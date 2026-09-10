import gzip
import hashlib
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch
import pyarrow as pa
import datasets as ds

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from qwen25_base_isolation_v1 import data
from qera_diag_g_isolation.storage import atomic_json, atomic_tensors, file_record, read_json


def save_prefix(folder, rows, expected=None):
    folder.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    with gzip.open(folder / "calibration.jsonl.gz", "wt", encoding="utf-8") as f:
        for row in rows:
            value = row["text"].encode("utf-8")
            digest.update(len(value).to_bytes(8, "little"))
            digest.update(value)
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    metadata = expected or {"dataset_id": "DKYoon/SlimPajama-6B", "resolved_revision": "test-revision",
                            "raw_prefix_rows": len(rows), "raw_text_sha256": digest.hexdigest()}
    atomic_json(folder / "calibration.source.json", {**metadata,
                "file_sha256": file_record(folder / "calibration.jsonl.gz")["sha256"]})
    return metadata


def arrow_splits(folder, column="text"):
    folder.mkdir()
    for split in ("train", "validation", "test"):
        table = pa.table({column: ["first", "", "last"]})
        path = folder / f"wikitext-{split}.arrow"
        with pa.OSFile(str(path), "wb") as f:
            with pa.ipc.new_stream(f, table.schema) as stream:
                stream.write_table(table)


def test_prefix_hash_count_order_and_revision(tmp_path):
    rows = [{"text": "one\n中文", "meta": {"tag": ["x"]}}, {"text": "two"}]
    expected = save_prefix(tmp_path, rows)
    loaded, records = data.offline_prefix(tmp_path, expected)
    assert loaded == rows and set(records) == {"prefix", "metadata"}
    with pytest.raises(RuntimeError, match="resolved_revision"):
        data.offline_prefix(tmp_path, {**expected, "resolved_revision": "other"})
    for changed in (rows[::-1], rows[:1], rows + [{"text": "extra"}]):
        save_prefix(tmp_path, changed, expected)
        with pytest.raises(RuntimeError, match="hash/count|extra rows"):
            data.offline_prefix(tmp_path, expected)


def test_prefix_file_corruption_rejected(tmp_path):
    expected = save_prefix(tmp_path, [{"text": "one"}])
    with (tmp_path / "calibration.jsonl.gz").open("ab") as f:
        f.write(b"corruption")
    with pytest.raises(RuntimeError, match="file hash"):
        data.offline_prefix(tmp_path, expected)


def test_wikitext_mapping_never_writes_old_cache(tmp_path):
    folder = tmp_path / "old"
    arrow_splits(folder)
    before = {p: (file_record(p), p.stat().st_mtime_ns) for p in folder.iterdir()}
    raw, records = data.offline_wikitext(folder)
    assert set(records) == {"train", "validation", "test"}
    for split in raw.values():
        assert not split.cache_files
        assert split[1]["text"] == ""  # Never drop empty lines.
    raw.map(lambda batch: {"length": [len(x) for x in batch["text"]]}, batched=True, num_proc=2)
    assert before == {p: (file_record(p), p.stat().st_mtime_ns) for p in folder.iterdir()}


def test_tokenized_wikitext_cache_not_accepted(tmp_path):
    folder = tmp_path / "old"
    arrow_splits(folder, column="input_ids")
    with pytest.raises(RuntimeError, match="original WikiText text"):
        data.offline_wikitext(folder)


def test_offline_prepare_data_replay_dynamic_windows_no_network(tmp_path, monkeypatch):
    prefix = tmp_path / "raw"
    metadata = save_prefix(prefix, [{"text": f"row {i}"} for i in range(5120)])
    cached = tmp_path / "old_cache"
    arrow_splits(cached)
    before = {p: (file_record(p), p.stat().st_mtime_ns) for p in cached.iterdir()}
    calibration_meta, eval_meta = tmp_path / "calibration.json", tmp_path / "eval.json"
    atomic_json(calibration_meta, metadata)
    atomic_json(eval_meta, {})
    baseline = {}
    for role, n in (("calibration", 256), ("wikitext2", 3)):
        path = tmp_path / f"llama-{role}.safetensors"
        ones = torch.ones(n, 2048, dtype=torch.int64)
        atomic_tensors(path, {"input_ids": ones, "attention_mask": ones.clone()})
        baseline[role] = file_record(path)
    reference = {"a_config": {"model_path": "llama", "num_workers": 8},
                 "a_calibration": file_record(calibration_meta), "a_wikitext2": file_record(eval_meta),
                 "llama_g_manifest": {"payload": {"data": baseline}}}
    config = {"run_dir": str(tmp_path / "new"), "model_path": "qwen"}
    def no_network(*_, **__):
        raise AssertionError("Offline preparation attempted Hub dataset loading")
    monkeypatch.setattr(ds, "load_dataset", no_network)
    mod = ModuleType("transformers")
    mod.PreTrainedTokenizerBase = type("PreTrainedTokenizerBase", (), {})
    mod.AutoTokenizer = SimpleNamespace(from_pretrained=lambda path, **_: path)
    monkeypatch.setitem(sys.modules, "transformers", mod)
    bad_replay = False
    def process(raw, name, tokenizer, **kwargs):
        assert kwargs == {"padding": "max_length", "max_length": 2048, "num_proc": 8}
        assert all("text" in split.column_names for split in raw.values())
        if name == "slim_pajama_6b":
            assert len(raw["train"]) == 5120
            n, split = 256, "train"
        else:
            n, split = (3 if tokenizer == "llama" else 4), "test"
        value = 1 if tokenizer == "llama" and not bad_replay else 2
        return ds.DatasetDict({split: ds.Dataset.from_dict({"input_ids": [[value]*2048]*n,
                                                          "attention_mask": [[1]*2048]*n})})
    monkeypatch.setattr(data, "import_official_qera", lambda _: {"preprocess_data_module": process})
    output = data.prepare_data(config, reference, offline_raw_dir=prefix, wikitext_cache_dir=cached)
    assert output["wikitext2"]["windows"] == 4
    assert output["calibration"]["llama_replay_exact"] is True
    assert output == data.prepare_data(config, reference, offline_raw_dir=prefix, wikitext_cache_dir=cached)
    assert before == {p: (file_record(p), p.stat().st_mtime_ns) for p in cached.iterdir()}
    bad_replay = True
    with pytest.raises(RuntimeError, match="Llama token replay"):
        data.prepare_data(config, reference, offline_raw_dir=prefix, wikitext_cache_dir=cached)


def test_partial_offline_configuration_rejected():
    with pytest.raises(RuntimeError, match="both directories"):
        data.prepare_data({}, {}, offline_raw_dir="somewhere")
    with pytest.raises(RuntimeError, match="forbid"):
        data.prepare_data({}, {}, allow_download=True, offline_raw_dir="a", wikitext_cache_dir="b")
