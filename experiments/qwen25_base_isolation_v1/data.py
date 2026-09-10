"""Same raw input protocol; different tokenizer; exact Llama replay gate."""
from pathlib import Path
import gzip
import hashlib
import json
import os

import torch

from qera_original_a_isolation.common import import_official_qera
from qera_original_a_isolation.pipeline import _take_streaming_rows
from qera_diag_g_isolation.storage import atomic_json, atomic_tensors, checked_tensors, file_record, read_json, verify, log


def windows(module, role):
    rows = module["train" if role == "calibration" else "test"]
    if role == "calibration":
        if len(rows) < 256:
            raise RuntimeError("Insufficient full calibration windows")
        rows = rows.select(range(256))
    values = rows[:]
    ids = torch.tensor(values["input_ids"], dtype=torch.int64)
    masks = torch.tensor(values.get("attention_mask", torch.ones_like(ids)), dtype=torch.int64)
    if ids.ndim != 2 or ids.shape[1] != 2048 or len(ids) == 0 or not (masks == 1).all():
        raise RuntimeError("Expected complete, unpadded windows; do not silently pad/drop differently")
    return {"input_ids": ids, "attention_mask": masks}


def replay_gate(observed, expected):
    if set(observed) != set(expected) or any(not torch.equal(observed[k], expected[k]) for k in observed):
        raise RuntimeError("Llama token replay differs: data revision/preprocessing/environment is NOT aligned")


def offline_prefix(folder, expected):
    folder = Path(folder).resolve()
    source = folder / "calibration.source.json"
    path = folder / "calibration.jsonl.gz"
    metadata = read_json(source)
    for key in ("dataset_id", "resolved_revision", "raw_prefix_rows", "raw_text_sha256"):
        if metadata.get(key) != expected.get(key):
            raise RuntimeError(f"Offline prefix provenance mismatch: {key}")
    record = file_record(path)
    if record["sha256"] != metadata.get("file_sha256"):
        raise RuntimeError("Offline prefix file hash mismatch")
    rows, digest = [], hashlib.sha256()
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if not isinstance(row, dict) or not isinstance(row.get("text"), str):
                raise RuntimeError("Offline prefix must contain original text rows")
            encoded = row["text"].encode("utf-8")
            digest.update(len(encoded).to_bytes(8, "little"))
            digest.update(encoded)
            rows.append(row)
            if len(rows) > expected["raw_prefix_rows"]:
                raise RuntimeError("Offline prefix has extra rows")
    if len(rows) != expected["raw_prefix_rows"] or digest.hexdigest() != expected["raw_text_sha256"]:
        raise RuntimeError("Offline prefix original text hash/count mismatch")
    log("data-qwen", f"offline SlimPajama raw rows={len(rows)} hash={digest.hexdigest()} PASS")
    return rows, {"prefix": record, "metadata": file_record(source)}


def offline_wikitext(folder):
    import datasets as ds
    import pyarrow as pa
    folder = Path(folder).resolve()
    splits, records = {}, {}
    for split in ("train", "validation", "test"):
        path = folder / f"wikitext-{split}.arrow"
        records[split] = file_record(path)
        # Read into memory before mapping. Dataset.from_file would associate
        # map-cache writes with the OLD Arrow directory, which is forbidden.
        with pa.memory_map(str(path), "r") as source:
            table = pa.ipc.open_stream(source).read_all()
            if table.column_names != ["text"]:
                raise RuntimeError("Require original WikiText text, not tokenized cache Arrow")
            rows = table.to_pylist()
        if not rows or any(not isinstance(row["text"], str) for row in rows):
            raise RuntimeError("Invalid original WikiText rows")
        splits[split] = ds.Dataset.from_list(rows)
        log("data-qwen", f"offline WikiText split={split} raw_rows={len(rows)}")
    return ds.DatasetDict(splits), records


def prepare_data(config, reference, allow_download=False, offline_raw_dir=None, wikitext_cache_dir=None):
    if bool(offline_raw_dir) != bool(wikitext_cache_dir) or (offline_raw_dir and allow_download):
        raise RuntimeError("Offline inputs require both directories and forbid --allow-download")
    for variable in ("HF_HUB_OFFLINE", "HF_DATASETS_OFFLINE", "TRANSFORMERS_OFFLINE"):
        if allow_download:
            os.environ.pop(variable, None)
        else:
            os.environ[variable] = "1"
    import datasets as ds
    from transformers import AutoTokenizer
    # All writable dataset/map caches are private to this Qwen output.
    private_cache = str(Path(config["run_dir"]) / "dataset_cache")
    a = reference["a_config"]
    official = import_official_qera(a)
    result = {}
    for role in ("calibration", "wikitext2"):
        offline_records = {}
        metadata = read_json(verify(reference["a_calibration" if role == "calibration" else "a_wikitext2"]))
        if role == "calibration":
            if metadata.get("raw_prefix_rows") != 5120 or not metadata.get("resolved_revision"):
                raise RuntimeError("Missing frozen SlimPajama revision/prefix provenance")
            if offline_raw_dir:
                rows, offline_records = offline_prefix(offline_raw_dir, metadata)
            else:
                raw = ds.load_dataset(metadata["dataset_id"], revision=metadata["resolved_revision"],
                                      split="train", streaming=True, cache_dir=private_cache)
                rows, digest = _take_streaming_rows(raw, 5120)
                if digest != metadata["raw_text_sha256"]:
                    raise RuntimeError("SlimPajama raw prefix changed")
            raw = ds.DatasetDict(train=ds.Dataset.from_list(rows))
            name = "slim_pajama_6b"
        else:
            # Old pipeline did not pin a WikiText revision. Exact replay of its
            # frozen token windows is mandatory before accepting the source.
            if wikitext_cache_dir:
                raw, offline_records = offline_wikitext(wikitext_cache_dir)
            else:
                raw = ds.load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", cache_dir=private_cache)
            name = "wikitext2"
        baseline_file = reference["llama_g_manifest"]["payload"]["data"][role]
        baseline = checked_tensors(baseline_file)
        def encode(model_path):
            tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
            processed = official["preprocess_data_module"](raw, name, tokenizer=tokenizer,
                        padding="max_length", max_length=2048, num_proc=a["num_workers"])
            return windows(processed, role)
        replay_gate(encode(a["model_path"]), baseline)
        log("data-qwen", f"{role} Llama token replay EXACT PASS")
        qwen = encode(config["model_path"])
        path = Path(config["run_dir"]) / "data" / f"{role}.safetensors"
        if path.exists():
            # A crash between the tensor commit and metadata commit is safe:
            # reconstruct and compare every token before adopting that file.
            metadata_path = path.with_suffix(".json")
            record = read_json(metadata_path)["file"] if metadata_path.exists() else file_record(path)
            replay_gate(checked_tensors(record), qwen)
        else:
            atomic_tensors(path, qwen)
            record = file_record(path)
        state = {"file": record, "role": role, "windows": len(qwen["input_ids"]),
                 "tokens": qwen["input_ids"].numel(), "prediction_tokens": len(qwen["input_ids"])*2047,
                 "llama_replay_exact": True, "llama_reference": baseline_file,
                 "raw_fingerprints": {k: getattr(v, "_fingerprint", None) for k, v in raw.items()},
                 "calibration_revision": metadata.get("resolved_revision"), "chat_template": False,
                 "offline_source_files": offline_records}
        atomic_json(path.with_suffix(".json"), state)
        result[role] = state
    return result
