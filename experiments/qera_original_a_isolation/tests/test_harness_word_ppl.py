from pathlib import Path
from types import SimpleNamespace
import hashlib
import sys

import pytest
import torch
from safetensors.torch import save_file

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from qera_original_a_isolation import harness_word_ppl as h
from qera_original_a_isolation.common import safe_name


def result(ppl=7.55):
    return {
        "results": {"wikitext": {"word_perplexity,none": ppl, "word_perplexity_stderr,none": "N/A",
                                 "byte_perplexity,none": 1.7}},
        "n-samples": {"wikitext": {"original": 60, "effective": 60}},
    }


def save_result(root, name="BF16", ppl=7.55, protocol="test"):
    path = root / name / "results.json"
    h.write_json(path, result(ppl))
    record = {"status": "PASS", "configuration": name, "word_ppl": ppl,
              "protocol_sha256": protocol, "results_sha256": h.file_hash(path), "correction_sha256": None}
    h.write_json(root / name / "complete.json", record)
    return record


def test_exact_word_metric_not_stderr_or_byte_metric():
    assert h.extract_word_ppl(result()) == 7.55
    broken = result()
    broken["results"]["wikitext"].pop("word_perplexity,none")
    with pytest.raises(RuntimeError, match="word_perplexity"):
        h.extract_word_ppl(broken)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), 0, -1])
def test_invalid_metric(value):
    with pytest.raises(RuntimeError):
        h.extract_word_ppl(result(value))


def test_reject_partial_evaluation():
    partial = result()
    partial["n-samples"]["wikitext"]["effective"] = 5
    with pytest.raises(RuntimeError, match="Incomplete"):
        h.extract_word_ppl(partial)


def test_resume_checks_protocol_and_result_bytes(tmp_path):
    save_result(tmp_path)
    assert h.existing_result(tmp_path, "BF16", "test")["word_ppl"] == 7.55
    with pytest.raises(RuntimeError, match="protocol"):
        h.existing_result(tmp_path, "BF16", "changed")
    h.write_json(tmp_path / "BF16/results.json", result(6.24))
    with pytest.raises(RuntimeError, match="checksum"):
        h.existing_result(tmp_path, "BF16", "test")


@pytest.mark.parametrize("stage,ppl,expected", [
    ("bf16", 7.55, ["BF16"]), ("all", 6.24, ["BF16"]),
    ("all", 7.57, list(h.REFERENCE)), ("compare", 7.55, list(h.REFERENCE)[1:]),
    ("compare", 6.24, []),
])
def test_stage_gating(tmp_path, monkeypatch, stage, ppl, expected):
    calls = []
    if stage == "compare":
        save_result(tmp_path, ppl=ppl)

    def evaluate(_config, _task, name, root, protocol_hash, _args):
        calls.append(name)
        return save_result(root, name, ppl if name == "BF16" else 8.4, protocol_hash)

    monkeypatch.setattr(h, "evaluate_one", evaluate)
    args = SimpleNamespace(stage=stage, bf16_tolerance=0.05)
    code = h.execute_stages({}, None, tmp_path, "test", args)
    assert calls == expected
    assert code == (0 if abs(ppl - 7.55) <= 0.05 else 2)


def test_compare_requires_baseline(tmp_path):
    with pytest.raises(RuntimeError, match="bf16 first"):
        h.execute_stages({}, None, tmp_path, "test", SimpleNamespace(stage="compare", bf16_tolerance=0.05))


def test_summary_does_not_drop_completed_comparisons(tmp_path):
    baseline = save_result(tmp_path)
    save_result(tmp_path, "QERA_DIAG_R32", 8.45)
    h.write_summary(tmp_path, [baseline], "test")
    assert "QERA_DIAG_R32" in (tmp_path / "word_ppl_summary.csv").read_text()


def test_source_verification_and_crlf(tmp_path, monkeypatch):
    path = tmp_path / "source.py"
    path.write_bytes(b"a\r\nb\r\n")
    monkeypatch.setattr(h, "HARNESS_HASHES", {"source.py": hashlib.sha256(b"a\nb\n").hexdigest()})
    h.verify_harness(tmp_path)
    path.write_bytes(b"different")
    with pytest.raises(RuntimeError, match="hash mismatch"):
        h.verify_harness(tmp_path)


def test_corrections_verified_without_loading_roots(tmp_path):
    name = "model.layers.0.self_attn.q_proj"
    config = {"run_dir": str(tmp_path)}
    h.write_json(tmp_path / "plan.json", {"target_module_count": 1,
        "shards": [[{"target": name, "shares": [], "in_features": 4}]]})
    path = tmp_path / "corrections/diag" / f"{safe_name(name)}.safetensors"
    path.parent.mkdir(parents=True)
    save_file({"A": torch.zeros(4, 64), "B": torch.zeros(64, 8)}, str(path))
    h.write_json(path.with_suffix(".json"), {"status": "PASS", "sha256": h.file_hash(path),
        "method": "diag", "layer": name, "rank": 64})
    assert list(h.correction_identity(config, "diag")) == [name]
    assert not (tmp_path / "statistics").exists()
    save_file({"A": torch.ones(4, 64), "B": torch.zeros(64, 8)}, str(path))
    with pytest.raises(RuntimeError, match="Invalid saved correction"):
        h.correction_identity(config, "diag")


def test_dataset_audit_is_document_based():
    task = SimpleNamespace(eval_docs=[{"page": "a b\nc"}, {"page": "d e"}])
    audit = h.dataset_identity(task)
    assert audit["documents"] == 2
    assert audit["raw_words_audit"] == 5
    assert audit["raw_pages_sha256"] == h.dataset_identity(task)["raw_pages_sha256"]
