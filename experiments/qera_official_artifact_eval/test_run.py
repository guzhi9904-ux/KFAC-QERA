from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from qera_official_artifact_eval import run


def valid_config(tmp_path):
    roots = {}
    for name in ("model", "official", "harness", "mxint4", "mxint3"):
        path = tmp_path / name
        path.mkdir(exist_ok=True)
        roots[name] = str(path)
    return {
        **run.FIXED_PROTOCOL,
        "ranks": [8, 16, 32, 64],
        "bf16_reference": 7.5527,
        "bf16_tolerance": 0.05,
        "expected_visible_gpus": 2,
        "official_qera_commit": run.OFFICIAL_QERA_COMMIT,
        "harness_commit": run.HARNESS_COMMIT,
        "model_path": roots["model"],
        "official_qera_root": roots["official"],
        "harness_source": roots["harness"],
        "output_dir": str(tmp_path / "output"),
        "artifact_runs": [
            {"name": "mxint4", "width": 4, "run_dir": roots["mxint4"]},
            {"name": "mxint3", "width": 3, "run_dir": roots["mxint3"]},
        ],
    }


def write_config(tmp_path, value):
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(value), encoding="utf-8")
    return path


def test_config_fixes_official_protocol_and_disjoint_output(tmp_path):
    value = valid_config(tmp_path)
    parsed = run.load_config(write_config(tmp_path, value))
    assert parsed["context_length"] == 4096
    assert parsed["methods"] == ["wq", "diag_gd", "full_gd"]

    value["context_length"] = 2048
    with pytest.raises(ValueError, match="context_length"):
        run.load_config(write_config(tmp_path, value))
    value = valid_config(tmp_path)
    value["output_dir"] = str(Path(value["artifact_runs"][0]["run_dir"]) / "evaluation")
    with pytest.raises(ValueError, match="disjoint"):
        run.load_config(write_config(tmp_path, value))


def test_configuration_names_and_selection():
    assert run._configuration_name(4, "wq", None) == "W4_MXINT"
    assert run._configuration_name(3, "diag_gd", 32) == "DIAG_GD_R32"
    assert run._configuration_name(4, "full_gd", 64) == "FULL_GD_R64"
    assert run._selected("mxint4:W4_MXINT", set())
    assert run._selected("mxint4:W4_MXINT", {"mxint4:W4_MXINT"})
    assert not run._selected("mxint3:W3_MXINT", {"mxint4:W4_MXINT"})


def test_correction_hook_matches_merged_weight_and_is_removed():
    torch.manual_seed(4)
    model = torch.nn.Sequential(torch.nn.Linear(5, 4, bias=False))
    x = torch.randn(2, 3, 5)
    a, b = torch.randn(5, 3), torch.randn(3, 4)
    baseline = model(x)
    expected = baseline + (x @ a[:, :2]) @ b[:2]
    with run.correction_hooks(model, {"0": (a, b)}, 2):
        torch.testing.assert_close(model(x), expected)
    torch.testing.assert_close(model(x), baseline)


def test_baseline_gate_fails_closed():
    config = {"bf16_reference": 7.5527, "bf16_tolerance": 0.05}
    assert run._baseline_gate(config, {"word_ppl": 7.5527})["status"] == "PASS"
    assert run._baseline_gate(config, {"word_ppl": 7.7})["status"] == "FAIL"


def test_existing_result_checks_protocol_and_result_hash(tmp_path, monkeypatch):
    root = tmp_path / "output"
    directory = root / "BF16"
    directory.mkdir(parents=True)
    result = {
        "results": {"wikitext": {"word_perplexity,none": 7.5527}},
        "n-samples": {"wikitext": {"original": 1, "effective": 1}},
    }
    result_path = directory / "results.json"
    result_path.write_text(json.dumps(result), encoding="utf-8")
    state = {
        "status": "PASS",
        "protocol_sha256": "protocol",
        "results_sha256": run.sha256_file(result_path),
        "word_ppl": 7.5527,
    }
    (directory / "complete.json").write_text(json.dumps(state), encoding="utf-8")
    assert run._existing_result(root, "BF16", "protocol")["word_ppl"] == pytest.approx(7.5527)
    with pytest.raises(RuntimeError, match="different protocol"):
        run._existing_result(root, "BF16", "changed")


def test_artifact_protocol_binds_tensor_hashes():
    record = {"path": "/immutable/wq.safetensors", "sha256": "abc", "bytes": 12}
    configuration = run.EvaluationConfiguration("W4_MXINT", "wq", None, {"layer": {"quant": record}})
    plan = run.ArtifactRun("mxint4", 4, Path("/run"), {}, {"sha256": "manifest"}, (configuration,))
    before = run._artifact_protocol_hash("base", plan, configuration)
    changed = run.EvaluationConfiguration(
        "W4_MXINT", "wq", None, {"layer": {"quant": {**record, "sha256": "changed"}}}
    )
    after = run._artifact_protocol_hash("base", plan, changed)
    assert before != after
