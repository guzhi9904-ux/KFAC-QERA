"""Verify solver equivalence and derived continuation without changing parent data."""
from copy import deepcopy
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from qera_diag_g_isolation import pipeline
from qera_diag_g_isolation.full_svd_v1 import run, solver
from qera_diag_g_isolation.storage import atomic_json, file_record, fingerprint, load_manifest, read_json


def official_inverse(scale, u):
    return torch.linalg.solve(torch.diag(scale) if scale.ndim == 1 else scale, u)


@pytest.mark.parametrize("full_a", [False, True])
def test_solver_calls_full_svd_for_both_identity_and_diagonal_g(monkeypatch, full_a):
    torch.manual_seed(3)
    error = torch.randn(9, 6, dtype=torch.float64)
    scale = torch.rand(9, dtype=torch.float64) + .5
    if full_a:
        scale = torch.diag(scale) + .01 * torch.ones(9, 9, dtype=torch.float64)
    actual_svd = torch.linalg.svd
    modes = []

    def spy(matrix, **kwargs):
        modes.append(kwargs.get("full_matrices"))
        return actual_svd(matrix, **kwargs)

    monkeypatch.setattr(torch.linalg, "svd", spy)
    for g in (torch.ones(6, dtype=torch.float64), torch.rand(6, dtype=torch.float64) + .5):
        left, right, metrics = solver.solve_weighted(error, scale, g, 3, official_inverse)
        weighted = solver.apply_a(scale, error) * g
        u, s, vh = actual_svd(weighted, full_matrices=True)
        expected_left = official_inverse(scale, u[:, :3])
        expected_right = (torch.diag(s[:3]) @ vh[:3]) / g
        torch.testing.assert_close(left @ right, expected_left @ expected_right, atol=1e-12, rtol=1e-12)
        assert metrics["weighted_sse_after"] == pytest.approx(float(s[3:].square().sum()), rel=1e-10)
        assert metrics["svd_full_matrices"] is True
    assert modes == [True, True]


def test_binding_restores_original_even_after_gate_failure():
    original = object()
    fake_pipeline = SimpleNamespace(solve_weighted=original)
    with pytest.raises(RuntimeError):
        with solver.bind_solver(fake_pipeline):
            assert fake_pipeline.solve_weighted is solver.solve_weighted
            raise RuntimeError("identity gate failed")
    assert fake_pipeline.solve_weighted is original


def evidence(manifest_sha, ranks):
    return {
        "status": "AUDIT_COMPLETE", "manifest_sha256": manifest_sha,
        "layer": "model.layers.0.self_attn.k_proj", "method": "diag",
        "inputs": {key: {"exact_equal": True} for key in (
            "fresh_official_q_vs_frozen_q", "new_error_vs_official_error", "new_svd_input_vs_official_svd_input",
        )},
        "variants": {name: {"vs_saved_production_gate": {str(r): value for r in ranks}} for name, value in (
            ("official_replay", 0.), ("old_input_full_svd", 0.), ("new_input_full_svd", 0.),
            ("current_solver", .003), ("old_input_reduced_svd", .003), ("new_input_reduced_svd", .003),
        )},
    }


def test_evidence_requires_matching_inputs_and_all_controlled_paths():
    good = evidence("abc", [8, 16, 32, 64])
    assert run.audit_supports_full_svd(good, "abc", .001, [8, 16, 32, 64])
    assert not run.audit_supports_full_svd(good, "different", .001, [8, 16, 32, 64])
    changed = deepcopy(good)
    changed["inputs"]["new_svd_input_vs_official_svd_input"]["exact_equal"] = False
    assert not run.audit_supports_full_svd(changed, "abc", .001, [8, 16, 32, 64])
    changed = deepcopy(good)
    changed["variants"]["new_input_full_svd"]["vs_saved_production_gate"]["64"] = .002
    assert not run.audit_supports_full_svd(changed, "abc", .001, [8, 16, 32, 64])
    changed["variants"]["new_input_full_svd"]["vs_saved_production_gate"]["64"] = float("nan")
    assert not run.audit_supports_full_svd(changed, "abc", .001, [8, 16, 32, 64])


@pytest.fixture
def retained_run(tmp_path):
    root = tmp_path / "g256"
    root.mkdir()
    config = {"run_dir": str(root), "ranks": [1, 2], "identity_product_tolerance": .001,
              "control_ppl_tolerance": .01}
    code = tmp_path / "original_solver.py"
    code.write_text("# immutable original solver\n")
    layer = {"name": "model.layers.0.self_attn.k_proj", "shape": [5, 9]}
    payload = {"config": config, "groups": [{"layers": [layer]}],
               "code": {str(code): file_record(code)}}
    parent = {"sha256": fingerprint(payload), "payload": payload}
    atomic_json(root / "manifest.json", parent)
    pipeline._save_g_checkpoint(config, parent, {layer["name"]: torch.ones(5, dtype=torch.float64)}, 256, 10.)
    path = pipeline.artifact_path(config, "quantized", layer["name"])
    pipeline.save_artifact(path, parent, {"weight_q": torch.ones(5, 9, dtype=torch.bfloat16)},
                           layer=layer["name"], fp32_bf16_quantization_equal=True)
    audit_path = root / "diagnostics/identity_g_20260908_185533_test/report.json"
    atomic_json(audit_path, evidence(parent["sha256"], config["ranks"]))
    return config, parent, layer, code


def snapshot_parent(config):
    return {path: path.read_bytes() for path in Path(config["run_dir"]).rglob("*")
            if path.is_file() and run.VARIANT not in path.relative_to(config["run_dir"]).parts}


def test_initialization_reuses_tensors_pins_solver_and_preserves_parent(retained_run):
    config, parent, layer, code = retained_run
    frozen = snapshot_parent(config)
    derived_config = run.derived_config(config)
    destination = Path(derived_config["run_dir"])
    destination.mkdir()
    derived = run.prepare_derived(config, parent, derived_config)
    assert derived["sha256"] != parent["sha256"]
    assert derived["payload"]["solver_transition"]["svd_full_matrices"] is True
    assert len(derived["payload"]["code"]) == len(parent["payload"]["code"]) + 2
    assert derived_config["identity_product_tolerance"] == config["identity_product_tolerance"]
    assert load_manifest(config) == parent
    assert load_manifest(derived_config) == derived
    raw, count, _ = pipeline._load_g_checkpoint(derived_config, derived)
    assert count == 256 and torch.equal(raw[layer["name"]], torch.ones(5, dtype=torch.float64))
    # Exercise the real solve/evaluation artifact reader with the derived metadata.
    quant = pipeline._evaluation_inputs(derived_config, derived, "wq")[layer["name"]]["quant"]
    assert Path(quant["path"]).is_relative_to(Path(config["run_dir"]))
    assert not Path(quant["path"]).is_relative_to(destination)
    assert not list(destination.rglob("*.safetensors"))
    assert all(path.read_bytes() == content for path, content in frozen.items())
    before_resume = {path: path.read_bytes() for path in destination.rglob("*") if path.is_file()}
    assert run.prepare_derived(config, parent, derived_config) == derived
    assert all(path.read_bytes() == content for path, content in before_resume.items())
    # Resume must reject changes to extension code records, rather than repin them.
    actual_record = run.file_record

    def changed_record(path):
        record = actual_record(path)
        if Path(path).name == "solver.py":
            record["sha256"] = "modified-extension"
        return record

    from unittest.mock import patch
    with patch.object(run, "file_record", changed_record), pytest.raises(RuntimeError, match="differs"):
        run.prepare_derived(config, parent, derived_config)
    assert all(path.read_bytes() == content for path, content in frozen.items())


def test_partial_setup_resumes_without_overwriting_parent(retained_run, monkeypatch):
    config, parent, _, _ = retained_run
    frozen = snapshot_parent(config)
    derived_config = run.derived_config(config)
    root = Path(derived_config["run_dir"])
    root.mkdir()
    original = pipeline.completed_artifact
    monkeypatch.setattr(pipeline, "completed_artifact", lambda *_: None)
    with pytest.raises(RuntimeError, match="Missing validated parent Wq"):
        run.prepare_derived(config, parent, derived_config)
    assert (root / "initialization.json").exists()
    assert not (root / "reuse_complete.json").exists()
    monkeypatch.setattr(pipeline, "completed_artifact", original)
    run.prepare_derived(config, parent, derived_config)
    assert read_json(root / "reuse_complete.json")["status"] == "PASS"
    assert all(path.read_bytes() == content for path, content in frozen.items())


def test_corrupt_parent_checkpoint_is_rejected(retained_run):
    config, parent, _, _ = retained_run
    state = read_json(Path(config["run_dir"]) / "statistics/collect_state.json")
    Path(state["file"]["path"]).write_bytes(b"corrupted checkpoint")
    with pytest.raises(RuntimeError, match="Immutable input"):
        run.validate_parent(config, parent)


def test_unrecognized_child_is_not_adopted(retained_run):
    config, parent, _, _ = retained_run
    derived_config = run.derived_config(config)
    root = Path(derived_config["run_dir"])
    root.mkdir()
    unrelated = root / "notes.txt"
    unrelated.write_text("keep me")
    with pytest.raises(RuntimeError, match="Unrecognized files"):
        run.prepare_derived(config, parent, derived_config)
    assert unrelated.read_text() == "keep me"
