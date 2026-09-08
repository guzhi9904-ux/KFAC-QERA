"""Tests of audit evidence and read-only production inputs; no CUDA required."""
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch

spec = importlib.util.spec_from_file_location("identity_g_audit", Path(__file__).with_name("identity_g.py"))
audit_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit_module)


def inverse(scale, u):
    return torch.linalg.solve(torch.diag(scale) if scale.ndim == 1 else scale, u)


def test_product_comparison_casts_before_multiplication():
    # Both FP32 dot products round to 1, while the represented factors differ.
    reference = (torch.tensor([[1., 1e-8]]), torch.ones(2, 1))
    actual = (torch.tensor([[1., 2e-8]]), torch.ones(2, 1))
    assert torch.equal(actual[0] @ actual[1], reference[0] @ reference[1])
    measured = audit_module.product_difference(actual, reference, [2])["2"]
    assert measured == pytest.approx(1e-8, rel=1e-6)


def test_product_comparison_ignores_paired_svd_signs():
    torch.manual_seed(7)
    left, right = torch.randn(11, 4), torch.randn(4, 5)
    signs = torch.tensor([1., -1., -1., 1.])
    changed = left * signs, right * signs[:, None]
    assert audit_module.product_difference(changed, (left, right), [1, 2, 4], row_chunk=3) == {
        "1": 0., "2": 0., "4": 0.,
    }


@pytest.mark.parametrize("full", [False, True])
def test_objective_matches_svd_tail_and_controlled_paths(full):
    torch.manual_seed(8)
    error = torch.randn(10, 6, dtype=torch.float64)
    scale = torch.rand(10, dtype=torch.float64) + .4
    if full:
        scale = torch.diag(scale) + .02 * torch.ones(10, 10, dtype=torch.float64)
    matrix = audit_module.apply_a(scale, error)
    ranks = [1, 3, 5]
    factors, spectrum = audit_module.controlled_svd(matrix, scale, ranks, inverse, True)
    reduced, _ = audit_module.controlled_svd(matrix, scale, ranks, inverse, False)
    assert max(audit_module.product_difference(factors, reduced, ranks).values()) < 1e-12
    objectives = audit_module.weighted_objectives(factors, scale, error, ranks, row_chunk=3)
    for rank in ranks:
        assert objectives[str(rank)]["sse"] == pytest.approx(
            spectrum["boundaries"][str(rank)]["optimal_tail_sse_for_this_matrix"], rel=1e-10,
        )


def test_spectral_boundary_gap_and_last_rank():
    spectrum = audit_module.spectral_description(torch.tensor([3., 2., 2., 1.]), [2, 4])
    assert spectrum["boundaries"]["2"]["relative_gap"] == 0
    assert spectrum["boundaries"]["4"]["relative_gap"] is None
    assert spectrum["boundaries"]["4"]["optimal_tail_sse_for_this_matrix"] == 0


def test_unique_layer_selection_rejects_missing_and_duplicate():
    manifest = {"payload": {"groups": [{"layers": [{"name": "x"}]}]}}
    assert audit_module.select_layer(manifest, "x")[1]["name"] == "x"
    with pytest.raises(ValueError):
        audit_module.select_layer(manifest, "absent")
    manifest["payload"]["groups"].append({"layers": [{"name": "x"}]})
    with pytest.raises(ValueError):
        audit_module.select_layer(manifest, "x")


def test_synthetic_audit_retains_inputs_and_records_controlled_comparisons(tmp_path, monkeypatch):
    from safetensors.torch import save_file

    torch.manual_seed(12)
    layer_name = "model.layers.0.self_attn.k_proj"
    weight = torch.randn(5, 9).bfloat16()
    quantized = (weight.float() * 4).round().div(4).bfloat16()
    scale = torch.rand(9) + .5
    error = (weight.float() - quantized.float()).T
    ranks = [1, 2, 4]
    reference, _ = audit_module.controlled_svd(scale[:, None] * error, scale, ranks, inverse, True)
    paths = {name: tmp_path / f"{name}.safetensors" for name in ("weights", "roots", "gi", "wq")}
    save_file({layer_name + ".weight": weight}, str(paths["weights"]))
    save_file({"diag": scale}, str(paths["roots"]))
    save_file({"A": reference[0].contiguous(), "B": reference[1].contiguous()}, str(paths["gi"]))
    save_file({"weight_q": quantized}, str(paths["wq"]))
    records = {name: audit_module.file_record(path) for name, path in paths.items()}
    state = tmp_path / "statistics" / "collect_state.json"
    audit_module.atomic_json(state, {"windows_completed": 256})
    frozen = {path: path.read_bytes() for path in (*paths.values(), state)}
    layer = {"name": layer_name, "weight_file": "weights", "gi": {"diag": records["gi"]}}
    manifest = {"payload": {"groups": [{"roots": records["roots"], "layers": [layer]}],
                            "model_files": {"weights": records["weights"]}, "source_config": {}}}
    config = {"run_dir": str(tmp_path), "ranks": ranks, "num_calibration_windows": 256,
              "identity_product_tolerance": 1e-3}
    monkeypatch.setattr(audit_module, "_load_g_checkpoint", lambda *_: ({}, 256, 1.))
    monkeypatch.setattr(audit_module, "completed_artifact", lambda *_: {"file": records["wq"]})
    official_calls = []

    def compute_ab(name, linear, local_scale, local_config):
        official_calls.append(name)
        local_error = (linear.weight - quantized.float()).T
        result, _ = audit_module.controlled_svd(
            torch.diag(local_scale) @ local_error, local_scale, [local_config["rank"]], inverse, True,
        )
        return {name + ".A": result[0], name + ".B": result[1]}, 0.

    monkeypatch.setattr(audit_module, "import_official_qera", lambda *_: {
        "compute_ab": compute_ab, "mxint_quantizer": lambda w, **_: (w * 4).round() / 4,
    })
    monkeypatch.setitem(sys.modules, "qera.approximate", SimpleNamespace(_compute_scale_inv_dot_U=inverse))
    report = {"variants": {}}
    args = SimpleNamespace(device="cpu", layer=layer_name, method="diag", skip_fp64_reference=True)
    report_path = tmp_path / "diagnostics" / "test" / "report.json"
    audit_module.audit(config, manifest, args, report, report_path)
    assert official_calls == [layer_name]
    assert report["gate_observations"]["official_replay_matches_saved"]
    assert report["gate_observations"]["current_solver_matches_saved"]
    assert report["inputs"]["fresh_official_q_vs_frozen_q"]["exact_equal"]
    assert len(report["variants"]) == 6
    assert len(report["controlled_comparisons_fp64_products"]) == 5
    assert all(path.read_bytes() == content for path, content in frozen.items())
