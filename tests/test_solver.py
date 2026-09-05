from pathlib import Path

import pytest
import torch

from qera_exp.solver import Metric, _completed_module_state, analyze_metric, solve_weighted
from qera_exp.utils import atomic_safetensors, ensure_layout, save_json, sha256_file


@pytest.mark.parametrize("root_method", ["eigh", "cholesky"])
def test_weighted_solver_reduces_objective(root_method: str) -> None:
    generator = torch.Generator().manual_seed(19)
    error = torch.randn(11, 9, generator=generator, dtype=torch.float64)
    xa = torch.randn(31, 9, generator=generator, dtype=torch.float64)
    xg = torch.randn(29, 11, generator=generator, dtype=torch.float64)
    a = xa.T @ xa / xa.shape[0]
    g = xg.T @ xg / xg.shape[0]
    _, ad, af = analyze_metric(
        a,
        a.diag(),
        label="A",
        damping_lambda=1e-4,
        negative_relative_tolerance=1e-6,
        root_method=root_method,
    )
    _, gd, gf = analyze_metric(
        g,
        g.diag(),
        label="G",
        damping_lambda=1e-4,
        negative_relative_tolerance=1e-6,
        root_method=root_method,
    )
    identity = Metric("I", 11)
    for a_metric, g_metric in ((ad, identity), (ad, gd), (ad, gf), (af, identity), (af, gd), (af, gf)):
        result = solve_weighted(error, a_metric, g_metric, rank=4, device="cpu")
        assert result["objective_after"] <= result["objective_before"]
        assert result["left"].dtype == torch.bfloat16
        assert result["right"].dtype == torch.bfloat16


def test_completed_module_state_validates_resume_artifacts(tmp_path: Path) -> None:
    ensure_layout(tmp_path)
    module = "model.layers.0.self_attn.q_proj"
    method = "AD_GI"
    maximum_rank = 4
    artifact = tmp_path / "corrections" / f"model__layers__0__self_attn__q_proj__{method}__r4.safetensors"
    atomic_safetensors(artifact, {"left": torch.ones(3, 4), "right": torch.ones(2, 4)})
    artifact_hash = sha256_file(artifact)
    save_json(artifact.with_suffix(".json"), {"artifact_sha256": artifact_hash})
    save_json(tmp_path / "statistics" / "metrics" / "model__layers__0__self_attn__q_proj.json", {"module": module})
    state = {
        "status": "PASS",
        "module": module,
        "methods": [method],
        "maximum_rank": maximum_rank,
        "artifacts": [{"method": method, "artifact": str(artifact), "artifact_sha256": artifact_hash}],
    }
    save_json(tmp_path / "state" / "solve_module_model__layers__0__self_attn__q_proj.json", state)
    assert _completed_module_state(tmp_path, module, [method], maximum_rank) == state
