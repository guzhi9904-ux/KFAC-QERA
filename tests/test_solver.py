import pytest
import torch

from qera_exp.solver import Metric, analyze_metric, solve_weighted


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
