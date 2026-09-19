"""Read-only FP64 contractions; no fitting, damping, or factor scaling."""
import torch


def norm_product(a, g):
    return float(a.square().sum() * g.square().sum())


def cross(s, a, g):
    return float(((g @ s @ a) * s).sum())


def projection_check(s, r, direct, tolerance=1e-10):
    cached = float((s * r).sum())
    scale = float((s * r).abs().sum())
    error = abs(cached - direct)
    # Same absolute / cancellation scale as the parent, including exact-zero case.
    assert error <= tolerance * scale, (error, scale)
    return dict(cached=cached, direct=direct, absolute_error=error,
                absolute_sum_scale=scale, tolerance=tolerance)


def small_checks():
    generator = torch.Generator().manual_seed(2026091804)
    samples = torch.randn(5, 3, 4, dtype=torch.float64, generator=generator)
    pairs = []
    for _ in range(2):
        x = torch.randn(4, 4, dtype=torch.float64, generator=generator)
        y = torch.randn(3, 3, dtype=torch.float64, generator=generator)
        a, g = x @ x.T, y @ y.T
        pairs.extend([(a, g), (a + .001*a.trace()/4*torch.eye(4),
                              g + .001*g.trace()/3*torch.eye(3))])
    vectors = samples.flatten(1)
    h = vectors.T @ vectors / (5 * 7)
    js = [norm_product(a, g) - 2*sum(cross(s, a, g) for s in samples)/(5*7)
          for a, g in pairs]
    errors = [float((h-torch.kron(g, a)).square().sum()) for a, g in pairs]
    checks = []
    for i, j in [(0, 2), (1, 3), (0, 1), (2, 3)]:
        difference = abs((js[i]-js[j])-(errors[i]-errors[j]))
        assert difference <= 1e-10*max(1, abs(js[i]), abs(js[j]))
        checks.append(dict(pair=[i,j], J_difference=js[i]-js[j],
                           explicit_error_difference=errors[i]-errors[j], difference=difference))
    return dict(passed=True, explicit_H_small_matrix_only=True, raw_solve_J_checks=checks)
