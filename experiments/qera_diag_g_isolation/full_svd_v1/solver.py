"""Versioned full-SVD variant of the frozen round-one weighted solver.

The only numerical algorithm change is full_matrices=True. This matches the
official QERA SVD mode for both the identity gate and the diagonal-G solution.
"""
from __future__ import annotations

from contextlib import contextmanager
import math

import torch

from qera_diag_g_isolation.math_ops import apply_a


def solve_weighted(error_t, scale_a, scale_g, rank, inverse_a=None):
    if not 0 < rank <= min(error_t.shape):
        raise ValueError("Invalid rank")
    if scale_g.shape != (error_t.shape[1],) or not torch.isfinite(scale_g).all() or (scale_g <= 0).any():
        raise ValueError("G root must be a finite positive vector")
    weighted = apply_a(scale_a, error_t) * scale_g[None, :]
    u, singular, vh = torch.linalg.svd(weighted, full_matrices=True)
    u, singular, vh = u[:, :rank], singular[:rank], vh[:rank]
    if inverse_a is not None:
        left = inverse_a(scale_a, u)
    elif scale_a.ndim == 1:
        left = u / scale_a.clamp_min(torch.finfo(scale_a.dtype).eps)[:, None]
    else:
        left = torch.linalg.solve(scale_a, u)
    right = (singular[:, None] * vh) / scale_g[None, :]
    if not torch.isfinite(left).all() or not torch.isfinite(right).all():
        raise FloatingPointError("Nonfinite correction")
    before = float(weighted.double().square().sum())
    after = float((apply_a(scale_a, error_t - left @ right) * scale_g[None, :]).double().square().sum())
    if not math.isfinite(after) or after > before * (1 + 1e-5) + 1e-20:
        raise RuntimeError(f"Weighted objective increased: {before} -> {after}")
    return left, right, {"weighted_sse_before": before, "weighted_sse_after": after,
                         "solver_variant": "full_svd_v1", "svd_full_matrices": True}


@contextmanager
def bind_solver(pipeline):
    """Explicit process-local dependency binding; never modify torch or a file.

    The unchanged pipeline resolves this module attribute for BOTH its identity
    regression and GD solve. Call only while holding the parent and child locks.
    """
    previous = pipeline.solve_weighted
    pipeline.solve_weighted = solve_weighted
    try:
        yield
    finally:
        pipeline.solve_weighted = previous
