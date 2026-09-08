"""Explicit precision contract: FP32 teacher, FP64 Gram, FP32 QERA solve."""
from __future__ import annotations

import math
import torch

from qera_diag_g_isolation.math_ops import apply_a, diagonal_increment


def relative_error(value, reference):
    numerator = float(torch.linalg.vector_norm((value.double() - reference.double()).reshape(-1)))
    denominator = max(float(torch.linalg.vector_norm(reference.double().reshape(-1))), 1e-30)
    result = numerator / denominator
    if not math.isfinite(result):
        raise FloatingPointError("Nonfinite relative error")
    return result


def accumulate_gram(gradient, mask, gram_sum, diagonal_sum, row_tile=512):
    """Bound GPU temporary by row tiles; no FP32/TF32 Gram multiplication.

    A failed/partial window may mutate RAM, but MUST NOT commit a checkpoint.
    The process must exit on error and reload the last whole-window checkpoint.
    """
    if gradient.dtype != torch.float32 or row_tile <= 0:
        raise ValueError("Expected FP32 gradient and positive row tile")
    d = gradient.shape[-1]
    if gram_sum.shape != (d, d) or diagonal_sum.shape != (d,):
        raise ValueError("Accumulator shape mismatch")
    if any(x.device.type != "cpu" or x.dtype != torch.float64 for x in (gram_sum, diagonal_sum)):
        raise ValueError("Accumulators must be FP64 CPU tensors")
    # Identical direct diagonal implementation to the frozen diagonal-G run.
    direct = diagonal_increment(gradient, mask, chunk_tokens=128)
    selected = gradient[mask.to(gradient.device)].double()
    if not torch.isfinite(selected).all():
        raise FloatingPointError("Nonfinite selected gradient")
    for start in range(0, d, row_tile):
        end = min(d, start + row_tile)
        block = selected[:, start:end].T @ selected
        if not torch.isfinite(block).all():
            raise FloatingPointError("Nonfinite FP64 Gram")
        gram_sum[start:end].add_(block.cpu())
    diagonal_sum.add_(direct)


def check_diagonal(gram, diagonal, tolerance=1e-10):
    error = relative_error(gram.diagonal(), diagonal)
    if error > tolerance:
        raise RuntimeError(f"Full-G/direct diagonal mismatch: {error} > {tolerance}")
    return error


def full_root(gram_sum, count, relative_floor=1e-6):
    """Trace-normalize, symmetrize and eigenvalue-floor in FP64, root -> FP32.

    This is the matrix extension of diagonal_scale, NOT extra damping of A.
    The raw matrix remains untouched and is retained on disk.
    """
    if count <= 0 or not 0 < relative_floor < 1 or gram_sum.ndim != 2:
        raise ValueError("Invalid full-G root arguments")
    if gram_sum.shape[0] != gram_sum.shape[1] or not torch.isfinite(gram_sum).all():
        raise ValueError("Full-G must be a finite square matrix")
    raw = gram_sum.double() / count
    asymmetry = relative_error(raw, raw.T)
    if asymmetry > 1e-10:
        raise RuntimeError(f"Full-G asymmetry exceeds FP64 roundoff gate: {asymmetry}")
    raw = (raw + raw.T) * .5
    mean = float(raw.diagonal().mean())
    if mean <= 0:
        raise ValueError("All-zero/negative Full-G trace")
    raw.div_(mean)
    eigenvalues, vectors = torch.linalg.eigh(raw)
    if not torch.isfinite(eigenvalues).all() or not torch.isfinite(vectors).all():
        raise FloatingPointError("Nonfinite G eigendecomposition")
    # Negative roundoff is possible for PSD Gram matrices; material negatives are not.
    negative_limit = 1e-10 * max(float(eigenvalues.abs().max()), 1.)
    if float(eigenvalues.min()) < -negative_limit:
        raise RuntimeError("Full-G is materially non-PSD")
    effective = eigenvalues.clamp_min(relative_floor)
    root = (vectors * effective.sqrt()[None, :]) @ vectors.T
    root = ((root + root.T) * .5).float()
    if not torch.isfinite(root).all():
        raise FloatingPointError("Nonfinite G square root")
    return root, {
        "raw_mean_diagonal": mean, "relative_floor": relative_floor,
        "floored_eigenvalues": int((eigenvalues < relative_floor).sum()),
        "negative_eigenvalues": int((eigenvalues < 0).sum()),
        "normalized_min_eigenvalue": float(eigenvalues.min()),
        "normalized_max_eigenvalue": float(eigenvalues.max()),
        "asymmetry": asymmetry, "eigh_dtype": "float64", "root_dtype": "float32",
    }


def solve_full(error_t, scale_a, scale_g, rank, inverse_a=None):
    """M=S_A E^T S_G; LR=S_A^-1 [M]_r S_G^-1, with FULL SVD."""
    if not 0 < rank <= min(error_t.shape) or scale_g.shape != (error_t.shape[1],) * 2:
        raise ValueError("Invalid rank/full-G root shape")
    if any(x.dtype != torch.float32 or not torch.isfinite(x).all() for x in (error_t, scale_a, scale_g)):
        raise ValueError("Full-G solver requires finite FP32 inputs")
    weighted = apply_a(scale_a, error_t) @ scale_g
    u, singular, vh = torch.linalg.svd(weighted, full_matrices=True)
    # Clone slices to release potentially huge unused full-SVD bases.
    u_r, s_r, vh_r = u[:, :rank].clone(), singular[:rank].clone(), vh[:rank].clone()
    del u, singular, vh
    if inverse_a is not None:
        left = inverse_a(scale_a, u_r)
    elif scale_a.ndim == 1:
        left = u_r / scale_a.clamp_min(torch.finfo(scale_a.dtype).eps)[:, None]
    else:
        left = torch.linalg.solve(scale_a, u_r)
    target = s_r[:, None] * vh_r
    right = torch.linalg.solve(scale_g.T, target.T).T
    if not torch.isfinite(left).all() or not torch.isfinite(right).all():
        raise FloatingPointError("Nonfinite full-G correction")
    inverse_residual = relative_error(right @ scale_g, target)
    if inverse_residual > 1e-3:
        raise RuntimeError(f"Full-G inverse residual too large: {inverse_residual}")
    before = float(weighted.double().square().sum())
    after = float((apply_a(scale_a, error_t - left @ right) @ scale_g).double().square().sum())
    if not math.isfinite(after) or after > before * (1 + 1e-5) + 1e-20:
        raise RuntimeError(f"Full-G weighted objective increased: {before} -> {after}")
    return left, right, {"weighted_sse_before": before, "weighted_sse_after": after,
                         "g_inverse_residual": inverse_residual, "svd_full_matrices": True,
                         "solver_variant": "full_g_v1"}
