from __future__ import annotations

import math
import torch
import torch.nn.functional as F


def ce_hidden_gradient(hidden, weight, ids, mask, chunk_tokens=128):
    """Exact analytic gradient of summed next-token CE, no vocabulary truncation.

    Teacher parameters are frozen. Backpropagating this seed through the decoder
    includes future-token paths through attention. It is NOT per-loss Fisher.
    """
    if chunk_tokens <= 0 or hidden.ndim != 3 or weight.ndim != 2:
        raise ValueError("Invalid hidden gradient arguments")
    if hidden.dtype != torch.float32 or weight.dtype != torch.float32:
        raise ValueError("This protocol requires FP32 teacher and CE gradient")
    ids, mask = ids.to(hidden.device), mask.to(hidden.device).bool()
    valid = mask[:, :-1] & mask[:, 1:]
    selected = valid.reshape(-1).nonzero().flatten()
    if selected.numel() == 0:
        raise ValueError("No prediction tokens")
    source = hidden[:, :-1].reshape(-1, hidden.shape[-1])
    target = ids[:, 1:].reshape(-1)
    derivative = torch.zeros_like(source)
    nll = 0.0
    with torch.no_grad():
        for positions in selected.split(chunk_tokens):
            logits = F.linear(source[positions], weight)
            log_probs = logits.log_softmax(-1)
            labels = target[positions]
            nll += float((-log_probs.gather(1, labels[:, None])).double().sum())
            grad = log_probs.exp()
            grad[torch.arange(len(labels), device=hidden.device), labels] -= 1
            derivative[positions] = grad @ weight
    result = torch.zeros_like(hidden)
    result[:, :-1] = derivative.reshape_as(hidden[:, :-1])
    if not torch.isfinite(result).all() or not math.isfinite(nll):
        raise FloatingPointError("Nonfinite teacher CE gradient")
    return result, nll, int(selected.numel())


def diagonal_increment(gradient, mask, chunk_tokens=128):
    """FP64 square/reduce, bounded GPU temporary; return an FP64 CPU vector."""
    if gradient.dtype != torch.float32:
        raise ValueError("G gradient must be FP32")
    selected = gradient[mask.to(gradient.device)]
    result = torch.zeros(gradient.shape[-1], dtype=torch.float64, device=gradient.device)
    for chunk in selected.split(chunk_tokens):
        result.add_(chunk.double().square().sum(0))
    if not torch.isfinite(result).all():
        raise FloatingPointError("Nonfinite G statistic")
    return result.cpu()


def diagonal_scale(g_sum, count, relative_floor=1e-6):
    """Normalize mean diagonal to 1, floor in normalized units, then sqrt.

    Constant normalization does not change the exact minimizer; flooring does.
    Both raw statistics and the effective metric are retained for audit.
    """
    if count <= 0 or not 0 < relative_floor < 1:
        raise ValueError("Invalid G count/floor")
    raw = g_sum.double() / count
    if raw.ndim != 1 or not torch.isfinite(raw).all() or (raw < 0).any():
        raise ValueError("G must be finite and nonnegative")
    mean = float(raw.mean())
    if mean <= 0:
        raise ValueError("All-zero G: review gradient collection")
    normalized = raw / mean
    effective = normalized.clamp_min(relative_floor)
    return effective.sqrt().float(), {
        "raw_mean": mean, "relative_floor": relative_floor,
        "floored_channels": int((normalized < relative_floor).sum()),
        "channels": len(raw), "raw_min": float(raw.min()), "raw_max": float(raw.max()),
    }


def apply_a(scale, matrix):
    return scale[:, None] * matrix if scale.ndim == 1 else scale @ matrix


def solve_weighted(error_t, scale_a, scale_g, rank, inverse_a=None):
    """M = S_A E^T S_G; correction = L R, L=S_A^-1 U, R=Sigma Vh S_G^-1.

    Saved QERA roots are used verbatim. No new damping of A is introduced.
    The inverse callback on the server is the pinned QERA implementation.
    """
    if not 0 < rank <= min(error_t.shape):
        raise ValueError("Invalid rank")
    if scale_g.shape != (error_t.shape[1],) or not torch.isfinite(scale_g).all() or (scale_g <= 0).any():
        raise ValueError("G root must be a finite positive vector")
    weighted = apply_a(scale_a, error_t) * scale_g[None, :]
    u, singular, vh = torch.linalg.svd(weighted, full_matrices=False)
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
    return left, right, {"weighted_sse_before": before, "weighted_sse_after": after}


def correction_drift(left, right, ref_left, ref_right, ranks):
    """Compare products, not SVD signs/bases. All requested ranks are checked."""
    result = {}
    for rank in ranks:
        reference = ref_left[:, :rank] @ ref_right[:rank]
        delta = left[:, :rank] @ right[:rank] - reference
        denominator = max(float(torch.linalg.vector_norm(reference.double())), 1e-30)
        value = float(torch.linalg.vector_norm(delta.double())) / denominator
        if not math.isfinite(value):
            raise FloatingPointError("Nonfinite identity-G drift")
        result[str(rank)] = value
    return result
