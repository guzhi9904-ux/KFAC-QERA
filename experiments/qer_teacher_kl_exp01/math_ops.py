"""Numerical definitions for protocol v2; no model, filesystem or scheduler code."""
from __future__ import annotations

import contextlib
import hashlib
import math
import time

import torch


def finite(x):
    if not torch.isfinite(x).all():
        raise FloatingPointError("Nonfinite tensor")
    return x


def digest_tensor(x):
    x = x.detach().contiguous().cpu()
    h = hashlib.sha256(str((str(x.dtype), tuple(x.shape))).encode())
    h.update(x.view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def relative(actual, reference):
    numerator = float(torch.linalg.vector_norm((actual - reference).double()))
    denominator = float(torch.linalg.vector_norm(reference.double()))
    return numerator / denominator if denominator else (0.0 if numerator == 0 else math.inf)


def comparison(actual, expected):
    a, b = actual.double().reshape(-1), expected.double().reshape(-1)
    na, nb = float(a.norm()), float(b.norm())
    if nb == 0:
        return {"zero_direction": True, "absolute_error": na, "relative_l2": 0.0 if na == 0 else math.inf,
                "cosine": 1.0 if na == 0 else 0.0}
    return {"zero_direction": False, "absolute_error": float((a-b).norm()),
            "relative_l2": float((a-b).norm())/nb,
            "cosine": float(torch.dot(a,b))/(na*nb) if na else 0.0}


def gram(x, chunk=256, double_product=False):
    """Uncentered Gram: FP32 product / FP64 accumulation, or audited FP64 fallback."""
    x = x.reshape(-1, x.shape[-1])
    result = torch.zeros((x.shape[-1], x.shape[-1]), dtype=torch.float64, device=x.device)
    for z in x.split(chunk):
        z = z.double() if double_product else z.float()
        result.add_((z.T @ z).double())
    return finite(result)


def objective(residual, metric):
    return float(((residual @ metric) * residual).sum())


def choose_metric(raw):
    raw = finite(raw.double())
    sym = (raw+raw.T)/2
    eigenvalues, vectors = torch.linalg.eigh(sym)
    low, high = float(eigenvalues[0]), float(eigenvalues[-1])
    if high <= 0 or low < -1e-6*high:
        raise RuntimeError(f"A_SOLVE_UNRESOLVED: raw spectrum {low}, {high}")
    scale = float(sym.diag().mean())
    identity = torch.eye(len(sym), dtype=sym.dtype, device=sym.device)
    for eta in (0., 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3):
        damping = eta*scale
        if low+damping <= 0 or (high+damping)/(low+damping) > 1e8:
            continue
        solve = sym+damping*identity
        started = time.perf_counter()
        lower = torch.linalg.cholesky(solve)
        cholesky_seconds = time.perf_counter()-started
        error = relative(lower@lower.T, solve)
        if error > 1e-10:
            continue
        started = time.perf_counter()
        root = (vectors*(eigenvalues+damping).sqrt()[None,:])@vectors.T
        root_seconds = time.perf_counter()-started
        root_error = relative(root@root.T, solve)
        if root_error > 1e-10:
            raise RuntimeError("Symmetric root reconstruction failed")
        return solve, lower, root, {"eta":eta,"lambda":damping,"raw_eig_min":low,"raw_eig_max":high,
            "solve_eig_min":low+damping,"solve_eig_max":high+damping,
            "condition":(high+damping)/(low+damping),"symmetry_error":relative(raw,raw.T),
            "symmetrization_change":relative(sym,raw),"cholesky_reconstruction":error,
            "symmetric_reconstruction":root_error,"cholesky_seconds":cholesky_seconds,
            "symmetric_root_construction_seconds":root_seconds}
    raise RuntimeError("A_SOLVE_UNRESOLVED: eta schedule exhausted")


def low_rank(error, root=None, rank=64, triangular=False):
    started = time.perf_counter()
    weighted = error if root is None else error@root
    u, s, vh = torch.linalg.svd(weighted, full_matrices=False)
    svd_seconds = time.perf_counter()-started
    p = u[:,:rank]*s[:rank].sqrt()[None,:]
    q = s[:rank].sqrt()[:,None]*vh[:rank]
    started = time.perf_counter()
    if root is not None:
        q = (torch.linalg.solve_triangular(root.T,q.T,upper=True) if triangular
             else torch.linalg.solve(root.T,q.T)).T
    recovery_seconds = time.perf_counter()-started
    return finite(p), finite(q), s, {"svd_seconds":svd_seconds,"recovery_seconds":recovery_seconds,
        "tail_objective":float(s[rank:].square().sum()),"singular_at_rank":float(s[rank-1]),
        "singular_after_rank":float(s[rank]) if rank<len(s) else 0.,
        "rank_gap":float(s[rank-1]-s[rank]) if rank<len(s) else float(s[rank-1])}


def stable_kl(reference, actual):
    """Complete-vocabulary FP64 KL with cancellation-free expm1 remainder."""
    ref, act = reference.double(), actual.double()
    p = ref.softmax(-1)
    d = act-ref
    v = d-(p*d).sum(-1,keepdim=True)
    # expm1(v)-v loses relative precision for tiny v; use a sixth-order remainder.
    small = v.abs() < 1e-3
    series = v.square()*(0.5+v*(1/6+v*(1/24+v*(1/120+v/720))))
    remainder = torch.where(small,series,torch.expm1(v)-v)
    result = torch.log1p((p*remainder).sum(-1))
    return finite(result)


def sampled_seed(hidden, weight, labels, chunk=128):
    """Derivative of summed FP64-softmax NLL through FP32 logits/head GEMM.

    Gradients of the FP64 probability path are cast back to FP32 at the logits
    cast boundary, matching direct autograd; the model forward remains FP32.
    """
    assert hidden.dtype == weight.dtype == torch.float32
    source = hidden.reshape(-1,hidden.shape[-1])[:-1]
    seed = torch.zeros_like(hidden).reshape(-1,hidden.shape[-1])
    with torch.no_grad():
        for start in range(0,len(source),chunk):
            end = min(start+chunk,len(source))
            probabilities = (source[start:end]@weight.T).double().softmax(-1)
            probabilities[torch.arange(end-start,device=weight.device),labels[start:end].to(weight.device)] -= 1
            seed[start:end] = probabilities.float()@weight
    return finite(seed.reshape_as(hidden))


def stream_seed(window, replicate, base=20260918):
    return int.from_bytes(hashlib.sha256(f"{base}:{window}:{replicate}".encode()).digest()[:8],"little")%(2**63-1)


@contextlib.contextmanager
def intervention(module, residual, alpha):
    old = module.weight.detach().clone()
    original_hash = digest_tensor(old)
    try:
        expected = -float(alpha)*residual.to(old.device).double()
        candidate = (old.double()+expected).float()
        with torch.no_grad():
            module.weight.copy_(candidate)
        audit = comparison(module.weight.detach().double()-old.double(),expected)
        yield audit
    finally:
        with torch.no_grad():
            module.weight.copy_(old)
        if digest_tensor(module.weight) != original_hash:
            raise RuntimeError("Teacher weight restoration failed")


def monte_carlo(records, direction):
    windows = sorted(set(row["window"] for row in records))
    values, counts, ks = [], [], []
    for c in windows:
        rows = sorted((r for r in records if r["window"]==c),key=lambda r:r["replicate"])
        if [r["replicate"] for r in rows] != list(range(len(rows))):
            raise RuntimeError("Noncontiguous or duplicate replicate IDs")
        bs = torch.tensor([r["directions"][direction]["b"] for r in rows],dtype=torch.float64)
        values.append(bs); counts.append(rows[0]["T"]); ks.append(len(rows))
    if len(set(ks)) != 1 or not ks or ks[0]<2:
        raise ValueError("MC summary needs equal K >= 2")
    total = sum(counts)
    mean = sum(float(v.mean())*t/total for v,t in zip(values,counts))
    se = math.sqrt(sum((t/total)**2*float(v.var(unbiased=True))/len(v) for v,t in zip(values,counts)))
    return {"K":ks[0],"q_hat":mean,"MC_SE":se,"ci_low":mean-1.96*se,"ci_high":mean+1.96*se,
            "relative_halfwidth":1.96*se/mean if mean>0 else math.inf}


def plateau(points):
    """Choose the smallest consecutive measured triple, independently of MC."""
    points=sorted(points,key=lambda p:p["alpha"])
    for i in range(len(points)-2):
        group=points[i:i+3]
        if not all(p["valid"] and p["KL_mean"]>0 for p in group):continue
        k=[p["KL_mean"]/p["alpha"]**2 for p in group]
        if max(k)/min(k)-1 <= 0.10:
            x=torch.tensor([math.log(p["alpha"]) for p in group],dtype=torch.float64)
            y=torch.tensor([math.log(p["KL_mean"]) for p in group],dtype=torch.float64)
            slope=float(((x-x.mean())*(y-y.mean())).sum()/((x-x.mean()).square().sum()))
            return {"alphas":[p["alpha"] for p in group],"q_KL":sum(k)/3,"kappa_min":min(k),
                    "kappa_max":max(k),"log_slope":slope}
    return None
