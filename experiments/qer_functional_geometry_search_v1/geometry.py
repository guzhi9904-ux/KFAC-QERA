"""Nine residual-SVD candidates in a fixed spectral family; no H(E) SVD."""
import math
import torch


def relative(a, b):
    numerator = float((a-b).double().norm()); denominator = float(b.double().norm())
    return numerator/denominator if denominator else (0.0 if numerator == 0 else math.inf)


def scalar_check(actual, expected, scale, tolerance=1e-8):
    allowed = tolerance*max(abs(actual), abs(expected)) + 64*torch.finfo(torch.float64).eps*abs(scale)
    if not math.isfinite(actual+expected+scale) or abs(actual-expected) > allowed:
        raise ArithmeticError(('scalar identity', actual, expected, allowed))
    return dict(absolute_error=abs(actual-expected), allowed=allowed)


def spectrum(factor):
    assert factor.dtype == torch.float64 and torch.isfinite(factor).all()
    assert factor.ndim == 2 and factor.shape[0] == factor.shape[1]
    assert relative(factor, factor.T) <= 1e-10, 'Asymmetric solve factor'
    mean = float(factor.diag().mean()); assert mean > 0
    normalized = factor/mean
    values, vectors = torch.linalg.eigh(normalized)
    assert float(values[0]) > 0, 'Nonpositive eigenvalue; no clipping or extra damping allowed'
    error = relative((vectors*values)@vectors.T, normalized)
    assert error <= 1e-8
    return values, vectors, dict(trace_mean=mean, condition=float(values[-1]/values[0]),
                                minimum=float(values[0]), maximum=float(values[-1]), reconstruction=error)


def candidate_key(a, b):
    return f'a{a:g}_b{b:g}'


def construct(error, av, au, gv, gu, a, b, rank, coordinates=None):
    assert error.dtype == torch.float64 and rank <= min(error.shape)
    # Eigenvectors are fixed. An orthogonal change of coordinates preserves the SVD objective.
    if coordinates is None: coordinates = gu.T@error@au
    left = gv.pow(b/2); right = av.pow(a/2)
    transformed = left[:, None]*coordinates*right[None, :]
    kwargs = dict(driver='gesvd') if transformed.is_cuda else {}
    u, singular, vh = torch.linalg.svd(transformed, full_matrices=False, **kwargs)
    reconstruction = relative((u*singular)@vh, transformed)
    assert reconstruction <= 1e-8
    p = gu@((u[:, :rank]*singular[:rank].sqrt())/left[:, None])
    q = ((singular[:rank].sqrt()[:, None]*vh[:rank])/right[None, :])@au.T
    correction = p@q
    actual = left[:, None]*(gu.T@(error-correction)@au)*right[None, :]
    tail = float(singular[rank:].square().sum()/2)
    objective = float(actual.square().sum()/2)
    check = scalar_check(objective, tail, float(transformed.square().sum()))
    prefix = (u[:, :rank]*singular[:rank])@vh[:rank]
    inverse_error = relative(left[:, None]*(gu.T@correction@au)*right[None, :], prefix)
    assert inverse_error <= 1e-8
    gap = float(singular[rank-1]-singular[rank]) if rank < len(singular) else None
    degenerate = gap is not None and gap <= 1e-10*float(singular[0])
    return p, q, dict(a=a, b=b, rank=rank, SVD_reconstruction=reconstruction,
        tail_energy_half=tail, objective=objective, tail_check=check, inverse_error=inverse_error,
        rank_boundary_gap=gap, rank_boundary_near_degenerate=degenerate,
        compensation_unique=not degenerate, norm_C=float(correction.norm()),
        condition_A=float((av[-1]/av[0])**a), condition_G=float((gv[-1]/gv[0])**b),
        algorithm='FP64 full-channel dense SVD; fixed eigenbasis; no randomized approximation')


def choose(scores, grid, none_score, floor=1e-12):
    assert all(math.isfinite(v) and v >= 0 for v in scores.values())
    assert set(scores) == set(grid)
    if none_score <= floor:
        return dict(selected=None, status='DENOMINATOR_UNRESOLVED', absolute_scores=scores)
    selected = min(scores, key=lambda key: (scores[key],
        (grid[key][0]-1)**2+(grid[key][1]-1)**2, grid[key][0], grid[key][1]))
    return dict(selected=selected, parameters=grid[selected], status='SELECTED',
                objective_ratio=scores[selected]/none_score,
                exact_ties=[k for k in scores if scores[k] == scores[selected]])
