"""Paired article and conditional-label intervals; no bootstrap selection."""
import numpy as np


def sign_state(lo, hi, guard):
    if lo > guard: return 'IMPROVED'
    if hi < -guard: return 'DEGRADED'
    return 'UNRESOLVED'


def ratio_summary(values, roles, seed, count=2000, floor=1e-12, kl=False, repeat_guard=0.):
    # values: article x label x role. KL has exactly one deterministic score per article.
    values = np.asarray(values, dtype=np.float64)
    assert values.ndim == 3 and values.shape[2] == len(roles) and np.isfinite(values).all()
    assert (values >= 0).all() and len(set(roles)) == len(roles)
    n, m, _ = values.shape
    assert n >= 2 and (m == 1 if kl else m >= 2)
    none, selected = roles.index('None'), roles.index('Selected')
    means = values.mean(axis=(0, 1)); window_means = values.mean(axis=1)
    rng = np.random.Generator(np.random.PCG64(seed))
    indices = rng.integers(n, size=(count, n))
    article_boot = window_means[indices].mean(axis=1)
    label_boot = None
    if not kl:
        # Independent stream; never nest a second resampling inside article bootstrap.
        rng = np.random.Generator(np.random.PCG64(seed+1))
        indices = rng.integers(m, size=(count, n, m))
        label_boot = values[np.arange(n)[None, :, None], indices].mean(axis=(1, 2))
    candidates = {name: dict(damage=float(means[i]), recovery=float(1-means[i]/means[none])
                    if means[none] > floor else None) for i, name in enumerate(roles)}
    result = []
    for baseline in ('Marginal-AG', 'A-only'):
        j = roles.index(baseline); delta = float(means[j]-means[selected])
        numerical_relative = 64*np.finfo(float).eps if kl else 2e-10
        absolute_guard = max(float(repeat_guard)*2, numerical_relative*max(float(means[j]), float(means[selected])))
        row = dict(baseline=baseline, selected='Selected', Delta=delta,
                   d=delta/float(means[none]) if means[none] > floor else None,
                   numerical_guard_absolute=absolute_guard)
        for label, samples in [('article', article_boot), ('conditional_labels', label_boot)]:
            if samples is None: continue
            deltas = samples[:, j]-samples[:, selected]
            absolute = np.quantile(deltas, [.025, .975])
            valid = bool(means[none] > floor and np.all(samples[:, none] > floor))
            lo, hi = np.quantile(deltas/samples[:, none], [.025, .975]) if valid else (None, None)
            guard = absolute_guard/float(means[none]) if valid else None
            row[label] = dict(ci_low=float(lo) if valid else None, ci_high=float(hi) if valid else None,
                absolute_ci=[float(v) for v in absolute], valid_ratio=valid,
                status=sign_state(lo, hi, guard) if valid else 'DENOMINATOR_UNRESOLVED')
        result.append(row)
    return dict(candidates=candidates, comparisons=result, bootstrap_count=count,
                article_scope='paired resampling of articles, holding observed per-article label means fixed',
                label_scope=None if kl else 'paired within-window label resampling, fixed articles; separate interval',
                KL_deterministic_per_article=kl)
