"""Full FP64 channel matrices; no diagonalization, clipping or trace sampling."""
import math
import torch


def full_pos(gradient, rx, prediction_count):
    product=gradient.double().reshape_as(rx)*rx
    a=product.sum(dim=-1)
    d=float(product.sum())
    return d,d*d/(2*prediction_count),float(a.square().sum())/(2*prediction_count)


def direct_sep(g, metrics, prediction_count):
    return {d:float(((g@m)*g).sum())/(2*prediction_count) for d,m in metrics.items()}


def gram_sep(g, metrics, prediction_count):
    gamma=g.T@g
    result={d:float((gamma*m.T).sum())/(2*prediction_count) for d,m in metrics.items()}
    return result


def validate_scores(scores, scale=1.):
    # Tiny negative roundoff is retained; materially negative PSD scores stop the run.
    for x in scores:
        if not math.isfinite(x) or x < -1e-12*max(scale,1e-30):
            raise FloatingPointError(f'Nonfinite/materially negative PSD score: {x}, scale={scale}')


def relative_scalar(actual, expected, zero_scale=1.):
    return abs(actual-expected)/abs(expected) if expected else abs(actual-expected)/max(zero_scale,1e-30)
