"""Preregistered statistics for the validation harness.

Pure numpy and the standard library (no scipy), so a receipt can be
recomputed on any install.  Every resampling procedure takes an explicit seed
from the spec; the same trials always give the same interval.
"""
from __future__ import annotations

import math
from typing import Dict, Sequence

import numpy as np


def bootstrap_mean(values: Sequence[float], *, seed: int, n_boot: int = 10000, ci: float = 0.95) -> dict:
    """Mean, SD and percentile bootstrap CI of the mean (the WP5 procedure)."""
    values = np.asarray(values, float)
    if len(values) == 0:
        return dict(mean=None, sd=None, ci=[None, None], n=0)
    rng = np.random.default_rng(seed)
    means = values[rng.integers(0, len(values), (n_boot, len(values)))].mean(axis=1)
    lo, hi = (1 - ci) / 2 * 100, (1 + ci) / 2 * 100
    sd = float(values.std(ddof=1)) if len(values) > 1 else 0.0
    return dict(mean=float(values.mean()), sd=sd,
                ci=[float(np.percentile(means, lo)), float(np.percentile(means, hi))],
                n=int(len(values)), n_positive=int((values > 0).sum()),
                n_negative=int((values < 0).sum()), n_zero=int((values == 0).sum()))


def bootstrap_ratio_of_means(num: Sequence[float], den: Sequence[float], *, seed: int,
                             n_boot: int = 10000, ci: float = 0.95) -> dict:
    """Ratio mean(num)/mean(den), resampling paired units (seeds) together.

    Resamples whose denominator mean is <= 0 are counted and make the interval
    undefined rather than being dropped silently.
    """
    num, den = np.asarray(num, float), np.asarray(den, float)
    if len(num) != len(den) or len(num) == 0:
        raise ValueError('num and den must be paired and non-empty')
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(num), (n_boot, len(num)))
    d = den[idx].mean(axis=1)
    bad = int((d <= 0).sum())
    point = float(num.mean() / den.mean()) if den.mean() > 0 else None
    if bad:
        return dict(ratio=point, ci=[None, None], n=int(len(num)), undefined_resamples=bad)
    r = num[idx].mean(axis=1) / d
    lo, hi = (1 - ci) / 2 * 100, (1 + ci) / 2 * 100
    return dict(ratio=point, ci=[float(np.percentile(r, lo)), float(np.percentile(r, hi))],
                n=int(len(num)), undefined_resamples=0)


def _binom_cdf(k: int, n: int, p: float) -> float:
    if p <= 0:
        return 1.0
    if p >= 1:
        return 1.0 if k >= n else 0.0
    lp, lq = math.log(p), math.log1p(-p)
    return min(1.0, sum(math.exp(math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1)
                                 + i * lp + (n - i) * lq) for i in range(0, k + 1)))


def clopper_pearson(k: int, n: int, ci: float = 0.95) -> list:
    """Exact binomial CI by bisection on the binomial CDF."""
    if n <= 0:
        return [None, None]
    alpha = 1 - ci
    lower = 0.0 if k == 0 else _cp_lower(k, n, alpha)
    upper = 1.0 if k == n else _cp_upper(k, n, alpha)
    return [lower, upper]


def _bisect(f, lo=0.0, hi=1.0, iters=200):
    """Root of a function that is increasing in p on [lo, hi]."""
    for _ in range(iters):
        mid = (lo + hi) / 2
        if f(mid) < 0:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def _cp_lower(k, n, alpha):
    # P(X >= k | p) = alpha/2, increasing in p.
    return _bisect(lambda p: (1 - _binom_cdf(k - 1, n, p)) - alpha / 2)


def _cp_upper(k, n, alpha):
    # P(X <= k | p) = alpha/2, decreasing in p.
    return _bisect(lambda p: alpha / 2 - _binom_cdf(k, n, p))


def proportion(k: int, n: int, ci: float = 0.95) -> dict:
    return dict(k=int(k), n=int(n), p=(k / n if n else None), ci=clopper_pearson(k, n, ci), method='clopper-pearson')


def bootstrap_slope(groups: Dict[float, Sequence[float]], *, seed: int, n_boot: int = 10000,
                    ci: float = 0.95) -> dict:
    """OLS slope of per-trial outcomes on x, resampling trials within each x.

    ``groups`` maps x (e.g. log10 r/v) to that condition's per-trial outcomes
    (e.g. 0/1 GF spike).  Stratified resampling keeps the design fixed.
    """
    xs = sorted(groups)
    arrays = [np.asarray(groups[x], float) for x in xs]
    if len(xs) < 2 or any(len(a) == 0 for a in arrays):
        raise ValueError('need at least two non-empty x levels')

    def slope(means):
        x = np.asarray(xs, float)
        weights = np.asarray([len(a) for a in arrays], float)
        xm = np.average(x, weights=weights)
        ym = np.average(means, weights=weights)
        return float(np.sum(weights * (x - xm) * (means - ym)) / np.sum(weights * (x - xm) ** 2))

    point = slope(np.asarray([a.mean() for a in arrays]))
    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot)
    samples = [a[rng.integers(0, len(a), (n_boot, len(a)))].mean(axis=1) for a in arrays]
    for b in range(n_boot):
        boots[b] = slope(np.asarray([s[b] for s in samples]))
    lo, hi = (1 - ci) / 2 * 100, (1 + ci) / 2 * 100
    return dict(slope=point, ci=[float(np.percentile(boots, lo)), float(np.percentile(boots, hi))],
                x=[float(x) for x in xs], means=[float(a.mean()) for a in arrays],
                n=[int(len(a)) for a in arrays], method='weighted OLS on condition means, stratified bootstrap')


def spearman(x: Sequence[float], y: Sequence[float]) -> float | None:
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 3:
        return None

    def rank(a):
        order = a.argsort(kind='mergesort')
        ranks = np.empty(len(a))
        ranks[order] = np.arange(len(a))
        for value in np.unique(a):   # average ties
            m = a == value
            ranks[m] = ranks[m].mean()
        return ranks
    rx, ry = rank(x), rank(y)
    if rx.std() == 0 or ry.std() == 0:
        return None
    return float(np.corrcoef(rx, ry)[0, 1])
