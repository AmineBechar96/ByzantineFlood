"""Ordinal (cumulative-link) maths and evaluation metrics -- numpy only.

The detector outputs four cumulative probabilities q_k = P(level > k),
k = 0..3, non-increasing in k. They convert to a belief over the five levels
    p_0 = 1 - q_0,  p_c = q_{c-1} - q_c,  p_4 = q_3,
whose ordinal median (bflood.levels.ordinal_median) is the number of k with
q_k > 1/2. Interval labels [lo, hi] supervise threshold k only when they
determine it: target 1 if lo > k, 0 if hi <= k, masked otherwise.
"""
from __future__ import annotations

import numpy as np
from scipy import stats

N_LEVELS = 5
N_THRESH = N_LEVELS - 1


def interval_targets(lo, hi):
    lo = np.asarray(lo, dtype=float)[:, None]
    hi = np.asarray(hi, dtype=float)[:, None]
    k = np.arange(N_THRESH)[None, :]
    pos, neg = lo > k, hi <= k
    return pos.astype(np.float32), (pos | neg)


def beliefs_from_cum(q):
    q = np.clip(np.asarray(q, dtype=float), 0.0, 1.0)
    q = np.minimum.accumulate(q, axis=1)  # enforce non-increasing
    p = np.concatenate([1 - q[:, :1], q[:, :-1] - q[:, 1:], q[:, -1:]], axis=1)
    p = np.clip(p, 0.0, None)
    return p / p.sum(1, keepdims=True)


def ordinal_level(p):
    return np.argmax(np.cumsum(p, axis=1) >= 0.5 - 1e-9, axis=1)


def expected_level(p):
    return p @ np.arange(N_LEVELS)


def ece(p, y, bins=10):
    """Expected calibration error of the ordinal-median prediction."""
    yhat = ordinal_level(p)
    conf = p[np.arange(len(p)), yhat]
    correct = (yhat == y).astype(float)
    edges = np.linspace(0, 1, bins + 1)
    e = 0.0
    for a, b in zip(edges[:-1], edges[1:]):
        m = (conf > a) & (conf <= b)
        if m.any():
            e += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(e)


def binary_auc(score, label):
    label = np.asarray(label).astype(bool)
    if label.all() or (~label).all():
        return float("nan")
    r = stats.rankdata(score)
    n1, n0 = label.sum(), (~label).sum()
    return float((r[label].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def bootstrap_ci(fn, *arrays, n=2000, seed=0, alpha=0.05):
    """Point estimate and percentile CI of fn(*arrays) over resampled rows."""
    est = fn(*arrays)
    rng = np.random.default_rng(seed)
    N = len(arrays[0])
    vals = []
    for _ in range(n):
        idx = rng.integers(0, N, N)
        v = fn(*(np.asarray(a)[idx] for a in arrays))
        if np.isfinite(v):
            vals.append(v)
    lo, hi = np.percentile(vals, [100 * alpha / 2, 100 * (1 - alpha / 2)]) if vals else (np.nan, np.nan)
    return float(est), float(lo), float(hi)


# metric functions used by report.py ------------------------------------------
def acc(p, y):
    return float((ordinal_level(p) == y).mean())


def within1(p, y):
    return float((np.abs(ordinal_level(p) - y) <= 1).mean())


def mae(p, y):
    return float(np.abs(ordinal_level(p) - y).mean())


def flooded_rate(p):
    """Fraction predicted at level >= 1 (false-alarm rate on dry images)."""
    return float((ordinal_level(p) >= 1).mean())


def kendall_tau(p, y):
    t = stats.kendalltau(expected_level(p), y).statistic
    return float(t) if np.isfinite(t) else float("nan")
