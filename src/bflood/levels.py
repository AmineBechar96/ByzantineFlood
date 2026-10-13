"""Ordinal levels, beliefs and distances.

Levels are 0..4 (dry .. severe). A belief is a probability vector over the
five levels. All v2 code summarises a belief by its ordinal median, never by
its argmax (CLAUDE.md, decision 3).
"""
from __future__ import annotations

import numpy as np

N_LEVELS = 5
EPS = 1e-12


def normalise(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), 0.0, None)
    s = p.sum()
    if s <= EPS:
        return np.full(N_LEVELS, 1.0 / N_LEVELS)
    return p / s


def ordinal_median(p: np.ndarray) -> int:
    """ell(p) = min{c : sum_{c' <= c} p_c' >= 1/2}."""
    cdf = np.cumsum(normalise(p))
    return int(np.argmax(cdf >= 0.5 - 1e-9))


def confidence(p: np.ndarray) -> float:
    """gamma(p): the largest probability, in [1/5, 1]."""
    return float(np.max(normalise(p)))


def hellinger(p: np.ndarray, q: np.ndarray) -> float:
    """Hellinger distance in [0, 1]; a true metric."""
    bc = float(np.sum(np.sqrt(normalise(p) * normalise(q))))
    return float(np.sqrt(max(0.0, 1.0 - bc)))


def sym_kl(p: np.ndarray, q: np.ndarray, eps: float = 1e-10) -> float:
    """Symmetric KL divergence as used by the v1 protocol (not a metric)."""
    p = normalise(np.clip(p, eps, 1.0))
    q = normalise(np.clip(q, eps, 1.0))
    return float(0.5 * np.sum((p - q) * np.log(p / q)))


def one_hot(level: int, mass: float = 0.992) -> np.ndarray:
    """A near-one-hot belief at `level`; remaining mass spread evenly."""
    p = np.full(N_LEVELS, (1.0 - mass) / (N_LEVELS - 1))
    p[int(level)] = mass
    return p


def peaked_belief(level: int, conf: float) -> np.ndarray:
    """Belief with mass `conf` (> 0.5) on `level`, the rest decaying with
    ordinal distance. Guarantees ordinal_median == level when conf > 0.5."""
    level = int(level)
    rest = np.exp(-np.abs(np.arange(N_LEVELS) - level).astype(float))
    rest[level] = 0.0
    rest = rest / rest.sum() * (1.0 - conf)
    rest[level] = conf
    return rest


def weighted_median_level(levels, weights) -> int:
    """Lower weighted median of integer levels.

    Returns the smallest level c whose cumulative weight reaches half the
    total. If the weight of honest inputs strictly exceeds the weight of the
    others, the result lies in [min honest, max honest].
    """
    levels = np.asarray(levels, dtype=int)
    weights = np.asarray(weights, dtype=float)
    if levels.size == 0:
        raise ValueError("empty input")
    order = np.argsort(levels, kind="stable")
    lv, w = levels[order], weights[order]
    cum = np.cumsum(w)
    return int(lv[np.argmax(cum >= 0.5 * cum[-1] - 1e-12)])
