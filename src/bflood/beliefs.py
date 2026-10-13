"""Belief sources: what an honest sensor's detector reports.

PlaceholderDetector is a stand-in until the belief bank of real detector
outputs exists. Results produced with it must be labelled 'placeholder'.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .levels import N_LEVELS, normalise, peaked_belief


class PlaceholderDetector:
    """Correct level with probability rho; otherwise off by one level.
    Wrong outputs are less confident than correct ones (a calibrated
    detector's typical behaviour)."""

    name = "placeholder"

    def __init__(self, rho: float = 0.87):
        if not 0.0 < rho <= 1.0:
            raise ValueError("rho must be in (0, 1]")
        self.rho = rho

    def sample(self, true_level: int, rng: np.random.Generator) -> np.ndarray:
        if rng.random() < self.rho:
            obs, conf = true_level, 0.5 + 0.49 * rng.beta(6, 2)
        else:
            if true_level == 0:
                obs = 1
            elif true_level == N_LEVELS - 1:
                obs = N_LEVELS - 2
            else:
                obs = true_level + (1 if rng.random() < 0.5 else -1)
            conf = 0.5 + 0.49 * rng.beta(2, 3)
        return peaked_belief(obs, conf)


class BeliefBank:
    """Real detector outputs indexed by true class.

    Expects an .npz with `labels` (int, shape [N]) and `probs` (float,
    shape [N, 5]) produced by the detector on real test images.
    `swap` replaces a belief by one from an adjacent class with that
    probability, to sweep the effective accuracy rho downward.
    """

    name = "belief_bank"

    def __init__(self, labels: np.ndarray, probs: np.ndarray, swap: float = 0.0):
        labels = np.asarray(labels, dtype=int)
        probs = np.asarray(probs, dtype=float)
        if probs.ndim != 2 or probs.shape[1] != N_LEVELS or len(labels) != len(probs):
            raise ValueError("bad belief bank shape")
        self.by_class = {c: probs[labels == c] for c in range(N_LEVELS)}
        missing = [c for c, a in self.by_class.items() if len(a) == 0]
        if missing:
            raise ValueError(f"belief bank has no samples for classes {missing}")
        self.swap = swap

    @classmethod
    def load(cls, path: str | Path, swap: float = 0.0) -> "BeliefBank":
        d = np.load(path)
        return cls(d["labels"], d["probs"], swap=swap)

    def sample(self, true_level: int, rng: np.random.Generator) -> np.ndarray:
        c = int(true_level)
        if self.swap > 0 and rng.random() < self.swap:
            nb = [x for x in (c - 1, c + 1) if 0 <= x < N_LEVELS]
            c = int(rng.choice(nb))
        pool = self.by_class[c]
        return normalise(pool[rng.integers(len(pool))])


ROOT = Path(__file__).resolve().parents[2]


def resolve_bank(spec: dict) -> Path:
    """`path` is used as given; `bank: <detector run name>` is looked up where
    Modal writes it (results/detector/<run>/ensemble/) and where
    `modal volume get` puts it locally (results/<run>/ensemble/). `file`
    names another bank of the same run (e.g. belief_bank_adverse.npz)."""
    if "path" in spec:
        return Path(spec["path"])
    b = spec["bank"]
    f = spec.get("file", "belief_bank.npz")
    cands = [ROOT / "results" / "detector" / b / "ensemble" / f,
             ROOT / "results" / b / "ensemble" / f]
    if f == "belief_bank.npz":
        cands.append(ROOT / "data" / "belief_bank" / f"{b}.npz")
    for c in cands:
        if c.exists():
            return c
    raise FileNotFoundError(f"belief bank {b!r} not found; tried: " + ", ".join(map(str, cands)))


_BANKS: dict = {}


def build_detector(spec: dict):
    kind = spec.get("type", "placeholder")
    if kind == "placeholder":
        return PlaceholderDetector(rho=float(spec.get("rho", 0.87)))
    if kind == "belief_bank":
        p = resolve_bank(spec)
        if p not in _BANKS:  # load once per process
            d = np.load(p)
            _BANKS[p] = (d["labels"], d["probs"])
        labels, probs = _BANKS[p]
        return BeliefBank(labels, probs, swap=float(spec.get("swap", 0.0)))
    raise ValueError(f"unknown detector {kind!r}")
