"""Correlated detector errors (exp13, Oct 2026).

Applied to honest cameras only, on top of the independent belief-bank draws
of sim.run_once, with a separate random stream ([seed, 6]): the base beliefs
are the same as without the error model, so cases are paired. Config key
`errors`:

  {type: spatial, p, h}    each round, with probability p, a centre camera is
                           drawn uniformly; every honest camera within h hops
                           (physical layout) draws its belief from the bank at
                           level clip(lambda + delta, 0, 4), one delta in
                           {-1, +1} (equal probability) for the whole patch.
  {type: temporal, q, mean}  each honest camera not already faulty starts a
                           stuck fault with probability q per round, lasting a
                           geometric number of rounds (mean `mean`, default 5);
                           at the start it draws a wrong belief (bank belief at
                           lambda + delta, delta = +-1, the other sign if out
                           of 0..4) and repeats it every round of the fault.
  {type: condition, a, bank, file}  one contiguous block of round(a * R)
                           rounds (start drawn uniformly so that the block
                           lies in the evaluated rounds t0..R-1) in which
                           every honest camera draws from the adverse bank
                           (default detector_base / belief_bank_adverse.npz).
"""
from __future__ import annotations

import networkx as nx
import numpy as np

from .beliefs import build_detector
from .levels import N_LEVELS

TOP = N_LEVELS - 1


def apply_errors(spec: dict, beliefs: list, truths: list, g: nx.Graph, honest: set,
                 detector, rng: np.random.Generator, t0: int) -> dict:
    """Modify beliefs[t][v] in place for honest v. Returns a small record of
    what was applied (for tests and logging)."""
    kind = spec["type"]
    R = len(beliefs)
    H = sorted(honest)
    rec = {"type": kind, "affected": 0}
    if kind == "spatial":
        p, h = float(spec["p"]), int(spec["h"])
        nodes = sorted(g.nodes)
        rec["patches"] = []
        for t in range(R):
            if rng.random() >= p:
                continue
            c = int(rng.choice(nodes))
            delta = int(rng.choice([-1, 1]))
            patch = sorted(nx.single_source_shortest_path_length(g, c, cutoff=h))
            rec["patches"].append((t, c, delta, patch))
            for v in patch:
                if v in honest:
                    lvl = int(np.clip(truths[t][v] + delta, 0, TOP))
                    beliefs[t][v] = detector.sample(lvl, rng)
                    rec["affected"] += 1
    elif kind == "temporal":
        q, mean = float(spec["q"]), float(spec.get("mean", 5.0))
        left = {v: 0 for v in H}
        stuck = {}
        rec["faults"] = []
        for t in range(R):
            for v in H:
                if left[v] > 0:
                    beliefs[t][v] = stuck[v]
                    left[v] -= 1
                    rec["affected"] += 1
                elif rng.random() < q:
                    lam = truths[t][v]
                    d = int(rng.choice([-1, 1]))
                    lvl = lam + d if 0 <= lam + d <= TOP else lam - d
                    stuck[v] = detector.sample(lvl, rng)
                    dur = int(rng.geometric(1.0 / mean))
                    left[v] = dur - 1
                    beliefs[t][v] = stuck[v]
                    rec["faults"].append((t, v, lvl, dur))
                    rec["affected"] += 1
    elif kind == "condition":
        a = float(spec["a"])
        bank = ({"path": spec["path"]} if "path" in spec else
                {"bank": spec.get("bank", "detector_base"), "file": spec.get("file", "belief_bank_adverse.npz")})
        adverse = build_detector({"type": "belief_bank", **bank})
        L = int(round(a * R))
        lo = min(t0, R - L)
        start = int(rng.integers(lo, R - L + 1))
        rec["block"] = (start, start + L)
        for t in range(start, start + L):
            for v in H:
                beliefs[t][v] = adverse.sample(truths[t][v], rng)
                rec["affected"] += 1
    else:
        raise ValueError(f"unknown error model {kind!r}")
    return rec
