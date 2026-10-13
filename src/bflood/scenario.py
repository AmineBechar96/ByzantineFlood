"""Flood propagation and fault placement."""
from __future__ import annotations

from dataclasses import dataclass

import networkx as nx
import numpy as np


@dataclass
class FloodScenario:
    """Front leaves `epicenter` at round t0 and moves `speed` hops per round.
    Level after arrival follows v1 Section III-F:
    min(4, max(1, floor(1 + growth * (t - t_arrival)))).

    Optional pond (exp11, a steep gradient between adjacent cameras):
      pond_mode "isolated": no spreading flood; from round t0 the pond node
                 has level `pond_level` and every other node level 0.
      pond_mode "inside":   the spreading flood as above; once the flood
                 reaches the pond node it has min(4, flood + pond_level - 1),
                 i.e. two levels above the flood at the default level 3.
    """

    g: nx.Graph
    epicenter: int
    t0: int = 3
    speed: float = 1.0
    growth: float = 0.8
    pond_node: int | None = None
    pond_level: int = 3
    pond_mode: str = "isolated"

    def __post_init__(self):
        self.hops = nx.single_source_shortest_path_length(self.g, self.epicenter)
        if self.pond_node is not None and self.pond_mode not in ("isolated", "inside"):
            raise ValueError(f"unknown pond mode {self.pond_mode!r}")

    def flood_level(self, v: int, t: int) -> int:
        h = self.hops.get(v)
        if h is None:
            return 0
        t_arr = self.t0 + h / self.speed
        if t < t_arr:
            return 0
        return int(min(4, max(1, np.floor(1 + self.growth * (t - t_arr)))))

    def level(self, v: int, t: int) -> int:
        if self.pond_node is None:
            return self.flood_level(v, t)
        if self.pond_mode == "isolated":
            return self.pond_level if (v == self.pond_node and t >= self.t0) else 0
        lv = self.flood_level(v, t)
        if v == self.pond_node and lv >= 1:
            return int(min(4, lv + self.pond_level - 1))
        return lv

    def levels(self, t: int) -> dict[int, int]:
        return {v: self.level(v, t) for v in self.g.nodes}


def choose_pond(g: nx.Graph, epicenter: int, spec, seed: int) -> int:
    """`spec` "random": a node of degree >= 4 other than the epicentre, drawn
    from the seed (the same node for every adversary and Byzantine count);
    an int: that node."""
    if spec == "random":
        cands = sorted(v for v in g.nodes if g.degree(v) >= 4 and v != epicenter)
        if not cands:
            raise ValueError("no pond candidate of degree >= 4")
        return int(np.random.default_rng([seed, 5]).choice(cands))
    return int(spec)


def place_faults(g: nx.Graph, epicenter: int, n_byz: int, n_crash: int,
                 placement: str, rng: np.random.Generator, exclude=()) -> tuple[set, set]:
    """Return (byzantine, crashed).

    'bfs': faults are the first nodes in BFS order from the epicentre
           (spatially correlated); Byzantine first, then crashes.
    'random': uniformly random nodes.
    Nodes in `exclude` (pond nodes) are never faulty; they are skipped in
    the order, which is unchanged otherwise.
    """
    nodes = list(g.nodes)
    total = n_byz + n_crash
    if total > len(nodes) - len(set(exclude)):
        raise ValueError("more faults than nodes")
    if placement == "bfs":
        order = [epicenter] + [v for _, v in nx.bfs_edges(g, epicenter)]
    elif placement == "random":
        order = list(rng.permutation(nodes))
    else:
        raise ValueError(f"unknown placement {placement!r}")
    if exclude:
        ex = set(exclude)
        order = [v for v in order if v not in ex]
    byz = set(int(v) for v in order[:n_byz])
    crash = set(int(v) for v in order[n_byz:total])
    return byz, crash


def max_local_byz(g: nx.Graph, byz: set, honest: set) -> int:
    """max over honest v of |B cap N(v)| -- the f actually faced."""
    return max((len(byz & set(g.neighbors(v))) for v in honest), default=0)
