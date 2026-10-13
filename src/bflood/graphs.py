"""Communication graphs: k-nearest-neighbour graphs on sensor positions."""
from __future__ import annotations

import networkx as nx
import numpy as np


def knn_graph(pos: np.ndarray, k: int) -> nx.Graph:
    """Join each node to its k nearest nodes, then symmetrise (union)."""
    pos = np.asarray(pos, dtype=float)
    n = len(pos)
    d = np.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=-1)
    np.fill_diagonal(d, np.inf)
    g = nx.Graph()
    for v in range(n):
        g.add_node(v, pos=tuple(pos[v]))
    for v in range(n):
        for u in np.argsort(d[v], kind="stable")[:k]:
            g.add_edge(v, int(u))
    return g


# Real camera positions (WGS84), recovered from the vector figure of the v1
# manuscript (Old/Overleaf/fig3_network_topology.pdf). Camera i is node i-1.
# Linking each camera to its 4 nearest neighbours reproduces the 43 links of
# that figure exactly (tests/test_graphs.py). Full data: data/testbed/cameras.json
TESTBED_LATLON = [
    (25.032933, 121.567135), (25.033041, 121.568230), (25.032868, 121.569297),
    (25.032838, 121.570710), (25.033001, 121.571862), (25.034027, 121.567141),
    (25.034271, 121.568114), (25.034076, 121.569305), (25.034112, 121.570607),
    (25.034151, 121.571737), (25.035405, 121.566892), (25.035309, 121.568160),
    (25.035358, 121.569486), (25.035281, 121.570604), (25.035399, 121.571664),
    (25.036603, 121.566901), (25.036441, 121.568335), (25.036711, 121.569493),
]


def latlon_to_local_m(latlon) -> np.ndarray:
    """Equirectangular projection to metres around the centroid (exact to
    well under 1 m over a 500 m site)."""
    a = np.asarray(latlon, dtype=float)
    lat0 = np.radians(a[:, 0].mean())
    x = (a[:, 1] - a[:, 1].mean()) * 111_320.0 * np.cos(lat0)
    y = (a[:, 0] - a[:, 0].mean()) * 110_574.0
    return np.c_[x, y]


def testbed_positions() -> np.ndarray:
    """Real 18-camera testbed in local metres (about 500 m x 430 m)."""
    return latlon_to_local_m(TESTBED_LATLON)


def grid_positions(jitter_m: float = 10.0, seed: int = 0) -> np.ndarray:
    """Synthetic 3 x 6 grid (the placeholder used before the real positions
    were recovered); kept for sensitivity checks."""
    rng = np.random.default_rng(seed)
    xs = np.linspace(0.0, 550.0, 6)
    ys = np.linspace(0.0, 400.0, 3)
    pos = np.array([(x, y) for y in ys for x in xs])
    return pos + rng.normal(0.0, jitter_m, pos.shape)


def random_geometric_knn(n: int, k: int, seed: int, density_per_km2: float = 80.0) -> nx.Graph:
    """Uniform random positions at roughly the testbed density, kNN graph."""
    rng = np.random.default_rng(seed)
    side = np.sqrt(n / density_per_km2) * 1000.0
    return knn_graph(rng.uniform(0.0, side, size=(n, 2)), k)


def build_graph(spec: dict, seed: int = 0) -> nx.Graph:
    kind = spec.get("type", "testbed")
    k = int(spec.get("k", 4))
    if kind == "testbed":
        return knn_graph(testbed_positions(), k)
    if kind == "grid":
        return knn_graph(grid_positions(seed=spec.get("layout_seed", 0)), k)
    if kind == "rgg":
        return random_geometric_knn(int(spec["n"]), k, seed=spec.get("layout_seed", seed))
    raise ValueError(f"unknown graph type {kind!r}")


def central_node(g: nx.Graph) -> int:
    pos = np.array([g.nodes[v]["pos"] for v in g.nodes])
    c = pos.mean(axis=0)
    return int(np.argmin(np.linalg.norm(pos - c, axis=1)))
