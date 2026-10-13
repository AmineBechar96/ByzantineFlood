import json
from pathlib import Path

from bflood.graphs import build_graph

ROOT = Path(__file__).resolve().parents[1]


def test_testbed_reproduces_v1_links():
    g = build_graph({"type": "testbed", "k": 4})
    links = {tuple(sorted((a + 1, b + 1))) for a, b in g.edges}
    v1 = {tuple(e) for e in json.loads((ROOT / "data/testbed/cameras.json").read_text())["edges_v1"]}
    assert g.number_of_nodes() == 18 and links == v1 and len(links) == 43


def test_testbed_degrees():
    g = build_graph({"type": "testbed", "k": 4})
    degs = sorted(d for _, d in g.degree())
    assert degs[0] == 4 and degs[-1] == 7
