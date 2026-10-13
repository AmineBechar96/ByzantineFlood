"""Shallow-water metrics, the alarm threshold at level 1 (exp12), correlated
detector errors (exp13) and named settings in the runner."""
import importlib.util
import tempfile
from pathlib import Path

import networkx as nx
import numpy as np

from bflood.errors import apply_errors
from bflood.graphs import build_graph, central_node
from bflood.levels import N_LEVELS, one_hot, ordinal_median, peaked_belief
from bflood.metrics import round_metrics
from bflood.rules import WMSR, ByzantineFloodV2
from bflood.sim import run_once, thresholded

ROOT = Path(__file__).resolve().parents[1]
BASE = {
    "graph": {"type": "testbed", "k": 4},
    "rounds": 12,
    "scenario": {"t0": 3, "speed": 1.0},
    "detector": {"type": "placeholder", "rho": 0.87},
    "methods": {"bf_v2": {"rule": "bf_v2"}},
}


class Exact:
    """Stub detector: a confident belief at exactly the requested level."""

    def sample(self, level, rng):
        return one_hot(int(level))


def test_wmsr_threshold_misses_unanimous_level1():
    """W-MSR (F=2) at tau = 0.35: own level 1 and four honest neighbours at
    level 1 give evidence 1/4 < tau, so the thresholded output is 0 (a miss);
    ByzantineFlood outputs 1."""
    w = WMSR(F=2)
    own, rec = 1.0, {u: 1.0 for u in range(1, 5)}
    y, _ = w.decide(0, own, rec, 1)
    score = w.alarm_score(0, own, rec, 1)
    assert y == 1 and abs(score - 0.25) < 1e-12
    assert thresholded(y, score, 0.35) == 0
    bf = ByzantineFloodV2()
    bf.reset(None)
    b = peaked_belief(1, 0.9)
    assert bf.decide(0, b, {u: peaked_belief(1, 0.9) for u in range(1, 5)}, 1)[0] == 1


def test_shallow_metrics():
    g = nx.path_graph(5)
    truth = {0: 0, 1: 1, 2: 1, 3: 1, 4: 3}
    out = {0: 0, 1: 0, 2: 1, 3: 2, 4: 3}
    m = round_metrics(g, set(g), truth, dict(out), out, {v: v for v in g})
    assert abs(m["shallow_recall"] - 2 / 3) < 1e-12 and abs(m["acc_level1"] - 1 / 3) < 1e-12
    m = round_metrics(g, set(g), {v: 0 for v in g}, dict(out), out, {v: v for v in g})
    assert np.isnan(m["shallow_recall"]) and np.isnan(m["acc_level1"])


def _setup(R=12):
    g = build_graph({"type": "testbed", "k": 4})
    truths = [{v: (v + t) % N_LEVELS for v in g} for t in range(R)]
    beliefs = [{v: peaked_belief(truths[t][v], 0.9) for v in g} for t in range(R)]
    return g, truths, beliefs


def test_spatial_patch_shares_one_delta_and_spares_byzantine():
    g, truths, beliefs = _setup()
    byz = {0, 7}
    honest = set(g) - byz
    before = [dict(b) for b in beliefs]
    rec = apply_errors({"type": "spatial", "p": 1.0, "h": 1}, beliefs, truths, g, honest, Exact(),
                       np.random.default_rng(3), 3)
    assert len(rec["patches"]) == len(beliefs)  # p = 1: one patch per round
    for t, c, delta, patch in rec["patches"]:
        assert set(patch) == set(nx.single_source_shortest_path_length(g, c, cutoff=1))
        for v in g:
            if v in patch and v in honest:
                assert ordinal_median(beliefs[t][v]) == int(np.clip(truths[t][v] + delta, 0, 4))
            else:
                assert beliefs[t][v] is before[t][v]


def test_temporal_fault_repeats_one_wrong_belief():
    g, truths, beliefs = _setup(R=30)
    honest = set(g)
    rec = apply_errors({"type": "temporal", "q": 0.2, "mean": 5}, beliefs, truths, g, honest, Exact(),
                       np.random.default_rng(4), 3)
    assert rec["faults"]
    for t, v, lvl, dur in rec["faults"]:
        assert lvl != truths[t][v] and abs(lvl - truths[t][v]) == 1
        for k in range(t, min(t + dur, len(beliefs))):
            assert beliefs[k][v] is beliefs[t][v]


def test_condition_block_uses_adverse_bank():
    g, truths, beliefs = _setup(R=20)
    honest = set(g) - {5}
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "adverse.npz"
        lab = np.arange(N_LEVELS).repeat(3)
        probs = np.array([one_hot((c + 1) % N_LEVELS) for c in lab])  # always wrong
        np.savez(p, labels=lab, probs=probs)
        before = [dict(b) for b in beliefs]
        rec = apply_errors({"type": "condition", "a": 0.25, "path": str(p)}, beliefs, truths, g, honest,
                           Exact(), np.random.default_rng(5), 3)
    s, e = rec["block"]
    assert e - s == 5 and s >= 3 and e <= 20
    for t in range(20):
        for v in g:
            if s <= t < e and v in honest:
                assert ordinal_median(beliefs[t][v]) == (truths[t][v] + 1) % N_LEVELS
            else:
                assert beliefs[t][v] is before[t][v]


def test_v2_range_holds_with_correlated_errors_and_aw():
    """Theorem 2 is relative to the honest observations, whatever their errors."""
    for spec in ({"type": "spatial", "p": 0.3, "h": 2}, {"type": "temporal", "q": 0.1}):
        row = run_once({**BASE, "errors": spec}, 2, "AW", 1)[0]
        assert row["f_local"] <= 2 and row["range_viol"] == 0.0


def test_errors_change_only_what_they_should():
    a = run_once({**BASE, "methods": {"local": {"rule": "local"}}}, 0, "none", 2)[0]
    b = run_once({**BASE, "methods": {"local": {"rule": "local"}},
                  "errors": {"type": "spatial", "p": 0.0, "h": 1}}, 0, "none", 2)[0]
    assert a["acc"] == b["acc"]  # p = 0: identical beliefs


def test_settings_expansion():
    spec = importlib.util.spec_from_file_location("run_experiment", ROOT / "scripts/run_experiment.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    cfg = {"seeds": 50, "graph": {"type": "testbed", "k": 4}, "detector": {"swap": 0.0},
           "settings": [{"name": "base", "set": {}}, {"name": "swap", "set": {"detector.swap": 0.2}},
                        {"name": "rgg", "set": {"graph": {"type": "rgg", "n": 100, "k": 4}, "seeds": 20}}]}
    pts = list(m.expand(cfg))
    assert [p["setting"] for p, _ in pts] == ["base", "swap", "rgg"]
    assert pts[1][1]["detector"]["swap"] == 0.2 and pts[0][1]["detector"]["swap"] == 0.0
    assert pts[2][1]["graph"] == {"type": "rgg", "n": 100, "k": 4} and pts[2][1]["seeds"] == 20
    assert cfg["detector"]["swap"] == 0.0 and "errors" not in cfg
