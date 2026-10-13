"""Worst-case adversary AW (bflood.sim.aw_search)."""
import networkx as nx
import numpy as np

from bflood.levels import one_hot, ordinal_median, peaked_belief
from bflood.rules import ByzantineFloodV1, ByzantineFloodV2, UnfilteredMean
from bflood.sim import aw_objective, aw_payloads, aw_search, aw_start, run_once

BASE = {
    "graph": {"type": "testbed", "k": 4},
    "rounds": 10,
    "scenario": {"t0": 3, "speed": 1.0},
    "detector": {"type": "placeholder", "rho": 0.87},
    "methods": {"bf_v2": {"rule": "bf_v2"}},
}


def _apply(rule, v, own, nbrs, honest, pick, hop=1):
    rec = {}
    for u in nbrs:
        if u in pick:
            if pick[u] is not None:
                rec[u] = pick[u]
        else:
            rec[u] = honest[u]
    return rule.decide(v, own, rec, hop)[0]


def _aw(rule, v, own, nbrs, honest, byz, truth, mode="auto"):
    P = aw_payloads(rule)
    pick, exact = aw_search(rule, v, own, nbrs, honest, byz, truth, 1, P,
                            aw_start(rule, P, truth), mode=mode)
    return pick, exact


def test_aw_candidate_set():
    P = aw_payloads(UnfilteredMean())
    assert len(P) == 11 and P[-1] is None  # 5 confident, 5 unconfident, silence


def test_v2_range_holds_under_aw_in_simulation():
    """(a) Theorem 2: with f_local <= 2, bf_v2 never leaves the honest range
    under AW (the search is exhaustive here, b <= 2)."""
    for nb in (2, 4, 5):
        for seed in (0,):
            row = run_once(BASE, nb, "AW", seed)[0]
            assert row["f_local"] <= 2
            assert row["range_viol"] == 0.0, (nb, seed)
            assert row["search_exact"] == 1.0


def test_v2_range_holds_under_aw_on_stars():
    """(a) The same at decision level, degrees 3-7, f <= 2, random honest
    beliefs and truths."""
    rng = np.random.default_rng(7)
    for _ in range(80):
        d = int(rng.integers(3, 8))
        f = int(rng.integers(1, min(2, d) + 1))
        rule = ByzantineFloodV2()
        rule.reset(None)
        own = peaked_belief(int(rng.integers(0, 5)), rng.uniform(0.51, 0.99))
        nbrs = list(range(1, d + 1))
        byz = nbrs[d - f:]
        honest = {u: peaked_belief(int(rng.integers(0, 5)), rng.uniform(0.51, 0.99))
                  for u in nbrs[:d - f]}
        hl = [ordinal_median(own)] + [ordinal_median(b) for b in honest.values()]
        truth = int(rng.integers(0, 5))
        pick, exact = _aw(rule, 0, own, nbrs, honest, byz, truth)
        assert exact
        y = _apply(rule, 0, own, nbrs, honest, pick)
        assert min(hl) <= y <= max(hl)


def test_aw_beats_a1_on_star_unfiltered_mean():
    """(b) 5-node star (centre 0, leaves 1-4, leaf 4 Byzantine), true level 2
    everywhere, honest detectors lean to level 3. A1 sends level 0 and pulls
    the centre back to the truth; AW finds the push that keeps it wrong."""
    g = nx.star_graph(4)
    truth = 2
    belief = {0: peaked_belief(3, 0.6), 1: peaked_belief(3, 0.6),
              2: peaked_belief(2, 0.7), 3: peaked_belief(3, 0.55)}
    honest = [0, 1, 2, 3]
    rule = UnfilteredMean()
    err = {}
    for attack in ("A1", "AW"):
        e = []
        for v in honest:
            nbrs = list(g.neighbors(v))
            byz = [u for u in nbrs if u == 4]
            if not byz:
                pick = {}
            elif attack == "A1":
                pick = {4: one_hot(0)}  # A1: far target of level 2 is 0
            else:
                pick, _ = _aw(rule, v, belief[v], nbrs, belief, byz, truth)
            e.append(abs(_apply(rule, v, belief[v], nbrs, belief, pick) - truth))
        err[attack] = float(np.mean(e))
    assert err["AW"] > err["A1"], err


def test_exhaustive_and_ascent_agree_b2():
    """(c) b = 2: exhaustive search and coordinate ascent reach the same
    objective (and the same output) for bf_v2 and the unfiltered mean."""
    own = peaked_belief(1, 0.7)
    honest = {1: peaked_belief(1, 0.8), 2: peaked_belief(2, 0.6), 3: peaked_belief(1, 0.9),
              4: peaked_belief(0, 0.55)}
    nbrs = [1, 2, 5, 3, 4, 6]
    byz = [5, 6]
    for rule in (ByzantineFloodV2(), UnfilteredMean()):
        rule.reset(None)
        for truth in (0, 1, 2):
            pe, ex_e = _aw(rule, 0, own, nbrs, honest, byz, truth, mode="exhaustive")
            pa, ex_a = _aw(rule, 0, own, nbrs, honest, byz, truth, mode="ascent")
            ye = _apply(rule, 0, own, nbrs, honest, pe)
            ya = _apply(rule, 0, own, nbrs, honest, pa)
            assert ex_e and not ex_a
            assert aw_objective(ye, truth) == aw_objective(ya, truth), (rule.name, truth)
            assert ye == ya


def test_search_leaves_stateful_rule_untouched():
    """bf_v1 is searched on deep copies: its temporal state is unchanged."""
    rule = ByzantineFloodV1()
    rule.reset(None)
    own = peaked_belief(2, 0.7)
    honest = {1: peaked_belief(2, 0.8), 2: peaked_belief(1, 0.6), 3: peaked_belief(2, 0.9)}
    rule.decide(0, own, honest, 1)  # creates prev[0]
    before = rule.prev[0].copy()
    pick, exact = _aw(rule, 0, own, [1, 2, 3, 4], honest, [4], 2)
    assert exact and np.array_equal(rule.prev[0], before) and set(rule.prev) == {0}


def test_aw_skips_local_and_oracle():
    cfg = {**BASE, "methods": {"local": {"rule": "local"}, "oracle": {"rule": "oracle"},
                               "bf_v2": {"rule": "bf_v2"}}}
    rows = {r["method"]: r for r in run_once(cfg, 2, "AW", 3)}
    assert np.isnan(rows["local"]["search_exact"]) and np.isnan(rows["oracle"]["search_exact"])
    clean = {r["method"]: r for r in run_once(cfg, 2, "A1", 3)}
    assert rows["oracle"]["acc"] == clean["oracle"]["acc"]
    assert rows["local"]["acc"] == clean["local"]["acc"]
