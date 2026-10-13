"""Alarm scores and the threshold sweep (exp10); pond scenario (exp11)."""
import numpy as np

from bflood.graphs import build_graph, central_node
from bflood.levels import peaked_belief
from bflood.rules import build_rule
from bflood.scenario import FloodScenario, choose_pond, place_faults
from bflood.sim import run_once

BASE = {
    "graph": {"type": "testbed", "k": 4},
    "rounds": 12,
    "scenario": {"t0": 3, "speed": 1.0},
    "detector": {"type": "placeholder", "rho": 0.87},
    "methods": {"local": {"rule": "local"}, "unfiltered": {"rule": "unfiltered_mean"},
                "majority": {"rule": "majority"}, "coord_median": {"rule": "coord_median"},
                "wmsr": {"rule": "wmsr", "F": 2}, "bf_v1": {"rule": "bf_v1"},
                "bf_v2": {"rule": "bf_v2"}, "oracle": {"rule": "oracle"}},
}


def test_alarm_scores_in_unit_interval_and_stateless():
    rng = np.random.default_rng(0)
    for spec in BASE["methods"].values():
        rule = build_rule(spec)
        rule.reset(None)
        for _ in range(50):
            own = rule.payload_from_belief(peaked_belief(int(rng.integers(0, 5)), rng.uniform(.51, .99)))
            rec = {u: rule.payload_from_belief(peaked_belief(int(rng.integers(0, 5)), rng.uniform(.51, .99)))
                   for u in range(1, int(rng.integers(2, 7)))}
            prev = dict(getattr(rule, "prev", {}))
            s = rule.alarm_score(0, own, rec, 1)
            assert 0.0 <= s <= 1.0
            assert dict(getattr(rule, "prev", {})).keys() == prev.keys()  # no state change
            rule.decide(0, own, rec, 1)
    dry = peaked_belief(0, 0.9)
    v2 = build_rule({"rule": "bf_v2"})
    v2.reset(None)
    assert v2.alarm_score(0, dry, {1: dry, 2: dry}, 1) < 0.2


def test_threshold_sweep_rows():
    cfg = {**BASE, "sweep_threshold": [0.05, 0.5, 0.95]}
    rows = run_once(cfg, 2, "A1", 1)
    assert len(rows) == 4 * len(BASE["methods"])
    plain = {r["method"]: r for r in run_once(BASE, 2, "A1", 1)}
    for m in BASE["methods"]:
        rs = {r["tau"]: r for r in rows if r["method"] == m}
        assert rs["none"]["acc"] == plain[m]["acc"]  # tau = none is the normal output
        # alarms (hence false alarms) can only fall as tau rises
        assert rs[0.05]["false_alarm"] >= rs[0.5]["false_alarm"] >= rs[0.95]["false_alarm"]


def test_no_attack_case_name():
    rows = run_once(BASE, 0, "none", 2)
    assert rows and all(r["f_local"] == 0 for r in rows)
    try:
        run_once(BASE, 2, "none", 2)
    except ValueError:
        pass
    else:
        raise AssertionError("adversary none with Byzantine cameras must fail")


def test_pond_isolated_levels_and_faults():
    g = build_graph({"type": "testbed", "k": 4})
    epi = central_node(g)
    for seed in range(20):
        p = choose_pond(g, epi, "random", seed)
        assert p != epi and g.degree(p) >= 4
        byz, _ = place_faults(g, epi, 6, 0, "bfs", np.random.default_rng(0), exclude=(p,))
        assert p not in byz and len(byz) == 6
    sc = FloodScenario(g, epi, t0=3, pond_node=5, pond_level=3, pond_mode="isolated")
    assert all(lv == 0 for lv in sc.levels(2).values())
    lv = sc.levels(10)
    assert lv[5] == 3 and sum(lv.values()) == 3
    sc = FloodScenario(g, epi, t0=3, pond_node=5, pond_level=3, pond_mode="inside")
    for t in range(20):
        f = sc.flood_level(5, t)
        assert sc.level(5, t) == (min(4, f + 2) if f >= 1 else 0)
        assert sc.level(epi, t) == sc.flood_level(epi, t)


def test_pond_metrics():
    cfg = {**BASE, "methods": {"local": {"rule": "local"}},
           "scenario": {"t0": 3, "pond": {"node": "random", "level": 3, "mode": "isolated"}}}
    r = run_once(cfg, 2, "A1", 0)[0]
    assert 0.0 <= r["acc_pond"] <= 1.0 and 0.0 <= r["alarm_pond"] <= 1.0
    assert r["alarm_pond"] >= r["acc_pond"]  # level 3 correct implies an alarm
    assert np.isnan(run_once(BASE, 2, "A1", 0)[0]["acc_pond"])  # no pond
