from bflood.sim import run_once

BASE = {
    "graph": {"type": "testbed", "k": 4},
    "rounds": 12,
    "scenario": {"t0": 3, "speed": 1.0},
    "detector": {"type": "placeholder", "rho": 0.87},
    "methods": {"bf_v2": {"rule": "bf_v2"}, "local": {"rule": "local"},
                "wmsr": {"rule": "wmsr", "F": 1}, "bf_v2_T2": {"rule": "bf_v2", "hops": 2}},
}


def test_run_is_deterministic():
    import math
    a = run_once(BASE, 2, "A1", seed=5)
    b = run_once(BASE, 2, "A1", seed=5)
    same = lambda x, y: x == y or (isinstance(x, float) and math.isnan(x) and math.isnan(y))
    assert len(a) == len(b)
    for ra, rb in zip(a, b):
        assert ra.keys() == rb.keys() and all(same(ra[k], rb[k]) for k in ra)


def test_local_only_ignores_adversary():
    a = {r["method"]: r for r in run_once(BASE, 0, "A1", seed=3)}
    b = {r["method"]: r for r in run_once(BASE, 3, "A1", seed=3)}
    # same beliefs, same detector; only honest set differs, so compare
    # 'induced' errors, which must be zero for local-only
    assert a["local"]["induced"] == 0.0 and b["local"]["induced"] == 0.0


def test_containment_one_hop():
    """Proposition 2: honest nodes with no Byzantine neighbour produce the
    same outputs whether the Byzantine nodes attack or behave honestly."""
    import numpy as np
    from bflood.adversaries import AdvContext, Adversary
    from bflood.beliefs import PlaceholderDetector
    from bflood.graphs import build_graph, central_node
    from bflood.levels import ordinal_median
    from bflood.rules import ByzantineFloodV2
    from bflood.scenario import FloodScenario, place_faults

    g = build_graph({"type": "testbed", "k": 4})
    epi = central_node(g)
    scen = FloodScenario(g, epi)
    byz, _ = place_faults(g, epi, 2, 0, "bfs", np.random.default_rng(0))
    det = PlaceholderDetector(0.87)
    rng = np.random.default_rng(1)
    beliefs = [{v: det.sample(scen.level(v, t), rng) for v in g} for t in range(10)]
    adv = Adversary("A1")
    outs = []
    for attack in (True, False):
        rule = ByzantineFloodV2(); rule.reset(g)
        o = []
        for t in range(10):
            ctx = AdvContext(t, 1, scen.levels(t), beliefs[t], epi, det, 0)
            res = {}
            for v in g:
                if v in byz:
                    continue
                rec = {u: (adv.message(u, v, ctx) if (attack and u in byz) else beliefs[t][u])
                       for u in g.neighbors(v)}
                res[v] = rule.decide(v, beliefs[t][v], rec, 1)[0]
            o.append(res)
        outs.append(o)
    far = [v for v in g if v not in byz and not (set(g.neighbors(v)) & byz)]
    assert far, "test needs nodes away from the Byzantine set"
    for t in range(10):
        for v in far:
            assert outs[0][t][v] == outs[1][t][v]
