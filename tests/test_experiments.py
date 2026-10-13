"""Step 25: oracle baseline, run-level metrics, Byzantine fractions, sweeps."""
import importlib.util
from pathlib import Path

import numpy as np

from bflood.rules import OracleMedian
from bflood.sim import resolve_n_byz, run_once

ROOT = Path(__file__).resolve().parents[1]
BASE = {
    "graph": {"type": "testbed", "k": 4},
    "rounds": 12,
    "scenario": {"t0": 3, "speed": 1.0},
    "detector": {"type": "placeholder", "rho": 0.87},
    "methods": {"bf_v2": {"rule": "bf_v2"}, "oracle": {"rule": "oracle"},
                "local": {"rule": "local"}},
}


def _load_runner():
    spec = importlib.util.spec_from_file_location("run_experiment", ROOT / "scripts/run_experiment.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_oracle_ignores_byzantine_inputs():
    r = OracleMedian()
    r.byz = {7, 8}
    own = np.array([0, 0, 1.0, 0, 0])
    honest = {1: own.copy(), 2: own.copy()}
    lie = np.array([0, 0, 0, 0, 1.0])
    assert r.decide(0, own, {**honest, 7: lie, 8: lie}, 1)[0] == 2


def test_oracle_unaffected_by_attack():
    a = {x["method"]: x for x in run_once(BASE, 3, "A1", seed=4)}
    b = {x["method"]: x for x in run_once(BASE, 3, "A0", seed=4)}
    assert a["oracle"]["acc"] == b["oracle"]["acc"]


def test_delay_and_miss_defined():
    rows = run_once(BASE, 0, "A0", seed=1)
    for r in rows:
        assert r["delay"] >= 0 and 0.0 <= r["miss"] <= 1.0
        assert np.isnan(r["decide_us"])  # timing off by default


def test_byz_fraction():
    assert resolve_n_byz(0.10, 200) == 20
    assert resolve_n_byz(3, 200) == 3
    assert resolve_n_byz(0.0, 200) == 0
    rows = run_once({**BASE, "graph": {"type": "rgg", "n": 40, "k": 4}}, 0.1, "A1", seed=2)
    assert rows[0]["byz_count"] == 4 and rows[0]["n_byz"] == 0.1


def test_sweep_expansion():
    m = _load_runner()
    cfg = {"detector": {"swap": 0.0}, "sweep": [{"key": "detector.swap", "values": [0.0, 0.2]},
                                                 {"key": "loss", "values": [0.0, 0.1]}]}
    pts = list(m.expand(cfg))
    assert len(pts) == 4
    assert {(c["detector"]["swap"], c["loss"]) for _, c in pts} == {(0.0, 0.0), (0.0, 0.1), (0.2, 0.0), (0.2, 0.1)}
    assert cfg["detector"]["swap"] == 0.0  # original untouched


def test_mixture_output_stays_in_kept_range():
    """Mixture of beliefs whose ordinal medians lie in [a, b] has its
    ordinal median in [a, b] (CDF argument), for random beliefs."""
    from bflood.levels import normalise, ordinal_median
    rng = np.random.default_rng(0)
    for _ in range(2000):
        bs = [normalise(rng.dirichlet(np.ones(5) * 0.5)) for _ in range(rng.integers(2, 7))]
        w = rng.random(len(bs)) + 1e-3
        meds = [ordinal_median(b) for b in bs]
        mix = normalise(np.sum(np.array(bs) * w[:, None], axis=0))
        assert min(meds) <= ordinal_median(mix) <= max(meds)


def test_v2_mixture_range_under_trim():
    """v2 with output=mixture and trim=2 stays within honest observations
    with up to 2 Byzantine neighbours sending extreme beliefs."""
    from bflood.levels import ordinal_median, peaked_belief
    from bflood.rules import ByzantineFloodV2
    rng = np.random.default_rng(1)
    for _ in range(500):
        r = ByzantineFloodV2(output="mixture", trim=2)
        r.reset(None)
        d = int(rng.integers(4, 8))
        own = peaked_belief(int(rng.integers(0, 5)), 0.6 + 0.39 * rng.random())
        rec = {u: peaked_belief(int(rng.integers(0, 5)), 0.5 + 0.49 * rng.random()) for u in range(1, d - 1)}
        hon = [ordinal_median(own)] + [ordinal_median(b) for b in rec.values()]
        for u in (d - 1, d):
            rec[u] = peaked_belief(int(rng.choice([0, 4])), 0.99)
        y = r.decide(0, own, rec, 1)[0]
        assert min(hon) <= y <= max(hon)


def test_reference_graph_fixes_flood_and_faults():
    """With scenario.reference_k, the flood and the Byzantine set do not
    depend on the communication degree; without it, k = 4 is unchanged."""
    base = {**BASE, "methods": {"local": {"rule": "local"}}}
    a = run_once({**base, "scenario": {**BASE["scenario"], "reference_k": 4}}, 2, "A1", seed=3)
    b = run_once(base, 2, "A1", seed=3)
    import math
    same = lambda x, y: x == y or (isinstance(x, float) and math.isnan(x) and math.isnan(y))
    assert all(same(a[0][k], b[0][k]) for k in a[0])  # unchanged when graph.k == reference_k
    c = run_once({**base, "graph": {"type": "testbed", "k": 17},
                  "scenario": {**BASE["scenario"], "reference_k": 4}}, 2, "A1", seed=3)
    # local-only output depends only on beliefs and truth: identical flood
    assert c[0]["acc"] == a[0]["acc"] and c[0]["false_alarm"] == a[0]["false_alarm"]
