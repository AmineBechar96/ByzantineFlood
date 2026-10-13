"""Per-round metrics, computed on honest, operating sensors only."""
from __future__ import annotations

import numpy as np

METRICS = ("acc", "acc_flooded", "acc_near", "false_alarm", "mae", "range_viol",
           "spatial", "corrected", "induced", "acc_pond", "alarm_pond",
           "shallow_recall", "acc_level1")
# shallow_recall, acc_level1 (Oct 2026): over the honest operating cameras
# whose true level is 1 (shallow water), the fraction with output >= 1 (an
# alarm) and the fraction with output exactly 1; nan when there is none.
# acc_pond, alarm_pond (exp11): accuracy and alarm rate (output >= 1) at the
# honest pond nodes, over the rounds in which the pond holds water (truth
# >= 1: from t0 in mode isolated, from flood arrival in mode inside); nan
# without a pond.
# Per-run (not per-round) metrics, computed in sim.run_once:
#   delay      mean rounds from flood arrival (truth >= 1) to first alarm
#              (output >= 1) at an honest node; nodes never alarmed count as
#              (rounds - arrival), i.e. right-censored at the horizon
#   miss       fraction of reached honest nodes never alarmed before the end
#   decide_us  mean CPU microseconds per rule.decide call (only when
#              cfg["timing"] is true; wall-clock, so not deterministic)
#   search_exact  adversary AW only: fraction of the searched decisions
#              (honest camera, round, hop with >= 1 Byzantine neighbour) in
#              which the search was exhaustive; nan when nothing was searched
RUN_METRICS = ("delay", "miss", "decide_us", "search_exact")


def round_metrics(g, honest, truth, obs, out, hops_from_epi, near_hops=2, pond=()):
    """truth/obs/out: node -> level. `pond`: the pond nodes (exp11). Returns
    a dict of metric -> value (nan if the metric's population is empty)."""
    H = sorted(honest)
    y = np.array([out[v] for v in H])
    lam = np.array([truth[v] for v in H])
    o = np.array([obs[v] for v in H])

    def frac(mask, cond):
        return float(np.mean(cond[mask])) if mask.any() else float("nan")

    correct = y == lam
    near = np.array([hops_from_epi.get(v, 10**9) <= near_hops for v in H])
    wet_pond = np.array([v in pond for v in H]) & (lam >= 1)
    res = {
        "acc": float(correct.mean()),
        "acc_flooded": frac(lam >= 1, correct),
        "acc_near": frac(near, correct),
        "false_alarm": frac(lam == 0, y >= 1),
        "mae": float(np.mean(np.abs(y - lam))),
        # own detector wrong -> fused output right
        "corrected": frac(o != lam, correct),
        # own detector right -> fused output wrong
        "induced": frac(o == lam, ~correct),
        "acc_pond": frac(wet_pond, correct),
        "alarm_pond": frac(wet_pond, y >= 1),
        "shallow_recall": frac(lam == 1, y >= 1),
        "acc_level1": frac(lam == 1, correct),
    }
    # Range (Definition 1a): output within honest observations of N[v]
    hs = set(honest)
    viol = []
    for v in H:
        honest_obs = [obs[v]] + [obs[u] for u in g.neighbors(v) if u in hs]
        viol.append(not (min(honest_obs) <= out[v] <= max(honest_obs)))
    res["range_viol"] = float(np.mean(viol))
    pairs = [(u, v) for u, v in g.edges if u in hs and v in hs]
    res["spatial"] = (float(np.mean([abs(out[u] - out[v]) <= 1 for u, v in pairs]))
                      if pairs else float("nan"))
    return res
