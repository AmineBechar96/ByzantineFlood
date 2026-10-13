"""Turn saved test-set predictions into the four validation numbers and the
belief bank. numpy only.

predictions: dict test_set -> {"probs": (N,5), "lo": (N,), "hi": (N,),
"level3": (N,), "ambiguous": (N,)} -- missing labels stored as -1.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from . import ordinal as O


def _exact(d):
    m = (d["lo"] == d["hi"]) & (d["lo"] >= 0)
    return m


def evaluate(pred: dict) -> dict:
    out = {}
    if "deepflood_test" in pred:
        d = pred["deepflood_test"]
        m = _exact(d)
        p, y = d["probs"][m], d["lo"][m].astype(int)
        out["deepflood_acc"] = O.bootstrap_ci(O.acc, p, y)
        out["deepflood_within1"] = O.bootstrap_ci(O.within1, p, y)
        out["deepflood_mae"] = O.bootstrap_ci(O.mae, p, y)
        out["deepflood_ece"] = O.ece(p, y)
        na = m & (d["ambiguous"] == 0)
        out["deepflood_acc_unambiguous"] = O.bootstrap_ci(O.acc, d["probs"][na], d["lo"][na].astype(int))
        out["deepflood_per_level_acc"] = {int(c): float((O.ordinal_level(p[y == c]) == c).mean())
                                          for c in np.unique(y)}
    for name in ("deepflood_test_fisheye",):
        if name in pred:
            d = pred[name]
            m = _exact(d)
            out[name.replace("_test", "") + "_acc"] = O.bootstrap_ci(O.acc, d["probs"][m], d["lo"][m].astype(int))
    if "sturm_test" in pred:
        d = pred["sturm_test"]
        flooded = (d["level3"] >= 1).astype(int)
        score = 1 - d["probs"][:, 0]
        out["sturm_dry_vs_flooded_auc"] = O.bootstrap_ci(O.binary_auc, score, flooded)
        out["sturm_dry_vs_flooded_acc"] = O.bootstrap_ci(
            lambda p, f: float(((O.ordinal_level(p) >= 1).astype(int) == f).mean()), d["probs"], flooded)
        fl = d["level3"] >= 1
        out["sturm_kendall_tau_flooded"] = O.bootstrap_ci(O.kendall_tau, d["probs"][fl], d["level3"][fl])
    if "alley_test" in pred:
        d = pred["alley_test"]
        dry = d["hi"] == 0
        out["alley_false_alarm"] = O.bootstrap_ci(O.flooded_rate, d["probs"][dry])
        out["alley_flood_recall"] = O.bootstrap_ci(O.flooded_rate, d["probs"][~dry])
        out["alley_auc"] = O.bootstrap_ci(O.binary_auc, 1 - d["probs"][:, 0], (~dry).astype(int))
    if "fe8k_test" in pred:
        out["fe8k_false_alarm"] = O.bootstrap_ci(O.flooded_rate, pred["fe8k_test"]["probs"])
    return out


def belief_bank(pred: dict):
    """Real detector beliefs indexed by true level: levels 1-4 from DeepFlood,
    level 0 from DeepFlood, FishEye8K test and AlleyFloodNet dry test images."""
    labels, probs, src = [], [], []
    for name in ("deepflood_test", "fe8k_test", "alley_test"):
        if name not in pred:
            continue
        d = pred[name]
        m = _exact(d)
        labels.append(d["lo"][m].astype(int))
        probs.append(d["probs"][m])
        src += [name] * int(m.sum())
    return np.concatenate(labels), np.concatenate(probs), np.array(src)


def write(pred: dict, out_dir: Path, title: str):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    met = evaluate(pred)
    (out_dir / "metrics.json").write_text(json.dumps(met, indent=1))
    labels, probs, src = belief_bank(pred)
    np.savez(out_dir / "belief_bank.npz", labels=labels, probs=probs, source=src)
    lines = [f"# {title}", "", "| metric | value | 95% CI |", "|---|---|---|"]
    for k, v in met.items():
        if isinstance(v, (list, tuple)) and len(v) == 3:
            lines.append(f"| {k} | {v[0]:.3f} | {v[1]:.3f}-{v[2]:.3f} |")
        elif isinstance(v, dict):
            lines.append(f"| {k} | " + ", ".join(f"L{a}: {b:.2f}" for a, b in v.items()) + " | |")
        else:
            lines.append(f"| {k} | {v:.3f} | |")
    lines += ["", f"Belief bank: {len(labels)} beliefs, per level "
              + str(np.bincount(labels, minlength=5).tolist())]
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n")
    return met
