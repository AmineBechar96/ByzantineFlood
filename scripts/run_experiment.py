"""Run an experiment config: every sweep point x adversary x Byzantine count x seed.

Usage:
  python scripts/run_experiment.py configs/<name>.yaml [--workers N]
         [--placeholder] [--bank RUN] [--quick]

  --placeholder  replace the detector by PlaceholderDetector(rho=0.87); output
                 goes to results/<name>_placeholder/ (dry runs before the
                 belief bank exists; such results must never be reported)
  --bank RUN     use the belief bank of detector run RUN (overrides config)
  --quick        2 seeds only, to check that a config runs

Optional config key `sweep`: a list of {key: dotted.path, values: [...]}.
The cartesian product of all sweeps is run; each key becomes a column
(e.g. key detector.swap -> column detector.swap).
`n_byz` entries are counts, or fractions of the nodes if floats in (0, 1).
Optional config key `cases`: a list of {adversary: ..., n_byz: ...} run
instead of the product adversaries x n_byz (adversary `none` = no attack,
n_byz 0; `AW` = the worst case searched in bflood.sim).
Optional config key `settings`: a list of {name, set: {dotted.key: value}}
run one after the other (column `setting`); a setting may set its own
`seeds`. Optional config key `errors`: correlated detector errors
(bflood.errors). Optional config key `sweep_threshold`: a list of alarm thresholds tau; rows
get a `tau` column ("none" = the normal output), see bflood.sim.run_once.

Writes results/<name>/runs.csv, summary.csv and summary.md.
"""
import argparse
import copy
import csv
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from itertools import product
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from bflood.metrics import METRICS, RUN_METRICS  # noqa: E402
from bflood.sim import run_once  # noqa: E402

ALL_METRICS = METRICS + RUN_METRICS


def set_dotted(cfg: dict, key: str, value):
    d = cfg
    parts = key.split(".")
    for p in parts[:-1]:
        d = d.setdefault(p, {})
    d[parts[-1]] = value


def expand(cfg: dict):
    """Yield (sweep_point_dict, cfg_copy) for every setting x sweep point.
    Optional `settings`: a list of {name, set: {dotted.key: value}}, each a
    named group of overrides (column `setting`), e.g. a test condition that
    changes several keys at once, or its own `seeds`."""
    sweeps = cfg.get("sweep", [])
    keys = [s["key"] for s in sweeps]
    for st in cfg.get("settings") or [None]:
        for vals in product(*[s["values"] for s in sweeps]):
            c = copy.deepcopy(cfg)
            point = {}
            if st is not None:
                for k, v in st.get("set", {}).items():
                    set_dotted(c, k, copy.deepcopy(v))
                point["setting"] = st["name"]
            for k, v in zip(keys, vals):
                set_dotted(c, k, v)
            point.update(zip(keys, vals))
            yield point, c


def _job(args):
    point, cfg, nb, adv, seed = args
    rows = run_once(cfg, nb, adv, seed)
    for r in rows:
        r.update(point)
    return rows


def summarise(rows, sweep_keys):
    """Mean and 95% CI half-width over seeds per (sweep..., adversary, n_byz, method)."""
    groups = {}
    for r in rows:
        key = tuple(r[k] for k in sweep_keys) + (r["adversary"], r["n_byz"], r["method"])
        groups.setdefault(key, []).append(r)
    out = []
    for key, rs in sorted(groups.items(), key=lambda kv: tuple(map(str, kv[0]))):
        s = dict(zip(sweep_keys, key[:len(sweep_keys)]))
        s.update({"adversary": key[-3], "n_byz": key[-2], "method": key[-1],
                  "byz_count": int(np.median([r["byz_count"] for r in rs])),
                  "f_local": int(np.median([r["f_local"] for r in rs])),
                  "n": int(np.median([r["n"] for r in rs])),
                  "messages_per_round": float(np.mean([r["messages_per_round"] for r in rs])),
                  "runs": len(rs)})
        for k in ALL_METRICS:
            v = np.array([r[k] for r in rs], dtype=float)
            v = v[np.isfinite(v)]
            s[k] = float(v.mean()) if v.size else float("nan")
            s[k + "_ci"] = float(1.96 * v.std(ddof=1) / np.sqrt(v.size)) if v.size > 1 else 0.0
        out.append(s)
    return out


def write_md(path, cfg, summ, sweep_keys, elapsed):
    show = ["acc", "acc_flooded", "false_alarm", "range_viol", "delay", "miss", "shallow_recall"]
    if cfg.get("timing"):
        show.append("decide_us")
    for extra in (("acc_pond", "alarm_pond"), ("search_exact",)):
        if any(np.isfinite(s[extra[0]]) for s in summ):
            show += list(extra)
    lines = [f"# {cfg['name']}", "", cfg.get("description", "").strip(), "",
             f"Detector: {cfg.get('detector')}  ",
             f"Seeds: {cfg['seeds']}, rounds: {cfg.get('rounds', 20)}, "
             f"runtime: {elapsed / 60:.1f} min", "",
             "| " + " | ".join(sweep_keys + ["adv", "n_byz", "f", "method"] + show) + " |",
             "|" + "---|" * (len(sweep_keys) + 4 + len(show))]
    for s in summ:
        cells = [str(s[k]) for k in sweep_keys] + [s["adversary"], str(s["n_byz"]),
                                                   str(s["f_local"]), s["method"]]
        cells += [f"{s[k]:.3f}" if np.isfinite(s[k]) else "-" for k in show]
        lines.append("| " + " | ".join(cells) + " |")
    Path(path).write_text("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--placeholder", action="store_true")
    ap.add_argument("--bank", default=None)
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    cfg = yaml.safe_load(Path(a.config).read_text())
    if a.placeholder:
        cfg["detector"] = {"type": "placeholder", "rho": 0.87}
        cfg["name"] += "_placeholder"
    elif a.bank:
        cfg.setdefault("detector", {"type": "belief_bank"})
        cfg["detector"] = {**cfg["detector"], "type": "belief_bank", "bank": a.bank}
        cfg["detector"].pop("path", None)
    if a.quick:
        cfg["seeds"] = 2
        cfg["name"] += "_quick"
    sweep_keys = (["setting"] if cfg.get("settings") else []) + [s["key"] for s in cfg.get("sweep", [])]
    if "cases" in cfg:
        cases = [(c["adversary"], c["n_byz"]) for c in cfg["cases"]]
    else:
        cases = list(product(cfg["adversaries"], cfg["n_byz"]))
    jobs = [(pt, c, nb, adv, s)
            for pt, c in expand(cfg)
            for (adv, nb), s in product(cases, range(min(c["seeds"], 2) if a.quick else c["seeds"]))]
    print(f"{cfg['name']}: {len(jobs)} runs x {len(cfg['methods'])} methods")
    t0 = time.time()
    with ProcessPoolExecutor(a.workers) as ex:
        rows = [r for rs in ex.map(_job, jobs, chunksize=4) for r in rs]
    elapsed = time.time() - t0
    outdir = ROOT / "results" / cfg["name"]
    outdir.mkdir(parents=True, exist_ok=True)
    with open(outdir / "runs.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    group_keys = sweep_keys + (["tau"] if "tau" in rows[0] else [])
    summ = summarise(rows, group_keys)
    with open(outdir / "summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summ[0]))
        w.writeheader(); w.writerows(summ)
    write_md(outdir / "summary.md", cfg, summ, group_keys, elapsed)
    print(f"{len(rows)} rows in {elapsed / 60:.1f} min -> {outdir}")


if __name__ == "__main__":
    main()
