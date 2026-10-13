"""Final detector report for the four runs trained with the human-checked Grok labels.

Reads results/<run>/ as downloaded from the bflood-results volume and writes
results/detector_final/ (CSV + SUMMARY.md). Each fold ensemble is rebuilt from
fold*_preds.npz exactly as detector.train.run does it (labels from fold 0,
probabilities averaged over folds) and checked against ensemble/metrics.json.
Added on top of report.py: ECE with a CI, flooded AUC and false-alarm rate on
DeepFlood, STURM false-alarm rate, the deepflood_test confusion matrix, a folds
0-2 ensemble of detector_base (like-for-like with the 3-fold ablations),
per-fold epochs and temperatures, belief-bank sizes, wall times and Modal cost.
CIs: report.py's own ordinal.bootstrap_ci (2000 resamples, seed 0, 95 % percentile).
"""
import csv
import json
import os
import re
import subprocess
import sys
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from detector import ordinal as O  # noqa: E402
from detector import report as R  # noqa: E402

RUNS = {"detector_base": ("ap-H8HY6Kx6VNyBQDkg0qtrfK", 5),
        "detector_no_fisheye": ("ap-yk0m0TkYLlSvU82alUm7Wj", 3),
        "detector_no_nuisance": ("ap-05r4RZ7H1rIS3dYoaQVw5y", 3),
        "detector_sim2real": ("ap-IBSqi6QluSCtNGbZsflhAf", 5)}
OUT = ROOT / "results" / "detector_final"
ENV = dict(os.environ, PYTHONIOENCODING="utf-8", MSYS_NO_PATHCONV="1")


def load_fold(run, f):
    z = np.load(ROOT / "results" / run / f"fold{f}_preds.npz")
    p = {}
    for k in z.files:
        ts, fld = k.split("__", 1)
        p.setdefault(ts, {})[fld] = z[k]
    return p


def ensemble(preds):
    for p in preds[1:]:
        for k in preds[0]:
            for fld in ("lo", "hi", "level3", "member"):
                assert np.array_equal(p[k][fld], preds[0][k][fld]), (k, fld)
    return {k: dict(preds[0][k], probs=np.mean([p[k]["probs"] for p in preds], axis=0)) for k in preds[0]}


def check_against_json(ens, run):
    ref = json.loads((ROOT / "results" / run / "ensemble" / "metrics.json").read_text())
    mine = R.evaluate(ens)
    for k, v in ref.items():
        a = np.array(list(v.values()) if isinstance(v, dict) else v, float)
        b = np.array(list(mine[k].values()) if isinstance(mine[k], dict) else mine[k], float)
        assert np.allclose(a, b, equal_nan=True), (run, k, a, b)
    return len(ref)


def bin_acc(p, flooded):
    return float(((O.ordinal_level(p) >= 1).astype(int) == flooded).mean())


def metrics(ens):
    rows = []

    def add(ts, name, fn, *a):
        est, lo, hi = O.bootstrap_ci(fn, *a)
        rows.append({"test_set": ts, "metric": name, "value": est, "ci_lo": lo, "ci_hi": hi, "n": len(a[0])})

    for ts in ("deepflood_test", "deepflood_test_fisheye"):
        d = ens[ts]
        m = (d["lo"] == d["hi"]) & (d["lo"] >= 0)
        p, y = d["probs"][m], d["lo"][m].astype(int)
        add(ts, "acc", O.acc, p, y)
        add(ts, "within1", O.within1, p, y)
        add(ts, "mae", O.mae, p, y)
        add(ts, "ece", O.ece, p, y)
        add(ts, "flooded_auc", O.binary_auc, 1 - p[:, 0], (y >= 1).astype(int))
        add(ts, "false_alarm", O.flooded_rate, p[y == 0])
        add(ts, "flood_recall", O.flooded_rate, p[y >= 1])
    d = ens["sturm_test"]
    fl = (d["level3"] >= 1).astype(int)
    add("sturm_test", "acc_dry_vs_flooded", bin_acc, d["probs"], fl)
    add("sturm_test", "flooded_auc", O.binary_auc, 1 - d["probs"][:, 0], fl)
    add("sturm_test", "false_alarm", O.flooded_rate, d["probs"][fl == 0])
    add("sturm_test", "flood_recall", O.flooded_rate, d["probs"][fl == 1])
    add("sturm_test", "kendall_tau_flooded", O.kendall_tau, d["probs"][fl == 1], d["level3"][fl == 1])
    d = ens["alley_test"]
    fl = (d["hi"] != 0).astype(int)
    add("alley_test", "acc_dry_vs_flooded", bin_acc, d["probs"], fl)
    add("alley_test", "flooded_auc", O.binary_auc, 1 - d["probs"][:, 0], fl)
    add("alley_test", "false_alarm", O.flooded_rate, d["probs"][fl == 0])
    add("alley_test", "flood_recall", O.flooded_rate, d["probs"][fl == 1])
    add("fe8k_test", "false_alarm", O.flooded_rate, ens["fe8k_test"]["probs"])
    return rows


def confusion(ens, ts="deepflood_test"):
    d = ens[ts]
    m = (d["lo"] == d["hi"]) & (d["lo"] >= 0)
    y, yh = d["lo"][m].astype(int), O.ordinal_level(d["probs"][m])
    cm = np.zeros((O.N_LEVELS, O.N_LEVELS), int)
    np.add.at(cm, (y, yh), 1)
    return cm


def auc_flooded(p, lab):
    return O.binary_auc(1 - p[:, 0], lab)


PAIRS = [(("detector_no_fisheye", "3 folds"), ("detector_base", "folds 0-2")),
         (("detector_no_nuisance", "3 folds"), ("detector_base", "folds 0-2")),
         (("detector_sim2real", "5 folds"), ("detector_base", "5 folds"))]


def paired(ens_a, ens_b):
    """Paired bootstrap of A - B on the same test images (same resampled rows for both)."""
    rows = []

    def add(ts, name, fn, *a):
        est, lo, hi = O.bootstrap_ci(fn, *a)
        rows.append({"test_set": ts, "metric": name, "diff": est, "ci_lo": lo, "ci_hi": hi, "n": len(a[0])})

    for k in ens_a:
        assert np.array_equal(ens_a[k]["member"], ens_b[k]["member"]), k
    for ts in ("deepflood_test", "deepflood_test_fisheye"):
        a, b = ens_a[ts], ens_b[ts]
        m = (a["lo"] == a["hi"]) & (a["lo"] >= 0)
        y = a["lo"][m].astype(int)
        for name, fn in (("acc", O.acc), ("mae", O.mae)):
            add(ts, name, lambda pa, pb, yy, fn=fn: fn(pa, yy) - fn(pb, yy), a["probs"][m], b["probs"][m], y)
    a, b = ens_a["sturm_test"], ens_b["sturm_test"]
    fl = (a["level3"] >= 1).astype(int)
    add("sturm_test", "flooded_auc", lambda pa, pb, f: auc_flooded(pa, f) - auc_flooded(pb, f), a["probs"], b["probs"], fl)
    a, b = ens_a["alley_test"], ens_b["alley_test"]
    fl = (a["hi"] != 0).astype(int)
    add("alley_test", "flooded_auc", lambda pa, pb, f: auc_flooded(pa, f) - auc_flooded(pb, f), a["probs"], b["probs"], fl)
    add("alley_test", "false_alarm", lambda pa, pb: O.flooded_rate(pa) - O.flooded_rate(pb),
        a["probs"][fl == 0], b["probs"][fl == 0])
    a, b = ens_a["fe8k_test"], ens_b["fe8k_test"]
    add("fe8k_test", "false_alarm", lambda pa, pb: O.flooded_rate(pa) - O.flooded_rate(pb), a["probs"], b["probs"])
    return rows


def parse_log(run):
    folds, cur = {}, None
    for line in open(ROOT / "results" / run / "train.log", encoding="utf-8"):
        if m := re.match(r"fold (\d+): train (\d+), val (\d+)", line):
            cur = int(m[1])
            folds[cur] = {"train": int(m[2]), "val": int(m[3]), "epochs": [], "early_stop": False}
        elif m := re.match(r"\s+epoch (\d+)/(\d+) train (\S+) val (\S+) \((\d+)s\)", line):
            folds[cur]["epochs"].append((int(m[1]), float(m[3]), float(m[4]), int(m[5])))
        elif "early stop" in line:
            folds[cur]["early_stop"] = True
        elif m := re.match(r"fold (\d+): best val (\S+), temperature (\S+)", line):
            folds[cur]["best_val"], folds[cur]["T"] = float(m[2]), float(m[3])
    for f in folds.values():  # best epoch with train.py's rule (improvement > 1e-4)
        best, f["best_epoch"] = float("inf"), None
        for ep, _, v, _ in f["epochs"]:
            if v < best - 1e-4:
                best, f["best_epoch"] = v, ep
    return folds


def modal_json(*args):
    r = subprocess.run([sys.executable, "-m", "modal", *args, "--json"], capture_output=True, text=True,
                       env=ENV, check=True, encoding="utf-8")
    return json.loads(r.stdout)


def fmt(r):
    return f"{r['value']:.3f} [{r['ci_lo']:.3f}, {r['ci_hi']:.3f}]"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    met_rows, cm_rows, fold_rows, bank_rows, cost_rows, perfold_rows = [], [], [], [], [], []
    ens_all = {}
    for run, (app, nf) in RUNS.items():
        files = sorted((ROOT / "results" / run).glob("fold*_preds.npz"))
        assert len(files) == nf, (run, len(files))
        preds = [load_fold(run, f) for f in range(nf)]
        ens = ensemble(preds)
        nchk = check_against_json(ens, run)
        print(f"{run}: {nf} folds, ensemble rebuilt; {nchk} metrics match ensemble/metrics.json")
        variants = {f"{nf} folds": ens}
        if run == "detector_base":
            variants["folds 0-2"] = ensemble(preds[:3])
        for vname, e in variants.items():
            ens_all[(run, vname)] = e
            for r in metrics(e):
                met_rows.append({"run": run, "ensemble": vname, **r})
            cm = confusion(e)
            for c in range(O.N_LEVELS):
                cm_rows.append({"run": run, "ensemble": vname, "true_level": c,
                                **{f"pred_{k}": int(cm[c, k]) for k in range(O.N_LEVELS)},
                                "n": int(cm[c].sum()), "recall": cm[c, c] / max(1, cm[c].sum())})
        for f in range(nf):
            fm = json.loads((ROOT / "results" / run / f"fold{f}" / "metrics.json").read_text())
            perfold_rows.append({"run": run, "fold": f, **{k: v[0] if isinstance(v, list) else v
                                                         for k, v in fm.items() if not isinstance(v, dict)}})
        for f, info in sorted(parse_log(run).items()):
            fold_rows.append({"run": run, "fold": f, "train": info["train"], "val": info["val"],
                              "epochs_run": len(info["epochs"]), "best_epoch": info["best_epoch"],
                              "early_stop": int(info["early_stop"]), "best_val_loss": info.get("best_val"),
                              "temperature": info.get("T"), "epoch_seconds": sum(e[3] for e in info["epochs"])})
        z = np.load(ROOT / "results" / run / "ensemble" / "belief_bank.npz")
        lv, src = np.bincount(z["labels"], minlength=5), Counter(z["source"].tolist())
        yh = O.ordinal_level(z["probs"])
        bank_rows.append({"run": run, "n": len(z["labels"]), **{f"level_{c}": int(lv[c]) for c in range(5)},
                          **{f"from_{k}": v for k, v in sorted(src.items())},
                          "acc": float((yh == z["labels"]).mean()),
                          "false_alarm_level0": float((yh[z["labels"] == 0] >= 1).mean()),
                          "miss_level1plus": float((yh[z["labels"] >= 1] == 0).mean())})
    pair_rows = []
    for a, b in PAIRS:
        for r in paired(ens_all[a], ens_all[b]):
            pair_rows.append({"a": f"{a[0]} ({a[1]})", "b": f"{b[0]} ({b[1]})", **r})
    # wall time and cost ------------------------------------------------------
    apps = {a["app_id"]: a for a in modal_json("app", "list")}
    bill = modal_json("billing", "report", "--for", "today", "--resolution", "h", "--show-resources", "--tz", "local")
    now = datetime.now().astimezone()
    for run, (app, nf) in RUNS.items():
        a = apps[app]
        t0 = datetime.fromisoformat(a["created_at"])
        t1 = datetime.fromisoformat(a["stopped_at"]) if a["stopped_at"] else None
        res = Counter()
        for b in bill:
            if b["object_id"] == app:
                res[b["resource"]] += float(b["cost"])
        complete = t1 is not None and now >= t1.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
        cost_rows.append({"run": run, "app_id": app, "state": a["state"], "created": a["created_at"],
                          "stopped": a["stopped_at"], "wall_min": round((t1 - t0).total_seconds() / 60, 1) if t1 else None,
                          **{f"cost_{k}": round(v, 4) for k, v in sorted(res.items())},
                          "cost_total": round(sum(res.values()), 4), "billing_complete": int(complete)})
    # write CSVs ---------------------------------------------------------------
    for name, rows in (("metrics.csv", met_rows), ("confusion_deepflood_test.csv", cm_rows),
                       ("folds.csv", fold_rows), ("per_fold_metrics.csv", perfold_rows),
                       ("belief_banks.csv", bank_rows), ("runs_cost.csv", cost_rows),
                       ("paired_differences.csv", pair_rows)):
        keys = list(dict.fromkeys(k for r in rows for k in r))
        with open(OUT / name, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)
    with open(Path(__file__).with_name("tables.json"), "w") as f:
        json.dump({"met": met_rows, "cm": cm_rows, "folds": fold_rows, "perfold": perfold_rows,
                   "bank": bank_rows, "cost": cost_rows, "paired": pair_rows}, f, indent=1, default=float)
    write_summary(met_rows, cm_rows, fold_rows, bank_rows, cost_rows, pair_rows, perfold_rows)
    print(f"wrote {OUT}")
    for r in cost_rows:
        print(r)


COLS = {"deepflood_test": ["acc", "within1", "mae", "ece", "flooded_auc", "false_alarm"],
        "deepflood_test_fisheye": ["acc", "within1", "mae", "ece", "flooded_auc", "false_alarm"],
        "sturm_test": ["acc_dry_vs_flooded", "flooded_auc", "false_alarm", "flood_recall", "kendall_tau_flooded"],
        "alley_test": ["acc_dry_vs_flooded", "flooded_auc", "false_alarm", "flood_recall"],
        "fe8k_test": ["false_alarm"]}


def write_summary(met_rows, cm_rows, fold_rows, bank_rows, cost_rows, pair_rows, perfold_rows_g):
    L = ["# Final detector runs (human-checked Grok labels), 2026-10-05", "",
         "Generated from results/<run>/ (fold predictions downloaded from Modal). Ensemble = mean of",
         "the fold beliefs; level = ordinal median. 95 % bootstrap CIs (2000 resamples, seed 0).",
         "`folds 0-2` = detector_base restricted to the three folds the ablations trained on.", ""]
    L += ["## Runs", "", "| run | folds | train / val per fold | epochs per fold (best) | wall time | cost |",
          "|---|---|---|---|---|---|"]
    for c in cost_rows:
        fr = [r for r in fold_rows if r["run"] == c["run"]]
        tv = f"{min(r['train'] for r in fr)}-{max(r['train'] for r in fr)} / {min(r['val'] for r in fr)}-{max(r['val'] for r in fr)}"
        ep = ", ".join(f"{r['epochs_run']} ({r['best_epoch']})" for r in fr)
        cost = f"${c['cost_total']:.2f}" + ("" if c["billing_complete"] else " (partial)")
        L.append(f"| {c['run']} | {len(fr)} | {tv} | {ep} | {c['wall_min']} min | {cost} |")
    for ts, cols in COLS.items():
        rs = [r for r in met_rows if r["test_set"] == ts]
        n = {r["metric"]: r["n"] for r in rs}
        L += ["", f"## {ts}", "", "| run | ensemble | " + " | ".join(f"{c} (n={n[c]})" for c in cols) + " |",
              "|---|---|" + "---|" * len(cols)]
        for key in dict.fromkeys((r["run"], r["ensemble"]) for r in rs):
            v = {r["metric"]: r for r in rs if (r["run"], r["ensemble"]) == key}
            L.append(f"| {key[0]} | {key[1]} | " + " | ".join(fmt(v[c]) for c in cols) + " |")
    L += ["", "## deepflood_test confusion matrix, detector_base (5 folds)", "",
          "| true \\ predicted | 0 | 1 | 2 | 3 | 4 | n | recall |", "|---|---|---|---|---|---|---|---|"]
    for r in cm_rows:
        if r["run"] == "detector_base" and r["ensemble"] == "5 folds":
            L.append(f"| {r['true_level']} | " + " | ".join(str(r[f"pred_{k}"]) for k in range(5))
                     + f" | {r['n']} | {r['recall']:.2f} |")
    L += ["", "## Paired differences (A - B, same test images, paired bootstrap)", "",
          "| A - B | test set | metric | difference [95 % CI] |", "|---|---|---|---|"]
    for r in pair_rows:
        L.append(f"| {r['a']} - {r['b']} | {r['test_set']} | {r['metric']} | "
                 f"{r['diff']:+.3f} [{r['ci_lo']:+.3f}, {r['ci_hi']:+.3f}] |")
    L += ["", "## Belief banks (ensemble/belief_bank.npz)", "",
          "| run | beliefs | per level 0-4 | sources | exact acc | level-0 predicted flooded | flooded predicted 0 |",
          "|---|---|---|---|---|---|---|"]
    for b in bank_rows:
        src = ", ".join(f"{k[5:]} {v}" for k, v in b.items() if k.startswith("from_"))
        L.append(f"| {b['run']} | {b['n']} | {[b[f'level_{c}'] for c in range(5)]} | {src} | {b['acc']:.3f} | "
                 f"{b['false_alarm_level0']:.3f} | {b['miss_level1plus']:.3f} |")
    fe = [r["fe8k_false_alarm"] for r in perfold_rows_g if r["run"] == "detector_base"]
    st = [r["sturm_dry_vs_flooded_auc"] for r in perfold_rows_g if r["run"] == "detector_base"]
    t_s2r = [r["temperature"] for r in fold_rows if r["run"] == "detector_sim2real"]
    st_base = [r for r in met_rows if r["run"] == "detector_base" and r["ensemble"] == "5 folds"
               and r["test_set"] == "sturm_test" and r["metric"] == "flood_recall"][0]
    L += ["", "## Notes", "",
          "- Paired CIs cover test-image sampling only, not training randomness. Across the five base folds the",
          f"  FishEye8K false-alarm rate ranges {min(fe):.3f}-{max(fe):.3f} and STURM AUC {min(st):.3f}-{max(st):.3f},",
          "  wider than every ablation difference; 16 ablation comparisons, no multiplicity correction. Do not",
          "  claim an ablation effect from these runs.",
          f"- detector_sim2real: validation folds contain only dry FishEye8K frames (all level 0), so early",
          f"  stopping and temperature scaling see dry images only; temperature = {sorted(set(t_s2r))} (clamp 0.2)",
          "  in every fold. Its ECE and belief sharpness are not meaningful. Predicted levels of a single fold do",
          "  not depend on the temperature, so the accuracy/AUC gap is not a calibration artefact, but the epoch",
          "  choice is biased towards dry predictions.",
          "- DeepFlood false-alarm rate and flooded AUC rest on 11 dry test images.",
          "- STURM dry-vs-flooded accuracy should be read against the all-flooded baseline 249/337 = 0.739;",
          f"  flood recall is {st_base['value']:.3f}, so the dry crops are mostly called flooded. AUC and Kendall",
          "  tau are the informative transfer numbers."]
    L += ["", "Files: metrics.csv (all metrics with CIs), confusion_deepflood_test.csv (every run),",
          "paired_differences.csv, folds.csv (epochs, best epoch, temperature), per_fold_metrics.csv,",
          "belief_banks.csv, runs_cost.csv (Modal app times and billed cost per resource)."]
    (OUT / "SUMMARY.md").write_text("\n".join(L) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
