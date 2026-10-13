"""Labels and a leak-free split for STURM-FloodDepth Flooded Cars
(Zenodo 10.5281/zenodo.14833532, CC BY 4.0).

Input : data/flooded_cars.zip, data/{train,val,test}.txt (official split)
Output: data/processed/sturm_labels.csv

Levels (STURM-FloodDepth paper, relative to the vehicle):
  0 not flooded; 1 below half the tyre / below chassis;
  2 half to whole tyre (~0.3-0.6 m); 3 tyre to side window (~0.6-1.0 m);
  4 above the side window (~1.0-1.5 m).
`level3` collapses to the scale shared with DeepFlood:
  0 none | 1 up to ~60 cm (STURM 1-2, DeepFlood <=64 cm) | 2 deeper.

The official split was drawn per crop. Crops come several per source photo,
and 221 of 340 official test crops share a photo with training crops, so the
official split leaks. `group_split` keeps every crop of a photo in one subset
(80/10/10, stratified by the photo's highest level). Use group_split.

Columns written: file, photo, level3, group_split (cross_dup is added by
scripts/crosscheck_datasets.py). The STURM level and the official split are
used here but not written; detector.manifest reads the official lists itself.
"""
import csv
import re
import sys
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PAT = re.compile(r"(car\d?-\d+)-Level(\d)_(\d+)\.jpg")
TO3 = {0: 0, 1: 1, 2: 1, 3: 2, 4: 2}


def main(data_dir=ROOT / "data", out=ROOT / "data/processed/sturm_labels.csv", seed=0):
    data_dir = Path(data_dir)
    z = zipfile.ZipFile(data_dir / "flooded_cars.zip")
    paths = {p.split("/")[-1]: p for p in z.namelist() if p.endswith(".jpg")}
    official = {}
    for s in ("train", "val", "test"):
        for line in open(data_dir / f"{s}.txt"):
            f = line.strip().replace("\\", "/").split("/")[-1]
            if f:
                official[f] = s
    rows = []
    for f, p in sorted(paths.items()):
        m = PAT.fullmatch(f)
        if not m:
            raise ValueError(f"unexpected file name {f}")
        lvl = int(m.group(2))
        if f"Level{lvl}" not in p:
            raise ValueError(f"level folder mismatch for {p}")
        rows.append({"file": p, "image": f, "photo": m.group(1), "sturm_level": lvl,
                     "level3": TO3[lvl], "official_split": official.get(f, "unlisted")})
    photo_max = defaultdict(int)
    for r in rows:
        photo_max[r["photo"]] = max(photo_max[r["photo"]], r["sturm_level"])
    rng = np.random.default_rng(seed)
    assign = {}
    for lvl in range(5):
        ph = sorted(p for p, m in photo_max.items() if m == lvl)
        rng.shuffle(ph)
        n = len(ph)
        n_test, n_val = int(round(0.1 * n)), int(round(0.1 * n))
        for i, p in enumerate(ph):
            assign[p] = "test" if i < n_test else "val" if i < n_test + n_val else "train"
    for r in rows:
        r["group_split"] = assign[r["photo"]]
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["file", "photo", "level3", "group_split"], extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    # report
    by = defaultdict(set)
    for r in rows:
        if r["official_split"] != "unlisted":
            by[r["photo"]].add(r["official_split"])
    leak = sum(1 for r in rows if r["official_split"] == "test" and "train" in by[r["photo"]])
    print(f"{len(rows)} crops from {len(photo_max)} photos -> {out}")
    print(f"official test crops sharing a photo with train: {leak}/"
          f"{sum(r['official_split'] == 'test' for r in rows)}")
    for s in ("train", "val", "test"):
        c = Counter(r["sturm_level"] for r in rows if r["group_split"] == s)
        print(f"group_split {s:5s}", [c[i] for i in range(5)])
    g = defaultdict(set)
    for r in rows:
        g[r["photo"]].add(r["group_split"])
    assert all(len(v) == 1 for v in g.values())


if __name__ == "__main__":
    main(*sys.argv[1:3])
