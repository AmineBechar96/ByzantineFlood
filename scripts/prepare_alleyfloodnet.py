"""Labels and a cleaned split for AlleyFloodNet (Kaggle, CC BY 3.0).

Input : data/AlleyFloodnet.zip (flood_train_/ and flood_test_/, each with
        flooded/ and non_flooded/ subfolders; 1,110 images, ~188 px short side)
Output: data/processed/alleyfloodnet_labels.csv

Roles in this project: non_flooded = level 0 (many are wet streets at night,
the hardest false-alarm case); flooded = "level >= 1" with unknown depth, so
it is used only for dry-vs-flooded checks, never for 5-level metrics.

Cleaning (3 October 2026 audit): 37 near-duplicate pairs (difference hash,
Hamming <= 20 of 256), 4 of them across the official train/test split, and one
test image (img_india_20151202_title-759x500.jpg) that is the same photo as
DeepFlood Flood_118 (a training image). `split` is the official split with:
duplicates of training images and repeats within test marked `drop`, the
DeepFlood duplicate marked `drop`, and 10% of training duplicate groups moved
to `val` (stratified by class).

Columns written: file, class (dry | flooded), split, dup_group, cross_dup.
"""
import csv
import io
import sys
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
CROSS_DEEPFLOOD = {"img_india_20151202_title-759x500.jpg": "Flood_118"}


def dhash(im, n=16):
    g = np.asarray(im.convert("L").resize((n + 1, n), Image.BILINEAR), dtype=float)
    return (g[:, 1:] > g[:, :-1]).ravel()


def main(zip_path=ROOT / "data/AlleyFloodnet.zip",
         out=ROOT / "data/processed/alleyfloodnet_labels.csv", seed=0):
    z = zipfile.ZipFile(zip_path)
    files = sorted(n for n in z.namelist() if not n.endswith("/"))
    rows = []
    for f in files:
        im = Image.open(io.BytesIO(z.read(f)))
        rows.append({"file": f, "official_split": "test" if "flood_test_" in f else "train",
                     "class": "flooded" if "/flooded/" in f else "dry", "_h": dhash(im)})
    H = np.array([r["_h"] for r in rows])
    D = (H[:, None, :] != H[None, :, :]).sum(-1)
    parent = list(range(len(rows)))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for a, b in zip(*np.nonzero(np.triu(D <= 20, 1))):
        parent[find(a)] = find(b)
    groups = defaultdict(list)
    for i, r in enumerate(rows):
        r["dup_group"] = find(i)
        groups[r["dup_group"]].append(i)
    # official split, cleaned
    for g, idx in groups.items():
        has_train = any(rows[i]["official_split"] == "train" for i in idx)
        kept_test = False
        for i in sorted(idx):
            r = rows[i]
            if r["official_split"] == "train":
                r["split"] = "train"
            elif has_train or kept_test:
                r["split"] = "drop"
            else:
                r["split"] = "test"
                kept_test = True
    for r in rows:
        name = r["file"].split("/")[-1]
        r["cross_dup"] = CROSS_DEEPFLOOD.get(name, "")
        if r["cross_dup"]:
            r["split"] = "drop"
    # validation: 10% of training groups per class
    rng = np.random.default_rng(seed)
    for cls in ("dry", "flooded"):
        gs = sorted({r["dup_group"] for r in rows if r["split"] == "train" and r["class"] == cls})
        rng.shuffle(gs)
        val = set(gs[: int(round(0.1 * len(gs)))])
        for r in rows:
            if r["split"] == "train" and r["class"] == cls and r["dup_group"] in val:
                r["split"] = "val"
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fields = ["file", "class", "split", "dup_group", "cross_dup"]
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in fields})
    cnt = defaultdict(int)
    for r in rows:
        cnt[(r["split"], r["class"])] += 1
    print(f"{len(rows)} images -> {out}")
    for s in ("train", "val", "test", "drop"):
        print(f"{s:5s} dry {cnt[(s, 'dry')]:4d}  flooded {cnt[(s, 'flooded')]:4d}")


if __name__ == "__main__":
    main(*sys.argv[1:3])
