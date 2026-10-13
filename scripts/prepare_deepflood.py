"""Build image-level labels for DeepFlood (Zenodo 10.5281/zenodo.7825989).

Input : data/Flood_Dataset.zip (as downloaded; research-use access granted by
        P. Chaudhary). Images are NOT redistributed; only IDs and labels.
Output: data/processed/deepflood_labels.csv

Per-object `level_id` uses the 11 body-proportion levels of Chaudhary et al.
(ISPRS Annals 2019, Table 1): 0 none, 1 <=1 cm, 2 1-10, 3 10-21, 4 21-43,
5 43-64, 6 64-85, 7 85-106, 8 106-128, 9 128-149, 10 149-170 cm.
Categories: 1 person, 2 bicycle, 3 car, 6 bus, 92 house, 91 flood (water
region, carries no level and is ignored).

Image label = median level of the *submerged* objects (level > 0), 0 if none.
The authors' trimmed mean over all objects is not used: in ~60% of images
some objects are dry (pedestrians on pavements, houses set back) while others
are submerged, and averaging them labels flooded scenes as dry.

Mapping to the paper's five levels (Section II, Table II):
  DF 0-1 -> 0 (<= 1 cm), 2 -> 1 (1-10 cm), 3 -> 2 (10-21 cm),
  4 -> 3 (21-43 cm), >= 5 -> 4 (> 43 cm).

Columns written: image_id, file, level5, dup_group, split (cross_dup is added
by scripts/crosscheck_datasets.py). No other value of the DeepFlood
annotation is written; detector.manifest reads the one it needs (the
`ambiguous` flag) from the zip.
"""
import csv
import io
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
ZIP = ROOT / "data" / "Flood_Dataset.zip"
OUT = ROOT / "data" / "processed" / "deepflood_labels.csv"
DF_TO_5 = {0: 0, 1: 0, 2: 1, 3: 2, 4: 3}


def to5(level: int) -> int:
    return DF_TO_5.get(int(level), 4)


def ahash(img: Image.Image, n: int = 16) -> np.ndarray:
    g = np.asarray(img.convert("L").resize((n, n), Image.BILINEAR), dtype=float)
    return (g > g.mean()).ravel()


def main(zip_path=ZIP, out=OUT, seed=0, test_frac=0.2):
    z = zipfile.ZipFile(zip_path)
    ann = json.loads(z.read("Flood_Dataset/flood_dict_new.json"))
    rows, hashes = [], {}
    for img_id, objs in sorted(ann.items(), key=lambda kv: int(kv[0].split("_")[1])):
        lv = np.array([o["level_id"] for o in objs if o["category_id"] != 91], dtype=int)
        sub = lv[lv > 0]
        df = int(round(float(np.median(sub)))) if sub.size else 0
        rows.append({"image_id": img_id, "file": f"Flood_Dataset/images_flood/{img_id}.jpg",
                     "level5": to5(df)})
        hashes[img_id] = ahash(Image.open(io.BytesIO(z.read(rows[-1]["file"]))))
    # near-duplicate groups (Hamming distance <= 10 of 256 bits)
    ids = list(hashes)
    H = np.array([hashes[i] for i in ids])
    dist = (H[:, None, :] != H[None, :, :]).sum(-1)
    group = list(range(len(ids)))

    def find(a):
        while group[a] != a:
            group[a] = group[group[a]]
            a = group[a]
        return a

    for a, b in zip(*np.nonzero(np.triu(dist <= 10, 1))):
        group[find(a)] = find(b)
    gid = {ids[i]: find(i) for i in range(len(ids))}
    # stratified split by level, whole duplicate groups kept together
    rng = np.random.default_rng(seed)
    by_group = {}
    for r in rows:
        by_group.setdefault(gid[r["image_id"]], []).append(r)
    test_groups = set()
    for lvl in range(5):
        gs = [g for g, rs in by_group.items() if rs[0]["level5"] == lvl]
        rng.shuffle(gs)
        test_groups.update(gs[: int(round(test_frac * len(gs)))])
    for r in rows:
        r["dup_group"] = gid[r["image_id"]]
        r["split"] = "test" if gid[r["image_id"]] in test_groups else "train"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    sizes = np.bincount([len(v) for v in by_group.values()])
    print(f"{len(rows)} images -> {out}")
    print("level5 counts:", np.bincount([r["level5"] for r in rows], minlength=5).tolist())
    print("near-duplicate groups of size >= 2:", int(sum(sizes[2:])),
          "covering", int(sum(k * c for k, c in enumerate(sizes) if k >= 2)), "images")
    for s in ("train", "test"):
        print(s, np.bincount([r["level5"] for r in rows if r["split"] == s], minlength=5).tolist())


if __name__ == "__main__":
    main(*(Path(a) for a in sys.argv[1:3]))
