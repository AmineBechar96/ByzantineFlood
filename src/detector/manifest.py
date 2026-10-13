"""Build data/processed/manifest.csv: every image the detector may use, with an
interval label, a role and (for test images) the test set it belongs to.

Label = [level_lo, level_hi] on the 5-level scale of paper Table II:
  exact level c  -> [c, c]      (DeepFlood, all dry images, labelled Grok)
  flooded, depth unknown -> [1, 4]   (AlleyFloodNet flooded)
STURM is used only as a 3-level transfer test (`level3`), never for training.

Roles: train | val | test | unused. Test sets:
  deepflood_test  5 levels, real scenes (primary)
  sturm_test      3 levels (none | <=~60 cm | deeper), car crops (transfer)
  alley_test      dry vs flooded, incl. wet streets at night
  fe8k_test       dry fisheye, FishEye8K test cameras 1, 2, 4, 7
Groups: duplicate group for web photos; the CAMERA for FishEye8K and Grok
images (a fixed camera shows the same street in every frame), so
cross-validation never splits one camera across train and validation.
Rules enforced (see check()): synthetic images never in val/test; nothing
marked as a duplicate across datasets is in a test set; no FishEye8K test
camera appears in training; no Grok image from a test camera.

The label files in data/processed keep only identifiers, our label, our groups
and our split. Two inputs come from the original datasets: the DeepFlood
`ambiguous` flag from the annotation file in Flood_Dataset.zip (kept from the
existing manifest.csv when the zip is absent), and the STURM crops that are
in the official split lists data/{train,val,test}.txt.
"""
from __future__ import annotations

import csv
import io
import json
import sys
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
P = ROOT / "data" / "processed"
FIELDS = ["source", "zip", "member", "level_lo", "level_hi", "level3", "role", "test_set",
          "is_fisheye", "synthetic", "camera", "ambiguous", "group"]
FE8K_TEST_CAMERAS = {1, 2, 4, 7}


def _row(**kw):
    r = {k: "" for k in FIELDS}
    r.update(kw)
    return r


def _val_groups(items, key_group, key_strat, frac, rng):
    """Pick ~frac of groups per stratum for validation."""
    val = set()
    for s in sorted({key_strat(x) for x in items}):
        gs = sorted({key_group(x) for x in items if key_strat(x) == s})
        rng.shuffle(gs)
        val.update(gs[: int(round(frac * len(gs)))])
    return val


def deepflood_ambiguous(data_dir):
    """member -> 1 if the submerged objects of a DeepFlood image span two or more
    of its 11 levels, else 0. Read from the annotation file of Flood_Dataset.zip;
    without the zip, the flags of the existing manifest.csv are kept."""
    zp = Path(data_dir) / "Flood_Dataset.zip"
    if not zp.exists():
        print("note: Flood_Dataset.zip not found; DeepFlood `ambiguous` flags kept from manifest.csv")
        return {r["member"]: int(r["ambiguous"])
                for r in csv.DictReader(open(P / "manifest.csv", encoding="utf-8"))
                if r["source"] == "deepflood"}
    ann = json.loads(zipfile.ZipFile(zp).read("Flood_Dataset/flood_dict_new.json"))
    out = {}
    for img_id, objs in ann.items():
        sub = [int(o["level_id"]) for o in objs if o["category_id"] != 91 and int(o["level_id"]) > 0]
        out[f"Flood_Dataset/images_flood/{img_id}.jpg"] = int(bool(sub) and max(sub) - min(sub) >= 2)
    return out


def sturm_listed(data_dir):
    """Names of the STURM crops in the official split lists data/{train,val,test}.txt."""
    names = set()
    for s in ("train", "val", "test"):
        for line in open(Path(data_dir) / f"{s}.txt"):
            f = line.strip().replace("\\", "/").split("/")[-1]
            if f:
                names.add(f)
    return names


def build(data_dir=ROOT / "data", seed=0):
    data_dir = Path(data_dir)
    rng = np.random.default_rng(seed)
    rows = []
    # DeepFlood -----------------------------------------------------------
    df = list(csv.DictReader(open(P / "deepflood_labels.csv")))
    amb = deepflood_ambiguous(data_dir)
    tr = [r for r in df if r["split"] == "train"]
    val = _val_groups(tr, lambda r: r["dup_group"], lambda r: r["level5"], 0.1, rng)
    for r in df:
        lvl = int(r["level5"])
        if r["split"] == "test":
            role, ts = ("unused", "") if r["cross_dup"] == "1" else ("test", "deepflood_test")
        else:
            role, ts = ("val" if r["dup_group"] in val else "train"), ""
        rows.append(_row(source="deepflood", zip="Flood_Dataset.zip", member=r["file"],
                         level_lo=lvl, level_hi=lvl, role=role, test_set=ts, is_fisheye=0,
                         synthetic=0, ambiguous=amb[r["file"]],
                         group=f"df{r['dup_group']}"))
    # STURM (transfer test only) -------------------------------------------
    listed = sturm_listed(data_dir)
    for r in csv.DictReader(open(P / "sturm_labels.csv")):
        ok = (r["group_split"] == "test" and r["file"].split("/")[-1] in listed
              and r["cross_dup"] != "1")
        rows.append(_row(source="sturm", zip="flooded_cars.zip", member=r["file"],
                         level3=r["level3"], role="test" if ok else "unused",
                         test_set="sturm_test" if ok else "", is_fisheye=0, synthetic=0,
                         group=f"st{r['photo']}"))
    # AlleyFloodNet ----------------------------------------------------------
    for r in csv.DictReader(open(P / "alleyfloodnet_labels.csv", encoding="utf-8")):
        lo, hi = (0, 0) if r["class"] == "dry" else (1, 4)
        role = {"drop": "unused"}.get(r["split"], r["split"])
        rows.append(_row(source="alleyfloodnet", zip="AlleyFloodnet.zip", member=r["file"],
                         level_lo=lo, level_hi=hi, role=role,
                         test_set="alley_test" if role == "test" else "", is_fisheye=0,
                         synthetic=0, group=f"al{r['dup_group']}"))
    # FishEye8K sample -------------------------------------------------------
    z = zipfile.ZipFile(data_dir / "fisheye8k_sample.zip")
    meta = [n for n in z.namelist() if n.endswith(".csv")][0]
    folder = meta.rsplit("/", 1)[0] + "/" if "/" in meta else ""
    fe = list(csv.DictReader(io.TextIOWrapper(z.open(meta))))
    grok_base = {r["file"] for r in csv.DictReader(open(data_dir / "grok_base_frames.csv"))}
    cand = [r for r in fe if int(r["camera"]) not in FE8K_TEST_CAMERAS and r["file"] not in grok_base]
    val_files = set(rng.choice([r["file"] for r in cand], size=round(0.1 * len(fe)) // 1, replace=False))
    for r in fe:
        cam = int(r["camera"])
        if cam in FE8K_TEST_CAMERAS:
            role, ts = "test", "fe8k_test"
        else:
            role, ts = ("val" if r["file"] in val_files else "train"), ""
        rows.append(_row(source="fisheye8k", zip="fisheye8k_sample.zip", member=folder + r["file"],
                         level_lo=0, level_hi=0, role=role, test_set=ts, is_fisheye=1,
                         synthetic=0, camera=cam, group=f"cam{cam}"))
    # Grok (only once labelled) --------------------------------------------
    gl = data_dir / "grok_labels.csv"
    if gl.exists():
        for r in csv.DictReader(open(gl)):
            if r.get("scene_changed", "").strip() in ("1", "y", "yes", "Y") or r.get("final_level", "") == "":
                continue
            cam = int(r["camera"])
            lvl = int(r["final_level"])
            rows.append(_row(source="grok", zip="", member=f"grok/{Path(r['file']).name}", level_lo=lvl,
                             level_hi=lvl, role="train", is_fisheye=1, synthetic=1, camera=cam,
                             group=f"cam{cam}"))
    else:
        print("note: data/grok_labels.csv not found; Grok images not included yet")
    check(rows)
    with open(P / "manifest.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    return rows


def check(rows):
    for r in rows:
        if r["synthetic"] == 1:
            assert r["role"] == "train", r
        if r["source"] == "fisheye8k" and r["role"] in ("train", "val"):
            assert int(r["camera"]) not in FE8K_TEST_CAMERAS, r
        if r["source"] == "grok":
            assert int(r["camera"]) not in FE8K_TEST_CAMERAS, r
    groups = {}
    for r in rows:
        groups.setdefault(r["group"], set()).add(r["role"])
    leaking = [g for g, roles in groups.items() if "test" in roles and ({"train", "val"} & roles)]
    assert not leaking, f"duplicate groups across train/test: {leaking[:5]}"


def summary(rows):
    c = Counter((r["source"], r["role"]) for r in rows)
    t = Counter(r["test_set"] for r in rows if r["test_set"])
    return c, t


if __name__ == "__main__":
    rows = build(*(sys.argv[1:2] or []))
    c, t = summary(rows)
    for (s, role), n in sorted(c.items()):
        print(f"{s:14s} {role:7s} {n:5d}")
    print("test sets:", dict(t))
