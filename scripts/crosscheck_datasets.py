"""Flag photos that appear in both DeepFlood and STURM-FloodDepth.

Cars annotated in DeepFlood are cropped as STURM did (box enlarged by 20%)
and compared with every STURM crop by a 256-bit difference hash. Pairs within
Hamming distance 40 were inspected by eye (3 October 2026): Flood_449 /
car-2109 is the same photo; Flood_613 / car3-2 and Flood_231 / car-1874 show
the same flood events. All are flagged `cross_dup = 1` in both label files
and must be excluded from any test split used for cross-dataset transfer.
"""
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# Flood_118 is the same photo as an AlleyFloodNet test image (see
# scripts/prepare_alleyfloodnet.py); flagged here for symmetry.
DEEPFLOOD = {"Flood_449", "Flood_613", "Flood_231", "Flood_118"}
STURM_PHOTOS = {"car-2109", "car3-2", "car-1874"}


def flag(path, key, ids):
    rows = list(csv.DictReader(open(path)))
    for r in rows:
        r["cross_dup"] = int(r[key] in ids)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    return sum(r["cross_dup"] for r in rows)


if __name__ == "__main__":
    p = ROOT / "data" / "processed"
    print("DeepFlood flagged:", flag(p / "deepflood_labels.csv", "image_id", DEEPFLOOD))
    print("STURM crops flagged:", flag(p / "sturm_labels.csv", "photo", STURM_PHOTOS))
