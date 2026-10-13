"""Choose the FishEye8K base frames for Grok flood editing (training only).

Reads data/fisheye8k_sample.zip (856 frames, 18 cameras, CSV inside) and
writes data/grok_base_frames.csv: 150 frames from the 14 FishEye8K
*training* cameras only (test cameras 1, 2, 4, 7 are never edited, so no
test scene leaks into training). Frames are evenly spaced in each camera's
sequence; daytime only (v2: night infrared frames could not show depth in
the pilot); camera 11 (hazy, green tint) contributes fewer. Rows are ordered
round-robin across cameras, starting with cameras that show vehicles, so
`--frames N` tests N different cameras. Only the frame names are written; the
camera and time of day are part of each name (grok_flood_edit.frame_info).
"""
import csv
import io
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
QUOTA = {3: 11, 5: 11, 6: 10, 8: 11, 9: 11, 10: 11, 11: 8, 12: 10,
         13: 11, 14: 11, 15: 11, 16: 11, 17: 11, 18: 12}
ORDER = [8, 12, 17, 3, 5, 6, 9, 10, 11, 13, 14, 15, 16, 18]


def spaced(rows, k):
    rows = sorted(rows, key=lambda r: int(r["frame_index"]))
    if k >= len(rows):
        return rows
    idx = [round(i * (len(rows) - 1) / (k - 1)) for i in range(k)]
    return [rows[i] for i in idx]


def main(zip_path=ROOT / "data/fisheye8k_sample.zip", out=ROOT / "data/grok_base_frames.csv"):
    z = zipfile.ZipFile(zip_path)
    meta = [n for n in z.namelist() if n.endswith(".csv")][0]
    rows = list(csv.DictReader(io.TextIOWrapper(z.open(meta))))
    assert sum(QUOTA.values()) == 150
    per_cam = {}
    for cam, k in QUOTA.items():
        rs = [r for r in rows if int(r["camera"]) == cam and r["split"] == "train"
              and r["time_of_day"] != "night"]
        per_cam[cam] = spaced(rs, k)
    chosen = []
    for i in range(max(len(v) for v in per_cam.values())):
        for cam in ORDER:
            if i < len(per_cam[cam]):
                chosen.append(per_cam[cam][i])
    assert len(chosen) == 150 and all(r["split"] == "train" for r in chosen)
    assert all(r["time_of_day"] != "night" for r in chosen)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["file"])
        w.writeheader()
        for r in chosen:
            w.writerow({k: r[k] for k in w.fieldnames})
    print(f"{len(chosen)} base frames -> {out}")


if __name__ == "__main__":
    import sys
    main(*(Path(a) for a in sys.argv[1:3]))
