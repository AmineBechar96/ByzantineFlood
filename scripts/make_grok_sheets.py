"""Blind labelling sheets for the Grok flood edits (one sheet per base frame).

Input : data/grok_base_frames.csv (150 base frames); originals and edits are
        located with the paths defined in scripts/grok_flood_edit.py
        (FRAMES_DIR, out_path), so the two scripts cannot drift apart.
Output: data/grok_sheets/sheet_001.jpg ... sheet_150.jpg (JPEG quality 85)
        data/grok_sheets_key.csv     sheet, panel, file, camera, requested_level
        data/grok_labels_blank.csv   sheet, panel, visible_level, scene_changed, notes

Layout, 3 x 2 grid of 512x512 image panels (letterboxed, aspect kept), each
under a white caption bar:
    ORIGINAL | A | B
    C        | D | sheet id, camera, frame name
A-D are the four edits (L1-L4) in random order. The order is seeded with
2026 + n, where n is the sheet number (the frame's 1-based position in
grok_base_frames.csv); the CSV's frame_index column repeats across cameras
and would give several sheets the same order. The requested level appears
only in the key, never on a sheet. A missing or unreadable edit is drawn as
a grey panel and reported.

The labels file is not overwritten if it already exists (it may hold labels).

Usage: python scripts/make_grok_sheets.py
"""
import csv
import random
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps

sys.path.insert(0, str(Path(__file__).resolve().parent))
import grok_flood_edit as gfe  # noqa: E402  (edit locations: FRAMES_DIR, out_path)

ROOT = gfe.ROOT
SHEETS_DIR = ROOT / "data/grok_sheets"
KEY_CSV = ROOT / "data/grok_sheets_key.csv"
LABELS_CSV = ROOT / "data/grok_labels_blank.csv"
SEED_BASE = 2026
PANEL = 512
BAR = 52
GAP = 8
QUALITY = 85
LETTERS = "ABCD"
GRID = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)]  # ORIGINAL, A, B, C, D, info
FONT = ImageFont.load_default(size=34)
SHEET_BG, LETTERBOX, MISSING_GREY = (200, 200, 200), (0, 0, 0), (128, 128, 128)


def load(path):
    """RGB image, or (None, reason) if the file is missing or cannot be decoded."""
    if not path.exists():
        return None, "MISSING"
    try:
        with Image.open(path) as im:
            return im.convert("RGB"), None
    except Exception:
        return None, "UNREADABLE"


def panel_image(im):
    out = Image.new("RGB", (PANEL, PANEL), LETTERBOX if im else MISSING_GREY)
    if im:
        im = ImageOps.contain(im, (PANEL, PANEL), Image.LANCZOS)
        out.paste(im, ((PANEL - im.width) // 2, (PANEL - im.height) // 2))
    return out


def info_image(lines):
    out = Image.new("RGB", (PANEL, PANEL), "white")
    d = ImageDraw.Draw(out)
    for i, line in enumerate(lines):
        d.text((32, 140 + i * 70), line, fill="black", font=FONT)
    return out


def place(sheet, cell, caption, body):
    r, c = cell
    x, y = GAP + c * (PANEL + GAP), GAP + r * (BAR + PANEL + GAP)
    d = ImageDraw.Draw(sheet)
    d.rectangle([x, y, x + PANEL - 1, y + BAR - 1], fill="white")
    d.text((x + PANEL // 2, y + BAR // 2), caption, fill="black", font=FONT, anchor="mm")
    sheet.paste(body, (x, y + BAR))


def main():
    frames = list(csv.DictReader(open(gfe.FRAMES_CSV, newline="", encoding="utf-8")))
    SHEETS_DIR.mkdir(parents=True, exist_ok=True)
    size = (3 * PANEL + 4 * GAP, 2 * (BAR + PANEL) + 3 * GAP)
    key, problems = [], []

    for n, fr in enumerate(frames, start=1):
        sheet_id = f"sheet_{n:03d}"
        frame = Path(fr["file"]).stem
        camera = gfe.frame_info(fr["file"])[0]
        order = list(gfe.LEVELS)
        random.Random(SEED_BASE + n).shuffle(order)

        sheet = Image.new("RGB", size, SHEET_BG)
        orig_path = gfe.FRAMES_DIR / fr["file"]
        im, err = load(orig_path)
        if err:
            problems.append((sheet_id, "ORIGINAL", orig_path, err))
        place(sheet, GRID[0], "ORIGINAL" if im else f"ORIGINAL - {err}", panel_image(im))

        for letter, level, cell in zip(LETTERS, order, GRID[1:5]):
            path = gfe.out_path(frame, level)
            im, err = load(path)
            if err:
                problems.append((sheet_id, letter, path, err))
            place(sheet, cell, letter if im else f"{letter} - {err}", panel_image(im))
            key.append(dict(sheet=sheet_id, panel=letter, file=path.relative_to(ROOT).as_posix(),
                            camera=camera, requested_level=level))

        place(sheet, GRID[5], sheet_id, info_image([sheet_id, f"camera {camera}", frame]))
        sheet.save(SHEETS_DIR / f"{sheet_id}.jpg", "JPEG", quality=QUALITY)

    with open(KEY_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["sheet", "panel", "file", "camera", "requested_level"])
        w.writeheader()
        w.writerows(key)
    if LABELS_CSV.exists():
        print(f"{LABELS_CSV.relative_to(ROOT)} exists; left untouched")
    else:
        with open(LABELS_CSV, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["sheet", "panel", "visible_level", "scene_changed", "notes"])
            w.writerows([k["sheet"], k["panel"], "", "", ""] for k in key)

    sheets = sorted(SHEETS_DIR.glob("sheet_*.jpg"))
    mb = sum(p.stat().st_size for p in sheets) / 1e6
    print(f"{len(sheets)} sheets in {SHEETS_DIR.relative_to(ROOT)}, {mb:.1f} MB; "
          f"key {len(key)} rows; problems: {len(problems)}")
    for p in problems:
        print("  ", p[0], p[1], p[2].relative_to(ROOT).as_posix(), p[3])


if __name__ == "__main__":
    main()
