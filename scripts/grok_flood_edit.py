"""Grok flood edits of FishEye8K training frames (synthetic: TRAINING ONLY).

Input : data/grok_base_frames.csv (frame names; images in data/fisheye8k_sample/;
        camera and time of day are read from the name, see frame_info),
        configs/grok_prompts.yaml (preamble, level and time-of-day text)
Output: data/grok/<frame>_L<level>.jpg, one edit per frame and level 1-4
        data/grok/log.csv, one row per API request

xAI API (docs.x.ai, read 3 Oct 2026): POST /v1/images/edits with a JSON body
{model, prompt, image: {url: <data URI>}}. The cheapest model that accepts an
input image is grok-imagine-image: $0.002 per input image + $0.02 per output
image at 1K or 2K = $0.022 per edit (docs.x.ai/developers/pricing, updated
29 Sep 2026; grok-imagine-image-2.0 costs $0.05 and -quality $0.06 per edit).

Spend is tracked from these documented prices and rebuilt from log.csv on
every start, so the cap holds across resumed runs. No request is sent if it
could take the total past BUDGET_USD. A request counts at the full price
unless the API rejected it before generating anything (4xx other than a
content-policy rejection, which the docs say is still billed); timeouts and
5xx count, since the image may have been generated. If the API reports a
higher cost (usage.cost_in_usd_ticks) than the documented price, the higher
value counts.

Resume: a (frame, level) whose output file exists is skipped.
The API key is read from the environment variable XAI_API_KEY only.

Usage: python scripts/grok_flood_edit.py [--frames N] [--dry-run]
"""
import argparse
import base64
import csv
import io
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import requests
import yaml
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
FRAMES_CSV = ROOT / "data/grok_base_frames.csv"
FRAMES_DIR = ROOT / "data/fisheye8k_sample"
PROMPTS = ROOT / "configs/grok_prompts.yaml"
OUT_DIR = ROOT / "data/grok"
LOG = OUT_DIR / "log.csv"

API_URL = "https://api.x.ai/v1/images/edits"
MODEL = "grok-imagine-image"
RESOLUTION = "1k"
PRICE_INPUT_USD = 0.002
PRICE_OUTPUT_USD = 0.02
PRICE_PER_EDIT = PRICE_INPUT_USD + PRICE_OUTPUT_USD
BUDGET_USD = 14.00
TICKS_PER_USD = 10_000_000_000
LEVELS = (1, 2, 3, 4)
MAX_ATTEMPTS = 3
TIMEOUT_S = 300
LOG_FIELDS = ["timestamp", "frame", "level", "model", "status", "http_status", "cost_usd",
              "api_cost_usd", "total_usd", "output_file", "detail"]
POLICY = re.compile(r"moderat|polic|violat|safety|inappropriate", re.I)
FRAME_NAME = re.compile(r"camera(\d+)_([AEMN])_(\d+)\.jpg")
TIME_OF_DAY = {"A": "afternoon", "E": "evening", "M": "morning", "N": "night"}


def frame_info(file):
    """(camera, time of day) of a FishEye8K frame, from its name camera<c>_<A|E|M|N>_<index>.jpg."""
    m = FRAME_NAME.fullmatch(Path(file).name)
    return int(m.group(1)), TIME_OF_DAY[m.group(2)]


def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def build_prompt(cfg, level, time_of_day):
    tod = cfg["time_of_day"].get(time_of_day, cfg["time_of_day"]["default"])
    parts = (cfg.get("preamble", ""), cfg["levels"][level], cfg.get("constraints", ""), tod)
    return " ".join(p.strip() for p in parts if p and p.strip())


def out_path(frame, level):
    return OUT_DIR / f"{frame}_L{level}.jpg"


class Log:
    def __init__(self, path):
        self.path = path
        rows = list(csv.DictReader(open(path, newline="", encoding="utf-8"))) if path.exists() else []
        self.total = sum(float(r["cost_usd"] or 0) for r in rows)
        self.ok = {(r["frame"], int(r["level"])) for r in rows if r["status"] in ("ok", "found_on_disk")}
        new = not path.exists()
        self.f = open(path, "a", newline="", encoding="utf-8")
        self.w = csv.DictWriter(self.f, fieldnames=LOG_FIELDS)
        if new:
            self.w.writeheader()

    def write(self, frame, level, status, http_status, cost, api_cost, output_file, detail):
        self.total += cost
        self.w.writerow(dict(timestamp=now(), frame=frame, level=level, model=MODEL, status=status,
                             http_status=http_status, cost_usd=f"{cost:.6f}",
                             api_cost_usd="" if api_cost is None else f"{api_cost:.6f}",
                             total_usd=f"{self.total:.6f}", output_file=output_file, detail=detail))
        self.f.flush()


def request_edit(api_key, image_path, prompt):
    data_uri = "data:image/jpeg;base64," + base64.b64encode(image_path.read_bytes()).decode()
    body = {"model": MODEL, "prompt": prompt, "image": {"url": data_uri, "type": "image_url"},
            "n": 1, "resolution": RESOLUTION, "response_format": "b64_json"}
    return requests.post(API_URL, json=body, timeout=TIMEOUT_S,
                         headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"})


def save_jpeg(item, dst):
    raw = base64.b64decode(item["b64_json"])
    tmp = dst.with_suffix(".part")
    with Image.open(io.BytesIO(raw)) as im:
        size, fmt = im.size, im.format
        if fmt == "JPEG":
            tmp.write_bytes(raw)  # keep the bytes the API returned
        else:
            im.convert("RGB").save(tmp, "JPEG", quality=95)
    os.replace(tmp, dst)
    return f"{size[0]}x{size[1]} {fmt}"


def edit_one(api_key, log, frame, level, image_path, prompt):
    """Up to MAX_ATTEMPTS requests for one (frame, level). Returns False if the run must stop."""
    dst = out_path(frame, level)
    rel = dst.relative_to(ROOT).as_posix()
    for attempt in range(1, MAX_ATTEMPTS + 1):
        if log.total + PRICE_PER_EDIT > BUDGET_USD + 1e-9:
            print(f"budget: ${log.total:.3f} spent, next edit would pass ${BUDGET_USD:.2f}; stopping")
            return False
        try:
            r = request_edit(api_key, image_path, prompt)
        except requests.RequestException as e:  # may have been generated and billed
            log.write(frame, level, "network_error", "", PRICE_PER_EDIT, None, "", repr(e)[:300])
            time.sleep(10 * attempt)
            continue
        text = r.text
        if r.status_code == 200:
            js = r.json()
            ticks = (js.get("usage") or {}).get("cost_in_usd_ticks")
            api_cost = ticks / TICKS_PER_USD if ticks is not None else None
            cost = max(PRICE_PER_EDIT, api_cost or 0)
            data = js.get("data") or []
            if not data or not data[0].get("b64_json"):
                log.write(frame, level, "no_image", 200, cost, api_cost, "", text[:300])
                return True
            info = save_jpeg(data[0], dst)
            log.write(frame, level, "ok", 200, cost, api_cost, rel, info)
            return True
        if r.status_code in (401, 403):
            log.write(frame, level, "auth_error", r.status_code, 0.0, None, "", text[:300])
            print(f"HTTP {r.status_code}: check XAI_API_KEY and team access to {MODEL}; stopping\n{text[:300]}")
            return False
        if r.status_code == 429:
            log.write(frame, level, "rate_limited", 429, 0.0, None, "", text[:300])
            time.sleep(float(r.headers.get("retry-after") or 20 * attempt))
            continue
        if r.status_code >= 500:
            log.write(frame, level, "server_error", r.status_code, PRICE_PER_EDIT, None, "", text[:300])
            time.sleep(10 * attempt)
            continue
        if POLICY.search(text):  # documented as billed; the same prompt would fail again
            log.write(frame, level, "policy_rejected", r.status_code, PRICE_PER_EDIT, None, "", text[:300])
            return True
        log.write(frame, level, "request_error", r.status_code, 0.0, None, "", text[:300])
        print(f"HTTP {r.status_code} for {frame} L{level}; stopping (request needs fixing)\n{text[:300]}")
        return False
    print(f"{frame} L{level}: gave up after {MAX_ATTEMPTS} attempts")
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--frames", type=int, default=None, help="only the first N frames of the CSV")
    ap.add_argument("--dry-run", action="store_true", help="print the plan and prompts, send nothing")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(PROMPTS, encoding="utf-8"))
    frames = list(csv.DictReader(open(FRAMES_CSV, newline="", encoding="utf-8")))[:args.frames]
    for fr in frames:
        if not (FRAMES_DIR / fr["file"]).exists():
            sys.exit(f"missing base frame {FRAMES_DIR / fr['file']}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    jobs = [(Path(fr["file"]).stem, lvl, FRAMES_DIR / fr["file"], build_prompt(cfg, lvl, frame_info(fr["file"])[1]))
            for fr in frames for lvl in LEVELS]

    if args.dry_run:
        todo = [j for j in jobs if not out_path(j[0], j[1]).exists()]
        for frame, lvl, _, prompt in todo[:8]:
            print(f"--- {frame} L{lvl}\n{prompt}\n")
        print(f"{len(frames)} frames, {len(jobs)} edits, {len(todo)} to do, "
              f"~${len(todo) * PRICE_PER_EDIT:.2f} at ${PRICE_PER_EDIT:.3f}/edit ({MODEL})")
        return

    api_key = os.environ.get("XAI_API_KEY")
    if not api_key:
        sys.exit("XAI_API_KEY is not set in the environment")

    log = Log(LOG)
    for frame, lvl, _, _ in jobs:  # outputs saved but never logged (crash between save and log)
        if out_path(frame, lvl).exists() and (frame, lvl) not in log.ok:
            log.write(frame, lvl, "found_on_disk", "", PRICE_PER_EDIT, None,
                      out_path(frame, lvl).relative_to(ROOT).as_posix(), "not in log; counted at list price")
    print(f"{len(frames)} frames x {len(LEVELS)} levels, model {MODEL}, "
          f"${PRICE_PER_EDIT:.3f}/edit, spent so far ${log.total:.3f} of ${BUDGET_USD:.2f}")

    done = skipped = 0
    for frame, lvl, image_path, prompt in jobs:
        if out_path(frame, lvl).exists():
            skipped += 1
            continue
        if not edit_one(api_key, log, frame, lvl, image_path, prompt):
            break
        done += out_path(frame, lvl).exists()
        print(f"{frame} L{lvl}: {'ok' if out_path(frame, lvl).exists() else 'FAILED'}  total ${log.total:.3f}",
              flush=True)
    print(f"new edits: {done}, already done: {skipped}, spent ${log.total:.3f} of ${BUDGET_USD:.2f}")


if __name__ == "__main__":
    main()
