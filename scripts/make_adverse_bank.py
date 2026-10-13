"""Build the adverse belief bank (exp13 condition model) from a detector run.

Usage: python scripts/make_adverse_bank.py --data-dir data --run-dir results/detector_base
On Modal (data and fold models on the volumes, CPU):
       python -m modal run modal_app.py::adverse --run detector_base
See src/detector/adverse_bank.py.
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from detector.adverse_bank import run  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--data-dir", default=str(ROOT / "data"))
ap.add_argument("--run-dir", default=str(ROOT / "results" / "detector_base"))
ap.add_argument("--threads", type=int, default=8)
a = ap.parse_args()
print(run(a.data_dir, a.run_dir, threads=a.threads))
