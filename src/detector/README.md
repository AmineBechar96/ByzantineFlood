# Detector

## Step 17 -- data and augmentation (done, 4 Oct 2026)
- `manifest.py`: builds `data/processed/manifest.csv` from the label files.
  Interval labels [level_lo, level_hi]; roles train/val/test/unused; test
  sets `deepflood_test` (5 levels, primary), `sturm_test` (3-level transfer),
  `alley_test` (dry vs flooded, wet night streets), `fe8k_test` (dry
  fisheye, cameras 1, 2, 4, 7). `check()` enforces the leak rules and runs on
  every build. Run: `PYTHONPATH=src python -m detector.manifest`.
- `transforms.py`: letterbox to 384 px (no cropping), fisheye warp
  (equidistant re-projection into a black circle) for non-fisheye sources,
  nuisance augmentations (low light, infrared night, glare, rain streaks,
  lens droplets, JPEG, colour jitter), all seeded and logged.
- `io.py`: reads images directly from the dataset zips.

Current manifest (without Grok): train 2,246 real images (DeepFlood 912,
AlleyFloodNet 764, FishEye8K 570), val 271, test 1,018 across four test sets.
Grok images join training once `data/grok_labels.csv` exists.

## Step 18 -- training on Modal (written 4 Oct 2026; smoke test pending)
Ordinal (cumulative-link) head: four outputs P(level > k), k = 0..3, trained
with masked binary cross-entropy so interval labels supervise only the
thresholds they determine. EfficientNet-B0, 384 px, 5 folds. Post-hoc
temperature scaling on val; the four validation numbers on the test sets;
belief bank = per-level probability vectors on deepflood_test.

Files: `ordinal.py` (numpy: interval targets, beliefs, metrics, bootstrap
CIs -- tested), `report.py` (four validation numbers + belief bank -- tested),
`train.py` (PyTorch: dataset, rank-consistent ordinal head, masked BCE,
camera-grouped folds, temperature scaling, ensemble). `train.py` could not be
executed in the authoring environment (no PyTorch); run
`configs/detector_smoke.yaml` first.

Configs: `detector_smoke` (1 fold, 64 images, 1 epoch), `detector_base`,
ablations `detector_no_fisheye`, `detector_no_nuisance`, and
`detector_sim2real` (train on Grok + dry fisheye only, test on real).

Transfer test note: STURM's levels do not share boundaries with the 5-level
scale, so transfer is reported as dry-vs-flooded accuracy/AUC and the Kendall
rank correlation between predicted and STURM levels among flooded crops.
