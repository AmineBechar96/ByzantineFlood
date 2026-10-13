# detector_sim2real rerun with `val_synthetic: true`, 2026-10-05

Generated from results/<run>/ (fold predictions downloaded from Modal). Ensemble = mean of
the fold beliefs; level = ordinal median. 95 % bootstrap CIs (2000 resamples, seed 0).
`folds 0-2` = detector_base restricted to the three folds the ablations trained on.

## Runs

| run | folds | train / val per fold | epochs per fold (best) | wall time | cost |
|---|---|---|---|---|---|
| detector_base | 5 | 2439-2532 / 473-522 | 19 (13), 22 (16), 13 (7), 23 (17), 16 (10) | 50.3 min | $1.11 |
| detector_sim2real_dryval | 5 | 950-1044 / 100-150 | 13 (7), 24 (18), 9 (3), 17 (11), 25 (23) | 27.3 min | $0.60 |
| detector_sim2real | 5 | 950-1044 / 187-281 | 13 (7), 13 (7), 13 (7), 23 (17), 21 (15) | 26.7 min | $0.59 (partial) |

## deepflood_test

| run | ensemble | acc (n=252) | within1 (n=252) | mae (n=252) | ece (n=252) | flooded_auc (n=252) | false_alarm (n=11) |
|---|---|---|---|---|---|---|---|
| detector_base | 5 folds | 0.508 [0.448, 0.572] | 0.893 [0.853, 0.929] | 0.623 [0.532, 0.718] | 0.089 [0.054, 0.153] | 0.781 [0.585, 0.941] | 0.545 [0.273, 0.818] |
| detector_base | folds 0-2 | 0.488 [0.425, 0.548] | 0.885 [0.841, 0.921] | 0.659 [0.567, 0.758] | 0.085 [0.050, 0.149] | 0.781 [0.589, 0.941] | 0.636 [0.364, 0.909] |
| detector_sim2real_dryval | 5 folds | 0.266 [0.214, 0.321] | 0.631 [0.571, 0.694] | 1.151 [1.040, 1.258] | 0.431 [0.375, 0.487] | 0.768 [0.620, 0.892] | 1.000 [1.000, 1.000] |
| detector_sim2real | 5 folds | 0.262 [0.210, 0.314] | 0.607 [0.544, 0.671] | 1.175 [1.063, 1.282] | 0.122 [0.069, 0.175] | 0.667 [0.480, 0.838] | 1.000 [1.000, 1.000] |

## deepflood_test_fisheye

| run | ensemble | acc (n=252) | within1 (n=252) | mae (n=252) | ece (n=252) | flooded_auc (n=252) | false_alarm (n=11) |
|---|---|---|---|---|---|---|---|
| detector_base | 5 folds | 0.476 [0.413, 0.536] | 0.901 [0.861, 0.937] | 0.655 [0.567, 0.750] | 0.097 [0.057, 0.158] | 0.794 [0.615, 0.934] | 0.818 [0.545, 1.000] |
| detector_base | folds 0-2 | 0.488 [0.425, 0.548] | 0.897 [0.857, 0.933] | 0.647 [0.560, 0.738] | 0.101 [0.060, 0.164] | 0.800 [0.627, 0.935] | 0.818 [0.545, 1.000] |
| detector_sim2real_dryval | 5 folds | 0.278 [0.226, 0.333] | 0.595 [0.536, 0.655] | 1.210 [1.091, 1.329] | 0.406 [0.351, 0.464] | 0.655 [0.476, 0.826] | 1.000 [1.000, 1.000] |
| detector_sim2real | 5 folds | 0.250 [0.198, 0.306] | 0.587 [0.524, 0.647] | 1.262 [1.139, 1.381] | 0.137 [0.084, 0.188] | 0.616 [0.416, 0.807] | 1.000 [1.000, 1.000] |

## sturm_test

| run | ensemble | acc_dry_vs_flooded (n=337) | flooded_auc (n=337) | false_alarm (n=88) | flood_recall (n=249) | kendall_tau_flooded (n=249) |
|---|---|---|---|---|---|---|
| detector_base | 5 folds | 0.783 [0.739, 0.825] | 0.822 [0.769, 0.870] | 0.830 [0.750, 0.898] | 1.000 [1.000, 1.000] | 0.405 [0.325, 0.486] |
| detector_base | folds 0-2 | 0.777 [0.733, 0.822] | 0.824 [0.772, 0.872] | 0.852 [0.773, 0.920] | 1.000 [1.000, 1.000] | 0.387 [0.301, 0.470] |
| detector_sim2real_dryval | 5 folds | 0.736 [0.688, 0.783] | 0.613 [0.543, 0.685] | 0.989 [0.966, 1.000] | 0.992 [0.980, 1.000] | 0.016 [-0.091, 0.117] |
| detector_sim2real | 5 folds | 0.739 [0.691, 0.786] | 0.601 [0.531, 0.671] | 0.977 [0.943, 1.000] | 0.992 [0.980, 1.000] | -0.053 [-0.154, 0.048] |

## alley_test

| run | ensemble | acc_dry_vs_flooded (n=229) | flooded_auc (n=229) | false_alarm (n=108) | flood_recall (n=121) |
|---|---|---|---|---|---|
| detector_base | 5 folds | 0.930 [0.895, 0.961] | 0.984 [0.971, 0.994] | 0.074 [0.028, 0.130] | 0.934 [0.884, 0.975] |
| detector_base | folds 0-2 | 0.917 [0.882, 0.952] | 0.983 [0.969, 0.993] | 0.083 [0.037, 0.139] | 0.917 [0.860, 0.959] |
| detector_sim2real_dryval | 5 folds | 0.541 [0.476, 0.607] | 0.858 [0.806, 0.904] | 0.972 [0.935, 1.000] | 1.000 [1.000, 1.000] |
| detector_sim2real | 5 folds | 0.537 [0.472, 0.603] | 0.840 [0.785, 0.890] | 0.981 [0.954, 1.000] | 1.000 [1.000, 1.000] |

## fe8k_test

| run | ensemble | false_alarm (n=200) |
|---|---|---|
| detector_base | 5 folds | 0.070 [0.035, 0.105] |
| detector_base | folds 0-2 | 0.035 [0.010, 0.065] |
| detector_sim2real_dryval | 5 folds | 0.065 [0.030, 0.100] |
| detector_sim2real | 5 folds | 0.105 [0.065, 0.150] |

## deepflood_test confusion matrix, detector_base (5 folds)

| true \ predicted | 0 | 1 | 2 | 3 | 4 | n | recall |
|---|---|---|---|---|---|---|---|
| 0 | 5 | 0 | 4 | 1 | 1 | 11 | 0.45 |
| 1 | 1 | 5 | 19 | 6 | 2 | 33 | 0.15 |
| 2 | 0 | 5 | 40 | 22 | 4 | 71 | 0.56 |
| 3 | 0 | 4 | 16 | 15 | 15 | 50 | 0.30 |
| 4 | 0 | 1 | 4 | 19 | 63 | 87 | 0.72 |

## Paired differences (A - B, same test images, paired bootstrap)

| A - B | test set | metric | difference [95 % CI] |
|---|---|---|---|
| detector_sim2real (5 folds) - detector_base (5 folds) | deepflood_test | acc | -0.246 [-0.325, -0.171] |
| detector_sim2real (5 folds) - detector_base (5 folds) | deepflood_test | mae | +0.552 [+0.409, +0.690] |
| detector_sim2real (5 folds) - detector_base (5 folds) | deepflood_test_fisheye | acc | -0.226 [-0.306, -0.147] |
| detector_sim2real (5 folds) - detector_base (5 folds) | deepflood_test_fisheye | mae | +0.607 [+0.456, +0.754] |
| detector_sim2real (5 folds) - detector_base (5 folds) | sturm_test | flooded_auc | -0.221 [-0.280, -0.159] |
| detector_sim2real (5 folds) - detector_base (5 folds) | alley_test | flooded_auc | -0.144 [-0.198, -0.095] |
| detector_sim2real (5 folds) - detector_base (5 folds) | alley_test | false_alarm | +0.907 [+0.852, +0.963] |
| detector_sim2real (5 folds) - detector_base (5 folds) | fe8k_test | false_alarm | +0.035 [-0.010, +0.085] |
| detector_sim2real (5 folds) - detector_sim2real_dryval (5 folds) | deepflood_test | acc | -0.004 [-0.028, +0.020] |
| detector_sim2real (5 folds) - detector_sim2real_dryval (5 folds) | deepflood_test | mae | +0.024 [-0.008, +0.060] |
| detector_sim2real (5 folds) - detector_sim2real_dryval (5 folds) | deepflood_test_fisheye | acc | -0.028 [-0.052, -0.004] |
| detector_sim2real (5 folds) - detector_sim2real_dryval (5 folds) | deepflood_test_fisheye | mae | +0.052 [+0.016, +0.087] |
| detector_sim2real (5 folds) - detector_sim2real_dryval (5 folds) | sturm_test | flooded_auc | -0.012 [-0.041, +0.017] |
| detector_sim2real (5 folds) - detector_sim2real_dryval (5 folds) | alley_test | flooded_auc | -0.018 [-0.037, -0.001] |
| detector_sim2real (5 folds) - detector_sim2real_dryval (5 folds) | alley_test | false_alarm | +0.009 [+0.000, +0.028] |
| detector_sim2real (5 folds) - detector_sim2real_dryval (5 folds) | fe8k_test | false_alarm | +0.040 [+0.010, +0.070] |

## Belief banks (ensemble/belief_bank.npz)

| run | beliefs | per level 0-4 | sources | exact acc | level-0 predicted flooded | flooded predicted 0 |
|---|---|---|---|---|---|---|
| detector_base | 560 | [319, 33, 71, 50, 87] | alley_test 108, deepflood_test 252, fe8k_test 200 | 0.739 | 0.088 | 0.004 |
| detector_sim2real_dryval | 560 | [319, 33, 71, 50, 87] | alley_test 108, deepflood_test 252, fe8k_test 200 | 0.459 | 0.404 | 0.004 |
| detector_sim2real | 560 | [319, 33, 71, 50, 87] | alley_test 108, deepflood_test 252, fe8k_test 200 | 0.441 | 0.433 | 0.004 |

## Folds (sim2real runs)

| run | fold | train | val | epochs run | best epoch | early stop | best val loss | temperature |
|---|---|---|---|---|---|---|---|---|
| detector_sim2real_dryval | 0 | 950 | 150 | 13 | 7 | 1 | 0.0811 | 0.2 |
| detector_sim2real_dryval | 1 | 979 | 130 | 24 | 18 | 1 | 0.003 | 0.2 |
| detector_sim2real_dryval | 2 | 971 | 133 | 9 | 3 | 1 | 0.0247 | 0.2 |
| detector_sim2real_dryval | 3 | 980 | 143 | 17 | 11 | 1 | 0.0458 | 0.2 |
| detector_sim2real_dryval | 4 | 1044 | 100 | 25 | 23 | 0 | 0.0152 | 0.2 |
| detector_sim2real | 0 | 950 | 281 | 13 | 7 | 1 | 0.2117 | 0.485 |
| detector_sim2real | 1 | 979 | 252 | 13 | 7 | 1 | 0.1819 | 0.754 |
| detector_sim2real | 2 | 971 | 260 | 13 | 7 | 1 | 0.1684 | 0.568 |
| detector_sim2real | 3 | 980 | 251 | 23 | 17 | 1 | 0.1674 | 0.569 |
| detector_sim2real | 4 | 1044 | 187 | 21 | 15 | 1 | 0.1724 | 0.659 |

## deepflood_test confusion matrix, detector_sim2real (5 folds)

| true \ predicted | 0 | 1 | 2 | 3 | 4 | n | recall |
|---|---|---|---|---|---|---|---|
| 0 | 0 | 4 | 7 | 0 | 0 | 11 | 0.00 |
| 1 | 0 | 3 | 24 | 6 | 0 | 33 | 0.09 |
| 2 | 0 | 4 | 54 | 13 | 0 | 71 | 0.76 |
| 3 | 0 | 4 | 37 | 9 | 0 | 50 | 0.18 |
| 4 | 1 | 9 | 72 | 5 | 0 | 87 | 0.00 |

## deepflood_test confusion matrix, detector_sim2real_dryval (5 folds)

| true \ predicted | 0 | 1 | 2 | 3 | 4 | n | recall |
|---|---|---|---|---|---|---|---|
| 0 | 0 | 4 | 7 | 0 | 0 | 11 | 0.00 |
| 1 | 0 | 5 | 25 | 3 | 0 | 33 | 0.15 |
| 2 | 0 | 3 | 54 | 14 | 0 | 71 | 0.76 |
| 3 | 0 | 4 | 38 | 8 | 0 | 50 | 0.16 |
| 4 | 1 | 10 | 68 | 8 | 0 | 87 | 0.00 |

## Notes

- detector_sim2real: `val_synthetic: true` (configs/detector_sim2real.yaml). Training parts are
  identical to the old run; validation = dry FishEye8K frames of the held-out cameras + the Grok
  images of those cameras (87-131 per fold), used for early stopping and temperature only.
- detector_sim2real_dryval: the earlier run (validation on dry FishEye8K frames only, temperature
  clamped at 0.2 in every fold), kept for reference; it is the run in results/detector_final/.
- DeepFlood false-alarm rate and flooded AUC rest on 11 dry test images.
- STURM dry-vs-flooded accuracy: all-flooded baseline 249/337 = 0.739.

Files: metrics.csv (all metrics with CIs), confusion_deepflood_test.csv (every run),
paired_differences.csv, folds.csv (epochs, best epoch, temperature), per_fold_metrics.csv,
belief_banks.csv, runs_cost.csv (Modal app times and billed cost per resource).
