# Final detector runs (human-checked Grok labels), 2026-10-05

Generated from results/<run>/ (fold predictions downloaded from Modal). Ensemble = mean of
the fold beliefs; level = ordinal median. 95 % bootstrap CIs (2000 resamples, seed 0).
`folds 0-2` = detector_base restricted to the three folds the ablations trained on.

## Runs

| run | folds | train / val per fold | epochs per fold (best) | wall time | cost |
|---|---|---|---|---|---|
| detector_base | 5 | 2439-2532 / 473-522 | 19 (13), 22 (16), 13 (7), 23 (17), 16 (10) | 50.3 min | $1.11 |
| detector_no_fisheye | 3 | 2439-2469 / 501-522 | 19 (13), 21 (15), 12 (6) | 32.1 min | $0.70 |
| detector_no_nuisance | 3 | 2439-2469 / 501-522 | 13 (7), 18 (12), 12 (5) | 14.1 min | $0.31 |
| detector_sim2real | 5 | 950-1044 / 100-150 | 13 (7), 24 (18), 9 (3), 17 (11), 25 (23) | 27.3 min | $0.60 |

## deepflood_test

| run | ensemble | acc (n=252) | within1 (n=252) | mae (n=252) | ece (n=252) | flooded_auc (n=252) | false_alarm (n=11) |
|---|---|---|---|---|---|---|---|
| detector_base | 5 folds | 0.508 [0.448, 0.572] | 0.893 [0.853, 0.929] | 0.623 [0.532, 0.718] | 0.089 [0.054, 0.153] | 0.781 [0.585, 0.941] | 0.545 [0.273, 0.818] |
| detector_base | folds 0-2 | 0.488 [0.425, 0.548] | 0.885 [0.841, 0.921] | 0.659 [0.567, 0.758] | 0.085 [0.050, 0.149] | 0.781 [0.589, 0.941] | 0.636 [0.364, 0.909] |
| detector_no_fisheye | 3 folds | 0.500 [0.437, 0.560] | 0.893 [0.853, 0.929] | 0.643 [0.552, 0.738] | 0.070 [0.041, 0.135] | 0.801 [0.623, 0.948] | 0.727 [0.455, 0.909] |
| detector_no_nuisance | 3 folds | 0.484 [0.425, 0.548] | 0.877 [0.837, 0.917] | 0.671 [0.579, 0.770] | 0.077 [0.051, 0.140] | 0.796 [0.606, 0.949] | 0.636 [0.364, 0.909] |
| detector_sim2real | 5 folds | 0.266 [0.214, 0.321] | 0.631 [0.571, 0.694] | 1.151 [1.040, 1.258] | 0.431 [0.375, 0.487] | 0.768 [0.620, 0.892] | 1.000 [1.000, 1.000] |

## deepflood_test_fisheye

| run | ensemble | acc (n=252) | within1 (n=252) | mae (n=252) | ece (n=252) | flooded_auc (n=252) | false_alarm (n=11) |
|---|---|---|---|---|---|---|---|
| detector_base | 5 folds | 0.476 [0.413, 0.536] | 0.901 [0.861, 0.937] | 0.655 [0.567, 0.750] | 0.097 [0.057, 0.158] | 0.794 [0.615, 0.934] | 0.818 [0.545, 1.000] |
| detector_base | folds 0-2 | 0.488 [0.425, 0.548] | 0.897 [0.857, 0.933] | 0.647 [0.560, 0.738] | 0.101 [0.060, 0.164] | 0.800 [0.627, 0.935] | 0.818 [0.545, 1.000] |
| detector_no_fisheye | 3 folds | 0.456 [0.397, 0.512] | 0.881 [0.837, 0.921] | 0.706 [0.615, 0.806] | 0.059 [0.036, 0.122] | 0.824 [0.666, 0.944] | 0.818 [0.545, 1.000] |
| detector_no_nuisance | 3 folds | 0.508 [0.444, 0.567] | 0.869 [0.825, 0.909] | 0.655 [0.563, 0.750] | 0.106 [0.061, 0.169] | 0.800 [0.618, 0.944] | 0.727 [0.455, 1.000] |
| detector_sim2real | 5 folds | 0.278 [0.226, 0.333] | 0.595 [0.536, 0.655] | 1.210 [1.091, 1.329] | 0.406 [0.351, 0.464] | 0.655 [0.476, 0.826] | 1.000 [1.000, 1.000] |

## sturm_test

| run | ensemble | acc_dry_vs_flooded (n=337) | flooded_auc (n=337) | false_alarm (n=88) | flood_recall (n=249) | kendall_tau_flooded (n=249) |
|---|---|---|---|---|---|---|
| detector_base | 5 folds | 0.783 [0.739, 0.825] | 0.822 [0.769, 0.870] | 0.830 [0.750, 0.898] | 1.000 [1.000, 1.000] | 0.405 [0.325, 0.486] |
| detector_base | folds 0-2 | 0.777 [0.733, 0.822] | 0.824 [0.772, 0.872] | 0.852 [0.773, 0.920] | 1.000 [1.000, 1.000] | 0.387 [0.301, 0.470] |
| detector_no_fisheye | 3 folds | 0.780 [0.736, 0.822] | 0.797 [0.740, 0.849] | 0.807 [0.716, 0.886] | 0.988 [0.972, 1.000] | 0.364 [0.280, 0.447] |
| detector_no_nuisance | 3 folds | 0.769 [0.724, 0.813] | 0.790 [0.732, 0.845] | 0.886 [0.818, 0.955] | 1.000 [1.000, 1.000] | 0.351 [0.265, 0.441] |
| detector_sim2real | 5 folds | 0.736 [0.688, 0.783] | 0.613 [0.543, 0.685] | 0.989 [0.966, 1.000] | 0.992 [0.980, 1.000] | 0.016 [-0.091, 0.117] |

## alley_test

| run | ensemble | acc_dry_vs_flooded (n=229) | flooded_auc (n=229) | false_alarm (n=108) | flood_recall (n=121) |
|---|---|---|---|---|---|
| detector_base | 5 folds | 0.930 [0.895, 0.961] | 0.984 [0.971, 0.994] | 0.074 [0.028, 0.130] | 0.934 [0.884, 0.975] |
| detector_base | folds 0-2 | 0.917 [0.882, 0.952] | 0.983 [0.969, 0.993] | 0.083 [0.037, 0.139] | 0.917 [0.860, 0.959] |
| detector_no_fisheye | 3 folds | 0.934 [0.900, 0.965] | 0.985 [0.970, 0.995] | 0.056 [0.019, 0.102] | 0.926 [0.876, 0.967] |
| detector_no_nuisance | 3 folds | 0.930 [0.895, 0.961] | 0.982 [0.968, 0.993] | 0.102 [0.046, 0.157] | 0.959 [0.917, 0.992] |
| detector_sim2real | 5 folds | 0.541 [0.476, 0.607] | 0.858 [0.806, 0.904] | 0.972 [0.935, 1.000] | 1.000 [1.000, 1.000] |

## fe8k_test

| run | ensemble | false_alarm (n=200) |
|---|---|---|
| detector_base | 5 folds | 0.070 [0.035, 0.105] |
| detector_base | folds 0-2 | 0.035 [0.010, 0.065] |
| detector_no_fisheye | 3 folds | 0.010 [0.000, 0.025] |
| detector_no_nuisance | 3 folds | 0.010 [0.000, 0.025] |
| detector_sim2real | 5 folds | 0.065 [0.030, 0.100] |

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
| detector_no_fisheye (3 folds) - detector_base (folds 0-2) | deepflood_test | acc | +0.012 [-0.036, +0.060] |
| detector_no_fisheye (3 folds) - detector_base (folds 0-2) | deepflood_test | mae | -0.016 [-0.075, +0.044] |
| detector_no_fisheye (3 folds) - detector_base (folds 0-2) | deepflood_test_fisheye | acc | -0.032 [-0.083, +0.024] |
| detector_no_fisheye (3 folds) - detector_base (folds 0-2) | deepflood_test_fisheye | mae | +0.060 [-0.004, +0.123] |
| detector_no_fisheye (3 folds) - detector_base (folds 0-2) | sturm_test | flooded_auc | -0.027 [-0.049, -0.005] |
| detector_no_fisheye (3 folds) - detector_base (folds 0-2) | alley_test | flooded_auc | +0.002 [-0.004, +0.008] |
| detector_no_fisheye (3 folds) - detector_base (folds 0-2) | alley_test | false_alarm | -0.028 [-0.065, +0.000] |
| detector_no_fisheye (3 folds) - detector_base (folds 0-2) | fe8k_test | false_alarm | -0.025 [-0.050, -0.005] |
| detector_no_nuisance (3 folds) - detector_base (folds 0-2) | deepflood_test | acc | -0.004 [-0.052, +0.044] |
| detector_no_nuisance (3 folds) - detector_base (folds 0-2) | deepflood_test | mae | +0.012 [-0.048, +0.075] |
| detector_no_nuisance (3 folds) - detector_base (folds 0-2) | deepflood_test_fisheye | acc | +0.020 [-0.032, +0.068] |
| detector_no_nuisance (3 folds) - detector_base (folds 0-2) | deepflood_test_fisheye | mae | +0.008 [-0.052, +0.067] |
| detector_no_nuisance (3 folds) - detector_base (folds 0-2) | sturm_test | flooded_auc | -0.034 [-0.059, -0.008] |
| detector_no_nuisance (3 folds) - detector_base (folds 0-2) | alley_test | flooded_auc | -0.001 [-0.007, +0.006] |
| detector_no_nuisance (3 folds) - detector_base (folds 0-2) | alley_test | false_alarm | +0.019 [-0.028, +0.074] |
| detector_no_nuisance (3 folds) - detector_base (folds 0-2) | fe8k_test | false_alarm | -0.025 [-0.050, -0.005] |
| detector_sim2real (5 folds) - detector_base (5 folds) | deepflood_test | acc | -0.242 [-0.321, -0.167] |
| detector_sim2real (5 folds) - detector_base (5 folds) | deepflood_test | mae | +0.528 [+0.385, +0.671] |
| detector_sim2real (5 folds) - detector_base (5 folds) | deepflood_test_fisheye | acc | -0.198 [-0.278, -0.119] |
| detector_sim2real (5 folds) - detector_base (5 folds) | deepflood_test_fisheye | mae | +0.556 [+0.409, +0.698] |
| detector_sim2real (5 folds) - detector_base (5 folds) | sturm_test | flooded_auc | -0.209 [-0.267, -0.146] |
| detector_sim2real (5 folds) - detector_base (5 folds) | alley_test | flooded_auc | -0.126 [-0.176, -0.082] |
| detector_sim2real (5 folds) - detector_base (5 folds) | alley_test | false_alarm | +0.898 [+0.833, +0.954] |
| detector_sim2real (5 folds) - detector_base (5 folds) | fe8k_test | false_alarm | -0.005 [-0.045, +0.035] |

## Belief banks (ensemble/belief_bank.npz)

| run | beliefs | per level 0-4 | sources | exact acc | level-0 predicted flooded | flooded predicted 0 |
|---|---|---|---|---|---|---|
| detector_base | 560 | [319, 33, 71, 50, 87] | alley_test 108, deepflood_test 252, fe8k_test 200 | 0.739 | 0.088 | 0.004 |
| detector_no_fisheye | 560 | [319, 33, 71, 50, 87] | alley_test 108, deepflood_test 252, fe8k_test 200 | 0.761 | 0.050 | 0.000 |
| detector_no_nuisance | 560 | [319, 33, 71, 50, 87] | alley_test 108, deepflood_test 252, fe8k_test 200 | 0.745 | 0.063 | 0.000 |
| detector_sim2real | 560 | [319, 33, 71, 50, 87] | alley_test 108, deepflood_test 252, fe8k_test 200 | 0.459 | 0.404 | 0.004 |

## Notes

- Paired CIs cover test-image sampling only, not training randomness. Across the five base folds the
  FishEye8K false-alarm rate ranges 0.010-0.145 and STURM AUC 0.761-0.820,
  wider than every ablation difference; 16 ablation comparisons, no multiplicity correction. Do not
  claim an ablation effect from these runs.
- detector_sim2real: validation folds contain only dry FishEye8K frames (all level 0), so early
  stopping and temperature scaling see dry images only; temperature = [0.2] (clamp 0.2)
  in every fold. Its ECE and belief sharpness are not meaningful. Predicted levels of a single fold do
  not depend on the temperature, so the accuracy/AUC gap is not a calibration artefact, but the epoch
  choice is biased towards dry predictions.
- DeepFlood false-alarm rate and flooded AUC rest on 11 dry test images.
- STURM dry-vs-flooded accuracy should be read against the all-flooded baseline 249/337 = 0.739;
  flood recall is 1.000, so the dry crops are mostly called flooded. AUC and Kendall
  tau are the informative transfer numbers.

Files: metrics.csv (all metrics with CIs), confusion_deepflood_test.csv (every run),
paired_differences.csv, folds.csv (epochs, best epoch, temperature), per_fold_metrics.csv,
belief_banks.csv, runs_cost.csv (Modal app times and billed cost per resource).
