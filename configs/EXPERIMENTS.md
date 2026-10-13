# Simulator experiments (step 25)

All configs read the real belief bank of a detector run
(`detector: {type: belief_bank, bank: detector_base}`), looked up at
`results/detector/<run>/ensemble/belief_bank.npz` (Modal) or
`results/<run>/ensemble/belief_bank.npz` (local). Run them only after the
final detector (`detector_base`, human-checked Grok labels) exists.

| Config | Question it answers | Feeds | Runs x methods | Est. time (Modal, 8 CPU) |
|---|---|---|---|---|
| exp1_main | Accuracy, false alarms, range violations, delay vs number of Byzantine cameras, every adversary, all baselines + oracle | Main table, Fig. accuracy-vs-f | 2,450 x 11 | ~10 min |
| exp2_rho | When does fusion help? Detector quality lowered by swapping beliefs (0-0.5) | Fig. gain-vs-detector-accuracy | 1,080 x 5 | ~5 min |
| exp3_scale | Scale 50-400 cameras, k = 4/6, 0-15 % Byzantine; messages and CPU time per decision | Table scale/cost | 1,920 x 6 | ~45 min |
| exp4_faults | Crashes (0-4) and message loss (0-30 %) with 0/2 Byzantine | Fig. benign faults | 720 x 5 | ~3 min |
| exp5_ablation | Which v2 component matters (filter, trust, confidence, trimming, alpha, distance, output) | Ablation table | 480 x 13 | ~6 min |
| exp6_scenario | Flood speed 0.5-2 hops/round; Byzantine next to flood vs random | Sensitivity figure | 1,620 x 5 | ~6 min |
| exp8_degree | Neighbours per camera k = 2-17 (complete graph); flood fixed on the 4-NN layout | Fig. degree, reply to R2 | 1,680 x 6 | ~3 min |
| exp7_hops | T = 1-3 hops (experimental only; theory is one hop) | Multi-hop paragraph | 270 x 7 | ~2 min |
| exp9_worstcase | Worst-case adversary AW (main adversary, decision 10): per honest camera, round and method, the Byzantine neighbours send the messages (11 candidates incl. silence) that maximise its error; 0-6 Byzantine | Main table, Fig. main; Theorem 2 check | 350 x 7 | 8.6 min |
| exp10_tradeoff | Matched false alarms: alarm threshold tau = 0.05-0.95 on each rule's own alarm score; no attack, A1 and AW with 2 Byzantine (+ AW with 4, not in the paper) | Table threshold, Fig. S4 tradeoff, `results/exp10_tradeoff/TRADEOFF.md` | 200 x 8 (x 20 outputs) | 3.1 min |
| exp11_pond | Steep gradient: one pond camera (level 2/3) isolated in a dry city or 2 levels above the flood; no attack, A1, AW | Table S4 pond | 600 x 8 | 4.7 min |
| exp12_transfer | Threshold transfer: each rule keeps its exp10 tau (Table IV) in 7 settings (base, detector swap 0.2/0.4, k = 6, flood speed 0.5/2, random layout of 100 cameras); full tau sweep to re-tune per setting; no attack and AW 2 (branch exp-transfer-corr) | tab_transfer, TRANSFER.md, VALIDITY.md | 640 x 8 (x 20 outputs) | 7.3 min |
| exp13_correlated | Correlated detector errors (bflood.errors): spatial patches p 0.1/0.3, h 1/2; stuck faults q 0.05/0.10; adverse condition block a 0.25/0.5 (adverse bank); no attack and AW 2 (branch exp-transfer-corr) | tab_correlated, CORRELATED.md | 900 x 8 | 8.8 min |

Commands (from the project folder; Git Bash: `export MSYS_NO_PATHCONV=1`):

    python scripts/run_experiment.py configs/exp1_main.yaml --bank detector_base --quick   # local check, 2 seeds
    python -m modal run --detach modal_app.py::main --config exp1_main.yaml --extra "--bank detector_base"
    python -m modal volume get bflood-results exp1_main results/

Flags: `--bank RUN` choose the detector run; `--quick` 2 seeds;
`--placeholder` synthetic detector (dry runs only, never reported).

Metrics (honest operating cameras only): acc, acc_flooded, acc_near,
false_alarm, mae, range_viol (Theorem 1 check: must be 0 for v2 when
f_v <= 2), spatial, corrected, induced; per run: delay (rounds from flood
arrival to first alarm), miss, decide_us (exp3 only), messages_per_round.
exp9-11 add: search_exact (AW only: fraction of searched decisions where the
search was exhaustive), acc_pond / alarm_pond (exp11: accuracy and alarm
rate at the pond camera while it holds water; nan without a pond),
pond_node / pond_degree, and a `tau` column (exp10: "none" = normal output).

Adversaries beyond A0-A8: `AW` worst case (searched in `bflood.sim.aw_search`,
exact for stateless rules, a lower bound on the damage for bf_v1 and trust
variants; skipped for local and oracle); `none` = no attack (n_byz 0, used
in `cases:` lists). Config keys: `cases` (list of {adversary, n_byz} instead
of the product), `sweep_threshold` (list of tau), `scenario.pond`
({node: random|int, level, mode: isolated|inside}).

exp12-13 add (Oct 2026): metrics shallow_recall (alarm at honest cameras of
true level 1) and acc_level1; config keys `settings` (named groups of
overrides, column `setting`, each may set its own `seeds`) and `errors`
(correlated detector errors, src/bflood/errors.py). The condition model
reads `belief_bank_adverse.npz`, built from the detector_base ensemble
under infrared night + rain streaks + lens droplets at full strength:
`python -m modal run modal_app.py::adverse --run detector_base` (5.4 min,
CPU; summary in results/detector_base/ensemble/ADVERSE_BANK.md). Validity
search for thresholded rules: `python scripts/validity_search.py`.

Methods: local (B0), unfiltered mean (B1), majority (B2), coordinate median
(B3), W-MSR F=1/2 (B4), ByzantineFlood v1 as published, ByzantineFlood v2
(paper rule), v2 with mixture output (`output: mixture`, candidate), oracle
(B5: knows the Byzantine set; upper reference, not deployable).

Dry run on the preliminary detector (fold 0, AI-drafted Grok labels,
NOT for the paper), 8 seeds, A1/A2/A5:
- v2 never violated the range bound; v1 violated it in 19-34 % of
  decisions at 4-6 Byzantine cameras.
- v1 has the highest raw accuracy at low f (0.74 vs v2 0.54, local 0.50),
  because it averages soft beliefs over time; v2 sells guarantees.
- v2 with mixture output: +4 points accuracy, -2 points false alarms,
  range still guaranteed (tests/test_experiments.py). Decide after the
  real bank whether the paper rule switches to it (theorems on distortion
  and correction quorum would need re-proof).
