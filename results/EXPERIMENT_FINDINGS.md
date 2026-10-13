# Simulator experiments: findings (step 26), 2026-10-05

**Rule:** the simplified paper rule (v2 = own-centred trimming F = 2,
confidence-weighted median of ordinal levels, alpha = 1.5, no filter, no
trust). This is the rerun after the 5 Oct simplification; the earlier runs
with filter and trust are in `results/_old_filter/`.

Final detector belief bank (`detector_base`, 560 beliefs). 18-camera layout
unless stated. Honest cameras only. Numbers are means over seeds (CIs in
each `results/exp*/summary.csv`). f = largest number of Byzantine
neighbours of an honest camera (theory bound for v2: f <= 2).

## 1. Main result (exp1, 50 seeds)

| A1 (confident lies) | n_byz=0 (f=0) | 2 (f=2) | 4 (f=2) | 6 (f=3) |
|---|---|---|---|---|
| **local only** acc / false alarm | 0.64 / 0.08 | 0.64 / 0.08 | 0.64 / 0.08 | 0.65 / 0.08 |
| **v2 (ours)** acc / FA / range viol. | 0.72 / 0.09 / 0 | 0.70 / 0.10 / 0 | 0.68 / 0.10 / 0 | 0.64 / 0.11 / 0.006 |
| v1 (published) acc / FA / range viol. | 0.81 / 0.27 / 0.004 | 0.81 / 0.27 / 0.007 | 0.71 / 0.28 / 0.15 | 0.62 / 0.34 / 0.27 |
| W-MSR F=2 acc / FA / range viol. | 0.73 / 0.36 / 0 | 0.71 / 0.36 / 0 | 0.67 / 0.36 / 0 | 0.59 / 0.42 / 0.07 |
| coord. median acc / FA | 0.75 / 0.37 | 0.75 / 0.36 | 0.74 / 0.35 | 0.68 / 0.40 |
| unfiltered mean acc / FA | 0.80 / 0.34 | 0.71 / 0.36 | 0.53 / 0.41 | 0.37 / 0.48 |
| oracle (knows attackers) acc / FA | 0.77 / 0.32 | 0.77 / 0.31 | 0.76 / 0.31 | 0.75 / 0.31 |

- **v2 is the only fusion rule that keeps false alarms near the
  single-camera level** (9-11 % vs 8 %); every other rule, including the
  oracle, reaches 27-48 % (flood-edge blurring).
- **Zero range violations within the bound**: 0 for every adversary at
  n_byz <= 4 (f <= 2), 50 seeds; 0.6 % at f = 3 (beyond the bound). v1:
  15-27 % at 4-6 attackers.
- **Level error** (MAE): local 0.45 -> v2 0.31 with no attack; v2 0.44 at
  the worst attack (A1, f = 3) vs v1 1.11 and unfiltered mean 1.25.
- **Accuracy is the price**: v2 +8 points over local. v1, the mean and the
  oracle gain more without attack (they average soft beliefs; v1 also
  smooths over time) but triple false alarms and collapse under attack.
- Detection delay: v2 alarms about 0.2 rounds later than a single camera;
  no reached camera is missed up to f = 2.

## 2. Detector quality (exp2)
Swap 0-0.5 (weaker detectors): v2 gain over local +5 to +8 points at every
quality; v2 false alarms stay within 1 point of local's or below.

## 3. Scale (exp3, 50-400 cameras, 0-15 % Byzantine)
- v2 gain over local: +6 to +8 points from 50 to 400 cameras, at every
  Byzantine fraction. Range violations <= 0.3 % even at 15 % Byzantine
  (f reaches 3-4, beyond the bound); v1 up to 7 %, W-MSR up to 2.6 %.
- Cost per decision is constant in network size: v2 0.16 ms (k = 4) /
  0.20 ms (k = 6) on one CPU core; v1 1.1 / 1.9 ms. Messages per round
  grow linearly (2|E|): 247 at 50 cameras, 1,956 at 400 (k = 4).

## 4. Crashes and message loss (exp4)
Up to 4 crashed cameras and 30 % loss with 2 attackers: v2 accuracy drops
by at most 7 points (0.72 -> 0.65); false alarms stay 5-8 %.

## 5. Ablation (exp5)
- **Trimming F = 2 is what keeps false alarms down**: no trimming 0.31,
  F = 1 0.31, F = 2 0.07. It costs 5 points of accuracy at f = 0 and cuts
  violations beyond the bound (f = 3: 8 % -> 0.6 %).
- **Confidence weights are needed**: without them false alarms 0.32.
- **Adding back the Hellinger filter, trust, or the symmetric-KL filter
  changes nothing measurable** (within 0.02 everywhere): justifies the
  simplification.
- Own weight alpha: 1.0 -> accuracy 0.73, false alarms 0.32; 3.0 -> 0.67 /
  0.05 (almost local). 1.5 is the balance.
- Mixture output: +4 points accuracy, false alarms 0.05, but 3.5-3.8 %
  range violations at f = 3 (vs 0.6 %). Paper keeps the median.

## 6. Flood speed and attacker placement (exp6)
Speeds 0.5-2 hops/round, attackers next to the flood (bfs) or random:
v2 range violations <= 0.14 %. **Weak spot:** with random placement the
one-level-up mimic (A2) raises v2 false alarms to 15-25 % (local 6-12 %),
because a dry camera next to two mimics and a flooded honest neighbour is
pushed to the top of its honest range. This is allowed by the theorem
(bounded distortion); the oracle is as high or higher (20-38 %). Report it.

## 7. Multi-hop (exp7, experimental)
T = 2/3 hops: v2 accuracy 0.72 -> 0.77 -> 0.79 but false alarms
0.07 -> 0.32 -> 0.31 and 2-3x messages. Supports the one-hop design.

## 8. Neighbours per camera (exp8, answers v1 reviewer R2.d)
18-camera layout, each camera linked to its k nearest cameras, k = 2-17
(17 = complete graph). The flood and the attackers are fixed on the
physical 4-NN layout (`scenario.reference_k: 4`); only the links change.
30 seeds, A1/A2, 0-6 attackers.
- **More links raise false alarms 5x**: v2 7 % at k <= 4, 26 % at k = 6,
  33-34 % at k >= 8 (complete graph 33 %). Far cameras see other water
  levels and pull dry cameras up; the oracle and W-MSR show the same.
- More links raise accuracy without attack (v2 0.72 at k = 4 -> 0.82 at
  k = 17) because flooded cameras get more flooded votes; the price is the
  false alarms above.
- **More links expose every camera to every attacker**: at k = 17 an honest
  camera faces all 4-6 attackers (f = 4-6, beyond any trimming bound). With
  6 attackers v2 accuracy falls below the single camera at k = 8-12 (0.55-
  0.59) and W-MSR collapses to 0.04-0.10 at k >= 8 (false alarms 92-100 %).
  Local links bound each camera's exposure (f <= 3 at k = 4).
- Too few links (k = 2-3): almost no gain over a single camera.
- Messages per round: 86 at k = 4, 306 for the complete graph.
- **Answer to R2:** k = 4 is the largest degree at which false alarms stay at
  the single-camera level and exposure stays within the bound; a complete
  graph is not "more reliable" for per-location estimates.

## Notes for figures
- `local` shows messages_per_round = 86 because the counter counts links;
  local sends nothing. Plot it as 0.
- All runs use `detector_base`; never mix with preliminary or smoke banks.

## Worst case, matched false alarms, pond (Oct 2026)
Branch `exp-worstcase` (12 Oct 2026). Configs exp9_worstcase, exp10_tradeoff,
exp11_pond; 50 seeds each, 18-camera layout, k = 4, `detector_base`. Paper
rule unchanged; the paper is not edited.

### Worst-case adversary AW (exp9)
For every honest camera, round and method separately, the Byzantine
neighbours send the messages that maximise the error of the camera output
(then: false alarm if dry, miss if wet), knowing the rule, the truth and the
honest messages. Candidates per Byzantine neighbour: full confidence at each
level (one_hot), low confidence at each level (peaked_belief(l, 0.2), whose
ordinal medians are 1, 1, 2, 3, 3), silence. On this layout an honest camera
has at most 3 Byzantine neighbours, so the search is exhaustive in every
decision (search_exact = 1.0; coordinate ascent never runs). Exact for the
stateless rules; a lower bound on the damage for bf_v1.

| AW: acc / FA / range viol. (%) | n=0 (f=0) | 1 (f=1) | 2 (f=2) | 4 (f=2) | 6 (f=3) |
|---|---|---|---|---|---|
| **v2 (ours)** | 0.72 / 0.09 / 0 | 0.69 / 0.09 / 0 | 0.67 / 0.10 / 0 | 0.63 / 0.11 / 0 | 0.60 / 0.12 / 0.65 |
| W-MSR F=2 | 0.73 / 0.36 / 0 | 0.70 / 0.36 / 0 | 0.67 / 0.36 / 0 | 0.62 / 0.37 / 0 | 0.54 / 0.44 / 7.4 |
| W-MSR F=1 | 0.75 / 0.39 / 0 | 0.72 / 0.39 / 0 | 0.65 / 0.42 / 0.8 | 0.47 / 0.57 / 15.6 | 0.26 / 0.73 / 30.6 |
| v1 (published) | 0.81 / 0.27 / 0.4 | 0.80 / 0.27 / 0.6 | 0.78 / 0.27 / 1.5 | 0.64 / 0.30 / 18.8 | 0.52 / 0.36 / 33.0 |
| coord. median | 0.75 / 0.37 / 0 | 0.73 / 0.37 / 0 | 0.69 / 0.37 / 0.1 | 0.58 / 0.40 / 3.6 | 0.46 / 0.46 / 12.3 |
| majority | 0.75 / 0.32 / 0 | 0.73 / 0.32 / 0 | 0.70 / 0.34 / 0 | 0.58 / 0.37 / 5.2 | 0.47 / 0.44 / 14.8 |
| unfiltered mean | 0.80 / 0.34 / 0 | 0.76 / 0.34 / 0.1 | 0.69 / 0.36 / 0.6 | 0.50 / 0.41 / 20.5 | 0.35 / 0.48 / 35.3 |

- **Theorem 2 holds against the searched worst case**: v2 range violations
  are 0 at every n_byz <= 5 (f <= 2), 0.65 % at f = 3. W-MSR F = 2 also
  keeps 0 at f <= 2 (its own guarantee) but 7.4 % at f = 3.
- AW is stronger than every fixed adversary of exp1 for every method:
  v2 loses 2-5 more accuracy points than under A1; majority and coordinate
  median 11-17 points more at 4-6 Byzantine; v1 10-11 points more at 5-6.
- At f = 3 v2 keeps the highest accuracy (0.60; next W-MSR F=2 0.54) and
  the lowest false alarm rate (0.12; others 0.36-0.73).
- AW maximises the level error, so at a camera with true level 1 it pushes
  the output up (an alarm): delays fall (v2 0.19 -> 0.05 rounds at n = 6).
  AW is not a delay attack.

### Matched false alarms (exp10)
Each rule also runs with an alarm threshold on its own alarm score
(rules.py): output 0 if score < tau, else max(1, y), tau = 0.05-0.95. Full
tables: `results/exp10_tradeoff/TRADEOFF.md`; figure `fig_tradeoff`.

| Accuracy at the FA of the normal v2 output | no attack (FA 0.093) | A1, 2 Byz. (0.096) | AW, 2 Byz. (0.099) |
|---|---|---|---|
| **v2 (ours)**, normal output | 0.721 (delay 0.19) | 0.696 (0.18) | 0.672 (0.18) |
| v1 (published) | 0.795 (delay 1.72) | 0.787 (1.53) | 0.765 (1.40) |
| unfiltered mean | 0.794 (0.44) | 0.691 (0.35) | 0.677 (0.36) |
| coordinate median | 0.749 (1.05) | 0.734 (0.98) | 0.687 (0.95) |
| majority | 0.754 (0.50) | 0.595 (0.50) | 0.579 (0.44) |
| W-MSR F=2 | 0.716 (0.30) | 0.702 (0.23) | 0.672 (0.23) |
| single camera | 0.646 (0.01) | 0.647 (0.00) | 0.647 (0.00) |
| oracle (not deployable) | 0.775 (0.33) | 0.772 (0.24) | 0.772 (0.24) |

- **v2 is not the only rule that can keep false alarms low**: a dry gate on
  the alarm score of any rule cuts its false alarms to the v2 level or
  below. In every panel some (method, tau) beat the normal v2 output on both
  accuracy and false alarm rate (no attack 17 pairs, A1 14, AW 10), e.g.
  without attack the unfiltered mean at tau = 0.75: 0.79 / 0.03 vs 0.72 /
  0.09.
- **None of them also matches the v2 delay**: every dominating point alarms
  later (0.25-3.8 rounds vs 0.18-0.19). Under AW the dominating deployable
  points are v1 at tau 0.9-0.95 (+5 to +8 points, delay 2.5-3.8 rounds) and
  the coordinate median at tau 0.85 (+1.2 points, delay 1.27); the rest is
  the oracle.
- v2 thresholded itself: tau = 0.7 gives FA 0.02 at accuracy 0.719 / 0.698 /
  0.675 (no attack / A1 / AW) and delay 0.23-0.32.
- The false alarm rate has wide CIs (+-0.03-0.04): its denominator is the
  few dry rounds at the flood edge.
- Paper protocol (decision 11, Table V `tab_threshold`): tau tuned without
  attack (max accuracy s.t. FA <= 0.093), kept under AW. Extra run, not in
  the paper, AW with **4** Byzantine at the same tau (acc / FA / delay /
  viol. %): v2 0.634 / 0.111 / 0.10 / 0; W-MSR F=2 (tau 0.35) 0.628 / 0.019
  / 0.14 / 1.0; coord. median (0.85) 0.575 / 0.051 / 0.99 / 9.7; FTM (0.90)
  0.569 / 0.027 / 1.91 / 32.0; unfiltered (0.75) 0.492 / 0.034 / 0.30 /
  27.5; majority (0.85) 0.333 / 0.007 / 0.58 / 40.0; single camera 0.644 /
  0.083; oracle (0.65) 0.765 / 0.054. Full rows in TRADEOFF.md.

### Pond, a steep gradient (exp11)
One pond camera (degree >= 4, not the epicentre, drawn from the seed, never
faulty). Isolated: the city stays dry, the pond has level 3 (or 2) from t0.
Inside: the flood spreads as in exp1 and the pond is 2 levels above it
(level 3) once the flood arrives. acc_pond / alarm_pond over the rounds in
which the pond holds water.

| Level 3: acc_pond / alarm_pond | isolated, no attack | isolated, AW | inside, no attack | inside, AW |
|---|---|---|---|---|
| **v2 (ours)**, alpha 1.5 | 0.005 / 0.22 | 0.005 / 0.19 | 0.73 / 1.00 | 0.66 / 0.99 |
| v2, alpha 3 | 0.16 / 0.62 | 0.16 / 0.60 | 0.69 / 1.00 | 0.67 / 0.99 |
| unfiltered mean | 0.00 / 0.01 | 0.00 / 0.01 | 0.77 / 0.99 | 0.65 / 0.99 |
| single camera | 0.29 / 1.00 | 0.29 / 1.00 | 0.67 / 1.00 | 0.67 / 1.00 |

- **An isolated pond is reported dry by v2** in 78-81 % of its rounds, as
  expected: with d_v dry neighbours, trimming drops 2 and the d_v - 2 left
  outweigh 1.5 (1 + gamma_v). The unfiltered mean, coordinate median, v1
  and the oracle almost never alarm (alarm_pond <= 0.032); W-MSR alarms (0.83) but only at
  level 1 (acc 0.00).
- **alpha = 3 keeps the pond only at degree 4**: alarm_pond 0.85 at degree 4
  (30 of 50 seeds), 0.32 / 0.28 / 0.16 at degree 5 / 6 / 7, where 3-5 dry
  levels survive trimming and their weight exceeds 3 (1 + gamma_v). Its
  accuracy (0.16) is capped by the detector (single camera 0.29 at level 3).
  It raises the false alarm rate elsewhere in the dry city from 0.024 to
  0.058.
- Inside the flood every rule alarms (>= 0.97). The flood reaches level 4
  within about 4 rounds of arrival, so most pond rounds have pond and
  neighbours at 3-4: this mode mostly measures the saturated flood.
- Range violations stay 0 for v2 in every pond case. Level 2 gives the same
  picture (isolated: v2 alarm 0.21, alpha 3 0.60).

## Threshold transfer and correlated errors (Oct 2026)
Branch `exp-transfer-corr` (13 Oct 2026). New metrics: shallow_recall (alarm
at honest cameras of true level 1) and acc_level1. Full tables:
`results/exp12_transfer/{TRANSFER,VALIDITY}.md`, `results/exp13_correlated/CORRELATED.md`.

### Validity of thresholded rules (VALIDITY.md)
Degree-4 camera, own and 4 honest neighbours all at true level c, beliefs
from the bank among those the detector gets right. At the exp10 thresholds:
- **Every thresholded rule except majority can output 0 (no alarm) for a
  unanimous honest level-1 neighbourhood**: W-MSR (0.35) always (evidence
  1/4); coordinate median (0.85) and FTM (0.90) always; unfiltered mean
  (0.75) in 92 % of random draws; oracle (0.65) in 13 %. FTM also misses
  unanimous level 2 in 94 % of draws, coordinate median in 15 %.
- ByzantineFlood outputs c for every such neighbourhood (weighted median).

### Threshold transfer (exp12, 50 seeds; 20 for 100 cameras)
- Base reproduces Table IV (re-tuned tau = fixed tau for every rule).
- **The tuned thresholds move**: re-tuned tau ranges FTM 0.55-0.90,
  unfiltered 0.60-0.75, majority 0.70-0.85, coordinate median 0.75-0.85,
  W-MSR 0.30-0.50, oracle 0.50-0.70 across the seven settings; the largest
  shifts at 100 cameras and with a degraded detector.
- **W-MSR with its fixed tau 0.35 stays close to v2 under AW** (accuracy -1.3
  to +0.3 points in every setting) at lower false alarms (e.g. base 0.020
  vs 0.099). Its price: shallow recall without attack 0.52-0.78 vs v2
  0.65-0.89, delay 0.14-0.54 vs 0.05-0.34 rounds, range violations 0.1-0.8
  % vs 0 under AW.
- FTM at 0.90: shallow recall 0.03-0.11, delay 2.3-2.9 rounds, range
  violations 12-14 % under AW. Unfiltered mean at 0.75 under AW: accuracy
  0.49-0.78, range violations 2-17 %.
- A degraded detector raises every false alarm rate: single camera 0.08 ->
  0.29 -> 0.46 (swap 0.2, 0.4); v2 follows it (0.09 -> 0.26 -> 0.44); the
  thresholded rules stay lower (W-MSR 0.18, 0.34; FTM 0.03, 0.05) at lower
  shallow recall. k = 6 raises v2 false alarms to 0.29 (as exp8).

### Correlated detector errors (exp13, 50 seeds)
- Adverse bank (infrared night + rain + lens droplets at full strength):
  accuracy 0.739 -> 0.479, dry false alarms 0.088 -> 0.367, level 4 0.724 ->
  0.046 (ADVERSE_BANK.md).
- **Theorem 2 holds**: v2 range violations 0 in every case, all 9 cases with
  AW at f = 2 included.
- v2 gain over the single camera, no attack: independent +0.078; spatial
  +0.068 to +0.076; stuck faults +0.074 (q 0.05), +0.059 (q 0.10); adverse
  block +0.055 (a 0.25), +0.021 (a 0.5). With AW: independent +0.029,
  spatial p 0.3 h 2 +0.024, stuck q 0.10 +0.003, adverse a 0.5 -0.007.
- v2 false alarms stay within 5 points of the single camera in every case
  without attack (others 0.23-0.57).
- FTM keeps the largest accuracy gains under correlated errors (+0.10 to
  +0.27; temporal smoothing averages out stuck and patch errors) at false
  alarms 0.23-0.32 and the range violations of exp9 under attack.
