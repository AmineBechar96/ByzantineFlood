# ByzantineFlood

Code and data for the paper

> **Spatial Fusion Resilient to Locally Bounded Byzantine Sensors for Flood
> Monitoring with Fisheye Camera Networks**
> (under review at *IEEE Transactions on Dependable and Secure Computing*).

Each camera of a flood monitoring network estimates the water level at its own
location from its own detector belief and the messages of its neighbours. The
rule (ByzantineFlood) trims up to F = 2 neighbour levels above and below the
level of the camera itself and outputs the weighted median of the ordinal
levels (own weight alpha = 1.5, weights from detector confidence). Its output
stays within the levels of the honest cameras around it whenever at most F of
its neighbours are Byzantine.

This repository holds the simulator, the detector code and configurations,
the labels and splits of every image used, the ratings of the two annotators,
the belief banks of the detector, and the results of every experiment.
**No image is redistributed**: the images are available from the authors of
each dataset under their licences (see [Data](#data)).

## Contents

| Path | What it holds |
|---|---|
| `src/bflood/` | Simulator: levels, graphs (18-camera layout, random layouts), flood scenario, beliefs, adversaries A0-A8 and the worst case AW, rules (ByzantineFlood, baselines B0-B4 incl. W-MSR), metrics, correlated detector errors |
| `src/detector/` | Flood level detector: manifest and leak checks, transforms (letterbox, fisheye warp, nuisance augmentations), ordinal head, training, report, adverse belief bank |
| `scripts/` | Entry points: data preparation, experiments, figures and tables, detector report |
| `configs/` | One YAML file per experiment (`exp1`-`exp13`) and per detector run; seeds are fixed in each file. `configs/EXPERIMENTS.md` describes every experiment |
| `modal_app.py` | Runs the experiments (CPU) and the detector training (GPU) on [Modal](https://modal.com) |
| `tests/` | Unit tests (58, about 25 s; none needs images) |
| `data/processed/` | Labels and splits (identifiers, our labels, our groups and splits only; see [Label files](#label-files)): `deepflood_labels.csv`, `sturm_labels.csv`, `alleyfloodnet_labels.csv`, and `manifest.csv` (every image the detector uses: dataset zip, member, label interval, role, test set) |
| `data/grok_labels.csv` | Levels of the 600 synthesised flood images (training only) |
| `data/grok_base_frames.csv` | Names of the 150 FishEye8K training frames that were edited |
| `data/human_check/` | Ratings of 120 synthesised images by two annotators (`rater_1.csv`, `rater_2.csv`) and the merged table (`merged.csv`) |
| `data/train.txt`, `val.txt`, `test.txt` | Official split lists of STURM-FloodDepth (used only to leave out crops that are not in the official lists) |
| `data/testbed/cameras.json` | Positions of the 18 cameras on a road network in Taipei (Xinyi District) |
| `results/detector_base/ensemble/` | Belief banks: `belief_bank.npz` (detector outputs on the test images, by true level; read by every experiment) and `belief_bank_adverse.npz` (same images under night, rain and lens droplets; exp13), with `ADVERSE_BANK.md` |
| `results/detector_final/`, `results/detector_final_sim2real/` | Detector metrics with confidence intervals (paper Table II) |
| `results/exp1_main/` ... `results/exp13_correlated/` | Results of every experiment: `runs.csv` (one row per run and method), `summary.csv`, `summary.md`, and reports (`TRADEOFF.md`, `TRANSFER.md`, `VALIDITY.md`, `CORRELATED.md`) |
| `results/EXPERIMENT_FINDINGS.md` | Findings of every experiment, in order |

## Installation

Python 3.10 or later.

```bash
git clone https://github.com/AmineBechar96/ByzantineFlood.git
cd ByzantineFlood
pip install -r requirements.txt
python scripts/run_tests.py          # or: pytest tests
```

The simulator needs only numpy, scipy, networkx, pyyaml, matplotlib and pandas.
The detector also needs PyTorch and OpenCV:

```bash
pip install torch==2.5.1 torchvision==0.20.1 opencv-python-headless pillow
```

`modal` is needed only to run on Modal (`pip install modal && modal setup`).

## Data

Nothing in this section is needed to run the experiments or to rebuild the
tables and figures: the experiments read the belief banks in
`results/detector_base/ensemble/`. The images are needed only to retrain the
detector. Put each dataset in `data/` as the zip file named below, without
extracting it (the code reads images straight from the zips).

| Dataset | Use | Licence | How to obtain | File in `data/` |
|---|---|---|---|---|
| DeepFlood (Chaudhary et al., *ISPRS J. Photogramm. Remote Sens.* 167, 2020) | Training, validation and the primary 5-level test set | Research use, granted by the authors | Request access from the dataset authors (P. Chaudhary) | `Flood_Dataset.zip` |
| STURM-FloodDepth, flooded cars (Notarangelo et al., *Geomatica* 77(2), 2025) | 3-level transfer test only | CC BY 4.0 | Zenodo, doi:10.5281/zenodo.14833532 | `flooded_cars.zip` |
| AlleyFloodNet (Joo and Lee, *Electronics* 14(10), 2025) | Dry vs flooded (no depth labels) | CC BY 3.0 | Kaggle (AlleyFloodNet) | `AlleyFloodnet.zip` |
| FishEye8K (Gochoo et al., CVPR Workshops 2023) | Dry fisheye frames: training (14 cameras) and test (cameras 1, 2, 4, 7) | CC BY-NC-SA 4.0 | Hugging Face copy `Voxel51/fisheye8k` | `fisheye8k_sample.zip` |
| Synthesised floods (Grok edits of FishEye8K training frames) | Training only | see below | Not redistributed; see below | `grok/*.jpg` (loose files) |

Notes on each dataset:

- **DeepFlood.** Image level = median level of the submerged objects,
  mapped to the five levels of the paper (`scripts/prepare_deepflood.py`).
  `deepflood_labels.csv` (image IDs, levels, split) is published with the
  agreement of the dataset authors; the images are not.
- **STURM-FloodDepth.** The official split was drawn per crop, and crops of
  one photo fall on both sides of it; `group_split` keeps every crop of a
  photo in one subset (`scripts/prepare_sturm.py`).
- **AlleyFloodNet.** `split` is the official split with duplicates across
  train and test removed (`scripts/prepare_alleyfloodnet.py`).
- **FishEye8K.** 856 frames of 18 cameras. The zip holds
  `fisheye8k_sample/<name>.jpg`, where `<name>.jpg` is the frame
  `data/<name>.png` of `Voxel51/fisheye8k` saved as JPEG; the list of frames
  is the rows of `data/processed/manifest.csv` with `source == fisheye8k`.
  Test cameras 1, 2, 4, 7 are never edited or trained on.
- **Synthesised floods.** 600 edits (150 frames of the training cameras,
  levels 1-4) made with the xAI image model `grok-imagine-image`
  (`scripts/grok_flood_edit.py`, key from the environment variable
  `XAI_API_KEY`, USD 0.022 per edit). They are used for training only and
  never in a validation or test split. The edits are not redistributed and a
  new run of the image model gives different images, so `data/grok_labels.csv`
  documents the labels of the paper but cannot label new edits. To train
  without them, remove `grok` from `sources` in the detector config (the
  results then differ from the paper).

### Label files

The label files of the third-party datasets hold only identifiers, our
label, the groups we created and our split. No other column of the original
annotations is included.

| File | Columns |
|---|---|
| `deepflood_labels.csv` | `image_id`, `file`, `level5` (our level 0-4), `dup_group` (near-duplicate group), `split`, `cross_dup` (same photo in another dataset) |
| `sturm_labels.csv` | `file`, `photo` (source photo of the crop), `level3` (our level 0-2), `group_split`, `cross_dup` |
| `alleyfloodnet_labels.csv` | `file`, `class` (dry or flooded), `split`, `dup_group`, `cross_dup` |
| `grok_base_frames.csv` | `file` (FishEye8K frame name) |

Where the code needs a value of an original annotation, it reads it from the
dataset itself:

- `detector.manifest` sets the `ambiguous` flag of a DeepFlood image
  (submerged objects that span two or more of the 11 DeepFlood levels; the
  detector report gives the accuracy without these images) from the
  annotation file in `Flood_Dataset.zip`. Without the zip, it keeps the flags
  of the included `manifest.csv`.
- `detector.manifest` leaves out the STURM crops that are not in the official
  lists `data/train.txt`, `val.txt`, `test.txt`.
- The Grok scripts read the camera and the time of day of a FishEye8K frame
  from its name (`camera<c>_<A|E|M|N>_<index>.jpg`: afternoon, evening,
  morning, night).

The `prepare_*.py` scripts write these columns only; run on the original zips
they reproduce the included files.

### Manifest

The detector reads `data/processed/manifest.csv`, which is included and was
checked against the leak rules (no synthesised image in validation or test,
no test camera in training, no cross-dataset duplicate in a test set).
Rebuilding it (`PYTHONPATH=src python -m detector.manifest`) also needs the
frame table inside `fisheye8k_sample.zip`, which is not part of this release.

## Retraining the detector

EfficientNet-B0 with an ordinal (cumulative link) head, 384 px letterbox,
fisheye warp and nuisance augmentations, 5 folds grouped by camera,
temperature scaling on validation. Configs: `detector_base` (paper),
`detector_no_fisheye`, `detector_no_nuisance`, `detector_sim2real`
(ablations), `detector_smoke` (one fold, 64 images, one epoch).

On Modal (as in the paper; one L4 GPU):

```bash
# Git Bash on Windows: export MSYS_NO_PATHCONV=1 first
python -m modal volume create bflood-data
python -m modal volume put bflood-data data/Flood_Dataset.zip /Flood_Dataset.zip
python -m modal volume put bflood-data data/flooded_cars.zip /flooded_cars.zip
python -m modal volume put bflood-data data/AlleyFloodnet.zip /AlleyFloodnet.zip
python -m modal volume put bflood-data data/fisheye8k_sample.zip /fisheye8k_sample.zip
python -m modal volume put bflood-data data/processed /processed
python -m modal volume put bflood-data data/grok /grok
python -m modal run modal_app.py::train --config detector_smoke.yaml
python -m modal run --detach modal_app.py::train --config detector_base.yaml
python -m modal volume get bflood-results detector/detector_base results/
python -m modal run modal_app.py::adverse --run detector_base
```

`detector_base` took 50 min (about USD 1.1); the ablations 14-32 min each.
The adverse bank takes about 5 min on CPU. Training writes
`results/detector_base/` (fold models, predictions, `ensemble/belief_bank.npz`),
which replaces the released belief bank; GPU training is not bit-for-bit
deterministic, so a new bank differs slightly from the released one.

Locally (a GPU is strongly advised):

```bash
PYTHONPATH=src python -c "from detector.train import run; run('configs/detector_base.yaml', 'data', 'results')"
python scripts/make_adverse_bank.py --data-dir data --run-dir results/detector_base
```

`scripts/detector_final_report.py` rebuilds `results/detector_final/` from the
fold predictions of the four runs; its wall time and cost columns query the
Modal account that ran them.

## Reproducing the tables and figures

**From the released results** (about 15 s):

```bash
python scripts/make_figures.py --out out
```

This writes `out/figures/*.pdf` (with PNG previews) and `out/tables/*.tex`,
which are identical to the tables of the paper, and refreshes the reports
`TRADEOFF.md`, `TRANSFER.md` and `CORRELATED.md` in `results/`.
`python scripts/validity_search.py` writes `results/exp12_transfer/VALIDITY.md`.
Fig. 1 of the paper is a drawing and is not generated by code.

**Running the experiments again.** Each experiment reads
`results/detector_base/ensemble/belief_bank.npz` and writes
`results/<name>/` (`runs.csv`, `summary.csv`, `summary.md`), replacing the
released results of that experiment. `--quick` runs 2 seeds and writes
`results/<name>_quick/` instead.

```bash
python scripts/run_experiment.py configs/exp9_worstcase.yaml --bank detector_base --workers 8
python scripts/run_experiment.py configs/exp9_worstcase.yaml --bank detector_base --quick
```

On Modal (8 CPUs), after copying the belief banks to the results volume:

```bash
python -m modal volume put bflood-results results/detector_base/ensemble/belief_bank.npz /detector/detector_base/ensemble/belief_bank.npz
python -m modal volume put bflood-results results/detector_base/ensemble/belief_bank_adverse.npz /detector/detector_base/ensemble/belief_bank_adverse.npz
python -m modal run --detach modal_app.py::main --config exp9_worstcase.yaml --extra "--bank detector_base"
python -m modal volume get --force bflood-results exp9_worstcase results/
```

| Config | Paper output | Run time (8 CPUs) |
|---|---|---|
| `exp1_main` | Table S3 (seven fixed adversaries); single camera and oracle rows of Table III and Fig. 3 | 2.3 min |
| `exp2_rho` | Fig. S1 (weaker detectors) | 1.2 min |
| `exp3_scale` | Table V and Fig. S3 (50-400 cameras, cost) | 31.5 min |
| `exp4_faults` | Fig. S2 (crashes and message loss) | 0.3 min |
| `exp5_ablation` | Table S2 (ablation) | 0.7 min |
| `exp6_scenario` | Text (flood speed, placement of the Byzantine cameras) | 1.8 min |
| `exp7_hops` | Table S1 (T = 1-3 hops) | 0.3 min |
| `exp8_degree` | Fig. S5 (neighbours per camera) | 5.0 min |
| `exp9_worstcase` | Table III and Fig. 3 (worst case adversary AW) | 8.6 min |
| `exp10_tradeoff` | Table IV and Fig. S4 (matched false alarm rate) | 3.1 min |
| `exp11_pond` | Table S4 (isolated pond) | 4.7 min |
| `exp12_transfer` | Table S5 (threshold transfer) | 7.3 min |
| `exp13_correlated` | Table S6 (correlated detector errors; needs `belief_bank_adverse.npz`) | 8.8 min |
| (no config) | Fig. 2 (layout, `data/testbed/cameras.json`); Table II (`results/detector_final*`) | - |

The run times are those recorded in each `summary.md` (8 worker processes).
All thirteen experiments take about 76 min on 8 CPUs. Locally, a `--quick`
run of `exp9_worstcase` (14 runs, 7 methods) took 1.5 min on an 8-core
Windows PC.

## Licence

- **Code** (`src/`, `scripts/`, `tests/`, `configs/`, `modal_app.py`): MIT, see
  [`LICENSE`](LICENSE).
- **Data annotations** (labels, splits, ratings, belief banks and results in
  `data/` and `results/`): CC BY-NC-SA 4.0, see [`data/LICENSE`](data/LICENSE).
  The share-alike, non-commercial terms follow FishEye8K, from which the
  synthesised images and fisheye labels derive. File names and image IDs
  belong to the datasets above, and `data/train.txt`, `val.txt`, `test.txt`
  are the official STURM-FloodDepth split lists (CC BY 4.0).
- **Images**: none are included; each dataset keeps its own licence.

## Citation

The citation will be added when the paper is published.
