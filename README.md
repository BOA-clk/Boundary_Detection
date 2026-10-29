# Groupwise Selective State-Space Filtering for Accurate and Streaming Action Boundary Detection

Code accompanying the revised manuscript by Mustafa Bora Çelik. The repository
provides model, data-loading, training, decoding and evaluation **code**.
**Dataset videos, third-party pre-extracted I3D features, annotations, official
split files, and trained weights are not redistributed here.** Obtain data and
features separately from their respective providers under their terms.

The three studies below use distinct protocols. In particular, the ablation
reference **0.611 ± 0.046** is not the main-table 50Salads result **0.577 ± 0.047**.

## Model

2048-dimensional input → 32 fixed groups of 64 channels → within-group LayerNorm
→ independent forward Mamba blocks → concatenation → learned 2048→64 fusion
→ residual symmetric TCN → 64→64→1 SiLU MLP → minimum-gap DP/refinement.
Each Mamba block has state dimension 16, local convolution width 4, expand 1,
and additive bypass D=0, frozen. Multiplicative gating and output projection remain.
The offline TCN dilations are (1,11,23), (1,7,13), and (1,17,35) for Breakfast,
GTEA, and 50Salads. Gaussian target sigma is 0.8/0.3/1.0 seconds; weighted
context BCE uses weights 1+4y. The frozen legacy direct head remains in offline
checkpoints: 695,234 total parameters, 693,121 trainable for the full G=32 model.

The public offline model API delegates to the actual offline experiment model;
it does not substitute the earlier simplified training implementation.

## 1. Main comparison: Table I

Video-macro boundary F1, mean ± population SD across official folds × seeds 0/1:

| Method | Params | Breakfast F1@1 s | GTEA F1@0.4 s | 50Salads F1@1 s |
|---|---:|---:|---:|---:|
| MS-TCN | 0.80M | 0.349 ± 0.018 | 0.557 ± 0.045 | 0.370 ± 0.041 |
| ASFormer | 1.13M | — | 0.575 ± 0.062 | 0.455 ± 0.054 |
| BaFormer | 1.63M | 0.313 ± 0.029 | 0.606 ± 0.024 | 0.524 ± 0.049 |
| DiffAct | 1.21M | 0.182 ± 0.013 | 0.285 ± 0.037 | 0.312 ± 0.050 |
| ASRF | 1.30M | 0.374 ± 0.042 | 0.601 ± 0.036 | **0.594 ± 0.040** |
| Proposed adapter | 0.70M | **0.470 ± 0.021** | **0.627 ± 0.023** | 0.577 ± 0.047 |

ASFormer was not evaluated on Breakfast within the reproduction budget.
Numbers are the revised manuscript's reported experiments, not results generated
by a CPU demo or a new local smoke test. Exploratory Welch p-values in the
manuscript use partly rounded summaries and overlapping runs; they are not
presented here as independent-replication significance evidence.

### Shared training and evaluation settings

- Official folds: Breakfast 4, GTEA 4, 50Salads 5; training seeds 0 and 1.
- Adam, lr 5e-4, zero weight decay, constant learning rate; no warm-up,
  scheduler or gradient clipping; at most 50 epochs, patience 10.
- Deterministic 20% validation from each official training partition; split
  seed equals training seed. Test videos never enter tuning or checkpoint selection.
- Model-specific architectures, losses and native decoders are retained.
- Checkpoint: validation primary F1 at the first native decoder-grid candidate;
  Our adapter uses lambda=0. After checkpoint selection, tune decoder on validation;
  larger parameter wins ties. The adapter's lambda grid is 0,0.2,...,4; its fixed
  minimum gaps are 2/0.8/4 seconds (30/12/60 steps) for Breakfast/GTEA/50Salads.
- Calibration: median of nearest-prediction minus reference residual for every
  validation reference in videos with predictions, subtracted from test predictions.
  No **post-calibration** clipping and no additional grid snapping. The DP
  decoder clips uncalibrated feature timestamps before this offset subtraction.
- All three main-table datasets use **legacy_greedy** one-to-one matching:
  predictions in supplied order take the nearest unused reference within tolerance;
  equal distances favor the last reference encountered. This is not guaranteed
  to maximize cardinality. Empty predictions/GT yield zero F1 in this scorer.
- Tolerances: (0.5,1,2) s for Breakfast/50Salads; (0.2,0.4,0.8) s for GTEA.
  Average F1 across videos per run, then across runs; SD uses **ddof=0**.
- The proposed adapter's main-table training is **FP32**, without autocast.

The authoritative runner is `experiments/offline_main.py`. Its optional
`native_full_diagnostic` and `hungarian_same_predictions` are audits, not the
main-table primary result. Use `calibrated` / `tuned_calib` for Table I.

## 2. Component and design ablations: 50Salads

Separate AdamW study, five official folds × training seeds 0/1:

| Configuration | F1@1 s |
|---|---:|
| Full G=32, D=0, independently trained reference | 0.611 ± 0.046 |
| G=16, width 128 | 0.613 ± 0.046 |
| G=64, width 32 | 0.585 ± 0.054 |
| Learnable D | 0.574 ± 0.045 |
| State reset / no recurrent SSM memory | 0.558 ± 0.041 |
| Linear head | 0.603 ± 0.032 |
| Without input normalization | 0.593 ± 0.040 |
| Without learned fusion (mean pooling) | 0.578 ± 0.056 |
| Without TCN | 0.559 ± 0.044 |
| Time-varying group permutation | 0.433 ± 0.029 |
| Time-varying channel permutation | 0.171 ± 0.023 |

| Setting | Main table | Ablation |
|---|---|---|
| Optimizer / cap | Adam / 50 | AdamW / 60 |
| Weight decay | 0 | 0.01; parameters marked no-weight-decay are exempt |
| Schedule / clip | Constant / none | 10% linear warm-up, cosine / 1.0 |
| Validation split seed | Training seed | Fixed 0 |
| Checkpoint | Fixed lambda=0 | Retune lambda on validation each epoch |
| Calibration residuals | Nearest prediction for every GT | Tolerance-matched pairs |
| Post-calibration clipping | None | Clip to video duration |
| Matching | legacy_greedy | SciPy optimal assignment |
| SD | ddof=0 | ddof=1 |
| Precision policy | FP32 | Automatic BF16 when supported, otherwise FP32 |

Ablation assignment maximizes cardinality, then minimizes total temporal error.
For np predictions, ng references, tolerance tau, P=(min(np,ng)+1)(tau+1), the
FP64 cost matrix has ng rows and np+ng columns: valid real edges have absolute
time error <=tau+1e-9, invalid edges cost 2P, dummy edges cost P. It is solved
with `scipy.optimize.linear_sum_assignment`. Lambda ties prefer closeness to 2,
then the smaller lambda. The original effective BF16/FP32 choice is unavailable.

Ablation conclusions apply within this protocol. The full reference was trained
separately; comparisons are descriptive. The main/ablation score difference
cannot be attributed solely to Adam versus AdamW: partitioning, checkpointing,
calibration, precision and matching also change. Do not substitute the Adam/50
full score into this ablation table.

Code: `experiments/offline_ablation.py` for component variants;
`experiments/group_d_controls.py` for G=16/G=64/D controls;
`experiments/permutation_controls.py` for group/channel permutations.
The original source filename for the group/D runner mentioned A3/A4, but its
current run list contains only G16-Doff, G64-Doff and G32-Dlearn.
The historical state-reset runner was not recovered in the available sources;
its reported result is documented without inventing a new historical implementation.
Permutation seed 1234 is independent of training seed; the time-indexed table
is shared across train/validation/test and outputs are not inverse-permuted.
Only disabling import-time autorun was necessary in the component runner.

## 3. Stateful streaming study

A separately trained TCN-free model applies the boundary head to fused features.
It maintains convolution/SSM caches, Welford running statistics, and a bounded
three-score peak buffer. Neural look-ahead is zero; peak confirmation needs one
subsequent feature. It does not reuse an offline TCN checkpoint as a trained
streaming model.

| Group width | Groups | Params | Conv/SSM FP32 state | 50Salads F1@1 s |
|---:|---:|---:|---:|---:|
| 2048 | 1 | 13.359M | 160 KiB | 0.4687 ± 0.0735 |
| 64 | 32 | 0.658M | 160 KiB | 0.4818 ± 0.0653 |
| 32 | 64 | 0.453M | 160 KiB | 0.4736 ± 0.0460 |

Manuscript settings: AdamW lr 5e-4, wd .01, 10% warm-up/cosine to 1e-6,
clip 1, 60 epochs; validation DP F1@1 at lambda=1 selects the checkpoint.
Test inference uses 15-Hz features, running normalization and parabolic peak
refinement. Current defaults: peak threshold .5 and minimum separation 1 s.
Historical fold-specific peak thresholds are unavailable. Streaming evaluation
uses SciPy assignment of absolute-time costs followed by tolerance checking;
it is separate from both the main greedy and ablation dummy-augmented scorer.

The manuscript measured 50.5 steps/s and p95 20.45 ms/feature for width 64,
excluding I3D extraction and waiting for the next feature. These are historical
measurements, not guarantees for every device or for the Python fallback.
160 KiB counts Conv/SSM caches; fixed normalization/decoder buffers add overhead.
Emitted outputs can be consumed incrementally. Keeping all emitted timestamps
is optional and adds output storage proportional to the number of events.

`train_streaming.py` supplies a separate current reference trainer/evaluator,
with recorded FP32 and seed-dependent validation. It is not a recovered exact
historical streaming runner; historical fold-specific settings remain unavailable.
`demo_streaming.py` is explicitly an untrained synthetic development demo.

## Data setup: code is shared, third-party data is not

Obtain the feature release and annotations separately. No dataset download or
third-party feature redistribution is performed by this repository.

```text
/path/to/dataset/
  features/
    video_id.npy
  groundTruth/
    video_id.txt
  splits/
    train.split1.bundle
    test.split1.bundle
    ...
```

- One float feature array per video: [2048,Tsource] or [Tsource,2048]. The loader
  checks orientation and selects float32 [T,2048] at 15 Hz. No average pooling.
- One nonempty action label per annotation frame, in source order, in the .txt.
  Boundary GT uses every label transition regardless of semantic action name.
- Bundle lines identify videos (commonly video_id.txt); suffixes are removed.
  Supply all official folds. Train/test must be disjoint; every video is held
  out exactly once across test folds. Expected videos: 1712/28/50.
- Source annotation rates: Breakfast/GTEA 15 Hz, 50Salads 30 Hz. Paper input
  feature rates: 15/15/30 Hz, strides 1/1/2, giving 15-Hz inputs. The core loader
  checks the inferred rate against feature/label lengths. Main preflight also
  verifies compatibility with the baseline sampling; inconsistent lengths fail.
- Input time t/15 is distinct from a transition midpoint (i-.5)/source_hz.
  External decoder indices use (j-.5)*stride/source_hz. All times are seconds on
  the source-video axis. No separate old low-rate decision-grid snapping is used.

`data/dataset.py` is a Python loader, **not the dataset**. It exposes
`DATASET_SPECS`, `VideoSample`, `load_dataset_split`, and `load_protocol_data`.
A sample exposes `video_id`, lazy `.features`, `.times`,
`.ground_truth_boundaries`, `.duration_sec`, and `.n_frames`.
The CLI paths use the same underlying canonical loader through
`load_protocol_data`; canonical video objects expose `.features()`.

```python
from data.dataset import load_dataset_split
validation = load_dataset_split('50salads', '/path/to/50salads',
                                fold=1, split_kind='val', seed=0, protocol='main')
sample = validation[0]
print(sample.features.shape, sample.times.shape, sample.ground_truth_boundaries)
```

Missing files, rate mismatches and empty/invalid splits fail explicitly.
Training/evaluation never silently switches to mock data or random weights.

## Installation and scan backend

Use a Python environment compatible with your chosen CUDA PyTorch/Mamba build.
Python 3.10+ is recommended for this release.

```bash
pip install -r requirements.txt
# Required for either offline runner and real streaming training/evaluation:
pip install mamba-ssm causal-conv1d
```

Both offline paper runners use **mamba_ssm.Mamba**, with package-default dt_rank
and fast-path settings. They require CUDA and perform finite-output/gradient
preflight checks; they use no custom scan or CPU fallback. Tests can inject a
small test double; this is not evidence of a real CUDA experiment.

The manuscript records PyTorch 2.11.0+cu130, CUDA 13.0 and RTX 6000 Blackwell
for the two offline studies. The **historical Mamba/causal-conv1d versions and
effective fast path were not recovered**. A fresh installation version must not
be presented as the historical experiment version. Runners record the current
environment in new manifests. The historical effective ablation precision is
also unavailable; its automatic policy is documented above.

`models/groupwise_mamba.py::FallbackScan` is a development-only implementation.
It uses **dt_rank=1**, whereas standard Mamba's default rank can differ (e.g.
4 at d_model=64). Similar recurrence equations do not establish identical
parameterization, initialization, outputs or checkpoint compatibility. The
fallback is not an exact/equivalent replacement and produced neither offline
paper result. CPU demonstrations are not reproduced paper experiments.

## Commands

All official folds and both seeds, one dataset at a time:

```bash
python train.py --protocol main --dataset gtea --data_root /path/to/gtea
python train.py --protocol main --dataset 50salads --data_root /path/to/50salads
python train.py --protocol main --dataset breakfast --data_root /path/to/breakfast
```

A single run (same recipe, no test-dependent overrides):

```bash
python train.py --protocol main --dataset 50salads --data_root /path/to/50salads --fold 1 --seed 0
python train.py --protocol ablation --dataset 50salads --data_root /path/to/50salads --variant linear_head
```

The CLI records per-run raw logs, manifest, history, checkpoint, calibration,
and result JSON. Main runs additionally save validation/test logits and
predictions; summaries are written only when all folds and both seeds run.
Official standalone runners are import-safe; edit their clearly marked root/output
settings before running directly. Group/D and permutation controls have their own CLI.

```bash
python eval.py --protocol main --dataset 50salads --data_root /path/to/50salads   --run_dir outputs/main/full/50salads/seed0_fold1
python eval.py --protocol ablation --dataset 50salads --data_root /path/to/50salads   --run_dir outputs/ablation/linear_head/full/50salads/seed0_fold1
# Re-score a saved main-run logits archive without data access or CUDA:
python -c "from experiments.offline_main import m5_rescore_saved; m5_rescore_saved('outputs/main/full/50salads/seed0_fold1')"
```

Separate streaming reference, explicit checkpoint path per fold/seed/group:

```bash
python train_streaming.py train --data_root /path/to/50salads --fold 1 --seed 0 --groups 32 --checkpoint outputs/streaming/f1s0/best.pt
python train_streaming.py eval --data_root /path/to/50salads --fold 1 --seed 0 --groups 32 --checkpoint outputs/streaming/f1s0/best.pt
python demo_streaming.py
```

Only load trusted checkpoints: PyTorch checkpoint loading may execute pickled code.

## Repository layout

```text
train.py / eval.py             Offline protocol entry points
train_streaming.py             Separate current streaming reference
experiments/offline_main.py    Main Adam/50 experiment
experiments/offline_ablation.py Component AdamW/60 experiment
experiments/group_d_controls.py Group-count and D controls
experiments/permutation_controls.py Time-indexed permutation controls
data/dataset.py                Public loading and official-split interfaces
models/                       Offline API and streaming/development modules
decoding/                     DP and streaming decoding utilities
evaluation/matching.py        Explicit greedy/optimal/streaming scorers
tests/                        Synthetic loader/protocol/model regression tests
```

Run local CPU checks with `python -m unittest discover -s tests -v`. These use
synthetic fixtures and an explicitly labelled test double, not third-party data
or a substitute for CUDA training. Full dataset training, GPU numerical behavior,
and the paper's reported scores cannot be verified from CPU tests alone.

## Citation and license

Use the manuscript title and author above when citing this work; final publication
metadata will be added when available. The MIT license applies to this repository's
code; it grants no redistribution rights over third-party features or datasets.
