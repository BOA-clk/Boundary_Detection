# Groupwise Selective State-Space Filtering for Accurate and Streaming Action Boundary Detection

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![IEEE SPL](https://img.shields.io/badge/Paper-IEEE%20SPL%202026-brightgreen.svg)](https://github.com/BOA-clk/Boundary_Detection)

Official reference implementation for **"Groupwise Selective State-Space Filtering for Accurate and Streaming Action Boundary Detection"** (IEEE Signal Processing Letters, 2026).

---

## Overview

Action boundary detection partitions untrimmed video sequences into intervals without requiring semantic action labels. This repository contains the complete, cleaned implementation of both:

1. **Full Pipeline (Offline / Bidirectional Context):**
   - **Groupwise Selective Scan:** Partitions pre-extracted 2048-dim I3D features into $G=32$ isolated channel groups (64 dims each), evaluated by independent Mamba blocks with frozen $D=0$ (pure state dynamics without appearance skip).
   - **Learned Group Fusion:** Projects concatenated state responses ($2048 \rightarrow 64$).
   - **Residual Dilated TCN:** 3 residual layers with symmetric padding for local two-sided temporal context.
   - **Penalized Dynamic Programming (DP):** Solves the global boundary selection problem in $O(T)$ time with separation constraints and parabolic sub-step refinement.

2. **Stateful Streaming Pipeline (Online / Real-Time):**
   - **Zero Neural Look-Ahead:** Replaces the symmetric TCN with a direct boundary head on $z_t$.
   - **Bounded Memory:** Exactly $O(1)$ working memory (**160 KiB** FP32 recurrent state), constant with sequence duration.
   - **Welford Running Normalization:** Constant-time score standardization without caching past logits.
   - **Causal Local Peak Confirmation:** Confirms local maxima with only 1-sample decision look-ahead ($q_{t-1} > q_{t-2}$ and $q_{t-1} \ge q_t$).

```
========================================================================================
FULL PIPELINE (OFFLINE)
Frame Feature x_t (2048) -> [Split 32 Groups] -> [32 Isolated Mamba Scans] 
                         -> [Concatenate r_t] -> [Linear Fusion -> z_t (64)]
                         -> [3-Layer Dilated TCN] -> [MLP Head] -> Penalized DP Decoder
========================================================================================

========================================================================================
STATEFUL STREAMING (ONLINE)
Feature Step x_t (2048)  -> [Single-Step Mamba Recurrence: O(1) Time, 160 KiB Memory]
                         -> [Linear Fusion -> z_t (64)] -> [Direct Boundary Head]
                         -> [Welford Running Z-Score]   -> [Causal 1-Step Peak Confirmation]
========================================================================================
```

---

## Experimental Results

Evaluated on the official splits of **Breakfast (4 folds)**, **GTEA (4 folds)**, and **50Salads (5 folds)** using strict **1-to-1 bipartite assignment matching** within temporal tolerances.

### Main Benchmark Results (Primary $F_1$)

| Method | Breakfast ($F_1@1.0\text{s}$) | GTEA ($F_1@0.4\text{s}$) | 50Salads ($F_1@1.0\text{s}$) | FLOPs ($T=1092$) |
| :--- | :---: | :---: | :---: | :---: |
| MS-TCN (Farha et al., 2019) | 0.349 | 0.584 | 0.582 | 1.740 GFLOPs |
| ASRF (Ishikawa et al., 2021) | 0.370 | 0.606 | 0.607 | 2.810 GFLOPs |
| BaFormer (2024) | 0.313 | 0.562 | 0.551 | 1.780 GFLOPs |
| **Proposed Full Pipeline** | **0.457** | **0.622** | **0.611** | **1.756 GFLOPs** |

### Stateful Streaming Performance (50Salads)

| Group Width | Groups ($G$) | Parameters | Persistent Memory | Primary $F_1@1.0\text{s}$ |
| :---: | :---: | :---: | :---: | :---: |
| 2048 | 1 | 13.359 M | 160 KiB | $0.4687 \pm 0.0735$ |
| 32 | 64 | 0.453 M | 160 KiB | $0.4736 \pm 0.0460$ |
| **64 (Ours)** | **32** | **0.658 M** | **160 KiB** | $\mathbf{0.4818 \pm 0.0653}$ |

*Measured latency: **50.5 steps/sec**, p95 processing time of **20.45 ms/feature**, easily satisfying the 15-Hz streaming requirement.*

---

## Repository Structure

```
Boundary_Detection/
├── models/
│   ├── groupwise_mamba.py     # 32 isolated Mamba blocks with Fallback CPU Scan
│   ├── full_pipeline.py       # Offline model with Residual Dilated TCN & soft target loss
│   └── streaming_pipeline.py  # Online model with StreamingDetector & Welford normalizer
├── decoding/
│   ├── dp_decoder.py          # O(T) Penalized Dynamic Programming decoder
│   └── streaming_decoder.py   # Welford running statistics & causal peak confirmation
├── evaluation/
│   └── matching.py            # Strict 1-to-1 bipartite assignment matching (F1, Precision, Recall)
├── data/
│   └── dataset.py             # Feature loader & ground truth parser for 50Salads, GTEA, Breakfast
├── train.py                   # Unified training script (AdamW + warmup + cosine decay)
├── eval.py                    # Unified evaluation script for full & streaming pipelines
├── demo_streaming.py          # Interactive real-time streaming demonstration
├── requirements.txt           # Dependencies
└── README.md                  # Documentation
```

---

## Installation

### Prerequisites
- Python 3.9 or higher
- PyTorch $\ge$ 2.0

```bash
git clone https://github.com/BOA-clk/Boundary_Detection.git
cd Boundary_Detection
pip install -r requirements.txt
```

*(Optional)* For CUDA acceleration of Mamba state-space scans:
```bash
pip install mamba-ssm causal-conv1d
```
*Note: If `mamba-ssm` or CUDA is unavailable, the repository automatically falls back to an exact, high-performance pure-PyTorch implementation (`FallbackScan`), allowing execution on any CPU, Mac, or GPU environment.*

---

## Quickstart

### 1. Interactive Streaming Demo
Run the frame-by-frame streaming demonstration to verify $O(1)$ memory consumption and step latency:
```bash
python demo_streaming.py
```

### 2. Training
To train the **Full Pipeline** (Offline with TCN):
```bash
python train.py --dataset 50salads --mode full --fold 1 --data_root /path/to/50salads
```

To train the **Stateful Streaming Model** (Online causal):
```bash
python train.py --dataset 50salads --mode streaming --fold 1 --data_root /path/to/50salads
```

### 3. Evaluation
To evaluate a trained checkpoint:
```bash
# Full Pipeline evaluation (DP decoding + parabolic refinement):
python eval.py --dataset 50salads --mode full --checkpoint checkpoints/50salads_full_fold1_best.pt

# Streaming evaluation (frame-by-frame causal peak confirmation):
python eval.py --dataset 50salads --mode streaming --checkpoint checkpoints/50salads_streaming_fold1_best.pt
```

---

## Citation

If you use this code in your research, please cite:

```bibtex
@article{celik2026groupwise,
  title={Groupwise Selective State-Space Filtering for Accurate and Streaming Action Boundary Detection},
  author={{\c{C}}elik, Mustafa Bora},
  journal={IEEE Signal Processing Letters},
  year={2026}
}
```

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
