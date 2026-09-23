"""Unified Training Script for Full and Streaming Boundary Detection Pipelines.

Usage:
    python train.py --dataset 50salads --mode full --fold 1 --data_root /path/to/50salads
    python train.py --dataset 50salads --mode streaming --fold 1 --data_root /path/to/50salads
"""

from __future__ import annotations

import argparse
import math
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

from data.dataset import DATASET_SPECS, load_dataset_split
from decoding.dp_decoder import decode_boundaries_dp
from evaluation.matching import compute_boundary_prf
from models.full_pipeline import FullBoundaryModel, compute_gaussian_target, weighted_bce_loss
from models.streaming_pipeline import StreamingBoundaryModel


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_epoch(
    model: nn.Module,
    train_samples: list,
    optimizer: torch.optim.Optimizer,
    spec,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    random.shuffle(train_samples)

    for sample in train_samples:
        feats = sample.features.to(device)  # (T, 2048)
        targets = compute_gaussian_target(
            sample.times, sample.ground_truth_boundaries, sigma_sec=spec.sigma_sec
        ).to(device)

        optimizer.zero_grad()
        out = model(feats)
        loss = weighted_bce_loss(out["logits"], targets, pos_weight_alpha=4.0)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += float(loss.item())

    return total_loss / max(1, len(train_samples))


@torch.no_grad()
def evaluate_validation(
    model: nn.Module,
    val_samples: list,
    spec,
    device: torch.device,
    penalty_lambda: float = 1.0,
) -> float:
    model.eval()
    f1_scores = []

    for sample in val_samples:
        feats = sample.features.to(device)
        out = model(feats)
        logits = out["logits"].cpu().numpy()

        preds = decode_boundaries_dp(
            logits=logits,
            times=sample.times,
            duration_sec=sample.duration_sec,
            penalty_lambda=penalty_lambda,
            min_gap_sec=spec.min_gap_sec,
            feature_hz=spec.target_fps,
        )

        res = compute_boundary_prf(preds, sample.ground_truth_boundaries, tolerance_sec=spec.primary_tolerance)
        f1_scores.append(res["f1"])

    return float(np.mean(f1_scores)) if f1_scores else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Boundary Detection Models")
    parser.add_argument("--dataset", type=str, default="50salads", choices=["50salads", "gtea", "breakfast"])
    parser.add_argument("--mode", type=str, default="full", choices=["full", "streaming"])
    parser.add_argument("--fold", type=int, default=1)
    parser.add_argument("--data_root", type=str, default="./data/50salads")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="./checkpoints")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    spec = DATASET_SPECS[args.dataset]

    print(f"=== Training {args.mode.upper()} Pipeline on {args.dataset.upper()} (Fold {args.fold}) ===")
    print(f"Device: {device} | Max Epochs: {args.epochs} | Initial LR: {args.lr}")

    # Build model
    if args.mode == "full":
        model = FullBoundaryModel(dataset=args.dataset).to(device)
    else:
        model = StreamingBoundaryModel().to(device)

    model.init_prior(prior_prob=0.05)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # Data loaders
    train_samples = []
    val_samples = []
    if Path(args.data_root).exists():
        try:
            train_samples = load_dataset_split(args.dataset, args.data_root, fold=args.fold, split_kind="train")
            val_samples = load_dataset_split(args.dataset, args.data_root, fold=args.fold, split_kind="val")
        except Exception as e:
            print(f"Warning: Could not load data from {args.data_root}: {e}")

    if not train_samples:
        print("Note: No dataset files found. Generating synthetic mock samples for smoke testing...")
        from data.dataset import VideoSample
        for i in range(4):
            T = 300
            feats = np.random.randn(T, 2048).astype(np.float32)
            gts = [5.0, 10.0, 15.0]
            train_samples.append(VideoSample(f"train_{i}", feats, gts, source_fps=spec.source_fps, subsample_factor=spec.subsample_factor))
            val_samples.append(VideoSample(f"val_{i}", feats, gts, source_fps=spec.source_fps, subsample_factor=spec.subsample_factor))

    # Optimizer and scheduler (10% linear warmup + cosine decay)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    warmup_epochs = max(1, int(0.1 * args.epochs))
    sched_warmup = LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs)
    sched_cosine = CosineAnnealingLR(optimizer, T_max=max(1, args.epochs - warmup_epochs), eta_min=1e-6)
    scheduler = SequentialLR(optimizer, schedulers=[sched_warmup, sched_cosine], milestones=[warmup_epochs])

    os.makedirs(args.output_dir, exist_ok=True)
    best_f1 = -1.0
    checkpoint_path = Path(args.output_dir) / f"{args.dataset}_{args.mode}_fold{args.fold}_best.pt"

    for epoch in range(1, args.epochs + 1):
        loss = train_epoch(model, train_samples, optimizer, spec, device)
        scheduler.step()

        val_f1 = evaluate_validation(model, val_samples, spec, device)
        print(f"Epoch {epoch:02d}/{args.epochs:02d} | Train Loss: {loss:.4f} | Val F1@{spec.primary_tolerance}s: {val_f1:.4f}")

        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_f1": val_f1,
                "dataset": args.dataset,
                "mode": args.mode,
            }, checkpoint_path)
            print(f"  -> Saved new best checkpoint (F1: {best_f1:.4f})")

    print(f"\nTraining completed. Best validation F1: {best_f1:.4f}")


if __name__ == "__main__":
    main()
