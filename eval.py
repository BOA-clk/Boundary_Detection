"""Evaluation Script for Full Pipeline and Stateful Streaming Inference.

Usage:
    python eval.py --dataset 50salads --mode full --checkpoint checkpoints/50salads_full_fold1_best.pt
    python eval.py --dataset 50salads --mode streaming --checkpoint checkpoints/50salads_streaming_fold1_best.pt
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

from data.dataset import DATASET_SPECS, VideoSample, load_dataset_split
from decoding.dp_decoder import decode_boundaries_dp
from evaluation.matching import evaluate_dataset_macro_f1
from models.full_pipeline import FullBoundaryModel
from models.streaming_pipeline import StreamingBoundaryModel, StreamingDetector


def evaluate_full_pipeline(
    model: FullBoundaryModel,
    test_samples: List[VideoSample],
    spec,
    device: torch.device,
    penalty_lambda: float = 1.0,
    temporal_offset_sec: float = 0.0,
) -> Dict[float, Dict[str, float]]:
    model.eval()
    preds_dict: Dict[str, List[float]] = {}
    gts_dict: Dict[str, List[float]] = {}

    start_time = time.perf_counter()
    total_frames = 0

    with torch.no_grad():
        for sample in test_samples:
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
                refine_parabolic=True,
                temporal_offset_sec=temporal_offset_sec,
            )

            preds_dict[sample.video_id] = preds
            gts_dict[sample.video_id] = sample.ground_truth_boundaries
            total_frames += sample.n_frames

    elapsed = time.perf_counter() - start_time
    fps = total_frames / elapsed if elapsed > 0 else 0.0
    print(f"Processed {total_frames} frames in {elapsed:.3f}s ({fps:.1f} frames/s)")

    results = evaluate_dataset_macro_f1(preds_dict, gts_dict, tolerances=spec.tolerances)
    return results


def evaluate_streaming_pipeline(
    model: StreamingBoundaryModel,
    test_samples: List[VideoSample],
    spec,
    device: torch.device,
    peak_threshold: float = 0.5,
    temporal_offset_sec: float = 0.0,
) -> Dict[float, Dict[str, float]]:
    model.eval()
    detector = StreamingDetector(
        model=model,
        feature_hz=spec.target_fps,
        peak_threshold=peak_threshold,
        min_gap_sec=spec.min_gap_sec,
        temporal_offset_sec=temporal_offset_sec,
        device=device,
    )

    preds_dict: Dict[str, List[float]] = {}
    gts_dict: Dict[str, List[float]] = {}

    latencies_ms: List[float] = []

    for sample in test_samples:
        detector.reset()
        for t in range(sample.n_frames):
            x_t = sample.features[t]
            t0 = time.perf_counter()
            _ = detector.push_feature(x_t)
            t1 = time.perf_counter()
            latencies_ms.append((t1 - t0) * 1000.0)

        preds_dict[sample.video_id] = list(detector.detected_boundaries)
        gts_dict[sample.video_id] = sample.ground_truth_boundaries

    p50 = float(np.percentile(latencies_ms, 50))
    p95 = float(np.percentile(latencies_ms, 95))
    mem_kib = detector.get_memory_bytes() / 1024.0
    print(f"Streaming Latency per step: median={p50:.2f}ms, p95={p95:.2f}ms | Working Memory: {mem_kib:.1f} KiB")

    results = evaluate_dataset_macro_f1(preds_dict, gts_dict, tolerances=spec.tolerances)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Boundary Detection Models")
    parser.add_argument("--dataset", type=str, default="50salads", choices=["50salads", "gtea", "breakfast"])
    parser.add_argument("--mode", type=str, default="full", choices=["full", "streaming"])
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--fold", type=int, default=1)
    parser.add_argument("--data_root", type=str, default="./data/50salads")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    spec = DATASET_SPECS[args.dataset]

    if args.mode == "full":
        model = FullBoundaryModel(dataset=args.dataset).to(device)
    else:
        model = StreamingBoundaryModel().to(device)

    if args.checkpoint and Path(args.checkpoint).exists():
        ckpt = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"Loaded checkpoint from {args.checkpoint}")

    # Load test samples
    test_samples: List[VideoSample] = []
    if Path(args.data_root).exists():
        try:
            test_samples = load_dataset_split(args.dataset, args.data_root, fold=args.fold, split_kind="test")
        except Exception as e:
            print(f"Warning: Could not load test split from {args.data_root}: {e}")

    if not test_samples:
        print("Note: Dataset not found on disk. Running smoke test on synthetic samples...")
        for i in range(2):
            T = 300
            feats = np.random.randn(T, 2048).astype(np.float32)
            gts = [5.0, 10.0, 15.0]
            test_samples.append(VideoSample(f"test_{i}", feats, gts, source_fps=spec.source_fps, subsample_factor=spec.subsample_factor))

    print(f"\n=== Evaluating {args.mode.upper()} Pipeline on {args.dataset.upper()} ===")
    if args.mode == "full":
        results = evaluate_full_pipeline(model, test_samples, spec, device)
    else:
        results = evaluate_streaming_pipeline(model, test_samples, spec, device)

    print("\n---------------- RESULTS ----------------")
    print(f"{'Tolerance':<12} | {'Precision':<10} | {'Recall':<10} | {'F1 Score':<10}")
    print("-" * 52)
    for tol, metrics in results.items():
        is_primary = " (primary)" if abs(tol - spec.primary_tolerance) < 1e-4 else ""
        tol_str = f"{tol:.1f}s{is_primary}"
        print(f"{tol_str:<12} | {metrics['precision']:<10.4f} | {metrics['recall']:<10.4f} | {metrics['f1']:<10.4f}")
    print("-----------------------------------------")


if __name__ == "__main__":
    main()
