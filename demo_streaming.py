"""Interactive Real-Time Feature-Streaming Demo for Action Boundary Detection.

Demonstrates:
- O(1) computation per incoming feature sample
- Bounded O(1) persistent state memory (160 KiB FP32)
- Online Welford running normalization
- Causal local peak confirmation with 1-sample decision look-ahead
"""

from __future__ import annotations

import time
import numpy as np
import torch

from models.streaming_pipeline import StreamingBoundaryModel, StreamingDetector


def run_streaming_demo():
    print("=====================================================================")
    print("  Stateful Feature-Streaming Action Boundary Detection Demo")
    print("  Synthetic, untrained development demo; not a paper benchmark")
    print("=====================================================================")

    # 1. Initialize model and detector
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = StreamingBoundaryModel(input_dim=2048, n_groups=32, context_dim=64).to(device)
    model.eval()

    feature_hz = 15.0  # 15 frames per second
    detector = StreamingDetector(
        model=model,
        feature_hz=feature_hz,
        peak_threshold=0.6,
        min_gap_sec=1.0,
        device=device,
        retain_boundaries=True,
    )

    mem_bytes = detector.get_memory_bytes()
    print(f"Model Width: 64 channels x 32 groups = 2048 input dimensions")
    print(f"Persistent Conv/SSM State: {mem_bytes:,} bytes ({mem_bytes / 1024:.1f} KiB)")
    print(f"Streaming Rate: {feature_hz} Hz (1 step every {1000.0 / feature_hz:.1f} ms)")
    print("-" * 69)

    # 2. Simulate streaming features for a 20-second video sequence
    duration_sec = 20.0
    total_steps = int(duration_sec * feature_hz)
    print(f"Streaming {total_steps} feature vectors (duration: {duration_sec}s)...")

    # Generate synthetic features with synthetic transition energy bursts
    np.random.seed(42)
    torch.manual_seed(42)

    synthetic_transitions = [3.5, 7.2, 11.8, 16.0]  # Ground truth boundary seconds
    features = np.random.randn(total_steps, 2048).astype(np.float32) * 0.1

    # Inject distinctive transition activity
    for gt in synthetic_transitions:
        idx = int(gt * feature_hz)
        if 0 <= idx < total_steps:
            features[idx:idx + 2] += np.random.randn(min(2, total_steps - idx), 2048) * 1.5

    latencies_ms = []

    print(f"{'Step':<6} | {'Time (s)':<9} | {'Logit':<8} | {'Z-Score':<8} | {'Status'}")
    print("-" * 69)

    for step in range(total_steps):
        feat = torch.from_numpy(features[step])

        t0 = time.perf_counter()
        result = detector.push_feature(feat)
        t1 = time.perf_counter()
        latency = (t1 - t0) * 1000.0
        latencies_ms.append(latency)

        status = ""
        if result["is_confirmed_boundary"]:
            status = f"--> [BOUNDARY CONFIRMED at {result['confirmed_boundary_sec']:.2f}s!]"

        # Print periodic progress or confirmed boundaries
        if result["is_confirmed_boundary"] or step % 30 == 0:
            print(
                f"{result['step']:<6d} | "
                f"{result['timestamp']:<9.2f} | "
                f"{result['logit']:<8.3f} | "
                f"{result['normalized_score']:<8.3f} | "
                f"{status}"
            )

    print("=" * 69)
    print("STREAMING SUMMARY:")
    print(f"Total steps processed: {total_steps}")
    print(f"Total boundaries detected: {len(detector.detected_boundaries)}")
    print(f"Detected timestamps: {[round(t, 2) for t in detector.detected_boundaries]}")
    print(f"Ground-truth transitions: {synthetic_transitions}")
    print(f"Step latency: median={np.percentile(latencies_ms, 50):.2f} ms, "
          f"p95={np.percentile(latencies_ms, 95):.2f} ms, "
          f"max={np.max(latencies_ms):.2f} ms")
    print(f"Conv/SSM state (excludes retained output list): {detector.get_memory_bytes() / 1024:.1f} KiB")
    print("=" * 69)


if __name__ == "__main__":
    run_streaming_demo()
