"""Penalized Dynamic Programming Decoder for Action Boundary Detection.

Solves the global boundary selection problem:
    maximize sum_{t in S} q_t - lambda * |S|
    subject to |t_i - t_j| >= m for all distinct transitions in S.

Runs in strictly O(T) time and extracts candidate boundaries via backtracking,
followed by parabolic 3-point sub-step refinement and validation temporal calibration.
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

import numpy as np


def normalize_logits(logits: np.ndarray, std_floor: float = 0.5) -> np.ndarray:
    """Per-video z-score standardization with standard deviation floor."""
    x = np.asarray(logits, dtype=np.float64)
    if x.ndim != 1 or not np.isfinite(x).all():
        raise ValueError("Logits must be a finite 1D array.")
    if len(x) == 0:
        return x.copy()

    std = float(np.std(x, ddof=1)) if len(x) > 1 else 0.0
    denom = max(std, float(std_floor))
    return (x - float(np.mean(x))) / denom


def solve_penalized_dp(
    scores: np.ndarray,
    penalty_lambda: float,
    min_gap_steps: int,
) -> List[int]:
    """Exact O(T) dynamic programming solver with separation constraint."""
    q = np.asarray(scores, dtype=np.float64)
    T = len(q)
    gap = max(1, int(min_gap_steps))
    lam = float(penalty_lambda)

    # V[t] represents optimal cumulative score up to index t-1
    V = np.zeros(T + 1, dtype=np.float64)
    take = np.zeros(T, dtype=bool)

    for i in range(T):
        prev_idx = max(0, i + 1 - gap)
        cut_val = V[prev_idx] + q[i] - lam
        if cut_val > V[i]:
            take[i] = True
            V[i + 1] = cut_val
        else:
            V[i + 1] = V[i]

    # Backtracking
    selected: List[int] = []
    i = T - 1
    while i >= 0:
        if take[i]:
            selected.append(i)
            i -= gap
        else:
            i -= 1

    return list(reversed(selected))


def decode_boundaries_dp(
    logits: np.ndarray,
    times: np.ndarray,
    duration_sec: float,
    penalty_lambda: float = 1.0,
    min_gap_sec: float = 1.0,
    feature_hz: float = 15.0,
    refine_parabolic: bool = True,
    temporal_offset_sec: float = 0.0,
) -> List[float]:
    """Decodes boundary timestamps from raw model logits."""
    q = normalize_logits(logits)
    min_gap_steps = max(1, int(math.ceil(min_gap_sec * feature_hz - 1e-9)))

    indices = solve_penalized_dp(q, penalty_lambda=penalty_lambda, min_gap_steps=min_gap_steps)

    predictions: List[float] = []
    for i in indices:
        shift = 0.0
        if refine_parabolic and 0 < i < len(q) - 1:
            if q[i] >= q[i - 1] and q[i] >= q[i + 1]:
                denom = q[i - 1] - 2.0 * q[i] + q[i + 1]
                if denom < -1e-12:
                    shift = float(np.clip(0.5 * (q[i - 1] - q[i + 1]) / denom, -0.5, 0.5))

        refined_time = times[i] + shift / feature_hz - temporal_offset_sec
        refined_time = float(np.clip(refined_time, 0.0, duration_sec))
        predictions.append(refined_time)

    return predictions
