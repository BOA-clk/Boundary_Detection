"""One-to-One Bipartite Assignment Matching for Action Boundary Evaluation.

Ensures strict 1-to-1 matching between predicted and ground-truth boundary timestamps
using maximum-cardinality, minimum-error bipartite assignment (linear sum assignment).
Prevents over-segmentation cheating where multiple predictions could otherwise claim
a single transition.
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment


def bipartite_boundary_match(
    predictions: Sequence[float],
    ground_truth: Sequence[float],
    tolerance_sec: float,
) -> List[Tuple[int, int]]:
    """Finds maximum-cardinality, minimum-distance 1-to-1 matches within tolerance.

    Returns:
        List of (pred_idx, gt_idx) pairs.
    """
    pred = np.asarray(predictions, dtype=np.float64)
    gt = np.asarray(ground_truth, dtype=np.float64)

    if len(pred) == 0 or len(gt) == 0:
        return []

    # Pairwise temporal distance matrix: (len(gt), len(pred))
    distances = np.abs(gt[:, None] - pred[None, :])

    # Augment cost matrix for unmatchable pairs
    penalty = 1e6
    n_gt, n_pred = len(gt), len(pred)
    dim = max(n_gt, n_pred)
    costs = np.full((dim, dim), 2.0 * penalty, dtype=np.float64)

    valid_mask = distances <= (tolerance_sec + 1e-9)
    costs[:n_gt, :n_pred] = np.where(valid_mask, distances, penalty)

    row_ind, col_ind = linear_sum_assignment(costs)

    matches: List[Tuple[int, int]] = []
    for r, c in zip(row_ind, col_ind):
        if r < n_gt and c < n_pred and distances[r, c] <= (tolerance_sec + 1e-9):
            matches.append((int(c), int(r)))

    return matches


def compute_boundary_prf(
    predictions: Sequence[float],
    ground_truth: Sequence[float],
    tolerance_sec: float,
) -> Dict[str, float]:
    """Computes Precision, Recall, and F1 score for a single video."""
    n_pred = len(predictions)
    n_gt = len(ground_truth)

    if n_pred == 0 and n_gt == 0:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0, "tp": 0, "n_pred": 0, "n_gt": 0}
    if n_pred == 0 or n_gt == 0:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "tp": 0, "n_pred": n_pred, "n_gt": n_gt}

    matches = bipartite_boundary_match(predictions, ground_truth, tolerance_sec=tolerance_sec)
    tp = len(matches)

    precision = tp / n_pred if n_pred > 0 else 0.0
    recall = tp / n_gt if n_gt > 0 else 0.0
    f1 = (2.0 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "n_pred": n_pred,
        "n_gt": n_gt,
    }


def evaluate_dataset_macro_f1(
    predictions_per_video: Dict[str, Sequence[float]],
    ground_truth_per_video: Dict[str, Sequence[float]],
    tolerances: Sequence[float],
) -> Dict[float, Dict[str, float]]:
    """Computes video-macro averaged Precision, Recall, and F1 across multiple tolerances."""
    video_ids = sorted(predictions_per_video.keys())
    results: Dict[float, Dict[str, float]] = {}

    for tol in tolerances:
        precs, recs, f1s = [], [], []
        for vid in video_ids:
            pred = predictions_per_video[vid]
            gt = ground_truth_per_video.get(vid, [])
            m = compute_boundary_prf(pred, gt, tolerance_sec=tol)
            precs.append(m["precision"])
            recs.append(m["recall"])
            f1s.append(m["f1"])

        results[tol] = {
            "precision": float(np.mean(precs)),
            "recall": float(np.mean(recs)),
            "f1": float(np.mean(f1s)),
        }

    return results
