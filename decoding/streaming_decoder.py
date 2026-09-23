"""Streaming Decoders for Online Feature Feeds."""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

import numpy as np


class OnlineWelfordStats:
    """Welford algorithm for incremental sample mean and sample variance."""

    def __init__(self, std_floor: float = 0.5) -> None:
        self.std_floor = float(std_floor)
        self.count = 0
        self.mean = 0.0
        self.m2 = 0.0

    def update(self, val: float) -> Tuple[float, float, float]:
        """Pushes new value and returns (normalized_zscore, running_mean, running_std)."""
        self.count += 1
        delta = val - self.mean
        self.mean += delta / self.count
        delta2 = val - self.mean
        self.m2 += delta * delta2

        if self.count > 1:
            variance = max(0.0, self.m2 / (self.count - 1))
            std = math.sqrt(variance)
        else:
            std = 0.0

        z_score = (val - self.mean) / max(std, self.std_floor)
        return z_score, self.mean, std


def causal_peak_confirm(
    scores: List[float],
    current_step: int,
    peak_threshold: float = 0.5,
    last_confirmed_step: int = -100,
    min_gap_steps: int = 15,
) -> Tuple[bool, Optional[int]]:
    """Checks whether the previous score (at current_step - 1) is a confirmed local peak.

    Requires:
      score[t-1] > score[t-2]
      score[t-1] >= score[t]
      score[t-1] >= peak_threshold
      (t - 1) - last_confirmed_step >= min_gap_steps
    """
    if len(scores) < 3:
        return False, None

    prev2, prev1, curr = scores[-3], scores[-2], scores[-1]
    candidate_step = current_step - 1

    if (
        prev1 > prev2
        and prev1 >= curr
        and prev1 >= peak_threshold
        and (candidate_step - last_confirmed_step >= min_gap_steps)
    ):
        return True, candidate_step

    return False, None
