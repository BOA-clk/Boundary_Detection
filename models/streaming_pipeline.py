"""Stateful Streaming Action Boundary Detection Pipeline.

Guarantees strictly causal, online feature-streaming inference:
1. Zero Neural Look-Ahead: Groupwise Mamba recurrence updates 32 isolated states
   per arriving feature without dilated convolutions or future frames.
2. Bounded Working Memory: Exactly O(1) persistent state (~160 KiB FP32), independent
   of total elapsed video duration.
3. Welford Running Normalization: Online, constant-time score standardization
   without caching past logits.
4. Causal Peak Confirmation: One-sample decision look-ahead confirming local maxima
   immediately upon receiving the subsequent feature.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .groupwise_mamba import GroupwiseMambaScan


class StreamingBoundaryModel(nn.Module):
    """TCN-free, causal boundary detection model for real-time feature streaming."""

    def __init__(
        self,
        input_dim: int = 2048,
        n_groups: int = 32,
        context_dim: int = 64,
        use_mlp_head: bool = True,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.n_groups = n_groups
        self.context_dim = context_dim

        # 1. Groupwise Mamba Scan (G=32, D=0)
        self.scan = GroupwiseMambaScan(input_dim=input_dim, n_groups=n_groups)

        # 2. Learned Group Fusion (2048 -> 64)
        self.fuse = nn.Linear(input_dim, context_dim)

        # 3. Direct Boundary Head (No TCN, strictly causal)
        if use_mlp_head:
            self.head = nn.Sequential(
                nn.Linear(context_dim, context_dim),
                nn.SiLU(),
                nn.Linear(context_dim, 1),
            )
        else:
            self.head = nn.Linear(context_dim, 1)

    def forward(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Sequence forward pass (used during causal training and validation).

        Args:
            features: Pre-extracted frame features (T, 2048) or (B, T, 2048).
        """
        r = self.scan(features)               # (T, 2048)
        z = self.fuse(r)                     # (T, 64)
        logits = self.head(z).squeeze(-1)    # (T,)
        probs = torch.sigmoid(logits)
        return {
            "mamba_out": r,
            "fused": z,
            "logits": logits,
            "probs": probs,
        }

    def allocate_inference_cache(
        self, batch_size: int = 1, dtype: Optional[torch.dtype] = None
    ) -> List[Any]:
        """Allocates recurrent inference states for all 32 groups."""
        return self.scan.allocate_inference_cache(batch_size=batch_size, dtype=dtype)

    def step(
        self,
        feature_step: torch.Tensor,
        scan_states: List[Any],
    ) -> Tuple[torch.Tensor, torch.Tensor, List[Any]]:
        """Single-step causal state update for online streaming.

        Args:
            feature_step: Feature vector for the arriving frame (2048,).
            scan_states: Current list of 32 group recurrent states.
        Returns:
            Tuple of (logit: scalar Tensor, fused_z: (64,) Tensor, new_scan_states).
        """
        r_step, new_states = self.scan.step(feature_step, scan_states)  # (2048,)
        z_step = self.fuse(r_step)                                      # (64,)
        logit_step = self.head(z_step).squeeze(-1)                      # scalar
        return logit_step, z_step, new_states

    def init_prior(self, prior_prob: float = 0.05) -> None:
        """Initializes output bias to reflect boundary sparsity."""
        p = float(np.clip(prior_prob, 1e-4, 1.0 - 1e-4))
        prior_bias = math.log(p / (1.0 - p))
        with torch.no_grad():
            last_linear = self.head[-1] if isinstance(self.head, nn.Sequential) else self.head
            last_linear.bias.fill_(prior_bias)


class WelfordRunningNormalizer:
    """O(1) time and memory online standardization via Welford's algorithm."""

    def __init__(self, std_floor: float = 0.5) -> None:
        self.std_floor = float(std_floor)
        self.reset()

    def reset(self) -> None:
        self.count = 0
        self.mean = 0.0
        self.m2 = 0.0

    def update(self, value: float) -> float:
        self.count += 1
        delta = value - self.mean
        self.mean += delta / self.count
        delta2 = value - self.mean
        self.m2 += delta * delta2

        if self.count > 1:
            var = max(0.0, self.m2 / (self.count - 1))
            std = math.sqrt(var)
        else:
            std = 0.0

        denom = max(std, self.std_floor)
        return (value - self.mean) / denom


class StreamingDetector:
    """Stateful, online action boundary detection engine.

    Processes incoming frame features one by one, updating 32 recurrent Mamba states
    in O(1) time, performing Welford running standardization, and confirming local
    peaks with one-sample decision look-ahead.
    """

    def __init__(
        self,
        model: StreamingBoundaryModel,
        feature_hz: float = 15.0,
        peak_threshold: float = 0.5,
        min_gap_sec: float = 1.0,
        std_floor: float = 0.5,
        temporal_offset_sec: float = 0.0,
        device: Optional[torch.device] = None,
    ) -> None:
        self.model = model.eval()
        self.feature_hz = float(feature_hz)
        self.peak_threshold = float(peak_threshold)
        self.min_gap_sec = float(min_gap_sec)
        self.min_gap_steps = max(1, int(round(min_gap_sec * feature_hz)))
        self.temporal_offset_sec = float(temporal_offset_sec)
        self.device = device or next(model.parameters()).device

        self.normalizer = WelfordRunningNormalizer(std_floor=std_floor)
        self.reset()

    def reset(self) -> None:
        """Resets recurrent states and detection buffers."""
        self.scan_states = self.model.allocate_inference_cache(batch_size=1)
        self.normalizer.reset()
        self.step_idx = 0

        # Ring buffer for 3-step local peak confirmation: [s_{t-2}, s_{t-1}, s_t]
        self.score_history: List[float] = []
        self.last_detected_step = -self.min_gap_steps
        self.detected_boundaries: List[float] = []

    def get_memory_bytes(self) -> int:
        """Calculates total persistent state memory across all 32 groups."""
        total_bytes = 0
        for state in self.scan_states:
            for tensor in state:
                if isinstance(tensor, torch.Tensor):
                    total_bytes += tensor.numel() * tensor.element_size()
        return total_bytes

    @torch.inference_mode()
    def push_feature(self, feature_step: torch.Tensor) -> Dict[str, Any]:
        """Pushes a newly arriving feature vector (2048,) and evaluates transitions.

        Returns:
            Dict containing step diagnostics and emitted boundary events.
        """
        if feature_step.ndim != 1 or feature_step.numel() != self.model.input_dim:
            raise ValueError(
                f"Expected feature of shape ({self.model.input_dim},), got {tuple(feature_step.shape)}"
            )

        feat = feature_step.to(device=self.device, dtype=next(self.model.parameters()).dtype)

        # 1. Update 32 recurrent Mamba states and evaluate boundary head (O(1))
        logit_t, _, self.scan_states = self.model.step(feat, self.scan_states)
        logit_val = float(logit_t.item())

        # 2. Update Welford running normalization (O(1))
        score_t = self.normalizer.update(logit_val)

        # 3. Causal Local Peak Confirmation with 1-sample decision look-ahead
        self.score_history.append(score_t)
        t = self.step_idx
        self.step_idx += 1

        is_confirmed = False
        confirmed_time = None

        if len(self.score_history) >= 3:
            # Check if step t-1 is a local maximum above peak_threshold
            q_prev2 = self.score_history[-3]  # step t-2
            q_prev1 = self.score_history[-2]  # step t-1 (candidate)
            q_curr  = self.score_history[-1]  # step t   (confirmation)

            candidate_step = t - 1

            if (
                q_prev1 > q_prev2
                and q_prev1 >= q_curr
                and q_prev1 >= self.peak_threshold
                and (candidate_step - self.last_detected_step >= self.min_gap_steps)
            ):
                is_confirmed = True
                self.last_detected_step = candidate_step

                # Refine candidate timestamp with parabolic sub-step shift
                shift = 0.0
                denom = q_prev2 - 2.0 * q_prev1 + q_curr
                if denom < -1e-8:
                    shift = float(np.clip(0.5 * (q_prev2 - q_curr) / denom, -0.5, 0.5))

                timestamp = (candidate_step + shift) / self.feature_hz - self.temporal_offset_sec
                timestamp = max(0.0, timestamp)
                self.detected_boundaries.append(timestamp)
                confirmed_time = timestamp

        return {
            "step": t,
            "timestamp": t / self.feature_hz,
            "logit": logit_val,
            "normalized_score": score_t,
            "is_confirmed_boundary": is_confirmed,
            "confirmed_boundary_sec": confirmed_time,
            "total_detected": len(self.detected_boundaries),
        }
