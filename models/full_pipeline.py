"""Full Action Boundary Detection Pipeline (Offline / Bidirectional Context).

Architecture:
1. Groupwise Mamba Scan (G=32 isolated channel groups, D=0 frozen)
2. Learned Group Fusion (2048 -> 64 linear projection)
3. 3-Layer Residual Dilated TCN (Dataset-specific dilations, symmetric padding)
4. Boundary MLP Head (64 -> 64 -> 1)
5. Positive-weighted Gaussian Soft-Target BCE Loss
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .groupwise_mamba import GroupwiseMambaScan

# Dataset-specific dilation triplets for 3-layer residual TCN
DATASET_DILATIONS = {
    "breakfast": (1, 11, 23),
    "gtea": (1, 7, 13),
    "50salads": (1, 17, 35),
}

# Dataset-specific Gaussian ground-truth target std (seconds)
DATASET_SIGMA_SEC = {
    "breakfast": 0.8,
    "gtea": 0.3,
    "50salads": 1.0,
}


class ResidualDilatedTCN(nn.Module):
    """3-layer residual TCN with LayerNorm, dilated Conv1d, and SiLU."""

    def __init__(self, dim: int = 64, dilations: Tuple[int, int, int] = (1, 7, 13)) -> None:
        super().__init__()
        self.dilations = tuple(dilations)
        self.norms = nn.ModuleList([nn.LayerNorm(dim) for _ in dilations])
        self.convs = nn.ModuleList([
            nn.Conv1d(dim, dim, kernel_size=3, dilation=d)
            for d in dilations
        ])

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Args:
            z: Tensor of shape (T, dim) or (B, T, dim).
        """
        has_batch = (z.ndim == 3)
        if not has_batch:
            z = z.unsqueeze(0)  # (1, T, dim)

        B, T, D = z.shape
        x = z
        for norm, conv, d in zip(self.norms, self.convs, self.dilations):
            normed = norm(x)  # (B, T, D)
            # Symmetric padding for two-sided temporal context
            padded = F.pad(normed.transpose(1, 2), (d, d))  # (B, D, T + 2*d)
            conv_out = conv(padded)  # (B, D, T)
            x = x + F.silu(conv_out.transpose(1, 2))

        return x if has_batch else x.squeeze(0)


class FullBoundaryModel(nn.Module):
    """Complete offline action boundary detector with two-sided temporal context."""

    def __init__(
        self,
        dataset: str = "50salads",
        input_dim: int = 2048,
        n_groups: int = 32,
        context_dim: int = 64,
        dilations: Optional[Tuple[int, int, int]] = None,
    ) -> None:
        super().__init__()
        self.dataset = dataset.lower()
        self.input_dim = input_dim
        self.n_groups = n_groups
        self.context_dim = context_dim

        # 1. Groupwise Mamba Scan (G=32, D=0)
        self.scan = GroupwiseMambaScan(input_dim=input_dim, n_groups=n_groups)

        # 2. Learned Group Fusion (2048 -> 64)
        self.fuse = nn.Linear(input_dim, context_dim)

        # 3. Residual Dilated Temporal Context
        dilations = dilations or DATASET_DILATIONS.get(self.dataset, (1, 7, 13))
        self.context = ResidualDilatedTCN(dim=context_dim, dilations=dilations)

        # 4. Boundary MLP Head (64 -> 64 -> 1)
        self.head = nn.Sequential(
            nn.Linear(context_dim, context_dim),
            nn.SiLU(),
            nn.Linear(context_dim, 1),
        )

    def forward(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Args:
            features: Pre-extracted frame features (T, 2048) or (B, T, 2048).
        Returns:
            Dict containing:
                'mamba_out': (T, 2048)
                'fused': (T, 64)
                'context': (T, 64)
                'logits': (T,) boundary transition logits
                'probs': (T,) sigmoid boundary probabilities
        """
        r = self.scan(features)               # (T, 2048)
        z = self.fuse(r)                     # (T, 64)
        c = self.context(z)                  # (T, 64)
        logits = self.head(c).squeeze(-1)    # (T,)
        probs = torch.sigmoid(logits)

        return {
            "mamba_out": r,
            "fused": z,
            "context": c,
            "logits": logits,
            "probs": probs,
        }

    def init_prior(self, prior_prob: float = 0.05) -> None:
        """Initializes the output head bias to reflect typical transition sparsity."""
        p = float(np.clip(prior_prob, 1e-4, 1.0 - 1e-4))
        prior_bias = math.log(p / (1.0 - p))
        with torch.no_grad():
            self.head[-1].bias.fill_(prior_bias)


def compute_gaussian_target(
    times: np.ndarray,
    ground_truth_boundaries: List[float],
    sigma_sec: float = 1.0,
) -> torch.Tensor:
    """Constructs continuous soft Gaussian targets around ground-truth boundaries."""
    if len(ground_truth_boundaries) == 0:
        return torch.zeros(len(times), dtype=torch.float32)

    gt = np.asarray(ground_truth_boundaries, dtype=np.float64)[:, None]  # (K, 1)
    t = np.asarray(times, dtype=np.float64)[None, :]                     # (1, T)

    # Max-projection across all ground-truth boundaries
    diff_sq = (t - gt) ** 2
    gaussian = np.exp(-diff_sq / (2.0 * (sigma_sec ** 2)))
    target = np.max(gaussian, axis=0)
    return torch.from_numpy(target).float()


def weighted_bce_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    pos_weight_alpha: float = 4.0,
) -> torch.Tensor:
    """Positive-weighted binary cross entropy loss accommodating transition sparsity."""
    weights = 1.0 + pos_weight_alpha * targets
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    loss = torch.sum(weights * bce) / torch.sum(weights).clamp_min(1e-6)
    return loss
