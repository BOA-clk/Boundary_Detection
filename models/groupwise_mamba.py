"""Groupwise Selective State-Space Scanning Module.

Adapts the Mamba state-space scan to multi-transition action boundary detection
by maintaining G=32 mutually isolated channel groups. Each group operates on a
64-dimensional feature subspace with an independent recurrent state, preventing
cross-group state leakage.
"""

from __future__ import annotations

import warnings
from typing import Any, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from mamba_ssm import Mamba
    MAMBA_AVAILABLE = True
except Exception:
    Mamba = None
    MAMBA_AVAILABLE = False


class FallbackScan(nn.Module):
    """Development-only pure-PyTorch SSM fallback, not the paper backend.

    Uses dt_rank=1; standard Mamba defaults can use another projection rank.
    Matching recurrence equations do NOT imply identical parameterization,
    initialization, weights, numerical outputs or trained-model equivalence.

    Implements causal 1D depthwise convolution, selective gating, and linear
    state recurrence, providing both full-sequence forward and single-step
    stateful streaming (`.step()`).
    """

    def __init__(
        self,
        d_model: int = 64,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 1,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.d_inner = int(expand * d_model)
        self.dt_rank = 1

        self.in_proj = nn.Linear(d_model, 2 * self.d_inner, bias=False)
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=0,
        )
        self.x_proj = nn.Linear(self.d_inner, d_state * 2 + 1, bias=False)
        self.dt_proj = nn.Linear(1, self.d_inner, bias=True)

        # S4D real parameterization
        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.zeros(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    def allocate_inference_cache(
        self, batch_size: int = 1, max_seqlen: int = 1, dtype: Optional[torch.dtype] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        device = next(self.parameters()).device
        dtype = dtype or next(self.parameters()).dtype
        conv_state = torch.zeros(batch_size, self.d_inner, self.d_conv, device=device, dtype=dtype)
        ssm_state = torch.zeros(batch_size, self.d_inner, self.d_state, device=device, dtype=dtype)
        return conv_state, ssm_state

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        """Parallel sequence forward pass. u: (B, L, D)."""
        B, L, _ = u.shape
        xz = self.in_proj(u)  # (B, L, 2*d_inner)
        x, z = xz.chunk(2, dim=-1)

        # Causal convolution with left padding
        x_conv = x.transpose(1, 2)  # (B, d_inner, L)
        x_conv = F.pad(x_conv, (self.d_conv - 1, 0))
        x_conv = self.conv1d(x_conv)[:, :, :L]
        x_conv = F.silu(x_conv).transpose(1, 2)  # (B, L, d_inner)

        # Selective projection
        ssm_params = self.x_proj(x_conv)  # (B, L, 2*d_state + 1)
        delta = ssm_params[..., :1]
        B_ssm = ssm_params[..., 1:1 + self.d_state]
        C_ssm = ssm_params[..., 1 + self.d_state:]

        delta = F.softplus(self.dt_proj(delta))  # (B, L, d_inner)
        A = -torch.exp(self.A_log.float())  # (d_inner, d_state)

        # Discretization
        dA = torch.exp(torch.einsum("bld,dn->bldn", delta, A))
        dB = torch.einsum("bld,bln->bldn", delta, B_ssm)

        # Recurrent scan
        h = torch.zeros(B, self.d_inner, self.d_state, device=u.device, dtype=u.dtype)
        y_list = []
        for t in range(L):
            h = dA[:, t] * h + dB[:, t] * x_conv[:, t, :, None]
            y_t = torch.einsum("bdn,bn->bd", h, C_ssm[:, t])
            if self.D is not None and (self.D != 0).any():
                y_t = y_t + self.D * x_conv[:, t]
            y_list.append(y_t)

        y = torch.stack(y_list, dim=1)  # (B, L, d_inner)
        out = self.out_proj(y * F.silu(z))
        return out

    def step(
        self,
        u_step: torch.Tensor,
        conv_state: torch.Tensor,
        ssm_state: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Single-step causal recurrent update. u_step: (B, 1, D)."""
        B, _, _ = u_step.shape
        xz = self.in_proj(u_step.squeeze(1))  # (B, 2*d_inner)
        x, z = xz.chunk(2, dim=-1)

        # Update 1D conv buffer
        conv_state = torch.roll(conv_state, shifts=-1, dims=-1)
        conv_state[:, :, -1] = x
        x_conv = torch.sum(conv_state * self.conv1d.weight[:, 0, :], dim=-1)
        if self.conv1d.bias is not None:
            x_conv = x_conv + self.conv1d.bias
        x_conv = F.silu(x_conv)  # (B, d_inner)

        # Selective projection
        ssm_params = self.x_proj(x_conv)
        delta = ssm_params[..., :1]
        B_ssm = ssm_params[..., 1:1 + self.d_state]
        C_ssm = ssm_params[..., 1 + self.d_state:]

        delta = F.softplus(self.dt_proj(delta))  # (B, d_inner)
        A = -torch.exp(self.A_log.float())  # (d_inner, d_state)

        dA = torch.exp(delta[:, :, None] * A[None, :, :])
        dB = delta[:, :, None] * B_ssm[:, None, :]

        # Recurrent state update: h_t = A_bar * h_{t-1} + B_bar * v_t
        ssm_state = dA * ssm_state + dB * x_conv[:, :, None]
        y_t = torch.sum(ssm_state * C_ssm[:, None, :], dim=-1)
        if self.D is not None and (self.D != 0).any():
            y_t = y_t + self.D * x_conv

        out = self.out_proj(y_t * F.silu(z)).unsqueeze(1)
        return out, conv_state, ssm_state


def create_mamba_block(
    d_model: int = 64,
    d_state: int = 16,
    d_conv: int = 4,
    expand: int = 1,
) -> nn.Module:
    """Instantiates a Mamba block, using CUDA mamba_ssm if available or FallbackScan."""
    if MAMBA_AVAILABLE and torch.cuda.is_available():
        block = Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
    else:
        warnings.warn('Development FallbackScan (dt_rank=1); not equivalent to standard Mamba '
                      'and not used by either offline paper runner.', RuntimeWarning, stacklevel=2)
        block = FallbackScan(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
    # Freeze D = 0 (no skip / appearance bypass)
    with torch.no_grad():
        block.D.zero_()
    block.D.requires_grad_(False)
    return block


class GroupwiseMambaScan(nn.Module):
    """Processes G=32 mutually isolated channel groups through independent Mamba blocks.

    Input features (T, 2048) are split into 32 groups of 64 dimensions each.
    Each group undergoes LayerNorm within its 64-dim subspace and is processed
    by its own dedicated Mamba block without cross-group recurrent state sharing.
    """

    def __init__(
        self,
        input_dim: int = 2048,
        n_groups: int = 32,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 1,
    ) -> None:
        super().__init__()
        if input_dim % n_groups != 0:
            raise ValueError(f"input_dim ({input_dim}) must be divisible by n_groups ({n_groups}).")
        self.input_dim = input_dim
        self.n_groups = n_groups
        self.group_dim = input_dim // n_groups

        # Layer normalization strictly within each 64-dim group
        self.norm = nn.LayerNorm(self.group_dim)
        self.scans = nn.ModuleList([
            create_mamba_block(
                d_model=self.group_dim,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
            )
            for _ in range(n_groups)
        ])

    def allocate_inference_cache(
        self, batch_size: int = 1, dtype: Optional[torch.dtype] = None
    ) -> List[Any]:
        """Allocates recurrent inference states for all 32 groups."""
        caches = []
        for scan in self.scans:
            try:
                cache = scan.allocate_inference_cache(
                    batch_size=batch_size, max_seqlen=1, dtype=dtype
                )
            except TypeError:
                cache = scan.allocate_inference_cache(batch_size, 1)
            caches.append(cache)
        return caches

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Full sequence parallel forward pass.

        Args:
            features: Tensor of shape (T, 2048) or (B, T, 2048).
        Returns:
            Scanned features of shape (T, 2048) or (B, T, 2048).
        """
        has_batch = (features.ndim == 3)
        if not has_batch:
            features = features.unsqueeze(0)  # (1, T, 2048)

        B, T, D = features.shape
        # Split into G=32 groups: (B, T, G, d_group) -> (G, B, T, d_group)
        groups = features.view(B, T, self.n_groups, self.group_dim).permute(2, 0, 1, 3)

        outputs = []
        for g, scan in enumerate(self.scans):
            u_g = self.norm(groups[g])  # (B, T, d_group)
            out_g = scan(u_g)  # (B, T, d_group)
            outputs.append(out_g)

        # Concatenate back to original group ordering: (B, T, 2048)
        stacked = torch.stack(outputs, dim=2)  # (B, T, G, d_group)
        out = stacked.view(B, T, D)
        return out if has_batch else out.squeeze(0)

    def step(
        self, feature_step: torch.Tensor, scan_states: List[Any]
    ) -> Tuple[torch.Tensor, List[Any]]:
        """Single-step recurrent update for online streaming.

        Args:
            feature_step: Tensor of shape (2048,) or (1, 2048).
            scan_states: List of 32 tuples containing (conv_state, ssm_state).
        Returns:
            Tuple of (step_output of shape (2048,), updated_scan_states).
        """
        feat = feature_step.reshape(1, 1, self.input_dim)
        groups = feat.view(1, 1, self.n_groups, self.group_dim).permute(2, 0, 1, 3)

        updated_states = []
        outputs = []
        for g, scan in enumerate(self.scans):
            u_g = self.norm(groups[g])  # (1, 1, d_group)
            state = scan_states[g]
            conv_state, ssm_state = state[0], state[1]
            out_g, new_conv, new_ssm = scan.step(u_g, conv_state, ssm_state)
            outputs.append(out_g)
            updated_states.append((new_conv, new_ssm))

        stacked = torch.stack(outputs, dim=2)  # (1, 1, G, d_group)
        out = stacked.view(self.input_dim)
        return out, updated_states
