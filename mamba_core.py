"""Mamba State Space Model — ported from SSM-MetaRL-TestCompute.

Provides MambaSSM and MambaBlockFallback (pure-PyTorch) for use as
the backbone of the SSMStabilityAgent in the heterogeneous swarm.

Origin: https://github.com/sunghunkwag/SSM-MetaRL-TestCompute/blob/main/core/ssm_mamba.py
Complexity: O(T·d) with official mamba-ssm, O(T·d·N) with fallback.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import math
import logging
from typing import Tuple, Optional, Dict, Any

logger = logging.getLogger(__name__)

# Attempt to import official Mamba
_MAMBA_AVAILABLE = False
try:
    from mamba_ssm import Mamba
    _MAMBA_AVAILABLE = True
    logger.info("Official mamba-ssm library loaded successfully.")
except ImportError:
    logger.warning(
        "mamba-ssm not installed. Using pure-PyTorch Mamba fallback. "
        "Install with: pip install mamba-ssm causal-conv1d>=1.4.0"
    )


class MambaBlockFallback(nn.Module):
    """Pure-PyTorch fallback implementation of a single Mamba block.

    Implements the selective scan mechanism using standard PyTorch operations:
        1. Input projection with expansion
        2. 1D depthwise convolution
        3. Input-dependent selective scan (SSM)
        4. Output gating with SiLU activation

    Works on CPU. Functionally equivalent to official Mamba but slower on GPU.
    """

    def __init__(self, d_model: int, d_state: int = 16,
                 d_conv: int = 4, expand: int = 2):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)

        # Input projection: projects to 2 * d_inner (for x and z branches)
        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=False)

        # 1D depthwise convolution
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner, out_channels=self.d_inner,
            kernel_size=d_conv, bias=True, groups=self.d_inner,
            padding=d_conv - 1,
        )

        # SSM parameters projection
        self.x_proj = nn.Linear(
            self.d_inner, self.d_state + self.d_state + self.d_inner, bias=False
        )

        # dt (delta) projection: low-rank factorization
        self.dt_proj = nn.Linear(self.d_inner, self.d_inner, bias=True)

        # SSM state matrix A (log-spaced initialization)
        A = torch.arange(1, self.d_state + 1, dtype=torch.float32)
        A = A.unsqueeze(0).expand(self.d_inner, -1)
        self.A_log = nn.Parameter(torch.log(A))

        # D parameter (skip connection)
        self.D = nn.Parameter(torch.ones(self.d_inner))

        # Output projection
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=False)

    def forward(self, x: torch.Tensor,
                hidden_state: Optional[torch.Tensor] = None
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass. Returns (output, next_hidden_state)."""
        batch, seq_len, _ = x.shape

        xz = self.in_proj(x)
        x_branch, z = xz.chunk(2, dim=-1)

        # 1D convolution (causal)
        x_conv = x_branch.transpose(1, 2)
        x_conv = self.conv1d(x_conv)[:, :, :seq_len]
        x_conv = x_conv.transpose(1, 2)
        x_conv = F.silu(x_conv)

        # SSM parameter computation (input-dependent / selective)
        x_ssm_proj = self.x_proj(x_conv)
        B_input = x_ssm_proj[:, :, :self.d_state]
        C_input = x_ssm_proj[:, :, self.d_state:2 * self.d_state]
        dt_input = x_ssm_proj[:, :, 2 * self.d_state:]

        dt = self.dt_proj(dt_input)
        dt = F.softplus(dt)

        A = -torch.exp(self.A_log)

        y, next_h = self._selective_scan(x_conv, dt, A, B_input, C_input, hidden_state)

        y = y + x_conv * self.D.unsqueeze(0).unsqueeze(0)
        y = y * F.silu(z)
        output = self.out_proj(y)
        return output, next_h

    def _selective_scan(self, u, delta, A, B, C, h_init=None):
        """Selective scan implementation with state propagation."""
        batch, seq_len, d_inner = u.shape
        n = A.shape[1]

        delta_A = torch.exp(delta.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0))
        delta_B_u = delta.unsqueeze(-1) * B.unsqueeze(2) * u.unsqueeze(-1)

        if h_init is None:
            h = torch.zeros(batch, d_inner, n, device=u.device, dtype=u.dtype)
        else:
            h = h_init

        outputs = []
        for t in range(seq_len):
            h = delta_A[:, t] * h + delta_B_u[:, t]
            y_t = torch.einsum("bdn,bdn->bd", h, C[:, t].unsqueeze(1).expand_as(h))
            outputs.append(y_t)

        return torch.stack(outputs, dim=1), h


class MambaSSM(nn.Module):
    """Mamba-based State Space Model for the SSM-Mamba Swarm.

    Architecture:
        Input Projection:  Linear(input_dim -> d_model)
        Mamba Block:       Mamba(d_model, d_state, d_conv, expand)
        Layer Norm:        LayerNorm(d_model)
        Output Projection: Linear(d_model -> output_dim)

    Accepts both single-step (B, D) and sequence (B, T, D) inputs.
    """

    def __init__(self, state_dim: int = 16, input_dim: int = 32,
                 output_dim: int = 32, d_model: int = 64,
                 d_conv: int = 4, expand: int = 2, device: str = "cpu"):
        super().__init__()
        self.state_dim = state_dim
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.d_model = d_model
        self.d_conv = d_conv
        self.expand = expand
        self.device = device

        self.input_projection = nn.Linear(input_dim, d_model)
        self.norm = nn.LayerNorm(d_model)

        if _MAMBA_AVAILABLE and device != "cpu":
            self.mamba_block = Mamba(
                d_model=d_model, d_state=state_dim,
                d_conv=d_conv, expand=expand,
            )
            self._using_official_mamba = True
        else:
            self.mamba_block = MambaBlockFallback(
                d_model=d_model, d_state=state_dim,
                d_conv=d_conv, expand=expand,
            )
            self._using_official_mamba = False

        self.output_projection = nn.Linear(d_model, output_dim)
        self.to(device)

    def init_hidden(self, batch_size: int = 1) -> Optional[torch.Tensor]:
        """Initialize hidden state. Returns None for official Mamba."""
        if not self._using_official_mamba:
            d_inner = int(self.expand * self.d_model)
            return torch.zeros(batch_size, d_inner, self.state_dim, device=self.device)
        return None

    def forward(self, x: torch.Tensor,
                hidden_state: Optional[torch.Tensor] = None
                ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass. Supports single-step (B, D) and sequence (B, T, D)."""
        single_step = x.dim() == 2
        if single_step:
            x = x.unsqueeze(1)

        x = self.input_projection(x)
        x = self.norm(x)

        if not self._using_official_mamba:
            x, next_hidden = self.mamba_block(x, hidden_state)
        else:
            x = self.mamba_block(x)
            next_hidden = None

        output = self.output_projection(x)
        if single_step:
            output = output.squeeze(1)

        return output, next_hidden

    def get_complexity_info(self) -> Dict[str, Any]:
        """Get model complexity metrics."""
        total_params = sum(p.numel() for p in self.parameters())
        return {
            "model_type": "MambaSSM",
            "using_official_mamba": self._using_official_mamba,
            "total_params": total_params,
            "d_model": self.d_model,
            "d_state": self.state_dim,
            "complexity": "O(T·d)" if self._using_official_mamba else "O(T·d·N)",
        }

    def save(self, path: str) -> None:
        """Save model checkpoint."""
        dir_name = os.path.dirname(path)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)
        torch.save({
            "state_dict": self.state_dict(),
            "config": {
                "state_dim": self.state_dim, "input_dim": self.input_dim,
                "output_dim": self.output_dim, "d_model": self.d_model,
                "d_conv": self.d_conv, "expand": self.expand, "device": self.device,
            },
        }, path)

    @staticmethod
    def load(path: str, device: Optional[str] = None) -> "MambaSSM":
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location="cpu")
        config = checkpoint["config"]
        if device is not None:
            config["device"] = device
        model = MambaSSM(**config)
        model.load_state_dict(checkpoint["state_dict"])
        return model

    def __repr__(self) -> str:
        backend = "official" if self._using_official_mamba else "fallback"
        return (f"MambaSSM(in={self.input_dim}, out={self.output_dim}, "
                f"d_model={self.d_model}, d_state={self.state_dim}, backend={backend})")
