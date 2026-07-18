"""Small blocks chosen to export as a few WebGPU-friendly GEMM kernels."""

from __future__ import annotations

import torch
from torch import nn


class ResidualGELUBlock(nn.Module):
    """Pre-norm residual MLP with one fused LayerNorm and two large GEMMs."""

    def __init__(self, width: int, expansion: int = 2) -> None:
        super().__init__()
        hidden = width * expansion
        self.norm = nn.LayerNorm(width)
        self.fc1 = nn.Linear(width, hidden)
        self.fc2 = nn.Linear(hidden, width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.fc2(torch.nn.functional.gelu(self.fc1(self.norm(x))))
        return x + residual


class TemporalSelfAttentionBlock(nn.Module):
    """Pre-norm self-attention plus a compact channel MLP.

    The implementation is deliberately explicit instead of using a fused
    framework attention operator.  Fixed-shape ONNX export then consists of
    MatMul, Softmax, Reshape, Transpose, LayerNormalization and GELU, which are
    the same primitive operator family used by the browser timing harness.
    """

    def __init__(self, width: int, heads: int, expansion: int = 2) -> None:
        super().__init__()
        if width % heads:
            raise ValueError(f"width ({width}) must be divisible by heads ({heads})")
        self.width = width
        self.heads = heads
        self.head_dim = width // heads
        self.scale = self.head_dim**-0.5
        self.attention_norm = nn.LayerNorm(width)
        self.qkv = nn.Linear(width, width * 3)
        self.attention_out = nn.Linear(width, width)
        self.channel = ResidualGELUBlock(width, expansion=expansion)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, tokens, _ = x.shape
        qkv = self.qkv(self.attention_norm(x)).reshape(
            batch,
            tokens,
            3,
            self.heads,
            self.head_dim,
        )
        q = qkv[:, :, 0].transpose(1, 2)
        k = qkv[:, :, 1].transpose(1, 2)
        v = qkv[:, :, 2].transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        weights = torch.softmax(scores, dim=-1)
        attended = torch.matmul(weights, v).transpose(1, 2).reshape(batch, tokens, self.width)
        x = x + self.attention_out(attended)
        return self.channel(x)


class TokenChannelMixerBlock(nn.Module):
    """Mix a fixed short time axis once, then spend most FLOPs in channel GEMMs."""

    def __init__(self, num_tokens: int, width: int, token_hidden: int, expansion: int = 2) -> None:
        super().__init__()
        self.token_norm = nn.LayerNorm(width)
        self.token_fc1 = nn.Linear(num_tokens, token_hidden)
        self.token_fc2 = nn.Linear(token_hidden, num_tokens)
        self.channel_block = ResidualGELUBlock(width, expansion=expansion)

    def forward(self, x: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        valid_3d = valid.unsqueeze(-1)
        token_input = self.token_norm(x).transpose(1, 2)
        token_delta = self.token_fc2(torch.nn.functional.gelu(self.token_fc1(token_input))).transpose(1, 2)
        x = (x + token_delta) * valid_3d
        return self.channel_block(x) * valid_3d
