"""Compact students for the fixed 4-frame encoder and 11-token decoder."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .common import ResidualGELUBlock, TokenChannelMixerBlock


@dataclass(frozen=True)
class CodecStudentConfig:
    body_dim: int = 325
    local_root_dim: int = 4
    latent_dim: int = 128
    frames_per_token: int = 4
    history_frames: int = 4
    decoder_tokens: int = 11
    encoder_width: int = 512
    encoder_blocks: int = 3
    decoder_width: int = 512
    decoder_blocks: int = 4
    decoder_token_hidden: int = 32
    expansion: int = 2


class HistoryEncoderStudent(nn.Module):
    """Map exactly four normalized body frames to one normalized FSQ token.

    The production continuation path always encodes four frames, so attention
    over a one-token sequence is unnecessary. Requantization stays outside the
    ONNX graph because ORT WebGPU does not currently cover Round.
    """

    def __init__(self, config: CodecStudentConfig = CodecStudentConfig()) -> None:
        super().__init__()
        self.config = config
        input_dim = config.history_frames * config.body_dim
        self.input_proj = nn.Linear(input_dim, config.encoder_width)
        self.blocks = nn.ModuleList(
            [
                ResidualGELUBlock(config.encoder_width, expansion=config.expansion)
                for _ in range(config.encoder_blocks)
            ]
        )
        self.output_norm = nn.LayerNorm(config.encoder_width)
        self.output_proj = nn.Linear(config.encoder_width, config.latent_dim)

    def forward(self, normalized_body: torch.Tensor) -> torch.Tensor:
        batch = normalized_body.shape[0]
        x = normalized_body.reshape(batch, self.config.history_frames * self.config.body_dim)
        x = torch.nn.functional.gelu(self.input_proj(x))
        for block in self.blocks:
            x = block(x)
        return self.output_proj(self.output_norm(x)).unsqueeze(1)


class MotionDecoderStudent(nn.Module):
    """Decode 11 latent tokens plus 44 local-root frames to normalized body motion.

    Slot zero is the real history token for continuation and an invalid dummy
    token for initial generation. This gives the browser one static graph while
    preserving the original 10-generated-token window in both modes.
    """

    def __init__(self, config: CodecStudentConfig = CodecStudentConfig()) -> None:
        super().__init__()
        self.config = config
        per_token_root_dim = config.frames_per_token * config.local_root_dim
        self.input_proj = nn.Linear(config.latent_dim + per_token_root_dim, config.decoder_width)
        self.blocks = nn.ModuleList(
            [
                TokenChannelMixerBlock(
                    num_tokens=config.decoder_tokens,
                    width=config.decoder_width,
                    token_hidden=config.decoder_token_hidden,
                    expansion=config.expansion,
                )
                for _ in range(config.decoder_blocks)
            ]
        )
        self.output_norm = nn.LayerNorm(config.decoder_width)
        self.output_proj = nn.Linear(
            config.decoder_width,
            config.frames_per_token * config.body_dim,
        )

    def forward(
        self,
        latent_tokens: torch.Tensor,
        local_root: torch.Tensor,
        token_valid: torch.Tensor,
    ) -> torch.Tensor:
        batch = latent_tokens.shape[0]
        root_by_token = local_root.reshape(
            batch,
            self.config.decoder_tokens,
            self.config.frames_per_token * self.config.local_root_dim,
        )
        valid = token_valid.to(dtype=latent_tokens.dtype)
        x = torch.cat([latent_tokens, root_by_token], dim=-1)
        x = torch.nn.functional.gelu(self.input_proj(x)) * valid.unsqueeze(-1)
        for block in self.blocks:
            x = block(x, valid)
        body_by_token = self.output_proj(self.output_norm(x)) * valid.unsqueeze(-1)
        return body_by_token.reshape(
            batch,
            self.config.decoder_tokens * self.config.frames_per_token,
            self.config.body_dim,
        )
