"""Training-only temporal critic for DMD2 motion refinement."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn


@dataclass(frozen=True)
class TemporalCriticConfig:
    input_dim: int = 658
    hybrid_dim: int = 148
    path_dim: int = 3
    heading_dim: int = 2
    width: int = 256
    blocks: int = 4
    expansion: int = 2
    num_diffusion_steps: int = 10


class DilatedTemporalBlock(nn.Module):
    def __init__(self, width: int, dilation: int, expansion: int = 2) -> None:
        super().__init__()
        hidden = width * expansion
        self.norm = nn.LayerNorm(width)
        self.conv1 = nn.Conv1d(
            width,
            hidden,
            kernel_size=3,
            padding=dilation,
            dilation=dilation,
        )
        self.conv2 = nn.Conv1d(hidden, width, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.norm(x).transpose(1, 2)
        residual = self.conv2(F.gelu(self.conv1(residual))).transpose(1, 2)
        return x + residual


class TemporalMotionCritic(nn.Module):
    """Conditional multi-dilation/multi-scale discriminator over 40 frames.

    The input contains normalized root/body motion plus physical FK joints,
    joint velocity/acceleration/jerk and foot contacts.  Path, heading and
    history are injected as conditioning so adversarial training cannot ignore
    the requested route.
    This network is training-only and is never exported to WebGPU.
    """

    def __init__(self, config: TemporalCriticConfig = TemporalCriticConfig()) -> None:
        super().__init__()
        self.config = config
        self.motion_proj = nn.Linear(config.input_dim, config.width)
        self.history_proj = nn.Linear(config.hybrid_dim, config.width)
        self.heading_proj = nn.Linear(config.heading_dim + 1, config.width)
        self.path_proj = nn.Linear(config.path_dim, config.width)
        self.path_summary = nn.Linear(config.width * 2, config.width)
        self.timestep_embedding = nn.Embedding(
            config.num_diffusion_steps, config.width
        )
        self.condition_norm = nn.LayerNorm(config.width)
        self.blocks = nn.ModuleList(
            [
                DilatedTemporalBlock(
                    config.width,
                    dilation=2**index,
                    expansion=config.expansion,
                )
                for index in range(config.blocks)
            ]
        )
        # mean/max at native, /2 and /4 scales, plus the condition vector.
        self.head = nn.Sequential(
            nn.LayerNorm(config.width * 7),
            nn.Linear(config.width * 7, config.width * 2),
            nn.GELU(),
            nn.Linear(config.width * 2, 1),
        )

    def _condition(
        self,
        history_hybrid: torch.Tensor,
        path_condition: torch.Tensor,
        first_heading: torch.Tensor,
        has_history: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        valid = (path_condition[..., 2:3] > 0.5).to(path_condition.dtype)
        path_hidden = F.gelu(self.path_proj(path_condition)) * valid
        denominator = valid.sum(dim=1).clamp_min(1.0)
        path_mean = path_hidden.sum(dim=1) / denominator
        masked_for_max = path_hidden.masked_fill(valid == 0, -torch.inf)
        path_max = masked_for_max.amax(dim=1)
        path_max = torch.where(torch.isfinite(path_max), path_max, torch.zeros_like(path_max))
        path_summary = self.path_summary(torch.cat([path_mean, path_max], dim=-1))
        history = self.history_proj(history_hybrid[:, 0])
        heading = self.heading_proj(
            torch.cat([first_heading, has_history.reshape(-1, 1)], dim=-1)
        )
        timestep = self.timestep_embedding(
            timesteps.reshape(-1).to(device=history.device, dtype=torch.long)
        )
        return self.condition_norm(history + heading + path_summary + timestep)

    def forward(
        self,
        motion_features: torch.Tensor,
        history_hybrid: torch.Tensor,
        path_condition: torch.Tensor,
        first_heading: torch.Tensor,
        has_history: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        condition = self._condition(
            history_hybrid,
            path_condition,
            first_heading,
            has_history,
            timesteps,
        )
        hidden = F.gelu(self.motion_proj(motion_features)) + condition.unsqueeze(1)
        for block in self.blocks:
            hidden = block(hidden)

        hidden_channels = hidden.transpose(1, 2)
        summaries = []
        for factor in (1, 2, 4):
            scaled = (
                hidden_channels
                if factor == 1
                else F.avg_pool1d(hidden_channels, kernel_size=factor, stride=factor)
            )
            summaries.extend([scaled.mean(dim=-1), scaled.amax(dim=-1)])
        packed = torch.cat([*summaries, condition], dim=-1)
        return self.head(packed).reshape(-1)


class ScoreBackboneCriticHead(nn.Module):
    """DMD2 classifier head over the fake-score network's latent features.

    The fake-score backbone has already fused noisy generation tokens with
    history, path, heading and diffusion time.  The head therefore only needs
    to aggregate the ten conditioned generation tokens.  It is training-only
    and never becomes part of the browser generator.
    """

    def __init__(self, width: int, blocks: int = 2, expansion: int = 2) -> None:
        super().__init__()
        if blocks < 0:
            raise ValueError("blocks must be non-negative")
        self.width = int(width)
        self.blocks = nn.ModuleList(
            [
                DilatedTemporalBlock(
                    self.width,
                    dilation=2**index,
                    expansion=expansion,
                )
                for index in range(blocks)
            ]
        )
        self.head = nn.Sequential(
            nn.LayerNorm(self.width * 2),
            nn.Linear(self.width * 2, self.width),
            nn.GELU(),
            nn.Linear(self.width, 1),
        )

    def forward(self, generation_features: torch.Tensor) -> torch.Tensor:
        if generation_features.ndim != 3:
            raise ValueError(
                "generation_features must have shape [B,T,C], got "
                f"{tuple(generation_features.shape)}"
            )
        if generation_features.shape[-1] != self.width:
            raise ValueError(
                f"expected feature width {self.width}, got "
                f"{generation_features.shape[-1]}"
            )
        hidden = generation_features
        for block in self.blocks:
            hidden = block(hidden)
        pooled = torch.cat([hidden.mean(dim=1), hidden.amax(dim=1)], dim=-1)
        return self.head(pooled).reshape(-1)


class IndependentScoreBackboneCriticHeads(nn.Module):
    """One independently trainable LADD classifier for every teacher tap.

    LADD applies independent discriminator heads to token sequences from
    different frozen diffusion-teacher layers.  Keeping this as one module
    lets DDP execute all heads in a single forward while preserving the tap
    ordering used by loss aggregation and checkpoint conversion.
    """

    expects_feature_map = True

    def __init__(
        self,
        width: int,
        taps: tuple[str, ...],
        blocks: int = 2,
        expansion: int = 2,
    ) -> None:
        super().__init__()
        if not taps:
            raise ValueError("independent critic heads require at least one tap")
        if len(set(taps)) != len(taps):
            raise ValueError("independent critic taps must be unique")
        self.width = int(width)
        self.taps = tuple(taps)
        self.heads = nn.ModuleDict(
            {
                tap: ScoreBackboneCriticHead(
                    self.width,
                    blocks=blocks,
                    expansion=expansion,
                )
                for tap in self.taps
            }
        )

    def forward(
        self, generation_features: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        if tuple(generation_features) != self.taps:
            raise ValueError(
                "independent critic feature keys must exactly match taps: "
                f"expected {self.taps}, got {tuple(generation_features)}"
            )
        return {
            tap: self.heads[tap](generation_features[tap]) for tap in self.taps
        }


def clone_shared_critic_state_for_taps(
    shared_state: dict[str, torch.Tensor],
    taps: tuple[str, ...],
) -> dict[str, torch.Tensor]:
    """Function-preserving shared-head -> independent-head state expansion."""

    if not shared_state:
        raise ValueError("shared critic state must not be empty")
    if not taps or len(set(taps)) != len(taps):
        raise ValueError("critic taps must be non-empty and unique")
    return {
        f"heads.{tap}.{name}": value.clone()
        for tap in taps
        for name, value in shared_state.items()
    }
