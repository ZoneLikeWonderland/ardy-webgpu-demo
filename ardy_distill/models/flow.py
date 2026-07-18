"""One-step conditional rectified-flow students for fixed ARDY windows."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from ..flow_matching import euler_flow_denoise
from .common import ResidualGELUBlock, TemporalSelfAttentionBlock


@dataclass(frozen=True)
class FlowStudentConfig:
    architecture: str = "temporal_attention_v2"
    generation_tokens: int = 10
    root_dim_per_token: int = 20
    body_latent_dim: int = 128
    history_tokens: int = 1
    path_frames: int = 64
    path_features: int = 3  # normalized local x, normalized local z, validity
    path_frames_per_token: int = 4
    heading_features: int = 2  # cos/sin of the first heading angle
    # Optional per-frame future heading condition: normalized cos, normalized
    # sin, and validity.  It is separate from the root-x/z path so old EMA
    # weights can be expanded with an exactly-zero residual projection.
    heading_condition_features: int = 0
    # The text encoder is external to the student.  The final text-aware model
    # consumes one precomputed pooled FLUX.2/Qwen3 feature per prompt.
    text_feature_dim: int = 0
    # Parameter-free normalization preserves the complete 7680D feature while
    # controlling the large native scale of concatenated FLUX.2/Qwen states.
    text_norm_eps: float = 1.0e-6
    width: int = 384
    heads: int = 6
    trunk_blocks: int = 4
    body_blocks: int = 2
    expansion: int = 2
    root_smoothing_passes: int = 0
    root_projection_kind: str = "binomial"
    root_control_points: int = 10

    @property
    def hybrid_dim(self) -> int:
        return self.root_dim_per_token + self.body_latent_dim

    @property
    def path_tokens(self) -> int:
        if self.path_frames % self.path_frames_per_token:
            raise ValueError("path_frames must be divisible by path_frames_per_token")
        return self.path_frames // self.path_frames_per_token

    @property
    def context_tokens(self) -> int:
        return 1 + self.history_tokens + self.path_tokens

    @property
    def sequence_tokens(self) -> int:
        return self.context_tokens + self.generation_tokens


class FlatOneStepFlowStudent(nn.Module):
    """Deprecated flattened v1 retained only to identify old audit artifacts."""

    def __init__(self, width: int = 768, trunk_blocks: int = 6, body_blocks: int = 2) -> None:
        super().__init__()
        flat_input_dim = 10 * 148 + 148 + 64 * 3 + 2 + 1 + 1
        self.input_proj = nn.Linear(flat_input_dim, width)
        self.trunk = nn.ModuleList(
            [ResidualGELUBlock(width, expansion=2) for _ in range(trunk_blocks)]
        )
        self.root_head = nn.Linear(width, 10 * 20)
        self.body_input = nn.Linear(width + 10 * 20, width)
        self.body_refiner = nn.ModuleList(
            [ResidualGELUBlock(width, expansion=2) for _ in range(body_blocks)]
        )
        self.body_head = nn.Linear(width, 10 * 128)

    def forward(
        self,
        noisy_generation: torch.Tensor,
        history_hybrid: torch.Tensor,
        path_condition: torch.Tensor,
        first_heading: torch.Tensor,
        has_history: torch.Tensor,
        flow_time: torch.Tensor,
    ) -> torch.Tensor:
        batch = noisy_generation.shape[0]
        packed = torch.cat(
            [
                noisy_generation.reshape(batch, -1),
                history_hybrid.reshape(batch, -1),
                path_condition.reshape(batch, -1),
                first_heading.reshape(batch, 2),
                has_history.reshape(batch, 1),
                flow_time.reshape(batch, 1),
            ],
            dim=-1,
        )
        hidden = torch.nn.functional.gelu(self.input_proj(packed))
        for block in self.trunk:
            hidden = block(hidden)
        root_flat = self.root_head(hidden)
        body_hidden = torch.nn.functional.gelu(self.body_input(torch.cat([hidden, root_flat], dim=-1)))
        for block in self.body_refiner:
            body_hidden = block(body_hidden)
        return torch.cat(
            [root_flat.reshape(batch, 10, 20), self.body_head(body_hidden).reshape(batch, 10, 128)],
            dim=-1,
        )


class OneStepFlowStudent(nn.Module):
    """Structured 28-token root-then-body conditional rectified-flow student.

    Training uses ``x_t=(1-t)*x0+t*noise`` and predicts
    ``velocity=noise-x0``. Production passes ``t=1`` once and computes
    ``x0=noise-velocity``; CFG has already been distilled into the weights.

    Sequence layout is one global token, one history token, sixteen four-frame
    path tokens, and ten noisy generation tokens.  Root velocity is predicted
    first and injected back into the generation tokens before body refinement.
    """

    def __init__(self, config: FlowStudentConfig = FlowStudentConfig()) -> None:
        super().__init__()
        if config.root_smoothing_passes < 0:
            raise ValueError("root_smoothing_passes must be non-negative")
        if config.root_projection_kind not in {"binomial", "cubic_controls"}:
            raise ValueError(
                "root_projection_kind must be 'binomial' or 'cubic_controls'"
            )
        if not 2 <= config.root_control_points <= 40:
            raise ValueError("root_control_points must be within [2, 40]")
        if config.text_norm_eps <= 0:
            raise ValueError("text_norm_eps must be positive")
        self.config = config
        self.register_buffer(
            "root_projection_basis",
            build_root_projection_basis(
                kind=config.root_projection_kind,
                passes=config.root_smoothing_passes,
                control_points=config.root_control_points,
            ),
            persistent=False,
        )
        self.global_proj = nn.Linear(config.heading_features + 2, config.width)
        self.text_proj = (
            nn.Linear(config.text_feature_dim, config.width, bias=False)
            if config.text_feature_dim > 0
            else None
        )
        self.history_proj = nn.Linear(config.hybrid_dim, config.width)
        self.path_proj = nn.Linear(
            config.path_frames_per_token * config.path_features,
            config.width,
        )
        self.heading_proj = (
            nn.Linear(
                config.path_frames_per_token * config.heading_condition_features,
                config.width,
            )
            if config.heading_condition_features > 0
            else None
        )
        self.generation_proj = nn.Linear(config.hybrid_dim, config.width)
        self.position = nn.Parameter(torch.zeros(1, config.sequence_tokens, config.width))
        self.global_type = nn.Parameter(torch.zeros(1, 1, config.width))
        self.history_type = nn.Parameter(torch.zeros(1, 1, config.width))
        self.path_type = nn.Parameter(torch.zeros(1, 1, config.width))
        self.generation_type = nn.Parameter(torch.zeros(1, 1, config.width))
        self.input_norm = nn.LayerNorm(config.width)
        self.trunk = nn.ModuleList(
            [
                TemporalSelfAttentionBlock(
                    config.width,
                    heads=config.heads,
                    expansion=config.expansion,
                )
                for _ in range(config.trunk_blocks)
            ]
        )
        self.root_head = nn.Linear(config.width, config.root_dim_per_token)
        self.root_injection = nn.Linear(config.root_dim_per_token, config.width)
        self.body_refiner = nn.ModuleList(
            [
                TemporalSelfAttentionBlock(
                    config.width,
                    heads=config.heads,
                    expansion=config.expansion,
                )
                for _ in range(config.body_blocks)
            ]
        )
        self.body_head = nn.Linear(config.width, config.body_latent_dim)

        nn.init.normal_(self.position, std=0.02)
        nn.init.normal_(self.global_type, std=0.02)
        nn.init.normal_(self.history_type, std=0.02)
        nn.init.normal_(self.path_type, std=0.02)
        nn.init.normal_(self.generation_type, std=0.02)
        # A text-aware student is always warm-started from an existing
        # path-only EMA.  Zero initialization makes that load functionally
        # identical until the new projection receives gradients.
        if self.text_proj is not None:
            nn.init.zeros_(self.text_proj.weight)
        if self.heading_proj is not None:
            nn.init.zeros_(self.heading_proj.weight)
            nn.init.zeros_(self.heading_proj.bias)

    def forward_with_features(
        self,
        noisy_generation: torch.Tensor,
        history_hybrid: torch.Tensor,
        path_condition: torch.Tensor,
        first_heading: torch.Tensor,
        has_history: torch.Tensor,
        flow_time: torch.Tensor,
        text_feature: torch.Tensor | None = None,
        heading_condition: torch.Tensor | None = None,
        *,
        feature_tap: str | tuple[str, ...] = "body_final",
    ) -> tuple[torch.Tensor, torch.Tensor | dict[str, torch.Tensor]]:
        """Return the velocity prediction and conditioned generation features.

        DMD2's training-only classifier reads frozen teacher feature taps.
        Every supported tap has shape ``[B, generation_tokens, width]``, so a
        feature-location experiment can retain the same critic parameters and
        complete optimizer state. Passing a tuple returns a dictionary in the
        requested order and permits one shared critic head to supervise several
        levels from one teacher forward. ``body_final`` is the historical
        default; the exported generator output is unchanged for every tap.
        """

        valid_feature_taps = {
            "trunk_final",
            "body_pre",
            "body_mid",
            "body_final",
        }
        requested_taps = (
            (feature_tap,) if isinstance(feature_tap, str) else tuple(feature_tap)
        )
        if not requested_taps:
            raise ValueError("feature_tap tuple must not be empty")
        if len(set(requested_taps)) != len(requested_taps):
            raise ValueError("feature_tap tuple must not contain duplicates")
        invalid_taps = set(requested_taps) - valid_feature_taps
        if invalid_taps:
            raise ValueError(
                f"feature_tap must be one of {sorted(valid_feature_taps)}, "
                f"got {sorted(invalid_taps)!r}"
            )

        batch = noisy_generation.shape[0]
        global_features = torch.cat(
            [
                first_heading.reshape(batch, self.config.heading_features),
                has_history.reshape(batch, 1),
                flow_time.reshape(batch, 1),
            ],
            dim=-1,
        )
        global_token = torch.nn.functional.gelu(self.global_proj(global_features)).unsqueeze(1)
        if self.text_proj is not None:
            if text_feature is None:
                text_feature = global_features.new_zeros(
                    batch,
                    self.config.text_feature_dim,
                )
            else:
                text_feature = text_feature.reshape(batch, -1)
                if text_feature.shape[-1] != self.config.text_feature_dim:
                    raise ValueError(
                        "text_feature last dimension must equal "
                        f"{self.config.text_feature_dim}, got {text_feature.shape[-1]}"
                    )
            # Use FP32 only for the reduction, then project in the caller's
            # BF16/FP16 dtype. An unconditional all-zero feature stays zero.
            text_feature_float = text_feature.float()
            text_feature = (
                text_feature_float
                * torch.rsqrt(
                    text_feature_float.square().mean(dim=-1, keepdim=True)
                    + self.config.text_norm_eps
                )
            ).to(dtype=text_feature.dtype)
            global_token = global_token + self.text_proj(text_feature).unsqueeze(1)
        global_token = global_token + self.global_type
        history_token = torch.nn.functional.gelu(self.history_proj(history_hybrid))
        history_token = history_token + self.history_type
        path_values = path_condition.reshape(
            batch,
            self.config.path_tokens,
            self.config.path_frames_per_token * self.config.path_features,
        )
        path_hidden = self.path_proj(path_values)
        if self.heading_proj is not None:
            if heading_condition is None:
                heading_condition = path_condition.new_zeros(
                    batch,
                    self.config.path_frames,
                    self.config.heading_condition_features,
                )
            if heading_condition.shape != (
                batch,
                self.config.path_frames,
                self.config.heading_condition_features,
            ):
                raise ValueError(
                    "heading_condition must have shape "
                    f"[{batch}, {self.config.path_frames}, "
                    f"{self.config.heading_condition_features}], got "
                    f"{list(heading_condition.shape)}"
                )
            heading_values = heading_condition.reshape(
                batch,
                self.config.path_tokens,
                self.config.path_frames_per_token
                * self.config.heading_condition_features,
            )
            path_hidden = path_hidden + self.heading_proj(heading_values)
        path_tokens = torch.nn.functional.gelu(path_hidden) + self.path_type
        generation_tokens = torch.nn.functional.gelu(self.generation_proj(noisy_generation))
        generation_tokens = generation_tokens + self.generation_type
        hidden = torch.cat(
            [global_token, history_token, path_tokens, generation_tokens],
            dim=1,
        )
        hidden = self.input_norm(hidden + self.position)
        for block in self.trunk:
            hidden = block(hidden)

        generation_hidden = hidden[:, -self.config.generation_tokens :]
        root_velocity = self.root_head(generation_hidden)
        root_context = self.root_injection(root_velocity)
        root_prefix = hidden[:, : self.config.context_tokens] * 0
        body_hidden = hidden + torch.cat([root_prefix, root_context], dim=1)
        body_pre_features = body_hidden[:, -self.config.generation_tokens :]
        body_mid_features = body_pre_features
        middle_block = max(1, len(self.body_refiner) // 2)
        for index, block in enumerate(self.body_refiner):
            body_hidden = block(body_hidden)
            if index + 1 == middle_block:
                body_mid_features = body_hidden[
                    :, -self.config.generation_tokens :
                ]
        body_final_features = body_hidden[:, -self.config.generation_tokens :]
        body_velocity = self.body_head(body_final_features)
        prediction = torch.cat([root_velocity, body_velocity], dim=-1)
        all_generation_features = {
            "trunk_final": generation_hidden,
            "body_pre": body_pre_features,
            "body_mid": body_mid_features,
            "body_final": body_final_features,
        }
        generation_features: torch.Tensor | dict[str, torch.Tensor]
        if isinstance(feature_tap, str):
            generation_features = all_generation_features[feature_tap]
        else:
            generation_features = {
                tap: all_generation_features[tap] for tap in requested_taps
            }
        return prediction, generation_features

    def forward(
        self,
        noisy_generation: torch.Tensor,
        history_hybrid: torch.Tensor,
        path_condition: torch.Tensor,
        first_heading: torch.Tensor,
        has_history: torch.Tensor,
        flow_time: torch.Tensor,
        text_feature: torch.Tensor | None = None,
        heading_condition: torch.Tensor | None = None,
        *,
        return_features: bool = False,
        feature_tap: str | tuple[str, ...] = "body_final",
    ) -> torch.Tensor | tuple[
        torch.Tensor, torch.Tensor | dict[str, torch.Tensor]
    ]:
        prediction, features = self.forward_with_features(
            noisy_generation,
            history_hybrid,
            path_condition,
            first_heading,
            has_history,
            flow_time,
            text_feature,
            heading_condition,
            feature_tap=feature_tap,
        )
        if return_features:
            return prediction, features
        return prediction

    def denoise_steps(
        self,
        noise: torch.Tensor,
        history_hybrid: torch.Tensor,
        path_condition: torch.Tensor,
        first_heading: torch.Tensor,
        has_history: torch.Tensor,
        *,
        steps: int,
        text_feature: torch.Tensor | None = None,
        heading_condition: torch.Tensor | None = None,
    ) -> torch.Tensor:
        clean = euler_flow_denoise(
            self,
            noise,
            history_hybrid,
            path_condition,
            first_heading,
            has_history,
            steps=steps,
            text_feature=text_feature,
            heading_condition=heading_condition,
        )
        return project_root_trajectory(
            clean,
            self.config.root_smoothing_passes,
            kind=self.config.root_projection_kind,
            control_points=self.config.root_control_points,
            basis=self.root_projection_basis,
        )

    def denoise_once(
        self,
        noise: torch.Tensor,
        history_hybrid: torch.Tensor,
        path_condition: torch.Tensor,
        first_heading: torch.Tensor,
        has_history: torch.Tensor,
        text_feature: torch.Tensor | None = None,
        heading_condition: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Backward-compatible NFE=1 wrapper used by existing exports."""

        return self.denoise_steps(
            noise,
            history_hybrid,
            path_condition,
            first_heading,
            has_history,
            steps=1,
            text_feature=text_feature,
            heading_condition=heading_condition,
        )


def _binomial_projection_basis(frames: int, passes: int) -> torch.Tensor:
    """Return the dense linear map equivalent to repeated binomial filtering."""

    basis = torch.eye(frames, dtype=torch.float32)
    for _ in range(passes):
        padded = torch.cat(
            [basis[:1], basis[:1], basis, basis[-1:], basis[-1:]], dim=0
        )
        filtered = (
            padded[0:frames]
            + 4.0 * padded[1 : frames + 1]
            + 6.0 * padded[2 : frames + 2]
            + 4.0 * padded[3 : frames + 3]
            + padded[4 : frames + 4]
        ) / 16.0
        basis = torch.cat([basis[:1], filtered[1:-1], basis[-1:]], dim=0)
    return basis


def _cubic_control_projection_basis(frames: int, control_points: int) -> torch.Tensor:
    """Build one dense cubic reconstruction from sparse predicted root frames.

    The controls include both endpoints. Centered finite-difference tangents
    preserve every linear trajectory exactly and make adjacent cubic-Hermite
    segments C1-continuous. Only control-frame columns are nonzero in the
    returned ``[frames, frames]`` map, so ONNX/WebGPU needs one constant MatMul
    rather than a chain of Gather/Slice/elementwise kernels.
    """

    positions = torch.linspace(0, frames - 1, control_points).round().to(torch.long)
    if torch.unique(positions).numel() != control_points:
        raise ValueError("rounded root control-frame positions must be unique")
    positions_f = positions.to(torch.float32)
    derivative = torch.zeros(control_points, control_points, dtype=torch.float32)
    derivative[0, 0] = -1.0 / (positions_f[1] - positions_f[0])
    derivative[0, 1] = -derivative[0, 0]
    derivative[-1, -2] = -1.0 / (positions_f[-1] - positions_f[-2])
    derivative[-1, -1] = -derivative[-1, -2]
    for index in range(1, control_points - 1):
        scale = 1.0 / (positions_f[index + 1] - positions_f[index - 1])
        derivative[index, index - 1] = -scale
        derivative[index, index + 1] = scale

    control_basis = torch.zeros(frames, control_points, dtype=torch.float32)
    segment = 0
    for frame in range(frames):
        while segment + 1 < control_points - 1 and frame > int(positions[segment + 1]):
            segment += 1
        left = segment
        right = segment + 1
        width = positions_f[right] - positions_f[left]
        phase = (float(frame) - positions_f[left]) / width
        phase2 = phase * phase
        phase3 = phase2 * phase
        h00 = 2.0 * phase3 - 3.0 * phase2 + 1.0
        h10 = phase3 - 2.0 * phase2 + phase
        h01 = -2.0 * phase3 + 3.0 * phase2
        h11 = phase3 - phase2
        control_basis[frame, left] += h00
        control_basis[frame] += h10 * width * derivative[left]
        control_basis[frame, right] += h01
        control_basis[frame] += h11 * width * derivative[right]

    basis = torch.zeros(frames, frames, dtype=torch.float32)
    basis[:, positions] = control_basis
    return basis


def build_root_projection_basis(
    *, kind: str, passes: int, control_points: int, frames: int = 40
) -> torch.Tensor:
    """Build the constant deployment matrix for one root projection variant."""

    if passes < 0:
        raise ValueError("passes must be non-negative")
    if kind == "binomial":
        return _binomial_projection_basis(frames, passes)
    if kind == "cubic_controls":
        if not 2 <= control_points <= frames:
            raise ValueError("control_points must be within [2, frames]")
        return _cubic_control_projection_basis(frames, control_points)
    raise ValueError(f"unsupported root projection kind: {kind}")


def project_root_trajectory(
    clean_generation: torch.Tensor,
    passes: int,
    *,
    kind: str = "binomial",
    control_points: int = 10,
    basis: torch.Tensor | None = None,
) -> torch.Tensor:
    """Apply a fixed temporal projection to the 40-frame global-root output.

    The one-step model otherwise predicts all forty normalized global-root
    frames independently.  Repeated ``[1, 4, 6, 4, 1] / 16`` filtering removes
    frame-scale modes that are absent from the released teacher.  The first and
    last frames are retained on every pass: history-seam and sparse-waypoint
    objectives therefore keep direct control of the two boundary values.

    ``binomial`` retains the existing repeated smoothing semantics.
    ``cubic_controls`` reconstructs all frames from sparse predicted controls,
    structurally removing frame-scale degrees of freedom. Both are
    zero-parameter parts of the endpoint graph, not renderer-side filters.
    """

    if kind == "binomial" and passes <= 0:
        return clean_generation
    if clean_generation.ndim != 3 or clean_generation.shape[1:] != (10, 148):
        raise ValueError(
            "clean_generation must have shape [B,10,148], got "
            f"{tuple(clean_generation.shape)}"
        )
    batch = clean_generation.shape[0]
    root = clean_generation[..., :20].reshape(batch, 40, 5)
    if basis is None:
        basis = build_root_projection_basis(
            kind=kind,
            passes=passes,
            control_points=control_points,
        )
    if tuple(basis.shape) != (40, 40):
        raise ValueError(
            "root projection basis must have shape [40,40], got "
            f"{tuple(basis.shape)}"
        )
    root = torch.matmul(basis.to(device=root.device, dtype=root.dtype), root)
    return torch.cat(
        [root.reshape(batch, 10, 20), clean_generation[..., 20:]],
        dim=-1,
    )
