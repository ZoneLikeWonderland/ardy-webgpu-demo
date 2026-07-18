"""Faithful path-only runtime for the compact ARDY student modules.

The browser graphs intentionally contain only the learned modules.  This file
keeps the surrounding PyTorch reference implementation in one place so rollout
tests and the WebGPU frontend can be compared against the same semantics:

* explicit four-frame history is recentered exactly like ``Ardy``;
* encoder and generated body latents are requantized with the released FSQ;
* world-space root constraints are translated into the current local frame;
* the static 11-token decoder graph uses a dummy leading token for an initial
  (history-free) window and a real history token for continuation windows.
"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import torch
from torch import nn

from ardy.model.ardy_model import translate_normalized_root_motion

from .losses import FSQRequantizer
from .models import HistoryEncoderStudent, MotionDecoderStudent, OneStepFlowStudent


@dataclass
class StudentWindowOutput:
    """One initial 40-frame or continuation 44-frame student window."""

    explicit_motion: torch.Tensor
    clean_generation: torch.Tensor
    history_hybrid: torch.Tensor
    path_condition: torch.Tensor
    first_heading: torch.Tensor
    has_history: torch.Tensor
    global_translation: torch.Tensor
    decoder_latent: torch.Tensor
    decoder_global_root: torch.Tensor
    decoder_local_root: torch.Tensor
    decoder_token_valid: torch.Tensor
    timings_ms: dict[str, float]


class _StageTimer:
    def __init__(self, device: torch.device, enabled: bool) -> None:
        self.device = device
        self.enabled = enabled
        self.timings_ms: dict[str, float] = {}

    def measure(self, name: str, function):
        if not self.enabled:
            return function()
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        start = perf_counter()
        result = function()
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        self.timings_ms[name] = (perf_counter() - start) * 1000.0
        return result


class StudentArdyRuntime(nn.Module):
    """Compose compact encoder, low-step flow, decoder and fixed ARDY math.

    Learned modules may run in FP32/FP16/BF16.  Motion representation math and
    the returned explicit motion stay FP32, matching the reference evaluator
    and avoiding precision loss in world-coordinate accumulation.
    """

    def __init__(
        self,
        encoder: HistoryEncoderStudent,
        flow: OneStepFlowStudent,
        decoder: MotionDecoderStudent,
        quantizer: FSQRequantizer,
        motion_rep,
        *,
        model_dtype: torch.dtype = torch.float32,
        flow_steps: int = 1,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.flow = flow
        self.decoder = decoder
        self.quantizer = quantizer
        self.motion_rep = motion_rep
        self.model_dtype = model_dtype
        if flow_steps < 1:
            raise ValueError("flow_steps must be positive")
        self.flow_steps = flow_steps

        codec = encoder.config
        decoder_config = decoder.config
        flow_config = flow.config
        if codec.history_frames != 4 or codec.frames_per_token != 4:
            raise ValueError("student runtime requires exactly four history frames")
        if decoder_config.decoder_tokens != 11 or flow_config.generation_tokens != 10:
            raise ValueError("student runtime requires the fixed 1+10-token layout")
        if flow_config.path_frames != 64:
            raise ValueError("student runtime currently requires a 64-frame path condition")
        if flow_config.root_dim_per_token != 4 * self.motion_rep.motion_root_dim:
            raise ValueError("flow root width does not match the motion representation")

    @property
    def device(self) -> torch.device:
        return next(self.flow.parameters()).device

    def _empty_path(self, batch: int) -> torch.Tensor:
        return torch.zeros(batch, 64, 3, device=self.device, dtype=torch.float32)

    def _normalize_constraint_length(
        self,
        tensor: torch.Tensor,
        *,
        batch: int,
        features: int,
        name: str,
    ) -> torch.Tensor:
        if tensor.ndim != 3 or tensor.shape[0] != batch or tensor.shape[2] != features:
            raise ValueError(
                f"{name} must have shape [B,T,{features}], got {tuple(tensor.shape)}"
            )
        if tensor.shape[1] > 64:
            return tensor[:, :64]
        if tensor.shape[1] == 64:
            return tensor
        padding = tensor.new_zeros(batch, 64 - tensor.shape[1], features)
        return torch.cat([tensor, padding], dim=1)

    def prepare_path_condition(
        self,
        *,
        batch: int,
        history_frames: int,
        global_translation: torch.Tensor,
        motion_mask: torch.Tensor | None,
        observed_motion: torch.Tensor | None,
    ) -> torch.Tensor:
        """Apply the released root-constraint preprocessing and pack ``[x,z,valid]``."""

        if motion_mask is None and observed_motion is None:
            return self._empty_path(batch)
        if motion_mask is None or observed_motion is None:
            raise ValueError("motion_mask and observed_motion must either both be set or both be None")

        feature_dim = self.motion_rep.motion_rep_dim
        motion_mask = self._normalize_constraint_length(
            motion_mask.to(device=self.device, dtype=torch.float32),
            batch=batch,
            features=feature_dim,
            name="motion_mask",
        )
        observed_motion = self._normalize_constraint_length(
            observed_motion.to(device=self.device, dtype=torch.float32),
            batch=batch,
            features=feature_dim,
            name="observed_motion",
        )

        current_mask = motion_mask.clone()
        if history_frames:
            current_mask[:, :history_frames] = 0
        observed_root = self.motion_rep.extract_root(observed_motion)
        observed_body = self.motion_rep.extract_body(observed_motion)
        translated_root = translate_normalized_root_motion(
            observed_root,
            -global_translation,
            self.motion_rep,
        )
        current_observed = self.motion_rep.concat_root_body(translated_root, observed_body)
        current_observed = current_observed * current_mask

        path = self._empty_path(batch)
        path[..., :2] = current_observed[..., [0, 2]]
        valid = (current_mask[..., 0] > 0) | (current_mask[..., 2] > 0)
        path[..., 2] = valid.to(dtype=path.dtype)
        path[..., :2] *= path[..., 2:3]
        return path

    def _encode_history(
        self,
        history: torch.Tensor,
        timer: _StageTimer,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if history.ndim != 3 or history.shape[1:] != (4, self.motion_rep.motion_rep_dim):
            raise ValueError(
                "init_history_sequence must have shape "
                f"[B,4,{self.motion_rep.motion_rep_dim}], got {tuple(history.shape)}"
            )
        history = history.to(device=self.device, dtype=torch.float32)
        batch = history.shape[0]
        normalized_body = self.motion_rep.extract_body(history)
        encoded = timer.measure(
            "encoder",
            lambda: self.encoder(normalized_body.to(dtype=self.model_dtype)).float(),
        )
        encoded = timer.measure(
            "encoder_fsq",
            lambda: self.quantizer(encoded, ste=False).float(),
        )

        root = self.motion_rep.extract_root(history)
        center_index = torch.full((batch,), 3, device=self.device, dtype=torch.long)
        local_root, center_position = self.motion_rep.recenter_root_motion(
            root,
            center_index,
            is_normalized=True,
            to_normalize=True,
            return_center_pos=True,
        )
        history_hybrid = torch.cat([local_root.reshape(batch, 1, -1), encoded], dim=-1)
        unnormalized_history = self.motion_rep.unnormalize(history)
        first_heading_angle = self.motion_rep.get_root_heading_angle(unnormalized_history)[:, 0]
        return history_hybrid, center_position, first_heading_angle

    def _decode(
        self,
        *,
        local_hybrid: torch.Tensor,
        global_translation: torch.Tensor,
        has_history: bool,
        timer: _StageTimer,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        batch = local_hybrid.shape[0]
        token_count = 11 if has_history else 10
        expected_tokens = token_count
        if local_hybrid.shape[1] != expected_tokens:
            raise ValueError(
                f"expected {expected_tokens} local tokens, got {local_hybrid.shape[1]}"
            )

        local_root = local_hybrid[..., :20].reshape(batch, token_count * 4, 5).float()
        latent = timer.measure(
            "generation_fsq",
            lambda: self.quantizer(local_hybrid[..., 20:].float(), ste=False).float(),
        )
        global_root = translate_normalized_root_motion(
            local_root,
            global_translation,
            self.motion_rep,
        )
        lengths = torch.full(
            (batch,),
            global_root.shape[1],
            device=self.device,
            dtype=torch.long,
        )
        decoder_local_actual = self.motion_rep.global_root_to_local_root(
            global_root,
            normalized=True,
            lengths=lengths,
        ).float()

        decoder_latent = torch.zeros(batch, 11, 128, device=self.device, dtype=torch.float32)
        decoder_global_root = torch.zeros(batch, 44, 5, device=self.device, dtype=torch.float32)
        decoder_local_root = torch.zeros(batch, 44, 4, device=self.device, dtype=torch.float32)
        decoder_token_valid = torch.ones(batch, 11, device=self.device, dtype=torch.float32)
        if has_history:
            decoder_latent.copy_(latent)
            decoder_global_root.copy_(global_root)
            decoder_local_root.copy_(decoder_local_actual)
        else:
            decoder_latent[:, 1:] = latent
            decoder_global_root[:, 4:] = global_root
            decoder_local_root[:, 4:] = decoder_local_actual
            decoder_token_valid[:, 0] = 0

        decoded_body = timer.measure(
            "decoder",
            lambda: self.decoder(
                decoder_latent.to(dtype=self.model_dtype),
                decoder_local_root.to(dtype=self.model_dtype),
                decoder_token_valid.to(dtype=self.model_dtype),
            ).float(),
        )
        if has_history:
            explicit_motion = self.motion_rep.concat_root_body(global_root, decoded_body)
        else:
            explicit_motion = self.motion_rep.concat_root_body(global_root, decoded_body[:, 4:])
        return (
            explicit_motion,
            decoder_latent,
            decoder_global_root,
            decoder_local_root,
            decoder_token_valid,
        )

    @torch.inference_mode()
    def decode_clean_generation(
        self,
        *,
        clean_generation: torch.Tensor,
        init_history_sequence: torch.Tensor | None = None,
        init_global_translation: torch.Tensor | None = None,
        init_first_heading_angle: torch.Tensor | None = None,
        profile: bool = False,
    ) -> StudentWindowOutput:
        """Decode an externally supplied clean 10-token endpoint.

        This is the oracle-flow codec ablation used by long-rollout evaluation:
        root/body generation tokens come from the exact teacher while history
        still passes through the compact encoder and the whole 11-token body
        sequence passes through the compact decoder.
        """

        clean_generation = clean_generation.to(device=self.device, dtype=torch.float32)
        if clean_generation.ndim != 3 or clean_generation.shape[1:] != (10, 148):
            raise ValueError(
                "clean_generation must have shape [B,10,148], got "
                f"{tuple(clean_generation.shape)}"
            )
        batch = clean_generation.shape[0]
        timer = _StageTimer(self.device, profile)
        total_start = perf_counter()
        if profile and self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

        if init_history_sequence is not None:
            history_hybrid, global_translation, first_heading_angle = self._encode_history(
                init_history_sequence,
                timer,
            )
            if history_hybrid.shape[0] != batch:
                raise ValueError("history and clean-generation batch sizes differ")
            has_history_bool = True
        else:
            global_translation = (
                torch.zeros(batch, 3, device=self.device, dtype=torch.float32)
                if init_global_translation is None
                else init_global_translation.to(device=self.device, dtype=torch.float32)
            )
            first_heading_angle = (
                torch.zeros(batch, device=self.device, dtype=torch.float32)
                if init_first_heading_angle is None
                else init_first_heading_angle.to(device=self.device, dtype=torch.float32)
            )
            history_hybrid = torch.zeros(batch, 1, 148, device=self.device, dtype=torch.float32)
            has_history_bool = False
        if global_translation.shape != (batch, 3):
            raise ValueError("global translation must have shape [B,3]")
        if first_heading_angle.shape != (batch,):
            raise ValueError("first heading angle must have shape [B]")
        if not torch.equal(global_translation[:, 1], torch.zeros_like(global_translation[:, 1])):
            raise ValueError("global translation y component must be exactly zero")

        first_heading = torch.stack(
            [torch.cos(first_heading_angle), torch.sin(first_heading_angle)],
            dim=-1,
        )
        has_history = torch.full(
            (batch, 1),
            float(has_history_bool),
            device=self.device,
            dtype=torch.float32,
        )
        local_hybrid = (
            torch.cat([history_hybrid, clean_generation], dim=1)
            if has_history_bool
            else clean_generation
        )
        (
            explicit_motion,
            decoder_latent,
            decoder_global_root,
            decoder_local_root,
            decoder_token_valid,
        ) = self._decode(
            local_hybrid=local_hybrid,
            global_translation=global_translation,
            has_history=has_history_bool,
            timer=timer,
        )
        if profile:
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            timer.timings_ms["total"] = (perf_counter() - total_start) * 1000.0
        return StudentWindowOutput(
            explicit_motion=explicit_motion,
            clean_generation=clean_generation,
            history_hybrid=history_hybrid,
            path_condition=self._empty_path(batch),
            first_heading=first_heading,
            has_history=has_history,
            global_translation=global_translation,
            decoder_latent=decoder_latent,
            decoder_global_root=decoder_global_root,
            decoder_local_root=decoder_local_root,
            decoder_token_valid=decoder_token_valid,
            timings_ms=timer.timings_ms,
        )

    @torch.inference_mode()
    def step_prepared(
        self,
        *,
        path_condition: torch.Tensor,
        text_feature: torch.Tensor | None = None,
        heading_condition: torch.Tensor | None = None,
        initial_noise: torch.Tensor | None = None,
        init_history_sequence: torch.Tensor | None = None,
        init_global_translation: torch.Tensor | None = None,
        init_first_heading_angle: torch.Tensor | None = None,
        profile: bool = False,
    ) -> StudentWindowOutput:
        """Generate one window from an already-local 64-frame path condition."""

        path_condition = path_condition.to(device=self.device, dtype=torch.float32)
        if path_condition.ndim != 3 or path_condition.shape[1:] != (64, 3):
            raise ValueError(f"path_condition must have shape [B,64,3], got {tuple(path_condition.shape)}")
        batch = path_condition.shape[0]
        if text_feature is not None and text_feature.shape[0] != batch:
            raise ValueError("text_feature and path_condition batch sizes differ")
        if heading_condition is not None and heading_condition.shape[0] != batch:
            raise ValueError("heading_condition and path_condition batch sizes differ")
        timer = _StageTimer(self.device, profile)
        total_start = perf_counter()
        if profile and self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

        if init_history_sequence is not None:
            history_hybrid, global_translation, first_heading_angle = self._encode_history(
                init_history_sequence,
                timer,
            )
            if history_hybrid.shape[0] != batch:
                raise ValueError("history and path batch sizes differ")
            has_history_bool = True
        else:
            if init_global_translation is None:
                global_translation = torch.zeros(batch, 3, device=self.device, dtype=torch.float32)
            else:
                global_translation = init_global_translation.to(device=self.device, dtype=torch.float32)
            if init_first_heading_angle is None:
                first_heading_angle = torch.zeros(batch, device=self.device, dtype=torch.float32)
            else:
                first_heading_angle = init_first_heading_angle.to(device=self.device, dtype=torch.float32)
            history_hybrid = torch.zeros(batch, 1, 148, device=self.device, dtype=torch.float32)
            has_history_bool = False

        if global_translation.shape != (batch, 3):
            raise ValueError(
                f"global translation must have shape [B,3], got {tuple(global_translation.shape)}"
            )
        if first_heading_angle.shape != (batch,):
            raise ValueError(
                f"first heading angle must have shape [B], got {tuple(first_heading_angle.shape)}"
            )
        if not torch.equal(global_translation[:, 1], torch.zeros_like(global_translation[:, 1])):
            raise ValueError("global translation y component must be exactly zero")

        first_heading = torch.stack(
            [torch.cos(first_heading_angle), torch.sin(first_heading_angle)],
            dim=-1,
        )
        has_history = torch.full(
            (batch, 1),
            float(has_history_bool),
            device=self.device,
            dtype=torch.float32,
        )
        if initial_noise is None:
            initial_noise = torch.randn(batch, 10, 148, device=self.device, dtype=torch.float32)
        else:
            initial_noise = initial_noise.to(device=self.device, dtype=torch.float32)
        if initial_noise.shape != (batch, 10, 148):
            raise ValueError(f"initial_noise must have shape [B,10,148], got {tuple(initial_noise.shape)}")

        clean_generation = timer.measure(
            "flow",
            lambda: self.flow.denoise_steps(
                initial_noise.to(dtype=self.model_dtype),
                history_hybrid.to(dtype=self.model_dtype),
                path_condition.to(dtype=self.model_dtype),
                first_heading.to(dtype=self.model_dtype),
                has_history.to(dtype=self.model_dtype),
                steps=self.flow_steps,
                text_feature=(
                    None
                    if text_feature is None
                    else text_feature.to(device=self.device, dtype=self.model_dtype)
                ),
                heading_condition=(
                    None
                    if heading_condition is None
                    else heading_condition.to(device=self.device, dtype=self.model_dtype)
                ),
            ).float(),
        )
        local_hybrid = (
            torch.cat([history_hybrid, clean_generation], dim=1)
            if has_history_bool
            else clean_generation
        )
        (
            explicit_motion,
            decoder_latent,
            decoder_global_root,
            decoder_local_root,
            decoder_token_valid,
        ) = self._decode(
            local_hybrid=local_hybrid,
            global_translation=global_translation,
            has_history=has_history_bool,
            timer=timer,
        )

        if profile:
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            timer.timings_ms["total"] = (perf_counter() - total_start) * 1000.0
        return StudentWindowOutput(
            explicit_motion=explicit_motion,
            clean_generation=clean_generation,
            history_hybrid=history_hybrid,
            path_condition=path_condition,
            first_heading=first_heading,
            has_history=has_history,
            global_translation=global_translation,
            decoder_latent=decoder_latent,
            decoder_global_root=decoder_global_root,
            decoder_local_root=decoder_local_root,
            decoder_token_valid=decoder_token_valid,
            timings_ms=timer.timings_ms,
        )

    @torch.inference_mode()
    def step(
        self,
        *,
        motion_mask: torch.Tensor | None,
        observed_motion: torch.Tensor | None,
        text_feature: torch.Tensor | None = None,
        heading_condition: torch.Tensor | None = None,
        initial_noise: torch.Tensor | None = None,
        init_history_sequence: torch.Tensor | None = None,
        init_global_translation: torch.Tensor | None = None,
        init_first_heading_angle: torch.Tensor | None = None,
        profile: bool = False,
    ) -> StudentWindowOutput:
        """Generate one window from world-space ARDY motion constraints."""

        candidates = [
            tensor
            for tensor in (
                init_history_sequence,
                init_global_translation,
                init_first_heading_angle,
                motion_mask,
                observed_motion,
                initial_noise,
                text_feature,
                heading_condition,
            )
            if tensor is not None
        ]
        if not candidates:
            raise ValueError("cannot infer batch size: provide an initial state, noise, or constraints")
        batch = candidates[0].shape[0]
        if any(tensor.shape[0] != batch for tensor in candidates):
            raise ValueError("all runtime inputs must use the same batch size")

        if init_history_sequence is not None:
            # Prepare the exact history coordinate frame once, then reuse it in
            # step_prepared.  The learned encoder is intentionally invoked only
            # there so profiling does not double-count it.
            history = init_history_sequence.to(device=self.device, dtype=torch.float32)
            root = self.motion_rep.extract_root(history)
            center_index = torch.full((batch,), 3, device=self.device, dtype=torch.long)
            _, global_translation = self.motion_rep.recenter_root_motion(
                root,
                center_index,
                is_normalized=True,
                to_normalize=True,
                return_center_pos=True,
            )
            history_frames = 4
        else:
            global_translation = (
                torch.zeros(batch, 3, device=self.device, dtype=torch.float32)
                if init_global_translation is None
                else init_global_translation.to(device=self.device, dtype=torch.float32)
            )
            history_frames = 0
        path_condition = self.prepare_path_condition(
            batch=batch,
            history_frames=history_frames,
            global_translation=global_translation,
            motion_mask=motion_mask,
            observed_motion=observed_motion,
        )
        return self.step_prepared(
            path_condition=path_condition,
            text_feature=text_feature,
            heading_condition=heading_condition,
            initial_noise=initial_noise,
            init_history_sequence=init_history_sequence,
            init_global_translation=init_global_translation,
            init_first_heading_angle=init_first_heading_angle,
            profile=profile,
        )
