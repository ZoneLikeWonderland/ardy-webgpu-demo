"""Exact, non-invasive tracing around the released ARDY autoregressive call."""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any

import torch


@dataclass
class TeacherWindowTrace:
    """One original 10-step teacher window, packed for all three students."""

    initial_noise: torch.Tensor  # [B,10,148]
    denoising_states: torch.Tensor  # [B,11,10,148], noise plus 10 DDIM states
    clean_generation: torch.Tensor  # [B,10,148], local root + requantized body latent
    history_hybrid: torch.Tensor  # [B,1,148], zero for initial mode
    path_condition: torch.Tensor  # [B,64,3], normalized local x/z + valid
    first_heading: torch.Tensor  # [B,2], cos/sin
    has_history: torch.Tensor  # [B,1]
    encoder_body: torch.Tensor  # [B,4,325], zero for initial mode
    encoder_valid: torch.Tensor  # [B,1]
    decoder_latent: torch.Tensor  # [B,11,128], slot zero is dummy for initial mode
    decoder_global_root: torch.Tensor  # [B,44,5], slot zero is dummy for initial mode
    decoder_local_root: torch.Tensor  # [B,44,4]
    decoder_token_valid: torch.Tensor  # [B,11]
    target_body: torch.Tensor  # [B,44,325], dummy frames zero in initial mode
    explicit_motion: torch.Tensor  # [B,40 or 44,330], exact original return
    global_translation: torch.Tensor  # [B,3]

    def cpu(self) -> "TeacherWindowTrace":
        return TeacherWindowTrace(
            **{
                field.name: getattr(self, field.name).detach().cpu()
                for field in fields(self)
            }
        )


def _select_masked_tokens(x: torch.Tensor, mask: torch.Tensor, expected_tokens: int) -> torch.Tensor:
    batch, _, width = x.shape
    selected = x[mask].reshape(batch, expected_tokens, width)
    return selected


def _pack_path_condition(
    observed_motion: torch.Tensor | None,
    motion_mask: torch.Tensor | None,
    batch: int,
    frames: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    result = torch.zeros(batch, frames, 3, device=device, dtype=dtype)
    if observed_motion is None or motion_mask is None:
        return result
    result[..., :2] = observed_motion[..., [0, 2]].to(dtype=dtype)
    valid = (motion_mask[..., 0] > 0) | (motion_mask[..., 2] > 0)
    result[..., 2] = valid.to(dtype=dtype)
    result[..., :2] *= result[..., 2:3]
    return result


def trace_autoregressive_step(model: Any, **call_kwargs: Any) -> TeacherWindowTrace:
    """Call ``model.autoregressive_step`` unchanged while capturing its exact internals.

    Instance methods are wrapped only for the duration of this synchronous call
    and restored in ``finally``. The release implementation, inputs, CFG, masks,
    DDIM sampler, recentering and decoder therefore remain the source of truth.
    """

    original_generate_window = model._generate_window
    original_denoising_step = model.denoising_step
    capture: dict[str, Any] = {"states": []}

    def traced_denoising_step(*args: Any, **kwargs: Any) -> torch.Tensor:
        x = args[0] if args else kwargs["x"]
        generation_mask = args[8] if len(args) > 8 else kwargs["generation_token_mask"]
        if "initial_noise" not in capture:
            generation_tokens = int(model.gen_horizon_len // model.num_frames_per_token)
            capture["initial_noise"] = _select_masked_tokens(x, generation_mask, generation_tokens).detach().clone()
            capture["history_token_mask"] = (
                args[7] if len(args) > 7 else kwargs["history_token_mask"]
            ).detach().clone()
            capture["first_x"] = x.detach().clone()
            capture["motion_mask"] = args[14] if len(args) > 14 else kwargs.get("motion_mask")
            capture["observed_motion"] = args[15] if len(args) > 15 else kwargs.get("observed_motion")
            capture["first_heading_angle"] = (
                args[13] if len(args) > 13 else kwargs.get("first_heading_angle")
            )
        output = original_denoising_step(*args, **kwargs)
        state = _select_masked_tokens(output, generation_mask, model.gen_horizon_len // model.num_frames_per_token)
        capture["states"].append(state.detach().clone())
        return output

    def traced_generate_window(*args: Any, **kwargs: Any) -> torch.Tensor:
        global_translation = args[1] if len(args) > 1 else kwargs["global_transl"]
        capture["global_translation"] = global_translation.detach().clone()
        output = original_generate_window(*args, **kwargs)
        capture["local_hybrid"] = output.detach().clone()
        return output

    model.denoising_step = traced_denoising_step
    model._generate_window = traced_generate_window
    try:
        with torch.inference_mode():
            explicit_motion = model.autoregressive_step(**call_kwargs)
    finally:
        model.denoising_step = original_denoising_step
        model._generate_window = original_generate_window

    required = {
        "initial_noise",
        "history_token_mask",
        "first_x",
        "global_translation",
        "local_hybrid",
    }
    missing = required - capture.keys()
    if missing:
        raise RuntimeError(f"teacher trace is incomplete: {sorted(missing)}")

    local_hybrid = capture["local_hybrid"]
    root, latent = model.hybrid.get_root_and_latent_body_motion_from_hybrid(local_hybrid)
    latent = model.autoencoder.requantize(latent)
    local_hybrid_quantized = model.hybrid.get_hybrid_motion_from_root_and_latent_body_motion(root, latent)
    generation_tokens = model.gen_horizon_len // model.num_frames_per_token
    clean_generation = local_hybrid_quantized[:, -generation_tokens:]

    first_x = capture["first_x"]
    history_token_mask = capture["history_token_mask"]
    batch, _, hybrid_dim = first_x.shape
    has_history_bool = history_token_mask.any(dim=1)
    history_hybrid = torch.zeros(batch, 1, hybrid_dim, device=first_x.device, dtype=first_x.dtype)
    if has_history_bool.any():
        for batch_index in range(batch):
            history_values = first_x[batch_index, history_token_mask[batch_index]]
            if len(history_values):
                history_hybrid[batch_index, 0] = history_values[-1]
    has_history = has_history_bool.to(dtype=first_x.dtype).unsqueeze(1)
    init_history_sequence = call_kwargs.get("init_history_sequence")
    encoder_body = torch.zeros(
        batch,
        model.num_frames_per_token,
        model.motion_rep.body_dim,
        device=first_x.device,
        dtype=first_x.dtype,
    )
    if init_history_sequence is not None:
        if init_history_sequence.shape[1] != model.num_frames_per_token:
            raise ValueError(
                "student encoder trace requires exactly "
                f"{model.num_frames_per_token} history frames, got {init_history_sequence.shape[1]}"
            )
        encoder_body.copy_(model.motion_rep.extract_body(init_history_sequence))
    encoder_valid = has_history.clone()

    first_heading_angle = capture.get("first_heading_angle")
    if first_heading_angle is None:
        first_heading_angle = torch.zeros(batch, device=first_x.device, dtype=first_x.dtype)
    first_heading = torch.stack(
        [torch.cos(first_heading_angle), torch.sin(first_heading_angle)],
        dim=-1,
    ).to(dtype=first_x.dtype)

    frame_count = int(first_x.shape[1] * model.num_frames_per_token)
    path_condition = _pack_path_condition(
        capture.get("observed_motion"),
        capture.get("motion_mask"),
        batch,
        frame_count,
        first_x.device,
        first_x.dtype,
    )

    valid_lengths = torch.full(
        (batch,),
        root.shape[1],
        device=root.device,
        dtype=torch.long,
    )
    local_root = model.motion_rep.global_root_to_local_root(
        root,
        normalized=True,
        lengths=valid_lengths,
    )
    target_body_native = model.motion_rep.extract_body(explicit_motion)

    decoder_tokens = generation_tokens + 1
    decoder_frames = decoder_tokens * model.num_frames_per_token
    decoder_latent = torch.zeros(
        batch,
        decoder_tokens,
        latent.shape[-1],
        device=latent.device,
        dtype=latent.dtype,
    )
    decoder_local_root = torch.zeros(
        batch,
        decoder_frames,
        local_root.shape[-1],
        device=local_root.device,
        dtype=local_root.dtype,
    )
    decoder_global_root = torch.zeros(
        batch,
        decoder_frames,
        root.shape[-1],
        device=root.device,
        dtype=root.dtype,
    )
    target_body = torch.zeros(
        batch,
        decoder_frames,
        target_body_native.shape[-1],
        device=target_body_native.device,
        dtype=target_body_native.dtype,
    )
    decoder_token_valid = torch.ones(
        batch,
        decoder_tokens,
        device=latent.device,
        dtype=latent.dtype,
    )

    if latent.shape[1] == generation_tokens:
        decoder_latent[:, 1:] = latent
        decoder_global_root[:, model.num_frames_per_token :] = root
        decoder_local_root[:, model.num_frames_per_token :] = local_root
        target_body[:, model.num_frames_per_token :] = target_body_native
        decoder_token_valid[:, 0] = 0
    elif latent.shape[1] == decoder_tokens:
        decoder_latent.copy_(latent)
        decoder_global_root.copy_(root)
        decoder_local_root.copy_(local_root)
        target_body.copy_(target_body_native)
    else:
        raise ValueError(
            f"student trace expects 0 or 1 history token, got {latent.shape[1] - generation_tokens}"
        )

    denoising_states = torch.stack([capture["initial_noise"], *capture["states"]], dim=1)
    return TeacherWindowTrace(
        initial_noise=capture["initial_noise"],
        denoising_states=denoising_states,
        clean_generation=clean_generation,
        history_hybrid=history_hybrid,
        path_condition=path_condition,
        first_heading=first_heading,
        has_history=has_history,
        encoder_body=encoder_body,
        encoder_valid=encoder_valid,
        decoder_latent=decoder_latent,
        decoder_global_root=decoder_global_root,
        decoder_local_root=decoder_local_root,
        decoder_token_valid=decoder_token_valid,
        target_body=target_body,
        explicit_motion=explicit_motion,
        global_translation=capture["global_translation"],
    )
