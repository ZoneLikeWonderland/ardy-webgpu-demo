"""DMD2-style training utilities adapted to ARDY's conditional motion diffusion.

The released ARDY denoiser predicts clean hybrid motion (``x0``), whereas the
small fake-score network used here predicts diffusion noise (``epsilon``).
This module keeps the conversion explicit and reconstructs the exact 16-token
teacher window from the bounded teacher-shard fields.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F

from ardy.skeleton.kinematics import fk
from ardy.skeleton.transforms import global_rots_to_local_rots

from .losses import (
    BODY_CONTACT,
    BODY_ROT6D,
    masked_mean,
    safe_cont6d_to_matrix,
)


@dataclass(frozen=True)
class TeacherDenoiserInputs:
    x: torch.Tensor
    history_len: torch.Tensor
    generation_len: torch.Tensor
    future_len: torch.Tensor
    history_mask: torch.Tensor
    generation_mask: torch.Tensor
    future_mask: torch.Tensor
    history_token_mask: torch.Tensor
    generation_token_mask: torch.Tensor
    future_token_mask: torch.Tensor
    text_feat: torch.Tensor
    text_mask: torch.Tensor
    first_heading_angle: torch.Tensor
    motion_mask: torch.Tensor
    observed_motion: torch.Tensor


class TeacherScoreAdapter:
    """Query the frozen released denoiser on an arbitrary noised generation.

    Teacher shards store exactly the information used by the path-only browser
    target: zero text, zero or one history token, ten generation tokens and a
    total 64-frame condition window.  ``pack_inputs`` reconstructs the original
    masks and token placement without calling the sampler.
    """

    def __init__(self, model: Any, cfg_constraint: float = 1.5) -> None:
        self.model = model
        self.cfg_constraint = float(cfg_constraint)
        self.frames = 64
        self.frames_per_token = int(model.num_frames_per_token)
        self.generation_frames = int(model.gen_horizon_len)
        self.generation_tokens = self.generation_frames // self.frames_per_token
        self.sequence_tokens = self.frames // self.frames_per_token
        self.hybrid_dim = int(model.denoiser.nframe_root_dim + model.denoiser.latent_embedding_dim)
        self.motion_dim = int(model.motion_rep.motion_rep_dim)
        self.text_dim = int(model.denoiser.llm_shape[-1])
        if self.frames_per_token != 4 or self.generation_tokens != 10 or self.sequence_tokens != 16:
            raise ValueError(
                "DMD2 adapter currently targets the released 4-frame/40-frame/64-frame Core model"
            )

    @property
    def alphas_cumprod(self) -> torch.Tensor:
        return self.model.diffusion.alphas_cumprod_base

    def pack_inputs(
        self,
        noised_generation: torch.Tensor,
        history_hybrid: torch.Tensor,
        path_condition: torch.Tensor,
        first_heading: torch.Tensor,
        has_history: torch.Tensor,
    ) -> TeacherDenoiserInputs:
        batch = noised_generation.shape[0]
        device = noised_generation.device
        dtype = noised_generation.dtype
        expected = (batch, self.generation_tokens, self.hybrid_dim)
        if tuple(noised_generation.shape) != expected:
            raise ValueError(f"noised_generation must have shape {expected}, got {tuple(noised_generation.shape)}")
        if tuple(history_hybrid.shape) != (batch, 1, self.hybrid_dim):
            raise ValueError(f"history_hybrid must have shape {(batch, 1, self.hybrid_dim)}")
        if tuple(path_condition.shape) != (batch, self.frames, 3):
            raise ValueError(f"path_condition must have shape {(batch, self.frames, 3)}")

        has_history_bool = has_history.reshape(batch) > 0.5
        history_tokens = has_history_bool.to(dtype=torch.long)
        history_len = history_tokens * self.frames_per_token
        generation_len = torch.full(
            (batch,), self.generation_frames, device=device, dtype=torch.long
        )
        future_len = self.frames - history_len - generation_len

        frame_index = torch.arange(self.frames, device=device).unsqueeze(0)
        history_mask = frame_index < history_len.unsqueeze(1)
        generation_mask = (frame_index >= history_len.unsqueeze(1)) & (
            frame_index < (history_len + generation_len).unsqueeze(1)
        )
        future_mask = frame_index >= (history_len + generation_len).unsqueeze(1)

        token_index = torch.arange(self.sequence_tokens, device=device).unsqueeze(0)
        history_token_mask = token_index < history_tokens.unsqueeze(1)
        generation_token_mask = (token_index >= history_tokens.unsqueeze(1)) & (
            token_index < (history_tokens + self.generation_tokens).unsqueeze(1)
        )
        future_token_base = token_index >= (
            history_tokens + self.generation_tokens
        ).unsqueeze(1)
        path_valid_by_token = (path_condition[..., 2] > 0.5).reshape(
            batch, self.sequence_tokens, self.frames_per_token
        ).any(dim=-1)
        future_token_mask = future_token_base & path_valid_by_token

        x = torch.zeros(
            batch,
            self.sequence_tokens,
            self.hybrid_dim,
            device=device,
            dtype=dtype,
        )
        generation_indices = history_tokens.unsqueeze(1) + torch.arange(
            self.generation_tokens, device=device
        ).unsqueeze(0)
        x.scatter_(
            1,
            generation_indices.unsqueeze(-1).expand(-1, -1, self.hybrid_dim),
            noised_generation,
        )
        x[:, 0] = torch.where(
            has_history_bool.unsqueeze(-1),
            history_hybrid[:, 0].to(dtype=dtype),
            x[:, 0],
        )

        valid = (path_condition[..., 2] > 0.5).to(dtype=dtype)
        motion_mask = torch.zeros(
            batch, self.frames, self.motion_dim, device=device, dtype=dtype
        )
        observed_motion = torch.zeros_like(motion_mask)
        motion_mask[..., 0] = valid
        motion_mask[..., 2] = valid
        observed_motion[..., 0] = path_condition[..., 0].to(dtype=dtype) * valid
        observed_motion[..., 2] = path_condition[..., 1].to(dtype=dtype) * valid

        first_heading_angle = torch.atan2(
            first_heading[:, 1].float(), first_heading[:, 0].float()
        ).to(device=device)
        text_feat = torch.zeros(batch, 1, self.text_dim, device=device, dtype=dtype)
        text_mask = torch.zeros(batch, 1, device=device, dtype=torch.bool)
        return TeacherDenoiserInputs(
            x=x,
            history_len=history_len,
            generation_len=generation_len,
            future_len=future_len,
            history_mask=history_mask,
            generation_mask=generation_mask,
            future_mask=future_mask,
            history_token_mask=history_token_mask,
            generation_token_mask=generation_token_mask,
            future_token_mask=future_token_mask,
            text_feat=text_feat,
            text_mask=text_mask,
            first_heading_angle=first_heading_angle,
            motion_mask=motion_mask,
            observed_motion=observed_motion,
        )

    @torch.inference_mode()
    def predict_x0(
        self,
        noised_generation: torch.Tensor,
        history_hybrid: torch.Tensor,
        path_condition: torch.Tensor,
        first_heading: torch.Tensor,
        has_history: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        packed = self.pack_inputs(
            noised_generation,
            history_hybrid,
            path_condition,
            first_heading,
            has_history,
        )
        batch = noised_generation.shape[0]
        timesteps = timesteps.reshape(batch).to(
            device=noised_generation.device, dtype=torch.long
        )
        output = self.model.denoiser(
            torch.zeros(1, device=noised_generation.device, dtype=torch.float32),
            torch.full(
                (1,),
                self.cfg_constraint,
                device=noised_generation.device,
                dtype=torch.float32,
            ),
            packed.x,
            packed.history_len,
            packed.generation_len,
            packed.future_len,
            packed.history_mask,
            packed.generation_mask,
            packed.future_mask,
            packed.history_token_mask,
            packed.generation_token_mask,
            packed.future_token_mask,
            packed.text_feat,
            packed.text_mask,
            timesteps,
            packed.first_heading_angle,
            packed.motion_mask,
            packed.observed_motion,
        )
        selected = output[packed.generation_token_mask]
        return selected.reshape(batch, self.generation_tokens, self.hybrid_dim).float()


def _diffusion_coefficients(
    alphas_cumprod: torch.Tensor,
    timesteps: torch.Tensor,
    reference: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    alpha_bar = alphas_cumprod.to(device=reference.device, dtype=torch.float32)[
        timesteps.to(device=reference.device, dtype=torch.long)
    ]
    view_shape = (reference.shape[0],) + (1,) * (reference.ndim - 1)
    sqrt_alpha = alpha_bar.sqrt().reshape(view_shape)
    sqrt_sigma = (1.0 - alpha_bar).clamp_min(0).sqrt().reshape(view_shape)
    return sqrt_alpha, sqrt_sigma


def q_sample(
    x0: torch.Tensor,
    timesteps: torch.Tensor,
    noise: torch.Tensor,
    alphas_cumprod: torch.Tensor,
) -> torch.Tensor:
    sqrt_alpha, sqrt_sigma = _diffusion_coefficients(
        alphas_cumprod, timesteps, x0
    )
    return sqrt_alpha * x0.float() + sqrt_sigma * noise.float()


def q_sample_with_exact_clean(
    x0: torch.Tensor,
    timesteps: torch.Tensor,
    noise: torch.Tensor,
    alphas_cumprod: torch.Tensor,
    exact_clean: torch.Tensor,
) -> torch.Tensor:
    """Diffuse a batch while allowing an explicit sigma-zero branch.

    ARDY's released ten-step cosine schedule still has substantial noise at
    timestep zero (sigma is about 0.167).  Calling that row "clean" is
    incorrect, especially before computing motion derivatives.  Samples
    selected by ``exact_clean`` bypass ``q_sample``; all other rows retain the
    ordinary diffusion calculation exactly.
    """

    if exact_clean.numel() != x0.shape[0]:
        raise ValueError(
            "exact_clean must contain one value per batch item, got "
            f"{tuple(exact_clean.shape)} for batch {x0.shape[0]}"
        )
    noised = q_sample(x0, timesteps, noise, alphas_cumprod)
    view_shape = (x0.shape[0],) + (1,) * (x0.ndim - 1)
    mask = exact_clean.to(device=x0.device, dtype=torch.bool).reshape(view_shape)
    return torch.where(mask, x0.float(), noised)


def epsilon_to_x0(
    xt: torch.Tensor,
    epsilon: torch.Tensor,
    timesteps: torch.Tensor,
    alphas_cumprod: torch.Tensor,
) -> torch.Tensor:
    sqrt_alpha, sqrt_sigma = _diffusion_coefficients(
        alphas_cumprod, timesteps, xt
    )
    return (xt.float() - sqrt_sigma * epsilon.float()) / sqrt_alpha.clamp_min(1.0e-5)


def normalized_diffusion_time(
    timesteps: torch.Tensor,
    num_timesteps: int,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    return timesteps.to(dtype=dtype).reshape(-1, 1) / max(1, num_timesteps - 1)


def sample_dmd_timesteps(
    batch_size: int,
    device: torch.device,
    *,
    minimum: int,
    maximum: int,
    exact_max_probability: float,
    high_noise_probability: float,
) -> torch.Tensor:
    """Sample score-query timesteps with an explicit high-noise bias.

    The deployment generator always starts at maximum noise.  Distribution
    matching still needs some lower-noise score information, so the remainder
    is sampled uniformly instead of collapsing every query to one endpoint.
    The high-noise branch uses ``maximum - floor(U^3 * range)``.
    """

    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if minimum < 0 or maximum < minimum:
        raise ValueError("timesteps must satisfy 0 <= minimum <= maximum")
    if not 0.0 <= exact_max_probability <= 1.0:
        raise ValueError("exact_max_probability must be in [0, 1]")
    if not 0.0 <= high_noise_probability <= 1.0:
        raise ValueError("high_noise_probability must be in [0, 1]")
    if exact_max_probability + high_noise_probability > 1.0:
        raise ValueError("timestep mixture probabilities exceed one")

    uniform = torch.randint(
        minimum,
        maximum + 1,
        (batch_size,),
        device=device,
        dtype=torch.long,
    )
    span = maximum - minimum + 1
    high = maximum - torch.floor(
        torch.rand(batch_size, device=device).pow(3) * span
    ).to(torch.long)
    high = high.clamp(min=minimum, max=maximum)
    mode = torch.rand(batch_size, device=device)
    result = torch.where(
        mode < exact_max_probability + high_noise_probability,
        high,
        uniform,
    )
    result = torch.where(
        mode < exact_max_probability,
        torch.full_like(result, maximum),
        result,
    )
    return result


def fake_score_loss(
    predicted_epsilon: torch.Tensor,
    target_epsilon: torch.Tensor,
) -> dict[str, torch.Tensor]:
    root_error = predicted_epsilon[..., :20].float() - target_epsilon[..., :20].float()
    body_error = predicted_epsilon[..., 20:].float() - target_epsilon[..., 20:].float()
    root_mse = root_error.square().mean()
    body_mse = body_error.square().mean()
    total = 2.0 * root_mse + body_mse
    return {
        "fake_score_root_mse": root_mse,
        "fake_score_body_mse": body_mse,
        "fake_score_total": total,
    }


def distribution_matching_loss(
    generated_x0: torch.Tensor,
    teacher_x0: torch.Tensor,
    fake_x0: torch.Tensor,
    xt: torch.Tensor,
    grad_clip: float = 0.0,
) -> dict[str, torch.Tensor]:
    """DMD surrogate whose gradient is the normalized real/fake score gap."""

    generated = generated_x0.float()
    teacher = teacher_x0.float()
    fake = fake_x0.float()
    xt = xt.float()
    # Match the official DMD/DMD2 surrogate: the normalization is the
    # generator-to-real-score denoising residual, not the noised sample's
    # distance to the real prediction.  Using ``xt - teacher`` makes the
    # denominator mostly a function of injected noise and suppresses the
    # generator correction at high timesteps.
    p_real = generated - teacher
    p_fake = generated - fake
    normalizer = p_real.abs().mean(dim=(1, 2), keepdim=True).clamp_min(1.0e-6)
    gradient = torch.nan_to_num((p_real - p_fake) / normalizer)
    if grad_clip > 0:
        gradient = gradient.clamp(min=-grad_clip, max=grad_clip)
    target = (generated - gradient).detach()
    root = 0.5 * (generated[..., :20] - target[..., :20]).square().mean()
    body = 0.5 * (generated[..., 20:] - target[..., 20:]).square().mean()
    return {
        "dmd_root": root,
        "dmd_body": body,
        "dmd_total": 2.0 * root + body,
        "dmd_gradient_abs": gradient.abs().mean().detach(),
        "dmd_normalizer": normalizer.mean().detach(),
        "dmd_teacher_fake_x0_l1": (teacher - fake).abs().mean().detach(),
    }


def path_constraint_losses(
    generated_x0: torch.Tensor,
    path_condition: torch.Tensor,
    has_history: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Compare generated normalized root x/z at constrained generation frames."""

    batch = generated_x0.shape[0]
    root = generated_x0[..., :20].reshape(batch, 40, 5)
    offsets = (has_history.reshape(batch) > 0.5).to(torch.long) * 4
    indices = offsets.unsqueeze(1) + torch.arange(40, device=root.device).unsqueeze(0)
    selected = path_condition.gather(
        1, indices.unsqueeze(-1).expand(-1, -1, path_condition.shape[-1])
    )
    valid = selected[..., 2]
    error = root[..., [0, 2]].float() - selected[..., :2].float()
    mse = masked_mean(error.square(), valid)
    distance = masked_mean(torch.linalg.vector_norm(error, dim=-1), valid)
    return {
        "path_constraint_mse": mse,
        "path_constraint_distance_normalized": distance,
    }


def seam_losses(
    generated_x0: torch.Tensor,
    decoded_body: torch.Tensor,
    history_hybrid: torch.Tensor,
    target_root: torch.Tensor,
    target_body: torch.Tensor,
    has_history: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Match teacher velocity across the history/generation boundary."""

    batch = generated_x0.shape[0]
    history_root = history_hybrid[..., :20].reshape(batch, 4, 5)
    generated_root = generated_x0[..., :20].reshape(batch, 40, 5)
    predicted_root = torch.cat([history_root, generated_root], dim=1)
    valid = has_history.reshape(batch)
    root_delta_error = (
        predicted_root[:, 4] - predicted_root[:, 3]
        - (target_root[:, 4] - target_root[:, 3])
    )
    body_delta_error = (
        decoded_body[:, 4] - decoded_body[:, 3]
        - (target_body[:, 4] - target_body[:, 3])
    )
    root_l1 = masked_mean(root_delta_error.abs(), valid)
    body_l1 = masked_mean(body_delta_error.abs(), valid)
    return {
        "seam_root_velocity_l1": root_l1,
        "seam_body_velocity_l1": body_l1,
        "seam_total": 2.0 * root_l1 + body_l1,
    }


def motion_critic_features(
    normalized_body: torch.Tensor,
    normalized_root: torch.Tensor,
    motion_rep: Any,
) -> torch.Tensor:
    """Build 658-D frame features used by the training-only critic.

    Layout: normalized root/body (330), physical FK joints (81), physical
    joint velocity/acceleration/jerk (81 each), and foot-contact channels (4).
    Explicit higher derivatives let the critic see the frame-scale jitter that
    invalidated the previous browser candidate instead of inferring it only
    through several convolution layers.
    """

    body = motion_rep.body_stats.unnormalize(normalized_body.float())
    root = motion_rep.global_root_stats.unnormalize(normalized_root.float())
    batch, frames = body.shape[:2]
    global_rotations = safe_cont6d_to_matrix(
        body[..., BODY_ROT6D].reshape(batch, frames, 27, 6)
    )
    local_rotations = global_rots_to_local_rots(
        global_rotations, motion_rep.skeleton
    )
    _, joints, _ = fk(local_rotations, root[..., :3], motion_rep.skeleton)
    joint_velocity = torch.diff(joints, dim=1, prepend=joints[:, :1])
    joint_acceleration = torch.diff(
        joint_velocity, dim=1, prepend=joint_velocity[:, :1]
    )
    joint_jerk = torch.diff(
        joint_acceleration, dim=1, prepend=joint_acceleration[:, :1]
    )
    contact = body[..., BODY_CONTACT]
    return torch.cat(
        [
            normalized_root.float(),
            normalized_body.float(),
            joints.reshape(batch, frames, -1),
            joint_velocity.reshape(batch, frames, -1),
            joint_acceleration.reshape(batch, frames, -1),
            joint_jerk.reshape(batch, frames, -1),
            contact,
        ],
        dim=-1,
    )


def diffusion_motion_critic_features(
    normalized_body: torch.Tensor,
    normalized_root: torch.Tensor,
    timesteps: torch.Tensor,
    alphas_cumprod: torch.Tensor,
    motion_rep: Any,
    noise: torch.Tensor | None = None,
    exact_clean: torch.Tensor | None = None,
) -> torch.Tensor:
    """Build critic features after diffusion-noising normalized motion.

    DMD2's adversarial branch is a diffusion GAN: its classifier sees samples
    corrupted at a random diffusion timestep instead of trivially separating
    clean real/fake samples.  ARDY body and global-root channels are already
    normalized, so they can share the released model's cosine schedule before
    the physical FK/derivative features are constructed.  Supplying ``noise``
    is useful for deterministic tests; real and fake training calls otherwise
    draw independent noise, matching the distributional objective.
    """

    if normalized_body.shape[:-1] != normalized_root.shape[:-1]:
        raise ValueError("body and root motion must share batch/frame dimensions")
    packed = torch.cat(
        [normalized_root.float(), normalized_body.float()], dim=-1
    )
    if noise is None:
        noise = torch.randn_like(packed)
    elif noise.shape != packed.shape:
        raise ValueError(
            f"noise must have shape {tuple(packed.shape)}, got {tuple(noise.shape)}"
        )
    if exact_clean is None:
        noised = q_sample(packed, timesteps, noise, alphas_cumprod)
    else:
        noised = q_sample_with_exact_clean(
            packed,
            timesteps,
            noise,
            alphas_cumprod,
            exact_clean,
        )
    root_dim = normalized_root.shape[-1]
    return motion_critic_features(
        noised[..., root_dim:], noised[..., :root_dim], motion_rep
    )


def discriminator_losses(
    real_logits: torch.Tensor,
    fake_logits: torch.Tensor,
) -> dict[str, torch.Tensor]:
    real = F.softplus(-real_logits).mean()
    fake = F.softplus(fake_logits).mean()
    return {
        "critic_real": real,
        "critic_fake": fake,
        "critic_total": real + fake,
        "critic_real_logit": real_logits.mean().detach(),
        "critic_fake_logit": fake_logits.mean().detach(),
    }


def generator_adversarial_loss(fake_logits: torch.Tensor) -> torch.Tensor:
    return F.softplus(-fake_logits).mean()
