"""Distillation losses and motion-aware metrics for the Core27 representation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from ardy.skeleton.kinematics import fk
from ardy.skeleton.transforms import global_rots_to_local_rots


BODY_POS = slice(0, 78)
BODY_ROT6D = slice(78, 240)
BODY_VELOCITY = slice(240, 321)
BODY_CONTACT = slice(321, 325)


def deployment_decoder_roots(
    clean_generation: torch.Tensor,
    history_hybrid: torch.Tensor,
    has_history: torch.Tensor,
    motion_rep,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build the exact static decoder root inputs used by deployment.

    Continuation windows convert the retained four history roots together with
    forty generated roots.  Initial windows first convert only the forty real
    generated roots and then prepend four invalid dummy frames.  Mixing these
    cases after conversion is essential: converting an initial dummy prefix as
    if it were real history would corrupt the first generated velocity.
    """

    if clean_generation.shape[-2:] != (10, 148):
        raise ValueError(
            "clean_generation must have shape [B,10,148], got "
            f"{tuple(clean_generation.shape)}"
        )
    if history_hybrid.shape[-2:] != (1, 148):
        raise ValueError(
            "history_hybrid must have shape [B,1,148], got "
            f"{tuple(history_hybrid.shape)}"
        )
    batch = clean_generation.shape[0]
    generated_root = clean_generation[..., :20].reshape(batch, 40, 5).float()
    history_root = history_hybrid[..., :20].reshape(batch, 4, 5).float()
    decoder_global_root = torch.cat([history_root, generated_root], dim=1)

    continuation_lengths = torch.full(
        (batch,),
        44,
        dtype=torch.long,
        device=generated_root.device,
    )
    continuation_local_root = motion_rep.global_root_to_local_root(
        decoder_global_root,
        normalized=True,
        lengths=continuation_lengths,
    ).float()
    initial_lengths = torch.full(
        (batch,),
        40,
        dtype=torch.long,
        device=generated_root.device,
    )
    initial_actual_local_root = motion_rep.global_root_to_local_root(
        generated_root,
        normalized=True,
        lengths=initial_lengths,
    ).float()
    initial_local_root = torch.cat(
        [initial_actual_local_root.new_zeros(batch, 4, 4), initial_actual_local_root],
        dim=1,
    )
    history_mask = (has_history.reshape(batch, 1, 1) > 0.5)
    decoder_local_root = torch.where(
        history_mask,
        continuation_local_root,
        initial_local_root,
    )
    return decoder_global_root, decoder_local_root


def masked_mean(values: torch.Tensor, mask: torch.Tensor, eps: float = 1.0e-8) -> torch.Tensor:
    while mask.ndim < values.ndim:
        mask = mask.unsqueeze(-1)
    mask = mask.to(dtype=values.dtype)
    return (values * mask).sum() / mask.expand_as(values).sum().clamp_min(eps)


def frame_valid_from_tokens(token_valid: torch.Tensor, frames_per_token: int = 4) -> torch.Tensor:
    return token_valid.repeat_interleave(frames_per_token, dim=1)


def safe_cont6d_to_matrix(cont6d: torch.Tensor, eps: float = 1.0e-6) -> torch.Tensor:
    x_raw = cont6d[..., :3]
    y_raw = cont6d[..., 3:]
    x = F.normalize(x_raw, dim=-1, eps=eps)
    z = F.normalize(torch.cross(x, y_raw, dim=-1), dim=-1, eps=eps)
    y = torch.cross(z, x, dim=-1)
    return torch.stack([x, y, z], dim=-1)


class FSQRequantizer(nn.Module):
    """Exact released 128-scalar, 64-level FSQ normalization with STE."""

    def __init__(self, stats_dir: str | Path, levels: int = 64) -> None:
        super().__init__()
        stats_dir = Path(stats_dir)
        mean = torch.from_numpy(np.load(stats_dir / "mean.npy")).float()
        std = torch.from_numpy(np.load(stats_dir / "std.npy")).float()
        self.register_buffer("mean", mean)
        self.register_buffer("std_eps", torch.sqrt(std.square() + 1.0e-5))
        self.half_width = levels // 2

    def unnormalize(self, latent: torch.Tensor) -> torch.Tensor:
        return latent * self.std_eps.to(dtype=latent.dtype) + self.mean.to(dtype=latent.dtype)

    def normalize(self, latent: torch.Tensor) -> torch.Tensor:
        return (latent - self.mean.to(dtype=latent.dtype)) / self.std_eps.to(dtype=latent.dtype)

    def forward(self, normalized_latent: torch.Tensor, ste: bool = True) -> torch.Tensor:
        raw = self.unnormalize(normalized_latent).clamp(-1, 1)
        scaled = raw * self.half_width
        rounded = scaled.round()
        if ste:
            rounded = scaled + (rounded - scaled).detach()
        return self.normalize(rounded / self.half_width)

    @torch.no_grad()
    def bin_accuracy(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_bin = (self.unnormalize(prediction).clamp(-1, 1) * self.half_width).round()
        target_bin = (self.unnormalize(target).clamp(-1, 1) * self.half_width).round()
        return (pred_bin == target_bin).float().mean()


@torch.no_grad()
def fsq_endpoint_diagnostics(
    prediction: torch.Tensor,
    target: torch.Tensor,
    quantizer: FSQRequantizer,
) -> dict[str, torch.Tensor]:
    """Measure deployment-bin robustness of predicted normalized FSQ latents.

    Released teacher endpoints are exactly on the 64-level FSQ centers.  A
    continuous endpoint can still have a small L1 error while sitting close to
    a rounding boundary, where tiny FP16/WebGPU differences change the decoded
    token.  Reporting both bin accuracy and boundary proximity exposes that
    failure mode without changing the supervised objective.
    """

    pred_raw = quantizer.unnormalize(prediction.float()).clamp(-1, 1)
    target_raw = quantizer.unnormalize(target.float()).clamp(-1, 1)
    pred_scaled = pred_raw * quantizer.half_width
    target_bins = (target_raw * quantizer.half_width).round()
    pred_bins = pred_scaled.round()
    distance_to_center = (pred_scaled - target_bins).abs()
    fractional_distance = (pred_scaled - pred_bins).abs()
    return {
        "fsq_bin_accuracy": (pred_bins == target_bins).float().mean(),
        "fsq_center_abs_bins": distance_to_center.mean(),
        "fsq_center_p95_bins": torch.quantile(distance_to_center.reshape(-1), 0.95),
        "fsq_near_boundary_fraction": (fractional_distance >= 0.45).float().mean(),
    }


def fsq_endpoint_losses(
    prediction: torch.Tensor,
    target: torch.Tensor,
    quantizer: FSQRequantizer,
) -> dict[str, torch.Tensor]:
    """Deployment-aligned loss for a continuous flow endpoint.

    Teacher body latents are already on the released FSQ grid.  The ordinary
    endpoint loss pulls the continuous prediction toward each bin center; this
    STE term additionally assigns the deployed, re-quantized token an explicit
    penalty whenever it lands in the wrong bin.
    """

    requantized = quantizer(prediction, ste=True)
    return {
        "fsq_quantized_l1": (requantized - target).abs().mean(),
    }


def encoder_distillation_losses(
    prediction: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor,
    quantizer: FSQRequantizer,
) -> dict[str, torch.Tensor]:
    continuous_l1 = masked_mean((prediction - target).abs(), valid)
    continuous_mse = masked_mean((prediction - target).square(), valid)
    requantized = quantizer(prediction, ste=True)
    quantized_l1 = masked_mean((requantized - target).abs(), valid)
    with torch.no_grad():
        bin_accuracy = quantizer.bin_accuracy(prediction[valid[:, 0] > 0], target[valid[:, 0] > 0]) if valid.any() else prediction.new_zeros(())
    return {
        "encoder_l1": continuous_l1,
        "encoder_mse": continuous_mse,
        "encoder_quantized_l1": quantized_l1,
        "encoder_bin_accuracy": bin_accuracy,
        "encoder_total": continuous_mse + 0.5 * continuous_l1 + quantized_l1,
    }


def decoder_feature_losses(
    prediction: torch.Tensor,
    target: torch.Tensor,
    token_valid: torch.Tensor,
) -> dict[str, torch.Tensor]:
    valid = frame_valid_from_tokens(token_valid)
    losses = {
        "decoder_body_l1": masked_mean((prediction - target).abs(), valid),
        "decoder_position_l1": masked_mean((prediction[..., BODY_POS] - target[..., BODY_POS]).abs(), valid),
        "decoder_rotation_l1": masked_mean((prediction[..., BODY_ROT6D] - target[..., BODY_ROT6D]).abs(), valid),
        "decoder_velocity_l1": masked_mean((prediction[..., BODY_VELOCITY] - target[..., BODY_VELOCITY]).abs(), valid),
        "decoder_contact_l1": masked_mean((prediction[..., BODY_CONTACT] - target[..., BODY_CONTACT]).abs(), valid),
    }
    losses["decoder_total"] = (
        losses["decoder_body_l1"]
        + 0.5 * losses["decoder_position_l1"]
        + losses["decoder_rotation_l1"]
        + 0.25 * losses["decoder_velocity_l1"]
        + 0.25 * losses["decoder_contact_l1"]
    )
    return losses


def motion_quality_losses(
    prediction_body: torch.Tensor,
    target_body: torch.Tensor,
    normalized_global_root: torch.Tensor,
    token_valid: torch.Tensor,
    motion_rep,
    target_normalized_global_root: torch.Tensor | None = None,
    frames_per_token: int = 4,
) -> dict[str, torch.Tensor]:
    """Differentiable rotation/FK/temporal/contact losses in physical units."""

    valid = frame_valid_from_tokens(token_valid)
    prediction = motion_rep.body_stats.unnormalize(prediction_body.float())
    target = motion_rep.body_stats.unnormalize(target_body.float())
    prediction_root = motion_rep.global_root_stats.unnormalize(normalized_global_root.float())
    if target_normalized_global_root is None:
        target_normalized_global_root = normalized_global_root
    target_root = motion_rep.global_root_stats.unnormalize(target_normalized_global_root.float())

    batch, frames = prediction.shape[:2]
    pred_global = safe_cont6d_to_matrix(prediction[..., BODY_ROT6D].reshape(batch, frames, 27, 6))
    target_global = safe_cont6d_to_matrix(target[..., BODY_ROT6D].reshape(batch, frames, 27, 6))
    relative = pred_global.transpose(-1, -2) @ target_global
    cosine = ((relative.diagonal(dim1=-2, dim2=-1).sum(-1) - 1.0) * 0.5).clamp(-1 + 1e-6, 1 - 1e-6)
    rotation_geodesic = masked_mean(torch.acos(cosine), valid)

    pred_rotation_velocity = pred_global[:, 1:] - pred_global[:, :-1]
    target_rotation_velocity = target_global[:, 1:] - target_global[:, :-1]
    pair_valid = valid[:, 1:] * valid[:, :-1]

    def token_boundary_valid(base_valid: torch.Tensor, order: int) -> torch.Tensor:
        """Select derivatives whose stencil crosses a four-frame token edge."""

        if frames_per_token < 1:
            raise ValueError("frames_per_token must be positive")
        endpoint = torch.arange(order, frames, device=base_valid.device)
        crosses_boundary = (endpoint % frames_per_token) < order
        return base_valid * crosses_boundary.to(dtype=base_valid.dtype).unsqueeze(0)

    boundary_pair_valid = token_boundary_valid(pair_valid, 1)
    rotation_velocity_l1 = masked_mean(
        (pred_rotation_velocity - target_rotation_velocity).abs(),
        pair_valid,
    )
    rotation_boundary_velocity_l1 = masked_mean(
        (pred_rotation_velocity - target_rotation_velocity).abs(),
        boundary_pair_valid,
    )
    if frames > 2:
        pred_rotation_acceleration = (
            pred_rotation_velocity[:, 1:] - pred_rotation_velocity[:, :-1]
        )
        target_rotation_acceleration = (
            target_rotation_velocity[:, 1:] - target_rotation_velocity[:, :-1]
        )
        triple_valid = pair_valid[:, 1:] * pair_valid[:, :-1]
        boundary_triple_valid = token_boundary_valid(triple_valid, 2)
        rotation_acceleration_l1 = masked_mean(
            (pred_rotation_acceleration - target_rotation_acceleration).abs(),
            triple_valid,
        )
        rotation_boundary_acceleration_l1 = masked_mean(
            (pred_rotation_acceleration - target_rotation_acceleration).abs(),
            boundary_triple_valid,
        )
    else:
        rotation_acceleration_l1 = prediction.new_zeros(())
        rotation_boundary_acceleration_l1 = prediction.new_zeros(())
        triple_valid = pair_valid[:, :0]
    if frames > 3:
        pred_rotation_jerk = (
            pred_rotation_acceleration[:, 1:] - pred_rotation_acceleration[:, :-1]
        )
        target_rotation_jerk = (
            target_rotation_acceleration[:, 1:] - target_rotation_acceleration[:, :-1]
        )
        quadruple_valid = triple_valid[:, 1:] * triple_valid[:, :-1]
        boundary_quadruple_valid = token_boundary_valid(quadruple_valid, 3)
        rotation_jerk_l1 = masked_mean(
            (pred_rotation_jerk - target_rotation_jerk).abs(),
            quadruple_valid,
        )
        rotation_boundary_jerk_l1 = masked_mean(
            (pred_rotation_jerk - target_rotation_jerk).abs(),
            boundary_quadruple_valid,
        )
    else:
        rotation_jerk_l1 = prediction.new_zeros(())
        rotation_boundary_jerk_l1 = prediction.new_zeros(())

    pred_local = global_rots_to_local_rots(pred_global, motion_rep.skeleton)
    target_local = global_rots_to_local_rots(target_global, motion_rep.skeleton)
    _, pred_joints, _ = fk(pred_local, prediction_root[..., :3], motion_rep.skeleton)
    _, target_joints, _ = fk(target_local, target_root[..., :3], motion_rep.skeleton)
    joint_error = torch.linalg.vector_norm(pred_joints - target_joints, dim=-1)
    fk_mpjpe = masked_mean(joint_error, valid)

    pred_velocity = pred_joints[:, 1:] - pred_joints[:, :-1]
    target_velocity = target_joints[:, 1:] - target_joints[:, :-1]
    joint_velocity_l1 = masked_mean((pred_velocity - target_velocity).abs(), pair_valid)
    joint_boundary_velocity_l1 = masked_mean(
        (pred_velocity - target_velocity).abs(), boundary_pair_valid
    )
    if frames > 2:
        pred_acceleration = pred_velocity[:, 1:] - pred_velocity[:, :-1]
        target_acceleration = target_velocity[:, 1:] - target_velocity[:, :-1]
        joint_acceleration_l1 = masked_mean((pred_acceleration - target_acceleration).abs(), triple_valid)
        joint_boundary_acceleration_l1 = masked_mean(
            (pred_acceleration - target_acceleration).abs(), boundary_triple_valid
        )
    else:
        joint_acceleration_l1 = prediction.new_zeros(())
        joint_boundary_acceleration_l1 = prediction.new_zeros(())
        triple_valid = pair_valid[:, :0]
    if frames > 3:
        pred_jerk = pred_acceleration[:, 1:] - pred_acceleration[:, :-1]
        target_jerk = target_acceleration[:, 1:] - target_acceleration[:, :-1]
        joint_jerk_l1 = masked_mean((pred_jerk - target_jerk).abs(), quadruple_valid)
        joint_boundary_jerk_l1 = masked_mean(
            (pred_jerk - target_jerk).abs(), boundary_quadruple_valid
        )
    else:
        joint_jerk_l1 = prediction.new_zeros(())
        joint_boundary_jerk_l1 = prediction.new_zeros(())

    pred_contact = prediction[..., BODY_CONTACT]
    target_contact = target[..., BODY_CONTACT]
    contact_l1 = masked_mean((pred_contact - target_contact).abs(), valid)
    foot_indices = torch.tensor(motion_rep.skeleton.foot_joint_idx, device=pred_joints.device)
    foot_speed = torch.linalg.vector_norm(pred_velocity[:, :, foot_indices], dim=-1)
    contact_pair = (target_contact[:, 1:] > 0.5).float() * pair_valid.unsqueeze(-1)
    foot_slide = masked_mean(foot_speed, contact_pair)

    return {
        "rotation_geodesic": rotation_geodesic,
        "rotation_velocity_l1": rotation_velocity_l1,
        "rotation_acceleration_l1": rotation_acceleration_l1,
        "rotation_jerk_l1": rotation_jerk_l1,
        "rotation_boundary_velocity_l1": rotation_boundary_velocity_l1,
        "rotation_boundary_acceleration_l1": rotation_boundary_acceleration_l1,
        "rotation_boundary_jerk_l1": rotation_boundary_jerk_l1,
        "fk_mpjpe": fk_mpjpe,
        "joint_velocity_l1": joint_velocity_l1,
        "joint_acceleration_l1": joint_acceleration_l1,
        "joint_jerk_l1": joint_jerk_l1,
        "joint_boundary_velocity_l1": joint_boundary_velocity_l1,
        "joint_boundary_acceleration_l1": joint_boundary_acceleration_l1,
        "joint_boundary_jerk_l1": joint_boundary_jerk_l1,
        "contact_physical_l1": contact_l1,
        "foot_slide": foot_slide,
    }


def physical_seam_losses(
    prediction_body: torch.Tensor,
    target_body: torch.Tensor,
    normalized_global_root: torch.Tensor,
    target_normalized_global_root: torch.Tensor,
    has_history: torch.Tensor,
    motion_rep,
) -> dict[str, torch.Tensor]:
    """Measure the deployed history/generation seam in physical units.

    The infinite runtime keeps the explicit history frames and appends the
    decoder's first generated frame.  Consequently the physical velocity-jump
    error at that boundary is determined by the first generated root/joint
    positions, not by a reconstructed history frame from the compact decoder.
    This loss uses the retained target history for frames 2/3 and the predicted
    frame 4, matching the rollout evaluator's buffer semantics.
    """

    prediction = motion_rep.body_stats.unnormalize(prediction_body.float())
    target = motion_rep.body_stats.unnormalize(target_body.float())
    prediction_root = motion_rep.global_root_stats.unnormalize(
        normalized_global_root.float()
    )
    target_root = motion_rep.global_root_stats.unnormalize(
        target_normalized_global_root.float()
    )
    batch = prediction.shape[0]
    valid = has_history.reshape(batch)
    fps = float(motion_rep.fps)

    target_global = safe_cont6d_to_matrix(
        target[:, 2:5, BODY_ROT6D].reshape(batch, 3, 27, 6)
    )
    prediction_first_global = safe_cont6d_to_matrix(
        prediction[:, 4:5, BODY_ROT6D].reshape(batch, 1, 27, 6)
    )
    target_local = global_rots_to_local_rots(target_global, motion_rep.skeleton)
    prediction_first_local = global_rots_to_local_rots(
        prediction_first_global, motion_rep.skeleton
    )
    _, target_joints, _ = fk(
        target_local,
        target_root[:, 2:5, :3],
        motion_rep.skeleton,
    )
    _, prediction_first_joints, _ = fk(
        prediction_first_local,
        prediction_root[:, 4:5, :3],
        motion_rep.skeleton,
    )

    # Frames 2/3 are retained verbatim by the deployed buffer update.  Writing
    # the complete jump equation makes the equivalence to evaluate_rollout.py
    # explicit even though the shared pre-seam velocity cancels algebraically.
    target_joint_velocity_before = (target_joints[:, 1] - target_joints[:, 0]) * fps
    target_joint_velocity_after = (target_joints[:, 2] - target_joints[:, 1]) * fps
    prediction_joint_velocity_after = (
        prediction_first_joints[:, 0] - target_joints[:, 1]
    ) * fps
    joint_jump_error = torch.linalg.vector_norm(
        (prediction_joint_velocity_after - target_joint_velocity_before)
        - (target_joint_velocity_after - target_joint_velocity_before),
        dim=-1,
    )

    target_root_velocity_before = (target_root[:, 3, :3] - target_root[:, 2, :3]) * fps
    target_root_velocity_after = (target_root[:, 4, :3] - target_root[:, 3, :3]) * fps
    prediction_root_velocity_after = (
        prediction_root[:, 4, :3] - target_root[:, 3, :3]
    ) * fps
    root_jump_error = torch.linalg.vector_norm(
        (prediction_root_velocity_after - target_root_velocity_before)
        - (target_root_velocity_after - target_root_velocity_before),
        dim=-1,
    )

    first_frame_fk = torch.linalg.vector_norm(
        prediction_first_joints[:, 0] - target_joints[:, 2], dim=-1
    )
    relative_rotation = (
        prediction_first_global[:, 0].transpose(-1, -2) @ target_global[:, 2]
    )
    cosine = (
        (relative_rotation.diagonal(dim1=-2, dim2=-1).sum(-1) - 1.0) * 0.5
    ).clamp(-1 + 1.0e-6, 1 - 1.0e-6)
    rotation_geodesic = torch.acos(cosine)

    root_jump = masked_mean(root_jump_error, valid)
    joint_jump = masked_mean(joint_jump_error, valid)
    first_frame_fk_mpjpe = masked_mean(first_frame_fk, valid)
    first_frame_rotation = masked_mean(rotation_geodesic, valid)
    return {
        "physical_seam_root_jump_error_mps": root_jump,
        "physical_seam_joint_jump_error_mps": joint_jump,
        "physical_seam_first_frame_fk_mpjpe": first_frame_fk_mpjpe,
        "physical_seam_rotation_geodesic": first_frame_rotation,
        "physical_seam_total": root_jump + joint_jump + 0.10 * first_frame_rotation,
    }


def flow_velocity_losses(
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> dict[str, torch.Tensor]:
    root_error = prediction[..., :20] - target[..., :20]
    body_error = prediction[..., 20:] - target[..., 20:]
    root_mse = root_error.square().mean()
    body_mse = body_error.square().mean()
    root_l1 = root_error.abs().mean()
    body_l1 = body_error.abs().mean()
    return {
        "flow_root_mse": root_mse,
        "flow_body_mse": body_mse,
        "flow_root_l1": root_l1,
        "flow_body_l1": body_l1,
        "flow_total": 2.0 * root_mse + body_mse + 0.25 * (root_l1 + body_l1),
    }


def root_temporal_losses(
    prediction: torch.Tensor,
    target: torch.Tensor,
    motion_rep,
) -> dict[str, torch.Tensor]:
    """Match physical root derivatives for a forty-frame clean endpoint.

    Framewise endpoint MSE assigns almost no special cost to alternating root
    errors.  At 20 FPS those errors become very large accelerations and jerk in
    the renderer.  These losses operate after any endpoint projection and use
    physical metres plus wrapped heading deltas, so their scale has a direct
    motion interpretation.
    """

    if prediction.shape[-2:] != (10, 148) or target.shape[-2:] != (10, 148):
        raise ValueError(
            "prediction and target must both have shape [B,10,148], got "
            f"{tuple(prediction.shape)} and {tuple(target.shape)}"
        )
    batch = prediction.shape[0]
    prediction_root = motion_rep.global_root_stats.unnormalize(
        prediction[..., :20].reshape(batch, 40, 5).float()
    )
    target_root = motion_rep.global_root_stats.unnormalize(
        target[..., :20].reshape(batch, 40, 5).float()
    )

    prediction_position = prediction_root[..., :3]
    target_position = target_root[..., :3]
    prediction_velocity = torch.diff(prediction_position, dim=1)
    target_velocity = torch.diff(target_position, dim=1)
    prediction_acceleration = torch.diff(prediction_velocity, dim=1)
    target_acceleration = torch.diff(target_velocity, dim=1)
    prediction_jerk = torch.diff(prediction_acceleration, dim=1)
    target_jerk = torch.diff(target_acceleration, dim=1)

    position_velocity = (prediction_velocity - target_velocity).abs().mean()
    position_acceleration = (prediction_acceleration - target_acceleration).abs().mean()
    position_jerk = (prediction_jerk - target_jerk).abs().mean()

    prediction_heading_raw = prediction_root[..., 3:5]
    target_heading_raw = target_root[..., 3:5]
    prediction_heading = F.normalize(prediction_heading_raw, dim=-1, eps=1.0e-6)
    target_heading = F.normalize(target_heading_raw, dim=-1, eps=1.0e-6)

    def heading_delta(heading: torch.Tensor) -> torch.Tensor:
        previous = heading[:, :-1]
        current = heading[:, 1:]
        cross = previous[..., 0] * current[..., 1] - previous[..., 1] * current[..., 0]
        dot = (previous * current).sum(dim=-1)
        return torch.atan2(cross, dot)

    prediction_turn = heading_delta(prediction_heading)
    target_turn = heading_delta(target_heading)
    heading_velocity = (prediction_turn - target_turn).abs().mean()
    heading_acceleration = (
        torch.diff(prediction_turn, dim=1) - torch.diff(target_turn, dim=1)
    ).abs().mean()
    heading_unit = (torch.linalg.vector_norm(prediction_heading_raw, dim=-1) - 1.0).abs().mean()

    total = (
        2.0 * position_velocity
        + 4.0 * position_acceleration
        + 8.0 * position_jerk
        + 0.25 * heading_velocity
        + 0.50 * heading_acceleration
        + 0.10 * heading_unit
    )
    return {
        "root_position_delta_l1_m": position_velocity,
        "root_position_second_delta_l1_m": position_acceleration,
        "root_position_third_delta_l1_m": position_jerk,
        "root_heading_delta_l1_rad": heading_velocity,
        "root_heading_second_delta_l1_rad": heading_acceleration,
        "root_heading_unit_l1": heading_unit,
        "root_temporal_total": total,
    }
