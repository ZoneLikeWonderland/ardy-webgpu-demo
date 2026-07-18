#!/usr/bin/env python
"""Audit raw and weighted loss-gradient scales before distillation.

This is intentionally a diagnostic, not a trainer.  It evaluates one fixed
batch in FP32 and reports how strongly each objective acts on the flow or
decoder parameters.  The endpoint path uses the same differentiable Euler
solver as training, including NFE > 1.  The resulting ratios make it possible
to choose temporal/path/physical coefficients from evidence instead of
comparing loss numbers with incompatible units.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from itertools import islice
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from ardy_distill.codec_cli import (
    add_codec_config_arguments,
    codec_config_from_args,
)
from ardy_distill.data import TeacherShardDataset
from ardy_distill.dmd2 import path_constraint_losses, seam_losses
from ardy_distill.losses import (
    FSQRequantizer,
    decoder_feature_losses,
    deployment_decoder_roots,
    flow_velocity_losses,
    fsq_endpoint_losses,
    motion_quality_losses,
    physical_seam_losses,
    root_temporal_losses,
)
from ardy_distill.flow_matching import euler_flow_denoise
from ardy_distill.models import (
    FlowStudentConfig,
    HistoryEncoderStudent,
    MotionDecoderStudent,
    OneStepFlowStudent,
    project_root_trajectory,
)
from ardy_distill.runtime import load_motion_rep, load_safetensor_weights
from ardy_distill.train_dmd2 import boundary_motion_quality_total
from ardy_distill.train_flow import endpoint_losses


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--encoder", type=Path, required=True)
    parser.add_argument("--decoder", type=Path, required=True)
    parser.add_argument("--flow", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path("ardy/checkpoints/ARDY-Core-RP-20FPS-Horizon40"),
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--batch-index",
        type=int,
        default=0,
        help="Zero-based deterministic DataLoader batch to audit.",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--flow-width", type=int, default=512)
    parser.add_argument("--flow-heads", type=int, default=8)
    parser.add_argument("--flow-trunk-blocks", type=int, default=8)
    parser.add_argument("--flow-body-blocks", type=int, default=8)
    parser.add_argument(
        "--solver-steps",
        type=int,
        default=1,
        help="Differentiable uniform-Euler NFE used by endpoint objectives.",
    )
    parser.add_argument("--flow-root-smoothing-passes", type=int, default=0)
    parser.add_argument(
        "--flow-root-projection-kind",
        choices=["binomial", "cubic_controls"],
        default="binomial",
    )
    parser.add_argument("--flow-root-control-points", type=int, default=10)
    add_codec_config_arguments(parser)
    parser.add_argument("--decoder-weight", type=float, default=0.1)
    parser.add_argument("--fsq-weight", type=float, default=1.0)
    parser.add_argument("--path-weight", type=float, default=0.1)
    parser.add_argument("--seam-weight", type=float, default=0.01)
    parser.add_argument("--physical-seam-weight", type=float, default=0.001)
    parser.add_argument("--root-temporal-weight", type=float, default=0.01)
    parser.add_argument("--quality-weight", type=float, default=0.10)
    parser.add_argument(
        "--quality-profile",
        choices=["compact", "dmd2_boundary100k"],
        default="compact",
        help="Use the exact DMD2 boundary-100k quality mix when auditing that stage.",
    )
    parser.add_argument("--foot-slide-quality-weight", type=float, default=0.1)
    parser.add_argument(
        "--clip-threshold",
        type=float,
        default=1.0,
        help="Training clip threshold, reported for scale interpretation only.",
    )
    return parser.parse_args()


def parameter_group(name: str, module_name: str) -> str:
    if module_name == "flow":
        if name.startswith(("root_head", "root_injection")):
            return "root_head"
        if name.startswith(("body_refiner", "body_head")):
            return "body_head"
        if name.startswith("trunk"):
            return "shared_trunk"
        return "conditioning_and_embeddings"
    if name.startswith("input_proj"):
        return "input"
    if name.startswith("blocks"):
        return "temporal_blocks"
    return "output"


def trainable_named_parameters(
    module: torch.nn.Module,
) -> list[tuple[str, torch.nn.Parameter]]:
    return [
        (name, parameter)
        for name, parameter in module.named_parameters()
        if parameter.requires_grad
    ]


def loss_gradients(
    loss: torch.Tensor,
    named_parameters: list[tuple[str, torch.nn.Parameter]],
) -> tuple[torch.Tensor | None, ...]:
    return torch.autograd.grad(
        loss,
        [parameter for _, parameter in named_parameters],
        retain_graph=True,
        allow_unused=True,
    )


def summarize_gradients(
    loss: torch.Tensor,
    named_parameters: list[tuple[str, torch.nn.Parameter]],
    gradients: tuple[torch.Tensor | None, ...],
    module_name: str,
) -> dict:
    squared_by_group: dict[str, float] = defaultdict(float)
    maximum = 0.0
    nonzero_parameters = 0
    nonfinite_elements = 0
    for (name, _), gradient in zip(named_parameters, gradients):
        if gradient is None:
            continue
        detached = gradient.detach().float()
        finite_mask = torch.isfinite(detached)
        nonfinite_elements += int((~finite_mask).sum())
        finite = torch.where(finite_mask, detached, torch.zeros_like(detached))
        squared_by_group[parameter_group(name, module_name)] += float(finite.square().sum())
        maximum = max(maximum, float(finite.abs().max()))
        if bool(finite.count_nonzero()):
            nonzero_parameters += 1
    group_l2 = {name: value**0.5 for name, value in sorted(squared_by_group.items())}
    return {
        "loss_value": float(loss.detach()),
        "gradient_l2": sum(squared_by_group.values()) ** 0.5,
        "gradient_max_abs": maximum,
        "all_finite": nonfinite_elements == 0,
        "nonfinite_elements": nonfinite_elements,
        "groups_l2": group_l2,
        "nonzero_parameter_tensors": nonzero_parameters,
    }


def gradient_summary(
    loss: torch.Tensor,
    module: torch.nn.Module,
    module_name: str,
) -> dict:
    named_parameters = trainable_named_parameters(module)
    gradients = loss_gradients(loss, named_parameters)
    return summarize_gradients(loss, named_parameters, gradients, module_name)


def gradient_cosine(
    first: tuple[torch.Tensor | None, ...],
    second: tuple[torch.Tensor | None, ...],
) -> float | None:
    dot = torch.zeros((), device=next(item for item in first if item is not None).device)
    first_squared = torch.zeros_like(dot)
    second_squared = torch.zeros_like(dot)
    for first_gradient, second_gradient in zip(first, second, strict=True):
        if first_gradient is None or second_gradient is None:
            continue
        first_float = torch.nan_to_num(first_gradient.detach().float())
        second_float = torch.nan_to_num(second_gradient.detach().float())
        dot = dot + (first_float * second_float).sum()
        first_squared = first_squared + first_float.square().sum()
        second_squared = second_squared + second_float.square().sum()
    denominator = (first_squared * second_squared).sqrt()
    if float(denominator) == 0.0:
        return None
    return float(dot / denominator)


def attach_reference_ratios(rows: dict[str, dict], reference: str) -> None:
    reference_norm = rows[reference]["gradient_l2"]
    for row in rows.values():
        norm = row["gradient_l2"]
        row["gradient_ratio_to_reference"] = (
            norm / reference_norm if reference_norm > 0 else None
        )
        row["coefficient_for_reference_gradient"] = (
            reference_norm / norm if norm > 0 else None
        )


def derivative_rms(
    values: torch.Tensor,
    order: int,
    valid: torch.Tensor | None = None,
) -> float:
    result = values.float()
    derivative_valid = valid
    for _ in range(order):
        result = torch.diff(result, dim=1)
        if derivative_valid is not None:
            derivative_valid = derivative_valid[:, 1:] * derivative_valid[:, :-1]
    if derivative_valid is None:
        return float(result.square().mean().sqrt())
    mask = derivative_valid
    while mask.ndim < result.ndim:
        mask = mask.unsqueeze(-1)
    mask = mask.expand_as(result).to(dtype=result.dtype)
    return float(((result.square() * mask).sum() / mask.sum().clamp_min(1)).sqrt())


def main() -> None:
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("batch size must be positive")
    if args.batch_index < 0:
        raise ValueError("batch index must be non-negative")
    if args.solver_steps < 1:
        raise ValueError("solver steps must be positive")
    nonnegative = {
        "decoder_weight": args.decoder_weight,
        "fsq_weight": args.fsq_weight,
        "path_weight": args.path_weight,
        "seam_weight": args.seam_weight,
        "physical_seam_weight": args.physical_seam_weight,
        "root_temporal_weight": args.root_temporal_weight,
        "quality_weight": args.quality_weight,
        "foot_slide_quality_weight": args.foot_slide_quality_weight,
        "clip_threshold": args.clip_threshold,
    }
    for name, value in nonnegative.items():
        if value < 0:
            raise ValueError(f"{name} must be non-negative")
    device = torch.device(args.device)
    torch.manual_seed(20260714)

    dataset = TeacherShardDataset(args.data, cache_shards=2)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    try:
        batch = next(islice(loader, args.batch_index, None))
    except StopIteration as error:
        raise ValueError(
            f"batch index {args.batch_index} is outside a dataset of {len(dataset)} samples"
        ) from error
    batch = {name: value.to(device=device).float() for name, value in batch.items()}
    motion_rep = load_motion_rep(args.checkpoint_dir)
    quantizer = FSQRequantizer(args.checkpoint_dir / "stats/post_quantization").to(device)

    codec_config = codec_config_from_args(args)
    encoder = (
        HistoryEncoderStudent(codec_config)
        .to(device)
        .eval()
        .requires_grad_(False)
    )
    decoder = MotionDecoderStudent(codec_config).to(device).eval()
    flow = OneStepFlowStudent(
        FlowStudentConfig(
            width=args.flow_width,
            heads=args.flow_heads,
            trunk_blocks=args.flow_trunk_blocks,
            body_blocks=args.flow_body_blocks,
            root_smoothing_passes=args.flow_root_smoothing_passes,
            root_projection_kind=args.flow_root_projection_kind,
            root_control_points=args.flow_root_control_points,
        )
    ).to(device).eval()
    load_safetensor_weights(encoder, args.encoder)
    load_safetensor_weights(decoder, args.decoder)
    load_safetensor_weights(flow, args.flow)

    with torch.no_grad():
        encoded_history = quantizer(encoder(batch["encoder_body"]), ste=False)
    history_valid = batch["has_history"].unsqueeze(-1)
    history_hybrid = torch.cat(
        [
            batch["history_hybrid"][..., :20] * history_valid,
            encoded_history * history_valid,
        ],
        dim=-1,
    )

    flow_time = torch.ones_like(batch["has_history"])
    velocity = flow(
        batch["initial_noise"],
        history_hybrid,
        batch["path_condition"],
        batch["first_heading"],
        batch["has_history"],
        flow_time,
    )
    endpoint = euler_flow_denoise(
        flow,
        batch["initial_noise"],
        history_hybrid,
        batch["path_condition"],
        batch["first_heading"],
        batch["has_history"],
        steps=args.solver_steps,
    )
    endpoint = project_root_trajectory(
        endpoint,
        args.flow_root_smoothing_passes,
        kind=args.flow_root_projection_kind,
        control_points=args.flow_root_control_points,
        basis=flow.root_projection_basis,
    )
    velocity_losses = flow_velocity_losses(
        velocity,
        batch["initial_noise"] - batch["clean_generation"],
    )
    paired = endpoint_losses(endpoint, batch["clean_generation"])
    fsq = fsq_endpoint_losses(
        endpoint[..., 20:],
        batch["clean_generation"][..., 20:],
        quantizer,
    )
    path = path_constraint_losses(endpoint, batch["path_condition"], batch["has_history"])
    root_temporal = root_temporal_losses(endpoint, batch["clean_generation"], motion_rep)
    predicted_root, predicted_local_root = deployment_decoder_roots(
        endpoint,
        history_hybrid,
        batch["has_history"],
        motion_rep,
    )
    flow_latent = torch.cat(
        [history_hybrid[..., 20:], quantizer(endpoint[..., 20:], ste=True)],
        dim=1,
    )
    decoded_flow = decoder(
        flow_latent,
        predicted_local_root,
        batch["decoder_token_valid"],
    )
    decoded_flow_losses = decoder_feature_losses(
        decoded_flow,
        batch["target_body"],
        batch["decoder_token_valid"],
    )
    flow_quality = motion_quality_losses(
        decoded_flow,
        batch["target_body"],
        predicted_root,
        batch["decoder_token_valid"],
        motion_rep,
        target_normalized_global_root=batch["decoder_global_root"],
    )
    flow_seam = seam_losses(
        endpoint,
        decoded_flow,
        history_hybrid,
        batch["decoder_global_root"],
        batch["target_body"],
        batch["has_history"],
    )
    flow_physical_seam = physical_seam_losses(
        decoded_flow,
        batch["target_body"],
        predicted_root,
        batch["decoder_global_root"],
        batch["has_history"],
        motion_rep,
    )
    if args.quality_profile == "dmd2_boundary100k":
        flow_quality_total = boundary_motion_quality_total(flow_quality)
    else:
        flow_quality_total = (
            0.10 * flow_quality["rotation_geodesic"]
            + flow_quality["fk_mpjpe"]
            + 0.10 * flow_quality["joint_velocity_l1"]
            + 0.05 * flow_quality["joint_acceleration_l1"]
            + args.foot_slide_quality_weight * flow_quality["foot_slide"]
        )

    flow_objectives = {
        "flow_velocity_total": velocity_losses["flow_total"],
        "endpoint_total": paired["endpoint_total"],
        "fsq_quantized_l1": fsq["fsq_quantized_l1"],
        "path_mse": path["path_constraint_mse"],
        "root_temporal_total": root_temporal["root_temporal_total"],
        "decoded_feature_total": decoded_flow_losses["decoder_total"],
        "seam_total": flow_seam["seam_total"],
        "physical_seam_total": flow_physical_seam["physical_seam_total"],
        "quality_total": flow_quality_total,
        "fk_mpjpe": flow_quality["fk_mpjpe"],
        "joint_velocity": flow_quality["joint_velocity_l1"],
        "joint_acceleration": flow_quality["joint_acceleration_l1"],
        "joint_jerk": flow_quality["joint_jerk_l1"],
        "rotation_velocity": flow_quality["rotation_velocity_l1"],
        "rotation_acceleration": flow_quality["rotation_acceleration_l1"],
        "rotation_jerk": flow_quality["rotation_jerk_l1"],
        "joint_boundary_velocity": flow_quality["joint_boundary_velocity_l1"],
        "joint_boundary_acceleration": flow_quality["joint_boundary_acceleration_l1"],
        "joint_boundary_jerk": flow_quality["joint_boundary_jerk_l1"],
        "rotation_boundary_velocity": flow_quality["rotation_boundary_velocity_l1"],
        "rotation_boundary_acceleration": flow_quality["rotation_boundary_acceleration_l1"],
        "rotation_boundary_jerk": flow_quality["rotation_boundary_jerk_l1"],
        "foot_slide": flow_quality["foot_slide"],
    }
    flow_rows = {
        name: gradient_summary(loss, flow, "flow")
        for name, loss in flow_objectives.items()
    }
    attach_reference_ratios(flow_rows, "endpoint_total")

    weighted_components = {
        "flow_velocity_total": flow_objectives["flow_velocity_total"],
        "endpoint_total": flow_objectives["endpoint_total"],
        "fsq_quantized_l1": (
            args.fsq_weight * flow_objectives["fsq_quantized_l1"]
        ),
        "path_mse": args.path_weight * flow_objectives["path_mse"],
        "root_temporal_total": (
            args.root_temporal_weight * flow_objectives["root_temporal_total"]
        ),
        "decoded_feature_total": (
            args.decoder_weight * flow_objectives["decoded_feature_total"]
        ),
        "seam_total": args.seam_weight * flow_objectives["seam_total"],
        "physical_seam_total": (
            args.physical_seam_weight * flow_objectives["physical_seam_total"]
        ),
        "quality_total": args.quality_weight * flow_objectives["quality_total"],
    }
    supervised_core = (
        weighted_components["flow_velocity_total"]
        + weighted_components["endpoint_total"]
        + weighted_components["fsq_quantized_l1"]
    )
    auxiliary_total = sum(
        loss
        for name, loss in weighted_components.items()
        if name not in {
            "flow_velocity_total",
            "endpoint_total",
            "fsq_quantized_l1",
        }
    )
    combined_total = supervised_core + auxiliary_total
    flow_named_parameters = trainable_named_parameters(flow)
    supervised_gradients = loss_gradients(supervised_core, flow_named_parameters)
    weighted_rows = {}
    for name, loss in {
        **weighted_components,
        "supervised_core": supervised_core,
        "auxiliary_total": auxiliary_total,
        "combined_total": combined_total,
    }.items():
        gradients = (
            supervised_gradients
            if name == "supervised_core"
            else loss_gradients(loss, flow_named_parameters)
        )
        row = summarize_gradients(
            loss,
            flow_named_parameters,
            gradients,
            "flow",
        )
        row["gradient_cosine_to_supervised_core"] = gradient_cosine(
            gradients,
            supervised_gradients,
        )
        weighted_rows[name] = row
    attach_reference_ratios(weighted_rows, "combined_total")

    decoder.zero_grad(set_to_none=True)
    decoded_teacher = decoder(
        batch["decoder_latent"],
        batch["decoder_local_root"],
        batch["decoder_token_valid"],
    )
    decoder_features = decoder_feature_losses(
        decoded_teacher,
        batch["target_body"],
        batch["decoder_token_valid"],
    )
    decoder_quality = motion_quality_losses(
        decoded_teacher,
        batch["target_body"],
        batch["decoder_global_root"],
        batch["decoder_token_valid"],
        motion_rep,
    )
    decoder_objectives = {
        "feature_total": decoder_features["decoder_total"],
        "body_l1": decoder_features["decoder_body_l1"],
        "rotation_geodesic": decoder_quality["rotation_geodesic"],
        "fk_mpjpe": decoder_quality["fk_mpjpe"],
        "joint_velocity": decoder_quality["joint_velocity_l1"],
        "joint_acceleration": decoder_quality["joint_acceleration_l1"],
        "joint_jerk": decoder_quality["joint_jerk_l1"],
        "rotation_velocity": decoder_quality["rotation_velocity_l1"],
        "rotation_acceleration": decoder_quality["rotation_acceleration_l1"],
        "rotation_jerk": decoder_quality["rotation_jerk_l1"],
        "joint_boundary_velocity": decoder_quality["joint_boundary_velocity_l1"],
        "joint_boundary_acceleration": decoder_quality["joint_boundary_acceleration_l1"],
        "joint_boundary_jerk": decoder_quality["joint_boundary_jerk_l1"],
        "rotation_boundary_velocity": decoder_quality["rotation_boundary_velocity_l1"],
        "rotation_boundary_acceleration": decoder_quality["rotation_boundary_acceleration_l1"],
        "rotation_boundary_jerk": decoder_quality["rotation_boundary_jerk_l1"],
        "foot_slide": decoder_quality["foot_slide"],
    }
    decoder_rows = {
        name: gradient_summary(loss, decoder, "decoder")
        for name, loss in decoder_objectives.items()
    }
    attach_reference_ratios(decoder_rows, "feature_total")

    batch_size = endpoint.shape[0]
    teacher_root = motion_rep.global_root_stats.unnormalize(
        batch["clean_generation"][..., :20].reshape(batch_size, 40, 5)
    )
    target_body_physical = motion_rep.body_stats.unnormalize(batch["target_body"])
    target_root_physical = motion_rep.global_root_stats.unnormalize(
        batch["decoder_global_root"]
    )
    target_motion = torch.cat([target_root_physical, target_body_physical], dim=-1)
    target_joints = motion_rep.inverse(target_motion, is_normalized=False)["posed_joints"]
    target_frame_valid = batch["decoder_token_valid"].repeat_interleave(4, dim=1)
    teacher_scales = {
        "root_position_delta_rms_m_per_frame": derivative_rms(teacher_root[..., :3], 1),
        "root_position_second_delta_rms_m_per_frame2": derivative_rms(
            teacher_root[..., :3], 2
        ),
        "root_position_third_delta_rms_m_per_frame3": derivative_rms(
            teacher_root[..., :3], 3
        ),
        "joint_delta_rms_m_per_frame": derivative_rms(
            target_joints, 1, target_frame_valid
        ),
        "joint_second_delta_rms_m_per_frame2": derivative_rms(
            target_joints, 2, target_frame_valid
        ),
        "joint_third_delta_rms_m_per_frame3": derivative_rms(
            target_joints, 3, target_frame_valid
        ),
    }

    history = batch["has_history"].reshape(batch_size) > 0.5
    path_valid = batch["path_condition"][..., 2] > 0.5
    path_frames = torch.arange(path_valid.shape[1], device=device).unsqueeze(0)
    generation_start = torch.where(
        history,
        torch.full_like(history, 4, dtype=torch.long),
        torch.zeros_like(history, dtype=torch.long),
    )
    path_inside = path_valid & (path_frames >= generation_start.unsqueeze(1)) & (
        path_frames < generation_start.unsqueeze(1) + 40
    )
    path_future = path_valid & (path_frames >= generation_start.unsqueeze(1) + 40)

    result = {
        "schema": "ardy_loss_gradient_audit_v3",
        "batch_size": args.batch_size,
        "batch_index": args.batch_index,
        "batch_composition": {
            "history_samples": int(history.sum()),
            "initial_samples": int((~history).sum()),
            "path_constraints": int(path_valid.sum()),
            "path_constraints_inside_generation": int(path_inside.sum()),
            "path_constraints_after_generation": int(path_future.sum()),
        },
        "device": str(device),
        "solver_steps": args.solver_steps,
        "flow_root_smoothing_passes": args.flow_root_smoothing_passes,
        "codec_config": {
            "encoder_width": codec_config.encoder_width,
            "encoder_blocks": codec_config.encoder_blocks,
            "decoder_width": codec_config.decoder_width,
            "decoder_blocks": codec_config.decoder_blocks,
            "decoder_token_hidden": codec_config.decoder_token_hidden,
            "expansion": codec_config.expansion,
        },
        "quality_profile": args.quality_profile,
        "training_coefficients": {
            "flow_velocity_total": 1.0,
            "endpoint_total": 1.0,
            "fsq_quantized_l1": args.fsq_weight,
            "path_mse": args.path_weight,
            "root_temporal_total": args.root_temporal_weight,
            "decoded_feature_total": args.decoder_weight,
            "seam_total": args.seam_weight,
            "physical_seam_total": args.physical_seam_weight,
            "quality_total": args.quality_weight,
            "foot_slide_inside_quality": args.foot_slide_quality_weight,
        },
        "clip_threshold": args.clip_threshold,
        "weights": {
            "encoder": str(args.encoder),
            "decoder": str(args.decoder),
            "flow": str(args.flow),
        },
        "teacher_temporal_rms": teacher_scales,
        "flow_objectives": flow_rows,
        "weighted_flow_objectives": weighted_rows,
        "combined_gradient_to_clip_ratio": (
            weighted_rows["combined_total"]["gradient_l2"] / args.clip_threshold
            if args.clip_threshold > 0
            else None
        ),
        "decoder_objectives": decoder_rows,
        "interpretation": {
            "coefficient_for_reference_gradient": (
                "coefficient that would give this raw objective the same total parameter-gradient "
                "L2 as endpoint_total (flow) or feature_total (decoder) on this audit batch"
            ),
            "warning": (
                "equal gradient norms are a calibration reference, not an automatic final loss recipe"
            ),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
