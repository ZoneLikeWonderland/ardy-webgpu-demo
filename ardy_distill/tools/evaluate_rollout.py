#!/usr/bin/env python
"""Faithful infinite-demo rollout comparison for teacher and compact student."""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import save_file

from ardy.constraints import Root2DConstraintSet
from ardy.model import load_model
from ardy_distill.codec_cli import add_codec_config_arguments, codec_config_from_args
from ardy_distill.losses import BODY_CONTACT, BODY_ROT6D, FSQRequantizer, safe_cont6d_to_matrix
from ardy_distill.models import (
    FlowStudentConfig,
    HistoryEncoderStudent,
    MotionDecoderStudent,
    OneStepFlowStudent,
)
from ardy_distill.runtime import load_safetensor_weights
from ardy_distill.student_runtime import StudentArdyRuntime
from ardy_distill.teacher import trace_autoregressive_step
from scripts.interactive_demo.window_budget import compute_window_num_frames


@dataclass
class Waypoint:
    frame: int
    target_xz: torch.Tensor
    speed_mps: float
    direction_rad: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--encoder", type=Path, required=True)
    parser.add_argument("--flow", type=Path, required=True)
    parser.add_argument("--decoder", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checkpoints-dir", type=Path, default=Path("ardy/checkpoints"))
    parser.add_argument("--model", default="ARDY-Core-RP-20FPS-Horizon40")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--model-dtype", choices=["fp32", "fp16", "bf16"], default="fp32")
    parser.add_argument("--windows", type=int, default=50)
    parser.add_argument("--checkpoints", type=int, nargs="+", default=[1, 5, 20, 50])
    parser.add_argument(
        "--log-every",
        type=int,
        default=1,
        help="Print one detailed rollout event every N windows; metrics still retain every event.",
    )
    parser.add_argument("--seed", type=int, default=20260714)
    parser.add_argument("--cfg-constraint", type=float, default=1.5)
    parser.add_argument("--waypoint-interval", type=int, default=60)
    parser.add_argument("--waypoint-speed-min", type=float, default=0.65)
    parser.add_argument("--waypoint-speed-max", type=float, default=1.15)
    parser.add_argument("--turn-std", type=float, default=0.65)
    parser.add_argument("--flow-width", type=int, default=384)
    parser.add_argument("--flow-heads", type=int, default=6)
    parser.add_argument("--flow-trunk-blocks", type=int, default=4)
    parser.add_argument("--flow-body-blocks", type=int, default=2)
    parser.add_argument("--flow-steps", type=int, default=1)
    parser.add_argument("--flow-root-smoothing-passes", type=int, default=0)
    parser.add_argument(
        "--flow-root-projection-kind",
        choices=["binomial", "cubic_controls"],
        default="binomial",
    )
    parser.add_argument("--flow-root-control-points", type=int, default=10)
    parser.add_argument(
        "--text-feature-dim",
        type=int,
        default=0,
        help=(
            "External pooled text-feature width expected by the flow. The rollout "
            "uses an all-zero/unconditional feature when this is non-zero."
        ),
    )
    parser.add_argument(
        "--heading-condition-features",
        type=int,
        choices=[0, 3],
        default=0,
        help=(
            "Per-frame future-heading feature width expected by the flow. This "
            "mouse-waypoint rollout leaves the optional heading condition at zero."
        ),
    )
    add_codec_config_arguments(parser)
    parser.add_argument(
        "--student-inertialization-frames",
        type=int,
        default=0,
        help=(
            "Deployment-side feature-space inertialization after each student seam; "
            "zero disables it."
        ),
    )
    parser.add_argument(
        "--student-inertialization-strength",
        type=float,
        default=1.0,
        help="Fraction of the initial constant-velocity seam offset to remove.",
    )
    parser.add_argument(
        "--on-policy-teacher-diagnostic",
        action="store_true",
        help=(
            "Also query the released teacher from the student's current four-frame "
            "history and compare recovery toward the same world-space waypoint."
        ),
    )
    return parser.parse_args()


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def percentile(values: list[float], quantile: float) -> float | None:
    if not values:
        return None
    return float(np.percentile(np.asarray(values, dtype=np.float64), quantile))


def numeric_summary(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "mean": None, "p50": None, "p95": None, "max": None}
    return {
        "count": len(values),
        "mean": float(np.mean(values)),
        "p50": percentile(values, 50),
        "p95": percentile(values, 95),
        "max": float(np.max(values)),
    }


def tensor_abs_summary(tensor: torch.Tensor) -> dict[str, float]:
    values = tensor.detach().float().abs()
    return {
        "mean": float(values.mean()),
        "max": float(values.max()),
        "l2": float(torch.linalg.vector_norm(values.reshape(values.shape[0], -1), dim=-1).mean()),
    }


def path_condition_summary(path: torch.Tensor) -> dict[str, float | int | None]:
    path = path.detach().float()
    valid = path[..., 2] > 0.5
    distances = torch.linalg.vector_norm(path[..., :2], dim=-1)[valid]
    if not distances.numel():
        return {"count": 0, "mean_norm": None, "max_norm": None}
    return {
        "count": int(distances.numel()),
        "mean_norm": float(distances.mean()),
        "max_norm": float(distances.max()),
    }


def root_positions(buffer: torch.Tensor, motion_rep) -> torch.Tensor:
    return motion_rep.get_root_pos(motion_rep.unnormalize(buffer.float()))


def make_waypoint(
    *,
    frame_index: int,
    teacher_buffer: torch.Tensor | None,
    initial_translation: torch.Tensor,
    previous_direction: float,
    generator: torch.Generator,
    interval: int,
    fps: int,
    speed_min: float,
    speed_max: float,
    turn_std: float,
    motion_rep,
) -> Waypoint:
    if teacher_buffer is None:
        current_xz = initial_translation[0, [0, 2]].detach().cpu()
    else:
        positions = root_positions(teacher_buffer, motion_rep)
        current_xz = positions[0, min(frame_index, positions.shape[1] - 1), [0, 2]].detach().cpu()
    turn = float(torch.randn((), generator=generator) * turn_std)
    direction = previous_direction + turn
    speed = float(speed_min + torch.rand((), generator=generator) * (speed_max - speed_min))
    distance = speed * interval / fps
    delta = torch.tensor([math.sin(direction), math.cos(direction)], dtype=torch.float32) * distance
    return Waypoint(
        frame=max(0, frame_index) + interval,
        target_xz=current_xz + delta,
        speed_mps=speed,
        direction_rad=direction,
    )


def make_conditions(
    teacher,
    waypoint: Waypoint,
    *,
    history_start: int,
    history_end: int,
    num_frames: int,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    if waypoint.frame <= history_end:
        return None, None
    relative_frame = waypoint.frame - history_start
    if relative_frame < 0 or relative_frame >= num_frames:
        return None, None
    constraint = Root2DConstraintSet(
        teacher.skeleton,
        frame_indices=torch.tensor([relative_frame], dtype=torch.long),
        root_2d=waypoint.target_xz.reshape(1, 2),
    )
    lengths = torch.tensor([num_frames], device=teacher.device, dtype=torch.long)
    observed_motion, motion_mask = teacher.motion_rep.create_conditions_from_constraints_batched(
        [[constraint]],
        lengths,
        to_normalize=True,
        device=teacher.device,
    )
    return motion_mask, observed_motion


def kinematics(buffer: torch.Tensor, motion_rep) -> dict[str, torch.Tensor]:
    unnormalized = motion_rep.unnormalize(buffer.float())
    inverse = motion_rep.inverse(unnormalized, is_normalized=False)
    return {
        "unnormalized": unnormalized,
        "root": motion_rep.get_root_pos(unnormalized),
        "joints": inverse["posed_joints"],
        "rotations": inverse["global_rot_mats"],
        "contacts": motion_rep.extract_body(unnormalized)[..., BODY_CONTACT],
    }


def foot_slide(kinematic: dict[str, torch.Tensor], foot_indices: list[int], fps: int) -> float | None:
    joints = kinematic["joints"]
    if joints.shape[1] < 2:
        return None
    speed = torch.linalg.vector_norm(
        (joints[:, 1:, foot_indices] - joints[:, :-1, foot_indices]) * fps,
        dim=-1,
    )
    contact = kinematic["contacts"][:, 1:] > 0.5
    if not bool(contact.any()):
        return None
    return float(speed[contact].mean())


def seam_metrics(buffer: torch.Tensor, seam_frame: int, motion_rep, fps: int) -> dict[str, float]:
    kin = kinematics(buffer[:, seam_frame - 1 : seam_frame + 2], motion_rep)
    root = kin["root"]
    joints = kin["joints"]
    root_before = (root[:, 1] - root[:, 0]) * fps
    root_after = (root[:, 2] - root[:, 1]) * fps
    joint_before = (joints[:, 1] - joints[:, 0]) * fps
    joint_after = (joints[:, 2] - joints[:, 1]) * fps
    return {
        "root_velocity_jump_mps": float(torch.linalg.vector_norm(root_after - root_before, dim=-1).mean()),
        "joint_velocity_jump_mps": float(
            torch.linalg.vector_norm(joint_after - joint_before, dim=-1).mean()
        ),
    }


def inertialize_generated_motion(
    generated: torch.Tensor,
    history: torch.Tensor | None,
    history_length: int,
    *,
    frames: int,
    strength: float,
) -> torch.Tensor:
    """Decay the first generated-frame discontinuity without another network pass.

    The target at the seam is a one-frame constant-velocity extrapolation of
    the retained explicit history.  Its feature-space offset from the generated
    first frame is faded with a cubic smoothstep.  Contact logits are excluded
    because blending their four binary channels creates ambiguous contacts.
    """

    if frames <= 0 or history is None or history.shape[1] < 2 or strength <= 0:
        return generated
    generation_start = history_length
    available = generated.shape[1] - generation_start
    count = min(frames, available)
    if count <= 0:
        return generated

    result = generated.clone()
    feature_end = generated.shape[-1] - 4
    previous = history[:, -2].to(dtype=generated.dtype)
    current = history[:, -1].to(dtype=generated.dtype)
    constant_velocity_first = 2.0 * current - previous
    offset = (
        constant_velocity_first[:, :feature_end]
        - generated[:, generation_start, :feature_end]
    )
    if count == 1:
        decay = generated.new_ones(1)
    else:
        phase = torch.arange(count, device=generated.device, dtype=generated.dtype)
        phase = phase / float(count - 1)
        decay = 1.0 - (3.0 * phase.square() - 2.0 * phase.pow(3))
    result[:, generation_start : generation_start + count, :feature_end] += (
        strength * decay.reshape(1, count, 1) * offset.unsqueeze(1)
    )
    return result


def snapshot_metrics(
    teacher_buffer: torch.Tensor,
    student_buffer: torch.Tensor,
    waypoints: list[Waypoint],
    teacher_seams: list[dict[str, float]],
    student_seams: list[dict[str, float]],
    motion_rep,
    fps: int,
) -> dict:
    teacher_kin = kinematics(teacher_buffer, motion_rep)
    student_kin = kinematics(student_buffer, motion_rep)
    root_delta = torch.linalg.vector_norm(
        student_kin["root"][..., [0, 2]] - teacher_kin["root"][..., [0, 2]],
        dim=-1,
    )
    joint_delta = torch.linalg.vector_norm(
        student_kin["joints"] - teacher_kin["joints"],
        dim=-1,
    )
    relative_rotation = student_kin["rotations"].transpose(-1, -2) @ teacher_kin["rotations"]
    cosine = (
        (relative_rotation.diagonal(dim1=-2, dim2=-1).sum(-1) - 1.0) * 0.5
    ).clamp(-1 + 1.0e-6, 1 - 1.0e-6)
    rotation = torch.acos(cosine)

    teacher_waypoint_errors: list[float] = []
    student_waypoint_errors: list[float] = []
    reached_waypoints = []
    for waypoint in waypoints:
        if waypoint.frame >= teacher_buffer.shape[1]:
            continue
        target = waypoint.target_xz.to(device=teacher_buffer.device)
        teacher_error = float(
            torch.linalg.vector_norm(teacher_kin["root"][0, waypoint.frame, [0, 2]] - target)
        )
        student_error = float(
            torch.linalg.vector_norm(student_kin["root"][0, waypoint.frame, [0, 2]] - target)
        )
        teacher_waypoint_errors.append(teacher_error)
        student_waypoint_errors.append(student_error)
        reached_waypoints.append(
            {
                "frame": waypoint.frame,
                "target_xz": waypoint.target_xz.tolist(),
                "teacher_error_m": teacher_error,
                "student_error_m": student_error,
            }
        )

    def seam_summary(rows: list[dict[str, float]], name: str) -> dict:
        return numeric_summary([row[name] for row in rows])

    body_l1 = float(
        (
            motion_rep.extract_body(student_buffer.float())
            - motion_rep.extract_body(teacher_buffer.float())
        ).abs().mean()
    )
    return {
        "frames": teacher_buffer.shape[1],
        "all_finite": bool(
            torch.isfinite(teacher_buffer).all() and torch.isfinite(student_buffer).all()
        ),
        "teacher_student": {
            "normalized_body_l1": body_l1,
            "root_drift_mean_m": float(root_delta.mean()),
            "root_drift_p95_m": float(torch.quantile(root_delta, 0.95)),
            "root_drift_final_m": float(root_delta[:, -1].mean()),
            "fk_mpjpe_m": float(joint_delta.mean()),
            "rotation_geodesic_rad": float(rotation.mean()),
        },
        "waypoints": {
            "reached": len(reached_waypoints),
            "teacher_error_m": numeric_summary(teacher_waypoint_errors),
            "student_error_m": numeric_summary(student_waypoint_errors),
            "details": reached_waypoints,
        },
        "teacher": {
            "foot_slide_mps": foot_slide(teacher_kin, motion_rep.skeleton.foot_joint_idx, fps),
            "seam_root_velocity_jump_mps": seam_summary(
                teacher_seams, "root_velocity_jump_mps"
            ),
            "seam_joint_velocity_jump_mps": seam_summary(
                teacher_seams, "joint_velocity_jump_mps"
            ),
        },
        "student": {
            "foot_slide_mps": foot_slide(student_kin, motion_rep.skeleton.foot_joint_idx, fps),
            "seam_root_velocity_jump_mps": seam_summary(
                student_seams, "root_velocity_jump_mps"
            ),
            "seam_joint_velocity_jump_mps": seam_summary(
                student_seams, "joint_velocity_jump_mps"
            ),
        },
    }


def main() -> None:
    args = parse_args()
    if args.text_feature_dim < 0:
        raise ValueError("--text-feature-dim must be non-negative")
    checkpoints = sorted(set(args.checkpoints))
    if args.windows < 1 or any(value < 1 or value > args.windows for value in checkpoints):
        raise ValueError("checkpoints must be within [1, windows]")
    if args.log_every < 1:
        raise ValueError("log interval must be positive")
    if args.waypoint_interval < 1:
        raise ValueError("waypoint interval must be positive")
    if not 0 < args.waypoint_speed_min <= args.waypoint_speed_max:
        raise ValueError("invalid waypoint speed range")
    if args.student_inertialization_frames < 0:
        raise ValueError("student inertialization frames must be non-negative")
    if args.flow_root_smoothing_passes < 0:
        raise ValueError("flow root smoothing passes must be non-negative")
    if not 2 <= args.flow_root_control_points <= 40:
        raise ValueError("flow root control points must be within [2, 40]")
    if args.flow_steps < 1:
        raise ValueError("flow steps must be positive")
    if not 0.0 <= args.student_inertialization_strength <= 1.0:
        raise ValueError("student inertialization strength must be in [0, 1]")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)
    dtype = {
        "fp32": torch.float32,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }[args.model_dtype]
    teacher = load_model(
        args.model,
        device=str(device),
        text_encoder=False,
        checkpoints_dir=str(args.checkpoints_dir),
    ).eval()
    flow_config = FlowStudentConfig(
        width=args.flow_width,
        heads=args.flow_heads,
        trunk_blocks=args.flow_trunk_blocks,
        body_blocks=args.flow_body_blocks,
        text_feature_dim=args.text_feature_dim,
        heading_condition_features=args.heading_condition_features,
        root_smoothing_passes=args.flow_root_smoothing_passes,
        root_projection_kind=args.flow_root_projection_kind,
        root_control_points=args.flow_root_control_points,
    )
    codec_config = codec_config_from_args(args)
    encoder = HistoryEncoderStudent(codec_config).to(device=device, dtype=dtype).eval()
    flow = OneStepFlowStudent(flow_config).to(device=device, dtype=dtype).eval()
    decoder = MotionDecoderStudent(codec_config).to(device=device, dtype=dtype).eval()
    load_safetensor_weights(encoder, args.encoder)
    load_safetensor_weights(flow, args.flow)
    load_safetensor_weights(decoder, args.decoder)
    checkpoint_dir = args.checkpoints_dir / args.model
    quantizer = FSQRequantizer(checkpoint_dir / "stats/post_quantization").to(device)
    runtime = StudentArdyRuntime(
        encoder,
        flow,
        decoder,
        quantizer,
        teacher.motion_rep,
        model_dtype=dtype,
        flow_steps=args.flow_steps,
    ).eval()

    fps = int(teacher.motion_rep.fps)
    gen_horizon = int(teacher.gen_horizon_len)
    frames_per_token = int(teacher.num_frames_per_token)
    trigger_threshold = 4
    replan_buffer = 1
    history_crop = 4
    max_window = 10 * fps // frames_per_token * frames_per_token
    future_crop = max_window - gen_horizon
    initial_translation = torch.zeros(1, 3, device=device)
    initial_heading = torch.zeros(1, device=device)
    text_feat = torch.zeros(1, 1, 4096, device=device)
    text_mask = torch.zeros(1, 1, dtype=torch.bool, device=device)
    condition_generator = torch.Generator(device="cpu").manual_seed(args.seed + 17)

    teacher_buffer: torch.Tensor | None = None
    student_buffer: torch.Tensor | None = None
    codec_buffer: torch.Tensor | None = None
    waypoints: list[Waypoint] = []
    active_waypoint: Waypoint | None = None
    direction = 0.0
    events = []
    teacher_seams: list[dict[str, float]] = []
    student_seams: list[dict[str, float]] = []
    codec_seams: list[dict[str, float]] = []
    teacher_times: list[float] = []
    student_times: list[float] = []
    codec_times: list[float] = []
    student_stage_times: dict[str, list[float]] = {}
    codec_stage_times: dict[str, list[float]] = {}
    snapshots: dict[int, dict] = {}
    codec_snapshots: dict[int, dict] = {}
    tensor_snapshots: dict[str, torch.Tensor] = {}

    for window_index in range(1, args.windows + 1):
        if teacher_buffer is None:
            frame_index = 0
            history_end = -1
            history_start = 0
            history_length = 0
            teacher_history = None
            student_history = None
            codec_history = None
        else:
            frame_index = teacher_buffer.shape[1] - 1 - trigger_threshold
            history_end = min(teacher_buffer.shape[1] - 1, frame_index + replan_buffer)
            history_end = max(history_end, frames_per_token - 1)
            history_length = min(history_end + 1, history_crop)
            history_length = history_length // frames_per_token * frames_per_token
            history_start = max(0, history_end - history_length + 1)
            teacher_history = teacher_buffer[:, history_start : history_end + 1]
            assert student_buffer is not None
            student_history = student_buffer[:, history_start : history_end + 1]
            assert codec_buffer is not None
            codec_history = codec_buffer[:, history_start : history_end + 1]

        if active_waypoint is None or active_waypoint.frame <= frame_index:
            active_waypoint = make_waypoint(
                frame_index=frame_index,
                teacher_buffer=teacher_buffer,
                initial_translation=initial_translation,
                previous_direction=direction,
                generator=condition_generator,
                interval=args.waypoint_interval,
                fps=fps,
                speed_min=args.waypoint_speed_min,
                speed_max=args.waypoint_speed_max,
                turn_std=args.turn_std,
                motion_rep=teacher.motion_rep,
            )
            direction = active_waypoint.direction_rad
            waypoints.append(active_waypoint)

        max_constraint = active_waypoint.frame if active_waypoint.frame > history_end else None
        num_frames = compute_window_num_frames(
            history_length=history_length,
            gen_horizon_len=gen_horizon,
            num_frames_per_token=frames_per_token,
            max_window_len=max_window,
            history_start_idx=history_start,
            max_constraint_idx=max_constraint,
            future_crop_length=future_crop,
        )
        motion_mask, observed_motion = make_conditions(
            teacher,
            active_waypoint,
            history_start=history_start,
            history_end=history_end,
            num_frames=num_frames,
        )

        sync(device)
        teacher_start = time.perf_counter()
        trace = trace_autoregressive_step(
            teacher,
            num_frames=num_frames,
            num_denoising_steps=10,
            motion_mask=motion_mask,
            observed_motion=observed_motion,
            cfg_weight=(0.0, args.cfg_constraint),
            texts=None,
            text_feat=text_feat,
            text_pad_mask=text_mask,
            init_history_sequence=teacher_history,
            init_global_translation=initial_translation if teacher_history is None else None,
            init_first_heading_angle=initial_heading if teacher_history is None else None,
        )
        sync(device)
        teacher_ms = (time.perf_counter() - teacher_start) * 1000.0
        teacher_times.append(teacher_ms)

        sync(device)
        student_start = time.perf_counter()
        student_output = runtime.step(
            motion_mask=motion_mask,
            observed_motion=observed_motion,
            initial_noise=trace.initial_noise,
            init_history_sequence=student_history,
            init_global_translation=initial_translation if student_history is None else None,
            init_first_heading_angle=initial_heading if student_history is None else None,
            profile=True,
        )
        sync(device)
        student_ms = (time.perf_counter() - student_start) * 1000.0
        student_times.append(student_ms)
        for name, value in student_output.timings_ms.items():
            student_stage_times.setdefault(name, []).append(value)
        student_explicit_motion = inertialize_generated_motion(
            student_output.explicit_motion,
            student_history,
            history_length,
            frames=args.student_inertialization_frames,
            strength=args.student_inertialization_strength,
        )

        on_policy_teacher_metrics = None
        if args.on_policy_teacher_diagnostic and student_history is not None:
            # The diagnostic must not perturb the fixed teacher/student rollout.
            # fork_rng restores both CPU and selected CUDA RNG states afterward.
            cuda_device = device.index if device.index is not None else torch.cuda.current_device()
            with torch.random.fork_rng(devices=[cuda_device]):
                diagnostic_seed = args.seed + 10_000_019 * window_index
                torch.manual_seed(diagnostic_seed)
                torch.cuda.manual_seed(diagnostic_seed)
                on_policy_trace = trace_autoregressive_step(
                    teacher,
                    num_frames=num_frames,
                    num_denoising_steps=10,
                    motion_mask=motion_mask,
                    observed_motion=observed_motion,
                    cfg_weight=(0.0, args.cfg_constraint),
                    texts=None,
                    text_feat=text_feat,
                    text_pad_mask=text_mask,
                    init_history_sequence=student_history,
                    init_global_translation=None,
                    init_first_heading_angle=None,
                )
            on_policy_student = runtime.step(
                motion_mask=motion_mask,
                observed_motion=observed_motion,
                initial_noise=on_policy_trace.initial_noise,
                init_history_sequence=student_history,
                profile=False,
            )
            target_xz = active_waypoint.target_xz.to(device=device, dtype=torch.float32)
            start_xz = root_positions(student_history, teacher.motion_rep)[:, -1, [0, 2]]
            teacher_xz = root_positions(on_policy_trace.explicit_motion, teacher.motion_rep)[
                :, history_length:, [0, 2]
            ]
            predicted_xz = root_positions(on_policy_student.explicit_motion, teacher.motion_rep)[
                :, history_length:, [0, 2]
            ]
            start_error = torch.linalg.vector_norm(start_xz - target_xz, dim=-1)
            teacher_errors = torch.linalg.vector_norm(teacher_xz - target_xz, dim=-1)
            predicted_errors = torch.linalg.vector_norm(predicted_xz - target_xz, dim=-1)
            on_policy_teacher_metrics = {
                "diagnostic_seed": diagnostic_seed,
                "path_condition": path_condition_summary(on_policy_trace.path_condition),
                "start_waypoint_error_m": float(start_error.mean()),
                "teacher_min_waypoint_error_m": float(teacher_errors.min(dim=1).values.mean()),
                "teacher_final_waypoint_error_m": float(teacher_errors[:, -1].mean()),
                "student_min_waypoint_error_m": float(predicted_errors.min(dim=1).values.mean()),
                "student_final_waypoint_error_m": float(predicted_errors[:, -1].mean()),
                "clean_generation_teacher_student_l1": float(
                    (
                        on_policy_trace.clean_generation.float()
                        - on_policy_student.clean_generation.float()
                    )
                    .abs()
                    .mean()
                ),
            }

        sync(device)
        codec_start = time.perf_counter()
        codec_output = runtime.decode_clean_generation(
            clean_generation=trace.clean_generation,
            init_history_sequence=codec_history,
            init_global_translation=initial_translation if codec_history is None else None,
            init_first_heading_angle=initial_heading if codec_history is None else None,
            profile=True,
        )
        sync(device)
        codec_ms = (time.perf_counter() - codec_start) * 1000.0
        codec_times.append(codec_ms)
        for name, value in codec_output.timings_ms.items():
            codec_stage_times.setdefault(name, []).append(value)

        if teacher_buffer is None:
            teacher_buffer = trace.explicit_motion.detach()
            student_buffer = student_explicit_motion.detach()
            codec_buffer = codec_output.explicit_motion.detach()
        else:
            assert student_buffer is not None and codec_buffer is not None
            teacher_buffer = torch.cat(
                [teacher_buffer[:, : history_end + 1], trace.explicit_motion[:, history_length:]],
                dim=1,
            )
            student_buffer = torch.cat(
                [student_buffer[:, : history_end + 1], student_explicit_motion[:, history_length:]],
                dim=1,
            )
            codec_buffer = torch.cat(
                [codec_buffer[:, : history_end + 1], codec_output.explicit_motion[:, history_length:]],
                dim=1,
            )
            teacher_seams.append(seam_metrics(teacher_buffer, history_end, teacher.motion_rep, fps))
            student_seams.append(seam_metrics(student_buffer, history_end, teacher.motion_rep, fps))
            codec_seams.append(seam_metrics(codec_buffer, history_end, teacher.motion_rep, fps))

        event = {
            "window": window_index,
            "frame_index": frame_index,
            "history_start": history_start,
            "history_end": history_end,
            "history_length": history_length,
            "num_frames_visible_to_model": num_frames,
            "buffer_frames_after_update": teacher_buffer.shape[1],
            "waypoint_frame": active_waypoint.frame,
            "waypoint_target_xz": active_waypoint.target_xz.tolist(),
            "waypoint_relative_frame": active_waypoint.frame - history_start,
            "teacher_ms": teacher_ms,
            "student_ms": student_ms,
            "student_stage_ms": student_output.timings_ms,
            "oracle_flow_codec_ms": codec_ms,
            "oracle_flow_codec_stage_ms": codec_output.timings_ms,
            "teacher_history_hybrid_abs": tensor_abs_summary(trace.history_hybrid),
            "student_history_hybrid_abs": tensor_abs_summary(student_output.history_hybrid),
            "history_hybrid_teacher_student_l1": float(
                (trace.history_hybrid.float() - student_output.history_hybrid.float()).abs().mean()
            ),
            "teacher_path_condition": path_condition_summary(trace.path_condition),
            "student_path_condition": path_condition_summary(student_output.path_condition),
            "path_condition_teacher_student_l1": float(
                (
                    trace.path_condition.float()
                    - student_output.path_condition[:, : trace.path_condition.shape[1]].float()
                )
                .abs()
                .mean()
            ),
            "student_clean_generation_abs": tensor_abs_summary(student_output.clean_generation),
            "on_policy_teacher": on_policy_teacher_metrics,
        }
        events.append(event)
        if window_index == 1 or window_index == args.windows or window_index % args.log_every == 0:
            print(json.dumps(event, ensure_ascii=False), flush=True)

        if window_index in checkpoints:
            assert codec_buffer is not None
            metrics = snapshot_metrics(
                teacher_buffer,
                student_buffer,
                waypoints,
                teacher_seams,
                student_seams,
                teacher.motion_rep,
                fps,
            )
            snapshots[window_index] = metrics
            codec_metrics = snapshot_metrics(
                teacher_buffer,
                codec_buffer,
                waypoints,
                teacher_seams,
                codec_seams,
                teacher.motion_rep,
                fps,
            )
            codec_snapshots[window_index] = codec_metrics
            prefix = f"window_{window_index:03d}"
            tensor_snapshots[f"teacher_{prefix}"] = teacher_buffer.detach().cpu().contiguous()
            tensor_snapshots[f"student_{prefix}"] = student_buffer.detach().cpu().contiguous()
            tensor_snapshots[f"oracle_flow_codec_{prefix}"] = codec_buffer.detach().cpu().contiguous()
            reached = [waypoint for waypoint in waypoints if waypoint.frame < teacher_buffer.shape[1]]
            tensor_snapshots[f"waypoint_frames_{prefix}"] = torch.tensor(
                [waypoint.frame for waypoint in reached], dtype=torch.int64
            )
            tensor_snapshots[f"waypoint_xz_{prefix}"] = (
                torch.stack([waypoint.target_xz for waypoint in reached])
                if reached
                else torch.empty(0, 2, dtype=torch.float32)
            )

    result = {
        "schema": "ardy_faithful_rollout_eval_v1",
        "seed": args.seed,
        "model_dtype": args.model_dtype,
        "flow_steps": args.flow_steps,
        "text_feature_dim": args.text_feature_dim,
        "heading_condition_features": args.heading_condition_features,
        "windows": args.windows,
        "timing_note": (
            "teacher includes exact tracing of ten DDIM states; student wall includes faithful "
            "Python preprocessing and synchronized module execution"
        ),
        "ui_semantics": {
            "fps": fps,
            "generation_horizon": gen_horizon,
            "trigger_threshold": trigger_threshold,
            "replan_buffer": replan_buffer,
            "history_crop": history_crop,
            "waypoint_interval": args.waypoint_interval,
            "buffer_update": "old[:history_end+1] + generated[history_length:]",
            "student_inertialization_frames": args.student_inertialization_frames,
            "student_inertialization_strength": args.student_inertialization_strength,
        },
        "weights": {
            "encoder": str(args.encoder),
            "flow": str(args.flow),
            "decoder": str(args.decoder),
        },
        "timings_ms": {
            "teacher_window_wall": numeric_summary(teacher_times),
            "student_window_wall": numeric_summary(student_times),
            "oracle_flow_codec_window_wall": numeric_summary(codec_times),
            "student_stages": {
                name: numeric_summary(values) for name, values in sorted(student_stage_times.items())
            },
            "oracle_flow_codec_stages": {
                name: numeric_summary(values) for name, values in sorted(codec_stage_times.items())
            },
        },
        "snapshots": {str(index): snapshots[index] for index in checkpoints},
        "oracle_flow_codec_snapshots": {
            str(index): codec_snapshots[index] for index in checkpoints
        },
        "waypoints": [
            {
                "frame": waypoint.frame,
                "target_xz": waypoint.target_xz.tolist(),
                "speed_mps": waypoint.speed_mps,
                "direction_rad": waypoint.direction_rad,
            }
            for waypoint in waypoints
        ],
        "events": events,
    }
    if not all(snapshot["all_finite"] for snapshot in snapshots.values()) or not all(
        snapshot["all_finite"] for snapshot in codec_snapshots.values()
    ):
        raise RuntimeError("rollout produced NaN/Inf")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "metrics.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    save_file(
        tensor_snapshots,
        args.output_dir / "fixed_cases.safetensors",
        metadata={
            "schema": "ardy_faithful_rollout_fixed_cases_v1",
            "seed": str(args.seed),
            "model_dtype": args.model_dtype,
            "flow_steps": str(args.flow_steps),
        },
    )
    print(json.dumps({"event": "rollout_complete", "output": str(args.output_dir)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
