#!/usr/bin/env python
"""Generate corrective teacher shards on the deployed student's rollout states.

The reference branch follows the exact infinite-demo buffer/replan semantics and
defines reachable world-space waypoints.  A second released-teacher query starts
from the student's current four-frame history and targets the same waypoint.
Those corrective traces are written as ordinary ``TeacherWindowTrace`` shards,
so the existing flow trainer can fit them without changing tensor semantics.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch

from ardy.constraints import Root2DConstraintSet
from ardy.model import load_model
from ardy_distill.codec_cli import add_codec_config_arguments, codec_config_from_args
from ardy_distill.data import TeacherShardWriter
from ardy_distill.losses import FSQRequantizer
from ardy_distill.models import (
    FlowStudentConfig,
    HistoryEncoderStudent,
    MotionDecoderStudent,
    OneStepFlowStudent,
)
from ardy_distill.runtime import load_safetensor_weights
from ardy_distill.student_runtime import StudentArdyRuntime
from ardy_distill.teacher import TeacherWindowTrace, trace_autoregressive_step
from scripts.interactive_demo.window_budget import compute_window_num_frames


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--encoder", type=Path, required=True)
    parser.add_argument("--flow", type=Path, required=True)
    parser.add_argument("--decoder", type=Path, required=True)
    parser.add_argument(
        "--checkpoints-dir",
        type=Path,
        default=PROJECT_ROOT / "ardy" / "checkpoints",
    )
    parser.add_argument("--model", default="ARDY-Core-RP-20FPS-Horizon40")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--windows", type=int, default=1024, help="Windows per rank")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--rollout-steps", type=int, default=16)
    parser.add_argument("--shard-size", type=int, default=256)
    parser.add_argument("--storage-dtype", choices=["float16", "float32"], default="float16")
    parser.add_argument("--student-dtype", choices=["fp32", "fp16", "bf16"], default="fp32")
    parser.add_argument("--seed", type=int, default=20260714)
    parser.add_argument("--cfg-constraint", type=float, default=1.5)
    parser.add_argument("--waypoint-interval", type=int, default=60)
    parser.add_argument("--waypoint-speed-min", type=float, default=0.65)
    parser.add_argument("--waypoint-speed-max", type=float, default=1.15)
    parser.add_argument("--turn-std", type=float, default=0.65)
    parser.add_argument("--flow-width", type=int, default=512)
    parser.add_argument("--flow-heads", type=int, default=8)
    parser.add_argument("--flow-trunk-blocks", type=int, default=8)
    parser.add_argument("--flow-body-blocks", type=int, default=8)
    parser.add_argument("--flow-root-smoothing-passes", type=int, default=0)
    parser.add_argument(
        "--flow-root-projection-kind",
        choices=["binomial", "cubic_controls"],
        default="binomial",
    )
    parser.add_argument("--flow-root-control-points", type=int, default=10)
    add_codec_config_arguments(parser)
    parser.add_argument("--log-every", type=int, default=128)
    return parser.parse_args()


def distributed_context() -> tuple[int, int, int]:
    return (
        int(os.environ.get("RANK", "0")),
        int(os.environ.get("LOCAL_RANK", "0")),
        int(os.environ.get("WORLD_SIZE", "1")),
    )


def root_xz_at(buffer: torch.Tensor | None, initial_translation: torch.Tensor, frame: int, motion_rep) -> torch.Tensor:
    if buffer is None:
        return initial_translation[:, [0, 2]].detach().cpu()
    unnormalized = motion_rep.unnormalize(buffer.float())
    positions = motion_rep.get_root_pos(unnormalized)
    index = min(max(0, frame), positions.shape[1] - 1)
    return positions[:, index, [0, 2]].detach().cpu()


def sample_waypoints(
    current_xz: torch.Tensor,
    previous_direction: torch.Tensor,
    generator: torch.Generator,
    *,
    interval: int,
    fps: int,
    speed_min: float,
    speed_max: float,
    turn_std: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch = current_xz.shape[0]
    direction = previous_direction + torch.randn(batch, generator=generator) * turn_std
    speed = speed_min + torch.rand(batch, generator=generator) * (speed_max - speed_min)
    distance = speed * interval / fps
    delta = torch.stack([torch.sin(direction), torch.cos(direction)], dim=-1) * distance[:, None]
    return current_xz + delta, direction


def make_conditions(
    teacher,
    targets_xz: torch.Tensor,
    *,
    waypoint_frame: int,
    history_start: int,
    history_end: int,
    num_frames: int,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    if waypoint_frame <= history_end:
        return None, None
    relative_frame = waypoint_frame - history_start
    if relative_frame < 0 or relative_frame >= num_frames:
        return None, None
    constraints = []
    for target in targets_xz:
        constraints.append(
            [
                Root2DConstraintSet(
                    teacher.skeleton,
                    frame_indices=torch.tensor([relative_frame], dtype=torch.long),
                    root_2d=target.reshape(1, 2),
                )
            ]
        )
    lengths = torch.full(
        (targets_xz.shape[0],),
        num_frames,
        device=teacher.device,
        dtype=torch.long,
    )
    observed_motion, motion_mask = teacher.motion_rep.create_conditions_from_constraints_batched(
        constraints,
        lengths,
        to_normalize=True,
        device=teacher.device,
    )
    return motion_mask, observed_motion


def update_buffer(
    buffer: torch.Tensor | None,
    generated: torch.Tensor,
    *,
    history_end: int,
    history_length: int,
) -> torch.Tensor:
    if buffer is None:
        return generated.detach()
    return torch.cat(
        [buffer[:, : history_end + 1], generated[:, history_length:]],
        dim=1,
    ).detach()


def trace_teacher(
    teacher,
    *,
    num_frames: int,
    motion_mask: torch.Tensor | None,
    observed_motion: torch.Tensor | None,
    cfg_constraint: float,
    text_feat: torch.Tensor,
    text_mask: torch.Tensor,
    history: torch.Tensor | None,
    initial_translation: torch.Tensor,
    initial_heading: torch.Tensor,
) -> TeacherWindowTrace:
    return trace_autoregressive_step(
        teacher,
        num_frames=num_frames,
        num_denoising_steps=10,
        motion_mask=motion_mask,
        observed_motion=observed_motion,
        cfg_weight=(0.0, cfg_constraint),
        texts=None,
        text_feat=text_feat,
        text_pad_mask=text_mask,
        init_history_sequence=history,
        init_global_translation=initial_translation if history is None else None,
        init_first_heading_angle=initial_heading if history is None else None,
    )


def pad_trace_path_condition(trace: TeacherWindowTrace, frames: int = 64) -> TeacherWindowTrace:
    """Keep the shard schema fixed when the faithful UI selects a 44-frame window."""

    current = trace.path_condition.shape[1]
    if current > frames:
        trace.path_condition = trace.path_condition[:, :frames]
    elif current < frames:
        padding = trace.path_condition.new_zeros(
            trace.path_condition.shape[0],
            frames - current,
            trace.path_condition.shape[2],
        )
        trace.path_condition = torch.cat([trace.path_condition, padding], dim=1)
    return trace


def distribution_summary(values: list[float]) -> dict[str, float | int | None]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(array.size),
        "mean": float(array.mean()) if array.size else None,
        "p50": float(np.percentile(array, 50)) if array.size else None,
        "p95": float(np.percentile(array, 95)) if array.size else None,
        "p99": float(np.percentile(array, 99)) if array.size else None,
        "max": float(array.max()) if array.size else None,
    }


def main() -> None:
    args = parse_args()
    rank, local_rank, world_size = distributed_context()
    if args.windows < args.batch_size or args.windows % args.batch_size:
        raise ValueError("--windows must be a positive multiple of --batch-size")
    if args.rollout_steps < 2:
        raise ValueError("--rollout-steps must be at least two")
    if args.waypoint_interval < 1 or args.log_every < 1:
        raise ValueError("waypoint interval and log interval must be positive")
    if not 0 < args.waypoint_speed_min <= args.waypoint_speed_max:
        raise ValueError("invalid waypoint speed range")
    if args.flow_root_smoothing_passes < 0:
        raise ValueError("flow root smoothing passes must be non-negative")
    if not 2 <= args.flow_root_control_points <= 40:
        raise ValueError("flow root control points must be within [2, 40]")

    device = torch.device(f"cuda:{local_rank}" if args.device == "auto" else args.device)
    rank_seed = args.seed + rank * 1_000_003
    random.seed(rank_seed)
    np.random.seed(rank_seed)
    torch.manual_seed(rank_seed)
    torch.cuda.manual_seed_all(rank_seed)
    condition_generator = torch.Generator(device="cpu").manual_seed(rank_seed + 17)

    rank_output = args.output_dir / f"rank-{rank:02d}"
    rank_output.mkdir(parents=True, exist_ok=True)
    if (rank_output / "manifest.json").exists():
        raise FileExistsError(f"refusing to overwrite existing shards: {rank_output}")
    (rank_output / "generation_config.json").write_text(
        json.dumps(
            {
                **{
                    name: str(value) if isinstance(value, Path) else value
                    for name, value in vars(args).items()
                },
                "rank": rank,
                "local_rank": local_rank,
                "world_size": world_size,
                "rank_seed": rank_seed,
                "resolved_device": str(device),
                "runtime_semantics": {
                    "trigger_threshold": 4,
                    "replan_buffer": 1,
                    "history_crop": 4,
                    "buffer_update": "old[:history_end+1] + generated[history_length:]",
                },
                "target_semantics": "released teacher conditioned on student rollout history",
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    writer = TeacherShardWriter(
        rank_output,
        shard_size=args.shard_size,
        storage_dtype=torch.float16 if args.storage_dtype == "float16" else torch.float32,
        prefix=f"onpolicy-r{rank:02d}",
    )

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
        root_smoothing_passes=args.flow_root_smoothing_passes,
        root_projection_kind=args.flow_root_projection_kind,
        root_control_points=args.flow_root_control_points,
    )
    codec_config = codec_config_from_args(args)
    student_dtype = {
        "fp32": torch.float32,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }[args.student_dtype]
    encoder = HistoryEncoderStudent(codec_config).to(device=device, dtype=student_dtype).eval()
    flow = OneStepFlowStudent(flow_config).to(device=device, dtype=student_dtype).eval()
    decoder = MotionDecoderStudent(codec_config).to(device=device, dtype=student_dtype).eval()
    load_safetensor_weights(encoder, args.encoder)
    load_safetensor_weights(flow, args.flow)
    load_safetensor_weights(decoder, args.decoder)
    quantizer = FSQRequantizer(
        args.checkpoints_dir / args.model / "stats" / "post_quantization"
    ).to(device)
    runtime = StudentArdyRuntime(
        encoder,
        flow,
        decoder,
        quantizer,
        teacher.motion_rep,
        model_dtype=student_dtype,
    ).eval()

    batch = args.batch_size
    text_feat = torch.zeros(batch, 1, 4096, device=device, dtype=torch.float32)
    text_mask = torch.zeros(batch, 1, device=device, dtype=torch.bool)
    fps = int(teacher.motion_rep.fps)
    gen_horizon = int(teacher.gen_horizon_len)
    frames_per_token = int(teacher.num_frames_per_token)
    trigger_threshold = 4
    replan_buffer = 1
    history_crop = 4
    max_window = 10 * fps // frames_per_token * frames_per_token
    future_crop = max_window - gen_horizon
    path_norms: list[float] = []
    student_reference_drifts: list[float] = []
    path_norms_by_depth: dict[int, list[float]] = {
        depth: [] for depth in range(args.rollout_steps)
    }
    drifts_by_depth: dict[int, list[float]] = {
        depth: [] for depth in range(args.rollout_steps)
    }
    history_windows = 0
    constrained_windows = 0
    start_time = time.perf_counter()

    while writer.written + writer.buffered < args.windows:
        initial_translation = torch.zeros(batch, 3, device=device)
        initial_translation[:, 0] = torch.rand(batch, device=device) * 10 - 5
        initial_translation[:, 2] = torch.rand(batch, device=device) * 10 - 5
        initial_heading = torch.rand(batch, device=device) * (2 * torch.pi) - torch.pi
        reference_buffer: torch.Tensor | None = None
        student_buffer: torch.Tensor | None = None
        waypoint_frame = -1
        waypoint_targets = torch.zeros(batch, 2)
        directions = torch.zeros(batch)

        for rollout_depth in range(args.rollout_steps):
            if writer.written + writer.buffered >= args.windows:
                break
            if reference_buffer is None:
                frame_index = 0
                history_end = -1
                history_start = 0
                history_length = 0
                reference_history = None
                student_history = None
            else:
                frame_index = reference_buffer.shape[1] - 1 - trigger_threshold
                history_end = min(reference_buffer.shape[1] - 1, frame_index + replan_buffer)
                history_end = max(history_end, frames_per_token - 1)
                history_length = min(history_end + 1, history_crop)
                history_length = history_length // frames_per_token * frames_per_token
                history_start = max(0, history_end - history_length + 1)
                reference_history = reference_buffer[:, history_start : history_end + 1]
                assert student_buffer is not None
                student_history = student_buffer[:, history_start : history_end + 1]

            reference_current_xz = root_xz_at(
                reference_buffer,
                initial_translation,
                frame_index,
                teacher.motion_rep,
            )
            student_current_xz = root_xz_at(
                student_buffer,
                initial_translation,
                frame_index,
                teacher.motion_rep,
            )
            drift_values = torch.linalg.vector_norm(
                student_current_xz - reference_current_xz, dim=-1
            ).tolist()
            student_reference_drifts.extend(drift_values)
            drifts_by_depth[rollout_depth].extend(drift_values)
            if student_history is not None:
                history_windows += batch

            if waypoint_frame <= frame_index:
                waypoint_targets, directions = sample_waypoints(
                    reference_current_xz,
                    directions,
                    condition_generator,
                    interval=args.waypoint_interval,
                    fps=fps,
                    speed_min=args.waypoint_speed_min,
                    speed_max=args.waypoint_speed_max,
                    turn_std=args.turn_std,
                )
                waypoint_frame = max(0, frame_index) + args.waypoint_interval

            num_frames = compute_window_num_frames(
                history_length=history_length,
                gen_horizon_len=gen_horizon,
                num_frames_per_token=frames_per_token,
                max_window_len=max_window,
                history_start_idx=history_start,
                max_constraint_idx=waypoint_frame,
                future_crop_length=future_crop,
            )
            motion_mask, observed_motion = make_conditions(
                teacher,
                waypoint_targets,
                waypoint_frame=waypoint_frame,
                history_start=history_start,
                history_end=history_end,
                num_frames=num_frames,
            )
            reference_trace = trace_teacher(
                teacher,
                num_frames=num_frames,
                motion_mask=motion_mask,
                observed_motion=observed_motion,
                cfg_constraint=args.cfg_constraint,
                text_feat=text_feat,
                text_mask=text_mask,
                history=reference_history,
                initial_translation=initial_translation,
                initial_heading=initial_heading,
            )
            corrective_trace = (
                reference_trace
                if student_history is None
                else trace_teacher(
                    teacher,
                    num_frames=num_frames,
                    motion_mask=motion_mask,
                    observed_motion=observed_motion,
                    cfg_constraint=args.cfg_constraint,
                    text_feat=text_feat,
                    text_mask=text_mask,
                    history=student_history,
                    initial_translation=initial_translation,
                    initial_heading=initial_heading,
                )
            )
            corrective_trace = pad_trace_path_condition(corrective_trace)
            writer.add(corrective_trace)
            valid = corrective_trace.path_condition[..., 2] > 0.5
            constrained_windows += int(valid.any(dim=1).sum())
            if bool(valid.any()):
                norms = torch.linalg.vector_norm(
                    corrective_trace.path_condition[..., :2].float(), dim=-1
                )[valid]
                norm_values = norms.detach().cpu().tolist()
                path_norms.extend(norm_values)
                path_norms_by_depth[rollout_depth].extend(norm_values)

            student_output = runtime.step(
                motion_mask=motion_mask,
                observed_motion=observed_motion,
                initial_noise=corrective_trace.initial_noise,
                init_history_sequence=student_history,
                init_global_translation=initial_translation if student_history is None else None,
                init_first_heading_angle=initial_heading if student_history is None else None,
                profile=False,
            )
            reference_buffer = update_buffer(
                reference_buffer,
                reference_trace.explicit_motion,
                history_end=history_end,
                history_length=history_length,
            )
            student_buffer = update_buffer(
                student_buffer,
                student_output.explicit_motion,
                history_end=history_end,
                history_length=history_length,
            )

            current = writer.written + writer.buffered
            if current % args.log_every == 0 or current >= args.windows:
                elapsed = time.perf_counter() - start_time
                values = np.asarray(path_norms, dtype=np.float64)
                print(
                    json.dumps(
                        {
                            "rank": rank,
                            "world_size": world_size,
                            "windows": current,
                            "target": args.windows,
                            "elapsed_s": round(elapsed, 3),
                            "windows_per_s": round(current / elapsed, 4),
                            "path_norm_p50": float(np.percentile(values, 50)) if values.size else None,
                            "path_norm_p95": float(np.percentile(values, 95)) if values.size else None,
                            "path_norm_max": float(values.max()) if values.size else None,
                            "student_reference_drift_p95_m": (
                                float(np.percentile(student_reference_drifts, 95))
                                if student_reference_drifts
                                else None
                            ),
                        }
                    ),
                    flush=True,
                )

    writer.close()
    elapsed = time.perf_counter() - start_time
    summary = {
        "schema": "ardy_onpolicy_teacher_generation_v2",
        "rank": rank,
        "world_size": world_size,
        "windows": writer.written,
        "elapsed_s": elapsed,
        "windows_per_s": writer.written / elapsed,
        "history_fraction": history_windows / writer.written,
        "constraint_fraction": constrained_windows / writer.written,
        "path_norm": distribution_summary(path_norms),
        "student_reference_root_drift_m": distribution_summary(
            student_reference_drifts
        ),
        "by_rollout_depth": {
            str(depth): {
                "windows": len(drifts_by_depth[depth]),
                "has_history": depth > 0,
                "path_norm": distribution_summary(path_norms_by_depth[depth]),
                "student_reference_root_drift_m": distribution_summary(
                    drifts_by_depth[depth]
                ),
            }
            for depth in range(args.rollout_steps)
        },
    }
    (rank_output / "onpolicy_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"event": "onpolicy_teacher_generation_complete", **summary}), flush=True)


if __name__ == "__main__":
    main()
