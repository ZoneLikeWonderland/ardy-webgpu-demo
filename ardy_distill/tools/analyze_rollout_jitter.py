#!/usr/bin/env python
"""Quantify temporal noise in faithful ARDY rollout snapshots.

The evaluator stores the complete visible buffer after selected generation
windows.  This tool compares four matched sequences:

* released teacher;
* teacher clean latents passed through the compact codec (codec oracle);
* the raw one-step flow student;
* the same student with deployment-side feature inertialization.

Metrics are reported both over the complete buffer and inside individual
generation segments.  Segment-local derivatives never cross a replan seam, so
they distinguish persistent within-window jitter from a boundary-only problem.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from safetensors.torch import load_file

from ardy_distill.runtime import load_motion_rep


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-cases", type=Path, required=True)
    parser.add_argument("--inertial-cases", type=Path, required=True)
    parser.add_argument(
        "--rollout-metrics",
        type=Path,
        required=True,
        help="metrics.json beside the inertialized fixed cases",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path("ardy/checkpoints/ARDY-Core-RP-20FPS-Horizon40"),
    )
    parser.add_argument("--snapshots", type=int, nargs="+", default=[1, 5, 20, 50])
    parser.add_argument("--high-frequency-hz", type=float, default=5.0)
    parser.add_argument("--spectral-floor-hz", type=float, default=0.5)
    parser.add_argument("--interior-margin-frames", type=int, default=4)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def summary(values: torch.Tensor | np.ndarray | Iterable[float]) -> dict[str, float | int | None]:
    values = torch.as_tensor(values, dtype=torch.float64).reshape(-1)
    values = values[torch.isfinite(values)]
    if not values.numel():
        return {"count": 0, "mean": None, "p50": None, "p95": None, "max": None}
    return {
        "count": int(values.numel()),
        "mean": float(values.mean()),
        "p50": float(torch.quantile(values, 0.50)),
        "p95": float(torch.quantile(values, 0.95)),
        "max": float(values.max()),
    }


def derivative_norm(series: torch.Tensor, order: int, fps: float) -> torch.Tensor:
    values = series.double()
    for _ in range(order):
        values = torch.diff(values, dim=0) * fps
    if not values.numel():
        return values.new_empty(0)
    return torch.linalg.vector_norm(values, dim=-1).reshape(-1)


def segment_slices(length: int, seams: list[int]) -> list[slice]:
    """Return spans whose internal differences do not cross a rollout seam.

    A seam value is the retained history frame; the discontinuous transition,
    if any, is from ``seam`` to ``seam + 1``.
    """

    starts = [0] + [seam + 1 for seam in seams if 0 <= seam < length - 1]
    stops = [seam + 1 for seam in seams if 0 <= seam < length - 1] + [length]
    return [slice(start, stop) for start, stop in zip(starts, stops) if stop > start]


def concatenate_derivatives(
    series: torch.Tensor,
    segments: list[slice],
    order: int,
    fps: float,
    margin: int = 0,
) -> torch.Tensor:
    rows = []
    for span in segments:
        start = span.start + margin
        stop = span.stop - margin
        if stop - start <= order:
            continue
        rows.append(derivative_norm(series[start:stop], order, fps))
    return torch.cat(rows) if rows else torch.empty(0, dtype=torch.float64)


def spectral_ratio(
    series: torch.Tensor,
    segments: list[slice],
    fps: float,
    floor_hz: float,
    high_hz: float,
    margin: int,
) -> dict[str, float | int | None]:
    """Velocity power above ``high_hz`` divided by power above ``floor_hz``."""

    high_power = 0.0
    reference_power = 0.0
    per_segment = []
    for span in segments:
        start = span.start + margin
        stop = span.stop - margin
        values = series[start:stop].double()
        if values.shape[0] < 12:
            continue
        velocity = torch.diff(values, dim=0) * fps
        velocity = velocity - velocity.mean(dim=0, keepdim=True)
        spectrum = torch.fft.rfft(velocity, dim=0)
        power = spectrum.real.square() + spectrum.imag.square()
        power = power.reshape(power.shape[0], -1).sum(dim=1)
        frequencies = torch.fft.rfftfreq(velocity.shape[0], d=1.0 / fps)
        reference = float(power[frequencies >= floor_hz].sum())
        high = float(power[frequencies >= high_hz].sum())
        if reference <= 1.0e-20:
            continue
        high_power += high
        reference_power += reference
        per_segment.append(high / reference)
    return {
        "definition": f"velocity power >= {high_hz:g} Hz / power >= {floor_hz:g} Hz",
        "aggregate_ratio": high_power / reference_power if reference_power > 0 else None,
        "per_segment": summary(per_segment),
    }


def direction_change_degrees(
    root_xz: torch.Tensor,
    segments: list[slice],
    fps: float,
    margin: int,
    minimum_speed: float = 0.10,
) -> dict[str, float | int | None]:
    values = []
    for span in segments:
        start = span.start + margin
        stop = span.stop - margin
        positions = root_xz[start:stop].double()
        if positions.shape[0] < 3:
            continue
        velocity = torch.diff(positions, dim=0) * fps
        speed = torch.linalg.vector_norm(velocity, dim=-1)
        valid = (speed[:-1] >= minimum_speed) & (speed[1:] >= minimum_speed)
        if not bool(valid.any()):
            continue
        cosine = (
            (velocity[:-1] * velocity[1:]).sum(dim=-1)
            / (speed[:-1] * speed[1:]).clamp_min(1.0e-12)
        ).clamp(-1.0, 1.0)
        values.append(torch.rad2deg(torch.acos(cosine[valid])))
    return summary(torch.cat(values) if values else torch.empty(0))


def seam_jumps(
    series: torch.Tensor,
    seams: list[int],
    fps: float,
) -> dict[str, float | int | None]:
    jumps = []
    for seam in seams:
        if seam < 1 or seam + 1 >= series.shape[0]:
            continue
        before = (series[seam] - series[seam - 1]) * fps
        after = (series[seam + 1] - series[seam]) * fps
        jumps.append(torch.linalg.vector_norm(after - before, dim=-1).reshape(-1))
    return summary(torch.cat(jumps) if jumps else torch.empty(0))


def temporal_metrics(
    series: torch.Tensor,
    seams: list[int],
    fps: float,
    floor_hz: float,
    high_hz: float,
    margin: int,
) -> dict:
    segments = segment_slices(series.shape[0], seams)
    return {
        "shape": list(series.shape),
        "complete_buffer": {
            "speed": summary(derivative_norm(series, 1, fps)),
            "acceleration": summary(derivative_norm(series, 2, fps)),
            "jerk": summary(derivative_norm(series, 3, fps)),
        },
        "segment_local_no_cross_seam": {
            "speed": summary(concatenate_derivatives(series, segments, 1, fps)),
            "acceleration": summary(concatenate_derivatives(series, segments, 2, fps)),
            "jerk": summary(concatenate_derivatives(series, segments, 3, fps)),
        },
        f"segment_interior_margin_{margin}_frames": {
            "speed": summary(concatenate_derivatives(series, segments, 1, fps, margin)),
            "acceleration": summary(concatenate_derivatives(series, segments, 2, fps, margin)),
            "jerk": summary(concatenate_derivatives(series, segments, 3, fps, margin)),
        },
        "segment_local_high_frequency": spectral_ratio(
            series,
            segments,
            fps,
            floor_hz,
            high_hz,
            margin,
        ),
        "seam_velocity_jump": seam_jumps(series, seams, fps),
    }


def angular_metrics(
    heading_xy: torch.Tensor,
    seams: list[int],
    fps: float,
    floor_hz: float,
    high_hz: float,
    margin: int,
) -> dict:
    heading = heading_xy.double()
    angle = torch.atan2(heading[:, 1], heading[:, 0]).numpy()
    unwrapped = torch.from_numpy(np.unwrap(angle)).unsqueeze(-1)
    return temporal_metrics(unwrapped, seams, fps, floor_hz, high_hz, margin)


def motion_metrics(
    normalized: torch.Tensor,
    seams: list[int],
    motion_rep,
    floor_hz: float,
    high_hz: float,
    margin: int,
) -> dict:
    if normalized.shape[0] != 1:
        raise ValueError(f"expected batch size one, got {tuple(normalized.shape)}")
    unnormalized = motion_rep.unnormalize(normalized.float())
    inverse = motion_rep.inverse(unnormalized, is_normalized=False)
    root = motion_rep.get_root_pos(unnormalized)[0].cpu()
    joints = inverse["posed_joints"][0].cpu()
    heading = inverse["global_root_heading"][0].cpu()
    root_relative_joints = joints - root.unsqueeze(1)
    fps = float(motion_rep.fps)
    root_xz = root[:, [0, 2]]
    segments = segment_slices(root.shape[0], seams)
    return {
        "frames": int(root.shape[0]),
        "segments": [[span.start, span.stop] for span in segments],
        "root_horizontal_m": temporal_metrics(
            root_xz, seams, fps, floor_hz, high_hz, margin
        ),
        "root_vertical_m": temporal_metrics(
            root[:, 1:2], seams, fps, floor_hz, high_hz, margin
        ),
        "root_heading_rad": angular_metrics(
            heading, seams, fps, floor_hz, high_hz, margin
        ),
        "world_joints_m": temporal_metrics(
            joints, seams, fps, floor_hz, high_hz, margin
        ),
        "root_relative_joints_m": temporal_metrics(
            root_relative_joints, seams, fps, floor_hz, high_hz, margin
        ),
        "root_direction_change_degrees": direction_change_degrees(
            root_xz, segments, fps, margin
        ),
    }


def ratio(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator is None or abs(denominator) < 1.0e-20:
        return None
    return numerator / denominator


def diagnostic_ratios(metrics: dict[str, dict]) -> dict:
    teacher = metrics["teacher"]
    result = {}
    paths = {
        "root_horizontal_interior_acceleration_p95": (
            "root_horizontal_m",
            "segment_interior_margin_4_frames",
            "acceleration",
            "p95",
        ),
        "root_horizontal_interior_jerk_p95": (
            "root_horizontal_m",
            "segment_interior_margin_4_frames",
            "jerk",
            "p95",
        ),
        "root_horizontal_high_frequency_ratio": (
            "root_horizontal_m",
            "segment_local_high_frequency",
            "aggregate_ratio",
        ),
        "root_relative_joint_interior_acceleration_p95": (
            "root_relative_joints_m",
            "segment_interior_margin_4_frames",
            "acceleration",
            "p95",
        ),
        "root_relative_joint_interior_jerk_p95": (
            "root_relative_joints_m",
            "segment_interior_margin_4_frames",
            "jerk",
            "p95",
        ),
        "root_relative_joint_high_frequency_ratio": (
            "root_relative_joints_m",
            "segment_local_high_frequency",
            "aggregate_ratio",
        ),
        "root_seam_velocity_jump_mean": (
            "root_horizontal_m",
            "seam_velocity_jump",
            "mean",
        ),
        "joint_seam_velocity_jump_mean": (
            "world_joints_m",
            "seam_velocity_jump",
            "mean",
        ),
    }

    def lookup(source: dict, path: tuple[str, ...]):
        value = source
        for key in path:
            if key.startswith("segment_interior_margin_"):
                key = next(name for name in value if name.startswith("segment_interior_margin_"))
            value = value[key]
        return value

    for source_name, source in metrics.items():
        if source_name == "teacher":
            continue
        result[source_name] = {
            name: {
                "value": lookup(source, path),
                "teacher_value": lookup(teacher, path),
                "ratio_to_teacher": ratio(lookup(source, path), lookup(teacher, path)),
            }
            for name, path in paths.items()
        }
    return result


def main() -> None:
    args = parse_args()
    if args.high_frequency_hz <= args.spectral_floor_hz:
        raise ValueError("high-frequency threshold must exceed spectral floor")
    if args.interior_margin_frames < 0:
        raise ValueError("interior margin must be non-negative")

    raw = load_file(args.raw_cases, device="cpu")
    inertial = load_file(args.inertial_cases, device="cpu")
    rollout = json.loads(args.rollout_metrics.read_text(encoding="utf-8"))
    motion_rep = load_motion_rep(args.checkpoint_dir)

    result = {
        "schema": "ardy_rollout_jitter_analysis_v1",
        "fps": float(motion_rep.fps),
        "definitions": {
            "derivatives": "finite differences scaled to seconds; norms use the final xyz axis",
            "segment_local": "derivatives/FFT computed separately per generation span and never across a replan seam",
            "segment_interior": (
                f"also removes {args.interior_margin_frames} frames from both ends of every span"
            ),
            "high_frequency": (
                f"velocity power >= {args.high_frequency_hz:g} Hz divided by velocity power "
                f">= {args.spectral_floor_hz:g} Hz"
            ),
        },
        "inputs": {
            "raw_cases": str(args.raw_cases),
            "inertial_cases": str(args.inertial_cases),
            "rollout_metrics": str(args.rollout_metrics),
            "checkpoint_dir": str(args.checkpoint_dir),
        },
        "snapshot_consistency": {},
        "snapshots": {},
    }

    for window in sorted(set(args.snapshots)):
        suffix = f"window_{window:03d}"
        teacher_key = f"teacher_{suffix}"
        oracle_key = f"oracle_flow_codec_{suffix}"
        student_key = f"student_{suffix}"
        for key in (teacher_key, oracle_key, student_key):
            if key not in raw or key not in inertial:
                raise KeyError(f"missing snapshot key: {key}")
        teacher_difference = float((raw[teacher_key] - inertial[teacher_key]).abs().max())
        oracle_difference = float((raw[oracle_key] - inertial[oracle_key]).abs().max())
        result["snapshot_consistency"][str(window)] = {
            "raw_vs_inertial_teacher_max_abs": teacher_difference,
            "raw_vs_inertial_oracle_max_abs": oracle_difference,
        }
        if teacher_difference > 1.0e-6 or oracle_difference > 1.0e-6:
            raise RuntimeError("raw and inertial evaluations are not matched")

        length = raw[teacher_key].shape[1]
        seams = [
            int(event["history_end"])
            for event in rollout["events"]
            if 1 < int(event["window"]) <= window
            and 0 <= int(event["history_end"]) < length - 1
        ]
        sources = {
            "teacher": raw[teacher_key],
            "codec_oracle": raw[oracle_key],
            "raw_student": raw[student_key],
            "inertialized_student": inertial[student_key],
        }
        source_metrics = {
            name: motion_metrics(
                tensor,
                seams,
                motion_rep,
                args.spectral_floor_hz,
                args.high_frequency_hz,
                args.interior_margin_frames,
            )
            for name, tensor in sources.items()
        }
        result["snapshots"][str(window)] = {
            "frames": length,
            "seam_after_frames": seams,
            "sources": source_metrics,
            "ratios_to_teacher": diagnostic_ratios(source_metrics),
        }
        print(
            json.dumps(
                {
                    "window": window,
                    "frames": length,
                    "ratios_to_teacher": result["snapshots"][str(window)][
                        "ratios_to_teacher"
                    ],
                },
                ensure_ascii=False,
            ),
            flush=True,
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"event": "complete", "output": str(args.output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
