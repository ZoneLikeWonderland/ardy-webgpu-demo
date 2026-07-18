#!/usr/bin/env python
"""Build a reproducible leaderboard for codec checkpoint sweeps.

The evaluator deliberately stores raw evidence in separate files: static
validation, multi-seed rollout jitter, and token-phase diagnostics.  This tool
joins those files without rerunning inference so checkpoint selection does not
depend on hand-copied numbers.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any


STATIC_KEYS = {
    "decoder_total": "decoder/decoder_total",
    "fk_mpjpe_m": "decoder_quality/fk_mpjpe",
    "foot_slide": "decoder_quality/foot_slide",
    "joint_jerk": "decoder_quality/joint_jerk_l1",
    "boundary_joint_jerk": "decoder_quality/joint_boundary_jerk_l1",
    "first_frame_joint_seam_mps": (
        "decoder_seam/physical_seam_joint_jump_error_mps"
    ),
}

ROLLOUT_KEYS = {
    "acceleration_p95_ratio": "root_relative_joint_interior_acceleration_p95",
    "jerk_p95_ratio": "root_relative_joint_interior_jerk_p95",
    "high_frequency_ratio": "root_relative_joint_high_frequency_ratio",
    "joint_seam_ratio": "joint_seam_velocity_jump_mean",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, nargs="+", required=True)
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[20260714, 20260715, 20260716, 20260717, 20260718],
    )
    parser.add_argument(
        "--prefix", default="codec_boundary_4gpu_b512", help="Evidence file prefix"
    )
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def first_existing(paths: list[Path]) -> Path:
    for path in paths:
        if path.is_file():
            return path
    raise FileNotFoundError("none of the evidence paths exist: " + ", ".join(map(str, paths)))


def static_metrics(eval_dir: Path, prefix: str, step: int, variant: str) -> dict[str, float]:
    path = eval_dir / f"{prefix}_step{step:07d}_{variant}_fp32.json"
    metrics = read_json(path)["metrics"]
    return {name: float(metrics[key]) for name, key in STATIC_KEYS.items()}


def seed_suffix(seed: int, first_seed: int) -> str:
    return "" if seed == first_seed else f"_seed{seed}"


def jitter_path(
    eval_dir: Path, prefix: str, step: int, seed: int, first_seed: int
) -> Path:
    suffix = seed_suffix(seed, first_seed)
    stem = f"{prefix}_step{step:07d}_raw_jitter{suffix}"
    candidates = [eval_dir / f"{stem}.json"]
    if not suffix:
        candidates.append(eval_dir / f"{prefix}_step{step:07d}_raw_jitter_analysis.json")
    return first_existing(candidates)


def phase_path(
    eval_dir: Path, prefix: str, step: int, seed: int, first_seed: int
) -> Path:
    suffix = seed_suffix(seed, first_seed)
    candidates = [
        eval_dir / f"{prefix}_step{step:07d}_raw_token_phase{suffix}.json"
    ]
    if not suffix:
        candidates.append(
            eval_dir / f"{prefix}_step{step:07d}_raw_token_phase_jitter.json"
        )
    return first_existing(candidates)


def rollout_metrics(path: Path) -> dict[str, float]:
    snapshot = read_json(path)["snapshots"]["50"]
    codec = snapshot["ratios_to_teacher"]["codec_oracle"]
    return {
        name: float(codec[key]["ratio_to_teacher"])
        for name, key in ROLLOUT_KEYS.items()
    }


def phase_metrics(path: Path) -> dict[str, list[float]]:
    metrics = read_json(path)["metrics"]
    result: dict[str, list[float]] = {}
    for name in ("velocity", "acceleration", "jerk"):
        result[name + "_p95_ratio"] = [
            float(metrics[name]["codec_norm"][str(phase)]["p95"])
            / float(metrics[name]["teacher_norm"][str(phase)]["p95"])
            for phase in range(4)
        ]
    return result


def aggregate_scalars(rows: list[dict[str, float]]) -> dict[str, dict[str, float]]:
    return {
        key: {
            "mean": statistics.fmean(row[key] for row in rows),
            "median": statistics.median(row[key] for row in rows),
            "min": min(row[key] for row in rows),
            "max": max(row[key] for row in rows),
        }
        for key in rows[0]
    }


def aggregate_phases(
    rows: list[dict[str, list[float]]],
) -> dict[str, dict[str, list[float]]]:
    result: dict[str, dict[str, list[float]]] = {}
    for key in rows[0]:
        result[key] = {
            "mean": [
                statistics.fmean(row[key][phase] for row in rows)
                for phase in range(4)
            ],
            "max": [max(row[key][phase] for row in rows) for phase in range(4)],
        }
    return result


def percent_delta(current: float, previous: float) -> float:
    if previous == 0:
        raise ZeroDivisionError("cannot compute checkpoint delta from zero")
    return (current / previous - 1.0) * 100.0


def change_counts(values: list[float], tolerance: float = 1.0e-12) -> dict[str, int]:
    """Count paired lower-is-better changes without hiding seed regressions."""

    improved = sum(value < -tolerance for value in values)
    regressed = sum(value > tolerance for value in values)
    return {
        "improved_count": improved,
        "regressed_count": regressed,
        "unchanged_count": len(values) - improved - regressed,
    }


def aggregate_paired_scalars(
    rows: list[dict[str, float]],
) -> dict[str, dict[str, float | int]]:
    result: dict[str, dict[str, float | int]] = {}
    for key in rows[0]:
        values = [row[key] for row in rows]
        result[key] = {
            "mean_delta_percent": statistics.fmean(values),
            "median_delta_percent": statistics.median(values),
            "best_delta_percent": min(values),
            "worst_delta_percent": max(values),
            **change_counts(values),
        }
    return result


def aggregate_paired_phases(
    rows: list[dict[str, list[float]]],
) -> dict[str, dict[str, list[float] | list[int]]]:
    result: dict[str, dict[str, list[float] | list[int]]] = {}
    for key in rows[0]:
        by_phase = [
            [row[key][phase] for row in rows]
            for phase in range(4)
        ]
        counts = [change_counts(values) for values in by_phase]
        result[key] = {
            "mean_delta_percent": [statistics.fmean(values) for values in by_phase],
            "median_delta_percent": [statistics.median(values) for values in by_phase],
            "best_delta_percent": [min(values) for values in by_phase],
            "worst_delta_percent": [max(values) for values in by_phase],
            "improved_count": [item["improved_count"] for item in counts],
            "regressed_count": [item["regressed_count"] for item in counts],
            "unchanged_count": [item["unchanged_count"] for item in counts],
        }
    return result


def checkpoint_record(
    eval_dir: Path, prefix: str, step: int, seeds: list[int]
) -> dict[str, Any]:
    first_seed = seeds[0]
    jitter_rows = [
        rollout_metrics(jitter_path(eval_dir, prefix, step, seed, first_seed))
        for seed in seeds
    ]
    phase_rows = [
        phase_metrics(phase_path(eval_dir, prefix, step, seed, first_seed))
        for seed in seeds
    ]
    return {
        "step": step,
        "static": {
            variant: static_metrics(eval_dir, prefix, step, variant)
            for variant in ("raw", "ema")
        },
        "rollout_50": {
            "per_seed": {
                str(seed): row for seed, row in zip(seeds, jitter_rows, strict=True)
            },
            "aggregate": aggregate_scalars(jitter_rows),
        },
        "token_phase": {
            "per_seed": {
                str(seed): row for seed, row in zip(seeds, phase_rows, strict=True)
            },
            "aggregate": aggregate_phases(phase_rows),
        },
    }


def add_deltas(records: list[dict[str, Any]]) -> None:
    for previous, current in zip(records, records[1:]):
        seed_order = list(current["rollout_50"]["per_seed"])
        if seed_order != list(previous["rollout_50"]["per_seed"]):
            raise ValueError("checkpoint seed sets or ordering do not match")
        rollout_paired = {
            seed: {
                key: percent_delta(value, previous["rollout_50"]["per_seed"][seed][key])
                for key, value in current["rollout_50"]["per_seed"][seed].items()
            }
            for seed in seed_order
        }
        phase_paired = {
            seed: {
                key: [
                    percent_delta(value, previous["token_phase"]["per_seed"][seed][key][phase])
                    for phase, value in enumerate(values)
                ]
                for key, values in current["token_phase"]["per_seed"][seed].items()
            }
            for seed in seed_order
        }
        current["delta_percent_vs_previous"] = {
            "static_raw": {
                key: percent_delta(value, previous["static"]["raw"][key])
                for key, value in current["static"]["raw"].items()
            },
            "rollout_50_mean": {
                key: percent_delta(
                    value["mean"],
                    previous["rollout_50"]["aggregate"][key]["mean"],
                )
                for key, value in current["rollout_50"]["aggregate"].items()
            },
            "rollout_50_paired": {
                "per_seed": rollout_paired,
                "aggregate": aggregate_paired_scalars(list(rollout_paired.values())),
            },
            "token_phase_paired": {
                "per_seed": phase_paired,
                "aggregate": aggregate_paired_phases(list(phase_paired.values())),
            },
        }


def print_table(records: list[dict[str, Any]]) -> None:
    print("step\tfk_mm\tjerk\thf_ratio\tseam_ratio\tphase1_jerk")
    for record in records:
        rollout = record["rollout_50"]["aggregate"]
        phase = record["token_phase"]["aggregate"]["jerk_p95_ratio"]["mean"]
        print(
            f"{record['step']}\t"
            f"{record['static']['raw']['fk_mpjpe_m'] * 1000:.4f}\t"
            f"{rollout['jerk_p95_ratio']['mean']:.6f}\t"
            f"{rollout['high_frequency_ratio']['mean']:.6f}\t"
            f"{rollout['joint_seam_ratio']['mean']:.6f}\t"
            f"{phase[1]:.6f}"
        )


def main() -> None:
    args = parse_args()
    steps = sorted(set(args.steps))
    seeds = list(dict.fromkeys(args.seeds))
    if any(step < 1 for step in steps) or not seeds:
        raise ValueError("steps must be positive and at least one seed is required")
    records = [
        checkpoint_record(args.eval_dir, args.prefix, step, seeds) for step in steps
    ]
    add_deltas(records)
    result = {
        "schema": "ardy_codec_checkpoint_sweep_v2",
        "eval_dir": str(args.eval_dir),
        "prefix": args.prefix,
        "steps": steps,
        "seeds": seeds,
        "checkpoint_count": len(records),
        "checkpoints": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print_table(records)
    print(json.dumps({"event": "complete", "output": str(args.output)}))


if __name__ == "__main__":
    main()
