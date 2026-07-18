#!/usr/bin/env python
"""Join static and faithful-rollout evidence for NFE flow checkpoints.

The training log is deliberately not used for model selection.  This tool
joins independent validation, five-seed faithful rollout, and temporal/frequency
analysis so every later checkpoint is compared with the same paths and metrics.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any


STATIC_KEYS = {
    "endpoint_total": "nfe{nfe}/endpoint_total",
    "endpoint_root_mse": "nfe{nfe}/endpoint_root_mse",
    "endpoint_body_mse": "nfe{nfe}/endpoint_body_mse",
    "fk_mpjpe_m": "nfe{nfe}_quality/fk_mpjpe",
    "foot_slide_mps": "nfe{nfe}_quality/foot_slide",
    "joint_acceleration": "nfe{nfe}_quality/joint_acceleration_l1",
    "joint_jerk": "nfe{nfe}_quality/joint_jerk_l1",
    "boundary_joint_jerk": "nfe{nfe}_quality/joint_boundary_jerk_l1",
    "root_temporal_total": "nfe{nfe}_root_temporal/root_temporal_total",
    "path_error_m": "path/path_error_m",
    "future_waypoint_remaining_m": "path/future_waypoint_remaining_m",
}

JITTER_KEYS = {
    "root_acceleration_p95_ratio": "root_horizontal_interior_acceleration_p95",
    "root_jerk_p95_ratio": "root_horizontal_interior_jerk_p95",
    "root_high_frequency_ratio": "root_horizontal_high_frequency_ratio",
    "body_acceleration_p95_ratio": "root_relative_joint_interior_acceleration_p95",
    "body_jerk_p95_ratio": "root_relative_joint_interior_jerk_p95",
    "body_high_frequency_ratio": "root_relative_joint_high_frequency_ratio",
    "joint_seam_ratio": "joint_seam_velocity_jump_mean",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-dir", type=Path, required=True)
    parser.add_argument("--rollout-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, nargs="+", required=True)
    parser.add_argument("--nfe", type=int, default=4)
    parser.add_argument("--prefix", default="flow_nfe4_formal")
    parser.add_argument(
        "--variants",
        choices=["raw", "ema"],
        nargs="+",
        default=["raw", "ema"],
        help="Checkpoint variants that must be present and compared.",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[20260714, 20260715, 20260716, 20260717, 20260718],
    )
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def static_path(eval_dir: Path, prefix: str, step: int, variant: str) -> Path:
    return eval_dir / f"{prefix}_step{step:07d}_{variant}_fp32.json"


def jitter_path(
    eval_dir: Path, prefix: str, step: int, variant: str, seed: int
) -> Path:
    return eval_dir / f"{prefix}_step{step:07d}_{variant}_jitter_seed{seed}.json"


def rollout_path(
    rollout_dir: Path, prefix: str, step: int, variant: str, seed: int
) -> Path:
    return (
        rollout_dir
        / f"{prefix}_step{step:07d}_{variant}_oracle50_seed{seed}"
        / "metrics.json"
    )


def static_metrics(path: Path, nfe: int) -> dict[str, float]:
    metrics = read_json(path)["metrics"]
    return {
        name: float(metrics[key.format(nfe=nfe)])
        for name, key in STATIC_KEYS.items()
    }


def rollout_metrics(rollout: Path, jitter: Path) -> dict[str, float]:
    rollout_snapshot = read_json(rollout)["snapshots"]["50"]
    jitter_snapshot = read_json(jitter)["snapshots"]["50"]
    ratios = jitter_snapshot["ratios_to_teacher"]["raw_student"]
    result = {
        "waypoint_error_m": float(
            rollout_snapshot["waypoints"]["student_error_m"]["mean"]
        ),
        "root_drift_mean_m": float(
            rollout_snapshot["teacher_student"]["root_drift_mean_m"]
        ),
        "root_drift_final_m": float(
            rollout_snapshot["teacher_student"]["root_drift_final_m"]
        ),
        "fk_mpjpe_m": float(rollout_snapshot["teacher_student"]["fk_mpjpe_m"]),
        "foot_slide_mps": float(rollout_snapshot["student"]["foot_slide_mps"]),
        "joint_seam_mps": float(
            rollout_snapshot["student"]["seam_joint_velocity_jump_mps"]["mean"]
        ),
    }
    result.update(
        {
            name: float(ratios[key]["ratio_to_teacher"])
            for name, key in JITTER_KEYS.items()
        }
    )
    return result


def aggregate(rows: list[dict[str, float]]) -> dict[str, dict[str, float]]:
    if not rows:
        raise ValueError("cannot aggregate an empty metric set")
    return {
        key: {
            "mean": statistics.fmean(row[key] for row in rows),
            "median": statistics.median(row[key] for row in rows),
            "min": min(row[key] for row in rows),
            "max": max(row[key] for row in rows),
        }
        for key in rows[0]
    }


def percent_delta(current: float, previous: float) -> float:
    if previous == 0:
        raise ZeroDivisionError("cannot compute checkpoint delta from zero")
    return (current / previous - 1.0) * 100.0


def change_counts(values: list[float], tolerance: float = 1.0e-12) -> dict[str, int]:
    improved = sum(value < -tolerance for value in values)
    regressed = sum(value > tolerance for value in values)
    return {
        "improved_count": improved,
        "regressed_count": regressed,
        "unchanged_count": len(values) - improved - regressed,
    }


def paired_aggregate(rows: list[dict[str, float]]) -> dict[str, dict[str, float | int]]:
    return {
        key: {
            "mean_delta_percent": statistics.fmean(row[key] for row in rows),
            "median_delta_percent": statistics.median(row[key] for row in rows),
            "best_delta_percent": min(row[key] for row in rows),
            "worst_delta_percent": max(row[key] for row in rows),
            **change_counts([row[key] for row in rows]),
        }
        for key in rows[0]
    }


def variant_record(
    eval_dir: Path,
    rollout_dir: Path,
    prefix: str,
    step: int,
    variant: str,
    seeds: list[int],
    nfe: int,
) -> dict[str, Any] | None:
    static = static_path(eval_dir, prefix, step, variant)
    if not static.is_file():
        return None
    rows = [
        rollout_metrics(
            rollout_path(rollout_dir, prefix, step, variant, seed),
            jitter_path(eval_dir, prefix, step, variant, seed),
        )
        for seed in seeds
    ]
    return {
        "static": static_metrics(static, nfe),
        "rollout_50": {
            "per_seed": {
                str(seed): row for seed, row in zip(seeds, rows, strict=True)
            },
            "aggregate": aggregate(rows),
        },
    }


def checkpoint_record(
    eval_dir: Path,
    rollout_dir: Path,
    prefix: str,
    step: int,
    requested_variants: list[str],
    seeds: list[int],
    nfe: int,
) -> dict[str, Any]:
    variants = {}
    for variant in requested_variants:
        record = variant_record(
            eval_dir, rollout_dir, prefix, step, variant, seeds, nfe
        )
        if record is None:
            raise FileNotFoundError(static_path(eval_dir, prefix, step, variant))
        variants[variant] = record
    return {"step": step, "variants": variants}


def add_variant_deltas(records: list[dict[str, Any]]) -> None:
    for previous, current in zip(records, records[1:]):
        for variant, current_evidence in current["variants"].items():
            if variant not in previous["variants"]:
                continue
            previous_evidence = previous["variants"][variant]
            seed_order = list(current_evidence["rollout_50"]["per_seed"])
            if seed_order != list(previous_evidence["rollout_50"]["per_seed"]):
                raise ValueError("checkpoint seed sets or ordering do not match")
            paired = {
                seed: {
                    key: percent_delta(
                        value,
                        previous_evidence["rollout_50"]["per_seed"][seed][key],
                    )
                    for key, value in current_evidence["rollout_50"]["per_seed"][seed].items()
                }
                for seed in seed_order
            }
            current[f"{variant}_delta_percent_vs_previous"] = {
                "static": {
                    key: percent_delta(value, previous_evidence["static"][key])
                    for key, value in current_evidence["static"].items()
                },
                "rollout_50_mean": {
                    key: percent_delta(
                        value["mean"],
                        previous_evidence["rollout_50"]["aggregate"][key]["mean"],
                    )
                    for key, value in current_evidence["rollout_50"]["aggregate"].items()
                },
                "rollout_50_paired": {
                    "per_seed": paired,
                    "aggregate": paired_aggregate(list(paired.values())),
                },
            }


def print_table(records: list[dict[str, Any]]) -> None:
    print("step\tvariant\tstatic_fk_mm\twaypoint_m\trollout_fk_m\tbody_jerk\tbody_hf\tseam")
    for record in records:
        for variant, evidence in record["variants"].items():
            rollout = evidence["rollout_50"]["aggregate"]
            print(
                f"{record['step']}\t{variant}\t"
                f"{evidence['static']['fk_mpjpe_m'] * 1000:.4f}\t"
                f"{rollout['waypoint_error_m']['mean']:.6f}\t"
                f"{rollout['fk_mpjpe_m']['mean']:.6f}\t"
                f"{rollout['body_jerk_p95_ratio']['mean']:.6f}\t"
                f"{rollout['body_high_frequency_ratio']['mean']:.6f}\t"
                f"{rollout['joint_seam_ratio']['mean']:.6f}"
            )


def main() -> None:
    args = parse_args()
    steps = sorted(set(args.steps))
    seeds = list(dict.fromkeys(args.seeds))
    if not steps or any(step < 0 for step in steps):
        raise ValueError("steps must be non-negative")
    if args.nfe < 1 or not seeds:
        raise ValueError("NFE and seed set must be positive/non-empty")
    records = [
        checkpoint_record(
            args.eval_dir,
            args.rollout_dir,
            args.prefix,
            step,
            list(dict.fromkeys(args.variants)),
            seeds,
            args.nfe,
        )
        for step in steps
    ]
    add_variant_deltas(records)
    result = {
        "schema": "ardy_flow_checkpoint_sweep_v1",
        "eval_dir": str(args.eval_dir),
        "rollout_dir": str(args.rollout_dir),
        "prefix": args.prefix,
        "nfe": args.nfe,
        "steps": steps,
        "variants": list(dict.fromkeys(args.variants)),
        "seeds": seeds,
        "checkpoints": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print_table(records)
    print(json.dumps({"event": "complete", "output": str(args.output)}))


if __name__ == "__main__":
    main()
