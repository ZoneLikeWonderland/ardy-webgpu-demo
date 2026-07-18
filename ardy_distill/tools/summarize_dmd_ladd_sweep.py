"""Summarize controlled DMD/LADD arms against one frozen stage-one baseline."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


LOWER_IS_BETTER = (
    "endpoint_total",
    "fk_mpjpe",
    "foot_slide",
    "joint_jerk",
    "root_temporal",
    "path_error",
    "mouse_dense_endpoint",
    "unconditional_endpoint",
    "prompt_switch_endpoint",
)

ROLLOUT_LOWER_IS_BETTER = (
    "waypoint_error_m",
    "root_drift_mean_m",
    "root_drift_final_m",
    "fk_mpjpe_m",
    "foot_slide_mps",
    "seam_joint_velocity_jump_mps",
    "body_jerk_p95",
    "body_high_frequency_ratio",
    "root_horizontal_high_frequency_ratio",
    "root_heading_high_frequency_ratio",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--baseline-full", type=Path, required=True)
    parser.add_argument("--baseline-text", type=Path, required=True)
    parser.add_argument("--step", default="step000200")
    parser.add_argument("--rollout-seeds", type=int, nargs="*", default=[])
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def full_metrics(document: dict) -> dict[str, float]:
    metrics = document["metrics"]
    return {
        "endpoint_total": float(metrics["nfe1/endpoint_total"]),
        "fsq_bin_accuracy": float(metrics["nfe1/fsq_bin_accuracy"]),
        "fk_mpjpe": float(metrics["nfe1_quality/fk_mpjpe"]),
        "foot_slide": float(metrics["nfe1_quality/foot_slide"]),
        "joint_jerk": float(metrics["nfe1_quality/joint_jerk_l1"]),
        "root_temporal": float(metrics["nfe1_root_temporal/root_temporal_total"]),
        "path_error": float(metrics["path/path_error_m"]),
    }


def group_metric(document: dict, group: str, metric: str) -> float:
    return float(document["groups"][group]["metrics"][metric]["mean"])


def text_metrics(document: dict) -> dict[str, float]:
    return {
        "text_gain_over_zero": group_metric(document, "all", "text_gain_over_zero"),
        "heading_gain_over_zero": group_metric(document, "all", "heading_gain_over_zero"),
        "mouse_dense_endpoint": group_metric(
            document, "control/mouse_dense", "endpoint_total"
        ),
        "unconditional_endpoint": group_metric(
            document, "prompt/unconditional", "endpoint_total"
        ),
        "prompt_switch_endpoint": group_metric(
            document, "prompt_switch/true", "endpoint_total"
        ),
    }


def relative_percent(value: float, baseline: float) -> float:
    return 100.0 * (value - baseline) / abs(baseline)


def aggregate(values: list[float]) -> dict[str, float | int]:
    if not values:
        raise ValueError("cannot aggregate an empty metric list")
    return {
        "count": len(values),
        "mean": statistics.fmean(values),
        "std_population": statistics.pstdev(values),
        "min": min(values),
        "max": max(values),
    }


def rollout_metrics(root: Path, tag: str, seed: int) -> dict[str, float | bool]:
    directory = root / f"rollout_seed{seed}" / tag
    rollout_path = directory / "metrics.json"
    jitter_path = directory / "jitter.json"
    if not rollout_path.is_file() or not jitter_path.is_file():
        raise FileNotFoundError(
            f"missing matched rollout/jitter evidence for {tag} seed {seed}: {directory}"
        )
    snapshot = load(rollout_path)["snapshots"]["50"]
    jitter = load(jitter_path)["snapshots"]["50"]["sources"]["raw_student"]
    body = jitter["root_relative_joints_m"]
    root_horizontal = jitter["root_horizontal_m"]
    root_heading = jitter["root_heading_rad"]
    return {
        "all_finite": bool(snapshot["all_finite"]),
        "waypoint_error_m": float(snapshot["waypoints"]["student_error_m"]["mean"]),
        "root_drift_mean_m": float(snapshot["teacher_student"]["root_drift_mean_m"]),
        "root_drift_final_m": float(snapshot["teacher_student"]["root_drift_final_m"]),
        "fk_mpjpe_m": float(snapshot["teacher_student"]["fk_mpjpe_m"]),
        "foot_slide_mps": float(snapshot["student"]["foot_slide_mps"]),
        "seam_joint_velocity_jump_mps": float(
            snapshot["student"]["seam_joint_velocity_jump_mps"]["mean"]
        ),
        "body_jerk_p95": float(
            body["segment_interior_margin_4_frames"]["jerk"]["p95"]
        ),
        "body_high_frequency_ratio": float(
            body["segment_local_high_frequency"]["aggregate_ratio"]
        ),
        "root_horizontal_high_frequency_ratio": float(
            root_horizontal["segment_local_high_frequency"]["aggregate_ratio"]
        ),
        "root_heading_high_frequency_ratio": float(
            root_heading["segment_local_high_frequency"]["aggregate_ratio"]
        ),
    }


def rollout_summary(
    root: Path,
    tag: str,
    seeds: list[int],
    baseline_by_seed: dict[int, dict[str, float | bool]],
) -> dict:
    per_seed = {seed: rollout_metrics(root, tag, seed) for seed in seeds}
    metrics = {
        name: aggregate([float(per_seed[seed][name]) for seed in seeds])
        for name in ROLLOUT_LOWER_IS_BETTER
    }
    paired_percent = {
        name: aggregate(
            [
                relative_percent(
                    float(per_seed[seed][name]),
                    float(baseline_by_seed[seed][name]),
                )
                for seed in seeds
            ]
        )
        for name in ROLLOUT_LOWER_IS_BETTER
    }
    wins = {
        name: sum(
            float(per_seed[seed][name]) < float(baseline_by_seed[seed][name])
            for seed in seeds
        )
        for name in ROLLOUT_LOWER_IS_BETTER
    }
    return {
        "all_finite": all(bool(per_seed[seed]["all_finite"]) for seed in seeds),
        "per_seed": {str(seed): per_seed[seed] for seed in seeds},
        "aggregate": metrics,
        "paired_percent_vs_stage1": paired_percent,
        "paired_wins_vs_stage1": wins,
    }


def completed_training_metrics(path: Path) -> dict:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    completion = next(
        (row for row in reversed(rows) if row.get("event") == "training_complete"),
        None,
    )
    if completion is None:
        raise RuntimeError(f"missing training_complete: {path}")
    last_update = next(
        (row for row in reversed(rows) if row.get("event") is None),
        None,
    )
    return {
        "generator_updates": int(completion["step"]),
        "guidance_updates": int(completion["guidance_updates"]),
        "elapsed_s": float(completion["elapsed_s"]),
        "last_logged": last_update,
    }


def pareto_flags(rows: list[dict]) -> None:
    for row in rows:
        row["pareto_nondominated"] = True
        for other in rows:
            if other is row:
                continue
            no_worse = all(
                other["metrics"][name] <= row["metrics"][name]
                for name in LOWER_IS_BETTER
            )
            strictly_better = any(
                other["metrics"][name] < row["metrics"][name]
                for name in LOWER_IS_BETTER
            )
            if no_worse and strictly_better:
                row["pareto_nondominated"] = False
                break


def main() -> None:
    args = parse_args()
    baseline = {
        **full_metrics(load(args.baseline_full)),
        **text_metrics(load(args.baseline_text)),
    }
    rows: list[dict] = []
    for arm in sorted(path for path in args.root.iterdir() if path.is_dir()):
        config_path = arm / "config.json"
        full_path = arm / "eval" / f"{args.step}_ema_full_fp16.json"
        text_path = arm / "eval" / f"{args.step}_ema_text_control_fp16.json"
        metrics_path = arm / "metrics.jsonl"
        if not all(path.is_file() for path in (config_path, full_path, text_path, metrics_path)):
            continue
        config = load(config_path)
        metrics = {
            **full_metrics(load(full_path)),
            **text_metrics(load(text_path)),
        }
        rows.append(
            {
                "arm": arm.name,
                "generator_learning_rate": float(config["generator_learning_rate"]),
                "score_learning_rate": float(config["score_learning_rate"]),
                "critic_learning_rate": float(config["critic_learning_rate"]),
                "steps": int(config["steps"]),
                "guidance_updates_per_generator": int(
                    config["guidance_updates_per_generator"]
                ),
                "warmup_guidance_updates": int(config["warmup_guidance_updates"]),
                "metrics": metrics,
                "delta_percent_vs_stage1": {
                    name: relative_percent(value, baseline[name])
                    for name, value in metrics.items()
                },
                "training": completed_training_metrics(metrics_path),
            }
        )
    if not rows:
        raise RuntimeError(f"no complete evaluated arms under {args.root}")
    pareto_flags(rows)
    rollout = None
    if args.rollout_seeds:
        seeds = list(dict.fromkeys(args.rollout_seeds))
        baseline_by_seed = {
            seed: rollout_metrics(args.root, "baseline", seed) for seed in seeds
        }
        rollout = {
            "seeds": seeds,
            "snapshot_windows": 50,
            "model_dtype": "fp16",
            "lower_is_better": list(ROLLOUT_LOWER_IS_BETTER),
            "baseline": rollout_summary(
                args.root, "baseline", seeds, baseline_by_seed
            ),
        }
        for row in rows:
            rollout_dir = args.root / f"rollout_seed{seeds[0]}" / row["arm"]
            if rollout_dir.is_dir():
                row["rollout"] = rollout_summary(
                    args.root, row["arm"], seeds, baseline_by_seed
                )
    result = {
        "schema": "ardy_dmd_ladd_sweep_v1",
        "root": str(args.root),
        "baseline": {
            "full": str(args.baseline_full),
            "text": str(args.baseline_text),
            "metrics": baseline,
        },
        "lower_is_better": list(LOWER_IS_BETTER),
        "arms": rows,
        "rollout": rollout,
        "note": (
            "No single weighted score is used. Pareto flags cover endpoint, physical, "
            "path, and hard conditioning groups; visual/WebGPU rollout remains a separate gate."
        ),
    }
    output = args.output or args.root / "sweep_summary.json"
    output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print("arm\tG_lr\tendpoint\tFK_m\tfoot\tjerk\troot\tpath\tpareto")
    for row in rows:
        metrics = row["metrics"]
        print(
            f"{row['arm']}\t{row['generator_learning_rate']:.1e}\t"
            f"{metrics['endpoint_total']:.6f}\t{metrics['fk_mpjpe']:.6f}\t"
            f"{metrics['foot_slide']:.6f}\t{metrics['joint_jerk']:.6f}\t"
            f"{metrics['root_temporal']:.6f}\t{metrics['path_error']:.6f}\t"
            f"{row['pareto_nondominated']}"
        )
    if rollout is not None:
        print("arm\twaypoint%\tFK%\tbody_jerk%\tbody_HF%\tseam%\tfinite")
        for row in rows:
            if "rollout" not in row:
                continue
            paired = row["rollout"]["paired_percent_vs_stage1"]
            print(
                f"{row['arm']}\t{paired['waypoint_error_m']['mean']:+.3f}\t"
                f"{paired['fk_mpjpe_m']['mean']:+.3f}\t"
                f"{paired['body_jerk_p95']['mean']:+.3f}\t"
                f"{paired['body_high_frequency_ratio']['mean']:+.3f}\t"
                f"{paired['seam_joint_velocity_jump_mps']['mean']:+.3f}\t"
                f"{row['rollout']['all_finite']}"
            )
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
