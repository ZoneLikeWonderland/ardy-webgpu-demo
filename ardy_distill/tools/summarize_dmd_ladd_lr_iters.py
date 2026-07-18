"""Summarize the corrected-framework generator-LR x iteration experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .summarize_dmd_ladd_guidance_ratio import paired_rollout_summary
from .summarize_dmd_ladd_sweep import (
    LOWER_IS_BETTER,
    ROLLOUT_LOWER_IS_BETTER,
    aggregate,
    full_metrics,
    load,
    relative_percent,
    rollout_metrics,
    text_metrics,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--baseline-rollout-root", type=Path, required=True)
    parser.add_argument("--baseline-full", type=Path, required=True)
    parser.add_argument("--baseline-text", type=Path, required=True)
    parser.add_argument(
        "--control-g400-root",
        type=Path,
        help=(
            "Optional adversarial-time experiment root containing the exact "
            "u05_legacy_advtime g400 control that precedes lr1e7_control."
        ),
    )
    parser.add_argument(
        "--arms",
        nargs="+",
        default=["lr1e8", "lr3e8", "lr1e7_control", "lr3e7", "lr1e6"],
    )
    parser.add_argument("--steps", type=int, nargs="+", default=[400, 700, 1000, 1500])
    parser.add_argument(
        "--rollout-seeds",
        type=int,
        nargs="+",
        default=[20260714, 20260715, 20260716, 20260717],
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def checkpoint_rollout_metrics(root: Path, arm: str, step: int, seed: int) -> dict:
    adapter_root = root / f"rollout_step{step:06d}_seed{seed}"
    # rollout_metrics expects <root>/rollout_seed<seed>/<tag>. Point it at a
    # synthetic parent by reading the same schema directly from our step-aware
    # directory; keeping the extraction identical avoids metric drift.
    directory = adapter_root / arm
    rollout_path = directory / "metrics.json"
    jitter_path = directory / "jitter.json"
    if not rollout_path.is_file() or not jitter_path.is_file():
        raise FileNotFoundError(
            f"missing rollout/jitter evidence for {arm} step {step} seed {seed}: "
            f"{directory}"
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


def paired_rollout(
    root: Path,
    arm: str,
    step: int,
    baseline_rollout_root: Path,
    seeds: list[int],
) -> dict:
    per_seed = {
        seed: checkpoint_rollout_metrics(root, arm, step, seed) for seed in seeds
    }
    baseline = {
        seed: rollout_metrics(baseline_rollout_root, "baseline", seed)
        for seed in seeds
    }
    paired = {
        name: aggregate(
            [
                relative_percent(
                    float(per_seed[seed][name]),
                    float(baseline[seed][name]),
                )
                for seed in seeds
            ]
        )
        for name in ROLLOUT_LOWER_IS_BETTER
    }
    return {
        "all_finite": all(bool(per_seed[seed]["all_finite"]) for seed in seeds),
        "per_seed": {str(seed): per_seed[seed] for seed in seeds},
        "paired_percent_vs_stage1": paired,
        "paired_wins_vs_stage1": {
            name: sum(
                float(per_seed[seed][name]) < float(baseline[seed][name])
                for seed in seeds
            )
            for name in ROLLOUT_LOWER_IS_BETTER
        },
    }


def main() -> None:
    args = parse_args()
    seeds = list(dict.fromkeys(args.rollout_seeds))
    baseline = {
        **full_metrics(load(args.baseline_full)),
        **text_metrics(load(args.baseline_text)),
    }
    rows = []
    missing = []
    for arm in args.arms:
        arm_root = args.root / arm
        configs = sorted(arm_root.glob("resume_config_step-*.json"))
        if not configs:
            missing.append(f"{arm}: resume config")
            continue
        config = load(configs[0])
        for step in args.steps:
            external_control = bool(
                arm == "lr1e7_control"
                and step == 400
                and args.control_g400_root is not None
            )
            evidence_root = (
                args.control_g400_root / "u05_legacy_advtime"
                if external_control
                else arm_root
            )
            full_path = evidence_root / "eval" / f"step{step:06d}_ema_full_fp16.json"
            text_path = (
                evidence_root / "eval" / f"step{step:06d}_ema_text_control_fp16.json"
            )
            if not full_path.is_file() or not text_path.is_file():
                missing.append(f"{arm}: static/text step {step}")
                continue
            try:
                if external_control:
                    rollout = paired_rollout_summary(
                        args.control_g400_root,
                        "u05_legacy_advtime",
                        args.baseline_rollout_root,
                        seeds,
                    )
                else:
                    rollout = paired_rollout(
                        args.root,
                        arm,
                        step,
                        args.baseline_rollout_root,
                        seeds,
                    )
            except FileNotFoundError as error:
                missing.append(str(error))
                continue
            metrics = {
                **full_metrics(load(full_path)),
                **text_metrics(load(text_path)),
            }
            rows.append(
                {
                    "arm": arm,
                    "step": step,
                    "generator_learning_rate": float(
                        config["generator_learning_rate"]
                    ),
                    "score_learning_rate": float(config["score_learning_rate"]),
                    "critic_learning_rate": float(config["critic_learning_rate"]),
                    "resume_learning_rate_policy": config.get(
                        "override_learning_rates_on_resume", False
                    ),
                    "evidence_root": str(evidence_root),
                    "metrics": metrics,
                    "delta_percent_vs_stage1": {
                        name: relative_percent(value, baseline[name])
                        for name, value in metrics.items()
                    },
                    "rollout": rollout,
                }
            )
    if not rows:
        raise RuntimeError(f"no fully evaluated LR/iteration rows under {args.root}")

    result = {
        "schema": "ardy_dmd_ladd_generator_lr_iters_v1",
        "root": str(args.root),
        "baseline": baseline,
        "seeds": seeds,
        "lower_is_better": list(LOWER_IS_BETTER),
        "rollout_lower_is_better": list(ROLLOUT_LOWER_IS_BETTER),
        "rows": rows,
        "missing_or_pending": missing,
        "note": (
            (
                "All non-control arms resume the same u05 g300 full state; the "
                "1e-7 g400 control is supplied by the preceding adversarial-time "
                "experiment. "
            )
            if args.control_g400_root is not None
            else (
                "All arms, including the independently reproduced 1e-7 control, "
                "resume the same u05 g300 full state. "
            )
        )
        + (
            "Adam moments, EMA, sampler and RNG are retained while constant LRs "
            "are explicitly overridden. No weighted aggregate score is used."
        ),
    }
    output = args.output or args.root / "generator_lr_iters_summary.json"
    output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    print("arm\tstep\tendpoint%\twaypoint%\tFK%\tfoot%\tseam%\tjerk%\tbodyHF%")
    for row in rows:
        static = row["delta_percent_vs_stage1"]
        rollout = row["rollout"]["paired_percent_vs_stage1"]
        print(
            f"{row['arm']}\t{row['step']}\t{static['endpoint_total']:+.3f}\t"
            f"{rollout['waypoint_error_m']['mean']:+.3f}\t"
            f"{rollout['fk_mpjpe_m']['mean']:+.3f}\t"
            f"{rollout['foot_slide_mps']['mean']:+.3f}\t"
            f"{rollout['seam_joint_velocity_jump_mps']['mean']:+.3f}\t"
            f"{rollout['body_jerk_p95']['mean']:+.3f}\t"
            f"{rollout['body_high_frequency_ratio']['mean']:+.3f}"
        )
    print(f"pending={len(missing)} wrote {output}")


if __name__ == "__main__":
    main()
