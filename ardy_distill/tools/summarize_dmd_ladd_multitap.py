"""Summarize shared-head multi-level LADD aggregation experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .summarize_dmd_ladd_feature_taps import paired_rollout, static_metrics
from .summarize_dmd_ladd_lr_iters import checkpoint_rollout_metrics
from .summarize_dmd_ladd_sweep import (
    LOWER_IS_BETTER,
    ROLLOUT_LOWER_IS_BETTER,
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
    parser.add_argument("--active-root", type=Path)
    parser.add_argument("--active-arm", default="g1_s1_d1")
    parser.add_argument("--active-step", type=int, default=400)
    parser.add_argument(
        "--arms",
        nargs="+",
        default=[
            "single_body_final",
            "multi_deep_mean_loss",
            "multi_three_mean_loss",
            "multi_three_mean_logit",
        ],
    )
    parser.add_argument("--control-arm", default="single_body_final")
    parser.add_argument("--steps", type=int, nargs="+", default=[350, 400, 450, 500])
    parser.add_argument(
        "--rollout-seeds",
        type=int,
        nargs="+",
        default=[20260714, 20260715, 20260716, 20260717],
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seeds = list(dict.fromkeys(args.rollout_seeds))
    if args.control_arm not in args.arms:
        raise ValueError("--control-arm must be included in --arms")

    baseline_static = {
        **full_metrics(load(args.baseline_full)),
        **text_metrics(load(args.baseline_text)),
    }
    baseline_rollout = {
        seed: rollout_metrics(args.baseline_rollout_root, "baseline", seed)
        for seed in seeds
    }

    active_static = None
    active_rollout = None
    if args.active_root is not None:
        active_static = static_metrics(
            args.active_root, args.active_arm, args.active_step
        )
        active_rollout = {
            seed: checkpoint_rollout_metrics(
                args.active_root, args.active_arm, args.active_step, seed
            )
            for seed in seeds
        }

    rows = []
    missing = []
    for step in args.steps:
        try:
            control_static = static_metrics(args.root, args.control_arm, step)
            control_rollout = {
                seed: checkpoint_rollout_metrics(
                    args.root, args.control_arm, step, seed
                )
                for seed in seeds
            }
        except FileNotFoundError as error:
            missing.append(str(error))
            continue

        for arm in args.arms:
            arm_root = args.root / arm
            configs = sorted(arm_root.glob("resume_config_step-*.json"))
            if not configs:
                missing.append(f"{arm}: resume config")
                continue
            config = load(configs[0])
            try:
                metrics = static_metrics(args.root, arm, step)
                rollout_stage1 = paired_rollout(
                    args.root, arm, step, baseline_rollout, seeds
                )
                rollout_control = paired_rollout(
                    args.root, arm, step, control_rollout, seeds
                )
                rollout_active = (
                    paired_rollout(args.root, arm, step, active_rollout, seeds)
                    if active_rollout is not None
                    else None
                )
            except FileNotFoundError as error:
                missing.append(str(error))
                continue

            row = {
                "arm": arm,
                "step": step,
                "critic_feature_taps": config["critic_feature_taps"],
                "critic_tap_aggregation": config["critic_tap_aggregation"],
                "metrics": metrics,
                "delta_percent_vs_stage1": {
                    name: relative_percent(value, baseline_static[name])
                    for name, value in metrics.items()
                },
                "delta_percent_vs_same_step_single": {
                    name: relative_percent(value, control_static[name])
                    for name, value in metrics.items()
                },
                "rollout_vs_stage1": rollout_stage1,
                "rollout_vs_same_step_single": rollout_control,
            }
            if active_static is not None and rollout_active is not None:
                row["delta_percent_vs_active_frontend"] = {
                    name: relative_percent(value, active_static[name])
                    for name, value in metrics.items()
                }
                row["rollout_vs_active_frontend"] = rollout_active
            rows.append(row)

    if not rows:
        raise RuntimeError(f"no fully evaluated multi-tap rows under {args.root}")

    result = {
        "schema": "ardy_dmd_ladd_shared_head_multitap_v1",
        "root": str(args.root),
        "control_arm": args.control_arm,
        "baseline_static": baseline_static,
        "active_frontend": (
            {
                "root": str(args.active_root),
                "arm": args.active_arm,
                "step": args.active_step,
                "metrics": active_static,
            }
            if args.active_root is not None
            else None
        ),
        "seeds": seeds,
        "lower_is_better": list(LOWER_IS_BETTER),
        "rollout_lower_is_better": list(ROLLOUT_LOWER_IS_BETTER),
        "rows": rows,
        "missing_or_pending": missing,
        "note": (
            "Every arm resumes the identical corrected-u05 g300 full state, "
            "performs the same S/D 700->800 recovery, and uses one inherited "
            "checkpoint-compatible critic head. Multi-level arms differ only "
            "in selected frozen-teacher taps and mean-loss versus mean-logit "
            "aggregation. No weighted quality score is used."
        ),
    }
    output = args.output or args.root / "multitap_summary.json"
    output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    print("arm\tstep\tendpoint%\twaypoint%\tFK%\tfoot%\tseam%\tjerk%\tbodyHF%\theadingHF%")
    for row in rows:
        static = row["delta_percent_vs_stage1"]
        rollout = row["rollout_vs_stage1"]["paired_percent_vs_reference"]
        print(
            f"{row['arm']}\t{row['step']}\t{static['endpoint_total']:+.3f}\t"
            f"{rollout['waypoint_error_m']['mean']:+.3f}\t"
            f"{rollout['fk_mpjpe_m']['mean']:+.3f}\t"
            f"{rollout['foot_slide_mps']['mean']:+.3f}\t"
            f"{rollout['seam_joint_velocity_jump_mps']['mean']:+.3f}\t"
            f"{rollout['body_jerk_p95']['mean']:+.3f}\t"
            f"{rollout['body_high_frequency_ratio']['mean']:+.3f}\t"
            f"{rollout['root_heading_high_frequency_ratio']['mean']:+.3f}"
        )
    print(f"pending={len(missing)} wrote {output}")


if __name__ == "__main__":
    main()
