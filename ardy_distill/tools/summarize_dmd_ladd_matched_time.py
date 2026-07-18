"""Summarize informative-time versus legacy-time LADD at matched weights."""

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
    parser.add_argument("--legacy-root", type=Path, required=True)
    parser.add_argument("--baseline-rollout-root", type=Path, required=True)
    parser.add_argument("--baseline-full", type=Path, required=True)
    parser.add_argument("--baseline-text", type=Path, required=True)
    parser.add_argument("--active-root", type=Path, required=True)
    parser.add_argument("--active-arm", default="g1_s1_d1")
    parser.add_argument("--active-step", type=int, default=400)
    parser.add_argument(
        "--arms", nargs="+", default=["adv1e2", "adv3e2", "adv1e1", "adv3e1"]
    )
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
    baseline_static = {
        **full_metrics(load(args.baseline_full)),
        **text_metrics(load(args.baseline_text)),
    }
    baseline_rollout = {
        seed: rollout_metrics(args.baseline_rollout_root, "baseline", seed)
        for seed in seeds
    }
    active_static = static_metrics(args.active_root, args.active_arm, args.active_step)
    active_rollout = {
        seed: checkpoint_rollout_metrics(
            args.active_root, args.active_arm, args.active_step, seed
        )
        for seed in seeds
    }

    rows = []
    missing = []
    for step in args.steps:
        for arm in args.arms:
            configs = sorted((args.root / arm).glob("resume_config_step-*.json"))
            legacy_configs = sorted(
                (args.legacy_root / arm).glob("resume_config_step-*.json")
            )
            if not configs or not legacy_configs:
                missing.append(f"{arm}: matched or legacy resume config")
                continue
            config = load(configs[0])
            legacy_config = load(legacy_configs[0])
            try:
                metrics = static_metrics(args.root, arm, step)
                legacy_static = static_metrics(args.legacy_root, arm, step)
                legacy_rollout = {
                    seed: checkpoint_rollout_metrics(
                        args.legacy_root, arm, step, seed
                    )
                    for seed in seeds
                }
                rollout_stage1 = paired_rollout(
                    args.root, arm, step, baseline_rollout, seeds
                )
                rollout_legacy = paired_rollout(
                    args.root, arm, step, legacy_rollout, seeds
                )
                rollout_active = paired_rollout(
                    args.root, arm, step, active_rollout, seeds
                )
            except FileNotFoundError as error:
                missing.append(str(error))
                continue

            rows.append(
                {
                    "arm": arm,
                    "step": step,
                    "adversarial_weight": float(config["adversarial_weight"]),
                    "adversarial_time_sampler": config[
                        "adversarial_time_sampler"
                    ],
                    "legacy_time_sampler": legacy_config[
                        "adversarial_time_sampler"
                    ],
                    "metrics": metrics,
                    "delta_percent_vs_stage1": {
                        name: relative_percent(value, baseline_static[name])
                        for name, value in metrics.items()
                    },
                    "delta_percent_vs_same_weight_legacy_time": {
                        name: relative_percent(value, legacy_static[name])
                        for name, value in metrics.items()
                    },
                    "delta_percent_vs_active_frontend": {
                        name: relative_percent(value, active_static[name])
                        for name, value in metrics.items()
                    },
                    "rollout_vs_stage1": rollout_stage1,
                    "rollout_vs_same_weight_legacy_time": rollout_legacy,
                    "rollout_vs_active_frontend": rollout_active,
                }
            )

    if not rows:
        raise RuntimeError(f"no fully evaluated rows under {args.root}")
    result = {
        "schema": "ardy_dmd_ladd_matched_informative_time_v1",
        "root": str(args.root),
        "legacy_root": str(args.legacy_root),
        "baseline_static": baseline_static,
        "active_frontend": {
            "root": str(args.active_root),
            "arm": args.active_arm,
            "step": args.active_step,
            "metrics": active_static,
        },
        "seeds": seeds,
        "lower_is_better": list(LOWER_IS_BETTER),
        "rollout_lower_is_better": list(ROLLOUT_LOWER_IS_BETTER),
        "rows": rows,
        "missing_or_pending": missing,
        "note": (
            "Matched arms differ from their same-weight legacy controls only in "
            "generator adversarial-time sampler: critic uses informative uniform "
            "[0,0.5] with no exact-t1 mass; score is the legacy high-noise mixture. "
            "No weighted quality score is used."
        ),
    }
    output = args.output or args.root / "matched_time_summary.json"
    output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    print(
        "arm\tstep\tendpoint%\twaypoint%\tFK%\tfoot%\tseam%\tjerk%\t"
        "bodyHF%\theadingHF%"
    )
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
