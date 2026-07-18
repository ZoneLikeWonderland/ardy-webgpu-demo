"""Summarize the corrected-u05 independent fake-score/critic ratio matrix."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .summarize_dmd_ladd_lr_iters import checkpoint_rollout_metrics
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
        "--arms",
        nargs="+",
        default=["g1_s1_d1", "g1_s2_d1", "g1_s1_d2", "g1_s2_d2"],
    )
    parser.add_argument("--control-arm", default="g1_s1_d1")
    parser.add_argument("--steps", type=int, nargs="+", default=[350, 400, 450, 500])
    parser.add_argument(
        "--rollout-seeds",
        type=int,
        nargs="+",
        default=[20260714, 20260715, 20260716, 20260717],
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def paired_rollout(
    root: Path,
    arm: str,
    step: int,
    reference: dict[int, dict],
    seeds: list[int],
) -> dict:
    per_seed = {
        seed: checkpoint_rollout_metrics(root, arm, step, seed) for seed in seeds
    }
    return {
        "all_finite": all(bool(per_seed[seed]["all_finite"]) for seed in seeds),
        "per_seed": {str(seed): per_seed[seed] for seed in seeds},
        "paired_percent_vs_reference": {
            name: aggregate(
                [
                    relative_percent(
                        float(per_seed[seed][name]),
                        float(reference[seed][name]),
                    )
                    for seed in seeds
                ]
            )
            for name in ROLLOUT_LOWER_IS_BETTER
        },
        "paired_wins_vs_reference": {
            name: sum(
                float(per_seed[seed][name]) < float(reference[seed][name])
                for seed in seeds
            )
            for name in ROLLOUT_LOWER_IS_BETTER
        },
    }


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

    rows = []
    missing = []
    for step in args.steps:
        try:
            control_rollout = {
                seed: checkpoint_rollout_metrics(
                    args.root, args.control_arm, step, seed
                )
                for seed in seeds
            }
            control_static_root = args.root / args.control_arm / "eval"
            control_static = {
                **full_metrics(
                    load(control_static_root / f"step{step:06d}_ema_full_fp16.json")
                ),
                **text_metrics(
                    load(
                        control_static_root
                        / f"step{step:06d}_ema_text_control_fp16.json"
                    )
                ),
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
            full_path = arm_root / "eval" / f"step{step:06d}_ema_full_fp16.json"
            text_path = (
                arm_root / "eval" / f"step{step:06d}_ema_text_control_fp16.json"
            )
            if not full_path.is_file() or not text_path.is_file():
                missing.append(f"{arm}: static/text step {step}")
                continue
            try:
                rollout_stage1 = paired_rollout(
                    args.root, arm, step, baseline_rollout, seeds
                )
                rollout_control = paired_rollout(
                    args.root, arm, step, control_rollout, seeds
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
                    "update_ratio": {
                        "generator": 1,
                        "fake_score": int(config["score_updates_per_generator"]),
                        "critic": int(config["critic_updates_per_generator"]),
                    },
                    "learning_rates": {
                        "generator": float(config["generator_learning_rate"]),
                        "fake_score": float(config["score_learning_rate"]),
                        "critic": float(config["critic_learning_rate"]),
                    },
                    "metrics": metrics,
                    "delta_percent_vs_stage1": {
                        name: relative_percent(value, baseline_static[name])
                        for name, value in metrics.items()
                    },
                    "delta_percent_vs_same_step_control": {
                        name: relative_percent(value, control_static[name])
                        for name, value in metrics.items()
                    },
                    "rollout_vs_stage1": rollout_stage1,
                    "rollout_vs_same_step_control": rollout_control,
                }
            )

    if not rows:
        raise RuntimeError(f"no fully evaluated ratio rows under {args.root}")
    result = {
        "schema": "ardy_dmd_ladd_independent_update_ratio_v1",
        "root": str(args.root),
        "control_arm": args.control_arm,
        "baseline_static": baseline_static,
        "seeds": seeds,
        "lower_is_better": list(LOWER_IS_BETTER),
        "rollout_lower_is_better": list(ROLLOUT_LOWER_IS_BETTER),
        "rows": rows,
        "missing_or_pending": missing,
        "note": (
            "All arms resume an identical full state. The ratio origin is "
            "anchored at the loaded counters after any common warmup, so no "
            "branch performs retroactive catch-up. The named control supplies "
            "the same-step reference; byte-level reproduction is audited by "
            "the experiment launcher. No weighted aggregate score is used."
        ),
    }
    output = args.output or args.root / "independent_ratio_summary.json"
    output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    print(
        "arm\tstep\tendpoint%\twaypoint%\tFK%\tfoot%\tseam%\tjerk%\tbodyHF%"
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
            f"{rollout['body_high_frequency_ratio']['mean']:+.3f}"
        )
    print(f"pending={len(missing)} wrote {output}")


if __name__ == "__main__":
    main()
