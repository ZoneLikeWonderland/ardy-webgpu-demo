"""Summarize the controlled guidance/critic update-frequency experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .summarize_dmd_ladd_sweep import (
    LOWER_IS_BETTER,
    ROLLOUT_LOWER_IS_BETTER,
    aggregate,
    completed_training_metrics,
    full_metrics,
    load,
    relative_percent,
    rollout_metrics,
    text_metrics,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--ratio-one-root", type=Path, required=True)
    parser.add_argument("--baseline-rollout-root", type=Path, required=True)
    parser.add_argument("--baseline-full", type=Path, required=True)
    parser.add_argument("--baseline-text", type=Path, required=True)
    parser.add_argument(
        "--rollout-seeds",
        type=int,
        nargs="+",
        default=[20260714, 20260715, 20260716, 20260717],
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def paired_rollout_summary(
    candidate_root: Path,
    candidate_tag: str,
    baseline_root: Path,
    seeds: list[int],
) -> dict:
    per_seed = {
        seed: rollout_metrics(candidate_root, candidate_tag, seed) for seed in seeds
    }
    baseline = {
        seed: rollout_metrics(baseline_root, "baseline", seed) for seed in seeds
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

    specs = [
        {
            "arm": "g1_s1_c1",
            "ratio": 1,
            "root": args.ratio_one_root,
            "config": args.ratio_one_root / "resume_config_step-0000200.json",
            "metrics": args.ratio_one_root / "metrics.jsonl",
            "full": args.ratio_one_root / "eval/step300_ema_full_fp16.json",
            "text": args.ratio_one_root / "eval/step300_ema_text_control_fp16.json",
            "rollout_root": args.ratio_one_root,
            "rollout_tag": "g300",
        }
    ]
    for ratio in (2, 4):
        arm = f"g1_s{ratio}_c{ratio}"
        root = args.root / arm
        specs.append(
            {
                "arm": arm,
                "ratio": ratio,
                "root": root,
                "config": root / "resume_config_step-0000200.json",
                "metrics": root / "metrics.jsonl",
                "full": root / "eval/step000300_ema_full_fp16.json",
                "text": root / "eval/step000300_ema_text_control_fp16.json",
                "rollout_root": args.root,
                "rollout_tag": arm,
            }
        )

    rows = []
    for spec in specs:
        required = [spec["config"], spec["metrics"], spec["full"], spec["text"]]
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"missing evidence for {spec['arm']}: {missing}")
        config = load(spec["config"])
        metrics = {
            **full_metrics(load(spec["full"])),
            **text_metrics(load(spec["text"])),
        }
        rows.append(
            {
                "arm": spec["arm"],
                "generator_score_critic_ratio": [1, spec["ratio"], spec["ratio"]],
                "guidance_updates_per_generator": int(
                    config["guidance_updates_per_generator"]
                ),
                "generator_learning_rate": float(config["generator_learning_rate"]),
                "score_learning_rate": float(config["score_learning_rate"]),
                "critic_learning_rate": float(config["critic_learning_rate"]),
                "metrics": metrics,
                "delta_percent_vs_stage1": {
                    name: relative_percent(value, baseline[name])
                    for name, value in metrics.items()
                },
                "training": completed_training_metrics(spec["metrics"]),
                "rollout": paired_rollout_summary(
                    spec["rollout_root"],
                    spec["rollout_tag"],
                    args.baseline_rollout_root,
                    seeds,
                ),
            }
        )

    ratio_one = rows[0]["metrics"]
    for row in rows:
        row["delta_percent_vs_ratio_one"] = {
            name: relative_percent(value, ratio_one[name])
            for name, value in row["metrics"].items()
        }

    result = {
        "schema": "ardy_dmd_ladd_guidance_ratio_v1",
        "root": str(args.root),
        "ratio_one_root": str(args.ratio_one_root),
        "baseline": baseline,
        "seeds": seeds,
        "lower_is_better": list(LOWER_IS_BETTER),
        "rollout_lower_is_better": list(ROLLOUT_LOWER_IS_BETTER),
        "arms": rows,
        "note": (
            "All arms branch from the exact g200 state and end at generator step 300. "
            "Only the number of jointly scheduled fake-score/critic updates per new "
            "generator update changes; no weighted aggregate score is used."
        ),
    }
    output = args.output or args.root / "guidance_ratio_summary.json"
    output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    print("arm\tendpoint%\twaypoint%\tFK%\tfoot%\tseam%\tjerk%\tbodyHF%")
    for row in rows:
        static = row["delta_percent_vs_stage1"]
        paired = row["rollout"]["paired_percent_vs_stage1"]
        print(
            f"{row['arm']}\t{static['endpoint_total']:+.3f}\t"
            f"{paired['waypoint_error_m']['mean']:+.3f}\t"
            f"{paired['fk_mpjpe_m']['mean']:+.3f}\t"
            f"{paired['foot_slide_mps']['mean']:+.3f}\t"
            f"{paired['seam_joint_velocity_jump_mps']['mean']:+.3f}\t"
            f"{paired['body_jerk_p95']['mean']:+.3f}\t"
            f"{paired['body_high_frequency_ratio']['mean']:+.3f}"
        )
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
