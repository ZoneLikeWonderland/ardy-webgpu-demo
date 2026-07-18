"""Summarize the controlled adversarial-weight experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .summarize_dmd_ladd_guidance_ratio import paired_rollout_summary
from .summarize_dmd_ladd_sweep import (
    LOWER_IS_BETTER,
    ROLLOUT_LOWER_IS_BETTER,
    completed_training_metrics,
    full_metrics,
    load,
    relative_percent,
    text_metrics,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--current-root", type=Path, required=True)
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


def main() -> None:
    args = parse_args()
    seeds = list(dict.fromkeys(args.rollout_seeds))
    baseline = {
        **full_metrics(load(args.baseline_full)),
        **text_metrics(load(args.baseline_text)),
    }
    specs = [
        {
            "arm": "adv1e3",
            "root": args.current_root,
            "config": args.current_root / "resume_config_step-0000200.json",
            "metrics_log": args.current_root / "metrics.jsonl",
            "full": args.current_root / "eval/step300_ema_full_fp16.json",
            "text": args.current_root / "eval/step300_ema_text_control_fp16.json",
            "rollout_root": args.current_root,
            "rollout_tag": "g300",
        }
    ]
    for arm in ("adv0", "adv3e4", "adv3e3", "adv1e2"):
        root = args.root / arm
        specs.append(
            {
                "arm": arm,
                "root": root,
                "config": root / "resume_config_step-0000200.json",
                "metrics_log": root / "metrics.jsonl",
                "full": root / "eval/step000300_ema_full_fp16.json",
                "text": root / "eval/step000300_ema_text_control_fp16.json",
                "rollout_root": args.root,
                "rollout_tag": arm,
            }
        )

    rows = []
    for spec in specs:
        required = [
            spec["config"],
            spec["metrics_log"],
            spec["full"],
            spec["text"],
        ]
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
                "adversarial_weight": float(config["adversarial_weight"]),
                "metrics": metrics,
                "delta_percent_vs_stage1": {
                    name: relative_percent(value, baseline[name])
                    for name, value in metrics.items()
                },
                "training": completed_training_metrics(spec["metrics_log"]),
                "rollout": paired_rollout_summary(
                    spec["rollout_root"],
                    spec["rollout_tag"],
                    args.baseline_rollout_root,
                    seeds,
                ),
            }
        )

    current = rows[0]["metrics"]
    for row in rows:
        row["delta_percent_vs_adv1e3"] = {
            name: relative_percent(value, current[name])
            for name, value in row["metrics"].items()
        }

    result = {
        "schema": "ardy_dmd_ladd_adversarial_weight_v1",
        "root": str(args.root),
        "current_root": str(args.current_root),
        "baseline": baseline,
        "seeds": seeds,
        "lower_is_better": list(LOWER_IS_BETTER),
        "rollout_lower_is_better": list(ROLLOUT_LOWER_IS_BETTER),
        "arms": rows,
        "note": (
            "All arms branch from the exact g200 state and end at generator step 300. "
            "Only adversarial_weight changes; no weighted aggregate score is used."
        ),
    }
    output = args.output or args.root / "adversarial_weight_summary.json"
    output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    print("arm\tadv\tendpoint%\twaypoint%\tFK%\tfoot%\tseam%\tjerk%\tbodyHF%")
    for row in rows:
        static = row["delta_percent_vs_stage1"]
        paired = row["rollout"]["paired_percent_vs_stage1"]
        print(
            f"{row['arm']}\t{row['adversarial_weight']:.1e}\t"
            f"{static['endpoint_total']:+.3f}\t"
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
