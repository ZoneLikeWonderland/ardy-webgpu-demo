"""Summarize aligned versus legacy generator adversarial-time sampling."""

from __future__ import annotations

import argparse
import json
import statistics
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


ARMS = (
    "u05_legacy_advtime",
    "u05_aligned_advtime",
    "high50_legacy_advtime",
    "high50_aligned_advtime",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
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


def final_diagnostics(path: Path) -> dict:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    row = next(
        item
        for item in reversed(rows)
        if item.get("event") is None and int(item.get("step", -1)) == 400
    )
    keys = (
        "critic_total",
        "critic_real_logit",
        "critic_fake_logit",
        "generator_adversarial",
        "generator_grad_norm",
        "adversarial_time_mean",
        "adversarial_time_min",
        "adversarial_time_max",
        "adversarial_time_exact_t1_fraction",
    )
    return {key: float(row[key]) for key in keys}


def direct_pair(candidate: dict, control: dict) -> dict:
    candidate_seeds = candidate["rollout"]["per_seed"]
    control_seeds = control["rollout"]["per_seed"]
    metrics = {}
    for name in ROLLOUT_LOWER_IS_BETTER:
        values = [
            relative_percent(candidate_seeds[seed][name], control_seeds[seed][name])
            for seed in control_seeds
        ]
        metrics[name] = {
            "mean": statistics.fmean(values),
            "std_population": statistics.pstdev(values),
            "wins": sum(value < 0 for value in values),
            "per_seed": dict(zip(control_seeds, values)),
        }
    return metrics


def main() -> None:
    args = parse_args()
    seeds = list(dict.fromkeys(args.rollout_seeds))
    baseline = {
        **full_metrics(load(args.baseline_full)),
        **text_metrics(load(args.baseline_text)),
    }
    rows = []
    for arm in ARMS:
        root = args.root / arm
        config_path = root / "resume_config_step-0000300.json"
        metrics_log = root / "metrics.jsonl"
        full_path = root / "eval/step000400_ema_full_fp16.json"
        text_path = root / "eval/step000400_ema_text_control_fp16.json"
        required = [config_path, metrics_log, full_path, text_path]
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"missing evidence for {arm}: {missing}")
        config = load(config_path)
        metrics = {
            **full_metrics(load(full_path)),
            **text_metrics(load(text_path)),
        }
        rows.append(
            {
                "arm": arm,
                "adversarial_time_sampler": config["adversarial_time_sampler"],
                "critic_time_sampling": {
                    "exact_t1_probability": float(
                        config["critic_time_exact_t1_probability"]
                    ),
                    "high_noise_probability": float(
                        config["critic_time_high_noise_probability"]
                    ),
                    "upper_bound": float(config["critic_time_upper_bound"]),
                },
                "metrics": metrics,
                "delta_percent_vs_stage1": {
                    name: relative_percent(value, baseline[name])
                    for name, value in metrics.items()
                },
                "training": completed_training_metrics(metrics_log),
                "final_diagnostics": final_diagnostics(metrics_log),
                "rollout": paired_rollout_summary(
                    args.root,
                    arm,
                    args.baseline_rollout_root,
                    seeds,
                ),
            }
        )

    by_arm = {row["arm"]: row for row in rows}
    pairwise = {
        "u05_aligned_vs_legacy": direct_pair(
            by_arm["u05_aligned_advtime"], by_arm["u05_legacy_advtime"]
        ),
        "high50_aligned_vs_legacy": direct_pair(
            by_arm["high50_aligned_advtime"], by_arm["high50_legacy_advtime"]
        ),
    }
    result = {
        "schema": "ardy_dmd_ladd_adversarial_time_v1",
        "root": str(args.root),
        "baseline": baseline,
        "seeds": seeds,
        "lower_is_better": list(LOWER_IS_BETTER),
        "rollout_lower_is_better": list(ROLLOUT_LOWER_IS_BETTER),
        "arms": rows,
        "pairwise": pairwise,
        "note": (
            "Each aligned/control pair resumes the same exact g300 state and runs "
            "100 additional generator updates. Only generator adversarial-time "
            "sampling differs within a pair."
        ),
    }
    output = args.output or args.root / "adversarial_time_summary.json"
    output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    print("arm\tadv_t\tadv_exact\tendpoint%\twaypoint%\tFK%\tfoot%\tseam%\tjerk%\tbodyHF%")
    for row in rows:
        static = row["delta_percent_vs_stage1"]
        paired = row["rollout"]["paired_percent_vs_stage1"]
        diagnostic = row["final_diagnostics"]
        print(
            f"{row['arm']}\t{diagnostic['adversarial_time_mean']:.3f}\t"
            f"{diagnostic['adversarial_time_exact_t1_fraction']:.3f}\t"
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
