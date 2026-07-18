"""Summarize the controlled critic flow-time sampling experiment."""

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


ARMS = ("legacy_t1p70", "uniform_u09", "high50_u09", "uniform_u05")
DIAGNOSTIC_KEYS = (
    "critic_total",
    "critic_real_logit",
    "critic_fake_logit",
    "critic_grad_norm",
    "critic_time_mean",
    "critic_time_min",
    "critic_time_max",
    "critic_time_exact_t1_fraction",
    "critic_feature_gap_l1",
    "critic_exact_t1_feature_gap_l1",
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


def selected_diagnostics(path: Path) -> dict:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    recovery_end = next(
        row
        for row in rows
        if row.get("event") == "guidance_warmup"
        and int(row.get("guidance_updates", -1)) == 600
    )
    final_update = next(
        row
        for row in reversed(rows)
        if row.get("event") is None and int(row.get("step", -1)) == 300
    )
    return {
        "recovery_end_guidance600": {
            key: float(recovery_end[key]) for key in DIAGNOSTIC_KEYS
        },
        "final_generator300": {
            key: float(final_update[key]) for key in DIAGNOSTIC_KEYS
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
    for arm in ARMS:
        root = args.root / arm
        config_path = root / "resume_config_step-0000200.json"
        metrics_log = root / "metrics.jsonl"
        full_path = root / "eval/step000300_ema_full_fp16.json"
        text_path = root / "eval/step000300_ema_text_control_fp16.json"
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
                "critic_diagnostics": selected_diagnostics(metrics_log),
                "rollout": paired_rollout_summary(
                    args.root,
                    arm,
                    args.baseline_rollout_root,
                    seeds,
                ),
            }
        )

    legacy = rows[0]["metrics"]
    for row in rows:
        row["delta_percent_vs_legacy"] = {
            name: relative_percent(value, legacy[name])
            for name, value in row["metrics"].items()
        }

    result = {
        "schema": "ardy_dmd_ladd_critic_time_v1",
        "root": str(args.root),
        "baseline": baseline,
        "seeds": seeds,
        "lower_is_better": list(LOWER_IS_BETTER),
        "rollout_lower_is_better": list(ROLLOUT_LOWER_IS_BETTER),
        "arms": rows,
        "note": (
            "All arms resume the exact g200 state, receive the same 200 additional "
            "score/critic-only recovery updates, then end at generator step 300. "
            "Only critic flow-time sampling changes; score/DMD time sampling remains "
            "70% exact t=1 plus 20% high-noise."
        ),
    }
    output = args.output or args.root / "critic_time_summary.json"
    output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    print(
        "arm\trecovery_D\trecovery_gap\tendpoint%\twaypoint%\tFK%\t"
        "foot%\tseam%\tjerk%\tbodyHF%"
    )
    for row in rows:
        static = row["delta_percent_vs_stage1"]
        paired = row["rollout"]["paired_percent_vs_stage1"]
        diagnostic = row["critic_diagnostics"]["recovery_end_guidance600"]
        print(
            f"{row['arm']}\t{diagnostic['critic_total']:.6f}\t"
            f"{diagnostic['critic_feature_gap_l1']:.6f}\t"
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
