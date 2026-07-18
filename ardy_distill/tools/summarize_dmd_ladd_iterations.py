"""Summarize one continued DMD/LADD generator-iteration trajectory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .summarize_dmd_ladd_sweep import (
    ROLLOUT_LOWER_IS_BETTER,
    full_metrics,
    load,
    relative_percent,
    rollout_metrics,
    rollout_summary,
    text_metrics,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--round1-root", type=Path, required=True)
    parser.add_argument("--iteration-root", type=Path, required=True)
    parser.add_argument("--baseline-full", type=Path, required=True)
    parser.add_argument("--baseline-text", type=Path, required=True)
    parser.add_argument("--steps", type=int, nargs="+", default=[300, 500, 700, 1000])
    parser.add_argument(
        "--rollout-seeds",
        type=int,
        nargs="+",
        default=[20260714, 20260715, 20260716, 20260717],
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def static_metrics(full: Path, text: Path) -> dict[str, float]:
    return {**full_metrics(load(full)), **text_metrics(load(text))}


def main() -> None:
    args = parse_args()
    seeds = list(dict.fromkeys(args.rollout_seeds))
    baseline_static = static_metrics(args.baseline_full, args.baseline_text)
    baseline_by_seed = {
        seed: rollout_metrics(args.round1_root, "baseline", seed) for seed in seeds
    }

    points: list[dict] = []
    specifications = [
        (
            200,
            args.round1_root / "g1e7/eval/step000200_ema_full_fp16.json",
            args.round1_root / "g1e7/eval/step000200_ema_text_control_fp16.json",
            args.round1_root,
            "g1e7",
        ),
        *[
            (
                step,
                args.iteration_root / f"eval/step{step}_ema_full_fp16.json",
                args.iteration_root / f"eval/step{step}_ema_text_control_fp16.json",
                args.iteration_root,
                f"g{step}",
            )
            for step in args.steps
        ],
    ]
    for step, full_path, text_path, rollout_root, rollout_tag in specifications:
        static = static_metrics(full_path, text_path)
        points.append(
            {
                "generator_updates": step,
                "static": static,
                "static_delta_percent_vs_stage1": {
                    name: relative_percent(value, baseline_static[name])
                    for name, value in static.items()
                },
                "rollout": rollout_summary(
                    rollout_root, rollout_tag, seeds, baseline_by_seed
                ),
            }
        )

    result = {
        "schema": "ardy_dmd_ladd_iteration_sweep_v1",
        "generator_learning_rate": 1.0e-7,
        "baseline": {
            "full": str(args.baseline_full),
            "text": str(args.baseline_text),
            "static": baseline_static,
        },
        "rollout": {
            "seeds": seeds,
            "windows": 50,
            "model_dtype": "fp16",
            "lower_is_better": list(ROLLOUT_LOWER_IS_BETTER),
        },
        "points": points,
        "note": (
            "All points are one continued trajectory from the exact g200 state; "
            "no optimizer, EMA, fake-score, critic, sampler, or RNG reset occurs."
        ),
    }
    output = args.output or args.iteration_root / "iteration_summary.json"
    output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    print("step\tstatic_endpoint%\twaypoint%\tFK%\tfoot%\tseam%\tjerk%\tbody_HF%")
    for point in points:
        paired = point["rollout"]["paired_percent_vs_stage1"]
        print(
            f"{point['generator_updates']}\t"
            f"{point['static_delta_percent_vs_stage1']['endpoint_total']:+.3f}\t"
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
