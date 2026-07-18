#!/usr/bin/env python
"""Aggregate independently validated teacher ranks without rescanning shards."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("reports", nargs="+", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--expected-prompts", type=int, default=8192)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    reports = [json.loads(path.read_text(encoding="utf-8")) for path in args.reports]
    for path, report in zip(args.reports, reports, strict=True):
        if report.get("schema") != "ardy_teacher_validation_v1":
            raise ValueError(f"not a teacher validation report: {path}")
        if not report.get("valid") or not report.get("hashes_checked"):
            raise ValueError(f"report did not pass full validation: {path}")

    total = sum(int(report["count"]) for report in reports)
    if total <= 0:
        raise ValueError("aggregate count must be positive")

    histograms: dict[str, Counter[int]] = {}
    for name in (
        "prompt_histogram",
        "control_mode_histogram",
        "rollout_depth_histogram",
    ):
        histogram: Counter[int] = Counter()
        for report in reports:
            histogram.update(
                {int(key): int(value) for key, value in report[name].items()}
            )
        histograms[name] = histogram

    prompt_histogram = histograms["prompt_histogram"]
    missing_prompts = [
        prompt_id
        for prompt_id in range(args.expected_prompts)
        if prompt_histogram[prompt_id] == 0
    ]
    generated_counts = [
        prompt_histogram[prompt_id]
        for prompt_id in range(11, args.expected_prompts)
        if prompt_histogram[prompt_id] > 0
    ]

    combined_stats = {}
    for name in reports[0]["stats"]:
        mean = sum(
            int(report["count"]) * float(report["stats"][name]["mean"])
            for report in reports
        ) / total
        second_moment = sum(
            int(report["count"])
            * (
                float(report["stats"][name]["std"]) ** 2
                + float(report["stats"][name]["mean"]) ** 2
            )
            for report in reports
        ) / total
        combined_stats[name] = {
            "mean": mean,
            "std": max(0.0, second_moment - mean * mean) ** 0.5,
        }

    def weighted_fraction(name: str) -> float:
        return sum(
            int(report["count"]) * float(report[name]) for report in reports
        ) / total

    result = {
        "schema": "ardy_teacher_validation_aggregate_v1",
        "valid": True,
        "reports": [str(path) for path in args.reports],
        "manifests": sum(int(report["manifests"]) for report in reports),
        "shards": sum(int(report["shards"]) for report in reports),
        "count": total,
        "bytes": sum(int(report["bytes"]) for report in reports),
        "storage_dtypes": sorted(
            {dtype for report in reports for dtype in report["storage_dtypes"]}
        ),
        "hashes_checked": True,
        "history_fraction": weighted_fraction("history_fraction"),
        "constrained_fraction": weighted_fraction("constrained_fraction"),
        "prompt_switch_fraction": weighted_fraction("prompt_switch_fraction"),
        "prompt_coverage": {
            "expected": args.expected_prompts,
            "observed": len(prompt_histogram),
            "missing": missing_prompts,
            "unconditional_fraction": prompt_histogram[0] / total,
            "official_fraction": sum(prompt_histogram[index] for index in range(1, 11))
            / total,
            "generated_min_count": min(generated_counts) if generated_counts else 0,
            "generated_max_count": max(generated_counts) if generated_counts else 0,
        },
        "control_mode_histogram": dict(sorted(histograms["control_mode_histogram"].items())),
        "rollout_depth_histogram": dict(sorted(histograms["rollout_depth_histogram"].items())),
        "stats": combined_stats,
    }
    encoded = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")


if __name__ == "__main__":
    main()
