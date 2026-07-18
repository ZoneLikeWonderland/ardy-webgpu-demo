"""Summarize non-invasive DMD/LADD generator endpoint-gradient diagnostics."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path


COMPONENTS = ("dmd", "adversarial", "paired_fsq", "control_physics")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def summary(values: list[float]) -> dict[str, float | int]:
    if not values or not all(math.isfinite(value) for value in values):
        raise ValueError("gradient diagnostics must be non-empty and finite")
    return {
        "count": len(values),
        "mean": statistics.fmean(values),
        "std_population": statistics.pstdev(values),
        "min": min(values),
        "max": max(values),
    }


def main() -> None:
    args = parse_args()
    metrics_path = args.root / "metrics.jsonl"
    rows = [
        json.loads(line)
        for line in metrics_path.read_text(encoding="utf-8").splitlines()
        if "generator_component_total_endpoint_grad_rms" in line
    ]
    if not rows:
        raise RuntimeError(f"no component-gradient rows in {metrics_path}")

    result: dict[str, object] = {
        "schema": "ardy_dmd_ladd_endpoint_gradient_groups_v1",
        "root": str(args.root),
        "steps": [int(row["step"]) for row in rows],
        "components": {},
        "pairwise_raw_endpoint_cosine": {},
        "total_endpoint_gradient_rms": summary(
            [row["generator_component_total_endpoint_grad_rms"] for row in rows]
        ),
        "component_sum_relative_reconstruction_error": summary(
            [
                row[
                    "generator_component_sum_vs_total_endpoint_grad_relative_rms"
                ]
                for row in rows
            ]
        ),
        "generator_preclip_parameter_gradient_norm": summary(
            [row["generator_grad_norm"] for row in rows]
        ),
        "effective_global_clip_scale": summary(
            [min(1.0, 0.01 / row["generator_grad_norm"]) for row in rows]
        ),
        "note": (
            "Four weighted scalar groups reconstruct the actual BF16 total endpoint "
            "gradient. Diagnostics use autograd.grad with retain_graph and leave the "
            "normal distributed backward/optimizer trajectory unchanged. Ratios are "
            "endpoint-space measurements, not parameter-space norms."
        ),
    }
    components = result["components"]
    assert isinstance(components, dict)
    for component in COMPONENTS:
        prefix = f"generator_component_{component}_"
        components[component] = {
            "raw_endpoint_gradient_rms": summary(
                [row[prefix + "endpoint_grad_rms"] for row in rows]
            ),
            "weighted_endpoint_gradient_rms": summary(
                [row[prefix + "weighted_endpoint_grad_rms"] for row in rows]
            ),
            "weighted_to_total_rms_ratio": summary(
                [row[prefix + "weighted_to_total_rms_ratio"] for row in rows]
            ),
            "weighted_vs_total_cosine": summary(
                [
                    row[prefix + "weighted_vs_total_endpoint_cosine"]
                    for row in rows
                ]
            ),
        }

    pairwise = result["pairwise_raw_endpoint_cosine"]
    assert isinstance(pairwise, dict)
    for left_index, left in enumerate(COMPONENTS):
        for right in COMPONENTS[left_index + 1 :]:
            name = f"{left}_vs_{right}"
            key = f"generator_component_{name}_endpoint_cosine"
            pairwise[name] = summary([row[key] for row in rows])

    output = args.output or args.root / "gradient_group_summary.json"
    output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"rows={len(rows)} wrote {output}")


if __name__ == "__main__":
    main()
