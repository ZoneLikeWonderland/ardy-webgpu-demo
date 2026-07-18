#!/usr/bin/env python
"""Calibrate the training-only FP16 simulator against the real Edge probe."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

from ardy_distill.models import (
    CodecStudentConfig,
    FlowStudentConfig,
    HistoryEncoderStudent,
    MotionDecoderStudent,
    OneStepFlowStudent,
)
from ardy_distill.runtime import load_safetensor_weights
from ardy_distill.webgpu_numerics import (
    CapturePoint,
    EDGE_149_AMPERE_FP16_PROFILES,
    simulate_webgpu_fp16,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUN = PROJECT_ROOT / "distill_runs" / "first_12h_20260714_014643"
DEFAULT_CODEC = (
    DEFAULT_RUN
    / "codec_standard_e4_d8_4gpu_b256_step5e5_to1e5_100k"
    / "weights"
    / "step-0100000"
)
DEFAULT_FLOW = (
    DEFAULT_RUN
    / "flow_standard48m_dmd2_adv001_anchor01_onpolicy20kema_replay50_4gpu_b32_g1000_const_1e6_5e6_1e6_ema995_20260715"
    / "weights"
    / "step-0001000"
    / "flow_ema.safetensors"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--probe-case",
        type=Path,
        default=PROJECT_ROOT / "webgpu_toy/infinite_demo/webgpu_precision_probe_case.json",
    )
    parser.add_argument(
        "--edge-result",
        type=Path,
        default=DEFAULT_RUN / "eval/edge_webgpu_precision_probe_blockwise.json",
    )
    parser.add_argument("--encoder", type=Path, default=DEFAULT_CODEC / "encoder_ema.safetensors")
    parser.add_argument("--decoder", type=Path, default=DEFAULT_CODEC / "decoder_ema.safetensors")
    parser.add_argument("--flow", type=Path, default=DEFAULT_FLOW)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_RUN / "eval/webgpu_fp16_simulator_calibration.json",
    )
    parser.add_argument("--device", default="cuda:6")
    parser.add_argument("--repeats", type=int, default=8)
    parser.add_argument(
        "--noise-scales",
        type=float,
        nargs="+",
        default=[0.0, 0.25, 0.5, 0.75, 1.0, 1.25],
    )
    parser.add_argument("--seed", type=int, default=20260715)
    return parser.parse_args()


def tensor_from_spec(spec: dict, device: torch.device) -> torch.Tensor:
    return torch.tensor(spec["values"], device=device, dtype=torch.float32).reshape(spec["shape"])


def capture_points(module: str) -> list[CapturePoint]:
    if module == "encoder":
        rows = [CapturePoint("input_gelu", "blocks.0.norm", "pre")]
        rows.extend(CapturePoint(f"block_{index}", f"blocks.{index}") for index in range(4))
        rows.extend(
            [
                CapturePoint("output_norm", "output_norm"),
                CapturePoint("output", ""),
            ]
        )
        return rows
    if module == "flow":
        rows = [CapturePoint("input_norm", "input_norm")]
        for index in range(8):
            rows.extend(
                [
                    CapturePoint(
                        f"trunk_{index}_attention",
                        f"trunk.{index}.channel.norm",
                        "pre",
                    ),
                    CapturePoint(f"trunk_{index}_channel", f"trunk.{index}"),
                ]
            )
        rows.extend(
            [
                CapturePoint("root_velocity", "root_head"),
                CapturePoint("body_input", "body_refiner.0.attention_norm", "pre"),
            ]
        )
        for index in range(8):
            rows.extend(
                [
                    CapturePoint(
                        f"body_{index}_attention",
                        f"body_refiner.{index}.channel.norm",
                        "pre",
                    ),
                    CapturePoint(f"body_{index}_channel", f"body_refiner.{index}"),
                ]
            )
        rows.extend(
            [
                CapturePoint("body_velocity", "body_head"),
                # The root module returns velocity.  It is converted to the
                # exported NFE=1 clean endpoint after the forward.
                CapturePoint("output", ""),
            ]
        )
        return rows
    if module == "decoder":
        rows = [CapturePoint("input_gelu_valid", "blocks.0.token_norm", "pre")]
        for index in range(8):
            rows.extend(
                [
                    CapturePoint(
                        f"block_{index}_token",
                        f"blocks.{index}.channel_block.norm",
                        "pre",
                    ),
                    CapturePoint(f"block_{index}_channel", f"blocks.{index}"),
                ]
            )
        rows.extend(
            [
                CapturePoint("output_norm", "output_norm"),
                CapturePoint("output", ""),
            ]
        )
        return rows
    raise ValueError(module)


def model_inputs(case: dict, device: torch.device) -> dict[str, torch.Tensor]:
    return {name: tensor_from_spec(spec, device) for name, spec in case["inputs"].items()}


def references(case: dict, device: torch.device) -> dict[str, torch.Tensor]:
    return {
        row["label"]: tensor_from_spec(row["reference"], device)
        for row in case["outputs"]
    }


def edge_targets(edge: dict) -> dict[str, dict[str, dict]]:
    return {
        module: {row["label"]: row for row in rows}
        for module, rows in edge["cases"].items()
    }


def cosine(left: torch.Tensor, right: torch.Tensor) -> float:
    left64 = left.reshape(-1).double()
    right64 = right.reshape(-1).double()
    denominator = float(torch.linalg.vector_norm(left64) * torch.linalg.vector_norm(right64))
    return float(torch.dot(left64, right64) / denominator) if denominator > 0.0 else 1.0


def metrics(actual: torch.Tensor, reference: torch.Tensor) -> dict[str, float | bool]:
    difference = (actual.float() - reference.float()).abs().reshape(-1)
    return {
        "all_finite": bool(torch.isfinite(actual).all()),
        "max_abs_error": float(difference.max()),
        "mean_abs_error": float(difference.mean()),
        "p95_abs_error": float(torch.quantile(difference, 0.95)),
        "p99_abs_error": float(torch.quantile(difference, 0.99)),
        "cosine_similarity": cosine(actual, reference),
    }


def instantiate_models(device: torch.device, args: argparse.Namespace) -> dict[str, torch.nn.Module]:
    codec_config = CodecStudentConfig(
        encoder_width=512,
        encoder_blocks=4,
        decoder_width=512,
        decoder_blocks=8,
        decoder_token_hidden=32,
        expansion=2,
    )
    flow_config = FlowStudentConfig(
        width=512,
        heads=8,
        trunk_blocks=8,
        body_blocks=8,
        expansion=2,
        root_smoothing_passes=0,
    )
    models: dict[str, torch.nn.Module] = {
        "encoder": HistoryEncoderStudent(codec_config).to(device).eval(),
        "flow": OneStepFlowStudent(flow_config).to(device).eval(),
        "decoder": MotionDecoderStudent(codec_config).to(device).eval(),
    }
    load_safetensor_weights(models["encoder"], args.encoder)
    load_safetensor_weights(models["flow"], args.flow)
    load_safetensor_weights(models["decoder"], args.decoder)
    return models


def run_module(
    name: str,
    model: torch.nn.Module,
    inputs: dict[str, torch.Tensor],
    *,
    noise_scale: float,
    generator: torch.Generator,
) -> dict[str, torch.Tensor]:
    if name == "encoder":
        positional = (inputs["normalized_body"],)
        keyword = {}
    elif name == "flow":
        positional = (
            inputs["noise"],
            inputs["history_hybrid"],
            inputs["path_condition"],
            inputs["first_heading"],
            inputs["has_history"],
            torch.ones_like(inputs["has_history"]),
        )
        keyword = {}
    elif name == "decoder":
        positional = (
            inputs["latent_tokens"],
            inputs["local_root"],
            inputs["token_valid"],
        )
        keyword = {}
    else:
        raise ValueError(name)
    _, captures = simulate_webgpu_fp16(
        model,
        *positional,
        profile=EDGE_149_AMPERE_FP16_PROFILES[name],
        severity=1.0,
        noise_scale=noise_scale,
        generator=generator,
        capture_points=capture_points(name),
        **keyword,
    )
    if name == "flow":
        captures["output"] = inputs["noise"].float() - captures["output"]
    return captures


def average_rows(rows: list[dict[str, float | bool]]) -> dict[str, float | bool]:
    numeric_keys = [key for key in rows[0] if key != "all_finite"]
    return {
        "all_finite": all(bool(row["all_finite"]) for row in rows),
        **{
            key: float(np.mean([float(row[key]) for row in rows]))
            for key in numeric_keys
        },
    }


def calibration_score(
    simulated: dict[str, dict[str, dict]],
    targets: dict[str, dict[str, dict]],
) -> float:
    log_errors = []
    for module, rows in simulated.items():
        for label, row in rows.items():
            if label not in targets[module]:
                continue
            measured = float(targets[module][label]["mean_abs_error"])
            candidate = float(row["mean_abs_error"])
            log_errors.append(abs(math.log((candidate + 1.0e-8) / (measured + 1.0e-8))))
    return float(np.mean(log_errors))


def main() -> None:
    args = parse_args()
    if args.repeats < 1 or not args.noise_scales:
        raise ValueError("repeats and noise scale list must be non-empty")
    if any(scale < 0.0 for scale in args.noise_scales):
        raise ValueError("noise scales must be non-negative")
    device = torch.device(args.device)
    probe = json.loads(args.probe_case.read_text(encoding="utf-8"))
    edge = json.loads(args.edge_result.read_text(encoding="utf-8"))
    targets = edge_targets(edge)
    models = instantiate_models(device, args)
    cases = probe["cases"]
    prepared_inputs = {name: model_inputs(cases[name], device) for name in models}
    prepared_references = {name: references(cases[name], device) for name in models}
    results: dict[str, dict] = {}

    with torch.inference_mode():
        for noise_scale in args.noise_scales:
            repeated: dict[str, dict[str, list[dict]]] = {
                name: {label: [] for label in targets[name]}
                for name in models
            }
            for repeat in range(args.repeats):
                generator = torch.Generator(device=device).manual_seed(
                    args.seed + repeat + int(round(noise_scale * 10_000))
                )
                for name, model in models.items():
                    captures = run_module(
                        name,
                        model,
                        prepared_inputs[name],
                        noise_scale=noise_scale,
                        generator=generator,
                    )
                    for label in repeated[name]:
                        repeated[name][label].append(
                            metrics(captures[label], prepared_references[name][label])
                        )
            averaged = {
                name: {
                    label: average_rows(rows)
                    for label, rows in module_rows.items()
                }
                for name, module_rows in repeated.items()
            }
            score = calibration_score(averaged, targets)
            results[f"{noise_scale:g}"] = {
                "noise_scale": noise_scale,
                "mean_absolute_log_ratio_score": score,
                "modules": averaged,
            }
            print(json.dumps({"noise_scale": noise_scale, "score": score}), flush=True)

    best = min(results.values(), key=lambda row: row["mean_absolute_log_ratio_score"])
    result = {
        "schema": "ardy_webgpu_fp16_simulator_calibration_v1",
        "probe_case": str(args.probe_case),
        "edge_result": str(args.edge_result),
        "device": str(device),
        "repeats": args.repeats,
        "weights": {
            "encoder": str(args.encoder),
            "flow": str(args.flow),
            "decoder": str(args.decoder),
        },
        "profile_sources": {
            name: profile.source
            for name, profile in EDGE_149_AMPERE_FP16_PROFILES.items()
        },
        "score_definition": (
            "mean absolute log ratio between simulated and real Edge mean-absolute "
            "error over the common blockwise observation points"
        ),
        "best_noise_scale": best["noise_scale"],
        "best_score": best["mean_absolute_log_ratio_score"],
        "real_edge_targets": targets,
        "candidates": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"best_noise_scale": result["best_noise_scale"], "output": str(args.output)}))


if __name__ == "__main__":
    main()

