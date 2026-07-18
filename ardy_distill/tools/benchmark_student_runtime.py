#!/usr/bin/env python
"""Benchmark trained student modules in the faithful one-window runtime."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from ardy_distill.codec_cli import add_codec_config_arguments, codec_config_from_args
from ardy_distill.data import TeacherShardDataset
from ardy_distill.losses import FSQRequantizer
from ardy_distill.models import (
    FlowStudentConfig,
    HistoryEncoderStudent,
    MotionDecoderStudent,
    OneStepFlowStudent,
)
from ardy_distill.runtime import load_motion_rep, load_safetensor_weights
from ardy_distill.student_runtime import StudentArdyRuntime


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--encoder", type=Path, required=True)
    parser.add_argument("--flow", type=Path, required=True)
    parser.add_argument("--decoder", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path("ardy/checkpoints/ARDY-Core-RP-20FPS-Horizon40"),
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--model-dtype", choices=["fp32", "fp16", "bf16"], default="fp32")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--flow-width", type=int, default=384)
    parser.add_argument("--flow-heads", type=int, default=6)
    parser.add_argument("--flow-trunk-blocks", type=int, default=4)
    parser.add_argument("--flow-body-blocks", type=int, default=2)
    add_codec_config_arguments(parser)
    return parser.parse_args()


def select_cases(dataset: TeacherShardDataset) -> dict[str, dict[str, torch.Tensor]]:
    selected: dict[str, dict[str, torch.Tensor]] = {}
    for index in range(len(dataset)):
        sample = dataset[index]
        mode = "continuation" if float(sample["has_history"].reshape(-1)[0]) > 0.5 else "initial"
        selected.setdefault(mode, sample)
        if len(selected) == 2:
            return selected
    raise RuntimeError("teacher dataset must contain both initial and continuation windows")


def percentile(values: list[float], quantile: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), quantile))


def summarize(timings: dict[str, list[float]]) -> dict[str, dict[str, float]]:
    return {
        name: {
            "mean_ms": float(np.mean(values)),
            "p50_ms": percentile(values, 50),
            "p95_ms": percentile(values, 95),
            "min_ms": float(np.min(values)),
            "max_ms": float(np.max(values)),
        }
        for name, values in sorted(timings.items())
    }


def run_case(
    runtime: StudentArdyRuntime,
    sample: dict[str, torch.Tensor],
    mode: str,
    *,
    warmup: int,
    repeats: int,
) -> dict:
    device = runtime.device
    tensors = {name: value.unsqueeze(0).to(device=device, dtype=torch.float32) for name, value in sample.items()}
    kwargs = {
        "path_condition": tensors["path_condition"],
        "initial_noise": tensors["initial_noise"],
    }
    if mode == "continuation":
        kwargs["init_history_sequence"] = runtime.motion_rep.concat_root_body(
            tensors["decoder_global_root"][:, :4],
            tensors["target_body"][:, :4],
        )
    else:
        kwargs["init_global_translation"] = tensors["global_translation"]
        kwargs["init_first_heading_angle"] = torch.atan2(
            tensors["first_heading"][:, 1],
            tensors["first_heading"][:, 0],
        )

    for _ in range(warmup):
        runtime.step_prepared(**kwargs)
    if device.type == "cuda":
        torch.cuda.synchronize(device)

    timings: dict[str, list[float]] = defaultdict(list)
    output = None
    for _ in range(repeats):
        output = runtime.step_prepared(**kwargs, profile=True)
        for name, value in output.timings_ms.items():
            timings[name].append(value)
    assert output is not None
    finite = bool(torch.isfinite(output.explicit_motion).all())
    if not finite:
        raise RuntimeError(f"{mode} output contains NaN/Inf")
    expected_frames = 44 if mode == "continuation" else 40
    if output.explicit_motion.shape != (1, expected_frames, runtime.motion_rep.motion_rep_dim):
        raise RuntimeError(
            f"unexpected {mode} output shape: {tuple(output.explicit_motion.shape)}"
        )
    return {
        "mode": mode,
        "output_shape": list(output.explicit_motion.shape),
        "finite": finite,
        "timings": summarize(timings),
    }


def main() -> None:
    args = parse_args()
    if args.warmup < 1 or args.repeats < 1:
        raise ValueError("warmup and repeats must be positive")
    device = torch.device(args.device)
    dtype = {
        "fp32": torch.float32,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }[args.model_dtype]
    flow_config = FlowStudentConfig(
        width=args.flow_width,
        heads=args.flow_heads,
        trunk_blocks=args.flow_trunk_blocks,
        body_blocks=args.flow_body_blocks,
    )
    codec_config = codec_config_from_args(args)
    encoder = HistoryEncoderStudent(codec_config).to(device=device, dtype=dtype).eval()
    flow = OneStepFlowStudent(flow_config).to(device=device, dtype=dtype).eval()
    decoder = MotionDecoderStudent(codec_config).to(device=device, dtype=dtype).eval()
    load_safetensor_weights(encoder, args.encoder)
    load_safetensor_weights(flow, args.flow)
    load_safetensor_weights(decoder, args.decoder)
    quantizer = FSQRequantizer(args.checkpoint_dir / "stats/post_quantization").to(device)
    motion_rep = load_motion_rep(args.checkpoint_dir)
    runtime = StudentArdyRuntime(
        encoder,
        flow,
        decoder,
        quantizer,
        motion_rep,
        model_dtype=dtype,
    ).eval()
    cases = select_cases(TeacherShardDataset(args.data, cache_shards=1))
    case_results = [
        run_case(runtime, cases[mode], mode, warmup=args.warmup, repeats=args.repeats)
        for mode in ("initial", "continuation")
    ]
    parameter_counts = {
        "encoder": sum(parameter.numel() for parameter in encoder.parameters()),
        "flow": sum(parameter.numel() for parameter in flow.parameters()),
        "decoder": sum(parameter.numel() for parameter in decoder.parameters()),
    }
    result = {
        "schema": "ardy_student_runtime_benchmark_v1",
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "model_dtype": args.model_dtype,
        "warmup": args.warmup,
        "repeats": args.repeats,
        "parameter_counts": {
            **parameter_counts,
            "total": sum(parameter_counts.values()),
        },
        "weights": {
            "encoder": str(args.encoder),
            "flow": str(args.flow),
            "decoder": str(args.decoder),
        },
        "flow_config": {
            "width": args.flow_width,
            "heads": args.flow_heads,
            "trunk_blocks": args.flow_trunk_blocks,
            "body_blocks": args.flow_body_blocks,
        },
        "codec_config": {
            "encoder_width": args.encoder_width,
            "encoder_blocks": args.encoder_blocks,
            "decoder_width": args.decoder_width,
            "decoder_blocks": args.decoder_blocks,
            "decoder_token_hidden": args.decoder_token_hidden,
            "expansion": args.codec_expansion,
        },
        "cases": case_results,
    }
    if any(
        not math.isfinite(metric)
        for case in case_results
        for stage in case["timings"].values()
        for metric in stage.values()
    ):
        raise RuntimeError("benchmark produced a non-finite timing")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
