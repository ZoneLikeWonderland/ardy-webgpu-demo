#!/usr/bin/env python
"""Export deterministic random-weight student graphs for browser WebGPU timing.

These cases answer an architecture question before training: whether the chosen
operator mix and static shapes meet the latency budget. They are not quality
models and are deliberately labelled diagnostic-only in their manifests.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Callable

import numpy as np
import onnx
import onnxruntime as ort
import torch
from torch import nn


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TOY_ROOT = PROJECT_ROOT / "webgpu_toy"
sys.path.insert(0, str(PROJECT_ROOT))

from ardy_distill.models import (  # noqa: E402
    CodecStudentConfig,
    FlowStudentConfig,
    HistoryEncoderStudent,
    MotionDecoderStudent,
    OneStepFlowStudent,
)
from ardy_distill.runtime import load_safetensor_weights  # noqa: E402


class OneStepExportWrapper(nn.Module):
    def __init__(self, model: OneStepFlowStudent) -> None:
        super().__init__()
        self.model = model

    def forward(
        self,
        noise: torch.Tensor,
        history_hybrid: torch.Tensor,
        path_condition: torch.Tensor,
        first_heading: torch.Tensor,
        has_history: torch.Tensor,
    ) -> torch.Tensor:
        return self.model.denoise_once(
            noise,
            history_hybrid,
            path_condition,
            first_heading,
            has_history,
        )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def numeric_metrics(actual: np.ndarray, reference: np.ndarray) -> dict[str, float | None]:
    actual64 = actual.astype(np.float64)
    reference64 = reference.astype(np.float64)
    absolute = np.abs(actual64 - reference64)
    relative = absolute / np.maximum(np.abs(reference64), 1.0e-8)
    denom = np.linalg.norm(actual64.ravel()) * np.linalg.norm(reference64.ravel())
    cosine = None if denom == 0 else float(np.dot(actual64.ravel(), reference64.ravel()) / denom)
    return {
        "max_abs_error": float(absolute.max()),
        "mean_abs_error": float(absolute.mean()),
        "max_rel_error": float(relative.max()),
        "mean_rel_error": float(relative.mean()),
        "cosine_similarity": cosine,
    }


def tensor_spec(name: str, tensor: torch.Tensor, case_id: str) -> dict:
    filename = f"input_{name}.bin"
    array = tensor.detach().cpu().numpy()
    array.tofile(TOY_ROOT / "cases" / case_id / filename)
    return {
        "name": name,
        "dtype": str(array.dtype),
        "shape": list(array.shape),
        "url": f"cases/{case_id}/{filename}",
    }


def make_encoder(
    dtype: torch.dtype,
    config: CodecStudentConfig = CodecStudentConfig(),
) -> tuple[nn.Module, tuple[torch.Tensor, ...], list[str], str]:
    model = HistoryEncoderStudent(config).eval().to(dtype=dtype)
    inputs = (torch.randn(1, 4, 325, dtype=dtype) * 0.2,)
    return model, inputs, ["normalized_body"], "Compact 4-frame history encoder"


def make_decoder(
    dtype: torch.dtype,
    config: CodecStudentConfig = CodecStudentConfig(),
) -> tuple[nn.Module, tuple[torch.Tensor, ...], list[str], str]:
    model = MotionDecoderStudent(config).eval().to(dtype=dtype)
    latent = torch.randn(1, 11, 128, dtype=dtype) * 0.2
    local_root = torch.randn(1, 44, 4, dtype=dtype) * 0.2
    valid = torch.ones(1, 11, dtype=dtype)
    return model, (latent, local_root, valid), ["latent_tokens", "local_root", "token_valid"], "Compact 11-token motion decoder"


def make_flow(
    dtype: torch.dtype,
    config: FlowStudentConfig = FlowStudentConfig(),
) -> tuple[nn.Module, tuple[torch.Tensor, ...], list[str], str]:
    model = OneStepExportWrapper(OneStepFlowStudent(config)).eval().to(dtype=dtype)
    noise = torch.randn(1, 10, 148, dtype=dtype)
    history = torch.randn(1, 1, 148, dtype=dtype) * 0.2
    path = torch.zeros(1, 64, 3, dtype=dtype)
    path[:, [20, 40, 60], :2] = torch.tensor(
        [[[0.25, -0.1], [0.8, 0.2], [1.2, 0.5]]],
        dtype=dtype,
    )
    path[:, [20, 40, 60], 2] = 1
    heading = torch.tensor([[0.995, 0.1]], dtype=dtype)
    has_history = torch.ones(1, 1, dtype=dtype)
    return (
        model,
        (noise, history, path, heading, has_history),
        ["noise", "history_hybrid", "path_condition", "first_heading", "has_history"],
        "NFE=1 shared-trunk flow student",
    )


BUILDERS: dict[str, Callable] = {
    "encoder": make_encoder,
    "flow": make_flow,
    "decoder": make_decoder,
}


def export_case(
    module_name: str,
    precision: str,
    *,
    weights: Path | None = None,
    case_tag: str | None = None,
    flow_config: FlowStudentConfig | None = None,
    codec_config: CodecStudentConfig | None = None,
) -> dict:
    dtype = torch.float16 if precision == "fp16" else torch.float32
    tag = f"_{case_tag}" if case_tag else ""
    case_id = f"ardy_student{tag}_{module_name}_{precision}"
    case_dir = TOY_ROOT / "cases" / case_id
    case_dir.mkdir(parents=True, exist_ok=True)

    if module_name == "flow" and flow_config is not None:
        model, inputs, input_names, description = make_flow(dtype, flow_config)
    elif module_name == "encoder" and codec_config is not None:
        model, inputs, input_names, description = make_encoder(dtype, codec_config)
    elif module_name == "decoder" and codec_config is not None:
        model, inputs, input_names, description = make_decoder(dtype, codec_config)
    else:
        model, inputs, input_names, description = BUILDERS[module_name](dtype)
    if weights is not None:
        target = model.model if isinstance(model, OneStepExportWrapper) else model
        load_safetensor_weights(target, weights)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    with torch.inference_mode():
        reference = model(*inputs).contiguous()
    if not torch.isfinite(reference).all():
        raise RuntimeError(f"{case_id}: random-weight reference contains NaN/Inf")

    model_path = case_dir / "model.onnx"
    torch.onnx.export(
        model,
        inputs,
        model_path,
        input_names=input_names,
        output_names=["output"],
        opset_version=17,
        do_constant_folding=True,
        dynamo=False,
    )
    graph = onnx.load(model_path)
    onnx.checker.check_model(graph)

    input_specs = [tensor_spec(name, tensor, case_id) for name, tensor in zip(input_names, inputs, strict=True)]
    reference_np = reference.detach().cpu().numpy()
    reference_np.tofile(case_dir / "reference_output.bin")

    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    feeds = {name: tensor.detach().cpu().numpy() for name, tensor in zip(input_names, inputs, strict=True)}
    ort_output = session.run(["output"], feeds)[0]
    native_metrics = numeric_metrics(ort_output, reference_np)

    if precision == "fp16":
        tolerances = {
            "max_abs_error": 0.08,
            "mean_abs_error": 0.008,
            "min_cosine_similarity": 0.999,
        }
    else:
        tolerances = {
            "max_abs_error": 0.003,
            "mean_abs_error": 0.0003,
            "min_cosine_similarity": 0.99999,
        }

    manifest = {
        "id": case_id,
        "title": (
            f"ARDY student {module_name} {precision.upper()} / trained weights"
            if weights is not None
            else f"ARDY student {module_name} {precision.upper()} / random weights"
        ),
        "description": (
            f"{description}; trained-weight latency/parity case from {weights}."
            if weights is not None
            else f"{description}; architecture-only latency/parity case, not a trained quality model."
        ),
        "model_url": f"cases/{case_id}/model.onnx",
        "model_sha256": sha256(model_path),
        "model_size_bytes": model_path.stat().st_size,
        "parameter_count": parameter_count,
        "opset": 17,
        "operators": sorted({node.op_type for node in graph.graph.node}),
        "inputs": input_specs,
        "outputs": [
            {
                "name": "output",
                "dtype": str(reference_np.dtype),
                "shape": list(reference_np.shape),
                "reference_url": f"cases/{case_id}/reference_output.bin",
                "compare_url": f"/api/compare/{case_id}/output",
                "tolerances": tolerances,
            }
        ],
        "tolerances": tolerances,
        "benchmark": {"warmup_runs": 3, "timed_runs": 20},
        "native_ort_cpu_vs_pytorch": native_metrics,
        "diagnostic_only": weights is None,
        "trained": weights is not None,
        "weights_path": str(weights) if weights is not None else None,
        "nfe": 1 if module_name == "flow" else None,
        "flow_config": asdict(flow_config) if module_name == "flow" and flow_config is not None else None,
        "codec_config": (
            asdict(codec_config)
            if module_name in {"encoder", "decoder"} and codec_config is not None
            else None
        ),
    }
    (case_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def update_index(manifests: list[dict]) -> None:
    index_path = TOY_ROOT / "cases" / "index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    ids = {manifest["id"] for manifest in manifests}
    index["cases"] = [case for case in index["cases"] if case["id"] not in ids]
    index["cases"].extend(
        {
            "id": manifest["id"],
            "title": manifest["title"],
            "manifest_url": f"cases/{manifest['id']}/manifest.json",
        }
        for manifest in manifests
    )
    index_path.write_text(json.dumps(index, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--modules", nargs="+", choices=sorted(BUILDERS), default=sorted(BUILDERS))
    parser.add_argument("--precisions", nargs="+", choices=["fp32", "fp16"], default=["fp32", "fp16"])
    parser.add_argument("--encoder-weights", type=Path)
    parser.add_argument("--flow-weights", type=Path)
    parser.add_argument("--decoder-weights", type=Path)
    parser.add_argument(
        "--case-tag",
        help="Optional alphanumeric/underscore tag inserted into case ids; required when any weights are supplied.",
    )
    parser.add_argument("--flow-width", type=int, default=384)
    parser.add_argument("--flow-heads", type=int, default=6)
    parser.add_argument("--flow-trunk-blocks", type=int, default=4)
    parser.add_argument("--flow-body-blocks", type=int, default=2)
    parser.add_argument("--encoder-width", type=int, default=512)
    parser.add_argument("--encoder-blocks", type=int, default=3)
    parser.add_argument("--decoder-width", type=int, default=512)
    parser.add_argument("--decoder-blocks", type=int, default=4)
    parser.add_argument("--decoder-token-hidden", type=int, default=32)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    weights = {
        "encoder": args.encoder_weights,
        "flow": args.flow_weights,
        "decoder": args.decoder_weights,
    }
    flow_config = FlowStudentConfig(
        width=args.flow_width,
        heads=args.flow_heads,
        trunk_blocks=args.flow_trunk_blocks,
        body_blocks=args.flow_body_blocks,
    )
    codec_config = CodecStudentConfig(
        encoder_width=args.encoder_width,
        encoder_blocks=args.encoder_blocks,
        decoder_width=args.decoder_width,
        decoder_blocks=args.decoder_blocks,
        decoder_token_hidden=args.decoder_token_hidden,
    )
    if args.case_tag and any(
        character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_"
        for character in args.case_tag
    ):
        raise ValueError("--case-tag may contain only letters, digits, and underscores")
    if any(path is not None for path in weights.values()):
        if not args.case_tag:
            raise ValueError("--case-tag is required when exporting trained weights")
        missing = [module for module in args.modules if weights[module] is None]
        if missing:
            raise ValueError(f"missing weights for selected modules: {', '.join(missing)}")
    torch.manual_seed(20260714)
    np.random.seed(20260714)
    torch.set_grad_enabled(False)
    manifests = [
        export_case(
            module_name,
            precision,
            weights=weights[module_name],
            case_tag=args.case_tag,
            flow_config=flow_config,
            codec_config=codec_config,
        )
        for module_name in args.modules
        for precision in args.precisions
    ]
    update_index(manifests)
    for manifest in manifests:
        native = manifest["native_ort_cpu_vs_pytorch"]
        print(
            f"{manifest['id']}: params={manifest['parameter_count']:,} "
            f"onnx={manifest['model_size_bytes']:,} B "
            f"cpu_max_abs={native['max_abs_error']:.6g}"
        )


if __name__ == "__main__":
    main()
