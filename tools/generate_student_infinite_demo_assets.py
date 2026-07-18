#!/usr/bin/env python
"""Export a selected all-student ARDY infinite-demo release.

The browser keeps learned compute in three FP16 WebGPU graphs:

* four-frame compact history encoder;
* one-evaluation rectified-flow endpoint predictor;
* fixed eleven-token compact motion decoder.

Two small FP32 WASM graphs retain the released ARDY root-coordinate and FSQ
semantics.  Initial and continuation finalization are separate static graphs so
the dummy leading decoder token can never be confused with a real history
token.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
from torch import nn


TOY_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = TOY_ROOT.parent
DEMO_ROOT = TOY_ROOT / "infinite_demo"
CHECKPOINT_ROOT = WORKSPACE / "ardy" / "checkpoints" / "ARDY-Core-RP-20FPS-Horizon40"

if str(WORKSPACE) not in sys.path:
    sys.path.insert(0, str(WORKSPACE))
if str(WORKSPACE / "ardy") not in sys.path:
    sys.path.insert(0, str(WORKSPACE / "ardy"))

from ardy.model.ardy_model import translate_normalized_root_motion  # noqa: E402
from ardy_distill.losses import FSQRequantizer  # noqa: E402
from ardy_distill.models import (  # noqa: E402
    CodecStudentConfig,
    FlowStudentConfig,
    HistoryEncoderStudent,
    MotionDecoderStudent,
    OneStepFlowStudent,
)
from ardy_distill.runtime import load_motion_rep, load_safetensor_weights  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--encoder", type=Path, required=True)
    parser.add_argument("--flow", type=Path, required=True)
    parser.add_argument("--decoder", type=Path, required=True)
    parser.add_argument("--model-release", required=True)
    parser.add_argument("--manifest", type=Path, default=DEMO_ROOT / "manifest.json")
    parser.add_argument("--precision", choices=["fp16", "fp32"], default="fp16")
    parser.add_argument("--method", default="flow_dmd2_adversarial")
    parser.add_argument("--generator-step", type=int, default=0)
    parser.add_argument("--encoder-width", type=int, default=512)
    parser.add_argument("--encoder-blocks", type=int, default=4)
    parser.add_argument("--decoder-width", type=int, default=512)
    parser.add_argument("--decoder-blocks", type=int, default=8)
    parser.add_argument("--decoder-token-hidden", type=int, default=32)
    parser.add_argument("--codec-expansion", type=int, default=2)
    parser.add_argument("--flow-width", type=int, default=512)
    parser.add_argument("--flow-heads", type=int, default=8)
    parser.add_argument("--flow-trunk-blocks", type=int, default=8)
    parser.add_argument("--flow-body-blocks", type=int, default=8)
    parser.add_argument("--flow-expansion", type=int, default=2)
    parser.add_argument("--text-feature-dim", type=int, default=0)
    parser.add_argument(
        "--heading-condition-features",
        type=int,
        choices=[0, 3],
        default=0,
    )
    parser.add_argument(
        "--budget-bytes",
        type=int,
        default=100_000_000,
        help="Configurable approximate release budget; text pilots may use a modestly larger cap.",
    )
    parser.add_argument("--flow-root-smoothing-passes", type=int, default=0)
    parser.add_argument(
        "--flow-root-projection-kind",
        choices=["binomial", "cubic_controls"],
        default="binomial",
    )
    parser.add_argument("--flow-root-control-points", type=int, default=10)
    return parser.parse_args()


def configs_from_args(
    args: argparse.Namespace,
) -> tuple[CodecStudentConfig, FlowStudentConfig]:
    structural_values = (
        args.encoder_width,
        args.encoder_blocks,
        args.decoder_width,
        args.decoder_blocks,
        args.decoder_token_hidden,
        args.codec_expansion,
        args.flow_width,
        args.flow_heads,
        args.flow_trunk_blocks,
        args.flow_body_blocks,
        args.flow_expansion,
    )
    if min(structural_values) < 1:
        raise ValueError("all widths, head counts, block counts and expansions must be positive")
    if args.flow_width % args.flow_heads:
        raise ValueError("flow width must be divisible by head count")
    if args.flow_root_smoothing_passes < 0:
        raise ValueError("root smoothing passes must be non-negative")
    codec = CodecStudentConfig(
        encoder_width=args.encoder_width,
        encoder_blocks=args.encoder_blocks,
        decoder_width=args.decoder_width,
        decoder_blocks=args.decoder_blocks,
        decoder_token_hidden=args.decoder_token_hidden,
        expansion=args.codec_expansion,
    )
    flow = FlowStudentConfig(
        width=args.flow_width,
        heads=args.flow_heads,
        trunk_blocks=args.flow_trunk_blocks,
        body_blocks=args.flow_body_blocks,
        expansion=args.flow_expansion,
        text_feature_dim=args.text_feature_dim,
        heading_condition_features=args.heading_condition_features,
        root_smoothing_passes=args.flow_root_smoothing_passes,
        root_projection_kind=args.flow_root_projection_kind,
        root_control_points=args.flow_root_control_points,
    )
    return codec, flow


class OneStepExportWrapper(nn.Module):
    """Expose the production ``noise -> clean endpoint`` operation."""

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


class ConditionedOneStepExportWrapper(nn.Module):
    """Expose text and future-heading inputs without embedding either encoder."""

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
        text_feature: torch.Tensor,
        heading_condition: torch.Tensor,
    ) -> torch.Tensor:
        return self.model.denoise_once(
            noise,
            history_hybrid,
            path_condition,
            first_heading,
            has_history,
            text_feature,
            heading_condition,
        )


class InitialFinalize(nn.Module):
    """Prepare the dummy+generation decoder layout for a history-free window."""

    def __init__(self, motion_rep, quantizer: FSQRequantizer) -> None:
        super().__init__()
        self.motion_rep = motion_rep
        self.quantizer = quantizer

    def forward(
        self,
        clean_generation: torch.Tensor,
        global_translation: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch = clean_generation.shape[0]
        local_root = clean_generation[..., :20].reshape(batch, 40, 5)
        latent = self.quantizer(clean_generation[..., 20:], ste=False)
        global_root = translate_normalized_root_motion(
            local_root,
            global_translation,
            self.motion_rep,
        )
        lengths = torch.full((batch,), 40, dtype=torch.long, device=clean_generation.device)
        decoder_local_actual = self.motion_rep.global_root_to_local_root(
            global_root,
            normalized=True,
            lengths=lengths,
        )
        decoder_latent = torch.cat([latent.new_zeros(batch, 1, 128), latent], dim=1)
        decoder_local_root = torch.cat(
            [decoder_local_actual.new_zeros(batch, 4, 4), decoder_local_actual],
            dim=1,
        )
        token_valid = torch.cat(
            [latent.new_zeros(batch, 1), latent.new_ones(batch, 10)],
            dim=1,
        )
        return global_root, decoder_latent, decoder_local_root, token_valid


class ContinuationFinalize(nn.Module):
    """Restore world roots and prepare all eleven real decoder tokens."""

    def __init__(self, motion_rep, quantizer: FSQRequantizer) -> None:
        super().__init__()
        self.motion_rep = motion_rep
        self.quantizer = quantizer

    def forward(
        self,
        local_hybrid: torch.Tensor,
        global_translation: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch = local_hybrid.shape[0]
        local_root = local_hybrid[..., :20].reshape(batch, 44, 5)
        latent = self.quantizer(local_hybrid[..., 20:], ste=False)
        global_root = translate_normalized_root_motion(
            local_root,
            global_translation,
            self.motion_rep,
        )
        lengths = torch.full((batch,), 44, dtype=torch.long, device=local_hybrid.device)
        decoder_local_root = self.motion_rep.global_root_to_local_root(
            global_root,
            normalized=True,
            lengths=lengths,
        )
        token_valid = latent.new_ones(batch, 11)
        return global_root, latent, decoder_local_root, token_valid


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def numeric_metrics(actual: np.ndarray, reference: np.ndarray) -> dict[str, float]:
    delta = np.abs(actual.astype(np.float64) - reference.astype(np.float64))
    return {
        "max_abs_error": float(delta.max(initial=0.0)),
        "mean_abs_error": float(delta.mean()) if delta.size else 0.0,
    }


def cast_float_args(
    args: tuple[torch.Tensor, ...],
    dtype: torch.dtype,
) -> tuple[torch.Tensor, ...]:
    return tuple(
        tensor.to(dtype=dtype) if tensor.is_floating_point() else tensor
        for tensor in args
    )


def export_module(
    module: nn.Module,
    args: tuple[torch.Tensor, ...],
    path: Path,
    input_names: list[str],
    output_names: list[str],
    *,
    dtype: torch.dtype,
) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    module = module.float().eval()
    fp32_args = cast_float_args(args, torch.float32)
    with torch.inference_mode():
        fp32_raw = module(*fp32_args)
    fp32_references = fp32_raw if isinstance(fp32_raw, tuple) else (fp32_raw,)
    module = module.to(dtype=dtype).eval()
    export_args = cast_float_args(args, dtype)
    with torch.inference_mode():
        native_raw = module(*export_args)
    native_references = native_raw if isinstance(native_raw, tuple) else (native_raw,)
    torch.onnx.export(
        module,
        export_args,
        path,
        input_names=input_names,
        output_names=output_names,
        opset_version=17,
        do_constant_folding=True,
        dynamo=False,
    )
    graph = onnx.load(path)
    onnx.checker.check_model(graph)
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    feeds = {
        name: tensor.detach().cpu().numpy()
        for name, tensor in zip(input_names, export_args, strict=True)
    }
    actuals = session.run(output_names, feeds)
    native_parity = {
        name: numeric_metrics(actual, reference.detach().float().cpu().numpy())
        for name, actual, reference in zip(
            output_names,
            actuals,
            native_references,
            strict=True,
        )
    }
    precision_parity = {
        name: numeric_metrics(actual, reference.detach().cpu().numpy())
        for name, actual, reference in zip(
            output_names,
            actuals,
            fp32_references,
            strict=True,
        )
    }
    if not all(np.isfinite(actual).all() for actual in actuals):
        raise RuntimeError(f"{path}: ORT output contains NaN/Inf")
    return {
        "url": str(path.relative_to(TOY_ROOT)).replace("\\", "/"),
        "size_bytes": path.stat().st_size,
        "sha256": sha256(path),
        "parameter_count": sum(parameter.numel() for parameter in module.parameters()),
        "inputs": [item.name for item in graph.graph.input],
        "outputs": [item.name for item in graph.graph.output],
        "operators": sorted({node.op_type for node in graph.graph.node}),
        "native_ort_cpu_vs_pytorch_same_precision": native_parity,
        "exported_precision_vs_pytorch_fp32": precision_parity,
    }


def display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(WORKSPACE.resolve()))
    except ValueError:
        return str(path.resolve())


def main() -> None:
    args = parse_args()
    codec_config, flow_config = configs_from_args(args)
    dtype = torch.float16 if args.precision == "fp16" else torch.float32
    models_root = DEMO_ROOT / "models" / "student" / args.model_release
    torch.manual_seed(20260714)
    np.random.seed(20260714)
    torch.set_grad_enabled(False)

    for path in (args.encoder, args.flow, args.decoder):
        if not path.is_file():
            raise FileNotFoundError(path)

    motion_rep = load_motion_rep(CHECKPOINT_ROOT)
    quantizer = FSQRequantizer(CHECKPOINT_ROOT / "stats/post_quantization").float().eval()
    encoder = HistoryEncoderStudent(codec_config).float().eval()
    flow = OneStepFlowStudent(flow_config).float().eval()
    decoder = MotionDecoderStudent(codec_config).float().eval()
    load_safetensor_weights(encoder, args.encoder)
    load_safetensor_weights(flow, args.flow)
    load_safetensor_weights(decoder, args.decoder)

    normalized_body = torch.randn(1, 4, 325) * 0.2
    noise = torch.randn(1, 10, 148)
    history_hybrid = torch.randn(1, 1, 148) * 0.2
    path_condition = torch.zeros(1, 64, 3)
    path_condition[:, [20, 40, 60], :2] = torch.tensor(
        [[[0.25, -0.1], [0.8, 0.2], [1.2, 0.5]]]
    )
    path_condition[:, [20, 40, 60], 2] = 1
    first_heading = torch.tensor([[0.995, 0.1]])
    has_history = torch.ones(1, 1)
    text_feature = torch.randn(1, max(1, flow_config.text_feature_dim))
    heading_condition = torch.zeros(
        1,
        flow_config.path_frames,
        max(1, flow_config.heading_condition_features),
    )
    if flow_config.heading_condition_features:
        heading_condition[:, :, 0] = 1
        heading_condition[:, :, 2] = 1
    local_generation = torch.randn(1, 10, 148) * 0.2
    local_hybrid = torch.randn(1, 11, 148) * 0.2
    translation = torch.tensor([[1.25, 0.0, -0.75]])
    latent = torch.randn(1, 11, 128) * 0.2
    local_root = torch.randn(1, 44, 4) * 0.2
    token_valid = torch.ones(1, 11)

    conditioned_flow = bool(
        flow_config.text_feature_dim or flow_config.heading_condition_features
    )
    flow_export_module: nn.Module
    flow_export_args: tuple[torch.Tensor, ...]
    flow_input_names: list[str]
    if conditioned_flow:
        if not flow_config.text_feature_dim or not flow_config.heading_condition_features:
            raise ValueError(
                "the browser conditioned-flow export currently requires both text and heading"
            )
        flow_export_module = ConditionedOneStepExportWrapper(flow)
        flow_export_args = (
            noise,
            history_hybrid,
            path_condition,
            first_heading,
            has_history,
            text_feature[:, : flow_config.text_feature_dim],
            heading_condition[:, :, : flow_config.heading_condition_features],
        )
        flow_input_names = [
            "noise",
            "history_hybrid",
            "path_condition",
            "first_heading",
            "has_history",
            "text_feature",
            "heading_condition",
        ]
    else:
        flow_export_module = OneStepExportWrapper(flow)
        flow_export_args = (
            noise,
            history_hybrid,
            path_condition,
            first_heading,
            has_history,
        )
        flow_input_names = [
            "noise",
            "history_hybrid",
            "path_condition",
            "first_heading",
            "has_history",
        ]

    models = {
        "encoder": export_module(
            encoder,
            (normalized_body,),
            models_root / args.precision / "encoder.onnx",
            ["normalized_body"],
            ["output"],
            dtype=dtype,
        ),
        "flow": export_module(
            flow_export_module,
            flow_export_args,
            models_root / args.precision / "flow.onnx",
            flow_input_names,
            ["output"],
            dtype=dtype,
        ),
        "decoder": export_module(
            decoder,
            (latent, local_root, token_valid),
            models_root / args.precision / "decoder.onnx",
            ["latent_tokens", "local_root", "token_valid"],
            ["output"],
            dtype=dtype,
        ),
    }
    utilities = {
        "finalize_initial": export_module(
            InitialFinalize(motion_rep, quantizer),
            (local_generation, translation),
            models_root / "shared/finalize_initial.onnx",
            ["clean_generation", "global_translation"],
            ["global_root", "decoder_latent", "decoder_local_root", "token_valid"],
            dtype=torch.float32,
        ),
        "finalize_continuation": export_module(
            ContinuationFinalize(motion_rep, quantizer),
            (local_hybrid, translation),
            models_root / "shared/finalize_continuation.onnx",
            ["local_hybrid", "global_translation"],
            ["global_root", "decoder_latent", "decoder_local_root", "token_valid"],
            dtype=torch.float32,
        ),
    }

    skeleton = motion_rep.skeleton
    learned_bytes = sum(spec["size_bytes"] for spec in models.values())
    utility_bytes = sum(spec["size_bytes"] for spec in utilities.values())
    total_bytes = learned_bytes + utility_bytes
    if args.budget_bytes < 1:
        raise ValueError("--budget-bytes must be positive")
    if learned_bytes >= args.budget_bytes:
        raise RuntimeError(
            f"learned ONNX graphs exceed the configured budget: "
            f"{learned_bytes:,} >= {args.budget_bytes:,} bytes"
        )
    if total_bytes >= args.budget_bytes:
        raise RuntimeError(
            f"complete demo ONNX download exceeds the configured budget: "
            f"{total_bytes:,} >= {args.budget_bytes:,} bytes"
        )
    manifest = {
        "schema_version": 2,
        "model": f"ARDY-Student-{args.model_release}-{args.precision.upper()}",
        "model_release": args.model_release,
        "all_student": True,
        "release_status": (
            "experimental_pilot"
            if "pilot" in args.model_release.lower() or "pilot" in args.method.lower()
            else "candidate"
        ),
        "text_encoder_loaded": False,
        "text_conditioning_enabled": bool(flow_config.text_feature_dim),
        "text_conditioning": {
            "encoder": "external_FLUX.2_Qwen3_precomputed",
            "feature_dim": flow_config.text_feature_dim,
            "inside_motion_weight_budget": False,
            "compression": "none",
            "prompt_bundle_url": "infinite_demo/prompt_features.json",
        },
        "heading_conditioning_enabled": bool(flow_config.heading_condition_features),
        "constraint_conditioning_enabled": True,
        "precision": args.precision,
        "nfe": 1,
        "fps": float(motion_rep.fps),
        "frames_per_token": 4,
        "history_frames": 4,
        "generation_frames": 40,
        "decode_frames": 44,
        "path_frames": 64,
        "waypoint_interval_frames": 60,
        "motion_dim": int(motion_rep.motion_rep_dim),
        "body_dim": 325,
        "hybrid_dim": 148,
        "root_dim": int(motion_rep.motion_root_dim),
        "latent_dim": 128,
        "inertialization": {
            "frames": 0,
            "strength": 0.0,
            "excluded_tail_features": 4,
            "mode": "disabled_model_graph_projection_only",
        },
        "distillation": {
            "deployment_nfe": 1,
            "training_nfe": 1,
            "method": args.method,
            "generator_step": args.generator_step,
            "runtime_cfg": False,
            "root_projection": {
                "kind": flow_config.root_projection_kind,
                "passes": flow_config.root_smoothing_passes,
                "kernel": [1.0 / 16.0, 4.0 / 16.0, 6.0 / 16.0, 4.0 / 16.0, 1.0 / 16.0],
                "inside_exported_flow_graph": True,
            },
        },
        "root_stats": {
            "mean": motion_rep.global_root_stats.mean.detach().cpu().tolist(),
            "std_eps": motion_rep.global_root_stats.std_eps.detach().cpu().tolist(),
        },
        "post_quantization_stats": {
            "mean": quantizer.mean.detach().cpu().tolist(),
            "std_eps": quantizer.std_eps.detach().cpu().tolist(),
            "levels": 64,
        },
        "skeleton": {
            "joint_names": list(skeleton.bone_order_names),
            "parents": skeleton.joint_parents.detach().cpu().tolist(),
            "root_index": int(skeleton.root_idx),
        },
        "model_config": {
            "codec": asdict(codec_config),
            "flow": asdict(flow_config),
        },
        "weights": {
            "encoder": display_path(args.encoder),
            "flow": display_path(args.flow),
            "decoder": display_path(args.decoder),
        },
        "models": models,
        "utilities": utilities,
        "download": {
            "learned_graph_bytes": learned_bytes,
            "utility_graph_bytes": utility_bytes,
            "total_onnx_bytes": total_bytes,
            "budget_bytes": args.budget_bytes,
            "under_budget": True,
        },
    }
    manifest_path = args.manifest
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {manifest_path}")
    for name, spec in {**models, **utilities}.items():
        worst = max(
            metrics["max_abs_error"]
            for metrics in spec["exported_precision_vs_pytorch_fp32"].values()
        )
        print(
            f"{name}: params={spec['parameter_count']:,} "
            f"onnx={spec['size_bytes']:,} B cpu_max_abs={worst:.6g}"
        )
    print(
        f"download: learned={learned_bytes:,} B utilities={utility_bytes:,} B "
        f"total={total_bytes:,} B"
    )


if __name__ == "__main__":
    main()
