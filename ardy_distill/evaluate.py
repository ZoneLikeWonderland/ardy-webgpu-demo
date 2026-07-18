#!/usr/bin/env python
"""Fixed-dataset codec, low-step flow, motion, and path metrics."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .codec_cli import add_codec_config_arguments, codec_config_from_args
from .data import SHARD_FIELDS, TeacherShardDataset
from .losses import (
    FSQRequantizer,
    decoder_feature_losses,
    deployment_decoder_roots,
    encoder_distillation_losses,
    fsq_endpoint_diagnostics,
    motion_quality_losses,
    physical_seam_losses,
    root_temporal_losses,
)
from .models import (
    FlowStudentConfig,
    HistoryEncoderStudent,
    MotionDecoderStudent,
    OneStepFlowStudent,
)
from .runtime import load_motion_rep, load_safetensor_weights
from .text_features import PromptFeatureTable
from .train_flow import endpoint_losses


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--encoder", type=Path, required=True)
    parser.add_argument("--decoder", type=Path, required=True)
    parser.add_argument(
        "--flow",
        type=Path,
        help="Optional flow weights. Omit for an encoder/decoder-only validation pass.",
    )
    parser.add_argument(
        "--text-features",
        type=Path,
        help="Complete Qwen prompt-feature table used by a text-conditioned flow.",
    )
    parser.add_argument(
        "--heading-condition-features",
        type=int,
        choices=[0, 3],
        default=0,
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path("ardy/checkpoints/ARDY-Core-RP-20FPS-Horizon40"),
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-batches", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--model-dtype",
        choices=["fp32", "fp16", "bf16"],
        default="fp32",
        help="Arithmetic dtype inside student modules; metrics are accumulated in FP32.",
    )
    parser.add_argument(
        "--flow-history",
        choices=["student", "teacher"],
        default="student",
        help="History body source for flow evaluation; student matches browser deployment.",
    )
    parser.add_argument("--flow-width", type=int, default=384)
    parser.add_argument("--flow-heads", type=int, default=6)
    parser.add_argument("--flow-trunk-blocks", type=int, default=4)
    parser.add_argument("--flow-body-blocks", type=int, default=2)
    parser.add_argument("--flow-steps", type=int, default=1)
    parser.add_argument("--flow-root-smoothing-passes", type=int, default=0)
    parser.add_argument(
        "--flow-root-projection-kind",
        choices=["binomial", "cubic_controls"],
        default="binomial",
    )
    parser.add_argument("--flow-root-control-points", type=int, default=10)
    add_codec_config_arguments(parser)
    parser.add_argument(
        "--student-history-quantization",
        choices=["fsq", "none"],
        default="fsq",
        help="FSQ matches released _encode_init_history; none exists only to audit older flow runs.",
    )
    return parser.parse_args()


def root_xz_to_meters(normalized_xz: torch.Tensor, motion_rep) -> torch.Tensor:
    indices = torch.tensor([0, 2], device=normalized_xz.device)
    mean = motion_rep.global_root_stats.mean.to(normalized_xz.device)[indices]
    scale = motion_rep.global_root_stats.std_eps.to(normalized_xz.device)[indices]
    return normalized_xz.float() * scale + mean


def path_metrics(
    predicted_generation: torch.Tensor,
    teacher_generation: torch.Tensor,
    path_condition: torch.Tensor,
    has_history: torch.Tensor,
    motion_rep,
) -> dict[str, tuple[float, int]]:
    pred_root = predicted_generation[..., :20].reshape(-1, 40, 5)
    teacher_root = teacher_generation[..., :20].reshape(-1, 40, 5)
    totals = defaultdict(float)
    counts = defaultdict(int)
    for batch_index in range(pred_root.shape[0]):
        generation_start = 4 if has_history[batch_index, 0] > 0.5 else 0
        valid_frames = torch.nonzero(path_condition[batch_index, :, 2] > 0.5).flatten()
        for frame_tensor in valid_frames:
            frame = int(frame_tensor)
            target_m = root_xz_to_meters(path_condition[batch_index, frame, :2], motion_rep)
            if generation_start <= frame < generation_start + 40:
                generated_index = frame - generation_start
                pred_m = root_xz_to_meters(pred_root[batch_index, generated_index, [0, 2]], motion_rep)
                teacher_m = root_xz_to_meters(teacher_root[batch_index, generated_index, [0, 2]], motion_rep)
                totals["path_error_m"] += float(torch.linalg.vector_norm(pred_m - target_m))
                totals["teacher_path_error_m"] += float(torch.linalg.vector_norm(teacher_m - target_m))
                counts["path_error_m"] += 1
                counts["teacher_path_error_m"] += 1
            elif frame >= generation_start + 40:
                pred_m = root_xz_to_meters(pred_root[batch_index, -1, [0, 2]], motion_rep)
                teacher_m = root_xz_to_meters(teacher_root[batch_index, -1, [0, 2]], motion_rep)
                totals["future_waypoint_remaining_m"] += float(torch.linalg.vector_norm(pred_m - target_m))
                totals["teacher_future_waypoint_remaining_m"] += float(torch.linalg.vector_norm(teacher_m - target_m))
                counts["future_waypoint_remaining_m"] += 1
                counts["teacher_future_waypoint_remaining_m"] += 1
    return {name: (total, counts[name]) for name, total in totals.items()}


def main() -> None:
    args = parse_args()
    if args.flow_root_smoothing_passes < 0:
        raise ValueError("flow root smoothing passes must be non-negative")
    if not 2 <= args.flow_root_control_points <= 40:
        raise ValueError("flow root control points must be within [2, 40]")
    if args.flow_steps < 1:
        raise ValueError("flow steps must be positive")
    device = torch.device(args.device)
    model_dtype = {
        "fp32": torch.float32,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }[args.model_dtype]
    codec_config = codec_config_from_args(args)
    text_feature_dim = 0
    if args.text_features is not None:
        text_manifest = json.loads(
            (args.text_features / "manifest.json").read_text(encoding="utf-8")
        )
        if text_manifest.get("encoder") != "qwen" or not text_manifest.get("complete"):
            raise ValueError("--text-features must be a complete Qwen feature table")
        text_feature_dim = int(text_manifest["feature_dim"])
    evaluation_fields = SHARD_FIELDS + (
        *(("prompt_id",) if args.text_features is not None else ()),
        *(("heading_condition",) if args.heading_condition_features else ()),
    )
    dataset = TeacherShardDataset(args.data, fields=evaluation_fields)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    encoder = HistoryEncoderStudent(codec_config).to(device=device, dtype=model_dtype).eval()
    decoder = MotionDecoderStudent(codec_config).to(device=device, dtype=model_dtype).eval()
    flow_config = FlowStudentConfig(
        width=args.flow_width,
        heads=args.flow_heads,
        trunk_blocks=args.flow_trunk_blocks,
        body_blocks=args.flow_body_blocks,
        text_feature_dim=text_feature_dim,
        heading_condition_features=args.heading_condition_features,
        root_smoothing_passes=args.flow_root_smoothing_passes,
        root_projection_kind=args.flow_root_projection_kind,
        root_control_points=args.flow_root_control_points,
    )
    flow = (
        OneStepFlowStudent(flow_config).to(device=device, dtype=model_dtype).eval()
        if args.flow is not None
        else None
    )
    load_safetensor_weights(encoder, args.encoder)
    load_safetensor_weights(decoder, args.decoder)
    if flow is not None:
        load_safetensor_weights(flow, args.flow)
    text_features = None
    if args.text_features is not None:
        if flow is None:
            raise ValueError("--text-features requires --flow")
        text_features = PromptFeatureTable(
            args.text_features,
            expected_encoder="qwen",
        ).to(device, dtype=model_dtype)
    quantizer = FSQRequantizer(args.checkpoint_dir / "stats/post_quantization").to(device)
    motion_rep = load_motion_rep(args.checkpoint_dir)

    sums = defaultdict(float)
    weights = defaultdict(float)
    sample_count = 0
    with torch.inference_mode():
        for batch_index, batch in enumerate(loader):
            if args.max_batches and batch_index >= args.max_batches:
                break
            batch = {
                name: (
                    value.to(device).float()
                    if value.is_floating_point()
                    else value.to(device)
                )
                for name, value in batch.items()
            }
            batch_size = batch["initial_noise"].shape[0]
            sample_count += batch_size

            encoder_prediction = encoder(
                batch["encoder_body"].to(dtype=model_dtype)
            ).float()
            encoder_results = encoder_distillation_losses(
                encoder_prediction,
                batch["history_hybrid"][..., 20:],
                batch["encoder_valid"],
                quantizer,
            )
            decoder_prediction = decoder(
                batch["decoder_latent"].to(dtype=model_dtype),
                batch["decoder_local_root"].to(dtype=model_dtype),
                batch["decoder_token_valid"].to(dtype=model_dtype),
            ).float()
            decoder_results = decoder_feature_losses(
                decoder_prediction,
                batch["target_body"],
                batch["decoder_token_valid"],
            )
            decoder_quality = motion_quality_losses(
                decoder_prediction,
                batch["target_body"],
                batch["decoder_global_root"],
                batch["decoder_token_valid"],
                motion_rep,
            )
            decoder_seam = physical_seam_losses(
                decoder_prediction,
                batch["target_body"],
                batch["decoder_global_root"],
                batch["decoder_global_root"],
                batch["has_history"],
                motion_rep,
            )

            groups = {
                "encoder": encoder_results,
                "decoder": decoder_results,
                "decoder_quality": decoder_quality,
                "decoder_seam": decoder_seam,
            }
            clean_prediction = None
            if flow is not None:
                if args.flow_history == "student":
                    history_body = encoder_prediction
                    if args.student_history_quantization == "fsq":
                        history_body = quantizer(history_body, ste=False)
                    history_valid = batch["has_history"].unsqueeze(-1)
                    history_for_flow = torch.cat(
                        [
                            batch["history_hybrid"][..., :20] * history_valid,
                            history_body * history_valid,
                        ],
                        dim=-1,
                    )
                else:
                    history_for_flow = batch["history_hybrid"]
                text_feature = (
                    None
                    if text_features is None
                    else text_features.lookup(batch["prompt_id"])
                )
                heading_condition = (
                    batch.get("heading_condition")
                    if args.heading_condition_features
                    else None
                )
                clean_prediction = flow.denoise_steps(
                    batch["initial_noise"].to(dtype=model_dtype),
                    history_for_flow.to(dtype=model_dtype),
                    batch["path_condition"].to(dtype=model_dtype),
                    batch["first_heading"].to(dtype=model_dtype),
                    batch["has_history"].to(dtype=model_dtype),
                    steps=args.flow_steps,
                    text_feature=(
                        None
                        if text_feature is None
                        else text_feature.to(dtype=model_dtype)
                    ),
                    heading_condition=(
                        None
                        if heading_condition is None
                        else heading_condition.to(dtype=model_dtype)
                    ),
                ).float()
                endpoint_results = endpoint_losses(clean_prediction, batch["clean_generation"])
                endpoint_results.update(
                    fsq_endpoint_diagnostics(
                        clean_prediction[..., 20:],
                        batch["clean_generation"][..., 20:],
                        quantizer,
                    )
                )
                endpoint_temporal = root_temporal_losses(
                    clean_prediction,
                    batch["clean_generation"],
                    motion_rep,
                )
                latent_sequence = torch.cat(
                    [
                        history_for_flow[..., 20:],
                        quantizer(clean_prediction[..., 20:], ste=False),
                    ],
                    dim=1,
                )
                predicted_root, predicted_decoder_local_root = deployment_decoder_roots(
                    clean_prediction,
                    history_for_flow,
                    batch["has_history"],
                    motion_rep,
                )
                flow_body = decoder(
                    latent_sequence.to(dtype=model_dtype),
                    predicted_decoder_local_root.to(dtype=model_dtype),
                    batch["decoder_token_valid"].to(dtype=model_dtype),
                ).float()
                flow_quality = motion_quality_losses(
                    flow_body,
                    batch["target_body"],
                    predicted_root,
                    batch["decoder_token_valid"],
                    motion_rep,
                    target_normalized_global_root=batch["decoder_global_root"],
                )
                flow_prefix = f"nfe{args.flow_steps}"
                groups[flow_prefix] = endpoint_results
                groups[flow_prefix + "_root_temporal"] = endpoint_temporal
                groups[flow_prefix + "_quality"] = flow_quality
            for prefix, metrics in groups.items():
                for name, value in metrics.items():
                    key = f"{prefix}/{name}"
                    sums[key] += float(value) * batch_size
                    weights[key] += batch_size
            if clean_prediction is not None:
                for name, (total, count) in path_metrics(
                    clean_prediction,
                    batch["clean_generation"],
                    batch["path_condition"],
                    batch["has_history"],
                    motion_rep,
                ).items():
                    key = f"path/{name}"
                    sums[key] += total
                    weights[key] += count

    metrics = {
        name: sums[name] / weights[name]
        for name in sorted(sums)
        if weights[name] > 0
    }
    result = {
        "schema": "ardy_distill_eval_v2",
        "data": str(args.data),
        "samples": sample_count,
        "batch_size": args.batch_size,
        "max_batches": args.max_batches,
        "weights": {
            "encoder": str(args.encoder),
            "decoder": str(args.decoder),
            **({"flow": str(args.flow)} if args.flow is not None else {}),
        },
        "flow_history": args.flow_history if args.flow is not None else None,
        "student_history_quantization": (
            args.student_history_quantization
            if args.flow is not None and args.flow_history == "student"
            else None
        ),
        "flow_config": asdict(flow_config) if args.flow is not None else None,
        "flow_steps": args.flow_steps if args.flow is not None else None,
        "text_features": str(args.text_features) if args.text_features is not None else None,
        "heading_condition_features": args.heading_condition_features,
        "model_dtype": args.model_dtype,
        "metrics": metrics,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
