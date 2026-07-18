#!/usr/bin/env python
"""Paired FP16-vs-FP32 parity audit, including deployment FSQ bin flips."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from ardy_distill.codec_cli import add_codec_config_arguments, codec_config_from_args
from ardy_distill.data import TeacherShardDataset
from ardy_distill.losses import (
    FSQRequantizer,
    deployment_decoder_roots,
    frame_valid_from_tokens,
)
from ardy_distill.models import (
    FlowStudentConfig,
    HistoryEncoderStudent,
    MotionDecoderStudent,
    OneStepFlowStudent,
)
from ardy_distill.runtime import load_motion_rep, load_safetensor_weights


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--encoder", type=Path, required=True)
    parser.add_argument("--decoder", type=Path, required=True)
    parser.add_argument("--flow", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path("ardy/checkpoints/ARDY-Core-RP-20FPS-Horizon40"),
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Deployment parity defaults to the browser's batch size of one.",
    )
    parser.add_argument("--max-batches", type=int, default=0)
    parser.add_argument("--flow-width", type=int, default=512)
    parser.add_argument("--flow-heads", type=int, default=8)
    parser.add_argument("--flow-trunk-blocks", type=int, default=8)
    parser.add_argument("--flow-body-blocks", type=int, default=8)
    parser.add_argument("--flow-steps", type=int, default=1)
    parser.add_argument("--flow-root-smoothing-passes", type=int, default=0)
    parser.add_argument(
        "--flow-root-projection-kind",
        choices=["binomial", "cubic_controls"],
        default="binomial",
    )
    parser.add_argument("--flow-root-control-points", type=int, default=10)
    add_codec_config_arguments(parser)
    return parser.parse_args()


class PairStats:
    def __init__(self) -> None:
        self.count = 0
        self.sum_abs = 0.0
        self.sum_sq = 0.0
        self.max_abs = 0.0
        self.dot = 0.0
        self.norm_a_sq = 0.0
        self.norm_b_sq = 0.0
        self.all_finite = True

    def update(
        self,
        fp16_value: torch.Tensor,
        fp32_value: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> None:
        a = fp16_value.detach().float()
        b = fp32_value.detach().float()
        if mask is not None:
            while mask.ndim < a.ndim:
                mask = mask.unsqueeze(-1)
            mask = mask.expand_as(a).bool()
            a = a[mask]
            b = b[mask]
        else:
            a = a.reshape(-1)
            b = b.reshape(-1)
        if a.numel() == 0:
            return
        delta = a - b
        self.count += a.numel()
        self.sum_abs += float(delta.abs().double().sum())
        self.sum_sq += float(delta.double().square().sum())
        self.max_abs = max(self.max_abs, float(delta.abs().max()))
        self.dot += float((a.double() * b.double()).sum())
        self.norm_a_sq += float(a.double().square().sum())
        self.norm_b_sq += float(b.double().square().sum())
        self.all_finite = self.all_finite and bool(
            torch.isfinite(a).all() and torch.isfinite(b).all()
        )

    def result(self) -> dict[str, float | int | bool | None]:
        denominator = (self.norm_a_sq * self.norm_b_sq) ** 0.5
        return {
            "elements": self.count,
            "max_abs": self.max_abs if self.count else None,
            "mean_abs": self.sum_abs / self.count if self.count else None,
            "rmse": (self.sum_sq / self.count) ** 0.5 if self.count else None,
            "cosine": self.dot / denominator if denominator > 0 else None,
            "all_finite": self.all_finite,
        }


class BinStats:
    def __init__(self) -> None:
        self.elements = 0
        self.mismatches = 0

    def update(
        self,
        fp16_quantized: torch.Tensor,
        fp32_quantized: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> None:
        mismatch = fp16_quantized != fp32_quantized
        if mask is not None:
            while mask.ndim < mismatch.ndim:
                mask = mask.unsqueeze(-1)
            mismatch = mismatch[mask.expand_as(mismatch).bool()]
        else:
            mismatch = mismatch.reshape(-1)
        self.elements += mismatch.numel()
        self.mismatches += int(mismatch.sum())

    def result(self) -> dict[str, float | int | None]:
        return {
            "elements": self.elements,
            "mismatches": self.mismatches,
            "mismatch_fraction": (
                self.mismatches / self.elements if self.elements else None
            ),
        }


def parameter_count(module: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters())


def main() -> None:
    args = parse_args()
    if args.batch_size < 1 or args.max_batches < 0 or args.flow_steps < 1:
        raise ValueError("batch size/flow steps must be positive and max batches non-negative")
    device = torch.device(args.device)
    codec_config = codec_config_from_args(args)
    flow_config = FlowStudentConfig(
        width=args.flow_width,
        heads=args.flow_heads,
        trunk_blocks=args.flow_trunk_blocks,
        body_blocks=args.flow_body_blocks,
        root_smoothing_passes=args.flow_root_smoothing_passes,
        root_projection_kind=args.flow_root_projection_kind,
        root_control_points=args.flow_root_control_points,
    )

    encoder32 = HistoryEncoderStudent(codec_config).to(device).eval()
    encoder16 = HistoryEncoderStudent(codec_config).to(device=device, dtype=torch.float16).eval()
    decoder32 = MotionDecoderStudent(codec_config).to(device).eval()
    decoder16 = MotionDecoderStudent(codec_config).to(device=device, dtype=torch.float16).eval()
    for module in (encoder32, encoder16):
        load_safetensor_weights(module, args.encoder)
    for module in (decoder32, decoder16):
        load_safetensor_weights(module, args.decoder)

    flow32 = flow16 = None
    if args.flow is not None:
        flow32 = OneStepFlowStudent(flow_config).to(device).eval()
        flow16 = OneStepFlowStudent(flow_config).to(device=device, dtype=torch.float16).eval()
        load_safetensor_weights(flow32, args.flow)
        load_safetensor_weights(flow16, args.flow)

    quantizer = FSQRequantizer(args.checkpoint_dir / "stats/post_quantization").to(device).eval()
    motion_rep = load_motion_rep(args.checkpoint_dir)
    loader = DataLoader(
        TeacherShardDataset(args.data),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )
    pairs = {
        "encoder_continuous": PairStats(),
        "decoder_teacher_inputs": PairStats(),
    }
    bins = {"encoder_fsq": BinStats()}
    if flow32 is not None:
        pairs.update(
            {
                "flow_endpoint": PairStats(),
                "flow_endpoint_common_fp32_history": PairStats(),
                "flow_root": PairStats(),
                "full_chain_decoder_body": PairStats(),
            }
        )
        bins["flow_endpoint_fsq"] = BinStats()
        bins["flow_endpoint_fsq_common_fp32_history"] = BinStats()

    samples = 0
    with torch.inference_mode():
        for batch_index, batch in enumerate(loader):
            if args.max_batches and batch_index >= args.max_batches:
                break
            batch = {name: value.to(device).float() for name, value in batch.items()}
            samples += batch["initial_noise"].shape[0]
            encoder_valid = batch["encoder_valid"] > 0.5
            token_valid = batch["decoder_token_valid"] > 0.5
            frame_valid = frame_valid_from_tokens(token_valid)

            enc32 = encoder32(batch["encoder_body"])
            enc16 = encoder16(batch["encoder_body"].half()).float()
            pairs["encoder_continuous"].update(enc16, enc32, encoder_valid)
            enc_q32 = quantizer(enc32, ste=False)
            enc_q16 = quantizer(enc16, ste=False)
            bins["encoder_fsq"].update(enc_q16, enc_q32, encoder_valid)

            dec32 = decoder32(
                batch["decoder_latent"],
                batch["decoder_local_root"],
                batch["decoder_token_valid"],
            )
            dec16 = decoder16(
                batch["decoder_latent"].half(),
                batch["decoder_local_root"].half(),
                batch["decoder_token_valid"].half(),
            ).float()
            pairs["decoder_teacher_inputs"].update(dec16, dec32, frame_valid)

            if flow32 is None or flow16 is None:
                continue
            history_valid = batch["has_history"].unsqueeze(-1)
            history32 = torch.cat(
                [
                    batch["history_hybrid"][..., :20] * history_valid,
                    enc_q32 * history_valid,
                ],
                dim=-1,
            )
            history16 = torch.cat(
                [
                    batch["history_hybrid"][..., :20] * history_valid,
                    enc_q16 * history_valid,
                ],
                dim=-1,
            )
            endpoint32 = flow32.denoise_steps(
                batch["initial_noise"],
                history32,
                batch["path_condition"],
                batch["first_heading"],
                batch["has_history"],
                steps=args.flow_steps,
            )
            endpoint16 = flow16.denoise_steps(
                batch["initial_noise"].half(),
                history16.half(),
                batch["path_condition"].half(),
                batch["first_heading"].half(),
                batch["has_history"].half(),
                steps=args.flow_steps,
            ).float()
            endpoint16_common_history = flow16.denoise_steps(
                batch["initial_noise"].half(),
                history32.half(),
                batch["path_condition"].half(),
                batch["first_heading"].half(),
                batch["has_history"].half(),
                steps=args.flow_steps,
            ).float()
            pairs["flow_endpoint"].update(endpoint16, endpoint32)
            pairs["flow_endpoint_common_fp32_history"].update(
                endpoint16_common_history, endpoint32
            )
            endpoint_q32 = quantizer(endpoint32[..., 20:], ste=False)
            endpoint_q16 = quantizer(endpoint16[..., 20:], ste=False)
            endpoint_q16_common_history = quantizer(
                endpoint16_common_history[..., 20:], ste=False
            )
            bins["flow_endpoint_fsq"].update(endpoint_q16, endpoint_q32)
            bins["flow_endpoint_fsq_common_fp32_history"].update(
                endpoint_q16_common_history, endpoint_q32
            )

            root32, local_root32 = deployment_decoder_roots(
                endpoint32, history32, batch["has_history"], motion_rep
            )
            root16, local_root16 = deployment_decoder_roots(
                endpoint16, history16, batch["has_history"], motion_rep
            )
            pairs["flow_root"].update(root16, root32)
            latent32 = torch.cat([history32[..., 20:], endpoint_q32], dim=1)
            latent16 = torch.cat([history16[..., 20:], endpoint_q16], dim=1)
            chain32 = decoder32(latent32, local_root32, batch["decoder_token_valid"])
            chain16 = decoder16(
                latent16.half(),
                local_root16.half(),
                batch["decoder_token_valid"].half(),
            ).float()
            pairs["full_chain_decoder_body"].update(chain16, chain32, frame_valid)

    result = {
        "schema": "ardy_precision_parity_v1",
        "samples": samples,
        "reference": "pytorch_fp32",
        "candidate": "pytorch_fp16",
        "weights": {
            "encoder": str(args.encoder),
            "decoder": str(args.decoder),
            **({"flow": str(args.flow)} if args.flow is not None else {}),
        },
        "codec_config": asdict(codec_config),
        "flow_config": asdict(flow_config) if args.flow is not None else None,
        "flow_steps": args.flow_steps if args.flow is not None else None,
        "parameter_counts": {
            "encoder": parameter_count(encoder32),
            "decoder": parameter_count(decoder32),
            **({"flow": parameter_count(flow32)} if flow32 is not None else {}),
        },
        "numeric_parity": {name: value.result() for name, value in pairs.items()},
        "fsq_bin_parity": {name: value.result() for name, value in bins.items()},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
