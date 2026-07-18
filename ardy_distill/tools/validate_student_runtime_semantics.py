#!/usr/bin/env python
"""Check student preprocessing against an exact traced release-model call."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from ardy.constraints import Root2DConstraintSet
from ardy.model import load_model
from ardy_distill.losses import FSQRequantizer
from ardy_distill.models import (
    CodecStudentConfig,
    FlowStudentConfig,
    HistoryEncoderStudent,
    MotionDecoderStudent,
    OneStepFlowStudent,
)
from ardy_distill.runtime import load_safetensor_weights
from ardy_distill.student_runtime import StudentArdyRuntime
from ardy_distill.teacher import trace_autoregressive_step


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoints-dir",
        type=Path,
        default=Path("ardy/checkpoints"),
    )
    parser.add_argument("--model", default="ARDY-Core-RP-20FPS-Horizon40")
    parser.add_argument("--encoder", type=Path, required=True)
    parser.add_argument("--flow", type=Path, required=True)
    parser.add_argument("--decoder", type=Path, required=True)
    parser.add_argument("--encoder-width", type=int, default=512)
    parser.add_argument("--encoder-blocks", type=int, default=3)
    parser.add_argument("--decoder-width", type=int, default=512)
    parser.add_argument("--decoder-blocks", type=int, default=4)
    parser.add_argument("--decoder-token-hidden", type=int, default=32)
    parser.add_argument("--flow-width", type=int, default=384)
    parser.add_argument("--flow-heads", type=int, default=6)
    parser.add_argument("--flow-trunk-blocks", type=int, default=4)
    parser.add_argument("--flow-body-blocks", type=int, default=2)
    parser.add_argument("--expansion", type=int, default=2)
    parser.add_argument("--root-smoothing-passes", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def make_constraints(model, target_xz: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    constraint = Root2DConstraintSet(
        model.skeleton,
        frame_indices=torch.tensor([20, 40, 60], dtype=torch.long),
        root_2d=target_xz,
    )
    lengths = torch.tensor([64], device=model.device, dtype=torch.long)
    return model.motion_rep.create_conditions_from_constraints_batched(
        [[constraint]],
        lengths,
        to_normalize=True,
        device=model.device,
    )


def errors(student, teacher) -> dict[str, float]:
    pairs = {
        "path_condition": (student.path_condition, teacher.path_condition),
        "first_heading": (student.first_heading, teacher.first_heading),
        "has_history": (student.has_history, teacher.has_history),
        "global_translation": (student.global_translation, teacher.global_translation),
        "history_root": (student.history_hybrid[..., :20], teacher.history_hybrid[..., :20]),
    }
    return {
        name: float((actual.float() - expected.float()).abs().max())
        for name, (actual, expected) in pairs.items()
    }


def main() -> None:
    args = parse_args()
    torch.manual_seed(20260714)
    device = torch.device(args.device)
    teacher = load_model(
        args.model,
        device=str(device),
        text_encoder=False,
        checkpoints_dir=str(args.checkpoints_dir),
    ).eval()
    codec_config = CodecStudentConfig(
        encoder_width=args.encoder_width,
        encoder_blocks=args.encoder_blocks,
        decoder_width=args.decoder_width,
        decoder_blocks=args.decoder_blocks,
        decoder_token_hidden=args.decoder_token_hidden,
        expansion=args.expansion,
    )
    flow_config = FlowStudentConfig(
        width=args.flow_width,
        heads=args.flow_heads,
        trunk_blocks=args.flow_trunk_blocks,
        body_blocks=args.flow_body_blocks,
        expansion=args.expansion,
        root_smoothing_passes=args.root_smoothing_passes,
    )
    encoder = HistoryEncoderStudent(codec_config).to(device).eval()
    flow = OneStepFlowStudent(flow_config).to(device).eval()
    decoder = MotionDecoderStudent(codec_config).to(device).eval()
    load_safetensor_weights(encoder, args.encoder)
    load_safetensor_weights(flow, args.flow)
    load_safetensor_weights(decoder, args.decoder)
    checkpoint_dir = args.checkpoints_dir / args.model
    quantizer = FSQRequantizer(checkpoint_dir / "stats/post_quantization").to(device)
    runtime = StudentArdyRuntime(
        encoder,
        flow,
        decoder,
        quantizer,
        teacher.motion_rep,
    ).eval()
    text_feat = torch.zeros(1, 1, 4096, device=device)
    text_mask = torch.zeros(1, 1, dtype=torch.bool, device=device)

    targets_initial = torch.tensor(
        [[0.35, 0.15], [0.9, -0.2], [1.6, 0.45]],
        dtype=torch.float32,
    )
    observed, mask = make_constraints(teacher, targets_initial)
    initial_translation = torch.tensor([[1.25, 0.0, -0.75]], device=device)
    initial_heading = torch.tensor([0.6], device=device)
    initial_trace = trace_autoregressive_step(
        teacher,
        num_frames=64,
        num_denoising_steps=10,
        motion_mask=mask,
        observed_motion=observed,
        cfg_weight=(0.0, 1.5),
        texts=None,
        text_feat=text_feat,
        text_pad_mask=text_mask,
        init_history_sequence=None,
        init_global_translation=initial_translation,
        init_first_heading_angle=initial_heading,
    )
    initial_student = runtime.step(
        motion_mask=mask,
        observed_motion=observed,
        initial_noise=initial_trace.initial_noise,
        init_global_translation=initial_translation,
        init_first_heading_angle=initial_heading,
    )

    history = initial_trace.explicit_motion[:, -4:]
    root_position = teacher.motion_rep.get_root_pos(teacher.motion_rep.unnormalize(history))[:, -1, [0, 2]]
    offsets = torch.tensor(
        [[0.25, 0.10], [0.85, -0.15], [1.45, 0.35]],
        device=device,
    )
    observed, mask = make_constraints(teacher, root_position[0].cpu() + offsets.cpu())
    continuation_trace = trace_autoregressive_step(
        teacher,
        num_frames=64,
        num_denoising_steps=10,
        motion_mask=mask,
        observed_motion=observed,
        cfg_weight=(0.0, 1.5),
        texts=None,
        text_feat=text_feat,
        text_pad_mask=text_mask,
        init_history_sequence=history,
        init_global_translation=None,
        init_first_heading_angle=None,
    )
    continuation_student = runtime.step(
        motion_mask=mask,
        observed_motion=observed,
        initial_noise=continuation_trace.initial_noise,
        init_history_sequence=history,
    )

    cases = {
        "initial": {
            "semantic_max_abs_error": errors(initial_student, initial_trace),
            "output_shape": list(initial_student.explicit_motion.shape),
            "finite": bool(torch.isfinite(initial_student.explicit_motion).all()),
        },
        "continuation": {
            "semantic_max_abs_error": errors(continuation_student, continuation_trace),
            "output_shape": list(continuation_student.explicit_motion.shape),
            "finite": bool(torch.isfinite(continuation_student.explicit_motion).all()),
        },
    }
    max_error = max(
        value
        for case in cases.values()
        for value in case["semantic_max_abs_error"].values()
    )
    passed = (
        max_error <= 1.0e-6
        and all(case["finite"] for case in cases.values())
        and cases["initial"]["output_shape"] == [1, 40, 330]
        and cases["continuation"]["output_shape"] == [1, 44, 330]
    )
    result = {
        "schema": "ardy_student_runtime_semantics_v1",
        "passed": passed,
        "max_semantic_abs_error": max_error,
        "architecture": {
            "encoder_width": args.encoder_width,
            "encoder_blocks": args.encoder_blocks,
            "decoder_width": args.decoder_width,
            "decoder_blocks": args.decoder_blocks,
            "decoder_token_hidden": args.decoder_token_hidden,
            "flow_width": args.flow_width,
            "flow_heads": args.flow_heads,
            "flow_trunk_blocks": args.flow_trunk_blocks,
            "flow_body_blocks": args.flow_body_blocks,
            "expansion": args.expansion,
            "root_smoothing_passes": args.root_smoothing_passes,
        },
        "cases": cases,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
