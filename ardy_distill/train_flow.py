#!/usr/bin/env python
"""Accelerate training for continuous flow and the final NFE=1 endpoint."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import torch
from accelerate import Accelerator
from accelerate.utils import set_seed
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from .data import (
    IndexedTeacherMixtureDataset,
    RandomTeacherMixtureDataset,
    ShardMixtureSampler,
    ShardShuffleSampler,
    TeacherShardDataset,
)
from .codec_cli import add_codec_config_arguments, codec_config_from_args
from .dmd2 import path_constraint_losses, seam_losses
from .ema import ModelEMA, ema_99_percent_horizon
from .flow_matching import euler_flow_denoise, make_flow_pair, sample_flow_time
from .losses import (
    FSQRequantizer,
    decoder_feature_losses,
    deployment_decoder_roots,
    flow_velocity_losses,
    fsq_endpoint_diagnostics,
    fsq_endpoint_losses,
    motion_quality_losses,
    physical_seam_losses,
    root_temporal_losses,
)
from .models import (
    FlowStudentConfig,
    HistoryEncoderStudent,
    MotionDecoderStudent,
    OneStepFlowStudent,
    project_root_trajectory,
)
from .runtime import (
    MetricLogger,
    load_motion_rep,
    load_safetensor_weights,
    save_safetensor_weights,
    tensor_metrics_to_float,
)
from .text_features import PromptFeatureTable


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument(
        "--text-features",
        type=Path,
        help=(
            "Optional complete Qwen prompt-feature table. When set, V3 prompt ids "
            "are looked up on GPU and the flow learns a full text projection."
        ),
    )
    parser.add_argument(
        "--heading-condition-features",
        type=int,
        choices=[0, 3],
        default=0,
        help="Use V3 normalized future heading cos/sin/valid inputs when set to 3.",
    )
    parser.add_argument(
        "--replay-data",
        type=Path,
        help="Optional replay corpus mixed with --data to prevent on-policy forgetting.",
    )
    parser.add_argument(
        "--primary-data-prob",
        type=float,
        default=0.5,
        help="Per-sample probability for --data when --replay-data is set.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--init-flow",
        type=Path,
        help="Initialize flow weights while starting a fresh optimizer/scheduler/EMA.",
    )
    parser.add_argument(
        "--expandable-init",
        action="store_true",
        help=(
            "Load a shallower same-width flow and initialize newly appended attention "
            "blocks as exact residual identities."
        ),
    )
    parser.add_argument("--decoder", type=Path, required=True, help="Trained decoder EMA safetensors")
    parser.add_argument(
        "--encoder",
        type=Path,
        help="Optional trained history encoder. When set, flow sees deployment-style encoded history.",
    )
    parser.add_argument(
        "--teacher-history-prob",
        type=float,
        default=0.0,
        help="Probability of retaining teacher body history when --encoder is set.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path("ardy/checkpoints/ARDY-Core-RP-20FPS-Horizon40"),
    )
    parser.add_argument("--steps", type=int, default=100_000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--gradient-accumulation", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=5.0e-5)
    parser.add_argument(
        "--condition-learning-rate",
        type=float,
        help=(
            "Optional learning rate for newly added text/heading projections. "
            "Defaults to --learning-rate."
        ),
    )
    parser.add_argument(
        "--lr-schedule",
        choices=["constant", "step"],
        default="step",
        help="Formal teacher fitting uses one 5e-5 to 1e-5 step decay.",
    )
    parser.add_argument(
        "--lr-decay-step",
        type=int,
        default=50_000,
        help="Optimizer step where the step schedule switches to --final-learning-rate.",
    )
    parser.add_argument(
        "--final-learning-rate",
        type=float,
        default=1.0e-5,
        help="Learning rate after --lr-decay-step when --lr-schedule=step.",
    )
    parser.add_argument(
        "--final-condition-learning-rate",
        type=float,
        help=(
            "Optional post-decay learning rate for text/heading projections. "
            "Defaults to --final-learning-rate."
        ),
    )
    parser.add_argument(
        "--condition-only",
        action="store_true",
        help=(
            "Freeze the warm-started flow and optimize only text/heading projections. "
            "This is the projection warm-up stage before joint continuation."
        ),
    )
    parser.add_argument(
        "--base-gradient-clip",
        type=float,
        default=1.0,
        help="Independent L2 gradient clipping threshold for the existing flow; zero disables it.",
    )
    parser.add_argument(
        "--condition-gradient-clip",
        type=float,
        default=1.0,
        help="Independent L2 gradient clipping threshold for text/heading projections; zero disables it.",
    )
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--warmup-steps", type=int, default=0)
    parser.add_argument("--mixed-precision", choices=["no", "fp16", "bf16"], default="bf16")
    parser.add_argument(
        "--frozen-codec-dtype",
        choices=["fp16", "fp32"],
        default="fp16",
        help=(
            "Arithmetic dtype for the frozen encoder/decoder used inside flow training; "
            "fp16 matches the browser deployment path while the trainable flow can remain BF16."
        ),
    )
    parser.add_argument(
        "--ema-decay",
        type=float,
        default=0.9995,
        help="Constant FP32 EMA decay; 0.9995 has an approximately 9.2k-step 99%% horizon.",
    )
    parser.add_argument(
        "--override-ema-decay-on-resume",
        action="store_true",
        help=(
            "Keep the command-line --ema-decay after loading a resume state instead of "
            "restoring the checkpoint's decay. Model/optimizer/RNG and EMA shadow still resume."
        ),
    )
    parser.add_argument("--decode-every", type=int, default=1)
    parser.add_argument("--quality-every", type=int, default=1)
    parser.add_argument("--decoder-weight", type=float, default=0.1)
    parser.add_argument(
        "--fsq-weight",
        type=float,
        default=1.0,
        help="Weight of the STE re-quantized endpoint latent L1.",
    )
    parser.add_argument("--quality-weight", type=float, default=0.10)
    parser.add_argument(
        "--foot-slide-quality-weight",
        type=float,
        default=0.1,
        help="Foot-slide coefficient inside quality_total before --quality-weight.",
    )
    parser.add_argument(
        "--path-weight",
        type=float,
        default=0.1,
        help="Weight of explicit in-generation x/z waypoint MSE.",
    )
    parser.add_argument(
        "--seam-weight",
        type=float,
        default=0.01,
        help="Weight of root/body velocity continuity at the history-generation seam.",
    )
    parser.add_argument(
        "--physical-seam-weight",
        type=float,
        default=0.001,
        help=(
            "Weight of rollout-aligned root/joint seam error in physical units; "
            "zero preserves earlier training behavior."
        ),
    )
    parser.add_argument(
        "--root-temporal-weight",
        type=float,
        default=0.01,
        help="Weight of physical root velocity/acceleration/jerk matching.",
    )
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--save-every", type=int, default=5_000)
    parser.add_argument(
        "--state-every",
        type=int,
        default=25_000,
        help="Full optimizer/RNG recovery interval; weights use --save-every.",
    )
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--cache-shards", type=int, default=2)
    parser.add_argument(
        "--sample-order",
        choices=["shard_shuffle", "global_shuffle", "sequential"],
        default="shard_shuffle",
        help="Shard-local order avoids random safetensors opens and is the default.",
    )
    parser.add_argument("--flow-width", type=int, default=512)
    parser.add_argument("--flow-heads", type=int, default=8)
    parser.add_argument("--flow-trunk-blocks", type=int, default=8)
    parser.add_argument("--flow-body-blocks", type=int, default=8)
    parser.add_argument(
        "--solver-steps",
        type=int,
        default=1,
        help="Differentiable uniform-Euler NFE used for paired endpoint/decoder losses.",
    )
    parser.add_argument(
        "--flow-root-smoothing-passes",
        type=int,
        default=0,
        help="Differentiable binomial projection passes in the NFE=1 endpoint graph.",
    )
    parser.add_argument(
        "--flow-root-projection-kind",
        choices=["binomial", "cubic_controls"],
        default="binomial",
    )
    parser.add_argument("--flow-root-control-points", type=int, default=10)
    add_codec_config_arguments(parser)
    parser.add_argument(
        "--exact-t1-probability",
        type=float,
        default=0.70,
        help="Fraction of flow-matching samples placed exactly at maximum noise t=1.",
    )
    parser.add_argument(
        "--high-noise-probability",
        type=float,
        default=0.20,
        help="Additional fraction sampled from t=1-U^3; the remainder is uniform.",
    )
    parser.add_argument("--seed", type=int, default=20260714)
    parser.add_argument("--max-runtime-s", type=float, default=43_200)
    parser.add_argument("--resume", type=Path)
    parser.add_argument(
        "--override-learning-rates-on-resume",
        action="store_true",
        help=(
            "After restoring model/optimizer/scheduler/EMA/RNG state, apply the "
            "learning rates and schedule from this command. This permits an exact "
            "checkpoint branch without discarding Adam or EMA."
        ),
    )
    return parser.parse_args()


def endpoint_losses(prediction: torch.Tensor, target: torch.Tensor) -> dict[str, torch.Tensor]:
    root = prediction[..., :20] - target[..., :20]
    body = prediction[..., 20:] - target[..., 20:]
    result = {
        "endpoint_root_mse": root.square().mean(),
        "endpoint_body_mse": body.square().mean(),
        "endpoint_root_l1": root.abs().mean(),
        "endpoint_body_l1": body.abs().mean(),
    }
    result["endpoint_total"] = (
        2.0 * result["endpoint_root_mse"]
        + result["endpoint_body_mse"]
        + 0.25 * (result["endpoint_root_l1"] + result["endpoint_body_l1"])
    )
    return result


@torch.no_grad()
def gradient_l2_norm(parameters: list[nn.Parameter]) -> torch.Tensor:
    """FP32 pre-clip L2 norm without flattening a multi-million parameter group."""

    reference = next((parameter for parameter in parameters if parameter.grad is not None), None)
    if reference is None:
        return torch.zeros((), dtype=torch.float32)
    total = torch.zeros((), device=reference.grad.device, dtype=torch.float32)
    for parameter in parameters:
        if parameter.grad is not None:
            total.add_(parameter.grad.detach().float().square().sum())
    return total.sqrt()


@torch.no_grad()
def initialize_flow_residuals_as_identity(flow: OneStepFlowStudent) -> None:
    """Make every attention block an identity before overlaying old weights."""

    for block in [*flow.trunk, *flow.body_refiner]:
        nn.init.zeros_(block.attention_out.weight)
        nn.init.zeros_(block.attention_out.bias)
        nn.init.zeros_(block.channel.fc2.weight)
        nn.init.zeros_(block.channel.fc2.bias)


def load_expandable_flow_weights(flow: OneStepFlowStudent, path: Path) -> list[str]:
    initialize_flow_residuals_as_identity(flow)
    incompatible = load_safetensor_weights(flow, path, strict=False)
    if incompatible.unexpected_keys:
        raise RuntimeError(
            f"unexpected flow keys during expansion: {incompatible.unexpected_keys}"
        )
    allowed_prefixes = (
        "trunk.",
        "body_refiner.",
        "text_proj.",
        "heading_proj.",
    )
    invalid_missing = [
        name
        for name in incompatible.missing_keys
        if not name.startswith(allowed_prefixes)
    ]
    if invalid_missing:
        raise RuntimeError(
            f"non-block flow tensors missing during expansion: {invalid_missing}"
        )
    return list(incompatible.missing_keys)


def save_weights(flow: OneStepFlowStudent, ema: ModelEMA, output: Path, step: int) -> None:
    checkpoint = output / "weights" / f"step-{step:07d}"
    save_safetensor_weights(flow, checkpoint / "flow.safetensors")
    ema_state = flow.state_dict()
    ema_state.update(ema.shadow)
    save_safetensor_weights(flow, checkpoint / "flow_ema.safetensors", ema_state)


def main() -> None:
    args = parse_args()
    condition_learning_rate = (
        args.learning_rate
        if args.condition_learning_rate is None
        else args.condition_learning_rate
    )
    final_condition_learning_rate = (
        args.final_learning_rate
        if args.final_condition_learning_rate is None
        else args.final_condition_learning_rate
    )
    if not 0.0 <= args.teacher_history_prob <= 1.0:
        raise ValueError("--teacher-history-prob must be in [0, 1]")
    if args.replay_data is not None and not 0.0 < args.primary_data_prob < 1.0:
        raise ValueError("--primary-data-prob must be strictly between zero and one")
    if args.replay_data is None and args.primary_data_prob != 0.5:
        raise ValueError("--primary-data-prob only applies with --replay-data")
    nonnegative = {
        "decoder_weight": args.decoder_weight,
        "fsq_weight": args.fsq_weight,
        "quality_weight": args.quality_weight,
        "foot_slide_quality_weight": args.foot_slide_quality_weight,
        "path_weight": args.path_weight,
        "seam_weight": args.seam_weight,
        "physical_seam_weight": args.physical_seam_weight,
        "root_temporal_weight": args.root_temporal_weight,
    }
    for name, value in nonnegative.items():
        if value < 0:
            raise ValueError(f"--{name.replace('_', '-')} must be non-negative")
    if args.init_flow is not None and args.resume is not None:
        raise ValueError("--init-flow and --resume are mutually exclusive")
    if args.expandable_init and args.init_flow is None:
        raise ValueError("--expandable-init requires --init-flow")
    if args.save_every < 1 or args.state_every < 1:
        raise ValueError("save/state intervals must be positive")
    if args.base_gradient_clip < 0 or args.condition_gradient_clip < 0:
        raise ValueError("gradient clipping thresholds must be non-negative")
    if args.learning_rate <= 0 or condition_learning_rate <= 0:
        raise ValueError("learning rates must be positive")
    if not 0.0 < args.ema_decay < 1.0:
        raise ValueError("--ema-decay must be in (0, 1)")
    if args.override_ema_decay_on_resume and args.resume is None:
        raise ValueError("--override-ema-decay-on-resume requires --resume")
    if args.override_learning_rates_on_resume and args.resume is None:
        raise ValueError("--override-learning-rates-on-resume requires --resume")
    if args.flow_root_smoothing_passes < 0:
        raise ValueError("--flow-root-smoothing-passes must be non-negative")
    if not 2 <= args.flow_root_control_points <= 40:
        raise ValueError("--flow-root-control-points must be within [2, 40]")
    if args.solver_steps < 1:
        raise ValueError("--solver-steps must be positive")
    if not 0.0 <= args.exact_t1_probability <= 1.0:
        raise ValueError("--exact-t1-probability must be in [0, 1]")
    if not 0.0 <= args.high_noise_probability <= 1.0:
        raise ValueError("--high-noise-probability must be in [0, 1]")
    if args.exact_t1_probability + args.high_noise_probability > 1.0:
        raise ValueError("flow-time mixture probabilities exceed one")
    if args.lr_schedule == "step":
        if not 0 < args.lr_decay_step < args.steps:
            raise ValueError("step LR decay must occur strictly inside the training run")
        if not 0.0 < args.final_learning_rate <= args.learning_rate:
            raise ValueError("final learning rate must be in (0, initial learning rate]")
        if not 0.0 < final_condition_learning_rate <= condition_learning_rate:
            raise ValueError(
                "final condition learning rate must be in (0, condition learning rate]"
            )
    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
        gradient_accumulation_steps=args.gradient_accumulation,
        step_scheduler_with_optimizer=False,
    )
    set_seed(args.seed, device_specific=True)
    text_feature_dim = 0
    text_feature_manifest = None
    if args.text_features is not None:
        text_feature_manifest = json.loads(
            (args.text_features / "manifest.json").read_text(encoding="utf-8")
        )
        if text_feature_manifest.get("encoder") != "qwen":
            raise ValueError("--text-features must point to the Qwen feature table")
        if not text_feature_manifest.get("complete"):
            raise ValueError("--text-features table is incomplete")
        text_feature_dim = int(text_feature_manifest["feature_dim"])
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
    codec_config = codec_config_from_args(args)
    flow = OneStepFlowStudent(flow_config)
    if args.init_flow is not None:
        if args.expandable_init:
            load_expandable_flow_weights(flow, args.init_flow)
        else:
            load_safetensor_weights(flow, args.init_flow)
    condition_parameters = [
        parameter
        for module in (flow.text_proj, flow.heading_proj)
        if module is not None
        for parameter in module.parameters()
    ]
    condition_parameter_ids = {id(parameter) for parameter in condition_parameters}
    base_parameters = [
        parameter
        for parameter in flow.parameters()
        if id(parameter) not in condition_parameter_ids
    ]
    if args.condition_only:
        if not condition_parameters:
            raise ValueError("--condition-only requires text and/or heading projections")
        for parameter in base_parameters:
            parameter.requires_grad_(False)
    trainable_base_parameters = [
        parameter for parameter in base_parameters if parameter.requires_grad
    ]
    trainable_condition_parameters = [
        parameter for parameter in condition_parameters if parameter.requires_grad
    ]
    args.output.mkdir(parents=True, exist_ok=True)
    if accelerator.is_main_process:
        config_name = (
            "config.json"
            if args.resume is None
            else f"resume_config_{args.resume.name}.json"
        )
        (args.output / config_name).write_text(
            json.dumps(
                {
                    **vars(args),
                    "resolved_condition_learning_rate": condition_learning_rate,
                    "resolved_final_condition_learning_rate": final_condition_learning_rate,
                    "base_trainable_parameters": sum(
                        parameter.numel() for parameter in trainable_base_parameters
                    ),
                    "condition_trainable_parameters": sum(
                        parameter.numel() for parameter in trainable_condition_parameters
                    ),
                    "ema_99_percent_horizon_steps": ema_99_percent_horizon(args.ema_decay),
                    "model_config": asdict(flow_config),
                    "codec_model_config": asdict(codec_config),
                },
                default=str,
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

    training_fields = (
        "initial_noise",
        "clean_generation",
        "history_hybrid",
        "path_condition",
        "first_heading",
        "has_history",
        *(("prompt_id",) if args.text_features is not None else ()),
        *(("heading_condition",) if args.heading_condition_features else ()),
        "decoder_global_root",
        "decoder_token_valid",
        "target_body",
        *(("encoder_body",) if args.encoder is not None else ()),
    )
    primary_dataset = TeacherShardDataset(
        args.data,
        cache_shards=args.cache_shards,
        fields=training_fields,
    )
    sampler = None
    if args.replay_data is None:
        dataset = primary_dataset
        if args.sample_order == "shard_shuffle":
            sampler = ShardShuffleSampler(primary_dataset, seed=args.seed)
    else:
        replay_dataset = TeacherShardDataset(
            args.replay_data,
            cache_shards=args.cache_shards,
            fields=training_fields,
        )
        if args.sample_order == "shard_shuffle":
            dataset = IndexedTeacherMixtureDataset(primary_dataset, replay_dataset)
            sampler = ShardMixtureSampler(
                dataset,
                primary_probability=args.primary_data_prob,
                chunk_size=args.batch_size,
                seed=args.seed,
            )
        else:
            dataset = RandomTeacherMixtureDataset(
                primary_dataset,
                replay_dataset,
                primary_probability=args.primary_data_prob,
            )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=args.sample_order == "global_shuffle",
        sampler=sampler,
        drop_last=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    optimizer_groups = []
    optimizer_group_names = []
    final_group_learning_rates = []
    if trainable_base_parameters:
        optimizer_groups.append(
            {
                "params": trainable_base_parameters,
                "lr": args.learning_rate,
                "weight_decay": args.weight_decay,
            }
        )
        optimizer_group_names.append("base")
        final_group_learning_rates.append(args.final_learning_rate)
    if trainable_condition_parameters:
        optimizer_groups.append(
            {
                "params": trainable_condition_parameters,
                "lr": condition_learning_rate,
                "weight_decay": args.weight_decay,
            }
        )
        optimizer_group_names.append("condition")
        final_group_learning_rates.append(final_condition_learning_rate)
    optimizer = AdamW(optimizer_groups)
    if args.lr_schedule == "constant":
        scheduler = LambdaLR(
            optimizer,
            [lambda _step: 1.0 for _ in optimizer_groups],
        )
    else:
        final_ratios = [
            final_learning_rate / group["lr"]
            for group, final_learning_rate in zip(
                optimizer_groups, final_group_learning_rates
            )
        ]
        scheduler = LambdaLR(
            optimizer,
            [
                (lambda step, ratio=ratio: 1.0 if step < args.lr_decay_step else ratio)
                for ratio in final_ratios
            ],
        )
    flow, optimizer, loader, scheduler = accelerator.prepare(flow, optimizer, loader, scheduler)
    ema = ModelEMA(
        accelerator.unwrap_model(flow),
        decay=args.ema_decay,
        override_decay_on_load=args.override_ema_decay_on_resume,
    )
    accelerator.register_for_checkpointing(ema)
    if sampler is not None:
        accelerator.register_for_checkpointing(sampler)

    frozen_codec_dtype = {
        "fp16": torch.float16,
        "fp32": torch.float32,
    }[args.frozen_codec_dtype]
    if frozen_codec_dtype == torch.float16 and accelerator.device.type != "cuda":
        raise ValueError("--frozen-codec-dtype=fp16 requires a CUDA training device")
    decoder = MotionDecoderStudent(codec_config).to(
        device=accelerator.device, dtype=frozen_codec_dtype
    ).eval()
    load_safetensor_weights(decoder, args.decoder)
    decoder.requires_grad_(False)
    encoder = None
    if args.encoder is not None:
        encoder = HistoryEncoderStudent(codec_config).to(
            device=accelerator.device, dtype=frozen_codec_dtype
        ).eval()
        load_safetensor_weights(encoder, args.encoder)
        encoder.requires_grad_(False)
    quantizer = FSQRequantizer(args.checkpoint_dir / "stats/post_quantization").to(accelerator.device)
    motion_rep = load_motion_rep(args.checkpoint_dir)
    text_features = None
    if args.text_features is not None:
        text_features = PromptFeatureTable(
            args.text_features,
            expected_encoder="qwen",
        ).to(accelerator.device, dtype=torch.bfloat16)
        if text_features.feature_dim != flow_config.text_feature_dim:
            raise ValueError("Qwen feature table and flow config dimensions differ")
    logger = MetricLogger(args.output, enabled=accelerator.is_main_process)

    global_step = 0
    start_time = time.perf_counter()
    if args.resume is not None:
        accelerator.load_state(str(args.resume))
        try:
            global_step = int(args.resume.name.split("-")[-1])
        except ValueError:
            global_step = 0
        if args.override_learning_rates_on_resume:
            initial_learning_rates = [
                (
                    args.learning_rate
                    if name == "base"
                    else condition_learning_rate
                )
                for name in optimizer_group_names
            ]
            final_learning_rates = [
                (
                    args.final_learning_rate
                    if name == "base"
                    else final_condition_learning_rate
                )
                for name in optimizer_group_names
            ]
            if args.lr_schedule == "constant" or global_step < args.lr_decay_step:
                resumed_learning_rates = initial_learning_rates
            else:
                resumed_learning_rates = final_learning_rates

            for group, initial_lr, resumed_lr in zip(
                optimizer.param_groups,
                initial_learning_rates,
                resumed_learning_rates,
            ):
                group["initial_lr"] = initial_lr
                group["lr"] = resumed_lr

            base_scheduler = getattr(scheduler, "scheduler", scheduler)
            base_scheduler.base_lrs = list(initial_learning_rates)
            base_scheduler.last_epoch = global_step
            base_scheduler._step_count = global_step + 1
            base_scheduler._last_lr = list(resumed_learning_rates)
            if accelerator.is_main_process:
                event = {
                    "event": "resume_learning_rates_overridden",
                    "step": global_step,
                    "schedule": args.lr_schedule,
                    "learning_rates": dict(
                        zip(optimizer_group_names, resumed_learning_rates)
                    ),
                }
                logger.log(
                    global_step,
                    {},
                    event=event["event"],
                    schedule=event["schedule"],
                    learning_rates=event["learning_rates"],
                )
                print(json.dumps(event), flush=True)

    optimizer.zero_grad(set_to_none=True)
    stop = False
    optimizer_steps_skipped = 0
    while global_step < args.steps and not stop:
        for batch in loader:
            batch = {
                name: (
                    value.to(accelerator.device, non_blocking=True).float()
                    if value.is_floating_point()
                    else value.to(accelerator.device, non_blocking=True)
                )
                for name, value in batch.items()
            }
            text_feature = (
                None
                if text_features is None
                else text_features.lookup(batch["prompt_id"])
            )
            heading_condition = batch.get("heading_condition")
            history_hybrid = batch["history_hybrid"]
            if encoder is not None:
                with torch.no_grad(), torch.autocast(
                    device_type=accelerator.device.type, enabled=False
                ):
                    encoded_history = encoder(
                        batch["encoder_body"].to(dtype=frozen_codec_dtype)
                    ).float()
                # The released _encode_init_history path returns an FSQ token.
                # Keep the student deployment path identical: the encoder
                # predicts the normalized token, then the released 64-level
                # quantization grid is applied before flow and decoder use it.
                encoded_history = quantizer(encoded_history, ste=False)
                if args.teacher_history_prob > 0:
                    keep_teacher = (
                        torch.rand(
                            encoded_history.shape[0],
                            1,
                            1,
                            device=encoded_history.device,
                        )
                        < args.teacher_history_prob
                    )
                    history_body = torch.where(
                        keep_teacher,
                        batch["history_hybrid"][..., 20:],
                        encoded_history,
                    )
                else:
                    history_body = encoded_history
                history_valid = batch["has_history"].unsqueeze(-1)
                history_hybrid = torch.cat(
                    [
                        batch["history_hybrid"][..., :20] * history_valid,
                        history_body * history_valid,
                    ],
                    dim=-1,
                )
            with accelerator.accumulate(flow):
                time_value = sample_flow_time(
                    batch["clean_generation"].shape[0],
                    accelerator.device,
                    dtype=batch["clean_generation"].dtype,
                    exact_t1_probability=args.exact_t1_probability,
                    high_noise_probability=args.high_noise_probability,
                )
                noisy, target_velocity = make_flow_pair(
                    batch["clean_generation"],
                    batch["initial_noise"],
                    time_value,
                )
                with accelerator.autocast():
                    velocity = flow(
                        noisy,
                        history_hybrid,
                        batch["path_condition"],
                        batch["first_heading"],
                        batch["has_history"],
                        time_value,
                        text_feature,
                        heading_condition,
                    )
                    velocity_losses = flow_velocity_losses(velocity, target_velocity)
                    endpoint = euler_flow_denoise(
                        flow,
                        batch["initial_noise"],
                        history_hybrid,
                        batch["path_condition"],
                        batch["first_heading"],
                        batch["has_history"],
                        steps=args.solver_steps,
                        text_feature=text_feature,
                        heading_condition=heading_condition,
                    )
                    unwrapped_flow = flow.module if hasattr(flow, "module") else flow
                    endpoint = project_root_trajectory(
                        endpoint,
                        args.flow_root_smoothing_passes,
                        kind=args.flow_root_projection_kind,
                        control_points=args.flow_root_control_points,
                        basis=unwrapped_flow.root_projection_basis,
                    )
                    paired_losses = endpoint_losses(endpoint, batch["clean_generation"])
                    fsq_diagnostics = fsq_endpoint_diagnostics(
                        endpoint[..., 20:],
                        batch["clean_generation"][..., 20:],
                        quantizer,
                    )
                    fsq_losses = fsq_endpoint_losses(
                        endpoint[..., 20:],
                        batch["clean_generation"][..., 20:],
                        quantizer,
                    )
                    temporal_root_losses = root_temporal_losses(
                        endpoint,
                        batch["clean_generation"],
                        motion_rep,
                    )
                    path_losses = path_constraint_losses(
                        endpoint,
                        batch["path_condition"],
                        batch["has_history"],
                    )
                    total = (
                        velocity_losses["flow_total"]
                        + paired_losses["endpoint_total"]
                        + args.fsq_weight * fsq_losses["fsq_quantized_l1"]
                        + args.path_weight * path_losses["path_constraint_mse"]
                        + args.root_temporal_weight
                        * temporal_root_losses["root_temporal_total"]
                    )

                    decoder_losses = {}
                    decoder_prediction = None
                    predicted_root = None
                    predicted_decoder_local_root = None
                    if global_step % args.decode_every == 0:
                        predicted_root, predicted_decoder_local_root = (
                            deployment_decoder_roots(
                                endpoint,
                                history_hybrid,
                                batch["has_history"],
                                motion_rep,
                            )
                        )
                        generated_latent = quantizer(endpoint[..., 20:], ste=True)
                        latent_sequence = torch.cat(
                            [history_hybrid[..., 20:], generated_latent],
                            dim=1,
                        )
                        with torch.autocast(
                            device_type=accelerator.device.type, enabled=False
                        ):
                            decoder_prediction = decoder(
                                latent_sequence.to(dtype=frozen_codec_dtype),
                                predicted_decoder_local_root.to(dtype=frozen_codec_dtype),
                                batch["decoder_token_valid"].to(dtype=frozen_codec_dtype),
                            ).float()
                        decoder_losses = decoder_feature_losses(
                            decoder_prediction,
                            batch["target_body"],
                            batch["decoder_token_valid"],
                        )
                        total = total + args.decoder_weight * decoder_losses["decoder_total"]

                quality_losses = {}
                boundary_losses = {}
                physical_boundary_losses = {}
                if predicted_root is not None and args.seam_weight > 0:
                    boundary_losses = seam_losses(
                        endpoint,
                        decoder_prediction,
                        history_hybrid,
                        batch["decoder_global_root"],
                        batch["target_body"],
                        batch["has_history"],
                    )
                    total = total + args.seam_weight * boundary_losses["seam_total"]
                if predicted_root is not None and args.physical_seam_weight > 0:
                    physical_boundary_losses = physical_seam_losses(
                        decoder_prediction,
                        batch["target_body"],
                        predicted_root,
                        batch["decoder_global_root"],
                        batch["has_history"],
                        motion_rep,
                    )
                    total = total + (
                        args.physical_seam_weight
                        * physical_boundary_losses["physical_seam_total"]
                    )
                if predicted_root is not None and global_step % args.quality_every == 0:
                    quality_losses = motion_quality_losses(
                        decoder_prediction,
                        batch["target_body"],
                        predicted_root,
                        batch["decoder_token_valid"],
                        motion_rep,
                        target_normalized_global_root=batch["decoder_global_root"],
                    )
                    quality_total = (
                        0.10 * quality_losses["rotation_geodesic"]
                        + quality_losses["fk_mpjpe"]
                        + 0.10 * quality_losses["joint_velocity_l1"]
                        + 0.05 * quality_losses["joint_acceleration_l1"]
                        + args.foot_slide_quality_weight * quality_losses["foot_slide"]
                    )
                    quality_losses["quality_total"] = quality_total
                    total = total + args.quality_weight * quality_total

                accelerator.backward(total)
                if accelerator.sync_gradients:
                    # Unscale once, then keep the inherited flow and the newly
                    # introduced condition adapters from suppressing each
                    # other's gradients through a single global norm.
                    accelerator.unscale_gradients(optimizer)
                    if trainable_base_parameters:
                        base_grad_norm = (
                            torch.nn.utils.clip_grad_norm_(
                                trainable_base_parameters,
                                args.base_gradient_clip,
                            )
                            if args.base_gradient_clip > 0
                            else gradient_l2_norm(trainable_base_parameters)
                        ).to(total.device)
                    else:
                        base_grad_norm = total.new_zeros(())
                    if trainable_condition_parameters:
                        condition_grad_norm = (
                            torch.nn.utils.clip_grad_norm_(
                                trainable_condition_parameters,
                                args.condition_gradient_clip,
                            )
                            if args.condition_gradient_clip > 0
                            else gradient_l2_norm(trainable_condition_parameters)
                        ).to(total.device)
                    else:
                        condition_grad_norm = total.new_zeros(())
                    grad_norm = torch.sqrt(
                        base_grad_norm.square() + condition_grad_norm.square()
                    )
                optimizer.step()
                optimizer_step_skipped = bool(
                    accelerator.sync_gradients
                    and accelerator.optimizer_step_was_skipped
                )
                if accelerator.sync_gradients and not optimizer_step_skipped:
                    scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            if not accelerator.sync_gradients:
                continue
            if optimizer_step_skipped:
                optimizer_steps_skipped += 1
                if accelerator.is_main_process:
                    logger.log(
                        global_step,
                        {},
                        event="optimizer_step_skipped",
                        skipped_updates=optimizer_steps_skipped,
                    )
                    print(
                        json.dumps(
                            {
                                "step": global_step,
                                "event": "optimizer_step_skipped",
                                "skipped_updates": optimizer_steps_skipped,
                            }
                        ),
                        flush=True,
                    )
                continue
            global_step += 1
            unwrapped = accelerator.unwrap_model(flow)
            ema.update(unwrapped)

            if global_step % args.log_every == 0 or global_step == 1:
                group_learning_rates = dict(
                    zip(optimizer_group_names, scheduler.get_last_lr())
                )
                base_parameter_count = sum(
                    parameter.numel() for parameter in trainable_base_parameters
                )
                condition_parameter_count = sum(
                    parameter.numel() for parameter in trainable_condition_parameters
                )
                metrics = {
                    **velocity_losses,
                    **paired_losses,
                    **fsq_losses,
                    **fsq_diagnostics,
                    **path_losses,
                    **temporal_root_losses,
                    **decoder_losses,
                    **boundary_losses,
                    **physical_boundary_losses,
                    **quality_losses,
                    "loss_total": total,
                    "flow_time_mean": time_value.mean(),
                    "flow_time_t1_fraction": (time_value == 1).float().mean(),
                    "solver_steps": total.new_tensor(float(args.solver_steps)),
                    "grad_norm": grad_norm if isinstance(grad_norm, torch.Tensor) else total.new_tensor(float(grad_norm)),
                    "base_grad_norm": base_grad_norm,
                    "condition_grad_norm": condition_grad_norm,
                    "base_grad_rms": base_grad_norm
                    / max(1, base_parameter_count) ** 0.5,
                    "condition_grad_rms": condition_grad_norm
                    / max(1, condition_parameter_count) ** 0.5,
                    "learning_rate": total.new_tensor(scheduler.get_last_lr()[0]),
                    **(
                        {
                            "base_learning_rate": total.new_tensor(
                                group_learning_rates["base"]
                            )
                        }
                        if "base" in group_learning_rates
                        else {}
                    ),
                    **(
                        {
                            "condition_learning_rate": total.new_tensor(
                                group_learning_rates["condition"]
                            )
                        }
                        if "condition" in group_learning_rates
                        else {}
                    ),
                }
                numeric = tensor_metrics_to_float(metrics, accelerator)
                logger.log(
                    global_step,
                    numeric,
                    samples_seen=global_step * args.batch_size * accelerator.num_processes * args.gradient_accumulation,
                )
                if accelerator.is_main_process:
                    print(json.dumps({"step": global_step, **numeric}), flush=True)

            if global_step % args.state_every == 0:
                state_dir = args.output / "state" / f"step-{global_step:07d}"
                accelerator.save_state(str(state_dir))
            if global_step % args.save_every == 0 and accelerator.is_main_process:
                save_weights(unwrapped, ema, args.output, global_step)

            if time.perf_counter() - start_time >= args.max_runtime_s:
                stop = True
            if global_step >= args.steps or stop:
                break

    accelerator.wait_for_everyone()
    unwrapped = accelerator.unwrap_model(flow)
    if accelerator.is_main_process:
        save_weights(unwrapped, ema, args.output, global_step)
        logger.log(global_step, {}, event="training_complete", stopped_by_runtime=stop)
    logger.close()
    accelerator.end_training()


if __name__ == "__main__":
    main()
