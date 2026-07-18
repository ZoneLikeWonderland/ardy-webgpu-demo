#!/usr/bin/env python
"""Accelerate training for the compact history encoder and motion decoder."""

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

from .data import ShardShuffleSampler, TeacherShardDataset
from .ema import ModelEMA, ema_99_percent_horizon
from .losses import (
    FSQRequantizer,
    decoder_feature_losses,
    encoder_distillation_losses,
    motion_quality_losses,
)
from .models import CodecStudentConfig, HistoryEncoderStudent, MotionDecoderStudent
from .runtime import (
    MetricLogger,
    load_motion_rep,
    load_safetensor_weights,
    save_safetensor_weights,
    tensor_metrics_to_float,
)


class CodecBundle(nn.Module):
    def __init__(self, config: CodecStudentConfig = CodecStudentConfig()) -> None:
        super().__init__()
        self.encoder = HistoryEncoderStudent(config)
        self.decoder = MotionDecoderStudent(config)

    def forward(
        self,
        encoder_body: torch.Tensor | None,
        decoder_latent: torch.Tensor,
        decoder_local_root: torch.Tensor,
        decoder_token_valid: torch.Tensor,
        run_encoder: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if run_encoder:
            if encoder_body is None:
                raise ValueError("encoder_body is required when run_encoder=True")
            encoded = self.encoder(encoder_body)
        else:
            encoded = decoder_latent.new_empty(0)
        return (
            encoded,
            self.decoder(decoder_latent, decoder_local_root, decoder_token_valid),
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path("ardy/checkpoints/ARDY-Core-RP-20FPS-Horizon40"),
    )
    parser.add_argument("--steps", type=int, default=100_000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--gradient-accumulation", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=5.0e-5)
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
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--warmup-steps", type=int, default=0)
    parser.add_argument("--mixed-precision", choices=["no", "fp16", "bf16"], default="bf16")
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
    parser.add_argument("--encoder-width", type=int, default=512)
    parser.add_argument("--encoder-blocks", type=int, default=4)
    parser.add_argument("--decoder-width", type=int, default=512)
    parser.add_argument("--decoder-blocks", type=int, default=8)
    parser.add_argument("--decoder-token-hidden", type=int, default=32)
    parser.add_argument("--expansion", type=int, default=2)
    parser.add_argument("--quality-every", type=int, default=1)
    parser.add_argument(
        "--quality-weight",
        type=float,
        default=1.0,
        help="Multiplier for differentiable FK/temporal/contact quality losses.",
    )
    parser.add_argument(
        "--joint-jerk-weight",
        type=float,
        default=0.25,
        help="Coefficient of physical joint third-difference matching inside quality loss.",
    )
    parser.add_argument("--rotation-velocity-weight", type=float, default=1.0)
    parser.add_argument("--rotation-acceleration-weight", type=float, default=0.5)
    parser.add_argument("--rotation-jerk-weight", type=float, default=0.25)
    parser.add_argument("--boundary-joint-velocity-weight", type=float, default=0.5)
    parser.add_argument("--boundary-joint-acceleration-weight", type=float, default=0.25)
    parser.add_argument("--boundary-joint-jerk-weight", type=float, default=0.1)
    parser.add_argument("--boundary-rotation-velocity-weight", type=float, default=0.5)
    parser.add_argument("--boundary-rotation-acceleration-weight", type=float, default=0.25)
    parser.add_argument("--boundary-rotation-jerk-weight", type=float, default=0.1)
    parser.add_argument("--rotation-geodesic-weight", type=float, default=0.25)
    parser.add_argument("--fk-weight", type=float, default=2.0)
    parser.add_argument("--joint-velocity-weight", type=float, default=1.0)
    parser.add_argument("--joint-acceleration-weight", type=float, default=0.5)
    parser.add_argument("--contact-weight", type=float, default=0.10)
    parser.add_argument("--foot-slide-weight", type=float, default=0.25)
    parser.add_argument("--log-every", type=int, default=250)
    parser.add_argument("--save-every", type=int, default=5_000)
    parser.add_argument(
        "--state-every",
        type=int,
        default=25_000,
        help="Save the larger optimizer/RNG resume state at this interval.",
    )
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--cache-shards", type=int, default=2)
    parser.add_argument(
        "--sample-order",
        choices=["shard_shuffle", "global_shuffle", "sequential"],
        default="shard_shuffle",
        help="Shard-local shuffle avoids random multi-megabyte shard reloads.",
    )
    parser.add_argument("--seed", type=int, default=20260714)
    parser.add_argument("--max-runtime-s", type=float, default=43_200)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--init-encoder", type=Path)
    parser.add_argument("--init-decoder", type=Path)
    parser.add_argument(
        "--expandable-init",
        action="store_true",
        help=(
            "Load matching tensors from a shallower same-width codec and initialize only "
            "the newly appended residual blocks as exact identities."
        ),
    )
    parser.add_argument(
        "--freeze-encoder",
        action="store_true",
        help="Keep the initialized encoder fixed while refining only the decoder.",
    )
    return parser.parse_args()


@torch.no_grad()
def initialize_codec_residuals_as_identity(module: nn.Module) -> None:
    """Initialize residual branches as identities before overlaying a checkpoint."""

    blocks = getattr(module, "blocks", None)
    if blocks is None:
        raise TypeError(f"unsupported expandable codec module: {type(module).__name__}")
    for block in blocks:
        if hasattr(block, "fc2"):
            nn.init.zeros_(block.fc2.weight)
            nn.init.zeros_(block.fc2.bias)
            continue
        if hasattr(block, "token_fc2") and hasattr(block, "channel_block"):
            nn.init.zeros_(block.token_fc2.weight)
            nn.init.zeros_(block.token_fc2.bias)
            nn.init.zeros_(block.channel_block.fc2.weight)
            nn.init.zeros_(block.channel_block.fc2.bias)
            continue
        raise TypeError(f"unsupported residual block: {type(block).__name__}")


def load_expandable_codec_weights(module: nn.Module, path: Path) -> list[str]:
    """Load a shallower same-width checkpoint without changing its function."""

    initialize_codec_residuals_as_identity(module)
    incompatible = load_safetensor_weights(module, path, strict=False)
    if incompatible.unexpected_keys:
        raise RuntimeError(
            f"unexpected keys while expanding {type(module).__name__}: "
            f"{incompatible.unexpected_keys}"
        )
    invalid_missing = [
        name for name in incompatible.missing_keys if not name.startswith("blocks.")
    ]
    if invalid_missing:
        raise RuntimeError(
            f"non-block tensors missing while expanding {type(module).__name__}: "
            f"{invalid_missing}"
        )
    return list(incompatible.missing_keys)


def save_weights(codec: CodecBundle, ema: ModelEMA, output: Path, step: int) -> None:
    checkpoint = output / "weights" / f"step-{step:07d}"
    save_safetensor_weights(codec.encoder, checkpoint / "encoder.safetensors")
    save_safetensor_weights(codec.decoder, checkpoint / "decoder.safetensors")
    full_state = codec.state_dict()
    ema_state = {name: value for name, value in full_state.items()}
    ema_state.update(ema.shadow)
    encoder_ema = {
        name.removeprefix("encoder."): value
        for name, value in ema_state.items()
        if name.startswith("encoder.")
    }
    decoder_ema = {
        name.removeprefix("decoder."): value
        for name, value in ema_state.items()
        if name.startswith("decoder.")
    }
    save_safetensor_weights(codec.encoder, checkpoint / "encoder_ema.safetensors", encoder_ema)
    save_safetensor_weights(codec.decoder, checkpoint / "decoder_ema.safetensors", decoder_ema)


def main() -> None:
    args = parse_args()
    nonnegative = {
        "quality_weight": args.quality_weight,
        "joint_jerk_weight": args.joint_jerk_weight,
        "rotation_velocity_weight": args.rotation_velocity_weight,
        "rotation_acceleration_weight": args.rotation_acceleration_weight,
        "rotation_jerk_weight": args.rotation_jerk_weight,
        "boundary_joint_velocity_weight": args.boundary_joint_velocity_weight,
        "boundary_joint_acceleration_weight": args.boundary_joint_acceleration_weight,
        "boundary_joint_jerk_weight": args.boundary_joint_jerk_weight,
        "boundary_rotation_velocity_weight": args.boundary_rotation_velocity_weight,
        "boundary_rotation_acceleration_weight": args.boundary_rotation_acceleration_weight,
        "boundary_rotation_jerk_weight": args.boundary_rotation_jerk_weight,
        "rotation_geodesic_weight": args.rotation_geodesic_weight,
        "fk_weight": args.fk_weight,
        "joint_velocity_weight": args.joint_velocity_weight,
        "joint_acceleration_weight": args.joint_acceleration_weight,
        "contact_weight": args.contact_weight,
        "foot_slide_weight": args.foot_slide_weight,
    }
    if any(value < 0 for value in nonnegative.values()):
        raise ValueError("quality and temporal weights must be non-negative")
    if args.lr_schedule == "step":
        if not 0 < args.lr_decay_step < args.steps:
            raise ValueError("step LR decay must occur strictly inside the training run")
        if not 0.0 < args.final_learning_rate <= args.learning_rate:
            raise ValueError("final learning rate must be in (0, initial learning rate]")
    if args.save_every < 1 or args.state_every < 1:
        raise ValueError("save intervals must be positive")
    if not 0.0 < args.ema_decay < 1.0:
        raise ValueError("--ema-decay must be in (0, 1)")
    if args.override_ema_decay_on_resume and args.resume is None:
        raise ValueError("--override-ema-decay-on-resume requires --resume")
    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
        gradient_accumulation_steps=args.gradient_accumulation,
        step_scheduler_with_optimizer=False,
    )
    set_seed(args.seed, device_specific=True)
    if min(
        args.encoder_width,
        args.encoder_blocks,
        args.decoder_width,
        args.decoder_blocks,
        args.decoder_token_hidden,
        args.expansion,
    ) < 1:
        raise ValueError("codec widths, block counts, token hidden size and expansion must be positive")
    if args.expandable_init and (args.init_encoder is None or args.init_decoder is None):
        raise ValueError("--expandable-init requires both --init-encoder and --init-decoder")
    codec_config = CodecStudentConfig(
        encoder_width=args.encoder_width,
        encoder_blocks=args.encoder_blocks,
        decoder_width=args.decoder_width,
        decoder_blocks=args.decoder_blocks,
        decoder_token_hidden=args.decoder_token_hidden,
        expansion=args.expansion,
    )
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
                    "ema_99_percent_horizon_steps": ema_99_percent_horizon(args.ema_decay),
                    "model_config": asdict(codec_config),
                },
                default=str,
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

    decoder_fields = (
        "decoder_latent",
        "decoder_global_root",
        "decoder_local_root",
        "decoder_token_valid",
        "target_body",
    )
    training_fields = (
        decoder_fields
        if args.freeze_encoder
        else (
            "history_hybrid",
            "encoder_body",
            "encoder_valid",
            *decoder_fields,
        )
    )
    dataset = TeacherShardDataset(
        args.data,
        cache_shards=args.cache_shards,
        fields=training_fields,
    )
    sampler = (
        ShardShuffleSampler(dataset, seed=args.seed)
        if args.sample_order == "shard_shuffle"
        else None
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
    codec = CodecBundle(codec_config)
    if args.init_encoder is not None:
        if args.expandable_init:
            load_expandable_codec_weights(codec.encoder, args.init_encoder)
        else:
            load_safetensor_weights(codec.encoder, args.init_encoder)
    if args.init_decoder is not None:
        if args.expandable_init:
            load_expandable_codec_weights(codec.decoder, args.init_decoder)
        else:
            load_safetensor_weights(codec.decoder, args.init_decoder)
    if args.freeze_encoder:
        codec.encoder.requires_grad_(False)
    optimizer = AdamW(
        [parameter for parameter in codec.parameters() if parameter.requires_grad],
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    if args.lr_schedule == "constant":
        scheduler = LambdaLR(optimizer, lambda _step: 1.0)
    else:
        final_ratio = args.final_learning_rate / args.learning_rate
        scheduler = LambdaLR(
            optimizer,
            lambda step: 1.0 if step < args.lr_decay_step else final_ratio,
        )
    codec, optimizer, loader, scheduler = accelerator.prepare(codec, optimizer, loader, scheduler)
    ema = ModelEMA(
        accelerator.unwrap_model(codec),
        decay=args.ema_decay,
        override_decay_on_load=args.override_ema_decay_on_resume,
    )
    accelerator.register_for_checkpointing(ema)
    if sampler is not None:
        accelerator.register_for_checkpointing(sampler)

    motion_rep = load_motion_rep(args.checkpoint_dir)
    quantizer = FSQRequantizer(args.checkpoint_dir / "stats/post_quantization").to(accelerator.device)
    logger = MetricLogger(args.output, enabled=accelerator.is_main_process)
    global_step = 0
    start_time = time.perf_counter()
    if args.resume is not None:
        accelerator.load_state(str(args.resume))
        try:
            global_step = int(args.resume.name.split("-")[-1])
        except ValueError:
            global_step = 0

    optimizer.zero_grad(set_to_none=True)
    stop = False
    optimizer_steps_skipped = 0
    while global_step < args.steps and not stop:
        for batch in loader:
            batch = {
                name: value.to(accelerator.device, non_blocking=True).float()
                for name, value in batch.items()
            }
            with accelerator.accumulate(codec):
                with accelerator.autocast():
                    encoder_prediction, decoder_prediction = codec(
                        batch.get("encoder_body"),
                        batch["decoder_latent"],
                        batch["decoder_local_root"],
                        batch["decoder_token_valid"],
                        run_encoder=not args.freeze_encoder,
                    )
                    if args.freeze_encoder:
                        encoder_losses = {}
                    else:
                        encoder_target = batch["history_hybrid"][..., 20:]
                        encoder_losses = encoder_distillation_losses(
                            encoder_prediction,
                            encoder_target,
                            batch["encoder_valid"],
                            quantizer,
                        )
                    decoder_losses = decoder_feature_losses(
                        decoder_prediction,
                        batch["target_body"],
                        batch["decoder_token_valid"],
                    )
                    total = decoder_losses["decoder_total"]
                    if not args.freeze_encoder:
                        total = total + encoder_losses["encoder_total"]

                quality_losses = {}
                if global_step % args.quality_every == 0:
                    quality_losses = motion_quality_losses(
                        decoder_prediction,
                        batch["target_body"],
                        batch["decoder_global_root"],
                        batch["decoder_token_valid"],
                        motion_rep,
                    )
                    quality_total = (
                        args.rotation_geodesic_weight
                        * quality_losses["rotation_geodesic"]
                        + args.rotation_velocity_weight
                        * quality_losses["rotation_velocity_l1"]
                        + args.rotation_acceleration_weight
                        * quality_losses["rotation_acceleration_l1"]
                        + args.rotation_jerk_weight
                        * quality_losses["rotation_jerk_l1"]
                        + args.boundary_rotation_velocity_weight
                        * quality_losses["rotation_boundary_velocity_l1"]
                        + args.boundary_rotation_acceleration_weight
                        * quality_losses["rotation_boundary_acceleration_l1"]
                        + args.boundary_rotation_jerk_weight
                        * quality_losses["rotation_boundary_jerk_l1"]
                        + args.fk_weight * quality_losses["fk_mpjpe"]
                        + args.joint_velocity_weight
                        * quality_losses["joint_velocity_l1"]
                        + args.joint_acceleration_weight
                        * quality_losses["joint_acceleration_l1"]
                        + args.joint_jerk_weight * quality_losses["joint_jerk_l1"]
                        + args.boundary_joint_velocity_weight
                        * quality_losses["joint_boundary_velocity_l1"]
                        + args.boundary_joint_acceleration_weight
                        * quality_losses["joint_boundary_acceleration_l1"]
                        + args.boundary_joint_jerk_weight
                        * quality_losses["joint_boundary_jerk_l1"]
                        + args.contact_weight * quality_losses["contact_physical_l1"]
                        + args.foot_slide_weight * quality_losses["foot_slide"]
                    )
                    total = total + args.quality_weight * quality_total
                    quality_losses["quality_total"] = quality_total

                accelerator.backward(total)
                if accelerator.sync_gradients:
                    grad_norm = accelerator.clip_grad_norm_(codec.parameters(), 1.0)
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
            unwrapped = accelerator.unwrap_model(codec)
            ema.update(unwrapped)

            if global_step % args.log_every == 0 or global_step == 1:
                metrics = {
                    **encoder_losses,
                    **decoder_losses,
                    **quality_losses,
                    "loss_total": total,
                    "grad_norm": grad_norm if isinstance(grad_norm, torch.Tensor) else total.new_tensor(float(grad_norm)),
                    "learning_rate": total.new_tensor(scheduler.get_last_lr()[0]),
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
    unwrapped = accelerator.unwrap_model(codec)
    if accelerator.is_main_process:
        save_weights(unwrapped, ema, args.output, global_step)
        logger.log(global_step, {}, event="training_complete", stopped_by_runtime=stop)
    logger.close()
    accelerator.end_training()


if __name__ == "__main__":
    main()
