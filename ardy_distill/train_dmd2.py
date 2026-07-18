#!/usr/bin/env python
"""Two-time-scale DMD2-style refinement for the conditional NFE=1 flow.

The fake score and temporal critic update every iteration.  The deployable
generator updates less frequently with a DMD surrogate, paired endpoint/path/
seam objectives and a non-saturating adversarial loss.  Teacher, fake score,
critic, encoder and decoder are training-only dependencies; only generator/EMA
weights are candidates for browser export.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import torch
from accelerate import Accelerator
from accelerate.utils import set_seed
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from ardy.model import load_model

from .codec_cli import add_codec_config_arguments, codec_config_from_args
from .data import (
    IndexedTeacherMixtureDataset,
    RandomTeacherMixtureDataset,
    ShardMixtureSampler,
    ShardShuffleSampler,
    TeacherShardDataset,
)
from .dmd2 import (
    TeacherScoreAdapter,
    discriminator_losses,
    diffusion_motion_critic_features,
    distribution_matching_loss,
    epsilon_to_x0,
    fake_score_loss,
    generator_adversarial_loss,
    normalized_diffusion_time,
    path_constraint_losses,
    q_sample,
    q_sample_with_exact_clean,
    sample_dmd_timesteps,
    seam_losses,
)
from .ema import ModelEMA, ema_99_percent_horizon
from .losses import (
    FSQRequantizer,
    decoder_feature_losses,
    deployment_decoder_roots,
    motion_quality_losses,
    physical_seam_losses,
    root_temporal_losses,
)
from .models import (
    FlowStudentConfig,
    HistoryEncoderStudent,
    MotionDecoderStudent,
    OneStepFlowStudent,
    ScoreBackboneCriticHead,
    TemporalCriticConfig,
    TemporalMotionCritic,
    project_root_trajectory,
)
from .runtime import (
    MetricLogger,
    load_motion_rep,
    load_safetensor_weights,
    save_safetensor_weights,
    tensor_metrics_to_float,
)
from .train_flow import endpoint_losses


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument(
        "--replay-data",
        type=Path,
        help="Optional replay teacher corpus mixed with the primary on-policy corpus.",
    )
    parser.add_argument("--primary-data-prob", type=float, default=0.5)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--generator", type=Path, required=True)
    parser.add_argument("--decoder", type=Path, required=True)
    parser.add_argument("--encoder", type=Path, required=True)
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path("ardy/checkpoints/ARDY-Core-RP-20FPS-Horizon40"),
    )
    parser.add_argument("--model", default="ARDY-Core-RP-20FPS-Horizon40")
    parser.add_argument(
        "--steps",
        type=int,
        default=2_000,
        help="Generator update count; distribution distillation is intentionally short",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--generator-update-ratio",
        type=int,
        default=5,
        help=(
            "Fake-score/critic iterations per generator update. DMD2 TTUR uses 5; "
            "--steps counts generator updates only."
        ),
    )
    parser.add_argument(
        "--warmup-iterations",
        type=int,
        default=200,
        help=(
            "Fake-score/critic-only iterations before the first generator update; "
            "the four-GPU direct-NFE1 probe selected 200."
        ),
    )
    parser.add_argument("--generator-learning-rate", type=float, default=1.0e-6)
    parser.add_argument("--score-learning-rate", type=float, default=5.0e-6)
    parser.add_argument("--critic-learning-rate", type=float, default=2.0e-6)
    parser.add_argument(
        "--critic-kind",
        choices=["score_backbone", "physical_features"],
        default="score_backbone",
        help=(
            "score_backbone matches DMD2 by classifying the fake-score "
            "bottleneck; physical_features retains the earlier ablation."
        ),
    )
    parser.add_argument(
        "--guidance-classifier-weight",
        type=float,
        default=1.0e-2,
        help=(
            "Classifier weight inside the joint fake-score guidance update. "
            "Only used by --critic-kind score_backbone."
        ),
    )
    parser.add_argument(
        "--critic-max-timestep",
        type=int,
        default=9,
        help=(
            "Inclusive maximum diffusion-GAN timestep. The released ARDY "
            "schedule has ten steps, so 9 exposes the critic to the full "
            "noise range instead of letting it saturate on clean motion."
        ),
    )
    parser.add_argument(
        "--critic-exact-clean-probability",
        type=float,
        default=0.0,
        help=(
            "Probability that the adversarial classifier sees the exact "
            "sigma-zero teacher/generator endpoint. ARDY timestep zero is "
            "still noised, so this is a separate branch."
        ),
    )
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument(
        "--lr-schedule",
        choices=["constant"],
        default="constant",
        help="The short distribution-distillation stage only permits fixed LR.",
    )
    parser.add_argument("--mixed-precision", choices=["no", "fp16", "bf16"], default="bf16")
    parser.add_argument(
        "--frozen-codec-dtype",
        choices=["fp16", "fp32"],
        default="fp16",
        help=(
            "Arithmetic dtype for the frozen encoder/decoder used inside DMD2; "
            "fp16 matches the browser deployment path while trainable modules can remain BF16."
        ),
    )
    parser.add_argument(
        "--ema-decay",
        type=float,
        default=0.995,
        help="Constant FP32 EMA decay; 0.995 has an approximately 919-step 99%% horizon.",
    )
    parser.add_argument(
        "--override-ema-decay-on-resume",
        action="store_true",
        help=(
            "Keep the command-line --ema-decay after loading a resume state instead of "
            "restoring the checkpoint's decay. All learned and optimizer states still resume."
        ),
    )
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--cache-shards", type=int, default=2)
    parser.add_argument(
        "--sample-order",
        choices=["shard_shuffle", "global_shuffle", "sequential"],
        default="shard_shuffle",
    )
    parser.add_argument("--flow-width", type=int, default=384)
    parser.add_argument("--flow-heads", type=int, default=6)
    parser.add_argument("--flow-trunk-blocks", type=int, default=4)
    parser.add_argument("--flow-body-blocks", type=int, default=2)
    parser.add_argument(
        "--flow-root-smoothing-passes",
        type=int,
        default=0,
        help=(
            "Differentiable fixed root projection inside the deployed NFE1 "
            "endpoint. Zero exactly reproduces legacy checkpoints."
        ),
    )
    parser.add_argument(
        "--flow-root-projection-kind",
        choices=["binomial", "cubic_controls"],
        default="binomial",
        help="Zero-parameter root endpoint parameterization used by deployment.",
    )
    parser.add_argument(
        "--flow-root-control-points",
        type=int,
        default=10,
        help="Predicted root control-frame count for cubic_controls.",
    )
    add_codec_config_arguments(parser)
    parser.add_argument("--critic-width", type=int, default=256)
    parser.add_argument("--critic-blocks", type=int, default=4)
    parser.add_argument("--cfg-constraint", type=float, default=1.5)
    parser.add_argument("--dmd-min-timestep", type=int, default=0)
    parser.add_argument("--dmd-max-timestep", type=int, default=8)
    parser.add_argument("--dmd-exact-max-probability", type=float, default=0.0)
    parser.add_argument("--dmd-high-noise-probability", type=float, default=0.0)
    parser.add_argument("--dmd-grad-clip", type=float, default=0.0)
    parser.add_argument(
        "--paired-weight",
        type=float,
        default=0.05,
        help="Weak conditional anchor; set to zero for pure DMD2/adv ablation.",
    )
    parser.add_argument("--dmd-weight", type=float, default=0.05)
    parser.add_argument("--adversarial-weight", type=float, default=0.005)
    parser.add_argument("--decoder-weight", type=float, default=0.25)
    parser.add_argument("--quality-weight", type=float, default=0.10)
    parser.add_argument("--path-weight", type=float, default=0.05)
    parser.add_argument("--seam-weight", type=float, default=0.10)
    parser.add_argument("--physical-seam-weight", type=float, default=0.02)
    parser.add_argument("--root-temporal-weight", type=float, default=0.10)
    parser.add_argument("--quality-every", type=int, default=1)
    parser.add_argument("--log-every", type=int, default=10, help="Iteration interval")
    parser.add_argument("--save-every", type=int, default=100, help="Generator update interval")
    parser.add_argument("--seed", type=int, default=20260714)
    parser.add_argument("--max-runtime-s", type=float, default=7_200)
    parser.add_argument("--resume", type=Path)
    return parser.parse_args()


def validate_args(args: argparse.Namespace, num_diffusion_steps: int = 10) -> None:
    positive = {
        "steps": args.steps,
        "batch_size": args.batch_size,
        "generator_update_ratio": args.generator_update_ratio,
        "quality_every": args.quality_every,
        "log_every": args.log_every,
        "save_every": args.save_every,
    }
    for name, value in positive.items():
        if value < 1:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if not 0.0 < args.ema_decay < 1.0:
        raise ValueError("--ema-decay must be in (0, 1)")
    if args.override_ema_decay_on_resume and args.resume is None:
        raise ValueError("--override-ema-decay-on-resume requires --resume")
    if args.steps > 5_000:
        raise ValueError(
            "distribution distillation is capped at 5000 generator updates; "
            "select a checkpoint from short paired evaluations instead of extending it"
        )
    if args.warmup_iterations < 0:
        raise ValueError("--warmup-iterations must be non-negative")
    if args.flow_root_smoothing_passes < 0:
        raise ValueError("--flow-root-smoothing-passes must be non-negative")
    if not 2 <= args.flow_root_control_points <= 40:
        raise ValueError("--flow-root-control-points must be within [2, 40]")
    if args.replay_data is not None and not 0.0 < args.primary_data_prob < 1.0:
        raise ValueError("--primary-data-prob must be strictly between zero and one")
    if args.replay_data is None and args.primary_data_prob != 0.5:
        raise ValueError("--primary-data-prob only applies with --replay-data")
    learning_rates = {
        "generator_learning_rate": args.generator_learning_rate,
        "score_learning_rate": args.score_learning_rate,
        "critic_learning_rate": args.critic_learning_rate,
    }
    for name, value in learning_rates.items():
        if not 0.0 < value <= 1.0e-5:
            raise ValueError(
                f"--{name.replace('_', '-')} must be in the short-stage range (0, 1e-5]"
            )
    nonnegative = {
        "weight_decay": args.weight_decay,
        "paired_weight": args.paired_weight,
        "dmd_weight": args.dmd_weight,
        "adversarial_weight": args.adversarial_weight,
        "decoder_weight": args.decoder_weight,
        "quality_weight": args.quality_weight,
        "path_weight": args.path_weight,
        "seam_weight": args.seam_weight,
        "physical_seam_weight": args.physical_seam_weight,
        "root_temporal_weight": args.root_temporal_weight,
        "guidance_classifier_weight": args.guidance_classifier_weight,
    }
    for name, value in nonnegative.items():
        if value < 0:
            raise ValueError(f"--{name.replace('_', '-')} must be non-negative")
    if not 0 <= args.dmd_min_timestep <= args.dmd_max_timestep < num_diffusion_steps:
        raise ValueError(
            f"DMD timesteps must satisfy 0 <= min <= max < {num_diffusion_steps}"
        )
    if not 0 <= args.critic_max_timestep < num_diffusion_steps:
        raise ValueError(
            "critic max timestep must satisfy "
            f"0 <= max < {num_diffusion_steps}"
        )
    if not 0.0 <= args.critic_exact_clean_probability <= 1.0:
        raise ValueError("--critic-exact-clean-probability must be in [0, 1]")
    if not 0.0 <= args.dmd_exact_max_probability <= 1.0:
        raise ValueError("--dmd-exact-max-probability must be in [0, 1]")
    if not 0.0 <= args.dmd_high_noise_probability <= 1.0:
        raise ValueError("--dmd-high-noise-probability must be in [0, 1]")
    if (
        args.dmd_exact_max_probability + args.dmd_high_noise_probability
        > 1.0
    ):
        raise ValueError("DMD timestep mixture probabilities exceed one")


def deployment_history(
    batch: dict[str, torch.Tensor],
    encoder: HistoryEncoderStudent,
    quantizer: FSQRequantizer,
    accelerator: Accelerator,
) -> torch.Tensor:
    frozen_codec_dtype = next(encoder.parameters()).dtype
    with torch.no_grad(), torch.autocast(
        device_type=accelerator.device.type, enabled=False
    ):
        encoded = encoder(
            batch["encoder_body"].to(dtype=frozen_codec_dtype)
        ).float()
    encoded = quantizer(encoded, ste=False)
    valid = batch["has_history"].unsqueeze(-1)
    return torch.cat(
        [
            batch["history_hybrid"][..., :20] * valid,
            encoded * valid,
        ],
        dim=-1,
    )


def generate_endpoint(
    generator: OneStepFlowStudent,
    batch: dict[str, torch.Tensor],
    history_hybrid: torch.Tensor,
) -> torch.Tensor:
    flow_time = torch.ones_like(batch["has_history"])
    velocity = generator(
        batch["initial_noise"],
        history_hybrid,
        batch["path_condition"],
        batch["first_heading"],
        batch["has_history"],
        flow_time,
    )
    endpoint = batch["initial_noise"] - velocity
    unwrapped = generator.module if hasattr(generator, "module") else generator
    config = unwrapped.config
    return project_root_trajectory(
        endpoint,
        int(config.root_smoothing_passes),
        kind=config.root_projection_kind,
        control_points=int(config.root_control_points),
        basis=unwrapped.root_projection_basis,
    )


def predict_fake_epsilon(
    fake_score: OneStepFlowStudent,
    xt: torch.Tensor,
    timesteps: torch.Tensor,
    num_timesteps: int,
    batch: dict[str, torch.Tensor],
    history_hybrid: torch.Tensor,
) -> torch.Tensor:
    score_time = normalized_diffusion_time(
        timesteps, num_timesteps, dtype=xt.dtype
    ).to(device=xt.device)
    return fake_score(
        xt,
        history_hybrid,
        batch["path_condition"],
        batch["first_heading"],
        batch["has_history"],
        score_time,
    )


def score_backbone_critic_logits(
    fake_score: OneStepFlowStudent,
    critic: ScoreBackboneCriticHead,
    xt: torch.Tensor,
    timesteps: torch.Tensor,
    num_timesteps: int,
    batch: dict[str, torch.Tensor],
    history_hybrid: torch.Tensor,
) -> torch.Tensor:
    """Classify a noised endpoint through the conditional fake-score trunk."""

    score_time = normalized_diffusion_time(
        timesteps, num_timesteps, dtype=xt.dtype
    ).to(device=xt.device)
    result = fake_score(
        xt,
        history_hybrid,
        batch["path_condition"],
        batch["first_heading"],
        batch["has_history"],
        score_time,
        return_features=True,
    )
    if not isinstance(result, tuple):
        raise RuntimeError("fake-score backbone did not return classifier features")
    _prediction, generation_features = result
    return critic(generation_features)


def parameter_gradient_l2(parameters) -> torch.Tensor:
    """Return an FP32 pre-clipping gradient norm for one parameter group."""

    squared = None
    device = None
    for parameter in parameters:
        device = parameter.device
        if parameter.grad is None:
            continue
        value = parameter.grad.detach().float().square().sum()
        squared = value if squared is None else squared + value
    if squared is None:
        return torch.zeros((), device=device or torch.device("cpu"))
    return squared.sqrt()


def decode_endpoint(
    endpoint: torch.Tensor,
    history_hybrid: torch.Tensor,
    batch: dict[str, torch.Tensor],
    decoder: MotionDecoderStudent,
    quantizer: FSQRequantizer,
    motion_rep,
    ste: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    generated_latent = quantizer(endpoint[..., 20:], ste=ste)
    latent_sequence = torch.cat(
        [history_hybrid[..., 20:], generated_latent], dim=1
    )
    root_sequence, decoder_local_root = deployment_decoder_roots(
        endpoint,
        history_hybrid,
        batch["has_history"],
        motion_rep,
    )
    frozen_codec_dtype = next(decoder.parameters()).dtype
    with torch.autocast(device_type=endpoint.device.type, enabled=False):
        decoded = decoder(
            latent_sequence.to(dtype=frozen_codec_dtype),
            decoder_local_root.to(dtype=frozen_codec_dtype),
            batch["decoder_token_valid"].to(dtype=frozen_codec_dtype),
        ).float()
    return decoded, root_sequence


def boundary_motion_quality_total(
    losses: dict[str, torch.Tensor],
) -> torch.Tensor:
    """Apply the decoder-100k boundary profile to deployed NFE=1 motion."""

    return (
        0.25 * losses["rotation_geodesic"]
        + 0.90 * losses["rotation_velocity_l1"]
        + 0.45 * losses["rotation_acceleration_l1"]
        + 0.22 * losses["rotation_jerk_l1"]
        + 0.38 * losses["rotation_boundary_velocity_l1"]
        + 0.30 * losses["rotation_boundary_acceleration_l1"]
        + 0.19 * losses["rotation_boundary_jerk_l1"]
        + 2.00 * losses["fk_mpjpe"]
        + 1.50 * losses["joint_velocity_l1"]
        + 0.75 * losses["joint_acceleration_l1"]
        + 0.35 * losses["joint_jerk_l1"]
        + 0.64 * losses["joint_boundary_velocity_l1"]
        + 0.44 * losses["joint_boundary_acceleration_l1"]
        + 0.28 * losses["joint_boundary_jerk_l1"]
        + 0.10 * losses["contact_physical_l1"]
        + 0.25 * losses["foot_slide"]
    )


def save_weights(
    generator: OneStepFlowStudent,
    fake_score: OneStepFlowStudent,
    critic: torch.nn.Module,
    ema: ModelEMA,
    output: Path,
    generator_step: int,
) -> None:
    checkpoint = output / "weights" / f"step-{generator_step:07d}"
    save_safetensor_weights(generator, checkpoint / "flow.safetensors")
    ema_state = generator.state_dict()
    ema_state.update(ema.shadow)
    save_safetensor_weights(
        generator, checkpoint / "flow_ema.safetensors", ema_state
    )
    save_safetensor_weights(fake_score, checkpoint / "fake_score.safetensors")
    save_safetensor_weights(critic, checkpoint / "critic.safetensors")


def main() -> None:
    args = parse_args()
    validate_args(args)
    # ARDY-Core is trained with the fixed ten-step cosine schedule.  The
    # loaded teacher below verifies the count before the first training batch;
    # it is needed here to size the critic's discrete time embedding.
    num_diffusion_steps = 10
    # We call every scheduler explicitly after its corresponding optimizer.
    # Accelerate's default wrapper advances a scheduler once per process when
    # batches are not split, which would compress a four-GPU schedule by 4x.
    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
        step_scheduler_with_optimizer=False,
    )
    set_seed(args.seed, device_specific=True)
    frozen_codec_dtype = {
        "fp16": torch.float16,
        "fp32": torch.float32,
    }[args.frozen_codec_dtype]
    if frozen_codec_dtype == torch.float16 and accelerator.device.type != "cuda":
        raise ValueError("--frozen-codec-dtype=fp16 requires a CUDA training device")

    flow_config = FlowStudentConfig(
        width=args.flow_width,
        heads=args.flow_heads,
        trunk_blocks=args.flow_trunk_blocks,
        body_blocks=args.flow_body_blocks,
        root_smoothing_passes=args.flow_root_smoothing_passes,
        root_projection_kind=args.flow_root_projection_kind,
        root_control_points=args.flow_root_control_points,
    )
    codec_config = codec_config_from_args(args)
    generator = OneStepFlowStudent(flow_config)
    load_safetensor_weights(generator, args.generator)
    fake_score = OneStepFlowStudent(flow_config)
    fake_score.load_state_dict(generator.state_dict())
    shared_score_critic = args.critic_kind == "score_backbone"
    if shared_score_critic:
        critic = ScoreBackboneCriticHead(
            width=flow_config.width,
            blocks=args.critic_blocks,
        )
        critic_config: dict[str, object] = {
            "architecture": "fake_score_bottleneck_head",
            "backbone_width": flow_config.width,
            "generation_tokens": flow_config.generation_tokens,
            "blocks": args.critic_blocks,
            "joint_guidance_optimizer": True,
        }
        critic_input = "noised_hybrid_via_fake_score_bottleneck"
    else:
        temporal_critic_config = TemporalCriticConfig(
            width=args.critic_width,
            blocks=args.critic_blocks,
            num_diffusion_steps=num_diffusion_steps,
        )
        critic = TemporalMotionCritic(temporal_critic_config)
        critic_config = asdict(temporal_critic_config)
        critic_input = "timestep_conditioned_diffusion_motion_features"

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
                    "generator_config": asdict(flow_config),
                    "codec_model_config": asdict(codec_config),
                    "critic_config": critic_config,
                    "algorithm": {
                        "deployment_generator": "single_call_x1_to_x0",
                        "fake_score_target": "epsilon",
                        "teacher_target": "x0",
                        "generator_update_schedule": "two_time_scale",
                        "guidance_updates_per_generator": args.generator_update_ratio,
                        "critic_loss": "softplus_nonsaturating",
                        "critic_input": critic_input,
                        "critic_timestep_range": [0, args.critic_max_timestep],
                        "critic_exact_clean_probability": (
                            args.critic_exact_clean_probability
                        ),
                        "dmd_timestep_sampling": (
                            "uniform"
                            if args.dmd_exact_max_probability == 0
                            and args.dmd_high_noise_probability == 0
                            else "mixture"
                        ),
                        "quality_profile": "decoder_boundary_100k",
                        "scheduler_step_semantics": "one_step_per_optimizer_update",
                    },
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
        "decoder_global_root",
        "decoder_token_valid",
        "target_body",
        "encoder_body",
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
    generator_optimizer = AdamW(
        generator.parameters(),
        lr=args.generator_learning_rate,
        weight_decay=args.weight_decay,
    )
    if shared_score_critic:
        score_optimizer = AdamW(
            [
                {
                    "params": list(fake_score.parameters()),
                    "lr": args.score_learning_rate,
                },
                {
                    "params": list(critic.parameters()),
                    "lr": args.critic_learning_rate,
                },
            ],
            weight_decay=args.weight_decay,
        )
        critic_optimizer = None
    else:
        score_optimizer = AdamW(
            fake_score.parameters(),
            lr=args.score_learning_rate,
            weight_decay=args.weight_decay,
        )
        critic_optimizer = AdamW(
            critic.parameters(),
            lr=args.critic_learning_rate,
            weight_decay=args.weight_decay,
        )
    generator_scheduler = LambdaLR(generator_optimizer, lambda _step: 1.0)
    score_scheduler = LambdaLR(score_optimizer, lambda _step: 1.0)
    if shared_score_critic:
        critic_scheduler = None
        (
            generator,
            fake_score,
            critic,
            generator_optimizer,
            score_optimizer,
            loader,
            generator_scheduler,
            score_scheduler,
        ) = accelerator.prepare(
            generator,
            fake_score,
            critic,
            generator_optimizer,
            score_optimizer,
            loader,
            generator_scheduler,
            score_scheduler,
        )
    else:
        if critic_optimizer is None:
            raise RuntimeError("physical critic optimizer was not created")
        critic_scheduler = LambdaLR(critic_optimizer, lambda _step: 1.0)
        (
            generator,
            fake_score,
            critic,
            generator_optimizer,
            score_optimizer,
            critic_optimizer,
            loader,
            generator_scheduler,
            score_scheduler,
            critic_scheduler,
        ) = accelerator.prepare(
            generator,
            fake_score,
            critic,
            generator_optimizer,
            score_optimizer,
            critic_optimizer,
            loader,
            generator_scheduler,
            score_scheduler,
            critic_scheduler,
        )
    ema = ModelEMA(
        accelerator.unwrap_model(generator),
        decay=args.ema_decay,
        override_decay_on_load=args.override_ema_decay_on_resume,
    )
    accelerator.register_for_checkpointing(ema)
    if sampler is not None:
        accelerator.register_for_checkpointing(sampler)

    decoder = MotionDecoderStudent(codec_config).to(
        device=accelerator.device, dtype=frozen_codec_dtype
    ).eval()
    load_safetensor_weights(decoder, args.decoder)
    decoder.requires_grad_(False)
    encoder = HistoryEncoderStudent(codec_config).to(
        device=accelerator.device, dtype=frozen_codec_dtype
    ).eval()
    load_safetensor_weights(encoder, args.encoder)
    encoder.requires_grad_(False)
    quantizer = FSQRequantizer(
        args.checkpoint_dir / "stats/post_quantization"
    ).to(accelerator.device)
    motion_rep = load_motion_rep(args.checkpoint_dir)

    teacher = load_model(
        args.model,
        device=str(accelerator.device),
        text_encoder=False,
        checkpoints_dir=str(args.checkpoint_dir.parent),
    ).eval()
    teacher.requires_grad_(False)
    teacher_score = TeacherScoreAdapter(
        teacher, cfg_constraint=args.cfg_constraint
    )
    alphas_cumprod = teacher_score.alphas_cumprod.detach()
    num_diffusion_steps = int(alphas_cumprod.numel())
    validate_args(args, num_diffusion_steps=num_diffusion_steps)
    logger = MetricLogger(args.output, enabled=accelerator.is_main_process)

    iteration = 0
    generator_step = 0
    if args.resume is not None:
        accelerator.load_state(str(args.resume))
        progress_path = args.resume / "progress.json"
        if progress_path.exists():
            progress = json.loads(progress_path.read_text(encoding="utf-8"))
            iteration = int(progress.get("iteration", 0))
            generator_step = int(progress.get("generator_step", 0))

    start_time = time.perf_counter()
    stop = False
    while generator_step < args.steps and not stop:
        for batch in loader:
            batch = {
                name: value.to(
                    accelerator.device, non_blocking=True
                ).float()
                for name, value in batch.items()
            }
            history_hybrid = deployment_history(
                batch, encoder, quantizer, accelerator
            )
            local_batch_size = batch["clean_generation"].shape[0]
            with torch.no_grad(), accelerator.autocast():
                fake_endpoint_detached = generate_endpoint(
                    generator, batch, history_hybrid
                ).float()

            # Guidance update.  In the DMD2-aligned mode the classifier reads
            # the fake-score bottleneck and both losses update that shared
            # backbone in one optimizer step.  The generator endpoint remains
            # detached throughout the guidance turn.
            score_optimizer.zero_grad(set_to_none=True)
            score_t = torch.randint(
                0,
                num_diffusion_steps,
                (local_batch_size,),
                device=accelerator.device,
            )
            score_noise = torch.randn_like(fake_endpoint_detached)
            score_xt = q_sample(
                fake_endpoint_detached,
                score_t,
                score_noise,
                alphas_cumprod,
            )
            with accelerator.autocast():
                predicted_epsilon = predict_fake_epsilon(
                    fake_score,
                    score_xt,
                    score_t,
                    num_diffusion_steps,
                    batch,
                    history_hybrid,
                )
                score_losses = fake_score_loss(
                    predicted_epsilon, score_noise
                )

            critic_t = torch.randint(
                0,
                args.critic_max_timestep + 1,
                (local_batch_size,),
                device=accelerator.device,
            )
            critic_exact_clean = (
                torch.rand(local_batch_size, device=accelerator.device)
                < args.critic_exact_clean_probability
            )
            critic_model_t = torch.where(
                critic_exact_clean,
                torch.zeros_like(critic_t),
                critic_t,
            )
            if shared_score_critic:
                real_critic_xt = q_sample_with_exact_clean(
                    batch["clean_generation"],
                    critic_t,
                    torch.randn_like(batch["clean_generation"]),
                    alphas_cumprod,
                    critic_exact_clean,
                )
                fake_critic_xt = q_sample_with_exact_clean(
                    fake_endpoint_detached,
                    critic_t,
                    torch.randn_like(fake_endpoint_detached),
                    alphas_cumprod,
                    critic_exact_clean,
                )
                with accelerator.autocast():
                    real_logits = score_backbone_critic_logits(
                        fake_score,
                        critic,
                        real_critic_xt,
                        critic_model_t,
                        num_diffusion_steps,
                        batch,
                        history_hybrid,
                    )
                    fake_logits = score_backbone_critic_logits(
                        fake_score,
                        critic,
                        fake_critic_xt,
                        critic_model_t,
                        num_diffusion_steps,
                        batch,
                        history_hybrid,
                    )
                    critic_losses = discriminator_losses(
                        real_logits, fake_logits
                    )
                    guidance_total = (
                        score_losses["fake_score_total"]
                        + args.guidance_classifier_weight
                        * critic_losses["critic_total"]
                    )
                accelerator.backward(guidance_total)
                score_grad_norm = parameter_gradient_l2(
                    fake_score.parameters()
                )
                critic_grad_norm = parameter_gradient_l2(
                    critic.parameters()
                )
                guidance_grad_norm = accelerator.clip_grad_norm_(
                    [*fake_score.parameters(), *critic.parameters()], 1.0
                )
                score_optimizer.step()
                score_scheduler.step()
            else:
                accelerator.backward(score_losses["fake_score_total"])
                score_grad_norm = accelerator.clip_grad_norm_(
                    fake_score.parameters(), 1.0
                )
                score_optimizer.step()
                score_scheduler.step()
                if critic_optimizer is None or critic_scheduler is None:
                    raise RuntimeError("physical critic optimizer is unavailable")

                critic_optimizer.zero_grad(set_to_none=True)
                with torch.no_grad(), accelerator.autocast():
                    decoded_fake_detached, fake_root_detached = decode_endpoint(
                        fake_endpoint_detached,
                        history_hybrid,
                        batch,
                        decoder,
                        quantizer,
                        motion_rep,
                        ste=False,
                    )
                with torch.no_grad():
                    real_features = diffusion_motion_critic_features(
                        batch["target_body"][:, -40:],
                        batch["decoder_global_root"][:, -40:],
                        critic_t,
                        alphas_cumprod,
                        motion_rep,
                        exact_clean=critic_exact_clean,
                    )
                    fake_features = diffusion_motion_critic_features(
                        decoded_fake_detached[:, -40:],
                        fake_root_detached[:, -40:],
                        critic_t,
                        alphas_cumprod,
                        motion_rep,
                        exact_clean=critic_exact_clean,
                    )
                with accelerator.autocast():
                    real_logits = critic(
                        real_features,
                        history_hybrid,
                        batch["path_condition"],
                        batch["first_heading"],
                        batch["has_history"],
                        critic_model_t,
                    )
                    fake_logits = critic(
                        fake_features,
                        history_hybrid,
                        batch["path_condition"],
                        batch["first_heading"],
                        batch["has_history"],
                        critic_model_t,
                    )
                    critic_losses = discriminator_losses(
                        real_logits, fake_logits
                    )
                accelerator.backward(critic_losses["critic_total"])
                critic_grad_norm = accelerator.clip_grad_norm_(
                    critic.parameters(), 1.0
                )
                critic_optimizer.step()
                critic_scheduler.step()
                guidance_total = score_losses["fake_score_total"]
                guidance_grad_norm = score_grad_norm

            should_update_generator = (
                iteration >= args.warmup_iterations
                and (iteration - args.warmup_iterations)
                % args.generator_update_ratio
                == 0
            )
            iteration_generator_metrics: dict[str, torch.Tensor] = {}
            if should_update_generator:
                generator_optimizer.zero_grad(set_to_none=True)
                with accelerator.autocast():
                    endpoint = generate_endpoint(
                        generator, batch, history_hybrid
                    )
                    paired_losses = endpoint_losses(
                        endpoint, batch["clean_generation"]
                    )
                    path_losses = path_constraint_losses(
                        endpoint,
                        batch["path_condition"],
                        batch["has_history"],
                    )
                    decoded, predicted_root = decode_endpoint(
                        endpoint,
                        history_hybrid,
                        batch,
                        decoder,
                        quantizer,
                        motion_rep,
                        ste=True,
                    )
                    decoder_losses = decoder_feature_losses(
                        decoded,
                        batch["target_body"],
                        batch["decoder_token_valid"],
                    )
                    boundary_losses = seam_losses(
                        endpoint,
                        decoded,
                        history_hybrid,
                        batch["decoder_global_root"],
                        batch["target_body"],
                        batch["has_history"],
                    )

                dmd_t = sample_dmd_timesteps(
                    local_batch_size,
                    accelerator.device,
                    minimum=args.dmd_min_timestep,
                    maximum=args.dmd_max_timestep,
                    exact_max_probability=args.dmd_exact_max_probability,
                    high_noise_probability=args.dmd_high_noise_probability,
                )
                dmd_noise = torch.randn_like(endpoint)
                dmd_xt = q_sample(
                    endpoint.detach(), dmd_t, dmd_noise, alphas_cumprod
                )
                teacher_x0 = teacher_score.predict_x0(
                    dmd_xt.float(),
                    history_hybrid.float(),
                    batch["path_condition"].float(),
                    batch["first_heading"].float(),
                    batch["has_history"].float(),
                    dmd_t,
                )
                with torch.no_grad(), accelerator.autocast():
                    fake_epsilon = predict_fake_epsilon(
                        fake_score,
                        dmd_xt,
                        dmd_t,
                        num_diffusion_steps,
                        batch,
                        history_hybrid,
                    )
                fake_x0 = epsilon_to_x0(
                    dmd_xt,
                    fake_epsilon,
                    dmd_t,
                    alphas_cumprod,
                )
                dmd_losses = distribution_matching_loss(
                    endpoint,
                    teacher_x0,
                    fake_x0,
                    dmd_xt,
                    grad_clip=args.dmd_grad_clip,
                )

                temporal_root_losses = root_temporal_losses(
                    endpoint,
                    batch["clean_generation"],
                    motion_rep,
                )
                physical_boundary_losses = physical_seam_losses(
                    decoded,
                    batch["target_body"],
                    predicted_root,
                    batch["decoder_global_root"],
                    batch["has_history"],
                    motion_rep,
                )
                quality_losses: dict[str, torch.Tensor] = {}
                if generator_step % args.quality_every == 0:
                    quality_losses = motion_quality_losses(
                        decoded,
                        batch["target_body"],
                        predicted_root,
                        batch["decoder_token_valid"],
                        motion_rep,
                        target_normalized_global_root=batch[
                            "decoder_global_root"
                        ],
                    )
                    quality_losses["quality_total"] = (
                        boundary_motion_quality_total(quality_losses)
                    )
                else:
                    quality_losses["quality_total"] = endpoint.new_zeros(())

                unwrapped_fake_score = accelerator.unwrap_model(fake_score)
                unwrapped_critic = accelerator.unwrap_model(critic)
                adversarial_parameters = [
                    *unwrapped_fake_score.parameters(),
                    *unwrapped_critic.parameters(),
                ]
                for parameter in adversarial_parameters:
                    parameter.requires_grad_(False)
                try:
                    adversarial_t = torch.randint(
                        0,
                        args.critic_max_timestep + 1,
                        (local_batch_size,),
                        device=accelerator.device,
                    )
                    adversarial_exact_clean = (
                        torch.rand(local_batch_size, device=accelerator.device)
                        < args.critic_exact_clean_probability
                    )
                    adversarial_model_t = torch.where(
                        adversarial_exact_clean,
                        torch.zeros_like(adversarial_t),
                        adversarial_t,
                    )
                    if shared_score_critic:
                        adversarial_xt = q_sample_with_exact_clean(
                            endpoint,
                            adversarial_t,
                            torch.randn_like(endpoint),
                            alphas_cumprod,
                            adversarial_exact_clean,
                        )
                        with accelerator.autocast():
                            adversarial_logits = score_backbone_critic_logits(
                                unwrapped_fake_score,
                                unwrapped_critic,
                                adversarial_xt,
                                adversarial_model_t,
                                num_diffusion_steps,
                                batch,
                                history_hybrid,
                            )
                    else:
                        adversarial_features = diffusion_motion_critic_features(
                            decoded[:, -40:],
                            predicted_root[:, -40:],
                            adversarial_t,
                            alphas_cumprod,
                            motion_rep,
                            exact_clean=adversarial_exact_clean,
                        )
                        with accelerator.autocast():
                            adversarial_logits = unwrapped_critic(
                            adversarial_features,
                            history_hybrid,
                            batch["path_condition"],
                            batch["first_heading"],
                            batch["has_history"],
                            adversarial_model_t,
                        )
                    adversarial_loss = generator_adversarial_loss(
                        adversarial_logits
                    )
                finally:
                    for parameter in adversarial_parameters:
                        parameter.requires_grad_(True)

                generator_total = (
                    args.paired_weight * paired_losses["endpoint_total"]
                    + args.dmd_weight * dmd_losses["dmd_total"]
                    + args.adversarial_weight * adversarial_loss
                    + args.decoder_weight * decoder_losses["decoder_total"]
                    + args.quality_weight * quality_losses["quality_total"]
                    + args.path_weight * path_losses["path_constraint_mse"]
                    + args.seam_weight * boundary_losses["seam_total"]
                    + args.physical_seam_weight
                    * physical_boundary_losses["physical_seam_total"]
                    + args.root_temporal_weight
                    * temporal_root_losses["root_temporal_total"]
                )
                accelerator.backward(generator_total)
                generator_grad_norm = accelerator.clip_grad_norm_(
                    generator.parameters(), 1.0
                )
                generator_optimizer.step()
                generator_scheduler.step()
                generator_step += 1
                ema.update(accelerator.unwrap_model(generator))
                iteration_generator_metrics = {
                    **paired_losses,
                    **path_losses,
                    **decoder_losses,
                    **boundary_losses,
                    **physical_boundary_losses,
                    **temporal_root_losses,
                    **dmd_losses,
                    **quality_losses,
                    "generator_adversarial": adversarial_loss,
                    "generator_total": generator_total,
                    "generator_grad_norm": generator_grad_norm
                    if isinstance(generator_grad_norm, torch.Tensor)
                    else generator_total.new_tensor(float(generator_grad_norm)),
                    "dmd_timestep_mean": dmd_t.float().mean(),
                    "adversarial_timestep_mean": adversarial_t.float().mean(),
                    "adversarial_exact_clean_fraction": (
                        adversarial_exact_clean.float().mean()
                    ),
                }

            iteration += 1
            # Generator updates are offset from the iteration counter by the
            # warmup and update ratio, so periodic iteration logging alone can
            # systematically miss every DMD/adversarial metric.
            if should_update_generator or iteration % args.log_every == 0 or iteration == 1:
                zero = score_losses["fake_score_total"].new_zeros(())
                metrics = {
                    **score_losses,
                    **critic_losses,
                    **iteration_generator_metrics,
                    "guidance_total": guidance_total,
                    "guidance_grad_norm": guidance_grad_norm
                    if isinstance(guidance_grad_norm, torch.Tensor)
                    else zero + float(guidance_grad_norm),
                    "score_grad_norm": score_grad_norm
                    if isinstance(score_grad_norm, torch.Tensor)
                    else zero + float(score_grad_norm),
                    "critic_grad_norm": critic_grad_norm
                    if isinstance(critic_grad_norm, torch.Tensor)
                    else zero + float(critic_grad_norm),
                    "score_timestep_mean": score_t.float().mean(),
                    "critic_timestep_mean": critic_t.float().mean(),
                    "critic_exact_clean_fraction": (
                        critic_exact_clean.float().mean()
                    ),
                    "generator_learning_rate": zero
                    + generator_scheduler.get_last_lr()[0],
                    "score_learning_rate": zero
                    + score_scheduler.get_last_lr()[0],
                    "critic_learning_rate": zero
                    + (
                        score_scheduler.get_last_lr()[1]
                        if shared_score_critic
                        else critic_scheduler.get_last_lr()[0]
                    ),
                    "generator_step": zero + generator_step,
                    "generator_updated": zero
                    + float(should_update_generator),
                    "warmup_active": zero
                    + float(iteration <= args.warmup_iterations),
                }
                numeric = tensor_metrics_to_float(metrics, accelerator)
                logger.log(
                    iteration,
                    numeric,
                    samples_seen=iteration
                    * args.batch_size
                    * accelerator.num_processes,
                )
                if accelerator.is_main_process:
                    print(
                        json.dumps(
                            {"iteration": iteration, **numeric}
                        ),
                        flush=True,
                    )

            if (
                should_update_generator
                and generator_step % args.save_every == 0
            ):
                state_dir = (
                    args.output
                    / "state"
                    / f"step-{generator_step:07d}-iter-{iteration:07d}"
                )
                accelerator.save_state(str(state_dir))
                if accelerator.is_main_process:
                    (state_dir / "progress.json").write_text(
                        json.dumps(
                            {
                                "iteration": iteration,
                                "generator_step": generator_step,
                            },
                            indent=2,
                        )
                        + "\n",
                        encoding="utf-8",
                    )
                    save_weights(
                        accelerator.unwrap_model(generator),
                        accelerator.unwrap_model(fake_score),
                        accelerator.unwrap_model(critic),
                        ema,
                        args.output,
                        generator_step,
                    )

            if time.perf_counter() - start_time >= args.max_runtime_s:
                stop = True
            if generator_step >= args.steps or stop:
                break

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        save_weights(
            accelerator.unwrap_model(generator),
            accelerator.unwrap_model(fake_score),
            accelerator.unwrap_model(critic),
            ema,
            args.output,
            generator_step,
        )
        logger.log(
            iteration,
            {},
            event="training_complete",
            generator_step=generator_step,
            stopped_by_runtime=stop,
        )
    logger.close()
    accelerator.end_training()


if __name__ == "__main__":
    main()
