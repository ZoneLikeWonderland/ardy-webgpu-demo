#!/usr/bin/env python
"""Flow-native DMD2 + teacher-feature adversarial refinement for ARDY.

Stage one fits a conditional rectified-flow model to the released ARDY
10-step+CFG endpoints.  This stage-two trainer freezes that fitted flow as the
real score teacher, clones a fake score for the generator distribution, and
updates the deployable generator directly at NFE=1.  No cosine-diffusion
coefficient or released denoiser score is used here.
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

from .codec_cli import add_codec_config_arguments, codec_config_from_args
from .data import (
    IndexedTeacherMixtureDataset,
    ShardMixtureSampler,
    ShardShuffleSampler,
    TeacherShardDataset,
)
from .dmd2 import (
    discriminator_losses,
    distribution_matching_loss,
    fake_score_loss,
    generator_adversarial_loss,
    path_constraint_losses,
    seam_losses,
)
from .ema import ModelEMA, ema_99_percent_horizon
from .flow_matching import flow_velocity_to_x0, make_flow_pair, sample_flow_time
from .losses import (
    FSQRequantizer,
    decoder_feature_losses,
    deployment_decoder_roots,
    fsq_endpoint_diagnostics,
    fsq_endpoint_losses,
    motion_quality_losses,
    physical_seam_losses,
    root_temporal_losses,
)
from .models import (
    FlowStudentConfig,
    HistoryEncoderStudent,
    IndependentScoreBackboneCriticHeads,
    MotionDecoderStudent,
    OneStepFlowStudent,
    ScoreBackboneCriticHead,
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
from .train_flow import endpoint_losses


class TrainingCounters:
    def __init__(self) -> None:
        self.iterations = 0
        self.guidance_updates = 0
        self.score_updates = 0
        self.critic_updates = 0
        self.generator_updates = 0
        self.update_ratio_origin_generator = -1
        self.update_ratio_origin_score = -1
        self.update_ratio_origin_critic = -1

    def state_dict(self) -> dict[str, int]:
        return {
            "iterations": self.iterations,
            "guidance_updates": self.guidance_updates,
            "score_updates": self.score_updates,
            "critic_updates": self.critic_updates,
            "generator_updates": self.generator_updates,
            "update_ratio_origin_generator": self.update_ratio_origin_generator,
            "update_ratio_origin_score": self.update_ratio_origin_score,
            "update_ratio_origin_critic": self.update_ratio_origin_critic,
        }

    def load_state_dict(self, state: dict[str, int]) -> None:
        self.iterations = int(state["iterations"])
        self.guidance_updates = int(state["guidance_updates"])
        # Checkpoints written before score/critic ratios were decoupled contain
        # one joint guidance counter.  Each old guidance batch updated both
        # optimizers exactly once, so this is an exact migration rather than an
        # estimate.
        self.score_updates = int(state.get("score_updates", self.guidance_updates))
        self.critic_updates = int(state.get("critic_updates", self.guidance_updates))
        self.generator_updates = int(state["generator_updates"])
        self.update_ratio_origin_generator = int(
            state.get("update_ratio_origin_generator", -1)
        )
        self.update_ratio_origin_score = int(
            state.get("update_ratio_origin_score", -1)
        )
        self.update_ratio_origin_critic = int(
            state.get("update_ratio_origin_critic", -1)
        )

    @property
    def has_update_ratio_origin(self) -> bool:
        return (
            self.update_ratio_origin_generator >= 0
            and self.update_ratio_origin_score >= 0
            and self.update_ratio_origin_critic >= 0
        )

    def reset_update_ratio_origin(self) -> None:
        """Start a new ratio schedule without retroactive catch-up updates."""

        self.update_ratio_origin_generator = self.generator_updates
        self.update_ratio_origin_score = self.score_updates
        self.update_ratio_origin_critic = self.critic_updates


def guidance_update_plan(
    counters: TrainingCounters,
    *,
    warmup_updates: int,
    score_updates_per_generator: int,
    critic_updates_per_generator: int,
) -> tuple[bool, bool, bool, int, int]:
    """Return score/D work due before the next generator update.

    The ratio origin is anchored to the state being resumed.  This matters for
    controlled branches: changing from 1:1:1 to 1:2:1 at generator step 300
    must request two *future* score updates, not retroactively catch up the
    first 300 generator steps.
    """

    in_warmup = (
        counters.score_updates < warmup_updates
        or counters.critic_updates < warmup_updates
    )
    if in_warmup:
        return (
            counters.score_updates < warmup_updates,
            counters.critic_updates < warmup_updates,
            True,
            warmup_updates,
            warmup_updates,
        )

    if not counters.has_update_ratio_origin:
        counters.reset_update_ratio_origin()
    generator_updates_since_origin = (
        counters.generator_updates - counters.update_ratio_origin_generator
    )
    if generator_updates_since_origin < 0:
        raise ValueError("generator counter predates the update-ratio origin")
    score_target = (
        counters.update_ratio_origin_score
        + (generator_updates_since_origin + 1) * score_updates_per_generator
    )
    critic_target = (
        counters.update_ratio_origin_critic
        + (generator_updates_since_origin + 1) * critic_updates_per_generator
    )
    return (
        counters.score_updates < score_target,
        counters.critic_updates < critic_target,
        False,
        score_target,
        critic_target,
    )


def override_constant_learning_rate(
    optimizer: torch.optim.Optimizer,
    scheduler: object,
    learning_rate: float,
) -> None:
    """Change only the LR after a full-state resume of a constant-LR run.

    Accelerate restores both optimizer param-group LRs and LambdaLR base LRs.
    Updating only the optimizer would therefore be undone by the next scheduler
    step.  Keep Adam moments and every other restored state intact while making
    the requested experimental LR effective immediately and persistently.
    """

    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive")
    for group in optimizer.param_groups:
        group["lr"] = learning_rate
        group["initial_lr"] = learning_rate

    raw_scheduler = getattr(scheduler, "scheduler", scheduler)
    if not hasattr(raw_scheduler, "base_lrs"):
        raise TypeError("constant-LR override requires a scheduler with base_lrs")
    raw_scheduler.base_lrs = [learning_rate] * len(optimizer.param_groups)
    if hasattr(raw_scheduler, "_last_lr"):
        raw_scheduler._last_lr = [learning_rate] * len(optimizer.param_groups)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument(
        "--text-features",
        type=Path,
        help="Complete Qwen prompt-feature table for a text-conditioned generator.",
    )
    parser.add_argument(
        "--heading-condition-features",
        type=int,
        choices=[0, 3],
        default=0,
    )
    parser.add_argument(
        "--replay-data",
        type=Path,
        help="Optional original teacher corpus mixed with the primary corrective corpus.",
    )
    parser.add_argument(
        "--primary-data-prob",
        type=float,
        default=0.5,
        help="Per-batch probability for --data when --replay-data is set.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--teacher-flow", type=Path, required=True)
    parser.add_argument("--generator", type=Path, required=True)
    parser.add_argument("--encoder", type=Path, required=True)
    parser.add_argument("--decoder", type=Path, required=True)
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path("ardy/checkpoints/ARDY-Core-RP-20FPS-Horizon40"),
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=1_000,
        help="Generator optimizer updates; guidance/critic updates are counted separately.",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--guidance-updates-per-generator",
        type=int,
        default=1,
        help=(
            "Backward-compatible joint score/D ratio. Independent ratio flags "
            "default to this value when omitted."
        ),
    )
    parser.add_argument(
        "--score-updates-per-generator",
        type=int,
        help="Independent fake-score optimizer updates per generator update.",
    )
    parser.add_argument(
        "--critic-updates-per-generator",
        type=int,
        help="Independent discriminator optimizer updates per generator update.",
    )
    parser.add_argument("--warmup-guidance-updates", type=int, default=200)
    parser.add_argument("--generator-learning-rate", type=float, default=1.0e-6)
    parser.add_argument("--score-learning-rate", type=float, default=5.0e-6)
    parser.add_argument("--critic-learning-rate", type=float, default=1.0e-6)
    parser.add_argument(
        "--override-learning-rates-on-resume",
        action="store_true",
        help=(
            "After loading the complete resume state, keep Adam moments/EMA/RNG "
            "but replace all three optimizer and constant-scheduler LRs with the "
            "command-line values. Required for an honest resumed LR sweep."
        ),
    )
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--mixed-precision", choices=["no", "fp16", "bf16"], default="bf16")
    parser.add_argument(
        "--frozen-codec-dtype",
        choices=["fp16", "fp32"],
        default="fp16",
        help="Run the frozen codec in deployment FP16 while flow/score training remains BF16.",
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
    parser.add_argument("--generator-grad-clip", type=float, default=0.01)
    parser.add_argument("--score-grad-clip", type=float, default=1.0)
    parser.add_argument("--critic-grad-clip", type=float, default=1.0)
    parser.add_argument("--flow-width", type=int, default=512)
    parser.add_argument("--flow-heads", type=int, default=8)
    parser.add_argument("--flow-trunk-blocks", type=int, default=8)
    parser.add_argument("--flow-body-blocks", type=int, default=8)
    parser.add_argument("--flow-root-smoothing-passes", type=int, default=0)
    parser.add_argument("--critic-blocks", type=int, default=2)
    parser.add_argument(
        "--critic-head-mode",
        choices=["shared", "independent"],
        default="shared",
        help=(
            "Use one checkpoint-compatible classifier across all taps, or a "
            "separate LADD classifier per tap. Independent mode requires a "
            "matching expanded resume checkpoint when resuming shared state."
        ),
    )
    parser.add_argument(
        "--critic-feature-tap",
        choices=["trunk_final", "body_pre", "body_mid", "body_final"],
        default="body_final",
        help=(
            "Frozen teacher generation-token representation consumed by the "
            "training-only critic. All taps retain shape [B,T,width]."
        ),
    )
    parser.add_argument(
        "--critic-feature-taps",
        nargs="+",
        choices=["trunk_final", "body_pre", "body_mid", "body_final"],
        help=(
            "Optional ordered multi-level feature set. Overrides "
            "--critic-feature-tap when provided. Head sharing is controlled "
            "by --critic-head-mode."
        ),
    )
    parser.add_argument(
        "--critic-tap-aggregation",
        choices=["mean_loss", "mean_logit"],
        default="mean_loss",
        help=(
            "For multiple taps, either average independent per-tap GAN losses "
            "or average logits before applying the GAN loss."
        ),
    )
    parser.add_argument(
        "--log-critic-tap-gradient-rms",
        action="store_true",
        help=(
            "At logged generator steps, measure each tap's unweighted "
            "adversarial gradient RMS with respect to the generated endpoint."
        ),
    )
    parser.add_argument(
        "--generator-component-gradient-every",
        type=int,
        default=0,
        help=(
            "At every Nth generator update, measure unweighted and weighted "
            "endpoint gradients for every generator-loss group. Zero "
            "disables the diagnostic. This never changes the optimized loss."
        ),
    )
    add_codec_config_arguments(parser)

    parser.add_argument("--time-exact-t1-probability", type=float, default=0.70)
    parser.add_argument("--time-high-noise-probability", type=float, default=0.20)
    parser.add_argument(
        "--critic-time-exact-t1-probability",
        type=float,
        help=(
            "Critic-only exact-t=1 mass. Defaults to --time-exact-t1-probability "
            "for backward compatibility; set to zero to avoid identical real/fake "
            "inputs at exact t=1 when paired noise is shared."
        ),
    )
    parser.add_argument(
        "--critic-time-high-noise-probability",
        type=float,
        help=(
            "Critic-only high-noise mixture mass. Defaults to "
            "--time-high-noise-probability."
        ),
    )
    parser.add_argument(
        "--critic-time-upper-bound",
        type=float,
        default=1.0,
        help="Scale critic flow times into (approximately) [0, upper_bound].",
    )
    parser.add_argument(
        "--adversarial-time-sampler",
        choices=["score", "critic"],
        default="score",
        help=(
            "Time distribution used when the trained critic supplies generator "
            "adversarial gradients. 'score' preserves legacy behavior; 'critic' "
            "matches the critic's own informative time support."
        ),
    )
    parser.add_argument("--dmd-grad-clip", type=float, default=0.0)
    parser.add_argument("--dmd-weight", type=float, default=1.0)
    parser.add_argument("--adversarial-weight", type=float, default=1.0e-3)
    parser.add_argument("--paired-weight", type=float, default=0.10)
    parser.add_argument("--fsq-weight", type=float, default=0.10)
    parser.add_argument("--path-weight", type=float, default=0.01)
    parser.add_argument("--decoder-weight", type=float, default=0.01)
    parser.add_argument("--root-temporal-weight", type=float, default=0.01)
    parser.add_argument("--quality-weight", type=float, default=0.01)
    parser.add_argument("--seam-weight", type=float, default=0.001)
    parser.add_argument("--physical-seam-weight", type=float, default=0.001)

    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--cache-shards", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260714)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--save-every", type=int, default=100)
    parser.add_argument("--state-every", type=int, default=500)
    parser.add_argument("--max-runtime-s", type=float, default=7_200)
    parser.add_argument("--resume", type=Path)
    parser.add_argument(
        "--reset-update-ratio-on-resume",
        action="store_true",
        help=(
            "Anchor a newly requested score/D ratio at the loaded counters. "
            "Weights, optimizer moments, EMA, sampler and RNG remain restored."
        ),
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.score_updates_per_generator is None:
        args.score_updates_per_generator = args.guidance_updates_per_generator
    if args.critic_updates_per_generator is None:
        args.critic_updates_per_generator = args.guidance_updates_per_generator
    if args.critic_time_exact_t1_probability is None:
        args.critic_time_exact_t1_probability = args.time_exact_t1_probability
    if args.critic_time_high_noise_probability is None:
        args.critic_time_high_noise_probability = args.time_high_noise_probability
    args.critic_feature_taps = tuple(
        args.critic_feature_taps
        if args.critic_feature_taps is not None
        else (args.critic_feature_tap,)
    )
    if len(set(args.critic_feature_taps)) != len(args.critic_feature_taps):
        raise ValueError("--critic-feature-taps must not contain duplicates")
    positive = {
        "steps": args.steps,
        "batch_size": args.batch_size,
        "guidance_updates_per_generator": args.guidance_updates_per_generator,
        "score_updates_per_generator": args.score_updates_per_generator,
        "critic_updates_per_generator": args.critic_updates_per_generator,
        "log_every": args.log_every,
        "save_every": args.save_every,
        "state_every": args.state_every,
    }
    for name, value in positive.items():
        if value < 1:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if args.steps > 5_000:
        raise ValueError("flow DMD2 is capped at 5000 generator updates")
    if args.warmup_guidance_updates < 0:
        raise ValueError("--warmup-guidance-updates must be non-negative")
    if args.generator_component_gradient_every < 0:
        raise ValueError("--generator-component-gradient-every must be non-negative")
    learning_rates = (
        args.generator_learning_rate,
        args.score_learning_rate,
        args.critic_learning_rate,
    )
    if any(rate <= 0 or rate > 1.0e-5 for rate in learning_rates):
        raise ValueError("all DMD2 learning rates must be in (0, 1e-5]")
    if args.weight_decay != 0:
        raise ValueError("flow DMD2 uses weight decay 0")
    if not 0.0 < args.ema_decay < 1.0:
        raise ValueError("--ema-decay must be in (0, 1)")
    if args.override_ema_decay_on_resume and args.resume is None:
        raise ValueError("--override-ema-decay-on-resume requires --resume")
    if args.override_learning_rates_on_resume and args.resume is None:
        raise ValueError("--override-learning-rates-on-resume requires --resume")
    if args.reset_update_ratio_on_resume and args.resume is None:
        raise ValueError("--reset-update-ratio-on-resume requires --resume")
    if args.replay_data is not None and not 0.0 < args.primary_data_prob < 1.0:
        raise ValueError("--primary-data-prob must be strictly between zero and one")
    if args.replay_data is None and args.primary_data_prob != 0.5:
        raise ValueError("--primary-data-prob only applies with --replay-data")
    if args.flow_root_smoothing_passes < 0:
        raise ValueError("root smoothing passes must be non-negative")
    if args.critic_blocks < 0:
        raise ValueError("critic blocks must be non-negative")
    probability_pairs = {
        "score/dmd": (
            args.time_exact_t1_probability,
            args.time_high_noise_probability,
        ),
        "critic": (
            args.critic_time_exact_t1_probability,
            args.critic_time_high_noise_probability,
        ),
    }
    for name, probabilities in probability_pairs.items():
        if (
            any(value < 0 or value > 1 for value in probabilities)
            or sum(probabilities) > 1
        ):
            raise ValueError(
                f"{name} flow-time mixture probabilities must be in [0,1] "
                "and sum to <= 1"
            )
    if not 0.0 < args.critic_time_upper_bound <= 1.0:
        raise ValueError("--critic-time-upper-bound must be in (0,1]")
    nonnegative = {
        "generator_grad_clip": args.generator_grad_clip,
        "score_grad_clip": args.score_grad_clip,
        "critic_grad_clip": args.critic_grad_clip,
        "dmd_grad_clip": args.dmd_grad_clip,
        "dmd_weight": args.dmd_weight,
        "adversarial_weight": args.adversarial_weight,
        "paired_weight": args.paired_weight,
        "fsq_weight": args.fsq_weight,
        "path_weight": args.path_weight,
        "decoder_weight": args.decoder_weight,
        "root_temporal_weight": args.root_temporal_weight,
        "quality_weight": args.quality_weight,
        "seam_weight": args.seam_weight,
        "physical_seam_weight": args.physical_seam_weight,
    }
    if any(value < 0 for value in nonnegative.values()):
        raise ValueError("loss weights and gradient clips must be non-negative")


def nfe1_endpoint(
    model: torch.nn.Module,
    noise: torch.Tensor,
    history_hybrid: torch.Tensor,
    path_condition: torch.Tensor,
    first_heading: torch.Tensor,
    has_history: torch.Tensor,
    text_feature: torch.Tensor | None = None,
    heading_condition: torch.Tensor | None = None,
    *,
    root_smoothing_passes: int,
) -> torch.Tensor:
    time_value = torch.ones_like(has_history)
    velocity = model(
        noise,
        history_hybrid,
        path_condition,
        first_heading,
        has_history,
        time_value,
        text_feature,
        heading_condition,
    )
    endpoint = noise - velocity
    return project_root_trajectory(endpoint, root_smoothing_passes)


def flow_teacher_features(
    teacher: OneStepFlowStudent,
    clean: torch.Tensor,
    noise: torch.Tensor,
    time_value: torch.Tensor,
    history_hybrid: torch.Tensor,
    path_condition: torch.Tensor,
    first_heading: torch.Tensor,
    has_history: torch.Tensor,
    text_feature: torch.Tensor | None = None,
    heading_condition: torch.Tensor | None = None,
    *,
    feature_tap: str | tuple[str, ...] = "body_final",
) -> tuple[torch.Tensor, torch.Tensor | dict[str, torch.Tensor]]:
    noisy, _ = make_flow_pair(clean, noise, time_value)
    velocity, features = teacher(
        noisy,
        history_hybrid,
        path_condition,
        first_heading,
        has_history,
        time_value,
        text_feature,
        heading_condition,
        return_features=True,
        feature_tap=feature_tap,
    )
    return flow_velocity_to_x0(noisy, velocity, time_value), features


def critic_feature_request(taps: tuple[str, ...]) -> str | tuple[str, ...]:
    """Keep the historical single-tap execution path bit-compatible."""

    return taps[0] if len(taps) == 1 else taps


def critic_feature_map(
    features: torch.Tensor | dict[str, torch.Tensor],
    taps: tuple[str, ...],
) -> dict[str, torch.Tensor]:
    if isinstance(features, torch.Tensor):
        if len(taps) != 1:
            raise ValueError("a feature tensor can only represent one critic tap")
        return {taps[0]: features}
    if tuple(features) != taps:
        raise ValueError(
            f"critic feature keys {tuple(features)} do not match requested taps {taps}"
        )
    return features


def critic_logits(
    critic: torch.nn.Module,
    features: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Apply either one shared head or one independently trained head per tap."""

    unwrapped = getattr(critic, "module", critic)
    if getattr(unwrapped, "expects_feature_map", False):
        logits = critic(features)
        if tuple(logits) != tuple(features):
            raise ValueError("independent critic logits changed feature ordering")
        return logits
    return {tap: critic(value) for tap, value in features.items()}


def aggregate_discriminator_losses(
    real_logits: dict[str, torch.Tensor],
    fake_logits: dict[str, torch.Tensor],
    aggregation: str,
) -> dict[str, torch.Tensor]:
    if not real_logits or tuple(real_logits) != tuple(fake_logits):
        raise ValueError("real/fake critic taps must be non-empty and identical")
    per_tap = {
        tap: discriminator_losses(real_logits[tap], fake_logits[tap])
        for tap in real_logits
    }
    if len(per_tap) == 1:
        # Avoid even an extra stack/mean in the control path so its optimizer
        # trajectory can be checked byte-for-byte against historical runs.
        combined = dict(next(iter(per_tap.values())))
    elif aggregation == "mean_loss":
        combined = {
            name: torch.stack([losses[name] for losses in per_tap.values()]).mean()
            for name in next(iter(per_tap.values()))
        }
    elif aggregation == "mean_logit":
        combined = discriminator_losses(
            torch.stack(list(real_logits.values())).mean(dim=0),
            torch.stack(list(fake_logits.values())).mean(dim=0),
        )
    else:
        raise ValueError(f"unsupported critic tap aggregation: {aggregation}")
    for tap, losses in per_tap.items():
        for name, value in losses.items():
            suffix = name.removeprefix("critic_")
            combined[f"critic_tap_{tap}_{suffix}"] = value
    return combined


def aggregate_generator_adversarial_loss(
    fake_logits: dict[str, torch.Tensor],
    aggregation: str,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if not fake_logits:
        raise ValueError("generator adversarial logits must not be empty")
    per_tap = {
        tap: generator_adversarial_loss(logits)
        for tap, logits in fake_logits.items()
    }
    if len(per_tap) == 1:
        combined = next(iter(per_tap.values()))
    elif aggregation == "mean_loss":
        combined = torch.stack(list(per_tap.values())).mean()
    elif aggregation == "mean_logit":
        combined = generator_adversarial_loss(
            torch.stack(list(fake_logits.values())).mean(dim=0)
        )
    else:
        raise ValueError(f"unsupported critic tap aggregation: {aggregation}")
    return combined, per_tap


def endpoint_component_gradient_diagnostics(
    generated: torch.Tensor,
    total: torch.Tensor,
    components: dict[str, tuple[torch.Tensor, float]],
) -> dict[str, torch.Tensor]:
    """Measure generator-loss directions without modifying parameter gradients.

    Loss coefficients are not comparable when their raw gradient scales and
    directions differ. Compute each component gradient with respect to the
    generated endpoint, then report raw and coefficient-weighted RMS,
    alignment with the total update, and all pairwise raw-direction cosines.
    ``autograd.grad`` does not populate ``.grad``; retaining the graph lets the
    caller perform the normal distributed backward afterwards.
    """

    if not components:
        raise ValueError("generator gradient diagnostics require components")
    if not generated.requires_grad:
        raise ValueError("generated endpoint must require gradients")

    gradients: dict[str, torch.Tensor] = {}
    weighted: dict[str, torch.Tensor] = {}
    for name, (loss, coefficient) in components.items():
        if not name or not name.replace("_", "").isalnum():
            raise ValueError(f"invalid generator component name: {name!r}")
        raw_gradient = torch.autograd.grad(
            loss,
            generated,
            retain_graph=True,
            create_graph=False,
            allow_unused=True,
        )[0]
        weighted_gradient = torch.autograd.grad(
            loss * float(coefficient),
            generated,
            retain_graph=True,
            create_graph=False,
            allow_unused=True,
        )[0]
        if raw_gradient is None:
            raw_gradient = torch.zeros_like(generated)
        if weighted_gradient is None:
            weighted_gradient = torch.zeros_like(generated)
        gradients[name] = raw_gradient.detach().float()
        # Differentiate the weighted scalar directly. With BF16 endpoints,
        # scaling an already-quantized raw gradient in FP32 does not reproduce
        # the gradient that the real total loss sends through autograd.
        weighted[name] = weighted_gradient.detach().float()

    total_gradient = torch.autograd.grad(
        total,
        generated,
        retain_graph=True,
        create_graph=False,
        allow_unused=False,
    )[0].detach().float()
    summed_gradient = torch.zeros_like(total_gradient)
    for gradient in weighted.values():
        summed_gradient.add_(gradient)

    epsilon = total_gradient.new_tensor(torch.finfo(torch.float32).eps)

    def rms(value: torch.Tensor) -> torch.Tensor:
        return value.square().mean().sqrt()

    def cosine(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        left_flat = left.reshape(-1)
        right_flat = right.reshape(-1)
        denominator = left_flat.norm() * right_flat.norm()
        return torch.dot(left_flat, right_flat) / denominator.clamp_min(epsilon)

    total_rms = rms(total_gradient)
    reconstruction_error = summed_gradient - total_gradient
    metrics: dict[str, torch.Tensor] = {
        "generator_component_total_endpoint_grad_rms": total_rms,
        "generator_component_sum_endpoint_grad_rms": rms(summed_gradient),
        "generator_component_sum_vs_total_endpoint_grad_max_abs": (
            reconstruction_error.abs().max()
        ),
        "generator_component_sum_vs_total_endpoint_grad_relative_rms": (
            rms(reconstruction_error) / total_rms.clamp_min(epsilon)
        ),
    }
    for name in components:
        raw_gradient = gradients[name]
        weighted_gradient = weighted[name]
        weighted_rms = rms(weighted_gradient)
        metrics[f"generator_component_{name}_endpoint_grad_rms"] = rms(raw_gradient)
        metrics[f"generator_component_{name}_weighted_endpoint_grad_rms"] = (
            weighted_rms
        )
        metrics[f"generator_component_{name}_weighted_to_total_rms_ratio"] = (
            weighted_rms / total_rms.clamp_min(epsilon)
        )
        metrics[
            f"generator_component_{name}_weighted_vs_total_endpoint_cosine"
        ] = cosine(weighted_gradient, total_gradient)

    names = list(components)
    for left_index, left in enumerate(names):
        for right in names[left_index + 1 :]:
            metrics[
                f"generator_component_{left}_vs_{right}_endpoint_cosine"
            ] = cosine(gradients[left], gradients[right])
    return metrics


def save_weights(
    generator: OneStepFlowStudent,
    fake_score: OneStepFlowStudent,
    critic: torch.nn.Module,
    ema: ModelEMA,
    output: Path,
    step: int,
) -> None:
    checkpoint = output / "weights" / f"step-{step:07d}"
    save_safetensor_weights(generator, checkpoint / "flow.safetensors")
    ema_state = generator.state_dict()
    ema_state.update(ema.shadow)
    save_safetensor_weights(generator, checkpoint / "flow_ema.safetensors", ema_state)
    save_safetensor_weights(fake_score, checkpoint / "fake_score.safetensors")
    save_safetensor_weights(critic, checkpoint / "critic.safetensors")


def main() -> None:
    args = parse_args()
    validate_args(args)
    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
        step_scheduler_with_optimizer=False,
    )
    set_seed(args.seed, device_specific=True)

    text_feature_dim = 0
    text_feature_manifest = None
    if args.text_features is not None:
        text_feature_manifest = json.loads(
            (args.text_features / "manifest.json").read_text(encoding="utf-8")
        )
        if (
            text_feature_manifest.get("encoder") != "qwen"
            or not text_feature_manifest.get("complete")
        ):
            raise ValueError("--text-features must be a complete Qwen feature table")
        text_feature_dim = int(text_feature_manifest["feature_dim"])

    flow_config = FlowStudentConfig(
        width=args.flow_width,
        heads=args.flow_heads,
        trunk_blocks=args.flow_trunk_blocks,
        body_blocks=args.flow_body_blocks,
        text_feature_dim=text_feature_dim,
        heading_condition_features=args.heading_condition_features,
        root_smoothing_passes=args.flow_root_smoothing_passes,
    )
    codec_config = codec_config_from_args(args)
    generator = OneStepFlowStudent(flow_config)
    load_safetensor_weights(generator, args.generator)
    fake_score = OneStepFlowStudent(flow_config)
    fake_score.load_state_dict(generator.state_dict())
    if args.critic_head_mode == "shared":
        critic = ScoreBackboneCriticHead(
            width=flow_config.width,
            blocks=args.critic_blocks,
        )
    else:
        critic = IndependentScoreBackboneCriticHeads(
            width=flow_config.width,
            taps=args.critic_feature_taps,
            blocks=args.critic_blocks,
        )

    teacher_flow = OneStepFlowStudent(flow_config).to(accelerator.device).eval()
    load_safetensor_weights(teacher_flow, args.teacher_flow)
    teacher_flow.requires_grad_(False)
    frozen_codec_dtype = {
        "fp16": torch.float16,
        "fp32": torch.float32,
    }[args.frozen_codec_dtype]
    if frozen_codec_dtype == torch.float16 and accelerator.device.type != "cuda":
        raise ValueError("--frozen-codec-dtype=fp16 requires a CUDA training device")
    encoder = HistoryEncoderStudent(codec_config).to(
        device=accelerator.device, dtype=frozen_codec_dtype
    ).eval()
    decoder = MotionDecoderStudent(codec_config).to(
        device=accelerator.device, dtype=frozen_codec_dtype
    ).eval()
    load_safetensor_weights(encoder, args.encoder)
    load_safetensor_weights(decoder, args.decoder)
    encoder.requires_grad_(False)
    decoder.requires_grad_(False)

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
                    "flow_config": asdict(flow_config),
                    "codec_config": asdict(codec_config),
                    "algorithm": {
                        "noise_path": "x_t=(1-t)*x0+t*epsilon",
                        "prediction": "velocity=epsilon-x0",
                        "x0_reconstruction": "x0=x_t-t*velocity",
                        "real_score_teacher": "frozen_stage1_flow",
                        "fake_score_training_distribution": "nfe1_generator_outputs",
                        "critic_features": {
                            "backbone": "frozen_stage1_flow",
                            "tokens": "generation",
                            "taps": list(args.critic_feature_taps),
                            "head_mode": args.critic_head_mode,
                            "shared_head": args.critic_head_mode == "shared",
                            "aggregation": args.critic_tap_aggregation,
                        },
                        "score_time_sampling": {
                            "exact_t1_probability": args.time_exact_t1_probability,
                            "high_noise_probability": args.time_high_noise_probability,
                            "upper_bound": 1.0,
                        },
                        "critic_time_sampling": {
                            "exact_t1_probability": (
                                args.critic_time_exact_t1_probability
                            ),
                            "high_noise_probability": (
                                args.critic_time_high_noise_probability
                            ),
                            "upper_bound": args.critic_time_upper_bound,
                            "paired_noise": True,
                        },
                        "generator_adversarial_time_sampler": (
                            args.adversarial_time_sampler
                        ),
                        "optimizer_update_ratio": {
                            "generator": 1,
                            "fake_score": args.score_updates_per_generator,
                            "critic": args.critic_updates_per_generator,
                            "ratio_origin": "loaded_counters",
                        },
                        "released_ardy_role": "offline_10step_cfg_endpoint_corpus_only",
                        "deployment_nfe": 1,
                        "cfg_at_deployment": False,
                        "lr_schedule": "constant",
                        "resume_learning_rate_policy": (
                            "override_from_command_line"
                            if args.override_learning_rates_on_resume
                            else "restore_checkpoint"
                        ),
                        "initialization": {
                            "generator": "warm_start_from_previous_stage_final_ema",
                            "fake_score": "exact_generator_clone",
                            "critic": "new_only_when_no_compatible_stage_state_exists",
                            "same_stage_continuation": "full_accelerate_resume_state",
                        },
                    },
                },
                default=str,
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

    fields = (
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
        "encoder_body",
    )
    primary_dataset = TeacherShardDataset(
        args.data,
        cache_shards=args.cache_shards,
        fields=fields,
    )
    if args.replay_data is None:
        dataset = primary_dataset
        sampler = ShardShuffleSampler(dataset, seed=args.seed)
    else:
        replay_dataset = TeacherShardDataset(
            args.replay_data,
            cache_shards=args.cache_shards,
            fields=fields,
        )
        dataset = IndexedTeacherMixtureDataset(primary_dataset, replay_dataset)
        sampler = ShardMixtureSampler(
            dataset,
            primary_probability=args.primary_data_prob,
            chunk_size=args.batch_size,
            seed=args.seed,
        )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        drop_last=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    generator_optimizer = AdamW(
        generator.parameters(), lr=args.generator_learning_rate, weight_decay=0
    )
    score_optimizer = AdamW(
        fake_score.parameters(), lr=args.score_learning_rate, weight_decay=0
    )
    critic_optimizer = AdamW(
        critic.parameters(), lr=args.critic_learning_rate, weight_decay=0
    )
    generator_scheduler = LambdaLR(generator_optimizer, lambda _step: 1.0)
    score_scheduler = LambdaLR(score_optimizer, lambda _step: 1.0)
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
    counters = TrainingCounters()
    accelerator.register_for_checkpointing(ema)
    accelerator.register_for_checkpointing(counters)
    accelerator.register_for_checkpointing(sampler)
    quantizer = FSQRequantizer(args.checkpoint_dir / "stats/post_quantization").to(
        accelerator.device
    )
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
    if args.resume is not None:
        accelerator.load_state(str(args.resume))
        if args.override_learning_rates_on_resume:
            override_constant_learning_rate(
                generator_optimizer,
                generator_scheduler,
                args.generator_learning_rate,
            )
            override_constant_learning_rate(
                score_optimizer,
                score_scheduler,
                args.score_learning_rate,
            )
            override_constant_learning_rate(
                critic_optimizer,
                critic_scheduler,
                args.critic_learning_rate,
            )
        if args.reset_update_ratio_on_resume:
            counters.reset_update_ratio_origin()

    start_time = time.perf_counter()
    stop = False
    latest_guidance: dict[str, torch.Tensor] = {}
    while counters.generator_updates < args.steps and not stop:
        for batch in loader:
            counters.iterations += 1
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
            with torch.no_grad(), torch.autocast(
                device_type=accelerator.device.type, enabled=False
            ):
                encoded_history = encoder(
                    batch["encoder_body"].to(dtype=frozen_codec_dtype)
                ).float()
            encoded_history = quantizer(encoded_history, ste=False)
            history_valid = batch["has_history"].unsqueeze(-1)
            history_hybrid = torch.cat(
                [
                    batch["history_hybrid"][..., :20] * history_valid,
                    encoded_history * history_valid,
                ],
                dim=-1,
            )

            (
                run_score_update,
                run_critic_update,
                in_guidance_warmup,
                score_target,
                critic_target,
            ) = guidance_update_plan(
                counters,
                warmup_updates=args.warmup_guidance_updates,
                score_updates_per_generator=args.score_updates_per_generator,
                critic_updates_per_generator=args.critic_updates_per_generator,
            )

            if run_score_update or run_critic_update:
                with torch.no_grad(), accelerator.autocast():
                    generated_detached = nfe1_endpoint(
                        generator,
                        batch["initial_noise"],
                        history_hybrid,
                        batch["path_condition"],
                        batch["first_heading"],
                        batch["has_history"],
                        text_feature,
                        heading_condition,
                        root_smoothing_passes=args.flow_root_smoothing_passes,
                    ).detach()

            if run_score_update:
                score_time = sample_flow_time(
                    generated_detached.shape[0],
                    accelerator.device,
                    dtype=generated_detached.dtype,
                    exact_t1_probability=args.time_exact_t1_probability,
                    high_noise_probability=args.time_high_noise_probability,
                )
                score_noise = torch.randn_like(generated_detached)
                score_input, score_target_velocity = make_flow_pair(
                    generated_detached, score_noise, score_time
                )
                score_optimizer.zero_grad(set_to_none=True)
                with accelerator.autocast():
                    score_prediction = fake_score(
                        score_input,
                        history_hybrid,
                        batch["path_condition"],
                        batch["first_heading"],
                        batch["has_history"],
                        score_time,
                        text_feature,
                        heading_condition,
                    )
                    score_losses = fake_score_loss(
                        score_prediction, score_target_velocity
                    )
                accelerator.backward(score_losses["fake_score_total"])
                score_grad_norm = accelerator.clip_grad_norm_(
                    fake_score.parameters(), args.score_grad_clip
                )
                score_optimizer.step()
                score_scheduler.step()
                counters.score_updates += 1
                latest_guidance.update(
                    {
                        **score_losses,
                        "score_grad_norm": (
                            score_grad_norm
                            if isinstance(score_grad_norm, torch.Tensor)
                            else score_losses["fake_score_total"].new_tensor(
                                float(score_grad_norm)
                            )
                        ),
                        "score_time_mean": score_time.mean(),
                    }
                )

            if run_critic_update:
                critic_time = sample_flow_time(
                    generated_detached.shape[0],
                    accelerator.device,
                    dtype=generated_detached.dtype,
                    exact_t1_probability=args.critic_time_exact_t1_probability,
                    high_noise_probability=args.critic_time_high_noise_probability,
                )
                critic_time = critic_time * args.critic_time_upper_bound
                critic_noise = torch.randn_like(generated_detached)
                with torch.no_grad(), accelerator.autocast():
                    _, real_feature_values = flow_teacher_features(
                        teacher_flow,
                        batch["clean_generation"],
                        critic_noise,
                        critic_time,
                        history_hybrid,
                        batch["path_condition"],
                        batch["first_heading"],
                        batch["has_history"],
                        text_feature,
                        heading_condition,
                        feature_tap=critic_feature_request(
                            args.critic_feature_taps
                        ),
                    )
                    _, fake_feature_values = flow_teacher_features(
                        teacher_flow,
                        generated_detached,
                        critic_noise,
                        critic_time,
                        history_hybrid,
                        batch["path_condition"],
                        batch["first_heading"],
                        batch["has_history"],
                        text_feature,
                        heading_condition,
                        feature_tap=critic_feature_request(
                            args.critic_feature_taps
                        ),
                    )
                    real_features = critic_feature_map(
                        real_feature_values, args.critic_feature_taps
                    )
                    fake_features = critic_feature_map(
                        fake_feature_values, args.critic_feature_taps
                    )
                    critic_feature_gap_by_tap = {
                        tap: (
                            real_features[tap].float()
                            - fake_features[tap].float()
                        )
                        .abs()
                        .mean(dim=(1, 2))
                        for tap in args.critic_feature_taps
                    }
                    critic_feature_gap_per_sample = torch.stack(
                        list(critic_feature_gap_by_tap.values())
                    ).mean(dim=0)
                    critic_exact_t1_mask = critic_time.reshape(-1) == 1
                    critic_exact_t1_count = critic_exact_t1_mask.sum()
                    critic_exact_t1_feature_gap = (
                        critic_feature_gap_per_sample
                        * critic_exact_t1_mask.to(critic_feature_gap_per_sample.dtype)
                    ).sum() / critic_exact_t1_count.clamp_min(1)
                critic_optimizer.zero_grad(set_to_none=True)
                with accelerator.autocast():
                    real_logits = critic_logits(critic, real_features)
                    fake_logits = critic_logits(critic, fake_features)
                    critic_losses = aggregate_discriminator_losses(
                        real_logits,
                        fake_logits,
                        args.critic_tap_aggregation,
                    )
                accelerator.backward(critic_losses["critic_total"])
                critic_grad_norm = accelerator.clip_grad_norm_(
                    critic.parameters(), args.critic_grad_clip
                )
                critic_optimizer.step()
                critic_scheduler.step()
                counters.critic_updates += 1
                latest_guidance.update(
                    {
                        **critic_losses,
                        "critic_grad_norm": (
                            critic_grad_norm
                            if isinstance(critic_grad_norm, torch.Tensor)
                            else critic_losses["critic_total"].new_tensor(
                                float(critic_grad_norm)
                            )
                        ),
                        "critic_time_mean": critic_time.mean(),
                        "critic_time_min": critic_time.min(),
                        "critic_time_max": critic_time.max(),
                        "critic_time_exact_t1_fraction": (
                            critic_exact_t1_mask.float().mean()
                        ),
                        "critic_feature_gap_l1": (
                            critic_feature_gap_per_sample.mean()
                        ),
                        "critic_exact_t1_feature_gap_l1": (
                            critic_exact_t1_feature_gap
                        ),
                        **{
                            f"critic_tap_{tap}_feature_gap_l1": gap.mean()
                            for tap, gap in critic_feature_gap_by_tap.items()
                        },
                        **{
                            f"critic_tap_{tap}_exact_t1_feature_gap_l1": (
                                gap
                                * critic_exact_t1_mask.to(gap.dtype)
                            ).sum()
                            / critic_exact_t1_count.clamp_min(1)
                            for tap, gap in critic_feature_gap_by_tap.items()
                        },
                    }
                )

            if run_score_update or run_critic_update:
                counters.guidance_updates += 1

            if in_guidance_warmup:
                warmup_complete = (
                    counters.score_updates >= args.warmup_guidance_updates
                    and counters.critic_updates >= args.warmup_guidance_updates
                )
                if warmup_complete and not counters.has_update_ratio_origin:
                    counters.reset_update_ratio_origin()
                if counters.guidance_updates % args.log_every == 0:
                    numeric = tensor_metrics_to_float(latest_guidance, accelerator)
                    logger.log(
                        counters.generator_updates,
                        numeric,
                        event="guidance_warmup",
                        guidance_updates=counters.guidance_updates,
                        score_updates=counters.score_updates,
                        critic_updates=counters.critic_updates,
                    )
                    if accelerator.is_main_process:
                        print(
                            json.dumps(
                                {
                                    "generator_step": counters.generator_updates,
                                    "guidance_updates": counters.guidance_updates,
                                    "score_updates": counters.score_updates,
                                    "critic_updates": counters.critic_updates,
                                    **numeric,
                                }
                            ),
                            flush=True,
                        )
                continue
            if (
                counters.score_updates < score_target
                or counters.critic_updates < critic_target
            ):
                continue

            # Direct NFE=1 generator update with flow-native DMD and GAN terms.
            generator_optimizer.zero_grad(set_to_none=True)
            critic.requires_grad_(False)
            adversarial_tap_gradient_metrics: dict[str, torch.Tensor] = {}
            component_gradient_metrics: dict[str, torch.Tensor] = {}
            with accelerator.autocast():
                generated = nfe1_endpoint(
                    generator,
                    batch["initial_noise"],
                    history_hybrid,
                    batch["path_condition"],
                    batch["first_heading"],
                    batch["has_history"],
                    text_feature,
                    heading_condition,
                    root_smoothing_passes=args.flow_root_smoothing_passes,
                )

                dmd_time = sample_flow_time(
                    generated.shape[0],
                    accelerator.device,
                    dtype=generated.dtype,
                    exact_t1_probability=args.time_exact_t1_probability,
                    high_noise_probability=args.time_high_noise_probability,
                )
                dmd_noise = torch.randn_like(generated)
                dmd_input, _ = make_flow_pair(generated.detach(), dmd_noise, dmd_time)
                with torch.no_grad():
                    teacher_velocity = teacher_flow(
                        dmd_input,
                        history_hybrid,
                        batch["path_condition"],
                        batch["first_heading"],
                        batch["has_history"],
                        dmd_time,
                        text_feature,
                        heading_condition,
                    )
                    fake_velocity = fake_score(
                        dmd_input,
                        history_hybrid,
                        batch["path_condition"],
                        batch["first_heading"],
                        batch["has_history"],
                        dmd_time,
                        text_feature,
                        heading_condition,
                    )
                    teacher_x0 = flow_velocity_to_x0(
                        dmd_input, teacher_velocity, dmd_time
                    )
                    fake_x0 = flow_velocity_to_x0(dmd_input, fake_velocity, dmd_time)
                dmd_losses = distribution_matching_loss(
                    generated,
                    teacher_x0,
                    fake_x0,
                    dmd_input,
                    grad_clip=args.dmd_grad_clip,
                )

                if args.adversarial_time_sampler == "critic":
                    adversarial_exact_t1_probability = (
                        args.critic_time_exact_t1_probability
                    )
                    adversarial_high_noise_probability = (
                        args.critic_time_high_noise_probability
                    )
                    adversarial_time_upper_bound = args.critic_time_upper_bound
                else:
                    adversarial_exact_t1_probability = args.time_exact_t1_probability
                    adversarial_high_noise_probability = (
                        args.time_high_noise_probability
                    )
                    adversarial_time_upper_bound = 1.0
                adversarial_time = sample_flow_time(
                    generated.shape[0],
                    accelerator.device,
                    dtype=generated.dtype,
                    exact_t1_probability=adversarial_exact_t1_probability,
                    high_noise_probability=adversarial_high_noise_probability,
                )
                adversarial_time = adversarial_time * adversarial_time_upper_bound
                adversarial_noise = torch.randn_like(generated)
                _, adversarial_feature_values = flow_teacher_features(
                    teacher_flow,
                    generated,
                    adversarial_noise,
                    adversarial_time,
                    history_hybrid,
                    batch["path_condition"],
                    batch["first_heading"],
                    batch["has_history"],
                    text_feature,
                    heading_condition,
                    feature_tap=critic_feature_request(
                        args.critic_feature_taps
                    ),
                )
                adversarial_features = critic_feature_map(
                    adversarial_feature_values, args.critic_feature_taps
                )
                adversarial_logits = critic_logits(
                    critic, adversarial_features
                )
                adversarial_loss, adversarial_loss_by_tap = (
                    aggregate_generator_adversarial_loss(
                        adversarial_logits,
                        args.critic_tap_aggregation,
                    )
                )
                next_generator_step = counters.generator_updates + 1
                if (
                    args.log_critic_tap_gradient_rms
                    and len(args.critic_feature_taps) > 1
                    and (
                        next_generator_step % args.log_every == 0
                        or next_generator_step == 1
                    )
                ):
                    for tap, tap_loss in adversarial_loss_by_tap.items():
                        tap_gradient = torch.autograd.grad(
                            tap_loss,
                            generated,
                            retain_graph=True,
                            create_graph=False,
                        )[0]
                        adversarial_tap_gradient_metrics[
                            f"generator_adversarial_{tap}_endpoint_grad_rms"
                        ] = tap_gradient.float().square().mean().sqrt().detach()

                paired_losses = endpoint_losses(generated, batch["clean_generation"])
                fsq_losses = fsq_endpoint_losses(
                    generated[..., 20:],
                    batch["clean_generation"][..., 20:],
                    quantizer,
                )
                fsq_diagnostics = fsq_endpoint_diagnostics(
                    generated[..., 20:],
                    batch["clean_generation"][..., 20:],
                    quantizer,
                )
                path_losses = path_constraint_losses(
                    generated, batch["path_condition"], batch["has_history"]
                )
                temporal_losses = root_temporal_losses(
                    generated, batch["clean_generation"], motion_rep
                )
                predicted_root, predicted_local_root = deployment_decoder_roots(
                    generated,
                    history_hybrid,
                    batch["has_history"],
                    motion_rep,
                )
                generated_latent = quantizer(generated[..., 20:], ste=True)
                latent_sequence = torch.cat(
                    [history_hybrid[..., 20:], generated_latent], dim=1
                )
                with torch.autocast(
                    device_type=accelerator.device.type, enabled=False
                ):
                    decoded = decoder(
                        latent_sequence.to(dtype=frozen_codec_dtype),
                        predicted_local_root.to(dtype=frozen_codec_dtype),
                        batch["decoder_token_valid"].to(dtype=frozen_codec_dtype),
                    ).float()
                decoder_losses = decoder_feature_losses(
                    decoded, batch["target_body"], batch["decoder_token_valid"]
                )
                quality_losses = motion_quality_losses(
                    decoded,
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
                    + 0.05 * quality_losses["foot_slide"]
                )
                seam = seam_losses(
                    generated,
                    decoded,
                    history_hybrid,
                    batch["decoder_global_root"],
                    batch["target_body"],
                    batch["has_history"],
                )
                physical_seam = physical_seam_losses(
                    decoded,
                    batch["target_body"],
                    predicted_root,
                    batch["decoder_global_root"],
                    batch["has_history"],
                    motion_rep,
                )
                total = (
                    args.dmd_weight * dmd_losses["dmd_total"]
                    + args.adversarial_weight * adversarial_loss
                    + args.paired_weight * paired_losses["endpoint_total"]
                    + args.fsq_weight * fsq_losses["fsq_quantized_l1"]
                    + args.path_weight * path_losses["path_constraint_mse"]
                    + args.decoder_weight * decoder_losses["decoder_total"]
                    + args.root_temporal_weight
                    * temporal_losses["root_temporal_total"]
                    + args.quality_weight * quality_total
                    + args.seam_weight * seam["seam_total"]
                    + args.physical_seam_weight
                    * physical_seam["physical_seam_total"]
                )

                if (
                    args.generator_component_gradient_every > 0
                    and next_generator_step
                    % args.generator_component_gradient_every
                    == 0
                ):
                    component_gradient_metrics = (
                        endpoint_component_gradient_diagnostics(
                            generated,
                            total,
                            {
                                "dmd": (dmd_losses["dmd_total"], args.dmd_weight),
                                "adversarial": (
                                    adversarial_loss,
                                    args.adversarial_weight,
                                ),
                                "paired_fsq": (
                                    args.paired_weight
                                    * paired_losses["endpoint_total"]
                                    + args.fsq_weight
                                    * fsq_losses["fsq_quantized_l1"],
                                    1.0,
                                ),
                                "control_physics": (
                                    args.path_weight
                                    * path_losses["path_constraint_mse"]
                                    + args.decoder_weight
                                    * decoder_losses["decoder_total"]
                                    + args.root_temporal_weight
                                    * temporal_losses["root_temporal_total"]
                                    + args.quality_weight * quality_total
                                    + args.seam_weight * seam["seam_total"]
                                    + args.physical_seam_weight
                                    * physical_seam["physical_seam_total"],
                                    1.0,
                                ),
                            },
                        )
                    )

            accelerator.backward(total)
            generator_grad_norm = accelerator.clip_grad_norm_(
                generator.parameters(), args.generator_grad_clip
            )
            generator_optimizer.step()
            generator_scheduler.step()
            critic.requires_grad_(True)
            counters.generator_updates += 1
            unwrapped_generator = accelerator.unwrap_model(generator)
            ema.update(unwrapped_generator)

            generator_metrics = {
                **dmd_losses,
                **paired_losses,
                **fsq_losses,
                **fsq_diagnostics,
                **path_losses,
                **temporal_losses,
                **decoder_losses,
                **seam,
                **physical_seam,
                "quality_total": quality_total,
                "generator_adversarial": adversarial_loss,
                **{
                    f"generator_adversarial_{tap}": value
                    for tap, value in adversarial_loss_by_tap.items()
                },
                **adversarial_tap_gradient_metrics,
                **component_gradient_metrics,
                "loss_total": total,
                "generator_grad_norm": (
                    generator_grad_norm
                    if isinstance(generator_grad_norm, torch.Tensor)
                    else total.new_tensor(float(generator_grad_norm))
                ),
                "dmd_time_mean": dmd_time.mean(),
                "adversarial_time_mean": adversarial_time.mean(),
                "adversarial_time_min": adversarial_time.min(),
                "adversarial_time_max": adversarial_time.max(),
                "adversarial_time_exact_t1_fraction": (
                    adversarial_time.reshape(-1) == 1
                ).float().mean(),
                "generator_learning_rate": total.new_tensor(
                    generator_scheduler.get_last_lr()[0]
                ),
                "score_learning_rate": total.new_tensor(
                    score_scheduler.get_last_lr()[0]
                ),
                "critic_learning_rate": total.new_tensor(
                    critic_scheduler.get_last_lr()[0]
                ),
            }
            if (
                counters.generator_updates % args.log_every == 0
                or counters.generator_updates == 1
            ):
                numeric = tensor_metrics_to_float(
                    {**latest_guidance, **generator_metrics}, accelerator
                )
                logger.log(
                    counters.generator_updates,
                    numeric,
                    guidance_updates=counters.guidance_updates,
                    score_updates=counters.score_updates,
                    critic_updates=counters.critic_updates,
                    samples_seen=(
                        counters.iterations
                        * args.batch_size
                        * accelerator.num_processes
                    ),
                )
                if accelerator.is_main_process:
                    print(
                        json.dumps(
                            {
                                "generator_step": counters.generator_updates,
                                "guidance_updates": counters.guidance_updates,
                                "score_updates": counters.score_updates,
                                "critic_updates": counters.critic_updates,
                                **numeric,
                            }
                        ),
                        flush=True,
                    )

            if counters.generator_updates % args.state_every == 0:
                state_dir = (
                    args.output
                    / "state"
                    / f"step-{counters.generator_updates:07d}"
                )
                accelerator.save_state(str(state_dir))
            if (
                counters.generator_updates % args.save_every == 0
                and accelerator.is_main_process
            ):
                save_weights(
                    unwrapped_generator,
                    accelerator.unwrap_model(fake_score),
                    accelerator.unwrap_model(critic),
                    ema,
                    args.output,
                    counters.generator_updates,
                )

            if time.perf_counter() - start_time >= args.max_runtime_s:
                stop = True
            if counters.generator_updates >= args.steps or stop:
                break

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        save_weights(
            accelerator.unwrap_model(generator),
            accelerator.unwrap_model(fake_score),
            accelerator.unwrap_model(critic),
            ema,
            args.output,
            counters.generator_updates,
        )
        logger.log(
            counters.generator_updates,
            {},
            event="training_complete",
            guidance_updates=counters.guidance_updates,
            score_updates=counters.score_updates,
            critic_updates=counters.critic_updates,
            stopped_by_runtime=stop,
        )
    logger.close()
    accelerator.end_training()


if __name__ == "__main__":
    main()
