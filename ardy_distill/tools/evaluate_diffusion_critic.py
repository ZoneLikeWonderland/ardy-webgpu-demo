#!/usr/bin/env python
"""Measure a trained motion critic separately at every diffusion timestep."""

from __future__ import annotations

import argparse
import json
from contextlib import nullcontext
from dataclasses import fields
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ardy_distill.data import TeacherShardDataset
from ardy_distill.dmd2 import diffusion_motion_critic_features, q_sample
from ardy_distill.losses import FSQRequantizer
from ardy_distill.models import (
    FlowStudentConfig,
    HistoryEncoderStudent,
    MotionDecoderStudent,
    OneStepFlowStudent,
    ScoreBackboneCriticHead,
    TemporalCriticConfig,
    TemporalMotionCritic,
)
from ardy_distill.runtime import load_motion_rep, load_safetensor_weights
from ardy_distill.train_dmd2 import (
    decode_endpoint,
    generate_endpoint,
    score_backbone_critic_logits,
)


TRAINING_FIELDS = (
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-config", type=Path, required=True)
    parser.add_argument("--critic", type=Path, required=True)
    parser.add_argument(
        "--fake-score",
        type=Path,
        help="Required for a score_backbone critic run.",
    )
    parser.add_argument("--generator", type=Path, required=True)
    parser.add_argument("--decoder", type=Path, required=True)
    parser.add_argument("--encoder", type=Path, required=True)
    parser.add_argument("--data", type=Path, nargs="+", required=True)
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path("ardy/checkpoints/ARDY-Core-RP-20FPS-Horizon40"),
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--batches-per-data", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=20260714)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def dataclass_kwargs(cls, values: dict) -> dict:
    names = {field.name for field in fields(cls)}
    return {name: value for name, value in values.items() if name in names}


def main() -> None:
    args = parse_args()
    if args.batch_size < 1 or args.batches_per_data < 1:
        raise ValueError("batch size and batches per data source must be positive")

    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    run_config = json.loads(args.run_config.read_text(encoding="utf-8"))
    flow_config = FlowStudentConfig(
        **dataclass_kwargs(FlowStudentConfig, run_config["generator_config"])
    )
    critic_kind = run_config.get("critic_kind", "physical_features")
    shared_score_critic = critic_kind == "score_backbone"
    generator = OneStepFlowStudent(flow_config).to(device).eval()
    fake_score = None
    if shared_score_critic:
        if args.fake_score is None:
            raise ValueError("--fake-score is required for score_backbone critic")
        fake_score = OneStepFlowStudent(flow_config).to(device).eval()
        critic = ScoreBackboneCriticHead(
            width=flow_config.width,
            blocks=int(run_config["critic_config"]["blocks"]),
        ).to(device).eval()
        num_diffusion_steps = 10
    else:
        critic_config = TemporalCriticConfig(
            **dataclass_kwargs(
                TemporalCriticConfig, run_config["critic_config"]
            )
        )
        critic = TemporalMotionCritic(critic_config).to(device).eval()
        num_diffusion_steps = critic_config.num_diffusion_steps
    decoder = MotionDecoderStudent().to(device).eval()
    encoder = HistoryEncoderStudent().to(device).eval()
    load_safetensor_weights(generator, args.generator)
    load_safetensor_weights(critic, args.critic)
    if fake_score is not None and args.fake_score is not None:
        load_safetensor_weights(fake_score, args.fake_score)
    load_safetensor_weights(decoder, args.decoder)
    load_safetensor_weights(encoder, args.encoder)
    generator.requires_grad_(False)
    if fake_score is not None:
        fake_score.requires_grad_(False)
    critic.requires_grad_(False)
    decoder.requires_grad_(False)
    encoder.requires_grad_(False)

    quantizer = FSQRequantizer(
        args.checkpoint_dir / "stats/post_quantization"
    ).to(device)
    motion_rep = load_motion_rep(args.checkpoint_dir)
    from ardy.model.diffusion import Diffusion

    alphas_cumprod = Diffusion(num_diffusion_steps).alphas_cumprod_base.to(
        device
    )
    accumulators = {
        timestep: {
            "count": 0,
            "real_logit_sum": 0.0,
            "fake_logit_sum": 0.0,
            "real_loss_sum": 0.0,
            "fake_loss_sum": 0.0,
            "real_correct": 0,
            "fake_correct": 0,
        }
        for timestep in range(num_diffusion_steps)
    }

    autocast = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if device.type == "cuda"
        else nullcontext()
    )
    evaluated_sources: list[dict[str, int | str]] = []
    with torch.inference_mode():
        for data_path in args.data:
            dataset = TeacherShardDataset(
                data_path, cache_shards=2, fields=TRAINING_FIELDS
            )
            loader = DataLoader(
                dataset,
                batch_size=args.batch_size,
                shuffle=False,
                drop_last=True,
                num_workers=0,
                pin_memory=device.type == "cuda",
            )
            source_batches = 0
            source_samples = 0
            for batch in loader:
                batch = {
                    name: value.to(device, non_blocking=True).float()
                    for name, value in batch.items()
                }
                with autocast:
                    encoded = encoder(batch["encoder_body"]).float()
                encoded = quantizer(encoded, ste=False)
                valid = batch["has_history"].unsqueeze(-1)
                history_hybrid = torch.cat(
                    [
                        batch["history_hybrid"][..., :20] * valid,
                        encoded * valid,
                    ],
                    dim=-1,
                )
                with autocast:
                    endpoint = generate_endpoint(generator, batch, history_hybrid)
                    if shared_score_critic:
                        decoded_fake = None
                        fake_root = None
                    else:
                        decoded_fake, fake_root = decode_endpoint(
                            endpoint,
                            history_hybrid,
                            batch,
                            decoder,
                            quantizer,
                            motion_rep,
                            ste=False,
                        )

                local_batch = endpoint.shape[0]
                for timestep, accumulator in accumulators.items():
                    timesteps = torch.full(
                        (local_batch,), timestep, device=device, dtype=torch.long
                    )
                    if shared_score_critic:
                        if fake_score is None:
                            raise RuntimeError("fake-score model is unavailable")
                        real_xt = q_sample(
                            batch["clean_generation"],
                            timesteps,
                            torch.randn_like(batch["clean_generation"]),
                            alphas_cumprod,
                        )
                        fake_xt = q_sample(
                            endpoint,
                            timesteps,
                            torch.randn_like(endpoint),
                            alphas_cumprod,
                        )
                        with autocast:
                            real_logits = score_backbone_critic_logits(
                                fake_score,
                                critic,
                                real_xt,
                                timesteps,
                                num_diffusion_steps,
                                batch,
                                history_hybrid,
                            ).float()
                            fake_logits = score_backbone_critic_logits(
                                fake_score,
                                critic,
                                fake_xt,
                                timesteps,
                                num_diffusion_steps,
                                batch,
                                history_hybrid,
                            ).float()
                    else:
                        if decoded_fake is None or fake_root is None:
                            raise RuntimeError("decoded physical critic input is unavailable")
                        real_features = diffusion_motion_critic_features(
                            batch["target_body"][:, -40:],
                            batch["decoder_global_root"][:, -40:],
                            timesteps,
                            alphas_cumprod,
                            motion_rep,
                        )
                        fake_features = diffusion_motion_critic_features(
                            decoded_fake[:, -40:],
                            fake_root[:, -40:],
                            timesteps,
                            alphas_cumprod,
                            motion_rep,
                        )
                        with autocast:
                            real_logits = critic(
                                real_features,
                                history_hybrid,
                                batch["path_condition"],
                                batch["first_heading"],
                                batch["has_history"],
                                timesteps,
                            ).float()
                            fake_logits = critic(
                                fake_features,
                                history_hybrid,
                                batch["path_condition"],
                                batch["first_heading"],
                                batch["has_history"],
                                timesteps,
                            ).float()
                    accumulator["count"] += local_batch
                    accumulator["real_logit_sum"] += float(real_logits.sum())
                    accumulator["fake_logit_sum"] += float(fake_logits.sum())
                    accumulator["real_loss_sum"] += float(
                        F.softplus(-real_logits).sum()
                    )
                    accumulator["fake_loss_sum"] += float(
                        F.softplus(fake_logits).sum()
                    )
                    accumulator["real_correct"] += int((real_logits > 0).sum())
                    accumulator["fake_correct"] += int((fake_logits < 0).sum())

                source_batches += 1
                source_samples += local_batch
                if source_batches >= args.batches_per_data:
                    break
            evaluated_sources.append(
                {
                    "path": str(data_path),
                    "batches": source_batches,
                    "samples": source_samples,
                }
            )

    per_timestep = {}
    for timestep, accumulator in accumulators.items():
        count = accumulator["count"]
        real_loss = accumulator["real_loss_sum"] / count
        fake_loss = accumulator["fake_loss_sum"] / count
        real_logit = accumulator["real_logit_sum"] / count
        fake_logit = accumulator["fake_logit_sum"] / count
        per_timestep[str(timestep)] = {
            "alpha_cumprod": float(alphas_cumprod[timestep]),
            "snr": float(
                alphas_cumprod[timestep]
                / (1.0 - alphas_cumprod[timestep]).clamp_min(1.0e-12)
            ),
            "samples": count,
            "critic_total": real_loss + fake_loss,
            "real_loss": real_loss,
            "fake_loss": fake_loss,
            "real_logit": real_logit,
            "fake_logit": fake_logit,
            "logit_gap": real_logit - fake_logit,
            "real_accuracy": accumulator["real_correct"] / count,
            "fake_accuracy": accumulator["fake_correct"] / count,
            "balanced_accuracy": (
                accumulator["real_correct"] + accumulator["fake_correct"]
            )
            / (2 * count),
        }

    result = {
        "schema": "ardy_diffusion_critic_timestep_probe_v1",
        "inputs": {
            "run_config": str(args.run_config),
            "critic": str(args.critic),
            "fake_score": str(args.fake_score) if args.fake_score else None,
            "generator": str(args.generator),
            "decoder": str(args.decoder),
            "encoder": str(args.encoder),
            "seed": args.seed,
        },
        "sources": evaluated_sources,
        "per_timestep": per_timestep,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    for timestep, metrics in per_timestep.items():
        print(
            f"t={timestep} snr={metrics['snr']:.4f} "
            f"loss={metrics['critic_total']:.4f} "
            f"logits={metrics['real_logit']:+.3f}/{metrics['fake_logit']:+.3f} "
            f"balanced_acc={metrics['balanced_accuracy']:.3f}"
        )
    print(json.dumps({"event": "complete", "output": str(args.output)}))


if __name__ == "__main__":
    main()
