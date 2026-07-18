#!/usr/bin/env python
"""Generate the broad text × UI-control teacher corpus on four independent ranks."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from ardy.constraints import Root2DConstraintSet
from ardy.model import load_model
from ardy_distill.control_distribution import (
    CONTROL_MODE_TO_ID,
    PromptControlSampler,
    load_prompt_bank,
    pack_heading_condition,
)
from ardy_distill.data import TeacherShardWriter
from ardy_distill.teacher import trace_autoregressive_step
from ardy_distill.text_features import PromptFeatureTable


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ROLLOUT_LENGTHS = (1, 2, 4, 8, 16)
ROLLOUT_LENGTH_WEIGHTS = (0.25, 0.20, 0.25, 0.20, 0.10)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--prompt-bank",
        type=Path,
        default=PROJECT_ROOT / "distill_data" / "text_control_v1" / "prompt_bank.jsonl",
    )
    parser.add_argument(
        "--llama-features",
        type=Path,
        default=PROJECT_ROOT / "distill_data" / "text_control_v1" / "features" / "llama",
    )
    parser.add_argument(
        "--checkpoints-dir",
        type=Path,
        default=PROJECT_ROOT / "ardy" / "checkpoints",
    )
    parser.add_argument("--model", default="ARDY-Core-RP-20FPS-Horizon40")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--windows", type=int, default=4096, help="Windows per rank")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--shard-size", type=int, default=256)
    parser.add_argument("--storage-dtype", choices=["float16", "float32"], default="float16")
    parser.add_argument("--seed", type=int, default=20260715)
    parser.add_argument("--cfg-text", type=float, default=2.0)
    parser.add_argument("--cfg-constraint", type=float, default=2.0)
    parser.add_argument("--prompt-switch-prob", type=float, default=0.15)
    parser.add_argument("--unconditional-prob", type=float, default=0.10)
    parser.add_argument("--official-prompt-prob", type=float, default=0.10)
    parser.add_argument("--log-every", type=int, default=256)
    return parser.parse_args()


def distributed_context() -> tuple[int, int, int]:
    return (
        int(os.environ.get("RANK", "0")),
        int(os.environ.get("LOCAL_RANK", "0")),
        int(os.environ.get("WORLD_SIZE", "1")),
    )


def sample_rollout_length(generator: torch.Generator) -> int:
    weights = torch.tensor(ROLLOUT_LENGTH_WEIGHTS, dtype=torch.float64)
    index = int(torch.multinomial(weights, 1, generator=generator))
    return ROLLOUT_LENGTHS[index]


def root_state(
    teacher,
    history: torch.Tensor | None,
    initial_translation: torch.Tensor,
    initial_heading: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch = initial_translation.shape[0]
    if history is None:
        return (
            initial_translation[:, [0, 2]].detach().cpu(),
            initial_heading.detach().cpu(),
            torch.zeros(batch, 2),
        )
    unnormalized = teacher.motion_rep.unnormalize(history.float())
    positions = teacher.motion_rep.get_root_pos(unnormalized)
    headings = teacher.motion_rep.get_root_heading_angle(unnormalized)
    velocity = (positions[:, -1, [0, 2]] - positions[:, -2, [0, 2]]) * float(
        teacher.motion_rep.fps
    )
    return (
        positions[:, -1, [0, 2]].detach().cpu(),
        headings[:, -1].detach().cpu(),
        velocity.detach().cpu(),
    )


def make_conditions(
    teacher,
    sampler: PromptControlSampler,
    prompt_ids: torch.Tensor,
    start_xz: torch.Tensor,
    start_heading: torch.Tensor,
    current_velocity: torch.Tensor,
    *,
    has_history: bool,
    frames: int = 64,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    constraints: list[list[Root2DConstraintSet]] = []
    mode_ids = []
    for index, prompt_id_tensor in enumerate(prompt_ids):
        prompt_id = int(prompt_id_tensor)
        mode = sampler.sample_control_mode(prompt_id)
        trajectory = sampler.sample_trajectory(
            prompt_id,
            mode,
            start_xz=start_xz[index],
            start_heading=float(start_heading[index]),
            current_velocity_xz=current_velocity[index],
            history_frames=4 if has_history else 0,
            frames=frames,
            fps=int(teacher.motion_rep.fps),
        )
        mode_ids.append(CONTROL_MODE_TO_ID[mode])
        if not len(trajectory.frame_indices):
            constraints.append([])
            continue
        constraints.append(
            [
                Root2DConstraintSet(
                    teacher.skeleton,
                    frame_indices=trajectory.frame_indices,
                    root_2d=trajectory.root_xz,
                    global_root_heading=trajectory.heading_angles,
                )
            ]
        )
    lengths = torch.full(
        (len(prompt_ids),),
        frames,
        device=teacher.device,
        dtype=torch.long,
    )
    observed_motion, motion_mask = teacher.motion_rep.create_conditions_from_constraints_batched(
        constraints,
        lengths,
        to_normalize=True,
        device=teacher.device,
    )
    return (
        motion_mask,
        observed_motion,
        torch.tensor(mode_ids, dtype=torch.int64),
    )


def main() -> None:
    args = parse_args()
    rank, local_rank, world_size = distributed_context()
    if args.windows < args.batch_size or args.windows % args.batch_size:
        raise ValueError("--windows must be a positive multiple of --batch-size")
    if not 0 <= args.prompt_switch_prob <= 1:
        raise ValueError("--prompt-switch-prob must be in [0, 1]")
    if args.log_every < 1:
        raise ValueError("--log-every must be positive")

    device = torch.device(f"cuda:{local_rank}" if args.device == "auto" else args.device)
    rank_seed = args.seed + rank * 1_000_003
    random.seed(rank_seed)
    np.random.seed(rank_seed)
    torch.manual_seed(rank_seed)
    torch.cuda.manual_seed_all(rank_seed)
    generator = torch.Generator(device="cpu").manual_seed(rank_seed + 17)

    rank_output = args.output_dir / f"rank-{rank:02d}"
    rank_output.mkdir(parents=True, exist_ok=True)
    if (rank_output / "manifest.json").exists():
        raise FileExistsError(f"refusing to overwrite existing shards: {rank_output}")

    prompt_records = load_prompt_bank(args.prompt_bank)
    sampler = PromptControlSampler(
        prompt_records,
        generator,
        unconditional_probability=args.unconditional_prob,
        official_probability=args.official_prompt_prob,
    )
    llama = PromptFeatureTable(
        args.llama_features,
        expected_encoder="llama",
    ).to(device, dtype=torch.bfloat16)
    if len(llama) != len(prompt_records):
        raise ValueError("prompt bank and Llama feature table counts differ")

    generation_config = {
        **{
            name: str(value) if isinstance(value, Path) else value
            for name, value in vars(args).items()
        },
        "rank": rank,
        "local_rank": local_rank,
        "world_size": world_size,
        "rank_seed": rank_seed,
        "resolved_device": str(device),
        "rollout_lengths": list(ROLLOUT_LENGTHS),
        "rollout_length_weights": list(ROLLOUT_LENGTH_WEIGHTS),
        "expected_initial_fraction": 1.0
        / sum(
            length * weight
            for length, weight in zip(ROLLOUT_LENGTHS, ROLLOUT_LENGTH_WEIGHTS)
        ),
        "control_mode_ids": CONTROL_MODE_TO_ID,
        "text_encoder_compute_dtype": "bfloat16",
        "text_feature_storage_dtype": "float16",
    }
    (rank_output / "generation_config.json").write_text(
        json.dumps(generation_config, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    writer = TeacherShardWriter(
        rank_output,
        shard_size=args.shard_size,
        storage_dtype=(
            torch.float16 if args.storage_dtype == "float16" else torch.float32
        ),
        prefix=f"joint-r{rank:02d}",
        extra_fields=(
            "prompt_id",
            "control_mode_id",
            "rollout_depth",
            "prompt_switch",
            "heading_condition",
        ),
        manifest_metadata={
            "prompt_bank_sha256": llama.manifest["prompt_bank_sha256"],
            "llama_feature_manifest": str(args.llama_features / "manifest.json"),
            "control_mode_ids": CONTROL_MODE_TO_ID,
        },
    )

    teacher = load_model(
        args.model,
        device=str(device),
        text_encoder=False,
        checkpoints_dir=str(args.checkpoints_dir),
    ).eval()
    batch = args.batch_size
    prompt_histogram: Counter[int] = Counter()
    family_histogram: Counter[str] = Counter()
    source_histogram: Counter[str] = Counter()
    control_histogram: Counter[int] = Counter()
    depth_histogram: Counter[int] = Counter()
    prompt_switches = 0
    constrained = 0
    start_time = time.perf_counter()
    next_log = args.log_every

    while writer.written + writer.buffered < args.windows:
        rollout_length = sample_rollout_length(generator)
        initial_translation = torch.zeros(batch, 3, device=device)
        initial_translation[:, 0] = torch.rand(batch, device=device) * 20 - 10
        initial_translation[:, 2] = torch.rand(batch, device=device) * 20 - 10
        initial_heading = torch.rand(batch, device=device) * (2 * torch.pi) - torch.pi
        history = None
        prompt_ids = sampler.sample_prompt_ids(batch)

        for rollout_depth in range(rollout_length):
            if writer.written + writer.buffered >= args.windows:
                break
            prompt_switch = torch.zeros(batch, dtype=torch.bool)
            if rollout_depth > 0:
                prompt_switch = (
                    torch.rand(batch, generator=generator) < args.prompt_switch_prob
                )
                replacements = sampler.sample_prompt_ids(batch)
                prompt_ids = torch.where(prompt_switch, replacements, prompt_ids)

            start_xz, start_heading, current_velocity = root_state(
                teacher,
                history,
                initial_translation,
                initial_heading,
            )
            motion_mask, observed_motion, control_mode_ids = make_conditions(
                teacher,
                sampler,
                prompt_ids,
                start_xz,
                start_heading,
                current_velocity,
                has_history=history is not None,
            )
            device_prompt_ids = prompt_ids.to(device)
            text_feat = llama.lookup(device_prompt_ids).unsqueeze(1)
            text_mask = (device_prompt_ids != 0).unsqueeze(1)
            trace = trace_autoregressive_step(
                teacher,
                num_frames=64,
                num_denoising_steps=10,
                motion_mask=motion_mask,
                observed_motion=observed_motion,
                cfg_weight=(args.cfg_text, args.cfg_constraint),
                texts=None,
                text_feat=text_feat,
                text_pad_mask=text_mask,
                init_history_sequence=history,
                init_global_translation=initial_translation if history is None else None,
                init_first_heading_angle=initial_heading if history is None else None,
            )
            heading_condition = pack_heading_condition(
                observed_motion,
                motion_mask,
            ).cpu()
            depth_tensor = torch.full((batch,), rollout_depth, dtype=torch.int64)
            writer.add(
                trace,
                {
                    "prompt_id": prompt_ids,
                    "control_mode_id": control_mode_ids,
                    "rollout_depth": depth_tensor,
                    "prompt_switch": prompt_switch,
                    "heading_condition": heading_condition,
                },
            )
            history = trace.explicit_motion[:, -4:].detach()

            prompt_histogram.update(prompt_ids.tolist())
            control_histogram.update(control_mode_ids.tolist())
            depth_histogram.update(depth_tensor.tolist())
            prompt_switches += int(prompt_switch.sum())
            constrained += int((control_mode_ids != CONTROL_MODE_TO_ID["none"]).sum())
            for prompt_id in prompt_ids.tolist():
                record = prompt_records[prompt_id]
                family_histogram[record["family"]] += 1
                source_histogram[record["source"]] += 1

            current = writer.written + writer.buffered
            if current >= next_log or current >= args.windows:
                elapsed = time.perf_counter() - start_time
                print(
                    json.dumps(
                        {
                            "rank": rank,
                            "world_size": world_size,
                            "windows": current,
                            "target": args.windows,
                            "elapsed_s": round(elapsed, 3),
                            "windows_per_s": round(current / elapsed, 4),
                            "initial_fraction": depth_histogram[0] / current,
                            "constraint_fraction": constrained / current,
                            "prompt_switch_fraction": prompt_switches / current,
                            "unconditional_fraction": source_histogram["unconditional"]
                            / current,
                        }
                    ),
                    flush=True,
                )
                next_log += args.log_every

    writer.close()
    elapsed = time.perf_counter() - start_time
    summary = {
        "schema": "ardy_joint_teacher_generation_v1",
        "rank": rank,
        "world_size": world_size,
        "windows": writer.written,
        "elapsed_s": elapsed,
        "windows_per_s": writer.written / elapsed,
        "initial_fraction": depth_histogram[0] / writer.written,
        "constraint_fraction": constrained / writer.written,
        "prompt_switch_fraction": prompt_switches / writer.written,
        "prompt_source_histogram": dict(sorted(source_histogram.items())),
        "prompt_family_histogram": dict(sorted(family_histogram.items())),
        "control_mode_histogram": {
            str(key): value for key, value in sorted(control_histogram.items())
        },
        "rollout_depth_histogram": {
            str(key): value for key, value in sorted(depth_histogram.items())
        },
        "unique_prompt_ids": len(prompt_histogram),
    }
    (rank_output / "joint_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps({"event": "joint_teacher_generation_complete", **summary}),
        flush=True,
    )


if __name__ == "__main__":
    main()
