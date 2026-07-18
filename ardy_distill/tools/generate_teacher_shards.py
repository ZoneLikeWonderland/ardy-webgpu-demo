#!/usr/bin/env python
"""Generate bounded path-only teacher shards with the released FP32 sampler."""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "ardy"))

from ardy.constraints import Root2DConstraintSet  # noqa: E402
from ardy.model import load_model  # noqa: E402
from ardy_distill.data import TeacherShardWriter  # noqa: E402
from ardy_distill.teacher import trace_autoregressive_step  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checkpoints-dir", type=Path, default=PROJECT_ROOT / "ardy" / "checkpoints")
    parser.add_argument("--model", default="ARDY-Core-RP-20FPS-Horizon40")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--windows", type=int, default=256, help="Windows per distributed rank")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--rollout-steps", type=int, default=8)
    parser.add_argument("--shard-size", type=int, default=128)
    parser.add_argument("--storage-dtype", choices=["float16", "float32"], default="float16")
    parser.add_argument("--seed", type=int, default=20260714)
    parser.add_argument("--cfg-constraint", type=float, default=1.5)
    parser.add_argument("--no-constraint-prob", type=float, default=0.25)
    parser.add_argument("--log-every", type=int, default=256, help="Progress interval per rank")
    return parser.parse_args()


def distributed_context() -> tuple[int, int, int]:
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    return rank, local_rank, world_size


def current_root_xz(model, history: torch.Tensor | None, initial_translation: torch.Tensor) -> torch.Tensor:
    if history is None:
        return initial_translation[:, [0, 2]]
    unnormalized = model.motion_rep.unnormalize(history)
    return model.motion_rep.get_root_pos(unnormalized)[:, -1, [0, 2]]


def random_constraint_lists(
    model,
    start_xz: torch.Tensor,
    generator: torch.Generator,
    no_constraint_prob: float,
) -> list[list[Root2DConstraintSet]]:
    result: list[list[Root2DConstraintSet]] = []
    for sample in range(start_xz.shape[0]):
        if torch.rand((), generator=generator).item() < no_constraint_prob:
            result.append([])
            continue
        multi = torch.rand((), generator=generator).item() < 0.2
        frame_indices = torch.tensor([24, 40, 60] if multi else [60], dtype=torch.long)
        targets = []
        position = start_xz[sample].detach().cpu()
        direction = (torch.rand((), generator=generator) * (2 * torch.pi)).item()
        previous_frame = 0
        for frame_index in frame_indices.tolist():
            seconds = (frame_index - previous_frame) / 20.0
            speed = (0.45 + 1.15 * torch.rand((), generator=generator)).item()
            direction += (torch.randn((), generator=generator) * (0.55 if multi else 0.85)).item()
            delta = torch.tensor(
                [np.sin(direction), np.cos(direction)],
                dtype=torch.float32,
            ) * (seconds * speed)
            position = position + delta
            targets.append(position.clone())
            previous_frame = frame_index
        result.append(
            [
                Root2DConstraintSet(
                    model.skeleton,
                    frame_indices=frame_indices,
                    root_2d=torch.stack(targets),
                )
            ]
        )
    return result


def main() -> None:
    args = parse_args()
    rank, local_rank, world_size = distributed_context()
    if args.windows < args.batch_size:
        raise ValueError("--windows must be at least --batch-size")
    if args.rollout_steps < 1:
        raise ValueError("--rollout-steps must be positive")
    if args.log_every < 1:
        raise ValueError("--log-every must be positive")

    device = f"cuda:{local_rank}" if args.device == "auto" else args.device
    rank_seed = args.seed + rank * 1_000_003
    random.seed(rank_seed)
    np.random.seed(rank_seed)
    torch.manual_seed(rank_seed)
    if device.startswith("cuda"):
        torch.cuda.manual_seed_all(rank_seed)
    condition_generator = torch.Generator(device="cpu").manual_seed(rank_seed + 17)

    output_dir = args.output_dir / f"rank-{rank:02d}"
    output_dir.mkdir(parents=True, exist_ok=True)
    if (output_dir / "manifest.json").exists():
        raise FileExistsError(f"refusing to overwrite existing teacher shards: {output_dir}")
    (output_dir / "generation_config.json").write_text(
        json.dumps(
            {
                **{name: str(value) if isinstance(value, Path) else value for name, value in vars(args).items()},
                "rank": rank,
                "local_rank": local_rank,
                "world_size": world_size,
                "rank_seed": rank_seed,
                "resolved_device": device,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    storage_dtype = torch.float16 if args.storage_dtype == "float16" else torch.float32
    writer = TeacherShardWriter(
        output_dir,
        shard_size=args.shard_size,
        storage_dtype=storage_dtype,
        prefix=f"teacher-r{rank:02d}",
    )

    start_time = time.perf_counter()
    model = load_model(
        args.model,
        device=device,
        text_encoder=False,
        checkpoints_dir=str(args.checkpoints_dir),
    )
    model.eval()
    batch = args.batch_size
    text_feat = torch.zeros(batch, 1, 4096, device=device, dtype=torch.float32)
    text_mask = torch.zeros(batch, 1, device=device, dtype=torch.bool)

    while writer.written + writer.buffered < args.windows:
        initial_translation = torch.zeros(batch, 3, device=device)
        initial_translation[:, 0] = torch.rand(batch, device=device) * 10 - 5
        initial_translation[:, 2] = torch.rand(batch, device=device) * 10 - 5
        initial_heading = torch.rand(batch, device=device) * (2 * torch.pi) - torch.pi
        history = None

        for _ in range(args.rollout_steps):
            if writer.written + writer.buffered >= args.windows:
                break
            start_xz = current_root_xz(model, history, initial_translation)
            constraints = random_constraint_lists(
                model,
                start_xz,
                condition_generator,
                args.no_constraint_prob,
            )
            lengths = torch.full((batch,), 64, device=device, dtype=torch.long)
            observed_motion, motion_mask = model.motion_rep.create_conditions_from_constraints_batched(
                constraints,
                lengths,
                to_normalize=True,
                device=device,
            )
            trace = trace_autoregressive_step(
                model,
                num_frames=64,
                num_denoising_steps=10,
                motion_mask=motion_mask,
                observed_motion=observed_motion,
                cfg_weight=(0.0, args.cfg_constraint),
                texts=None,
                text_feat=text_feat,
                text_pad_mask=text_mask,
                init_history_sequence=history,
                init_global_translation=initial_translation if history is None else None,
                init_first_heading_angle=initial_heading if history is None else None,
            )
            writer.add(trace)
            history = trace.explicit_motion[:, -4:].detach()

            elapsed = time.perf_counter() - start_time
            current = writer.written + writer.buffered
            if current % args.log_every == 0 or current >= args.windows:
                print(
                    json.dumps(
                        {
                            "rank": rank,
                            "world_size": world_size,
                            "windows": current,
                            "target": args.windows,
                            "elapsed_s": round(elapsed, 3),
                            "windows_per_s": round(current / elapsed, 4),
                        }
                    ),
                    flush=True,
                )

    writer.close()
    elapsed = time.perf_counter() - start_time
    print(
        json.dumps(
            {
                "event": "teacher_generation_complete",
                "rank": rank,
                "world_size": world_size,
                "windows": writer.written,
                "elapsed_s": round(elapsed, 3),
                "output_dir": str(output_dir),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
