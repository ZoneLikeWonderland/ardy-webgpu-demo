#!/usr/bin/env python
"""Prove that the DMD teacher-score adapter reproduces stored DDIM states."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "ardy"))

from ardy.model import load_model  # noqa: E402
from ardy_distill.data import TeacherShardDataset  # noqa: E402
from ardy_distill.dmd2 import TeacherScoreAdapter  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument(
        "--checkpoints-dir", type=Path, default=PROJECT_ROOT / "ardy" / "checkpoints"
    )
    parser.add_argument("--model", default="ARDY-Core-RP-20FPS-Horizon40")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--cfg-constraint", type=float, default=1.5)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def ddim_step(
    xt: torch.Tensor,
    predicted_x0: torch.Tensor,
    timestep: int,
    alphas_cumprod: torch.Tensor,
) -> torch.Tensor:
    alpha = alphas_cumprod[timestep].float()
    alpha_prev = (
        alphas_cumprod[timestep - 1].float()
        if timestep > 0
        else alphas_cumprod.new_tensor(1.0).float()
    )
    epsilon = (xt / alpha.sqrt() - predicted_x0) / torch.sqrt(1.0 / alpha - 1.0)
    return predicted_x0 * alpha_prev.sqrt() + (1.0 - alpha_prev).sqrt() * epsilon


def main() -> None:
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")
    device = torch.device(args.device)
    model = load_model(
        args.model,
        device=str(device),
        text_encoder=False,
        checkpoints_dir=str(args.checkpoints_dir),
    ).eval()
    adapter = TeacherScoreAdapter(model, cfg_constraint=args.cfg_constraint)
    dataset = TeacherShardDataset(args.data, cache_shards=1)
    batch = next(
        iter(
            DataLoader(
                dataset,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=0,
            )
        )
    )
    batch = {name: value.to(device=device).float() for name, value in batch.items()}

    per_step = []
    for state_index, timestep in enumerate(range(9, -1, -1)):
        xt = batch["denoising_states"][:, state_index]
        prediction = adapter.predict_x0(
            xt,
            batch["history_hybrid"],
            batch["path_condition"],
            batch["first_heading"],
            batch["has_history"],
            torch.full((args.batch_size,), timestep, device=device, dtype=torch.long),
        )
        reconstructed_next = ddim_step(
            xt,
            prediction,
            timestep,
            adapter.alphas_cumprod,
        )
        expected_next = batch["denoising_states"][:, state_index + 1]
        error = (reconstructed_next - expected_next).abs()
        per_step.append(
            {
                "timestep": timestep,
                "max_abs_error": float(error.max().item()),
                "mean_abs_error": float(error.mean().item()),
            }
        )

    packed = adapter.pack_inputs(
        batch["denoising_states"][:, 0],
        batch["history_hybrid"],
        batch["path_condition"],
        batch["first_heading"],
        batch["has_history"],
    )
    max_abs = max(record["max_abs_error"] for record in per_step)
    result = {
        "schema": "ardy_teacher_score_adapter_validation_v1",
        "batch_size": args.batch_size,
        "max_abs_error": max_abs,
        "mean_abs_error": sum(record["mean_abs_error"] for record in per_step)
        / len(per_step),
        "passed": max_abs <= 5.0e-5,
        "history_token_counts": packed.history_token_mask.sum(dim=1).cpu().tolist(),
        "generation_token_counts": packed.generation_token_mask.sum(dim=1).cpu().tolist(),
        "future_token_counts": packed.future_token_mask.sum(dim=1).cpu().tolist(),
        "per_step": per_step,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
