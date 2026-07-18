#!/usr/bin/env python
"""Fixed-corpus text/control validation for the compact ARDY flow.

This evaluator complements :mod:`ardy_distill.evaluate`: the latter measures
decoder/FK/temporal quality, while this tool keeps per-sample endpoint metrics
so they can be stratified by prompt, control mode, rollout depth and history.
It also runs zero/shuffled-text and zero-heading counterfactuals to verify that
the student actually uses the newly introduced conditions.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from ardy_distill.codec_cli import add_codec_config_arguments, codec_config_from_args
from ardy_distill.data import TeacherShardDataset
from ardy_distill.losses import FSQRequantizer
from ardy_distill.models import FlowStudentConfig, HistoryEncoderStudent, OneStepFlowStudent
from ardy_distill.runtime import load_motion_rep, load_safetensor_weights
from ardy_distill.text_features import PromptFeatureTable


CONTROL_NAMES = {
    0: "none",
    1: "mouse_sparse",
    2: "mouse_dense",
    3: "keyboard_velocity",
    4: "keyboard_heading",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--flow", type=Path, required=True)
    parser.add_argument("--encoder", type=Path, required=True)
    parser.add_argument("--text-features", type=Path, required=True)
    parser.add_argument("--prompt-bank", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path("ardy/checkpoints/ARDY-Core-RP-20FPS-Horizon40"),
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--model-dtype", choices=["fp32", "fp16", "bf16"], default="fp16")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-batches", type=int, default=0)
    parser.add_argument("--flow-width", type=int, default=512)
    parser.add_argument("--flow-heads", type=int, default=8)
    parser.add_argument("--flow-trunk-blocks", type=int, default=8)
    parser.add_argument("--flow-body-blocks", type=int, default=8)
    parser.add_argument("--flow-steps", type=int, default=1)
    parser.add_argument("--heading-condition-features", type=int, choices=[3], default=3)
    add_codec_config_arguments(parser)
    return parser.parse_args()


class GroupAccumulator:
    def __init__(self) -> None:
        self.samples: dict[str, int] = defaultdict(int)
        self.sums: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
        self.counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

    def add(
        self,
        groups: dict[str, torch.Tensor],
        metrics: dict[str, torch.Tensor],
    ) -> None:
        for group, mask in groups.items():
            mask = mask.bool()
            self.samples[group] += int(mask.sum())
            for name, values in metrics.items():
                valid = mask & torch.isfinite(values)
                count = int(valid.sum())
                if count:
                    self.sums[group][name] += float(values[valid].double().sum())
                    self.counts[group][name] += count

    def result(self) -> dict[str, object]:
        output: dict[str, object] = {}
        for group in sorted(self.samples):
            output[group] = {
                "samples": self.samples[group],
                "metrics": {
                    name: {
                        "mean": self.sums[group][name] / self.counts[group][name],
                        "count": self.counts[group][name],
                    }
                    for name in sorted(self.sums[group])
                },
            }
        return output


def load_prompt_families(path: Path) -> tuple[torch.Tensor, list[str]]:
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    if [int(record["prompt_id"]) for record in records] != list(range(len(records))):
        raise ValueError("prompt bank ids must be contiguous")
    names = sorted({str(record["family"]) for record in records})
    name_to_id = {name: index for index, name in enumerate(names)}
    ids = torch.tensor([name_to_id[str(record["family"])] for record in records], dtype=torch.long)
    return ids, names


def endpoint_metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
    quantizer: FSQRequantizer,
) -> dict[str, torch.Tensor]:
    root = prediction[..., :20] - target[..., :20]
    body = prediction[..., 20:] - target[..., 20:]
    root_mse = root.square().mean(dim=(1, 2))
    body_mse = body.square().mean(dim=(1, 2))
    root_l1 = root.abs().mean(dim=(1, 2))
    body_l1 = body.abs().mean(dim=(1, 2))
    pred_bin = (
        quantizer.unnormalize(prediction[..., 20:]).clamp(-1, 1) * quantizer.half_width
    ).round()
    target_bin = (
        quantizer.unnormalize(target[..., 20:]).clamp(-1, 1) * quantizer.half_width
    ).round()
    return {
        "endpoint_root_mse": root_mse,
        "endpoint_body_mse": body_mse,
        "endpoint_root_l1": root_l1,
        "endpoint_body_l1": body_l1,
        "endpoint_total": 2.0 * root_mse + body_mse + 0.25 * (root_l1 + body_l1),
        "fsq_bin_accuracy": (pred_bin == target_bin).float().mean(dim=(1, 2)),
    }


def path_metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
    path_condition: torch.Tensor,
    has_history: torch.Tensor,
    motion_rep,
) -> dict[str, torch.Tensor]:
    batch = prediction.shape[0]
    pred_root = prediction[..., :20].reshape(batch, 40, 5).float()
    target_root = target[..., :20].reshape(batch, 40, 5).float()
    frames = torch.arange(64, device=prediction.device).unsqueeze(0).expand(batch, -1)
    generation_start = (has_history.reshape(batch, 1) > 0.5).long() * 4
    generation_index = frames - generation_start
    within = (generation_index >= 0) & (generation_index < 40)
    valid = within & (path_condition[..., 2] > 0.5)
    gather_index = generation_index.clamp(0, 39)
    pred_xz = pred_root[..., [0, 2]].gather(
        1, gather_index.unsqueeze(-1).expand(-1, -1, 2)
    )
    target_xz = target_root[..., [0, 2]].gather(
        1, gather_index.unsqueeze(-1).expand(-1, -1, 2)
    )
    wanted_xz = path_condition[..., :2].float()
    indices = torch.tensor([0, 2], device=prediction.device)
    mean = motion_rep.global_root_stats.mean.to(prediction.device)[indices]
    scale = motion_rep.global_root_stats.std_eps.to(prediction.device)[indices]

    def meters(values: torch.Tensor) -> torch.Tensor:
        return values.float() * scale + mean

    pred_error = torch.linalg.vector_norm(meters(pred_xz) - meters(wanted_xz), dim=-1)
    teacher_error = torch.linalg.vector_norm(meters(target_xz) - meters(wanted_xz), dim=-1)
    count = valid.sum(dim=1)
    denominator = count.clamp_min(1)
    pred_mean = (pred_error * valid).sum(dim=1) / denominator
    teacher_mean = (teacher_error * valid).sum(dim=1) / denominator
    nan = torch.full_like(pred_mean, float("nan"))
    return {
        "path_error_m": torch.where(count > 0, pred_mean, nan),
        "teacher_path_error_m": torch.where(count > 0, teacher_mean, nan),
        "path_target_count": count.float(),
    }


def build_groups(
    batch: dict[str, torch.Tensor],
    family_lookup: torch.Tensor,
    family_names: list[str],
) -> dict[str, torch.Tensor]:
    prompt = batch["prompt_id"].long()
    control = batch["control_mode_id"].long()
    depth = batch["rollout_depth"].long()
    history = batch["has_history"].reshape(-1) > 0.5
    switched = batch["prompt_switch"].reshape(-1) > 0
    groups = {
        "all": torch.ones_like(prompt, dtype=torch.bool),
        "prompt/unconditional": prompt == 0,
        "prompt/official": (prompt >= 1) & (prompt <= 10),
        "prompt/generated": prompt >= 11,
        "rollout/initial": depth == 0,
        "rollout/depth_1": depth == 1,
        "rollout/depth_2_3": (depth >= 2) & (depth <= 3),
        "rollout/depth_4_7": (depth >= 4) & (depth <= 7),
        "rollout/depth_8_15": (depth >= 8) & (depth <= 15),
        "history/false": ~history,
        "history/true": history,
        "prompt_switch/false": ~switched,
        "prompt_switch/true": switched,
        "heading/valid": batch["heading_condition"][..., 2].sum(dim=1) > 0,
    }
    for control_id, name in CONTROL_NAMES.items():
        groups[f"control/{name}"] = control == control_id
    family_id = family_lookup[prompt]
    for index, name in enumerate(family_names):
        groups[f"family/{name}"] = family_id == index
    return groups


def main() -> None:
    args = parse_args()
    if args.flow_steps < 1:
        raise ValueError("--flow-steps must be positive")
    device = torch.device(args.device)
    dtype = {
        "fp32": torch.float32,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }[args.model_dtype]
    if dtype == torch.float16 and device.type != "cuda":
        raise ValueError("FP16 validation requires CUDA")

    text_features = PromptFeatureTable(
        args.text_features,
        expected_encoder="qwen",
    ).to(device, dtype=dtype)
    family_lookup, family_names = load_prompt_families(args.prompt_bank)
    if len(family_lookup) != len(text_features):
        raise ValueError("prompt bank and feature table lengths differ")
    family_lookup = family_lookup.to(device)

    flow_config = FlowStudentConfig(
        width=args.flow_width,
        heads=args.flow_heads,
        trunk_blocks=args.flow_trunk_blocks,
        body_blocks=args.flow_body_blocks,
        text_feature_dim=text_features.feature_dim,
        heading_condition_features=args.heading_condition_features,
    )
    codec_config = codec_config_from_args(args)
    flow = OneStepFlowStudent(flow_config).to(device=device, dtype=dtype).eval()
    encoder = HistoryEncoderStudent(codec_config).to(device=device, dtype=dtype).eval()
    load_safetensor_weights(flow, args.flow)
    load_safetensor_weights(encoder, args.encoder)
    quantizer = FSQRequantizer(args.checkpoint_dir / "stats/post_quantization").to(device)
    motion_rep = load_motion_rep(args.checkpoint_dir)

    fields = (
        "initial_noise",
        "clean_generation",
        "history_hybrid",
        "path_condition",
        "first_heading",
        "has_history",
        "prompt_id",
        "control_mode_id",
        "rollout_depth",
        "prompt_switch",
        "heading_condition",
        "encoder_body",
    )
    dataset = TeacherShardDataset(args.data, fields=fields, cache_shards=4)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    accumulator = GroupAccumulator()
    samples = 0
    start = time.perf_counter()
    with torch.inference_mode():
        for batch_index, batch in enumerate(loader):
            if args.max_batches and batch_index >= args.max_batches:
                break
            batch = {
                name: value.to(device, non_blocking=True)
                for name, value in batch.items()
            }
            batch_size = batch["initial_noise"].shape[0]
            samples += batch_size
            encoded_history = encoder(batch["encoder_body"].to(dtype=dtype)).float()
            encoded_history = quantizer(encoded_history, ste=False)
            history_valid = batch["has_history"].float().unsqueeze(-1)
            history_hybrid = torch.cat(
                [
                    batch["history_hybrid"][..., :20].float() * history_valid,
                    encoded_history * history_valid,
                ],
                dim=-1,
            )
            prompt_feature = text_features.lookup(batch["prompt_id"])
            heading_condition = batch["heading_condition"].to(dtype=dtype)

            def predict(
                text: torch.Tensor,
                heading: torch.Tensor,
            ) -> torch.Tensor:
                return flow.denoise_steps(
                    batch["initial_noise"].to(dtype=dtype),
                    history_hybrid.to(dtype=dtype),
                    batch["path_condition"].to(dtype=dtype),
                    batch["first_heading"].to(dtype=dtype),
                    batch["has_history"].to(dtype=dtype),
                    steps=args.flow_steps,
                    text_feature=text,
                    heading_condition=heading,
                ).float()

            prediction = predict(prompt_feature, heading_condition)
            zero_text_prediction = predict(torch.zeros_like(prompt_feature), heading_condition)
            shuffled_text_prediction = predict(prompt_feature.roll(1, dims=0), heading_condition)
            zero_heading_prediction = predict(prompt_feature, torch.zeros_like(heading_condition))
            for name, tensor in {
                "prediction": prediction,
                "zero_text_prediction": zero_text_prediction,
                "shuffled_text_prediction": shuffled_text_prediction,
                "zero_heading_prediction": zero_heading_prediction,
            }.items():
                if not torch.isfinite(tensor).all():
                    raise RuntimeError(f"non-finite {name} at batch {batch_index}")

            target = batch["clean_generation"].float()
            metrics = endpoint_metrics(prediction, target, quantizer)
            zero_text = endpoint_metrics(zero_text_prediction, target, quantizer)
            shuffled_text = endpoint_metrics(shuffled_text_prediction, target, quantizer)
            zero_heading = endpoint_metrics(zero_heading_prediction, target, quantizer)
            metrics.update(path_metrics(
                prediction,
                target,
                batch["path_condition"].float(),
                batch["has_history"].float(),
                motion_rep,
            ))
            metrics.update(
                {
                    "zero_text_endpoint_total": zero_text["endpoint_total"],
                    "shuffled_text_endpoint_total": shuffled_text["endpoint_total"],
                    "zero_heading_endpoint_total": zero_heading["endpoint_total"],
                    "text_gain_over_zero": zero_text["endpoint_total"] - metrics["endpoint_total"],
                    "text_gain_over_shuffle": shuffled_text["endpoint_total"] - metrics["endpoint_total"],
                    "heading_gain_over_zero": zero_heading["endpoint_total"] - metrics["endpoint_total"],
                    "prediction_delta_zero_text_l1": (
                        prediction - zero_text_prediction
                    ).abs().mean(dim=(1, 2)),
                    "prediction_delta_shuffled_text_l1": (
                        prediction - shuffled_text_prediction
                    ).abs().mean(dim=(1, 2)),
                    "prediction_delta_zero_heading_l1": (
                        prediction - zero_heading_prediction
                    ).abs().mean(dim=(1, 2)),
                }
            )
            groups = build_groups(batch, family_lookup, family_names)
            accumulator.add(groups, metrics)

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start
    result = {
        "schema": "ardy_text_control_eval_v1",
        "data": str(args.data),
        "weights": {"flow": str(args.flow), "encoder": str(args.encoder)},
        "text_features": str(args.text_features),
        "prompt_bank": str(args.prompt_bank),
        "model_dtype": args.model_dtype,
        "flow_steps": args.flow_steps,
        "samples": samples,
        "batch_size": args.batch_size,
        "elapsed_s": elapsed,
        "samples_per_s": samples / elapsed,
        "flow_config": flow_config.__dict__,
        "groups": accumulator.result(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    all_metrics = result["groups"]["all"]["metrics"]
    print(json.dumps(
        {
            "event": "text_control_validation_complete",
            "output": str(args.output),
            "samples": samples,
            "elapsed_s": elapsed,
            "endpoint_total": all_metrics["endpoint_total"]["mean"],
            "text_gain_over_zero": all_metrics["text_gain_over_zero"]["mean"],
            "heading_gain_over_zero": all_metrics["heading_gain_over_zero"]["mean"],
        },
        ensure_ascii=False,
    ))


if __name__ == "__main__":
    main()
