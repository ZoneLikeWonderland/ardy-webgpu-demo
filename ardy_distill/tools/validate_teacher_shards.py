#!/usr/bin/env python
"""Validate teacher shard manifests, hashes, tensor schemas and finite values."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from ardy_distill.data import (  # noqa: E402
    OPTIONAL_SHARD_FIELDS,
    SHARD_FIELDS,
    SUPPORTED_SHARD_FIELDS,
    sha256,
)


EXPECTED_TRAILING_SHAPES = {
    "initial_noise": (10, 148),
    "denoising_states": (11, 10, 148),
    "clean_generation": (10, 148),
    "history_hybrid": (1, 148),
    "path_condition": (64, 3),
    "first_heading": (2,),
    "has_history": (1,),
    "encoder_body": (4, 325),
    "encoder_valid": (1,),
    "decoder_latent": (11, 128),
    "decoder_global_root": (44, 5),
    "decoder_local_root": (44, 4),
    "decoder_token_valid": (11,),
    "target_body": (44, 325),
    "global_translation": (3,),
    "prompt_id": (),
    "control_mode_id": (),
    "rollout_depth": (),
    "prompt_switch": (),
    "heading_condition": (64, 3),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--skip-hashes", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifests = sorted(args.root.rglob("manifest.json"))
    if not manifests:
        raise FileNotFoundError(f"no manifests under {args.root}")

    total_count = 0
    total_bytes = 0
    history_count = 0.0
    constrained_count = 0.0
    moments = {
        name: {"sum": 0.0, "sum2": 0.0, "count": 0}
        for name in ("initial_noise", "clean_generation")
    }
    shard_count = 0
    storage_dtypes: set[str] = set()
    prompt_histogram: dict[int, int] = {}
    control_mode_histogram: dict[int, int] = {}
    rollout_depth_histogram: dict[int, int] = {}
    prompt_switch_count = 0
    teacher_manifests = 0

    for manifest_path in manifests:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("schema") not in {
            "ardy_teacher_window_v2",
            "ardy_teacher_window_v3",
        }:
            continue
        teacher_manifests += 1
        storage_dtype = str(manifest["storage_dtype"])
        storage_dtypes.add(storage_dtype)
        fields = tuple(manifest["fields"])
        if not set(SHARD_FIELDS).issubset(fields) or not set(fields).issubset(
            SUPPORTED_SHARD_FIELDS
        ):
            raise ValueError(f"field schema mismatch in {manifest_path}")
        manifest_count = 0
        for record in manifest["shards"]:
            path = manifest_path.parent / record["file"]
            if path.stat().st_size != int(record["bytes"]):
                raise ValueError(f"size mismatch: {path}")
            if not args.skip_hashes and sha256(path) != record["sha256"]:
                raise ValueError(f"SHA256 mismatch: {path}")
            tensors = load_file(path, device="cpu")
            if set(tensors) != set(fields):
                raise ValueError(f"tensor fields mismatch: {path}")
            count = int(record["count"])
            for name in fields:
                tensor = tensors[name]
                if tensor.shape[0] != count or tuple(tensor.shape[1:]) != EXPECTED_TRAILING_SHAPES[name]:
                    raise ValueError(f"shape mismatch {path}:{name}: {tuple(tensor.shape)}")
                if tensor.is_floating_point() and not torch.isfinite(tensor).all():
                    raise ValueError(f"NaN/Inf in {path}:{name}")
                if (
                    name not in OPTIONAL_SHARD_FIELDS
                    and tensor.is_floating_point()
                    and str(tensor.dtype).removeprefix("torch.") != storage_dtype
                ):
                    raise ValueError(f"dtype mismatch {path}:{name}: {tensor.dtype} != {storage_dtype}")
            history_count += float(tensors["has_history"].sum())
            constrained_count += float((tensors["path_condition"][..., 2].sum(dim=1) > 0).sum())
            for name, accum in moments.items():
                values = tensors[name].double()
                accum["sum"] += float(values.sum())
                accum["sum2"] += float(values.square().sum())
                accum["count"] += values.numel()
            for field, histogram in (
                ("prompt_id", prompt_histogram),
                ("control_mode_id", control_mode_histogram),
                ("rollout_depth", rollout_depth_histogram),
            ):
                if field in tensors:
                    values, counts = torch.unique(tensors[field].long(), return_counts=True)
                    for value, frequency in zip(values.tolist(), counts.tolist()):
                        histogram[value] = histogram.get(value, 0) + frequency
            if "prompt_switch" in tensors:
                prompt_switch_count += int(tensors["prompt_switch"].bool().sum())
            manifest_count += count
            total_count += count
            total_bytes += int(record["bytes"])
            shard_count += 1
        if manifest_count != int(manifest["count"]):
            raise ValueError(f"manifest count mismatch: {manifest_path}")

    if teacher_manifests == 0:
        raise FileNotFoundError(f"no teacher manifests under {args.root}")
    stats = {}
    for name, accum in moments.items():
        mean = accum["sum"] / accum["count"]
        variance = max(0.0, accum["sum2"] / accum["count"] - mean * mean)
        stats[name] = {"mean": mean, "std": variance**0.5}
    result = {
        "schema": "ardy_teacher_validation_v1",
        "root": str(args.root),
        "valid": True,
        "manifests": teacher_manifests,
        "shards": shard_count,
        "count": total_count,
        "bytes": total_bytes,
        "storage_dtypes": sorted(storage_dtypes),
        "history_fraction": history_count / total_count,
        "constrained_fraction": constrained_count / total_count,
        "prompt_histogram": dict(sorted(prompt_histogram.items())),
        "control_mode_histogram": dict(sorted(control_mode_histogram.items())),
        "rollout_depth_histogram": dict(sorted(rollout_depth_histogram.items())),
        "prompt_switch_fraction": prompt_switch_count / total_count,
        "stats": stats,
        "hashes_checked": not args.skip_hashes,
    }
    encoded = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")


if __name__ == "__main__":
    main()
