#!/usr/bin/env python
"""Merge disjoint prompt-feature manifests without duplicating shard data."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from ardy_distill.data import sha256  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parts-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--encoder", choices=["qwen", "llama"], required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifests = sorted(args.parts_root.rglob("manifest.json"))
    manifests = [
        path
        for path in manifests
        if json.loads(path.read_text(encoding="utf-8")).get("encoder") == args.encoder
    ]
    if not manifests:
        raise FileNotFoundError(f"no {args.encoder} feature manifests under {args.parts_root}")

    parts = [json.loads(path.read_text(encoding="utf-8")) for path in manifests]
    invariant_keys = (
        "schema",
        "encoder",
        "prompt_bank",
        "prompt_bank_sha256",
        "storage_dtype",
        "max_sequence_length",
        "qwen_hidden_state_layers",
        "pooling",
        "empty_prompt_is_zero",
        "feature_dim",
    )
    reference = parts[0]
    for path, part in zip(manifests, parts):
        if not part.get("complete"):
            raise RuntimeError(f"feature part is incomplete: {path}")
        for key in invariant_keys:
            if part.get(key) != reference.get(key):
                raise ValueError(f"feature part mismatch for {key}: {path}")

    ordered: list[tuple[int, int, Path, dict]] = []
    for manifest_path, part in zip(manifests, parts):
        for shard in part["shards"]:
            ordered.append(
                (
                    int(shard["start_prompt_id"]),
                    int(shard["end_prompt_id"]),
                    manifest_path.parent / shard["file"],
                    shard,
                )
            )
    ordered.sort(key=lambda item: item[0])
    expected = 0
    for start, end, path, shard in ordered:
        if start != expected or end < start:
            raise ValueError(
                f"non-contiguous prompt feature coverage at {path}: "
                f"expected {expected}, got [{start}, {end}]"
            )
        if end - start + 1 != int(shard["count"]):
            raise ValueError(f"count/id range mismatch: {path}")
        if sha256(path) != shard["sha256"]:
            raise ValueError(f"feature shard checksum mismatch: {path}")
        expected = end + 1

    source_counts = {
        int(part.get("source_prompt_count", part["planned_count"])) for part in parts
    }
    if len(source_counts) != 1 or expected != next(iter(source_counts)):
        raise ValueError(
            f"merged coverage {expected} does not match source counts {sorted(source_counts)}"
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "manifest.json"
    if manifest_path.exists():
        raise FileExistsError(f"refusing to overwrite {manifest_path}")
    merged_shards = []
    for index, (start, end, source, shard) in enumerate(ordered):
        filename = f"features-{index:05d}.safetensors"
        target = args.output_dir / filename
        if target.exists():
            raise FileExistsError(f"refusing to overwrite {target}")
        os.link(source, target)
        merged_shards.append({**shard, "file": filename})

    merged = {
        **{key: reference.get(key) for key in invariant_keys},
        "planned_count": expected,
        "count": expected,
        "complete": True,
        "batch_sizes_by_part": [int(part["batch_size"]) for part in parts],
        "chunk_sizes_by_part": [int(part["chunk_size"]) for part in parts],
        "parts": [str(path) for path in manifests],
        "shards": merged_shards,
    }
    manifest_path.write_text(
        json.dumps(merged, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "event": "prompt_features_merged",
                "encoder": args.encoder,
                "count": expected,
                "parts": len(parts),
                "shards": len(merged_shards),
                "output": str(args.output_dir),
            }
        )
    )


if __name__ == "__main__":
    main()
