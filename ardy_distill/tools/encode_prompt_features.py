#!/usr/bin/env python
"""Encode a prompt bank into resumable Llama-teacher or Qwen-student shards."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import torch
from safetensors.torch import save_file


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "ardy"))

from ardy_distill.text_features import (  # noqa: E402
    LLM2VEC_FEATURE_DIM,
    QWEN_FEATURE_DIM,
    ArdyLlamaFeatureEncoder,
    Flux2QwenFeatureEncoder,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt-bank", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--encoder", choices=["qwen", "llama"], required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--chunk-size", type=int, default=64)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--start-id", type=int, default=0)
    parser.add_argument("--end-id", type=int)
    parser.add_argument("--max-sequence-length", type=int, default=128)
    parser.add_argument(
        "--flux2-root",
        type=Path,
        default=Path("/mnt/newdisk/HUGGINGFACE_MODELS/black-forest-labs/FLUX.2-klein-4B"),
    )
    parser.add_argument(
        "--models-root",
        type=Path,
        default=Path("/mnt/newdisk/HUGGINGFACE_MODELS"),
    )
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def load_prompt_bank(
    path: Path,
    limit: int | None,
    start_id: int,
    end_id: int | None,
) -> tuple[list[dict], int]:
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    if [record["prompt_id"] for record in records] != list(range(len(records))):
        raise ValueError("prompt ids must be contiguous and ordered from zero")
    source_count = len(records)
    if start_id < 0 or start_id >= source_count:
        raise ValueError("--start-id is outside the prompt bank")
    if end_id is None:
        end_id = source_count
    if not start_id < end_id <= source_count:
        raise ValueError("--end-id must be greater than start and inside the prompt bank")
    if limit is not None and (start_id != 0 or end_id != source_count):
        raise ValueError("--limit cannot be combined with a prompt-id range")
    records = records[start_id:end_id]
    if limit is not None:
        if limit < 1:
            raise ValueError("--limit must be positive")
        records = records[:limit]
    return records, source_count


def build_encoder(args: argparse.Namespace):
    if args.encoder == "qwen":
        return Flux2QwenFeatureEncoder(
            args.flux2_root,
            args.device,
            max_sequence_length=args.max_sequence_length,
        ), QWEN_FEATURE_DIM
    return ArdyLlamaFeatureEncoder(args.models_root, args.device), LLM2VEC_FEATURE_DIM


def main() -> None:
    args = parse_args()
    if args.batch_size < 1 or args.chunk_size < 1:
        raise ValueError("batch and chunk sizes must be positive")
    if args.chunk_size % args.batch_size:
        raise ValueError("--chunk-size must be divisible by --batch-size")
    prompt_bank = args.prompt_bank.resolve()
    records, source_count = load_prompt_bank(
        prompt_bank,
        args.limit,
        args.start_id,
        args.end_id,
    )
    bank_sha = file_sha256(prompt_bank)
    output = args.output_dir.resolve() / args.encoder
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "manifest.json"
    configuration = {
        "schema": "ardy_prompt_features_v1",
        "encoder": args.encoder,
        "prompt_bank": str(prompt_bank),
        "prompt_bank_sha256": bank_sha,
        "planned_count": len(records),
        "storage_dtype": "float16",
        "batch_size": args.batch_size,
        "chunk_size": args.chunk_size,
        "max_sequence_length": args.max_sequence_length if args.encoder == "qwen" else None,
        "qwen_hidden_state_layers": [9, 18, 27] if args.encoder == "qwen" else None,
        "pooling": "valid_token_mean",
        "empty_prompt_is_zero": True,
    }
    range_is_full = records[0]["prompt_id"] == 0 and records[-1]["prompt_id"] == source_count - 1
    if not range_is_full:
        configuration.update(
            {
                "source_prompt_count": source_count,
                "start_prompt_id": records[0]["prompt_id"],
                "end_prompt_id_exclusive": records[-1]["prompt_id"] + 1,
            }
        )
    shard_records: list[dict] = []
    completed = 0
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        for key, expected in configuration.items():
            if previous.get(key) != expected:
                raise ValueError(
                    f"resume manifest mismatch for {key}: {previous.get(key)!r} != {expected!r}"
                )
        shard_records = list(previous.get("shards", []))
        completed = sum(int(record["count"]) for record in shard_records)
        if completed > len(records):
            raise ValueError("manifest contains more features than requested")
        for shard in shard_records:
            path = output / shard["file"]
            if not path.is_file() or file_sha256(path) != shard["sha256"]:
                raise RuntimeError(f"existing feature shard failed checksum: {path}")

    if completed == len(records):
        print(json.dumps({"event": "prompt_features_already_complete", "encoder": args.encoder, "count": completed}))
        return

    initial_completed = completed
    start_time = time.perf_counter()
    encoder, feature_dim = build_encoder(args)
    chunk_index = len(shard_records)
    while completed < len(records):
        end = min(len(records), completed + args.chunk_size)
        chunk_records = records[completed:end]
        feature_batches = []
        for batch_start in range(0, len(chunk_records), args.batch_size):
            batch_records = chunk_records[batch_start : batch_start + args.batch_size]
            features = encoder([record["text"] for record in batch_records])
            if features.shape != (len(batch_records), feature_dim):
                raise RuntimeError(
                    f"unexpected {args.encoder} feature shape {tuple(features.shape)}"
                )
            if not torch.isfinite(features).all():
                raise RuntimeError(f"non-finite {args.encoder} features")
            feature_batches.append(features)
        features = torch.cat(feature_batches, dim=0).to(torch.float16).contiguous()
        prompt_ids = torch.tensor(
            [record["prompt_id"] for record in chunk_records],
            dtype=torch.int64,
        )
        filename = f"features-{chunk_index:05d}.safetensors"
        path = output / filename
        temporary = output / f".{filename}.tmp"
        save_file(
            {"features": features, "prompt_ids": prompt_ids},
            temporary,
            metadata={
                "schema": "ardy_prompt_features_v1",
                "encoder": args.encoder,
                "feature_dim": str(feature_dim),
                "start_prompt_id": str(int(prompt_ids[0])),
                "end_prompt_id": str(int(prompt_ids[-1])),
            },
        )
        temporary.replace(path)
        shard_records.append(
            {
                "file": filename,
                "count": len(chunk_records),
                "start_prompt_id": int(prompt_ids[0]),
                "end_prompt_id": int(prompt_ids[-1]),
                "bytes": path.stat().st_size,
                "sha256": file_sha256(path),
            }
        )
        completed = end
        elapsed = time.perf_counter() - start_time
        manifest = {
            **configuration,
            "feature_dim": feature_dim,
            "count": completed,
            "complete": completed == len(records),
            "shards": shard_records,
        }
        atomic_json(manifest_path, manifest)
        print(
            json.dumps(
                {
                    "event": "prompt_feature_progress",
                    "encoder": args.encoder,
                    "count": completed,
                    "planned_count": len(records),
                    "elapsed_s": round(elapsed, 3),
                    "features_per_s": round(
                        (completed - initial_completed) / max(elapsed, 1e-9), 4
                    ),
                }
            ),
            flush=True,
        )
        chunk_index += 1

    print(
        json.dumps(
            {
                "event": "prompt_features_complete",
                "encoder": args.encoder,
                "count": completed,
                "feature_dim": feature_dim,
                "output": str(output),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
