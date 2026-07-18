#!/usr/bin/env python
"""Bundle a small, exact FP16 Qwen prompt table for the browser pilot.

The motion model still receives the full 7680-D feature.  This utility only
selects representative rows from the complete offline table; it performs no
PCA, low-rank projection, quantization beyond the existing FP16 cache, or text
encoding in the browser.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file


TOY_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = TOY_ROOT.parent
DEMO_ROOT = TOY_ROOT / "infinite_demo"
DEFAULT_FEATURE_ROOT = WORKSPACE / "distill_data/text_control_v1/features/qwen"
DEFAULT_PROMPT_BANK = WORKSPACE / "distill_data/text_control_v1/prompt_bank.jsonl"

# Official ARDY presets plus representatives from every generated motion
# family.  Keeping the list explicit makes frontend comparisons reproducible.
DEFAULT_PROMPT_IDS = (
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
    1186, 1187,
    1773, 2235,
    2738, 2739,
    3241, 3242,
    3577, 3578,
    4206,
    4709, 4710,
    5045, 5674, 6848,
    7268, 7269,
    7604, 7605,
    7940, 7941,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-root", type=Path, default=DEFAULT_FEATURE_ROOT)
    parser.add_argument("--prompt-bank", type=Path, default=DEFAULT_PROMPT_BANK)
    parser.add_argument("--output-json", type=Path, default=DEMO_ROOT / "prompt_features.json")
    parser.add_argument(
        "--output-bin",
        type=Path,
        default=DEMO_ROOT / "prompt_features.fp16.bin",
    )
    parser.add_argument("--prompt-ids", type=int, nargs="*", default=list(DEFAULT_PROMPT_IDS))
    parser.add_argument("--default-prompt-id", type=int, default=1)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    feature_manifest = json.loads(
        (args.feature_root / "manifest.json").read_text(encoding="utf-8")
    )
    if feature_manifest.get("schema") != "ardy_prompt_features_v1":
        raise ValueError("unexpected source feature schema")
    if not feature_manifest.get("complete") or feature_manifest.get("storage_dtype") != "float16":
        raise ValueError("source Qwen feature table must be complete FP16")

    prompt_rows = {
        int(row["prompt_id"]): row
        for row in (
            json.loads(line)
            for line in args.prompt_bank.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    }
    prompt_ids = list(dict.fromkeys(int(value) for value in args.prompt_ids))
    if not prompt_ids:
        raise ValueError("at least one prompt id is required")
    if args.default_prompt_id not in prompt_ids:
        raise ValueError("default prompt id must be included in the bundle")

    shard_records = feature_manifest["shards"]
    loaded_shards: dict[str, dict[str, torch.Tensor]] = {}
    rows: list[torch.Tensor] = []
    entries: list[dict] = []
    for prompt_id in prompt_ids:
        if prompt_id not in prompt_rows:
            raise KeyError(f"prompt id {prompt_id} missing from prompt bank")
        record = next(
            (
                item
                for item in shard_records
                if int(item["start_prompt_id"]) <= prompt_id <= int(item["end_prompt_id"])
            ),
            None,
        )
        if record is None:
            raise KeyError(f"prompt id {prompt_id} missing from feature shards")
        filename = str(record["file"])
        tensors = loaded_shards.get(filename)
        if tensors is None:
            tensors = load_file(args.feature_root / filename, device="cpu")
            loaded_shards[filename] = tensors
        offset = prompt_id - int(record["start_prompt_id"])
        if int(tensors["prompt_ids"][offset]) != prompt_id:
            raise ValueError(f"feature row mismatch for prompt id {prompt_id}")
        feature = tensors["features"][offset]
        if feature.dtype != torch.float16 or not torch.isfinite(feature).all():
            raise ValueError(f"invalid FP16 feature for prompt id {prompt_id}")
        rows.append(feature.contiguous())
        source = prompt_rows[prompt_id]
        entries.append(
            {
                "index": len(entries),
                "prompt_id": prompt_id,
                "text": source["text"],
                "family": source["family"],
                "group": source["group"],
                "source": source["source"],
            }
        )

    features = torch.stack(rows).contiguous()
    expected_shape = (len(prompt_ids), int(feature_manifest["feature_dim"]))
    if tuple(features.shape) != expected_shape:
        raise ValueError(f"bundle shape {tuple(features.shape)} != {expected_shape}")

    args.output_bin.parent.mkdir(parents=True, exist_ok=True)
    # NumPy's explicit little-endian dtype fixes the wire format consumed by
    # Uint16Array in all supported browsers.
    features.numpy().astype(np.dtype("<f2"), copy=False).tofile(args.output_bin)
    expected_bytes = features.numel() * 2
    if args.output_bin.stat().st_size != expected_bytes:
        raise RuntimeError("prompt feature binary size mismatch")

    metadata = {
        "schema": "ardy_browser_prompt_features_v1",
        "encoder": "FLUX.2_Qwen3_precomputed",
        "storage_dtype": "float16_le",
        "feature_dim": expected_shape[1],
        "count": expected_shape[0],
        "compression": "none",
        "default_prompt_id": args.default_prompt_id,
        "feature_url": f"infinite_demo/{args.output_bin.name}",
        "size_bytes": expected_bytes,
        "sha256": sha256(args.output_bin),
        "entries": entries,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "metadata": str(args.output_json),
                "features": str(args.output_bin),
                "shape": list(features.shape),
                "size_bytes": expected_bytes,
                "sha256": metadata["sha256"],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
