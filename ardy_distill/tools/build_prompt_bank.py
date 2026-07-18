#!/usr/bin/env python
"""Build the deterministic prompt bank used by joint text/control distillation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from ardy_distill.prompt_distribution import (  # noqa: E402
    DEFAULT_PROMPT_BANK_SEED,
    DEFAULT_PROMPT_BANK_SIZE,
    build_prompt_bank,
    prompt_bank_jsonl,
    summarize_prompt_bank,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--size", type=int, default=DEFAULT_PROMPT_BANK_SIZE)
    parser.add_argument("--seed", type=int, default=DEFAULT_PROMPT_BANK_SEED)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    bank_path = args.output_dir / "prompt_bank.jsonl"
    summary_path = args.output_dir / "prompt_bank_summary.json"
    existing = [path for path in (bank_path, summary_path) if path.exists()]
    if existing and not args.force:
        raise FileExistsError(f"refusing to overwrite existing prompt bank files: {existing}")

    records = build_prompt_bank(size=args.size, seed=args.seed)
    payload = prompt_bank_jsonl(records)
    summary = summarize_prompt_bank(records, seed=args.seed)
    bank_path.write_text(payload, encoding="utf-8")
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "event": "prompt_bank_complete",
                "count": len(records),
                "sha256": summary["sha256"],
                "bank": str(bank_path),
                "summary": str(summary_path),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()

