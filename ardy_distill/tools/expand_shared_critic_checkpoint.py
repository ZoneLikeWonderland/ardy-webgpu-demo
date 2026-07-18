"""Expand one shared critic checkpoint into cloned independent LADD heads.

Generator, fake-score, EMA, sampler, counters, schedulers and every rank's RNG
state are hard-linked when possible.  Only the critic model and Adam state are
materialized: each requested tap receives an exact copy of the shared head and
its optimizer moments.  This preserves all learned state while changing the
intended discriminator parameter sharing.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import shutil
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

from ardy_distill.models import clone_shared_critic_state_for_taps


def expand_single_group_optimizer_state(
    optimizer_state: dict,
    copies: int,
) -> dict:
    """Duplicate one Adam parameter group and all moments in parameter order."""

    if copies < 1:
        raise ValueError("optimizer state copies must be positive")
    groups = optimizer_state.get("param_groups")
    states = optimizer_state.get("state")
    if not isinstance(groups, list) or len(groups) != 1 or not isinstance(states, dict):
        raise ValueError("expected one optimizer parameter group")
    old_ids = list(groups[0].get("params", ()))
    if not old_ids or any(parameter_id not in states for parameter_id in old_ids):
        raise ValueError("optimizer parameter group/state mapping is incomplete")

    expanded_states = {}
    expanded_ids = []
    for _copy_index in range(copies):
        for old_id in old_ids:
            new_id = len(expanded_ids)
            expanded_ids.append(new_id)
            expanded_states[new_id] = copy.deepcopy(states[old_id])

    expanded_group = copy.deepcopy(groups[0])
    expanded_group["params"] = expanded_ids
    return {"state": expanded_states, "param_groups": [expanded_group]}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def link_or_copy(source: Path, destination: Path) -> None:
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def expand_checkpoint(source: Path, output: Path, taps: tuple[str, ...]) -> None:
    if not source.is_dir():
        raise ValueError(f"checkpoint source does not exist: {source}")
    if not taps or len(set(taps)) != len(taps):
        raise ValueError("--taps must be non-empty and unique")
    critic_model = source / "model_2.safetensors"
    critic_optimizer = source / "optimizer_2.bin"
    if not critic_model.is_file() or not critic_optimizer.is_file():
        raise ValueError("source lacks model_2.safetensors or optimizer_2.bin")
    if output.exists():
        raise ValueError(f"refusing to overwrite expanded checkpoint: {output}")

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp-{os.getpid()}")
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir()
    try:
        for path in source.iterdir():
            if not path.is_file() or path.name in {
                critic_model.name,
                critic_optimizer.name,
            }:
                continue
            link_or_copy(path, temporary / path.name)

        shared_state = load_file(critic_model, device="cpu")
        expanded_model = clone_shared_critic_state_for_taps(shared_state, taps)
        with safe_open(critic_model, framework="pt", device="cpu") as handle:
            metadata = handle.metadata()
        save_file(
            expanded_model,
            temporary / critic_model.name,
            metadata=metadata,
        )

        old_optimizer = torch.load(
            critic_optimizer,
            map_location="cpu",
            weights_only=False,
        )
        expanded_optimizer = expand_single_group_optimizer_state(
            old_optimizer,
            copies=len(taps),
        )
        torch.save(expanded_optimizer, temporary / critic_optimizer.name)

        manifest = {
            "schema": "ardy_independent_ladd_checkpoint_v1",
            "source": str(source),
            "source_critic_model_sha256": sha256(critic_model),
            "source_critic_optimizer_sha256": sha256(critic_optimizer),
            "taps": list(taps),
            "shared_parameter_tensors": len(shared_state),
            "independent_parameter_tensors": len(expanded_model),
            "optimizer_parameter_states": len(expanded_optimizer["state"]),
            "passthrough": "hardlink_when_possible",
        }
        (temporary / "independent_critic_expansion.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary.rename(output)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--taps", nargs="+", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    expand_checkpoint(args.source, args.output, tuple(args.taps))
    print(f"expanded shared critic checkpoint -> {args.output}")


if __name__ == "__main__":
    main()
