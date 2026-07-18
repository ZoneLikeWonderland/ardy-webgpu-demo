"""Shared logging, motion-representation loading, and weight I/O."""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf
from safetensors.torch import load_file, save_file
from torch.utils.tensorboard import SummaryWriter


def load_motion_rep(checkpoint_dir: str | Path):
    checkpoint_dir = Path(checkpoint_dir).resolve()
    config = OmegaConf.merge(
        OmegaConf.load(checkpoint_dir / "config.yaml"),
        OmegaConf.create({"checkpoint_dir": str(checkpoint_dir)}),
    )
    return instantiate(config.autoencoder.motion_rep)


def cosine_warmup_lambda(step: int, warmup_steps: int, total_steps: int, min_ratio: float = 0.05) -> float:
    if step < warmup_steps:
        return max(1.0e-8, step / max(1, warmup_steps))
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    progress = min(max(progress, 0.0), 1.0)
    return min_ratio + (1.0 - min_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))


class MetricLogger:
    def __init__(self, output_dir: str | Path, enabled: bool) -> None:
        self.enabled = enabled
        self.output_dir = Path(output_dir)
        self.writer = None
        self.jsonl = None
        self.start = time.perf_counter()
        if enabled:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            self.writer = SummaryWriter(self.output_dir / "tensorboard")
            self.jsonl = (self.output_dir / "metrics.jsonl").open("a", encoding="utf-8")

    def log(self, step: int, metrics: dict[str, float], **metadata) -> None:
        if not self.enabled:
            return
        elapsed = time.perf_counter() - self.start
        record = {"step": step, "elapsed_s": elapsed, **metadata, **metrics}
        assert self.jsonl is not None and self.writer is not None
        self.jsonl.write(json.dumps(record, ensure_ascii=False) + "\n")
        self.jsonl.flush()
        for name, value in metrics.items():
            if isinstance(value, (float, int)) and math.isfinite(float(value)):
                self.writer.add_scalar(name, float(value), step)

    def close(self) -> None:
        if self.writer is not None:
            self.writer.close()
        if self.jsonl is not None:
            self.jsonl.close()


def tensor_metrics_to_float(metrics: dict[str, torch.Tensor], accelerator) -> dict[str, float]:
    result = {}
    for name, value in metrics.items():
        gathered = accelerator.gather_for_metrics(value.detach().float().reshape(1))
        result[name] = float(gathered.mean().item())
    return result


def save_safetensor_weights(module: torch.nn.Module, path: str | Path, state_dict: dict | None = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    source = module.state_dict() if state_dict is None else state_dict
    tensors = {
        name: value.detach().cpu().contiguous()
        for name, value in source.items()
        if isinstance(value, torch.Tensor)
    }
    save_file(tensors, path)


def load_safetensor_weights(module: torch.nn.Module, path: str | Path, strict: bool = True):
    state = load_file(path, device="cpu")
    return module.load_state_dict(state, strict=strict)
