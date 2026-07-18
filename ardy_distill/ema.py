"""Project-local constant-decay FP32 exponential moving average."""

from __future__ import annotations

from contextlib import contextmanager
import math
from typing import Iterator

import torch
from torch import nn


def ema_99_percent_horizon(decay: float) -> float:
    """Return updates required for a constant-decay EMA to replace 99% of history."""

    if not 0.0 < decay < 1.0:
        raise ValueError("EMA decay must be in (0, 1)")
    return math.log(0.01) / math.log(decay)


class ModelEMA:
    def __init__(
        self,
        model: nn.Module,
        decay: float = 0.9995,
        *,
        override_decay_on_load: bool = False,
    ) -> None:
        if not 0.0 < decay < 1.0:
            raise ValueError("EMA decay must be in (0, 1)")
        self.decay = float(decay)
        self.override_decay_on_load = bool(override_decay_on_load)
        self.num_updates = 0
        self.shadow = {
            name: parameter.detach().float().clone()
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        }

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        self.num_updates += 1
        parameters = dict(model.named_parameters())
        for name, average in self.shadow.items():
            current = parameters[name].detach().float()
            average.lerp_(current, 1.0 - self.decay)

    def state_dict(self) -> dict:
        return {
            "decay": self.decay,
            "num_updates": self.num_updates,
            "shadow": self.shadow,
        }

    def load_state_dict(self, state: dict) -> None:
        loaded_decay = float(state["decay"])
        if not 0.0 < loaded_decay < 1.0:
            raise ValueError("Loaded EMA decay must be in (0, 1)")
        if not self.override_decay_on_load:
            self.decay = loaded_decay
        self.num_updates = int(state["num_updates"])
        self.shadow = {
            name: value.to(device=self.shadow[name].device, dtype=torch.float32).clone()
            for name, value in state["shadow"].items()
        }

    @contextmanager
    def apply(self, model: nn.Module) -> Iterator[None]:
        parameters = dict(model.named_parameters())
        backup = {
            name: parameters[name].detach().clone()
            for name in self.shadow
        }
        try:
            with torch.no_grad():
                for name, average in self.shadow.items():
                    parameters[name].copy_(average.to(device=parameters[name].device, dtype=parameters[name].dtype))
            yield
        finally:
            with torch.no_grad():
                for name, value in backup.items():
                    parameters[name].copy_(value)
