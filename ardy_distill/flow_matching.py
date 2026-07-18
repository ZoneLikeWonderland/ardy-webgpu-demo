"""Rectified-flow interpolation and the high-noise-biased time sampler."""

from __future__ import annotations

import torch
from torch import nn


def sample_flow_time(
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
    exact_t1_probability: float = 0.50,
    high_noise_probability: float = 0.35,
) -> torch.Tensor:
    if not 0.0 <= exact_t1_probability <= 1.0:
        raise ValueError("exact_t1_probability must be in [0, 1]")
    if not 0.0 <= high_noise_probability <= 1.0:
        raise ValueError("high_noise_probability must be in [0, 1]")
    if exact_t1_probability + high_noise_probability > 1.0:
        raise ValueError("time mixture probabilities exceed one")
    selector = torch.rand(batch_size, 1, device=device)
    uniform = torch.rand(batch_size, 1, device=device)
    high_noise = 1.0 - uniform.pow(3)
    result = uniform
    result = torch.where(
        selector < exact_t1_probability + high_noise_probability,
        high_noise,
        result,
    )
    result = torch.where(
        selector < exact_t1_probability,
        torch.ones_like(result),
        result,
    )
    return result.to(dtype=dtype)


def make_flow_pair(clean: torch.Tensor, noise: torch.Tensor, time: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    time_3d = time.reshape(clean.shape[0], 1, 1).to(dtype=clean.dtype)
    noisy = (1.0 - time_3d) * clean + time_3d * noise
    velocity = noise - clean
    return noisy, velocity


def flow_velocity_to_x0(
    noisy: torch.Tensor,
    velocity: torch.Tensor,
    time: torch.Tensor,
) -> torch.Tensor:
    """Recover ``x0`` from the rectified-flow velocity parameterization."""

    time_view = time.reshape(noisy.shape[0], *([1] * (noisy.ndim - 1))).to(
        device=noisy.device,
        dtype=noisy.dtype,
    )
    return noisy - time_view * velocity


def euler_flow_denoise(
    model: nn.Module,
    noise: torch.Tensor,
    history_hybrid: torch.Tensor,
    path_condition: torch.Tensor,
    first_heading: torch.Tensor,
    has_history: torch.Tensor,
    *,
    steps: int,
    text_feature: torch.Tensor | None = None,
    heading_condition: torch.Tensor | None = None,
) -> torch.Tensor:
    """Integrate the rectified-flow ODE from t=1 to t=0 with uniform Euler steps.

    The training path is ``x_t=(1-t)*x0+t*noise`` and the network predicts
    ``dx/dt=noise-x0``.  Generation therefore moves backward in time using
    ``x_{t-dt}=x_t-dt*v_theta(x_t,t)``.  Calling the passed module for every
    stage keeps DDP forward hooks active during end-to-end solver training.
    """

    if steps < 1:
        raise ValueError("flow solver steps must be positive")
    state = noise
    step_size = 1.0 / steps
    for index in range(steps, 0, -1):
        flow_time = torch.full_like(has_history, index / steps)
        if text_feature is None and heading_condition is None:
            # Keep compatibility with small path-only/test modules whose
            # forward signature predates the optional condition arguments.
            velocity = model(
                state,
                history_hybrid,
                path_condition,
                first_heading,
                has_history,
                flow_time,
            )
        else:
            velocity = model(
                state,
                history_hybrid,
                path_condition,
                first_heading,
                has_history,
                flow_time,
                text_feature,
                heading_condition,
            )
        state = state - step_size * velocity
    return state
