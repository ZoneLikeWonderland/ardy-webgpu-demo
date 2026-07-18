"""Differentiable deployment-numerics simulation for ARDY's FP16 WebGPU path.

The released ONNX graphs store weights and activations as FP16.  CPU ORT and
CUDA do not reproduce the reduction order used by ORT Web's WebGPU kernels, so
ordinary ``autocast`` is not a sufficient robustness test.  This module keeps
the production models unchanged and provides a training-only functional
forward with two ingredients:

* parameters, buffers and tensor inputs are cast to native FP16 for the whole
  forward (casts remain differentiable back to the FP32 master parameters);
* zero-mean perturbations are inserted at the residual boundaries where the
  real Edge 149 / NVIDIA Ampere probe measured error growth.

The perturbation standard deviations are derived from the measured cumulative
mean-absolute errors under a zero-mean Gaussian approximation.  They are not
present in exported models.  Training samples their severity so the student is
made robust to a deployment error envelope rather than overfitted to one fixed
golden input or one exact shader reduction order.
"""

from __future__ import annotations

import math
from contextlib import ExitStack
from dataclasses import dataclass
from typing import Any, Literal, Mapping, Sequence

import torch
from torch import nn
from torch.func import functional_call


@dataclass(frozen=True)
class CapturePoint:
    """Capture one module boundary during a simulated forward."""

    label: str
    module_name: str
    location: Literal["pre", "post"] = "post"


@dataclass(frozen=True)
class WebGpuNoiseProfile:
    """Absolute activation-noise profile calibrated from a browser probe."""

    module: Literal["encoder", "flow", "decoder"]
    source: str
    module_noise_std: Mapping[str, float]
    cumulative_mean_abs_targets: Mapping[str, float]


def _incremental_stds(
    rows: Sequence[tuple[str | None, str, float]],
) -> tuple[dict[str, float], dict[str, float]]:
    """Convert cumulative mean-absolute errors to incremental Gaussian stds.

    For a zero-mean Gaussian, ``E|x| = sigma * sqrt(2/pi)``.  Independent
    boundary perturbations add in variance, so each injected standard
    deviation is the positive variance increment needed to reach the next
    measured boundary.  A ``None`` module name advances the measured baseline
    at a functional boundary where there is no convenient ``nn.Module`` hook.
    """

    previous_sigma = 0.0
    stds: dict[str, float] = {}
    targets: dict[str, float] = {}
    for module_name, label, mean_abs in rows:
        if mean_abs < 0.0:
            raise ValueError(f"negative calibration error for {label}: {mean_abs}")
        sigma = float(mean_abs) * math.sqrt(math.pi / 2.0)
        variance_increment = max(0.0, sigma * sigma - previous_sigma * previous_sigma)
        if module_name is not None and variance_increment > 0.0:
            stds[module_name] = math.sqrt(variance_increment)
        targets[label] = float(mean_abs)
        previous_sigma = sigma
    return stds, targets


_PROBE_SOURCE = (
    "distill_runs/first_12h_20260714_014643/eval/"
    "edge_webgpu_precision_probe_blockwise.json"
)


def _make_profiles() -> dict[str, WebGpuNoiseProfile]:
    encoder_stds, encoder_targets = _incremental_stds(
        [
            ("input_proj", "input_gelu", 0.0014444440603256226),
            ("blocks.0", "block_0", 0.00240325927734375),
            ("blocks.1", "block_1", 0.002698145806789398),
            ("blocks.2", "block_2", 0.0029826611280441284),
            ("blocks.3", "block_3", 0.0033093374222517014),
            ("output_norm", "output_norm", 0.00595330074429512),
            ("output_proj", "output", 0.005159974098205566),
        ]
    )
    flow_rows: list[tuple[str | None, str, float]] = [
        ("input_norm", "input_norm", 0.0011364020174369216)
    ]
    trunk_targets = [
        (0.0017975942630852973, 0.0020692140512567547),
        (0.0023967444015267703, 0.0025703665118531455),
        (0.002856114031081753, 0.0030117132992018014),
        (0.0032461312283495708, 0.0035388070142029653),
        (0.0037099181208759546, 0.004122620448470116),
        (0.004144600592553616, 0.004163146018981934),
        (0.004193062428385019, 0.004221081733703613),
        (0.0042676241137087345, 0.004303097724914551),
    ]
    for index, (attention, channel) in enumerate(trunk_targets):
        flow_rows.extend(
            [
                (
                    f"trunk.{index}.attention_out",
                    f"trunk_{index}_attention",
                    attention,
                ),
                (f"trunk.{index}", f"trunk_{index}_channel", channel),
            ]
        )
    # Root projection has a lower-dimensional error, while body_input resumes
    # the full hidden stream.  Advance at body_input rather than incorrectly
    # resetting the hidden-state variance to the root-head value.
    flow_rows.append((None, "body_input", 0.004379549063742161))
    body_targets = [
        (0.0050563812255859375, 0.005135876792328698),
        (0.005777984857559204, 0.005903410259634256),
        (0.006033267825841904, 0.006017695181071758),
        (0.006204135250300169, 0.006242469884455204),
        (0.006441256031394005, 0.006477171089500189),
        (0.006907362025231123, 0.00706041743978858),
        (0.0073048570193350315, 0.00750536797568202),
        (0.007798166126012802, 0.0079111373052001),
    ]
    for index, (attention, channel) in enumerate(body_targets):
        flow_rows.extend(
            [
                (
                    f"body_refiner.{index}.attention_out",
                    f"body_{index}_attention",
                    attention,
                ),
                (
                    f"body_refiner.{index}",
                    f"body_{index}_channel",
                    channel,
                ),
            ]
        )
    flow_rows.extend(
        [
            (None, "body_velocity", 0.0069205681793391705),
            (None, "output", 0.006352597381919622),
        ]
    )
    flow_stds, flow_targets = _incremental_stds(flow_rows)

    decoder_rows: list[tuple[str | None, str, float]] = [
        ("input_proj", "input_gelu_valid", 0.00047195330262184143)
    ]
    decoder_targets = [
        (0.00040933568333275616, 0.0030488281045109034),
        (0.002909096423536539, 0.003984274342656136),
        (0.0033992368262261152, 0.004506103694438934),
        (0.004356850404292345, 0.005920854397118092),
        (0.005330588202923536, 0.005252006463706493),
        (0.004756783600896597, 0.004789693746715784),
        (0.004574596416205168, 0.00466002384185791),
        (0.00460596801713109, 0.004794355481863022),
    ]
    for index, (token, channel) in enumerate(decoder_targets):
        decoder_rows.extend(
            [
                (f"blocks.{index}.token_fc2", f"block_{index}_token", token),
                (f"blocks.{index}", f"block_{index}_channel", channel),
            ]
        )
    decoder_rows.extend(
        [
            ("output_norm", "output_norm", 0.009458223357796669),
            ("output_proj", "output", 0.010377192869782448),
        ]
    )
    decoder_stds, decoder_cumulative_targets = _incremental_stds(decoder_rows)
    return {
        "encoder": WebGpuNoiseProfile(
            module="encoder",
            source=_PROBE_SOURCE,
            module_noise_std=encoder_stds,
            cumulative_mean_abs_targets=encoder_targets,
        ),
        "flow": WebGpuNoiseProfile(
            module="flow",
            source=_PROBE_SOURCE,
            module_noise_std=flow_stds,
            cumulative_mean_abs_targets=flow_targets,
        ),
        "decoder": WebGpuNoiseProfile(
            module="decoder",
            source=_PROBE_SOURCE,
            module_noise_std=decoder_stds,
            cumulative_mean_abs_targets=decoder_cumulative_targets,
        ),
    }


EDGE_149_AMPERE_FP16_PROFILES = _make_profiles()


def _map_tensors(value: Any, function) -> Any:
    if isinstance(value, torch.Tensor):
        return function(value)
    if isinstance(value, tuple):
        return tuple(_map_tensors(item, function) for item in value)
    if isinstance(value, list):
        return [_map_tensors(item, function) for item in value]
    if isinstance(value, dict):
        return {key: _map_tensors(item, function) for key, item in value.items()}
    return value


def _to_fp16(value: torch.Tensor) -> torch.Tensor:
    return value.to(dtype=torch.float16) if value.is_floating_point() else value


def _to_fp32(value: torch.Tensor) -> torch.Tensor:
    return value.float() if value.is_floating_point() else value


def _noise_like(
    value: torch.Tensor,
    *,
    std: float,
    severity: float | torch.Tensor,
    generator: torch.Generator | None,
) -> torch.Tensor:
    if not value.is_floating_point() or std == 0.0:
        return value
    noise = torch.randn(
        value.shape,
        dtype=torch.float32,
        device=value.device,
        generator=generator,
    )
    if isinstance(severity, torch.Tensor):
        scale = severity.to(device=value.device, dtype=torch.float32)
        if scale.ndim == 1 and value.ndim > 1 and scale.shape[0] == value.shape[0]:
            scale = scale.reshape(scale.shape[0], *([1] * (value.ndim - 1)))
    else:
        scale = float(severity)
    # The returned cast is part of the computation graph.  Gradients therefore
    # reach FP32 master parameters through the FP16 functional forward, while
    # the sampled perturbation itself is deliberately non-differentiable.
    return (value.float() + noise * (float(std) * scale)).to(dtype=torch.float16)


def simulate_webgpu_fp16(
    module: nn.Module,
    *args: Any,
    profile: WebGpuNoiseProfile,
    severity: float | torch.Tensor = 1.0,
    noise_scale: float = 1.0,
    generator: torch.Generator | None = None,
    capture_points: Sequence[CapturePoint] = (),
    **kwargs: Any,
) -> tuple[Any, dict[str, torch.Tensor]]:
    """Run ``module`` through a differentiable FP16 + calibrated-noise path.

    The original module is never converted in place.  ``functional_call`` uses
    FP16 views of its current parameters and buffers, so the clean and noisy
    branches can share one FP32 master model and one EMA.  Outputs are returned
    as FP32 for stable loss evaluation.  Captures are detached FP32 tensors and
    are intended only for calibration/audit tooling.
    """

    if noise_scale < 0.0:
        raise ValueError("noise_scale must be non-negative")
    if profile.module not in {"encoder", "flow", "decoder"}:
        raise ValueError(f"unknown WebGPU profile module: {profile.module}")

    named_modules = dict(module.named_modules())
    unknown_noise_modules = sorted(set(profile.module_noise_std) - set(named_modules))
    if unknown_noise_modules:
        raise ValueError(
            f"{profile.module} profile does not match model; missing modules: "
            f"{unknown_noise_modules}"
        )
    unknown_capture_modules = sorted(
        {point.module_name for point in capture_points} - set(named_modules)
    )
    if unknown_capture_modules:
        raise ValueError(f"capture modules not found: {unknown_capture_modules}")

    state: dict[str, torch.Tensor] = {}
    for name, parameter in module.named_parameters():
        state[name] = _to_fp16(parameter)
    for name, buffer in module.named_buffers():
        state[name] = _to_fp16(buffer)
    fp16_args = _map_tensors(args, _to_fp16)
    fp16_kwargs = _map_tensors(kwargs, _to_fp16)
    captures: dict[str, torch.Tensor] = {}

    with ExitStack() as stack:
        for module_name, std in profile.module_noise_std.items():
            target = named_modules[module_name]

            def inject_noise(_module, _inputs, output, *, boundary_std=std):
                return _map_tensors(
                    output,
                    lambda tensor: _noise_like(
                        tensor,
                        std=boundary_std * noise_scale,
                        severity=severity,
                        generator=generator,
                    ),
                )

            stack.callback(target.register_forward_hook(inject_noise).remove)

        # Capture hooks are registered after perturbation hooks.  Post captures
        # therefore see the actual value passed downstream by the simulator.
        for point in capture_points:
            target = named_modules[point.module_name]
            if point.location == "post":

                def capture_post(_module, _inputs, output, *, label=point.label):
                    tensors: list[torch.Tensor] = []
                    _map_tensors(
                        output,
                        lambda tensor: tensors.append(tensor) or tensor,
                    )
                    if len(tensors) != 1:
                        raise ValueError(
                            f"capture {label} expected one tensor, got {len(tensors)}"
                        )
                    captures[label] = tensors[0].detach().float().clone()

                stack.callback(target.register_forward_hook(capture_post).remove)
            else:

                def capture_pre(_module, inputs, *, label=point.label):
                    tensors: list[torch.Tensor] = []
                    _map_tensors(
                        inputs,
                        lambda tensor: tensors.append(tensor) or tensor,
                    )
                    if len(tensors) < 1:
                        raise ValueError(f"capture {label} received no tensor input")
                    captures[label] = tensors[0].detach().float().clone()

                stack.callback(target.register_forward_pre_hook(capture_pre).remove)

        # Disable any surrounding autocast: this branch deliberately executes
        # native FP16 instead of inheriting the BF16 training policy.
        device_type = next(module.parameters()).device.type
        with torch.autocast(device_type=device_type, enabled=False):
            output = functional_call(
                module,
                state,
                fp16_args,
                fp16_kwargs,
                strict=True,
            )
    return _map_tensors(output, _to_fp32), captures


def sample_severity(
    batch_size: int,
    *,
    device: torch.device | str,
    low: float = 0.5,
    high: float = 1.5,
) -> torch.Tensor:
    """Sample one deployment-noise severity per batch item."""

    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if not 0.0 <= low <= high:
        raise ValueError("severity range must satisfy 0 <= low <= high")
    return torch.empty(batch_size, device=device, dtype=torch.float32).uniform_(low, high)

