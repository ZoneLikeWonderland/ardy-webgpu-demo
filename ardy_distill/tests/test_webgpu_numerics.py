from __future__ import annotations

import unittest

import torch
from torch import nn

from ardy_distill.webgpu_numerics import (
    CapturePoint,
    EDGE_149_AMPERE_FP16_PROFILES,
    WebGpuNoiseProfile,
    sample_severity,
    simulate_webgpu_fp16,
)


class TinyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.first = nn.Linear(4, 8)
        self.activation = nn.GELU()
        self.second = nn.Linear(8, 2)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.second(self.activation(self.first(value)))


class WebGpuNumericsTest(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(7)
        self.model = TinyModel()
        self.profile = WebGpuNoiseProfile(
            module="encoder",
            source="unit-test",
            module_noise_std={"first": 0.01},
            cumulative_mean_abs_targets={"first": 0.008},
        )
        self.inputs = torch.randn(3, 4)

    def test_forward_preserves_fp32_master_parameters_and_gradients(self) -> None:
        output, _ = simulate_webgpu_fp16(
            self.model,
            self.inputs,
            profile=self.profile,
            severity=torch.ones(3),
        )
        self.assertEqual(output.dtype, torch.float32)
        self.assertTrue(all(parameter.dtype == torch.float32 for parameter in self.model.parameters()))
        output.square().mean().backward()
        for parameter in self.model.parameters():
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())
            self.assertGreater(float(parameter.grad.abs().sum()), 0.0)

    def test_seeded_noise_and_capture_are_reproducible(self) -> None:
        capture = [CapturePoint("first_output", "first")]
        generator_a = torch.Generator().manual_seed(123)
        output_a, captures_a = simulate_webgpu_fp16(
            self.model,
            self.inputs,
            profile=self.profile,
            generator=generator_a,
            capture_points=capture,
        )
        generator_b = torch.Generator().manual_seed(123)
        output_b, captures_b = simulate_webgpu_fp16(
            self.model,
            self.inputs,
            profile=self.profile,
            generator=generator_b,
            capture_points=capture,
        )
        self.assertTrue(torch.equal(output_a, output_b))
        self.assertTrue(torch.equal(captures_a["first_output"], captures_b["first_output"]))

    def test_zero_noise_is_native_fp16_functional_forward(self) -> None:
        output, _ = simulate_webgpu_fp16(
            self.model,
            self.inputs,
            profile=self.profile,
            noise_scale=0.0,
        )
        reference = TinyModel().half()
        reference.load_state_dict(self.model.state_dict())
        expected = reference(self.inputs.half()).float()
        self.assertTrue(torch.equal(output, expected))

    def test_profile_model_mismatch_fails_loudly(self) -> None:
        invalid = WebGpuNoiseProfile(
            module="encoder",
            source="unit-test",
            module_noise_std={"missing": 0.1},
            cumulative_mean_abs_targets={},
        )
        with self.assertRaisesRegex(ValueError, "profile does not match model"):
            simulate_webgpu_fp16(self.model, self.inputs, profile=invalid)

    def test_severity_range_and_locked_profiles(self) -> None:
        severity = sample_severity(128, device="cpu", low=0.5, high=1.5)
        self.assertEqual(severity.shape, (128,))
        self.assertGreaterEqual(float(severity.min()), 0.5)
        self.assertLessEqual(float(severity.max()), 1.5)
        self.assertEqual(set(EDGE_149_AMPERE_FP16_PROFILES), {"encoder", "flow", "decoder"})
        self.assertEqual(len(EDGE_149_AMPERE_FP16_PROFILES["encoder"].module_noise_std), 6)
        self.assertEqual(len(EDGE_149_AMPERE_FP16_PROFILES["flow"].module_noise_std), 32)
        self.assertEqual(len(EDGE_149_AMPERE_FP16_PROFILES["decoder"].module_noise_std), 10)


if __name__ == "__main__":
    unittest.main()

