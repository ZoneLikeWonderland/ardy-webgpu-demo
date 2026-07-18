#!/usr/bin/env python
"""Generate a deterministic PyTorch/ONNX smoke case for browser WebGPU parity."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[1]
CASE_DIR = ROOT / "cases" / "smoke_fp32"


class SmokeModel(nn.Module):
    """ML-like graph: Linear -> GELU -> residual projection -> LayerNorm."""

    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(64, 32)
        self.residual = nn.Linear(64, 32, bias=False)
        self.norm = nn.LayerNorm(32)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = torch.nn.functional.gelu(self.linear(x), approximate="none")
        return self.norm(hidden + 0.125 * self.residual(x))


def metrics(actual: np.ndarray, reference: np.ndarray) -> dict[str, float]:
    actual64 = actual.astype(np.float64)
    reference64 = reference.astype(np.float64)
    abs_error = np.abs(actual64 - reference64)
    rel_error = abs_error / np.maximum(np.abs(reference64), 1.0e-8)
    denom = np.linalg.norm(actual64.ravel()) * np.linalg.norm(reference64.ravel())
    cosine = float(np.dot(actual64.ravel(), reference64.ravel()) / denom)
    return {
        "max_abs_error": float(abs_error.max()),
        "mean_abs_error": float(abs_error.mean()),
        "max_rel_error": float(rel_error.max()),
        "mean_rel_error": float(rel_error.mean()),
        "cosine_similarity": cosine,
    }


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    torch.manual_seed(20260713)
    np.random.seed(20260713)
    torch.set_grad_enabled(False)

    CASE_DIR.mkdir(parents=True, exist_ok=True)
    model = SmokeModel().eval()
    input_tensor = torch.randn(4, 64, dtype=torch.float32)
    reference = model(input_tensor).contiguous()

    model_path = CASE_DIR / "model.onnx"
    torch.onnx.export(
        model,
        (input_tensor,),
        model_path,
        input_names=["input"],
        output_names=["output"],
        opset_version=17,
        do_constant_folding=True,
        dynamo=False,
    )
    onnx.checker.check_model(onnx.load(model_path))

    input_np = input_tensor.cpu().numpy()
    reference_np = reference.cpu().numpy()
    input_np.tofile(CASE_DIR / "input.bin")
    reference_np.tofile(CASE_DIR / "reference.bin")

    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    native_output = session.run(["output"], {"input": input_np})[0]
    native_metrics = metrics(native_output, reference_np)

    op_types = sorted({node.op_type for node in onnx.load(model_path).graph.node})
    manifest = {
        "id": "smoke_fp32",
        "title": "FP32 Linear + GELU + LayerNorm",
        "description": "固定小图，先验证 PyTorch → ONNX → ORT WebGPU → 服务器回传链路。",
        "model_url": "cases/smoke_fp32/model.onnx",
        "model_sha256": sha256(model_path),
        "opset": 17,
        "operators": op_types,
        "inputs": [
            {
                "name": "input",
                "dtype": "float32",
                "shape": list(input_np.shape),
                "url": "cases/smoke_fp32/input.bin",
            }
        ],
        "outputs": [
            {
                "name": "output",
                "dtype": "float32",
                "shape": list(reference_np.shape),
                "reference_url": "cases/smoke_fp32/reference.bin",
                "compare_url": "/api/compare/smoke_fp32/output",
            }
        ],
        "tolerances": {
            "max_abs_error": 2.0e-4,
            "mean_abs_error": 2.0e-5,
            "min_cosine_similarity": 0.999999,
        },
        "native_ort_cpu_vs_pytorch": native_metrics,
    }
    with (CASE_DIR / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    index = {
        "schema_version": 1,
        "cases": [
            {
                "id": manifest["id"],
                "title": manifest["title"],
                "manifest_url": "cases/smoke_fp32/manifest.json",
            }
        ],
    }
    with (ROOT / "cases" / "index.json").open("w", encoding="utf-8") as handle:
        json.dump(index, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    print(f"MODEL={model_path}")
    print(f"MODEL_SHA256={manifest['model_sha256']}")
    print(f"OPERATORS={','.join(op_types)}")
    print(f"NATIVE_MAX_ABS={native_metrics['max_abs_error']:.9g}")
    print(f"NATIVE_COSINE={native_metrics['cosine_similarity']:.12f}")


if __name__ == "__main__":
    main()
