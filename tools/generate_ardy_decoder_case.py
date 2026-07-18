#!/usr/bin/env python
"""Export a fixed-shape ARDY Core decoder and deterministic PyTorch references."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch


TOY_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = TOY_ROOT.parent
REPO = WORKSPACE / "ardy"
CHECKPOINTS = REPO / "checkpoints"
CASE_DIR = TOY_ROOT / "cases" / "ardy_decoder_fp32"

if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from ardy.model.load_model import load_model  # noqa: E402
from scripts.export_onnx import _onnx_export_mode, make_decoder_dummy_inputs  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def update_case_index(case_entry: dict) -> None:
    index_path = TOY_ROOT / "cases" / "index.json"
    if index_path.exists():
        index = json.loads(index_path.read_text(encoding="utf-8"))
    else:
        index = {"schema_version": 1, "cases": []}
    index["cases"] = [item for item in index["cases"] if item["id"] != case_entry["id"]]
    index["cases"].append(case_entry)
    index_path.write_text(json.dumps(index, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    torch.manual_seed(20260713)
    np.random.seed(20260713)
    torch.set_grad_enabled(False)
    CASE_DIR.mkdir(parents=True, exist_ok=True)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this export path")
    device = "cuda:0"
    print("Loading ARDY Core with text_encoder=False...", flush=True)
    model = load_model(
        "core",
        device=device,
        text_encoder=False,
        checkpoints_dir=str(CHECKPOINTS),
    )
    if model.text_encoder is not None:
        raise RuntimeError("text encoder unexpectedly loaded")
    autoencoder = model.autoencoder.eval()

    num_tokens = 10
    dummy = make_decoder_dummy_inputs(autoencoder, num_tokens=num_tokens, device=device)
    # Real FSQ tokens are on-grid. Snapping before both PyTorch and ONNX runs
    # avoids an artificial precision boundary around Round().
    dummy["latent_tokens"] = autoencoder.requantize(dummy["latent_tokens"]).contiguous()
    dummy["external_cond"] = dummy["external_cond"].contiguous()
    dummy["motion_pad_mask"] = dummy["motion_pad_mask"].contiguous()

    with torch.no_grad():
        reference_dict = autoencoder(
            dummy["latent_tokens"],
            dummy["external_cond"],
            dummy["motion_pad_mask"] > 0.5,
        )
    output_names = list(reference_dict)

    model_path = CASE_DIR / "model.onnx"
    print(f"Exporting fixed decoder to {model_path}...", flush=True)
    with _onnx_export_mode(), torch.no_grad():
        torch.onnx.export(
            autoencoder,
            (
                dummy["latent_tokens"],
                dummy["external_cond"],
                dummy["motion_pad_mask"],
            ),
            model_path,
            input_names=["latent_tokens", "external_cond", "motion_pad_mask"],
            output_names=output_names,
            opset_version=17,
            do_constant_folding=True,
            dynamo=False,
        )

    onnx_model = onnx.load(model_path)
    onnx.checker.check_model(onnx_model)
    graph_outputs = [item.name for item in onnx_model.graph.output]
    if graph_outputs != output_names:
        raise RuntimeError(f"unexpected graph outputs: {graph_outputs}, expected {output_names}")

    inputs = []
    for name in ("latent_tokens", "external_cond", "motion_pad_mask"):
        array = dummy[name].detach().float().cpu().numpy()
        filename = f"input_{name}.bin"
        array.tofile(CASE_DIR / filename)
        inputs.append(
            {
                "name": name,
                "dtype": "float32",
                "shape": list(array.shape),
                "url": f"cases/ardy_decoder_fp32/{filename}",
            }
        )

    outputs = []
    reference_arrays: dict[str, np.ndarray] = {}
    for name, tensor in reference_dict.items():
        array = tensor.detach().float().contiguous().cpu().numpy()
        reference_arrays[name] = array
        filename = f"reference_{name}.bin"
        array.tofile(CASE_DIR / filename)
        outputs.append(
            {
                "name": name,
                "dtype": "float32",
                "shape": list(array.shape),
                "reference_url": f"cases/ardy_decoder_fp32/{filename}",
                "compare_url": f"/api/compare/ardy_decoder_fp32/{name}",
            }
        )

    del model
    torch.cuda.empty_cache()

    print("Running native ORT CPU cross-check...", flush=True)
    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    cpu_feeds = {
        item["name"]: np.fromfile(CASE_DIR / f"input_{item['name']}.bin", dtype=np.float32).reshape(item["shape"])
        for item in inputs
    }
    native_outputs = session.run(output_names, cpu_feeds)
    native_metrics = {
        name: metrics(actual, reference_arrays[name])
        for name, actual in zip(output_names, native_outputs)
    }
    output_tolerances = {
        "root": {
            "max_abs_error": 1.5e-2,
            "mean_abs_error": 3.0e-3,
            "min_cosine_similarity": 0.9999,
        },
        "body": {
            "max_abs_error": 4.0e-2,
            "mean_abs_error": 3.0e-3,
            "min_cosine_similarity": 0.9999,
        },
    }
    for output in outputs:
        output["tolerances"] = output_tolerances[output["name"]]

    op_types = sorted({node.op_type for node in onnx_model.graph.node})
    manifest = {
        "id": "ardy_decoder_fp32",
        "title": "ARDY Core FSQ decoder FP32 / 10 tokens",
        "description": "无 Llama；固定 10 tokens/40 frames；输入 latent 已对齐 FSQ grid。",
        "model_url": "cases/ardy_decoder_fp32/model.onnx",
        "model_sha256": sha256(model_path),
        "model_size_bytes": model_path.stat().st_size,
        "opset": 17,
        "operators": op_types,
        "inputs": inputs,
        "outputs": outputs,
        "tolerances": output_tolerances,
        "native_ort_cpu_vs_pytorch": native_metrics,
        "benchmark": {"warmup_runs": 3, "timed_runs": 10},
        "text_encoder_loaded": False,
    }
    (CASE_DIR / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    update_case_index(
        {
            "id": manifest["id"],
            "title": manifest["title"],
            "manifest_url": "cases/ardy_decoder_fp32/manifest.json",
        }
    )

    print(f"MODEL_SIZE={manifest['model_size_bytes']}")
    print(f"MODEL_SHA256={manifest['model_sha256']}")
    print(f"OPERATORS={','.join(op_types)}")
    for name, values in native_metrics.items():
        print(f"NATIVE_{name}_MAX_ABS={values['max_abs_error']:.9g}")
        print(f"NATIVE_{name}_COSINE={values['cosine_similarity']:.12f}")


if __name__ == "__main__":
    main()
