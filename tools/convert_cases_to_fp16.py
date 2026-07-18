#!/usr/bin/env python
"""Directly export ARDY Core half models and FP16 PyTorch references."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import onnx
import torch


TOY_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = TOY_ROOT.parent
REPO = WORKSPACE / "ardy"
CHECKPOINTS = REPO / "checkpoints"
CASES_ROOT = TOY_ROOT / "cases"

if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from ardy.model.load_model import load_model  # noqa: E402
from scripts.export_onnx import _onnx_export_mode  # noqa: E402


NP_DTYPES = {
    "float32": np.dtype("<f4"),
    "float16": np.dtype("<f2"),
    "int64": np.dtype("<i8"),
    "int32": np.dtype("<i4"),
    "bool": np.dtype("u1"),
}


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
        "max_abs_error": float(abs_error.max(initial=0.0)),
        "mean_abs_error": float(abs_error.mean()) if abs_error.size else 0.0,
        "max_rel_error": float(rel_error.max(initial=0.0)),
        "mean_rel_error": float(rel_error.mean()) if rel_error.size else 0.0,
        "cosine_similarity": cosine,
    }


def load_array(spec: dict) -> np.ndarray:
    return np.fromfile(TOY_ROOT / spec["url"], dtype=NP_DTYPES[spec["dtype"]]).reshape(spec["shape"])


def half_tensor(spec: dict, device: str) -> torch.Tensor:
    tensor = torch.from_numpy(load_array(spec)).to(device)
    return tensor.half() if tensor.is_floating_point() else tensor


def save_input(target_id: str, name: str, tensor: torch.Tensor) -> dict:
    array = tensor.detach().contiguous().cpu().numpy()
    if tensor.dtype == torch.float16:
        dtype_name = "float16"
    elif tensor.dtype == torch.int64:
        dtype_name = "int64"
    elif tensor.dtype == torch.int32:
        dtype_name = "int32"
    elif tensor.dtype == torch.bool:
        dtype_name = "bool"
    else:
        raise RuntimeError(f"unsupported graph input dtype: {name}={tensor.dtype}")
    filename = f"input_{name}.bin"
    array.tofile(CASES_ROOT / target_id / filename)
    return {
        "name": name,
        "dtype": dtype_name,
        "shape": list(array.shape),
        "url": f"cases/{target_id}/{filename}",
    }


def secondary_reference(target_id: str, name: str, shape: list[int], tolerances: dict) -> dict:
    return {
        "label": "pytorch_fp16",
        "dtype": "float16",
        "url": f"cases/{target_id}/reference_{name}_pytorch_fp16.bin",
        "shape": shape,
        "tolerances": tolerances,
    }


def finalize_manifest(
    *, target_id: str, source_manifest: dict, title: str, description: str,
    model_path: Path, onnx_model, inputs: list[dict], outputs: list[dict],
    local_metrics: dict, benchmark_runs: int,
) -> dict:
    manifest = {
        "id": target_id,
        "title": title,
        "description": description,
        "source_case": source_manifest["id"],
        "model_url": f"cases/{target_id}/model.onnx",
        "model_sha256": sha256(model_path),
        "model_size_bytes": model_path.stat().st_size,
        "source_model_size_bytes": source_manifest["model_size_bytes"],
        "size_ratio_vs_fp32": model_path.stat().st_size / source_manifest["model_size_bytes"],
        "opset": 17,
        "operators": sorted({node.op_type for node in onnx_model.graph.node}),
        "inputs": inputs,
        "outputs": outputs,
        "tolerances": {item["name"]: item["tolerances"] for item in outputs},
        "pytorch_fp16_vs_pytorch_fp32": local_metrics,
        "benchmark": {"warmup_runs": 3, "timed_runs": benchmark_runs},
        "precision": "float16",
        "reference_precision": "float32",
        "conversion_method": "direct torch.nn.Module.half() export",
        "text_encoder_loaded": False,
    }
    (CASES_ROOT / target_id / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"{target_id}: {manifest['model_size_bytes']} bytes "
        f"({manifest['size_ratio_vs_fp32']:.3%} of FP32), SHA256={manifest['model_sha256']}"
    )
    for name, values in local_metrics.items():
        print(
            f"  PyTorch FP16 vs FP32 {name}: max_abs={values['max_abs_error']:.9g}, "
            f"mean_abs={values['mean_abs_error']:.9g}, cosine={values['cosine_similarity']:.12f}"
        )
    return {"id": target_id, "title": title, "manifest_url": f"cases/{target_id}/manifest.json"}


def export_decoder(model, device: str) -> dict:
    source_id, target_id = "ardy_decoder_fp32", "ardy_decoder_fp16"
    source_manifest = json.loads((CASES_ROOT / source_id / "manifest.json").read_text(encoding="utf-8"))
    target_dir = CASES_ROOT / target_id
    target_dir.mkdir(parents=True, exist_ok=True)
    source_inputs = {item["name"]: item for item in source_manifest["inputs"]}
    tensors = {name: half_tensor(spec, device) for name, spec in source_inputs.items()}
    args = (tensors["latent_tokens"], tensors["external_cond"], tensors["motion_pad_mask"])
    with torch.no_grad():
        reference_dict = model.autoencoder(*args)
    output_names = list(reference_dict)

    model_path = target_dir / "model.onnx"
    print(f"Direct-exporting {target_id}...", flush=True)
    with _onnx_export_mode(), torch.no_grad():
        torch.onnx.export(
            model.autoencoder, args, model_path,
            input_names=["latent_tokens", "external_cond", "motion_pad_mask"],
            output_names=output_names, opset_version=17, do_constant_folding=True, dynamo=False,
        )
    onnx_model = onnx.load(model_path)
    onnx.checker.check_model(onnx_model)
    graph_inputs = [item.name for item in onnx_model.graph.input]
    inputs = [save_input(target_id, name, tensors[name]) for name in graph_inputs]

    primary_tolerances = {
        "root": {"max_abs_error": 0.06, "mean_abs_error": 0.006, "min_cosine_similarity": 0.998},
        "body": {"max_abs_error": 0.12, "mean_abs_error": 0.006, "min_cosine_similarity": 0.998},
    }
    fp16_tolerances = {
        "root": {"max_abs_error": 0.03, "mean_abs_error": 0.004, "min_cosine_similarity": 0.999},
        "body": {"max_abs_error": 0.06, "mean_abs_error": 0.004, "min_cosine_similarity": 0.999},
    }
    outputs, local_metrics = [], {}
    source_outputs = {item["name"]: item for item in source_manifest["outputs"]}
    for name, tensor in reference_dict.items():
        half_array = tensor.detach().contiguous().cpu().numpy()
        half_array.tofile(target_dir / f"reference_{name}_pytorch_fp16.bin")
        source_spec = source_outputs[name]
        fp32_reference = np.fromfile(
            TOY_ROOT / source_spec["reference_url"], dtype=np.float32
        ).reshape(source_spec["shape"])
        local_metrics[name] = metrics(half_array, fp32_reference)
        outputs.append(
            {
                "name": name, "dtype": "float16", "shape": list(half_array.shape),
                "reference_dtype": "float32", "reference_url": source_spec["reference_url"],
                "compare_url": f"/api/compare/{target_id}/{name}",
                "tolerances": primary_tolerances[name],
                "secondary_references": [
                    secondary_reference(target_id, name, list(half_array.shape), fp16_tolerances[name])
                ],
            }
        )
    return finalize_manifest(
        target_id=target_id, source_manifest=source_manifest,
        title="ARDY Core FSQ decoder FP16 / 10 tokens",
        description="无 Llama；固定 10 tokens/40 frames；直接从 PyTorch half 模型导出。",
        model_path=model_path, onnx_model=onnx_model, inputs=inputs, outputs=outputs,
        local_metrics=local_metrics, benchmark_runs=10,
    )


def export_denoiser(model, device: str) -> dict:
    source_id, target_id = "ardy_denoiser_fp32", "ardy_denoiser_fp16"
    source_manifest = json.loads((CASES_ROOT / source_id / "manifest.json").read_text(encoding="utf-8"))
    target_dir = CASES_ROOT / target_id
    target_dir.mkdir(parents=True, exist_ok=True)
    source_specs = {item["name"]: item for item in source_manifest["inputs"]}
    tensors = {name: half_tensor(spec, device) for name, spec in source_specs.items()}
    tensors["future_len"] = torch.zeros(1, device=device, dtype=torch.int64)
    tensors["future_mask"] = torch.zeros(1, 40, device=device, dtype=torch.float16)
    tensors["text_feat_pad_mask"] = torch.zeros(1, 1, device=device, dtype=torch.float16)
    input_order = [
        "cfg_weight_text", "cfg_weight_cstr", "x", "history_len", "generation_len", "future_len",
        "history_mask", "generation_mask", "future_mask", "history_token_mask",
        "generation_token_mask", "future_token_mask", "text_feat", "text_feat_pad_mask",
        "timesteps", "first_heading_angle", "motion_mask", "observed_motion",
    ]
    args = tuple(tensors[name].contiguous() for name in input_order)
    with torch.no_grad():
        reference = model.denoiser(*args).detach().contiguous()

    model_path = target_dir / "model.onnx"
    print(f"Direct-exporting {target_id}...", flush=True)
    with _onnx_export_mode(), torch.no_grad():
        torch.onnx.export(
            model.denoiser, args, model_path,
            input_names=input_order, output_names=["output"], opset_version=17,
            do_constant_folding=True, dynamo=False,
        )
    onnx_model = onnx.load(model_path)
    onnx.checker.check_model(onnx_model)
    graph_inputs = [item.name for item in onnx_model.graph.input]
    inputs = [save_input(target_id, name, tensors[name]) for name in graph_inputs]

    half_array = reference.cpu().numpy()
    half_array.tofile(target_dir / "reference_output_pytorch_fp16.bin")
    source_output = source_manifest["outputs"][0]
    fp32_reference = np.fromfile(
        TOY_ROOT / source_output["reference_url"], dtype=np.float32
    ).reshape(source_output["shape"])
    local_metrics = {"output": metrics(half_array, fp32_reference)}
    primary_tolerances = {"max_abs_error": 0.025, "mean_abs_error": 0.004, "min_cosine_similarity": 0.999}
    fp16_tolerances = {"max_abs_error": 0.015, "mean_abs_error": 0.002, "min_cosine_similarity": 0.9999}
    outputs = [
        {
            "name": "output", "dtype": "float16", "shape": list(half_array.shape),
            "reference_dtype": "float32", "reference_url": source_output["reference_url"],
            "compare_url": f"/api/compare/{target_id}/output", "tolerances": primary_tolerances,
            "secondary_references": [
                secondary_reference(target_id, "output", list(half_array.shape), fp16_tolerances)
            ],
        }
    ]
    return finalize_manifest(
        target_id=target_id, source_manifest=source_manifest,
        title="ARDY Core denoiser FP16 / one separated-CFG step",
        description="无 Llama；固定 10 tokens/40 frames；zero text、非零约束；直接从 PyTorch half 模型导出。",
        model_path=model_path, onnx_model=onnx_model, inputs=inputs, outputs=outputs,
        local_metrics=local_metrics, benchmark_runs=10,
    )


def main() -> None:
    torch.manual_seed(20260713)
    np.random.seed(20260713)
    torch.set_grad_enabled(False)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = "cuda:0"
    print("Loading ARDY Core with text_encoder=False and converting model to half...", flush=True)
    model = load_model("core", device=device, text_encoder=False, checkpoints_dir=str(CHECKPOINTS)).half().eval()
    if model.text_encoder is not None:
        raise RuntimeError("text encoder unexpectedly loaded")
    entries = [export_decoder(model, device), export_denoiser(model, device)]
    index_path = CASES_ROOT / "index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    ids = {item["id"] for item in entries}
    index["cases"] = [item for item in index["cases"] if item["id"] not in ids] + entries
    index_path.write_text(json.dumps(index, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
