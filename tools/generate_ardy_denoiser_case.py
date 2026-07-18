#!/usr/bin/env python
"""Export one fixed ARDY Core CFG denoiser step and PyTorch references."""

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
CASE_ID = "ardy_denoiser_fp32"
CASE_DIR = TOY_ROOT / "cases" / CASE_ID

if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from ardy.model.load_model import load_model  # noqa: E402
from scripts.export_onnx import _onnx_export_mode, make_denoiser_dummy_inputs  # noqa: E402


NP_DTYPES = {
    torch.float32: (np.float32, "float32"),
    torch.int64: (np.int64, "int64"),
    torch.int32: (np.int32, "int32"),
    torch.bool: (np.bool_, "bool"),
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
        "max_abs_error": float(abs_error.max()),
        "mean_abs_error": float(abs_error.mean()),
        "max_rel_error": float(rel_error.max()),
        "mean_rel_error": float(rel_error.mean()),
        "cosine_similarity": cosine,
    }


def update_case_index(case_entry: dict) -> None:
    index_path = TOY_ROOT / "cases" / "index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
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
    model = load_model("core", device=device, text_encoder=False, checkpoints_dir=str(CHECKPOINTS))
    if model.text_encoder is not None:
        raise RuntimeError("text encoder unexpectedly loaded")
    denoiser = model.denoiser.eval()

    dummy = make_denoiser_dummy_inputs(denoiser, num_tokens=10, num_text_tokens=1, device=device)
    # This path-only case explicitly removes text conditioning. Constraints are
    # kept non-zero to exercise the constraint branch of separated CFG.
    dummy["cfg_weight_text"].zero_()
    dummy["cfg_weight_cstr"].fill_(1.5)
    dummy["text_feat"].zero_()
    dummy["text_feat_pad_mask"].zero_()
    dummy["timesteps"].fill_(500)
    dummy["first_heading_angle"].fill_(0.35)
    dummy["motion_mask"].zero_()
    dummy["observed_motion"].zero_()
    # Three deterministic generation-frame constraints. The concrete feature
    # values are normalized-space test values; this tests graph transport and
    # numerics, while end-to-end path construction remains a separate case.
    for frame, feature, value in ((8, 0, 0.4), (20, 2, -0.25), (36, 0, 0.75)):
        dummy["motion_mask"][0, frame, feature] = 1.0
        dummy["observed_motion"][0, frame, feature] = value

    input_names = list(dummy)
    args = tuple(dummy[name].contiguous() for name in input_names)
    with torch.no_grad():
        reference = denoiser(*args).detach().float().contiguous().cpu().numpy()

    model_path = CASE_DIR / "model.onnx"
    print(f"Exporting fixed denoiser to {model_path}...", flush=True)
    with _onnx_export_mode(), torch.no_grad():
        torch.onnx.export(
            denoiser,
            args,
            model_path,
            input_names=input_names,
            output_names=["output"],
            opset_version=17,
            do_constant_folding=True,
            dynamo=False,
        )

    onnx_model = onnx.load(model_path)
    onnx.checker.check_model(onnx_model)
    graph_input_names = [item.name for item in onnx_model.graph.input]
    # Constant folding may remove semantically unused CFG/text inputs. Publish
    # exactly the graph inputs that ORT expects, in graph order.
    inputs = []
    cpu_feeds = {}
    for name in graph_input_names:
        tensor = dummy[name].detach().contiguous().cpu()
        if tensor.dtype not in NP_DTYPES:
            raise RuntimeError(f"unsupported input dtype for {name}: {tensor.dtype}")
        np_dtype, dtype_name = NP_DTYPES[tensor.dtype]
        array = tensor.numpy().astype(np_dtype, copy=False)
        filename = f"input_{name}.bin"
        array.tofile(CASE_DIR / filename)
        inputs.append(
            {
                "name": name,
                "dtype": dtype_name,
                "shape": list(array.shape),
                "url": f"cases/{CASE_ID}/{filename}",
            }
        )
        cpu_feeds[name] = array

    reference.tofile(CASE_DIR / "reference_output.bin")
    output_spec = {
        "name": "output",
        "dtype": "float32",
        "shape": list(reference.shape),
        "reference_url": f"cases/{CASE_ID}/reference_output.bin",
        "compare_url": f"/api/compare/{CASE_ID}/output",
        "tolerances": {
            "max_abs_error": 5.0e-2,
            "mean_abs_error": 5.0e-3,
            "min_cosine_similarity": 0.999,
        },
    }

    del denoiser, model, args
    torch.cuda.empty_cache()
    print("Running native ORT CPU cross-check...", flush=True)
    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    native = session.run(["output"], cpu_feeds)[0]
    native_metrics = metrics(native, reference)

    op_types = sorted({node.op_type for node in onnx_model.graph.node})
    manifest = {
        "id": CASE_ID,
        "title": "ARDY Core denoiser FP32 / one separated-CFG step",
        "description": "无 Llama；固定 10 tokens/40 frames；zero text；非零路径类约束输入。",
        "model_url": f"cases/{CASE_ID}/model.onnx",
        "model_sha256": sha256(model_path),
        "model_size_bytes": model_path.stat().st_size,
        "opset": 17,
        "operators": op_types,
        "inputs": inputs,
        "outputs": [output_spec],
        "tolerances": output_spec["tolerances"],
        "native_ort_cpu_vs_pytorch": {"output": native_metrics},
        "benchmark": {"warmup_runs": 3, "timed_runs": 10},
        "text_encoder_loaded": False,
        "text_conditioning_enabled": False,
        "constraint_conditioning_enabled": True,
    }
    (CASE_DIR / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    update_case_index(
        {"id": CASE_ID, "title": manifest["title"], "manifest_url": f"cases/{CASE_ID}/manifest.json"}
    )

    print(f"MODEL_SIZE={manifest['model_size_bytes']}")
    print(f"MODEL_SHA256={manifest['model_sha256']}")
    print(f"OPERATORS={','.join(op_types)}")
    print(f"GRAPH_INPUTS={','.join(graph_input_names)}")
    print(f"OUTPUT_SHAPE={reference.shape}")
    print(f"NATIVE_MAX_ABS={native_metrics['max_abs_error']:.9g}")
    print(f"NATIVE_MEAN_ABS={native_metrics['mean_abs_error']:.9g}")
    print(f"NATIVE_COSINE={native_metrics['cosine_similarity']:.12f}")


if __name__ == "__main__":
    main()
