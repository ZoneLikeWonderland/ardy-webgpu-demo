#!/usr/bin/env python
"""Create a parity/latency case for the exact 16-token infinite-demo denoiser."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort


TOY_ROOT = Path(__file__).resolve().parents[1]
CASE_ID = "ardy_denoiser_window16_fp32"
CASE_DIR = TOY_ROOT / "cases" / CASE_ID
MODEL_PATH = TOY_ROOT / "infinite_demo" / "models" / "fp32" / "denoiser.onnx"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    CASE_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20260713)
    arrays: dict[str, np.ndarray] = {
        "cfg_weight_text": np.asarray([0.0], dtype=np.float32),
        "cfg_weight_cstr": np.asarray([1.5], dtype=np.float32),
        "x": (rng.standard_normal((1, 16, 148)).astype(np.float32) * np.float32(0.2)),
        "history_len": np.asarray([4], dtype=np.int64),
        "generation_len": np.asarray([40], dtype=np.int64),
        "history_mask": np.zeros((1, 64), dtype=np.float32),
        "generation_mask": np.zeros((1, 64), dtype=np.float32),
        "history_token_mask": np.zeros((1, 16), dtype=np.float32),
        "generation_token_mask": np.zeros((1, 16), dtype=np.float32),
        "future_token_mask": np.zeros((1, 16), dtype=np.float32),
        "text_feat": np.zeros((1, 1, 4096), dtype=np.float32),
        "timesteps": np.asarray([5], dtype=np.int64),
        "first_heading_angle": np.asarray([0.1], dtype=np.float32),
        "motion_mask": np.zeros((1, 64, 330), dtype=np.float32),
        "observed_motion": np.zeros((1, 64, 330), dtype=np.float32),
    }
    arrays["history_mask"][:, :4] = 1
    arrays["generation_mask"][:, 4:44] = 1
    arrays["history_token_mask"][:, :1] = 1
    arrays["generation_token_mask"][:, 1:11] = 1

    session = ort.InferenceSession(str(MODEL_PATH), providers=["CPUExecutionProvider"])
    graph_input_names = [item.name for item in session.get_inputs()]
    missing = set(graph_input_names) - arrays.keys()
    if missing:
        raise RuntimeError(f"missing generated inputs: {sorted(missing)}")
    feeds = {name: arrays[name] for name in graph_input_names}
    reference = session.run(["output"], feeds)[0].astype(np.float32)
    reference.tofile(CASE_DIR / "reference_output.bin")

    inputs = []
    for name in graph_input_names:
        array = arrays[name]
        filename = f"input_{name}.bin"
        array.tofile(CASE_DIR / filename)
        inputs.append(
            {
                "name": name,
                "dtype": "int64" if array.dtype == np.int64 else "float32",
                "shape": list(array.shape),
                "url": f"cases/{CASE_ID}/{filename}",
            }
        )

    graph = onnx.load(MODEL_PATH)
    onnx.checker.check_model(graph)
    tolerance = {
        "max_abs_error": 0.005,
        "mean_abs_error": 0.001,
        "min_cosine_similarity": 0.99999,
    }
    manifest = {
        "id": CASE_ID,
        "title": "ARDY Core denoiser FP32 / current 16-token window",
        "description": "无限前端实际 denoiser；B1 输入，图内 separated CFG B3；4 history + 40 generation + 20 visible future frames。",
        "model_url": "infinite_demo/models/fp32/denoiser.onnx",
        "model_sha256": sha256(MODEL_PATH),
        "model_size_bytes": MODEL_PATH.stat().st_size,
        "opset": 17,
        "operators": sorted({node.op_type for node in graph.graph.node}),
        "inputs": inputs,
        "outputs": [
            {
                "name": "output",
                "dtype": "float32",
                "shape": list(reference.shape),
                "reference_url": f"cases/{CASE_ID}/reference_output.bin",
                "compare_url": f"/api/compare/{CASE_ID}/output",
                "tolerances": tolerance,
            }
        ],
        "tolerances": tolerance,
        "benchmark": {"warmup_runs": 3, "timed_runs": 20},
        "text_encoder_loaded": False,
        "text_conditioning_enabled": False,
        "constraint_conditioning_enabled": True,
    }
    (CASE_DIR / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    index_path = TOY_ROOT / "cases" / "index.json"
    case_index = json.loads(index_path.read_text(encoding="utf-8"))
    case_index["cases"] = [item for item in case_index["cases"] if item["id"] != CASE_ID]
    case_index["cases"].append(
        {
            "id": CASE_ID,
            "title": manifest["title"],
            "manifest_url": f"cases/{CASE_ID}/manifest.json",
        }
    )
    index_path.write_text(json.dumps(case_index, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"CASE={CASE_ID}")
    print(f"MODEL_SIZE={manifest['model_size_bytes']}")
    print(f"INPUTS={graph_input_names}")
    print(f"OUTPUT_SHAPE={list(reference.shape)}")


if __name__ == "__main__":
    main()
