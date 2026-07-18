#!/usr/bin/env python
"""Instrument final FP16 ONNX graphs for layerwise CPU-ORT/WebGPU parity.

The production graphs remain unchanged.  This tool adds selected existing
intermediate tensors as graph outputs, evaluates the instrumented graphs with
CPU ORT, and writes one continuation-window golden case for the browser.  The
probe is diagnostic-only and is not counted as part of the release download.
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort


TOY_ROOT = Path(__file__).resolve().parents[1]
DEMO_ROOT = TOY_ROOT / "infinite_demo"
MANIFEST = DEMO_ROOT / "manifest.json"
RELEASE = json.loads(MANIFEST.read_text(encoding="utf-8"))["model_release"]
MODEL_ROOT = DEMO_ROOT / "models" / "student" / RELEASE
SOURCE_ROOT = MODEL_ROOT / "fp16"
PROBE_ROOT = MODEL_ROOT / "diagnostics" / "precision_probe"
BASE_CASE = DEMO_ROOT / "browser_validation_case.json"
OUTPUT_CASE = DEMO_ROOT / "webgpu_precision_probe_case.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def selected_outputs(module: str) -> list[tuple[str, str]]:
    if module == "encoder":
        rows = [("input_gelu", "/Mul_1_output_0")]
        for index in range(4):
            if index in {0, 3}:
                rows.extend(
                    [
                        (f"block_{index}_norm", f"/blocks.{index}/norm/LayerNormalization_output_0"),
                        (f"block_{index}_fc1", f"/blocks.{index}/fc1/Gemm_output_0"),
                        (f"block_{index}_gelu", f"/blocks.{index}/Mul_1_output_0"),
                        (f"block_{index}_fc2", f"/blocks.{index}/fc2/Gemm_output_0"),
                    ]
                )
            rows.append((f"block_{index}", f"/blocks.{index}/Add_1_output_0"))
        rows.append(("output_norm", "/output_norm/LayerNormalization_output_0"))
        rows.append(("output", "output"))
        return rows
    if module == "flow":
        rows = [("input_norm", "/model/input_norm/LayerNormalization_output_0")]
        for index in range(8):
            if index in {0, 7}:
                prefix = f"/model/trunk.{index}"
                rows.extend(
                    [
                        (f"trunk_{index}_attention_norm", f"{prefix}/attention_norm/LayerNormalization_output_0"),
                        (f"trunk_{index}_qkv", f"{prefix}/qkv/Add_output_0"),
                        (f"trunk_{index}_qk", f"{prefix}/MatMul_output_0"),
                        (f"trunk_{index}_softmax", f"{prefix}/Softmax_output_0"),
                        (f"trunk_{index}_weighted_v", f"{prefix}/MatMul_1_output_0"),
                        (f"trunk_{index}_attention_proj", f"{prefix}/attention_out/Add_output_0"),
                    ]
                )
            rows.extend(
                [
                    (f"trunk_{index}_attention", f"/model/trunk.{index}/Add_output_0"),
                ]
            )
            if index in {0, 7}:
                prefix = f"/model/trunk.{index}/channel"
                rows.extend(
                    [
                        (f"trunk_{index}_channel_norm", f"{prefix}/norm/LayerNormalization_output_0"),
                        (f"trunk_{index}_channel_fc1", f"{prefix}/fc1/Add_output_0"),
                        (f"trunk_{index}_channel_gelu", f"{prefix}/Mul_1_output_0"),
                        (f"trunk_{index}_channel_fc2", f"{prefix}/fc2/Add_output_0"),
                    ]
                )
            rows.extend(
                [
                    (f"trunk_{index}_channel", f"/model/trunk.{index}/channel/Add_1_output_0"),
                ]
            )
        rows.extend(
            [
                ("root_velocity", "/model/root_head/Add_output_0"),
                ("body_input", "/model/Add_9_output_0"),
            ]
        )
        for index in range(8):
            if index in {0, 7}:
                prefix = f"/model/body_refiner.{index}"
                rows.extend(
                    [
                        (f"body_{index}_attention_norm", f"{prefix}/attention_norm/LayerNormalization_output_0"),
                        (f"body_{index}_qkv", f"{prefix}/qkv/Add_output_0"),
                        (f"body_{index}_qk", f"{prefix}/MatMul_output_0"),
                        (f"body_{index}_softmax", f"{prefix}/Softmax_output_0"),
                        (f"body_{index}_weighted_v", f"{prefix}/MatMul_1_output_0"),
                        (f"body_{index}_attention_proj", f"{prefix}/attention_out/Add_output_0"),
                    ]
                )
            rows.extend(
                [
                    (
                        f"body_{index}_attention",
                        f"/model/body_refiner.{index}/Add_output_0",
                    ),
                ]
            )
            if index in {0, 7}:
                prefix = f"/model/body_refiner.{index}/channel"
                rows.extend(
                    [
                        (f"body_{index}_channel_norm", f"{prefix}/norm/LayerNormalization_output_0"),
                        (f"body_{index}_channel_fc1", f"{prefix}/fc1/Add_output_0"),
                        (f"body_{index}_channel_gelu", f"{prefix}/Mul_1_output_0"),
                        (f"body_{index}_channel_fc2", f"{prefix}/fc2/Add_output_0"),
                    ]
                )
            rows.extend(
                [
                    (
                        f"body_{index}_channel",
                        f"/model/body_refiner.{index}/channel/Add_1_output_0",
                    ),
                ]
            )
        rows.extend(
            [
                ("body_velocity", "/model/body_head/Add_output_0"),
                ("output", "output"),
            ]
        )
        return rows
    if module == "decoder":
        rows = [("input_gelu_valid", "/Mul_2_output_0")]
        for index in range(8):
            if index in {0, 3, 7}:
                prefix = f"/blocks.{index}"
                rows.extend(
                    [
                        (f"block_{index}_token_norm", f"{prefix}/token_norm/LayerNormalization_output_0"),
                        (f"block_{index}_token_fc1", f"{prefix}/token_fc1/Add_output_0"),
                        (f"block_{index}_token_gelu", f"{prefix}/Mul_1_output_0"),
                        (f"block_{index}_token_fc2", f"{prefix}/token_fc2/Add_output_0"),
                    ]
                )
            rows.extend(
                [
                    (f"block_{index}_token", f"/blocks.{index}/Mul_2_output_0"),
                ]
            )
            if index in {0, 3, 7}:
                prefix = f"/blocks.{index}/channel_block"
                rows.extend(
                    [
                        (f"block_{index}_channel_norm", f"{prefix}/norm/LayerNormalization_output_0"),
                        (f"block_{index}_channel_fc1", f"{prefix}/fc1/Add_output_0"),
                        (f"block_{index}_channel_gelu", f"{prefix}/Mul_1_output_0"),
                        (f"block_{index}_channel_fc2", f"{prefix}/fc2/Add_output_0"),
                    ]
                )
            rows.extend(
                [
                    (f"block_{index}_channel", f"/blocks.{index}/Mul_3_output_0"),
                ]
            )
        rows.extend(
            [
                ("output_norm", "/output_norm/LayerNormalization_output_0"),
                ("output", "output"),
            ]
        )
        return rows
    raise ValueError(module)


def instrument(module: str) -> tuple[Path, list[tuple[str, str]]]:
    source = SOURCE_ROOT / f"{module}.onnx"
    destination = PROBE_ROOT / f"{module}.onnx"
    destination.parent.mkdir(parents=True, exist_ok=True)
    model = onnx.load(source)
    inferred = onnx.shape_inference.infer_shapes(model)
    value_info = {
        item.name: item
        for item in [
            *inferred.graph.input,
            *inferred.graph.value_info,
            *inferred.graph.output,
        ]
    }
    existing = {item.name for item in model.graph.output}
    rows = selected_outputs(module)
    for _, tensor_name in rows:
        if tensor_name not in value_info:
            raise KeyError(f"{module}: missing inferred tensor {tensor_name}")
        if tensor_name not in existing:
            model.graph.output.append(copy.deepcopy(value_info[tensor_name]))
            existing.add(tensor_name)
    onnx.checker.check_model(model)
    onnx.save(model, destination)
    return destination, rows


def tensor_spec(array: np.ndarray) -> dict:
    values = np.asarray(array)
    return {
        "shape": list(values.shape),
        "values": values.astype(np.float32).reshape(-1).tolist(),
    }


def run_probe(
    model_path: Path,
    rows: list[tuple[str, str]],
    *,
    inputs: dict[str, np.ndarray],
) -> dict:
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(
        str(model_path),
        sess_options=options,
        providers=["CPUExecutionProvider"],
    )
    output_names = [name for _, name in rows]
    actual = session.run(output_names, inputs)
    outputs = []
    for (label, tensor_name), value in zip(rows, actual, strict=True):
        if not np.isfinite(value).all():
            raise RuntimeError(f"{model_path}: non-finite output {label}")
        outputs.append(
            {
                "label": label,
                "tensor_name": tensor_name,
                "reference": tensor_spec(value),
            }
        )
    return {
        "model_url": str(model_path.relative_to(TOY_ROOT)).replace("\\", "/"),
        "size_bytes": model_path.stat().st_size,
        "sha256": sha256(model_path),
        "inputs": {name: tensor_spec(value) for name, value in inputs.items()},
        "outputs": outputs,
    }


def continuation_inputs() -> tuple[dict, dict]:
    golden = json.loads(BASE_CASE.read_text(encoding="utf-8"))
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if golden["model_release"] != RELEASE or manifest["model_release"] != RELEASE:
        raise RuntimeError("probe release does not match current browser release")
    spec = golden["continuation"]
    history = np.asarray(spec["history"], dtype=np.float32).reshape(1, 4, 330)
    root_mean = np.asarray(manifest["root_stats"]["mean"], dtype=np.float32)
    root_std = np.asarray(manifest["root_stats"]["std_eps"], dtype=np.float32)
    translation = np.array(
        [[
            history[0, 3, 0] * root_std[0] + root_mean[0],
            0.0,
            history[0, 3, 2] * root_std[2] + root_mean[2],
        ]],
        dtype=np.float32,
    )
    heading_cos = history[0, 0, 3] * root_std[3] + root_mean[3]
    heading_sin = history[0, 0, 4] * root_std[4] + root_mean[4]
    angle = np.arctan2(heading_sin, heading_cos)
    first_heading = np.asarray([[np.cos(angle), np.sin(angle)]], dtype=np.float32)
    history_hybrid = np.asarray(
        spec["expected"]["history_hybrid"], dtype=np.float32
    ).reshape(1, 1, 148)
    clean = np.asarray(spec["expected"]["clean"], dtype=np.float32).reshape(1, 10, 148)
    return {
        "history": history,
        "translation": translation,
        "first_heading": first_heading,
        "history_hybrid": history_hybrid,
        "clean": clean,
        "noise": np.asarray(spec["noise"], dtype=np.float32).reshape(1, 10, 148),
        "path_condition": np.asarray(spec["path_condition"], dtype=np.float32).reshape(1, 64, 3),
        "text_feature": np.asarray(
            golden["conditions"]["text_feature"], dtype=np.float32
        ).reshape(1, manifest["text_conditioning"]["feature_dim"]),
        "heading_condition": np.asarray(
            golden["conditions"]["heading_condition"], dtype=np.float32
        ).reshape(1, 64, 3),
    }, manifest


def decoder_inputs(
    prepared: dict,
) -> dict[str, np.ndarray]:
    utility_path = MODEL_ROOT / "shared" / "finalize_continuation.onnx"
    session = ort.InferenceSession(
        str(utility_path), providers=["CPUExecutionProvider"]
    )
    local_hybrid = np.concatenate(
        [prepared["history_hybrid"], prepared["clean"]], axis=1
    ).astype(np.float32)
    _, latent, local_root, token_valid = session.run(
        ["global_root", "decoder_latent", "decoder_local_root", "token_valid"],
        {
            "local_hybrid": local_hybrid,
            "global_translation": prepared["translation"],
        },
    )
    return {
        "latent_tokens": latent.astype(np.float16),
        "local_root": local_root.astype(np.float16),
        "token_valid": token_valid.astype(np.float16),
    }


def main() -> None:
    prepared, manifest = continuation_inputs()
    paths_and_rows = {
        module: instrument(module) for module in ("encoder", "flow", "decoder")
    }
    cases = {
        "encoder": run_probe(
            *paths_and_rows["encoder"],
            inputs={
                "normalized_body": prepared["history"][..., 5:].astype(np.float16),
            },
        ),
        "flow": run_probe(
            *paths_and_rows["flow"],
            inputs={
                "noise": prepared["noise"].astype(np.float16),
                "history_hybrid": prepared["history_hybrid"].astype(np.float16),
                "path_condition": prepared["path_condition"].astype(np.float16),
                "first_heading": prepared["first_heading"].astype(np.float16),
                "has_history": np.ones((1, 1), dtype=np.float16),
                "text_feature": prepared["text_feature"].astype(np.float16),
                "heading_condition": prepared["heading_condition"].astype(np.float16),
            },
        ),
        "decoder": run_probe(
            *paths_and_rows["decoder"],
            inputs=decoder_inputs(prepared),
        ),
    }
    result = {
        "schema": "ardy_webgpu_precision_probe_v1",
        "model_release": RELEASE,
        "precision": "fp16",
        "source_validation_case": str(BASE_CASE.relative_to(TOY_ROOT)),
        "model_config": manifest["model_config"],
        "cases": cases,
    }
    OUTPUT_CASE.write_text(
        json.dumps(result, ensure_ascii=False, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {OUTPUT_CASE} ({OUTPUT_CASE.stat().st_size:,} B)")
    for name, case in cases.items():
        print(
            f"{name}: graph={case['size_bytes']:,} B, "
            f"outputs={len(case['outputs'])}"
        )


if __name__ == "__main__":
    main()
