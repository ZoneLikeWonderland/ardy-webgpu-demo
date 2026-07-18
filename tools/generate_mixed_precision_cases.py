#!/usr/bin/env python
"""Build mixed FP16/FP32 ARDY ONNX cases from validated FP32 exports."""

from __future__ import annotations

import hashlib
import json
import warnings
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnxconverter_common import float16


TOY_ROOT = Path(__file__).resolve().parents[1]
CASES_ROOT = TOY_ROOT / "cases"
DTYPES = {
    "float32": np.dtype("<f4"),
    "float16": np.dtype("<f2"),
    "int64": np.dtype("<i8"),
    "int32": np.dtype("<i4"),
    "bool": np.dtype("u1"),
}
FP32_OPS = [
    "LayerNormalization",
    "Softmax",
    "Div",
    "Sqrt",
    "Atan",
    "Sin",
    "Cos",
    "Erf",
    "Sigmoid",
    "Round",
    "ReduceSum",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def metrics(actual: np.ndarray, reference: np.ndarray) -> dict[str, float]:
    actual64 = actual.astype(np.float64)
    reference64 = reference.astype(np.float64)
    absolute = np.abs(actual64 - reference64)
    relative = absolute / np.maximum(np.abs(reference64), 1.0e-8)
    denom = np.linalg.norm(actual64.ravel()) * np.linalg.norm(reference64.ravel())
    return {
        "max_abs_error": float(absolute.max(initial=0.0)),
        "mean_abs_error": float(absolute.mean()) if absolute.size else 0.0,
        "max_rel_error": float(relative.max(initial=0.0)),
        "mean_rel_error": float(relative.mean()) if relative.size else 0.0,
        "cosine_similarity": float(np.dot(actual64.ravel(), reference64.ravel()) / denom),
    }


def repair_cast_value_info(model) -> int:
    """Repair an onnxconverter-common bug around pre-existing Cast nodes."""
    values = {
        item.name: item
        for item in list(model.graph.value_info) + list(model.graph.input) + list(model.graph.output)
    }
    repaired = 0
    for node in model.graph.node:
        if node.op_type != "Cast":
            continue
        target_type = next((attribute.i for attribute in node.attribute if attribute.name == "to"), None)
        for output_name in node.output:
            value_info = values.get(output_name)
            if value_info is not None and value_info.type.tensor_type.elem_type != target_type:
                value_info.type.tensor_type.elem_type = target_type
                repaired += 1
    return repaired


def retarget_nonblocked_casts(model) -> int:
    """Retarget source FP32 mask/text casts when their consumer remains FP16."""
    value_infos = {
        item.name: item
        for item in list(model.graph.value_info) + list(model.graph.input) + list(model.graph.output)
    }
    types = {
        name: item.type.tensor_type.elem_type
        for name, item in value_infos.items()
        if item.type.tensor_type.elem_type
    }
    types.update({item.name: item.data_type for item in model.graph.initializer})
    producers = {output: node for node in model.graph.node for output in node.output}
    retargeted = 0
    for node in model.graph.node:
        known_types = [types.get(name) for name in node.input if name]
        if (
            node.op_type in FP32_OPS
            or onnx.TensorProto.FLOAT16 not in known_types
            or onnx.TensorProto.FLOAT not in known_types
        ):
            continue
        for input_name in node.input:
            if types.get(input_name) != onnx.TensorProto.FLOAT:
                continue
            producer = producers.get(input_name)
            if producer is None or producer.op_type != "Cast":
                continue
            attribute = next((item for item in producer.attribute if item.name == "to"), None)
            if attribute is None or attribute.i != onnx.TensorProto.FLOAT:
                continue
            attribute.i = onnx.TensorProto.FLOAT16
            types[input_name] = onnx.TensorProto.FLOAT16
            if input_name in value_infos:
                value_infos[input_name].type.tensor_type.elem_type = onnx.TensorProto.FLOAT16
            retargeted += 1
    return retargeted


def build_case(source_id: str, pure_fp16_id: str, target_id: str, title: str) -> dict:
    source_dir = CASES_ROOT / source_id
    target_dir = CASES_ROOT / target_id
    target_dir.mkdir(parents=True, exist_ok=True)
    source_manifest = json.loads((source_dir / "manifest.json").read_text(encoding="utf-8"))
    fp16_manifest = json.loads((CASES_ROOT / pure_fp16_id / "manifest.json").read_text(encoding="utf-8"))

    # The installed converter has a cleanup bug for adjacent casts. Skipping
    # that optional cleanup preserves a valid graph; we then repair stale type
    # annotations on Cast outputs and validate with both ONNX checker and ORT.
    original_cleanup = float16.remove_unnecessary_cast_node
    float16.remove_unnecessary_cast_node = lambda graph: None
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            model = float16.convert_float_to_float16(
                onnx.load(source_dir / "model.onnx"),
                keep_io_types=False,
                op_block_list=float16.DEFAULT_OP_BLOCK_LIST + FP32_OPS,
            )
    finally:
        float16.remove_unnecessary_cast_node = original_cleanup
    repaired_casts = repair_cast_value_info(model)
    retargeted_casts = retarget_nonblocked_casts(model)
    repaired_casts += repair_cast_value_info(model)
    model_path = target_dir / "model.onnx"
    onnx.checker.check_model(model)
    onnx.save(model, model_path)

    inputs = []
    feeds = {}
    fp16_inputs = {item["name"]: item for item in fp16_manifest["inputs"]}
    for graph_input in model.graph.input:
        source_spec = fp16_inputs[graph_input.name]
        array = np.fromfile(TOY_ROOT / source_spec["url"], dtype=DTYPES[source_spec["dtype"]]).reshape(
            source_spec["shape"]
        )
        filename = f"input_{graph_input.name}.bin"
        array.tofile(target_dir / filename)
        spec = {
            "name": graph_input.name,
            "dtype": source_spec["dtype"],
            "shape": source_spec["shape"],
            "url": f"cases/{target_id}/{filename}",
        }
        inputs.append(spec)
        feeds[graph_input.name] = array

    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    output_names = [item.name for item in model.graph.output]
    native_outputs = session.run(output_names, feeds)
    source_outputs = {item["name"]: item for item in source_manifest["outputs"]}
    outputs = []
    native_metrics = {}
    for graph_output, actual in zip(model.graph.output, native_outputs):
        name = graph_output.name
        source_spec = source_outputs[name]
        reference = np.fromfile(
            TOY_ROOT / source_spec["reference_url"], dtype=DTYPES[source_spec["dtype"]]
        ).reshape(source_spec["shape"])
        native_metrics[name] = metrics(actual, reference)
        outputs.append(
            {
                "name": name,
                "dtype": "float16",
                "shape": source_spec["shape"],
                "reference_dtype": source_spec["dtype"],
                "reference_url": source_spec["reference_url"],
                "compare_url": f"/api/compare/{target_id}/{name}",
                "tolerances": source_spec["tolerances"],
            }
        )

    manifest = {
        "id": target_id,
        "title": title,
        "description": source_manifest["description"] + "；FP16 大矩阵/权重，敏感算子 FP32。",
        "source_case": source_id,
        "model_url": f"cases/{target_id}/model.onnx",
        "model_sha256": sha256(model_path),
        "model_size_bytes": model_path.stat().st_size,
        "source_model_size_bytes": source_manifest["model_size_bytes"],
        "size_ratio_vs_fp32": model_path.stat().st_size / source_manifest["model_size_bytes"],
        "opset": 17,
        "operators": sorted({node.op_type for node in model.graph.node}),
        "inputs": inputs,
        "outputs": outputs,
        "tolerances": {item["name"]: item["tolerances"] for item in outputs},
        "native_ort_cpu_vs_pytorch_fp32": native_metrics,
        "benchmark": {"warmup_runs": 3, "timed_runs": 10},
        "precision": "mixed_float16_float32",
        "fp32_operator_types": FP32_OPS,
        "repaired_cast_value_info_count": repaired_casts,
        "retargeted_nonblocked_cast_count": retargeted_casts,
        "text_encoder_loaded": False,
    }
    (target_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"{target_id}: {manifest['model_size_bytes']} bytes "
        f"({manifest['size_ratio_vs_fp32']:.3%} of FP32), repaired_casts={repaired_casts}, "
        f"retargeted_casts={retargeted_casts}, "
        f"SHA256={manifest['model_sha256']}"
    )
    for name, values in native_metrics.items():
        print(
            f"  ORT mixed vs PyTorch FP32 {name}: max_abs={values['max_abs_error']:.9g}, "
            f"mean_abs={values['mean_abs_error']:.9g}, cosine={values['cosine_similarity']:.12f}"
        )
    return {"id": target_id, "title": title, "manifest_url": f"cases/{target_id}/manifest.json"}


def main() -> None:
    entries = [
        build_case(
            "ardy_decoder_fp32",
            "ardy_decoder_fp16",
            "ardy_decoder_mixed_fp16",
            "ARDY Core FSQ decoder mixed FP16/FP32",
        ),
        build_case(
            "ardy_denoiser_fp32",
            "ardy_denoiser_fp16",
            "ardy_denoiser_mixed_fp16",
            "ARDY Core denoiser mixed FP16/FP32",
        ),
    ]
    index_path = CASES_ROOT / "index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    ids = {item["id"] for item in entries}
    index["cases"] = [item for item in index["cases"] if item["id"] not in ids] + entries
    index_path.write_text(json.dumps(index, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
