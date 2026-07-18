#!/usr/bin/env python
"""Export root/body denoiser Transformer stages for browser WebGPU timing.

The production denoiser runs separated CFG as B=3. These fixed-shape cases
therefore benchmark each inner Transformer with B=3, T=16 and D=1024 while
preserving the checkpoint weights, prefix conditioning and positional logic.
They are diagnostic graphs only and do not replace the end-to-end model.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import onnx
import torch
from torch import nn


TOY_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = TOY_ROOT.parent
REPO = WORKSPACE / "ardy"
CHECKPOINTS = REPO / "checkpoints"

if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from ardy.model.load_model import load_model  # noqa: E402
from scripts.export_onnx import _onnx_export_mode  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class StageWrapper(nn.Module):
    def __init__(self, stage: nn.Module):
        super().__init__()
        self.stage = stage

    def forward(self, x, x_pad_mask, text_feat, timesteps, first_heading_angle, token_index):
        text_pad_mask = torch.ones(
            text_feat.shape[:2],
            dtype=torch.bool,
            device=text_feat.device,
        )
        return self.stage(
            x,
            x_pad_mask > 0.5,
            text_feat,
            text_pad_mask,
            timesteps,
            first_heading_angle,
            token_index,
        )


def update_case_index(entries: list[dict]) -> None:
    index_path = TOY_ROOT / "cases" / "index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    ids = {entry["id"] for entry in entries}
    index["cases"] = [item for item in index["cases"] if item["id"] not in ids]
    index["cases"].extend(entries)
    index_path.write_text(json.dumps(index, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def export_stage(name: str, stage: nn.Module, output_dim: int, device: str) -> dict:
    case_id = f"ardy_denoiser_{name}_stage_fp32"
    case_dir = TOY_ROOT / "cases" / case_id
    case_dir.mkdir(parents=True, exist_ok=True)
    wrapper = StageWrapper(stage).eval()

    generator = torch.Generator(device=device).manual_seed(20260713 + (0 if name == "root" else 1))
    x = torch.randn(3, 16, 1024, generator=generator, device=device) * 0.2
    x_pad_mask = torch.zeros(3, 16, dtype=torch.float32, device=device)
    x_pad_mask[:, :11] = 1
    text_feat = torch.zeros(3, 1, 4096, dtype=torch.float32, device=device)
    timesteps = torch.full((3,), 5, dtype=torch.int64, device=device)
    first_heading_angle = torch.full((3,), 0.1, dtype=torch.float32, device=device)
    token_index = torch.arange(-1, 15, dtype=torch.int64, device=device)[None].repeat(3, 1)
    args = (x, x_pad_mask, text_feat, timesteps, first_heading_angle, token_index)

    with torch.no_grad():
        reference = wrapper(*args).detach().float().contiguous().cpu().numpy()
    model_path = case_dir / "model.onnx"
    print(f"Exporting {name} stage to {model_path}...", flush=True)
    with _onnx_export_mode(), torch.no_grad():
        torch.onnx.export(
            wrapper,
            args,
            model_path,
            input_names=[
                "x",
                "x_pad_mask",
                "text_feat",
                "timesteps",
                "first_heading_angle",
                "token_index",
            ],
            output_names=["output"],
            opset_version=17,
            do_constant_folding=True,
            dynamo=False,
        )

    graph = onnx.load(model_path)
    onnx.checker.check_model(graph)
    graph_inputs = {item.name for item in graph.graph.input}
    inputs = []
    arrays = {
        "x": x.detach().cpu().numpy().astype(np.float32),
        "x_pad_mask": x_pad_mask.detach().cpu().numpy().astype(np.float32),
        "text_feat": text_feat.detach().cpu().numpy().astype(np.float32),
        "timesteps": timesteps.detach().cpu().numpy().astype(np.int64),
        "first_heading_angle": first_heading_angle.detach().cpu().numpy().astype(np.float32),
        "token_index": token_index.detach().cpu().numpy().astype(np.int64),
    }
    for input_name, array in arrays.items():
        if input_name not in graph_inputs:
            continue
        filename = f"input_{input_name}.bin"
        array.tofile(case_dir / filename)
        inputs.append(
            {
                "name": input_name,
                "dtype": "int64" if array.dtype == np.int64 else "float32",
                "shape": list(array.shape),
                "url": f"cases/{case_id}/{filename}",
            }
        )

    reference.tofile(case_dir / "reference_output.bin")
    tolerance = {
        "max_abs_error": 0.01,
        "mean_abs_error": 0.001,
        "min_cosine_similarity": 0.99999,
    }
    manifest = {
        "id": case_id,
        "title": f"ARDY Core denoiser {name} Transformer FP32 / B3 T16",
        "description": (
            "Separated-CFG inner stage diagnostic graph; exact checkpoint block; "
            f"[3,16,1024] to [3,16,{output_dim}]; no Llama encoder."
        ),
        "model_url": f"cases/{case_id}/model.onnx",
        "model_sha256": sha256(model_path),
        "model_size_bytes": model_path.stat().st_size,
        "parameter_count": sum(parameter.numel() for parameter in stage.parameters()),
        "opset": 17,
        "operators": sorted({node.op_type for node in graph.graph.node}),
        "inputs": inputs,
        "outputs": [
            {
                "name": "output",
                "dtype": "float32",
                "shape": list(reference.shape),
                "reference_url": f"cases/{case_id}/reference_output.bin",
                "compare_url": f"/api/compare/{case_id}/output",
                "tolerances": tolerance,
            }
        ],
        "tolerances": tolerance,
        "benchmark": {"warmup_runs": 3, "timed_runs": 20},
        "diagnostic_only": True,
        "text_encoder_loaded": False,
    }
    (case_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        f"{name}: params={manifest['parameter_count']} size={manifest['model_size_bytes']} "
        f"inputs={[item['name'] for item in inputs]} output={list(reference.shape)}",
        flush=True,
    )
    return {
        "id": case_id,
        "title": manifest["title"],
        "manifest_url": f"cases/{case_id}/manifest.json",
    }


def main() -> None:
    torch.manual_seed(20260713)
    np.random.seed(20260713)
    torch.set_grad_enabled(False)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this export path")
    device = "cuda:0"
    print("Loading ARDY Core with text_encoder=False...", flush=True)
    model = load_model(
        "core",
        device=device,
        text_encoder=False,
        checkpoints_dir=str(CHECKPOINTS),
    ).eval()
    inner = model.denoiser.model
    entries = [
        export_stage("root", inner.root_model, 20, device),
        export_stage("body", inner.body_model, 128, device),
    ]
    update_case_index(entries)


if __name__ == "__main__":
    main()
