#!/usr/bin/env python
"""Export the fixed-shape models used by the minimal infinite WebGPU demo.

The browser mirrors ARDY's interactive default: one 4-frame history token plus
10 generation tokens (40 frames).  Text conditioning is deliberately disabled;
the constraint branch remains active for mouse-selected root waypoints.
"""

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
DEMO_ROOT = TOY_ROOT / "infinite_demo"
MODELS_ROOT = DEMO_ROOT / "models"

if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from ardy.model.ardy_model import translate_normalized_root_motion  # noqa: E402
from ardy.model.load_model import load_model  # noqa: E402
from scripts.export_onnx import _onnx_export_mode, make_denoiser_dummy_inputs, make_decoder_dummy_inputs  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def export_module(
    module,
    args,
    path: Path,
    input_names: list[str],
    output_names: list[str],
    *,
    do_constant_folding: bool = True,
) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    with _onnx_export_mode(), torch.no_grad():
        torch.onnx.export(
            module,
            args,
            path,
            input_names=input_names,
            output_names=output_names,
            opset_version=17,
            do_constant_folding=do_constant_folding,
            dynamo=False,
        )
    graph = onnx.load(path)
    onnx.checker.check_model(graph)
    return {
        "url": str(path.relative_to(TOY_ROOT)).replace("\\", "/"),
        "size_bytes": path.stat().st_size,
        "sha256": sha256(path),
        "inputs": [item.name for item in graph.graph.input],
        "outputs": [item.name for item in graph.graph.output],
        "operators": sorted({node.op_type for node in graph.graph.node}),
    }


class PrepareHistory(torch.nn.Module):
    """Recenter the last global history token exactly as ARDY does."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, global_history):
        root, latent = self.model.hybrid.get_root_and_latent_body_motion_from_hybrid(global_history)
        root_world = self.model.motion_rep.global_root_stats.unnormalize(root)
        heading = torch.atan2(root_world[:, 0, 4], root_world[:, 0, 3])
        center_idx = torch.full(
            (global_history.shape[0],),
            root.shape[1] - 1,
            dtype=torch.long,
            device=global_history.device,
        )
        local_root, center = self.model.motion_rep.recenter_root_motion(
            root,
            center_idx,
            is_normalized=True,
            to_normalize=True,
            return_center_pos=True,
        )
        local_history = self.model.hybrid.get_hybrid_motion_from_root_and_latent_body_motion(local_root, latent)
        return local_history, center, heading


class EncodeHistory(torch.nn.Module):
    """Encode four explicit frames and recenter them like autoregressive_step."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, normalized_history):
        return self.model._encode_init_history(normalized_history, normalized_history.shape[0])


class FinalizeHybrid(torch.nn.Module):
    """Restore world translation, quantize FSQ latents and form decoder inputs."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, local_hybrid, global_translation, motion_len):
        local_root, latent = self.model.hybrid.get_root_and_latent_body_motion_from_hybrid(local_hybrid)
        global_root = translate_normalized_root_motion(local_root, global_translation, self.model.motion_rep)
        latent = self.model.autoencoder.requantize(latent)
        external_cond = self.model.motion_rep.global_root_to_local_root(
            global_root,
            normalized=True,
            lengths=motion_len,
        )
        global_hybrid = self.model.hybrid.get_hybrid_motion_from_root_and_latent_body_motion(global_root, latent)
        return global_hybrid, global_root, latent, external_cond


class MotionToJoints(torch.nn.Module):
    """Convert the normalized explicit representation to posed skeleton joints."""

    def __init__(self, motion_rep):
        super().__init__()
        self.motion_rep = motion_rep

    def forward(self, normalized_motion):
        motion = self.motion_rep.unnormalize(normalized_motion)
        # The representation already carries joint positions.  This official
        # inverse branch avoids exporting the rotation-based FK Python loop,
        # which legacy torch.onnx cannot lower, and is sufficient for the
        # browser's lightweight skeleton visualization.
        decoded = self.motion_rep.inverse(
            motion,
            is_normalized=False,
            posed_joints_from="positions",
        )
        return decoded["posed_joints"]


def denoiser_args(model, *, dtype: torch.dtype, device: str):
    dummy = make_denoiser_dummy_inputs(model.denoiser, num_tokens=16, num_text_tokens=1, device=device)
    # The original waypoint is current+60 frames.  With four history frames,
    # this needs a 64-frame visible window: 4 history + 40 generation + 20 future.
    dummy["history_len"].fill_(4)
    dummy["generation_len"].fill_(40)
    dummy["future_len"].fill_(20)
    dummy["history_mask"].zero_()
    dummy["history_mask"][:, :4] = 1
    dummy["generation_mask"].zero_()
    dummy["generation_mask"][:, 4:44] = 1
    dummy["future_mask"].zero_()
    dummy["future_mask"][:, 44:] = 1
    dummy["history_token_mask"].zero_()
    dummy["history_token_mask"][:, :1] = 1
    dummy["generation_token_mask"].zero_()
    dummy["generation_token_mask"][:, 1:11] = 1
    dummy["future_token_mask"].zero_()
    dummy["future_token_mask"][:, 11:] = 1
    dummy["cfg_weight_text"].zero_()
    dummy["cfg_weight_cstr"].fill_(1.5)
    dummy["text_feat"].zero_()
    dummy["text_feat_pad_mask"].zero_()
    dummy["timesteps"].fill_(500)
    dummy["first_heading_angle"].zero_()
    dummy["motion_mask"].zero_()
    dummy["observed_motion"].zero_()
    for name, tensor in dummy.items():
        if tensor.is_floating_point():
            dummy[name] = tensor.to(dtype=dtype)
    order = list(dummy)
    return dummy, order, tuple(dummy[name].contiguous() for name in order)


def decoder_args(model, *, dtype: torch.dtype, device: str):
    dummy = make_decoder_dummy_inputs(model.autoencoder, num_tokens=11, device=device)
    dummy["latent_tokens"] = model.autoencoder.requantize(dummy["latent_tokens"].to(dtype=dtype)).contiguous()
    dummy["external_cond"] = dummy["external_cond"].to(dtype=dtype).contiguous()
    dummy["motion_pad_mask"] = dummy["motion_pad_mask"].to(dtype=dtype).contiguous()
    return (
        dummy["latent_tokens"],
        dummy["external_cond"],
        dummy["motion_pad_mask"],
    )


def export_precision(model, precision: str, device: str) -> dict:
    dtype = torch.float16 if precision == "fp16" else torch.float32
    model = model.to(dtype=dtype).eval()
    precision_root = MODELS_ROOT / precision

    _, denoiser_order, dargs = denoiser_args(model, dtype=dtype, device=device)
    print(f"Exporting {precision} 16-token denoiser...", flush=True)
    denoiser = export_module(
        model.denoiser,
        dargs,
        precision_root / "denoiser.onnx",
        denoiser_order,
        ["output"],
    )

    dec_args = decoder_args(model, dtype=dtype, device=device)
    with torch.no_grad():
        decoder_outputs = list(model.autoencoder(*dec_args))
    print(f"Exporting {precision} 11-token decoder...", flush=True)
    decoder = export_module(
        model.autoencoder,
        dec_args,
        precision_root / "decoder.onnx",
        ["latent_tokens", "external_cond", "motion_pad_mask"],
        decoder_outputs,
    )
    return {"denoiser": denoiser, "decoder": decoder}


def export_utilities(model, device: str) -> dict:
    model = model.float().eval()
    shared = MODELS_ROOT / "shared"

    global_history = torch.randn(1, 1, 148, device=device)
    global_history[:, :, 20:] = model.autoencoder.requantize(global_history[:, :, 20:])
    prepare = PrepareHistory(model).eval()
    print("Exporting prepare-history utility...", flush=True)
    prepare_spec = export_module(
        prepare,
        (global_history,),
        shared / "prepare_history.onnx",
        ["global_history"],
        ["local_history", "global_translation", "first_heading_angle"],
        do_constant_folding=False,
    )

    normalized_history = torch.randn(1, 4, model.motion_rep.motion_rep_dim, device=device)
    encode = EncodeHistory(model).eval()
    print("Exporting four-frame history encoder...", flush=True)
    encode_spec = export_module(
        encode,
        (normalized_history,),
        shared / "encode_history.onnx",
        ["normalized_history"],
        ["local_history", "global_translation", "first_heading_angle"],
        do_constant_folding=False,
    )

    local_hybrid = torch.randn(1, 11, 148, device=device)
    translation = torch.tensor([[1.25, 0.0, -0.75]], device=device)
    motion_len = torch.tensor([44], dtype=torch.int64, device=device)
    finalize = FinalizeHybrid(model).eval()
    print("Exporting finalize-hybrid utility...", flush=True)
    finalize_spec = export_module(
        finalize,
        (local_hybrid, translation, motion_len),
        shared / "finalize_hybrid.onnx",
        ["local_hybrid", "global_translation", "motion_len"],
        ["global_hybrid", "global_root", "latent_tokens", "external_cond"],
        do_constant_folding=False,
    )

    normalized_motion = torch.randn(1, 40, model.motion_rep.motion_rep_dim, device=device)
    joints = MotionToJoints(model.motion_rep).eval()
    print("Exporting motion-to-joints utility...", flush=True)
    joints_spec = export_module(
        joints,
        (normalized_motion,),
        shared / "motion_to_joints.onnx",
        ["normalized_motion"],
        ["posed_joints"],
        do_constant_folding=False,
    )

    # CPU ORT smoke checks catch unsupported or malformed utility graphs now,
    # before the browser has to diagnose them.
    checks = [
        (prepare_spec, {"global_history": global_history.detach().cpu().numpy()}),
        (encode_spec, {"normalized_history": normalized_history.detach().cpu().numpy()}),
        (
            finalize_spec,
            {
                "local_hybrid": local_hybrid.detach().cpu().numpy(),
                "global_translation": translation.detach().cpu().numpy(),
                "motion_len": motion_len.detach().cpu().numpy(),
            },
        ),
        (joints_spec, {"normalized_motion": normalized_motion.detach().cpu().numpy()}),
    ]
    for spec, feeds in checks:
        session = ort.InferenceSession(str(TOY_ROOT / spec["url"]), providers=["CPUExecutionProvider"])
        values = session.run(None, feeds)
        if not all(np.isfinite(value).all() for value in values):
            raise RuntimeError(f"non-finite utility output: {spec['url']}")
    return {
        "prepare_history": prepare_spec,
        "encode_history": encode_spec,
        "finalize_hybrid": finalize_spec,
        "motion_to_joints": joints_spec,
    }


def main() -> None:
    torch.manual_seed(20260713)
    np.random.seed(20260713)
    torch.set_grad_enabled(False)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for model export")
    device = "cuda:0"
    MODELS_ROOT.mkdir(parents=True, exist_ok=True)

    print("Loading ARDY Core with text_encoder=False...", flush=True)
    model = load_model("core", device=device, text_encoder=False, checkpoints_dir=str(CHECKPOINTS)).eval()
    if model.text_encoder is not None:
        raise RuntimeError("text encoder unexpectedly loaded")

    steps = 10
    use_timesteps, map_tensor = model.diffusion.space_timesteps(steps)
    model.diffusion.calc_diffusion_vars(use_timesteps)
    schedule = {
        "mapped_timesteps": map_tensor.detach().cpu().tolist(),
        "sqrt_recip_alphas_cumprod": model.diffusion.sqrt_recip_alphas_cumprod.detach().cpu().tolist(),
        "sqrt_recipm1_alphas_cumprod": model.diffusion.sqrt_recipm1_alphas_cumprod.detach().cpu().tolist(),
        "alphas_cumprod_prev": model.diffusion.alphas_cumprod_prev.detach().cpu().tolist(),
    }

    motion_rep = model.motion_rep
    skeleton = motion_rep.skeleton
    root_mean = motion_rep.global_root_stats.mean.detach().float().cpu().tolist()
    root_std_eps = motion_rep.global_root_stats.std_eps.detach().float().cpu().tolist()

    utilities = export_utilities(model, device)
    precisions = {
        "fp32": export_precision(model, "fp32", device),
        "fp16": export_precision(model, "fp16", device),
    }

    manifest = {
        "schema_version": 1,
        "model": "ARDY-Core-RP-20FPS-Horizon40",
        "text_encoder_loaded": False,
        "text_conditioning_enabled": False,
        "constraint_conditioning_enabled": True,
        "fps": float(motion_rep.fps),
        "frames_per_token": int(model.denoiser.num_frames_per_token),
        "history_frames": 4,
        "generation_frames": 40,
        "decode_frames": 44,
        "denoiser_window_frames": 64,
        "denoiser_window_tokens": 16,
        "waypoint_interval_frames": 60,
        "motion_dim": int(motion_rep.motion_rep_dim),
        "hybrid_dim": 148,
        "root_dim": int(motion_rep.motion_root_dim),
        "latent_dim": int(model.denoiser.latent_embedding_dim),
        "cfg": {"text": 0.0, "constraint": 1.5},
        "diffusion": schedule,
        "root_stats": {"mean": root_mean, "std_eps": root_std_eps},
        "root_position_feature_indices": [0, 1, 2],
        "skeleton": {
            "joint_names": list(skeleton.bone_order_names),
            "parents": skeleton.joint_parents.detach().cpu().tolist(),
            "root_index": int(skeleton.root_idx),
        },
        "models": precisions,
        "utilities": utilities,
    }
    manifest_path = DEMO_ROOT / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {manifest_path}", flush=True)
    for precision, entries in precisions.items():
        for name, spec in entries.items():
            print(f"{precision}/{name}: {spec['size_bytes']} bytes, {spec['sha256']}")


if __name__ == "__main__":
    main()
