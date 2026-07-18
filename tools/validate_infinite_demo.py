#!/usr/bin/env python
"""Server-side ONNX/PyTorch parity for the all-student infinite demo.

This exercises the same two branches as the browser: an initial 40-frame
window and a continuation window with four retained history frames.  It also
checks the JavaScript-side history preparation, sparse waypoint conversion,
eight-frame inertialization, and the original 40 -> 77 frame buffer update.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch


TOY_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = TOY_ROOT.parent
DEMO_ROOT = TOY_ROOT / "infinite_demo"
CHECKPOINT_ROOT = WORKSPACE / "ardy/checkpoints/ARDY-Core-RP-20FPS-Horizon40"
MANIFEST: dict = {}
PROMPT_BUNDLE = DEMO_ROOT / "prompt_features.json"

if str(WORKSPACE) not in sys.path:
    sys.path.insert(0, str(WORKSPACE))
if str(WORKSPACE / "ardy") not in sys.path:
    sys.path.insert(0, str(WORKSPACE / "ardy"))

from ardy_distill.losses import FSQRequantizer  # noqa: E402
from ardy_distill.models import (  # noqa: E402
    CodecStudentConfig,
    FlowStudentConfig,
    HistoryEncoderStudent,
    MotionDecoderStudent,
    OneStepFlowStudent,
)
from ardy_distill.runtime import load_motion_rep, load_safetensor_weights  # noqa: E402
from ardy_distill.student_runtime import StudentArdyRuntime  # noqa: E402
from ardy_distill.tools.evaluate_rollout import inertialize_generated_motion  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEMO_ROOT / "manifest.json")
    parser.add_argument("--output", type=Path, default=DEMO_ROOT / "server_validation.json")
    parser.add_argument(
        "--browser-case",
        type=Path,
        default=DEMO_ROOT / "browser_validation_case.json",
        help="Deterministic CPU-ORT golden case consumed by the live WebGPU demo.",
    )
    return parser.parse_args()


def learned_numpy_dtype() -> type[np.float16] | type[np.float32]:
    return np.float16 if MANIFEST["precision"] == "fp16" else np.float32


def load_browser_conditions() -> tuple[np.ndarray, np.ndarray, dict]:
    """Load the exact FP16 prompt row and no-keyboard-heading condition."""

    if not MANIFEST.get("text_conditioning_enabled"):
        return (
            np.zeros((1, 0), dtype=np.float32),
            np.zeros((1, MANIFEST["path_frames"], 0), dtype=np.float32),
            {"prompt_id": None, "text": None},
        )
    metadata = json.loads(PROMPT_BUNDLE.read_text(encoding="utf-8"))
    feature_dim = int(MANIFEST["text_conditioning"]["feature_dim"])
    if (
        metadata.get("schema") != "ardy_browser_prompt_features_v1"
        or metadata.get("storage_dtype") != "float16_le"
        or metadata.get("compression") != "none"
        or int(metadata.get("feature_dim", -1)) != feature_dim
    ):
        raise ValueError("browser prompt bundle does not match manifest")
    binary_path = TOY_ROOT / metadata["feature_url"]
    feature_table = np.fromfile(binary_path, dtype=np.dtype("<f2"))
    expected_shape = (int(metadata["count"]), feature_dim)
    feature_table = feature_table.reshape(expected_shape)
    default_prompt_id = int(metadata["default_prompt_id"])
    index = next(
        int(entry["index"])
        for entry in metadata["entries"]
        if int(entry["prompt_id"]) == default_prompt_id
    )
    entry = metadata["entries"][index]
    text_feature = feature_table[index : index + 1].astype(np.float32)
    heading_features = int(
        MANIFEST["model_config"]["flow"]["heading_condition_features"]
    )
    heading_condition = np.zeros(
        (1, MANIFEST["path_frames"], heading_features),
        dtype=np.float32,
    )
    return text_feature, heading_condition, entry


def session(spec: dict) -> ort.InferenceSession:
    return ort.InferenceSession(
        str(TOY_ROOT / spec["url"]),
        providers=["CPUExecutionProvider"],
    )


def metrics(actual: np.ndarray, reference: np.ndarray) -> dict[str, float | bool]:
    actual64 = actual.astype(np.float64)
    reference64 = reference.astype(np.float64)
    delta = np.abs(actual64 - reference64)
    denom = np.linalg.norm(actual64.ravel()) * np.linalg.norm(reference64.ravel())
    cosine = float(np.dot(actual64.ravel(), reference64.ravel()) / denom) if denom else 1.0
    return {
        "all_finite": bool(np.isfinite(actual).all()),
        "max_abs_error": float(delta.max(initial=0.0)),
        "mean_abs_error": float(delta.mean()) if delta.size else 0.0,
        "p95_abs_error": float(np.quantile(delta, 0.95)) if delta.size else 0.0,
        "p99_abs_error": float(np.quantile(delta, 0.99)) if delta.size else 0.0,
        "fraction_abs_error_gt_0_01": float((delta > 0.01).mean()) if delta.size else 0.0,
        "cosine_similarity": cosine,
    }


def round_ties_to_even(values: np.ndarray) -> np.ndarray:
    # np.rint has the same ties-to-even rule as torch.round and WGSL's round.
    return np.rint(values)


def requantize_numpy(values: np.ndarray) -> np.ndarray:
    stats = MANIFEST["post_quantization_stats"]
    mean = np.asarray(stats["mean"], dtype=np.float32)
    std = np.asarray(stats["std_eps"], dtype=np.float32)
    half_width = stats["levels"] // 2
    raw = np.clip(values * std + mean, -1.0, 1.0)
    quantized = round_ties_to_even(raw * half_width) / half_width
    return ((quantized - mean) / std).astype(np.float32)


def fsq_bin_diagnostics_numpy(
    values: np.ndarray,
    reference: np.ndarray,
) -> dict[str, float]:
    stats = MANIFEST["post_quantization_stats"]
    mean = np.asarray(stats["mean"], dtype=np.float32)
    std = np.asarray(stats["std_eps"], dtype=np.float32)
    half_width = stats["levels"] // 2
    scaled = np.clip(values[..., 20:] * std + mean, -1.0, 1.0) * half_width
    reference_scaled = (
        np.clip(reference[..., 20:] * std + mean, -1.0, 1.0) * half_width
    )
    bins = np.rint(scaled)
    reference_bins = np.rint(reference_scaled)
    fractional_distance = np.abs(scaled - bins)
    return {
        "bin_match_fraction": float((bins == reference_bins).mean()),
        "near_boundary_fraction": float((fractional_distance >= 0.45).mean()),
        "distance_to_nearest_center_mean_bins": float(fractional_distance.mean()),
        "distance_to_nearest_center_p95_bins": float(
            np.quantile(fractional_distance, 0.95)
        ),
    }


def prepare_history_numpy(
    history: np.ndarray,
    encoder: ort.InferenceSession,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Mirror the browser's compact encoder and released recentering math."""

    encoded_raw = encoder.run(
        ["output"],
        {"normalized_body": history[..., 5:].astype(learned_numpy_dtype())},
    )[0]
    encoded = requantize_numpy(encoded_raw)
    root_mean = np.asarray(MANIFEST["root_stats"]["mean"], dtype=np.float32)
    root_std = np.asarray(MANIFEST["root_stats"]["std_eps"], dtype=np.float32)
    root_world = history[..., :5].astype(np.float32) * root_std + root_mean
    translation = root_world[:, 3, :3].copy()
    translation[:, 1] = 0
    local_world = root_world.copy()
    local_world[..., 0] -= translation[:, None, 0]
    local_world[..., 2] -= translation[:, None, 2]
    local_root = (local_world - root_mean) / root_std
    angle = np.arctan2(root_world[:, 0, 4], root_world[:, 0, 3])
    first_heading = np.stack([np.cos(angle), np.sin(angle)], axis=-1).astype(np.float32)
    history_hybrid = np.concatenate([local_root.reshape(-1, 1, 20), encoded], axis=-1).astype(np.float32)
    return history_hybrid, translation.astype(np.float32), first_heading, encoded_raw


def build_sparse_path(
    *,
    history_start: int,
    history_end: int,
    translation: np.ndarray,
    waypoint_frame: int,
    target_xz: tuple[float, float],
) -> np.ndarray:
    path = np.zeros((1, MANIFEST["path_frames"], 3), dtype=np.float32)
    relative = waypoint_frame - history_start
    if waypoint_frame <= history_end or relative < 0 or relative >= MANIFEST["path_frames"]:
        return path
    mean = MANIFEST["root_stats"]["mean"]
    std = MANIFEST["root_stats"]["std_eps"]
    path[0, relative, 0] = ((target_xz[0] - translation[0, 0]) - mean[0]) / std[0]
    path[0, relative, 1] = ((target_xz[1] - translation[0, 2]) - mean[2]) / std[2]
    path[0, relative, 2] = 1
    return path


def inertialize_numpy(
    explicit_motion: np.ndarray,
    history: np.ndarray | None,
    history_length: int,
) -> np.ndarray:
    config = MANIFEST["inertialization"]
    if history is None or history.shape[1] < 2 or config["frames"] <= 0:
        return explicit_motion.copy()
    result = explicit_motion.copy()
    count = min(config["frames"], result.shape[1] - history_length)
    feature_end = result.shape[-1] - config["excluded_tail_features"]
    constant_velocity_first = 2.0 * history[:, -1] - history[:, -2]
    offset = constant_velocity_first[:, :feature_end] - result[:, history_length, :feature_end]
    if count == 1:
        decay = np.ones(1, dtype=np.float32)
    else:
        phase = np.arange(count, dtype=np.float32) / np.float32(count - 1)
        decay = 1.0 - (3.0 * phase**2 - 2.0 * phase**3)
    result[:, history_length : history_length + count, :feature_end] += (
        np.float32(config["strength"]) * decay[None, :, None] * offset[:, None, :]
    )
    return result


def ort_step(
    bundle: dict[str, ort.InferenceSession],
    *,
    noise: np.ndarray,
    path_condition: np.ndarray,
    text_feature: np.ndarray,
    heading_condition: np.ndarray,
    history: np.ndarray | None,
) -> dict[str, np.ndarray | dict[str, float]]:
    timings: dict[str, float] = {}
    if history is None:
        history_hybrid = np.zeros((1, 1, 148), dtype=np.float32)
        translation = np.zeros((1, 3), dtype=np.float32)
        first_heading = np.asarray([[1.0, 0.0]], dtype=np.float32)
        has_history = np.zeros((1, 1), dtype=np.float32)
    else:
        start = time.perf_counter()
        history_hybrid, translation, first_heading, encoded_raw = prepare_history_numpy(
            history,
            bundle["encoder"],
        )
        timings["encoder_ms"] = (time.perf_counter() - start) * 1000.0
        has_history = np.ones((1, 1), dtype=np.float32)

    start = time.perf_counter()
    flow_feeds = {
        "noise": noise.astype(learned_numpy_dtype()),
        "history_hybrid": history_hybrid.astype(learned_numpy_dtype()),
        "path_condition": path_condition.astype(learned_numpy_dtype()),
        "first_heading": first_heading.astype(learned_numpy_dtype()),
        "has_history": has_history.astype(learned_numpy_dtype()),
    }
    if MANIFEST.get("text_conditioning_enabled"):
        flow_feeds["text_feature"] = text_feature.astype(learned_numpy_dtype())
    if MANIFEST.get("heading_conditioning_enabled"):
        flow_feeds["heading_condition"] = heading_condition.astype(learned_numpy_dtype())
    clean = bundle["flow"].run(["output"], flow_feeds)[0]
    timings["flow_ms"] = (time.perf_counter() - start) * 1000.0

    if history is None:
        finalizer = bundle["finalize_initial"]
        finalize_feeds = {
            "clean_generation": clean.astype(np.float32),
            "global_translation": translation,
        }
        history_length = 0
    else:
        finalizer = bundle["finalize_continuation"]
        finalize_feeds = {
            "local_hybrid": np.concatenate(
                [history_hybrid, clean.astype(np.float32)],
                axis=1,
            ),
            "global_translation": translation,
        }
        history_length = 4
    start = time.perf_counter()
    global_root, decoder_latent, decoder_local_root, token_valid = finalizer.run(None, finalize_feeds)
    timings["finalize_ms"] = (time.perf_counter() - start) * 1000.0
    start = time.perf_counter()
    body = bundle["decoder"].run(
        ["output"],
        {
            "latent_tokens": decoder_latent.astype(learned_numpy_dtype()),
            "local_root": decoder_local_root.astype(learned_numpy_dtype()),
            "token_valid": token_valid.astype(learned_numpy_dtype()),
        },
    )[0]
    timings["decoder_ms"] = (time.perf_counter() - start) * 1000.0

    if history is None:
        explicit = np.concatenate([global_root, body[:, 4:]], axis=-1)
    else:
        explicit = np.concatenate([global_root, body], axis=-1)
    inertialized = inertialize_numpy(explicit, history, history_length)
    segment = inertialized[:, history_length:]
    return {
        "history_hybrid": history_hybrid,
        "translation": translation,
        "first_heading": first_heading,
        "clean": clean,
        "global_root": global_root,
        "decoder_latent": decoder_latent,
        "decoder_local_root": decoder_local_root,
        "token_valid": token_valid,
        "body": body,
        "explicit": explicit,
        "inertialized": inertialized,
        "segment": segment,
        "timings_ms": timings,
    }


def build_runtime(model_dtype: torch.dtype) -> StudentArdyRuntime:
    codec_config = CodecStudentConfig(**MANIFEST["model_config"]["codec"])
    flow_config = FlowStudentConfig(**MANIFEST["model_config"]["flow"])
    encoder = HistoryEncoderStudent(codec_config).eval()
    flow = OneStepFlowStudent(flow_config).eval()
    decoder = MotionDecoderStudent(codec_config).eval()
    load_safetensor_weights(encoder, WORKSPACE / MANIFEST["weights"]["encoder"])
    load_safetensor_weights(flow, WORKSPACE / MANIFEST["weights"]["flow"])
    load_safetensor_weights(decoder, WORKSPACE / MANIFEST["weights"]["decoder"])
    quantizer = FSQRequantizer(CHECKPOINT_ROOT / "stats/post_quantization").eval()
    encoder.to(dtype=model_dtype)
    flow.to(dtype=model_dtype)
    decoder.to(dtype=model_dtype)
    return StudentArdyRuntime(
        encoder,
        flow,
        decoder,
        quantizer,
        load_motion_rep(CHECKPOINT_ROOT),
        model_dtype=model_dtype,
    ).eval()


def main() -> None:
    global MANIFEST
    args = parse_args()
    MANIFEST = json.loads(args.manifest.read_text(encoding="utf-8"))
    if MANIFEST.get("schema_version") != 2 or not MANIFEST.get("all_student"):
        raise RuntimeError("manifest is not the all-student schema")
    if MANIFEST.get("precision") not in {"fp16", "fp32"}:
        raise RuntimeError(f"unsupported manifest precision: {MANIFEST.get('precision')}")
    bundle = {
        "encoder": session(MANIFEST["models"]["encoder"]),
        "flow": session(MANIFEST["models"]["flow"]),
        "decoder": session(MANIFEST["models"]["decoder"]),
        "finalize_initial": session(MANIFEST["utilities"]["finalize_initial"]),
        "finalize_continuation": session(MANIFEST["utilities"]["finalize_continuation"]),
    }
    deployment_dtype = torch.float16 if MANIFEST["precision"] == "fp16" else torch.float32
    runtime = build_runtime(deployment_dtype)
    fp32_runtime = build_runtime(torch.float32)
    rng = np.random.default_rng(20260714)
    text_feature, heading_condition, prompt_entry = load_browser_conditions()

    noise1 = rng.standard_normal((1, 10, 148), dtype=np.float32)
    zero_translation = np.zeros((1, 3), dtype=np.float32)
    path1 = build_sparse_path(
        history_start=0,
        history_end=-1,
        translation=zero_translation,
        waypoint_frame=60,
        target_xz=(2.0, 3.0),
    )
    initial_ort = ort_step(
        bundle,
        noise=noise1,
        path_condition=path1,
        text_feature=text_feature,
        heading_condition=heading_condition,
        history=None,
    )
    initial_torch = runtime.step_prepared(
        path_condition=torch.from_numpy(path1),
        text_feature=torch.from_numpy(text_feature),
        heading_condition=torch.from_numpy(heading_condition),
        initial_noise=torch.from_numpy(noise1),
        init_global_translation=torch.zeros(1, 3),
        init_first_heading_angle=torch.zeros(1),
    )
    initial_torch_fp32 = fp32_runtime.step_prepared(
        path_condition=torch.from_numpy(path1),
        text_feature=torch.from_numpy(text_feature),
        heading_condition=torch.from_numpy(heading_condition),
        initial_noise=torch.from_numpy(noise1),
        init_global_translation=torch.zeros(1, 3),
        init_first_heading_angle=torch.zeros(1),
    )

    history_start = 33
    history_end = 36
    history = np.asarray(initial_ort["explicit"][:, history_start : history_end + 1], dtype=np.float32)
    prepared_history, translation, _, _ = prepare_history_numpy(history, bundle["encoder"])
    path2 = build_sparse_path(
        history_start=history_start,
        history_end=history_end,
        translation=translation,
        waypoint_frame=95,
        target_xz=(-1.5, 4.0),
    )
    noise2 = rng.standard_normal((1, 10, 148), dtype=np.float32)
    continuation_ort = ort_step(
        bundle,
        noise=noise2,
        path_condition=path2,
        text_feature=text_feature,
        heading_condition=heading_condition,
        history=history,
    )
    continuation_torch = runtime.step_prepared(
        path_condition=torch.from_numpy(path2),
        text_feature=torch.from_numpy(text_feature),
        heading_condition=torch.from_numpy(heading_condition),
        initial_noise=torch.from_numpy(noise2),
        init_history_sequence=torch.from_numpy(history),
    )
    continuation_torch_fp32 = fp32_runtime.step_prepared(
        path_condition=torch.from_numpy(path2),
        text_feature=torch.from_numpy(text_feature),
        heading_condition=torch.from_numpy(heading_condition),
        initial_noise=torch.from_numpy(noise2),
        init_history_sequence=torch.from_numpy(history),
    )
    torch_inertialized = inertialize_generated_motion(
        continuation_torch.explicit_motion,
        torch.from_numpy(history),
        4,
        frames=MANIFEST["inertialization"]["frames"],
        strength=MANIFEST["inertialization"]["strength"],
    ).detach().numpy()

    comparisons = {
        "initial_clean": metrics(initial_ort["clean"], initial_torch.clean_generation.detach().numpy()),
        "initial_clean_vs_pytorch_fp32": metrics(
            initial_ort["clean"],
            initial_torch_fp32.clean_generation.detach().numpy(),
        ),
        "initial_global_root": metrics(
            initial_ort["global_root"],
            initial_torch.decoder_global_root[:, 4:].detach().numpy(),
        ),
        "initial_decoder_latent": metrics(
            initial_ort["decoder_latent"],
            initial_torch.decoder_latent.detach().numpy(),
        ),
        "initial_decoder_local_root": metrics(
            initial_ort["decoder_local_root"],
            initial_torch.decoder_local_root.detach().numpy(),
        ),
        "initial_decoder_body": metrics(
            initial_ort["body"][:, 4:],
            initial_torch.explicit_motion[..., MANIFEST["root_dim"] :].detach().numpy(),
        ),
        "initial_explicit": metrics(initial_ort["explicit"], initial_torch.explicit_motion.detach().numpy()),
        "initial_explicit_vs_pytorch_fp32": metrics(
            initial_ort["explicit"],
            initial_torch_fp32.explicit_motion.detach().numpy(),
        ),
        "continuation_history_hybrid": metrics(
            prepared_history,
            continuation_torch.history_hybrid.detach().numpy(),
        ),
        "continuation_clean": metrics(
            continuation_ort["clean"],
            continuation_torch.clean_generation.detach().numpy(),
        ),
        "continuation_clean_vs_pytorch_fp32": metrics(
            continuation_ort["clean"],
            continuation_torch_fp32.clean_generation.detach().numpy(),
        ),
        "continuation_global_root": metrics(
            continuation_ort["global_root"],
            continuation_torch.decoder_global_root.detach().numpy(),
        ),
        "continuation_decoder_latent": metrics(
            continuation_ort["decoder_latent"],
            continuation_torch.decoder_latent.detach().numpy(),
        ),
        "continuation_decoder_local_root": metrics(
            continuation_ort["decoder_local_root"],
            continuation_torch.decoder_local_root.detach().numpy(),
        ),
        "continuation_decoder_body": metrics(
            continuation_ort["body"],
            continuation_torch.explicit_motion[..., MANIFEST["root_dim"] :].detach().numpy(),
        ),
        "continuation_explicit": metrics(
            continuation_ort["explicit"],
            continuation_torch.explicit_motion.detach().numpy(),
        ),
        "continuation_explicit_vs_pytorch_fp32": metrics(
            continuation_ort["explicit"],
            continuation_torch_fp32.explicit_motion.detach().numpy(),
        ),
        "continuation_inertialized": metrics(
            continuation_ort["inertialized"],
            torch_inertialized,
        ),
    }
    max_abs = max(float(row["max_abs_error"]) for row in comparisons.values())
    all_finite = all(bool(row["all_finite"]) for row in comparisons.values())
    initial_frames = int(initial_ort["segment"].shape[1])
    continuation_frames = int(continuation_ort["segment"].shape[1])
    buffer = np.concatenate(
        [initial_ort["segment"][:, : history_end + 1], continuation_ort["segment"]],
        axis=1,
    )
    download_bytes = int(MANIFEST["download"]["total_onnx_bytes"])
    fsq_diagnostics = {
        "initial_vs_pytorch_same_precision": fsq_bin_diagnostics_numpy(
            initial_ort["clean"],
            initial_torch.clean_generation.detach().numpy(),
        ),
        "initial_vs_pytorch_fp32": fsq_bin_diagnostics_numpy(
            initial_ort["clean"],
            initial_torch_fp32.clean_generation.detach().numpy(),
        ),
        "continuation_vs_pytorch_same_precision": fsq_bin_diagnostics_numpy(
            continuation_ort["clean"],
            continuation_torch.clean_generation.detach().numpy(),
        ),
        "continuation_vs_pytorch_fp32": fsq_bin_diagnostics_numpy(
            continuation_ort["clean"],
            continuation_torch_fp32.clean_generation.detach().numpy(),
        ),
    }
    if MANIFEST["precision"] == "fp16":
        thresholds = {
            "same_precision_endpoint_max_abs": 2.5e-2,
            "fp32_endpoint_max_abs": 2.0e-2,
            "fp32_explicit_mean_abs": 1.5e-2,
            "fp32_explicit_p99_abs": 7.5e-2,
            "fp32_explicit_cosine": 0.9999,
            "fsq_bin_match_fraction": 0.94,
        }
        browser_thresholds = {
            "endpoint_max_abs": 5.0e-2,
            "explicit_mean_abs": 3.0e-2,
            "explicit_p99_abs": 1.5e-1,
            "explicit_cosine": 0.999,
            "fsq_bin_match_fraction": 0.90,
            "history_mean_abs": 2.0e-2,
            "history_cosine": 0.999,
        }
    else:
        thresholds = {
            "same_precision_endpoint_max_abs": 5.0e-4,
            "fp32_endpoint_max_abs": 5.0e-4,
            "fp32_explicit_mean_abs": 5.0e-4,
            "fp32_explicit_p99_abs": 1.0e-3,
            "fp32_explicit_cosine": 0.999999,
            "fsq_bin_match_fraction": 0.995,
        }
        browser_thresholds = {
            "endpoint_max_abs": 1.0e-3,
            "explicit_mean_abs": 2.0e-3,
            "explicit_p99_abs": 1.0e-2,
            "explicit_cosine": 0.99999,
            "fsq_bin_match_fraction": 0.995,
            "history_mean_abs": 2.0e-3,
            "history_cosine": 0.99999,
        }
    observed = {
        "same_precision_endpoint_max_abs": max(
            float(comparisons[name]["max_abs_error"])
            for name in ("initial_clean", "continuation_clean")
        ),
        "fp32_endpoint_max_abs": max(
            float(comparisons[name]["max_abs_error"])
            for name in (
                "initial_clean_vs_pytorch_fp32",
                "continuation_clean_vs_pytorch_fp32",
            )
        ),
        "fp32_explicit_mean_abs": max(
            float(comparisons[name]["mean_abs_error"])
            for name in (
                "initial_explicit_vs_pytorch_fp32",
                "continuation_explicit_vs_pytorch_fp32",
            )
        ),
        "fp32_explicit_p99_abs": max(
            float(comparisons[name]["p99_abs_error"])
            for name in (
                "initial_explicit_vs_pytorch_fp32",
                "continuation_explicit_vs_pytorch_fp32",
            )
        ),
        "fp32_explicit_cosine": min(
            float(comparisons[name]["cosine_similarity"])
            for name in (
                "initial_explicit_vs_pytorch_fp32",
                "continuation_explicit_vs_pytorch_fp32",
            )
        ),
        "fsq_bin_match_fraction": min(
            float(row["bin_match_fraction"])
            for row in fsq_diagnostics.values()
        ),
    }
    numeric_passed = bool(
        observed["same_precision_endpoint_max_abs"]
        <= thresholds["same_precision_endpoint_max_abs"]
        and observed["fp32_endpoint_max_abs"] <= thresholds["fp32_endpoint_max_abs"]
        and observed["fp32_explicit_mean_abs"] <= thresholds["fp32_explicit_mean_abs"]
        and observed["fp32_explicit_p99_abs"] <= thresholds["fp32_explicit_p99_abs"]
        and observed["fp32_explicit_cosine"] >= thresholds["fp32_explicit_cosine"]
        and observed["fsq_bin_match_fraction"] >= thresholds["fsq_bin_match_fraction"]
    )
    passed = bool(
        all_finite
        and numeric_passed
        and initial_frames == 40
        and continuation_frames == 40
        and buffer.shape[1] == 77
        and MANIFEST["nfe"] == 1
        and download_bytes < int(MANIFEST["download"]["budget_bytes"])
    )
    result = {
        "passed": passed,
        "provider": "CPUExecutionProvider",
        "precision": MANIFEST["precision"],
        "nfe": MANIFEST["nfe"],
        "comparisons": comparisons,
        "fsq_endpoint_diagnostics": fsq_diagnostics,
        "numeric_acceptance": {
            "passed": numeric_passed,
            "observed": observed,
            "thresholds": thresholds,
            "note": (
                "Continuous flow endpoints retain max-error gates. After discontinuous "
                "FSQ rounding, acceptance uses bin agreement plus mean/p99/cosine while "
                "the worst elementwise error remains reported separately."
            ),
        },
        "max_abs_error_across_pipeline": max_abs,
        "shapes": {
            "initial_explicit": list(initial_ort["explicit"].shape),
            "continuation_explicit": list(continuation_ort["explicit"].shape),
            "browser_initial_segment": list(initial_ort["segment"].shape),
            "browser_continuation_segment": list(continuation_ort["segment"].shape),
            "buffer_after_second_window": list(buffer.shape),
        },
        "sparse_path_indices": {
            "initial": np.flatnonzero(path1[0, :, 2]).tolist(),
            "continuation": np.flatnonzero(path2[0, :, 2]).tolist(),
        },
        "timings_ms_cpu_single_run": {
            "initial": initial_ort["timings_ms"],
            "continuation": continuation_ort["timings_ms"],
        },
        "download": {
            **MANIFEST["download"],
            "total_mib": download_bytes / 1048576.0,
            "under_configured_budget": (
                download_bytes < int(MANIFEST["download"]["budget_bytes"])
            ),
        },
        "conditioning": {
            "prompt_id": prompt_entry.get("prompt_id"),
            "prompt_text": prompt_entry.get("text"),
            "text_feature_shape": list(text_feature.shape),
            "heading_condition_shape": list(heading_condition.shape),
            "heading_valid_count": int((heading_condition[..., -1] > 0).sum()),
        },
        "tolerance": thresholds,
    }
    browser_case = {
        "schema": "ardy_webgpu_e2e_golden_v1",
        "model_release": MANIFEST["model_release"],
        "precision": MANIFEST["precision"],
        "nfe": MANIFEST["nfe"],
        "thresholds": browser_thresholds,
        "conditions": {
            "prompt_id": prompt_entry.get("prompt_id"),
            "prompt_text": prompt_entry.get("text"),
            "text_feature": text_feature.reshape(-1).tolist(),
            "heading_condition": heading_condition.reshape(-1).tolist(),
        },
        "initial": {
            "noise": noise1.reshape(-1).tolist(),
            "path_condition": path1.reshape(-1).tolist(),
            "expected": {
                "clean": np.asarray(initial_ort["clean"], dtype=np.float32).reshape(-1).tolist(),
                "motion": np.asarray(initial_ort["segment"], dtype=np.float32).reshape(-1).tolist(),
            },
        },
        "continuation": {
            "history": history.reshape(-1).tolist(),
            "noise": noise2.reshape(-1).tolist(),
            "path_condition": path2.reshape(-1).tolist(),
            "expected": {
                "history_hybrid": prepared_history.reshape(-1).tolist(),
                "clean": np.asarray(continuation_ort["clean"], dtype=np.float32).reshape(-1).tolist(),
                "motion": np.asarray(continuation_ort["segment"], dtype=np.float32).reshape(-1).tolist(),
            },
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    args.browser_case.parent.mkdir(parents=True, exist_ok=True)
    args.browser_case.write_text(
        json.dumps(browser_case, ensure_ascii=False, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(json.dumps({"browser_case": str(args.browser_case)}, ensure_ascii=False))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
