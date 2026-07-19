#!/usr/bin/env python3
"""Validate the allowlisted, backend-free ARDY GitHub Pages bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path


MAX_GIT_FILE_BYTES = 100 * 1024 * 1024
MAX_PAGES_BYTES = 1024 * 1024 * 1024
MOTION_STATS_CANONICAL_SHA256 = "5c791b8785ec6b431ea04fa574f4a7d5af3b7141df0a309f56a540fc594a197c"
CORE_SKIN_SHA256 = "a4affa8c939feb91f78e460d5273850b816c5e2e0d0368e93e77843c491fc9e9"
EXPECTED_ORT_FILES = (
    "vendor/onnxruntime-web-1.27.0/ort.webgpu.bundle.min.mjs",
    "vendor/onnxruntime-web-1.27.0/ort-wasm-simd-threaded.asyncify.mjs",
    "vendor/onnxruntime-web-1.27.0/ort-wasm-simd-threaded.asyncify.wasm",
)
FORBIDDEN_SITE_TEXT = (
    "10.15.89.180",
    "/mnt/newdisk/",
    "/root/",
    "hf_token",
    "BEGIN OPENSSH PRIVATE KEY",
    "BEGIN RSA PRIVATE KEY",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def check_file(site: Path, record: dict, expected_onnx: set[Path]) -> int:
    path = site / record["url"]
    assert path.is_file(), f"missing asset: {path}"
    size = path.stat().st_size
    assert size == record["size_bytes"], f"size mismatch: {path}"
    assert size < MAX_GIT_FILE_BYTES, f"GitHub 100 MiB file limit exceeded: {path}"
    actual_hash = sha256(path)
    assert actual_hash == record["sha256"], f"sha256 mismatch: {path}"
    expected_onnx.add(path.resolve())
    return size


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("site", type=Path)
    args = parser.parse_args()
    site = args.site.resolve()
    assert site.is_dir(), site
    assert (site / "index.html").is_file()
    assert (site / "infinite_demo.html").is_file()

    manifest = json.loads((site / "infinite_demo/manifest.json").read_text())
    assert manifest["schema_version"] == 2
    assert manifest["all_student"] is True
    assert manifest["precision"] == "fp16"
    assert manifest["nfe"] == 1
    assert manifest["release_status"] == "experimental_pilot"

    expected_onnx: set[Path] = set()
    model_bytes = sum(
        check_file(site, record, expected_onnx)
        for record in manifest["models"].values()
    )
    utility_bytes = sum(
        check_file(site, record, expected_onnx)
        for record in manifest["utilities"].values()
    )
    assert model_bytes == manifest["download"]["learned_graph_bytes"]
    assert utility_bytes == manifest["download"]["utility_graph_bytes"]
    assert model_bytes + utility_bytes == manifest["download"]["total_onnx_bytes"]

    actual_onnx = {path.resolve() for path in site.rglob("*.onnx")}
    assert actual_onnx == expected_onnx, "site contains non-active or missing ONNX files"

    prompt_meta_path = site / manifest["text_conditioning"]["prompt_bundle_url"]
    prompt_meta = json.loads(prompt_meta_path.read_text())
    prompt_data = site / prompt_meta["feature_url"]
    assert prompt_data.stat().st_size == prompt_meta["size_bytes"]
    assert sha256(prompt_data) == prompt_meta["sha256"]
    assert prompt_meta["count"] == len(prompt_meta["entries"])
    assert prompt_meta["feature_dim"] == manifest["text_conditioning"]["feature_dim"]

    motion_stats = json.loads((site / "infinite_demo/motion_stats.json").read_text())
    for name in ("mean", "std_eps"):
        values = motion_stats[name]
        assert len(values) == manifest["motion_dim"], f"bad {name} dimension"
        assert all(math.isfinite(value) for value in values), f"non-finite {name}"
    assert all(value > 0 for value in motion_stats["std_eps"])
    canonical_stats = json.dumps(
        motion_stats,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    assert hashlib.sha256(canonical_stats).hexdigest() == MOTION_STATS_CANONICAL_SHA256

    for relative in EXPECTED_ORT_FILES:
        assert (site / relative).is_file(), f"missing ORT runtime: {relative}"

    core_skin_meta_path = site / "infinite_demo/assets/core_skin.json"
    core_skin_data_path = site / "infinite_demo/assets/core_skin.bin"
    core_skin_meta = json.loads(core_skin_meta_path.read_text())
    assert core_skin_meta["vertex_count"] == 9084
    assert core_skin_meta["triangle_count"] == 18152
    assert core_skin_meta["joint_count"] == 27
    assert core_skin_meta["influences_per_vertex"] == 5
    assert core_skin_data_path.stat().st_size == core_skin_meta["binary_size_bytes"]
    assert core_skin_meta["binary_sha256"] == CORE_SKIN_SHA256
    assert sha256(core_skin_data_path) == CORE_SKIN_SHA256

    demo_js_path = site / "infinite_demo/demo.js"
    demo_js = demo_js_path.read_text()
    assert "demoAssetUrl('motion_stats.json')" in demo_js
    assert "const serverBridgeEnabled = urlOptions.get('bridge') === '1'" in demo_js
    assert "if (serverBridgeEnabled)" in demo_js
    assert "loadCoreSkinRenderer" in demo_js
    assert "./infinite_demo/assets/core_skin.json" in demo_js
    assert (site / "infinite_demo/skin_renderer.js").is_file()
    for page_name in ("index.html", "infinite_demo.html"):
        page = (site / page_name).read_text()
        assert 'id="model-viewport"' in page
        assert 'id="overlay-viewport"' in page
        assert 'id="model-toggle-button"' in page

    for path in site.rglob("*"):
        if not path.is_file():
            continue
        assert path.stat().st_size < MAX_GIT_FILE_BYTES, f"oversized file: {path}"
        if path.suffix.lower() in {".html", ".js", ".mjs", ".json", ".txt", ".md", ".css"}:
            text = path.read_text(errors="replace")
            for forbidden in FORBIDDEN_SITE_TEXT:
                assert forbidden not in text, f"forbidden public text {forbidden!r}: {path}"

    total_bytes = sum(path.stat().st_size for path in site.rglob("*") if path.is_file())
    assert total_bytes < MAX_PAGES_BYTES
    print(
        json.dumps(
            {
                "model_release": manifest["model_release"],
                "onnx_files": len(actual_onnx),
                "onnx_bytes": model_bytes + utility_bytes,
                "site_bytes": total_bytes,
                "largest_file_bytes": max(
                    path.stat().st_size for path in site.rglob("*") if path.is_file()
                ),
                "status": "PUBLIC_BUNDLE_OK",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
