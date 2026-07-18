# GitHub Pages deployment

The public demo is a static application. HTML, JavaScript, precomputed text
features, ONNX Runtime Web, and five ONNX graphs are all served from the same
HTTPS origin; inference runs on the visitor's WebGPU adapter.

The Pages build uses an explicit `site/` allowlist. It does not upload local
training data, checkpoints other than the active ONNX release, raw evaluations,
credentials, TLS material, or the local Python comparison server.

The public frontend differs from the local comparison frontend in two ways:

1. motion normalization statistics are loaded from
   `infinite_demo/motion_stats.json` instead of `/api/demo/motion-stats`;
2. telemetry and remote command polling are disabled unless the page is opened
   with the explicit development-only `?bridge=1` option.

The runtime fixes ONNX Runtime WASM to one thread, so the demo does not require
`SharedArrayBuffer` or cross-origin isolation headers that GitHub Pages cannot
customize. WebGPU still requires HTTPS, a supported browser/GPU, hardware
acceleration, and `shader-f16` for this release.

The GitHub Actions workflow verifies every model size and SHA-256 from the
manifest, rejects extra ONNX files and workstation-only strings in `site/`, runs
the display-postprocessing tests, uploads the static artifact, and deploys it
through the official Pages actions.

