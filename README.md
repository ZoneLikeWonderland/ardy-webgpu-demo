# ARDY WebGPU Student

[Open the live GitHub Pages demo](https://zonelikewonderland.github.io/ardy-webgpu-demo/)

This repository is a research snapshot of a compact, browser-native student of
[NVIDIA ARDY](https://github.com/nv-tlabs/ardy). The demo performs infinite
autoregressive human-motion generation entirely in the visitor's browser with
ONNX Runtime WebGPU. There is no Python inference service behind the public
page and no motion or prompt data is uploaded.

> **Status:** experimental pilot, not a final quality release. The active model
> is intentionally the currently selected `u05legacy_g400_ema` checkpoint; the
> repository does not claim that every numerical or motion-quality gate passes.

## Try it

Open the link above in a recent hardware-accelerated Chrome or Edge browser.
The page requires a secure HTTPS context, WebGPU, and `shader-f16` support.
The first visit downloads approximately 123 MiB of static runtime/model assets
and may spend additional time compiling WebGPU pipelines.

The public demo provides:

- NFE=1 FP16 autoregressive generation at a 20 FPS motion timeline;
- start, pause, restart, and continuous 40-frame continuation;
- mouse-selected sparse root waypoints;
- a visible frame/time timeline and waypoint markers;
- 33 precomputed Qwen prompt features;
- raw, light, balanced, and strong display-only postprocessing modes.

The public page is static. Arbitrary new text is not encoded in the browser;
the selector uses an offline-precomputed prompt bank. The Llama teacher text
encoder and the Qwen student text encoder are not part of the deployed motion
network and are not downloaded at runtime.

## Deployed network

| Module | Structure | Parameters | Browser input | Browser output |
| --- | --- | ---: | --- | --- |
| History encoder | width 512, 4 residual MLP blocks | 4,937,344 | `[B,4,325]` normalized body | `[B,1,128]` latent history |
| Flow model | width 512, 8 heads, 8 trunk + 8 body blocks | 37,849,236 | noise/history/path/heading/text | `[B,10,148]` clean generation |
| Motion decoder | width 512, 8 token/channel mixer blocks | 9,165,420 | `[B,11,128]` latent + local root | `[B,44,325]` body motion |
| Finalizers | fixed tensor/root/FSQ layout graphs | 0 | initial or continuation tensors | decoder-ready tensors + global root |

The learned motion network has 51,952,000 parameters by the manifest component
sum; the five deployed ONNX graphs total 99.29 MiB. The authoritative exact
counts, tensor names, operators, hashes, and export errors are recorded in
[`site/infinite_demo/manifest.json`](site/infinite_demo/manifest.json).

## Repository map

- `site/`: the exact GitHub Pages bundle and active ONNX student.
- `ardy_distill/`: compact codec, flow, DMD2/LADD, data, EMA, evaluation, and
  WebGPU-numerics research code.
- `tools/`: dataset, prompt, export, parity, profiling, and sweep-summary tools.
- `recipes/`: the recorded multi-GPU training and evaluation launchers. These
  are experiment recipes and require local datasets/checkpoints.
- `docs/PROJECT_FINDINGS.md`: the sanitized stage-by-stage technical log.
- `docs/DISTILLATION_PLAN.md`: network design, parameter budget, profiling, and
  distillation plan.
- `docs/experiments/`: selected machine-readable sweep summaries only; no
  checkpoints, optimizer states, raw datasets, or private credentials.
- `dev/compare_server.py`: optional local PyTorch/NumPy comparison bridge; it is
  not used by GitHub Pages.

The upstream ARDY checkout is not duplicated. This snapshot was developed
against `nv-tlabs/ardy@693f74d13b3d04a0a22ce127ee79c929dd89756b`.

## Validate the static bundle

No package installation is required for the public page or bundle verifier:

```bash
python3 tools/verify_public_bundle.py site
python3 -m http.server 8768 --bind 127.0.0.1 --directory site
```

The Pages workflow runs the same integrity checks and the frontend
postprocessing tests before deployment. Browser inference still needs to be
validated on a real WebGPU-capable client.

## Licenses

Software and model assets have separate terms. See `LICENSE`, `NOTICE`,
`THIRD_PARTY_NOTICES.md`, and `LICENSES/`. In particular, the distilled ARDY
student is accompanied by the NVIDIA Open Model License Agreement and its
required attribution notice.
