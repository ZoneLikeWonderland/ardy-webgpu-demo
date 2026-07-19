# GitHub Pages deployment

The public demo is a static application. HTML, JavaScript, precomputed text
features, ONNX Runtime Web, and five ONNX graphs are all served from the same
HTTPS origin; inference runs on the visitor's WebGPU adapter.

The visualization also includes ARDY's official CoreSkin asset (9,084
vertices, 18,152 triangles, 27 joints, five LBS influences per vertex). It is
rendered by a dependency-free WebGL2 layer above the grid and below the
trajectory/skeleton overlay. The character surface is opaque; hiding it does
not change ONNX state, playback, waypoints, or display postprocessing.

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
manifest, verifies the CoreSkin bundle and five-weight LBS regression, rejects
extra ONNX files and workstation-only strings in `site/`, runs the
display-postprocessing tests, uploads the static artifact, and deploys it
through the official Pages actions.

## Repository visibility

GitHub Free supports Pages only from public repositories. A personal private
repository requires GitHub Pro (or an applicable organization Team/Enterprise
plan); changing a Free repository from public to private automatically
unpublishes its Pages site. A private source repository does not by itself make
the site private: Pages remains public on the internet by default. Authenticated
private Pages access is limited to eligible project sites owned by a GitHub
Enterprise Cloud organization. See GitHub's official documentation for
[Pages plan support](https://docs.github.com/en/pages/getting-started-with-github-pages/what-is-github-pages),
[repository visibility effects](https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/managing-repository-settings/setting-repository-visibility),
and [Pages access control](https://docs.github.com/en/enterprise-cloud@latest/pages/getting-started-with-github-pages/changing-the-visibility-of-your-github-pages-site).
