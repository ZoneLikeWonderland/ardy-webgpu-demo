# Third-party notices

- **ARDY**: NVIDIA ARDY source code, Apache-2.0. The upstream repository is
  [`nv-tlabs/ardy`](https://github.com/nv-tlabs/ardy), pinned by this work at
  commit `693f74d13b3d04a0a22ce127ee79c929dd89756b`.
- **ARDY model**: the deployed student is derived from
  `ARDY-Core-RP-20FPS-Horizon40` and is distributed under the NVIDIA Open Model
  License Agreement included in `LICENSES/NVIDIA-OPEN-MODEL-AGREEMENT.txt`.
  The browser bundle also includes the upstream ARDY CoreSkin mesh and skinning
  data used solely to visualize generated motion.
- **ONNX Runtime Web 1.27.0**: Copyright Microsoft Corporation, MIT License.
- **FLUX.2-klein-4B / Qwen text encoder**: used offline to precompute the fixed
  prompt feature bank. No text-encoder weights are included in the web bundle.
- **LLM2Vec**: the upstream ARDY project includes LLM2Vec code under the MIT
  License; its notice is retained in `LICENSE` and `LICENSES/ARDY-APACHE-2.0.txt`.

The full license texts and upstream attribution file are under `LICENSES/`.
