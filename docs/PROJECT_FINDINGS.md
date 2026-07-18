# ARDY 阶段性项目结论

更新日期：2026-07-13

## 1. 当前结论

1. `ARDY-Core-RP-20FPS-Horizon40` 已在本机现有 `py311` 环境中成功加载并运行交互 demo。没有创建或切换 Conda 环境，也没有执行任何 `conda` 命令。
2. ARDY Core 本体是约 1.91 亿参数的两阶段 Transformer 扩散动作模型；Llama-3-8B/LLM2Vec 只负责把文本提示变成 4096 维条件，不参与数值路径点约束的解析。
3. 仅使用路径/朝向约束时，算法上不需要 Llama。可以给 denoiser 输入全零的 `[B, 1, 4096]` 文本特征并令 text CFG 权重为 0；当前 demo 仍无条件加载文本编码器，需要增加一个明确的 no-text 模式。
4. GPU 占用看起来低不是因为模型没有在 GPU 上运行，而是生成速度远快于播放速度：A6000 上 Core40 一次生成 2 秒动作约需 0.113 秒，纯核心吞吐约为实时的 17.7 倍；交互 demo 实际常见 0.17～0.21 秒生成约 2 秒动作，计算 duty cycle 只有约 9%～11%。`nvidia-smi` 的 1 秒采样很容易看到 0%。
5. WebGPU 适配是可行的，最合理的首个目标是“无文本、固定 Core40、仅路径/姿态约束”的浏览器版本。完整 LLM2Vec/Llama-3-8B 浏览器化不是首选：它不是 ARDY 核心动作网络的必需部分，且会显著增加下载、显存、导出和兼容工作。
6. 已建立一个零第三方依赖的原生 WebGPU toy，服务器端口为 `8766`。它验证浏览器安全上下文、adapter、`shader-f16`、buffer limits、WGSL 编译、矩阵乘法和数值结果。
7. 浏览器 WebGPU toy 已由用户实际验证通过。后续 WebGPU 对拍明确不包含 Llama/LLM2Vec，只处理 ARDY Core 动作模型及数值路径/heading 约束。
8. 已完成精简版 ARDY WebGPU 无限生成前端并在 Edge 149 / NVIDIA Ampere 上实际运行。它不是 40 帧循环播放，而是执行 4 帧历史重编码、40 帧续写、replan buffer 1、剩余 4 帧自动重规划和当前帧后 60 帧的稀疏 waypoint；生成期间播放头冻结，生成完成后从原帧连续播放。

## 2. 本地项目与权重

### 2.1 仓库

- 项目目录：`<PROJECT_ROOT>/ardy`
- Git commit：`693f74d13b3d04a0a22ce127ee79c929dd89756b`
- 当前项目内代码改动：`ardy/model/llm2vec/llm2vec.py`
- Core checkpoint：`<PROJECT_ROOT>/ardy/checkpoints/ARDY-Core-RP-20FPS-Horizon40`
- checkpoint 文件：
  - `denoiser.safetensors`：623,324,336 bytes
  - `tokenizer.safetensors`：141,746,112 bytes
  - `config.yaml`、动作统计量和 skeleton 资源

### 2.2 文本编码器权重

- Llama 基座：`<HUGGINGFACE_MODELS>/meta-llama/Meta-Llama-3-8B-Instruct`
- MNTP adapter：`<HUGGINGFACE_MODELS>/McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp`
- supervised adapter：`<HUGGINGFACE_MODELS>/McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp-supervised`
- 已在 `<HUGGINGFACE_MODELS>/hub` 建立标准 Hugging Face cache 目录/引用，使仓库可按原始 repo id 离线解析这些本地权重。
- token 文件位于 `<PROJECT_ROOT>/<HF_TOKEN_FILE>`；本文不记录 token 内容。

## 3. 环境与兼容处理

### 3.1 固定约束

- 只使用 `<PY311_ENV>/bin/python`。
- 禁止执行 `conda` 命令。
- 不按仓库声明盲目降级已有依赖。
- 未经审核不安装新包；本次 WebGPU 调研和 toy 没有安装任何包。

### 3.2 当前 py311 主要版本

| 组件 | 版本 |
|---|---:|
| Python | 3.11.11 |
| PyTorch | 2.7.1+cu118 |
| Transformers | 5.13.1 |
| NumPy | 2.2.6 |
| viser（全局） | 1.0.30 |
| PEFT（全局） | 0.18.1 |
| vector-quantize-pytorch | 1.30.1 |
| ONNX | 1.18.0 |
| ONNX Runtime（Python） | 1.20.1 |

为避免扰动全局环境，仓库兼容包放在隔离目录：

`<PROJECT_ROOT>/vendor/ardy_runtime`

其中包括固定的 `kimodo-viser` commit `7c82ad...`（包版本 1.0.16）、PEFT 0.19.1、nodeenv 1.9.1 和 yourdfpy 0.0.60。它们通过 `PYTHONPATH` 使用，没有覆盖 py311 的全局包。

### 3.3 已做的必要修复

1. `ardy/model/llm2vec/llm2vec.py`
   - 显式识别 LLM2Vec 的 MNTP checkpoint 是 PEFT adapter。
   - 先加载真正的 Meta-Llama 基座，再通过 PEFT 加载并 merge adapter。
   - 避免 Transformers 5.13 自动 adapter 路径对权重 key 的错误重写。
2. `vendor/ardy_runtime/peft/import_utils.py`
   - 将与当前环境不兼容的可选全局 `torchao 0.12` 判定为不可用，避免其阻断 BF16 adapter 注入。
3. 早期错误 adapter 加载产生的 embedding cache 已移动到 `.cache/text_embeddings.invalid-auto-adapter` 留档，随后重新生成了干净 cache。
4. vendor viser 前端已构建成功。其 npm 安装树报告 10 个 audit 项（1 low、5 moderate、4 high），未擅自执行 `npm audit fix`。

## 4. 网络结构与算法

### 4.1 总体数据流

```text
文本提示 ── Llama-3-8B + LLM2Vec adapters ──> [B, 1, 4096] 文本条件 ─┐
                                                                    │
历史动作 / 路径点 / 朝向 / 帧索引 ──> 数值约束与 mask ───────────────┤
                                                                    v
噪声 hybrid tokens ── 10 步 DDIM ──> 两阶段 Transformer denoiser
                                      ① global root ② local body
                                              │
                                              v
                                  FSQ Transformer decoder
                                              │
                                              v
                               20 FPS、27 关节、330 维/帧动作
```

当前 Viser 页面只是 React/Vite/Three.js 交互与渲染前端；模型推理发生在 Python/PyTorch 服务器。浏览器通过 HTTP/WebSocket 与服务器通信。

### 4.2 动作表示

- Core skeleton：27 joints。
- 输出：20 FPS，每帧 330 维完整动作表示。
- global root 每帧 5 维：`x, y, z, cos(heading), sin(heading)`。
- 每 4 帧组成一个 motion token。
- 每个 hybrid token：20 维 root（4×5）+ 128 维 body latent = 148 维。
- Horizon40：40 帧 = 2 秒 = 10 个 motion tokens。

### 4.3 FSQ VAE tokenizer/decoder

- Transformer encoder 和 decoder 各 8 层。
- hidden size 512，4 heads，FFN 1024。
- latent embedding 128 维。
- 128 个 FSQ 维度，每维 64 个 level。
- encoder/decoder 都为 causal 配置。
- 推理时 detokenize 会执行 clamp + round，之后由 Transformer decoder 恢复动作。

### 4.4 两阶段 denoiser

- 两个独立的 8 层 Transformer encoder block。
- 第一阶段预测 global root；第二阶段在 local root 条件下预测 body latent。
- 每阶段 hidden size 1024，8 heads，FFN 2048，无 dropout。
- 条件包括文本 token、diffusion timestep、首帧 heading、历史/生成/future mask 和数值动作约束。
- 默认 10 个 denoising steps。

### 4.5 separated CFG

当前实现把 batch 扩成三路：

1. text-only；
2. constraint-only；
3. unconditional。

组合公式为：

```text
uncond + w_text × (text - uncond) + w_cstr × (constraint - uncond)
```

因此即使 `w_text=0`，当前通用 wrapper 仍会计算三路。纯路径 WebGPU 版可改为两路（constraint + unconditional），减少约三分之一 denoiser batch 计算。

## 5. 参数量与权重规模

以下由 safetensors tensor shape 直接统计：

| 组件 | 参数量 | 当前权重大小/理论大小 |
|---|---:|---:|
| ARDY tokenizer/autoencoder | 35,430,052 | 135.2 MiB FP32 文件 |
| ARDY denoiser | 155,823,252 | 594.4 MiB FP32 文件 |
| ARDY motion model 合计 | 191,253,304 | 约 729.6 MiB FP32；理论 FP16 约 364.8 MiB |
| Meta-Llama-3-8B-Instruct | 8,030,261,248 | 约 14.96 GiB BF16 |
| MNTP LoRA | 41,943,040 | 约 160.1 MiB |
| supervised LoRA | 41,943,040 | 约 160.1 MiB |

完整 demo 的约 16.7 GB 显存主要来自 Llama；只加载 ARDY Core 模型时，CUDA allocated 约 872 MiB。

## 6. 已验证运行与性能

### 6.1 Demo

- 正确 demo 已成功在 `0.0.0.0:2333` 启动过。
- Llama + 两个 LLM2Vec adapter 加载正常；日志中仅剩 encoder 场景下预期的 base `lm_head.weight` unexpected key。
- 用户已实际连续运行数千帧，并使用路径点约束。
- UI 日志中常见一次生成耗时约 0.16～0.23 秒，多数为 0.18～0.20 秒，对应 40/44/64 帧窗口。
- 当前阶段 demo 已停止，2333 不再由本次进程占用。

### 6.2 Core40 干净基准

测试条件：RTX A6000、batch 1、10 DDIM steps、当前三路 separated CFG、无 TensorRT、无 `torch.compile`、zero text、40 帧输出；3 次 warmup + 30 次连续运行。

| 指标 | 结果 |
|---|---:|
| 输出 shape | `(1, 40, 330)` |
| 平均耗时 | 0.113174 s |
| 中位数 | 0.112706 s |
| min / max | 0.111667 / 0.120056 s |
| 30 次 wall time | 3.396166 s |
| 生成动作时长 | 60 s |
| realtime factor | 17.67× |
| 等效动作 FPS | 353.3 |
| peak CUDA allocated | 883.9 MiB |
| 连续压测 GPU utilization | 约 33%～35% |

低利用率原因是 batch=1、序列只有 10 tokens、许多短 kernel 和 launch/同步开销，无法把 A6000 的大规模并行单元持续灌满。交互时又要等待用户操作和按真实时间播放，平均占用进一步降到约 3%～4%。

## 7. 2026-07 WebGPU 支持现状

### 7.1 浏览器

| 浏览器/平台 | 当前状态 |
|---|---|
| Chrome / Edge | Windows、macOS、Android、ChromeOS 已可直接使用；Chrome 147～148 开始把 Linux NVIDIA 支持扩展到使用 2024-05 之后现代驱动的 Wayland 环境。 |
| Safari | Safari 26.0 起在 macOS、iOS、iPadOS、visionOS 正式支持 WebGPU。 |
| Firefox | Windows 从 142 起默认启用；Apple Silicon macOS 从 147 起默认启用；Linux 和 Intel Mac 仍主要在 Nightly/开关范围。 |

来源：[Chrome 147–148 的 Linux NVIDIA WebGPU 说明](https://developer.chrome.com/blog/new-in-webgpu-147-148)、[Safari 26 WebGPU](https://webkit.org/blog/17333/webkit-features-in-safari-26-0/)、[Mozilla WebGPU 平台状态](https://developer.mozilla.org/en-US/docs/Mozilla/Firefox/Experimental_features)。

WebGPU 只在 secure context 暴露，即 HTTPS，或浏览器认定为本机的 `http://localhost` / `http://127.0.0.1`。远程服务器 IP 的普通 HTTP 页面通常拿不到 `navigator.gpu`。[ONNX Runtime Web 部署文档](https://onnxruntime.ai/docs/tutorials/web/deploy.html)也明确要求这一点。

### 7.2 框架与网络结构

WebGPU 本身不是“支持 BERT、不支持某个动作网络”的高层框架；它提供 WGSL compute shader。实际网络兼容性取决于上层 runtime 的算子覆盖。

1. **ONNX Runtime Web WebGPU EP**
   - 可以执行自定义 ONNX 图。
   - WASM EP 支持全部 ONNX operators；WebGPU EP 支持一个不断扩展的子集。
   - 当前表中已覆盖 Transformer/ARDY 主要算子：MatMul、Gemm、Softmax、LayerNormalization、GELU/Erf、Gather、ScatterND、Where、Sin/Cos/Atan、Reduce、Concat、Slice、Transpose、Pad、Cast 等。
   - 因而可适配普通 Transformer encoder/decoder、CNN、视觉 Transformer、embedding、语音、扩散网络，以及由这些算子组成的自定义网络。
   - 官方算子表：[ONNX Runtime WebGPU operators](https://github.com/microsoft/onnxruntime/blob/main/js/web/docs/webgpu-operators.md)。
2. **Transformers.js**
   - 是 ONNX Runtime 之上的高层 tokenizer/model/pipeline 封装，有已注册架构列表。
   - 当前 v4 使用重写的 WebGPU runtime，官方称已测试约 200 个模型架构；除 BERT/Llama/Whisper/ViT/CNN 外，也覆盖部分 Mamba、MLA、MoE 等新结构，并能在浏览器及服务端 JS runtime 使用 WebGPU。[Transformers.js v4 发布说明](https://github.com/huggingface/transformers.js/releases/tag/4.0.0)。
   - ARDY 不是 Transformers.js 注册架构，不能把 checkpoint 直接交给 `pipeline()`；但可以绕过该高级 API，直接用 ONNX Runtime Web 创建 session。

### 7.3 WebGPU 对 ARDY 的逐项判断

| 组件 | 适配判断 | 主要工作/风险 |
|---|---|---|
| 两阶段 Transformer denoiser | 可适配 | Linear/MatMul、LayerNorm、GELU、Softmax 和 mask 算子均有覆盖。仓库已有 ONNX opset 17 导出器。 |
| SDPA attention | 可适配 | 当前 ONNX 导出会分解为 MatMul + scale/mask + Softmax + MatMul，均在 WebGPU EP 支持表内。后续可测试是否融合为 MHA 以提速。 |
| global-root 到 local-root | 可适配 | `atan2` 在 opset 17 下实测分解为 Div、Atan、Greater/Less、Add/Sub、Where；这些均受支持。 |
| FSQ decoder | 基本可适配 | Transformer 部分可运行；量化前的 `Round` 不在当前 WebGPU operator 表，应移到 JS/WGSL，或让 decoder 接收已 round 的 latent。 |
| 动态 shape | 能跑但不理想 | `Shape`/`Reshape` 是元数据/CPU 路径，且动态 shape 不利于 graph capture。首版固定 batch=1、10 tokens、40 frames。 |
| 10 步 diffusion loop | 可适配 | JS 循环调用 denoiser session 10 次；应使用 GPU tensor/IO binding，避免每步 CPU↔GPU 拷贝。 |
| separated CFG | 可适配 | 当前三路 batch 可原样导出；纯路径版建议变成两路。 |
| 路径/heading constraints | 可适配 | 只是数值 tensor 和 mask，可在 JS 中构造，不需要 Llama。 |
| skeleton/FK/渲染 | 可适配 | 后处理可在 JS/Three.js 实现；Viser 前端本身已经使用 Three.js。 |
| Llama-3-8B + LLM2Vec adapters | 非首选、非即插即用 | Transformers.js 支持 Llama 类架构，但 ARDY 使用双向 LLM2Vec 行为及两层 adapter；需要 merge/专门导出并验证 4096 维 embedding。下载和内存成本远高于 ARDY Core。 |

仓库的 `scripts/export_onnx.py` 已能在不加载文本 encoder 的情况下导出 CFG+denoiser 和 decoder；这显著降低了迁移风险。但当前脚本的 FP16 选项只用于后续 TensorRT engine build，ONNX 本身仍按已加载 PyTorch dtype 导出。WebGPU 版本需要单独做 FP16 ONNX 转换/验证。

### 7.4 大小、显存和性能预期

- ARDY Core FP32 权重约 730 MiB，FP16 理论约 365 MiB。加上 runtime、重复 session、激活和临时 buffer，浏览器实际峰值会明显高于权重本身；粗略预期桌面端需预留约 0.7～1.5 GiB，必须以原型实测为准。
- WebGPU 基准保证的单个 storage buffer binding 下限是 128 MiB，但 ARDY 的单个线性层矩阵远小于这一值；总模型可以拆成多个 buffer。仍需在 toy/ORT 页面读取实际 adapter limits。[WebGPU 规范 limits](https://gpuweb.github.io/gpuweb/)。
- ONNX Runtime Web 支持 external data、浏览器 cache 和大模型加载；浏览器仍有约 2 GB ArrayBuffer、4 GB WASM address space 等约束。[ONNX Runtime Web 大模型说明](https://onnxruntime.ai/docs/tutorials/web/large-models.html)。
- A6000 PyTorch 的 0.113 秒不能直接外推浏览器。高端桌面独显有机会实时；集成显卡、移动端或不同浏览器可能慢数倍。必须通过 WebGPU prototype 实测。
- 固定 shape 且所有计算 kernel 都落在 WebGPU 后，可以尝试 graph capture；每步输出继续留在 GPU 可减少拷贝。[ONNX Runtime WebGPU EP](https://onnxruntime.ai/docs/tutorials/web/ep-webgpu.html)、[性能诊断](https://onnxruntime.ai/docs/tutorials/web/performance-diagnosis.html)。

## 8. WebGPU toy 与本机调试方式

### 8.1 已创建内容

- 目录：`<PROJECT_ROOT>/webgpu_toy`
- 页面：`index.html`
- 原生 WebGPU 测试逻辑：`app.js`
- 使用说明：`README.md`
- 当前服务：`0.0.0.0:8766`
- 服务只使用 py311 的 `http.server` 标准库，没有安装 npm/Python 包。

客户端推荐使用现有 SSH 登录地址建立安全的 localhost 访问：

```bash
ssh -N -L 8766:127.0.0.1:8766 <服务器登录信息>
```

然后打开：

```text
http://127.0.0.1:8766/
```

选择矩阵大小并运行；出现 `WEBGPU_TOY_PASS` 表示：

- 安全上下文正常；
- `navigator.gpu` 和 adapter 正常；
- WGSL compute pipeline 创建成功；
- GPU buffer 上传/计算/读回正常；
- 矩阵乘法数值校验通过。

这一步验证的是用户浏览器所在电脑的 GPU，而不是服务器 A6000。

### 8.2 已验证的客户端结果

用户于 2026-07-13 使用 Windows 10、Edge/Chromium 149 和 NVIDIA Ampere GPU 完成测试：

| 项目 | 结果 |
|---|---:|
| secure context | 通过 |
| WebGPU API / adapter | 通过 / NVIDIA Ampere |
| `shader-f16` | 支持 |
| `maxBufferSize` | 2.0 GiB |
| `maxStorageBufferBindingSize` | 2.0 GiB |
| `maxComputeWorkgroupStorageSize` | 32.0 KiB |
| 首次 WGSL/pipeline 创建 | 27.0 ms |
| 后续 pipeline 创建 | 2.9～3.7 ms |
| 256×256 FP32 MatMul | 约 0.80～1.29 ms/次（朴素 shader） |
| 数值误差 | `max_abs_error = 0` |
| 最终状态 | `WEBGPU_TOY_PASS` |

这说明该客户端具备运行 ARDY WebGPU 原型所需的基础 API、FP16 特性和足够宽松的单 buffer limit。它仍不代表 ONNX 图已通过，需要继续做 runtime 和模型逐层对拍。

### 8.3 服务器自身能否用 A6000 做 WebGPU

硬件上可行，但当前服务器还不能直接完成服务器 A6000 的浏览器 WebGPU 硬件测试：

- 8× NVIDIA RTX A6000，driver 560.35.03，属于 Chrome Linux NVIDIA 文档要求的现代驱动时间范围。
- 当前未发现 Chrome/Chromium/Firefox 可执行文件。
- 当前没有 DISPLAY/Wayland 图形会话。
- 系统能看到 NVIDIA GL 库，但 Vulkan ICD 目录中没有发现 NVIDIA ICD JSON，`vulkaninfo` 也未安装。

若以后需要在服务器 headless Chrome 上真正调用 A6000，需要单独审核浏览器和 NVIDIA Vulkan/图形运行时方案。Chrome 官方说明 headless Linux 可通过 Vulkan 相关启动参数使用 GPU，但驱动与 ICD 必须完整。[Chrome Headless WebGPU 调试说明](https://developer.chrome.com/blog/supercharge-web-ai-testing)。当前没有为此安装或修改任何系统包。

### 8.4 服务器 PyTorch 与浏览器 WebGPU 对拍方式

后续不会再依赖截图人工判断，而是建立可复现的自动对拍链路：

```text
服务器 py311/PyTorch
  ├─ 固定随机种子并生成输入
  ├─ 运行 ARDY Core，保存参考输出和中间张量
  └─ 提供 model/input/reference manifest
                 │
                 v
用户浏览器 ONNX Runtime WebGPU
  ├─ 读取完全相同的 ONNX 和输入
  ├─ 运行 WebGPU session
  └─ 将输出 tensor 回传到服务器
                 │
                 v
服务器 Python 对拍
  ├─ shape / dtype / NaN / Inf
  ├─ max/mean absolute error
  ├─ max/mean relative error
  ├─ cosine similarity
  └─ 逐阶段 pass/fail 与结果留档
```

对拍范围固定为：

- 不加载、导出或运行 Llama/LLM2Vec；
- `text_feat = zeros([1, 1, 4096])`，`cfg_weight_text = 0`；
- ARDY Core denoiser、FSQ decoder、diffusion loop 和路径/heading 数值约束；
- 先 FP32 判断转换正确性，再测 FP16 的允许误差和性能；
- 先逐组件（decoder、denoiser 单步），通过后再做 10 步端到端，便于定位误差来源。

### 8.5 推荐的递进测试顺序

1. **原生 WebGPU toy**：当前已就绪，验证客户端浏览器链路。
2. **ONNX Runtime Web toy**：用一个很小的 MatMul/LayerNorm ONNX 图验证 WebGPU EP、profiling 和 WASM fallback；需要先审核采用本地 vendor 文件还是 CDN。
3. **ARDY decoder-only**：导出固定 10-token decoder，处理 `Round`，与 PyTorch 输出逐元素比较。
4. **ARDY denoiser 单步**：固定 Core40 shape，确认全部关键算子落在 WebGPU，并记录每个 op 耗时。
5. **10 步无文本路径版**：JS 管理 diffusion loop、两路 CFG 和路径约束，GPU tensor 不回传 CPU。
6. **交互与渲染整合**：接入现有 Three.js UI。
7. **文本可选项**：继续保留服务器 LLM2Vec，或另做浏览器文本 encoder 方案；不阻塞路径版。

### 8.6 已落地的自动对拍通道与结果

2026-07-13 已把 `webgpu_toy` 从手动页面扩展为浏览器 worker：

- `compare_server.py` 同时提供静态文件、浏览器心跳、localhost-only 任务入队、任务状态和 tensor 对拍接口；
- Edge 页面每秒轮询任务，自动读取 ONNX/输入、创建 ONNX Runtime WebGPU session、warmup、执行并回传原始输出；
- 服务器用 NumPy 独立读取 PyTorch reference，检查 shape、dtype、NaN/Inf、max/mean absolute/relative error 和 cosine；
- 每次浏览器原始输出与 JSON 判定都保存在 `webgpu_toy/results/`；
- 用户只需保持通过 SSH tunnel 打开的页面在线，不需要逐次点击。

ONNX Runtime Web 采用已审核的 1.27.0 静态资源，没有安装 npm/Python 包：

| 文件 | 大小 | SHA-256（base64） |
|---|---:|---|
| `ort.webgpu.bundle.min.mjs` | 113,035 B | `OhjH8mHgXUShWy7vGKKOX1PARNXRQLzqYGa68fCci1M=` |
| `ort-wasm-simd-threaded.asyncify.mjs` | 47,507 B | `cjZlO4Vl2kBG5FnNDidBI0GaHZ8fjxj9NsKAWDRsplU=` |
| `ort-wasm-simd-threaded.asyncify.wasm` | 24,254,953 B | `foPNbO535Hi8lqfpGxmBRPteQSYofa8fm1S7GV681Vo=` |

自动对拍均由 Windows 10、Edge 149、NVIDIA Ampere、`shader-f16=true` 的同一浏览器 worker 执行：

| 用例 | ONNX 大小 | 输出 | max abs | mean abs | cosine | 结果 |
|---|---:|---|---:|---:|---:|---|
| smoke FP32 | 18,059 B | `[4,32]` | 1.788e-6 | 1.640e-7 | 0.999999999999968 | PASS |
| ARDY decoder FP32 | 71,552,286 B | root `[1,40,4]` | 0.003825 | 0.000571 | 0.999999537 | PASS |
| ARDY decoder FP32 | 同上 | body `[1,40,325]` | 0.011674 | 0.000502 | 0.999999934 | PASS |
| ARDY denoiser FP32 | 685,044,417 B | `[1,10,148]` | 1.144e-5 | 1.112e-6 | 0.999999999998527 | PASS |

ARDY denoiser 用例的具体范围是固定 10 tokens / 40 frames、单个 separated-CFG denoising step、`text_feat=0`、`cfg_weight_text=0`、`cfg_weight_cstr=1.5`，并带三个非零的路径类约束输入；加载模型时明确使用 `text_encoder=False`。它覆盖了完整 155.8M 参数的双阶段 denoiser 图，不是裁剪后的 toy 网络。

本地基线也已分别通过：

- decoder ONNX Runtime CPU vs PyTorch：root max abs 0.003829、body max abs 0.011671；
- denoiser ONNX Runtime CPU vs PyTorch：max abs 5.245e-6、mean abs 6.106e-7、cosine 约 1.0；
- denoiser ONNX SHA-256：`7dafc43e4ffc77ba2cb067689127a33c468b8273d15ca2264999fd2a84817e3d`；
- decoder ONNX SHA-256：`90994b53a001f413859aae78689b0f66704002934e35613715a1f75c99b9d2b9`。

自动任务的 wall-clock：decoder 从领取到回报约 5.22 秒；denoiser 约 40.47 秒。但它们分别包含 71.55 MB / 685.04 MB 的首次下载、ONNX 解析、session/pipeline 创建、warmup、正式推理和回传，不能当作纯推理时延。下一轮需要让更新后的浏览器 worker 单独回报 session create、warmup 和 inference 时间，并利用浏览器缓存测稳态。

### 8.7 是否能完全在服务器模拟浏览器

数值转换的大部分验证可以完全在服务器完成：PyTorch、ONNX checker、ONNX Runtime CPU/CUDA 与参考输出都不需要浏览器。真实 WebGPU EP 则必须依赖具体浏览器实现。

当前服务器只读探测结果：

- 没有 Chrome/Chromium/Edge/Firefox 可执行文件；
- py311 中存在 Selenium，但 Selenium 不是浏览器本体；
- 没有 `DISPLAY` / `WAYLAND_DISPLAY`，也未发现 Xvfb；
- 8 张 RTX A6000 均为 compute 可见但 display disabled。

因此在不新增浏览器和图形/Vulkan 运行时的前提下，服务器不能本机模拟真实 WebGPU。现有 Edge 页面作为远程 WebGPU worker 是当前零安装方案；任务、输入、输出和判定都由服务器自动控制。

### 8.8 FP16 / mixed precision 浏览器实测

同一 Windows Edge 149 / NVIDIA Ampere worker 对固定 10-token 单步图的稳态结果：

| 图 | 精度 | session create | 稳态中位数 | 相对 FP32 PyTorch 主要误差 | 判断 |
|---|---|---:|---:|---|---|
| decoder | FP32 | 4232 ms | 91.45 ms | root max abs 0.00382；body 0.01167 | PASS |
| decoder | pure FP16 | 2112 ms | 59.92 ms | root max abs 0.1217；body 0.2679 | 严格对拍 FAIL |
| denoiser | FP32 | 38408 ms | 169.32 ms | max abs 1.14e-5 | PASS |
| denoiser | pure FP16 | 17755 ms | 113.90 ms | max abs 0.9693；cosine 0.99576 | 严格对拍 FAIL |
| decoder | coarse mixed FP16 | 2148 ms | 104.82 ms | root max abs 0.1566；body 0.2010 | FAIL，且慢于 FP32 |
| denoiser | coarse mixed FP16 | 17794 ms | 211.88 ms | max abs 1.001；cosine 0.99558 | FAIL，且慢于 FP32 |

pure FP16 将模型文件缩小约 50%，单组件稳态推理曾快约三分之一，但逐组件严格数值误差不可接受。粗粒度地把 LayerNorm、Softmax、Div、Sqrt、Atan、Sin、Cos、Erf、Sigmoid、Round、ReduceSum 留在 FP32 没有改善，反而因大量 Cast 和 pipeline 切换变慢。2026-07-14 用户又在完整无限生成前端完成最终视觉验收，确认 FP16 动作效果严重失真、实际不可用。因此前端默认精度已改为 FP32；FP16 仅作为明确标注的诊断选项保留，不再作为可交付推理模式。

BF16 当前不是浏览器备选：WGSL 原生数值类型有 `f16`/`f32`，没有 `bf16`；而当前问题表现为多层舍入累积，不是 FP16 exponent 不够导致 overflow/NaN。BF16 mantissa 还少于 FP16，不应假设会改善这组误差。

### 8.9 精简无限 WebGPU 前端

入口：

```text
http://127.0.0.1:8766/infinite_demo.html
```

实现文件：

- `webgpu_toy/infinite_demo.html`：最小 UI；
- `webgpu_toy/infinite_demo/demo.js`：10 步 DDIM、history/replan、waypoint、session 和播放状态机；
- `webgpu_toy/infinite_demo/demo.css`：全屏 Canvas 骨架页面；
- `webgpu_toy/infinite_demo/manifest.json`：shape、diffusion schedule、统计量、skeleton 和模型清单；
- `webgpu_toy/tools/generate_infinite_demo_assets.py`：可复现导出；
- `webgpu_toy/tools/validate_infinite_demo.py`：服务器 ORT 整链路回归。

无限逻辑按原版默认值实现：

```text
当前 frame_idx
  -> history_end = min(max_frame, frame_idx + 1)
  -> 取 history_end 前后共 4 帧显式动作
  -> autoencoder encoder 重新编码并以最后历史帧重心居中
  -> 64 帧 denoiser 可见窗口：4 history + 40 generation + 最多 20 future
  -> 10 步 DDIM，只更新 10 个 generation tokens
  -> FSQ requantize + decoder
  -> 保留旧序列到 history_end，覆盖未播放尾部，再拼接 40 帧
  -> 剩余帧数 <= 4 时继续下一轮
```

原版 `Use Dense Root` 默认关闭。普通鼠标点击只在绝对帧 `frame_idx + 60` 放置一个稀疏 2D root x/z keyframe，不自动记录当前 root，也不在当前位置和目标之间逐帧插值；多个不同绝对帧的 waypoint 会累积，同一绝对帧再次点击则更新该帧目标。点击新目标会立即触发 replan。暂停只停止 20 FPS 播放，不清除动作缓存。

模型产物：

| 模型 | 大小 | SHA-256 |
|---|---:|---|
| FP32 denoiser，16 tokens / 64 frames | 685,045,620 B | `77cfe0d95b9644dda099a1e6f3c6f236ecbab44c087ee18cb149272390fef79d` |
| FP32 decoder，11 tokens / 44 frames | 71,554,445 B | `1b371bfd42d8fe01ef49ea6dd75bef69975c081112780e46d7ed6ce1b1fadcc8` |
| FP16 denoiser，16 tokens / 64 frames | 342,682,460 B | `896b4670ecd842bc6611f0be2115c9b47b5494a8afa9559aa76769300926daf0` |
| FP16 decoder，11 tokens / 44 frames | 35,874,863 B | `5ff43703983e0026f3d57dde8004813f673c9b8dc8eebd507b18c10a8d9e3ee5` |
| FP32 4-frame history encoder | 70,578,774 B | 见 manifest |

渲染层刻意精简：原版生成和 history 状态逻辑保留，但不复制 Viser/React/Three.js 的复杂 UI、mesh、文本输入和后处理。页面底部已有滚动时间轴，显示秒数、绝对帧号、已生成区间、当前播放头和稀疏 waypoint，并允许点击已生成区间定位。Canvas 使用 decoder 的 global 6D rotations、Core27 neutral skeleton 和父子层级执行旋转 FK，不再使用 joint-position 捷径。

2026-07-14 针对首次交互观感暴露的问题完成以下修正：

- 旧版在异步 WebGPU 推理时仍让播放时钟前进，导致生成完成后出现跳帧/闪现。现在 `generating=true` 期间完全冻结 `frameIndex`，每个 DDIM step 后主动让浏览器合成冻结画面，完成后清空累计时间再继续 20 FPS 播放。
- 旧版误把 waypoint 实现成 dense 直线约束；现已按原版默认 `dense_path=false` 改成绝对帧稀疏约束。
- checkpoint 的原始 motion stats 是 334 维（5 global-root + 4 local-root + 325 body），页面使用的显式动作是 330 维（5 global-root + 325 body）。现按原版 `MotionRepBase.stats` 取 `[0:5] + [9:334]`，避免 body rotation 统计错位。
- JS 旋转 FK 已与原版 PyTorch `motion_rep.inverse(..., posed_joints_from="rotations")` 对拍：Core27 neutral skeleton 常量最大差约 `1.0e-16`，随机 40 帧动作的 joint position 最大绝对误差 `3.58e-7`、平均绝对误差 `4.32e-8`。

修正版已由同一 Edge/NVIDIA Ampere 客户端实际加载。遥测可见首段在 frame 0 生成 40 帧约 1.72 秒；自动续写在 frame 35 冻结，完成后缓存扩为 77 帧，后续 FP16 重规划约 1.35～1.58 秒。连续采样中 `generating=true` 时播放头分别稳定在 frame 178 和 215，未再推进到缓存末尾。随后客户端切到 FP32 并连续完成 31 轮生成，稳态总耗时中位数约 1.21 秒、无 `init_error`/`generation_error`，无限序列已越过 1000 帧。虽然 FP16 数值始终 finite，但用户视觉验收确认其动作质量完全不可接受；“没有 NaN/Inf”不能作为低精度可用的判断标准。

服务器 ORT CPU 已完成两轮整链路回归，FP32/FP16 均通过：

| 精度 | 首段 | 续写 | joints | finite | 拼接 root 位移 |
|---|---|---|---|---|---:|
| FP16 | `[40,330]` | `[40,330]` | `[40,27,3]` | PASS | 0.05620 m |
| FP32 | `[40,330]` | `[40,330]` | `[40,27,3]` | PASS | 0.05609 m |

既有服务器 ORT 回归验证导出图、10 步 sampler、4 帧重新编码、decoder、轻量 position 分支和第二轮续写的数据流；它不是 WebGPU 替代测试。修正版浏览器的旋转 FK 另按上述方式与 PyTorch 对拍。真实浏览器页面会向 `/api/demo/status` 回报模型加载、每轮耗时、NaN/Inf 和错误，因此客户端打开一次后，服务器可直接继续观察，无需人工抄日志或动作 tensor。

### 8.10 当前浏览器网络的精确结构和参数量

当前无文本的 WebGPU 推理链包含 3 个有参 ONNX 图，从语义上可分成 4 个学习模块：history encoder、root denoiser、body denoiser 和 motion decoder。DDIM 更新、FSQ 重量化/坐标还原、motion 拼接和旋转 FK 没有学习参数。Llama/LLM2Vec 不在这条路径中。

| 模块 | 核心结构 | 学习参数 |
|---|---|---:|
| history encoder | `d=512`，8 层 causal Transformer，4 heads，FFN 1024；4 帧 body 拼成 1 token，325×4 投影到 512，输出 128 维 FSQ latent | 17,554,048 |
| root denoiser Transformer | `d=1024`，8 层 Transformer，8 heads，FFN 2048；包含 text/time/heading prefix 和 20 维 root 输出头 | 73,520,148 |
| body denoiser Transformer | `d=1024`，8 层 Transformer，8 heads，FFN 2048；以 root stage 预测的 local root 为条件，输出 128 维 body latent | 73,630,848 |
| denoiser 共享/阶段投影 | global-root history/generation projection、local-root history/generation projection、future-constraint projection | 8,672,256 |
| motion decoder（浏览器实际导出图） | `d=512`，8 层 causal Transformer，4 heads，FFN 1024；128 维 latent + 每帧 4 维 local-root condition，输出 329 维 local pose | 17,834,276 |
| **浏览器有效合计** | **32 个不共享的 Transformer layer** | **191,211,576** |

PyTorch 原始 Core 模型（不含 Llama）的总参数是 191,253,304，其中 denoiser 155,823,252，autoencoder encoder 17,554,048，完整 decoder 17,876,004。浏览器导出 decoder 没有传 `target_cond`，因此 ONNX 常量折叠删掉了 41,728 个无用参数，得到上表的 17,834,276。

精确 tensor 数据流如下（当前固定 batch=1、20 FPS、4 frames/token）：

| 阶段 | 输入 | 中间/输出 |
|---|---|---|
| history encoder | `normalized_history [1,4,330] FP32` | 取 325 维 body → `input_proj [1300→512]` → 8 层 Transformer → FSQ/重定中 → `local_history [1,1,148]`、`global_translation [1,3]`、`first_heading_angle [1]` |
| denoiser 入口 | `x [1,16,148]`；frame masks `[1,64]`；token masks `[1,16]`；`motion_mask/observed_motion [1,64,330]`；`timestep [1]`；`text_feat [1,1,4096]` 为全 0 | separated CFG 导出图内部仍构造 3 个 branch batch；当前无文本时 text branch 存在可删冗余 |
| root stage | 全局 root 20 + latent 128，另拼约束与 mask 后经投影得 `root_stage_input [3,16,1024]` | 8 层 root Transformer → `[3,16,20]` → reshape 为 `[3,64,5]` global root → 转 local root `[3,64,4]` |
| body stage | local root 每 token 16 + body latent 128，拼约束后经投影得 `body_stage_input [3,16,1024]` | 8 层 body Transformer → `[3,16,128]`；与 root 合并、CFG 合成为 `output [1,16,148]` |
| 10-step DDIM | 上述 denoiser 调用 10 次，只更新 10 个 generation tokens | 保留 initial 的 10 tokens，或 continuation 的 1 history + 10 generation tokens；得 `local_hybrid [1,11,148]` |
| finalize（WASM，无参） | `local_hybrid [1,11,148]`、translation `[1,3]`、length `[1]` | `global_root [1,44,5]`、`latent_tokens [1,11,128]`、`external_cond [1,44,4]` |
| motion decoder | latent `[1,11,128]`、local-root condition `[1,44,4]`、pad mask `[1,44]` | 8 层 decoder Transformer → `root [1,44,4]` + `body [1,44,325]`。当前页面只使用 body，root output 也是可删的无损冗余 |
| 合并 + FK（JS，无参） | `global_root [1,44,5]` + `body [1,44,325]` | crop 为 `motion [40,330]` → rotation-6D FK → `joints [40,27,3]` |

三个有参 FP32 ONNX 图和 finalize 图合计 827,217,523 B。这比 `191,211,576 × 4` 大约多 62 MB，主要是导出图内展开的 sinusoidal position encoding 常量：denoiser 的 `[9999,1024]` 约 40.96 MB，autoencoder 的 `[5000,1024]` 约 20.48 MB。实际只用 16/11 个 motion tokens，这两个大表是首个应先做的无损删减项之一。

### 8.11 WebGPU 小模型路线：先测速，再蒸馏

项目判断已明确：当前 1.91 亿参数、32 层 Transformer 对无文本、单人、40 帧短窗口的浏览器用途明显偏大。目标不是先将现有图粗暴转 FP16，而是在 FP32 teacher 质量基准上训练真正较小的 WebGPU student。

执行顺序固定为：

1. **建立稳态基线**：在同一 Edge/NVIDIA WebGPU 客户端上，分别采集 history encoder、root stage、body stage、每个 DDIM step、finalize、motion decoder、tensor 回读、JS 拼接和 FK 的 warmup 与稳态 p50/p90/p95；同时记录 session create、模型下载和峰值显存。
2. **先做无损精简**：删除无文本的第三个 CFG branch/重复计算，裁掉过长 PE 常量、不使用的 decoder root output 和其他导出冗余，再重测。这一阶段不改权重、不引入质量损失。
3. **训练小 student 拟合 teacher**：先用 teacher 的 clean motion、latent/velocity target、中间 feature 和 rollout 进行监督蒸馏，保住路径响应、动作细节、足接触、速度/加速度和续写接缝。分模块实测已排除保留原 encoder/decoder 的方案：两者合计 p50 约 102 ms、35.39M 参数、FP32 ONNX 约 142.1 MB，因此 encoder、flow 与 decoder 全部进入 student 设计。
4. **将最终生成器蒸馏到 NFE=1**：先建立连续时间 flow-matching 基线和少步 teacher-fitting，再用 distribution matching + temporal adversarial objective 将引导后的 teacher 分布压入单步 student。最终浏览器图不保留 CFG 分支，固定从 `t=1` 噪声一步得到 clean latent。过渡的 4/2 步模型只用于训练诊断，不是最终交付。
5. **最后才量化**：学生网络的 FP32 质量达标后，再评估 weight-only INT8/INT4 或局部 mixed precision。当前 pure FP16 整链路已被视觉验收判定不可用，不再将“文件小一半”当作成功。

浏览器硬预算是：每次 40 帧续写 p50 不高于 100 ms、p95 不高于 170 ms，从而在当前“剩余 4 帧”触发策略下不冻结 20 FPS 播放。子预算为 encoder ≤ 10 ms、NFE=1 flow ≤ 110 ms、finalize + decoder + FK ≤ 25 ms、JS/同步余量 ≤ 15 ms。最终交付的 encoder + flow + decoder 全部权重文件合计必须严格 `<100 MB`；因此不使用参数量代替实际 ONNX external-data 文件大小验收。

### 8.12 FP32 WebGPU 第一轮分模块稳态测速

2026-07-14 在同一 Windows Edge 149 / NVIDIA Ampere 客户端上运行了 1 次 initial warmup、3 次 continuation warmup 和 20 次正式 continuation。页面始终暂停在同一个播放帧，每轮都从相同的前 4 帧重新编码，shape 固定为 encoder `[1,4,330]`、denoiser `[1,16,148] × 10`、decoder `[1,11,128]`。这轮关闭 GPU profiler，所以数据用作真实 wall-time 基线。

| 项目 | p50 | p90 | p95 | 均值 |
|---|---:|---:|---:|---:|
| 40 帧 continuation 端到端 | 1184.19 ms | 1311.45 ms | 1358.31 ms | 1195.68 ms |
| history encoder wall | 52.05 ms | 56.84 ms | 60.51 ms | 54.59 ms |
| history encoder `session.run` | 52.03 ms | 56.82 ms | 60.48 ms | 54.56 ms |
| denoiser 整段 wall（10 步） | 1084.35 ms | 1202.93 ms | 1236.13 ms | 1089.00 ms |
| denoiser `session.run` 合计（10 步） | 938.45 ms | 1078.89 ms | 1085.84 ms | 959.11 ms |
| 为显示冻结姿势主动 `requestAnimationFrame` 等待（10 步） | 123.15 ms | 166.49 ms | 175.26 ms | 129.07 ms |
| finalize（WASM） | 0.76 ms | 1.17 ms | 1.21 ms | 0.84 ms |
| motion decoder `session.run` | 50.29 ms | 57.16 ms | 58.54 ms | 51.07 ms |
| global-root/body 合并 + crop | 0.030 ms | 0.035 ms | 0.035 ms | 0.029 ms |
| rotation-6D FK（40 帧、27 joints） | 0.105 ms | 0.110 ms | 0.114 ms | 0.108 ms |
| finalize + decoder + JS 后处理 wall | 51.55 ms | 58.15 ms | 59.71 ms | 52.08 ms |

10 个 diffusion step 的 `session.run` p50 分别是 91.65、96.27、90.53、91.55、90.17、92.73、91.34、92.91、91.21 和 92.68 ms，没有特定 timestep 形成主要尖峰。因此不是“某一步异常慢”，而是相同的大 root/body Transformer 被连续调用 10 次。

第一轮已能得出两个结论：

- 主瓶颈仍是 denoiser：只计 10 次 `session.run` 就占端到端 p50 的约 79.2%。删掉为调试保留的逐步合成帧等待，可减少约 123 ms，但不能解决约 1.04 秒的 encoder + denoiser + decoder 核心计算。
- encoder 和 decoder 也不应默认保留：它们单次各约 50–52 ms，合计约 102 ms；而且合计 35,388,324 参数、两个 FP32 ONNX 文件合计约 142.1 MB。两者已经几乎吃满暂定的整链路延迟和下载预算，所以都进入蒸馏/缩小候选；finalize、数组拼接和 FK 则可以保留。

测速入口为 `infinite_demo.html?benchmark=1`。它会自动暂停播放、完成 warmup/正式轮次，并将结构化结果回传到 `/api/demo/status`。GPU kernel 归因使用独立的 `?benchmark=1&profile=1` 模式，避免 profiler 额外开销污染本节 wall-time 数据。

#### root/body 归因补充测试

[ONNX Runtime Web performance diagnosis](https://onnxruntime.ai/docs/tutorials/web/performance-diagnosis.html) 文档定义的 `ort.env.webgpu.profiling = {mode: "default", ondata}` 在当前 ORT Web 1.27.0 bundle / Edge 149 组合上没有报错，但两轮生成最终都回传 0 条 kernel timestamp。因此不用空 profiler 数据猜测 root/body 占比，而是从同一 checkpoint 另导出只含 root Transformer 和只含 body Transformer 的诊断图，两者都使用 separated-CFG 内部的 `B=3, T=16, D=1024`。

| 同一 WebGPU worker 诊断图 | session create | 稳态 median | mean | min | max | 数值对拍 |
|---|---:|---:|---:|---:|---:|---|
| root 8-layer Transformer | 20.25 s | 54.90 ms | 56.86 ms | 46.35 ms | 93.88 ms | vs PyTorch max abs `1.78e-5`，cosine `0.9999999999986` |
| body 8-layer Transformer | 20.07 s | 61.08 ms | 74.54 ms | 51.69 ms | 253.56 ms | vs PyTorch max abs `5.63e-6`，cosine `0.9999999999953` |
| 当前完整 16-token denoiser step | 38.34 s | 179.31 ms | 186.58 ms | 164.20 ms | 258.71 ms | vs ORT CPU max abs `4.11e-5`，cosine `0.9999999999046` |

这三个任务在同一后台 worker 上顺序执行，可用来看相对重量，但后台标签页调度使绝对时延约为前台无限页的两倍，所以不能用 179.31 ms 替换上表前台每步约 90–96 ms 的主基线。在同条件内粗分，root stage 占完整 step 约 30.6%，body stage 约 34.1%，其余至少 35.3% 来自输入/约束投影、mask/concat、global-to-local root 转换、CFG 分支拆合和图 I/O。由于拆分图各自多承担一次 session-run 固定开销，35.3% 还是“其余部分”的保守下界，不是精确上界。

这一结果排除了“只有 body 慢”或“只有 root 慢”的假设：root/body 两个 8-layer backbone 基本同等重，而非 Transformer 的约束处理也很重。小模型设计应同时缩减两个 backbone，并重做无文本 CFG/约束数据流，不应只替换单个阶段。

### 8.13 第一版 student 的 WebGPU 架构筛选（flow v1 后续已拒绝）

在任何训练之前，已将随机初始化的固定 shape student 分别导出为 FP32/FP16 ONNX，用同一 Edge/NVIDIA Ampere 自动 worker 完成 PyTorch 数值对拍和 3 warmup + 20 timed runs。该测试只回答“算子组合、shape 和 dispatch 是否值得训”，随机权重不能回答动作质量。表中的 22.19M flatten flow 是 v1 历史数据；训练后因严重过拟合被拒绝，当前 7.30M temporal flow v2 见 8.15。

| student 图 | 参数 | ONNX 大小 | session create | 稳态 median | mean | 对拍 |
|---|---:|---:|---:|---:|---:|---|
| encoder FP32 | 3,886,208 | 15,539,291 B | 1101.48 ms | 14.37 ms | 15.30 ms | PASS |
| encoder FP16 | 同上 | 7,772,981 B | 676.18 ms | 16.90 ms | 16.89 ms | PASS |
| NFE=1 flow v1 FP32（已拒绝） | 22,189,000 | 88,734,002 B | 5156.27 ms | 33.30 ms | 33.05 ms | PASS |
| NFE=1 flow v1 FP16（已拒绝） | 同上 | 44,377,440 B | 2649.76 ms | 43.45 ms | 46.91 ms | PASS |
| decoder FP32 | 4,953,792 | 19,803,931 B | 1575.55 ms | 34.94 ms | 35.03 ms | PASS |
| decoder FP16 | 同上 | 9,912,651 B | 750.35 ms | 38.78 ms | 39.14 ms | PASS |

该 v1 组合的 FP16 三图权重实际合计 62,063,072 B，满足 `<100 MB` 且留有约 37.9 MB 质量调整余量。后台三图 median 直接相加为 FP32 82.60 ms、FP16 99.14 ms；这只是当时允许进入训练的架构门槛，不能覆盖其后独立验证作出的质量拒绝。

三个 FP16 图都比同结构 FP32 慢，所以不宣称 `shader-f16` 会自动加速。当前选 FP16 的必要性来自严格体积上限；最终可使用 FP16 权重与局部 FP32 归一化/累加的 mixed graph，但必须重新对拍和前台测速。

### 8.14 训练前的目标预算和候选规格（历史方案）

本节保留训练前的决策审计；实际实现已经改为固定 NFE=1 的 28-token temporal flow v2 和 compact codec，见 8.15 与 `ARDY_WEBGPU_DISTILL_PLAN.md`。以下早期 2-step/独立 root-body 方案不再代表当前实现。

当前页面在剩余 4 帧时启动续写，20 FPS 下只有 200 ms 的自然提前量。因此在同一 Edge/NVIDIA Ampere 客户端上定义两级验收线：

- **理想线**：40 帧 continuation 端到端 p50 ≤ 100 ms。
- **必过线**：端到端 p95 ≤ 170 ms，给 200 ms 窗口留至少 30 ms 调度/抖动余量。任何超时轮次仍冻结当前帧而不跳帧，保留现在的可调试语义。

| 模块 | 当前 p95 | student p95 预算 | 要求 |
|---|---:|---:|---|
| history encoder | 60.48 ms | ≤ 10 ms | 至少约 6× 延迟改善 |
| denoiser（全部步数） | 1085.84 ms（不含 rAF 等待） | ≤ 110–120 ms | 步数和单步网络必须同时缩减 |
| finalize | 1.21 ms | ≤ 2 ms | 保留，不蒸馏 |
| motion decoder | 58.54 ms | ≤ 20–25 ms | 至少约 2.3–3× 改善 |
| JS merge + FK | 0.15 ms 以内 | ≤ 1 ms | 保留 |
| 其他调度/I/O 余量 | 未独立硬分 | ≤ 15 ms | 优先 GPU tensor binding/图捕获实验 |

当前为了调试冻结姿势，每个 diffusion step 后都主动 `requestAnimationFrame`，10 步 p50 累计等待 123.15 ms。这个逻辑在当前 teacher 调试页继续保留；最终 student 只有 2 步且整轮小于 170 ms 后，可以不再强制每步等一个合成帧，但仍然不允许用播放头跳时间来掩盖慢推理。

第一版最终 student 的建议规格是：

| 部分 | 当前 | student 初始规格 | 预估参数 |
|---|---|---|---:|
| motion encoder | 8 层，d=512，FFN=1024，latent=128 | 4 层，d=256，FFN=512，4 heads，latent=64 | 约 2.5M |
| root denoiser | 8 层，d=1024，FFN=2048 | 4 层，d=512，FFN=1024，删除 LLM 4096→d prefix | 约 8–9M |
| body denoiser | 8 层，d=1024，FFN=2048 | 4 层，d=512，FFN=1024，与 root 对称起步 | 约 8–9M |
| 约束/输入投影 | 8.67M，latent=128，3-way CFG | latent=64，d=512，path-only 2-way CFG；评估进一步合并 unconditional/constraint 数据流 | 约 3–4M |
| motion decoder | 8 层，d=512，FFN=1024 | 4 层，d=256，FFN=512，4 heads | 约 2.5M |
| **合计** | **191.21M** | **先做独立 root/body 的稳妥版** | **约 25–28M** |

这一规格把每个 denoising step 的 Transformer layer 调用从 16 层减到 8 层，宽度从 1024 减到 512；再从 10 步减到 2 步后，Transformer 层调用次数从 160 次降到 16 次，且单层主要矩阵计算约为原来的四分之一。这只是理论上界，WebGPU dispatch、mask 和 I/O 不会按参数线性缩放；所以必须每个版本都重复本节的真实浏览器测速。

训练不建议一次同时更换 codec、denoiser 和步数，而是使用可定位的分阶段路线：

1. 先保持现有 128-dim codec，训练小 root/body denoiser 在相同 timestep 上拟合 FP32 teacher 的 clean/noise prediction 和中间 root/body feature；这一阶段先保持 10 步，用来验证容量而不是最终速度。
2. 与此同时单独训练 64-dim、d=256 的小 motion codec，在真实 motion 和 teacher rollout 上使用 explicit-motion、rotation、joint FK、velocity/acceleration、foot-contact/slide 和接缝 loss；不用“latent MSE 小”代替视觉质量。
3. 将小 denoiser 迁移到 64-dim student latent，用 teacher 解码后动作再编码为 student target，重做 10-step teacher fitting 和长 autoregressive rollout。
4. 按 10→4→2 步逐级蒸馏；4 步是质量中间点，2 步是当前 170 ms 必过线的主目标，1 步只在 2 步质量稳定后尝试。
5. 最后在 2-step student 上增加 DMD2 类 distribution matching 和 temporal adversarial critic。critic 应同时看 explicit motion、FK joints、速度和 foot contact 的多尺度时间窗，并保留 waypoint、history seam 和长 rollout 约束；不直接照搬图像 discriminator。

更激进的第二个架构候选是“共享 4-layer d=512 trunk，root 头后只用 1–2 层 body refinement”，或直接单次 joint root/body prediction。它们可进一步删掉重复 backbone 和 global-to-local 中间数据流，但风险是破坏当前明确的“先 root、后 body”条件关系。应当在上述独立 root/body student 成为可用质量基线后再对照训练，不作为第一个无对照实验。

### 8.15 首轮有界蒸馏实测：codec 可冻结，flow 仍是质量瓶颈

首轮训练没有沿用已被否定的 flatten-MLP flow。当前实际网络、浏览器固定输入输出和参数如下：

| 模块 | 固定输入 | 固定输出 | 结构摘要 | 参数 |
|---|---|---|---|---:|
| history encoder student | body `[B,4,325]` | latent `[B,1,128]` | flatten 1300→512；3 个 `512→1024→512` residual GELU block；512→128 | 3,886,208 |
| NFE=1 flow v2 | noise `[B,10,148]`、history `[B,1,148]`、path `[B,64,3]`、heading `[B,2]`、has-history `[B,1]` | clean hybrid `[B,10,148]` | 1 global + 1 history + 16 path + 10 generation tokens；width 384、6 heads、4 trunk attention blocks；20-dim root head 注回后 2 body blocks、128-dim body head | 7,302,676 |
| motion decoder student | latent `[B,11,128]`、local root `[B,44,4]`、valid `[B,11]` | body `[B,44,325]` | 144→512；4 个固定 11-token/channel mixer；512→1300 后 reshape | 4,953,792 |
| **合计** |  |  |  | **16,142,676** |

flow v1 的 22,189,000 参数 flatten MLP 在 8192 窗口上严重记忆训练集，独立验证 FK-MPJPE 约 1.10 m，已明确拒绝。v2 用 28-token 明确保留时间、稀疏 path 和 root→body 结构，参数反而降至 7.30M。

大 teacher corpus 由原版 FP32、10-step、constraint CFG 1.5 的推理链无侵入采样：131,072 个窗口、256 shards、9,700,210,688 B；87.5% continuation、74.91% constrained。落盘用 FP16 控制体积，但 teacher 计算本身没有降精度。全量 SHA-256、schema、15 个 tensor 的 shape/dtype/finite 校验已通过。

codec v3 在四张最空闲 A6000 上使用 BF16 autocast + FP32 EMA 训练 4000 step，训练器 wall time 132.30 s。独立 1024-window 验证逐一比较 1000/2000/3000/4000 的 raw 和 EMA 后，选择 step-4000 encoder EMA + decoder raw：

| codec 指标 | codec v2 | codec v3 selected |
|---|---:|---:|
| encoder requantized latent L1 | 0.12102 | **0.07629** |
| encoder exact FSQ bin accuracy | 25.94% | **39.09%** |
| decoder FK-MPJPE | 0.03463 m | **0.02307 m** |
| decoder rotation geodesic | 0.12546 rad | **0.08969 rad** |
| decoder foot slide | — | **0.02097** |

flow v2 large 使用同一大 corpus、90% student-encoded history + 10% teacher-history 正则，BF16 + FP32 EMA 训练 4000 step，训练器 wall time 210.88 s。独立验证按浏览器实际的 student history 运行：

| raw checkpoint | root/body MSE | FK-MPJPE | rotation | foot slide | 当前约束点 path error |
|---:|---:|---:|---:|---:|---:|
| 1000 | 0.15267 / 0.72622 | 0.48042 m | 0.50967 rad | 0.41600 | 0.44224 m |
| 2000 | 0.11727 / 0.65362 | 0.38836 m | 0.46312 rad | 0.28671 | 0.33757 m |
| 3000 | 0.10316 / 0.62060 | 0.33522 m | 0.43646 rad | 0.23317 | 0.28652 m |
| 4000 | **0.09803 / 0.61151** | **0.32338 m** | **0.42772 rad** | **0.21908** | **0.27283 m** |

step-4000 EMA 略差，FK 为 0.32427 m。将同一 raw flow 改用精确 teacher history 后，FK 仅从 0.32338 m 变为 0.32310 m，path error 从 0.27283 m 变为 0.27364 m；因此 encoder 误差几乎没有贡献当前 flow 误差。decoder 独立 FK 只有 2.31 cm，也明显不是当前主瓶颈。

低精度不是这版 student 的误差来源：在同一 1024-window 验证集上让 encoder/flow/decoder 内部全部使用 FP16，最终 FK 为 0.323383 m（FP32 0.323380 m），path error 为 0.272752 m（FP32 0.272826 m），decoder 独立 FK 的变化小于 `5e-8 m`。BF16 的 FK/path 为 0.323497/0.273955 m，同样接近 FP32。这个结果说明 compact student 没有复现原版 pure-FP16 的严重数值退化，但它只是服务器 CUDA 验证，不能替代真实浏览器 WebGPU 对拍。

这回答了“encoder/decoder 是否需要动”的问题：原版两者合计约 102 ms、142.1 MB，所以必须蒸馏；蒸馏后 codec 的体积、速度和独立质量已经足以冻结，不应继续无依据砍小。下一轮资源应优先给 flow。大数据 flow 相对 8k probe 已将 FK 从 0.570 m 降到 0.323 m、path error 从 0.576 m 降到 0.273 m，但仍未达到最终动画质量，不能因为模型只有 7.3M 或输出 finite 就直接宣称可交付。

选中训练权重直接导出的文件体积为：

| 模块 | FP32 ONNX | FP16 ONNX |
|---|---:|---:|
| encoder | 15,551,369 B | 7,778,915 B |
| flow v2 | 29,256,743 B | 14,651,309 B |
| decoder | 19,836,043 B | 9,928,379 B |
| **合计** | **64,644,155 B** | **32,358,603 B** |

完整配置、所有检查点评估和失败审计位于 `distill_runs/first_12h_20260714_014643/RUN_PLAN.md`。本轮 teacher 采样、训练、评估和 ONNX 导出没有安装任何新包。

## 9. 下一步建议

完整的 NFE=1 student 固定设计、精确 tensor/参数/文件体积、teacher 采样和蒸馏验收协议已另存于 [`ARDY_WEBGPU_DISTILL_PLAN.md`](./ARDY_WEBGPU_DISTILL_PLAN.md)。该文档覆盖本节早期的 2-step 临时方案：最终交付固定为 NFE=1，encoder/flow/decoder 均为 student，实际 ONNX 权重合计严格 `<100 MB`。

当前下一里程碑改为：

> 冻结已经达到厘米级独立误差的 compact codec，集中解决 NFE=1 flow 的动作分布质量；任何改进都必须同时通过浏览器训练权重对拍和长 rollout。

具体动作：

1. 完成已入队的六个 trained-weight FP32/FP16 WebGPU 用例，记录真实权重的数值误差、session create 和稳态分模块延迟；随后只在前台动画页测整链路，后台三图延迟求和不作为最终结论。
2. 以 step-4000 raw flow 为监督基线，先做受控的容量/结构 A/B：优先把剩余约 67.6 MB FP16 预算给 width 512 或更深的 temporal flow，并保持相同数据、step、seed 和评估集；不改 codec，避免无法归因。
   两档 width-512 候选随后已用完全相同的 131,072-window corpus、4000 step、四卡 BF16、FP32 EMA、FSQ history 和 seed 完成训练；4+2 与 5+2 的训练器 wall time 分别为 `226.83/239.42 s`。独立验证和长 rollout 结果见第 13 节。真实浏览器 latency job 仍在排队，因此这里不提前按参数量选择最终部署结构。
3. 加强最终 `t=1` endpoint、root/path、FK、速度、foot 和 seam loss，并做 1/5/20/50 次自回归 rollout。paired teacher 指标继续保留，但要增加 teacher-feature/distribution 指标，避免把“另一条合理动作”误算成纯逐元素失败。
4. teacher fitting 稳定后进入 DMD2 类两时间尺度 distribution matching + temporal adversarial critic。critic 同时看 normalized motion、FK joints、速度、foot-contact 和多尺度时间窗；paired waypoint/history/seam loss 始终保留，防止条件坍塌。teacher/critic 只参与训练，不进入浏览器。
5. 只有固定视觉样例、长 rollout、path success、NaN/Inf、FP16 数值与前台 p50/p95 全部过线，才把 student 接到精简无限生成前端；推理期间仍冻结播放头，禁止跳帧掩盖延迟。

此路线不需要先解决 Llama 浏览器化，也不要求修改现有 py311 全局依赖。任何新包或浏览器/系统组件安装仍需先提交清单审核。

## 10. 2026-07-14 codec 再审计：忠实 runtime、FSQ 修正与缩小候选

在决定是否继续砍 encoder/decoder 前，已新增 `ardy_distill/student_runtime.py`，把三张 student 图放回原版窗口语义中，而不是只测互不相干的随机 tensor。continuation 的实际顺序为：4 帧显式 history 重心平移 → student encoder → 原版 FSQ requantize → world waypoint 转当前局部坐标 → NFE=1 flow → generated latent 再 requantize → root 还原到 world → 11-token student decoder。初始窗用 decoder 的第 0 个 dummy token，返回 40 帧；续写窗第 0 个 token 是真实 history，返回 44 帧。

用原版 `trace_autoregressive_step` 生成 initial/continuation 两个固定调用后，逐项比较 student runtime 的 `path_condition`、`first_heading`、`has_history`、`global_translation` 和 history root，最大绝对误差全部为 **0**；输出分别为 `[1,40,330]` 和 `[1,44,330]`，均 finite。机器记录是 `distill_runs/first_12h_20260714_014643/eval/student_runtime_semantics.json`。这只证明外围数据流与原版一致，不代表当前 flow 动作质量已经合格。

同时发现旧 flow v2 训练/评估曾把 student encoder 的连续输出直接交给 flow，而原版 `_encode_init_history` 实际使用 FSQ token。训练器和评估器现已统一在 encoder 后执行精确 FSQ；旧 step-4000 权重没有重训，所以仍标记为 provisional。对旧权重重新做 1024-window 忠实 FSQ 评估后，FP32 为 root/body MSE `0.098115/0.611594`、FK-MPJPE `0.323774 m`、rotation `0.428109 rad`、path error `0.272751 m`；FP16 为 `0.098143/0.611603`、`0.323804 m`、`0.428159 rad`、`0.272633 m`。与旧的连续-history 结果只差约 0.4 mm FK，说明该语义错误必须修，但不是当前 32 cm flow 误差的来源。以后所有 flow 训练默认使用 FSQ history。

在 A6000 上对忠实 runtime 做 20 次 warmup + 100 次 batch-1 测速，得到：

| CUDA arithmetic | 窗口 | encoder p50/p95 | decoder p50/p95 | flow p50/p95 | 整个 Python runtime p50/p95 |
|---|---|---:|---:|---:|---:|
| FP32 | initial | — | 1.280 / 1.358 ms | 2.644 / 2.727 ms | 5.402 / 5.606 ms |
| FP32 | continuation | 0.488 / 0.504 ms | 1.265 / 1.359 ms | 2.624 / 2.798 ms | 6.384 / 6.586 ms |
| FP16 | initial | — | 1.339 / 1.436 ms | 2.869 / 3.009 ms | 5.704 / 5.895 ms |
| FP16 | continuation | 0.445 / 0.454 ms | 1.321 / 1.368 ms | 2.837 / 3.005 ms | 6.623 / 6.856 ms |

这些 CUDA 数字说明 codec 在服务器 GPU 上不是瓶颈，也说明 batch-1 小图不会仅因改 FP16 就自动更快；它们不能替代 WebGPU。真实 Edge 后台架构筛选中，当前 codec 仍是 encoder FP32/FP16 `14.37/16.90 ms`、decoder `34.94/38.78 ms`。因此当前 codec 作为**质量基线**冻结，但 decoder 没达到 20–25 ms 子预算，仍值得先做无训练的 WebGPU 结构筛选。

在 width-512、5+2 block 的 NFE=1 flow 训练完成后，又用相同 continuation 语义、batch 1、20 次 warmup + 100 次正式重复重测了一次，以排除 flow 结构改变对整链路判断的影响：FP32 encoder/decoder p50 为 `0.490/1.272 ms`，FP16 为 `0.443/1.317 ms`；对应整个 Python runtime p50 为 `6.797/6.958 ms`。机器记录位于 `eval/runtime_w512_t5_raw_{fp32,fp16}.json`。因此 codec 在 CUDA 上可以冻结；是否继续缩小只由真实浏览器候选延迟决定，不能把 CUDA 的约 1.8 ms codec 合计延迟外推到 WebGPU。

已导出并通过 ONNX checker、CPU ORT 对拍的两档 codec 候选：

| 候选 | encoder 结构/参数/FP16 ONNX | decoder 结构/参数/FP16 ONNX | codec FP16 合计 |
|---|---|---|---:|
| 保守 | width 384、2 residual blocks；1,733,120；3,468,201 B | width 384、3 mixer blocks；2,336,173；4,679,791 B | **8,147,992 B** |
| 激进 | width 256、2 residual blocks；893,312；1,789,608 B | width 256、3 mixer blocks；1,164,677；2,339,871 B | **4,129,479 B** |

两档 FP32/FP16 encoder/decoder 共 8 个真实 WebGPU latency job 已入队。浏览器 worker 最后心跳仍是 `2026-07-14 02:13:50 CST`，当前离线，所以尚无可声称的浏览器结果。决策规则是：若保守档不能把 decoder 明显压到约 25 ms，就不为体积继续损失质量；若能显著省时，再用完全相同 teacher 数据分别蒸馏 encoder/decoder，并以 latent/FK/seam/长 rollout 选择，而不是按参数量直接采用。整个再审计没有安装任何包。

## 11. 2026-07-14 原版无限续写 rollout 与 codec 责任分离

已新增 `ardy_distill/tools/evaluate_rollout.py`，按原版 UI 的实际时间语义做闭环评估，不再用孤立的 40 帧随机窗口代替无限生成。固定为 20 FPS、剩 4 帧时触发续写、回看 4 帧 history、每次生成 40 帧；第一窗输出 40 帧，之后每次保留原 buffer 并追加 37 帧，因而 buffer 长度为 `40, 77, 114, ...`。每个窗口 teacher/student 共用同一个 world-space waypoint 和同一份初始噪声，teacher 执行原版 10-step DDIM，student 执行 NFE=1。所有时间轴事件、1/5/20/50 窗口快照和动作 tensor 都已落盘。

当前 provisional flow + codec v3 的 FP32 长 rollout 结果是：

| 续写窗口 | 累计帧数 | root drift mean / final | FK-MPJPE | student waypoint mean | student foot slide | student root / joint seam jump |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 40 | 0.236 / 0.377 m | 0.374 m | — | 6.788 m/s | — |
| 5 | 188 | 1.347 / 1.696 m | 1.369 m | 1.524 m | 5.050 m/s | 4.283 / 8.946 m/s |
| 20 | 743 | 11.453 / 17.068 m | 11.531 m | 12.021 m | 4.946 m/s | 5.304 / 9.466 m/s |
| 50 | 1853 | 17.642 / 43.491 m | 17.683 m | 18.146 m | 5.093 m/s | 5.194 / 9.338 m/s |

同一路径上 teacher 在 50 窗口的 waypoint mean 仅 `0.0105 m`、foot slide `0.0640 m/s`、root/joint seam jump `0.139/0.257 m/s`。student 虽然全程 finite，但从第一窗就有 37 cm FK 误差，长续写后是明确的闭环崩溃，不能进入前端验收。FP16 与 FP32 几乎重合：50 窗口 FP16 的 root drift mean/final 为 `17.619/43.000 m`、FK `17.659 m`、waypoint mean `18.118 m`。因此这次崩溃不是 FP16 数值精度造成的。

为把 flow 与 codec 的责任分开，又运行了 **oracle-flow codec ablation**：每窗直接给 student encoder/decoder 同窗 teacher 生成的 clean latent 和 root，但仍使用 student codec 并真实闭环回馈它解码后的 body history。结果为：

| 续写窗口 | root drift | FK-MPJPE | waypoint mean | codec foot slide | codec joint seam jump |
|---:|---:|---:|---:|---:|---:|
| 1 | 0 | 0.0135 m | — | 0.265 m/s | — |
| 5 | 0 | 0.0145 m | 0.0051 m | 0.239 m/s | 0.471 m/s |
| 20 | 0 | 0.0326 m | 0.0157 m | 0.318 m/s | 1.232 m/s |
| 50 | 0 | 0.0229 m | 0.0105 m | 0.332 m/s | 0.874 m/s |

这个消融说明：当前米级路径偏移和整体动作崩溃由 **flow** 主导，不是 encoder/decoder 容量导致；codec 的位置和骨架还原仍在 1.3–3.3 cm，可以先作为质量基线。但 decoder 也不是完全免责：它把 teacher 的 foot slide 从约 `0.064` 放大到 `0.332 m/s`，50 窗口 joint seam jump 从 `0.257` 放大到 `0.874 m/s`。因此决策是：

1. flow 必须先扩容/重训，并加入 endpoint、路径、速度、脚接触和 seam 的长 rollout loss；这是当前首要瓶颈。
2. 原版 encoder/decoder 仍必须替换：两者浏览器 p50 合计约 102 ms、FP32 ONNX 约 142.1 MB，并不“很快”。
3. codec v3 只是临时质量基线，不是宣布永久冻结。当前 WebGPU decoder median 仍约 35–39 ms；若 width-384/256 候选在真实浏览器能明显降到约 25 ms 或以下，就单独蒸馏 codec，并把 foot-contact/slide 和 joint-seam 作为硬指标；若延迟收益很小，则不为进一步减参破坏质量。

结果文件位于 `distill_runs/first_12h_20260714_014643/rollout/faithful_50_{fp32,fp16}*/`，每个目录含 `metrics.json` 和固定动作样例。浏览器候选仍是 18 个 pending，worker 未上线，所以本节没有伪造 codec 候选的 WebGPU 延迟结论。

## 12. 2026-07-14 DMD2 类两时间尺度与 temporal adversarial 实现

已按 [DMD2 论文](https://arxiv.org/abs/2405.14867) 和[官方实现](https://github.com/tianweiy/DMD2)的核心更新方式，新增 `ardy_distill/dmd2.py`、`ardy_distill/models/critic.py`、`ardy_distill/train_dmd2.py` 和 teacher-score 验证工具。这里不是把图像 DMD2 代码直接搬到动作上，而是保留其两个关键机制并显式适配 ARDY：

1. fake-score 每个 iteration 更新，在当前 generator 样本加原版 cosine diffusion 噪声后预测 epsilon；generator 默认每 5 步更新一次，形成 two-time-scale 训练。
2. teacher 直接使用冻结的原版 155.8M denoiser 预测 real `x0`；fake-score epsilon 转为 fake `x0`。DMD 梯度按 `p_real = x_t - x0_real`、`p_fake = x_t - x0_fake`、`(p_real - p_fake) / mean(abs(p_real))` 归一化，generator 使用 stop-gradient surrogate loss。teacher/fake-score 均不随 generator 反传。
3. temporal critic 每步更新，使用 softplus non-saturating real/fake loss。它不只看 latent，而是看 40 帧的 normalized root/body、物理 FK joints、joint velocity 和 foot contact，并通过多 dilation 时序块与 native/÷2/÷4 三个时间尺度判别。history、waypoint path、heading 和 `has_history` 都进入 critic 条件，避免无条件 GAN 忽略路径。
4. 动作适配中仍保留 paired teacher endpoint，并同时保留 path、decoder body、rotation/FK、joint velocity/acceleration、foot slide 和 history-generation seam loss。原因是当前 supervised baseline 还有明显条件误差，如果只优化无配对分布目标，可能生成“像动作但不跟点”的样本。

teacher score adapter 的语义不是用 shape test 猜测，而是用 corpus 保存的完整 DDIM 轨迹反向验证。一个 16-sample batch 同时包含 8 个 initial 和 8 个 continuation；对 `t=9...0` 的每步 noised generation 调 teacher `x0`，再用原 DDIM 公式重建下一状态。10 步的最大绝对误差为 **`7.03e-6`**、平均误差 `4.42e-7`，history token count 精确为 0/1，generation 始终为 10，验证通过。记录为 `distill_runs/first_12h_20260714_014643/eval/teacher_score_adapter.json`。

单卡 A6000 BF16 smoke 使用当前 7.30M flow 初始化 generator 和 fake-score，用缩小的 0.175M critic 执行 1 个 warmup iteration + 3 个 generator update。fake-score、critic、DMD teacher、paired/path/decoder/quality/seam/adversarial 全部完成 forward/backward/gradient clipping/optimizer/scheduler/EMA；中途从 step 2 状态恢复后续跑到 step 3，断点恢复通过。generator、EMA、fake-score 和 critic 的所有权重均 finite。smoke 记录位于 `distill_runs/first_12h_20260714_014643/dmd2_smoke/`。默认完整 critic 为 3,323,649 参数，fake-score 为 7,302,676 参数；两者和 teacher 都只是训练辅助，**不进入 WebGPU 权重体积**。

这个 smoke 只证明算法、条件语义、梯度、checkpoint 和 resume 闭环可用，3 步不构成质量提升证据。正式训练仍应先用真实 WebGPU 延迟选定 width/depth，再先做 faithful-FSQ supervised fitting，最后在选中 checkpoint 上进行 DMD2+adv refinement。当前浏览器 worker 仍离线，18 个 job 仍 pending。本节没有安装任何包。

## 13. 2026-07-14 width-512 受控扩容、分模块测速与长闭环结果

为避免把结构、数据量和训练语义同时改变，width-512 的 4+2 与 5+2 block 候选严格复用同一份 `teacher_train_large_v2`、global batch 128、4000 step、学习率、timestep sampling、teacher-history 概率、FSQ history、seed 和 codec v3。两者都在物理 GPU `6,4,3,5` 上用 Accelerate/DDP、BF16 autocast 和 FP32 EMA 完成；未安装任何包。

独立 `teacher_val` 1024-window 验证中，raw checkpoint 一致优于 EMA，5+2 raw 最优：

| flow | endpoint total | FK-MPJPE | rotation | foot slide | path error |
|---|---:|---:|---:|---:|---:|
| 旧 width-384 4+2 raw（忠实 FSQ） | 1.01069 | 0.32377 m | 0.42811 rad | — | 0.27275 m |
| width-512 4+2 raw | 0.94861 | 0.30175 m | 0.40550 rad | 0.18576 | 0.25039 m |
| **width-512 5+2 raw** | **0.93586** | **0.29571 m** | **0.40270 rad** | **0.18094** | **0.21891 m** |

5+2 raw 相对旧 width-384 的 endpoint/FK/path 分别改善约 `7.4%/8.7%/19.7%`，说明容量确实有帮助，但 29.6 cm 的单窗 FK 仍不够。把 encoder、flow、decoder 全部改为 FP16 后，endpoint/FK/path 为 `0.93594/0.29560 m/0.21885 m`，与 FP32 基本一致；当前失败不是低精度造成的。

同一权重在 A6000 上以忠实 continuation runtime、20 次 warmup + 100 次重复测得：

| CUDA arithmetic | encoder p50/p95 | flow p50/p95 | decoder p50/p95 | runtime total p50/p95 |
|---|---:|---:|---:|---:|
| FP32 | 0.490 / 0.516 ms | 2.997 / 3.058 ms | 1.272 / 1.320 ms | 6.797 / 7.123 ms |
| FP16 | 0.443 / 0.461 ms | 3.171 / 3.262 ms | 1.317 / 1.426 ms | 6.958 / 7.331 ms |

机器记录为 `eval/runtime_w512_t5_raw_{fp32,fp16}.json`。CUDA 上 compact codec 合计约 1.8 ms，可以冻结；浏览器上已测 codec v3 仍是约 49–56 ms 合计，不能用 CUDA 数字替它下结论。width-384/256 codec 的 8 个 FP32/FP16 WebGPU job 继续等待 Edge worker。

5+2 raw 的忠实 50-window rollout 仍然失败：

| checkpoint | 1-window FK | 5-window FK | 20-window FK | 50-window FK | 50-window waypoint mean | 50-window foot slide |
|---|---:|---:|---:|---:|---:|---:|
| 旧 width-384 FP32 | 0.374 m | 1.369 m | 11.531 m | 17.683 m | 18.146 m | 5.093 m/s |
| width-512 5+2 FP32 | **0.262 m** | 2.101 m | 11.644 m | **16.825 m** | **17.270 m** | **3.664 m/s** |
| width-512 5+2 FP16 | 0.261 m | 2.114 m | 11.716 m | 16.921 m | 17.368 m | 3.673 m/s |

扩容改善了第一窗和最终部分指标，却没有修复自回归闭环，而且第 5 窗反而更差；不能把它宣称为动画质量成功。当前合理动作是以 5+2 raw 作为 supervised 容量上界候选进入有界 DMD2+temporal adversarial refinement，同时继续保留 paired endpoint/path/seam 约束；浏览器选型若证明 5+2 latency 不划算，再回退到更快结构，而不是隐瞒延迟或质量任一侧。

对应结果为 `eval/flow_w512_t{4,5}_b2_{raw,ema}_fp32.json`、`eval/flow_w512_t5_b2_raw_fp16.json` 和 `rollout/faithful_50_w512_t5_raw_{fp32,fp16}/`。

## 14. 2026-07-14 四卡 DMD2 探针审计：先继续 teacher fitting

以 width-512 5+2 raw 为初始化，在物理 GPU `6,4,3,5` 上运行了 global batch 32、完整 3,323,649 参数 temporal critic 的四卡探针。第一轮暴露出 DMD2 训练器特有的 scheduler 问题：`Accelerator` 默认会在未 split batch 的多卡环境把一次显式 `scheduler.step()` 扩成 `num_processes` 次，20-step 探针的学习率因此被压缩 4 倍。`train_flow.py` 与 `train_codec.py` 原本已经显式关闭该行为，只有新写的 `train_dmd2.py` 漏掉。现已同样设置 `step_scheduler_with_optimizer=False`，并把 generator update 强制纳入日志条件，避免周期日志恰好永远错开 DMD/adversarial 指标。

修复后重新运行 20 个 generator update：四卡 DDP、BF16 forward/backward、teacher score、fake-score、full critic、EMA、checkpoint 和所有梯度路径均完成；generator/fake-score/critic/EMA 全部 finite。学习率按 `4e-6, 8e-6, ... 2e-5` 正常 warmup，而不是每次跨 4 个 scheduler step。记录位于 `dmd2_w512_t5_probe20_v2/`。

但是独立 1024-window 验证没有质量收益：raw 的 endpoint/FK/path 从 supervised 基线 `0.93586 / 0.29571 m / 0.21891 m` 回退到 `0.94864 / 0.30134 m / 0.23149 m`，foot slide 从 `0.18094` 回退到 `0.18814`；EMA 也没有逆转。critic loss 在 20 个 generator update 内从约 `1.39` 降到约 `0.68`，generator adversarial loss 上升到约 `1.72`，说明默认 critic 对当前尚未充分拟合 teacher 的 generator 变强过快。

因此不把这个探针包装成“DMD2 已提升质量”，也不直接放大默认配置。顺序改为：同一 5+2 架构先做更充分的监督 teacher fitting，并用独立集与长 rollout 选 checkpoint；只有 paired/FK/path 达到稳定基线后，再降低 critic 相对更新强度并做 DMD/adv 权重消融。这个决策符合“先拟合 teacher，再 DMD2+adv”，而不是指望 GAN 修复尚未学会的条件映射。

## 15. 2026-07-14 20k teacher fitting：单窗继续收敛，长闭环仍发散

在不改架构、corpus、seed、global batch 128、faithful FSQ history、BF16 autocast 和 FP32 EMA 的前提下，width-512 5+2 从头训练到 20,000 step。四张 A6000 的训练 wall time 为 `1172.80 s`（约 19.55 分钟），共看到 2.56M samples，全程 finite。每 2000 step 的 raw/EMA 共20个 checkpoint 都在独立 1024-window `teacher_val` 上评估，后半程 EMA 稳定优于 raw：

| checkpoint | endpoint total | FK-MPJPE | rotation | foot slide | path error |
|---|---:|---:|---:|---:|---:|
| 4k raw 旧基线 | 0.93586 | 0.29571 m | 0.40270 rad | 0.18094 | 0.21891 m |
| 14k EMA | **0.74587** | 0.22723 m | 0.34260 rad | 0.10038 | 0.13691 m |
| 20k EMA | 0.75093 | **0.22286 m** | **0.34119 rad** | 0.09498 | **0.13188 m** |

20k EMA 相对 4k raw 的独立集 FK/path 分别改善约 `24.6%/39.8%`，说明 teacher fitting 并没有在 4k 就完全饱和。但单窗指标不能代替无限生成闭环；用原版的 4 帧历史重编码、40 帧续写、replan buffer 和 waypoint 时间语义测得：

| checkpoint | 1-window FK | 5-window FK | 20-window FK | 50-window FK | 50-window waypoint mean | 50-window foot slide |
|---|---:|---:|---:|---:|---:|---:|
| 4k raw FP32 | 0.262 m | 2.101 m | 11.644 m | 16.825 m | 17.270 m | 3.664 m/s |
| 14k EMA FP32 | 0.206 m | 0.849 m | 11.385 m | 17.369 m | 17.794 m | 1.922 m/s |
| **20k EMA FP32** | **0.187 m** | **0.718 m** | **10.310 m** | **16.753 m** | **17.220 m** | **1.812 m/s** |
| 20k EMA FP16 | 0.187 m | 0.693 m | 10.515 m | 17.196 m | 17.670 m | 1.810 m/s |

20k 把第 5 窗 FK 从 `2.101 m` 降到 `0.718 m`，且 foot slide 约减半，但到第 20/50 窗仍发散到 `10.31/16.75 m`；FP16 与 FP32 趋势一致，仍不是低精度数值问题。所以不再无限增加单窗监督 step，下一阶段以 20k EMA 为质量基线，优先做弱 critic / DMD-only / 低 adversarial-weight 的受控消融，并把 5/20/50-window rollout 作为选择指标，不再只看 one-window validation。

encoder/decoder 的执行决策也因此固定为两层：

1. **服务器训练/质量主线先冻结 compact codec v3**：忠实 batch-1 CUDA p50 为 encoder `0.490 ms` + decoder `1.272 ms`，50 窗 oracle-flow 又证明它不会造成米级 root/path 崩溃。当前不同时重训 codec 和 flow，以保持可归因性。
2. **部署端是否再蒸馏只由真实 WebGPU 决定**：当前 Edge p50 是 encoder/decoder FP32 `14.37/34.94 ms`、FP16 `16.90/38.78 ms`，合计仍约 `49.31/55.68 ms`，不能称为“已经快到永久不动”。只有 width-384/256 无训练架构能在同一前台 Edge 将 decoder p50 明显压到约 `25 ms` 或以下，且 codec 合计时延降低至少约 `25%`，才值得单独启动 codec 蒸馏；否则保留 v3，先解决 flow 闭环质量。重训 codec 时必须同时验收 FK、foot slide、joint seam 和 50-window oracle-flow，不以 latent loss 或参数量单独定案。

训练、全20个独立集评估和三组长 rollout 的机器记录分别位于 `flow_w512_t5_b2_fsq_20k/`、`eval/flow_w512_t5_20k_step*_fp32.json` 和 `rollout/faithful_50_w512_t5_20k_*/`。本节未安装任何包。

## 16. 2026-07-14 长闭环根因：不是 FP16，也不是 codec，而是 path-condition exposure bias

20k EMA 之后又做了两个受控 DMD 探针。DMD-only 从 20k EMA 继续 200 个 generator step，独立集 path 只改善约 `3%`，但 1/5/20-window rollout 均回退；弱 adversarial 版本采用 adversarial weight `0.005`、DMD weight `0.05`、critic width 128 和 critic LR `2e-5`，同样没有形成稳定的闭环收益。因此当前不延长 DMD/adv，也不把 critic loss 下降误报成动作质量提升。

随后给 `evaluate_rollout.py` 增加每窗口输入分布统计和 `--on-policy-teacher-diagnostic`。完整 131,072-window 原始 teacher corpus 的有效 path-condition norm 为 mean `1.813`、p95 `3.106`、p99 `3.272`、绝对最大值 `3.329`；20k EMA 的忠实 rollout 从第 7 窗开始越过 corpus 最大值，50 窗中有 41 窗超过 `3.329`，第 50 窗达到 `29.120`。相反，history-hybrid L2 大体仍在训练分布内，故首要分布偏移来自“学生没到 waypoint 后，下一个相对目标越来越远”，而不是 history tensor 单独爆炸。

从学生当前状态重新调用原版 10-step teacher 证明这些状态仍可纠正。第 50 窗学生起点距目标 `42.07 m`，teacher 一窗可把最小误差改善约 `10.1 m`，学生只改善约 `0.47 m`。按 path norm 分桶的单窗最小进展如下：

| path norm | teacher progress | student progress |
|---|---:|---:|
| `0–3.33` | 2.095 m | 0.296 m |
| `3.33–10` | 6.448 m | 0.159 m |
| `10–20` | 8.207 m | 0.202 m |
| `>20` | 7.923 m | 0.137 m |

这给出了明确的算法结论：需要在学生真实访问状态上向 teacher 查询纠偏动作，即 DAgger/on-policy teacher fitting；继续堆原始 teacher-window 或直接加 GAN 不能覆盖缺失的条件分布。诊断机器记录位于 `eval/teacher_corpus_distribution.json`、`rollout/faithful_50_w512_t5_20k_step0020000_ema_fp32_shift_diag/` 和 `..._onpolicy_diag/`。

## 17. 三轮 DAgger：先解决跟路，再回收动作物理质量

新增 `generate_onpolicy_teacher_shards.py`，严格复现原版 UI 的 trigger=4、replan buffer=1、4-frame history crop 和 `old[:history_end+1] + generated[history_length:]` 更新。reference teacher branch 定义可达 world waypoint，student branch 产生真实部署状态，再从 student history 向原版 teacher 查询同一 waypoint 的纠偏动作。首轮 4096-window corpus 的 path norm 中位数约 `6.1`、p95 约 `15`、最大约 `24`，完整 shape/finite/SHA 校验通过。FP16 落盘会让 DDIM state replay 的严格 score-adapter 误差达到 `0.002721`，而 FP32 smoke 为 `1.24e-5`；所以这些 FP16 shards 只用于 paired supervised/path fitting，不作为精确 score-adapter 证据。

第一轮若用 50% on-policy、path weight `1.0`，50-window waypoint 从 `17.220 m` 降到 `0.990 m`，但脚滑爆到 `13.495 m/s`，说明模型在用 root 硬追路径而 body/foot 没有同步。随后改为 25% on-policy、75% replay、path weight `0.05`，每步加入 decoder/FK/foot/seam loss；最佳 step-750 把 waypoint 保持在 `0.955 m`，脚滑降到 `3.554 m/s`，但仍高于物理基线。

第二轮由该闭环模型重新采 4096 个较温和状态，path norm p50 约 `1.78`、p95 约 `3.68`、最大 `6.89`。从 step-750 继续以 12.5% round2 + 87.5% replay、无显式 path loss、较强物理损失训练，step-500 达到 waypoint `0.791 m`、脚滑 `2.806 m/s`、joint seam `6.548 m/s`。再把它与 20k EMA 在同一权重盆地线性插值，发现 50-window 稳定性在 alpha `0.65–0.75` 附近发生相变：alpha `0.50` 到第 50 窗又漂到 `9.201 m`，alpha `0.75` 则为 `1.056 m`，脚滑 `2.290 m/s`。

第三轮以 alpha `0.70` 状态再采 4096 窗，path norm p50 约 `2.05`、p95 约 `3.64`，只用 5% on-policy + 95% replay、无显式 path loss继续 500 step。当前网络候选为 round3 step-400 raw：

| 候选 | 50-window FK | waypoint mean | foot slide | joint seam mean |
|---|---:|---:|---:|---:|
| 20k EMA supervised | 16.753 m | 17.220 m | **1.812 m/s** | **4.002 m/s** |
| round1 naive step-1000 | 1.801 m | 0.990 m | 13.495 m/s | 12.067 m/s |
| round1 physical step-750 | 1.783 m | 0.955 m | 3.554 m/s | 8.750 m/s |
| round2 step-500 | 1.908 m | **0.791 m** | 2.806 m/s | 6.548 m/s |
| interpolation alpha 0.75 | **1.478 m** | 1.056 m | 2.290 m/s | 5.533 m/s |
| **round3 step-400** | 1.576 m | 1.054 m | **2.224 m/s** | 5.549 m/s |

round3 step-400 在五条不同随机 50-window 路径上的 waypoint mean 为 `0.894–1.278 m`、平均 `1.108 m`；foot slide 为 `2.197–2.347 m/s`、平均 `2.256 m/s`；joint seam mean 为 `5.542–6.751 m/s`、平均 `5.896 m/s`。因此闭环收益不是单 seed 偶然，但拼接仍是剩余主要视觉问题。数据、训练和评估记录分别位于 `onpolicy_teacher_{pilot_v1,round2_physical750,round3_interp070}/`、`flow_w512_t5_{20k_onpolicy_physical_pilot1k,round2_physics_recovery750,round3_minimal_corrective500}/` 与对应 `eval/`、`rollout/` 目录。

## 18. 拼接惯性化与当前部署候选

训练侧新增与 rollout 定义对齐的 physical seam loss：只在真实 buffer 会保留的 history 与首个生成帧之间计算 root/joint velocity-jump error，默认权重为 0，保证旧实验语义不变。300-step 小步实验的最佳 joint seam 为 `5.51 m/s`，与 round3 的 `5.55 m/s` 仅是噪声量级，说明单纯继续调全局 flow 权重的收益已经很低。另一方面，直接把 teacher clean generation 送入同一 compact codec 的 50-window oracle 仍有 FK `0.0229 m`、foot slide `0.332 m/s`、joint seam `0.874 m/s`，所以 codec 不是当前 5–6 m/s 拼接跳变的主因。

为此在 evaluator 中加入可完全关闭的部署侧短时惯性化。每次 continuation 只对前 `K` 帧施加“上一帧常速度外推与学生首帧之间的 feature offset”，用 cubic smoothstep 衰减到 0，contact 四通道不插值；播放时间、buffer 长度、帧号和 waypoint 均不改变，也没有跳帧。固定 seed 的结果为：

| 后处理 | waypoint mean | FK | foot slide | root seam | joint seam | joint seam p95 |
|---|---:|---:|---:|---:|---:|---:|
| 关闭 | 1.054 m | 1.576 m | **2.224 m/s** | 2.767 m/s | 5.549 m/s | 10.950 m/s |
| 4 帧 / 50% | 1.054 m | 1.576 m | 2.211 m/s | 1.383 m/s | 2.813 m/s | 5.901 m/s |
| **8 帧 / 100%** | **1.054 m** | 1.579 m | 2.257 m/s | **0.000 m/s** | **1.179 m/s** | **3.635 m/s** |

8-frame/100% 在五个 seed 上保持 waypoint mean 平均 `1.108 m`，joint seam mean 平均 `1.358 m/s`（范围 `1.179–1.823`），p95 平均 `4.045 m/s`，foot slide 平均 `2.299 m/s`。这是当前推荐给浏览器整体动画对拍的候选：round3 step-400 raw + 8-frame inertialization；它仍需要真实 WebGPU/视觉验收，不能只凭离线指标宣布完成。

当前三份 FP32 safetensors 的精确体积为 encoder `15,546,912 B`、flow `59,951,304 B`、decoder `19,820,312 B`，合计 `95,318,528 B`（`90.90 MiB`），NFE=1 且已低于 100 MB 原始权重目标。最终 ONNX graph/external data 仍需重新导出并按实际文件求和，不能把 safetensors 体积直接冒充浏览器下载体积。当前 CUDA continuation 常见 stage 约 encoder `0.5 ms`、flow `3.0–3.2 ms`、decoder `1.3 ms`、Python 整窗约 `7–8 ms`；它只说明服务器计算量小，不替代 Edge WebGPU 前台测速。

下一步固定为：把同一 8-frame inertialization 端到端接入精简无限生成前端；导出 round3 step-400 的 FP32 与受控 mixed-precision 图；恢复真实 Edge worker，先做 ONNX 数值对拍，再测下载/session/warmup/p50/p95/峰值显存和完整时间轴动画。codec v3 在该阶段继续冻结；只有更小 codec 在同一前台 Edge 达到此前约定的至少 25% 合计提速且质量过线，才单独蒸馏。以上所有实验未安装任何新包，也未修改用户已有的 `ardy/model/llm2vec/llm2vec.py`。

## 19. round3 FP32 浏览器导出、全链路对拍与无限前端接入

当前候选已按真实部署结构重新导出，不再复用原版 10-step denoiser。三个 learned ONNX 分别是 compact encoder `15,551,369 B`、round3 step-400 NFE=1 flow `59,993,269 B`、compact decoder `19,836,043 B`；首段/续写两个固定 ARDY root/FSQ 状态图合计 `45,264 B`。浏览器实际需要的五个 ONNX 总计 **`95,425,945 B`（`91.005 MiB`）**，低于 100 MiB，且没有 external-data 隐藏文件。模型参数为 encoder `3,886,208`、flow `14,985,364`、decoder `4,953,792`，合计 `23,825,364`。

新的服务器 validator 同时运行 PyTorch reference 与 CPU ORT，覆盖首段、续写、JS 侧 history recenter + FSQ、稀疏 waypoint、首段 dummy token、续写真实 history token、8-frame inertialization 和 buffer 更新。结果如下：

- initial clean/explicit 的最大绝对误差分别为 `3.22e-6 / 4.63e-5`；continuation clean/explicit 为 `2.21e-6 / 6.95e-5`。
- JS 语义的 history hybrid 与 PyTorch reference **零误差**；8-frame inertialization 后全动作最大误差仍为 `6.95e-5`。
- 首段输出 `[1,40,330]`，续写内部 `[1,44,330]`、裁掉 history 后仍追加 40 帧；第二窗 buffer 精确为 `40 -> 77`。首段/续写 waypoint 分别落在局部帧 `60/62`，与绝对时间轴选择一致。

真实 Edge 149 + NVIDIA Ampere WebGPU worker 已直接接收当前训练权重的三个任务，三者均通过 WebGPU 对 PyTorch 数值校验：

| 模块 | session create | warm 后 p50 / p95 | WebGPU max abs | 结果 |
|---|---:|---:|---:|---:|
| encoder | 1023.95 ms | 14.160 / 16.121 ms | `1.55e-6` | PASS |
| NFE=1 flow | 3577.81 ms | 64.390 / 77.036 ms | `3.46e-6` | PASS |
| decoder | 1260.12 ms | 33.613 / 56.985 ms | `7.63e-6` | PASS |

三个独立模块 p50 直接相加为 `112.163 ms`，p95 相加为 `150.142 ms`。这个结果已经低于 `170 ms` 必过预算，但略高于 `100 ms` 理想线；它只是同一前台 worker 的分模块预算，不能伪装成完整页面端到端分位数。真实记录已固化为 `webgpu_toy/infinite_demo/webgpu_module_validation.json`，服务器全链路记录为 `server_validation.json`。

`/infinite_demo.html` 已切换为这套 FP32 全 student：只执行一次 flow，不再加载原版 685 MB denoiser，也不提供质量不可用的全 FP16 选项；保留原版 20 FPS、剩 4 帧触发、replan buffer=1、4-frame history、`40,77,114...` 无限 buffer、绝对帧时间轴、当前帧 `+60` 稀疏鼠标 waypoint 和生成期间播放冻结。8-frame/100% 惯性化已逐公式移植，contact 四通道不混合，不改变任何帧号、时间或目标位置。页面还会分别上报 encoder/flow/decoder/finalize/inertialization/FK 耗时。

尚未完成的最后一项是刷新后的完整动画视觉验收与前台整链路 benchmark；已有在线 worker 是算子测试页，不会被服务器擅自导航到动画页。因此当前可以声称“导出、CPU 全链路、真实 WebGPU 三模块数值与延迟通过”，不能声称最终视觉效果已经由用户验收。无限页现已增加 localhost-only 命令队列；用户只需打开/刷新一次页面，服务器之后可直接下发 `start/pause/restart/benchmark/waypoint` 并读取结果，不再要求逐项手点。整个阶段未安装任何包，也未修改 `ardy/model/llm2vec/llm2vec.py`。

## 20. 视觉否决、抖动根因与训练流程重置

2026-07-14 的真实动画视觉验收已经明确否决上一节候选：FP32 页面虽然数值 finite、WebGPU 对拍和延迟均通过，但动作持续抖动、轨迹含大量高频噪声，不能交付。`webgpu_toy/infinite_demo/webgpu_e2e_validation.json` 已改为 `passed=false`，其中性能/数值仍为 true、视觉为 false。此前“下一步只剩刷新验收”的描述因此只保留为历史状态，不再代表当前结论。

离线诊断将问题分成三层：

- 旧 NFE=1 flow 在首个 40-frame 窗口、完全不跨 rollout seam 时，root acceleration P95 已是 teacher 的约 `61×`，jerk 约 `251×`，高频比约 `13.7×`。所以噪声不是前端播放追帧或长闭环累计才产生。
- 对 root endpoint 做 4 次固定 binomial projection 后，50-window root acceleration/jerk 分别降到 teacher 的约 `1.77×/1.33×`，root 高频低于 teacher；但 root-relative body acceleration/jerk/高频仍约为 `2.07×/4.22×/15.96×`。这证明只修 root 或接缝滤波不能解决 body 噪声。
- teacher clean latent 直接送入 compact decoder 的 50-window codec oracle 也有 body acceleration/jerk/高频约 `1.99×/3.98×/10.02×`，说明 decoder 本身同样需要重新蒸馏，而不是继续冻结。

训练审计还发现一个已修正的 train/deploy 语义错误：旧 flow 与 DMD2 训练在解码预测 latent 时使用了 batch 中的 teacher `decoder_local_root`，而浏览器运行时使用从 student 预测 root 推导的 local-root。现在 `deployment_decoder_roots()` 统一生成与部署完全相同的 initial/continuation decoder 输入；验证对 teacher trace 的 global-root 最大误差为 0、有效 local-root 最大误差约 `1.30e-5`。旧的 decoded-body 静态指标不再作为可靠选模依据。

旧 loss 虽然名义上包含速度、加速度和脚滑，但逐目标参数梯度审计显示 decoder 的这些项只相当于主 feature 重建梯度的约 `1%–3%`，且旧训练每 8 步才计算一次；joint/rotation jerk 均缺失。flow 的物理质量项同样远弱于 endpoint loss，并且每 20 步才计算一次。原 codec v3 还从 `1e-4` 开始训练。因而旧版本不能称为充分的平滑蒸馏。

新的 decoder 诊断先冻结 encoder，用固定 `5e-6` 跑 1k-step 方向消融；它在独立 1024-window 验证集使 joint velocity/acceleration/jerk 分别改善约 `6.6%/8.8%/9.9%`，但 50-window 高频仍为 teacher 的 `8.76×`，所以 1k 只证明梯度方向，不作为训练完成或部署候选。相位分析进一步定位到 compact decoder 的 4-frame 输出结构：旧 codec 的 phase-0/1 acceleration P95 是 teacher 的 `2.73×/2.51×`，phase-1 jerk 达 `5.44×`，显著高于 phase-2/3。新增跨 token 边界的 joint/rotation velocity、acceleration、jerk 六项，并在 teacher-val 与 on-policy batch 上逐项做梯度校准。

最初错误地只在 GPU 6 上以 batch 64 启动了该协议；它已在 20k checkpoint 后停止，目录 `codec_decoder_boundary_const3e6_100k/` 已写入 `SINGLE_GPU_PIPELINE_CONTROL.md`，只允许作为早期 loss/静态趋势对照，不能再称为正式 100k，也不继承其 Adam 状态。错误原因是把其他卡上的驻留显存误判成活跃任务，违背了用户明确指定的四卡要求。

冻结 encoder 后的数据读取现已只加载 decoder 所需的 5 个字段，并用 shard-local shuffle 避免全局随机索引反复打开大 shard。四卡吞吐检查均从原始 codec checkpoint 独立初始化：每卡 batch 256（global 1024）在 200 step 中处理 204,800 samples，训练段耗时约 `19.97 s`，约 `10,256 samples/s`；每卡 batch 512（global 2048）在 500 step 中处理 1,024,000 samples，训练段耗时约 `86.55 s`，约 `11,831 samples/s`。后者有效区间四张 A6000 的 SM 通常为 `92%–95%`、功耗约 `218–230 W/卡`，训练自身新增显存约 `2.0 GiB/卡`，因此选用 batch 512。两个目录均写入 `THROUGHPUT_ONLY.md`，不作为模型结果。

正式训练已于 2026-07-14 13:06 启动：物理 GPU `3,4,5,6`，四进程 DDP，每卡 batch 512、global batch 2048，encoder frozen，BF16 forward/backward + FP32 optimizer/EMA，weight decay 0，固定 LR `3e-6`，无 warmup、无 cosine，预算完整 100k optimizer steps（共 204.8M sample draws）。质量与六个 token-boundary 项每一步计算，每 5k 保存权重、每 25k 保存完整可恢复状态。正式目录为 `distill_runs/first_12h_20260714_014643/codec_decoder_boundary_4gpu_b512_const3e6_100k/`；首步已完成，启动后复采四卡 SM 为 `90%–97%`。后续必须用独立 validation、token-phase P95/频域指标及 50-window codec oracle 选模，不能用训练 loss、吞吐或有限性代替视觉质量。本次更正未安装任何包，也未修改用户已有的 `ardy/model/llm2vec/llm2vec.py`。

为给正式 checkpoint 提供长序列控制组，已对停止在 20k 的单卡版本补做独立 1024-window 与 50-window codec-oracle。静态 raw 全面优于 EMA：raw 的 decoder total/FK/foot-slide 为 `0.33024 / 0.01850 m / 0.01546`，EMA 为 `0.33287 / 0.01877 m / 0.01564`。50-window 中 raw 相对 teacher 的 root-relative joint acceleration P95、jerk P95、高频比和 joint seam mean 分别为 `1.403× / 2.412× / 4.705× / 2.582×`；EMA 为 `1.421× / 2.480× / 4.946× / 2.612×`，因此不存在“EMA 静态略差但长序列更平滑”的逆转。raw 的 phase-0/1 acceleration P95 仍为 teacher 的 `1.678×/1.762×`，phase-0/1 jerk 为 `2.241×/3.260×`；边界项已明显改善旧 codec，但仍远未达到视觉通过线。记录位于 `eval/codec_boundary_single_control_step0020000_{raw,ema}_{fp32,jitter_analysis,token_phase_jitter}.json` 与对应 `rollout/..._oracle50/`，仅作为控制组，不影响四卡任务继续到 100k。

四卡正式任务的首个 5k checkpoint 已落盘并在不中断训练的情况下完成同口径评估。独立集 raw 仍全面优于 EMA：raw 的 decoder total/FK/foot-slide/joint jerk 为 `0.33266 / 0.01880 m / 0.01559 / 0.01370`，EMA 为 `0.33549 / 0.01908 m / 0.01579 / 0.01400`。5k raw 的 50-window acceleration P95、jerk P95、高频比和 joint seam mean 分别为 teacher 的 `1.419× / 2.476× / 4.896× / 2.638×`，phase-0/1 acceleration 为 `1.722×/1.793×`，phase-0/1 jerk 为 `2.346×/3.381×`；它已接近、但尚未超过单卡 20k 控制。相同权重的 CUDA FP16 算术相对 FP32 在 decoder total/FK/foot-slide/joint jerk 上仅相差约 `0.0031%/0.0077%/0.0018%/0.0048%`，BF16 最大相对差约 `0.32%`；模型本身的低精度算术不是当前抖动来源，但这不替代最终 ONNX/WebGPU 图级对拍。因此不提前停止、不按样本访问量冒充 optimizer update，继续完整固定 LR 训练并在 10k/15k/... 重复独立集评估。记录位于 `eval/codec_boundary_4gpu_b512_step0005000_*` 与 `rollout/codec_boundary_4gpu_b512_step0005000_raw_oracle50/`。

10k checkpoint 已产生超过旧单卡 20k 控制的新增收益，且训练继续运行。独立集 raw/EMA 的 decoder total 为 `0.30753/0.31195`，FK 为 `0.01659/0.01695 m`，raw 仍全面优于 EMA；raw 相对单卡 20k raw 的 FK、joint jerk、boundary jerk 和 continuation 首帧 joint seam error 分别改善约 `10.3%/13.6%/13.8%/12.0%`。50-window raw 相对 teacher 的 acceleration P95、jerk P95、高频比和 joint seam mean 从 5k 的 `1.419×/2.476×/4.896×/2.638×` 降至 `1.342×/2.147×/3.521×/2.360×`；phase-1 jerk 从 `3.381×` 降至 `2.885×`。这证明四卡大 batch 的提升同时存在于独立静态集与长序列，但高频和 phase-1 jerk 仍明显超线，不能切换到 flow 或浏览器阶段。记录位于 `eval/codec_boundary_4gpu_b512_step0010000_*` 与 `rollout/codec_boundary_4gpu_b512_step0010000_raw_oracle50/`。

15k checkpoint 已在四卡训练不中断的情况下完成同口径评估。独立集 raw/EMA 的 decoder total 为 `0.28810/0.29338`，FK 为 `0.01513/0.01551 m`，joint jerk 为 `0.01028/0.01061`，boundary joint jerk 为 `0.01197/0.01237`，continuation 首帧 joint seam error 为 `0.41494/0.42600 m/s`，因此 raw 继续优于 EMA。首个 50-window seed 的 codec-oracle acceleration P95、jerk P95、高频比和 joint seam mean 相对 teacher 分别为 `1.287×/1.953×/2.935×/2.197×`，相比 10k 的四项均继续下降；token phase-0/1 acceleration P95 为 `1.439×/1.496×`，phase-0/1 jerk P95 为 `1.871×/2.581×`。补做四条独立路径后，五个 seed 的四项均值为 `1.160×/1.612×/2.283×/1.812×`，范围为 `1.101–1.287×/1.478–1.953×/1.993–2.935×/1.589–2.197×`。其中四条新增路径还用 10k 权重做了严格 paired comparison：10k→15k 的 acceleration/jerk/high-frequency/seam 均值从 `1.163×/1.652×/2.492×/1.805×` 降至 `1.128×/1.526×/2.120×/1.716×`；四个 seed 的 jerk 均下降 `7.4%–7.9%`、高频均下降 `13.8%–16.0%`，不是随机路径带来的假趋势。这说明边界监督仍在持续削弱 compact decoder 的 4-frame 周期噪声，但最坏 seed 高频仍约为 teacher 的 2.94 倍、phase-1 jerk 约为 2.58 倍，不能视为视觉通过。正式训练保持 GPU `3,4,5,6` 四进程 DDP、global batch 2048、固定 LR `3e-6` 继续到 100k；检查时四卡利用率约 `91%–96%`、功耗约 `226–235 W/卡`。记录位于 `eval/codec_boundary_4gpu_b512_step00{10000,15000}_*` 与 `rollout/codec_boundary_4gpu_b512_step00{10000,15000}_raw_oracle50*/`。

20k checkpoint 继续给出跨独立集和跨路径的一致收益。1024-window raw/EMA 的 decoder total 为 `0.27299/0.27862`，FK 为 `0.01413/0.01450 m`，joint jerk 为 `0.00941/0.00972`，boundary joint jerk 为 `0.01094/0.01131`，continuation 首帧 joint seam error 为 `0.38742/0.39734 m/s`，raw 仍全面优于 EMA。相对 15k raw，FK、joint jerk、boundary joint jerk 和首帧 seam 分别再改善约 `6.6%/8.5%/8.6%/6.6%`。相同五条 50-window 路径的 acceleration/jerk/high-frequency/seam 均值从 15k 的 `1.160×/1.612×/2.283×/1.812×` 降至 `1.136×/1.521×/2.033×/1.723×`，最坏 seed 也从 `1.287×/1.953×/2.935×/2.197×` 降至 `1.257×/1.828×/2.541×/2.081×`；五个 seed 的四项全部逐条改善。五 seed token-phase 汇总的 acceleration P95 四相位均值为 `1.176×/1.302×/1.016×/1.058×`，jerk P95 为 `1.381×/1.940×/1.459×/1.092×`，phase-1 jerk 最坏 seed 为 `2.415×`；这确认 phase-1 仍是当前 decoder 的结构性周期残差，而非单路径偶然。同一 raw 权重的 CUDA FP16 相对 FP32 在 decoder total/FK/joint jerk/seam 上仅为 `+0.0033%/+0.0082%/+0.0240%/-0.0014%`，BF16 则约为 `+0.22%/+0.32%/+0.72%/+0.23%`；因此当前算术精度上 FP16 更接近 FP32，但仍需最终 ONNX/WebGPU 图级对拍。记录位于 `eval/codec_boundary_4gpu_b512_step0020000_*` 与 `rollout/codec_boundary_4gpu_b512_step0020000_raw_oracle50*/`；这些结果支持继续完整训练，不支持提前宣布视觉通过。

25k checkpoint 的完整恢复状态已经产生并通过只读反序列化检查：模型文件含 78 个 tensor，optimizer 含 state/param groups，scheduler 状态完整，EMA checkpoint 含 decay/update-count/shadow，sampler 含 seed/epoch，四个 rank 各自具备 Python/NumPy/Torch/CUDA RNG 状态。独立集 raw/EMA 的 decoder total 为 `0.26101/0.26673`，FK 为 `0.01342/0.01376 m`，joint jerk 为 `0.00881/0.00908`，boundary joint jerk 为 `0.01024/0.01056`，首帧 joint seam error 为 `0.36941/0.37787 m/s`；相对 20k raw，FK、joint jerk、boundary joint jerk 和 seam 再改善约 `5.0%/6.4%/6.5%/4.6%`。同五 seed 的 acceleration/jerk/high-frequency/seam 均值从 20k 的 `1.136×/1.521×/2.033×/1.723×` 降到 `1.120×/1.460×/1.858×/1.688×`，最坏 seed 为 `1.225×/1.733×/2.250×/2.031×`；每个 seed 的四项仍全部改善。五 seed phase-1 acceleration/jerk 均值从 20k 的 `1.302×/1.940×` 降到 `1.274×/1.848×`，四个 phase 的 jerk 均值全部下降；phase-1 仍是残余硬指标，但不是单 seed 偶然。训练收益开始边际收窄但尚未平台，因此继续既定 100k，不提前停止。记录位于 `state/step-0025000/`、`eval/codec_boundary_4gpu_b512_step0025000_*` 与 `rollout/codec_boundary_4gpu_b512_step0025000_raw_oracle50*/`。

30k checkpoint 仍有跨独立集与跨路径的一致收益。独立集 raw/EMA 的 decoder total 为 `0.25124/0.25693`，FK 为 `0.01289/0.01320 m`，joint jerk 为 `0.00837/0.00861`，boundary joint jerk 为 `0.00972/0.01001`，首帧 joint seam error 为 `0.35556/0.36362 m/s`；raw 继续全面优于 EMA。相对 25k raw，FK、joint jerk、boundary joint jerk 和 seam 再改善约 `4.0%/5.0%/5.1%/3.7%`。同五 seed acceleration/jerk/high-frequency/seam 均值从 `1.120×/1.460×/1.858×/1.688×` 降至 `1.110×/1.413×/1.758×/1.668×`，最坏 seed 为 `1.205×/1.675×/2.110×/2.001×`；五条路径四项全部下降。五 seed phase-1 acceleration/jerk 均值从 `1.274×/1.848×` 降至 `1.248×/1.773×`，phase-1 jerk 最坏值从 `2.297×` 降至 `2.209×`。边际收益继续收窄但未反转或平台，训练继续。新增 `summarize_codec_sweep.py` 只读汇总 raw/EMA、五 seed rollout、频域和 token-phase，并自动计算相邻 checkpoint 变化；20k/25k/30k 已固化到 `eval/codec_boundary_4gpu_b512_sweep_020k_030k.json`，避免后续手工抄数造成选模误差。原始记录位于 `eval/codec_boundary_4gpu_b512_step0030000_*` 与 `rollout/codec_boundary_4gpu_b512_step0030000_raw_oracle50*/`。

35k checkpoint 延续同一趋势。独立集 raw/EMA 的 decoder total 为 `0.24300/0.24866`，FK 为 `0.01244/0.01274 m`，joint jerk 为 `0.00802/0.00825`，boundary joint jerk 为 `0.00932/0.00958`，首帧 joint seam error 为 `0.34414/0.35204 m/s`；相对 30k raw，FK、joint jerk、boundary joint jerk 和 seam 分别再改善约 `3.5%/4.1%/4.1%/3.2%`。五 seed acceleration/jerk/high-frequency/seam 均值为 `1.101×/1.384×/1.682×/1.620×`，最坏 seed 为 `1.185×/1.628×/2.013×/1.933×`；相对 30k 均值，长期 jerk/high-frequency/seam 分别再降 `2.1%/4.3%/2.9%`。phase-1 jerk 五 seed 均值为 `1.715×`、最坏为 `2.113×`，仍未过线。统一 leaderboard 已更新到 `eval/codec_boundary_4gpu_b512_sweep.json`；原始证据位于 `eval/codec_boundary_4gpu_b512_step0035000_*` 与 `rollout/codec_boundary_4gpu_b512_step0035000_raw_oracle50*/`。

40k checkpoint 继续稳定改善。独立集 raw/EMA 的 decoder total 为 `0.23611/0.24161`，FK 为 `0.01207/0.01236 m`，joint jerk 为 `0.00774/0.00795`，boundary joint jerk 为 `0.00898/0.00923`，首帧 joint seam error 为 `0.33495/0.34237 m/s`；相对 35k raw，FK、joint jerk、boundary joint jerk 和 seam 再改善约 `3.0%/3.6%/3.6%/2.7%`。五 seed acceleration/jerk/high-frequency/seam 均值为 `1.096×/1.357×/1.621×/1.602×`，最坏 seed 为 `1.171×/1.606×/1.937×/1.918×`；相对 35k 的长期 jerk/high-frequency/seam 再降 `1.9%/3.7%/1.1%`。phase-1 jerk 均值为 `1.675×`、最坏为 `2.055×`。高频最坏 seed 首次低于 `2×`，但 phase-1 jerk 最坏仍略高于 `2×`，因此只记录进展，不宣称质量通过。leaderboard 与原始证据位于 `eval/codec_boundary_4gpu_b512_sweep.json`、`eval/codec_boundary_4gpu_b512_step0040000_*` 和 `rollout/codec_boundary_4gpu_b512_step0040000_raw_oracle50*/`。

45k checkpoint 的独立集 raw/EMA decoder total 为 `0.23035/0.23556`，FK 为 `0.01173/0.01202 m`，joint jerk 为 `0.00752/0.00770`，boundary joint jerk 为 `0.00872/0.00894`，首帧 joint seam error 为 `0.32570/0.33377 m/s`；相对 40k raw，FK、joint jerk、boundary joint jerk 和 seam 分别再改善约 `2.8%/2.9%/2.9%/2.8%`。五 seed acceleration/jerk/high-frequency/seam 均值为 `1.091×/1.340×/1.580×/1.581×`，最坏 seed 为 `1.167×/1.578×/1.885×/1.896×`；相对 40k 的长期 jerk/high-frequency/seam 再降 `1.3%/2.5%/1.3%`。phase-1 jerk 均值为 `1.645×`、最坏为 `2.015×`，已接近但仍略高于 `2×`。统一 leaderboard 与原始证据已更新。

50k checkpoint 的第二份完整恢复状态已验证：EMA update count 精确为 `50,000`，sampler epoch 为 `782`，optimizer/scheduler 与四 rank RNG 文件均存在且可反序列化。独立集 raw/EMA decoder total 为 `0.22523/0.23034`，FK 为 `0.01145/0.01173 m`，joint jerk 为 `0.00732/0.00750`，boundary joint jerk 为 `0.00848/0.00870`，首帧 joint seam error 为 `0.31863/0.32595 m/s`；相对 45k raw，FK、joint jerk、boundary joint jerk 和 seam 再改善约 `2.4%/2.6%/2.7%/2.2%`。五 seed acceleration/jerk/high-frequency/seam 均值为 `1.083×/1.317×/1.541×/1.557×`，最坏 seed 为 `1.150×/1.547×/1.834×/1.865×`；相对 45k 的长期 jerk/high-frequency/seam 再降 `1.7%/2.5%/1.5%`。phase-1 jerk 均值为 `1.604×`、最坏为 `1.959×`，最坏值首次低于 `2×`；但高频和 seam 仍明显高于 teacher，训练继续。leaderboard 与原始证据已更新至 50k。

55k checkpoint 继续采用相同 1024-window validation 和五个固定 50-window path seed。独立集 raw/EMA decoder total 为 `0.22083/0.22578`，FK 为 `0.01119/0.01147 m`，joint jerk 为 `0.00716/0.00732`，boundary joint jerk 为 `0.00830/0.00849`，首帧 joint seam error 为 `0.31022/0.31855 m/s`，raw 仍一致优于 EMA。相对 50k raw，FK、joint jerk、boundary joint jerk 和首帧 seam 分别再改善 `2.30%/2.19%/2.21%/2.64%`。五 seed acceleration/jerk/high-frequency/seam 均值为 `1.0808×/1.3040×/1.5076×/1.5491×`，最坏 seed 为 `1.1405×/1.5253×/1.7845×/1.8517×`；phase-1 jerk 均值/最坏为 `1.5861×/1.9346×`。相对 50k 的长期 jerk/high-frequency/seam 均值再降 `1.01%/2.16%/0.49%`，但边际收益缩小且高频、seam 仍未过线，所以不提前停止 100k 协议，也不开始 flow/DMD2 阶段。完整 sweep 已更新到 `eval/codec_boundary_4gpu_b512_sweep.json`。

60k checkpoint 的同口径评估也已完成。独立集 raw/EMA decoder total 为 `0.21724/0.22182`，FK 为 `0.01095/0.01123 m`，joint jerk 为 `0.00703/0.00717`，boundary joint jerk 为 `0.00815/0.00831`，首帧 joint seam error 为 `0.30405/0.31182 m/s`；raw 继续全面优于 EMA。相对 55k raw，FK、joint jerk、boundary joint jerk 和首帧 seam 分别改善 `2.08%/1.80%/1.82%/1.99%`。五 seed acceleration/jerk/high-frequency/seam 均值为 `1.0775×/1.2912×/1.4893×/1.5172×`，最坏 seed 为 `1.1362×/1.5026×/1.7460×/1.8003×`；phase-1 jerk 均值/最坏为 `1.5747×/1.9084×`。相对 55k，长期 jerk/high-frequency/seam 均值再降 `0.98%/1.21%/2.05%`，说明继续训练仍有可测收益；但最坏 HF/seam 仍约 `1.75×/1.80×`，不构成视觉通过。

65k checkpoint 继续收敛。独立集 raw/EMA decoder total 为 `0.21402/0.21837`，FK 为 `0.01077/0.01103 m`，joint jerk 为 `0.00690/0.00703`，boundary joint jerk 为 `0.00798/0.00815`，首帧 joint seam error 为 `0.29743/0.30572 m/s`。相对 60k raw，FK、joint jerk、boundary joint jerk 和首帧 seam 分别改善 `1.73%/1.93%/1.99%/2.18%`。五 seed acceleration/jerk/high-frequency/seam 均值为 `1.0742×/1.2781×/1.4557×/1.5060×`，最坏 seed 为 `1.1300×/1.4963×/1.7106×/1.7807×`；phase-1 jerk 均值/最坏为 `1.5353×/1.8455×`。相对 60k，长期 jerk/HF/seam 均值再降 `1.02%/2.26%/0.74%`；趋势仍为严格 paired 改善，但 seam 尚未接近 teacher，因此继续原定 100k，不用训练批 loss 代替最终判断。

70k checkpoint 的静态 raw/EMA decoder total 为 `0.21112/0.21532`，FK 为 `0.01059/0.01084 m`，joint jerk 为 `0.00677/0.00692`，boundary joint jerk 为 `0.00784/0.00802`，首帧 joint seam error 为 `0.29196/0.29993 m/s`；相对 65k raw 继续改善 `1.35%/1.64%/1.81%/1.84%/1.84%`。五 seed acceleration/jerk/HF/seam 均值为 `1.0722×/1.2695×/1.4417×/1.5124×`，最坏 seed 为 `1.1289×/1.4894×/1.6939×/1.7873×`。其中 jerk/HF 均值相对 65k 改善 `0.67%/0.96%`，且逐 seed paired delta 是 `5/5` 全部改善；但 seam 均值小幅回退 `0.43%`，并非单个离群路径，而是 `4/5` seed 回退，phase-1 jerk 均值也从 `1.5353×` 轻微回到 `1.5366×`。这是首个长序列局部反转。幅度尚小，不能据单点提前停训，也不能隐瞒为单调收敛；继续到 100k 后按全部 checkpoint、更多 seed 和视觉共同选模。

75k checkpoint 同时保存并验证了第三份完整恢复状态：model 78 个 tensor、optimizer 54 个 state、scheduler、sampler epoch `1172` 和四 rank RNG 均可反序列化；EMA decay 为 `0.9999`、update count 精确为 `75,000`、54 个 shadow tensor 全 finite。独立集 raw/EMA decoder total 为 `0.20858/0.21256`，FK 为 `0.01042/0.01067 m`，joint jerk 为 `0.00667/0.00681`，boundary joint jerk 为 `0.00772/0.00789`，首帧 seam 为 `0.28645/0.29467 m/s`。五 seed acceleration/jerk/HF/seam 均值为 `1.0694×/1.2582×/1.4138×/1.5083×`，最坏 seed 为 `1.1256×/1.4672×/1.6535×/1.7925×`，phase-1 jerk 均值/最坏为 `1.5111×/1.8226×`。相对 70k，长期 jerk/HF 均为 `5/5` seed 改善，均值下降 `0.89%/1.94%`；seam 均值只改善 `0.27%`，逐 seed 为 `3/5` 改善、`2/5` 回退，且总体仍略差于 65k 的 `1.5060×`。因此最终必须按多指标选择 checkpoint，不能机械采用最后一步。

80k checkpoint 的独立集 raw/EMA decoder total 为 `0.20597/0.21004`，FK 为 `0.01026/0.01051 m`，joint jerk 为 `0.00658/0.00672`，boundary joint jerk 为 `0.00761/0.00778`，首帧 seam 为 `0.28089/0.28938 m/s`；raw 仍全面优于 EMA。五 seed acceleration/jerk/HF/seam 均值为 `1.0693×/1.2501×/1.4078×/1.4884×`，最坏 seed 为 `1.1250×/1.4569×/1.6359×/1.7527×`，phase-1 jerk 均值/最坏为 `1.4988×/1.8109×`。相对 75k，seam 均值下降 `1.32%`，逐 seed 为 `4/5` 改善，首次低于 `1.50×`；jerk/HF 均值下降 `0.65%/0.42%`，也都是 `4/5` seed 改善。困难路径的最坏 HF/seam 仍约 `1.64×/1.75×`，因此 80k 只是当前新 Pareto 候选，不是质量过线或提前终止依据。

85k checkpoint 再次确认 80k 后的改善不是单 seed 偶然。独立集 raw/EMA decoder total 为 `0.20385/0.20769`，FK 为 `0.01012/0.01036 m`，joint jerk 为 `0.00650/0.00663`，boundary joint jerk 为 `0.00751/0.00767`，首帧 seam 为 `0.27633/0.28440 m/s`。五 seed acceleration/jerk/HF/seam 均值为 `1.0657×/1.2472×/1.3903×/1.4728×`，最坏 seed 为 `1.1183×/1.4556×/1.6135×/1.7466×`，phase-1 jerk 均值/最坏为 `1.4842×/1.7894×`。相对 80k，HF 和 seam 均值再降 `1.25%/1.04%`，且都是 `5/5` 路径改善；jerk 均值只降 `0.23%`，逐 seed 为 `3/5` 改善。85k 是当前最强候选，但困难路径 seam 仍约 `1.75×`，不能跳过 90/95/100k 与最终视觉验证。

90k checkpoint 的独立集 raw/EMA decoder total 为 `0.20175/0.20552`，FK 为 `0.00998/0.01022 m`，joint jerk 为 `0.00644/0.00655`，boundary joint jerk 为 `0.00744/0.00758`，首帧 seam 为 `0.27189/0.27975 m/s`；raw FK 首次低于 `10 mm`。五 seed acceleration/jerk/HF/seam 均值为 `1.0653×/1.2446×/1.3911×/1.4615×`，最坏 seed 为 `1.1164×/1.4414×/1.6135×/1.6972×`，phase-1 jerk 均值/最坏为 `1.4786×/1.7773×`。相对 85k，jerk/seam 均值改善 `0.21%/0.77%`，seam 为 `4/5` seed 改善且最坏值明显降低；但 HF 均值轻微回退 `0.059%`，逐 seed 为 `3/5` 改善、`2/5` 回退。85k 与 90k 因而构成很小的 HF-versus-seam Pareto 权衡，不能只按 step 选模。

等待 decoder 完成期间已完成下一阶段 flow 训练器预审，但没有提前启动正式 flow。原 `train_flow.py` 硬编码 cosine，且 endpoint 始终是 t=1 单步，不能称为“先 4-step、再 2-step”。现已把默认改为 constant LR `1e-5`，加入显式 exact-t1/high-noise mixture，并实现从 `t=1→0` 的 uniform backward-Euler solver；paired endpoint、decoder 与物理 loss 可端到端穿过 4/2 个 stage，静态 evaluator、无限 rollout 和 runtime 也都记录真实 NFE。旧 `denoise_once` 与新 NFE1 在真实权重上 bitwise 一致（max abs 0），NFE2/4 forward finite；solver/sampling 5 个单测与 shape 测试通过。GPU 2 的 1-step NFE4 BF16 工程 smoke 完成全 loss 反传、optimizer、EMA 和状态保存，105 个权重 tensor 全 finite，但裁剪前 grad norm 为 `106`，正式训练前仍需梯度尺度与四卡吞吐审计。该目录已写明 `ENGINEERING_SMOKE_ONLY.md`，不得作为质量结果。

同一时点重新做了不依赖显存总量的四卡归属审计：PID `151704/151705/151706/151707` 分别是 `LOCAL_RANK=0/1/2/3`，`WORLD_SIZE=4`，共同设置 `CUDA_VISIBLE_DEVICES=3,4,5,6`；NVML 将四个计算 PID 分别映射到物理 GPU `3/4/5/6`。连续 8 秒采样中除一次数据/同步间隙外，四卡 SM 通常为 `92%–95%`、功耗约 `227–235 W/卡`，日志从 54,250 实际推进到 55,500。机器上其余驻留显存属于别的测试任务，不能也不会再被作为本训练占卡证据；今后的健康结论只由 rank/PID 映射、持续 SM/功耗和 optimizer step 增量共同给出。

在 94k 附近再次按相同口径复核：上述四个 rank/PID 仍分别驻留物理 GPU `3/4/5/6`；连续 6 秒逐卡采样的 SM 为 `89%–97%`、功耗为 `196–237 W/卡`，没有把 GPU `0/1/2/7` 上其他测试任务的显存计入。训练日志从 94,000 推进到 94,250 的 wall time 为约 `43.2 s`，即约 `5.78 optimizer step/s`；global batch 2048 对应约 `11.8k windows/s`。这里再次明确：显存驻留只能说明有 context/缓存，既不能证明任务正在计算，也不能据此判定一张卡不可用于新训练；后续选卡和健康检查必须区分“其他任务的驻留显存”与“活跃 SM/功耗”。

## 21. decoder 100k 收口与 NFE4 四卡正式训练启动

decoder 四卡任务已正常完成完整 `100,000` optimizer steps，累计 `204.8M` sample draws，wall time 约 `4.85 h`，不是按样本数折算出来的伪 step。最终完整状态的 model/raw encoder/raw decoder/optimizer/scheduler/EMA/sampler/四 rank RNG 均已只读验证：全部 tensor finite，54 组 Adam state 的 step 均为 `100000`，scheduler `last_epoch=100000` 且 LR 保持 `3e-6`，EMA update count 为 `100000`。raw 继续优于 EMA，因此后续 flow 固定使用 `weights/step-0100000/decoder.safetensors`。

100k raw 的独立集 decoder total/FK/foot-slide/joint jerk/boundary jerk/首帧 joint seam 分别为 `0.197999 / 9.733 mm / 0.010043 / 0.006285 / 0.007258 / 0.263393 m/s`。五个固定 50-window seed 的 acceleration/jerk/high-frequency/seam 均值为 `1.0594×/1.2254×/1.3583×/1.4552×`，最坏 seed 为 `1.1066×/1.4037×/1.5566×/1.7078×`；phase-1 jerk 均值/最坏为 `1.4555×/1.7470×`。95k→100k 的 acceleration/HF/seam 为 `5/5` seed 改善，jerk 为 `4/5` 改善；四项均值分别下降约 `0.33%/0.63%/1.78%/2.02%`。因此 100k raw 是本轮 codec 的明确选择，但困难路径 seam 仍高于 teacher，不能把 codec 通过等同于整套动画视觉通过。完整 sweep 已更新到 `eval/codec_boundary_4gpu_b512_sweep.json`。

正式 flow 前的逐 loss 梯度审计发现，原候选 physical seam 系数 `0.1` 在 teacher/on-policy batch 上产生约 `49.4/38.6` 的加权梯度范数，而 on-policy velocity+endpoint 核心梯度仅约 `17.6`；原因是 m/s 单位天然含 20 FPS 放大。将该系数校准为 `0.02` 后，teacher batch 的 core/physical/aux/combined 梯度范数为 `31.18/9.89/20.12/39.84`，combined 对 core cosine 为 `0.867`；on-policy 为 `17.62/7.71/15.29/24.54`，cosine 为 `0.785`。其余系数保持 decoder/path/normalized-seam/root-temporal/quality 为 `0.25/0.05/0.1/0.1/0.1`，foot-slide 在 quality 内为 `0.05`。这只是梯度尺度校准，不把“范数相近”误当成质量证明。

数据读取已改成 batch-local shard mixture：on-policy 4096-window 主集与 131072-window replay 按 `0.5/0.5` 采样，但一个 batch chunk 只顺序读取单一 shard，避免旧 per-sample 随机跨 shard seek；训练只加载实际使用的 10 个字段。新的 sampler 可随 Accelerate 完整 checkpoint，相关数据/solver 测试共 9 项通过。四卡 NFE4、每卡 batch 512、global batch 2048、每步计算完整 decoder/quality/seam loss 的 200-step 工程扫描正常结束；step 20→200 为 `76.479 s`，约 `2.3536 optimizer step/s`、`4820.1 samples/s`，全部日志 scalar finite。该目录已写入 `THROUGHPUT_ONLY.md`，不得当作质量模型。

随后启动正式 NFE4 50k 训练，目录为 `flow_nfe4_4gpu_b512_const3e6_50k_phys002/`：width-512、5+2 blocks、NFE4 uniform backward Euler、每卡 batch 512/global 2048、固定 LR `3e-6`、BF16 forward/backward、FP32 AdamW/EMA、weight decay 0、无 warmup、无 cosine，70% exact-t1 + 25% high-noise + 5% uniform，每步完整辅助 loss，每 5k 保存可恢复状态。首次命令因漏传仓库内层 `PYTHONPATH` 在 import 阶段立即退出，未进入训练也未产生 step；补正后四个训练 PID `1114008/1114009/1114010/1114011` 的 `LOCAL_RANK=0/1/2/3`、`WORLD_SIZE=4`，共同设置 `CUDA_VISIBLE_DEVICES=3,4,5,6`，NVML 分别映射物理 GPU `3/4/5/6`。连续 12 秒有效采样中四卡多数为 `89%–98% SM`、约 `238–274 W/卡`，optimizer step 已从 1 推进到 200，LR 恒定且所有记录 finite。机器上已有显存继续明确归属于其他测试任务；本段四卡结论只由 rank/PID 映射、持续 SM/功耗和 optimizer-step 增量建立。

为防止后续只按训练 loss 判断，已用最终 decoder100k 对正式 flow 初始化补齐同口径 step-0 基线。独立 teacher-val 1024-window 的 NFE4 endpoint total/FK/path error 为 `0.78593 / 0.23893 m / 0.16037 m`。五个固定 50-window seed 的 waypoint/root drift/FK/foot-slide/joint seam 均值为 `1.707 m / 2.017 m / 2.032 m / 1.284 m/s / 2.588 m/s`；root-relative joint acceleration P95、jerk P95、高频和 seam 相对 teacher 的均值为 `2.793×/5.410×/13.520×/8.806×`，最坏为 `3.100×/6.063×/15.634×/10.890×`。这些结果确认初始化即使改成 NFE4 仍远未达到动画质量，正式 5k/10k checkpoint 必须在相同五 seed 上证明下降，不能把 NFE4 或 finite 本身当作成功。

DMD2/对抗阶段与当前长程 supervised fitting 明确分离。旧 `train_dmd2.py` 默认 `10k` generator updates、generator LR `2e-5`、score/critic LR `1e-4` 且 cosine，容易在短程分布蒸馏中破坏已拟合的 teacher 行为，现已改为保守短程默认：`2000` generator updates、generator/score/critic LR `1e-6/5e-6/2e-6`、constant、无 scheduler warmup、DMD/adv weight `0.05/0.005`、EMA `0.995`、每 100 generator step 保存、runtime 上限 2h。参数校验对 `>5000` generator updates 或任一 LR `>1e-5` 直接拒绝。正式执行仍按 `200→500→1000→最多2000` 做固定多 seed paired rollout，任何 waypoint/FK/foot-slide/seam/jerk/HF 或视觉反转立即停止；critic/score loss 下降不允许成为延长训练的理由。

## 22. NFE4 5k 多种子闭环检查：方向正确，但尚未达到质量线

正式 NFE4 的 5k raw/EMA 已在训练不中断时完成同一批 1024-window 静态验证和五个固定 seed、每个 50 次续写的 faithful rollout。汇总如下：

| checkpoint | static FK | waypoint mean | rollout FK | foot slide | body jerk / teacher | body HF / teacher | joint seam / teacher |
|---|---:|---:|---:|---:|---:|---:|---:|
| step 0 raw | 238.93 mm | 1.707 m | 2.032 m | 1.284 m/s | 5.410× | 13.520× | 8.806× |
| **step 5k raw** | **211.69 mm** | 0.178 m | **1.274 m** | **0.643 m/s** | **2.532×** | **5.156×** | **3.775×** |
| step 5k EMA | 212.31 mm | **0.158 m** | 1.280 m | 0.650 m/s | 2.601× | 5.372× | 3.874× |

step 0→5k raw 的五 seed paired comparison 中，waypoint、平均 root drift、FK、foot slide、joint seam、root/body acceleration、jerk 和高频指标均为 `5/5` 改善；只有 final root drift 为 `3/5` 改善、`2/5` 回退。均值相对变化为 waypoint `-89.6%`、FK `-37.3%`、foot slide `-49.9%`、body jerk `-53.2%`、body HF `-61.9%`、joint seam `-57.1%`。静态 endpoint/FK/path 也分别改善约 `4.2%/11.4%/25.8%`。因此训练方向和 loss 校准得到了跨路径证据，不是只看单次训练 loss。

这仍不是可用模型：5k raw 的 root acceleration/jerk/HF 均值仍为 teacher 的 `7.37×/15.84×/8.70×`，rollout FK 仍为 `1.27 m`，final root drift 还存在两条路径反弹。EMA 只在 waypoint 均值上略好，其余主要平滑、FK、foot 和 seam 均值普遍略差，所以当前选 raw 作为 5k 候选，继续 supervised NFE4，不提前切 NFE2 或 DMD2。记录时正式日志已推进到 step 6000，LR 仍为固定 `3e-6`；5k 完整 optimizer/scheduler/EMA/sampler/四 rank RNG 状态均已验证 finite。

机器汇总位于 `eval/flow_nfe4_formal_sweep.json`，原始静态结果位于 `eval/flow_nfe4_formal_step0005000_{raw,ema}_fp32.json`，五 seed rollout/jitter 位于对应 `rollout/flow_nfe4_formal_step0005000_*` 和 `eval/*_jitter_seed*.json`。本次检查未安装任何包；环境中没有 `pytest` 时没有擅自补装，而是继续使用现有 `unittest`，也未修改用户已有的 `ardy/model/llm2vec/llm2vec.py`。

## 23. 路线纠正：取消 NFE4→NFE2 课程，直接优化部署用 NFE1

终态浏览器生成器只调用一次，因此继续优化 NFE>1 不能作为 NFE1 质量的替代证据。已对正式 NFE4 step-5000 raw 做了同权重、同 decoder 的直接单步复测：它在 NFE4 静态评估中 FK 约为 `0.212 m`，但强制按最终部署的 NFE1 运行后，FK 变为 `0.244 m`、endpoint 约为 `0.784`；相同权重的单步 joint jerk 约为 `0.143`，明显高于四步组合的 `0.030`。这证明五千步收益主要存在于四次调用的数值轨迹中，不会自动转移给单步生成器。

因此 NFE4 任务已主动停止于日志 step 7100，最后完整可恢复 checkpoint 为 step 5000；目录 `flow_nfe4_4gpu_b512_const3e6_50k_phys002/STOPPED_FOR_DIRECT_NFE1_PIVOT.md` 已明确它只是历史对照。不再启动 NFE2，也不再用 NFE4/NFE2 作为 DMD2 的中间 teacher-fitting 课程。第 9 节和第 21–22 节中“继续 NFE4 后转 NFE2”的文字只保留为历史记录，本节正式取代其后续执行意义。

新的固定基线是 round3 step-400 raw 的原生单步权重，并使用 decoder100k raw。它在独立静态集的 endpoint/FK/path 约为 `0.753/0.231 m/0.133 m`。五个固定 seed、50-window、不开惯性化的忠实闭环基线为：waypoint `1.785 m`、rollout FK `2.388 m`、foot slide `2.581 m/s`；root jerk、body jerk、body HF 和 joint seam 均值分别是 teacher 的 `86.38×/14.81×/23.62×/20.25×`。所有后续 DMD2/adv checkpoint 都必须与这五条路径 paired 对比，不允许用静态 loss 替代长闭环。

DMD2 实现已按单步目标重新审计。修正了 distribution-matching 归一化的参照量：`p_real`/`p_fake` 现在是 generated sample 分别减 teacher/fake-score 预测的 x0，不再错用加噪后 `x_t`。fake-score/critic 与 generator 的 TTUR 固定为 `5:1`；DMD 查询对最高噪声步偏置，paired 项只保留 `0.05` 的条件锚点，adversarial 起始为 `0.005`，所有学习率为 constant。critic 输入扩展为 `658` 维，显式包含 FK joint 的 velocity/acceleration/jerk 和 contact，teacher/fake-score/critic 都不导出到浏览器。

四卡工程探针先证明仅 2 次 fake-score warmup 不可用：首个 generator step 的 fake-score total 仍约 `2.85`，teacher/fake x0 L1 约 `3.19`，DMD 梯度平均绝对值约 `9.07`，generator 裁剪前梯度范数约 `27.3`，因此该权重已标为 `ENGINEERING_SMOKE_ONLY`。随后在物理 GPU `3,4,5,6` 上使用四进程、每卡 batch 64/global 256 完成 200 次辅助网络预热 + 1 次 generator 更新：fake-score total 从 `3.041` 降到 `0.328`，teacher/fake x0 L1 降到 `0.848`，DMD 梯度平均绝对值降到 `1.931`，generator 裁剪前梯度范数降到 `6.48`；critic total 仍约 `1.27`，未饱和。这只证明直接 NFE1 的 DMD 梯度已进入可试验范围，不声称 1 个 generator step 有质量收益。

后续协议只比较三个直接 NFE1 分支：DMD-only、DMD + 低权重 conditional temporal adversarial、以及前者 + 弱 paired conditional anchor。先到 200 generator updates 做完同 seed 静态/长闭环/视觉消融，只有 paired 主指标方向正确才续到 `500→1000→最多 2000`。这些 step 只计 generator optimizer update，不把五次 score/critic update 或 global sample 数换算成虚假训练步数。本路线更正未安装任何包，也未修改用户已有的 `ardy/model/llm2vec/llm2vec.py`。

### 23.1 直接 NFE1 首轮选模与 500-step 早停

三个分支均从同一个 round3 step-400 NFE1 权重开始，四卡 global batch 256，generator/score/critic 固定 LR 为 `1e-6/5e-6/2e-6`，TTUR 为 `5:1`，无 cosine。200 generator updates 后，带弱 conditional/几何锚点的 adversarial 分支 EMA 是首轮相对最好点；其五 seed、50-window、无惯性化均值为 waypoint `0.849 m`、root drift `1.883 m`、final root drift `2.450 m`、FK `1.912 m`、foot slide `2.422 m/s`、root/joint seam `3.253/5.859 m/s`。root acceleration/jerk/HF 为 teacher 的 `38.93×/85.94×/14.33×`，body acceleration/jerk/HF 为 `7.26×/14.58×/21.73×`。它优于原生 NFE1 基线的条件跟随与部分长期指标，但抖动和接缝仍远未达到可用线，因此只称“当前相对最好”，不能称为成功模型。

该分支从完整 step-200 状态继续到 step 500 后出现明确 Pareto 反转。五 seed waypoint 从 `0.849` 恶化到 `0.932 m`，root drift 从 `1.883` 到 `2.403 m`，FK 从 `1.912` 到 `2.432 m`，且 FK 为 `0/5` seed 改善；foot slide 从 `2.422` 到 `2.442 m/s`，root seam 从 `3.253` 到 `3.337 m/s`。虽然 final root drift、joint seam 和部分频域值小幅改善，但 critic 已接近饱和（最终 real/fake logit 约 `+5.88/-7.16`，generator adversarial 约 `7.15`）。因此 step 500 被拒绝，不再续到 1000；选模点保留 step-200 EMA。

### 23.2 timestep-conditioned diffusion critic 消融

对照 DMD2 官方实现后确认，fake-score 与 classifier/critic 同属 guidance cadence，`5:1` 更新比本身不是错误；真正缺少的是 diffusion-GAN 对真实/生成动作在随机时间步加噪后再判别的机制。为此 critic 增加 10-step timestep embedding，真实和生成动作分别独立加噪，输入仍包含 normalized motion、FK、velocity、acceleration、jerk、contact 和条件。探针显示可分性随噪声单调下降：`t=0` accuracy 约 `0.914`，到 `t=9` 约 `0.508`，说明实现行为符合预期。

然而把 critic 覆盖到完整 `t=0..9` 并训练 200 generator updates 后，单 seed 虽改善了 waypoint/global geometry，却恶化视觉关键指标：EMA waypoint/root drift/final drift/FK 为 `0.954/1.957/0.547/1.977 m`，但 root seam、joint seam 分别升到 `3.279/6.078 m/s`，root acceleration/jerk 为 teacher 的 `47.50×/105.10×`，body jerk 为 `17.38×`。step-100 也没有形成更好折中。该 full-range diffusion critic 分支因此被拒绝，不补做五 seed，也不继续训练。

### 23.3 辅助梯度审计与低噪声 adversarial 配对实验

在全 history batch 上对 step-200 EMA 做逐项反传，当前加权梯度 L2 为：path `0.0149`、root-temporal `0.3575`、decoder `0.3665`、seam `0.2605`、physical-seam `0.4640`、完整 quality `0.1314`，辅助项合计约 `1.0718`。这证明所谓“弱锚点”的实际梯度并不弱，不能机械整体加倍。由此只做一个严格配对：critic 限于 `t=0..1`、critic LR 降为 `1e-6`、adversarial weight 降为 `0.002`；A 保持 quality/root-temporal `0.004/0.004`，B 在近似保持总梯度预算下改为 `0.008/0.002`。两者使用相同 seed、相同初始化和相同 200 次辅助预热，只跑到 100 generator updates。

同一 seed `20260714` 的 50-window 结果如下，均为真实 NFE1、FP32、无惯性化：

| checkpoint | waypoint | root drift | final drift | FK | foot slide | root seam | joint seam | root jerk / teacher | body jerk / teacher |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 旧 clean g200 EMA | 1.082 m | 1.963 m | 2.393 m | 1.991 m | 2.398 m/s | 3.333 m/s | **5.819 m/s** | **98.96×** | **15.94×** |
| low-t A g100 raw | **1.052 m** | **1.701 m** | 3.823 m | **1.717 m** | 2.497 m/s | 3.383 m/s | 5.998 m/s | 106.02× | 16.47× |
| low-t A g100 EMA | 1.352 m | 1.934 m | 3.340 m | 1.948 m | 2.648 m/s | 3.667 m/s | 6.141 m/s | 107.37× | 17.41× |
| low-t B g100 raw | 1.167 m | 1.958 m | 4.123 m | 1.948 m | 2.861 m/s | 3.381 m/s | 6.115 m/s | 112.61× | 18.94× |
| low-t B g100 EMA | 1.352 m | 2.031 m | 4.072 m | 2.022 m | 2.793 m/s | 3.446 m/s | 6.269 m/s | 104.43× | 18.36× |

A/raw 的平均路径与 FK 更好，但 final drift、foot slide、seam、acceleration/jerk 全部反转；B 的 quality/root 权重重平衡进一步恶化。两分支都在 g100 淘汰，不续 g200、不扩到五 seed。这个结果也说明当前问题不是简单把某个平滑项翻倍即可解决。

截至本节，执行路线固定为：只优化部署结构本身的直接 NFE1，不再建立 NFE4→NFE2 课程，也不把多次 solver 调用的成功当成单步成功。当前仅保留 `dmd2_direct_nfe1_4gpu_dmd_adv005_weakanchor04x_g200/weights/step-0000200/flow_ema.safetensors` 作为相对最好研究 checkpoint；它仍不满足动画质量，浏览器候选不得据此更新。上述新实验没有安装任何包，直接使用 `<PY311_ENV>/bin/python`；用户已有的 `ardy/model/llm2vec/llm2vec.py` 未被修改。

### 23.4 官方式 score-backbone adversarial + 均匀 DMD 时间步

进一步审计 DMD2 官方源码后发现两个结构差异。第一，旧实验把 `85%` 的 DMD query 人为集中在最高噪声附近，而官方 distribution matching 在有效区间均匀采样；这会过度强调粗略全局结构。第二，官方 adversarial classifier 读取 fake-score denoiser 的 bottleneck feature，并与 fake-score 联合更新；旧 ARDY 实验则另建 physical critic，在给 normalized motion 加白噪声后再计算 FK/velocity/acceleration/jerk。后者会用时间差分放大注入噪声，critic 的判别依据与一步 latent 分布不完全一致。

实现现已增加 `OneStepFlowStudent.forward(return_features=True)` 和训练专用 `ScoreBackboneCriticHead`。公开生成器 `forward` 与 feature 路径的预测经过逐元素相等测试，浏览器导出图没有新增模块。新 guidance turn 在一个 optimizer step 内联合更新 fake-score epsilon loss 与 `0.01 × classifier loss`；score/head 使用独立固定 LR 参数组 `5e-6/2e-6`。generator 仍只做一次 `x1→x0`，DMD 改为 `t=0..8` 均匀采样，adversarial 使用全 `t=0..9`，没有任何 NFE>1 过程。

第一次四卡工程 smoke 因 DDP 不代理自定义方法而在第一次 guidance forward 同步退出，未发生 optimizer update；接口改成标准 `forward(return_features=True)` 后，第二次四卡 smoke 的反传、两个 LR 参数组、EMA、状态和 safetensors 保存全部通过。200 次 guidance-only 校准后，fake-score total 从 `3.02` 降到约 `0.31`；独立 critic profile 的 balanced accuracy 从 `t=0` 的 `0.750`、`t=4` 的 `0.648` 单调降到 `t=8/9` 的 `0.482/0.492`，没有 clean saturation 或高噪声伪可分性。首个 generator update 的 DMD gradient abs 约 `0.99`，generator grad norm 约 `0.42`，因此只启动一个正式配置。

正式任务 `dmd2_direct_nfe1_4gpu_scorebackbone_uniformt_adv003_anchor04x_g100` 使用物理 GPU `3,4,5,6`、global batch 256、200 次 guidance 预热、`5:1` TTUR、generator/score/head 固定 LR `1e-6/5e-6/2e-6`、DMD/adv `0.05/0.003`，并保持上一相对最好分支的条件/几何锚点不变，以便单变量归因。最终 g100 fake-score total 为 `0.287`，critic real/fake logits 为 `+0.386/-0.395`，没有饱和。

静态验证表明该方法比旧 g200 更少破坏初始化器：g50 EMA 的 endpoint/FK/path/joint jerk 为 `0.764/0.234 m/0.134 m/0.118`，g100 EMA 为 `0.775/0.236 m/0.135 m/0.118`；旧 clean g200 EMA 的对应值为 `0.814/0.247 m/0.139 m/0.120`。但同一 seed 的 50-window 结果仍未同时改善：

| checkpoint | waypoint | final drift | FK | foot slide | root seam | joint seam | root jerk / teacher | body jerk / teacher |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 旧 clean g200 EMA | **1.082 m** | **2.393 m** | 1.991 m | **2.398 m/s** | **3.333 m/s** | **5.819 m/s** | **98.96×** | 15.94× |
| score-backbone g50 EMA | 1.425 m | 3.541 m | 2.004 m | 2.605 m/s | 3.695 m/s | 5.900 m/s | 100.02× | **15.63×** |
| score-backbone g100 raw | 1.100 m | 3.322 m | 1.992 m | 2.642 m/s | 3.724 m/s | 6.600 m/s | 112.06× | 18.11× |
| score-backbone g100 EMA | 1.143 m | 3.464 m | **1.988 m** | 2.592 m/s | 3.777 m/s | 6.575 m/s | 109.04× | 17.50× |

因此 g50/g100 都未通过单 seed gate：共享 score backbone 与均匀时间步修正了 DMD2 方法对齐，并明显减缓静态质量破坏，但仍把路径收益换成 final drift、foot、seam 和 root jerk 回退。正式任务停止于 g100，不续到 g200、不补五 seed，也不更新浏览器模型。这个负结果不推翻“直接 NFE1 DMD2/adv”路线；它排除了“旧 critic 形式和高噪声偏置是唯一根因”的假设。后续若继续，必须直接针对一步输出的时域/频域分布梯度设计，而不是恢复 NFE 课程或机械增加 generator step。

本轮新增的 shape/finite、公开 forward 等价、uniform sampler 和输入梯度测试通过，原有 9 项数据/solver 单测仍通过。所有运行均使用已有 py311 环境，没有执行 `conda` 命令、没有安装包，也没有修改 `ardy/model/llm2vec/llm2vec.py`。

### 23.5 图内 root temporal projection 与直接 NFE1 重训

旧 clean g200 EMA 的噪声主要集中在一次输出的 40 帧 root 序列。为避免依赖前端滤波，生成器端点加入固定、可微、零参数的 `[1,4,6,4,1]/16` 时间投影；首尾帧每次投影都原样保留，使 history seam 与末端 waypoint 仍可被 loss 直接控制。它位于 `OneStepFlowStudent.denoise_steps()` 内，训练、PyTorch 部署和 ONNX 导出共用同一实现，不增加 NFE，也不进入 renderer。旧 g200 EMA 的四次投影先做五 seed 后验消融：相对不投影，foot slide、root/joint seam、root acceleration/jerk/HF、body acceleration/jerk/HF 都是 `5/5` seed 改善；均值从 `2.422/3.253/5.859 m/s、38.93×/85.94×/14.33×、7.26×/14.58×/21.73×` 降到 `0.728/1.433/1.845 m/s、1.91×/1.42×/0.51×、1.20×/2.08×/7.59×`。代价是 waypoint 从 `0.849` 退到 `1.493 m`，说明结构投影确实消除了高频自由度，但需要在投影图内重新学习条件跟随。

梯度审计后，正式任务 `dmd2_direct_nfe1_4gpu_rootproj4_scorebackbone_uniformt_path02_g100` 从旧 clean g200 EMA 开始，训练与部署都固定为直接 NFE1 + projection×4。物理 GPU `3,4,5,6`、global batch 256、200 次 guidance 预热、`5:1` TTUR，generator/score/head 固定 LR `1e-6/5e-6/2e-6`、无 warmup、无 cosine；DMD/adv 为 `0.05/0.003`，path/decoder/quality/seam/physical 权重为 `0.02/0.01/0.004/0.001/0.0002`。g50/g100 raw/EMA 均完成独立 1024-window 静态验证与同 seed 50-window gate；EMA 在百步内明显滞后，最终只将 g50/g100 raw 扩展到五 seed。

五 seed、50-window、FP32、无前端惯性化的均值如下：

| checkpoint | waypoint | mean drift | final drift | FK | foot slide | root seam | joint seam | root jerk / teacher | body jerk / teacher | body HF / teacher |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 旧 g200 EMA + projection×4 | 1.493 m | 2.045 m | 1.345 m | 2.083 m | 0.728 m/s | 1.433 m/s | 1.845 m/s | **1.422×** | **2.079×** | 7.595× |
| 新 g50 raw | 1.302 m | 1.930 m | 2.277 m | 1.965 m | **0.682 m/s** | 1.445 m/s | 1.837 m/s | 1.507× | 2.081× | **6.075×** |
| 新 g100 raw | **1.097 m** | **1.893 m** | **1.075 m** | **1.929 m** | 0.692 m/s | **1.366 m/s** | **1.794 m/s** | 1.608× | 2.171× | 6.376× |

g100 raw 相对投影基线的 waypoint/mean drift/final drift/FK/foot/root seam/joint seam/body HF 分别改善 `26.53%/7.46%/20.07%/7.39%/5.01%/4.69%/2.75%/16.05%`，waypoint 为逐 seed `5/5` 改善；但 root acceleration/jerk/HF 分别回退 `15.56%/13.06%/23.37%`，body jerk 回退 `4.41%`。g50→g100 已显示继续优化路径会侵蚀部分时域平滑，因此任务在 g100 主动停止，不机械续到 g200。`step-0000100/flow.safetensors` 只被选为当前路径/平滑 Pareto 候选进入浏览器视觉验证，并不宣称最终质量成功。

### 23.6 g100 浏览器图导出与服务器对拍

浏览器版本已切到 `ARDY-Student-Direct-DMD2-G100-RootProj4-FP32`：encoder 使用 step-4000 EMA，flow 使用上述 g100 raw，decoder 使用 boundary-100k raw。三个 learned ONNX 为 encoder `15,551,369 B`、flow（含四次 root projection）`60,014,182 B`、decoder `19,836,043 B`；两个无参数 finalize 图合计 `45,264 B`。总下载 `95,446,858 B`（`91.025 MiB`），按十进制和二进制口径都低于 100 MB/MiB，参数量仍为 `23,825,364`。模型放在版本化目录 `infinite_demo/models/student/direct_dmd2_g100_rootproj4/`，避免浏览器复用旧 ONNX 缓存。

服务器端用相同随机输入对拍 PyTorch 与 ONNX Runtime CPU：初始 clean/explicit、续写 history/clean/explicit 全 finite，整链路最大绝对误差 `2.9683e-5`；初始输出 40 帧、续写输出 40 帧、第二窗 buffer 为 77 帧，稀疏 waypoint 索引为 initial `60`、continuation `62`，全部通过。前端额外 8-frame inertialization 已关闭，浏览器行为与上述五 seed 无惯性化评测一致；推理期间播放时间仍冻结，不补跳停顿帧。真实 Edge WebGPU 仍需页面重新加载新 manifest 后完成 session 编译、整链路生成与视觉验收。

本轮没有执行任何 `conda` 命令、没有安装包；只直接调用现有 py311 环境。用户已有的 `ardy/model/llm2vec/llm2vec.py` 保持未修改。

### 23.7 直接 NFE1 最终路线与真实前台 WebGPU 耗时

最终产品只执行一次 `x1→x0` flow，因此后续训练不再设计 NFE4→NFE2→NFE1 的中间课程，也不会把 NFE>1 上的收敛当成最终部署质量证据。固定路线为：以 paired teacher endpoint/condition 约束提供初始锚点，随后直接用短程 DMD2 + 低强度 conditional adversarial 校正一步输出分布；选模只看同 seed 的一步静态评估、1/5/20/50-window 闭环、频域/时域指标与视觉结果。

为判断是否还需要更复杂的 root 结构，又对旧 clean g200 EMA 做了零参数、低秩 cubic control-point projection 的未训练消融。8 控制点是其中最好的一档，但相对当前 binomial projection×4，endpoint `0.81250 vs 0.80684`、FK `0.23845 vs 0.23328 m`、foot slide `0.03385 vs 0.03171`、root temporal `0.12126 vs 0.11304`、path error `0.13075 vs 0.12514 m` 全部更差；10/12/16 控制点也没有形成 Pareto 改善。因此该方向在训练前否决，不跑 rollout、不启动四卡实验，避免对一个已被静态指标全面支配的结构继续消耗算力。

Edge 149 + NVIDIA Ampere 已加载新版 `direct_dmd2_g100_rootproj4`，由服务器远程下发 restart 与 3 warmup + 20 timed-run 的前台整链路 benchmark。本次结果确实属于新 release：40 帧一次生成的 total p50/p95 为 `96.35/140.36 ms`，history encoder `13.02/23.05 ms`，NFE1 flow `57.71/69.42 ms`，decoder pipeline `24.23/29.07 ms`，均通过预定的 p50 `100 ms` 理想线和 p95 `170 ms` 必过线。这是每段 40 帧的生成时间，不是单帧时间；在 20 FPS 下一段覆盖 2 秒播放，当前主要计算瓶颈是单次 flow，codec 不是首要重训对象。该性能通过不等于视觉质量通过，后者仍需以用户看到的无限 rollout 为准。

### 23.8 正式 student 容量修正：保留 flow 深度，只缩通道

原版浏览器运动网络（不含 Llama/LLM2Vec）共有 `191,211,576` 个参数：history encoder `17,554,048`（width 512、8 层、4 heads、FFN 1024），root denoiser `73,520,148`（width 1024、8 层、8 heads、FFN 2048），body denoiser `73,630,848`（同为 width 1024、8 层、8 heads、FFN 2048），flow 共享嵌入/投影 `8,672,256`，browser decoder `17,834,276`（width 512、8 层、4 heads、FFN 1024）。因此原版 flow 部分是 `155,823,252` 参数、root 8 + body 8 共 16 个 Transformer block；完整运动网络共 32 个 Transformer block。全 FP16 仅参数本体约 `382.42 MB`，无法满足浏览器小权重目标。

此前已训练的小 student 只有 `23,825,364` 参数：encoder `3,886,208`、flow `14,985,364`、decoder `4,953,792`。其 flow 是 width 512、8 heads、shared trunk 5 + body refinement 2，root 只经过 5 个 attention block，body 经过 7 个，确实过度削减深度。该容量不再作为正式重训架构，只保留为历史性能/失败视觉对照。

正式候选改为“flow 不减层、仅减 dim”，并遵守常见维度/超参数：encoder width 512、4 个 residual GELU block，共 `4,937,344` 参数；flow width 512、8 heads、head dim 64、shared/root trunk 8 + body refinement 8、FFN expansion 2，共 `33,910,420` 参数；decoder width 512、8 个 fixed-token channel-mixer block、token hidden 32，共 `9,165,420` 参数。总计 `48,013,184` 参数。所有主宽度均为 512，head 数为 8，head dim 为 64，block 数为 4/8/8+8，避免 width 560、7 heads 等不规整配置。训练采用 BF16/FP16 autocast + FP32 optimizer/master weights/EMA，部署推理与 WebGPU 图统一为 FP16；训练 checkpoint 是否以 FP32 保存不改变部署精度。

已经实际导出随机权重的三模块 FP16 ONNX 做结构/预算验证，而非只按参数量估算：encoder `9,874,881 B`、flow `67,866,721 B`、decoder `18,338,683 B`，合计 `96,080,285 B`（十进制 `96.08 MB`，约 `91.63 MiB`），严格低于 100 MB。三图的 ONNX Runtime CPU 对 PyTorch max-abs 分别为 `0.001953/0.003906/0.004639`，均可执行且数值一致。对应 case id 为 `ardy_student_stage1_depth16_standard48m_{encoder,flow,decoder}_fp16`。真实 Edge 149/NVIDIA Ampere 已对同一 width-512、8+8 flow 图完成 FP16 对拍：max abs `0.01953125`、mean abs `0.004259`、cosine `0.99999210`，全部 finite 并通过阈值；20 次稳态 median/mean/max 为 `67.105/69.908/85.705 ms`。同结构 8-block decoder 的 median 为 `33.895 ms`。这些仍是随机权重架构诊断图，不能当作训练质量结果；但已经证明标准完整深度图的 WebGPU 算子、数值和延迟足以进入四卡、分阶段学习率、无 cosine 的正式 teacher fitting。

容量换算必须保持明确：参数数量与训练/推理精度是两回事；纯 FP16 每参数至少 2 bytes，因此 `<100 MB` 的部署权重上限约为 50M 参数。若使用 100M 参数，纯 FP16 权重本体约为 200MB，除非改为 INT8，否则与当前 `<100 MB` 约束不能同时成立。当前 48.01M 方案是在 FP16 和 100MB 约束下接近上限、同时给 ONNX 元数据留出约 3.9MB 安全余量的质量优先设计，不再通过砍 flow 层数节省容量。

### 23.9 标准配置 WebGPU 定案与正式重训启动

Edge 149/NVIDIA Ampere 对标准 48.01M 架构的三个随机权重 FP16 图已全部实测，而不是按 CUDA 或参数比例外推：encoder width-512/4-block median/max 为 `9.940/12.265 ms`，flow width-512/8-head/head-dim-64/8+8-block 为 `67.105/85.705 ms`，decoder width-512/8-block 为 `33.895/63.990 ms`；三个后台图 median 直接相加为 `110.940 ms/40帧`。flow 的 max/mean abs 和 cosine 为 `0.019531/0.004259/0.99999210`，全 finite 并通过。因每窗覆盖 2 秒，该结果证明无需为速度继续削减 flow 深度。曾做过 width-560/7-head 的只读架构诊断，但该超参不规整，已经明确淘汰，不进入任何训练或最终模型。

codec 采用 function-preserving expansion：将已训练 width-512 encoder 3-block 与 decoder 4-block 权重完整覆盖到新的 4/8-block 模型；新增 residual 分支的最后投影置零，使 step-0 encoder/decoder 对原模型的 max abs 都精确为 `0`。四卡 BF16 探针使用物理 GPU `3,4,5,6`、global batch 1024；100 optimizer step 用时 `13.14 s`，约 `7,792 samples/s`；四卡为 `84%–89% SM`、约 `188–199 W/卡`，全部日志与权重 finite。原先随后启动的正式目录 `codec_standard_e4_d8_4gpu_b256_const3e6_100k/` 错把最终微调/分布蒸馏量级的固定 `3e-6` 用于第一阶段 teacher fitting，已在日志 step 12,000 停止并写入 `INVALID_WRONG_STAGE_LR.md`，其中 checkpoint 不进入选模或导出。更正后的第一阶段使用最普通的单次阶梯衰减：前 50k optimizer steps 为 `5e-5`，后 50k 为 `1e-5`，无 warmup、无 cosine、weight decay 0；同一协议也用于后续第一阶段 flow 拟合。只有最终 DMD2/ADV 分布蒸馏才使用 `1e-6` 量级低学习率。

被否决的 `3e-6` run 曾在 step-5000 完成独立 1024-window `teacher_val` 评估：相对旧 decoder-100k raw，5k raw 将 encoder bin accuracy 从 `0.39152` 提到 `0.41279`、encoder L1 从 `0.07974` 降到 `0.07501`；decoder total/FK/joint jerk/boundary jerk/foot slide/首帧 joint seam 从 `0.19800 / 0.009733 m / 0.006285 / 0.007258 / 0.010043 / 0.26339 m/s` 改善到 `0.19353 / 0.009465 m / 0.006085 / 0.007019 / 0.010032 / 0.25523 m/s`。这些数值仅保留作失败审计，不能因为有限或略有改善而恢复该错误学习率 run，也不进入正式 raw/EMA 选模。

更正后的正式目录为 `codec_standard_e4_d8_4gpu_b256_step5e5_to1e5_100k/`。物理 GPU `3,4,5,6` 上运行四进程 Accelerate/DDP，每卡 batch 256、global batch 1024、BF16 forward/backward、FP32 AdamW/EMA、EMA decay `0.9999`、gradient clip `1.0`。学习率严格为前 50k `5e-5`、后 50k `1e-5` 的单次 step decay；weight decay 0、warmup 0、无 cosine。为避免难解释的超参数，完整辅助监督只使用规整系数：rotation velocity/acceleration/jerk 为 `1.0/0.5/0.25`，joint velocity/acceleration/jerk 为 `1.0/0.5/0.25`，joint boundary velocity/acceleration/jerk 为 `0.5/0.25/0.1`，rotation boundary velocity/acceleration/jerk 为 `0.5/0.25/0.1`，rotation geodesic `0.25`、FK `2.0`、contact `0.1`、foot slide `0.25`。启动后 step 1 的实际 LR 为 `5e-5`，四卡 SM 为 `83%–88%`，早期 step 1/250/500 的 loss、权重与梯度均 finite；第一次漏传内层源码 `PYTHONPATH` 的启动在 import 阶段退出，没有产生 optimizer step。

正确学习率的 5k/10k 独立 1024-window 验证已经完成。5k EMA 的 encoder bin accuracy、encoder L1、decoder total、FK、joint seam 分别为 `0.50709 / 0.05911 / 0.17994 / 8.772 mm / 0.23983 m/s`；10k EMA 进一步达到 `0.54915 / 0.05346 / 0.16252 / 7.759 mm / 0.20759 m/s`，且 raw/EMA 的主要指标均由 EMA 胜出。作为严格 paired 对照，被否决的固定 `3e-6` 10k raw 对应为 `0.43104 / 0.07149 / 0.18892 / 9.212 mm / 0.24789 m/s`，证明第一阶段 `5e-5→1e-5` 不是仅改变日志尺度，而是显著提高 teacher fitting。10k EMA 的原版 50-window codec-oracle（seed `20260714`、1,853 帧）相对旧 4-block decoder 100k，将 FK `8.850→7.181 mm`、normalized body L1 `0.06288→0.05135`、foot slide/teacher `2.171×→1.879×`、joint seam/teacher `1.708×→1.483×`；segment-local joint acceleration P95、jerk P95、高频能量相对 teacher 分别为 `1.098×/1.306×/1.358×`，旧模型为 `1.107×/1.404×/1.557×`。这些仍只是 10k 中途结果，训练继续到 100k，不据此提前结束。

WebGPU 最终资产工具链也已同步到标准架构：`generate_student_infinite_demo_assets.py` 不再硬编码旧 `encoder3/decoder4/flow5+2` 或 FP32，而是显式读取 encoder/flow/decoder 路径、完整模型超参数和 release id，默认导出 FP16 learned graphs + FP32 root/FSQ utilities，并同时记录 ORT 同精度对拍及 FP16 相对 PyTorch FP32 误差；学习图和完整 ONNX 下载均强制 `<100,000,000 B`。`infinite_demo/demo.js` 已从 manifest 读取 learned precision，encoder/flow/decoder 使用 float16 tensor，utility 边界显式转回 FP32，并要求 adapter 提供 `shader-f16`。使用明确标记为 `tooling_smoke_invalid_weights_standard48m` 的无效旧权重做结构 smoke 时，三图均为真实 ONNX FLOAT16，encoder/flow/decoder 参数量为 `4,937,344 / 33,910,420 / 9,165,420`，CPU ORT 相对 FP32 max abs 为 `0.00147/0.00359/0.00422`；因 identity-expanded flow 的零残差被常量折叠，本次临时下载为 `83,594,895 B`，正式训练权重仍按此前非零随机图的 `96,080,285 B` 预算。

FP16 串联验证没有只看单图就宣布通过。无效权重 smoke 中，flow 的约 `0.004` 数值差会使少量连续 latent 跨越 FSQ bin 边界，导致 decoder latent/local-root 的稀疏最大差约 `0.12/0.13`，最终 explicit motion max/mean abs 约 `0.104/0.0066`，cosine 仍约 `0.99995`。这说明 FSQ 是不连续放大点，后续正式 checkpoint 必须同时报告 flow clean parity、FSQ bin mismatch、完整 explicit motion 误差和真实 WebGPU 动画，不能通过放宽一个 max-abs 阈值掩盖；必要时在第一阶段 flow 中加入 FP16 consistency/bin-margin 监督。该 smoke 使用无效权重，只证明验证器能发现问题，不代表正在训练的 codec 或未来 flow 的最终视觉质量。

第二阶段已新增独立的 `train_flow_dmd2.py`，不再沿用旧 `train_dmd2.py` 的 released cosine-diffusion score adapter。新实现严格使用参考项目的 rectified-flow 公式：`x_t=(1-t)x_0+tε`、目标 `v=ε-x_0`、重建 `x̂_0=x_t-t v`；冻结第一阶段 flow 作为 real score teacher，fake score 从 generator 克隆并只在 NFE1 generator 分布上拟合，adversarial head 读取冻结 flow teacher 的 conditioned generation feature。原版 ARDY 在该阶段仅通过离线语料提供 10-step+CFG `x0`，不再参与加噪或 score 公式；最终 generator 始终单次 `t=1→0`、无 CFG。连续时间重建测试、冻结 teacher feature 对 generator 输入梯度测试、codec/flow identity expansion 测试均通过。正式 DMD2 必须等待第一阶段新 flow 训练和多 seed gate，不得用当前代码通过冒充模型质量。

### 23.10 正确第一阶段学习率的 20k/25k codec 审计

正式 `5e-5→1e-5` codec 任务在不中断四卡训练的情况下完成了 20k 与 25k 独立 1024-window 检查。20k EMA 的 encoder bin accuracy、decoder total、FK、joint jerk、foot slide、joint seam 分别为 `0.59194 / 0.14173 / 6.760 mm / 0.004764 / 0.008399 / 0.17927 m/s`；25k EMA 继续改善到 `0.60564 / 0.13392 / 6.437 mm / 0.004558 / 0.008219 / 0.17108 m/s`。25k raw 对应为 `0.58149 / 0.13545 / 6.741 mm / 0.004766 / 0.008374 / 0.17696 m/s`，因此 EMA 仍是当前候选，但训练不会因 25k 中途结果提前停止。

25k 完整 Accelerate 状态已做只读恢复审计：模型共 132 个 tensor、optimizer 共 396 个状态 tensor、EMA 共 132 个 shadow tensor，全部 finite；132 组 Adam state 的 step 最小值和最大值均为 `25000`，scheduler `last_epoch=25000`、当前 LR `5e-5`，EMA update count 为 `25000`，四个 rank 的 RNG 状态及 shard sampler 状态齐全。这证明 checkpoint 是真实可恢复的 25k optimizer step，而不是仅保存推理权重或按 sample 数折算。

25k EMA 又按原版无限生成的 buffer/history/waypoint 语义完成 seed `20260714..20260718` 五条各 50-window codec-oracle：平均 FK 为 `5.098 mm`，foot slide 为 teacher 的 `1.644×`，segment-local joint acceleration P95、jerk P95、高频能量和 joint seam 分别为 teacher 的 `1.033× / 1.118× / 1.182× / 1.227×`；五 seed 最坏值为 `1.072× / 1.247× / 1.312× / 1.360×`。同 seed `20260714` 的 raw/EMA 对拍中，EMA 的 FK、foot、jerk、高频分别为 `6.118 mm / 1.744× / 1.247× / 1.312×`，优于 raw 的 `6.122 mm / 1.831× / 1.265× / 1.341×`；raw 只在 acceleration 与 seam 上有极小优势。与 10k EMA 单 seed 的 acceleration/jerk/HF/seam `1.098×/1.306×/1.358×/1.483×` 相比，25k 已实质改善，但 foot slide 平均仍高于 teacher 约 64%，所以继续完整 100k 协议。

为使下一阶段 flow 系数建立在真实结构而非旧模型上，`audit_loss_gradients.py` 已增加当前 codec 架构参数入口；此前它写死 3-block encoder/4-block decoder，不能用于 4/8-block 标准 codec。修正后编译与 CLI 检查通过。环境没有 `pytest`，未安装任何包；改用测试文件的原生入口运行，7 项 rectified-flow/DMD2 单测和完整 student shape/identity-expansion/finite 测试均通过。

标准 8+8 flow 又用 25k EMA codec 做了 initial 与 continuation+in-window-waypoint 两类逐项梯度审计。旧候选的 decoder/root-temporal/seam/physical 系数会使 auxiliary 梯度超过 supervised core，因此正式入口改用规整的 FSQ `1.0`、decoder/path/quality `0.1`、root-temporal/seam `0.01`、physical seam `0.001`。continuation batch 的 supervised/auxiliary 梯度 L2 为 `23.52/12.01`，合并梯度与 supervised core cosine 为 `0.901`；initial batch 为 `44.94/16.97`、cosine `0.942`。另外新增 STE FSQ endpoint L1：连续 endpoint loss 把预测拉向 teacher bin center，量化 L1 明确惩罚部署时落入错误 bin；它在两类 batch 中与 core 梯度高度同向。`train_flow.py` 默认值已同步为 width-512、8 heads、8+8 blocks、NFE1、100k、前后 `5e-5/1e-5`、70% exact-t1 + 20% high-noise + 10% uniform、weight decay/warmup `0/0`，避免漏参回退到历史小模型或错误学习率。

单卡 batch-32 的 10-step 全链路工程 smoke 已实际经过 encoder→history FSQ→33.91M flow→endpoint FSQ→8-block decoder→FK/temporal/seam losses，BF16 forward/backward、FP32 AdamW/EMA 全 finite；稳态约 `168 ms/optimizer step`。缩短的测试边界中，step 1–4 日志为 `5e-5`，完成第 5 次更新后准确变为 `1e-5`；213 个 raw/EMA/model tensor 及 213 组 Adam state 全 finite，optimizer step 均为 10。按正式四卡每 rank batch 32 的同量级 step time估算，100k flow 约 4.7 小时，连同约 3.3 小时 codec 和短 DMD2/导出仍在首版 12 小时预算内。目录 `flow_standard48m_stage1_protocol_smoke10_codec25k/` 已写 `ENGINEERING_SMOKE_ONLY.md`，不得选模或导出。

相同配置的 FP16 autocast 对照在第 2/3 次尝试出现裁剪前 grad norm `inf`，GradScaler 跳过更新；旧计数逻辑却把日志/scheduler/EMA推进到 10，而 213 组 Adam state 实际都只有 8，属于不可接受的伪 step。`train_codec.py` 与 `train_flow.py` 现已在 `accelerator.optimizer_step_was_skipped` 时同时冻结 global step、scheduler 和 EMA。修复后的 FP16 复测记录两次 skip 后继续完成 10 次真实更新，Adam/scheduler/EMA 均精确为 10，且 LR 只按真实更新数在第 5 次切换。正式 flow 仍选择无 overflow 的 BF16 混合精度；部署保持 FP16，并通过导出后 FP16/FSQ/完整动画 gate 保证训练部署匹配，而不是容忍 scaler 跳步。两个 FP16 目录均已标为工程或否决结果，不进入选模。

### 23.11 第一阶段与分布蒸馏的学习率边界、35k codec 进展

学习率协议再次明确分阶段锁定，不能把最终分布蒸馏的低学习率套到初始 teacher fitting：codec teacher fitting 和直接 NFE1 flow teacher fitting 都完整训练 `100,000` 个真实 optimizer steps，前 `50,000` 步固定 `5e-5`，后 `50,000` 步固定 `1e-5`，只做一次硬切换；无 warmup、无 cosine、weight decay 为 0。只有完成第一阶段、多 seed 闭环和视觉 gate 后，短程 DMD2/ADV 才使用 generator/fake-score/critic 的 `1e-6/5e-6/1e-6` 量级，且限定为数百到最多数千个 generator update。历史固定 `3e-6` 的第一阶段任务仍保持 invalid，不允许恢复或用于选模。

正确协议下，35k 独立 1024-window 验证的 EMA encoder bin accuracy、encoder L1、decoder total、FK、joint jerk、foot slide、joint seam 为 `0.62508 / 0.04502 / 0.12259 / 5.922 mm / 0.004234 / 0.007927 / 0.15932 m/s`；raw 对应为 `0.60106 / 0.04718 / 0.12415 / 6.424 mm / 0.004561 / 0.008035 / 0.17115 m/s`，EMA 在所有列出的核心项上继续领先。记录时正式任务已推进到 42k，四进程仍正常运行，当前 LR 仍为 `5e-5`；必须到第 50,000 次真实 optimizer update 后才切到 `1e-5`，不会根据中途 loss 手工改学习率或提前停止。

8+8 flow 的 function-preserving expansion 也完成了实际更新验证：10 个 BF16 optimizer step 后，新增 `trunk.5..7` 与 `body_refiner.2..7` 每层约 210 万个元素发生变化，各层相对初始化的参数 L2 变化约 `0.160–0.162`。因此新增深度不是零初始化后永久失活的空层；正式第一阶段可以从旧模型保留函数起点，同时让全部新增 block 接受梯度并以 `5e-5→1e-5` 学习。

40k 权重在训练不中断时完成独立 1024-window 对拍。EMA 相对 raw 的 encoder bin accuracy 为 `0.63394 vs 0.60825`，encoder L1 为 `0.04418 vs 0.04646`，decoder total 为 `0.11823 vs 0.12084`，FK 为 `5.717 vs 6.223 mm`，joint jerk 为 `0.004110 vs 0.004319`，foot slide 为 `0.007668 vs 0.007875`，joint seam 为 `0.15541 vs 0.16280 m/s`；EMA 在列出的核心项上继续一致领先。同一 EMA 用 CUDA FP16 arithmetic 重跑，FK `5.719 mm`、joint jerk `0.004116`、foot slide `0.007674`、joint seam `0.15547 m/s`、encoder bin accuracy `0.63379`，均与 FP32 非常接近。这证明 40k codec 的 FP16 单模块行为稳定，但最终仍需把 flow 输出跨 FSQ 边界的完整链路误差单独 gate，不能由 codec 指标外推。

teacher 语料的来源也由每个 rank 的 `generation_config.json` 直接复核：原版 `ARDY-Core-RP-20FPS-Horizon40` 以 FP32 teacher 执行每窗 10 个 denoising step，constraint CFG 固定为 `1.5`，25% 无 constraint，text encoder 未加载；131,072 个 clean endpoint 仅在落盘时转成 FP16。第一阶段 student 因此拟合的正是原版 10-step + constraint CFG 结果，部署与最终 DMD2 后不再执行 teacher 或 CFG。

50k 单次 LR 边界已由完整恢复状态而非日志推断确认：132 组 Adam state 的 step 最小/最大值均为 `50000`，scheduler `last_epoch=50000`、`_last_lr=1e-5`，EMA `num_updates=50000`、decay `0.9999`；model/optimizer/EMA 全部 floating tensor finite，四 rank RNG 和 shard sampler 均存在。日志 step 49,750 仍为 `5e-5`，step 50,000 完成后显示 `1e-5`，所以前 50k 个 update 实际用 `5e-5`、后续 update 才用 `1e-5`，没有 off-by-one 或伪切换。

50k 独立固定集继续选择 EMA。raw/EMA FP32 的 encoder bin accuracy 为 `0.59953/0.64614`，decoder total `0.11511/0.11088`，FK `5.733/5.374 mm`，joint jerk `0.004105/0.003894`，foot slide `0.007707/0.007479`，joint seam `0.15214/0.14682 m/s`。EMA FP16 对应为 bin accuracy `0.64607`、decoder total `0.11094`、FK `5.378 mm`、joint jerk `0.003899`、foot `0.007484`、joint seam `0.14687 m/s`，仍与 FP32 紧密一致。

同一 50k EMA 又完成 seed `20260714..20260718` 五条、每条 50-window、无前端惯性化的原版无限 buffer/history/waypoint codec-oracle。五 seed 均值为 normalized body L1 `0.03074`、FK `4.278 mm`、rotation geodesic `0.02246 rad`；foot slide、segment-local joint acceleration P95、jerk P95、高频能量和 joint seam 相对 teacher 分别为 `1.517× / 1.022× / 1.080× / 1.132× / 1.159×`，最坏 seed 为 `1.608× / 1.055× / 1.189× / 1.250× / 1.236×`。相对 25k EMA 均值 `5.098 mm / 1.644× / 1.033× / 1.118× / 1.182× / 1.227×`，FK、foot、acceleration、jerk、高频与 seam 全部继续改善。codec 仍按既定协议训练到 100k，不因 50k 中途结果提前停止。

50k codec 的 flow 梯度复审先发现审计工具仍残留历史 `root smoothing passes=4` 默认值；两份错误口径文件已经由 `INVALID_codec50k_gradient_audit_rootp4.md` 明确排除，正式训练本身始终是 projection `0`。工具默认值现已同步为 width-512、8+8、NFE1、projection 0 以及正式规整系数，并用显式 `rootp0` 文件名重跑。initial batch 的 supervised/auxiliary/combined 梯度 L2 为 `44.94/16.30/49.96`，combined 对 supervised cosine `0.9465`；continuation + 2 个窗内/2 个未来 waypoint batch 为 `23.16/11.76/27.05`，cosine `0.9018`。FSQ 梯度对 supervised core 的 cosine 为 `0.891/0.972`。因此辅助项没有压过 teacher fitting 主目标，且 25k→50k codec 改善没有推翻已选的规整系数；最终 100k codec 仍会再做一次审计。

新增 `audit_precision_parity.py`，以浏览器 batch size 1 为默认，直接在相同输入和相同权重上累计 PyTorch FP16 vs FP32 的连续误差、cosine、FSQ bin flip，以及可选的完整 encoder→flow→decoder parity。50k EMA codec 在 1024-window、batch-1 上的 encoder 连续 max/mean abs 为 `0.00611/0.000766`、cosine `0.99999949`；114,688 个有效 history latent 元素中有 929 个 FP16/FP32 bin 不同，比例 `0.810%`。decoder teacher-input 输出 max/mean abs 为 `0.0511/0.001126`、cosine `0.99999918`。作为能否识别坏 FP16 的对照，旧 5+2 flow 在相同 50k codec 上的 endpoint FSQ bin flip 为 `3.688%`，完整 decoder body max/mean abs `1.100/0.00946`、cosine `0.9999216`；这与此前旧 FP16 动画劣化一致。最终新 8+8 flow 必须显著压低该整链路误差并通过真实 WebGPU 视觉 gate，不能仅凭 codec 单模块通过。

### 23.12 EMA horizon 更正与已有权重续训原则

60k 固定验证出现了明确的 EMA 滞后信号。raw/EMA 的 encoder bin accuracy 为 `0.64917/0.65388`，encoder L1 为 `0.04269/0.04234`，说明 encoder 的 EMA 仍略优；但 decoder total 为 `0.10300/0.10493`、FK 为 `5.031/5.070 mm`、joint jerk 为 `0.003667/0.003699`、foot slide 为 `0.007313/0.007334`、joint seam 为 `0.13800/0.13960 m/s`，raw 在 decoder 指标上暂时略优。这个现象不能解释为应当发布 raw 或拼接 raw decoder + EMA encoder，而应当解释为 LR 在 50k 从 `5e-5` 切到 `1e-5` 后，旧 EMA 的有效历史过长、尚未跟上新的参数轨迹；最终候选仍统一选择 EMA。

70k 的同一独立 1024-window 对拍复现了相同分工而非偶然波动：EMA/raw encoder bin accuracy 为 `0.65908/0.65090`，encoder L1 为 `0.04189/0.04253`，EMA encoder 仍优；raw/EMA decoder total 为 `0.10021/0.10150`、FK 为 `4.853/4.892 mm`、foot slide 为 `0.007249/0.007266`、首帧 joint seam 为 `0.13385/0.13533 m/s`，raw decoder 仍略优。连续两个 checkpoint 都说明旧 EMA 对后 50k 低 LR 轨迹响应过慢，支持缩短 horizon，而不支持拆模块混用 raw/EMA。

只读检查了 `<REFERENCE_3DLM_PROJECT>/20260316_zlw_train_vertices_edge_vae_min.py`、`20260104_zlw_clayv3_train_vae.py`、`20260106_zlw_clayv3_train_ldm.py`、`20260418_zlw_train_s1vae.py`、`20260419_zlw_train_s1ldm.py` 和用户指定的 `20250425_zlw_clayv2_train_ldm_fsdp2_dmd2.py`，没有修改 `<REFERENCE_3DLM_PROJECT>` 下任何文件。较新的长程 VAE/LDM 训练普遍使用固定 `decay=0.9995`，并显式关闭 EMA warmup；短 DMD2 generator 使用固定 `0.999`。按 `h_99=ln(0.01)/ln(decay)` 计算，`0.9999/0.9995/0.999` 的 99% 历史替换长度分别约为 `46.05k/9.21k/4.60k` step。

本项目此前的 `ModelEMA` 还有第二个问题：它把 effective decay 写成 `min(decay, (1+n)/(10+n))`，并不是真正的 constant decay；在 60k 时有效 decay 约 `0.99985`，而且越训练越迟钝，约到 90k 才碰到 `0.9999` 上限。实现现已改为从第一次 update 起严格使用固定 decay：100k codec/flow teacher fitting 默认 `0.9995`；默认 1000 个 generator update 的短 DMD2/ADV 使用 `0.995`，99% horizon 约 `919` step。参考 DMD2 中的 `0.999` 配套的是命令注释里最长 `200k` 的任务；若机械用于本项目 1000-step 阶段，结束时仍会保留约 `36.8%` 的初始权重，正好重复 EMA 滞后问题。`0.995` 在 1000 step 后只保留约 `0.665%` 初始权重，仍有平滑作用又能使最终 EMA 代表当前 generator。恢复 checkpoint 时默认仍精确恢复原 decay；只有显式 `--override-ema-decay-on-resume` 才保留命令行的新 decay，model、AdamW、scheduler、sampler、各 rank RNG、EMA shadow 和 update count 均继续恢复。新增四项 EMA 单测验证 horizon、首步 constant update、默认 decay 恢复和只覆盖 decay 的续训语义；连同 flow/shape 测试共 11 项通过。

当前 codec 不重新初始化，也不丢弃前 75k 学到的任何权重。计划在 75k 完整 Accelerate resume state 原子落盘并审计后，从同一个 model/optimizer/scheduler/sampler/RNG/EMA shadow 继续后 25k，只把后续 EMA decay 显式改为固定 `0.9995`；25k 个新 update 后，旧 EMA shadow 的残余系数约为 `0.9995^25000≈3.7e-6`。正式 8+8 flow 同样优先继承已有同 width flow 的兼容层，并只把新增 residual block 做 identity 初始化；第一阶段 DMD2/ADV generator 必须从监督阶段最终 EMA 原样启动，fake-score 从该 generator 复制。除非某个新增张量确实没有兼容来源，否则禁止为改配置而全量随机重启；所有局部初始化都必须列出未继承的 tensor 名称和参数量。

修正后的 warm-start + 混合精度路径又完成一次真实 CUDA update smoke：冻结的 50k EMA encoder/decoder 以部署一致的 FP16 运算，33.91M 参数 8+8 flow 以 BF16 autocast 反传，FP32 AdamW 和固定 `0.9995` FP32 EMA 更新；flow 从已训练 width-512 5+2 EMA 加载所有兼容 tensor，只将新增 residual block 做 identity 初始化。step 1 的 total loss `1.49555`、grad norm `25.4863` 和全部时域/物理分支均 finite，raw/EMA 各 213 个 tensor、`33,910,420` 参数全部 finite，无 optimizer skip。目录 `flow_standard48m_fp16codec_bf16_warmstart_smoke1_20260715/` 已标为 `ENGINEERING_SMOKE_ONLY`，不得选模；它只证明正式 flow 无需随机重启且训练/部署精度路径可反传。

另用空闲 GPU 对真实四 rank 50k state 做了隔离的一步 Accelerate 恢复演练，原 state 只读且未被修改。恢复后的 update 编号精确为 `50001`，scheduler 保持 LR `1e-5`；model、AdamW、sampler/RNG、EMA shadow 和 update count 均从旧 state 继承。全部 132 个 EMA tensor 都满足 `ema_50001 = 0.9995 × ema_50000 + 0.0005 × raw_50001`，保存后最大绝对核对误差仅 `2.384e-7`，raw/EMA 全 finite。目录 `codec_resume_decay_override_smoke_50k_to50001_20260715/` 同样已标为 `ENGINEERING_SMOKE_ONLY`。这证明 75k 切换可以只改 EMA horizon，不需要也不允许重启模型权重或 optimizer。

正式 8+8 flow 的 warm-start 候选进一步锁定为此前 width-512、5+2、直接 NFE1 的 g200 EMA，而不是随机权重、raw 权重或工程 smoke 权重。对真实 `dmd2_direct_nfe1_4gpu_dmd_adv005_weakanchor04x_g200/weights/step-0000200/flow_ema.safetensors` 的逐 tensor 扩展审计表明：旧模型 `14,985,364` 个参数全部继承；新模型总计 `33,910,420` 个参数；只有新增 `trunk.5..7` 和 `body_refiner.2..7` 的 108 个 parameter tensor、`18,925,056` 个参数没有旧来源，并按 residual identity 初始化。相同随机输入上，扩展前后 velocity 输出 max abs 精确为 `0`。因此正式训练 step-0 完整保留已有 EMA 函数，只让新增深度和所有旧层共同继续学习；最终是否采用该来源仍会在 100k codec 固定后做一次初始静态/闭环审计，但不会退回全量噪声初始化。

75k 完整状态已经实际完成切换，而不是只写计划。旧进程在 `state/step-0075000` 和四份 step-75k 推理权重全部写完后停止；审计确认 132 个 model tensor、132 组 Adam state、132 个 EMA shadow 全 finite，所有 Adam step、scheduler epoch 和 EMA update count 都精确为 `75000`，LR 为 `1e-5`，四 rank RNG 与 sampler epoch `586` 齐全；state model 对 raw export、state EMA 对 EMA export 的 max abs 都为 `0`。随后在同一物理 GPU `3,4,5,6` 从该 state 恢复，命令显式使用固定 `ema_decay=0.9995` 与 `override_ema_decay_on_resume=true`；记录文件 `resume_config_step-0075000.json` 给出的 99% horizon 为 `9208.04` step。恢复后的首个日志点为 step `75250`、LR `1e-5`，证明 global step/scheduler 没有重置；四卡实时 SM 为约 `80%–88%`，续训正常。旧 75k EMA shadow 完整保留，后续只改变滤波时间常数。

80k 是切换固定 `0.9995` 后仅经过 5k update 的过渡点，此时旧 75k shadow 理论残余仍约 `8.2%`，但新 EMA 已在独立 1024-window 固定集上全面优于 raw：encoder bin accuracy `0.66428 vs 0.65898`、encoder L1 `0.04143 vs 0.04191`、decoder total `0.09788 vs 0.09819`、FK `4.714 vs 4.777 mm`、joint jerk `0.003497 vs 0.003528`、foot slide `0.007189 vs 0.007210`、首帧 joint seam `0.13052 vs 0.13097 m/s`。同一 EMA 以部署 FP16 运算重跑，decoder total `0.09796`、FK `4.720 mm`、jerk `0.003505`、foot `0.007194`、seam `0.13070 m/s`，与 FP32 紧密一致。该结果初步证明修正 horizon 后无需 raw/EMA 混选；85k 经过完整约 9.2k horizon 后仍需重复验证。

85k 已经过切换后的完整 99% horizon，独立集再次由 EMA 全面且更明显地胜过 raw：encoder bin accuracy `0.66584 vs 0.65904`、encoder L1 `0.04126 vs 0.04176`、decoder total `0.09677 vs 0.09739`、FK `4.667 vs 4.780 mm`、joint jerk `0.003469 vs 0.003541`、foot slide `0.007158 vs 0.007226`、首帧 joint seam `0.12926 vs 0.13137 m/s`。所以 60k/70k 的 raw decoder 暂时领先已被证实是旧 EMA 时间常数过长，而不是 raw 更适合发布；后续 codec、flow 和 DMD2/ADV 的正式候选统一使用阶段长度匹配的 EMA，不再做 raw/EMA 拼接。

85k EMA 又完成 seed `20260714..20260718` 五条、每条 50-window、无前端惯性化的原版无限 buffer/history/waypoint codec-oracle。相对 50k EMA，五 seed 均值的 normalized body L1 从 `0.03074→0.02682`、FK 从 `4.278→3.770 mm`、rotation geodesic 从 `0.02246→0.01955 rad`；foot slide、segment-local joint acceleration P95、jerk P95、高频能量和 joint seam 相对 teacher 从 `1.517/1.022/1.080/1.132/1.159×` 改善到 `1.443/1.019/1.064/1.099/1.125×`。五 seed 最坏值也分别从 `1.608/1.055/1.189/1.250/1.236×` 降到 `1.514/1.048/1.155/1.197/1.171×`，所有列都改善。因此新 EMA 的优势不是固定静态集过拟合或单 seed 偶然；codec 仍继续完整 100k，再以最终同口径结果锁定。

90k 独立固定集继续复现 EMA 全面领先：EMA/raw encoder bin accuracy `0.66797/0.65914`、encoder L1 `0.04113/0.04178`、decoder total `0.09587/0.09694`、FK `4.620/4.704 mm`、joint jerk `0.003445/0.003518`、foot slide `0.007130/0.007183`、首帧 joint seam `0.12778/0.12952 m/s`。90k EMA 自身又优于 85k EMA 的 decoder total `0.09677`、FK `4.667 mm`、jerk `0.003469` 和 seam `0.12926 m/s`，说明修正后的 EMA 并非只在一个 checkpoint 偶然领先，低 LR 阶段仍在稳定改善。

95k 独立固定集仍保持相同趋势。raw/EMA encoder bin accuracy 为 `0.65889/0.66806`、encoder L1 为 `0.04180/0.04104`、decoder total 为 `0.09546/0.09505`、FK 为 `4.650/4.580 mm`、joint jerk 为 `0.003456/0.003418`、foot slide 为 `0.007129/0.007109`、首帧 joint seam 为 `0.12830/0.12686 m/s`。相对 90k EMA，95k 的 decoder total、FK、jerk、foot slide 和 seam 仍有小幅一致改善；因此继续同一个已恢复的训练状态完成 100k，不另起模型，也不以训练 batch 的瞬时波动替代固定集判断。

已有权重继承被设为后续所有阶段的默认硬约束，而不只适用于本次 codec：同构且训练超参数兼容时，从最近的完整 checkpoint 恢复 model、optimizer、scheduler、sampler、各 rank RNG、EMA shadow 与 update count；只有推理权重可用时，优先从表现最好的 EMA warm-start，并明确记录哪些训练状态无法继承。结构扩宽或加深时，逐 tensor 继承所有 shape 兼容权重，以 function-preserving/identity 方式初始化新增分支；超参数相同绝不因为改目录、改运行时长或进入下一次会话而重新做全量噪声初始化。只有不存在任何兼容来源，或继承会在数学上改变任务定义时，才允许新初始化，并必须先写清原因、未继承 tensor 名称和参数量。

### 23.13 100k codec 锁定、部署精度与 flow 交接审计

正式 codec 已从 75k 完整状态连续训练到第 `100,000` 个真实 optimizer update，并以 `training_complete / stopped_by_runtime=false` 正常退出，没有重新初始化，也没有 optimizer skip。100k 恢复状态包含 132 个 model tensor、132 组 Adam state（396 个浮点状态 tensor）、132 个 EMA shadow、scheduler、sampler 和四份 rank RNG；全部 finite。所有 Adam step 的最小/最大值均为 `100000`，scheduler `last_epoch=100000`、LR `1e-5`，EMA `num_updates=100000`、固定 decay `0.9995`，sampler epoch 为 `782`。state model 对 raw encoder/decoder 导出、state EMA shadow 对 EMA encoder/decoder 导出的逐 tensor max abs 全部为 `0`，因此该 checkpoint 可以无损续训，不是仅供推理的残缺权重。

100k 独立 1024-window 固定集继续明确选择 EMA。raw/EMA FP32 的 encoder bin accuracy 为 `0.65969/0.66880`、encoder L1 为 `0.04173/0.04095`、decoder total 为 `0.09559/0.09432`、FK 为 `4.670/4.546 mm`、joint jerk 为 `0.003515/0.003387`、foot slide 为 `0.007150/0.007071`、首帧 joint seam 为 `0.12798/0.12592 m/s`。同一 EMA 用部署 FP16 算术得到 decoder total `0.09441`、FK `4.553 mm`、jerk `0.003395`、foot `0.007081`、seam `0.12608 m/s`、encoder bin accuracy `0.66839`，与 FP32 静态指标紧密一致。

batch-1 的逐元素 FP16/FP32 对拍给出 encoder continuous max/mean abs `0.006268/0.000843`、cosine `0.99999938`，114,688 个 encoder FSQ 元素中 1,019 个跨 bin，mismatch `0.8885%`；decoder teacher-input max/mean abs `0.06658/0.001301`、cosine `0.99999890`，全部 finite。codec 单模块的连续误差很小，但 FSQ bin mismatch 相比 50k 的 `0.810%` 略升，所以这只允许冻结 codec 进入 flow 训练，不能替代最终 encoder→flow→FSQ→decoder 的整链路 FP16/WebGPU gate。

100k EMA 又完成 seed `20260714..20260718` 五条各 50-window、无前端惯性化的原版无限 buffer/history/waypoint codec-oracle。五 seed 均值为 normalized body L1 `0.02615`、FK `3.672 mm`、rotation geodesic `0.01904 rad`；foot slide、segment-local joint acceleration P95、jerk P95、高频能量与 joint seam 分别为 teacher 的 `1.423/1.016/1.060/1.095/1.138×`。最坏 seed 分别为 body L1 `0.02965`、FK `4.370 mm`、rotation `0.02120 rad`，以及 foot/acceleration/jerk/HF/seam `1.495/1.042/1.151/1.194/1.165×`。相对 85k，前七项均值改善约 `0.25%–2.60%`，且所有最坏值改善；唯一 trade-off 是 seam 均值从 `1.125×` 小幅升至 `1.138×`（`+1.14%`），但 seam 最坏值从 `1.171×` 降至 `1.165×`。综合静态、五 seed 均值和最坏值，100k EMA 被锁为后续冻结 codec；该小幅 seam 均值回摆保留在结论中，不以“全面改善”掩盖。

用锁定的 100k EMA codec、正式 8+8 identity-expanded g200 EMA flow 起点重新做了两类梯度审计。initial batch 的 supervised/auxiliary/combined 梯度 L2 为 `44.94/18.14/50.57`，combined 对 supervised cosine `0.9346`；continuation + 2 个窗内/2 个未来 waypoint batch 为 `23.67/12.90/27.95`，cosine `0.8882`。两类 batch 的所有目标和梯度均 finite，auxiliary 仍小于 supervised core，FSQ 对 core cosine 分别为 `0.8912/0.9728`；因此继续使用已锁定的规整系数，不因 codec 更新临时重调。正式 flow 不从随机噪声启动：继续以 `dmd2_direct_nfe1_4gpu_dmd_adv005_weakanchor04x_g200/.../flow_ema.safetensors` 为来源，继承旧 5+2 模型全部 `14,985,364` 个参数，只对新增 `trunk.5..7`、`body_refiner.2..7` 的 `18,925,056` 个参数做 residual identity 初始化，并从该函数保持不变的 step-0 继续训练。

正式 flow 已按上述交接实际启动，目录为 `flow_standard48m_8p8_nfe1_warm_g200ema_codec100k_fp16codec_4gpu_b32_step5e5_to1e5_100k/`。它使用物理 GPU `3,4,5,6`、四进程 DDP、每卡 batch 32/global batch 128、trainable flow BF16 forward/backward、FP32 AdamW/固定 `0.9995` EMA，以及冻结 100k EMA encoder/decoder 的部署一致 FP16 运算；100k update 前 50k 为 `5e-5`、后 50k 为 `1e-5`，无 warmup、无 cosine、weight decay 0。step 1 与 step 500 的所有 loss/梯度均 finite、无 optimizer skip，step 500 仍为 `5e-5`；step 1→500 用时约 `75.8 s`，稳态约 `0.152 s/update`、`838 samples/s`，估算完整 100k 约 `4.2 h`。五次 `nvidia-smi dmon` 采样中四卡 SM 约 `31%–74%`、功耗约 `109–137 W`；该图由多个短 encoder/flow/decoder/FK/时域 loss kernel 与同步构成，属于 dispatch/同步受限，不能为抬高利用率读数而在正式 run 中途改变 batch 或重新初始化。

正式 flow 的 5k 首个快照已经完成早期审计。raw/EMA 各 213 个 tensor、`33,910,420` 参数全部 finite，没有 optimizer skip；EMA 在固定 1024-window 上优于 raw。EMA FP32 的 endpoint total/body L1、FSQ bin accuracy、FK、foot slide、joint jerk、root temporal 与 path error 分别为 `0.73008 / 0.48973 / 0.06898 / 0.20753 m / 0.04676 / 0.05874 / 0.84246 / 0.10308 m`，部署 FP16 对应为 `0.73028 / 0.48982 / 0.06893 / 0.20758 m / 0.04679 / 0.05878 / 0.84293 / 0.10310 m`，说明当前 8+8 图的 FP16 静态数值一致性良好。

同口径的 identity-expanded g200 EMA step-0 起点为 endpoint total `0.75168`、FK `0.22888 m`、foot slide `0.07698`、joint jerk `0.10683`、root temporal `1.38691`、path error `0.13191 m`。所以前 5k 已分别改善约 `2.9%/9.3%/39.3%/45.0%/39.3%/21.9%`，证明继承旧 EMA 后继续 teacher fitting 是有效路线，而不是新增深度完全不工作。但绝对误差仍远未可用：seed `20260714` 的原版 50-window 闭环中，root acceleration/jerk 为 teacher 的 `20.45×/45.15×`，joint acceleration/jerk/HF 为 `2.63×/5.14×/13.95×`，joint seam 为 `7.96×`，并出现明显路径漂移。该 5k checkpoint 只作收敛基线，禁止用于 demo、DMD2 或导出；继续完整 100k 第一阶段，不能用几千步的早期结果冒充蒸馏完成。

10k 快照继续沿用同一继承链，没有重新初始化。raw/EMA 各 213 个 tensor、`33,910,420` 参数全部 finite，固定 1024-window 仍由 EMA 明显胜出。10k EMA FP32 的 endpoint total/body L1、FSQ bin accuracy/center abs bins、FK、foot slide、joint jerk、root temporal 和 path error 为 `0.36166 / 0.35421 / 0.09314 / 3.70461 / 0.14499 m / 0.03063 / 0.04217 / 0.61875 / 0.08884 m`；相对 5k EMA 分别改善约 `50.5% / 27.7% / +35.0% / 28.4% / 30.1% / 34.5% / 28.2% / 26.6% / 13.8%`。同权重 FP16 的对应值为 `0.36169 / 0.35424 / 0.09316 / 3.70490 / 0.14502 m / 0.03065 / 0.04220 / 0.61913 / 0.08880 m`，这些主要指标对 FP32 的相对差异均小于约 `0.07%`。

但同一 seed `20260714` 的原版 50-window 闭环仍不可用，且静态收益没有全部转化为控制收益。5k→10k 的 root acceleration/jerk/HF 比例从 `20.45×/45.15×/13.62×` 降到 `14.70×/32.77×/13.50×`，joint acceleration/jerk/HF 从 `2.63×/5.14×/13.95×` 降到 `1.87×/3.66×/12.53×`，root/joint seam 从 `17.65×/7.96×` 降到 `12.69×/5.93×`，foot slide 从 `0.933` 降到 `0.710 m/s`；这些时域项改善约 `24%–29%`。反之，waypoint mean、root drift mean 和 rollout FK 从 `14.37/14.00/13.98 m` 反弹到 `15.32/14.90/14.94 m`，恶化约 `6%–7%`。因此 10k 继续只是中途证据：可以证明 inherited EMA + identity expansion 正在学习，不能证明闭环路径已解决；不更新 demo、不导出 WebGPU、不提前进入 DMD2。记录位于 `eval/flow_standard48m_step0010000_{raw_fp32,ema_fp32,ema_fp16}.json`、`rollout/flow_standard48m_step0010000_ema_oracle50_seed20260714/` 和 `eval/flow_standard48m_step0010000_ema_jitter_seed20260714.json`；正式四卡训练在评估期间未停止。

15k 继续从同一训练状态向前，不重启、不重新初始化。raw/EMA 的 213 个 tensor 和 `33,910,420` 参数全部 finite，EMA 静态继续全面胜出 raw。相对 10k EMA，15k EMA FP32 的 endpoint total、FK、foot slide、joint jerk、root temporal 和 path error 从 `0.36166/0.14499/0.03063/0.04217/0.61875/0.08884 m` 改善到 `0.31890/0.13381/0.02710/0.03648/0.53905/0.08251 m`，分别下降约 `11.8%/7.7%/11.5%/13.5%/12.9%/7.1%`。FP16 对应为 `0.31892/0.13383/0.02712/0.03652/0.53965/0.08249 m`；除 FSQ bin accuracy 的相对差约 `-0.177%` 外，主要连续指标对 FP32 的相对差小于约 `0.13%`。

15k 的原版闭环进一步定位了分布偏移的时间尺度：第 5 个续写窗的 waypoint mean 从 10k 的 `1.209 m` 大幅改善到 `0.404 m`，说明短程路径跟随正在学会；但第 20/50 窗的 waypoint 为 `6.724/16.417 m`，50-window mean drift/final drift/FK 为 `15.909/45.195/15.943 m`。与 10k 比，50-window foot、root acceleration/jerk/seam、joint acceleration/jerk/HF/seam 继续改善约 `10%–15%`，但 waypoint/mean drift/FK 再恶化约 `6.7%–7.1%`、final drift 恶化 `21.9%`。这不是“网络完全没学会路径”，而是学生续写数窗后离开 offline teacher state distribution，随后的单窗静态改善无法自行修复长闭环。因此保留既定 100k supervised 主训练，同时把“从最终 EMA 真实 rollout state 向原版 teacher 查询纠偏目标”设为 DMD2 前的必要 gate；不用重启随机学生代替 on-policy 覆盖。

为此已将 `generate_onpolicy_teacher_shards.py` 从旧的默认 3-block encoder/4-block decoder/5+2 flow 硬编码改为显式可配置，当前默认与正式 4-block encoder、8+8 flow、8-block decoder 一致，并记录 root projection 配置。一份 2-window、student FP16、teacher FP32 shard 的工程 smoke 已确认：student 执行直接 NFE1，原版 teacher 执行 10-step + CFG constraint `1.5`，history fraction `0.5`、constraint fraction `1.0`，shape/finite/SHA 全部通过。目录 `onpolicy_teacher_standard48m_step0015000_smoke2_20260715/` 已写入 `ENGINEERING_SMOKE_ONLY.md`，不得混入训练；正式 corpus 只会从后续选定的 supervised EMA 生成。改造通过编译、CLI 和 11 项现有 shape/flow/EMA 单测，没有安装任何包。

20k 的独立 1024-window 静态验证继续改善且 EMA 继续胜 raw。15k→20k EMA FP32 的 endpoint total、FK、foot slide、joint jerk、root temporal 和 path error 从 `0.31890/0.13381/0.02710/0.03648/0.53905/0.08251 m` 改善到 `0.28017/0.12164/0.02483/0.03300/0.49010/0.07930 m`，下降约 `12.1%/9.1%/8.4%/9.5%/9.1%/3.9%`。EMA FP16 对应为 `0.28015/0.12164/0.02484/0.03304/0.49068/0.07924 m`，主要相对差继续在约 `0.15%` 以内。这证明 supervised mapping 仍在学习，不支持因闭环失败就在 20k 丢掉已有权重或提前放弃既定长程训练。

但 20k EMA 的 seed `20260714..20260718` 五条原版 50-window 评估表明，长闭环条件覆盖不能由静态收敛代替。第 5/20/50 窗的 waypoint mean 五 seed 均值为 `1.502/8.882/21.868 m`；50 窗范围为 `10.262–31.114 m`，mean drift、final drift 和 FK 均值为 `21.374/43.741/21.405 m`。50-window foot slide 为 `0.565 m/s`，root jerk、joint jerk、joint HF、root seam、joint seam 对 teacher 的均值为 `23.77×/2.56×/12.26×/10.98×/4.50×`，且五条路径都没有达到可用控制。所以单 seed 15k 在第 5 窗的 `0.404 m` 只能说明网络存在短程能力，不是可泛化的 checkpoint 结论。正式 supervised run 继续；最终选定 EMA 后先用同一权重生成 student-state teacher correction corpus、再与原 corpus replay 混合续训，不新建随机 student，也不用 DMD2/critic 去弥补缺失的条件分布。记录位于 `eval/flow_standard48m_step0020000_*`、`rollout/flow_standard48m_step0020000_ema_oracle50_seed*/` 和同名 jitter JSON。

25k 首个完整 flow 恢复点已通过只读审计。state model/EMA 各有 213 个 tensor、`33,910,420` 参数，Adam 有 213 组 state/639 个浮点 tensor，全部 finite；所有 Adam step 精确为 `25000`，scheduler `last_epoch/_step_count=25000/25001`、LR `5e-5`，EMA decay/update count 为 `0.9995/25000`，sampler seed/epoch 为 `20260714/25`，四份 rank RNG 状态齐全。state model 对 raw export、state EMA shadow 对 EMA export 的逐 tensor max abs 都为 `0`；所以该点能完整恢复 model/optimizer/scheduler/EMA/sampler/RNG，不是只能 warm-start 的推理权重。

25k 静态仍由 EMA 胜 raw，且 20k→25k EMA FP32 的 endpoint total、FK、foot slide、joint jerk、root temporal 和 path error 从 `0.28017/0.12164/0.02483/0.03300/0.49010/0.07930 m` 改善到 `0.25667/0.11281/0.02327/0.03065/0.45688/0.07671 m`，分别下降约 `8.4%/7.3%/6.3%/7.1%/6.8%/3.3%`。FP16 对应为 `0.25660/0.11280/0.02329/0.03069/0.45737/0.07670 m`，主要相对差小于约 `0.16%`。同 seed `20260714` 的第 5/20/50 窗 waypoint 相对 20k 从 `1.182/9.185/18.779 m` 改善到 `0.767/3.811/16.273 m`，joint HF 从 `11.09/9.04/10.24×` 降到 `7.27/7.28/8.98×`；但 50-window mean/final drift 仍为 `15.732/46.356 m`，第 5 窗 root/joint seam 还从 `9.41/5.28×` 反弹到 `17.56/6.03×`。因此当前证据支持从已保存的完整 25k state 继续既定长程训练，不支持发布 25k、停掉 on-policy correction 或改用随机初始化重训。记录位于 `state/step-0025000/`、`eval/flow_standard48m_step0025000_*` 和 `rollout/flow_standard48m_step0025000_ema_oracle50_seed20260714/`。

on-policy 生成器又以中途 25k EMA 完成了一次更接近正式参数的 batch-8 smoke：16 个窗口、student FP16、shard FP16、原版 teacher 10-step + CFG constraint `1.5`，实测单卡 `17.4 windows/s`；history fraction `0.5`、constraint fraction `1.0`，shape/finite/SHA 均通过。按四卡线性粗估，32k corrective windows 约为数分钟到十分钟量级，不是 12h 主要瓶颈。目录 `onpolicy_teacher_standard48m_step0025000_b8_smoke16_20260715/` 已标记 `ENGINEERING_SMOKE_ONLY`；它只证明批处理路径，因来自中途权重，禁止混入正式训练。

30k 静态仍未平台且 EMA 仍明显胜 raw。25k→30k EMA FP32 的 endpoint total、FK、foot slide、joint jerk、root temporal 和 path error 从 `0.25667/0.11281/0.02327/0.03065/0.45688/0.07671 m` 改善到 `0.23575/0.10541/0.02216/0.02880/0.42999/0.07280 m`，分别下降约 `8.1%/6.6%/4.8%/6.0%/5.9%/5.1%`。EMA FP16 对应为 `0.23567/0.10541/0.02219/0.02887/0.43082/0.07278 m`，主要指标相对差仍小于约 `0.22%`。同 seed `20260714` 的第 5/20 窗 waypoint 从 25k 的 `0.767/3.811 m` 回摆到 `1.247/8.361 m`，但 50-window waypoint/final drift 又从 `16.273/46.356 m` 改善到 `14.824/33.022 m`，joint jerk/HF/seam 仅小幅变化。这是长闭环对小条件误差敏感的又一证据：不按单 seed 某一中间窗挑 checkpoint，不因回摆丢掉完整续训状态；最终以五 seed + on-policy 纠偏后的长闭环为准。记录位于 `eval/flow_standard48m_step0030000_*` 和 `rollout/flow_standard48m_step0030000_ema_oracle50_seed20260714/`。

用户再次确认“已有权重继承优先”为项目全局硬规则，其优先级高于为了目录整洁、改运行时长或切换训练会话而重新初始化。当前正式 8+8 flow 在记录时已连续训练至 step `33250`，起点仍是 g200 EMA 的全部 `14,985,364` 个兼容参数，只有无来源的新增残差 block 做了 identity init；之后的 on-policy 纠偏 student、DMD2 generator 和 fake score 都必须从当时最优已有 EMA 继承。同构且是同一训练阶段的中断恢复必须优先加载完整 state，不能退化为只加载权重；只有训练目标或数据分布已切换成新 stage 时，才可以保留最优 EMA 权重而重建新 stage 的 optimizer。训练专用 critic 等模块也要先审计旧权重的结构和输入定义；只有无兼容来源或目标在数学上不兼容时才能新初始化，并记录例外理由。

后续 DMD2 训练的部署精度一致性也完成了代码审计：`train_dmd2.py` 原先会让冻结 encoder/decoder 被外层 BF16 autocast 接管，已改为显式 `--frozen-codec-dtype fp16`，codec 权重和算术均使用 FP16，且在 codec 前向内关闭 autocast；可训 flow/score/critic 仍使用 BF16。真实 100k EMA codec 的 CUDA batch-2 检查确认 encoder/decoder 参数均为 FP16，history 路径与显式 FP16 reference 逐元素完全相等，decoder 输出 shape 为 `[2,44,325]`、root 为 `[2,44,5]`，全部 finite，且 endpoint 反传梯度 L2 为 `1.04366`。代码编译、CLI 和现有 15 项单测通过；本轮没有安装包、没有执行 `conda` 命令。

35k 快照继续沿用同一继承链并由 EMA 明显胜 raw。与 30k EMA FP32 的 endpoint total/body L1、FSQ center abs bins、FK、foot slide、joint jerk、root temporal 和 path error `0.23575/0.27331/2.84808/0.10541/0.02216/0.02880/0.42999/0.07280 m` 相比，35k EMA 为 `0.21765/0.26024/2.70989/0.09901/0.02125/0.02728/0.40746/0.07036 m`，分别改善约 `7.68%/4.78%/4.85%/6.07%/4.11%/5.29%/5.24%/3.36%`；FSQ bin accuracy 从 `0.12067` 提到 `0.12619`。35k EMA FP16 的对应值为 `0.21761/0.26021/2.70956/0.09899/0.02128/0.02733/0.40816/0.07034 m`，与 FP32 主要相对差不超过约 `0.21%`。本次评估首先暴露出历史 JSON 未记录 data root，一次误用 `teacher_val` 的输出已在发现冻结 codec 指标不可对齐后作废并用原固定 1024-window `teacher_train_large_v2` 口径覆盖重跑；`evaluate.py` 现在强制在结果里记录 `data/batch_size/max_batches`，避免再把数据分布变化误当模型变化。

35k EMA 的同 seed 50-window 闭环仍只能当中途诊断。30k→35k 的第 5/20/50 窗 waypoint 从 `1.247/8.361/14.824 m` 变为 `1.220/8.739/14.362 m`；50-window mean drift/FK 从 `14.430/14.469 m` 改善到 `13.926/13.964 m`，但 final drift 从 `33.022` 回摆到 `38.310 m`。foot slide 从 `0.544→0.525 m/s`，root acceleration/jerk、joint acceleration/jerk、joint HF、root/joint seam 对 teacher 从 `11.20/24.46/1.44/2.77/8.90/9.58/4.72×` 变为 `10.57/22.87/1.32/2.55/8.78/9.20/4.57×`；除 root HF `13.24→13.25×` 基本不变外其余平滑项有小幅改善。这再次证明 offline 静态收敛与单 seed 长闭环并非单调对应；不因 35k 的 final drift 回摆重启权重，也不用该点替换既定 100k + on-policy 纠偏路线。记录位于 `eval/flow_standard48m_step0035000_*` 和 `rollout/flow_standard48m_step0035000_ema_oracle50_seed20260714/`。

on-policy/replay 纠偏训练的继承链已做真实两次 update 工程演练，而不是只验证 CLI。同构 8+8 flow 从 35k EMA 完整加载，没有任何随机重启；第 1 个 batch 来自 on-policy smoke shard，第 2 个 batch 来自 131,072-window 原语料 replay，两次都以固定 `1e-5`、BF16 trainable flow、FP16 冻结 codec、FP32 AdamW/EMA 完成。step-2 raw/EMA 的 213 个 tensor、`33,910,420` 参数全 finite，213 组 Adam step 都精确为 2，scheduler `last_epoch=2`、LR `1e-5`，mixture sampler 的 seed/epoch/chunk/probability 为 `20260714/1/8/0.25`，RNG 也落盘。目录 `flow_standard48m_onpolicy_replay_warm35k_smoke2_20260715/` 已写 `ENGINEERING_SMOKE_ONLY.md`；它只证明“从已有 EMA 继承 + 新 stage optimizer + 可恢复 replay sampler”的链路，正式语料仍必须由最终 supervised EMA 重新生成。

on-policy corpus QA 已升级为 v2：除总体 path norm 外，强制按 rollout depth 记录样本数、history 标记、path norm 与 student/reference root drift 的 mean/p50/p95/p99/max，并记录总 history/constraint fraction。一份 35k EMA、batch-2、rollout-depth-2 的 4-window 工程 shard 已通过 shape/finite/SHA 验证：总 path p95 `1.849 m`、第 0/1 层 student-reference drift p95 `0/0.701 m`，history fraction `0.5`、constraint fraction `1.0`。目录 `onpolicy_teacher_standard48m_step0035000_v2stats_smoke4_20260715/` 也已标记为不可训练的 smoke。正式 corpus 将以这些分层统计判断后半 rollout 是否充斥不可达条件，不再用一个总均值掩盖 OOD 尾部。

40k 固定 1024-window 继续给出明确监督收益。35k→40k EMA FP32 的 endpoint total/body L1、FSQ center abs bins、FK、foot、joint jerk、root temporal、path error 从 `0.21765/0.26024/2.70989/0.09901/0.02125/0.02728/0.40746/0.07036 m` 改善到 `0.20187/0.24832/2.58425/0.09238/0.02042/0.02593/0.38734/0.06718 m`，分别下降约 `7.25%/4.58%/4.64%/6.70%/3.91%/4.94%/4.94%/4.52%`；FSQ bin accuracy 再提升 `4.71%`。40k EMA 对 raw 的 endpoint/FK/foot/jerk/root-temporal/path 改善约 `9.83%/11.84%/25.43%/31.88%/29.10%/13.25%`。FP16 对 FP32 主要相对差不超过约 `0.18%`，所以继续以 EMA 和部署 FP16 为主候选。

40k EMA 又完成 seed `20260714..20260718` 五条、每条 50-window、无惯性化的原版语义闭环。相对同五 seed 的 20k EMA，waypoint 第 5/20/50 窗均值从 `1.502/8.882/21.868 m` 改善到 `1.367/7.269/19.315 m`；50-window mean drift/final drift/FK/foot 从 `21.374/43.741/21.405 m/0.565 m/s` 改善到 `18.886/40.187/18.917 m/0.497 m/s`。root acceleration/jerk/HF、joint acceleration/jerk/HF、root/joint seam 均值从 `10.76/23.77/15.10/1.34/2.56/12.26/10.98/4.50×` 改善到 `8.97/20.16/14.93/1.12/2.15/10.12/9.17/3.87×`。foot、seam、acceleration、jerk 和 joint HF 均为逐 seed `5/5` 改善；waypoint/mean drift/FK 为 `3/5` 改善，final drift `4/5`，root HF `4/5`。因此 20k→40k 的训练收益在多 seed 时依然成立，不是 35k 单 seed 的偶然；但 40k 的 waypoint 仍为 `11.514–31.898 m`，绝对质量远未可用，仍必须完成 100k 后进入 on-policy 纠偏。机器汇总位于 `eval/flow_standard48m_formal_sweep_020k_040k_ema.json`。

首选的 rectified-flow DMD2/ADV 入口 `train_flow_dmd2.py` 已增加原语料 replay 混合与可恢复的 mixture sampler，并把继承语义写入运行配置：generator 必须从纠偏阶段选定的现有 EMA warm-start，fake score 在 stage 起点逐 tensor 精确复制 generator；只有新训练目标下确实没有兼容状态的 critic 才允许新初始化，同一 DMD2 stage 的后续运行必须完整恢复 generator/fake-score/critic、三组 optimizer/scheduler、EMA、计数器、sampler 和 RNG。这里的“新 stage optimizer”不等于“新模型”：进入纠偏或 DMD2 时仍继承上一阶段最佳 EMA，只因目标函数和数据分布发生了数学变化而新建对应 optimizer；若同一 stage 中断，则优先完整 state resume，禁止退化为只载入权重。

上述路径完成了一次真实 generator update 工程烟测，目录 `flow_dmd2_standard48m_warm35k_replay_smoke1_20260715/` 已标为 `ENGINEERING_SMOKE_ONLY`，禁止选模、导出或展示。generator 与 frozen real-score teacher 均由同一个 35k supervised EMA 加载，fake score 起点为 generator 的精确副本；训练专用 critic 因该 rectified-flow stage 没有任何结构/输入定义兼容的旧 state 才作为唯一例外新初始化。primary/replay mixture、冻结 FP16 codec、可训 BF16 flow/score/critic、FP32 AdamW/EMA 均完成；step 1 的 generator/fake-score/critic 分别为 `33,910,420 / 33,910,420 / 4,726,785` 参数，三组 optimizer state 数量为 `213/213/18`，scheduler、sampler、EMA 与 RNG 均落盘且全部 finite。裁剪前 generator/score/critic grad norm 为 `2.3755/6.7609/0.6069`；LR 为 `1e-6/5e-6/1e-6`，这只是验证实现的单步烟测，不代表训练充分或质量可用。

只读核对用户指定的 `<REFERENCE_3DLM_PROJECT>/20250425_zlw_clayv2_train_ldm_fsdp2_dmd2.py` 后，参考代码对 student generator、fake score、discriminator 的 gradient clip 分别是 `1e-2/1.0/1.0`；当前首选入口与其逐项一致，且没有修改 `<REFERENCE_3DLM_PROJECT>`。单步烟测显示 generator 的裁剪前梯度明显高于 `1e-2`，所以正式短程 DMD2 会把“实际更新幅度、EMA 相对初始纠偏权重的位移、闭环指标随 gate 的变化”作为必审项目；不能只看 loss finite 就认为分布蒸馏有效，也不能在没有多 gate 证据时随意改掉参考裁剪值。正式 DMD2 只在 100k supervised EMA 和 on-policy/replay 纠偏通过后启动，并继续继承当时最佳 EMA，绝不从噪声重新训练 generator。

45k 快照继续由同一完整训练状态向前产生，没有任何重启。固定 1024-window 上，40k→45k EMA FP32 的 endpoint total、body L1、FSQ center abs bins、FK、foot、joint jerk、root temporal 和 path error 从 `0.20187/0.24832/2.58425/0.09238/0.02042/0.02593/0.38734/0.06718 m` 改善到 `0.19126/0.23991/2.49557/0.08820/0.01986/0.02496/0.37251/0.06494 m`，分别下降约 `5.26%/3.39%/3.43%/4.52%/2.77%/3.76%/3.83%/3.33%`，FSQ bin accuracy 从 `0.13213` 提到 `0.13708`。45k EMA 对 raw 的 endpoint/FK/foot/jerk/root-temporal/path 分别改善约 `13.65%/13.25%/23.22%/28.76%/26.69%/21.04%`。EMA FP16 的 endpoint/FK/foot/jerk/root-temporal/path 为 `0.19117/0.08816/0.01988/0.02499/0.37282/0.06488 m`，相对 FP32 的主要差异仍不超过约 `0.13%`；因此监督映射尚未平台，EMA 与部署 FP16 路线都继续成立。

45k EMA 的固定 seed `20260714` 50-window 原版语义闭环同时保留了分布偏移反例。40k→45k 的第 5/20 窗 waypoint 从 `1.176/5.249 m` 改善到 `0.999/3.448 m`，foot slide 从 `0.504→0.495 m/s`；root jerk/HF、joint acceleration/jerk/HF、root/joint seam 从 `21.85/12.97/1.327/2.542/8.340/9.001/4.351×` 改善到 `21.76/12.80/1.305/2.478/7.615/8.368/4.089×`，仅 root acceleration 从 `9.84×` 小幅回摆到 `10.00×`。但第 50 窗 waypoint、mean drift、final drift、FK 从 `13.985/13.536/40.110/13.551 m` 回摆到 `15.483/14.941/45.951/14.924 m`。所以 45k 的短程控制与局部平滑确实继续学习，长闭环仍被 student-state 分布偏移支配；该单 seed 回摆不构成随机重启、提前选模或跳过 100k 的理由，反而继续支持“最终 EMA → on-policy teacher correction → 原语料 replay”的既定继承链。

### 23.14 100k flow 锁定与正式 on-policy 纠偏

正式 8+8 flow 已从既有 g200 EMA 继承链连续跑满 `100,000` 个真实 optimizer update，以 `training_complete / stopped_by_runtime=false` 正常退出，没有重启、随机回退或 optimizer skip。50k 与 100k 两个完整 Accelerate state 均通过只读审计：model/EMA 各为 213 个 tensor、`33,910,420` 参数，Adam 为 213 组 state、639 个浮点 tensor；对应 Adam step、scheduler epoch 和 EMA update count 分别严格为 `50000/100000`，全部元素 finite。EMA decay 固定为 `0.9995`；sampler、四 rank RNG 均存在。两个 state 的 model 对 raw export、EMA shadow 对 EMA export 的逐 tensor max abs 都为 `0`。日志中 step 49,750 仍显示 `5e-5`，完成 step 50,000 后显示 `1e-5`，后 50k 始终固定 `1e-5`，无 cosine、warmup 或 off-by-one。

50k→100k 每 5k 的独立固定 1024-window EMA 静态曲线在 endpoint total/body L1、FSQ bin accuracy/center distance、FK、foot slide、joint jerk、root temporal 和 path error 九个核心项上全部严格单调改善；该扫点只用于验证训练协议和末段是否退化，不再用于耗时挑 checkpoint。用户明确指出只要蒸馏协议正确即可机械采用最后一步，后续因此直接锁定 `100k EMA` 为唯一继承源。100k EMA FP32 的 endpoint total/body L1、FSQ bin accuracy/center abs bins、FK、foot、joint jerk、root temporal、path error 为 `0.14228/0.19874/0.16962/2.06140/0.06524 m/0.01584/0.01792/0.26693/0.05658 m`。相对 100k raw，EMA 的 FK、foot、joint jerk、root temporal、path 分别改善约 `1.75%/4.60%/9.08%/8.44%/3.58%`。部署 FP16 对 EMA FP32 的 endpoint/FK/path 差异约 `-0.073%/-0.076%/-0.085%`，foot、joint jerk、root temporal为 `+0.233%/+0.540%/+0.490%`；静态仍可接受，但最终整链路 WebGPU gate 不能省略。

100k EMA 的五 seed 原版 50-window 闭环说明离线监督已把动作本身显著变稳，却没有解决 student-state 条件分布。40k→100k 的静态 FK 降低 `29.37%`；五 seed foot、joint seam、root acceleration/jerk、body acceleration/jerk/HF、joint seam ratio 均值分别改善约 `19.17%/17.72%/27.67%/26.95%/17.32%/20.91%/29.10%/18.02%`，且这些项逐 seed 全部 `5/5` 改善。但 50-window waypoint 均值为 `19.332 m`，与 40k 的 `19.315 m` 基本不变；mean drift/FK 也仍约 `18.88/18.90 m`，final drift 均值为 `45.26 m`。这不是改拿旧 checkpoint 的理由，而是直接进入最终 EMA on-policy correction 的证据。机器汇总为 `eval/flow_standard48m_formal_sweep_040k_100k_ema.json`。

正式 corrective corpus 已由锁定的 100k EMA student 生成，不从旧中途 smoke 或随机 student 生成。物理 GPU `3,4,5,6` 各写 8,192 窗，共 `32,768` 窗；student 使用部署 FP16 和 NFE1，released teacher 使用原版 FP32 10-step + constraint CFG `1.5`，rollout depth 为 8。目录 `onpolicy_teacher_standard48m_step0100000_ema_r8_4gpu_32k_fp16_20260715/` 含 4 个 manifest、128 个 FP16 shard、总计 `2,425,142,272` bytes；SHA256、字段 schema、shape 与 finite 全通过。history fraction 为 `0.875`，constraint fraction 为 `1.0`。四 rank 总体 path norm p95 约 `5.82–5.95 m`，student-reference drift p95 约 `7.17–7.41 m`；按 depth 看，第 1 层 drift p95 约 `0.70–0.73 m`，第 7 层约 `9.62–9.79 m`。这些晚层是故意覆盖 recovery/OOD 的困难条件，并与 50% 原语料 replay 配合，不能把它误当普通 iid teacher corpus。

正式纠偏 run 已启动为 `flow_standard48m_onpolicy_r8_replay50_warm100kema_4gpu_b32_const1e5_20k/`。它严格加载 100k `flow_ema.safetensors` 的全部 213 个 tensor/`33,910,420` 参数，没有任何新 flow tensor 或噪声初始化；冻结 codec 仍为 100k EMA FP16。由于数据分布和 sampler 切换为新的 on-policy/replay stage，optimizer/EMA 从该 EMA 权重点新建，而不是伪装成同一 sampler 的完整 resume；后续若这个纠偏 stage 中断，则必须完整恢复它自己的 model/optimizer/scheduler/mixture sampler/RNG/EMA state。训练使用四卡、每卡 batch 32/global 128、BF16 flow、FP32 AdamW/固定 `0.9995` EMA、50% on-policy + 50% 原 131,072-window corpus、固定 LR `1e-5`、20k update，所有 loss 系数、NFE1、70% exact-t1 + 20% high-noise 均与监督末段一致，无 cosine/warmup/weight decay。step 1 全链路 loss/gradient finite；该点不是质量结论，必须完成 20k 后按相同五 seed 闭环证明 recovery 是否改善。

纠偏 stage 的 10k 完整状态已在训练不中断时通过只读审计。state model 与 EMA 各含 213 个 tensor、`33,910,420` 参数，Adam 含 213 组 state/639 个 tensor；全部 floating tensor finite，所有 Adam step 精确为 `10000`。scheduler `last_epoch/_step_count=10000/10001`，LR 固定 `1e-5`、weight decay 0；EMA decay/update count 为 `0.9995/10000`。mixture sampler 保存 `seed=20260714`、epoch `10`、chunk size `32`、samples/epoch `131072`、primary probability `0.5`，四份 rank RNG 均含 Python/NumPy/Torch/CUDA 状态。state model 对 10k raw export、EMA shadow 对 10k EMA export 的 key 完全一致且 max abs 均为 `0`。这个中间点只证明同一 stage 可完整恢复，不参与效果选模；仍机械训练到并继承 20k 最终 EMA。

纠偏 stage 随后完整跑满 `20,000` 个真实 optimizer update，以 `training_complete / stopped_by_runtime=false` 正常退出，wall time 为 `3064.73 s`。最终完整状态再次通过同口径审计：model/EMA 各 213 个 tensor、`33,910,420` 参数，Adam 213 组/639 个 tensor全部 finite；Adam step 全为 `20000`，scheduler `last_epoch/_step_count=20000/20001`，LR `1e-5`、weight decay 0，EMA decay/update count 为 `0.9995/20000`。mixture sampler 为 epoch `20`、primary probability `0.5`，四 rank RNG 齐全；state model 对 raw export、EMA shadow 对 EMA export 的逐 tensor max abs 均为 `0`。依照最终步规则，后续唯一纠偏继承源固定为 `weights/step-0020000/flow_ema.safetensors`，不再评估 10k 或其他中间权重作为候选。

20k 最终 EMA 的固定 1024-window 对拍证明纠偏确实学到了 student-state recovery。相对纠偏前 100k EMA，on-policy FP32 endpoint total 从 `2.79327→0.25227`、FK 从 `1.33372→0.14505 m`、path error 从 `3.17909→0.47635 m`，分别改善约 `90.97%/89.12%/85.02%`；同 batch teacher path error 为 `0.82095 m`。原语料代价是 endpoint/FK/path 从 `0.14228/0.06524/0.05658 m` 回到 `0.15986/0.07588/0.06274 m`，约回退 `12.35%/16.30%/10.88%`，这是 50/50 recovery 与 replay 的明确 trade-off，不隐瞒为全面静态改善。FP16 与 FP32 紧密一致：on-policy endpoint/FK/path 为 `0.25218/0.14500/0.47620 m`，原语料为 `0.15976/0.07584/0.06268 m`。

更关键的五 seed、每条 50-window、无惯性化、部署 FP16 原版无限闭环已把 exposure-bias 失败真正修复：waypoint mean `19.332→0.142 m`、mean/final root drift `18.882/45.258→1.041/1.661 m`、FK `18.901→1.107 m`，分别改善 `99.26%/94.48%/96.33%/94.14%`，且逐 seed 全部 `5/5` 改善。joint seam、root acceleration/jerk/HF、body HF 与 joint seam ratio 也分别改善约 `15.12%/13.26%/16.70%/50.59%/48.79%/14.22%`，均为 `5/5`；代价是 foot slide `0.402→0.434 m/s`（`+8.19%`）以及 body acceleration/jerk ratio `0.929/1.699→1.352/2.080`（`+45.43%/+22.40%`），同样是五条都回退。机器汇总为 `eval/flow_standard48m_correction20k_vs_supervised100k.json`。因此纠偏最终 EMA 作为 DMD2 warm-start 的依据充分，但短程分布/对抗阶段仍必须重点压 body jerk/foot，而不能破坏已恢复的 waypoint。

### 23.15 DMD2 最终步、FP16 WebGPU 导出与整链路 gate

短程 DMD2/ADV 严格从纠偏 `20k EMA` 继承 generator，fake score 在 stage 起点逐 tensor 克隆 generator；只有输入定义和目标均无兼容旧状态的 2-block critic 新初始化。正式 run 使用物理 GPU `3,4,5,6`、每卡 batch 32/global 128、BF16 generator/fake-score/critic、冻结 FP16 codec、FP32 AdamW/EMA；generator/score/critic 固定 LR `1e-6/5e-6/1e-6`，weight decay/warmup/cosine 均为 0，EMA decay `0.995`，先做 200 次 guidance warmup，再做 1000 次 generator update。DMD/adv/paired/FSQ 权重为 `1/0.001/0.1/0.1`，generator/score/critic clip 为 `0.01/1/1`。任务以 `training_complete`、`stopped_by_runtime=false` 在 generator step 1000 正常结束，wall time `321.80 s`；全程 loss、gradient、权重均 finite。

最终 `step-0001000` 完整状态审计通过：generator/fake-score 各 213 个 tensor、`33,910,420` 参数，critic 18 个 tensor、`4,726,785` 参数；三组 optimizer state 数量为 `213/213/18`，step 精确为 `1000/1200/1200`，三组 scheduler 同步，EMA update/decay 为 `1000/0.995`，计数器为 iterations/guidance/generator `1200/1200/1000`，mixture sampler 与四份 rank RNG 齐全。state generator/fake-score/critic 对各自 raw export、EMA shadow 对 EMA export 的逐 tensor max abs 均为 `0`。500-step state 只作恢复审计，没有参与选择；按锁定规则机械采用最终 `weights/step-0001000/flow_ema.safetensors`。

最终 EMA 的 1024-window 静态 FP16/FP32 仍一致。原语料 FP32 endpoint/FK/path 为 `0.17291/0.07813/0.06180 m`，FP16 为 `0.17281/0.07811/0.06175 m`；on-policy FP32 为 `0.26725/0.14441/0.51130 m`，FP16 为 `0.26711/0.14438/0.51116 m`。五 seed、每条 50-window、无惯性化的最终 FP16 闭环没有崩溃，但 DMD2 相对纠偏 20k 末步不是全面收益：waypoint `0.14213→0.15073 m`（`+6.05%`）、FK `1.10727→1.11143 m`（`+0.38%`）、joint seam `0.80067→0.85773 m/s`（`+7.13%`）、body jerk ratio `2.07961→2.14105`（`+2.95%`）；body 高频比 `3.67395→3.21424`（`-12.51%`）是唯一明显且逐 seed `5/5` 的稳定改善。该结果用于如实判断 stage 效果，不触发回头挑 500-step；完整汇总为 `eval/flow_standard48m_dmd2_final1000_vs_correction20k.json`。

最终发布 `standard48m_dmd2_final1000_ema_fp16` 已按相同 4/8+8/8-block 结构导出：encoder `9,882,793 B`、flow `67,929,761 B`、decoder `18,370,139 B`，学习图 `96,182,693 B`；两个 FP32 root/FSQ utility 图 `45,264 B`，完整 ONNX 下载 `96,227,957 B`（`91.77 MiB`），严格低于 `100,000,000 B`。三个 learned graph 的 CPU ORT 相对 PyTorch FP32 max abs 为 `0.001197/0.006625/0.006593`，全部 finite。

服务器端初始段+续写段整链路 gate 已通过：连续 flow endpoint 对 PyTorch FP32 的最坏 max abs 为 `0.01552`；跨不连续 FSQ 后，最终 explicit motion 对 FP32 的最坏 mean/p99 abs 为 `0.01021/0.05062`，最低 cosine `0.9999326`，最低 FSQ bin match `94.84375%`。稀疏单元素 max abs `0.2583` 仍原样记录，不用放宽一个总 max 掩盖；验收对连续 endpoint 保留 max gate，对离散 FSQ 后输出使用 bin match + mean/p99 + cosine。首段/续写均为 40 帧，第二次生成后的原版 buffer 精确为 77 帧，NFE=1、无 runtime CFG、无 text/Llama、无前端惯性化。

页面新增不改 demo 状态的浏览器 golden-case 整链路对拍：Edge 会用 WebGPU encoder/flow/decoder + WASM root/FSQ utility 运行固定 initial/continuation 输入，并与服务器 CPU ORT 的 clean endpoint、history hybrid 和 40-frame motion 直接比较。8766 服务已重启并排入远程 `validate` job `408442ee426a49f1b1a71bc45639678f`；当前尚无浏览器 client，因此真实 WebGPU 数值、前台 p50/p95 和动画视觉仍明确标记为待验收，不能用服务器通过代替。最终页面为 `http://127.0.0.1:8766/infinite_demo.html`。本阶段没有安装任何包、没有执行 `conda` 命令，也没有修改 `<REFERENCE_3DLM_PROJECT>`。

导出后的代码回归也已用现有 py311 环境完成：9 个 rectified-flow/DMD2 语义测试、4 个 shard-mixture/sampler 恢复测试、4 个 EMA/horizon/恢复测试，以及 5 组 codec/flow/critic 结构、形状、finite 与 function-preserving expansion 检查全部通过。环境没有 `pytest`，因此直接使用测试文件自带的 `unittest`/原生入口；没有为测试安装任何依赖。首次直接启动两项测试时缺少仓库内层包路径，加入 `PYTHONPATH=ardy:.` 后通过，这属于启动路径问题，不是模型或测试失败。

最终 4/8+8/8-block 权重又重新经过原版 runtime 条件语义审计。历史验证器原先写死类默认的 3/4+2/4-block 配置，无法加载正式权重；现已只为该验证器补充显式 encoder/flow/decoder 架构参数，不修改模型或 checkpoint。使用最终 EMA 后，initial 与 continuation 的 `path_condition`、`first_heading`、`has_history`、`global_translation`、`history_root` 对 released teacher trace 的最大绝对误差全部精确为 `0`，输出形状严格为 `[1,40,330]` 与 `[1,44,330]`，全部 finite。机器结果为 `distill_runs/first_12h_20260714_014643/eval/final_dmd2_1000_runtime_semantics.json`。

### 23.16 真实 Edge FP16 失败基线与部署数值纠偏

Edge 149/NVIDIA Ampere 已实际领取最终 release 的 golden-case 任务。结果全部 finite，但严格数值 gate 失败，证明 CPU ORT 对 FP16 ONNX 的通过不能替代真实 WebGPU：initial/continuation flow endpoint max abs 为 `0.08789/0.08643`，FSQ bin match 仅 `85.47%/83.05%`；跨 FSQ 与 decoder 后 motion mean abs 为 `0.04454/0.03881`、p99 为 `0.33399/0.22956`、cosine 为 `0.99718/0.99888`。这些均明显差于预先锁定门槛，因此不放宽阈值、不把“finite”冒充精度通过，也与此前用户直接观察到 FP16 动画质量差一致。机器摘要为 `webgpu_toy/infinite_demo/edge_fp16_baseline_20260715.json`。

同一真实前台做了 3 warmup + 20 timed-run：每 40 帧 total p50/p95 为 `127.97/154.86 ms`，encoder `13.53/22.79 ms`，NFE1 flow `75.32/93.14 ms`，decoder `36.86/47.28 ms`。延迟仍过当前 p95 `170 ms` 硬线，所以当前首要问题是 WebGPU FP16 数值而不是继续砍网络。页面也已真实运行到数百帧并接收多个 `current+60` 稀疏 waypoint，但数值 gate 未过前不能据此宣布视觉版本交付。

下一步采用 deployment-numerics-aware distillation，但不盲加独立高斯噪声。已生成诊断专用的 encoder 7 点、flow 37 点、decoder 19 点逐层 ONNX 探针及同一 continuation golden case；探针只复用最终 FP16 权重、暴露既有中间 tensor，不改变生产图或模型。将先用 Edge 对拍定位误差是在 attention residual、channel MLP、token mixer 还是 FSQ 前输出逐层累积，再据此实现带 STE 的 FP16 fake-round/低精度累加模拟器。训练必须从当前最终 EMA 继承，使用四卡做部署数值纠偏，不能重新随机初始化；重新导出后仍需同时满足 `<100MB`、NFE1、真实 Edge 数值、延迟和无限动画视觉 gate。

### 23.17 文本条件、全分布 teacher corpus 与四卡训练准备（2026-07-15）

运行时文本条件已按用户指定的双编码器职责实现：released teacher 继续使用原版 Llama-3-8B/LLM2Vec；student 使用本地 `<HUGGINGFACE_MODELS>/black-forest-labs/FLUX.2-klein-4B/` 内的 Qwen3 text encoder。两者都只是 prompt 预处理器，不计入当前运动 student 的参数或 WebGPU 每窗延迟。Qwen 以 BF16 推理，取 FLUX.2 对应的第 9/18/27 层 hidden state 拼接后做有效 token mean pooling，得到完整 `7680D` feature；Llama 得到 `4096D` teacher feature。没有 PCA、SVD、低秩或其他文本压缩。两张表均以 FP16 缓存且全部 finite：Qwen `[8192,7680]`，Llama `[8192,4096]`，prompt 0 为严格全零的 unconditional feature。

student flow 保留 width `512`、8 heads、8 trunk blocks、8 body blocks和直接 NFE1，不砍层数。在已有 path-only EMA 上只新增 full `Linear(7680,512,bias=False)` 文本投影和四帧一组的 heading `Linear(12,512)` 投影，二者均零初始化，所以 step 0 对任意文本/朝向输入都与旧 EMA 逐元素完全一致。文本投影增加 `3,932,160` 参数，heading 投影增加 `6,656`；encoder + flow + decoder 总计从 `48,013,184` 增至 `51,952,000` 参数，纯 FP16 参数为 `103,904,000 B`。Qwen/Llama 不计在该数值内，100 MB 仍只作近似容量目标，优先级低于数据分布和效果。

prompt bank 已确定性扩展为 `8,192` 条，包含 16 个动作 family、10 条官方 preset、unconditional、不同速度/方向/风格、前进/后退/侧移/转向/启停、idle/crouch/jump/dance/gesture/combat/sport/everyday 等。控制采样只选择与 prompt 语义兼容的模式，覆盖 none、mouse sparse、mouse dense、keyboard velocity、keyboard heading；续写 rollout 以 15% 概率切换 prompt，深度从 `1/2/4/8/16` 采样，并记录 prompt id、control id、rollout depth、prompt switch 和每帧 heading condition。V3 shard 与旧 V2 unconditional replay 向后兼容，shape/finite/SHA 和 categorical provenance 已有测试。

正式 text × control teacher corpus 已使用物理 GPU `1/2/3/6` 四路独立生成完毕，每 rank `131,072` 窗，总计 `524,288` 窗；teacher 是原版 10-step、text/constraint CFG 均为 `2.0`，Llama feature 直接查表，输出落盘 FP16。四路均约 `46.7–47.2 windows/s`，总吞吐约 `187 windows/s`，wall time 约 46.8 分钟。随后四个独立 CPU 进程并行完成全部 `2,048` 个 shard 的 SHA256、schema、shape、dtype 和 finite 校验，聚合结果为 `valid=true`、总字节数 `39,017,512,960`。完整统计为 history `79.37%`、constraint `76.09%`、prompt switch `11.89%`、unconditional `9.87%`；`8,192/8,192` 个 prompt 全部出现，generated prompt 最少 7 次、最多 128 次；none/mouse-sparse/mouse-dense/keyboard-velocity/keyboard-heading 分别为 `125,351/156,603/79,375/133,182/29,777` 窗，rollout depth `0..15` 全覆盖。机器结果在 `distill_data/text_control_v1/qa/train_aggregate.json`。四张数据卡同时工作，不再把多卡只留给训练。

单卡 teacher batch 吞吐已在 GPU4 实测而非凭经验设置：batch `8/32/64/128/256` 分别约为 `46.8/70.4/76.8/81.5/82.0 windows/s`。batch 128 相比旧 batch 8 提升约 74%，256 已基本饱和；因为主 corpus 在探针期间已推进到 41%，中途全量重启与继续完成的 ETA 已相同，所以没有丢弃已有约 213k 窗。独立 validation 随后固定用 batch 128，在四卡约 51 秒内生成 `16,384` 窗；64 个 shard、`1,219,297,280` bytes 的 SHA/shape/dtype/finite 全通过，结果在 `distill_data/text_control_v1/qa/val_aggregate.json`。Qwen 全表单卡约 88 秒完成；Llama 全表已经按四卡各 2048 prompt 并行编码后校验合并。

新增投影的首次 20-step 诊断发现 FLUX.2 Qwen raw feature RMS 约 `12.67`、平均行 L2 约 `1110.4`。在全 `7680D→512` 零初始化投影直接吃 raw feature 时，Adam 第一次更新导致下一步 loss 从 `3.61` 短暂升到 `16.19`。修正没有降成异常小 LR，也没有压缩特征，而是在 full projection 前加入无参数、逐 prompt、FP32 reduction 的 RMSNorm（`eps=1e-6`，输出回到 BF16/FP16）；all-zero unconditional 保持严格为零。相同 seed 对拍后第二步为 `3.61→4.53`，20 步全部 finite，condition gradient RMS 从坏探针峰值 `6.38` 降至通常约 `0.02–0.05`。scale-invariance、zero warm-start、text/heading effect 和参数增量测试均通过。

正式优化器也已拆成 inherited base 与新增 condition 两组：分别记录 pre-clip L2/RMS、独立裁剪、独立 LR，避免一个 3.93M 参数矩阵的总 L2 数把旧 flow 的梯度一起缩放。20-step joint smoke 在 base `1e-5`、condition `5e-5` 下全程 finite，step 20 的 base/condition grad RMS 为 `0.00681/0.01093`，total loss 为 `3.41`。执行顺序锁定为：先冻结旧 flow、四卡训练新增 full projection，使用正常 `5e-5→1e-5`；再从该 EMA 原样继承并四卡联合续训，旧 flow 使用 `1e-5→5e-6`，而不是从噪声重启；最后才进入数百到数千 generator update 的低 LR DMD2/ADV。

正式训练前已经用完整 loss/decoder/FK 路径做四卡真实 update 吞吐扫描，而不是只测前向：每卡 batch `128/256/512`、对应 global batch `512/1024/2048` 的稳态总吞吐约为 `3,444/4,786/5,643 samples/s`；后两档相对前一档分别提升约 `39%/18%`，三档均无 OOM且 loss/gradient 全 finite。最终采用每卡 512。正式 condition-only run 为 `text_control_condition_warmup_fullproj_4gpu_b512_step5e5_to1e5_10k_ema995_20260715/`：物理 GPU `1/2/3/6`、global batch `2,048`、10,000 update、约 `20.48M` 样本实例（完整 corpus 约 39.1 轮）、step 5,000 硬切 `5e-5→1e-5`、EMA `0.9995`，无 cosine/warmup/weight decay。实测稳态四卡 SM 为 `84%–96%`、约 `258–274 W/卡`；step 1→100 的 loss 为 `4.016→3.594`，condition grad RMS 为 `0.01260→0.00653`，全部 finite。该 stage 仍逐 tensor继承此前最终 path-only DMD2 EMA，只有新增 full text/heading projection 从 function-preserving zero 开始，没有重启既有 flow。本轮没有安装任何包、没有运行任何 `conda` 命令，所有命令只调用既有 py311 环境的绝对路径。

### 23.18 条件 warmup 10k 完成与固定集验证（2026-07-15）

condition-only warmup 已正常完成 `10,000` 个真实 optimizer update，以 `training_complete` 退出；累计处理 `20,480,000` 个窗口实例，训练净耗时 `4,012.27 s`。最终 raw/EMA flow 各含 `216` 个 tensor、`37,849,236` 个参数，文件均为 `151,417,904 B`，全部元素 finite。完整恢复状态也已审计：condition optimizer 的 3 组 state 的 Adam step 最小/最大值均为 `10,000`，scheduler `last_epoch=10,000`、当前 LR `1e-5`，EMA decay/update 为 `0.9995/10,000`，optimizer moment、model、raw、EMA 均无非有限值。这里的 `37.85M` 是 flow 本身；encoder + flow + decoder 的部署总参数仍为 `51.952M`。

在固定 `16,384` 窗 validation corpus 上，10k EMA 的 PyTorch FP16 NFE1 指标为 endpoint total `1.115672`、FSQ bin accuracy `0.054592`、FK MPJPE `0.310877 m`、foot slide `0.049678`、joint jerk `0.070323`、root temporal total `0.906272`、path error `0.275383 m`；teacher path error 为 `0.034256 m`。相同 5k EMA 分别为 `1.151641 / 0.054560 / 0.315296 m / 0.052086 / 0.074247 / 0.941612 / 0.286894 m`，因此 5k→10k 的 endpoint、FK、foot slide、jerk、root temporal 和 path error 分别改善约 `3.12% / 1.40% / 4.62% / 5.29% / 3.75% / 4.01%`；训练末段仍有实质收益，不能把 5k 当成已经收敛。

同一 10k EMA 的 FP32 对拍为 endpoint `1.115680`、FK `0.310887 m`、foot slide `0.049644`、joint jerk `0.070272`、root temporal `0.905827`、path error `0.275349 m`。FP16 相对 FP32 的绝对差依次约为 `7.4e-6 / 1.0e-5 m / 3.5e-5 / 5.1e-5 / 4.4e-4 / 3.5e-5 m`，当前 PyTorch FP16 算术稳定；这不替代之后真实 WebGPU FP16 + FSQ 串联 gate。

文本/控制反事实评估同样覆盖全部 `16,384` 窗。10k 全局正确文本相对 zero/shuffled text 的 endpoint 收益为 `0.216881/0.166485`，正确预测与 zero text 的平均 L1 差为有效非零；去掉 heading 的 endpoint 损失为 `0.095200`。keyboard-heading 子集从 5k 的 endpoint `1.077427` 改善到 `0.986341`（`-8.45%`），且 zero heading 的惩罚从 `1.536858` 增至 `1.614663`，证明朝向输入被强使用。mouse-dense、mouse-sparse、keyboard-velocity、none 分别由 `1.671276/1.235132/0.949550/0.950515` 改善到 `1.613148/1.203679/0.924741/0.924807`；prompt unconditional/official/generated 分别由 `1.588315/1.120376/1.099810` 改善到 `1.540872/1.078645/1.066047`。prompt-switch true 由 `1.233861` 改善到 `1.202336`，rollout depth `1/2-3/4-7/8-15` 也全部改善。逐项检查的全部 35 个分组 endpoint 均改善，无一回退；但 mouse-dense、unconditional、prompt-switch 仍是明显困难分布，必须在 joint fitting 与最终闭环验证中单列。

机器结果为 `eval/step010000_ema_full_fp16.json`、`eval/step010000_ema_full_fp32.json` 和 `eval/step010000_ema_text_control_fp16.json`，均位于本次 condition warmup 目录。结论是条件投影 warmup 已通过，可以从 10k EMA 原样进入四卡 joint fitting；不能直接把冻结 path-only backbone 的 condition-only 权重当成最终 student，也不能在这里随机重启或直接跳到 DMD2。joint stage 先做真实吞吐/显存探针，再以 base/condition `1e-5` 起步、单次硬切到 `5e-6`、EMA `0.9995`、无 cosine/warmup/weight decay；完成短 pilot 固定集与闭环 gate 后才进入低 LR 的短 DMD2/ADV。

### 23.19 joint fitting、固定集增益与基于平台期的 LR 更正（2026-07-15）

四卡 joint 的完整反向吞吐不能沿用 condition-only 结论，因此从 condition-10k EMA 做了每卡 batch `128/256/512` 的同配置 20-update 探针。去掉起始编译/加载后，step 5→20 的实测总吞吐分别约为 `2,831 / 4,098 / 1,461 samples/s`；b512 在全量 backbone 反传时进入明显低效区，最终锁定每卡 `256`、global batch `1,024`。三个探针均为人工 10-step LR 边界并已有 `ENGINEERING_SMOKE_ONLY.md`，不参与选模。正式 joint 从 condition-10k EMA 逐 tensor 继承全部 flow 权重，base/condition 同时训练，BF16 flow、FP16 frozen codec、EMA `0.9995`，无 cosine/warmup/weight decay。

正式 joint 的 step-2500 EMA 在固定 `16,384` 窗 FP16 validation 上已经将 endpoint total、FK、foot slide、joint jerk、root temporal 和 path error 从 condition-10k 的 `1.115672 / 0.310877 m / 0.049678 / 0.070323 / 0.906272 / 0.275383 m` 改善到 `0.826640 / 0.209874 m / 0.032933 / 0.041000 / 0.554977 / 0.109689 m`，FSQ bin accuracy 从 `0.054592` 升至 `0.066244`。全部 35 个文本/控制/rollout 分组 endpoint 均改善，无一回退；mouse-dense、unconditional、prompt-switch 和 keyboard-heading 分别达到 `0.973132 / 0.932453 / 0.910321 / 0.762068`。正确文本相对 zero 的收益仍为 `0.186159`，zero heading 惩罚为 `0.104458`，证明 joint 的收益不是通过忽略文本或朝向得到。

step-5000 EMA 相对 step-2500 又继续大幅改善：endpoint `0.826640→0.744729`（`-9.91%`）、FK `0.209874→0.175332 m`（`-16.46%`）、foot slide `0.032933→0.026132`（`-20.65%`）、joint jerk `0.041000→0.028804`（`-29.75%`）、root temporal `0.554977→0.399477`（`-28.02%`）、path error `0.109689→0.065781 m`（`-40.03%`），FSQ bin accuracy 升至 `0.070824`。分组再次全部改善，无一回退；全局正确文本收益回升到 `0.199563`，zero heading 惩罚增至 `0.110532`。因此 step5000 明确远未进入平台期。

最初命令曾预设 step5000 把 `1e-5` 硬切到 `5e-6`；固定验证完成后确认该衰减过早，任务在日志 step6400 被主动停止。旧目录已写 `BRANCH_AFTER_STEP5000_INVALID_PREMATURE_LR_DECAY.md`：只有完整 step5000 checkpoint 可用，5000 之后的 1400 个低 LR 更新全部丢弃，不能用于选模、DMD2、导出或结果报告。没有在该低 LR 轨迹上重新升 LR，而是从原始 step5000 分叉点恢复完整 model、216 组 Adam、EMA、sampler 和四 rank RNG 状态，再重走后续轨迹。

`train_flow.py` 为此新增显式 `--override-learning-rates-on-resume`：先由 Accelerate 恢复所有状态，再按新命令重设 optimizer 两组 LR 与 scheduler 的 base/current LR、epoch 和 step count。四 rank 的 5000→5001 工程 smoke 已实际验证：恢复事件明确报告 base/condition `1e-5/1e-5`，更新后 216 组 Adam step 最小/最大值、scheduler epoch、EMA update 均精确为 `5001`，optimizer/scheduler LR 都为 `1e-5`；该 tiny-batch smoke 已排除选模。正式恢复目录为 `text_control_joint_fullproj_resume5000_4gpu_b256_const1e5_until_plateau_step20000_ema995_20260715/`，从 step5000 继续固定 `1e-5`，每 2500 step 保存完整状态并做同一固定集验证。

joint 的 LR 规则更正为基于验证平台而不是预设步数：只有连续两个 2500-step 区间中，endpoint/FK/path/jerk 等多数核心指标的相对改善都低于约 `2%`，且 mouse-dense、unconditional、prompt-switch 等困难分组也无实质改善，才从 `1e-5` 减到 `5e-6`；单个训练 batch loss、一次噪声波动或任意固定 step 都不能触发衰减。减半后不再升回；若平台条件未满足，则保持 `1e-5` 继续已有状态。

文本条件版 `train_flow_dmd2.py` 也已用 joint-5k EMA 做单 GPU、batch8、2 次 guidance warmup + 1 次 generator update 的接口 smoke。Qwen text、heading、fake score、critic、generator、FSQ 和 frozen codec 全链路均 finite，generator/score/critic 实际 LR 为 `1e-6/5e-6/1e-6`；该目录已有 `ENGINEERING_SMOKE_ONLY.md`，只证明接口和状态保存可运行，不代表质量，也不允许替代最终 joint EMA 或正式 200-step guidance warmup。

### 23.20 joint 15k 平台审计、首个文本 DMD/LADD pilot 与 WebGPU 前端闭环（2026-07-15）

joint 恢复分支持续使用物理 GPU `1/2/3/6`、每卡 batch 256、global batch 1024、base/condition LR `1e-5/1e-5`。7.5k EMA 的 endpoint/FK/foot/joint-jerk/root-temporal/path 为 `0.716004 / 0.163850 m / 0.023183 / 0.023654 / 0.329904 / 0.053949 m`；10k 为 `0.704140 / 0.158547 m / 0.022263 / 0.022225 / 0.309392 / 0.050070 m`；12.5k 为 `0.699739 / 0.155557 m / 0.021864 / 0.021557 / 0.299254 / 0.048120 m`；15k 为 `0.699396 / 0.153739 m / 0.021644 / 0.021170 / 0.293194 / 0.046833 m`。10k→12.5k 的相对改善约为 `0.63% / 1.89% / 1.80% / 3.00% / 3.28% / 3.89%`；12.5k→15k 为 `0.05% / 1.17% / 1.00% / 1.80% / 2.03% / 2.67%`。后一段多数指标接近平台，但前一段 jerk/root/path 仍明显超过 2%，尚不满足“连续两段多数核心项低于约 2%”的减半条件，因此继续 `1e-5` 到 17.5k 再验证，不根据 endpoint 单项提前减 LR。

条件反事实也不支持把 15k 视为完全停止学习：12.5k→15k 的正确文本相对 zero-text endpoint 收益由 `0.226439` 增至 `0.233428`，zero-heading 惩罚由 `0.116428` 增至 `0.117482`。7.5k/10k 的正确文本收益为 `0.209718/0.218548`，heading 收益为 `0.113387/0.115128`。7.5k 与 10k 的 35 个固定分组全部改善；12.5k/15k 的完整 full 与 stratified 机器结果分别在正式目录的 `eval/step012500_ema_*_fp16.json` 和 `eval/step015000_ema_*_fp16.json`。主训练验证始终在其余空闲 GPU 并行运行，没有暂停四卡主任务。

首个正式文本条件 DMD/LADD pilot 从 joint-10k EMA 同时初始化冻结 teacher 与 generator，不从噪声重启。物理 GPU `0/4/5/7`、每卡 batch 32、global batch 128，先做 200 次 guidance warmup，再做 200 次 generator update；每次 generator 对应一次 score/critic 更新，generator/score/critic LR 为 `1e-6/5e-6/1e-6`，EMA `0.995`，adv/paired/FSQ 系数为 `0.001/0.1/0.1`，无 cosine/warmup/weight decay。总耗时 `91.72 s`。最终状态审计为 generator/fake-score 各 `216` 个 tensor、`37,849,236` 参数，critic `18` 个 tensor、`4,726,785` 参数，全部 finite；generator optimizer 的 216 组 state 均为 step200，score/critic 分别为 step400，EMA update200，计数器与计划一致。

该 g200 EMA 的固定 FP16 静态验证相对 joint-10k 不是全面提升：endpoint `0.704140→0.720242`（`+2.29%`）、FK `0.158547→0.159669 m`（`+0.71%`）、foot `0.022263→0.022610`（`+1.56%`）、jerk `0.022225→0.022361`（`+0.61%`）、root-temporal `0.309392→0.309642`（`+0.08%`），path `0.050070→0.050021 m`（`-0.10%`）。文本与 heading 反事实收益仍为 `0.230471/0.118065`，没有静态塌缩，但这只能叫首个 `pilot_g200`，不能称最优 DMD/LADD。分布蒸馏明确按实验科学处理：后续固定 teacher/data/seeds/frontend 协议，系统扫 generator LR `1e-8..1e-6`、score/critic LR、warmup、generator iteration `100/200/500/1000`、G:score:dis 更新频率、critic feature/condition 设计、adv/paired 权重与是否共享框架；比较静态固定集、长 rollout、真实 WebGPU 延迟和前端盲测，而不是由单次配置或论文默认值决定。

为尽快验证完整闭环，g200 EMA 已导出为 `text_qwen_dmd2_joint10k_g200_ema_pilot_20260715`，manifest 明确标记 `release_status=experimental_pilot`。结构仍为 encoder width512/4-block、flow width512/8-head/8+8-block、decoder width512/8-block；参数量 `4,937,344 / 37,849,236 / 9,165,420`，总计 `51,952,000`。FP16 ONNX 大小为 `9,882,793 / 75,809,680 / 18,370,139 B`，learned graphs 共 `104,062,612 B`；加两个 FP32 root/FSQ utility 后完整 ONNX 为 `104,107,876 B`，约 `99.285 MiB`，低于本 pilot 的近似预算 `110,000,000 B`。导出时 CPU ORT 对同精度 PyTorch 的 encoder/flow/decoder max abs 为 `0.001290/0.003615/0.007401`，全部 finite。

前端没有在浏览器运行 Qwen，也没有把 Qwen 计入 motion student：从完整 8192 条离线 Qwen 表中确定性抽取官方 preset 与全部 17 个 family 的 33 条代表 prompt，保留每条完整 `7680D` FP16 feature，无 PCA、低秩、INT8 或其他压缩；额外下载 `506,880 B`。页面可直接切换文本，切换后从当前时刻重新规划；flow 的真实输入新增 `text_feature [1,7680]` 与 `heading_condition [1,64,3]`。鼠标仍使用原版稀疏 `path_condition [1,64,3]`、目标位于当前帧 `+60`；没有键盘朝向命令时 heading validity 全零，不伪造约束。开始/暂停、重新开始、清空 waypoint、无限 buffer 续写、时间轴和“推理期间冻结时间、不跳帧”语义均保留。远程调试协议升级到 v3，并增加 prompt 切换和 reload 命令，后续浏览器一旦加载新版即可由服务器自动下发 validate/benchmark/waypoint/prompt/reload。

带文本条件的服务器端整链路验证已经通过，不是只验证单模块：prompt 为 `A person is walking.`、heading validity 为 0，initial/continuation 的 ONNX FP16 对同精度 PyTorch flow endpoint max abs 为 `0.0078125/0.0097656`，相对 PyTorch FP32 最坏 max abs 为 `0.0060055`；最终 continuation explicit motion 对 FP32 的 mean/p99 abs 为 `0.004615/0.028637`、cosine `0.9999768`，最低 FSQ bin match `97.65625%`。首段和续写段均为 40 帧，第二窗后 buffer 精确为 77 帧，完整验证 `passed=true`。机器结果为 `webgpu_toy/infinite_demo/server_validation.json` 与 `browser_validation_case.json`。

8766 服务已用现有 py311 环境重新启动，页面仍是 `http://127.0.0.1:8766/infinite_demo.html`。记录时服务器只看到已过期的旧 release 浏览器心跳，真实 Edge 尚未刷新到本次 g200 文本 pilot，所以真实 WebGPU golden 对拍、实际延迟和视觉效果仍待用户刷新后验收；服务器 CPU 对拍不能替代这一关。全过程没有安装任何包、没有执行 `conda` 命令，也没有修改 `<REFERENCE_3DLM_PROJECT>`。

### 23.21 joint 20k 平台确认、单向减半续训与真实 Edge/WebGPU 审计（2026-07-15）

17.5k EMA 的 endpoint/FSQ-bin-accuracy/FK/foot/joint-jerk/root-temporal/path 为 `0.700846 / 0.075732 / 0.152598 m / 0.021506 / 0.020847 / 0.287948 / 0.045868 m`；20k 为 `0.703428 / 0.075929 / 0.151741 m / 0.021437 / 0.020644 / 0.284805 / 0.045050 m`。15k→17.5k→20k 时 endpoint 连续回退约 `0.21%/0.37%`；17.5k→20k 的 FK、foot、jerk、root、path 只改善约 `0.56%/0.32%/0.98%/1.09%/1.78%`。正确文本相对 zero-text 的收益仍由 `0.233428→0.239024→0.244055` 增长，heading 收益为 `0.117482→0.118289→0.118956`，但这并未转化为绝对拟合误差继续下降。

困难分组进一步确认已经进入平台而非单个全局指标噪声：15k→17.5k→20k 的 mouse-dense endpoint 为 `0.786933→0.789053→0.793038`，unconditional 为 `0.759936→0.766240→0.773335`，prompt-switch 为 `0.789305→0.793382→0.799295`，mouse-sparse 为 `0.791421→0.794601→0.798756`；均连续小幅回退。keyboard-heading 为 `0.614491→0.614308→0.614721`，基本持平。因此满足此前“连续两个 2500-step 区间多数核心项低于约 2%，且困难组无实质改善”的单向减半条件；不能因为 text gain 单项仍增长而继续用 `1e-5`。

20k 已以 `training_complete` 正常结束并保存完整 model/Adam/EMA/scheduler/RNG/sampler 状态。随后从 `state/step-0020000` 原样恢复到 `text_control_joint_fullproj_resume20000_4gpu_b256_const5e6_until_plateau_step40000_ema995_20260715/`，没有重新初始化；物理 GPU 仍为 `1/2/3/6`、每卡 batch 256、global batch 1024、BF16 flow、FP16 frozen codec、EMA `0.9995`。base/condition LR 同时固定为 `5e-6`，无 cosine、warmup 或 weight decay；恢复事件明确记录 step 20000 的 LR 已覆盖为 `5e-6/5e-6`，step 20100 以后实际日志也是 `4.9999999e-6`。暂设上限 40k，但仍每 2500 step 做固定 full + stratified validation，是否提前结束由平台和困难组决定，而不是机械跑满。

真实 Edge 已加载 `text_qwen_dmd2_joint10k_g200_ema_pilot_20260715`，adapter 为 NVIDIA Ampere 且 `shader-f16=true`。严格 WebGPU golden 对拍没有通过，不能用服务器 CPU 通过冒充：initial/continuation flow endpoint max abs 为 `0.082764/0.036743`，explicit motion mean abs 为 `0.036699/0.038628`、p99 为 `0.243193/0.235352`、cosine 为 `0.998362/0.998601`，FSQ bin match 为 `88.594%/89.766%`；所有 tensor finite，但低于预设 `90%` bin match 和 motion 误差/余弦门槛，门槛没有被放宽。

针对当前带完整 `7680D` Qwen feature 与 heading 输入的图，逐层 WebGPU/CPU-ORT FP16 探针已经重建并在同一 Edge 上执行。encoder 最终输出 mean/max abs 为 `0.005379/0.022949`、cosine `0.999976`；flow 为 `0.006464/0.030273`、cosine `0.999955`；decoder 为 `0.009719/0.115234`、cosine `0.999930`。attention QK logits 的局部 max 可达 `0.625`，但 softmax 后误差很小；没有发现漏输入、shape 错误、非有限值或单个算子直接失效。当前证据更符合深层 FP16 backend 差异逐层累积，随后由不连续 FSQ bin crossing 与 8 层 decoder 放大。后续需要把 WebGPU-aware precision simulation/bin-margin 或蒸馏鲁棒性纳入实验变量，不能只看 PyTorch FP16。

真实 Edge 的 3 次 warmup + 20 次 continuation-window 计时已经完成。每窗总耗时 p50/p95 为 `205.61/328.18 ms`；encoder session p50 `21.23 ms`，NFE1 flow `121.06 ms`，decoder `55.26 ms`，FK 后处理 `0.238 ms`。因此本 pilot 的 WebGPU 计算速度足够支持无限生成，当前优先问题是 DMD/LADD 视觉效果与 FP16 稳健性，不是算力延迟。DMD/LADD 仍严格视为实验科学：首个 g200 只提供完整闭环基线，后续固定 teacher/data/seeds/前端协议后系统扫描 generator LR `1e-8..1e-6`、score/critic LR、iterations、G:score:dis 更新频率、critic 条件与特征层、adv/paired/FSQ 权重，并以静态集、长 rollout、真实 WebGPU 数值和盲测共同选择。

当前页面仍为 `http://127.0.0.1:8766/infinite_demo.html`，真实 Edge 已刷新到 protocol v3，prompt、reload、waypoint、开始/暂停及验证任务现在都可由服务器远程控制。用户已经在页面中实际连续播放并切换多个 prompt；最近 20 个自然续写窗的 total/encoder/flow/decoder p50 为 `136.13/14.80/81.63/38.38 ms`，比固定 benchmark 更快。现有 waypoint telemetry 中既有约 `0.22–0.28 m` 的命中，也有约 `1.96/3.95 m` 的明显偏差，进一步说明首个 pilot 只适合暴露问题，不能宣布效果完成。刷新过程中第二次 golden job 被旧文档领取后中断，服务器中的 `running` 只是 orphaned 状态、没有产生新验证结果；严格数值仍以上文第一次完整完成的 WebGPU 对拍为准。全过程没有安装任何包、没有执行 `conda` 命令，也没有修改 `<REFERENCE_3DLM_PROJECT>`。

### 23.22 joint 30k、首轮 DMD/LADD 实验矩阵与新 WebGPU 候选（2026-07-15）

主 joint 在物理 GPU `1/2/3/6` 上保持四卡不中断运行，仍是每卡 batch 256、global batch 1024、base/condition 固定 LR `5e-6`、BF16 flow、FP16 frozen codec、EMA `0.9995`，无 cosine/warmup/weight decay。22.5k/25k/27.5k/30k EMA 的 endpoint 分别为 `0.707362/0.710802/0.713658/0.716355`，FK 为 `0.151242/0.150911/0.150664/0.150468 m`，joint jerk 为 `0.020448/0.020321/0.020282/0.019998`，root temporal 为 `0.281401/0.279115/0.278278/0.274333`，path error 为 `0.044450/0.044037/0.043722/0.043403 m`。22.5k→30k 时 FK、jerk、root temporal、path 分别改善约 `0.51%/2.20%/2.51%/2.36%`，但 endpoint 回退约 `1.27%`。

困难条件同样显示这个可复现的 Pareto trade-off：22.5k→30k 的 mouse-dense、unconditional、prompt-switch endpoint 从 `0.798817/0.780905/0.806351` 变为 `0.811197/0.795734/0.822005`，分别回退约 `1.55%/1.90%/1.94%`；text/heading 反事实收益则由 `0.248471/0.119464` 增至 `0.255331/0.120395`。因此 30k 不是无条件优于 22.5k，旧 checkpoint 全部保留；主训练按用户要求继续挂着以提供更多可选 Pareto 点，但首轮分布蒸馏固定从 22.5k EMA 分叉，避免实验过程中移动 teacher/init 基线。

首轮 DMD/LADD 把 teacher、generator 初始化、data、seed、batch、warmup、score/critic LR、更新频率、loss 权重和 EMA 全部固定，只扫 generator LR。四臂均使用 joint-22.5k EMA 同时初始化 frozen teacher 与 generator，物理 GPU `0/4/5/7`、global batch 128、200 次 guidance warmup + 200 次 generator update、G:score:critic=`1:1:1`、score/critic LR=`5e-6/1e-6`、EMA `0.995`、generator grad clip `0.01`、adv/paired/FSQ=`0.001/0.1/0.1`；generator LR 为 `1e-8/1e-7/3e-7/1e-6`。每臂耗时约 `88–91 s`，全部完成 200 个 generator 与 400 个 score/critic update，模型/优化器/指标均 finite。

静态固定集显示随 generator LR 墦大而单调偏离 teacher。stage-one 基线的 endpoint/FK/foot/jerk/root/path 为 `0.707362/0.151242/0.021390/0.020448/0.281401/0.044450`；`1e-8` 为 `0.708040/0.151294/0.021393/0.020409/0.280882/0.044533`，基本近似不动；`1e-7` 为 `0.712884/0.151712/0.021559/0.020524/0.282003/0.044850`；`3e-7` 为 `0.719746/0.152192/0.021769/0.020663/0.283266/0.044678`；`1e-6` 为 `0.731271/0.152907/0.022083/0.020852/0.284927/0.044360`。`1e-7` 的静态 endpoint 相对基线回退 `0.78%`，`3e-7/1e-6` 则回退 `1.75%/3.38%`。这些结果只约束 teacher 配对保真度，不能替代分布/动画判断。

为避免单条轨迹的混沌分岔误导选择，又对 baseline、`1e-8`、`1e-7`、`3e-7` 做 seed `20260714..20260717` 四条严格配对、每条 50-window、FP16 NFE1 的原版无限 buffer/history/稀疏 waypoint rollout；每条均同时跑精确 10-step teacher，有限差分和 FFT 按每个生成段独立计算，不跨 replan seam。相对各 seed 的 stage-one 基线，`1e-7` 的 waypoint error、rollout FK、foot slide、joint seam、body jerk P95、body 高频分别平均变化 `-13.52%/-0.08%/-3.39%/-0.68%/-2.61%/-8.16%`；root horizontal/root heading 高频分别为 `-5.57%/-21.90%`。其中 body jerk 和两项 root 高频在 `4/4` seed 改善，waypoint/foot/seam/body-HF 在 `3/4` seed 改善，全部输出 finite。`1e-8` 的 waypoint/foot 改善 `14.03%/2.87%`，但 seam 回退 `0.51%`，body jerk/HF 只改善 `1.49%/3.35%`；`3e-7` 的 body-HF、foot、seam改善更大，为 `12.20%/5.05%/1.98%`，但 waypoint 只改善 `5.90%`，静态 endpoint 已明显回退。首轮因此选 `1e-7` 作为最均衡的视觉候选，而不是把单 seed 或单个静态 loss 当结论。

机器汇总为 `distill_runs/dmd_ladd_lr_round1_joint22500_warm200_g200_20260715/sweep_summary.json`；它保留逐 seed 原始值、配对百分比、population std 和胜出 seed 数，不使用任意加权总分。新增的 `summarize_dmd_ladd_sweep.py` 可同时汇总静态 full/text 与多 seed rollout；`evaluate_rollout.py` 也已支持按 checkpoint 的 `text_feature_dim=7680` 和 `heading_condition_features=3` 实例化条件分支，本次纯鼠标滚动显式使用零文本/零朝向条件。训练、静态评估、四 seed rollout 和 jitter 的可复现实行脚本均保存在 `scripts/`。

`1e-7` g200 EMA 已导出为新实验 release `text_qwen_dmd_ladd_joint22500_g1e7_g200_ema_pilot_20260715`。结构与参数量仍为 encoder `4,937,344`、flow `37,849,236`、decoder `9,165,420`，总计 `51,952,000`；FP16 learned graphs 为 `104,062,612 B`，加 FP32 root/FSQ utilities 后为 `104,107,876 B = 99.285 MiB`，低于本实验的近似 `110,000,000 B` 上限。导出时 CPU ORT 对同精度 PyTorch 的三模块 max abs 为 `0.001290/0.003308/0.007401`，全部 finite；33 条浏览器 prompt 仍保留完整 `7680D` FP16 Qwen feature，无 PCA、低秩或量化，文本编码器不计入 motion student 也不在浏览器运行。

完整 initial+continuation CPU 语义链的 shape、路径索引和全部 tensor 均 finite，但严格 FP16 numeric gate 未通过，不能伪称通过：continuation endpoint 对同精度 PyTorch max abs 为 `0.235352`，对 FP32 为 `0.179216`；最终 explicit 对 FP32 mean/p99/cosine 为 `0.020692/0.175247/0.999404`，FSQ bin match 为 `66.875%`。这仍是深层 FP16 累积误差经不连续 FSQ 与 decoder 放大的已知风险。新的 browser golden 与 15/77/43 个 encoder/flow/decoder 逐层 probe 已按本 release 重建；真实 Edge/WebGPU 结果仍作为独立 gate，不会用 CPU 模块级对拍替代。

前端 manifest 已切到该新候选，页面地址不变：`http://127.0.0.1:8766/infinite_demo.html`。服务器已排队 protocol-v3 `reload`；记录时浏览器标签页心跳暂停，故尚未领取命令，用户重新聚焦页面后会自动刷新。该 release 明确是首轮实验候选，不是最终 DMD/LADD。后续保持 stage-one 主训练并行，继续从已有 g200 完整状态测试 generator iterations、G:score:critic 更新频率、score/critic LR、critic feature/condition 框架及 adv/paired 权重，所有分支仍需静态、多 seed rollout、真实 WebGPU 数值和用户视觉测试共同决策。全过程没有安装任何包、没有执行 `conda` 命令，也没有修改 `<REFERENCE_3DLM_PROJECT>`。

随后真实 Edge 已自动领取 `reload` 并确认当前 heartbeat 的 model release 正是 `text_qwen_dmd_ladd_joint22500_g1e7_g200_ema_pilot_20260715`，不再是旧 joint-10k pilot。新版逐层 precision probe 完成且任务本身通过，encoder/flow/decoder 最终输出的 WebGPU 对 CPU-ORT FP16 mean/max abs 分别为 `0.005298/0.020508`、`0.015154/0.144531`、`0.010236/0.104584`，cosine 为 `0.999976/0.999667/0.999924`，全部 finite。相比旧 release，encoder/decoder 同量级，但 flow 的 backend 累积误差更大，说明 DMD 选模还必须显式考虑 WebGPU 数值鲁棒性。

同一新版的真实 Edge golden validation 仍严格判为失败：initial/continuation endpoint max abs 为 `0.050781/0.075439`，FSQ bin match 为 `88.125%/84.453%`；explicit mean abs 为 `0.034631/0.038705`、p99 为 `0.224629/0.237314`、cosine 为 `0.998432/0.998631`。全部 tensor finite，且最终 motion 误差量级与旧 pilot 接近，但 continuation FSQ 稳健性更差，门槛没有被放宽。该结果与用户视觉反馈一起决定是否保留 g200；不能仅凭多 seed PyTorch rollout 改善就宣布 WebGPU 版本优胜。

iterations 轴已经从 `g1e7` 的完整 step-200 Accelerate 状态原样恢复到 `dmd_ladd_iters_g1e7_resume200_to1000_joint22500_20260715/`：generator、fake-score、critic、三组 Adam/scheduler、EMA、sampler/RNG 和 update counters 全部继承，不重复 guidance warmup、不重启任何已有权重。generator LR 仍为 `1e-7`，其余首轮超参全部固定；每 100 generator step 保存 raw/EMA 与完整恢复状态，用于隔离比较 g200/g300/.../g1000 的最佳停止点。恢复后的首个日志为 generator step210、guidance410，证明确实从 200/400 计数继续；四卡更新已实际运行，主 joint 也继续独立训练。

### 23.23 DMD/LADD iterations 轴、g300 WebGPU 候选与后续实验原则（2026-07-15）

`generator_lr=1e-7` 的同一完整状态已从 g200 连续训练到 g1000，新增部分耗时 `269.67 s`；g300/g500/g700/g1000 均保存 raw、EMA 和完整恢复状态。所有配置保持 teacher、数据、seed、batch、score/critic LR、G:score:critic 更新比、loss 权重、EMA 与 generator grad clip 不变，因此该分支只回答“同一轨迹多跑多少 generator update”。训练全程 finite；critic loss 长期约 `1.385–1.386`、接近二分类随机基线，generator pre-clip gradient L2 约 `5–6` 且始终被裁到 `0.01`，这两个现象都必须作为后续更新频率、critic 框架和裁剪强度实验的独立变量，不能靠继续增加 generator step 掩盖。

静态固定集随迭代数增加而单调偏离 stage-one。g200/g300/g500/g700/g1000 的 endpoint 相对 joint-22.5k 分别回退 `0.781%/1.400%/2.588%/3.457%/4.359%`。四 seed、每 seed 50-window 的严格配对无限 rollout 则呈现非单调 Pareto：g200 的 waypoint/FK/foot/seam/body-jerk/body-HF 相对 stage-one 分别变化 `-13.519%/-0.082%/-3.388%/-0.683%/-2.609%/-8.159%`；g300 为 `-14.424%/-4.016%/-4.435%/-4.711%/-2.088%/-11.072%`；g500 为 `-10.070%/+1.679%/-2.922%/-0.737%/-1.625%/-12.012%`；g700 为 `-12.241%/+0.645%/-2.513%/+1.504%/+1.241%/-13.368%`；g1000 为 `-17.239%/-4.496%/+1.964%/+3.425%/+4.898%/-21.031%`。负值表示误差降低。g300 是当前最均衡的停止点：路径、FK、foot、seam、jerk 与高频项同时改善，静态 endpoint 代价为 `+1.40%`；g1000 虽继续追逐 waypoint/body-HF，却已经明显损害 foot、seam 与 jerk。机器汇总为 `distill_runs/dmd_ladd_iters_g1e7_resume200_to1000_joint22500_20260715/iteration_summary.json`，不使用人为加权总分。

g300 EMA 已先以独立 candidate manifest 导出并做服务器门禁，再切换活动前端；release 为 `text_qwen_dmd_ladd_joint22500_g1e7_g300_ema_pilot_20260715`。模型结构、总参数量和下载体积与 g200 完全相同：motion student `51,952,000` 参数，FP16 learned graphs `104,062,612 B`，含 FP32 utilities 共 `104,107,876 B = 99.285 MiB`。模块导出时 encoder/flow/decoder 对 PyTorch FP32 的 max abs 为 `0.001290/0.004135/0.007401`，全部 finite。

g300 的完整 CPU-ORT FP16 initial+continuation gate 只在一项极紧阈值上失败：same-precision continuation endpoint max abs 为 `0.032715`，略高于阈值 `0.025`；相对 FP32 的 endpoint max abs `0.017689`、explicit mean/p99/cosine `0.006761/0.034810/0.999965`、FSQ bin match `94.6875%` 均通过各自阈值。与活动前任 g200 的 `0.235352/0.179216/0.020692/0.175247/0.999404/66.875%` 相比，g300 在同一服务器协议下明显更稳定，因此满足切换候选的条件，但仍不能把服务器结果当作真实 WebGPU 通过。活动页面地址不变，仍为 `http://127.0.0.1:8766/infinite_demo.html`；真实 Edge 已完成 g300 reload，逐层 precision probe 与 golden validation 作为独立 gate 排队执行。

DMD+LADD 后续明确按实验科学而非单次 recipe 推进。当前证据只支持把 generator LR 的有效搜索重点放在 `1e-8..1e-6` 和 few-hundred-step 区域，不支持宣布唯一最优值。后续保持 stage-one/初始化、训练 batch、数据采样、验证 seeds 和前端协议固定，分别扫描 generator/score/critic LR、generator iterations、G:score:critic 更新频率、guidance warmup、critic feature/condition/framework、adv/paired/FSQ 权重、gradient clip 与 EMA；每项同时报告静态固定集、多 seed 长 rollout、真实 WebGPU 数值、延迟和用户前端视觉测试。尤其不能只依据 critic loss、训练 loss、某个 seed 或 PyTorch rollout 作最终选择。主 joint 继续在物理 GPU `1/2/3/6` 上运行，分布实验使用其余四卡并行，不因候选导出而中断。全过程没有安装任何包、没有执行 `conda` 命令，也没有修改 `<REFERENCE_3DLM_PROJECT>`。

真实 Edge 已随后完成 g300 的逐层 probe 与完整 golden。逐层 probe 任务通过且全部 finite：encoder/flow/decoder 最终输出对 CPU-ORT FP16 的 mean/max abs 为 `0.005309/0.016602`、`0.006466/0.032227`、`0.010281/0.119141`，cosine 为 `0.999977/0.999955/0.999923`。flow 层面比 g200 的 `0.015154/0.144531/0.999667` 明显稳定，说明 g300 并非整体更不适合 FP16 WebGPU。

但完整 golden 仍失败，而且暴露明显输入依赖：g300 initial/continuation endpoint max abs 为 `0.189453/0.087891`，FSQ bin match 为 `66.406%/85.234%`，最终 explicit mean/p99/cosine 为 `0.041676/0.278320/0.997917` 与 `0.036630/0.227559/0.998885`。与 g200 的 initial `0.050781/88.125%` 相比，g300 首窗显著更差；续写窗相对 g200 的 `0.075439/84.453%` 则接近、略有取舍。逐层 probe 使用一个 continuation case，而 golden 同时覆盖 initial 与另一 continuation state，因此两者并不矛盾，而是说明 FP16 backend 差异和 FSQ crossing 对输入状态高度敏感。g300 继续作为活动视觉候选供用户在原 URL 测试，但不能标记为 WebGPU 数值验收版；后续 WebGPU gate 必须扩成多输入、多 seed，而不能再由单个 probe 或单个 golden 决定。

G:score:critic 更新频率的第一轮受控实验也已完成。两条分支均从同一个 g200 完整 Accelerate 状态继续到 generator step300；`1:2:2` 最终 guidance update600、增量训练耗时 `51.95 s`，`1:4:4` 最终 guidance update800、耗时 `77.20 s`。两者末次 critic total 仍为 `1.386436/1.386414`，real/fake logits 都接近零；增加 guidance/critic 次数没有使当前 teacher-feature critic 脱离随机基线。与 `1:1:1` g300 EMA 相比，两臂 EMA 权重的 relative L2 仅为 `7.49e-6/7.70e-6`，静态 endpoint/FK/foot/jerk/root/path 也几乎相同：`1:1:1` endpoint `0.717264`，`1:2:2/1:4:4` 为 `0.717294/0.717292`。

四 seed 无限 rollout 进一步否定“多做几次判别器更新自然更好”。相对 stage-one，`1:1:1` 的 waypoint/root-drift-mean/FK/foot/seam/jerk/body-HF 为 `-14.424%/-6.692%/-4.016%/-4.435%/-4.711%/-2.088%/-11.072%`；`1:2:2` 为 `-6.182%/-0.808%/+1.237%/-4.618%/-0.735%/-2.318%/-5.916%`；`1:4:4` 为 `-11.820%/-1.367%/-6.021%/-3.140%/-2.938%/-2.516%/-7.993%`。直接对 `1:1:1` 配对时，`1:2:2` 的 waypoint/root-drift/FK/seam 在 `0/4` seed 胜出；`1:4:4` 的 waypoint/root-drift 也只在 `1/4` 和 `0/4` seed 胜出，FK 的平均改善主要由单个 seed 驱动。当前继续增加更新频率没有稳健收益，下一步应测试 critic feature/condition/framework、score 与 critic 独立频率以及 adv 权重，而不是继续扩大同一绑定 ratio。机器汇总为 `distill_runs/dmd_ladd_guidance_ratio_from_g200_joint22500_20260715/guidance_ratio_summary.json`，包含逐 seed 原始值、配对变化和胜出数。

主 joint 已在 step40000 以 `training_complete` 正常结束，累计本分支 `40,960,000` 个样本实例、净训练耗时 `5,410.09 s`，并保存完整 weight 与 Accelerate state；没有异常退出。32.5k/35k/37.5k/40k 固定 full + stratified validation 已随后在四张空闲卡并行补齐。endpoint 依次为 `0.718826/0.721356/0.723770/0.726102`，FK 为 `0.150285/0.150131/0.150024/0.149906 m`，foot 为 `0.021285/0.021253/0.021274/0.021282`，jerk 为 `0.019860/0.019761/0.019736/0.019659`，root temporal 为 `0.272309/0.270558/0.270079/0.268989`，path 为 `0.043124/0.042922/0.042666/0.042483 m`。30k 以后每个 2500-step 区间的 FK/root/path/jerk 改善只约 `0.07–0.74%`，已经进入平台区，foot 在 35k 后也不再稳定改善。

同一阶段的分布代价却持续累积：22.5k→40k 的 endpoint、mouse-dense、unconditional、prompt-switch 分别回退 `2.65%/3.31%/3.27%/3.96%`，而 FK/jerk/root/path 改善 `0.88%/3.86%/4.41%/4.43%`，text/heading 反事实收益增加 `4.73%/1.49%`。因此 40k 是“更平滑、root/path 更好但配对 endpoint 与困难分布更差”的 Pareto checkpoint，不是无条件最优；22.5k 仍是当前 DMD/LADD 初始化和配对保真的合理基线。基于完整验证，不再用同一数据、同一 loss 和同一 `5e-6` 机械续跑 supervised joint；若要继续改善困难分布，应改变采样/reweighting 或目标设计，而不是增加相同步数。所有 22.5k–40k checkpoint 均保留。

对 critic 长期停在 `1.3863` 的代码级审计发现了一个确定性框架问题，而不只是经验上的“可能更新不足”。当前 score 与 critic 共用 `exact_t1_probability=0.7`，real/fake teacher-feature critic 又对两者使用同一份 noise。rectified-flow 插值为 `x_t=(1-t)x0+tε`；精确 `t=1` 时 real/fake 输入都严格等于同一个 `ε`，冻结 teacher 得到的中间 feature 也逐元素相同。新增单元测试已经以不同 real/fake clean、相同 noise 验证 exact-t1 feature 的 `rtol=0/atol=0` 严格相等；完整 flow-matching 测试共 10 项通过。训练日志也新增 feature-gap 诊断，legacy sampler 实测 exact-t1 fraction 约 `0.63–0.73`，且该子集 feature-gap 始终严格为 `0.0`，与理论完全一致。此时同一输入被同时监督为 real/fake，最优 logit 为零、loss 为 `2·log(2)=1.386294`，正好解释此前观测。

`train_flow_dmd2.py` 已因此把 score/DMD 与 critic 的 timestep sampler 解耦，同时保持旧默认完全向后兼容。新增 `critic_time_exact_t1_probability`、`critic_time_high_noise_probability` 和 `critic_time_upper_bound`，并记录每批 critic time 的 mean/min/max、exact-t1 fraction、全体 feature gap 与 exact-t1 feature gap。score/DMD 仍保留面向 NFE1 的高噪声与 `t=1` 偏置；只有 critic 实验可以去掉无信息的精确 `t=1` 并限制最高噪声。下一轮从同一个 g200 完整状态分叉，先统一追加 200 次 score/critic-only recovery，再走 100 次 generator update，对比 legacy、uniform `[0,0.9]`、high-noise-biased `[0,0.9]` 与 uniform `[0,0.5]`；control arm 也做相同 recovery，从而只隔离 critic-time 框架差异。

旧 sampler 下的 adversarial-weight 消融已完成静态和四 seed 无限 rollout。所有臂都从同一 g200 状态走到 g300，只改 adv weight；静态 endpoint 均约 `0.71725`，无法区分。相对 stage-one，当前 `1e-3` 的 waypoint/root-drift-mean/FK/foot/seam/jerk/body-HF 为 `-14.42%/-6.69%/-4.02%/-4.44%/-4.71%/-2.09%/-11.07%`。adv=0 为 `-10.22%/-0.47%/+1.48%/-1.97%/+0.13%/-0.50%/-3.65%`，直接对当前 `1e-3` 时 waypoint/root/FK/foot/seam/jerk/body-HF 只在 `1/4/1/1/1/1/0` 个 seed 胜出，说明 adversarial 项虽被旧 timestep 大量削弱，但并非完全无效。`3e-3/1e-2` 的 waypoint、root、FK 和 seam 整体更差；`3e-4` 可换得略好的 jerk/body-HF/heading-HF，却牺牲 waypoint/root/FK，且 root final drift 方差很大。因此当前不把 adv weight 盲目增大或删零，保留 `1e-3` 作为下一轮框架对照。机器结果为 `distill_runs/dmd_ladd_adv_weight_from_g200_joint22500_20260715/adversarial_weight_summary.json`。

critic-time 四臂也已完成。统一 recovery 到 guidance600 时，legacy 的 critic loss/feature-gap 为 `1.386112/0.066242`；uniform `[0,0.9]` 为 `1.359645/0.398618`；high-noise-biased `[0,0.9]` 为 `1.372476/0.303721`；uniform `[0,0.5]` 为 `1.333480/0.511505`。这证明去掉 exact-t1 后判别器得到真实可分信号，且噪声上限越低越容易区分；但分类更容易本身不等于生成更好。

四 seed rollout 显示不同 sampler 是真实 Pareto，而非按 critic loss 单调排序。相对各自完全相同的 legacy recovery control，uniform `[0,0.9]` 的 FK 在 `4/4` seed 改善、均值 `-5.49%`，但 waypoint、foot、jerk、body-HF 分别回退约 `9.64%/3.29%/2.82%/4.62%`。high-noise `[0,0.9]` 的 FK/seam/root-horizontal-HF 分别平均改善 `2.62%/2.52%/2.17%`，且均为 `4/4` seed，代价是 waypoint `+2.28%`、foot `+1.21%`、jerk `+1.00%`。uniform `[0,0.5]` 的 waypoint 基本持平 `-0.05%`，FK/seam/body-HF 改善 `8.68%/4.89%/2.83%`，但 jerk 与 root-heading-HF 回退 `1.53%/5.24%`。因此 `[0,0.5]` 是当前更强的候选，但不能仅凭该轮宣布最优。机器结果为 `distill_runs/dmd_ladd_critic_time_from_g200_joint22500_20260715/critic_time_summary.json`。

进一步审计发现 generator 计算 adversarial loss 时仍沿用 legacy score timestep；即便 critic 用了有信息的 sampler，约 70% generator adversarial batch 仍在 exact `t=1` 上得到零输入梯度。为保持历史实验可复现，新增 `adversarial_time_sampler={score,critic}`，默认 `score` 不改变旧行为；选择 `critic` 时，generator adversarial feature 使用与 critic 训练相同的 exact/high-noise/upper-bound，并记录 mean/min/max/exact-t1 fraction。下一轮不从头训练：从已完成的 uniform-u05 与 high50-u09 g300 完整状态分别继续到 g400，各做 legacy/aligned 一对，形成严格 2×2；每对只改变 generator adversarial-time sampler，generator/fake-score/critic、三组 Adam、EMA、sampler 和 RNG 起点完全相同。

### 23.24 Generator adversarial-time 2×2 与 u05-g400 前端候选（2026-07-15）

上述 2×2 已全部从各自完全相同的 g300 full state 继续到 g400，并完成固定 full/text validation、seed `20260714..20260717` 四条各 50-window 原版无限 rollout 和 jitter。对齐 generator adversarial-time 的实现确实生效：u05/high50 aligned 分支记录的 adversarial exact-t1 fraction 都从约 `0.75` 降为 `0`，采样均值分别从约 `0.912` 降到 `0.235/0.599`；这修复了大部分 adversarial batch 在 exact `t=1` 没有输入梯度的代码问题。但方法更合理没有自动转化为更好的动作分布。

严格同起点、同 seed 的 aligned-vs-legacy 直接对比中，u05 aligned 的 waypoint 平均改善 `1.03%`，但 FK/foot/seam/jerk/body-HF/root-horizontal-HF/root-heading-HF 分别回退 `2.55%/1.65%/1.73%/0.57%/10.14%/4.40%/46.50%`，其中 body-HF 与 heading-HF 为 `0/4` seed 胜出。high50 aligned 的 waypoint 改善 `1.36%`，但 root-drift-mean/final、FK、seam、jerk、body-HF 分别回退 `2.99%/44.73%/2.26%/1.36%/2.44%/19.04%`；jerk 为 `0/4` seed 胜出。aligned 分支因此不进入本轮前端，不会因为它修了理论 bug 就掩盖经验上的负结果。机器汇总为 `distill_runs/dmd_ladd_adversarial_time_from_g300_joint22500_20260715/adversarial_time_summary.json`。

四臂中用于视觉实验的是 `u05_legacy_advtime`，不是宣称全局最优。它相对 stage-one 的 waypoint/FK/foot/seam/body-jerk/body-HF/root-horizontal-HF/root-heading-HF 分别为 `-12.91%/-2.08%/-4.98%/-2.50%/-0.86%/-14.10%/-5.49%/-33.52%`，其中 waypoint/FK/foot/body-HF/root-horizontal-HF/root-heading-HF 在 `3/4/3/3/3/4` seed 改善；代价是 root-final-drift 均值回退 `17.28%` 且 seed 方差很大。相较活动 g300，它牺牲部分 waypoint/FK/seam/jerk，换取更低的 body 与 heading 高频，正好适合用户肉眼判断“抖动是否真的减少”。

该候选已导出为 `text_qwen_dmd_ladd_joint22500_u05legacy_g400_ema_pilot_20260715`：motion student 仍为 `51,952,000` 参数，FP16 learned graphs `104,062,612 B`，含 FP32 utilities 共 `104,107,876 B = 99.285 MiB`。encoder/flow/decoder 模块导出相对 PyTorch FP32 的 max abs 为 `0.001290/0.003054/0.007401`，全部 finite。完整 CPU FP16 gate 保留失败事实：continuation same-precision/FP32 endpoint max abs 为 `0.042969/0.040972`，explicit mean/p99/cosine 为 `0.011885/0.064091/0.999890`，FSQ match `91.484%`；没有放宽 `0.025/0.02/0.015/0.075/0.9999/94%` 阈值。

尽管 CPU 单 case 较 g300 弱，真实 Edge full golden 对本候选反而相对 g300 全面改善且全部 finite：initial endpoint/FSQ/explicit-mean 从 g300 的 `0.189453/66.406%/0.041676` 改善为 `0.172363/69.609%/0.034661`；continuation 从 `0.087891/85.234%/0.036630` 改善为 `0.050781/89.375%/0.036768`。它仍未通过绝对 WebGPU 阈值，所以 release 保持 `experimental_pilot`，但相对结果支持将其保留给视觉测试。活动页面已加载该 release 且处于暂停、frame 0；g300 manifest、golden 与权重完整保存在 `webgpu_toy/infinite_demo/candidates/g300/` 和版本化模型目录，可随时无损回滚。

DMD+LADD 的总体方法论据此继续按实验科学执行，而非固定 recipe：LR 至少覆盖 `1e-8..1e-6`，generator iteration 看 few-hundred 到 few-thousand 的非单调 Pareto，G/score/critic 频率、critic 与 generator timestep、warmup、adv/paired/FSQ 权重、clip、EMA 和 critic feature/framework 均视为独立变量。每次只做可归因的小矩阵，保留完整 state 续训；静态、分层条件集、多 seed 长闭环、真实 WebGPU 数值、延迟和用户视觉反馈共同构成证据，任何单一 loss、单 seed 或理论一致性都不能被称为“最优效果”。

### 23.25 修正 critic 框架后的 generator LR × iterations 完整矩阵（2026-07-15）

首轮 generator-LR sweep 发生在 exact-`t=1` 退化 critic 下，不能直接外推到修正后的框架。因此第二轮固定 uniform-u05 critic、legacy generator adversarial-time、score/critic LR `5e-6/1e-6`、G:score:critic `1:1:1`、adv/paired/FSQ `1e-3/0.1/0.1`、EMA `0.995` 和全部数据/seed，只扫描 generator LR `1e-8/3e-8/1e-7/3e-7/1e-6`。非 control 臂都从同一个 uniform-u05 g300 完整状态开始；`1e-7` control 从其完全相同轨迹的 u05-legacy g400 完整状态继续，避免重复前 100 step。五臂都正常完成到 g1500、guidance1900，每 100 generator update 保存 raw/EMA/full Accelerate state，所有权重和日志 finite。

为防止假 sweep，训练器新增显式 `--override-learning-rates-on-resume`：先完整恢复 generator/fake-score/critic、三组 Adam moments/scheduler、EMA、sampler 和各 rank RNG，再同时覆盖 optimizer param-group LR 与 LambdaLR base/last LR。否则 Accelerate 会用 checkpoint LR 覆盖命令行值，下一次 scheduler step 还会把只改 optimizer 的 LR 改回去。首个恢复日志实测 generator LR 为 `9.99999994e-9`、score/critic 为 `5e-6/1e-6`；新增单测验证覆盖值在 optimizer+scheduler step 后仍保持，flow-matching 共 11 项测试通过。没有重新初始化任何已有权重或 Adam state。

每条 LR 的 g400/g700/g1000/g1500（`1e-7` 以 g500 代替已在原目录验证的 g400）均完成 16k full、分层 text/control validation；四个关键点又分别完成 seed `20260714..20260717` 四条各 50-window 原版无限 rollout 和 jitter，总计 80 条长 rollout。相对同 seed stage-one 的主要结果如下，负值表示更好：

| generator LR @ step | static endpoint | waypoint | FK | foot | seam | body jerk | body HF |
|---|---:|---:|---:|---:|---:|---:|---:|
| `1e-8 @ 1000` | `+2.841%` | `-16.665%` | `-0.948%` | `-3.416%` | `-3.485%` | `-2.247%` | `-9.696%` |
| `3e-8 @ 1000` | `+3.256%` | `-13.763%` | `-1.395%` | `-3.593%` | `-1.906%` | `-2.152%` | `-15.777%` |
| `3e-8 @ 1500` | `+3.674%` | `-20.566%` | `-7.767%` | `-2.169%` | `+0.948%` | `-2.392%` | `-19.103%` |
| `1e-7 @ 700` | `+3.366%` | `-8.763%` | `-7.471%` | `-2.476%` | `-0.401%` | `-2.243%` | `-15.077%` |
| `3e-7 @ 400` | `+2.128%` | `-17.322%` | `-2.486%` | `-2.828%` | `-1.968%` | `-3.734%` | `-16.436%` |
| `3e-7 @ 700` | `+4.467%` | `-17.359%` | `-5.860%` | `-1.363%` | `+4.822%` | `+2.221%` | `-27.008%` |
| `1e-6 @ 700` | `+6.234%` | `-14.454%` | `+1.220%` | `+3.550%` | `+3.308%` | `+6.339%` | `-17.756%` |
| `1e-6 @ 1500` | `+8.864%` | `-12.734%` | `-5.270%` | `+4.901%` | `+7.409%` | `+16.254%` | `-24.192%` |

曲线明显非单调。`1e-8` 的 waypoint 在 g400/g700/g1000/g1500 为 `-8.49/-11.01/-16.67/-10.62%`，且 g1500 jerk 已从 g1000 的 `-2.25%` 回到 `+0.71%`；`1e-7` g700 的 FK/foot/jerk/body-HF 为 `-7.47/-2.48/-2.24/-15.08%`，g1000 却变成 `+2.51/+1.45/+3.83/-17.15%`。这直接排除了按更多 iterations、training loss 或单个 HF 指标机械选模。`3e-7` 仍有一个很窄但有效的 g400 早停窗口：只从共同 g300 再走 100 update，waypoint/FK/foot/seam/jerk/body-HF 同时改善；继续到 g700 后 seam/jerk 已转为 `+4.82/+2.22%`，g1500 更到 `+7.99/+8.16%`。`1e-6` 连 g400 的 seam 已回退 `2.35%`，后续只会把能量与路径指标换成更严重的 seam/jerk/foot 和配对保真损失，在当前框架下没有成为合理 Pareto，不再延长。

当前未宣布唯一最优。较均衡的 Pareto 是 `1e-8@g1000`、`3e-8@g1000`、`3e-8@g1500`、`1e-7@g700` 与短程 `3e-7@g400`，其中 `3e-8@g1500` 的 waypoint/root-drift/FK/body-HF 为 `4/4` seed 改善，但 seam 平均回退且只 `1/4` seed 胜出。与活动前端 u05-g400 直接比较，新 Pareto 点可换得更好的 path/FK/jerk，但 root-heading 高频普遍明显回退；例如 `3e-7@g400` 直接改善 waypoint/jerk/body-HF `4.81/2.74/2.10%`，同时使 foot/seam/root-heading-HF 回退 `2.44/0.55/23.92%`。因此在用户完成视觉测试前不覆盖活动版本。完整机器证据位于 `distill_runs/dmd_ladd_u05_generator_lr_iters_to1500_joint22500_20260715/generator_lr_iters_summary.json`，不包含人为加权总分。

新增可复现实行脚本为 `scripts/run_dmd_ladd_u05_generator_lr_iters.sh`、`scripts/evaluate_dmd_ladd_lr_checkpoint_static.sh`、`scripts/evaluate_dmd_ladd_lr_checkpoint_rollout.sh`，汇总器为 `ardy_distill/tools/summarize_dmd_ladd_lr_iters.py`。训练使用物理 GPU `0/4/5/7`，full/text/四 seed rollout 同时使用其余 `1/2/3/6`，训练与验证并行而不互相覆盖。全过程没有安装任何包、没有执行 `conda` 命令，也没有修改 `<REFERENCE_3DLM_PROJECT>`。

前端无人值守诊断还修复了一个与模型无关的后台-tab 卡点：`yieldToAnimationFrame()` 原先在隐藏 Edge tab 中可能永久暂停，使服务器 precision-probe 一直显示 running；现改为 rAF 或 100 ms timer 先到者并通过 ES-module 语法检查。当前已加载旧脚本的标签页需要刷新一次才使用该 fallback；模型 release、manifest 与暂停 frame 0 状态不受影响。

### 23.26 修正 u05 critic 后独立 G:score:D 更新频率矩阵（2026-07-16）

旧训练器的 `guidance_updates_per_generator` 会把 fake-score 与 discriminator 强制绑定，因此此前只能测 `1:n:n`，无法区分多做 score 与多做 D 的作用。训练器现新增独立 `score_updates_per_generator` 与 `critic_updates_per_generator`，旧参数仍作为两者的默认值以保持向后兼容。`TrainingCounters` 同时保存 score、critic 与 generator 三个计数；读取旧 checkpoint 时，由于旧代码每个 guidance batch 确实各更新 score/D 一次，两个新计数可以精确继承旧 `guidance_updates`，不是估计值。ratio 原点锚定在 resume 后的当前计数，改变 ratio 只约束未来更新，不会为历史 generator step 追补数百次 score/D。完整 Adam moments、scheduler、EMA、sampler 和四 rank RNG 均继续继承。计数迁移、warmup 边界、`1:2:1` 推进、LR override 与 flow 数学共 14 项单测通过。

本轮固定 uniform-u05 critic、legacy generator adversarial-time、generator/score/D LR `1e-7/5e-6/1e-6`、adv/paired/FSQ `1e-3/0.1/0.1`、EMA `0.995`、数据、seed 和 batch，只比较 `G:score:D = 1:1:1 / 1:2:1 / 1:1:2 / 1:2:2`。四臂都从同一个 u05 g300 full Accelerate state 分叉，继续到 g500，在 g350/g400/g450/g500 保存 raw、EMA 和 full state。最终 score/D 计数分别为 `900/900`、`1100/900`、`900/1100`、`1100/1100`，与设计严格一致；所有日志和权重 finite。新调度器的 `1:1:1@g400` EMA SHA256 为 `8310298168218b8563ccf370cebf5b35e7212cd9d42c6f257b45011881adac85`，与重构前活动 u05-g400 release 逐字节相同，证明兼容路径没有改变原轨迹。

每个 checkpoint 均完成 16k full、分层 text/control validation，以及 seed `20260714..20260717` 四条各 50-window 原版无限 rollout 和 jitter，共 16 个静态点、64 条长闭环。下表均相对同 seed stage-one，负值表示更好：

| G:score:D @ step | static endpoint | waypoint | FK | foot | seam | body jerk | body HF |
|---|---:|---:|---:|---:|---:|---:|---:|
| `1:1:1 @ 350` | `+1.679%` | `-10.128%` | `+2.465%` | `-3.496%` | `+0.368%` | `-1.256%` | `-6.569%` |
| `1:2:1 @ 350` | `+1.675%` | `-8.190%` | `+0.725%` | `-3.185%` | `-1.778%` | `-1.383%` | `-8.610%` |
| `1:1:2 @ 350` | `+1.677%` | `-7.576%` | `+4.732%` | `-0.397%` | `+3.534%` | `-0.068%` | `-6.054%` |
| `1:2:2 @ 350` | `+1.677%` | `-8.308%` | `-3.002%` | `-3.450%` | `+0.325%` | `+0.115%` | `-5.590%` |
| `1:1:1 @ 400` | `+1.961%` | `-12.909%` | `-2.080%` | `-4.980%` | `-2.497%` | `-0.855%` | `-14.101%` |
| `1:2:1 @ 400` | `+1.947%` | `-0.342%` | `-0.011%` | `-3.020%` | `-2.000%` | `-0.310%` | `-3.426%` |
| `1:1:2 @ 400` | `+1.956%` | `-14.544%` | `-2.340%` | `-1.769%` | `-2.068%` | `+0.442%` | `-12.355%` |
| `1:2:2 @ 400` | `+1.947%` | `-11.753%` | `+1.645%` | `-3.636%` | `-1.797%` | `-1.395%` | `-10.618%` |
| `1:1:1 @ 450` | `+2.221%` | `-8.066%` | `-2.158%` | `-3.288%` | `-0.096%` | `-1.595%` | `-10.183%` |
| `1:2:1 @ 450` | `+2.203%` | `-12.173%` | `-4.941%` | `-2.098%` | `-0.940%` | `+0.647%` | `-13.098%` |
| `1:1:2 @ 450` | `+2.223%` | `-13.526%` | `+1.653%` | `-2.520%` | `+1.909%` | `-0.653%` | `-10.229%` |
| `1:2:2 @ 450` | `+2.213%` | `-10.578%` | `+0.074%` | `-3.777%` | `-0.148%` | `+0.386%` | `-15.870%` |
| `1:1:1 @ 500` | `+2.472%` | `-9.406%` | `-5.324%` | `-1.839%` | `-2.743%` | `-0.165%` | `-12.356%` |
| `1:2:1 @ 500` | `+2.449%` | `-15.301%` | `-0.288%` | `-4.081%` | `+0.006%` | `-4.464%` | `-16.103%` |
| `1:1:2 @ 500` | `+2.476%` | `-12.560%` | `+0.187%` | `-2.598%` | `+0.375%` | `-1.245%` | `-11.713%` |
| `1:2:2 @ 500` | `+2.460%` | `-15.838%` | `-0.774%` | `-4.071%` | `-1.240%` | `+0.544%` | `-12.191%` |

同 step 对照与活动前端对照给出不同但一致的结论。`1:2:1@g500` 相对同 step `1:1:1@g500` 改善 waypoint/foot/jerk/body-HF `6.30/2.27/4.18/3.39%`，说明额外 fake-score 更新并非始终无效；但相对实际活动前端 `1:1:1@g400`，它仅改善 waypoint/jerk/body-HF `1.63/3.37/0.41%`，同时回退 root-drift-mean/FK/foot/seam/root-heading-HF `1.66/1.90/1.11/2.72/39.86%`，heading-HF 为 `0/4` seed 胜出。`1:1:2@g400` 虽将 waypoint 再改善 `1.62%`，却使 foot/jerk/body-HF/root-heading-HF 回退 `3.55/1.41/2.64/45.91%`。`1:2:2@g500` 的 waypoint/root-drift-mean 相对活动版改善 `3.61/1.74%`，代价是 root-final-drift/FK/foot/seam/jerk/root-heading-HF 回退 `18.27/1.42/1.16/1.48/1.68/50.39%`。因此当前证据不支持把多 D 或多 score 版本替换到前端，活动 `1:1:1@g400` 继续保留；这不是宣称其全局最优，而是没有新臂通过当前平衡 guardrail。

完整机器汇总为 `distill_runs/dmd_ladd_u05_independent_ratio_from_g300_joint22500_20260716/independent_ratio_summary.json`，不含人为加权总分。训练脚本为 `scripts/run_dmd_ladd_u05_independent_ratio.sh`，验证编排为 `scripts/evaluate_dmd_ladd_u05_independent_ratio.sh`，汇总器为 `ardy_distill/tools/summarize_dmd_ladd_independent_ratio.py`。证据审计成功解析 160 份 static/rollout/jitter JSON，64/64 条 rollout 的 final snapshot 全部 finite，64/64 份 safetensors 均可完整读取，日志错误扫描为空。g500 的 `1:2:1` 曾被两个验证编排短暂重复调度；停止重复进程后又在独立目录重跑四个 seed，原结果与重跑结果的 rollout snapshots、jitter snapshots 和每个 safetensors tensor value 均逐元素完全一致，故没有把潜在并发写坏文件纳入结论。

本轮说明频率与 iterations 强耦合：相同 ratio 在 g350/g400/g450/g500 可从收益变成损失，不能把 `n_critic=2` 或 `n_score=2` 固定成 recipe。下一项框架实验应固定短程迭代和 ratio，独立比较 critic feature taps、聚合方式、head 容量/正则与 generator adversarial coupling；继续增加同一 D 更新次数的优先级较低。全过程没有安装任何包、没有执行 `conda` 命令，也没有修改 `<REFERENCE_3DLM_PROJECT>`。

### 23.27 单变量 critic feature-tap 框架实验（2026-07-16）

为隔离 LADD 判别特征层级，flow 新增了不改变 state dict 的 `forward_with_features(feature_tap=...)`。四个 tap 都输出 `[B,10,512]`：`trunk_final` 是 8 层 trunk 后、root head 前的 generation token；`body_pre` 是 root 预测注入后、body refiner 前；`body_mid` 是 8 层 body refiner 的第 4 层后；`body_final` 是历史实现使用的第 8 层后特征。相同输入下四种 tap 的 generator prediction 逐元素完全相同，模型参数量、部署图和浏览器推理均不变；只有训练期 frozen teacher feature 与 critic head 的输入改变。新增的 tap 形状、输出恒等、特征非恒等和非法参数测试，与此前计数/LR/flow 测试合计 15 项全部通过。

四臂均从完全相同的 corrected uniform-u05 g300 full Accelerate state 分叉。源状态的 generator/score/D 计数为 `300/700/700`；每臂先统一把 score/D 恢复到 `800/800`，期间 generator 保持 g300，再以 `G:score:D=1:1:1` 训练到 g500，最终计数严格为 `500/1000/1000`。generator/score/D LR 固定为 `1e-7/5e-6/1e-6`，critic time 为 uniform `[0,0.5]`，generator adversarial time 保持 legacy score sampler，adv/paired/FSQ 权重为 `1e-3/0.1/0.1`，EMA 为 `0.995`；数据、batch、seed、optimizer moments、scheduler、sampler 和四 rank RNG 均继承一致。四臂都以 `training_complete`、`stopped_by_runtime=false` 正常结束，g350/g400/g450/g500 的 raw、EMA 和完整 state 齐全。

训练诊断本身不能用于选模，但揭示了框架尺度问题。`body_final/body_pre/body_mid/trunk_final` 在公共 recovery 段的 feature-gap 均值分别为 `0.5324/0.2594/0.3655/0.2609`，generator 段为 `0.5410/0.2669/0.3726/0.2682`；对应 generator 段 critic loss 为 `1.3180/1.3422/1.3265/1.3469`，generator adversarial loss 为 `0.8092/0.6737/0.5329/0.6772`。这说明固定同一个 `adv_weight` 时，不同层的可分性、logit 与传给 generator 的梯度尺度并不天然可比。单 tap 结果只能回答“当前未归一化实现中哪个组合更合适”，不能把层级差异错误解释成普适 LADD 结论。

16 个 checkpoint 均完成 16k full、分层 text/control validation，以及 seed `20260714..20260717` 四条各 50-window 原版无限 rollout 和 jitter。下表相对同 seed stage-one，负值表示更好；短程静态 endpoint 在同 step 的四个 tap 间最多只相差约 `0.003` 个百分点，因此主要判别力来自长闭环。

| critic tap @ step | static endpoint | waypoint | FK | foot | seam | body jerk | body HF | heading HF |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `body_final @ 350` | `+1.672%` | `-7.729%` | `-7.198%` | `-3.071%` | `-2.133%` | `-0.937%` | `-5.713%` | `-16.462%` |
| `body_pre @ 350` | `+1.672%` | `-10.217%` | `-5.908%` | `-5.023%` | `+0.166%` | `-1.928%` | `-12.454%` | `-11.877%` |
| `body_mid @ 350` | `+1.671%` | `-12.817%` | `-0.384%` | `-2.893%` | `-0.491%` | `-2.023%` | `-10.812%` | `-21.656%` |
| `trunk_final @ 350` | `+1.671%` | `-13.286%` | `-2.564%` | `-4.702%` | `-1.559%` | `-3.859%` | `-9.296%` | `-11.938%` |
| `body_final @ 400` | `+1.945%` | `-8.771%` | `+0.974%` | `-1.476%` | `-0.663%` | `+0.902%` | `-9.332%` | `-18.729%` |
| `body_pre @ 400` | `+1.944%` | `-12.019%` | `-5.188%` | `-2.069%` | `+0.449%` | `-1.877%` | `-11.585%` | `-4.999%` |
| `body_mid @ 400` | `+1.943%` | `-3.805%` | `-0.344%` | `-0.997%` | `+0.305%` | `+2.272%` | `-4.492%` | `-16.715%` |
| `trunk_final @ 400` | `+1.945%` | `-8.314%` | `-7.854%` | `-0.986%` | `-3.253%` | `-2.035%` | `-7.370%` | `-12.455%` |
| `body_final @ 450` | `+2.209%` | `-10.393%` | `+2.892%` | `-2.494%` | `+0.353%` | `-1.050%` | `-10.625%` | `-7.997%` |
| `body_pre @ 450` | `+2.209%` | `-10.438%` | `-1.973%` | `-0.142%` | `-3.473%` | `-0.752%` | `-11.128%` | `-14.403%` |
| `body_mid @ 450` | `+2.209%` | `-10.937%` | `-1.921%` | `-4.387%` | `-0.580%` | `-1.943%` | `-7.598%` | `-9.574%` |
| `trunk_final @ 450` | `+2.209%` | `-12.092%` | `-11.465%` | `-1.477%` | `-1.374%` | `-2.384%` | `-10.501%` | `-16.238%` |
| `body_final @ 500` | `+2.464%` | `-16.325%` | `+1.692%` | `+0.041%` | `+2.278%` | `+0.148%` | `-7.652%` | `-26.778%` |
| `body_pre @ 500` | `+2.462%` | `-12.548%` | `+0.884%` | `-4.511%` | `-1.343%` | `-1.986%` | `-13.429%` | `-17.165%` |
| `body_mid @ 500` | `+2.463%` | `-8.413%` | `-5.549%` | `-2.048%` | `-1.803%` | `-0.057%` | `-15.538%` | `-6.057%` |
| `trunk_final @ 500` | `+2.461%` | `-17.036%` | `-4.034%` | `-2.806%` | `+0.907%` | `-1.511%` | `-16.030%` | `-18.859%` |

结果再次表现出强烈的 tap × iterations 非单调性。`trunk_final@g350` 偏 waypoint/jerk，`trunk_final@g400/g450` 分别偏 FK+seam 与 FK，`body_pre@g500` 偏 foot/body-HF，而 `trunk_final@g500` 在 waypoint、FK、jerk、body-HF 上较均衡。即使如此，没有新点通过活动前端 guardrail。最接近的 `trunk_final@g500` 相对当前 `body_final 1:1:1@g400` 活动候选改善 waypoint/root-mean/root-final/FK/jerk/body-HF `3.63/0.83/7.43/1.93/0.43/1.13%`，同时回退 foot/seam/root-horizontal-HF/root-heading-HF `2.42/3.70/3.11/28.19%`；seam 与 heading-HF 都是 `0/4` seed 胜出。`trunk_final@g350` 虽改善 waypoint/FK/jerk `0.07/0.44/2.87%`，但 final drift、body-HF、heading-HF 回退 `8.16/5.54/43.76%`。因此不把任何 feature-tap 分支替换到网页，活动 release 继续是 `text_qwen_dmd_ladd_joint22500_u05legacy_g400_ema_pilot_20260715`；这是证据不足以换版，不是把旧候选宣布为全局最优。

机器汇总为 `distill_runs/dmd_ladd_u05_feature_tap_from_g300_joint22500_20260716/feature_tap_summary.json`，训练、验证和汇总脚本分别为 `scripts/run_dmd_ladd_u05_feature_taps.sh`、`scripts/evaluate_dmd_ladd_u05_feature_taps.sh` 和 `ardy_distill/tools/summarize_dmd_ladd_feature_taps.py`。最终审计成功解析 165 份 JSON；64/64 条 rollout 与 64/64 份 jitter 全部 finite；16 个 EMA checkpoint 与 64 份固定 rollout case safetensors 均逐 tensor finite；日志错误扫描为空。评估编排对实验目录加了非持久 `flock`，避免第二套任务重复写相同证据目录。

下一轮框架变量不应继续机械扫描单 tap。更接近 LADD 的可检验方向是：对 `trunk_final/body_mid/body_final` 使用独立小 head 的多层判别，分别记录 real/fake logit、D 梯度和传给 generator 的梯度 RMS；先比较等权多头与按 generator-gradient RMS 归一化的多头，再单独测试 head 容量或正则，不能把三者一次混在一起。仍须从已有完整状态继承，以 few-hundred 到 few-thousand 的多 checkpoint、matched-seed 长闭环和真实浏览器视觉作为最终证据。全过程没有安装任何包、没有执行 `conda` 命令，也没有修改 `<REFERENCE_3DLM_PROJECT>`。

### 23.28 共享 head 的多层 LADD 聚合、梯度尺度与独立 A/B 前端（2026-07-16）

单 tap 结果显示不同 frozen-teacher 层的 feature gap 与 generator adversarial 尺度不一致，因此没有直接引入多个随机 head 或同时改正则。第一轮多层框架采用一个完全继承的 `ScoreBackboneCriticHead`，对每个 tap 独立 forward，再比较两种无新增参数的聚合：`mean_loss` 分别计算每层 real/fake BCE 和 generator non-saturating loss 后等权平均；`mean_logit` 先平均各层 logits，再计算一次对应 loss。teacher 的一次 forward 同时返回所需 feature dict，generator prediction 与 state dict 不变；critic 参数、Adam state 和所有部署图也不变。多层顺序、key/shape/输出恒等、两种聚合公式、旧计数/LR/flow 数学合计 16 项测试通过。

四臂为旧 `body_final` 单层精确控制、`body_mid+body_final / mean_loss`、`trunk_final+body_mid+body_final / mean_loss`、同三层 `mean_logit`。它们都从同一 corrected-u05 g300 full state 开始，先做相同的 score/D `700→800` recovery，再以 `G:score:D=1:1:1` 到 g500；generator/score/D LR、timestep、adv/paired/FSQ、EMA、数据和 seed 与 23.27 完全相同。最终四臂均为 G/S/D `500/1000/1000`，以 `training_complete/stopped_by_runtime=false` 正常退出，训练 wall time 分别为 `107.74/112.87/113.08/115.44 s`。控制臂 g350/g400/g450/g500 的 EMA SHA256 与 23.27 的 `body_final` 四个对应 checkpoint 全部逐字节相同，证明新代码没有改变旧轨迹。

逐层 endpoint-gradient RMS 只作为机制诊断，不参与质量选模。两层 mean-loss 的 body-mid/body-final 全程记录均值为 `4.42e-5/4.19e-5`，最大/最小仅 `1.05×`；三层 mean-loss 的 trunk/body-mid/body-final 为 `7.71e-5/4.35e-5/4.03e-5`，最大/最小 `1.91×`；三层 mean-logit 为 `7.33e-5/5.54e-5/4.44e-5`，降到 `1.65×`。这证明共享 head 自适应后深层两 tap 可以近似平衡，但加入 trunk 会让其梯度系统性占优；“loss 等权”不能冒充“generator 梯度等权”。

四臂的 g350/g400/g450/g500 均完成相同的 16k full、分层 text/control、四 seed 50-window 无限 rollout 与 jitter。下表相对同 seed stage-one，负值表示更好：

| 聚合 @ step | static endpoint | waypoint | FK | foot | seam | body jerk | body HF | heading HF |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `single body_final @ 350` | `+1.672%` | `-7.729%` | `-7.198%` | `-3.071%` | `-2.133%` | `-0.937%` | `-5.713%` | `-16.462%` |
| `deep-2 mean-loss @ 350` | `+1.672%` | `-13.798%` | `+0.873%` | `-2.809%` | `+2.491%` | `-0.432%` | `-10.366%` | `-8.814%` |
| `three mean-loss @ 350` | `+1.671%` | `-13.276%` | `+2.542%` | `-2.075%` | `+3.935%` | `+0.019%` | `-8.079%` | `-9.436%` |
| `three mean-logit @ 350` | `+1.672%` | `-15.473%` | `-3.130%` | `-4.536%` | `-2.523%` | `-2.127%` | `-9.021%` | `-19.339%` |
| `single body_final @ 400` | `+1.945%` | `-8.771%` | `+0.974%` | `-1.476%` | `-0.663%` | `+0.902%` | `-9.332%` | `-18.729%` |
| `deep-2 mean-loss @ 400` | `+1.945%` | `-14.986%` | `-0.000%` | `-5.139%` | `-1.841%` | `-2.436%` | `-9.828%` | `-12.006%` |
| `three mean-loss @ 400` | `+1.945%` | `-15.397%` | `+3.416%` | `-3.470%` | `+1.385%` | `-0.242%` | `-12.147%` | `-19.265%` |
| `three mean-logit @ 400` | `+1.944%` | `-15.240%` | `+0.187%` | `-2.916%` | `-0.510%` | `-1.110%` | `-9.519%` | `-12.777%` |
| `single body_final @ 450` | `+2.209%` | `-10.393%` | `+2.892%` | `-2.494%` | `+0.353%` | `-1.050%` | `-10.625%` | `-7.997%` |
| `deep-2 mean-loss @ 450` | `+2.210%` | `-17.781%` | `-4.374%` | `-5.858%` | `-3.576%` | `-3.671%` | `-13.034%` | `-19.308%` |
| `three mean-loss @ 450` | `+2.209%` | `-15.619%` | `+0.041%` | `-3.453%` | `+1.098%` | `-1.041%` | `-10.510%` | `-26.365%` |
| `three mean-logit @ 450` | `+2.210%` | `-8.766%` | `-1.148%` | `-0.874%` | `+0.229%` | `+2.478%` | `-13.798%` | `-11.467%` |
| `single body_final @ 500` | `+2.464%` | `-16.325%` | `+1.692%` | `+0.041%` | `+2.278%` | `+0.148%` | `-7.652%` | `-26.778%` |
| `deep-2 mean-loss @ 500` | `+2.461%` | `-13.802%` | `-2.757%` | `-0.913%` | `+0.130%` | `+0.892%` | `-14.108%` | `-13.074%` |
| `three mean-loss @ 500` | `+2.463%` | `-8.871%` | `-5.173%` | `-4.315%` | `-2.500%` | `-3.538%` | `-13.735%` | `-20.313%` |
| `three mean-logit @ 500` | `+2.463%` | `-10.759%` | `+3.846%` | `+1.644%` | `+3.072%` | `+0.369%` | `-7.814%` | `-11.947%` |

多层与聚合同样强烈依赖 iteration。三层 mean-logit 在 g350 是本轮最早的均衡窗口，相对 stage-one 同时改善 waypoint/FK/foot/seam/jerk；到 g450 后 waypoint、foot、seam、jerk 已明显转坏。两层 mean-loss 在 g450 的主要均值最强，但不能只看表中七项。直接对活动前端时，它改善 waypoint/root-mean/FK/foot/seam/jerk `5.94/3.26/2.38/0.80/0.91/2.74%`，其中 root-mean 为 `4/4` seed 改善；同时 root-final-drift 回退 `64.80%`，body-HF/root-horizontal-HF/heading-HF 回退 `1.48/1.59/27.33%`，heading-HF 为 `0/4` seed 胜出。三层 mean-logit g350 相对活动版改善 waypoint/FK/jerk `3.05/1.11/1.11%`，但 final drift、body-HF、heading-HF 回退 `10.43/6.53/32.15%`。因此没有任何多层分支满足默认 release guardrail，活动 `u05legacy-g400` 不被覆盖。

为获得用户视觉证据而不污染默认版本，`deep-2 mean-loss@g450` 已另外导出成只读 A/B 候选 `text_qwen_dmd_ladd_joint22500_multideep_meanloss_g450_ema_visual_20260716`。页面新增受限于 `[a-z0-9_-]+` 的 `candidate` 查询参数，默认 manifest 与旧 URL 行为不变；候选 URL 为 `http://127.0.0.1:8766/infinite_demo.html?candidate=multitap_deep_meanloss_g450`，页面日志会明确显示“A/B 候选、默认 release 未替换”。候选 learned graphs 为 `104,062,612 B`，含 utilities 总下载 `104,107,876 B = 99.285 MiB`；encoder/flow/decoder 的 CPU ORT 模块 max abs 为 `0.001290/0.004032/0.007401`，全部 finite。

候选专属 CPU 整链路 golden 已生成。same-precision endpoint max abs 为 `0.010742`，FP32 explicit mean/p99/cosine 与 FSQ match 为 `0.008567/0.045297/0.9999433/96.484%`，均通过既定线；唯一失败是 FP32 endpoint max abs `0.021363` 略高于 `0.020`，因此 server validation 仍如实为 `passed=false`，没有放宽阈值。真实 Edge WebGPU 数值与视觉尚待用户打开上述 A/B URL 后验证，不能以 CPU 结果代替。

机器汇总为 `distill_runs/dmd_ladd_u05_multitap_shared_head_from_g300_joint22500_20260716/multitap_summary.json`；训练、验证、汇总脚本分别为 `scripts/run_dmd_ladd_u05_multitap_shared_head.sh`、`scripts/evaluate_dmd_ladd_u05_multitap_shared_head.sh`、`ardy_distill/tools/summarize_dmd_ladd_multitap.py`。审计成功解析 165 份 JSON，64/64 rollout、64/64 jitter、16 个 EMA 与 64 份 fixed case 均逐元素 finite，日志错误扫描为空。下一轮若继续追两层 mean-loss，应先单变量扫描更低 adversarial weight，检验能否保留 g450 的 path/FK/foot/seam/jerk 收益并收回 final-drift/heading-HF；不能同时改 root-temporal/path 权重，否则无法归因。全过程没有安装任何包、没有执行 `conda` 命令，也没有修改 `<REFERENCE_3DLM_PROJECT>`。

### 23.29 deep-2 mean-loss 的 adversarial-weight 消融（2026-07-16）

为检验上一节 `multi_deep_mean_loss@g450` 的 final-drift/heading-HF 回退是否来自过强 LADD coupling，本轮严格固定 corrected uniform-u05 critic、`body_mid+body_final` 两层共享 head、`mean_loss` 聚合、legacy score adversarial-time、generator/score/D LR `1e-7/5e-6/1e-6`、`G:score:D=1:1:1`、DMD/paired/FSQ 与全部物理正则、EMA、数据、batch 和 seed，只扫描 generator loss 中的 adversarial weight `0/1e-4/3e-4/1e-3`。`adv=0` 是仍训练 score/D、但不向 generator 注入 LADD 梯度的必要消融；`1e-3` 是上一轮控制臂。四臂均从同一个 g300 full Accelerate state 恢复，重复完全相同的 score/D `700→800` recovery，再训练到 g500；每臂 g350/g400/g450/g500 的 raw、EMA 和完整 state 齐全，最终计数均为 `G/S/D=500/1000/1000`，wall time 分别为 `112.44/112.09/110.70/114.36 s`，全部以 `training_complete`、`stopped_by_runtime=false` 结束。

矩阵可归因性通过了强复现检查：四臂 resume config 的顶层差异只有 `output` 与 `adversarial_weight`；新 `adv1e3` 在 g350/g400/g450/g500 的 EMA SHA256 与上一轮 `multi_deep_mean_loss` 对应 checkpoint 逐字节完全相同。所有 16 个 checkpoint 均完成 16k full、分层 text/control、seed `20260714..20260717` 四条各 50-window 原版无限 rollout 和 jitter。下表相对同 seed stage-one，负值表示更好：

| adv @ step | static endpoint | waypoint | FK | foot | seam | body jerk | body HF | heading HF |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `0 @ 350` | `+1.671%` | `-17.668%` | `-7.332%` | `-4.421%` | `-1.159%` | `-2.916%` | `-11.188%` | `-12.257%` |
| `1e-4 @ 350` | `+1.671%` | `-9.058%` | `-2.359%` | `-5.310%` | `-3.976%` | `-2.554%` | `-9.144%` | `-6.665%` |
| `3e-4 @ 350` | `+1.671%` | `-13.245%` | `-0.343%` | `-5.534%` | `+1.248%` | `-1.576%` | `-6.270%` | `-9.221%` |
| `1e-3 @ 350` | `+1.672%` | `-13.798%` | `+0.873%` | `-2.809%` | `+2.491%` | `-0.432%` | `-10.366%` | `-8.814%` |
| `0 @ 400` | `+1.944%` | `-12.460%` | `+2.127%` | `-3.133%` | `-2.625%` | `-2.165%` | `-8.979%` | `-8.648%` |
| `1e-4 @ 400` | `+1.943%` | `-10.201%` | `+1.332%` | `-3.477%` | `-2.138%` | `+0.559%` | `-9.294%` | `-15.273%` |
| `3e-4 @ 400` | `+1.944%` | `-15.425%` | `-0.307%` | `-2.768%` | `+0.817%` | `-1.977%` | `-13.839%` | `-24.576%` |
| `1e-3 @ 400` | `+1.945%` | `-14.986%` | `-0.000%` | `-5.139%` | `-1.841%` | `-2.436%` | `-9.828%` | `-12.006%` |
| `0 @ 450` | `+2.209%` | `-15.076%` | `-2.862%` | `-2.468%` | `+2.340%` | `-1.319%` | `-15.260%` | `-27.002%` |
| `1e-4 @ 450` | `+2.209%` | `-6.328%` | `+3.071%` | `-1.141%` | `+1.540%` | `+0.170%` | `-7.705%` | `-3.189%` |
| `3e-4 @ 450` | `+2.210%` | `-9.595%` | `+2.609%` | `-2.678%` | `+2.485%` | `-0.330%` | `-5.412%` | `-16.566%` |
| `1e-3 @ 450` | `+2.210%` | `-17.781%` | `-4.374%` | `-5.858%` | `-3.576%` | `-3.671%` | `-13.034%` | `-19.308%` |
| `0 @ 500` | `+2.462%` | `-13.193%` | `-0.811%` | `-4.115%` | `-1.227%` | `-0.502%` | `-10.201%` | `-11.712%` |
| `1e-4 @ 500` | `+2.462%` | `-13.242%` | `+1.429%` | `-2.993%` | `+2.157%` | `-0.087%` | `-14.044%` | `-14.532%` |
| `3e-4 @ 500` | `+2.462%` | `-16.619%` | `-2.832%` | `-4.062%` | `-1.872%` | `-3.903%` | `-14.102%` | `-22.230%` |
| `1e-3 @ 500` | `+2.461%` | `-13.802%` | `-2.757%` | `-0.913%` | `+0.130%` | `+0.892%` | `-14.108%` | `-13.074%` |

降低 adv 并没有形成单调的“漂移和 heading 恢复”曲线。相对当前默认前端，`adv0@g350` 虽改善 waypoint/root-drift-mean/FK/jerk `4.93/5.36/5.47/1.89%`，却使 final drift、body-HF、root-horizontal-HF、heading-HF 回退 `31.13/3.64/3.19/37.72%`；`adv1e4@g350` 可让 final drift/foot/seam/jerk 改善 `2.71/0.24/1.41/1.51%`，但 waypoint/root-drift-mean/body-HF/heading-HF 回退 `4.54/2.75/6.59/46.87%`。`adv3e4@g400` 的 waypoint/jerk 改善 `3.23/1.01%`，heading-HF 回退缩到本轮较低的 `12.97%`，但 final drift 仍回退 `64.93%`。`adv3e4@g500` 改善 waypoint/FK/jerk `3.42/0.81/2.87%`，代价仍是 final drift/foot/seam/body-HF/root-HF/heading-HF 回退 `26.34/1.10/0.76/0.45/1.67/17.29%`。没有任何新点通过默认版的平衡 guardrail，因此不新增或替换前端 release；上一节的独立视觉候选仍保留供用户观察，但不能因本轮静态 endpoint 接近而升级。

这项负结果说明 adversarial weight 与 iteration、随机闭环分岔及全局 generator grad clip 的耦合很强，仅继续细扫 `1e-4..1e-3` 的收益有限。下一步在改 LADD normalization、gradient balancing 或 G/D 框架前，应先测量 DMD、LADD、paired/FSQ 与物理正则在 endpoint 及 generator 参数空间中的实际梯度 RMS、夹角和 clip 前后贡献；否则 loss coefficient 不是可比尺度，继续网格搜索仍难解释。机器汇总位于 `distill_runs/dmd_ladd_u05_multideep_adv_weight_from_g300_joint22500_20260716/multideep_adversarial_weight_summary.json`。审计成功解析 165 份 JSON 和 124 条 JSONL，64/64 rollout、64/64 jitter、16 个 EMA、64 份 fixed case 全部可读且逐元素 finite，日志错误扫描为空，flow-matching 16 项回归测试通过。训练/验证/汇总入口分别为 `scripts/run_dmd_ladd_u05_multideep_adv_weight.sh`、`scripts/evaluate_dmd_ladd_u05_multideep_adv_weight.sh`、`ardy_distill/tools/summarize_dmd_ladd_multideep_adv_weight.py`。全过程没有安装任何包、没有执行 `conda` 命令，也没有修改 `<REFERENCE_3DLM_PROJECT>`。

### 23.30 不改变轨迹的 generator endpoint 梯度分解（2026-07-16）

训练器新增默认关闭的 `--generator-component-gradient-every N` 诊断。它在指定 generator step 使用 `torch.autograd.grad(..., retain_graph=True)`，只读取各 scalar loss 对生成 endpoint 的梯度，不写入参数 `.grad`，随后仍执行原来的单次 distributed backward、global clip、Adam 和 EMA。正式分成四个加权 scalar group：DMD、LADD adversarial、paired+FSQ、其余 control+physics；四组之和与实际 total loss 完全一致。最初将十个 BF16 分量分别求导再相加会产生约 `4–6%` 的独立量化残差，故没有用那组近似值下结论；四组 scalar 直接加权求导后，component-sum 对真实 total endpoint gradient 的相对 RMS 误差降到 `4.22e-8`。

诊断固定上一轮 deep-2 mean-loss `adv=1e-3` 控制配置，从同一个 corrected-u05 g300 full state 重做 S/D `700→800` recovery，再走到 g350，在 g310/320/330/340/g350 五个截面记录。开启诊断的 g350 EMA SHA256 为 `c86aa86915657a5db6cbf234c740440199a1e9d089703d3b3fc81214fcdac506`，与未开启诊断的历史控制 checkpoint 逐字节相同；三次不同诊断实现均保持该 hash，因此诊断没有消耗 RNG、改变 loss、写坏 gradient 或扰动 optimizer 轨迹。17 项 flow-matching/聚合/诊断测试全部通过。

五个截面的 endpoint-space 结果如下。ratio 是各组加权 gradient RMS 除以 total gradient RMS；由于方向可近似正交，ratio 不应机械相加：

| group | raw RMS | weighted RMS | weighted/total RMS | cosine vs total |
|---|---:|---:|---:|---:|
| DMD | `2.812e-5` | `2.812e-5` | `32.02%` | `0.3227` |
| LADD adversarial (`1e-3`) | `3.145e-5` | `3.146e-8` | `0.0361%` | `0.0013` |
| paired + FSQ | `7.391e-6` | `7.391e-6` | `8.42%` | `0.1305` |
| control + physics | `8.253e-5` | `8.253e-5` | `93.85%` | `0.9424` |

总 endpoint gradient RMS 均值为 `8.792e-5`。原始 DMD 与 LADD gradient RMS 同量级，说明 critic 并非完全没有信号；问题在于 `1e-3` 系数把 LADD 实际贡献压到 total 的 `0.036%`。DMD/LADD、DMD/control-physics、LADD/control-physics 的 raw endpoint cosine 分别只有 `0.0077/0.0011/-0.0018`，近似正交；当前 generator 方向主要由 control+physics 决定，DMD 提供第二条近正交方向，而 LADD 在数值上几乎不参与。与此同时，generator parameter pre-clip norm 为 `3.82..5.62`、均值 `4.66`，相对 clip `0.01` 的实际全局缩放只有 `0.00178..0.00261`。endpoint 比例不能替代 parameter-space norm，但足以否定“继续把 adv 从 `1e-3` 往下调会形成更有效 LADD”的假设。

据 raw LADD/DMD RMS 比值，`adv=0.01/0.03/0.1/0.3` 在共同起点约对应 DMD endpoint gradient 的 `1%/3%/11%/34%`，相对 total 约 `0.36%/1.08%/3.61%/10.83%`。所以下一轮应保持所有其他变量不动，扫描这组向上的、由实际梯度校准的 LADD coupling；先以短程 g350/g400/g450/g500 和多 seed 闭环检查稳定性，再决定是否需要显式 gradient normalization，而不是直接引入未经验证的动态 balancing。机器汇总为 `distill_runs/dmd_ladd_u05_multideep_gradient_groups_g350_joint22500_20260716/gradient_group_summary.json`，复现实行脚本为 `scripts/run_dmd_ladd_u05_multideep_gradient_diagnostics.sh`，汇总器为 `ardy_distill/tools/summarize_dmd_ladd_gradient_groups.py`。默认前端与独立 A/B 候选均未改变；全过程没有安装任何包、没有执行 `conda` 命令，也没有修改 `<REFERENCE_3DLM_PROJECT>`。

### 23.31 梯度校准后的高 LADD coupling 矩阵（2026-07-16）

本轮固定 corrected-u05、deep-2 shared-head mean-loss、legacy score adversarial-time、generator/score/D LR `1e-7/5e-6/1e-6`、`G:score:D=1:1:1`、全部非 adversarial loss、clip、EMA、数据和 seed，只将 adversarial weight 向上扫描为 `0.01/0.03/0.1/0.3`。四臂均从相同 g300 full state 恢复并重做 S/D `700→800` recovery，随后到 g500；最终计数全部 `500/1000/1000`，wall time 为 `113.88/114.78/113.32/115.60 s`，四个保存点的 raw/EMA/full state 完整。配置审计确认顶层差异只有 `output` 与 `adversarial_weight`，所有训练和权重 finite。

每个 g350/g400/g450/g500 checkpoint 都完成 16k full、分层 text/control、四 seed 各 50-window 原版无限 rollout 与 jitter。相对同 seed stage-one 的结果如下，负值更好：

| adv @ step | static endpoint | waypoint | FK | foot | seam | body jerk | body HF | heading HF |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `0.01 @ 350` | `+1.671%` | `-10.587%` | `-1.207%` | `-3.225%` | `-0.541%` | `-1.980%` | `-13.637%` | `-12.399%` |
| `0.03 @ 350` | `+1.672%` | `-11.671%` | `-3.283%` | `-3.014%` | `-1.132%` | `-1.701%` | `-4.702%` | `-12.326%` |
| `0.1 @ 350` | `+1.675%` | `-10.096%` | `-10.171%` | `-2.460%` | `-2.956%` | `-1.641%` | `-8.664%` | `-6.532%` |
| `0.3 @ 350` | `+1.682%` | `-7.024%` | `+0.768%` | `-5.100%` | `-0.852%` | `-0.707%` | `-4.971%` | `-25.269%` |
| `0.01 @ 400` | `+1.945%` | `-9.287%` | `+0.163%` | `-2.525%` | `-0.141%` | `+0.073%` | `+0.620%` | `-14.066%` |
| `0.03 @ 400` | `+1.949%` | `-15.981%` | `-2.040%` | `-3.755%` | `+0.391%` | `-2.285%` | `-14.397%` | `-20.368%` |
| `0.1 @ 400` | `+1.958%` | `-12.202%` | `-0.815%` | `-3.163%` | `-1.387%` | `-1.064%` | `-9.494%` | `-12.326%` |
| `0.3 @ 400` | `+1.983%` | `-11.930%` | `+4.090%` | `-2.933%` | `-0.283%` | `-3.320%` | `-12.153%` | `-8.518%` |
| `0.01 @ 450` | `+2.212%` | `-15.172%` | `-3.517%` | `-4.028%` | `-2.721%` | `-2.788%` | `-16.987%` | `-13.597%` |
| `0.03 @ 450` | `+2.218%` | `-13.441%` | `-0.098%` | `-3.674%` | `-0.989%` | `-0.348%` | `-13.776%` | `-8.059%` |
| `0.1 @ 450` | `+2.237%` | `-12.022%` | `-1.189%` | `-3.273%` | `+0.431%` | `-0.825%` | `-9.022%` | `-10.937%` |
| `0.3 @ 450` | `+2.293%` | `-13.684%` | `-6.067%` | `-2.075%` | `-0.329%` | `-0.062%` | `-10.932%` | `-11.532%` |
| `0.01 @ 500` | `+2.466%` | `-11.742%` | `+0.361%` | `-1.464%` | `-0.227%` | `-0.082%` | `-17.195%` | `-17.154%` |
| `0.03 @ 500` | `+2.476%` | `-14.054%` | `-1.399%` | `+0.012%` | `+0.224%` | `+0.681%` | `-17.107%` | `-15.919%` |
| `0.1 @ 500` | `+2.506%` | `-6.802%` | `-1.607%` | `-2.712%` | `-0.333%` | `+0.058%` | `+3.980%` | `-23.033%` |
| `0.3 @ 500` | `+2.594%` | `-12.309%` | `-5.884%` | `-2.995%` | `-4.818%` | `-1.939%` | `-11.524%` | `-7.576%` |

提高系数确实按预期提高了 LADD 数值参与度：g350/g400/g450/g500 四点均值下，`adv=0.01/0.03/0.1/0.3` 的 weighted LADD/total endpoint RMS 分别为 `0.368/1.108/3.652/10.534%`；对应 cosine-vs-total 为 `0.0067/0.0131/0.0381/0.1093`。所以这是有效 coupling sweep，不是假配置。但它仍没有产生可替换默认版的平衡点。相对当前默认前端，最接近的是 `0.01@g450`：waypoint/root-drift-mean/root-drift-final/FK/jerk/body-HF 改善 `2.55/1.15/2.91/1.43/1.81/2.80%`，但 foot/root-horizontal-HF/heading-HF 回退 `1.23/1.74/40.91%`，seam 仅近似持平 `+0.04%`。`0.3@g450` 虽改善 root-mean/final/FK `6.28/31.97/4.18%`，却回退 foot/seam/jerk/body-HF/heading-HF `3.25/2.44/1.08/4.21/43.91%`。其余点也没有同时守住路径、漂移、foot/seam、jerk/HF 与 heading，因此不新增前端 candidate。

诊断同时暴露了当前 legacy adversarial-time 的结构性低效：四臂在保存点的 `adversarial_time_exact_t1_fraction` 均值都是 `71.875%`。对 rectified flow 的 `x_t=(1-t)x_0+tε`，exact `t=1` 时 teacher feature 输入严格等于噪声、与 generator endpoint 无关，因此这些样本对 generator 的 LADD gradient 严格为零；只有约 `28%` batch 样本提供 adversarial gradient，造成稀疏、高方差 coupling。此前 adversarial-time 2×2 是在 `adv=1e-3`、实际 LADD/total 仅约 `0.036%` 时完成，无法排除“matched informative time 在有效 coupling 下更好”。下一项应固定 deep-2 与本轮四个 weight，只把 generator adversarial-time 切到 critic 的 uniform `[0,0.5]`、exact-t1=0，再做严格 matched-time 对照；不能同时改 critic、LR 或物理项。

机器汇总为 `distill_runs/dmd_ladd_u05_multideep_high_adv_from_g300_joint22500_20260716/multideep_adversarial_weight_summary.json`。证据审计为 165 份 JSON、124 条 JSONL、64/64 rollout、64/64 jitter、16 个 EMA、64 份 fixed case 全部可读且 finite，日志错误扫描为空，17 项回归测试通过。训练和验证入口为 `scripts/run_dmd_ladd_u05_multideep_high_adv.sh` 与 `scripts/evaluate_dmd_ladd_u05_multideep_high_adv.sh`；默认 FP16 release 仍为 `text_qwen_dmd_ladd_joint22500_u05legacy_g400_ema_pilot_20260715`。全过程没有安装任何包、没有执行 `conda` 命令，也没有修改 `<REFERENCE_3DLM_PROJECT>`。

### 23.32 有效 coupling 下的 matched informative-time 矩阵与独立 WebGPU A/B（2026-07-16）

上一轮 legacy adversarial-time 有 `71.875%` 的 generator adversarial 样本落在 exact `t=1`，这些样本对 generator endpoint 的 LADD 梯度严格为零。本轮因此不再用 `adv=1e-3` 的近零 coupling 判断 matched-time，而是在完全相同的 high-weight 四臂上只把 generator `adversarial_time_sampler` 从 legacy `score` 改为 `critic`：采样为 uniform `[0,0.5]`、exact-`t=1` 比例为 0。其余 corrected-u05 critic、deep-2 `body_mid+body_final` shared-head mean-loss、generator/score/D LR `1e-7/5e-6/1e-6`、`G:score:D=1:1:1`、DMD/paired/FSQ 与全部控制/物理 loss、global generator clip `0.01`、EMA `0.995`、数据、batch 和 seed 均保持不变。四个 adversarial weight 仍为 `0.01/0.03/0.1/0.3`，全部从同一个 corrected-u05 g300 full state 恢复并重做 S/D `700→800` recovery，再训练到 g500。

配置审计确认 matched 四臂之间的顶层差异只有 `output` 与 `adversarial_weight`；每个 matched arm 与同权重 legacy arm 的实质差异只有 `adversarial_time_sampler: score→critic`，配置中的 algorithm 描述同步记录该变化。最终四臂计数均为 `G/S/D=500/1000/1000`，`metrics.jsonl` 记录的完成 elapsed 分别为 `116.22/115.15/227.54/113.29 s`，全部 `training_complete`、`stopped_by_runtime=false`。训练过程中 `<DATA_VOLUME>` 一度 100% 满，`adv=0.3` 第一次在 g350 写完整 optimizer state 时触发一次 `PytorchStreamWriter file write failed`；这不是 NaN 或训练失稳。仅删除了本轮可重建的损坏 state 和前三臂 g350/g400/g450 的冗余中间 Accelerate state，保留全部 raw/EMA 权重和每臂 g500 完整可恢复 state；随后从同一个 g300 full state 干净重跑 `adv=0.3`，通过原故障点并正常完成。最终训练重跑日志和评估日志均无错误。

matched-time 将实际 LADD coupling 提高了约四倍。g350/g400/g450/g500 四个诊断截面的平均 weighted-LADD/total endpoint-gradient RMS 及 cosine-vs-total 如下；所有 matched arm 的 adversarial exact-`t=1` 均为 `0%`：

| adversarial weight | matched weighted LADD / total RMS | cosine vs total | legacy weighted LADD / total RMS |
|---:|---:|---:|---:|
| `0.01` | `1.653%` | `0.0112` | `0.368%` |
| `0.03` | `4.905%` | `0.0400` | `1.108%` |
| `0.1` | `15.655%` | `0.1411` | `3.652%` |
| `0.3` | `40.756%` | `0.3790` | `10.534%` |

所以这轮不是只改变 loss 日志而没有改变 generator 的假消融；`adv=0.3` 已让 LADD 成为真正有影响力、且与 control/physics 近正交的方向。16 个 g350/g400/g450/g500 checkpoint 均完成相同的 16k full、分层 text/control、seed `20260714..20260717` 四条各 50-window 原版无限 rollout 与 jitter。相对同 seed stage-one 的结果如下，负值表示更好：

| matched adv @ step | static endpoint | waypoint | FK | foot | seam | body jerk | body HF | heading HF |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `0.01 @ 350` | `+1.673%` | `-14.985%` | `+3.856%` | `-1.402%` | `+0.225%` | `-2.751%` | `-8.624%` | `-13.502%` |
| `0.03 @ 350` | `+1.680%` | `-4.440%` | `-5.254%` | `-2.554%` | `-3.430%` | `-0.903%` | `-10.234%` | `-4.384%` |
| `0.1 @ 350` | `+1.696%` | `-0.245%` | `+2.369%` | `-1.473%` | `+1.230%` | `-0.942%` | `-7.868%` | `-7.147%` |
| `0.3 @ 350` | `+1.740%` | `-14.658%` | `+2.120%` | `-3.957%` | `-1.222%` | `-0.738%` | `-9.750%` | `-12.534%` |
| `0.01 @ 400` | `+1.954%` | `-12.663%` | `+0.648%` | `-2.829%` | `+1.200%` | `-1.154%` | `-8.163%` | `-23.479%` |
| `0.03 @ 400` | `+1.974%` | `-13.980%` | `-0.036%` | `-1.772%` | `+0.173%` | `+0.545%` | `-8.227%` | `-18.576%` |
| `0.1 @ 400` | `+2.048%` | `-13.096%` | `-3.038%` | `-2.395%` | `+0.248%` | `-0.176%` | `-9.753%` | `-21.882%` |
| `0.3 @ 400` | `+2.221%` | `-15.278%` | `-2.470%` | `-4.275%` | `-2.444%` | `-0.960%` | `-15.299%` | `-19.612%` |
| `0.01 @ 450` | `+2.230%` | `-8.229%` | `-1.242%` | `-3.462%` | `-1.819%` | `-0.999%` | `-7.474%` | `-14.713%` |
| `0.03 @ 450` | `+2.277%` | `-10.496%` | `-4.666%` | `-3.414%` | `-2.341%` | `+1.804%` | `-13.586%` | `-6.938%` |
| `0.1 @ 450` | `+2.422%` | `-14.222%` | `+2.127%` | `-1.361%` | `-0.857%` | `-0.121%` | `-10.342%` | `-17.135%` |
| `0.3 @ 450` | `+2.783%` | `-8.634%` | `+0.475%` | `+1.784%` | `-2.622%` | `+1.464%` | `-2.054%` | `-14.939%` |
| `0.01 @ 500` | `+2.498%` | `-9.228%` | `+1.538%` | `-2.524%` | `+0.684%` | `+2.536%` | `-12.212%` | `-24.324%` |
| `0.03 @ 500` | `+2.567%` | `+0.570%` | `+1.416%` | `+1.528%` | `+0.281%` | `+3.725%` | `-13.454%` | `-5.706%` |
| `0.1 @ 500` | `+2.804%` | `-17.854%` | `-0.942%` | `-1.837%` | `+0.813%` | `-0.677%` | `-10.026%` | `-18.916%` |
| `0.3 @ 500` | `+3.380%` | `-8.487%` | `+1.973%` | `-0.085%` | `-0.037%` | `+2.549%` | `-15.040%` | `-9.541%` |

matched-time 的收益仍然强烈依赖 weight×iterations，并非单调。最有研究价值的点是 `adv=0.3@g400`：直接相对同权重、同 step 的 legacy-time，它改善 waypoint/root-drift-mean/root-drift-final/FK/foot/seam/body-HF/root-horizontal-HF/heading-HF `3.05/5.49/1.60/6.08/1.29/2.10/2.59/3.03/12.03%`，代价是 static endpoint 与 body jerk 分别回退 `0.23/2.41%`。这证明 informative-time 在有效 coupling 下可以同时改善多个闭环指标，否定了此前 `adv=1e-3` 2×2 对 matched-time 的过早否定。

但 `adv=0.3@g400` 仍不满足默认前端替换门槛。直接相对当前 active `u05legacy-g400`，它改善 waypoint/FK/jerk/body-HF `2.64/0.39/0.03/0.67%`，但 static endpoint、root-mean/final、foot、seam、root-horizontal-HF 分别回退 `0.26/0.55/1.29/0.90/0.27/0.59%`，heading-HF 更回退 `32.62%`。因此默认 `webgpu_toy/infinite_demo/manifest.json` 保持 `text_qwen_dmd_ladd_joint22500_u05legacy_g400_ema_pilot_20260715`，没有把局部 matched-time 胜出包装成全局最优。

为获得真实视觉反馈，`adv=0.3@g400` 已另行导出为只读 A/B candidate：release `text_qwen_dmd_ladd_joint22500_matched_u05_adv3e1_g400_ema_visual_20260716`，候选 URL 为 `http://127.0.0.1:8766/infinite_demo.html?candidate=matched_u05_adv3e1_g400`。页面仍使用原版无限生成逻辑，只切换 candidate manifest；HTTP 已确认 HTML、manifest 和 75.8 MB flow 均返回 `200 OK`。learned FP16 graphs 为 `104,062,612 B`，加 FP32 utilities 后总下载 `104,107,876 B = 99.285 MiB`。encoder/flow/decoder 模块对拍 max abs 为 `0.001290/0.003765/0.007401`，全部 finite。

候选专属 CPU-ORT initial+continuation 整链路 gate 为 `passed=true`：same-precision endpoint max abs 为 `0.007813/0.006348`，FP32 endpoint max abs 为 `0.007177/0.004388`；验收摘要的 FP32 explicit mean/p99/cosine 为 `0.004883/0.030362/0.9999669`，FSQ bin match 为 `98.516%`，均通过既定阈值且没有放宽门槛。真实 Edge WebGPU 数值和用户视觉仍是独立证据，不能用 CPU gate 代替。

机器汇总为 `distill_runs/dmd_ladd_u05_multideep_high_adv_matched_time_from_g300_joint22500_20260716/matched_time_summary.json`；训练/验证/汇总入口为 `scripts/run_dmd_ladd_u05_multideep_high_adv.sh`、`scripts/evaluate_dmd_ladd_u05_multideep_high_adv.sh` 和 `ardy_distill/tools/summarize_dmd_ladd_matched_time.py`。最终审计为 `JSON_OK=165 JSONL_ROWS_OK=124 ROLLOUT_FINITE=64 JITTER_FINITE=64 FLOW_EMA_OK=16 FIXED_CASES_OK=64`，全部逐元素 finite；标准库 `unittest` 的 17 项 flow-matching/critic/multitap/gradient 回归测试全部通过。当前环境没有 `pytest`，没有为此安装任何包；全过程没有执行 `conda` 命令，也没有修改 `<REFERENCE_3DLM_PROJECT>`。

结论仍是实验科学而非 recipe：本轮只建立了“有效 matched-time coupling 可以改善同权重 legacy，但尚未胜过 active”的证据。下一轮应在收到前端视觉反馈后，保持 candidate 框架与验证 seeds 不变，分别扫描 generator LR `1e-8..1e-6`、few-hundred 到 few-thousand iterations，以及独立的 G/score/D 更新频率；不能把 LR、ratio、critic head、gradient normalization 和物理 loss 同时改变。每轮继续报告实际 gradient contribution、静态与条件分层集、多 seed 长闭环、WebGPU 数值和用户视觉，而不是从单个 loss、单个 checkpoint 或单一加权总分宣布最优。

### 23.33 matched high-LADD 框架的 generator LR × iterations 首轮筛选（2026-07-16）

本轮把上一节的 `adv=0.3`、matched informative-time 作为固定框架，只扫描 generator LR 与停止点。所有 arm 都从同一个 corrected-u05 `G=300` 完整 Accelerate state 开始，完整继承 generator/fake-score/critic、三组 Adam 与 scheduler、EMA、sampler、各 rank RNG 和 update counters；统一重复 S/D `700→800` recovery。固定项包括 deep-2 `body_mid+body_final` shared-head `mean_loss`、critic/generator adversarial time uniform `[0,0.5]`、exact-`t=1` 为 0、score/D LR `5e-6/1e-6`、`G:score:D=1:1:1`、DMD/paired/FSQ/控制与物理 loss、generator global clip `0.01`、EMA `0.995`、数据、global batch 128 和 seed。唯一实验变量是 generator LR `1e-8/3e-8/1e-7/3e-7/1e-6`；每条轨迹保存 `G=400/500/600/700` EMA，并只在 `G=700` 保留完整恢复 state 以控制磁盘占用。

矩阵先运行 `1e-7` 复现控制，再进入其他 LR。新控制的 `G=400/G=500` EMA SHA256 分别为 `b4dcce346ae71454926be37a96c38f2c38a2b7fb9b2e58e7efdd8a1281333bcc` 和 `00cd8a1d82d00f83182cdb22428290f091fc2b2cda4d7aca459d8dc415456295`，与上一节 `adv=0.3` matched-time 对应权重逐字节相同；因此数据顺序、随机状态、四卡归约和训练框架确实固定。五条 config 相对控制的顶层差异只有 `output` 和 `generator_learning_rate`。五臂均在物理 GPU `1/2/3/4`、BF16 下正常到 `G=700`，完成事件 elapsed 为 `162.96/162.33/162.49/161.82/161.54 s`，全部 `training_complete`、`stopped_by_runtime=false`，没有 NaN、OOM 或异常退出。

20 个 checkpoint 全部完成同一 16k full、分层 text/control，以及 seed `20260714..20260717` 四条各 50-window 原版无限 rollout 和 jitter，总计 20 组静态、80 条长 rollout、80 份 jitter。下表相对同 seed stage-one，负值表示误差降低；这里不构造人为加权总分：

| generator LR @ step | static endpoint | waypoint | FK | foot | seam | body jerk | body HF | heading HF |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `1e-8 @ 400` | `+1.906%` | `-9.652%` | `+2.435%` | `-1.530%` | `-0.601%` | `-1.656%` | `-8.627%` | `-8.948%` |
| `1e-8 @ 500` | `+2.318%` | `-6.059%` | `-0.907%` | `-0.767%` | `-3.329%` | `+0.337%` | `-10.650%` | `-18.755%` |
| `1e-8 @ 600` | `+2.654%` | `-10.537%` | `+2.348%` | `+0.719%` | `+1.474%` | `+2.866%` | `-7.397%` | `-23.454%` |
| `1e-8 @ 700` | `+2.941%` | `-16.554%` | `+1.812%` | `-2.942%` | `+2.868%` | `+0.104%` | `-7.692%` | `-4.744%` |
| `3e-8 @ 400` | `+1.981%` | `-18.007%` | `+0.842%` | `-2.975%` | `-0.152%` | `-2.078%` | `-11.151%` | `-11.694%` |
| `3e-8 @ 500` | `+2.593%` | `-12.080%` | `-2.736%` | `-0.933%` | `-1.240%` | `-1.143%` | `-9.231%` | `-16.075%` |
| `3e-8 @ 600` | `+3.185%` | `-17.542%` | `+1.297%` | `-0.819%` | `+1.223%` | `+1.599%` | `-8.276%` | `-22.220%` |
| `3e-8 @ 700` | `+3.754%` | `-23.323%` | `-1.605%` | `+0.930%` | `-1.799%` | `+1.012%` | `-11.311%` | `-13.946%` |
| `1e-7 @ 400` | `+2.221%` | `-15.278%` | `-2.470%` | `-4.275%` | `-2.444%` | `-0.960%` | `-15.299%` | `-19.612%` |
| `1e-7 @ 500` | `+3.380%` | `-8.487%` | `+1.973%` | `-0.085%` | `-0.037%` | `+2.549%` | `-15.040%` | `-9.541%` |
| `1e-7 @ 600` | `+4.569%` | `-18.287%` | `-0.990%` | `+4.284%` | `+4.269%` | `+7.422%` | `-5.854%` | `-21.683%` |
| `1e-7 @ 700` | `+5.653%` | `-12.454%` | `-0.435%` | `+3.463%` | `+2.183%` | `+6.373%` | `-9.963%` | `-3.344%` |
| `3e-7 @ 400` | `+2.744%` | `-17.777%` | `+2.100%` | `-2.209%` | `+0.393%` | `-0.707%` | `+22.477%` | `-26.252%` |
| `3e-7 @ 500` | `+4.793%` | `-8.707%` | `-1.199%` | `+1.551%` | `+5.734%` | `+5.780%` | `-14.540%` | `-14.252%` |
| `3e-7 @ 600` | `+6.666%` | `-13.326%` | `+0.396%` | `+3.857%` | `+2.289%` | `+7.473%` | `-16.577%` | `-10.221%` |
| `3e-7 @ 700` | `+8.132%` | `-17.403%` | `-3.782%` | `+4.876%` | `+7.221%` | `+11.286%` | `-25.263%` | `-27.399%` |
| `1e-6 @ 400` | `+3.781%` | `-15.182%` | `-3.850%` | `-0.982%` | `+5.426%` | `+3.502%` | `-17.135%` | `-17.169%` |
| `1e-6 @ 500` | `+6.937%` | `-15.817%` | `-5.163%` | `+2.872%` | `+5.931%` | `+8.077%` | `-21.012%` | `-23.632%` |
| `1e-6 @ 600` | `+9.373%` | `-8.211%` | `-3.960%` | `+5.556%` | `+6.777%` | `+15.173%` | `-29.145%` | `-7.076%` |
| `1e-6 @ 700` | `+11.041%` | `-16.243%` | `-4.709%` | `+7.915%` | `+9.535%` | `+17.919%` | `-28.698%` | `-13.309%` |

结果再次证明 LR 与 iterations 是强非单调耦合。较低 LR 可以在继续压 waypoint 时避免高 LR 的快速静态偏离，但没有自动恢复 foot/seam/jerk/heading；`1e-6` 则明显把更低 FK/body-HF 换成越来越差的 endpoint、foot、seam 和 jerk，当前无需机械延长。最强 waypoint 点 `3e-8@g700` 直接相对 active 改善 `10.14%`，但 endpoint、root-mean、FK、foot、seam、jerk、body-HF、root-HF、heading-HF 分别回退 `1.76/0.90/0.48/6.48/0.76/2.01/3.98/8.18/37.18%`。`1e-6@g400` 虽直接改善 active 的 root-mean/final、FK 和 body-HF `1.30/14.20/1.81/3.40%`，却回退 foot/seam/jerk/heading-HF `4.26/7.95/4.57/30.76%`。这两类都不是适合用户所强调“不要抖动和噪声”的默认候选。

因此本轮 19 个新点没有一个通过 active 或 matched-control 的平衡 guardrail，不为追求表面上的“更新”而替换默认权重。上一节的 `1e-7@g400` 仍保留为独立视觉 A/B，而不是被宣布为唯一最优；它也是本轮逐字节复现的控制点。候选 URL 继续为 `http://127.0.0.1:8766/infinite_demo.html?candidate=matched_u05_adv3e1_g400`，重新检查 HTML、candidate manifest 和 75.8 MB flow 均为 HTTP `200`；默认 `manifest.json` 未改变。

机器汇总为 `distill_runs/dmd_ladd_matched_adv03_generator_lr_to700_joint22500_20260716/generator_lr_iters_summary.json`。训练与评估入口为 `scripts/run_dmd_ladd_matched_adv03_generator_lr_screen.sh` 和 `scripts/evaluate_dmd_ladd_matched_adv03_generator_lr_screen.sh`；通用汇总器修正了“所有 arm 都从 g300 开始”时的 provenance 说明。最终审计为 `JSON=206`、`JSONL=5/255 rows`、`flow EMA=20`、`rollout fixed cases=80` 全部可读且逐 tensor finite，80/80 rollout 与 80/80 jitter 齐全；17 项 flow-matching/critic/multitap/gradient 回归测试通过。没有安装任何包、没有执行 `conda` 命令，也没有修改 `<REFERENCE_3DLM_PROJECT>`。

下一项只改变 D 更新频率，检验当前约 40% endpoint-space LADD coupling 在 discriminator 多更新时是否更有效或反而过强；score 频率、三组 LR、matched timestep、head、loss、clip、数据和 seed 保持不变。先从同一 g300 full state比较 `G:score:D=1:1:1/1:1:2/1:1:4` 的 g400/g500，复用已经逐字节验证的 `1:1:1` 控制，仍以静态、多 seed 长闭环、真实前端视觉共同判断，而不从 D loss 单独下结论。

### 23.34 matched high-LADD 的独立 discriminator 更新频率矩阵（2026-07-16）

本轮严格继承 23.33 的共同 `G=300` full state、`adv=0.3`、matched uniform `[0,0.5]` adversarial-time、deep-2 shared-head mean-loss、generator/score/D LR `1e-7/5e-6/1e-6`、score ratio 1、全部 loss/clip/EMA/数据/batch/seed，只将未来的 `critic_updates_per_generator` 设为 `1/2/4`。共同 S/D warmup 仍从 `700→800`，ratio origin 在 warmup 完成后的 loaded counters 锚定，不对历史 300 个 generator update 做追补。三臂最终计数实测为 `G/S/D=500/1000/1000、500/1000/1200、500/1000/1600`，证明调度真实生效；完成 elapsed 为 `97.50/111.68/138.80 s`，均正常退出且全部 finite。

三条 config 归一化掉 `output`、`critic_updates_per_generator` 及 algorithm provenance 中同一 ratio 镜像后完全相同。`D=1` 控制在 g400/g500 的 EMA SHA256 再次为 `b4dcce346ae71454926be37a96c38f2c38a2b7fb9b2e58e7efdd8a1281333bcc` 与 `00cd8a1d82d00f83182cdb22428290f091fc2b2cda4d7aca459d8dc415456295`，和 23.33 控制逐字节一致。因此 D2/D4 的差异可归因到未来 D 更新频率，而不是初始化、RNG 或调度重构。

三臂的 g350/g400/g450/g500 全部完成 16k full、分层 text/control、四 seed 各 50-window 无限 rollout 与 jitter。下表相对同 seed stage-one，负值表示更好：

| G:score:D @ step | static endpoint | waypoint | root mean | root final | FK | foot | seam | jerk | body HF | root HF | heading HF |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `1:1:1 @ 350` | `+1.740%` | `-14.658%` | `-0.460%` | `+22.344%` | `+2.120%` | `-3.957%` | `-1.222%` | `-0.738%` | `-9.750%` | `-4.657%` | `-12.534%` |
| `1:1:2 @ 350` | `+1.750%` | `-6.565%` | `+1.851%` | `+14.808%` | `-0.839%` | `-0.665%` | `-3.398%` | `+1.490%` | `-7.422%` | `-2.407%` | `-15.024%` |
| `1:1:4 @ 350` | `+1.744%` | `-10.796%` | `-4.512%` | `+4.560%` | `-7.427%` | `-3.582%` | `-6.128%` | `-1.729%` | `-7.876%` | `-1.243%` | `-13.316%` |
| `1:1:1 @ 400` | `+2.221%` | `-15.278%` | `-2.384%` | `+0.533%` | `-2.470%` | `-4.275%` | `-2.444%` | `-0.960%` | `-15.299%` | `-4.935%` | `-19.612%` |
| `1:1:2 @ 400` | `+2.248%` | `-8.293%` | `-2.957%` | `-14.496%` | `-7.110%` | `-3.261%` | `-3.125%` | `-1.886%` | `-11.145%` | `-3.323%` | `-13.505%` |
| `1:1:4 @ 400` | `+2.245%` | `-16.981%` | `+0.425%` | `-19.601%` | `-0.224%` | `-2.648%` | `-4.248%` | `-1.743%` | `-9.325%` | `-3.778%` | `-15.839%` |
| `1:1:1 @ 450` | `+2.783%` | `-8.634%` | `+0.194%` | `+28.670%` | `+0.475%` | `+1.784%` | `-2.622%` | `+1.464%` | `-2.054%` | `-2.753%` | `-14.939%` |
| `1:1:2 @ 450` | `+2.820%` | `-9.152%` | `-1.938%` | `+3.319%` | `+0.191%` | `-0.241%` | `+1.871%` | `+0.923%` | `-10.084%` | `+0.891%` | `-6.913%` |
| `1:1:4 @ 450` | `+2.834%` | `-15.014%` | `-0.949%` | `+2.596%` | `-2.106%` | `+1.525%` | `+3.654%` | `+1.541%` | `-10.349%` | `-2.247%` | `-17.894%` |
| `1:1:1 @ 500` | `+3.380%` | `-8.487%` | `+0.097%` | `+13.908%` | `+1.973%` | `-0.085%` | `-0.037%` | `+2.549%` | `-15.040%` | `-0.741%` | `-9.541%` |
| `1:1:2 @ 500` | `+3.422%` | `-4.548%` | `-0.266%` | `+10.921%` | `+3.029%` | `+0.369%` | `+2.321%` | `+5.583%` | `-12.596%` | `+3.122%` | `-3.311%` |
| `1:1:4 @ 500` | `+3.467%` | `-17.574%` | `-3.592%` | `+0.425%` | `-4.491%` | `+0.485%` | `+0.784%` | `+5.200%` | `-18.171%` | `+0.062%` | `-18.065%` |

结果是明显的 ratio×iterations Pareto，而非 D 越多越好。相对同 step D1，`D4@g350` 改善 root-mean/final、FK、seam、jerk `3.92/5.00/9.12/5.02/1.02%`，但 waypoint/foot/body-HF/root-HF 回退 `5.34/0.41/2.81/3.62%`；seam 为 `4/4` seed 改善，FK 为 `3/4`。`D2@g400` 改善 root-mean/final、FK、seam、jerk `0.46/12.47/4.67/0.71/0.94%`，但 waypoint/foot/body-HF/root-HF/heading-HF 回退 `8.49/1.02/5.27/1.84/8.34%`。`D4@g400` 改善 waypoint/final-drift/seam/jerk `1.82/10.85/1.81/0.72%`，但 root-mean/FK/foot/body-HF/root-HF/heading-HF 回退 `2.99/2.31/1.73/6.71/1.27/6.33%`。`D4@g500` 又切换为 waypoint/root/FK/body-HF 更好，却使 foot/seam/jerk 回退 `0.59/0.87/2.61%`。没有单一频率或更多 iterations 全面占优。

相对当前 active 的前端 guardrail，D2/D4 仍未解决 heading 高频。`D4@g400` 虽改善 active 的 waypoint/seam/jerk `4.52/1.56/0.71%`，heading-HF、body-HF、root-HF、foot 与 FK 却回退 `36.88/5.83/1.79/2.64/1.92%`；heading 为 `0/4` seed 胜出。`D4@g500` 改善 waypoint/root-mean/final/FK/body-HF `4.23/0.51/3.09/2.45/3.73%`，但 foot/seam/jerk/root-HF/heading-HF 回退 `5.99/3.51/6.33/5.99/26.76%`。因此不新增前端 candidate，已有 matched `D1@g400` A/B 与默认 active 都保持不变。

机制指标说明更多 D update 的确改变了判别轨迹，但分类 loss 也不与动作质量单调对应。g500 的 critic total 为 `D1/D2/D4 = 1.3741/1.3744/1.3336`；D4 的 real/fake logit 为 `0.0408/-0.0996`，比 D1 的 `0.0743/+0.0193` 更可分。对应 generator weighted-LADD/total endpoint-gradient RMS 为 `35.22/44.46/47.16%`，cosine-vs-total 为 `0.325/0.411/0.449`。D4 确实给 generator 更强、更一致的 LADD 方向，但这同时放大了某些 seam/jerk/heading trade-off，不能把更低 D loss 当作成功。

机器汇总为 `distill_runs/dmd_ladd_matched_adv03_critic_ratio_from_g300_joint22500_20260716/critic_ratio_summary.json`。训练/验证入口为 `scripts/run_dmd_ladd_matched_adv03_critic_ratio.sh` 和 `scripts/evaluate_dmd_ladd_matched_adv03_critic_ratio.sh`；通用 ratio 汇总器的 provenance 文案已改为不绑定旧低-weight 实验。最终审计为 `JSON=124`、`JSONL=3/93 rows`、12 个 flow EMA、48 份 fixed cases、48/48 rollout 和 48/48 jitter 全部可读且逐 tensor finite。没有安装任何包、没有执行 `conda` 命令，也没有修改 `<REFERENCE_3DLM_PROJECT>`。

下一轮固定 `G:score:D=1:1:4`，只扫描 critic LR `1e-8/3e-8/1e-7/3e-7/1e-6`。目的不是假设更低 LR 必然好，而是检验“多次小 D step”能否保留 D4 的 FK/seam/路径收益，同时减轻当前约 47% LADD coupling 对 foot/jerk/heading 的副作用；generator/score LR、adv weight、matched time、head、数据、loss 和 seeds 均保持不动。

### 23.35 固定 D4 的 critic-LR × iterations 矩阵（2026-07-16）

本轮严格固定 23.34 的共同 corrected-u05 G300 full state、generator/score LR `1e-7/5e-6`、`G:score:D=1:1:4`、`adv=0.3`、matched uniform `[0,0.5]` adversarial-time、deep-2 `body_mid+body_final` shared-head mean-loss、DMD/paired/FSQ/控制与物理 loss、clip、EMA、数据、batch 与 seed，仅扫描 critic LR `1e-8/3e-8/1e-7/3e-7/1e-6`。五臂都完成 G350/G400/G450/G500，最终计数均为 `G/S/D=500/1000/1600`；wall time 分别为 `139.37/138.79/139.64/140.18/139.18 s`，全部正常退出。`1e-6` 控制的 G400/G500 EMA SHA256 为 `1b38dcf9...ae7b` 和 `fde1ef91...448e`，与 23.34 的 D4 对应权重逐字节一致，证明新矩阵没有改变控制轨迹。

20 个 checkpoint 均完成 16k full、分层 text/control，以及 seed `20260714..20260717` 四条各 50-window 原版无限 rollout 与 jitter。下表相对同 seed Stage1，负值表示误差下降：

| critic LR @ step | static endpoint | waypoint | root mean | root final | FK | foot | seam | jerk | body HF | root HF | heading HF |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `1e-08 @ 350` | `+1.747%` | `-12.385%` | `-8.955%` | `-14.431%` | `-4.246%` | `-5.842%` | `+0.298%` | `-1.597%` | `-11.474%` | `-0.801%` | `-9.344%` |
| `3e-08 @ 350` | `+1.745%` | `-10.428%` | `-3.332%` | `-2.983%` | `-0.350%` | `-5.248%` | `-0.784%` | `-2.690%` | `-11.018%` | `-1.190%` | `-16.753%` |
| `1e-07 @ 350` | `+1.743%` | `-5.053%` | `-1.029%` | `+33.447%` | `-2.651%` | `-4.117%` | `-1.765%` | `-1.859%` | `-5.897%` | `-1.331%` | `-3.525%` |
| `3e-07 @ 350` | `+1.739%` | `-11.842%` | `-5.283%` | `+7.220%` | `-1.387%` | `-5.978%` | `+0.033%` | `-2.190%` | `-8.653%` | `-0.434%` | `-10.386%` |
| `1e-06 @ 350` | `+1.744%` | `-10.796%` | `-4.512%` | `+4.560%` | `-7.427%` | `-3.582%` | `-6.128%` | `-1.729%` | `-7.876%` | `-1.243%` | `-13.316%` |
| `1e-08 @ 400` | `+2.247%` | `-12.475%` | `-1.686%` | `+4.879%` | `+0.077%` | `-3.782%` | `+1.674%` | `-0.660%` | `-5.468%` | `-3.102%` | `-20.500%` |
| `3e-08 @ 400` | `+2.241%` | `-10.757%` | `-5.970%` | `-37.054%` | `-2.450%` | `-2.312%` | `-0.981%` | `+0.001%` | `-10.534%` | `-2.335%` | `-9.942%` |
| `1e-07 @ 400` | `+2.226%` | `-12.537%` | `-0.791%` | `-5.831%` | `+1.217%` | `-1.118%` | `+0.156%` | `-0.523%` | `-12.171%` | `-2.392%` | `-20.167%` |
| `3e-07 @ 400` | `+2.219%` | `-8.406%` | `+1.237%` | `-8.658%` | `+0.545%` | `-4.523%` | `-0.972%` | `+0.295%` | `-13.708%` | `-4.591%` | `-15.611%` |
| `1e-06 @ 400` | `+2.245%` | `-16.981%` | `+0.425%` | `-19.601%` | `-0.224%` | `-2.648%` | `-4.248%` | `-1.743%` | `-9.325%` | `-3.778%` | `-15.839%` |
| `1e-08 @ 450` | `+2.830%` | `-11.184%` | `-0.655%` | `-15.770%` | `-1.369%` | `+0.442%` | `+0.204%` | `+2.611%` | `-13.402%` | `-3.517%` | `-15.690%` |
| `3e-08 @ 450` | `+2.813%` | `-14.139%` | `-0.931%` | `+20.820%` | `+0.563%` | `-2.835%` | `+0.861%` | `+1.140%` | `-8.215%` | `-2.829%` | `-15.843%` |
| `1e-07 @ 450` | `+2.778%` | `-11.514%` | `-0.520%` | `+7.911%` | `+0.142%` | `-2.398%` | `-0.487%` | `+2.622%` | `-15.011%` | `-5.029%` | `-4.205%` |
| `3e-07 @ 450` | `+2.767%` | `-5.580%` | `+3.443%` | `-35.532%` | `+0.207%` | `+0.289%` | `+1.017%` | `+3.776%` | `-10.310%` | `-3.023%` | `-21.139%` |
| `1e-06 @ 450` | `+2.834%` | `-15.014%` | `-0.949%` | `+2.596%` | `-2.106%` | `+1.525%` | `+3.654%` | `+1.541%` | `-10.349%` | `-2.247%` | `-17.894%` |
| `1e-08 @ 500` | `+3.449%` | `-14.580%` | `-0.828%` | `+33.604%` | `+0.053%` | `-0.163%` | `+0.375%` | `+2.348%` | `-12.559%` | `-2.974%` | `-17.274%` |
| `3e-08 @ 500` | `+3.414%` | `-10.742%` | `+5.152%` | `+3.381%` | `+3.790%` | `-0.049%` | `+0.680%` | `+2.203%` | `-10.423%` | `-2.517%` | `-8.625%` |
| `1e-07 @ 500` | `+3.351%` | `-11.195%` | `-1.375%` | `-11.905%` | `+0.249%` | `+0.507%` | `+0.463%` | `+5.563%` | `-9.087%` | `+0.849%` | `-9.422%` |
| `3e-07 @ 500` | `+3.339%` | `-9.806%` | `-3.787%` | `+12.333%` | `-2.603%` | `+1.577%` | `+0.312%` | `+3.438%` | `-13.215%` | `-2.312%` | `-11.199%` |
| `1e-06 @ 500` | `+3.467%` | `-17.574%` | `-3.592%` | `+0.425%` | `-4.491%` | `+0.485%` | `+0.784%` | `+5.200%` | `-18.171%` | `+0.062%` | `-18.065%` |

低 critic LR 改变了 D 的优化状态，但没有形成平衡更好的闭环轨迹。`1e-8@g350` 相对同 step `1e-6` 控制改善 waypoint/root-mean/root-final/foot/body-HF `0.79/4.46/18.98/2.21/2.78%`，代价是 FK/seam/root-HF/heading-HF 回退 `3.89/7.15/0.48/6.63%`，其中 seam 仅 `1/4` seed 胜出。`3e-8@g400` 的 root-mean/root-final/FK/body-HF 相对控制改善 `6.30/10.54/2.06/1.26%`，但 waypoint/foot/seam/jerk/root-HF/heading-HF 回退 `7.69/0.35/3.47/1.78/1.64/7.14%`，seam 为 `0/4`。最低 active-heading 回退点是 `3e-7@g450`，仍相对当前 active 回退 `27.33%`，同时 waypoint/root-mean/FK/foot/seam/jerk/body-HF 分别回退 `9.02/6.58/2.47/5.78/3.90/5.08/6.32%`。全 20 个点相对 active 的 heading-HF 回退范围为 `+27.33%..+55.15%`，所以没有新前端候选；默认与已有 matched A/B 均不替换。

机制上，降低 critic LR 确实单调削弱了分类器：G350/G400/G450/G500 四截面的 critic-total 均值由 `1e-6` 的 `1.3491` 上升至 `1e-8` 的 `1.3988`，real-minus-fake logit margin 由 `0.1157` 降到 `0.0488`。但它没有按同样比例降低 generator 的实际 LADD coupling：五档 LR 的 weighted-LADD/total endpoint-gradient RMS 均值依次为 `41.59/40.42/38.72/38.84/44.33%`，cosine-vs-total 为 `0.389/0.377/0.362/0.363/0.417`。这是因为所有 arm 继承了同一已经训练的 G300 head，且 head 对 endpoint 的 raw gradient 仍大；“更小 D LR”不等价于“更小 adversarial coefficient”。分类 margin、GAN loss 和动作质量也再次表现为非单调关系。

机器汇总位于 `distill_runs/dmd_ladd_matched_adv03_d4_critic_lr_from_g300_joint22500_20260716/critic_lr_summary.json`，训练与验证入口为 `scripts/run_dmd_ladd_matched_adv03_d4_critic_lr.sh` 和 `scripts/evaluate_dmd_ladd_matched_adv03_d4_critic_lr.sh`。清理前的完整审计为 `JSON=206`、`JSONL=5/155 rows`、20 个 EMA、80 份 fixed cases、80/80 rollout、80/80 jitter 全部逐元素 finite；17 项 flow-matching/critic/multitap/gradient `unittest` 通过。随后按 23.36 的保留策略删除可重建 raw/fixed/state，不影响已落盘汇总与逐 seed metrics。

下一框架变量选择“每个 feature tap 使用独立 discriminator head”，不是继续机械扫 LR。LADD 原论文明确在冻结 diffusion teacher 的每个 attention-block token sequence 上放置独立 head，而当前实现为了 checkpoint 兼容让 `body_mid/body_final` 共用一套 head；这不是等价实现。第一轮应维持相同 taps、mean-loss、时间分布、loss、LR 与 seeds，用从 G300 shared head 逐字节复制出的两个独立 head 做 function-preserving 初始化，并与已有 shared D1/D4 控制作 factorial 对拍；暂不同时改变 head 容量、正则或 feature taps。

### 23.36 存储清理与可恢复锚点（2026-07-16）

用户指出实验目录占用过大后，先暂停新增矩阵并完成两阶段清理。工作区由约 `333G` 降至 `52G`，`distill_runs` 由约 `287G` 降至 `5.9G`，共释放约 `281G`。保留的 `39G distill_data` 是后续大规模全分布训练所需数据，`5.7G webgpu_toy` 包含当前默认/A-B release、数值用例和浏览器服务，均未因清理中断。

删除项包括：1058 份约 `10.59 GiB` rollout fixed case、约 `48.05 GiB` 非 EMA flow、约 `35.97 GiB` fake-score 导出、约 `4.65 GiB` critic 导出、绝大多数 completed sweep optimizer state，以及旧 codec raw/非入选 EMA、早期 teacher/on-policy 临时 shard。所有矩阵的 config、训练 JSONL、静态评估、逐 seed metrics/jitter、汇总和日志均保留。

仅保留四个完整可续训 state：监督主干 step40000、所有后续实验共同的 corrected-u05 G300、当前 active G400、matched D4 G500。另保留 codec encoder/decoder EMA、Stage1 G22500/监督 G40000、corrected-u05 G300、active G400、matched A/B G400、D4 G400/G500，共 9 份 EMA。四个 state 均检查必需 model/optimizer/scheduler，9 份 EMA 均逐 tensor finite；默认页面、matched A/B 页面与 candidate manifest 清理后仍为 HTTP 200。精确路径、字节数与 SHA256 见 `存储清理记录_20260716.md`。全过程没有安装任何包、没有执行 `conda` 命令，也没有修改 `<REFERENCE_3DLM_PROJECT>`。

### 23.37 LADD 每个 feature tap 独立判别头对拍与即时收口（2026-07-16）

本轮只改变 LADD 判别头的参数共享方式，验证 23.35 提出的结构变量。新增 `IndependentScoreBackboneCriticHeads`：`body_mid` 与 `body_final` 各自使用一套独立的二层 critic head，参数量由共享头约 `4.73M` 增至两头合计 `9,453,570`；tap、teacher feature、mean-loss 聚合、matched uniform `[0,0.5]` 时间、generator/score/D LR `1e-7/5e-6/1e-6`、`adv=0.3`、所有辅助 loss、clip、EMA、数据、batch 和 seed 均不变。为避免重新随机初始化，转换工具从 corrected-u05 G300 的共享 critic 逐 tensor 克隆到两个独立 head，同时复制对应 Adam moments；单测验证转换前后两个 tap 的初始 logits 与 optimizer state 精确一致。

只比较 `G:score:D=1:1:1` 与 `1:1:4`，分别记为 D1/D4。两组都从同一个 function-preserving G300 state 恢复，最终计数为 `G/S/D=500/1000/1000` 和 `500/1000/1600`，完成用时 `88.23/130.14 s`，均 `training_complete`、`stopped_by_runtime=false`。8 个 G350/G400/G450/G500 EMA 全部完成 16k full、分层 text/control，以及 seed `20260714..20260717` 四条各 50-window 无限 rollout 与 jitter。相对同 seed Stage1 的结果如下，负值表示更好：

| 独立头 ratio @ step | static endpoint | waypoint | FK | foot | seam | body jerk | body HF | heading HF |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `D1 @ 350` | `+1.742%` | `-11.078%` | `+2.447%` | `-3.385%` | `-2.842%` | `-0.711%` | `-5.942%` | `-22.106%` |
| `D4 @ 350` | `+1.747%` | `-13.196%` | `-2.738%` | `-3.206%` | `-2.337%` | `-3.196%` | `-11.278%` | `-13.529%` |
| `D1 @ 400` | `+2.236%` | `-11.741%` | `-5.154%` | `-2.936%` | `-4.592%` | `-1.568%` | `-9.093%` | `-17.996%` |
| `D4 @ 400` | `+2.258%` | `-10.342%` | `+0.139%` | `-1.484%` | `-0.711%` | `+3.075%` | `-13.897%` | `-7.620%` |
| `D1 @ 450` | `+2.816%` | `-12.614%` | `+1.694%` | `+0.364%` | `+0.602%` | `+1.499%` | `-1.939%` | `-23.078%` |
| `D4 @ 450` | `+2.859%` | `-4.325%` | `+1.279%` | `-1.572%` | `+1.396%` | `-1.529%` | `-8.797%` | `-8.045%` |
| `D1 @ 500` | `+3.439%` | `-7.566%` | `+2.339%` | `-2.061%` | `+1.134%` | `+1.697%` | `-10.912%` | `-6.371%` |
| `D4 @ 500` | `+3.501%` | `-19.410%` | `-1.369%` | `+0.039%` | `+0.452%` | `+2.709%` | `-13.393%` | `-14.868%` |

独立 head 并未形成优于共享 head 的稳定方向。同 ratio、同 step 逐种子比较中，它有时改善 seam、jerk 或 FK，但常同时回退 waypoint、foot 或 HF；例如 D1@g400 相对共享 D1 改善 FK/seam/jerk `2.67/2.13/0.53%`，却回退 waypoint/foot/body-HF/heading-HF `4.38/1.47/7.88/1.04%`。D4@g400 相对共享 D4 的 body-HF 改善 `4.09%`，但 waypoint/seam/jerk/heading-HF 回退 `8.25/3.80/4.90/10.03%`。因此“按论文拆成独立 heads”在本任务与当前 teacher features 上不是自动收益。

相对当前默认前端 `u05_legacy_advtime@g400`，8 个点全部未通过抗抖动守门：heading-HF 回退范围为 `+20.26%..+52.12%`，body-HF 多数回退，foot 也全部按四 seed 均值回退。最接近平衡的 D4@g350 虽改善 waypoint/FK/jerk `0.13/0.65/2.19%`，仍回退 foot/seam/body-HF/heading-HF `2.04/0.13/3.39/41.17%`；所以没有导出 WebGPU candidate，也没有替换默认或已有 matched A/B。

清理前审计为 83 份 JSON、2 份 JSONL/62 rows、8 个 EMA、32 条 rollout 与 32 份 jitter 全部 finite，19 项 flow-matching/critic/独立-head/optimizer-conversion 回归测试通过。机器汇总保留在 `distill_runs/dmd_ladd_matched_adv03_independent_heads_from_g300_joint22500_20260716/independent_heads_summary.json`。由于 8 个点全部落选，随即删除两份完整 state、8 个 EMA、32 份 `fixed_cases.safetensors` 和可重建的转换 state，只保留 config、训练指标、静态评估、逐 seed rollout/jitter、汇总、日志及 4.7 KB 转换 manifest；实验目录由 `3.4G` 收缩为 `16M`。四个有用续训锚点与 9 份保留 EMA 完全不变。

本轮说明下一步不应继续在相同 high-LADD 框架上机械增加判别容量。更合理的是回到数据与 teacher/student residual 的可解释诊断：按控制类型、prompt、速度、转向强度和 rollout 时长定位 heading 高频来源，再决定是调整 feature/time/loss 还是补充监督数据。全过程没有安装任何包、没有执行 `conda` 命令，也没有修改 `<REFERENCE_3DLM_PROJECT>`。

### 23.38 WebGPU infinite demo 显示后处理 A/B（2026-07-17）

针对当前线上 FP16 NFE=1 infinite demo 的根路径抖动、动作高频噪声和 20 FPS sample-and-hold 观感，网页新增七档可即时切换的显示后处理：`raw` 原始对照、`interp` 仅显示插帧、`seam` 仅 8 帧接缝惯性化、`root` 接缝加根轨迹低通、`pose` 接缝加关节旋转低通、`balanced` 温和根/动作低通加插帧、`strong` 两遍强平滑加插帧。默认档改为 `balanced`；URL 也可显式使用 `?post=raw|interp|seam|root|pose|balanced|strong`。

实现严格分离模型状态与显示状态：`motionFrames` 始终保存未经后处理的 330 维模型输出，并且只有它进入下一轮 history encoder；`displayMotionFrames` 与 FK joint 仅用于轨迹、相机和骨架绘制，不回灌自回归。因此后处理不会把自身误差积累进无限续写。切换模式也不再重新运行模型、跳回 frame 0 或改变噪声，而是在同一条 raw 序列、同一 frame index 上重建显示副本，适合直接视觉 A/B。每次生成段的真实边界和稀疏约束帧单独记录，重建时仍能恢复接缝处理和 waypoint 保护。

三点时域低通只作用于 5 维 global root 和 27×6D 旋转区间；接缝惯性化作用于除末尾四个接触/离散特征外的连续特征，四个 tail features 始终逐元素保持原值。低通核为廉价的 `[1,2,1]/4` 并按强度与原值混合；接缝用前两帧速度外推得到偏移并以 smoothstep 衰减。根轨迹首尾使用分段线性 correction 保持，落在前 8–10 帧接缝窗口内的 sparse waypoint x/z 也明确锚回 raw 模型值。所有 root 编辑完成后在物理 cos/sin 空间重新归一化 heading，修复了接缝外推可能离开单位圆的问题。20 FPS 到浏览器刷新率的显示插帧不是直接 lerp joint xyz，而是先插值 6D rotation、Gram–Schmidt 正交化，再做一次 27-joint FK，因而保持骨长。

真实 `browser_validation_case.json` 的 continuation 40×330 动作上，root 二阶差分 RMS 从 raw `0.04349` 降为 balanced `0.03234`、strong `0.01855`，即下降约 `25.6%/57.3%`；27×6D rotation 二阶差分 RMS 从 `0.16919` 降为 `0.14110/0.10504`，下降约 `16.6%/37.9%`。接缝首帧相对历史速度外推的 normalized RMS，root/rotation 约从 `0.0121/0.2336` 降至 `0.0030/0.0584`。processed modes 的 heading 单位圆最大误差小于 `4.3e-8`，接缝窗口内 protected frame 2 的 x/z 与 raw 逐元素相等，段尾 root 偏差为零；输入、raw 输出与四个 tail features 均未被修改。

本机 Node 真实尺寸暖机后每 40 帧均值为 raw `0.1785 ms`、balanced `0.2635 ms`、strong `0.2925 ms`，即 balanced/strong 相对 raw 只增加约 `0.085/0.114 ms`。真实 Edge 149 + Ampere WebGPU 首段成功且全 finite：首次编译轮 balanced display filter `2.045 ms`、FK `2.525 ms`，同期 flow `701.19 ms`、decoder session `122.26 ms`。同一 40 帧 raw 序列即时切到 strong 的浏览器实测为 filter `0.990 ms`、FK `1.650 ms`、总重建 `3.490 ms`；切回 raw 总重建 `1.075 ms`。随后真实跑了一次 continuation waypoint，`generation_count=2`、总序列 44 帧且全 finite，目标帧 raw/display root 坐标逐值相同；该续写段 balanced filter/FK 为 `1.560/0.155 ms`。包含两个真实段边界的 44 帧序列再切 strong 也通过，总重建 `3.385 ms`。这不是持续每帧成本：滤波/FK 只在新段或切换模式时运行；正常 RAF 仅在 `interp/balanced/strong` 做单帧 27-joint FK。

回归入口为 `node --experimental-default-type=module webgpu_toy/infinite_demo/postprocess.test.mjs` 与 `node --experimental-default-type=module webgpu_toy/infinite_demo/postprocess.real.test.mjs`，两者均通过；`demo.js/postprocess.js` ESM 语法检查与 `compare_server.py` Python 编译检查也通过。8766 服务已升级到 frontend protocol 4，支持服务器端远程切换模式；当前 Edge 客户端已经实际完成 `balanced -> strong -> raw` 的同序列切换且 job 全部 passed，最终会留在推荐的 balanced。页面为 `http://127.0.0.1:8766/infinite_demo.html`。全过程没有安装任何包、没有执行 `conda` 命令，也没有修改 `<REFERENCE_3DLM_PROJECT>`。

### 23.39 WebGPU 局域网 HTTPS 入口（2026-07-17）

直接访问 `http://<SERVER_IP>:8766` 时 WebGPU 不可用并非模型或前端故障，而是 Secure Context 规则：WebGPU 的 `navigator.gpu` 只暴露给安全上下文；HTTP 的 `127.0.0.0/8`、`::1` 与 localhost 名称有开发特例，普通 HTTP 私网 IP 没有。127.0.0.1 始终指向浏览器所在机器，SSH 本地转发有效是因为浏览器看到的 origin 仍是 loopback。

机器上没有现成 Nginx/Caddy、域名或证书，因此保留 HTTP 8766，并用 Python 标准库为同一站点新增独立 HTTPS 8767。创建了受保护的本地测试 CA 和带 `IP:<SERVER_IP>` SAN 的服务器叶子证书；CA/server 私钥均位于静态根目录之外且权限 600，只有公开 CA 证书复制到静态站点供受控测试机下载。`compare_server.py` 新增可选 `--certfile/--keyfile`，TLS 最低版本为 1.2；当前实际协商 TLS 1.3 `TLS_AES_256_GCM_SHA384`，CA 链与 IP 校验 `Verify return code: 0`。HTTPS 预检页、infinite demo、公开 CA 和 motion-stats API 均实测 HTTP 200。

局域网/VPN 用户现在可访问 `https://<SERVER_IP>:8767/` 或 `https://<SERVER_IP>:8767/infinite_demo.html`；原 HTTP 预检页也新增了这两个入口和 CA 下载说明。第一次可在证书警告中选择高级继续；反复测试可在受控 Windows 测试机安装 `ARDY WebGPU LAN Test CA`，其 SHA-256 为 `14:83:D9:C7:43:00:EC:5A:C7:D0:DD:32:FA:39:95:76:BD:24:C4:22:34:19:E9:34:2B:0F:FE:95:75:41:27:6B`。正式无警告分享仍应使用域名加公网/组织 CA，因为公网 CA 不会直接给 `<SERVER_IP>` 私网 IP 签普通可信证书。完整操作与安全注意事项记录在 `WebGPU_HTTPS测试说明.md`。全过程没有安装任何包、没有执行 conda 命令，也没有修改 `<REFERENCE_3DLM_PROJECT>`。
