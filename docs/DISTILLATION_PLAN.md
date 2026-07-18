# ARDY path-only WebGPU student：冻结设计与首轮实验协议

更新时间：2026-07-14

## 1. 已冻结的交付约束

- 只支持原版默认的无文本、稀疏 2D root waypoint、20 FPS、每轮生成 40 帧和无限续写；不加载 Llama/LLM2Vec。
- 保留原版 history 裁剪、最后历史帧重定中、世界坐标平移、heading、帧号/waypoint 和播放冻结语义。
- 最终浏览器生成器必须是 `NFE=1`，不保留运行时 CFG 分支；teacher 的 constraint guidance 被蒸馏进 student。
- encoder、flow、decoder 都重做。最终三者的实际 ONNX 权重文件合计严格 `<100 MB`，不是只看参数量估算。
- 混合精度训练，FP32 EMA；最终模型必须针对 FP16 推理训练和验收，禁止把 FP32 teacher 事后粗暴转半精度当成结果。
- 训练只使用现有 `py311` 环境。没有审核前不安装任何包。

## 2. 为什么 encoder/decoder 也必须蒸馏

同一 Edge 149 / NVIDIA Ampere WebGPU 客户端的前台稳态实测：

| 模块 | 当前参数 | 当前 FP32 ONNX | `session.run` p50 / p95 | 结论 |
|---|---:|---:|---:|---|
| history encoder | 17,554,048 | 70.58 MB | 52.03 / 60.48 ms | 必须替换 |
| 10-step denoiser | 155,823,252 | 685.05 MB | 938.45 / 1085.84 ms（10 步合计） | 必须替换并降到 NFE=1 |
| motion decoder | 17,834,276 | 71.55 MB | 50.29 / 58.54 ms | 必须替换 |

encoder + decoder 已经合计 35.39M 参数、约 142.1 MB、p50 约 102 ms。即使 flow 变成零耗时，保留它们也无法同时满足 `<100 MB` 和理想端到端 p50 `≤100 ms`。

## 3. 第一版 WebGPU student

### 3.1 固定 tensor 数据流

```text
normalized history [B,4,330]
  -> 原版无参 root 重定中 / translation / heading
  -> 取 body [B,4,325]
  -> HistoryEncoderStudent
  -> history latent [B,1,128]
  -> 原版 FSQ 64-level requantize（JS/WGSL，无学习参数）

noise [B,10,148]
history hybrid [B,1,148]            # initial 时全 0
path condition [B,64,3]             # normalized local x/z + valid
first heading [B,2]                 # sin/cos
has_history [B,1]
  -> OneStepFlowStudent, t=1
  -> clean generation hybrid [B,10,148]

[history,dummy] + generation -> hybrid [B,11,148]
  -> 原版无参 global-root/local-root/finalize
  -> latent [B,11,128], local root [B,44,4], token valid [B,11]
  -> MotionDecoderStudent
  -> normalized body [B,44,325]
  -> 原版 world translation / merge / rotation-6D FK
```

initial 与 continuation 共用一个静态 11-token decoder 图：initial 的 slot 0 是 invalid dummy，生成 token 放在 slot 1..10，最后只取 frame 4..43；continuation 的 slot 0 是真实 4-frame history，生成 token仍是 slot 1..10。`token_valid` 会在每个 mixer block 后清零 dummy，不能让 padding 污染生成帧。

普通鼠标 waypoint 仍对应原版绝对帧 `current + 60`。进入模型前只做原版已有的 window-relative 索引和 `-global_translation`，然后提取 root x/z 及 valid mask；不改成 dense 直线路径。

### 3.2 encoder：固定四帧 MLP

- 输入：`[B,4,325]`，flatten 为 1300。
- `Linear(1300,512) + GELU`。
- 3 个 `LayerNorm -> Linear(512,1024) -> GELU -> Linear(1024,512) -> residual`。
- `LayerNorm -> Linear(512,128)`，输出 `[B,1,128]`。
- 参数：**3,886,208**。

原 encoder 在浏览器续写时只有四帧，reshape 后其实只有一个 motion token；在该固定路径上，8 层 self-attention 没有可用的跨 token 关系，因此改成少量大 GEMM 的 residual MLP 是有针对性的删冗余，不是任意减层。

### 3.3 flow v2：保留时间结构、root 先行、body refinement

最初的 flatten-MLP v1（22,189,000 参数）已经被独立验证拒绝：它把所有时间位置压成一个向量，在 8192 个窗口上严重记忆训练集，验证 FK-MPJPE 约 1.10 m。该类只为识别旧审计权重而保留，不能用于浏览器。

当前 v2 的静态序列共 28 tokens：

- 1 个 global token：`first_heading [B,2] + has_history [B,1] + t [B,1]`，经 `Linear(4,384)`。
- 1 个 history token：`history_hybrid [B,1,148]`，经 `Linear(148,384)`。
- 16 个 path token：`path_condition [B,64,3]` 每 4 帧打成 12 维，经共享 `Linear(12,384)`；不会把稀疏 waypoint 改成 dense path。
- 10 个 generation token：`noisy_generation [B,10,148]`，经共享 `Linear(148,384)`。
- 加 28×384 learned position 和四类 type embedding；主干为 4 个 width 384、6 heads（head dim 64）、MLP expansion 2 的显式 self-attention block。
- 在最后 10 个 token 上先用 `Linear(384,20)` 预测 root velocity；再以 `Linear(20,384)` 注回 generation token，经过 2 个同规格 attention block，用 `Linear(384,128)` 预测 body velocity。
- ONNX 只用 MatMul、Softmax、Reshape、Transpose、LayerNorm、GELU 等当前 WebGPU 已实测可用的基础算子。
- 参数：**7,302,676**。

训练约定：

```text
x_t = (1 - t) * x0 + t * epsilon
target_velocity = epsilon - x0
x0_hat_at_t1 = epsilon - velocity_theta(epsilon, t=1, condition)
```

这保留原版“root 决定 body”的因果意图，但删除两套完整 8-layer backbone、3-way CFG 和 16-token Transformer 的重复开销。最终导出 wrapper 固定 `t=1`，只返回一次 `noise - velocity`。

### 3.4 decoder：短序列 token/channel mixer

- 输入：latent `[B,11,128]`、local root `[B,44,4]`、valid `[B,11]`。
- 每 token 将 4×4 local-root flatten 后与 latent 拼成 144 维，`Linear(144,512)`。
- 4 个 mixer block：一次固定 11-token 的 `11 -> 32 -> 11` token MLP；一次 `512 -> 1024 -> 512` channel MLP；两者 residual。
- `LayerNorm -> Linear(512,1300)`，每 token 输出 4×325 body；不再计算页面从未使用的 local-root output。
- 参数：**4,953,792**。

### 3.5 精确体积

下表是当前选中训练权重用现有 PyTorch/ONNX 工具链直接导出的实际文件，不是参数量乘字节数估算：

| 模块 | 参数 | FP32 ONNX | FP16 ONNX |
|---|---:|---:|---:|
| encoder | 3,886,208 | 15,551,369 B | 7,778,915 B |
| NFE=1 flow v2 | 7,302,676 | 29,256,743 B | 14,651,309 B |
| decoder | 4,953,792 | 19,836,043 B | 9,928,379 B |
| **合计** | **16,142,676** | **64,644,155 B** | **32,358,603 B** |

FP16 图离 100 MB 上限尚有约 67.6 MB，可用于 flow 加宽、增加 refinement 或局部保留 FP32。codec 已达到厘米级独立 FK 误差，当前余量优先给仍未过质量线的 NFE=1 flow。

## 4. teacher 数据必须严格来自原版推理

仓库没有训练集，因此首轮使用原版 FP32 模型自采样形成有界、可复现的 teacher shards：

- 初始窗口和 continuation 各占一部分；continuation history 永远来自此前 teacher rollout，不能用随机 latent 冒充真实历史。
- 文本 feature/mask 为全零；teacher constraint CFG 使用原版页面当前值。
- 条件分布覆盖无 waypoint、单 waypoint、多 waypoint、窗口内与 visible-future waypoint、急转和不可达边界。
- 调用原版 `autoregressive_step`；只用无侵入 trace wrapper 捕获首次 10-token Gaussian、每步输出和 `_generate_window` 返回的局部 hybrid。默认方法和结果不修改。
- 保存 teacher seed、绝对/相对帧索引、history、world translation、heading、motion mask、observed motion、初始 noise、10-step endpoint、显式 motion 和 FK joints，保证以后能重放。
- `<DATA_VOLUME>` 空间紧张，采用有上限的分片和边训练边回收策略，不无限堆 teacher 样本。

## 5. 训练顺序

### A. codec teacher fitting

encoder 独立 loss：teacher normalized FSQ latent、requantized latent、一致 code-bin 比例，以及通过 teacher/student decoder 后的动作误差。decoder 独立 loss：normalized body、6D rotation、FK joint、速度/加速度、foot-contact/slide 和窗口 seam。encoder 与 decoder 必须各自通过，不能只看端到端平均 loss。

### B. flow matching 基线

先用 teacher endpoint 训练连续 flow。首轮 timestep 混合分布：50% 精确 `t=1`、35% 高噪声偏置、15% uniform；这样单步目标占多数，同时保留中间时间的向量场约束。generation token 才加噪和计算主 loss，history 保持原值。

### C. NFE=1 endpoint / rollout fitting

固定相同 noise、history 和 waypoint，直接拟合原版 10-step guided teacher endpoint；加入 decoder 后 root/path、FK、速度、foot 和 seam loss，并做多轮 autoregressive rollout。该阶段即开始以最终 `t=1` 图为主要验证对象。

### D. distribution matching + temporal adversarial

基础拟合稳定后再加 DMD2 类两时间尺度 distribution matching 与 temporal adversarial critic。critic 可复用冻结 teacher encoder/denoiser 中间 feature，但最终浏览器不携带 critic 或 teacher。判别输入同时包含 normalized motion、FK joints、速度、foot contact 和多尺度时间窗；保留 paired loss 防止 waypoint 与 history 条件被无条件 GAN 目标冲掉。

训练始终使用现有 Accelerate；BF16/FP16 autocast、FP32 master/EMA。项目本地实现 EMA 和 JSONL + TensorBoard 记录，不依赖缺失的 `ema_pytorch`/`wandb`。

## 6. 验收指标

每个 checkpoint 在固定 validation seeds/conditions 上记录：

- encoder：latent L1/L2、FSQ bin accuracy、teacher-decoder 后 FK MPJPE 和 seam。
- decoder：normalized body、rotation geodesic、FK MPJPE、root/body velocity、acceleration、foot contact F1、foot slide。
- flow：paired teacher endpoint latent/root、解码动作/FK、waypoint x/z error、heading error、diversity 与 teacher-feature distribution distance。
- rollout：1/5/20/50 次续写的接缝、root drift、foot slide、constraint success、NaN/Inf 和视觉固定样例。
- WebGPU：每图下载、session create、warmup、p50/p90/p95、数值对拍、峰值显存，以及 40 帧完整 continuation 的 wall time。

硬门槛：总权重 `<100 MB`、NFE=1、端到端 p50 `≤100 ms` 为理想线、p95 `≤170 ms` 为必过线；推理期间播放头冻结，不允许用跳帧掩盖延迟。

## 7. 首轮不超过 12 小时的资源协议

- 启动前重新读取 8 张卡的显存/利用率，动态选 4 张最空闲卡，不把之前的 GPU 编号写死。
- 先做单卡 20-step overfit，再做 4 GPU 200-step smoke；shape、loss、梯度、EMA、resume 和评测全部通过后才启动长实验。
- teacher 采样、codec fitting、flow baseline 和 NFE=1 验证分别记 wall time；到 12 小时硬截止时保存 checkpoint、优化器、EMA、随机状态和完整指标，不拖延成无限任务。
- 首轮目标是验证数据/结构/收敛/浏览器闭环，不承诺在 12 小时内完成最终 DMD2+adv 质量。

## 8. 当前状态

- 原版结构、参数、端到端和分模块 WebGPU 基线：完成。
- flatten-MLP flow v1：独立验证拒绝，验证 FK-MPJPE 约 `1.10 m`；只保留审计文件。
- temporal-attention flow v2：shape、FP32/FP16 ONNX、CPU ORT 与真实 Edge/NVIDIA WebGPU 对拍完成。随机权重架构图的 flow median 为 FP32 `73.11 ms`、FP16 `54.34 ms`；配合同规格 encoder/decoder 的后台 median 直接求和约 FP32 `122.4 ms`、FP16 `110.0 ms`。这是后台 worker 的相对筛选，不是前台整链路验收。
- 131,072-window 大 teacher corpus 已完成全量 schema、shape、dtype、finite 与 SHA-256 校验：256 shards、9,700,210,688 B、87.5% continuation、74.91% constrained；teacher 计算保持 FP32，仅落盘使用 FP16。
- codec v3 在四张最空闲 A6000 上用 BF16 + FP32 EMA 训练 4000 step，训练 wall time `132.30 s`。独立 1024-window 验证选择 step-4000 encoder EMA + decoder raw：encoder quantized latent L1 `0.07629`、bin accuracy `39.09%`；decoder FK-MPJPE `0.02307 m`、rotation `0.08969 rad`、foot slide `0.02097`。
- flow v2 large 用同一 corpus、90% student history + 10% teacher history、BF16 + FP32 EMA 训练 4000 step，训练 wall time `210.88 s`。独立验证选择 step-4000 raw：endpoint root/body MSE `0.09803/0.61151`、FK-MPJPE `0.32338 m`、rotation `0.42772 rad`、path error `0.27283 m`。EMA 略差。
- teacher-history 对照的 FK-MPJPE `0.32310 m`、path error `0.27364 m`，与部署用 student-history 基本相同；因此当前瓶颈是 flow，而不是 encoder 历史误差。decoder 自身只有 `2.31 cm` FK 误差，也不是当前主瓶颈。
- 同一 1024-window 验证集上将 encoder/flow/decoder 内部算术全部切到 FP16 后，FK-MPJPE 为 `0.323383 m`（FP32 `0.323380 m`），path error 为 `0.272752 m`（FP32 `0.272826 m`）；BF16 分别为 `0.323497/0.273955 m`。所以当前动作误差来自 flow 能力，不是学生 FP16 数值崩坏；但真实 WebGPU FP16 对拍仍是独立门槛。
- 当前 codec 可以冻结作为下一阶段基线；flow 虽较 8k probe 的 `0.570 m` FK / `0.576 m` path error 明显改善，但仍未达到动画交付质量，禁止仅因 finite 或速度合格就集成成最终前端。
- 已先导出两个 width-512 容量候选再决定训练：4+2 blocks 为 12,882,580 参数 / 25,787,457 B FP16 flow，连同 codec 共 43,494,751 B；5+2 blocks 为 14,985,364 参数 / 29,994,808 B，连同 codec 共 47,702,102 B。两者 CPU ORT 对拍通过，真实 WebGPU 延迟任务已排队；在客户端结果回来前不盲训。
- 选中训练权重的六个 FP32/FP16 浏览器 parity/latency 用例已经单独导出，不覆盖随机权重诊断用例；真实浏览器结果记录后再决定 mixed precision。
- 从 teacher 采样、训练到验证均未安装新包，只使用现有 `py311` 可执行文件；完整审计在 `distill_runs/first_12h_20260714_014643/RUN_PLAN.md`。
- 已补齐忠实 student one-window runtime；和原版 trace 对拍时 initial/continuation 的 path、heading、history root 与 world translation 最大绝对误差均为 0，输出 shape/finite 通过。CUDA batch-1 continuation 的 FP32 p50 为 encoder `0.488 ms`、flow `2.624 ms`、decoder `1.265 ms`、整链路 `6.384 ms`；FP16 为 `0.445/2.837/1.321/6.623 ms`。这是服务器参考，不替代 Edge WebGPU。
- 已修正 student history 的 FSQ 语义：旧 flow v2 权重训练时错误使用了连续 encoder 输出；以后 encoder 后必须 requantize。旧权重用忠实 FSQ history 重评后 FP32 FK/path 为 `0.323774/0.272751 m`，与旧结果仅差约 0.4 mm FK，因此错误不是当前质量瓶颈，但旧权重仍只作 provisional baseline。
- codec v3 继续作为质量基线冻结，但真实 Edge 架构筛选中 decoder median 仍约 `34.94/38.78 ms`（FP32/FP16），高于 20–25 ms 子预算。为避免盲目重训，已先导出两档无训练候选：width-384 e2/d3 的 FP16 codec 为 `8,147,992 B`，width-256 e2/d3 为 `4,129,479 B`；8 个 FP32/FP16 WebGPU latency job 已排队。只有候选带来明显浏览器时延收益时才重新蒸馏 codec。
- 已完成忠实原版 UI 缓冲/重规划语义的 1/5/20/50-window rollout。provisional flow 在 FP32 50 窗口的 root drift mean/final 为 `17.642/43.491 m`、waypoint mean `18.146 m`、FK `17.683 m`；FP16 几乎相同，证明崩溃不是精度导致。
- oracle-flow codec 闭环消融在 50 窗口的 root drift 为 `0`、waypoint mean 与 teacher 相同为 `0.0105 m`、FK 为 `0.0229 m`，确认当前米级崩溃是 flow 瓶颈。但 codec 的 foot slide/joint seam jump 仍为 `0.332/0.874 m/s`，teacher 为 `0.064/0.257 m/s`，所以若小 codec 有真实 WebGPU 延迟收益，其再蒸馏必须包含针对性 foot-contact/slide 和 seam loss，不能只拟合单帧 FK。
- 因而“encoder/decoder 快则不动”的执行规则已量化：原版两者合计约 `102 ms`/约 `142.1 MB`，必须替换；codec v3 作为质量对照保留，但 decoder 还未达约 25 ms 子预算。只在 width-384/256 候选的真实 WebGPU 实测明显更快时重训 codec；否则先不动它，集中解决 flow。
- DMD2 类训练骨架已实现并通过 smoke：fake-score/critic 每步更新，generator 默认每 5 步更新；teacher `x0` 和 fake `x0` 构成 normalized distribution-matching surrogate，temporal critic 使用 normalized motion + FK joints + velocity + foot contact 与 path/history/heading 条件。teacher adapter 通过 10-step DDIM corpus 回放，最大误差 `7.03e-6`。
- 1 warmup + 3 generator-update BF16 smoke 的所有反传、EMA、保存和 step-2 断点恢复通过，四份权重均 finite。这仅是工程闭环，不声称 3 步带来质量提升。正式顺序仍是 WebGPU 选架构 → faithful-FSQ supervised fitting → DMD2+adv → 1/5/20/50 rollout。teacher/fake-score/critic 均不导出到浏览器。
- width-512 5+2 的 supervised fitting 已扩展到 20k step，四卡 wall time `1172.80 s`。独立集 20k EMA 的 FK/path 为 `0.22286/0.13188 m`，比 4k raw 的 `0.29571/0.21891 m` 明显改善；但 faithful rollout 在 5/20/50 窗的 FK 仍为 `0.718/10.310/16.753 m`。所以监督拟合已作为新基线，下一步转向弱 critic、DMD-only 与低 adversarial-weight 消融，用 5/20/50 窗而不是单窗指标选择。
- codec v3 在 CUDA 的 encoder/decoder p50 为 `0.490/1.272 ms`，因此训练主线先冻结；但 Edge FP32 仍为 `14.37/34.94 ms`。只有 width-384/256 候选在同一真实浏览器把 decoder 压到约 `25 ms` 或以下且 codec 合计至少快约 `25%`，才单独重训 encoder/decoder；否则不为了减参而牺牲已有质量。
- DMD-only 200-step 与弱 adversarial 探针均未改善 1/5/20-window 闭环，因此暂不扩大 critic。完整 corpus 的 path-condition norm 最大仅 `3.329`，20k EMA rollout 从第 7 窗越界且第 50 窗达到 `29.120`；on-policy teacher 从同一学生状态仍能大幅接近 waypoint，根因已锁定为 path-condition exposure bias。
- 已实现忠实 UI buffer/replan 语义的 on-policy teacher 采样，并完成三轮 DAgger。第一轮大 OOD 数据证明可把 50-window waypoint 从 `17.220 m` 压到约 `1 m`，但 naive path loss 会把 foot slide 推到 `13.495 m/s`；通过 replay、物理 loss、较温和的第二/三轮状态聚合和权重插值，当前 round3 step-400 raw 达到 waypoint `1.054 m`、foot slide `2.224 m/s`、joint seam `5.549 m/s`。
- round3 step-400 已在五个随机 50-window 轨迹上复验：waypoint mean 平均 `1.108 m`（`0.894–1.278`），foot slide 平均 `2.256 m/s`，不是单 seed 偶然。训练侧 physical seam loss 对 rollout 提升不显著；teacher clean latent 经相同 compact codec 的 oracle joint seam 仅 `0.874 m/s`，剩余拼接误差仍来自 one-step flow 首帧。
- evaluator 已加入默认关闭的 8-frame feature-space inertialization，严格保留播放时间、帧号、buffer 和 waypoint 语义。五 seed joint seam mean 从无后处理平均 `5.896` 降到 `1.358 m/s`，waypoint mean 仍为 `1.108 m`，foot slide 小幅变为 `2.299 m/s`。当前浏览器候选固定为 round3 step-400 raw + 8-frame/100% inertialization，下一步把完全相同的逻辑接入 JS runtime 做视觉和真实 WebGPU 对拍。
- 当前候选 FP32 safetensors 精确合计 `95,318,528 B`（encoder `15,546,912` + flow `59,951,304` + decoder `19,820,312`），已低于 100 MB 原始权重线且 NFE=1；最终 ONNX graph/external data 必须重新按实际文件求和。CUDA 整窗约 `7–8 ms` 只作服务器参考，不能替代 Edge 前台 p50/p95。
- round3 step-400 已重新导出为浏览器最终候选：encoder/flow/decoder 加两个固定状态图共 `95,425,945 B`（`91.005 MiB`），无 external-data；CPU ORT 对 PyTorch 的首段+续写+惯性化整链路最大绝对误差 `6.95e-5`，buffer 精确为 `40 -> 77`。
- 当前权重的真实 Edge 149/NVIDIA Ampere WebGPU 对拍三项全过：encoder p50/p95 `14.160/16.121 ms`，flow `64.390/77.036 ms`，decoder `33.613/56.985 ms`，最大绝对误差分别 `1.55e-6/3.46e-6/7.63e-6`。独立 stage p50/p95 和为 `112.163/150.142 ms`，过 170 ms 必过线但略高于 100 ms 理想线；必须继续测动画页前台整链路，不能把独立分位数之和冒充端到端。
- `/infinite_demo.html` 已切换到 FP32 全 student NFE=1，并接入相同的 8-frame inertialization、原版无限 buffer/replan/时间轴和鼠标 `current+60` 稀疏 waypoint。生成期间播放头冻结且不补跳停顿时间。下一步只剩刷新页面后的完整视觉验收、前台 benchmark 与必要的细节修正。

## 9. 视觉否决后的正式重训协议

真实页面验收已否决 round3 step-400：性能和 PyTorch/ORT/WebGPU 数值对拍通过，但 FP32 动画仍持续抖动并呈噪声状。该候选只能作为失败基线，不能继续导出或展示为完成版本。

新的执行顺序如下：

1. 先重训 compact decoder。首窗口与 50-window codec-oracle 已证明 decoder 的 root-relative body jerk 和高频显著高于 teacher；必须先让 teacher clean latent 的解码动作过线，flow 才有可信目标。
2. decoder 固定 LR `3e-6`、100k optimizer steps、四卡 DDP、每卡 batch 512（global batch 2048）、encoder frozen、BF16 + FP32 EMA、无 cosine/无 weight decay。feature/FK/rotation/contact/foot 与 joint/rotation velocity-acceleration-jerk 每步计算；额外强化 4-frame token 边界 stencil。每 5k 静态验证，每个关键 checkpoint 做 50-window oracle 与 phase/P95/FFT 验证。
3. decoder 通过后，直接训练部署用 NFE1 generator；不再先训 4-step/2-step student。学习率限定在 `1e-5–1e-6` 量级，paired teacher endpoint/condition 只作为一步映射的初始锚点，所有 decoder conditioning 使用预测 root 推导的部署 local-root。
4. 一步基础映射稳定后，直接加 DMD2 + conditional temporal adversarial 做短程分布蒸馏；critic/score loss 下降不算成功，仍以多 seed 1/5/20/50-window waypoint、FK、foot slide、seam、P95 jerk、频域和视觉验收选模。DMD2 中为估计分布梯度而对辅助 score 网络抽样随机 `t` 不等于生成器 NFE>1；学生始终只做一次 forward。
5. 只有 FP32 student 通过上述质量门槛后才重新导出 WebGPU 图并考虑低精度。上一版 pure FP16 已被视觉否决，不再优先。

当前 1k fixed-LR decoder run 仅是 loss 方向消融；均匀 temporal 5k run 仅是 token-boundary 对照。误用单卡的 `codec_decoder_boundary_const3e6_100k` 已在 20k 后停止，只作流水线对照。正式训练从原始 checkpoint 干净重启，使用 GPU `3,4,5,6`，目录为 `codec_decoder_boundary_4gpu_b512_const3e6_100k`，预算为完整 100k，不得将单卡、吞吐扫描或 pilot checkpoint 冒充最终模型。四卡 B512 吞吐检查为约 `11,831 samples/s`，有效区间 SM 通常 `92%–95%`。

正式任务截至 65k 的严格 paired sweep 已完成。20k→65k 的独立 FK 从 `14.134 mm` 降到 `10.765 mm`；五 seed 50-window codec-oracle 的 jerk/HF/seam 均值从 `1.521×/2.033×/1.723×` 降到 `1.278×/1.456×/1.506×`，phase-1 jerk 均值从 `1.940×` 降到 `1.535×`。65k 最坏 seed 的 jerk/HF/seam 仍为 `1.496×/1.711×/1.781×`，所以继续 100k 且不提前进入 flow。完整原始记录和逐 checkpoint delta 位于 `eval/codec_boundary_4gpu_b512_sweep.json`。四卡健康检查以 rank/PID、SM/功耗和 step 增量为证据，不能把机器上其他任务的驻留显存当作本训练占用。

flow 代码预审已修正一个与本协议直接冲突的问题：`train_flow.py` 原先硬编码 cosine scheduler，且 t=1/high-noise mixture 不能从命令行固定。现在默认 LR 为 `1e-5`、默认 constant schedule，并显式提供 `--exact-t1-probability` 与 `--high-noise-probability`，两者会进入 config 且有边界/和校验。现已进一步实现真正的 uniform backward-Euler rectified-flow solver：从 `t=1` 到 `t=0` 逐 stage 调用同一 time-conditioned flow，paired endpoint、decoder 与物理 loss 端到端穿过全部 NFE；静态 evaluator、无限 rollout 和 `StudentArdyRuntime` 均有显式 `flow_steps`，旧 NFE1 wrapper 与新 `steps=1` 在真实权重上最大误差为 0。5 个 solver/time-sampling 单测、模型 shape/finite 测试、NFE2/4 真实权重 forward 均通过。

GPU 2 上另做了严格标记的单步 NFE4 BF16 工程 smoke：constant LR `1e-6`，70% exact-t1 + 25% high-noise + 5% uniform，完整 velocity/endpoint/decoder/path/root-temporal/normalized-seam/physical-seam 反传、clip、optimizer、EMA、保存和恢复链路通过；105 个 raw/EMA/model tensor 全 finite。裁剪前 grad norm 为 `106`、clip threshold 为 `1.0`，因此正式四卡训练前还必须审计各 loss 梯度尺度和吞吐。目录 `flow_nfe4_solver_smoke_step1/` 已写入 `ENGINEERING_SMOKE_ONLY.md`，明确禁止把 1 step 当质量模型。正式 flow 仍须等待 decoder 100k sweep 完成后才启动。

decoder 正式任务现已完整结束于 100k，最终状态中 optimizer/EMA/scheduler/sampler/四 rank RNG 均完整且 finite。100k raw 的独立 FK 为 `9.733 mm`；五 seed 50-window acceleration/jerk/HF/seam 均值为 `1.0594×/1.2254×/1.3583×/1.4552×`，phase-1 jerk 均值为 `1.4555×`。相对 95k，acceleration/HF/seam 五条路径全部改善，jerk 为四条改善一条轻微回退；四项均值仍全部下降，所以后续固定选择 100k raw，而不是 EMA 或机械沿用 95k。完整证据位于 `eval/codec_boundary_4gpu_b512_sweep.json`。

NFE4 梯度审计将 physical seam 从初始 `0.1` 校准到 `0.02`：在 teacher/on-policy batch 上，加权 physical 梯度范数由约 `49.4/38.6` 降到 `9.89/7.71`；最终 combined/core cosine 为 `0.867/0.785`，避免 20 FPS 的 m/s 尺度淹没 velocity+endpoint 核心目标。训练数据同时切换为可 checkpoint 的 shard-local mixture sampler，只加载实际使用字段，9 项数据与 solver 测试通过。四卡每卡 B512/global 2048 的 200-step NFE4 全 loss 扫描达到约 `2.3536 step/s`、`4820 samples/s`，全部 scalar finite，并已标记为 `THROUGHPUT_ONLY`。

正式 NFE4 任务已在物理 GPU `3,4,5,6` 启动：50k optimizer steps、固定 LR `3e-6`、BF16 + FP32 optimizer/EMA、无 warmup/无 cosine、70% exact-t1 + 25% high-noise、每步完整 decoder/quality/seam，目录 `flow_nfe4_4gpu_b512_const3e6_50k_phys002/`。四个训练 PID `1114008/1114009/1114010/1114011` 分别是 rank 0/1/2/3，并由 NVML 映射到物理卡 3/4/5/6；连续采样多数为 `89%–98% SM`、`238–274 W/卡`，日志已从 step 1 推进到 200。其他任务的驻留显存不参与这一归属判断。下一检查点为 5k：必须同时评估 raw/EMA 的静态、NFE4 faithful 1/5/20/50-window、waypoint/FK/foot-slide/seam/P95 jerk/频域，再决定继续 50k 与进入 NFE2，不能只看训练 loss。

已补齐最终 decoder100k 下的正式 step-0 对照：独立 1024-window 的 endpoint/FK/path 为 `0.78593/0.23893 m/0.16037 m`；五 seed 50-window waypoint/FK/foot-slide/joint seam 均值为 `1.707 m/2.032 m/1.284 m/s/2.588 m/s`，body acceleration/jerk/HF/seam 相对 teacher 为 `2.793×/5.410×/13.520×/8.806×`。因此初始 NFE4 仍是明确的失败质量基线，5k 后必须与这些固定路径 paired 比较。

分布蒸馏阶段采用独立的短程协议，不能继承 supervised 50k 的步数。`train_dmd2.py` 默认已从危险的 10k + `2e-5/1e-4/1e-4` cosine 改为最多 2k 的保守起点：generator/score/critic 固定 LR `1e-6/5e-6/2e-6`，无 warmup，ratio 1，DMD/adv `0.05/0.005`，每 100 generator update 保存；代码硬拒绝超过 5k generator updates或任一 LR 高于 `1e-5`。实际选模点固定为 200/500/1000/2000，按相同多 seed rollout 和视觉早停，而不是等到 2k 才首次查看。

NFE4 的首个正式 5k 检查点已完成五 seed、50-window paired sweep。step 0→5k raw 的 waypoint/FK/foot-slide/body jerk/body HF/joint seam 均值从 `1.707 m/2.032 m/1.284 m/s/5.410×/13.520×/8.806×` 降到 `0.178 m/1.274 m/0.643 m/s/2.532×/5.156×/3.775×`，这些指标逐 seed 均为 `5/5` 改善；final root drift 只有 `3/5` 改善。EMA 的 waypoint 略好到 `0.158 m`，但 FK、foot、jerk、HF、seam 普遍略差，所以暂选 5k raw 并继续 supervised NFE4。它仍有 root jerk `15.84× teacher` 和米级 rollout FK，绝不据此提前进入 NFE2/DMD2。完整汇总位于 `eval/flow_nfe4_formal_sweep.json`。

后续 DMD2+conditional adversarial 只统计 **generator update**，而不是 dataloader iteration、critic/score update 或样本数。计划上限保持 2k、代码硬上限 5k，generator LR `1e-6`；在 200/500/1000/2000 做固定多 seed + 视觉检查，任一主要指标持续反转即以前一 checkpoint 停止。该短阶段的目的只是把已经合格的 NFE2 分布压到 CFG-free NFE1，不承担修复尚未拟合好的轨迹或高频噪声。

### 9.1 2026-07-14 执行路线更正：直接 NFE1

本小节正式取代上文“NFE4→NFE2→NFE1”的后续执行顺序。最终部署只调用一次 flow，而同一 NFE4 step-5000 权重的实测已表明，四步质量改善不会自动迁移到单步：NFE4 的静态 FK 约 `0.212 m`，强制 NFE1 后约 `0.244 m`，endpoint 约 `0.784`。因此 NFE4 已停在日志 step 7100（最后完整 checkpoint 5000），不再启动 NFE2。

新路线从已有 round3 step-400 的原生 NFE1 生成器直接开始，只训练三类短程候选：`DMD-only`、`DMD + low conditional temporal adversarial`、`DMD + low adversarial + weak paired conditional anchor`。fake-score/critic 对 generator 的更新比固定为 `5:1`，generator/score/critic 固定 LR 起点为 `1e-6/5e-6/2e-6`，不使用 cosine。经四卡 global batch 256 实测，200 次辅助预热可使 fake-score total 从 `3.041` 降到 `0.328`、DMD gradient abs 从早期 smoke 的 `9.07` 降到 `1.93`，因此正式消融每个分支先预热 200 次，但这些预热不计作 generator step。

检查点为 `200/500/1000/2000` generator updates，每次必须与相同五个 seed、50-window、无惯性化的原生 NFE1 基线 paired 对比。基线均值是 waypoint `1.785 m`、FK `2.388 m`、foot slide `2.581 m/s`、body jerk/HF/joint seam 分别为 teacher 的 `14.81×/23.62×/20.25×`。不以 fake-score/critic loss 下降作为生成器进步；只有上述 paired 指标、视觉和条件跟随同时不退化，才进入下一检查点。

首轮直接 NFE1 消融已经执行并触发早停。弱 anchor + adversarial 的 step-200 EMA 暂为相对最好点：五 seed 50-window waypoint/FK/foot slide/root seam/joint seam 为 `0.849 m/1.912 m/2.422 m/s/3.253 m/s/5.859 m/s`，但 root/body jerk 仍为 teacher 的 `85.94×/14.58×`，不具备浏览器交付质量。从完整状态继续到 step 500 后 waypoint、平均 root drift 和 FK 分别恶化 `9.7%/27.6%/27.2%`，且 FK 为 `0/5` 改善；critic 同时饱和，因此不续 1000。

已实现并验证 timestep-conditioned diffusion critic。完整 `t=0..9` 的判别准确率从无噪声约 `0.914` 单调降至最高噪声约 `0.508`，但正式 g200 虽改善单 seed 路径与 final drift，却把 root/joint seam 推到 `3.279/6.078 m/s`，root jerk 推到 teacher 的 `105.10×`，因此拒绝。逐 loss 梯度审计又证明辅助锚点总梯度约 `1.0718`，不是简单“权重太小”。随后严格配对的 `t=0..1`、critic LR `1e-6`、adv `0.002` 两个 g100 分支也均使 final drift、seam 和 jerk 反转；quality/root-temporal 从 `0.004/0.004` 调为 `0.008/0.002` 的分支更差。两者不续、不补五 seed。

因此后续计划不再包含任何 NFE>1 中间课程；历史 NFE4 只作为“多调用改善不等于单调用改善”的负对照。所有新方法必须直接作用于一次 `x1→x0` 调用，并先在 `≤100` generator updates 的同 seed 50-window gate 同时改善 waypoint、final drift、FK、foot、seam 和 jerk，才允许扩到五 seed 与 200-step。当前相对最好仍是上述 clean g200 EMA，但其质量不合格，不能导出替换浏览器模型。

已完成一个更贴近官方 DMD2 的直接 NFE1 实现：DMD query 从旧的最高噪声偏置改为 `t=0..8` 均匀采样；adversarial head 不再另行编码加噪后的 FK/高阶差分，而是读取 conditional fake-score backbone 的 generation bottleneck，并与 epsilon loss 在同一 guidance optimizer step 联合更新。generator 的公开输出逐元素不变，classifier/teacher/fake-score 都是训练期模块，不进入 WebGPU。

四卡 200-iteration 校准显示 score loss和按 timestep 单调下降的 critic accuracy都正常；唯一正式 g100 也没有 critic saturation。但固定 seed 50-window gate 仍拒绝：g100 EMA waypoint 为 `1.143 m`，而 final drift/foot/root seam/joint seam/root jerk 为 `3.464 m/2.592 m/s/3.777 m/s/6.575 m/s/109.04×`，均劣于旧 clean g200 EMA 的 `2.393 m/2.398 m/s/3.333 m/s/5.819 m/s/98.96×`。g50 EMA 只有 body jerk 小幅更好，其余关键项仍回退。因此停止于 g100，不续、不扩 seed。

下一步仍只允许直接 NFE1。新的研究问题已收窄为：怎样让一次输出获得针对 seam 与高频时域分布的有效梯度，同时不破坏 waypoint/final drift；不得再用 NFE4/NFE2 课程规避这个问题。任何新 objective 必须先做逐项生成器梯度方向/尺度审计，并在同一 g50/g100 rollout gate 证明路径与平滑同时改善，才有继续价值。

### 9.2 直接 NFE1 图内 root projection 的门控结果

后续没有恢复任何 NFE>1 课程。固定 `[1,4,6,4,1]/16` root temporal projection×4 已放进一次 `x1→x0` 的训练/部署图，零参数、零额外模型调用；它把旧 g200 EMA 五 seed 的 root jerk 从 `85.94× teacher` 降到 `1.42×`、foot slide 从 `2.422` 降到 `0.728 m/s`，但 waypoint 从 `0.849` 退到 `1.493 m`。随后只在这个最终 NFE1 图上做 4 卡短程 DMD2+adv，固定 LR `1e-6/5e-6/2e-6`、无 cosine，并在 g50/g100 门控。

g100 raw 五 seed相对“旧 g200 EMA + projection×4”使 waypoint/FK/final drift 改善 `26.53%/7.39%/20.07%`，root/joint seam 也改善 `4.69%/2.75%`；代价是 root jerk 从 `1.42×` 回升到 `1.61×`。由于 g50→g100 已出现路径继续变好而时域平滑变差的清晰 trade-off，训练停止在 g100，不延长到 g200。该 raw checkpoint 已按 FP32 导出为版本化浏览器候选，五张 ONNX 合计 `95,446,858 B`，服务器 ORT/PyTorch 整链路最大误差 `2.97e-5`；前端 inertialization 关闭。下一门槛是实际 Edge WebGPU 整体视觉与 p50/p95，不由训练 loss 决定是否继续。

### 9.3 一步目标对齐与前台实测

后续路线固定为直接 NFE1 teacher anchor → 短程 DMD2 + conditional adversarial，不再建立任何 NFE>1 中间过程。原因不只是节省训练：多步 solver 优化的是多次调用后的积分轨迹，会引入最终浏览器图不存在的中间状态。最终验收也只测一次 `x1→x0` 的条件分布、一步时域质量和闭环 rollout。

未训练的 cubic control-point root projection 已做 8/10/12/16 点静态消融。最好的 8 点版仍被现有 projection×4 在 endpoint/FK/foot/root-temporal/path 全面支配，因此在 rollout 和四卡训练前直接否决。这一负结果保留为结构审计，不改变默认 binomial 图。

新 release 已在实际 Edge 149/NVIDIA Ampere 动画页完成服务器远程触发的前台基准：40 帧 continuation 的 total p50/p95 `96.35/140.36 ms`，encoder `13.02/23.05 ms`，NFE1 flow `57.71/69.42 ms`，decoder pipeline `24.23/29.07 ms`。所有时延门槛通过，且 flow 占 p50 的约 60%，下一轮减参/蒸馏的性能优先级仍是直接 NFE1 flow。性能结论不代替视觉验收。

### 9.4 最终步继承规则（2026-07-15）

当前正式路线不再为相同协议内的中间 checkpoint 做选模搜索。只要该阶段的实现、完整恢复状态、数值有限性和训练/部署语义审计通过，就机械采用计划终点的 EMA：监督 flow 使用 100k EMA，on-policy/replay 纠偏使用 20k EMA，短程 DMD2/ADV 使用既定最终 generator step 的 EMA。中间 checkpoint 只用于断点续训和故障定位；固定集、多 seed 闭环与视觉检查仍作为阶段是否有效、能否进入导出/交付的验收证据，不用于改拿更早一步。

### 9.5 最终 DMD2 与浏览器交付状态（2026-07-15）

DMD2/ADV 已按锁定协议跑满 1000 个 generator update，并机械采用最终 EMA；500-step 只用于完整恢复状态审计，没有任何 checkpoint 搜索。最终 state 的 generator/score/critic optimizer step 为 `1000/1200/1200`，EMA update 为 `1000`，sampler、四 rank RNG、三组 scheduler 与所有权重均完整 finite，state/export 逐 tensor 一致。

五 seed 50-window 表明最终 DMD2 相对纠偏末步主要是小幅 trade-off，而不是新的大幅质量跃升：body 高频比平均改善 `12.51%` 且 `5/5` seed 改善；waypoint、FK、joint seam 和 body jerk 分别回退约 `6.05%/0.38%/7.13%/2.95%`。这些结果不改变最终步规则，但必须带到视觉验收，不能仅凭 DMD/critic loss 宣称质量提升。

最终 FP16 release 为 `standard48m_dmd2_final1000_ema_fp16`，学习图 `96,182,693 B`、含 utility 总下载 `96,227,957 B`，低于 100MB；NFE=1、runtime CFG=false、text/Llama=false、inertialization=0。服务器 CPU ORT 对 PyTorch 的连续 endpoint max abs `0.01552`，完整 motion mean/p99 `0.01021/0.05062`、cosine `0.9999326`、FSQ bin match 最低 `94.84375%`，40→77 buffer 语义通过。

`/infinite_demo.html` 已指向该 release，并新增远程 `validate`：浏览器将固定 initial+continuation golden case 通过 WebGPU/WASM 整链运行，与服务器 CPU ORT 对拍后回报 endpoint/motion/FSQ/history 指标。8766 已排队 validation；真实 Edge client 尚未连接。下一步只剩用户刷新页面后由服务器自动收取 WebGPU 数值结果，再远程触发 3 warmup + 20-run 前台 benchmark 和 start/waypoint 无限播放；最终是否可交付仍以用户看到的连续动作、抖动、路径响应和时间轴行为为准。

### 9.6 文本版最终训练路线与多 GPU 资源策略（2026-07-15）

本节取代“不含文本”的最终 release 路线。teacher 仍用原版 Llama-3-8B/LLM2Vec；student prompt 预处理使用 FLUX.2-klein-4B 自带 Qwen3，BF16 推理并缓存完整 `7680D` feature。文本编码器不计入 51.952M 运动 student；运动图新增 full `7680→512` projection、无参数 RMSNorm 和 future-heading projection，不做 PCA/低秩，flow 深度仍为 8+8/NFE1。推理运动图统一 FP16，文本 feature 缓存 FP16。

数据优先于容量：8192 prompt × none/mouse sparse/mouse dense/keyboard velocity/keyboard heading × initial/continuation × prompt switch × rollout depth 1/2/4/8/16。正式 teacher train set 为 524,288 窗，四 GPU 并行；另生成不同 seed 的独立 validation。teacher batch 实测 128 已接近饱和，后续生成不再用 batch 8。训练前先验证全部 shard SHA/finite/分布覆盖。

训练按继承链执行：`path-only final EMA → condition-only full projection warm-up (5e-5→1e-5, 4 GPU) → joint supervised continuation (base 1e-5→5e-6, condition 可独立 LR, 4 GPU, BF16+FP32 Adam/EMA) → short DMD2+conditional adversarial (hundreds–thousands generator updates)`。禁止 cosine，禁止相同超参数下随机重启。四卡正式启动前扫 per-GPU batch 32/64/128，以真实 SM、功耗、显存、samples/s 和 step time 选最大有效吞吐；数据预处理同样分卡并行。

最终 gate 仍是完整无限生成而非单窗 loss：prompt 可辨识性、鼠标/键盘控制、prompt switch、1/5/20/50-window waypoint、FK、foot slide、seam、P95 acceleration/jerk、频域、FP16 PyTorch/ONNX/WebGPU 数值，以及 Edge 前台动画视觉和 p50/p95。文本前处理耗时与每窗运动推理解耦报告，不能把 Qwen 一次性编码耗时错误计入每个 2 秒窗口。
