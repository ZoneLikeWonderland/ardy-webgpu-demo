# WebGPU Toy / ARDY parity worker

页面包含两部分：原生 WGSL WebGPU 自检，以及 ONNX Runtime WebGPU 与服务器 PyTorch 参考输出的自动对拍。浏览器页面会注册为任务 worker；服务器可下发用例并自动接收逐元素比较结果，不需要在页面逐次点击。

没有 Llama/LLM2Vec。ARDY 用例固定 `text_encoder=False`，路径版 denoiser 使用 zero text 和 `cfg_weight_text=0`。

服务器启动命令（只使用现有 py311 环境及其中已有的 NumPy；不安装新包）：

```bash
cd /mnt/newdisk/20260713_test_ardy/webgpu_toy
/root/anaconda3/envs/py311/bin/python compare_server.py --bind 0.0.0.0 --port 8766
```

WebGPU 仅在 HTTPS 或 localhost 安全上下文可用。推荐在客户端执行端口转发：

```bash
ssh -N -L 8766:127.0.0.1:8766 <服务器登录信息>
```

然后在客户端浏览器打开：

```text
http://127.0.0.1:8766/
```

完整的无文本 ARDY Student 无限生成页面：

```text
http://127.0.0.1:8766/infinite_demo.html
```

该页面是完整动画循环，并复刻交互 demo 的默认自回归状态机：20 FPS、每次续写 40 帧、最近 4 帧 history、剩余 4 帧触发重规划、replan buffer 1 帧、鼠标 waypoint 位于当前后 60 帧。浏览器用 WebGPU 运行 FP32 compact history encoder、NFE=1 flow 和 compact decoder；两个很小的 WASM 图保留原版 root/FSQ/token 排布。部署固定为已验收的 FP32 student（全部 ONNX 为 91.01 MiB），不再暴露质量不可用的全 FP16 版本。8-frame inertialization 不改变时间、帧号、buffer 或 waypoint。

服务器可直接读取页面的调试状态，不上传动作 tensor：

```bash
curl -fsS http://127.0.0.1:8766/api/demo/status
```

服务器侧整链路回归（使用现有 Python ORT CPU，不模拟 WebGPU）：

```bash
/root/anaconda3/envs/py311/bin/python tools/validate_infinite_demo.py
```

结果写入 `infinite_demo/server_validation.json`。

服务器还可以直接控制已经打开的无限页；`enqueue` 仅接受服务器 localhost 请求：

```bash
curl -fsS -X POST \
  -H 'Content-Type: application/json' \
  --data '{"action":"benchmark"}' \
  http://127.0.0.1:8766/api/demo/enqueue

curl -fsS -X POST \
  -H 'Content-Type: application/json' \
  --data '{"action":"waypoint","command":{"x":2.0,"z":3.0,"frame_offset":60}}' \
  http://127.0.0.1:8766/api/demo/enqueue

curl -fsS -X POST \
  -H 'Content-Type: application/json' \
  --data '{"action":"start"}' \
  http://127.0.0.1:8766/api/demo/enqueue
```

允许的 action 是 `start`、`pause`、`restart`、`benchmark` 和 `waypoint`。状态、任务与最近报告统一从 `/api/demo/status` 读取。

页面出现 `WEBGPU_TOY_PASS` 表示原生 WebGPU 基础链路通过。页面显示“自动执行代理：在线”后，可在服务器本地入队：

```bash
curl -fsS -X POST \
  -H 'Content-Type: application/json' \
  --data '{"case_id":"ardy_decoder_fp32"}' \
  http://127.0.0.1:8766/api/control/enqueue

curl -fsS http://127.0.0.1:8766/api/control/status
```

`/api/control/enqueue` 仅接受服务器 localhost 请求。浏览器只领取已生成的固定用例，输出由服务器 NumPy 独立计算 max/mean absolute/relative error、cosine、NaN/Inf 并留档在 `results/`。

当前用例：

- `smoke_fp32`：Linear + GELU + residual + LayerNorm；
- `ardy_decoder_fp32`：ARDY Core FSQ decoder，10 tokens / 40 frames；
- `ardy_denoiser_fp32`：ARDY Core 双阶段 Transformer denoiser 单步、separated CFG、zero text、非零约束。
- `ardy_student_round3_step400_{encoder,flow,decoder}_fp32`：当前无限页的三张真实训练权重图；真实 Edge/NVIDIA WebGPU 数值和分模块延迟结果固化在 `infinite_demo/webgpu_module_validation.json`。

ONNX Runtime Web 1.27.0 的三份运行时静态文件保存在 `vendor/onnxruntime-web-1.27.0/`。它们是项目静态资源，不是 Python/npm 环境安装。
