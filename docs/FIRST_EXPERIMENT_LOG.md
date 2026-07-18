# First bounded ARDY WebGPU distillation experiment

- Start audit: 2026-07-14 01:46:43 CST
- Hard maximum runtime: 12 hours
- Python: `<PY311_ENV>/bin/python`
- No packages installed
- GPU selection immediately before launch, ordered by free memory: 6, 4, 3, 5
- Selection snapshot (MiB free): GPU6 48557, GPU4 42359, GPU3 41241, GPU5 38125
- Teacher: released FP32 ARDY Core, 10 DDIM steps, text feature zero, constraint CFG 1.5
- Train data: 8192 windows total, four independent ranks, FP32 safetensors
- Validation data: 1024 windows total, independent seed, FP32 safetensors
- Codec: BF16 autocast, FP32 EMA, 4 GPUs
- Flow: BF16 autocast, FP32 EMA, continuous high-noise-biased flow + explicit t=1 endpoint, 4 GPUs
- Final target: NFE=1, no CFG, encoder+flow+decoder ONNX weights <100 MB

This is the first bounded convergence/quality experiment. DMD2-like distribution matching and
temporal adversarial refinement start only after the supervised codec/flow baseline has measurable
quality; they are not mixed into the first optimizer step.

## Codec scheduler incident

- `codec/` is an invalidated trial and must not be used for evaluation or downstream flow training.
- Cause: Accelerate's prepared scheduler advanced once per process while the script also stepped it
  explicitly, compressing the intended 2000-step schedule by 4x.
- The run was stopped at about optimizer step 980; its step-500 checkpoint is retained only as an
  audit artifact.
- Both codec and flow trainers now use `step_scheduler_with_optimizer=False`; a four-GPU smoke test
  verified the intended LR sequence exactly.
- The clean restart is written to `codec_v2/` and starts from random initialization.

## Codec v2 result

- `codec_v2/` completed 2000 optimizer steps with the corrected schedule on GPUs 6,4,3,5.
- LR audit passed: step 20 `2e-5`, step 100 `1e-4`, step 200 `2e-4`, step 2000 `1e-5`.
- Independent 1024-window validation is stored under `eval/codec_v2_{raw,ema}.json`.
- Raw decoder was slightly better than the short-horizon EMA: FK-MPJPE `0.03463 m`, rotation
  geodesic `0.12546 rad`, body L1 `0.18046`; it is selected for the first flow baseline.
- Encoder validation remains provisional: latent L1 `0.12322`, requantized L1 `0.12102`, and exact
  quantization-bin accuracy `25.94%`. Long autoregressive rollout error is required before acceptance.
- First flow baseline is `flow_v1/`: 4000 steps, BF16 training, FP32 EMA, NFE=1 endpoint objective.

## Flow v1 rejection and v2 probe

- `flow_v1/` is a rejected flattened-MLP baseline, retained only for comparison.
- It overfit 8192 teacher windows: training endpoint root/body MSE reached about `0.0033/0.564`,
  while independent validation was about `0.497/1.563`; validation FK-MPJPE was about `1.10 m`.
- Train/validation distribution audits matched closely, so the failure is not a split-generation bug.
- v2 preserves fixed temporal structure: 1 global + 1 history + 16 path + 10 generation tokens,
  four shared attention blocks, root prediction, then two root-conditioned body blocks.
- v2 flow has 7,302,676 parameters (down from 22,189,000); total student parameters are 16,142,676.
- Random-weight ONNX is 29.22 MB FP32 / 14.63 MB FP16. Real Edge/NVIDIA WebGPU parity passed;
  flow median latency was 73.11 ms FP32 / 54.34 ms FP16. With the previously measured codec,
  projected FP16 total is about 110 ms and 32.32 MB.
- `flow_v2_8k_probe/` is a bounded sample-efficiency probe before spending time and disk on a much
  larger teacher corpus. It must pass independent validation before any browser integration.

## Flow v2 8k probe result

- At step 1000, independent validation root/body endpoint MSE was `0.18275/0.76062`,
  FK-MPJPE `0.56999 m`, rotation geodesic `0.54397 rad`, and constrained path error `0.57644 m`.
- This is substantially better than v1 and the train/validation gap is small, validating the
  temporal architecture, but animation quality is still unacceptable.
- The next corpus `teacher_train_large_v2/` contains 131,072 windows (32,768 per rank), uses the
  same released FP32 teacher semantics, and stores tensors in FP16 to control disk use. Existing
  independent FP32 `teacher_val/` remains the selection set.

## Large teacher corpus validation

- `teacher_train_large_v2/` completed on four ranks with 131,072 windows in 256 safetensors
  shards. Teacher inference stayed FP32; only persisted tensors were cast to FP16.
- Full validation, including every shard hash, schema, all 15 tensor shapes/dtypes, and finite
  checks, passed. The corpus is 9,700,210,688 bytes, 87.5% continuation windows and 74.913%
  constrained windows.
- Initial noise mean/std are `-0.0001872/0.9999473`; clean endpoint mean/std are
  `-0.0208610/0.9433964`. The machine-readable audit is `teacher_train_large_v2/validation.json`.

## Codec v3 large result

- `codec_v3_large/` warm-started from the selected codec-v2 raw weights and completed 4000
  optimizer steps, global batch 128, BF16 autocast and FP32 EMA on GPUs 6,4,3,5.
- Trainer-reported wall time was 132.30 s. No runtime cutoff was hit and checkpoints were saved at
  steps 1000, 2000, 3000, and 4000.
- Every raw and EMA checkpoint was evaluated on the independent 1024-window FP32 validation set.
  Results improved monotonically through step 4000. The selected deployment pair deliberately
  mixes the best variants: step-4000 encoder EMA and step-4000 decoder raw.

| selected codec metric | value |
|---|---:|
| encoder latent L1 | 0.07993 |
| encoder requantized latent L1 | 0.07629 |
| encoder exact FSQ-bin accuracy | 39.09% |
| decoder body L1 | 0.12644 |
| decoder rotation geodesic | 0.08969 rad |
| decoder FK-MPJPE | 0.02307 m |
| decoder foot slide | 0.02097 |

- Against codec v2, decoder FK-MPJPE improved from 0.03463 m to 0.02307 m and encoder
  requantized L1 improved from 0.12102 to 0.07629. The compact codec is accepted as the next flow
  baseline, but long autoregressive rollout remains a final acceptance requirement.

## Flow v2 large result

- Before the formal run, a two-step four-GPU smoke verified deployment-style student encoder
  history, decoder/FK quality losses, DDP backward, finite gradients, raw/EMA saving, and static
  shapes.
- `flow_v2_large/` completed 4000 optimizer steps, global batch 128, BF16 autocast, FP32 EMA,
  90% student-encoded history and 10% teacher-history regularization. Trainer-reported wall time
  was 210.88 s.
- Selection used the independent FP32 validation set with `flow_history=student`, matching the
  browser. Raw checkpoints were better than or equal to EMA; step-4000 raw was selected.

| raw checkpoint | root MSE | body MSE | FK-MPJPE | rotation | foot slide | constrained path error |
|---:|---:|---:|---:|---:|---:|---:|
| 1000 | 0.15267 | 0.72622 | 0.48042 m | 0.50967 rad | 0.41600 | 0.44224 m |
| 2000 | 0.11727 | 0.65362 | 0.38836 m | 0.46312 rad | 0.28671 | 0.33757 m |
| 3000 | 0.10316 | 0.62060 | 0.33522 m | 0.43646 rad | 0.23317 | 0.28652 m |
| 4000 | 0.09803 | 0.61151 | 0.32338 m | 0.42772 rad | 0.21908 | 0.27283 m |

- Step-4000 EMA was slightly worse: FK-MPJPE 0.32427 m and path error 0.27284 m.
- Re-evaluating step-4000 raw with exact teacher history changed FK-MPJPE only from 0.32338 m to
  0.32310 m and path error from 0.27283 m to 0.27364 m. Encoder approximation is therefore not
  the limiting factor.
- Arithmetic precision was evaluated on the same 1024 windows with all student modules actually
  running in the requested dtype and metrics accumulated in FP32. FP16 matched FP32 closely:
  FK-MPJPE `0.323383` vs `0.323380 m`, path error `0.272752` vs `0.272826 m`, and decoder-only FK
  differed by less than `5e-8 m`. BF16 gave FK `0.323497 m` and path error `0.273955 m`.
- Therefore this baseline's large motion error is not caused by FP16 arithmetic. Browser FP16
  parity is still a separate gate; server CUDA precision agreement does not replace WebGPU.
- Large-data v2 improves strongly over the 8k probe (FK 0.570 -> 0.323 m; path 0.576 -> 0.273 m),
  but it is not accepted for final animation integration. The compact decoder's independent FK
  error is only 0.023 m, so the remaining quality bottleneck is the NFE=1 flow mapping.

## Current selection and next controlled stage

- Frozen codec baseline:
  `codec_v3_large/weights/step-0004000/encoder_ema.safetensors` and
  `codec_v3_large/weights/step-0004000/decoder.safetensors`.
- Best supervised flow baseline, diagnostic only:
  `flow_v2_large/weights/step-0004000/flow.safetensors`.
- Selected trained graphs contain 16,142,676 parameters. Directly exported ONNX totals are
  64,644,155 B FP32 and 32,358,603 B FP16.
- The codec is fast/small enough and no longer the dominant error source, so the next experiment
  should spend the available size budget on flow capacity and endpoint/distribution refinement,
  not re-distill encoder/decoder blindly.
- Two capacity candidates were exported before training, both using the same 28-token operator
  family: width-512/8-head/4+2 blocks is 12,882,580 parameters and 25,787,457 B FP16 ONNX;
  width-512/8-head/5+2 blocks is 14,985,364 parameters and 29,994,808 B FP16 ONNX. Including the
  selected codec, totals are 43,494,751 B and 47,702,102 B. CPU ORT parity passed; browser jobs are
  queued and neither candidate will be trained until the WebGPU latency screen returns.
- A final browser front end must not be declared ready until trained-weight WebGPU parity,
  front-tab end-to-end latency, fixed visual cases, and 1/5/20/50-window autoregressive rollouts
  pass. DMD2-like distribution matching and a temporal adversarial critic remain the next quality
  stage; paired path/history/seam losses must remain active to prevent condition collapse.
- No package was installed during corpus generation, training, export, or evaluation.

## Faithful runtime and codec latency re-audit

- Added `ardy_distill/student_runtime.py` to compose history recentering, student encoder, exact FSQ,
  local path preprocessing, NFE=1 flow, world-root restoration, and the static 11-token decoder.
  Initial and continuation calls return `[1,40,330]` and `[1,44,330]` respectively.
- Exact release-model trace comparison passed with max absolute error `0` for path condition, first
  heading, history flag, world translation, and history root in both modes. The machine-readable
  result is `eval/student_runtime_semantics.json`.
- A semantic audit found that the old flow-v2 trainer passed continuous student encoder output to
  flow, while released `_encode_init_history` uses FSQ-quantized history. Training/evaluation now
  requantize by default. The old step-4000 flow was re-evaluated rather than silently relabelled:
  faithful-FSQ FP32 FK/path are `0.323774/0.272751 m`; FP16 gives `0.323804/0.272633 m`. The tiny
  delta shows this was not the main quality bottleneck, but all future flow runs must use FSQ.
- Trained runtime timing on an A6000, batch 1, 20 warmups and 100 repeats:
  continuation FP32 p50/p95 is encoder `0.488/0.504 ms`, decoder `1.265/1.359 ms`, flow
  `2.624/2.798 ms`, total `6.384/6.586 ms`; FP16 is `0.445/0.454`, `1.321/1.368`,
  `2.837/3.005`, and `6.623/6.856 ms`. Files are `bench/student_runtime_{fp32,fp16}.json`.
- CUDA says codec compute is small, but the prior Edge WebGPU medians remain encoder
  `14.37/16.90 ms` and decoder `34.94/38.78 ms` for FP32/FP16. Codec v3 therefore remains the
  quality baseline while smaller graphs are latency-screened before any retraining.
- Conservative codec candidate (encoder width384/2 blocks, decoder width384/3 blocks) has
  1,733,120 + 2,336,173 parameters and `8,147,992 B` total FP16 ONNX. Aggressive width256/2+3
  has 893,312 + 1,164,677 parameters and `4,129,479 B` FP16. All eight FP32/FP16 module graphs
  passed ONNX checker and CPU ORT parity and are queued for real WebGPU timing. The browser worker
  is currently offline, so no browser latency claim has been made.
- No package was installed for this re-audit.

## Faithful 1/5/20/50-window rollout and codec responsibility split

- Added `ardy_distill/tools/evaluate_rollout.py`. It reproduces the released UI's 20 FPS,
  four-frame replan threshold, four-frame history crop, 40-frame generation horizon, and exact
  append rule. Buffer sizes are `40, 77, 114, ...`; teacher and student share the same world-space
  waypoint and initial noise at each window. Fixed tensors and timeline events are saved beside
  every `metrics.json`.
- With the provisional flow and codec-v3, FP32 root drift mean/final grows from `0.236/0.377 m`
  at one window to `1.347/1.696 m` at five, `11.453/17.068 m` at twenty, and
  `17.642/43.491 m` at fifty. The corresponding FK errors are `0.374`, `1.369`, `11.531`, and
  `17.683 m`; 50-window waypoint mean is `18.146 m`. FP16 gives essentially the same failure
  (`17.619/43.000 m` root mean/final and `18.118 m` waypoint mean), so low precision is not the
  cause.
- Oracle-flow codec ablation feeds the teacher clean generation/root through the student codec
  while retaining closed-loop student-decoded body history. At 1/5/20/50 windows its FK error is
  `0.0135/0.0145/0.0326/0.0229 m`; root drift remains exactly zero and waypoint error remains the
  teacher value (`0.0105 m` mean at fifty). This isolates the meter-scale collapse to flow.
- Codec is not fully perfect: at fifty oracle-flow windows, foot slide is `0.332 m/s` versus the
  teacher's `0.064`, and joint seam velocity jump is `0.874 m/s` versus `0.257`. Any codec
  retraining therefore needs explicit foot-contact/slide and seam objectives, not only latent or
  single-frame FK losses.
- Decision: the original encoder/decoder remain rejected because their Edge p50 is about 102 ms
  combined and their FP32 ONNX files are about 142.1 MB. Codec-v3 is retained as the quality
  baseline, but its 35--39 ms WebGPU decoder is not automatically accepted. Retrain the width-384
  or width-256 codec only if the pending real-browser latency screen shows a material improvement
  toward the roughly 25 ms codec budget; otherwise preserve codec quality and prioritize flow.
- Records are under `rollout/faithful_50_{fp32,fp16}*/`. The browser worker remains offline with
  18 jobs pending, so no candidate-codec WebGPU latency has been claimed. No package was installed.

## DMD2-like two-time-scale and temporal-adversarial smoke

- Added `ardy_distill/dmd2.py`, `models/critic.py`, `train_dmd2.py`, and
  `tools/validate_teacher_score_adapter.py`. The implementation follows the core update in the
  official [DMD2 paper](https://arxiv.org/abs/2405.14867) and
  [reference code](https://github.com/tianweiy/DMD2), while retaining paired endpoint, path,
  decoder, FK/rotation/velocity/foot, and seam objectives needed for conditional motion.
- Fake score predicts epsilon on noised, detached generator samples every iteration. Frozen ARDY
  predicts real x0; fake epsilon is converted to fake x0; generator receives the normalized
  `(p_real - p_fake)` stop-gradient surrogate. The fake score and critic update every iteration,
  while generator defaults to every fifth iteration.
- The conditional critic consumes 40-frame normalized root/body, physical FK joints, joint
  velocity, and foot contacts through dilated temporal blocks and native,/2,/4 summaries. History,
  path, heading and history-valid state are explicit conditions. Default critic size is 3,323,649
  parameters; fake score is 7,302,676. Both are training-only and do not affect browser size.
- Exact teacher adapter validation replayed all ten stored DDIM transitions on a 16-sample batch
  containing eight initial and eight continuation windows. Worst max absolute error was `7.03e-6`
  and the mean was `4.42e-7`; record: `eval/teacher_score_adapter.json`.
- A one-GPU BF16 engineering smoke ran one warmup iteration and three generator updates with every
  loss/optimizer/scheduler/gradient clip/EMA path active. It resumed from the saved step-2 state
  and reached step 3; generator, EMA, fake score and critic safetensors are all finite. Records are
  in `dmd2_smoke/`. This proves execution and resume semantics only, not quality improvement.
- Formal training remains gated on real-browser architecture selection, then faithful-FSQ
  supervised fitting before DMD2+adv refinement and long rollout evaluation. All 18 browser jobs
  are still pending because the worker remains offline. No package was installed.
