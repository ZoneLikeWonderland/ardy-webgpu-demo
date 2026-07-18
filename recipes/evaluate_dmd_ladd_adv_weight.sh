#!/usr/bin/env bash
set -euo pipefail

cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

ROOT=${DMD_EVAL_ROOT:-distill_runs/dmd_ladd_adv_weight_from_g200_joint22500_20260715}
DATA=distill_data/text_control_v1/joint_teacher_val_16k
CODEC=distill_runs/first_12h_20260714_014643/codec_standard_e4_d8_4gpu_b256_step5e5_to1e5_100k/weights/step-0100000
if [[ -n "${DMD_EVAL_ARMS:-}" ]]; then
  read -r -a ARMS <<<"$DMD_EVAL_ARMS"
else
  ARMS=(adv0 adv3e4 adv3e3 adv1e2)
fi
[[ "${#ARMS[@]}" -eq 4 ]] || {
  echo "DMD_EVAL_ARMS must contain exactly four whitespace-separated arms" >&2
  exit 2
}
if [[ -n "${DMD_EVAL_GPUS:-}" ]]; then
  read -r -a GPUS <<<"$DMD_EVAL_GPUS"
else
  GPUS=(0 4 5 7)
fi
[[ "${#GPUS[@]}" -eq 4 ]] || {
  echo "DMD_EVAL_GPUS must contain exactly four whitespace-separated GPU ids" >&2
  exit 2
}
STEP=${DMD_EVAL_STEP:-300}
[[ "$STEP" =~ ^[1-9][0-9]*$ ]] || {
  echo "DMD_EVAL_STEP must be a positive integer" >&2
  exit 2
}
WEIGHT_TAG="$(printf '%07d' "$STEP")"
OUTPUT_TAG="$(printf '%06d' "$STEP")"
SEEDS=(20260714 20260715 20260716 20260717)

export PYTHONPATH=.:ardy
export OMP_NUM_THREADS=4

for arm in "${ARMS[@]}"; do
  mkdir -p "$ROOT/$arm/eval"
done

evaluate_full() {
  local arm="$1"
  local gpu="$2"
  local output="$ROOT/$arm/eval/step${OUTPUT_TAG}_ema_full_fp16.json"
  [[ -f "$output" ]] && return 0
  env CUDA_VISIBLE_DEVICES="$gpu" python \
    -m ardy_distill.evaluate \
    --data "$DATA" \
    --encoder "$CODEC/encoder_ema.safetensors" \
    --decoder "$CODEC/decoder_ema.safetensors" \
    --flow "$ROOT/$arm/weights/step-$WEIGHT_TAG/flow_ema.safetensors" \
    --text-features distill_data/text_control_v1/features/qwen \
    --heading-condition-features 3 \
    --output "$output" \
    --batch-size 256 \
    --device cuda:0 \
    --model-dtype fp16 \
    --flow-history student \
    --flow-width 512 \
    --flow-heads 8 \
    --flow-trunk-blocks 8 \
    --flow-body-blocks 8 \
    --flow-steps 1 \
    --flow-root-smoothing-passes 0 \
    --student-history-quantization fsq
}

evaluate_text() {
  local arm="$1"
  local gpu="$2"
  local output="$ROOT/$arm/eval/step${OUTPUT_TAG}_ema_text_control_fp16.json"
  [[ -f "$output" ]] && return 0
  env CUDA_VISIBLE_DEVICES="$gpu" python \
    -m ardy_distill.tools.evaluate_text_control \
    --data "$DATA" \
    --flow "$ROOT/$arm/weights/step-$WEIGHT_TAG/flow_ema.safetensors" \
    --encoder "$CODEC/encoder_ema.safetensors" \
    --text-features distill_data/text_control_v1/features/qwen \
    --prompt-bank distill_data/text_control_v1/prompt_bank.jsonl \
    --output "$output" \
    --device cuda:0 \
    --model-dtype fp16 \
    --batch-size 256 \
    --num-workers 4 \
    --flow-width 512 \
    --flow-heads 8 \
    --flow-trunk-blocks 8 \
    --flow-body-blocks 8 \
    --flow-steps 1 \
    --heading-condition-features 3
}

for mode in full text; do
  pids=()
  for index in "${!ARMS[@]}"; do
    arm="${ARMS[$index]}"
    gpu="${GPUS[$index]}"
    if [[ "$mode" == full ]]; then
      evaluate_full "$arm" "$gpu" >"$ROOT/$arm/eval/step${OUTPUT_TAG}_ema_full_fp16.log" 2>&1 &
    else
      evaluate_text "$arm" "$gpu" >"$ROOT/$arm/eval/step${OUTPUT_TAG}_ema_text_control_fp16.log" 2>&1 &
    fi
    pids+=("$!")
  done
  failed=0
  for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
      failed=1
    fi
  done
  [[ "$failed" -eq 0 ]] || exit "$failed"
done

for arm in "${ARMS[@]}"; do
  pids=()
  for index in "${!SEEDS[@]}"; do
    seed="${SEEDS[$index]}"
    gpu="${GPUS[$index]}"
    output="$ROOT/rollout_seed${seed}/$arm"
    mkdir -p "$output"
    [[ -f "$output/metrics.json" ]] && continue
    CUDA_VISIBLE_DEVICES="$gpu" python \
      -m ardy_distill.tools.evaluate_rollout \
      --encoder "$CODEC/encoder_ema.safetensors" \
      --flow "$ROOT/$arm/weights/step-$WEIGHT_TAG/flow_ema.safetensors" \
      --decoder "$CODEC/decoder_ema.safetensors" \
      --output-dir "$output" \
      --device cuda:0 \
      --model-dtype fp16 \
      --windows 50 \
      --checkpoints 1 5 20 50 \
      --log-every 50 \
      --seed "$seed" \
      --flow-width 512 \
      --flow-heads 8 \
      --flow-trunk-blocks 8 \
      --flow-body-blocks 8 \
      --flow-steps 1 \
      --text-feature-dim 7680 \
      --heading-condition-features 3 \
      --encoder-width 512 \
      --encoder-blocks 4 \
      --decoder-width 512 \
      --decoder-blocks 8 \
      --decoder-token-hidden 32 \
      --codec-expansion 2 \
      >"$output/run.log" 2>&1 &
    pids+=("$!")
  done
  failed=0
  for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
      failed=1
    fi
  done
  [[ "$failed" -eq 0 ]] || exit "$failed"

  pids=()
  for seed in "${SEEDS[@]}"; do
    output="$ROOT/rollout_seed${seed}/$arm"
    [[ -f "$output/jitter.json" ]] && continue
    python \
      -m ardy_distill.tools.analyze_rollout_jitter \
      --raw-cases "$output/fixed_cases.safetensors" \
      --inertial-cases "$output/fixed_cases.safetensors" \
      --rollout-metrics "$output/metrics.json" \
      --output "$output/jitter.json" \
      >"$output/jitter.log" 2>&1 &
    pids+=("$!")
  done
  failed=0
  for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
      failed=1
    fi
  done
  [[ "$failed" -eq 0 ]] || exit "$failed"
done
