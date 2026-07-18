#!/usr/bin/env bash
set -euo pipefail

cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

ROOT=distill_runs/text_control_joint_fullproj_resume20000_4gpu_b256_const5e6_until_plateau_step40000_ema995_20260715
DATA=distill_data/text_control_v1/joint_teacher_val_16k
CODEC=distill_runs/first_12h_20260714_014643/codec_standard_e4_d8_4gpu_b256_step5e5_to1e5_100k/weights/step-0100000
STEPS=(32500 35000 37500 40000)
GPUS=(0 4 5 7)

export PYTHONPATH=.:ardy
export OMP_NUM_THREADS=4

evaluate_full() {
  local step="$1"
  local gpu="$2"
  local output_tag weight_tag
  output_tag="$(printf '%06d' "$step")"
  weight_tag="$(printf '%07d' "$step")"
  local output="$ROOT/eval/step${output_tag}_ema_full_fp16.json"
  [[ -f "$output" ]] && return 0
  env CUDA_VISIBLE_DEVICES="$gpu" python \
    -m ardy_distill.evaluate \
    --data "$DATA" \
    --encoder "$CODEC/encoder_ema.safetensors" \
    --decoder "$CODEC/decoder_ema.safetensors" \
    --flow "$ROOT/weights/step-$weight_tag/flow_ema.safetensors" \
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
  local step="$1"
  local gpu="$2"
  local output_tag weight_tag
  output_tag="$(printf '%06d' "$step")"
  weight_tag="$(printf '%07d' "$step")"
  local output="$ROOT/eval/step${output_tag}_ema_text_control_fp16.json"
  [[ -f "$output" ]] && return 0
  env CUDA_VISIBLE_DEVICES="$gpu" python \
    -m ardy_distill.tools.evaluate_text_control \
    --data "$DATA" \
    --flow "$ROOT/weights/step-$weight_tag/flow_ema.safetensors" \
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
  for index in "${!STEPS[@]}"; do
    step="${STEPS[$index]}"
    gpu="${GPUS[$index]}"
    if [[ "$mode" == full ]]; then
      evaluate_full "$step" "$gpu" >"$ROOT/eval/step$(printf '%06d' "$step")_ema_full_fp16.log" 2>&1 &
    else
      evaluate_text "$step" "$gpu" >"$ROOT/eval/step$(printf '%06d' "$step")_ema_text_control_fp16.log" 2>&1 &
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
