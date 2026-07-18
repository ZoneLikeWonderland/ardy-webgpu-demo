#!/usr/bin/env bash
set -euo pipefail

cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

ROOT=distill_runs/dmd_ladd_lr_round1_joint22500_warm200_g200_20260715
DATA=distill_data/text_control_v1/joint_teacher_val_16k
CODEC=distill_runs/first_12h_20260714_014643/codec_standard_e4_d8_4gpu_b256_step5e5_to1e5_100k/weights/step-0100000
TAGS=(g1e8 g1e7 g3e7 g1e6)
GPUS=(0 4 5 7)

evaluate_arm() {
  local tag="$1"
  local gpu="$2"
  local arm="$ROOT/$tag"
  local flow="$arm/weights/step-0000200/flow_ema.safetensors"
  if [[ ! -f "$flow" ]]; then
    echo "missing final EMA: $flow" >&2
    return 1
  fi
  mkdir -p "$arm/eval"
  env CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH=.:ardy \
    python -m ardy_distill.evaluate \
      --data "$DATA" \
      --encoder "$CODEC/encoder_ema.safetensors" \
      --decoder "$CODEC/decoder_ema.safetensors" \
      --flow "$flow" \
      --text-features distill_data/text_control_v1/features/qwen \
      --heading-condition-features 3 \
      --output "$arm/eval/step000200_ema_full_fp16.json" \
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
  env CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH=.:ardy \
    python -m ardy_distill.tools.evaluate_text_control \
      --data "$DATA" \
      --flow "$flow" \
      --encoder "$CODEC/encoder_ema.safetensors" \
      --text-features distill_data/text_control_v1/features/qwen \
      --prompt-bank distill_data/text_control_v1/prompt_bank.jsonl \
      --output "$arm/eval/step000200_ema_text_control_fp16.json" \
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

pids=()
for index in "${!TAGS[@]}"; do
  evaluate_arm "${TAGS[$index]}" "${GPUS[$index]}" &
  pids+=("$!")
done

failed=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    failed=1
  fi
done
exit "$failed"
