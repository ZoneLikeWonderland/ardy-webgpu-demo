#!/usr/bin/env bash
set -euo pipefail

cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ "$#" -ne 4 ]]; then
  echo "usage: $0 ROOT ARM STEP PHYSICAL_GPU" >&2
  exit 2
fi

ROOT="$1"
ARM="$2"
STEP="$3"
GPU="$4"
DATA=distill_data/text_control_v1/joint_teacher_val_16k
CODEC=distill_runs/first_12h_20260714_014643/codec_standard_e4_d8_4gpu_b256_step5e5_to1e5_100k/weights/step-0100000

[[ "$STEP" =~ ^[1-9][0-9]*$ ]] || {
  echo "STEP must be a positive integer" >&2
  exit 2
}
WEIGHT_TAG=$(printf '%07d' "$STEP")
OUTPUT_TAG=$(printf '%06d' "$STEP")
WEIGHT="$ROOT/$ARM/weights/step-$WEIGHT_TAG/flow_ema.safetensors"
[[ -f "$WEIGHT" ]] || {
  echo "missing weight: $WEIGHT" >&2
  exit 2
}

export PYTHONPATH=.:ardy
export OMP_NUM_THREADS=4
mkdir -p "$ROOT/$ARM/eval"

FULL="$ROOT/$ARM/eval/step${OUTPUT_TAG}_ema_full_fp16.json"
if [[ ! -f "$FULL" ]]; then
  env CUDA_VISIBLE_DEVICES="$GPU" python \
    -m ardy_distill.evaluate \
    --data "$DATA" \
    --encoder "$CODEC/encoder_ema.safetensors" \
    --decoder "$CODEC/decoder_ema.safetensors" \
    --flow "$WEIGHT" \
    --text-features distill_data/text_control_v1/features/qwen \
    --heading-condition-features 3 \
    --output "$FULL" \
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
fi

TEXT="$ROOT/$ARM/eval/step${OUTPUT_TAG}_ema_text_control_fp16.json"
if [[ ! -f "$TEXT" ]]; then
  env CUDA_VISIBLE_DEVICES="$GPU" python \
    -m ardy_distill.tools.evaluate_text_control \
    --data "$DATA" \
    --flow "$WEIGHT" \
    --encoder "$CODEC/encoder_ema.safetensors" \
    --text-features distill_data/text_control_v1/features/qwen \
    --prompt-bank distill_data/text_control_v1/prompt_bank.jsonl \
    --output "$TEXT" \
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
fi

echo "$ARM step=$STEP static+text complete"

