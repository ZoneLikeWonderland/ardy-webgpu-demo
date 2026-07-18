#!/usr/bin/env bash
set -euo pipefail

cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ "$#" -ne 3 ]]; then
  echo "usage: $0 ROOT ARM STEP" >&2
  exit 2
fi

ROOT="$1"
ARM="$2"
STEP="$3"
CODEC=distill_runs/first_12h_20260714_014643/codec_standard_e4_d8_4gpu_b256_step5e5_to1e5_100k/weights/step-0100000
SEEDS=(20260714 20260715 20260716 20260717)
IFS=',' read -r -a GPUS <<< "${ARDY_ROLLOUT_GPUS:-1,2,3,6}"
if [[ "${#GPUS[@]}" -ne "${#SEEDS[@]}" ]]; then
  echo "ARDY_ROLLOUT_GPUS must contain exactly four comma-separated GPUs" >&2
  exit 2
fi

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

pids=()
for index in "${!SEEDS[@]}"; do
  seed="${SEEDS[$index]}"
  gpu="${GPUS[$index]}"
  output="$ROOT/rollout_step${OUTPUT_TAG}_seed${seed}/$ARM"
  mkdir -p "$output"
  [[ -f "$output/metrics.json" ]] && continue
  env CUDA_VISIBLE_DEVICES="$gpu" python \
    -m ardy_distill.tools.evaluate_rollout \
    --encoder "$CODEC/encoder_ema.safetensors" \
    --flow "$WEIGHT" \
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
  output="$ROOT/rollout_step${OUTPUT_TAG}_seed${seed}/$ARM"
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

echo "$ARM step=$STEP four-seed rollout+jitter complete"
