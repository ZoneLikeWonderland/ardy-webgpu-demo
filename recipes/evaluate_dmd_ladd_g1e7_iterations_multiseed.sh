#!/usr/bin/env bash
set -euo pipefail

cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

ROOT=distill_runs/dmd_ladd_iters_g1e7_resume200_to1000_joint22500_20260715
CODEC=distill_runs/first_12h_20260714_014643/codec_standard_e4_d8_4gpu_b256_step5e5_to1e5_100k/weights/step-0100000
STEPS=(300 500 700 1000)
GPUS=(0 4 5 7)
SEEDS=(20260714 20260715 20260716 20260717)

export PYTHONPATH=.:ardy
export OMP_NUM_THREADS=4

for step in "${STEPS[@]}"; do
  tag="g$step"
  weight_tag="$(printf '%07d' "$step")"
  flow="$ROOT/weights/step-$weight_tag/flow_ema.safetensors"
  pids=()
  echo "start seed wave: $tag"
  for seed_index in "${!SEEDS[@]}"; do
    seed="${SEEDS[$seed_index]}"
    gpu="${GPUS[$seed_index]}"
    output="$ROOT/rollout_seed${seed}/$tag"
    mkdir -p "$output"
    if [[ -f "$output/metrics.json" ]]; then
      echo "skip completed rollout: $tag seed=$seed"
      continue
    fi
    CUDA_VISIBLE_DEVICES="$gpu" python \
      -m ardy_distill.tools.evaluate_rollout \
      --encoder "$CODEC/encoder_ema.safetensors" \
      --flow "$flow" \
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
done
