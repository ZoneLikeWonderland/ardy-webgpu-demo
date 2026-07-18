#!/usr/bin/env bash
set -euo pipefail

cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

STAGE1=distill_runs/text_control_joint_fullproj_resume20000_4gpu_b256_const5e6_until_plateau_step40000_ema995_20260715
DMD_ROOT=distill_runs/dmd_ladd_lr_round1_joint22500_warm200_g200_20260715
CODEC=distill_runs/first_12h_20260714_014643/codec_standard_e4_d8_4gpu_b256_step5e5_to1e5_100k/weights/step-0100000
ROLLOUT_ROOT="$DMD_ROOT/rollout_seed20260715"

export PYTHONPATH=.:ardy
export OMP_NUM_THREADS=4

TAGS=(baseline g1e8 g1e7 g3e7)
GPUS=(0 4 5 7)
FLOWS=(
  "$STAGE1/weights/step-0022500/flow_ema.safetensors"
  "$DMD_ROOT/g1e8/weights/step-0000200/flow_ema.safetensors"
  "$DMD_ROOT/g1e7/weights/step-0000200/flow_ema.safetensors"
  "$DMD_ROOT/g3e7/weights/step-0000200/flow_ema.safetensors"
)

pids=()
for index in "${!TAGS[@]}"; do
  tag="${TAGS[$index]}"
  gpu="${GPUS[$index]}"
  flow="${FLOWS[$index]}"
  output="$ROLLOUT_ROOT/$tag"
  mkdir -p "$output"
  if [[ -f "$output/metrics.json" ]]; then
    echo "skip completed rollout: $tag"
    continue
  fi
  echo "start rollout: $tag gpu=$gpu"
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
    --seed 20260715 \
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

status=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    status=1
  fi
done
exit "$status"
