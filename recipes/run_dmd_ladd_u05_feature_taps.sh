#!/usr/bin/env bash
set -euo pipefail

cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

STAGE1=distill_runs/text_control_joint_fullproj_resume20000_4gpu_b256_const5e6_until_plateau_step40000_ema995_20260715
FLOW="$STAGE1/weights/step-0022500/flow_ema.safetensors"
CODEC=distill_runs/first_12h_20260714_014643/codec_standard_e4_d8_4gpu_b256_step5e5_to1e5_100k/weights/step-0100000
SOURCE=distill_runs/dmd_ladd_critic_time_from_g200_joint22500_20260715/uniform_u05/state/step-0000300
ROOT=distill_runs/dmd_ladd_u05_feature_tap_from_g300_joint22500_20260716

export CUDA_VISIBLE_DEVICES=1,2,3,4
export PYTHONPATH=.:ardy
export OMP_NUM_THREADS=4

latest_state_or_source() {
  local output="$1"
  local latest=""
  if [[ -d "$output/state" ]]; then
    latest=$(find "$output/state" -mindepth 1 -maxdepth 1 -type d -name 'step-*' | sort | tail -n 1)
  fi
  if [[ -n "$latest" ]]; then
    printf '%s\n' "$latest"
  else
    printf '%s\n' "$SOURCE"
  fi
}

run_arm() {
  local tap="$1"
  local port="$2"
  local output="$ROOT/$tap"
  local resume
  resume=$(latest_state_or_source "$output")

  if [[ -f "$output/weights/step-0000500/flow_ema.safetensors" ]]; then
    echo "$tap already complete"
    return
  fi

  accelerate launch \
    --num_machines 1 \
    --num_processes 4 \
    --mixed_precision bf16 \
    --dynamo_backend no \
    --main_process_port "$port" \
    -m ardy_distill.train_flow_dmd2 \
    --data distill_data/text_control_v1/joint_teacher_train_524k \
    --text-features distill_data/text_control_v1/features/qwen \
    --heading-condition-features 3 \
    --output "$output" \
    --teacher-flow "$FLOW" \
    --generator "$FLOW" \
    --encoder "$CODEC/encoder_ema.safetensors" \
    --decoder "$CODEC/decoder_ema.safetensors" \
    --steps 500 \
    --batch-size 32 \
    --guidance-updates-per-generator 1 \
    --score-updates-per-generator 1 \
    --critic-updates-per-generator 1 \
    --warmup-guidance-updates 800 \
    --generator-learning-rate 1e-7 \
    --score-learning-rate 5e-6 \
    --critic-learning-rate 1e-6 \
    --override-learning-rates-on-resume \
    --weight-decay 0 \
    --mixed-precision bf16 \
    --frozen-codec-dtype fp16 \
    --ema-decay 0.995 \
    --generator-grad-clip 0.01 \
    --score-grad-clip 1 \
    --critic-grad-clip 1 \
    --flow-width 512 \
    --flow-heads 8 \
    --flow-trunk-blocks 8 \
    --flow-body-blocks 8 \
    --flow-root-smoothing-passes 0 \
    --critic-blocks 2 \
    --critic-feature-tap "$tap" \
    --encoder-width 512 \
    --encoder-blocks 4 \
    --decoder-width 512 \
    --decoder-blocks 8 \
    --decoder-token-hidden 32 \
    --codec-expansion 2 \
    --time-exact-t1-probability 0.7 \
    --time-high-noise-probability 0.2 \
    --critic-time-exact-t1-probability 0 \
    --critic-time-high-noise-probability 0 \
    --critic-time-upper-bound 0.5 \
    --adversarial-time-sampler score \
    --dmd-grad-clip 0 \
    --dmd-weight 1 \
    --adversarial-weight 0.001 \
    --paired-weight 0.1 \
    --fsq-weight 0.1 \
    --path-weight 0.01 \
    --decoder-weight 0.01 \
    --root-temporal-weight 0.01 \
    --quality-weight 0.01 \
    --seam-weight 0.001 \
    --physical-seam-weight 0.001 \
    --num-workers 2 \
    --cache-shards 2 \
    --seed 20260715 \
    --log-every 10 \
    --save-every 50 \
    --state-every 50 \
    --max-runtime-s 1200 \
    --resume "$resume"
}

# body_final is the historical tap. Every arm first moves score/D counters
# from 700 to 800 while generator remains at 300, then performs 200 generator
# updates. Omitting ratio-origin reset is intentional: the origin is anchored
# after the common recovery, not before it.
run_arm body_final 29721
run_arm body_pre 29722
run_arm body_mid 29723
run_arm trunk_final 29724
