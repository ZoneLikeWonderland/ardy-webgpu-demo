#!/usr/bin/env bash
set -euo pipefail

cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

STAGE1=distill_runs/text_control_joint_fullproj_resume20000_4gpu_b256_const5e6_until_plateau_step40000_ema995_20260715
FLOW="$STAGE1/weights/step-0022500/flow_ema.safetensors"
CODEC=distill_runs/first_12h_20260714_014643/codec_standard_e4_d8_4gpu_b256_step5e5_to1e5_100k/weights/step-0100000
SOURCE=distill_runs/dmd_ladd_lr_round1_joint22500_warm200_g200_20260715/g1e7
ROOT=distill_runs/dmd_ladd_critic_time_from_g200_joint22500_20260715

export CUDA_VISIBLE_DEVICES=0,4,5,7
export PYTHONPATH=.:ardy
export OMP_NUM_THREADS=4

run_arm() {
  local tag="$1"
  local critic_exact_t1="$2"
  local critic_high_noise="$3"
  local critic_upper_bound="$4"
  local port="$5"
  local output="$ROOT/$tag"

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
    --steps 300 \
    --batch-size 32 \
    --guidance-updates-per-generator 1 \
    --warmup-guidance-updates 600 \
    --generator-learning-rate 1e-7 \
    --score-learning-rate 5e-6 \
    --critic-learning-rate 1e-6 \
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
    --encoder-width 512 \
    --encoder-blocks 4 \
    --decoder-width 512 \
    --decoder-blocks 8 \
    --decoder-token-hidden 32 \
    --codec-expansion 2 \
    --time-exact-t1-probability 0.7 \
    --time-high-noise-probability 0.2 \
    --critic-time-exact-t1-probability "$critic_exact_t1" \
    --critic-time-high-noise-probability "$critic_high_noise" \
    --critic-time-upper-bound "$critic_upper_bound" \
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
    --save-every 100 \
    --state-every 100 \
    --max-runtime-s 1800 \
    --resume "$SOURCE/state/step-0000200"
}

# All arms receive 200 additional score/critic-only updates (guidance 400->600)
# before generator updates resume. This isolates critic-time design from recovery time.
run_arm legacy_t1p70 0.7 0.2 1.0 29648
run_arm uniform_u09 0 0 0.9 29649
run_arm high50_u09 0 0.5 0.9 29650
run_arm uniform_u05 0 0 0.5 29651
