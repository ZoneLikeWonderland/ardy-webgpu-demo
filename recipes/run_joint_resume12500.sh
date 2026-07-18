#!/usr/bin/env bash
set -euo pipefail

cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

OUTPUT=distill_runs/text_control_joint_fullproj_resume5000_4gpu_b256_const1e5_until_plateau_step20000_ema995_20260715
STATE="$OUTPUT/state/step-0012500"
CODEC=distill_runs/first_12h_20260714_014643/codec_standard_e4_d8_4gpu_b256_step5e5_to1e5_100k/weights/step-0100000

export CUDA_VISIBLE_DEVICES=1,2,3,6
export PYTHONPATH=.:ardy
export OMP_NUM_THREADS=4

exec accelerate launch \
  --num_machines 1 \
  --num_processes 4 \
  --mixed_precision bf16 \
  --main_process_port 29617 \
  -m ardy_distill.train_flow \
  --data distill_data/text_control_v1/joint_teacher_train_524k \
  --text-features distill_data/text_control_v1/features/qwen \
  --heading-condition-features 3 \
  --output "$OUTPUT" \
  --decoder "$CODEC/decoder_ema.safetensors" \
  --encoder "$CODEC/encoder_ema.safetensors" \
  --teacher-history-prob 0 \
  --checkpoint-dir ardy/checkpoints/ARDY-Core-RP-20FPS-Horizon40 \
  --steps 20000 \
  --batch-size 256 \
  --gradient-accumulation 1 \
  --learning-rate 1e-5 \
  --condition-learning-rate 1e-5 \
  --lr-schedule constant \
  --lr-decay-step 1000000 \
  --final-learning-rate 1e-5 \
  --final-condition-learning-rate 1e-5 \
  --base-gradient-clip 1 \
  --condition-gradient-clip 1 \
  --weight-decay 0 \
  --warmup-steps 0 \
  --mixed-precision bf16 \
  --frozen-codec-dtype fp16 \
  --ema-decay 0.9995 \
  --decode-every 1 \
  --quality-every 1 \
  --decoder-weight 0.1 \
  --fsq-weight 1 \
  --quality-weight 0.1 \
  --foot-slide-quality-weight 0.1 \
  --path-weight 0.1 \
  --seam-weight 0.01 \
  --physical-seam-weight 0.001 \
  --root-temporal-weight 0.01 \
  --log-every 100 \
  --save-every 2500 \
  --state-every 2500 \
  --num-workers 8 \
  --cache-shards 4 \
  --sample-order shard_shuffle \
  --flow-width 512 \
  --flow-heads 8 \
  --flow-trunk-blocks 8 \
  --flow-body-blocks 8 \
  --solver-steps 1 \
  --flow-root-smoothing-passes 0 \
  --flow-root-projection-kind binomial \
  --flow-root-control-points 10 \
  --encoder-width 512 \
  --encoder-blocks 4 \
  --decoder-width 512 \
  --decoder-blocks 8 \
  --decoder-token-hidden 32 \
  --codec-expansion 2 \
  --exact-t1-probability 0.7 \
  --high-noise-probability 0.2 \
  --seed 20260715 \
  --max-runtime-s 172800 \
  --resume "$STATE" \
  --override-learning-rates-on-resume
