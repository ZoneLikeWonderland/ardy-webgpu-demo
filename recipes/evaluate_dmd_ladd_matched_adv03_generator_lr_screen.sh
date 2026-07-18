#!/usr/bin/env bash
set -euo pipefail

cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

ROOT=${DMD_MATCHED_LR_EVAL_ROOT:-distill_runs/dmd_ladd_matched_adv03_generator_lr_to700_joint22500_20260716}
ARMS=(lr1e8 lr3e8 lr1e7_control lr3e7 lr1e6)
STEPS=(400 500 600 700)
STATIC_GPUS=(1 2 3 4)

exec 9<"$ROOT"
if ! flock -n 9; then
  echo "matched-time generator-LR training/evaluation is already running" >&2
  exit 75
fi

run_static_arm() {
  local arm="$1"
  local gpu="$2"
  local step
  for step in "${STEPS[@]}"; do
    bash scripts/evaluate_dmd_ladd_lr_checkpoint_static.sh \
      "$ROOT" "$arm" "$step" "$gpu"
  done
}

# Four independent arms occupy one GPU each; the fifth runs as a second batch
# so no physical GPU is oversubscribed.
pids=()
for index in 0 1 2 3; do
  arm="${ARMS[$index]}"
  gpu="${STATIC_GPUS[$index]}"
  run_static_arm "$arm" "$gpu" >"$ROOT/$arm/static_eval.log" 2>&1 &
  pids+=("$!")
done

failed=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    failed=1
  fi
done
[[ "$failed" -eq 0 ]] || exit "$failed"

run_static_arm "${ARMS[4]}" "${STATIC_GPUS[0]}" \
  >"$ROOT/${ARMS[4]}/static_eval.log" 2>&1

export ARDY_ROLLOUT_GPUS=1,2,3,4
for step in "${STEPS[@]}"; do
  for arm in "${ARMS[@]}"; do
    bash scripts/evaluate_dmd_ladd_lr_checkpoint_rollout.sh \
      "$ROOT" "$arm" "$step"
  done
done

python \
  -m ardy_distill.tools.summarize_dmd_ladd_lr_iters \
  --root "$ROOT" \
  --baseline-rollout-root \
    distill_runs/dmd_ladd_lr_round1_joint22500_warm200_g200_20260715 \
  --baseline-full \
    distill_runs/text_control_joint_fullproj_resume20000_4gpu_b256_const5e6_until_plateau_step40000_ema995_20260715/eval/step022500_ema_full_fp16.json \
  --baseline-text \
    distill_runs/text_control_joint_fullproj_resume20000_4gpu_b256_const5e6_until_plateau_step40000_ema995_20260715/eval/step022500_ema_text_control_fp16.json \
  --arms "${ARMS[@]}" \
  --steps "${STEPS[@]}" \
  --output "$ROOT/generator_lr_iters_summary.json"
