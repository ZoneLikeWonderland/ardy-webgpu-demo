#!/usr/bin/env bash
set -euo pipefail

cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

ROOT=${DMD_MATCHED_D4_LR_EVAL_ROOT:-distill_runs/dmd_ladd_matched_adv03_d4_critic_lr_from_g300_joint22500_20260716}
ARMS=(clr1e8 clr3e8 clr1e7 clr3e7 clr1e6_control)
STEPS=(350 400 450 500)
STATIC_GPUS=(1 2 3 4)

exec 9<"$ROOT"
if ! flock -n 9; then
  echo "matched-time D4 critic-LR training/evaluation is already running" >&2
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
  -m ardy_distill.tools.summarize_dmd_ladd_independent_ratio \
  --root "$ROOT" \
  --baseline-rollout-root \
    distill_runs/dmd_ladd_lr_round1_joint22500_warm200_g200_20260715 \
  --baseline-full \
    distill_runs/text_control_joint_fullproj_resume20000_4gpu_b256_const5e6_until_plateau_step40000_ema995_20260715/eval/step022500_ema_full_fp16.json \
  --baseline-text \
    distill_runs/text_control_joint_fullproj_resume20000_4gpu_b256_const5e6_until_plateau_step40000_ema995_20260715/eval/step022500_ema_text_control_fp16.json \
  --arms "${ARMS[@]}" \
  --control-arm clr1e6_control \
  --steps "${STEPS[@]}" \
  --output "$ROOT/critic_lr_summary.json"
