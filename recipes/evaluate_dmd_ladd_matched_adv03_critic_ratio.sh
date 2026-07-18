#!/usr/bin/env bash
set -euo pipefail

cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

ROOT=${DMD_MATCHED_RATIO_EVAL_ROOT:-distill_runs/dmd_ladd_matched_adv03_critic_ratio_from_g300_joint22500_20260716}
ARMS=(g1_s1_d1_control g1_s1_d2 g1_s1_d4)
STEPS=(350 400 450 500)
STATIC_GPUS=(1 2 3)

exec 9<"$ROOT"
if ! flock -n 9; then
  echo "matched-time critic-ratio training/evaluation is already running" >&2
  exit 75
fi

pids=()
for index in "${!ARMS[@]}"; do
  arm="${ARMS[$index]}"
  gpu="${STATIC_GPUS[$index]}"
  (
    for step in "${STEPS[@]}"; do
      bash scripts/evaluate_dmd_ladd_lr_checkpoint_static.sh \
        "$ROOT" "$arm" "$step" "$gpu"
    done
  ) >"$ROOT/$arm/static_eval.log" 2>&1 &
  pids+=("$!")
done

failed=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    failed=1
  fi
done
[[ "$failed" -eq 0 ]] || exit "$failed"

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
  --control-arm g1_s1_d1_control \
  --steps "${STEPS[@]}" \
  --output "$ROOT/critic_ratio_summary.json"
