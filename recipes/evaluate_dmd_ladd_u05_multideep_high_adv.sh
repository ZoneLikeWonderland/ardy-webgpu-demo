#!/usr/bin/env bash
set -euo pipefail

cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

ROOT=${DMD_HIGH_ADV_EVAL_ROOT:-distill_runs/dmd_ladd_u05_multideep_high_adv_from_g300_joint22500_20260716}
ARMS=(adv1e2 adv3e2 adv1e1 adv3e1)
STEPS=(350 400 450 500)
STATIC_GPUS=(1 2 4 3)

exec 9<"$ROOT"
if ! flock -n 9; then
  echo "multi-deep high-adversarial evaluation is already running" >&2
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

export ARDY_ROLLOUT_GPUS=1,2,4,3
for step in "${STEPS[@]}"; do
  for arm in "${ARMS[@]}"; do
    bash scripts/evaluate_dmd_ladd_lr_checkpoint_rollout.sh \
      "$ROOT" "$arm" "$step"
  done
done
