#!/usr/bin/env bash
set -euo pipefail

cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

ROOT=distill_runs/dmd_ladd_u05_independent_ratio_from_g300_joint22500_20260716
ARMS=(g1_s1_d1 g1_s2_d1 g1_s1_d2 g1_s2_d2)
STEPS=(350 400 450 500)
STATIC_GPUS=(1 2 4 3)

# Each arm gets a dedicated GPU so all 16 fixed-set checkpoints are evaluated
# in four parallel, non-overlapping workers.
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

# Four rollout seeds run concurrently on the same four empty physical GPUs.
# Checkpoints remain serial to avoid GPU oversubscription and retain clear logs.
export ARDY_ROLLOUT_GPUS=1,2,4,3
for step in "${STEPS[@]}"; do
  for arm in "${ARMS[@]}"; do
    bash scripts/evaluate_dmd_ladd_lr_checkpoint_rollout.sh \
      "$ROOT" "$arm" "$step"
  done
done
