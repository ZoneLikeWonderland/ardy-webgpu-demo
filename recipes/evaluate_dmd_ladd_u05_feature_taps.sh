#!/usr/bin/env bash
set -euo pipefail

cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

ROOT=distill_runs/dmd_ladd_u05_feature_tap_from_g300_joint22500_20260716
ARMS=(body_final body_pre body_mid trunk_final)
STEPS=(350 400 450 500)
STATIC_GPUS=(1 2 4 3)

# Lock the experiment directory itself. This prevents a second orchestrator
# from launching duplicate rollout jobs without creating a persistent lockfile.
exec 9<"$ROOT"
if ! flock -n 9; then
  echo "feature-tap evaluation is already running" >&2
  exit 75
fi

# Evaluate each arm serially on one dedicated GPU; the four arms run in
# parallel, so no physical GPU is oversubscribed.
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

# One checkpoint is scheduled at a time. Its four matched rollout seeds use
# four GPUs concurrently; this preserves unambiguous evidence directories.
export ARDY_ROLLOUT_GPUS=1,2,4,3
for step in "${STEPS[@]}"; do
  for arm in "${ARMS[@]}"; do
    bash scripts/evaluate_dmd_ladd_lr_checkpoint_rollout.sh \
      "$ROOT" "$arm" "$step"
  done
done
