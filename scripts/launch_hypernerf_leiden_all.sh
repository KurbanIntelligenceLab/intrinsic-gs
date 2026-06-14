#!/usr/bin/env bash
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_PREFIX="${ENV_PREFIX:-/workspace/.envs/trase}"
PYTHON_BIN="${PYTHON_BIN:-$ENV_PREFIX/bin/python}"
LOG_ROOT="${LOG_ROOT:-$ROOT/output/hypernerf_leiden_all_logs}"

mkdir -p "$LOG_ROOT"
cd "$ROOT" || exit 1

run_scene() {
  local gpu="$1"
  local scene="$2"
  local log="$LOG_ROOT/${scene}.log"
  printf '[%(%F %T)T] START gpu=%s scene=%s log=%s\n' -1 "$gpu" "$scene" "$log" | tee -a "$LOG_ROOT/launcher.log"
  (
    set -o pipefail
    export CUDA_VISIBLE_DEVICES="$gpu"
    export PATH="$ENV_PREFIX/bin:$PATH"
    "$PYTHON_BIN" scripts/run_hypernerf_leiden_scene.py --scene "$scene" --iteration 30000
  ) >"$log" 2>&1
  local status=$?
  if [[ "$status" -eq 0 ]]; then
    printf '[%(%F %T)T] DONE  gpu=%s scene=%s\n' -1 "$gpu" "$scene" | tee -a "$LOG_ROOT/launcher.log"
  else
    printf '[%(%F %T)T] FAIL  gpu=%s scene=%s status=%s log=%s\n' -1 "$gpu" "$scene" "$status" "$log" | tee -a "$LOG_ROOT/launcher.log"
  fi
  return "$status"
}

gpu0_scenes=(
  chickchicken
  cut-lemon1
  hand1-dense-v2
  slice-banana
  torchocolate
)

gpu1_scenes=(
  americano
  espresso
  keyboard
  oven-mitts
  split-cookie
)

failures=0

(
  for scene in "${gpu0_scenes[@]}"; do
    run_scene 0 "$scene" || exit 1
  done
) &
pid0="$!"

(
  for scene in "${gpu1_scenes[@]}"; do
    run_scene 1 "$scene" || exit 1
  done
) &
pid1="$!"

for pid in "$pid0" "$pid1"; do
  if ! wait "$pid"; then
    failures=$((failures + 1))
  fi
done

printf '[%(%F %T)T] ALL_DONE failures=%s\n' -1 "$failures" | tee -a "$LOG_ROOT/launcher.log"
exit "$failures"
