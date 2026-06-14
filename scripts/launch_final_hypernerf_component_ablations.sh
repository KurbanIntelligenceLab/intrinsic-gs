#!/usr/bin/env bash
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_PREFIX="${ENV_PREFIX:-/workspace/.envs/trase}"
PYTHON_BIN="${PYTHON_BIN:-$ENV_PREFIX/bin/python}"
LOG_ROOT="${LOG_ROOT:-$ROOT/output/final_hypernerf_component_ablation_logs}"

mkdir -p "$LOG_ROOT"
cd "$ROOT" || exit 1
export PATH="$ENV_PREFIX/bin:$PATH"

run_variant() {
  local gpu="$1"
  local variant="$2"
  local scene="$3"
  shift 3
  local log="$LOG_ROOT/${variant}_${scene}.log"
  printf '[%(%F %T)T] START gpu=%s variant=%s scene=%s\n' -1 "$gpu" "$variant" "$scene" | tee -a "$LOG_ROOT/launcher.log"
  (
    set -o pipefail
    export CUDA_VISIBLE_DEVICES="$gpu"
    "$PYTHON_BIN" scripts/run_hypernerf_leiden_scene.py \
      --scene "$scene" \
      --iteration 30000 \
      --clusterer leiden \
      --leiden_resolution 0.03 \
      --skip_done \
      "$@"
  ) >"$log" 2>&1
  local status=$?
  if [[ "$status" -eq 0 ]]; then
    printf '[%(%F %T)T] DONE  gpu=%s variant=%s scene=%s\n' -1 "$gpu" "$variant" "$scene" | tee -a "$LOG_ROOT/launcher.log"
  else
    printf '[%(%F %T)T] FAIL  gpu=%s variant=%s scene=%s status=%s log=%s\n' -1 "$gpu" "$variant" "$scene" "$status" "$log" | tee -a "$LOG_ROOT/launcher.log"
  fi
  return "$status"
}

run_scene_set() {
  local gpu="$1"
  shift
  local scenes=("$@")
  local scene
  for scene in "${scenes[@]}"; do
    run_variant "$gpu" default "$scene" --use_motion --use_boundary || exit 1
    run_variant "$gpu" no_geo "$scene" --no_geo --use_motion --use_boundary || exit 1
    run_variant "$gpu" no_motion "$scene" --use_boundary || exit 1
    run_variant "$gpu" no_boundary "$scene" --use_motion || exit 1
    run_variant "$gpu" no_motion_no_boundary "$scene" || exit 1
  done
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

run_scene_set 0 "${gpu0_scenes[@]}" &
pid0="$!"

run_scene_set 1 "${gpu1_scenes[@]}" &
pid1="$!"

for pid in "$pid0" "$pid1"; do
  if ! wait "$pid"; then
    failures=$((failures + 1))
  fi
done

printf '[%(%F %T)T] ALL_DONE failures=%s\n' -1 "$failures" | tee -a "$LOG_ROOT/launcher.log"
exit "$failures"
