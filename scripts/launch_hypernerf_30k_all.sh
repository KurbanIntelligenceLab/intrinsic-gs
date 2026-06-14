#!/usr/bin/env bash
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_PREFIX="${ENV_PREFIX:-/workspace/.envs/trase}"
PYTHON_BIN="${PYTHON_BIN:-$ENV_PREFIX/bin/python}"
LOG_ROOT="${LOG_ROOT:-$ROOT/output/hypernerf_30k_logs}"
ITERATIONS="${ITERATIONS:-30000}"

mkdir -p "$LOG_ROOT"
cd "$ROOT" || exit 1

run_scene() {
  local gpu="$1"
  local rel_scene="$2"
  local scene
  scene="$(basename "$rel_scene")"
  local model="output/${scene}_30k_gs"
  local log="$LOG_ROOT/${scene}.log"

  if [[ -f "$model/point_cloud/iteration_${ITERATIONS}/point_cloud.ply" ]]; then
    printf '[%(%F %T)T] SKIP %-20s checkpoint exists at %s\n' -1 "$scene" "$model" | tee -a "$LOG_ROOT/launcher.log"
    return 0
  fi

  printf '[%(%F %T)T] START gpu=%s scene=%s model=%s log=%s\n' -1 "$gpu" "$rel_scene" "$model" "$log" | tee -a "$LOG_ROOT/launcher.log"

  (
    set -o pipefail
    export CUDA_VISIBLE_DEVICES="$gpu"
    export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
    "$PYTHON_BIN" train.py \
      -s "data/HyperNeRF/${rel_scene}" \
      -m "$model" \
      --warm_up 1500 \
      --warm_up_3d_features 30001 \
      --iterative_opt_interval 20000 \
      --iterations "$ITERATIONS" \
      --test_iterations 5000 10000 15000 20000 "$ITERATIONS" \
      --save_iterations 20000 "$ITERATIONS" \
      --monitor_mem \
      --densify_until_iter 9000 \
      --lambda_reg_deform 0.0 \
      --eval
  ) >"$log" 2>&1
  local status=$?

  if [[ "$status" -eq 0 && -f "$model/deform/iteration_${ITERATIONS}/deform.pth" ]]; then
    ln -f "$model/deform/iteration_${ITERATIONS}/deform.pth" "$model/deform/deform_${scene}.pth"
  fi

  if [[ "$status" -eq 0 ]]; then
    printf '[%(%F %T)T] DONE  gpu=%s scene=%s\n' -1 "$gpu" "$rel_scene" | tee -a "$LOG_ROOT/launcher.log"
  else
    printf '[%(%F %T)T] FAIL  gpu=%s scene=%s status=%s log=%s\n' -1 "$gpu" "$rel_scene" "$status" "$log" | tee -a "$LOG_ROOT/launcher.log"
  fi
  return "$status"
}

# Balanced by approximate resident image tensor size plus per-process training state.
GPU0_SCENES=(
  "interp/hand1-dense-v2"
  "misc/espresso"
  "misc/keyboard"
  "misc/split-cookie"
  "interp/slice-banana"
)

GPU1_SCENES=(
  "interp/cut-lemon1"
  "interp/torchocolate"
  "misc/oven-mitts"
  "misc/americano"
  "interp/chickchicken"
)

pids=()
for rel_scene in "${GPU0_SCENES[@]}"; do
  run_scene 0 "$rel_scene" &
  pids+=("$!")
done

for rel_scene in "${GPU1_SCENES[@]}"; do
  run_scene 1 "$rel_scene" &
  pids+=("$!")
done

failures=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    failures=$((failures + 1))
  fi
done

printf '[%(%F %T)T] ALL_DONE failures=%s\n' -1 "$failures" | tee -a "$LOG_ROOT/launcher.log"
exit "$failures"
