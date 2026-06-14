#!/usr/bin/env bash
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_PREFIX="${ENV_PREFIX:-/workspace/.envs/trase}"
PYTHON_BIN="${PYTHON_BIN:-$ENV_PREFIX/bin/python}"
LOG_ROOT="${LOG_ROOT:-$ROOT/output/final_neu3d_leiden018_component_ablation_logs_single_lane}"
LEIDEN_RESOLUTION="${LEIDEN_RESOLUTION:-0.018}"
OUTPUT_ROOT="${OUTPUT_ROOT:-output/final_neu3d_leiden018_component_ablations}"
GPU="${GPU:-0}"

mkdir -p "$LOG_ROOT"
cd "$ROOT" || exit 1
export PATH="$ENV_PREFIX/bin:$PATH"

run_variant() {
  local variant="$1"
  local scene="$2"
  shift 2
  local log="$LOG_ROOT/${variant}_${scene}.log"

  printf '[%(%F %T)T] START gpu=%s variant=%s scene=%s res=%s\n' -1 "$GPU" "$variant" "$scene" "$LEIDEN_RESOLUTION" | tee -a "$LOG_ROOT/launcher.log"
  (
    set -o pipefail
    export CUDA_VISIBLE_DEVICES="$GPU"
    "$PYTHON_BIN" scripts/run_neu3d_leiden_scene.py \
      --scene "$scene" \
      --variant "$variant" \
      --iteration 30000 \
      --leiden_resolution "$LEIDEN_RESOLUTION" \
      --output_root "$OUTPUT_ROOT" \
      --skip_done \
      "$@"
  ) >"$log" 2>&1
  local status=$?
  if [[ "$status" -eq 0 ]]; then
    printf '[%(%F %T)T] DONE  gpu=%s variant=%s scene=%s\n' -1 "$GPU" "$variant" "$scene" | tee -a "$LOG_ROOT/launcher.log"
  else
    printf '[%(%F %T)T] FAIL  gpu=%s variant=%s scene=%s status=%s log=%s\n' -1 "$GPU" "$variant" "$scene" "$status" "$log" | tee -a "$LOG_ROOT/launcher.log"
  fi
  return "$status"
}

printf '[%(%F %T)T] SINGLE_LANE_START res=%s gpu=%s\n' -1 "$LEIDEN_RESOLUTION" "$GPU" | tee -a "$LOG_ROOT/launcher.log"
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits | tee -a "$LOG_ROOT/launcher.log"

failures=0
run_variant no_geo sear_steak --no_geo --use_motion --use_boundary || failures=$((failures + 1))
run_variant no_motion sear_steak --use_boundary || failures=$((failures + 1))
run_variant no_boundary sear_steak --use_motion || failures=$((failures + 1))
run_variant no_motion_no_boundary cut_roasted_beef || failures=$((failures + 1))
run_variant no_motion_no_boundary sear_steak || failures=$((failures + 1))

printf '[%(%F %T)T] SINGLE_LANE_DONE failures=%s\n' -1 "$failures" | tee -a "$LOG_ROOT/launcher.log"
"$PYTHON_BIN" scripts/summarize_neu3d_component_ablations.py --root "$OUTPUT_ROOT" >> "$LOG_ROOT/launcher.log" 2>&1
printf '[%(%F %T)T] SUMMARY_REFRESHED\n' -1 | tee -a "$LOG_ROOT/launcher.log"
exit "$failures"
