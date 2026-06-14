#!/usr/bin/env bash
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_PREFIX="${ENV_PREFIX:-/workspace/.envs/trase}"
PYTHON_BIN="${PYTHON_BIN:-$ENV_PREFIX/bin/python}"
LOG_ROOT="${LOG_ROOT:-$ROOT/output/final_neu3d_leiden018_component_ablation_logs_two_lane}"
LEIDEN_RESOLUTION="${LEIDEN_RESOLUTION:-0.018}"
OUTPUT_ROOT="${OUTPUT_ROOT:-output/final_neu3d_leiden018_component_ablations}"

mkdir -p "$LOG_ROOT"
cd "$ROOT" || exit 1
export PATH="$ENV_PREFIX/bin:$PATH"

run_variant() {
  local gpu="$1"
  local variant="$2"
  local scene="$3"
  shift 3
  local log="$LOG_ROOT/${variant}_${scene}.log"

  printf '[%(%F %T)T] START gpu=%s variant=%s scene=%s res=%s\n' -1 "$gpu" "$variant" "$scene" "$LEIDEN_RESOLUTION" | tee -a "$LOG_ROOT/launcher.log"
  (
    set -o pipefail
    export CUDA_VISIBLE_DEVICES="$gpu"
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
    printf '[%(%F %T)T] DONE  gpu=%s variant=%s scene=%s\n' -1 "$gpu" "$variant" "$scene" | tee -a "$LOG_ROOT/launcher.log"
  else
    printf '[%(%F %T)T] FAIL  gpu=%s variant=%s scene=%s status=%s log=%s\n' -1 "$gpu" "$variant" "$scene" "$status" "$log" | tee -a "$LOG_ROOT/launcher.log"
  fi
  return "$status"
}

lane0() {
  local failures=0
  run_variant 0 default sear_steak --use_motion --use_boundary || failures=$((failures + 1))
  run_variant 0 no_geo sear_steak --no_geo --use_motion --use_boundary || failures=$((failures + 1))
  run_variant 0 no_boundary sear_steak --use_motion || failures=$((failures + 1))
  return "$failures"
}

lane1() {
  local failures=0
  run_variant 1 no_boundary cut_roasted_beef --use_motion || failures=$((failures + 1))
  run_variant 1 no_motion_no_boundary cut_roasted_beef || failures=$((failures + 1))
  run_variant 1 no_motion sear_steak --use_boundary || failures=$((failures + 1))
  run_variant 1 no_motion_no_boundary sear_steak || failures=$((failures + 1))
  return "$failures"
}

printf '[%(%F %T)T] TWO_LANE_START res=%s\n' -1 "$LEIDEN_RESOLUTION" | tee -a "$LOG_ROOT/launcher.log"
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits | tee -a "$LOG_ROOT/launcher.log"

lane0 &
pid0="$!"
lane1 &
pid1="$!"
printf '%s\n%s\n' "$pid0" "$pid1" > "$LOG_ROOT/pids.txt"
printf '[%(%F %T)T] LANES pids=%s %s\n' -1 "$pid0" "$pid1" | tee -a "$LOG_ROOT/launcher.log"

failures=0
if ! wait "$pid0"; then
  failures=$((failures + 1))
fi
if ! wait "$pid1"; then
  failures=$((failures + 1))
fi

printf '[%(%F %T)T] LANES_DONE failures=%s\n' -1 "$failures" | tee -a "$LOG_ROOT/launcher.log"
"$PYTHON_BIN" scripts/summarize_neu3d_component_ablations.py --root "$OUTPUT_ROOT" >> "$LOG_ROOT/launcher.log" 2>&1
printf '[%(%F %T)T] SUMMARY_REFRESHED\n' -1 | tee -a "$LOG_ROOT/launcher.log"
exit "$failures"
