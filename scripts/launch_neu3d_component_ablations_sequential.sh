#!/usr/bin/env bash
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_PREFIX="${ENV_PREFIX:-/workspace/.envs/trase}"
PYTHON_BIN="${PYTHON_BIN:-$ENV_PREFIX/bin/python}"
LOG_ROOT="${LOG_ROOT:-$ROOT/output/final_neu3d_component_ablation_logs_seq}"
LEIDEN_RESOLUTION="${LEIDEN_RESOLUTION:-0.03}"
OUTPUT_ROOT="${OUTPUT_ROOT:-output/final_neu3d_component_ablations}"

mkdir -p "$LOG_ROOT"
cd "$ROOT" || exit 1
export PATH="$ENV_PREFIX/bin:$PATH"

run_variant() {
  local variant="$1"
  local scene="$2"
  shift 2
  local log="$LOG_ROOT/${variant}_${scene}.log"
  printf '[%(%F %T)T] START variant=%s scene=%s res=%s\n' -1 "$variant" "$scene" "$LEIDEN_RESOLUTION" | tee -a "$LOG_ROOT/launcher.log"
  (
    set -o pipefail
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
    printf '[%(%F %T)T] DONE  variant=%s scene=%s\n' -1 "$variant" "$scene" | tee -a "$LOG_ROOT/launcher.log"
  else
    printf '[%(%F %T)T] FAIL  variant=%s scene=%s status=%s log=%s\n' -1 "$variant" "$scene" "$status" "$log" | tee -a "$LOG_ROOT/launcher.log"
  fi
  return "$status"
}

run_scene() {
  local scene="$1"
  run_variant default "$scene" --use_motion --use_boundary || return 1
  run_variant no_geo "$scene" --no_geo --use_motion --use_boundary || return 1
  run_variant no_motion "$scene" --use_boundary || return 1
  run_variant no_boundary "$scene" --use_motion || return 1
  run_variant no_motion_no_boundary "$scene" || return 1
}

scenes=(
  coffee_martini
  cook_spinach
  cut_roasted_beef
  flame_steak
  sear_steak
)

failures=0
for scene in "${scenes[@]}"; do
  if ! run_scene "$scene"; then
    failures=$((failures + 1))
  fi
done

printf '[%(%F %T)T] ALL_DONE failures=%s\n' -1 "$failures" | tee -a "$LOG_ROOT/launcher.log"
exit "$failures"
