#!/usr/bin/env bash
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_PREFIX="${ENV_PREFIX:-/workspace/.envs/trase}"
PYTHON_BIN="${PYTHON_BIN:-$ENV_PREFIX/bin/python}"
LOG_ROOT="${LOG_ROOT:-$ROOT/output/final_neu3d_leiden018_component_ablation_logs_resume}"
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

printf '[%(%F %T)T] RESUME_START res=%s\n' -1 "$LEIDEN_RESOLUTION" | tee -a "$LOG_ROOT/launcher.log"
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits | tee -a "$LOG_ROOT/launcher.log"

failures=0
pids=()

run_variant 0 default sear_steak --use_motion --use_boundary &
pids+=("$!")
run_variant 0 no_geo sear_steak --no_geo --use_motion --use_boundary &
pids+=("$!")
run_variant 0 no_boundary sear_steak --use_motion &
pids+=("$!")

run_variant 1 no_motion sear_steak --use_boundary &
pids+=("$!")
run_variant 1 no_motion_no_boundary sear_steak &
pids+=("$!")
run_variant 1 no_boundary cut_roasted_beef --use_motion &
pids+=("$!")
run_variant 1 no_motion_no_boundary cut_roasted_beef &
pids+=("$!")

printf '%s\n' "${pids[@]}" > "$LOG_ROOT/pids.txt"
printf '[%(%F %T)T] LAUNCHED pids=%s\n' -1 "$(tr '\n' ' ' < "$LOG_ROOT/pids.txt")" | tee -a "$LOG_ROOT/launcher.log"

for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    failures=$((failures + 1))
  fi
done

printf '[%(%F %T)T] WORKERS_DONE failures=%s\n' -1 "$failures" | tee -a "$LOG_ROOT/launcher.log"
"$PYTHON_BIN" scripts/summarize_neu3d_component_ablations.py --root "$OUTPUT_ROOT" >> "$LOG_ROOT/launcher.log" 2>&1
printf '[%(%F %T)T] SUMMARY_REFRESHED\n' -1 | tee -a "$LOG_ROOT/launcher.log"
exit "$failures"
