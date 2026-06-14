#!/usr/bin/env bash
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_PREFIX="${ENV_PREFIX:-/workspace/.envs/trase}"
PYTHON_BIN="${PYTHON_BIN:-$ENV_PREFIX/bin/python}"
LOG_ROOT="${LOG_ROOT:-$ROOT/output/neu3d_pidinet_boundary_operator_logs}"
OUTPUT_ROOT="${OUTPUT_ROOT:-output/neu3d_pidinet_boundary_operator}"
LEIDEN_RESOLUTION="${LEIDEN_RESOLUTION:-0.018}"
PIDINET_VARIANT="${PIDINET_VARIANT:-full}"

mkdir -p "$LOG_ROOT"
cd "$ROOT" || exit 1
export PATH="$ENV_PREFIX/bin:$PATH"

run_scene() {
  local gpu="$1"
  local scene="$2"
  local log="$LOG_ROOT/pidinet_${PIDINET_VARIANT}_${scene}.log"
  printf '[%(%F %T)T] START gpu=%s scene=%s pidinet=%s res=%s\n' -1 "$gpu" "$scene" "$PIDINET_VARIANT" "$LEIDEN_RESOLUTION" | tee -a "$LOG_ROOT/launcher.log"
  (
    set -o pipefail
    export CUDA_VISIBLE_DEVICES="$gpu"
    "$PYTHON_BIN" scripts/run_neu3d_leiden_scene.py \
      --scene "$scene" \
      --variant "pidinet_${PIDINET_VARIANT}" \
      --iteration 30000 \
      --leiden_resolution "$LEIDEN_RESOLUTION" \
      --output_root "$OUTPUT_ROOT" \
      --use_motion \
      --use_boundary \
      --rgb_edge_method pidinet \
      --pidinet_variant "$PIDINET_VARIANT" \
      --skip_done
  ) >"$log" 2>&1
  local status=$?
  if [[ "$status" -eq 0 ]]; then
    printf '[%(%F %T)T] DONE  gpu=%s scene=%s\n' -1 "$gpu" "$scene" | tee -a "$LOG_ROOT/launcher.log"
  else
    printf '[%(%F %T)T] FAIL  gpu=%s scene=%s status=%s log=%s\n' -1 "$gpu" "$scene" "$status" "$log" | tee -a "$LOG_ROOT/launcher.log"
  fi
  return "$status"
}

printf '[%(%F %T)T] PIDINET_NEU3D_START output_root=%s\n' -1 "$OUTPUT_ROOT" | tee -a "$LOG_ROOT/launcher.log"
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits | tee -a "$LOG_ROOT/launcher.log"

# Two concurrent workers per visible A100. This keeps the launch aggressive
# without assuming every scene has identical memory behavior.
run_scene 0 coffee_martini &
pid0="$!"
run_scene 0 cut_roasted_beef &
pid1="$!"
run_scene 1 cook_spinach &
pid2="$!"
run_scene 1 flame_steak &
pid3="$!"

failures=0
for pid in "$pid0" "$pid1" "$pid2" "$pid3"; do
  if ! wait "$pid"; then
    failures=$((failures + 1))
  fi
done

# Fill whichever GPU frees last with the fifth scene.
if ! run_scene 0 sear_steak; then
  failures=$((failures + 1))
fi

printf '[%(%F %T)T] PIDINET_NEU3D_DONE failures=%s\n' -1 "$failures" | tee -a "$LOG_ROOT/launcher.log"
exit "$failures"
