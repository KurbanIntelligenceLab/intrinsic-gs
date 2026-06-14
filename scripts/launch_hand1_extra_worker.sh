#!/usr/bin/env bash
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_PREFIX="${ENV_PREFIX:-/workspace/.envs/trase}"
PYTHON_BIN="${PYTHON_BIN:-$ENV_PREFIX/bin/python}"
LOG_ROOT="${LOG_ROOT:-$ROOT/output/hypernerf_30k_recovery_logs}"
ITERATIONS="${ITERATIONS:-30000}"
GPU="${GPU:-0}"
SCENE="hand1-dense-v2"
MODEL="output/${SCENE}_30k_gs"
LOG="$LOG_ROOT/${SCENE}.log"
SENTINEL="$MODEL/point_cloud/iteration_${ITERATIONS}/point_cloud.ply"
INCOMPLETE="$MODEL/point_cloud/iteration_${ITERATIONS}/.extra_worker_incomplete"

mkdir -p "$LOG_ROOT" "$(dirname "$SENTINEL")"
cd "$ROOT" || exit 1

if pgrep -af "train.py .*data/HyperNeRF/interp/${SCENE}" >/dev/null; then
  printf '[%(%F %T)T] SKIP  scene=%s already running\n' -1 "$SCENE" | tee -a "$LOG_ROOT/launcher.log"
  exit 0
fi

if [[ -s "$SENTINEL" && ! -f "$INCOMPLETE" ]]; then
  printf '[%(%F %T)T] SKIP  scene=%s checkpoint exists\n' -1 "$SCENE" | tee -a "$LOG_ROOT/launcher.log"
  exit 0
fi

# The recovery launcher has hand1 queued after keyboard. This placeholder makes
# that queued call skip while this extra worker owns the real training run.
: > "$SENTINEL"
: > "$INCOMPLETE"

printf '[%(%F %T)T] START gpu=%s scene=interp/%s model=%s log=%s extra_worker=1\n' -1 "$GPU" "$SCENE" "$MODEL" "$LOG" | tee -a "$LOG_ROOT/launcher.log"

(
  set -o pipefail
  export CUDA_VISIBLE_DEVICES="$GPU"
  export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
  "$PYTHON_BIN" train.py \
    -s "data/HyperNeRF/interp/${SCENE}" \
    -m "$MODEL" \
    --warm_up 1500 \
    --warm_up_3d_features 30001 \
    --iterative_opt_interval 20000 \
    --iterations "$ITERATIONS" \
    --test_iterations 5000 10000 15000 20000 "$ITERATIONS" \
    --save_iterations 20000 "$ITERATIONS" \
    --monitor_mem \
    --densify_until_iter 9000 \
    --lambda_reg_deform 0.0 \
    --load2gpu_on_the_fly \
    --eval
) >"$LOG" 2>&1
status=$?

if [[ "$status" -eq 0 ]]; then
  rm -f "$INCOMPLETE"
  if [[ -f "$MODEL/deform/iteration_${ITERATIONS}/deform.pth" ]]; then
    ln -f "$MODEL/deform/iteration_${ITERATIONS}/deform.pth" "$MODEL/deform/deform_${SCENE}.pth"
  fi
  printf '[%(%F %T)T] DONE  gpu=%s scene=interp/%s extra_worker=1\n' -1 "$GPU" "$SCENE" | tee -a "$LOG_ROOT/launcher.log"
else
  if [[ ! -s "$SENTINEL" ]]; then
    rm -f "$SENTINEL"
  fi
  rm -f "$INCOMPLETE"
  printf '[%(%F %T)T] FAIL  gpu=%s scene=interp/%s status=%s log=%s extra_worker=1\n' -1 "$GPU" "$SCENE" "$status" "$LOG" | tee -a "$LOG_ROOT/launcher.log"
fi

exit "$status"
