#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON_BIN="${PYTHON_BIN:-/workspace/.envs/trase/bin/python}"
INPUT_CSV="${INPUT_CSV:-output/final_hypernerf_1/maskbenchmark_full10_res003_global_longrange_input.csv}"
LOG_ROOT="${LOG_ROOT:-output/hypernerf_tau_lr_sweep_logs}"
PER_GPU="${PER_GPU:-2}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"

SCENES=(
  chickchicken
  cut-lemon1
  hand1-dense-v2
  slice-banana
  torchocolate
  americano
  espresso
  keyboard
  oven-mitts
  split-cookie
)

if [[ "$#" -gt 0 ]]; then
  TAUS=("$@")
else
  TAUS=(0.60 0.75 0.85)
fi
mkdir -p "$LOG_ROOT"

MAX_JOBS=$((PER_GPU * 2))
NEXT_GPU=0
PIDS=()
GPU_SLOT=0

refresh_pids() {
  local alive=()
  local pid
  for pid in "${PIDS[@]}"; do
    if kill -0 "$pid" 2>/dev/null; then
      alive+=("$pid")
    fi
  done
  PIDS=("${alive[@]}")
}

wait_for_slot() {
  refresh_pids
  while [[ "${#PIDS[@]}" -ge "$MAX_JOBS" ]]; do
    wait -n || true
    refresh_pids
  done
  GPU_SLOT="$NEXT_GPU"
  NEXT_GPU=$((1 - NEXT_GPU))
}

run_one() {
  local gpu="$1"
  local tau="$2"
  local scene="$3"
  local tau_tag="${tau/./}"
  local tag="objectness_lr_tau${tau_tag}"
  local log="${LOG_ROOT}/${scene}_${tag}.log"
  local skip_args=()
  if [[ "$SKIP_EXISTING" == "1" ]]; then
    skip_args+=(--skip-existing)
  fi
  (
    export PATH="/workspace/.envs/trase/bin:$PATH"
    export CUDA_VISIBLE_DEVICES="$gpu"
    "$PYTHON_BIN" scripts/run_objectness_longrange_res003.py \
      --input-csv "$INPUT_CSV" \
      --tag "$tag" \
      --pair-tau "$tau" \
      --scenes "$scene" \
      "${skip_args[@]}"
  ) >"$log" 2>&1 &
  local pid=$!
  PIDS+=("$pid")
  echo "[launch] gpu=${gpu} tau=${tau} scene=${scene} pid=${pid} active=${#PIDS[@]}/${MAX_JOBS} log=${log}"
}

for tau in "${TAUS[@]}"; do
  for scene in "${SCENES[@]}"; do
    wait_for_slot
    run_one "$GPU_SLOT" "$tau" "$scene"
  done
done

wait
echo "[done] tau sweep finished"
