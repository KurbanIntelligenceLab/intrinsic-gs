#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-/workspace/.envs/trase/bin/python}"
LOG_ROOT="${LOG_ROOT:-$ROOT/output/trase_sam_feature_logs_new4}"
mkdir -p "$LOG_ROOT"

SCENES=(slice-banana cut-lemon1 hand1-dense-v2 oven-mitts)

source_path() {
  local scene="$1"
  if [[ -d "$ROOT/data/HyperNeRF/misc/$scene" ]]; then
    printf "%s\n" "$ROOT/data/HyperNeRF/misc/$scene"
  else
    printf "%s\n" "$ROOT/data/HyperNeRF/interp/$scene"
  fi
}

prepare_model_dir() {
  local scene="$1"
  local src="$ROOT/output/${scene}_30k_gs"
  local dst="$ROOT/output/${scene}_trase_sam"
  mkdir -p "$dst/point_cloud" "$dst/deform"
  cp -n "$src/cfg_args" "$dst/cfg_args" 2>/dev/null || true
  cp -n "$src/input.ply" "$dst/input.ply" 2>/dev/null || true
  cp -n "$src/cameras.json" "$dst/cameras.json" 2>/dev/null || true
  cp -rn "$src/point_cloud/iteration_30000" "$dst/point_cloud/" 2>/dev/null || true
  cp -rn "$src/deform/iteration_30000" "$dst/deform/" 2>/dev/null || true
  cp -n "$src/deform/deform_${scene}.pth" "$dst/deform/deform_${scene}.pth" 2>/dev/null || true
}

run_scene() {
  local gpu="$1"
  local scene="$2"
  local src
  src="$(source_path "$scene")"
  local model="$ROOT/output/${scene}_trase_sam"
  local log="$LOG_ROOT/${scene}.log"

  prepare_model_dir "$scene"
  if [[ -f "$model/point_cloud/iteration_40000/point_cloud.ply" ]]; then
    echo "[skip] $scene already has iteration_40000"
    return 0
  fi

  echo "[train] gpu=$gpu scene=$scene log=$log"
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_BIN" train.py \
    -s "$src" \
    -m "$model" \
    --load_iteration 30000 \
    --warm_up 1500 \
    --warm_up_3d_features 0 \
    --iterative_opt_interval 20000 \
    --iterations 40000 \
    --save_iterations 40000 \
    --monitor_mem \
    --densify_until_iter 0 \
    --lambda_reg_deform 0.0 \
    --eval \
    --load2gpu_on_the_fly \
    --load_image_on_the_fly \
    --load_mask_on_the_fly \
    --num_sampled_pixels 5000 \
    --num_sampled_masks 25 \
    --smooth_K 16 \
    --contrastive_mode soft \
    >"$log" 2>&1
}

"$PYTHON_BIN" scripts/convert_sam3_npz_to_trase_masks.py

(
  run_scene 0 cut-lemon1
  run_scene 0 hand1-dense-v2
) &
pid0=$!

(
  run_scene 1 slice-banana
  run_scene 1 oven-mitts
) &
pid1=$!

wait "$pid0"
wait "$pid1"
echo "ALL_DONE"
