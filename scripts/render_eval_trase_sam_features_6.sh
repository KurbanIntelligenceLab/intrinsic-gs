#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-/workspace/.envs/trase/bin/python}"
LOG_ROOT="${LOG_ROOT:-$ROOT/output/trase_sam_render_logs}"
K_VALUES="${K_VALUES:-30}"
mkdir -p "$LOG_ROOT" "$ROOT/output/sam_mask_eval"

SCENES=(
  americano
  chickchicken
  espresso
  keyboard
  split-cookie
  torchocolate
)

source_path() {
  local scene="$1"
  if [[ -d "$ROOT/data/HyperNeRF/misc/$scene" ]]; then
    printf "%s\n" "$ROOT/data/HyperNeRF/misc/$scene"
  else
    printf "%s\n" "$ROOT/data/HyperNeRF/interp/$scene"
  fi
}

render_scene() {
  local gpu="$1"
  local scene="$2"
  local k="$3"
  local model="$ROOT/output/${scene}_trase_sam"
  local ckpt="$model/point_cloud/iteration_40000/point_cloud.ply"
  local out_dir="$model/trase_features_k${k}"
  local ids_dir="$out_dir/cluster_ids_train"
  local log="$LOG_ROOT/${scene}_k${k}.log"

  if [[ ! -f "$ckpt" ]]; then
    echo "[missing] $scene checkpoint not found: $ckpt" >&2
    return 1
  fi

  if find "$ids_dir" -maxdepth 1 -name '*.png' -print -quit 2>/dev/null | grep -q .; then
    echo "[skip] $scene k=$k already has cluster-id renders"
    return 0
  fi

  echo "[render] gpu=$gpu scene=$scene k=$k log=$log"
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_BIN" self_supervised_scripts/render_clusters.py \
    -s "$(source_path "$scene")" \
    --model_path "$model" \
    --load_iteration 40000 \
    --n_clusters "$k" \
    --output_dir "$out_dir" \
    --save_cluster_ids \
    --load2gpu_on_the_fly \
    --load_image_on_the_fly \
    --load_mask_on_the_fly \
    >"$log" 2>&1
}

for k in $K_VALUES; do
  pids=()
  render_scene 0 americano "$k" &
  pids+=("$!")
  render_scene 1 chickchicken "$k" &
  pids+=("$!")

  for pid in "${pids[@]}"; do wait "$pid"; done

  pids=()
  render_scene 0 espresso "$k" &
  pids+=("$!")
  render_scene 1 keyboard "$k" &
  pids+=("$!")

  for pid in "${pids[@]}"; do wait "$pid"; done

  pids=()
  render_scene 0 split-cookie "$k" &
  pids+=("$!")
  render_scene 1 torchocolate "$k" &
  pids+=("$!")

  for pid in "${pids[@]}"; do wait "$pid"; done

  "$PYTHON_BIN" scripts/eval_trase_sam_zip.py \
    --pred_pattern "output/{scene}_trase_sam/trase_features_k${k}/cluster_ids_train" \
    --out_csv "$ROOT/output/sam_mask_eval/trase_sam_k${k}_vs_sam_object_masks.csv"
done

echo "ALL_DONE"
