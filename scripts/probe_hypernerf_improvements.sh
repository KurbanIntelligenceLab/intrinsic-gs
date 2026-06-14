#!/usr/bin/env bash
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_PREFIX="${ENV_PREFIX:-/workspace/.envs/trase}"
PYTHON_BIN="${PYTHON_BIN:-$ENV_PREFIX/bin/python}"
LOG_ROOT="${LOG_ROOT:-$ROOT/output/hypernerf_improve_probe_logs}"

mkdir -p "$LOG_ROOT"
cd "$ROOT" || exit 1
export PATH="$ENV_PREFIX/bin:$PATH"

run_probe() {
  local gpu="$1"
  local name="$2"
  shift 2
  local log="$LOG_ROOT/${name}.log"
  printf '[%(%F %T)T] START gpu=%s probe=%s\n' -1 "$gpu" "$name" | tee -a "$LOG_ROOT/launcher.log"
  (
    set -o pipefail
    export CUDA_VISIBLE_DEVICES="$gpu"
    "$PYTHON_BIN" scripts/run_hypernerf_leiden_scene.py --iteration 30000 "$@"
  ) >"$log" 2>&1
  local status=$?
  if [[ "$status" -eq 0 ]]; then
    printf '[%(%F %T)T] DONE  gpu=%s probe=%s\n' -1 "$gpu" "$name" | tee -a "$LOG_ROOT/launcher.log"
  else
    printf '[%(%F %T)T] FAIL  gpu=%s probe=%s status=%s log=%s\n' -1 "$gpu" "$name" "$status" "$log" | tee -a "$LOG_ROOT/launcher.log"
  fi
  return "$status"
}

failures=0

(
  run_probe 0 americano_k30 --scene americano --clusterer kmeans --n_clusters 30 --eigengap_k 31 || exit 1
  run_probe 0 americano_leiden003 --scene americano --clusterer leiden --leiden_resolution 0.03 || exit 1
  run_probe 0 torchocolate_k30 --scene torchocolate --clusterer kmeans --n_clusters 30 --eigengap_k 31 || exit 1
  run_probe 0 torchocolate_hdb005 --scene torchocolate --clusterer hdbscan --eigengap_k 20 --hdbscan_min_cluster_size_frac 0.005 --hdbscan_min_samples 5 || exit 1
) &
pid0="$!"

(
  run_probe 1 keyboard_k30 --scene keyboard --clusterer kmeans --n_clusters 30 --eigengap_k 31 || exit 1
  run_probe 1 keyboard_boundary --scene keyboard --clusterer kmeans --n_clusters 14 --eigengap_k 15 --use_boundary --boundary_views 8 --presmooth_sigma 3.0 || exit 1
  run_probe 1 slice_hdb005 --scene slice-banana --clusterer hdbscan --eigengap_k 20 --hdbscan_min_cluster_size_frac 0.005 --hdbscan_min_samples 5 || exit 1
  run_probe 1 hand1_motion --scene hand1-dense-v2 --clusterer kmeans --n_clusters 14 --eigengap_k 15 --use_motion || exit 1
) &
pid1="$!"

for pid in "$pid0" "$pid1"; do
  if ! wait "$pid"; then
    failures=$((failures + 1))
  fi
done

printf '[%(%F %T)T] ALL_DONE failures=%s\n' -1 "$failures" | tee -a "$LOG_ROOT/launcher.log"
exit "$failures"
