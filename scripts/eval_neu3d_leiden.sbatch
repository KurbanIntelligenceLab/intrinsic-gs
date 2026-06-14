#!/bin/bash
#SBATCH --partition=longq
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=16:00:00
#SBATCH --job-name=neu3d_eval_leiden
#SBATCH --output=output/neu3d_eval_leiden_%j.log

# Leiden variant of eval_neu3d_baseline.sbatch. Runs spectral (Leiden cluster)
# → render → mIoU → mAcc on all 5 Neu3D scenes sequentially.
# Outputs land under multiple_ablation/neu3d_leiden/<scene>/leiden_res0.018_*/.

set -uo pipefail

cd /okyanus/users/mtuncel/TRASE

set +u
source /okyanus/progs/anaconda3/etc/profile.d/conda.sh
conda activate trase_env
set -u

echo "python    = $(which python)"
echo "CONDA_PREFIX = ${CONDA_PREFIX:-<unset>}"
if ! python -c "import torch; print('torch', torch.__version__)" 2>&1; then
    echo "ERROR: trase_env not active (torch not importable). Aborting."
    exit 1
fi

export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

SCENES=(coffee_martini cook_spinach cut_roasted_beef flame_steak sear_steak)
FAILED=()

for SCENE in "${SCENES[@]}"; do
    echo
    echo "=== [eval-leiden] $SCENE ==="
    if python self_supervised_scripts/pipeline_multi_scene.py \
        --dataset neu3d \
        --scenes "$SCENE" \
        --ablation_name neu3d_leiden \
        --clusterer leiden \
        --leiden_resolution 0.018 \
        --sigma_color 0.8 --sigma_scale 1.0 --power 2.0 --knn_k 20 \
        --n_time_steps 20 --motion_floor 0.2 \
        --boundary_views 12 --alpha_depth 5.0 --beta_rgb 2.0 --gamma 2.0 \
        --opacity_thresh 0.05; then
        echo "[ok] $SCENE"
    else
        rc=$?
        echo "[FAIL] $SCENE (rc=$rc)"
        FAILED+=("$SCENE")
    fi
done

echo
if (( ${#FAILED[@]} > 0 )); then
    echo "Failed scenes: ${FAILED[*]}"
    exit 1
fi
echo "All 5 scenes evaluated (Leiden)."
