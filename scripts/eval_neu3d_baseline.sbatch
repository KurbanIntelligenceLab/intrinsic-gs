#!/bin/bash
#SBATCH --partition=longq
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --job-name=neu3d_eval_k14
#SBATCH --output=output/neu3d_eval_baseline_k14_%j.log

# Runs spectral → render → mIoU → mAcc on all 5 Neu3D scenes sequentially.
# Each scene is invoked in its own pipeline_multi_scene.py call so a single
# scene's failure (OOM, transient cluster issue) does not kill the rest.
# Trained checkpoints + published deform weights must already exist —
# see train_neu3d.sbatch.
#
# Outputs land under multiple_ablation/neu3d_baseline_k14/<scene>/spectral_k14_*/
# with miou_results{,_greedy}.json and macc_results{,_greedy}.json per run.

set -uo pipefail   # NOT -e: continue on a single-scene failure

cd /okyanus/users/mtuncel/TRASE

# `source activate <env>` (legacy syntax) silently no-ops under non-interactive
# sbatch shells because the conda init from ~/.bashrc isn't sourced — `python`
# then resolves to the system anaconda3 (Python 3.7, no torch). The job runs
# but every subprocess call from pipeline_multi_scene.py crashes with
# `ModuleNotFoundError: No module named 'torch'` (see job 3399 / 3401 logs).
# Fix: explicitly source conda.sh and use `conda activate`. Disable `nounset`
# around the init because conda.sh references $PS1, which is unset in
# non-interactive shells (this aborted activation mid-way in job 3402).
set +u
source /okyanus/progs/anaconda3/etc/profile.d/conda.sh
conda activate trase_env
set -u

# Sanity-check the env. Fail loud if python isn't trase_env's python.
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
    echo "=== [eval] $SCENE ==="
    if python self_supervised_scripts/pipeline_multi_scene.py \
        --dataset neu3d \
        --scenes "$SCENE" \
        --ablation_name neu3d_baseline_k14 \
        --max_palette_views 100 \
        --n_clusters 14 --eigengap_k 15 \
        --sigma_color 0.8 --sigma_scale 1.0 --power 2.0 --knn_k 20 \
        --n_time_steps 20 --motion_floor 0.2 \
        --boundary_views 12 --alpha_depth 5.0 --beta_rgb 2.0 --gamma 2.0; then
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
echo "All 5 scenes evaluated."
