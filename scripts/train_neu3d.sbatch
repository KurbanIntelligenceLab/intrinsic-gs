#!/bin/bash
#SBATCH --partition=longq
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=20:00:00
#SBATCH --job-name=neu3d_train
#SBATCH --output=output/neu3d_train_%j.log

# Trains all 5 Neu3D scenes sequentially on a single GPU allocation.
# Skips any scene whose point_cloud/iteration_30000/point_cloud.ply already
# exists, so it's safe to re-submit if some scenes were trained elsewhere.

set -eo pipefail

cd /okyanus/users/mtuncel/TRASE
source activate trase_env
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

ITER=30000
SCENES=(coffee_martini cook_spinach cut_roasted_beef flame_steak sear_steak)

for SCENE in "${SCENES[@]}"; do
    DATA=/okyanus/users/mtuncel/datasets/neu3d/$SCENE
    MODEL=output/${SCENE}_run
    CKPT=$MODEL/point_cloud/iteration_${ITER}/point_cloud.ply

    if [[ -f "$CKPT" ]]; then
        echo "[skip] $SCENE already trained ($CKPT exists)"
    else
        echo "[train] $SCENE"
        python train.py \
            -s "$DATA" --model_path "$MODEL" \
            --iterations $ITER --warm_up_3d_features $((ITER + 1)) \
            --warm_up 3000 \
            --iterative_opt_interval 20000 \
            --densify_until_iter 8000 \
            --lambda_reg_deform 0 --eval \
            --load2gpu_on_the_fly \
            --load_image_on_the_fly \
            --load_mask_on_the_fly \
            --num_sampled_pixels 10000 --num_sampled_masks 50 --smooth_K 16 \
            --contrastive_mode soft \
            --test_iterations 10000 15000 20000 ${ITER} \
            --save_iterations 10000 15000 20000 ${ITER}
    fi

    DEFORM_SRC=$MODEL/deform/iteration_${ITER}/deform.pth
    DEFORM_DST=deform/deform_${SCENE}.pth
    if [[ -f "$DEFORM_SRC" && ! -f "$DEFORM_DST" ]]; then
        mkdir -p deform
        cp "$DEFORM_SRC" "$DEFORM_DST"
        echo "[publish] $DEFORM_DST"
    fi
done

echo "All 5 scenes processed."
