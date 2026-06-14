#!/usr/bin/env bash
# Re-eval all HyperNeRF-Mask runs after the compute_miou.py off-by-one fix
# (commit 75e5f26). Re-computes mIoU + mAcc in best_cluster and greedy_union
# modes, in-place — overwrites the existing JSONs in each run directory.
#
# Neu3D and other-dataset runs are skipped: they use frame_alignment.method =
# "direct" and never entered the broken indexed remap.
#
# Usage:
#   bash self_supervised_scripts/reeval_hypernerf_mask.sh
#
# Override defaults via env vars if needed:
#   REPO_ROOT, ABLATION_ROOT, HYPERNERF_DATA

set -u

REPO_ROOT="${REPO_ROOT:-/okyanus/users/mtuncel/TRASE}"
ABLATION_ROOT="${ABLATION_ROOT:-$REPO_ROOT/multiple_ablation}"
HYPERNERF_DATA="${HYPERNERF_DATA:-/okyanus/users/mtuncel/datasets/hypernerf}"

HYPERNERF_SCENES=(americano chickchicken cut-lemon1 espresso hand1-dense-v2 \
                  keyboard oven-mitts slice-banana split-cookie torchocolate)

MIOU=$REPO_ROOT/self_supervised_scripts/compute_miou.py
MACC=$REPO_ROOT/self_supervised_scripts/compute_macc.py

is_hypernerf() {
    local scene="$1"
    for s in "${HYPERNERF_SCENES[@]}"; do
        [ "$s" = "$scene" ] && return 0
    done
    return 1
}

total=0; done_count=0; skipped=0

for ABL_DIR in "$ABLATION_ROOT"/*/; do
    [ -d "$ABL_DIR" ] || continue
    ABL=$(basename "$ABL_DIR")
    echo
    echo "============================================================"
    echo " Ablation: $ABL"
    echo "============================================================"

    for SCENE_DIR in "$ABL_DIR"*/; do
        [ -d "$SCENE_DIR" ] || continue
        SCENE=$(basename "$SCENE_DIR")
        is_hypernerf "$SCENE" || continue

        GT_DIR="$HYPERNERF_DATA/$SCENE/gt_masks"
        if [ ! -d "$GT_DIR" ]; then
            echo "  [no GT for $SCENE at $GT_DIR]"
            continue
        fi

        for RUN_DIR in "$SCENE_DIR"*/; do
            [ -d "$RUN_DIR" ] || continue
            RUN_NAME=$(basename "$RUN_DIR")
            PRED_DIR="${RUN_DIR%/}/cluster_ids_train"
            REPORT="${RUN_DIR%/}/report.md"
            total=$((total + 1))

            if [ ! -d "$PRED_DIR" ]; then
                echo "  [skip $SCENE/$RUN_NAME — no cluster_ids_train]"
                skipped=$((skipped + 1))
                continue
            fi

            echo "  >> $SCENE / $RUN_NAME"

            python "$MIOU" \
                --pred_dir "$PRED_DIR" \
                --gt_dir "$GT_DIR" \
                --report_md "$REPORT" \
                --output_json "${RUN_DIR%/}/miou_results.json" \
                2>&1 | grep -E "mIoU \(scene-wide" | sed 's/^/     /'

            python "$MIOU" \
                --pred_dir "$PRED_DIR" \
                --gt_dir "$GT_DIR" \
                --report_md "$REPORT" \
                --selection_mode greedy_union \
                --output_json "${RUN_DIR%/}/miou_results_greedy.json" \
                2>&1 | grep -E "mIoU \(scene-wide" | sed 's/^/     /'

            # mAcc reads selected clusters from the corresponding miou JSON,
            # so it must run *after* the mIoU re-eval above.
            python "$MACC" \
                --pred_dir "$PRED_DIR" \
                --gt_dir "$GT_DIR" \
                --miou_json "${RUN_DIR%/}/miou_results.json" \
                --output_json "${RUN_DIR%/}/macc_results.json" \
                2>&1 | grep -E "^mAcc" | sed 's/^/     /'

            python "$MACC" \
                --pred_dir "$PRED_DIR" \
                --gt_dir "$GT_DIR" \
                --miou_json "${RUN_DIR%/}/miou_results_greedy.json" \
                --output_json "${RUN_DIR%/}/macc_results_greedy.json" \
                2>&1 | grep -E "^mAcc" | sed 's/^/     /'

            done_count=$((done_count + 1))
        done
    done
done

echo
echo "============================================================"
echo " Done. $done_count runs re-evaluated, $skipped skipped, $total total."
echo "============================================================"
