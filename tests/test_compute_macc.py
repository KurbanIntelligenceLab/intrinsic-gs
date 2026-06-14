import json

import numpy as np
from PIL import Image

from self_supervised_scripts.compute_macc import (
    compute_macc,
    selected_clusters_from_miou_json,
)


def _write_cluster_id_map(path, arr):
    Image.fromarray(arr.astype(np.uint8), mode="L").save(path)


def _write_binary_mask(path, arr):
    Image.fromarray((arr.astype(np.uint8) * 255), mode="L").save(path)


def test_per_frame_pixel_accuracy_matches_hand_count(tmp_path):
    # Frame 0: 4-pixel image. GT FG = top-left only.
    # pred selects cluster 1 → top-left & top-right → 1 TP, 1 FP, 0 FN, 2 TN → 3/4 correct.
    pred0 = np.array([[1, 1], [2, 0]], dtype=np.uint8)
    gt0   = np.array([[1, 0], [0, 0]], dtype=bool)
    # Frame 1: pred=cluster1 covers exactly GT → 4/4 correct.
    pred1 = np.array([[1, 1], [0, 0]], dtype=np.uint8)
    gt1   = np.array([[1, 1], [0, 0]], dtype=bool)

    pred_dir = tmp_path / "pred"
    gt_dir = tmp_path / "gt"
    pred_dir.mkdir()
    gt_dir.mkdir()
    _write_cluster_id_map(pred_dir / "00000.png", pred0)
    _write_cluster_id_map(pred_dir / "00001.png", pred1)
    _write_binary_mask(gt_dir / "00000.png", gt0)
    _write_binary_mask(gt_dir / "00001.png", gt1)

    pred_paths = {0: str(pred_dir / "00000.png"), 1: str(pred_dir / "00001.png")}
    gt_paths   = {0: str(gt_dir / "00000.png"),  1: str(gt_dir / "00001.png")}

    result = compute_macc(pred_paths, gt_paths, selected_clusters=[1])

    assert result["n_frames"] == 2
    assert result["per_frame_acc"]["0"] == 0.75
    assert result["per_frame_acc"]["1"] == 1.0
    # mAcc averages per-frame accuracies, not pixel totals.
    assert result["mAcc"] == 0.875
    assert result["selected_clusters"] == [1]


def test_selected_clusters_union_projects_pred_correctly(tmp_path):
    # One frame, 4 pixels, GT marks 3 of them. Pred uses clusters {2,3,4,1}.
    pred = np.array([[2, 3], [4, 1]], dtype=np.uint8)
    gt   = np.array([[1, 1], [1, 0]], dtype=bool)
    pred_dir = tmp_path / "pred"
    gt_dir = tmp_path / "gt"
    pred_dir.mkdir()
    gt_dir.mkdir()
    _write_cluster_id_map(pred_dir / "00000.png", pred)
    _write_binary_mask(gt_dir / "00000.png", gt)

    paths = {0: str(pred_dir / "00000.png")}
    gts   = {0: str(gt_dir / "00000.png")}

    # Union {2,3,4} = first three pixels FG, 4th BG.
    # GT       = first three pixels FG, 4th BG.
    # → 4/4 correct, acc=1.0
    perfect = compute_macc(paths, gts, selected_clusters=[2, 3, 4])
    assert perfect["mAcc"] == 1.0

    # Pick only cluster 1 → covers pixel (1,1) which is GT-BG.
    # → 0 TP, 3 FN, 1 FP, 0 TN → 0/4 correct
    worst = compute_macc(paths, gts, selected_clusters=[1])
    assert worst["mAcc"] == 0.0


def test_skips_shape_mismatched_frames(tmp_path):
    pred = np.array([[1, 1], [0, 0]], dtype=np.uint8)
    gt_small = np.array([[1, 1]], dtype=bool)  # 1x2 vs pred 2x2

    pred_dir = tmp_path / "pred"
    gt_dir = tmp_path / "gt"
    pred_dir.mkdir()
    gt_dir.mkdir()
    _write_cluster_id_map(pred_dir / "00000.png", pred)
    _write_binary_mask(gt_dir / "00000.png", gt_small)

    paths = {0: str(pred_dir / "00000.png")}
    gts   = {0: str(gt_dir / "00000.png")}

    result = compute_macc(paths, gts, selected_clusters=[1])
    assert result["n_frames"] == 0
    assert result["skipped_shape_mismatch"] == 1
    assert result["mAcc"] == 0.0


def test_selected_clusters_from_miou_json_handles_legacy_best_cluster_only(tmp_path):
    # Pre-selection_mode JSON has only best_cluster (no selected_clusters list).
    miou_json = tmp_path / "miou_results.json"
    miou_json.write_text(json.dumps({
        "results": {"mIoU": 0.92, "best_cluster": 12, "selected_clusters": None}
    }))
    assert selected_clusters_from_miou_json(miou_json) == [12]


def test_selected_clusters_from_miou_json_prefers_explicit_list(tmp_path):
    miou_json = tmp_path / "miou_results_greedy.json"
    miou_json.write_text(json.dumps({
        "results": {
            "mIoU": 0.95,
            "best_cluster": 12,
            "selected_clusters": [12, 3, 7],
            "selection_mode": "greedy_union",
        }
    }))
    assert selected_clusters_from_miou_json(miou_json) == [12, 3, 7]


def test_selected_clusters_from_miou_json_returns_none_when_unevaluable(tmp_path):
    miou_json = tmp_path / "miou_results.json"
    miou_json.write_text(json.dumps({"results": {"mIoU": 0.0}}))  # no clusters at all
    assert selected_clusters_from_miou_json(miou_json) is None
