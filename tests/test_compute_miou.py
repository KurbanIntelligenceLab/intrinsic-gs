import json

import numpy as np
from PIL import Image

from self_supervised_scripts.compute_miou import align_single_object_gt_paths
from self_supervised_scripts.compute_miou import evaluate_single_object
from self_supervised_scripts.compute_miou import save_visualizations
from self_supervised_scripts.compute_miou import select_original_dir


def test_aligns_sequential_gt_masks_to_hypernerf_validation_ids(tmp_path):
    scene_dir = tmp_path / "americano"
    gt_dir = tmp_path / "gt_masks"
    scene_dir.mkdir()
    gt_dir.mkdir()

    ids = [f"{idx:06d}" for idx in range(1, 13)]
    (scene_dir / "dataset.json").write_text(json.dumps({"ids": ids}))

    gt_paths = {}
    for idx in range(3):
        path = gt_dir / f"{idx:05d}.png"
        path.write_text("")
        gt_paths[idx] = str(path)

    pred_paths = {idx: f"pred/{idx:06d}.png" for idx in [1, 3, 5, 7, 9, 11]}

    aligned, info = align_single_object_gt_paths(pred_paths, gt_paths, str(gt_dir))

    assert sorted(aligned) == [3, 7, 11]
    assert aligned[3] == gt_paths[0]
    assert aligned[7] == gt_paths[1]
    assert aligned[11] == gt_paths[2]
    assert info["method"] == "hypernerf_val"


def test_aligns_one_based_hypernerf_masks_to_validation_render_ids(tmp_path):
    # GT filename K identifies val_ids[K] directly — the numbering base (0 or
    # 1) is incidental, not a 1-based-counting convention to subtract from.
    # GT 5 must map to val_ids[5], not val_ids[4].
    scene_dir = tmp_path / "chickchicken"
    benchmark_dir = tmp_path / "Mask-Benchmark" / "HyperNeRF-Mask" / "chickchicken"
    gt_dir = benchmark_dir / "gt_masks"
    scene_dir.mkdir()
    gt_dir.mkdir(parents=True)

    ids = [str(idx) for idx in range(1, 25)]
    (scene_dir / "dataset.json").write_text(json.dumps({"ids": ids}))
    # val_ids = ids[2::4] = ['3','7','11','15','19','23']

    gt_paths = {}
    for idx in [1, 2, 5]:
        path = gt_dir / f"{idx:05d}.png"
        path.write_text("")
        gt_paths[idx] = str(path)

    pred_paths = {idx: f"pred/{idx:06d}.png" for idx in range(1, 24, 2)}

    aligned, info = align_single_object_gt_paths(pred_paths, gt_paths, str(gt_dir))

    assert sorted(aligned) == [7, 11, 23]
    assert aligned[7] == gt_paths[1]
    assert aligned[11] == gt_paths[2]
    assert aligned[23] == gt_paths[5]
    assert info["method"] == "hypernerf_val_indexed"


def test_save_visualizations_removes_stale_panels(tmp_path):
    pred_path = tmp_path / "000003.png"
    gt_path = tmp_path / "gt_00000.png"
    original_dir = tmp_path / "rgb" / "1x"
    vis_dir = tmp_path / "visualizations"
    original_dir.mkdir(parents=True)
    vis_dir.mkdir()
    stale_path = vis_dir / "f00099_iou0.000.png"
    stale_path.write_text("stale")

    original = np.zeros((2, 2, 3), dtype=np.uint8)
    original[..., 0] = 255
    Image.fromarray(original).save(original_dir / "000003.png")
    Image.fromarray(np.ones((2, 2), dtype=np.uint8)).save(pred_path)
    Image.fromarray(np.full((2, 2), 255, dtype=np.uint8)).save(gt_path)

    save_visualizations(
        str(vis_dir),
        {3: str(pred_path)},
        {3: str(gt_path)},
        best_cluster=1,
        original_dir=str(original_dir),
    )

    assert not stale_path.exists()
    out_path = vis_dir / "f00003_iou1.000.png"
    assert out_path.exists()
    assert Image.open(out_path).size == (8, 2)


def test_cli_original_dir_overrides_alignment_default():
    assert select_original_dir(
        cli_original_dir="chickchicken/rgb/2x",
        frame_alignment={"original_dir": "chickchicken/rgb/1x"},
    ) == "chickchicken/rgb/2x"


def test_greedy_union_selects_multiple_clusters_when_iou_improves(tmp_path):
    pred_path = tmp_path / "000001.png"
    gt_path = tmp_path / "gt_000001.png"
    pred = np.array([[1, 1, 2], [2, 3, 3]], dtype=np.uint8)
    gt = np.array([[255, 255, 255], [255, 0, 0]], dtype=np.uint8)
    Image.fromarray(pred, mode="L").save(pred_path)
    Image.fromarray(gt, mode="L").save(gt_path)

    results = evaluate_single_object(
        {1: str(pred_path)},
        {1: str(gt_path)},
        K=3,
        selection_mode="greedy_union",
    )

    assert results["selected_clusters"] == [1, 2]
    assert results["best_cluster"] == 1
    assert results["mIoU"] == 1.0
