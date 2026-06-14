#!/usr/bin/env python3
"""Evaluate cluster-ID masks with TRASE's Mask-Benchmark metric convention."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from self_supervised_scripts.compute_miou import (
    align_single_object_gt_paths,
    index_pngs,
    load_binary_mask,
    load_cluster_id_map,
)


SCENE_ORDER = (
    "chickchicken",
    "cut-lemon1",
    "hand1-dense-v2",
    "slice-banana",
    "torchocolate",
    "americano",
    "espresso",
    "keyboard",
    "oven-mitts",
    "split-cookie",
)

PAPER_TRASE_HYPERNERF = {
    "americano": (0.8144, 0.9922),
    "chickchicken": (0.9308, 0.9835),
    "cut-lemon1": (0.8795, 0.9769),
    "espresso": (0.7164, 0.9860),
    "hand1-dense-v2": (0.9006, 0.9849),
    "keyboard": (0.8952, 0.9827),
    "oven-mitts": (0.9295, 0.9876),
    "slice-banana": (0.8782, 0.9617),
    "split-cookie": (0.8581, 0.9930),
    "torchocolate": (0.8599, 0.9964),
}

SCENE_SPLITS = {
    "chickchicken": "interp",
    "cut-lemon1": "interp",
    "hand1-dense-v2": "interp",
    "slice-banana": "interp",
    "torchocolate": "interp",
    "americano": "misc",
    "espresso": "misc",
    "keyboard": "misc",
    "oven-mitts": "misc",
    "split-cookie": "misc",
}


def max_cluster_id(pred_paths: dict[int, str]) -> int:
    k_max = 0
    for path in pred_paths.values():
        pred = load_cluster_id_map(path)
        k_max = max(k_max, int(pred.max()))
    return k_max


def frame_iou_acc(pred: np.ndarray, gt: np.ndarray) -> tuple[float, float]:
    inter = int(np.logical_and(pred, gt).sum())
    union = int(np.logical_or(pred, gt).sum())
    iou = float(inter / union) if union > 0 else 0.0
    acc = float((pred == gt).sum() / gt.size) if gt.size else 0.0
    return iou, acc


def evaluate_scene(pred_dir: Path, gt_dir: Path) -> dict[str, object]:
    pred_paths = index_pngs(str(pred_dir))
    raw_gt_paths = index_pngs(str(gt_dir))
    gt_paths, alignment = align_single_object_gt_paths(pred_paths, raw_gt_paths, str(gt_dir))
    matched = sorted(set(pred_paths) & set(gt_paths))
    if not matched:
        raise RuntimeError(f"No matched frames for pred={pred_dir} gt={gt_dir}")

    k_max = max_cluster_id(pred_paths)
    iou_by_k: list[float] = [0.0] * (k_max + 1)
    acc_by_k: list[float] = [0.0] * (k_max + 1)

    gt_cache = {fid: load_binary_mask(gt_paths[fid]) for fid in matched}
    pred_cache = {fid: load_cluster_id_map(pred_paths[fid]) for fid in matched}

    valid_matched = []
    for fid in matched:
        if pred_cache[fid].shape == gt_cache[fid].shape:
            valid_matched.append(fid)
    if not valid_matched:
        raise RuntimeError(f"All matched frames had shape mismatch for pred={pred_dir} gt={gt_dir}")

    for k in range(1, k_max + 1):
        frame_ious = []
        frame_accs = []
        for fid in valid_matched:
            pred = pred_cache[fid] == k
            gt = gt_cache[fid]
            iou, acc = frame_iou_acc(pred, gt)
            frame_ious.append(iou)
            frame_accs.append(acc)
        iou_by_k[k] = float(np.mean(frame_ious))
        acc_by_k[k] = float(np.mean(frame_accs))

    best_k = int(np.argmax(iou_by_k[1:]) + 1) if k_max else 0
    return {
        "best_k": best_k,
        "trase_style_miou": iou_by_k[best_k] if best_k else 0.0,
        "trase_style_macc": acc_by_k[best_k] if best_k else 0.0,
        "matched_frames": len(valid_matched),
        "K_seen": k_max,
        "alignment": alignment.get("method", "unknown"),
    }


def load_run_rows(path: Path) -> dict[str, dict[str, str]]:
    with path.open(newline="") as f:
        rows = {row["scene"]: row for row in csv.DictReader(f)}
    missing = [scene for scene in SCENE_ORDER if scene not in rows]
    if missing:
        raise RuntimeError(f"Missing scenes in {path}: {', '.join(missing)}")
    return rows


def write_markdown(rows: list[dict[str, object]], out_md: Path) -> None:
    mean_miou = float(np.mean([float(row["trase_style_miou"]) for row in rows]))
    mean_macc = float(np.mean([float(row["trase_style_macc"]) for row in rows]))
    paper_mean_miou = float(np.mean([PAPER_TRASE_HYPERNERF[str(row["scene"])][0] for row in rows]))
    paper_mean_macc = float(np.mean([PAPER_TRASE_HYPERNERF[str(row["scene"])][1] for row in rows]))

    lines = [
        "# HyperNeRF Mask-Benchmark, TRASE-style evaluation",
        "",
        "GT masks: `data/Mask-Benchmark/Mask-Benchmark/HyperNeRF-Mask/<scene>/gt_masks`.",
        "",
        "Metric: for each scene, select the best single predicted cluster by mean per-frame IoU, then report mean per-frame IoU and mean per-frame pixel accuracy. This matches the averaging convention in TRASE's released `metrics_segmentation.py`.",
        "",
        f"- Our res0.03 mean mIoU: **{mean_miou:.4f}**",
        f"- Our res0.03 mean mAcc: **{mean_macc:.4f}**",
        f"- Paper TRASE mean mIoU: **{paper_mean_miou:.4f}**",
        f"- Paper TRASE mean mAcc: **{paper_mean_macc:.4f}**",
        "",
        "| Scene | Our mIoU | Our mAcc | Best k | Paper TRASE mIoU | Paper TRASE mAcc | Frames |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        scene = str(row["scene"])
        paper_iou, paper_acc = PAPER_TRASE_HYPERNERF[scene]
        lines.append(
            f"| {scene} | {float(row['trase_style_miou']):.4f} | "
            f"{float(row['trase_style_macc']):.4f} | {row['best_k']} | "
            f"{paper_iou:.4f} | {paper_acc:.4f} | {row['matched_frames']} |"
        )
    out_md.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", default="output/final_hypernerf_1/maskbenchmark_full10_res003.csv")
    parser.add_argument("--out-csv", default="output/final_hypernerf_1/maskbenchmark_full10_res003_trase_style.csv")
    parser.add_argument("--out-md", default="output/final_hypernerf_1/maskbenchmark_full10_res003_trase_style.md")
    parser.add_argument("--data-root", default="data/HyperNeRF")
    parser.add_argument(
        "--pred-subdir",
        default="",
        help="Optional subdir under each run_dir containing cluster_ids_train, e.g. objectness_longrange_default.",
    )
    args = parser.parse_args()

    source_rows = load_run_rows(Path(args.input_csv))
    rows = []
    for scene in SCENE_ORDER:
        source = source_rows[scene]
        pred_root = Path(source["run_dir"]) / args.pred_subdir if args.pred_subdir else Path(source["run_dir"])
        pred_dir = pred_root / "cluster_ids_train"
        gt_dir = Path(args.data_root) / SCENE_SPLITS[scene] / scene / "gt_masks"
        if not gt_dir.exists():
            gt_dir = Path(source["mask_dir"])
        result = evaluate_scene(pred_dir, gt_dir)
        row = {
            "scene": scene,
            "pred_dir": str(pred_dir),
            "gt_dir": str(gt_dir),
            **result,
        }
        rows.append(row)
        print(f"{scene}: mIoU={row['trase_style_miou']:.4f} mAcc={row['trase_style_macc']:.4f} k={row['best_k']}")

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    columns = ["scene", "trase_style_miou", "trase_style_macc", "best_k", "matched_frames", "K_seen", "alignment", "pred_dir", "gt_dir"]
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)

    write_markdown(rows, Path(args.out_md))
    print(f"Mean mIoU: {np.mean([float(row['trase_style_miou']) for row in rows]):.4f}")
    print(f"Mean mAcc: {np.mean([float(row['trase_style_macc']) for row in rows]):.4f}")
    print(f"Wrote {out_csv}")
    print(f"Wrote {args.out_md}")


if __name__ == "__main__":
    main()
