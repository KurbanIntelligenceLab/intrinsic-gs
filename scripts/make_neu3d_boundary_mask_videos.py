#!/usr/bin/env python3
"""Create Neu3D boundary and mask comparison videos from rendered cluster IDs."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np
from PIL import Image


SCENES = [
    "coffee_martini",
    "cook_spinach",
    "cut_roasted_beef",
    "flame_steak",
    "sear_steak",
]


def natural_key(path: Path) -> list[object]:
    return [int(s) if s.isdigit() else s for s in re.split(r"(\d+)", path.name)]


def read_rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"))


def read_label(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path))


def read_mask(path: Path, shape: tuple[int, int]) -> np.ndarray:
    if not path.exists():
        return np.zeros(shape, dtype=bool)
    mask = np.asarray(Image.open(path).convert("L")) > 0
    if mask.shape != shape:
        mask = cv2.resize(mask.astype(np.uint8), (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST) > 0
    return mask


def selected_clusters(run_dir: Path, greedy: bool = False) -> list[int]:
    result = run_dir / ("miou_results_greedy.json" if greedy else "miou_results.json")
    with result.open() as f:
        data = json.load(f)
    clusters = data["results"].get("selected_clusters")
    if clusters:
        return [int(k) for k in clusters]
    return [int(data["results"]["best_cluster"])]


def selected_miou(run_dir: Path, greedy: bool = False) -> float:
    result = run_dir / ("miou_results_greedy.json" if greedy else "miou_results.json")
    with result.open() as f:
        data = json.load(f)
    return float(data["results"]["mIoU"])


def latest_run(root: Path, variant: str, scene: str) -> Path:
    scene_root = root / "runs" / variant / scene
    runs = sorted([p for p in scene_root.iterdir() if p.is_dir()])
    if not runs:
        raise FileNotFoundError(f"No runs under {scene_root}")
    return runs[-1]


def edge_panel(rgb: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (0, 0), 1.0)
    gx = cv2.Sobel(blur, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(blur, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    q = np.percentile(mag, 98)
    if q > 0:
        mag = np.clip(mag / q, 0, 1)
    edges = (mag * 255).astype(np.uint8)
    return cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)


def overlay(rgb: np.ndarray, pred: np.ndarray, gt: np.ndarray | None = None) -> np.ndarray:
    out = rgb.copy().astype(np.float32)
    blue = np.array([35, 165, 255], dtype=np.float32)
    yellow = np.array([255, 215, 30], dtype=np.float32)
    red = np.array([255, 55, 55], dtype=np.float32)
    out[pred] = 0.55 * out[pred] + 0.45 * blue
    if gt is not None:
        both = pred & gt
        gt_only = gt & ~pred
        out[gt_only] = 0.55 * out[gt_only] + 0.45 * red
        out[both] = 0.45 * out[both] + 0.55 * yellow
    return np.clip(out, 0, 255).astype(np.uint8)


def fit_panel(img: np.ndarray, width: int) -> np.ndarray:
    h, w = img.shape[:2]
    height = int(round(h * width / w))
    return cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)


def label_panel(img: np.ndarray, text: str) -> np.ndarray:
    out = img.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 34), (0, 0, 0), -1)
    cv2.putText(out, text, (10, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2, cv2.LINE_AA)
    return out


def make_scene_video(
    *,
    scene: str,
    data_root: Path,
    run_root: Path,
    out_dir: Path,
    panel_width: int,
    fps: int,
    max_frames: int | None,
) -> dict[str, object]:
    default_run = latest_run(run_root, "default", scene)
    best_run = latest_run(run_root, "no_motion_no_boundary", scene)
    default_clusters = selected_clusters(default_run)
    best_clusters = selected_clusters(best_run)

    default_ids = sorted((default_run / "cluster_ids_test").glob("*.png"), key=natural_key)
    best_ids = sorted((best_run / "cluster_ids_test").glob("*.png"), key=natural_key)
    if len(default_ids) != len(best_ids):
        raise ValueError(f"{scene}: mismatched frame counts: default={len(default_ids)} best={len(best_ids)}")
    frames = list(zip(default_ids, best_ids))
    if max_frames:
        frames = frames[:max_frames]

    gt_paths = sorted((data_root / scene / "gt_masks").glob("*.png"), key=natural_key)
    out_path = out_dir / f"{scene}_boundary_mask_compare.mp4"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with imageio.get_writer(out_path, fps=fps, codec="libx264", quality=8, macro_block_size=16) as writer:
        for idx, (default_id_path, best_id_path) in enumerate(frames):
            rgb_path = data_root / scene / "images_2x" / default_id_path.name
            rgb = read_rgb(rgb_path)
            h, w = rgb.shape[:2]
            gt = read_mask(gt_paths[idx], (h, w)) if idx < len(gt_paths) else None

            default_ids_arr = read_label(default_id_path)
            best_ids_arr = read_label(best_id_path)
            default_pred = np.isin(default_ids_arr, default_clusters)
            best_pred = np.isin(best_ids_arr, best_clusters)

            panels = [
                label_panel(fit_panel(rgb, panel_width), "RGB"),
                label_panel(fit_panel(edge_panel(rgb), panel_width), "Sobel boundary map"),
                label_panel(
                    fit_panel(overlay(rgb, default_pred, gt), panel_width),
                    f"full mask k={','.join(map(str, default_clusters))}",
                ),
                label_panel(
                    fit_panel(overlay(rgb, best_pred, gt), panel_width),
                    f"no motion/boundary k={','.join(map(str, best_clusters))}",
                ),
            ]
            writer.append_data(np.concatenate(panels, axis=1))

    return {
        "scene": scene,
        "frames": len(frames),
        "video": str(out_path),
        "default_miou": selected_miou(default_run),
        "best_miou": selected_miou(best_run),
        "default_clusters": " ".join(map(str, default_clusters)),
        "best_clusters": " ".join(map(str, best_clusters)),
        "default_run": str(default_run),
        "best_run": str(best_run),
    }


def render_dir(run_dir: Path) -> Path:
    matches = sorted(run_dir.glob("renders_k*_loaded_test"))
    if not matches:
        raise FileNotFoundError(f"No renders_k*_loaded_test under {run_dir}")
    return matches[-1]


def make_scene_all_masks_video(
    *,
    scene: str,
    data_root: Path,
    run_root: Path,
    out_dir: Path,
    panel_width: int,
    fps: int,
    max_frames: int | None,
) -> dict[str, object]:
    default_run = latest_run(run_root, "default", scene)
    best_run = latest_run(run_root, "no_motion_no_boundary", scene)
    default_renders = sorted(render_dir(default_run).glob("*.png"), key=natural_key)
    best_renders = sorted(render_dir(best_run).glob("*.png"), key=natural_key)
    if len(default_renders) != len(best_renders):
        raise ValueError(f"{scene}: mismatched frame counts: default={len(default_renders)} best={len(best_renders)}")
    frames = list(zip(default_renders, best_renders))
    if max_frames:
        frames = frames[:max_frames]

    out_path = out_dir / f"{scene}_boundary_all_masks_compare.mp4"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with imageio.get_writer(out_path, fps=fps, codec="libx264", quality=8, macro_block_size=16) as writer:
        for default_render_path, best_render_path in frames:
            rgb_path = data_root / scene / "images_2x" / default_render_path.name
            rgb = read_rgb(rgb_path)
            default_palette = read_rgb(default_render_path)
            best_palette = read_rgb(best_render_path)

            panels = [
                label_panel(fit_panel(rgb, panel_width), "RGB"),
                label_panel(fit_panel(edge_panel(rgb), panel_width), "Sobel boundary map"),
                label_panel(fit_panel(default_palette, panel_width), "full all masks"),
                label_panel(fit_panel(best_palette, panel_width), "no motion/boundary all masks"),
            ]
            writer.append_data(np.concatenate(panels, axis=1))

    return {
        "scene": scene,
        "frames": len(frames),
        "video": str(out_path),
        "default_miou": selected_miou(default_run),
        "best_miou": selected_miou(best_run),
        "default_run": str(default_run),
        "best_run": str(best_run),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("data/Neu3D"))
    parser.add_argument("--run-root", type=Path, default=Path("output/final_neu3d_leiden018_component_ablations"))
    parser.add_argument("--out-dir", type=Path, default=Path("output/neu3d_boundary_mask_videos"))
    parser.add_argument("--scenes", nargs="*", default=SCENES)
    parser.add_argument("--panel-width", type=int, default=384)
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--all-masks", action="store_true")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    make_fn = make_scene_all_masks_video if args.all_masks else make_scene_video
    rows = [
        make_fn(
            scene=scene,
            data_root=args.data_root,
            run_root=args.run_root,
            out_dir=args.out_dir,
            panel_width=args.panel_width,
            fps=args.fps,
            max_frames=args.max_frames,
        )
        for scene in args.scenes
    ]

    summary = args.out_dir / "summary.csv"
    with summary.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {summary}")
    for row in rows:
        print(row["video"])


if __name__ == "__main__":
    main()
