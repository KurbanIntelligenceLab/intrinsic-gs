#!/usr/bin/env python3
"""Render RGB videos with predicted cluster-mask overlays."""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


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


def numeric_stem(path: Path) -> int | None:
    if re.fullmatch(r"\d+", path.stem):
        return int(path.stem)
    return None


def image(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"))


def mask_image(path: Path) -> np.ndarray:
    arr = np.asarray(Image.open(path))
    if arr.ndim == 3:
        arr = arr[..., 0]
    return arr


def resize_nearest(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    height, width = shape
    if mask.shape == (height, width):
        return mask
    return cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)


def cluster_palette(max_label: int) -> np.ndarray:
    labels = np.arange(max_label + 1, dtype=np.uint32)
    values = labels * np.uint32(2654435761)
    colors = np.stack(
        [
            (values & 255),
            ((values >> 8) & 255),
            ((values >> 16) & 255),
        ],
        axis=1,
    ).astype(np.uint8)
    colors = ((colors.astype(np.uint16) + 80) % 256).astype(np.uint8)
    return colors


def overlay_single(rgb: np.ndarray, mask: np.ndarray, alpha: float) -> np.ndarray:
    out = rgb.astype(np.float32)
    color = np.array([0, 210, 255], dtype=np.float32)
    edge_color = (0, 180, 255)
    out[mask] = out[mask] * (1.0 - alpha) + color * alpha
    out = np.clip(out, 0, 255).astype(np.uint8)
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, contours, -1, edge_color, 1, lineType=cv2.LINE_AA)
    return out


def overlay_full(rgb: np.ndarray, pred: np.ndarray, alpha: float, draw_edges: bool = True) -> np.ndarray:
    pred = pred.astype(np.int64, copy=False)
    max_label = int(pred.max(initial=0))
    colors = cluster_palette(max_label)
    color_img = colors[np.clip(pred, 0, max_label)]
    valid = pred >= 0

    out = rgb.astype(np.float32)
    out[valid] = out[valid] * (1.0 - alpha) + color_img[valid].astype(np.float32) * alpha
    out = np.clip(out, 0, 255).astype(np.uint8)

    if draw_edges:
        edges = np.zeros(pred.shape, dtype=np.uint8)
        edges[:, 1:] |= pred[:, 1:] != pred[:, :-1]
        edges[1:, :] |= pred[1:, :] != pred[:-1, :]
        out[edges.astype(bool)] = np.array([255, 255, 255], dtype=np.uint8)
    return out


def label(frame_rgb: np.ndarray, text: str) -> np.ndarray:
    bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.55
    thickness = 1
    (tw, th), base = cv2.getTextSize(text, font, scale, thickness)
    pad = 8
    cv2.rectangle(bgr, (0, 0), (tw + pad * 2, th + base + pad * 2), (0, 0, 0), -1)
    cv2.putText(bgr, text, (pad, pad + th), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)
    return bgr


def rgb_index(rgb_dir: Path) -> dict[int, Path]:
    out: dict[int, Path] = {}
    for path in sorted(rgb_dir.iterdir()):
        if path.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
            continue
        n = numeric_stem(path)
        if n is not None:
            out[n] = path
    return out


def choose_rgb(fid: int, rgbs: dict[int, Path]) -> Path | None:
    return rgbs.get(fid) or rgbs.get(fid + 1) or rgbs.get(fid - 1)


def render_scene(
    row: dict[str, str],
    data_root: Path,
    out_dir: Path,
    fps: float,
    alpha: float,
    mode: str,
) -> Path:
    scene = row["scene"]
    target = row["target"]
    pred_dir = Path(row["pred_dir"])
    k = int(row["best_k"])
    split = SCENE_SPLITS[scene]
    rgb_dir = data_root / split / scene / "rgb" / "2x"
    rgbs = rgb_index(rgb_dir)
    pred_paths = []
    for path in sorted(pred_dir.glob("*.png")):
        fid = numeric_stem(path)
        if fid is None:
            continue
        rgb_path = choose_rgb(fid, rgbs)
        if rgb_path is not None:
            pred_paths.append((fid, path, rgb_path))
    if not pred_paths:
        raise RuntimeError(f"No RGB/pred frame matches for {scene}")

    first_rgb = image(pred_paths[0][2])
    height, width = first_rgb.shape[:2]
    out_dir.mkdir(parents=True, exist_ok=True)
    if mode == "single":
        out_name = f"{scene}_pred_{target}_k{k}_miou{float(row['best_miou']):.3f}.mp4"
    else:
        out_name = f"{scene}_full_pred_clusters.mp4"
    out_path = out_dir / out_name
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open {out_path}")

    try:
        total = len(pred_paths)
        for idx, (fid, pred_path, rgb_path) in enumerate(pred_paths, start=1):
            rgb = image(rgb_path)
            if rgb.shape[:2] != (height, width):
                rgb = np.asarray(Image.fromarray(rgb).resize((width, height), Image.Resampling.BILINEAR))
            pred = resize_nearest(mask_image(pred_path), (height, width))
            if mode == "single":
                pred_mask = pred == k
                frame = overlay_single(rgb, pred_mask, alpha)
                text = f"{scene} pred={target} k={k} frame={idx}/{total}"
            else:
                frame = overlay_full(rgb, pred, alpha)
                text = f"{scene} full predicted clusters frame={idx}/{total}"
            writer.write(label(frame, text))
    finally:
        writer.release()
    return out_path


def load_top_rows(csv_path: Path) -> list[dict[str, str]]:
    by_scene: dict[str, list[dict[str, str]]] = {}
    with csv_path.open(newline="") as f:
        for row in csv.DictReader(f):
            by_scene.setdefault(row["scene"], []).append(row)
    rows = []
    for scene in SCENE_SPLITS:
        candidates = by_scene[scene]
        rows.append(max(candidates, key=lambda row: float(row["best_miou"])))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", default="output/final_hypernerf_1/combined_zip6_plus_sam3new4_res003.csv")
    parser.add_argument("--data-root", default="data/HyperNeRF")
    parser.add_argument("--out-dir", default="output/final_hypernerf_1/prediction_overlay_videos")
    parser.add_argument("--fps", type=float, default=15.0)
    parser.add_argument("--alpha", type=float, default=0.45)
    parser.add_argument("--mode", choices=["single", "full"], default="single")
    args = parser.parse_args()

    for row in load_top_rows(Path(args.csv)):
        out = render_scene(row, Path(args.data_root), Path(args.out_dir), args.fps, args.alpha, args.mode)
        print(out)


if __name__ == "__main__":
    main()
