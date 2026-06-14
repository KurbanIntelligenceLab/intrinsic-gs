#!/usr/bin/env python3
"""Render RGB videos with SAM3 mask overlays for visual QA."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


DEFAULT_SCENES = {
    "slice-banana": "data/HyperNeRF/interp/slice-banana/rgb/2x",
    "cut-lemon1": "data/HyperNeRF/interp/cut-lemon1/rgb/2x",
    "hand1-dense-v2": "data/HyperNeRF/interp/hand1-dense-v2/rgb/2x",
    "oven-mitts": "data/HyperNeRF/misc/oven-mitts/rgb/2x",
}

PALETTE = np.array(
    [
        (230, 57, 70),
        (42, 157, 143),
        (245, 166, 35),
        (67, 97, 238),
        (156, 39, 176),
        (46, 196, 182),
        (255, 214, 10),
        (255, 112, 67),
        (76, 175, 80),
        (3, 169, 244),
        (233, 30, 99),
        (139, 195, 74),
    ],
    dtype=np.float32,
)


def frame_number(key: str) -> int:
    match = re.search(r"(\d+)$", key)
    return int(match.group(1)) if match else 0


def load_rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"))


def normalize_masks(masks: np.ndarray, height: int, width: int) -> np.ndarray:
    if masks.ndim == 2:
        masks = masks[None]
    if masks.size == 0:
        return np.zeros((0, height, width), dtype=bool)
    masks = masks.astype(bool, copy=False)
    if masks.shape[-2:] == (height, width):
        return masks
    if masks.shape[-2:] == (width, height):
        return np.transpose(masks, (0, 2, 1))

    resized = []
    for mask in masks:
        resized.append(
            cv2.resize(
                mask.astype(np.uint8),
                (width, height),
                interpolation=cv2.INTER_NEAREST,
            ).astype(bool)
        )
    return np.stack(resized, axis=0) if resized else np.zeros((0, height, width), dtype=bool)


def overlay_masks(rgb: np.ndarray, masks: np.ndarray, alpha: float) -> np.ndarray:
    out = rgb.astype(np.float32)
    for idx, mask in enumerate(masks):
        if not mask.any():
            continue
        color = PALETTE[idx % len(PALETTE)]
        out[mask] = out[mask] * (1.0 - alpha) + color * alpha

    out_u8 = np.clip(out, 0, 255).astype(np.uint8)
    for idx, mask in enumerate(masks):
        if not mask.any():
            continue
        color = tuple(int(c) for c in PALETTE[idx % len(PALETTE)][::-1])
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out_u8, contours, -1, color, 1, lineType=cv2.LINE_AA)
    return out_u8


def draw_label(frame_rgb: np.ndarray, text: str) -> np.ndarray:
    bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.55
    thickness = 1
    (text_w, text_h), baseline = cv2.getTextSize(text, font, scale, thickness)
    pad = 8
    cv2.rectangle(bgr, (0, 0), (text_w + pad * 2, text_h + baseline + pad * 2), (0, 0, 0), -1)
    cv2.putText(
        bgr,
        text,
        (pad, pad + text_h),
        font,
        scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA,
    )
    return bgr


def render_scene(
    scene: str,
    rgb_dir: Path,
    mask_path: Path,
    out_path: Path,
    fps: float,
    alpha: float,
    max_frames: int | None,
) -> None:
    rgb_paths = sorted(p for p in rgb_dir.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"})
    if not rgb_paths:
        raise FileNotFoundError(f"No RGB frames in {rgb_dir}")
    if not mask_path.exists():
        raise FileNotFoundError(mask_path)

    masks_npz = np.load(mask_path, allow_pickle=False)
    keys = sorted(masks_npz.files, key=frame_number)
    frame_count = min(len(rgb_paths), len(keys))
    if max_frames:
        frame_count = min(frame_count, max_frames)
    if frame_count == 0:
        raise RuntimeError(f"No matching frames for {scene}")

    first = load_rgb(rgb_paths[0])
    height, width = first.shape[:2]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(out_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer for {out_path}")

    try:
        for idx in range(frame_count):
            rgb = load_rgb(rgb_paths[idx])
            if rgb.shape[:2] != (height, width):
                rgb = np.asarray(Image.fromarray(rgb).resize((width, height), Image.Resampling.BILINEAR))
            masks = normalize_masks(masks_npz[keys[idx]], height, width)
            overlay = overlay_masks(rgb, masks, alpha)
            label = f"{scene} frame {idx + 1}/{frame_count} masks={masks.shape[0]}"
            writer.write(draw_label(overlay, label))
    finally:
        writer.release()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mask-root", type=Path, default=Path("data/sam3_hypernerf_missing"))
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/sam3_mask_overlay_videos"))
    parser.add_argument("--fps", type=float, default=15.0)
    parser.add_argument("--alpha", type=float, default=0.45)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--scene", action="append", choices=sorted(DEFAULT_SCENES), help="Render only this scene")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    scenes = args.scene or list(DEFAULT_SCENES)
    for scene in scenes:
        out_path = args.out_dir / f"{scene}_sam3_overlay.mp4"
        render_scene(
            scene=scene,
            rgb_dir=Path(DEFAULT_SCENES[scene]),
            mask_path=args.mask_root / scene / "masks.npz",
            out_path=out_path,
            fps=args.fps,
            alpha=args.alpha,
            max_frames=args.max_frames,
        )
        print(out_path)


if __name__ == "__main__":
    main()
