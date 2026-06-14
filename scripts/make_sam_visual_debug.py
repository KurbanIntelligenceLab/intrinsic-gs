#!/usr/bin/env python3
"""Create RGB/GT/prediction contact sheets for SAM-object eval rows."""

from __future__ import annotations

import argparse
import csv
import io
import re
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def parse_pair(text: str) -> tuple[str, str]:
    if ":" not in text:
        raise argparse.ArgumentTypeError("pairs must look like scene:object")
    scene, obj = text.split(":", 1)
    return scene, obj


def image_array(path: Path) -> np.ndarray:
    arr = np.asarray(Image.open(path).convert("RGB"))
    return arr


def find_rgb(root: Path, scene: str, fid: int) -> Path:
    for family in ("misc", "interp"):
        for stem in (f"{fid:06d}", f"{fid - 1:06d}"):
            path = root / family / scene / "rgb" / "2x" / f"{stem}.png"
            if path.exists():
                return path
    raise FileNotFoundError(f"Could not find RGB for {scene} frame {fid}")


def read_gt(zip_file: zipfile.ZipFile, scene: str, obj: str, fid: int) -> np.ndarray | None:
    name = f"sam_masks_hypernerf/{scene}/masks_{obj}.npz"
    npz = np.load(io.BytesIO(zip_file.read(name)), allow_pickle=False)
    for key in (f"frame_{fid - 1:06d}", f"frame_{fid:06d}"):
        if key in npz.files:
            gt = npz[key]
            if gt.ndim == 3:
                gt = np.any(gt, axis=0)
            return gt.astype(bool)
    return None


def tint(mask: np.ndarray, color: tuple[int, int, int]) -> np.ndarray:
    out = np.zeros((*mask.shape, 3), dtype=np.uint8)
    out[mask] = color
    return out


def overlay(rgb: np.ndarray, gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
    out = rgb.copy().astype(np.float32)
    only_gt = gt & ~pred
    only_pred = pred & ~gt
    both = gt & pred
    out[only_gt] = out[only_gt] * 0.35 + np.array([255, 40, 40]) * 0.65
    out[only_pred] = out[only_pred] * 0.35 + np.array([30, 190, 255]) * 0.65
    out[both] = out[both] * 0.35 + np.array([255, 230, 30]) * 0.65
    return np.clip(out, 0, 255).astype(np.uint8)


def label(tile: Image.Image, text: str) -> Image.Image:
    tile = tile.copy()
    draw = ImageDraw.Draw(tile)
    draw.rectangle((0, 0, tile.width, 18), fill=(0, 0, 0))
    draw.text((5, 3), text, fill=(255, 255, 255))
    return tile


def resized(arr: np.ndarray, width: int) -> Image.Image:
    im = Image.fromarray(arr)
    height = int(round(im.height * (width / im.width)))
    return im.resize((width, height), Image.Resampling.BILINEAR)


def make_sheet(row: dict[str, str], zip_file: zipfile.ZipFile, rgb_root: Path, out_dir: Path, frames: int) -> Path:
    scene = row["scene"]
    obj = row["object"]
    pred_dir = Path(row["pred_dir"])
    k = int(row["best_k"])

    pred_paths = []
    for path in sorted(pred_dir.glob("*.png")):
        if re.match(r"^\d+$", path.stem):
            pred_paths.append((int(path.stem), path))
    if len(pred_paths) > frames:
        idxs = np.linspace(0, len(pred_paths) - 1, frames).round().astype(int)
        pred_paths = [pred_paths[i] for i in idxs]

    rows = []
    for fid, pred_path in pred_paths:
        gt = read_gt(zip_file, scene, obj, fid)
        if gt is None:
            continue
        pred = np.asarray(Image.open(pred_path))
        if pred.ndim == 3:
            pred = pred[..., 0]
        pred_mask = pred == k
        rgb = image_array(find_rgb(rgb_root, scene, fid))
        if rgb.shape[:2] != gt.shape:
            rgb = np.asarray(Image.fromarray(rgb).resize((gt.shape[1], gt.shape[0]), Image.Resampling.BILINEAR))
        if pred_mask.shape != gt.shape:
            continue

        cols = [
            label(resized(rgb, 240), f"{scene} {fid:06d} rgb"),
            label(resized(tint(gt, (40, 220, 80)), 240), f"GT {obj}"),
            label(resized(tint(pred_mask, (30, 190, 255)), 240), f"pred k={k}"),
            label(resized(overlay(rgb, gt, pred_mask), 240), "red GT / blue pred / yellow both"),
        ]
        rows.append(cols)

    if not rows:
        raise RuntimeError(f"No visual rows created for {scene}:{obj}")

    gap = 6
    w = sum(c.width for c in rows[0]) + gap * (len(rows[0]) - 1)
    h = sum(r[0].height for r in rows) + gap * (len(rows) - 1)
    sheet = Image.new("RGB", (w, h), (245, 245, 245))
    y = 0
    for row_tiles in rows:
        x = 0
        for tile in row_tiles:
            sheet.paste(tile, (x, y))
            x += tile.width + gap
        y += row_tiles[0].height + gap

    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{scene}_{obj}_k{k}_miou{float(row['best_miou']):.3f}.png"
    sheet.save(out)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", default="output/sam_mask_eval/no_mask_candidate_sweep_best_cluster.csv")
    parser.add_argument("--candidate", default="leiden_base")
    parser.add_argument("--zip", default="/workspace/sam_masks_hypernerf.zip")
    parser.add_argument("--rgb-root", default="data/HyperNeRF")
    parser.add_argument("--out-dir", default="output/sam_mask_eval/visual_debug")
    parser.add_argument("--frames", type=int, default=6)
    parser.add_argument("--pair", action="append", type=parse_pair, required=True)
    args = parser.parse_args()

    with open(args.csv, newline="") as f:
        all_rows = list(csv.DictReader(f))
    rows = {}
    for row in all_rows:
        if row["candidate"] == args.candidate:
            rows[(row["scene"], row["object"])] = row

    with zipfile.ZipFile(args.zip) as zip_file:
        for pair in args.pair:
            if pair not in rows:
                raise KeyError(f"No row for {args.candidate}:{pair[0]}:{pair[1]}")
            out = make_sheet(rows[pair], zip_file, Path(args.rgb_root), Path(args.out_dir), args.frames)
            print(out)


if __name__ == "__main__":
    main()
