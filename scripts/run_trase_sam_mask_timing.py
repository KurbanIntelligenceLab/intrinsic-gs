#!/usr/bin/env python3
"""Run TRASE-style SAM mask extraction and record timing.

This follows upstream TRASE's extract_masks.py settings, but adds scene
discovery, resumable output, and CSV/JSON timing summaries.
"""

from __future__ import annotations

import csv
import json
import os
import time
from argparse import ArgumentParser
from pathlib import Path

import cv2
import torch
from bitarray import bitarray
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
from tqdm import tqdm


HYPERNERF_SCENES = [
    ("interp", "chickchicken"),
    ("interp", "cut-lemon1"),
    ("interp", "hand1-dense-v2"),
    ("interp", "slice-banana"),
    ("interp", "torchocolate"),
    ("misc", "americano"),
    ("misc", "espresso"),
    ("misc", "keyboard"),
    ("misc", "oven-mitts"),
    ("misc", "split-cookie"),
]


def image_files(rgb_dir: Path) -> list[Path]:
    suffixes = {".jpg", ".jpeg", ".png"}
    return [p for p in sorted(rgb_dir.iterdir()) if p.suffix.lower() in suffixes]


def save_masks(mask_list: list[torch.Tensor], out_path: Path) -> tuple[int, int, int]:
    if not mask_list:
        torch.save({"masks": bitarray(), "N": 0, "H": 0, "W": 0}, out_path)
        return 0, 0, 0
    masks = torch.stack(mask_list, dim=0)
    n_masks, height, width = masks.shape
    packed = {
        "masks": bitarray(masks.reshape(-1).cpu().numpy().tolist()),
        "N": n_masks,
        "H": height,
        "W": width,
    }
    torch.save(packed, out_path)
    return n_masks, height, width


def main() -> None:
    parser = ArgumentParser(description="TRASE-style SAM mask timing")
    parser.add_argument("--data-root", default="data/HyperNeRF")
    parser.add_argument("--rgb-scale", default="2x")
    parser.add_argument("--output-root", default="data/sam_trase_style_hypernerf_2x")
    parser.add_argument("--timing-dir", default="output/sam_mask_timing")
    parser.add_argument("--checkpoint", default="dependency/sam_vit_h_4b8939.pth")
    parser.add_argument("--sam-arch", default="vit_h")
    parser.add_argument("--iou-th", default=0.88, type=float)
    parser.add_argument("--stability-score-th", default=0.95, type=float)
    parser.add_argument("--points-per-side", default=32, type=int)
    parser.add_argument("--box-nms-th", default=0.7, type=float)
    parser.add_argument("--min-mask-region-area", default=100, type=int)
    parser.add_argument(
        "--scenes",
        nargs="*",
        default=None,
        help="Optional scene names to run. Defaults to all HyperNeRF scenes.",
    )
    parser.add_argument("--max-frames-per-scene", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    data_root = Path(args.data_root)
    output_root = Path(args.output_root)
    timing_dir = Path(args.timing_dir)
    timing_dir.mkdir(parents=True, exist_ok=True)

    total_start = time.perf_counter()
    model_start = time.perf_counter()
    sam = sam_model_registry[args.sam_arch](checkpoint=args.checkpoint).to("cuda")
    mask_generator = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=args.points_per_side,
        pred_iou_thresh=args.iou_th,
        box_nms_thresh=args.box_nms_th,
        stability_score_thresh=args.stability_score_th,
        crop_n_layers=0,
        crop_n_points_downscale_factor=1,
        min_mask_region_area=args.min_mask_region_area,
    )
    model_load_sec = time.perf_counter() - model_start

    requested = set(args.scenes) if args.scenes else None
    scene_specs = [
        (split, scene)
        for split, scene in HYPERNERF_SCENES
        if requested is None or scene in requested
    ]
    if requested:
        found = {scene for _, scene in scene_specs}
        missing = sorted(requested - found)
        if missing:
            raise SystemExit(f"Unknown scene(s): {', '.join(missing)}")

    rows = []
    for split, scene in scene_specs:
        rgb_dir = data_root / split / scene / "rgb" / args.rgb_scale
        files = image_files(rgb_dir)
        if args.max_frames_per_scene is not None:
            files = files[: args.max_frames_per_scene]
        scene_out = output_root / split / scene / "masks"
        scene_out.mkdir(parents=True, exist_ok=True)

        scene_start = time.perf_counter()
        processed = skipped = failed = total_masks = 0
        first_shape = ""
        per_frame_times = []
        for img_path in tqdm(files, desc=f"{split}/{scene}", unit="frame"):
            out_path = scene_out / f"{img_path.stem}.pt"
            if out_path.exists() and not args.overwrite:
                skipped += 1
                continue
            frame_start = time.perf_counter()
            img = cv2.imread(str(img_path))
            if img is None:
                failed += 1
                continue
            try:
                masks = mask_generator.generate(img)
                mask_list = []
                for mask in masks:
                    mask_tensor = torch.from_numpy(mask["segmentation"]).float().to("cuda")
                    if len(mask_tensor.unique()) >= 2:
                        mask_list.append(mask_tensor.bool())
                n_masks, height, width = save_masks(mask_list, out_path)
                if not first_shape and height and width:
                    first_shape = f"{height}x{width}"
                total_masks += n_masks
                processed += 1
                per_frame_times.append(time.perf_counter() - frame_start)
            except Exception as exc:
                failed += 1
                err_path = scene_out / f"{img_path.stem}.error.txt"
                err_path.write_text(str(exc) + "\n")

        scene_sec = time.perf_counter() - scene_start
        denom = max(processed, 1)
        row = {
            "split": split,
            "scene": scene,
            "rgb_dir": str(rgb_dir),
            "output_dir": str(scene_out),
            "frames_found": len(files),
            "processed": processed,
            "skipped": skipped,
            "failed": failed,
            "total_masks": total_masks,
            "mean_masks_per_frame": total_masks / denom,
            "scene_total_sec": scene_sec,
            "mean_processed_frame_sec": sum(per_frame_times) / denom,
            "first_mask_shape": first_shape,
        }
        rows.append(row)

        with (timing_dir / "sam_trase_style_timing_partial.json").open("w") as fh:
            json.dump({"model_load_sec": model_load_sec, "rows": rows}, fh, indent=2)

    total_sec = time.perf_counter() - total_start
    summary = {
        "method": "TRASE-style SAM automatic mask generation",
        "checkpoint": args.checkpoint,
        "sam_arch": args.sam_arch,
        "rgb_scale": args.rgb_scale,
        "model_load_sec": model_load_sec,
        "total_wall_sec": total_sec,
        "total_processed_frames": sum(r["processed"] for r in rows),
        "total_failed_frames": sum(r["failed"] for r in rows),
        "total_masks": sum(r["total_masks"] for r in rows),
        "output_root": str(output_root),
        "rows": rows,
    }
    json_path = timing_dir / "sam_trase_style_timing.json"
    csv_path = timing_dir / "sam_trase_style_timing.csv"
    json_path.write_text(json.dumps(summary, indent=2))
    with csv_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()) if rows else [])
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    main()
