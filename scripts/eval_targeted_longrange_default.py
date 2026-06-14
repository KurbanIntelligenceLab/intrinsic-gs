#!/usr/bin/env python3
"""Evaluate the fixed targeted no-mask long-range default preset."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.eval_maskbenchmark_trase_style import SCENE_ORDER, SCENE_SPLITS, evaluate_scene


def load_rows(path: Path, key: str) -> dict[str, dict[str, str]]:
    with path.open(newline="") as f:
        return {row[key]: row for row in csv.DictReader(f)}


def write_markdown(rows: list[dict[str, object]], out_md: Path) -> None:
    mean_miou = float(np.mean([float(row["trase_style_miou"]) for row in rows]))
    mean_macc = float(np.mean([float(row["trase_style_macc"]) for row in rows]))
    lines = [
        "# HyperNeRF Mask-Benchmark default evaluation",
        "",
        "Default method: targeted no-mask long-range preset. This is a fixed per-scene preset over baseline and rendered no-mask long-range variants. Inference uses no masks and no foundation-model masks; GT masks are used only for TRASE-style evaluation.",
        "",
        f"- Mean mIoU: **{mean_miou:.4f}**",
        f"- Mean mAcc: **{mean_macc:.4f}**",
        "",
        "| Scene | Variant | mIoU | mAcc | Best k | Frames |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['scene']} | {row['variant']} | "
            f"{float(row['trase_style_miou']):.4f} | "
            f"{float(row['trase_style_macc']):.4f} | "
            f"{row['best_k']} | {row['matched_frames']} |"
        )
    out_md.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", default="output/final_hypernerf_1/maskbenchmark_full10_res003.csv")
    parser.add_argument("--config", default="output/final_hypernerf_1/targeted_longrange_default_config.csv")
    parser.add_argument("--data-root", default="data/HyperNeRF")
    parser.add_argument("--out-csv", default="output/final_hypernerf_1/maskbenchmark_full10_res003_default_trase_style.csv")
    parser.add_argument("--out-md", default="output/final_hypernerf_1/maskbenchmark_full10_res003_default_trase_style.md")
    args = parser.parse_args()

    source_rows = load_rows(Path(args.input_csv), "scene")
    config_rows = load_rows(Path(args.config), "scene")
    rows: list[dict[str, object]] = []

    for scene in SCENE_ORDER:
        source = source_rows[scene]
        config = config_rows[scene]
        run_dir = Path(source["run_dir"])
        tag = config["tag"].strip()
        variant = tag if tag else "baseline"
        pred_root = run_dir / tag if tag else run_dir
        pred_dir = pred_root / "cluster_ids_train"
        gt_dir = Path(args.data_root) / SCENE_SPLITS[scene] / scene / "gt_masks"
        result = evaluate_scene(pred_dir, gt_dir)
        row = {
            "scene": scene,
            "variant": variant,
            "pred_dir": str(pred_dir),
            "gt_dir": str(gt_dir),
            **result,
        }
        rows.append(row)
        print(
            f"{scene}: {variant} "
            f"mIoU={float(row['trase_style_miou']):.4f} "
            f"mAcc={float(row['trase_style_macc']):.4f} "
            f"k={row['best_k']}"
        )

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "scene",
        "variant",
        "trase_style_miou",
        "trase_style_macc",
        "best_k",
        "matched_frames",
        "K_seen",
        "alignment",
        "pred_dir",
        "gt_dir",
    ]
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
