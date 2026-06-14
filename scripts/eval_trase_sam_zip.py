#!/usr/bin/env python3
"""Evaluate rendered cluster-ID maps against sam_masks_hypernerf.zip objects.

This is intentionally independent of Mask-Benchmark layout. It reads each
object-specific `masks_<object>.npz` in the zip and scores a prediction
directory containing uint8 cluster-ID PNGs.
"""

from __future__ import annotations

import argparse
import csv
import glob
import io
import re
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image


SCENES = (
    "americano",
    "chickchicken",
    "espresso",
    "keyboard",
    "split-cookie",
    "torchocolate",
)


def pred_dir_for(root: Path, scene: str, pattern: str) -> Path:
    rendered = pattern.format(scene=scene)
    full = root / rendered
    if any(ch in rendered for ch in "*?["):
        matches = sorted(Path(path) for path in glob.glob(str(full)))
        if not matches:
            raise FileNotFoundError(f"No prediction directories match: {full}")
        return matches[-1]
    return Path(str(full))


def index_pred(pred_dir: Path):
    paths = {}
    k_max = 0
    for path in sorted(pred_dir.glob("*.png")):
        try:
            fid = int(path.stem)
        except ValueError:
            continue
        paths[fid] = path
        arr = np.asarray(Image.open(path))
        if arr.ndim == 3:
            arr = arr[..., 0]
        k_max = max(k_max, int(arr.max()))
    if not paths:
        raise FileNotFoundError(f"No cluster-id PNGs found: {pred_dir}")
    return paths, k_max


def greedy_from_stats(inter, pred_count, gt_count, k_max):
    selected = []
    cur_i = 0
    cur_p = 0
    cur = 0.0
    remaining = set(range(1, k_max + 1))
    while remaining:
        best = None
        for k in remaining:
            ni = cur_i + int(inter[k])
            npred = cur_p + int(pred_count[k])
            union = npred + gt_count - ni
            val = ni / union if union > 0 else 0.0
            if best is None or val > best[0]:
                best = (val, k, ni, npred)
        if best is None or best[0] <= cur + 1e-12:
            break
        cur, k, cur_i, cur_p = best
        selected.append(k)
        remaining.remove(k)
    return cur, selected


def eval_npz(zip_file, npz_name: str, pred_paths: dict[int, Path], k_max: int):
    npz = np.load(io.BytesIO(zip_file.read(npz_name)), allow_pickle=False)
    inter = np.zeros(k_max + 1, dtype=np.int64)
    pred_count = np.zeros(k_max + 1, dtype=np.int64)
    gt_count = 0
    matched = 0

    for fid, pred_path in pred_paths.items():
        key = f"frame_{fid - 1:06d}" if f"frame_{fid - 1:06d}" in npz.files else f"frame_{fid:06d}"
        if key not in npz.files:
            continue

        pred = np.asarray(Image.open(pred_path))
        if pred.ndim == 3:
            pred = pred[..., 0]
        pred = pred.astype(np.int32)

        gt = npz[key]
        if gt.ndim == 3:
            gt = np.any(gt, axis=0)
        if gt.shape != pred.shape:
            continue
        gt = gt.astype(bool)

        pc = np.bincount(pred.ravel(), minlength=k_max + 1)
        pred_count[: len(pc)] += pc
        ic = np.bincount(pred[gt].ravel(), minlength=k_max + 1)
        inter[: len(ic)] += ic
        gt_count += int(gt.sum())
        matched += 1

    union = pred_count + gt_count - inter
    ious = np.zeros(k_max + 1, dtype=np.float64)
    valid = union > 0
    ious[valid] = inter[valid] / union[valid]
    best_k = int(ious[1:].argmax() + 1) if k_max else 0
    best = float(ious[best_k]) if best_k else 0.0
    greedy, selected = greedy_from_stats(inter, pred_count, gt_count, k_max)
    return {
        "matched": matched,
        "gt_pixels": int(gt_count),
        "best_miou": best,
        "best_k": best_k,
        "greedy_miou": float(greedy),
        "selected": " ".join(str(k) for k in selected),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zip", default="/workspace/sam_masks_hypernerf.zip")
    parser.add_argument("--pred_root", default=".")
    parser.add_argument(
        "--pred_pattern",
        required=True,
        help="Format string relative to pred_root, e.g. "
        "'output/{scene}_trase_sam/trase_features_k30/cluster_ids_train'",
    )
    parser.add_argument("--out_csv", required=True)
    parser.add_argument("--scenes", nargs="*", default=list(SCENES))
    args = parser.parse_args()

    pred_root = Path(args.pred_root)
    rows = []
    with zipfile.ZipFile(args.zip) as zip_file:
        members = defaultdict(list)
        for name in zip_file.namelist():
            match = re.match(r"sam_masks_hypernerf/([^/]+)/masks_(.+)\.npz$", name)
            if match:
                members[match.group(1)].append((match.group(2), name))

        for scene in args.scenes:
            pred_dir = pred_dir_for(pred_root, scene, args.pred_pattern)
            pred_paths, k_max = index_pred(pred_dir)
            for obj, member in sorted(members[scene]):
                result = eval_npz(zip_file, member, pred_paths, k_max)
                result.update(scene=scene, object=obj, K=k_max, pred_dir=str(pred_dir))
                rows.append(result)

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    cols = [
        "scene",
        "object",
        "K",
        "matched",
        "gt_pixels",
        "best_miou",
        "best_k",
        "greedy_miou",
        "selected",
        "pred_dir",
    ]
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        writer.writerows(rows)

    by_scene = defaultdict(list)
    for row in rows:
        by_scene[row["scene"]].append(row)
    top_best = [max(vals, key=lambda r: r["best_miou"]) for vals in by_scene.values()]
    top_greedy = [max(vals, key=lambda r: r["greedy_miou"]) for vals in by_scene.values()]

    print("Per-scene top best-cluster:")
    for row in top_best:
        print(f"  {row['scene']}: {row['best_miou']:.4f} ({row['object']}, k={row['best_k']})")
    print(f"Mean top best-cluster: {np.mean([r['best_miou'] for r in top_best]):.4f}")
    print(f"Mean top greedy-union:  {np.mean([r['greedy_miou'] for r in top_greedy]):.4f}")
    print(f"Wrote {out_csv}")


if __name__ == "__main__":
    main()
