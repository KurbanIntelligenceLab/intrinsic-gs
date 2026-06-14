#!/usr/bin/env python3
"""Evaluate cluster-ID maps against SAM3 mask directories.

The SAM3 exports have two useful structures:

* aggregate_union: scene/masks.npz, all deduplicated proposals per frame.
* per_prompt_union: scene/masks_<prompt>.npz, proposals unioned per prompt.

Both are scored with the same best-cluster and greedy-union metrics used by
scripts/eval_trase_sam_zip.py.
"""

from __future__ import annotations

import argparse
import csv
import glob
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image


SCENES = ("slice-banana", "cut-lemon1", "hand1-dense-v2", "oven-mitts")


def resolve_pred_dir(root: Path, scene: str, pattern: str) -> Path:
    rendered = pattern.format(scene=scene)
    full = root / rendered
    if any(ch in rendered for ch in "*?["):
        matches = sorted(Path(p) for p in glob.glob(str(full)))
        if not matches:
            raise FileNotFoundError(f"No prediction directories match: {full}")
        return matches[-1]
    return full


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


def frame_key(npz, fid: int) -> str | None:
    candidates = (f"frame_{fid - 1:06d}", f"frame_{fid:06d}", f"frame_{fid + 1:06d}")
    for key in candidates:
        if key in npz.files:
            return key
    return None


def normalize_gt(gt: np.ndarray, pred_shape: tuple[int, int]) -> tuple[np.ndarray | None, str]:
    if gt.ndim == 3:
        gt = np.any(gt, axis=0)
    if gt.shape == pred_shape:
        return gt.astype(bool), "matched"
    if gt.ndim == 2 and gt.T.shape == pred_shape:
        return gt.T.astype(bool), "transposed"
    return None, f"shape_mismatch:{gt.shape}->{pred_shape}"


def eval_npz(npz_path: Path, pred_paths: dict[int, Path], k_max: int):
    npz = np.load(npz_path, allow_pickle=False)
    inter = np.zeros(k_max + 1, dtype=np.int64)
    pred_count = np.zeros(k_max + 1, dtype=np.int64)
    gt_count = 0
    matched = 0
    transposed = 0
    skipped_shape = 0
    skipped_key = 0

    for fid, pred_path in pred_paths.items():
        key = frame_key(npz, fid)
        if key is None:
            skipped_key += 1
            continue

        pred = np.asarray(Image.open(pred_path))
        if pred.ndim == 3:
            pred = pred[..., 0]
        pred = pred.astype(np.int32)

        gt, status = normalize_gt(npz[key], pred.shape)
        if gt is None:
            skipped_shape += 1
            continue
        if status == "transposed":
            transposed += 1

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
        "transposed": transposed,
        "skipped_shape": skipped_shape,
        "skipped_key": skipped_key,
        "gt_pixels": int(gt_count),
        "best_miou": best,
        "best_k": best_k,
        "greedy_miou": float(greedy),
        "selected": " ".join(str(k) for k in selected),
    }


def scene_targets(scene_dir: Path, structures: list[str]):
    targets = []
    if "aggregate_union" in structures:
        aggregate = scene_dir / "masks.npz"
        if aggregate.exists():
            targets.append(("aggregate_union", "sam3_union", aggregate))
    if "per_prompt_union" in structures:
        for path in sorted(scene_dir.glob("masks_*.npz")):
            target = path.stem.removeprefix("masks_")
            targets.append(("per_prompt_union", target, path))
    return targets


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mask-root", default="data/sam3_hypernerf_missing")
    parser.add_argument("--pred_root", default=".")
    parser.add_argument(
        "--pred_pattern",
        required=True,
        help="Format string relative to pred_root. Glob wildcards are allowed.",
    )
    parser.add_argument("--out_csv", required=True)
    parser.add_argument("--scenes", nargs="*", default=list(SCENES))
    parser.add_argument(
        "--structures",
        nargs="+",
        choices=("aggregate_union", "per_prompt_union"),
        default=["aggregate_union", "per_prompt_union"],
    )
    args = parser.parse_args()

    mask_root = Path(args.mask_root)
    pred_root = Path(args.pred_root)
    rows = []

    for scene in args.scenes:
        scene_dir = mask_root / scene
        pred_dir = resolve_pred_dir(pred_root, scene, args.pred_pattern)
        pred_paths, k_max = index_pred(pred_dir)
        for structure, target, npz_path in scene_targets(scene_dir, args.structures):
            result = eval_npz(npz_path, pred_paths, k_max)
            result.update(
                scene=scene,
                structure=structure,
                target=target,
                K=k_max,
                pred_dir=str(pred_dir),
                mask_npz=str(npz_path),
            )
            rows.append(result)

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    cols = [
        "scene",
        "structure",
        "target",
        "K",
        "matched",
        "transposed",
        "skipped_shape",
        "skipped_key",
        "gt_pixels",
        "best_miou",
        "best_k",
        "greedy_miou",
        "selected",
        "pred_dir",
        "mask_npz",
    ]
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {out_csv}")
    for structure in args.structures:
        by_scene = defaultdict(list)
        for row in rows:
            if row["structure"] == structure:
                by_scene[row["scene"]].append(row)
        if not by_scene:
            continue
        top_best = [max(vals, key=lambda r: r["best_miou"]) for vals in by_scene.values()]
        top_greedy = [max(vals, key=lambda r: r["greedy_miou"]) for vals in by_scene.values()]
        print(f"{structure}:")
        for row in top_best:
            print(
                f"  {row['scene']}: {row['best_miou']:.4f} "
                f"({row['target']}, k={row['best_k']}, matched={row['matched']})"
            )
        print(f"  Mean top best-cluster: {np.mean([r['best_miou'] for r in top_best]):.4f}")
        print(f"  Mean top greedy-union:  {np.mean([r['greedy_miou'] for r in top_greedy]):.4f}")


if __name__ == "__main__":
    main()
