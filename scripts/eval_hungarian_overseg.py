#!/usr/bin/env python3
"""Score saved cluster-ID renders with Hungarian matching and over-seg stats.

The existing Mask-Benchmark scripts report the best single cluster for a
single foreground object. This script keeps that behavior for flat GT masks,
and also supports nested multi-object GT folders by solving a one-to-one
Hungarian assignment between predicted clusters and GT objects.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from self_supervised_scripts.compute_miou import (  # noqa: E402
    align_single_object_gt_paths,
    index_pngs,
    load_binary_mask,
    load_cluster_id_map,
    load_gt_layout,
)

try:
    from scipy.optimize import linear_sum_assignment
except Exception:  # pragma: no cover - fallback keeps tiny cases usable.
    linear_sum_assignment = None


HYPERNERF_SPLITS = {
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


def infer_dataset(scene: str) -> str:
    if scene in HYPERNERF_SPLITS:
        return "hypernerf"
    return "neu3d"


def default_gt_dir(dataset: str, scene: str) -> Path:
    if dataset == "hypernerf":
        return Path("data") / "HyperNeRF" / HYPERNERF_SPLITS[scene] / scene / "gt_masks"
    if dataset == "neu3d":
        return Path("data") / "Mask-Benchmark" / "Mask-Benchmark" / "Neu3D-Mask" / scene / "gt_masks"
    raise ValueError(f"Unknown dataset: {dataset}")


def resolve_pred_dir(row: dict[str, str], pred_subdir: str = "") -> Path:
    if row.get("pred_dir"):
        pred = Path(row["pred_dir"])
        if pred.name == "cluster_ids_train" or pred.name == "cluster_ids_test":
            return pred
    run_dir = Path(row["run_dir"])
    root = run_dir / pred_subdir if pred_subdir else run_dir
    for leaf in ("cluster_ids_train", "cluster_ids_test"):
        candidate = root / leaf
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"No cluster_ids_train/test under {root}")


def count_clusters(pred_paths: dict[int, str]) -> tuple[int, int, list[int]]:
    seen: set[int] = set()
    max_id = 0
    for path in pred_paths.values():
        pred = load_cluster_id_map(path)
        ids = np.unique(pred)
        seen.update(int(x) for x in ids if int(x) > 0)
        max_id = max(max_id, int(ids.max()) if ids.size else 0)
    return len(seen), max_id, sorted(seen)


def hungarian(cost: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if linear_sum_assignment is not None:
        return linear_sum_assignment(cost)
    # Very small fallback for environments without scipy. The common flat-mask
    # case has one GT object, so this path is enough for that too.
    if cost.shape[0] == 1:
        return np.array([0]), np.array([int(np.argmin(cost[0]))])
    raise RuntimeError("scipy is required for multi-object Hungarian matching")


def aligned_gt_objects(
    pred_paths: dict[int, str],
    gt_dir: Path,
) -> tuple[dict[str, dict[int, str]], dict[str, object]]:
    kind, payload = load_gt_layout(str(gt_dir))
    if kind == "single":
        gt_paths, alignment = align_single_object_gt_paths(pred_paths, payload, str(gt_dir))
        return {"foreground": gt_paths}, alignment
    return payload, {"method": "nested_direct", "gt_objects": len(payload)}


def evaluate(pred_dir: Path, gt_dir: Path) -> dict[str, object]:
    pred_paths = index_pngs(str(pred_dir))
    if not pred_paths:
        raise RuntimeError(f"No prediction PNGs in {pred_dir}")

    gt_objects, alignment = aligned_gt_objects(pred_paths, gt_dir)
    object_names = sorted(gt_objects)
    if not object_names:
        raise RuntimeError(f"No GT masks in {gt_dir}")

    cluster_count, max_cluster_id, cluster_ids = count_clusters(pred_paths)
    if not cluster_ids:
        cluster_ids = []
    cluster_to_col = {cluster_id: idx for idx, cluster_id in enumerate(cluster_ids)}

    num_objects = len(object_names)
    num_clusters = len(cluster_ids)
    inter = np.zeros((num_objects, num_clusters), dtype=np.float64)
    gt_count = np.zeros(num_objects, dtype=np.float64)
    pred_count = np.zeros(num_clusters, dtype=np.float64)
    total_pixels = np.zeros(num_objects, dtype=np.float64)
    valid_frames_by_object = np.zeros(num_objects, dtype=np.int32)

    # Prediction counts are object-independent, so accumulate once per frame.
    valid_pred_frames: set[int] = set()
    for obj_idx, obj_name in enumerate(object_names):
        gt_paths = gt_objects[obj_name]
        matched = sorted(set(pred_paths) & set(gt_paths))
        for fid in matched:
            pred = load_cluster_id_map(pred_paths[fid])
            gt = load_binary_mask(gt_paths[fid])
            if pred.shape != gt.shape:
                continue
            if fid not in valid_pred_frames:
                ids, counts = np.unique(pred[pred > 0], return_counts=True)
                for cluster_id, count in zip(ids, counts):
                    col = cluster_to_col.get(int(cluster_id))
                    if col is not None:
                        pred_count[col] += int(count)
                valid_pred_frames.add(fid)
            total_pixels[obj_idx] += gt.size
            gt_count[obj_idx] += int(gt.sum())
            valid_frames_by_object[obj_idx] += 1
            if num_clusters:
                ids, counts = np.unique(pred[gt], return_counts=True)
                for cluster_id, count in zip(ids, counts):
                    if int(cluster_id) <= 0:
                        continue
                    col = cluster_to_col.get(int(cluster_id))
                    if col is not None:
                        inter[obj_idx, col] += int(count)

    union = gt_count[:, None] + pred_count[None, :] - inter
    iou = np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)
    accuracy = np.divide(
        total_pixels[:, None] - pred_count[None, :] - gt_count[:, None] + (2.0 * inter),
        total_pixels[:, None],
        out=np.zeros_like(inter),
        where=total_pixels[:, None] > 0,
    )

    if num_clusters:
        gt_rows, cluster_cols = hungarian(-iou)
    else:
        gt_rows = np.array([], dtype=np.int32)
        cluster_cols = np.array([], dtype=np.int32)

    matched_iou = np.zeros(num_objects, dtype=np.float64)
    matched_acc = np.zeros(num_objects, dtype=np.float64)
    matched_cluster_ids = [None] * num_objects
    for gt_row, cluster_col in zip(gt_rows, cluster_cols):
        matched_iou[int(gt_row)] = float(iou[int(gt_row), int(cluster_col)])
        matched_acc[int(gt_row)] = float(accuracy[int(gt_row), int(cluster_col)])
        matched_cluster_ids[int(gt_row)] = cluster_ids[int(cluster_col)]

    if num_clusters == 0:
        matched_acc = np.divide(
            total_pixels - gt_count,
            total_pixels,
            out=np.zeros_like(total_pixels),
            where=total_pixels > 0,
        )

    gt_object_count = num_objects
    overseg_ratio = float(cluster_count / gt_object_count) if gt_object_count else 0.0
    excess_clusters_per_gt = (
        float(max(0, cluster_count - gt_object_count) / gt_object_count)
        if gt_object_count else 0.0
    )

    return {
        "hungarian_miou": float(np.mean(matched_iou)) if num_objects else 0.0,
        "hungarian_macc": float(np.mean(matched_acc)) if num_objects else 0.0,
        "gt_object_count": gt_object_count,
        "matched_cluster_count": int(sum(x is not None for x in matched_cluster_ids)),
        "cluster_count": int(cluster_count),
        "max_cluster_id": int(max_cluster_id),
        "overseg_ratio": overseg_ratio,
        "excess_clusters_per_gt": excess_clusters_per_gt,
        "valid_pred_frames": int(len(valid_pred_frames)),
        "valid_gt_frames_min": int(valid_frames_by_object.min()) if len(valid_frames_by_object) else 0,
        "valid_gt_frames_max": int(valid_frames_by_object.max()) if len(valid_frames_by_object) else 0,
        "alignment": alignment.get("method", "unknown"),
        "object_matches": [
            {
                "object": object_names[idx],
                "cluster_id": matched_cluster_ids[idx],
                "iou": float(matched_iou[idx]),
                "macc": float(matched_acc[idx]),
                "valid_frames": int(valid_frames_by_object[idx]),
            }
            for idx in range(num_objects)
        ],
    }


def load_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def rows_from_component_csv(path: Path, dataset: str) -> list[dict[str, str]]:
    rows = []
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            scene = row["scene"]
            rows.append(
                {
                    "dataset": dataset,
                    "variant": row.get("variant", "default"),
                    "scene": scene,
                    "run_dir": row["run_dir"],
                    "gt_dir": str(default_gt_dir(dataset, scene)),
                }
            )
    return rows


def rows_from_hypernerf_global() -> list[dict[str, str]]:
    source = Path("output/final_hypernerf_1/maskbenchmark_full10_res003_global_longrange_input.csv")
    if not source.exists():
        source = Path("output/final_hypernerf_1/maskbenchmark_full10_res003_default_trase_style.csv")
    rows = []
    with source.open(newline="") as f:
        for row in csv.DictReader(f):
            scene = row["scene"]
            if row.get("run_dir"):
                run_dir = Path(row["run_dir"])
            else:
                baseline_pred = Path(row["pred_dir"])
                run_dir = baseline_pred.parent
                if baseline_pred.name in {"cluster_ids_train", "cluster_ids_test"}:
                    run_dir = baseline_pred.parent
                if (run_dir / "objectness_longrange_default" / "cluster_ids_train").exists():
                    pass
                elif (run_dir.parent / "objectness_longrange_default" / "cluster_ids_train").exists():
                    run_dir = run_dir.parent
            rows.append(
                {
                    "dataset": "hypernerf",
                    "variant": "single_global_objectness_longrange",
                    "scene": scene,
                    "run_dir": str(run_dir),
                    "pred_subdir": "objectness_longrange_default",
                    "gt_dir": row.get("gt_dir") or str(default_gt_dir("hypernerf", scene)),
                }
            )
    return rows


def build_default_manifest() -> list[dict[str, str]]:
    rows = []
    neu3d = Path("output/final_neu3d_leiden018_component_ablations/component_ablation_per_scene.csv")
    hyper = Path("output/final_hypernerf_component_ablation_logs/component_ablation_per_scene.csv")
    if neu3d.exists():
        rows.extend(rows_from_component_csv(neu3d, "neu3d"))
    if hyper.exists():
        rows.extend(rows_from_component_csv(hyper, "hypernerf"))
    if Path("output/final_hypernerf_1/maskbenchmark_full10_res003_default_trase_style.csv").exists():
        rows.extend(rows_from_hypernerf_global())
    return rows


def summarize(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    groups: dict[tuple[str, str], list[dict[str, object]]] = {}
    for row in rows:
        groups.setdefault((str(row["dataset"]), str(row["variant"])), []).append(row)
    summaries = []
    for (dataset, variant), group in sorted(groups.items()):
        summaries.append(
            {
                "dataset": dataset,
                "variant": variant,
                "scenes": len(group),
                "hungarian_miou_mean": float(np.mean([float(r["hungarian_miou"]) for r in group])),
                "hungarian_macc_mean": float(np.mean([float(r["hungarian_macc"]) for r in group])),
                "cluster_count_mean": float(np.mean([float(r["cluster_count"]) for r in group])),
                "overseg_ratio_mean": float(np.mean([float(r["overseg_ratio"]) for r in group])),
                "excess_clusters_per_gt_mean": float(np.mean([float(r["excess_clusters_per_gt"]) for r in group])),
            }
        )
    return summaries


def write_csv(path: Path, rows: list[dict[str, object]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, help="Optional CSV with dataset,scene,variant,run_dir/pred_dir,gt_dir.")
    parser.add_argument("--out-dir", type=Path, default=Path("output/non_oracle_metrics"))
    args = parser.parse_args()

    manifest = load_manifest(args.manifest) if args.manifest else build_default_manifest()
    if not manifest:
        raise RuntimeError("No runs found to evaluate")

    rows: list[dict[str, object]] = []
    details: list[dict[str, object]] = []
    for source in manifest:
        scene = source["scene"]
        dataset = source.get("dataset") or infer_dataset(scene)
        variant = source.get("variant") or "default"
        gt_dir = Path(source.get("gt_dir") or default_gt_dir(dataset, scene))
        pred_dir = resolve_pred_dir(source, source.get("pred_subdir", ""))
        result = evaluate(pred_dir, gt_dir)
        row = {
            "dataset": dataset,
            "variant": variant,
            "scene": scene,
            "hungarian_miou": result["hungarian_miou"],
            "hungarian_macc": result["hungarian_macc"],
            "gt_object_count": result["gt_object_count"],
            "matched_cluster_count": result["matched_cluster_count"],
            "cluster_count": result["cluster_count"],
            "max_cluster_id": result["max_cluster_id"],
            "overseg_ratio": result["overseg_ratio"],
            "excess_clusters_per_gt": result["excess_clusters_per_gt"],
            "valid_pred_frames": result["valid_pred_frames"],
            "valid_gt_frames_min": result["valid_gt_frames_min"],
            "valid_gt_frames_max": result["valid_gt_frames_max"],
            "alignment": result["alignment"],
            "pred_dir": str(pred_dir),
            "gt_dir": str(gt_dir),
        }
        rows.append(row)
        details.append({**row, "object_matches": result["object_matches"]})
        print(
            f"{dataset}/{variant}/{scene}: "
            f"H-mIoU={float(row['hungarian_miou']):.4f} "
            f"H-mAcc={float(row['hungarian_macc']):.4f} "
            f"K={row['cluster_count']} overseg={float(row['overseg_ratio']):.1f}x"
        )

    per_scene_cols = [
        "dataset",
        "variant",
        "scene",
        "hungarian_miou",
        "hungarian_macc",
        "gt_object_count",
        "matched_cluster_count",
        "cluster_count",
        "max_cluster_id",
        "overseg_ratio",
        "excess_clusters_per_gt",
        "valid_pred_frames",
        "valid_gt_frames_min",
        "valid_gt_frames_max",
        "alignment",
        "pred_dir",
        "gt_dir",
    ]
    summary_cols = [
        "dataset",
        "variant",
        "scenes",
        "hungarian_miou_mean",
        "hungarian_macc_mean",
        "cluster_count_mean",
        "overseg_ratio_mean",
        "excess_clusters_per_gt_mean",
    ]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.out_dir / "hungarian_overseg_per_scene.csv", rows, per_scene_cols)
    write_csv(args.out_dir / "hungarian_overseg_summary.csv", summarize(rows), summary_cols)
    (args.out_dir / "hungarian_overseg_details.json").write_text(json.dumps(details, indent=2))
    print(f"Wrote {args.out_dir / 'hungarian_overseg_per_scene.csv'}")
    print(f"Wrote {args.out_dir / 'hungarian_overseg_summary.csv'}")


if __name__ == "__main__":
    main()
