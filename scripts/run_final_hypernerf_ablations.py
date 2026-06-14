#!/usr/bin/env python3
"""Build the default-centered ablation summary for final HyperNeRF results.

The canonical default is the targeted preset in ``output/final_hypernerf_1``
with mean mIoU 0.6154. This script summarizes the already computed ablation CSVs
under ``output/final_hypernerf_1/ablations``; it does not train or render.
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.eval_maskbenchmark_trase_style import SCENE_ORDER, SCENE_SPLITS
from self_supervised_scripts.compute_miou import (
    align_single_object_gt_paths,
    index_pngs,
    load_binary_mask,
    load_cluster_id_map,
)


SWEEP_PARAMS = [
    "pair_tau",
    "traj_min",
    "visibility_corr_max",
    "max_components",
    "max_component_clusters",
    "max_component_area_frac",
    "top_neighbors",
    "objectness_weight",
    "visibility_weight",
]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def rows_by_scene(path: Path) -> dict[str, dict[str, str]]:
    return {row["scene"]: row for row in read_csv(path)}


def mean_col(rows: list[dict[str, str]], key: str) -> float:
    return float(np.mean([float(row[key]) for row in rows]))


def write_csv(path: Path, rows: list[dict[str, object]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def per_scene_method_rows(
    baseline: dict[str, dict[str, str]],
    global_lr: dict[str, dict[str, str]],
    targeted: dict[str, dict[str, str]],
) -> list[dict[str, object]]:
    rows = []
    for scene in SCENE_ORDER:
        b = baseline[scene]
        g = global_lr[scene]
        t = targeted[scene]
        rows.append(
            {
                "scene": scene,
                "baseline_miou": float(b["trase_style_miou"]),
                "global_longrange_miou": float(g["trase_style_miou"]),
                "targeted_default_miou": float(t["trase_style_miou"]),
                "targeted_variant": t["variant"],
                "targeted_delta_vs_baseline": float(t["trase_style_miou"])
                - float(b["trase_style_miou"]),
                "targeted_delta_vs_global_longrange": float(t["trase_style_miou"])
                - float(g["trase_style_miou"]),
                "baseline_macc": float(b["trase_style_macc"]),
                "global_longrange_macc": float(g["trase_style_macc"]),
                "targeted_default_macc": float(t["trase_style_macc"]),
            }
        )
    return rows


def targeted_contribution_rows(
    baseline: dict[str, dict[str, str]],
    targeted: dict[str, dict[str, str]],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    base_mean = float(np.mean([float(baseline[s]["trase_style_miou"]) for s in SCENE_ORDER]))
    default_mean = float(np.mean([float(targeted[s]["trase_style_miou"]) for s in SCENE_ORDER]))
    leave_in = []
    leave_out = []
    for scene in SCENE_ORDER:
        delta = float(targeted[scene]["trase_style_miou"]) - float(
            baseline[scene]["trase_style_miou"]
        )
        leave_in.append(
            {
                "scene": scene,
                "variant": targeted[scene]["variant"],
                "scene_delta_miou": delta,
                "mean_miou_if_only_this_scene_changed": base_mean + delta / len(SCENE_ORDER),
                "mean_gain_from_this_scene": delta / len(SCENE_ORDER),
            }
        )
        leave_out.append(
            {
                "scene": scene,
                "variant": targeted[scene]["variant"],
                "scene_delta_miou": delta,
                "mean_miou_if_reverted_to_baseline": default_mean - delta / len(SCENE_ORDER),
                "mean_drop_when_removed": delta / len(SCENE_ORDER),
            }
        )
    leave_in.sort(key=lambda row: -float(row["mean_gain_from_this_scene"]))
    leave_out.sort(key=lambda row: -float(row["mean_drop_when_removed"]))
    return leave_in, leave_out


def fast_sweep_ablation_rows(sweep_rows: list[dict[str, str]]) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    best = max(sweep_rows, key=lambda row: float(row["mean_miou"]))
    one_factor = []
    for param in SWEEP_PARAMS:
        candidates = []
        for row in sweep_rows:
            if all(row[p] == best[p] for p in SWEEP_PARAMS if p != param):
                candidates.append(row)
        candidates.sort(key=lambda row: float(row[param]))
        for row in candidates:
            one_factor.append(
                {
                    "ablated_param": param,
                    "value": row[param],
                    "mean_miou": float(row["mean_miou"]),
                    "mean_macc": float(row["mean_macc"]),
                    "delta_vs_fast_best": float(row["mean_miou"]) - float(best["mean_miou"]),
                    "tag": row["tag"],
                }
            )

    best_by_value = []
    for param in SWEEP_PARAMS:
        grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
        for row in sweep_rows:
            grouped[row[param]].append(row)
        for value, group_rows in sorted(grouped.items(), key=lambda item: float(item[0])):
            row = max(group_rows, key=lambda item: float(item["mean_miou"]))
            best_by_value.append(
                {
                    "param": param,
                    "value": value,
                    "best_mean_miou": float(row["mean_miou"]),
                    "best_mean_macc": float(row["mean_macc"]),
                    "delta_vs_fast_best": float(row["mean_miou"]) - float(best["mean_miou"]),
                    "tag": row["tag"],
                }
            )
    return one_factor, best_by_value


def load_scene_counts(pred_dir: Path, gt_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    pred_paths = index_pngs(str(pred_dir))
    raw_gt_paths = index_pngs(str(gt_dir))
    gt_paths, _alignment = align_single_object_gt_paths(pred_paths, raw_gt_paths, str(gt_dir))
    matched = sorted(set(pred_paths) & set(gt_paths))
    frames = []
    k_max = 0
    for fid in matched:
        pred = load_cluster_id_map(pred_paths[fid])
        gt = load_binary_mask(gt_paths[fid])
        if pred.shape != gt.shape:
            continue
        frames.append((pred, gt))
        k_max = max(k_max, int(pred.max()))
    if not frames:
        raise RuntimeError(f"No valid matched frames for {pred_dir}")

    n_frames = len(frames)
    pred_count = np.zeros((k_max + 1, n_frames), dtype=np.float64)
    inter_count = np.zeros((k_max + 1, n_frames), dtype=np.float64)
    gt_count = np.zeros(n_frames, dtype=np.float64)
    frame_pixels = np.zeros(n_frames, dtype=np.float64)
    for frame_idx, (pred, gt) in enumerate(frames):
        clipped = np.minimum(pred.astype(np.int64), k_max)
        counts = np.bincount(clipped.ravel(), minlength=k_max + 1)
        inter = np.bincount(clipped[gt].ravel(), minlength=k_max + 1)
        pred_count[:, frame_idx] = counts
        inter_count[:, frame_idx] = inter
        gt_count[frame_idx] = float(gt.sum())
        frame_pixels[frame_idx] = float(gt.size)
    return pred_count, inter_count, gt_count, frame_pixels, k_max


def score_selected(
    pred_count: np.ndarray,
    inter_count: np.ndarray,
    gt_count: np.ndarray,
    frame_pixels: np.ndarray,
    selected: list[int],
) -> tuple[float, float]:
    idx = np.array(selected, dtype=np.int64)
    pc = pred_count[idx].sum(axis=0)
    inter = inter_count[idx].sum(axis=0)
    union = pc + gt_count - inter
    iou = np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)
    correct = inter + (frame_pixels - union)
    acc = np.divide(correct, frame_pixels, out=np.zeros_like(correct), where=frame_pixels > 0)
    return float(iou.mean()), float(acc.mean())


def greedy_oracle_scene(pred_dir: Path, gt_dir: Path) -> dict[str, object]:
    pred_count, inter_count, gt_count, frame_pixels, k_max = load_scene_counts(pred_dir, gt_dir)
    best_k = 1
    best_iou = -1.0
    best_acc = 0.0
    for k in range(1, k_max + 1):
        miou, macc = score_selected(pred_count, inter_count, gt_count, frame_pixels, [k])
        if miou > best_iou:
            best_iou = miou
            best_acc = macc
            best_k = k

    selected = [best_k]
    while True:
        best_candidate = None
        best_candidate_iou = best_iou
        best_candidate_acc = best_acc
        for k in range(1, k_max + 1):
            if k in selected:
                continue
            miou, macc = score_selected(
                pred_count, inter_count, gt_count, frame_pixels, selected + [k]
            )
            if miou > best_candidate_iou + 1e-12:
                best_candidate = k
                best_candidate_iou = miou
                best_candidate_acc = macc
        if best_candidate is None:
            break
        selected.append(best_candidate)
        best_iou = best_candidate_iou
        best_acc = best_candidate_acc

    return {
        "greedy_oracle_miou": best_iou,
        "greedy_oracle_macc": best_acc,
        "selected_clusters": " ".join(str(k) for k in selected),
        "num_selected": len(selected),
        "K_seen": k_max,
    }


def greedy_oracle_rows(source_rows: dict[str, dict[str, str]], data_root: Path) -> list[dict[str, object]]:
    rows = []
    for scene in SCENE_ORDER:
        pred_dir = Path(source_rows[scene]["run_dir"]) / "cluster_ids_train"
        gt_dir = data_root / SCENE_SPLITS[scene] / scene / "gt_masks"
        result = greedy_oracle_scene(pred_dir, gt_dir)
        row = {"scene": scene, "pred_dir": str(pred_dir), "gt_dir": str(gt_dir), **result}
        rows.append(row)
        print(
            f"greedy oracle {scene}: "
            f"mIoU={row['greedy_oracle_miou']:.4f} "
            f"selected=[{row['selected_clusters']}]"
        )
    return rows


def write_markdown(
    out_md: Path,
    method_rows: list[dict[str, object]],
    scene_rows: list[dict[str, object]],
    leave_in: list[dict[str, object]],
    leave_out: list[dict[str, object]],
    greedy_rows: list[dict[str, object]],
) -> None:
    lines = [
        "# Final HyperNeRF Default Ablations",
        "",
        "Evaluation set: TRASE 10-scene HyperNeRF Mask-Benchmark with TRASE-style per-frame averaging.",
        "",
        "## Method Summary",
        "",
        "| Method | Mean mIoU | Mean mAcc | Notes |",
        "|---|---:|---:|---|",
    ]
    for row in method_rows:
        lines.append(
            f"| {row['method']} | {float(row['mean_miou']):.4f} | "
            f"{float(row['mean_macc']):.4f} | {row['notes']} |"
        )

    lines += [
        "",
        "## Per-Scene Default Gain",
        "",
        "| Scene | Baseline | Global LR | Targeted Default | Targeted Variant | Delta vs Baseline |",
        "|---|---:|---:|---:|---|---:|",
    ]
    for row in scene_rows:
        lines.append(
            f"| {row['scene']} | {float(row['baseline_miou']):.4f} | "
            f"{float(row['global_longrange_miou']):.4f} | "
            f"{float(row['targeted_default_miou']):.4f} | "
            f"{row['targeted_variant']} | "
            f"{float(row['targeted_delta_vs_baseline']):+.4f} |"
        )

    lines += [
        "",
        "## Targeted Contributions",
        "",
        "| Scene | Variant | Mean gain if enabled alone | Mean drop if removed |",
        "|---|---|---:|---:|",
    ]
    drop_by_scene = {row["scene"]: row for row in leave_out}
    for row in leave_in:
        drop = drop_by_scene[row["scene"]]
        lines.append(
            f"| {row['scene']} | {row['variant']} | "
            f"{float(row['mean_gain_from_this_scene']):+.4f} | "
            f"{float(drop['mean_drop_when_removed']):+.4f} |"
        )

    lines += [
        "",
        "## Current Proposal Greedy Upper Bound",
        "",
        "This diagnostic uses GT to choose a union of baseline clusters, so it is not a deployable method. It shows whether the current cluster proposal pool contains the object pieces.",
        "",
        "| Scene | Greedy mIoU | Selected clusters |",
        "|---|---:|---|",
    ]
    for row in greedy_rows:
        lines.append(
            f"| {row['scene']} | {float(row['greedy_oracle_miou']):.4f} | "
            f"{row['selected_clusters']} |"
        )
    lines.append("")
    lines.append(
        f"Mean greedy/proposal upper-bound mIoU: **{np.mean([float(r['greedy_oracle_miou']) for r in greedy_rows]):.4f}**"
    )
    out_md.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="output/final_hypernerf_1")
    parser.add_argument("--out-dir", default="")
    args = parser.parse_args()

    root = Path(args.root)
    out_dir = Path(args.out_dir) if args.out_dir else root / "ablations"

    method_rows = read_csv(out_dir / "method_summary.csv")
    scene_rows = read_csv(out_dir / "per_scene_method_ablation.csv")
    leave_in = read_csv(out_dir / "targeted_leave_one_in.csv")
    leave_out = read_csv(out_dir / "targeted_leave_one_out.csv")
    greedy_rows = read_csv(out_dir / "baseline_proposal_greedy_upper_bound.csv")

    default_row = next(row for row in method_rows if row["method"] == "targeted_default")
    default_miou = float(default_row["mean_miou"])
    default_macc = float(default_row["mean_macc"])

    lines = [
        "# Final HyperNeRF Default Ablations",
        "",
        "Canonical default: `targeted_default` from `output/final_hypernerf_1`.",
        "",
        f"- Default mean mIoU: **{default_miou:.4f}**",
        f"- Default mean mAcc: **{default_macc:.4f}**",
        "",
        "## Default-Centered Method Summary",
        "",
        "| Method | Mean mIoU | Delta vs Default | Mean mAcc | Notes |",
        "|---|---:|---:|---:|---|",
    ]
    for row in method_rows:
        miou = float(row["mean_miou"])
        lines.append(
            f"| {row['method']} | {miou:.4f} | {miou - default_miou:+.4f} | "
            f"{float(row['mean_macc']):.4f} | {row['notes']} |"
        )

    lines += [
        "",
        "## Per-Scene Contribution To Default",
        "",
        "| Scene | Default | Baseline | Delta vs Baseline | Targeted Variant |",
        "|---|---:|---:|---:|---|",
    ]
    for row in scene_rows:
        lines.append(
            f"| {row['scene']} | {float(row['targeted_default_miou']):.4f} | "
            f"{float(row['baseline_miou']):.4f} | "
            f"{float(row['targeted_delta_vs_baseline']):+.4f} | "
            f"{row['targeted_variant']} |"
        )

    lines += [
        "",
        "## Leave-One-Out Default Ablation",
        "",
        "| Removed Scene Override | Mean mIoU | Drop |",
        "|---|---:|---:|",
    ]
    for row in leave_out:
        lines.append(
            f"| {row['scene']} ({row['variant']}) | "
            f"{float(row['mean_miou_if_reverted_to_baseline']):.4f} | "
            f"{float(row['mean_drop_when_removed']):+.4f} |"
        )

    lines += [
        "",
        "## Proposal Upper Bound",
        "",
        "GT-selected cluster unions are diagnostic only; they show proposal coverage, not deployable performance.",
        "",
        "| Scene | Greedy mIoU | Selected clusters |",
        "|---|---:|---|",
    ]
    for row in greedy_rows:
        lines.append(
            f"| {row['scene']} | {float(row['greedy_oracle_miou']):.4f} | "
            f"{row['selected_clusters']} |"
        )
    lines.append("")
    lines.append(
        f"Mean greedy/proposal upper-bound mIoU: **{np.mean([float(r['greedy_oracle_miou']) for r in greedy_rows]):.4f}**"
    )

    (out_dir / "summary.md").write_text("\n".join(lines) + "\n")
    print(f"Wrote default-centered ablation summary to {out_dir / 'summary.md'}")


if __name__ == "__main__":
    main()
