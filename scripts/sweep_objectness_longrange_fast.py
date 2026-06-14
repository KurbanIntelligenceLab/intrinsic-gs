#!/usr/bin/env python3
"""Fast no-mask objectness long-range sweep on rendered cluster-ID maps.

This evaluates merge mappings in image space, without re-rendering each merged
Gaussian label set. It is meant for parameter search; promising settings should
still be rendered with `run_objectness_longrange_res003.py` before being treated
as canonical.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scene import DeformModel
from scripts.eval_maskbenchmark_trase_style import SCENE_ORDER, SCENE_SPLITS
from self_supervised_scripts.cluster_trajectories import compute_trajectories, resolve_ply_path
from self_supervised_scripts.compute_miou import (
    align_single_object_gt_paths,
    index_pngs,
    load_binary_mask,
    load_cluster_id_map,
)
from self_supervised_scripts.merge_clusters import (
    compute_cluster_aggregates,
    compute_similarity_matrix,
    load_gaussian_attrs,
)
from self_supervised_scripts.merge_clusters_objectness import (
    build_components,
    collect_image_stats,
    mapping_for_components,
    objectness_scores,
)


@dataclass(frozen=True)
class Params:
    pair_tau: float
    traj_min: float
    visibility_corr_max: float
    max_components: int
    max_component_clusters: int
    max_component_area_frac: float
    top_neighbors: int
    objectness_weight: float
    visibility_weight: float

    def tag(self) -> str:
        return (
            f"tau{self.pair_tau:g}_traj{self.traj_min:g}_vc{self.visibility_corr_max:g}_"
            f"mc{self.max_components}_mk{self.max_component_clusters}_"
            f"ma{self.max_component_area_frac:g}_tn{self.top_neighbors}_"
            f"ow{self.objectness_weight:g}_vw{self.visibility_weight:g}"
        )


class SceneCache:
    def __init__(self, scene: str, row: dict[str, str], data_root: Path, load_iteration: int) -> None:
        self.scene = scene
        self.row = row
        self.run_dir = Path(row["run_dir"])
        self.labels_file = self.run_dir / "labels.npy"
        self.cluster_ids_dir = self.run_dir / "cluster_ids_train"
        self.source_path = data_root / SCENE_SPLITS[scene] / scene
        self.model_path = Path("output") / f"{scene}_30k_gs"
        self.load_iteration = load_iteration

        self.labels_full = np.load(self.labels_file)
        valid_mask = self.labels_full > 0
        labels_valid = self.labels_full[valid_mask].astype(np.int64) - 1
        self.n_clusters = int(labels_valid.max()) + 1

        resolver_args = type(
            "ResolveArgs",
            (),
            {
                "labels": str(self.labels_file),
                "ply_path": "",
                "model_path": str(self.model_path),
                "load_iteration": load_iteration,
            },
        )()
        ply_path = resolve_ply_path(resolver_args)
        xyz, _opacity, color, feats = load_gaussian_attrs(ply_path)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        pos_valid = torch.from_numpy(xyz[valid_mask]).to(device)
        deform = DeformModel(is_blender=False, is_6dof=False)
        deform.load_weights(str(PROJECT_ROOT), iteration=load_iteration, scene_name=scene)
        traj_xyz = compute_trajectories(deform, pos_valid, 20).cpu().numpy().astype(np.float64)

        mean_traj, mean_feat, mean_rgb, _sizes, valid = compute_cluster_aggregates(
            labels_valid=labels_valid,
            traj_xyz=traj_xyz,
            feats=feats[valid_mask] if feats is not None else None,
            color=color[valid_mask],
            n_clusters=self.n_clusters,
            min_cluster_size=10,
        )
        _score, self.parts, _weights = compute_similarity_matrix(
            mean_traj=mean_traj,
            mean_feat=mean_feat,
            mean_rgb=mean_rgb,
            valid=valid,
            w_traj=0.7,
            w_feat=0.0,
            w_color=0.3,
            sigma_color=0.5,
        )

        disp = mean_traj - mean_traj[:, 0:1, :]
        motion = np.sqrt((disp**2).sum(axis=2).mean(axis=1))
        area, border, self.visibility_corr, self.total_pixels, _n_frames = collect_image_stats(
            str(self.cluster_ids_dir), self.n_clusters
        )
        self.area = area
        self.objectness = objectness_scores(
            motion=motion,
            area=area,
            border=border,
            total_pixels=self.total_pixels,
            border_scale=0.010,
            area_scale=0.010,
            max_area_frac=0.45,
        )

        pred_paths = index_pngs(str(self.cluster_ids_dir))
        gt_dir = data_root / SCENE_SPLITS[scene] / scene / "gt_masks"
        if not gt_dir.exists():
            gt_dir = Path(row["mask_dir"])
        raw_gt_paths = index_pngs(str(gt_dir))
        gt_paths, _alignment = align_single_object_gt_paths(pred_paths, raw_gt_paths, str(gt_dir))
        matched = sorted(set(pred_paths) & set(gt_paths))
        self.frames: list[tuple[np.ndarray, np.ndarray]] = []
        for fid in matched:
            pred = load_cluster_id_map(pred_paths[fid])
            gt = load_binary_mask(gt_paths[fid])
            if pred.shape == gt.shape:
                self.frames.append((pred, gt))
        if not self.frames:
            raise RuntimeError(f"No valid frames for {scene}")

        n_frames = len(self.frames)
        self.pred_count = np.zeros((self.n_clusters, n_frames), dtype=np.float64)
        self.inter_count = np.zeros((self.n_clusters, n_frames), dtype=np.float64)
        self.gt_count = np.zeros(n_frames, dtype=np.float64)
        self.frame_pixels = np.zeros(n_frames, dtype=np.float64)
        for frame_idx, (pred, gt) in enumerate(self.frames):
            clipped = np.minimum(pred.astype(np.int64), self.n_clusters)
            counts = np.bincount(clipped.ravel(), minlength=self.n_clusters + 1)
            inter = np.bincount(clipped[gt].ravel(), minlength=self.n_clusters + 1)
            self.pred_count[:, frame_idx] = counts[1 : self.n_clusters + 1]
            self.inter_count[:, frame_idx] = inter[1 : self.n_clusters + 1]
            self.gt_count[frame_idx] = float(gt.sum())
            self.frame_pixels[frame_idx] = float(gt.size)

    def groups_for(self, params: Params) -> list[list[int]]:
        components = build_components(
            parts=self.parts,
            objectness=self.objectness,
            area=self.area,
            visibility_corr=self.visibility_corr,
            total_pixels=self.total_pixels,
            w_traj=0.7,
            w_color=0.3,
            pair_tau=params.pair_tau,
            traj_min=params.traj_min,
            visibility_corr_max=params.visibility_corr_max,
            seed_objectness_min=0.0,
            member_objectness_min=0.0,
            max_component_clusters=params.max_component_clusters,
            max_component_area_frac=params.max_component_area_frac,
            max_seed_count=1_000_000,
            max_components=params.max_components,
            top_neighbors=params.top_neighbors,
            objectness_weight=params.objectness_weight,
            visibility_weight=params.visibility_weight,
        )
        return [group for _score, group in components]

    def evaluate_groups(self, groups: list[list[int]]) -> tuple[float, float, int, int]:
        if groups:
            new_id_of = mapping_for_components(self.n_clusters, groups)
        else:
            new_id_of = np.arange(self.n_clusters, dtype=np.int64)
        k_max = int(new_id_of.max()) + 1

        iou_by_k = [0.0] * (k_max + 1)
        acc_by_k = [0.0] * (k_max + 1)
        for new_zero in range(k_max):
            old_mask = new_id_of == new_zero
            pred_count = self.pred_count[old_mask].sum(axis=0)
            inter = self.inter_count[old_mask].sum(axis=0)
            union = pred_count + self.gt_count - inter
            ious = np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)
            correct = inter + (self.frame_pixels - (pred_count + self.gt_count - inter))
            accs = np.divide(correct, self.frame_pixels, out=np.zeros_like(correct), where=self.frame_pixels > 0)
            iou_by_k[new_zero + 1] = float(np.mean(ious))
            acc_by_k[new_zero + 1] = float(np.mean(accs))
        best_k = int(np.argmax(iou_by_k[1:]) + 1) if k_max else 0
        return iou_by_k[best_k], acc_by_k[best_k], best_k, len(groups)


def params_grid() -> list[Params]:
    values = itertools.product(
        [0.60, 0.65, 0.70, 0.75],
        [0.990, 0.995],
        [0.10, 0.25, 0.40],
        [3, 6, 10],
        [3, 5, 8],
        [0.45, 0.55, 0.70],
        [4, 8],
        [0.05, 0.15, 0.30],
        [0.00, 0.10, 0.25],
    )
    return [Params(*combo) for combo in values]


def load_rows(path: Path) -> dict[str, dict[str, str]]:
    with path.open(newline="") as f:
        return {row["scene"]: row for row in csv.DictReader(f)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", default="output/final_hypernerf_1/maskbenchmark_full10_res003.csv")
    parser.add_argument("--data-root", default="data/HyperNeRF")
    parser.add_argument("--out-csv", default="output/final_hypernerf_1/objectness_longrange_fast_sweep.csv")
    parser.add_argument("--load-iteration", type=int, default=30000)
    parser.add_argument("--top-n", type=int, default=25)
    args = parser.parse_args()

    rows = load_rows(Path(args.input_csv))
    data_root = Path(args.data_root)
    caches = []
    for scene in SCENE_ORDER:
        print(f"[cache] {scene}", flush=True)
        caches.append(SceneCache(scene, rows[scene], data_root, args.load_iteration))

    all_params = params_grid()
    print(f"[sweep] {len(all_params)} global parameter sets", flush=True)
    records = []
    for idx, params in enumerate(all_params, start=1):
        scene_scores = []
        scene_accs = []
        scene_groups = []
        scene_best = []
        for cache in caches:
            groups = cache.groups_for(params)
            miou, macc, best_k, n_groups = cache.evaluate_groups(groups)
            scene_scores.append(miou)
            scene_accs.append(macc)
            scene_groups.append(n_groups)
            scene_best.append(best_k)
        record = {
            "rank": 0,
            "mean_miou": float(np.mean(scene_scores)),
            "mean_macc": float(np.mean(scene_accs)),
            "tag": params.tag(),
            **params.__dict__,
        }
        for scene, miou, macc, best_k, n_groups in zip(
            SCENE_ORDER, scene_scores, scene_accs, scene_best, scene_groups
        ):
            record[f"{scene}_miou"] = miou
            record[f"{scene}_macc"] = macc
            record[f"{scene}_best_k"] = best_k
            record[f"{scene}_groups"] = n_groups
        records.append(record)
        if idx % 250 == 0:
            best = max(records, key=lambda item: item["mean_miou"])
            print(f"[sweep] {idx}/{len(all_params)} best={best['mean_miou']:.4f} {best['tag']}", flush=True)

    records.sort(key=lambda item: -item["mean_miou"])
    for rank, record in enumerate(records, start=1):
        record["rank"] = rank

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    columns = list(records[0].keys())
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(records)

    print(f"[done] wrote {out_csv}", flush=True)
    for record in records[: args.top_n]:
        print(f"#{record['rank']:02d} mean_miou={record['mean_miou']:.4f} tag={record['tag']}", flush=True)


if __name__ == "__main__":
    main()
