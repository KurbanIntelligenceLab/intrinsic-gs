"""No-mask post-cluster merge using motion and rendered image objectness.

This is an experimental second layer after the first clustering pass. It does
not read GT masks or SAM masks. Given a `labels.npy` plus already-rendered
`cluster_ids_train/*.png`, it:

  1. computes the same long-range trajectory/color similarities used by
     merge_clusters.py,
  2. scores each cluster with an image-space objectness prior: enough visible
     area, low image-border occupancy, and high relative motion,
  3. greedily builds constrained foreground-like components, and
  4. writes labels_merged.npy plus a report.

Unlike merge_clusters.py, this avoids transitive closure across all strong
pairs. By default it writes multiple disjoint merge proposals and leaves
unrelated clusters separate, which is closer to the "greedy union" evaluation
behavior for single-object HyperNeRF mask benchmarks while still using no masks
as input.
"""

from __future__ import annotations

import math
import os
import sys
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import torch
from PIL import Image

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from scene import DeformModel
from self_supervised_scripts.cluster_trajectories import (
    compute_trajectories,
    resolve_ply_path,
)
from self_supervised_scripts.merge_clusters import (
    compute_cluster_aggregates,
    compute_similarity_matrix,
    load_gaussian_attrs,
    remap_labels,
)


def collect_image_stats(cluster_ids_dir: str, n_clusters: int):
    """Return per-cluster image stats from rendered ID PNGs."""
    paths = sorted(Path(cluster_ids_dir).glob("*.png"))
    if not paths:
        raise FileNotFoundError(f"No PNG cluster maps found in {cluster_ids_dir}")

    area = np.zeros(n_clusters + 1, dtype=np.int64)
    border = np.zeros(n_clusters + 1, dtype=np.int64)
    visible_area = np.zeros((n_clusters + 1, len(paths)), dtype=np.float64)
    total_pixels = 0

    for frame_idx, path in enumerate(paths):
        labels = np.asarray(Image.open(path))
        if labels.ndim == 3:
            labels = labels[..., 0]
        labels = labels.astype(np.int32)
        total_pixels += labels.size

        counts = np.bincount(labels.ravel(), minlength=n_clusters + 1)
        area[: len(counts)] += counts
        visible_area[: len(counts), frame_idx] = counts

        edge_pixels = np.concatenate(
            [labels[0, :], labels[-1, :], labels[:, 0], labels[:, -1]]
        )
        edge_counts = np.bincount(edge_pixels, minlength=n_clusters + 1)
        border[: len(edge_counts)] += edge_counts

    visibility_corr = np.zeros((n_clusters, n_clusters), dtype=np.float64)
    for i in range(n_clusters):
        vi = visible_area[i + 1]
        for j in range(i + 1, n_clusters):
            vj = visible_area[j + 1]
            if vi.std() > 1e-9 and vj.std() > 1e-9:
                corr = float(np.corrcoef(vi, vj)[0, 1])
            else:
                corr = 0.0
            visibility_corr[i, j] = corr
            visibility_corr[j, i] = corr

    return area, border, visibility_corr, total_pixels, len(paths)


def objectness_scores(
    motion: np.ndarray,
    area: np.ndarray,
    border: np.ndarray,
    total_pixels: int,
    border_scale: float,
    area_scale: float,
    max_area_frac: float,
):
    """Cluster prior in [0, 1-ish]: moving, visible, and not stuck to borders."""
    cluster_area = area[1:].astype(np.float64)
    area_frac = cluster_area / max(1, total_pixels)
    border_ratio = border[1:].astype(np.float64) / np.maximum(cluster_area, 1.0)

    lo, hi = np.percentile(motion, [5, 85])
    motion_norm = np.clip((motion - lo) / max(hi - lo, 1e-9), 0.0, 1.0)

    visible = 1.0 - np.exp(-area_frac / max(area_scale, 1e-9))
    border_ok = np.exp(-border_ratio / max(border_scale, 1e-9))
    not_huge = np.exp(-np.maximum(area_frac - max_area_frac, 0.0) / 0.10)
    return (0.2 + 0.8 * motion_norm) * visible * border_ok * not_huge


def build_component(
    parts: dict[str, np.ndarray],
    objectness: np.ndarray,
    area: np.ndarray,
    total_pixels: int,
    *,
    w_traj: float,
    w_color: float,
    pair_tau: float,
    traj_min: float,
    seed_objectness_min: float,
    member_objectness_min: float,
    max_component_clusters: int,
    max_component_area_frac: float,
    max_seed_count: int,
    objectness_weight: float,
    target_area_frac: float,
):
    """Greedily choose a single foreground-like merge component."""
    n_clusters = objectness.shape[0]
    total_w = max(w_traj + w_color, 1e-9)
    pair = (w_traj / total_w) * parts["traj"] + (w_color / total_w) * parts["color"]
    np.fill_diagonal(pair, 0.0)

    seeds = [
        int(idx)
        for idx in np.argsort(-objectness)[:max_seed_count]
        if objectness[idx] >= seed_objectness_min
    ]
    candidates = []

    for seed in seeds:
        group = [seed]
        while len(group) < max_component_clusters:
            best = None
            for other in range(n_clusters):
                if other in group:
                    continue
                if objectness[other] < member_objectness_min:
                    continue
                traj = max(parts["traj"][other, old] for old in group)
                sim = max(pair[other, old] for old in group)
                if traj < traj_min or sim < pair_tau:
                    continue
                union_area = sum(area[old + 1] for old in group) + area[other + 1]
                if union_area / max(1, total_pixels) > max_component_area_frac:
                    continue
                score = sim + objectness_weight * objectness[other]
                if best is None or score > best[0]:
                    best = (score, int(other))
            if best is None:
                break
            group.append(best[1])

        if len(group) < 2:
            continue

        union_area_frac = sum(area[old + 1] for old in group) / max(1, total_pixels)
        mean_objectness = float(np.mean([objectness[old] for old in group]))
        mean_pair = float(
            np.mean(
                [
                    pair[a, b]
                    for pos, a in enumerate(group)
                    for b in group[pos + 1 :]
                ]
            )
        )
        area_prior = 1.0 - abs(union_area_frac - target_area_frac) / max(
            target_area_frac, 1e-9
        )
        group_score = mean_objectness + 0.35 * mean_pair + 0.15 * area_prior
        candidates.append((group_score, group, union_area_frac, mean_objectness, mean_pair))

    candidates.sort(key=lambda item: -item[0])
    return candidates[0] if candidates else None


def build_components(
    parts: dict[str, np.ndarray],
    objectness: np.ndarray,
    area: np.ndarray,
    visibility_corr: np.ndarray,
    total_pixels: int,
    *,
    w_traj: float,
    w_color: float,
    pair_tau: float,
    traj_min: float,
    visibility_corr_max: float,
    seed_objectness_min: float,
    member_objectness_min: float,
    max_component_clusters: int,
    max_component_area_frac: float,
    max_seed_count: int,
    max_components: int,
    top_neighbors: int,
    objectness_weight: float,
    visibility_weight: float,
):
    """Build multiple disjoint merge proposals from strict pair candidates."""
    n_clusters = objectness.shape[0]
    total_w = max(w_traj + w_color, 1e-9)
    pair = (w_traj / total_w) * parts["traj"] + (w_color / total_w) * parts["color"]
    np.fill_diagonal(pair, 0.0)

    candidates = []
    seed_ids = [
        int(idx)
        for idx in np.argsort(-objectness)[:max_seed_count]
        if objectness[idx] >= seed_objectness_min
    ]

    for seed in seed_ids:
        neighbors = []
        for other in range(n_clusters):
            if other == seed:
                continue
            if objectness[other] < member_objectness_min:
                continue
            if parts["traj"][seed, other] < traj_min:
                continue
            if pair[seed, other] < pair_tau:
                continue
            if visibility_corr[seed, other] > visibility_corr_max:
                continue
            if (area[seed + 1] + area[other + 1]) / max(1, total_pixels) > max_component_area_frac:
                continue
            corr_bonus = max(0.0, visibility_corr_max - visibility_corr[seed, other])
            score = (
                pair[seed, other]
                + objectness_weight * (objectness[seed] + objectness[other]) / 2.0
                + visibility_weight * corr_bonus
            )
            neighbors.append((score, int(other)))

        neighbors.sort(key=lambda item: -item[0])
        for score, other in neighbors[:top_neighbors]:
            candidates.append((score, [seed, other]))

        group = [seed]
        for _score, other in neighbors:
            if len(group) >= max_component_clusters:
                break
            if other in group:
                continue
            union_area = sum(area[old + 1] for old in group) + area[other + 1]
            if union_area / max(1, total_pixels) > max_component_area_frac:
                continue
            if max(parts["traj"][other, old] for old in group) < traj_min:
                continue
            if min(visibility_corr[other, old] for old in group) > visibility_corr_max:
                continue
            group.append(other)

        if len(group) >= 2:
            mean_pair = float(
                np.mean(
                    [
                        pair[a, b]
                        for pos, a in enumerate(group)
                        for b in group[pos + 1 :]
                    ]
                )
            )
            mean_obj = float(np.mean([objectness[old] for old in group]))
            mean_corr = float(
                np.mean(
                    [
                        visibility_corr[a, b]
                        for pos, a in enumerate(group)
                        for b in group[pos + 1 :]
                    ]
                )
            )
            corr_bonus = max(0.0, visibility_corr_max - mean_corr)
            union_area_frac = sum(area[old + 1] for old in group) / max(1, total_pixels)
            score = (
                mean_pair
                + objectness_weight * mean_obj
                + visibility_weight * corr_bonus
                - 0.05 * max(0.0, union_area_frac - max_component_area_frac * 0.75)
            )
            candidates.append((score, group))

    seen = set()
    deduped = []
    for score, group in sorted(candidates, key=lambda item: -item[0]):
        unique_group = tuple(sorted(set(group)))
        if len(unique_group) < 2 or unique_group in seen:
            continue
        seen.add(unique_group)
        deduped.append((score, list(unique_group)))

    used = set()
    groups = []
    for score, group in deduped:
        if any(old in used for old in group):
            continue
        groups.append((score, group))
        used.update(group)
        if len(groups) >= max_components:
            break
    return groups


def mapping_for_single_component(n_clusters: int, group: list[int]):
    """Return old 0-based cluster id -> new 0-based cluster id."""
    group_set = set(group)
    new_id_of = np.zeros(n_clusters, dtype=np.int64)
    next_id = 1
    for old in range(n_clusters):
        if old in group_set:
            new_id_of[old] = 0
        else:
            new_id_of[old] = next_id
            next_id += 1
    return new_id_of


def mapping_for_components(n_clusters: int, groups: list[list[int]]):
    """Return old 0-based cluster id -> new 0-based cluster id."""
    new_id_of = np.full(n_clusters, -1, dtype=np.int64)
    next_id = 0
    for group in groups:
        for old in group:
            new_id_of[old] = next_id
        next_id += 1
    for old in range(n_clusters):
        if new_id_of[old] < 0:
            new_id_of[old] = next_id
            next_id += 1
    return new_id_of


def write_report(path: str, args, component, components, objectness, motion, area, border):
    lines = ["# Objectness long-range merge report", ""]
    lines.append("- **Uses GT/SAM masks:** no")
    lines.append(f"- **cluster_ids_dir:** {args.cluster_ids_dir}")
    lines.append(f"- **pair_tau:** {args.pair_tau}")
    lines.append(f"- **traj_min:** {args.traj_min}")
    lines.append(f"- **seed_objectness_min:** {args.seed_objectness_min}")
    lines.append(f"- **member_objectness_min:** {args.member_objectness_min}")
    lines.append("")

    if components:
        lines.append("## Selected Components")
        lines.append("| new_component | png_ids | score |")
        lines.append("|---:|---|---:|")
        for idx, (score, group) in enumerate(components):
            lines.append(f"| {idx} | {[old + 1 for old in group]} | {score:.4f} |")
    elif component is not None:
        score, group, area_frac, mean_obj, mean_pair = component
        lines.append(f"- **Selected old ids (0-based):** {group}")
        lines.append(f"- **Selected cluster PNG ids (1-based):** {[old + 1 for old in group]}")
        lines.append(f"- **component_score:** {score:.4f}")
        lines.append(f"- **component_area_frac:** {area_frac:.4f}")
        lines.append(f"- **mean_objectness:** {mean_obj:.4f}")
        lines.append(f"- **mean_pair_similarity:** {mean_pair:.4f}")
    else:
        lines.append("No component passed the merge constraints.")

    lines.append("")
    lines.append("## Cluster Scores")
    lines.append("| png_id | objectness | motion | area | border_ratio |")
    lines.append("|---:|---:|---:|---:|---:|")
    for old in np.argsort(-objectness):
        png_id = int(old + 1)
        ar = int(area[png_id])
        br = float(border[png_id]) / max(1, ar)
        lines.append(
            f"| {png_id} | {objectness[old]:.4f} | {motion[old]:.5f} | {ar} | {br:.5f} |"
        )

    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def main(args):
    if not os.path.exists(args.labels_file):
        raise FileNotFoundError(args.labels_file)

    labels_full = np.load(args.labels_file)
    valid_mask = labels_full > 0
    if not valid_mask.any():
        raise RuntimeError("labels.npy has no valid non-zero entries.")

    labels_valid = labels_full[valid_mask].astype(np.int64) - 1
    n_clusters = int(labels_valid.max()) + 1

    resolver_args = type(
        "ResolveArgs",
        (),
        {
            "labels": args.labels_file,
            "ply_path": args.ply_path,
            "model_path": args.model_path,
            "load_iteration": args.load_iteration,
        },
    )()
    ply_path = resolve_ply_path(resolver_args)
    xyz, _opacity, color, feats = load_gaussian_attrs(ply_path)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    pos_valid = torch.from_numpy(xyz[valid_mask]).to(device)
    scene_name = os.path.basename(os.path.normpath(args.source_path))
    deform_path = args.deform_path if args.deform_path else PROJECT_ROOT
    deform = DeformModel(is_blender=False, is_6dof=False)
    deform.load_weights(deform_path, iteration=args.load_iteration, scene_name=scene_name)

    traj_xyz_t = compute_trajectories(deform, pos_valid, args.n_time_steps)
    traj_xyz = traj_xyz_t.cpu().numpy().astype(np.float64)

    mean_traj, mean_feat, mean_rgb, sizes, valid = compute_cluster_aggregates(
        labels_valid=labels_valid,
        traj_xyz=traj_xyz,
        feats=feats[valid_mask] if feats is not None else None,
        color=color[valid_mask],
        n_clusters=n_clusters,
        min_cluster_size=args.min_cluster_size,
    )
    _score, parts, _weights = compute_similarity_matrix(
        mean_traj=mean_traj,
        mean_feat=mean_feat,
        mean_rgb=mean_rgb,
        valid=valid,
        w_traj=args.w_traj,
        w_feat=0.0,
        w_color=args.w_color,
        sigma_color=args.sigma_color,
    )

    disp = mean_traj - mean_traj[:, 0:1, :]
    motion = np.sqrt((disp**2).sum(axis=2).mean(axis=1))
    area, border, visibility_corr, total_pixels, n_frames = collect_image_stats(
        args.cluster_ids_dir, n_clusters
    )
    objectness = objectness_scores(
        motion=motion,
        area=area,
        border=border,
        total_pixels=total_pixels,
        border_scale=args.border_scale,
        area_scale=args.area_scale,
        max_area_frac=args.objectness_max_area_frac,
    )

    components = []
    component = None
    if args.single_component:
        component = build_component(
            parts=parts,
            objectness=objectness,
            area=area,
            total_pixels=total_pixels,
            w_traj=args.w_traj,
            w_color=args.w_color,
            pair_tau=args.pair_tau,
            traj_min=args.traj_min,
            seed_objectness_min=args.seed_objectness_min,
            member_objectness_min=args.member_objectness_min,
            max_component_clusters=args.max_component_clusters,
            max_component_area_frac=args.max_component_area_frac,
            max_seed_count=args.max_seed_count,
            objectness_weight=args.objectness_weight,
            target_area_frac=args.target_area_frac,
        )
    else:
        components = build_components(
            parts=parts,
            objectness=objectness,
            area=area,
            visibility_corr=visibility_corr,
            total_pixels=total_pixels,
            w_traj=args.w_traj,
            w_color=args.w_color,
            pair_tau=args.pair_tau,
            traj_min=args.traj_min,
            visibility_corr_max=args.visibility_corr_max,
            seed_objectness_min=args.seed_objectness_min,
            member_objectness_min=args.member_objectness_min,
            max_component_clusters=args.max_component_clusters,
            max_component_area_frac=args.max_component_area_frac,
            max_seed_count=args.max_seed_count,
            max_components=args.max_components,
            top_neighbors=args.top_neighbors,
            objectness_weight=args.objectness_weight,
            visibility_weight=args.visibility_weight,
        )

    out_dir = args.output_dir or os.path.dirname(os.path.abspath(args.labels_file))
    os.makedirs(out_dir, exist_ok=True)

    if components:
        groups = [group for _score, group in components]
        new_id_of = mapping_for_components(n_clusters, groups)
        merged = remap_labels(labels_full, new_id_of)
        print(
            "Selected merge components "
            f"png_ids={[[old + 1 for old in group] for group in groups]} "
            f"from {n_frames} rendered frames."
        )
    elif component is None:
        merged = labels_full
        print("No objectness component selected; writing pass-through labels.")
    else:
        _score, group, _area_frac, _mean_obj, _mean_pair = component
        new_id_of = mapping_for_single_component(n_clusters, group)
        merged = remap_labels(labels_full, new_id_of)
        print(
            "Selected merge component "
            f"png_ids={[old + 1 for old in group]} "
            f"(old_ids={group}) from {n_frames} rendered frames."
        )

    out_labels = os.path.join(out_dir, "labels_merged.npy")
    np.save(out_labels, merged)
    write_report(
        os.path.join(out_dir, "merge_objectness_report.md"),
        args,
        component,
        components,
        objectness,
        motion,
        area,
        border,
    )
    print(f"Wrote {out_labels}")


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("-s", "--source_path", required=True)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--load_iteration", type=int, default=30000)
    parser.add_argument("--labels_file", required=True)
    parser.add_argument("--cluster_ids_dir", required=True)
    parser.add_argument("--output_dir", default="")
    parser.add_argument("--ply_path", default="")
    parser.add_argument("--deform_path", default="")

    parser.add_argument("--n_time_steps", type=int, default=20)
    parser.add_argument("--min_cluster_size", type=int, default=10)
    parser.add_argument("--w_traj", type=float, default=0.7)
    parser.add_argument("--w_color", type=float, default=0.3)
    parser.add_argument("--sigma_color", type=float, default=0.5)

    parser.add_argument("--pair_tau", type=float, default=0.70)
    parser.add_argument("--traj_min", type=float, default=0.995)
    parser.add_argument("--visibility_corr_max", type=float, default=0.25)
    parser.add_argument("--seed_objectness_min", type=float, default=0.0)
    parser.add_argument("--member_objectness_min", type=float, default=0.0)
    parser.add_argument("--max_component_clusters", type=int, default=5)
    parser.add_argument("--max_component_area_frac", type=float, default=0.55)
    parser.add_argument("--max_seed_count", type=int, default=1000000)
    parser.add_argument("--max_components", type=int, default=6)
    parser.add_argument("--top_neighbors", type=int, default=8)
    parser.add_argument("--objectness_weight", type=float, default=0.15)
    parser.add_argument("--visibility_weight", type=float, default=0.10)
    parser.add_argument("--target_area_frac", type=float, default=0.28)
    parser.add_argument("--single_component", action="store_true")

    parser.add_argument("--border_scale", type=float, default=0.010)
    parser.add_argument("--area_scale", type=float, default=0.010)
    parser.add_argument("--objectness_max_area_frac", type=float, default=0.45)

    main(parser.parse_args())
