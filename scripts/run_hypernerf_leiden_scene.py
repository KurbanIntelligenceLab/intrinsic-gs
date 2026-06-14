#!/usr/bin/env python
"""Run segmentation + mIoU/mAcc for one 30k HyperNeRF scene."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from self_supervised_scripts.pipeline_common import (
    ScenePaths,
    SpectralParams,
    detect_actual_k,
    run_macc,
    run_miou,
    run_render,
    run_spectral,
)


SCENE_RELS = {
    "chickchicken": "interp/chickchicken",
    "cut-lemon1": "interp/cut-lemon1",
    "hand1-dense-v2": "interp/hand1-dense-v2",
    "slice-banana": "interp/slice-banana",
    "torchocolate": "interp/torchocolate",
    "americano": "misc/americano",
    "espresso": "misc/espresso",
    "keyboard": "misc/keyboard",
    "oven-mitts": "misc/oven-mitts",
    "split-cookie": "misc/split-cookie",
}


def force_symlink(src: Path, dst: Path) -> None:
    if dst.is_symlink() or dst.exists():
        if dst.is_symlink() and Path(os.readlink(dst)) == src:
            return
        if dst.is_file():
            dst.unlink()
        else:
            return
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.symlink_to(src)


def prepare_scene(scene: str, repo: Path) -> ScenePaths:
    rel = SCENE_RELS[scene]
    data = repo / "data" / "HyperNeRF" / rel
    model = repo / "output" / f"{scene}_30k_gs"
    gt = repo / "data" / "Mask-Benchmark" / "Mask-Benchmark" / "HyperNeRF-Mask" / scene / "gt_masks"
    deform_src = model / "deform" / "iteration_30000" / "deform.pth"
    deform_dst = repo / "deform" / f"deform_{scene}.pth"

    for path in (data, model, gt, deform_src):
        if not path.exists():
            raise FileNotFoundError(path)
    if not (model / "point_cloud" / "iteration_30000" / "point_cloud.ply").exists():
        raise FileNotFoundError(model / "point_cloud" / "iteration_30000" / "point_cloud.ply")

    force_symlink(gt, data / "gt_masks")
    force_symlink(deform_src, deform_dst)
    return ScenePaths(scene=scene, data=str(data), model=str(model))


def _fmt_float(value: float) -> str:
    return str(value)


def _expected_run_tokens(args: argparse.Namespace) -> list[str]:
    tokens = [
        f"{args.clusterer}_res{_fmt_float(args.leiden_resolution)}",
        f"sc{_fmt_float(args.sigma_color)}",
        f"ss{_fmt_float(args.sigma_scale)}",
        f"p{_fmt_float(args.power)}",
        f"k{args.knn_k}",
    ]
    if args.use_motion:
        tokens.append(f"mot{args.n_time_steps}")
    if args.use_boundary:
        if args.rgb_edge_method == "pidinet":
            tokens.append(f"B-pd{args.pidinet_variant[0].upper()}")
        else:
            tokens.append("B-v")
    return tokens


def resolve_matching_run(paths: ScenePaths, args: argparse.Namespace, candidate: str) -> str:
    """Recover the run matching this invocation when other jobs share a scene.

    pipeline_common selects the newest run directory after spectral clustering.
    That is fragile when independent sweeps are running for the same scene.
    The directory name encodes the key parameters, so reselect by signature.
    """
    tokens = _expected_run_tokens(args)
    candidate_path = Path(candidate)
    if all(token in candidate_path.name for token in tokens):
        return candidate

    model = Path(paths.model)
    matches = [
        path
        for path in model.glob(f"*/{args.clusterer}_*")
        if path.is_dir() and all(token in path.name for token in tokens)
    ]
    if not matches:
        return candidate
    selected = max(matches, key=lambda path: (path.stat().st_mtime, path.name))
    print(f"RESELECTED_SPEC_RUN={selected} (was {candidate})", flush=True)
    return str(selected)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", required=True, choices=sorted(SCENE_RELS))
    parser.add_argument("--iteration", type=int, default=30000)
    parser.add_argument(
        "--clusterer",
        choices=("kmeans", "leiden", "hdbscan"),
        default="leiden",
    )
    parser.add_argument("--n_clusters", type=int, default=14)
    parser.add_argument("--eigengap_k", type=int, default=15)
    parser.add_argument("--leiden_resolution", type=float, default=0.018)
    parser.add_argument("--hdbscan_min_cluster_size_frac", type=float, default=0.015)
    parser.add_argument("--hdbscan_min_samples", type=int, default=5)
    parser.add_argument("--no_geo", dest="use_geometry", action="store_false")
    parser.set_defaults(use_geometry=True)
    parser.add_argument("--use_motion", action="store_true")
    parser.add_argument("--n_time_steps", type=int, default=20)
    parser.add_argument("--motion_floor", type=float, default=0.2)
    parser.add_argument("--static_motion_thresh", type=float, default=1e-3)
    parser.add_argument("--use_boundary", action="store_true")
    parser.add_argument("--boundary_views", type=int, default=12)
    parser.add_argument("--alpha_depth", type=float, default=5.0)
    parser.add_argument("--beta_rgb", type=float, default=2.0)
    parser.add_argument("--gamma", type=float, default=2.0)
    parser.add_argument("--presmooth_sigma", type=float, default=0.0)
    parser.add_argument("--sigma_color", type=float, default=0.8)
    parser.add_argument("--sigma_scale", type=float, default=1.0)
    parser.add_argument("--power", type=float, default=2.0)
    parser.add_argument("--knn_k", type=int, default=20)
    parser.add_argument("--rgb_edge_method", choices=("sobel", "pidinet"), default="sobel")
    parser.add_argument("--pidinet_variant", choices=("full", "small", "tiny"), default="full")
    parser.add_argument("--skip_done", action="store_true")
    args = parser.parse_args()

    repo = Path(__file__).resolve().parents[1]
    os.chdir(repo)
    paths = prepare_scene(args.scene, repo)

    params = SpectralParams(
        clusterer=args.clusterer,
        n_clusters=args.n_clusters,
        eigengap_k=args.eigengap_k,
        leiden_resolution=args.leiden_resolution,
        hdbscan_min_cluster_size_frac=args.hdbscan_min_cluster_size_frac,
        hdbscan_min_samples=args.hdbscan_min_samples,
        use_geometry=args.use_geometry,
        sigma_color=args.sigma_color,
        sigma_scale=args.sigma_scale,
        power=args.power,
        knn_k=args.knn_k,
        use_motion=args.use_motion,
        n_time_steps=args.n_time_steps,
        motion_floor=args.motion_floor,
        static_motion_thresh=args.static_motion_thresh,
        use_boundary=args.use_boundary,
        boundary_views=args.boundary_views,
        alpha_depth=args.alpha_depth,
        beta_rgb=args.beta_rgb,
        gamma=args.gamma,
        presmooth_sigma=args.presmooth_sigma,
        rgb_edge_method=args.rgb_edge_method,
        pidinet_variant=args.pidinet_variant,
        max_palette_views=0,
    )

    spec_run = run_spectral(paths, params, args.iteration)
    spec_run = resolve_matching_run(paths, args, spec_run)
    if args.skip_done and (Path(spec_run) / "miou_results_greedy.json").exists():
        print(f"[skip eval] {args.scene}: {spec_run} already has greedy mIoU")
        return

    run_render(paths, spec_run, args.iteration)
    actual_k = detect_actual_k(spec_run)
    run_miou(paths, spec_run, actual_k, "best_cluster")
    run_macc(paths, spec_run, "best_cluster")
    run_miou(paths, spec_run, actual_k, "greedy_union")
    run_macc(paths, spec_run, "greedy_union")
    print(f"SCENE_DONE {args.scene} {spec_run}")


if __name__ == "__main__":
    main()
