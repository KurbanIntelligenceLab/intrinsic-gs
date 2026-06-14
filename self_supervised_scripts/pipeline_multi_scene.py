"""Run one spectral/eval parameter config across multiple scenes.

Examples:
    # HyperNeRF (default)
    python self_supervised_scripts/pipeline_multi_scene.py \
        --scenes americano cut-lemon1 \
        --ablation_name baseline \
        --n_clusters 14 --eigengap_k 15

    # Neu3D — picks up images_2x, test-camera split, iteration 30000
    python self_supervised_scripts/pipeline_multi_scene.py \
        --dataset neu3d \
        --scenes coffee_martini cook_spinach cut_roasted_beef flame_steak sear_steak \
        --ablation_name neu3d_baseline

    python self_supervised_scripts/pipeline_multi_scene.py --scenes all

Run from the repository root so train.py and self_supervised_scripts/ are
available at the expected relative paths.
"""

from __future__ import annotations

import argparse
import os
import sys
import traceback

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from self_supervised_scripts.pipeline_common import (  # noqa: E402
    DATASETS,
    DEFAULT_DATASET_ROOT,
    DEFAULT_ITERATION,
    DatasetConfig,
    MergeParams,
    SpectralParams,
    discover_scenes,
    run_pipeline_pass,
)


def add_spectral_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--clusterer",
        type=str,
        default="kmeans",
        choices=["kmeans", "leiden", "hdbscan"],
        help="Clustering algorithm.",
    )
    parser.add_argument("--n_clusters", type=int, default=14)
    parser.add_argument("--eigengap_k", type=int, default=15)
    parser.add_argument("--leiden_resolution", type=float, default=1.0)
    parser.add_argument("--hdbscan_min_cluster_size_frac", type=float, default=0.015)
    parser.add_argument("--hdbscan_min_samples", type=int, default=5)
    parser.add_argument("--sigma_color", type=float, default=0.8)
    parser.add_argument("--sigma_scale", type=float, default=1.0)
    parser.add_argument("--power", type=float, default=2.0)
    parser.add_argument("--knn_k", type=int, default=20)
    geo_group = parser.add_mutually_exclusive_group()
    geo_group.add_argument("--use_geo", dest="use_geometry", action="store_true")
    geo_group.add_argument("--no_geo", dest="use_geometry", action="store_false")
    parser.set_defaults(use_geometry=True)
    motion_group = parser.add_mutually_exclusive_group()
    motion_group.add_argument("--use_motion", dest="use_motion", action="store_true")
    motion_group.add_argument("--no_motion", dest="use_motion", action="store_false")
    parser.set_defaults(use_motion=True)
    parser.add_argument("--n_time_steps", type=int, default=20)
    parser.add_argument("--motion_floor", type=float, default=0.2)
    parser.add_argument("--static_motion_thresh", type=float, default=1e-3)
    boundary_group = parser.add_mutually_exclusive_group()
    boundary_group.add_argument("--use_boundary", dest="use_boundary", action="store_true")
    boundary_group.add_argument("--no_boundary", dest="use_boundary", action="store_false")
    parser.set_defaults(use_boundary=True)
    parser.add_argument("--boundary_views", type=int, default=12)
    parser.add_argument("--alpha_depth", type=float, default=5.0)
    parser.add_argument("--beta_rgb", type=float, default=2.0)
    parser.add_argument("--gamma", type=float, default=2.0)
    parser.add_argument("--opacity_thresh", type=float, default=0.05)
    parser.add_argument("--presmooth_sigma", type=float, default=0.0)
    parser.add_argument(
        "--solver",
        type=str,
        default="cupy",
        choices=["lobpcg", "cupy", "randomized", "arpack"],
    )
    parser.add_argument(
        "--rgb_edge_method",
        type=str,
        default="sobel",
        choices=["sobel", "pidinet"],
        help="RGB-branch edge detector for boundary suppression.",
    )
    parser.add_argument(
        "--pidinet_variant",
        type=str,
        default="full",
        choices=["full", "small", "tiny"],
        help="(pidinet only) Model size.",
    )
    parser.add_argument(
        "--write_annotated_ply",
        action="store_true",
        help="Write run-dir point_cloud.ply (checkpoint-sized ~0.5GB + cls). "
        "Default off; use export_segmented_ply.py when you need a PLY.",
    )
    parser.add_argument(
        "--max_palette_views",
        type=int,
        default=-1,
        help="Cap the train-camera palette render at N evenly-spaced views. "
        "<0 (default) renders all (~1 hr/scene on Neu3D); 0 skips entirely; "
        "N>0 renders N for quick visualisation (e.g. --max_palette_views 100 "
        "→ ~80 sec instead of ~1 hr). mIoU/mAcc are unaffected — only the "
        "'segmented' column of compute_miou's vis grid uses these PNGs.",
    )
    parser.add_argument(
        "--tc_camera_idx",
        type=int,
        default=-1,
        help="If >=0, render a fixed-camera time sweep at this view index and "
        "compute temporal consistency (TC). Default -1 disables TC.",
    )
    parser.add_argument(
        "--tc_n_steps",
        type=int,
        default=20,
        help="Number of time samples in the TC sweep (only used if tc_camera_idx >= 0).",
    )


def add_merge_args(parser: argparse.ArgumentParser) -> None:
    """Long-range cluster merge flags. Off by default; identical knobs
    across pipeline_multi_scene and pipeline_param_sweep so configs port."""
    parser.add_argument(
        "--enable_long_range_merge",
        action="store_true",
        help="After baseline eval, run merge_clusters.py and re-evaluate on "
             "labels_merged.npy. Adds *_merged variants of the four JSONs.",
    )
    parser.add_argument("--merge_w_traj", type=float, default=0.5)
    parser.add_argument("--merge_w_feat", type=float, default=0.3)
    parser.add_argument("--merge_w_color", type=float, default=0.2)
    parser.add_argument("--merge_tau", type=float, default=0.85,
                        help="Merge threshold on the weighted score.")
    parser.add_argument("--merge_sigma_color", type=float, default=0.5)
    parser.add_argument("--merge_n_time_steps", type=int, default=20)
    parser.add_argument("--merge_min_cluster_size", type=int, default=10)


def args_to_merge_params(args: argparse.Namespace) -> MergeParams:
    return MergeParams(
        enabled=getattr(args, "enable_long_range_merge", False),
        w_traj=getattr(args, "merge_w_traj", 0.5),
        w_feat=getattr(args, "merge_w_feat", 0.3),
        w_color=getattr(args, "merge_w_color", 0.2),
        tau=getattr(args, "merge_tau", 0.85),
        sigma_color=getattr(args, "merge_sigma_color", 0.5),
        n_time_steps=getattr(args, "merge_n_time_steps", 20),
        min_cluster_size=getattr(args, "merge_min_cluster_size", 10),
    )


def args_to_params(args: argparse.Namespace) -> SpectralParams:
    return SpectralParams(
        n_clusters=args.n_clusters,
        eigengap_k=args.eigengap_k,
        use_geometry=getattr(args, "use_geometry", True),
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
        opacity_thresh=getattr(args, "opacity_thresh", 0.05),
        presmooth_sigma=args.presmooth_sigma,
        solver=args.solver,
        clusterer=getattr(args, "clusterer", "kmeans"),
        leiden_resolution=getattr(args, "leiden_resolution", 1.0),
        hdbscan_min_cluster_size_frac=getattr(args, "hdbscan_min_cluster_size_frac", 0.015),
        hdbscan_min_samples=getattr(args, "hdbscan_min_samples", 5),
        rgb_edge_method=getattr(args, "rgb_edge_method", "sobel"),
        pidinet_variant=getattr(args, "pidinet_variant", "full"),
        write_annotated_ply=getattr(args, "write_annotated_ply", False),
        max_palette_views=getattr(args, "max_palette_views", -1),
        tc_camera_idx=getattr(args, "tc_camera_idx", -1),
        tc_n_steps=getattr(args, "tc_n_steps", 20),
    )


def resolve_scenes(scenes: list[str], dataset_root: str = DEFAULT_DATASET_ROOT) -> list[str]:
    if scenes == ["all"]:
        return discover_scenes(dataset_root)
    if "all" in scenes:
        raise ValueError("Use either '--scenes all' or explicit scene names, not both.")
    return scenes


def resolve_dataset(args: argparse.Namespace) -> tuple[DatasetConfig, str, int]:
    """Resolve dataset config + the effective dataset_root/iteration.

    `--dataset_root` and `--iteration` (when explicitly passed) override the
    DatasetConfig defaults; this lets the user point at a staging dataset
    root while keeping Neu3D's render/eval split semantics intact.
    """
    dataset = DATASETS[args.dataset]
    dataset_root = args.dataset_root or dataset.dataset_root
    iteration = args.iteration if args.iteration is not None else dataset.iteration
    return dataset, dataset_root, iteration


def assert_repo_root() -> None:
    if not os.path.exists("train.py"):
        raise SystemExit(
            "Run this script from the repository root, where train.py exists."
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scenes",
        nargs="+",
        required=True,
        help="Scene names, or the single keyword 'all' to discover every dataset scene.",
    )
    parser.add_argument(
        "--dataset",
        choices=sorted(DATASETS),
        default="hypernerf",
        help="Dataset preset: drives dataset_root, Stage 1 iteration, render/eval "
             "camera split, and original-image subdir. 'neu3d' uses test cameras + "
             "images_2x + iteration 30000.",
    )
    # `default=None` so DatasetConfig defaults can apply when user is silent.
    parser.add_argument(
        "--dataset_root",
        default=None,
        help="Override the dataset root (default: from --dataset preset).",
    )
    parser.add_argument(
        "--iteration",
        type=int,
        default=None,
        help="Override the Stage 1 iteration (default: from --dataset preset).",
    )
    parser.add_argument(
        "--ablation_name",
        required=True,
        help="Per-scene runs are grouped under multiple_ablation/<ablation_name>/<scene>/.",
    )
    parser.add_argument(
        "--retrain",
        action="store_true",
        help="Force training and republish the scene deform checkpoint.",
    )
    parser.add_argument(
        "--gt_subdir",
        default="gt_masks",
        help="GT mask subdirectory under each scene's data dir "
             "(e.g. 'gt_masks_sam' to eval against SAM-derived masks).",
    )
    add_spectral_args(parser)
    add_merge_args(parser)
    return parser


def main() -> None:
    assert_repo_root()
    parser = build_parser()
    args = parser.parse_args()

    dataset, dataset_root, iteration = resolve_dataset(args)
    scenes = resolve_scenes(args.scenes, dataset_root)
    params = args_to_params(args)
    merge = args_to_merge_params(args)
    output_group = os.path.join("multiple_ablation", args.ablation_name)
    failures = []

    print(f"Dataset: {dataset.name} (root={dataset_root}, iter={iteration}, "
          f"test_cams={dataset.use_test_cameras})")
    print(f"Running one parameter config on {len(scenes)} scene(s): {', '.join(scenes)}")
    print(f"Grouped outputs will be written under: {output_group}/<scene>/")
    for scene in scenes:
        print(f"\n{'#' * 72}\n# Scene: {scene}\n{'#' * 72}")
        try:
            run_pipeline_pass(
                scene,
                params,
                iteration=iteration,
                retrain=args.retrain,
                dataset_root=dataset_root,
                output_group=output_group,
                gt_subdir=args.gt_subdir,
                dataset=dataset,
                merge=merge,
            )
        except Exception as exc:
            traceback.print_exc()
            failures.append((scene, str(exc)))

    if failures:
        print(f"\n{len(failures)} scene(s) failed:")
        for scene, error in failures:
            print(f"  {scene}: {error}")
        raise SystemExit(1)

    print("\nAll scenes complete.")


if __name__ == "__main__":
    main()
