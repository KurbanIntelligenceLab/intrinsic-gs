#!/usr/bin/env python
"""Run one Neu3D Leiden component-ablation scene.

This mirrors ``run_hypernerf_leiden_scene.py`` but uses the local Neu3D data,
test-camera evaluation, and a lean metric pass without visualization panels.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from self_supervised_scripts.pipeline_common import (  # noqa: E402
    ScenePaths,
    SpectralParams,
    detect_actual_k,
    run_render,
    run_spectral,
)


SCENES = (
    "coffee_martini",
    "cook_spinach",
    "cut_roasted_beef",
    "flame_steak",
    "sear_steak",
)


def run_cmd(cmd: list[str]) -> None:
    print(" ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def prepare_scene(scene: str, repo: Path, dataset_root: Path) -> ScenePaths:
    data = dataset_root / scene
    model = repo / "output" / f"{scene}_run"
    deform = repo / "deform" / f"deform_{scene}.pth"
    for path in (
        data,
        data / "gt_masks",
        model,
        model / "point_cloud" / "iteration_30000" / "point_cloud.ply",
        deform,
    ):
        if not path.exists():
            raise FileNotFoundError(path)
    return ScenePaths(scene=scene, data=str(data), model=str(model))


def existing_done(output_root: Path, variant: str, scene: str) -> Path | None:
    scene_root = output_root / "runs" / variant / scene
    if not scene_root.exists():
        return None
    candidates = sorted(
        (
            path
            for path in scene_root.iterdir()
            if path.is_dir()
            and (path / "miou_results.json").exists()
            and (path / "miou_results_greedy.json").exists()
            and (path / "macc_results.json").exists()
            and (path / "macc_results_greedy.json").exists()
        ),
        key=lambda path: (path.stat().st_mtime, path.name),
        reverse=True,
    )
    return candidates[0] if candidates else None


def relocate_run(spec_run: str, output_root: Path, variant: str, scene: str) -> str:
    src = Path(spec_run)
    dst = output_root / "runs" / variant / scene / src.name
    if dst.exists():
        suffix = 2
        while dst.with_name(f"{dst.name}_{suffix}").exists():
            suffix += 1
        dst = dst.with_name(f"{dst.name}_{suffix}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(src), str(dst))
    print(f"RELOCATED_SPEC_RUN={dst}", flush=True)
    return str(dst)


def eval_metrics(paths: ScenePaths, spec_run: str, iteration: int) -> None:
    for mode in ("best_cluster", "greedy_union"):
        suffix = "_greedy" if mode == "greedy_union" else ""
        miou_json = f"{spec_run}/miou_results{suffix}.json"
        miou_cmd = [
            "python",
            "self_supervised_scripts/compute_miou.py",
            "--pred_dir",
            f"{spec_run}/cluster_ids_test",
            "--gt_dir",
            f"{paths.data}/gt_masks",
            "--report_md",
            f"{spec_run}/report.md",
            "--output_json",
            miou_json,
        ]
        if mode == "greedy_union":
            miou_cmd.extend(["--selection_mode", "greedy_union"])
        run_cmd(miou_cmd)
        run_cmd(
            [
                "python",
                "self_supervised_scripts/compute_macc.py",
                "--pred_dir",
                f"{spec_run}/cluster_ids_test",
                "--gt_dir",
                f"{paths.data}/gt_masks",
                "--miou_json",
                miou_json,
                "--output_json",
                f"{spec_run}/macc_results{suffix}.json",
            ]
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", required=True, choices=SCENES)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--dataset_root", default="data/Neu3D")
    parser.add_argument("--output_root", default="output/final_neu3d_component_ablations")
    parser.add_argument("--iteration", type=int, default=30000)
    parser.add_argument("--leiden_resolution", type=float, default=0.03)
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

    repo = PROJECT_ROOT
    os.chdir(repo)
    output_root = repo / args.output_root
    if args.skip_done:
        done = existing_done(output_root, args.variant, args.scene)
        if done is not None:
            print(f"[skip done] {args.variant}/{args.scene}: {done}", flush=True)
            return

    paths = prepare_scene(args.scene, repo, repo / args.dataset_root)
    params = SpectralParams(
        clusterer="leiden",
        leiden_resolution=args.leiden_resolution,
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
    spec_run = relocate_run(spec_run, output_root, args.variant, args.scene)
    run_render(paths, spec_run, args.iteration, use_test_cameras=True)
    eval_metrics(paths, spec_run, args.iteration)
    print(f"SCENE_DONE {args.variant} {args.scene} {spec_run}", flush=True)


if __name__ == "__main__":
    main()
