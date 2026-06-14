#!/usr/bin/env python3
"""Run no-mask objectness long-range merge and render merged cluster IDs."""

from __future__ import annotations

import argparse
import csv
import subprocess
from pathlib import Path

import numpy as np


SCENE_SPLITS = {
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


def run(cmd: list[str]) -> None:
    print(" ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def scene_model(scene: str) -> Path:
    return Path("output") / f"{scene}_30k_gs"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", default="output/final_hypernerf_1/maskbenchmark_full10_res003.csv")
    parser.add_argument("--data-root", default="data/HyperNeRF")
    parser.add_argument("--tag", default="objectness_longrange_default")
    parser.add_argument("--python", default="/workspace/.envs/trase/bin/python")
    parser.add_argument("--load-iteration", type=int, default=30000)
    parser.add_argument("--scenes", nargs="*", default=list(SCENE_SPLITS))
    parser.add_argument("--skip-existing", action="store_true")

    parser.add_argument("--pair-tau", type=float, default=0.70)
    parser.add_argument("--traj-min", type=float, default=0.995)
    parser.add_argument("--visibility-corr-max", type=float, default=0.25)
    parser.add_argument("--max-components", type=int, default=6)
    parser.add_argument("--max-component-clusters", type=int, default=5)
    parser.add_argument("--max-component-area-frac", type=float, default=0.55)
    parser.add_argument("--w-traj", type=float, default=0.7)
    parser.add_argument("--w-color", type=float, default=0.3)
    parser.add_argument("--top-neighbors", type=int, default=8)
    parser.add_argument("--objectness-weight", type=float, default=0.15)
    parser.add_argument("--visibility-weight", type=float, default=0.10)
    args = parser.parse_args()

    rows = {row["scene"]: row for row in csv.DictReader(open(args.input_csv, newline=""))}
    for scene in args.scenes:
        row = rows[scene]
        run_dir = Path(row["run_dir"])
        out_dir = run_dir / args.tag
        pred_dir = out_dir / "cluster_ids_train"
        if args.skip_existing and pred_dir.exists() and any(pred_dir.glob("*.png")):
            print(f"[skip] {scene}: {pred_dir} exists", flush=True)
            continue

        data_path = Path(args.data_root) / SCENE_SPLITS[scene] / scene
        model_path = scene_model(scene)
        labels_file = run_dir / "labels.npy"
        cluster_ids_dir = run_dir / "cluster_ids_train"

        run(
            [
                args.python,
                "self_supervised_scripts/merge_clusters_objectness.py",
                "-s",
                str(data_path),
                "--model_path",
                str(model_path),
                "--load_iteration",
                str(args.load_iteration),
                "--labels_file",
                str(labels_file),
                "--cluster_ids_dir",
                str(cluster_ids_dir),
                "--output_dir",
                str(out_dir),
                "--pair_tau",
                str(args.pair_tau),
                "--traj_min",
                str(args.traj_min),
                "--visibility_corr_max",
                str(args.visibility_corr_max),
                "--max_components",
                str(args.max_components),
                "--max_component_clusters",
                str(args.max_component_clusters),
                "--max_component_area_frac",
                str(args.max_component_area_frac),
                "--w_traj",
                str(args.w_traj),
                "--w_color",
                str(args.w_color),
                "--top_neighbors",
                str(args.top_neighbors),
                "--objectness_weight",
                str(args.objectness_weight),
                "--visibility_weight",
                str(args.visibility_weight),
            ]
        )

        merged_labels = out_dir / "labels_merged.npy"
        k_prime = int(np.load(merged_labels).max())
        run(
            [
                args.python,
                "self_supervised_scripts/render_clusters.py",
                "-s",
                str(data_path),
                "--model_path",
                str(model_path),
                "--load_iteration",
                str(args.load_iteration),
                "--labels_file",
                str(merged_labels),
                "--output_dir",
                str(out_dir),
                "--save_cluster_ids",
                "--load_image_on_the_fly",
                "--load_mask_on_the_fly",
                "--load2gpu_on_the_fly",
            ]
        )
        print(f"[done] {scene}: K'={k_prime} -> {pred_dir}", flush=True)


if __name__ == "__main__":
    main()
