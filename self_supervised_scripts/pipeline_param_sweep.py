"""Run multiple spectral/eval parameter configs on one scene.

Examples:
    # HyperNeRF (default), zip mode: 3 runs
    python self_supervised_scripts/pipeline_param_sweep.py \
        --scene cut-lemon1 \
        --sweep_mode zip \
        --knn_k 20 30 40 \
        --beta_rgb 0.1 0.2 0.5

    # Neu3D — test cameras + images_2x + iter 30000
    python self_supervised_scripts/pipeline_param_sweep.py \
        --dataset neu3d \
        --scene coffee_martini \
        --sweep_mode zip \
        --beta_rgb 1.5 2.0 2.5

Run from the repository root so train.py and self_supervised_scripts/ are
available at the expected relative paths.
"""

from __future__ import annotations

import argparse
import itertools
import os
import sys
import traceback
from dataclasses import asdict
from pathlib import Path

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from self_supervised_scripts.pipeline_common import (  # noqa: E402
    DATASETS,
    DEFAULT_DATASET_ROOT,
    DEFAULT_ITERATION,
    DatasetConfig,
    MergeParams,
    SpectralParams,
    run_pipeline_pass,
)
from self_supervised_scripts.pipeline_multi_scene import (  # noqa: E402
    add_merge_args,
    args_to_merge_params,
)

# Each entry: (cli_name, type, default, choices_or_None).
# `choices_or_None=None` means free-form numeric value; a list means an
# argparse `choices=` constraint.
SWEEP_PARAMS = [
    ("clusterer", str, "kmeans", ["kmeans", "leiden", "hdbscan"]),
    ("n_clusters", int, 14, None),
    ("eigengap_k", int, 15, None),
    ("leiden_resolution", float, 1.0, None),
    ("hdbscan_min_cluster_size_frac", float, 0.015, None),
    ("hdbscan_min_samples", int, 5, None),
    ("use_geometry", bool, True, None),
    ("rgb_edge_method", str, "sobel", ["sobel", "pidinet"]),
    ("pidinet_variant", str, "full", ["full", "small", "tiny"]),
    ("sigma_color", float, 0.8, None),
    ("sigma_scale", float, 1.0, None),
    ("power", float, 2.0, None),
    ("knn_k", int, 20, None),
    ("n_time_steps", int, 20, None),
    ("motion_floor", float, 0.2, None),
    ("static_motion_thresh", float, 1e-3, None),
    ("boundary_views", int, 12, None),
    ("alpha_depth", float, 5.0, None),
    ("beta_rgb", float, 2.0, None),
    ("gamma", float, 2.0, None),
    ("opacity_thresh", float, 0.05, None),
    ("presmooth_sigma", float, 0.0, None),
]


def str_to_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    normalized = value.lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected boolean value, got {value!r}")


def expand_config_dicts(values: dict[str, list], sweep_mode: str) -> list[dict]:
    if sweep_mode == "zip":
        return _expand_zip(values)
    if sweep_mode == "cartesian":
        return _expand_cartesian(values)
    raise ValueError(f"Unknown sweep_mode: {sweep_mode}")


def _expand_zip(values: dict[str, list]) -> list[dict]:
    lengths = {name: len(items) for name, items in values.items()}
    non_singleton_lengths = {length for length in lengths.values() if length != 1}
    if not non_singleton_lengths:
        count = 1
    elif len(non_singleton_lengths) == 1:
        count = non_singleton_lengths.pop()
    else:
        raise ValueError(f"zip sweep values must have the same length: {lengths}")

    return [
        {
            name: items[index] if len(items) > 1 else items[0]
            for name, items in values.items()
        }
        for index in range(count)
    ]


def _expand_cartesian(values: dict[str, list]) -> list[dict]:
    names = list(values)
    return [
        dict(zip(names, combination))
        for combination in itertools.product(*(values[name] for name in names))
    ]


def expand_configs(args: argparse.Namespace) -> list[SpectralParams]:
    sweep_values = {entry[0]: getattr(args, entry[0]) for entry in SWEEP_PARAMS}
    config_dicts = expand_config_dicts(sweep_values, args.sweep_mode)
    return [
        SpectralParams(
            **config,
            use_motion=args.use_motion,
            use_boundary=args.use_boundary,
            solver=args.solver,
            write_annotated_ply=args.write_annotated_ply,
            max_palette_views=args.max_palette_views,
            tc_camera_idx=args.tc_camera_idx,
            tc_n_steps=args.tc_n_steps,
        )
        for config in config_dicts
    ]


def next_output_group_path(outputs_root: str | Path = "outputs") -> Path:
    root = Path(outputs_root)
    index = 1
    while (root / f"multiple-{index}").exists():
        index += 1
    return root / f"multiple-{index}"


def assert_repo_root() -> None:
    if not os.path.exists("train.py"):
        raise SystemExit(
            "Run this script from the repository root, where train.py exists."
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", required=True, help="Single scene name to sweep.")
    parser.add_argument(
        "--dataset",
        choices=sorted(DATASETS),
        default="hypernerf",
        help="Dataset preset: drives dataset_root, Stage 1 iteration, render/eval "
             "camera split, and original-image subdir.",
    )
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
        "--retrain",
        action="store_true",
        help="Force training and republish the scene deform checkpoint on pass 1.",
    )
    parser.add_argument(
        "--sweep_mode",
        choices=["zip", "cartesian"],
        default="zip",
        help="zip pairs same-index values; cartesian runs every value combination.",
    )
    parser.add_argument(
        "--output_group",
        default="",
        help=(
            "Grouped output directory for sweep runs. Defaults to the next "
            "outputs/multiple-N directory."
        ),
    )
    for name, value_type, default, choices in SWEEP_PARAMS:
        kwargs = {
            "type": str_to_bool if value_type is bool else value_type,
            "nargs": "+",
            "default": [default],
        }
        if choices is not None:
            kwargs["choices"] = choices
        parser.add_argument(f"--{name}", **kwargs)

    motion_group = parser.add_mutually_exclusive_group()
    motion_group.add_argument("--use_motion", dest="use_motion", action="store_true")
    motion_group.add_argument("--no_motion", dest="use_motion", action="store_false")
    parser.set_defaults(use_motion=True)

    boundary_group = parser.add_mutually_exclusive_group()
    boundary_group.add_argument("--use_boundary", dest="use_boundary", action="store_true")
    boundary_group.add_argument("--no_boundary", dest="use_boundary", action="store_false")
    parser.set_defaults(use_boundary=True)

    parser.add_argument(
        "--solver",
        type=str,
        default="cupy",
        choices=["lobpcg", "cupy", "randomized", "arpack"],
    )
    parser.add_argument(
        "--write_annotated_ply",
        action="store_true",
        help="Write run-dir point_cloud.ply (~checkpoint size + cls). Default off.",
    )
    parser.add_argument(
        "--max_palette_views",
        type=int,
        default=-1,
        help="Cap the train-camera palette render at N evenly-spaced views. "
        "<0 (default) renders all; 0 skips entirely; N>0 strides through "
        "views (e.g. 100 → ~80 sec). mIoU/mAcc unaffected.",
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
    add_merge_args(parser)
    return parser


def main() -> None:
    assert_repo_root()
    parser = build_parser()
    args = parser.parse_args()

    dataset = DATASETS[args.dataset]
    dataset_root = args.dataset_root or dataset.dataset_root
    iteration = args.iteration if args.iteration is not None else dataset.iteration

    configs = expand_configs(args)
    merge = args_to_merge_params(args)
    failures = []
    output_group = args.output_group or str(next_output_group_path("outputs"))

    print(f"Dataset: {dataset.name} (root={dataset_root}, iter={iteration}, "
          f"test_cams={dataset.use_test_cameras})")
    print(
        f"Running {len(configs)} parameter config(s) on scene {args.scene} "
        f"with sweep_mode={args.sweep_mode}"
    )
    print(f"Grouped outputs will be written under: {output_group}/{args.scene}")
    for index, params in enumerate(configs, start=1):
        print(f"\n{'#' * 72}\n# Pass {index}/{len(configs)}\n{'#' * 72}")
        print(asdict(params))
        try:
            run_pipeline_pass(
                args.scene,
                params,
                iteration=iteration,
                retrain=args.retrain if index == 1 else False,
                dataset_root=dataset_root,
                output_group=output_group,
                dataset=dataset,
                merge=merge,
            )
        except Exception as exc:
            traceback.print_exc()
            failures.append((index, str(exc)))

    if failures:
        print(f"\n{len(failures)} config(s) failed:")
        for index, error in failures:
            print(f"  pass {index}: {error}")
        raise SystemExit(1)

    print("\nSweep complete.")


if __name__ == "__main__":
    main()
