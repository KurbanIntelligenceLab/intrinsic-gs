"""Shared helpers for running full self-supervised segmentation evaluations."""

from __future__ import annotations

import glob
import os
import shlex
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DatasetConfig:
    """Per-dataset defaults that diverge between HyperNeRF and Neu3D.

    The wrappers resolve one of these from `--dataset` and thread it through
    every command builder so render/eval target the correct camera split and
    source-image layout.
    """
    name: str
    dataset_root: str
    iteration: int                       # Stage 1 iteration to load.
    use_test_cameras: bool               # Render & eval against test split (Neu3D).
    original_subdir: str                 # Source-image subdir under each scene dir.
    train_extra_args: tuple[str, ...] = ()  # Extra train.py flags for fresh trainings.


DATASETS: dict[str, DatasetConfig] = {
    "hypernerf": DatasetConfig(
        name="hypernerf",
        dataset_root="/okyanus/users/mtuncel/datasets/hypernerf",
        iteration=20000,
        use_test_cameras=False,
        original_subdir="rgb/2x",
    ),
    "neu3d": DatasetConfig(
        name="neu3d",
        dataset_root="/okyanus/users/mtuncel/datasets/neu3d",
        iteration=30000,
        # GT masks are tied to the held-out cam00 test split (300 frames).
        use_test_cameras=True,
        original_subdir="images_2x",
        # Neu3D multi-view recipe from implementation_docs/COMMANDS.md §2.
        train_extra_args=(
            "--warm_up", "3000",
            "--iterative_opt_interval", "20000",
            "--densify_until_iter", "8000",
            "--lambda_reg_deform", "0",
            "--eval",
            "--load2gpu_on_the_fly",
            "--load_image_on_the_fly",
            "--load_mask_on_the_fly",
            "--num_sampled_pixels", "10000",
            "--num_sampled_masks", "50",
            "--smooth_K", "16",
            "--contrastive_mode", "soft",
            "--test_iterations", "10000", "15000", "20000", "30000",
            "--save_iterations", "10000", "15000", "20000", "30000",
        ),
    ),
}


# Back-compat aliases — older callers (tests, ad-hoc scripts) import these
# directly. They track the HyperNeRF defaults, which was the only supported
# dataset before the registry was introduced.
DEFAULT_DATASET_ROOT = DATASETS["hypernerf"].dataset_root
DEFAULT_ITERATION = DATASETS["hypernerf"].iteration


@dataclass(frozen=True)
class ScenePaths:
    scene: str
    data: str
    model: str


@dataclass(frozen=True)
class MergeParams:
    """Long-range cluster-merge knobs. Forwarded to merge_clusters.py."""
    enabled: bool = False
    w_traj: float = 0.5
    w_feat: float = 0.3
    w_color: float = 0.2
    tau: float = 0.85
    sigma_color: float = 0.5
    n_time_steps: int = 20
    min_cluster_size: int = 10


@dataclass(frozen=True)
class SpectralParams:
    n_clusters: int = 14
    eigengap_k: int = 15
    use_geometry: bool = True
    sigma_color: float = 0.8
    sigma_scale: float = 1.0
    power: float = 2.0
    knn_k: int = 20
    use_motion: bool = True
    n_time_steps: int = 20
    motion_floor: float = 0.2
    static_motion_thresh: float = 1e-3
    use_boundary: bool = True
    boundary_views: int = 12
    alpha_depth: float = 5.0
    beta_rgb: float = 2.0
    gamma: float = 2.0
    opacity_thresh: float = 0.05
    presmooth_sigma: float = 0.0
    solver: str = "cupy"
    clusterer: str = "kmeans"
    leiden_resolution: float = 1.0
    hdbscan_min_cluster_size_frac: float = 0.015
    hdbscan_min_samples: int = 5
    rgb_edge_method: str = "sobel"
    pidinet_variant: str = "full"
    # Run-dir point_cloud.ply is a full checkpoint copy + cls (~0.5GB). Off by default.
    write_annotated_ply: bool = False
    # Cap the train-camera palette render in spectral_cluster.py:
    #   <0 → render all (current behaviour; ~1 hour on Neu3D)
    #    0 → skip entirely (equivalent to --no_render)
    #    N → render N evenly-spaced views (stride sampling)
    # The palette only drives the optional "segmented" column in compute_miou
    # visualizations; mIoU/mAcc themselves don't depend on it.
    max_palette_views: int = -1
    # Temporal-consistency eval. tc_camera_idx < 0 disables the TC sweep + metric.
    tc_camera_idx: int = -1
    tc_n_steps: int = 20


# Output directory prefix per clusterer — used to find the freshly-created
# run dir after spectral_cluster.py exits (`newest_run`) and to detect which
# algorithm produced an existing run.
CLUSTERER_PREFIX = {
    "kmeans": "spectral",
    "leiden": "leiden",
    "hdbscan": "hdbscan",
}


def discover_scenes(dataset_root: str = DEFAULT_DATASET_ROOT) -> list[str]:
    """Return scene directory names under the given dataset root."""
    root = Path(dataset_root)
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset root does not exist: {dataset_root}")
    return sorted(path.name for path in root.iterdir() if path.is_dir())


def scene_paths(scene: str, dataset_root: str = DEFAULT_DATASET_ROOT) -> ScenePaths:
    return ScenePaths(
        scene=scene,
        data=f"{dataset_root.rstrip('/')}/{scene}",
        model=f"output/{scene}_run",
    )


def is_trained(model_path: str, iteration: int = DEFAULT_ITERATION) -> bool:
    ply_path = Path(model_path) / "point_cloud" / f"iteration_{iteration}" / "point_cloud.ply"
    return ply_path.exists()


def deform_published(scene: str) -> bool:
    return (Path("deform") / f"deform_{scene}.pth").exists()


def publish_deform_paths(scene: str, iteration: int = DEFAULT_ITERATION) -> tuple[str, str]:
    src = f"output/{scene}_run/deform/iteration_{iteration}/deform.pth"
    dst = f"deform/deform_{scene}.pth"
    return src, dst


def build_train_cmd(
    paths: ScenePaths,
    iteration: int = DEFAULT_ITERATION,
    extra_args: tuple[str, ...] | list[str] = (),
) -> list[str]:
    cmd = [
        "python",
        "train.py",
        "-s",
        paths.data,
        "--model_path",
        paths.model,
        "--iterations",
        str(iteration),
        "--warm_up_3d_features",
        str(iteration + 1),
    ]
    cmd.extend(extra_args)
    return cmd


def build_spectral_cmd(
    paths: ScenePaths,
    params: SpectralParams,
    iteration: int = DEFAULT_ITERATION,
) -> list[str]:
    cmd = [
        "python",
        "self_supervised_scripts/spectral_cluster.py",
        "-s",
        paths.data,
        "--model_path",
        paths.model,
        "--load_iteration",
        str(iteration),
        "--clusterer",
        params.clusterer,
        "--sigma_color",
        str(params.sigma_color),
        "--sigma_scale",
        str(params.sigma_scale),
        "--power",
        str(params.power),
        "--k",
        str(params.knn_k),
        "--opacity_thresh",
        str(params.opacity_thresh),
        # Keep Camera.original_image on CPU; transfer to GPU lazily on access.
        # Without this, loading 5400 Neu3D cameras blows the 32GB V100 budget
        # (each pushes ~2MB onto GPU at construction → OOM ~camera 2000).
        # Negligible cost for spectral_cluster since only the 12 boundary
        # views actually read original_image.
        "--load2gpu_on_the_fly",
    ]
    # NOTE: do NOT pass --load_image_on_the_fly here. boundary_suppression.py
    # reads view.original_image when use_gt_rgb=True (the default), and with
    # the on-the-fly flag that attribute is None — the boundary term silently
    # falls back to rendered RGB, which shifts the affinity matrix and changes
    # downstream mIoU/mAcc. Keep GT-PNG boundary edges for metric stability.
    # Clusterer-specific args.
    if params.clusterer == "kmeans":
        cmd.extend([
            "--n_clusters", str(params.n_clusters),
            "--eigengap_k", str(params.eigengap_k),
            "--solver", params.solver,
        ])
    elif params.clusterer == "leiden":
        cmd.extend(["--leiden_resolution", str(params.leiden_resolution)])
    elif params.clusterer == "hdbscan":
        cmd.extend([
            "--eigengap_k", str(params.eigengap_k),  # embedding dim for hdbscan
            "--solver", params.solver,
            "--hdbscan_min_cluster_size_frac",
            str(params.hdbscan_min_cluster_size_frac),
            "--hdbscan_min_samples", str(params.hdbscan_min_samples),
        ])
    if not params.use_geometry:
        cmd.append("--no_geo")
    if params.use_motion:
        cmd.extend(
            [
                "--use_motion",
                "--n_time_steps",
                str(params.n_time_steps),
                "--motion_floor",
                str(params.motion_floor),
                "--static_motion_thresh",
                str(params.static_motion_thresh),
            ]
        )
    if params.use_boundary:
        cmd.extend(
            [
                "--use_boundary",
                "--boundary_views",
                str(params.boundary_views),
                "--alpha_depth",
                str(params.alpha_depth),
                "--beta_rgb",
                str(params.beta_rgb),
                "--gamma",
                str(params.gamma),
                "--presmooth_sigma",
                str(params.presmooth_sigma),
                "--rgb_edge_method",
                params.rgb_edge_method,
            ]
        )
        if params.rgb_edge_method == "pidinet":
            cmd.extend(["--pidinet_variant", params.pidinet_variant])
    if not params.write_annotated_ply:
        cmd.append("--no_annotated_ply")
    if params.max_palette_views >= 0:
        cmd.extend(["--max_palette_views", str(params.max_palette_views)])
    return cmd


def newest_spec_run(model_path: str, prefix: str = "spectral") -> str | None:
    """Return the most-recently-modified <model>/<dd_mm>/<prefix>_* dir.

    Default prefix is 'spectral' (kmeans output); pass 'leiden' or 'hdbscan'
    for the corresponding clusterers.
    """
    matches = glob.glob(f"{model_path}/*/{prefix}_*")
    if not matches:
        return None
    return max(matches, key=lambda path: (os.path.getmtime(path), path))


def detect_actual_k(spec_run: str) -> int:
    """Recover the actual cluster count produced by the run.

    Prefers parsing the cheap `renders_k{K}/` directory name; falls back
    to reading labels.npy when no renders dir exists (e.g., --no_render).
    """
    for path in glob.glob(f"{spec_run}/renders_k*"):
        name = os.path.basename(path)
        suffix = name[len("renders_k"):]
        if name.startswith("renders_k") and suffix.isdigit():
            return int(suffix)
    import numpy as np  # local import: not always needed
    labels = np.load(f"{spec_run}/labels.npy")
    return int(labels.max())


def build_render_cmd(
    paths: ScenePaths,
    spec_run: str,
    iteration: int = DEFAULT_ITERATION,
    tc_camera_idx: int = -1,
    tc_n_steps: int = 20,
    use_test_cameras: bool = False,
    labels_filename: str = "labels.npy",
    output_subdir: str = "",
) -> list[str]:
    """Build the render_clusters.py command line.

    `labels_filename` and `output_subdir` cover the long-range-merge second
    pass: pointing at `labels_merged.npy` and writing renders/cluster IDs
    under `<spec_run>/merged/` keeps merged artefacts isolated from the
    baseline outputs.
    """
    output_dir = os.path.join(spec_run, output_subdir) if output_subdir else spec_run
    cmd = [
        "python",
        "self_supervised_scripts/render_clusters.py",
        "-s",
        paths.data,
        "--model_path",
        paths.model,
        "--load_iteration",
        str(iteration),
        "--labels_file",
        f"{spec_run}/{labels_filename}",
        "--output_dir",
        output_dir,
        "--save_cluster_ids",
        # Renderer regenerates pixels from the Gaussian field — it never reads
        # cam.image, so the source PNG decode in the camera loader is wasted
        # work. Defer image/mask loads to keep cam loading at seconds, not minutes.
        "--load_image_on_the_fly",
        "--load_mask_on_the_fly",
        # Keep Camera tensors on CPU; cheap to move per-view at render time.
        # Prevents the Neu3D 32GB-V100 OOM seen in spectral_cluster.
        "--load2gpu_on_the_fly",
    ]
    if use_test_cameras:
        cmd.append("--use_test_cameras")
        # readMultiViewInfo merges test→train when eval=False (Neu3D loader);
        # scene.getTestCameras() then returns []. --eval keeps the test split
        # populated so --use_test_cameras has something to iterate over.
        cmd.append("--eval")
    if tc_camera_idx >= 0:
        cmd.extend([
            "--tc_camera_idx", str(tc_camera_idx),
            "--tc_n_steps", str(tc_n_steps),
        ])
    return cmd


def build_miou_cmd(
    paths: ScenePaths,
    spec_run: str,
    n_clusters: int,
    mode: str = "best_cluster",
    tc_camera_idx: int = -1,
    gt_subdir: str = "gt_masks",
    use_test_cameras: bool = False,
    original_subdir: str = "rgb/2x",
    pred_subdir: str = "",
    result_tag: str = "",
    palette_subdir: str = "",
    palette_suffix: str = "",
) -> list[str]:
    """Build the compute_miou.py command line.

    `pred_subdir`, `result_tag`, `palette_subdir`, and `palette_suffix` are
    set for the long-range-merge pass so cluster IDs are read from
    `<spec_run>/merged/cluster_ids_*` and outputs land beside the baseline
    JSONs with a `_merged` (or `_merged_greedy`) suffix.
    """
    if mode not in {"best_cluster", "greedy_union"}:
        raise ValueError(f"Unknown mIoU selection mode: {mode}")
    suffix = "_greedy" if mode == "greedy_union" else ""
    # render_clusters.py writes cluster IDs under cluster_ids_<split>.
    pred_split = "test" if use_test_cameras else "train"
    pred_root = os.path.join(spec_run, pred_subdir) if pred_subdir else spec_run
    palette_root = os.path.join(spec_run, palette_subdir) if palette_subdir else spec_run
    tag = result_tag  # e.g. "_merged"
    cmd = [
        "python",
        "self_supervised_scripts/compute_miou.py",
        "--pred_dir",
        f"{pred_root}/cluster_ids_{pred_split}",
        "--gt_dir",
        f"{paths.data}/{gt_subdir}",
        "--report_md",
        f"{spec_run}/report.md",
        "--output_json",
        f"{spec_run}/miou_results{tag}{suffix}.json",
        "--vis_dir",
        f"{spec_run}/visualizations{tag}{suffix}",
        "--palette_dir",
        f"{palette_root}/renders_k{n_clusters}{palette_suffix}",
        "--original_dir",
        f"{paths.data}/{original_subdir}",
    ]
    if mode == "greedy_union":
        cmd.extend(["--selection_mode", "greedy_union"])
    if tc_camera_idx >= 0:
        cmd.extend(["--tc_dir", f"{pred_root}/cluster_ids_tc_v{tc_camera_idx}"])
    return cmd


def build_macc_cmd(
    paths: ScenePaths,
    spec_run: str,
    mode: str = "best_cluster",
    gt_subdir: str = "gt_masks",
    use_test_cameras: bool = False,
    pred_subdir: str = "",
    result_tag: str = "",
) -> list[str]:
    if mode not in {"best_cluster", "greedy_union"}:
        raise ValueError(f"Unknown mAcc selection mode: {mode}")
    suffix = "_greedy" if mode == "greedy_union" else ""
    pred_split = "test" if use_test_cameras else "train"
    pred_root = os.path.join(spec_run, pred_subdir) if pred_subdir else spec_run
    tag = result_tag
    return [
        "python",
        "self_supervised_scripts/compute_macc.py",
        "--pred_dir",
        f"{pred_root}/cluster_ids_{pred_split}",
        "--gt_dir",
        f"{paths.data}/{gt_subdir}",
        "--miou_json",
        f"{spec_run}/miou_results{tag}{suffix}.json",
        "--output_json",
        f"{spec_run}/macc_results{tag}{suffix}.json",
    ]


def build_merge_cmd(
    paths: ScenePaths,
    spec_run: str,
    merge: MergeParams,
    iteration: int = DEFAULT_ITERATION,
) -> list[str]:
    return [
        "python",
        "self_supervised_scripts/merge_clusters.py",
        "-s", paths.data,
        "--model_path", paths.model,
        "--load_iteration", str(iteration),
        "--labels_file", f"{spec_run}/labels.npy",
        "--output_dir", spec_run,
        "--n_time_steps", str(merge.n_time_steps),
        "--w_traj", str(merge.w_traj),
        "--w_feat", str(merge.w_feat),
        "--w_color", str(merge.w_color),
        "--tau", str(merge.tau),
        "--sigma_color", str(merge.sigma_color),
        "--min_cluster_size", str(merge.min_cluster_size),
    ]


def _run_cmd(cmd: list[str]) -> None:
    # shlex.join is Python 3.8+; the server env appears to be older. Inline the
    # equivalent (shell-escape each arg, space-separate) for back-compat.
    print(" ".join(shlex.quote(c) for c in cmd))
    subprocess.run(cmd, check=True)


def run_train(
    paths: ScenePaths,
    iteration: int = DEFAULT_ITERATION,
    extra_args: tuple[str, ...] | list[str] = (),
) -> None:
    print(f"\n=== Train: {paths.scene} ===")
    _run_cmd(build_train_cmd(paths, iteration, extra_args=extra_args))


def publish_deform(scene: str, iteration: int = DEFAULT_ITERATION) -> None:
    src, dst = publish_deform_paths(scene, iteration)
    if not Path(src).exists():
        raise FileNotFoundError(f"Deform checkpoint not found: {src}")
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(src, dst)
    print(f"Published deform checkpoint: {dst}")


def run_spectral(
    paths: ScenePaths,
    params: SpectralParams,
    iteration: int = DEFAULT_ITERATION,
) -> str:
    print(f"\n=== {params.clusterer} clustering: {paths.scene} ===")
    _run_cmd(build_spectral_cmd(paths, params, iteration))
    prefix = CLUSTERER_PREFIX.get(params.clusterer, "spectral")
    spec_run = newest_spec_run(paths.model, prefix=prefix)
    if not spec_run:
        raise RuntimeError(
            f"No {prefix}_* run directory found under {paths.model}"
        )
    print(f"SPEC_RUN={spec_run}")
    return spec_run


def run_render(
    paths: ScenePaths,
    spec_run: str,
    iteration: int = DEFAULT_ITERATION,
    tc_camera_idx: int = -1,
    tc_n_steps: int = 20,
    use_test_cameras: bool = False,
    labels_filename: str = "labels.npy",
    output_subdir: str = "",
) -> None:
    print(f"\n=== Render cluster IDs: {paths.scene}"
          f"{' [' + output_subdir + ']' if output_subdir else ''} ===")
    _run_cmd(build_render_cmd(
        paths, spec_run, iteration,
        tc_camera_idx=tc_camera_idx, tc_n_steps=tc_n_steps,
        use_test_cameras=use_test_cameras,
        labels_filename=labels_filename,
        output_subdir=output_subdir,
    ))


def run_merge(
    paths: ScenePaths,
    spec_run: str,
    merge: MergeParams,
    iteration: int = DEFAULT_ITERATION,
) -> int:
    """Run merge_clusters.py and return K' (the merged cluster count)."""
    import numpy as np  # local import — only needed in the merge path
    print(f"\n=== Long-range cluster merge: {paths.scene} ===")
    _run_cmd(build_merge_cmd(paths, spec_run, merge, iteration))
    merged_path = Path(spec_run) / "labels_merged.npy"
    if not merged_path.exists():
        raise RuntimeError(
            f"merge_clusters.py exited successfully but {merged_path} is missing."
        )
    merged = np.load(merged_path)
    k_prime = int(merged.max())
    print(f"Merged cluster count K'={k_prime}")
    return k_prime


def run_miou(
    paths: ScenePaths,
    spec_run: str,
    n_clusters: int,
    mode: str,
    tc_camera_idx: int = -1,
    gt_subdir: str = "gt_masks",
    use_test_cameras: bool = False,
    original_subdir: str = "rgb/2x",
    pred_subdir: str = "",
    result_tag: str = "",
    palette_subdir: str = "",
    palette_suffix: str = "",
) -> None:
    label = f"{mode}{(' ' + result_tag) if result_tag else ''}"
    print(f"\n=== mIoU {label}: {paths.scene} ===")
    _run_cmd(build_miou_cmd(
        paths, spec_run, n_clusters, mode,
        tc_camera_idx=tc_camera_idx, gt_subdir=gt_subdir,
        use_test_cameras=use_test_cameras,
        original_subdir=original_subdir,
        pred_subdir=pred_subdir, result_tag=result_tag,
        palette_subdir=palette_subdir, palette_suffix=palette_suffix,
    ))


def run_macc(
    paths: ScenePaths,
    spec_run: str,
    mode: str,
    gt_subdir: str = "gt_masks",
    use_test_cameras: bool = False,
    pred_subdir: str = "",
    result_tag: str = "",
) -> None:
    label = f"{mode}{(' ' + result_tag) if result_tag else ''}"
    print(f"\n=== mAcc {label}: {paths.scene} ===")
    _run_cmd(build_macc_cmd(
        paths, spec_run, mode,
        gt_subdir=gt_subdir, use_test_cameras=use_test_cameras,
        pred_subdir=pred_subdir, result_tag=result_tag,
    ))


def relocate_spec_run(spec_run: str, output_group: str, scene: str) -> str:
    """Move one spectral run under <output_group>/<scene>/ and return its new path."""
    src = Path(spec_run)
    dst = Path(output_group) / scene / src.name
    if dst.exists():
        raise FileExistsError(f"Grouped output already exists: {dst}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(src), str(dst))
    relocated = str(dst)
    print(f"Relocated SPEC_RUN={relocated}")
    return relocated


def run_pipeline_pass(
    scene: str,
    params: SpectralParams,
    iteration: int = DEFAULT_ITERATION,
    retrain: bool = False,
    dataset_root: str = DEFAULT_DATASET_ROOT,
    output_group: str | None = None,
    gt_subdir: str = "gt_masks",
    dataset: DatasetConfig | None = None,
    merge: MergeParams | None = None,
) -> str:
    """Run train, spectral clustering, cluster-ID render, and both mIoU modes.

    When `dataset` is provided it supplies the render/eval split and source
    image layout (HyperNeRF uses train cameras + rgb/2x, Neu3D uses test
    cameras + images_2x). Omitting it keeps the legacy HyperNeRF defaults.
    """
    paths = scene_paths(scene, dataset_root)
    use_test_cameras = dataset.use_test_cameras if dataset else False
    original_subdir = dataset.original_subdir if dataset else "rgb/2x"
    train_extra_args = dataset.train_extra_args if dataset else ()

    if retrain or not is_trained(paths.model, iteration):
        run_train(paths, iteration, extra_args=train_extra_args)
    else:
        print(f"\n[skip train] {paths.model} already has iteration_{iteration}")

    if retrain or not deform_published(scene):
        publish_deform(scene, iteration)
    else:
        print(f"[skip deform] deform/deform_{scene}.pth exists")

    spec_run = run_spectral(paths, params, iteration)
    if output_group:
        spec_run = relocate_spec_run(spec_run, output_group, scene)
    run_render(
        paths, spec_run, iteration,
        tc_camera_idx=params.tc_camera_idx,
        tc_n_steps=params.tc_n_steps,
        use_test_cameras=use_test_cameras,
    )
    # The actual k may differ from params.n_clusters for auto-k clusterers
    # (leiden, hdbscan); detect it from the run dir before mIoU eval.
    actual_k = detect_actual_k(spec_run)
    run_miou(paths, spec_run, actual_k, "best_cluster",
             tc_camera_idx=params.tc_camera_idx, gt_subdir=gt_subdir,
             use_test_cameras=use_test_cameras,
             original_subdir=original_subdir)
    run_macc(paths, spec_run, "best_cluster",
             gt_subdir=gt_subdir, use_test_cameras=use_test_cameras)
    run_miou(paths, spec_run, actual_k, "greedy_union",
             tc_camera_idx=params.tc_camera_idx, gt_subdir=gt_subdir,
             use_test_cameras=use_test_cameras,
             original_subdir=original_subdir)
    run_macc(paths, spec_run, "greedy_union",
             gt_subdir=gt_subdir, use_test_cameras=use_test_cameras)

    if merge is not None and merge.enabled:
        k_prime = run_merge(paths, spec_run, merge, iteration)
        # Run the same render + mIoU/mAcc grid against labels_merged, scoped
        # to <spec_run>/merged/ so baseline artefacts stay untouched. Skip
        # the second pass when merging didn't change the partition — the
        # baseline JSONs already cover that case.
        if k_prime == actual_k:
            print(f"[skip merge eval] K'={k_prime} == K={actual_k}; merge fired no edges.")
        else:
            run_render(
                paths, spec_run, iteration,
                tc_camera_idx=params.tc_camera_idx,
                tc_n_steps=params.tc_n_steps,
                use_test_cameras=use_test_cameras,
                labels_filename="labels_merged.npy",
                output_subdir="merged",
            )
            for mode in ("best_cluster", "greedy_union"):
                run_miou(paths, spec_run, k_prime, mode,
                         tc_camera_idx=params.tc_camera_idx, gt_subdir=gt_subdir,
                         use_test_cameras=use_test_cameras,
                         original_subdir=original_subdir,
                         pred_subdir="merged", result_tag="_merged",
                         palette_subdir="merged",
                         palette_suffix=f"_loaded_{'test' if use_test_cameras else 'train'}")
                run_macc(paths, spec_run, mode,
                         gt_subdir=gt_subdir, use_test_cameras=use_test_cameras,
                         pred_subdir="merged", result_tag="_merged")
    return spec_run
