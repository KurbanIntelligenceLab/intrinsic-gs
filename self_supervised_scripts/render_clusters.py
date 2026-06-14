"""
Render learned feature clusters onto the original scene.

Runs K-means on the saved Gaussian features, assigns cluster colors,
and renders all training views using the existing Gaussian renderer.

Usage:
    python self_supervised_scripts/render_clusters.py \
        -s data/HyperNeRF/americano \
        --model_path output/8abe732a-1 \
        --deform_path /okyanus/users/mtuncel/TRASE \
        --load_iteration 20000 \
        --n_clusters 8

Evaluation mode (K-pass binary rendering for mIoU):
    python self_supervised_scripts/render_clusters.py \
        -s data/HyperNeRF/chicken \
        --model_path output/<scene_run> \
        --load_iteration 20000 \
        --labels_file outputs/spectral_*/labels.npy \
        --output_dir outputs/spectral_*/ \
        --use_test_cameras \
        --save_cluster_ids
"""

import os
import sys
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
import torch
import numpy as np
import torchvision
from PIL import Image
from argparse import ArgumentParser
from tqdm import tqdm
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import normalize

from scene import Scene, GaussianModel, DeformModel
from arguments import ModelParams, PipelineParams, OptimizationParams
from utils.general_utils import safe_state
from gaussian_renderer import render
from self_supervised_scripts.render_cluster_io import feature_checkpoint_path
from self_supervised_scripts.timing import TimingRecorder


CLUSTER_PALETTE = torch.tensor([
    [230,  25,  75], [60,  180,  75], [ 67,  99, 216], [255, 225,  25],
    [245, 130,  49], [145,  30, 180], [ 66, 212, 244], [240,  50, 230],
    [188, 246,  12], [250, 190, 212], [  0, 128, 128], [220, 190, 255],
    [154,  99,  36], [255, 250, 200], [128,   0,   0], [170, 255, 195],
], dtype=torch.float32) / 255.0  # [P, 3]


@torch.no_grad()
def main(dataset, opt, pipe, args):
    # ------------------------------------------------------------------
    # Load Stage 1 checkpoint
    # ------------------------------------------------------------------
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, load_iteration=args.load_iteration)

    deform = DeformModel(
        is_blender=dataset.is_blender,
        is_6dof=dataset.is_6dof,
        model_type=opt.deform_type,
    )
    scene_name  = os.path.basename(os.path.normpath(dataset.source_path))
    deform_path = args.deform_path if args.deform_path else PROJECT_ROOT
    deform.load_weights(deform_path, iteration=args.load_iteration, scene_name=scene_name)

    feat_ply = feature_checkpoint_path(
        dataset.model_path,
        args.load_iteration,
        run_name=args.run_name,
        labels_file=args.labels_file,
    )
    if feat_ply:
        print(f"Loading features from: {feat_ply}")

        # Re-load gaussian features from the feature .ply for internal clustering.
        gaussians.load_ply(feat_ply)
        feats = gaussians.get_gaussian_features.squeeze(1)  # [N, 32]
        print(f"Features: {feats.shape}")
    else:
        feats = None

    # ------------------------------------------------------------------
    # Clustering — load pre-computed labels OR run k-means/DBSCAN
    # ------------------------------------------------------------------
    if args.labels_file:
        print(f"Loading pre-computed labels from: {args.labels_file}")
        labels = np.load(args.labels_file).astype(np.int64)
        n_gaussians = gaussians.get_xyz.shape[0]
        if labels.shape[0] != n_gaussians:
            raise ValueError(
                f"labels.npy length ({labels.shape[0]}) does not match Gaussian "
                f"count ({n_gaussians}). Wrong run or wrong checkpoint?"
            )
        valid = labels[labels > 0]
        n_loaded_clusters = int(valid.max()) if valid.size else 0
        n_filtered = int((labels == 0).sum())
        counts = np.bincount(valid, minlength=n_loaded_clusters + 1)[1:] if n_loaded_clusters else np.array([])
        print(f"Loaded {labels.shape[0]} labels: {n_loaded_clusters} clusters, "
              f"{n_filtered} filtered (label=0)")
        print(f"Cluster sizes: {sorted(counts.tolist(), reverse=True)}")
        out_suffix = f"k{n_loaded_clusters}_loaded"
    else:
        f_np = normalize(feats.cpu().numpy(), norm='l2')

        if args.cluster_method == 'dbscan':
            from sklearn.neighbors import NearestNeighbors
            N = len(f_np)
            # Subsample for DBSCAN — 1.2M points is too large for exact DBSCAN
            if N > args.dbscan_subsample:
                print(f"Subsampling {N} → {args.dbscan_subsample} points for DBSCAN...")
                idx = np.random.choice(N, args.dbscan_subsample, replace=False)
                f_sub = f_np[idx]
            else:
                idx = np.arange(N)
                f_sub = f_np

            print(f"Running DBSCAN (eps={args.dbscan_eps}, min_samples={args.dbscan_min_samples}) on {len(f_sub)} points...")
            db = DBSCAN(eps=args.dbscan_eps, min_samples=args.dbscan_min_samples, metric='euclidean', n_jobs=-1)
            sub_labels = db.fit_predict(f_sub)
            n_clusters = len(set(sub_labels)) - (1 if -1 in sub_labels else 0)
            n_noise = np.sum(sub_labels == -1)
            print(f"Found {n_clusters} clusters, {n_noise} noise points ({100*n_noise/len(sub_labels):.1f}%)")

            # Assign all Gaussians to nearest subsampled point's label
            if N > args.dbscan_subsample:
                print("Assigning all Gaussians via nearest neighbor...")
                nn = NearestNeighbors(n_neighbors=1, metric='euclidean', n_jobs=-1)
                nn.fit(f_sub)
                _, nn_idx = nn.kneighbors(f_np)
                labels = sub_labels[nn_idx[:, 0]]
            else:
                labels = sub_labels

            counts = np.bincount(labels[labels >= 0], minlength=max(n_clusters, 1))
            print(f"Cluster sizes: {sorted(counts, reverse=True)}")
            out_suffix = f"dbscan_eps{args.dbscan_eps}_min{args.dbscan_min_samples}"
        else:
            print(f"Running K-means (k={args.n_clusters})...")
            km = KMeans(n_clusters=args.n_clusters, random_state=0, n_init='auto')
            labels = km.fit_predict(f_np)
            counts = np.bincount(labels, minlength=args.n_clusters)
            print(f"Cluster sizes: {sorted(counts, reverse=True)}")
            out_suffix = f"k{args.n_clusters}"

    # Normalize to "0 = no cluster, 1..K = valid clusters" convention.
    # labels_file already follows this. Internal k-means is 0..K-1 → shift by +1.
    # DBSCAN noise (-1) → 0 (treated as no cluster).
    if not args.labels_file:
        labels = labels.copy()
        labels[labels < 0] = -1            # mark noise
        labels = labels + 1                # shift: -1→0 (noise), 0..K-1 → 1..K

    # Assign cluster colors — label=0 → black (no-cluster / filtered / noise)
    filtered_color = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32)
    colors_list = []
    for lbl in labels:
        if lbl == 0:
            colors_list.append(filtered_color)
        else:
            colors_list.append(CLUSTER_PALETTE[(lbl - 1) % len(CLUSTER_PALETTE)])
    cluster_colors = torch.stack(colors_list).cuda()  # [N, 3]

    # Cluster IDs to render as binary masks (skip 0 = no-cluster)
    unique_labels = sorted(int(x) for x in set(labels.tolist()) if x > 0)
    K_render = len(unique_labels)

    # ------------------------------------------------------------------
    # Render
    # ------------------------------------------------------------------
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    # Camera split — train (default) or test
    if args.use_test_cameras:
        views = scene.getTestCameras()
        split_tag = "test"
    else:
        views = scene.getTrainCameras()
        split_tag = "train"

    if not views:
        raise RuntimeError(
            f"No {split_tag} cameras found in scene. "
            f"Try the other split via --use_test_cameras."
        )

    # Output directory: --output_dir overrides the default model-relative location
    if args.output_dir:
        base_out = args.output_dir
    else:
        base_out = os.path.join(dataset.model_path, "cluster_renders",
                                f"iteration_{args.load_iteration}_{out_suffix}")

    palette_dir = os.path.join(base_out, f"renders_{out_suffix}_{split_tag}")
    os.makedirs(palette_dir, exist_ok=True)
    print(f"Rendering {len(views)} {split_tag} views to: {palette_dir}\n")

    if args.save_cluster_ids:
        ids_dir = os.path.join(base_out, f"cluster_ids_{split_tag}")
        os.makedirs(ids_dir, exist_ok=True)
        print(f"Cluster ID maps will be written to: {ids_dir}")
        print(f"K-pass binary rendering over {K_render} clusters: {unique_labels}\n")

    # Pre-build per-cluster binary color tensors (white for cluster k, black otherwise)
    binary_color_cache = {}
    if args.save_cluster_ids:
        labels_np = labels  # already numpy int
        for k in unique_labels:
            mask = (labels_np == k)
            colors = np.zeros((labels_np.shape[0], 3), dtype=np.float32)
            colors[mask] = 1.0
            binary_color_cache[k] = torch.from_numpy(colors).cuda()

    timer = TimingRecorder(n_valid_gaussians=int((labels > 0).sum()))

    with timer.stage("render_main"):
        for idx, view in enumerate(tqdm(views, desc="Rendering")):
            xyz = gaussians.get_xyz
            # view.fid lives on CPU when --load2gpu_on_the_fly is set; the
            # deform MLP runs on GPU, so align explicitly.
            fid = view.fid.to(xyz.device)
            time_input = fid.unsqueeze(0).expand(xyz.shape[0], -1)

            d_xyz, d_rotation, d_scaling = deform.step(
                xyz.detach(), time_input
            ) if opt.deform_type == 'DeformNetwork' else deform.step(
                xyz.detach(), time_input, gaussians.get_gaussian_features.squeeze(1)
            )

            # Palette render (for visual inspection)
            result = render(view, gaussians, pipe, background,
                            d_xyz, d_rotation, d_scaling,
                            is_6dof=dataset.is_6dof,
                            override_color=cluster_colors)

            img_name = os.path.splitext(view.image_name)[0]
            torchvision.utils.save_image(
                result["render"].cpu(),
                os.path.join(palette_dir, f"{img_name}.png")
            )

            # K-pass binary rendering → per-pixel cluster ID map
            if args.save_cluster_ids:
                H, W = result["render"].shape[-2], result["render"].shape[-1]
                response = torch.zeros((K_render, H, W), dtype=torch.float32, device="cuda")
                for ki, k in enumerate(unique_labels):
                    bin_result = render(view, gaussians, pipe, background,
                                        d_xyz, d_rotation, d_scaling,
                                        is_6dof=dataset.is_6dof,
                                        override_color=binary_color_cache[k])
                    # All channels equal for binary input → take channel 0
                    response[ki] = bin_result["render"][0]

                # argmax over K → cluster index in [0..K-1], map back to actual label
                best_ki = response.argmax(dim=0)                  # [H, W]
                best_val = response.max(dim=0).values             # [H, W]
                id_map_np = np.zeros((H, W), dtype=np.uint8)
                label_lookup = np.array(unique_labels, dtype=np.uint8)
                valid_pixel = (best_val > 0.5).cpu().numpy()
                id_map_np[valid_pixel] = label_lookup[best_ki.cpu().numpy()[valid_pixel]]
                Image.fromarray(id_map_np, mode='L').save(
                    os.path.join(ids_dir, f"{img_name}.png")
                )

    print(f"\nDone. Palette images saved to: {palette_dir}")
    if args.save_cluster_ids:
        print(f"      Cluster ID maps saved to: {ids_dir}")

    # ------------------------------------------------------------------
    # Temporal consistency sweep (optional)
    # ------------------------------------------------------------------
    # Fix a single camera pose and render cluster-ID maps at uniformly
    # spaced times t ∈ [0, 1]. Decouples deformation flicker from camera
    # motion so the TC metric measures only label stability over time.
    if args.tc_camera_idx >= 0 and args.save_cluster_ids:
        if args.tc_camera_idx >= len(views):
            raise ValueError(
                f"--tc_camera_idx={args.tc_camera_idx} out of range "
                f"(0..{len(views) - 1})"
            )
        tc_view = views[args.tc_camera_idx]
        tc_dir = os.path.join(base_out, f"cluster_ids_tc_v{args.tc_camera_idx}")
        os.makedirs(tc_dir, exist_ok=True)
        print(f"\nTemporal sweep: camera idx={args.tc_camera_idx}, "
              f"T={args.tc_n_steps} steps, → {tc_dir}")

        xyz = gaussians.get_xyz
        N = xyz.shape[0]
        times = torch.linspace(0.0, 1.0, args.tc_n_steps, device="cuda")

        with timer.stage("render_tc_sweep"):
            for ti, t_val in enumerate(tqdm(times, desc="TC sweep")):
                time_input = t_val.view(1, 1).expand(N, 1)
                d_xyz, d_rotation, d_scaling = deform.step(
                    xyz.detach(), time_input
                ) if opt.deform_type == 'DeformNetwork' else deform.step(
                    xyz.detach(), time_input, gaussians.get_gaussian_features.squeeze(1)
                )

                # K-pass binary rendering at this fixed pose, swept time.
                ref = render(tc_view, gaussians, pipe, background,
                             d_xyz, d_rotation, d_scaling,
                             is_6dof=dataset.is_6dof,
                             override_color=cluster_colors)
                H, W = ref["render"].shape[-2], ref["render"].shape[-1]
                response = torch.zeros((K_render, H, W), dtype=torch.float32, device="cuda")
                for ki, k in enumerate(unique_labels):
                    bin_result = render(tc_view, gaussians, pipe, background,
                                        d_xyz, d_rotation, d_scaling,
                                        is_6dof=dataset.is_6dof,
                                        override_color=binary_color_cache[k])
                    response[ki] = bin_result["render"][0]

                best_ki  = response.argmax(dim=0)
                best_val = response.max(dim=0).values
                id_map_np = np.zeros((H, W), dtype=np.uint8)
                label_lookup = np.array(unique_labels, dtype=np.uint8)
                valid_pixel = (best_val > 0.5).cpu().numpy()
                id_map_np[valid_pixel] = label_lookup[best_ki.cpu().numpy()[valid_pixel]]
                Image.fromarray(id_map_np, mode='L').save(
                    os.path.join(tc_dir, f"{ti:04d}.png")
                )

        print(f"      Temporal sweep saved to: {tc_dir}")

    # ------------------------------------------------------------------
    # Persist render-stage timings
    # ------------------------------------------------------------------
    # output_dir takes precedence; otherwise the run dir lives under model_path
    # alongside the spectral run's report.md / timings.json.
    timings_target = args.output_dir if args.output_dir else base_out
    if os.path.isdir(timings_target):
        timer.append_to_report_md(
            os.path.join(timings_target, "report.md"), step="render"
        )
        timer.merge_into_json(
            os.path.join(timings_target, "timings.json"), step="render"
        )


if __name__ == "__main__":
    parser = ArgumentParser()
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)

    parser.add_argument("--load_iteration", type=int, default=20000)
    parser.add_argument("--deform_path", type=str, default="")
    parser.add_argument("--n_clusters", type=int, default=8)
    parser.add_argument("--cluster_method", type=str, default="kmeans", choices=["kmeans", "dbscan"])
    parser.add_argument("--dbscan_eps", type=float, default=0.3)
    parser.add_argument("--dbscan_min_samples", type=int, default=10)
    parser.add_argument("--dbscan_subsample", type=int, default=100000,
                        help="Max points to run DBSCAN on; rest assigned via nearest neighbor")
    parser.add_argument("--run_name", type=str, default="",
                        help="Must match the --run_name used during training")

    # Evaluation-mode flags
    parser.add_argument("--labels_file", type=str, default="",
                        help="Path to a pre-computed labels.npy from spectral_cluster.py. "
                             "If set, skips internal clustering and uses these labels. "
                             "Convention: 0 = filtered, 1..K = clusters.")
    parser.add_argument("--use_test_cameras", action="store_true",
                        help="Render test cameras instead of train cameras.")
    parser.add_argument("--save_cluster_ids", action="store_true",
                        help="Run K-pass binary rendering to also save per-pixel "
                             "cluster ID maps (uint8 PNG, value = cluster label).")
    parser.add_argument("--output_dir", type=str, default="",
                        help="Override output directory. If unset, falls back to "
                             "<model_path>/cluster_renders/iteration_<iter>_<suffix>/.")

    # Temporal-consistency sweep (optional)
    parser.add_argument("--tc_camera_idx", type=int, default=-1,
                        help="If ≥0, render an extra temporal sweep using this "
                             "view's camera pose at uniform times t∈[0,1]. "
                             "Requires --save_cluster_ids. Output: "
                             "<base_out>/cluster_ids_tc_v<idx>/.")
    parser.add_argument("--tc_n_steps", type=int, default=20,
                        help="Number of time samples in the temporal sweep.")

    args = parser.parse_args(sys.argv[1:])
    safe_state(args.quiet if hasattr(args, 'quiet') else False)

    main(lp.extract(args), op.extract(args), pp.extract(args), args)
