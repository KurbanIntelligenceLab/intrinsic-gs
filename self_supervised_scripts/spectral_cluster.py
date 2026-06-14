"""
Spectral clustering on the Gaussian affinity graph (Phase 1 / Mode A).

Pipeline:
  1. Build Ageo affinity graph (reuses AffinityGraph)
  2. Symmetrize the directed k-NN graph
  3. Compute normalized graph Laplacian
  4. Extract bottom-k eigenvectors via ARPACK (sparse, CPU)
  5. K-means on eigenvector embedding → cluster labels
  6. Save labels as .npy and annotate the .ply checkpoint
  7. Render all training views with cluster colours

Usage:
    python self_supervised_scripts/spectral_cluster.py \\
        -s data/HyperNeRF/americano \\
        --model_path output/8abe732a-1 \\
        --load_iteration 20000 \\
        --n_clusters 5 \\
        --sigma_pos 0.0036 \\
        --sigma_color 0.5160
"""

import os
import sys
from datetime import datetime
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from tqdm import tqdm
import torchvision
from plyfile import PlyData, PlyElement

from scene import Scene, GaussianModel, DeformModel
from arguments import ModelParams, OptimizationParams, PipelineParams
from utils.general_utils import safe_state
from gaussian_renderer import render

from self_supervised_scripts.affinity_graph import AffinityGraph
from self_supervised_scripts.boundary_suppression import BoundarySuppression
from self_supervised_scripts.spectral_solver import (
    symmetrize,
    normalized_laplacian,
)
from self_supervised_scripts.clusterers import make_clusterer
from self_supervised_scripts.rgb_edge import make_edge_method
from self_supervised_scripts.timing import TimingRecorder


CLUSTER_PALETTE = torch.tensor([
    [  0,   0,   0],  # index 0 — filtered-out / invalid Gaussians (black = invisible)
    [230,  25,  75], [ 60, 180,  75], [ 67,  99, 216], [255, 225,  25],
    [245, 130,  49], [145,  30, 180], [ 66, 212, 244], [240,  50, 230],
    [188, 246,  12], [250, 190, 212], [  0, 128, 128], [220, 190, 255],
    [154,  99,  36], [255, 250, 200], [128,   0,   0], [170, 255, 195],
], dtype=torch.float32) / 255.0  # [P, 3]  — index 0 reserved for invalid


# ── I/O ───────────────────────────────────────────────────────────────────────

def save_run_report(out_dir, args, graph_stats, algo_stats, clusterer):
    """
    Write a human-readable run summary to <out_dir>/report.md.

    graph_stats: dict with keys: n_total, n_valid, n_edges, nnz
    algo_stats:  dict returned by clusterer.fit(); must contain 'cluster_sizes'
                 and 'k_used'. KMeans path additionally has 'spectral'.
    clusterer:   Clusterer instance — supplies the algorithm-specific
                 section via clusterer.report_section(algo_stats, args).
    """
    import shutil
    os.makedirs(out_dir, exist_ok=True)

    cluster_sizes = algo_stats['cluster_sizes']

    # Eigengap diagnostic plot is a kmeans-only artifact.
    if clusterer.name == 'kmeans':
        tmp_plot = '/tmp/eigengap.png'
        if os.path.exists(tmp_plot):
            shutil.copy(tmp_plot, os.path.join(out_dir, 'eigengap.png'))

    # Solver line — kmeans/hdbscan use the spectral solver; leiden doesn't.
    solver_str = algo_stats.get('spectral', {}).get('solver', '—')

    lines = [
        "# Spectral Clustering Run Report",
        "",
        "## Parameters",
        f"| param         | value |",
        f"|---------------|-------|",
        f"| n_clusters    | {args.n_clusters} |",
        f"| k (kNN)       | {args.k} |",
        f"| sigma_color   | {args.sigma_color} |",
        f"| sigma_scale   | {args.sigma_scale} |",
        f"| sigma_pos     | {args.sigma_pos} |",
        f"| power         | {args.power} |",
        f"| opacity_thresh| {args.opacity_thresh} |",
        f"| use_geometry  | {args.use_geometry} |",
        f"| use_motion    | {args.use_motion} |",
        f"| n_time_steps  | {args.n_time_steps if args.use_motion else '—'} |",
        f"| static_thresh | {args.static_motion_thresh if args.use_motion else '—'} |",
        f"| motion_floor  | {args.motion_floor if args.use_motion else '—'} |",
        f"| use_boundary  | {args.use_boundary} |",
        f"| boundary_views| {args.boundary_views if args.use_boundary else '—'} |",
        f"| alpha_depth   | {args.alpha_depth if args.use_boundary else '—'} |",
        f"| beta_rgb      | {args.beta_rgb if args.use_boundary else '—'} |",
        f"| gamma         | {args.gamma if args.use_boundary else '—'} |",
        f"| presmooth_σ   | {args.presmooth_sigma if args.use_boundary else '—'} |",
        f"| rgb_edge_method | {args.rgb_edge_method if args.use_boundary else '—'} |",
        f"| pidinet_variant | {args.pidinet_variant if (args.use_boundary and args.rgb_edge_method == 'pidinet') else '—'} |",
        f"| pidinet_bin_thr | {args.pidinet_binarize_threshold if (args.use_boundary and args.rgb_edge_method == 'pidinet' and args.pidinet_binarize_threshold is not None) else '—'} |",
        f"| solver        | {solver_str} |",
        f"| load_iteration| {args.load_iteration} |",
        "",
        "## Graph Stats",
        f"| stat           | value |",
        f"|----------------|-------|",
        f"| total gaussians| {graph_stats['n_total']:,} |",
        f"| valid (opacity)| {graph_stats['n_valid']:,} ({100*graph_stats['n_valid']/graph_stats['n_total']:.1f}%) |",
        f"| edges          | {graph_stats['n_edges']:,} |",
        f"| nnz (sym)      | {graph_stats['nnz']:,} |",
        "",
    ]

    # Algorithm-specific section (KMeans → eigenvalues; Leiden → modularity;
    # HDBSCAN → cluster persistence).
    lines.extend(clusterer.report_section(algo_stats, args))

    lines += [
        "",
        "## Cluster Sizes (descending)",
        "| cluster | size |",
        "|---------|------|",
    ]
    for idx, sz in enumerate(cluster_sizes):
        lines.append(f"| {idx+1} | {sz:,} |")

    lines += [
        "",
        f"- Total valid: {sum(cluster_sizes):,}",
        f"- Max/min ratio: {cluster_sizes[0]/cluster_sizes[-1]:.2f}x",
        "",
        "## Outputs",
        "- `labels.npy` — full-N label array (0 = filtered out)",
        "- `cluster_scatter.png` — XY/XZ/YZ scatter",
    ]
    if clusterer.name == 'kmeans':
        lines.append("- `eigengap.png` — eigenvalue & eigengap plot")
    if getattr(args, "no_annotated_ply", False):
        lines.append(
            "- `point_cloud.ply` — **not written** (`--no_annotated_ply`); "
            "use checkpoint `point_cloud/iteration_*/point_cloud.ply` + "
            "`labels.npy`, or `export_segmented_ply.py`"
        )
    else:
        lines.append("- `point_cloud.ply` — annotated PLY with `cls` property")
    lines += [
        "- `renders_k{}/` — per-view cluster renders".format(args.n_clusters),
    ]

    path = os.path.join(out_dir, 'report.md')
    with open(path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    print(f"  Saved report: {path}")


def save_labels(labels_full, out_dir, n_clusters):
    """Save full-N label array as .npy (invalid Gaussians → label 0)."""
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "labels.npy")
    np.save(path, labels_full)
    print(f"  Saved labels: {path}")
    return path


def annotate_ply(ply_path, out_ply_path, labels_full):
    """
    Read original .ply, append/overwrite the 'cls' property, write new .ply.
    labels_full: [N] int array aligned to all Gaussians in the .ply.
    """
    plydata = PlyData.read(ply_path)
    vertex = plydata.elements[0]
    data   = vertex.data

    # Rebuild as structured array with cls column
    old_names  = data.dtype.names
    new_dtype  = [(n, data.dtype[n]) for n in old_names if n != 'cls']
    new_dtype += [('cls', 'f4')]

    new_data = np.empty(len(data), dtype=new_dtype)
    for n in old_names:
        if n != 'cls':
            new_data[n] = data[n]
    new_data['cls'] = labels_full.astype(np.float32)

    new_vertex = PlyElement.describe(new_data, 'vertex')
    PlyData([new_vertex], text=False).write(out_ply_path)
    print(f"  Saved annotated .ply: {out_ply_path}")


def save_cluster_scatter(gaussians, valid, labels_full, out_dir, n_clusters, subsample=50000):
    """Quick 2-D scatter coloured by cluster label."""
    pos = gaussians.get_xyz[valid].cpu().float().numpy()
    labels = labels_full[valid.cpu().numpy()]

    N = pos.shape[0]
    if N > subsample:
        idx = np.random.choice(N, subsample, replace=False)
        pos    = pos[idx]
        labels = labels[idx]

    palette = (CLUSTER_PALETTE.numpy() * 255).astype(np.uint8)
    colors  = np.array([palette[l % len(palette)] for l in labels]) / 255.0

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(f"Spectral Clusters (k={n_clusters})", fontsize=12)
    planes = [('XY', 0, 1), ('XZ', 0, 2), ('YZ', 1, 2)]
    for ax, (label, a, b) in zip(axes, planes):
        ax.scatter(pos[:, a], pos[:, b], c=colors, s=0.4, alpha=0.6)
        ax.set_title(label); ax.set_xlabel(label[0]); ax.set_ylabel(label[1])
        ax.set_aspect('equal')
    plt.tight_layout()
    path = os.path.join(out_dir, "cluster_scatter.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved scatter: {path}")


# ── Rendering ─────────────────────────────────────────────────────────────────

@torch.no_grad()
def render_clusters(gaussians, scene, deform, dataset, opt, pipe, labels_full,
                    n_clusters, out_dir, max_views=-1):
    labels_t = torch.from_numpy(labels_full).long().cuda()  # [N]
    palette  = CLUSTER_PALETTE.cuda()                        # [P, 3]
    cluster_colors = palette[labels_t % len(palette)]        # [N, 3]

    bg_color   = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device='cuda')

    all_views = scene.getTrainCameras()
    if 0 < max_views < len(all_views):
        stride = max(1, len(all_views) // max_views)
        views = all_views[::stride][:max_views]
        print(f"\nSubsampling palette render: {len(views)} of {len(all_views)} "
              f"train views (stride={stride})")
    else:
        views = all_views
    render_dir = os.path.join(out_dir, f"renders_k{n_clusters}")
    os.makedirs(render_dir, exist_ok=True)
    print(f"\nRendering {len(views)} views → {render_dir}")

    for view in tqdm(views, desc="Rendering"):
        xyz        = gaussians.get_xyz
        # view.fid lives on CPU when --load2gpu_on_the_fly is set; the deform
        # MLP runs on GPU, so align explicitly (matches boundary_suppression).
        fid        = view.fid.to(xyz.device)
        time_input = fid.unsqueeze(0).expand(xyz.shape[0], -1)

        if opt.deform_type == 'DeformNetwork':
            d_xyz, d_rotation, d_scaling = deform.step(xyz.detach(), time_input)
        else:
            d_xyz, d_rotation, d_scaling = deform.step(
                xyz.detach(), time_input,
                gaussians.get_gaussian_features.squeeze(1)
            )

        result = render(view, gaussians, pipe, background,
                        d_xyz, d_rotation, d_scaling,
                        is_6dof=dataset.is_6dof,
                        override_color=cluster_colors)

        img_name = os.path.splitext(view.image_name)[0]
        torchvision.utils.save_image(
            result['render'].cpu(),
            os.path.join(render_dir, f"{img_name}.png")
        )


# ── Main ──────────────────────────────────────────────────────────────────────

@torch.no_grad()
def main(dataset, opt, pipe, args):
    # ── 1. Load checkpoint ────────────────────────────────────────────────
    gaussians = GaussianModel(dataset.sh_degree)
    scene     = Scene(dataset, gaussians, load_iteration=args.load_iteration,
                      shuffle=False)

    deform      = DeformModel(is_blender=dataset.is_blender,
                              is_6dof=dataset.is_6dof,
                              model_type=opt.deform_type)
    scene_name  = os.path.basename(os.path.normpath(dataset.source_path))
    deform_path = args.deform_path if args.deform_path else PROJECT_ROOT
    deform.load_weights(deform_path, iteration=args.load_iteration, scene_name=scene_name)

    N_total = gaussians.get_xyz.shape[0]
    print(f"\nTotal Gaussians: {N_total:,}")

    # ── 2. Build affinity graph ───────────────────────────────────────────
    boundary = None
    if args.use_boundary:
        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        edge_method = make_edge_method(args)
        print(f"\nRGB edge method: {edge_method.name}"
              + (f" (variant={args.pidinet_variant})"
                 if edge_method.name == "pidinet" else ""))
        boundary = BoundarySuppression(
            views=scene.getTrainCameras(),
            pipe=pipe,
            background=background,
            deform_model=deform if args.use_motion else None,
            is_6dof=dataset.is_6dof,
            n_views=args.boundary_views,
            n_samples=args.boundary_samples,
            alpha_depth=args.alpha_depth,
            beta_rgb=args.beta_rgb,
            gamma=args.gamma,
            edge_chunk=args.boundary_edge_chunk,
            use_gt_rgb=not args.boundary_use_rendered_rgb,
            presmooth_sigma=args.presmooth_sigma,
            edge_method=edge_method,
        )

    timer = TimingRecorder()

    print(f"\nBuilding affinity graph (k={args.k})...")
    graph = AffinityGraph(
        gaussians,
        k=args.k,
        opacity_thresh=args.opacity_thresh,
        use_geometry=args.use_geometry,
        sigma_pos=args.sigma_pos,
        sigma_color=args.sigma_color,
        sigma_scale=args.sigma_scale,
        power=args.power,
        deform_model=deform if args.use_motion else None,
        n_time_steps=args.n_time_steps,
        static_motion_thresh=args.static_motion_thresh,
        motion_floor=args.motion_floor,
        boundary=boundary,
    )
    with timer.stage("graph_build"):
        edge_index, W, valid = graph.build(return_components=False)

    N_valid = valid.sum().item()
    timer.set_n_valid(N_valid)
    print(f"Gaussians after opacity filter: {N_valid:,} / {N_total:,} "
          f"({100 * N_valid / N_total:.1f}%)")
    print(f"Edges: {W.shape[0]:,}")

    # ── 3. Symmetrize → normalized affinity matrix ────────────────────────
    print("\nBuilding symmetric normalized Laplacian...")
    with timer.stage("symmetrize_laplacian"):
        W_sym  = symmetrize(edge_index, W, N_valid)
        A_norm = normalized_laplacian(W_sym)
    print(f"  Sparse matrix: {W_sym.shape}, nnz={W_sym.nnz:,}")

    graph_stats = {
        'n_total': N_total,
        'n_valid': N_valid,
        'n_edges': W.shape[0],
        'nnz':     W_sym.nnz,
    }

    # ── 4. Cluster (pluggable: kmeans / leiden / hdbscan) ─────────────────
    clusterer = make_clusterer(args)
    print(f"\nClusterer: {clusterer.name}")
    with timer.stage("clusterer"):
        labels_valid, algo_stats = clusterer.fit(W_sym, A_norm, args)
    args.n_clusters = algo_stats['k_used']  # propagate to report / paths / renders
    cluster_sizes = algo_stats['cluster_sizes']

    # Map back to all N Gaussians.
    # labels_valid is 0-indexed from KMeans → shift by +1 so valid clusters
    # occupy indices 1..k. Index 0 is reserved for filtered-out Gaussians,
    # which map to the black entry in CLUSTER_PALETTE and are invisible in renders.
    labels_full = np.zeros(N_total, dtype=np.int32)
    valid_np    = valid.cpu().numpy()
    labels_full[valid_np] = labels_valid + 1

    # ── 6. Save outputs ───────────────────────────────────────────────────
    now = datetime.now()
    date_dir = now.strftime("%d_%m")
    motion_tag = f"_mot{args.n_time_steps}" if args.use_motion else ""
    geo_tag = "" if args.use_geometry else "_nogeo"
    if args.use_boundary:
        # PiDiNet variants get a 3-char tag (pdF/pdS/pdT) inserted right after
        # `B-`. Sobel keeps the original (no tag) so existing run dirs reproduce.
        if args.rgb_edge_method == "pidinet":
            edge_tag = {"full": "pdF", "small": "pdS", "tiny": "pdT"}[args.pidinet_variant]
            if args.pidinet_binarize_threshold is not None:
                edge_tag += f"t{args.pidinet_binarize_threshold}"
            boundary_tag = (
                f"_B-{edge_tag}-v{args.boundary_views}"
                f"a{args.alpha_depth}b{args.beta_rgb}g{args.gamma}"
            )
        else:
            boundary_tag = (
                f"_B-v{args.boundary_views}a{args.alpha_depth}b{args.beta_rgb}g{args.gamma}"
            )
        if args.presmooth_sigma > 0:
            boundary_tag += f"ps{args.presmooth_sigma}"
    else:
        boundary_tag = ""
    prefix = clusterer.output_prefix(args, args.n_clusters)
    run_name = (
        f"{prefix}"
        f"_sc{args.sigma_color}"
        f"_ss{args.sigma_scale}"
        f"_p{args.power}"
        f"_k{args.k}"
        f"{geo_tag}"
        f"{motion_tag}"
        f"{boundary_tag}"
        f"_{now.strftime('%H-%M')}"
    )
    out_dir = os.path.join(dataset.model_path, date_dir, run_name)
    os.makedirs(out_dir, exist_ok=True)
    print(f"\nSaving outputs to: {out_dir}")

    save_run_report(out_dir, args, graph_stats, algo_stats, clusterer)
    save_labels(labels_full, out_dir, args.n_clusters)
    save_cluster_scatter(gaussians, valid, labels_full, out_dir, args.n_clusters)

    # Annotate the original .ply with cluster ids (optional; large ~checkpoint size)
    if not getattr(args, "no_annotated_ply", False):
        ply_in = os.path.join(
            dataset.model_path,
            "point_cloud",
            f"iteration_{args.load_iteration}",
            "point_cloud.ply",
        )
        ply_out = os.path.join(out_dir, "point_cloud.ply")
        if os.path.exists(ply_in):
            annotate_ply(ply_in, ply_out, labels_full)
        else:
            print(f"  [WARN] .ply not found at {ply_in}, skipping annotation")
    else:
        print("  Skipping run-dir point_cloud.ply (--no_annotated_ply).")

    # ── 7. Render ─────────────────────────────────────────────────────────
    if not args.no_render and args.max_palette_views != 0:
        with timer.stage("render_train_palette"):
            render_clusters(gaussians, scene, deform, dataset, opt, pipe,
                            labels_full, args.n_clusters, out_dir,
                            max_views=args.max_palette_views)

    # ── 8. Persist timings ────────────────────────────────────────────────
    timer.append_to_report_md(os.path.join(out_dir, "report.md"), step="spectral")
    timer.merge_into_json(os.path.join(out_dir, "timings.json"), step="spectral")

    print("\nDone.")


if __name__ == "__main__":
    parser = ArgumentParser()
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)

    parser.add_argument("--load_iteration",    type=int,   default=20000)
    parser.add_argument("--deform_path",       type=str,   default="")
    parser.add_argument("--clusterer",         type=str,   default="kmeans",
                        choices=["kmeans", "leiden", "hdbscan"],
                        help="Clustering algorithm. kmeans=spectral embedding + "
                             "k-means (eigengap auto-k). leiden=graph-direct "
                             "modularity (auto-k). hdbscan=density on spectral "
                             "embedding (auto-k).")
    parser.add_argument("--n_clusters",        type=int,   default=5,
                        help="(kmeans only) Number of clusters. Pass 0 (or "
                             "negative) to auto-pick via the eigengap heuristic. "
                             "Ignored for leiden/hdbscan.")
    parser.add_argument("--eigengap_k",        type=int,   default=15,
                        help="Number of eigenvalues to compute. For kmeans this "
                             "is the eigengap window; for hdbscan this is the "
                             "embedding dimensionality. Ignored for leiden.")
    parser.add_argument("--leiden_resolution", type=float, default=1.0,
                        help="(leiden only) RBConfigurationVertexPartition "
                             "resolution parameter. Higher → more communities.")
    parser.add_argument("--hdbscan_min_cluster_size_frac", type=float, default=0.015,
                        help="(hdbscan only) Minimum cluster size as a fraction "
                             "of N_valid Gaussians. Default 0.015 = 1.5%%.")
    parser.add_argument("--hdbscan_min_samples", type=int, default=5,
                        help="(hdbscan only) min_samples for density estimation.")
    parser.add_argument("--k",                 type=int,   default=20)
    parser.add_argument("--opacity_thresh",    type=float, default=0.05)
    parser.add_argument(
        "--no_geo",
        dest="use_geometry",
        action="store_false",
        help="No-geometry ablation: keep spatial kNN topology but replace "
             "Acolor/Aorient/Ascale edge weights with ones before optional "
             "motion and boundary terms.",
    )
    parser.set_defaults(use_geometry=True)
    parser.add_argument("--sigma_pos",         type=float, default=0.0036)
    parser.add_argument("--sigma_color",       type=float, default=0.5160)
    parser.add_argument("--sigma_scale",       type=float, default=1.0)
    parser.add_argument("--power",             type=float, default=1.0,
                        help="Sharpening exponent on W (p>1 boosts eigengap; try 4 or 8)")
    parser.add_argument("--solver",             type=str,   default="cupy",
                        choices=["lobpcg", "cupy", "randomized", "arpack"],
                        help="Eigensolver: cupy=GPU thick-restart Lanczos (default), "
                             "lobpcg=GPU/torch, arpack=CPU/scipy")
    parser.add_argument("--no_render",         action="store_true",
                        help="Skip rendering, only save labels and scatter")
    parser.add_argument("--max_palette_views", type=int, default=-1,
                        help="Cap the train-camera palette render at N evenly-"
                             "spaced views. <0 (default) renders all; 0 skips "
                             "entirely (same as --no_render); N>0 strides "
                             "through views to produce ~N palette PNGs. "
                             "Useful for ablation sweeps where the full "
                             "5400-view render dominates wall time.")
    parser.add_argument(
        "--no_annotated_ply",
        action="store_true",
        help="Do not write run-dir point_cloud.ply (duplicate of checkpoint + cls; "
             "often ~0.5GB). labels.npy remains the source of truth.",
    )
    parser.add_argument("--use_motion",        action="store_true",
                        help="Enable Amotion: fuse deformation MLP signal into affinity")
    parser.add_argument("--n_time_steps",       type=int,   default=20,
                        help="T: number of uniformly-spaced time steps in [0,1] for Amotion")
    parser.add_argument("--static_motion_thresh", type=float, default=1e-3,
                        help="‖d_xyz‖ below this threshold → Gaussian treated as static")
    parser.add_argument("--motion_floor",         type=float, default=0.2,
                        help="Minimum Amotion weight (0=hard cut, 1=disable motion); prevents graph fragmentation")
    # ── Boundary suppression (Sec. 4.3, Eq. 6) ────────────────────────────────
    parser.add_argument("--use_boundary",          action="store_true",
                        help="Enable B(i,j): multiply W by (1 - B) where B is a "
                             "sigmoid over image-space depth+RGB edge responses "
                             "along each projected Gaussian pair.")
    parser.add_argument("--boundary_views",        type=int,   default=12,
                        help="Number of uniformly-spaced train views used to "
                             "compute boundary responses.")
    parser.add_argument("--boundary_samples",      type=int,   default=8,
                        help="Samples along each projected segment for max-pooling.")
    parser.add_argument("--alpha_depth",           type=float, default=5.0,
                        help="α: weight on depth-edge response inside the sigmoid.")
    parser.add_argument("--beta_rgb",              type=float, default=2.0,
                        help="β: weight on RGB-edge response inside the sigmoid.")
    parser.add_argument("--gamma",                 type=float, default=2.0,
                        help="γ: bias inside the sigmoid (higher ⇒ less suppression).")
    parser.add_argument("--boundary_edge_chunk",   type=int,   default=500_000,
                        help="Chunk size for grid_sample over edges (memory knob).")
    parser.add_argument("--boundary_use_rendered_rgb", action="store_true",
                        help="Use rendered RGB instead of GT RGB (default uses GT).")
    parser.add_argument("--presmooth_sigma",       type=float, default=0.0,
                        help="σ (px) of Gaussian blur applied to luminance & "
                             "depth before Sobel. >0 suppresses high-frequency "
                             "texture (e.g. woven mats) so boundaries respond "
                             "to coarse silhouettes only. Try 3.0–6.0.")
    parser.add_argument("--rgb_edge_method",       type=str,   default="sobel",
                        choices=["sobel", "pidinet"],
                        help="RGB-branch edge detector. sobel = luminance "
                             "Sobel (default, fast, no deps). pidinet = "
                             "pretrained boundary detector — handles texture "
                             "vs. semantic boundary far better. Requires "
                             "weights/pidinet_<variant>.pth (run "
                             "scripts/download_pidinet.sh).")
    parser.add_argument("--pidinet_variant",       type=str,   default="full",
                        choices=["full", "small", "tiny"],
                        help="(pidinet only) Model size. full=~710K params "
                             "(best quality), small=~184K, tiny=~73K.")
    parser.add_argument("--pidinet_binarize_threshold", type=float, default=None,
                        help="(pidinet only) If set (e.g. 0.5), hard-threshold "
                             "the sigmoid edge map to {0,1} before percentile "
                             "normalization. Sparsifies dense PiDiNet outputs "
                             "so silhouettes dominate over surface texture. "
                             "Leave unset to keep the soft probability map.")

    args = parser.parse_args(sys.argv[1:])
    safe_state(False)

    main(lp.extract(args), op.extract(args), pp.extract(args), args)
