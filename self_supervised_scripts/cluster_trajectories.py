"""
Per-cluster mean-trajectory diagnostic visualisation.

Standalone one-shot script. Given an existing spectral run directory
(containing labels.npy) plus the corresponding 4DGS checkpoint, it
renders three artefacts into <run_dir>/trajectories/:

  - trajectories_3d.png       static ORB-SLAM-style figure: subsampled
                              cluster-coloured Gaussians + K trajectory
                              polylines on a black background
  - trajectories_3d.mp4       same view, trajectory lines grow over t
                              with a current-frame marker per cluster
                              (falls back to .gif if ffmpeg is missing)
  - trajectory_correlation.png  K×K cosine-similarity heatmap of mean
                                trajectories — off-diagonal ~1 ⇒ merge
                                candidates.

Server-only: needs CUDA (DeformModel runs on GPU) and the trained deform
weights at <deform_path>/deform/deform_<scene>.pth.

Usage:
    python self_supervised_scripts/cluster_trajectories.py \\
        --labels outputs/<scene>/<run_name>/labels.npy \\
        --scene_name <scene> \\
        --n_time_steps 60
"""

import os
import sys
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from argparse import ArgumentParser

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers 3d projection)
from plyfile import PlyData

from scene import DeformModel


# ── Checkpoint loading ──────────────────────────────────────────────────────

def resolve_ply_path(args):
    """Resolve which .ply to read xyz/opacity from. Tries, in order:
       1. --ply_path (explicit override)
       2. <labels_dir>/point_cloud.ply (the annotated run-dir copy)
       3. <model_path>/point_cloud/iteration_<X>/point_cloud.ply
    """
    if args.ply_path:
        return args.ply_path
    run_dir_ply = os.path.join(os.path.dirname(args.labels), "point_cloud.ply")
    if os.path.exists(run_dir_ply):
        return run_dir_ply
    if args.model_path:
        return os.path.join(args.model_path, "point_cloud",
                            f"iteration_{args.load_iteration}",
                            "point_cloud.ply")
    raise FileNotFoundError(
        "Could not locate a .ply. Pass --ply_path explicitly, or provide "
        "--model_path with a point_cloud/iteration_<X>/ tree, or place "
        "point_cloud.ply next to labels.npy.")


def load_xyz_opacity(ply_path):
    """Read positions and (raw, pre-sigmoid) opacity from the 4DGS .ply.

    We don't need the full GaussianModel — just xyz for trajectory
    extraction and opacity (rank-only) for top-K subsampling.
    """
    if not os.path.exists(ply_path):
        raise FileNotFoundError(f"point_cloud.ply not found: {ply_path}")
    print(f"Reading point cloud: {ply_path}")
    plydata = PlyData.read(ply_path)
    el = plydata.elements[0]
    xyz = np.stack([np.asarray(el["x"]),
                    np.asarray(el["y"]),
                    np.asarray(el["z"])], axis=1).astype(np.float32)
    opacity = np.asarray(el["opacity"]).astype(np.float32)  # raw, monotonic
    return xyz, opacity


# ── Trajectory extraction ────────────────────────────────────────────────────

@torch.no_grad()
def compute_trajectories(deform, pos, n_time_steps):
    """Query the deform MLP at T uniformly-spaced times in [0,1].

    Mirrors AffinityGraph._compute_trajectories. Returns Δμ(t) only;
    rotation deltas are not needed for spatial trajectory plotting.

    Returns:
        traj_xyz: [T, N, 3] displacement Δμ(t).
    """
    device = pos.device
    N = pos.shape[0]
    T = n_time_steps
    times = torch.linspace(0.0, 1.0, T, device=device)
    out = []
    for t_val in times:
        time_input = torch.full((N, 1), t_val.item(), dtype=torch.float32, device=device)
        d_xyz, _, _ = deform.step(pos.detach(), time_input)
        out.append(d_xyz)
    return torch.stack(out, dim=0)  # [T, N, 3]


def aggregate_per_cluster(world_pos_t, labels_valid, n_clusters):
    """world_pos_t: [T, N, 3] absolute positions per timestep.
    labels_valid: [N] in 0..K-1.

    Returns:
        mean_traj: [K, T, 3]
        std_traj:  [K, T] mean per-axis std of position within cluster
        sizes:     [K]
    """
    T = world_pos_t.shape[0]
    mean_traj = np.zeros((n_clusters, T, 3), dtype=np.float64)
    std_traj  = np.zeros((n_clusters, T), dtype=np.float64)
    sizes     = np.zeros(n_clusters, dtype=np.int64)
    for k in range(n_clusters):
        mask = labels_valid == k
        if not mask.any():
            continue
        cluster_pts = world_pos_t[:, mask, :]  # [T, M, 3]
        mean_traj[k] = cluster_pts.mean(axis=1)
        std_traj[k]  = cluster_pts.std(axis=1).mean(axis=-1)  # mean over xyz
        sizes[k]     = int(mask.sum())
    return mean_traj, std_traj, sizes


def trajectory_correlation(mean_traj):
    """Cosine similarity of centred mean trajectories. [K, K] in [-1, 1]."""
    K, T, _ = mean_traj.shape
    flat = mean_traj.reshape(K, T * 3)
    flat = flat - flat.mean(axis=1, keepdims=True)
    norms = np.linalg.norm(flat, axis=1, keepdims=True).clip(min=1e-12)
    flat = flat / norms
    return flat @ flat.T


# ── Visualisation ────────────────────────────────────────────────────────────

def get_cluster_colors(n_clusters):
    """[K, 3] RGB in [0,1]. tab20 covers up to 20 clusters; hsv beyond."""
    cmap = plt.get_cmap("tab20" if n_clusters <= 20 else "hsv")
    return np.stack([cmap(i % cmap.N)[:3] for i in range(n_clusters)], axis=0)


def subsample_per_cluster(opacity, labels_valid, n_clusters, per_cluster):
    """Top-N highest-opacity Gaussian indices per cluster, concatenated."""
    idx_keep = []
    for k in range(n_clusters):
        cluster_idx = np.where(labels_valid == k)[0]
        if cluster_idx.size == 0:
            continue
        if cluster_idx.size <= per_cluster:
            idx_keep.append(cluster_idx)
            continue
        opa = opacity[cluster_idx]
        top = np.argpartition(-opa, per_cluster - 1)[:per_cluster]
        idx_keep.append(cluster_idx[top])
    return np.concatenate(idx_keep) if idx_keep else np.empty(0, dtype=np.int64)


def _style_3d_axes(ax):
    ax.set_facecolor("black")
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.set_pane_color((0, 0, 0, 1))
        axis._axinfo["grid"]["color"] = (0.3, 0.3, 0.3, 0.5)
        axis.label.set_color("white")
    ax.tick_params(colors="white", labelsize=8)
    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")


def render_static(out_path, sub_pos, sub_colors, mean_traj, colors, sizes):
    fig = plt.figure(figsize=(12, 10), facecolor="black")
    ax = fig.add_subplot(111, projection="3d")
    _style_3d_axes(ax)

    ax.scatter(sub_pos[:, 0], sub_pos[:, 1], sub_pos[:, 2],
               c=sub_colors, s=1.5, alpha=0.35, depthshade=False,
               linewidths=0)

    K = mean_traj.shape[0]
    for k in range(K):
        if sizes[k] == 0:
            continue
        traj = mean_traj[k]
        ax.plot(traj[:, 0], traj[:, 1], traj[:, 2],
                color=colors[k], linewidth=2.8,
                label=f"Cluster {k+1} (n={sizes[k]:,})")
        ax.scatter([traj[0, 0]], [traj[0, 1]], [traj[0, 2]],
                   c=[colors[k]], s=40, marker="o", edgecolors="white",
                   linewidths=0.6, depthshade=False)
        ax.scatter([traj[-1, 0]], [traj[-1, 1]], [traj[-1, 2]],
                   c=[colors[k]], s=60, marker="X", edgecolors="white",
                   linewidths=0.6, depthshade=False)

    leg = ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0),
                    fontsize=8, framealpha=0.0, labelcolor="white")
    for text in leg.get_texts():
        text.set_color("white")

    fig.savefig(out_path, dpi=150, facecolor="black", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def render_animation(out_path, sub_pos, sub_colors, mean_traj, colors,
                      sizes, fps=10):
    fig = plt.figure(figsize=(12, 10), facecolor="black")
    ax = fig.add_subplot(111, projection="3d")
    _style_3d_axes(ax)
    ax.scatter(sub_pos[:, 0], sub_pos[:, 1], sub_pos[:, 2],
               c=sub_colors, s=1.5, alpha=0.35, depthshade=False,
               linewidths=0)

    K, T, _ = mean_traj.shape
    line_artists, head_artists = [], []
    for k in range(K):
        line, = ax.plot([], [], [], color=colors[k], linewidth=2.8)
        head = ax.scatter([], [], [], c=[colors[k]], s=70, marker="o",
                          edgecolors="white", linewidths=0.6,
                          depthshade=False)
        line_artists.append(line)
        head_artists.append(head)

    title = ax.set_title("", color="white", fontsize=11)

    def update(frame):
        for k in range(K):
            if sizes[k] == 0:
                continue
            traj = mean_traj[k, : frame + 1]
            line_artists[k].set_data(traj[:, 0], traj[:, 1])
            line_artists[k].set_3d_properties(traj[:, 2])
            head_artists[k]._offsets3d = (
                [mean_traj[k, frame, 0]],
                [mean_traj[k, frame, 1]],
                [mean_traj[k, frame, 2]],
            )
        title.set_text(f"t = {frame + 1}/{T}")
        return line_artists + head_artists + [title]

    anim = FuncAnimation(fig, update, frames=T, interval=1000 / fps, blit=False)

    try:
        writer = FFMpegWriter(fps=fps, bitrate=4000)
        anim.save(out_path, writer=writer, savefig_kwargs={"facecolor": "black"})
        print(f"  Saved: {out_path}")
    except (RuntimeError, FileNotFoundError) as e:
        gif_path = os.path.splitext(out_path)[0] + ".gif"
        print(f"  ffmpeg unavailable ({e}); falling back to GIF: {gif_path}")
        anim.save(gif_path, writer=PillowWriter(fps=fps),
                  savefig_kwargs={"facecolor": "black"})
        print(f"  Saved: {gif_path}")
    plt.close(fig)


def render_correlation(out_path, corr, sizes):
    K = corr.shape[0]
    fig, ax = plt.subplots(figsize=(max(6, 0.4 * K + 4), max(5, 0.4 * K + 3)))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1.0, vmax=1.0)
    ax.set_xticks(range(K)); ax.set_yticks(range(K))
    labels = [f"{k+1}\nn={sizes[k]:,}" for k in range(K)]
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_title("Cluster mean-trajectory cosine similarity\n"
                 "(off-diagonal → 1 ⇒ merge candidates)")
    for i in range(K):
        for j in range(K):
            if i == j:
                continue
            v = corr[i, j]
            if abs(v) > 0.4:
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        color="white" if abs(v) > 0.7 else "black",
                        fontsize=6)
    fig.colorbar(im, ax=ax, fraction=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ── Main ─────────────────────────────────────────────────────────────────────

@torch.no_grad()
def main(args):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required (DeformModel runs on GPU).")

    labels_full = np.load(args.labels)
    print(f"Loaded labels: {args.labels}  shape={labels_full.shape}  "
          f"unique_count={len(np.unique(labels_full))}")

    xyz_np, opacity_np = load_xyz_opacity(resolve_ply_path(args))
    N_total = xyz_np.shape[0]
    if labels_full.shape[0] != N_total:
        raise ValueError(f"labels.npy length {labels_full.shape[0]} != "
                         f"N_total {N_total} from .ply")

    valid_mask = labels_full > 0
    pos_valid = torch.from_numpy(xyz_np[valid_mask]).cuda()
    labels_valid = labels_full[valid_mask] - 1
    n_clusters = int(labels_valid.max()) + 1
    print(f"Valid Gaussians: {labels_valid.size:,} / {N_total:,}  "
          f"K={n_clusters}")

    deform = DeformModel(is_blender=args.is_blender,
                         is_6dof=args.is_6dof,
                         model_type=args.deform_type)
    deform_path = args.deform_path if args.deform_path else PROJECT_ROOT
    deform.load_weights(deform_path,
                        iteration=args.load_iteration,
                        scene_name=args.scene_name or None)

    print(f"Computing trajectories (T={args.n_time_steps}) ...")
    traj_xyz = compute_trajectories(deform, pos_valid,
                                     args.n_time_steps)  # [T, N, 3]

    pos_valid_np = pos_valid.detach().cpu().numpy()
    world_pos_t = pos_valid_np[None, :, :] + traj_xyz.detach().cpu().numpy()  # [T, N, 3]

    mean_traj, std_traj, sizes = aggregate_per_cluster(
        world_pos_t, labels_valid, n_clusters)
    corr = trajectory_correlation(mean_traj)

    print("\nPer-cluster motion (RMS displacement over T):")
    for k in range(n_clusters):
        if sizes[k] == 0:
            continue
        disp = mean_traj[k] - mean_traj[k, 0]
        rms = float(np.sqrt((disp ** 2).sum(axis=-1).mean()))
        print(f"  cluster {k+1:>2}  n={sizes[k]:>8,}  "
              f"rms_disp={rms:.4f}  mean_intra_std={std_traj[k].mean():.4f}")

    if args.subsample_per_cluster > 0:
        keep = subsample_per_cluster(opacity_np[valid_mask], labels_valid,
                                      n_clusters, args.subsample_per_cluster)
        sub_pos = pos_valid_np[keep]
        sub_labels = labels_valid[keep]
    else:
        sub_pos = pos_valid_np
        sub_labels = labels_valid
    print(f"Subsampled background point cloud: {sub_pos.shape[0]:,} points")

    colors = get_cluster_colors(n_clusters)
    sub_colors = colors[sub_labels]

    out_dir = args.out_dir or os.path.join(os.path.dirname(args.labels),
                                            "trajectories")
    os.makedirs(out_dir, exist_ok=True)
    print(f"\nWriting outputs to: {out_dir}")

    render_static(os.path.join(out_dir, "trajectories_3d.png"),
                  sub_pos, sub_colors, mean_traj, colors, sizes)
    render_animation(os.path.join(out_dir, "trajectories_3d.mp4"),
                     sub_pos, sub_colors, mean_traj, colors, sizes,
                     fps=args.fps)
    render_correlation(os.path.join(out_dir, "trajectory_correlation.png"),
                       corr, sizes)

    np.savez(os.path.join(out_dir, "trajectories.npz"),
             mean_traj=mean_traj, std_traj=std_traj, sizes=sizes,
             correlation=corr)
    print(f"  Saved: {os.path.join(out_dir, 'trajectories.npz')}")
    print("\nDone.")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--labels",      type=str, required=True,
                        help="Path to labels.npy from a prior spectral run.")
    parser.add_argument("--ply_path",    type=str, default="",
                        help="Explicit .ply to read xyz/opacity from. If "
                             "omitted, tries <labels_dir>/point_cloud.ply, "
                             "then <model_path>/point_cloud/iteration_X/...")
    parser.add_argument("--model_path", "-m", type=str, default="",
                        help="4DGS checkpoint root. Only used as a fallback "
                             "when --ply_path is not given and no .ply "
                             "exists next to labels.npy.")
    parser.add_argument("--load_iteration", type=int, default=20000)
    parser.add_argument("--out_dir",     type=str, default="",
                        help="Output dir; defaults to <labels_dir>/trajectories.")

    parser.add_argument("--deform_path", type=str, default="",
                        help="Root containing deform/ weights. Defaults to "
                             "PROJECT_ROOT (matches spectral_cluster.py).")
    parser.add_argument("--scene_name",  type=str, default="",
                        help="Used to resolve "
                             "<deform_path>/deform/deform_<scene>.pth. "
                             "Leave empty to use the iteration-based path.")
    parser.add_argument("--deform_type", type=str, default="DeformNetwork",
                        help="DeformModel class name (default: DeformNetwork).")
    parser.add_argument("--is_blender",  action="store_true")
    parser.add_argument("--is_6dof",     action="store_true")

    parser.add_argument("--n_time_steps", type=int, default=60,
                        help="T: number of uniformly-spaced times in [0,1].")
    parser.add_argument("--subsample_per_cluster", type=int, default=2000,
                        help="Top-N highest-opacity Gaussians per cluster "
                             "for the background point cloud (0 = keep all).")
    parser.add_argument("--fps", type=int, default=10)

    main(parser.parse_args(sys.argv[1:]))
