"""Long-range cluster merge: collapse spectral clusters that share trajectory,
feature, and color statistics.

Reads an existing spectral run's `labels.npy` plus the corresponding 4DGS
checkpoint, computes a K x K similarity matrix from three signals, thresholds
it, takes the transitive closure (union-find), and writes a relabelled
`labels_merged.npy` alongside a `merge_report.md` documenting which clusters
fused and why.

Signals:
  S_traj(i,j)  = Pearson correlation of per-cluster mean displacement
                 trajectories over T MLP queries (matches AffinityGraph's
                 Atraj convention).
  S_feat(i,j)  = cosine similarity of per-cluster mean 32-d
                 `gaussian_feats_*` embedding.
  S_color(i,j) = exp(-||mean_rgb_i - mean_rgb_j||^2 / (2 * sigma_color^2))
                 on the DC SH coefficient (same color used by AffinityGraph).

Merge edge: w_traj * S_traj + w_feat * S_feat + w_color * S_color > tau.

Usage:
    python self_supervised_scripts/merge_clusters.py \\
        -s data/Neu3D/coffee_martini \\
        --model_path output/coffee_martini_run \\
        --load_iteration 30000 \\
        --labels_file outputs/.../labels.npy
"""

from __future__ import annotations

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from argparse import ArgumentParser

import numpy as np
import torch
from plyfile import PlyData

from scene import DeformModel
from self_supervised_scripts.cluster_trajectories import (
    compute_trajectories,
    resolve_ply_path,
)


# ── Checkpoint loading ─────────────────────────────────────────────────────────

def load_gaussian_attrs(ply_path: str, feature_dim: int = 32):
    """Read xyz, opacity, DC SH color, and gaussian_feats from the 4DGS .ply.

    Returns:
        xyz:      [N, 3] float32
        opacity:  [N]    float32 (raw, pre-sigmoid)
        color:    [N, 3] float32 (DC SH coefficient, matches AffinityGraph)
        feats:    [N, F] float32 (None if PLY has no gaussian_feats_*)
    """
    if not os.path.exists(ply_path):
        raise FileNotFoundError(f"point_cloud.ply not found: {ply_path}")
    print(f"Reading point cloud: {ply_path}")
    plydata = PlyData.read(ply_path)
    el = plydata.elements[0]
    xyz = np.stack([
        np.asarray(el["x"]),
        np.asarray(el["y"]),
        np.asarray(el["z"]),
    ], axis=1).astype(np.float32)
    opacity = np.asarray(el["opacity"]).astype(np.float32)
    color = np.stack([
        np.asarray(el["f_dc_0"]),
        np.asarray(el["f_dc_1"]),
        np.asarray(el["f_dc_2"]),
    ], axis=1).astype(np.float32)

    feat_names = [p.name for p in el.properties
                  if p.name.startswith("gaussian_feats_")]
    if feat_names:
        feat_names.sort(key=lambda n: int(n.split("_")[-1]))
        F = min(feature_dim, len(feat_names))
        feats = np.stack(
            [np.asarray(el[f"gaussian_feats_{i}"]) for i in range(F)],
            axis=1,
        ).astype(np.float32)
    else:
        feats = None
    return xyz, opacity, color, feats


# ── Pure functions (unit-testable) ────────────────────────────────────────────

def compute_cluster_aggregates(
    labels_valid: np.ndarray,
    traj_xyz: np.ndarray,
    feats: np.ndarray | None,
    color: np.ndarray,
    n_clusters: int,
    min_cluster_size: int,
):
    """Aggregate per-cluster means used by the similarity matrix.

    Args:
        labels_valid:  [N_v] int in 0..K-1 (already shifted from 1..K).
        traj_xyz:      [T, N_v, 3] displacement trajectories.
        feats:         [N_v, F] gaussian feature embeddings, or None.
        color:         [N_v, 3] DC SH RGB.
        n_clusters:    K.
        min_cluster_size: clusters with fewer Gaussians are flagged invalid
                          (their rows/cols in S are zeroed downstream).

    Returns:
        mean_traj:  [K, T, 3]
        mean_feat:  [K, F]  (L2-normalised) or None if feats is None
        mean_rgb:   [K, 3]
        sizes:      [K]    int
        valid:      [K]    bool — sizes >= min_cluster_size
    """
    T = traj_xyz.shape[0]
    F = feats.shape[1] if feats is not None else 0
    mean_traj = np.zeros((n_clusters, T, 3), dtype=np.float64)
    mean_feat = np.zeros((n_clusters, F), dtype=np.float64) if F else None
    mean_rgb = np.zeros((n_clusters, 3), dtype=np.float64)
    sizes = np.zeros(n_clusters, dtype=np.int64)

    for k in range(n_clusters):
        mask = labels_valid == k
        m = int(mask.sum())
        sizes[k] = m
        if m == 0:
            continue
        mean_traj[k] = traj_xyz[:, mask, :].mean(axis=1)
        mean_rgb[k] = color[mask].mean(axis=0)
        if mean_feat is not None:
            mean_feat[k] = feats[mask].mean(axis=0)

    if mean_feat is not None:
        norms = np.linalg.norm(mean_feat, axis=1, keepdims=True).clip(min=1e-12)
        mean_feat = mean_feat / norms

    valid = sizes >= min_cluster_size
    return mean_traj, mean_feat, mean_rgb, sizes, valid


def compute_similarity_matrix(
    mean_traj: np.ndarray,
    mean_feat: np.ndarray | None,
    mean_rgb: np.ndarray,
    valid: np.ndarray,
    w_traj: float,
    w_feat: float,
    w_color: float,
    sigma_color: float,
):
    """Build the K x K weighted similarity matrix S(i,j) and its components.

    Off-diagonal only; the diagonal is zeroed. Rows/cols of invalid clusters
    (size < min_cluster_size, or feature mode requested but `mean_feat` is
    None for that cluster) are zeroed so they never form merge edges.

    When `mean_feat is None` (PLY had no gaussian_feats_*), w_feat is
    treated as 0 and the remaining weights renormalised to sum 1.

    When all trajectories are static (var < 1e-6 across clusters) the
    trajectory weight is similarly redistributed — see `merge_clusters.py`'s
    static-scene edge case.

    Returns:
        S:        [K, K] weighted total score.
        parts:    dict with keys {"traj","feat","color"} → [K, K] arrays.
        weights:  dict of the (possibly renormalised) weights actually used.
    """
    K = mean_traj.shape[0]

    # Trajectory: Pearson over flattened (T, 3) → centred cosine on [K, T*3].
    flat = mean_traj.reshape(K, -1)
    centred = flat - flat.mean(axis=1, keepdims=True)
    norms = np.linalg.norm(centred, axis=1, keepdims=True).clip(min=1e-12)
    flat_norm = centred / norms
    S_traj = flat_norm @ flat_norm.T  # [K, K] in [-1, 1]
    # Detect "no motion at all" — all centred norms tiny.
    traj_alive = float(norms.max()) > 1e-6

    # Feature similarity (already L2-normalised in compute_cluster_aggregates).
    if mean_feat is not None:
        S_feat = mean_feat @ mean_feat.T  # [K, K] in [-1, 1]
    else:
        S_feat = np.zeros((K, K), dtype=np.float64)

    # Color: RBF on Euclidean distance in DC SH space.
    diff = mean_rgb[:, None, :] - mean_rgb[None, :, :]  # [K, K, 3]
    d2 = (diff ** 2).sum(axis=-1)
    S_color = np.exp(-d2 / (2.0 * sigma_color ** 2))

    # Renormalise weights if a signal is unavailable.
    w_t = w_traj if traj_alive else 0.0
    w_f = w_feat if mean_feat is not None else 0.0
    w_c = w_color
    total = w_t + w_f + w_c
    if total <= 0:
        raise ValueError(
            "All merge signals are disabled (no trajectory, no features, "
            "no color weight). At least one must be active."
        )
    w_t, w_f, w_c = w_t / total, w_f / total, w_c / total

    S = w_t * S_traj + w_f * S_feat + w_c * S_color

    # Zero diagonal and invalid rows/cols.
    np.fill_diagonal(S, 0.0)
    bad = ~valid
    S[bad, :] = 0.0
    S[:, bad] = 0.0

    parts = {"traj": S_traj, "feat": S_feat, "color": S_color}
    weights = {"w_traj": w_t, "w_feat": w_f, "w_color": w_c}
    return S, parts, weights


def union_find_merge(S: np.ndarray, tau: float):
    """Connected-component merge over edges {(i,j) : S[i,j] > tau}.

    Returns:
        new_id_of:  [K] int — new cluster id per old cluster (0..K'-1).
        groups:     list[list[int]] — old ids per merged group, ordered by
                    the new id (group 0 is the largest group, ties broken
                    by min original id for determinism).
        edges:      list[(i, j, S_ij)] sorted by S_ij descending, only
                    pairs with S_ij > tau and i < j.
    """
    K = S.shape[0]
    parent = list(range(K))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            # Keep the smaller-index root for determinism.
            if ra < rb:
                parent[rb] = ra
            else:
                parent[ra] = rb

    edges = []
    for i in range(K):
        for j in range(i + 1, K):
            if S[i, j] > tau:
                edges.append((i, j, float(S[i, j])))
                union(i, j)
    edges.sort(key=lambda e: -e[2])

    # Bucket by root, then sort buckets so the largest group → new id 0.
    buckets: dict[int, list[int]] = {}
    for k in range(K):
        buckets.setdefault(find(k), []).append(k)
    groups = sorted(buckets.values(), key=lambda g: (-len(g), min(g)))

    new_id_of = np.zeros(K, dtype=np.int64)
    for new_id, group in enumerate(groups):
        for old in group:
            new_id_of[old] = new_id
    return new_id_of, groups, edges


# ── Report ────────────────────────────────────────────────────────────────────

def write_merge_report(
    path: str,
    K: int,
    K_prime: int,
    weights: dict,
    tau: float,
    sigma_color: float,
    min_cluster_size: int,
    groups: list,
    edges: list,
    parts: dict,
    sizes: np.ndarray,
    valid: np.ndarray,
) -> None:
    lines = []
    lines.append("# Long-range cluster merge report\n")
    lines.append(f"- **Input clusters (K):** {K}")
    lines.append(f"- **Output clusters (K'):** {K_prime}")
    lines.append(f"- **Merges fired:** {K - K_prime}")
    lines.append(f"- **Weights:** w_traj={weights['w_traj']:.3f}, "
                 f"w_feat={weights['w_feat']:.3f}, "
                 f"w_color={weights['w_color']:.3f}")
    lines.append(f"- **tau:** {tau}")
    lines.append(f"- **sigma_color:** {sigma_color}")
    lines.append(f"- **min_cluster_size:** {min_cluster_size}\n")

    lines.append("## Merge edges (S[i,j] > tau)\n")
    if not edges:
        lines.append("_No edges crossed the threshold._\n")
    else:
        lines.append("| i | j | S | S_traj | S_feat | S_color | size_i | size_j |")
        lines.append("|---|---|---|--------|--------|---------|--------|--------|")
        for i, j, s in edges:
            lines.append(
                f"| {i} | {j} | {s:.4f} | "
                f"{parts['traj'][i, j]:.4f} | "
                f"{parts['feat'][i, j]:.4f} | "
                f"{parts['color'][i, j]:.4f} | "
                f"{sizes[i]} | {sizes[j]} |"
            )
        lines.append("")

    lines.append("## Merged groups\n")
    lines.append("| new_id | old_ids | total_size |")
    lines.append("|--------|---------|------------|")
    for new_id, group in enumerate(groups):
        total = int(sum(sizes[k] for k in group))
        lines.append(f"| {new_id} | {group} | {total} |")
    lines.append("")

    if (~valid).any():
        skipped = [int(k) for k in np.where(~valid)[0]]
        lines.append(f"## Skipped clusters (size < min_cluster_size)\n")
        lines.append(f"`{skipped}` — these stay in their own group, but never "
                     "match a merge edge.\n")

    with open(path, "w") as f:
        f.write("\n".join(lines))
    print(f"Wrote merge report: {path}")


# ── Main driver ───────────────────────────────────────────────────────────────

def remap_labels(labels_full: np.ndarray, new_id_of: np.ndarray) -> np.ndarray:
    """Apply old→new id mapping. Filtered entries (label==0) stay 0; valid
    labels 1..K shift by -1, get remapped, then shift back by +1."""
    merged = labels_full.copy()
    mask = labels_full > 0
    old_idx = labels_full[mask] - 1
    merged[mask] = new_id_of[old_idx] + 1
    return merged


def main(args):
    # Resolve PLY + labels.
    if not os.path.exists(args.labels_file):
        raise FileNotFoundError(f"labels.npy not found: {args.labels_file}")
    print(f"Reading labels: {args.labels_file}")
    labels_full = np.load(args.labels_file)

    ns = type("ResolveArgs", (), {
        "labels": args.labels_file,
        "ply_path": args.ply_path,
        "model_path": args.model_path,
        "load_iteration": args.load_iteration,
    })()
    ply_path = resolve_ply_path(ns)
    xyz, opacity, color, feats = load_gaussian_attrs(ply_path)

    # `labels==0` already encodes the opacity filter applied at affinity
    # build time — no need to re-threshold.
    valid_mask = labels_full > 0
    if not valid_mask.any():
        raise RuntimeError("labels.npy has no valid (non-zero) entries.")
    labels_valid = labels_full[valid_mask].astype(np.int64) - 1  # → 0..K-1
    K = int(labels_valid.max()) + 1
    print(f"Valid Gaussians: {valid_mask.sum():,} / {len(labels_full):,}  K={K}")

    if K <= 1:
        print("K <= 1 → nothing to merge. Writing pass-through labels_merged.npy.")
        out_dir = args.output_dir or os.path.dirname(os.path.abspath(args.labels_file))
        os.makedirs(out_dir, exist_ok=True)
        np.save(os.path.join(out_dir, "labels_merged.npy"), labels_full)
        with open(os.path.join(out_dir, "merge_report.md"), "w") as f:
            f.write(f"# Long-range cluster merge report\n\nK={K}; nothing to merge.\n")
        return

    # Trajectories: query deform MLP at T uniformly-spaced timesteps.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pos_valid = torch.from_numpy(xyz[valid_mask]).to(device)

    scene_name = os.path.basename(os.path.normpath(args.source_path))
    deform_path = args.deform_path if args.deform_path else PROJECT_ROOT
    print(f"Loading deform weights (scene='{scene_name}', iter={args.load_iteration})...")
    deform = DeformModel(is_blender=False, is_6dof=False)
    deform.load_weights(deform_path, iteration=args.load_iteration, scene_name=scene_name)

    print(f"Computing trajectories (T={args.n_time_steps})...")
    traj_xyz_t = compute_trajectories(deform, pos_valid, args.n_time_steps)  # [T, N_v, 3]
    traj_xyz = traj_xyz_t.cpu().numpy().astype(np.float64)

    color_valid = color[valid_mask]
    feats_valid = feats[valid_mask] if feats is not None else None

    mean_traj, mean_feat, mean_rgb, sizes, valid = compute_cluster_aggregates(
        labels_valid=labels_valid,
        traj_xyz=traj_xyz,
        feats=feats_valid,
        color=color_valid,
        n_clusters=K,
        min_cluster_size=args.min_cluster_size,
    )

    S, parts, weights = compute_similarity_matrix(
        mean_traj=mean_traj,
        mean_feat=mean_feat,
        mean_rgb=mean_rgb,
        valid=valid,
        w_traj=args.w_traj,
        w_feat=args.w_feat,
        w_color=args.w_color,
        sigma_color=args.sigma_color,
    )

    new_id_of, groups, edges = union_find_merge(S, tau=args.tau)
    K_prime = len(groups)
    print(f"Merge: K={K} → K'={K_prime}  ({K - K_prime} merges, "
          f"{len(edges)} qualifying edges)")

    out_dir = args.output_dir or os.path.dirname(os.path.abspath(args.labels_file))
    os.makedirs(out_dir, exist_ok=True)

    merged = remap_labels(labels_full, new_id_of)
    out_labels = os.path.join(out_dir, "labels_merged.npy")
    np.save(out_labels, merged)
    print(f"Wrote labels_merged.npy: {out_labels}  (max id={merged.max()})")

    write_merge_report(
        path=os.path.join(out_dir, "merge_report.md"),
        K=K,
        K_prime=K_prime,
        weights=weights,
        tau=args.tau,
        sigma_color=args.sigma_color,
        min_cluster_size=args.min_cluster_size,
        groups=groups,
        edges=edges,
        parts=parts,
        sizes=sizes,
        valid=valid,
    )


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("-s", "--source_path", type=str, required=True,
                        help="Scene data directory (used only to derive scene "
                             "name for deform weight lookup).")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Stage 1 output dir (used to locate point_cloud.ply).")
    parser.add_argument("--load_iteration", type=int, default=30000)
    parser.add_argument("--labels_file", type=str, required=True,
                        help="Path to labels.npy produced by spectral_cluster.py.")
    parser.add_argument("--output_dir", type=str, default="",
                        help="Where to write labels_merged.npy and merge_report.md. "
                             "Default: same dir as --labels_file.")
    parser.add_argument("--ply_path", type=str, default="",
                        help="Explicit point_cloud.ply override (otherwise resolved "
                             "from --model_path + --load_iteration).")
    parser.add_argument("--deform_path", type=str, default="",
                        help="Where deform_<scene>.pth lives (default: repo root).")

    parser.add_argument("--n_time_steps", type=int, default=20,
                        help="T: deform MLP query count for trajectory signal.")
    parser.add_argument("--w_traj", type=float, default=0.5)
    parser.add_argument("--w_feat", type=float, default=0.3)
    parser.add_argument("--w_color", type=float, default=0.2)
    parser.add_argument("--tau", type=float, default=0.85,
                        help="Merge threshold on weighted score.")
    parser.add_argument("--sigma_color", type=float, default=0.5,
                        help="RBF bandwidth for DC SH color distance.")
    parser.add_argument("--min_cluster_size", type=int, default=10,
                        help="Clusters smaller than this are excluded from merging.")

    args = parser.parse_args()
    main(args)
