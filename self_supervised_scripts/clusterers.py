"""Pluggable clusterers for the Gaussian affinity graph.

Three implementations behind a common interface:

  - KMeansClusterer:  spectral embedding + k-means (existing baseline).
  - LeidenClusterer:  graph-direct community detection (modularity).
  - HDBSCANClusterer: density-based clustering on the spectral embedding.

Each clusterer takes the symmetric sparse affinity (W_sym) and the normalized
affinity (A_norm) and produces a 0-indexed label array of shape [N_valid] plus
an algorithm-specific stats dict for the run report.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import scipy.sparse as sp


class Clusterer(ABC):
    """Common interface for all clusterers."""

    name: str = ""

    @abstractmethod
    def fit(
        self,
        W_sym: sp.csr_matrix,
        A_norm: sp.csr_matrix,
        args: Any,
    ) -> tuple[np.ndarray, dict]:
        """Run the clusterer.

        Returns:
            labels: int32 array of shape [N_valid], values in [0, k-1].
            stats:  algorithm-specific dict; must include key 'k_used' (int)
                    and 'cluster_sizes' (sorted-descending list of int).
        """

    @abstractmethod
    def output_prefix(self, args: Any, k_actual: int) -> str:
        """Algorithm-specific portion of the output dir name (no leading slash)."""

    @abstractmethod
    def report_section(self, stats: dict, args: Any) -> list[str]:
        """Return the algorithm-specific section(s) of report.md as a list of lines."""


# ── KMeans (existing baseline) ─────────────────────────────────────────────────


class KMeansClusterer(Clusterer):
    """Spectral embedding + k-means. Wraps the original spectral_cluster pipeline."""

    name = "kmeans"

    def fit(self, W_sym, A_norm, args):
        from self_supervised_scripts.spectral_solver import (
            spectral_embed,
            plot_eigengap,
            run_kmeans,
        )

        requested_k = None if args.n_clusters <= 0 else args.n_clusters
        embedding, k_used, spectral_stats = spectral_embed(
            A_norm, requested_k, eigengap_k=args.eigengap_k, solver=args.solver
        )

        # Eigengap diagnostic plot — kmeans-only artifact.
        plot_eigengap(
            np.array(spectral_stats["eigenvalues"]),
            np.array(spectral_stats["eigengaps"]),
            spectral_stats["suggested_k"],
            k_used,
        )

        labels, cluster_sizes = run_kmeans(embedding, k_used)
        stats = {
            "k_used": int(k_used),
            "spectral": spectral_stats,
            "cluster_sizes": cluster_sizes,
        }
        return labels.astype(np.int32), stats

    def output_prefix(self, args, k_actual):
        return f"spectral_k{k_actual}"

    def report_section(self, stats, args):
        spec = stats["spectral"]
        ev = spec["eigenvalues"]
        gap = spec["eigengaps"]
        lines = [
            "## Spectral Analysis",
            f"- Eigengap suggested k: **{spec['suggested_k']}** (requested: {args.n_clusters})",
            f"- ρ = δ_max/δ_2nd: **{spec['rho']:.3f}**  (higher ⇒ more decisive cluster count)",
            "",
            f"### Eigenvalues (top-{spec['k_compute']})",
            "| i | λ_i | gap (λ_i − λ_i+1) |",
            "|---|-----|-------------------|",
        ]
        for i, lam in enumerate(ev):
            g = f"{gap[i]:.6f}" if i < len(gap) else "—"
            lines.append(f"| {i + 1} | {lam:.10f} | {g} |")
        return lines


# ── Leiden (graph-direct community detection) ──────────────────────────────────


class LeidenClusterer(Clusterer):
    """Modularity-based community detection on the symmetric affinity graph.

    Uses leidenalg with RBConfigurationVertexPartition (resolution-aware
    modularity). Skips spectral embedding entirely.
    """

    name = "leiden"

    def fit(self, W_sym, A_norm, args):
        try:
            import igraph as ig
            import leidenalg
        except ImportError as exc:
            raise ImportError(
                "leiden requires leidenalg + python-igraph. Install with:\n"
                "  pip install leidenalg python-igraph"
            ) from exc

        # scipy CSR (symmetric) → upper-triangle edges → undirected igraph.
        W_coo = W_sym.tocoo()
        upper = W_coo.row < W_coo.col
        edges = list(zip(W_coo.row[upper].tolist(), W_coo.col[upper].tolist()))
        weights = W_coo.data[upper].tolist()
        n = W_sym.shape[0]

        print(f"  Building igraph (n={n:,}, edges={len(edges):,}) ...")
        g = ig.Graph(n=n, edges=edges, directed=False)
        g.es["weight"] = weights

        resolution = float(args.leiden_resolution)
        print(f"  Running Leiden (resolution={resolution}) ...")
        partition = leidenalg.find_partition(
            g,
            leidenalg.RBConfigurationVertexPartition,
            weights="weight",
            resolution_parameter=resolution,
            seed=0,
        )
        labels = np.asarray(partition.membership, dtype=np.int32)
        modularity_q = float(partition.modularity)

        n_clusters = int(labels.max()) + 1
        counts = np.bincount(labels, minlength=n_clusters)
        sizes = sorted(counts.tolist(), reverse=True)
        print(
            f"  Leiden → {n_clusters} communities, modularity Q={modularity_q:.4f}"
        )
        print(f"  Cluster sizes (top-10): {sizes[:10]}")

        stats = {
            "k_used": n_clusters,
            "modularity_q": modularity_q,
            "resolution": resolution,
            "cluster_sizes": sizes,
        }
        return labels, stats

    def output_prefix(self, args, k_actual):
        return f"leiden_res{args.leiden_resolution}"

    def report_section(self, stats, args):
        return [
            "## Leiden Community Detection",
            f"- Resolution parameter: **{stats['resolution']}**",
            f"- Communities found: **{stats['k_used']}**",
            f"- Modularity Q: **{stats['modularity_q']:.4f}**  (higher ⇒ stronger community structure)",
        ]


# ── HDBSCAN (density on spectral embedding) ────────────────────────────────────


class HDBSCANClusterer(Clusterer):
    """Density-based clustering on the spectral embedding.

    Uses eigengap_k as the embedding dimensionality (no eigengap heuristic;
    the full top-eigengap_k eigenvectors form the feature space). Noise points
    are reassigned to the nearest non-noise cluster in embedding space.
    """

    name = "hdbscan"

    def fit(self, W_sym, A_norm, args):
        try:
            import hdbscan as hdbscan_lib
        except ImportError as exc:
            raise ImportError(
                "hdbscan requires the hdbscan package. Install with:\n"
                "  pip install hdbscan"
            ) from exc
        from sklearn.neighbors import NearestNeighbors
        from self_supervised_scripts.spectral_solver import spectral_embed

        # Embedding: top-eigengap_k eigenvectors (no eigengap heuristic).
        # Passing n_clusters=eigengap_k forces k_used == eigengap_k.
        embedding, _k_unused, spectral_stats = spectral_embed(
            A_norm,
            n_clusters=args.eigengap_k,
            eigengap_k=args.eigengap_k,
            solver=args.solver,
        )

        n = embedding.shape[0]
        min_cluster_size = max(2, int(round(n * args.hdbscan_min_cluster_size_frac)))
        min_samples = int(args.hdbscan_min_samples)
        print(
            f"  Running HDBSCAN (min_cluster_size={min_cluster_size}, "
            f"min_samples={min_samples}, embedding_dim={embedding.shape[1]}) ..."
        )
        clusterer = hdbscan_lib.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            core_dist_n_jobs=-1,
        )
        raw_labels = clusterer.fit_predict(embedding).astype(np.int32)

        n_noise = int((raw_labels == -1).sum())
        n_clustered = int((raw_labels != -1).sum())
        print(
            f"  HDBSCAN raw: {n_clustered:,} clustered, "
            f"{n_noise:,} noise (will be reassigned)"
        )

        labels = raw_labels.copy()
        if n_noise > 0 and n_clustered > 0:
            print("  Reassigning noise to nearest non-noise cluster (k=1 NN in embedding) ...")
            non_noise = labels != -1
            nn = NearestNeighbors(n_neighbors=1, n_jobs=-1)
            nn.fit(embedding[non_noise])
            _, idx = nn.kneighbors(embedding[~non_noise])
            labels[~non_noise] = labels[non_noise][idx.ravel()]
        elif n_clustered == 0:
            print("  [WARN] HDBSCAN found 0 clusters; assigning all points to cluster 0.")
            labels[:] = 0

        # Compact label range to a contiguous [0, k-1].
        unique = np.unique(labels)
        remap = {old: new for new, old in enumerate(sorted(unique.tolist()))}
        labels = np.asarray([remap[int(v)] for v in labels], dtype=np.int32)

        n_clusters_final = int(labels.max()) + 1
        counts = np.bincount(labels, minlength=n_clusters_final)
        sizes = sorted(counts.tolist(), reverse=True)

        try:
            persistence = clusterer.cluster_persistence_.tolist()
        except AttributeError:
            persistence = []

        print(f"  HDBSCAN final: {n_clusters_final} clusters")
        print(f"  Cluster sizes (top-10): {sizes[:10]}")

        stats = {
            "k_used": n_clusters_final,
            "min_cluster_size": min_cluster_size,
            "min_cluster_size_frac": float(args.hdbscan_min_cluster_size_frac),
            "min_samples": min_samples,
            "n_noise_before_reassign": n_noise,
            "cluster_persistence": persistence,
            "cluster_sizes": sizes,
        }
        return labels, stats

    def output_prefix(self, args, k_actual):
        return (
            f"hdbscan_mcs{args.hdbscan_min_cluster_size_frac}"
            f"_ms{args.hdbscan_min_samples}"
        )

    def report_section(self, stats, args):
        lines = [
            "## HDBSCAN Density-Based Clustering",
            f"- min_cluster_size_frac: **{stats['min_cluster_size_frac']}** "
            f"(resolved to {stats['min_cluster_size']:,} points)",
            f"- min_samples: **{stats['min_samples']}**",
            f"- Clusters found: **{stats['k_used']}**",
            f"- Noise points reassigned to nearest cluster: **{stats['n_noise_before_reassign']:,}**",
        ]
        if stats["cluster_persistence"]:
            lines += [
                "",
                "### Cluster Persistence (descending)",
                "| i | persistence |",
                "|---|-------------|",
            ]
            for i, p in enumerate(sorted(stats["cluster_persistence"], reverse=True)):
                lines.append(f"| {i + 1} | {p:.4f} |")
        return lines


# ── Factory ────────────────────────────────────────────────────────────────────


CLUSTERER_REGISTRY = {
    "kmeans": KMeansClusterer,
    "leiden": LeidenClusterer,
    "hdbscan": HDBSCANClusterer,
}


def make_clusterer(args: Any) -> Clusterer:
    """Instantiate the clusterer named in args.clusterer (default 'kmeans')."""
    name = getattr(args, "clusterer", "kmeans")
    cls = CLUSTERER_REGISTRY.get(name)
    if cls is None:
        choices = " | ".join(sorted(CLUSTERER_REGISTRY))
        raise ValueError(f"Unknown clusterer '{name}'. Choose: {choices}")
    return cls()
