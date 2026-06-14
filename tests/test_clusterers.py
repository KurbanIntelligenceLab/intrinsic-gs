"""Unit tests for the pluggable clusterer infrastructure.

Covers factory dispatch, the Clusterer interface contract, and
KMeansClusterer's wiring to the existing spectral_cluster primitives.

Leiden / HDBSCAN integration smoke tests are gated on the optional
libraries being installed; if absent, those tests are skipped.
"""
from __future__ import annotations

import sys
from argparse import Namespace
from pathlib import Path

import numpy as np
import pytest
import scipy.sparse as sp


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from self_supervised_scripts.clusterers import (  # noqa: E402
    Clusterer,
    HDBSCANClusterer,
    KMeansClusterer,
    LeidenClusterer,
    make_clusterer,
)


def _toy_block_graph(n_per_block: int = 20, n_blocks: int = 3, intra_w: float = 1.0,
                    inter_w: float = 0.01, seed: int = 0):
    """Build a small symmetric sparse W with clear block structure.

    Returns (W_sym, A_norm) where A_norm = D^{-1/2} W D^{-1/2}.
    """
    rng = np.random.default_rng(seed)
    n = n_per_block * n_blocks
    rows, cols, vals = [], [], []
    for b in range(n_blocks):
        start = b * n_per_block
        for i in range(start, start + n_per_block):
            for j in range(i + 1, start + n_per_block):
                rows.append(i)
                cols.append(j)
                vals.append(intra_w)
    # A few weak inter-block edges so the graph is connected.
    for b in range(n_blocks - 1):
        for _ in range(2):
            i = rng.integers(b * n_per_block, (b + 1) * n_per_block)
            j = rng.integers((b + 1) * n_per_block, (b + 2) * n_per_block)
            rows.append(int(i))
            cols.append(int(j))
            vals.append(inter_w)
    W_upper = sp.csr_matrix((vals, (rows, cols)), shape=(n, n))
    W_sym = (W_upper + W_upper.T).tocsr()
    d = np.asarray(W_sym.sum(axis=1)).flatten()
    d_inv_sqrt = np.where(d > 0, 1.0 / np.sqrt(d), 0.0)
    D = sp.diags(d_inv_sqrt)
    A_norm = (D @ W_sym @ D).tocsr()
    return W_sym, A_norm


def _kmeans_args(n_clusters=3, eigengap_k=5, solver="arpack"):
    return Namespace(
        clusterer="kmeans",
        n_clusters=n_clusters,
        eigengap_k=eigengap_k,
        solver=solver,
    )


# ── Factory ────────────────────────────────────────────────────────────────────


def test_make_clusterer_default_is_kmeans():
    args = Namespace()
    c = make_clusterer(args)
    assert isinstance(c, KMeansClusterer)
    assert c.name == "kmeans"


def test_make_clusterer_dispatches_all_three():
    for name, cls in [("kmeans", KMeansClusterer),
                      ("leiden", LeidenClusterer),
                      ("hdbscan", HDBSCANClusterer)]:
        c = make_clusterer(Namespace(clusterer=name))
        assert isinstance(c, cls)
        assert c.name == name


def test_make_clusterer_rejects_unknown():
    with pytest.raises(ValueError, match="Unknown clusterer"):
        make_clusterer(Namespace(clusterer="meanshift"))


def test_clusterer_is_abstract():
    with pytest.raises(TypeError):
        Clusterer()  # type: ignore[abstract]


# ── Output-dir prefix ──────────────────────────────────────────────────────────


def test_kmeans_output_prefix_uses_actual_k():
    args = Namespace(n_clusters=14)
    assert KMeansClusterer().output_prefix(args, k_actual=14) == "spectral_k14"
    assert KMeansClusterer().output_prefix(args, k_actual=7) == "spectral_k7"


def test_leiden_output_prefix_includes_resolution():
    args = Namespace(leiden_resolution=1.0)
    assert LeidenClusterer().output_prefix(args, k_actual=5) == "leiden_res1.0"


def test_hdbscan_output_prefix_includes_min_cluster_size_frac():
    args = Namespace(hdbscan_min_cluster_size_frac=0.005, hdbscan_min_samples=5)
    assert (
        HDBSCANClusterer().output_prefix(args, k_actual=4)
        == "hdbscan_mcs0.005_ms5"
    )


# ── KMeansClusterer integration ────────────────────────────────────────────────


def test_kmeans_clusterer_recovers_block_structure():
    """End-to-end: 3 blocks → 3 clusters, no points lost."""
    W_sym, A_norm = _toy_block_graph(n_per_block=20, n_blocks=3)
    args = _kmeans_args(n_clusters=3)
    labels, stats = KMeansClusterer().fit(W_sym, A_norm, args)

    assert labels.shape == (60,)
    assert labels.dtype == np.int32
    assert set(labels.tolist()) == {0, 1, 2}
    assert stats["k_used"] == 3
    assert sum(stats["cluster_sizes"]) == 60
    # Each block should fall mostly into a single cluster (>= 18/20).
    block_majority = []
    for b in range(3):
        block_lbls = labels[b * 20:(b + 1) * 20]
        majority = np.bincount(block_lbls).max()
        block_majority.append(majority)
    assert min(block_majority) >= 18, f"Block purity too low: {block_majority}"


def test_kmeans_clusterer_stats_carries_spectral_block():
    W_sym, A_norm = _toy_block_graph()
    args = _kmeans_args(n_clusters=3)
    _labels, stats = KMeansClusterer().fit(W_sym, A_norm, args)
    spec = stats["spectral"]
    assert "eigenvalues" in spec
    assert "eigengaps" in spec
    assert spec["solver"] == "arpack"
    assert spec["k_compute"] >= 3


def test_kmeans_report_section_has_eigenvalues_table():
    W_sym, A_norm = _toy_block_graph()
    args = _kmeans_args(n_clusters=3)
    _labels, stats = KMeansClusterer().fit(W_sym, A_norm, args)
    section = KMeansClusterer().report_section(stats, args)
    text = "\n".join(section)
    assert "Spectral Analysis" in text
    assert "Eigenvalues" in text
    assert "ρ = δ_max/δ_2nd" in text


# ── LeidenClusterer integration (skipped if leidenalg missing) ─────────────────


leiden_or_skip = pytest.importorskip


def test_leiden_clusterer_finds_block_communities():
    leiden_or_skip("leidenalg")
    leiden_or_skip("igraph")
    W_sym, A_norm = _toy_block_graph(n_per_block=20, n_blocks=3, inter_w=0.001)
    args = Namespace(clusterer="leiden", leiden_resolution=1.0)
    labels, stats = LeidenClusterer().fit(W_sym, A_norm, args)
    assert labels.shape == (60,)
    assert labels.dtype == np.int32
    # Should find ~3 communities; allow 2-4 because of small-graph noise.
    assert 2 <= stats["k_used"] <= 4
    assert "modularity_q" in stats
    assert stats["resolution"] == 1.0


def test_leiden_report_section_mentions_modularity():
    leiden_or_skip("leidenalg")
    leiden_or_skip("igraph")
    W_sym, A_norm = _toy_block_graph()
    args = Namespace(clusterer="leiden", leiden_resolution=1.0)
    _labels, stats = LeidenClusterer().fit(W_sym, A_norm, args)
    section = LeidenClusterer().report_section(stats, args)
    text = "\n".join(section)
    assert "Leiden" in text
    assert "Modularity Q" in text
    assert "Resolution" in text


# ── HDBSCANClusterer integration (skipped if hdbscan missing) ──────────────────


def test_hdbscan_clusterer_assigns_every_point():
    leiden_or_skip("hdbscan")
    W_sym, A_norm = _toy_block_graph(n_per_block=30, n_blocks=3)
    args = Namespace(
        clusterer="hdbscan",
        eigengap_k=5,
        solver="arpack",
        hdbscan_min_cluster_size_frac=0.1,
        hdbscan_min_samples=3,
    )
    labels, stats = HDBSCANClusterer().fit(W_sym, A_norm, args)
    assert labels.shape == (90,)
    assert labels.dtype == np.int32
    # Noise reassignment guarantee: every point has a non-negative label.
    assert int(labels.min()) >= 0
    assert stats["k_used"] >= 1


def test_hdbscan_report_section_lists_params():
    leiden_or_skip("hdbscan")
    W_sym, A_norm = _toy_block_graph(n_per_block=30, n_blocks=3)
    args = Namespace(
        clusterer="hdbscan",
        eigengap_k=5,
        solver="arpack",
        hdbscan_min_cluster_size_frac=0.1,
        hdbscan_min_samples=3,
    )
    _labels, stats = HDBSCANClusterer().fit(W_sym, A_norm, args)
    section = HDBSCANClusterer().report_section(stats, args)
    text = "\n".join(section)
    assert "HDBSCAN" in text
    assert "min_cluster_size" in text
    assert "min_samples" in text
