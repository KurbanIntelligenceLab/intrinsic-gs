"""Spectral graph primitives shared between spectral_cluster.py and clusterers.py.

Pure numpy / scipy / sklearn / torch — no GPU 3D pipeline dependencies (no
plyfile, no scene, no gaussian_renderer). Safe to import in unit tests that
only need the math.

Public surface:
  symmetrize(edge_index, W, N) -> scipy.sparse.csr_matrix
  normalized_laplacian(W_sym)  -> scipy.sparse.csr_matrix (returns A_norm)
  spectral_embed(A_norm, n_clusters, eigengap_k, solver) -> (embedding, k_used, stats)
  run_kmeans(embedding, n_clusters, seed=0)              -> (labels, sizes)
  plot_eigengap(eigenvalues, gaps, suggested_k, requested_k, path='/tmp/eigengap.png')
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize


def symmetrize(edge_index, W, N):
    """Convert directed k-NN edges to a symmetric sparse matrix.

    W_sym[i,j] = W_sym[j,i] = max(W[i,j], W[j,i]).
    """
    i = edge_index[0].cpu().numpy()
    j = edge_index[1].cpu().numpy()
    w = W.cpu().float().numpy()
    W_ij = sp.csr_matrix((w, (i, j)), shape=(N, N))
    W_ji = sp.csr_matrix((w, (j, i)), shape=(N, N))
    W_sym = W_ij.maximum(W_ji)
    W_sym.eliminate_zeros()
    return W_sym


def normalized_laplacian(W_sym):
    """A_norm = D^{-1/2} W D^{-1/2}.

    Eigenvectors of L_sym = I - A_norm bottom-k ↔ eigenvectors of A_norm top-k.
    """
    d = np.asarray(W_sym.sum(axis=1)).flatten()
    d_inv_sqrt = np.where(d > 0, 1.0 / np.sqrt(d), 0.0)
    D_inv_sqrt = sp.diags(d_inv_sqrt)
    return D_inv_sqrt @ W_sym @ D_inv_sqrt


def _eigsh_arpack(A_norm, k):
    print(f"  Computing top-{k} eigenvectors (ARPACK / CPU) ...")
    eigenvalues, eigenvectors = spla.eigsh(A_norm, k=k, which="LM")
    return eigenvalues, eigenvectors


def _eigsh_randomized(A_norm, k, n_iter=10, random_state=42):
    from sklearn.utils.extmath import randomized_svd
    print(f"  Computing top-{k} eigenvectors (randomized SVD / CPU) ...")
    U, S, _ = randomized_svd(
        A_norm, n_components=k, n_iter=n_iter,
        n_oversamples=20, random_state=random_state,
    )
    return S, U


def _eigsh_lobpcg(A_norm, k, device="cuda"):
    import torch
    print(f"  Computing top-{k} eigenvectors (torch.lobpcg / GPU) ...")
    N = A_norm.shape[0]
    M = A_norm.tocsr().astype(np.float32)
    crow = torch.from_numpy(M.indptr.copy().astype(np.int64)).to(device)
    col = torch.from_numpy(M.indices.copy().astype(np.int64)).to(device)
    val = torch.from_numpy(M.data.copy().astype(np.float32)).to(device)
    A_t = torch.sparse_csr_tensor(crow, col, val, size=(N, N), device=device)
    X0 = torch.randn(N, k, dtype=torch.float32, device=device)
    eigenvalues_t, eigenvectors_t = torch.lobpcg(
        A_t, k=k, X=X0, largest=True, niter=1000, tol=1e-5
    )
    return eigenvalues_t.cpu().numpy(), eigenvectors_t.cpu().numpy()


def _eigsh_cupy(A_norm, k, maxiter=10000, tol=0.0, ncv=None, seed=0):
    try:
        import cupy as cp
        import cupyx.scipy.sparse as cpsp
        import cupyx.scipy.sparse.linalg as cpsla
    except ImportError as exc:
        raise ImportError(
            "cupy not found. Install with:  pip install cupy-cuda118\n"
            "Or use --solver arpack (CPU)."
        ) from exc

    if ncv is None:
        ncv = min(4 * k, A_norm.shape[0] - 1)
    print(f"  Computing top-{k} eigenvectors (cupyx eigsh / GPU, "
          f"ncv={ncv}, maxiter={maxiter}, tol={tol:.0e}) ...")
    A_cp = cpsp.csr_matrix(A_norm.astype(np.float32))
    cp.random.seed(seed)
    eigenvalues, eigenvectors = cpsla.eigsh(
        A_cp, k=k, which="LA", ncv=ncv, maxiter=maxiter, tol=tol,
    )
    resid = cp.linalg.norm(A_cp @ eigenvectors - eigenvectors * eigenvalues, axis=0)
    print(f"  Residuals: max={float(resid.max()):.2e}, "
          f"mean={float(resid.mean()):.2e}, min={float(resid.min()):.2e}")
    return eigenvalues.get(), eigenvectors.get()


def spectral_embed(A_norm, n_clusters, eigengap_k=15, solver="cupy"):
    """Top-k eigenvectors of A_norm, with eigengap-based auto-k selection.

    Args:
      n_clusters: target k, or None to auto-pick via eigengap.
      eigengap_k: number of eigenvalues to compute for the eigengap window.
      solver: 'lobpcg' | 'cupy' | 'randomized' | 'arpack'.

    Returns:
      (embedding [N, k_used] float32, k_used int, stats dict).
    """
    auto_k = n_clusters is None
    k_compute = eigengap_k if auto_k else max(n_clusters, eigengap_k)

    if solver == "lobpcg":
        eigenvalues, eigenvectors = _eigsh_lobpcg(A_norm, k_compute)
    elif solver == "cupy":
        eigenvalues, eigenvectors = _eigsh_cupy(A_norm, k_compute)
    elif solver == "randomized":
        eigenvalues, eigenvectors = _eigsh_randomized(A_norm, k_compute)
    elif solver == "arpack":
        eigenvalues, eigenvectors = _eigsh_arpack(A_norm, k_compute)
    else:
        raise ValueError(
            f"Unknown solver '{solver}'. Choose: lobpcg | cupy | randomized | arpack"
        )

    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]

    gaps = eigenvalues[:-1] - eigenvalues[1:]
    suggested_k = int(np.argmax(gaps)) + 1

    sorted_gaps = np.sort(gaps)[::-1]
    if len(sorted_gaps) >= 2 and sorted_gaps[1] > 1e-12:
        rho = float(sorted_gaps[0] / sorted_gaps[1])
    else:
        rho = float("inf")

    k_used = suggested_k if auto_k else n_clusters

    print(f"  Eigenvalues (top-{k_compute}): {eigenvalues.round(4).tolist()}")
    print(f"  Eigengaps:                     {gaps.round(4).tolist()}")
    print(f"  Eigengap suggests k = {suggested_k}  "
          f"({'AUTO → using ' + str(k_used) if auto_k else 'requested k = ' + str(n_clusters)})")
    print(f"  ρ = δ_max/δ_2nd = {rho:.3f}  (higher ⇒ more decisive cluster count)")

    spectral_stats = {
        "eigenvalues": eigenvalues.tolist(),
        "eigengaps": gaps.tolist(),
        "suggested_k": suggested_k,
        "rho": rho,
        "solver": solver,
        "k_compute": k_compute,
    }

    embedding = normalize(eigenvectors[:, :k_used], norm="l2")
    return embedding.astype(np.float32), k_used, spectral_stats


def plot_eigengap(eigenvalues, gaps, suggested_k, requested_k, path="/tmp/eigengap.png"):
    """Eigengap diagnostic plot (kmeans-only). Saved to `path`."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    fig.suptitle("Eigengap Heuristic", fontsize=12)

    ks = np.arange(1, len(eigenvalues) + 1)
    ax1.plot(ks, eigenvalues, "o-", markersize=4)
    ax1.axvline(suggested_k, color="red", linestyle="--",
                label=f"suggested k={suggested_k}")
    ax1.axvline(requested_k, color="orange", linestyle="--",
                label=f"requested k={requested_k}")
    ax1.set_xlabel("k"); ax1.set_ylabel("Eigenvalue"); ax1.set_title("Eigenvalues")
    ax1.legend(fontsize=8)

    gap_ks = np.arange(1, len(gaps) + 1)
    ax2.bar(gap_ks, gaps, color="steelblue", alpha=0.8)
    ax2.axvline(suggested_k, color="red", linestyle="--",
                label=f"suggested k={suggested_k}")
    ax2.axvline(requested_k, color="orange", linestyle="--",
                label=f"requested k={requested_k}")
    ax2.set_xlabel("k"); ax2.set_ylabel("Gap (λ_k − λ_{k+1})"); ax2.set_title("Eigengaps")
    ax2.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Eigengap plot (tmp): {path}")


def run_kmeans(embedding, n_clusters, seed=0):
    """K-means on spectral embedding. Returns (labels [N], sizes desc list)."""
    print(f"  Running k-means (k={n_clusters})...")
    km = KMeans(n_clusters=n_clusters, random_state=seed, n_init="auto", max_iter=300)
    labels = km.fit_predict(embedding)
    counts = np.bincount(labels, minlength=n_clusters)
    sizes = sorted(counts.tolist(), reverse=True)
    print(f"  Cluster sizes: {sizes}")
    return labels, sizes
