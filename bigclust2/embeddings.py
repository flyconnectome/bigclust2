from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from sklearn.neighbors import NearestNeighbors

from .utils import check_finite_features


@dataclass
class KNNGraph:
    """A precomputed k-nearest-neighbors graph.

    Used as an alternative to a full pairwise distance matrix for large
    projects. Neighbors are stored as **row positions** into the metadata (the
    project file supplies these directly as 0-based ``nn_idx_*`` columns), so
    the graph stays unambiguous even when neuron IDs are duplicated. Each row is
    ordered nearest-first and excludes the point itself. Neighbors that were
    dropped by filtering/subsetting are encoded as the sentinel ``-1`` and
    left-compacted to the end of the row.

    Attributes
    ----------
    indices : np.ndarray of shape (N, k), dtype int64
        Neighbor row positions, nearest-first. ``-1`` marks a dropped neighbor
        (one that fell outside a filter/selection), left-compacted to the tail.
    dists : np.ndarray of shape (N, k), dtype float64
        Distance to each neighbor, aligned with ``indices``. For a ``-1`` slot
        this is the distance the neighbor had *before* it was dropped: edge
        consumers (sparse graph, KNN lines, fidelity) mask by index and ignore
        it, but UMAP keeps it so the local-scale estimate stays faithful to the
        original neighborhood.
    ids : np.ndarray of shape (N,)
        Neuron IDs in metadata row order (used to validate alignment).
    k : int
        Number of neighbor columns (``indices.shape[1]``).
    """

    indices: np.ndarray
    dists: np.ndarray
    ids: np.ndarray
    k: int

    def __len__(self):
        return self.indices.shape[0]


def is_knn_graph(x):
    """Whether ``x`` is a :class:`KNNGraph` instance."""
    return isinstance(x, KNNGraph)


def knn_to_sparse(indices, dists, n, *, symmetrize=True, eps=1e-9):
    """Build a sparse CSR distance graph from neighbor indices + distances.

    Parameters
    ----------
    indices : array-like of shape (N, k), int
        Neighbor row positions; ``-1`` entries are skipped (missing neighbors).
    dists : array-like of shape (N, k), float
        Distance to each neighbor. Exact zeros are replaced with ``eps`` so
        CSR's implicit-zero semantics do not silently drop those edges.
    n : int
        Number of nodes (``N``); the resulting matrix is ``(n, n)``.
    symmetrize : bool, default True
        If True, take the element-wise maximum with the transpose so the graph
        is symmetric (required by ``SpectralClustering`` and well-behaved for
        ``HDBSCAN``).
    eps : float, default 1e-9
        Minimum stored distance (see above).

    Returns
    -------
    scipy.sparse.csr_matrix of shape (n, n)
        Sparse distance graph, sorted by row values.
    """
    import scipy.sparse as sp

    indices = np.asarray(indices)
    dists = np.asarray(dists, dtype=np.float64)

    valid = indices >= 0
    rows = np.repeat(np.arange(indices.shape[0]), indices.shape[1])[valid.ravel()]
    cols = indices.ravel()[valid.ravel()].astype(np.int64)
    data = np.maximum(dists.ravel()[valid.ravel()], eps)

    G = sp.csr_matrix((data, (rows, cols)), shape=(n, n))
    if symmetrize:
        G = G.maximum(G.T)

    G = sp.csr_matrix(G)
    try:
        from sklearn.neighbors import sort_graph_by_row_values

        sort_graph_by_row_values(G, warn_when_not_sorted=False)
    except Exception:
        # Sorting is purely an efficiency optimization for downstream
        # estimators; skip it silently if the helper is unavailable.
        pass
    return G


def is_precomputed_distance_matrix(arr):
    """Heuristic: a square 2D matrix with zero diagonal is treated as precomputed pairwise distances."""
    if arr is None or not hasattr(arr, "shape") or len(arr.shape) != 2:
        return False
    if arr.shape[0] != arr.shape[1]:
        return False
    if np.diag(arr).min() != 0:
        return False
    return True


def estimate_embedding_input_bytes(data):
    """Estimate the in-memory footprint (bytes) of the embedding input.

    Cost scales very differently by source: a precomputed distance matrix is
    O(N^2), feature vectors O(N*D) and a KNN graph O(N*k). Returns ``(n, bytes)``
    where ``n`` is the number of points being embedded.
    """
    if is_knn_graph(data):
        n = len(data)
        # indices (int64) + dists (float64) -> 16 bytes per stored edge.
        return n, int(n) * int(data.k) * 16
    # ndarrays and pandas DataFrames both expose shape/size/ndim, so duck-type
    # rather than importing pandas (this module stays numpy/sklearn-only).
    if data is None or not hasattr(data, "shape") or getattr(data, "ndim", 0) < 1:
        return 0, 0
    n = int(data.shape[0])
    # data.size already equals N*N for a square distance matrix and N*D for
    # features, so no per-type branch is needed. DataFrames have no `.dtype`,
    # so itemsize falls back to 8 (float64) -- a safe, slightly high estimate.
    itemsize = getattr(getattr(data, "dtype", None), "itemsize", 8) or 8
    return n, int(data.size) * itemsize


def _normalize_feature_rows(features):
    """Row-wise normalize a numeric feature DataFrame.

    Divides each row by its total; when columns are a MultiIndex the
    normalization is applied per top-level group. Qt-free reimplementation of
    ``FeatureComparisonWidget._normalize_features`` (gui/widgets/features.py).
    """
    import pandas as pd

    if isinstance(features.columns, pd.MultiIndex):
        parts = []
        for level in features.columns.get_level_values(0).unique():
            part = (
                features.loc[:, features.columns.get_level_values(0) == level]
                .copy()
                .astype(float)
            )
            row_totals = part.sum(axis=1).replace(0, float("nan"))
            parts.append(part.div(row_totals, axis=0).fillna(0))
        return pd.concat(parts, axis=1)

    features = features.copy().astype(float)
    row_totals = features.sum(axis=1).replace(0, float("nan"))
    return features.div(row_totals, axis=0).fillna(0)


def distance_matrix_from_features(features, ids, metric="cosine", normalize=False):
    """Compute an (N, N) pairwise-distance DataFrame from a feature table.

    ``features`` is a pandas DataFrame whose rows are assumed to be in the same
    order as ``ids`` (positional alignment with the scatter point order); its
    columns may be a MultiIndex. The result is indexed by ``ids`` on both axes
    and is symmetric with a zero diagonal, so it is a drop-in replacement for a
    precomputed distance matrix.

    Rows are never reindexed by the feature table's own index: position ``i`` in
    the returned matrix corresponds to ``ids[i]`` (and to scatter point ``i``),
    which is what the heatmap's positional selection sync relies on. This mirrors
    ``DistancesTable._prepare_distances`` for ndarray inputs.

    Parameters
    ----------
    features : pandas.DataFrame
        Feature table; non-numeric and all-zero columns are dropped.
    ids : array-like
        Row labels (neuron IDs) in row order; used as both index and columns.
    metric : str
        Any metric accepted by ``sklearn.metrics.pairwise_distances``.
    normalize : bool
        If True, normalize each neuron's feature vector row-wise before
        computing distances (per top-level group when columns are a MultiIndex).
    """
    import pandas as pd
    from sklearn.metrics import pairwise_distances

    num = features.select_dtypes(include="number")

    # Drop columns that are zero for every row (mirrors
    # FeatureComparisonWidget._drop_all_zero_feature_columns); improves the
    # robustness of cosine/correlation distances.
    if num.shape[1] > 0:
        all_zero = (num.fillna(0) == 0).all(axis=0)
        if all_zero.any():
            num = num.loc[:, ~all_zero]

    if normalize and num.shape[1] > 0:
        num = _normalize_feature_rows(num)

    # Use ``.values`` positionally so the matrix follows ``ids`` row order, not
    # the feature table's own index. ``.values`` also flattens a MultiIndex.
    X = np.asarray(num.values, dtype=np.float64)
    if X.shape[1] == 0:
        raise ValueError("no numeric feature columns to compute distances from")
    if len(ids) != X.shape[0]:
        raise ValueError("ids length does not match the number of feature rows")

    check_finite_features(X, "distance matrix computation")
    D = pairwise_distances(X, metric=metric)
    # cosine/correlation can leave tiny non-zero diagonal values from float
    # error; force an exact zero diagonal so the matrix reads as precomputed
    # distances (Hide-diagonal UX and the Linkage squareform path).
    np.fill_diagonal(D, 0.0)

    index = pd.Index(ids)
    return pd.DataFrame(D, index=index, columns=index)


def rebalance_feature_matrix(arr, mode="none"):
    """Apply feature rebalancing to reduce dominance of individual dimensions."""
    data = np.asarray(arr, dtype=np.float64)
    mode = str(mode or "none").strip().lower()

    if mode in ("", "none"):
        return data

    if mode == "log1p + z-score":
        min_per_feature = np.nanmin(data, axis=0)
        shifts = np.where(min_per_feature < 0, -min_per_feature, 0.0)
        data = np.log1p(data + shifts)

    if mode in ("z-score", "log1p + z-score"):
        means = np.nanmean(data, axis=0)
        stds = np.nanstd(data, axis=0)
        stds[~np.isfinite(stds) | (stds == 0)] = 1.0
        return (data - means) / stds

    if mode == "robust (median/iqr)":
        medians = np.nanmedian(data, axis=0)
        q1 = np.nanpercentile(data, 25, axis=0)
        q3 = np.nanpercentile(data, 75, axis=0)
        iqr = q3 - q1
        iqr[~np.isfinite(iqr) | (iqr == 0)] = 1.0
        return (data - medians) / iqr

    return data


def make_embedding_estimator(
    method,
    *,
    metric,
    is_precomputed,
    random_state,
    umap_n_neighbors=10,
    umap_min_dist=0.1,
    umap_spread=1.0,
    umap_set_op_mix_ratio=0.5,
    umap_densmap=False,
    umap_dens_lambda=2.0,
    mds_n_init=4,
    mds_max_iter=300,
    mds_eps=0.001,
    tsne_perplexity=30.0,
    tsne_learning_rate=200.0,
    tsne_n_iter=1000,
):
    """Create an embedding estimator instance from standardized settings."""
    method = str(method).strip()
    method_key = method.replace("-", "").upper()
    if method_key == "UMAP":
        import umap

        return umap.UMAP(
            metric=metric,
            n_components=2,
            n_neighbors=umap_n_neighbors,
            min_dist=umap_min_dist,
            spread=umap_spread,
            set_op_mix_ratio=umap_set_op_mix_ratio,
            densmap=umap_densmap,
            dens_lambda=umap_dens_lambda,
            random_state=random_state,
        )

    if method_key == "MDS":
        from sklearn.manifold import MDS

        return MDS(
            n_components=2,
            n_init=mds_n_init,
            max_iter=mds_max_iter,
            eps=mds_eps,
            dissimilarity="precomputed" if is_precomputed else "euclidean",
            random_state=random_state,
        )

    if method_key == "PCA":
        from sklearn.decomposition import KernelPCA

        return KernelPCA(n_components=2, kernel=metric)

    if method_key == "TSNE":
        from sklearn.manifold import TSNE

        return TSNE(
            n_components=2,
            perplexity=float(tsne_perplexity),
            learning_rate=float(tsne_learning_rate),
            max_iter=int(tsne_n_iter),
            metric="precomputed" if is_precomputed else metric,
            random_state=random_state,
        )

    if method_key == "PACMAP":
        import pacmap

        return pacmap.PaCMAP(n_components=2, n_neighbors=10, MN_ratio=0.5, FP_ratio=2.0)

    raise ValueError(f"Unsupported embedding method: {method}")


def make_knn_embedding_estimator(
    method,
    *,
    knn: KNNGraph,
    random_state,
    umap_n_neighbors=10,
    umap_min_dist=0.1,
    umap_spread=1.0,
    umap_set_op_mix_ratio=0.5,
    umap_densmap=False,
    umap_dens_lambda=2.0,
    tsne_perplexity=30.0,
    tsne_learning_rate=200.0,
    tsne_n_iter=1000,
):
    """Build an embedding estimator that consumes a precomputed KNN graph.

    Only UMAP and t-SNE can be driven directly from a KNN graph; ``MDS``,
    ``PCA`` and ``PaCMAP`` require a full distance matrix or feature vectors and
    raise ``ValueError`` here.

    Returns
    -------
    tuple[estimator, fit_input]
        ``fit_input`` is the array/sparse-matrix to pass to
        ``estimator.fit_transform(...)``: a placeholder ``(N, 1)`` array for
        UMAP (which only reads ``X.shape[0]`` when ``precomputed_knn`` is set)
        or the sparse CSR distance graph for t-SNE.
    """
    method = str(method).strip()
    method_key = method.replace("-", "").upper()
    n_samples = len(knn)

    if method_key == "UMAP":
        import umap

        # n_neighbors must be < n_samples and <= the graph's k.
        n_nb = max(1, min(int(umap_n_neighbors), int(knn.k), n_samples - 1))
        knn_indices = knn.indices[:, :n_nb]

        # With no edges at all (e.g. a scattered selection whose neurons aren't
        # in each other's KNN), UMAP's fuzzy set is empty and it crashes on an
        # empty-array reduction. Fail clearly instead.
        if not bool((knn_indices >= 0).any()):
            raise ValueError(
                "The selected neurons share no nearest-neighbor links in the "
                "KNN graph, so a UMAP layout cannot be computed from it. Widen "
                "the selection, or recompute from a feature/distance source if "
                "the project provides one."
            )

        estimator = umap.UMAP(
            n_components=2,
            n_neighbors=n_nb,
            min_dist=umap_min_dist,
            spread=umap_spread,
            set_op_mix_ratio=umap_set_op_mix_ratio,
            densmap=umap_densmap,
            dens_lambda=umap_dens_lambda,
            precomputed_knn=(knn_indices, knn.dists[:, :n_nb]),
            random_state=random_state,
        )
        # UMAP only uses X for its row count when precomputed_knn is given.
        fit_input = np.zeros((n_samples, 1), dtype=np.float32)
        return estimator, fit_input

    if method_key == "TSNE":
        from sklearn.manifold import TSNE

        graph = knn_to_sparse(knn.indices, knn.dists, n_samples, symmetrize=True)

        # sklearn's t-SNE requests n_neighbors = int(3*perplexity + 1) and the
        # precomputed-graph check additionally reserves one slot for the point
        # itself, so each row must store at least n_neighbors + 1 edges. Cap
        # perplexity against the *actual* minimum row degree (which can be < k
        # once filtering drops neighbors) to stay safely under that bound.
        min_nnz = int(np.diff(graph.indptr).min()) if graph.nnz else 0
        max_neighbors = max(1, min_nnz - 1)
        max_perp = max(1.0, (max_neighbors - 1) / 3.0)
        perplexity = min(float(tsne_perplexity), max_perp)

        # sklearn requests int(3*perplexity + 1) neighbors plus a self slot. If
        # the sparsest row can't supply that even at the minimum perplexity, the
        # graph is too sparse for t-SNE (e.g. isolated neurons); fail clearly
        # rather than letting sklearn raise a cryptic error.
        required = int(3.0 * perplexity + 1) + 1
        if required > min_nnz:
            raise ValueError(
                "t-SNE cannot run on this KNN graph: some neurons have too few "
                f"neighbors (need at least {required}, the sparsest has "
                f"{min_nnz}). Use UMAP instead, widen the selection/filter, or "
                "rebuild the KNN graph with a larger k."
            )

        estimator = TSNE(
            n_components=2,
            perplexity=perplexity,
            learning_rate=float(tsne_learning_rate),
            max_iter=int(tsne_n_iter),
            metric="precomputed",
            # "pca" init is incompatible with metric="precomputed".
            init="random",
            random_state=random_state,
        )
        return estimator, graph

    raise ValueError(
        f"Embedding method '{method}' cannot be computed from a KNN graph. "
        "Only UMAP and t-SNE are supported; MDS/PCA/PaCMAP require a full "
        "distance matrix or feature vectors."
    )


def sanitize_embedding(xy):
    """Relocate non-finite embedding rows to the layout periphery.

    UMAP/t-SNE leave fully disconnected vertices (neurons whose KNN neighbors
    all fall outside the current subset/selection, so they have no edges at all)
    at ``NaN``. A single non-finite row poisons the whole layout downstream: the
    shared-frame normalization computes ``min``/``max`` over every point, so one
    ``NaN`` turns *all* coordinates into ``NaN`` and the entire scatter vanishes.

    To keep the layout usable, the bad rows are replaced with finite coordinates
    fanned out around the edge of the placed (finite) points -- they read as
    outliers, matching how UMAP normally banishes disconnected vertices, instead
    of corrupting the embedding.

    Parameters
    ----------
    xy : array-like of shape (N, 2)
        Freshly computed embedding coordinates.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        ``(xy_clean, bad_mask)`` -- a float64 copy with non-finite rows relocated,
        and the boolean mask (shape ``(N,)``) of the rows that were relocated.
    """
    xy = np.array(xy, dtype=np.float64, copy=True)
    bad = ~np.isfinite(xy).all(axis=1)
    if not bad.any():
        return xy, bad

    good = ~bad
    if good.any():
        lo = xy[good].min(axis=0)
        hi = xy[good].max(axis=0)
        center = (lo + hi) / 2.0
        span = float((hi - lo).max())
        if not np.isfinite(span) or span <= 0:
            span = 1.0
    else:
        center = np.zeros(xy.shape[1])
        span = 1.0

    # Fan the disconnected points out around the periphery so they don't stack
    # on top of one another.
    n_bad = int(bad.sum())
    angles = 2.0 * np.pi * np.arange(n_bad) / n_bad
    radius = span * 0.65
    offsets = np.column_stack([np.cos(angles), np.sin(angles)]) * radius
    xy[bad] = center + offsets
    return xy, bad


def prepare_embedding_input(
    data,
    *,
    is_precomputed,
    method,
    metric,
    rebalance_mode="none",
    pca_n_components=None,
    random_state=None,
):
    """Prepare input matrix for embedding based on configured preprocessing."""
    arr = np.asarray(data, dtype=np.float64)

    if not is_precomputed:
        arr = rebalance_feature_matrix(arr, mode=rebalance_mode)
        # Checked post-rebalance: rebalancing is NaN-aware but never fills, so
        # missing values in the input survive it.
        check_finite_features(arr, "embedding computation")

    if (not is_precomputed) and (pca_n_components is not None):
        from sklearn.decomposition import PCA

        pca = PCA(n_components=int(pca_n_components), random_state=random_state)
        arr = pca.fit_transform(arr)

    # MDS with feature vectors supports Euclidean dissimilarity directly.
    # For other metrics, convert features to pairwise distances first.
    if method == "MDS" and (not is_precomputed) and metric != "euclidean":
        from sklearn.metrics import pairwise_distances

        arr = pairwise_distances(arr, metric=metric)

    return arr


def _knn_from_features(data, k, metric, algorithm="auto"):
    """Return k-nearest neighbor indices per row for feature vectors."""
    check_finite_features(data, "nearest-neighbor computation")
    # Ask for one extra neighbor to account for the point itself.
    nbrs = NearestNeighbors(
        n_neighbors=min(k + 1, data.shape[0]),
        metric=metric,
        algorithm=algorithm,
    )
    nbrs.fit(data)
    _, idx = nbrs.kneighbors(data)

    # Remove self-neighbor robustly (not always guaranteed to be at position 0).
    out = np.empty((data.shape[0], k), dtype=np.int64)
    for i in range(data.shape[0]):
        row = idx[i]
        row = row[row != i]
        if row.shape[0] < k:
            # Extremely rare (e.g. heavy duplication edge cases): pad deterministically.
            pad = np.full(k - row.shape[0], row[-1] if row.shape[0] else i, dtype=np.int64)
            row = np.concatenate([row, pad])
        out[i] = row[:k]

    return out


def neighborhood_fidelity(
    embedding,
    *,
    distances=None,
    features=None,
    knn_neighbors=None,
    metric="euclidean",
    k=15,
    rank=False,
):
    """Compute per-point neighborhood preservation fidelity for an embedding.

    Parameters
    ----------
    embedding : array-like of shape (N, D_emb)
        Embedded coordinates (typically ``(N, 2)`` for UMAP scatter).
    distances : array-like of shape (N, N), optional
        Precomputed pairwise distances in the original space.
    features : array-like of shape (N, D_feat), optional
        Feature matrix in the original space. If provided, pairwise distances are
        computed using ``metric``.
    knn_neighbors : array-like of shape (N, k_graph), optional
        Precomputed nearest-neighbor row positions (nearest-first), e.g. from a
        :class:`KNNGraph`. The high-dimensional neighbor set is read directly
        from the first ``k`` columns; no distance matrix is needed.
    metric : str, default "euclidean"
        Distance metric used when ``features`` are supplied.
    k : int, default 15
        Number of nearest neighbors used for preservation scoring.
    rank : bool, default False
        If False, compute overlap fidelity ``|N_high(i) ∩ N_emb(i)| / k``.
        If True, compute a rank-aware fidelity that multiplies overlap
        coverage by agreement in neighbor ordering within the top-k sets.

    Returns
    -------
    np.ndarray of shape (N,)
        Per-point neighborhood fidelity score in ``[0, 1]``.

    Notes
    -----
    Exactly one of ``distances``, ``features`` or ``knn_neighbors`` must be
    provided.
    """
    emb = np.asarray(embedding, dtype=np.float64)
    if emb.ndim != 2:
        raise ValueError("`embedding` must be a 2D array-like of shape (N, D).")

    n_samples = emb.shape[0]
    if n_samples < 2:
        raise ValueError("Need at least 2 samples to compute neighborhood fidelity.")

    if sum(x is not None for x in (distances, features, knn_neighbors)) != 1:
        raise ValueError(
            "Provide exactly one of `distances`, `features` or `knn_neighbors`."
        )

    k_eff = min(int(k), n_samples - 1)
    if knn_neighbors is not None:
        k_eff = min(k_eff, np.asarray(knn_neighbors).shape[1])
    if k_eff < 1:
        raise ValueError("`k` must be >= 1.")

    if knn_neighbors is not None:
        # High-dim neighbors come straight from the precomputed graph; the
        # graph is nearest-first so the leading columns are the top-k.
        high_neighbors = np.asarray(knn_neighbors)[:, :k_eff]
        if high_neighbors.shape[0] != n_samples:
            raise ValueError(
                "`embedding` and `knn_neighbors` must contain the same number of samples."
            )
    elif distances is not None:
        high_d = np.asarray(distances, dtype=np.float64)
        if high_d.ndim != 2 or high_d.shape[0] != high_d.shape[1]:
            raise ValueError("`distances` must be a square matrix of shape (N, N).")
        if high_d.shape[0] != n_samples:
            raise ValueError(
                "`embedding` and `distances` must contain the same number of samples."
            )

        # Extract neighbors directly from precomputed distances.
        high_d = high_d.copy()
        np.fill_diagonal(high_d, np.inf)
        high_neighbors = np.argpartition(high_d, kth=k_eff - 1, axis=1)[:, :k_eff]
    else:
        feats = np.asarray(features, dtype=np.float64)
        if feats.ndim != 2:
            raise ValueError("`features` must be a 2D array-like of shape (N, D).")
        if feats.shape[0] != n_samples:
            raise ValueError(
                "`embedding` and `features` must contain the same number of samples."
            )

        # Use tree search depending on metric support.
        high_neighbors = _knn_from_features(feats, k_eff, metric=metric, algorithm="auto")

    # Embedding space is Euclidean: tree-based kNN avoids full O(N^2) matrix.
    emb_neighbors = _knn_from_features(emb, k_eff, metric="euclidean", algorithm="kd_tree")

    scores = np.empty(n_samples, dtype=np.float64)
    for i in range(n_samples):
        shared = np.intersect1d(
            high_neighbors[i],
            emb_neighbors[i],
            assume_unique=False,
        )

        if not rank:
            scores[i] = shared.size / k_eff
            continue

        if shared.size == 0:
            scores[i] = 0.0
            continue

        high_rank = {int(nei): pos for pos, nei in enumerate(high_neighbors[i])}
        emb_rank = {int(nei): pos for pos, nei in enumerate(emb_neighbors[i])}
        diffs = np.array(
            [abs(high_rank[int(nei)] - emb_rank[int(nei)]) for nei in shared],
            dtype=np.float64,
        )
        denom = max(k_eff - 1, 1)
        rank_agreement = 1.0 - (diffs.mean() / denom)
        overlap_coverage = shared.size / k_eff
        scores[i] = overlap_coverage * rank_agreement

    return scores


def cluster_missing_member_scores(
    selection,
    *,
    distances=None,
    features=None,
    metric="euclidean",
    k=15,
    threshold=0.6,
):
    """Score likely missing members and estimate cluster completeness.

    Parameters
    ----------
    selection : array-like of int or bool mask, shape (N,) or (n_selected,)
        Candidate cluster membership. Can be a boolean mask over all neurons or
        a list/array of selected neuron indices.
    distances : array-like of shape (N, N), optional
        Precomputed pairwise distances in the original space.
    features : array-like of shape (N, D), optional
        Feature matrix in the original space. If provided, nearest neighbors are
        computed using ``metric``.
    metric : str, default "euclidean"
        Distance metric used when ``features`` are supplied.
    k : int, default 15
        Number of nearest neighbors used to compute missing-member scores.
    threshold : float, default 0.6
        Candidate threshold for the completeness proxy. Unselected neurons with
        score >= ``threshold`` are counted as likely missing members.

    Returns
    -------
    tuple[np.ndarray, float]
        ``(scores, completeness_proxy)`` where:
        - ``scores`` is shape ``(N,)`` and gives, per neuron, the fraction of its
          top-k neighbors that are in ``selection``.
        - ``completeness_proxy`` is
          ``|S| / (|S| + |{j not in S: score_j >= threshold}|)``.

    Notes
    -----
    Exactly one of ``distances`` or ``features`` must be provided.
    The missing-member interpretation applies to unselected neurons.
    """
    if (distances is None) == (features is None):
        raise ValueError("Provide exactly one of `distances` or `features`.")

    if not (0.0 <= float(threshold) <= 1.0):
        raise ValueError("`threshold` must be between 0 and 1.")

    if distances is not None:
        high_d = np.asarray(distances, dtype=np.float64)
        if high_d.ndim != 2 or high_d.shape[0] != high_d.shape[1]:
            raise ValueError("`distances` must be a square matrix of shape (N, N).")
        n_samples = high_d.shape[0]
    else:
        feats = np.asarray(features, dtype=np.float64)
        if feats.ndim != 2:
            raise ValueError("`features` must be a 2D array-like of shape (N, D).")
        n_samples = feats.shape[0]

    if n_samples < 2:
        raise ValueError("Need at least 2 samples to compute missing-member scores.")

    sel = np.asarray(selection)
    if sel.dtype == bool:
        if sel.ndim != 1 or sel.shape[0] != n_samples:
            raise ValueError("Boolean `selection` mask must have shape (N,).")
        selected_mask = sel.copy()
    else:
        sel = np.asarray(selection, dtype=np.int64).ravel()
        if sel.size == 0:
            raise ValueError("`selection` must contain at least one neuron.")
        if np.any(sel < 0) or np.any(sel >= n_samples):
            raise ValueError("`selection` contains indices outside [0, N).")
        selected_mask = np.zeros(n_samples, dtype=bool)
        selected_mask[sel] = True

    if not np.any(selected_mask):
        raise ValueError("`selection` must contain at least one neuron.")

    k_eff = min(int(k), n_samples - 1)
    if k_eff < 1:
        raise ValueError("`k` must be >= 1.")

    if distances is not None:
        # Extract top-k neighbors directly from precomputed distances.
        high_d = high_d.copy()
        np.fill_diagonal(high_d, np.inf)
        neighbors = np.argpartition(high_d, kth=k_eff - 1, axis=1)[:, :k_eff]
    else:
        neighbors = _knn_from_features(feats, k_eff, metric=metric, algorithm="auto")

    scores = selected_mask[neighbors].mean(axis=1, dtype=np.float64)

    outside_selection = ~selected_mask
    missing_candidates = outside_selection & (scores >= float(threshold))
    n_selected = int(selected_mask.sum())
    n_missing_candidates = int(missing_candidates.sum())
    completeness_proxy = n_selected / (n_selected + n_missing_candidates)

    return scores, float(completeness_proxy)


def selection_silhouette_scores(
    selection,
    *,
    distances=None,
    features=None,
    metric="euclidean",
    candidate_threshold=0.0,
):
    """Compute silhouette for a selected cluster and suggest missing candidates.

    Parameters
    ----------
    selection : array-like of int or bool mask, shape (N,) or (n_selected,)
        Selected cluster membership as boolean mask or selected indices.
    distances : array-like of shape (N, N), optional
        Precomputed pairwise distances in the original space.
    features : array-like of shape (N, D), optional
        Feature matrix in the original space. If provided, pairwise distances are
        computed using ``metric``.
    metric : str, default "euclidean"
        Distance metric used when ``features`` are supplied.
    candidate_threshold : float, default 0.0
        Unselected neurons with hypothetical silhouette > threshold are returned
        as candidate missing members.

    Returns
    -------
    tuple[np.ndarray, float, np.ndarray, np.ndarray]
        ``(selected_scores, selected_mean, candidate_indices, candidate_scores)`` where:
        - ``selected_scores`` are per-neuron silhouette scores for selected neurons.
        - ``selected_mean`` is the mean silhouette of selected neurons.
        - ``candidate_indices`` are unselected neuron indices whose hypothetical
          silhouette as selected exceeds ``candidate_threshold``.
        - ``candidate_scores`` are the corresponding hypothetical silhouettes.

    Notes
    -----
    This function treats the problem as a binary partition: selected vs unselected.
    For unselected candidates, silhouette is computed under a hypothetical move into
    the selected cluster while all other assignments remain unchanged.
    """
    if (distances is None) == (features is None):
        raise ValueError("Provide exactly one of `distances` or `features`.")

    if distances is not None:
        dmat = np.asarray(distances, dtype=np.float64)
        if dmat.ndim != 2 or dmat.shape[0] != dmat.shape[1]:
            raise ValueError("`distances` must be a square matrix of shape (N, N).")
        n_samples = dmat.shape[0]
    else:
        feats = np.asarray(features, dtype=np.float64)
        if feats.ndim != 2:
            raise ValueError("`features` must be a 2D array-like of shape (N, D).")
        n_samples = feats.shape[0]
        from sklearn.metrics import pairwise_distances

        check_finite_features(feats, "silhouette computation")
        dmat = pairwise_distances(feats, metric=metric)

    if n_samples < 2:
        raise ValueError("Need at least 2 samples to compute silhouette scores.")

    sel = np.asarray(selection)
    if sel.dtype == bool:
        if sel.ndim != 1 or sel.shape[0] != n_samples:
            raise ValueError("Boolean `selection` mask must have shape (N,).")
        selected_mask = sel.copy()
    else:
        sel = np.asarray(selection, dtype=np.int64).ravel()
        if sel.size == 0:
            raise ValueError("`selection` must contain at least one neuron.")
        if np.any(sel < 0) or np.any(sel >= n_samples):
            raise ValueError("`selection` contains indices outside [0, N).")
        selected_mask = np.zeros(n_samples, dtype=bool)
        selected_mask[sel] = True

    n_selected = int(selected_mask.sum())
    n_unselected = int((~selected_mask).sum())
    if n_selected < 2:
        raise ValueError("Need at least 2 selected neurons for silhouette.")
    if n_unselected < 1:
        raise ValueError("Need at least 1 unselected neuron for silhouette.")

    # Ignore self-distances for all mean calculations.
    dmat = dmat.copy()
    np.fill_diagonal(dmat, np.nan)

    selected_ix = np.where(selected_mask)[0]
    unselected_ix = np.where(~selected_mask)[0]

    # Silhouette for currently selected neurons.
    selected_scores = np.empty(n_selected, dtype=np.float64)
    for out_i, i in enumerate(selected_ix):
        a_i = np.nanmean(dmat[i, selected_ix])
        b_i = np.nanmean(dmat[i, unselected_ix])
        denom = max(a_i, b_i)
        selected_scores[out_i] = 0.0 if denom <= 0 else (b_i - a_i) / denom

    selected_mean = float(np.nanmean(selected_scores))

    # Hypothetical silhouette if each unselected neuron were moved into selection.
    candidate_scores = np.empty(unselected_ix.shape[0], dtype=np.float64)
    for out_j, j in enumerate(unselected_ix):
        a_j = np.nanmean(dmat[j, selected_ix])

        # In the hypothetical move, j leaves unselected cluster.
        rest_unselected = unselected_ix[unselected_ix != j]
        if rest_unselected.size == 0:
            b_j = np.nan
        else:
            b_j = np.nanmean(dmat[j, rest_unselected])

        if not np.isfinite(a_j) or not np.isfinite(b_j):
            candidate_scores[out_j] = np.nan
            continue

        denom = max(a_j, b_j)
        candidate_scores[out_j] = 0.0 if denom <= 0 else (b_j - a_j) / denom

    keep = np.isfinite(candidate_scores) & (candidate_scores > float(candidate_threshold))
    return (
        selected_scores,
        selected_mean,
        unselected_ix[keep],
        candidate_scores[keep],
    )

