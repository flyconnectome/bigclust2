from __future__ import annotations

import numpy as np

from sklearn.neighbors import NearestNeighbors


def is_precomputed_distance_matrix(arr):
    """Heuristic: a square 2D matrix with zero diagonal is treated as precomputed pairwise distances."""
    if arr is None or not hasattr(arr, "shape") or len(arr.shape) != 2:
        return False
    if arr.shape[0] != arr.shape[1]:
        return False
    if np.diag(arr).min() != 0:
        return False
    return True


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
    Exactly one of ``distances`` or ``features`` must be provided.
    """
    emb = np.asarray(embedding, dtype=np.float64)
    if emb.ndim != 2:
        raise ValueError("`embedding` must be a 2D array-like of shape (N, D).")

    n_samples = emb.shape[0]
    if n_samples < 2:
        raise ValueError("Need at least 2 samples to compute neighborhood fidelity.")

    if (distances is None) == (features is None):
        raise ValueError("Provide exactly one of `distances` or `features`.")

    k_eff = min(int(k), n_samples - 1)
    if k_eff < 1:
        raise ValueError("`k` must be >= 1.")

    if distances is not None:
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

