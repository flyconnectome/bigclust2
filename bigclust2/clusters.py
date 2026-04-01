"""Clustering algorithms for bigclust2."""

from __future__ import annotations

import warnings

import numpy as np


def run_clustering(
    data,
    *,
    method: str,
    is_precomputed: bool,
    metric: str = "euclidean",
    hdbscan_min_cluster_size: int = 5,
    hdbscan_min_samples: int | None = None,
    hdbscan_cluster_selection_epsilon: float = 0.0,
    hdbscan_cluster_selection_method: str = "eom",
    agg_n_clusters: int = 10,
    agg_linkage: str = "ward",
    kmeans_n_clusters: int = 10,
    kmeans_n_init: int = 10,
    kmeans_max_iter: int = 300,
    random_state: int | None = 42,
):
    """Partition ``data`` into clusters and return an integer label array.

    Parameters
    ----------
    data : array-like
        Either a precomputed pairwise distance matrix (N×N) when
        ``is_precomputed=True``, or a feature matrix (N×D) otherwise.
    method : str
        One of ``"HDBSCAN"``, ``"Agglomerative"``, ``"K-Means"``.
    is_precomputed : bool
        Whether ``data`` is a precomputed pairwise distance matrix.
    metric : str
        Distance metric used when ``is_precomputed=False``.
    hdbscan_min_cluster_size : int
        Minimum number of points to form a cluster (HDBSCAN).
    hdbscan_min_samples : int or None
        Minimum number of core points. ``None`` defaults to
        ``min_cluster_size`` (HDBSCAN).
    hdbscan_cluster_selection_epsilon : float
        Distance threshold below which clusters are merged (HDBSCAN).
    hdbscan_cluster_selection_method : str
        ``"eom"`` (excess of mass) or ``"leaf"`` (HDBSCAN).
    agg_n_clusters : int
        Target number of clusters (Agglomerative).
    agg_linkage : str
        Linkage criterion (Agglomerative). ``"ward"`` is silently replaced
        with ``"average"`` when a precomputed distance matrix is used.
    kmeans_n_clusters : int
        Number of clusters (K-Means).
    kmeans_n_init : int
        Number of random initialisations (K-Means).
    kmeans_max_iter : int
        Maximum iterations per run (K-Means).
    random_state : int or None
        Random seed for reproducible K-Means results.

    Returns
    -------
    labels : np.ndarray of shape (N,), dtype int
        Cluster label per point.  HDBSCAN assigns ``-1`` to noise points.
        All other methods return labels in ``[0, n_clusters)``.
    """
    arr = np.asarray(data, dtype=np.float64)
    method = str(method)

    if method == "HDBSCAN":
        from sklearn.cluster import HDBSCAN

        _metric = "precomputed" if is_precomputed else metric
        clusterer = HDBSCAN(
            min_cluster_size=int(hdbscan_min_cluster_size),
            min_samples=(
                int(hdbscan_min_samples) if hdbscan_min_samples is not None else None
            ),
            cluster_selection_epsilon=float(hdbscan_cluster_selection_epsilon),
            cluster_selection_method=str(hdbscan_cluster_selection_method),
            metric=_metric,
        )
        with warnings.catch_warnings(action="ignore"):
            labels = clusterer.fit_predict(arr)

    elif method == "Agglomerative":
        from sklearn.cluster import AgglomerativeClustering

        _metric = "precomputed" if is_precomputed else metric
        linkage = str(agg_linkage)
        # Ward linkage requires Euclidean distances; fall back gracefully.
        if is_precomputed and linkage == "ward":
            linkage = "average"
        clusterer = AgglomerativeClustering(
            n_clusters=int(agg_n_clusters),
            metric=_metric,
            linkage=linkage,
        )
        labels = clusterer.fit_predict(arr)

    elif method == "K-Means":
        if is_precomputed:
            raise ValueError(
                "K-Means requires feature vectors, not a precomputed distance matrix."
            )
        from sklearn.cluster import KMeans

        clusterer = KMeans(
            n_clusters=int(kmeans_n_clusters),
            n_init=int(kmeans_n_init),
            max_iter=int(kmeans_max_iter),
            random_state=random_state,
        )
        with warnings.catch_warnings(action="ignore"):
            labels = clusterer.fit_predict(arr)

    else:
        raise ValueError(
            f"Unknown clustering method '{method}'. "
            "Expected 'HDBSCAN', 'Agglomerative', or 'K-Means'."
        )

    return labels.astype(int)


def labels_to_colors(labels, palette="vispy:husl", noise_color=(0.5, 0.5, 0.5, 1.0)):
    """Map integer cluster labels to RGBA colours.

    Parameters
    ----------
    labels : array-like of int
        Cluster assignment per point.  ``-1`` is treated as noise/unassigned.
    palette : str
        A ``cmap``-compatible colormap name used for distinct cluster colours.
    noise_color : tuple of 4 floats
        RGBA colour for noise points (label == -1).

    Returns
    -------
    colors : np.ndarray of shape (N, 4), dtype float32
        Per-point RGBA colours in ``[0, 1]`` range.
    """
    import cmap as _cmap

    labels = np.asarray(labels, dtype=int)
    unique_cluster_ids = sorted(set(labels[labels >= 0].tolist()))
    n = max(len(unique_cluster_ids), 1)

    colormap_obj = _cmap.Colormap(palette)
    palette_colors = list(colormap_obj.iter_colors(n))
    cluster_to_color = {
        lbl: np.array(palette_colors[i].rgba, dtype=np.float32)
        for i, lbl in enumerate(unique_cluster_ids)
    }

    noise = np.array(noise_color, dtype=np.float32)
    colors = np.empty((len(labels), 4), dtype=np.float32)
    for i, lbl in enumerate(labels):
        colors[i] = cluster_to_color.get(lbl, noise)

    return colors
