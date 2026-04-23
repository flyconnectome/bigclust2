"""Clustering algorithms for bigclust2."""

from __future__ import annotations

import logging
import warnings

import numpy as np
import networkx as nx
import scipy.cluster.hierarchy as sch

from scipy.spatial.distance import squareform, pdist

logger = logging.getLogger(__name__)


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
    hbdscan_label_noise: bool = False,
    hbdscan_label_noise_threshold: float | None = None,
    agg_stop_criterion: str = "n_clusters",
    agg_n_clusters: int | None = 10,
    agg_distance_threshold: float | None = None,
    agg_linkage: str = "ward",
    agg_homogeneous_labels=None,
    agg_homogeneous_eval_func=None,
    agg_homogeneous_max_dist: float | None = None,
    agg_homogeneous_min_dist: float | None = None,
    agg_homogeneous_min_dist_diff: float | None = None,
    agg_homogeneous_verbose: bool = False,
    kmeans_n_clusters: int = 10,
    kmeans_n_init: int = 10,
    kmeans_max_iter: int = 300,
    spectral_n_clusters: int = 10,
    spectral_affinity: str = "nearest_neighbors",
    spectral_gamma: float | None = None,
    spectral_n_neighbors: int = 10,
    spectral_n_init: int = 10,
    random_state: int | None = 42,
    allow_singletons: bool = False,
):
    """Partition ``data`` into clusters and return an integer label array.

    Parameters
    ----------
    data : array-like
        Either a precomputed pairwise distance matrix (N×N) when
        ``is_precomputed=True``, or a feature matrix (N×D) otherwise.
    method : str
        One of ``"HDBSCAN"``, ``"Agglomerative"``, ``"K-Means"``,
        or ``"Spectral"``.
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
    hbdscan_label_noise : bool
        If True, reassign HDBSCAN noise points (``-1``) to the closest
        non-noise cluster.
    hbdscan_label_noise_threshold : float or None
        Optional maximum distance allowed for reassignment. Noise points
        farther than this threshold from their closest cluster remain ``-1``.
        If None, all noise points are reassigned.
    agg_stop_criterion : str
        Agglomerative stop criterion. One of ``"n_clusters"``,
        ``"distance_threshold"``, or ``"homogeneous_composition"``.
    agg_n_clusters : int or None
        Target number of clusters (Agglomerative). Mutually exclusive with
        ``agg_distance_threshold``.
    agg_distance_threshold : float or None
        Linkage distance threshold for terminating tree merges
        (Agglomerative). Mutually exclusive with ``agg_n_clusters``.
    agg_linkage : str
        Linkage criterion (Agglomerative). ``"ward"`` is silently replaced
        with ``"average"`` when a precomputed distance matrix is used.
    agg_homogeneous_labels : array-like, optional
        Labels used to evaluate cluster composition in
        ``"homogeneous_composition"`` mode.
    agg_homogeneous_eval_func : callable, optional
        Custom evaluation function for cluster composition in
        ``"homogeneous_composition"`` mode. If not provided, ``is_good``
        is used.
    agg_homogeneous_max_dist : float, optional
        Upper split-distance bound for homogeneous composition mode.
    agg_homogeneous_min_dist : float, optional
        Lower split-distance bound for homogeneous composition mode.
    agg_homogeneous_min_dist_diff : float, optional
        Merge-nearby-clusters threshold for homogeneous composition mode.
    agg_homogeneous_verbose : bool
        Verbose logging in homogeneous composition mode.
    kmeans_n_clusters : int
        Number of clusters (K-Means).
    kmeans_n_init : int
        Number of random initialisations (K-Means).
    kmeans_max_iter : int
        Maximum iterations per run (K-Means).
    spectral_n_clusters : int
        Number of clusters (Spectral).
    spectral_affinity : str
        Affinity used by Spectral clustering. One of ``"nearest_neighbors"`` or ``"rbf"``.
    spectral_gamma : float or None
        Kernel scale for spectral affinity. Defaults to 1.0 when needed.
    spectral_n_neighbors : int
        Number of neighbors used with ``affinity='nearest_neighbors'``.
    spectral_n_init : int
        Number of initializations for Spectral clustering.
    random_state : int or None
        Random seed for reproducible K-Means and Spectral results.
    allow_singletons : bool
        If False, treat single-observation clusters as noise (i.e. assign
        label ``-1``).

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

        if hbdscan_label_noise:
            labels = labels.copy()
            non_noise_mask = labels >= 0
            noise_mask = labels == -1

            # Nothing to relabel if there is no noise or no valid cluster.
            if non_noise_mask.any() and noise_mask.any():
                clusters = np.unique(labels[non_noise_mask])

                if is_precomputed:
                    # Assign by nearest distance to any member in each cluster.
                    dmat = np.asarray(arr, dtype=np.float64)
                    noise_ix = np.flatnonzero(noise_mask)
                    for i in noise_ix:
                        best_cluster = -1
                        best_dist = np.inf
                        for c in clusters:
                            members = np.flatnonzero(labels == c)
                            if members.size == 0:
                                continue
                            d = float(np.min(dmat[i, members]))
                            if d < best_dist:
                                best_dist = d
                                best_cluster = int(c)

                        if best_cluster >= 0 and (
                            hbdscan_label_noise_threshold is None
                            or best_dist <= float(hbdscan_label_noise_threshold)
                        ):
                            labels[i] = best_cluster
                else:
                    # Assign by nearest cluster centroid in feature space.
                    feats = np.asarray(arr, dtype=np.float64)
                    from sklearn.metrics import pairwise_distances

                    cluster_centroids = []
                    cluster_ids = []
                    for c in clusters:
                        members = feats[labels == c]
                        if members.shape[0] == 0:
                            continue
                        cluster_ids.append(int(c))
                        cluster_centroids.append(np.mean(members, axis=0))

                    if cluster_centroids:
                        cluster_centroids = np.asarray(cluster_centroids)
                        noise_ix = np.flatnonzero(noise_mask)
                        d = pairwise_distances(
                            feats[noise_ix],
                            cluster_centroids,
                            metric=metric,
                        )
                        nearest = np.argmin(d, axis=1)
                        nearest_dist = d[np.arange(d.shape[0]), nearest]

                        for row_i, obs_i in enumerate(noise_ix):
                            if (
                                hbdscan_label_noise_threshold is None
                                or nearest_dist[row_i]
                                <= float(hbdscan_label_noise_threshold)
                            ):
                                labels[obs_i] = cluster_ids[nearest[row_i]]

    elif method == "Agglomerative":
        from sklearn.cluster import AgglomerativeClustering

        _metric = "precomputed" if is_precomputed else metric
        linkage = str(agg_linkage)
        criterion = str(agg_stop_criterion).strip().lower().replace(" ", "_")
        # Ward linkage requires Euclidean distances
        if linkage == "ward" and (is_precomputed or _metric != "euclidean"):
            raise ValueError(
                "Ward linkage should not be used with non-euclidean distances.\n"
                "Please use a different linkage method (e.g. 'average') or provide\n"
                "feature vectors + Euclidean metric instead of a distance matrix."
            )

        if criterion in ("homogeneous_composition", "homogenous_composition"):
            if not is_precomputed:
                arr = squareform(pdist(arr, metric=_metric))
                is_precomputed = True

            if agg_homogeneous_labels is None:
                raise ValueError(
                    "`agg_homogeneous_labels` are required for homogeneous composition."
                )

            label_arr = np.asarray(agg_homogeneous_labels)
            if label_arr.shape[0] != arr.shape[0]:
                raise ValueError(
                    "`agg_homogeneous_labels` length must match number of samples."
                )

            import pandas as pd

            eval_func = agg_homogeneous_eval_func or is_good
            labels = extract_homogeneous_clusters(
                dists=pd.DataFrame(arr),
                labels=label_arr,
                eval_func=eval_func,
                max_dist=agg_homogeneous_max_dist,
                min_dist=agg_homogeneous_min_dist,
                min_dist_diff=agg_homogeneous_min_dist_diff,
                link_method=linkage,
                verbose=agg_homogeneous_verbose,
            )
        else:
            if criterion not in ("n_clusters", "distance_threshold"):
                raise ValueError(
                    f"Unknown Agglomerative stop criterion '{agg_stop_criterion}'. "
                    "Expected 'n_clusters', 'distance_threshold' or "
                    "'homogeneous_composition'."
                )

            if criterion == "n_clusters":
                if agg_n_clusters is None:
                    raise ValueError(
                        "`agg_n_clusters` must be provided when using "
                        "stop criterion 'n_clusters'."
                    )
                if agg_distance_threshold is not None:
                    raise ValueError(
                        "Do not provide `agg_distance_threshold` when using "
                        "stop criterion 'n_clusters'."
                    )
            elif criterion == "distance_threshold":
                if agg_distance_threshold is None:
                    raise ValueError(
                        "`agg_distance_threshold` must be provided when using "
                        "stop criterion 'distance_threshold'."
                    )
                if agg_n_clusters is not None:
                    raise ValueError(
                        "Do not provide `agg_n_clusters` when using "
                        "stop criterion 'distance_threshold'."
                    )

            kwargs = {
                "metric": _metric,
                "linkage": linkage,
            }
            if criterion == "distance_threshold":
                kwargs["n_clusters"] = None
                kwargs["distance_threshold"] = float(agg_distance_threshold)
            else:
                kwargs["n_clusters"] = int(agg_n_clusters)

            clusterer = AgglomerativeClustering(**kwargs)
            labels = clusterer.fit_predict(arr)

    elif method == "Spectral":
        from sklearn.cluster import SpectralClustering

        affinity = str(spectral_affinity)
        if is_precomputed:
            dmat = np.asarray(arr, dtype=np.float64)
            if dmat.ndim != 2 or dmat.shape[0] != dmat.shape[1]:
                raise ValueError(
                    "Spectral clustering requires a square precomputed distance matrix "
                    "when is_precomputed=True."
                )
            gamma = 1.0 if spectral_gamma is None else float(spectral_gamma)
            affinity_matrix = np.exp(-gamma * dmat)
            clusterer = SpectralClustering(
                n_clusters=int(spectral_n_clusters),
                affinity="precomputed",
                n_init=int(spectral_n_init),
                random_state=random_state,
            )
            labels = clusterer.fit_predict(affinity_matrix)
        else:
            if affinity not in ("nearest_neighbors", "rbf"):
                raise ValueError(
                    "Spectral affinity must be either 'nearest_neighbors' or 'rbf'."
                )
            kwargs = {
                "n_clusters": int(spectral_n_clusters),
                "affinity": affinity,
                "n_init": int(spectral_n_init),
                "n_neighbors": int(spectral_n_neighbors),
                "random_state": random_state,
            }
            if affinity == "rbf":
                kwargs["gamma"] = 1.0 if spectral_gamma is None else float(spectral_gamma)
            clusterer = SpectralClustering(**kwargs)
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
            "Expected 'HDBSCAN', 'Agglomerative', 'K-Means', or 'Spectral'."
        )

    if not allow_singletons:
        # Treat single-observation clusters as noise
        unique_labels, counts = np.unique(labels, return_counts=True)
        singleton_labels = unique_labels[counts == 1]
        if singleton_labels.size:
            labels = labels.copy()
            labels[np.isin(labels, singleton_labels)] = -1

    return labels.astype(int)


def find_k(
    data,
    *,
    method: str,
    is_precomputed: bool = False,
    metric: str = "euclidean",
    k_min: int = 2,
    k_max: int | str = "auto",
    score_method: str = "silhouette",
    spectral_affinity: str = "nearest_neighbors",
    spectral_gamma: float | None = None,
    spectral_n_neighbors: int = 10,
    spectral_n_init: int = 10,
    kmeans_n_init: int = 10,
    kmeans_max_iter: int = 300,
    random_state: int | None = 42,
    progress_callback=None,
):
    """Find an optimal number of clusters for K-Means or Spectral clustering.

    Parameters
    ----------
    data : array-like
        Either a precomputed pairwise distance matrix (NxN) when
        ``is_precomputed=True``, or a feature matrix (NxD) otherwise.
    method : str
        One of ``"K-Means"`` or ``"Spectral"``.
    is_precomputed : bool
        Whether ``data`` is a precomputed pairwise distance matrix.
    metric : str
        Distance metric used when ``is_precomputed=False``.
    k_min : int
        Minimum number of clusters to consider.
    k_max : int | "auto"
        Maximum number of clusters to consider. If "auto", a heuristic based on the number
        of samples is used to set a reasonable upper bound.
    score_method : str
        One of ``"silhouette"``, ``"calinski_harabasz"``, or
        ``"davies_bouldin"``.
    spectral_affinity : str
        Affinity used by Spectral clustering. One of
        ``"nearest_neighbors"`` or ``"rbf"``.
    spectral_gamma : float or None
        Gamma scale for RBF affinity.
    spectral_n_neighbors : int
        Number of neighbors for spectral nearest-neighbors affinity.
    spectral_n_init : int
        Number of initializations for Spectral clustering.
    kmeans_n_init : int
        Number of initializations for K-Means.
    kmeans_max_iter : int
        Maximum iterations for K-Means.
    random_state : int or None
        Random seed.
    progress_callback : callable, optional
        Optional callback called on each candidate k iteration with
        ``progress_callback(k, k_max)``.

    Returns
    -------
    int
        Estimated optimal number of clusters.
    """
    method = str(method)
    if method not in ("K-Means", "Spectral"):
        raise ValueError("find_k supports only 'K-Means' or 'Spectral'.")

    if k_min < 2:
        raise ValueError("k_min must be >= 2.")

    score_method = str(score_method).strip().lower()
    if score_method not in ("silhouette", "calinski_harabasz", "davies_bouldin"):
        raise ValueError(
            "score_method must be one of 'silhouette', 'calinski_harabasz', or 'davies_bouldin'."
        )

    arr = np.asarray(data, dtype=np.float64)
    n_samples = arr.shape[0]
    if n_samples < 2:
        raise ValueError("Need at least 2 samples to choose k.")

    auto_max = False
    if isinstance(k_max, str):
        if k_max.lower() != "auto":
            raise ValueError("k_max must be an integer or 'auto'.")
        auto_max = True
        if n_samples <= 50:
            k_max = max(k_min, n_samples - 1)
        else:
            k_max = min(
                n_samples - 1,
                max(k_min, int(np.sqrt(n_samples) * 2)),
            )

    k_max = int(k_max)
    if k_max < k_min:
        raise ValueError("k_min must be >= 2 and k_max must be >= k_min.")
    if is_precomputed and method == "K-Means":
        raise ValueError("K-Means requires feature vectors, not a precomputed distance matrix.")

    if is_precomputed and score_method in ("calinski_harabasz", "davies_bouldin"):
        raise ValueError(
            f"Score method '{score_method}' requires feature vectors, not precomputed distances."
        )

    from sklearn.metrics import (
        silhouette_score,
        calinski_harabasz_score,
        davies_bouldin_score,
    )

    best_k = None
    best_score = -np.inf if score_method in ("silhouette", "calinski_harabasz") else np.inf
    better = (
        (lambda new, old: new > old)
        if score_method in ("silhouette", "calinski_harabasz")
        else (lambda new, old: new < old)
    )

    initial_k_max = k_max
    if progress_callback is not None:
        try:
            progress_callback(0, initial_k_max)
        except Exception:
            pass

    k = k_min
    current_limit = initial_k_max
    while k <= current_limit:
        if progress_callback is not None:
            try:
                progress_callback(k, current_limit)
            except Exception:
                pass

        try:
            labels = run_clustering(
                arr,
                method=method,
                is_precomputed=is_precomputed,
                metric=metric,
                kmeans_n_clusters=k,
                kmeans_n_init=kmeans_n_init,
                kmeans_max_iter=kmeans_max_iter,
                spectral_n_clusters=k,
                spectral_affinity=spectral_affinity,
                spectral_gamma=spectral_gamma,
                spectral_n_neighbors=spectral_n_neighbors,
                spectral_n_init=spectral_n_init,
                random_state=random_state,
                allow_singletons=True,
            )
        except Exception:
            if auto_max and k == n_samples and best_k == initial_k_max:
                raise ValueError(
                    "Could not determine an optimal number of clusters; "
                    "search reached k == n_samples."
                )
            k += 1
            continue

        unique_labels = np.unique(labels[labels >= 0])
        if unique_labels.size < 2:
            if auto_max and k == n_samples and best_k == initial_k_max:
                raise ValueError(
                    "Could not determine an optimal number of clusters; "
                    "search reached k == n_samples."
                )
            k += 1
            continue

        try:
            if score_method == "silhouette":
                score = silhouette_score(arr, labels, metric="precomputed" if is_precomputed else metric)
            elif score_method == "calinski_harabasz":
                score = calinski_harabasz_score(arr, labels)
            else:
                score = davies_bouldin_score(arr, labels)
        except Exception:
            if auto_max and k == n_samples and best_k == initial_k_max:
                raise ValueError(
                    "Could not determine an optimal number of clusters; "
                    "search reached k == n_samples."
                )
            k += 1
            continue

        if best_k is None or better(score, best_score):
            best_score = score
            best_k = k
        elif auto_max and k > initial_k_max:
            break

        if auto_max and k == initial_k_max and best_k == initial_k_max:
            current_limit = n_samples

        k += 1

    if auto_max and best_k == n_samples:
        raise ValueError(
            "Could not determine an optimal number of clusters; "
            "search reached k == n_samples."
        )

    if best_k is None:
        raise ValueError(
            "Could not determine an optimal number of clusters for the given data and parameters."
        )

    return best_k


def evaluate_clustering(
    data,
    labels,
    *,
    method: str,
    is_precomputed: bool = False,
    metric: str = "cosine",
    true_labels=None,
    k_neighbors: int = 10,
    random_state: int | None = None,
):
    """Evaluate cluster labels using a selected metric.

    Parameters
    ----------
    data : array-like
        Either a precomputed pairwise distance matrix (NxN) when
        ``is_precomputed=True``, or a feature matrix (NxD) otherwise.
    labels : array-like of shape (N,)
        Cluster labels to evaluate.
    method : str
        One of ``"silhouette"``, ``"davies_bouldin"``, ``"calinski_harabasz"``,
        ``"adjusted_rand_index"``, ``"adjusted_mutual_info"``,
        ``"neighbor_consistency"``, or ``"graph_modularity"``.
    is_precomputed : bool
        Whether ``data`` is a precomputed pairwise distance matrix.
    metric : str
        Distance metric used when ``is_precomputed=False`` and for
        neighborhood-based metrics.
    true_labels : array-like, optional
        Ground truth labels used for supervised metrics such as ARI/AMI.
    k_neighbors : int
        Number of neighbors used for neighborhood-based metrics.
    random_state : int or None
        Random seed for any metric that may need it.

    Returns
    -------
    float
        Score computed by the selected metric.
    """
    labels = np.asarray(labels, dtype=np.int64)
    if labels.ndim != 1:
        raise ValueError("`labels` must be a 1D array-like of cluster IDs.")
    n_samples = labels.shape[0]
    if n_samples < 2:
        raise ValueError("Need at least 2 samples to evaluate clustering.")
    method = str(method).strip().lower()
    if int(k_neighbors) < 1:
        raise ValueError("`k_neighbors` must be >= 1.")

    if method in ("silhouette",):
        if is_precomputed:
            dmat = np.asarray(data, dtype=np.float64)
            _check_distance_matrix(dmat)
            from sklearn.metrics import silhouette_score

            return float(silhouette_score(dmat, labels, metric="precomputed"))

        feats = np.asarray(data, dtype=np.float64)
        if feats.ndim != 2:
            raise ValueError(
                "`data` must be a 2D feature matrix when `is_precomputed=False`."
            )
        from sklearn.metrics import silhouette_score

        return float(silhouette_score(feats, labels, metric=metric))

    if method in ("davies_bouldin", "db"):
        if is_precomputed:
            raise ValueError(
                "Davies-Bouldin score requires feature vectors, not a distance matrix."
            )
        feats = np.asarray(data, dtype=np.float64)
        if feats.ndim != 2:
            raise ValueError("`data` must be a 2D feature matrix for Davies-Bouldin.")
        from sklearn.metrics import davies_bouldin_score

        return float(davies_bouldin_score(feats, labels))

    if method in ("calinski_harabasz", "ch"):
        if is_precomputed:
            raise ValueError(
                "Calinski-Harabasz score requires feature vectors, not a distance matrix."
            )
        feats = np.asarray(data, dtype=np.float64)
        if feats.ndim != 2:
            raise ValueError(
                "`data` must be a 2D feature matrix for Calinski-Harabasz."
            )
        from sklearn.metrics import calinski_harabasz_score

        return float(calinski_harabasz_score(feats, labels))

    if method in ("adjusted_rand_index", "ari"):
        if true_labels is None:
            raise ValueError("`true_labels` must be provided for Adjusted Rand Index.")
        from sklearn.metrics import adjusted_rand_score

        return float(adjusted_rand_score(np.asarray(true_labels), labels))

    if method in ("adjusted_mutual_info", "ami"):
        if true_labels is None:
            raise ValueError(
                "`true_labels` must be provided for Adjusted Mutual Information."
            )
        from sklearn.metrics import adjusted_mutual_info_score

        return float(adjusted_mutual_info_score(np.asarray(true_labels), labels))

    if method in ("neighbor_consistency", "nn_consistency"):
        neighbor_ix = _knn_neighbors(data, int(k_neighbors), is_precomputed, metric)
        same_label = labels[neighbor_ix] == labels[:, None]
        return float(np.mean(same_label))

    if method in ("graph_modularity", "modularity"):
        neighbor_ix = _knn_neighbors(data, int(k_neighbors), is_precomputed, metric)
        G = _knn_graph_from_indices(n_samples, neighbor_ix)
        communities = [set(np.flatnonzero(labels == c)) for c in np.unique(labels)]
        from networkx.algorithms.community.quality import modularity

        return float(modularity(G, communities))

    raise ValueError(
        f"Unknown evaluation method '{method}'. "
        "Expected one of 'silhouette', 'davies_bouldin', 'calinski_harabasz', "
        "'adjusted_rand_index', 'adjusted_mutual_info', 'neighbor_consistency', "
        "or 'graph_modularity'."
    )


def evaluate_clustering_sample(
    data,
    labels,
    *,
    method: str,
    is_precomputed: bool = False,
    metric: str = "euclidean",
    k_neighbors: int = 10,
):
    """Compute a per-sample clustering quality score.

    Parameters
    ----------
    data : array-like
        Either a precomputed pairwise distance matrix (NxN) when
        ``is_precomputed=True``, or a feature matrix (NxD) otherwise.
    labels : array-like of shape (N,)
        Cluster labels for each sample.
    method : str
        One of ``"silhouette"`` or ``"neighbor_consistency"``.
    is_precomputed : bool
        Whether ``data`` is a precomputed pairwise distance matrix.
    metric : str
        Distance metric used when ``is_precomputed=False`` and for
        neighborhood-based metrics.
    k_neighbors : int
        Number of neighbors used for ``neighbor_consistency``.

    Returns
    -------
    np.ndarray of shape (N,)
        Per-sample quality scores.
    """
    labels = np.asarray(labels, dtype=np.int64)
    if labels.ndim != 1:
        raise ValueError("`labels` must be a 1D array-like of cluster IDs.")
    n_samples = labels.shape[0]
    if n_samples < 2:
        raise ValueError("Need at least 2 samples to evaluate clustering.")
    method = str(method).strip().lower()
    if int(k_neighbors) < 1:
        raise ValueError("`k_neighbors` must be >= 1.")

    if method in ("silhouette", "silhouette_samples"):
        from sklearn.metrics import silhouette_samples

        if is_precomputed:
            dmat = np.asarray(data, dtype=np.float64)
            _check_distance_matrix(dmat)
            return silhouette_samples(dmat, labels, metric="precomputed")

        feats = np.asarray(data, dtype=np.float64)
        if feats.ndim != 2:
            raise ValueError(
                "`data` must be a 2D feature matrix when `is_precomputed=False`."
            )
        return silhouette_samples(feats, labels, metric=metric)

    if method in ("neighbor_consistency", "nn_consistency"):
        neighbor_ix = _knn_neighbors(data, int(k_neighbors), is_precomputed, metric)
        same_label = labels[neighbor_ix] == labels[:, None]
        return np.mean(same_label, axis=1).astype(np.float64)

    raise ValueError(
        f"Unknown per-sample evaluation method '{method}'. "
        "Expected one of 'silhouette' or 'neighbor_consistency'."
    )


def _check_distance_matrix(dmat):
    dmat = np.asarray(dmat, dtype=np.float64)
    if dmat.ndim != 2 or dmat.shape[0] != dmat.shape[1]:
        raise ValueError("`data` must be a square distance matrix when `is_precomputed=True`.")
    return dmat


def _knn_neighbors(data, k_neighbors, is_precomputed, metric):
    from sklearn.neighbors import NearestNeighbors

    if is_precomputed:
        dmat = _check_distance_matrix(data)
        n_samples = dmat.shape[0]
        n_neighbors = min(k_neighbors + 1, n_samples)
        nn = NearestNeighbors(
            n_neighbors=n_neighbors,
            metric="precomputed",
            algorithm="brute",
        )
        nn.fit(dmat)
        _, neighbors = nn.kneighbors(dmat, return_distance=True)
    else:
        feats = np.asarray(data, dtype=np.float64)
        if feats.ndim != 2:
            raise ValueError(
                "`data` must be a 2D feature matrix when `is_precomputed=False`."
            )
        n_samples = feats.shape[0]
        n_neighbors = min(k_neighbors + 1, n_samples)
        nn = NearestNeighbors(n_neighbors=n_neighbors, metric=metric)
        nn.fit(feats)
        _, neighbors = nn.kneighbors(feats, return_distance=True)

    if neighbors.shape[1] <= 1:
        return np.empty((n_samples, 0), dtype=np.int64)
    return neighbors[:, 1:].astype(np.int64, copy=False)


def _knn_graph_from_indices(n_samples, neighbor_ix):
    G = nx.Graph()
    G.add_nodes_from(range(n_samples))
    edges = []
    for i, neighbors in enumerate(neighbor_ix):
        for j in neighbors:
            if i != int(j):
                edges.append((int(i), int(j)))
    G.add_edges_from(edges)
    return G


def is_good(v, n_unique_ds):
    """Check if composition of labels.

    `v` is a len(N) vector with counts per label (i.e. per dataset): e.g. [1, 2, 1]
    """
    if isinstance(v, dict):
        v = list(v.values())
    if len(v) < n_unique_ds:  # if not all datasets present
        return False
    mn = min(v)
    mx = max(v)
    if ((mx - mn) > 3) and ((mx / mn) >= 2):
        return False
    return True


def extract_homogeneous_clusters(
    dists,
    labels,
    eval_func=is_good,
    max_dist=None,
    min_dist=None,
    min_dist_diff=None,
    link_method="ward",
    linkage=None,
    verbose=False,
):
    """Make clusters that contains representatives of each unique label.

    Parameters
    ----------
    dists :         pd.DataFrame
                    Distances from which to find clusters.
    labels :        np.ndarray
                    Labels for each row in `dists`.
    eval_func :     callable
                    Must accept two positional arguments:
                     1. A numpy array of label counts (e.g. `[1, 1, 2]`)
                     2. An integer describing how many unique labels we expect
                    Must return True if cluster composition is acceptable and
                    False if it isn't.
    min/max_dist :  float, optional
                    Use this to set a range of between-cluster distances at which
                    we are allowed to make clusters. For example:
                     - ``min_dist=.1`` means that we will not split further if
                       a cluster is already more similar than .1
                     - ``max_dist=1`` means that we will keep splitting clusters
                       that are more dissimilar than 1 even if they don't fullfil
                       the `eval_func`
    min_dist_diff : float, optional
                    Consider two homogenous clusters that are adjacent to each
                    other in the dendrogram: if the difference in distance
                    between the two clusters and their supercluster is smaller
                    than `min_dist_diff` they will be merged. Or in other words:
                    we will merge if the three horizontal lines in the dendrograms
                    are closer together than `min_dist_diff`.
    link_method :   str
                    Method to use for generating the linkage.
    linkage :       np.ndarray
                    Precomputed linkage. If this is given `link_method` is ignored.

    Returns
    -------
    cl :        np.ndarray

    """
    if dists.values[0, 0] >= 0.999:
        dists = 1 - dists

    # Make linkage
    if linkage is None:
        Z = sch.linkage(squareform(dists, checks=False), method=link_method)
    else:
        Z = linkage

    # Turn linkage into graph
    G = linkage_to_graph(Z, labels=labels)

    # Add origin as node attribute
    n_unique_ds = len(np.unique(labels))

    # Prepare eval function
    def _eval_func(x):
        return eval_func(x, n_unique_ds)

    # Find clusters recursively
    clusters = {}
    label_dict = nx.get_node_attributes(G, "label")
    _ = _find_clusters_rec(
        G,
        clusters=clusters,
        eval_func=_eval_func,
        label_dict=label_dict,
        max_dist=max_dist,
        min_dist=min_dist,
        verbose=verbose,
    )

    # Keep only clusters labels for the leaf nodes
    clusters = {k: v for k, v in clusters.items() if k in label_dict}

    # Clusters are currently labels based at which hinge they were created
    # We have to renumber them
    reind = {c: i for i, c in enumerate(np.unique(list(clusters.values())))}
    clusters = {k: reind[v] for k, v in clusters.items()}

    # At this point singletons might not be assigned a cluster - we need
    # to account for that and give them a unique cluster
    for i in np.arange((len(dists))):
        if i not in clusters:
            clusters[i] = len(set(clusters.values()))

    cl = np.array([clusters[i] for i in np.arange(len(dists))])

    if min_dist_diff:
        cl = _merge_similar_clusters(
            cl=cl, G=G, dist_thresh=min_dist_diff, verbose=verbose
        )

    return cl


def _find_clusters_rec(
    G, clusters, eval_func, label_dict, max_dist=None, min_dist=None, verbose=False
):
    """Recursively find clusters."""
    if G.is_directed():
        G = G.to_undirected()

    # The root node should always be the last in the graph
    root = max(G.nodes)

    try:
        dist = G.nodes[root]["distance"]  # the distance between the two prior clusters
    except KeyError:
        # If this is a leaf-node it won't have a "distance" property
        dist = 0

    # Remove the root in this (sub)graph
    G2 = G.copy()
    G2.remove_node(root)

    # Split into the two connected components
    CC = list(nx.connected_components(G2))
    # Count the number of labels (i.e. datasets) present in each subgraph
    counts = [_count_labels(c, label_dict=label_dict) for c in CC]
    # Evaluate the counts
    is_good = [eval_func(c) for c in counts]

    # Check if we should stop here
    stop = False
    # If we are below the minimum distance we have to stop
    if min_dist and (dist <= min_dist):
        stop = True
    # If one or both of the clusters are bad...
    elif not all(is_good):
        # ... and the distance between the two clusters below is not too big
        # we can stop
        if max_dist and (dist <= max_dist):
            stop = True
        elif not max_dist:
            stop = True

    if not stop:
        for c in CC:
            _find_clusters_rec(
                G.subgraph(c),
                clusters=clusters,
                eval_func=eval_func,
                label_dict=label_dict,
                max_dist=max_dist,
                min_dist=min_dist,
                verbose=verbose,
            )
    else:
        if verbose:
            print(
                f"Found cluster of {sum([c.sum() for c in counts])} at distance {dist} ({root})"
            )
        clusters.update({n: root for n in G.nodes})

    return


def _count_labels(cluster, label_dict):
    """Takes a list of node IDs and counts labels among those."""
    cluster = list(cluster) if isinstance(cluster, set) else cluster
    cluster = np.asarray(cluster)
    cluster = cluster[np.isin(cluster, list(label_dict))]
    _, cnt = np.unique([label_dict[n] for n in cluster], return_counts=True)
    return cnt


def _merge_similar_clusters(cl, G, dist_thresh, verbose=False):
    """Merge similar clusters.

    Parameters
    ----------
    cl :        np.ndarray
                Clusters membership that is to be checked.
    G :         nx.DiGraph
                Graph representing the linkage.
    dist_thresh : float
                Distance under which to merge clusters.

    Returns
    -------
    cl :        np.ndarray
                Fixed cluster membership.

    """
    ix = np.arange(len(cl))
    to_merge = []
    for c1 in np.unique(cl):
        # Get the connected subgraph for this cluster
        p = nx.shortest_path(G.to_undirected(), source=ix[cl == c1][0], target=None)
        sg = [p[i] for i in ix[cl == c1][1:]]
        sg = np.unique([i for l in sg for i in l])  # flatten

        if len(sg) == 0:
            continue

        # The root for this cluster
        root = max(sg)

        dist_c1 = G.nodes[root].get("distance", 0)

        # The cluster one above this one
        try:
            top = next(G.predecessors(root))
        except StopIteration:
            continue
        except BaseException:
            raise

        # Distance between our original cluster and the closest
        dist_top = G.nodes[top].get("distance", np.inf)

        # Distance for the neighbouring cluster
        other = [s for s in G.successors(top) if s not in sg][0]
        dist_c2 = G.nodes[other].get("distance", 0)

        # If merging this and the next cluster are very similar
        th = 0.1
        if (dist_top - dist_c1) <= th and (dist_top - dist_c2) < th:
            # Get the index of the other cluster
            sg_other = list(nx.dfs_postorder_nodes(G, other))
            c2 = cl[[i for i in sg_other if i in ix]][0]

            if verbose:
                print(
                    f"Merging {c1} and {c2} (top={dist_top}; left={dist_c1}, right={dist_c2}"
                )

            to_merge.append([c1, c2])

    # Deduplicate
    to_merge = list(set([tuple(sorted(p)) for p in to_merge]))

    cl2 = cl.copy()
    for p in to_merge:
        cl2[cl2 == p[1]] = p[0]

    return cl2


def linkage_to_graph(Z, labels=None):
    """Turn linkage into a directed graph.

    Parameters
    ----------
    Z :         linkage
    labels :    iterable, optional
                A label for each of the original observations in Z.

    Returns
    -------
    nx.DiGraph
                A graph representing the dendrogram. Each node corresponds to
                either a leaf or a hinge in the dendrogram. Edges are directed
                and point from the root node (i.e. the top hinge) toward the
                leafs. Nodes representing clusters (i.e. non-leafs) have a
                "distance" property indicating the distance between the two
                downstream clusters/leafs.

    """
    # The number of original observations
    n = len(Z) + 1

    edges = []
    cl_dists = {}
    for i, row in enumerate(Z):
        edges.append((int(n + i), int(row[0])))
        edges.append((int(n + i), int(row[1])))
        cl_dists[int(n + i)] = row[2]

    G = nx.DiGraph()
    G.add_edges_from(edges)

    nx.set_node_attributes(G, {i: i < n for i in G.nodes},
                           name='is_original')
    nx.set_node_attributes(G, cl_dists, name='distance')

    if labels is not None:
        if len(labels) != n:
            raise ValueError(f'Expected {n} labels, got {len(labels)}')
        nx.set_node_attributes(G, dict(zip(np.arange(n), labels)), name='label')

    return G