"""Grow a point selection by pulling in the nearest unselected neighbours.

Pure, GUI-free helpers used by :class:`~bigclust2.scatter.ScatterFigure` to grow
the current selection. "Nearest" is single-linkage: a point's distance to the
selection is its minimum distance to *any* selected point. This is the only
notion that also works on a sparse KNN graph, where we have no full pairwise
matrix and fall back to shortest-path (flood-fill) distances.

Distance can be measured in four spaces, mirroring what the figure already holds:

* ``embedding``  – the 2D layout shown on screen (always available).
* ``distances``  – a precomputed pairwise distance matrix.
* ``features``   – per-point feature vectors (various metrics).
* ``knn``        – a sparse KNN graph (large projects where a full matrix is too
  big); single-linkage distance becomes shortest-path distance on that graph.

Shrinking is *not* handled here: the figure implements it as a plain reversal of
prior grow steps (an undo stack), so no distance computation is needed.

Heavy dependencies (scipy / sklearn — both already project deps) are imported
lazily inside the functions that use them.
"""

from __future__ import annotations

import numpy as np

from .utils import check_finite_features

SOURCE_EMBEDDING = "embedding"
SOURCE_DISTANCES = "distances"
SOURCE_FEATURES = "features"
SOURCE_KNN = "knn"

# Display / auto-default preference order. Embedding comes first because it is
# the default source (the layout the user actually sees on screen).
_SOURCE_ORDER = (SOURCE_EMBEDDING, SOURCE_KNN, SOURCE_DISTANCES, SOURCE_FEATURES)

# KD-trees only prune effectively in low dimensions; beyond this the curse of
# dimensionality makes them degenerate to a slow, non-BLAS brute force, so we
# fall back to (BLAS-backed) chunked ``pairwise_distances`` instead. The 2D
# embedding stays well under this; high-dimensional feature spaces are well over.
_KDTREE_MAX_DIM = 16

# Chunking for the brute-force pairwise path. The block size caps the
# materialised (chunk, m) distance matrix, but for a tiny selection that alone
# would allow one enormous call — which makes sklearn's float32 euclidean upcast
# path scale pathologically (a selection of 1 ends up *slower* than 100). The row
# cap keeps each call cache/RAM-friendly regardless of selection size.
_PAIRWISE_MAX_BLOCK = 2_000_000  # max materialised (chunk x m) elements
_PAIRWISE_MAX_CHUNK_ROWS = 20_000  # max rows per chunk, regardless of m


def _pairwise_chunk_rows(m):
    """Rows per chunk for the brute-force pairwise loop (see constants above)."""
    return max(1, min(_PAIRWISE_MAX_CHUNK_ROWS, _PAIRWISE_MAX_BLOCK // max(1, m)))


class GrowShrinkUnavailable(Exception):
    """Raised when a requested distance source is not available on the figure."""


class GrowShrinkThresholdUnavailable(GrowShrinkUnavailable):
    """Raised when no similarity threshold can be derived from the selection.

    Subclasses :class:`GrowShrinkUnavailable` so callers that already handle the
    latter surface this one's (more specific) message too.
    """


def available_sources(dists, positions):
    """Return the distance sources usable for grow/shrink right now.

    ``embedding`` whenever ``positions`` exist (always, in practice);
    ``distances`` / ``features`` / ``knn`` when the matching key is present in
    the figure's ``dists`` dict. Ordered by :data:`_SOURCE_ORDER`.
    """
    keys = set(dists) if dists else set()
    sources = []
    for src in _SOURCE_ORDER:
        if src == SOURCE_EMBEDDING:
            if positions is not None and len(positions):
                sources.append(src)
        elif src in keys:
            sources.append(src)
    return sources


def nearest_distance_to_selection(
    selected, *, source, positions, dists, metric="euclidean"
):
    """Single-linkage distance of every point to the current selection.

    Parameters
    ----------
    selected :  array of int
                Row indices of the currently selected points.
    source :    str
                One of the ``SOURCE_*`` constants.
    positions : (N, 2) array or None
                Embedding coordinates (used when ``source == 'embedding'``).
    dists :     dict or None
                The figure's distance dict (``distances`` / ``features`` /
                ``knn``).
    metric :    str
                Distance metric, only used when ``source == 'features'``.

    Returns
    -------
    (N,) float array
        Each point's minimum distance to any selected point. ``np.inf`` for
        points whose distance is unknown. For the dense sources this is the true
        single-linkage distance; for the sparse KNN graph it is the distance to
        the nearest selected point *that lists it as a neighbour* — i.e. the
        one-hop frontier (everything further out is ``inf`` and is reached
        progressively as repeated grows advance the frontier).
    """
    selected = np.asarray(selected, dtype=int)

    if source == SOURCE_EMBEDDING:
        if positions is None or not len(positions):
            raise GrowShrinkUnavailable("No embedding positions available.")
        # Keep the native dtype (typically float32): cKDTree upcasts internally
        # and pairwise_distances handles float32 — forcing float64 here would copy
        # the whole feature matrix (≈2× memory) for no benefit.
        return _nearest_from_coords(np.asarray(positions), selected, "euclidean")

    if source == SOURCE_FEATURES:
        if not dists or "features" not in dists:
            raise GrowShrinkUnavailable("No feature vectors available.")
        X = np.asarray(dists["features"])
        try:
            score = _nearest_from_coords(X, selected, metric)
        except ValueError:
            # cKDTree/sklearn refuse non-finite input with a generic message;
            # replace it with one that carries row/column counts if that is
            # indeed the cause (a full-matrix scan is fine on the error path).
            check_finite_features(X, "grow/shrink selection")
            raise
        if np.isnan(score).any():
            # The euclidean GEMM path computes straight through NaNs, which
            # would otherwise silently exclude those points from the grow.
            check_finite_features(X, "grow/shrink selection")
        return score

    if source == SOURCE_DISTANCES:
        if not dists or "distances" not in dists:
            raise GrowShrinkUnavailable("No distance matrix available.")
        # DataFrame is ID-indexed; convert and index by row position.
        D = np.asarray(dists["distances"])
        return D[:, selected].min(axis=1)

    if source == SOURCE_KNN:
        if not dists or "knn" not in dists:
            raise GrowShrinkUnavailable("No KNN graph available.")
        return _nearest_from_knn(dists["knn"], selected)

    raise GrowShrinkUnavailable(f"Unknown distance source: {source!r}")


def grow_selection(
    selected, step, *, source, positions, dists, metric="euclidean", scope_mask=None
):
    """Return the row indices to ADD when growing the selection by ``step``.

    Picks the up-to-``step`` smallest-distance points that are unselected, have a
    finite distance, and (if ``scope_mask`` is given) lie in scope. Ties are
    broken by ascending row index so repeated grows are deterministic. Returns an
    empty array when nothing can be added.
    """
    selected = np.asarray(selected, dtype=int)
    step = int(step)
    if step <= 0 or len(selected) == 0:
        return np.empty(0, dtype=int)

    score = nearest_distance_to_selection(
        selected, source=source, positions=positions, dists=dists, metric=metric
    )
    n = len(score)

    candidate = np.isfinite(score)
    candidate[selected] = False  # never re-add already-selected points
    if scope_mask is not None and len(scope_mask) == n:
        candidate &= np.asarray(scope_mask, dtype=bool)

    cand_idx = np.where(candidate)[0]
    if not len(cand_idx):
        return np.empty(0, dtype=int)

    # Primary key: distance; secondary: row index (cand_idx is already ascending,
    # so this gives a stable, reproducible tie-break).
    order = np.lexsort((cand_idx, score[cand_idx]))
    return np.sort(cand_idx[order[:step]])


def within_selection_neighbor_distances(
    selected, *, source, positions, dists, metric="euclidean"
):
    """Each selected point's distance to its nearest *other* selected point.

    Returns
    -------
    (m,) float array
        Aligned with ``selected``. ``np.inf`` where a point has no measurable
        in-selection neighbour (only the sparse KNN graph can produce this, when
        a selected point lists no other selected point among its neighbours).
        Callers detect "no internal structure" via ``np.isfinite(...).any()``.
    """
    sel = np.asarray(selected, dtype=int)
    if len(sel) <= 1:
        return np.full(len(sel), np.inf)

    if source == SOURCE_EMBEDDING:
        if positions is None or not len(positions):
            raise GrowShrinkUnavailable("No embedding positions available.")
        # Index first so only the (small) selection is materialised; keep the
        # native dtype to avoid a full-matrix float64 copy.
        return _within_from_coords(np.asarray(positions)[sel], "euclidean")

    if source == SOURCE_FEATURES:
        if not dists or "features" not in dists:
            raise GrowShrinkUnavailable("No feature vectors available.")
        # NaN handling mirrors `nearest_distance_to_selection` (which see);
        # only the selected rows are used, so only those are checked.
        X_sel = np.asarray(dists["features"])[sel]
        try:
            w = _within_from_coords(X_sel, metric)
        except ValueError:
            check_finite_features(X_sel, "grow/shrink selection")
            raise
        if np.isnan(w).any():
            check_finite_features(X_sel, "grow/shrink selection")
        return w

    if source == SOURCE_DISTANCES:
        if not dists or "distances" not in dists:
            raise GrowShrinkUnavailable("No distance matrix available.")
        # DataFrame is ID-indexed; convert and index by row position.
        sub = np.asarray(dists["distances"])[np.ix_(sel, sel)].astype(np.float64)
        np.fill_diagonal(sub, np.inf)  # ignore the self-distance
        return sub.min(axis=1)

    if source == SOURCE_KNN:
        if not dists or "knn" not in dists:
            raise GrowShrinkUnavailable("No KNN graph available.")
        return _within_from_knn(dists["knn"], sel)

    raise GrowShrinkUnavailable(f"Unknown distance source: {source!r}")


def selection_distance_threshold(
    selected, *, source, positions, dists, metric="euclidean", factor=1.0
):
    """Auto distance threshold ``factor * max(within-selection NN distances)``.

    Raises
    ------
    GrowShrinkThresholdUnavailable
        If fewer than 2 points are selected, or (KNN) no selected point has a
        selected neighbour — i.e. there is no internal structure to measure.
    GrowShrinkUnavailable
        If the requested ``source`` data is missing.
    """
    sel = np.asarray(selected, dtype=int)
    if len(sel) < 2:
        raise GrowShrinkThresholdUnavailable(
            "Select at least 2 points to grow by similarity."
        )
    w = within_selection_neighbor_distances(
        sel, source=source, positions=positions, dists=dists, metric=metric
    )
    finite = w[np.isfinite(w)]
    if not finite.size:
        raise GrowShrinkThresholdUnavailable(
            "Selection has no internal neighbour links to derive a distance."
        )
    return float(factor) * float(finite.max())


def grow_within_threshold(
    selected,
    threshold,
    *,
    source,
    positions,
    dists,
    metric="euclidean",
    scope_mask=None,
):
    """Single-pass similarity grow: indices to ADD within ``threshold``.

    Adds every unselected, in-scope point whose
    :func:`nearest_distance_to_selection` is finite and ``<= threshold`` (one
    pass, no flood). For the KNN source this is bounded to the one-hop frontier,
    matching that function's semantics. Returns ascending indices (empty array
    when nothing qualifies or the selection is empty).
    """
    sel = np.asarray(selected, dtype=int)
    if len(sel) == 0:
        return np.empty(0, dtype=int)

    score = nearest_distance_to_selection(
        sel, source=source, positions=positions, dists=dists, metric=metric
    )
    n = len(score)

    candidate = np.isfinite(score) & (score <= float(threshold))
    candidate[sel] = False  # never re-add already-selected points
    if scope_mask is not None and len(scope_mask) == n:
        candidate &= np.asarray(scope_mask, dtype=bool)

    return np.where(candidate)[0]


def _within_from_coords(X, metric):
    """Min distance from each row of ``X`` to any *other* row of ``X``."""
    m = X.shape[0]
    if m <= 1:
        return np.full(m, np.inf)
    # Low-dimensional euclidean: a KD-tree prunes effectively and is fastest.
    if metric == "euclidean" and X.shape[1] <= _KDTREE_MAX_DIM:
        from scipy.spatial import cKDTree

        # k=2: the first hit is the point itself (distance 0); take the second.
        dist, _ = cKDTree(X).query(X, k=2)
        return np.asarray(dist[:, 1], dtype=np.float64)

    # High-dimensional euclidean: chunked squared-norm GEMM (see
    # `_nearest_euclidean_gemm`), excluding each row's own zero self-distance.
    if metric == "euclidean":
        return _within_euclidean_gemm(X)

    # Any other metric: brute force.
    from sklearn.metrics import pairwise_distances  

    D = pairwise_distances(X, X, metric=metric)
    np.fill_diagonal(D, np.inf)
    return D.min(axis=1)


def _within_euclidean_gemm(X):
    """Each row's min euclidean distance to any *other* row of ``X``.

    Squared-norm GEMM like :func:`_nearest_euclidean_gemm`, chunked over rows and
    excluding each row's own (zero) self-distance before taking the per-row min.
    """
    SS = np.einsum("ij,ij->i", X, X, dtype=np.float64)  # ‖x‖² per row
    m = X.shape[0]
    out = np.empty(m, dtype=np.float64)
    chunk = _pairwise_chunk_rows(m)
    for start in range(0, m, chunk):
        stop = min(start + chunk, m)
        d2 = SS[start:stop, None] + SS[None, :] - 2.0 * (X[start:stop] @ X.T)
        np.maximum(d2, 0.0, out=d2)
        # Drop each row's self-distance so it doesn't win the min.
        local = np.arange(stop - start)
        d2[local, np.arange(start, stop)] = np.inf
        out[start:stop] = np.sqrt(d2.min(axis=1))
    return out


def _within_from_knn(graph, selected):
    """Each selected point's smallest KNN-edge weight to another selected point."""
    indices = np.asarray(graph.indices)
    gdist = np.asarray(graph.dists, dtype=np.float64)
    n = indices.shape[0]

    sel_mask = np.zeros(n, dtype=bool)
    sel_mask[selected] = True

    neigh = indices[selected]  # (m, k) neighbour row positions, -1 = missing
    ndist = gdist[selected]  # (m, k) edge weights
    # Keep only edges to neighbours that are themselves selected.
    valid = (neigh >= 0) & sel_mask[np.clip(neigh, 0, n - 1)]
    masked = np.where(valid, ndist, np.inf)
    return masked.min(axis=1)  # inf for rows with no selected neighbour


def _nearest_from_coords(X, selected, metric):
    """Min distance from every row of ``X`` to the rows ``X[selected]``."""
    n = X.shape[0]
    # Low-dimensional euclidean: a KD-tree prunes effectively and is fastest.
    if metric == "euclidean" and X.shape[1] <= _KDTREE_MAX_DIM:
        from scipy.spatial import cKDTree

        dist, _ = cKDTree(X[selected]).query(X, k=1)
        return np.asarray(dist, dtype=np.float64)

    # High-dimensional euclidean: chunked squared-norm GEMM — far faster than
    # sklearn's pairwise_distances, which upcasts float32 -> float64 per block
    # (slow, and pathologically so for tiny selections).
    if metric == "euclidean":
        return _nearest_euclidean_gemm(X, selected)

    # Any other metric: BLAS-backed brute force via chunked pairwise_distances.
    from sklearn.metrics import pairwise_distances

    sel_X = X[selected]
    out = np.empty(n, dtype=np.float64)
    chunk = _pairwise_chunk_rows(len(selected))
    for start in range(0, n, chunk):
        stop = min(start + chunk, n)
        block = pairwise_distances(X[start:stop], sel_X, metric=metric)
        out[start:stop] = block.min(axis=1)
    return out


def _nearest_euclidean_gemm(X, selected):
    """Min euclidean distance from every row of ``X`` to the rows ``X[selected]``.

    Uses the squared-norm identity ``d² = ‖x‖² + ‖s‖² − 2·x·s``: the cross term is
    a float32 BLAS matmul (fast) and the norms are accumulated in float64 (so the
    subtraction is stable). Negative ``d²`` from float32 cancellation is clamped to
    0. This is accurate enough for nearest/threshold ranking; sklearn instead
    upcasts the whole matmul to float64 for full precision, which is much slower.
    """
    sel = np.ascontiguousarray(X[selected])
    SS = np.einsum("ij,ij->i", sel, sel, dtype=np.float64)  # ‖s‖² per selected
    n = X.shape[0]
    out = np.empty(n, dtype=np.float64)
    chunk = _pairwise_chunk_rows(len(selected))
    for start in range(0, n, chunk):
        stop = min(start + chunk, n)
        Xc = X[start:stop]
        XX = np.einsum("ij,ij->i", Xc, Xc, dtype=np.float64)  # ‖x‖² per row
        d2 = XX[:, None] + SS[None, :] - 2.0 * (Xc @ sel.T)
        np.maximum(d2, 0.0, out=d2)
        out[start:stop] = np.sqrt(d2.min(axis=1))
    return out


def _nearest_from_knn(graph, selected):
    """One-hop frontier distance on the KNN graph.

    For every unselected point that is a stored neighbour of at least one
    selected point, the score is the smallest such edge weight; all other points
    are ``inf``. This grows the selection one neighbour-ring at a time along the
    graph's own edges (each selected neuron pulls in its nearest neighbours) —
    O(m·k) per press, versus a full multi-source shortest-path flood over the
    whole graph, which is what made repeated grows sluggish on large data.
    """
    indices = np.asarray(graph.indices)
    dists = np.asarray(graph.dists, dtype=np.float64)
    n = indices.shape[0]

    sel_mask = np.zeros(n, dtype=bool)
    sel_mask[selected] = True

    # Gather the outgoing edges of the selected points (the candidate frontier).
    neigh = indices[selected]  # (m, k) neighbour row positions, -1 = missing
    ndist = dists[selected]  # (m, k) edge weights
    # Keep edges to real, not-already-selected neighbours.
    valid = (neigh >= 0) & ~sel_mask[np.clip(neigh, 0, n - 1)]

    score = np.full(n, np.inf)
    # A candidate adjacent to several selected points keeps its smallest edge.
    np.minimum.at(score, neigh[valid], ndist[valid])
    return score
