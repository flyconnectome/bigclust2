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

SOURCE_EMBEDDING = "embedding"
SOURCE_DISTANCES = "distances"
SOURCE_FEATURES = "features"
SOURCE_KNN = "knn"

# Display / auto-default preference order. Embedding comes first because it is
# the default source (the layout the user actually sees on screen).
_SOURCE_ORDER = (SOURCE_EMBEDDING, SOURCE_KNN, SOURCE_DISTANCES, SOURCE_FEATURES)


class GrowShrinkUnavailable(Exception):
    """Raised when a requested distance source is not available on the figure."""


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
        return _nearest_from_coords(
            np.asarray(positions, dtype=np.float64), selected, "euclidean"
        )

    if source == SOURCE_FEATURES:
        if not dists or "features" not in dists:
            raise GrowShrinkUnavailable("No feature vectors available.")
        return _nearest_from_coords(
            np.asarray(dists["features"], dtype=np.float64), selected, metric
        )

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


def _nearest_from_coords(X, selected, metric):
    """Min distance from every row of ``X`` to the rows ``X[selected]``."""
    n = X.shape[0]
    if metric == "euclidean":
        from scipy.spatial import cKDTree

        dist, _ = cKDTree(X[selected]).query(X, k=1)
        return np.asarray(dist, dtype=np.float64)

    # Non-euclidean: compute the (chunk, m) distance block row-wise and take the
    # per-row minimum, capping the materialised block to ~2M entries.
    from sklearn.metrics import pairwise_distances

    sel_X = X[selected]
    out = np.empty(n, dtype=np.float64)
    chunk = max(1, int(2_000_000 // max(1, len(selected))))
    for start in range(0, n, chunk):
        stop = min(start + chunk, n)
        block = pairwise_distances(X[start:stop], sel_X, metric=metric)
        out[start:stop] = block.min(axis=1)
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
