"""Tests for the figure's chunked point visuals.

``make_points`` builds one ``gfx.Points`` per (marker, spatial chunk) rather
than one per marker, so the renderer can later cull and decimate per chunk.
Everything downstream addresses points through ``vis._point_ix``, so the
invariants that matter are that the chunks partition the points exactly and
that each chunk's uploaded attributes line up with its own indices.

Selection is deliberately *not* read off the visuals any more: it must consider
every point, including those a future draw budget leaves undrawn.
"""

from __future__ import annotations

import numpy as np
import pytest

@pytest.fixture
def chunked(scatter_factory, force_chunks):
    """A figure whose points are split into several chunks per marker.

    Production thresholds would leave a test-sized scatter unchunked, so the
    chunk settings are lowered - but still applied through the real
    `set_points` -> `make_visuals` path, so the visuals are wired up (hover
    handlers included) exactly as they are in the app.

    Uses `scatter_factory` rather than `make_scatter`: these tests force
    chunking anyway, so running them under both of `make_scatter`'s layouts
    would just duplicate every result.
    """

    def _make(**kwargs):
        kwargs.setdefault("n", 6000)
        for key, value in force_chunks.items():
            kwargs.setdefault(key, value)
        return scatter_factory(**kwargs)

    return _make


def _markers(n, groups=("a", "b", "c"), seed=0):
    return np.random.default_rng(seed).choice(list(groups), n)


# --------------------------------------------------------------------------- #
# Chunk structure
# --------------------------------------------------------------------------- #


def test_points_are_split_into_several_chunks(chunked):
    fig, _, _ = chunked(n=6000)
    assert len(fig.point_visuals) > 1



def test_each_chunk_holds_exactly_one_marker(chunked):
    """A `gfx.Points` carries a single marker in its material."""
    markers = _markers(6000)
    fig, _, _ = chunked(n=6000, markers=markers)

    for vis in fig.point_visuals:
        in_chunk = fig.marker_symbols_at(vis._point_ix)
        assert set(in_chunk) == {vis.material.marker}


def test_cached_chunk_bounds_contain_their_points(chunked):
    """The figure caches one (n_chunks, 4) array, aligned with `point_visuals`.

    Reading pygfx's per-object box for ~1000 chunks costs ~16ms, which is far
    too slow to repeat per frame, so the view test gets a single array instead.
    """
    fig, _, points = chunked(n=4000)
    assert fig._chunks.bounds.shape == (len(fig.point_visuals), 4)

    for i, vis in enumerate(fig.point_visuals):
        pts = points[vis._point_ix]
        xmin, ymin, xmax, ymax = fig._chunks.bounds[i]
        assert pts[:, 0].min() >= xmin - 1e-6
        assert pts[:, 1].min() >= ymin - 1e-6
        assert pts[:, 0].max() <= xmax + 1e-6
        assert pts[:, 1].max() <= ymax + 1e-6


def test_chunk_bounds_follow_moved_points(chunked):
    """Cached bounds must be refreshed when positions change."""
    fig, _, _ = chunked(n=4000)
    before = fig._chunks.bounds.copy()

    fig.positions = (fig.positions + 10.0).astype(np.float32)
    fig.update_point_position()

    assert np.allclose(fig._chunks.bounds, before + 10.0)


def test_draw_order_matches_the_visuals(chunked):
    """The layout's arrays must stay aligned with `point_visuals`."""
    fig, _, _ = chunked(n=6000, markers=_markers(6000))
    assert len(fig._chunks.starts) == len(fig.point_visuals) + 1
    assert np.array_equal(np.sort(fig._chunks.order), np.arange(6000))

    for i, vis in enumerate(fig.point_visuals):
        assert np.array_equal(
            vis._point_ix, fig._chunks.order[fig._chunks.slice_for(i)]
        )
        # ...and be a *view* into it, not another copy of the indices.
        # (`base` can collapse through a dtype view, so compare the memory.)
        assert np.shares_memory(vis._point_ix, fig._chunks.order)


def test_selection_overlay_is_not_chunked(chunked):
    """The overlay is neither culled nor decimated, so chunking it is waste."""
    fig, _, _ = chunked(n=6000, markers=_markers(6000))
    fig.selected = np.arange(0, 6000, 2)

    # One visual per marker present in the selection, not per spatial chunk.
    assert len(fig.highlight_visuals) <= len(set(fig.marker_symbols_at(fig.selected)))


@pytest.mark.parametrize("attr", ["positions", "colors"])
def test_chunks_share_one_backing_array_per_marker_group(chunked, attr):
    """Chunks are views, so N points cost one array, not one per chunk.

    Both halves matter: every chunk must *be* a view (a copy would multiply the
    memory), and they must all view the same few arrays (one per marker group).
    """
    markers = _markers(6000, groups=("a", "b"))
    fig, _, _ = chunked(n=6000, markers=markers)
    assert len(fig.point_visuals) > 4

    bases = []
    for vis in fig.point_visuals:
        data = getattr(vis.geometry, attr).data
        # Disjoint slices of one array do not share memory with each other, so
        # group them by the parent they are views of.
        assert data.base is not None, f"{attr} chunk is a copy, not a view"
        bases.append(id(data.base))

    assert len(set(bases)) <= 2, f"expected one {attr} array per marker group"


# --------------------------------------------------------------------------- #
# Selection reads all points, not just the drawn ones
# --------------------------------------------------------------------------- #


def test_selection_ignores_what_is_drawn(chunked):
    """Hiding every visual must not change what a lasso selects."""
    fig, _, points = chunked(n=6000)
    polygon = np.array([[0.1, 0.1], [0.9, 0.2], [0.5, 0.9]])

    fig.select_points_lasso(polygon)
    with_visuals = fig.selected.copy()

    for vis in fig.point_visuals:
        vis.visible = False
    fig.select_points_lasso(polygon)

    assert np.array_equal(fig.selected, with_visuals)
    assert len(with_visuals), "the test polygon should catch something"



def test_moving_points_marks_the_layout_stale(chunked):
    """Chunk *membership* goes stale when points move, not just the bounds."""
    fig, _, _ = chunked(n=6000)
    assert not fig._chunks_stale

    fig.positions = np.random.default_rng(7).random((6000, 2)).astype(np.float32)
    fig.update_point_position()
    assert fig._chunks_stale, "a move must flag the grouping as stale"


def test_rechunk_restores_locality_after_a_move(chunked):
    """After re-chunking, chunk extents must be small again.

    A move leaves each chunk holding points that are no longer neighbours, so
    its box balloons to span most of the scene - which would leave per-chunk
    culling with nothing to cull.
    """
    # More chunks than the default so coherent vs. scrambled extents separate
    # clearly: 32 chunks of a unit square cover ~1/32 each when coherent, but
    # nearly all of it once their points are scattered.
    fig, _, _ = chunked(n=6000, chunk_target=32)

    def mean_area(f):
        b = f._chunks.bounds
        return float(np.nanmean((b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])))

    before = mean_area(fig)

    fig.positions = np.random.default_rng(7).random((6000, 2)).astype(np.float32)
    fig.update_point_position()
    scrambled = mean_area(fig)
    assert scrambled > 5 * before, "a move should balloon the chunk extents"

    fig.rechunk()
    assert not fig._chunks_stale
    assert mean_area(fig) < scrambled / 5, "re-chunking should shrink them again"

    # ...and the layout must still describe the visuals correctly.
    covered = np.sort(np.concatenate([v._point_ix for v in fig.point_visuals]))
    assert np.array_equal(covered, np.arange(6000))
    for i, vis in enumerate(fig.point_visuals):
        assert np.array_equal(
            vis._point_ix, fig._chunks.order[fig._chunks.slice_for(i)]
        )


def test_rechunk_is_a_noop_when_nothing_moved(chunked):
    fig, _, _ = chunked(n=6000)
    before = [v._point_ix for v in fig.point_visuals]
    fig.rechunk()
    assert [v._point_ix for v in fig.point_visuals] == before


def test_finishing_an_animated_move_rechunks(chunked):
    """`process_moves` must settle the layout once the points stop moving."""
    fig, _, _ = chunked(n=6000, chunk_target=32)

    target = np.random.default_rng(7).random((6000, 2)).astype(np.float32)
    fig.move_points(target, n_frames=3)

    # Drive the animation to completion, as the render loop would.
    for _ in range(4):
        fig.process_moves()

    assert not hasattr(fig, "to_move"), "the animation should have finished"
    assert not fig._chunks_stale, "finishing a move must restore chunk locality"
    assert np.allclose(fig.positions, target, atol=1e-5)

    covered = np.sort(np.concatenate([v._point_ix for v in fig.point_visuals]))
    assert np.array_equal(covered, np.arange(6000))
