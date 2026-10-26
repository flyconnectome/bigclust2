"""Tests for the selection overlay under level-of-detail drawing.

The overlay rings the selected points. It used to be one unchunked visual
drawn in full, which was fine until the scatter started culling and sampling:
zoomed in on a large project the scatter would draw a few hundred thousand
points while the overlay still drew every one of several million selected
ones - more work for the annotation than for the thing being annotated.

A large selection now *follows* the scatter: each overlay visual carries one
chunk's selected points in the scatter's own draw order, so the selected
members of the prefix the scatter draws are a prefix of the overlay too. The
property that buys is exact, and is what these tests pin: the overlay rings
the selected points the scatter drew, no more and no less.

Small selections keep the old behaviour - aligning costs a pass over every
point in the project, which is not worth paying to ring a handful.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def selectable(chunked, look_at):
    """A chunked figure with the budget forced low enough to decimate.

    `select` deliberately does *not* reset `_draw_budget_key`: forcing the
    budget pass after a selection change is the figure's own job, and resetting
    it here would hide whether it does it.
    """

    def _make(budget=600, **kwargs):
        fig, meta, points = chunked(**kwargs)
        fig.draw_budget = budget
        fig.draw_min_per_chunk = 1
        look_at(fig, 0, 1, 0, 1)

        def select(ix):
            fig.selected = np.asarray(ix)
            fig._apply_draw_budget()

        return fig, select, points

    return _make


# --------------------------------------------------------------------------- #
# Which path is taken
# --------------------------------------------------------------------------- #


def test_a_small_selection_is_drawn_whole(selectable, drawn_points):
    fig, select, _ = selectable(budget=10_000)
    select(np.arange(50))

    assert fig._highlight is None
    assert drawn_points(fig.highlight_visuals) == set(range(50))


def test_a_large_selection_follows_the_scatter(selectable):
    fig, select, _ = selectable(budget=600)
    select(np.arange(3000))

    assert fig._highlight.structure == fig._chunks.structure
    assert fig._highlight is not None


def test_the_threshold_tracks_the_draw_budget(selectable):
    """Lowering the budget has to tighten the overlay with it."""
    fig, _, _ = selectable(budget=10_000)
    assert fig.highlight_whole_limit == 1_000

    fig.draw_budget = 600
    assert fig.highlight_whole_limit == 60


def test_the_limit_is_at_least_one(selectable):
    """So a tiny budget cannot make every selection take the aligned path."""
    fig, _, _ = selectable(budget=600)
    fig.draw_budget = 1
    assert fig.highlight_whole_limit == 1


# --------------------------------------------------------------------------- #
# The guarantee
# --------------------------------------------------------------------------- #


def test_the_overlay_rings_exactly_what_the_scatter_drew(selectable, drawn_points):
    """Equality both ways: no drawn point looks unselected, no ring floats.

    The first half is the one that would read as a bug - a point plainly on
    screen, inside the lasso you just drew, with no ring on it.
    """
    fig, select, _ = selectable(budget=600)
    selected = np.arange(0, 6000, 2)

    select(selected)

    scatter = drawn_points(fig.point_visuals)
    assert scatter, "the scatter has to be drawing something"
    assert drawn_points(fig.highlight_visuals) == scatter & set(selected.tolist())


def test_raising_the_budget_rings_more(selectable, drawn_points):
    fig, select, _ = selectable(budget=600)
    select(np.arange(0, 6000, 2))
    tight = len(drawn_points(fig.highlight_visuals))

    fig.draw_budget = 10**9  # part of the budget key, so this forces the pass
    fig._apply_draw_budget()

    assert len(drawn_points(fig.highlight_visuals)) > tight
    assert drawn_points(fig.highlight_visuals) == set(range(0, 6000, 2))


def test_zooming_in_culls_the_overlay_with_the_scatter(selectable, drawn_points, look_at):
    """The case the whole change is for: the overlay must cull, not just LOD.

    The budget is set high enough that nothing is decimated, so what the zoom
    changes is culling alone - but still low enough that the selection is over
    `highlight_whole_limit` and takes the following path.
    """
    fig, select, points = selectable(budget=10_000)
    select(np.arange(6000))
    assert len(drawn_points(fig.highlight_visuals)) == 6000

    look_at(fig, 0.4, 0.6, 0.4, 0.6)
    fig._apply_draw_budget()

    overlay = drawn_points(fig.highlight_visuals)
    assert len(overlay) < 6000, "the overlay was not culled"
    assert overlay == drawn_points(fig.point_visuals)


# --------------------------------------------------------------------------- #
# Staying consistent
# --------------------------------------------------------------------------- #


def test_a_new_selection_reaches_the_screen_on_an_unchanged_view(selectable, drawn_points):
    """The budget pass short-circuits on an unchanged view; selecting must not.

    A fresh overlay starts out drawing its buffers in full, so "it changed" is
    not enough to prove the pass ran - it has to have been *ranged*.
    """
    fig, select, _ = selectable(budget=600)
    select(np.arange(0, 6000, 2))

    selected = np.arange(1, 6000, 2)
    fig.selected = selected
    fig._apply_draw_budget()  # the selection alone has to force the pass

    assert drawn_points(fig.highlight_visuals) == (
        drawn_points(fig.point_visuals) & set(selected.tolist())
    )


def test_rechunking_rebuilds_the_overlay_against_the_new_layout(selectable, drawn_points):
    fig, select, _ = selectable(budget=600)
    select(np.arange(0, 6000, 2))

    fig.positions = np.random.default_rng(9).random((6000, 2)).astype(np.float32)
    fig.update_point_position()
    fig.rechunk()
    fig._apply_draw_budget()

    assert fig._highlight.structure == fig._chunks.structure
    assert drawn_points(fig.highlight_visuals) == (
        drawn_points(fig.point_visuals) & set(range(0, 6000, 2))
    )


def test_a_stale_overlay_is_left_alone_rather_than_crashing(selectable):
    """Chunk numbers from a previous layout address nothing in the new one.

    `rechunk` rebuilds the overlay alongside the visuals, so this is a guard
    rather than a path the app takes - but a layout with fewer chunks than the
    overlay remembers would index past the end of the draw counts.
    """
    from bigclust2.chunking import ChunkLayout

    fig, select, _ = selectable(budget=600)
    select(np.arange(0, 6000, 2))
    assert fig._highlight.chunks.max() > 0

    # A new layout with a single chunk: every remembered chunk number but 0 is
    # now out of range.
    n = len(fig)
    fig._chunks = ChunkLayout(
        order=np.arange(n, dtype=np.intp),
        starts=np.array([0, n], dtype=np.intp),
        bounds=np.array([[0.0, 0.0, 1.0, 1.0]]),
    )
    fig._apply_draw_budget()  # must not raise

    assert fig._highlight.structure != fig._chunks.structure


def test_clearing_the_selection_removes_the_overlay(selectable):
    fig, select, _ = selectable(budget=600)
    select(np.arange(0, 6000, 2))
    assert fig.highlight_visuals

    fig.selected = None

    assert fig.highlight_visuals == []
    assert fig._highlight is None


def test_an_overlay_drawn_whole_stays_a_fraction_of_the_budget(selectable, look_at):
    """The whole path is not culled, so its size is what bounds its cost.

    `highlight_whole_limit` is the contract: an overlay drawn in full can
    never be more than a tenth of what the scatter itself is allowed to draw,
    whatever the view.
    """
    fig, select, _ = selectable(budget=30_000)
    select(np.arange(3000))
    assert fig._highlight is None, "this has to be the whole path"

    look_at(fig, 0.4, 0.6, 0.4, 0.6)
    fig._apply_draw_budget()

    overlay = sum(v._pos_buffer.draw_range[1] for v in fig.highlight_visuals if v.visible)
    assert overlay <= fig.highlight_whole_limit <= fig.draw_budget // 10


def test_the_overlay_keeps_following_while_the_points_move(selectable, drawn_points):
    """Regression: a bounds refresh is not a rebuild.

    Every frame of a move animation calls `refresh_bounds`, which moves the
    chunk boxes but leaves chunk *membership* alone - so the ranks the overlay
    carries stay valid. Keying the overlay on the layout's general version
    made it bail there, freezing the rings while the scatter kept re-culling.
    """
    fig, select, _ = selectable(budget=600)
    select(np.arange(0, 6000, 2))

    before = fig._chunks.structure
    fig.positions = np.random.default_rng(9).random((6000, 2)).astype(np.float32)
    fig.update_point_position()  # one animation frame; no rechunk yet

    assert fig._chunks.version != before, "a refresh has to bump the version"
    assert fig._chunks.structure == before, "...but not the structure"

    fig._apply_draw_budget()
    assert drawn_points(fig.highlight_visuals) == (
        drawn_points(fig.point_visuals) & set(range(0, 6000, 2))
    )


def test_lowering_the_budget_rebuilds_an_overlay_that_no_longer_qualifies(
    selectable, drawn_points
):
    """Regression: the path was latched at selection time, the limit was live.

    Select while the budget is generous and the selection is drawn whole; drop
    the budget and the limit drops with it, leaving an unculled overlay drawing
    more than the scatter - the exact thing the follow path exists to prevent.
    """
    fig, select, _ = selectable(budget=30_000)
    select(np.arange(2000))
    assert fig._highlight is None, "this has to start on the whole path"

    fig.draw_budget = 600
    fig._apply_draw_budget()

    assert fig._highlight is not None, "it should have switched to following"
    assert len(drawn_points(fig.highlight_visuals)) <= fig.drawn_point_count


def test_raising_the_budget_goes_back_to_drawing_the_selection_whole(selectable):
    fig, select, _ = selectable(budget=600)
    select(np.arange(2000))
    assert fig._highlight is not None

    fig.draw_budget = 10**9

    assert fig._highlight is None


def test_loading_a_new_dataset_drops_the_overlay(selectable):
    """Regression: the visuals outlived the points they indexed.

    `set_points` rebuilds the scatter without going through `highlight_points`,
    so the overlay was left holding indices into the previous dataset - and the
    next position update walked them straight off the end.
    """
    fig, select, _ = selectable(budget=600)
    select(np.arange(0, 6000, 2))
    assert fig.highlight_visuals

    meta = pd.DataFrame(
        {
            "id": np.arange(20),
            "label": np.full(20, "x", dtype=object),
            "dataset": np.full(20, "ds", dtype=object),
            "_color": np.full(20, "red", dtype=object),
        }
    )
    points = np.random.default_rng(2).random((20, 2)).astype(np.float32)
    fig.set_points(points=points, metadata=meta, color_col="_color")

    assert fig.highlight_visuals == []
    assert fig._highlight is None
    fig.update_point_position()  # must not walk off the end of the new data
