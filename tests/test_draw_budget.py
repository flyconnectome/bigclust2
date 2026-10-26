"""Tests for view culling and the per-frame draw budget.

At tens of millions of points, drawing everything is the bottleneck: the cost
is linear in the number of points submitted, and zoomed out almost all of them
land on the same few pixels. So the figure draws only the chunks in view, and
only as many points from each as a budget allows.

What makes that honest rather than arbitrary is the ordering: points within a
chunk are shuffled at build time, so the drawn prefix is an unbiased sample of
that chunk. Zooming in narrows the view, fewer chunks share the budget, and the
survivors are drawn more densely until everything in view is on screen.

Nothing is re-uploaded as the budget changes - the points stay resident and
only `Buffer.draw_range` moves.
"""

from __future__ import annotations

import numpy as np
import pytest



# --------------------------------------------------------------------------- #
# The figure applies it
# --------------------------------------------------------------------------- #


@pytest.fixture
def budgeted(chunked):
    """A chunked figure with the draw budget forced to bite."""

    def _make(budget, **kwargs):
        fig, meta, points = chunked(**kwargs)
        fig.draw_budget = budget
        fig.draw_min_per_chunk = 1
        fig.renderer.render(fig.scene, fig.camera)
        fig.canvas.draw()
        fig._apply_draw_budget()
        return fig, meta, points

    return _make


def test_everything_is_drawn_under_a_generous_budget(budgeted):
    fig, _, _ = budgeted(budget=10**9)
    assert fig.drawn_point_count == 6000
    assert all(v.visible for v in fig.point_visuals)


def test_a_tight_budget_decimates(budgeted):
    fig, _, _ = budgeted(budget=600)
    assert fig.drawn_point_count <= 600
    assert fig.drawn_point_count > 0


def test_decimation_only_moves_draw_range(budgeted):
    """The points stay resident; nothing is re-uploaded."""
    fig, _, points = budgeted(budget=600)

    for vis in fig.point_visuals:
        # The buffer still holds every point of the chunk...
        assert len(vis.geometry.positions.data) == len(vis._point_ix)
        # ...only the drawn range is shorter.
        offset, count = vis.geometry.positions.draw_range
        assert offset == 0
        assert count <= len(vis._point_ix)


def test_raising_the_budget_draws_more(budgeted):
    fig, _, _ = budgeted(budget=600)
    tight = fig.drawn_point_count

    fig.draw_budget = 10**9
    fig._apply_draw_budget()

    assert fig.drawn_point_count > tight
    assert fig.drawn_point_count == 6000


def test_selection_is_unaffected_by_decimation(budgeted):
    """Selection reads `positions`, so it must not depend on the budget."""
    fig, _, points = budgeted(budget=100)
    bounds = np.array([[0.2, 0.2], [0.8, 0.8]])

    fig.select_points(bounds)
    decimated = fig.selected.copy()

    fig.draw_budget = 10**9
    fig._apply_draw_budget()
    fig.select_points(bounds)

    assert np.array_equal(fig.selected, decimated)
    expected = np.where(
        (points[:, 0] >= 0.2)
        & (points[:, 0] <= 0.8)
        & (points[:, 1] >= 0.2)
        & (points[:, 1] <= 0.8)
    )[0]
    assert np.array_equal(decimated, expected)


def test_repeated_passes_are_a_no_op(budgeted):
    """It runs per frame, so an unchanged view must short-circuit."""
    fig, _, _ = budgeted(budget=600)
    before = [v.geometry.positions.draw_range for v in fig.point_visuals]

    fig._render_stale = False
    fig._apply_draw_budget()

    assert [v.geometry.positions.draw_range for v in fig.point_visuals] == before
    assert not fig._render_stale, "an unchanged view should not force a redraw"


def test_rebuilding_the_layout_reapplies_the_budget(budgeted):
    fig, _, _ = budgeted(budget=600)
    fig.positions = np.random.default_rng(9).random((6000, 2)).astype(np.float32)
    fig.update_point_position()
    fig.rechunk()

    fig._apply_draw_budget()
    assert fig.drawn_point_count <= 600


# --------------------------------------------------------------------------- #
# The Settings control
# --------------------------------------------------------------------------- #


def test_setting_the_budget_marks_the_figure_for_redraw(make_scatter):
    """The slider just assigns the property, so the property must do the work."""
    fig, _, _ = make_scatter(n=50)
    fig._render_stale = False

    fig.draw_budget = 1_234_567

    assert fig.draw_budget == 1_234_567
    assert fig._render_stale, "a new budget has to reach the screen"


def test_the_budget_is_coerced_to_an_int(make_scatter):
    fig, _, _ = make_scatter(n=50)
    fig.draw_budget = 2.5e6
    assert fig.draw_budget == 2_500_000
    assert isinstance(fig.draw_budget, int)


@pytest.mark.parametrize(
    "budget, label",
    [(100_000, "100k"), (500_000, "500k"), (3_000_000, "3.0M"), (20_000_000, "20.0M")],
)
def test_the_slider_label_formats_the_budget(budget, label):
    from bigclust2.gui.controls.scatter_control import ScatterControls

    assert ScatterControls._format_draw_budget(budget) == label
