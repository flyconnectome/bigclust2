"""Tests for the label system's cost at scale.

Labels are only ever *shown* for a few hundred points (`label_vis_limit`), but
deciding which ones used to cost a pass over every point in the project, and
the per-point bookkeeping was a dense list with one `None` per point. At tens
of millions of points that is a multi-second freeze on every label tick and a
gigabyte of `None`.

The behaviour must not change: the tests below pin the new paths against the
straightforward full scan they replace, rather than against hand-written
expectations that could drift along with the implementation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from bigclust2.utils import SparseMap


# --------------------------------------------------------------------------- #
# SparseMap
# --------------------------------------------------------------------------- #


def test_sparse_map_reads_as_a_list_of_none():
    s = SparseMap(5)
    assert len(s) == 5
    assert all(s[i] is None for i in range(5))


def test_sparse_map_round_trips_values():
    s = SparseMap(5)
    s[2] = "a"
    assert s[2] == "a"
    assert s[1] is None
    assert dict(s.items()) == {2: "a"}
    assert list(s.values()) == ["a"]


def test_sparse_map_handles_negative_indices_like_a_list():
    s = SparseMap(5)
    s[-1] = "last"
    assert s[4] == "last"
    assert s[-1] == "last"


def test_assigning_none_clears_the_entry():
    """So that `items()` never has to filter, which is the whole point."""
    s = SparseMap(5)
    s[3] = "x"
    s[3] = None
    assert s[3] is None
    assert dict(s.items()) == {}


@pytest.mark.parametrize("ix", [5, 6, -6, -99])
def test_out_of_range_raises(ix):
    with pytest.raises(IndexError):
        SparseMap(5)[ix]


def test_sparse_map_is_not_iterable():
    """`for v in s` could only yield the set values, disagreeing with `len`."""
    with pytest.raises(TypeError):
        list(SparseMap(3))


# --------------------------------------------------------------------------- #
# The figure's visibility query
# --------------------------------------------------------------------------- #


def _full_scan(fig):
    """What `_visible_label_indices` replaced: test every point."""
    return np.where(fig.is_visible_pos(fig.positions))[0]


@pytest.fixture
def viewed(make_scatter):
    """A rendered figure plus a helper to point the camera somewhere."""

    def _make(**kwargs):
        kwargs.setdefault("n", 400)
        fig, meta, points = make_scatter(**kwargs)

        def look_at(x0, x1, y0, y1):
            fig.camera.show_rect(x0, x1, y0, y1)
            fig.renderer.render(fig.scene, fig.camera)
            fig.canvas.draw()

        look_at(0, 1, 0, 1)
        return fig, look_at, points

    return _make


def test_a_tall_sliver_overlaps_many_chunks_but_contains_few_points(viewed):
    """The shape the containment fast path cannot help with."""
    fig, look_at, _ = viewed()
    look_at(0.45, 0.55, -5.0, 5.0)

    got = fig._visible_label_indices(len(fig))

    assert got is not None
    assert np.array_equal(got, _full_scan(fig))


def test_an_empty_view_is_an_empty_answer_not_a_bail(viewed):
    """`None` means "too many", so nothing visible must not report it."""
    fig, look_at, _ = viewed()
    look_at(5.0, 6.0, 5.0, 6.0)

    got = fig._visible_label_indices(0)
    assert got is not None
    assert len(got) == 0


def test_without_a_viewport_everything_counts_as_visible(viewed, monkeypatch):
    """The safe answer for a caller that uses this to *hide* things."""
    fig, _, _ = viewed(n=50)
    monkeypatch.setattr(fig, "view_bounds", lambda: None)

    assert np.array_equal(fig._visible_label_indices(50), np.arange(50))
    assert fig._visible_label_indices(49) is None


def test_random_views_agree_with_the_full_scan(viewed):
    """The cheap path has to be exact, not approximately right."""
    fig, look_at, _ = viewed(n=500, seed=3)
    rng = np.random.default_rng(0)

    for _ in range(25):
        cx, cy = rng.random(2)
        w = float(10 ** rng.uniform(-2, 0.5))
        look_at(cx - w, cx + w, cy - w, cy + w)

        expected = _full_scan(fig)
        for limit in (0, 3, 25, len(fig)):
            got = fig._visible_label_indices(limit)
            if len(expected) > limit:
                assert got is None, f"expected a bail at limit={limit}"
            else:
                assert np.array_equal(got, expected), f"limit={limit}"


# --------------------------------------------------------------------------- #
# hide / highlight no longer scale with the project
# --------------------------------------------------------------------------- #


def test_hiding_everything_only_touches_labels_that_exist(viewed):
    fig, _, _ = viewed(n=200)
    fig.show_labels([1, 7, 9])
    assert len(list(fig.label_visuals.values())) == 3

    fig.hide_labels()

    assert not any(v.visible for v in fig.label_visuals.values())
    assert len(list(fig.label_visuals.values())) == 3, "hiding must not create any"


def test_highlighting_does_not_build_a_visual_per_match(viewed):
    """A match can run to thousands of points; each visual costs about a ms."""
    fig, _, _ = viewed(n=200)
    fig.labels = np.array(["a"] * 150 + ["b"] * 50, dtype=object)
    assert len(list(fig.label_visuals.values())) == 0

    fig.highlight_labels("a")

    assert fig._label_highlighted == set(range(150))
    assert len(list(fig.label_visuals.values())) == 0


def test_a_label_created_later_picks_up_its_highlight(viewed):
    """Which is what makes the lazy highlight invisible to the user."""
    fig, _, _ = viewed(n=200)
    fig.labels = np.array(["a"] * 150 + ["b"] * 50, dtype=object)
    fig.highlight_labels("a", color="y")

    fig.show_labels([0, 199])

    assert fig.label_visuals[0].material.color.rgba[:3] == (1, 1, 0)
    assert fig.label_visuals[199].material.color.rgba[:3] != (1, 1, 0)


def test_highlighting_recolors_labels_that_already_exist(viewed):
    fig, _, _ = viewed(n=200)
    fig.labels = np.array(["a"] * 150 + ["b"] * 50, dtype=object)
    fig.show_labels([3])

    fig.highlight_labels("a", color="y")

    assert fig.label_visuals[3].material.color.rgba[:3] == (1, 1, 0)


def test_clearing_the_highlight_restores_the_color(viewed):
    fig, _, _ = viewed(n=200)
    fig.labels = np.array(["a"] * 200, dtype=object)
    fig.show_labels([3])
    original = fig.label_visuals[3].material.color.rgba

    fig.highlight_labels("a", color="y")
    fig.highlight_labels(None)

    assert fig.label_visuals[3].material.color.rgba == original
    assert fig._label_highlighted == set()


def test_highlighting_a_missing_label_matches_nothing(viewed):
    fig, _, _ = viewed(n=20)
    fig.labels = np.array(["a"] * 20, dtype=object)

    fig.highlight_labels("not-a-label")

    assert fig._label_highlighted == set()


# --------------------------------------------------------------------------- #
# make_label_lines
# --------------------------------------------------------------------------- #


def test_label_lines_skip_groups_too_small_to_enclose(viewed):
    fig, _, _ = viewed(n=60)
    # Only "a" has the three points a hull needs.
    fig.labels = np.array(["a"] * 58 + ["b", "c"], dtype=object)

    fig.make_label_lines()

    assert len(fig.label_line_group.children) == 1


def test_no_group_is_large_enough_leaves_the_mesh_empty(viewed):
    """Regression: concatenating an empty list of hulls used to raise."""
    fig, _, _ = viewed(n=20)
    fig.labels = np.array([f"g{i}" for i in range(20)], dtype=object)

    fig.make_label_lines()  # must not raise

    assert len(fig.label_line_group.children) == 0


def test_label_lines_match_the_per_value_grouping(viewed):
    """Pinned against the `np.where(labels == l)` scan the sort replaced.

    Drives `make_label_lines` itself and compares its hulls against hulls built
    from the per-value grouping, so it fails if the sort ever groups wrongly.
    """
    from scipy.spatial import ConvexHull

    fig, _, points = viewed(n=90, seed=11)
    labels = np.array([f"g{i % 5}" for i in range(90)], dtype=object)
    fig.labels = labels

    fig.make_label_lines()
    (mesh,) = fig.label_line_group.children
    got = {tuple(v) for v in mesh.geometry.positions.data[:, :2].tolist()}

    expected = set()
    for value in pd.unique(labels):
        member = fig.positions[np.where(labels == value)[0]]
        hull = ConvexHull(member)
        expected.update(tuple(v) for v in member[hull.vertices].tolist())

    assert got == expected


# --------------------------------------------------------------------------- #
# Cost, not just correctness
# --------------------------------------------------------------------------- #


def test_a_crowded_view_bails_without_touching_a_single_point(chunked, monkeypatch):
    """The zoomed-out case is the common one, and the one that has to be cheap.

    Chunks lying wholly inside the viewport are counted from their sizes, so
    once those alone are over the limit the answer is known without reading any
    positions. Without this the query is back to a pass over every point - the
    results would still be right, so only a cost test can catch the regression.
    """
    fig, _, _ = chunked()
    fig.camera.show_rect(0, 1, 0, 1)
    fig.renderer.render(fig.scene, fig.camera)
    fig.canvas.draw()

    def _boom(i):
        raise AssertionError(f"chunk {i} should not have been read point by point")

    monkeypatch.setattr(fig._chunks, "points_in", _boom)

    assert fig._visible_label_indices(400) is None


def test_the_visibility_tick_shows_the_labels_in_view(viewed):
    """End to end through the animation callback, as the figure runs it."""
    fig, look_at, points = viewed(n=400, seed=5)
    tick = fig._label_visibility_tick
    fig.label_refresh_rate = 1
    # Isolate "which labels does the tick pick" from the declutter solver,
    # which hides labels it cannot place without overlap.
    fig.smart_label_placement = False

    # Zoomed out: everything is over the limit, so nothing is labelled.
    look_at(0, 1, 0, 1)
    fig.label_vis_limit = 10
    tick()
    assert not any(v.visible for v in fig.label_visuals.values())

    # Zoomed in far enough, the labels for the points in view come up.
    look_at(0.35, 0.65, 0.35, 0.65)
    fig.label_vis_limit = 400
    tick()

    expected = set(_full_scan(fig).tolist())
    shown = {ix for ix, v in fig.label_visuals.items() if v.visible}
    assert shown == expected
    assert expected, "the zoomed-in view has to contain something to label"


def test_an_unchunked_figure_falls_back_to_the_full_scan(viewed, monkeypatch):
    """Not every figure has a layout - `make_points(chunk=False)` builds none.

    Reporting "nothing visible" there would silently hide every label.
    """
    fig, look_at, _ = viewed(n=200)
    look_at(0, 1, 0, 1)
    monkeypatch.setattr(fig, "_chunks", None)

    got = fig._visible_label_indices(len(fig))

    assert np.array_equal(got, _full_scan(fig))
    assert fig._visible_label_indices(0) is None


@pytest.mark.parametrize("declutter", [False, True])
def test_an_unchanged_tick_does_not_rewrite_visibility(viewed, declutter):
    """`visible` is tracked in pygfx: writing one notifies even if unchanged.

    The tick runs twice a second forever, so re-asserting every label each time
    is pure cost (the same trap `_apply_draw_budget` guards against). Run with
    decluttering both off and on - the placement solver re-asserts suppressed
    labels on its cached path, which is the default and was the one gap.
    """
    fig, look_at, _ = viewed(n=400, seed=5)
    fig.label_refresh_rate = 1
    fig.smart_label_placement = declutter
    look_at(0.35, 0.65, 0.35, 0.65)
    fig._label_visibility_tick()
    assert any(v.visible for v in fig.label_visuals.values()), "nothing to re-assert"

    writes = []
    cls = type(next(iter(fig.label_visuals.values())))
    original = cls.visible
    monkey = property(
        original.fget,
        lambda self, value: (writes.append(value), original.fset(self, value))[1],
    )
    cls.visible = monkey
    try:
        fig._label_visibility_tick()
    finally:
        cls.visible = original

    assert writes == [], f"an unchanged view rewrote {len(writes)} visibility flags"


def test_the_debug_report_answers_for_the_frame_the_tick_saw(viewed):
    """It is the tool you open *because* labels are missing - it must agree."""
    fig, look_at, _ = viewed(n=400, seed=5)
    look_at(0, 1, 0, 1)
    fig.label_vis_limit = 10

    report = fig.label_debug_report()

    assert fig._visible_label_indices(fig.label_vis_limit) is None
    assert "more than the limit" in report
    assert "Over the visibility limit" in report


def test_labels_are_hidden_once_they_leave_the_view(viewed):
    """The tick no longer blanket-hides, so leaving the view has to be seen."""
    fig, look_at, _ = viewed(n=400, seed=5)
    fig.label_refresh_rate = 1
    fig.smart_label_placement = False

    look_at(0.0, 0.3, 0.0, 0.3)
    fig._label_visibility_tick()
    first = {ix for ix, v in fig.label_visuals.items() if v.visible}
    assert first, "the first view has to label something"

    look_at(0.7, 1.0, 0.7, 1.0)
    fig._label_visibility_tick()
    second = {ix for ix, v in fig.label_visuals.items() if v.visible}

    assert second, "the second view has to label something too"
    assert not (first & second), "the two views must not overlap for this test"
    assert second == set(_full_scan(fig).tolist())


def test_resetting_placement_state_takes_down_what_it_forgets(viewed):
    """Regression: the tick hides incrementally, so the record must not vanish.

    `_reset_label_placement_state` clears `_shown_labels`. If labels were on
    screen at that moment, nothing afterwards knows they are up - the next tick
    only hides what it remembers showing - so they stay visible forever.
    Toggling smart placement is the reachable route to it.
    """
    fig, look_at, _ = viewed(n=400, seed=5)
    fig.label_refresh_rate = 1
    fig.smart_label_placement = False

    look_at(0.0, 0.3, 0.0, 0.3)
    fig._label_visibility_tick()
    assert any(v.visible for v in fig.label_visuals.values()), "nothing was shown"

    # Any reset of the placement state, mid-flight
    fig.smart_label_placement = True
    fig.smart_label_placement = False

    look_at(0.7, 1.0, 0.7, 1.0)
    fig._label_visibility_tick()

    shown = {ix for ix, v in fig.label_visuals.items() if v.visible}
    assert shown == set(_full_scan(fig).tolist())


def test_loading_a_smaller_dataset_over_a_larger_one(viewed):
    """Regression: `set_points` replaces the label store before the reset runs.

    Indices remembered from the previous, bigger dataset then address nothing,
    and looking them up in the new store raised `IndexError`.
    """
    fig, look_at, _ = viewed(n=400, seed=5)
    fig.label_refresh_rate = 1
    fig.smart_label_placement = False
    look_at(0.0, 0.3, 0.0, 0.3)
    fig._label_visibility_tick()
    assert max(fig._shown_labels) > 20, "need a remembered index beyond the new size"

    meta = pd.DataFrame(
        {
            "id": np.arange(20),
            "label": [f"p{i}" for i in range(20)],
            "dataset": np.full(20, "ds", dtype=object),
            "_color": np.full(20, "red", dtype=object),
        }
    )
    points = np.random.default_rng(2).random((20, 2)).astype(np.float32)

    fig.set_points(points=points, metadata=meta, color_col="_color")  # must not raise

    assert len(fig) == 20
    assert fig._shown_labels == set()
