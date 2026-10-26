"""Tests for marker classification and point-visual grouping.

Markers used to be carried as a per-point array of pygfx marker *names* and
re-derived on every ``make_points`` call - which happens on every selection
change, via ``highlight_points``. They are now classified once in
``set_points`` into small integer codes plus a list of names, and
``make_points`` groups by those codes.

The invariant the rest of the figure depends on is that every visual carries a
``_point_ix`` mapping its vertices back to figure rows, and that the union of
those indices is exactly the points asked for.
"""

from __future__ import annotations

import numpy as np
import pytest

from bigclust2.plotly_export import _resolve_marker_symbols
from bigclust2.scatter import AVAILABLE_MARKERS, marker_codes


def _oracle_marker_symbols(markers):
    """The mapping `make_points` used before codes were introduced."""
    unique_types = np.unique(markers)
    marker_map = dict(zip(unique_types, AVAILABLE_MARKERS))
    return np.array([marker_map[t] for t in markers], dtype=object)


# --------------------------------------------------------------------------- #
# marker_codes
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "markers",
    [
        np.random.default_rng(0).choice(list("dcba"), 200),
        np.array(["only"] * 20),
        np.random.default_rng(1).choice(np.array([3, 1, 2]), 200),
    ],
    ids=["strings", "single", "ints"],
)
def test_codes_reproduce_the_old_sorted_mapping(markers):
    codes, names = marker_codes(markers)
    got = np.asarray(names, dtype=object)[codes]
    assert np.array_equal(got, _oracle_marker_symbols(markers))


def test_codes_are_small_ints():
    """568MB of intp per figure at scatter sizes would be wasteful."""
    codes, _ = marker_codes(np.random.default_rng(0).choice(list("abc"), 100))
    assert codes.dtype == np.int8


def test_none_markers_means_one_default_symbol():
    codes, names = marker_codes(None)
    assert codes is None
    assert names == [AVAILABLE_MARKERS[0]] == ["circle"]


def test_too_many_distinct_markers_is_rejected():
    too_many = np.array([f"ds{i}" for i in range(len(AVAILABLE_MARKERS) + 1)])
    with pytest.raises(AssertionError, match="unique types are supported"):
        marker_codes(too_many)


def test_same_value_maps_to_same_symbol_regardless_of_row_order():
    a = np.array(["c", "a", "b", "a"])
    b = np.array(["a", "b", "c", "a"])
    codes_a, names_a = marker_codes(a)
    codes_b, names_b = marker_codes(b)
    assert names_a == names_b
    for value in "abc":
        assert names_a[codes_a[a == value][0]] == names_b[codes_b[b == value][0]]


# --------------------------------------------------------------------------- #
# make_points grouping (real visuals, offscreen canvas)
# --------------------------------------------------------------------------- #


def test_single_marker_builds_one_visual_covering_every_point(make_scatter):
    fig, _, points = make_scatter(n=150)
    assert len(fig.point_visuals) == 1

    vis = fig.point_visuals[0]
    assert np.array_equal(vis._point_ix, np.arange(150))
    assert np.allclose(vis.geometry.positions.data[:, :2], points)
    # Points are stored (N, 2); the visual gets z=0.
    assert np.allclose(vis.geometry.positions.data[:, 2], 0.0)


def test_multiple_markers_partition_the_points(make_scatter):
    markers = np.random.default_rng(0).choice(["a", "b", "c"], 150)
    fig, _, points = make_scatter(n=150, markers=markers)

    assert len(fig.point_visuals) == 3
    assert sorted(v.material.marker for v in fig.point_visuals) == sorted(
        AVAILABLE_MARKERS[:3]
    )

    # Exactly a partition: every point once, no gaps, no overlaps.
    covered = np.sort(np.concatenate([v._point_ix for v in fig.point_visuals]))
    assert np.array_equal(covered, np.arange(150))

    # Each visual holds the positions its indices point at.
    for vis in fig.point_visuals:
        assert np.allclose(vis.geometry.positions.data[:, :2], points[vis._point_ix])


def test_marker_symbols_at_gathers_only_the_requested_points(make_scatter):
    markers = np.random.default_rng(0).choice(["a", "b"], 100)
    fig, _, _ = make_scatter(n=100, markers=markers)

    wanted = np.array([3, 17, 42])
    got = fig.marker_symbols_at(wanted)
    assert np.array_equal(got, _oracle_marker_symbols(markers)[wanted])


def test_marker_symbols_at_without_markers(make_scatter):
    fig, _, _ = make_scatter(n=10)
    fig._marker_codes, fig._marker_names = None, ["circle"]
    assert list(fig.marker_symbols_at(np.array([0, 1]))) == ["circle", "circle"]

    fig._marker_names = None
    assert fig.marker_symbols_at(np.array([0, 1])) is None


def test_per_point_sizes_with_multiple_markers(make_scatter):
    """Used to raise in `points2gfx`: the full size array met a point subset."""
    markers = np.random.default_rng(0).choice(["a", "b"], 60)
    fig, _, _ = make_scatter(n=60, markers=markers)

    sizes = np.linspace(1.0, 2.0, 60).astype(np.float32)
    fig.point_size = sizes

    rebuilt = fig.make_points()
    assert len(rebuilt) == 2
    for vis in rebuilt:
        assert len(vis.geometry.sizes.data) == len(vis._point_ix)
        assert np.allclose(
            vis.geometry.sizes.data, sizes[vis._point_ix] * fig.point_scale
        )


# --------------------------------------------------------------------------- #
# Plotly symbol resolution
# --------------------------------------------------------------------------- #


def test_plotly_symbols_collapse_to_a_scalar_when_uniform(make_scatter):
    fig, _, _ = make_scatter(n=20)
    assert _resolve_marker_symbols(fig) == "circle"


def test_plotly_symbols_are_per_point_when_mixed(make_scatter):
    markers = np.random.default_rng(0).choice(["a", "b"], 40)
    fig, _, _ = make_scatter(n=40, markers=markers)

    symbols = _resolve_marker_symbols(fig)
    assert isinstance(symbols, np.ndarray)
    assert len(symbols) == 40
    # Same partition as the gfx markers, just renamed for Plotly.
    assert len(set(symbols.tolist())) == 2
