"""Tests for the spatial chunking of the scatter's points.

``bigclust2.chunking`` splits a point cloud into spatially coherent, roughly
equal-sized chunks so the figure can build one visual per chunk and later cull
or decimate per chunk. The properties the rest of the code relies on are:

* the order is a *permutation* - no point is lost or duplicated;
* chunks are contiguous slices of that order;
* chunks are near-equal in point count, whatever the density;
* each chunk's reported bounds really do contain its points.
"""

from __future__ import annotations

import numpy as np
import pytest

from bigclust2.chunking import (
    chunk_bounds,
    chunk_count,
    draw_counts,
    group_chunk_target,
    morton_keys,
    spatial_chunks,
    visible_chunks,
)


# --------------------------------------------------------------------------- #
# morton_keys
# --------------------------------------------------------------------------- #


def test_keys_interleave_coordinate_bits():
    """A 2x2 grid maps to the canonical Z-order 0, 1, 2, 3."""
    xy = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    keys = morton_keys(xy, bits=1)
    assert list(keys) == [0, 1, 2, 3]


def test_keys_preserve_locality():
    """Points close together get closer keys than far-apart ones.

    All points must be in one call: the quantisation grid is derived from the
    extent of the input, so keys are only comparable within a single call.
    """
    keys = morton_keys(
        np.array([[0.10, 0.10], [0.11, 0.11], [0.99, 0.99]]), bits=16
    ).astype(np.int64)
    assert abs(keys[0] - keys[1]) < abs(keys[0] - keys[2])


def test_non_finite_points_get_key_zero():
    """Non-finite rows must not poison the quantisation of the finite ones."""
    xy = np.array(
        [[np.nan, 0.0], [0.9, 0.9], [0.0, np.inf], [0.0, 0.0], [0.5, 0.5]]
    )
    keys = morton_keys(xy)
    assert keys[0] == 0 and keys[2] == 0, "non-finite rows sort together at 0"
    assert keys[3] == 0, "the finite minimum also lands at 0"
    assert keys[1] != 0 and keys[4] != 0, "other finite points are spread out"
    assert keys[4] < keys[1], "and still ordered by locality"


def test_all_non_finite_is_handled():
    assert list(morton_keys(np.full((3, 2), np.nan))) == [0, 0, 0]


def test_degenerate_axis_does_not_divide_by_zero():
    """All points on a vertical line: the x span is zero."""
    xy = np.column_stack([np.zeros(10), np.linspace(0, 1, 10)])
    keys = morton_keys(xy)
    assert np.isfinite(keys.astype(np.float64)).all()
    assert len(np.unique(keys)) == 10


def test_empty_input():
    assert morton_keys(np.zeros((0, 2))).shape == (0,)


@pytest.mark.parametrize("bits", [0, 17])
def test_bits_are_validated(bits):
    with pytest.raises(ValueError, match="bits"):
        morton_keys(np.zeros((4, 2)), bits=bits)


def test_shape_is_validated():
    with pytest.raises(ValueError, match=r"\(N, 2\)"):
        morton_keys(np.zeros((4, 3)))


# --------------------------------------------------------------------------- #
# chunk_count
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "n, target, min_size, expected",
    [
        (0, 1000, 100, 0),
        (50, 1000, 100, 1),  # below min_size -> unchunked
        (1000, 1000, 100, 10),
        (10**9, 1000, 100, 1000),  # capped at target
        (1000, 1, 100, 1),  # target of 1 disables chunking
    ],
)
def test_chunk_count(n, target, min_size, expected):
    assert chunk_count(n, target, min_size) == expected


# --------------------------------------------------------------------------- #
# spatial_chunks
# --------------------------------------------------------------------------- #


def test_order_is_a_permutation():
    xy = np.random.default_rng(0).random((5000, 2)).astype(np.float32)
    order, starts = spatial_chunks(xy, target=8, min_size=100)
    assert np.array_equal(np.sort(order), np.arange(len(xy)))
    assert starts[0] == 0 and starts[-1] == len(xy)
    assert (np.diff(starts) > 0).all()


def test_small_input_is_left_unchunked():
    """`None` lets the caller skip materializing an identity permutation."""
    xy = np.random.default_rng(0).random((500, 2)).astype(np.float32)
    order, starts = spatial_chunks(xy, target=100, min_size=100_000)
    assert order is None
    assert list(starts) == [0, 500]


def test_chunks_are_near_equal_in_size():
    xy = np.random.default_rng(0).random((10_000, 2)).astype(np.float32)
    _, starts = spatial_chunks(xy, target=7, min_size=100)
    sizes = np.diff(starts)
    assert sizes.max() - sizes.min() <= 1


def test_chunk_extent_shrinks_with_density():
    """Dense regions get small chunks, sparse regions large ones.

    This is the property that makes per-chunk culling worthwhile: a zoomed-in
    view touches few chunks regardless of how dense that area is.
    """
    rng = np.random.default_rng(0)
    dense = rng.normal(0.5, 0.005, (9000, 2))
    sparse = rng.uniform(0, 1, (1000, 2))
    xy = np.vstack([dense, sparse]).astype(np.float32)
    # Shuffle, so that only real spatial ordering can group the dense points
    # together - leaving the input grouped would let any ordering pass.
    xy = xy[rng.permutation(len(xy))]

    order, starts = spatial_chunks(xy, target=10, min_size=100)
    bounds = chunk_bounds(xy[order], starts)
    areas = (bounds[:, 2] - bounds[:, 0]) * (bounds[:, 3] - bounds[:, 1])

    assert np.ptp(np.diff(starts)) <= 1, "sizes should still be uniform"
    assert areas.max() > 100 * areas.min(), "extents should vary with density"


def test_identical_points_still_chunk_cleanly():
    xy = np.full((2000, 2), 3.0, dtype=np.float32)
    order, starts = spatial_chunks(xy, target=4, min_size=100)
    assert np.array_equal(np.sort(order), np.arange(2000))
    assert list(np.diff(starts)) == [500, 500, 500, 500]


def test_empty_input_chunks_to_nothing():
    order, starts = spatial_chunks(
        np.zeros((0, 2), dtype=np.float32), target=1000, min_size=100
    )
    assert order is None
    assert list(starts) == [0, 0]


def test_chunk_membership_follows_the_morton_order():
    """Which points share a chunk is the contract; their order within is not.

    Asserted as membership rather than the exact permutation so that a later
    step can shuffle within each chunk without touching this test.
    """
    xy = np.random.default_rng(0).random((20_000, 2)).astype(np.float32)
    order, starts = spatial_chunks(xy, target=4, min_size=100)
    expected = np.argsort(morton_keys(xy), kind="stable")

    for i in range(len(starts) - 1):
        sl = slice(starts[i], starts[i + 1])
        assert set(order[sl].tolist()) == set(expected[sl].tolist())


# --------------------------------------------------------------------------- #
# group_chunk_target
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "target, n_group, total, expected",
    [
        (1000, 1000, 1000, 1000),  # one group takes the whole budget
        (1000, 500, 1000, 500),  # half the points, half the budget
        (1000, 1, 1_000_000, 1),  # never below one chunk
        (1000, 0, 0, 1),  # degenerate totals do not divide by zero
    ],
)
def test_group_chunk_target(target, n_group, total, expected):
    assert group_chunk_target(target, n_group, total) == expected


def test_budget_is_shared_across_groups():
    """Splitting by group must not multiply the total chunk count."""
    target, total = 1000, 10_000
    groups = [5000, 3000, 2000]
    shares = [group_chunk_target(target, n, total) for n in groups]
    assert sum(shares) <= target + len(groups)


# --------------------------------------------------------------------------- #
# chunk_bounds
# --------------------------------------------------------------------------- #


def test_bounds_match_a_per_chunk_loop():
    xy = np.random.default_rng(1).random((3000, 2)).astype(np.float32)
    order, starts = spatial_chunks(xy, target=6, min_size=100)
    ordered = xy[order]

    expected = np.empty((len(starts) - 1, 4))
    for i in range(len(starts) - 1):
        pts = ordered[starts[i] : starts[i + 1]]
        expected[i, :2] = pts.min(axis=0)
        expected[i, 2:] = pts.max(axis=0)

    assert np.allclose(chunk_bounds(ordered, starts), expected)


def test_bounds_contain_their_points():
    xy = np.random.default_rng(0).random((5000, 2)).astype(np.float32)
    order, starts = spatial_chunks(xy, target=8, min_size=100)
    ordered = xy[order]
    bounds = chunk_bounds(ordered, starts)

    for i in range(len(starts) - 1):
        pts = ordered[starts[i] : starts[i + 1]]
        assert (pts[:, 0] >= bounds[i, 0] - 1e-6).all()
        assert (pts[:, 1] >= bounds[i, 1] - 1e-6).all()
        assert (pts[:, 0] <= bounds[i, 2] + 1e-6).all()
        assert (pts[:, 1] <= bounds[i, 3] + 1e-6).all()


def test_non_finite_points_are_ignored_in_bounds():
    xy = np.array([[0.0, 0.0], [np.nan, np.nan], [1.0, 1.0]])
    assert np.allclose(chunk_bounds(xy, np.array([0, 3]))[0], [0.0, 0.0, 1.0, 1.0])


def test_chunk_with_no_finite_point_gets_nan_bounds():
    """NaN bounds never intersect a view rectangle, so such a chunk is skipped."""
    assert np.isnan(chunk_bounds(np.full((4, 2), np.nan), np.array([0, 4]))).all()


def test_bounds_of_empty_input():
    assert chunk_bounds(np.zeros((0, 2)), np.array([0, 0])).shape == (1, 4)


def test_bounds_are_per_chunk_not_global():
    """Two well-separated chunks must not both report the whole extent."""
    xy = np.vstack(
        [np.zeros((100, 2)), np.full((100, 2), 5.0)]
    ).astype(np.float32)
    bounds = chunk_bounds(xy, np.array([0, 100, 200]))
    assert np.allclose(bounds[0], [0, 0, 0, 0])
    assert np.allclose(bounds[1], [5, 5, 5, 5])


# --------------------------------------------------------------------------- #
# visible_chunks
# --------------------------------------------------------------------------- #


@pytest.fixture
def bounds():
    # Four unit squares in a row along x, at x = 0..1, 2..3, 4..5, 6..7.
    return np.array(
        [
            [0.0, 0.0, 1.0, 1.0],
            [2.0, 0.0, 3.0, 1.0],
            [4.0, 0.0, 5.0, 1.0],
            [6.0, 0.0, 7.0, 1.0],
        ]
    )


def test_only_overlapping_chunks_are_visible(bounds):
    assert list(visible_chunks(bounds, (2.5, 0.0, 4.5, 1.0))) == [
        False,
        True,
        True,
        False,
    ]


def test_a_view_containing_everything_sees_everything(bounds):
    assert visible_chunks(bounds, (-10, -10, 10, 10)).all()


def test_a_view_outside_everything_sees_nothing(bounds):
    assert not visible_chunks(bounds, (100, 100, 200, 200)).any()


def test_touching_edges_count_as_visible(bounds):
    """A chunk exactly at the viewport edge must not pop out of existence."""
    mask = visible_chunks(bounds, (1.0, 0.0, 2.0, 1.0))
    assert mask[0] and mask[1]


def test_nan_bounds_are_never_visible(bounds):
    """A chunk with no finite point has NaN bounds."""
    bounds = np.vstack([bounds, np.full((1, 4), np.nan)])
    assert not visible_chunks(bounds, (-10, -10, 10, 10))[-1]


# --------------------------------------------------------------------------- #
# draw_counts
# --------------------------------------------------------------------------- #


def test_everything_is_drawn_when_it_fits_the_budget():
    sizes = np.array([100, 100, 100])
    counts = draw_counts(sizes, np.ones(3, bool), budget=1000)
    assert list(counts) == [100, 100, 100]


def test_invisible_chunks_draw_nothing():
    sizes = np.array([100, 100, 100])
    counts = draw_counts(sizes, np.array([True, False, True]), budget=1000)
    assert list(counts) == [100, 0, 100]


def test_the_budget_is_shared_in_proportion_to_size():
    sizes = np.array([600, 300, 100])
    counts = draw_counts(sizes, np.ones(3, bool), budget=100, min_per_chunk=1)
    assert counts.sum() <= 100
    # Same ordering as the sizes, roughly proportional.
    assert counts[0] > counts[1] > counts[2]


def test_no_chunk_is_asked_for_more_than_it_holds():
    sizes = np.array([10, 1000])
    counts = draw_counts(sizes, np.ones(2, bool), budget=500)
    assert (counts <= sizes).all()


def test_a_visible_chunk_keeps_a_floor():
    """Otherwise a chunk in view could vanish when many share the budget."""
    sizes = np.full(10, 1000)
    counts = draw_counts(sizes, np.ones(10, bool), budget=100, min_per_chunk=4)
    assert (counts >= 4).all()
    assert counts.sum() <= 100


def test_the_floor_never_overruns_the_budget():
    """With enough visible chunks the floor cannot be paid for in full.

    `min_per_chunk` exists to stop a chunk disappearing, not to override the
    budget - 1000 chunks at a floor of 32 would be 32x a budget of 1000.
    """
    sizes = np.full(1000, 1000)
    counts = draw_counts(sizes, np.ones(1000, bool), budget=1000, min_per_chunk=32)

    assert counts.sum() <= 1000
    assert (counts >= 1).all(), "every visible chunk should still draw something"


def test_nothing_visible_draws_nothing():
    counts = draw_counts(np.array([10, 20]), np.zeros(2, bool), budget=1000)
    assert list(counts) == [0, 0]


def test_narrowing_the_view_draws_the_survivors_more_densely():
    """The property that makes zooming in reveal detail."""
    sizes = np.full(10, 1000)
    budget = 2000

    wide = draw_counts(sizes, np.ones(10, bool), budget=budget)
    narrow = draw_counts(sizes, np.array([True] * 2 + [False] * 8), budget=budget)

    assert narrow[0] > wide[0]
    assert narrow.sum() <= budget


# --------------------------------------------------------------------------- #
# The within-chunk shuffle
# --------------------------------------------------------------------------- #


@pytest.fixture
def shuffle_xy():
    return np.random.default_rng(0).random((2000, 2)).astype(np.float32)


def test_shuffling_keeps_chunk_membership(shuffle_xy):
    xy = shuffle_xy
    plain, starts = spatial_chunks(xy, target=8, min_size=10)
    shuffled, starts2 = spatial_chunks(
        xy, target=8, min_size=10, rng=np.random.default_rng(0)
    )

    assert np.array_equal(starts, starts2)
    for i in range(len(starts) - 1):
        sl = slice(starts[i], starts[i + 1])
        assert set(shuffled[sl].tolist()) == set(plain[sl].tolist())


def test_shuffling_changes_the_order_within_a_chunk(shuffle_xy):
    xy = shuffle_xy
    plain, _ = spatial_chunks(xy, target=8, min_size=10)
    shuffled, _ = spatial_chunks(
        xy, target=8, min_size=10, rng=np.random.default_rng(0)
    )
    assert not np.array_equal(plain, shuffled)


def test_shuffling_is_still_a_permutation(shuffle_xy):
    xy = shuffle_xy
    order, _ = spatial_chunks(xy, target=8, min_size=10, rng=np.random.default_rng(1))
    assert np.array_equal(np.sort(order), np.arange(2000))


def test_an_unchunked_cloud_still_shuffles():
    """With one chunk there is no ordering to preserve, but still one to sample."""
    xy = np.random.default_rng(0).random((500, 2)).astype(np.float32)
    order, starts = spatial_chunks(
        xy, target=8, min_size=10_000, rng=np.random.default_rng(0)
    )
    assert list(starts) == [0, 500]
    assert order is not None
    assert np.array_equal(np.sort(order), np.arange(500))
    assert not np.array_equal(order, np.arange(500))


def test_a_drawn_prefix_is_an_unbiased_sample():
    """The property the whole budget scheme rests on.

    A prefix of a shuffled chunk must look like the chunk as a whole - here,
    half the points carry a marker value and the prefix should contain roughly
    half of them, not a spatially-sorted run of one or the other.
    """
    rng = np.random.default_rng(0)
    xy = rng.random((20_000, 2)).astype(np.float32)
    flagged = xy[:, 0] > 0.5  # a spatially-correlated property

    order, starts = spatial_chunks(xy, target=10, min_size=10, rng=rng)

    for i in range(len(starts) - 1):
        chunk = order[starts[i] : starts[i + 1]]
        prefix = chunk[: max(len(chunk) // 10, 1)]
        whole_rate = flagged[chunk].mean()
        prefix_rate = flagged[prefix].mean()
        assert abs(prefix_rate - whole_rate) < 0.15, (
            f"chunk {i}: prefix {prefix_rate:.2f} vs whole {whole_rate:.2f}"
        )
