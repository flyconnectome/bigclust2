"""Tests for greedy label placement (headless: no GUI / wgpu canvas).

The pure solver in ``bigclust2.label_placement`` is exercised directly. The
figure-side caching/invalidation logic is tested by binding the real
``ScatterFigure`` methods onto a lightweight stub (importing the figure module
is headless-safe; instantiating its wgpu window is not).
"""

from types import SimpleNamespace

import numpy as np
import pygfx as gfx
import pytest

from bigclust2.figure import BaseFigure
from bigclust2.label_placement import (
    N_SLOTS,
    SLOT_LEFT,
    SLOT_RIGHT,
    SLOT_TOP_LEFT,
    SLOT_TOP_RIGHT,
    connector_offsets,
    estimate_text_wh,
    measure_text_wh,
    slot_box,
    solve_label_placement,
    spatial_components,
)
from bigclust2.scatter import (
    LABEL_CONNECTOR_STANDOFF,
    LABEL_CONNECTOR_Z,
    LABEL_Z,
    LABEL_Z_BEHIND,
    ScatterFigure,
)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _boxes_from_solution(points, extents, slots, offsets):
    """Reconstruct label boxes from the anchor offsets (middle-left anchor)."""
    boxes = []
    for i in range(len(points)):
        if slots[i] < 0:
            boxes.append(None)
            continue
        w, h = extents[i]
        x0 = points[i, 0] + offsets[i, 0]
        yc = points[i, 1] + offsets[i, 1]
        boxes.append((x0, yc - h / 2, x0 + w, yc + h / 2))
    return boxes


def _overlaps(a, b):
    return a[0] < b[2] and a[2] > b[0] and a[1] < b[3] and a[3] > b[1]


def _assert_valid(points, radii, extents, slots, offsets):
    """No placed label may overlap another label or another point's marker."""
    boxes = _boxes_from_solution(points, extents, slots, offsets)
    radii = np.broadcast_to(np.asarray(radii, dtype=float), (len(points),))
    for i, a in enumerate(boxes):
        if a is None:
            continue
        for j in range(i + 1, len(boxes)):
            if boxes[j] is None:
                continue
            assert not _overlaps(a, boxes[j]), f"labels {i} and {j} overlap"
        for j in range(len(points)):
            if j == i:
                continue
            r = radii[j]
            marker = (
                points[j, 0] - r,
                points[j, 1] - r,
                points[j, 0] + r,
                points[j, 1] + r,
            )
            assert not _overlaps(a, marker), f"label {i} overlaps marker {j}"


# --------------------------------------------------------------------------- #
# Solver
# --------------------------------------------------------------------------- #


def test_isolated_points_use_preferred_slot():
    points = np.array([[0.0, 0.0], [10.0, 10.0]])
    extents = np.tile((1.0, 0.25), (2, 1))

    slots, offsets = solve_label_placement(points, 0.1, extents)

    assert (slots == SLOT_RIGHT).all()
    assert (offsets[:, 0] > 0).all()  # to the right of the point
    assert np.allclose(offsets[:, 1], 0)  # vertically centered
    _assert_valid(points, 0.1, extents, slots, offsets)


def test_coincident_points_get_distinct_slots():
    # Four wide labels on the same spot: diagonal slots collide with the
    # other points' markers here, so we expect right, left, top and bottom.
    points = np.zeros((4, 2))
    extents = np.tile((1.0, 0.2), (4, 1))

    slots, offsets = solve_label_placement(points, 0.2, extents)

    assert (slots >= 0).all()
    assert len(set(slots)) == 4
    _assert_valid(points, 0.2, extents, slots, offsets)


def test_overflow_gets_suppressed():
    # 20 points on the same spot cannot all fit a label
    points = np.zeros((20, 2))
    extents = np.tile((1.0, 0.2), (20, 1))

    slots, offsets = solve_label_placement(points, 0.2, extents)

    assert 0 < (slots >= 0).sum() <= N_SLOTS
    assert (slots == -1).sum() >= 20 - N_SLOTS

    # Unplaced labels still carry the preferred (right-hand) slot's offset
    # so callers can draw them anyway
    pad = 0.2 * 0.2  # default pad: 20% of the median label height
    for i in np.where(slots == -1)[0]:
        assert offsets[i] == pytest.approx((0.2 + pad, 0.0))


def test_prev_slot_hysteresis():
    points = np.array([[0.0, 0.0]])
    extents = np.array([[1.0, 0.25]])

    # Even though the (preferred) right slot is free, the label sticks to
    # the slot it occupied before
    slots, _ = solve_label_placement(
        points, 0.1, extents, prev_slots=np.array([SLOT_LEFT])
    )

    assert slots[0] == SLOT_LEFT


def test_hysteresis_keeps_solution_stable():
    rng = np.random.default_rng(0)
    points = rng.uniform(0, 3, size=(80, 2))
    extents = np.column_stack([rng.uniform(0.1, 0.6, 80), np.full(80, 0.12)])

    slots1, _ = solve_label_placement(points, 0.02, extents)
    slots2, _ = solve_label_placement(points, 0.02, extents, prev_slots=slots1)

    # Re-solving with the previous solution as hysteresis must not shuffle
    assert (slots1 == slots2).all()


def test_priority_wins_preferred_slot():
    points = np.zeros((2, 2))
    extents = np.tile((1.0, 0.2), (2, 1))

    slots, _ = solve_label_placement(
        points, 0.2, extents, priority=np.array([1.0, 0.0])
    )

    assert slots[1] == SLOT_RIGHT
    assert slots[0] != SLOT_RIGHT


def test_random_configuration_is_overlap_free():
    rng = np.random.default_rng(42)
    points = rng.uniform(0, 3, size=(150, 2))
    extents = np.column_stack([rng.uniform(0.1, 0.8, 150), np.full(150, 0.12)])
    radii = np.full(150, 0.02)

    slots, offsets = solve_label_placement(points, radii, extents)

    assert np.isfinite(offsets).all()
    assert (slots >= 0).sum() > 0
    _assert_valid(points, radii, extents, slots, offsets)


# --------------------------------------------------------------------------- #
# Text extents
# --------------------------------------------------------------------------- #


def test_connector_offsets_geometry():
    w, h, r, pad = 1.0, 0.25, 0.1, 0.05
    for slot in range(N_SLOTS):
        box = slot_box(slot, 0.0, 0.0, r, w, h, pad)
        anchor = np.array([box[0], (box[1] + box[3]) / 2])
        start, end_rel = connector_offsets(slot, w, h, r)

        # The line starts on the marker edge (0.7071 approximates sqrt(2)/2,
        # hence the loose tolerance) ...
        assert np.hypot(*start) == pytest.approx(r, rel=1e-3)

        # ... or, with a standoff, just outside of it
        start_gap, end_gap = connector_offsets(slot, w, h, r, gap=0.03)
        assert np.hypot(*start_gap) == pytest.approx(r + 0.03, rel=1e-3)
        assert end_gap == end_rel
        # ... pointing towards the label
        attach = anchor + end_rel
        assert np.dot(start, attach) > 0

        # And it ends on the boundary of the label box
        x0, y0, x1, y1 = box
        assert x0 - 1e-9 <= attach[0] <= x1 + 1e-9
        assert y0 - 1e-9 <= attach[1] <= y1 + 1e-9
        assert (
            np.isclose(attach[0], x0)
            or np.isclose(attach[0], x1)
            or np.isclose(attach[1], y0)
            or np.isclose(attach[1], y1)
        )

    # Ring slots keep their base direction ...
    assert connector_offsets(N_SLOTS + 2, w, h, r) == connector_offsets(2, w, h, r)
    # ... but negative slots are invalid
    with pytest.raises(ValueError):
        connector_offsets(-1, w, h, r)


def test_ring_slots_extend_reach():
    w, h, r, pad = 0.3, 0.2, 0.2, 0.04

    # Ring-1 boxes have the same base geometry, just farther out
    for base in range(N_SLOTS):
        b0 = np.array(slot_box(base, 0.0, 0.0, r, w, h, pad))
        b1 = np.array(slot_box(N_SLOTS + base, 0.0, 0.0, r, w, h, pad))
        assert b1[2] - b1[0] == pytest.approx(b0[2] - b0[0])  # same width
        assert b1[3] - b1[1] == pytest.approx(b0[3] - b0[1])  # same height
        c0 = np.hypot((b0[0] + b0[2]) / 2, (b0[1] + b0[3]) / 2)
        c1 = np.hypot((b1[0] + b1[2]) / 2, (b1[1] + b1[3]) / 2)
        assert c1 > c0  # farther from the anchor

    # Eight coincident narrow labels: with one ring only the four
    # non-diagonal slots work (diagonals hit the other markers), a second
    # ring gives everyone a spot
    points = np.zeros((8, 2))
    extents = np.tile((w, h), (8, 1))
    slots1, _ = solve_label_placement(points, r, extents, pad=pad)
    assert (slots1 >= 0).sum() == 4
    slots2, offsets2 = solve_label_placement(points, r, extents, pad=pad, rings=2)
    assert (slots2 >= 0).all()
    _assert_valid(points, r, extents, slots2, offsets2)


def test_bounds_rejects_out_of_view_slots():
    points = np.array([[0.3, 0.0]])
    extents = np.array([[1.0, 0.25]])

    # The right-side slots would stick out beyond x=0.5, so the label goes
    # to the first candidate that fits fully inside the bounds
    slots, offsets = solve_label_placement(
        points, 0.1, extents, bounds=(-2.0, -2.0, 0.5, 2.0)
    )
    assert slots[0] == SLOT_TOP_LEFT
    x0 = points[0, 0] + offsets[0, 0]
    assert x0 >= -2.0 and x0 + extents[0, 0] <= 0.5

    # Bounds too tight for any slot -> unplaced
    slots, _ = solve_label_placement(
        points, 0.1, extents, bounds=(0.0, -0.05, 0.6, 0.05)
    )
    assert slots[0] == -1


def test_exact_obstacles_instead_of_anchor_discs():
    # A "group" anchor with a big disc: with anchor_obstacles=True the disc
    # blocks a second anchor entirely ...
    anchors = np.array([[0.0, 0.0], [0.05, 0.0]])
    radii = np.array([10.0, 0.01])
    extents = np.tile((0.2, 0.1), (2, 1))
    slots, _ = solve_label_placement(anchors, radii, extents)
    assert slots[1] == -1

    # ... with anchor_obstacles=False both labels place fine
    slots, _ = solve_label_placement(anchors, radii, extents, anchor_obstacles=False)
    assert (slots >= 0).all()

    # Explicit obstacles do block, and are identified in the debug output
    debug = []
    slots, _ = solve_label_placement(
        np.array([[0.0, 0.0]]),
        0.1,
        np.array([[1.0, 0.25]]),
        obstacles=np.array([[0.6, 0.0]]),
        obstacle_radii=0.05,
        anchor_obstacles=False,
        debug=debug,
    )
    assert slots[0] == SLOT_TOP_RIGHT
    assert debug[0][0] == (SLOT_RIGHT, "collision", ("obstacle", 0))


def test_solver_debug_records_rejections():
    # Overflow case: coincident points where not everything fits
    points = np.zeros((20, 2))
    extents = np.tile((1.0, 0.2), (20, 1))
    debug = []
    slots, _ = solve_label_placement(points, 0.2, extents, debug=debug)

    assert len(debug) == 20
    for i in np.where(slots < 0)[0]:
        # Every candidate slot was rejected, each with a concrete blocker
        assert len(debug[i]) == N_SLOTS
        assert {kind for _, kind, _ in debug[i]} == {"collision"}
        for _, _, blocker in debug[i]:
            # Marker (anchor index) or an already-placed label
            assert isinstance(blocker, tuple) or int(blocker) >= 0

    # Bounds rejections are reported as out-of-view
    debug = []
    slots, _ = solve_label_placement(
        np.array([[0.3, 0.0]]),
        0.1,
        np.array([[1.0, 0.25]]),
        bounds=(0.0, -0.05, 0.6, 0.05),
        debug=debug,
    )
    assert slots[0] == -1
    assert {kind for _, kind, _ in debug[0]} == {"out-of-view"}


def test_estimate_text_wh_scales_with_text():
    w1, h1 = estimate_text_wh("abc", 1.0)
    w2, h2 = estimate_text_wh("abcdef", 1.0)
    assert w2 > w1 and h2 == h1

    w3, h3 = estimate_text_wh("abc\nabc", 1.0)
    assert h3 > h1 and w3 == w1


class _FakeText:
    """Duck-typed stand-in for a laid-out pygfx Text visual."""

    def __init__(self, blocks, positions):
        self._text_blocks = [
            SimpleNamespace(_rect=SimpleNamespace(left=l, right=r, top=t, bottom=b))
            for (l, r, t, b) in blocks
        ]
        self.geometry = SimpleNamespace(
            positions=SimpleNamespace(data=np.asarray(positions, dtype=np.float32))
        )


def test_measure_text_wh():
    # Single line: 12 units wide, 1.25 units tall
    fake = _FakeText([(0.0, 12.0, 0.35, -0.9)], [(0.0, 0.0, 0.0)])
    assert measure_text_wh(fake, 1.0) == pytest.approx((12.0, 1.25))

    # Two lines stacked below each other
    fake = _FakeText(
        [(0.0, 5.0, 0.3, -0.95), (0.0, 5.0, 0.3, -0.95)],
        [(0.0, 0.0, 0.0), (0.0, -1.25, 0.0)],
    )
    assert measure_text_wh(fake, 1.0) == pytest.approx((5.0, 2.5))


def test_measure_text_wh_falls_back_on_garbage():
    # Not a text visual at all
    assert measure_text_wh(object(), 1.0) is None
    # No blocks yet
    assert measure_text_wh(_FakeText([], np.zeros((0, 3))), 1.0) is None
    # Implausible extent for the font size (e.g. layout units changed)
    fake = _FakeText([(0.0, 12.0, 50.0, -50.0)], [(0.0, 0.0, 0.0)])
    assert measure_text_wh(fake, 1.0) is None


# --------------------------------------------------------------------------- #
# Figure-side caching / invalidation
# --------------------------------------------------------------------------- #


class _StubVisual:
    def __init__(self):
        self.visible = True
        self.local = SimpleNamespace(position=np.zeros(3, dtype=np.float32))
        self.material = SimpleNamespace(opacity=1.0, color="w")


class _StubFigure:
    """Binds the real placement methods onto a minimal stand-in figure."""

    _update_label_placement = ScatterFigure._update_label_placement
    _invalidate_label_placement = ScatterFigure._invalidate_label_placement
    _reset_label_placement_state = ScatterFigure._reset_label_placement_state
    _label_extent = ScatterFigure._label_extent
    _animate_label_placement = ScatterFigure._animate_label_placement
    _rebuild_label_connector_visual = ScatterFigure._rebuild_label_connector_visual
    _sync_label_connector_positions = ScatterFigure._sync_label_connector_positions
    _update_group_labels = ScatterFigure._update_group_labels
    _ensure_group_label_grp = ScatterFigure._ensure_group_label_grp
    _make_group_label_visual = ScatterFigure._make_group_label_visual
    _rebuild_group_connectors = ScatterFigure._rebuild_group_connectors
    _sync_group_label_positions = ScatterFigure._sync_group_label_positions
    _group_label_extent = ScatterFigure._group_label_extent
    _label_codes = ScatterFigure._label_codes
    view_bounds = BaseFigure.view_bounds
    is_visible_pos = BaseFigure.is_visible_pos
    label_debug_report = ScatterFigure.label_debug_report
    _format_group_label_debug = ScatterFigure._format_group_label_debug
    _format_individual_label_debug = ScatterFigure._format_individual_label_debug
    _describe_blocker = staticmethod(ScatterFigure._describe_blocker)
    hide_labels = ScatterFigure.hide_labels
    highlight_labels = ScatterFigure.highlight_labels
    smart_label_placement = ScatterFigure.smart_label_placement
    declutter_mode = ScatterFigure.declutter_mode
    unplaced_labels = ScatterFigure.unplaced_labels
    label_connectors = ScatterFigure.label_connectors

    def __init__(self, positions, labels):
        self.positions = np.asarray(positions, dtype=np.float32)
        self.labels = np.asarray(labels)
        self.label_visuals = [_StubVisual() for _ in labels]
        self.label_group = gfx.Group()
        self.canvas = SimpleNamespace(request_draw=lambda: None)
        self._smart_label_placement = True
        self._declutter_mode = "individual"
        self._unplaced_labels = "hide"
        self._label_connectors = True
        self._label_colors = None
        self._font_size = 0.01
        self._point_size = 10
        self._point_scale = 0.001
        self.label_vis_limit = 400
        self._selected = None
        self._render_stale = False
        self._label_layout_version = 0
        self._reset_label_placement_state()

    def __len__(self):
        return len(self.positions)


def test_figure_places_and_caches():
    # Two near-coincident points plus one far away
    fig = _StubFigure(
        [[0, 0], [0.001, 0], [100, 100]], ["AAAA", "BBBB", "CCCC"]
    )
    visible = np.arange(3)

    fig._update_label_placement(visible)

    # The isolated point keeps the preferred right-hand slot ...
    assert fig._label_slots[2] == SLOT_RIGHT
    # ... while the two overlapping ones end up in different slots
    assert fig._label_slots[0] != fig._label_slots[1]
    assert all(hasattr(v, "_label_target") for v in fig.label_visuals)

    # A second pass with an unchanged view is a no-op (solution is cached)
    key = fig._label_placement_key
    fig._update_label_placement(visible)
    assert fig._label_placement_key == key


def test_figure_suppresses_when_crowded():
    # 12 points on the same spot -> with the default geometry only the
    # right/left/top/bottom slots are free, the rest gets suppressed
    fig = _StubFigure(np.zeros((12, 2)), [f"label_{i}" for i in range(12)])
    visible = np.arange(12)

    fig._update_label_placement(visible)

    assert len(fig._label_suppressed) > 0
    assert all(not fig.label_visuals[ix].visible for ix in fig._label_suppressed)

    # The cached pass re-asserts suppression (labels get re-shown by the
    # visibility tick in the real figure)
    for ix in fig._label_suppressed:
        fig.label_visuals[ix].visible = True
    fig._update_label_placement(visible)
    assert all(not fig.label_visuals[ix].visible for ix in fig._label_suppressed)


def test_figure_reacts_to_text_change():
    fig = _StubFigure([[0, 0], [100, 100]], ["short", "other"])
    visible = np.arange(2)

    fig._update_label_placement(visible)
    key = fig._label_placement_key
    w_before, _ = fig._label_extent(0)

    # Changing the label text (e.g. user picks another meta column) must
    # invalidate both the cached extents and the cached solution
    fig.labels = np.asarray(["something considerably longer", "other"])
    fig._invalidate_label_placement()
    fig._update_label_placement(visible)

    assert fig._label_placement_key != key
    w_after, _ = fig._label_extent(0)
    assert w_after > w_before


def test_figure_unplaced_policies():
    fig = _StubFigure(np.zeros((12, 2)), [f"label_{i}" for i in range(12)])
    visible = np.arange(12)

    # "dim": unplaceable labels stay visible but faded, at the default
    # anchor - and on a z-layer behind the points
    fig.unplaced_labels = "dim"
    fig._update_label_placement(visible)
    assert len(fig._label_suppressed) > 0
    for ix in fig._label_suppressed:
        vis = fig.label_visuals[ix]
        assert vis.visible
        assert vis.material.opacity < 1.0
        assert vis._label_target[2] == LABEL_Z_BEHIND
    for ix in set(range(12)) - fig._label_suppressed:
        assert fig.label_visuals[ix].material.opacity == 1.0
        assert fig.label_visuals[ix]._label_target[2] == LABEL_Z

    # "show": fully opaque and above the points
    fig.unplaced_labels = "show"
    fig._update_label_placement(visible)
    for ix in fig._label_suppressed:
        assert fig.label_visuals[ix].visible
        assert fig.label_visuals[ix].material.opacity == 1.0
        assert fig.label_visuals[ix]._label_target[2] == LABEL_Z

    # back to "hide": hidden again
    fig.unplaced_labels = "hide"
    fig._update_label_placement(visible)
    for ix in fig._label_suppressed:
        assert not fig.label_visuals[ix].visible

    with pytest.raises(ValueError):
        fig.unplaced_labels = "nonsense"


def test_figure_builds_connectors():
    fig = _StubFigure([[0, 0], [0.001, 0], [100, 100]], ["AAAA", "BBBB", "CCCC"])
    visible = np.arange(3)

    fig._update_label_placement(visible)

    vis = fig._label_connector_vis
    assert vis is not None and vis.visible
    n = len(fig._label_connector_ix)
    assert n == 3  # all three labels were placed
    assert vis.geometry.positions.data.shape == (2 * n, 3)
    # Connectors render on their own layer just below the points
    assert (vis.geometry.positions.data[:, 2] == LABEL_CONNECTOR_Z).all()

    # Let the labels finish easing to their targets; the connectors must then
    # run from the marker edge to the label box edge
    for _ in range(100):
        fig._animate_label_placement()
    assert not fig._label_anim_active

    radius = fig._point_size * fig._point_scale / 2
    # The line starts just outside the marker edge: radius plus the standoff
    # (a fraction of the marker-to-label padding, which with connectors on is
    # half the label height)
    pad = 0.5 * 1.25 * fig._font_size
    start_dist = radius + pad * LABEL_CONNECTOR_STANDOFF
    data = vis.geometry.positions.data
    for row, gix in enumerate(fig._label_connector_ix):
        start = data[2 * row, :2]
        end = data[2 * row + 1, :2]
        assert np.hypot(*(start - fig.positions[gix, :2])) == pytest.approx(
            start_dist, rel=1e-3
        )
        expected = (
            np.asarray(fig.label_visuals[gix].local.position)[:2]
            + fig._label_connector_end_rel[row]
        )
        assert end == pytest.approx(expected, rel=1e-3, abs=1e-6)

    # Toggling connectors off hides the visual and clears the segments
    fig.label_connectors = False
    fig._update_label_placement(visible)
    assert not vis.visible
    assert len(fig._label_connector_ix) == 0

    # Re-enabling swaps a fresh geometry into the existing visual
    fig.label_connectors = True
    fig._update_label_placement(visible)
    assert vis.visible
    assert vis.geometry.positions.data.shape == (2 * 3, 3)


def test_figure_grouped_labels():
    positions = [
        [0, 0], [0.004, 0], [0, 0.004],  # group A
        [1, 1], [1.004, 1],  # group B
        [5, 5],  # group C (singleton)
    ]
    labels = ["A", "A", "A", "B", "B", "C"]
    fig = _StubFigure(positions, labels)
    fig.declutter_mode = "grouped"
    visible = np.arange(6)

    fig._update_group_labels(visible)

    # One visual per unique value, all shown
    assert set(fig._group_label_visuals) == {("A", 0), ("B", 0), ("C", 0)}
    assert all(v.visible for v in fig._group_label_visuals.values())
    data = fig._group_data
    assert data is not None and data["shown"].all()

    # One connector segment per member point, rendered behind the points
    vis = fig._group_connector_vis
    assert vis is not None and vis.visible
    assert vis.geometry.positions.data.shape == (2 * 6, 3)
    assert (vis.geometry.positions.data[:, 2] == LABEL_CONNECTOR_Z).all()

    # Every segment starts at its group label's attach point
    g = data["group_ids"]
    expected_attach = data["label_pos"][g][:, :2] + data["attach_rel"][g]
    assert vis.geometry.positions.data[0::2, :2] == pytest.approx(
        expected_attach, abs=1e-6
    )

    # Caching: an unchanged view is a no-op
    key = fig._group_placement_key
    fig._update_group_labels(visible)
    assert fig._group_placement_key == key

    # Moving all points drags labels and connectors along
    old_pos = data["label_pos"].copy()
    fig.positions = fig.positions + np.array([10, 0], dtype=np.float32)
    fig._sync_group_label_positions()
    assert fig._group_data["label_pos"][:, 0] == pytest.approx(
        old_pos[:, 0] + 10, abs=1e-4
    )


def test_figure_grouped_diffuse_group_does_not_block():
    # Regression: a diffuse "background" group whose bounding disc covers an
    # embedded tight group must not block the tight group's label - only its
    # actual points count as obstacles
    angles = np.linspace(0, 2 * np.pi, 24, endpoint=False)
    ring = 0.25 * np.column_stack([np.cos(angles), np.sin(angles)])
    tight = np.array([[0.0, 0.0], [0.003, 0.0], [0.0, 0.003]])
    positions = np.vstack([ring, tight])
    labels = np.array(["bg"] * 24 + ["fg"] * 3)
    fig = _StubFigure(positions, labels)
    fig.declutter_mode = "grouped"

    fig._update_group_labels(np.arange(len(positions)))

    assert fig._group_data["shown"].all()
    assert fig._group_label_visuals[("fg", 0)].visible
    assert fig._group_label_visuals[("bg", 0)].visible


def test_spatial_components():
    # Two well-separated pairs
    pts = np.array([[0.0, 0.0], [0.1, 0.0], [5.0, 5.0], [5.1, 5.0]])
    comp = spatial_components(pts, 0.5)
    assert comp[0] == comp[1] and comp[2] == comp[3]
    assert comp[0] != comp[2]

    # Chain connectivity: linked through intermediate points
    chain = np.column_stack([np.arange(5) * 0.4, np.zeros(5)])
    assert (spatial_components(chain, 0.5) == 0).all()
    assert len(np.unique(spatial_components(chain, 0.3))) == 5

    assert len(spatial_components(np.zeros((0, 2)), 1.0)) == 0


def test_figure_grouped_splits_disjoint_islands():
    # Two far-apart islands of 'untyped' each get their own label; the
    # contiguous 'typed' cluster stays a single group
    island = np.array(
        [[0, 0], [0.004, 0], [0, 0.004], [0.004, 0.004], [0.002, 0.002]]
    )
    positions = np.vstack([island, island + [5, 5], island[:3] + [0, 5]])
    labels = np.array(["untyped"] * 10 + ["typed"] * 3)
    fig = _StubFigure(positions, labels)
    fig.declutter_mode = "grouped"

    dbg = {}
    fig._update_group_labels(np.arange(len(positions)), debug=dbg)

    assert sorted(dbg["keys"]) == [("typed", 0), ("untyped", 0), ("untyped", 1)]
    assert fig._group_data["shown"].all()

    # Each island's label sits near its own points, not midway between them
    for gi in range(len(dbg["keys"])):
        label_xy = fig._group_data["label_pos"][gi, :2]
        assert np.linalg.norm(label_xy - dbg["centroids"][gi]) < 1.0

    # ... and no connector line spans between the islands
    seg = fig._group_connector_vis.geometry.positions.data
    lengths = np.linalg.norm(seg[0::2, :2] - seg[1::2, :2], axis=1)
    assert lengths.max() < 1.0

    # The debug report tells the islands apart
    report = "\n".join(fig._format_group_label_debug(dbg))
    assert "(island 1)" in report and "(island 2)" in report


def test_figure_grouped_unique_labels_before_extra_islands():
    # A's far singleton becomes an extra island that coincides with the
    # unique value B: the unique label must win the preferred slot, the
    # extra island takes what's left
    blob = np.array(
        [[0, 0], [0.002, 0], [0, 0.002], [0.002, 0.002], [0.001, 0.001]]
    )
    positions = np.vstack([blob + [-5, 0], [[0, 0]], [[0, 0]]])
    labels = np.array(["A"] * 5 + ["A", "B"])
    fig = _StubFigure(positions, labels)
    fig.declutter_mode = "grouped"

    dbg = {}
    fig._update_group_labels(np.arange(len(positions)), debug=dbg)

    keys = dbg["keys"]
    assert ("A", 1) in keys  # the far singleton became its own island
    a_extra = keys.index(("A", 1))
    b = keys.index(("B", 0))

    # The big island is A's primary, the singleton is not; B is primary
    assert dbg["primary"][keys.index(("A", 0))]
    assert not dbg["primary"][a_extra]
    assert dbg["primary"][b]

    # The unique label gets the preferred slot, the coincident extra island
    # comes second in line (but still finds a spot here)
    assert dbg["slots"][b] == SLOT_RIGHT
    assert dbg["slots"][a_extra] != SLOT_RIGHT
    assert dbg["slots"][a_extra] >= 0


def test_figure_grouped_oversized_group_uses_centroid_fallback():
    # A group larger than the viewport can't place a label on its rim (all
    # slots out of view) - it falls back to a spot near its centroid
    angles = np.linspace(0, 2 * np.pi, 24, endpoint=False)
    ring = 0.25 * np.column_stack([np.cos(angles), np.sin(angles)])
    tight = np.array([[0.0, 0.0], [0.003, 0.0], [0.0, 0.003]])
    positions = np.vstack([ring, tight])
    labels = np.array(["bg"] * 24 + ["fg"] * 3)
    fig = _StubFigure(positions, labels)
    fig.declutter_mode = "grouped"
    # View much smaller than the ring: every rim slot is out of view
    fig.view_bounds = lambda: (-0.15, -0.15, 0.15, 0.15)

    dbg = {}
    fig._update_group_labels(np.arange(len(positions)), debug=dbg)

    assert (dbg["slots"] >= 0).all()
    assert dbg["fallback"][dbg["texts"].index("bg")]
    assert fig._group_data["shown"].all()
    # The fallback label must sit inside the view bounds
    bg_pos = fig._group_data["label_pos"][dbg["texts"].index("bg")]
    assert -0.15 <= bg_pos[0] <= 0.15 and -0.15 <= bg_pos[1] <= 0.15


def _rgba(color):
    return pytest.approx(gfx.Color(color).rgba, abs=1e-3)


def test_figure_grouped_highlight_labels():
    # Double-clicking a label highlights the value: in grouped mode this
    # must recolor the *group* label - not create/show per-point labels
    positions = [[0, 0], [0.004, 0], [0, 0.004], [1, 1], [1.004, 1]]
    labels = ["A", "A", "A", "B", "B"]
    fig = _StubFigure(positions, labels)
    fig.declutter_mode = "grouped"
    visible = np.arange(5)

    fig._update_group_labels(visible)
    for v in fig.label_visuals:
        v.visible = False  # grouped mode keeps the per-point labels hidden

    fig.highlight_labels("A")

    # No per-point label was shown or recolored ...
    assert not any(v.visible for v in fig.label_visuals)
    assert not any(hasattr(v.material, "_original_color") for v in fig.label_visuals)

    # ... the highlight invalidated the cached solution ...
    key = fig._group_placement_key
    fig._update_group_labels(visible)
    assert fig._group_placement_key != key

    # ... and the re-solve recolored the group label (and only that one)
    assert fig._group_label_visuals[("A", 0)].material.color.rgba == _rgba("y")
    assert fig._group_label_visuals[("B", 0)].material.color.rgba == _rgba("w")

    # Index-based highlighting (what double-click passes) works the same
    fig.highlight_labels([3])
    fig._update_group_labels(visible)
    assert fig._group_label_visuals[("A", 0)].material.color.rgba == _rgba("w")
    assert fig._group_label_visuals[("B", 0)].material.color.rgba == _rgba("y")

    # Clearing the highlights restores the normal colors
    fig.highlight_labels(None)
    fig._update_group_labels(visible)
    assert fig._group_label_visuals[("A", 0)].material.color.rgba == _rgba("w")
    assert fig._group_label_visuals[("B", 0)].material.color.rgba == _rgba("w")


def test_figure_grouped_highlight_priority():
    # Two coincident groups: normally the bigger one wins the preferred
    # right-hand slot, but a highlighted value takes precedence
    positions = [[0, 0], [0.001, 0], [0, 0]]
    labels = ["AAAA", "AAAA", "BBBB"]

    fig = _StubFigure(positions, labels)
    fig.declutter_mode = "grouped"
    fig._update_group_labels(np.arange(3))
    assert fig._group_label_slots[("AAAA", 0)] == SLOT_RIGHT

    fig2 = _StubFigure(positions, labels)
    fig2.declutter_mode = "grouped"
    fig2.highlight_labels("BBBB")
    fig2._update_group_labels(np.arange(3))
    assert fig2._group_label_slots[("BBBB", 0)] == SLOT_RIGHT


def test_figure_grouped_highlight_survives_mode_switch():
    # Highlights set while in grouped mode carry over to the per-point
    # labels when switching back to individual mode
    fig = _StubFigure([[0, 0], [1, 1]], ["A", "B"])
    fig.declutter_mode = "grouped"
    fig.highlight_labels("A")

    fig.declutter_mode = "individual"
    assert fig.label_visuals[0].material.color == "y"
    assert fig.label_visuals[1].material.color == "w"
    assert fig._label_highlighted == {0}


def test_figure_grouped_mode_switch():
    fig = _StubFigure([[0, 0], [0.001, 0], [100, 100]], ["AAAA", "BBBB", "CCCC"])
    visible = np.arange(3)

    # Individual mode first: per-point labels placed, connectors visible
    fig._update_label_placement(visible)
    assert fig._label_connector_vis.visible

    # Switching to grouped hides the per-point machinery
    fig.declutter_mode = "grouped"
    assert all(not v.visible for v in fig.label_visuals)
    assert not fig._label_connector_vis.visible

    fig._update_group_labels(visible)
    assert fig._group_label_grp.visible
    assert set(fig._group_label_visuals) == {
        ("AAAA", 0),
        ("BBBB", 0),
        ("CCCC", 0),
    }

    # ... and switching back hides the group visuals again
    fig.declutter_mode = "individual"
    assert not fig._group_label_grp.visible

    with pytest.raises(ValueError):
        fig.declutter_mode = "nonsense"


def test_label_debug_report():
    # Individual mode with a crowded spot: the report explains what was
    # dropped and why
    fig = _StubFigure(np.zeros((12, 2)), [f"label_{i}" for i in range(12)])
    report = fig.label_debug_report()
    assert "Visible points: 12 / 12" in report
    assert "UNPLACED" in report
    assert "blocked by" in report

    # Grouped mode: one line per group
    fig2 = _StubFigure(
        [[0, 0], [0.004, 0], [1, 1], [5, 5]], ["A", "A", "B", "C"]
    )
    fig2.declutter_mode = "grouped"
    report2 = fig2.label_debug_report()
    assert "Groups: 3" in report2
    assert "'A'" in report2
    assert "placed" in report2

    # No labels at all
    fig3 = _StubFigure(np.zeros((1, 2)), ["X"])
    fig3.labels = None
    assert "No labels loaded" in fig3.label_debug_report()


def test_figure_disable_smart_placement_restores_defaults():
    fig = _StubFigure(np.zeros((12, 2)), [f"label_{i}" for i in range(12)])
    visible = np.arange(12)

    fig._update_label_placement(visible)
    assert len(fig._label_suppressed) > 0

    fig.smart_label_placement = False

    # All labels are back at the legacy fixed offset, undimmed and above the
    # points
    for i, vis in enumerate(fig.label_visuals):
        assert vis._label_offset is None
        assert vis.material.opacity == 1.0
        assert vis.local.position[0] == pytest.approx(fig.positions[i, 0] + 0.005)
        assert vis.local.position[2] == LABEL_Z

    # ... and the placement tick has become a no-op
    fig._update_label_placement(visible)
    assert fig._label_placement_key is None
