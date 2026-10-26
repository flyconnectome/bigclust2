"""Tests for the selection setter, the colour upload path and hover picking.

These three share a cause: each used to do O(N) Python-level work on every
interaction. The selection setter built Python ``set``s of indices, the colour
refresh copied every colour buffer twice per call, and hover scanned every
point in a visual to find the nearest one. All three now work off arrays or
the renderer's own pick buffer.

The behaviours worth locking down are the user-visible ones: what ends up
selected, what the GPU buffers hold, and which point a hover resolves to.
"""

from __future__ import annotations

import numpy as np
import pytest

from bigclust2.scatter import SCOPE_FADE_ALPHA


# --------------------------------------------------------------------------- #
# The `selected` setter
# --------------------------------------------------------------------------- #


def test_selection_is_sorted_and_deduplicated(make_scatter):
    fig, _, _ = make_scatter(n=50)
    fig.selected = np.array([5, 3, 3, 1])
    assert np.array_equal(fig.selected, [1, 3, 5])


@pytest.mark.parametrize(
    "value, expected",
    [
        (None, []),
        (7, [7]),
        ([9, 2], [2, 9]),
        (np.array([4, 4]), [4]),
    ],
    ids=["none", "scalar", "list", "array"],
)
def test_selection_accepts_the_documented_input_types(make_scatter, value, expected):
    fig, _, _ = make_scatter(n=50)
    fig.selected = value
    assert np.array_equal(fig.selected, expected)
    assert fig.selected.dtype == np.dtype(int)


def test_boolean_mask_is_converted_to_indices(make_scatter):
    fig, _, _ = make_scatter(n=50)
    mask = np.zeros(50, dtype=bool)
    mask[[2, 7]] = True
    fig.selected = mask
    assert np.array_equal(fig.selected, [2, 7])


def test_wrong_length_mask_is_rejected(make_scatter):
    fig, _, _ = make_scatter(n=50)
    with pytest.raises(AssertionError, match="same length"):
        fig.selected = np.zeros(10, dtype=bool)


def test_newly_selected_points_are_stamped_once(make_scatter):
    """`_last_selected` records the selection generation a point joined in."""
    fig, meta, _ = make_scatter(n=50)
    stamps = meta["_last_selected"]
    assert stamps.dtype == np.int32
    assert (stamps == -1).all()

    fig.selected = np.array([1, 2])
    first = fig._selection_counter
    assert (stamps.values[[1, 2]] == first).all()

    # Re-selecting the same points adds nothing new, so the counter stands still
    fig.selected = np.array([1, 2])
    assert fig._selection_counter == first

    # A genuinely new point bumps it, and only that point is re-stamped.
    fig.selected = np.array([1, 2, 3])
    assert fig._selection_counter == first + 1
    assert stamps.values[3] == first + 1
    assert (stamps.values[[1, 2]] == first).all()


def test_selection_is_restricted_to_the_scope(make_scatter):
    fig, _, _ = make_scatter(n=50)
    scope = np.zeros(50, dtype=bool)
    scope[:10] = True
    fig.set_scope(scope)

    fig.selected = np.array([5, 25])
    assert np.array_equal(fig.selected, [5]), "out-of-scope point must not select"


def test_selection_counter_label_is_updated(make_scatter):
    fig, _, _ = make_scatter(n=50)
    fig.selected = np.array([1, 2, 3])
    assert "3" in fig.selection_counter.text


# --------------------------------------------------------------------------- #
# The selection overlay
# --------------------------------------------------------------------------- #


def test_overlay_covers_exactly_the_selection(make_scatter):
    markers = np.random.default_rng(0).choice(["a", "b"], 80)
    fig, _, _ = make_scatter(n=80, markers=markers)

    fig.selected = np.array([4, 11, 70])
    covered = np.sort(np.concatenate([v._point_ix for v in fig.highlight_visuals]))
    assert np.array_equal(covered, [4, 11, 70])


def test_overlay_is_rebuilt_not_accumulated(make_scatter):
    fig, _, _ = make_scatter(n=80)
    fig.selected = np.array([1, 2, 3])
    fig.selected = np.array([9])
    covered = np.concatenate([v._point_ix for v in fig.highlight_visuals])
    assert np.array_equal(np.sort(covered), [9])


def test_highlight_points_accepts_scalars_and_lists(make_scatter):
    """`make_points` indexes with this, so a plain list used to raise."""
    markers = np.random.default_rng(0).choice(["a", "b"], 40)
    fig, _, _ = make_scatter(n=40, markers=markers)
    fig._selected = np.array([7, 9])

    for value, expected in [(7, [7]), ([7, 9], [7, 9]), (np.array([7, 9]), [7, 9])]:
        fig.highlight_points(value)
        covered = np.sort(np.concatenate([v._point_ix for v in fig.highlight_visuals]))
        assert np.array_equal(covered, expected)


# --------------------------------------------------------------------------- #
# Colour upload and scope fading
# --------------------------------------------------------------------------- #


def _buffer_colors(fig):
    return fig.point_visuals[0].geometry.colors.data


def test_colors_are_normalized_to_rgba_at_the_boundary(make_scatter):
    fig, _, _ = make_scatter(n=30, colors="red")
    assert fig.colors.shape == (30, 4)
    assert fig.colors.dtype == np.float32
    assert np.allclose(_buffer_colors(fig), [1, 0, 0, 1])


def test_set_colors_reaches_the_gpu_buffer(make_scatter):
    fig, _, _ = make_scatter(n=30, colors="red")
    fig.set_colors(np.array(["blue"] * 30), sync_to_viewer=False)
    assert np.allclose(_buffer_colors(fig), [0, 0, 1, 1])


def test_set_colors_rejects_a_length_mismatch(make_scatter):
    fig, _, _ = make_scatter(n=30)
    with pytest.raises(ValueError, match="Expected 30 colors"):
        fig.set_colors(np.array(["blue"] * 5), sync_to_viewer=False)


def test_scope_fades_out_of_scope_points_and_restores_them(make_scatter):
    fig, _, _ = make_scatter(n=40, colors="red")
    base = _buffer_colors(fig).copy()

    scope = np.zeros(40, dtype=bool)
    scope[:10] = True
    fig.set_scope(scope)

    alpha = _buffer_colors(fig)[:, 3]
    assert np.allclose(alpha[:10], 1.0)
    assert np.allclose(alpha[10:], base[10:, 3] * SCOPE_FADE_ALPHA)

    fig.set_scope(None)
    assert np.allclose(_buffer_colors(fig), base)


def test_recoloring_while_scoped_keeps_the_fade(make_scatter):
    fig, _, _ = make_scatter(n=40, colors="red")
    scope = np.zeros(40, dtype=bool)
    scope[:10] = True
    fig.set_scope(scope)

    fig.set_colors(np.array(["blue"] * 40), sync_to_viewer=False)

    colors = _buffer_colors(fig)
    assert np.allclose(colors[:10], [0, 0, 1, 1])
    assert np.allclose(colors[10:, :3], [0, 0, 1])
    assert np.allclose(colors[10:, 3], SCOPE_FADE_ALPHA)


def test_update_point_position_keeps_indices_aligned(make_scatter):
    markers = np.random.default_rng(0).choice(["a", "b"], 40)
    fig, _, _ = make_scatter(n=40, markers=markers)

    moved = np.random.default_rng(1).random((40, 2)).astype(np.float32)
    fig.positions = moved
    fig.update_point_position()

    for vis in fig.point_visuals:
        assert np.allclose(vis.geometry.positions.data[:, :2], moved[vis._point_ix])
        assert np.allclose(vis.geometry.positions.data[:, 2], 1.0)


# --------------------------------------------------------------------------- #
# Hover picking
# --------------------------------------------------------------------------- #


def _hover_at(fig, x, y):
    """Send a pointer_move through the real event pipeline and flush it."""
    fig.canvas.submit_event(
        {
            "event_type": "pointer_move",
            "x": x,
            "y": y,
            "button": 0,
            "buttons": [],
            "modifiers": [],
            "ntouches": 0,
            "touches": {},
            "pointer_id": 1,
        }
    )
    fig.canvas._events.flush()
    fig.renderer.render(fig.scene, fig.camera)
    fig.canvas.draw()


def test_hover_resolves_the_point_under_the_cursor(make_scatter):
    fig, _, points = make_scatter(n=200)
    fig.point_scale = 0.02
    fig.renderer.render(fig.scene, fig.camera)
    fig.canvas.draw()

    seen = {}
    fig._update_hover_widget = lambda text, *_: seen.update(text=text)

    target = 42
    screen = fig.world_to_screen((float(points[target, 0]), float(points[target, 1]), 0.0))
    _hover_at(fig, *screen)

    assert seen.get("text") == f"label: point-{target}"
    assert fig.hover_widget.visible


def test_moving_off_a_point_hides_the_widget(make_scatter):
    """The widget is hidden by the `pointer_leave` the renderer sends."""
    fig, _, points = make_scatter(n=200)
    fig.point_scale = 0.02
    fig.renderer.render(fig.scene, fig.camera)
    fig.canvas.draw()

    fig._update_hover_widget = lambda text, *_: None

    screen = fig.world_to_screen((float(points[0, 0]), float(points[0, 1]), 0.0))
    _hover_at(fig, *screen)
    assert fig.hover_widget.visible

    _hover_at(fig, 2, 2)  # top-left corner, outside the point cloud
    assert not fig.hover_widget.visible


def test_hover_is_suppressed_while_hidden(make_scatter):
    fig, _, points = make_scatter(n=200)
    fig.point_scale = 0.02
    fig.renderer.render(fig.scene, fig.camera)
    fig.canvas.draw()

    fig._hide_hover = True
    seen = {}
    fig._update_hover_widget = lambda text, *_: seen.update(text=text)

    screen = fig.world_to_screen((float(points[0, 0]), float(points[0, 1]), 0.0))
    _hover_at(fig, *screen)

    assert "text" not in seen


def test_no_hover_handler_when_hover_is_disabled(make_scatter):
    fig, _, _ = make_scatter(n=20, hover_col=None)
    assert not fig._hover_enabled
    assert not hasattr(fig, "hover_widget")
