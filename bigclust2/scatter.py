import re
import textwrap
import cmap
import inspect

import numpy as np
import pygfx as gfx
import pandas as pd
import pylinalg as la

from numbers import Number
from functools import partial
from itertools import combinations
from scipy.spatial import ConvexHull, Delaunay

from .figure import BaseFigure, update_figure
from .embeddings import neighborhood_fidelity
from .label_placement import (
    SLOT_RIGHT,
    connector_offsets,
    estimate_text_wh,
    measure_text_wh,
    slot_name,
    solve_label_placement,
    spatial_components,
)
from .selection import LassoGizmo, SelectionGizmo
from .utils import check_finite_features
from .visuals import points2gfx, text2gfx, lines2gfx

AVAILABLE_MARKERS = list(gfx.MarkerShape)
# Drop markers which look too similar to others
AVAILABLE_MARKERS.remove("ring")

# Alpha multiplier for points outside the current selection scope
SCOPE_FADE_ALPHA = 0.05

# Opacity for labels that could not be placed without overlap when
# `unplaced_labels` is set to "dim"
UNPLACED_LABEL_ALPHA = 0.15

# Color of the short connector lines between points and their labels
LABEL_CONNECTOR_COLOR = (0.3, 0.3, 0.3, 0.5)

# Candidate rings for group labels: crowded scenes get fallback slots farther
# out instead of dropping the label (see `slot_box`)
GROUP_LABEL_RINGS = 3

# Same-label islands further apart than this multiple of the typical point
# spacing each get their own group label (instead of one label with connector
# lines running across the whole view)
GROUP_SPLIT_FACTOR = 5.0

# Standoff between the marker edge and the start of its connector line, as a
# fraction of the marker-to-label padding
LABEL_CONNECTOR_STANDOFF = 0.35

# Z-layers for the label system. Points sit at z=0/1 (`make_points` /
# `update_point_position`); placed labels render above them, unplaced "dimmed"
# labels behind them, connector lines just below the points.
LABEL_Z = 2.0
LABEL_Z_BEHIND = -1.0
LABEL_CONNECTOR_Z = -0.5


def auto_point_size(n, base=10.0, ref=1000, min_size=0.1):
    """Default point size scaled by dataset size (inverse-sqrt, density-preserving).

    Up to `ref` points the `base` size reads well; beyond that the plot gets
    crowded, so shrink ~1/sqrt(n) (point area is proportional to 1/n, which keeps
    the inked area roughly constant) down to a `min_size` floor.
    """
    if not n or n <= ref:
        return base
    return max(min_size, base * (ref / n) ** 0.5)


def _pts_in_polygon(points, polygon):
    """Test which points lie inside a polygon (vectorised ray-casting).

    Parameters
    ----------
    points : (N, 2) array
    polygon : (M, 2) array  — open or closed polygon vertices

    Returns
    -------
    inside : (N,) bool array
    """
    px = points[:, 0]
    py = points[:, 1]
    x = polygon[:, 0]
    y = polygon[:, 1]
    n = len(polygon)
    inside = np.zeros(len(points), dtype=bool)
    j = n - 1
    for i in range(n):
        xi, yi = x[i], y[i]
        xj, yj = x[j], y[j]
        dy = yj - yi
        # Guard against horizontal edges
        safe_dy = np.where(dy == 0, 1e-10, dy)
        cross = ((yi > py) != (yj > py)) & (
            px < (xj - xi) * (py - yi) / safe_dy + xi
        )
        inside ^= cross
        j = i
    return inside


class ScatterFigure(BaseFigure):
    """A 3D scatter plot figure for visualizing point clouds with metadata."""

    selection_color = "y"
    distance_edges_threshold_default = 0.5

    def __init__(self, selection_counter=None, debug=False, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.selection_counter = selection_counter

        # Some internal state
        self.labels = None
        self.label_visuals = None
        self._label_colors = None
        self.point_visuals = None
        self.positions = None
        self.metadata = None
        # Multiple-embedding support. `embedding_entries` is a list of dicts
        # {"name", "embedding" (N,2), "features"|None, "distances"|None}; the
        # active entry's features/distances populate `self.dists`.
        self.embedding_entries = []
        self.active_embedding = None
        self._embedding_frame = None  # shared (center, scale) for normalization
        self.controls = None  # back-reference set by ScatterControls once built
        self._selected = None
        self._selection_scope_mask = None
        self.deselect_on_empty = (
            False  # whether to deselect everything on empty selection
        )
        self.deselect_on_dclick = (
            False  # whether to deselect everything on double click
        )
        self._font_size = 0.01
        self._point_scale = 0.001  # used for scaling points uniformly
        self.label_vis_limit = 400  # number of labels shown at once before hiding all
        self.label_refresh_rate = 30  # update labels every n frames
        self._smart_label_placement = True  # de-clutter labels via greedy placement
        self._declutter_mode = "individual"  # individual | grouped (one per value)
        self._unplaced_labels = "hide"  # labels that don't fit: hide | dim | show
        self._label_connectors = True  # draw lines from points to their labels
        self._label_layout_version = 0  # bumped by anything invalidating placement
        self._reset_label_placement_state()
        self._viewer_sync_enabled = True

        # Add the selection gizmo (Shift+drag → rectangle)
        self.selection_gizmo = SelectionGizmo(
            self.renderer,
            self.camera,
            self.scene,
            callback_after=lambda x: self.select_points(
                x.bounds, additive="Control" in x._event_modifiers
            ),
        )

        # Add the lasso gizmo (Shift+Command+drag → freehand polygon)
        self.lasso_gizmo = LassoGizmo(
            self.renderer,
            self.camera,
            self.scene,
            callback_after=lambda x: self.select_points_lasso(
                x.polygon, additive="Control" in x._event_modifiers
            ),
        )

        # This group will hold text labels that need to move but not scale with the figure
        self.text_group = gfx.Group()
        self.scene.add(self.text_group)

        # Generate the a container for labels
        self.label_group = gfx.Group()
        self.label_group.visible = True
        self.text_group.add(self.label_group)

        # Add some keyboard shortcuts for moving and scaling the figure
        def move_camera(x, y):
            self.camera.world.x += x
            self.camera.world.y += y
            self._render_stale = True
            self.canvas.request_draw()

        self.key_events["ArrowLeft"] = lambda: setattr(
            self, "font_size", max(self.font_size - 1, 1) if self.font_size > 1 else max(self.font_size - .1, 0.1)
        )
        self.key_events["ArrowRight"] = lambda: setattr(
            self, "font_size", self.font_size + 1 if self.font_size >= 1 else self.font_size + .1
        )
        self.key_events["ArrowDown"] = lambda: setattr(
            self, "point_scale", max(self.point_scale * 0.9, 0.0001)
        )
        self.key_events["ArrowUp"] = lambda: setattr(
            self, "point_scale", self.point_scale * 1.1
        )
        self.key_events["Escape"] = lambda: self.deselect_all()
        self.key_events["l"] = lambda: self.toggle_labels()
        # Shift+C centers on the selection. The key arrives as the typed
        # character ("C", or "c" with Caps Lock on), so register both.
        self.key_events[("C", ("Shift",))] = lambda: self.center_on_selection()
        self.key_events[("c", ("Shift",))] = self.key_events[("C", ("Shift",))]

        def _hide_selection():
            """Hide the selected neurons via the scope (see ScatterControls)."""
            if self.controls is not None:
                self.controls.hide_selection()

        self.key_events["h"] = _hide_selection

        def _remove_selection():
            """Irreversibly remove selected neurons from the current (sub)view.

            Walks up to the hosting window (mirrors `open_selection_in_new_tab`),
            which guards against removal on the main view.
            """
            window = self.canvas.window()
            while window is not None and not hasattr(
                window, "on_remove_selection_from_view"
            ):
                window = window.parent()
            if window is not None:
                window.on_remove_selection_from_view()

        self.key_events["Backspace"] = _remove_selection
        # Space cycles through the available embeddings (the Qt rendercanvas key
        # map has no entry for the space bar, so it arrives as " ").
        self.key_events[" "] = lambda: self._cycle_embedding()

        self.debug = debug

        def _control_label_vis():
            """Show only labels currently visible."""
            # Skip entirely if no labels or labels are hidden
            if self.labels is None or not self.label_group.visible:
                return

            # Check if we need to run this frame
            if self.control_label_vis_tick % self.label_refresh_rate:
                self.control_label_vis_tick += 1
                return

            # Reset the tick counter
            self.control_label_vis_tick = 1

            # Check which leafs are currently visible
            iv = self.is_visible_pos(self.positions)

            # If more points than the limit are visible, don't show any
            # labels (this applies per-point in both declutter modes)
            if iv.sum() > self.label_vis_limit:
                for i, t in enumerate(self.label_group.children):
                    t.visible = False
            elif self._smart_label_placement and self._declutter_mode == "grouped":
                # One label per unique value among the visible points
                self._update_group_labels(np.where(iv)[0])
            else:
                vis_ix = np.where(iv)[0]
                self.show_labels(vis_ix)
                self.hide_labels(np.where(~iv)[0])
                self._update_label_placement(vis_ix)

        # Turns out this is too slow to be run every frame - we're throttling it to every N frames
        self.control_label_vis_tick = 1
        self.add_animation(_control_label_vis)
        self.add_animation(self._animate_label_placement)

        self.add_animation(self.process_moves)

    def __len__(self):
        return len(self.positions) if self.positions is not None else 0

    def center_camera(self):
        """Center the camera on the scatter plot."""
        if self.positions is None:
            return

        # If any point is non-finite, pygfx can't build a bounding sphere for the
        # group and raises. Compute one from the finite points and pass it
        # explicitly (the form pygfx itself suggests in that error).
        finite = np.isfinite(self.positions).all(axis=1)
        if not finite.any():
            return
        if not finite.all():
            pts = np.asarray(self.positions[finite], dtype=np.float64)
            lo = pts.min(axis=0)
            hi = pts.max(axis=0)
            center = (lo + hi) / 2.0
            radius = float(np.linalg.norm(hi - lo)) / 2.0
            if not np.isfinite(radius) or radius <= 0:
                radius = 1.0
            self.camera.show_object((center[0], center[1], 1.0, radius))
            return

        self.camera.show_object(self.scatter_group)

    def center_on_selection(self):
        """Center and zoom the camera on the currently selected points."""
        if self.positions is None or self.selected is None or not len(self.selected):
            return

        pts = np.asarray(self.positions[self.selected], dtype=np.float64)
        finite = np.isfinite(pts).all(axis=1)
        if not finite.any():
            return
        pts = pts[finite]

        lo = pts.min(axis=0)
        hi = pts.max(axis=0)
        center = (lo + hi) / 2.0
        radius = float(np.linalg.norm(hi - lo)) / 2.0
        if not np.isfinite(radius) or radius <= 0:
            # Single point / degenerate box -> sane zoom level
            radius = 1.0
        self.camera.show_object((center[0], center[1], 1.0, radius))
        self._render_stale = True

    def clear(self):
        """Clear contents of the scatter plot."""
        for vis in ("label_group", "scatter_group"):
            vis = getattr(self, vis, None)
            if vis is not None:
                vis.clear()

        self.labels = None
        self.label_visuals = None
        self._label_colors = None
        self._reset_label_placement_state()
        self.point_visuals = None
        self.positions = None
        self.metadata = None
        self.embedding_entries = []
        self.active_embedding = None
        self._embedding_frame = None
        self._selected = None

    @property
    def labels(self):
        """Return the labels of leafs in the figure."""
        return self._labels

    @labels.setter
    @update_figure
    def labels(self, x):
        """Set the labels of leafs in the figure."""
        if x is None:
            self._labels = None
            self._label_visuals = None
            return
        assert len(x) == len(self), "Number of labels must match number of leafs."
        self._labels = np.asarray(x)
        self.update_point_labels()  # updates the visuals

    @property
    def font_size(self):
        return self._font_size * 100

    @font_size.setter
    @update_figure
    def font_size(self, size):
        size = size / 100
        self._font_size = size
        for t in self.label_visuals:
            if isinstance(t, gfx.Text):
                t.font_size = size
        for t in getattr(self, "_group_label_visuals", {}).values():
            t.font_size = size
        self._invalidate_label_placement()

    @property
    def smart_label_placement(self):
        """Whether labels are automatically arranged to avoid overlaps."""
        return self._smart_label_placement

    @smart_label_placement.setter
    @update_figure
    def smart_label_placement(self, x):
        assert isinstance(x, bool), "`smart_label_placement` must be a boolean."
        if x == self._smart_label_placement:
            return
        self._smart_label_placement = x

        if not x:
            # Restore the legacy fixed label positions and undo any
            # dimming/suppression the placement may have applied
            self._reset_label_placement_state()
            if self.label_visuals is not None and self.positions is not None:
                for i, vis in enumerate(self.label_visuals):
                    if vis is None:
                        continue
                    vis._label_offset = None
                    vis._label_target = None
                    vis.material.opacity = 1.0
                    vis.local.position = (
                        self.positions[i, 0] + 0.005,
                        self.positions[i, 1],
                        LABEL_Z,
                    )
        self._invalidate_label_placement()

    @property
    def declutter_mode(self):
        """How labels are decluttered while `smart_label_placement` is on.

        One of:
         - "individual": every visible point gets its own label, arranged to
           avoid overlaps (default)
         - "grouped": a single label per unique label value, with connector
           lines pointing to all associated points
        """
        return self._declutter_mode

    @declutter_mode.setter
    @update_figure
    def declutter_mode(self, x):
        if x not in ("individual", "grouped"):
            raise ValueError(f"Expected 'individual' or 'grouped', got {x!r}.")
        if x == self._declutter_mode:
            return
        self._declutter_mode = x

        if x == "grouped":
            # Hide the per-point machinery; the next refresh tick builds the
            # group labels
            self.hide_labels()
            vis = getattr(self, "_label_connector_vis", None)
            if vis is not None:
                vis.visible = False
            self._label_anim_active.clear()
            self._label_suppressed = set()
            self._label_placement_key = None
        else:
            grp = getattr(self, "_group_label_grp", None)
            if grp is not None:
                grp.visible = False
            self._group_placement_key = None
            self._group_data = None
            # Grouped mode only tracks highlighted indices - the per-point
            # labels still need their highlight color applied
            if self._label_highlighted:
                self.highlight_labels(
                    sorted(self._label_highlighted),
                    color=self._label_highlight_color,
                )

        self._invalidate_label_placement()

    @property
    def label_connectors(self):
        """Whether placed labels get a short connector line to their point.

        Only relevant while `smart_label_placement` is enabled.
        """
        return self._label_connectors

    @label_connectors.setter
    @update_figure
    def label_connectors(self, x):
        assert isinstance(x, bool), "`label_connectors` must be a boolean."
        if x == self._label_connectors:
            return
        self._label_connectors = x
        if not x:
            vis = getattr(self, "_label_connector_vis", None)
            if vis is not None:
                vis.visible = False
        self._invalidate_label_placement()

    @property
    def unplaced_labels(self):
        """What to do with labels that can not be placed without overlap.

        One of:
         - "hide": don't show them at all (default)
         - "dim": show them faded at their default position
         - "show": show them normally at their default position

        Only relevant while `smart_label_placement` is enabled.
        """
        return self._unplaced_labels

    @unplaced_labels.setter
    @update_figure
    def unplaced_labels(self, x):
        if x not in ("hide", "dim", "show"):
            raise ValueError(f"Expected 'hide', 'dim' or 'show', got {x!r}.")
        if x == self._unplaced_labels:
            return
        self._unplaced_labels = x
        self._invalidate_label_placement()

    @property
    def point_size(self):
        """Size for points."""
        return self._point_size

    @point_size.setter
    @update_figure
    def point_size(self, size):
        # For single sizes we can go via the material
        if not isinstance(size, (list, np.ndarray)):
            self._point_size = size
            for vis in self.point_visuals:
                if not isinstance(vis, gfx.Points):
                    continue
                vis.material.size = size * self._point_scale
                vis.material.size_mode = "uniform"
        # For variable sizes we need to go via the geometry
        else:
            self._point_size = np.array(size, dtype=np.float32)

            for vis in self.point_visuals:
                if not isinstance(vis, gfx.Points):
                    continue

                # Point sizes for this visual (in case we have multiple visuals for different markers)
                this_point_size = self._point_size[vis._point_ix]

                # Check if we have a size buffer
                if not hasattr(vis.geometry, "sizes"):
                    vis.geometry.sizes = gfx.resources._buffer.Buffer(
                        this_point_size * self._point_scale
                    )
                else:
                    vis.geometry.sizes.set_data(this_point_size * self._point_scale)

                vis.material.size_mode = "vertex"

        self._invalidate_label_placement()

    @property
    def point_scale(self):
        """Uniform scale factor for points."""
        return getattr(self, "_point_scale", 1.0)

    @point_scale.setter
    @update_figure
    def point_scale(self, scale):
        self._point_scale = np.float32(scale)  # avoid dtype issues with buffers
        self.point_size = self._point_size  # trigger a size update on the visuals

    @property
    def selected(self):
        """Return the indices of selected points in the plot."""
        return self._selected

    @selected.setter
    @update_figure
    def selected(self, x):
        """Select given points in the plot (indices!)."""
        # Any selection change that does not come from grow/shrink itself resets
        # the grow history, so shrinking only ever reverses our own grow steps
        # (see `grow_selection`/`shrink_selection`).
        if not getattr(self, "_gs_internal_update", False):
            self._gs_history = []

        if isinstance(x, type(None)):
            x = []
        elif isinstance(x, int):
            x = [x]

        if isinstance(x, np.ndarray) and x.dtype == bool:
            assert len(x) == len(
                self
            ), "Selection mask must be the same length as the plot."
            x = np.where(x)[0]

        # Which points are newly selected
        # Update the selection counter and last selected time
        newly_selected = (
            set(x) - set(self._selected) if self._selected is not None else set(x)
        )
        if newly_selected:
            self._selection_counter += 1
            self.metadata.loc[list(newly_selected), "_last_selected"] = (
                self._selection_counter
            )

        # Set the selected points (make sure to sort them)
        self._selected = np.asarray(sorted(x), dtype=int)

        # Restrict selection to the current scope (see `set_scope`)
        mask = self._selection_scope_mask
        if mask is not None and len(mask) == len(self.metadata):
            self._selected = self._selected[mask[self._selected]]

        # Create the new selection visuals
        self.highlight_points(self._selected, color=self.selection_color)

        # Update the controls
        # if hasattr(self, "_controls"):
        #     self._controls.update_ann_combo_box()

        self._sync_selection_to_viewer()

        if hasattr(self, "synced_widgets"):
            for w, func in self.synced_widgets:
                try:
                    if (
                        "datasets" in inspect.signature(func).parameters
                        and self.datasets is not None
                    ):
                        func(
                            self.selected,
                            datasets=self.datasets[self.selected],
                        )
                    else:
                        func(self.selected)
                except BaseException as e:
                    print(f"Failed to sync widget {w}:\n", e)

        if self.show_knn_edges and self.show_knn_edges["mode"] == "selected":
            self.show_knn_edges = (
                self._show_knn_edges
            )  # trigger an update of the KNN edges

        self.selection_counter.setText(f"Selected: {len(self._selected):,}  ")

    @property
    def selected_ids(self):
        """Return the IDs of selected points in the figure."""
        if self.selected is None or not len(self.selected):
            return None
        if self.ids is None:
            raise ValueError("No IDs were provided.")
        return self.ids[self.selected]

    @selected_ids.setter
    def selected_ids(self, x):
        """Select given IDs in the plot."""
        if self.ids is None:
            raise ValueError("No IDs were provided.")
        if isinstance(x, str):
            x = [x]
        elif isinstance(x, int):
            x = [x]
        elif isinstance(x, list):
            x = np.array(x)
        elif not isinstance(x, np.ndarray):
            raise ValueError(f"Expected array or list, got {type(x)}.")

        ind = np.where(np.isin(self.ids, x))[0]
        self.selected = ind

    def open_selection_in_new_tab(self, ids=None, ind=None):
        """Open the current or given selection in a new tab.

        Parameters
        ----------
        ids : array-like of int or str, optional
            IDs of the leafs to open in a new tab. If None, the currently
            selected leafs will be used.
        ind : array-like of int, optional
            Indices of the leafs to open in a new tab. If None, the currently
            selected leafs will be used.
        """
        if ids is not None and ind is not None:
            raise ValueError("Cannot specify both `ids` and `ind`.")

        curr_sel = self.selected
        if ind is not None:
            self.selected = ind
        elif ids is not None:
            self.selected_ids = ids

        window = self.canvas.window()
        while window is not None and not hasattr(
            window, "on_open_selection_in_new_view"
        ):
            window = window.parent()

        if window is None:
            raise RuntimeError(
                "Unable to find parent window to open selection in a new view."
            )

        window.on_open_selection_in_new_view()

        self.selected = (
            curr_sel  # restore the current selection after opening the new view
        )

    # Backwards-compatible alias
    open_selection_in_new_window = open_selection_in_new_tab

    @property
    def selected_ids_dataset(self):
        """Return the IDs and datasets of selected leafs in the figure."""
        if self.selected is None or not len(self.selected):
            return None
        if self.ids is None or self.datasets is None:
            raise ValueError("IDs and/or datasets were not provided.")
        return self.ids[self.selected], self.datasets[self.selected]

    @property
    def selected_labels(self):
        """Return the labels of selected leafs in the figure."""
        if self.selected is None or not len(self.selected):
            return None
        if self.labels is None:
            raise ValueError("No labels were provided.")
        return self.labels[self.selected]

    @property
    def selected_meta(self):
        """Return the metadata of selected leafs in the figure."""
        if self.selected is None or not len(self.selected):
            return None
        if self.metadata is None:
            raise ValueError("No metadata was provided.")
        return self.metadata.iloc[self.selected]

    @property
    def show_label_lines(self):
        """Show or hide the label outlines."""
        if not hasattr(self, "_show_label_lines"):
            return False
        return self._show_label_lines

    @show_label_lines.setter
    @update_figure
    def show_label_lines(self, x):
        assert isinstance(x, bool), "`show_label_lines` must be a boolean."

        if x == self.show_label_lines:
            return

        if not x and getattr(self, "label_line_group", None):
            self.label_line_group.visible = False
        elif x:
            # We're always remaking the label lines to make sure they
            # are up-to-date with the current positions and labels.
            # This is not super efficient but it is simpler than
            # trying to keep track if the lines are stale or not.
            self.make_label_lines()
            self.label_line_group.visible = True

        self._show_label_lines = x

    @property
    def show_distance_edges(self):
        """Show or hide the distance edges."""
        if not hasattr(self, "_show_distance_edges"):
            return False
        return self._show_distance_edges

    @show_distance_edges.setter
    @update_figure
    def show_distance_edges(self, x):
        assert isinstance(x, bool), "`show_distance_edges` must be a boolean."

        if x == self.show_distance_edges:
            return

        if x:
            if getattr(self, "dists", None) is None:
                raise ValueError("No distance matrix provided.")

            if not getattr(self, "distance_edge_group", None):
                self.make_distance_edges()
            self.distance_edge_group.visible = True
        elif not x and getattr(self, "distance_edge_group", None):
            self.distance_edge_group.visible = False

        self._show_distance_edges = x

    @property
    def distance_edges_threshold(self):
        """Get the distance edges threshold."""
        if not hasattr(self, "_distance_edges_threshold"):
            return self.distance_edges_threshold_default
        return self._distance_edges_threshold

    @distance_edges_threshold.setter
    @update_figure
    def distance_edges_threshold(self, x):
        """Set the distance edges threshold."""
        assert isinstance(
            x, (int, float)
        ), "`distance_edges_threshold` must be a number."

        if x == self.distance_edges_threshold:
            return

        self._distance_edges_threshold = x

        if getattr(self, "distance_edge_group", None):
            self.make_distance_edges()

    @property
    def show_knn_edges(self):
        """Show or hide the KNN edges."""
        if not hasattr(self, "_show_knn_edges"):
            return False
        return self._show_knn_edges

    @show_knn_edges.setter
    @update_figure
    def show_knn_edges(self, x):
        assert isinstance(
            x, (bool, dict)
        ), "`show_knn_edges` must be a boolean or dictionary."

        if isinstance(x, bool):
            x = {"mode": x}

        # If mode is the same and not "selected", we don't need to do anything
        if x == self.show_knn_edges and x.get("mode", None) != "selected":
            return

        if not x["mode"]:
            if getattr(self, "neighbors_edge_group", None):
                self.neighbors_edge_group.visible = False
        else:
            if x["mode"] == "selected":
                mask = np.zeros(len(self), dtype=bool)

                if self.selected is not None:
                    mask[self.selected] = True
            else:
                mask = None

            k = x.get("k", 15)
            metric = x.get("metric", "auto")
            if not self._knn_edges_drawable(metric):
                # The source backing these edges is gone (e.g. after switching
                # to an embedding without a KNN graph). Degrade gracefully
                # instead of raising inside a render/animation/selection
                # callback; the controls reconcile the GUI on switch.
                if getattr(self, "neighbors_edge_group", None):
                    self.neighbors_edge_group.visible = False
                self._show_knn_edges = False
                return
            color = x.get("color", (1, 1, 1, 0.1))
            linewidth = x.get("linewidth", 1)
            try:
                self.make_neighbour_edges(
                    k=k, metric=metric, mask=mask, color=color, linewidth=linewidth
                )
            except ValueError as e:
                # E.g. non-finite values in the feature matrix. Like the
                # missing-source case above, degrade instead of raising — this
                # setter also fires from render/animation/selection callbacks.
                self.show_message(str(e), color="red", duration=4)
                if getattr(self, "neighbors_edge_group", None):
                    self.neighbors_edge_group.visible = False
                self._show_knn_edges = False
                return

        self._show_knn_edges = x

    def _knn_edges_drawable(self, metric):
        """Whether KNN edges with `metric` can be drawn from the current sources.

        Used to degrade gracefully (rather than raise) when an overlay is
        re-applied after the active embedding's `self.dists` no longer carries
        the source the edges were drawn from.
        """
        dists = getattr(self, "dists", None)
        if dists is None:
            return False
        if not isinstance(dists, dict):
            # Bare (legacy) distance matrix -> only "precomputed"/"auto" work.
            return metric in ("precomputed", "auto")
        if not dists:  # empty dict -> no sources
            return False
        if metric == "knn":
            return "knn" in dists
        if metric == "precomputed":
            return "distances" in dists
        if metric == "auto":
            return bool(dists)
        return "features" in dists

    def _distance_edges_drawable(self):
        """Whether distance edges can be drawn from the current sources."""
        dists = getattr(self, "dists", None)
        if dists is None:
            return False
        if isinstance(dists, dict):
            return "distances" in dists
        return True  # bare (legacy) distance matrix

    @property
    def debug(self):
        """Activate/Deactive debug mode for the Scatter figure."""
        return getattr(self, "_debug", False)

    @debug.setter
    def debug(self, x):
        assert isinstance(x, bool), "`debug` must be a boolean."

        if x == self.debug:
            return

        self._debug = x
        self.selection_gizmo.debug = x
        self.lasso_gizmo.debug = x

        if self.debug:
            print("Debug mode activated.")

        else:
            print("Debug mode deactivated: hiding point indices.")

    def deselect_all(self):
        self.selected = None

    def _get_features_checked(self, context):
        """The stored feature matrix, refusing to hand out non-finite data.

        Finiteness is checked once in :meth:`set_points` (cached flag); on
        failure the full check is re-run to raise with row/column counts.
        """
        features = self.dists["features"]
        if not getattr(self, "_features_finite", True):
            check_finite_features(features, context)
        return features

    def _gs_resolve_settings(self):
        """Resolve the effective (source, metric, step) for grow/shrink.

        Falls back to the embedding (always available) when no source is set or
        the stored source is no longer valid for the current data.
        """
        from . import grow_shrink as gs

        sources = gs.available_sources(self.dists, self.positions)
        source = self._gs_source
        if source not in sources:
            source = sources[0] if sources else None
        return source, self._gs_metric, int(getattr(self, "_gs_step", 10))

    def _gs_apply(self, new_selected):
        """Assign `new_selected` without clearing the grow history."""
        self._gs_internal_update = True
        try:
            self.selected = new_selected
        finally:
            self._gs_internal_update = False

    def grow_selection(self, step=None, confirm=None):
        """Grow the current selection by pulling in nearby unselected points.

        In the default "count" mode this adds the `step` (default
        :attr:`_gs_step`) points closest to the current selection. In "threshold"
        mode (:attr:`_gs_mode`) it instead adds, in a single pass, every point
        within an auto-derived similarity distance of the selection (see
        :meth:`_grow_threshold`). In "per_neuron" mode it adds each selected
        neuron's :attr:`_gs_knn_k` nearest unselected neighbours (so each neuron
        pulls in its own matches, rather than the selection as a whole). Distance
        is measured in the configured source (embedding by default). Each grow is
        pushed onto a history stack so it can be reversed by
        :meth:`shrink_selection`.

        Parameters
        ----------
        step : int, optional
            Points to add in count mode; ignored in threshold mode.
        confirm : callable, optional
            Called with the prospective total selection size before applying; if
            it returns False the grow is aborted. Used by the GUI to confirm very
            large jumps; ``None`` (default) applies unconditionally.
        """
        from . import grow_shrink as gs

        if self._selected is None or not len(self._selected):
            self.show_message(
                "Select at least one point first", color="red", duration=2
            )
            return

        source, metric, default_step = self._gs_resolve_settings()
        if source is None:
            self.show_message(
                "No data available for grow/shrink", color="red", duration=2
            )
            return

        if source == gs.SOURCE_FEATURES and not getattr(
            self, "_features_finite", True
        ):
            self.show_message(
                "Feature vectors contain missing values (NaN) — cannot grow "
                "by feature distance",
                color="red",
                duration=4,
            )
            return

        mode = getattr(self, "_gs_mode", "count")
        try:
            if mode == "threshold":
                added = self._grow_threshold(gs, source, metric)
            elif mode == "per_neuron":
                added = gs.grow_selection_per_neuron(
                    self._selected,
                    int(getattr(self, "_gs_knn_k", 1)),
                    source=source,
                    positions=self.positions,
                    dists=self.dists,
                    metric=metric,
                    scope_mask=self._selection_scope_mask,
                )
            else:
                step = int(step) if step else default_step
                added = gs.grow_selection(
                    self._selected,
                    step,
                    source=source,
                    positions=self.positions,
                    dists=self.dists,
                    metric=metric,
                    scope_mask=self._selection_scope_mask,
                )
        except gs.GrowShrinkUnavailable as e:
            self.show_message(str(e), color="red", duration=2)
            return

        if not len(added):
            if mode == "threshold":
                msg = "No further similar points within distance"
            elif mode == "per_neuron":
                msg = "No in-scope neighbours left to add"
            else:
                msg = "All in-scope points already selected"
            self.show_message(msg, duration=2)
            return

        new_selected = np.union1d(self._selected, added)
        if confirm is not None and not confirm(len(new_selected)):
            return

        # Describe how wide a net was cast (uses the pre-grow selection).
        reach_note = self._gs_reach_note(gs, added, source, metric)

        self._gs_history.append(np.asarray(self._selected, dtype=int))
        self._gs_apply(new_selected)
        self.show_message(
            f"Selection grown by {len(added):,}{reach_note}", duration=1.5
        )

    def _gs_reach_note(self, gs, added, source, metric):
        """A ``" (reach R× spacing)"`` suffix describing how far the grow reached.

        ``reach`` is the farthest newly-added point's distance to the (pre-grow)
        selection; ``spacing`` is the median nearest-neighbour distance *within*
        that selection. The ratio is unit-free, so it reads consistently across
        distance sources. Returns ``""`` when spacing can't be derived (a single
        selected point, or a KNN selection with no internal edges) so the message
        degrades to just the count. Never raises — a stats hiccup must not break
        the (already-applied) grow.
        """
        try:
            score = gs.nearest_distance_to_selection(
                self._selected,
                source=source,
                positions=self.positions,
                dists=self.dists,
                metric=metric,
            )
            reach = float(np.max(score[added]))
            within = gs.within_selection_neighbor_distances(
                self._selected,
                source=source,
                positions=self.positions,
                dists=self.dists,
                metric=metric,
            )
            finite = within[np.isfinite(within)]
            if not finite.size:
                return ""
            spacing = float(np.median(finite))
            if spacing <= 0:
                return ""
            return f" (reach {reach / spacing:.2g}× spacing)"
        except gs.GrowShrinkUnavailable:
            return ""

    def _grow_threshold(self, gs, source, metric):
        """Indices to add for a one-shot similarity grow.

        The threshold is ``factor * max(within-selection nearest-neighbour
        distances)``; all unselected, in-scope points within that distance are
        returned (single pass). May raise ``gs.GrowShrinkThresholdUnavailable``
        (fewer than 2 points, or a KNN selection with no internal edges).
        """
        threshold = gs.selection_distance_threshold(
            self._selected,
            source=source,
            positions=self.positions,
            dists=self.dists,
            metric=metric,
            factor=float(getattr(self, "_gs_threshold_factor", 1.0)),
        )
        return gs.grow_within_threshold(
            self._selected,
            threshold,
            source=source,
            positions=self.positions,
            dists=self.dists,
            metric=metric,
            scope_mask=self._selection_scope_mask,
        )

    def shrink_selection(self):
        """Reverse the last grow step, restoring the prior selection.

        Cannot shrink below the initial (pre-grow) selection; any manual
        selection change resets the history (see the `selected` setter).
        """
        if not self._gs_history:
            self.show_message("Nothing to shrink — grow first", duration=2)
            return

        prev = self._gs_history.pop()
        removed = len(self._selected) - len(prev)
        self._gs_apply(prev)
        self.show_message(f"Selection shrunk by {removed:,}", duration=1.5)

    def highlight_points(self, points, color="y"):
        """Highlight given points in the plot.

        Parameters
        ----------
        points : array of int or bool | None
            Either indices of points to highlight or a boolean mask.
            Use `None` to clear all highlights.
        color : str
            Color to use for highlighting.

        """
        # Clear existing selection
        if hasattr(self, "highlight_visuals"):
            for vis in self.highlight_visuals:
                if vis.parent:
                    vis.parent.remove(vis)
            del self.highlight_visuals

        # If no points are given, return
        if points is None:
            return

        # If a boolean mask is given, convert it to indices
        if isinstance(points, np.ndarray) and points.dtype == bool:
            assert len(points) == len(
                self
            ), "Selection mask must be the same length as the figure."
            points = np.where(points)[0]
        elif isinstance(points, int):
            points = [points]
        elif isinstance(points, list):
            points = np.array(points)
        elif not isinstance(points, np.ndarray):
            raise ValueError(f"Expected array or list, got {type(points)}.")

        if len(points) == 0:
            return

        # Create the new selection visuals
        if len(self.selected) > 0:
            self.highlight_visuals = self.make_points(
                mask=np.isin(np.arange(len(self)), points)
            )
            for vis in self.highlight_visuals:
                vis.material.edge_color = color
                vis.material.edge_width = .2 * self.point_scale  # 20% of the point size
                vis.material.color = (1, 1, 1, 0)
                vis.material.edge_mode = "outer"
                vis.material.min_edge_width = 2  # min size in pixels
                self.scatter_group.add(vis)

    @update_figure
    def toggle_labels(self):
        """Toggle the visibility of labels."""
        self.label_group.visible = not self.label_group.visible

    def make_visuals(self, labels=True, clear=False):
        """Generate the pygfx visuals for the scatterplot."""
        if clear:
            self.clear()

        # Create the group for the points
        self.scatter_group = gfx.Group()
        self.scatter_group._object_id = "scatter"
        self.scene.add(self.scatter_group)

        self.point_visuals = self.make_points()
        for p in self.point_visuals:
            self.scatter_group.add(p)

    def make_points(self, mask=None):
        """Create the visuals for the points."""
        visuals = []

        if self.markers is None:
            self._marker_symbols = np.full(len(self), "circle")
        else:
            assert len(self.markers) == len(
                self
            ), "Length of leaf_types must match length of figure."
            unique_types = np.unique(self.markers)

            assert len(unique_types) <= len(
                AVAILABLE_MARKERS
            ), "Only 10 unique types are supported."
            marker_map = dict(zip(unique_types, AVAILABLE_MARKERS))
            self._marker_symbols = np.array([marker_map[t] for t in self.markers])

        # Create the visuals
        for m in np.unique(self._marker_symbols):
            color = "w"
            if mask is None:
                ix = np.where(self._marker_symbols == m)[0]
                this_meta = self.metadata.iloc[ix]
                this_pos = self.positions[ix]
                if self.colors is not None:
                    color = np.array(
                        [
                            tuple(cmap.Color(c).rgba)
                            for c in self.colors[self._marker_symbols == m]
                        ]
                    )
                this_size = self.point_size
            else:
                this_meta = self.metadata.iloc[mask & (self._marker_symbols == m)]
                this_pos = self.positions[mask & (self._marker_symbols == m)]
                ix = np.where(mask & (self._marker_symbols == m))[0]
                if self.colors is not None:
                    color = np.array(
                        [
                            tuple(cmap.Color(c).rgba)
                            for c in self.colors[mask & (self._marker_symbols == m)]
                        ]
                    )
                if isinstance(self.point_size, (int, float)):
                    this_size = self.point_size
                else:
                    this_size = self.point_size[mask & (self._marker_symbols == m)]
            if this_meta.empty:
                continue
            vis = points2gfx(
                np.append(
                    this_pos,
                    np.zeros(len(this_pos)).reshape(-1, 1),
                    axis=1,
                ),
                color=color,
                size=this_size * self.point_scale,
                marker=m,
                size_space="world",
                pick_write=self.hover_info_org is not None,
            )
            vis._point_ix = ix
            visuals.append(vis)

        return visuals

    def make_hover_widget(self, color=(0, 0, 0, 0.75), font_color=(1, 1, 1, 1)):
        """Generate a widget for hover info.

        The widget is rendered in the overlay scene, so coordinates are in NDC
        space and range from [-1, 1] in both dimensions.
        """
        widget = gfx.Group()
        widget.visible = False

        text = text2gfx(
            "",
            position=(0, 0, 0),
            color=(1, 1, 1, 1),
            font_size=12,
            anchor="top-left",
            screen_space=True,
        )
        widget.add(text)
        widget._text = text
        widget._border = None
        widget._patch = None

        return widget

    def _format_hover_text(self, hover_value):
        """Format hover content into a compact multi-line string.

        We typically expect the input to be a single string with newlines
        but we're adding some extra handling for other data types.
        """
        if isinstance(hover_value, pd.Series):
            hover_value = hover_value.to_dict()

        if isinstance(hover_value, dict):
            lines = [f"{k}: {v}" for k, v in hover_value.items()]
        elif isinstance(hover_value, (list, tuple, np.ndarray)):
            lines = [str(item) for item in hover_value]
        else:
            lines = str(hover_value).splitlines()

        wrapped = []
        for line in lines:
            if len(line) <= 40:
                wrapped.append(line)
            else:
                wrapped.extend(textwrap.wrap(line, width=40))

        return "\n".join(wrapped)

    def _screen_to_ndc(self, pos):
        """Convert a screen coordinate to normalized device coordinates."""
        width, height = self.size
        if width == 0 or height == 0:
            return 0, 0

        x = pos[0] / width * 2 - 1
        y = -(pos[1] / height * 2 - 1)
        return x, y

    def _update_hover_widget(self, hover_value, screen_pos_event, screen_pos_point):
        """Update hover widget content and place it near the screen position."""
        text = self._format_hover_text(hover_value)
        self.hover_widget._text.set_text(text)
        outline, patch = self.draw_bounding_box_around_text(self.hover_widget._text)

        if self.hover_widget._border:
            self.hover_widget.remove(self.hover_widget._border)
        if self.hover_widget._patch:
            self.hover_widget.remove(self.hover_widget._patch)

        if patch:
            self.hover_widget._patch = patch
            self.hover_widget.add(patch)

        if outline:
            self.hover_widget._border = outline
            self.hover_widget.add(outline)

        # We can get the dimensions of the widget from the lines geometry
        bounds = (
            outline.get_geometry_bounding_box()
        )  # [[min_x, min_y, min_z], [max_x, max_y, max_z]] in NDC space
        widget_width = bounds[1, 0] - bounds[0, 0]
        widget_height = bounds[1, 1] - bounds[0, 1]

        # Convert the screen position to NDC space. Remember, in NDC:
        # - (0, 0) = center of the screen
        # - (-1, -1) = bottom-left corner
        # - (1, 1) = top-right corner
        ndc_x, ndc_y = self._screen_to_ndc(screen_pos_point)

        # By default we will anchor the widget to the top-left corner of the cursor position.
        # However, if that would put it outside the screen, we will change the anchor to keep it inside.
        move_above = (ndc_y - widget_height) < -1
        move_left = (ndc_x + widget_width) > 1
        if move_above:
            ndc_y += widget_height
        if move_left:
            ndc_x -= widget_width

        self.hover_widget.local.position = (ndc_x, ndc_y, 0)

        if move_above and not move_left:
            anchor = "bottom-left"
        elif not move_above and move_left:
            anchor = "top-right"
        elif move_above and move_left:
            anchor = "bottom-right"
        else:
            anchor = "top-left"

        # Update the connector line to point from the widget to the cursor position
        self.udpate_connector_line(anchor, widget_width, widget_height)

    def udpate_connector_line(
        self, anchor, width, height, edge_color=(1, 1, 1, 0.25), offset=0.01
    ):
        """Update the little line connecting the hover widget to the point."""
        # Remove the existing line if it exists
        if (
            hasattr(self.hover_widget, "_connector_line")
            and self.hover_widget._connector_line
        ):
            self.hover_widget.remove(self.hover_widget._connector_line)
            del self.hover_widget._connector_line

        # Current position of the widget
        current_pos = np.array(self.hover_widget.local.position)

        # Add line and offset at the appropriate position
        if anchor == "top-left":
            start = (0, 0, 0)
            end = (-offset, offset, 0)
            new_pos = current_pos + np.array([offset, -offset, 0])
        elif anchor == "top-right":
            start = (width, 0, 0)
            end = (width + offset, offset, 0)
            new_pos = current_pos + np.array([-offset, -offset, 0])
        elif anchor == "bottom-left":
            start = (0, -height, 0)
            end = (-offset, -height - offset, 0)
            new_pos = current_pos + np.array([offset, offset, 0])
        elif anchor == "bottom-right":
            start = (width, -height, 0)
            end = (width + offset, -height - offset, 0)
            new_pos = current_pos + np.array([-offset, offset, 0])
        else:
            raise ValueError(f"Invalid anchor: {anchor}")

        line = lines2gfx(
            np.array([start, end], dtype=np.float32),
            color=edge_color,
            linewidth=2,
        )
        self.hover_widget._connector_line = line
        self.hover_widget.add(line)
        self.hover_widget.local.position = new_pos  # add the offset to the widget position to account for the connector line

    def draw_bounding_box_around_text(
        self,
        text,
        edge_color=(1, 1, 1, 0.25),
        bg_color=(0, 0, 0, 0.9),
        round_corners=True,
    ):
        """Draw a bounding box around a given text object.

        This function makes the following assumptions:
        - text is in screen space (i.e. `text.screen_space == True`)
        - text is top-left anchored (i.e. `text.anchor == "top-left"`)
        - renderer is using an NDC camera (i.e. `camera` is an instance of `gfx.NDCCamera`)

        """
        # Calculate the bounding box of the text in screen space (i.e. in pixels)
        blocks = text._text_blocks
        positions = text.geometry.positions.data
        left = np.inf
        right = -np.inf
        top = -np.inf
        bottom = np.inf
        for i, block in enumerate(blocks):
            pos_x, pos_y, _ = positions[i]
            left = min(left, pos_x + block._rect.left)
            right = max(right, pos_x + block._rect.right)
            top = max(top, pos_y + block._rect.top)
            bottom = min(bottom, pos_y + block._rect.bottom)

        # Convert the bounding box from screen space (pixels) to world space (NDC)
        canvas_width, canvas_height = (
            self.canvas.size().width(),
            self.canvas.size().height(),
        )
        left_screen = (left / canvas_width) * 2 - 1
        right_screen = (right / canvas_width) * 2 - 1
        top_screen = 1 - (top / canvas_height) * 2
        bottom_screen = 1 - (bottom / canvas_height) * 2
        width = right_screen - left_screen
        height = top_screen - bottom_screen

        if not round_corners:
            positions = np.array(
                [
                    [0, 0, 0],
                    [width, 0, 0],
                    [width, height, 0],
                    [0, height, 0],
                    [0, 0, 0],
                ],
                dtype=np.float32,
            )
        else:
            positions = _round_corners(width, 0, 0, height, radius=0.01, num_points=10)

        lines = None
        if edge_color is not None:
            lines = gfx.Line(
                gfx.Geometry(positions=positions),
                gfx.LineMaterial(color=edge_color, thickness=2),
            )
            lines.local.position = text.local.position

        patch = None
        if bg_color is not None:
            delaunay = Delaunay(positions[:, :2])
            vertices, faces = delaunay.points, delaunay.simplices
            vertices = np.append(
                vertices, np.zeros((vertices.shape[0], 1), dtype=np.float32), axis=1
            )
            patch = gfx.Mesh(
                gfx.Geometry(
                    positions=vertices.astype(np.float32),
                    indices=faces.astype(np.int32),
                ),
                gfx.MeshBasicMaterial(color=bg_color, alpha_mode="blend"),
            )
            patch.local.position = text.local.position

        return lines, patch

    def make_label_lines(self):
        """Generate the polygons around each unique label."""
        # Create a group and add to scene
        if not getattr(self, "label_line_group", None):
            self.label_line_group = gfx.Group()
            self.label_line_group.visible = self.show_label_lines
            self.scene.add(self.label_line_group)

        # Clear the group (we might call this function to update the lines)
        self.label_line_group.clear()

        # Generate a dictionary mapping a unique label to the indices
        labels = {
            l: np.where(self.labels == l)[0]
            for l in np.unique(self.labels[~pd.isnull(self.labels)])
        }

        # Generate a line for each label
        vertices = []
        faces = []
        n_vertices = 0
        for l, indices in labels.items():
            # Get the points for this label
            points = self.positions[indices]

            # Generate a convex hull around the points
            if len(points) < 3:
                continue
            hull = ConvexHull(points)

            tri = Delaunay(points[hull.vertices])

            # Add vertices and faces to the list
            vertices.append(points[hull.vertices])
            faces.append(tri.simplices + n_vertices)
            n_vertices += len(hull.vertices)

        # Concatenate all vertices and faces
        vertices = np.concatenate(vertices, axis=0)
        faces = np.concatenate(faces, axis=0)

        # Add a third coordinate to the vertices
        vertices = np.append(
            vertices, np.zeros((vertices.shape[0], 1), dtype=np.float32), axis=1
        )
        # Make sure vertices are in the back
        vertices[:, 2] = -1

        vis = gfx.Mesh(
            gfx.Geometry(
                indices=faces.astype(np.int32), positions=vertices.astype(np.float32)
            ),
            gfx.MeshBasicMaterial(color=(1, 1, 1, 0.1), alpha_mode="add"),
        )

        # Create a mesh for the label lines and add to the group
        self.label_line_group.add(vis)

    def make_distance_edges(self):
        """Generate the lines between each scatter point."""
        if self.dists is None:
            raise ValueError("No distance matrix provided.")

        # Create a group and add to scene
        if not getattr(self, "distance_edge_group", None):
            self.distance_edge_group = gfx.Group()
            self.distance_edge_group.visible = self.show_distance_edges
            self.scene.add(self.distance_edge_group)

        # Clear the group (we might call this function to update the lines)
        self.distance_edge_group.clear()

        # Get the distances. Distance edges need the square pairwise-distance
        # matrix, so prefer the "distances" source and fall back to whatever is
        # available.
        if isinstance(self.dists, dict):
            if "distances" in self.dists:
                dists = self.dists["distances"]
            else:
                dists = list(self.dists.values())[0]
        else:
            dists = self.dists

        if isinstance(dists, pd.DataFrame):
            dists = dists.values

        if dists.shape[0] != dists.shape[1]:
            raise ValueError(f"Distance matrix must be square, got {dists.shape}.")

        # Grab the threshold once
        threshold = self.distance_edges_threshold

        # We really only want one line between each point, so we need to
        # make sure to only draw the line in one direction
        # (i.e. only draw the line from i to j, not from j to i)
        lines = []
        widths = []
        for i, j in combinations(np.arange(len(dists)), 2):
            val = dists[i, j]

            if val > threshold:
                continue

            # Get the points for this line
            lines.append(np.array([self.positions[i], self.positions[j], [None, None]]))

            # Thickness of the line
            widths.append((threshold - val) / threshold * 1 + 0.2)

        # We can't afford making a visual for each line but unfortunately,
        # pygfx doesn't support lines with variable width. To work around this
        # we will digitize the lines into N bins and then create a visual for each bin.
        lines = np.array(lines)
        widths = np.array(widths)
        bins = np.histogram_bin_edges(widths, bins="auto")
        widths_bins = np.digitize(widths, bins)

        for i, width in enumerate(bins):
            this_bin = widths_bins == (i + 1)

            if not np.any(this_bin):
                continue

            # Get the points for this line
            points = lines[this_bin].reshape(-1, 2)

            # Create a line between the two points
            vis = lines2gfx(
                points,
                color=(1, 1, 1, (width - widths.min()) / (widths.max() - widths.min())),
                linewidth=width * 5,
            )

            # Add the visual to the group
            self.distance_edge_group.add(vis)

    def make_neighbour_edges(
        self, k, metric="auto", mask=None, color=(1, 1, 1, 0.1), linewidth=1
    ):
        """Generate lines between nearest neighbours.

        Important: existing edges will be cleared when calling this function!

        Parameters
        ----------
        k : int
            Number of nearest neighbours to show.
        metric : str, optional
            Distance metric to use for finding nearest neighbours.
            Default ("auto") will use precomputed distances if
            available and fall back to Euclidean distance on the
            feature vector if not.
        mask : array of bool, optional
            Boolean mask to specify which points to which to mark the nearest
            neighbour(s). Note that the nearest neighbor can still be outside
            of the mask. If None, all points are included.

        """
        if self.dists is None:
            raise ValueError("No distance matrix/feature vector provided.")

        if metric == "auto":
            if "distances" in self.dists:
                metric = "precomputed"
            elif "knn" in self.dists:
                metric = "knn"
            else:
                metric = "euclidean"

        if metric == "precomputed" and "distances" not in self.dists:
            raise ValueError("No precomputed distances available.")

        if metric == "knn" and "knn" not in self.dists:
            raise ValueError("No KNN graph available.")

        if metric not in ("precomputed", "knn"):
            if "features" not in self.dists:
                raise ValueError(
                    "No feature vectors available for distance calculation."
                )
            # Fails loudly (with counts) if the features contain NaN/inf.
            self._get_features_checked("nearest-neighbour edges")

        # Create a group and add to scene
        if not getattr(self, "neighbors_edge_group", None):
            self.neighbors_edge_group = gfx.Group()
            self.scene.add(self.neighbors_edge_group)

        # Clear the group (we might call this function to update the lines)
        self.neighbors_edge_group.clear()
        # (Re)building edges means they should be shown. Without this, a group
        # hidden by a prior "off" toggle — or by the move-completion re-apply,
        # which toggles off-then-on — would stay invisible and the edges could
        # never be brought back. (The distance-edge group does the same.)
        self.neighbors_edge_group.visible = True

        # If nothing to show, we can just return here
        if mask is not None and mask.sum() == 0:
            return

        # At this point we should have completed all checks, so we can safely
        # grab the required data. If we have a mask, we only compute nearest
        # neighbours for the masked points (the neighbours themselves can
        # still be anywhere).
        if mask is not None:
            ind = np.where(mask)[0]

        if metric == "knn":
            # Neighbors come straight from the precomputed graph (nearest-first,
            # already self-excluded). Cap k to what the graph stores; -1 padding
            # marks missing neighbors and is dropped after edge construction.
            graph = self.dists["knn"]
            kk = min(k, int(graph.k))
            if mask is None:
                ind = np.arange(len(graph))
                knn = graph.indices[:, :kk]
            else:
                knn = graph.indices[ind, :kk]
        elif metric == "precomputed":
            dists = np.asarray(self.dists["distances"])
            if mask is None:
                knn = np.argsort(dists, axis=1)[:, 1 : (k + 1)]
                ind = np.arange(len(knn))
            else:
                # Fancy indexing copies the masked rows, so we can safely
                # overwrite the self-distances below
                sub = dists[ind]
                sub[np.arange(len(ind)), ind] = np.inf  # exclude self
                # argpartition because we only need the k smallest - the order
                # doesn't matter since edges get sorted/deduplicated anyway
                knn = np.argpartition(sub, k, axis=1)[:, :k]
        else:
            from sklearn.neighbors import NearestNeighbors

            nn = NearestNeighbors(n_neighbors=k, metric=metric).fit(
                self.dists["features"]
            )
            if mask is None:
                _, knn = nn.kneighbors()  # excludes self by default
                ind = np.arange(len(knn))
            else:
                # Query only the masked points; each point comes back as its
                # own first hit, so we ask for one extra neighbour and drop it
                _, knn = nn.kneighbors(
                    np.asarray(self.dists["features"])[ind], n_neighbors=k + 1
                )
                knn = knn[:, 1:]

        # Convert to edges (i.e. pairs of points). `knn` may have fewer than `k`
        # columns (KNN graph capped to its own k).
        edges = []
        for i in range(knn.shape[1]):
            edges.append(np.stack((ind, knn[:, i]), axis=1))
        edges = np.concatenate(edges, axis=0)

        # Drop edges to missing neighbors (sentinel -1 from a filtered KNN graph).
        edges = edges[(edges >= 0).all(axis=1)]

        # Find unique edges (since the same edge can be found from both directions)
        edges = np.sort(edges, axis=1)  # sort each edge to make them comparable
        edges = np.unique(edges, axis=0)

        # No surviving edges (e.g. a sparse KNN selection whose neighbors were
        # all dropped to -1). Nothing to draw -> leave the cleared group empty
        # rather than handing lines2gfx an empty list (np.vstack would raise).
        if len(edges) == 0:
            return

        # Translate edge into coordiantes
        lines = []
        for i, j in edges:
            lines.append(np.array([self.positions[i], self.positions[j], [None, None]]))

        # Create a line between the two points
        vis = lines2gfx(
            lines,
            color=color,
            linewidth=linewidth,
        )

        # Add the visual to the group
        self.neighbors_edge_group.add(vis)

    @update_figure
    def show_labels(self, which=None):
        """Show labels for the leafs.

        Parameters
        ----------
        which : list, optional
            Indices of points for which to show the labels. If None, all labels are shown.

        """
        # Return early if no labels
        if self.labels is None:
            return

        if which is None:
            which = np.arange(len(self))
        elif isinstance(which, Number):
            which = np.array([which])
        elif isinstance(which, list):
            which = np.array(which)

        if not isinstance(which, (list, np.ndarray)):
            raise ValueError(f"Expected list or array, got {type(which)}.")

        for ix in which:
            if ix < 0:
                ix = len(self) + ix

            if self.label_visuals[ix] is None:
                label_color = (
                    self._label_colors[ix] if self._label_colors is not None else "w"
                )
                if label_color is None:
                    label_color = "w"

                t = text2gfx(
                    str(self.labels[ix]),
                    position=(
                        self.positions[ix, 0] + 0.005,
                        self.positions[ix, 1],
                        LABEL_Z,
                    ),
                    color=label_color,
                    font_size=self._font_size,
                    anchor="middle-left",
                    pickable=True,
                )

                def _highlight(event, text):
                    ls = self.find_label(text._text, go_to_first=False)
                    if "Shift" in event.modifiers:
                        ls.select_all(add="Control" in event.modifiers)

                t.add_event_handler(partial(_highlight, text=t), "double_click")

                # `_label_visuals` is in the same order as `_labels`
                self.label_visuals[ix] = t
                self.label_group.add(t)

                # Track where this label is supposed to show up (for scaling)
                t._absolute_position = self.positions[ix]

                # Center the text
                t.text_align = "center"

            self.label_visuals[ix].visible = True

    @update_figure
    def hide_labels(self, which=None):
        """Hide labels for the leafs.

        Parameters
        ----------
        which : list, optional
            Indices of points for which to hide the labels. If None, all labels are hidden.

        """
        if self.labels is None:
            return

        if which is None:
            which = np.arange(len(self))
        elif isinstance(which, int):
            which = np.array([which])
        elif isinstance(which, list):
            which = np.array(which)

        if not isinstance(which, (list, np.ndarray)):
            raise ValueError(f"Expected list or array, got {type(which)}.")

        for ix in which:
            if self.label_visuals[ix] is None:
                continue

            self.label_visuals[ix].visible = False

    def _invalidate_label_placement(self):
        """Mark the current label placement as stale.

        Called whenever something invalidates the solution: label text,
        font size, point size/scale or point positions. The actual re-solve
        happens at the next label refresh tick (see `_update_label_placement`).
        """
        self._label_layout_version += 1

    def _reset_label_placement_state(self):
        """Reset all cached label placement state (e.g. after `set_points`)."""
        self._label_placement_key = None  # (version, visible set) of last solve
        self._label_slots = {}  # label index -> last assigned slot (hysteresis)
        self._label_suppressed = set()  # labels hidden because no free slot
        self._label_extent_cache = {}  # label index -> (text, font size, (w, h), measured)
        self._label_anim_active = set()  # labels currently easing to a new position
        self._label_highlighted = set()  # labels highlighted via `highlight_labels`
        self._label_highlight_color = "y"  # color applied to highlighted labels
        # Connector lines: label index, start offset (relative to the point)
        # and end offset (relative to the label anchor) per placed label
        self._label_connector_ix = np.empty(0, dtype=int)
        self._label_connector_start = np.empty((0, 2), dtype=np.float32)
        self._label_connector_end_rel = np.empty((0, 2), dtype=np.float32)
        vis = getattr(self, "_label_connector_vis", None)
        if vis is not None:
            vis.visible = False
        # Grouped mode (one label per unique value, see `declutter_mode`)
        self._group_placement_key = None
        self._group_label_slots = {}  # label text -> last slot (hysteresis)
        self._group_label_extent_cache = {}  # label text -> (font size, (w, h), measured)
        self._group_label_visuals = {}  # label text -> gfx.Text
        self._group_data = None  # solution of the last grouped solve
        self._label_codes_cache = None  # cached factorization of self.labels
        self._group_connector_vis = None  # child of _group_label_grp (see below)
        grp = getattr(self, "_group_label_grp", None)
        if grp is not None:
            grp.clear()
            grp.visible = False

    def _label_extent(self, ix):
        """Return the (width, height) of a label in world units.

        Prefers exact extents measured from the glyph layout of an existing
        text visual; falls back to a character-count estimate for labels
        whose visual has not been created yet. Cached per label until the
        text or font size changes.
        """
        text = str(self.labels[ix])
        fs = self._font_size
        cached = self._label_extent_cache.get(ix)
        if cached is not None and cached[0] == text and cached[1] == fs and cached[3]:
            return cached[2]

        vis = self.label_visuals[ix]
        wh = measure_text_wh(vis, fs) if vis is not None else None
        measured = wh is not None
        if wh is None:
            # Re-use a previous estimate if still valid, else make a new one
            if cached is not None and cached[0] == text and cached[1] == fs:
                return cached[2]
            wh = estimate_text_wh(text, fs)

        self._label_extent_cache[ix] = (text, fs, wh, measured)
        return wh

    def _update_label_placement(self, visible_idx, debug=None):
        """(Re-)solve the placement for the currently visible labels.

        Greedily moves labels into free candidate slots around their points
        so they overlap neither markers nor each other; labels with no free
        slot are temporarily hidden ("suppressed"). Because points and labels
        are both sized in world units, a solution stays valid under pan/zoom -
        we only re-solve when the set of visible labels changes or the layout
        version was bumped (label text, font/point size, positions).

        Pass a dict as `debug` to force a re-solve and collect diagnostics
        (see `label_debug_report`).
        """
        if not self.smart_label_placement:
            return
        if self.labels is None or self.label_visuals is None:
            return

        key = (self._label_layout_version, hash(visible_idx.tobytes()))
        if debug is None and key == self._label_placement_key:
            # Solution still valid. Just re-assert suppression: show_labels()
            # in the refresh tick will have re-shown suppressed labels.
            if self._unplaced_labels == "hide":
                for ix in self._label_suppressed:
                    if self.label_visuals[ix] is not None:
                        self.label_visuals[ix].visible = False
            return
        self._label_placement_key = key

        # Point marker radii in world units (see `make_points`: world-space
        # size = point_size * point_scale, and size is the diameter)
        point_size = getattr(self, "_point_size", 1)
        scale = float(getattr(self, "_point_scale", 1.0))
        if isinstance(point_size, np.ndarray):
            radii = point_size[visible_idx].astype(float) * scale / 2
        else:
            radii = np.full(len(visible_idx), float(point_size) * scale / 2)

        extents = np.array([self._label_extent(ix) for ix in visible_idx]).reshape(
            -1, 2
        )

        # Highlighted labels get placed first (= best slots), then labels of
        # selected points, then the rest
        priority = np.full(len(visible_idx), 2.0)
        if self._selected is not None and len(self._selected):
            priority[np.isin(visible_idx, self._selected)] = 1
        if self._label_highlighted:
            priority[np.isin(visible_idx, list(self._label_highlighted))] = 0

        prev_slots = np.array(
            [self._label_slots.get(int(ix), -1) for ix in visible_idx], dtype=int
        )

        # Leave a wider gap between marker and label when connector lines are
        # drawn, so the lines are actually visible
        if len(extents):
            pad = (0.5 if self._label_connectors else 0.2) * float(
                np.median(extents[:, 1])
            )
        else:
            pad = None

        candidates = [] if debug is not None else None
        slots, offsets = solve_label_placement(
            self.positions[visible_idx],
            radii,
            extents,
            priority=priority,
            prev_slots=prev_slots,
            pad=pad,
            debug=candidates,
        )
        if debug is not None:
            debug.update(
                mode="individual",
                n_visible=len(visible_idx),
                visible_idx=visible_idx,
                pad=pad,
                extents=extents,
                slots=slots,
                prev_slots=prev_slots,
                candidates=candidates,
                policy=self._unplaced_labels,
            )

        policy = self._unplaced_labels
        self._label_suppressed = set()
        for j, ix in enumerate(visible_idx):
            ix = int(ix)
            vis = self.label_visuals[ix]
            if vis is None:
                continue

            placed = slots[j] >= 0
            if placed:
                self._label_slots[ix] = int(slots[j])
            else:
                self._label_suppressed.add(ix)
                if policy == "hide":
                    vis.visible = False
                    continue

            # Unplaced labels that are shown anyway ("dim"/"show") sit at the
            # preferred right-hand slot (the solver returns that offset);
            # dimmed ones additionally go *behind* the points
            dimmed = not placed and policy == "dim"
            vis.material.opacity = UNPLACED_LABEL_ALPHA if dimmed else 1.0
            # N.B. the third component is not an offset but the absolute
            # z-layer of the label
            offset = np.array(
                [
                    offsets[j, 0],
                    offsets[j, 1],
                    LABEL_Z_BEHIND if dimmed else LABEL_Z,
                ],
                dtype=np.float32,
            )
            target = np.array(
                [
                    self.positions[ix, 0] + offset[0],
                    self.positions[ix, 1] + offset[1],
                    offset[2],
                ],
                dtype=np.float32,
            )
            vis._label_offset = offset
            vis._label_target = target
            if not np.allclose(vis.local.position, target):
                self._label_anim_active.add(ix)

        # (Re-)build the connector lines pointing from markers to their labels
        connector_ix = []
        connector_start = []
        connector_end_rel = []
        if self._label_connectors:
            for j, ix in enumerate(visible_idx):
                ix = int(ix)
                if slots[j] < 0 or self.label_visuals[ix] is None:
                    continue
                start, end_rel = connector_offsets(
                    int(slots[j]),
                    extents[j, 0],
                    extents[j, 1],
                    radii[j],
                    gap=pad * LABEL_CONNECTOR_STANDOFF,
                )
                connector_ix.append(ix)
                connector_start.append(start)
                connector_end_rel.append(end_rel)
        self._label_connector_ix = np.array(connector_ix, dtype=int)
        self._label_connector_start = np.array(
            connector_start, dtype=np.float32
        ).reshape(-1, 2)
        self._label_connector_end_rel = np.array(
            connector_end_rel, dtype=np.float32
        ).reshape(-1, 2)
        self._rebuild_label_connector_visual()

        self._render_stale = True

    def _rebuild_label_connector_visual(self):
        """(Re-)create the line visual holding the point-to-label connectors."""
        n = len(self._label_connector_ix)
        vis = getattr(self, "_label_connector_vis", None)

        if n == 0:
            if vis is not None:
                vis.visible = False
            return

        # The number of segments changes between solves, so we swap in a
        # fresh geometry; the endpoints themselves are filled in (and kept
        # up-to-date during animations) by `_sync_label_connector_positions`
        positions = np.zeros((2 * n, 3), dtype=np.float32)
        positions[:, 2] = LABEL_CONNECTOR_Z  # just below the points
        geometry = gfx.Geometry(positions=positions)
        if vis is None:
            vis = gfx.Line(
                geometry,
                gfx.LineSegmentMaterial(
                    thickness=1.0, color=LABEL_CONNECTOR_COLOR, aa=True
                ),
            )
            self._label_connector_vis = vis
        else:
            vis.geometry = geometry

        # (Re-)parent - e.g. after `clear()` emptied the label group
        if vis.parent is not self.label_group:
            self.label_group.add(vis)
        vis.visible = True

        self._sync_label_connector_positions()

    def _sync_label_connector_positions(self):
        """Update the connector endpoints from the current label positions.

        Called after each placement solve and while labels are moving (label
        animation, point moves) so the lines stay attached to both the marker
        and the label.
        """
        vis = getattr(self, "_label_connector_vis", None)
        ix = self._label_connector_ix
        if vis is None or not len(ix) or not vis.visible:
            return

        positions = vis.geometry.positions
        for row, gix in enumerate(ix):
            label = self.label_visuals[gix]
            if label is None:
                continue
            positions.data[2 * row, :2] = (
                self.positions[gix, :2] + self._label_connector_start[row]
            )
            positions.data[2 * row + 1, :2] = (
                np.asarray(label.local.position, dtype=np.float32)[:2]
                + self._label_connector_end_rel[row]
            )
        positions.update_full()

    def _view_bounds(self):
        """Return the current viewport in world coordinates as (x0, y0, x1, y1).

        Returns None if the bounds can not be determined (e.g. headless).
        """
        try:
            top_left = self.screen_to_world((0, 0))
            bottom_right = self.screen_to_world(self.size)
            if top_left is None or bottom_right is None:
                return None
            return (top_left[0], bottom_right[1], bottom_right[0], top_left[1])
        except Exception:
            return None

    def _label_codes(self):
        """Return a cached factorization (codes, uniques) of `self.labels`."""
        cached = self._label_codes_cache
        if cached is not None and cached[0] is self.labels:
            return cached[1], cached[2]
        codes, uniques = pd.factorize(self.labels)
        self._label_codes_cache = (self.labels, codes, uniques)
        return codes, uniques

    def _group_label_extent(self, text):
        """Like `_label_extent` but for group labels (keyed by text)."""
        fs = self._font_size
        cached = self._group_label_extent_cache.get(text)
        if cached is not None and cached[0] == fs and cached[2]:
            return cached[1]

        vis = self._group_label_visuals.get(text)
        wh = measure_text_wh(vis, fs) if vis is not None else None
        measured = wh is not None
        if wh is None:
            if cached is not None and cached[0] == fs:
                return cached[1]
            wh = estimate_text_wh(text, fs)

        self._group_label_extent_cache[text] = (fs, wh, measured)
        return wh

    def _ensure_group_label_grp(self):
        """Return the (lazily created) group holding the grouped-mode visuals."""
        grp = getattr(self, "_group_label_grp", None)
        if grp is None:
            grp = gfx.Group()
            self._group_label_grp = grp
        # (Re-)parent - e.g. after `clear()` emptied the label group
        if grp.parent is not self.label_group:
            self.label_group.add(grp)
        return grp

    def _make_group_label_visual(self, key, text):
        """Create (and register) the text visual for a group label.

        `key` is a ``(text, island)`` tuple - a label value split into
        multiple spatial islands gets one visual (with the same text) per
        island.
        """
        t = text2gfx(
            text,
            position=(0, 0, LABEL_Z),
            color="w",
            font_size=self._font_size,
            anchor="middle-left",
            pickable=True,
        )
        t.text_align = "center"

        # Same interaction as per-point labels: double-click highlights,
        # Shift+double-click selects all points with this label
        def _highlight(event, text_vis):
            ls = self.find_label(text_vis._text, go_to_first=False)
            if "Shift" in event.modifiers:
                ls.select_all(add="Control" in event.modifiers)

        t.add_event_handler(partial(_highlight, text_vis=t), "double_click")

        self._group_label_visuals[key] = t
        self._group_label_grp.add(t)
        return t

    def _update_group_labels(self, visible_idx, debug=None):
        """(Re-)compute grouped labels: one label per unique value.

        Each group of visible same-labeled points gets a single label,
        anchored at the group's centroid and placed by the same greedy solver
        as individual labels - the group's spread acts as the "marker radius",
        so labels end up just outside their group and avoid the other groups.
        Connector lines fan out from the label to all member points.

        Pass a dict as `debug` to force a re-solve and collect diagnostics
        (see `label_debug_report`).
        """
        if self.labels is None:
            return

        grp = self._ensure_group_label_grp()

        key = (self._label_layout_version, hash(visible_idx.tobytes()))
        if debug is None and key == self._group_placement_key:
            # Solution still valid - but the group may have been hidden
            # wholesale by the over-limit branch of the visibility tick
            grp.visible = self._group_data is not None
            return
        self._group_placement_key = key

        # Group the visible points by label value
        codes, uniques = self._label_codes()
        vis_codes = codes[visible_idx]
        val_codes, val_ids = np.unique(vis_codes, return_inverse=True)

        if len(val_codes) == 0:
            grp.visible = False
            self._group_data = None
            self._rebuild_group_connectors()
            if debug is not None:
                debug.update(
                    mode="grouped",
                    n_visible=len(visible_idx),
                    n_groups=0,
                    over_limit=False,
                )
            return

        pts = self.positions[visible_idx].astype(float)

        # Typical point spacing, used to decide when same-label islands are
        # "disjoint": spatial clusters of one value that are further apart
        # than a multiple of it each get their own label
        if len(pts) > 1:
            pair_d = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=-1)
            np.fill_diagonal(pair_d, np.inf)
            global_spacing = float(np.median(pair_d.min(axis=1)))
        else:
            pair_d = None
            global_spacing = 0.0

        group_ids = np.zeros(len(visible_idx), dtype=int)
        texts = []  # label text per group (duplicated for split islands)
        keys = []  # (text, island) - keys `_group_label_visuals` etc.
        first_members = []  # first member (index into visible_idx) per group
        for vi in range(len(val_codes)):
            members = np.where(val_ids == vi)[0]
            text = str(uniques[val_codes[vi]])
            # The value's own spacing keeps sparse-but-contiguous groups in
            # one piece; only meaningful with >2 members (for fewer, their
            # mutual distance IS the spacing and nothing would ever split)
            if pair_d is not None and len(members) > 2:
                sub_d = pair_d[np.ix_(members, members)]
                value_spacing = float(np.median(sub_d.min(axis=1)))
            else:
                value_spacing = 0.0
            threshold = max(
                GROUP_SPLIT_FACTOR * max(global_spacing, value_spacing),
                2 * self._font_size,  # never split what sits label-close
            )
            comps = spatial_components(pts[members], threshold)
            for k in range(int(comps.max()) + 1):
                island = members[comps == k]
                group_ids[island] = len(texts)
                texts.append(text)
                keys.append((text, k))
                first_members.append(int(island[0]))

        n_groups = len(texts)
        if n_groups > self.label_vis_limit:
            grp.visible = False
            self._group_data = None
            self._rebuild_group_connectors()
            if debug is not None:
                debug.update(
                    mode="grouped",
                    n_visible=len(visible_idx),
                    n_groups=n_groups,
                    over_limit=True,
                )
            return
        grp.visible = True

        counts = np.bincount(group_ids)
        centroids = (
            np.column_stack(
                [
                    np.bincount(group_ids, weights=pts[:, 0]),
                    np.bincount(group_ids, weights=pts[:, 1]),
                ]
            )
            / counts[:, None]
        )

        # Marker radii of the members
        point_size = getattr(self, "_point_size", 1)
        scale = float(getattr(self, "_point_scale", 1.0))
        if isinstance(point_size, np.ndarray):
            member_radii = point_size[visible_idx].astype(float) * scale / 2
        else:
            member_radii = np.full(len(visible_idx), float(point_size) * scale / 2)

        # Effective group radius: the full member spread, capped at twice the
        # 90th percentile so stray outliers don't push the label far away.
        # Add the group's *largest* marker radius so size-mapped points at
        # the rim stay clear of the label.
        dists = np.linalg.norm(pts - centroids[group_ids], axis=1)
        order = np.lexsort((dists, group_ids))
        sorted_d = dists[order]
        starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
        q_ix = starts + np.minimum((0.9 * (counts - 1)).astype(int), counts - 1)
        spread = np.minimum(sorted_d[starts + counts - 1], 2 * sorted_d[q_ix])
        max_marker = np.zeros(n_groups)
        np.maximum.at(max_marker, group_ids, member_radii)
        radii = spread + max_marker

        # Create any missing visuals up front so the solve works with
        # measured text extents rather than estimates
        for key, text in zip(keys, texts):
            if key not in self._group_label_visuals:
                self._make_group_label_visual(key, text)
        extents = np.array([self._group_label_extent(t) for t in texts]).reshape(
            -1, 2
        )

        pad = (0.5 if self._label_connectors else 0.2) * float(
            np.median(extents[:, 1])
        )

        # A value's largest island is its "primary": unique labels place
        # first, extra islands from splitting only get labels if there is
        # space left afterwards
        primary = np.zeros(n_groups, dtype=bool)
        best_by_text = {}
        for gi, text in enumerate(texts):
            best = best_by_text.get(text)
            if best is None or counts[gi] > counts[best]:
                best_by_text[text] = gi
        primary[list(best_by_text.values())] = True

        has_selected = np.zeros(n_groups, dtype=bool)
        if self._selected is not None and len(self._selected):
            sel = np.isin(visible_idx, self._selected).astype(float)
            has_selected = np.bincount(group_ids, weights=sel) > 0

        # Groups containing a highlighted point (see `highlight_labels`) get
        # the highlight color and top placement priority
        highlighted = np.zeros(n_groups, dtype=bool)
        if self._label_highlighted:
            hl = np.isin(visible_idx, list(self._label_highlighted)).astype(float)
            highlighted = np.bincount(group_ids, weights=hl) > 0

        # Keep group labels inside the current viewport (a label at a
        # group's rim can otherwise easily stick out of view), with farther
        # fallback rings for crowded scenes
        bounds = self._view_bounds()

        slots = np.full(n_groups, -1, dtype=int)
        offsets = np.zeros((n_groups, 2), dtype=float)
        candidates = [None] * n_groups if debug is not None else None

        def _placed_boxes():
            """Boxes of the labels placed so far - obstacles for later passes."""
            return np.array(
                [
                    (
                        centroids[gi, 0] + offsets[gi, 0],
                        centroids[gi, 1] + offsets[gi, 1] - extents[gi, 1] / 2,
                        centroids[gi, 0] + offsets[gi, 0] + extents[gi, 0],
                        centroids[gi, 1] + offsets[gi, 1] + extents[gi, 1] / 2,
                    )
                    for gi in np.where(slots >= 0)[0]
                ]
            ).reshape(-1, 4)

        def _solve(subset, obstacle_boxes=None):
            """Solve one pass for `subset` (bigger/selected groups first)."""
            priority = -counts[subset].astype(float)
            priority[has_selected[subset]] -= 1e9
            priority[highlighted[subset]] -= 2e9
            prev = np.array(
                [self._group_label_slots.get(keys[gi], -1) for gi in subset],
                dtype=int,
            )
            cand = [] if debug is not None else None
            sub_slots, sub_offsets = solve_label_placement(
                centroids[subset],
                radii[subset],
                extents[subset],
                priority=priority,
                prev_slots=prev,
                pad=pad,
                rings=GROUP_LABEL_RINGS,
                bounds=bounds,
                # Avoid the actual points, not the groups' bounding discs -
                # a diffuse group's disc can easily cover the whole view and
                # would block everyone else's label (the disc still anchors
                # the group's own candidate slots via `radii`)
                obstacles=pts,
                obstacle_radii=member_radii,
                obstacle_boxes=obstacle_boxes,
                anchor_obstacles=False,
                debug=cand,
            )
            slots[subset] = sub_slots
            offsets[subset] = sub_offsets
            if debug is not None:
                for j, gi in enumerate(subset):
                    candidates[gi] = cand[j]

        prim_ix = np.where(primary)[0]
        _solve(prim_ix)

        # Second chance for unplaced unique labels: a group larger than the
        # viewport can never place a label on its rim (every slot is out of
        # view), so retry anchored at the - view-clamped - centroid with a
        # small radius. The label then sits over the group, still avoiding
        # the actual points and the labels placed above.
        fallback = np.zeros(n_groups, dtype=bool)
        failed = prim_ix[slots[prim_ix] < 0]
        if len(failed):
            anchors = centroids[failed].copy()
            if bounds is not None:
                anchors[:, 0] = np.clip(anchors[:, 0], bounds[0], bounds[2])
                anchors[:, 1] = np.clip(anchors[:, 1], bounds[1], bounds[3])

            fb_slots, fb_offsets = solve_label_placement(
                anchors,
                float(np.median(member_radii)),
                extents[failed],
                pad=pad,
                rings=GROUP_LABEL_RINGS,
                bounds=bounds,
                obstacles=pts,
                obstacle_radii=member_radii,
                obstacle_boxes=_placed_boxes(),
                anchor_obstacles=False,
            )
            for k, gi in enumerate(failed):
                if fb_slots[k] >= 0:
                    slots[gi] = fb_slots[k]
                    offsets[gi] = anchors[k] + fb_offsets[k] - centroids[gi]
                    fallback[gi] = True

        # Extra islands of already-labeled values fill the remaining space
        sec_ix = np.where(~primary)[0]
        if len(sec_ix):
            _solve(sec_ix, obstacle_boxes=_placed_boxes())

        if debug is not None:
            debug.update(
                mode="grouped",
                fallback=fallback,
                primary=primary,
                n_visible=len(visible_idx),
                visible_idx=visible_idx,
                n_groups=n_groups,
                over_limit=False,
                bounds=bounds,
                pad=pad,
                texts=texts,
                keys=keys,
                counts=counts,
                centroids=centroids,
                radii=radii,
                extents=extents,
                slots=slots,
                candidates=candidates,
                policy=self._unplaced_labels,
            )

        # Hide labels of groups that are no longer in view
        active = set(keys)
        for key, vis in self._group_label_visuals.items():
            if key not in active:
                vis.visible = False

        policy = self._unplaced_labels
        label_pos = np.zeros((n_groups, 3), dtype=np.float32)
        attach_rel = np.zeros((n_groups, 2), dtype=np.float32)
        shown = np.zeros(n_groups, dtype=bool)
        for gi, key in enumerate(keys):
            vis = self._group_label_visuals.get(key)
            if vis is None:
                vis = self._make_group_label_visual(key, texts[gi])

            # Group labels take the color of their first visible member -
            # unless the value is highlighted (e.g. via double-click)
            if highlighted[gi]:
                vis.material.color = self._label_highlight_color
            else:
                color = None
                if self._label_colors is not None:
                    color = self._label_colors[int(visible_idx[first_members[gi]])]
                vis.material.color = color if color is not None else "w"

            placed = slots[gi] >= 0
            if placed:
                self._group_label_slots[key] = int(slots[gi])
            elif policy == "hide":
                vis.visible = False
                continue

            dimmed = not placed and policy == "dim"
            vis.material.opacity = UNPLACED_LABEL_ALPHA if dimmed else 1.0
            lx = centroids[gi, 0] + offsets[gi, 0]
            ly = centroids[gi, 1] + offsets[gi, 1]
            if not placed and bounds is not None:
                # Labels shown despite not fitting get pulled back into view
                w_, h_ = extents[gi]
                lx = np.clip(lx, bounds[0], max(bounds[0], bounds[2] - w_))
                ly = np.clip(
                    ly,
                    bounds[1] + h_ / 2,
                    max(bounds[1] + h_ / 2, bounds[3] - h_ / 2),
                )
                # Store the effective offset so move-syncs stay consistent
                offsets[gi] = (lx - centroids[gi, 0], ly - centroids[gi, 1])
            label_pos[gi] = (lx, ly, LABEL_Z_BEHIND if dimmed else LABEL_Z)
            vis.local.position = label_pos[gi]
            vis.visible = True
            shown[gi] = True

            slot = int(slots[gi]) if placed else SLOT_RIGHT
            attach_rel[gi] = connector_offsets(
                slot, extents[gi, 0], extents[gi, 1], 0.0
            )[1]

        self._group_data = {
            "member_ix": visible_idx,
            "group_ids": group_ids,
            "texts": texts,
            "keys": keys,
            "offsets": offsets,
            "label_pos": label_pos,
            "attach_rel": attach_rel,
            "shown": shown,
            "member_radii": member_radii,
            "gap": pad * LABEL_CONNECTOR_STANDOFF,
        }
        self._rebuild_group_connectors()

        self._render_stale = True

    def _rebuild_group_connectors(self):
        """(Re-)build the connector lines from group labels to their members.

        One segment per member point, from the label's attach point to just
        outside the member's marker; fully vectorized since member counts can
        be large. Rendered behind the points (see LABEL_CONNECTOR_Z).
        """
        data = self._group_data
        vis = self._group_connector_vis

        if data is None or not self._label_connectors or not data["shown"].any():
            if vis is not None:
                vis.visible = False
            return

        g = data["group_ids"]
        mask = data["shown"][g]
        member_pos = self.positions[data["member_ix"][mask]][:, :2].astype(
            np.float32
        )
        g = g[mask]

        attach = data["label_pos"][g][:, :2] + data["attach_rel"][g]
        dvec = attach - member_pos
        dist = np.linalg.norm(dvec, axis=1, keepdims=True)
        direction = np.where(dist > 1e-12, dvec / np.maximum(dist, 1e-12), 0.0)
        standoff = (data["member_radii"][mask] + data["gap"])[:, None]
        # Clamp so the line never overshoots a member that sits closer to the
        # label than the standoff
        ends = member_pos + direction * np.minimum(standoff, dist)

        n = len(member_pos)
        positions = np.empty((2 * n, 3), dtype=np.float32)
        positions[0::2, :2] = attach
        positions[1::2, :2] = ends
        positions[:, 2] = LABEL_CONNECTOR_Z

        geometry = gfx.Geometry(positions=positions)
        if vis is None:
            vis = gfx.Line(
                geometry,
                gfx.LineSegmentMaterial(
                    thickness=1.0, color=LABEL_CONNECTOR_COLOR, aa=True
                ),
            )
            self._group_connector_vis = vis
        else:
            vis.geometry = geometry
        if vis.parent is not self._group_label_grp:
            self._group_label_grp.add(vis)
        vis.visible = True

    def _sync_group_label_positions(self):
        """Re-anchor group labels and their connectors after point moves."""
        data = self._group_data
        if data is None:
            return

        pts = self.positions[data["member_ix"]].astype(float)
        group_ids = data["group_ids"]
        counts = np.bincount(group_ids)
        centroids = (
            np.column_stack(
                [
                    np.bincount(group_ids, weights=pts[:, 0]),
                    np.bincount(group_ids, weights=pts[:, 1]),
                ]
            )
            / counts[:, None]
        )

        for gi, key in enumerate(data["keys"]):
            if not data["shown"][gi]:
                continue
            vis = self._group_label_visuals.get(key)
            if vis is None:
                continue
            data["label_pos"][gi, :2] = centroids[gi] + data["offsets"][gi]
            vis.local.position = data["label_pos"][gi]

        self._rebuild_group_connectors()

    def label_debug_report(self):
        """Diagnose label placement for the current view.

        Re-runs the placement solve with instrumentation and returns a
        human-readable report - useful to understand why a particular label
        is not showing (all candidate slots blocked, out of view, over the
        visibility limit, ...). Wired to Help -> Debug -> Labels in the GUI.
        """
        lines = ["=== Label placement debug report ==="]
        if self.labels is None or self.positions is None:
            lines.append("No labels loaded.")
            return "\n".join(lines)

        lines.append(f"Labels shown (L toggle): {self.label_group.visible}")
        lines.append(
            f"Declutter: {'on' if self._smart_label_placement else 'OFF'}"
            f" | mode: {self._declutter_mode}"
            f" | unplaced policy: {self._unplaced_labels}"
            f" | connectors: {self._label_connectors}"
            f" | font size: {self._font_size * 100:g}"
        )

        try:
            iv = self.is_visible_pos(self.positions)
            vis_note = ""
        except Exception:
            iv = np.ones(len(self.positions), dtype=bool)
            vis_note = " (visibility check unavailable - assuming all visible)"
        n_vis = int(iv.sum())
        lines.append(
            f"Visible points: {n_vis} / {len(self.positions)}"
            f" (limit: {self.label_vis_limit}){vis_note}"
        )
        bounds = self._view_bounds()
        if bounds is not None:
            bounds_str = ", ".join(f"{float(b):.4g}" for b in bounds)
            lines.append(f"View bounds (x0, y0, x1, y1): ({bounds_str})")
        else:
            lines.append("View bounds: unknown")

        if n_vis > self.label_vis_limit:
            lines.append(
                "=> Over the visibility limit: ALL labels are hidden at this "
                "zoom level. Zoom in further to see labels."
            )
            return "\n".join(lines)
        if not self._smart_label_placement:
            lines.append(
                "=> Decluttering is off: labels use fixed positions, "
                "nothing to diagnose."
            )
            return "\n".join(lines)

        vis_ix = np.where(iv)[0]
        dbg = {}
        if self._declutter_mode == "grouped":
            self._update_group_labels(vis_ix, debug=dbg)
            lines.extend(self._format_group_label_debug(dbg))
        else:
            self._update_label_placement(vis_ix, debug=dbg)
            lines.extend(self._format_individual_label_debug(dbg))
        return "\n".join(lines)

    @staticmethod
    def _describe_blocker(kind, blocker, names, marker_word, obstacle_names=None):
        """One-phrase description of why a candidate slot was rejected."""
        if kind == "out-of-view":
            return "out of view"
        if isinstance(blocker, tuple):
            tag, k = blocker
            if tag == "obstacle" and obstacle_names is not None:
                return f"blocked by {obstacle_names[int(k)]}"
            if tag == "box":
                return "blocked by an already-placed label"
            return f"blocked by the label {names[int(k)]!r}"
        return f"blocked by {marker_word} {names[int(blocker)]!r}"

    def _format_group_label_debug(self, dbg):
        """Format the grouped-mode part of `label_debug_report`."""
        if not dbg:
            return ["No solve ran (no debug data collected)."]
        if dbg.get("over_limit"):
            return [
                f"=> {dbg['n_groups']} groups exceed the limit of "
                f"{self.label_vis_limit} - all group labels are hidden."
            ]
        if dbg.get("n_groups", 0) == 0:
            return ["No label groups among the visible points."]

        slots = dbg["slots"]
        texts = dbg["texts"]
        # Obstacles are the individual visible points, in visible_idx order
        obstacle_names = [
            f"point #{int(gix)} ({str(self.labels[int(gix)])!r})"
            for gix in dbg["visible_idx"]
        ]
        n_placed = int((slots >= 0).sum())
        lines = [
            f"Groups: {dbg['n_groups']} | placed: {n_placed} | unplaced: "
            f"{dbg['n_groups'] - n_placed} (policy: {dbg['policy']})"
        ]
        dbg_keys = dbg.get("keys")
        for gi, text in enumerate(texts):
            display = f"{text!r}"
            # A value split into multiple spatial islands gets one label each
            if dbg_keys is not None and texts.count(text) > 1:
                display += f" (island {dbg_keys[gi][1] + 1})"
            cx, cy = dbg["centroids"][gi]
            w, h = dbg["extents"][gi]
            head = (
                f"[{display}] members={int(dbg['counts'][gi])}"
                f" centroid=({cx:.4g}, {cy:.4g})"
                f" radius={dbg['radii'][gi]:.4g}"
                f" label={w:.4g}x{h:.4g}"
            )
            fallback = dbg.get("fallback")
            prim_mask = dbg.get("primary")
            secondary = prim_mask is not None and not prim_mask[gi]
            if slots[gi] >= 0:
                rejected = len(dbg["candidates"][gi])
                note = f" (after {rejected} rejected slots)" if rejected else ""
                if fallback is not None and fallback[gi]:
                    note += " [via centroid fallback]"
                if secondary:
                    note += " [extra island]"
                lines.append(f"{head} -> placed {slot_name(slots[gi])}{note}")
            else:
                if secondary:
                    lines.append(
                        f"{head} -> UNPLACED extra island (no space left; "
                        "the value keeps its other label)"
                    )
                else:
                    lines.append(
                        f"{head} -> UNPLACED (policy: {dbg['policy']}; "
                        "centroid fallback also failed)"
                    )
                for slot, kind, blocker in dbg["candidates"][gi]:
                    reason = self._describe_blocker(
                        kind, blocker, texts, "group", obstacle_names
                    )
                    lines.append(f"    {slot_name(slot)}: {reason}")
        return lines

    def _format_individual_label_debug(self, dbg):
        """Format the individual-mode part of `label_debug_report`."""
        if not dbg:
            return ["No solve ran (no debug data collected)."]

        slots = dbg["slots"]
        vis_ix = dbg["visible_idx"]
        names = [str(self.labels[int(i)]) for i in vis_ix]
        n = len(vis_ix)
        n_placed = int((slots >= 0).sum())
        lines = [
            f"Labels: {n} visible | placed: {n_placed} | unplaced: "
            f"{n - n_placed} (policy: {dbg['policy']})"
        ]
        if n == n_placed:
            lines.append("All labels placed - nothing was dropped.")
        for j in np.where(slots < 0)[0]:
            gix = int(vis_ix[j])
            w, h = dbg["extents"][j]
            lines.append(
                f"[{names[j]!r}] point #{gix}"
                f" at ({self.positions[gix, 0]:.4g}, {self.positions[gix, 1]:.4g})"
                f" label={w:.4g}x{h:.4g} -> UNPLACED (policy: {dbg['policy']})"
            )
            for slot, kind, blocker in dbg["candidates"][j]:
                reason = self._describe_blocker(kind, blocker, names, "the marker of")
                lines.append(f"    {slot_name(slot)}: {reason}")
        return lines

    def _animate_label_placement(self):
        """Ease label visuals towards their assigned positions.

        Runs as an animation but early-exits (cheaply) unless a placement
        update has just moved labels.
        """
        if not self._label_anim_active or self.label_visuals is None:
            self._label_anim_active.clear()
            return

        done = []
        for ix in self._label_anim_active:
            vis = self.label_visuals[ix] if ix < len(self.label_visuals) else None
            target = getattr(vis, "_label_target", None)
            if vis is None or target is None or not vis.visible:
                done.append(ix)
                continue

            current = np.asarray(vis.local.position, dtype=np.float32)
            delta = target - current
            # Snap once we're within a fraction of the font size
            if np.abs(delta).max() < self._font_size * 0.05:
                vis.local.position = target
                done.append(ix)
            else:
                eased = current + delta * 0.35
                # Never ease z: layer changes apply instantly so labels don't
                # transiently pop through the points mid-animation
                eased[2] = target[2]
                vis.local.position = eased

        for ix in done:
            self._label_anim_active.discard(ix)

        # Keep the connector lines attached to the moving labels
        self._sync_label_connector_positions()

        self._render_stale = True

    @update_figure
    def find_label(
        self, label, regex=False, highlight=True, go_to_first=True, verbose=True
    ):
        """Find and center the plot on a given label.

        Parameters
        ----------
        label : str
            The label to search for.
        highlight : bool, optional
            Whether to highlight the found label.
        go_to_first : bool, optional
            Whether to go to the first occurrence of the label.
        verbose : bool, optional
            Whether to show a message when the label is found.

        Returns
        -------
        LabelSearch
            An object that can be used to iterate over the found labels.

        """
        ls = LabelSearch(self, label, go_to_first=go_to_first, regex=regex)

        if highlight:
            self.highlight_labels(ls.indices)

        if verbose:
            self.show_message(
                f"Found {len(ls)} occurrences of '{label}'",
                duration=3,
                color="g" if len(ls) else "r",
            )

        return ls

    @update_figure
    def highlight_labels(self, x, color="y"):
        """Highlight given label(s) in the plot.

        Parameters
        ----------
        x : str | iterable | None
            Can be either:
             - a string with a label to highlight.
             - an iterable with indices  of points to highlight
             - `None` to clear the highlights.
        color : str, optional
            The color to use for highlighting.

        """
        if self.labels is None:
            return

        # Reset existing highlights
        for vis in self.label_visuals:
            if vis is None:
                continue
            if hasattr(vis.material, "_original_color"):
                vis.material.color = vis.material._original_color

        # Highlighted labels are exempt from placement suppression and get
        # priority slots - changing the highlights hence requires a re-solve
        # (in grouped mode the re-solve is also what (re-)colors the labels)
        self._label_highlighted = set()
        self._label_highlight_color = color
        self._invalidate_label_placement()

        # Return here if we're only clearing the highlights
        if x is None:
            return

        if isinstance(x, str):
            indices = [i for i, label in enumerate(self.labels) if label == x]
        elif isinstance(x, (list, np.ndarray)):
            indices = x
        else:
            raise ValueError(f"Expected str or list, got {type(x)}.")

        # In grouped mode there are no per-point labels to recolor (creating
        # them here would wrongly show individual labels on top of the group
        # ones) - the group labels pick up the highlight on the re-solve
        grouped = self._smart_label_placement and self._declutter_mode == "grouped"

        for ix in indices:
            # Index in the original order
            ix = int(ix)
            self._label_highlighted.add(ix)
            self._label_suppressed.discard(ix)

            if grouped:
                continue

            if self.label_visuals[ix] is None:
                self.show_labels(ix)
            visual = self.label_visuals[ix]

            visual.material._original_color = visual.material.color
            visual.material.color = color

    @update_figure
    def select_points(self, bounds, additive=False):
        """Select all selectable objects in the region.

        Parameters
        ----------
        bounds : np.ndarray
            A (2, 2) array with the selection bounds in world coordinates.
        additive : bool, optional
            Whether to add to the existing selection.

        """
        # Get the positions and original indices of the leaf nodes
        positions_abs = []
        indices = []
        for l in self.point_visuals:
            positions_abs.append(
                la.vec_transform(l.geometry.positions.data, l.world.matrix)
            )
            indices.append(l._point_ix)
        positions_abs = np.vstack(positions_abs)
        indices = np.concatenate(indices)

        # Check if any of the points are within the selection region
        selected = (
            (positions_abs[:, 0] >= bounds[0, 0])
            & (positions_abs[:, 0] <= bounds[1, 0])
            & (positions_abs[:, 1] >= bounds[0, 1])
            & (positions_abs[:, 1] <= bounds[1, 1])
        )
        selected = indices[selected]

        if not len(selected) and not self.deselect_on_empty:
            return

        if additive and self.selected is not None:
            selected = np.unique(np.concatenate((self.selected, selected)))

        self.selected = selected

    @update_figure
    def select_points_lasso(self, polygon, additive=False):
        """Select all points inside a freehand lasso polygon.

        Parameters
        ----------
        polygon : (N, 2) array | None
            Polygon vertices in world coordinates. None is a no-op.
        additive : bool
            Whether to add to the existing selection.
        """
        if polygon is None:
            return

        positions_abs = []
        indices = []
        for l in self.point_visuals:
            positions_abs.append(
                la.vec_transform(l.geometry.positions.data, l.world.matrix)
            )
            indices.append(l._point_ix)
        positions_abs = np.vstack(positions_abs)
        indices = np.concatenate(indices)

        selected = indices[_pts_in_polygon(positions_abs[:, :2], polygon)]

        if not len(selected) and not self.deselect_on_empty:
            return

        if additive and self.selected is not None:
            selected = np.unique(np.concatenate((self.selected, selected)))

        self.selected = selected

    @update_figure
    def set_points(
        self,
        points,
        metadata,
        label_col="label",
        id_col="id",
        color_col="color",
        marker_col="dataset",
        hover_col="hover_info",
        dataset_col="dataset",
        point_size=None,
        distances=None,
        features=None,
        knn=None,
    ):
        """Set the scatter points and associated metadata.

        Parameters
        ----------
        points : np.ndarray
            An (N, 3) array of 3D coordinates for the scatter points.
        metadata : pandas DataFrame
            A DataFrame containing metadata for each point.
        label_col : str, optional
            Column name for point labels, by default "label"
        id_col : str, optional
            Column name for point IDs, by default "id"
        color_col : str, optional
            Column name for point colors, by default "color"
        marker_col : str, optional
            Column name for point markers, by default "dataset"
        hover_col : str, optional
            Column name for hover information, by default "hover_info". Can also be a format string using other columns.
        dataset_col : str, optional
            Column name for dataset identifiers, by default "dataset"
        point_size : int, optional
            Size of the points. If ``None`` (default), the size is auto-scaled by
            the number of points (see :func:`auto_point_size`): 10 for up to ~1000
            points, shrinking for larger datasets to avoid overlap.
        distances/features : np.ndarray, optional
            An (N, N) array of pairwise distances between points, or an (N, M) array of features.
            If provided, these can be used to re-compute point positions based on dimensionality reduction techniques.
        knn : KNNGraph, optional
            A precomputed k-nearest-neighbors graph (see :class:`bigclust2.embeddings.KNNGraph`).
            Used as a lightweight stand-in for a full distance matrix: enables
            recomputing embeddings (UMAP/t-SNE), KNN-based clustering and fidelity
            without the full (N, N) matrix.
        """
        # Make sure metadata has RangeIndex
        assert isinstance(
            metadata.index, pd.RangeIndex
        ), "Metadata index must be a RangeIndex."

        self.positions = points.astype(np.float32)
        self.metadata = metadata

        for col in [label_col, id_col, color_col, marker_col, dataset_col]:
            if col not in metadata.columns:
                raise ValueError(f"Column '{col}' not found in metadata.")

        # Add column to track when this row was last selected (for selection history tracking)
        self.metadata["_last_selected"] = -1
        self._selection_counter = 0

        self.default_label_col = label_col
        self.default_color_col = color_col
        self.label_visuals = [None] * len(metadata) if label_col else None
        self._label_colors = [None] * len(metadata) if label_col else None
        self._reset_label_placement_state()
        self._invalidate_label_placement()
        self.labels = metadata[label_col].astype(str).values if label_col else None
        self.ids = metadata[id_col].values if id_col else None
        self.colors = metadata[color_col].values if color_col else None
        self.markers = metadata[marker_col].values if marker_col else None

        # Make sure distances and features have the same order as the points
        # N.B. we're tracking both distances and features in a dictionary
        # to allow for multiple distance metrics
        if distances is None and features is None and knn is None:
            self.dists = None
        else:
            self.dists = {}

        if distances is not None:
            if distances.shape[0] != distances.shape[1]:
                raise ValueError(
                    f"Distance matrix must be square, got {distances.shape}."
                )
            if not distances.shape[0] == len(metadata):
                raise ValueError(
                    f"Distance matrix must have the same number of rows as metadata, got {distances.shape[0]} and {len(metadata)}."
                )
            if not np.all(distances.index == self.ids) or not np.all(
                distances.columns == self.ids
            ):
                raise ValueError(
                    "Index and columns of distance matrix must match IDs in metadata."
                )

            self.dists["distances"] = distances

        # Assume finite until a feature matrix proves otherwise; checked once
        # here so the latency-sensitive consumers (e.g. grow/shrink) don't have
        # to rescan a potentially large matrix on every call.
        self._features_finite = True

        if features is not None:
            if features.shape[0] != len(metadata):
                raise ValueError(
                    f"Number of rows in features must match number of points, got {features.shape[0]} and {len(metadata)}."
                )
            if not np.all(features.index == self.ids):
                raise ValueError("Index of features must match IDs in metadata.")

            self._features_finite = check_finite_features(
                features, "distance computations", action="warn"
            )
            self.dists["features"] = features

        if knn is not None:
            if len(knn) != len(metadata):
                raise ValueError(
                    f"KNN graph must have the same number of rows as metadata, "
                    f"got {len(knn)} and {len(metadata)}."
                )
            if self.ids is not None and not np.array_equal(
                np.asarray(knn.ids), np.asarray(self.ids)
            ):
                raise ValueError(
                    "KNN graph ids must match the IDs in metadata (same order)."
                )

            self.dists["knn"] = knn

        # Datasets are used to avoid collisions when the same ID is used in different datasets
        self.datasets = self.metadata[dataset_col].values if dataset_col else None

        self.hover_info_org = hover_col  # keep the original, unprocessed hover info
        # Track which columns are available and which are currently shown
        self._hover_col_names_all = [
            c for c in metadata.columns if not str(c).startswith("_")
        ] if hover_col is not None else []
        self._hover_col_names_active = None  # None means "show all"
        # Whether to build hover text from the active columns instead of the
        # original hover_col (flips on first hover-column change)
        self._hover_use_cols = False

        # Update some internal state
        # (note that we're writing to the protected member variables here)
        self._selected = None
        self._point_size = 1
        if point_size is None:
            # point_size = auto_point_size(len(metadata))
            point_size = 0.003
        self._point_scale = point_size

        # Grow/shrink-selection state (see `grow_selection`/`shrink_selection`)
        self._gs_source = None  # None => default to embedding at call time
        self._gs_metric = "euclidean"  # only used when source == "features"
        self._gs_step = 10  # points added per grow press (count mode)
        self._gs_knn_k = 1  # neighbours per selected neuron (per_neuron mode)
        self._gs_mode = "count"  # {"count", "threshold", "per_neuron"}
        self._gs_threshold_factor = 1.0  # multiplier for the similarity threshold
        self._gs_history = []  # stack of pre-grow selection snapshots
        self._gs_internal_update = False  # True while we drive the `selected` setter

        # Generate the visuals
        self.make_visuals()

        # Re-apply scope fading to the new visuals (ignored if the
        # scope mask no longer matches the number of points)
        self._apply_scope_to_visuals()

        # Setup hover info
        if hover_col is not None:

            def hover(event):
                # Note: we could use e.g. shift-hover to show more/different info?
                if event.type == "pointer_enter" and not getattr(
                    self, "_hide_hover", False
                ):
                    pos = self.screen_to_world((event.x, event.y))
                    vis = event.current_target
                    coords = vis.geometry.positions.data
                    dist = np.linalg.norm(coords[:, :2] - pos[:2], axis=1)
                    closest = np.argmin(dist)
                    point_ix = vis._point_ix[closest]

                    self.hover_widget.visible = True
                    self._update_hover_widget(
                        self.get_hover_info(point_ix),
                        (event.x, event.y),
                        self.world_to_screen(coords[closest]),
                    )
                elif self.hover_widget.visible:
                    self.hover_widget.visible = False

                # Request a redraw
                self.canvas.request_draw()

            for vis in self.point_visuals:
                vis.add_event_handler(hover, "pointer_enter", "pointer_leave")

            self.hover_widget = self.make_hover_widget()
            self.overlay_scene.add(self.hover_widget)

        # Show center camera on the new points
        self.camera.show_object(self.scatter_group)

    def sync_viewer(self, viewer):
        """Sync the figure with a neuroglancer viewer."""
        self.ngl_viewer = viewer

    @property
    def viewer_sync_enabled(self):
        """Whether scatter selection updates are synced to the 3D viewer."""
        return self._viewer_sync_enabled

    @viewer_sync_enabled.setter
    def viewer_sync_enabled(self, enabled):
        self._viewer_sync_enabled = bool(enabled)

    def set_viewer_sync(self, enabled, sync_now=False):
        """Enable/disable viewer sync and optionally sync current selection once."""
        self.viewer_sync_enabled = enabled
        if self.viewer_sync_enabled and sync_now:
            self._sync_selection_to_viewer()

    @property
    def hover_col_names(self):
        """List of metadata column names currently shown in hover tooltips."""
        if self._hover_col_names_active is not None:
            return list(self._hover_col_names_active)
        return list(getattr(self, "_hover_col_names_all", []))

    @hover_col_names.setter
    def hover_col_names(self, col_names):
        """Set which metadata columns are shown in hover tooltips."""
        all_cols = getattr(self, "_hover_col_names_all", [])
        # Validate and filter to known columns, preserving original order
        valid = [c for c in all_cols if c in col_names]
        self._hover_col_names_active = valid if valid != all_cols else None
        self._recompute_hover_info()

    def _recompute_hover_info(self):
        """Switch hover text to the active hover columns (computed lazily on hover)."""
        self._hover_use_cols = True

    @property
    def hover_info(self):
        """Hover text for all points (computed on demand)."""
        if getattr(self, "hover_info_org", None) is None:
            return None
        return np.array(
            [
                self._format_hover_text(self.get_hover_info(i))
                for i in range(len(self.metadata))
            ]
        )

    def get_hover_info(self, ix):
        """Compute hover info for a single point on demand."""
        if getattr(self, "_hover_use_cols", False):
            # Series of the active columns; _format_hover_text renders
            # them as "column: value" lines
            return self.metadata.iloc[ix][self.hover_col_names]
        hover_col = self.hover_info_org
        if "{" in hover_col:
            return hover_col.format_map(self.metadata.iloc[ix])
        return self.metadata[hover_col].iloc[ix]

    def _sync_selection_to_viewer(self):
        """Push the current scatter selection to the synced viewer."""
        if not hasattr(self, "ngl_viewer") or not self.viewer_sync_enabled:
            return

        selected = self._selected if self._selected is not None else np.array([], dtype=int)
        if len(selected) > 0:
            final_group_name = None
            if getattr(self, "_add_as_group", False):
                # Find a sensible name for the group based on selected labels.
                selected_labels = list(set(self.selected_labels.astype(str)))
                if len(selected_labels) == 1:
                    group_name = selected_labels[0]
                elif len(selected_labels) <= 3:
                    group_name = "+".join(selected_labels)
                else:
                    group_name = "mixed"

                # Ensure a unique group name in the viewer
                i = 1
                final_group_name = group_name
                while final_group_name in self.ngl_viewer.viewer.objects:
                    final_group_name = group_name + f" - #{i}"
                    i += 1

                i = 1
                final_group_name = group_name
                while final_group_name in self.ngl_viewer.viewer.objects_grouped:
                    final_group_name = group_name + f" - #{i}"
                    i += 1

            self.ngl_viewer.show(
                self.ids[selected],
                datasets=self.datasets[selected],
                add_as_group=final_group_name,
            )
        else:
            self.ngl_viewer.clear()

    def sync_widget(self, widget, callback=None):
        """Connect a widget to the figure.

        Parameters
        ----------
        widget
                The widget to sync.
        callback
                The function to call. If `None`, the widget must implement a
                `.select()` method that takes a list of indices to select.

        """
        if callback is None:
            assert hasattr(widget, "select") and callable(
                widget.select
            ), "Widget must have a `select` method that takes a list of indices to select."
            callback = widget.select

        if not hasattr(self, "synced_widgets"):
            self.synced_widgets = []

        if (widget, callback) not in self.synced_widgets:
            self.synced_widgets.append((widget, callback))

    def unsync_widget(self, widget):
        """Disconnect a widget from the figure."""
        if not hasattr(self, "synced_widgets"):
            return

        self.synced_widgets = [(w, cb) for w, cb in self.synced_widgets if w != widget]

    def set_colors(self, colors, sync_to_viewer=True):
        """Set the colors for the points in the scatterplot.

        Parameters
        ----------
        colors : np.ndarray
            Can be:
                - (N, ) list of string colors
                - (N, 3) array of RGB colors
                - (N, 4) array of RGBA colors
            N must match the number of points in the scatterplot.
            Same order as the points provided in `set_points()`.
        sync_to_viewer : bool
            Whether to also update the colors in the synced neuroglancer viewer (if any).

        """
        if isinstance(colors, pd.arrays.ArrowStringArray):
            colors = np.array(colors)

        if not isinstance(colors, (list, np.ndarray)):
            raise ValueError(f"Expected list or array, got {type(colors)}.")

        if len(colors) != len(self):
            raise ValueError(f"Expected {len(self)} colors, got {len(colors)}.")

        if isinstance(colors, list) and isinstance(colors[0], str):
            colors = np.array([tuple(cmap.Color(c).rgba) for c in colors])
        elif isinstance(colors, np.ndarray):
            if colors.ndim == 1:
                colors = np.array([tuple(cmap.Color(c).rgba) for c in colors])
            elif colors.ndim == 2 and colors.shape[1] == 3:
                # Add an alpha channel if not provided
                colors = np.hstack((colors, np.ones((len(colors), 1))))

        self.colors = colors.astype(np.float32)

        for vis in self.point_visuals:
            vis._base_colors = self.colors[vis._point_ix]
        self._apply_scope_to_visuals()

        if sync_to_viewer and hasattr(self, "ngl_viewer"):
            self.set_viewer_colors(colors)

    def set_scope(self, mask):
        """Restrict selections to a subset of points.

        Points outside the scope can not be selected and are shown faded.

        Parameters
        ----------
        mask :  (N,) bool array | None
                True = selectable. N must match the number of points in the
                scatterplot. Use `None` to clear the scope.

        """
        if mask is not None:
            mask = np.asarray(mask).astype(bool, copy=False)
            if len(mask) != len(self):
                raise ValueError(
                    f"Expected mask of length {len(self)}, got {len(mask)}."
                )
        self._selection_scope_mask = mask

        # Prune the current selection to the new scope (re-assigning routes
        # through the `selected` setter, which applies the mask, refreshes
        # the highlights and syncs widgets/viewer)
        if mask is not None and self._selected is not None and len(self._selected):
            self.selected = self._selected

        self._apply_scope_to_visuals()

    def _apply_scope_to_visuals(self):
        """(Re-)apply scope fading on top of the base point colors."""
        mask = self._selection_scope_mask
        if mask is not None and len(mask) != len(self):
            mask = None  # stale mask (e.g. after set_points) -> ignore

        if self.point_visuals is None:
            return

        for vis in self.point_visuals:
            if not hasattr(vis, "_base_colors"):
                # Lazily capture the unfaded colors created by make_points()
                vis._base_colors = vis.geometry.colors.data.copy()
            colors = vis._base_colors.copy()
            if mask is not None:
                colors[~mask[vis._point_ix], 3] *= SCOPE_FADE_ALPHA
            vis.geometry.colors.set_data(colors)
            vis.geometry.colors.update_full()

        self._render_stale = True
        self.canvas.request_draw()

    def set_viewer_colors(self, colors):
        """Set the colors for the neuroglancer viewer.

        Parameters
        ----------
        colors :    dict | array-like
                    Dictionary of colors keyed by IDs ({id: color, ...}
                    or{(id, dataset): color, ...}) or array-like of colors
        """
        if not hasattr(self, "ngl_viewer"):
            raise ValueError("No neuroglancer viewer is connected.")

        if isinstance(colors, pd.arrays.ArrowStringArray):
            colors = np.array(colors)

        if isinstance(colors, (np.ndarray, list)):
            if self.ids is None:
                colors = {i: c for i, c in enumerate(colors)}
            elif self.datasets is None:
                colors = {i: c for i, c in zip(self.ids, colors)}
            else:
                colors = {(i, d): c for i, d, c in zip(self.ids, self.datasets, colors)}
        elif not isinstance(colors, dict):
            raise ValueError("Colors must be a dictionary.")

        self.ngl_viewer.set_colors(colors)

    @update_figure
    def update_point_labels(self):
        """Update the point labels."""
        if self.labels is None:
            return

        for i, l in enumerate(self.label_visuals):
            if l is None:
                continue
            l.set_text(self.labels[i])
            l._text = self.labels[i]

        # New texts = new label extents and new grouping
        self._label_codes_cache = None
        self._invalidate_label_placement()

    @update_figure
    def update_point_position(self):
        """Update the point positions from the figure's `positions` property."""
        # Add a z coordinate of 1
        xyz = np.append(
            self.positions, np.ones((len(self.positions), 1)), axis=1
        ).astype(np.float32)

        # Update the positions of the points
        for vis in self.point_visuals:
            vis.geometry.positions.set_data(xyz[vis._point_ix])
            vis.geometry.positions.update_full()

        # Update the positions of the labels (keeping any offset - and
        # z-layer, stored in the third component - the smart placement has
        # assigned)
        if self.label_visuals is not None:
            for i, l in enumerate(self.label_visuals):
                if l is None:
                    continue
                offset = getattr(l, "_label_offset", None)
                if offset is None:
                    offset = (0.005, 0.0, LABEL_Z)
                l.local.position = (
                    self.positions[i, 0] + offset[0],
                    self.positions[i, 1] + offset[1],
                    offset[2],
                )

        # Keep the connector lines attached while points (and labels) move
        self._sync_label_connector_positions()
        self._sync_group_label_positions()

        # Point positions changed -> current placement (and any in-flight
        # label animation targets) are stale
        self._label_anim_active.clear()
        self._invalidate_label_placement()

        # Update the positions of selected points
        if hasattr(self, "highlight_visuals"):
            for vis in self.highlight_visuals:
                vis.geometry.positions.set_data(xyz[vis._point_ix])
                vis.geometry.positions.update_full()

    def move_points(self, new_positions, n_frames=20):
        """Move the points to new positions.

        Parameters
        ----------
        new_positions : array-like
                        The new positions to move to.
        n_frames :      int, optional
                        The number of frames to move over.
                        Default is 100.

        """
        # Check if the new positions are the same length as the old ones
        if len(new_positions) != len(self.positions):
            raise ValueError(
                f"New positions must be the same length as the old ones. "
                f"Got {len(new_positions)} and {len(self.positions)}."
            )

        # Calculate the vector from new to old positions
        vec = new_positions - self.positions

        # Slice the vector into n_frames segments
        # (we may want to do non-linear interpolation later)
        steps = vec / n_frames

        # Clear the label outlines if they exist
        if hasattr(self, "label_line_group"):
            self.label_line_group.clear()

        # Clear the distance edges if they exist
        if hasattr(self, "distance_edge_group"):
            self.distance_edge_group.clear()

        # Clear the distance edges if they exist
        if hasattr(self, "neighbors_edge_group"):
            self.neighbors_edge_group.clear()

        # Stack n_frame times in a new dimension
        self.to_move = np.repeat(steps.reshape(-1, 2, 1), n_frames, axis=2)

    def process_moves(self):
        """Move the points to the new positions."""
        # Check if we have any points to move
        if (
            not hasattr(self, "to_move")
            or self.to_move is None
            or self.to_move.size == 0
        ):
            return

        # Pop the first step from the `to_move` array
        new_positions = self.positions + self.to_move[:, :, 0]
        self.positions = new_positions
        self.to_move = self.to_move[:, :, 1:]

        # Update the positions of the points
        self.update_point_position()

        # If there are no more steps, remove the `to_move` attribute
        if self.to_move.size == 0:
            del self.to_move
            if self.show_label_lines:
                self.make_label_lines()
            if self.show_distance_edges:
                # The active embedding may no longer carry a distance matrix
                # (e.g. after a switch); turn the overlay off rather than raise.
                if self._distance_edges_drawable():
                    self.make_distance_edges()
                else:
                    self.show_distance_edges = False
            if self.show_knn_edges:
                # The setter degrades gracefully if the stored metric's source
                # is no longer available (see `_knn_edges_drawable`).
                knn_mode = self.show_knn_edges
                self.show_knn_edges = False
                self.show_knn_edges = knn_mode
            self._render_stale = True

    @staticmethod
    def _compute_frame(xy):
        """Return ``(center, scale)`` for an (N, 2) embedding.

        `center` is the bounding-box center; `scale` is a single scalar (the
        larger of the two axis ranges) so normalization stays uniform and
        preserves aspect ratio (and therefore 2D neighbor ordering / fidelity).
        """
        xy = np.asarray(xy, dtype=np.float64)
        # Ignore non-finite rows: a single NaN/inf would otherwise propagate
        # through min/max and turn the whole normalized layout into NaN.
        finite = np.isfinite(xy).all(axis=1)
        if not finite.any():
            return np.zeros(xy.shape[1]), 1.0
        xy = xy[finite]
        lo = xy.min(axis=0)
        hi = xy.max(axis=0)
        center = (lo + hi) / 2.0
        scale = float((hi - lo).max())
        if scale <= 0:
            scale = 1.0
        return center, scale

    def normalize_to_frame(self, xy):
        """Uniformly map `xy` into the shared embedding frame.

        Used so freshly computed embeddings (or alternative embeddings) occupy
        roughly the same region as the active one, keeping `space`/recompute
        morphs in view. Returns `xy` unchanged if no frame has been set.
        """
        xy = np.asarray(xy, dtype=np.float32)
        if self._embedding_frame is None:
            return xy
        frame_center, frame_scale = self._embedding_frame
        own_center, own_scale = self._compute_frame(xy)
        return (
            (xy - own_center) * (frame_scale / own_scale) + frame_center
        ).astype(np.float32)

    def set_embeddings(self, entries, active=0):
        """Register the full list of embedding entries (no animation).

        Parameters
        ----------
        entries : list of dict
            Each dict has keys "name", "embedding" (N, 2), "features" and
            "distances" (either may be None).
        active : int
            Index of the currently displayed embedding. Its layout defines the
            shared normalization frame; all other embeddings are scaled into it.

        Notes
        -----
        The caller must have already shown the active embedding via
        `set_points` (which frames the camera). This only stores the
        collection and normalizes the non-active embeddings into the active
        embedding's frame.
        """
        self.embedding_entries = list(entries) if entries else []

        if not self.embedding_entries:
            self.active_embedding = None
            self._embedding_frame = None
            if self.controls is not None:
                self.controls.update_embedding_selector()
            return

        active = int(active) % len(self.embedding_entries)
        self.active_embedding = active
        self._embedding_frame = self._compute_frame(
            self.embedding_entries[active]["embedding"]
        )

        # Normalize every non-active embedding into the active frame. The
        # active embedding defines the frame, so it is left untouched (this
        # also keeps it exactly equal to what `set_points` is displaying).
        for i, entry in enumerate(self.embedding_entries):
            emb = np.asarray(entry["embedding"], dtype=np.float32)
            if i == active:
                entry["embedding"] = emb
            else:
                entry["embedding"] = self.normalize_to_frame(emb)

        if self.controls is not None:
            self.controls.update_embedding_selector()

    def switch_embedding(self, idx, animate=True):
        """Make embedding `idx` the active one, animating the transition.

        Swaps in the entry's paired features/distances, then moves the points
        to the new (already normalized) layout via `move_points` and notifies
        the controls so fidelity/options refresh.
        """
        if not self.embedding_entries:
            return

        idx = int(idx) % len(self.embedding_entries)
        if idx == self.active_embedding:
            return

        self.active_embedding = idx
        entry = self.embedding_entries[idx]

        # Swap the high-dim sources paired with this embedding.
        feats = entry.get("features")
        dists_mat = entry.get("distances")
        knn = entry.get("knn")
        if feats is None and dists_mat is None and knn is None:
            self.dists = None
        else:
            new_dists = {}
            if dists_mat is not None:
                new_dists["distances"] = dists_mat
            if feats is not None:
                new_dists["features"] = feats
            if knn is not None:
                new_dists["knn"] = knn
            self.dists = new_dists

        new_pos = np.asarray(entry["embedding"], dtype=np.float32)
        if self.positions is not None and len(self.positions) == len(new_pos):
            # Same number of points -> animate (or jump in a single frame).
            self.move_points(new_pos, n_frames=20 if animate else 1)
        else:
            # No current positions (or a mismatch): assign directly.
            self.positions = new_pos
            if self.point_visuals is not None:
                self.update_point_position()

        # Let the user know which embedding is now shown.
        self.show_message(
            f"Embedding: {entry.get('name', f'#{idx + 1}')}",
            position="top-center",
            color="w",
            duration=2,
        )

        if self.controls is not None:
            self.controls.on_embedding_switched()

    def _cycle_embedding(self):
        """Advance to the next embedding (bound to the space key)."""
        if len(self.embedding_entries) > 1 and self.active_embedding is not None:
            self.switch_embedding(self.active_embedding + 1)

    def update_active_embedding_positions(self, xy):
        """Persist recomputed positions into the active entry (supersede).

        `xy` is expected to already be normalized into the shared frame.
        """
        if self.active_embedding is not None and self.embedding_entries:
            self.embedding_entries[self.active_embedding]["embedding"] = np.asarray(
                xy, dtype=np.float32
            )

    @update_figure
    def set_labels(self, indices, new_label):
        """Change the label of given point(s) in the figure.

        Parameters
        ----------
        indices : int or list of int
            Index or list of indices of the points to relabel.
        new_label : str or list of str
            New label or list of new labels to assign.

        """
        if self.labels is None:
            raise ValueError("No labels were provided.")

        if not isinstance(indices, (list, np.ndarray, tuple, set)):
            indices = [indices]

        if isinstance(new_label, str):
            new_label = [new_label] * len(indices)

        assert len(indices) == len(
            new_label
        ), "Number of indices and new labels must match."

        self.labels[indices] = new_label
        for ix, label in zip(indices, new_label):
            if self.label_visuals[ix] is not None:
                self.label_visuals[ix].set_text(label)
                self.label_visuals[ix]._text = label

        # New texts = new label extents and new grouping (N.B. the labels
        # array was mutated in place, so the factorization cache is stale)
        self._label_codes_cache = None
        self._invalidate_label_placement()

    @update_figure
    def set_label_color(self, indices, new_color):
        """Change the color of given point labels in the figure.

        Parameters
        ----------
        indices : int or list of int
            Index or list of indices of the point labels to recolor.
        new_color : str or tuple or list
            Color string/tuple or list of colors to apply.
        """
        if self.labels is None:
            raise ValueError("No labels were provided.")

        if not isinstance(indices, (list, np.ndarray, tuple, set)):
            indices = [indices]

        if isinstance(new_color, (str, tuple)):
            new_color = [new_color] * len(indices)

        assert len(indices) == len(
            new_color
        ), "Number of indices and colors must match."

        if self._label_colors is None:
            self._label_colors = [None] * len(self)

        for ix, color in zip(indices, new_color):
            if ix < 0:
                ix = len(self) + ix
            self._label_colors[ix] = color
            if self.label_visuals is not None and self.label_visuals[ix] is not None:
                self.label_visuals[ix].material.color = color

    def calculate_embedding_fidelity(self, k=10, metric="auto", rank=True, positions=None):
        """Calculate the neighborhood fidelity of the embedding.

        Parameters
        ----------
        positions : (N, 2) array, optional
                    Embedding positions to evaluate. Defaults to the current
                    point positions; pass e.g. the target positions of an
                    ongoing move animation to evaluate those instead.
        """
        if not hasattr(self, "dists") or self.dists is None:
            raise ValueError(
                "No distance matrix or features provided. Cannot calculate embedding fidelity."
            )

        has_dist = "distances" in self.dists and self.dists["distances"] is not None
        has_feat = "features" in self.dists and self.dists["features"] is not None
        has_knn = "knn" in self.dists and self.dists["knn"] is not None
        if not has_dist and not has_feat and not has_knn:
            raise ValueError(
                "Must have a distance matrix, features or a KNN graph to "
                "calculate embedding fidelity."
            )

        if metric == "auto":
            if has_dist:
                metric = "precomputed"
            elif has_knn:
                metric = "knn"
            else:
                metric = "euclidean"

        # A KNN graph supplies the true top-k neighbors directly.
        if metric == "knn":
            if not has_knn:
                raise ValueError("KNN metric requires a KNN graph.")
            graph = self.dists["knn"]
            return neighborhood_fidelity(
                embedding=positions if positions is not None else self.positions,
                knn_neighbors=graph.indices,
                k=min(int(k), int(graph.k)),
                rank=rank,
            )

        if metric == "precomputed" and not has_dist:
            raise ValueError("Precomputed metric requires a distance matrix.")
        elif metric != "precomputed" and not has_feat:
            raise ValueError(f"Metric '{metric}' requires features.")

        # Use precomputed distance if requested or metric not specified
        if metric == "precomputed":
            dists = self.dists["distances"]
            features = None
        else:
            dists = None
            # Fails loudly (with counts) if the features contain NaN/inf.
            features = self._get_features_checked("neighborhood fidelity")

        return neighborhood_fidelity(
            embedding=positions if positions is not None else self.positions,
            distances=dists,
            features=features,
            k=k,
            metric=metric,
            rank=rank,
        )

    def to_plotly(
        self,
        include_selected=True,
        selected_size=10,
        unselected_opacity=0.85,
        marker_line_width=0,
    ):
        """Export the current 2D embedding to an interactive Plotly figure.

        Parameters
        ----------
        include_selected : bool, optional
            If True, selected points are rendered as a separate highlighted trace.
        selected_size : int, optional
            Marker size used for the selected-points highlight trace.
        unselected_opacity : float, optional
            Opacity of the main scatter trace.
        marker_line_width : float, optional
            Outline width for markers in both traces.

        Returns
        -------
        plotly.graph_objects.Figure
            Interactive figure containing the current embedding.
        """
        from .plotly_export import scatter_to_plotly_figure

        return scatter_to_plotly_figure(
            self,
            include_selected=include_selected,
            selected_size=selected_size,
            unselected_opacity=unselected_opacity,
            marker_line_width=marker_line_width,
        )

    def to_plotly_dashboard(self, top_n=20, include_selected=True):
        """Export a compact multi-panel Plotly dashboard figure."""
        from .plotly_export import scatter_to_dashboard_figure

        return scatter_to_dashboard_figure(
            self,
            top_n=top_n,
            include_selected=include_selected,
        )

    def write_plotly_dashboard_html(
        self,
        file_path,
        top_n=20,
        include_selected=True,
        include_plotlyjs="cdn",
    ):
        """Write a standalone multi-panel Plotly dashboard HTML file."""
        from .plotly_export import write_scatter_dashboard_html

        return write_scatter_dashboard_html(
            self,
            file_path=file_path,
            top_n=top_n,
            include_selected=include_selected,
            include_plotlyjs=include_plotlyjs,
        )


class LabelSearch:
    """Class to search for and iterate over point labels.

    Parameters
    ----------
    scatter : Scatterplot
        The plot to search in.
    query : str
        The label to search for.
    rotate : bool, optional
        Whether to rotate through all occurrences of the label.
    go_to_first : bool, optional
        Whether to go to the first occurrence of the label at
        initialization.
    regex : bool, optional
        Whether to interpret the label as a regular expression.

    """

    def __init__(self, scatter, query, rotate=True, go_to_first=True, regex=False):
        self.scatter = scatter
        self.query = query
        self.ix = None  # start with no index (will be initialized in `next` or `prev`)
        self.regex = regex
        self.rotate = rotate

        if scatter._labels is None:
            print("No labels available.")
            return

        # Search labels
        self.indices = self.search_labels(query)

        # Search IDs if no labels were found
        if len(self.indices) == 0:
            if isinstance(query, Number) or query.isdigit():
                self.indices = self.search_ids(int(query))
            elif "," in query:
                # Comma-separated list of IDs. Ignore empty parts so a trailing
                # comma (e.g. "1, 2, 3,") still matches.
                parts = [q.strip() for q in query.split(",") if q.strip()]
                if parts and all(p.isdigit() for p in parts):
                    self.indices = self.search_ids([int(p) for p in parts])

        # If still no labels found, return
        if len(self.indices) == 0:
            print(f"Label '{query}' not found.")
            return

        # Start at the first label
        if go_to_first:
            self.next()

    def __len__(self):
        return len(self.indices)

    def search_labels(self, query):
        """Search for a label in the scatter."""
        if not self.regex:
            return np.where(
                (self.scatter._labels == query) | (self.scatter._labels == str(query))
            )[0]
        else:
            return np.where(
                [re.search(str(query), str(l)) is not None for l in self.scatter.labels]
            )[0]

    def search_ids(self, id):
        """Search for an ID in the scatter."""
        if isinstance(id, (int, np.integer)):
            return np.where(self.scatter.ids == id)[0]
        else:
            return np.where(np.isin(self.scatter.ids, id))[0]

    def select_all(self, add=False):
        """Select all labels found by the search."""
        if add and self.scatter.selected is not None:
            self.scatter.selected = np.union1d(self.scatter.selected, self.indices)
        else:
            self.scatter.selected = self.indices

    def next(self):
        """Go to the next label."""
        if self.ix is None:
            self.ix = 0
        elif self.ix >= (len(self.indices) - 1):
            if not self.rotate:
                raise StopIteration
            else:
                self.ix = 0
        else:
            self.ix += 1

        self.scatter.camera.local.x = self.scatter.positions[self.indices[self.ix], 0]
        self.scatter.camera.local.y = self.scatter.positions[self.indices[self.ix], 1]

        self.scatter._render_stale = True

    def prev(self):
        """Go to the previous label."""
        if self.ix is None:
            self.ix = len(self.indices) - 1
        elif self.ix <= 0:
            if not self.rotate:
                raise StopIteration
            else:
                self.ix = len(self.indices) - 1
        else:
            self.ix -= 1

        self.scatter.camera.local.x = self.scatter.positions[self.indices[self.ix], 0]
        self.scatter.camera.local.y = self.scatter.positions[self.indices[self.ix], 1]

        self.scatter._render_stale = True


def _round_corners(left, right, top, bottom, radius, num_points=10):
    """Round the corners of a bounding box defined by left, right, top and bottom coordinates.

    Returns
    -------
    numpy array defining the positions of the vertices of the rounded bounding box, inclusive of the last vertex which is the same as the first vertex to close the loop.
    """
    # If the rectangle is much larger than it is wider or vice versa, we get corners that are elongated along the longer axis.
    # To prevent this, we will scale the radius for x- and y- axis separately
    ratio = (max(left, right) - min(left, right)) / (
        max(top, bottom) - min(top, bottom)
    )

    radius_y = radius / ratio if ratio > 1 else radius
    radius_x = radius * ratio if ratio < 1 else radius

    # Define the center points of the rounded corners
    corner_centers = np.array(
        [
            [left + radius_x, top - radius_y, 0],
            [right - radius_x, top - radius_y, 0],
            [right - radius_x, bottom + radius_y, 0],
            [left + radius_x, bottom + radius_y, 0],
        ],
        dtype=np.float32,
    )

    # Generate the vertices for the rounded corners
    vertices = []
    for i in range(4):
        start_angle = i * np.pi / 2
        end_angle = start_angle + np.pi / 2
        angles = np.linspace(start_angle, end_angle, num=num_points)
        for angle in angles:
            vertex = corner_centers[i] + np.array(
                [radius_x * np.cos(angle), radius_y * np.sin(angle), 0],
                dtype=np.float32,
            )
            vertices.append(vertex)
    vertices.append(vertices[0])

    return np.array(vertices, dtype=np.float32)
