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
from .selection import SelectionGizmo
from .visuals import points2gfx, text2gfx, lines2gfx

AVAILABLE_MARKERS = list(gfx.MarkerShape)
# Drop markers which look too similar to others
AVAILABLE_MARKERS.remove("ring")


class ScatterFigure(BaseFigure):
    """A 3D scatter plot figure for visualizing point clouds with metadata."""

    selection_color = "y"
    distance_edges_threshold_default = 0.5

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Some internal state
        self.labels = None
        self.label_visuals = None
        self._label_colors = None
        self.point_visuals = None
        self.positions = None
        self.metadata = None
        self._selected = None
        self.deselect_on_empty = (
            False  # whether to deselect everything on empty selection
        )
        self.deselect_on_dclick = (
            False  # whether to deselect everything on double click
        )
        self._font_size = 0.01
        self._points_scale = 1  # used for scaling points uniformly
        self.label_vis_limit = 400  # number of labels shown at once before hiding all
        self.label_refresh_rate = 30  # update labels every n frames

        # Add the selection gizmo
        self.selection_gizmo = SelectionGizmo(
            self.renderer,
            self.camera,
            self.scene,
            callback_after=lambda x: self.select_points(
                x.bounds, additive="Control" in x._event_modifiers
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
            self, "point_scale", max(self.point_scale * 0.9, 0.01)
        )
        self.key_events["ArrowUp"] = lambda: setattr(
            self, "point_scale", self.point_scale * 1.1
        )
        self.key_events["Escape"] = lambda: self.deselect_all()
        self.key_events["l"] = lambda: self.toggle_labels()

        def _toggle_last_label():
            """Toggle between the last label and the original labels."""
            if hasattr(self, "controls"):
                self.controls.switch_labels()

        self.key_events["m"] = _toggle_last_label

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

            # If more than the limit, don't show any labels
            if iv.sum() > self.label_vis_limit:
                for i, t in enumerate(self.label_group.children):
                    t.visible = False
            else:
                self.show_labels(np.where(iv)[0])
                self.hide_labels(np.where(~iv)[0])

        # Turns out this is too slow to be run every frame - we're throttling it to every N frames
        self.control_label_vis_tick = 1
        self.add_animation(_control_label_vis)

        self.add_animation(self.process_moves)

    def __len__(self):
        return len(self.positions) if self.positions is not None else 0

    def center_camera(self):
        """Center the camera on the scatter plot."""
        if self.positions is None:
            return

        self.camera.show_object(self.scatter_group)

    def clear(self):
        """Clear contents of the scatter plot."""
        for vis in ("label_group", "scatter_group"):
            vis = getattr(self, vis, None)
            if vis is not None:
                vis.clear()

        self.labels = None
        self.label_visuals = None
        self._label_colors = None
        self.point_visuals = None
        self.positions = None
        self.metadata = None
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

    @property
    def point_scale(self):
        """Uniform scale factor for points."""
        return self._point_scale

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

        # Restrict selections if applicable
        if hasattr(self, "_restrict_selection"):
            self._selected = self._selected[
                np.isin(self.datasets[self._selected], self._restrict_selection)
            ]

        # Create the new selection visuals
        self.highlight_points(self._selected, color=self.selection_color)

        # Update the controls
        # if hasattr(self, "_controls"):
        #     self._controls.update_ann_combo_box()

        if hasattr(self, "ngl_viewer"):
            if len(self._selected) > 0:
                if not getattr(self, "_add_as_group", False):
                    group_name = None
                else:
                    # Find a sensible name for the group based on selected labels
                    selected_labels = list(set(self.selected_labels))
                    if len(selected_labels) == 1:
                        group_name = selected_labels[0]
                    elif len(selected_labels) <= 3:
                        group_name = "+".join(selected_labels)
                    else:
                        group_name = "mixed"

                    # But we also want to make sure that if there is already a group
                    # with this name, we don't add to it but rather make a new group
                    # with a unique name.
                    i = 1
                    final_group_name = group_name
                    while final_group_name in self.ngl_viewer.viewer.objects:
                        final_group_name = group_name + f" - #{i}"

                self.ngl_viewer.show(
                    self.ids[self.selected],
                    datasets=self.datasets[self._selected],
                    add_as_group=final_group_name,
                )
            else:
                self.ngl_viewer.clear()

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

    def open_selection_in_new_window(self, ids=None, ind=None):
        """Open the current or given selection in a new window.

        Parameters
        ----------
        ids : array-like of int or str, optional
            IDs of the leafs to open in a new window. If None, the currently
            selected leafs will be used.
        ind : array-like of int, optional
            Indices of the leafs to open in a new window. If None, the currently
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
            window, "on_open_selection_in_new_window"
        ):
            window = window.parent()

        if window is None:
            raise RuntimeError(
                "Unable to find parent window to open selection in a new window."
            )

        window.on_open_selection_in_new_window()

        self.selected = (
            curr_sel  # restore the current selection after opening the new window
        )

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
            self.neighbors_edge_group.visible = False
        else:
            if x["mode"] == "selected":
                mask = np.zeros(len(self), dtype=bool)
                mask[self.selected] = True
            else:
                mask = None

            k = x.get("k", 15)
            metric = x.get("metric", "auto")
            color = x.get("color", (1, 1, 1, 0.1))
            linewidth = x.get("linewidth", 1)
            self.make_neighbour_edges(
                k=k, metric=metric, mask=mask, color=color, linewidth=linewidth
            )

        self._show_knn_edges = x

    @property
    def fidelity_mode(self):
        """Show or hide the neighborhood fidelity."""
        if not hasattr(self, "_fidelity_mode"):
            return False
        return self._fidelity_mode

    @fidelity_mode.setter
    @update_figure
    def fidelity_mode(self, x):
        # No need to change anything
        if x == self.fidelity_mode:
            return

        if not x:
            self._fidelity_mode = False
            self.point_size = 1  # reset to default point size
            return

        if not isinstance(x, dict):
            x = {"mode": x}

        # If dictionary, parse parameters
        mode = x.get("mode", "point_size")
        rank = x.get("rank", False)
        distance = x.get("distance", "euclidean")
        k = x.get("k", 15)

        assert mode in ("point_size",), f"Unsupported fidelity mode: {mode}"

        # Calculate fidelity scores
        self.point_size = self.calculate_embedding_fidelity(
            k=k, rank=rank, metric=distance
        )

        # Make sure no point vanishes entirely
        if np.any(self.point_size == 0):
            self.point_size += 1e-2

        self._fidelity_mode = x

    def deselect_all(self):
        self.selected = None

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
                vis.material.edge_color = "yellow"
                vis.material.edge_width = 2
                vis.material.color = (1, 1, 1, 0)
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
                pick_write=self.hover_info is not None,
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

        # Get the distances
        if isinstance(self.dists, dict):
            if hasattr(self, "controls"):
                dists = self.dists[self.controls.umap_dist_combo_box.currentText()]
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
            else:
                metric = "euclidean"

        if metric == "precomputed" and "distances" not in self.dists:
            raise ValueError("No precomputed distances available.")

        if metric != "precomputed" and "features" not in self.dists:
            raise ValueError("No feature vectors available for distance calculation.")

        # Create a group and add to scene
        if not getattr(self, "neighbors_edge_group", None):
            self.neighbors_edge_group = gfx.Group()
            self.scene.add(self.neighbors_edge_group)

        # Clear the group (we might call this function to update the lines)
        self.neighbors_edge_group.clear()

        # If nothing to show, we can just return here
        if mask is not None and mask.sum() == 0:
            return

        # At this point we should have completed all checks, so we can safely grab the required data
        if metric == "precomputed":
            knn = np.argsort(self.dists["distances"], axis=1)[:, 1 : (k + 1)]
        else:
            from sklearn.neighbors import NearestNeighbors

            _, knn = (
                NearestNeighbors(n_neighbors=k, metric=metric)
                .fit(self.dists["features"])
                .kneighbors()
            )

        ind = np.arange(len(knn))
        if mask is not None:
            knn = knn[mask]
            ind = ind[mask]

        # Convert to edges (i.e. pairs of points)
        edges = []
        for i in range(k):
            edges.append(np.stack((ind, knn[:, i]), axis=1))
        edges = np.concatenate(edges, axis=0)

        # Find unique edges (since the same edge can be found from both directions)
        edges = np.sort(edges, axis=1)  # sort each edge to make them comparable
        edges = np.unique(edges, axis=0)

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
                        0,
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
                f"Found {len(ls)} occurrences of '{label}'", duration=3, color="g"
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

        # Return here if we're only clearing the highlights
        if x is None:
            return

        if isinstance(x, str):
            for i, label in enumerate(self.labels):
                if label != x:
                    continue

                if self.label_visuals[i] is None:
                    self.show_labels(i)
                visual = self.label_visuals[i]

                visual.material._original_color = visual.material.color
                visual.material.color = color
        elif isinstance(x, (list, np.ndarray)):
            for ix in x:
                # Index in the original order
                if self.label_visuals[ix] is None:
                    self.show_labels(ix)
                visual = self.label_visuals[ix]

                visual.material._original_color = visual.material.color
                visual.material.color = color
        else:
            raise ValueError(f"Expected str or list, got {type(x)}.")

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
        point_size=10,
        distances=None,
        features=None,
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
            Size of the points, by default 10
        distances/features : np.ndarray, optional
            An (N, N) array of pairwise distances between points, or an (N, M) array of features.
            If provided, these can be used to re-compute point positions based on dimensionality reduction techniques.
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
        self.labels = metadata[label_col].astype(str).values if label_col else None
        self.ids = metadata[id_col].values if id_col else None
        self.colors = metadata[color_col].values if color_col else None
        self.markers = metadata[marker_col].values if marker_col else None

        # Make sure distances and features have the same order as the points
        # N.B. we're tracking both distances and features in a dictionary
        # to allow for multiple distance metrics
        if distances is None and features is None:
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

        if features is not None:
            if features.shape[0] != len(metadata):
                raise ValueError(
                    f"Number of rows in features must match number of points, got {features.shape[0]} and {len(metadata)}."
                )
            if not np.all(features.index == self.ids):
                raise ValueError("Index of features must match IDs in metadata.")

            self.dists["features"] = features

        # Datasets are used to avoid collisions when the same ID is used in different datasets
        self.datasets = self.metadata[dataset_col].values if dataset_col else None

        self.hover_info_org = hover_col  # keep the original, unprocessed hover info
        if hover_col is not None:
            if "{" in hover_col:
                hover_info = metadata.apply(hover_col.format_map, axis=1)
            else:
                hover_info = metadata[hover_col]
        self.hover_info = np.asarray(hover_info) if hover_col else None

        # Update some internal state
        # (note that we're writing to the protected member variables here)
        self._selected = None
        self._point_size = 1
        self._point_scale = point_size

        # Generate the visuals
        self.make_visuals()

        # Setup hover info
        if hover_info is not None:

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
                        hover_info[point_ix],
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

        # Activate the neuroglancer controls tab
        if hasattr(self, "controls"):
            self.controls.tabs.setTabEnabled(2, True)

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
            vis.geometry.colors.set_data(self.colors[vis._point_ix])
            vis.geometry.colors.update_full()

        if sync_to_viewer and hasattr(self, "ngl_viewer"):
            self.set_viewer_colors(colors)

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

        # Update the positions of the labels
        for i, l in enumerate(self.label_visuals):
            if l is None:
                continue
            l.local.position = (
                self.positions[i, 0] + 0.005,
                self.positions[i, 1],
                0,
            )

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
                self.make_distance_edges()
            if self.fidelity_mode:
                fid_mode = self.fidelity_mode
                self.fidelity_mode = None
                self.fidelity_mode = fid_mode
            if self.show_knn_edges:
                knn_mode = self.show_knn_edges
                self.show_knn_edges = False
                self.show_knn_edges = knn_mode
            self._render_stale = True

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

    def calculate_embedding_fidelity(self, k=10, metric="auto", rank=True):
        """Calculate the neighborhood fidelity of the embedding."""
        if not hasattr(self, "dists") or self.dists is None:
            raise ValueError(
                "No distance matrix or features provided. Cannot calculate embedding fidelity."
            )

        has_dist = "distances" in self.dists and self.dists["distances"] is not None
        has_feat = "features" in self.dists and self.dists["features"] is not None
        if not (has_dist) and not (has_feat):
            raise ValueError(
                "Must have either distance matrix or features to calculate embedding fidelity."
            )

        if metric == "auto":
            metric = "procomputed" if has_dist else "euclidean"

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
            features = self.dists["features"]

        return neighborhood_fidelity(
            embedding=self.positions,
            distances=dists,
            features=features,
            k=k,
            metric=metric,
            rank=rank,
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
            elif "," in query and all([q.strip().isdigit() for q in query.split(",")]):
                self.indices = self.search_ids(
                    [int(q.strip()) for q in query.split(",")]
                )

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
