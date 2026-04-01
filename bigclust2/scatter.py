import re
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
        self.label_vis_limit = 200  # number of labels shown at once before hiding all
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

        # This group will hold text labels that need to move but not scale with the dendrogram
        self.text_group = gfx.Group()
        self.scene.add(self.text_group)

        # Generate the a container for labels
        self.label_group = gfx.Group()
        self.label_group.visible = True
        self.text_group.add(self.label_group)

        # Add some keyboard shortcuts for moving and scaling the dendrogam
        def move_camera(x, y):
            self.camera.world.x += x
            self.camera.world.y += y
            self._render_stale = True
            self.canvas.request_draw()

        self.key_events["ArrowLeft"] = lambda: setattr(
            self, "font_size", max(self.font_size - 1, 1)
        )
        self.key_events["ArrowRight"] = lambda: setattr(
            self, "font_size", self.font_size + 1
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

    def clear(self):
        """Clear contents of the scatter plot."""
        for vis in ("label_group", "scatter_group"):
            vis = getattr(self, vis, None)
            if vis is not None:
                vis.clear()

        self.labels = None
        self.label_visuals = None
        self.point_visuals = None
        self.positions = None
        self.metadata = None
        self._selected = None

    @property
    def labels(self):
        """Return the labels of leafs in the dendrogram."""
        return self._labels

    @labels.setter
    @update_figure
    def labels(self, x):
        """Set the labels of leafs in the dendrogram."""
        if x is None:
            self._labels = None
            self._label_visuals = None
            return
        assert len(x) == len(self), "Number of labels must match number of leafs."
        self._labels = np.asarray(x)
        self.update_point_labels()  # updates the visuals

    @property
    def font_size(self):
        return int(self._font_size * 100)

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
        """Select given points in the plot."""
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
                self.ngl_viewer.show(
                    self.ids[self.selected],
                    datasets=self.datasets[self._selected],
                    add_as_group=getattr(self, "_add_as_group", False),
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
                            self.ids[self.selected],
                            datasets=self.datasets[self.selected],
                        )
                    else:
                        func(self.ids[self.selected])
                except BaseException as e:
                    print(f"Failed to sync widget {w}:\n", e)

        if self.show_knn_edges and self.show_knn_edges["mode"] == "selected":
            self.show_knn_edges = (
                self._show_knn_edges
            )  # trigger an update of the KNN edges

    @property
    def selected_ids(self):
        """Return the IDs of selected leafs in the dendrogram."""
        if self.selected is None or not len(self.selected):
            return None
        if self.ids is None:
            raise ValueError("No IDs were provided.")
        return self.ids[self.selected]

    @property
    def selected_labels(self):
        """Return the labels of selected leafs in the dendrogram."""
        if self.selected is None or not len(self.selected):
            return None
        if self.labels is None:
            raise ValueError("No labels were provided.")
        return self.labels[self.selected]

    @property
    def selected_meta(self):
        """Return the metadata of selected leafs in the dendrogram."""
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

        if x == self._distance_edges_threshold:
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
            ), "Selection mask must be the same length as the dendrogram."
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
            markers = np.full(len(self), "circle")
        else:
            assert len(self.markers) == len(
                self
            ), "Length of leaf_types must match length of dendrogram."
            unique_types = np.unique(self.markers)

            assert len(unique_types) <= len(
                AVAILABLE_MARKERS
            ), "Only 10 unique types are supported."
            marker_map = dict(zip(unique_types, AVAILABLE_MARKERS))
            markers = np.array([marker_map[t] for t in self.markers])

        # Create the visuals
        for m in np.unique(markers):
            color = "w"
            if mask is None:
                ix = np.where(markers == m)[0]
                this_meta = self.metadata.iloc[ix]
                this_pos = self.positions[ix]
                if self.colors is not None:
                    color = np.array(
                        [tuple(cmap.Color(c).rgba) for c in self.colors[markers == m]]
                    )
                this_size = self.point_size
            else:
                this_meta = self.metadata.iloc[mask & (markers == m)]
                this_pos = self.positions[mask & (markers == m)]
                ix = np.where(mask & (markers == m))[0]
                if self.colors is not None:
                    color = np.array(
                        [
                            tuple(cmap.Color(c).rgba)
                            for c in self.colors[mask & (markers == m)]
                        ]
                    )
                if isinstance(self.point_size, (int, float)):
                    this_size = self.point_size
                else:
                    this_size = self.point_size[mask & (markers == m)]
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

    def make_hover_widget(self, color=(1, 1, 1, 0.5), font_color=(0, 0, 0, 1)):
        """Generate a widget for hover info.

        The widget will be added to the overlay scene which uses a NDC camera
        which means the coordinates will be in the range [-1, 1] regardless of
        the actual size of the scene/window.

        """
        widget = gfx.Group()
        widget.visible = False
        width = 2 / 6
        left = 1 - width / 2
        widget.local.position = (left, 0, 0)

        widget.add(
            gfx.Mesh(
                gfx.plane_geometry(width, 2),  # full screen height, 1/6 width
                gfx.MeshBasicMaterial(color=color, alpha_mode="blend"),
            )
        )
        widget.add(
            text2gfx(
                "Hover info",
                position=(-(width / 2) + 0.025, 0, 0),
                color=font_color,
                font_size=12,  # fix font size
                anchor="middle-left",
                screen_space=True,  # without this the text would be scewed
            )
        )

        return widget

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
                t = text2gfx(
                    str(self.labels[ix]),
                    position=(
                        self.positions[ix, 0] + 0.005,
                        self.positions[ix, 1],
                        0,
                    ),
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
            self.show_message(f"Found {len(ls)} occurrences of '{label}'", duration=3)

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
        self.label_visuals = [None] * len(metadata) if label_col else None
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
            self.dists["distances"] = distances.loc[self.ids, self.ids]

        if features is not None:
            self.dists["features"] = features.loc[self.ids]

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
                # Note: we could use e.g. shift-hover to show
                # more/different info?
                if event.type == "pointer_enter":
                    # Translate position to world coordinates
                    pos = self.screen_to_world((event.x, event.y))

                    # Find the closest leaf
                    vis = event.current_target
                    coords = vis.geometry.positions.data
                    dist = np.linalg.norm(coords[:, :2] - pos[:2], axis=1)
                    closest = np.argmin(dist)
                    point_ix = vis._point_ix[closest]

                    # Position and show the hover widget
                    # self._hover_widget.local.position = coords[closest]
                    self.hover_widget.visible = True

                    # Set the text
                    # N.B. there is some funny behaviour where repeatedly setting the same
                    # text will cause the bounding box to increase every time. To avoid this
                    # we have to reset the text to anything but an empty string.
                    self.hover_widget.children[1].set_text("asdfgasdfasdfsdafsfasdfasg")
                    self.hover_widget.children[1].set_text(str(hover_info[point_ix]))

                    # Scale the background to fit the text
                    # bb = self._hover_widget.children[1].get_world_bounding_box()
                    # extent = bb[1] - bb[0]

                    # The text bounding box is currently not very accurate. For example,
                    # a single-line text has no height. Hence, we need to add some padding:
                    # extent = (extent + [0, 1.2, 0]) * 1.2
                    # self._hover_widget.children[0].local.scale_x = extent[0]
                    # self._hover_widget.children[0].local.scale_y = extent[1]

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
        """Sync the dendrogram with a neuroglancer viewer."""
        self.ngl_viewer = viewer

        # Activate the neuroglancer controls tab
        if hasattr(self, "controls"):
            self.controls.tabs.setTabEnabled(2, True)

    def sync_widget(self, widget, callback=None):
        """Connect a widget to the dendrogram.

        Parameters
        ----------
        widget
                The widget to sync.
        callback
                The function to call. If `None`, the widget must implement a
                `.select()` method that takes a list of IDs to select.
                If either method accepts a `datasets` parameter, the dataset for
                each ID will also be passed to the method.

        """
        if callback is None:
            assert hasattr(widget, "select") and callable(
                widget.select
            ), "Widget must have a `select` method that takes a list of IDs to select."
            callback = widget.select

        if not hasattr(self, "synced_widgets"):
            self.synced_widgets = []

        self.synced_widgets.append((widget, callback))

    def set_colors(self, colors):
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

        """
        if not isinstance(colors, (list, np.ndarray)):
            raise ValueError(f"Expected list or array, got {type(colors)}.")

        if len(colors) != len(self):
            raise ValueError(f"Expected {len(self)} colors, got {len(colors)}.")

        if isinstance(colors, list) and isinstance(colors[0], str):
            colors = np.array([tuple(cmap.Color(c).rgba) for c in colors])
        elif isinstance(colors, np.ndarray) and colors.shape[1] == 3:
            # Add an alpha channel if not provided
            colors = np.hstack((colors, np.ones((len(colors), 1))))

        self.colors = colors.astype(np.float32)

        for vis in self.point_visuals:
            vis.geometry.colors.set_data(self.colors[vis._point_ix])
            vis.geometry.colors.update_full()

    def set_viewer_colors(self, colors):
        """Set the colors for the neuroglancer viewer.

        Parameters
        ----------
        colors :    dict
                    Dictionary of colors keyed by IDs: {id: color, ...}
        """
        if not hasattr(self, "ngl_viewer"):
            raise ValueError("No neuroglancer viewer is connected.")

        assert isinstance(colors, dict), "Colors must be a dictionary."
        self.ngl_viewer.set_colors(colors)

    def set_viewer_color_mode(self, mode, palette="vispy:husl"):
        """Set the color mode for the neuroglancer viewer.

        Parameters
        ----------
        mode :  "dataset" | "cluster" | "label" | "default"
                The color mode to use.

        """
        if not hasattr(self, "ngl_viewer"):
            raise ValueError("No neuroglancer viewer connected to this figure.")

        if mode == "cluster":
            # Collect colors for each leaf
            colors = {}
            for vis in self.point_visuals:
                this_ids = self.ids[vis._point_ix]

                if self.datasets is None:
                    colors.update(zip(this_ids, vis.geometry.colors.data))
                else:
                    this_datasets = self.datasets[vis._point_ix]
                    colors.update(
                        zip(
                            zip(this_ids, this_datasets),
                            vis.geometry.colors.data,
                        )
                    )
        elif mode == "label":
            labels_unique = np.unique(self.labels.astype(str))

            # To avoid similar labels getting a similar color we will jumble the labels
            rng = np.random.default_rng(seed=42)
            rng.shuffle(labels_unique)

            palette = cmap.Colormap(palette)
            colormap = {
                l: c.hex
                for l, c in zip(labels_unique, palette.iter_colors(len(labels_unique)))
            }
            if self.datasets is None:
                colors = {i: colormap[str(l)] for i, l in zip(self.ids, self.labels)}
            else:
                colors = {
                    (i, d): colormap[str(l)]
                    for i, l, d in zip(self.ids, self.labels, self.datasets)
                }
        elif mode == "dataset":
            palette = cmap.Colormap(palette)
            colormap = {
                i: c.hex
                for i, c in zip(
                    range(len(self.point_visuals)),
                    palette.iter_colors(len(self.point_visuals)),
                )
            }
            colors = {}
            for i, vis in enumerate(self.point_visuals):
                this_ids = self.ids[vis._point_ix]
                this_c = colormap[i]
                if self.datasets is None:
                    colors.update({i: this_c for this_id in this_ids})
                else:
                    this_datasets = self.datasets[vis._point_ix]
                    colors.update(
                        {
                            (this_id, this_dataset): this_c
                            for this_id, this_dataset in zip(this_ids, this_datasets)
                        }
                    )
        elif mode == "default":
            self.ngl_viewer.set_default_colors()
            return
        else:
            raise ValueError(
                f"Unknown color mode '{mode}'. "
                f"Expected 'dataset', 'cluster', 'label' or 'default'."
            )

        self.set_viewer_colors(colors)

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
        return np.where(self.scatter.ids == id)[0]

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
