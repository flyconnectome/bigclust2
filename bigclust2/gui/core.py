import sys
import cmap
import csv
import io
import itertools
import json
import logging
from datetime import datetime
from html import escape

import numpy as np
import pandas as pd

from pathlib import Path
from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QMenu,
    QWidget,
    QScrollArea,
    QVBoxLayout,
    QHBoxLayout,
    QSplitter,
    QPushButton,
    QFrame,
    QLabel,
    QDialog,
    QPlainTextEdit,
    QComboBox,
    QSizePolicy,
    QProgressDialog,
    QWidgetAction,
    QFileDialog,
    QCheckBox,
    QDialogButtonBox,
)
from PySide6.QtGui import QIcon, QAction, QActionGroup, QKeySequence, QShortcut, QDesktopServices, QPainter, QPainterPath, QColor, QPen, QBrush, QPixmap
from PySide6.QtCore import Qt, QSize, QSettings, QPoint, QTimer, QEvent, Signal, QUrl, QRectF, QPointF
from importlib.resources import files

from .loaders import OpenProjectDialog
from .tabs import ViewTabWidget
from ..data import parse_directory, SingleProjectLoader
from .controls import ScatterControls
from .widgets.connectivity import ConnectivityTable
from .widgets.distances import DistancesTable
from .widgets.features import FeatureExplorerWidget
from .widgets.meta_explorer import MetaExplorerDialog
from .widgets.project_details import ProjectDetailsDialog
from .widgets.annotations import AnnotationDialog, SelectionRecord
from ..scatter import ScatterFigure
from ..neuroglancer import NglViewer
from ..embeddings import KNNGraph
from ..__version__ import __version__


__all__ = ["MainWindow", "MainWidget", "main"]


logger = logging.getLogger(__name__)


def _subset_knn(graph, selected_indices):
    """Subset a KNNGraph to a selection, remapping neighbor row positions.

    Neighbor positions are translated from the old (full) index space to the
    new (selection) index space; neighbors that fall outside the selection (or
    were already missing) become the sentinel ``-1`` and are left-compacted to
    the end of each row so valid neighbors stay first. ``k`` is preserved.

    The distance of a dropped slot is *retained* (not zeroed): edge-building
    consumers mask by index, but UMAP's local-scale estimate benefits from
    knowing how far the now-absent neighbor was (see ``KNNGraph``).
    """
    sel = np.asarray(selected_indices, dtype=np.int64)
    n_old = len(graph)

    # old position -> new position (-1 for rows not in the selection)
    remap = np.full(n_old, -1, dtype=np.int64)
    remap[sel] = np.arange(sel.shape[0])

    sub_idx = graph.indices[sel]
    sub_dist = graph.dists[sel]

    valid = sub_idx >= 0
    # Look up remapped positions; invalid/dropped neighbors -> -1.
    mapped = np.where(valid, remap[np.where(valid, sub_idx, 0)], -1)

    # Left-compact so valid neighbors come first; keep distances aligned.
    order = np.argsort(mapped < 0, axis=1, kind="stable")
    mapped = np.take_along_axis(mapped, order, axis=1)
    sub_dist = np.take_along_axis(sub_dist, order, axis=1).astype(np.float64, copy=True)

    return KNNGraph(
        indices=mapped.astype(np.int64),
        dists=sub_dist,
        ids=np.asarray(graph.ids)[sel],
        k=int(graph.k),
    )


# Ask for confirmation before Select All / Invert Selection grabs more
# neurons than this.
LARGE_SELECTION_THRESHOLD = 5000

# Keep strong references to top-level windows so spawned windows stay alive.
_OPEN_WINDOWS = []

# Accent colors assigned to tabs/views so users can match auxiliary widgets to
# their tab: a painted dot in the tab plus a matching circle character in
# window titles (native macOS title bars can't show icons).
_TAB_ACCENTS = [
    ("#e74c3c", "\U0001f534"),  # red
    ("#3498db", "\U0001f535"),  # blue
    ("#2ecc71", "\U0001f7e2"),  # green
    ("#e67e22", "\U0001f7e0"),  # orange
    ("#9b59b6", "\U0001f7e3"),  # purple
    ("#f1c40f", "\U0001f7e1"),  # yellow
    ("#8d6e63", "\U0001f7e4"),  # brown
]
_NEXT_TAB_ACCENT = itertools.count()


def _make_dot_icon(color, size=10):
    """Create a round single-color icon (used as per-tab accent dot)."""
    pixmap = QPixmap(size, size)
    pixmap.fill(Qt.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.Antialiasing)
    painter.setBrush(QBrush(QColor(color)))
    painter.setPen(Qt.NoPen)
    painter.drawEllipse(0, 0, size, size)
    painter.end()
    return QIcon(pixmap)


def _dismiss_active_popup():
    """Close any active popup window (e.g. a completer dropdown).

    Tearing a widget out from under its open popup (closing or moving its
    tab) leaves a stale entry in Qt's application-wide popup stack; key
    events are then redirected to the dead popup, i.e. swallowed, until a
    mouse click dismisses it.
    """
    popup = QApplication.activePopupWidget()
    if popup is not None:
        popup.close()

try:
    ASSETS_FILE_PATH = files("bigclust2") / "assets"
except ModuleNotFoundError:
    ASSETS_FILE_PATH = Path(__file__).parent.parent / "assets"


def resize_figures(func):
    """Decorator to resize figures after the wrapped function is called."""

    def wrapper(self, *args, **kwargs):
        result = func(self, *args, **kwargs)
        self.resize_figures()
        return result

    return wrapper


class MainWidget(QWidget):
    """A single view into a project: scatter figure, 3D viewer and per-view state.

    One instance per tab in the main window. Owns the project data and any
    auxiliary widgets (connectivity table, explorers, ...) opened for it.
    """

    def __init__(self, selection_counter=None):
        super().__init__()
        self._teardown_done = False
        if selection_counter is None:
            selection_counter = QLabel("Selected: N/A  ")
        self._selection_counter = selection_counter

        # Per-view project state
        self._data = None
        self._current_project_loader = None
        self.view_title = "untitled"
        # Accent color identifying this view across tab + widget titles;
        # assigned by MainWindow.add_new_tab and kept for life (incl. detach).
        self.accent_color = None
        self.accent_dot = ""

        # Per-view auxiliary widgets (created on demand by the main window).
        # Connectivity/feature widgets are keyed by embedding so each embedding
        # gets its own widget (showing that embedding's features).
        self._connectivity_widgets = {}
        self._feature_explorer_widgets = {}
        self._annotation_dialog = None
        self._meta_explorer_dialog = None
        self._distance_widgets = []

        self.init_ui()

    @property
    def selection_counter(self):
        """Status bar label showing this view's selection count."""
        return self._selection_counter

    def aux_widgets(self):
        """All live auxiliary widgets owned by this view.

        Excludes the annotation dialog: it is application-modal (tabs cannot
        be switched while it is open) and manages its own window title.
        """
        candidates = [
            *self._connectivity_widgets.values(),
            *self._feature_explorer_widgets.values(),
            self._meta_explorer_dialog,
            *self._distance_widgets,
        ]
        widgets = []
        for widget in candidates:
            if widget is None:
                continue
            try:
                widget.objectName()  # raises RuntimeError if Qt object is gone
            except RuntimeError:
                continue
            widgets.append(widget)
        return widgets

    def focus_canvas(self):
        """Give keyboard focus to the scatter canvas.

        The global key bindings (Escape, space, arrows, ...) are rendercanvas
        events and only fire while the canvas widget has Qt focus.
        """
        canvas = getattr(getattr(self, "fig_scatter", None), "canvas", None)
        # The RenderCanvas wrapper proxies events to an inner subwidget; that
        # subwidget is the one with a focus policy and the key handlers.
        target = getattr(canvas, "_subwidget", None) or canvas
        if target is None:
            return
        try:
            target.setFocus(Qt.OtherFocusReason)
        except RuntimeError:
            pass  # canvas already deleted

    def teardown_rendering(self):
        """Stop render backends before Qt starts deleting child widgets."""
        if self._teardown_done:
            return
        self._teardown_done = True

        try:
            fig = getattr(self, "fig_scatter", None)
            canvas = getattr(fig, "canvas", None)
            if canvas is not None:
                canvas.close()
        except Exception as e:
            logger.debug(f"Failed to close scatter canvas cleanly: {e}")

        try:
            viewer = getattr(self, "ngl_viewer", None)
            if viewer is not None:
                viewer.close()
        except Exception as e:
            logger.debug(f"Failed to close neuroglancer viewer cleanly: {e}")

    def closeEvent(self, event):
        self.teardown_rendering()
        super().closeEvent(event)

    def init_ui(self):
        """Initialize the user interface."""
        layout = QVBoxLayout()

        # Create a vertical splitter with two widgets
        self.splitter = QSplitter(Qt.Vertical)

        # Create the scatter and viewer panes
        self.scatter_widget = QWidget()
        self.setup_scatter_widget()
        self.viewer_widget = QWidget()
        self.setup_viewer_widget()

        # Add widgets to splitter
        self.splitter.addWidget(self.scatter_widget)
        self.splitter.addWidget(self.viewer_widget)

        # Make the divider more visible
        self.splitter.setHandleWidth(1)
        self.splitter.setStyleSheet(
            "QSplitter::handle { background-color: #cccccc; border: 1px solid #999999; }"
        )

        # Set splitter to expand equally (50/50 split)
        self.splitter.setStretchFactor(0, 1)
        self.splitter.setStretchFactor(1, 1)
        self.splitter.setSizes(
            [int(self.splitter.height() / 2), int(self.splitter.height() / 2)]
        )

        # Current arrangement of the two panes; persisted across sessions so the
        # window reopens the way it was left (see MainWindow save/restore). Kept
        # in sync by show_only / show_side_by_side / show_stacked.
        self._layout_mode = "stacked"

        layout.addWidget(self.splitter)
        self.setLayout(layout)

        # Wire overlay button actions
        self.configure_overlay_actions()

    @resize_figures
    def show_only(self, target_widget):
        """Show only the target widget inside the splitter."""
        self.scatter_widget.setVisible(True)
        self.viewer_widget.setVisible(True)

        if target_widget is self.scatter_widget:
            self.scatter_widget.show()
            self.viewer_widget.hide()
            self.splitter.setSizes([1, 0])
            self._layout_mode = "only_scatter"
        elif target_widget is self.viewer_widget:
            self.viewer_widget.show()
            self.scatter_widget.hide()
            self.splitter.setSizes([0, 1])
            self._layout_mode = "only_viewer"

    @resize_figures
    def show_side_by_side(self):
        """Show both widgets side-by-side (horizontal)."""
        self.scatter_widget.show()
        self.viewer_widget.show()
        self.splitter.setOrientation(Qt.Horizontal)
        self._set_equal_sizes_horizontal()
        self._layout_mode = "side_by_side"

    @resize_figures
    def show_stacked(self):
        """Show both widgets stacked vertically."""
        self.scatter_widget.show()
        self.viewer_widget.show()
        self.splitter.setOrientation(Qt.Vertical)
        self._set_equal_sizes_vertical()
        self._layout_mode = "stacked"

    def apply_layout_mode(self, mode, ratio=None):
        """Restore a saved arrangement; defaults to stacked for unknown values.

        ``ratio`` is the fraction of the splitter the scatter pane should get,
        applied on top of the two-pane layouts (ignored when one pane is
        maximised). It overrides the 50/50 split the show_* methods set.
        """
        if mode == "side_by_side":
            self.show_side_by_side()
        elif mode == "only_scatter":
            self.show_only(self.scatter_widget)
        elif mode == "only_viewer":
            self.show_only(self.viewer_widget)
        else:
            self.show_stacked()

        if ratio is not None and self._layout_mode in ("stacked", "side_by_side"):
            self._apply_ratio(ratio)

    def layout_ratio(self):
        """Fraction of the splitter allotted to the scatter pane, or None.

        Only meaningful for the two-pane layouts; returns None when one pane is
        maximised (where the split reads as all-or-nothing).
        """
        if self._layout_mode not in ("stacked", "side_by_side"):
            return None
        sizes = self.splitter.sizes()
        total = sum(sizes)
        if total <= 0 or len(sizes) < 2:
            return None
        return sizes[0] / total

    def _apply_ratio(self, ratio):
        """Position the divider so the scatter pane gets ``ratio`` of the extent."""
        ratio = min(max(float(ratio), 0.0), 1.0)
        if self.splitter.orientation() == Qt.Horizontal:
            total = max(1, self.splitter.size().width())
        else:
            total = max(1, self.splitter.size().height())
        first = int(total * ratio)
        self.splitter.setSizes([first, max(0, total - first)])

    def _set_equal_sizes_horizontal(self):
        width = max(1, self.splitter.size().width())
        self.splitter.setSizes([width // 2, width // 2])

    def _set_equal_sizes_vertical(self):
        height = max(1, self.splitter.size().height())
        self.splitter.setSizes([height // 2, height // 2])

    def _set_sidebar_width(self, splitter, sidebar_width):
        """Set a splitter sidebar to a fixed pixel width when possible."""
        total_width = max(1, splitter.size().width())
        if total_width > sidebar_width:
            splitter.setSizes([sidebar_width, total_width - sidebar_width])
        else:
            splitter.setSizes([total_width // 2, total_width - (total_width // 2)])

    def configure_overlay_actions(self):
        """Connect overlay buttons to actions."""
        if hasattr(self.scatter_widget, "buttons") and self.scatter_widget.buttons:
            # Left-most button: fullscreen scatter
            self.scatter_widget.buttons[0].clicked.connect(
                lambda: self.show_only(self.scatter_widget)
            )
            # Middle button: stacked vertically
            self.scatter_widget.buttons[1].clicked.connect(self.show_side_by_side)
            # Right button: side-by-side
            self.scatter_widget.buttons[2].clicked.connect(self.show_stacked)

        if hasattr(self.viewer_widget, "buttons") and self.viewer_widget.buttons:
            # Left-most button: fullscreen viewer
            self.viewer_widget.buttons[0].clicked.connect(
                lambda: self.show_only(self.viewer_widget)
            )
            # Middle button: stacked vertically
            self.viewer_widget.buttons[1].clicked.connect(self.show_side_by_side)
            # Right button: side-by-side
            self.viewer_widget.buttons[2].clicked.connect(self.show_stacked)

    def update_left_button_position(self, host_widget):
        """Position the left sidebar toggle relative to sidebar visibility."""
        if not hasattr(host_widget, "left_button") or not hasattr(
            host_widget, "sidebar"
        ):
            return

        button = host_widget.left_button
        sidebar = host_widget.sidebar

        # Update label based on sidebar visibility
        button.setText("<<" if sidebar.isVisible() else ">>")

        x_offset = sidebar.width() + 5 if sidebar.isVisible() else 5
        y_offset = (host_widget.height() - button.height()) // 2
        button.move(x_offset, y_offset)

    def toggle_figure_controls(self):
        """Toggle visibility of the scatter figure controls sidebar."""
        sidebar = getattr(self.scatter_widget, "sidebar", None)
        if sidebar is None:
            return

        sidebar.setVisible(not sidebar.isVisible())
        # Size the sidebar on first open: the splitter is laid out at its real
        # width here, unlike during construction (tab pages get an early layout
        # pass at a junk size that would corrupt the splitter's size state).
        if sidebar.isVisible() and not self.scatter_widget._initial_sidebar_width_applied:
            self._set_sidebar_width(self.scatter_widget.splitter, 300)
            self.scatter_widget._initial_sidebar_width_applied = True
        self.update_left_button_position(self.scatter_widget)

        # Force an update to prevent transparency artifacts on sidebar toggles.
        self.force_update()
        self.resize_figures()
        self.fig_scatter.force_single_render()
        self.ngl_viewer.force_single_render()

    def toggle_viewer_controls(self):
        """Toggle visibility of the neuroglancer viewer controls sidebar."""
        sidebar = getattr(self.viewer_widget, "sidebar", None)
        if sidebar is None:
            return

        sidebar.setVisible(not sidebar.isVisible())
        if sidebar.isVisible() and not self.viewer_widget._initial_sidebar_width_applied:
            self._set_sidebar_width(self.viewer_widget.splitter, 300)
            self.viewer_widget._initial_sidebar_width_applied = True
        self.update_left_button_position(self.viewer_widget)

        # Force an update to prevent transparency artifacts on sidebar toggles.
        self.force_update()
        self.resize_figures()
        self.fig_scatter.force_single_render()
        self.ngl_viewer.force_single_render()

    def setup_scatter_widget(self):
        """Set up the scatter pane with overlay buttons and a left-positioned button."""
        # Initialize and connect the figure to the scatter pane
        self.fig_scatter = ScatterFigure(selection_counter=self._selection_counter, parent=self.scatter_widget)
        self.fig_scatter.show()

        # Create main layout for scatter pane
        main_layout = QHBoxLayout()
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # Create splitter for resizable sidebar
        splitter = QSplitter(Qt.Horizontal)
        splitter.setHandleWidth(1)
        splitter.setStyleSheet("QSplitter::handle { background-color: #cccccc; }")

        # Create sidebar
        sidebar = QFrame()
        sidebar.setMinimumWidth(250)
        sidebar_layout = QVBoxLayout()
        sidebar_layout.setContentsMargins(10, 10, 10, 10)

        # Add scatter controls to sidebar
        self.scatter_controls = ScatterControls(self.fig_scatter)
        # Keep controls usable in short panes by scrolling instead of enforcing
        # a large minimum height on the scatter pane.
        scatter_controls_scroll = QScrollArea()
        scatter_controls_scroll.setWidgetResizable(True)
        scatter_controls_scroll.setFrameShape(QFrame.NoFrame)
        scatter_controls_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scatter_controls_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.scatter_controls.setSizePolicy(
            QSizePolicy.Expanding, QSizePolicy.Preferred
        )
        scatter_controls_scroll.setWidget(self.scatter_controls)

        original_scroll_resize = scatter_controls_scroll.resizeEvent

        def on_scatter_controls_scroll_resize(event):
            original_scroll_resize(event)
            # Keep controls width locked to the viewport to avoid horizontal
            # clipping/overflow while still allowing vertical scrolling.
            viewport_width = max(1, scatter_controls_scroll.viewport().width())
            self.scatter_controls.setFixedWidth(viewport_width)

        scatter_controls_scroll.resizeEvent = on_scatter_controls_scroll_resize
        sidebar_layout.addWidget(scatter_controls_scroll, 1)

        sidebar_layout.addStretch()
        sidebar.setLayout(sidebar_layout)
        sidebar.setVisible(False)

        # Create content area
        content = QWidget()
        content_layout = QVBoxLayout()
        content_layout.setContentsMargins(0, 0, 0, 0)
        content.setLayout(content_layout)

        # Add the scatter figure to the content area
        content_layout.addWidget(self.fig_scatter.canvas)

        splitter.addWidget(sidebar)
        splitter.addWidget(content)
        # Resizes go to the content pane; the sidebar keeps its width. The
        # sidebar width itself is set when it is first shown (see
        # toggle_figure_controls).
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)

        main_layout.addWidget(splitter)
        self.scatter_widget.setLayout(main_layout)
        self.scatter_widget.splitter = splitter
        self.scatter_widget.sidebar = sidebar
        self.scatter_widget.content = content
        self.scatter_widget._initial_sidebar_width_applied = False

        # Create a button positioned on the left, centered vertically
        left_button = QPushButton()
        left_button.setFixedSize(40, 40)
        left_button.setParent(self.scatter_widget)
        left_button.setFlat(True)
        left_button.setAutoFillBackground(True)
        left_button.setAttribute(Qt.WA_TranslucentBackground, True)
        left_button.setAttribute(Qt.WA_NoSystemBackground, True)
        left_button.setStyleSheet(
            "QPushButton { background-color: transparent; border: none; }"
            "QPushButton:hover { background-color: rgba(0, 0, 0, 0.08); border-radius: 4px; }"
            "QPushButton:pressed { background-color: rgba(0, 0, 0, 0.12); border-radius: 4px; }"
            "QPushButton:focus { outline: none; }"
        )
        left_button.setToolTip("Toggle scatter sidebar")

        def toggle_sidebar():
            self.toggle_figure_controls()

        left_button.clicked.connect(toggle_sidebar)
        self.fig_scatter.key_events["c"] = toggle_sidebar

        # Create three square buttons for the overlay (icon buttons)
        button_size = 25
        button_spacing = 3
        buttons = []

        icon_defs = [
            (
                str(ASSETS_FILE_PATH / "button_fullscreen.png"),
                "Show only scatter",
            ),
            (
                str(ASSETS_FILE_PATH / "button_split_vertical.png"),
                "Stack widgets vertically",
            ),
            (
                str(ASSETS_FILE_PATH / "button_split_horizontal.png"),
                "Place widgets side by side",
            ),
        ]

        for icon_path, tip in icon_defs:
            button = QPushButton()
            button.setFixedSize(button_size, button_size)
            button.setIcon(QIcon(str(icon_path)))
            button.setIconSize(QSize(button_size - 12, button_size - 12))
            button.setParent(self.scatter_widget)
            button.setFlat(True)
            button.setAutoFillBackground(False)
            button.setAttribute(Qt.WA_TranslucentBackground, True)
            button.setAttribute(Qt.WA_NoSystemBackground, True)
            button.setStyleSheet(
                "QPushButton { background-color: transparent; border: none; }"
                "QPushButton:hover { background-color: rgba(0, 0, 0, 0.08); border-radius: 4px; }"
                "QPushButton:pressed { background-color: rgba(0, 0, 0, 0.12); border-radius: 4px; }"
                "QPushButton:focus { outline: none; }"
            )
            button.setToolTip(tip)
            buttons.append(button)

        # Position buttons in the top right corner
        for i, button in enumerate(buttons):
            x = (
                self.scatter_widget.width()
                - (button_size + button_spacing) * (3 - i)
                + button_spacing
            )
            y = button_spacing
            button.move(x, y)

        # Initial position of left button
        self.update_left_button_position(self.scatter_widget)

        # Store reference to adjust position on resize
        self.scatter_widget.left_button = left_button
        self.scatter_widget.sidebar = sidebar
        self.scatter_widget.buttons = buttons
        self.scatter_widget.button_size = button_size
        self.scatter_widget.button_spacing = button_spacing

        # Override resizeEvent to reposition buttons
        original_resize = self.scatter_widget.resizeEvent

        def on_resize(event):
            original_resize(event)
            for i, button in enumerate(buttons):
                x = (
                    self.scatter_widget.width()
                    - (button_size + button_spacing) * (3 - i)
                    + button_spacing
                )
                y = button_spacing
                button.move(x, y)
            # Reposition left button
            self.update_left_button_position(self.scatter_widget)
            # Resize figure if necessary
            self.resize_figures()

        self.scatter_widget.resizeEvent = on_resize

        # Update button position when splitter is moved
        splitter.splitterMoved.connect(
            lambda: self.update_left_button_position(self.scatter_widget)
        )

    def setup_viewer_widget(self):
        """Set up the viewer pane with a left-positioned button and sidebar."""
        # Initialize and connect the neuroglancer viewer in the viewer pane
        self.ngl_viewer = NglViewer(figure=self.fig_scatter, viewer_kwargs=dict(parent=self.viewer_widget))
        self.ngl_viewer.viewer.show()

        # Hook the viewer up to the figure
        self.fig_scatter.sync_viewer(self.ngl_viewer)

        # Create main layout for viewer pane
        main_layout = QHBoxLayout()
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # Create splitter for resizable sidebar
        splitter = QSplitter(Qt.Horizontal)
        splitter.setHandleWidth(1)
        splitter.setStyleSheet("QSplitter::handle { background-color: #cccccc; }")

        # Create sidebar
        sidebar = QFrame()
        sidebar.setMinimumWidth(250)
        sidebar_layout = QVBoxLayout()
        sidebar_layout.setContentsMargins(10, 10, 10, 10)

        # Add viewer controls to sidebar
        self.ngl_viewer.viewer.show_controls()
        self.viewer_controls = self.ngl_viewer.viewer._controls
        viewer_controls_scroll = QScrollArea()
        viewer_controls_scroll.setWidgetResizable(True)
        viewer_controls_scroll.setFrameShape(QFrame.NoFrame)
        viewer_controls_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        viewer_controls_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.viewer_controls.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        viewer_controls_scroll.setWidget(self.viewer_controls)

        original_viewer_scroll_resize = viewer_controls_scroll.resizeEvent

        def on_viewer_controls_scroll_resize(event):
            original_viewer_scroll_resize(event)
            # Keep controls width locked to the viewport to avoid horizontal
            # clipping/overflow while still allowing vertical scrolling.
            viewport_width = max(1, viewer_controls_scroll.viewport().width())
            self.viewer_controls.setFixedWidth(viewport_width)

        viewer_controls_scroll.resizeEvent = on_viewer_controls_scroll_resize
        sidebar_layout.addWidget(viewer_controls_scroll, 1)

        sidebar_layout.addStretch()
        sidebar.setLayout(sidebar_layout)
        sidebar.setVisible(False)

        # Create content area
        content = QWidget()
        content_layout = QVBoxLayout()
        content_layout.setContentsMargins(0, 0, 0, 0)
        content.setLayout(content_layout)

        # Add the neuroglancer viewer to the content area
        content_layout.addWidget(self.ngl_viewer.viewer.canvas)

        splitter.addWidget(sidebar)
        splitter.addWidget(content)
        # Resizes go to the content pane; the sidebar keeps its width. The
        # sidebar width itself is set when it is first shown (see
        # toggle_viewer_controls).
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)

        main_layout.addWidget(splitter)
        self.viewer_widget.setLayout(main_layout)
        self.viewer_widget.splitter = splitter
        self.viewer_widget.sidebar = sidebar
        self.viewer_widget.content = content
        self.viewer_widget._initial_sidebar_width_applied = False

        # Create a button positioned on the left, centered vertically
        left_button = QPushButton()
        left_button.setFixedSize(40, 40)
        left_button.setParent(self.viewer_widget)
        left_button.setFlat(True)
        left_button.setAutoFillBackground(False)
        left_button.setAttribute(Qt.WA_TranslucentBackground, True)
        left_button.setAttribute(Qt.WA_NoSystemBackground, True)
        left_button.setStyleSheet(
            "QPushButton { background-color: transparent; border: none; }"
            "QPushButton:hover { background-color: rgba(0, 0, 0, 0.08); border-radius: 4px; }"
            "QPushButton:pressed { background-color: rgba(0, 0, 0, 0.12); border-radius: 4px; }"
            "QPushButton:focus { outline: none; }"
        )
        left_button.setToolTip("Toggle viewer sidebar")

        def toggle_sidebar():
            self.toggle_viewer_controls()

        left_button.clicked.connect(toggle_sidebar)

        # Make it so that pressing 'c' in the viewer opens/closes the controls sidebar
        self.ngl_viewer.viewer._key_events["c"] = toggle_sidebar

        # Create three square buttons for the overlay (icon buttons)
        button_size = 25
        button_spacing = 3
        buttons = []

        icon_defs = [
            (
                str(ASSETS_FILE_PATH / "button_fullscreen.png"),
                "Show only 3D viewer",
            ),
            (
                str(ASSETS_FILE_PATH / "button_split_vertical.png"),
                "Stack widgets vertically",
            ),
            (
                str(ASSETS_FILE_PATH / "button_split_horizontal.png"),
                "Place widgets side by side",
            ),
        ]

        for icon_path, tip in icon_defs:
            button = QPushButton()
            button.setFixedSize(button_size, button_size)
            button.setIcon(QIcon(icon_path))
            button.setIconSize(QSize(button_size - 12, button_size - 12))
            button.setParent(self.viewer_widget)
            button.setFlat(True)
            button.setAutoFillBackground(False)
            button.setAttribute(Qt.WA_TranslucentBackground, True)
            button.setAttribute(Qt.WA_NoSystemBackground, True)
            button.setStyleSheet(
                "QPushButton { background-color: transparent; border: none; }"
                "QPushButton:hover { background-color: rgba(0, 0, 0, 0.08); border-radius: 4px; }"
                "QPushButton:pressed { background-color: rgba(0, 0, 0, 0.12); border-radius: 4px; }"
                "QPushButton:focus { outline: none; }"
            )
            button.setToolTip(tip)
            buttons.append(button)

        # Position overlay buttons in the top right corner
        for i, button in enumerate(buttons):
            x = (
                self.viewer_widget.width()
                - (button_size + button_spacing) * (3 - i)
                + button_spacing
            )
            y = button_spacing
            button.move(x, y)

        # Initial position of left button
        self.update_left_button_position(self.viewer_widget)

        # Store reference to adjust position on resize
        self.viewer_widget.left_button = left_button
        self.viewer_widget.buttons = buttons
        self.viewer_widget.button_size = button_size
        self.viewer_widget.button_spacing = button_spacing
        self.viewer_widget.sidebar = sidebar

        # Override resizeEvent to reposition buttons
        original_resize = self.viewer_widget.resizeEvent

        def on_resize(event):
            original_resize(event)
            for i, button in enumerate(buttons):
                x = (
                    self.viewer_widget.width()
                    - (button_size + button_spacing) * (3 - i)
                    + button_spacing
                )
                y = button_spacing
                button.move(x, y)
                button.raise_()
            # Reposition left button
            self.update_left_button_position(self.viewer_widget)

        self.viewer_widget.resizeEvent = on_resize

        # Update button position when splitter is moved
        splitter.splitterMoved.connect(
            lambda: self.update_left_button_position(self.viewer_widget)
        )

        # Initial positions
        self.update_left_button_position(self.viewer_widget)
        for i, button in enumerate(buttons):
            x = (
                self.viewer_widget.width()
                - (button_size + button_spacing) * (3 - i)
                + button_spacing
            )
            y = button_spacing
            button.move(x, y)

        def apply_initial_viewer_overlay_layout():
            for i, button in enumerate(buttons):
                x = (
                    self.viewer_widget.width()
                    - (button_size + button_spacing) * (3 - i)
                    + button_spacing
                )
                y = button_spacing
                button.move(x, y)
                button.raise_()

            self.update_left_button_position(self.viewer_widget)

        # Ensure overlays are correctly placed after the first real layout pass.
        QTimer.singleShot(0, apply_initial_viewer_overlay_layout)

    def resize_figures(self):
        """Resize figures to match their parent widgets."""
        self.fig_scatter.size = (
            self.scatter_widget.content.width(),
            self.scatter_widget.content.height(),
        )
        self.ngl_viewer.size = (
            self.viewer_widget.content.width(),
            self.viewer_widget.content.height(),
        )

    def force_update(self):
        """Force a repaint of the main window to avoid visual glitches."""
        window = self.window()
        if window:
            orig_size = window.size()
            if orig_size.width() > 1 and orig_size.height() > 1:
                window.resize(orig_size.width() - 1, orig_size.height() - 1)
                window.repaint()
                QApplication.processEvents()
                window.resize(orig_size)


class _KeyBadge(QLabel):
    """A keyboard key-cap badge."""

    def __init__(self, text, parent=None):
        super().__init__(text, parent)
        self.setStyleSheet(
            "QLabel {"
            "  background-color: #f0f0f0;"
            "  color: #111111;"
            "  border: 1px solid #b0b0b0;"
            "  border-bottom: 2px solid #888;"
            "  border-radius: 3px;"
            "  padding: 1px 5px;"
            "  font-size: 11px;"
            "}"
        )
        self.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)


class _MouseGlyph(QWidget):
    """Tiny painted mouse icon with one button zone highlighted."""

    def __init__(self, area="left", hold=False, double=False, parent=None):
        super().__init__(parent)
        self._area = area      # "left", "right", "middle", "scroll"
        self._hold = hold
        self._double = double
        self.setFixedSize(20, 32)
        self.setAttribute(Qt.WA_TransparentForMouseEvents)

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)

        W = float(self.width())
        H_body = 24.0
        r = 5.0
        btn_h = H_body * 0.44

        body_fill = QColor("#f0f0f0")
        body_line = QColor("#888")
        hi_fill = QColor("#b8d4f0")
        hi_line = QColor("#5b9bd5")

        body = QRectF(0.5, 0.5, W - 1, H_body - 1)
        clip_path = QPainterPath()
        clip_path.addRoundedRect(body, r, r)

        # Body fill
        p.setPen(QPen(body_line, 1))
        p.setBrush(QBrush(body_fill))
        p.drawPath(clip_path)

        # Three button zones: left | scroll-wheel | right
        zone_w = (W - 1) / 3.0
        zone_rects = {
            "left":   QRectF(0.5,               0.5, zone_w, btn_h),
            "middle": QRectF(0.5 + zone_w,      0.5, zone_w, btn_h),
            "scroll": QRectF(0.5 + zone_w,      0.5, zone_w, btn_h),
            "right":  QRectF(0.5 + 2 * zone_w,  0.5, zone_w, btn_h),
        }

        # Highlight active zone, clipped to rounded body
        if self._area in zone_rects:
            p.save()
            p.setClipPath(clip_path)
            p.setPen(Qt.NoPen)
            p.setBrush(QBrush(hi_fill))
            p.drawRect(zone_rects[self._area])
            p.restore()

        # Redraw body outline
        p.setPen(QPen(body_line, 1))
        p.setBrush(Qt.NoBrush)
        p.drawPath(clip_path)

        # Zone dividers (two vertical lines in button area)
        p.setPen(QPen(QColor("#c0c0c0"), 0.8))
        for i in (1, 2):
            x = 0.5 + i * zone_w
            p.drawLine(QPointF(x, 0.5), QPointF(x, btn_h))

        # Horizontal separator below button area
        p.drawLine(QPointF(0.5, btn_h), QPointF(W - 0.5, btn_h))

        # Scroll wheel indicator in middle zone (when not highlighted)
        if self._area not in ("middle", "scroll"):
            sw = QRectF(0.5 + zone_w + 1.5, 2.0, zone_w - 3.0, btn_h - 3.5)
            p.setPen(QPen(QColor("#aaa"), 0.8))
            p.setBrush(QBrush(QColor("#d0d0d0")))
            p.drawRoundedRect(sw, 2, 2)

        # Annotation text below body (for hold / double-click variants)
        if self._double or self._hold:
            text = "2×" if self._double else "hold"
            color = hi_line if self._double else QColor("#777")
            f = p.font()
            f.setPixelSize(7)
            p.setFont(f)
            p.setPen(QPen(color, 1))
            p.drawText(QRectF(0, H_body + 0.5, W, 8), Qt.AlignCenter, text)

        p.end()


def build_project_summary(loader, data):
    """Assemble an ordered dict of derived project stats for the details dialog.

    ``loader`` is a :class:`SingleProjectLoader` (or None) and ``data`` is the
    compiled project dict (or None). Every field is guarded so a missing source
    is omitted rather than rendered as ``"None"``. Kept as a free function (no
    ``MainWindow``) so it stays unit-testable with a duck-typed fake loader.
    """
    summary = {}
    if loader is not None:
        name = getattr(loader, "name", None)
        if name is not None:
            summary["name"] = str(name)
        path = getattr(loader, "path", None)
        if path is not None:
            summary["path"] = str(path)
        try:
            summary["location"] = "remote" if loader.is_remote else "local"
        except Exception:
            pass

    if isinstance(data, dict) and data.get("meta") is not None:
        try:
            summary["observations"] = len(data["meta"])
        except Exception:
            pass

    # Column count: prefer the loader's lazy schema (no meta materialisation);
    # fall back to the compiled meta frame.
    n_columns = None
    if loader is not None:
        try:
            n_columns = len(loader.meta_columns)
        except Exception:
            n_columns = None
    if n_columns is None and isinstance(data, dict) and data.get("meta") is not None:
        try:
            n_columns = len(data["meta"].columns)
        except Exception:
            n_columns = None
    if n_columns is not None:
        summary["meta columns"] = n_columns

    # Embedding names, with the active one marked. Prefer the compiled entries
    # (they carry the active index); fall back to the loader's specs.
    names = None
    active = None
    if isinstance(data, dict):
        entries = data.get("embedding_entries")
        if entries:
            names = [str(e.get("name", i)) for i, e in enumerate(entries)]
            idx = data.get("active_embedding")
            if isinstance(idx, int) and 0 <= idx < len(names):
                active = idx
    if names is None and loader is not None:
        try:
            names = [str(s["name"]) for s in loader.normalized_embedding_specs]
        except Exception:
            names = None
    if names:
        summary["embeddings"] = [
            f"{n} (active)" if i == active else n for i, n in enumerate(names)
        ]

    if loader is not None:
        has_distances = getattr(loader, "has_distances", None)
        if has_distances is not None:
            summary["has distances"] = bool(has_distances)
        has_features = getattr(loader, "has_features", None)
        if has_features is not None:
            feat_type = getattr(loader, "feature_type", None)
            summary["has features"] = (
                f"yes ({feat_type})" if has_features and feat_type else bool(has_features)
            )

    return summary


class AnnotationLogDialog(QDialog):
    """Dialog that shows the current window's annotation log."""

    FORMAT_PLAIN = "Plain Text"
    FORMAT_JSON = "JSON"
    FORMAT_CSV = "CSV"

    def __init__(self, parent=None, entries=None):
        super().__init__(parent)
        self.setWindowTitle("Annotation Log")
        self.resize(700, 520)

        self._entries = entries or []

        layout = QVBoxLayout(self)

        header_layout = QHBoxLayout()
        header_layout.addStretch(1)
        header_layout.addWidget(QLabel("Format:"))

        self._format_combo = QComboBox(self)
        self._format_combo.addItems(
            [
                self.FORMAT_PLAIN,
                self.FORMAT_JSON,
                self.FORMAT_CSV,
            ]
        )
        self._format_combo.setCurrentText(self.FORMAT_PLAIN)
        self._format_combo.currentTextChanged.connect(self._on_format_changed)
        header_layout.addWidget(self._format_combo)
        layout.addLayout(header_layout)

        self._text = QPlainTextEdit(self)
        self._text.setReadOnly(True)
        self._text.setLineWrapMode(QPlainTextEdit.NoWrap)
        layout.addWidget(self._text)

        self.update_entries(self._entries)

    def _on_format_changed(self, value):
        self.update_entries(self._entries)

    def update_entries(self, entries):
        self._entries = entries or []
        if not self._entries:
            self._text.setPlainText("No annotation log entries.")
            return

        fmt = self._format_combo.currentText()
        if fmt == self.FORMAT_JSON:
            self._text.setPlainText(json.dumps(self._entries, indent=2))
            return

        if fmt == self.FORMAT_CSV:
            self._text.setPlainText(self._format_csv(self._entries))
            return

        self._text.setPlainText(self._format_plain(self._entries))

    def _format_plain(self, entries):
        lines = []
        for entry in entries:
            dataset = entry.get("dataset")
            fields = entry.get("fields", [])
            ids = entry.get("ids", [])
            value = entry.get("value")
            lines.append(f"Dataset: {dataset}")
            lines.append("Fields: " + ", ".join(str(f) for f in fields))
            lines.append("Value: " + str(value))
            lines.append("IDs: " + ", ".join(str(i) for i in ids))
            lines.append("---")
        return "\n".join(lines).rstrip("\n-")

    def _format_csv(self, entries):
        buffer = io.StringIO()
        writer = csv.writer(buffer)
        writer.writerow(["dataset", "fields", "value", "ids"])
        for entry in entries:
            dataset = entry.get("dataset")
            fields = entry.get("fields", [])
            value = entry.get("value")
            ids = entry.get("ids", [])
            writer.writerow(
                [
                    dataset,
                    ";".join(str(f) for f in fields),
                    value,
                    ";".join(str(i) for i in ids),
                ]
            )
        return buffer.getvalue().rstrip("\n")


class MainWindow(QMainWindow):
    """Main application window."""

    RECENT_PROJECTS_KEY = "openRecentProjects/v1"
    MAX_RECENT_PROJECTS = 10
    annotation_submit_result_received = Signal(str)
    annotation_changed = Signal(object)

    def __init__(self, adopt_view=None):
        super().__init__()
        self.setAttribute(Qt.WA_DeleteOnClose, True)
        self.settings = QSettings("BigClust", "BigClustGUI")
        self.open_recent_menu = None
        self.connectivity_table_action = None
        self.distances_table_action = None
        self.feature_explorer_action = None
        self.meta_explorer_action = None
        self.sync_viewer_action = None
        self._hover_columns_menu = None
        self._adopt_view = adopt_view
        self._active_selection_counter = None
        # Window this one was detached from (for "merge back"), if any.
        self._parent_window = None
        # Whether widgets of background tabs are hidden (global preference).
        self._hide_inactive_aux_widgets = self.settings.value(
            "auxWidgets/hideInactiveTabWidgets", True, type=bool
        )
        self._annotation_log = []
        self._annotation_log_dir = Path.home() / ".bigclust"
        self._annotation_log_dir.mkdir(parents=True, exist_ok=True)
        self._annotation_log_file = self._annotation_log_dir / (
            f"annotation_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.log"
        )
        self._annotation_log_file.touch(exist_ok=True)
        self.annotation_submit_result_received.connect(
            self._handle_annotation_submit_result
        )
        self.annotation_changed.connect(self._handle_annotation_changed)
        _OPEN_WINDOWS.append(self)
        self.destroyed.connect(
            lambda _obj=None, w=self: (
                _OPEN_WINDOWS.remove(w) if w in _OPEN_WINDOWS else None
            )
        )
        self.init_ui()

    def current_view(self):
        """Return the currently active view (MainWidget) or None."""
        tabs = getattr(self, "_tabs", None)
        return tabs.currentWidget() if tabs is not None else None

    def views(self):
        """Return all views (MainWidgets) hosted in this window."""
        tabs = getattr(self, "_tabs", None)
        if tabs is None:
            return []
        return [tabs.widget(i) for i in range(tabs.count())]

    # Project data lives on the per-tab view; these properties keep the many
    # existing `self._data` / `self._current_project_loader` references working
    # by delegating to the active tab.
    @property
    def _data(self):
        view = self.current_view()
        return getattr(view, "_data", None) if view is not None else None

    @_data.setter
    def _data(self, value):
        view = self.current_view()
        if view is not None:
            view._data = value

    @property
    def _current_project_loader(self):
        view = self.current_view()
        return (
            getattr(view, "_current_project_loader", None)
            if view is not None
            else None
        )

    @_current_project_loader.setter
    def _current_project_loader(self, value):
        view = self.current_view()
        if view is not None:
            view._current_project_loader = value

    def eventFilter(self, obj, event):
        if event.type() == QEvent.Close:
            owner = getattr(obj, "_owner_view", None)
            if owner is not None:
                # User closed an aux widget: don't restore it on tab switch.
                obj._visible_in_tab = False
                if obj in getattr(owner, "_connectivity_widgets", {}).values():
                    self._unsync_connectivity_widget(owner, obj)
        return super().eventFilter(obj, event)

    def init_ui(self):
        """Initialize the main window."""
        self.setWindowTitle("BigClust")
        self.setGeometry(100, 100, 800, 600)

        # Restore window geometry/state if available
        try:
            geom = self.settings.value("mainWindow/geometry")
            if geom is not None:
                self.restoreGeometry(geom)
            is_max = self.settings.value("mainWindow/isMaximized", False, type=bool)
            if is_max:
                self.showMaximized()
        except Exception:
            # Fall back silently if settings are missing or incompatible
            pass

        # Status bar
        self.status_bar = self.statusBar()
        self.status_bar.showMessage("Ready", timeout=5000)

        # Permanent indicator mirroring the active view's "Active embedding"
        # selector. Added before any per-view selection counter so it sits to
        # the left of it. Hidden until a view has more than one embedding.
        self.embedding_status_label = QLabel("")
        self.status_bar.addPermanentWidget(self.embedding_status_label)
        self.embedding_status_label.hide()

        # Tab widget hosting one view (MainWidget) per tab
        self._tabs = ViewTabWidget()
        self._tabs.currentChanged.connect(self._on_current_tab_changed)
        self._tabs.tabCloseRequested.connect(self._on_tab_close_requested)
        # Queued so the tab bar's mouse handling finishes before we mutate it.
        self._tabs.detach_requested.connect(
            self._on_tab_detach_requested, Qt.QueuedConnection
        )
        self.setCentralWidget(self._tabs)

        if self._adopt_view is not None:
            view = self._adopt_view
            self._adopt_view = None
            self.add_new_tab(
                title=getattr(view, "view_title", "untitled"), view=view
            )
        else:
            view = self.add_new_tab(title="untitled")
            # Reopen the initial tab in the arrangement (and divider position)
            # last used, saved in closeEvent. Deferred a turn so the splitter
            # has a real size by the time the sizing runs. New mid-session tabs
            # keep the default stacked 50/50 layout.
            mode = self.settings.value("mainWindow/layoutMode", "stacked")
            ratio = self._read_layout_ratio_setting()
            if (mode and mode != "stacked") or ratio is not None:
                QTimer.singleShot(
                    0, lambda v=view, m=mode, r=ratio: v.apply_layout_mode(m, r)
                )

        # Menu bar with File -> Open Project
        menu_bar = self.menuBar()
        menu_bar.clear()
        menu_bar.setNativeMenuBar(
            True
        )  # keep it visible inside the window across platforms
        file_menu = menu_bar.addMenu("File")
        open_project_action = QAction("Open Project", self)
        # Keyboard shortcut
        open_project_action.setShortcut("Ctrl+O")
        open_project_action.triggered.connect(self.show_open_project_dialog)
        file_menu.addAction(open_project_action)

        new_tab_action = QAction("New Tab", self)
        new_tab_action.setShortcut(QKeySequence("Ctrl+T"))
        new_tab_action.triggered.connect(lambda: self.add_new_tab(title="untitled"))
        file_menu.addAction(new_tab_action)

        new_window_action = QAction("New Window", self)
        new_window_action.setShortcut(QKeySequence("Shift+Meta+N"))
        new_window_action.triggered.connect(self.open_new_window)
        file_menu.addAction(new_window_action)

        close_tab_action = QAction("Close Tab", self)
        close_tab_action.setShortcut(QKeySequence.Close)
        close_tab_action.triggered.connect(self.close_current_tab)
        file_menu.addAction(close_tab_action)

        close_window_action = QAction("Close Window", self)
        close_window_action.setShortcut(QKeySequence("Shift+Ctrl+W"))
        close_window_action.triggered.connect(self.close)
        file_menu.addAction(close_window_action)

        self.open_recent_menu = file_menu.addMenu("Open Recent")
        self.refresh_open_recent_menu()

        # View menu
        view_menu = menu_bar.addMenu("View")
        self.connectivity_table_action = QAction("Connectivity Table", self)
        self.connectivity_table_action.setShortcut(QKeySequence("Shift+Meta+C"))
        self.connectivity_table_action.setEnabled(False)
        self.connectivity_table_action.triggered.connect(self.show_connectivity_table)
        view_menu.addAction(self.connectivity_table_action)

        self.distances_table_action = QAction("Distance Heatmap", self)
        self.distances_table_action.setShortcut(QKeySequence("Shift+Meta+D"))
        self.distances_table_action.setEnabled(False)
        self.distances_table_action.triggered.connect(self.show_distances_table)
        view_menu.addAction(self.distances_table_action)

        self.feature_explorer_action = QAction("Feature Explorer", self)
        self.feature_explorer_action.setShortcut(QKeySequence("Shift+Meta+F"))
        self.feature_explorer_action.setEnabled(False)
        self.feature_explorer_action.triggered.connect(self.show_feature_explorer)
        view_menu.addAction(self.feature_explorer_action)

        self.meta_explorer_action = QAction("Meta Data Explorer", self)
        self.meta_explorer_action.setShortcut(QKeySequence("Shift+Meta+M"))
        self.meta_explorer_action.setEnabled(False)
        self.meta_explorer_action.triggered.connect(self.show_meta_explorer)
        view_menu.addAction(self.meta_explorer_action)

        view_menu.addSeparator()

        center_menu = view_menu.addMenu("Center")

        center_scatter_action = QAction("Scatter", self)
        center_scatter_action.triggered.connect(
            lambda: self.current_view().fig_scatter.center_camera()
        )
        center_menu.addAction(center_scatter_action)

        center_3d_action = QAction("3D Viewer", self)
        center_3d_action.triggered.connect(
            lambda: self.current_view().ngl_viewer.viewer.center_camera()
        )
        center_menu.addAction(center_3d_action)

        view_menu.addSeparator()

        toggle_figure_controls_action = QAction("Toggle Figure Controls", self)
        toggle_figure_controls_action.triggered.connect(
            lambda: self.current_view().toggle_figure_controls()
        )
        view_menu.addAction(toggle_figure_controls_action)

        toggle_viewer_controls_action = QAction("Toggle Viewer Controls", self)
        toggle_viewer_controls_action.triggered.connect(
            lambda: self.current_view().toggle_viewer_controls()
        )
        view_menu.addAction(toggle_viewer_controls_action)

        reset_layout_action = QAction("Reset Layout", self)
        reset_layout_action.setToolTip(
            "Restore the default stacked 50/50 arrangement of the two panes."
        )
        reset_layout_action.triggered.connect(self.on_reset_layout)
        view_menu.addAction(reset_layout_action)

        self.sync_viewer_action = QAction("Synchronize Viewer", self)
        self.sync_viewer_action.setCheckable(True)
        self.sync_viewer_action.setChecked(True)
        self.sync_viewer_action.setToolTip(
            "When enabled, scatter selections are synchronized to the 3D viewer."
        )
        self.sync_viewer_action.setStatusTip(
            "Toggle synchronization from scatter selection to 3D viewer"
        )
        self.sync_viewer_action.toggled.connect(
            lambda checked: self.current_view().fig_scatter.set_viewer_sync(
                checked, sync_now=checked
            )
        )
        view_menu.addAction(self.sync_viewer_action)

        self.show_hoverinfo_action = QAction("Show Hoverinfo", self)
        self.show_hoverinfo_action.setCheckable(True)
        self.show_hoverinfo_action.setChecked(True)
        self.show_hoverinfo_action.toggled.connect(
            lambda checked: setattr(
                self.current_view().fig_scatter,
                "_hide_hover",
                not checked,
            )
        )
        view_menu.addAction(self.show_hoverinfo_action)

        self._hover_columns_menu = QMenu("Hover Columns", self)
        view_menu.addMenu(self._hover_columns_menu)
        self._hover_columns_menu.aboutToShow.connect(self._refresh_hover_columns_menu)
        self._hover_columns_menu.aboutToHide.connect(self._on_hover_columns_menu_hidden)

        # The view menu automatically contains an item to toggle fullscreen mode
        # This separator keeps it visually separate from the custom view actions we added above.
        view_menu.addSeparator()

        # Selection menu
        selection_menu = menu_bar.addMenu("Selection")

        select_all_action = QAction("Select All", self)
        # On other platforms Ctrl+A is taken by "Set Annotations" (see below).
        if sys.platform == "darwin":
            select_all_action.setShortcut(QKeySequence("Ctrl+A"))
        select_all_action.triggered.connect(self.on_select_all)
        selection_menu.addAction(select_all_action)

        invert_selection_action = QAction("Invert Selection", self)
        invert_selection_action.setShortcut(QKeySequence("Ctrl+I"))
        invert_selection_action.triggered.connect(self.on_invert_selection)
        selection_menu.addAction(invert_selection_action)

        deselect_all_action = QAction("Deselect All", self)
        deselect_all_action.triggered.connect(self.on_deselect_all)
        selection_menu.addAction(deselect_all_action)

        set_annotations_action = QAction("Set Annotations", self)
        set_annotations_action.setShortcut(
            QKeySequence("Meta+A")
            if sys.platform == "darwin"
            else QKeySequence("Ctrl+A")
        )
        set_annotations_action.triggered.connect(self.show_annotation_dialog)
        selection_menu.addAction(set_annotations_action)

        selection_menu.addSeparator()
        grow_action = QAction("Grow Selection", self)
        # On macOS Qt maps Ctrl -> Cmd. Bind both "=" and "+" so the grow
        # shortcut fires whether or not Shift is held for the "+" key.
        grow_action.setShortcuts([QKeySequence("Ctrl+="), QKeySequence("Ctrl++")])
        grow_action.triggered.connect(self.on_grow_selection)
        selection_menu.addAction(grow_action)

        shrink_action = QAction("Shrink Selection", self)
        shrink_action.setShortcut(QKeySequence("Ctrl+-"))
        shrink_action.triggered.connect(self.on_shrink_selection)
        selection_menu.addAction(shrink_action)

        # Rebuilt on show so it reflects the current tab's figure (see
        # `_refresh_grow_shrink_menu`).
        self._grow_shrink_menu = selection_menu.addMenu("Grow/Shrink Options")
        self._grow_shrink_menu.aboutToShow.connect(self._refresh_grow_shrink_menu)

        selection_menu.addSeparator()
        open_selection_in_new_tab_action = QAction("Open in New Tab", self)
        open_selection_in_new_tab_action.setShortcut(
            QKeySequence("Shift+Ctrl+Meta+N")
        )
        open_selection_in_new_tab_action.triggered.connect(
            self.on_open_selection_in_new_tab
        )
        selection_menu.addAction(open_selection_in_new_tab_action)

        open_in_neuroglancer_action = QAction("Open in Neuroglancer", self)
        open_in_neuroglancer_action.triggered.connect(self.on_open_in_neuroglancer)
        selection_menu.addAction(open_in_neuroglancer_action)

        copy_menu = selection_menu.addMenu("Copy to Clipboard")
        copy_ids_action = QAction("IDs", self)
        copy_ids_action.setShortcut("Ctrl+C")
        copy_ids_action.triggered.connect(self.on_copy_ids_to_clipboard)
        copy_menu.addAction(copy_ids_action)
        copy_meta_action = QAction("Meta Data", self)
        copy_meta_action.triggered.connect(self.on_copy_meta_to_clipboard)
        copy_menu.addAction(copy_meta_action)

        # Export menu
        export_menu = menu_bar.addMenu("Export")
        export_meta_menu = export_menu.addMenu("Meta Data")

        export_meta_clipboard_action = QAction("To Clipboard", self)
        export_meta_clipboard_action.triggered.connect(self.on_export_meta_to_clipboard)
        export_meta_menu.addAction(export_meta_clipboard_action)

        export_meta_csv_action = QAction("To CSV", self)
        export_meta_csv_action.triggered.connect(self.on_export_meta_to_csv)
        export_meta_menu.addAction(export_meta_csv_action)

        export_embedding_menu = export_menu.addMenu("Embedding")

        export_embedding_plotly_action = QAction("To Plotly", self)
        export_embedding_plotly_action.triggered.connect(
            self.on_export_embedding_to_plotly_html
        )
        export_embedding_menu.addAction(export_embedding_plotly_action)

        export_embedding_plotly_dashboard_action = QAction("To Dashboard", self)
        export_embedding_plotly_dashboard_action.triggered.connect(
            self.on_export_embedding_to_plotly_dashboard_html
        )
        export_embedding_menu.addAction(export_embedding_plotly_dashboard_action)

        # Window menu with Zoom and Minimize
        window_menu = menu_bar.addMenu("Window")

        zoom_action = QAction("Zoom", self)

        def toggle_zoom():
            if self.isMaximized():
                self.showNormal()
            else:
                self.showMaximized()

        zoom_action.triggered.connect(toggle_zoom)
        window_menu.addAction(zoom_action)

        minimize_action = QAction("Minimize", self)
        minimize_action.setShortcut("Ctrl+M")
        minimize_action.triggered.connect(self.showMinimized)
        window_menu.addAction(minimize_action)

        window_menu.addSeparator()

        self.detach_tab_action = QAction("Move Tab to New Window", self)
        self.detach_tab_action.setToolTip(
            "Open the current tab as a separate window (same as dragging the "
            "tab out of the tab bar)."
        )
        self.detach_tab_action.triggered.connect(self.detach_current_tab)
        self.detach_tab_action.setEnabled(False)
        window_menu.addAction(self.detach_tab_action)

        self.merge_window_action = QAction("Merge Window into Parent", self)
        self.merge_window_action.setToolTip(
            "Move this window's tabs back into the window it was detached from."
        )
        self.merge_window_action.triggered.connect(self.merge_into_parent_window)
        self.merge_window_action.setEnabled(False)
        window_menu.addAction(self.merge_window_action)

        # Enablement depends on tab count / detach lineage; refresh on open.
        window_menu.aboutToShow.connect(self._refresh_window_menu_actions)

        window_menu.addSeparator()

        self.hide_inactive_widgets_action = QAction(
            "Hide Widgets of Inactive Tabs", self
        )
        self.hide_inactive_widgets_action.setCheckable(True)
        self.hide_inactive_widgets_action.setChecked(self._hide_inactive_aux_widgets)
        self.hide_inactive_widgets_action.setToolTip(
            "When enabled, widgets (connectivity, explorers, heatmaps) of "
            "background tabs are hidden and restored with their tab."
        )
        self.hide_inactive_widgets_action.toggled.connect(
            self._on_hide_inactive_widgets_toggled
        )
        window_menu.addAction(self.hide_inactive_widgets_action)

        window_menu.addSeparator()

        show_annotation_log_action = QAction("Show Annotation Log", self)
        show_annotation_log_action.triggered.connect(self.show_annotation_log)
        window_menu.addAction(show_annotation_log_action)

        show_project_details_action = QAction("Show Project Details", self)
        show_project_details_action.triggered.connect(self.show_project_details)
        window_menu.addAction(show_project_details_action)

        # Help menu
        help_menu = menu_bar.addMenu("Help")

        # About action
        about_action = QAction("About", self)
        # Set role so macOS places this in the app menu (not Help menu)
        about_action.setMenuRole(QAction.MenuRole.AboutRole)

        def show_about_dialog():
            about_dialog = QDialog(self)
            about_dialog.setWindowTitle("About BigClust GUI")
            about_dialog.setModal(True)
            layout = QVBoxLayout()
            about_label = QLabel(
                "BigClust<br><br>"
                f"Version {__version__}<br>"
                "A graphical interface for inspecting large clusterings.<br><br>"
                "For more information visit<br>"
                '<a href="https://github.com/flyconnectome/bigclust2">https://github.com/flyconnectome/bigclust2</a>'
            )
            about_label.setAlignment(Qt.AlignCenter)
            layout.addWidget(about_label)
            ok_button = QPushButton("OK")
            ok_button.clicked.connect(about_dialog.accept)
            layout.addWidget(ok_button, alignment=Qt.AlignCenter)
            about_dialog.setLayout(layout)
            about_dialog.exec()

        about_action.triggered.connect(show_about_dialog)
        help_menu.addAction(about_action)

        help_menu.addSeparator()

        github_repo_action = QAction("GitHub Repository", self)
        github_repo_action.triggered.connect(
            lambda: QDesktopServices.openUrl(QUrl("https://github.com/flyconnectome/bigclust2"))
        )
        help_menu.addAction(github_repo_action)

        report_problem_action = QAction("Report a Problem", self)
        report_problem_action.triggered.connect(
            lambda: QDesktopServices.openUrl(QUrl("https://github.com/flyconnectome/bigclust2/issues"))
        )
        help_menu.addAction(report_problem_action)

        help_menu.addSeparator()

        # Debug submenu
        debug_menu = help_menu.addMenu("Debug")

        self.debug_scatter_action = QAction("Scatter", self)
        self.debug_scatter_action.setCheckable(True)
        self.debug_scatter_action.toggled.connect(
            lambda checked: setattr(self.current_view().fig_scatter, "debug", checked)
        )

        self.debug_viewer_action = QAction("Viewer", self)
        self.debug_viewer_action.setCheckable(True)
        self.debug_viewer_action.toggled.connect(
            lambda checked: setattr(self.current_view().ngl_viewer, "debug", checked)
        )

        self.debug_all_action = QAction("All", self)
        self.debug_all_action.setCheckable(True)
        self.debug_all_action.toggled.connect(
            lambda checked: (
                self.debug_scatter_action.setChecked(checked),
                self.debug_viewer_action.setChecked(checked),
            )
        )

        debug_menu.addAction(self.debug_all_action)
        debug_menu.addSeparator()
        debug_menu.addAction(self.debug_scatter_action)
        debug_menu.addAction(self.debug_viewer_action)

        help_menu.addSeparator()

        keyboard_shortcuts_action = QAction("Keyboard Shortcuts", self)
        keyboard_shortcuts_action.triggered.connect(self.show_keyboard_shortcuts)
        help_menu.addAction(keyboard_shortcuts_action)

        # Cmd/Ctrl+1..9 switch between tabs (9 jumps to the last tab).
        for i in range(1, 10):
            shortcut = QShortcut(QKeySequence(f"Ctrl+{i}"), self)
            shortcut.activated.connect(
                lambda idx=i: self._activate_tab_by_number(idx)
            )

        # The first tab was added before the menus existed; sync window title,
        # action states and status bar to it now.
        self._on_current_tab_changed(self._tabs.currentIndex())

    def show_keyboard_shortcuts(self):
        """Show a dialog listing all keyboard shortcuts."""

        def _make_combo(keys):
            w = QWidget()
            lay = QHBoxLayout(w)
            lay.setContentsMargins(0, 0, 0, 0)
            lay.setSpacing(3)
            first = True
            for key in keys:
                if key == "/":
                    sep = QLabel("/")
                    sep.setStyleSheet("color: #999; font-size: 11px;")
                    lay.addWidget(sep)
                    first = True
                    continue
                if not first:
                    plus = QLabel("+")
                    plus.setStyleSheet("color: #666; font-size: 11px; padding: 0 1px;")
                    lay.addWidget(plus)
                first = False
                if key.startswith("mouse:"):
                    spec = key[6:]
                    double = spec.startswith("double-")
                    hold = spec.endswith("-hold")
                    area = spec.replace("double-", "").replace("-hold", "")
                    lay.addWidget(_MouseGlyph(area=area, hold=hold, double=double))
                else:
                    lay.addWidget(_KeyBadge(key))
            lay.addStretch()
            w.setFixedWidth(200)
            return w

        def _add_section(layout, title):
            lbl = QLabel(f"<b>{title}</b>")
            lbl.setContentsMargins(4, 8, 4, 2)
            layout.addWidget(lbl)
            line = QFrame()
            line.setFrameShape(QFrame.HLine)
            line.setStyleSheet("color: #ddd;")
            layout.addWidget(line)

        def _add_row(layout, keys, description):
            row = QWidget()
            rl = QHBoxLayout(row)
            rl.setContentsMargins(4, 2, 4, 2)
            rl.setSpacing(12)
            rl.setAlignment(Qt.AlignVCenter)
            combo = _make_combo(keys)
            rl.addWidget(combo)
            desc = QLabel(description)
            desc.setStyleSheet("font-size: 12px;")
            rl.addWidget(desc, 1)
            layout.addWidget(row)

        dialog = QDialog(self)
        dialog.setWindowTitle("Keyboard Shortcuts")
        dialog.setModal(False)
        dialog.resize(520, 560)

        dlg_layout = QVBoxLayout(dialog)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        content = QWidget()
        cl = QVBoxLayout(content)
        cl.setSpacing(1)
        cl.setContentsMargins(8, 8, 8, 8)

        _add_section(cl, "Scatterplot")
        _add_row(cl, ["mouse:left"], "Move the view")
        _add_row(cl, ["⇧", "mouse:left"], "Draw a selection box")
        _add_row(cl, ["⇧", "mouse:left", "⌘"], "Add box selection to current selection")
        _add_row(cl, ["⇧", "⌃", "mouse:left"], "Draw a lasso selection")
        _add_row(cl, ["⇧", "⌃", "mouse:left", "⌘"], "Add lasso selection to current selection")
        _add_row(cl, ["Esc"], "Deselect all points")
        _add_row(cl, ["C"], "Toggle the control panel")
        _add_row(cl, ["L"], "Toggle labels")
        _add_row(cl, ["←", "/", "→"], "Increase / decrease label font size")
        _add_row(cl, ["↑", "/", "↓"], "Increase / decrease marker size")
        _add_row(cl, ["Space"], "Cycle through embeddings")
        _add_row(cl, ["mouse:double-left"], "Highlight points with the same label")
        _add_row(cl, ["⇧", "mouse:double-left"], "Select points with the same label")
        _add_row(cl, ["⌘", "⇧", "mouse:double-left"], "Add same-label points to selection")

        _add_section(cl, "Selection")
        _add_row(cl, ["⌘", "A"], "Select all points")
        _add_row(cl, ["⌘", "I"], "Invert the selection")
        _add_row(cl, ["⌘", "+"], "Grow the selection (add nearest points)")
        _add_row(cl, ["⌘", "−"], "Shrink the selection (undo last grow)")
        _add_row(cl, ["⌘", "C"], "Copy selected IDs to the clipboard")

        _add_section(cl, "Tabs")
        _add_row(cl, ["⌘", "T"], "Open a new tab")
        _add_row(cl, ["⌘", "W"], "Close the current tab")
        _add_row(cl, ["⌘", "1-9"], "Switch to the n-th tab (9 = last tab)")

        _add_section(cl, "3D Viewer")
        _add_row(cl, ["mouse:left-hold"], "Rotate the view")
        _add_row(cl, ["mouse:middle-hold"], "Pan")
        _add_row(cl, ["mouse:scroll"], "Zoom in and out")
        _add_row(cl, ["C"], "Toggle the legend")
        _add_row(cl, ["1"], "Align view: front")
        _add_row(cl, ["2"], "Align view: side")
        _add_row(cl, ["3"], "Align view: top")

        cl.addStretch()
        scroll.setWidget(content)
        dlg_layout.addWidget(scroll)

        ok_btn = QPushButton("OK")
        ok_btn.clicked.connect(dialog.accept)
        dlg_layout.addWidget(ok_btn, alignment=Qt.AlignCenter)

        dialog.exec()

    def open_new_window(self):
        """Open a second top-level BigClust window."""
        window = MainWindow()
        window.move(self.pos() + QPoint(40, 40))
        window.show()

    def add_new_tab(self, title="untitled", switch=True, view=None):
        """Create (or adopt) a view and add it as a new tab."""
        if view is None:
            view = MainWidget()
        view.view_title = title

        # Assign a per-view accent color; adopted views keep theirs.
        if view.accent_color is None:
            color, dot = _TAB_ACCENTS[next(_NEXT_TAB_ACCENT) % len(_TAB_ACCENTS)]
            view.accent_color = color
            view.accent_dot = dot

        index = self._tabs.addTab(view, title)
        self._tabs.setTabIcon(index, _make_dot_icon(view.accent_color))
        if switch:
            self._tabs.setCurrentIndex(index)
        return view

    def set_view_title(self, view, title, tooltip=None):
        """Set a view's title and mirror it on its tab, its widgets and the window."""
        if view is None:
            return
        view.view_title = title
        index = self._tabs.indexOf(view)
        if index >= 0:
            self._tabs.setTabText(index, title)
            self._tabs.setTabToolTip(index, tooltip if tooltip else title)
        for widget in view.aux_widgets():
            self._decorate_aux_widget_title(view, widget)
        self._sync_window_title()

    def _sync_window_title(self):
        view = self.current_view()
        title = getattr(view, "view_title", "") if view is not None else ""
        dot = getattr(view, "accent_dot", "") if view is not None else ""
        prefix = f"{dot} " if dot else ""
        if title and title != "untitled":
            self.setWindowTitle(f"{prefix}BigClust - {title}")
        else:
            self.setWindowTitle(f"{prefix}BigClust")

    def _on_current_tab_changed(self, index):
        """Sync window chrome (title, actions, status bar) to the active tab."""
        _dismiss_active_popup()
        view = self._tabs.widget(index) if index >= 0 else None
        self._sync_window_title()
        self._update_view_actions()
        self._refresh_window_menu_actions()
        self._sync_actions_to_view(view)

        # Hand keyboard focus to the canvas. Qt does not reliably reassign
        # focus when the tab holding the focused widget goes away (e.g.
        # Cmd+W while typing in a text field), which would leave the global
        # key bindings dead until the user clicks a focusable widget.
        if view is not None:
            view.focus_canvas()
            self._reassert_native_key_focus()

        # Only the active tab's auxiliary widgets are shown.
        self._update_aux_widget_visibility()

        # Swap in the active view's selection counter.
        status_bar = getattr(self, "status_bar", None)
        if status_bar is not None:
            prev = self._active_selection_counter
            counter = getattr(view, "selection_counter", None)
            if prev is not None and prev is not counter:
                try:
                    status_bar.removeWidget(prev)
                except RuntimeError:
                    pass
            if counter is not None and counter is not prev:
                status_bar.addPermanentWidget(counter)
                counter.show()
            self._active_selection_counter = counter

        # Reflect the newly active view's embedding in the status bar.
        self._update_embedding_status()

        # A freshly exposed view may hold a stale frame; force one render.
        # Skip while hidden (e.g. during startup) - rendering to a canvas
        # whose surface doesn't exist yet can upset the wgpu backend.
        if view is not None and self.isVisible():
            try:
                view.resize_figures()
                view.fig_scatter.force_single_render()
                view.ngl_viewer.force_single_render()
            except Exception as e:
                logger.debug(f"Failed to refresh canvases after tab switch: {e}")

    def _update_embedding_status(self):
        """Sync the status-bar embedding indicator with the active view.

        Mirrors the controls' "Active embedding" selector: shows the active
        embedding's name and is hidden unless the view has more than one
        embedding to switch between. Safe to call from any view's controls;
        it always reads the currently visible view.
        """
        label = getattr(self, "embedding_status_label", None)
        if label is None:
            return
        view = self.current_view()
        fig = getattr(view, "fig_scatter", None) if view is not None else None
        entries = getattr(fig, "embedding_entries", None) or []
        active = getattr(fig, "active_embedding", None)
        if len(entries) > 1 and active is not None and 0 <= active < len(entries):
            name = entries[active].get("name", f"#{active + 1}")
            label.setText(f"Embedding: {name}  ")
            label.show()
        else:
            label.clear()
            label.hide()

    def _reassert_native_key_focus(self):
        """Point the OS-level key routing back at this window.

        The wgpu canvases force native window handles onto parts of the
        widget tree, so a text field can live inside a native subwindow.
        Typing there makes that subwindow's view the macOS first responder,
        and Qt-side focus changes do not move it back. When the subwindow
        dies with its tab, the first responder dangles and macOS drops all
        key events before Qt sees them. Re-activating the (already active)
        window makes its top-level content view the first responder again,
        restoring delivery to Qt's focus widget.
        """
        handle = self.windowHandle()
        if handle is not None and self.isActiveWindow():
            handle.requestActivate()

    def _sync_actions_to_view(self, view):
        """Mirror the per-view checkable menu actions to the given view's state."""
        if view is None:
            return
        fig = getattr(view, "fig_scatter", None)
        ngl = getattr(view, "ngl_viewer", None)

        states = [
            ("sync_viewer_action", bool(getattr(fig, "_viewer_sync_enabled", True))),
            ("show_hoverinfo_action", not getattr(fig, "_hide_hover", False)),
            ("debug_scatter_action", bool(getattr(fig, "debug", False))),
            ("debug_viewer_action", bool(getattr(ngl, "debug", False))),
        ]
        for name, checked in states:
            action = getattr(self, name, None)
            if action is None:
                continue
            action.blockSignals(True)
            action.setChecked(checked)
            action.blockSignals(False)

    def _register_aux_widget(self, view, widget):
        """Mark a widget as owned by a view and decorate its window title."""
        widget._owner_view = view
        widget._visible_in_tab = True
        if not hasattr(widget, "_base_window_title"):
            widget._base_window_title = widget.windowTitle()
        self._decorate_aux_widget_title(view, widget)
        # Lets eventFilter() detect user closes (and connectivity unsyncs).
        widget.installEventFilter(self)

    def _decorate_aux_widget_title(self, view, widget):
        """Prefix a widget's title with its view's accent dot and tab title."""
        base = getattr(widget, "_base_window_title", widget.windowTitle())
        dot = getattr(view, "accent_dot", "")
        title = getattr(view, "view_title", "")
        prefix = f"{dot} " if dot else ""
        suffix = f" — {title}" if title else ""
        widget.setWindowTitle(f"{prefix}{base}{suffix}")

    def _present_aux_widget(self, widget):
        """(Re-)show a cached auxiliary widget and bring it to the front."""
        widget._visible_in_tab = True
        widget.showNormal()
        widget.show()
        widget.raise_()
        widget.activateWindow()

    def _update_aux_widget_visibility(self):
        """Show only the active tab's auxiliary widgets (if so configured).

        Widgets of background tabs are hidden; switching back restores those
        that were open. `_visible_in_tab` tracks user intent: set on
        creation/re-show, cleared when the user closes the widget (via the
        close-event filter) - hiding here does not touch it. With the
        "Hide Widgets of Inactive Tabs" preference off, every tab's open
        widgets stay visible.
        """
        current = self.current_view()
        for view in self.views():
            if view is None:
                continue
            for widget in view.aux_widgets():
                try:
                    if view is current or not self._hide_inactive_aux_widgets:
                        if getattr(widget, "_visible_in_tab", False) and not widget.isVisible():
                            widget.show()
                    elif widget.isVisible():
                        widget.hide()
                except RuntimeError:
                    continue

    def _on_hide_inactive_widgets_toggled(self, checked):
        """Apply and persist the hide-inactive-tab-widgets preference globally."""
        try:
            self.settings.setValue("auxWidgets/hideInactiveTabWidgets", checked)
        except Exception:
            pass

        # The preference is global; mirror it into every open window.
        for win in list(_OPEN_WINDOWS):
            if not isinstance(win, MainWindow):
                continue
            try:
                win._hide_inactive_aux_widgets = checked
                action = getattr(win, "hide_inactive_widgets_action", None)
                if action is not None and action.isChecked() != checked:
                    action.blockSignals(True)
                    action.setChecked(checked)
                    action.blockSignals(False)
                win._update_aux_widget_visibility()
            except RuntimeError:
                continue

    def _transfer_aux_widgets(self, view, new_window):
        """Re-home a detached view's auxiliary widgets to its new window.

        Without this, widgets stay Qt-children of the old window and get
        destroyed with it even though their view lives on.
        """
        widgets = view.aux_widgets()

        # The annotation dialog is excluded from aux_widgets() but must move too.
        dialog = view._annotation_dialog
        if dialog is not None:
            try:
                dialog.objectName()
                widgets.append(dialog)
                # Annotation logging is per-window; rewire to the new one.
                try:
                    dialog.annotations_logged.disconnect(self._log_annotation_entries)
                except (RuntimeError, TypeError):
                    pass
                dialog.annotations_logged.connect(new_window._log_annotation_entries)
            except RuntimeError:
                pass

        for widget in widgets:
            try:
                if widget.parent() is self:
                    visible = widget.isVisible()
                    widget.setParent(new_window, widget.windowFlags())
                    if visible:
                        widget.show()
                # The close-event filter lives on the owning window.
                widget.removeEventFilter(self)
                widget.installEventFilter(new_window)
            except RuntimeError:
                continue

    def _refresh_window_menu_actions(self):
        """Update enablement of the tab/window organization actions."""
        if getattr(self, "detach_tab_action", None) is not None:
            self.detach_tab_action.setEnabled(self._tabs.count() > 1)
        if getattr(self, "merge_window_action", None) is not None:
            self.merge_window_action.setEnabled(self.parent_main_window() is not None)

    def _activate_tab_by_number(self, number):
        """Switch to the n-th tab; 9 jumps to the last tab."""
        count = self._tabs.count()
        index = count - 1 if number == 9 else number - 1
        if 0 <= index < count:
            self._tabs.setCurrentIndex(index)

    def _on_tab_close_requested(self, index):
        view = self._tabs.widget(index)
        if view is not None:
            self.close_view(view)

    def close_current_tab(self):
        view = self.current_view()
        if view is not None:
            self.close_view(view)

    def close_view(self, view):
        """Tear down and remove a view; closing the last tab closes the window."""
        _dismiss_active_popup()
        if self._tabs.count() <= 1:
            # closeEvent takes care of tearing down the remaining view.
            self.close()
            return

        self._teardown_view(view)
        index = self._tabs.indexOf(view)
        if index >= 0:
            self._tabs.removeTab(index)
        view.deleteLater()

    def _teardown_view(self, view):
        """Stop a view's render backends and dispose its auxiliary widgets."""
        try:
            view.teardown_rendering()
        except Exception as e:
            logger.debug(f"Failed to tear down view rendering: {e}")
        self._dispose_view_aux_widgets(view)

    def _on_tab_detach_requested(self, index, global_pos):
        """Move a tab into its own window (tab dragged out of the bar)."""
        self.detach_view(self._tabs.widget(index), global_pos)

    def detach_current_tab(self):
        """Move the current tab into its own window."""
        return self.detach_view(self.current_view())

    def detach_view(self, view, global_pos=None):
        """Move a view out of this window into a new window of its own."""
        if view is None or self._tabs.count() <= 1:
            return None
        index = self._tabs.indexOf(view)
        if index < 0:
            return None

        _dismiss_active_popup()
        self._tabs.removeTab(index)
        window = MainWindow(adopt_view=view)
        window._parent_window = self
        self._transfer_aux_widgets(view, window)
        if global_pos is not None:
            window.move(global_pos - QPoint(100, 20))
        else:
            window.move(self.pos() + QPoint(40, 40))
        window.show()
        return window

    def parent_main_window(self):
        """The window this one was detached from, if it still exists."""
        parent = self._parent_window
        if parent is None or parent is self or parent not in _OPEN_WINDOWS:
            return None
        return parent

    def merge_into_parent_window(self):
        """Move this window's tabs back into the parent window, then close."""
        parent = self.parent_main_window()
        if parent is None:
            return

        _dismiss_active_popup()
        views = self.views()
        for view in views:
            index = self._tabs.indexOf(view)
            if index < 0:
                continue
            self._tabs.removeTab(index)
            parent.add_new_tab(title=view.view_title, switch=False, view=view)
            self._transfer_aux_widgets(view, parent)

        if views:
            parent._tabs.setCurrentWidget(views[0])
        parent.show()
        parent.raise_()
        parent.activateWindow()
        self.close()

    def _can_open_connectivity_table(self):
        """Whether the connectivity table can be opened for the current project."""
        if not hasattr(self, "_data") or not isinstance(self._data, dict):
            return False

        features = self._data.get("features")
        if (
            features is None
            or not hasattr(features, "shape")
            or len(features.shape) != 2
        ):
            return False

        return features.shape[0] > 0 and features.shape[1] > 0

    def _can_open_distances_table(self):
        """Whether the distance heatmap can be opened for the current project."""
        if not hasattr(self, "_data") or not isinstance(self._data, dict):
            return False

        dists = self._data.get("distances")
        if dists is None or not hasattr(dists, "shape") or len(dists.shape) != 2:
            return False

        return dists.shape[0] == dists.shape[1]

    def _can_open_feature_explorer(self):
        """Whether the feature explorer can be opened for the current project."""
        if not hasattr(self, "_data") or not isinstance(self._data, dict):
            return False

        features = self._data.get("features")
        meta = self._data.get("meta")
        if not isinstance(features, pd.DataFrame) or not isinstance(meta, pd.DataFrame):
            return False

        return (not features.empty) and (not meta.empty)

    def _can_open_meta_explorer(self):
        if not hasattr(self, "_data") or not isinstance(self._data, dict):
            return False

        meta = self._data.get("meta")
        if not isinstance(meta, pd.DataFrame):
            return False

        return not meta.empty

    def _update_view_actions(self):
        """Update View menu action states."""
        if self.connectivity_table_action is not None:
            self.connectivity_table_action.setEnabled(
                self._can_open_connectivity_table()
            )
        if self.distances_table_action is not None:
            self.distances_table_action.setEnabled(self._can_open_distances_table())
        if self.feature_explorer_action is not None:
            self.feature_explorer_action.setEnabled(self._can_open_feature_explorer())
        if self.meta_explorer_action is not None:
            self.meta_explorer_action.setEnabled(self._can_open_meta_explorer())

    def _active_embedding_key(self, view):
        """Return ``(key, name)`` identifying the active embedding for a view.

        `key` keys the per-embedding widget caches (the active embedding index,
        or a sentinel when there are no entries). `name` is the embedding's
        display name, used in widget titles - only set when more than one
        embedding exists, so single-embedding projects keep plain titles.
        """
        fig = getattr(view, "fig_scatter", None)
        idx = getattr(fig, "active_embedding", None) if fig is not None else None
        entries = (getattr(fig, "embedding_entries", None) or []) if fig is not None else []
        name = None
        if idx is not None and 0 <= idx < len(entries) and len(entries) > 1:
            name = str(entries[idx].get("name", ""))
        key = idx if idx is not None else "__default__"
        return key, name

    def show_connectivity_table(self):
        """Open the connectivity table for the active embedding of the current view.

        Widgets are cached per embedding, so switching embeddings and reopening
        gives each embedding its own table (showing that embedding's features).
        """
        if not self._can_open_connectivity_table():
            return

        view = self.current_view()
        if view is None:
            return

        key, emb_name = self._active_embedding_key(view)

        existing = view._connectivity_widgets.get(key)
        if existing is not None:
            try:
                self._sync_connectivity_widget(view, existing)
                # Reuse this embedding's widget and bring it to front.
                self._present_aux_widget(existing)
                return
            except RuntimeError:
                # Underlying Qt object was deleted elsewhere; rebuild on demand.
                view._connectivity_widgets.pop(key, None)

        data = view._data if isinstance(view._data, dict) else {}
        features = data.get("features")
        meta_data = data.get("meta")
        fig = view.fig_scatter

        if features is None or meta_data is None:
            return

        title = "Connectivity widget"
        if emb_name:
            title = f"Connectivity widget — {emb_name}"

        widget = ConnectivityTable(
            features,
            figure=fig,
            meta_data=meta_data,
            title=title,
            parent=self,
        )
        self._register_aux_widget(view, widget)
        self._sync_connectivity_widget(view, widget)
        widget.show()

        # Keep a strong reference so the window can be reopened instead of recreated.
        view._connectivity_widgets[key] = widget
        widget.destroyed.connect(
            lambda _obj=None, v=view, k=key: v._connectivity_widgets.pop(k, None)
        )
        widget.destroyed.connect(
            lambda _obj=None, w=widget, f=fig: f.unsync_widget(w)
        )

    def show_feature_explorer(self):
        """Open the Feature Explorer for the active embedding of the current view.

        Cached per embedding so each embedding gets its own explorer (showing
        that embedding's features).
        """
        if not self._can_open_feature_explorer():
            return

        view = self.current_view()
        if view is None:
            return

        key, emb_name = self._active_embedding_key(view)

        existing = view._feature_explorer_widgets.get(key)
        if existing is not None:
            try:
                self._present_aux_widget(existing)
                return
            except RuntimeError:
                view._feature_explorer_widgets.pop(key, None)

        data = view._data if isinstance(view._data, dict) else {}
        features = data.get("features")
        meta_data = data.get("meta")
        if features is None or meta_data is None:
            return

        widget = FeatureExplorerWidget(
            metadata=meta_data,
            features=features,
            figure=view.fig_scatter,
            parent=None,
        )
        widget.setAttribute(Qt.WA_DeleteOnClose, True)
        widget.setWindowFlag(Qt.Window, True)
        title = "Feature Explorer"
        if emb_name:
            title = f"Feature Explorer — {emb_name}"
        widget.setWindowTitle(title)
        self._register_aux_widget(view, widget)
        # widget.resize(1100, 700)
        widget.show()

        view._feature_explorer_widgets[key] = widget
        widget.destroyed.connect(
            lambda _obj=None, v=view, k=key: v._feature_explorer_widgets.pop(k, None)
        )

    def show_meta_explorer(self):
        """Open the meta data explorer dialog for the current view's project."""
        if not self._can_open_meta_explorer():
            return

        view = self.current_view()
        if view is None:
            return

        existing = view._meta_explorer_dialog
        if existing is not None:
            try:
                self._present_aux_widget(existing)
                return
            except RuntimeError:
                view._meta_explorer_dialog = None

        data = view._data if isinstance(view._data, dict) else {}
        meta_data = data.get("meta")
        if meta_data is None:
            return

        dialog = MetaExplorerDialog(
            meta_data,
            figure=view.fig_scatter,
            parent=self,
        )
        self._register_aux_widget(view, dialog)
        dialog.setAttribute(Qt.WA_DeleteOnClose, True)
        dialog.show()

        view._meta_explorer_dialog = dialog
        dialog.destroyed.connect(
            lambda _obj=None, v=view: setattr(v, "_meta_explorer_dialog", None)
        )

    def _sync_connectivity_widget(self, view, widget):
        """Sync a connectivity widget to figure selection (idempotent)."""
        if widget is None:
            return
        view.fig_scatter.sync_widget(widget)

    def _unsync_connectivity_widget(self, view, widget):
        """Unsync a connectivity widget to avoid heavy updates while hidden."""
        if widget is None:
            return
        try:
            view.fig_scatter.unsync_widget(widget)
        except RuntimeError:
            # Qt object may already be deleted.
            pass

    def _dispose_view_aux_widgets(self, view):
        """Dispose all auxiliary widgets owned by a view (project context change)."""
        if view is None:
            return
        self._dispose_connectivity_widget(view)
        self._dispose_feature_explorer_widget(view)
        self._dispose_annotation_dialog(view)
        self._dispose_meta_explorer_dialog(view)
        self._dispose_distance_widgets(view)

    def _dispose_connectivity_widget(self, view):
        """Dispose all of a view's connectivity widgets."""
        widgets = list(view._connectivity_widgets.values())
        view._connectivity_widgets = {}
        for widget in widgets:
            self._unsync_connectivity_widget(view, widget)
            try:
                widget.removeEventFilter(self)
                widget.close()
                widget.deleteLater()
            except RuntimeError:
                # Qt object may already be deleted.
                pass

    def _dispose_feature_explorer_widget(self, view):
        """Dispose all of a view's feature explorer widgets."""
        widgets = list(view._feature_explorer_widgets.values())
        view._feature_explorer_widgets = {}
        for widget in widgets:
            try:
                widget.close()
                widget.deleteLater()
            except RuntimeError:
                # Qt object may already be deleted.
                pass

    def _dispose_meta_explorer_dialog(self, view):
        """Dispose a view's meta explorer dialog."""
        dialog = view._meta_explorer_dialog
        if dialog is None:
            return

        view._meta_explorer_dialog = None
        try:
            dialog.close()
            dialog.deleteLater()
        except RuntimeError:
            # Qt object may already be deleted.
            pass

    def _dispose_distance_widgets(self, view):
        """Dispose all of a view's distance heatmap widgets."""
        widgets = list(view._distance_widgets)
        view._distance_widgets = []
        for widget in widgets:
            try:
                view.fig_scatter.unsync_widget(widget)
            except Exception:
                pass
            try:
                widget.close()
                widget.deleteLater()
            except RuntimeError:
                # Qt object may already be deleted.
                pass

    def _selected_annotation_records(self):
        """Build annotation dialog selection records from scatter selection."""
        fig = self.current_view().fig_scatter
        selected_meta = getattr(fig, "selected_meta", None)
        if selected_meta is None or selected_meta.empty:
            return []

        if "id" not in selected_meta.columns or "dataset" not in selected_meta.columns:
            return []

        records = []
        for neuron_id, dataset in zip(
            selected_meta["id"].tolist(),
            selected_meta["dataset"].tolist(),
        ):
            records.append(
                SelectionRecord(neuron_id=int(neuron_id), dataset=str(dataset))
            )
        return records

    def _project_annotation_datasets(self):
        """Return all dataset names available in the currently loaded project."""
        meta = self._project_meta_data()
        if meta is None or "dataset" not in meta.columns:
            return []

        values = []
        for dataset in meta["dataset"].dropna().tolist():
            text = str(dataset).strip()
            if text:
                values.append(text)
        return sorted(set(values))

    def show_annotation_dialog(self):
        """Show a single reusable annotation dialog for the current selection."""
        view = self.current_view()
        if view is None:
            return

        selection = self._selected_annotation_records()
        project_datasets = self._project_annotation_datasets()
        if not selection:
            view.fig_scatter.show_message(
                "No points selected", color="red", duration=2
            )
            return

        existing = view._annotation_dialog
        if existing is not None:
            try:
                existing.set_selection(selection, project_datasets=project_datasets)
                existing.showNormal()
                existing.show()
                existing.raise_()
                existing.activateWindow()
                return
            except RuntimeError:
                # Underlying Qt object was deleted elsewhere; rebuild on demand.
                view._annotation_dialog = None

        dialog = AnnotationDialog(
            selection=selection,
            parent=self,
            project_datasets=project_datasets,
        )
        dialog._owner_view = view
        dialog.annotations_logged.connect(self._log_annotation_entries)
        dialog.show()
        dialog.raise_()
        dialog.activateWindow()

        view._annotation_dialog = dialog
        dialog.destroyed.connect(
            lambda _obj=None, v=view: setattr(v, "_annotation_dialog", None)
        )

    def on_annotation_submit_result(self, message, changed_neurons=None):
        """Receive async submit result message and forward to UI + console."""
        text = str(message)
        print(text)
        self.annotation_submit_result_received.emit(text)
        if changed_neurons:
            self.annotation_changed.emit(changed_neurons)

    def _handle_annotation_submit_result(self, message):
        """Show user-facing submit result status on the scatter figure."""
        text = str(message)
        lowered = text.lower()
        is_error = ("error" in lowered) or ("failed" in lowered)
        color = "red" if is_error else "green"

        try:
            fig = self.current_view().fig_scatter
            fig.show_message(text, color=color, duration=3)
        except Exception as e:
            logger.debug(f"Failed to show annotation submit status message: {e}")

    def _handle_annotation_changed(self, changed_neurons):
        """Highlight labels for neurons whose annotations were changed."""
        try:
            fig = self.current_view().fig_scatter
            ids = getattr(fig, "ids", None)
            datasets = getattr(fig, "datasets", None)
            if ids is None or len(ids) == 0:
                return

            matches = []
            for entry in changed_neurons:
                if isinstance(entry, dict):
                    neuron_id = entry.get("id")
                    dataset = entry.get("dataset")
                elif isinstance(entry, tuple) and len(entry) == 2:
                    dataset, neuron_id = entry
                else:
                    continue

                if datasets is None:
                    idx = np.where(ids == neuron_id)[0]
                else:
                    idx = np.where((ids == neuron_id) & (datasets == dataset))[0]
                matches.extend(idx.tolist())

            if matches:
                fig.set_label_color(matches, "#ff69b4")
        except Exception as e:
            logger.debug(f"Failed to highlight changed annotation labels: {e}")

    def _log_annotation_entries(self, entries):
        """Store per-window annotation log entries for successful dataset writes."""
        if not isinstance(entries, list):
            return
        self._annotation_log.extend(entries)
        self._write_annotation_log_entries(entries)

    def _write_annotation_log_entries(self, entries):
        try:
            with self._annotation_log_file.open("a", encoding="utf-8") as fh:
                for entry in entries:
                    fh.write(json.dumps(entry, ensure_ascii=False))
                    fh.write("\n")
        except Exception as e:
            logger.debug(f"Failed to write annotation log file: {e}")

    def show_annotation_log(self):
        """Open a dialog showing the current window's annotation log."""
        dialog = AnnotationLogDialog(self, entries=self._annotation_log)
        dialog.exec()

    def show_project_details(self):
        """Open a dialog summarizing the current project's metadata."""
        loader = self._current_project_loader
        data = self._data
        summary = build_project_summary(loader, data)
        try:
            info = loader.info if loader is not None else None
        except Exception:
            info = None
        ProjectDetailsDialog(self, summary=summary, info=info).exec()

    @property
    def annotation_log(self):
        """Return the per-window annotation log."""
        return list(self._annotation_log)

    def _dispose_annotation_dialog(self, view):
        """Dispose a view's annotation dialog."""
        dialog = view._annotation_dialog
        if dialog is None:
            return

        view._annotation_dialog = None
        try:
            dialog.close()
            dialog.deleteLater()
        except RuntimeError:
            # Qt object may already be deleted.
            pass

    def show_distances_table(self):
        """Open the pairwise distance heatmap widget for the current view's project."""
        if not self._can_open_distances_table():
            return

        view = self.current_view()
        if view is None:
            return

        data = view._data if isinstance(view._data, dict) else {}
        distances = data.get("distances")
        meta_data = data.get("meta")
        fig = view.fig_scatter

        if distances is None or meta_data is None:
            return

        _key, emb_name = self._active_embedding_key(view)
        title = "Distance heatmap"
        if emb_name:
            title = f"Distance heatmap — {emb_name}"

        widget = DistancesTable(
            distances,
            figure=fig,
            meta_data=meta_data,
            title=title,
            parent=self,
        )
        self._register_aux_widget(view, widget)
        fig.sync_widget(widget)
        widget.show()

        view._distance_widgets.append(widget)
        widget.destroyed.connect(
            lambda _obj=None, w=widget, v=view: (
                v._distance_widgets.remove(w)
                if w in v._distance_widgets
                else None
            )
        )

    def _normalize_recent_state(self, state):
        """Normalize and validate a recent project state payload."""
        if not isinstance(state, dict):
            return None

        path = str(state.get("path", "")).strip()
        if not path:
            return None

        project_name = str(state.get("project_name", "")).strip()
        filter_expr = str(state.get("filter_expr", "")).strip()
        embedding_mode = str(state.get("embedding_mode", "")).strip()

        try:
            project_index = int(state.get("project_index", -1))
        except (TypeError, ValueError):
            project_index = -1

        source_type = str(state.get("source_type", "")).strip().lower()
        if source_type not in ("local", "remote"):
            source_type = (
                "remote" if path.startswith(("http://", "https://")) else "local"
            )

        return {
            "path": path,
            "source_type": source_type,
            "project_name": project_name,
            "project_index": project_index,
            "filter_expr": filter_expr,
            "embedding_mode": embedding_mode,
        }

    def _recent_state_key(self, state):
        """Stable identity for deduplicating recent items."""
        return (
            state.get("path", ""),
            state.get("project_name", ""),
            state.get("project_index", -1),
            state.get("filter_expr", ""),
            state.get("embedding_mode", ""),
        )

    def load_recent_projects(self):
        """Load the saved recent project list from settings."""
        raw = self.settings.value(self.RECENT_PROJECTS_KEY, "[]")
        if isinstance(raw, (list, tuple)):
            parsed = raw
        else:
            try:
                parsed = json.loads(raw)
            except Exception:
                parsed = []

        recent = []
        seen = set()
        for item in parsed:
            normalized = self._normalize_recent_state(item)
            if not normalized:
                continue
            key = self._recent_state_key(normalized)
            if key in seen:
                continue
            seen.add(key)
            recent.append(normalized)
            if len(recent) >= self.MAX_RECENT_PROJECTS:
                break
        return recent

    def save_recent_projects(self, recent):
        """Persist recent projects to settings."""
        self.settings.setValue(self.RECENT_PROJECTS_KEY, json.dumps(recent))

    def add_recent_project(self, state):
        """Insert a state at the top of the recent projects list."""
        normalized = self._normalize_recent_state(state)
        if not normalized:
            return

        recent = self.load_recent_projects()
        recent = [
            r
            for r in recent
            if self._recent_state_key(r) != self._recent_state_key(normalized)
        ]
        recent.insert(0, normalized)
        recent = recent[: self.MAX_RECENT_PROJECTS]
        self.save_recent_projects(recent)
        self.refresh_open_recent_menu()

    def clear_recent_projects(self):
        self.save_recent_projects([])
        self.refresh_open_recent_menu()

    def _recent_project_label(self, state):
        """Plain-text fallback label for recent project menu action."""
        project_name = state.get("project_name", "")
        path = state.get("path", "")
        filter_expr = state.get("filter_expr", "")

        project_part = project_name or "(unnamed project)"
        source_part = f"{path}"
        filter_part = (
            f"filters: {filter_expr}" if str(filter_expr).strip() else "filters: none"
        )
        return f"{project_part} | {source_part} | {filter_part}"

    def _recent_project_rich_label(self, state):
        """Rich-text label for recent project menu action."""
        project_name = escape(state.get("project_name", "") or "(unnamed project)")
        path = escape(state.get("path", ""))
        filter_expr = str(state.get("filter_expr", "")).strip()
        filter_part = escape(filter_expr) if filter_expr else "none"

        return (
            f"<span>{project_name}</span> "
            f"<span style='font-size: 11px; color: #666666;'>source: {path}</span> "
            f"<span style='font-size: 11px; color: #666666;'>filters: {filter_part}</span>"
        )

    def refresh_open_recent_menu(self):
        """Rebuild the File -> Open Recent submenu."""
        if self.open_recent_menu is None:
            return

        self.open_recent_menu.clear()
        recent = self.load_recent_projects()

        if not recent:
            empty_action = QAction("No recent projects", self)
            empty_action.setEnabled(False)
            self.open_recent_menu.addAction(empty_action)
            return

        for state in recent:
            # Native menus (macOS app menu) do not support custom rich-text widgets,
            # so we use plain text there and rich text elsewhere.
            if self.menuBar().isNativeMenuBar():
                action = QAction(self._recent_project_label(state), self)
                action.triggered.connect(
                    lambda _checked=False, s=state: self.open_recent_project(s)
                )
                self.open_recent_menu.addAction(action)
            else:
                rich_action = QWidgetAction(self.open_recent_menu)
                rich_action.setText(self._recent_project_label(state))
                label = QLabel(self._recent_project_rich_label(state))
                label.setTextFormat(Qt.RichText)
                label.setContentsMargins(4, 2, 4, 2)
                rich_action.setDefaultWidget(label)
                rich_action.triggered.connect(
                    lambda _checked=False, s=state: self.open_recent_project(s)
                )
                self.open_recent_menu.addAction(rich_action)

        self.open_recent_menu.addSeparator()
        clear_action = QAction("Clear Menu", self)
        clear_action.triggered.connect(self.clear_recent_projects)
        self.open_recent_menu.addAction(clear_action)

    def _select_project_from_state(self, state):
        """Resolve a saved state to a concrete project loader."""
        path = state.get("path", "")
        parsed = parse_directory(path)
        if isinstance(parsed, SingleProjectLoader):
            projects = [parsed]
        else:
            projects = list(parsed)

        if not projects:
            return None

        project_name = state.get("project_name", "")
        selected = None

        if project_name:
            for project in projects:
                if project is not None and project.name == project_name:
                    selected = project
                    break

        if selected is None:
            project_index = state.get("project_index", -1)
            if isinstance(project_index, int) and 0 <= project_index < len(projects):
                selected = projects[project_index]

        if selected is None:
            for project in projects:
                if project is not None:
                    selected = project
                    break

        if selected is None:
            return None

        filter_expr = str(state.get("filter_expr", "")).strip()
        if filter_expr:
            selected.filter_expr = filter_expr

        return selected

    def open_recent_project(self, state):
        """Load a project from a recent state entry."""
        normalized = self._normalize_recent_state(state)
        if not normalized:
            return

        try:
            project = self._select_project_from_state(normalized)
            if project is None:
                return
            self._load_project(project, normalized.get("embedding_mode", ""))
            self.add_recent_project(normalized)
        except Exception as e:
            logger.error(f"Failed to open recent project: {e}")

            # Fall back to opening the dialog with the saved state prefilled.
            dialog = OpenProjectDialog(self, initial_state=normalized)
            if dialog.exec() == QDialog.Accepted:
                self._load_project_from_dialog(dialog)

    def closeEvent(self, event):
        # A popup left open in this window would keep eating key events in
        # the remaining windows after this one is gone.
        _dismiss_active_popup()
        for view in self.views():
            if view is not None and hasattr(view, "teardown_rendering"):
                self._teardown_view(view)

        # Persist geometry, maximized state and the active view's layout
        try:
            self.settings.setValue("mainWindow/geometry", self.saveGeometry())
            self.settings.setValue("mainWindow/isMaximized", self.isMaximized())
            view = self.current_view()
            if view is not None:
                self.settings.setValue("mainWindow/layoutMode", view._layout_mode)
                ratio = view.layout_ratio()
                if ratio is not None:
                    self.settings.setValue("mainWindow/layoutRatio", ratio)
        except Exception:
            pass
        super().closeEvent(event)

    def on_reset_layout(self):
        """Reset the active view to the default stacked 50/50 arrangement."""
        view = self.current_view()
        if view is not None:
            view.show_stacked()

    def _read_layout_ratio_setting(self):
        """Read the persisted splitter ratio as a float in (0, 1), or None."""
        try:
            raw = self.settings.value("mainWindow/layoutRatio", None)
            if raw in (None, ""):
                return None
            ratio = float(raw)
            return ratio if 0.0 < ratio < 1.0 else None
        except (TypeError, ValueError):
            return None

    def _refresh_hover_columns_menu(self):
        """Rebuild the Hover Columns submenu with checkable column items."""
        if self._hover_columns_menu is None:
            return

        self._hover_columns_menu.clear()

        try:
            fig = self.current_view().fig_scatter
        except Exception:
            fig = None

        all_cols = getattr(fig, "_hover_col_names_all", None) if fig else None
        if not all_cols:
            no_data = QAction("No data loaded", self)
            no_data.setEnabled(False)
            self._hover_columns_menu.addAction(no_data)
            return

        active = set(fig.hover_col_names)

        for col in all_cols:
            checkbox = QCheckBox(col)
            checkbox.setChecked(col in active)
            checkbox.toggled.connect(
                lambda checked, c=col: self._on_hover_column_toggled(c, checked)
            )
            action = QWidgetAction(self._hover_columns_menu)
            action.setDefaultWidget(checkbox)
            self._hover_columns_menu.addAction(action)

        self._hover_columns_menu.addSeparator()

        show_all = QAction("Show All", self)
        show_all.triggered.connect(self._on_hover_show_all)
        self._hover_columns_menu.addAction(show_all)

        hide_all = QAction("Hide All", self)
        hide_all.triggered.connect(self._on_hover_hide_all)
        self._hover_columns_menu.addAction(hide_all)

    def _on_hover_column_toggled(self, col_name, checked):
        """Update hover column state and recompute hover_info immediately."""
        try:
            fig = self.current_view().fig_scatter
        except Exception:
            return

        all_cols = getattr(fig, "_hover_col_names_all", [])
        if not all_cols:
            return

        active = set(fig.hover_col_names)
        if checked:
            active.add(col_name)
        else:
            active.discard(col_name)

        valid = [c for c in all_cols if c in active]
        fig._hover_col_names_active = valid if valid != all_cols else None

    def _on_hover_show_all(self):
        """Show all metadata columns in the hover tooltip."""
        try:
            fig = self.current_view().fig_scatter
            if getattr(fig, "_hover_col_names_all", None):
                fig._hover_col_names_active = None  # None = all columns
        except Exception as e:
            logger.debug(f"Hover show all failed: {e}")

    def _on_hover_hide_all(self):
        """Hide all metadata columns from the hover tooltip."""
        try:
            fig = self.current_view().fig_scatter
            if getattr(fig, "_hover_col_names_all", None):
                fig._hover_col_names_active = []
        except Exception as e:
            logger.debug(f"Hover hide all failed: {e}")

    def _on_hover_columns_menu_hidden(self):
        """Apply the hover column selection after the Hover Columns menu closes."""
        try:
            fig = self.current_view().fig_scatter
            if getattr(fig, "metadata", None) is not None:
                fig._recompute_hover_info()
        except Exception as e:
            logger.debug(f"Hover recompute on menu close failed: {e}")

    def _refresh_grow_shrink_menu(self):
        """Rebuild the Grow/Shrink Options submenu for the current tab's figure."""
        from .. import grow_shrink as gs

        menu = self._grow_shrink_menu
        menu.clear()

        try:
            fig = self.current_view().fig_scatter
        except Exception:
            fig = None

        if fig is None or getattr(fig, "positions", None) is None:
            no_data = QAction("No data loaded", self)
            no_data.setEnabled(False)
            menu.addAction(no_data)
            return

        source, metric, step = fig._gs_resolve_settings()
        sources = gs.available_sources(fig.dists, fig.positions)

        # --- Source radio group ---
        src_labels = {
            gs.SOURCE_EMBEDDING: "Embedding (screen)",
            gs.SOURCE_KNN: "KNN graph",
            gs.SOURCE_DISTANCES: "Distance matrix",
            gs.SOURCE_FEATURES: "Feature space",
        }
        src_group = QActionGroup(self)
        src_group.setExclusive(True)
        for sid, label in src_labels.items():
            action = QAction(label, self)
            action.setCheckable(True)
            action.setEnabled(sid in sources)
            action.setChecked(sid == source)
            action.triggered.connect(lambda checked, s=sid: self._on_gs_set_source(s))
            src_group.addAction(action)
            menu.addAction(action)

        # --- Feature metric (only meaningful for the feature source) ---
        if source == gs.SOURCE_FEATURES:
            menu.addSeparator()
            metric_menu = menu.addMenu("Feature Metric")
            metric_group = QActionGroup(self)
            metric_group.setExclusive(True)
            for mname in ("euclidean", "cosine", "cityblock", "correlation"):
                action = QAction(mname, self)
                action.setCheckable(True)
                action.setChecked(mname == metric)
                action.triggered.connect(
                    lambda checked, m=mname: self._on_gs_set_metric(m)
                )
                metric_group.addAction(action)
                metric_menu.addAction(action)

        # --- Step size ---
        menu.addSeparator()
        step_menu = menu.addMenu("Step Size")
        step_group = QActionGroup(self)
        step_group.setExclusive(True)
        presets = [1, 5, 10, 50, 100]
        for preset in presets:
            action = QAction(str(preset), self)
            action.setCheckable(True)
            action.setChecked(preset == step)
            action.triggered.connect(lambda checked, n=preset: self._on_gs_set_step(n))
            step_group.addAction(action)
            step_menu.addAction(action)
        step_menu.addSeparator()
        custom = QAction(
            "Custom…" if step in presets else f"Custom… ({step})", self
        )
        custom.setCheckable(True)
        custom.setChecked(step not in presets)
        step_group.addAction(custom)
        custom.triggered.connect(self._on_gs_set_step_custom)
        step_menu.addAction(custom)

    def on_grow_selection(self):
        """Grow the current selection (Selection > Grow Selection)."""
        try:
            self.current_view().fig_scatter.grow_selection()
        except Exception as e:
            logger.debug(f"Grow Selection failed: {e}")

    def on_shrink_selection(self):
        """Shrink (reverse the last grow of) the current selection."""
        try:
            self.current_view().fig_scatter.shrink_selection()
        except Exception as e:
            logger.debug(f"Shrink Selection failed: {e}")

    def _on_gs_set_source(self, source):
        try:
            self.current_view().fig_scatter._gs_source = source
        except Exception as e:
            logger.debug(f"Set grow/shrink source failed: {e}")

    def _on_gs_set_metric(self, metric):
        try:
            self.current_view().fig_scatter._gs_metric = metric
        except Exception as e:
            logger.debug(f"Set grow/shrink metric failed: {e}")

    def _on_gs_set_step(self, step):
        try:
            self.current_view().fig_scatter._gs_step = int(step)
        except Exception as e:
            logger.debug(f"Set grow/shrink step failed: {e}")

    def _on_gs_set_step_custom(self):
        from PySide6.QtWidgets import QInputDialog

        try:
            fig = self.current_view().fig_scatter
            current = int(getattr(fig, "_gs_step", 10))
            value, ok = QInputDialog.getInt(
                self, "Step Size", "Points per grow/shrink:", current, 1, 100000, 1
            )
            if ok:
                fig._gs_step = int(value)
        except Exception as e:
            logger.debug(f"Set custom grow/shrink step failed: {e}")

    def _confirm_large_selection(self, n):
        """Ask the user to confirm a selection of `n` neurons if it is large."""
        if n <= LARGE_SELECTION_THRESHOLD:
            return True
        dialog = QDialog(self)
        dialog.setWindowTitle("Confirm Selection")
        dialog.setModal(True)

        layout = QVBoxLayout(dialog)
        layout.setContentsMargins(20, 18, 20, 14)
        layout.setSpacing(14)

        msg = QLabel(f"Confirm selection of <b>{n:,}</b> neurons?")
        msg.setWordWrap(True)
        layout.addWidget(msg)

        buttons = QDialogButtonBox()
        no_btn = buttons.addButton(QDialogButtonBox.No)
        yes_btn = buttons.addButton(QDialogButtonBox.Yes)
        no_btn.clicked.connect(dialog.reject)
        yes_btn.clicked.connect(dialog.accept)
        no_btn.setDefault(True)
        layout.addWidget(buttons)

        dialog.setMinimumWidth(300)
        return dialog.exec() == QDialog.Accepted

    def on_select_all(self):
        fig = self.current_view().fig_scatter
        indices = np.arange(len(getattr(fig, "ids", [])))
        if not self._confirm_large_selection(len(indices)):
            return
        fig.selected = indices

    def on_deselect_all(self):
        try:
            fig = self.current_view().fig_scatter
            if hasattr(fig, "deselect_all"):
                fig.deselect_all()
            elif hasattr(fig, "clear_selection"):
                fig.clear_selection()
            else:
                logger.info("Deselect All not supported by current figure")
        except Exception as e:
            logger.debug(f"Deselect All failed: {e}")

    def on_invert_selection(self):
        try:
            fig = self.current_view().fig_scatter
            all_indices = np.arange(len(getattr(fig, "ids", [])))
            current = fig.selected
            current_set = set(np.asarray(current, dtype=int).tolist()) if current is not None and len(current) > 0 else set()
            inverted = np.array([i for i in all_indices if i not in current_set], dtype=int)
            if not self._confirm_large_selection(len(inverted)):
                return
            fig.selected = inverted
        except Exception as e:
            logger.debug(f"Invert Selection failed: {e}")

    def on_open_in_neuroglancer(self):
        """Generate a Neuroglancer scene from current viewer state."""
        try:
            scene = self.current_view().ngl_viewer.neuroglancer_scene()
            scene.open()
        except Exception as e:
            logger.error(f"Open in Neuroglancer failed: {e}")

    def on_open_selection_in_new_tab(self):
        """Open currently selected scatter points in a new tab."""
        try:
            source_view = self.current_view()
            source_fig = source_view.fig_scatter
            selected = source_fig.selected
            selected_indices = (
                np.asarray(selected, dtype=int)
                if selected is not None
                else np.array([], dtype=int)
            )

            if selected_indices.size == 0:
                source_fig.show_message("No points selected", color="red", duration=2)
                return

            if source_fig.metadata is None or source_fig.positions is None:
                logger.info("No figure data available for opening a selection window")
                return

            selected_meta = (
                source_fig.metadata.iloc[selected_indices].copy().reset_index(drop=True)
            )

            def _subset_features(feats):
                """Subset features to the selection and drop now-empty columns."""
                feats = feats.iloc[selected_indices]
                return feats.loc[:, (feats.values.max(axis=0) > 0)].copy()

            # Carry over EVERY embedding (each subset to the selection).
            source_entries = getattr(source_fig, "embedding_entries", None) or []
            new_entries = []
            for entry in source_entries:
                emb = np.asarray(entry["embedding"])[selected_indices].copy()
                feats = entry.get("features")
                if feats is not None:
                    feats = _subset_features(feats)
                dists = entry.get("distances")
                if dists is not None:
                    dists = dists.iloc[selected_indices, selected_indices].copy()
                knn = entry.get("knn")
                if knn is not None:
                    knn = _subset_knn(knn, selected_indices)
                new_entries.append(
                    {
                        "name": entry["name"],
                        "embedding": emb,
                        "features": feats,
                        "distances": dists,
                        "knn": knn,
                        "features_info": entry.get("features_info"),
                        "distances_info": entry.get("distances_info"),
                    }
                )

            active = source_fig.active_embedding
            active = active if active is not None else 0

            if new_entries:
                active = active % len(new_entries)
                active_entry = new_entries[active]
                selected_points = active_entry["embedding"]
                selected_distances = active_entry["distances"]
                selected_features = active_entry["features"]
                selected_knn = active_entry["knn"]
            else:
                # No entry list (shouldn't normally happen): fall back to the
                # currently displayed positions and active sources.
                selected_points = np.asarray(source_fig.positions)[
                    selected_indices
                ].copy()
                selected_distances = None
                selected_features = None
                selected_knn = None
                source_matrices = getattr(source_fig, "dists", None)
                if isinstance(source_matrices, dict):
                    if source_matrices.get("distances") is not None:
                        selected_distances = source_matrices["distances"].iloc[
                            selected_indices, selected_indices
                        ].copy()
                    if source_matrices.get("features") is not None:
                        selected_features = _subset_features(
                            source_matrices["features"]
                        )
                    if source_matrices.get("knn") is not None:
                        selected_knn = _subset_knn(
                            source_matrices["knn"], selected_indices
                        )

            # Read the point size from the source view before the new tab
            # becomes the current one.
            source_data = (
                source_view._data if isinstance(source_view._data, dict) else {}
            )
            point_size = source_data.get("point_size", 10)

            # Snapshot the source view's display configuration (label/color/size
            # selections and their settings) so the new tab mirrors it. Must be
            # read before the new tab becomes the current one.
            display_state = source_view.scatter_controls.capture_display_state()

            # Carry the source view's pane arrangement into the new tab too.
            source_layout_mode = getattr(source_view, "_layout_mode", "stacked")
            source_layout_ratio = source_view.layout_ratio()

            title = f"selection ({len(selected_meta)})"
            view = self.add_new_tab(title=title)
            self._populate_view(
                view,
                meta=selected_meta,
                points=selected_points,
                entries=new_entries,
                active=active,
                distances=selected_distances,
                features=selected_features,
                knn=selected_knn,
                point_size=point_size,
                ngl_source_viewer=getattr(source_view, "ngl_viewer", None),
                display_state=display_state,
            )
            self._update_view_actions()
            self.set_view_title(view, title)

            # Apply the carried-over arrangement once the new tab has a real
            # size (deferred a turn, like the startup restore in init_ui).
            if source_layout_mode != "stacked" or source_layout_ratio is not None:
                QTimer.singleShot(
                    0,
                    lambda v=view, m=source_layout_mode, r=source_layout_ratio: (
                        v.apply_layout_mode(m, r)
                    ),
                )
        except Exception as e:
            logger.error(f"Open selection in new tab failed: {e}")

    # Backwards-compatible alias: external code (e.g. the figure's canvas walk)
    # looks this attribute up by name.
    on_open_selection_in_new_window = on_open_selection_in_new_tab

    def _populate_view(
        self,
        view,
        *,
        meta,
        points,
        entries,
        active,
        distances=None,
        features=None,
        knn=None,
        point_size=10,
        ngl_source_viewer=None,
        display_state=None,
    ):
        """Populate a view's figure and 3D viewer with a prepared data subset."""
        fig = view.fig_scatter
        fig.clear()
        fig.set_points(
            points=points,
            metadata=meta,
            label_col="label",
            id_col="id",
            color_col="_color",
            marker_col="dataset",
            hover_col="\n".join(
                [
                    f"{c}: {{{c}}}"
                    for c in meta.columns
                    if not str(c).startswith("_")
                ]
            ),
            dataset_col="dataset",
            point_size=point_size,
            distances=distances,
            features=features,
            knn=knn,
        )
        fig.set_embeddings(entries, active=active)
        view.scatter_controls.update_controls()

        try:
            src_data = getattr(ngl_source_viewer, "data", None)
            if src_data is not None and len(src_data):
                if (
                    isinstance(src_data.index, pd.MultiIndex)
                    and "dataset" in meta.columns
                ):
                    keys = list(
                        zip(
                            meta["id"].tolist(),
                            meta["dataset"].tolist(),
                        )
                    )
                    ngl_data = src_data.loc[keys].copy()
                else:
                    ngl_data = src_data.loc[meta["id"].tolist()].copy()

                view.ngl_viewer.set_data(ngl_data)

                neuropil_mesh = getattr(ngl_source_viewer, "neuropil_mesh", None)
                if neuropil_mesh is not None:
                    view.ngl_viewer.set_neuropil_mesh(
                        neuropil_mesh,
                        neuropil_source=getattr(
                            ngl_source_viewer, "neuropil_source", None
                        ),
                    )
        except Exception as e:
            logger.debug(
                f"Failed to propagate Neuroglancer data to selection view: {e}"
            )

        # Mirror the source view's display configuration. Runs after
        # update_controls() so the child's combo boxes already hold the items
        # we select from.
        if display_state is not None:
            try:
                view.scatter_controls.apply_display_state(display_state)
            except Exception as e:
                logger.debug(f"Failed to apply display state to selection view: {e}")

        view._data = {
            "meta": meta,
            "embeddings": points,
            "distances": distances,
            "features": features,
            "knn": knn,
            "embedding_entries": entries,
            "active_embedding": active if entries else None,
            "point_size": point_size,
        }

    def on_copy_ids_to_clipboard(self):
        fig = self.current_view().fig_scatter
        selected_ids = fig.selected_ids

        if selected_ids is not None:
            # Use TSV for broad compatibility
            QApplication.clipboard().setText(
                ",".join(map(str, np.asarray(selected_ids)))
            )
            logger.info("Copied selected IDs to clipboard")
        else:
            logger.info("No selection to copy metadata from")

    def on_copy_meta_to_clipboard(self):
        fig = self.current_view().fig_scatter
        selected_meta = fig.selected_meta

        if selected_meta is not None:
            # Use TSV for broad compatibility
            QApplication.clipboard().setText(
                selected_meta.to_csv(sep="\t", index=False)
            )
            logger.info("Copied selected metadata to clipboard")
        else:
            logger.info("No selection to copy metadata from")

    def _project_meta_data(self):
        """Return project-level metadata if available."""
        if not hasattr(self, "_data") or not isinstance(self._data, dict):
            return None

        meta = self._data.get("meta")
        if isinstance(meta, pd.DataFrame):
            return meta
        return None

    def on_export_meta_to_clipboard(self):
        """Copy full project metadata to clipboard as TSV."""
        meta_data = self._project_meta_data()
        if meta_data is None:
            logger.info("No project metadata available to export")
            return

        QApplication.clipboard().setText(meta_data.to_csv(sep="\t", index=False))
        logger.info("Copied full project metadata to clipboard")

    def on_export_meta_to_csv(self):
        """Export full project metadata to a CSV file."""
        meta_data = self._project_meta_data()
        if meta_data is None:
            logger.info("No project metadata available to export")
            return

        default_name = "meta_data.csv"
        project = getattr(self, "_current_project_loader", None)
        if project is not None and getattr(project, "name", None):
            default_name = f"{project.name}_meta_data.csv"

        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Meta Data to CSV",
            default_name,
            "CSV Files (*.csv);;All Files (*)",
        )
        if not file_path:
            return

        if not file_path.lower().endswith(".csv"):
            file_path = f"{file_path}.csv"

        try:
            meta_data.to_csv(file_path, index=False)
            logger.info(f"Exported full project metadata to {file_path}")
        except Exception as e:
            logger.error(f"Failed to export project metadata to CSV: {e}")

    def on_export_embedding_to_plotly_html(self):
        """Export the current embedding to an interactive Plotly HTML file."""
        fig = self.current_view().fig_scatter

        if getattr(fig, "positions", None) is None or len(fig) == 0:
            logger.info("No embedding available to export")
            return

        default_name = "embedding_plotly.html"
        project = getattr(self, "_current_project_loader", None)
        if project is not None and getattr(project, "name", None):
            default_name = f"{project.name}_embedding_plotly.html"

        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Embedding to Plotly HTML",
            default_name,
            "HTML Files (*.html);;All Files (*)",
        )
        if not file_path:
            return

        if not file_path.lower().endswith(".html"):
            file_path = f"{file_path}.html"

        try:
            plotly_fig = fig.to_plotly()
            plotly_fig.write_html(file_path, include_plotlyjs="cdn", full_html=True)
            logger.info(f"Exported embedding to Plotly HTML: {file_path}")
            fig.show_message("Exported Plotly HTML", color="green", duration=2)
        except ImportError:
            logger.error(
                "Plotly is not installed. Install plotly to export interactive HTML."
            )
            fig.show_message("Plotly is not installed", color="red", duration=3)
        except Exception as e:
            logger.error(f"Failed to export embedding to Plotly HTML: {e}")
            fig.show_message("Export failed", color="red", duration=3)

    def on_export_embedding_to_plotly_dashboard_html(self):
        """Export a compact multi-panel Plotly dashboard HTML file."""
        fig = self.current_view().fig_scatter

        if getattr(fig, "positions", None) is None or len(fig) == 0:
            logger.info("No embedding available to export")
            return

        default_name = "embedding_dashboard_plotly.html"
        project = getattr(self, "_current_project_loader", None)
        if project is not None and getattr(project, "name", None):
            default_name = f"{project.name}_embedding_dashboard_plotly.html"

        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Embedding Dashboard to Plotly HTML",
            default_name,
            "HTML Files (*.html);;All Files (*)",
        )
        if not file_path:
            return

        if not file_path.lower().endswith(".html"):
            file_path = f"{file_path}.html"

        try:
            fig.write_plotly_dashboard_html(file_path)
            logger.info(f"Exported embedding dashboard to Plotly HTML: {file_path}")
            fig.show_message("Exported Plotly dashboard", color="green", duration=2)
        except ImportError:
            logger.error(
                "Plotly is not installed. Install plotly to export interactive HTML."
            )
            fig.show_message("Plotly is not installed", color="red", duration=3)
        except Exception as e:
            logger.error(f"Failed to export embedding dashboard to Plotly HTML: {e}")
            fig.show_message("Export failed", color="red", duration=3)

    def show_open_project_dialog(self):
        """Show the open project dialog and load the selected project."""
        dialog = OpenProjectDialog(self)

        # Show the dialog and wait for user action
        if dialog.exec() != QDialog.Accepted:
            return

        self._load_project_from_dialog(dialog)

    def _load_project_from_dialog(self, dialog):
        """Load a project from a configured dialog instance."""
        state = dialog.current_state()

        # Load the selected project
        project = dialog.selected_project_loader()
        logger.info(f"Loading selected project: {project}")

        # No project? Just return
        if project is None:
            return

        self._load_project(project, state.get("embedding_mode", ""))
        self.add_recent_project(state)

    def _load_project(self, project, embedding_mode=""):
        """Load a resolved project loader into the current visualization."""
        embedding_mode = (embedding_mode or "").strip()

        # Auxiliary widgets are project-specific; reset cached instances on reload.
        self._dispose_view_aux_widgets(self.current_view())

        # Create progress dialog with range
        progress = QProgressDialog("Loading project data...", None, 0, 100, self)
        progress.setWindowModality(Qt.ApplicationModal)
        progress.setAutoClose(True)
        progress.setCancelButton(None)
        progress.show()
        progress.setValue(0)
        QApplication.processEvents()

        try:
            # This is a dictionary with the relevant data
            # Should already be filtered and ordered as needed
            self._data = project.compile(progress_callback=progress)
            progress.setValue(30)
            progress.setLabelText("Preparing embeddings...")
            QApplication.processEvents()

            entries = self._data.get("embedding_entries", []) or []
            active = self._data.get("active_embedding", 0 if entries else None)

            # (Re-)calculate embeddings if needed
            if embedding_mode in (
                "calculate from distances",
                "calculate from features",
            ):
                progress.setLabelText("Calculating embeddings...")
                progress.setValue(40)
                QApplication.processEvents()

                import umap

                if embedding_mode == "calculate from distances":
                    reducer = umap.UMAP(
                        n_components=2,
                        n_neighbors=10,
                        min_dist=0.1,
                        spread=1,
                        metric="precomputed",
                        random_state=42,
                    )
                    embeddings = reducer.fit_transform(self._data["distances"])
                else:
                    reducer = umap.UMAP(
                        n_components=2,
                        n_neighbors=10,
                        min_dist=0.1,
                        spread=1,
                        random_state=42,
                        metric="cosine",
                    )
                    embeddings = reducer.fit_transform(self._data["features"])

                # A load-time recompute supersedes the active embedding.
                if entries and active is not None:
                    entries[active]["embedding"] = embeddings
                self._data["embeddings"] = embeddings
            else:
                embeddings = self._data["embeddings"]

            progress.setValue(70)
            progress.setLabelText("Setting up visualization...")
            QApplication.processEvents()

            # First, let's clear existing data
            fig = self.current_view().fig_scatter
            fig.clear()

            # Now set the new data
            fig.set_points(
                points=embeddings,
                metadata=self._data["meta"],
                label_col="label",
                id_col="id",
                color_col="_color",  # this color is populated during data loading based on project info
                marker_col="dataset",
                hover_col="\n".join(
                    [
                        f"{c}: {{{c}}}"
                        for c in self._data["meta"].columns
                        if not str(c).startswith("_")
                    ]
                ),
                dataset_col="dataset",
                point_size=self._data.get("point_size", 10),
                distances=self._data.get("distances", None),
                features=self._data.get("features", None),
                knn=self._data.get("knn", None),
            )
            # Register the full set of embeddings (normalizes the non-active ones
            # into the active embedding's frame). No-op when there are none.
            fig.set_embeddings(entries, active=active if active is not None else 0)
            # We have to update bits and pieces on the controls panels based on the new data
            self.current_view().scatter_controls.update_controls()
            self._refresh_hover_columns_menu()

            # Set up the 3D viewer
            if "neuroglancer" in project.info:
                progress.setValue(85)
                progress.setLabelText("Setting up neuroglancer viewer...")
                QApplication.processEvents()

                ngl_viewer = self.current_view().ngl_viewer
                ngl_data = self._data["meta"][["id"]].copy()
                if "dataset" in self._data["meta"].columns:
                    ngl_data["dataset"] = self._data["meta"]["dataset"]

                if "source" in project.info["neuroglancer"]:
                    sources = project.info["neuroglancer"]["source"]
                    if isinstance(sources, dict):
                        # Map dataset to source URL
                        ngl_data["source"] = ngl_data.dataset.map(sources)
                    elif (
                        isinstance(sources, str)
                        and sources in self._data["meta"].columns
                    ):
                        # Column name with source URLs
                        ngl_data["source"] = self._data["meta"][sources]
                    else:
                        # Single source for all datasets
                        ngl_data["source"] = sources

                color = project.info["neuroglancer"].get("color", None)
                if color is not None:
                    if isinstance(color, str) and color in self._data["meta"].columns:
                        ngl_data["_color"] = self._data["meta"][color]
                    elif isinstance(color, dict):
                        ngl_data["_color"] = ngl_data.dataset.map(color)
                    else:
                        try:
                            color = tuple(cmap.Color(color).rgba)
                            ngl_data["_color"] = [color] * len(ngl_data)
                        except BaseException:
                            raise ValueError(f"Invalid color specification: {color}")

                ngl_viewer.set_data(ngl_data)

                neuropil_mesh = project.info["neuroglancer"].get("neuropil_mesh", None)
                if neuropil_mesh:
                    ngl_viewer.set_neuropil_mesh(neuropil_mesh)

            self._current_project_loader = project
            self._update_view_actions()
            self.set_view_title(
                self.current_view(),
                project.name,
                tooltip=str(getattr(project, "path", "") or project.name),
            )

            progress.setValue(100)
        finally:
            progress.close()


def main(dataset=None):
    """Main application entry point."""
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    if dataset is not None:
        from PySide6.QtCore import QTimer

        QTimer.singleShot(0, lambda: _load_dataset_from_arg(window, dataset))
    sys.exit(app.exec())


def _load_dataset_from_arg(window, dataset):
    """Load a dataset specified via the --from command-line argument."""
    try:
        parsed = parse_directory(dataset)
        if isinstance(parsed, SingleProjectLoader):
            project = parsed
        else:
            projects = [p for p in parsed if p is not None]
            if not projects:
                logger.error(f"No projects found at: {dataset}")
                return
            if len(projects) == 1:
                project = projects[0]
            else:
                # Multiple projects: show a quick selection dialog
                from PySide6.QtWidgets import QInputDialog

                names = [p.name for p in projects]
                name, ok = QInputDialog.getItem(
                    window,
                    "Select Project",
                    f"Multiple projects found in '{dataset}':\nSelect one to load:",
                    names,
                    0,
                    False,
                )
                if not ok:
                    return
                project = projects[names.index(name)]
        window._load_project(project)
    except Exception as e:
        logger.error(f"Failed to load dataset '{dataset}': {e}")
        from PySide6.QtWidgets import QMessageBox

        QMessageBox.critical(window, "Load Error", f"Could not load dataset:\n{e}")