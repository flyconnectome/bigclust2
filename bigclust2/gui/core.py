import sys
import cmap
import csv
import io
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
)
from PySide6.QtGui import QIcon, QAction, QKeySequence
from PySide6.QtCore import Qt, QSize, QSettings, QPoint, QTimer, QEvent, Signal
from importlib.resources import files

from .loaders import OpenProjectDialog
from ..data import parse_directory, SingleProjectLoader
from .controls import ScatterControls
from .widgets.connectivity import ConnectivityTable
from .widgets.distances import DistancesTable
from .widgets.features import FeatureExplorerWidget
from .widgets.meta_explorer import MetaExplorerDialog
from .widgets.annotations import AnnotationDialog, SelectionRecord
from ..scatter import ScatterFigure
from ..neuroglancer import NglViewer
from ..__version__ import __version__


__all__ = ["MainWindow", "MainWidget", "main"]


logger = logging.getLogger(__name__)


# Keep strong references to top-level windows so spawned windows stay alive.
_OPEN_WINDOWS = []

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
    """Main widget for the application."""

    def __init__(self):
        super().__init__()
        self._teardown_done = False
        self.init_ui()

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

        # Create top and bottom widgets
        self.top_widget = QWidget()
        self.setup_top_widget(self.top_widget)
        self.bottom_widget = QWidget()
        self.setup_bottom_widget(self.bottom_widget)

        # Add widgets to splitter
        self.splitter.addWidget(self.top_widget)
        self.splitter.addWidget(self.bottom_widget)

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

        layout.addWidget(self.splitter)
        self.setLayout(layout)

        # Wire overlay button actions
        self.configure_overlay_actions()

    @resize_figures
    def show_only(self, target_widget):
        """Show only the target widget inside the splitter."""
        self.top_widget.setVisible(True)
        self.bottom_widget.setVisible(True)

        if target_widget is self.top_widget:
            self.top_widget.show()
            self.bottom_widget.hide()
            self.splitter.setSizes([1, 0])
        elif target_widget is self.bottom_widget:
            self.bottom_widget.show()
            self.top_widget.hide()
            self.splitter.setSizes([0, 1])

    @resize_figures
    def show_side_by_side(self):
        """Show both widgets side-by-side (horizontal)."""
        self.top_widget.show()
        self.bottom_widget.show()
        self.splitter.setOrientation(Qt.Horizontal)
        self._set_equal_sizes_horizontal()

    @resize_figures
    def show_stacked(self):
        """Show both widgets stacked vertically."""
        self.top_widget.show()
        self.bottom_widget.show()
        self.splitter.setOrientation(Qt.Vertical)
        self._set_equal_sizes_vertical()

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
        if hasattr(self.top_widget, "buttons") and self.top_widget.buttons:
            # Left-most button: fullscreen top
            self.top_widget.buttons[0].clicked.connect(
                lambda: self.show_only(self.top_widget)
            )
            # Middle button: stacked vertically
            self.top_widget.buttons[1].clicked.connect(self.show_side_by_side)
            # Right button: side-by-side
            self.top_widget.buttons[2].clicked.connect(self.show_stacked)

        if hasattr(self.bottom_widget, "buttons") and self.bottom_widget.buttons:
            # Left-most button: fullscreen bottom
            self.bottom_widget.buttons[0].clicked.connect(
                lambda: self.show_only(self.bottom_widget)
            )
            # Middle button: stacked vertically
            self.bottom_widget.buttons[1].clicked.connect(self.show_side_by_side)
            # Right button: side-by-side
            self.bottom_widget.buttons[2].clicked.connect(self.show_stacked)

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
        sidebar = getattr(self.top_widget, "sidebar", None)
        if sidebar is None:
            return

        sidebar.setVisible(not sidebar.isVisible())
        self.update_left_button_position(self.top_widget)

        # Force an update to prevent transparency artifacts on sidebar toggles.
        self.force_update()
        self.resize_figures()
        self.fig_scatter.force_single_render()
        self.ngl_viewer.force_single_render()

    def toggle_viewer_controls(self):
        """Toggle visibility of the neuroglancer viewer controls sidebar."""
        sidebar = getattr(self.bottom_widget, "sidebar", None)
        if sidebar is None:
            return

        sidebar.setVisible(not sidebar.isVisible())
        self.update_left_button_position(self.bottom_widget)

        # Force an update to prevent transparency artifacts on sidebar toggles.
        self.force_update()
        self.resize_figures()
        self.fig_scatter.force_single_render()
        self.ngl_viewer.force_single_render()

    def setup_top_widget(self, top_widget):
        """Set up the top widget with overlay buttons and a left-positioned button."""
        # Initialize and connect the figure to the top widgets
        self.fig_scatter = ScatterFigure(parent=top_widget)
        self.fig_scatter.show()

        # Create main layout for top widget
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
        # a large minimum height on the top widget.
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
        splitter.setSizes([300, 1])  # Updated after first resize to exact pixel width

        main_layout.addWidget(splitter)
        top_widget.setLayout(main_layout)
        top_widget.splitter = splitter
        top_widget.sidebar = sidebar
        top_widget.content = content
        top_widget._initial_sidebar_width_applied = False

        # Create a button positioned on the left, centered vertically
        left_button = QPushButton()
        left_button.setFixedSize(40, 40)
        left_button.setParent(top_widget)
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
        left_button.setToolTip("Toggle top sidebar")

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
                "Show only top widget",
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
            button.setParent(top_widget)
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
                top_widget.width()
                - (button_size + button_spacing) * (3 - i)
                + button_spacing
            )
            y = button_spacing
            button.move(x, y)

        # Initial position of left button
        self.update_left_button_position(top_widget)

        # Store reference to adjust position on resize
        top_widget.left_button = left_button
        top_widget.sidebar = sidebar
        top_widget.buttons = buttons
        top_widget.button_size = button_size
        top_widget.button_spacing = button_spacing

        # Override resizeEvent to reposition buttons
        original_resize = top_widget.resizeEvent

        def on_resize(event):
            original_resize(event)
            if not top_widget._initial_sidebar_width_applied:
                self._set_sidebar_width(splitter, 300)
                top_widget._initial_sidebar_width_applied = True
            for i, button in enumerate(buttons):
                x = (
                    top_widget.width()
                    - (button_size + button_spacing) * (3 - i)
                    + button_spacing
                )
                y = button_spacing
                button.move(x, y)
            # Reposition left button
            self.update_left_button_position(top_widget)
            # Resize figure if necessary
            self.resize_figures()

        top_widget.resizeEvent = on_resize

        # Update button position when splitter is moved
        splitter.splitterMoved.connect(
            lambda: self.update_left_button_position(top_widget)
        )

    def setup_bottom_widget(self, bottom_widget):
        """Set up the bottom widget with a left-positioned button and sidebar."""
        # Initialize and connect the neuroglancer viewer in the bottom widgets
        self.ngl_viewer = NglViewer(viewer_kwargs=dict(parent=bottom_widget))
        self.ngl_viewer.viewer.show()

        # Hook the viewer up to the figure
        self.fig_scatter.sync_viewer(self.ngl_viewer)

        # Create main layout for bottom widget
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
        splitter.setSizes([300, 1])  # Updated after first resize to exact pixel width

        main_layout.addWidget(splitter)
        bottom_widget.setLayout(main_layout)
        bottom_widget.splitter = splitter
        bottom_widget.sidebar = sidebar
        bottom_widget.content = content
        bottom_widget._initial_sidebar_width_applied = False

        # Create a button positioned on the left, centered vertically
        left_button = QPushButton()
        left_button.setFixedSize(40, 40)
        left_button.setParent(bottom_widget)
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
        left_button.setToolTip("Toggle bottom sidebar")

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
                "Show only bottom widget",
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
            button.setParent(bottom_widget)
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
                bottom_widget.width()
                - (button_size + button_spacing) * (3 - i)
                + button_spacing
            )
            y = button_spacing
            button.move(x, y)

        # Initial position of left button
        self.update_left_button_position(bottom_widget)

        # Store reference to adjust position on resize
        bottom_widget.left_button = left_button
        bottom_widget.buttons = buttons
        bottom_widget.button_size = button_size
        bottom_widget.button_spacing = button_spacing
        bottom_widget.sidebar = sidebar

        # Override resizeEvent to reposition buttons
        original_resize = bottom_widget.resizeEvent

        def on_resize(event):
            original_resize(event)
            if not bottom_widget._initial_sidebar_width_applied:
                self._set_sidebar_width(splitter, 300)
                bottom_widget._initial_sidebar_width_applied = True
            for i, button in enumerate(buttons):
                x = (
                    bottom_widget.width()
                    - (button_size + button_spacing) * (3 - i)
                    + button_spacing
                )
                y = button_spacing
                button.move(x, y)
                button.raise_()
            # Reposition left button
            self.update_left_button_position(bottom_widget)

        bottom_widget.resizeEvent = on_resize

        # Update button position when splitter is moved
        splitter.splitterMoved.connect(
            lambda: self.update_left_button_position(bottom_widget)
        )

        # Initial positions
        self.update_left_button_position(bottom_widget)
        for i, button in enumerate(buttons):
            x = (
                bottom_widget.width()
                - (button_size + button_spacing) * (3 - i)
                + button_spacing
            )
            y = button_spacing
            button.move(x, y)

        def apply_initial_bottom_overlay_layout():
            if not bottom_widget._initial_sidebar_width_applied:
                self._set_sidebar_width(splitter, 300)
                bottom_widget._initial_sidebar_width_applied = True

            for i, button in enumerate(buttons):
                x = (
                    bottom_widget.width()
                    - (button_size + button_spacing) * (3 - i)
                    + button_spacing
                )
                y = button_spacing
                button.move(x, y)
                button.raise_()

            self.update_left_button_position(bottom_widget)

        # Ensure overlays are correctly placed after the first real layout pass.
        QTimer.singleShot(0, apply_initial_bottom_overlay_layout)

    def resize_figures(self):
        """Resize figures to match their parent widgets."""
        self.fig_scatter.size = (
            self.top_widget.content.width(),
            self.top_widget.content.height(),
        )
        self.ngl_viewer.size = (
            self.bottom_widget.content.width(),
            self.bottom_widget.content.height(),
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

    def __init__(self):
        super().__init__()
        self.setAttribute(Qt.WA_DeleteOnClose, True)
        self.settings = QSettings("BigClust", "BigClustGUI")
        self.open_recent_menu = None
        self.connectivity_table_action = None
        self.distances_table_action = None
        self.feature_explorer_action = None
        self._current_project_loader = None
        self._connectivity_widget = None
        self._connectivity_widget_synced = False
        self._feature_explorer_widget = None
        self._annotation_dialog = None
        self._annotation_log = []
        self._annotation_log_dir = Path.home() / ".bigclust"
        self._annotation_log_dir.mkdir(parents=True, exist_ok=True)
        self._annotation_log_file = self._annotation_log_dir / (
            f"annotation_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.log"
        )
        self._annotation_log_file.touch(exist_ok=True)
        self._distance_widgets = []
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

    def eventFilter(self, obj, event):
        if obj is self._connectivity_widget and event.type() == QEvent.Close:
            self._unsync_connectivity_widget(obj)
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

        # Create and set the main widget
        main_widget = MainWidget()
        self.setCentralWidget(main_widget)

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

        new_window_action = QAction("New Window", self)
        new_window_action.setShortcut(QKeySequence("Shift+Meta+N"))
        new_window_action.triggered.connect(self.open_new_window)
        file_menu.addAction(new_window_action)

        close_window_action = QAction("Close Window", self)
        close_window_action.setShortcut(QKeySequence.Close)
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
            lambda: self.centralWidget().fig_scatter.center_camera()
        )
        center_menu.addAction(center_scatter_action)

        center_3d_action = QAction("3D Viewer", self)
        center_3d_action.triggered.connect(
            lambda: self.centralWidget().ngl_viewer.viewer.center_camera()
        )
        center_menu.addAction(center_3d_action)

        view_menu.addSeparator()

        toggle_figure_controls_action = QAction("Toggle Figure Controls", self)
        toggle_figure_controls_action.triggered.connect(
            lambda: self.centralWidget().toggle_figure_controls()
        )
        view_menu.addAction(toggle_figure_controls_action)

        toggle_viewer_controls_action = QAction("Toggle Viewer Controls", self)
        toggle_viewer_controls_action.triggered.connect(
            lambda: self.centralWidget().toggle_viewer_controls()
        )
        view_menu.addAction(toggle_viewer_controls_action)

        # The view menu automatically contains an item to toggle fullscreen mode
        # This separator keeps it visually separate from the custom view actions we added above.
        view_menu.addSeparator()

        # Selection menu
        selection_menu = menu_bar.addMenu("Selection")

        select_all_action = QAction("Select All", self)
        select_all_action.triggered.connect(self.on_select_all)
        selection_menu.addAction(select_all_action)

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
        open_selection_in_new_window_action = QAction("Open in New Window", self)
        open_selection_in_new_window_action.setShortcut(
            QKeySequence("Shift+Ctrl+Meta+N")
        )
        open_selection_in_new_window_action.triggered.connect(
            self.on_open_selection_in_new_window
        )
        selection_menu.addAction(open_selection_in_new_window_action)

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

        show_annotation_log_action = QAction("Show Annotation Log", self)
        show_annotation_log_action.triggered.connect(self.show_annotation_log)
        window_menu.addAction(show_annotation_log_action)

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

    def open_new_window(self):
        """Open a second top-level BigClust window."""
        window = MainWindow()
        window.move(self.pos() + QPoint(40, 40))
        window.show()

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

    def show_connectivity_table(self):
        """Open the connectivity table widget for the current project."""
        if not self._can_open_connectivity_table():
            return

        existing = self._connectivity_widget
        if existing is not None:
            try:
                self._sync_connectivity_widget(existing)
                # Reuse the existing widget for this project and bring it to front.
                existing.showNormal()
                existing.show()
                existing.raise_()
                existing.activateWindow()
                return
            except RuntimeError:
                # Underlying Qt object was deleted elsewhere; rebuild on demand.
                self._connectivity_widget = None

        features = self._data.get("features") if hasattr(self, "_data") else None
        meta_data = self._data.get("meta") if hasattr(self, "_data") else None
        fig = self.centralWidget().fig_scatter

        if features is None or meta_data is None:
            return

        widget = ConnectivityTable(
            features,
            figure=fig,
            meta_data=meta_data,
            parent=self,
        )
        widget.installEventFilter(self)
        self._sync_connectivity_widget(widget)
        widget.show()

        # Keep a strong reference so the window can be reopened instead of recreated.
        self._connectivity_widget = widget
        widget.destroyed.connect(
            lambda _obj=None: setattr(self, "_connectivity_widget", None)
        )
        widget.destroyed.connect(
            lambda _obj=None: setattr(self, "_connectivity_widget_synced", False)
        )
        widget.destroyed.connect(lambda _obj=None, w=widget: fig.unsync_widget(w))

    def show_feature_explorer(self):
        """Open the Feature Explorer widget for the current project."""
        if not self._can_open_feature_explorer():
            return

        existing = self._feature_explorer_widget
        if existing is not None:
            try:
                existing.showNormal()
                existing.show()
                existing.raise_()
                existing.activateWindow()
                return
            except RuntimeError:
                self._feature_explorer_widget = None

        features = self._data.get("features")
        meta_data = self._data.get("meta")
        if features is None or meta_data is None:
            return

        widget = FeatureExplorerWidget(
            metadata=meta_data,
            features=features,
            figure=self.centralWidget().fig_scatter,
            parent=None,
        )
        widget.setAttribute(Qt.WA_DeleteOnClose, True)
        widget.setWindowFlag(Qt.Window, True)
        widget.setWindowTitle("Feature Explorer")
        # widget.resize(1100, 700)
        widget.show()

        self._feature_explorer_widget = widget
        widget.destroyed.connect(
            lambda _obj=None: setattr(self, "_feature_explorer_widget", None)
        )

    def show_meta_explorer(self):
        """Open the meta data explorer dialog for the current project."""
        if not self._can_open_meta_explorer():
            return

        existing = getattr(self, "_meta_explorer_dialog", None)
        if existing is not None:
            try:
                existing.showNormal()
                existing.show()
                existing.raise_()
                existing.activateWindow()
                return
            except RuntimeError:
                self._meta_explorer_dialog = None

        meta_data = self._data.get("meta") if hasattr(self, "_data") else None
        if meta_data is None:
            return

        dialog = MetaExplorerDialog(
            meta_data,
            figure=self.centralWidget().fig_scatter,
            parent=self,
        )
        dialog.setAttribute(Qt.WA_DeleteOnClose, True)
        dialog.show()

        self._meta_explorer_dialog = dialog
        dialog.destroyed.connect(
            lambda _obj=None: setattr(self, "_meta_explorer_dialog", None)
        )

    def _sync_connectivity_widget(self, widget):
        """Ensure the cached connectivity widget is synced to figure selection."""
        if widget is None or self._connectivity_widget_synced:
            return

        fig = self.centralWidget().fig_scatter
        fig.sync_widget(widget)
        self._connectivity_widget_synced = True

    def _unsync_connectivity_widget(self, widget=None):
        """Unsync the cached connectivity widget to avoid heavy updates while hidden."""
        if not self._connectivity_widget_synced:
            return

        if widget is None:
            widget = self._connectivity_widget
        if widget is None:
            self._connectivity_widget_synced = False
            return

        try:
            fig = self.centralWidget().fig_scatter
            fig.unsync_widget(widget)
        finally:
            self._connectivity_widget_synced = False

    def _dispose_connectivity_widget(self):
        """Dispose the cached connectivity widget when project context changes."""
        widget = self._connectivity_widget
        if widget is None:
            return

        self._connectivity_widget = None
        self._unsync_connectivity_widget(widget)

        try:
            widget.removeEventFilter(self)
            widget.close()
            widget.deleteLater()
        except RuntimeError:
            # Qt object may already be deleted.
            pass

    def _dispose_feature_explorer_widget(self):
        """Dispose the cached feature explorer widget when project context changes."""
        widget = self._feature_explorer_widget
        if widget is None:
            return

        self._feature_explorer_widget = None
        try:
            widget.close()
            widget.deleteLater()
        except RuntimeError:
            # Qt object may already be deleted.
            pass

    def _selected_annotation_records(self):
        """Build annotation dialog selection records from scatter selection."""
        fig = self.centralWidget().fig_scatter
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
        selection = self._selected_annotation_records()
        project_datasets = self._project_annotation_datasets()
        if not selection:
            fig = self.centralWidget().fig_scatter
            fig.show_message("No points selected", color="red", duration=2)
            return

        existing = self._annotation_dialog
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
                self._annotation_dialog = None

        dialog = AnnotationDialog(
            selection=selection,
            parent=self,
            project_datasets=project_datasets,
        )
        dialog.annotations_logged.connect(self._log_annotation_entries)
        dialog.show()
        dialog.raise_()
        dialog.activateWindow()

        self._annotation_dialog = dialog
        dialog.destroyed.connect(
            lambda _obj=None: setattr(self, "_annotation_dialog", None)
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
            fig = self.centralWidget().fig_scatter
            fig.show_message(text, color=color, duration=3)
        except Exception as e:
            logger.debug(f"Failed to show annotation submit status message: {e}")

    def _handle_annotation_changed(self, changed_neurons):
        """Highlight labels for neurons whose annotations were changed."""
        try:
            fig = self.centralWidget().fig_scatter
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

    @property
    def annotation_log(self):
        """Return the per-window annotation log."""
        return list(self._annotation_log)

    def _dispose_annotation_dialog(self):
        """Dispose the cached annotation dialog when project context changes."""
        dialog = self._annotation_dialog
        if dialog is None:
            return

        self._annotation_dialog = None
        try:
            dialog.close()
            dialog.deleteLater()
        except RuntimeError:
            # Qt object may already be deleted.
            pass

    def show_distances_table(self):
        """Open the pairwise distance heatmap widget for the current project."""
        if not self._can_open_distances_table():
            return

        distances = self._data.get("distances") if hasattr(self, "_data") else None
        meta_data = self._data.get("meta") if hasattr(self, "_data") else None
        fig = self.centralWidget().fig_scatter

        if distances is None or meta_data is None:
            return

        widget = DistancesTable(
            distances,
            figure=fig,
            meta_data=meta_data,
            parent=self,
        )
        fig.sync_widget(widget)
        widget.show()

        self._distance_widgets.append(widget)
        widget.destroyed.connect(
            lambda _obj=None, w=widget: (
                self._distance_widgets.remove(w)
                if w in self._distance_widgets
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
        main_widget = self.centralWidget()
        if main_widget is not None and hasattr(main_widget, "teardown_rendering"):
            main_widget.teardown_rendering()

        # Persist geometry and maximized state
        try:
            self.settings.setValue("mainWindow/geometry", self.saveGeometry())
            self.settings.setValue("mainWindow/isMaximized", self.isMaximized())
        except Exception:
            pass
        super().closeEvent(event)

    def on_select_all(self):
        try:
            fig = self.centralWidget().fig_scatter
            if hasattr(fig, "select_all"):
                fig.select_all()
            else:
                logger.info("Select All not supported by current figure")
        except Exception as e:
            logger.debug(f"Select All failed: {e}")

    def on_deselect_all(self):
        try:
            fig = self.centralWidget().fig_scatter
            if hasattr(fig, "deselect_all"):
                fig.deselect_all()
            elif hasattr(fig, "clear_selection"):
                fig.clear_selection()
            else:
                logger.info("Deselect All not supported by current figure")
        except Exception as e:
            logger.debug(f"Deselect All failed: {e}")

    def on_open_in_neuroglancer(self):
        """Generate a Neuroglancer scene from current viewer state."""
        try:
            scene = self.centralWidget().ngl_viewer.neuroglancer_scene()
            scene.open()
        except Exception as e:
            logger.error(f"Open in Neuroglancer failed: {e}")

    def on_open_selection_in_new_window(self):
        """Open currently selected scatter points in a new window."""
        try:
            source_main_widget = self.centralWidget()
            source_fig = source_main_widget.fig_scatter
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
            selected_points = np.asarray(source_fig.positions)[selected_indices].copy()
            selected_ids = selected_meta["id"].to_numpy()

            selected_distances = None
            selected_features = None
            source_matrices = getattr(source_fig, "dists", None)

            if isinstance(source_matrices, dict):
                source_distances = source_matrices.get("distances")
                source_features = source_matrices.get("features")

                if source_distances is not None:
                    if isinstance(source_distances, pd.DataFrame):
                        selected_distances = source_distances.iloc[
                            selected_indices, selected_indices
                        ].copy()
                    else:
                        raise ValueError(
                            "Expected distances matrix to be a DataFrame with metadata-aligned index and columns for selection export"
                        )

                if source_features is not None:
                    if isinstance(source_features, pd.DataFrame):
                        selected_features = source_features.iloc[selected_indices]

                        # Drop empty columns
                        selected_features = selected_features.loc[
                            :, (selected_features.values.max(axis=0) > 0)
                        ].copy()
                    else:
                        raise ValueError(
                            "Expected features matrix to be a DataFrame with metadata-aligned index for selection export"
                        )

            window = MainWindow()
            window.move(self.pos() + QPoint(40, 40))
            window.show()

            main_widget = window.centralWidget()
            fig = main_widget.fig_scatter
            fig.clear()
            fig.set_points(
                points=selected_points,
                metadata=selected_meta,
                label_col="label",
                id_col="id",
                color_col="_color",
                marker_col="dataset",
                hover_col="\n".join(
                    [
                        f"{c}: {{{c}}}"
                        for c in selected_meta.columns
                        if not str(c).startswith("_")
                    ]
                ),
                dataset_col="dataset",
                point_size=10,
                distances=selected_distances,
                features=selected_features,
            )
            main_widget.scatter_controls.update_controls()

            try:
                src_viewer = getattr(source_main_widget, "ngl_viewer", None)
                src_data = getattr(src_viewer, "data", None)
                if src_data is not None and len(src_data):
                    if (
                        isinstance(src_data.index, pd.MultiIndex)
                        and "dataset" in selected_meta.columns
                    ):
                        keys = list(
                            zip(
                                selected_meta["id"].tolist(),
                                selected_meta["dataset"].tolist(),
                            )
                        )
                        ngl_data = src_data.loc[keys].copy()
                    else:
                        ngl_data = src_data.loc[selected_meta["id"].tolist()].copy()

                    main_widget.ngl_viewer.set_data(ngl_data)

                    neuropil_mesh = getattr(src_viewer, "neuropil_mesh", None)
                    if neuropil_mesh is not None:
                        main_widget.ngl_viewer.set_neuropil_mesh(
                            neuropil_mesh,
                            neuropil_source=getattr(
                                src_viewer, "neuropil_source", None
                            ),
                        )
            except Exception as e:
                logger.debug(
                    f"Failed to propagate Neuroglancer data to selection window: {e}"
                )

            window._data = {
                "meta": selected_meta,
                "embeddings": selected_points,
                "distances": selected_distances,
                "features": selected_features,
            }
            window._update_view_actions()
            window.setWindowTitle(f"BigClust - selection ({len(selected_meta)})")
        except Exception as e:
            logger.error(f"Open selection in new window failed: {e}")

    def on_copy_ids_to_clipboard(self):
        fig = self.centralWidget().fig_scatter
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
        fig = self.centralWidget().fig_scatter
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

        # Connectivity table is project-specific; reset cached instance on reload.
        self._dispose_connectivity_widget()
        self._dispose_feature_explorer_widget()
        self._dispose_annotation_dialog()

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
            else:
                embeddings = self._data["embeddings"]

            progress.setValue(70)
            progress.setLabelText("Setting up visualization...")
            QApplication.processEvents()

            # First, let's clear existing data
            fig = self.centralWidget().fig_scatter
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
                point_size=10,
                distances=self._data.get("distances", None),
                features=self._data.get("features", None),
            )
            # We have to update bits and pieces on the controls panels based on the new data
            self.centralWidget().scatter_controls.update_controls()

            # Set up the 3D viewer
            if "neuroglancer" in project.info:
                progress.setValue(85)
                progress.setLabelText("Setting up neuroglancer viewer...")
                QApplication.processEvents()

                ngl_viewer = fig = self.centralWidget().ngl_viewer
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
            self.setWindowTitle(f"BigClust - {project.name}")

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
