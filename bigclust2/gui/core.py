import sys
import cmap
import json
import logging
from html import escape

import numpy as np

from pathlib import Path
from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QSplitter,
    QPushButton,
    QFrame,
    QLabel,
    QDialog,
    QSizePolicy,
    QProgressDialog,
    QWidgetAction,
)
from PySide6.QtGui import QIcon, QAction, QKeySequence
from PySide6.QtCore import Qt, QSize, QSettings
from importlib.resources import files

from .loaders import OpenProjectDialog
from ..data import parse_directory, SingleProjectLoader
from .controls import ScatterControls
from .widgets.connectivity import ConnectivityTable
from .widgets.distances import DistancesTable
from ..scatter import ScatterFigure
from ..neuroglancer import NglViewer
from ..__version__ import __version__


__all__ = ["MainWindow", "MainWidget", "main"]


logger = logging.getLogger(__name__)

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
        self.init_ui()

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
        sidebar.setMinimumWidth(80)
        sidebar_layout = QVBoxLayout()
        sidebar_layout.setContentsMargins(10, 10, 10, 10)

        # Add scatter controls to sidebar
        self.scatter_controls = ScatterControls(self.fig_scatter)
        self.scatter_controls.setSizePolicy(
            QSizePolicy.Expanding, QSizePolicy.Preferred
        )
        sidebar_layout.addWidget(self.scatter_controls, 1)

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
        splitter.setSizes([100, 600])  # Initial sizes

        main_layout.addWidget(splitter)
        top_widget.setLayout(main_layout)
        top_widget.splitter = splitter
        top_widget.sidebar = sidebar
        top_widget.content = content

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
            sidebar.setVisible(not sidebar.isVisible())
            self.update_left_button_position(top_widget)

            # We need to force a repaint to avoid glitches in the button transparency
            self.force_update()
            self.resize_figures()
            self.fig_scatter.force_single_render()
            self.ngl_viewer.force_single_render()

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
        sidebar.setMinimumWidth(80)
        sidebar_layout = QVBoxLayout()
        sidebar_layout.setContentsMargins(10, 10, 10, 10)

        # Add viewer controls to sidebar
        self.ngl_viewer.viewer.show_controls()
        self.viewer_controls = self.ngl_viewer.viewer._controls
        self.viewer_controls.setSizePolicy(
            QSizePolicy.Expanding, QSizePolicy.Preferred
        )
        sidebar_layout.addWidget(self.viewer_controls, 1)

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
        splitter.setSizes([100, 600])  # Initial sizes

        main_layout.addWidget(splitter)
        bottom_widget.setLayout(main_layout)
        bottom_widget.splitter = splitter
        bottom_widget.sidebar = sidebar
        bottom_widget.content = content

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
            sidebar.setVisible(not sidebar.isVisible())
            self.update_left_button_position(bottom_widget)

            # We need to force a repaint to avoid glitches in the button transparency
            self.force_update()

        left_button.clicked.connect(toggle_sidebar)

        # Make it so that pressing 'c' in the viewer opens/closes the controls sidebar
        self.ngl_viewer.viewer._key_events['c'] = toggle_sidebar

        # Create three square buttons for the overlay (icon buttons)
        button_size = 25
        button_spacing = 3
        buttons = []

        icon_defs = [
            ("assets/button_fullscreen.png", "Show only bottom widget"),
            ("assets/button_split_vertical.png", "Stack widgets vertically"),
            ("assets/button_split_horizontal.png", "Place widgets side by side"),
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
            for i, button in enumerate(buttons):
                x = (
                    bottom_widget.width()
                    - (button_size + button_spacing) * (3 - i)
                    + button_spacing
                )
                y = button_spacing
                button.move(x, y)
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


class MainWindow(QMainWindow):
    """Main application window."""

    RECENT_PROJECTS_KEY = "openRecentProjects/v1"
    MAX_RECENT_PROJECTS = 10

    def __init__(self):
        super().__init__()
        self.settings = QSettings("BigClust", "BigClustGUI")
        self.open_recent_menu = None
        self.connectivity_table_action = None
        self.distances_table_action = None
        self._current_project_loader = None
        self._connectivity_widgets = []
        self._distance_widgets = []
        self.init_ui()

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

        # Selection menu
        selection_menu = menu_bar.addMenu("Selection")

        select_all_action = QAction("Select All", self)
        select_all_action.triggered.connect(self.on_select_all)
        selection_menu.addAction(select_all_action)

        deselect_all_action = QAction("Deselect All", self)
        deselect_all_action.triggered.connect(self.on_deselect_all)
        selection_menu.addAction(deselect_all_action)

        selection_menu.addSeparator()
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

        # Export menu (to be populated later)
        export_menu = menu_bar.addMenu("Export")

        # Placeholder for future export actions
        export_placeholder = QAction("Placeholder", self)
        export_menu.addAction(export_placeholder)

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

    def _can_open_connectivity_table(self):
        """Whether the connectivity table can be opened for the current project."""
        project = self._current_project_loader
        if project is None:
            return False

        return project.feature_type == "connectivity"

    def _can_open_distances_table(self):
        """Whether the distance heatmap can be opened for the current project."""
        if not hasattr(self, "_data") or not isinstance(self._data, dict):
            return False

        dists = self._data.get("distances")
        if dists is None or not hasattr(dists, "shape") or len(dists.shape) != 2:
            return False

        return dists.shape[0] == dists.shape[1]

    def _update_view_actions(self):
        """Update View menu action states."""
        if self.connectivity_table_action is not None:
            self.connectivity_table_action.setEnabled(self._can_open_connectivity_table())
        if self.distances_table_action is not None:
            self.distances_table_action.setEnabled(self._can_open_distances_table())

    def show_connectivity_table(self):
        """Open the connectivity table widget for the current project."""
        if not self._can_open_connectivity_table():
            return

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
        fig.sync_widget(widget)
        widget.show()

        # Keep a strong reference so the window is not garbage collected.
        self._connectivity_widgets.append(widget)
        widget.destroyed.connect(
            lambda _obj=None, w=widget: self._connectivity_widgets.remove(w)
            if w in self._connectivity_widgets
            else None
        )

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
            lambda _obj=None, w=widget: self._distance_widgets.remove(w)
            if w in self._distance_widgets
            else None
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
            source_type = "remote" if path.startswith(("http://", "https://")) else "local"

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
        recent = [r for r in recent if self._recent_state_key(r) != self._recent_state_key(normalized)]
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
