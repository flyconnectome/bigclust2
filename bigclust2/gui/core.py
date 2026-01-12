import sys
import cmap
import logging

import numpy as np

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
    QMenuBar,
    QDialog,
    QSizePolicy,
    QProgressDialog,
)
from PySide6.QtGui import QIcon, QAction
from PySide6.QtCore import Qt, QSize, QSettings

from .loaders import OpenProjectDialog
from .controls import ScatterControls
from ..scatter import ScatterFigure
from ..neuroglancer import NglViewer
from ..__version__ import __version__


__all__ = ["MainWindow", "MainWidget", "main"]


logger = logging.getLogger(__name__)


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
        sidebar.setMinimumWidth(150)
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
        splitter.setSizes([200, 600])  # Initial sizes

        main_layout.addWidget(splitter)
        top_widget.setLayout(main_layout)
        top_widget.splitter = splitter

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

        # Create three square buttons for the overlay (icon buttons)
        button_size = 25
        button_spacing = 3
        buttons = []

        icon_defs = [
            ("assets/button_fullscreen.png", "Show only top widget"),
            ("assets/button_split_vertical.png", "Stack widgets vertically"),
            ("assets/button_split_horizontal.png", "Place widgets side by side"),
        ]

        for icon_path, tip in icon_defs:
            button = QPushButton()
            button.setFixedSize(button_size, button_size)
            button.setIcon(QIcon(icon_path))
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
        sidebar.setMinimumWidth(150)
        sidebar_layout = QVBoxLayout()
        sidebar_layout.setContentsMargins(10, 10, 10, 10)
        menu_items = [
            QPushButton("Menu Item 1"),
            QPushButton("Menu Item 2"),
            QPushButton("Menu Item 3"),
        ]
        for i, item in enumerate(menu_items, start=1):
            item.setToolTip(f"Bottom menu item {i}")
            sidebar_layout.addWidget(item)
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
        splitter.setSizes([200, 600])  # Initial sizes

        main_layout.addWidget(splitter)
        bottom_widget.setLayout(main_layout)
        bottom_widget.splitter = splitter

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
        self.fig_scatter.size = (self.top_widget.width(), self.top_widget.height())
        self.ngl_viewer.size = (self.bottom_widget.width(), self.bottom_widget.height())

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

    def __init__(self):
        super().__init__()
        self.settings = QSettings("BigClust", "BigClustGUI")
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
        menu_bar.setNativeMenuBar(True) # keep it visible inside the window across platforms
        file_menu = menu_bar.addMenu("File")
        open_project_action = QAction("Open Project", self)
        # Keyboard shortcut
        open_project_action.setShortcut("Ctrl+O")
        open_project_action.triggered.connect(self.show_open_project_dialog)
        file_menu.addAction(open_project_action)

        # Selection menu
        selection_menu = menu_bar.addMenu("Selection")

        select_all_action = QAction("Select All", self)
        select_all_action.triggered.connect(self.on_select_all)
        selection_menu.addAction(select_all_action)

        deselect_all_action = QAction("Deselect All", self)
        deselect_all_action.triggered.connect(self.on_deselect_all)
        selection_menu.addAction(deselect_all_action)

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
                "BigClust2<br><br>"
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
            logger.warning(f"Select All failed: {e}")

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
            logger.warning(f"Deselect All failed: {e}")

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
        dialog = OpenProjectDialog(self)

        # Show the dialog and wait for user action
        if dialog.exec() != QDialog.Accepted:
            return

        # Load the selected project
        project = dialog.selected_project_loader()
        logger.info(f"Loading selected project: {project}")

        # No project? Just return
        if project is None:
            return

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
            if dialog.embedding_combo.currentText() in (
                "calculate from distances",
                "calculate from features",
            ):
                progress.setLabelText("Calculating embeddings...")
                progress.setValue(40)
                QApplication.processEvents()

                import umap
                if dialog.embedding_combo.currentText() == "calculate from distances":
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
                hover_col="\n".join([f"{c}: {{{c}}}" for c in self._data["meta"].columns]),
                dataset_col="dataset",
                point_size=10,
                distances=self._data.get("distances", None),
                features=self._data.get("features", None),
            )
            # We have to update bits and pieces on the controls panels based on the new data
            self.centralWidget().scatter_controls.update_umap_options()

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
                    elif isinstance(sources, str) and sources in self._data["meta"].columns:
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

            progress.setValue(100)
        finally:
            progress.close()


def main():
    """Main application entry point."""
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
