import json
import logging

import requests
from PySide6.QtWidgets import (
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QLabel,
    QDialog,
    QLineEdit,
    QFileDialog,
    QDialogButtonBox,
    QComboBox,
    QTableWidget,
    QTableWidgetItem,
    QAbstractItemView,
    QGroupBox,
)
from PySide6.QtCore import QSettings, Qt

from ..data import parse_directory, SingleProjectLoader
from ..utils import string_to_polars_filter, is_list_of_ids

logger = logging.getLogger(__name__)


__all__ = ["OpenProjectDialog"]


class OpenProjectDialog(QDialog):
    """Simple dialog to choose a project source."""

    def __init__(self, parent=None, initial_state=None):
        super().__init__(parent)
        self.setWindowTitle("Open Project")
        self.setModal(True)
        self.resize(600, 480)
        self.settings = QSettings("BigClust", "BigClustGUI")
        self.projects = []  # Store parsed projects
        self.initial_state = initial_state or {}
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        label = QLabel("Choose local or remote data source")
        layout.addWidget(label)

        # Path entry with browse button
        path_row = QHBoxLayout()
        self.path_edit = QComboBox()
        self.path_edit.setEditable(True)
        self.path_edit.lineEdit().setPlaceholderText("Enter path or URL")
        self.path_edit.setInsertPolicy(QComboBox.NoInsert)
        # Load path history and keep the last path selected by default
        raw_history = self.settings.value("last_project_paths", [])
        if isinstance(raw_history, str):
            path_history = [raw_history] if raw_history else []
        elif isinstance(raw_history, (list, tuple)):
            path_history = [str(p) for p in raw_history if str(p).strip()]
        else:
            path_history = []

        last_path = str(self.settings.value("last_project_path", "")).strip()
        if last_path:
            path_history = [p for p in path_history if p != last_path]
            path_history.insert(0, last_path)
        path_history = path_history[:10]
        if path_history:
            self.path_edit.addItems(path_history)
            self.path_edit.setCurrentText(path_history[0])

        # Connect text change signal to validation
        self.path_edit.currentTextChanged.connect(self.scan_path)
        browse_btn = QPushButton("Browse…")
        browse_btn.setToolTip("Select a local folder")
        browse_btn.clicked.connect(self.open_file_dialog)

        path_row.addWidget(self.path_edit)
        path_row.addWidget(browse_btn)

        path_group = QVBoxLayout()
        path_group.setSpacing(4)
        path_group.addLayout(path_row)

        self.path_error_label = QLabel("")
        self.path_error_label.setWordWrap(True)
        self.path_error_label.setStyleSheet("color: #a00; margin-left: 4px; font-size: 12px;")
        self.path_error_label.setVisible(False)
        path_group.addWidget(self.path_error_label)

        layout.addLayout(path_group)

        # Container for showing/selecting projects
        project_label = QLabel("Select Project:")
        layout.addWidget(project_label)

        self.project_combo = QComboBox()
        self.project_combo.currentIndexChanged.connect(self.on_project_selected)
        layout.addWidget(self.project_combo)

        details_label = QLabel("Project Details:")
        layout.addWidget(details_label)

        self.details_table = QTableWidget(0, 2)
        self.details_table.setHorizontalHeaderLabels(["Field", "Value"])
        self.details_table.horizontalHeader().setStretchLastSection(True)
        self.details_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.details_table.setSelectionMode(QAbstractItemView.NoSelection)
        self.details_table.setFocusPolicy(Qt.NoFocus)
        self.details_table.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.details_table.verticalHeader().setVisible(False)
        layout.addWidget(self.details_table, 1)  # Add stretch factor

        # Options group box
        options_group = QGroupBox("Options")
        options_layout = QVBoxLayout()
        options_group.setLayout(options_layout)
        layout.addWidget(options_group)

        # Row for filter expression
        filter_row = QHBoxLayout()
        filter_label = QLabel("Filters:")
        filter_row.addWidget(filter_label)
        self.filter_edit = QLineEdit()
        self.filter_edit.setPlaceholderText('list of IDs or filter query (e.g., superclass == "central")')
        # Load last saved filter
        last_filter = self.settings.value("last_project_filter", "")
        if last_filter:
            self.filter_edit.setText(last_filter)
        # Connect textChanged signal to validation
        self.filter_edit.textChanged.connect(self.validate_filter)
        filter_row.addWidget(self.filter_edit)

        self.filter_help_btn = QPushButton("?")
        self.filter_help_btn.setFixedSize(25, 25)
        self.filter_help_btn.setToolTip("Available columns for filtering")
        filter_row.addWidget(self.filter_help_btn)

        options_layout.addLayout(filter_row)

        self.filter_hint_label = QLabel(
            "Tip: for filtered datasets, recomputing embeddings is usually better than using global precomputed embeddings. "
        )
        self.filter_hint_label.setWordWrap(True)
        self.filter_hint_label.setStyleSheet("color: #8a8a8a;")
        self.filter_hint_label.setVisible(bool(self.filter_edit.text().strip()))
        options_layout.addWidget(self.filter_hint_label)

        # Row with dropdown for embeddings
        emb_row = QHBoxLayout()
        emb_label = QLabel("Embeddings:")
        emb_row.addWidget(emb_label)
        self.embedding_combo = QComboBox()
        self.embedding_combo.currentTextChanged.connect(self.update_embedding_hint)
        emb_row.addWidget(self.embedding_combo)
        options_layout.addLayout(emb_row)

        self.embedding_hint_label = QLabel(
            'Tip: open the "Embeddings" tab in the control panel to fine-tune the embeddings.'
        )
        self.embedding_hint_label.setWordWrap(True)
        self.embedding_hint_label.setStyleSheet("color: #8a8a8a;")
        self.embedding_hint_label.setVisible(False)
        options_layout.addWidget(self.embedding_hint_label)

        # Dialog buttons at the very bottom
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.load_btn = buttons.button(QDialogButtonBox.Ok)
        self.load_btn.setText("Load")
        buttons.button(QDialogButtonBox.Cancel).setText("Cancel")
        buttons.accepted.connect(self.on_accept)
        buttons.rejected.connect(self.reject)

        layout.addWidget(buttons)

        self.setLayout(layout)

        # Once everything is setup, we can scan the last used path
        if last_path:
            self.scan_path()

        last_project = self.settings.value("last_project_index", -1, type=int)
        if last_project >= 0:
            self.project_combo.setCurrentIndex(last_project)

        if self.initial_state:
            self.apply_state(self.initial_state)

    def open_file_dialog(self):
        directory = QFileDialog.getExistingDirectory(self, "Select Project Directory")
        if directory:
            self.path_edit.setCurrentText(directory)

    def on_project_selected(self, index):
        """Handle project selection and display details."""
        if index < 0 or index >= len(self.projects):
            self.details_table.setRowCount(0)
            self.filter_help_btn.setToolTip("Available columns for filtering")
            return

        project = self.projects[index]
        # Populate project details into table
        self.details_table.setRowCount(0)
        if hasattr(project, "info") and isinstance(project.info, dict) and project.info:
            for row_idx, (k, v) in enumerate(project.info.items()):
                self.details_table.insertRow(row_idx)
                self.details_table.setItem(row_idx, 0, QTableWidgetItem(str(k)))
                self.details_table.setItem(row_idx, 1, QTableWidgetItem(str(v)))
        else:
            self.details_table.insertRow(0)
            self.details_table.setItem(0, 0, QTableWidgetItem("Info"))
            self.details_table.setItem(0, 1, QTableWidgetItem("No details available"))
        self.details_table.resizeColumnsToContents()

        # Add options for embeddings
        self.embedding_combo.clear()
        if project.info.get("embeddings", None):
            self.embedding_combo.addItem("use precomputed", 0)
        if project.info.get("distances", None):
            self.embedding_combo.addItem("calculate from distances", 0)
        if project.info.get("features", None):
            self.embedding_combo.addItem("calculate from features", 0)
        self.update_embedding_hint()

        # Update filter help tooltip with available columns
        try:
            columns = list(project.meta_columns)
            tooltip = "Available columns:\n" + "\n".join(columns)
            self.filter_help_btn.setToolTip(tooltip)
        except Exception:
            self.filter_help_btn.setToolTip("Error retrieving columns")

    def scan_path(self):
        """Validate the path by parsing it. Show red outline on error."""
        path = self.path_edit.currentText().strip()
        logger.info("Scanning path: %s", path)
        if not path:
            # Clear styling if field is empty
            self.path_edit.setStyleSheet("")
            self._clear_path_error()
            self.project_combo.clear()
            self.embedding_combo.clear()
            self.details_table.setRowCount(0)
            return

        try:
            self.projects = parse_directory(path)

            if isinstance(self.projects, SingleProjectLoader):
                self.projects = [self.projects]

            # Valid path - clear red outline and populate projects
            self.path_edit.setStyleSheet("")
            self._clear_path_error()
            self.project_combo.clear()
            self.embedding_combo.clear()
            self.details_table.setRowCount(0)
            # Add projects to dropdown
            for i, project in enumerate(self.projects):
                self.project_combo.addItem(project.name, i)
        except Exception as e:
            # Invalid path - add red outline and show error label
            logger.debug(f"Error validating path: {e}")
            self.path_edit.setStyleSheet(
                "QComboBox { border: 2px solid red; border-radius: 3px; }"
            )
            self._set_path_error(self._format_path_error(e))
            self.project_combo.clear()
            self.embedding_combo.clear()
            self.details_table.setRowCount(0)

    def _set_path_error(self, message):
        self.path_error_label.setText(message)
        self.path_error_label.setVisible(True)

    def _clear_path_error(self):
        self.path_error_label.setText("")
        self.path_error_label.setVisible(False)

    def _format_path_error(self, exc):
        if isinstance(exc, FileNotFoundError):
            text = str(exc)
            if "No 'info' file found" in text:
                return "No info file found in selected directory. Make sure the dataset folder contains an 'info' file."
            if "Path does not exist" in text:
                return "The path does not exist. Enter a valid local folder or URL."
            return "Path not found or inaccessible."
        if isinstance(exc, json.JSONDecodeError):
            return "Malformed info file. The 'info' file must contain valid JSON."
        if isinstance(exc, ValueError):
            if "Invalid info format" in str(exc):
                return "Malformed info file. Expected a JSON object or list in the 'info' file."
            return f"Invalid dataset info: {exc}"
        if isinstance(exc, requests.exceptions.RequestException):
            return "Unable to load remote info file. Check the URL and your network connection."
        return "Unable to parse path. Use a valid local dataset directory or remote URL."

    def validate_filter(self):
        """Validate the filter expression. Show red outline on error."""
        filter_expr = self.filter_edit.text().strip()
        self.filter_hint_label.setVisible(bool(filter_expr))
        if not filter_expr:
            # Clear styling if field is empty
            self.filter_edit.setStyleSheet("")
            self.load_btn.setEnabled(True)
            return

        try:
            if is_list_of_ids(filter_expr):
                # Handle list of IDs
                pass
            else:
                string_to_polars_filter(filter_expr)
            # Valid filter - clear red outline and enable Load button
            self.filter_edit.setStyleSheet("")
            self.load_btn.setEnabled(True)
        except Exception as e:
            # Invalid filter - add red outline and disable Load button
            logger.debug(f"Invalid filter expression: {e}")
            self.filter_edit.setStyleSheet(
                "QLineEdit { border: 2px solid red; border-radius: 3px; }"
            )
            self.load_btn.setEnabled(False)

    def update_embedding_hint(self):
        """Show guidance when embeddings are recomputed."""
        selected = self.embedding_combo.currentText().strip().lower()
        show_hint = bool(selected) and selected != "use precomputed"
        self.embedding_hint_label.setVisible(show_hint)

    def on_accept(self):
        """Save the path before accepting the dialog."""
        path = self.path_edit.currentText().strip()
        if path:
            self.settings.setValue("last_project_path", path)

            existing_paths = self.settings.value("last_project_paths", [])
            if isinstance(existing_paths, str):
                history = [existing_paths] if existing_paths else []
            elif isinstance(existing_paths, (list, tuple)):
                history = [str(p).strip() for p in existing_paths if str(p).strip()]
            else:
                history = []

            history = [p for p in history if p != path]
            history.insert(0, path)
            self.settings.setValue("last_project_paths", history[:10])

        project = self.project_combo.currentIndex()
        if project >= 0:
            self.settings.setValue("last_project_index", project)
        filter_expr = self.filter_edit.text().strip()
        self.settings.setValue("last_project_filter", filter_expr)
        self.accept()

    def apply_state(self, state):
        """Apply a previously captured dialog state."""
        if not isinstance(state, dict):
            return

        path = str(state.get("path", "")).strip()
        if path:
            self.path_edit.setCurrentText(path)
        else:
            self.scan_path()

        project_name = str(state.get("project_name", "")).strip()
        if project_name:
            idx = self.project_combo.findText(project_name)
            if idx >= 0:
                self.project_combo.setCurrentIndex(idx)
        elif "project_index" in state:
            try:
                idx = int(state.get("project_index", -1))
                if 0 <= idx < self.project_combo.count():
                    self.project_combo.setCurrentIndex(idx)
            except (TypeError, ValueError):
                pass

        filter_expr = str(state.get("filter_expr", "")).strip()
        self.filter_edit.setText(filter_expr)

        embedding_mode = str(state.get("embedding_mode", "")).strip()
        if embedding_mode:
            emb_idx = self.embedding_combo.findText(embedding_mode)
            if emb_idx >= 0:
                self.embedding_combo.setCurrentIndex(emb_idx)

    def current_state(self):
        """Capture the full current dialog state for reuse."""
        path = self.selected_path()
        return {
            "path": path,
            "source_type": "remote" if path.startswith(("http://", "https://")) else "local",
            "project_index": self.project_combo.currentIndex(),
            "project_name": self.project_combo.currentText(),
            "filter_expr": self.selected_filter(),
            "embedding_mode": self.embedding_combo.currentText(),
        }

    def selected_path(self):
        return self.path_edit.currentText().strip()

    def selected_filter(self):
        """Get the filter expression."""
        return self.filter_edit.text().strip()

    def selected_project_loader(self):
        """Get the selected project loader."""
        index = self.project_combo.currentIndex()
        if index < 0 or index >= len(self.projects):
            return None
        p = self.projects[index]
        if self.filter_edit.text().strip():
            p.filter_expr = self.filter_edit.text().strip()
        return p
