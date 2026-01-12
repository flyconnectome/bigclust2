import logging

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

from ..data import parse_directory, MultiProjectLoader, SingleProjectLoader
from ..utils import string_to_polars_filter, is_list_of_ids

logger = logging.getLogger(__name__)


__all__ = ["OpenProjectDialog"]


class OpenProjectDialog(QDialog):
    """Simple dialog to choose a project source."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Open Project")
        self.setModal(True)
        self.resize(600, 480)
        self.settings = QSettings("BigClust", "BigClustGUI")
        self.projects = []  # Store parsed projects
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        label = QLabel("Choose local or remote data source")
        layout.addWidget(label)

        # Path entry with browse button
        path_row = QHBoxLayout()
        self.path_edit = QLineEdit()
        self.path_edit.setPlaceholderText("Enter path or URL")
        # Load last saved path
        last_path = self.settings.value("last_project_path", "")
        if last_path:
            self.path_edit.setText(last_path)
        # Connect textChanged signal to validation
        self.path_edit.textChanged.connect(self.scan_path)
        browse_btn = QPushButton("Browse…")
        browse_btn.setToolTip("Select a local folder")
        browse_btn.clicked.connect(self.open_file_dialog)

        path_row.addWidget(self.path_edit)
        path_row.addWidget(browse_btn)
        layout.addLayout(path_row)

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

        # Row with dropdown for embeddings
        emb_row = QHBoxLayout()
        emb_label = QLabel("Embeddings:")
        emb_row.addWidget(emb_label)
        self.embedding_combo = QComboBox()
        emb_row.addWidget(self.embedding_combo)
        options_layout.addLayout(emb_row)

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

    def open_file_dialog(self):
        directory = QFileDialog.getExistingDirectory(self, "Select Project Directory")
        if directory:
            self.path_edit.setText(directory)

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

        # Update filter help tooltip with available columns
        try:
            columns = list(project.meta_columns)
            tooltip = "Available columns:\n" + "\n".join(columns)
            self.filter_help_btn.setToolTip(tooltip)
        except Exception:
            self.filter_help_btn.setToolTip("Error retrieving columns")

    def scan_path(self):
        """Validate the path by parsing it. Show red outline on error."""
        path = self.path_edit.text().strip()
        logger.info("Scanning path: %s", path)
        if not path:
            # Clear styling if field is empty
            self.path_edit.setStyleSheet("")
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
            self.project_combo.clear()
            self.embedding_combo.clear()
            self.details_table.setRowCount(0)
            # Add projects to dropdown
            for i, project in enumerate(self.projects):
                self.project_combo.addItem(project.name, i)
        except Exception as e:
            # Invalid path - add red outline and print error
            print(f"Error validating path: {e}")
            self.path_edit.setStyleSheet(
                "QLineEdit { border: 2px solid red; border-radius: 3px; }"
            )
            self.project_combo.clear()
            self.embedding_combo.clear()
            self.details_table.setRowCount(0)

    def validate_filter(self):
        """Validate the filter expression. Show red outline on error."""
        filter_expr = self.filter_edit.text().strip()
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
            logger.warning(f"Invalid filter expression: {e}")
            self.filter_edit.setStyleSheet(
                "QLineEdit { border: 2px solid red; border-radius: 3px; }"
            )
            self.load_btn.setEnabled(False)

    def on_accept(self):
        """Save the path before accepting the dialog."""
        path = self.path_edit.text().strip()
        if path:
            self.settings.setValue("last_project_path", path)
        project = self.project_combo.currentIndex()
        if project >= 0:
            self.settings.setValue("last_project_index", project)
        filter_expr = self.filter_edit.text().strip()
        if filter_expr:
            self.settings.setValue("last_project_filter", filter_expr)
        self.accept()

    def selected_path(self):
        return self.path_edit.text().strip()

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
