from __future__ import annotations

import json

from dataclasses import asdict, dataclass

from PySide6 import QtCore, QtGui, QtWidgets

try:
    from .annotation_backends import BACKEND_REGISTRY
except ImportError:
    # For testing we need absolute imports
    from annotation_backends import BACKEND_REGISTRY


@dataclass(frozen=True)
class SelectionRecord:
    """Minimal selected-neuron record used by the prototype dialog."""

    neuron_id: int
    dataset: str


class FieldChipInput(QtWidgets.QWidget):
    """Compact chip input that tokenizes on comma/enter."""

    changed = QtCore.Signal()

    def __init__(self, parent=None):
        """Initialize chip input state and child widgets."""
        super().__init__(parent)
        self._chips = []
        self._chip_buttons = {}
        self._invalid_chips = set()

        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        self._chips_layout = QtWidgets.QHBoxLayout()
        self._chips_layout.setContentsMargins(0, 0, 0, 0)
        self._chips_layout.setSpacing(4)

        chips_container = QtWidgets.QWidget()
        chips_container.setLayout(self._chips_layout)
        layout.addWidget(chips_container)

        self._line_edit = QtWidgets.QLineEdit()
        self._line_edit.setPlaceholderText("Fields (comma-separated)")
        self._line_edit.textChanged.connect(self._on_text_changed)
        self._line_edit.returnPressed.connect(self._commit_current_text)
        self._line_edit.installEventFilter(self)
        layout.addWidget(self._line_edit, stretch=1)

    def eventFilter(self, obj, event):
        """Commit pending text when the line edit loses focus."""
        if obj is self._line_edit and event.type() == QtCore.QEvent.FocusOut:
            self._commit_current_text()
        return super().eventFilter(obj, event)

    def set_placeholder_text(self, text):
        """Set placeholder text for the chip entry line edit."""
        self._line_edit.setPlaceholderText(text)

    def set_tooltip(self, text):
        """Set tooltip text for the chip entry line edit."""
        self._line_edit.setToolTip(text)

    def _on_text_changed(self, text):
        """Tokenize chips on commas while preserving the trailing fragment."""
        if "," not in text:
            return

        # Convert all completed comma-separated tokens to chips,
        # and keep only the unfinished trailing token in the editor.
        parts = text.split(",")
        for part in parts[:-1]:
            self._add_chip(part)
        self._line_edit.blockSignals(True)
        self._line_edit.setText(parts[-1])
        self._line_edit.blockSignals(False)
        self.changed.emit()

    def _commit_current_text(self):
        """Turn current line-edit text into a chip and clear the input."""
        self._add_chip(self._line_edit.text())
        self._line_edit.clear()
        self.changed.emit()

    def _add_chip(self, raw_text):
        """Add a unique chip button from raw text input."""
        text = raw_text.strip()
        if not text:
            return
        if text in self._chips:
            return

        self._chips.append(text)
        button = QtWidgets.QToolButton()
        button.setText(f"{text}  x")
        button.setCursor(QtCore.Qt.PointingHandCursor)
        self._apply_chip_style(button, is_invalid=text in self._invalid_chips)
        button.clicked.connect(
            lambda _checked=False, t=text, b=button: self._remove_chip(t, b)
        )
        self._chips_layout.addWidget(button)
        self._chip_buttons[text] = button

    def _apply_chip_style(self, button, is_invalid=False):
        """Apply valid or invalid visual styling to a chip button."""
        if is_invalid:
            button.setStyleSheet(
                "QToolButton {"
                " border: 1px solid #c62828;"
                " border-radius: 10px;"
                " padding: 2px 8px;"
                " background: #e53935;"
                " color: white;"
                " font-weight: 500;"
                " }"
                "QToolButton:hover {"
                " background: #d32f2f;"
                " }"
                "QToolButton:pressed {"
                " background: #b71c1c;"
                " }"
            )
            return

        button.setStyleSheet(
            "QToolButton {"
            " border: 1px solid #2f7d32;"
            " border-radius: 10px;"
            " padding: 2px 8px;"
            " background: #43a047;"
            " color: white;"
            " font-weight: 500;"
            " }"
            "QToolButton:hover {"
            " background: #388e3c;"
            " }"
            "QToolButton:pressed {"
            " background: #2e7d32;"
            " }"
        )

    def _remove_chip(self, text, button):
        """Remove a chip from state and delete its button widget."""
        if text in self._chips:
            self._chips.remove(text)
        self._chip_buttons.pop(text, None)
        self._invalid_chips.discard(text)
        button.deleteLater()
        self.changed.emit()

    def chips(self):
        """Return committed chips plus any pending typed value."""
        pending = self._line_edit.text().strip()
        if pending:
            return list(dict.fromkeys([*self._chips, pending]))
        return list(self._chips)

    def set_chips(self, values):
        """Replace all chips with the provided values."""
        for i in reversed(range(self._chips_layout.count())):
            item = self._chips_layout.takeAt(i)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        self._chips = []
        self._chip_buttons = {}
        self._invalid_chips = set()
        for value in values:
            self._add_chip(value)
        self.changed.emit()

    def set_invalid_chips(self, values):
        """Mark a subset of chips as invalid and refresh styles."""
        self._invalid_chips = {str(v).strip() for v in values if str(v).strip()}
        for chip_text, button in self._chip_buttons.items():
            self._apply_chip_style(button, is_invalid=chip_text in self._invalid_chips)


class FunctionRunnable(QtCore.QRunnable):
    """Simple QRunnable wrapper that executes a callable in a thread pool."""

    def __init__(self, fn, *args, **kwargs):
        """Store callable and invocation arguments for background execution."""
        super().__init__()
        self._fn = fn
        self._args = args
        self._kwargs = kwargs

    def run(self):
        """Run the wrapped callable."""
        self._fn(*self._args, **self._kwargs)


class AnnotationDialog(QtWidgets.QDialog):
    """Prototype annotation dialog for selecting backends and field writes.

    This widget is intentionally self-contained so it can be iterated in
    isolation before being wired into the main BigClust flow.
    """

    submitted = QtCore.Signal(dict)
    _session_config_validated = False

    UNSELECTED_BACKEND = "-- Select backend --"
    NOT_WRITEABLE_BACKEND = "Not Writeable"
    WRITEABLE_BACKENDS = BACKEND_REGISTRY
    BACKEND_CHOICES = (
        UNSELECTED_BACKEND,
        NOT_WRITEABLE_BACKEND,
        *WRITEABLE_BACKENDS.keys(),
    )

    SETTINGS_ORG = "bigclust2"
    SETTINGS_APP = "AnnotationDialog"
    SETTINGS_KEY_DATASET_CONFIGS = "annotation/dataset_configs"
    SETTINGS_KEY_RECENT_SUBMIT_PLANS = "annotation/recent_submit_plans"
    MAX_RECENT_SUBMIT_PLANS = 5

    def __init__(
        self,
        selection: list[SelectionRecord],
        parent=None,
        project_datasets: list[str] | None = None,
    ):
        """Initialize dialog state, load persisted settings, and build UI."""
        super().__init__(parent)
        self.selection = selection
        self._project_datasets = self._normalize_project_datasets(project_datasets)
        self._dataset_backends = {}
        self._dataset_row_index = {}
        self._dataset_config_inputs = {}
        self._submit_dataset_rows = {}
        self._validated_backends = {}
        self._last_invalid_fields_by_dataset = {}
        self._fallback_threadpool = QtCore.QThreadPool(self)
        self._settings = QtCore.QSettings(self.SETTINGS_ORG, self.SETTINGS_APP)
        self._saved_dataset_configs = self._load_saved_dataset_configs()
        self._recent_submit_plans = self._load_recent_submit_plans()
        self._validation_required = not self.__class__._session_config_validated

        self.setWindowTitle("Push Annotations")
        self.setModal(True)
        self.resize(860, 620)

        self._build_ui()
        self._populate_dataset_table()
        self._update_summary()
        self._update_submit_state()
        self._set_initial_tab()

    def _set_initial_tab(self):
        """Open Submit tab only when session validation is already satisfied."""
        if self.__class__._session_config_validated and self._all_datasets_configured():
            self.tabs.setCurrentIndex(self._submit_tab_index)
        else:
            self.tabs.setCurrentIndex(self._config_tab_index)

    def _normalize_project_datasets(self, datasets):
        """Normalize an optional project dataset list to unique sorted names."""
        if not datasets:
            return []
        return sorted(
            {
                str(dataset).strip()
                for dataset in datasets
                if str(dataset).strip()
            }
        )

    def set_selection(
        self,
        selection: list[SelectionRecord],
        project_datasets: list[str] | None = None,
    ):
        """Replace selection records and refresh dataset/submit state."""
        self.selection = list(selection)
        if project_datasets is not None:
            self._project_datasets = self._normalize_project_datasets(project_datasets)
        self._populate_dataset_table()
        self._update_summary()
        self._update_submit_state()

    def _build_ui(self):
        """Build the top-level tab container and register both tabs."""
        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(16, 16, 16, 16)
        root.setSpacing(12)

        self.tabs = QtWidgets.QTabWidget()
        root.addWidget(self.tabs, stretch=1)

        # Keep each tab's construction isolated in dedicated helpers.
        submit_tab = self._build_submit_tab()
        config_tab = self._build_config_tab()

        self._submit_tab_index = self.tabs.addTab(submit_tab, "Submit annotations")
        self._config_tab_index = self.tabs.addTab(config_tab, "Configuration")

        # Starts disabled until all datasets are mapped to backends.
        self.tabs.setTabEnabled(self._submit_tab_index, False)
        self.tabs.setCurrentIndex(self._config_tab_index)

    def _build_submit_tab(self):
        """Construct and return the submit tab widget hierarchy."""
        submit_tab = QtWidgets.QWidget()
        submit_tab_layout = QtWidgets.QVBoxLayout(submit_tab)
        submit_tab_layout.setContentsMargins(0, 8, 0, 0)
        submit_tab_layout.setSpacing(8)

        value_group = QtWidgets.QGroupBox("Value")
        value_layout = QtWidgets.QVBoxLayout(value_group)
        value_layout.setContentsMargins(10, 12, 10, 10)
        value_layout.setSpacing(8)

        self.shared_value_edit = QtWidgets.QLineEdit()
        self.shared_value_edit.setPlaceholderText("e.g. DA1_lPN")
        self.shared_value_edit.setStyleSheet(
            "QLineEdit {"
            " border: 2px solid #4a90e2;"
            " border-radius: 6px;"
            " padding: 4px 8px;"
            " background: #f7fbff;"
            " color: #111111;"
            " }"
            "QLineEdit:focus {"
            " border: 2px solid #1e6fd9;"
            " background: white;"
            " color: #111111;"
            " }"
            "QLineEdit:disabled {"
            " border: 2px solid #cbd5e1;"
            " background: #e5e7eb;"
            " color: #6b7280;"
            " }"
        )
        self.shared_value_edit.textChanged.connect(self._update_submit_state)

        self.clear_fields_checkbox = QtWidgets.QCheckBox("Clear fields")
        self.clear_fields_checkbox.setToolTip(
            "When checked, the specified fields will be cleared (set to empty string) instead of being set to the shared value. "
            "This allows you to clear existing annotations by selecting a shared value and enabling this option."
        )
        self.clear_fields_checkbox.toggled.connect(self._on_clear_mode_toggled)
        self.clear_fields_checkbox.setSizePolicy(
            QtWidgets.QSizePolicy.Maximum,
            QtWidgets.QSizePolicy.Fixed,
        )

        value_row = QtWidgets.QHBoxLayout()
        value_row.setContentsMargins(0, 0, 0, 0)
        value_row.setSpacing(8)
        value_row.addWidget(self.shared_value_edit, stretch=1)
        value_row.addWidget(
            self.clear_fields_checkbox,
            0,
            QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter,
        )
        value_layout.addLayout(value_row)

        submit_tab_layout.addWidget(value_group)

        fields_group = QtWidgets.QGroupBox("Fields")
        fields_layout = QtWidgets.QVBoxLayout(fields_group)
        fields_layout.setContentsMargins(10, 12, 10, 10)
        fields_layout.setSpacing(8)

        fields_help = QtWidgets.QLabel(
            "Set one value at the top, then choose target fields per dataset. "
            "Use commas for multiple fields (e.g. type, flywire_type)."
        )
        fields_help.setWordWrap(True)
        fields_help.setStyleSheet("font-size: 11px; color: #6b7280;")
        fields_layout.addWidget(fields_help)

        dataset_fields_scroll = QtWidgets.QScrollArea()
        dataset_fields_scroll.setWidgetResizable(True)
        dataset_fields_scroll.setFrameShape(QtWidgets.QFrame.NoFrame)

        self.submit_dataset_container = QtWidgets.QWidget()
        self.submit_dataset_layout = QtWidgets.QVBoxLayout(
            self.submit_dataset_container
        )
        self.submit_dataset_layout.setContentsMargins(0, 0, 0, 0)
        self.submit_dataset_layout.setSpacing(6)
        self.submit_dataset_layout.addStretch(1)
        dataset_fields_scroll.setWidget(self.submit_dataset_container)
        fields_layout.addWidget(dataset_fields_scroll, stretch=1)

        recent_group = QtWidgets.QGroupBox(
            "Re-use recent field plans. Click row to apply it."
        )
        recent_layout = QtWidgets.QVBoxLayout(recent_group)
        recent_layout.setContentsMargins(8, 8, 8, 8)
        recent_layout.setSpacing(6)

        self.recent_plans_table = QtWidgets.QTableWidget(0, 0)
        self.recent_plans_table.setEditTriggers(
            QtWidgets.QAbstractItemView.NoEditTriggers
        )
        self.recent_plans_table.setSelectionBehavior(
            QtWidgets.QAbstractItemView.SelectRows
        )
        self.recent_plans_table.setSelectionMode(
            QtWidgets.QAbstractItemView.SingleSelection
        )
        self.recent_plans_table.setStyleSheet(
            "QTableWidget::item { color: black; }"
            "QTableWidget::item:selected { color: white; }"
        )
        self.recent_plans_table.verticalHeader().setVisible(True)
        recent_header = self.recent_plans_table.horizontalHeader()
        recent_header.setStretchLastSection(False)
        recent_header.setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
        self.recent_plans_table.setMaximumHeight(160)
        self.recent_plans_table.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding,
            QtWidgets.QSizePolicy.Maximum,
        )
        self.recent_plans_table.cellClicked.connect(self._on_recent_plan_clicked)
        recent_layout.addWidget(self.recent_plans_table)
        recent_group.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding,
            QtWidgets.QSizePolicy.Maximum,
        )
        fields_layout.addWidget(recent_group)
        submit_tab_layout.addWidget(fields_group)

        submit_bottom_row = QtWidgets.QHBoxLayout()
        self.submit_status_label = QtWidgets.QLabel("")
        self.submit_status_label.setWordWrap(True)
        submit_bottom_row.addWidget(self.submit_status_label, stretch=1)
        submit_cancel_button = QtWidgets.QPushButton("Cancel")
        submit_cancel_button.clicked.connect(self.reject)
        submit_bottom_row.addWidget(submit_cancel_button)
        self.submit_button = QtWidgets.QPushButton("Submit")
        self.submit_button.clicked.connect(self._on_submit)
        submit_bottom_row.addWidget(self.submit_button)
        submit_tab_layout.addLayout(submit_bottom_row)

        return submit_tab

    def _build_config_tab(self):
        """Construct and return the configuration tab widget hierarchy."""
        config_tab = QtWidgets.QWidget()
        config_layout = QtWidgets.QVBoxLayout(config_tab)
        config_layout.setContentsMargins(0, 8, 0, 0)
        config_layout.setSpacing(8)

        backend_help = QtWidgets.QLabel(
            "Choose one repository/backend for each dataset in the current project. "
            "You'll only have to do this once per session, and the dialog will remember your choices for next time. "
        )
        backend_help.setWordWrap(True)
        config_layout.addWidget(backend_help)

        self.group_same_backend_checkbox = QtWidgets.QCheckBox(
            "Group same backend/repository"
        )
        self.group_same_backend_checkbox.setChecked(True)
        self.group_same_backend_checkbox.toggled.connect(self._on_grouping_toggled)
        config_layout.addWidget(self.group_same_backend_checkbox)

        self.dataset_table = QtWidgets.QTableWidget(0, 4)
        self.dataset_table.setHorizontalHeaderLabels(
            [
                "Dataset",
                "Selected neurons",
                "Backend",
                "Configuration",
            ]
        )
        self.dataset_table.horizontalHeaderItem(3).setTextAlignment(
            QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter
        )
        self.dataset_table.verticalHeader().setVisible(False)
        self.dataset_table.setShowGrid(True)
        self.dataset_table.setGridStyle(QtCore.Qt.SolidLine)
        self.dataset_table.setStyleSheet(
            "QTableView { gridline-color: rgba(140, 140, 140, 0.6); }"
            "QTableView::item { padding-top: 6px; padding-bottom: 6px; }"
        )
        self.dataset_table.setSelectionMode(QtWidgets.QAbstractItemView.NoSelection)
        self.dataset_table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        header = self.dataset_table.horizontalHeader()
        header.setStretchLastSection(True)
        self.dataset_table.horizontalHeader().setSectionResizeMode(
            0, QtWidgets.QHeaderView.ResizeToContents
        )
        self.dataset_table.horizontalHeader().setSectionResizeMode(
            1, QtWidgets.QHeaderView.ResizeToContents
        )
        self.dataset_table.horizontalHeader().setSectionResizeMode(
            2, QtWidgets.QHeaderView.ResizeToContents
        )
        self.dataset_table.horizontalHeader().setSectionResizeMode(
            3, QtWidgets.QHeaderView.Stretch
        )
        config_layout.addWidget(self.dataset_table)

        self.config_status_label = QtWidgets.QLabel("")
        self.config_status_label.setWordWrap(True)
        config_layout.addWidget(self.config_status_label)

        config_actions = QtWidgets.QHBoxLayout()
        config_actions.addStretch(1)
        config_cancel_button = QtWidgets.QPushButton("Cancel")
        config_cancel_button.clicked.connect(self.reject)
        config_actions.addWidget(config_cancel_button)
        self.validate_button = QtWidgets.QPushButton("Validate")
        self.validate_button.setEnabled(False)
        self.validate_button.clicked.connect(self._on_validate_config)
        config_actions.addWidget(self.validate_button)
        config_layout.addLayout(config_actions)

        return config_tab

    def _all_datasets_configured(self):
        """Return True when every dataset has complete backend configuration."""
        if not self._dataset_backends:
            return False
        return all(
            self._dataset_has_complete_config(ds) for ds in self._dataset_backends
        )

    def _on_validate_config(self):
        """Validate configured backends and unlock submit flow on success."""
        if not self._all_datasets_configured():
            self.config_status_label.setText(
                "Press Validate in Configuration to continue."
            )
            QtWidgets.QMessageBox.warning(
                self,
                "Configuration incomplete",
                "Fill required backend configuration for every dataset, then try Validate again.",
            )
            return

        ok, error_message = self._validate_backend_configs()
        if not ok:
            self.__class__._session_config_validated = False
            self._validation_required = True
            self.config_status_label.setText(
                "Press Validate in Configuration to continue."
            )
            self._update_submit_state()
            self.tabs.setCurrentIndex(self._config_tab_index)
            QtWidgets.QMessageBox.critical(
                self,
                "Backend validation failed",
                f"{error_message}\n\nPlease fix the configuration and try Validate again.",
            )
            return

        self.__class__._session_config_validated = True
        self._validation_required = False
        self.config_status_label.setText("Configuration is valid.")
        self._update_submit_state()
        self.tabs.setCurrentIndex(self._submit_tab_index)

    def _build_backend_instance(self, dataset):
        """Instantiate the runtime backend object for one dataset."""
        backend_name = self._dataset_backends.get(dataset, "")
        backend_cls = self._backend_class(backend_name)
        if backend_cls is None:
            return None

        config = self._build_dataset_config(dataset)
        if config is None:
            return None

        return backend_cls(config=config)

    def _backend_class(self, backend_name):
        """Return runtime backend class for a backend name."""
        return self.WRITEABLE_BACKENDS.get(backend_name)

    def _validate_backend_configs(self):
        """Validate backend configs and cache dataset-to-backend instances."""
        instances = {}
        unique_instances = {}
        for dataset in self._writable_datasets():
            backend_name = self._dataset_backends.get(dataset, "")
            config = self._build_dataset_config(dataset)
            if config is None:
                return (
                    False,
                    f"Validation failed for {dataset}: incomplete backend configuration.",
                )

            backend_cls = self._backend_class(backend_name)
            if backend_cls is None:
                return (
                    False,
                    f"Validation failed for {dataset}: unsupported backend '{backend_name}'.",
                )

            # Reuse one runtime backend for identical (backend, config) pairs.
            instance_key = (backend_name, config)
            backend = unique_instances.get(instance_key)
            if backend is None:
                backend = backend_cls(config=config)
                try:
                    backend.validate()
                except Exception as exc:
                    detail = str(exc).strip() or exc.__class__.__name__
                    return (
                        False,
                        f"Validation failed for {dataset} ({backend_name}): {detail}",
                    )
                unique_instances[instance_key] = backend

            instances[dataset] = backend

        self._validated_backends = instances
        return True, ""

    def _load_saved_dataset_configs(self):
        """Load and normalize persisted per-dataset backend configurations."""
        raw = self._settings.value(self.SETTINGS_KEY_DATASET_CONFIGS, "{}")
        if isinstance(raw, dict):
            return raw

        if raw is None:
            return {}

        if not isinstance(raw, str):
            raw = str(raw)

        try:
            data = json.loads(raw)
        except Exception:
            return {}

        if not isinstance(data, dict):
            return {}

        normalized = {}
        for dataset, entry in data.items():
            if not isinstance(entry, dict):
                continue

            backend = entry.get("backend", "")
            config = entry.get("config", {})
            if not isinstance(backend, str):
                backend = str(backend)
            if not isinstance(config, dict):
                config = {}

            normalized[str(dataset)] = {
                "backend": backend,
                "config": {str(k): v for k, v in config.items()},
            }

        return normalized

    def _load_recent_submit_plans(self):
        """Load and sanitize recent submit plans from settings storage."""
        raw = self._settings.value(self.SETTINGS_KEY_RECENT_SUBMIT_PLANS, "[]")
        if raw is None:
            return []
        if not isinstance(raw, str):
            raw = str(raw)

        try:
            data = json.loads(raw)
        except Exception:
            return []

        if not isinstance(data, list):
            return []

        normalized = []
        for plan in data:
            if not isinstance(plan, dict):
                continue
            cleaned = {}
            for dataset, field_names in plan.items():
                if not isinstance(field_names, list):
                    continue
                fields_clean = [str(f).strip() for f in field_names if str(f).strip()]
                if fields_clean:
                    cleaned[str(dataset)] = list(dict.fromkeys(fields_clean))
            if cleaned:
                normalized.append(cleaned)

        return normalized[: self.MAX_RECENT_SUBMIT_PLANS]

    def _on_grouping_toggled(self, _checked):
        """Rebuild submit rows when grouping mode changes."""
        self._populate_submit_dataset_rows()
        self._update_submit_state()

    def _invalidate_backend_validation(self):
        """Clear cached backend validation and require re-validation."""
        self._validated_backends = {}
        self.__class__._session_config_validated = False
        self._validation_required = True

    def _on_config_inputs_changed(self):
        """Handle config edits by invalidating validation and refreshing UI."""
        self._invalidate_backend_validation()
        if self.group_same_backend_checkbox.isChecked():
            self._populate_submit_dataset_rows()
        self._update_submit_state()

    def _save_recent_submit_plans(self):
        """Persist recent submit plans to settings."""
        self._settings.setValue(
            self.SETTINGS_KEY_RECENT_SUBMIT_PLANS,
            json.dumps(self._recent_submit_plans),
        )

    def _remember_recent_submit_plan(self):
        """Store the current submit field mapping in recent plans."""
        plan = self._collect_submit_dataset_field_map()
        if not plan:
            return

        # Deduplicate semantically identical plans and keep newest first.
        serialized = json.dumps(plan, sort_keys=True)
        pruned = [
            existing
            for existing in self._recent_submit_plans
            if json.dumps(existing, sort_keys=True) != serialized
        ]
        pruned.insert(0, plan)
        self._recent_submit_plans = pruned[: self.MAX_RECENT_SUBMIT_PLANS]
        self._save_recent_submit_plans()

    def _refresh_recent_plans_table(self):
        """Render the recent-plan table from persisted plan data."""
        datasets = self._writable_datasets()
        table = self.recent_plans_table
        table.blockSignals(True)
        table.clear()
        table.setColumnCount(len(datasets))
        table.setHorizontalHeaderLabels(datasets)
        table.setRowCount(
            min(len(self._recent_submit_plans), self.MAX_RECENT_SUBMIT_PLANS)
        )

        for row_idx in range(table.rowCount()):
            plan = self._recent_submit_plans[row_idx]
            table.setVerticalHeaderItem(
                row_idx,
                QtWidgets.QTableWidgetItem(f"#{row_idx + 1}"),
            )
            for col_idx, dataset in enumerate(datasets):
                fields_for_dataset = plan.get(dataset, [])
                if fields_for_dataset:
                    table.setCellWidget(
                        row_idx,
                        col_idx,
                        self._build_recent_plan_chip_cell(fields_for_dataset),
                    )
                else:
                    item = QtWidgets.QTableWidgetItem("")
                    item.setTextAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
                    table.setItem(row_idx, col_idx, item)

            table.resizeRowToContents(row_idx)
        table.blockSignals(False)

    def _build_recent_plan_chip_cell(self, field_names):
        """Build a read-only chip row widget for a recent-plan table cell."""
        cell_widget = QtWidgets.QWidget()
        layout = QtWidgets.QHBoxLayout(cell_widget)
        layout.setContentsMargins(6, 4, 6, 4)
        layout.setSpacing(4)

        for field_name in field_names:
            chip = QtWidgets.QLabel(str(field_name))
            chip.setStyleSheet(
                "QLabel {"
                " border: 1px solid #2f7d32;"
                " border-radius: 9px;"
                " padding: 1px 7px;"
                " background: #43a047;"
                " color: white;"
                " font-weight: 500;"
                " }"
            )
            layout.addWidget(chip)

        layout.addStretch(1)

        # Allow the table to receive clicks for row selection/apply action.
        cell_widget.setAttribute(QtCore.Qt.WA_TransparentForMouseEvents, True)
        return cell_widget

    def _on_recent_plan_clicked(self, row, _col):
        """Apply a selected recent plan to current submit field rows."""
        if row < 0 or row >= len(self._recent_submit_plans):
            return

        plan = self._recent_submit_plans[row]
        for submit_row in self._submit_dataset_rows.values():
            grouped_fields = []
            for dataset in submit_row["datasets"]:
                for field_name in plan.get(dataset, []):
                    if field_name not in grouped_fields:
                        grouped_fields.append(field_name)
            submit_row["fields"].set_chips(grouped_fields)

        self._set_submit_status(f"Applied recent plan #{row + 1}.")
        self._update_submit_state()

    def _collect_config_values(self, dataset):
        """Collect raw configuration text values for a dataset."""
        values = {}
        for key, widget in self._dataset_config_inputs.get(dataset, {}).items():
            values[key] = self._config_widget_value(widget)
        return values

    def _config_widget_value(self, widget):
        """Return a typed value from a config editor widget."""
        if isinstance(widget, QtWidgets.QCheckBox):
            return bool(widget.isChecked())
        return widget.text().strip()

    def _save_dataset_configs(self):
        """Persist current backend selections and config values by dataset."""
        snapshot = dict(self._saved_dataset_configs)
        for dataset in self._dataset_backends:
            backend = self._dataset_backends.get(dataset, "")
            if not backend:
                continue

            snapshot[dataset] = {
                "backend": backend,
                "config": self._collect_config_values(dataset),
            }

        self._saved_dataset_configs = snapshot
        self._settings.setValue(
            self.SETTINGS_KEY_DATASET_CONFIGS,
            json.dumps(snapshot),
        )

    def _backend_fields(self, backend):
        """Return UI-visible config field keys for a backend type."""
        backend_cls = self._backend_class(backend)
        if backend_cls is None:
            return ()
        return backend_cls.config_visible_fields()

    def _backend_config_fields(self, backend):
        """Return all config field keys (visible and hidden) for a backend type."""
        backend_cls = self._backend_class(backend)
        if backend_cls is None:
            return ()
        return backend_cls.config_fields()

    def _dataset_is_writeable(self, dataset):
        """Return whether the dataset currently maps to a writeable backend."""
        backend = self._dataset_backends.get(dataset, "")
        return backend in self.WRITEABLE_BACKENDS

    def _writable_datasets(self):
        """Return sorted datasets that are configured as writeable."""
        return [
            ds
            for ds in sorted(self._dataset_backends)
            if self._dataset_is_writeable(ds)
        ]

    def _dataset_group_signature(self, dataset):
        """Build grouping signature from backend and config values."""
        backend = self._dataset_backends.get(dataset, "")
        field_keys = self._backend_config_fields(backend)
        values = self._collect_config_values(dataset)

        # Hidden fields are not rendered in the UI, so derive them from defaults.
        for key in field_keys:
            if key in values:
                continue
            default_value = self._field_default(backend, key)
            if default_value is not None:
                values[key] = str(default_value)

        signature_values = tuple(
            (
                k,
                values.get(k, "").strip()
                if self._field_type(backend, k) != "bool"
                else self._coerce_bool(values.get(k, False)),
            )
            for k in field_keys
        )
        return backend, signature_values

    def _submit_dataset_groups(self):
        """Return dataset groups based on grouping toggle and signatures."""
        datasets = self._writable_datasets()
        if not datasets:
            return []

        if not self.group_same_backend_checkbox.isChecked():
            return [{"datasets": [dataset]} for dataset in datasets]

        grouped = {}
        ordered_keys = []
        for dataset in datasets:
            signature = self._dataset_group_signature(dataset)
            if signature not in grouped:
                # Preserve first-seen order to keep row ordering stable.
                grouped[signature] = []
                ordered_keys.append(signature)
            grouped[signature].append(dataset)

        return [{"datasets": grouped[key]} for key in ordered_keys]

    def _field_meta(self, backend, key):
        """Look up dataclass metadata for a backend config field."""
        backend_cls = self._backend_class(backend)
        if backend_cls is None:
            return {}
        return backend_cls.config_field_meta(key)

    def _field_label(self, backend, key):
        """Return user-facing label text for a config field."""
        meta = self._field_meta(backend, key)
        return meta.get("label", key.replace("_", " ").title())

    def _field_required(self, backend, key):
        """Return whether a config field is required."""
        meta = self._field_meta(backend, key)
        return bool(meta.get("required", True))

    def _field_type(self, backend, key):
        """Return logical field type from metadata (defaults to string)."""
        meta = self._field_meta(backend, key)
        value_type = meta.get("type", "string")
        if value_type in {bool, "bool", "boolean"}:
            return "bool"
        return "string"

    def _coerce_bool(self, value):
        """Coerce UI/default/saved values to a boolean."""
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in {"1", "true", "t", "yes", "y", "on"}:
                return True
            if lowered in {"0", "false", "f", "no", "n", "off", ""}:
                return False
        return bool(value)

    def _field_default(self, backend, key):
        """Return default value declared by the backend config, if any."""
        backend_cls = self._backend_class(backend)
        if backend_cls is None:
            return None
        return backend_cls.config_field_default(key)

    def _available_fields_for_dataset(self, dataset):
        """Return suggested annotation fields for a dataset backend."""
        if not self._dataset_is_writeable(dataset):
            return ()
        backend = self._dataset_backends.get(dataset, "")
        backend_cls = self._backend_class(backend)
        if backend_cls is None:
            return ()
        return tuple(backend_cls.FIELD_SUGGESTIONS)

    def _parse_field_list(self, text):
        """Parse comma-separated field text into ordered unique values."""
        parts = [p.strip() for p in text.split(",")]
        parts = [p for p in parts if p]
        # Keep order but remove duplicates.
        return list(dict.fromkeys(parts))

    def _populate_submit_dataset_rows(self):
        """Rebuild dataset field selector rows for current grouping state."""
        # Capture current per-dataset selections so edits survive regrouping.
        existing_fields_by_dataset = self._collect_submit_dataset_field_map()

        # Clear previously rendered dynamic rows but keep the trailing stretch.
        while self.submit_dataset_layout.count() > 1:
            item = self.submit_dataset_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

        self._submit_dataset_rows = {}
        for group in self._submit_dataset_groups():
            datasets = group["datasets"]
            primary_dataset = datasets[0]
            row_widget = QtWidgets.QWidget()
            row_layout = QtWidgets.QVBoxLayout(row_widget)
            row_layout.setContentsMargins(0, 0, 0, 0)
            row_layout.setSpacing(4)

            top_line = QtWidgets.QHBoxLayout()
            top_line.setContentsMargins(0, 0, 0, 0)
            top_line.setSpacing(8)

            if len(datasets) == 1:
                label_text = primary_dataset
            else:
                label_text = ", ".join(datasets)
            dataset_label = QtWidgets.QLabel(label_text)
            dataset_label.setMinimumWidth(140)

            fields_edit = QtWidgets.QLineEdit()
            fields_edit = FieldChipInput()
            fields_edit.changed.connect(self._update_submit_state)

            hint_label = QtWidgets.QLabel("")
            hint_label.setStyleSheet("color: #666;")

            top_line.addWidget(dataset_label)
            top_line.addWidget(fields_edit, stretch=1)
            top_line.addWidget(hint_label)
            row_layout.addLayout(top_line)

            self.submit_dataset_layout.insertWidget(
                self.submit_dataset_layout.count() - 1,
                row_widget,
            )

            row_key = "|".join(datasets)
            self._submit_dataset_rows[row_key] = {
                "widget": row_widget,
                "fields": fields_edit,
                "hint": hint_label,
                "datasets": datasets,
                "primary_dataset": primary_dataset,
            }

            # Merge prior field chips across all datasets represented by this row.
            initial_fields = []
            for dataset in datasets:
                for field_name in existing_fields_by_dataset.get(dataset, []):
                    if field_name not in initial_fields:
                        initial_fields.append(field_name)
            if initial_fields:
                fields_edit.set_chips(initial_fields)

            self._update_submit_row_hint_for_dataset(row_key)

        self._refresh_recent_plans_table()

    def _update_submit_row_hint_for_dataset(self, dataset):
        """Update placeholder, tooltip, and backend hint for one submit row."""
        row = self._submit_dataset_rows.get(dataset)
        if row is None:
            return

        primary_dataset = row["primary_dataset"]

        available = self._available_fields_for_dataset(primary_dataset)
        if available:
            row["fields"].set_placeholder_text("e.g. " + ", ".join(available[:3]))
            row["fields"].set_tooltip(
                "Press comma or Enter to create a field chip. Click chip to remove."
            )
            row["hint"].setText(self._dataset_backends.get(primary_dataset, ""))
        else:
            row["fields"].set_placeholder_text("Fields (comma-separated)")
            row["fields"].set_tooltip(
                "Press comma or Enter to create a field chip. Click chip to remove."
            )
            row["hint"].setText("")

    def _collect_submit_dataset_field_map(self):
        """Expand row chip selections into a per-dataset field mapping."""
        mapping = {}
        for row in self._submit_dataset_rows.values():
            fields = row["fields"].chips()
            if fields:
                # A grouped row fan-outs to every dataset in that group.
                for dataset in row["datasets"]:
                    mapping[dataset] = list(fields)
        return mapping

    def _set_submit_status(self, text, is_error=False):
        """Set submit status text and color-coded styling."""
        self.submit_status_label.setText(text)
        if not text:
            self.submit_status_label.setStyleSheet("")
        elif is_error:
            self.submit_status_label.setStyleSheet("color: #e53935;")
        else:
            self.submit_status_label.setStyleSheet("color: #43a047;")

    def _clear_mode_enabled(self):
        """Return whether submit should clear fields instead of writing a value."""
        checkbox = getattr(self, "clear_fields_checkbox", None)
        return bool(checkbox and checkbox.isChecked())

    def _on_clear_mode_toggled(self, checked):
        """Toggle value editor availability when clear mode is enabled."""
        self.shared_value_edit.setEnabled(not checked)
        self._update_submit_state()

    def _submit_plan_errors(self):
        """Return submit-time validation errors unrelated to backend field checks."""
        errors = []
        if not self._writable_datasets():
            errors.append(
                "No writeable datasets selected. Choose at least one writeable backend."
            )
            return errors

        if not self._clear_mode_enabled():
            value = self.shared_value_edit.text().strip()
            if not value:
                errors.append("Provide a value to write.")

        field_map = self._collect_submit_dataset_field_map()
        if not field_map:
            errors.append("Select at least one target field for at least one dataset.")

        return errors

    def _field_chip_validation_errors(self):
        """Validate selected field chips and values against backend rules."""
        self._last_invalid_fields_by_dataset = {}

        if not self.__class__._session_config_validated:
            return []

        field_map = self._collect_submit_dataset_field_map()
        if not field_map:
            return []

        # Session-level validation can outlive this dialog instance, so lazily
        # rebuild runtime backend instances when the local cache is missing.
        missing_cached_backend = any(
            dataset not in self._validated_backends for dataset in field_map
        )
        if missing_cached_backend:
            ok, error_message = self._validate_backend_configs()
            if not ok:
                return [error_message]

        shared_value = (
            None if self._clear_mode_enabled() else self.shared_value_edit.text().strip()
        )

        errors = []
        for dataset, field_names in field_map.items():
            backend = self._validated_backends.get(dataset)
            if backend is None:
                errors.append(
                    "Configuration changed. Press Validate in Configuration again."
                )
                return errors

            invalid_fields = []
            for field_name in field_names:
                try:
                    # Delegate field support checks to the backend implementation.
                    is_valid = backend.validate_field(field_name)
                except Exception as exc:
                    detail = str(exc).strip() or exc.__class__.__name__
                    errors.append(f"Field validation failed for {dataset}: {detail}")
                    break

                if not is_valid:
                    invalid_fields.append(field_name)

                try:
                    value_valid = backend.validate_field_value(field_name, shared_value)
                except Exception as exc:
                    detail = str(exc).strip() or exc.__class__.__name__
                    errors.append(
                        f"Value validation failed for {dataset}/{field_name}: {detail}"
                    )
                    break

                if not value_valid:
                    errors.append(
                        f"Invalid value for {dataset}/{field_name}: {shared_value!r}"
                    )
                    break

            if invalid_fields:
                backend_name = self._dataset_backends.get(dataset, "backend")
                self._last_invalid_fields_by_dataset[dataset] = set(invalid_fields)
                errors.append(
                    f"Invalid field(s) for {dataset} ({backend_name}): {', '.join(invalid_fields)}"
                )

        return errors

    def _dataset_has_complete_config(self, dataset):
        """Check whether required configuration values are present for a dataset."""
        backend = self._dataset_backends.get(dataset, "")
        if not backend:
            return False
        if backend == "Not Writeable":
            return True

        inputs = self._dataset_config_inputs.get(dataset, {})
        for key in self._backend_fields(backend):
            widget = inputs.get(key)
            if widget is None:
                return False

            value = self._config_widget_value(widget)
            if (
                self._field_required(backend, key)
                and self._field_type(backend, key) != "bool"
                and not str(value).strip()
            ):
                return False
        return True

    def _set_config_cell(self, dataset, backend, initial_values=None):
        """Render backend-specific configuration editors for one dataset row."""
        initial_values = initial_values or {}
        row = self._dataset_row_index[dataset]
        container = QtWidgets.QWidget()
        container_layout = QtWidgets.QVBoxLayout(container)
        container_layout.setContentsMargins(0, 4, 0, 4)
        container_layout.setSpacing(0)

        form_widget = QtWidgets.QWidget()
        form_widget.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding,
            QtWidgets.QSizePolicy.Preferred,
        )
        layout = QtWidgets.QFormLayout(form_widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        layout.setLabelAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
        layout.setFormAlignment(QtCore.Qt.AlignTop)
        layout.setFieldGrowthPolicy(QtWidgets.QFormLayout.AllNonFixedFieldsGrow)

        self._dataset_config_inputs[dataset] = {}
        fields = self._backend_fields(backend)
        if not fields:
            note_text = "Select a backend to configure required fields"
            if backend == self.NOT_WRITEABLE_BACKEND:
                note_text = "Dataset is marked as not writeable."
            note = QtWidgets.QLabel(note_text)
            note.setStyleSheet("color: #666;")
            layout.addRow(note)
        else:
            for key in fields:
                label = self._field_label(backend, key)
                meta = self._field_meta(backend, key)
                if self._field_type(backend, key) == "bool":
                    editor = QtWidgets.QCheckBox()
                    if meta.get("tooltip"):
                        editor.setToolTip(meta["tooltip"])
                    if key in initial_values:
                        editor.setChecked(self._coerce_bool(initial_values[key]))
                    else:
                        default_value = self._field_default(backend, key)
                        if default_value is not None:
                            editor.setChecked(self._coerce_bool(default_value))
                    editor.toggled.connect(self._on_config_inputs_changed)
                    editor.toggled.connect(self._save_dataset_configs)
                else:
                    editor = QtWidgets.QLineEdit()
                    editor.setSizePolicy(
                        QtWidgets.QSizePolicy.Expanding,
                        QtWidgets.QSizePolicy.Fixed,
                    )
                    editor.setPlaceholderText(meta.get("placeholder", label))
                    if meta.get("tooltip"):
                        editor.setToolTip(meta["tooltip"])
                    if key in initial_values:
                        editor.setText(str(initial_values[key]))
                    else:
                        default_value = self._field_default(backend, key)
                        if default_value is not None:
                            editor.setText(str(default_value))
                    editor.textChanged.connect(self._on_config_inputs_changed)
                    editor.textChanged.connect(self._save_dataset_configs)
                required_suffix = "" if not self._field_required(backend, key) else " *"
                layout.addRow(f"{label}{required_suffix}:", editor)
                self._dataset_config_inputs[dataset][key] = editor

        container_layout.addWidget(form_widget)
        container_layout.addStretch(1)

        self.dataset_table.setCellWidget(row, 3, container)
        self.dataset_table.resizeRowToContents(row)

    def _build_dataset_config(self, dataset):
        """Build validated backend config dataclass for a dataset."""
        backend = self._dataset_backends.get(dataset, "")
        backend_cls = self._backend_class(backend)
        if backend_cls is None or backend_cls.CONFIG_CLASS is None:
            return None

        values = {}
        for key in self._backend_config_fields(backend):
            widget = self._dataset_config_inputs.get(dataset, {}).get(key)
            if widget is not None:
                values[key] = self._config_widget_value(widget)
                continue

            default_value = self._field_default(backend, key)
            field_type = self._field_type(backend, key)
            if default_value is None:
                values[key] = False if field_type == "bool" else ""
            elif field_type == "bool":
                values[key] = self._coerce_bool(default_value)
            else:
                values[key] = str(default_value).strip()

        for key, value in values.items():
            if (
                self._field_required(backend, key)
                and self._field_type(backend, key) != "bool"
                and not str(value).strip()
            ):
                return None

        return backend_cls.CONFIG_CLASS(**values)

    def _populate_dataset_table(self):
        """Populate dataset configuration table from current selection."""
        counts = {}
        for rec in self.selection:
            counts[rec.dataset] = counts.get(rec.dataset, 0) + 1

        datasets = self._project_datasets or sorted(counts)
        self.dataset_table.setRowCount(len(datasets))
        self._dataset_backends = {}
        self._dataset_row_index = {}
        self._dataset_config_inputs = {}

        for row, dataset in enumerate(datasets):
            self._dataset_row_index[dataset] = row

            dataset_item = QtWidgets.QTableWidgetItem(dataset)
            dataset_item.setTextAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignTop)
            count_item = QtWidgets.QTableWidgetItem(str(counts.get(dataset, 0)))
            count_item.setTextAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignTop)
            self.dataset_table.setItem(row, 0, dataset_item)
            self.dataset_table.setItem(row, 1, count_item)

            combo = QtWidgets.QComboBox()
            combo.addItems(self.BACKEND_CHOICES)
            combo.currentTextChanged.connect(
                lambda text, ds=dataset: self._on_backend_changed(ds, text)
            )

            combo_container = QtWidgets.QWidget()
            combo_layout = QtWidgets.QVBoxLayout(combo_container)
            combo_layout.setContentsMargins(0, 4, 0, 4)
            combo_layout.setSpacing(0)
            combo_layout.addWidget(combo, alignment=QtCore.Qt.AlignTop)
            combo_layout.addStretch(1)

            self.dataset_table.setCellWidget(row, 2, combo_container)

            saved = self._saved_dataset_configs.get(dataset, {})
            saved_backend = saved.get("backend", "")
            saved_config = saved.get("config", {}) if isinstance(saved, dict) else {}

            if saved_backend in self.BACKEND_CHOICES:
                combo.blockSignals(True)
                combo.setCurrentText(saved_backend)
                combo.blockSignals(False)
                self._dataset_backends[dataset] = saved_backend
                self._set_config_cell(dataset, saved_backend, saved_config)
            else:
                self._dataset_backends[dataset] = ""
                self._set_config_cell(dataset, "")

        self._tune_dataset_table_columns()
        self._populate_submit_dataset_rows()

    def _tune_dataset_table_columns(self):
        """Keep fixed columns compact and reserve space for configuration widgets."""
        self.dataset_table.resizeColumnsToContents()

        max_widths = {
            0: 220,  # Dataset
            1: 170,  # Selected neurons
            2: 170,  # Backend
        }
        padding = 12
        for col, max_width in max_widths.items():
            width = min(self.dataset_table.columnWidth(col) + padding, max_width)
            self.dataset_table.setColumnWidth(col, width)

    def _update_summary(self):
        """Update dialog title with current selection and dataset breakdown."""
        total = len(self.selection)
        counts = {}
        for rec in self.selection:
            counts[rec.dataset] = counts.get(rec.dataset, 0) + 1

        if counts:
            dataset_parts = [
                f"{counts[dataset]} {dataset}" for dataset in sorted(counts)
            ]
            dataset_text = ", ".join(dataset_parts)
        else:
            dataset_text = "(none)"

        self.setWindowTitle(f"Push annotation for {total} neurons: {dataset_text}")

    def _on_backend_changed(self, dataset, backend):
        """Handle backend selection changes for a dataset row."""
        self._invalidate_backend_validation()
        if backend == self.UNSELECTED_BACKEND:
            self._dataset_backends[dataset] = ""
        else:
            self._dataset_backends[dataset] = backend
        self._set_config_cell(dataset, self._dataset_backends[dataset])
        self._save_dataset_configs()
        self._populate_submit_dataset_rows()
        self._update_submit_state()

    def _validation_errors(self):
        """Aggregate all current validation errors for the dialog state."""
        errors = []

        if not self.selection:
            errors.append("No selected neurons.")

        if any(not backend for backend in self._dataset_backends.values()):
            errors.append("Choose a backend for every dataset.")
        elif any(
            not self._dataset_has_complete_config(dataset)
            for dataset in self._dataset_backends
        ):
            errors.append("Fill required backend configuration for every dataset.")

        submit_errors = self._submit_plan_errors()
        errors.extend(submit_errors)

        field_validation_errors = self._field_chip_validation_errors()
        errors.extend(field_validation_errors)

        return errors

    def _update_submit_state(self):
        """Refresh control enablement, status messaging, and chip highlights."""
        all_configured = self._all_datasets_configured()
        submit_accessible = all_configured and self.__class__._session_config_validated
        self.tabs.setTabEnabled(self._submit_tab_index, submit_accessible)
        self.validate_button.setEnabled(all_configured)
        if not submit_accessible and self.tabs.currentIndex() == self._submit_tab_index:
            self.tabs.setCurrentIndex(self._config_tab_index)

        # Recompute full validation after any state change.
        errors = self._validation_errors()

        for row in self._submit_dataset_rows.values():
            invalid_fields = set()
            for dataset in row["datasets"]:
                invalid_fields.update(
                    self._last_invalid_fields_by_dataset.get(dataset, set())
                )
            # Highlight chips that failed backend field validation.
            row["fields"].set_invalid_chips(invalid_fields)

        can_submit = (not errors) and self.__class__._session_config_validated
        self.submit_button.setEnabled(can_submit)

        if not all_configured:
            self.config_status_label.setText(errors[0] if errors else "")
            self._set_submit_status("")
            return

        if not self.__class__._session_config_validated:
            self.config_status_label.setText(
                "Press Validate in Configuration to continue."
            )
            self._set_submit_status("Press Validate in Configuration to continue.")
            return

        self.config_status_label.setText("")
        if can_submit:
            ready_text = (
                "Ready to clear annotations"
                if self._clear_mode_enabled()
                else "Ready to push annotations"
            )
            self._set_submit_status(ready_text)
        else:
            self._set_submit_status(errors[0] if errors else "", is_error=bool(errors))

    def _build_payload(self):
        """Build the prototype submission payload from current UI state."""
        by_dataset = {}
        for rec in self.selection:
            by_dataset.setdefault(rec.dataset, []).append(rec.neuron_id)

        shared_value = (
            None
            if self._clear_mode_enabled()
            else self.shared_value_edit.text().strip()
        )
        dataset_field_map = self._collect_submit_dataset_field_map()

        write_rules = [
            {
                "dataset": dataset,
                "value": shared_value,
                "fields": fields,
            }
            for dataset, fields in dataset_field_map.items()
        ]

        # Flatten rules to one write operation per (dataset, field).
        writes = []
        for rule in write_rules:
            for field_name in rule["fields"]:
                writes.append(
                    {
                        "dataset": rule["dataset"],
                        "field": field_name,
                        "value": rule["value"],
                    }
                )

        return {
            "selection_count": len(self.selection),
            "datasets": {
                dataset: {
                    "backend": self._dataset_backends.get(dataset, ""),
                    "config": (
                        asdict(cfg)
                        if (cfg := self._build_dataset_config(dataset))
                        else {}
                    ),
                    "ids": ids,
                }
                for dataset, ids in by_dataset.items()
            },
            "write_rules": write_rules,
            "writes": writes,
        }

    def _submit_threadpool(self):
        """Return the parent widget thread pool when available."""
        parent = self.parent()
        if parent is not None:
            parent_pool = getattr(parent, "threadpool", None)
            if isinstance(parent_pool, QtCore.QThreadPool):
                return parent_pool
        return self._fallback_threadpool

    def _dispatch_backend_writes(self, callback):
        """Dispatch backend writes to a background thread with callback reporting."""
        runnable = FunctionRunnable(self._run_backend_writes, callback=callback)
        self._submit_threadpool().start(runnable)

    def _submit_result_callback(self):
        """Resolve callback used for background submit result reporting."""
        parent = self.parent()
        if parent is not None:
            parent_callback = getattr(parent, "on_annotation_submit_result", None)
            if callable(parent_callback):
                return parent_callback
        return print

    def _run_backend_writes(self, callback):
        """Write annotations across backends and callback on success or first error."""
        by_dataset = {}
        for rec in self.selection:
            by_dataset.setdefault(rec.dataset, []).append(rec.neuron_id)

        shared_value = (
            None
            if self._clear_mode_enabled()
            else self.shared_value_edit.text().strip()
        )
        dataset_field_map = self._collect_submit_dataset_field_map()

        used_backends = set()
        pushed_neuron_count = 0
        backend_batches = {}

        # Group dataset writes by backend instance and field set so shared
        # backend configs are written in fewer calls without widening fields.
        for dataset, field_names in dataset_field_map.items():
            ids = by_dataset.get(dataset, [])
            if not ids or not field_names:
                continue

            backend = self._validated_backends.get(dataset)
            backend_name = self._dataset_backends.get(dataset, "unknown")
            if backend is None:
                callback(
                    f"Error encountered pushing annotations to backend {backend_name}: backend is not validated"
                )
                return

            field_key = tuple(dict.fromkeys(field_names))
            batch_key = (id(backend), field_key)
            batch = backend_batches.get(batch_key)
            if batch is None:
                batch = {
                    "backend": backend,
                    "backend_name": backend_name,
                    "fields": list(field_key),
                    "ids": set(),
                }
                backend_batches[batch_key] = batch
            batch["ids"].update(ids)

        for batch in backend_batches.values():
            status, error = batch["backend"].write_annotations(
                ids=sorted(batch["ids"]),
                value=shared_value,
                fields=batch["fields"],
            )
            if error or "ERROR" in status:
                detail = error or "Unknown backend error"
                callback(
                    f"Error encountered pushing annotations to backend {batch['backend_name']}: {detail}"
                )
                return

            # Count unique backend instances used for this submit operation.
            used_backends.add(id(batch["backend"]))
            # Count neurons only for datasets that were actually written.
            pushed_neuron_count += len(batch["ids"])

        if not self._clear_mode_enabled():
            callback(
                f"Successfully pushed annotations for {pushed_neuron_count} neurons to {len(used_backends)} backends"
            )
        else:
            callback(
                f"Successfully cleared annotations for {pushed_neuron_count} neurons across {len(used_backends)} backends"
            )

    def _on_submit(self):
        """Validate and dispatch backend writes on a background thread pool."""
        if not self.__class__._session_config_validated:
            self._set_submit_status("Press Validate in Configuration to continue.")
            self.config_status_label.setText(
                "Press Validate in Configuration to continue."
            )
            self.tabs.setCurrentIndex(self._config_tab_index)
            return

        errors = self._validation_errors()
        if errors:
            self._set_submit_status(errors[0], is_error=True)
            return

        self._remember_recent_submit_plan()
        self._refresh_recent_plans_table()

        payload = self._build_payload()
        self._dispatch_backend_writes(callback=self._submit_result_callback())
        self.submitted.emit(payload)
        self.accept()


class AnnotationPrototypeWindow(QtWidgets.QWidget):
    """Simple host window for iterating on the annotation dialog."""

    submit_result_received = QtCore.Signal(str)

    def __init__(self):
        """Initialize prototype host window and mock selection state."""
        super().__init__()
        self.setWindowTitle("Annotation Dialog Prototype")
        self.resize(720, 380)
        self.threadpool = QtCore.QThreadPool(self)
        self.submit_result_received.connect(self._append_submit_result)

        self._selection = self._mock_selection()
        self._build_ui()
        self._update_selection_label()

    def _build_ui(self):
        """Build controls for launching the dialog and showing results."""
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(10)

        intro = QtWidgets.QLabel(
            "Prototype runner for the annotation dialog. "
            "Click the button or press Cmd+A to open it."
        )
        intro.setWordWrap(True)
        layout.addWidget(intro)

        self.selection_label = QtWidgets.QLabel()
        self.selection_label.setWordWrap(True)
        layout.addWidget(self.selection_label)

        open_button = QtWidgets.QPushButton("Open Annotation Dialog")
        open_button.clicked.connect(self.open_dialog)
        layout.addWidget(open_button)

        self.result_box = QtWidgets.QPlainTextEdit()
        self.result_box.setReadOnly(True)
        self.result_box.setPlaceholderText("Submission payload will be shown here.")
        layout.addWidget(self.result_box, stretch=1)

        shortcut = QtGui.QShortcut(
            QtGui.QKeySequence(QtGui.QKeySequence.SelectAll),
            self,
        )
        shortcut.activated.connect(self.open_dialog)

    def _mock_selection(self):
        """Return static mock neuron selection records for the prototype."""
        return [
            SelectionRecord(720575940628299419, "FwR"),
            SelectionRecord(720575940640052621, "FwL"),
            SelectionRecord(720575940631290082, "FwR"),
            SelectionRecord(529529442, "McnsR"),
            SelectionRecord(585510588, "McnsL"),
        ]

    def _update_selection_label(self):
        """Refresh summary label describing the current mock selection."""
        ids_preview = ", ".join(str(rec.neuron_id) for rec in self._selection[:6])
        dataset_count = len({rec.dataset for rec in self._selection})
        self.selection_label.setText(
            "Mock selection contains "
            f"{len(self._selection)} neurons across {dataset_count} dataset(s): "
            f"{ids_preview}"
        )

    def open_dialog(self):
        """Open the annotation dialog and hook up submit callback."""
        dialog = AnnotationDialog(selection=self._selection, parent=self)
        dialog.submitted.connect(self._handle_submitted)
        dialog.exec()

    def _handle_submitted(self, payload):
        """Display the emitted payload in the prototype result box."""
        self.result_box.setPlainText(str(payload))

    def on_annotation_submit_result(self, message):
        """Receive async submit result messages from the dialog worker."""
        self.submit_result_received.emit(str(message))

    def _append_submit_result(self, message):
        """Append async submit status lines to the prototype result box."""
        if self.result_box.toPlainText().strip():
            self.result_box.appendPlainText("")
        self.result_box.appendPlainText(str(message))


if __name__ in {"__main__", "main"}:
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    window = AnnotationPrototypeWindow()
    window.show()
    app.exec()
