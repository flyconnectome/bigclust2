"""UI for configuring per-dataset meta-data sources and pulling fresh data.

``MetaSourcesDialog`` mirrors the configuration tab of ``AnnotationDialog``
(dataset -> backend -> dynamic config form) but, instead of pushing
annotations, it *reads* from each backend and merges the result into the
project's meta table. It never writes to any backend.

The heavy lifting (parsing/merging) lives in the Qt-free ``bigclust2.meta_sources``
module; this file is only the Qt layer.
"""

from __future__ import annotations

import logging

import pandas as pd

from PySide6 import QtCore, QtWidgets

try:
    from .annotation_backends import BACKEND_REGISTRY, build_backend
    from .credentials import CredentialsManager
except ImportError:  # pragma: no cover - script execution fallback
    from annotation_backends import BACKEND_REGISTRY, build_backend
    from credentials import CredentialsManager

from ...credentials import service_key_for_backend

from ...meta_sources import (
    MetaSourceSpec,
    auto_match_columns,
    parse_meta_sources,
    update_meta,
    write_meta_sources,
)


logger = logging.getLogger(__name__)


def _is_reserved(column):
    """Project columns that are never valid meta-source targets."""
    text = str(column)
    return text in ("id", "dataset") or text.startswith("_")


class _FunctionRunnable(QtCore.QRunnable):
    """Run a callable in a thread pool (local copy to avoid cross-imports)."""

    def __init__(self, fn, *args, **kwargs):
        super().__init__()
        self._fn = fn
        self._args = args
        self._kwargs = kwargs

    def run(self):
        self._fn(*self._args, **self._kwargs)


class ColumnMappingDialog(QtWidgets.QDialog):
    """Edit the project-column -> source-column mapping for one dataset."""

    DO_NOT_UPDATE = "(do not update)"

    def __init__(self, dataset, project_columns, source_columns, mapping, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"Column mapping — {dataset}")
        self.resize(460, 560)

        self._combos = {}
        source_columns = sorted({str(c) for c in (source_columns or [])})

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)

        if source_columns:
            hint = (
                "Pick the source column that maps onto each project column. "
                "Leave as '(do not update)' to skip a column."
            )
        else:
            hint = (
                "No source columns fetched yet — type the source column name, "
                "or close and use 'Fetch columns' first."
            )
        hint_label = QtWidgets.QLabel(hint)
        hint_label.setWordWrap(True)
        layout.addWidget(hint_label)

        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        form_host = QtWidgets.QWidget()
        form = QtWidgets.QFormLayout(form_host)
        form.setLabelAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
        form.setFieldGrowthPolicy(QtWidgets.QFormLayout.AllNonFixedFieldsGrow)

        for col in project_columns:
            if _is_reserved(col):
                continue
            combo = QtWidgets.QComboBox()
            combo.setEditable(True)
            combo.addItem(self.DO_NOT_UPDATE)
            combo.addItems(source_columns)
            current = mapping.get(str(col))
            if current:
                idx = combo.findText(current)
                if idx >= 0:
                    combo.setCurrentIndex(idx)
                else:
                    combo.setEditText(current)
            else:
                combo.setCurrentIndex(0)
            self._combos[str(col)] = combo
            form.addRow(f"{col}:", combo)

        scroll.setWidget(form_host)
        layout.addWidget(scroll, 1)

        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def mapping(self):
        """Return the edited ``{project_col: source_col}`` mapping."""
        out = {}
        for col, combo in self._combos.items():
            text = combo.currentText().strip()
            if text and text != self.DO_NOT_UPDATE:
                out[col] = text
        return out


class MetaSourcesDialog(QtWidgets.QDialog):
    """Configure meta-data sources per dataset and pull fresh annotations."""

    # (updated_meta DataFrame, UpdateReport)
    metaUpdated = QtCore.Signal(object, object)
    sourcesSaved = QtCore.Signal()

    # Internal cross-thread signals (worker -> UI thread).
    _fetchFinished = QtCore.Signal(str, object, object)  # dataset, columns|None, error|None
    _updateFinished = QtCore.Signal(object, object, object)  # meta|None, report|None, error|None

    UNSELECTED_BACKEND = "-- Select backend --"
    BACKEND_CHOICES = (UNSELECTED_BACKEND, *BACKEND_REGISTRY.keys())
    SAMPLE_SIZE = 50

    def __init__(self, meta, info, parent=None):
        super().__init__(parent)
        if not isinstance(meta, pd.DataFrame):
            raise TypeError("meta must be a pandas DataFrame")

        self._meta = meta
        self._info = info if isinstance(info, dict) else {}

        self._project_columns = [c for c in meta.columns if not _is_reserved(c)]
        self._dataset_counts = (
            meta["dataset"].astype(str).value_counts().to_dict()
            if "dataset" in meta.columns
            else {}
        )
        self._datasets = sorted(self._dataset_counts)

        # Per-dataset UI/state.
        self._row_index = {}
        self._dataset_backends = {}
        self._dataset_config_inputs = {}
        self._dataset_columns = {}
        self._dataset_source_columns = {}
        self._dataset_last_updated = {}
        self._status_labels = {}
        self._map_buttons = {}

        self._fallback_threadpool = QtCore.QThreadPool(self)
        self._busy = False
        self._pending_fetch = set()
        self._progress = None
        self._dirty = False
        # Suppresses dirty-tracking while widgets are populated from presets.
        self._loading = False

        self.setWindowTitle("Meta Data Sources")
        self.setModal(False)
        self.resize(940, 600)

        self._seed_from_info()
        self._build_ui()
        self._loading = True
        self._populate_table()
        self._loading = False

        self._fetchFinished.connect(self._on_fetch_finished)
        self._updateFinished.connect(self._on_update_finished)

    # ------------------------------------------------------------------ #
    # Seed / state
    # ------------------------------------------------------------------ #

    def _seed_from_info(self):
        """Pre-fill backend/config/columns from existing info presets."""
        self._seed_specs = {
            spec.dataset: spec for spec in parse_meta_sources(self._info)
        }

    def _backend_cls(self, backend):
        return BACKEND_REGISTRY.get(backend)

    def _visible_fields(self, backend):
        backend_cls = self._backend_cls(backend)
        return backend_cls.config_visible_fields() if backend_cls else ()

    def _field_meta(self, backend, key):
        backend_cls = self._backend_cls(backend)
        return backend_cls.config_field_meta(key) if backend_cls else {}

    def _field_is_bool(self, backend, key):
        return self._field_meta(backend, key).get("type", "string") in (
            bool,
            "bool",
            "boolean",
        )

    def _field_required(self, backend, key):
        return bool(self._field_meta(backend, key).get("required", True))

    def _field_default(self, backend, key):
        backend_cls = self._backend_cls(backend)
        return backend_cls.config_field_default(key) if backend_cls else None

    def _collect_config_values(self, dataset):
        """Read the rendered config widgets for a dataset into a flat dict."""
        values = {}
        for key, widget in self._dataset_config_inputs.get(dataset, {}).items():
            if isinstance(widget, QtWidgets.QCheckBox):
                values[key] = bool(widget.isChecked())
            else:
                values[key] = widget.text().strip()
        return values

    # ------------------------------------------------------------------ #
    # UI construction
    # ------------------------------------------------------------------ #

    def _build_ui(self):
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(10)

        help_label = QtWidgets.QLabel(
            "Define where each dataset's meta data comes from, then pull fresh "
            "values. Use 'Auto-map' to list each source's columns and match them "
            "to the project columns, then refine the mapping. Sources are read "
            "only — nothing is ever written back."
        )
        help_label.setWordWrap(True)
        layout.addWidget(help_label)

        self.table = QtWidgets.QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(
            ["Dataset", "Rows", "Backend", "Configuration", "Column mapping"]
        )
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionMode(QtWidgets.QAbstractItemView.NoSelection)
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeToContents)
        header.setSectionResizeMode(1, QtWidgets.QHeaderView.ResizeToContents)
        header.setSectionResizeMode(2, QtWidgets.QHeaderView.ResizeToContents)
        header.setSectionResizeMode(3, QtWidgets.QHeaderView.Stretch)
        header.setSectionResizeMode(4, QtWidgets.QHeaderView.ResizeToContents)
        layout.addWidget(self.table, 1)

        # Auto-map sits under the "Column mapping" column (the rightmost one).
        automap_row = QtWidgets.QHBoxLayout()
        automap_row.addStretch(1)
        self.automap_button = QtWidgets.QPushButton("Auto-map")
        self.automap_button.setToolTip(
            "Fetch each source's columns and auto-match them to the project "
            "columns (samples a few ids)."
        )
        self.automap_button.clicked.connect(self._on_auto_map)
        automap_row.addWidget(self.automap_button)
        layout.addLayout(automap_row)

        self.status_label = QtWidgets.QLabel("")
        self.status_label.setWordWrap(True)
        self.status_label.setStyleSheet("color: #7a7a7a;")
        layout.addWidget(self.status_label)

        buttons = QtWidgets.QHBoxLayout()
        self.save_button = QtWidgets.QPushButton("Save sources")
        self.save_button.setToolTip("Store these source definitions in the project info.")
        self.save_button.clicked.connect(self._on_save_sources)
        self.save_button.setEnabled(False)  # enabled once something changes
        self.update_button = QtWidgets.QPushButton("Update meta now")
        self.update_button.setToolTip("Pull fresh values from the configured sources.")
        self.update_button.clicked.connect(self._on_update)
        self.close_button = QtWidgets.QPushButton("Close")
        self.close_button.clicked.connect(self.close)

        buttons.addWidget(self.save_button)
        buttons.addStretch(1)
        buttons.addWidget(self.update_button)
        buttons.addWidget(self.close_button)
        layout.addLayout(buttons)

    def _populate_table(self):
        self.table.setRowCount(len(self._datasets))
        for row, dataset in enumerate(self._datasets):
            self._row_index[dataset] = row

            name_item = QtWidgets.QTableWidgetItem(dataset)
            name_item.setTextAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignTop)
            count_item = QtWidgets.QTableWidgetItem(
                str(self._dataset_counts.get(dataset, 0))
            )
            count_item.setTextAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignTop)
            self.table.setItem(row, 0, name_item)
            self.table.setItem(row, 1, count_item)

            combo = QtWidgets.QComboBox()
            combo.addItems(self.BACKEND_CHOICES)
            combo.currentTextChanged.connect(
                lambda text, ds=dataset: self._on_backend_changed(ds, text)
            )
            combo_container = QtWidgets.QWidget()
            combo_layout = QtWidgets.QVBoxLayout(combo_container)
            combo_layout.setContentsMargins(0, 4, 0, 4)
            combo_layout.addWidget(combo, alignment=QtCore.Qt.AlignTop)
            combo_layout.addStretch(1)
            self.table.setCellWidget(row, 2, combo_container)

            self._build_mapping_cell(dataset, row)

            seed = self._seed_specs.get(dataset)
            if seed is not None and seed.backend in self.BACKEND_CHOICES:
                self._dataset_columns[dataset] = dict(seed.columns)
                self._dataset_last_updated[dataset] = seed.last_updated
                combo.blockSignals(True)
                combo.setCurrentText(seed.backend)
                combo.blockSignals(False)
                self._dataset_backends[dataset] = seed.backend
                self._render_config_cell(dataset, seed.backend, seed.config)
            else:
                self._dataset_backends[dataset] = ""
                self._render_config_cell(dataset, "")
            self._refresh_mapping_label(dataset)

        self.table.resizeRowsToContents()

    def _build_mapping_cell(self, dataset, row):
        container = QtWidgets.QWidget()
        vbox = QtWidgets.QVBoxLayout(container)
        vbox.setContentsMargins(4, 4, 4, 4)
        vbox.setSpacing(2)

        button = QtWidgets.QPushButton("Edit mapping…")
        button.clicked.connect(lambda _=False, ds=dataset: self._on_edit_mapping(ds))
        status = QtWidgets.QLabel("")
        status.setStyleSheet("color: #7a7a7a; font-size: 11px;")

        vbox.addWidget(button)
        vbox.addWidget(status)
        vbox.addStretch(1)

        self._map_buttons[dataset] = button
        self._status_labels[dataset] = status
        self.table.setCellWidget(row, 4, container)

    def _render_config_cell(self, dataset, backend, initial_values=None):
        initial_values = initial_values or {}
        row = self._row_index[dataset]

        container = QtWidgets.QWidget()
        container_layout = QtWidgets.QVBoxLayout(container)
        container_layout.setContentsMargins(0, 4, 0, 4)
        container_layout.setSpacing(0)

        form_widget = QtWidgets.QWidget()
        form = QtWidgets.QFormLayout(form_widget)
        form.setContentsMargins(0, 0, 0, 0)
        form.setSpacing(8)
        form.setLabelAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
        form.setFieldGrowthPolicy(QtWidgets.QFormLayout.AllNonFixedFieldsGrow)

        self._dataset_config_inputs[dataset] = {}
        fields = self._visible_fields(backend)
        if not fields:
            note = QtWidgets.QLabel("Select a backend to configure its fields.")
            note.setStyleSheet("color: #666;")
            form.addRow(note)
        else:
            for key in fields:
                meta = self._field_meta(backend, key)
                label = meta.get("label", key.replace("_", " ").title())
                if self._field_is_bool(backend, key):
                    editor = QtWidgets.QCheckBox()
                    if meta.get("tooltip"):
                        editor.setToolTip(meta["tooltip"])
                    if key in initial_values:
                        editor.setChecked(bool(initial_values[key]))
                    else:
                        default = self._field_default(backend, key)
                        if default is not None:
                            editor.setChecked(bool(default))
                    editor.toggled.connect(lambda _c=False: self._mark_dirty())
                else:
                    editor = QtWidgets.QLineEdit()
                    editor.setPlaceholderText(meta.get("placeholder", label))
                    if meta.get("tooltip"):
                        editor.setToolTip(meta["tooltip"])
                    if key in initial_values:
                        editor.setText(str(initial_values[key]))
                    else:
                        default = self._field_default(backend, key)
                        if default is not None:
                            editor.setText(str(default))
                    # A config change invalidates previously fetched columns.
                    editor.textChanged.connect(
                        lambda _t=None, ds=dataset: self._invalidate_fetched(ds)
                    )
                    editor.textChanged.connect(lambda _t=None: self._mark_dirty())
                suffix = " *" if self._field_required(backend, key) else ""
                form.addRow(f"{label}{suffix}:", editor)
                self._dataset_config_inputs[dataset][key] = editor

        container_layout.addWidget(form_widget)
        container_layout.addStretch(1)
        self.table.setCellWidget(row, 3, container)
        self.table.resizeRowToContents(row)

    # ------------------------------------------------------------------ #
    # Event handlers
    # ------------------------------------------------------------------ #

    def _on_backend_changed(self, dataset, backend):
        if backend == self.UNSELECTED_BACKEND:
            backend = ""
        self._dataset_backends[dataset] = backend
        # Changing the backend invalidates the source columns and mapping.
        self._dataset_source_columns.pop(dataset, None)
        self._dataset_columns[dataset] = {}
        self._render_config_cell(dataset, backend)
        self._refresh_mapping_label(dataset)
        self._mark_dirty()

    def _invalidate_fetched(self, dataset):
        """Drop fetched columns when a config value changes (mapping kept)."""
        if dataset in self._dataset_source_columns:
            self._dataset_source_columns.pop(dataset, None)
            self._refresh_mapping_label(dataset)

    def _on_edit_mapping(self, dataset):
        dialog = ColumnMappingDialog(
            dataset,
            self._project_columns,
            self._dataset_source_columns.get(dataset),
            self._dataset_columns.get(dataset, {}),
            parent=self,
        )
        if dialog.exec() == QtWidgets.QDialog.Accepted:
            self._dataset_columns[dataset] = dialog.mapping()
            self._refresh_mapping_label(dataset)
            self._mark_dirty()

    def _refresh_mapping_label(self, dataset):
        status = self._status_labels.get(dataset)
        if status is None:
            return
        n_map = len(self._dataset_columns.get(dataset, {}))
        cols = self._dataset_source_columns.get(dataset)
        fetched = "" if cols is None else f", {len(cols)} cols fetched"
        status.setText(f"{n_map} mapped{fetched}")
        button = self._map_buttons.get(dataset)
        if button is not None:
            configured = self._dataset_backends.get(dataset, "") in BACKEND_REGISTRY
            button.setEnabled(configured)

    def _configured_datasets(self):
        return [
            ds
            for ds in self._datasets
            if self._dataset_backends.get(ds, "") in BACKEND_REGISTRY
        ]

    # ------------------------------------------------------------------ #
    # Auto-map: fetch columns + auto-match (async)
    # ------------------------------------------------------------------ #

    def _credentials_manager(self):
        """Shared credentials manager (created on first use)."""
        manager = getattr(self, "_credentials_manager_instance", None)
        if manager is None:
            manager = CredentialsManager()
            self._credentials_manager_instance = manager
        return manager

    def _ensure_credentials_for(self, backend_names):
        """Prompt for missing tokens before dispatching to worker threads."""
        manager = self._credentials_manager()
        for name in dict.fromkeys(backend_names):
            service_key = service_key_for_backend(name)
            if service_key and not manager.ensure_credentials(service_key, parent=self):
                return False
        return True

    def _on_auto_map(self):
        if self._busy:
            return
        datasets = self._configured_datasets()
        if not datasets:
            self._set_status("Select a backend for at least one dataset first.")
            return

        # Workers cannot show dialogs, so resolve credentials up front.
        if not self._ensure_credentials_for(
            self._dataset_backends[ds] for ds in datasets
        ):
            self._set_status("Auto-map cancelled: credentials required.")
            return

        self._set_busy(True, f"Auto-mapping {len(datasets)} source(s)…")
        self._pending_fetch = set(datasets)
        for ds in datasets:
            backend_name = self._dataset_backends[ds]
            config = self._collect_config_values(ds)
            sample_ids = self._sample_ids(ds)
            runnable = _FunctionRunnable(
                self._fetch_worker, ds, backend_name, config, sample_ids
            )
            self._threadpool().start(runnable)

    def _sample_ids(self, dataset):
        mask = self._meta["dataset"].astype(str) == str(dataset)
        ids = self._meta.loc[mask, "id"].head(self.SAMPLE_SIZE)
        return ids.tolist()

    def _fetch_worker(self, dataset, backend_name, config, sample_ids):
        """Runs off the UI thread; emits results via a queued signal."""
        try:
            backend = build_backend(backend_name, config)
            columns = self._discover_columns(backend, sample_ids)
            self._fetchFinished.emit(dataset, list(columns), None)
        except Exception as exc:  # noqa: BLE001 - surfaced in UI
            logger.exception("Fetching source columns failed for %s", dataset)
            detail = str(exc).strip() or exc.__class__.__name__
            self._fetchFinished.emit(dataset, None, detail)

    def _discover_columns(self, backend, sample_ids):
        """Return a source's column names, preferring cheap schema accessors."""
        # Check the class so we don't trigger the property getter just to probe.
        for attr in ("data_columns", "available_fields"):
            if hasattr(type(backend), attr):
                try:
                    cols = list(getattr(backend, attr))
                    if cols:
                        return cols
                except Exception:  # noqa: BLE001 - fall back to sampled read
                    pass
        src = backend.read_annotations(ids=sample_ids)
        if src is None:
            return []
        return list(src.columns)

    def _on_fetch_finished(self, dataset, columns, error):
        if error is not None:
            label = self._status_labels.get(dataset)
            if label is not None:
                label.setText(f"fetch failed: {error}")
        else:
            self._dataset_source_columns[dataset] = columns
            # Auto-match any not-yet-mapped project columns.
            existing = dict(self._dataset_columns.get(dataset, {}))
            before = dict(existing)
            suggested = auto_match_columns(self._project_columns, columns)
            for proj, src in suggested.items():
                existing.setdefault(proj, src)
            self._dataset_columns[dataset] = existing
            if existing != before:
                self._mark_dirty()
            self._refresh_mapping_label(dataset)

        self._pending_fetch.discard(dataset)
        if not self._pending_fetch:
            self._set_busy(False)
            failed = [
                ds
                for ds in self._configured_datasets()
                if ds not in self._dataset_source_columns
            ]
            if failed:
                self._set_status(
                    f"Auto-mapped. Failed: {', '.join(failed)}.",
                    color=self.STATUS_WARNING,
                )
            else:
                self._set_status("Auto-mapped columns from sources.")

    # ------------------------------------------------------------------ #
    # Save
    # ------------------------------------------------------------------ #

    def _current_specs(self, *, require_columns=True):
        specs = []
        for ds in self._datasets:
            backend = self._dataset_backends.get(ds, "")
            if backend not in BACKEND_REGISTRY:
                continue
            columns = {
                str(p): str(s)
                for p, s in self._dataset_columns.get(ds, {}).items()
                if s and not _is_reserved(p)
            }
            if require_columns and not columns:
                continue
            specs.append(
                MetaSourceSpec(
                    dataset=ds,
                    backend=backend,
                    config=self._collect_config_values(ds),
                    columns=columns,
                    last_updated=self._dataset_last_updated.get(ds),
                )
            )
        return specs

    def _on_save_sources(self):
        specs = self._current_specs(require_columns=False)
        write_meta_sources(self._info, specs)
        self._set_status(f"Saved {len(specs)} source definition(s) to project.")
        self._mark_clean()
        self.sourcesSaved.emit()

    # ------------------------------------------------------------------ #
    # Update meta (async)
    # ------------------------------------------------------------------ #

    def _on_update(self):
        if self._busy:
            return
        specs = self._current_specs(require_columns=True)
        if not specs:
            self._set_status(
                "Configure a backend and map at least one column first."
            )
            return

        if not self._ensure_credentials_for(spec.backend for spec in specs):
            self._set_status("Update cancelled: credentials required.")
            return

        # Persist the source definitions before pulling.
        write_meta_sources(self._info, self._current_specs(require_columns=False))
        self._mark_clean()

        self._progress = QtWidgets.QProgressDialog(
            "Updating meta data…", None, 0, len(specs), self
        )
        self._progress.setWindowModality(QtCore.Qt.ApplicationModal)
        self._progress.setCancelButton(None)
        self._progress.setAutoClose(True)
        self._progress.setValue(0)

        self._set_busy(True, "Pulling fresh meta data…")
        runnable = _FunctionRunnable(self._update_worker, self._meta, specs)
        self._threadpool().start(runnable)

    def _update_worker(self, meta, specs):
        try:
            updated, report = update_meta(meta, specs)
            self._updateFinished.emit(updated, report, None)
        except Exception as exc:  # noqa: BLE001 - surfaced in UI
            logger.exception("Meta update failed")
            detail = str(exc).strip() or exc.__class__.__name__
            self._updateFinished.emit(None, None, detail)

    def _on_update_finished(self, updated, report, error):
        if getattr(self, "_progress", None) is not None:
            self._progress.setValue(self._progress.maximum())
            self._progress = None
        self._set_busy(False)

        if error is not None:
            self._set_status(f"Update failed: {error}", color=self.STATUS_ERROR)
            QtWidgets.QMessageBox.critical(self, "Meta update failed", error)
            return

        # Keep our reference current so a subsequent update sees new columns.
        self._meta = updated
        partial = bool(report.errors or report.columns_failed)
        self._set_status(
            report.summary(),
            color=self.STATUS_WARNING if partial else self.STATUS_SUCCESS,
        )
        self.metaUpdated.emit(updated, report)
        if partial:
            QtWidgets.QMessageBox.warning(self, "Meta update", report.summary())

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    def _threadpool(self):
        parent = self.parent()
        if parent is not None:
            pool = getattr(parent, "threadpool", None)
            if isinstance(pool, QtCore.QThreadPool):
                return pool
        return self._fallback_threadpool

    # Status colors: neutral grey, success green, warning amber, error red.
    STATUS_NEUTRAL = "#ededed"
    STATUS_SUCCESS = "#05fc6c"
    STATUS_WARNING = "#b26a00"
    STATUS_ERROR = "#c62828"

    def _set_status(self, text, color=None):
        """Set the footer status text and color (defaults to neutral grey)."""
        self.status_label.setStyleSheet(f"color: {color or self.STATUS_NEUTRAL};")
        self.status_label.setText(text)

    def _set_busy(self, busy, message=None):
        self._busy = busy
        self.automap_button.setEnabled(not busy)
        self.update_button.setEnabled(not busy)
        self._update_save_enabled()
        if message is not None:
            self._set_status(message)

    def _mark_dirty(self):
        """Flag unsaved changes (no-op while populating from presets)."""
        if self._loading:
            return
        self._dirty = True
        self._update_save_enabled()

    def _mark_clean(self):
        self._dirty = False
        self._update_save_enabled()

    def _update_save_enabled(self):
        self.save_button.setEnabled(self._dirty and not self._busy)
