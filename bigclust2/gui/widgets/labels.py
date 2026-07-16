import sys
from pathlib import Path

import traceback
import numpy as np
import pandas as pd
from PySide6 import QtCore, QtWidgets
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QComboBox,
    QHeaderView,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QProgressDialog,
    QPushButton,
    QMenu,
    QTableView,
    QVBoxLayout,
    QWidget,
    QCheckBox,
)


class FeatureTableProxyModel(QtCore.QSortFilterProxyModel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._row_filter = ""
        self._column_filter = ""
        self._hide_zero_rows = False
        self._visible_columns_cache = None

    def invalidateFilter(self):
        self._visible_columns_cache = None
        super().invalidateFilter()

    def _get_visible_source_columns(self):
        if self._visible_columns_cache is not None:
            return self._visible_columns_cache

        model = self.sourceModel()
        if model is None:
            return []

        visible = []
        for column in range(model.columnCount(QtCore.QModelIndex())):
            if self.filterAcceptsColumn(column, QtCore.QModelIndex()):
                visible.append(column)

        self._visible_columns_cache = visible
        return visible

    def setRowFilter(self, text):
        self._row_filter = text.strip().lower()
        self.invalidateFilter()

    def setColumnFilter(self, text):
        self._column_filter = text.strip().lower()
        self.invalidateFilter()

    def setHideZeroRows(self, hide):
        self._hide_zero_rows = bool(hide)
        self.invalidateFilter()

    def filterAcceptsRow(self, source_row, source_parent):
        model = self.sourceModel()
        if model is None:
            return True

        if self._hide_zero_rows:
            visible_columns = self._get_visible_source_columns()
            if visible_columns:
                values = model._values[source_row, visible_columns]
                try:
                    values = np.asarray(values, dtype=np.float64)
                except ValueError:
                    values = np.array(
                        [float(v) if pd.notna(v) else 0.0 for v in values],
                        dtype=np.float64,
                    )
                if np.all(np.nan_to_num(values) == 0):
                    return False
            else:
                return False

        if not self._row_filter:
            return True

        value = model.headerData(source_row, QtCore.Qt.Vertical, QtCore.Qt.DisplayRole)
        return self._row_filter in str(value).lower()

    def filterAcceptsColumn(self, source_column, source_parent):
        if not self._column_filter:
            return True
        model = self.sourceModel()
        header = model.headerData(source_column, QtCore.Qt.Horizontal, QtCore.Qt.DisplayRole)
        return self._column_filter in str(header).lower()


class DataFrameModel(QtCore.QAbstractTableModel):
    def __init__(self, dataframe=pd.DataFrame(), parent=None):
        super().__init__(parent)
        self._update_cache(dataframe)

    def _update_cache(self, dataframe):
        self._dataframe = dataframe
        self._values = dataframe.to_numpy()
        self._row_headers = []
        self._column_headers = []

        for idx in dataframe.index:
            if isinstance(idx, tuple):
                self._row_headers.append(" / ".join(str(part) for part in idx))
            else:
                self._row_headers.append(str(idx))

        for col in dataframe.columns:
            if isinstance(col, tuple):
                self._column_headers.append(" / ".join(str(part) for part in col))
            else:
                self._column_headers.append(str(col))

    def setDataFrame(self, dataframe):
        self.beginResetModel()
        self._update_cache(dataframe)
        self.endResetModel()

    def rowCount(self, parent=QtCore.QModelIndex()):
        if parent.isValid():
            return 0
        return len(self._values)

    def columnCount(self, parent=QtCore.QModelIndex()):
        if parent.isValid():
            return 0
        return len(self._values[0]) if self._values.size else 0

    def data(self, index, role=QtCore.Qt.DisplayRole):
        if not index.isValid():
            return None
        if role == QtCore.Qt.TextAlignmentRole:
            return QtCore.Qt.AlignCenter
        if role != QtCore.Qt.DisplayRole:
            return None
        value = self._values[index.row(), index.column()]
        return "" if pd.isna(value) else str(value)

    def headerData(self, section, orientation, role=QtCore.Qt.DisplayRole):
        if role != QtCore.Qt.DisplayRole:
            return None
        if orientation == QtCore.Qt.Horizontal:
            return self._column_headers[section] if section < len(self._column_headers) else str(section)
        return self._row_headers[section] if section < len(self._row_headers) else str(section)


class LabelTableProxyModel(QtCore.QSortFilterProxyModel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._row_filter = ""

    def setRowFilter(self, text):
        self._row_filter = text.strip().lower()
        self.invalidateFilter()

    def filterAcceptsRow(self, source_row, source_parent):
        if not self._row_filter:
            return True
        model = self.sourceModel()
        for column in range(model.columnCount(source_parent)):
            index = model.index(source_row, column, source_parent)
            value = model.data(index, QtCore.Qt.DisplayRole)
            if value is not None and self._row_filter in str(value).lower():
                return True
        return False


class LabelTableModel(QtCore.QAbstractTableModel):
    def __init__(self, meta=None, labels=None, parent=None):
        super().__init__(parent)
        self._meta = pd.DataFrame(columns=["id", "dataset"])
        self._label_series = pd.Series(dtype="string")
        if meta is not None:
            self.setMeta(meta)
        if labels is not None:
            self.setLabels(labels)

    def setMeta(self, meta):
        meta = meta.reset_index(drop=True).copy()
        self.beginResetModel()
        self._meta = meta
        self._label_series = self._label_series.reindex(self._meta.index, fill_value=pd.NA)
        self.endResetModel()

    def setLabels(self, labels):
        labels = labels.astype("string").copy()
        labels.index = self._meta.index
        self.beginResetModel()
        self._label_series = labels
        self.endResetModel()

    def get_labels_dataframe(self):
        return pd.DataFrame(
            {
                "id": self._meta["id"].astype(object),
                "dataset": self._meta["dataset"].astype(object),
                "label": self._label_series,
            }
        )

    def rowCount(self, parent=QtCore.QModelIndex()):
        if parent.isValid():
            return 0
        return len(self._meta)

    def columnCount(self, parent=QtCore.QModelIndex()):
        if parent.isValid():
            return 0
        return 3

    def data(self, index, role=QtCore.Qt.DisplayRole):
        if not index.isValid():
            return None
        if role == QtCore.Qt.TextAlignmentRole:
            return QtCore.Qt.AlignCenter
        if role not in (QtCore.Qt.DisplayRole, QtCore.Qt.EditRole):
            return None

        row = index.row()
        column = index.column()
        if column == 0:
            return str(self._meta.at[row, "id"])
        if column == 1:
            return str(self._meta.at[row, "dataset"])
        value = self._label_series.iat[row]
        return "" if pd.isna(value) else str(value)

    def flags(self, index):
        if not index.isValid():
            return QtCore.Qt.NoItemFlags
        flags = QtCore.Qt.ItemIsSelectable | QtCore.Qt.ItemIsEnabled
        if index.column() == 2:
            flags |= QtCore.Qt.ItemIsEditable
        return flags

    def setData(self, index, value, role=QtCore.Qt.EditRole):
        if not index.isValid() or index.column() != 2 or role != QtCore.Qt.EditRole:
            return False
        text = str(value).strip()
        if text == "":
            self._label_series.iat[index.row()] = pd.NA
        else:
            self._label_series.iat[index.row()] = text
        self.dataChanged.emit(index, index, [QtCore.Qt.DisplayRole, QtCore.Qt.EditRole])
        return True

    def headerData(self, section, orientation, role=QtCore.Qt.DisplayRole):
        if role != QtCore.Qt.DisplayRole:
            return None
        if orientation == QtCore.Qt.Horizontal:
            return ["id", "dataset", "label"][section]
        return str(section)


class ConnectivityLoadDialog(QDialog):
    def __init__(self, datasets, initial_paths=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Load connectivity edge lists")
        self.setModal(True)
        self._file_inputs = {}
        self._initial_paths = initial_paths or {}

        layout = QVBoxLayout(self)
        description = QLabel(
            "Select one edge list file per dataset. "
            "Expected columns: presynaptic id, postsynaptic id, weight."
        )
        description.setWordWrap(True)
        layout.addWidget(description)

        form_layout = QFormLayout()
        for dataset in datasets:
            row_container = QWidget()
            row_layout = QHBoxLayout(row_container)
            row_layout.setContentsMargins(0, 0, 0, 0)
            file_edit = QLineEdit()
            file_edit.setText(self._initial_paths.get(dataset, ""))
            file_edit.textChanged.connect(self._update_ok_button_state)
            browse_btn = QPushButton("Browse…")
            browse_btn.clicked.connect(
                lambda _, ds=dataset, edit=file_edit: self._browse_file(ds, edit)
            )
            row_layout.addWidget(file_edit)
            row_layout.addWidget(browse_btn)
            form_layout.addRow(QLabel(str(dataset)), row_container)
            self._file_inputs[dataset] = file_edit

        layout.addLayout(form_layout)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self._ok_button = buttons.button(QDialogButtonBox.Ok)
        self._ok_button.setText("Load")
        self._ok_button.setEnabled(False)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self._update_ok_button_state()

    def _browse_file(self, dataset, line_edit):
        path, _ = QFileDialog.getOpenFileName(
            self,
            f"Select edge list for dataset '{dataset}'",
            str(Path.home()),
            "Data files (*.parquet *.csv *.tsv *.feather);;All files (*)",
        )
        if path:
            line_edit.setText(path)

    def _update_ok_button_state(self):
        if getattr(self, "_ok_button", None) is None:
            return
        complete = all(edit.text().strip() for edit in self._file_inputs.values())
        self._ok_button.setEnabled(complete)

    def selected_files(self):
        return {
            dataset: edit.text().strip()
            for dataset, edit in self._file_inputs.items()
            if edit.text().strip()
        }


class LabelsWidget(QWidget):
    """Prototype widget for label-based connectivity aggregation."""

    features_computed = QtCore.Signal(object, object)

    def __init__(self, meta=None, labels=None, parent=None):
        super().__init__(parent)
        self.meta = None
        self.labels = None
        self.edge_list = None
        self._original_edge_list = None
        self.aggregated_features = None
        self._labels_source = None
        self._selected_meta_label_column = None
        self._computed_label_series = None
        self._computed_join_mode = None
        self._computed_common_labels = None
        self._ignore_item_changed = False
        self._visible_rows = []
        self._auto_compile_running = False
        self.settings = QtCore.QSettings("BigClust", "BigClustGUI")

        self.setWindowTitle("Labels and Connectivity Prototype")
        self._build_ui()

        if meta is not None:
            self.set_meta(meta)
        if labels is not None:
            self.set_labels(labels)

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setSpacing(8)
        root.setContentsMargins(10, 10, 10, 10)

        self.labels_label = QLabel("Labels:")
        self.labels_label.setToolTip("Metadata and label status.")
        self.labels_label.setStyleSheet("font-weight: bold;")

        self.labels_status_label = QLabel("No metadata loaded.")
        self.labels_status_label.setStyleSheet("color: #666666;")

        self._load_labels_menu = QMenu(self)
        self._load_labels_menu.addAction("From file...").triggered.connect(
            self._load_labels_from_file
        )
        self._load_labels_menu.addAction("From project...").triggered.connect(
            self._load_labels_from_project
        )

        self.load_labels_btn = QPushButton("Select Labels")
        self.load_labels_btn.setMenu(self._load_labels_menu)
        self.load_labels_btn.clicked.connect(self._load_labels_from_file)

        self.sync_labels_btn = QPushButton("Re-Sync")
        self.sync_labels_btn.setVisible(False)
        self.sync_labels_btn.setToolTip("Reload labels from the currently selected metadata label column.")
        self.sync_labels_btn.clicked.connect(self._sync_labels_from_meta)

        labels_row = QtWidgets.QHBoxLayout()
        labels_row.addWidget(self.labels_label)
        labels_row.addStretch()
        labels_row.addWidget(self.load_labels_btn)
        labels_row.addWidget(self.sync_labels_btn)
        root.addLayout(labels_row)

        labels_row = QtWidgets.QHBoxLayout()
        labels_row.addWidget(self.labels_status_label)
        root.addLayout(labels_row)

        search_layout = QtWidgets.QHBoxLayout()
        search_label = QLabel("Search:")
        self.search_edit = QLineEdit()
        self.search_edit.setPlaceholderText("Search id, dataset, or label")
        self.search_edit.textChanged.connect(self._schedule_refresh_table)
        search_layout.addWidget(search_label)
        search_layout.addWidget(self.search_edit)
        root.addLayout(search_layout)

        self.labels_model = LabelTableModel(parent=self)
        self.labels_proxy = LabelTableProxyModel(self)
        self.labels_proxy.setSourceModel(self.labels_model)

        self.table = QTableView()
        self.table.setModel(self.labels_proxy)
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.Interactive)
        header.setStretchLastSection(True)
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(
            QTableView.DoubleClicked | QTableView.EditKeyPressed
        )
        self.table.setSelectionBehavior(QTableView.SelectRows)
        self.table.setAlternatingRowColors(True)
        self.labels_model.dataChanged.connect(self._on_label_data_changed)
        root.addWidget(self.table, stretch=1)

        self.compute_btn = QPushButton("Compute aggregated features")
        self.compute_btn.setEnabled(False)
        self.compute_btn.setToolTip(
            "Compute aggregated features using the selected join strategy."
        )
        self.compute_btn.clicked.connect(self._compute_features)

        self.join_mode_combo = QComboBox()
        self.join_mode_combo.addItems(["Inner join", "Outer join"])
        self.join_mode_combo.setCurrentText("Inner join")
        self.join_mode_combo.setToolTip(
            "Select whether label/connectivity aggregation should use an inner join or outer join strategy."
        )

        self.auto_compile_checkbox = QCheckBox("Auto-compute")
        self.auto_compile_checkbox.setChecked(False)
        self.auto_compile_checkbox.setToolTip(
            "When enabled, compute aggregated features automatically when data changes."
        )
        self.auto_compile_checkbox.toggled.connect(self._on_auto_compile_toggled)

        compute_layout = QHBoxLayout()
        compute_layout.addWidget(self.compute_btn)
        compute_layout.addWidget(QLabel("Join:"))
        compute_layout.addWidget(self.join_mode_combo)
        compute_layout.addWidget(self.auto_compile_checkbox)
        compute_layout.addStretch()
        root.addLayout(compute_layout)

        self.features_label = QLabel("Aggregated Features:")
        self.features_label.setToolTip(
            "The computed aggregated features are shown in this table."
        )
        self.features_label.setStyleSheet("font-weight: bold;")
        self.features_status_label = QLabel("No computed features.")
        self.features_status_label.setStyleSheet("color: #666666;")
        self.features_model = DataFrameModel(pd.DataFrame(), self)
        self.features_proxy = FeatureTableProxyModel(self)
        self.features_proxy.setSourceModel(self.features_model)
        self._filter_timer = QtCore.QTimer(self)
        self._filter_timer.setSingleShot(True)
        self._filter_timer.setInterval(300)
        self._filter_timer.timeout.connect(self._update_feature_filters)
        self._search_timer = QtCore.QTimer(self)
        self._search_timer.setSingleShot(True)
        self._search_timer.setInterval(300)
        self._search_timer.timeout.connect(self._update_label_filter)
        self._auto_compute_timer = QtCore.QTimer(self)
        self._auto_compute_timer.setSingleShot(True)
        self._auto_compute_timer.setInterval(1000)
        self._auto_compute_timer.timeout.connect(self._auto_compute_if_enabled)
        self.features_table = QTableView()
        self.features_table.setModel(self.features_proxy)
        features_header = self.features_table.horizontalHeader()
        features_header.setSectionResizeMode(QHeaderView.Interactive)
        features_header.setStretchLastSection(True)
        self.features_table.verticalHeader().setVisible(True)
        self.features_table.verticalHeader().setSectionResizeMode(
            QHeaderView.Interactive
        )
        self.features_table.setEditTriggers(QTableView.NoEditTriggers)

        filter_layout = QHBoxLayout()
        self.row_filter_edit = QLineEdit()
        self.row_filter_edit.setPlaceholderText("Filter rows (id / dataset)")
        self.row_filter_edit.setToolTip("Filter feature table rows by id or dataset.")
        self.row_filter_edit.textChanged.connect(self._schedule_update_feature_filters)
        self.column_filter_edit = QLineEdit()
        self.column_filter_edit.setPlaceholderText("Filter columns")
        self.column_filter_edit.setToolTip("Filter feature table columns by feature name.")
        self.column_filter_edit.textChanged.connect(self._schedule_update_feature_filters)
        self.hide_zeros_checkbox = QCheckBox("Hide zero rows")
        self.hide_zeros_checkbox.setChecked(False)
        self.hide_zeros_checkbox.setToolTip(
            "Hide rows where all visible feature values are zero."
        )
        self.hide_zeros_checkbox.toggled.connect(self._on_hide_zero_rows_toggled)
        filter_layout.addWidget(QLabel("Row filter:"))
        filter_layout.addWidget(self.row_filter_edit)
        filter_layout.addWidget(QLabel("Column filter:"))
        filter_layout.addWidget(self.column_filter_edit)
        filter_layout.addWidget(self.hide_zeros_checkbox)

        separator = QtWidgets.QFrame()
        separator.setFrameShape(QtWidgets.QFrame.HLine)
        separator.setFrameShadow(QtWidgets.QFrame.Sunken)
        root.addWidget(separator)

        root.addWidget(self.features_label)
        root.addWidget(self.features_status_label)
        root.addLayout(filter_layout)
        root.addWidget(self.features_table, stretch=1)

        actions_layout = QHBoxLayout()
        self.load_meta_btn = QPushButton("Load metadata")
        self.load_meta_btn.clicked.connect(self._load_meta_file)
        self.load_connectivity_btn = QPushButton("Load connectivity")
        self.load_connectivity_btn.clicked.connect(self._load_connectivity_file)
        actions_layout.addWidget(self.load_meta_btn)
        actions_layout.addWidget(self.load_connectivity_btn)
        actions_layout.addStretch()
        self.push_to_project_btn = QPushButton("Push to Project")
        self.push_to_project_btn.setEnabled(False)
        self.push_to_project_btn.clicked.connect(self._push_to_project)
        self.export_to_parquet_btn = QPushButton("Export to Parquet")
        self.export_to_parquet_btn.setEnabled(False)
        self.export_to_parquet_btn.clicked.connect(self._export_to_parquet)
        actions_layout.addWidget(self.push_to_project_btn)
        actions_layout.addWidget(self.export_to_parquet_btn)
        root.addLayout(actions_layout)

        self.data_status_label = QLabel("No metadata or connectivity loaded.")
        self.data_status_label.setStyleSheet("color: #666666;")
        root.addWidget(self.data_status_label)

        self._update_connectivity_button_style()

    def _load_meta_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Open metadata file",
            str(Path.home()),
            "Data files (*.parquet *.csv *.tsv *.feather);;All files (*)",
        )
        if not path:
            return

        meta = self._read_table(path)
        if "id" not in meta.columns or "dataset" not in meta.columns:
            QtWidgets.QMessageBox.warning(
                self,
                "Invalid metadata",
                "Metadata must contain at least 'id' and 'dataset' columns.",
            )
            return

        self.set_meta(meta)

        label_cols = [c for c in meta.columns if c not in {"id", "dataset"}]
        if not label_cols:
            self.labels_status_label.setText(
                f"Loaded metadata ({len(meta):,} rows); no extra columns available for labels."
            )
            return

        choice, ok = QtWidgets.QInputDialog.getItem(
            self,
            "Choose label column",
            "Use metadata column for labels:",
            label_cols,
            0,
            False,
        )
        if ok and choice:
            self._selected_meta_label_column = choice
            self.set_labels(
                meta[["id", "dataset", choice]].rename(columns={choice: "label"}),
                source=f"{choice} (meta)"
            )

    def _load_labels_from_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Open labels file",
            str(Path.home()),
            "Label files (*.parquet *.csv *.tsv *.feather);;All files (*)",
        )
        if not path:
            return

        labels = self._read_table(path)
        if {"id", "dataset", "label"}.issubset(labels.columns):
            self.set_labels(labels[["id", "dataset", "label"]], source=Path(path).name)
            return

        if self.meta is None:
            QtWidgets.QMessageBox.warning(
                self,
                "Unable to load labels",
                "Labels file must contain 'id', 'dataset', and 'label' columns unless metadata is loaded first.",
            )
            return

        if "label" not in labels.columns:
            QtWidgets.QMessageBox.warning(
                self,
                "Unable to load labels",
                "Labels file must contain a 'label' column.",
            )
            return

        if len(labels) == len(self.meta):
            self.set_labels(labels[["label"]].reset_index(drop=True), source=Path(path).name)
        else:
            QtWidgets.QMessageBox.warning(
                self,
                "Unable to align labels",
                "The label file could not be aligned to the current metadata. Use a file with id/dataset/label.",
            )

    def _load_labels_from_project(self):
        if self.meta is None:
            QtWidgets.QMessageBox.warning(
                self,
                "Load metadata first",
                "Load metadata before selecting labels from project.",
            )
            return

        label_cols = [c for c in self.meta.columns if c not in {"id", "dataset"}]
        if not label_cols:
            QtWidgets.QMessageBox.warning(
                self,
                "No project labels",
                "No additional columns are available in the loaded metadata to use as labels.",
            )
            return

        choice, ok = QtWidgets.QInputDialog.getItem(
            self,
            "Choose label column",
            "Use metadata column for labels:",
            label_cols,
            0,
            False,
        )
        if ok and choice:
            self._selected_meta_label_column = choice
            self.set_labels(
                self.meta[["id", "dataset", choice]].rename(columns={choice: "label"}),
                source=f"{choice} (meta)"
            )

    def _load_connectivity_file(self):
        if self.meta is None:
            QtWidgets.QMessageBox.warning(
                self,
                "Load metadata first",
                "Load metadata before selecting connectivity edge lists.",
            )
            return

        datasets = sorted(self.meta["dataset"].dropna().unique().tolist())
        dialog = ConnectivityLoadDialog(
            datasets, self._load_saved_connectivity_paths(datasets), self
        )
        if dialog.exec() != QDialog.Accepted:
            return

        files_by_dataset = dialog.selected_files()
        missing = [ds for ds in datasets if ds not in files_by_dataset]
        if missing:
            QtWidgets.QMessageBox.warning(
                self,
                "Missing edge list files",
                "Please provide one edge list file for each dataset: "
                + ", ".join(str(ds) for ds in missing),
            )
            return

        progress = QProgressDialog(
            "Loading dataset edge lists...",
            None,
            0,
            len(datasets),
            self,
        )
        progress.setWindowTitle("Loading Connectivity")
        progress.setWindowModality(QtCore.Qt.WindowModal)
        progress.setAutoClose(False)
        progress.setMinimumDuration(0)
        progress.setValue(0)

        try:
            edge_list = self._load_connectivity_from_edge_lists(
                files_by_dataset, progress
            )
        except Exception as exc:
            progress.close()
            QtWidgets.QMessageBox.critical(
                self,
                "Unable to load connectivity",
                f"Failed to parse dataset edge lists: {exc}",
            )
            return

        progress.close()

        if edge_list.empty:
            QtWidgets.QMessageBox.warning(
                self,
                "Connectivity mismatch",
                "No valid edges were found for the loaded datasets.",
            )
            return

        self.edge_list = edge_list
        self._original_edge_list = edge_list.copy(deep=True)
        self._save_connectivity_paths(files_by_dataset)
        self.labels_status_label.setText(
            f"Loaded edge list connectivity for {len(edge_list):,} edges."
        )
        self.data_status_label.setText(
            f"Connectivity loaded ({len(edge_list):,} edges)."
        )
        self._refresh_compute_state()

    def _read_table(self, path):
        path = Path(path)
        try:
            if path.suffix == ".parquet":
                return pd.read_parquet(path)
            if path.suffix == ".feather":
                return pd.read_feather(path)
            if path.suffix == ".csv":
                return pd.read_csv(path)
            if path.suffix == ".tsv":
                return pd.read_csv(path, sep="\t")
            return pd.read_csv(path)
        except Exception as exc:
            QtWidgets.QMessageBox.warning(
                self,
                "Read error",
                f"Could not read file '{path.name}': {exc}",
            )
            return pd.DataFrame()

    def _infer_edge_columns(self, df):
        if df.shape[1] == 3:
            cols = list(df.columns[:3])
            return cols[0], cols[1], cols[2]

        lower = {col.lower().strip(): col for col in df.columns}

        def find(candidates):
            for candidate in candidates:
                if candidate in lower:
                    return lower[candidate]
            return None

        pre = find(["presynaptic id", "presynaptic_id", "presyn_id", "pre_id", "pre"])
        post = find(
            ["postsynaptic id", "postsynaptic_id", "postsyn_id", "post_id", "post"]
        )
        weight = find(["weight", "w", "weight_", "edge_weight"])

        if pre is None or post is None or weight is None:
            raise ValueError(
                "Edge list must contain columns named 'presynaptic id', 'postsynaptic id', and 'weight', or exactly three columns. "
                f"Found columns: {', '.join(df.columns)}"
            )

        return pre, post, weight

    def _load_connectivity_from_edge_lists(
        self, files_by_dataset, progress_dialog=None
    ):
        edge_parts = []
        ids_by_dataset = {}
        for dataset, subset in self.meta.groupby("dataset", sort=False):
            ids_by_dataset[dataset] = subset["id"].unique()

        total = len(files_by_dataset)
        current = 0
        file_cache = {}
        for dataset, path in files_by_dataset.items():
            current += 1
            if progress_dialog is not None:
                progress_dialog.setLabelText(
                    f"Loading edge list for dataset '{dataset}' ({current}/{total})..."
                )
                progress_dialog.setValue(current - 1)
                QApplication.processEvents()

            file_key = str(Path(path).resolve())
            if file_key in file_cache:
                df = file_cache[file_key]
            else:
                df = self._read_table(path)
                if df is None or df.empty:
                    raise ValueError(
                        f"Could not read edge list for dataset '{dataset}' or file is empty: {path}"
                    )
                file_cache[file_key] = df

            pre_col, post_col, weight_col = self._infer_edge_columns(df)
            df = df[[pre_col, post_col, weight_col]]
            df.columns = ["pre_id", "post_id", "weight"]

            df = df[
                df["pre_id"].isin(ids_by_dataset[dataset])
                & df["post_id"].isin(ids_by_dataset[dataset])
            ].copy()

            if df.empty:
                raise ValueError(
                    f"No valid edges found for dataset '{dataset}': {path}"
                )

            df["pre"] = list(zip(df["pre_id"].values, [dataset] * len(df)))
            df["post"] = list(zip(df["post_id"].values, [dataset] * len(df)))
            df["weight"] = pd.to_numeric(df["weight"], errors="coerce")
            df = df.dropna(subset=["weight"])
            if df.empty:
                raise ValueError(
                    f"No valid edges with numeric weights found for dataset '{dataset}': {path}"
                )

            edge_parts.append(df[["pre", "post", "weight"]])

        if progress_dialog is not None:
            progress_dialog.setValue(total)
            QApplication.processEvents()

        if not edge_parts:
            return pd.DataFrame(columns=["pre", "post", "weight"])

        return pd.concat(edge_parts, ignore_index=True)

    def _align_labels_to_meta(self, labels):
        if self.meta is None:
            return labels

        if {"id", "dataset", "label"}.issubset(labels.columns):
            aligned = self.meta[["id", "dataset"]].merge(
                labels[["id", "dataset", "label"]],
                on=["id", "dataset"],
                how="left",
                sort=False,
            )
            aligned.index = self.meta.index
            return aligned[["label"]]

        if "label" in labels.columns and len(labels) == len(self.meta):
            aligned = labels[["label"]].copy()
            aligned.index = self.meta.index
            return aligned

        return labels

    def set_meta(self, meta):
        if not isinstance(meta, pd.DataFrame):
            raise ValueError("Meta must be a pandas DataFrame.")
        if "id" not in meta.columns or "dataset" not in meta.columns:
            raise ValueError("Meta must contain id and dataset columns.")

        self.meta = meta.reset_index(drop=True).copy()
        self._selected_meta_label_column = None
        if self.labels is not None:
            self.labels = self._align_labels_to_meta(self.labels)
        self.labels_model.setMeta(self.meta)
        if self.labels is not None:
            self.labels_model.setLabels(self._lookup_labels())
        self.labels_status_label.setText(f"Loaded metadata ({len(self.meta):,} rows).")
        self.data_status_label.setText(f"Metadata loaded ({len(self.meta):,} rows).")
        self._refresh_table()
        self._refresh_compute_state()

    def set_labels(self, labels, source=None):
        if isinstance(labels, pd.DataFrame):
            if {"id", "dataset", "label"}.issubset(labels.columns):
                self.labels = labels[["id", "dataset", "label"]].copy()
            elif "label" in labels.columns:
                self.labels = labels[["label"]].copy()
            else:
                raise ValueError(
                    "Labels DataFrame must contain a column named 'label'."
                )
        elif isinstance(labels, pd.Series):
            self.labels = labels.rename("label").to_frame()
        else:
            raise ValueError("Labels must be a pandas Series or DataFrame.")

        if self.meta is not None:
            self.labels = self._align_labels_to_meta(self.labels)
            self.labels_model.setLabels(self._lookup_labels())
        else:
            self.labels = self.labels.reset_index(drop=True)

        self._labels_source = source
        self._update_labels_button_text()
        self.labels_status_label.setText(f"Loaded labels ({len(self.labels):,} rows).")
        self._refresh_table()
        self._refresh_compute_state()

    def _update_labels_button_text(self):
        if self._labels_source:
            self.load_labels_btn.setText(f"Loaded: {self._labels_source}")
        else:
            self.load_labels_btn.setText("Load labels")

    def _refresh_table(self):
        self.labels_proxy.invalidateFilter()

    def _on_label_data_changed(self, top_left, bottom_right, roles):
        if self._ignore_item_changed:
            return
        if top_left.column() > 2 or bottom_right.column() < 2:
            return

        self.labels = self.labels_model.get_labels_dataframe()
        self.labels_status_label.setText("Updated labels from table edits.")
        self._refresh_compute_state()

    def _lookup_labels(self):
        if self.labels is None:
            return pd.Series(
                [pd.NA] * len(self.meta), index=self.meta.index, dtype="string"
            )

        if isinstance(self.labels, pd.Series):
            return self.labels.astype("string")

        if isinstance(self.labels, pd.DataFrame):
            if {"id", "dataset", "label"}.issubset(self.labels.columns):
                labels_df = self.labels[["id", "dataset", "label"]].rename(
                    columns={"label": "__label"}
                )
                merged = self.meta.merge(
                    labels_df,
                    on=["id", "dataset"],
                    how="left",
                    sort=False,
                )
                return merged["__label"].astype("string")

            if "label" in self.labels.columns and len(self.labels) == len(self.meta):
                return self.labels["label"].astype("string")

            if self.labels.shape[1] == 1 and len(self.labels) == len(self.meta):
                return self.labels.iloc[:, 0].astype("string")

        raise ValueError(
            "Labels must be a pandas Series or DataFrame aligned to metadata with a 'label' column."
        )

    def _refresh_compute_state(self):
        self.compute_btn.setEnabled(
            self.meta is not None
            and self.labels is not None
            and self.edge_list is not None
        )
        self._update_connectivity_button_style()
        self._update_load_buttons_visibility()
        self._update_sync_button_visibility()
        self._schedule_auto_compute()

    def _update_sync_button_visibility(self):
        if hasattr(self, "sync_labels_btn"):
            self.sync_labels_btn.setVisible(
                self.meta is not None and self._selected_meta_label_column is not None
            )

    def _update_load_buttons_visibility(self):
        if hasattr(self, "load_meta_btn"):
            self.load_meta_btn.setVisible(self.meta is None)
        if hasattr(self, "load_connectivity_btn"):
            self.load_connectivity_btn.setVisible(self.edge_list is None)

    def _determine_common_labels(self, label_series):
        if self.join_mode_combo.currentText().lower().startswith("inner"):
            dataset_labels = (
                pd.DataFrame({"dataset": self.meta["dataset"], "label": label_series})
                .dropna()
                .groupby(["dataset", "label"])
                .size()
                .unstack(fill_value=0)
            )
            return dataset_labels.columns[(dataset_labels > 0).all()]
        return pd.Index(label_series.dropna().astype("string").unique())

    def _build_node_label_map(self, label_series, common_labels):
        index = pd.MultiIndex.from_frame(self.meta[["id", "dataset"]])
        labels = pd.Series(label_series.values, index=index, dtype="string")
        if common_labels is not None:
            labels = labels.where(labels.isin(common_labels))
        return labels

    def _get_current_edge_list(self):
        return self._original_edge_list if self._original_edge_list is not None else self.edge_list

    def _compute_feature_deltas(
        self,
        old_labels,
        new_labels,
        old_common,
        new_common,
        changed_nodes,
        progress_dialog=None,
    ):
        source_edge_list = self._get_current_edge_list()
        edge_subset = source_edge_list[
            source_edge_list["pre"].isin(changed_nodes)
            | source_edge_list["post"].isin(changed_nodes)
        ].copy()

        if edge_subset.empty:
            return pd.DataFrame(), pd.DataFrame()

        old_map = self._build_node_label_map(old_labels, old_common)
        new_map = self._build_node_label_map(new_labels, new_common)

        old_edge = edge_subset.copy()
        old_edge["pre_label"] = old_edge["pre"].map(old_map)
        old_edge["post_label"] = old_edge["post"].map(old_map)

        new_edge = edge_subset.copy()
        new_edge["pre_label"] = new_edge["pre"].map(new_map)
        new_edge["post_label"] = new_edge["post"].map(new_map)

        old_upstream = (
            old_edge[old_edge.pre_label.notna()]
            .groupby(["pre", "post_label"], sort=False)["weight"]
            .sum()
            .unstack(fill_value=0)
            .astype(np.int64)
        )
        new_upstream = (
            new_edge[new_edge.pre_label.notna()]
            .groupby(["pre", "post_label"], sort=False)["weight"]
            .sum()
            .unstack(fill_value=0)
            .astype(np.int64)
        )

        old_downstream = (
            old_edge[old_edge.post_label.notna()]
            .groupby(["post", "pre_label"], sort=False)["weight"]
            .sum()
            .unstack(fill_value=0)
            .astype(np.int64)
        )
        new_downstream = (
            new_edge[new_edge.post_label.notna()]
            .groupby(["post", "pre_label"], sort=False)["weight"]
            .sum()
            .unstack(fill_value=0)
            .astype(np.int64)
        )

        if progress_dialog is not None:
            progress_dialog.setLabelText("Computing incremental label deltas...")
            progress_dialog.setValue(2)
            QApplication.processEvents()

        upstream_index = old_upstream.index.union(new_upstream.index)
        upstream_columns = old_upstream.columns.union(new_upstream.columns)
        downstream_index = old_downstream.index.union(new_downstream.index)
        downstream_columns = old_downstream.columns.union(new_downstream.columns)

        old_upstream = old_upstream.reindex(index=upstream_index, columns=upstream_columns, fill_value=0)
        new_upstream = new_upstream.reindex(index=upstream_index, columns=upstream_columns, fill_value=0)
        old_downstream = old_downstream.reindex(index=downstream_index, columns=downstream_columns, fill_value=0)
        new_downstream = new_downstream.reindex(index=downstream_index, columns=downstream_columns, fill_value=0)

        return new_upstream - old_upstream, new_downstream - old_downstream

    def _ensure_feature_columns(self, features, labels):
        for label in labels:
            for direction in ["downstream", "upstream"]:
                key = (direction, label)
                if key not in features.columns:
                    features[key] = 0
        return features

    def _ensure_feature_index(self, features, index_values):
        if len(index_values) == 0:
            return features
        return features.reindex(features.index.union(index_values), fill_value=0)

    def _apply_feature_delta(self, features, delta_upstream, delta_downstream):
        missing_columns = []
        if not delta_upstream.empty:
            missing_columns.extend(
                [("upstream", label) for label in delta_upstream.columns]
            )
        if not delta_downstream.empty:
            missing_columns.extend(
                [("downstream", label) for label in delta_downstream.columns]
            )

        for col in missing_columns:
            if col not in features.columns:
                features[col] = 0

        if not delta_upstream.empty:
            upstream_cols = pd.MultiIndex.from_tuples(
                [("upstream", label) for label in delta_upstream.columns]
            )
            features.loc[delta_upstream.index, upstream_cols] = (
                features.loc[delta_upstream.index, upstream_cols].fillna(0).values
                + delta_upstream.values
            )

        if not delta_downstream.empty:
            downstream_cols = pd.MultiIndex.from_tuples(
                [("downstream", label) for label in delta_downstream.columns]
            )
            features.loc[delta_downstream.index, downstream_cols] = (
                features.loc[delta_downstream.index, downstream_cols].fillna(0).values
                + delta_downstream.values
            )

        return features

    def _prune_feature_columns(self, features, current_common_labels):
        keep = []
        for direction, label in features.columns:
            if label in current_common_labels:
                keep.append((direction, label))
        return features.loc[:, pd.MultiIndex.from_tuples(keep)]

    def _incremental_aggregate_connectivity(self, new_label_series, progress_dialog=None):
        if self._computed_label_series is None:
            return self._aggregate_connectivity(progress_dialog)

        changed_mask = ~self._computed_label_series.fillna("").eq(
            new_label_series.fillna("")
        )
        if not changed_mask.any():
            if progress_dialog is not None:
                progress_dialog.setLabelText("No label changes detected.")
                progress_dialog.setValue(5)
                QApplication.processEvents()
            return self.aggregated_features

        changed_nodes = set(
            self.meta.loc[changed_mask, ["id", "dataset"]]
            .apply(tuple, axis=1)
            .tolist()
        )
        if not changed_nodes:
            if progress_dialog is not None:
                progress_dialog.setLabelText("No relevant label changes detected.")
                progress_dialog.setValue(5)
                QApplication.processEvents()
            return self.aggregated_features

        old_common = self._computed_common_labels
        new_common = self._determine_common_labels(new_label_series)

        if progress_dialog is not None:
            progress_dialog.setLabelText("Preparing incremental aggregation...")
            progress_dialog.setValue(1)
            QApplication.processEvents()

        delta_upstream, delta_downstream = self._compute_feature_deltas(
            self._computed_label_series,
            new_label_series,
            old_common,
            new_common,
            changed_nodes,
            progress_dialog,
        )

        features = self.aggregated_features.fillna(0).astype(np.int64)
        features = self._ensure_feature_columns(features, old_common.union(new_common))
        features = self._ensure_feature_index(
            features, delta_upstream.index.union(delta_downstream.index)
        )
        features = self._apply_feature_delta(features, delta_upstream, delta_downstream)
        features = self._prune_feature_columns(features, new_common)
        features = features.clip(lower=0).astype(np.uint32)
        if not features.empty:
            features = features.loc[(features != 0).any(axis=1)]
            if not features.empty:
                features = features.loc[:, (features != 0).any(axis=0)]
            else:
                features = features.iloc[:, :0]
        else:
            features = features.iloc[:, :0]

        if progress_dialog is not None:
            progress_dialog.setLabelText("Finalizing incremental features...")
            progress_dialog.setValue(4)
            QApplication.processEvents()

        return features

    def _update_connectivity_button_style(self):
        button_style = (
            "background-color: #f0ad4e; color: #000000; "
            "border: 1px solid #d58512; border-radius: 4px; padding: 4px 10px; min-height: 12px;"
        )

        if self.meta is None and hasattr(self, "load_meta_btn"):
            self.load_meta_btn.setStyleSheet(button_style)
        elif hasattr(self, "load_meta_btn"):
            self.load_meta_btn.setStyleSheet("")

        if self.edge_list is None and hasattr(self, "load_connectivity_btn"):
            self.load_connectivity_btn.setStyleSheet(button_style)
        elif hasattr(self, "load_connectivity_btn"):
            self.load_connectivity_btn.setStyleSheet("")

    def _on_auto_compile_toggled(self, checked):
        if checked:
            self._schedule_auto_compute()
        elif hasattr(self, "_auto_compute_timer"):
            self._auto_compute_timer.stop()

    def _schedule_auto_compute(self):
        if not getattr(self, "auto_compile_checkbox", None):
            return
        if self.auto_compile_checkbox.isChecked() and self.compute_btn.isEnabled():
            self._auto_compute_timer.start()
        elif hasattr(self, "_auto_compute_timer"):
            self._auto_compute_timer.stop()

    def _auto_compute_if_enabled(self):
        if (
            self.auto_compile_checkbox.isChecked()
            and self.compute_btn.isEnabled()
            and not self._auto_compile_running
        ):
            self._auto_compile_running = True
            try:
                self._compute_features()
            finally:
                self._auto_compile_running = False

    def _sync_labels_from_meta(self):
        if self.meta is None or self._selected_meta_label_column is None:
            return
        column = self._selected_meta_label_column
        self.set_labels(
            self.meta[["id", "dataset", column]].rename(columns={column: "label"}),
            source=f"{column} (meta)"
        )
        self.data_status_label.setText(
            f"Labels synced from metadata column '{column}'."
        )

    def _settings_key_for_dataset(self, dataset):
        return f"labels/edge_file/{str(dataset).replace('/', '_')}"

    def _load_saved_connectivity_paths(self, datasets):
        saved = {}
        for dataset in datasets:
            value = self.settings.value(self._settings_key_for_dataset(dataset), "")
            if isinstance(value, str) and value.strip():
                saved[dataset] = value
        return saved

    def _save_connectivity_paths(self, files_by_dataset):
        for dataset, path in files_by_dataset.items():
            self.settings.setValue(self._settings_key_for_dataset(dataset), path)

    def _compute_features(self):
        progress = QProgressDialog(
            "Computing aggregated features...",
            None,
            0,
            5,
            self,
        )
        progress.setWindowTitle("Computing Features")
        progress.setWindowModality(QtCore.Qt.WindowModal)
        progress.setAutoClose(False)
        progress.setMinimumDuration(0)
        progress.setValue(0)

        label_series = self._lookup_labels()
        try:
            if (
                self.aggregated_features is not None
                and self._computed_label_series is not None
                and self._computed_join_mode == self.join_mode_combo.currentText()
            ):
                self.aggregated_features = self._incremental_aggregate_connectivity(
                    label_series, progress
                )
            else:
                self.aggregated_features = self._aggregate_connectivity(progress)
        except Exception as exc:
            progress.close()
            traceback.print_exc()
            QtWidgets.QMessageBox.critical(
                self,
                "Feature computation failed",
                f"Could not compute aggregated features:\n{type(exc).__name__}: {exc}",
            )
            return

        progress.close()
        self._computed_label_series = label_series.copy()
        self._computed_join_mode = self.join_mode_combo.currentText()
        self._computed_common_labels = self._determine_common_labels(label_series)
        self.features_status_label.setText(
            f"Computed aggregated features ({self.aggregated_features.shape[1]:,} columns)."
        )
        self._populate_features_table()
        self.features_computed.emit(self.aggregated_features, None)

    def _aggregate_connectivity(self, progress_dialog=None):
        if self.edge_list is None:
            raise ValueError("Connectivity not loaded.")
        if self.labels is None:
            raise ValueError("Labels not loaded.")

        label_series = self._lookup_labels()
        all_labels = pd.Index(label_series.dropna()).drop_duplicates().values
        inner_join = self.join_mode_combo.currentText().lower().startswith("inner")

        if progress_dialog is not None:
            progress_dialog.setLabelText("Preparing labels...")
            progress_dialog.setValue(0)
            QApplication.processEvents()

        if inner_join:
            dataset_labels = (
                pd.DataFrame({"dataset": self.meta["dataset"], "label": label_series})
                .dropna()
                .groupby(["dataset", "label"])
                .size()
                .unstack(fill_value=0)
            )
            common_labels = dataset_labels.columns[(dataset_labels > 0).all()]
        else:
            common_labels = all_labels

        if progress_dialog is not None:
            progress_dialog.setLabelText("Mapping labels onto edges...")
            progress_dialog.setValue(1)
            QApplication.processEvents()

        labels = (
            pd.DataFrame(
                {
                    "id": self.meta["id"],
                    "dataset": self.meta["dataset"],
                    "label": label_series,
                }
            )
            .set_index(["id", "dataset"])
            .label
        )

        labels = labels[labels.isin(common_labels)]

        edge_df = self._get_current_edge_list().copy()
        edge_df["pre_label"] = edge_df["pre"].map(labels)
        edge_df["post_label"] = edge_df["post"].map(labels)

        if progress_dialog is not None:
            progress_dialog.setLabelText("Grouping upstream edges...")
            progress_dialog.setValue(2)
            QApplication.processEvents()

        upstream = (
            edge_df[edge_df.pre_label.notna()]
            .groupby(["pre", "post_label"], sort=False)["weight"]
            .sum()
            .unstack(fill_value=0)
            .astype(np.uint32)
        )
        if progress_dialog is not None:
            progress_dialog.setLabelText("Grouping downstream edges...")
            progress_dialog.setValue(3)
            QApplication.processEvents()
        downstream = (
            edge_df[edge_df.post_label.notna()]
            .groupby(["post", "pre_label"], sort=False)["weight"]
            .sum()
            .unstack(fill_value=0)
            .astype(np.uint32)
        )

        if progress_dialog is not None:
            progress_dialog.setLabelText("Finalizing features...")
            progress_dialog.setValue(4)
            QApplication.processEvents()

        features = pd.concat(
            [downstream, upstream],
            axis=1,
            keys=["downstream", "upstream"],
        )
        return features

    def _populate_features_table(self):
        if self.aggregated_features is None or self.aggregated_features.empty:
            self.features_model.setDataFrame(pd.DataFrame())
            self.push_to_project_btn.setEnabled(False)
            self.export_to_parquet_btn.setEnabled(False)
            return

        self.features_model.setDataFrame(self.aggregated_features)
        self.push_to_project_btn.setEnabled(True)
        self.export_to_parquet_btn.setEnabled(True)

    def _schedule_update_feature_filters(self):
        if hasattr(self, "_filter_timer"):
            self._filter_timer.start()

    def _schedule_refresh_table(self):
        if hasattr(self, "_search_timer"):
            self._search_timer.start()

    def _update_label_filter(self):
        if hasattr(self, "labels_proxy") and hasattr(self, "search_edit"):
            self.labels_proxy.setRowFilter(self.search_edit.text())

    def _update_feature_filters(self):
        row_text = self.row_filter_edit.text() if hasattr(self, "row_filter_edit") else ""
        column_text = self.column_filter_edit.text() if hasattr(self, "column_filter_edit") else ""
        hide_zero = self.hide_zeros_checkbox.isChecked() if hasattr(self, "hide_zeros_checkbox") else False
        self.features_proxy.setRowFilter(row_text)
        self.features_proxy.setColumnFilter(column_text)
        self.features_proxy.setHideZeroRows(hide_zero)

    def _on_hide_zero_rows_toggled(self, checked):
        self._schedule_update_feature_filters()

    def _clear_features_table(self):
        self.features_model.setDataFrame(pd.DataFrame())
        if hasattr(self, "push_to_project_btn"):
            self.push_to_project_btn.setEnabled(False)
        if hasattr(self, "export_to_parquet_btn"):
            self.export_to_parquet_btn.setEnabled(False)

    def _push_to_project(self):
        QtWidgets.QMessageBox.information(
            self,
            "Push to Project",
            "Push to Project is not implemented yet.",
        )

    def _export_to_parquet(self):
        if self.aggregated_features is None or self.aggregated_features.empty:
            QtWidgets.QMessageBox.warning(
                self,
                "Export to Parquet",
                "No aggregated features are available to export.",
            )
            return

        path, _ = QFileDialog.getSaveFileName(
            self,
            "Export aggregated features to Parquet",
            str(Path.home() / "aggregated_features.parquet"),
            "Parquet files (*.parquet);;All files (*)",
        )
        if not path:
            return

        try:
            self.aggregated_features.to_parquet(path)
            QtWidgets.QMessageBox.information(
                self,
                "Export complete",
                f"Aggregated features exported to {path}",
            )
        except Exception as exc:
            QtWidgets.QMessageBox.critical(
                self,
                "Export failed",
                f"Could not export aggregated features: {exc}",
            )


if __name__ == "__main__":
    app = QApplication.instance() or QApplication(sys.argv)
    widget = LabelsWidget()
    widget.resize(900, 600)

    default_path = Path("/Users/philipps/Downloads/MaleCNS_MANC_bigclust/meta.parquet")
    if default_path.exists():
        try:
            meta = pd.read_parquet(default_path)
            widget.set_meta(meta)
            widget.labels_status_label.setText(
                f"Loaded metadata from {default_path.name} ({len(meta):,} rows)."
            )
        except Exception as exc:
            widget.labels_status_label.setText(f"Could not load default metadata: {exc}")
    else:
        widget.labels_status_label.setText(
            "No default metadata found. Use Load metadata to open a file."
        )

    widget.show()
    sys.exit(app.exec())
