from __future__ import annotations

import numpy as np
import pandas as pd

from pathlib import Path

from scipy import stats as _scipy_stats

from sklearn.metrics import roc_auc_score as _roc_auc_score
from sklearn.model_selection import StratifiedKFold as _StratifiedKFold
from sklearn.linear_model import LogisticRegression as _LogisticRegression
from sklearn.feature_selection import mutual_info_classif as _mutual_info_classif
from sklearn.inspection import permutation_importance as _permutation_importance

from PySide6 import QtCore, QtGui, QtWidgets


class _SortableNumberItem(QtWidgets.QTableWidgetItem):
    """Table item that sorts by numeric value stored in UserRole."""

    def __lt__(self, other):
        if isinstance(other, QtWidgets.QTableWidgetItem):
            left = self.data(QtCore.Qt.UserRole)
            right = other.data(QtCore.Qt.UserRole)
            if left is not None and right is not None:
                try:
                    return float(left) < float(right)
                except (TypeError, ValueError):
                    pass
        return super().__lt__(other)


class _OrderedTriStateCheckBox(QtWidgets.QCheckBox):
    """Tri-state checkbox cycling as off -> checked -> partially checked."""

    def nextCheckState(self):
        state = self.checkState()
        if state == QtCore.Qt.Unchecked:
            self.setCheckState(QtCore.Qt.Checked)
        elif state == QtCore.Qt.Checked:
            self.setCheckState(QtCore.Qt.PartiallyChecked)
        else:
            self.setCheckState(QtCore.Qt.Unchecked)


class _SelectIdsDialog(QtWidgets.QDialog):
    """Reusable list-selection dialog for metadata IDs."""

    def __init__(self, parent, meta, figure=None):
        super().__init__(parent)
        self.meta = meta
        self.figure = figure
        self._build_ui()
        self._populate_items()

    def _build_ui(self):
        self.resize(540, 460)
        root = QtWidgets.QVBoxLayout(self)

        self.search = QtWidgets.QLineEdit()
        self.search.setPlaceholderText("Filter by ID or label...")
        self.search.setToolTip(
            "Filter list entries by ID or label. Use comma-separated IDs for exact multi-ID filtering."
        )
        root.addWidget(self.search)

        self.id_list = QtWidgets.QListWidget()
        self.id_list.setSelectionMode(QtWidgets.QAbstractItemView.MultiSelection)
        self.id_list.setToolTip("Multi-select IDs. Cmd+A selects all visible IDs.")
        root.addWidget(self.id_list, stretch=1)

        tools_row = QtWidgets.QHBoxLayout()
        self.select_all_btn = QtWidgets.QPushButton("Select all")
        self.clear_selection_btn = QtWidgets.QPushButton("Clear selection")
        self.copy_figure_selection_btn = QtWidgets.QPushButton("Copy Figure Selection")
        self.select_all_btn.setToolTip("Select all currently visible rows.")
        self.clear_selection_btn.setToolTip("Clear the current selection.")
        self.copy_figure_selection_btn.setToolTip(
            "Copy selected IDs from parent figure into this selection."
        )
        tools_row.addWidget(self.select_all_btn)
        tools_row.addWidget(self.clear_selection_btn)
        tools_row.addWidget(self.copy_figure_selection_btn)
        tools_row.addStretch(1)
        root.addLayout(tools_row)

        self.status = QtWidgets.QLabel()
        self.status.setToolTip("Current number of selected IDs.")
        root.addWidget(self.status)

        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        shortcut_hint = QtWidgets.QLabel(
            "Shortcuts: Cmd+A select all, Cmd+Shift+A clear selection"
        )
        shortcut_hint.setStyleSheet("color: #777; font-size: 11px;")

        footer_row = QtWidgets.QHBoxLayout()
        footer_row.setContentsMargins(0, 0, 0, 0)
        footer_row.setSpacing(8)
        footer_row.addWidget(shortcut_hint)
        footer_row.addStretch(1)
        footer_row.addWidget(buttons)
        root.addLayout(footer_row)

        self.search.textChanged.connect(self._apply_filter)
        self.id_list.itemSelectionChanged.connect(self._update_status)
        self.select_all_btn.clicked.connect(self._select_all_visible)
        self.clear_selection_btn.clicked.connect(self._clear_selection)
        self.copy_figure_selection_btn.clicked.connect(self._copy_figure_selection)

        deselect_shortcut = QtGui.QShortcut(QtGui.QKeySequence("Meta+Shift+A"), self)
        deselect_shortcut.activated.connect(self._clear_selection)

        self._update_status()

    def _populate_items(self):
        self.id_list.clear()
        ids = self.meta["id"].values.astype(int)
        if "label" in self.meta.columns:
            labels = self.meta["label"].fillna("").values.astype(str)
        else:
            labels = [""] * len(ids)

        self._all_items = []
        self._item_ids = ids.tolist()

        for id_value, label_value in zip(ids, labels):
            text = f"{id_value}    {label_value}".rstrip()
            item = QtWidgets.QListWidgetItem(text)
            item.setData(QtCore.Qt.UserRole, id_value)
            self.id_list.addItem(item)
            self._all_items.append(item)

    def reset(self, title, preselected_ids):
        self.setWindowTitle(title)
        self.search.blockSignals(True)
        self.search.setText("")
        self.search.blockSignals(False)

        selected_set = {int(i) for i in preselected_ids}
        for item, item_id in zip(self._all_items, self._item_ids):
            item.setHidden(False)
            item.setSelected(item_id in selected_set)

        self._update_status()
        self.search.setFocus()

    def set_figure(self, figure):
        self.figure = figure

    def _update_status(self):
        self.status.setText(f"{len(self.id_list.selectedItems())} selected")

    def _apply_filter(self, text):
        query = text.strip().lower()
        token_list = [t.strip() for t in query.split(",") if t.strip()]
        is_multi_id_query = "," in query and len(token_list) > 0

        for i in range(self.id_list.count()):
            item = self.id_list.item(i)
            if not query:
                item.setHidden(False)
                continue

            if is_multi_id_query:
                item_id = str(item.data(QtCore.Qt.UserRole))
                item.setHidden(item_id not in token_list)
            else:
                item.setHidden(query not in item.text().lower())

    def _select_all_visible(self):
        for i in range(self.id_list.count()):
            item = self.id_list.item(i)
            if not item.isHidden():
                item.setSelected(True)

    def _clear_selection(self):
        self.id_list.clearSelection()

    def _copy_figure_selection(self):
        selected_ids = self.figure.selected_ids if self.figure is not None else None
        if selected_ids is None:
            QtWidgets.QMessageBox.information(
                self,
                "No figure selection",
                "Could not find parent.fig_scatter.selected_ids.",
            )
            return

        wanted = {int(i) for i in selected_ids}
        self.id_list.clearSelection()
        for i in range(self.id_list.count()):
            item = self.id_list.item(i)
            if int(item.data(QtCore.Qt.UserRole)) in wanted:
                item.setSelected(True)


class FeatureExplorerWidget(QtWidgets.QWidget):
    """Skeleton widget for prototyping feature ranking UI."""

    def __init__(
        self,
        metadata=None,
        group_a_name="Group A",
        group_b_name="Group B",
        group_a_ids=None,
        group_b_ids=None,
        features=None,
        figure=None,
        parent=None,
    ):
        super().__init__(parent)
        self.meta = (
            metadata.copy() if isinstance(metadata, pd.DataFrame) else pd.DataFrame()
        )
        if not self.meta.empty and "id" in self.meta.columns:
            self.meta = self.meta.dropna(subset=["id"]).copy()
            self.meta["id"] = self.meta["id"].astype("int64")

        raw_features = (
            features if isinstance(features, pd.DataFrame) else pd.DataFrame()
        )
        if not raw_features.empty and not self.meta.empty:
            candidate_ids = set(self.meta["id"].tolist())
            try:
                raw_features = _set_feature_index(raw_features, candidate_ids)
            except Exception:
                # Fall back to original table if index inference fails.
                pass

            if raw_features.index.isin(candidate_ids).sum() > 0:
                meta_order = pd.Index(self.meta["id"].tolist(), dtype="int64")
                raw_features = raw_features.loc[
                    raw_features.index.isin(candidate_ids)
                ].copy()
                raw_features = raw_features.reindex(
                    meta_order.intersection(raw_features.index)
                )

        self.features = self._drop_all_zero_feature_columns(raw_features)
        self.figure = figure
        self._cached_select_ids_dialog = None

        available_ids = set(self.features.index.tolist())
        self.group_a_ids = [
            int(i) for i in (group_a_ids or []) if int(i) in available_ids
        ]
        self.group_b_ids = [
            int(i) for i in (group_b_ids or []) if int(i) in available_ids
        ]

        self.group_a_name = str(group_a_name)
        self.group_b_name = str(group_b_name)
        self._perm_scoring_user_overridden = False
        self._updating_perm_scoring_default = False
        self._figure_sync_callbacks = {
            "a": lambda ids, datasets=None: self._apply_figure_selection_to_group(
                "a", ids, invert=self._group_sync_is_inverse("a")
            ),
            "b": lambda ids, datasets=None: self._apply_figure_selection_to_group(
                "b", ids, invert=self._group_sync_is_inverse("b")
            ),
        }

        self._distribution_dialogs: list = []
        self._n_hierarchy_levels = 0
        if isinstance(self.features.columns, pd.MultiIndex):
            self._n_hierarchy_levels = self.features.columns.nlevels - 1
        self._build_ui()

    def _build_ui(self):
        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(10)

        title = QtWidgets.QLabel("Connection Feature Ranking")
        title.setStyleSheet("font-size: 18px; font-weight: 600;")
        root.addWidget(title)

        self.subtitle = QtWidgets.QLabel(
            "Compare two neuron groups and rank connections that best" " separate them."
        )
        self.subtitle.setWordWrap(True)
        self.subtitle.setStyleSheet("color: #555;")
        root.addWidget(self.subtitle)

        top_row = QtWidgets.QHBoxLayout()
        top_row.setSpacing(10)
        group_box = self._make_group_box()
        group_box.setSizePolicy(
            QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Preferred
        )
        data_box = self._make_data_box()
        controls_box = self._make_controls_box()
        top_row.addWidget(group_box, stretch=0, alignment=QtCore.Qt.AlignTop)
        top_row.addWidget(data_box, stretch=0, alignment=QtCore.Qt.AlignTop)
        top_row.addWidget(controls_box, stretch=1, alignment=QtCore.Qt.AlignTop)
        root.addLayout(top_row)

        self.metric_warning_label = QtWidgets.QLabel()
        self.metric_warning_label.setWordWrap(True)
        self.metric_warning_label.setStyleSheet(
            "background-color: #fff3cd; color: #6b4f00; border: 1px solid #f0d98c; border-radius: 4px; padding: 6px 8px;"
        )
        self.metric_warning_label.hide()
        root.addWidget(self.metric_warning_label)

        filter_row = QtWidgets.QHBoxLayout()
        filter_row.setContentsMargins(0, 0, 0, 0)
        filter_row.setSpacing(6)
        filter_row.addStretch(1)
        self.feature_filter_edit = QtWidgets.QLineEdit()
        self.feature_filter_edit.setPlaceholderText("Filter features...")
        self.feature_filter_edit.setToolTip(
            "Filter visible rows by text in the Feature column."
        )
        self.feature_filter_edit.setClearButtonEnabled(True)
        self.feature_filter_edit.setMaximumWidth(220)
        self.feature_filter_edit.textChanged.connect(self._apply_feature_text_filter)
        filter_row.addWidget(self.feature_filter_edit)
        root.addLayout(filter_row)

        # Compute column layout based on MultiIndex hierarchy
        n_cols = 6 + self._n_hierarchy_levels
        self.table = QtWidgets.QTableWidget(0, n_cols)

        headers = ["Feature"]
        # Add hierarchy level headers (all but the finest grain)
        if self._n_hierarchy_levels > 0 and isinstance(
            self.features.columns, pd.MultiIndex
        ):
            for i in range(self._n_hierarchy_levels):
                level_name = self.features.columns.names[i]
                # Use a default name if level is unnamed
                if level_name is None:
                    level_name = f"Level {i + 1}"
                headers.append(str(level_name))
        # Add remaining standard headers
        headers.extend(
            [
                "Score (abs)",
                "Score (raw)",
                "Group A\n(mean +/- sd)",
                "Group B\n(mean +/- sd)",
                "Flags",
            ]
        )
        self.table.setHorizontalHeaderLabels(headers)
        header = self.table.horizontalHeader()
        header.setStretchLastSection(False)
        for col in range(self.table.columnCount()):
            header.setSectionResizeMode(
                col,
                QtWidgets.QHeaderView.Stretch
                if col == self._get_col_flags()
                else QtWidgets.QHeaderView.ResizeToContents,
            )
        header.setSortIndicator(self._get_col_score_abs(), QtCore.Qt.DescendingOrder)
        header.setSortIndicatorShown(True)
        self.table.verticalHeader().setVisible(False)
        self.table.setAlternatingRowColors(True)
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.table.setSortingEnabled(True)
        self.table.cellClicked.connect(self._on_feature_clicked)
        root.addWidget(self.table, stretch=1)

        self.table_hint_label = QtWidgets.QLabel("Hint: click a row to open the feature distribution plot.")
        self.table_hint_label.setStyleSheet("color: #777; font-size: 11px;")
        root.addWidget(self.table_hint_label)

        footer = QtWidgets.QHBoxLayout()
        footer.addWidget(QtWidgets.QLabel("Show Top N:"))
        self.top_n_spin = QtWidgets.QSpinBox()
        self.top_n_spin.setRange(0, 500)
        self.top_n_spin.setSpecialValueText("All")
        self.top_n_spin.setValue(25)
        footer.addWidget(self.top_n_spin)
        footer.addWidget(QtWidgets.QLabel("Top N by:"))

        self.top_n_mode_combo = QtWidgets.QComboBox()
        self.top_n_mode_combo.addItem("Absolute", userData="abs")
        self.top_n_mode_combo.addItem("Raw", userData="raw")
        self.top_n_mode_combo.setToolTip(
            "How Top N features are selected before table display."
        )
        footer.addWidget(self.top_n_mode_combo)

        self.keep_on_top_check = QtWidgets.QCheckBox("Always on top")
        self.keep_on_top_check.setChecked(False)
        self.keep_on_top_check.setToolTip("Keep this window above other windows.")
        self.keep_on_top_check.stateChanged.connect(self._toggle_keep_on_top)
        footer.addWidget(self.keep_on_top_check)

        footer.addStretch(1)

        self.status_label = QtWidgets.QLabel()
        self.status_label.setStyleSheet("color: #888;")
        self.status_label.hide()
        footer.addWidget(self.status_label)

        self.recompute_button = QtWidgets.QPushButton("Refresh")
        self.recompute_button.clicked.connect(self._populate_feature_rows)
        footer.addWidget(self.recompute_button)

        style = self.style()

        self.copy_btn = QtWidgets.QToolButton()
        self.copy_btn.setIcon(
            style.standardIcon(QtWidgets.QStyle.SP_FileDialogListView)
        )
        self.copy_btn.setToolTip("Copy table to clipboard")
        self.copy_btn.setAutoRaise(True)
        self.copy_btn.clicked.connect(self._copy_to_clipboard)
        footer.addWidget(self.copy_btn)

        self.export_btn = QtWidgets.QToolButton()
        self.export_btn.setIcon(
            style.standardIcon(QtWidgets.QStyle.SP_DialogSaveButton)
        )
        self.export_btn.setToolTip("Export table to CSV")
        self.export_btn.setAutoRaise(True)
        self.export_btn.clicked.connect(self._export_to_csv)
        footer.addWidget(self.export_btn)

        root.addLayout(footer)

        self.top_n_spin.valueChanged.connect(self._populate_feature_rows)
        self.top_n_mode_combo.currentIndexChanged.connect(self._populate_feature_rows)
        self._update_flags_column_visibility()
        self._update_group_summary()
        self._populate_feature_rows()

    def _make_group_box(self):
        box = QtWidgets.QGroupBox("Groups")
        layout = QtWidgets.QVBoxLayout(box)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(0)

        self.group_a_box = QtWidgets.QGroupBox("Group A")
        group_a_layout = QtWidgets.QVBoxLayout(self.group_a_box)
        group_a_layout.setContentsMargins(8, 8, 8, 8)
        group_a_layout.setSpacing(4)

        group_a_row = QtWidgets.QHBoxLayout()
        group_a_row.setContentsMargins(0, 0, 0, 0)
        group_a_row.setSpacing(6)

        self.group_a_count_label = QtWidgets.QLabel("0 selected")
        self.group_a_count_label.setStyleSheet("color: #666;")
        group_a_row.addWidget(self.group_a_count_label)
        group_a_row.addStretch(1)

        self.group_a_button = QtWidgets.QPushButton()
        self.group_a_button.setToolTip("Select IDs for Group A")
        self.group_a_button.clicked.connect(lambda: self._open_group_selection("a"))
        group_a_row.addWidget(self.group_a_button)
        group_a_layout.addLayout(group_a_row)
        if self.figure is not None:
            self.group_a_sync_check = _OrderedTriStateCheckBox("sync w/ figure")
            self.group_a_sync_check.setProperty("sync_target", "a")
            self.group_a_sync_check.setTristate(True)
            self.group_a_sync_check.setToolTip(
                "Unchecked: no sync. Checked: selected IDs. Partially checked: IDs NOT selected in figure."
            )
            self.group_a_sync_check.stateChanged.connect(
                self._on_group_sync_checkbox_changed
            )
            group_a_layout.addWidget(self.group_a_sync_check)
        layout.addWidget(self.group_a_box)

        self.swap_groups_btn = QtWidgets.QToolButton()
        self.swap_groups_btn.setIcon(
            self.style().standardIcon(QtWidgets.QStyle.SP_BrowserReload)
        )
        self.swap_groups_btn.setToolTip("Swap Group A and Group B")
        self.swap_groups_btn.setAutoRaise(True)
        self.swap_groups_btn.setFixedSize(26, 26)
        self.swap_groups_btn.setIconSize(QtCore.QSize(14, 14))
        self.swap_groups_btn.setStyleSheet(
            "QToolButton {"
            "background-color: palette(base);"
            "border: 1px solid #bdbdbd;"
            "border-radius: 13px;"
            "margin-top: -8px;"
            "margin-bottom: -8px;"
            "}"
            "QToolButton:hover { background-color: #f4f4f4; }"
            "QToolButton:pressed { background-color: #e9e9e9; }"
        )
        self.swap_groups_btn.clicked.connect(self._swap_groups)
        layout.addWidget(self.swap_groups_btn, alignment=QtCore.Qt.AlignHCenter)

        self.group_b_box = QtWidgets.QGroupBox("Group B")
        group_b_layout = QtWidgets.QVBoxLayout(self.group_b_box)
        group_b_layout.setContentsMargins(8, 8, 8, 8)
        group_b_layout.setSpacing(4)

        group_b_row = QtWidgets.QHBoxLayout()
        group_b_row.setContentsMargins(0, 0, 0, 0)
        group_b_row.setSpacing(6)

        self.group_b_count_label = QtWidgets.QLabel("0 selected")
        self.group_b_count_label.setStyleSheet("color: #666;")
        group_b_row.addWidget(self.group_b_count_label)
        group_b_row.addStretch(1)

        self.group_b_button = QtWidgets.QPushButton()
        self.group_b_button.setToolTip("Select IDs for Group B")
        self.group_b_button.clicked.connect(lambda: self._open_group_selection("b"))
        group_b_row.addWidget(self.group_b_button)
        group_b_layout.addLayout(group_b_row)
        if self.figure is not None:
            self.group_b_sync_check = _OrderedTriStateCheckBox("sync w/ figure")
            self.group_b_sync_check.setProperty("sync_target", "b")
            self.group_b_sync_check.setTristate(True)
            self.group_b_sync_check.setToolTip(
                "Unchecked: no sync. Checked: selected IDs. Partially checked: IDs NOT selected in figure."
            )
            self.group_b_sync_check.stateChanged.connect(
                self._on_group_sync_checkbox_changed
            )
            group_b_layout.addWidget(self.group_b_sync_check)
        layout.addWidget(self.group_b_box)

        self.group_feature_count_label = QtWidgets.QLabel()
        self.group_feature_count_label.setStyleSheet("color: #777; font-size: 11px;")
        self.group_feature_count_label.setToolTip(
            "Numeric feature columns surviving the current top-level and minimum-value filters for the selected groups."
        )
        layout.addWidget(self.group_feature_count_label)

        return box

    def _count_group_filtered_features(self):
        """Count features that survive the current scoring filters for the selected groups."""
        numeric = self._get_numeric_feature_table()
        total_numeric = int(numeric.shape[1])
        if total_numeric == 0:
            return 0, 0

        if not self.group_a_ids or not self.group_b_ids:
            return 0, total_numeric

        group_a = self._get_group_features(self.group_a_ids)
        group_b = self._get_group_features(self.group_b_ids)
        if group_a.empty or group_b.empty:
            return 0, total_numeric

        kept_cols = self._get_feature_columns_passing_threshold(group_a, group_b)
        return int(len(kept_cols)), total_numeric

    def _update_group_feature_count_label(self, kept_n=None, total_n=None):
        """Update the feature-count label using the current scoring filters."""
        if not hasattr(self, "group_feature_count_label"):
            return

        if len(self.group_a_ids) == 0 or len(self.group_b_ids) == 0:
            self.group_feature_count_label.setText("")
            return

        if kept_n is None or total_n is None:
            kept_n, total_n = self._count_group_filtered_features()

        self.group_feature_count_label.setText(f"Feature count: {kept_n}/{total_n}")

    def _make_data_box(self):
        box = QtWidgets.QGroupBox("Data")
        box.setSizePolicy(
            QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Preferred
        )
        box.setToolTip("Choose how features are filtered, normalized, and summarized before scoring.")
        layout = QtWidgets.QFormLayout(box)

        self.aggregation_combo = QtWidgets.QComboBox()
        self.aggregation_combo.addItems(
            ["Mean ± SD", "Median ± IQR", "Trimmed mean (10%) ± SD", "Max"]
        )
        self.aggregation_combo.setToolTip("Statistic used to summarize each group per feature.")
        self.aggregation_combo.currentIndexChanged.connect(self._populate_feature_rows)
        layout.addRow("Aggregation", self.aggregation_combo)

        self.top_level_checks = {}
        self.top_level_filter_widget = QtWidgets.QWidget()
        self.top_level_filter_widget.setToolTip("Filter MultiIndex features by top-level category.")
        top_level_filter_layout = QtWidgets.QVBoxLayout(self.top_level_filter_widget)
        top_level_filter_layout.setContentsMargins(0, 0, 0, 0)
        top_level_filter_layout.setSpacing(4)

        if isinstance(self.features.columns, pd.MultiIndex):
            top_levels = self.features.columns.get_level_values(0).unique().tolist()
            for level_name in top_levels:
                check = QtWidgets.QCheckBox(str(level_name))
                check.setChecked(True)
                check.setToolTip(f"Include features from top-level group '{level_name}'.")
                check.stateChanged.connect(self._populate_feature_rows)
                top_level_filter_layout.addWidget(check)
                self.top_level_checks[level_name] = check
        else:
            label = QtWidgets.QLabel("Not available (single-level columns)")
            label.setStyleSheet("color: #666;")
            top_level_filter_layout.addWidget(label)

        layout.addRow("Top-level filters", self.top_level_filter_widget)

        self.min_count_spin = QtWidgets.QSpinBox()
        self.min_count_spin.setRange(0, 10000)
        self.min_count_spin.setValue(3)
        self.min_count_spin.setToolTip("Drop features whose values are below this threshold in both groups.")
        self.min_count_spin.valueChanged.connect(self._populate_feature_rows)

        self.min_count_spin_norm = QtWidgets.QDoubleSpinBox()
        self.min_count_spin_norm.setRange(0.0, 1.0)
        self.min_count_spin_norm.setDecimals(3)
        self.min_count_spin_norm.setSingleStep(0.001)
        self.min_count_spin_norm.setValue(0.005)
        self.min_count_spin_norm.setToolTip("Drop normalized features below this threshold in both groups.")
        self.min_count_spin_norm.valueChanged.connect(self._populate_feature_rows)

        self.min_count_label = QtWidgets.QLabel("Min synapse count")
        self.min_count_label.setToolTip("Active minimum threshold used to filter low-value features.")
        self.min_count_stack = QtWidgets.QStackedWidget()
        self.min_count_stack.addWidget(self.min_count_spin)
        self.min_count_stack.addWidget(self.min_count_spin_norm)
        self.min_count_spin.setSizePolicy(
            QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed
        )
        self.min_count_spin_norm.setSizePolicy(
            QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed
        )
        self.min_count_stack.setSizePolicy(
            QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed
        )
        self.min_count_stack.setToolTip("Threshold control switches with normalization mode.")
        layout.addRow(self.min_count_label, self.min_count_stack)

        self.normalize_check = QtWidgets.QCheckBox("Normalize per neuron")
        self.normalize_check.setChecked(False)
        self.normalize_check.setToolTip("Normalize each neuron's feature vector before scoring.")
        self.normalize_check.stateChanged.connect(self._populate_feature_rows)
        self.normalize_check.stateChanged.connect(
            self._update_min_count_control_visibility
        )
        layout.addRow("", self.normalize_check)
        self._update_min_count_control_visibility()

        return box

    def _make_controls_box(self):
        box = QtWidgets.QGroupBox("Scoring")
        box.setToolTip("Choose ranking metric and method-specific scoring options.")
        layout = QtWidgets.QFormLayout(box)

        self.metric_combo = QtWidgets.QComboBox()
        self.metric_combo.addItems(
            [
                "Mean difference",
                "Permutation Importance",
                "Logistic (L1)",
                "Hurdle score",
                "Rank-biserial correlation",
                "Mutual information",
                "AUC (single-feature)",
                "Brunner-Munzel",
                "T-test (Welch's)",
                "Effect size (Cohen d)",
                "One-way ANOVA",
            ]
        )
        self.metric_combo.currentIndexChanged.connect(self._populate_feature_rows)
        self.metric_combo.currentIndexChanged.connect(self._update_metric_help_tooltip)
        self.metric_combo.currentIndexChanged.connect(
            self._update_metric_specific_controls
        )
        self.metric_combo.currentIndexChanged.connect(
            self._update_flags_column_visibility
        )

        self.metric_help_btn = QtWidgets.QToolButton()
        self.metric_help_btn.setText("?")
        self.metric_help_btn.setAutoRaise(True)
        self.metric_help_btn.clicked.connect(self._show_metric_help)

        metric_row = QtWidgets.QWidget()
        metric_row_layout = QtWidgets.QHBoxLayout(metric_row)
        metric_row_layout.setContentsMargins(0, 0, 0, 0)
        metric_row_layout.setSpacing(4)
        metric_row_layout.addWidget(self.metric_combo, stretch=1)
        metric_row_layout.addWidget(self.metric_help_btn)
        layout.addRow("Metric", metric_row)

        self.l1_settings_widget = QtWidgets.QWidget()
        self.l1_settings_widget.setToolTip("Options for Logistic (L1) scoring.")
        l1_layout = QtWidgets.QFormLayout(self.l1_settings_widget)
        l1_layout.setContentsMargins(0, 0, 0, 0)
        l1_layout.setVerticalSpacing(4)

        self.l1_c_spin = QtWidgets.QDoubleSpinBox()
        self.l1_c_spin.setRange(0.001, 1000.0)
        self.l1_c_spin.setDecimals(3)
        self.l1_c_spin.setSingleStep(0.1)
        self.l1_c_spin.setValue(1.0)
        self.l1_c_spin.setToolTip(
            "Inverse regularization strength. Smaller C means stronger sparsity."
        )
        self.l1_c_spin.valueChanged.connect(self._populate_feature_rows)

        self.l1_solver_combo = QtWidgets.QComboBox()
        self.l1_solver_combo.addItems(["liblinear", "saga"])
        self.l1_solver_combo.setToolTip(
            "Optimization algorithm for L1 logistic regression."
        )
        self.l1_solver_combo.currentIndexChanged.connect(self._populate_feature_rows)

        self.l1_max_iter_spin = QtWidgets.QSpinBox()
        self.l1_max_iter_spin.setRange(100, 50000)
        self.l1_max_iter_spin.setSingleStep(100)
        self.l1_max_iter_spin.setValue(3000)
        self.l1_max_iter_spin.setToolTip(
            "Maximum optimization iterations before stopping."
        )
        self.l1_max_iter_spin.valueChanged.connect(self._populate_feature_rows)

        self.l1_standardize_check = QtWidgets.QCheckBox("Standardize features")
        self.l1_standardize_check.setChecked(True)
        self.l1_standardize_check.setToolTip(
            "Z-score features before fitting so coefficients are comparable."
        )
        self.l1_standardize_check.stateChanged.connect(self._populate_feature_rows)

        # Compact two-column layout for core Logistic (L1) controls.
        l1_core_widget = QtWidgets.QWidget()
        l1_core_grid = QtWidgets.QGridLayout(l1_core_widget)
        l1_core_grid.setContentsMargins(0, 0, 0, 0)
        l1_core_grid.setHorizontalSpacing(8)
        l1_core_grid.setVerticalSpacing(4)
        l1_core_grid.addWidget(QtWidgets.QLabel("L1 C"), 0, 0)
        l1_core_grid.addWidget(self.l1_c_spin, 0, 1)
        l1_core_grid.addWidget(QtWidgets.QLabel("Solver"), 0, 2)
        l1_core_grid.addWidget(self.l1_solver_combo, 0, 3)
        l1_core_grid.addWidget(QtWidgets.QLabel("Max iter"), 1, 0)
        l1_core_grid.addWidget(self.l1_max_iter_spin, 1, 1)
        l1_core_grid.addWidget(self.l1_standardize_check, 1, 2, 1, 2)
        l1_layout.addRow("", l1_core_widget)

        self.l1_stability_check = QtWidgets.QCheckBox("Use stability selection")
        self.l1_stability_check.setChecked(True)
        self.l1_stability_check.setToolTip(
            "Use bootstrap L1 refits and rank by signed selection frequency."
        )
        self.l1_stability_check.stateChanged.connect(self._populate_feature_rows)
        self.l1_stability_check.stateChanged.connect(self._update_l1_stability_controls)
        l1_layout.addRow("", self.l1_stability_check)

        self.l1_stability_options_widget = QtWidgets.QWidget()
        l1_stability_layout = QtWidgets.QFormLayout(self.l1_stability_options_widget)
        l1_stability_layout.setContentsMargins(0, 0, 0, 0)

        self.l1_stability_bootstraps_spin = QtWidgets.QSpinBox()
        self.l1_stability_bootstraps_spin.setRange(10, 2000)
        self.l1_stability_bootstraps_spin.setSingleStep(10)
        self.l1_stability_bootstraps_spin.setValue(200)
        self.l1_stability_bootstraps_spin.setToolTip(
            "Number of bootstrap refits used to estimate selection frequency."
        )
        self.l1_stability_bootstraps_spin.valueChanged.connect(
            self._populate_feature_rows
        )
        l1_stability_layout.addRow("Bootstraps", self.l1_stability_bootstraps_spin)

        self.l1_stability_subsample_spin = QtWidgets.QDoubleSpinBox()
        self.l1_stability_subsample_spin.setRange(0.3, 1.0)
        self.l1_stability_subsample_spin.setDecimals(2)
        self.l1_stability_subsample_spin.setSingleStep(0.05)
        self.l1_stability_subsample_spin.setValue(0.75)
        self.l1_stability_subsample_spin.setToolTip(
            "Per-class bootstrap sample fraction for each refit."
        )
        self.l1_stability_subsample_spin.valueChanged.connect(
            self._populate_feature_rows
        )
        l1_stability_layout.addRow("Sample frac", self.l1_stability_subsample_spin)

        l1_layout.addRow("", self.l1_stability_options_widget)

        layout.addRow("", self.l1_settings_widget)

        self.perm_settings_widget = QtWidgets.QWidget()
        self.perm_settings_widget.setToolTip("Options for Permutation Importance scoring.")
        perm_layout = QtWidgets.QFormLayout(self.perm_settings_widget)
        perm_layout.setContentsMargins(0, 0, 0, 0)

        self.perm_repeats_spin = QtWidgets.QSpinBox()
        self.perm_repeats_spin.setRange(1, 100)
        self.perm_repeats_spin.setValue(5)
        self.perm_repeats_spin.setToolTip(
            "Number of permutations per feature. More repeats improve stability but take longer."
        )
        self.perm_repeats_spin.valueChanged.connect(self._populate_feature_rows)
        perm_layout.addRow("Repeats", self.perm_repeats_spin)

        self.perm_scoring_combo = QtWidgets.QComboBox()
        self.perm_scoring_combo.addItems(
            ["neg_log_loss", "roc_auc", "accuracy", "average_precision"]
        )
        self.perm_scoring_combo.setToolTip(
            "Scoring function used to measure performance drop after permutation."
        )
        self.perm_scoring_combo.currentIndexChanged.connect(
            self._on_perm_scoring_changed
        )
        perm_layout.addRow("Scoring", self.perm_scoring_combo)

        self.perm_eval_combo = QtWidgets.QComboBox()
        self.perm_eval_combo.addItems(["In-sample", "3-fold CV", "5-fold CV"])
        self.perm_eval_combo.setCurrentText("In-sample")
        self.perm_eval_combo.setToolTip(
            "Compute importances on training data or held-out CV folds."
        )
        self.perm_eval_combo.currentIndexChanged.connect(self._populate_feature_rows)
        perm_layout.addRow("Evaluation", self.perm_eval_combo)

        layout.addRow("", self.perm_settings_widget)

        self._apply_permutation_scoring_default(force=True)
        self._update_metric_help_tooltip()
        self._set_metric_item_tooltips()
        self._update_metric_specific_controls()
        self._update_l1_stability_controls()
        return box

    _METRIC_DESCRIPTIONS = {
        "Mean difference": (
            "Center difference between the two groups using the selected aggregation method.\n"
            "Makes no distributional assumptions — suitable for count data."
        ),
        "Rank-biserial correlation": (
            "Non-parametric effect size from the Mann-Whitney U test.\n"
            "r = (U1 - U2) / (n_A x n_B), ranges from -1 to +1.\n"
            "Positive = Group A stochastically larger; negative = Group B larger.\n"
            "Makes no distributional assumptions — recommended for count data."
        ),
        "Brunner-Munzel": (
            "Non-parametric two-sample test robust to unequal variances/distribution shapes.\n"
            "Score is the Brunner-Munzel W statistic (signed by group ordering).\n"
            "No normality or homoscedasticity assumptions."
        ),
        "Hurdle score": (
            "Two-part sparse-count score: (presence difference) + (log-median difference among non-zeros).\n"
            "Captures both occurrence shifts and non-zero magnitude shifts.\n"
            "Heuristic, assumption-light metric tailored to zero-inflated counts."
        ),
        "AUC (single-feature)": (
            "ROC AUC using one feature at a time to classify group label.\n"
            "Internally centered as (AUC - 0.5), so score ranges from -0.5 to +0.5.\n"
            "No distributional assumptions; monotonic-separation metric."
        ),
        "Logistic (L1)": (
            "Sparse multivariate classifier using L1-regularized logistic regression.\n"
            "Default score is signed feature coefficient (larger magnitude = more important).\n"
            "Enable 'Use stability selection' to rank by signed bootstrap selection frequency.\n"
            "Settings in Scoring control regularization strength (C), solver, and iterations."
        ),
        "Permutation Importance": (
            "Model-based importance from performance drop after randomly permuting each feature.\n"
            "Uses a logistic classifier and a user-selectable scoring objective.\n"
            "For small groups the default scoring is neg_log_loss, which is usually less degenerate than ROC AUC.\n"
            "Displayed score is signed by (center_A - center_B); magnitude reflects importance."
        ),
        "Effect size (Cohen d)": (
            "Standardised mean difference: (mean_A - mean_B) / pooled SD.\n"
            "Assumes normally distributed data with equal variances (homoscedasticity).\n"
            "Violations are flagged per feature in the Flags column.\n"
            "Note: synapse count data typically violates both assumptions."
        ),
        "T-test (Welch's)": (
            "Welch's two-sample t-test statistic (signed: positive = Group A > Group B).\n"
            "Does not assume equal variances — robust to heteroscedasticity.\n"
            "Assumes normally distributed data; violations flagged as !normal."
        ),
        "One-way ANOVA": (
            "F-statistic from a one-way ANOVA across the two groups.\n"
            "With two groups, F = t^2, so ranking is equivalent to Welch's t-test.\n"
            "Score is always positive (no direction).\n"
            "Assumes normality and equal variances; violations flagged."
        ),
        "Mutual information": (
            "Mutual information between each feature and the group label.\n"
            "Estimated via k-NN (sklearn mutual_info_classif).\n"
            "Always non-negative; higher = more informative. No direction.\n"
            "Non-parametric — no distributional assumptions. Can capture non-linear relationships."
        ),
    }

    # Maps metric name → which assumption tests to run.
    _METRICS_WITH_ASSUMPTIONS: dict[str, dict] = {
        "Effect size (Cohen d)": {"check_normality": True, "check_equal_var": True},
        "T-test (Welch's)": {"check_normality": True, "check_equal_var": False},
        "One-way ANOVA": {"check_normality": True, "check_equal_var": True},
    }

    def _get_col_feature(self):
        """Get column index for Feature."""
        return 0

    def _get_col_score_abs(self):
        """Get column index for Score (abs)."""
        return self._n_hierarchy_levels + 1

    def _get_col_score_raw(self):
        """Get column index for Score (raw)."""
        return self._n_hierarchy_levels + 2

    def _get_col_group_a(self):
        """Get column index for Group A."""
        return self._n_hierarchy_levels + 3

    def _get_col_group_b(self):
        """Get column index for Group B."""
        return self._n_hierarchy_levels + 4

    def _get_col_flags(self):
        """Get column index for Flags."""
        return self._n_hierarchy_levels + 5

    def _update_metric_help_tooltip(self):
        desc = self._METRIC_DESCRIPTIONS.get(self.metric_combo.currentText(), "")
        self.metric_combo.setToolTip(desc)
        self.metric_help_btn.setToolTip(desc)

    def _set_metric_item_tooltips(self):
        """Set dropdown hover tooltips for each metric option."""
        for i in range(self.metric_combo.count()):
            metric_name = self.metric_combo.itemText(i)
            desc = self._METRIC_DESCRIPTIONS.get(metric_name, "")
            self.metric_combo.setItemData(i, desc, QtCore.Qt.ToolTipRole)

    def _show_metric_help(self):
        desc = self._METRIC_DESCRIPTIONS.get(self.metric_combo.currentText(), "")
        if desc:
            QtWidgets.QToolTip.showText(
                self.metric_help_btn.mapToGlobal(
                    QtCore.QPoint(0, self.metric_help_btn.height())
                ),
                desc,
                self.metric_help_btn,
            )

    def _update_metric_specific_controls(self):
        """Show method-specific controls only for the active metric."""
        metric = self.metric_combo.currentText()
        is_l1 = metric == "Logistic (L1)"
        is_perm = metric == "Permutation Importance"
        self.l1_settings_widget.setVisible(is_l1)
        self.perm_settings_widget.setVisible(is_perm)

    def _preferred_permutation_scoring(self):
        """Return the default permutation scoring for the current group sizes."""
        if not self.group_a_ids or not self.group_b_ids:
            return "roc_auc"

        min_group_n = min(len(self.group_a_ids), len(self.group_b_ids))
        if min_group_n <= 20:
            return "neg_log_loss"
        return "roc_auc"

    def _apply_permutation_scoring_default(self, force=False):
        """Apply the preferred permutation scoring unless the user chose one explicitly."""
        if not hasattr(self, "perm_scoring_combo"):
            return
        if self._perm_scoring_user_overridden and not force:
            return

        preferred = self._preferred_permutation_scoring()
        if self.perm_scoring_combo.currentText() == preferred:
            return

        self._updating_perm_scoring_default = True
        try:
            self.perm_scoring_combo.setCurrentText(preferred)
        finally:
            self._updating_perm_scoring_default = False

    def _on_perm_scoring_changed(self):
        """Persist user scoring choices while still allowing automatic defaults before override."""
        if not self._updating_perm_scoring_default:
            self._perm_scoring_user_overridden = True
        if not hasattr(self, "status_label"):
            return
        self._populate_feature_rows()

    def _update_l1_stability_controls(self):
        """Show stability-specific L1 controls only when enabled."""
        is_enabled = self.l1_stability_check.isChecked()
        self.l1_stability_options_widget.setVisible(is_enabled)

    def _update_flags_column_visibility(self):
        """Show Flags only for metrics that surface assumption checks."""
        has_flags = self.metric_combo.currentText() in self._METRICS_WITH_ASSUMPTIONS
        self.table.setColumnHidden(self._get_col_flags(), not has_flags)
        header = self.table.horizontalHeader()
        for col in range(self.table.columnCount()):
            header.setSectionResizeMode(
                col,
                QtWidgets.QHeaderView.Stretch
                if col == self._get_col_flags()
                else QtWidgets.QHeaderView.ResizeToContents,
            )

    def _update_group_summary(self):
        group_a_n = len(self.group_a_ids)
        group_b_n = len(self.group_b_ids)

        self._apply_permutation_scoring_default()

        if hasattr(self, "group_a_box"):
            self.group_a_box.setTitle(f"Group A: {self.group_a_name}")
            a_color = "#d9534f" if group_a_n == 0 else "#2e8b57"
            self._apply_group_box_state_style(self.group_a_box, a_color)
        if hasattr(self, "group_b_box"):
            self.group_b_box.setTitle(f"Group B: {self.group_b_name}")
            b_color = "#d9534f" if group_b_n == 0 else "#2e8b57"
            self._apply_group_box_state_style(self.group_b_box, b_color)

        if hasattr(self, "group_a_button"):
            self.group_a_button.setText("Select ..." if group_a_n == 0 else "Edit")
        if hasattr(self, "group_a_count_label"):
            self.group_a_count_label.setText(f"{group_a_n} selected")

        self._update_group_feature_count_label()

        if hasattr(self, "group_b_button"):
            self.group_b_button.setText("Select ..." if group_b_n == 0 else "Edit")
        if hasattr(self, "group_b_count_label"):
            self.group_b_count_label.setText(f"{group_b_n} selected")

        feature_shape = self.features.shape
        self.subtitle.setText(
            "Compare two neuron groups and rank connections that best "
            f"separate them. Full feature matrix: {feature_shape[0]} x {feature_shape[1]}."
        )

    def _swap_groups(self):
        """Swap group IDs and display names, then refresh computed rankings."""
        self.group_a_ids, self.group_b_ids = self.group_b_ids, self.group_a_ids
        self.group_a_name, self.group_b_name = self.group_b_name, self.group_a_name

        if hasattr(self, "group_a_sync_check") and hasattr(self, "group_b_sync_check"):
            a_state = self.group_a_sync_check.checkState()
            b_state = self.group_b_sync_check.checkState()

            self.group_a_sync_check.blockSignals(True)
            self.group_b_sync_check.blockSignals(True)
            self.group_a_sync_check.setCheckState(b_state)
            self.group_b_sync_check.setCheckState(a_state)
            self.group_a_sync_check.blockSignals(False)
            self.group_b_sync_check.blockSignals(False)
            self._refresh_figure_sync_callbacks()

        self._update_group_summary()
        self._populate_feature_rows()

    def _group_name_from_ids(self, ids, fallback):
        """Infer group display name from selected IDs and metadata labels."""
        if not ids or self.meta.empty or "label" not in self.meta.columns:
            return fallback

        selected_labels = (
            self.meta.loc[self.meta["id"].isin(ids), "label"]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )
        if not selected_labels:
            return fallback
        if len(selected_labels) == 1:
            return selected_labels[0]
        return f"Mixed ({len(selected_labels)} labels)"

    def _open_group_selection(self, target):
        """Open an ID selection dialog for one group."""
        current_ids = self.group_a_ids if target == "a" else self.group_b_ids
        title = "Select Group A IDs" if target == "a" else "Select Group B IDs"
        selected_ids = self._select_ids_dialog(title=title, preselected_ids=current_ids)
        if selected_ids is None:
            return

        if target == "a":
            self.group_a_ids = selected_ids
            self.group_a_name = self._group_name_from_ids(selected_ids, "Group A")
        else:
            self.group_b_ids = selected_ids
            self.group_b_name = self._group_name_from_ids(selected_ids, "Group B")

        self._update_group_summary()
        self._populate_feature_rows()

    def _on_group_sync_checkbox_changed(self, _state):
        """Register or unregister figure callbacks based on checkbox states."""
        sender = self.sender()
        target = sender.property("sync_target") if sender is not None else None
        is_enabled = (
            sender.checkState() != QtCore.Qt.Unchecked
            if isinstance(sender, QtWidgets.QCheckBox)
            else False
        )

        self._refresh_figure_sync_callbacks()

        # On initial check, apply current figure selection immediately.
        if (
            is_enabled
            and target in ("a", "b")
            and self.figure is not None
            and hasattr(self.figure, "selected_ids")
        ):
            self._apply_figure_selection_to_group(
                target,
                self.figure.selected_ids,
                invert=self._group_sync_is_inverse(target),
            )

    def _group_sync_is_inverse(self, target):
        """Return whether a group's sync mode is inverse (not selected)."""
        check = (
            self.group_a_sync_check
            if target == "a"
            else self.group_b_sync_check
            if target == "b"
            else None
        )
        if check is None:
            return False
        return check.checkState() == QtCore.Qt.PartiallyChecked

    def _refresh_figure_sync_callbacks(self):
        """Sync current checkbox state to figure widget callbacks."""
        if self.figure is None:
            return
        if not hasattr(self.figure, "sync_widget") or not hasattr(
            self.figure, "unsync_widget"
        ):
            return

        # Reset callbacks for this widget, then re-register active ones.
        self.figure.unsync_widget(self)

        if (
            getattr(self, "group_a_sync_check", None) is not None
            and self.group_a_sync_check.checkState() != QtCore.Qt.Unchecked
        ):
            self.figure.sync_widget(self, callback=self._figure_sync_callbacks["a"])
        if (
            getattr(self, "group_b_sync_check", None) is not None
            and self.group_b_sync_check.checkState() != QtCore.Qt.Unchecked
        ):
            self.figure.sync_widget(self, callback=self._figure_sync_callbacks["b"])

    def _apply_figure_selection_to_group(self, target, selected_ids, invert=False):
        """Apply figure selection IDs to one group and refresh results."""
        available_ids = set(self.features.index.tolist())
        selected_set = (
            {int(i) for i in selected_ids if int(i) in available_ids}
            if selected_ids is not None
            else set()
        )
        if invert:
            new_ids = sorted(available_ids - selected_set)
        else:
            new_ids = sorted(selected_set)

        if target == "a":
            if new_ids == self.group_a_ids:
                return
            self.group_a_ids = new_ids
            self.group_a_name = self._group_name_from_ids(new_ids, "Group A")
        else:
            if new_ids == self.group_b_ids:
                return
            self.group_b_ids = new_ids
            self.group_b_name = self._group_name_from_ids(new_ids, "Group B")

        self._update_group_summary()
        self._populate_feature_rows()

    def _select_ids_dialog(self, title, preselected_ids):
        """Show a modal dialog to pick IDs from metadata."""
        if self.meta.empty or "id" not in self.meta.columns:
            QtWidgets.QMessageBox.warning(
                self, "No metadata", "Metadata with an 'id' column is required."
            )
            return None

        if self._cached_select_ids_dialog is None:
            self._cached_select_ids_dialog = _SelectIdsDialog(
                self, self.meta, figure=self.figure
            )
        else:
            self._cached_select_ids_dialog.set_figure(self.figure)

        self._cached_select_ids_dialog.reset(title, preselected_ids)
        if self._cached_select_ids_dialog.exec() != QtWidgets.QDialog.Accepted:
            return None

        return [
            int(item.data(QtCore.Qt.UserRole))
            for item in self._cached_select_ids_dialog.id_list.selectedItems()
        ]

    def _drop_all_zero_feature_columns(self, features):
        """Drop numeric columns that are zero for every row."""
        if features.empty:
            return features

        numeric_cols = features.select_dtypes(include="number").columns
        if len(numeric_cols) == 0:
            return features

        all_zero_numeric = (features.loc[:, numeric_cols].fillna(0) == 0).all(axis=0)
        if not all_zero_numeric.any():
            return features

        drop_cols = list(all_zero_numeric[all_zero_numeric].index)
        return features.drop(columns=drop_cols)

    def _update_min_count_control_visibility(self):
        """Show the threshold control relevant to current normalization mode."""
        use_normalized = self.normalize_check.isChecked()
        if use_normalized:
            self.min_count_stack.setCurrentWidget(self.min_count_spin_norm)
            self.min_count_label.setText("Min normalized value")
        else:
            self.min_count_stack.setCurrentWidget(self.min_count_spin)
            self.min_count_label.setText("Min synapse count")

    def _toggle_keep_on_top(self, state):
        """Toggle top-level window always-on-top behavior."""
        win = self.window()
        if win is None:
            return
        keep_on_top = bool(state)
        win.setWindowFlag(QtCore.Qt.WindowStaysOnTopHint, keep_on_top)
        # Re-show to ensure window flag changes take effect on all platforms.
        win.show()

    def _populate_feature_rows(self):
        self.status_label.setText("⏳ Computing…")
        self.status_label.show()
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.BusyCursor)
        QtWidgets.QApplication.processEvents()
        try:
            self._do_populate_feature_rows()
        finally:
            self.status_label.hide()
            QtWidgets.QApplication.restoreOverrideCursor()

    def _do_populate_feature_rows(self):
        was_sorting_enabled = self.table.isSortingEnabled()
        if was_sorting_enabled:
            self.table.setSortingEnabled(False)

        try:
            self._set_metric_warning(None)
            numeric_features = self.features.select_dtypes(include="number")
            numeric_features = self._filter_by_top_level(numeric_features)

            if numeric_features.shape[1] == 0:
                self._update_group_feature_count_label(0, 0)
                self._set_single_status_row(
                    feature_text="No features selected",
                    flags_text="Enable at least one top-level filter",
                )
                return

            if numeric_features.empty:
                self._update_group_feature_count_label(0, 0)
                self._set_single_status_row(
                    feature_text="No numeric features",
                    flags_text="Check connections.feather",
                )
                return

            group_a = self._get_group_features(self.group_a_ids)
            group_b = self._get_group_features(self.group_b_ids)

            if group_a.empty or group_b.empty:
                self._update_group_feature_count_label(0, int(numeric_features.shape[1]))
                self._set_single_status_row(
                    feature_text="Missing group data",
                    flags_text="Could not match IDs between metadata and features",
                )
                return

            valid_cols = self._get_feature_columns_passing_threshold(group_a, group_b)
            self._update_group_feature_count_label(
                int(len(valid_cols)), int(numeric_features.shape[1])
            )
            if len(valid_cols) == 0:
                self._set_single_status_row(
                    feature_text="No features above threshold",
                    flags_text="Lower the minimum value or change the group selection",
                )
                return

            group_a = group_a.reindex(columns=valid_cols)
            group_b = group_b.reindex(columns=valid_cols)

            group_a_center, group_a_spread = self._aggregate_group(group_a)
            group_b_center, group_b_spread = self._aggregate_group(group_b)

            scores = self._compute_metric_scores(
                group_a,
                group_b,
                group_a_center,
                group_b_center,
            )
            self._update_metric_warning_banner(
                scores,
                group_a,
                group_b,
                group_a_center,
                group_b_center,
            )
            # Rank by selected Top-N mode (absolute or raw).
            # When scores tie exactly, prefer larger displayed group separation.
            ranking = scores.reindex(
                self._feature_ranking_order(
                    scores,
                    group_a_center,
                    group_b_center,
                    mode=str(self.top_n_mode_combo.currentData() or "abs"),
                )
            )
            top_n_value = int(self.top_n_spin.value())
            if top_n_value > 0:
                ranking = ranking.iloc[: min(top_n_value, len(ranking))]

            aggr_label = self._current_aggr_header_label()
            self.table.setHorizontalHeaderItem(
                self._get_col_group_a(),
                QtWidgets.QTableWidgetItem(f"Group A\n{aggr_label}"),
            )
            self.table.setHorizontalHeaderItem(
                self._get_col_group_b(),
                QtWidgets.QTableWidgetItem(f"Group B\n{aggr_label}"),
            )

            assumption_cfg = self._METRICS_WITH_ASSUMPTIONS.get(
                self.metric_combo.currentText()
            )

            # Pre-compute assumption flags for all displayed features for metrics that have them.
            assumption_flags: dict[object, list[str]] = {}
            if assumption_cfg is not None:
                group_a_valid = group_a.reindex(columns=valid_cols)
                group_b_valid = group_b.reindex(columns=valid_cols)
                for feature_name in ranking.index:
                    assumption_flags[feature_name] = self._compute_assumption_flags(
                        group_a_valid[feature_name],
                        group_b_valid[feature_name],
                        **assumption_cfg,
                    )

            # Use one shared color scale across both Group A and Group B columns.
            if len(ranking) > 0:
                ordered_features = list(ranking.index)
                combined_values = [
                    float(group_a_center.loc[f]) for f in ordered_features
                ] + [float(group_b_center.loc[f]) for f in ordered_features]
                color_min = 0.0
                color_max = max(combined_values)
            else:
                color_min = 0.0
                color_max = 0.0

            self.table.setRowCount(len(ranking))
            for row, (feature_name, score) in enumerate(ranking.items()):
                center_a = float(group_a_center.loc[feature_name])
                center_b = float(group_b_center.loc[feature_name])
                spread_a = (
                    float(group_a_spread.loc[feature_name])
                    if group_a_spread is not None
                    else None
                )
                spread_b = (
                    float(group_b_spread.loc[feature_name])
                    if group_b_spread is not None
                    else None
                )
                flags = assumption_flags.get(feature_name, [])
                notes = ", ".join(flags) if flags else ""

                # Extract hierarchy levels if MultiIndex
                if isinstance(feature_name, tuple):
                    hierarchy_parts = feature_name[:-1]
                    finest_grain = feature_name[-1]
                else:
                    hierarchy_parts = ()
                    finest_grain = feature_name

                # Populate Feature column (finest grain) first
                self.table.setItem(
                    row,
                    self._get_col_feature(),
                    QtWidgets.QTableWidgetItem(str(finest_grain)),
                )

                # Populate hierarchy columns after Feature
                for h_idx, h_value in enumerate(hierarchy_parts):
                    self.table.setItem(
                        row, h_idx + 1, QtWidgets.QTableWidgetItem(str(h_value))
                    )

                raw_score = float(score)
                abs_score = abs(raw_score)

                score_abs_item = _SortableNumberItem(f"{abs_score:.6g}")
                score_abs_item.setData(QtCore.Qt.UserRole, abs_score)
                self.table.setItem(row, self._get_col_score_abs(), score_abs_item)

                score_raw_item = _SortableNumberItem(f"{raw_score:.6g}")
                score_raw_item.setData(QtCore.Qt.UserRole, raw_score)
                self.table.setItem(row, self._get_col_score_raw(), score_raw_item)

                # Keep hidden numeric items under widget columns so they can be sorted.
                group_a_item = _SortableNumberItem("")
                group_a_item.setData(QtCore.Qt.UserRole, center_a)
                self.table.setItem(row, self._get_col_group_a(), group_a_item)
                self.table.setCellWidget(
                    row,
                    self._get_col_group_a(),
                    self._make_group_stat_label(
                        center_a,
                        spread_a,
                        color_value=center_a,
                        color_min=color_min,
                        color_max=color_max,
                    ),
                )

                group_b_item = _SortableNumberItem("")
                group_b_item.setData(QtCore.Qt.UserRole, center_b)
                self.table.setItem(row, self._get_col_group_b(), group_b_item)
                self.table.setCellWidget(
                    row,
                    self._get_col_group_b(),
                    self._make_group_stat_label(
                        center_b,
                        spread_b,
                        color_value=center_b,
                        color_min=color_min,
                        color_max=color_max,
                    ),
                )

                self.table.setItem(
                    row, self._get_col_flags(), QtWidgets.QTableWidgetItem(notes)
                )

            self._autosize_table_columns()
        finally:
            if was_sorting_enabled:
                self.table.setSortingEnabled(True)
            self._apply_feature_text_filter()

    def _table_to_text(self, separator):
        """Serialise visible table rows to delimited text."""
        headers = []
        for col in range(self.table.columnCount()):
            item = self.table.horizontalHeaderItem(col)
            headers.append(item.text().replace("\n", " ") if item else "")
        lines = [separator.join(headers)]
        for row in range(self.table.rowCount()):
            cells = []
            for col in range(self.table.columnCount()):
                widget = self.table.cellWidget(row, col)
                if widget is not None:
                    # Extract plain text from the rich label.
                    if isinstance(widget, QtWidgets.QLabel):
                        import re as _re

                        text = _re.sub(r"<[^>]+>", "", widget.text()).strip()
                    else:
                        text = ""
                else:
                    item = self.table.item(row, col)
                    text = item.text() if item else ""
                cells.append(text)
            lines.append(separator.join(cells))
        return "\n".join(lines)

    def _copy_to_clipboard(self):
        """Copy the current table to the system clipboard as tab-separated text."""
        text = self._table_to_text("\t")
        QtWidgets.QApplication.clipboard().setText(text)

    def _export_to_csv(self):
        """Export the current table to a user-chosen CSV file."""
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Export CSV", "feature_ranking.csv", "CSV files (*.csv)"
        )
        if not path:
            return
        text = self._table_to_text(",")
        try:
            with open(path, "w", encoding="utf-8", newline="") as fh:
                fh.write(text)
        except OSError as exc:
            QtWidgets.QMessageBox.warning(self, "Export failed", str(exc))

    def _on_feature_clicked(self, row, _col):
        """Open a distribution popup for the feature in the clicked row."""
        # Reconstruct the full feature name from hierarchy and fine-grain columns
        # Feature is in column 0, hierarchy levels are in columns 1 to _n_hierarchy_levels
        fine_item = self.table.item(row, self._get_col_feature())
        if fine_item is None:
            return
        finest_grain = fine_item.text()

        if not finest_grain or finest_grain in (
            "No features selected",
            "No numeric features",
            "Missing group data",
        ):
            return

        # Extract hierarchy levels and reconstruct tuple if MultiIndex
        if self._n_hierarchy_levels > 0:
            parts = []
            for h_idx in range(self._n_hierarchy_levels):
                item = self.table.item(row, h_idx + 1)
                if item:
                    parts.append(item.text())
            feature_name = tuple(parts + [finest_grain])
        else:
            feature_name = finest_grain

        try:
            self._show_feature_distribution(feature_name)
        except Exception:
            import traceback as _traceback

            QtWidgets.QMessageBox.critical(self, "Error", _traceback.format_exc())

    def _show_feature_distribution(self, feature_name):
        """Open a non-modal popup with a violin+strip plot for one feature."""
        import os as _os
        import traceback as _traceback

        _os.environ.setdefault("QT_API", "pyside6")

        try:
            from matplotlib.backends.backend_qtagg import (
                FigureCanvasQTAgg as _FigureCanvas,
            )
            from matplotlib.figure import Figure as _Figure
        except Exception:
            QtWidgets.QMessageBox.warning(
                self,
                "Matplotlib unavailable",
                f"Could not load matplotlib Qt backend:\n\n{_traceback.format_exc()}",
            )
            return

        numeric_features = self._get_numeric_feature_table()

        if feature_name not in numeric_features.columns:
            # Column names may be tuples (MultiIndex); fall back to matching by representation
            col_match = [
                c for c in numeric_features.columns if str(c) == str(feature_name)
            ]
            if not col_match:
                return
            feature_name = col_match[0]

        if self.normalize_check.isChecked():
            vals_a = (
                self._get_group_features(self.group_a_ids)[feature_name]
                .dropna()
                .values.astype(float)
            )
            vals_b = (
                self._get_group_features(self.group_b_ids)[feature_name]
                .dropna()
                .values.astype(float)
            )
        else:
            vals_a = (
                numeric_features.loc[
                    numeric_features.index.isin(self.group_a_ids), feature_name
                ]
                .dropna()
                .values.astype(float)
            )
            vals_b = (
                numeric_features.loc[
                    numeric_features.index.isin(self.group_b_ids), feature_name
                ]
                .dropna()
                .values.astype(float)
            )

        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle(str(feature_name))
        dlg.setAttribute(QtCore.Qt.WA_DeleteOnClose)
        dlg.resize(420, 360)

        fig = _Figure(figsize=(4.5, 3.5), layout="tight")
        ax = fig.add_subplot(111)

        rng = np.random.default_rng(1)
        groups = [
            (vals_a, self.group_a_name, "#5b9bd5"),
            (vals_b, self.group_b_name, "#ed7d31"),
        ]
        for idx, (vals, label, color) in enumerate(groups, start=1):
            if len(vals) >= 3:
                vp = ax.violinplot(
                    vals,
                    positions=[idx],
                    showmedians=True,
                    showextrema=False,
                    widths=0.55,
                )
                for pc in vp["bodies"]:
                    pc.set_facecolor(color)
                    pc.set_alpha(0.45)
                    pc.set_edgecolor("none")
                vp["cmedians"].set_color(color)
                vp["cmedians"].set_linewidth(2)
            jitter = rng.uniform(-0.12, 0.12, size=len(vals))
            ax.scatter(
                np.full(len(vals), idx) + jitter,
                vals,
                color=color,
                s=22,
                alpha=0.7,
                linewidths=0,
                label=f"{label} (n={len(vals)})",
            )

        ax.set_xticks([1, 2])
        ax.set_xticklabels([self.group_a_name, self.group_b_name])
        ax.set_ylabel(
            "Normalized value" if self.normalize_check.isChecked() else "Synapse count"
        )
        ax.set_title(str(feature_name), fontsize=10)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_xlim(0.4, 2.6)

        canvas = _FigureCanvas(fig)
        layout = QtWidgets.QVBoxLayout(dlg)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.addWidget(canvas)
        self._distribution_dialogs.append(dlg)
        dlg.destroyed.connect(
            lambda: (
                self._distribution_dialogs.remove(dlg)
                if dlg in self._distribution_dialogs
                else None
            )
        )
        dlg.show()

    def _autosize_table_columns(self):
        """Autoscale all table columns to current content."""
        self.table.resizeColumnsToContents()

    def _apply_feature_text_filter(self):
        """Hide rows whose Feature text does not match the current filter query."""
        if not hasattr(self, "table") or not hasattr(self, "feature_filter_edit"):
            return

        query = self.feature_filter_edit.text().strip().lower()
        feature_col = self._get_col_feature()

        for row in range(self.table.rowCount()):
            item = self.table.item(row, feature_col)
            text = item.text().lower() if item is not None else ""
            hide_row = bool(query) and (query not in text)
            self.table.setRowHidden(row, hide_row)

    def _set_single_status_row(
        self,
        feature_text,
        flags_text,
        score_text="-",
        group_a_text="-",
        group_b_text="-",
    ):
        """Render a one-row status message in the table."""
        self.table.setRowCount(1)
        self._clear_value_cell_widgets(0)
        values = (
            (self._get_col_feature(), feature_text),
            (self._get_col_score_abs(), score_text),
            (self._get_col_score_raw(), score_text),
            (self._get_col_group_a(), group_a_text),
            (self._get_col_group_b(), group_b_text),
            (self._get_col_flags(), flags_text),
        )
        for col, text in values:
            self.table.setItem(0, col, QtWidgets.QTableWidgetItem(text))
        self._autosize_table_columns()

    def _clear_value_cell_widgets(self, row):
        """Remove rich-text value widgets for a table row if present."""
        for col in (self._get_col_group_a(), self._get_col_group_b()):
            widget = self.table.cellWidget(row, col)
            if widget is not None:
                self.table.removeCellWidget(row, col)
                widget.deleteLater()

    def _apply_group_box_state_style(self, box, color):
        """Apply the selected/empty state border style to a group box."""
        box.setStyleSheet(
            f"QGroupBox {{ border: 1px solid {color}; border-radius: 4px; margin-top: 6px; }} "
            f"QGroupBox::title {{ subcontrol-origin: margin; left: 8px; padding: 0 3px; color: {color}; }}"
        )

    def _value_to_rgb(self, color_value, color_min, color_max):
        """Map a value to a soft blue color on a shared [min, max] scale."""
        if (
            not np.isfinite(color_value)
            or not np.isfinite(color_min)
            or not np.isfinite(color_max)
        ):
            return 245, 247, 250
        if color_max <= color_min:
            ratio = 0.5
        else:
            ratio = (color_value - color_min) / (color_max - color_min)
            ratio = max(0.0, min(1.0, ratio))

        # Interpolate from very light gray-blue to deeper blue.
        start = (245, 247, 250)
        end = (139, 184, 255)
        r = int(start[0] + ratio * (end[0] - start[0]))
        g = int(start[1] + ratio * (end[1] - start[1]))
        b = int(start[2] + ratio * (end[2] - start[2]))
        return r, g, b

    def _make_group_stat_label(
        self, center_value, spread_value, color_value=None, color_min=0.0, color_max=0.0
    ):
        """Create a label that styles spread text and colors background by value."""
        if spread_value is None:
            html = f"{center_value:.3g}"
        else:
            html = (
                f"{center_value:.3g} "
                f'<span style="color:#9a9a9a; font-size:10px;">± {spread_value:.3g}</span>'
            )
        label = QtWidgets.QLabel(html)
        label.setTextFormat(QtCore.Qt.RichText)
        label.setAlignment(QtCore.Qt.AlignCenter)
        label.setContentsMargins(0, 0, 0, 0)
        label.setStyleSheet("color: #000;")
        if color_value is not None:
            r, g, b = self._value_to_rgb(color_value, color_min, color_max)
            label.setStyleSheet(
                f"color: #000; background-color: rgb({r}, {g}, {b}); border-radius: 2px;"
            )
        return label

    def _current_aggr_header_label(self):
        """Return a short parenthesized label for the active aggregation method."""
        return {
            "Mean ± SD": "(mean ± sd)",
            "Median ± IQR": "(median ± IQR)",
            "Trimmed mean (10%) ± SD": "(trim. mean ± sd)",
            "Max": "(max)",
        }.get(self.aggregation_combo.currentText(), "")

    def _aggregate_group(self, group):
        """Return (center, spread) Series for the selected aggregation method.

        spread is None for methods that carry no secondary statistic.
        """
        method = self.aggregation_combo.currentText()
        if method == "Mean ± SD":
            return group.mean(axis=0), group.std(axis=0).fillna(0)
        elif method == "Median ± IQR":
            q75 = group.quantile(0.75, axis=0)
            q25 = group.quantile(0.25, axis=0)
            return group.median(axis=0), (q75 - q25)
        elif method == "Trimmed mean (10%) ± SD":

            def _tmean(col):
                arr = np.sort(col.dropna().values)
                k = int(np.floor(0.1 * len(arr)))
                trimmed = arr[k : len(arr) - k] if k > 0 and len(arr) - k > k else arr
                return float(trimmed.mean()) if len(trimmed) else float("nan")

            def _tstd(col):
                arr = np.sort(col.dropna().values)
                k = int(np.floor(0.1 * len(arr)))
                trimmed = arr[k : len(arr) - k] if k > 0 and len(arr) - k > k else arr
                return float(trimmed.std()) if len(trimmed) > 1 else 0.0

            return group.apply(_tmean), group.apply(_tstd)
        elif method == "Max":
            return group.max(axis=0), None
        # fallback
        return group.mean(axis=0), group.std(axis=0).fillna(0)

    def _center_difference(self, center_a, center_b):
        """Return signed center difference as float to avoid unsigned wrap-around."""
        a = pd.to_numeric(center_a, errors="coerce").astype(float)
        b = pd.to_numeric(center_b, errors="coerce").astype(float)
        return a - b

    def _feature_ranking_order(self, scores, center_a, center_b, mode="abs"):
        """Return a stable feature order using selected score mode and center shift tie-break."""
        score_mode = str(mode).lower()
        if score_mode not in {"abs", "raw"}:
            score_mode = "abs"

        ranking_df = pd.DataFrame(
            {
                "abs_score": scores.abs(),
                "raw_score": scores,
                "abs_center_diff": self._center_difference(center_a, center_b)
                .abs()
                .reindex(scores.index),
            },
            index=scores.index,
        ).fillna({"abs_score": -np.inf, "raw_score": -np.inf, "abs_center_diff": -np.inf})

        if score_mode == "raw":
            ranking_df = ranking_df.sort_values(
                ["raw_score", "abs_center_diff"],
                ascending=[False, False],
                kind="mergesort",
            )
        else:
            ranking_df = ranking_df.sort_values(
                ["abs_score", "abs_center_diff"],
                ascending=[False, False],
                kind="mergesort",
            )
        return ranking_df.index

    def _set_metric_warning(self, text):
        """Show or hide the metric warning banner."""
        if not hasattr(self, "metric_warning_label"):
            return
        if not text:
            self.metric_warning_label.clear()
            self.metric_warning_label.hide()
            return
        self.metric_warning_label.setText(text)
        self.metric_warning_label.show()

    def _label_entropy(self, n_a, n_b):
        """Return the binary label entropy in nats for the current group sizes."""
        total = int(n_a + n_b)
        if total <= 0:
            return 0.0

        probs = np.array([n_a / total, n_b / total], dtype=float)
        probs = probs[probs > 0]
        if probs.size == 0:
            return 0.0
        return float(-(probs * np.log(probs)).sum())

    def _count_near_perfect_single_feature_separators(self, group_a, group_b):
        """Count features whose single-feature AUC is effectively perfect."""
        y = np.array([0] * len(group_a) + [1] * len(group_b), dtype=int)
        count = 0
        for col in group_a.columns:
            x = np.concatenate(
                [
                    group_a[col].values.astype(float),
                    group_b[col].values.astype(float),
                ]
            )
            valid = np.isfinite(x)
            if valid.sum() < 2:
                continue
            try:
                auc = float(_roc_auc_score(y[valid], x[valid]))
            except Exception:
                continue
            if max(auc, 1.0 - auc) >= 0.99:
                count += 1
        return count

    def _update_metric_warning_banner(
        self, scores, group_a, group_b, center_a, center_b
    ):
        """Surface metric-specific warnings for saturated or degenerate outputs."""
        metric = self.metric_combo.currentText()
        finite_scores = scores[np.isfinite(scores.values)].astype(float)
        if finite_scores.empty:
            self._set_metric_warning(None)
            return

        warning_text = None

        if metric == "Permutation Importance":
            scoring = str(self.perm_scoring_combo.currentText())
            finite_abs = np.abs(finite_scores.values.astype(float))
            rounded_abs = np.unique(np.round(finite_abs, decimals=10))
            nonzero_abs = rounded_abs[rounded_abs > 1e-10]
            compressed_scores = finite_abs.size > 0 and (
                finite_abs.max() <= 5e-2 and nonzero_abs.size <= 3
            )
            near_perfect = self._count_near_perfect_single_feature_separators(
                group_a, group_b
            )
            min_group_n = min(len(group_a), len(group_b))
            if self._is_degenerate_permutation_importance(
                finite_scores.values, scoring
            ) or compressed_scores or (
                scoring in {"roc_auc", "average_precision"}
                and (
                    (near_perfect >= 1 and min_group_n <= 50)
                    or (near_perfect >= 2)
                )
            ) or (
                scoring == "accuracy" and near_perfect >= 1 and min_group_n <= 30
            ):
                warning_text = (
                    "Permutation importance may be unreliable here because the groups look very easy to separate, "
                    "which can make the reported importances coarse, unstable, or less informative than they appear. "
                    "Treat this ranking cautiously; consider neg_log_loss, cross-validation, or a simpler single-feature metric."
                )

        elif metric == "Mutual information":
            entropy = self._label_entropy(len(group_a), len(group_b))
            if entropy > 0:
                near_ceiling = np.isclose(
                    finite_scores.values,
                    entropy,
                    rtol=2e-2,
                    atol=max(1e-6, entropy * 2e-2),
                )
                if int(near_ceiling.sum()) >= 2:
                    warning_text = (
                        "Mutual information is saturated: several features nearly determine the group label, "
                        "so they receive the same maximal score. Use mean difference, AUC, or inspect raw group values to break ties."
                    )

        elif metric == "Logistic (L1)":
            nonzero_scores = int((np.abs(finite_scores.values) > 1e-12).sum())
            near_perfect = self._count_near_perfect_single_feature_separators(
                group_a, group_b
            )
            min_group_n = min(len(group_a), len(group_b))
            sparse_explanation = nonzero_scores <= 5
            if (
                (near_perfect >= 1 and min_group_n <= 50)
                or near_perfect >= 2
                or (near_perfect >= 1 and sparse_explanation)
                or (near_perfect >= 1 and not self.l1_stability_check.isChecked())
            ):
                warning_text = (
                    "Logistic (L1) ranking may be unstable here because multiple features may separate the groups well, "
                    "while the model still picks one sparse explanation. Coefficients can therefore reflect one convenient solution, "
                    "not necessarily the most biologically important feature set. Treat this ranking cautiously."
                )

        self._set_metric_warning(warning_text)

    def _fit_permutation_model(self, X, y):
        """Fit the shared classifier used for permutation-importance scoring."""
        model = _LogisticRegression(
            penalty="l2",
            C=1.0,
            solver="lbfgs",
            max_iter=2000,
            fit_intercept=True,
            random_state=0,
        )
        model.fit(X, y)
        return model

    def _collect_permutation_importances(self, X, y, scoring, n_repeats, eval_mode):
        """Return mean permutation importances for the requested scoring/eval mode."""
        if eval_mode == "In-sample":
            model = self._fit_permutation_model(X, y)
            perm = _permutation_importance(
                model,
                X,
                y,
                scoring=scoring,
                n_repeats=n_repeats,
                random_state=0,
                n_jobs=1,
            )
            return perm.importances_mean.astype(float)

        n_splits = 3 if eval_mode == "3-fold CV" else 5
        skf = _StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=0)
        fold_importances = []
        for train_idx, test_idx in skf.split(X, y):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            if np.unique(y_train).size < 2 or np.unique(y_test).size < 2:
                continue
            model = self._fit_permutation_model(X_train, y_train)
            perm = _permutation_importance(
                model,
                X_test,
                y_test,
                scoring=scoring,
                n_repeats=n_repeats,
                random_state=0,
                n_jobs=1,
            )
            fold_importances.append(perm.importances_mean.astype(float))

        if not fold_importances:
            return None

        return np.mean(np.vstack(fold_importances), axis=0)

    def _is_degenerate_permutation_importance(self, importances, scoring):
        """Detect coarse or collapsed permutation scores that are not useful for ranking."""
        finite = np.asarray(importances, dtype=float)
        finite = finite[np.isfinite(finite)]
        if finite.size == 0:
            return True

        finite_abs = np.abs(finite)
        if np.allclose(finite_abs, 0.0):
            return True

        if scoring not in {"roc_auc", "average_precision"}:
            return False

        rounded = np.unique(np.round(finite_abs, decimals=12))
        nonzero = rounded[rounded > 1e-12]
        return finite_abs.max() <= 1e-2 and nonzero.size <= 1

    def _compute_assumption_flags(
        self, col_a, col_b, check_normality=True, check_equal_var=True
    ):
        """Return flag strings for violated statistical assumptions.

        check_normality: Shapiro-Wilk on each group (p < 0.05 → "!normal")
        check_equal_var: Levene test (p < 0.05 → "!equal-var")
        """
        alpha = 0.05
        flags = []
        a = col_a.dropna().values.astype(float)
        b = col_b.dropna().values.astype(float)

        if check_normality:
            # Shapiro-Wilk requires n in [3, 5000]; skip groups outside that range.
            for arr in (a, b):
                if 3 <= len(arr) <= 5000:
                    _, p = _scipy_stats.shapiro(arr)
                    if p < alpha:
                        flags.append("!normal")
                        break  # one violation is enough to flag

        if check_equal_var and len(a) >= 2 and len(b) >= 2:
            _, p = _scipy_stats.levene(a, b)
            if p < alpha:
                flags.append("!equal-var")

        return flags

    def _compute_metric_scores(self, group_a, group_b, center_a, center_b):
        """Return a signed score Series per feature for the selected metric."""
        method = self.metric_combo.currentText()
        if method == "Effect size (Cohen d)":
            n_a, n_b = len(group_a), len(group_b)
            denom = max(n_a + n_b - 2, 1)
            var_a = group_a.var(axis=0, ddof=1).fillna(0)
            var_b = group_b.var(axis=0, ddof=1).fillna(0)
            pooled_std = np.sqrt(((n_a - 1) * var_a + (n_b - 1) * var_b) / denom)
            pooled_std = pooled_std.replace(0, float("nan"))
            return (group_a.mean(axis=0) - group_b.mean(axis=0)) / pooled_std
        elif method == "T-test (Welch's)":

            def _t_stat(col):
                a = group_a[col].dropna().values.astype(float)
                b = group_b[col].dropna().values.astype(float)
                if len(a) < 2 or len(b) < 2:
                    return float("nan")
                return float(_scipy_stats.ttest_ind(a, b, equal_var=False).statistic)

            return pd.Series(
                {col: _t_stat(col) for col in group_a.columns}, dtype=float
            )
        elif method == "One-way ANOVA":

            def _f_stat(col):
                a = group_a[col].dropna().values.astype(float)
                b = group_b[col].dropna().values.astype(float)
                if len(a) < 2 or len(b) < 2:
                    return float("nan")
                return float(_scipy_stats.f_oneway(a, b).statistic)

            return pd.Series(
                {col: _f_stat(col) for col in group_a.columns}, dtype=float
            )
        elif method == "Rank-biserial correlation":

            def _rbc(col):
                a = group_a[col].dropna().values.astype(float)
                b = group_b[col].dropna().values.astype(float)
                if len(a) < 1 or len(b) < 1:
                    return float("nan")
                # U1: number of (a_i, b_j) pairs where a_i > b_j.
                u1 = float(
                    _scipy_stats.mannwhitneyu(a, b, alternative="greater").statistic
                )
                u2 = len(a) * len(b) - u1
                return (u1 - u2) / (len(a) * len(b))

            return pd.Series({col: _rbc(col) for col in group_a.columns}, dtype=float)
        elif method == "Brunner-Munzel":

            def _bm(col):
                a = group_a[col].dropna().values.astype(float)
                b = group_b[col].dropna().values.astype(float)
                if len(a) < 2 or len(b) < 2:
                    return float("nan")
                try:
                    return float(_scipy_stats.brunnermunzel(a, b).statistic)
                except Exception:
                    return float("nan")

            return pd.Series({col: _bm(col) for col in group_a.columns}, dtype=float)
        elif method == "Hurdle score":

            def _hurdle(col):
                a = group_a[col].dropna().values.astype(float)
                b = group_b[col].dropna().values.astype(float)
                if len(a) < 1 or len(b) < 1:
                    return float("nan")
                presence_diff = float((a > 0).mean() - (b > 0).mean())
                nz_a = a[a > 0]
                nz_b = b[b > 0]
                if len(nz_a) == 0 or len(nz_b) == 0:
                    magnitude_diff = 0.0
                else:
                    magnitude_diff = float(
                        np.log1p(np.median(nz_a)) - np.log1p(np.median(nz_b))
                    )
                return presence_diff + magnitude_diff

            return pd.Series(
                {col: _hurdle(col) for col in group_a.columns}, dtype=float
            )
        elif method == "AUC (single-feature)":
            y = np.array([0] * len(group_a) + [1] * len(group_b), dtype=int)

            def _auc(col):
                a = group_a[col].values.astype(float)
                b = group_b[col].values.astype(float)
                x = np.concatenate([a, b])
                valid = np.isfinite(x)
                if valid.sum() < 2:
                    return float("nan")
                try:
                    auc = float(_roc_auc_score(y[valid], x[valid]))
                    return auc - 0.5
                except Exception:
                    return float("nan")

            return pd.Series({col: _auc(col) for col in group_a.columns}, dtype=float)
        elif method == "Logistic (L1)":
            X_df = pd.concat([group_a, group_b], axis=0).fillna(0.0)
            X = X_df.values.astype(float)
            y = np.array([0] * len(group_a) + [1] * len(group_b), dtype=int)

            try:
                c_value = float(self.l1_c_spin.value())
                solver = self.l1_solver_combo.currentText()
                max_iter = int(self.l1_max_iter_spin.value())
                use_standardize = self.l1_standardize_check.isChecked()

                if self.l1_stability_check.isChecked():
                    if X.shape[1] == 0 or np.unique(y).size < 2:
                        return pd.Series(np.nan, index=group_a.columns, dtype=float)

                    n_boot = int(self.l1_stability_bootstraps_spin.value())
                    subsample = float(self.l1_stability_subsample_spin.value())

                    idx_a = np.where(y == 0)[0]
                    idx_b = np.where(y == 1)[0]
                    if len(idx_a) < 2 or len(idx_b) < 2:
                        return pd.Series(np.nan, index=group_a.columns, dtype=float)

                    n_a = max(2, int(round(subsample * len(idx_a))))
                    n_b = max(2, int(round(subsample * len(idx_b))))

                    rng = np.random.default_rng(0)
                    select_counts = np.zeros(X.shape[1], dtype=float)
                    direction_sum = np.zeros(X.shape[1], dtype=float)
                    n_success = 0

                    for _ in range(n_boot):
                        boot_a = rng.choice(idx_a, size=n_a, replace=True)
                        boot_b = rng.choice(idx_b, size=n_b, replace=True)
                        boot_idx = np.concatenate([boot_a, boot_b])
                        X_boot = X[boot_idx].copy()
                        y_boot = y[boot_idx]

                        if np.unique(y_boot).size < 2:
                            continue

                        if use_standardize and X_boot.shape[1] > 0:
                            mu = X_boot.mean(axis=0)
                            sigma = X_boot.std(axis=0)
                            sigma[sigma == 0] = 1.0
                            X_boot = (X_boot - mu) / sigma

                        try:
                            model = _LogisticRegression(
                                penalty="l1",
                                C=c_value,
                                solver=solver,
                                max_iter=max_iter,
                                fit_intercept=True,
                                random_state=0,
                            )
                            model.fit(X_boot, y_boot)
                            coef = model.coef_[0].astype(float)
                        except Exception:
                            continue

                        n_success += 1
                        selected = np.abs(coef) > 1e-12
                        select_counts += selected.astype(float)
                        direction_sum += np.sign(coef)

                    if n_success == 0:
                        return pd.Series(np.nan, index=group_a.columns, dtype=float)

                    freq = select_counts / float(n_success)
                    direction = np.sign(direction_sum)
                    fallback_dir = np.sign(
                        self._center_difference(center_a, center_b)
                        .reindex(group_a.columns)
                        .values.astype(float)
                    )
                    direction[direction == 0] = fallback_dir[direction == 0]

                    scores = freq * direction
                    return pd.Series(scores, index=group_a.columns, dtype=float)

                if use_standardize and X.shape[1] > 0:
                    mu = X.mean(axis=0)
                    sigma = X.std(axis=0)
                    sigma[sigma == 0] = 1.0
                    X = (X - mu) / sigma

                model = _LogisticRegression(
                    penalty="l1",
                    C=c_value,
                    solver=solver,
                    max_iter=max_iter,
                    fit_intercept=True,
                    random_state=0,
                )
                model.fit(X, y)
                coef = model.coef_[0]
                return pd.Series(coef, index=group_a.columns, dtype=float)
            except Exception:
                return pd.Series(np.nan, index=group_a.columns, dtype=float)
        elif method == "Permutation Importance":
            X_df = pd.concat([group_a, group_b], axis=0).fillna(0.0)
            X = X_df.values.astype(float)
            y = np.array([0] * len(group_a) + [1] * len(group_b), dtype=int)

            if X.shape[1] == 0 or np.unique(y).size < 2:
                return pd.Series(np.nan, index=group_a.columns, dtype=float)

            try:
                n_repeats = int(self.perm_repeats_spin.value())
                scoring = str(self.perm_scoring_combo.currentText())
                eval_mode = str(self.perm_eval_combo.currentText())

                direction = np.sign(
                    self._center_difference(center_a, center_b)
                    .reindex(group_a.columns)
                    .values.astype(float)
                )

                importances = self._collect_permutation_importances(
                    X,
                    y,
                    scoring=scoring,
                    n_repeats=n_repeats,
                    eval_mode=eval_mode,
                )
                if importances is None:
                    return pd.Series(np.nan, index=group_a.columns, dtype=float)

                current_scoring = scoring
                if self._is_degenerate_permutation_importance(
                    importances, current_scoring
                ) and scoring != "neg_log_loss":
                    fallback_importances = self._collect_permutation_importances(
                        X,
                        y,
                        scoring="neg_log_loss",
                        n_repeats=n_repeats,
                        eval_mode=eval_mode,
                    )
                    if fallback_importances is not None:
                        importances = fallback_importances
                        current_scoring = "neg_log_loss"

                if self._is_degenerate_permutation_importance(
                    importances, current_scoring
                ):
                    fallback_model = self._fit_permutation_model(X, y)
                    importances = np.abs(fallback_model.coef_[0]).astype(float)

                scores = importances * direction
                return pd.Series(scores, index=group_a.columns, dtype=float)
            except Exception:
                return pd.Series(np.nan, index=group_a.columns, dtype=float)
        elif method == "Mutual information":
            X_df = pd.concat([group_a, group_b], axis=0).fillna(0.0)
            X = X_df.values.astype(float)
            y = np.array([0] * len(group_a) + [1] * len(group_b), dtype=int)

            # Use a per-feature discrete mask for count-like columns when not normalized.
            # This avoids continuous-kNN MI artifacts on integer-valued sparse counts.
            if self.normalize_check.isChecked():
                discrete_mask = False
            else:
                discrete_mask = np.all(np.isclose(X, np.rint(X)), axis=0)

            min_class_n = int(min(len(group_a), len(group_b)))
            n_neighbors = max(1, min(5, min_class_n - 1)) if min_class_n > 1 else 1

            if self.normalize_check.isChecked():
                # On normalized sparse vectors, kNN MI can appear quantized for small groups.
                # Average a few tiny-jitter estimates to smooth plateaus while staying deterministic.
                rng = np.random.default_rng(0)
                std = X.std(axis=0)
                std[~np.isfinite(std)] = 0.0
                jitter_scale = np.where(std > 0, std * 1e-8, 1e-12)
                mi_runs = []
                for _ in range(4):
                    X_jitter = X + rng.normal(0.0, jitter_scale, size=X.shape)
                    mi_runs.append(
                        _mutual_info_classif(
                            X_jitter,
                            y,
                            discrete_features=False,
                            n_neighbors=n_neighbors,
                            random_state=0,
                        )
                    )
                mi = np.mean(np.vstack(mi_runs), axis=0)
            else:
                mi = _mutual_info_classif(
                    X,
                    y,
                    discrete_features=discrete_mask,
                    n_neighbors=n_neighbors,
                    random_state=0,
                )

            # In near-perfect separation regimes, MI estimates can differ by tiny
            # numerical amounts while all features are effectively at the entropy
            # ceiling. Collapse those near-ceiling scores to stabilize ranking.
            entropy = self._label_entropy(len(group_a), len(group_b))
            if entropy > 0:
                mi = np.minimum(mi, entropy)
                near_ceiling = np.isclose(
                    mi,
                    entropy,
                    rtol=2e-2,
                    atol=max(1e-6, entropy * 2e-2),
                )
                if int(near_ceiling.sum()) >= 2:
                    mi = mi.copy()
                    mi[near_ceiling] = entropy
            return pd.Series(mi, index=group_a.columns, dtype=float)
        # Default: center difference (respects aggregation choice)
        return self._center_difference(center_a, center_b)

    def _normalize_features(self, features):
        """Normalize features row-wise, per top-level group when MultiIndex."""
        if isinstance(features.columns, pd.MultiIndex):
            normalized_parts = []
            for level in features.columns.get_level_values(0).unique():
                part = (
                    features.loc[:, features.columns.get_level_values(0) == level]
                    .copy()
                    .astype(float)
                )
                row_totals = part.sum(axis=1).replace(0, float("nan"))
                normalized_parts.append(part.div(row_totals, axis=0).fillna(0))
            return pd.concat(normalized_parts, axis=1)
        else:
            features = features.copy().astype(float)
            row_totals = features.sum(axis=1).replace(0, float("nan"))
            return features.div(row_totals, axis=0).fillna(0)

    def _filter_by_top_level(self, features):
        """Filter a feature table by checked top-level MultiIndex columns."""
        if not isinstance(features.columns, pd.MultiIndex):
            return features

        if not self.top_level_checks:
            return features

        selected_levels = [
            level_name
            for level_name, check in self.top_level_checks.items()
            if check.isChecked()
        ]
        if not selected_levels:
            return features.iloc[:, 0:0]

        keep_mask = features.columns.get_level_values(0).isin(selected_levels)
        return features.loc[:, keep_mask]

    def _get_numeric_feature_table(self):
        """Return the currently visible numeric feature table after top-level filtering."""
        return self._filter_by_top_level(self.features.select_dtypes(include="number"))

    def _current_min_feature_value(self):
        """Return the active raw feature-value threshold."""
        if self.normalize_check.isChecked():
            return float(self.min_count_spin_norm.value())
        return float(self.min_count_spin.value())

    def _get_feature_columns_passing_threshold(self, group_a, group_b):
        """Return features whose raw selected values clear the active threshold in either group."""
        if group_a.empty or group_b.empty:
            return group_a.columns[:0]

        min_value = self._current_min_feature_value()
        if min_value <= 0:
            return group_a.columns

        max_a = group_a.fillna(0.0).max(axis=0)
        max_b = group_b.fillna(0.0).max(axis=0)
        keep_mask = (max_a >= min_value) | (max_b >= min_value)
        return group_a.columns[keep_mask]

    def _get_group_features(self, group_ids):
        """Return the selected group's numeric feature rows, normalized if enabled."""
        features = self._get_numeric_feature_table()
        group = features.loc[features.index.isin(group_ids)]
        if self.normalize_check.isChecked():
            return self._normalize_features(group)
        return group

    def closeEvent(self, event):
        """Ensure figure callbacks are removed when this widget closes."""
        if self.figure is not None and hasattr(self.figure, "unsync_widget"):
            self.figure.unsync_widget(self)
        super().closeEvent(event)


def _set_feature_index(connections, candidate_ids):
    """Return a feature table indexed by neuron IDs."""
    if connections.empty:
        return connections

    if connections.index.isin(candidate_ids).sum() > 0:
        return connections

    best_col = None
    best_hits = -1
    for col in connections.columns:
        hits = connections[col].isin(candidate_ids).sum()
        if hits > best_hits:
            best_col = col
            best_hits = hits

    if best_col is None or best_hits <= 0:
        raise ValueError(
            "Could not infer ID index for connections.feather. "
            "Expected an index or column overlapping with meta['id']."
        )

    indexed = connections.set_index(best_col, drop=True)
    indexed.index.name = "id"
    return indexed


def _load_prototype_data():
    """Load metadata and full feature table for standalone group selection."""
    base = Path("~/projects/mcns_paper/JO_cluster/JO_Male_v_Female").expanduser()
    meta_path = base / "meta.feather"
    connections_path = base / "connections.feather"

    meta = pd.read_feather(meta_path)
    if "id" not in meta.columns or "label" not in meta.columns:
        raise ValueError("meta.feather must contain 'id' and 'label' columns.")

    meta = meta.dropna(subset=["id", "label"]).copy()
    meta["id"] = meta["id"].astype("int64")
    meta["label"] = meta["label"].astype(str)

    connections = pd.read_feather(connections_path)
    candidate_ids = set(meta["id"].tolist())
    features = _set_feature_index(connections, candidate_ids)
    # Keep all metadata rows with matching feature rows and preserve metadata order.
    selected_ids = pd.Index(meta["id"].tolist(), dtype="int64")
    filtered = features.loc[features.index.isin(selected_ids)].copy()
    filtered = filtered.reindex(selected_ids.intersection(filtered.index))

    return meta, filtered


def _run_prototype_app():
    """Start a minimal app for rapid widget prototyping."""
    meta, filtered_features = _load_prototype_data()

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    window = QtWidgets.QMainWindow()
    window.setWindowTitle("FeatureExplorerWidget Prototype")
    # window.resize(980, 640)

    feature_widget = FeatureExplorerWidget(
        metadata=meta,
        features=filtered_features,
    )
    window.setCentralWidget(feature_widget)
    window.show()

    app.exec()


if __name__ == "__main__":
    _run_prototype_app()
