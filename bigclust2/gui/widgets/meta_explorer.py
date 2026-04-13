import logging

import numpy as np
import pandas as pd

from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtCore import Qt


logger = logging.getLogger(__name__)


class DescriptorComboDelegate(QtWidgets.QStyledItemDelegate):
	def paint(self, painter, option, index):
		if index.row() == 0:
			option = QtWidgets.QStyleOptionViewItem(option)
			option.palette.setColor(QtGui.QPalette.Text, Qt.gray)
		QtWidgets.QStyledItemDelegate.paint(self, painter, option, index)


class MetaTableModel(QtCore.QAbstractTableModel):
	"""Simple read-only table model backed by a pandas DataFrame."""

	def __init__(self, data):
		super().__init__()
		if not isinstance(data, pd.DataFrame):
			raise TypeError("data must be a pandas DataFrame")
		self._data = data
		self._row_positions = np.arange(len(self._data), dtype=np.int64)

	def set_row_positions(self, row_positions):
		self.beginResetModel()
		self._row_positions = np.asarray(row_positions, dtype=np.int64)
		self.endResetModel()

	@property
	def row_positions(self):
		return self._row_positions

	def rowCount(self, parent=None):
		if parent and parent.isValid():
			return 0
		return int(len(self._row_positions))

	def columnCount(self, parent=None):
		if parent and parent.isValid():
			return 0
		return int(self._data.shape[1])

	def headerData(self, section, orientation, role):
		if role != Qt.DisplayRole:
			return None

		if orientation == Qt.Horizontal:
			return str(self._data.columns[section])

		if section >= len(self._row_positions):
			return None

		row_pos = self._row_positions[section]
		return str(self._data.index[row_pos])

	def sort(self, column, order=Qt.AscendingOrder):
		"""Sort visible rows by the given column."""
		positions = np.asarray(self._row_positions, dtype=np.int64)
		column_values = self._data.iloc[positions, column].values
		series = pd.Series(column_values, index=positions)
		try:
			sorted_positions = series.sort_values(
				kind="mergesort",
				ascending=(order == Qt.AscendingOrder),
				na_position="last",
			).index.to_numpy(dtype=np.int64)
		except TypeError:
			sorted_positions = series.astype(str).sort_values(
				kind="mergesort",
				ascending=(order == Qt.AscendingOrder),
				na_position="last",
			).index.to_numpy(dtype=np.int64)
		self.beginResetModel()
		self._row_positions = sorted_positions
		self.endResetModel()

	def data(self, index, role):
		if not index.isValid():
			return None

		if role == Qt.TextAlignmentRole:
			return Qt.AlignLeft | Qt.AlignVCenter

		if role not in (Qt.DisplayRole, Qt.ToolTipRole):
			return None

		row_pos = int(self._row_positions[index.row()])
		value = self._data.iat[row_pos, index.column()]

		if pd.isna(value):
			return ""
		if isinstance(value, float):
			return f"{value:.6g}"
		return str(value)


class ColumnFilterRow(QtWidgets.QWidget):
	"""One row of filter controls: column + operation + value."""

	changed = QtCore.Signal()
	removeRequested = QtCore.Signal(object)

	OPERATORS = [
		"contains",
		"equals",
		"not equals",
		">",
		">=",
		"<",
		"<=",
		"is empty",
		"is not empty",
	]

	def __init__(self, columns, parent=None):
		super().__init__(parent)

		layout = QtWidgets.QHBoxLayout(self)
		layout.setContentsMargins(0, 0, 0, 0)
		layout.setSpacing(6)

		self.column_combo = QtWidgets.QComboBox()
		self.column_combo.addItems([str(c) for c in columns])

		self.op_combo = QtWidgets.QComboBox()
		self.op_combo.addItems(self.OPERATORS)

		self.value_edit = QtWidgets.QLineEdit()
		self.value_edit.setPlaceholderText("value")

		self.remove_btn = QtWidgets.QPushButton("Remove")
		self.remove_btn.setToolTip("Remove this filter")

		layout.addWidget(self.column_combo, 2)
		layout.addWidget(self.op_combo, 1)
		layout.addWidget(self.value_edit, 3)
		layout.addWidget(self.remove_btn)

		self.column_combo.currentTextChanged.connect(self._emit_changed)
		self.op_combo.currentTextChanged.connect(self._on_operator_changed)
		self.value_edit.textChanged.connect(self._emit_changed)
		self.remove_btn.clicked.connect(lambda: self.removeRequested.emit(self))

		self._on_operator_changed(self.op_combo.currentText())

	def _emit_changed(self):
		self.value_edit.setStyleSheet("")
		self.changed.emit()

	def _on_operator_changed(self, op):
		needs_value = op not in ("is empty", "is not empty")
		self.value_edit.setEnabled(needs_value)
		self._emit_changed()

	def filter_spec(self):
		return {
			"column": self.column_combo.currentText(),
			"op": self.op_combo.currentText(),
			"value": self.value_edit.text(),
		}


class MetaExplorerDialog(QtWidgets.QDialog):
	"""Dialog for exploring and filtering metadata."""

	selectRequested = QtCore.Signal(object)
	openInNewWindowRequested = QtCore.Signal(object)

	def __init__(self, meta, figure=None, parent=None):
		super().__init__(parent)
		if not isinstance(meta, pd.DataFrame):
			raise TypeError("meta must be a pandas DataFrame")

		self._meta = meta
		self._figure = figure
		self._string_cache = {}
		self._numeric_cache = {}
		self._filter_rows = []

		self.setWindowTitle("Meta Explorer")
		self.resize(1200, 760)
		self.setModal(False)

		self._init_ui()
		self.add_filter_row()
		self.apply_filters()

	def _init_ui(self):
		layout = QtWidgets.QVBoxLayout(self)
		layout.setContentsMargins(10, 10, 10, 10)
		layout.setSpacing(8)

		# Filter panel
		filter_group = QtWidgets.QGroupBox("Filters")
		filter_group_layout = QtWidgets.QVBoxLayout(filter_group)
		filter_group_layout.setContentsMargins(10, 10, 10, 10)
		filter_group_layout.setSpacing(8)

		filter_header_row = QtWidgets.QHBoxLayout()
		self.filter_toggle_btn = QtWidgets.QToolButton()
		self.filter_toggle_btn.setArrowType(Qt.DownArrow)
		self.filter_toggle_btn.setToolButtonStyle(Qt.ToolButtonIconOnly)
		self.filter_toggle_btn.setCheckable(True)
		self.filter_toggle_btn.setChecked(True)
		self.filter_toggle_btn.setFixedSize(18, 18)
		self.filter_toggle_btn.clicked.connect(self._toggle_filter_panel)

		filter_header_label = QtWidgets.QLabel("Column Filters")
		filter_header_label.setStyleSheet("font-weight: 600;")
		self.add_filter_btn = QtWidgets.QPushButton("Add Filter")
		self.add_filter_btn.clicked.connect(self.add_filter_row)
		self.clear_filters_btn = QtWidgets.QPushButton("Clear Filters")
		self.clear_filters_btn.clicked.connect(self.clear_filters)
		self.filter_status_label = QtWidgets.QLabel("")
		self.filter_status_label.setStyleSheet("color: #7a7a7a;")

		filter_header_row.addWidget(self.filter_toggle_btn)
		filter_header_row.addWidget(filter_header_label)
		filter_header_row.addStretch(1)
		filter_header_row.addWidget(self.filter_status_label)
		filter_header_row.addWidget(self.clear_filters_btn)
		filter_header_row.addWidget(self.add_filter_btn)
		filter_group_layout.addLayout(filter_header_row)

		self.filters_container = QtWidgets.QWidget()
		self.filters_layout = QtWidgets.QVBoxLayout(self.filters_container)
		self.filters_layout.setContentsMargins(0, 0, 0, 0)
		self.filters_layout.setSpacing(4)
		filter_group_layout.addWidget(self.filters_container)

		self.count_label = QtWidgets.QLabel("")
		self.count_label.setStyleSheet("color: #7a7a7a; font-size: 11px;")
		filter_group_layout.addWidget(self.count_label)

		layout.addWidget(filter_group)

		# Table
		self.table_model = MetaTableModel(self._meta)
		self.table = QtWidgets.QTableView()
		self.table.setModel(self.table_model)
		self.table.selectionModel().selectionChanged.connect(self.update_selection_info)
		self.table.setSortingEnabled(True)
		self.table.setAlternatingRowColors(True)
		self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
		self.table.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
		self.table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
		self.table.horizontalHeader().setStretchLastSection(False)
		self.table.horizontalHeader().setSectionsMovable(True)
		self.table.horizontalHeader().setSectionsClickable(True)
		self.table.verticalHeader().setVisible(False)
		self.table.setHorizontalScrollMode(QtWidgets.QAbstractItemView.ScrollPerPixel)
		self.table.setVerticalScrollMode(QtWidgets.QAbstractItemView.ScrollPerPixel)
		layout.addWidget(self.table, 1)

		# Footer buttons and selection mode
		buttons_row = QtWidgets.QHBoxLayout()
		self.selection_mode_combo = QtWidgets.QComboBox()
		self.selection_mode_combo.addItems(["Filtered rows", "Highlighted rows"])
		self.selection_mode_combo.setToolTip(
			"Choose whether actions apply to the filtered result set or currently highlighted table rows."
		)
		self.selection_mode_combo.currentTextChanged.connect(self.update_selection_info)

		self.selection_info_label = QtWidgets.QLabel("")
		self.selection_info_label.setStyleSheet("color: #7a7a7a;")
		buttons_row.addWidget(self.selection_mode_combo)
		buttons_row.addWidget(self.selection_info_label)
		buttons_row.addStretch(1)

		save_icon = self.style().standardIcon(QtWidgets.QStyle.SP_DialogSaveButton)

		self.copy_combo = QtWidgets.QComboBox()
		self.copy_combo.addItems(["To Clipboard", "Selected rows", "IDs only"])
		self.copy_combo.setItemDelegate(DescriptorComboDelegate(self.copy_combo))
		self.copy_combo.setToolTip("Choose what to copy to the clipboard.")
		self.copy_combo.currentTextChanged.connect(self._on_copy_selection)

		self.save_btn = QtWidgets.QPushButton(save_icon, "Save")
		self.save_btn.setToolTip("Save selected results to CSV")
		self.save_btn.clicked.connect(self._on_save)

		self.select_btn = QtWidgets.QPushButton("Select in Main Window")
		self.select_btn.clicked.connect(self._on_select)
		self.open_new_btn = QtWidgets.QPushButton("Open as New Project")
		self.open_new_btn.clicked.connect(self._on_open_new)
		self.close_btn = QtWidgets.QPushButton("Close")
		self.close_btn.clicked.connect(self.close)

		buttons_row.addWidget(self.copy_combo)
		buttons_row.addWidget(self.save_btn)
		buttons_row.addWidget(self.select_btn)
		buttons_row.addWidget(self.open_new_btn)
		buttons_row.addWidget(self.close_btn)
		layout.addLayout(buttons_row)

	def add_filter_row(self):
		row = ColumnFilterRow(self._meta.columns)
		row.changed.connect(self.apply_filters)
		row.removeRequested.connect(self._remove_filter_row)
		self._filter_rows.append(row)
		self.filters_layout.addWidget(row)
		self.apply_filters()

	def _remove_filter_row(self, row):
		if row not in self._filter_rows:
			return
		self._filter_rows.remove(row)
		self.filters_layout.removeWidget(row)
		row.deleteLater()
		self.apply_filters()

	def clear_filters(self):
		for row in list(self._filter_rows):
			self._remove_filter_row(row)
		self.add_filter_row()

	def _series_as_lower_str(self, col):
		if col not in self._string_cache:
			self._string_cache[col] = self._meta[col].astype(str).str.lower()
		return self._string_cache[col]

	def _series_as_numeric(self, col):
		if col not in self._numeric_cache:
			self._numeric_cache[col] = pd.to_numeric(self._meta[col], errors="coerce")
		return self._numeric_cache[col]

	def _apply_filter_spec(self, mask, spec, row_widget):
		col = spec["column"]
		op = spec["op"]
		value = (spec["value"] or "").strip()

		series = self._meta[col]

		if op == "is empty":
			return mask & (series.isna() | (series.astype(str).str.strip() == ""))
		if op == "is not empty":
			return mask & ~(series.isna() | (series.astype(str).str.strip() == ""))

		if not value:
			return mask

		if op in (">", ">=", "<", "<="):
			try:
				value_num = float(value)
			except ValueError:
				row_widget.value_edit.setStyleSheet(
					"QLineEdit { border: 1px solid #cc3a3a; border-radius: 3px; }"
				)
				return mask

			vals = self._series_as_numeric(col)
			if op == ">":
				return mask & (vals > value_num).fillna(False)
			if op == ">=":
				return mask & (vals >= value_num).fillna(False)
			if op == "<":
				return mask & (vals < value_num).fillna(False)
			return mask & (vals <= value_num).fillna(False)

		low_value = value.lower()
		vals = self._series_as_lower_str(col)

		if op == "contains":
			return mask & vals.str.contains(low_value, na=False, regex=False)
		if op == "equals":
			return mask & (vals == low_value)
		if op == "not equals":
			return mask & (vals != low_value)

		return mask

	def apply_filters(self):
		mask = np.ones(len(self._meta), dtype=bool)

		active_filters = 0
		for row in self._filter_rows:
			spec = row.filter_spec()
			if spec["op"] not in ("is empty", "is not empty") and not spec["value"].strip():
				row.value_edit.setStyleSheet("")
				continue

			active_filters += 1
			mask = self._apply_filter_spec(mask, spec, row)

		rows = np.flatnonzero(mask)
		self.table_model.set_row_positions(rows)

		self.count_label.setText(f"Showing {len(rows):,} / {len(self._meta):,} rows")
		self.filter_status_label.setText(f"Active filters: {active_filters}")
		self.update_selection_info()

	def filtered_row_positions(self):
		return self.table_model.row_positions

	def selected_row_positions(self):
		mode = self.selection_mode_combo.currentText()
		if mode == "Highlighted rows":
			selected = self.table.selectionModel().selectedRows()
			if not selected:
				return np.array([], dtype=np.int64)
			rows = np.array([idx.row() for idx in selected], dtype=np.int64)
			return self.table_model.row_positions[np.unique(rows)]
		return self.filtered_row_positions()

	def filtered_ids(self):
		rows = self.filtered_row_positions()
		if "id" in self._meta.columns:
			return self._meta.iloc[rows]["id"].tolist()
		return self._meta.index[rows].tolist()

	def selected_ids(self):
		rows = self.selected_row_positions()
		if "id" in self._meta.columns:
			return self._meta.iloc[rows]["id"].tolist()
		return self._meta.index[rows].tolist()

	def update_selection_info(self):
		mode = self.selection_mode_combo.currentText()
		rows = self.selected_row_positions()
		self.selection_info_label.setText(f"Selected rows: {len(rows):,} ({mode})")

	def _toggle_filter_panel(self, checked):
		self.filters_container.setVisible(checked)
		self.count_label.setVisible(checked)
		self.filter_toggle_btn.setArrowType(Qt.DownArrow if checked else Qt.RightArrow)

	def _on_copy_selection(self, mode):
		if mode == "To Clipboard":
			return
		rows = self.selected_row_positions()
		if len(rows) == 0:
			self.copy_combo.blockSignals(True)
			self.copy_combo.setCurrentIndex(0)
			self.copy_combo.blockSignals(False)
			return
		if mode == "IDs only":
			ids = self.selected_ids()
			text = "\n".join(str(i) for i in ids)
		else:
			text = self._meta.iloc[rows].to_csv(sep="\t", index=False)
		QtWidgets.QApplication.clipboard().setText(text)
		self.copy_combo.blockSignals(True)
		self.copy_combo.setCurrentIndex(0)
		self.copy_combo.blockSignals(False)

	def _on_save(self):
		path, _ = QtWidgets.QFileDialog.getSaveFileName(
			self,
			"Save selected metadata",
			"selected_meta.csv",
			"CSV Files (*.csv);;All Files (*)",
		)
		if not path:
			return
		rows = self.selected_row_positions()
		self._meta.iloc[rows].to_csv(path, index=False)

	def _on_select(self):
		ids = self.selected_ids()
		if self._figure is not None and hasattr(self._figure, "select"):
			try:
				self._figure.select(ids)
				return
			except Exception:
				pass
		self.selectRequested.emit(ids)

	def _on_open_new(self):
		ids = self.selected_ids()
		if self._figure is not None:
			for method_name in ("open_selection_in_new_window", "open_in_new_window", "open_new_window"):
				if hasattr(self._figure, method_name):
					try:
						getattr(self._figure, method_name)(ids)
						return
					except Exception:
						pass
		self.openInNewWindowRequested.emit(ids)


if __name__ == "__main__":
	import sys

	logging.basicConfig(level=logging.INFO)

	test_path = (
		"/Users/philipps/Downloads/"
		"MaleCNS_FlyWire_hemibrain_central_brain_bigclust/meta.feather"
	)

	app = QtWidgets.QApplication(sys.argv)

	try:
		meta = pd.read_feather(test_path)
	except Exception as exc:
		msg = QtWidgets.QMessageBox()
		msg.setIcon(QtWidgets.QMessageBox.Critical)
		msg.setWindowTitle("Meta Explorer")
		msg.setText("Failed to load test metadata")
		msg.setInformativeText(str(exc))
		msg.exec()
		raise

	dlg = MetaExplorerDialog(meta)

	def _print_select(ids):
		logger.info("Select clicked with %d filtered neurons", len(ids))

	def _print_open(ids):
		logger.info("Open in New Window clicked with %d filtered neurons", len(ids))

	dlg.selectRequested.connect(_print_select)
	dlg.openInNewWindowRequested.connect(_print_open)

	dlg.show()
	sys.exit(app.exec())
