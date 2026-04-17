import numpy as np
import pandas as pd

from numbers import Number
from PySide6 import QtWidgets, QtCore, QtGui
from PySide6.QtCore import Qt
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import squareform


class RotatedHeaderView(QtWidgets.QHeaderView):
    """Horizontal header that renders section labels rotated by 90 degrees."""

    def __init__(self, orientation, parent=None):
        super().__init__(orientation, parent)
        self.setDefaultAlignment(Qt.AlignCenter)
        self.setMinimumHeight(130)

    def paintSection(self, painter, rect, logical_index):
        if not rect.isValid():
            return

        option = QtWidgets.QStyleOptionHeader()
        self.initStyleOption(option)
        option.rect = rect
        option.section = logical_index
        option.text = ""

        self.style().drawControl(QtWidgets.QStyle.CE_Header, option, painter, self)

        model = self.model()
        if model is None:
            return

        text = model.headerData(logical_index, self.orientation(), Qt.DisplayRole)
        if text is None:
            return

        painter.save()
        painter.setFont(self.font())
        painter.setPen(option.palette.color(QtGui.QPalette.ButtonText))
        painter.translate(rect.left(), rect.bottom())
        painter.rotate(-90)
        text_rect = QtCore.QRect(0, 0, rect.height(), rect.width())
        text_rect = text_rect.adjusted(2, 2, -2, -2)
        painter.setClipRect(text_rect)
        painter.drawText(
            text_rect,
            Qt.AlignHCenter | Qt.AlignVCenter | Qt.TextWordWrap,
            str(text),
        )
        painter.restore()

    def sectionSizeFromContents(self, logical_index):
        """Return a size hint that accounts for the 90-degree label rotation."""
        size = super().sectionSizeFromContents(logical_index)

        # For rotated horizontal headers, text length should contribute to height,
        # while width can stay compact and driven mostly by cell content.
        if self.orientation() == Qt.Horizontal:
            return QtCore.QSize(size.height(), size.width())
        return size


class TableModel(QtCore.QAbstractTableModel):
    def __init__(self, data, meta_data):
        super(TableModel, self).__init__()
        assert isinstance(data, pd.DataFrame), "data must be a pandas DataFrame"
        self._data = data
        self._meta_data = meta_data
        self._view = self._data.iloc[0:0, 0:0]
        self._selected_ids = self._data.index[:0]
        self._row_labels = "ID"
        self._ordering = "Linkage"
        self._hide_diagonal = False
        self._hide_upper_triangle = True
        self._color_cells = True
        self._decimals = 2
        self._min_color = QtGui.QColor("#0b2a6f")
        self._max_color = QtGui.QColor("#ffffff")
        self.update_indices()

    def data(self, index, role):
        if role == Qt.DisplayRole:
            if self._is_hidden_upper_triangle(index):
                return ""

            row_key = self._view.index[index.row()]
            col_key = self._view.columns[index.column()]
            if self._hide_diagonal and row_key == col_key:
                return ""

            value = self._view.values[index.row(), index.column()]
            if pd.isna(value):
                return ""

            if isinstance(value, Number):
                return f"{float(value):.{self._decimals}f}"
            return str(value)

        elif role == Qt.TextAlignmentRole:
            return Qt.AlignHCenter | Qt.AlignVCenter

        elif role == Qt.ItemDataRole.ForegroundRole:
            if self._is_hidden_upper_triangle(index):
                return None

            row_key = self._view.index[index.row()]
            col_key = self._view.columns[index.column()]
            if self._hide_diagonal and row_key == col_key:
                return None

            value = self._view.values[index.row(), index.column()]
            if not isinstance(value, Number) or pd.isna(value):
                return None

            background = self._background_color_for_index(index)
            if background is None:
                return None

            # Perceived luminance threshold for black vs white text.
            luminance = (
                0.299 * background.red()
                + 0.587 * background.green()
                + 0.114 * background.blue()
            )
            if luminance >= 170:
                return QtGui.QBrush(QtGui.QColor("black"))
            return QtGui.QBrush(QtGui.QColor("white"))

        elif role == Qt.ItemDataRole.BackgroundRole and self._color_cells:
            return self._background_color_for_index(index)

        elif role == Qt.ToolTipRole:
            if self._is_hidden_upper_triangle(index):
                return None

            row_key = self._view.index[index.row()]
            col_key = self._view.columns[index.column()]
            if self._hide_diagonal and row_key == col_key:
                return None

            value = self._view.values[index.row(), index.column()]
            if pd.isna(value):
                return None

            row_label = str(self._indices[index.row()])
            col_label = str(self._columns[index.column()])
            if isinstance(value, Number):
                return f"{row_label} × {col_label}: {float(value):.6g}"
            return f"{row_label} × {col_label}: {value}"

        return None

    def _interpolate_color(self, t):
        """Interpolate between the minimum and maximum heatmap colors."""
        t = float(np.clip(t, 0.0, 1.0))
        start = self._min_color
        end = self._max_color
        red = round(start.red() + (end.red() - start.red()) * t)
        green = round(start.green() + (end.green() - start.green()) * t)
        blue = round(start.blue() + (end.blue() - start.blue()) * t)
        return QtGui.QColor(red, green, blue)

    def _background_color_for_index(self, index):
        if not self._color_cells or self._is_hidden_upper_triangle(index):
            return None

        row_key = self._view.index[index.row()]
        col_key = self._view.columns[index.column()]
        if self._hide_diagonal and row_key == col_key:
            return None

        value = self._view.values[index.row(), index.column()]
        if not isinstance(value, Number) or pd.isna(value):
            return None

        vals = self._view.values.astype(float)
        if self._hide_upper_triangle and vals.shape[0] == vals.shape[1]:
            vals = vals.copy()
            vals[np.triu_indices_from(vals, k=1)] = np.nan
        if self._hide_diagonal and vals.shape[0] == vals.shape[1]:
            vals = vals.copy()
            np.fill_diagonal(vals, np.nan)

        vmin = 0.0
        vmax = np.nanmax(vals)
        if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
            return None

        value_norm = (float(value) - vmin) / (vmax - vmin)
        return self._interpolate_color(value_norm)

    def _is_hidden_upper_triangle(self, index):
        return self._hide_upper_triangle and index.column() > index.row()

    def rowCount(self, index):
        return len(self._view)

    def columnCount(self, index):
        return self._view.shape[1]

    def headerData(self, section, orientation, role):
        if role == Qt.DisplayRole:
            if orientation == Qt.Horizontal:
                return str(self._columns[section])
            if orientation == Qt.Vertical:
                return str(self._indices[section])

        if role == Qt.ToolTipRole:
            if orientation == Qt.Horizontal:
                return str(self._view.columns[section])
            return str(self._view.index[section])

        return None

    def select_rows(self, indices, use_index=True):
        """Select rows by IDs or by integer positions."""
        if use_index:
            selected = pd.Index(indices)
            self._selected_ids = self._data.index[self._data.index.isin(selected)]
        else:
            self._selected_ids = self._data.iloc[indices].index

        if len(self._selected_ids):
            self._view = self._data.loc[self._selected_ids, self._selected_ids]
        else:
            self._view = self._data.iloc[0:0, 0:0]

        self._apply_ordering()

        self.update_indices()
        self.layoutChanged.emit()

    def set_row_labels(self, col):
        self._row_labels = col
        self.select_rows(self._selected_ids)

    def set_ordering(self, ordering):
        self._ordering = ordering
        self.select_rows(self._selected_ids)

    def set_hide_diagonal(self, hide_diagonal):
        self._hide_diagonal = hide_diagonal
        self.layoutChanged.emit()

    def set_hide_upper_triangle(self, hide_upper_triangle):
        self._hide_upper_triangle = hide_upper_triangle
        self.layoutChanged.emit()

    def set_color_cells(self, color_cells):
        self._color_cells = color_cells
        self.layoutChanged.emit()

    def set_decimals(self, decimals):
        self._decimals = max(0, int(decimals))
        self.layoutChanged.emit()

    def update_indices(self):
        """Update the indices and columns according to the current view."""
        if self._row_labels == "ID":
            self._indices = self._view.index.astype(str)
            self._columns = self._view.columns.astype(str)
            return

        row_labels = self._meta_data[self._row_labels].loc[self._view.index].astype(str)
        col_labels = self._meta_data[self._row_labels].loc[self._view.columns].astype(str)
        self._indices = row_labels.values
        self._columns = col_labels.values

    def _apply_ordering(self):
        """Apply matrix ordering to rows/columns in lockstep."""
        if self._view.empty or self._view.shape[0] <= 1:
            return

        if self._ordering == "None":
            return

        if self._ordering == "Label":
            if self._row_labels == "ID":
                labels = self._view.index.astype(str).values
            else:
                labels = (
                    self._meta_data.loc[self._view.index, self._row_labels]
                    .astype(str)
                    .values
                )
            order = np.argsort(labels, kind="stable")
            self._view = self._view.iloc[order, order]
            return

        if self._ordering == "Linkage":
            try:
                d = self._view.values.astype(float)
                d = np.nan_to_num(d, nan=0.0, posinf=0.0, neginf=0.0)
                np.fill_diagonal(d, 0.0)
                condensed = squareform(d, checks=False)
                order = leaves_list(linkage(condensed, method="ward"))
                self._view = self._view.iloc[order, order]
            except Exception:
                # Keep current order if clustering fails for this matrix.
                return
            return

        raise ValueError(f"Unknown ordering mode: {self._ordering}")


class DistancesTable(QtWidgets.QWidget):
    """Widget to display a pairwise distance heatmap as a table."""

    def __init__(
        self,
        distances,
        meta_data,
        figure=None,
        width=720,
        height=520,
        title="Distance heatmap",
        parent=None,
    ):
        super().__init__(parent, Qt.Window)

        self._figure = figure
        self._meta_data = meta_data.copy()
        self._data = self._prepare_distances(distances, self._meta_data)
        self._meta_data = self._meta_data.set_index("id").loc[self._data.index].copy()

        self.setWindowTitle(title)
        self.resize(width, height)

        self._layout = QtWidgets.QHBoxLayout()
        self._layout.setContentsMargins(4, 4, 4, 4)
        self._layout.setSpacing(6)
        self.setLayout(self._layout)

        self._control_panel = QtWidgets.QWidget()
        self._control_panel.setMinimumWidth(240)
        self._control_panel.setMaximumWidth(360)
        control_layout = QtWidgets.QVBoxLayout()
        control_layout.setContentsMargins(0, 0, 0, 0)
        control_layout.setSpacing(4)
        self._control_panel.setLayout(control_layout)

        self._layout.addWidget(self._control_panel)

        self._table = QtWidgets.QTableView()
        self._table.setHorizontalHeader(RotatedHeaderView(Qt.Horizontal, self._table))
        self._table.horizontalHeader().setDefaultAlignment(Qt.AlignHCenter | Qt.AlignVCenter)
        self._table.verticalHeader().setDefaultAlignment(Qt.AlignHCenter | Qt.AlignVCenter)
        self._model = TableModel(self._data, self._meta_data)
        self._table.setModel(self._model)

        self._base_table_font = QtGui.QFont(self._table.font())
        self._base_horizontal_header_font = QtGui.QFont(
            self._table.horizontalHeader().font()
        )
        self._base_vertical_header_font = QtGui.QFont(
            self._table.verticalHeader().font()
        )
        self._base_rotated_header_min_height = self._table.horizontalHeader().minimumHeight()

        self._table_scale = 100
        self._table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Interactive)
        self._table.verticalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Interactive)

        self._layout.addWidget(self._table, 1)

        self._table.horizontalHeader().sectionDoubleClicked.connect(self.find_header)
        self._table.verticalHeader().sectionDoubleClicked.connect(self.find_index)

        controls_group = QtWidgets.QGroupBox("Display")
        controls_form = QtWidgets.QFormLayout()
        controls_form.setContentsMargins(6, 6, 6, 6)
        controls_form.setVerticalSpacing(4)
        controls_form.setHorizontalSpacing(6)
        controls_group.setLayout(controls_form)

        self._row_label_dropdown = QtWidgets.QComboBox()
        self._row_label_dropdown.addItems(["ID"] + list(self._meta_data.columns))
        self._row_label_dropdown.currentIndexChanged.connect(self.update_row_labels)
        controls_form.addRow("Labels:", self._row_label_dropdown)

        self._ordering_dropdown = QtWidgets.QComboBox()
        self._ordering_dropdown.addItems(["None", "Label", "Linkage"])
        self._ordering_dropdown.setCurrentText("Linkage")
        self._ordering_dropdown.setToolTip("Order rows/columns together")
        self._ordering_dropdown.currentIndexChanged.connect(self.update_ordering)
        controls_form.addRow("Ordering:", self._ordering_dropdown)

        self._hide_diagonal = QtWidgets.QCheckBox("Hide diagonal")
        self._hide_diagonal.stateChanged.connect(self.update_hide_diagonal)
        self._hide_diagonal.setChecked(True)
        controls_form.addRow(self._hide_diagonal)

        self._hide_upper_triangle = QtWidgets.QCheckBox("Hide upper triangle")
        self._hide_upper_triangle.stateChanged.connect(self.update_hide_upper_triangle)
        self._hide_upper_triangle.setChecked(True)
        controls_form.addRow(self._hide_upper_triangle)

        self._color_cells = QtWidgets.QCheckBox("Color cells")
        self._color_cells.setChecked(True)
        self._color_cells.stateChanged.connect(self.update_color_cells)
        controls_form.addRow(self._color_cells)

        self._decimals = QtWidgets.QSpinBox()
        self._decimals.setRange(0, 8)
        self._decimals.setSingleStep(1)
        self._decimals.setValue(2)
        self._decimals.setToolTip("Number of decimals shown in each heatmap cell")
        self._decimals.valueChanged.connect(self.update_decimals)
        controls_form.addRow("Decimals:", self._decimals)

        self._cell_size = QtWidgets.QSpinBox()
        self._cell_size.setRange(25, 200)
        self._cell_size.setSingleStep(5)
        self._cell_size.setValue(self._table_scale)
        self._cell_size.setSuffix("%")
        self._cell_size.setToolTip("Scale table fonts and cell size relative to content")
        self._cell_size.valueChanged.connect(self.update_cell_size)
        controls_form.addRow("Scale:", self._cell_size)

        self._model.layoutChanged.connect(self.update_cell_size)

        self._always_on_top = QtWidgets.QCheckBox("Always on top")
        self._always_on_top.setToolTip("Keep this window above other BigClust windows")
        self._always_on_top.stateChanged.connect(self.update_always_on_top)
        self._always_on_top.setChecked(True)
        controls_form.addRow(self._always_on_top)

        self._copy_button = QtWidgets.QPushButton("Copy table to clipboard")
        self._copy_button.clicked.connect(self.copy_to_clipboard)

        control_layout.addWidget(controls_group)
        control_layout.addStretch()
        control_layout.addWidget(self._copy_button)

        if self._figure is not None and self._figure.selected_ids is not None:
            if len(self._figure.selected_ids) > 0:
                self.select(self._figure.selected_ids)

        self.update_cell_size()

    @staticmethod
    def _prepare_distances(distances, meta_data):
        if isinstance(distances, pd.DataFrame):
            return distances

        if not isinstance(distances, np.ndarray):
            raise ValueError("distances must be a pandas DataFrame or numpy array")

        if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
            raise ValueError("distances array must have shape (N, N)")

        if "id" not in meta_data.columns:
            raise ValueError("meta_data must contain an 'id' column for ndarray distances")

        ids = meta_data["id"].values
        if len(ids) != distances.shape[0]:
            raise ValueError("meta_data length does not match distances shape")

        return pd.DataFrame(distances, index=ids, columns=ids)

    def update_row_labels(self, *args, **kwargs):
        self._model.set_row_labels(self._row_label_dropdown.currentText())

    def update_hide_diagonal(self, *args, **kwargs):
        self._model.set_hide_diagonal(self._hide_diagonal.isChecked())

    def update_hide_upper_triangle(self, *args, **kwargs):
        self._model.set_hide_upper_triangle(self._hide_upper_triangle.isChecked())

    def update_ordering(self, *args, **kwargs):
        self._model.set_ordering(self._ordering_dropdown.currentText())

    def update_color_cells(self, *args, **kwargs):
        self._model.set_color_cells(self._color_cells.isChecked())

    def update_decimals(self, *args, **kwargs):
        self._model.set_decimals(self._decimals.value())

    def update_cell_size(self, *args, **kwargs):
        if hasattr(self, "_cell_size"):
            self._table_scale = self._cell_size.value()

        scale = self._table_scale / 100.0

        table_font = QtGui.QFont(self._base_table_font)
        table_font.setPointSizeF(max(6.0, self._base_table_font.pointSizeF() * scale))
        self._table.setFont(table_font)

        h_header_font = QtGui.QFont(self._base_horizontal_header_font)
        h_header_font.setPointSizeF(
            max(6.0, self._base_horizontal_header_font.pointSizeF() * scale)
        )
        self._table.horizontalHeader().setFont(h_header_font)

        v_header_font = QtGui.QFont(self._base_vertical_header_font)
        v_header_font.setPointSizeF(
            max(6.0, self._base_vertical_header_font.pointSizeF() * scale)
        )
        self._table.verticalHeader().setFont(v_header_font)

        rotated_header_min_height = max(
            60, int(round(self._base_rotated_header_min_height * scale))
        )
        self._table.horizontalHeader().setMinimumHeight(rotated_header_min_height)

        self._table.resizeColumnsToContents()
        self._table.resizeRowsToContents()

    def keyPressEvent(self, event):
        if event.modifiers() & Qt.ControlModifier:
            if event.key() in (Qt.Key_Plus, Qt.Key_Equal):
                self._cell_size.setValue(self._cell_size.value() + self._cell_size.singleStep())
                event.accept()
                return
            if event.key() == Qt.Key_Minus:
                self._cell_size.setValue(self._cell_size.value() - self._cell_size.singleStep())
                event.accept()
                return
        super().keyPressEvent(event)

    def update_always_on_top(self, *args, **kwargs):
        self.setWindowFlag(Qt.Tool, self._always_on_top.isChecked())
        self.setWindowFlag(Qt.WindowStaysOnTopHint, False)
        self.show()

    def select(self, indices):
        self._model.select_rows(indices, use_index=False)

    def find_header(self):
        curr_col = self._table.currentIndex().column()
        if curr_col < 0:
            return
        label = self._model._view.columns[curr_col]
        if self._figure:
            self._figure.find_label(label, regex=True)

    def find_index(self):
        curr_row = self._table.currentIndex().row()
        if curr_row < 0:
            return
        label = self._model._view.index[curr_row]
        if self._figure:
            self._figure.find_label(label, regex=True)

    def copy_to_clipboard(self):
        if self._model._view.shape[0] > 500:
            raise ValueError("Too many rows to copy to clipboard.")
        self._model._view.to_clipboard(index=True)
