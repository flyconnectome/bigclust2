import cmap

import numpy as np
import pandas as pd
import pyqtgraph as pg

from numbers import Number
from PySide6 import QtWidgets, QtCore, QtGui
from PySide6.QtCore import Qt
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import linkage, leaves_list

from ...utils import rgb_from_segment_id

# See here for a good tutorial on tables:
# https://www.pythonguis.com/tutorials/pyside6-qtableview-modelviews-numpy-pandas/

# Activate antialiasing for better graphics
pg.setConfigOptions(antialias=True)

# Shorten the upstream/downstream labels for display
SHORT = {"upstream": "up", "downstream": "ds"}


def trigger_graph_update(func):
    """Decorator to update the graph after the function is called."""

    def wrapper(self, *args, **kwargs):
        func(self, *args, **kwargs)

        if hasattr(self, "_graph_widget"):
            self.update_graph()

    return wrapper


class TableModel(QtCore.QAbstractTableModel):
    def __init__(self, data, meta_data):
        super(TableModel, self).__init__()
        assert isinstance(data, pd.DataFrame), "data must be a pandas DataFrame"
        self._data = data
        self._meta_data = meta_data
        self._view = self._data.iloc[0:0, 0:0]  # start with an empty view
        self._selected_ids = self._data.index[:0]
        self._row_labels = "ID"  # Default row labels
        self._collapse_rows = False
        self.update_indices()  # pre-compute the indices and columns
        self._hide_zeros = True
        self._synapse_threshold = 1
        self._col_sort = None
        self._row_sort = None
        self._col_filt = None
        self._row_filt = None
        self._upstream = True
        self._downstream = True
        self._color_cells = True
        self._normalize = False
        self.colormap = cmap.Colormap("matlab:cool", interpolation="linear")

    def data(self, index, role):
        if role == Qt.DisplayRole:
            # Get the current value
            value = self._view.values[index.row(), index.column()]
            if self._hide_zeros and value == 0:
                return ""
            else:
                if self._normalize:
                    return f"{value:.3f}"
                return str(value)
        elif role == Qt.TextAlignmentRole:
            # Center the text in both vertical and horizontal directions
            return Qt.AlignCenter + Qt.AlignVCenter
        elif role == Qt.ItemDataRole.BackgroundRole and self._color_cells:
            # Get the current value
            value = self._view.values[index.row(), index.column()]
            if isinstance(value, Number) and value > 0:
                # Normalise to range 0 - 1
                value_norm = value / self._view.values.max()

                # Generate a color
                c = self.colormap(value_norm).hex

                return QtGui.QColor(c)

    def rowCount(self, index):
        # The length of the outer list.
        return len(self._view)

    def columnCount(self, index):
        # The following takes the first sub-list, and returns
        # the length (only works if all rows are an equal length)
        return self._view.shape[1]

    def headerData(self, section, orientation, role):
        # section is the index of the column/row.
        if role == Qt.DisplayRole:
            if orientation == Qt.Horizontal:
                return str(self._columns[section])

            if orientation == Qt.Vertical:
                return str(self._indices[section])

        if role == Qt.ToolTipRole:
            if orientation == Qt.Horizontal:
                return str(self._view.columns[section])
            else:
                return str(self._view.index[section])

    def select_rows(self, indices, drop_empty_cols=True, use_index=True):
        """Select rows by indices."""
        if use_index:
            self._selected_ids = self._data.loc[indices].index
            self._view = _view_temp = self._data.loc[self._selected_ids]
        else:
            self._selected_ids = self._data.iloc[indices].index
            self._view = _view_temp = self._data.loc[self._selected_ids]

        if isinstance(self._view.columns, pd.MultiIndex):
            if not self._upstream:
                self._view = self._view.drop(columns=["upstream"])
            if not self._downstream:
                self._view = self._view.drop(columns=["downstream"])

        if self._collapse_rows and self._row_labels != "ID" and not self._view.empty:
            row_labels = self._meta_data.loc[self._view.index, self._row_labels].astype(str)
            self._view = self._view.groupby(row_labels, sort=False).sum()
            _view_temp = _view_temp.groupby(row_labels, sort=False).sum()

        # Apply synapse threshold
        if self._synapse_threshold:
            self._view = self._view.iloc[
                :, self._view.max(axis=0).values >= self._synapse_threshold
            ]

        # Apply column filter
        if self._col_filt:
            self._view = self._view.filter(regex=self._col_filt, axis=1)

        # Apply row filter
        if self._row_filt:
            self._view = self._view.filter(regex=self._row_filt, axis=0)

        # Apply normalisation
        if self._normalize:
            self._view = self._view.astype(float)
            _view_temp = _view_temp.astype(float)

            # Normalise up- and downstream separately
            if isinstance(self._view.columns, pd.MultiIndex):
                for d in ("upstream", "downstream"):
                    this_level = self._view.columns.get_level_values(0) == d
                    this_level_full = _view_temp.columns.get_level_values(0) == d
                    denom = _view_temp.loc[:, this_level_full].sum(axis=1)
                    denom = denom.replace(0, np.nan)
                    self._view.loc[:, this_level] = (
                        self._view.loc[:, this_level]
                        .div(denom, axis=0)
                        .fillna(0)
                    )
            else:
                denom = _view_temp.sum(axis=1).replace(0, np.nan)
                self._view = self._view.div(denom, axis=0).fillna(0)

        # Apply column sort
        if self._col_sort not in (None, "No sort") and self._view.shape[1] > 1:
            if self._col_sort == "By synapse count":
                srt = np.argsort(self._view.sum(axis=0).values)[::-1]
            elif self._col_sort == "By label":
                if isinstance(self._view.columns, pd.MultiIndex):
                    # Ignore upstream/downstream and sort by label
                    _, srt = self._view.columns.sortlevel(1)
                else:
                    srt = np.argsort(self._view.columns)
            elif self._col_sort == "By distance":
                # Sort by Euclidean distance
                d = pdist(self._view.T)
                srt = leaves_list(linkage(d, method="ward"))
            else:
                raise ValueError(f"Unknown sort order: {self._col_sort}")

            self._view = self._view.iloc[:, srt]

        # Apply row sort
        if self._row_sort not in (None, "No sort") and self._view.shape[0] > 1:
            if self._row_sort == "By label":
                srt = np.argsort(self._view.index)
            elif self._row_sort == "By distance":
                # Sort by Euclidean distance
                d = pdist(self._view.values)
                srt = leaves_list(linkage(d, method="ward"))
            else:
                raise ValueError(f"Unknown sort order: {self._row_sort}")

            self._view = self._view.iloc[srt, :]

        self.update_indices()

        # Emit signal to trigger update
        # This is where 99.999% of time is spent
        self.layoutChanged.emit()

    def set_synapse_threshold(self, threshold):
        """Set the synapse threshold."""
        self._synapse_threshold = threshold
        self.select_rows(self._selected_ids)  # reselect rows

    def set_hide_zeros(self, hide_zeros):
        """Set whether to hide zeros."""
        self._hide_zeros = hide_zeros

        # Emit signal to trigger update
        self.layoutChanged.emit()

    def set_col_sort(self, sort):
        """Set the column sort."""
        self._col_sort = sort
        self.select_rows(self._selected_ids)  # reselect rows

    def set_row_sort(self, sort):
        """Set the rows sort."""
        self._row_sort = sort
        self.select_rows(self._selected_ids)  # reselect rows

    def set_filter_columns(self, filt):
        """Set the column filter."""
        self._col_filt = filt
        self.select_rows(self._selected_ids)

    def set_filter_rows(self, filt):
        """Set the row filter."""
        self._row_filt = filt
        self.select_rows(self._selected_ids)

    def set_row_labels(self, col):
        self._row_labels = col
        self.select_rows(self._selected_ids)

    def set_direction(self, upstream=True, downstream=True):
        """Set the direction of the table."""
        self._upstream = upstream
        self._downstream = downstream
        self.select_rows(self._selected_ids)

    def set_collapse_rows(self, collapse_rows):
        """Collapse rows by the active row label and sum their values."""
        self._collapse_rows = collapse_rows
        self.select_rows(self._selected_ids)

    def update_indices(self):
        """Update the indices and columns according to the current view."""

        def fmt(i):
            return f"{SHORT.get(i[0], i[0])}:{i[1]}"

        if self._collapse_rows and self._row_labels != "ID":
            self._indices = self._view.index.astype(str)
        elif self._row_labels == "ID":
            if isinstance(self._view.index, pd.MultiIndex):
                self._indices = [fmt(i) for i in self._view.index.to_flat_index()]
            else:
                self._indices = self._view.index.astype(str)
        else:
            self._indices = (
                self._meta_data[self._row_labels]
                .loc[self._view.index]
                .astype(str)
                .values
            )

        if isinstance(self._view.columns, pd.MultiIndex):
            self._columns = [fmt(i) for i in self._view.columns.to_flat_index()]
        else:
            self._columns = self._view.columns.astype(str)


class ConnectivityTable(QtWidgets.QWidget):
    """A widget to display a table of connectivity data.

    Parameters
    ----------
    data : pd.DataFrame
        The connectivity data to display.
    figure : Dendrogram, optional
        The dendrogram figure to connect to.
    width : int, optional
        The width of the widget.
    height : int, optional
        The height of the widget.

    """

    def __init__(
        self,
        data,
        meta_data,
        figure=None,
        width=600,
        height=400,
        title="Connectivity widget",
        parent=None,
    ):
        assert isinstance(data, pd.DataFrame), "data must be a pandas DataFrame"

        super().__init__(parent, Qt.Window)

        self._data = data
        self._figure = figure
        self._meta_data = meta_data
        self.setWindowTitle(title)
        self.resize(width, height)

        # Prep data
        self._meta_data = (
            self._meta_data.set_index("id").loc[self._data.index].copy()
        )  # sort to match the data & make a copy

        # Set up layout
        self._layout = QtWidgets.QHBoxLayout()
        self._layout.setContentsMargins(4, 4, 4, 4)
        self._layout.setSpacing(6)
        self.setLayout(self._layout)

        self._control_panel = QtWidgets.QWidget()
        self._control_panel.setMinimumWidth(280)
        self._control_panel.setMaximumWidth(420)
        self._control_panel.setSizePolicy(
            QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Expanding
        )
        control_layout = QtWidgets.QVBoxLayout()
        control_layout.setContentsMargins(0, 0, 0, 0)
        control_layout.setSpacing(4)
        self._control_panel.setLayout(control_layout)

        content_panel = QtWidgets.QWidget()
        content_layout = QtWidgets.QVBoxLayout()
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(0)
        content_panel.setLayout(content_layout)

        self._layout.addWidget(self._control_panel)
        self._layout.addWidget(content_panel, 1)

        self._tabs = QtWidgets.QTabWidget()
        content_layout.addWidget(self._tabs)
        # self.tabs.setDocumentMode(True)
        # self.tabs.setTabPosition(QtWidgets.QTabWidget.North)
        # self.tabs.setMovable(True)

        # Add a small corner button to show/hide the options panel.
        self._toggle_options_button = QtWidgets.QToolButton()
        self._toggle_options_button.setCheckable(True)
        self._toggle_options_button.setChecked(True)
        self._toggle_options_button.setAutoRaise(True)
        self._toggle_options_button.setToolTip("Hide options panel")
        self._toggle_options_button.setIconSize(QtCore.QSize(14, 14))
        icon = QtGui.QIcon.fromTheme("preferences-system")
        if icon.isNull():
            icon = QtGui.QIcon.fromTheme("settings")
        if not icon.isNull():
            self._toggle_options_button.setIcon(icon)
        else:
            icon = self.style().standardIcon(QtWidgets.QStyle.SP_FileDialogListView)
            self._toggle_options_button.setIcon(icon)
        self._toggle_options_button.toggled.connect(self.toggle_options_panel)
        self._tabs.setCornerWidget(self._toggle_options_button, Qt.TopRightCorner)

        # Build gui
        self._tab_table = QtWidgets.QWidget()
        self._tabs.addTab(self._tab_table, "Table")

        self._tab_table_layout = QtWidgets.QVBoxLayout()
        self._tab_table_layout.setContentsMargins(0, 0, 0, 0)
        self._tab_table_layout.setSpacing(0)
        self._tab_table.setLayout(self._tab_table_layout)

        # First up: the table
        self._table = QtWidgets.QTableView()
        self._model = TableModel(self._data, self._meta_data)
        self._table.setModel(self._model)
        self._tab_table_layout.addWidget(self._table)

        self._base_table_font = QtGui.QFont(self._table.font())
        self._base_horizontal_header_font = QtGui.QFont(
            self._table.horizontalHeader().font()
        )
        self._base_vertical_header_font = QtGui.QFont(
            self._table.verticalHeader().font()
        )

        self._table_scale = 100
        self._last_applied_font_sizes = None
        self._table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Interactive)
        self._table.verticalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Interactive)
        self._model.layoutChanged.connect(self.update_cell_size)

        # Add a double click event for header and rows
        self._table.horizontalHeader().sectionDoubleClicked.connect(self.find_header)
        self._table.verticalHeader().sectionDoubleClicked.connect(self.find_index)

        self._control_tabs = QtWidgets.QTabWidget()
        control_layout.addWidget(self._control_tabs)

        self._table_controls_tab = QtWidgets.QWidget()
        self._control_tabs.addTab(self._table_controls_tab, "Table Controls")
        table_controls_layout = QtWidgets.QVBoxLayout()
        table_controls_layout.setContentsMargins(0, 0, 0, 0)
        table_controls_layout.setSpacing(4)
        self._table_controls_tab.setLayout(table_controls_layout)

        self._display_controls_tab = QtWidgets.QWidget()
        self._control_tabs.addTab(self._display_controls_tab, "Display")
        display_controls_layout = QtWidgets.QVBoxLayout()
        display_controls_layout.setContentsMargins(0, 0, 0, 0)
        display_controls_layout.setSpacing(4)
        self._display_controls_tab.setLayout(display_controls_layout)

        rows_group = QtWidgets.QGroupBox("Rows")
        rows_form = QtWidgets.QFormLayout()
        rows_form.setContentsMargins(6, 6, 6, 6)
        rows_form.setVerticalSpacing(4)
        rows_form.setHorizontalSpacing(6)
        rows_form.setLabelAlignment(Qt.AlignLeft)
        rows_group.setLayout(rows_form)

        # Add a dropdown for row labels
        self._row_label_dropdown = QtWidgets.QComboBox()
        self._row_label_dropdown.setToolTip("Select row labels")
        self._row_label_dropdown.addItems(["ID"] + list(self._meta_data.columns))
        self._row_label_dropdown.currentIndexChanged.connect(self.update_row_labels)
        rows_form.addRow("Labels:", self._row_label_dropdown)

        self._sort_rows_dropdown = QtWidgets.QComboBox()
        self._sort_rows_dropdown.addItem("No sort")
        self._sort_rows_dropdown.addItem("By label")
        self._sort_rows_dropdown.addItem("By distance")
        self._sort_rows_dropdown.currentIndexChanged.connect(self.update_sort_rows)
        rows_form.addRow("Sort:", self._sort_rows_dropdown)

        self._row_search = QtWidgets.QLineEdit()
        self._row_search.setToolTip("Filter rows by name")
        self._row_search.setPlaceholderText("Filter rows")
        self._row_search.textChanged.connect(self.filter_rows)
        rows_form.addRow("Filter:", self._row_search)

        self._collapse_rows = QtWidgets.QCheckBox("Collapse rows by label")
        self._collapse_rows.setChecked(False)
        self._collapse_rows.setToolTip(
            "Group rows by the current row label and sum their values"
        )
        self._collapse_rows.stateChanged.connect(self.update_collapse_rows)
        rows_form.addRow(self._collapse_rows)

        table_controls_layout.addWidget(rows_group)

        cols_group = QtWidgets.QGroupBox("Columns")
        cols_form = QtWidgets.QFormLayout()
        cols_form.setContentsMargins(6, 6, 6, 6)
        cols_form.setVerticalSpacing(4)
        cols_form.setHorizontalSpacing(6)
        cols_form.setLabelAlignment(Qt.AlignLeft)
        cols_group.setLayout(cols_form)

        direction_row = QtWidgets.QHBoxLayout()
        direction_row.setContentsMargins(0, 0, 0, 0)
        direction_row.setSpacing(4)

        # Add checkboxes for up- and downstream
        self._upstream = QtWidgets.QCheckBox("Upstream")
        self._upstream.setChecked(True)
        self._upstream.setToolTip("Show upstream connections")
        self._upstream.stateChanged.connect(self.update_direction)
        direction_row.addWidget(self._upstream)

        self._downstream = QtWidgets.QCheckBox("Downstream")
        self._downstream.setChecked(True)
        self._downstream.setToolTip("Show downstream connections")
        self._downstream.stateChanged.connect(self.update_direction)
        direction_row.addWidget(self._downstream)
        direction_row.addStretch()
        cols_form.addRow("Direction:", direction_row)

        # Add a QSpinBox for the synapse threshold
        self._synapse_threshold = QtWidgets.QSpinBox()
        self._synapse_threshold.setToolTip("Set the synapse threshold")
        self._synapse_threshold.setRange(0, 1000)
        self._synapse_threshold.setValue(1)
        self._synapse_threshold.setSingleStep(1)
        self._synapse_threshold.valueChanged.connect(self.update_synapse_threshold)
        cols_form.addRow("Threshold:", self._synapse_threshold)

        self._sort_cols_dropdown = QtWidgets.QComboBox()
        self._sort_cols_dropdown.addItem("No sort")
        self._sort_cols_dropdown.addItem("By synapse count")
        self._sort_cols_dropdown.addItem("By label")
        self._sort_cols_dropdown.addItem("By distance")
        self._sort_cols_dropdown.currentIndexChanged.connect(self.update_sort_cols)
        self._sort_cols_dropdown.setCurrentIndex(1)  # default to sorting by synapse count
        cols_form.addRow("Sort:", self._sort_cols_dropdown)

        self._column_search = QtWidgets.QLineEdit()
        self._column_search.setToolTip("Filter columns by name")
        self._column_search.setPlaceholderText("Filter columns")
        self._column_search.textChanged.connect(self.filter_columns)
        cols_form.addRow("Filter:", self._column_search)

        table_controls_layout.addWidget(cols_group)

        self._copy_button = QtWidgets.QPushButton("Copy table to clipboard")
        self._copy_button.setToolTip("Copy the current table view to the clipboard")
        self._copy_button.clicked.connect(self.copy_to_clipboard)
        table_controls_layout.addStretch()
        table_controls_layout.addWidget(self._copy_button)

        display_group = QtWidgets.QGroupBox("Display")
        display_form = QtWidgets.QFormLayout()
        display_form.setContentsMargins(6, 6, 6, 6)
        display_form.setVerticalSpacing(4)
        display_form.setHorizontalSpacing(6)
        display_form.setLabelAlignment(Qt.AlignLeft)
        display_group.setLayout(display_form)

        self._hide_zeros = QtWidgets.QCheckBox("Hide zero values")
        self._hide_zeros.setChecked(True)
        self._hide_zeros.setToolTip("Hide zero values")
        self._hide_zeros.stateChanged.connect(self.update_hide_zeros)
        display_form.addRow(self._hide_zeros)

        self._normalize = QtWidgets.QCheckBox("Normalize")
        self._normalize.setChecked(False)
        self._normalize.setToolTip("Normalize synapse counts")
        self._normalize.stateChanged.connect(self.update_normalize)
        display_form.addRow(self._normalize)

        self._cell_size = QtWidgets.QSpinBox()
        self._cell_size.setRange(25, 200)
        self._cell_size.setSingleStep(5)
        self._cell_size.setValue(self._table_scale)
        self._cell_size.setSuffix("%")
        self._cell_size.setToolTip("Scale table cells relative to content size")
        self._cell_size.valueChanged.connect(self.update_cell_size)
        display_form.addRow("Scale:", self._cell_size)

        self._always_on_top = QtWidgets.QCheckBox("Always on top")
        self._always_on_top.setToolTip(
            "Keep this window above other BigClust windows"
        )
        self._always_on_top.stateChanged.connect(self.update_always_on_top)
        self._always_on_top.setChecked(True)
        display_form.addRow(self._always_on_top)

        display_controls_layout.addWidget(display_group)
        display_controls_layout.addStretch()
        control_layout.addStretch()

        ### Now the graph
        self._tab_graph = QtWidgets.QWidget()
        self._tabs.addTab(self._tab_graph, "Graph")

        self._tab_graph_layout = QtWidgets.QVBoxLayout()
        self._tab_graph_layout.setContentsMargins(0, 0, 0, 0)
        self._tab_graph_layout.setSpacing(4)
        self._tab_graph.setLayout(self._tab_graph_layout)

        self._graph_widget = pg.PlotWidget()
        self._graph_widget.setBackground("k")
        self._tab_graph_layout.addWidget(self._graph_widget)
        self._graph_widget.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding
        )

        # A couple settings for graph
        row = QtWidgets.QHBoxLayout()
        self._tab_graph_layout.addLayout(row)
        row.setContentsMargins(0, 0, 0, 0)  # Remove margins for a tighter fit
        row.setSpacing(4)

        # Dropdown for colors
        row.addWidget(QtWidgets.QLabel("Color scheme:"))
        self._color_dropdown = QtWidgets.QComboBox()
        self._color_dropdown.setToolTip("Set the color scheme for the graph")
        self._color_dropdown.addItems(
            ["Up/Downstream", "ID"] + list(self._meta_data.columns)
        )
        self._color_dropdown.currentIndexChanged.connect(self.update_graph)
        row.addWidget(self._color_dropdown)

        # Spinbox for linewidth
        row.addWidget(QtWidgets.QLabel("Line width:"))
        self._line_width = QtWidgets.QDoubleSpinBox()
        self._line_width.setRange(1, 10)
        self._line_width.setValue(2)
        self._line_width.setSingleStep(0.1)
        self._line_width.setToolTip("Set the line width for the graph")
        self._line_width.valueChanged.connect(self.update_graph)
        row.addWidget(self._line_width)

        # Spinbox for maximum number of columns to show
        row.addWidget(QtWidgets.QLabel("Max partners:"))
        self._max_cols = QtWidgets.QSpinBox()
        self._max_cols.setRange(1, 1000)
        self._max_cols.setValue(30)
        self._max_cols.setSingleStep(1)
        self._max_cols.setToolTip(
            "Set the maximum number of columns to show in the graph"
        )
        self._max_cols.valueChanged.connect(self.update_graph)
        row.addWidget(self._max_cols)

        # Stretch the row
        row.addStretch()

        # TODOs:
        # add toggles for:
        # - setting colors (perhaps based on dendrogram)
        # - toggle for normalized weight

        # Now that we are done, we need to check if the figure has already
        # something connected
        if not isinstance(self._figure.selected_ids, type(None)) and len(self._figure.selected_ids) > 0:
            self.select(self._figure.selected_ids)

        self.update_cell_size()

    def update_row_labels(self, *args, **kwargs):
        self._model.set_row_labels(self._row_label_dropdown.currentText())

    @trigger_graph_update
    def update_synapse_threshold(self, *args, **kwargs):
        """Update the synapse threshold."""
        self._model.set_synapse_threshold(self._synapse_threshold.value())

    def update_hide_zeros(self, *args, **kwargs):
        """Update the hide zeros setting."""
        self._model.set_hide_zeros(self._hide_zeros.isChecked())

    @trigger_graph_update
    def update_sort_cols(self, *args, **kwargs):
        """Update the column sorting of the table."""
        self._model.set_col_sort(self._sort_cols_dropdown.currentText())

    def update_sort_rows(self, *args, **kwargs):
        """Update the row sorting of the table."""
        self._model.set_row_sort(self._sort_rows_dropdown.currentText())

    @trigger_graph_update
    def update_direction(self, *args, **kwargs):
        """Update the direction of the table."""
        self._model.set_direction(
            upstream=self._upstream.isChecked(), downstream=self._downstream.isChecked()
        )

    @trigger_graph_update
    def update_normalize(self, *args, **kwargs):
        """Update the normalization of the table."""
        self._model._normalize = self._normalize.isChecked()
        self._model.select_rows(self._model._selected_ids)

    @trigger_graph_update
    def update_collapse_rows(self, *args, **kwargs):
        """Collapse rows by the currently selected row label."""
        self._model.set_collapse_rows(self._collapse_rows.isChecked())

    def update_cell_size(self, *args, **kwargs):
        if hasattr(self, "_cell_size"):
            self._table_scale = self._cell_size.value()

        scale = self._table_scale / 100.0
        table_font_size = max(6.0, self._base_table_font.pointSizeF() * scale)
        h_header_font_size = max(
            6.0, self._base_horizontal_header_font.pointSizeF() * scale
        )
        v_header_font_size = max(
            6.0, self._base_vertical_header_font.pointSizeF() * scale
        )
        font_sizes = (table_font_size, h_header_font_size, v_header_font_size)

        if font_sizes == self._last_applied_font_sizes:
            return

        # Scale table/header fonts and then let Qt recompute geometry from content.
        table_font = QtGui.QFont(self._base_table_font)
        table_font.setPointSizeF(table_font_size)
        self._table.setFont(table_font)

        h_header_font = QtGui.QFont(self._base_horizontal_header_font)
        h_header_font.setPointSizeF(h_header_font_size)
        self._table.horizontalHeader().setFont(h_header_font)

        v_header_font = QtGui.QFont(self._base_vertical_header_font)
        v_header_font.setPointSizeF(v_header_font_size)
        self._table.verticalHeader().setFont(v_header_font)

        self._last_applied_font_sizes = font_sizes

        # These are the expensive calls
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
        """Toggle whether this widget should float above other BigClust windows."""
        self.setWindowFlag(Qt.Tool, self._always_on_top.isChecked())
        self.setWindowFlag(Qt.WindowStaysOnTopHint, False)
        # Re-show to apply the updated window flag on all platforms.
        self.show()

    def toggle_options_panel(self, show_options):
        """Show or hide the options panel in the sidebar."""
        self._control_panel.setVisible(show_options)
        if show_options:
            self._toggle_options_button.setToolTip("Hide options panel")
        else:
            self._toggle_options_button.setToolTip("Show options panel")

    def update_graph(self):
        """Update the graph view with the selected IDs."""
        # First clear
        self._graph_widget.clear()

        # Set plot y-axis label
        if not self._normalize.isChecked():
            self._graph_widget.setLabel("left", "Synapse count")
        else:
            self._graph_widget.setLabel("left", "Synapse count (norm.)")

        # Get the data
        data = self._model._view

        # Plot only the first N columns
        data = data.iloc[:, : self._max_cols.value()]

        if data.empty:
            return

        if isinstance(data.columns, pd.MultiIndex):
            # Get all available columns
            all_cols = data.columns.get_level_values(1)
            # Get unique columns but keep the order
            _, idx = np.unique(all_cols, return_index=True)
            cols = all_cols[np.sort(idx)]
            x = np.arange(len(cols))
        else:
            cols = data.columns
            x = np.arange(len(cols))

        if self._color_dropdown.currentText() in self._meta_data.columns:
            this_meta = self._meta_data.loc[data.index]
            vals = this_meta[self._color_dropdown.currentText()].unique()
            colormap = cmap.Colormap("seaborn:tab20")
            colors = {v: c.hex for v, c in zip(vals, colormap.iter_colors(len(vals)))}
            id2color = {
                i: colors[v]
                for i, v in zip(
                    this_meta.index,
                    this_meta[self._color_dropdown.currentText()].values,
                )
            }

        for index, row in data.iterrows():
            if self._color_dropdown.currentText() == "ID":
                # Generate a random color for each row
                color = rgb_from_segment_id(color_seed=1985, segment_id=index)
            elif self._color_dropdown.currentText() != "Up/Downstream":
                # Use the selected column to color the lines
                color = id2color.get(index, "w")

            if isinstance(data.columns, pd.MultiIndex):
                for label in ["upstream", "downstream"]:
                    if self._color_dropdown.currentText() == "Up/Downstream":
                        color = {"upstream": "cyan", "downstream": "red"}[label]

                    if label not in row:
                        continue

                    pen = pg.mkPen(color=color, width=self._line_width.value())
                    y = row[label].reindex(cols).fillna(0).values
                    line = self._graph_widget.plot(
                        x,
                        y,
                        name=str(index),
                        pen=pen,
                        symbol="+",
                        symbolSize=5,
                        symbolBrush=(color),
                    )
                    line.setAlpha(0.8, False)

                    # Add hover tooltip to the line
                    def hovered(sig, points):
                        """Handle hover events on the line."""
                        if not points:
                            return
                        point = points[0]
                        # Show the index and label in the tooltip
                        point.setToolTip(
                            f"{index} ({SHORT[label]})\n"
                            f"Synapse count: {point.y():.2f}"
                        )
                    line.sigPointsHovered.connect(hovered)

                    # Add text at the beginning of the line
                    text = pg.TextItem(
                        f"{index} ({SHORT[label]})",
                        anchor=(1, 0.5),
                        color=color,
                        border=None,
                    )
                    text.setPos(-0.1, y[0])
                    text.setFont(QtGui.QFont("Arial", 8))
                    self._graph_widget.addItem(text)
            else:
                color = "w"
                pen = pg.mkPen(color=color, alpha=0.1, width=self._line_width.value())
                self._graph_widget.plot(
                    x,
                    row.fillna(0).values,
                    name=str(index),
                    pen=pen,
                    symbol="+",
                    symbolSize=5,
                    symbolBrush=(color),
                )
                cols = data.columns

        # Set the x-ticks to the column names
        self._graph_widget.getAxis("bottom").setTicks(
            [list(enumerate(cols.astype(str)))]
        )
        # Note to self: apparently rotating the x-ticks is not supported in pyqtgraph

        # Only allow horizontal scrolling (disable vertical panning/zooming)
        self._graph_widget.setMouseEnabled(x=True, y=False)
        # self._graph_widget.setYRange(data.values.min(), data.values.max(), padding=0.1)

        # Show only the first 20 columns
        if data.shape[1] > 20:
            self._graph_widget.setXRange(0, 20)

    @trigger_graph_update
    def filter_columns(self, *args, **kwargs):
        """Filter the columns based on the search field."""
        search = self._column_search.text()
        self._model.set_filter_columns(search)

    @trigger_graph_update
    def filter_rows(self, *args, **kwargs):
        """Filter the rows based on the search field."""
        search = self._row_search.text()
        self._model.set_filter_rows(search)

    @trigger_graph_update
    def select(self, ids):
        """Select rows by IDs."""
        self._model.select_rows(ids, use_index=True)

    def find_header(self):
        """Find the currently selected header."""
        curr_col = self._table.currentIndex().column()
        label = self._model._view.columns[curr_col]

        # Drop the "upstream" or "downstream" prefix
        # if this is a multi-index
        if isinstance(label, tuple):
            label = label[1]

        if self._figure:
            self._figure.find_label(label, regex=True)

    def find_index(self):
        """Find the currently selected index."""
        curr_row = self._table.currentIndex().row()
        label = self._model._view.index[curr_row]

        if self._figure:
            self._figure.find_label(label, regex=True)

    def copy_to_clipboard(self):
        """Copy the table to the clipboard."""
        # Let's enforce some sensible limits to how many rows we can copy
        if self._model._view.shape[0] > 200:
            raise ValueError("Too many rows to copy to clipboard.")

        self._model._view.to_clipboard(index=True)
