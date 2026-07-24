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

# Colors used for the two connection directions across the profile and network
# plots (keep these in sync so the two views read the same way)
DIRECTION_COLORS = {"upstream": "cyan", "downstream": "red"}

# Above this many lines, per-line text labels become unreadable (and pyqtgraph
# has no batched text primitive), so we skip them
MAX_GRAPH_LABELS = 30

# Skip the hoverable "+" symbols beyond this many points to keep the profile snappy
MAX_GRAPH_POINTS = 50_000

# Network guard rails: beyond these the diagram is unreadable and slow to draw
MAX_NETWORK_NODES = 600
MAX_NETWORK_EDGES = 4_000
MAX_NETWORK_LABELS = 120

# Rows above this are refused by "Copy to clipboard" (use "Export CSV" instead)
MAX_CLIPBOARD_ROWS = 200


def trigger_plot_update(func):
    """Decorator to refresh the profile/network plots after the function runs."""

    def wrapper(self, *args, **kwargs):
        func(self, *args, **kwargs)

        if getattr(self, "_ready", False):
            self.update_plots()

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
        self._hide_zeros = True
        self._synapse_threshold = 1
        self._top_n = 0  # 0 == no limit
        self._col_sort = "By synapse count"
        self._row_sort = None
        self._col_filt = None
        self._row_filt = None
        self._upstream = True
        self._downstream = True
        self._color_cells = True
        self._normalize = False
        # Denominator for the cell background colors; recomputed whenever the
        # view changes so `data()` does not have to scan the values per cell
        self._color_scale = 1.0
        self.colormap = cmap.Colormap("matlab:cool", interpolation="linear")
        self.update_indices()  # pre-compute the indices and columns

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
                # Normalise to range 0 - 1 against the cached scale
                value_norm = min(1.0, value / self._color_scale)

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
        """Select rows by indices.

        Parameters
        ----------
        indices : list-like
            The indices of the rows to select. Can be either the index labels (IDs)
            or integer positions.
        drop_empty_cols : bool, optional
            Whether to drop columns that are zero for all selected rows. Default is True.
        use_index : bool, optional
            Whether the provided indices are index labels (IDs) or integer positions.
            Default is True, i.e. the indices are index labels. If False, the
            indices are treated as integer positions.

        """
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

        # Keep only the N strongest partners (per direction, if we have both)
        self._apply_top_n()

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
        self.update_color_scale()

        # Emit signal to trigger update
        # This is where 99.999% of time is spent
        self.layoutChanged.emit()

    def _apply_top_n(self):
        """Restrict the view to the `_top_n` strongest partners."""
        if not self._top_n or self._view.empty:
            return

        totals = self._view.sum(axis=0)
        if isinstance(self._view.columns, pd.MultiIndex):
            # Top N per direction, so neither direction crowds out the other
            keep = []
            for d in self._view.columns.get_level_values(0).unique():
                mask = self._view.columns.get_level_values(0) == d
                keep.extend(totals[mask].sort_values(ascending=False).index[: self._top_n])
            if len(keep) < self._view.shape[1]:
                self._view = self._view.loc[:, self._view.columns.isin(keep)]
        elif self._top_n < self._view.shape[1]:
            keep = totals.sort_values(ascending=False).index[: self._top_n]
            self._view = self._view.loc[:, self._view.columns.isin(keep)]

    def update_color_scale(self):
        """Recompute the denominator used to color cells.

        Normalised values are fractions, so they get an absolute 0-1 scale which
        keeps up- and downstream cells comparable (they are normalised
        separately). Raw counts are scaled against the largest value in view.
        """
        if self._normalize:
            self._color_scale = 1.0
            return

        values = self._view.values
        if values.size == 0:
            self._color_scale = 1.0
            return

        vmax = np.nanmax(values)
        self._color_scale = float(vmax) if vmax and vmax > 0 else 1.0

    def _refresh_cells(self):
        """Repaint the current cells without forcing a full re-layout."""
        if self._view.size == 0:
            return
        top_left = self.index(0, 0)
        bottom_right = self.index(self._view.shape[0] - 1, self._view.shape[1] - 1)
        self.dataChanged.emit(
            top_left,
            bottom_right,
            [Qt.DisplayRole, Qt.ItemDataRole.BackgroundRole],
        )

    def set_synapse_threshold(self, threshold):
        """Set the synapse threshold."""
        self._synapse_threshold = threshold
        self.select_rows(self._selected_ids)  # reselect rows

    def set_top_n(self, top_n):
        """Limit the view to the N strongest partners (0 == no limit)."""
        self._top_n = top_n
        self.select_rows(self._selected_ids)

    def set_hide_zeros(self, hide_zeros):
        """Set whether to hide zeros."""
        self._hide_zeros = hide_zeros
        self._refresh_cells()

    def set_color_cells(self, color_cells):
        """Set whether cells are colored by value."""
        self._color_cells = color_cells
        self._refresh_cells()

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

    def set_normalize(self, normalize):
        """Show fractions of a neuron's total input/output instead of counts."""
        self._normalize = normalize
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

    Shows the up- and downstream partners of the current selection three ways:
    as a table, as a per-neuron connectivity profile, and as a node-link
    network. All three read from the same filtered view, which is shaped by the
    "Data" controls in the sidebar.

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

    # Main tabs, in order. The sidebar's second tab mirrors the active one.
    _VIEW_TABS = ("Table", "Profile", "Network")

    def __init__(
        self,
        data,
        meta_data,
        figure=None,
        width=900,
        height=600,
        title="Connectivity widget",
        parent=None,
    ):
        assert isinstance(data, pd.DataFrame), "data must be a pandas DataFrame"

        super().__init__(parent, Qt.Window)

        self._ready = False
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

        # The sidebar has a "Data" tab (shapes the view for every plot) and a
        # second tab that follows whichever main tab is active
        self._control_tabs = QtWidgets.QTabWidget()
        control_layout.addWidget(self._control_tabs, 1)

        self._data_controls_tab = QtWidgets.QWidget()
        self._control_tabs.addTab(self._data_controls_tab, "Data")
        data_controls_layout = QtWidgets.QVBoxLayout()
        data_controls_layout.setContentsMargins(0, 0, 0, 0)
        data_controls_layout.setSpacing(4)
        self._data_controls_tab.setLayout(data_controls_layout)

        self._view_controls_stack = QtWidgets.QStackedWidget()
        self._control_tabs.addTab(self._view_controls_stack, "Table")

        self._build_table_tab()
        self._build_data_controls(data_controls_layout)
        self._build_profile_tab()
        self._build_network_tab()
        self._build_view_controls()
        self._build_control_footer(control_layout)

        self._tabs.currentChanged.connect(self._on_tab_changed)

        self._ready = True

        # Now that we are done, we need to check if the figure has already something selected
        if not isinstance(self._figure.selected, type(None)) and len(self._figure.selected) > 0:
            self.select(self._figure.selected)

        self.update_cell_size()

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    def _build_table_tab(self):
        """The table view itself."""
        self._tab_table = QtWidgets.QWidget()
        self._tabs.addTab(self._tab_table, "Table")

        self._tab_table_layout = QtWidgets.QVBoxLayout()
        self._tab_table_layout.setContentsMargins(0, 0, 0, 0)
        self._tab_table_layout.setSpacing(0)
        self._tab_table.setLayout(self._tab_table_layout)

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

    def _build_data_controls(self, layout):
        """Controls that shape the view shared by all three tabs."""
        rows_group = QtWidgets.QGroupBox("Rows")
        rows_form = self._make_form(rows_group)

        # Add a dropdown for row labels
        self._row_label_dropdown = QtWidgets.QComboBox()
        self._row_label_dropdown.setToolTip("Select row labels")
        self._row_label_dropdown.addItems(["ID"] + list(self._meta_data.columns))
        self._row_label_dropdown.currentIndexChanged.connect(self.update_row_labels)
        rows_form.addRow("Labels:", self._row_label_dropdown)

        self._sort_rows_dropdown = QtWidgets.QComboBox()
        self._sort_rows_dropdown.addItems(["No sort", "By label", "By distance"])
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

        layout.addWidget(rows_group)

        cols_group = QtWidgets.QGroupBox("Columns (partners)")
        cols_form = self._make_form(cols_group)

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

        # Add a QSpinBox for the synapse threshold. Set the value before
        # connecting so the widget and the model agree on the default.
        self._synapse_threshold = QtWidgets.QSpinBox()
        self._synapse_threshold.setToolTip(
            "Hide partners whose strongest connection is below this many synapses"
        )
        self._synapse_threshold.setRange(0, 1000)
        self._synapse_threshold.setSingleStep(1)
        self._synapse_threshold.setValue(self._model._synapse_threshold)
        self._synapse_threshold.valueChanged.connect(self.update_synapse_threshold)
        cols_form.addRow("Threshold:", self._synapse_threshold)

        self._top_n = QtWidgets.QSpinBox()
        self._top_n.setRange(0, 10_000)
        self._top_n.setSingleStep(5)
        self._top_n.setValue(0)
        self._top_n.setSpecialValueText("All")
        self._top_n.setToolTip(
            "Keep only the N strongest partners per direction (0 = all)"
        )
        self._top_n.valueChanged.connect(self.update_top_n)
        cols_form.addRow("Top N:", self._top_n)

        self._sort_cols_dropdown = QtWidgets.QComboBox()
        self._sort_cols_dropdown.addItems(
            ["No sort", "By synapse count", "By label", "By distance"]
        )
        self._sort_cols_dropdown.setCurrentIndex(1)  # default to sorting by synapse count
        self._sort_cols_dropdown.currentIndexChanged.connect(self.update_sort_cols)
        cols_form.addRow("Sort:", self._sort_cols_dropdown)

        self._column_search = QtWidgets.QLineEdit()
        self._column_search.setToolTip("Filter columns by name")
        self._column_search.setPlaceholderText("Filter columns")
        self._column_search.textChanged.connect(self.filter_columns)
        cols_form.addRow("Filter:", self._column_search)

        self._normalize = QtWidgets.QCheckBox("Normalize")
        self._normalize.setChecked(False)
        self._normalize.setToolTip(
            "Show each connection as a fraction of that neuron's total "
            "input/output rather than a raw synapse count"
        )
        self._normalize.stateChanged.connect(self.update_normalize)
        cols_form.addRow(self._normalize)

        layout.addWidget(cols_group)
        layout.addStretch()

    def _build_view_controls(self):
        """Per-tab display options; the stack follows the active main tab."""
        # --- Table -----------------------------------------------------
        table_page = QtWidgets.QWidget()
        table_layout = QtWidgets.QVBoxLayout()
        table_layout.setContentsMargins(0, 0, 0, 0)
        table_layout.setSpacing(4)
        table_page.setLayout(table_layout)

        table_group = QtWidgets.QGroupBox("Table display")
        table_form = self._make_form(table_group)

        self._hide_zeros = QtWidgets.QCheckBox("Hide zero values")
        self._hide_zeros.setChecked(True)
        self._hide_zeros.setToolTip("Leave cells with no connection blank")
        self._hide_zeros.stateChanged.connect(self.update_hide_zeros)
        table_form.addRow(self._hide_zeros)

        self._color_cells = QtWidgets.QCheckBox("Color cells")
        self._color_cells.setChecked(True)
        self._color_cells.setToolTip("Shade cells by connection strength")
        self._color_cells.stateChanged.connect(self.update_color_cells)
        table_form.addRow(self._color_cells)

        self._cell_size = QtWidgets.QSpinBox()
        self._cell_size.setRange(25, 200)
        self._cell_size.setSingleStep(5)
        self._cell_size.setValue(self._table_scale)
        self._cell_size.setSuffix("%")
        self._cell_size.setToolTip("Scale table cells relative to content size")
        self._cell_size.valueChanged.connect(self.update_cell_size)
        table_form.addRow("Scale:", self._cell_size)

        table_layout.addWidget(table_group)
        table_layout.addStretch()
        self._view_controls_stack.addWidget(table_page)

        # --- Profile ---------------------------------------------------
        profile_page = QtWidgets.QWidget()
        profile_layout = QtWidgets.QVBoxLayout()
        profile_layout.setContentsMargins(0, 0, 0, 0)
        profile_layout.setSpacing(4)
        profile_page.setLayout(profile_layout)

        profile_group = QtWidgets.QGroupBox("Profile display")
        profile_form = self._make_form(profile_group)

        self._color_dropdown = QtWidgets.QComboBox()
        self._color_dropdown.setToolTip("Set the color scheme for the profile lines")
        self._color_dropdown.addItems(
            ["Up/Downstream", "ID"] + list(self._meta_data.columns)
        )
        self._color_dropdown.currentIndexChanged.connect(self.update_plots)
        profile_form.addRow("Color by:", self._color_dropdown)

        self._line_width = QtWidgets.QDoubleSpinBox()
        self._line_width.setRange(1, 10)
        self._line_width.setValue(2)
        self._line_width.setSingleStep(0.1)
        self._line_width.setToolTip("Set the line width for the profile")
        self._line_width.valueChanged.connect(self.update_plots)
        profile_form.addRow("Line width:", self._line_width)

        self._max_cols = QtWidgets.QSpinBox()
        self._max_cols.setRange(1, 1000)
        self._max_cols.setValue(5)
        self._max_cols.setSingleStep(1)
        self._max_cols.setToolTip("Maximum number of partners to plot")
        self._max_cols.valueChanged.connect(self.update_plots)
        profile_form.addRow("Max partners:", self._max_cols)

        self._max_rows = QtWidgets.QSpinBox()
        self._max_rows.setRange(1, 100_000)
        self._max_rows.setValue(100)
        self._max_rows.setSingleStep(10)
        self._max_rows.setToolTip("Maximum number of neurons to plot")
        self._max_rows.valueChanged.connect(self.update_plots)
        profile_form.addRow("Max rows:", self._max_rows)

        profile_layout.addWidget(profile_group)
        profile_layout.addStretch()
        self._view_controls_stack.addWidget(profile_page)

        # --- Network ---------------------------------------------------
        network_page = QtWidgets.QWidget()
        network_layout = QtWidgets.QVBoxLayout()
        network_layout.setContentsMargins(0, 0, 0, 0)
        network_layout.setSpacing(4)
        network_page.setLayout(network_layout)

        network_group = QtWidgets.QGroupBox("Network display")
        network_form = self._make_form(network_group)

        self._net_layout_dropdown = QtWidgets.QComboBox()
        self._net_layout_dropdown.addItems(["Layered", "Spring"])
        self._net_layout_dropdown.setToolTip(
            "Layered puts upstream partners left, the selection in the middle "
            "and downstream partners right. Spring uses a force-directed layout."
        )
        self._net_layout_dropdown.currentIndexChanged.connect(self.update_plots)
        network_form.addRow("Layout:", self._net_layout_dropdown)

        self._net_color_dropdown = QtWidgets.QComboBox()
        self._net_color_dropdown.setToolTip("Color scheme for the selected neurons")
        self._net_color_dropdown.addItems(["ID"] + list(self._meta_data.columns))
        self._net_color_dropdown.currentIndexChanged.connect(self.update_plots)
        network_form.addRow("Color by:", self._net_color_dropdown)

        self._net_max_cols = QtWidgets.QSpinBox()
        self._net_max_cols.setRange(1, 500)
        self._net_max_cols.setValue(10)
        self._net_max_cols.setSingleStep(1)
        self._net_max_cols.setToolTip("Maximum number of partners to draw")
        self._net_max_cols.valueChanged.connect(self.update_plots)
        network_form.addRow("Max partners:", self._net_max_cols)

        self._net_max_rows = QtWidgets.QSpinBox()
        self._net_max_rows.setRange(1, 2000)
        self._net_max_rows.setValue(50)
        self._net_max_rows.setSingleStep(5)
        self._net_max_rows.setToolTip("Maximum number of selected neurons to draw")
        self._net_max_rows.valueChanged.connect(self.update_plots)
        network_form.addRow("Max rows:", self._net_max_rows)

        self._net_edge_width = QtWidgets.QDoubleSpinBox()
        self._net_edge_width.setRange(0.5, 20)
        self._net_edge_width.setValue(6)
        self._net_edge_width.setSingleStep(0.5)
        self._net_edge_width.setToolTip("Width of the strongest edge")
        self._net_edge_width.valueChanged.connect(self.update_plots)
        network_form.addRow("Max edge width:", self._net_edge_width)

        self._net_labels = QtWidgets.QCheckBox("Show node labels")
        self._net_labels.setChecked(True)
        self._net_labels.setToolTip(
            f"Labels are skipped above {MAX_NETWORK_LABELS} nodes"
        )
        self._net_labels.stateChanged.connect(self.update_plots)
        network_form.addRow(self._net_labels)

        network_layout.addWidget(network_group)

        hint = QtWidgets.QLabel("Click a node to find it in the scatter plot.")
        hint.setWordWrap(True)
        hint.setStyleSheet("color: gray;")
        network_layout.addWidget(hint)
        network_layout.addStretch()
        self._view_controls_stack.addWidget(network_page)

    def _build_control_footer(self, layout):
        """Window-level actions, always visible below the sidebar tabs."""
        self._always_on_top = QtWidgets.QCheckBox("Always on top")
        self._always_on_top.setToolTip("Keep this window above other BigClust windows")
        self._always_on_top.stateChanged.connect(self.update_always_on_top)
        self._always_on_top.setChecked(True)
        layout.addWidget(self._always_on_top)

        button_row = QtWidgets.QHBoxLayout()
        button_row.setContentsMargins(0, 0, 0, 0)
        button_row.setSpacing(4)

        self._copy_button = QtWidgets.QPushButton("Copy")
        self._copy_button.setToolTip("Copy the current table view to the clipboard")
        self._copy_button.clicked.connect(self.copy_to_clipboard)
        button_row.addWidget(self._copy_button)

        self._export_button = QtWidgets.QPushButton("Export CSV")
        self._export_button.setToolTip("Save the current table view as a CSV file")
        self._export_button.clicked.connect(self.export_to_csv)
        button_row.addWidget(self._export_button)

        layout.addLayout(button_row)

    def _build_profile_tab(self):
        """Per-neuron connectivity profile (one line per neuron)."""
        self._tab_profile = QtWidgets.QWidget()
        self._tabs.addTab(self._tab_profile, "Profile")

        profile_layout = QtWidgets.QVBoxLayout()
        profile_layout.setContentsMargins(0, 0, 0, 0)
        profile_layout.setSpacing(0)
        self._tab_profile.setLayout(profile_layout)

        self._profile_widget = pg.PlotWidget()
        self._profile_widget.setBackground("k")
        self._profile_widget.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding
        )
        profile_layout.addWidget(self._profile_widget)

        # Building the plots can be expensive, so we defer them until the tab
        # actually becomes visible (Qt sends a Show event to the tab page both
        # on tab switch and when the whole window is re-shown)
        self._profile_dirty = True
        self._tab_profile.installEventFilter(self)

    def _build_network_tab(self):
        """Node-link diagram of the selection and its partners."""
        self._tab_network = QtWidgets.QWidget()
        self._tabs.addTab(self._tab_network, "Network")

        network_layout = QtWidgets.QVBoxLayout()
        network_layout.setContentsMargins(0, 0, 0, 0)
        network_layout.setSpacing(0)
        self._tab_network.setLayout(network_layout)

        self._network_widget = pg.PlotWidget()
        self._network_widget.setBackground("k")
        self._network_widget.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding
        )
        self._network_widget.hideAxis("bottom")
        self._network_widget.hideAxis("left")
        self._network_widget.setAspectLocked(False)
        network_layout.addWidget(self._network_widget)

        self._network_item = None
        self._network_dirty = True
        self._tab_network.installEventFilter(self)

    @staticmethod
    def _make_form(group_box):
        """Build the form layout we use inside every control group box."""
        form = QtWidgets.QFormLayout()
        form.setContentsMargins(6, 6, 6, 6)
        form.setVerticalSpacing(4)
        form.setHorizontalSpacing(6)
        form.setLabelAlignment(Qt.AlignLeft)
        group_box.setLayout(form)
        return form

    # ------------------------------------------------------------------
    # Control callbacks
    # ------------------------------------------------------------------

    @trigger_plot_update
    def update_row_labels(self, *args, **kwargs):
        self._model.set_row_labels(self._row_label_dropdown.currentText())

    @trigger_plot_update
    def update_synapse_threshold(self, *args, **kwargs):
        """Update the synapse threshold."""
        self._model.set_synapse_threshold(self._synapse_threshold.value())

    @trigger_plot_update
    def update_top_n(self, *args, **kwargs):
        """Limit the view to the N strongest partners."""
        self._model.set_top_n(self._top_n.value())

    def update_hide_zeros(self, *args, **kwargs):
        """Update the hide zeros setting."""
        self._model.set_hide_zeros(self._hide_zeros.isChecked())

    def update_color_cells(self, *args, **kwargs):
        """Update whether table cells are colored by value."""
        self._model.set_color_cells(self._color_cells.isChecked())

    @trigger_plot_update
    def update_sort_cols(self, *args, **kwargs):
        """Update the column sorting of the table."""
        self._model.set_col_sort(self._sort_cols_dropdown.currentText())

    @trigger_plot_update
    def update_sort_rows(self, *args, **kwargs):
        """Update the row sorting of the table."""
        self._model.set_row_sort(self._sort_rows_dropdown.currentText())

    @trigger_plot_update
    def update_direction(self, *args, **kwargs):
        """Update the direction of the table."""
        self._model.set_direction(
            upstream=self._upstream.isChecked(), downstream=self._downstream.isChecked()
        )

    @trigger_plot_update
    def update_normalize(self, *args, **kwargs):
        """Update the normalization of the table."""
        self._model.set_normalize(self._normalize.isChecked())

    @trigger_plot_update
    def update_collapse_rows(self, *args, **kwargs):
        """Collapse rows by the currently selected row label."""
        self._model.set_collapse_rows(self._collapse_rows.isChecked())

    @trigger_plot_update
    def filter_columns(self, *args, **kwargs):
        """Filter the columns based on the search field."""
        self._model.set_filter_columns(self._column_search.text())

    @trigger_plot_update
    def filter_rows(self, *args, **kwargs):
        """Filter the rows based on the search field."""
        self._model.set_filter_rows(self._row_search.text())

    @trigger_plot_update
    def select(self, indices):
        """Select rows by Indices."""
        self._model.select_rows(indices, use_index=False)

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

    def eventFilter(self, obj, event):
        # Rebuild a plot when its tab becomes visible (tab switch or window
        # re-show) and an update was deferred in the meantime
        if event.type() == QtCore.QEvent.Type.Show:
            if obj is self._tab_profile and self._profile_dirty:
                self._rebuild_profile()
            elif obj is self._tab_network and self._network_dirty:
                self._rebuild_network()
        return super().eventFilter(obj, event)

    def _on_tab_changed(self, index):
        """Keep the sidebar's view tab in step with the active main tab."""
        if 0 <= index < len(self._VIEW_TABS):
            self._view_controls_stack.setCurrentIndex(index)
            self._control_tabs.setTabText(1, self._VIEW_TABS[index])

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

    # ------------------------------------------------------------------
    # Plots
    # ------------------------------------------------------------------

    def update_plots(self, *args, **kwargs):
        """Mark both plots stale and rebuild whichever one is on screen.

        The hidden plot is rebuilt lazily when its tab is next shown.
        """
        self._profile_dirty = True
        self._network_dirty = True

        if self._tab_profile.isVisible():
            self._rebuild_profile()
        elif self._tab_network.isVisible():
            self._rebuild_network()

    def _find_in_figure(self, label):
        """Ask the scatter plot to locate `label`."""
        if self._figure:
            self._figure.find_label(label, regex=True)

    def _graph_line_colors(self, index, label=None, id2color=None, scheme=None):
        """Determine the line color for each row in `index`."""
        scheme = scheme if scheme is not None else self._color_dropdown.currentText()
        if scheme == "Up/Downstream":
            return [DIRECTION_COLORS.get(label, "w")] * len(index)
        elif scheme == "ID":
            # Generate a stable color for each row
            colors = []
            for ix in index:
                # Collapsed rows have (string) labels as index which can't
                # be translated into a color
                try:
                    colors.append(rgb_from_segment_id(color_seed=1985, segment_id=ix))
                except (TypeError, ValueError):
                    colors.append("w")
            return colors
        else:
            # Use the selected meta data column to color the lines
            id2color = id2color if id2color is not None else {}
            return [id2color.get(ix, "w") for ix in index]

    def _meta_color_map(self, index, column):
        """Map each row in `index` to a color from a meta data column."""
        if column not in self._meta_data.columns:
            return None
        try:
            this_meta = self._meta_data.loc[index]
        except KeyError:
            # Collapsed rows are labels, not IDs, and have no meta data
            return None
        vals = this_meta[column].unique()
        colormap = cmap.Colormap("seaborn:tab20")
        colors = {v: c.hex for v, c in zip(vals, colormap.iter_colors(len(vals)))}
        return {i: colors[v] for i, v in zip(this_meta.index, this_meta[column].values)}

    def _plot_line_batches(self, x, V, row_colors):
        """Plot rows of `V` as lines, batched into one curve item per color."""
        groups = {}
        for i, c in enumerate(row_colors):
            qcol = pg.mkColor(c)
            groups.setdefault(qcol.name(), (qcol, []))[1].append(i)

        # NaN separators let us draw all same-colored lines as a single item
        x_nan = np.append(x.astype(float), np.nan)
        for qcol, rows in groups.values():
            xs = np.tile(x_nan, len(rows))
            ys = np.column_stack([V[rows], np.full(len(rows), np.nan)]).ravel()
            curve = pg.PlotCurveItem(
                x=xs,
                y=ys,
                connect="finite",
                pen=pg.mkPen(color=qcol, width=self._line_width.value()),
            )
            curve.setOpacity(0.8)
            self._profile_widget.addItem(curve)

    def _rebuild_profile(self):
        """Rebuild the connectivity profile from the current table view."""
        self._profile_dirty = False

        # First clear
        self._profile_widget.clear()

        # Set plot y-axis label
        if not self._normalize.isChecked():
            self._profile_widget.setLabel("left", "Synapse count")
        else:
            self._profile_widget.setLabel("left", "Synapse count (norm.)")

        # Get the data
        data = self._model._view

        # Plot only the first N columns
        data = data.iloc[:, : self._max_cols.value()]

        # Plot only the first N rows
        n_total = len(data)
        data = data.iloc[: self._max_rows.value()]
        if len(data) < n_total:
            self._profile_widget.setTitle(
                f"Showing first {len(data)} of {n_total} rows",
                color="#aaaaaa",
                size="9pt",
            )
        else:
            self._profile_widget.setTitle(None)

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

        id2color = self._meta_color_map(data.index, self._color_dropdown.currentText())

        # Collect per-point data for the "+" symbols and per-line text labels
        # as we go; both are added in a single batch at the end
        pts_x, pts_y, pts_brush, pts_data = [], [], [], []
        labels = []

        if isinstance(data.columns, pd.MultiIndex):
            for label in ("upstream", "downstream"):
                if label not in data.columns.get_level_values(0):
                    continue

                # All of this direction's values as a (n_rows, n_cols) array
                V = data[label].reindex(columns=cols).fillna(0).values
                row_colors = self._graph_line_colors(data.index, label, id2color)

                self._plot_line_batches(x, V, row_colors)

                pts_x.append(np.tile(x, len(V)))
                pts_y.append(V.ravel())
                pts_brush += [c for c in row_colors for _ in range(len(cols))]
                pts_data += [
                    (ix, SHORT[label]) for ix in data.index for _ in range(len(cols))
                ]
                labels += [
                    (f"{ix} ({SHORT[label]})", c, V[i, 0])
                    for i, (ix, c) in enumerate(zip(data.index, row_colors))
                ]
        else:
            V = data.fillna(0).values
            row_colors = ["w"] * len(data)

            self._plot_line_batches(x, V, row_colors)

            pts_x.append(np.tile(x, len(V)))
            pts_y.append(V.ravel())
            pts_brush += [c for c in row_colors for _ in range(len(cols))]
            pts_data += [(ix, None) for ix in data.index for _ in range(len(cols))]

        # Add the "+" symbols as a single batched scatter item with hover tooltips
        n_points = sum(len(p) for p in pts_x)
        if 0 < n_points <= MAX_GRAPH_POINTS:

            def tip(x, y, data):
                ix, direction = data
                head = f"{ix} ({direction})" if direction else f"{ix}"
                return f"{head}\nSynapse count: {y:.2f}"

            scatter = pg.ScatterPlotItem(
                x=np.concatenate(pts_x),
                y=np.concatenate(pts_y),
                symbol="+",
                size=5,
                pen=pg.mkPen(200, 200, 200),
                brush=pts_brush,
                data=pts_data,
                hoverable=True,
                tip=tip,
            )
            scatter.setOpacity(0.8)
            scatter.sigClicked.connect(self._on_profile_point_clicked)
            self._profile_widget.addItem(scatter)

        # Add text at the beginning of each line (unless there are too many)
        if len(labels) <= MAX_GRAPH_LABELS:
            for text, color, y0 in labels:
                item = pg.TextItem(text, anchor=(1, 0.5), color=color, border=None)
                item.setPos(-0.1, y0)
                item.setFont(QtGui.QFont("Arial", 8))
                self._profile_widget.addItem(item)

        # Set the x-ticks to the column names
        self._profile_widget.getAxis("bottom").setTicks(
            [list(enumerate(cols.astype(str)))]
        )
        # Note to self: apparently rotating the x-ticks is not supported in pyqtgraph

        # Only allow horizontal scrolling (disable vertical panning/zooming)
        self._profile_widget.setMouseEnabled(x=True, y=False)

        # Show only the first 20 columns
        if data.shape[1] > 20:
            self._profile_widget.setXRange(0, 20)

    def _on_profile_point_clicked(self, _scatter, points):
        """Find the neuron behind a clicked profile point in the scatter plot."""
        if not len(points):
            return
        ix = points[0].data()[0]
        self._find_in_figure(str(ix))

    def _network_edges(self, data):
        """Build the node/edge lists for the network from a table view.

        Returns ``(nodes, edges)`` where `nodes` is a list of
        ``(key, kind, direction, label)`` and `edges` is a list of
        ``(source_idx, target_idx, weight, direction)``. Partners are keyed by
        direction so a neuron that is both an input and an output shows up on
        both sides of a layered layout.
        """
        row_labels = dict(zip(data.index, self._model._indices[: len(data)]))

        nodes, node_index = [], {}

        def node_id(key, kind, direction, label):
            if key not in node_index:
                node_index[key] = len(nodes)
                nodes.append((key, kind, direction, label))
            return node_index[key]

        for ix in data.index:
            node_id(("row", ix), "row", None, str(row_labels.get(ix, ix)))

        edges = []
        values = data.values
        multi = isinstance(data.columns, pd.MultiIndex)
        for j, col in enumerate(data.columns):
            direction = col[0] if multi else "downstream"
            name = col[1] if multi else col
            target = node_id(("partner", direction, name), "partner", direction, str(name))
            for i, ix in enumerate(data.index):
                w = values[i, j]
                if not w or not np.isfinite(w):
                    continue
                edges.append((node_index[("row", ix)], target, float(w), direction))

        return nodes, edges

    def _network_positions(self, nodes, edges):
        """Compute node positions for the selected layout."""
        n = len(nodes)
        if self._net_layout_dropdown.currentText() == "Spring":
            import networkx as nx

            G = nx.Graph()
            G.add_nodes_from(range(n))
            for src, tgt, w, _ in edges:
                G.add_edge(src, tgt, weight=w)
            # Fixed seed so the layout does not jump around between rebuilds
            pos = nx.spring_layout(G, weight="weight", seed=1985)
            return np.array([pos[i] for i in range(n)], dtype=float)

        # Layered: upstream partners left, selection centre, downstream right
        columns = {"upstream": [], "row": [], "downstream": []}
        for i, (_key, kind, direction, _label) in enumerate(nodes):
            columns["row" if kind == "row" else direction].append(i)

        pos = np.zeros((n, 2), dtype=float)
        x_of = {"upstream": -1.0, "row": 0.0, "downstream": 1.0}

        # Place the selected neurons first, then order each partner column by
        # the mean height of the rows it connects to (a barycentre pass), which
        # cuts down on edge crossings considerably
        rows = columns["row"]
        row_y = {}
        for rank, i in enumerate(rows):
            y = 0.5 if len(rows) == 1 else rank / (len(rows) - 1)
            row_y[i] = y
            pos[i] = (x_of["row"], y)

        for direction in ("upstream", "downstream"):
            members = columns[direction]
            if not members:
                continue
            neighbours = {i: [] for i in members}
            for src, tgt, w, _ in edges:
                if tgt in neighbours:
                    neighbours[tgt].append(row_y.get(src, 0.5))
            members = sorted(
                members,
                key=lambda i: np.mean(neighbours[i]) if neighbours[i] else 0.5,
            )
            for rank, i in enumerate(members):
                y = 0.5 if len(members) == 1 else rank / (len(members) - 1)
                pos[i] = (x_of[direction], y)

        return pos

    def _rebuild_network(self):
        """Rebuild the node-link diagram from the current table view."""
        self._network_dirty = False
        self._network_widget.clear()
        self._network_item = None

        data = self._model._view
        n_rows_total, n_cols_total = data.shape
        data = data.iloc[: self._net_max_rows.value(), : self._net_max_cols.value()]

        notes = []
        if len(data) < n_rows_total:
            notes.append(f"{len(data)}/{n_rows_total} neurons")
        if data.shape[1] < n_cols_total:
            notes.append(f"{data.shape[1]}/{n_cols_total} partners")

        if data.empty:
            self._network_widget.setTitle(
                "Nothing to show - select neurons in the scatter plot",
                color="#aaaaaa",
                size="9pt",
            )
            return

        nodes, edges = self._network_edges(data)

        if len(nodes) > MAX_NETWORK_NODES or len(edges) > MAX_NETWORK_EDGES:
            self._network_widget.setTitle(
                f"Too dense to draw ({len(nodes)} nodes, {len(edges)} edges) - "
                "lower 'Max partners'/'Max rows' or raise the threshold",
                color="#ffaa55",
                size="9pt",
            )
            return

        pos = self._network_positions(nodes, edges)

        # --- edges ------------------------------------------------------
        adj = np.empty((0, 2), dtype=int)
        pen = None
        if edges:
            # Sorting by direction means pyqtgraph only switches pens twice
            edges = sorted(edges, key=lambda e: e[3])
            adj = np.array([[e[0], e[1]] for e in edges], dtype=int)

            weights = np.array([e[2] for e in edges], dtype=float)
            wmax = weights.max() if weights.max() > 0 else 1.0
            widths = 0.5 + (weights / wmax) * (self._net_edge_width.value() - 0.5)

            pen = np.zeros(
                len(edges),
                dtype=[
                    ("red", np.ubyte),
                    ("green", np.ubyte),
                    ("blue", np.ubyte),
                    ("alpha", np.ubyte),
                    ("width", float),
                ],
            )
            for i, (_src, _tgt, _w, direction) in enumerate(edges):
                c = pg.mkColor(DIRECTION_COLORS.get(direction, "w"))
                pen[i] = (c.red(), c.green(), c.blue(), 160, widths[i])

        # --- nodes ------------------------------------------------------
        strength = np.zeros(len(nodes), dtype=float)
        for src, tgt, w, _ in edges:
            strength[src] += w
            strength[tgt] += w
        smax = strength.max() if strength.size and strength.max() > 0 else 1.0
        sizes = 8.0 + np.sqrt(strength / smax) * 16.0

        row_index = [n[0][1] for n in nodes if n[1] == "row"]
        scheme = self._net_color_dropdown.currentText()
        id2color = self._meta_color_map(pd.Index(row_index), scheme)
        row_colors = dict(
            zip(row_index, self._graph_line_colors(row_index, None, id2color, scheme))
        )

        brushes, node_data = [], []
        for i, (key, kind, direction, label) in enumerate(nodes):
            if kind == "row":
                brushes.append(pg.mkBrush(pg.mkColor(row_colors.get(key[1], "w"))))
            else:
                c = pg.mkColor(DIRECTION_COLORS.get(direction, "w"))
                c.setAlpha(200)
                brushes.append(pg.mkBrush(c))
            node_data.append((kind, direction, label, float(strength[i])))

        total_label = "Total weight" if self._normalize.isChecked() else "Total synapses"

        def tip(x, y, data):
            kind, direction, label, total = data
            if kind == "row":
                head = f"{label}\nSelected neuron"
            else:
                head = f"{label}\n{direction.capitalize()} partner"
            return f"{head}\n{total_label}: {total:,.2f}"

        self._network_item = pg.GraphItem()
        self._network_widget.addItem(self._network_item)
        self._network_item.setData(
            pos=pos,
            adj=adj,
            pen=pen,
            size=sizes,
            symbol="o",
            symbolBrush=brushes,
            symbolPen=pg.mkPen(30, 30, 30),
            pxMode=True,
            data=node_data,
            hoverable=True,
            tip=tip,
        )
        self._network_item.scatter.sigClicked.connect(self._on_network_node_clicked)

        # --- labels -----------------------------------------------------
        if self._net_labels.isChecked() and len(nodes) <= MAX_NETWORK_LABELS:
            layered = self._net_layout_dropdown.currentText() == "Layered"
            # Labels point away from the centre of the graph so they grow into
            # empty space instead of over the edges (and off the right margin)
            x_mid = (pos[:, 0].min() + pos[:, 0].max()) / 2 if len(pos) else 0.0
            x_span = np.ptp(pos[:, 0]) if len(pos) else 1.0
            for i, (_key, kind, direction, label) in enumerate(nodes):
                if layered and kind == "row":
                    anchor, offset = (0.5, 1.2), 0.0
                else:
                    outward = pos[i][0] >= x_mid
                    anchor = (0, 0.5) if outward else (1, 0.5)
                    offset = (0.02 if outward else -0.02) * (x_span or 1.0)
                item = pg.TextItem(label, anchor=anchor, color="#dddddd")
                item.setFont(QtGui.QFont("Arial", 8))
                item.setPos(pos[i][0] + offset, pos[i][1])
                self._network_widget.addItem(item)

        title = ", ".join(notes)
        self._network_widget.setTitle(
            f"Showing {title}" if title else None, color="#aaaaaa", size="9pt"
        )
        self._network_widget.setMouseEnabled(x=True, y=True)
        # Generous padding leaves room for the (pixel-sized) node labels, which
        # autoRange does not account for
        self._network_widget.getViewBox().autoRange(padding=0.25)

    def _on_network_node_clicked(self, _scatter, points):
        """Find the neuron behind a clicked network node in the scatter plot."""
        if not len(points):
            return
        self._find_in_figure(points[0].data()[2])

    # ------------------------------------------------------------------
    # Table interactions & export
    # ------------------------------------------------------------------

    def find_header(self, section):
        """Find the double-clicked column header in the scatter plot."""
        try:
            label = self._model._view.columns[section]
        except IndexError:
            return

        # Drop the "upstream" or "downstream" prefix if this is a multi-index
        if isinstance(label, tuple):
            label = label[1]

        self._find_in_figure(str(label))

    def find_index(self, section):
        """Find the double-clicked row in the scatter plot."""
        try:
            label = self._model._view.index[section]
        except IndexError:
            return

        self._find_in_figure(str(label))

    def copy_to_clipboard(self):
        """Copy the current table view to the clipboard."""
        view = self._model._view
        # Let's enforce some sensible limits to how many rows we can copy
        if view.shape[0] > MAX_CLIPBOARD_ROWS:
            QtWidgets.QMessageBox.warning(
                self,
                "Too many rows",
                f"The current view has {view.shape[0]:,} rows, which is more "
                f"than the {MAX_CLIPBOARD_ROWS} we copy to the clipboard.\n\n"
                "Narrow the selection or use 'Export CSV' instead.",
            )
            return

        view.to_clipboard(index=True)

    def export_to_csv(self):
        """Save the current table view as a CSV file."""
        view = self._model._view
        if view.empty:
            QtWidgets.QMessageBox.information(
                self, "Nothing to export", "The current view is empty."
            )
            return

        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Export connectivity", "connectivity.csv", "CSV files (*.csv)"
        )
        if not path:
            return

        try:
            view.to_csv(path, index=True)
        except OSError as e:
            QtWidgets.QMessageBox.critical(self, "Export failed", str(e))
