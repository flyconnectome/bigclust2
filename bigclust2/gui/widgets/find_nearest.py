"""Find, for each selected neuron, its nearest neighbours from a filtered pool.

Answers "select a group on one side of the brain, find the closest neurons on the
other side": a *query* set of neurons (pinned from the figure selection when the
widget opens) is matched, per neuron, against a **candidate pool** you define
with metadata filters. The pool is independent of the figure's Scope, so it can
deliberately *exclude* the query neurons (which Scope cannot — scoping to the
other side would deselect them).

The similarity source (embedding / KNN graph / distance matrix / feature space)
and metric mirror the Grow/Shrink setup; the per-query top-N computation itself
lives in :func:`bigclust2.grow_shrink.per_query_neighbors`.
"""

import numpy as np
import pandas as pd

from PySide6 import QtWidgets, QtCore
from PySide6.QtCore import Qt

from ... import grow_shrink as gs
from ..controls.filters import ScopeFilterRow

# Human-readable labels for the similarity sources (kept in sync with the
# Grow/Shrink Options menu in core.py).
SOURCE_LABELS = {
    gs.SOURCE_EMBEDDING: "Embedding (screen)",
    gs.SOURCE_KNN: "KNN graph",
    gs.SOURCE_DISTANCES: "Distance matrix",
    gs.SOURCE_FEATURES: "Feature space",
}

# Metric list offered for the feature source (matches the Grow/Shrink menu).
FEATURE_METRICS = ("euclidean", "cosine", "cityblock", "correlation")

# Copy refuses to dump more than this many rows to the clipboard.
MAX_CLIPBOARD_ROWS = 500

# Muted grey for secondary text (status line, pool match count) so it reads as
# clearly subordinate to the normal labels.
MUTED_TEXT = "#8a8a8a"


class FindNearestModel(QtCore.QAbstractTableModel):
    """Results table with two interchangeable layouts.

    * **wide** — one row per query neuron; columns ``Query | Label | NN1…NNn``,
      each neighbour cell showing the id over its distance (label + full distance
      in the tooltip). Empty cells mark queries with fewer than N in-pool
      neighbours.
    * **long** — one row per (query, neighbour) match, with explicit ``Rank`` and
      ``Distance`` columns; padding matches are dropped.
    """

    LONG_HEADERS = (
        "Query",
        "Query label",
        "Rank",
        "Neighbour",
        "Neighbour label",
        "Distance",
    )

    def __init__(self):
        super().__init__()
        self._layout = "wide"
        self._distance_label = "Distance"
        self._n = 0
        self._query_ids = np.empty(0, dtype=object)
        self._query_labels = np.empty(0, dtype=object)
        self._neigh_ids = np.empty((0, 0), dtype=object)
        self._neigh_labels = np.empty((0, 0), dtype=object)
        self._neigh_dists = np.empty((0, 0), dtype=float)
        self._long_rows = []  # (qid, qlabel, rank, nid, nlabel, dist) per real match

    def set_results(self, query_ids, query_labels, neigh_ids, neigh_labels, neigh_dists):
        self.beginResetModel()
        self._query_ids = np.asarray(query_ids, dtype=object)
        self._query_labels = np.asarray(query_labels, dtype=object)
        self._neigh_ids = np.asarray(neigh_ids, dtype=object)
        self._neigh_labels = np.asarray(neigh_labels, dtype=object)
        self._neigh_dists = np.asarray(neigh_dists, dtype=float)
        self._n = self._neigh_ids.shape[1] if self._neigh_ids.size else 0
        self._long_rows = self._build_long_rows()
        self.endResetModel()

    def _build_long_rows(self):
        rows = []
        for i in range(len(self._query_ids)):
            for j in range(self._n):
                nid = self._neigh_ids[i, j]
                if nid is None:
                    continue
                rows.append(
                    (
                        self._query_ids[i],
                        self._query_labels[i],
                        j + 1,
                        nid,
                        self._neigh_labels[i, j],
                        self._neigh_dists[i, j],
                    )
                )
        return rows

    def set_layout(self, mode):
        if mode == self._layout:
            return
        self.beginResetModel()
        self._layout = mode
        self.endResetModel()

    def set_distance_label(self, label):
        """Label for the distance column/tooltip (e.g. 'Graph distance')."""
        self._distance_label = label

    def clear(self):
        self.set_results([], [], np.empty((0, 0)), np.empty((0, 0)), np.empty((0, 0)))

    def rowCount(self, index=QtCore.QModelIndex()):
        return len(self._long_rows) if self._layout == "long" else len(self._query_ids)

    def columnCount(self, index=QtCore.QModelIndex()):
        return 6 if self._layout == "long" else 2 + self._n

    def headerData(self, section, orientation, role):
        if role != Qt.DisplayRole or orientation != Qt.Horizontal:
            return None
        if self._layout == "long":
            if section == 5:
                return self._distance_label
            return self.LONG_HEADERS[section] if 0 <= section < 6 else None
        if section == 0:
            return "Query"
        if section == 1:
            return "Label"
        return f"NN{section - 1}"

    def data(self, index, role):
        row, col = index.row(), index.column()
        if role == Qt.TextAlignmentRole:
            return int(Qt.AlignCenter)

        if self._layout == "long":
            if role == Qt.DisplayRole:
                value = self._long_rows[row][col]
                return f"{value:.4g}" if col == 5 else str(value)
            return None

        if col == 0:
            if role == Qt.DisplayRole:
                return str(self._query_ids[row])
        elif col == 1:
            if role == Qt.DisplayRole:
                return str(self._query_labels[row])
        else:
            j = col - 2
            nid = self._neigh_ids[row, j]
            if nid is None:
                return None
            dist = self._neigh_dists[row, j]
            if role == Qt.DisplayRole:
                return f"{nid}\n{dist:.3g}"
            if role == Qt.ToolTipRole:
                label = self._neigh_labels[row, j]
                return f"{nid}\n{label}\n{self._distance_label.lower()} {dist:.4g}"
        return None


class FindNearestWidget(QtWidgets.QWidget):
    """Per-neuron nearest-neighbour finder over a filtered candidate pool.

    Parameters
    ----------
    figure : ScatterFigure
        The scatter plot to read similarity data from and push found neighbours
        back into. Read live (never snapshotted) so switching the active
        embedding is picked up on the next Find. The *query* set is pinned from
        the current selection when the widget opens; use **From Selection** to
        recapture it, or **Sync to Selection** to follow it live.
    title : str
        Window title.
    parent : QWidget, optional
    """

    def __init__(self, figure, *, title="Find Nearest", parent=None):
        super().__init__(parent, Qt.Window)
        self._figure = figure
        self.setWindowTitle(title)
        self.resize(840, 560)

        self._filter_rows = []
        self._last = None  # (neigh_positions, dist, query_positions) of the last Find

        outer = QtWidgets.QVBoxLayout()
        outer.setContentsMargins(6, 6, 6, 6)
        outer.setSpacing(6)
        self.setLayout(outer)

        top = QtWidgets.QHBoxLayout()
        top.setContentsMargins(0, 0, 0, 0)
        top.setSpacing(6)
        outer.addLayout(top, 1)

        self._build_controls(top)
        self._build_table(top)
        self._build_bottom_row(outer)

        # Pin the query set from the current selection at open time.
        sel = getattr(figure, "selected", None)
        self._query_positions = np.asarray(sel if sel is not None else [], dtype=int)
        self._populate_sources()
        self._update_query_label()
        self._on_pool_changed()
        self._set_status("Set a candidate pool, then Find Nearest Neighbours.")

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def _build_controls(self, layout):
        panel = QtWidgets.QWidget()
        panel.setFixedWidth(300)
        panel.setSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Expanding
        )
        control_layout = QtWidgets.QVBoxLayout()
        control_layout.setContentsMargins(0, 0, 0, 0)
        control_layout.setSpacing(6)
        panel.setLayout(control_layout)

        # --- Query set -------------------------------------------------
        query_group = QtWidgets.QGroupBox("Query neurons")
        query_layout = QtWidgets.QVBoxLayout()
        query_layout.setContentsMargins(6, 6, 6, 6)
        query_layout.setSpacing(4)
        query_group.setLayout(query_layout)

        self._query_label = QtWidgets.QLabel()
        self._query_label.setWordWrap(True)
        query_layout.addWidget(self._query_label)

        self._from_selection_button = QtWidgets.QPushButton("From Selection")
        self._from_selection_button.setToolTip(
            "Capture the current figure selection as the query set"
        )
        self._from_selection_button.clicked.connect(self._on_from_selection)
        query_layout.addWidget(self._from_selection_button)

        self._sync_cb = QtWidgets.QCheckBox("Sync to Selection")
        self._sync_cb.setToolTip(
            "Keep the query set in step with the figure selection.\n"
            "Off (default): the query set stays pinned to what was selected "
            "when you opened the widget."
        )
        self._sync_cb.toggled.connect(self._on_sync_toggled)
        query_layout.addWidget(self._sync_cb)

        control_layout.addWidget(query_group)

        # --- Similarity ------------------------------------------------
        sim_group = QtWidgets.QGroupBox("Similarity")
        sim_form = QtWidgets.QFormLayout()
        sim_form.setContentsMargins(6, 6, 6, 6)
        sim_form.setVerticalSpacing(4)
        sim_group.setLayout(sim_form)

        self._source_combo = QtWidgets.QComboBox()
        self._source_combo.setToolTip("Which similarity space to measure distance in")
        self._source_combo.currentIndexChanged.connect(self._on_source_changed)
        sim_form.addRow("Source:", self._source_combo)

        self._metric_combo = QtWidgets.QComboBox()
        self._metric_combo.addItems(FEATURE_METRICS)
        self._metric_combo.setCurrentText(
            getattr(self._figure, "_gs_metric", "cosine")
            if getattr(self._figure, "_gs_metric", None) in FEATURE_METRICS
            else "cosine"
        )
        self._metric_combo.setToolTip("Distance metric (feature source only)")
        sim_form.addRow("Metric:", self._metric_combo)

        self._neighbours_spin = QtWidgets.QSpinBox()
        self._neighbours_spin.setRange(1, 100)
        self._neighbours_spin.setValue(5)
        self._neighbours_spin.setToolTip("Neighbours to find per query neuron")
        sim_form.addRow("Neighbours:", self._neighbours_spin)

        self._graph_expand_cb = QtWidgets.QCheckBox("Search through graph (approximate)")
        self._graph_expand_cb.setToolTip(
            "KNN source only. A KNN graph stores just each neuron's top-k "
            "neighbours, so a strict pool can leave fewer than N results.\n"
            "When on, the search walks the graph outward to reach further "
            "in-pool neighbours — but the reported distances are then "
            "accumulated graph-path distances, an approximation of the true "
            "distance, not a real query-to-neighbour distance."
        )
        sim_form.addRow(self._graph_expand_cb)

        control_layout.addWidget(sim_group)

        # --- Candidate pool -------------------------------------------
        pool_group = QtWidgets.QGroupBox("Candidate pool")
        pool_layout = QtWidgets.QVBoxLayout()
        pool_layout.setContentsMargins(6, 6, 6, 6)
        pool_layout.setSpacing(4)
        pool_group.setLayout(pool_layout)

        add_btn = QtWidgets.QPushButton("+ Add filter")
        add_btn.setToolTip("Restrict the pool of candidate neighbours by metadata")
        add_btn.clicked.connect(self._add_filter_row)
        pool_layout.addWidget(add_btn)

        self._filter_container = QtWidgets.QWidget()
        self._filter_container_layout = QtWidgets.QVBoxLayout()
        self._filter_container_layout.setContentsMargins(0, 0, 0, 0)
        self._filter_container_layout.setSpacing(2)
        self._filter_container.setLayout(self._filter_container_layout)
        pool_layout.addWidget(self._filter_container)

        self._pool_label = QtWidgets.QLabel()
        self._pool_label.setStyleSheet(f"color: {MUTED_TEXT};")
        pool_layout.addWidget(self._pool_label)

        self._exclude_queries_cb = QtWidgets.QCheckBox("Exclude query neurons from pool")
        self._exclude_queries_cb.setChecked(True)
        self._exclude_queries_cb.setToolTip(
            "Keep the selected (query) neurons out of the candidate pool, so they "
            "cannot turn up as their own or each other's neighbours.\n"
            "Turn off to let query neurons match each other (a neuron still never "
            "matches itself)."
        )
        pool_layout.addWidget(self._exclude_queries_cb)

        control_layout.addWidget(pool_group)

        # --- Find + results layout ------------------------------------
        self._find_button = QtWidgets.QPushButton("Find Nearest Neighbours")
        self._find_button.setDefault(True)
        self._find_button.clicked.connect(self._run_find)
        control_layout.addWidget(self._find_button)

        layout_row = QtWidgets.QHBoxLayout()
        layout_row.setContentsMargins(0, 0, 0, 0)
        layout_row.setSpacing(4)
        layout_row.addWidget(QtWidgets.QLabel("Table layout:"))
        self._layout_combo = QtWidgets.QComboBox()
        self._layout_combo.addItems(["Wide", "Long"])
        self._layout_combo.setToolTip(
            "Wide: neighbours across columns (one row per query).\n"
            "Long: one row per query–neighbour match."
        )
        self._layout_combo.currentTextChanged.connect(self._on_layout_changed)
        layout_row.addWidget(self._layout_combo, 1)
        control_layout.addLayout(layout_row)

        self._status_label = QtWidgets.QLabel()
        self._status_label.setWordWrap(True)
        control_layout.addWidget(self._status_label)

        control_layout.addStretch()
        layout.addWidget(panel)

    def _build_table(self, layout):
        self._table = QtWidgets.QTableView()
        self._model = FindNearestModel()
        self._table.setModel(self._model)
        self._table.setWordWrap(True)

        # Nudge the table font up a little for readability.
        table_font = self._table.font()
        table_font.setPointSizeF(table_font.pointSizeF() + 1.5)
        self._table.setFont(table_font)
        self._table.verticalHeader().setVisible(False)
        self._table.horizontalHeader().setSectionResizeMode(
            QtWidgets.QHeaderView.ResizeToContents
        )
        self._table.doubleClicked.connect(self._on_cell_double_clicked)
        layout.addWidget(self._table, 1)

    def _build_bottom_row(self, outer):
        """Window/table actions: Always on top (left), table actions (right)."""
        row = QtWidgets.QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(4)

        self._always_on_top = QtWidgets.QCheckBox("Always on top")
        self._always_on_top.setChecked(True)
        self._always_on_top.stateChanged.connect(self._update_always_on_top)
        row.addWidget(self._always_on_top)

        row.addStretch()

        self._add_button = QtWidgets.QPushButton("Add to Figure")
        self._add_button.setToolTip(
            "Add the union of all found neighbours to the figure selection"
        )
        self._add_button.clicked.connect(self._on_add)
        row.addWidget(self._add_button)

        self._open_button = QtWidgets.QPushButton("Open in New View")
        self._open_button.clicked.connect(self._on_open_new_view)
        row.addWidget(self._open_button)

        self._copy_button = QtWidgets.QPushButton("Copy")
        self._copy_button.clicked.connect(self._on_copy)
        row.addWidget(self._copy_button)

        self._export_button = QtWidgets.QPushButton("Export CSV")
        self._export_button.clicked.connect(self._on_export)
        row.addWidget(self._export_button)

        outer.addLayout(row)

    # ------------------------------------------------------------------
    # Source / metric / layout
    # ------------------------------------------------------------------

    def _populate_sources(self):
        """(Re)fill the source combo from the figure's currently available data."""
        sources = gs.available_sources(
            getattr(self._figure, "dists", None),
            getattr(self._figure, "positions", None),
        )
        current = self._source_combo.currentData()
        self._source_combo.blockSignals(True)
        self._source_combo.clear()
        for sid in sources:
            self._source_combo.addItem(SOURCE_LABELS.get(sid, sid), sid)
        if current in sources:
            self._source_combo.setCurrentIndex(sources.index(current))
        elif sources:
            preferred = getattr(self._figure, "_gs_source", None)
            self._source_combo.setCurrentIndex(
                sources.index(preferred) if preferred in sources else 0
            )
        self._source_combo.blockSignals(False)
        self._on_source_changed()

    def _on_source_changed(self, *_):
        source = self._source_combo.currentData()
        self._metric_combo.setEnabled(source == gs.SOURCE_FEATURES)
        self._graph_expand_cb.setEnabled(source == gs.SOURCE_KNN)
        has_source = self._source_combo.count() > 0
        self._find_button.setEnabled(has_source)
        if not has_source:
            self._set_status("No similarity data available for this embedding.", error=True)

    def _current_source(self):
        return self._source_combo.currentData()

    def _on_layout_changed(self, text):
        self._model.set_layout("long" if text == "Long" else "wide")
        self._table.resizeColumnsToContents()

    def showEvent(self, event):
        # The active embedding may have changed while we were hidden; refresh.
        self._populate_sources()
        super().showEvent(event)

    # ------------------------------------------------------------------
    # Query set
    # ------------------------------------------------------------------

    def select(self, indices):
        """Sync hook: only follows the figure selection when Sync is enabled."""
        if not self._sync_cb.isChecked():
            return
        self._query_positions = np.asarray(indices, dtype=int)
        self._update_query_label()

    def _on_from_selection(self):
        sel = getattr(self._figure, "selected", None)
        self._query_positions = np.asarray(sel if sel is not None else [], dtype=int)
        self._update_query_label()
        self._set_status(
            f"Captured {len(self._query_positions):,} query neurons from the selection."
        )

    def _on_sync_toggled(self, checked):
        # Turning sync on adopts the current selection immediately.
        if checked:
            sel = getattr(self._figure, "selected", None)
            self._query_positions = np.asarray(sel if sel is not None else [], dtype=int)
            self._update_query_label()

    def _update_query_label(self):
        n = len(self._query_positions)
        self._query_label.setText(
            f"{n:,} query neuron{'' if n == 1 else 's'}"
            if n
            else "No query neurons — use 'From Selection'"
        )

    # ------------------------------------------------------------------
    # Candidate pool filters
    # ------------------------------------------------------------------

    def _pool_columns(self):
        meta = self._figure.metadata
        return [c for c in meta.columns if not str(c).startswith("_")]

    def _add_filter_row(self):
        row = ScopeFilterRow(df_getter=lambda: self._figure.metadata)
        row.set_columns(self._pool_columns())
        row.set_first(len(self._filter_rows) == 0)
        row.changed.connect(self._on_pool_changed)
        row.removed.connect(self._remove_filter_row)
        self._filter_rows.append(row)
        self._filter_container_layout.addWidget(row)
        self._on_pool_changed()

    def _remove_filter_row(self, row):
        if row in self._filter_rows:
            self._filter_rows.remove(row)
        row.setParent(None)
        row.deleteLater()
        if self._filter_rows:
            self._filter_rows[0].set_first(True)
        self._on_pool_changed()

    def _pool_mask(self):
        """Boolean mask (length N) of the metadata-filtered candidate pool."""
        df = self._figure.metadata
        if not self._filter_rows:
            return np.ones(len(df), dtype=bool)
        mask = np.asarray(self._filter_rows[0].mask(df), dtype=bool)
        for row in self._filter_rows[1:]:
            m = np.asarray(row.mask(df), dtype=bool)
            mask = (mask | m) if row.combinator.currentText() == "OR" else (mask & m)
        return mask

    def _on_pool_changed(self, *_):
        mask = self._pool_mask()
        self._pool_label.setText(
            f"Pool: {int(mask.sum()):,} of {len(mask):,} neurons match"
        )

    # ------------------------------------------------------------------
    # Compute
    # ------------------------------------------------------------------

    def _labels(self):
        """Positional label array aligned to the figure's rows."""
        labels = getattr(self._figure, "labels", None)
        if labels is not None:
            return np.asarray(labels)
        return np.asarray(self._figure.ids).astype(str)

    def _run_find(self):
        fig = self._figure
        query = np.asarray(self._query_positions, dtype=int)
        if not len(query):
            self._model.clear()
            self._set_status(
                "No query neurons — select some and press 'From Selection'.", error=True
            )
            return

        source = self._current_source()
        if source is None:
            self._set_status("No similarity data available.", error=True)
            return

        pool = self._pool_mask()
        if self._exclude_queries_cb.isChecked():
            pool = pool.copy()
            pool[query] = False
        if not pool.any():
            self._model.clear()
            self._set_status("Candidate pool is empty — loosen the filters.", error=True)
            return

        n = self._neighbours_spin.value()
        metric = self._metric_combo.currentText()
        graph_expand = source == gs.SOURCE_KNN and self._graph_expand_cb.isChecked()
        try:
            neigh, dist = gs.per_query_neighbors(
                query,
                n,
                source=source,
                positions=fig.positions,
                dists=fig.dists,
                metric=metric,
                pool_mask=pool,
                exclude_query=True,  # a neuron never matches itself
                graph_expand=graph_expand,
            )
        except gs.GrowShrinkUnavailable as e:
            self._set_status(str(e), error=True)
            return
        except ValueError:
            self._set_status(
                "Feature vectors contain NaNs — pick another source or metric.",
                error=True,
            )
            return

        ids = np.asarray(fig.ids)
        labels = self._labels()
        m, k = neigh.shape
        neigh_ids = np.empty((m, k), dtype=object)
        neigh_labels = np.empty((m, k), dtype=object)
        for i in range(m):
            for j in range(k):
                p = neigh[i, j]
                if p >= 0:
                    neigh_ids[i, j] = ids[p]
                    neigh_labels[i, j] = labels[p]

        self._model.set_distance_label("Graph distance" if graph_expand else "Distance")
        self._model.set_results(ids[query], labels[query], neigh_ids, neigh_labels, dist)
        self._last = (neigh, dist, query)
        self._table.resizeColumnsToContents()

        n_found = int(np.unique(neigh[neigh >= 0]).size)
        short = int((neigh < 0).any(axis=1).sum())
        msg = f"{n_found:,} distinct neighbours for {m:,} query neurons."
        if short:
            msg += f" {short:,} had fewer than {n} in-pool neighbours."
        if graph_expand:
            msg += " Distances are approximate graph-path distances."
        self._set_status(msg)

    def _neighbour_positions(self):
        """De-duplicated row positions of all neighbours from the last Find."""
        if self._last is None:
            return np.empty(0, dtype=int)
        neigh = self._last[0]
        return np.unique(neigh[neigh >= 0])

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------

    def _on_add(self):
        neighbours = self._neighbour_positions()
        if not len(neighbours):
            self._set_status(
                "Nothing to add — run Find Nearest Neighbours first.", error=True
            )
            return
        current = getattr(self._figure, "selected", None)
        current = np.asarray(current, dtype=int) if current is not None else np.empty(0, dtype=int)
        new_total = np.union1d(current, neighbours)
        added = len(new_total) - len(current)
        try:
            self._figure.selected = new_total
        except Exception as e:  # pragma: no cover - defensive
            self._set_status(f"Could not update selection: {e}", error=True)
            return
        self._set_status(
            f"Added {added:,} neighbour neuron{'' if added == 1 else 's'} to the selection."
        )

    def _on_open_new_view(self):
        positions = self._neighbour_positions()
        if not len(positions):
            self._set_status("Nothing to open — run Find Nearest Neighbours first.", error=True)
            return
        try:
            self._figure.open_selection_in_new_tab(ind=positions)
        except Exception as e:  # pragma: no cover - defensive
            self._set_status(f"Could not open new view: {e}", error=True)

    def _results_dataframe(self):
        """The last results as a flat DataFrame, matching the current layout."""
        if self._last is None:
            return None
        neigh, dist, query = self._last
        ids = np.asarray(self._figure.ids)
        labels = self._labels()
        m, k = neigh.shape

        if self._model._layout == "long":
            rows = []
            for i in range(m):
                for j in range(k):
                    p = neigh[i, j]
                    if p < 0:
                        continue
                    rows.append(
                        {
                            "query_id": ids[query[i]],
                            "query_label": labels[query[i]],
                            "rank": j + 1,
                            "neighbour_id": ids[p],
                            "neighbour_label": labels[p],
                            "distance": dist[i, j],
                        }
                    )
            return pd.DataFrame(
                rows,
                columns=[
                    "query_id",
                    "query_label",
                    "rank",
                    "neighbour_id",
                    "neighbour_label",
                    "distance",
                ],
            )

        data = {"query_id": ids[query], "query_label": labels[query]}
        for j in range(k):
            data[f"nn{j + 1}_id"] = [ids[p] if p >= 0 else "" for p in neigh[:, j]]
            data[f"nn{j + 1}_dist"] = [
                dist[i, j] if neigh[i, j] >= 0 else np.nan for i in range(m)
            ]
        return pd.DataFrame(data)

    def _on_copy(self):
        df = self._results_dataframe()
        if df is None or df.empty:
            self._set_status("Nothing to copy — run Find Nearest Neighbours first.", error=True)
            return
        if len(df) > MAX_CLIPBOARD_ROWS:
            QtWidgets.QMessageBox.warning(
                self,
                "Too many rows",
                f"{len(df):,} rows is more than the {MAX_CLIPBOARD_ROWS} we copy "
                "to the clipboard. Use Export CSV instead.",
            )
            return
        df.to_clipboard(index=False)
        self._set_status(f"Copied {len(df):,} rows to the clipboard.")

    def _on_export(self):
        df = self._results_dataframe()
        if df is None or df.empty:
            self._set_status("Nothing to export — run Find Nearest Neighbours first.", error=True)
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Export nearest neighbours", "nearest_neighbours.csv", "CSV files (*.csv)"
        )
        if not path:
            return
        try:
            df.to_csv(path, index=False)
        except OSError as e:
            QtWidgets.QMessageBox.critical(self, "Export failed", str(e))
            return
        self._set_status(f"Exported {len(df):,} rows.")

    def _on_cell_double_clicked(self, index):
        """Locate a double-clicked query/neighbour id in the scatter plot."""
        model = self._model
        row, col = index.row(), index.column()
        if model._layout == "long":
            value = model._long_rows[row][col] if col in (0, 3) else None
        elif col == 0:
            value = model._query_ids[row]
        elif col >= 2:
            value = model._neigh_ids[row, col - 2]
        else:
            value = None
        if value is None:
            return
        if hasattr(self._figure, "find_label"):
            self._figure.find_label(str(value), regex=True)

    # ------------------------------------------------------------------
    # Misc
    # ------------------------------------------------------------------

    def _update_always_on_top(self, *_):
        self.setWindowFlag(Qt.WindowStaysOnTopHint, self._always_on_top.isChecked())
        self.show()

    def _set_status(self, message, error=False):
        self._status_label.setStyleSheet(
            "color: #cc3a3a;" if error else f"color: {MUTED_TEXT};"
        )
        self._status_label.setText(message)
