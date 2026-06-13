import re
import logging
import warnings
import pyperclip

import cmap as _cmap
import pandas as pd
import numpy as np

from PySide6 import QtWidgets, QtCore, QtGui
from concurrent.futures import ThreadPoolExecutor

from ...embeddings import (
    is_knn_graph,
    is_precomputed_distance_matrix,
    make_embedding_estimator,
    make_knn_embedding_estimator,
    prepare_embedding_input,
    sanitize_embedding,
)
from ...clusters import evaluate_clustering_sample
from ...utils import labels_to_colors, is_color_column


CLIO_CLIENT = None
CLIO_ANN = None
NEUPRINT_CLIENT = None
FLYWIRE_ANN = None
HB_ANN = None

logger = logging.getLogger(__name__)


CLUSTER_DATA_EMBEDDING_OPTION = "embedding (2D)"
CLUSTER_DATA_OPTION = "cluster (bigclust)"
CLUSTER_DATA_COLUMN = "bigclust_cluster"
EVALUATE_DATA_COLUMN = "bigclust_label_evaluation"
FIDELITY_DATA_COLUMN = "bigclust_fidelity"
CLUSTER_HOMOGENEOUS_LABEL_CURRENT = "Current labels"


_DIVERGING_PALETTES = frozenset({
    "matplotlib:coolwarm",
    "colorbrewer:RdBu_r",
    "matplotlib:seismic",
})


def _normalize_sizes(values, smin=0.1, smax=1.0, vmin=None, vcenter=None, vmax=None):
    """Map numeric values to point sizes in [smin, smax].

    Values are mapped piecewise-linearly (with clipping) such that `vmin`
    -> smin, `vcenter` -> (smin + smax) / 2 and `vmax` -> smax; these
    default to the data minimum, midpoint and maximum, respectively.
    NaNs map to `smin`; constant or all-NaN input maps to uniform 1.0
    (the default point size).
    """
    values = np.asarray(values, dtype=float)
    finite = np.isfinite(values)

    if not finite.any():
        return np.ones(len(values), dtype=np.float32)

    if vmin is None:
        vmin = values[finite].min()
    if vmax is None:
        vmax = values[finite].max()
    if vcenter is None:
        vcenter = (vmin + vmax) / 2
    vcenter = min(max(vcenter, vmin), vmax)

    if vmax == vmin:
        return np.ones(len(values), dtype=np.float32)

    sizes = np.full(len(values), smin, dtype=np.float32)
    sizes[finite] = np.interp(
        values[finite], [vmin, vcenter, vmax], [smin, (smin + smax) / 2, smax]
    )
    return sizes


def _make_palette_pixmap(palette_name, width=100, height=14):
    """Return a QPixmap showing a gradient swatch for the given cmap palette."""
    colormap = _cmap.Colormap(palette_name)
    colors = list(colormap.iter_colors(width))
    pixmap = QtGui.QPixmap(width, height)
    painter = QtGui.QPainter(pixmap)
    for i, color in enumerate(colors):
        r, g, b, a = color.rgba
        painter.fillRect(i, 0, 1, height, QtGui.QColor.fromRgbF(r, g, b, a))
    painter.end()
    return pixmap


class _MultiHandleSlider(QtWidgets.QWidget):
    """Horizontal slider with 2 (min/max) or 3 (min/center/max) draggable handles.

    Emits ``valuesChanged(vmin, vcenter, vmax)`` whenever any handle moves.
    The centre handle is hidden unless ``set_diverging(True)`` is called.
    """

    valuesChanged = QtCore.Signal(float, float, float)

    _HANDLE_R = 6      # handle radius in px
    _TRACK_H = 4       # track height in px
    _H_MARGIN = 14     # left margin (≥ handle radius + a few px)
    _H_MARGIN_RIGHT = 28  # right margin — wider to accommodate the reset button
    _LABEL_AREA = 16   # px below the track reserved for value labels
    _MIN_H = 46        # total widget height
    _RESET_BTN_SIZE = 16  # reset button side length in px
    _HIST_H = 18       # px above the track reserved for the distribution histogram

    def __init__(self, parent=None):
        super().__init__(parent)
        self._lo: float = 0.0
        self._hi: float = 1.0
        self._vmin: float = 0.0
        self._vcenter: float = 0.5
        self._vmax: float = 1.0
        self._show_center: bool = False
        self._active: "str | None" = None   # 'min' | 'center' | 'max'
        self._hover: "str | None" = None
        self._dist_values: "np.ndarray | None" = None
        self._hist_cache: "tuple[tuple[int, float, float], np.ndarray] | None" = None
        self.setMinimumHeight(self._MIN_H)
        self.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Fixed,
        )
        self.setMouseTracking(True)
        self.setFocusPolicy(QtCore.Qt.FocusPolicy.NoFocus)

        # Small reset button — positioned in resizeEvent
        self._reset_btn = QtWidgets.QToolButton(self)
        self._reset_btn.setText("↺")
        self._reset_btn.setFixedSize(self._RESET_BTN_SIZE, self._RESET_BTN_SIZE)
        self._reset_btn.setToolTip("Reset range to full data extent")
        self._reset_btn.setAutoRaise(True)
        self._reset_btn.setStyleSheet(
            "QToolButton { border: none; padding: 0px; font-size: 11px; color: palette(mid); }"
            "QToolButton:hover { color: palette(text); }"
        )
        self._reset_btn.clicked.connect(self.reset)

        # Debounce timer: emit valuesChanged only after a short pause while dragging
        self._emit_timer = QtCore.QTimer(self)
        self._emit_timer.setSingleShot(True)
        self._emit_timer.setInterval(120)  # ms — adjust to taste
        self._emit_timer.timeout.connect(self._flush_emit)

    # ── public API ────────────────────────────────────────────────────────────

    @property
    def vmin(self) -> float:
        return self._vmin

    @property
    def vcenter(self) -> float:
        return self._vcenter

    @property
    def vmax(self) -> float:
        return self._vmax

    @property
    def show_center(self) -> bool:
        return self._show_center

    @property
    def lo(self) -> float:
        return self._lo

    @property
    def hi(self) -> float:
        return self._hi

    def set_data_range(self, lo: float, hi: float) -> None:
        """Update the track bounds. Does not emit."""
        self._lo = float(lo)
        self._hi = float(hi) if hi != lo else float(lo) + 1.0
        self._hist_cache = None
        self.update()

    def set_distribution(self, values=None) -> None:
        """Show a small histogram of `values` above the track; None removes it."""
        if values is not None:
            values = np.asarray(values, dtype=float)
            values = values[np.isfinite(values)]
            if not values.size:
                values = None
        self._dist_values = values
        self._hist_cache = None
        self.setMinimumHeight(
            self._MIN_H + (self._HIST_H if values is not None else 0)
        )
        self.updateGeometry()
        self.update()

    def set_values(self, vmin: float, vcenter: float, vmax: float) -> None:
        """Set all three handle positions. Does not emit."""
        self._vmin = float(vmin)
        self._vcenter = float(vcenter)
        self._vmax = float(vmax)
        self.update()

    def set_diverging(self, diverging: bool) -> None:
        """Show or hide the centre handle."""
        self._show_center = bool(diverging)
        self.update()

    def reset(self) -> None:
        """Reset handles to the full data range and emit valuesChanged."""
        self._vmin = self._lo
        self._vmax = self._hi
        self._vcenter = (self._lo + self._hi) / 2.0
        self.update()
        self.valuesChanged.emit(self._vmin, self._vcenter, self._vmax)

    # ── coordinate helpers ────────────────────────────────────────────────────

    def _track_y(self) -> int:
        hist_h = self._HIST_H if self._dist_values is not None else 0
        usable = self.height() - self._LABEL_AREA - hist_h
        return hist_h + max(self._HANDLE_R + 2, usable // 2)

    def _hist_counts(self) -> "np.ndarray | None":
        """Normalized bin heights for the distribution, cached per width/range."""
        if self._dist_values is None:
            return None
        key = (self._track_width(), self._lo, self._hi)
        if self._hist_cache is None or self._hist_cache[0] != key:
            n_bins = max(10, self._track_width() // 3)
            counts, _ = np.histogram(
                self._dist_values, bins=n_bins, range=(self._lo, self._hi)
            )
            peak = counts.max()
            heights = counts / peak if peak else counts.astype(float)
            self._hist_cache = (key, heights)
        return self._hist_cache[1]

    def _track_left(self) -> int:
        return self._H_MARGIN

    def _track_right(self) -> int:
        return self.width() - self._H_MARGIN_RIGHT

    def _track_width(self) -> int:
        return max(1, self._track_right() - self._track_left())

    def _val_to_x(self, val: float) -> int:
        span = self._hi - self._lo
        ratio = (val - self._lo) / span if span else 0.0
        return self._track_left() + int(ratio * self._track_width())

    def _x_to_val(self, x: int) -> float:
        tw = self._track_width()
        ratio = (x - self._track_left()) / tw if tw else 0.0
        return self._lo + max(0.0, min(1.0, ratio)) * (self._hi - self._lo)

    def _handles(self) -> "list[tuple[str, int, float]]":
        result = [
            ("min", self._val_to_x(self._vmin), self._vmin),
            ("max", self._val_to_x(self._vmax), self._vmax),
        ]
        if self._show_center:
            result.append(("center", self._val_to_x(self._vcenter), self._vcenter))
        return result

    def _hit_test(self, pos: QtCore.QPoint) -> "str | None":
        cy = self._track_y()
        r = self._HANDLE_R + 4
        best: "str | None" = None
        best_dist = r + 1.0
        for name, hx, _ in self._handles():
            dist = ((pos.x() - hx) ** 2 + (pos.y() - cy) ** 2) ** 0.5
            if dist <= r and dist < best_dist:
                best = name
                best_dist = dist
        return best

    # ── layout ───────────────────────────────────────────────────────────────

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        # Keep the reset button at the top-right corner
        s = self._RESET_BTN_SIZE
        self._reset_btn.setGeometry(self.width() - s - 1, 1, s, s)

    # ── events ────────────────────────────────────────────────────────────────

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:
        if event.button() == QtCore.Qt.MouseButton.LeftButton:
            hit = self._hit_test(event.position().toPoint())
            if hit:
                self._active = hit
                self._apply_drag(event.position().toPoint().x())
                event.accept()
                return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QtGui.QMouseEvent) -> None:
        pt = event.position().toPoint()
        if self._active and (event.buttons() & QtCore.Qt.MouseButton.LeftButton):
            self._apply_drag(pt.x())
            event.accept()
            return
        old_hover = self._hover
        self._hover = self._hit_test(pt)
        if self._hover != old_hover:
            self.update()
        self.setCursor(
            QtGui.QCursor(QtCore.Qt.CursorShape.SizeHorCursor)
            if self._hover
            else QtGui.QCursor(QtCore.Qt.CursorShape.ArrowCursor)
        )
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QtGui.QMouseEvent) -> None:
        if event.button() == QtCore.Qt.MouseButton.LeftButton and self._active:
            self._active = None
            # Flush any pending debounced emit immediately on mouse-up
            if self._emit_timer.isActive():
                self._emit_timer.stop()
                self._flush_emit()
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def leaveEvent(self, event) -> None:
        if self._hover:
            self._hover = None
            self.update()
        super().leaveEvent(event)

    def _apply_drag(self, x: int) -> None:
        val = self._x_to_val(x)
        if self._active == "min":
            new_min = min(val, self._vmax)
            if self._show_center:
                span = self._vmax - self._vmin
                ratio = (self._vcenter - self._vmin) / span if span else 0.5
                self._vcenter = new_min + ratio * (self._vmax - new_min)
            self._vmin = new_min
        elif self._active == "max":
            new_max = max(val, self._vmin)
            if self._show_center:
                span = self._vmax - self._vmin
                ratio = (self._vcenter - self._vmin) / span if span else 0.5
                self._vcenter = self._vmin + ratio * (new_max - self._vmin)
            self._vmax = new_max
        elif self._active == "center":
            self._vcenter = max(self._vmin, min(val, self._vmax))
        self.update()
        self._emit_timer.start()  # restarts the timer if already running

    def _flush_emit(self) -> None:
        self.valuesChanged.emit(self._vmin, self._vcenter, self._vmax)

    # ── painting ──────────────────────────────────────────────────────────────

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)

        pal = self.palette()
        cy = self._track_y()
        tl = self._track_left()
        tr_r = self._track_right()
        th = self._TRACK_H

        # Distribution histogram above the track
        heights = self._hist_counts()
        if heights is not None:
            baseline = cy - self._HANDLE_R - 2
            bin_w = (tr_r - tl) / len(heights)
            span = self._hi - self._lo
            in_col = QtGui.QColor(255, 255, 255, 200)
            out_col = QtGui.QColor(255, 255, 255, 80)
            painter.setPen(QtCore.Qt.PenStyle.NoPen)
            for i, h in enumerate(heights):
                if not h:
                    continue
                bin_center = self._lo + (i + 0.5) / len(heights) * span
                in_range = self._vmin <= bin_center <= self._vmax
                painter.setBrush(in_col if in_range else out_col)
                bar_h = h * self._HIST_H
                painter.drawRect(
                    QtCore.QRectF(tl + i * bin_w, baseline - bar_h, bin_w, bar_h)
                )

        # Groove
        groove = QtCore.QRectF(tl, cy - th / 2, tr_r - tl, th)
        painter.setPen(QtCore.Qt.PenStyle.NoPen)
        painter.setBrush(pal.color(QtGui.QPalette.ColorRole.Mid))
        painter.drawRoundedRect(groove, th / 2, th / 2)

        # Filled range between vmin and vmax
        x_min = self._val_to_x(self._vmin)
        x_max = self._val_to_x(self._vmax)
        if x_max > x_min:
            fill = QtCore.QRectF(x_min, cy - th / 2, x_max - x_min, th)
            painter.setBrush(pal.color(QtGui.QPalette.ColorRole.Highlight))
            painter.drawRoundedRect(fill, th / 2, th / 2)

        # Handles + labels
        font = painter.font()
        label_font = QtGui.QFont(font)
        label_font.setPointSizeF(max(7.0, font.pointSizeF() - 1.5))

        for name, hx, val in self._handles():
            is_active = name == self._active
            is_hover = name == self._hover
            is_center = name == "center"

            if is_active:
                brush = pal.color(QtGui.QPalette.ColorRole.Highlight)
                pen_col = pal.color(QtGui.QPalette.ColorRole.Dark)
            elif is_hover:
                brush = pal.color(QtGui.QPalette.ColorRole.Light)
                pen_col = pal.color(QtGui.QPalette.ColorRole.Dark)
            elif is_center:
                brush = pal.color(QtGui.QPalette.ColorRole.AlternateBase)
                pen_col = pal.color(QtGui.QPalette.ColorRole.Dark)
            else:
                brush = pal.color(QtGui.QPalette.ColorRole.Button)
                pen_col = pal.color(QtGui.QPalette.ColorRole.Dark)

            r = self._HANDLE_R
            painter.setPen(QtGui.QPen(pen_col, 1.5))
            painter.setBrush(brush)
            painter.drawEllipse(QtCore.QPointF(hx, cy), r, r)

            # Value label below the handle
            label = _MultiHandleSlider._fmt(val)
            label_rect = QtCore.QRectF(hx - 28, cy + r + 2, 56, self._LABEL_AREA - 2)
            painter.setPen(pal.color(QtGui.QPalette.ColorRole.Text))
            painter.setFont(label_font)
            painter.drawText(
                label_rect,
                QtCore.Qt.AlignmentFlag.AlignHCenter | QtCore.Qt.AlignmentFlag.AlignTop,
                label,
            )
            painter.setFont(font)
            painter.setPen(QtGui.QPen(pen_col, 1.5))

        painter.end()

    @staticmethod
    def _fmt(v: float) -> str:
        if v == 0.0:
            return "0"
        mag = abs(v)
        if 0.001 <= mag < 10_000:
            return f"{v:.4g}"
        return f"{v:.2e}"


def requires_selection(func):
    """Decorator to check if a selection is required."""

    def wrapper(self, *args, **kwargs):
        if self.figure.selected is None or len(self.figure.selected) == 0:
            self.figure.show_message("No neurons selected", color="red", duration=2)
            return
        return func(self, *args, **kwargs)

    return wrapper


class _FilterableComboBox(QtWidgets.QComboBox):
    """A QComboBox whose popup has a filter text field at the top.

    All standard QComboBox API (addItem, removeItem, findText, setCurrentText,
    currentText, currentIndexChanged, blockSignals, etc.) works unchanged.
    Only the visual popup is replaced with a custom frame containing a
    QLineEdit for filtering and a QListView for the matching items.
    """

    def __init__(self, parent=None):
        super().__init__(parent)

        # Build the custom popup frame (parented to None so it floats freely)
        self._popup = QtWidgets.QFrame(
            None,
            QtCore.Qt.WindowType.Popup | QtCore.Qt.WindowType.FramelessWindowHint,
        )
        self._popup.setFrameShape(QtWidgets.QFrame.Shape.StyledPanel)
        popup_layout = QtWidgets.QVBoxLayout(self._popup)
        popup_layout.setContentsMargins(4, 4, 4, 4)
        popup_layout.setSpacing(2)

        self._filter_edit = QtWidgets.QLineEdit()
        self._filter_edit.setPlaceholderText("Filter…")
        self._filter_edit.setClearButtonEnabled(True)
        popup_layout.addWidget(self._filter_edit)

        self._list_view = QtWidgets.QListView()
        self._list_view.setEditTriggers(
            QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers
        )
        self._list_view.setHorizontalScrollBarPolicy(
            QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        popup_layout.addWidget(self._list_view)

        # Proxy model for filtering — wraps this combo's own model
        self._proxy = QtCore.QSortFilterProxyModel(self)
        self._proxy.setFilterCaseSensitivity(QtCore.Qt.CaseSensitivity.CaseInsensitive)
        self._proxy.setFilterKeyColumn(0)
        self._list_view.setModel(self._proxy)

        # Connections
        self._filter_edit.textChanged.connect(self._on_filter_text_changed)
        self._list_view.clicked.connect(self._on_item_clicked)
        self._filter_edit.installEventFilter(self)
        self._list_view.installEventFilter(self)

    # ------------------------------------------------------------------
    # Filtering
    # ------------------------------------------------------------------

    def _on_filter_text_changed(self, text: str) -> None:
        escaped = QtCore.QRegularExpression.escape(text)
        self._proxy.setFilterRegularExpression(
            QtCore.QRegularExpression(escaped, QtCore.QRegularExpression.PatternOption.CaseInsensitiveOption)
        )
        # Select the first visible item in the list so Enter works immediately
        first = self._proxy.index(0, 0)
        if first.isValid():
            self._list_view.setCurrentIndex(first)
        else:
            self._list_view.clearSelection()

    # ------------------------------------------------------------------
    # Popup lifecycle
    # ------------------------------------------------------------------

    def showPopup(self) -> None:
        # Refresh the proxy source in case the underlying model was replaced
        self._proxy.setSourceModel(self.model())

        # Clear any previous filter text
        self._filter_edit.blockSignals(True)
        self._filter_edit.clear()
        self._filter_edit.blockSignals(False)
        self._proxy.setFilterRegularExpression(QtCore.QRegularExpression())

        # Size the popup
        min_width = max(self.width(), 200)
        row_height = self._list_view.sizeHintForRow(0)
        if row_height < 1:
            row_height = 22
        list_height = max(80, min(300, self.count() * row_height))
        self._list_view.setFixedHeight(list_height)
        self._popup.setFixedWidth(min_width)
        self._popup.adjustSize()

        # Position directly below the combo button
        global_pos = self.mapToGlobal(QtCore.QPoint(0, self.height()))
        self._popup.move(global_pos)
        self._popup.show()

        # Pre-select the item that is currently active in the combo
        current_source_idx = self.model().index(self.currentIndex(), 0)
        proxy_idx = self._proxy.mapFromSource(current_source_idx)
        if proxy_idx.isValid():
            self._list_view.setCurrentIndex(proxy_idx)
            self._list_view.scrollTo(
                proxy_idx, QtWidgets.QAbstractItemView.ScrollHint.PositionAtCenter
            )
        else:
            first = self._proxy.index(0, 0)
            if first.isValid():
                self._list_view.setCurrentIndex(first)

        self._filter_edit.setFocus(QtCore.Qt.FocusReason.PopupFocusReason)

    def hidePopup(self) -> None:
        self._popup.hide()
        super().hidePopup()

    # ------------------------------------------------------------------
    # Item selection from list view
    # ------------------------------------------------------------------

    def _on_item_clicked(self, proxy_index: QtCore.QModelIndex) -> None:
        source_index = self._proxy.mapToSource(proxy_index)
        if source_index.isValid():
            self.setCurrentIndex(source_index.row())
        self._popup.hide()

    def _select_current_and_close(self) -> None:
        """Select the currently highlighted list item and close the popup."""
        proxy_index = self._list_view.currentIndex()
        if not proxy_index.isValid():
            # Fall back to first visible item
            proxy_index = self._proxy.index(0, 0)
        if proxy_index.isValid():
            source_index = self._proxy.mapToSource(proxy_index)
            if source_index.isValid():
                self.setCurrentIndex(source_index.row())
        self._popup.hide()

    # ------------------------------------------------------------------
    # Keyboard navigation
    # ------------------------------------------------------------------

    def eventFilter(self, obj, event) -> bool:
        if event.type() == QtCore.QEvent.Type.KeyPress:
            key = event.key()
            if obj is self._filter_edit:
                if key in (QtCore.Qt.Key.Key_Return, QtCore.Qt.Key.Key_Enter):
                    self._select_current_and_close()
                    return True
                if key == QtCore.Qt.Key.Key_Escape:
                    self._popup.hide()
                    return True
                if key == QtCore.Qt.Key.Key_Down:
                    cur = self._list_view.currentIndex()
                    next_row = cur.row() + 1 if cur.isValid() else 0
                    next_idx = self._proxy.index(next_row, 0)
                    if next_idx.isValid():
                        self._list_view.setCurrentIndex(next_idx)
                    return True
                if key == QtCore.Qt.Key.Key_Up:
                    cur = self._list_view.currentIndex()
                    prev_row = max(0, cur.row() - 1) if cur.isValid() else 0
                    prev_idx = self._proxy.index(prev_row, 0)
                    if prev_idx.isValid():
                        self._list_view.setCurrentIndex(prev_idx)
                    return True
            elif obj is self._list_view:
                if key in (QtCore.Qt.Key.Key_Return, QtCore.Qt.Key.Key_Enter):
                    self._select_current_and_close()
                    return True
                if key == QtCore.Qt.Key.Key_Escape:
                    self._popup.hide()
                    return True
        return super().eventFilter(obj, event)


class _ScopeFilterRow(QtWidgets.QWidget):
    """A single scope filter: column picker plus a dtype-specific editor.

    Numeric columns get a range slider with editable min/max fields,
    low-cardinality categorical columns get checkboxes and high-cardinality
    ones a substring/regex filter field with a live match count.

    Emits ``changed`` whenever the filter may produce a different mask and
    ``removed(self)`` when the user clicks the remove button.
    """

    changed = QtCore.Signal()
    removed = QtCore.Signal(object)

    # Categorical columns with up to this many unique values get checkboxes,
    # larger ones a text filter field
    MAX_CHECKBOX_VALUES = 10

    def __init__(self, df_getter, parent=None):
        super().__init__(parent)
        self._df_getter = df_getter
        self._editor_kind = None
        self._value_checks = {}
        self._range_slider = None
        self._min_spin = None
        self._max_spin = None
        self._text_edit = None

        layout = QtWidgets.QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)
        self.setLayout(layout)

        header = QtWidgets.QHBoxLayout()
        header.setContentsMargins(0, 0, 0, 0)
        header.setSpacing(2)
        layout.addLayout(header)

        self.combinator = QtWidgets.QComboBox()
        self.combinator.addItems(["AND", "OR"])
        self.combinator.setFixedWidth(60)
        self.combinator.setToolTip("How to combine this filter with the one above")
        self.combinator.currentIndexChanged.connect(lambda *_: self.changed.emit())
        header.addWidget(self.combinator)

        self.column_combo = _FilterableComboBox()
        self.column_combo.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Fixed,
        )
        self.column_combo.currentIndexChanged.connect(self._on_column_changed)
        header.addWidget(self.column_combo)

        self.remove_btn = QtWidgets.QToolButton()
        self.remove_btn.setText("×")
        self.remove_btn.setAutoRaise(True)
        self.remove_btn.setToolTip("Remove this filter")
        self.remove_btn.clicked.connect(lambda: self.removed.emit(self))
        header.addWidget(self.remove_btn)

        self.editor_area = QtWidgets.QWidget()
        editor_layout = QtWidgets.QVBoxLayout()
        editor_layout.setContentsMargins(12, 0, 0, 0)
        editor_layout.setSpacing(2)
        self.editor_area.setLayout(editor_layout)
        layout.addWidget(self.editor_area)

    def set_first(self, is_first):
        """Hide the AND/OR combinator on the first row."""
        self.combinator.setVisible(not is_first)

    def set_columns(self, cols):
        """Refresh the column picker, keeping the current column if possible."""
        current = self.column_combo.currentText()
        self.column_combo.blockSignals(True)
        self.column_combo.clear()
        self.column_combo.addItems(cols)
        if current in cols:
            self.column_combo.setCurrentText(current)
        self.column_combo.blockSignals(False)
        # Rebuild the editor against the (possibly new) data
        self._on_column_changed()

    # ── editors ───────────────────────────────────────────────────────────────

    def _clear_editor(self):
        layout = self.editor_area.layout()
        while layout.count():
            item = layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        self._editor_kind = None
        self._value_checks = {}
        self._range_slider = None
        self._min_spin = None
        self._max_spin = None
        self._text_edit = None

    def _on_column_changed(self):
        self._clear_editor()
        layout = self.editor_area.layout()

        df = self._df_getter()
        col = self.column_combo.currentText()
        if df is not None and col in df.columns:
            series = df[col]
            if series.dtype.kind in "iuf":
                self._build_numeric_editor(layout, series)
            else:
                uniques = series.dropna().astype(str).unique()
                if len(uniques) <= self.MAX_CHECKBOX_VALUES:
                    self._build_checkbox_editor(layout, uniques)
                else:
                    self._build_text_editor(layout)
        self.changed.emit()

    def _build_numeric_editor(self, layout, series):
        vals = series.to_numpy(dtype=float)
        finite = vals[np.isfinite(vals)]
        if not len(finite):
            hint = QtWidgets.QLabel("No finite values in this column")
            hint.setStyleSheet("color: palette(mid);")
            layout.addWidget(hint)
            return
        lo, hi = float(finite.min()), float(finite.max())

        self._editor_kind = "numeric"
        self._range_slider = _MultiHandleSlider()
        self._range_slider.set_data_range(lo, hi)
        self._range_slider.set_values(lo, (lo + hi) / 2, hi)
        self._range_slider.set_distribution(finite)
        layout.addWidget(self._range_slider)

        spin_row = QtWidgets.QWidget()
        spin_layout = QtWidgets.QHBoxLayout()
        spin_layout.setContentsMargins(0, 0, 0, 0)
        spin_layout.setSpacing(2)
        spin_row.setLayout(spin_layout)
        decimals = 0 if series.dtype.kind in "iu" else 4
        step = (hi - lo) / 100 if hi > lo else 1.0
        self._min_spin = QtWidgets.QDoubleSpinBox()
        self._max_spin = QtWidgets.QDoubleSpinBox()
        for spin, value in ((self._min_spin, lo), (self._max_spin, hi)):
            spin.setDecimals(decimals)
            spin.setRange(lo, hi)
            spin.setSingleStep(step)
            spin.setValue(value)
            spin.setKeyboardTracking(False)
            spin_layout.addWidget(spin)
        layout.addWidget(spin_row)

        self._range_slider.valuesChanged.connect(self._on_slider_changed)
        self._min_spin.valueChanged.connect(self._on_spin_changed)
        self._max_spin.valueChanged.connect(self._on_spin_changed)

    def _build_checkbox_editor(self, layout, uniques):
        self._editor_kind = "checks"
        for value in sorted(uniques):
            check = QtWidgets.QCheckBox(value)
            check.setChecked(True)
            check.stateChanged.connect(lambda *_: self.changed.emit())
            layout.addWidget(check)
            self._value_checks[value] = check

    def _build_text_editor(self, layout):
        self._editor_kind = "text"
        self._text_edit = QtWidgets.QLineEdit()
        self._text_edit.setPlaceholderText("substring filter…")
        self._text_edit.setToolTip(
            "Case-insensitive substring filter; start with '/' for a regex pattern"
        )
        self._text_edit.setClearButtonEnabled(True)
        layout.addWidget(self._text_edit)

        self._text_edit.textChanged.connect(self._on_text_changed)

    # ── editor callbacks ──────────────────────────────────────────────────────

    def _on_slider_changed(self, vmin, vcenter, vmax):
        for spin, value in ((self._min_spin, vmin), (self._max_spin, vmax)):
            spin.blockSignals(True)
            spin.setValue(value)
            spin.blockSignals(False)
        self.changed.emit()

    def _on_spin_changed(self):
        self._range_slider.set_values(
            self._min_spin.value(), self._range_slider.vcenter, self._max_spin.value()
        )
        self.changed.emit()

    def _on_text_changed(self):
        # Flag an invalid regex directly on the field
        pattern, is_regex = self._text_pattern()
        valid = True
        if is_regex:
            try:
                re.compile(pattern)
            except re.error:
                valid = False
        self._text_edit.setStyleSheet("" if valid else "color: red;")
        self._text_edit.setToolTip(
            "Case-insensitive substring filter; start with '/' for a regex pattern"
            if valid
            else "Invalid regex pattern"
        )
        self.changed.emit()

    def _text_pattern(self):
        """Return (pattern, is_regex); a leading '/' marks a regex pattern."""
        text = self._text_edit.text()
        if text.startswith("/"):
            return text[1:], True
        return text, False

    def _text_mask(self, series):
        pattern, is_regex = self._text_pattern()
        return (
            series.astype(str)
            .str.contains(pattern, case=False, regex=is_regex, na=False)
            .to_numpy()
        )

    # ── mask ──────────────────────────────────────────────────────────────────

    def mask(self, df):
        """Boolean mask (length ``len(df)``) of rows passing this filter."""
        col = self.column_combo.currentText()
        if col not in df.columns:
            return np.ones(len(df), dtype=bool)

        if self._editor_kind == "numeric":
            vals = df[col].to_numpy(dtype=float)
            # NaN fails both comparisons and hence drops out
            return (vals >= self._min_spin.value()) & (vals <= self._max_spin.value())
        elif self._editor_kind == "checks":
            checked = {v for v, c in self._value_checks.items() if c.isChecked()}
            return (
                df[col].notna().to_numpy()
                & df[col].astype(str).isin(checked).to_numpy()
            )
        elif self._editor_kind == "text":
            if not self._text_edit.text():
                return np.ones(len(df), dtype=bool)
            try:
                return self._text_mask(df[col])
            except re.error:
                return np.ones(len(df), dtype=bool)
        return np.ones(len(df), dtype=bool)


class ScatterControls(QtWidgets.QWidget):
    """Controls for the scatter plot."""

    def __init__(self, figure):
        super().__init__()
        self.figure = figure
        # Back-reference so the figure can notify the controls (e.g. on
        # embedding switches and label toggles).
        self.figure.controls = self
        self.setWindowTitle("Controls")
        self.label_overrides = {}

        # Build gui
        self.tab_layout = QtWidgets.QVBoxLayout()
        self.tab_layout.setContentsMargins(0, 0, 0, 0)
        self.tab_layout.setSpacing(0)
        self.setLayout(self.tab_layout)

        self.tabs = QtWidgets.QTabWidget()
        self.tabs.setDocumentMode(True)
        self.tabs.setTabPosition(QtWidgets.QTabWidget.West)
        self.tabs.setMovable(True)

        self.tab_layout.addWidget(self.tabs)

        self.tab1 = QtWidgets.QWidget()
        self.tab3 = QtWidgets.QWidget()
        self.tab4 = QtWidgets.QWidget()
        self.tab5 = QtWidgets.QWidget()
        self.tab6 = QtWidgets.QWidget()
        self.tab7 = QtWidgets.QWidget()
        self.tab1_layout = QtWidgets.QVBoxLayout()
        self.tab4_layout = QtWidgets.QVBoxLayout()
        self.tab5_layout = QtWidgets.QVBoxLayout()
        self.tab6_layout = QtWidgets.QVBoxLayout()
        self.tab7_layout = QtWidgets.QVBoxLayout()
        for _tl in (
            self.tab1_layout, self.tab4_layout, self.tab5_layout,
            self.tab6_layout, self.tab7_layout,
        ):
            _tl.setContentsMargins(4, 4, 4, 4)
            _tl.setSpacing(4)
        self.tab1.setLayout(self.tab1_layout)
        self.tab4.setLayout(self.tab4_layout)
        self.tab5.setLayout(self.tab5_layout)
        self.tab6.setLayout(self.tab6_layout)
        self.tab7.setLayout(self.tab7_layout)
        self.tabs.addTab(self.tab1, "General")
        self.tabs.addTab(self.tab5, "Embeddings")
        self.tabs.addTab(self.tab6, "Fidelity")
        self.tabs.addTab(self.tab7, "Cluster")
        self.tabs.addTab(self.tab4, "Settings")

        self.build_control_gui()
        self.build_settings_gui()
        self.build_embeddings_gui()
        self.build_fidelity_gui()
        self.build_clusters_gui()
        # Holds the futures for requested data
        self.futures = {}
        self.pool = ThreadPoolExecutor(4)

    @property
    def meta_data(self):
        """Get the meta data."""
        return self.figure.metadata

    @property
    def labels(self):
        """Get labels for each observation."""
        return self.figure.labels

    @property
    def labels_unique(self):
        """Get unique labels."""
        return np.unique(self.figure.labels)

    @property
    def selected_indices(self):
        """Get the selected indices."""
        selected = getattr(self.figure, "selected", None)
        if selected is None:
            return np.array([], dtype=int)
        return np.asarray(selected, dtype=int)

    def build_control_gui(self):
        """Build the GUI."""
        ########
        # Search
        ########
        search_group = QtWidgets.QGroupBox("Search")
        search_form = QtWidgets.QFormLayout()
        search_form.setContentsMargins(2, 2, 2, 2)
        search_form.setVerticalSpacing(2)
        search_form.setFieldGrowthPolicy(
            QtWidgets.QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow
        )
        search_group.setLayout(search_form)
        self.tab1_layout.addWidget(search_group)

        self.searchbar = QtWidgets.QLineEdit()
        self.searchbar.setToolTip(
            "Search for label(s) in the project. Search syntax:\n"
            " - by default will search for exact match among the labels\n"
            " - use a leading '/' to search for a regex\n"
            " - you can also search for IDs; multiple IDs must be comma-separated, e.g. '1,2,3'"
        )
        self.searchbar.setPlaceholderText("Hover for search syntax")
        self.searchbar.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Fixed,
        )
        self.searchbar.returnPressed.connect(self.find_next)
        # self.searchbar.textChanged.connect(self.figure.highlight_cluster)
        self.searchbar_completer = QtWidgets.QCompleter(self.labels_unique)
        self.searchbar_completer.setCaseSensitivity(QtCore.Qt.CaseInsensitive)
        self.searchbar.setCompleter(self.searchbar_completer)
        search_form.addRow(self.searchbar)

        # Add buttons for previous/next
        self.button_layout = QtWidgets.QHBoxLayout()
        self.button_layout.setContentsMargins(0, 0, 0, 0)
        self.button_layout.setSpacing(0)
        self.prev_button = QtWidgets.QPushButton("Previous")
        self.prev_button.clicked.connect(self.find_previous)
        self.button_layout.addWidget(self.prev_button)
        self.find_sel_button = QtWidgets.QPushButton("Select")
        self.find_sel_button.setToolTip(
            "Select all objects matching the search term. Use Shift-Click to add to current selection."
        )
        self.find_sel_button.clicked.connect(self.find_select)
        self.button_layout.addWidget(self.find_sel_button)
        self.next_button = QtWidgets.QPushButton("Next")
        self.next_button.clicked.connect(self.find_next)
        self.button_layout.addWidget(self.next_button)
        search_form.addRow(self.button_layout)

        ########
        # Labels
        ########

        label_group = QtWidgets.QGroupBox("Labels")
        label_form = QtWidgets.QFormLayout()
        label_form.setContentsMargins(2, 2, 2, 2)
        label_form.setVerticalSpacing(2)
        label_group.setLayout(label_form)
        self.tab1_layout.addWidget(label_group)

        # Add dropdown to choose leaf labels
        self.label_combo_box = _FilterableComboBox()
        label_form.addRow(QtWidgets.QLabel("Labels:"), self.label_combo_box)
        self.label_combo_box.currentIndexChanged.connect(self.set_labels)
        self._current_leaf_labels = self.label_combo_box.currentText()

        # Checkbox for whether to show label counts
        self.label_count_check = QtWidgets.QCheckBox("Show label counts")
        self.label_count_check.setToolTip("Whether to add counts to the labels.")
        self.label_count_check.setChecked(False)
        self.label_count_check.stateChanged.connect(self.set_label_counts)
        label_form.addRow(self.label_count_check)

        # Checkbox for whether to show label outlines
        self.label_outlines_check = QtWidgets.QCheckBox("Show label outlines")
        self.label_outlines_check.setToolTip(
            "Whether to add draw polygons around neurons with the same label."
        )
        self.label_outlines_check.setChecked(False)
        self.label_outlines_check.stateChanged.connect(self.set_label_outlines)
        label_form.addRow(self.label_outlines_check)

        ########
        # Colors
        ########

        color_group = QtWidgets.QGroupBox("Colors")
        color_form = QtWidgets.QFormLayout()
        color_form.setContentsMargins(2, 2, 2, 2)
        color_form.setVerticalSpacing(2)
        color_group.setLayout(color_form)
        self.tab1_layout.addWidget(color_group)

        # Add dropdowns to choose color mode

        self.color_combo_box = _FilterableComboBox()
        color_form.addRow("Color by:", self.color_combo_box)
        self.palette_combo_box = QtWidgets.QComboBox()
        color_form.addRow("Palette:", self.palette_combo_box)
        self.palette_combo_box.setToolTip(
            "The color palette to use when coloring by labels. Ignored if column contains colors or if coloring by clusters."
        )
        _palettes = [
            # Categorical / qualitative
            "seaborn:tab10",
            "seaborn:tab20",
            "vispy:husl",
            "colorbrewer:Set1",
            "colorbrewer:Set2",
            "colorbrewer:Paired",
            "glasbey:glasbey",
            # Sequential
            "matplotlib:viridis",
            "matplotlib:plasma",
            "matplotlib:inferno",
            "matplotlib:magma",
            "google:turbo",
            # Diverging
            "matplotlib:coolwarm",
            "colorbrewer:RdBu_r",
            "matplotlib:seismic",
        ]
        for _p in _palettes:
            self.palette_combo_box.addItem(
                QtGui.QIcon(_make_palette_pixmap(_p)), _p
            )
        self.palette_combo_box.setIconSize(QtCore.QSize(100, 14))

        # Range slider — visible only when coloring by a numerical column.
        # Shows min/max handles; a third centre handle appears for diverging palettes.
        self.color_range_slider = _MultiHandleSlider()
        self.color_range_slider.setVisible(False)
        self.color_range_slider.setToolTip(
            "Drag the handles to set the colour-scale range.\n"
            "For diverging palettes a centre handle is also shown."
        )
        color_form.addRow(self.color_range_slider)

        self.color_range_slider.valuesChanged.connect(lambda *_: self.set_colors())

        # Set the action for the color combo box
        self.color_combo_box.currentIndexChanged.connect(self._on_color_column_changed)
        self.palette_combo_box.currentIndexChanged.connect(self._on_palette_changed)
        self.label_combo_box.currentIndexChanged.connect(
            self._maybe_recompute_evaluation
        )

        ########
        # Point sizes
        ########

        size_group = QtWidgets.QGroupBox("Point Size")
        size_form = QtWidgets.QFormLayout()
        size_form.setContentsMargins(2, 2, 2, 2)
        size_form.setVerticalSpacing(2)
        size_group.setLayout(size_form)
        self.tab1_layout.addWidget(size_group)

        # Add dropdown to choose which column to scale point sizes by
        self.size_combo_box = _FilterableComboBox()
        size_form.addRow("Size by:", self.size_combo_box)
        self.size_combo_box.setToolTip(
            "Scale point sizes by a numeric metadata column. Values are min-max "
            "normalized; combine with the point scale setting in the Settings tab."
        )

        # Range slider — visible only when sizing by a column.
        # Min/max handles set the values mapped to the smallest/largest point
        # size; the centre handle sets the value mapped to the middle size.
        self.size_range_slider = _MultiHandleSlider()
        self.size_range_slider.setVisible(False)
        self.size_range_slider.set_diverging(True)  # always show the centre handle
        self.size_range_slider.setToolTip(
            "Drag the handles to set the size-scale range.\n"
            "Values at/below the min handle get the smallest size, values "
            "at/above the max handle the largest; the centre handle sets "
            "the value mapped to the middle size."
        )
        size_form.addRow(self.size_range_slider)

        self.size_range_slider.valuesChanged.connect(lambda *_: self.set_sizes())
        self.size_combo_box.currentIndexChanged.connect(self._on_size_column_changed)

        ########
        # Scope
        ########

        scope_group = QtWidgets.QGroupBox("Scope")
        scope_group.setToolTip(
            "Restrict which points can be selected. Does not change the plot."
        )
        scope_layout = QtWidgets.QVBoxLayout()
        scope_layout.setContentsMargins(2, 2, 2, 2)
        scope_layout.setSpacing(2)
        scope_group.setLayout(scope_layout)
        self.tab1_layout.addWidget(scope_group)

        self.scope_rows = []
        self.scope_rows_layout = QtWidgets.QVBoxLayout()
        self.scope_rows_layout.setContentsMargins(0, 0, 0, 0)
        self.scope_rows_layout.setSpacing(2)
        scope_layout.addLayout(self.scope_rows_layout)

        self.scope_add_btn = QtWidgets.QPushButton("+ Add filter")
        self.scope_add_btn.clicked.connect(self._add_scope_row)
        scope_layout.addWidget(self.scope_add_btn)

        self.scope_match_label = QtWidgets.QLabel("")
        self.scope_match_label.setStyleSheet("color: white;")
        self.scope_match_label.setVisible(False)
        scope_layout.addWidget(self.scope_match_label)

        ########
        # Selection behavior
        ########

        selection_group = QtWidgets.QGroupBox("Selection Behavior")
        selection_form = QtWidgets.QFormLayout()
        selection_form.setContentsMargins(2, 2, 2, 2)
        selection_form.setVerticalSpacing(2)
        selection_group.setLayout(selection_form)
        self.tab1_layout.addWidget(selection_group)

        self.add_group_check = QtWidgets.QCheckBox("Add as group")
        self.add_group_check.setToolTip("Whether to add neurons as group to the viewer when selected")
        self.add_group_check.stateChanged.connect(self.set_add_group)
        self.add_group_check.setChecked(True)
        selection_form.addRow(self.add_group_check)

        self.dclick_deselect = QtWidgets.QCheckBox("Deselect on double-click")
        self.dclick_deselect.setToolTip("You can always deselect using ESC")
        self.dclick_deselect.setChecked(self.figure.deselect_on_dclick)
        self.dclick_deselect.stateChanged.connect(self.set_dclick_deselect)
        selection_form.addRow(self.dclick_deselect)

        self.empty_deselect = QtWidgets.QCheckBox("Deselect on empty selection")
        self.empty_deselect.setToolTip("You can always deselect using ESC")
        self.empty_deselect.setChecked(self.figure.deselect_on_empty)
        self.empty_deselect.stateChanged.connect(self.set_empty_deselect)
        selection_form.addRow(self.empty_deselect)

        # This would make it so the legend does not stretch when
        # we resize the window vertically
        self.tab1_layout.addStretch(1)

        return

    def build_settings_gui(self):
        # Add dropdown to determine render mode
        self.tab4_layout.addWidget(QtWidgets.QLabel("Render trigger:"))

        self.render_mode_dropdown = QtWidgets.QComboBox()
        self.render_mode_dropdown.setToolTip(
            "Set trigger for re-rendering the scene. See documentation for details."
        )
        self.render_mode_dropdown.addItems(["Continuous", "Reactive", "Active Window"])
        self.render_mode_dropdown.setItemData(
            0, "Continuously render the scene.", QtCore.Qt.ToolTipRole
        )
        self.render_mode_dropdown.setItemData(
            1,
            "Render only when the scene changes.",
            QtCore.Qt.ToolTipRole,
        )
        self.render_mode_dropdown.setItemData(
            2, "Render only when the window is active.", QtCore.Qt.ToolTipRole
        )
        render_trigger_vals = ["continuous", "reactive", "active_window"]
        # Set default item to whatever the currently set render trigger is
        self.render_mode_dropdown.setCurrentIndex(
            render_trigger_vals.index(self.figure.render_trigger)
        )
        self.render_mode_dropdown.currentIndexChanged.connect(
            lambda x: setattr(
                self.figure,
                "render_trigger",
                render_trigger_vals[self.render_mode_dropdown.currentIndex()],
            )
        )
        self.tab4_layout.addWidget(self.render_mode_dropdown)

        # Add slide for max frame rate
        label = QtWidgets.QLabel("Max frame rate:")
        label.setToolTip(
            "Set the maximum frame rate for the figure. Press F while the figure window is active to show current frame rate."
        )
        self.tab4_layout.addWidget(label)

        self.max_frame_rate_layout = QtWidgets.QHBoxLayout()
        self.tab4_layout.addLayout(self.max_frame_rate_layout)

        self.max_frame_rate_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.max_frame_rate_slider.setRange(5, 100)
        self.max_frame_rate_slider.setValue(self.figure.max_fps)
        self.max_frame_rate_slider.valueChanged.connect(
            lambda x: setattr(self.figure, "max_fps", int(x))
        )
        self.max_frame_rate_slider.valueChanged.connect(
            lambda x: self.max_frame_rate_value_label.setText(f"{x} FPS")
        )
        self.max_frame_rate_layout.addWidget(self.max_frame_rate_slider)

        self.max_frame_rate_value_label = QtWidgets.QLabel(
            f"{int(self.figure.max_fps)} FPS"
        )
        self.max_frame_rate_layout.addWidget(self.max_frame_rate_value_label)

        # Add SpinBox for font size
        hlayout = QtWidgets.QHBoxLayout()
        self.tab4_layout.addLayout(hlayout)
        label = QtWidgets.QLabel("Font size:")
        label.setToolTip("Set the font size for the labels in the figure.")
        hlayout.addWidget(label)
        self.font_size_slider = QtWidgets.QDoubleSpinBox()
        self.font_size_slider.setRange(.01, 200)
        self.font_size_slider.setValue(self.figure.font_size)
        self.font_size_slider.valueChanged.connect(
            lambda x: setattr(self.figure, "font_size", x)
        )
        hlayout.addWidget(self.font_size_slider)

        # Add SpinBox for point scale
        hlayout = QtWidgets.QHBoxLayout()
        self.tab4_layout.addLayout(hlayout)
        label = QtWidgets.QLabel("Point scale:")
        label.setToolTip("Set the scaling factor for the point sizes in the figure.")
        hlayout.addWidget(label)
        self.point_scale_spinbox = QtWidgets.QDoubleSpinBox()
        self.point_scale_spinbox.setRange(.01, 200)
        self.point_scale_spinbox.setValue(float(self.figure.point_scale))
        self.point_scale_spinbox.valueChanged.connect(
            lambda x: setattr(self.figure, "point_scale", x)
        )
        hlayout.addWidget(self.point_scale_spinbox)

        # Add slider for number of labels visible at once
        label = QtWidgets.QLabel("Max visible labels:")
        label.setToolTip(
            "Set the maximum number of labels visible at once. This is useful for large datasets. May negatively impact performance."
        )
        self.tab4_layout.addWidget(label)

        self.max_label_vis_layout = QtWidgets.QHBoxLayout()
        self.tab4_layout.addLayout(self.max_label_vis_layout)

        self.max_label_vis_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.max_label_vis_slider.setRange(1, 5_000)
        self.max_label_vis_slider.setValue(self.figure.label_vis_limit)
        self.max_label_vis_slider.valueChanged.connect(
            lambda x: setattr(self.figure, "label_vis_limit", int(x))
        )
        self.max_label_vis_slider.valueChanged.connect(
            lambda x: self.max_label_vis_value_label.setText(f"{x} labels")
        )
        self.max_label_vis_layout.addWidget(self.max_label_vis_slider)

        self.max_label_vis_value_label = QtWidgets.QLabel(
            f"{int(self.figure.label_vis_limit)} labels"
        )
        self.max_label_vis_layout.addWidget(self.max_label_vis_value_label)

        ########
        # Neuroglancer viewer
        ########

        ngl_group = QtWidgets.QGroupBox("Neuroglancer viewer")
        ngl_form = QtWidgets.QFormLayout()
        ngl_form.setContentsMargins(2, 2, 2, 2)
        ngl_form.setVerticalSpacing(2)
        ngl_group.setLayout(ngl_form)
        self.tab4_layout.addWidget(ngl_group)

        has_viewer = hasattr(self.figure, "ngl_viewer")

        self.ngl_cache_neurons = QtWidgets.QCheckBox("Cache neurons")
        self.ngl_cache_neurons.setToolTip(
            "Whether to keep already loaded neurons in memory so they re-display instantly."
        )
        self.ngl_cache_neurons.setChecked(
            self.figure.ngl_viewer.use_cache if has_viewer else True
        )
        self.ngl_cache_neurons.stateChanged.connect(self.set_ngl_cache)
        ngl_form.addRow(self.ngl_cache_neurons)

        cache_size_label = QtWidgets.QLabel("Max cache size:")
        cache_size_label.setToolTip(
            "Maximum number of neurons to keep in the cache; least-recently-used neurons are evicted first."
        )
        self.ngl_cache_size = QtWidgets.QSpinBox()
        self.ngl_cache_size.setRange(1, 10_000)
        self.ngl_cache_size.setValue(
            self.figure.ngl_viewer.max_cache_size if has_viewer else 100
        )
        self.ngl_cache_size.valueChanged.connect(self.set_ngl_cache_size)
        ngl_form.addRow(cache_size_label, self.ngl_cache_size)

        self.ngl_clear_cache_button = QtWidgets.QPushButton("Clear cache")
        self.ngl_clear_cache_button.setToolTip("Remove all neurons from the cache.")
        self.ngl_clear_cache_button.clicked.connect(self.clear_ngl_cache)
        ngl_form.addRow(self.ngl_clear_cache_button)

        # This makes it so the legend does not stretch
        self.tab4_layout.addStretch(1)

    def build_embeddings_gui(self):
        """Build the GUI for the Embeddings tab."""
        # Selector to switch between multiple embeddings (hidden when there is
        # only one). It lives outside the recompute container so it stays usable
        # even when the active embedding has no high-dim source to recompute from.
        self.embedding_selector_group = QtWidgets.QGroupBox("Active embedding")
        selector_row = QtWidgets.QHBoxLayout()
        selector_row.setContentsMargins(6, 4, 6, 4)
        self.embedding_selector_group.setLayout(selector_row)
        self.embedding_selector_combo = QtWidgets.QComboBox()
        self.embedding_selector_combo.setToolTip(
            "Switch the active embedding (or press the space bar to cycle)."
        )
        # `activated` only fires on user interaction, so programmatic
        # setCurrentIndex calls don't feed back into a switch.
        self.embedding_selector_combo.activated.connect(
            lambda idx: self.figure.switch_embedding(idx, animate=True)
        )
        selector_row.addWidget(self.embedding_selector_combo)
        self.tab5_layout.addWidget(self.embedding_selector_group)
        self.embedding_selector_group.setVisible(False)

        # Everything below recomputes the active embedding; it is disabled when
        # the active embedding has no paired features/distances.
        self.embedding_recompute_widget = QtWidgets.QWidget()
        recompute_layout = QtWidgets.QVBoxLayout()
        recompute_layout.setContentsMargins(0, 0, 0, 0)
        recompute_layout.setSpacing(4)
        self.embedding_recompute_widget.setLayout(recompute_layout)
        self.tab5_layout.addWidget(self.embedding_recompute_widget)

        # Top action row
        actions_row = QtWidgets.QHBoxLayout()
        recompute_layout.addLayout(actions_row)

        self.umap_button = QtWidgets.QPushButton("Re-calculate positions")
        self.umap_button.setToolTip(
            "Run dimensionality reduction on the current dataset. This will overwrite the current positions."
        )
        self.umap_button.clicked.connect(self.calculate_embeddings)
        actions_row.addWidget(self.umap_button)

        self.umap_auto_run = QtWidgets.QCheckBox("Auto run")
        self.umap_auto_run.setToolTip(
            "Automatically run dimensionality reduction when changing settings."
        )
        self.umap_auto_run.setChecked(False)
        self.umap_auto_run.stateChanged.connect(
            lambda: setattr(self.figure, "_auto_umap", self.umap_auto_run.isChecked())
        )
        actions_row.addWidget(self.umap_auto_run)

        # Input group
        input_group = QtWidgets.QGroupBox("Input")
        input_form = QtWidgets.QFormLayout()
        input_form.setContentsMargins(6, 4, 6, 4)
        input_form.setVerticalSpacing(6)
        input_group.setLayout(input_form)
        recompute_layout.addWidget(input_group)

        self.umap_method_combo_box = QtWidgets.QComboBox()
        self.umap_method_combo_box.setToolTip(
            "Select the method to use for dimensionality reduction."
        )
        self._populate_embedding_methods()
        self.umap_method_combo_box.currentIndexChanged.connect(
            self.update_embedding_settings
        )
        input_form.addRow("Method:", self.umap_method_combo_box)

        self.umap_dist_combo_box = QtWidgets.QComboBox()
        self.umap_dist_combo_box.setToolTip(
            "Select the data source used for embedding."
        )

        def update_and_calculate_embeddings_maybe():
            """Update the run button when the distance is changed."""
            self._update_embedding_input_controls()
            self.calculate_embeddings_maybe()

        self.umap_dist_combo_box.currentIndexChanged.connect(
            update_and_calculate_embeddings_maybe
        )
        input_form.addRow("Data:", self.umap_dist_combo_box)

        # Info about the selected source (shape + type/metric from the project info).
        self.umap_source_info_label = QtWidgets.QLabel("")
        self.umap_source_info_label.setWordWrap(True)
        self.umap_source_info_label.setStyleSheet("color: #8a8a8a; font-size: 11px;")
        # Bottom margin keeps it from crowding the Feature Subset group below.
        self.umap_source_info_label.setContentsMargins(0, 0, 0, 8)
        input_form.addRow(self.umap_source_info_label)

        # Optional sub-selection of top-level feature groups for MultiIndex columns.
        self.umap_feature_subset_group = QtWidgets.QGroupBox("Feature Subset")
        feature_subset_layout = QtWidgets.QVBoxLayout()
        feature_subset_layout.setContentsMargins(6, 4, 6, 4)
        feature_subset_layout.setSpacing(2)
        self.umap_feature_subset_group.setLayout(feature_subset_layout)
        input_form.addRow(self.umap_feature_subset_group)

        self.umap_feature_partition_widget = QtWidgets.QWidget()
        partition_layout = QtWidgets.QVBoxLayout()
        partition_layout.setContentsMargins(0, 0, 0, 0)
        partition_layout.setSpacing(2)
        self.umap_feature_partition_widget.setLayout(partition_layout)
        feature_subset_layout.addWidget(self.umap_feature_partition_widget)
        self._embedding_partition_checks = {}

        # Feature-dependent options are grouped together and enabled only for feature vectors.
        self.feature_options_group = QtWidgets.QGroupBox("Feature Options")
        prep_form = QtWidgets.QFormLayout()
        prep_form.setContentsMargins(6, 4, 6, 4)
        prep_form.setVerticalSpacing(2)
        self.feature_options_group.setLayout(prep_form)
        recompute_layout.addWidget(self.feature_options_group)

        # For non-square inputs (feature vectors), allow choosing a distance metric.
        self.umap_feature_metric_widget = QtWidgets.QWidget()
        metric_layout = QtWidgets.QHBoxLayout()
        metric_layout.setContentsMargins(0, 0, 0, 0)
        self.umap_feature_metric_widget.setLayout(metric_layout)
        self.umap_feature_metric_combo_box = QtWidgets.QComboBox()
        self.umap_feature_metric_combo_box.setToolTip(
            "Distance metric used when embedding feature vectors."
        )
        self.umap_feature_metric_combo_box.addItems(
            ["cosine", "euclidean", "manhattan", "correlation", "chebyshev"]
        )
        self.umap_feature_metric_combo_box.currentIndexChanged.connect(
            self.calculate_embeddings_maybe
        )
        metric_layout.addWidget(self.umap_feature_metric_combo_box)
        prep_form.addRow("Feature metric:", self.umap_feature_metric_widget)

        # Optional feature rebalancing to reduce dominance of a few dimensions.
        self.umap_feature_rebalance_widget = QtWidgets.QWidget()
        rebalance_layout = QtWidgets.QHBoxLayout()
        rebalance_layout.setContentsMargins(0, 0, 0, 0)
        self.umap_feature_rebalance_widget.setLayout(rebalance_layout)
        self.umap_feature_rebalance_combo_box = QtWidgets.QComboBox()
        self.umap_feature_rebalance_combo_box.setToolTip(
            "Preprocessing applied per feature before embedding to reduce feature dominance."
        )
        self.umap_feature_rebalance_combo_box.addItems(
            ["none", "z-score", "robust (median/IQR)", "log1p + z-score"]
        )
        self.umap_feature_rebalance_combo_box.currentIndexChanged.connect(
            self.calculate_embeddings_maybe
        )
        rebalance_layout.addWidget(self.umap_feature_rebalance_combo_box)
        prep_form.addRow("Feature rebalancing:", self.umap_feature_rebalance_widget)

        pca_widget = QtWidgets.QWidget()
        pca_row = QtWidgets.QHBoxLayout()
        pca_row.setContentsMargins(0, 0, 0, 0)
        pca_widget.setLayout(pca_row)

        self.pca_check = QtWidgets.QCheckBox("Enable PCA")
        self.pca_check.setToolTip(
            "Reduce dimensionality before embedding (feature vectors only)."
        )
        self.pca_check.setChecked(False)
        pca_row.addWidget(self.pca_check)

        self.pca_n_components_slider = QtWidgets.QSpinBox()
        self.pca_n_components_slider.setRange(1, 2000)
        self.pca_n_components_slider.setSingleStep(1)
        self.pca_n_components_slider.setValue(100)
        self.pca_n_components_slider.setToolTip(
            "Number of components to keep after PCA."
        )
        self.pca_n_components_slider.valueChanged.connect(
            self.calculate_embeddings_maybe
        )
        self.pca_n_components_slider.setEnabled(self.pca_check.isChecked())
        pca_row.addWidget(self.pca_n_components_slider)

        self.pca_check.stateChanged.connect(
            lambda _: self.pca_n_components_slider.setEnabled(
                self.pca_check.isChecked()
            )
        )
        prep_form.addRow("PCA:", pca_widget)

        # Method-specific settings
        settings_group = QtWidgets.QGroupBox("Method Settings")
        settings_layout = QtWidgets.QVBoxLayout()
        settings_layout.setContentsMargins(6, 4, 6, 4)
        settings_layout.setSpacing(2)
        settings_group.setLayout(settings_layout)
        recompute_layout.addWidget(settings_group)

        method_common_widget = QtWidgets.QWidget()
        method_common_form = QtWidgets.QFormLayout()
        method_common_form.setContentsMargins(0, 0, 0, 0)
        method_common_widget.setLayout(method_common_form)
        settings_layout.addWidget(method_common_widget)

        self.umap_random_seed = QtWidgets.QLineEdit()
        self.umap_random_seed.setToolTip(
            "Random seed. Leave empty for random initialization."
        )
        self.umap_random_seed.setPlaceholderText("random initialization")
        self.umap_random_seed.setText(str(42))
        self.umap_random_seed.textChanged.connect(
            lambda x: self.calculate_embeddings_maybe()
        )
        method_common_form.addRow("Random seed:", self.umap_random_seed)

        self.umap_settings_widget = QtWidgets.QWidget()
        self.umap_settings_layout = QtWidgets.QFormLayout()
        self.umap_settings_layout.setContentsMargins(0, 0, 0, 0)
        self.umap_settings_widget.setLayout(self.umap_settings_layout)
        settings_layout.addWidget(self.umap_settings_widget)

        self.umap_n_neighbors_slider = QtWidgets.QSpinBox()
        self.umap_n_neighbors_slider.setRange(1, 200)
        self.umap_n_neighbors_slider.setSingleStep(1)
        self.umap_n_neighbors_slider.setValue(10)
        self.umap_n_neighbors_slider.valueChanged.connect(
            self.calculate_embeddings_maybe
        )
        self.umap_settings_layout.addRow(
            "Number of neighbors:", self.umap_n_neighbors_slider
        )

        self.umap_min_dist_slider = QtWidgets.QDoubleSpinBox()
        self.umap_min_dist_slider.setRange(0.0, 100.0)
        self.umap_min_dist_slider.setSingleStep(0.05)
        self.umap_min_dist_slider.setValue(0.1)
        self.umap_min_dist_slider.valueChanged.connect(self.calculate_embeddings_maybe)
        self.umap_settings_layout.addRow("Minimum distance:", self.umap_min_dist_slider)

        self.umap_spread_slider = QtWidgets.QDoubleSpinBox()
        self.umap_spread_slider.setRange(0.0, 1000.0)
        self.umap_spread_slider.setSingleStep(0.05)
        self.umap_spread_slider.setValue(1)
        self.umap_spread_slider.valueChanged.connect(self.calculate_embeddings_maybe)
        self.umap_settings_layout.addRow("Spread:", self.umap_spread_slider)

        self.umap_set_op_mix_ratio_spinbox = QtWidgets.QDoubleSpinBox()
        self.umap_set_op_mix_ratio_spinbox.setRange(0.0, 1.0)
        self.umap_set_op_mix_ratio_spinbox.setSingleStep(0.01)
        self.umap_set_op_mix_ratio_spinbox.setDecimals(2)
        self.umap_set_op_mix_ratio_spinbox.setValue(1.0)
        self.umap_set_op_mix_ratio_spinbox.setToolTip(
            "Adjust the UMAP set_op_mix_ratio between 0 and 1. Lower values encourage the embedding to preserve outliers as outlying."
        )
        self.umap_set_op_mix_ratio_spinbox.valueChanged.connect(
            self.calculate_embeddings_maybe
        )
        self.umap_settings_layout.addRow(
            "Set operation mix ratio:", self.umap_set_op_mix_ratio_spinbox
        )

        self.umap_densmap_check = QtWidgets.QCheckBox("Use DensMAP")
        self.umap_densmap_check.setToolTip(
            "Use UMAP's densMAP extension to better preserve local density relationships in the embedding."
        )
        self.umap_densmap_check.setChecked(False)
        self.umap_densmap_check.stateChanged.connect(
            lambda _: (self._update_densmap_controls(), self.calculate_embeddings_maybe())
        )
        self.umap_settings_layout.addRow("DensMAP:", self.umap_densmap_check)

        self.umap_dens_lambda_label = QtWidgets.QLabel("DensMAP lambda:")
        self.umap_dens_lambda_spinbox = QtWidgets.QDoubleSpinBox()
        self.umap_dens_lambda_spinbox.setRange(0.0, 10.0)
        self.umap_dens_lambda_spinbox.setSingleStep(0.1)
        self.umap_dens_lambda_spinbox.setDecimals(3)
        self.umap_dens_lambda_spinbox.setValue(2.0)
        self.umap_dens_lambda_spinbox.setToolTip(
            "Strength of the DensMAP density-preservation penalty. Higher values preserve density more strongly."
        )
        self.umap_dens_lambda_spinbox.valueChanged.connect(self.calculate_embeddings_maybe)
        self.umap_settings_layout.addRow(
            self.umap_dens_lambda_label,
            self.umap_dens_lambda_spinbox,
        )
        self._update_densmap_controls()

        self.tsne_settings_widget = QtWidgets.QWidget()
        self.tsne_settings_layout = QtWidgets.QFormLayout()
        self.tsne_settings_layout.setContentsMargins(0, 0, 0, 0)
        self.tsne_settings_widget.setLayout(self.tsne_settings_layout)
        settings_layout.addWidget(self.tsne_settings_widget)

        self.tsne_perplexity_slider = QtWidgets.QDoubleSpinBox()
        self.tsne_perplexity_slider.setRange(1.0, 200.0)
        self.tsne_perplexity_slider.setSingleStep(1.0)
        self.tsne_perplexity_slider.setDecimals(1)
        self.tsne_perplexity_slider.setValue(30.0)
        self.tsne_perplexity_slider.valueChanged.connect(self.calculate_embeddings_maybe)
        self.tsne_settings_layout.addRow("Perplexity:", self.tsne_perplexity_slider)

        self.tsne_learning_rate_slider = QtWidgets.QDoubleSpinBox()
        self.tsne_learning_rate_slider.setRange(10.0, 1000.0)
        self.tsne_learning_rate_slider.setSingleStep(10.0)
        self.tsne_learning_rate_slider.setDecimals(1)
        self.tsne_learning_rate_slider.setValue(200.0)
        self.tsne_learning_rate_slider.valueChanged.connect(self.calculate_embeddings_maybe)
        self.tsne_settings_layout.addRow(
            "Learning rate:", self.tsne_learning_rate_slider
        )

        self.tsne_n_iter_slider = QtWidgets.QSpinBox()
        self.tsne_n_iter_slider.setRange(250, 10000)
        self.tsne_n_iter_slider.setSingleStep(50)
        self.tsne_n_iter_slider.setValue(1000)
        self.tsne_n_iter_slider.valueChanged.connect(self.calculate_embeddings_maybe)
        self.tsne_settings_layout.addRow("Iterations:", self.tsne_n_iter_slider)

        self.mds_settings_widget = QtWidgets.QWidget()
        self.mds_settings_layout = QtWidgets.QFormLayout()
        self.mds_settings_layout.setContentsMargins(0, 0, 0, 0)
        self.mds_settings_widget.setLayout(self.mds_settings_layout)
        settings_layout.addWidget(self.mds_settings_widget)

        self.mds_n_init_slider = QtWidgets.QSpinBox()
        self.mds_n_init_slider.setRange(1, 200)
        self.mds_n_init_slider.setSingleStep(1)
        self.mds_n_init_slider.setValue(4)
        self.mds_n_init_slider.valueChanged.connect(self.calculate_embeddings_maybe)
        self.mds_settings_layout.addRow(
            "Number of initializations:", self.mds_n_init_slider
        )

        self.mds_max_iter_slider = QtWidgets.QSpinBox()
        self.mds_max_iter_slider.setRange(1, 10000)
        self.mds_max_iter_slider.setSingleStep(1)
        self.mds_max_iter_slider.setValue(300)
        self.mds_max_iter_slider.valueChanged.connect(self.calculate_embeddings_maybe)
        self.mds_settings_layout.addRow("Max iterations:", self.mds_max_iter_slider)

        self.mds_eps_slider = QtWidgets.QDoubleSpinBox()
        self.mds_eps_slider.setRange(0.0000, 1.0000)
        self.mds_eps_slider.setSingleStep(0.001)
        self.mds_eps_slider.setDecimals(4)
        self.mds_eps_slider.setValue(0.001)
        self.mds_eps_slider.valueChanged.connect(self.calculate_embeddings_maybe)
        self.mds_settings_layout.addRow("Relative tolerance:", self.mds_eps_slider)

        # Populate options
        self.update_umap_options()

        self.tab5_layout.addStretch(1)

        # Make sure method settings are in sync by default.
        self.update_embedding_settings()

    def build_fidelity_gui(self):
        """Build the GUI for the Fidelity tab."""

        # Active-embedding indicator. Unlike the Embeddings tab (which offers a
        # selector), fidelity always applies to whichever embedding is active,
        # so this is a read-only label mirroring that selector's position.
        self.fidelity_active_embedding_group = QtWidgets.QGroupBox("Active embedding")
        active_row = QtWidgets.QHBoxLayout()
        active_row.setContentsMargins(6, 4, 6, 4)
        self.fidelity_active_embedding_group.setLayout(active_row)
        self.fidelity_active_embedding_label = QtWidgets.QLabel("—")
        self.fidelity_active_embedding_label.setToolTip(
            "The embedding that fidelity is evaluated against. Switch it on the "
            "Embeddings tab (or press the space bar to cycle)."
        )
        active_row.addWidget(self.fidelity_active_embedding_label)
        self.tab6_layout.addWidget(self.fidelity_active_embedding_group)

        # Shared input: which high-dimensional source defines the "true"
        # neighborhood that fidelity scores and KNN edges compare the 2D layout
        # against. Mirrors the Embeddings tab's "Input" group so the choice is
        # made once instead of per-panel.
        input_group = QtWidgets.QGroupBox("Input")
        input_form = QtWidgets.QFormLayout()
        input_form.setContentsMargins(6, 4, 6, 4)
        input_form.setVerticalSpacing(6)
        input_group.setLayout(input_form)
        self.fidelity_input_form = input_form
        self.tab6_layout.addWidget(input_group)

        self.add_tooltip(
            input_group,
            "Select the high-dimensional source used as ground truth for "
            "neighborhood fidelity and KNN edges. 'knn' uses the precomputed "
            "KNN graph directly; 'precomputed' uses the full distance matrix; "
            "'features' computes neighbors from feature vectors with the chosen "
            "metric.",
            anchor="group_label",
        )

        self.fidelity_data_combo_box = QtWidgets.QComboBox()
        self.fidelity_data_combo_box.setToolTip(
            "High-dimensional source used as ground truth for fidelity and KNN edges."
        )
        self.fidelity_data_combo_box.currentIndexChanged.connect(
            self._on_fidelity_source_changed
        )
        input_form.addRow("Data:", self.fidelity_data_combo_box)

        # The metric only applies to the feature source; the row is hidden for
        # 'knn'/'precomputed' (which carry their own distances).
        self.fidelity_metric_combo_box = QtWidgets.QComboBox()
        self.fidelity_metric_combo_box.setToolTip(
            "Distance metric used when neighbors are computed from feature vectors."
        )
        self.fidelity_metric_combo_box.addItems(
            ["euclidean", "cosine", "manhattan", "correlation", "chebyshev"]
        )
        self.fidelity_metric_combo_box.currentIndexChanged.connect(self.set_knn_edges)
        input_form.addRow("Metric:", self.fidelity_metric_combo_box)

        # Shape / KNN dimensions (+ metric) for the selected source.
        self.fidelity_source_info_label = QtWidgets.QLabel("")
        self.fidelity_source_info_label.setWordWrap(True)
        self.fidelity_source_info_label.setStyleSheet(
            "color: #8a8a8a; font-size: 11px;"
        )
        self.fidelity_source_info_label.setContentsMargins(0, 0, 0, 8)
        input_form.addRow(self.fidelity_source_info_label)

        # Neighboorhood fidelity settings
        settings_group = QtWidgets.QGroupBox("Neighborhood Fidelity")
        settings_form = QtWidgets.QFormLayout()
        settings_form.setContentsMargins(6, 4, 6, 4)
        settings_form.setVerticalSpacing(2)
        settings_group.setLayout(settings_form)
        self.tab6_layout.addWidget(settings_group)

        self.add_tooltip(
            settings_group,
            "Neighborhood fidelity measures how well local relationships are "
            "preserved by the 2D embedding. Higher values mean nearby neurons "
            "in the original space remain nearby in the plot. Computed scores "
            f"are stored as metadata column '{FIDELITY_DATA_COLUMN}'.",
            anchor="group_label",
        )

        self.fidelity_k_slider = QtWidgets.QSpinBox()
        self.fidelity_k_slider.setRange(1, 200)
        self.fidelity_k_slider.setSingleStep(1)
        self.fidelity_k_slider.setValue(10)
        self.fidelity_k_slider.setToolTip(
            "Number of nearest neighbors (k) used to compute neighborhood fidelity."
        )
        settings_form.addRow("k neighbors:", self.fidelity_k_slider)

        self.fidelity_use_rank_check = QtWidgets.QCheckBox("Use rank")
        self.fidelity_use_rank_check.setToolTip(
            "Use rank-based neighborhood overlap instead of distance-weighted overlap."
        )
        self.fidelity_use_rank_check.setChecked(True)
        settings_form.addRow(self.fidelity_use_rank_check)

        self.fidelity_compute_button = QtWidgets.QPushButton("Compute")
        self.fidelity_compute_button.setToolTip(
            "Compute per-point neighborhood fidelity with the current settings "
            f"and store it as metadata column '{FIDELITY_DATA_COLUMN}'. Point "
            "sizes will be scaled by the result (see 'Size by' on the General tab)."
        )
        self.fidelity_compute_button.clicked.connect(self._on_fidelity_compute)
        settings_form.addRow(self.fidelity_compute_button)

        # KNN lines
        settings_group = QtWidgets.QGroupBox("K-Nearest Neighbors")
        settings_form = QtWidgets.QFormLayout()
        settings_form.setContentsMargins(6, 4, 6, 4)
        settings_form.setVerticalSpacing(2)
        settings_group.setLayout(settings_form)
        self.tab6_layout.addWidget(settings_group)

        self.add_tooltip(
            settings_group,
            "Show lines connecting each point to its k nearest neighbors in the "
            "high-dimensional source selected above (Input → Data). This can "
            "help visualize how well local relationships are preserved in the "
            "embedding.",
            anchor="group_label",
        )

        self.fidelity_knn_combo_box = QtWidgets.QComboBox()
        self.fidelity_knn_combo_box.setToolTip(
            "Show lines connecting each point to its k nearest neighbors in the "
            "selected Data source."
        )
        self.fidelity_knn_combo_box.addItems(["Off", "Selected only", "All points"])
        self.fidelity_knn_combo_box.currentIndexChanged.connect(self.set_knn_edges)
        settings_form.addRow("Show:", self.fidelity_knn_combo_box)

        self.fidelity_knn_k_slider = QtWidgets.QSpinBox()
        self.fidelity_knn_k_slider.setRange(1, 200)
        self.fidelity_knn_k_slider.setSingleStep(1)
        self.fidelity_knn_k_slider.setValue(10)
        self.fidelity_knn_k_slider.setToolTip(
            "Number of nearest neighbors (k) to draw edges for."
        )
        self.fidelity_knn_k_slider.valueChanged.connect(self.set_knn_edges)
        settings_form.addRow("k neighbors:", self.fidelity_knn_k_slider)

        # Evaluate labels
        settings_group = QtWidgets.QGroupBox("Evaluate Labels")
        settings_form = QtWidgets.QFormLayout()
        settings_form.setContentsMargins(6, 4, 6, 4)
        settings_form.setVerticalSpacing(2)
        settings_group.setLayout(settings_form)
        self.evaluate_labels_form = settings_form
        self.tab6_layout.addWidget(settings_group)

        self.add_tooltip(
            settings_group,
            "Evaluate how well groups defined by the currently shown labels (e.g. from clustering) represent the structure of high-dimensional space.",
            anchor="group_label",
        )

        self.evaluate_labels_show_check = QtWidgets.QCheckBox()
        self.evaluate_labels_show_check.setToolTip(
            "Enable per-sample evaluation of label assignments based on given metric."
        )
        self.evaluate_labels_show_check.setChecked(False)
        self.evaluate_labels_show_check.stateChanged.connect(
            self._on_evaluate_labels_toggle
        )
        settings_form.addRow("Show:", self.evaluate_labels_show_check)

        self.evaluate_labels_data_combo_box = QtWidgets.QComboBox()
        self.evaluate_labels_data_combo_box.setToolTip(
            "Select the data source used for label evaluation."
        )
        self.evaluate_labels_data_combo_box.currentIndexChanged.connect(
            self._on_evaluate_labels_toggle
        )
        settings_form.addRow("Data:", self.evaluate_labels_data_combo_box)

        self.evaluate_labels_method_combo_box = QtWidgets.QComboBox()
        self.evaluate_labels_method_combo_box.setToolTip(
            "Select the evaluation method for per-sample label scores."
        )
        self.evaluate_labels_method_combo_box.addItems(
            ["Silhouette", "Neighbor consistency"]
        )
        self.evaluate_labels_method_combo_box.setItemData(
            0,
            "Use silhouette samples to score how well each point fits its assigned cluster.",
            QtCore.Qt.ToolTipRole,
        )
        self.evaluate_labels_method_combo_box.setItemData(
            1,
            "Score each point by how often its nearest neighbors share the same label.",
            QtCore.Qt.ToolTipRole,
        )
        self.evaluate_labels_method_combo_box.currentIndexChanged.connect(
            self._update_evaluate_labels_controls
        )
        self.evaluate_labels_method_combo_box.currentIndexChanged.connect(
            self._on_evaluate_labels_toggle
        )
        settings_form.addRow("Method:", self.evaluate_labels_method_combo_box)

        self.evaluate_labels_metric_combo_box = QtWidgets.QComboBox()
        self.evaluate_labels_metric_combo_box.setToolTip(
            "Distance metric used when computing per-sample label evaluation."
        )
        self.evaluate_labels_metric_combo_box.addItems(
            ["euclidean", "cosine", "manhattan", "correlation", "chebyshev"]
        )
        self.evaluate_labels_metric_combo_box.setCurrentText("cosine")
        self.evaluate_labels_metric_combo_box.currentIndexChanged.connect(
            self._on_evaluate_labels_toggle
        )
        settings_form.addRow("Metric:", self.evaluate_labels_metric_combo_box)

        self.evaluate_labels_sync_check = QtWidgets.QCheckBox("Sync w/ labels")
        self.evaluate_labels_sync_check.setToolTip(
            "Automatically recompute per-sample evaluation when the displayed labels change."
        )
        self.evaluate_labels_sync_check.setChecked(False)
        self.evaluate_labels_sync_check.stateChanged.connect(
            self._maybe_recompute_evaluation
        )
        settings_form.addRow(self.evaluate_labels_sync_check)

        self.evaluate_labels_k_spinbox = QtWidgets.QSpinBox()
        self.evaluate_labels_k_spinbox.setRange(1, 200)
        self.evaluate_labels_k_spinbox.setValue(10)
        self.evaluate_labels_k_spinbox.setToolTip(
            "Number of neighbors used for neighbor-consistency evaluation."
        )
        self.evaluate_labels_k_spinbox.valueChanged.connect(
            self._on_evaluate_labels_toggle
        )
        self.evaluate_labels_k_row = QtWidgets.QWidget()
        k_row_layout = QtWidgets.QHBoxLayout()
        k_row_layout.setContentsMargins(0, 0, 0, 0)
        k_row_layout.addWidget(self.evaluate_labels_k_spinbox)
        self.evaluate_labels_k_row.setLayout(k_row_layout)
        self.evaluate_labels_k_label = QtWidgets.QLabel("k neighbors:")
        settings_form.addRow(self.evaluate_labels_k_label, self.evaluate_labels_k_row)
        self.evaluate_labels_k_row.setVisible(False)
        self.evaluate_labels_k_label.setVisible(False)

        # Distances edges
        settings_group = QtWidgets.QGroupBox("Distances")
        settings_form = QtWidgets.QFormLayout()
        settings_form.setContentsMargins(6, 4, 6, 4)
        settings_form.setVerticalSpacing(2)
        settings_group.setLayout(settings_form)
        # Hidden entirely unless the project ships a full distance matrix
        # (see `update_distance_edges_controls`).
        self.distance_edges_group = settings_group
        self.tab6_layout.addWidget(settings_group)

        self.add_tooltip(
            settings_group,
            "Show lines connecting points closer than a given distance in the original feature space. "
            "This can help visualize how well local relationships are preserved in the embedding.",
            anchor="group_label",
        )

        # Add a checkbox + spinbox to show distances as edges
        self.show_distance_edges_check = QtWidgets.QCheckBox()
        self.show_distance_edges_check.setToolTip(
            "Whether to show actual distances as edges between points."
        )
        self.show_distance_edges_check.setChecked(False)
        settings_form.addRow("Show:", self.show_distance_edges_check)

        self.distance_edges_slider = QtWidgets.QDoubleSpinBox()
        self.distance_edges_slider.setRange(0.0, 1.0)
        self.distance_edges_slider.setSingleStep(0.05)
        self.distance_edges_slider.setValue(self.figure.distance_edges_threshold)
        self.distance_edges_slider.setDecimals(2)
        settings_form.addRow("Threshold:", self.distance_edges_slider)

        self.distance_edges_slider.valueChanged.connect(
            lambda x: setattr(self.figure, "distance_edges_threshold", float(x))
        )
        self.show_distance_edges_check.stateChanged.connect(
            lambda x: setattr(self.figure, "show_distance_edges", bool(x))
        )
        self.show_distance_edges_check.stateChanged.connect(
            lambda x: self.distance_edges_slider.setEnabled(
                self.show_distance_edges_check.isChecked()
            )
        )

        self.tab6_layout.addStretch(1)

        self.update_fidelity_options()

    def build_clusters_gui(self):
        """Build the GUI for the Clusters tab."""
        # Top action row
        actions_row = QtWidgets.QHBoxLayout()
        actions_row.setContentsMargins(0, 0, 0, 0)
        actions_row.setSpacing(0)
        self.tab7_layout.addLayout(actions_row)

        self.cluster_run_button = QtWidgets.QPushButton("Run")
        self.cluster_run_button.setToolTip(
            "Partition the data into clusters using the selected algorithm."
        )
        self.cluster_run_button.clicked.connect(self._run_clustering)
        actions_row.addWidget(self.cluster_run_button)

        self.cluster_clear_button = QtWidgets.QPushButton("Clear")
        self.cluster_clear_button.setToolTip("Reset points to their defaults.")
        self.cluster_clear_button.clicked.connect(self._clear_cluster)
        actions_row.addWidget(self.cluster_clear_button)

        self.cluster_load_button = QtWidgets.QPushButton()
        self.cluster_load_button.setIcon(
            self.style().standardIcon(QtWidgets.QStyle.SP_DialogOpenButton)
        )
        self.cluster_load_button.setAccessibleName("Load")
        self.cluster_load_button.setToolTip(
            "Load cluster assignments from a previously exported CSV file."
        )
        self.cluster_load_button.clicked.connect(self._load_cluster_labels)
        actions_row.addWidget(self.cluster_load_button)

        # Input group
        input_group = QtWidgets.QGroupBox("Input")
        input_form = QtWidgets.QFormLayout()
        input_form.setContentsMargins(6, 4, 6, 4)
        input_form.setVerticalSpacing(2)
        input_group.setLayout(input_form)
        self.tab7_layout.addWidget(input_group)

        self.cluster_data_combo_box = QtWidgets.QComboBox()
        self.cluster_data_combo_box.setToolTip("Select the data source for clustering.")
        self.cluster_data_combo_box.currentIndexChanged.connect(
            self._refresh_cluster_source_gating
        )
        input_form.addRow("Data:", self.cluster_data_combo_box)

        self.cluster_metric_combo_box = QtWidgets.QComboBox()
        self.cluster_metric_combo_box.setToolTip(
            "Distance metric used when clustering from feature vectors."
        )
        self.cluster_metric_combo_box.addItems(
            ["euclidean", "cosine", "manhattan", "correlation", "chebyshev"]
        )
        input_form.addRow("Metric:", self.cluster_metric_combo_box)

        # Small note shown only when a KNN graph is the selected input.
        self.cluster_knn_input_note = QtWidgets.QLabel(
            "Options limited when using KNN input"
        )
        self.cluster_knn_input_note.setWordWrap(True)
        self.cluster_knn_input_note.setStyleSheet("color: #a00; font-size: 11px;")
        self.cluster_knn_input_note.setVisible(False)
        input_form.addRow(self.cluster_knn_input_note)

        # Algorithm group
        algo_group = QtWidgets.QGroupBox("Algorithm")
        algo_layout = QtWidgets.QVBoxLayout()
        algo_layout.setContentsMargins(6, 4, 6, 4)
        algo_layout.setSpacing(2)
        algo_group.setLayout(algo_layout)
        self.tab7_layout.addWidget(algo_group)

        algo_form = QtWidgets.QFormLayout()
        algo_form.setContentsMargins(0, 0, 0, 0)
        algo_layout.addLayout(algo_form)

        self.cluster_method_combo_box = QtWidgets.QComboBox()
        self._populate_cluster_methods()
        self.cluster_method_combo_box.setToolTip("Select the clustering algorithm.")
        self.cluster_method_combo_box.currentIndexChanged.connect(
            self._update_cluster_method_settings
        )
        algo_form.addRow("Method:", self.cluster_method_combo_box)

        # --- HDBSCAN settings ---
        self.cluster_hdbscan_widget = QtWidgets.QWidget()
        hdbscan_form = QtWidgets.QFormLayout()
        hdbscan_form.setContentsMargins(0, 0, 0, 0)
        self.cluster_hdbscan_widget.setLayout(hdbscan_form)
        algo_layout.addWidget(self.cluster_hdbscan_widget)

        self.cluster_hdbscan_min_cluster_size = QtWidgets.QSpinBox()
        self.cluster_hdbscan_min_cluster_size.setRange(2, 10000)
        self.cluster_hdbscan_min_cluster_size.setValue(5)
        self.cluster_hdbscan_min_cluster_size.setToolTip(
            "Minimum number of points required to form a cluster."
        )
        hdbscan_form.addRow("Min cluster size:", self.cluster_hdbscan_min_cluster_size)

        self.cluster_hdbscan_min_samples = QtWidgets.QSpinBox()
        self.cluster_hdbscan_min_samples.setRange(0, 10000)
        self.cluster_hdbscan_min_samples.setValue(0)
        self.cluster_hdbscan_min_samples.setToolTip(
            "Minimum samples for a core point. 0 defaults to min_cluster_size."
        )
        hdbscan_form.addRow("Min samples:", self.cluster_hdbscan_min_samples)

        self.cluster_hdbscan_epsilon = QtWidgets.QDoubleSpinBox()
        self.cluster_hdbscan_epsilon.setRange(0.0, 100.0)
        self.cluster_hdbscan_epsilon.setSingleStep(0.01)
        self.cluster_hdbscan_epsilon.setDecimals(3)
        self.cluster_hdbscan_epsilon.setValue(0.0)
        self.cluster_hdbscan_epsilon.setToolTip(
            "Distance threshold below which clusters are merged."
        )
        hdbscan_form.addRow("Cluster epsilon:", self.cluster_hdbscan_epsilon)

        self.cluster_hdbscan_method_combo = QtWidgets.QComboBox()
        self.cluster_hdbscan_method_combo.addItems(["eom", "leaf"])
        self.cluster_hdbscan_method_combo.setToolTip(
            "'eom': excess of mass (larger clusters). "
            "'leaf': smaller, more granular clusters."
        )
        hdbscan_form.addRow("Selection method:", self.cluster_hdbscan_method_combo)

        self.cluster_hdbscan_label_noise_check = QtWidgets.QCheckBox(
            "Force noise point assignment"
        )
        self.cluster_hdbscan_label_noise_check.setToolTip(
            "Assign noise points (-1) to the nearest non-noise cluster."
        )
        self.cluster_hdbscan_label_noise_check.setChecked(False)
        self.cluster_hdbscan_label_noise_check.stateChanged.connect(
            self._update_hdbscan_noise_controls
        )
        hdbscan_form.addRow(self.cluster_hdbscan_label_noise_check)

        self.cluster_hdbscan_noise_threshold_row = QtWidgets.QWidget()
        noise_threshold_layout = QtWidgets.QHBoxLayout()
        noise_threshold_layout.setContentsMargins(0, 0, 0, 0)
        self.cluster_hdbscan_noise_threshold_row.setLayout(noise_threshold_layout)

        self.cluster_hdbscan_noise_threshold_check = QtWidgets.QCheckBox(
            "Use threshold"
        )
        self.cluster_hdbscan_noise_threshold_check.setToolTip(
            "Only relabel noise points if nearest-cluster distance is below threshold."
        )
        self.cluster_hdbscan_noise_threshold_check.setChecked(False)
        self.cluster_hdbscan_noise_threshold_check.stateChanged.connect(
            self._update_hdbscan_noise_controls
        )
        noise_threshold_layout.addWidget(self.cluster_hdbscan_noise_threshold_check)

        self.cluster_hdbscan_noise_threshold = QtWidgets.QDoubleSpinBox()
        self.cluster_hdbscan_noise_threshold.setRange(0.0, 1_000_000.0)
        self.cluster_hdbscan_noise_threshold.setSingleStep(0.05)
        self.cluster_hdbscan_noise_threshold.setDecimals(4)
        self.cluster_hdbscan_noise_threshold.setValue(1.0)
        self.cluster_hdbscan_noise_threshold.setToolTip(
            "Maximum distance for relabeling a noise point."
        )
        noise_threshold_layout.addWidget(self.cluster_hdbscan_noise_threshold)

        hdbscan_form.addRow(
            "Noise threshold:", self.cluster_hdbscan_noise_threshold_row
        )

        # --- Agglomerative settings ---
        self.cluster_agg_widget = QtWidgets.QWidget()
        agg_form = QtWidgets.QFormLayout()
        agg_form.setContentsMargins(0, 0, 0, 0)
        self.cluster_agg_widget.setLayout(agg_form)
        algo_layout.addWidget(self.cluster_agg_widget)

        self.cluster_agg_criterion_combo = QtWidgets.QComboBox()
        self.cluster_agg_criterion_combo.addItems(
            ["N clusters", "Distance threshold", "Homogeneous composition"]
        )
        self.cluster_agg_criterion_combo.setCurrentIndex(0)
        self.cluster_agg_criterion_combo.setToolTip(
            "Choose how to stop agglomerative merging: fixed number of "
            "clusters or a linkage distance threshold."
        )
        self.cluster_agg_criterion_combo.currentIndexChanged.connect(
            self._update_agg_controls
        )
        agg_form.addRow("Stop criterion:", self.cluster_agg_criterion_combo)

        self.cluster_agg_n_clusters = QtWidgets.QSpinBox()
        self.cluster_agg_n_clusters.setRange(2, 10000)
        self.cluster_agg_n_clusters.setValue(10)
        self.cluster_agg_n_clusters.setToolTip("Number of clusters to find.")
        agg_form.addRow("N clusters (k):", self.cluster_agg_n_clusters)

        self.cluster_agg_distance_threshold = QtWidgets.QDoubleSpinBox()
        self.cluster_agg_distance_threshold.setRange(0.0, 1_000_000.0)
        self.cluster_agg_distance_threshold.setSingleStep(0.05)
        self.cluster_agg_distance_threshold.setDecimals(4)
        self.cluster_agg_distance_threshold.setValue(1.0)
        self.cluster_agg_distance_threshold.setToolTip(
            "Stop merging when linkage distance exceeds this threshold."
        )
        agg_form.addRow("Distance threshold:", self.cluster_agg_distance_threshold)

        self.cluster_agg_homogeneous_widget = QtWidgets.QWidget()
        homogeneous_form = QtWidgets.QFormLayout()
        homogeneous_form.setContentsMargins(0, 0, 0, 0)
        self.cluster_agg_homogeneous_widget.setLayout(homogeneous_form)
        agg_form.addRow(self.cluster_agg_homogeneous_widget)

        self.cluster_agg_homogeneous_labels_combo = QtWidgets.QComboBox()
        self.cluster_agg_homogeneous_labels_combo.setToolTip(
            "Label source used to evaluate composition balance per cluster."
        )
        homogeneous_form.addRow(
            "Composition labels:", self.cluster_agg_homogeneous_labels_combo
        )

        self.cluster_agg_homogeneous_max_dist = QtWidgets.QDoubleSpinBox()
        self.cluster_agg_homogeneous_max_dist.setRange(0.0, 1_000_000.0)
        self.cluster_agg_homogeneous_max_dist.setDecimals(4)
        self.cluster_agg_homogeneous_max_dist.setSingleStep(0.05)
        self.cluster_agg_homogeneous_max_dist.setValue(0.0)
        self.cluster_agg_homogeneous_max_dist.setSpecialValueText("None")
        self.cluster_agg_homogeneous_max_dist.setToolTip(
            "Upper split-distance bound. 0 means no upper bound."
        )
        homogeneous_form.addRow(
            "Max split distance:", self.cluster_agg_homogeneous_max_dist
        )

        self.cluster_agg_homogeneous_min_dist = QtWidgets.QDoubleSpinBox()
        self.cluster_agg_homogeneous_min_dist.setRange(0.0, 1_000_000.0)
        self.cluster_agg_homogeneous_min_dist.setDecimals(4)
        self.cluster_agg_homogeneous_min_dist.setSingleStep(0.05)
        self.cluster_agg_homogeneous_min_dist.setValue(0.0)
        self.cluster_agg_homogeneous_min_dist.setSpecialValueText("None")
        self.cluster_agg_homogeneous_min_dist.setToolTip(
            "Lower split-distance bound. 0 means no lower bound."
        )
        homogeneous_form.addRow(
            "Min split distance:", self.cluster_agg_homogeneous_min_dist
        )

        self.cluster_agg_homogeneous_min_dist_diff = QtWidgets.QDoubleSpinBox()
        self.cluster_agg_homogeneous_min_dist_diff.setRange(0.0, 1_000_000.0)
        self.cluster_agg_homogeneous_min_dist_diff.setDecimals(4)
        self.cluster_agg_homogeneous_min_dist_diff.setSingleStep(0.05)
        self.cluster_agg_homogeneous_min_dist_diff.setValue(0.0)
        self.cluster_agg_homogeneous_min_dist_diff.setSpecialValueText("None")
        self.cluster_agg_homogeneous_min_dist_diff.setToolTip(
            "Merge nearby clusters when split distances differ less than this value."
        )
        homogeneous_form.addRow(
            "Merge distance diff:", self.cluster_agg_homogeneous_min_dist_diff
        )

        self.cluster_agg_linkage_combo = QtWidgets.QComboBox()
        self.cluster_agg_linkage_combo.addItems(
            ["ward", "complete", "average", "single"]
        )
        self.cluster_agg_linkage_combo.setToolTip(
            "Ward minimises within-cluster variance. complete/average/single use "
            "max/mean/min inter-cluster distances. "
            "Ward is replaced with 'average' when using a precomputed distance matrix."
        )
        agg_form.addRow("Linkage:", self.cluster_agg_linkage_combo)

        # --- K-Means settings ---
        self.cluster_kmeans_widget = QtWidgets.QWidget()
        kmeans_form = QtWidgets.QFormLayout()
        kmeans_form.setContentsMargins(0, 0, 0, 0)
        self.cluster_kmeans_widget.setLayout(kmeans_form)
        algo_layout.addWidget(self.cluster_kmeans_widget)

        self.cluster_kmeans_n_init = QtWidgets.QSpinBox()
        self.cluster_kmeans_n_init.setRange(1, 100)
        self.cluster_kmeans_n_init.setValue(10)
        self.cluster_kmeans_n_init.setToolTip(
            "Number of random initialisations to try."
        )
        kmeans_form.addRow("N init:", self.cluster_kmeans_n_init)

        self.cluster_kmeans_max_iter = QtWidgets.QSpinBox()
        self.cluster_kmeans_max_iter.setRange(10, 10000)
        self.cluster_kmeans_max_iter.setValue(300)
        self.cluster_kmeans_max_iter.setToolTip("Maximum iterations per run.")
        kmeans_form.addRow("Max iterations:", self.cluster_kmeans_max_iter)

        option_row = QtWidgets.QWidget()
        option_layout = QtWidgets.QHBoxLayout()
        option_layout.setContentsMargins(0, 0, 0, 0)
        option_layout.setSpacing(4)
        option_row.setLayout(option_layout)

        self.cluster_kmeans_n_clusters = QtWidgets.QSpinBox()
        self.cluster_kmeans_n_clusters.setRange(2, 10000)
        self.cluster_kmeans_n_clusters.setValue(10)
        self.cluster_kmeans_n_clusters.setToolTip("Number of clusters.")
        option_layout.addWidget(self.cluster_kmeans_n_clusters)

        self.cluster_kmeans_choose_k_button = QtWidgets.QToolButton()
        self.cluster_kmeans_choose_k_button.setText("Find k")
        self.cluster_kmeans_choose_k_button.setToolTip(
            "Click to choose a score method and automatically select the number of K-Means clusters."
        )
        self.cluster_kmeans_choose_k_button.setPopupMode(
            QtWidgets.QToolButton.InstantPopup
        )

        self.cluster_kmeans_score_menu = QtWidgets.QMenu(self)
        self.cluster_kmeans_score_group = QtGui.QActionGroup(self)
        self.cluster_kmeans_score_group.setExclusive(True)

        for score_name in ["Silhouette", "Calinski-Harabasz", "Davies-Bouldin"]:
            action = self.cluster_kmeans_score_menu.addAction(score_name)
            action.setCheckable(True)
            action.triggered.connect(
                lambda checked, score=score_name: self._choose_k_for_kmeans(score)
            )
            self.cluster_kmeans_score_group.addAction(action)

        self.cluster_kmeans_score_group.actions()[0].setChecked(True)
        self.cluster_kmeans_choose_k_button.setMenu(self.cluster_kmeans_score_menu)
        option_layout.addWidget(self.cluster_kmeans_choose_k_button)

        kmeans_form.addRow("N clusters:", option_row)

        # --- Spectral settings ---
        self.cluster_spectral_widget = QtWidgets.QWidget()
        spectral_form = QtWidgets.QFormLayout()
        spectral_form.setContentsMargins(0, 0, 0, 0)
        self.cluster_spectral_widget.setLayout(spectral_form)
        algo_layout.addWidget(self.cluster_spectral_widget)

        self.cluster_spectral_n_clusters = QtWidgets.QSpinBox()
        self.cluster_spectral_n_clusters.setRange(2, 10000)
        self.cluster_spectral_n_clusters.setValue(10)
        self.cluster_spectral_n_clusters.setToolTip(
            "Number of clusters to find with Spectral clustering."
        )

        spectral_option_row = QtWidgets.QWidget()
        spectral_option_layout = QtWidgets.QHBoxLayout()
        spectral_option_layout.setContentsMargins(0, 0, 0, 0)
        spectral_option_layout.setSpacing(4)
        spectral_option_row.setLayout(spectral_option_layout)
        spectral_option_layout.addWidget(self.cluster_spectral_n_clusters)

        self.cluster_spectral_choose_k_button = QtWidgets.QToolButton()
        self.cluster_spectral_choose_k_button.setText("Find k")
        self.cluster_spectral_choose_k_button.setToolTip(
            "Click to choose a score method and automatically select the number of Spectral clusters."
        )
        self.cluster_spectral_choose_k_button.setPopupMode(
            QtWidgets.QToolButton.InstantPopup
        )

        self.cluster_spectral_score_menu = QtWidgets.QMenu(self)
        self.cluster_spectral_score_group = QtGui.QActionGroup(self)
        self.cluster_spectral_score_group.setExclusive(True)

        for score_name in ["Silhouette", "Calinski-Harabasz", "Davies-Bouldin"]:
            action = self.cluster_spectral_score_menu.addAction(score_name)
            action.setCheckable(True)
            action.triggered.connect(
                lambda checked, score=score_name: self._choose_k_for_spectral(score)
            )
            self.cluster_spectral_score_group.addAction(action)

        self.cluster_spectral_score_group.actions()[0].setChecked(True)
        self.cluster_spectral_choose_k_button.setMenu(self.cluster_spectral_score_menu)
        spectral_option_layout.addWidget(self.cluster_spectral_choose_k_button)

        spectral_form.addRow("N clusters:", spectral_option_row)

        self.cluster_spectral_affinity_combo = QtWidgets.QComboBox()
        self.cluster_spectral_affinity_combo.addItems(["nearest_neighbors", "rbf"])
        self.cluster_spectral_affinity_combo.setToolTip(
            "Spectral affinity to use. 'nearest_neighbors' uses a graph affinity, 'rbf' uses a kernel affinity."
        )
        spectral_form.addRow("Affinity:", self.cluster_spectral_affinity_combo)

        self.cluster_spectral_gamma = QtWidgets.QDoubleSpinBox()
        self.cluster_spectral_gamma.setRange(0.001, 1000.0)
        self.cluster_spectral_gamma.setSingleStep(0.1)
        self.cluster_spectral_gamma.setDecimals(3)
        self.cluster_spectral_gamma.setValue(1.0)
        self.cluster_spectral_gamma.setToolTip(
            "Gamma value for RBF affinity. Only used when affinity is 'rbf'."
        )
        spectral_form.addRow("Gamma:", self.cluster_spectral_gamma)

        self.cluster_spectral_n_neighbors = QtWidgets.QSpinBox()
        self.cluster_spectral_n_neighbors.setRange(1, 1000)
        self.cluster_spectral_n_neighbors.setValue(10)
        self.cluster_spectral_n_neighbors.setToolTip(
            "Number of neighbors to use for nearest_neighbors affinity."
        )
        spectral_form.addRow("N neighbors:", self.cluster_spectral_n_neighbors)

        self.cluster_spectral_n_init = QtWidgets.QSpinBox()
        self.cluster_spectral_n_init.setRange(1, 100)
        self.cluster_spectral_n_init.setValue(10)
        self.cluster_spectral_n_init.setToolTip(
            "Number of initializations for Spectral clustering."
        )
        spectral_form.addRow("N init:", self.cluster_spectral_n_init)

        # Output group
        output_group = QtWidgets.QGroupBox("Output")
        output_form = QtWidgets.QFormLayout()
        output_form.setContentsMargins(6, 4, 6, 4)
        output_form.setVerticalSpacing(2)
        output_group.setLayout(output_form)
        self.tab7_layout.addWidget(output_group)

        self.cluster_result_label = QtWidgets.QLabel("N/A")
        self.cluster_result_label.setToolTip("Result of the last clustering run.")
        output_form.addRow("Result:", self.cluster_result_label)

        cluster_output_buttons = QtWidgets.QWidget()
        cluster_output_buttons_layout = QtWidgets.QHBoxLayout()
        cluster_output_buttons_layout.setContentsMargins(0, 0, 0, 0)
        cluster_output_buttons.setLayout(cluster_output_buttons_layout)

        self.cluster_apply_button = QtWidgets.QPushButton("Apply labels")
        self.cluster_apply_button.setToolTip(
            "Write cluster assignments as a metadata column and switch the "
            "label display to show cluster IDs."
        )
        self.cluster_apply_button.setEnabled(False)
        self.cluster_apply_button.clicked.connect(
            lambda _: self.label_combo_box.setCurrentText(CLUSTER_DATA_OPTION)
        )
        cluster_output_buttons_layout.addWidget(self.cluster_apply_button)

        self.cluster_export_button = QtWidgets.QPushButton("Export")
        self.cluster_export_button.setToolTip(
            "Save the current cluster assignments as a CSV file "
            "with 'id' and 'bigclust_cluster' columns."
        )
        self.cluster_export_button.setEnabled(False)
        self.cluster_export_button.clicked.connect(self._export_cluster_labels)
        cluster_output_buttons_layout.addWidget(self.cluster_export_button)

        output_form.addRow(cluster_output_buttons)

        # Manual refinement group (hidden until cluster labels are available)
        self.cluster_manual_group = QtWidgets.QGroupBox("Manual Refinement")
        manual_form = QtWidgets.QFormLayout()
        manual_form.setContentsMargins(6, 4, 6, 4)
        manual_form.setVerticalSpacing(2)
        self.cluster_manual_group.setLayout(manual_form)
        self.tab7_layout.addWidget(self.cluster_manual_group)

        cluster_set_row = QtWidgets.QWidget()
        cluster_set_layout = QtWidgets.QHBoxLayout()
        cluster_set_layout.setContentsMargins(0, 0, 0, 0)
        cluster_set_row.setLayout(cluster_set_layout)

        self.cluster_manual_set_combo = QtWidgets.QComboBox()
        self.cluster_manual_set_combo.setToolTip(
            "Select target cluster ID for the current selection."
        )
        self.cluster_manual_set_combo.setEditable(True)
        self.cluster_manual_set_combo.setInsertPolicy(
            QtWidgets.QComboBox.InsertPolicy.NoInsert
        )
        combo_completer = self.cluster_manual_set_combo.completer()
        if combo_completer is not None:
            combo_completer.setCaseSensitivity(QtCore.Qt.CaseInsensitive)
        cluster_set_layout.addWidget(self.cluster_manual_set_combo)

        self.cluster_manual_set_button = QtWidgets.QPushButton("Set Cluster")
        self.cluster_manual_set_button.setToolTip(
            "Assign current selection to the selected cluster ID."
        )
        self.cluster_manual_set_button.clicked.connect(self._manual_refine_set_cluster)
        cluster_set_layout.addWidget(self.cluster_manual_set_button)

        manual_form.addRow("Target cluster:", cluster_set_row)

        self.cluster_manual_new_button = QtWidgets.QPushButton("New Cluster")
        self.cluster_manual_new_button.setToolTip(
            "Assign current selection to a newly created cluster ID."
        )
        self.cluster_manual_new_button.clicked.connect(self._manual_refine_new_cluster)
        manual_form.addRow(self.cluster_manual_new_button)

        self.cluster_manual_reset_button = QtWidgets.QPushButton("Reset")
        self.cluster_manual_reset_button.setToolTip(
            "Reset current selection to the original cluster assignment."
        )
        self.cluster_manual_reset_button.clicked.connect(self._manual_refine_reset)
        manual_form.addRow(self.cluster_manual_reset_button)

        self.tab7_layout.addStretch(1)

        # Sync initial visibility
        self._update_cluster_method_settings()
        self._update_hdbscan_noise_controls()
        self.update_cluster_options()
        self._refresh_manual_refinement_controls()

    # ------------------------------------------------------------------
    # Clusters tab helpers
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # KNN-graph mode helpers
    # ------------------------------------------------------------------

    def _active_knn(self):
        """Return the active KNN graph, or None."""
        dists = getattr(self.figure, "dists", None)
        if isinstance(dists, dict):
            return dists.get("knn")
        return None

    def _is_knn_only(self):
        """Whether the active source is a KNN graph with no full distances/features."""
        dists = getattr(self.figure, "dists", None)
        if not isinstance(dists, dict):
            return False
        return (
            dists.get("knn") is not None
            and "distances" not in dists
            and "features" not in dists
        )

    def _selected_embedding_knn(self):
        """Return the selected embedding source iff it is a KNN graph, else None."""
        arr = self._selected_embedding_data()
        return arr if is_knn_graph(arr) else None

    def _selected_clustering_knn(self):
        """Return the selected clustering source iff it is a KNN graph, else None."""
        arr = self._selected_clustering_data()
        return arr if is_knn_graph(arr) else None

    def update_cluster_options(self):
        """Enable/disable the Clusters tab and populate the data combo box."""
        clusters_tab_index = self.tabs.indexOf(self.tab7)
        dists = getattr(self.figure, "dists", None)
        positions = getattr(self.figure, "positions", None)
        has_embedding = positions is not None and len(np.asarray(positions)) > 0

        if dists is None and not has_embedding:
            self.tabs.setTabEnabled(clusters_tab_index, False)
            self._refresh_manual_refinement_controls()
            return

        self.tabs.setTabEnabled(clusters_tab_index, True)

        current_text = self.cluster_data_combo_box.currentText()
        self.cluster_data_combo_box.blockSignals(True)
        self.cluster_data_combo_box.clear()
        if has_embedding:
            self.cluster_data_combo_box.addItem(CLUSTER_DATA_EMBEDDING_OPTION)
        if isinstance(dists, dict):
            for key in dists.keys():
                self.cluster_data_combo_box.addItem(key)
        self.cluster_data_combo_box.blockSignals(False)

        if current_text and self.cluster_data_combo_box.findText(current_text) >= 0:
            self.cluster_data_combo_box.setCurrentText(current_text)

        self._update_homogeneous_label_options()
        # Limit methods/sliders/metric and surface the warning for the selected
        # source (also runs `_update_cluster_method_settings` -> input controls).
        self._refresh_cluster_source_gating()
        self._refresh_manual_refinement_controls()

    def _refresh_cluster_source_gating(self):
        """Limit clustering methods/sliders/metric for the selected source.

        A KNN-graph source allows only HDBSCAN/Spectral (capped to k); the 2D
        embedding or a distance/feature source keeps all methods.
        """
        graph = self._selected_clustering_knn()
        self._populate_cluster_methods(is_knn=graph is not None)
        if graph is not None:
            k = int(graph.k)
            self.cluster_spectral_n_neighbors.setMaximum(max(1, k))
            self.cluster_hdbscan_min_samples.setMaximum(max(1, k))
        else:
            self.cluster_spectral_n_neighbors.setMaximum(1000)
            self.cluster_hdbscan_min_samples.setMaximum(10000)
        if hasattr(self, "cluster_knn_input_note"):
            self.cluster_knn_input_note.setVisible(graph is not None)
        # Repopulating may have changed the current method (signals blocked),
        # so re-sync which method-specific settings are shown + input controls.
        self._update_cluster_method_settings()

    def _populate_cluster_methods(self, is_knn=False):
        """Populate the clustering method combo, limited to HDBSCAN/Spectral for KNN."""
        combo = self.cluster_method_combo_box
        methods = (
            ["HDBSCAN", "Spectral"]
            if is_knn
            else ["HDBSCAN", "Agglomerative", "K-Means", "Spectral"]
        )
        current = combo.currentText()
        combo.blockSignals(True)
        combo.clear()
        combo.addItems(methods)
        combo.blockSignals(False)
        if current in methods:
            combo.setCurrentText(current)
        elif methods:
            combo.setCurrentIndex(0)

    def _update_homogeneous_label_options(self):
        """Populate homogeneous-composition label source choices."""
        if not hasattr(self, "cluster_agg_homogeneous_labels_combo"):
            return

        current = self.cluster_agg_homogeneous_labels_combo.currentText()
        self.cluster_agg_homogeneous_labels_combo.blockSignals(True)
        self.cluster_agg_homogeneous_labels_combo.clear()
        self.cluster_agg_homogeneous_labels_combo.addItem(
            CLUSTER_HOMOGENEOUS_LABEL_CURRENT
        )

        meta = getattr(self.figure, "metadata", None)
        if meta is not None:
            for col in sorted(meta.columns):
                if col.startswith("_"):
                    continue
                self.cluster_agg_homogeneous_labels_combo.addItem(col)

        self.cluster_agg_homogeneous_labels_combo.blockSignals(False)

        if current and self.cluster_agg_homogeneous_labels_combo.findText(current) >= 0:
            self.cluster_agg_homogeneous_labels_combo.setCurrentText(current)
        elif self.cluster_agg_homogeneous_labels_combo.findText("dataset") >= 0:
            self.cluster_agg_homogeneous_labels_combo.setCurrentText("dataset")

    def _selected_homogeneous_labels(self):
        """Return labels used for homogeneous-composition clustering."""
        key = self.cluster_agg_homogeneous_labels_combo.currentText()
        if (not key) or (key == CLUSTER_HOMOGENEOUS_LABEL_CURRENT):
            return np.asarray(self.figure.labels)

        meta = getattr(self.figure, "metadata", None)
        if meta is None or key not in meta.columns:
            return None

        return meta[key].astype(str).fillna("").to_numpy()

    def _selected_clustering_data(self):
        """Return the currently selected distance/feature matrix for clustering."""
        key = self.cluster_data_combo_box.currentText()
        if key == CLUSTER_DATA_EMBEDDING_OPTION:
            return getattr(self.figure, "positions", None)

        dists = getattr(self.figure, "dists", None)
        if dists is None:
            return None
        if isinstance(dists, dict):
            if not key:
                return None
            return dists.get(key)
        return dists

    def _update_cluster_input_controls(self):
        """Adjust metric availability based on the selected input type."""
        arr = self._selected_clustering_data()
        is_knn = is_knn_graph(arr)
        is_precomputed = arr is not None and is_precomputed_distance_matrix(arr)
        # No metric choice for a precomputed matrix or a KNN graph.
        self.cluster_metric_combo_box.setEnabled(not is_precomputed and not is_knn)
        if hasattr(self, "cluster_kmeans_choose_k_button"):
            self.cluster_kmeans_choose_k_button.setEnabled(
                not is_precomputed and not is_knn
                and self.cluster_method_combo_box.currentText() == "K-Means"
            )
        if hasattr(self, "cluster_spectral_choose_k_button"):
            # "Find k" needs a full distance matrix (silhouette/precomputed),
            # which a KNN graph cannot provide.
            self.cluster_spectral_choose_k_button.setEnabled(
                not is_knn
                and self.cluster_method_combo_box.currentText() == "Spectral"
            )

    def _update_cluster_method_settings(self):
        """Show/hide method-specific settings widgets."""
        method = self.cluster_method_combo_box.currentText()
        self.cluster_hdbscan_widget.setVisible(method == "HDBSCAN")
        self.cluster_agg_widget.setVisible(method == "Agglomerative")
        self.cluster_kmeans_widget.setVisible(method == "K-Means")
        self.cluster_spectral_widget.setVisible(method == "Spectral")
        self._update_agg_controls()
        self._update_cluster_input_controls()

    def _update_agg_controls(self):
        """Enable Agglomerative fields for the selected stop criterion."""
        if not hasattr(self, "cluster_agg_criterion_combo"):
            return

        criterion = self.cluster_agg_criterion_combo.currentText()
        use_threshold = criterion == "Distance threshold"
        use_homogeneous = criterion == "Homogeneous composition"

        show_n_clusters = not use_threshold and not use_homogeneous
        show_distance_threshold = use_threshold

        self.cluster_agg_n_clusters.setVisible(show_n_clusters)
        self.cluster_agg_distance_threshold.setVisible(show_distance_threshold)

        agg_form = self.cluster_agg_widget.layout()
        if isinstance(agg_form, QtWidgets.QFormLayout):
            n_label = agg_form.labelForField(self.cluster_agg_n_clusters)
            if n_label is not None:
                n_label.setVisible(show_n_clusters)

            d_label = agg_form.labelForField(self.cluster_agg_distance_threshold)
            if d_label is not None:
                d_label.setVisible(show_distance_threshold)

        self.cluster_agg_homogeneous_widget.setVisible(use_homogeneous)

    def _update_hdbscan_noise_controls(self):
        """Enable optional HDBSCAN noise-threshold controls."""
        if not hasattr(self, "cluster_hdbscan_label_noise_check"):
            return

        relabel_noise = self.cluster_hdbscan_label_noise_check.isChecked()
        use_threshold = (
            relabel_noise and self.cluster_hdbscan_noise_threshold_check.isChecked()
        )

        self.cluster_hdbscan_noise_threshold_check.setEnabled(relabel_noise)
        self.cluster_hdbscan_noise_threshold_row.setEnabled(relabel_noise)
        self.cluster_hdbscan_noise_threshold.setEnabled(use_threshold)

    def _choose_k_for_kmeans(self, score_method=None):
        """Automatically choose k for K-Means using the current data and score method."""
        if self.cluster_method_combo_box.currentText() != "K-Means":
            self.figure.show_message(
                "Choose K is only available for K-Means.", color="orange", duration=3
            )
            return

        arr = self._selected_clustering_data()
        if arr is None:
            self.figure.show_message(
                "No data available for choosing k", color="red", duration=3
            )
            return

        arr_np = arr.values if hasattr(arr, "values") else np.asarray(arr)
        if is_precomputed_distance_matrix(arr_np):
            self.figure.show_message(
                "K-Means requires feature vectors, not a precomputed distance matrix.",
                color="red",
                duration=4,
            )
            return

        from ...clusters import find_k

        if score_method is None:
            checked_action = self.cluster_kmeans_score_group.checkedAction()
            score_method = (
                checked_action.text() if checked_action is not None else "Silhouette"
            )
        score_method = score_method.lower().replace("-", "_").replace(" ", "_")

        def progress_callback(k, max_k):
            self.figure.show_message(
                f"Searching k: {k}/{max_k}", color="lightblue", duration=None
            )
            QtWidgets.QApplication.processEvents()

        try:
            k = find_k(
                arr_np,
                method="K-Means",
                is_precomputed=False,
                metric=self.cluster_metric_combo_box.currentText(),
                k_min=2,
                k_max="auto",
                score_method=score_method,
                kmeans_n_init=self.cluster_kmeans_n_init.value(),
                kmeans_max_iter=self.cluster_kmeans_max_iter.value(),
                random_state=42,
                progress_callback=progress_callback,
            )
        except Exception as exc:
            self.figure.show_message(
                f"Could not choose k: {exc}", color="red", duration=5
            )
            return

        self.cluster_kmeans_n_clusters.setValue(int(k))
        self.figure.show_message(
            f"Selected k={k} for K-Means.", color="lightgreen", duration=4
        )

    def _choose_k_for_spectral(self, score_method=None):
        """Automatically choose k for Spectral clustering using the current data and score method."""
        if self.cluster_method_combo_box.currentText() != "Spectral":
            self.figure.show_message(
                "Choose K is only available for Spectral.", color="orange", duration=3
            )
            return

        arr = self._selected_clustering_data()
        if arr is None:
            self.figure.show_message(
                "No data available for choosing k", color="red", duration=3
            )
            return

        arr_np = arr.values if hasattr(arr, "values") else np.asarray(arr)
        from ...clusters import find_k

        if score_method is None:
            checked_action = self.cluster_spectral_score_group.checkedAction()
            score_method = (
                checked_action.text() if checked_action is not None else "Silhouette"
            )
        score_method = score_method.lower().replace("-", "_").replace(" ", "_")

        def progress_callback(k, max_k):
            self.figure.show_message(
                f"Searching k: {k}/{max_k}", color="lightblue", duration=None
            )
            QtWidgets.QApplication.processEvents()

        try:
            k = find_k(
                arr_np,
                method="Spectral",
                is_precomputed=is_precomputed_distance_matrix(arr_np),
                metric=self.cluster_metric_combo_box.currentText(),
                k_min=2,
                k_max="auto",
                score_method=score_method,
                spectral_affinity=self.cluster_spectral_affinity_combo.currentText(),
                spectral_gamma=self.cluster_spectral_gamma.value(),
                spectral_n_neighbors=self.cluster_spectral_n_neighbors.value(),
                spectral_n_init=self.cluster_spectral_n_init.value(),
                random_state=42,
                progress_callback=progress_callback,
            )
        except Exception as exc:
            self.figure.show_message(
                f"Could not choose k: {exc}", color="red", duration=5
            )
            return

        self.cluster_spectral_n_clusters.setValue(int(k))
        self.figure.show_message(
            f"Selected k={k} for Spectral.", color="lightgreen", duration=4
        )

    def _run_clustering(self):
        """Run clustering with the current settings and colour points by cluster."""
        from ...clusters import run_clustering

        arr = self._selected_clustering_data()
        if arr is None:
            self.figure.show_message(
                "No data available for clustering", color="red", duration=2
            )
            return

        # KNN graph: cluster the sparse graph directly (HDBSCAN/Spectral only).
        if is_knn_graph(arr):
            self._run_clustering_from_knn(arr)
            return

        arr_np = arr.values if hasattr(arr, "values") else np.asarray(arr)
        is_precomputed = is_precomputed_distance_matrix(arr_np)
        metric = self.cluster_metric_combo_box.currentText()
        method = self.cluster_method_combo_box.currentText()

        if (
            method == "Agglomerative"
            and self.cluster_agg_linkage_combo.currentText() == "ward"
            and (is_precomputed or metric != "euclidean")
        ):
            if self.figure is not None:
                self.figure.show_message(
                    "Ward linkage should not be used with non-euclidean distances.\n"
                    "Please use a different linkage method (e.g. 'average') or provide\n"
                    "feature vectors + Euclidean metric instead of a distance matrix.",
                    duration=5,
                    color="orange",
                )

        try:
            labels = run_clustering(
                arr_np,
                method=method,
                is_precomputed=is_precomputed,
                metric=metric,
                hdbscan_min_cluster_size=self.cluster_hdbscan_min_cluster_size.value(),
                hdbscan_min_samples=(self.cluster_hdbscan_min_samples.value() or None),
                hdbscan_cluster_selection_epsilon=self.cluster_hdbscan_epsilon.value(),
                hdbscan_cluster_selection_method=self.cluster_hdbscan_method_combo.currentText(),
                hbdscan_label_noise=self.cluster_hdbscan_label_noise_check.isChecked(),
                hbdscan_label_noise_threshold=(
                    self.cluster_hdbscan_noise_threshold.value()
                    if (
                        self.cluster_hdbscan_label_noise_check.isChecked()
                        and self.cluster_hdbscan_noise_threshold_check.isChecked()
                    )
                    else None
                ),
                agg_stop_criterion={
                    "N clusters": "n_clusters",
                    "Distance threshold": "distance_threshold",
                    "Homogeneous composition": "homogeneous_composition",
                }[self.cluster_agg_criterion_combo.currentText()],
                agg_n_clusters=(
                    self.cluster_agg_n_clusters.value()
                    if self.cluster_agg_criterion_combo.currentText() == "N clusters"
                    else None
                ),
                agg_distance_threshold=(
                    self.cluster_agg_distance_threshold.value()
                    if self.cluster_agg_criterion_combo.currentText()
                    == "Distance threshold"
                    else None
                ),
                agg_linkage=self.cluster_agg_linkage_combo.currentText(),
                agg_homogeneous_labels=(
                    self._selected_homogeneous_labels()
                    if self.cluster_agg_criterion_combo.currentText()
                    == "Homogeneous composition"
                    else None
                ),
                agg_homogeneous_max_dist=(
                    self.cluster_agg_homogeneous_max_dist.value() or None
                    if self.cluster_agg_criterion_combo.currentText()
                    == "Homogeneous composition"
                    else None
                ),
                agg_homogeneous_min_dist=(
                    self.cluster_agg_homogeneous_min_dist.value() or None
                    if self.cluster_agg_criterion_combo.currentText()
                    == "Homogeneous composition"
                    else None
                ),
                agg_homogeneous_min_dist_diff=(
                    self.cluster_agg_homogeneous_min_dist_diff.value() or None
                    if self.cluster_agg_criterion_combo.currentText()
                    == "Homogeneous composition"
                    else None
                ),
                kmeans_n_clusters=self.cluster_kmeans_n_clusters.value(),
                kmeans_n_init=self.cluster_kmeans_n_init.value(),
                kmeans_max_iter=self.cluster_kmeans_max_iter.value(),
                spectral_n_clusters=self.cluster_spectral_n_clusters.value(),
                spectral_affinity=self.cluster_spectral_affinity_combo.currentText(),
                spectral_gamma=self.cluster_spectral_gamma.value(),
                spectral_n_neighbors=self.cluster_spectral_n_neighbors.value(),
                spectral_n_init=self.cluster_spectral_n_init.value(),
            )
        except Exception as e:
            logger.error(f"Clustering failed: {e}")
            self.figure.show_message(f"Clustering error: {e}", color="red", duration=4)
            return

        self._cluster_labels = labels
        self._cluster_labels_original = labels.copy()

        # This function takes care of writing cluster labels to metadata, updating the label/color combo boxes, and refreshing the display
        self._apply_cluster_labels()

    def _run_clustering_from_knn(self, graph):
        """Cluster a precomputed KNN graph (HDBSCAN/Spectral only)."""
        from ...clusters import run_clustering

        method = self.cluster_method_combo_box.currentText()
        try:
            labels = run_clustering(
                None,
                method=method,
                is_precomputed=False,
                knn=graph,
                hdbscan_min_cluster_size=self.cluster_hdbscan_min_cluster_size.value(),
                hdbscan_min_samples=(self.cluster_hdbscan_min_samples.value() or None),
                hdbscan_cluster_selection_epsilon=self.cluster_hdbscan_epsilon.value(),
                hdbscan_cluster_selection_method=self.cluster_hdbscan_method_combo.currentText(),
                hbdscan_label_noise=self.cluster_hdbscan_label_noise_check.isChecked(),
                spectral_n_clusters=self.cluster_spectral_n_clusters.value(),
                spectral_n_neighbors=self.cluster_spectral_n_neighbors.value(),
                spectral_n_init=self.cluster_spectral_n_init.value(),
            )
        except Exception as e:
            logger.error(f"KNN clustering failed: {e}")
            self.figure.show_message(f"Clustering error: {e}", color="red", duration=4)
            return

        self._cluster_labels = labels
        self._cluster_labels_original = labels.copy()
        self._apply_cluster_labels()

    def _clear_cluster(self):
        """Reset viewer to the defaults stored in figure metadata."""
        self._cluster_labels = None
        self._cluster_labels_original = None
        self._cluster_colors = None
        self.cluster_result_label.setText("N/A")
        self.cluster_apply_button.setEnabled(False)
        self.cluster_export_button.setEnabled(False)
        self.color_combo_box.setCurrentText(self._color_col_before_clustering)
        if getattr(self, "_label_before_clustering", None):
            self.label_combo_box.setCurrentText(self._label_before_clustering)
        self._refresh_manual_refinement_controls()
        self.update_label_combo_boxes()

    def _apply_cluster_labels(self):
        """Write cluster assignments to metadata and switch the label display."""
        if not hasattr(self, "_cluster_labels") or self._cluster_labels is None:
            return

        # (Re-)calculate colors
        self._cluster_colors = labels_to_colors(
            self._cluster_labels, palette="vispy:husl"
        )
        self._cluster_colors[self._cluster_labels < 0] = (
            0.5,
            0.5,
            0.5,
            1.0,
        )  # grey for noise/unassigned

        # Write new cluster labels to metadata
        if self.figure.metadata is not None:
            self.figure.metadata[CLUSTER_DATA_COLUMN] = self._cluster_labels.copy()

        # Save the current label and color mode so we can restore it when clearing
        self._label_before_clustering = self.label_combo_box.currentText()
        self._color_col_before_clustering = self.color_combo_box.currentText()

        # Update the cluster summary label
        self._update_cluster_result_label(self._cluster_labels)

        # Make sure the cluster buttons are enabled and the manual refinement controls are visible
        self.cluster_apply_button.setEnabled(True)
        self.cluster_export_button.setEnabled(True)
        self._refresh_manual_refinement_controls()

        # Apply cluster colors
        self.label_combo_box.blockSignals(True)
        self.update_label_combo_boxes()  # this update both label and color combo boxes
        self.label_combo_box.blockSignals(False)

        if self.color_combo_box.currentText() != CLUSTER_DATA_OPTION:
            self.color_combo_box.setCurrentText(
                CLUSTER_DATA_OPTION
            )  # this triggers a refresh
        else:
            self.set_colors()  # just refresh colors without changing the combo box text

        # For the labels: we don't automatically enforce setting them but if they are set to the cluster column,
        # we need to trigger an update
        if self.label_combo_box.currentText() == CLUSTER_DATA_OPTION:
            self.set_labels()

        if self.figure.show_label_lines:
            self.figure.make_label_lines()

        self._refresh_manual_refinement_controls()

    def _load_cluster_labels(self):
        """Open a load dialog and import cluster assignments from a CSV file."""
        import pandas as pd

        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Load cluster labels",
            "",
            "CSV files (*.csv);;All files (*)",
        )
        if not path:
            return

        try:
            df = pd.read_csv(path)
        except Exception as e:
            self.figure.show_message(
                f"Failed to read CSV: {e}", color="red", duration=4
            )
            return

        if "id" not in df.columns:
            self.figure.show_message(
                "CSV must contain 'id' column.", color="red", duration=4
            )
            return

        if CLUSTER_DATA_COLUMN in df.columns:
            cluster_col = CLUSTER_DATA_COLUMN
        elif "cluster" in df.columns:
            cluster_col = "cluster"
        else:
            self.figure.show_message(
                f"CSV must contain '{CLUSTER_DATA_COLUMN}' or 'cluster' column for cluster labels.",
                color="red",
                duration=4,
            )

        df[cluster_col] = df[cluster_col].fillna(-1).astype(self.figure.ids.dtype)

        df = df.set_index("id")
        labels_series = df[cluster_col].reindex(self.figure.ids)
        if labels_series.isna().any():
            self.figure.show_message(
                "Some IDs in the CSV do not match the current data. "
                "Missing entries will be marked as noise (-1).",
                color="orange",
                duration=4,
            )
        labels = labels_series.fillna(-1).astype(int).to_numpy()

        self._cluster_labels = labels
        self._cluster_labels_original = labels.copy()

        # This function takes care of writing cluster labels to metadata, updating the label/color combo boxes, and refreshing the display
        self._apply_cluster_labels()

    def _export_cluster_labels(self):
        """Open a save dialog and write cluster assignments to a CSV file."""
        if not hasattr(self, "_cluster_labels") or self._cluster_labels is None:
            return

        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Export cluster labels",
            "clusters.csv",
            "CSV files (*.csv);;All files (*)",
        )
        if not path:
            return

        import pandas as pd

        labels = self._cluster_labels
        ids = self.figure.ids if hasattr(self.figure, "ids") else np.arange(len(labels))

        df = pd.DataFrame({"id": ids, CLUSTER_DATA_COLUMN: labels})
        df.to_csv(path, index=False)

    def _update_cluster_result_label(self, labels):
        """Update clustering summary label from integer cluster assignments."""
        unique_clusters = (
            np.unique(labels[labels >= 0])
            if (labels >= 0).any()
            else np.array([], dtype=int)
        )
        n_clusters = len(unique_clusters)
        n_noise = int((labels == -1).sum())
        if n_noise > 0:
            self.cluster_result_label.setText(
                f"{n_clusters} clusters, {n_noise} noise points"
            )
        else:
            self.cluster_result_label.setText(f"{n_clusters} clusters")

    def _parse_cluster_series(self, values):
        """Parse a metadata cluster column into integer cluster IDs."""
        parsed = np.empty(len(values), dtype=int)
        for i, value in enumerate(values):
            if pd.isna(value):
                return None

            if isinstance(value, (int, np.integer)):
                parsed[i] = int(value)
                continue

            text = str(value).strip().lower()
            if text == "noise":
                parsed[i] = -1
                continue

            match = re.fullmatch(r"cluster_(-?\d+)", text)
            if match:
                parsed[i] = int(match.group(1))
                continue

            if re.fullmatch(r"-?\d+", text):
                parsed[i] = int(text)
                continue

            return None

        return parsed

    def _ensure_cluster_labels_loaded(self):
        """Ensure editable cluster labels exist, loading from metadata when possible."""
        labels = getattr(self, "_cluster_labels", None)
        if labels is not None:
            return np.asarray(labels, dtype=int)

        meta = getattr(self.figure, "metadata", None)
        if meta is None or CLUSTER_DATA_COLUMN not in meta.columns:
            return None

        parsed = self._parse_cluster_series(meta[CLUSTER_DATA_COLUMN].values)
        if parsed is None:
            return None

        self._cluster_labels = parsed.copy()
        self._cluster_labels_original = parsed.copy()
        return self._cluster_labels

    def _manual_cluster_target_id(self):
        """Return selected target cluster ID from manual refinement combo."""
        text = self.cluster_manual_set_combo.currentText().strip().lower()
        if text in ("noise", "noise (-1)"):
            return -1

        match = re.fullmatch(r"cluster_(-?\d+)", text)
        if match:
            return int(match.group(1))

        if re.fullmatch(r"-?\d+", text):
            return int(text)

        value = self.cluster_manual_set_combo.currentData()
        if value is not None:
            return int(value)

        return None

    def _refresh_manual_refinement_controls(self):
        """Show/hide and repopulate manual refinement controls."""
        if not hasattr(self, "cluster_manual_group"):
            return

        labels = self._ensure_cluster_labels_loaded()
        has_clusters = labels is not None and len(labels) > 0
        self.cluster_manual_group.setVisible(has_clusters)
        if not has_clusters:
            return

        current_target = self._manual_cluster_target_id()
        unique_labels = np.unique(labels)
        non_noise = np.sort(unique_labels[unique_labels >= 0])

        self.cluster_manual_set_combo.blockSignals(True)
        self.cluster_manual_set_combo.clear()
        self.cluster_manual_set_combo.addItem("noise (-1)", -1)
        for cluster_id in non_noise:
            self.cluster_manual_set_combo.addItem(str(int(cluster_id)), int(cluster_id))

        if current_target is None:
            self.cluster_manual_set_combo.setCurrentIndex(0)
        else:
            idx = self.cluster_manual_set_combo.findData(int(current_target))
            if idx >= 0:
                self.cluster_manual_set_combo.setCurrentIndex(idx)
            else:
                self.cluster_manual_set_combo.setEditText(str(int(current_target)))
        self.cluster_manual_set_combo.blockSignals(False)

    @requires_selection
    def _manual_refine_set_cluster(self):
        """Set current selection to an existing cluster ID."""
        labels = self._ensure_cluster_labels_loaded()
        if labels is None:
            self.figure.show_message("No clusters loaded", color="red", duration=2)
            return

        cluster_id = self._manual_cluster_target_id()
        if cluster_id is None:
            self.figure.show_message("Invalid cluster ID", color="red", duration=2)
            return

        indices = self.selected_indices
        if indices.size == 0:
            self.figure.show_message("No points selected", color="red", duration=2)
            return

        labels = labels.copy()
        labels[indices] = int(cluster_id)
        self._cluster_labels = labels
        self._apply_cluster_labels()
        self._refresh_manual_refinement_controls()

    @requires_selection
    def _manual_refine_new_cluster(self):
        """Assign current selection to a new cluster ID."""
        labels = self._ensure_cluster_labels_loaded()
        if labels is None:
            self.figure.show_message("No clusters loaded", color="red", duration=2)
            return

        existing = labels[labels >= 0]
        next_cluster_id = int(existing.max() + 1) if existing.size else 0

        indices = self.selected_indices
        if indices.size == 0:
            self.figure.show_message("No points selected", color="red", duration=2)
            return

        labels = labels.copy()
        labels[indices] = next_cluster_id
        self._cluster_labels = labels
        self._apply_cluster_labels()
        self._refresh_manual_refinement_controls()

    @requires_selection
    def _manual_refine_reset(self):
        """Reset current selection to original automatic cluster assignments."""
        labels = self._ensure_cluster_labels_loaded()
        original = getattr(self, "_cluster_labels_original", None)
        if labels is None or original is None:
            self.figure.show_message("No clusters loaded", color="red", duration=2)
            return

        indices = self.selected_indices
        if indices.size == 0:
            self.figure.show_message("No points selected", color="red", duration=2)
            return

        labels = labels.copy()
        labels[indices] = np.asarray(original, dtype=int)[indices]
        self._cluster_labels = labels
        self._apply_cluster_labels()
        self._refresh_manual_refinement_controls()

    def add_tooltip(self, widget, text, anchor="group_label"):
        """Add a reusable round help icon tooltip to a widget."""
        if not hasattr(self, "_tooltip_anchors"):
            self._tooltip_anchors = {}

        button = QtWidgets.QToolButton(widget)
        button.setText("?")
        button.setAutoRaise(True)
        button.setFixedSize(14, 14)
        button.setStyleSheet(
            "QToolButton {"
            "border: 1px solid palette(mid);"
            "border-radius: 7px;"
            "font-weight: bold;"
            "padding: 0px;"
            "}"
        )
        button.setToolTip(text)

        self._tooltip_anchors.setdefault(widget, []).append((button, anchor))
        widget.installEventFilter(self)
        QtCore.QTimer.singleShot(0, lambda: self._position_tooltips(widget))
        return button

    def _position_tooltips(self, widget):
        """Position one or more tooltip icons for a widget."""
        items = getattr(self, "_tooltip_anchors", {}).get(widget, [])
        if not items:
            return

        for i, (button, anchor) in enumerate(items):
            if anchor == "group_label" and isinstance(widget, QtWidgets.QGroupBox):
                opt = QtWidgets.QStyleOptionGroupBox()
                widget.initStyleOption(opt)
                label_rect = widget.style().subControlRect(
                    QtWidgets.QStyle.CC_GroupBox,
                    opt,
                    QtWidgets.QStyle.SC_GroupBoxLabel,
                    widget,
                )
                if label_rect.isValid():
                    x = label_rect.right() + 4 + i * (button.width() + 3)
                    y = label_rect.center().y() - (button.height() // 2)
                else:
                    x = widget.width() - button.width() - 6 - i * (button.width() + 3)
                    y = 6
            else:
                x = widget.width() - button.width() - 6 - i * (button.width() + 3)
                y = 6

            max_x = max(0, widget.width() - button.width() - 6)
            button.move(max(0, min(x, max_x)), max(0, y))
            button.raise_()

    def eventFilter(self, obj, event):
        """Keep custom tooltip buttons aligned when host widgets change."""
        if obj in getattr(self, "_tooltip_anchors", {}) and event.type() in (
            QtCore.QEvent.Resize,
            QtCore.QEvent.Show,
            QtCore.QEvent.LayoutRequest,
        ):
            self._position_tooltips(obj)
        return super().eventFilter(obj, event)

    def add_split(self, layout):
        """Add horizontal divider."""
        # layout.addSpacing(5)
        layout.addWidget(QHLine())
        # layout.addSpacing(5)

    def update_controls(self):
        """Update the controls based on the current figure state."""
        # self.update_ann_combo_box()
        self.update_embedding_selector()
        self.update_umap_options()
        self.update_fidelity_options()
        self.update_cluster_options()
        self.update_label_combo_boxes()
        self.update_searchbar_completer()
        self.update_distance_edges_controls()
        self.update_scope_options()

    def sync_point_scale_spinbox(self):
        """Refresh the point-scale spinbox to match the figure's current scale.

        ``set_points`` may auto-scale the global point size by dataset size, so
        after a load the spinbox needs to be re-synced to show that value.
        """
        self.point_scale_spinbox.blockSignals(True)
        self.point_scale_spinbox.setValue(float(self.figure.point_scale))
        self.point_scale_spinbox.blockSignals(False)

    def capture_display_state(self):
        """Snapshot the label/color/size selections and their settings.

        Returns a plain dict that ``apply_display_state`` can restore onto
        another ``ScatterControls`` instance (e.g. when opening a selection in a
        new tab) to make it mirror this view's appearance.
        """

        def _slider_state(slider):
            if not slider.isVisible():
                return None
            return {
                "lo": slider.lo,
                "hi": slider.hi,
                "vmin": slider.vmin,
                "vcenter": slider.vcenter,
                "vmax": slider.vmax,
                "show_center": slider.show_center,
            }

        return {
            "label_col": self.label_combo_box.currentText(),
            "label_counts": self.label_count_check.isChecked(),
            "label_outlines": self.label_outlines_check.isChecked(),
            "color_col": self.color_combo_box.currentText(),
            "palette": self.palette_combo_box.currentText(),
            "color_range": _slider_state(self.color_range_slider),
            "size_col": self.size_combo_box.currentText(),
            "size_range": _slider_state(self.size_range_slider),
            "font_size": self.figure.font_size,
            "point_scale": self.figure.point_scale,
            "label_vis_limit": self.figure.label_vis_limit,
        }

    def _set_combo_text(self, combo_box, text):
        """Select ``text`` in ``combo_box`` (signals blocked); no-op if absent.

        Silently skips values not offered by this box — e.g. the cluster option
        when the target view has no clustering — leaving the default selection.
        """
        if not text:
            return
        idx = combo_box.findText(text)
        if idx < 0:
            return
        combo_box.blockSignals(True)
        combo_box.setCurrentIndex(idx)
        combo_box.blockSignals(False)

    def _apply_slider_state(self, slider, slider_state):
        """Configure ``slider`` to match a captured snapshot, or hide it."""
        if not slider_state:
            slider.setVisible(False)
            return
        slider.blockSignals(True)
        slider.set_data_range(slider_state["lo"], slider_state["hi"])
        slider.set_values(
            slider_state["vmin"], slider_state["vcenter"], slider_state["vmax"]
        )
        slider.set_diverging(slider_state["show_center"])
        slider.blockSignals(False)
        slider.setVisible(True)

    def apply_display_state(self, state):
        """Restore a ``capture_display_state`` snapshot onto this instance.

        Widget values are set with their signals blocked, then applied once via
        the public setters. The explicit apply ensures the configuration takes
        effect even when a selection matches the current index (which would emit
        no change signal) and that the parent's palette/range are honored.
        """
        if not state:
            return

        # Settings-tab knobs first; their valueChanged lambdas write to the
        # figure, and set_labels/set_sizes below pick up the new font/scale.
        if state.get("font_size") is not None:
            self.font_size_slider.setValue(float(state["font_size"]))
        if state.get("point_scale") is not None:
            self.point_scale_spinbox.setValue(float(state["point_scale"]))
        if state.get("label_vis_limit") is not None:
            self.max_label_vis_slider.setValue(int(state["label_vis_limit"]))

        # --- Labels ---
        self._set_combo_text(self.label_combo_box, state.get("label_col"))
        self.label_count_check.blockSignals(True)
        self.label_count_check.setChecked(bool(state.get("label_counts")))
        self.label_count_check.blockSignals(False)
        self.label_outlines_check.blockSignals(True)
        self.label_outlines_check.setChecked(bool(state.get("label_outlines")))
        self.label_outlines_check.blockSignals(False)
        # set_label_outlines only flips this flag; mirror it since we blocked
        # the checkbox's signal above.
        self.figure.show_label_lines = self.label_outlines_check.isChecked()

        # --- Colors --- configure the range slider directly (rather than via
        # _on_color_column_changed, which would re-fit it to the subset).
        self._set_combo_text(self.color_combo_box, state.get("color_col"))
        self._set_combo_text(self.palette_combo_box, state.get("palette"))
        self._apply_slider_state(self.color_range_slider, state.get("color_range"))
        # Palette is meaningless for colour-typed or cluster columns; mirror the
        # enabled state that _on_color_column_changed would set.
        color_col = self.color_combo_box.currentText()
        is_color_col = (
            color_col not in ("", "Default", CLUSTER_DATA_OPTION)
            and self.meta_data is not None
            and color_col in self.meta_data.columns
            and is_color_column(self.meta_data[color_col])
        )
        self.palette_combo_box.setEnabled(
            not is_color_col and color_col != CLUSTER_DATA_OPTION
        )

        # --- Sizes ---
        self._set_combo_text(self.size_combo_box, state.get("size_col"))
        self._apply_slider_state(self.size_range_slider, state.get("size_range"))

        # Apply everything once now that all widgets are configured.
        self.set_labels()
        self.set_colors()
        self.set_sizes()

    def update_embedding_selector(self):
        """Refresh the active-embedding selector from the figure's entries."""
        entries = getattr(self.figure, "embedding_entries", None) or []
        combo = self.embedding_selector_combo
        combo.blockSignals(True)
        combo.clear()
        for entry in entries:
            combo.addItem(str(entry.get("name", "embedding")))
        active = getattr(self.figure, "active_embedding", None)
        if active is not None and 0 <= active < len(entries):
            combo.setCurrentIndex(active)
        combo.blockSignals(False)
        # Only worth showing when there is something to switch between.
        self.embedding_selector_group.setVisible(len(entries) > 1)
        self._sync_window_embedding_status()

    def _scope_columns(self):
        """Columns available for scope filters."""
        if self.meta_data is None:
            return []
        return [c for c in self.meta_data.columns if not str(c).startswith("_")]

    def _refresh_scope_row_flags(self):
        """Hide the AND/OR combinator on the first scope row."""
        for i, row in enumerate(self.scope_rows):
            row.set_first(i == 0)

    def _add_scope_row(self, *_):
        """Add a new scope filter row."""
        row = _ScopeFilterRow(df_getter=lambda: self.meta_data)
        row.set_columns(self._scope_columns())
        row.changed.connect(self._update_scope_mask)
        row.removed.connect(self._remove_scope_row)
        self.scope_rows.append(row)
        self.scope_rows_layout.addWidget(row)
        self._refresh_scope_row_flags()
        self._update_scope_mask()

    def _remove_scope_row(self, row):
        """Remove a scope filter row."""
        if row in self.scope_rows:
            self.scope_rows.remove(row)
        self.scope_rows_layout.removeWidget(row)
        row.deleteLater()
        self._refresh_scope_row_flags()
        self._update_scope_mask()

    def update_scope_options(self):
        """Refresh scope filter columns after the underlying data changed."""
        if not hasattr(self, "scope_rows"):
            return
        cols = self._scope_columns()
        for row in self.scope_rows:
            row.set_columns(cols)
        self._update_scope_mask()

    def _update_scope_mask(self):
        """Recompute the selection scope mask and pass it to the figure."""
        df = self.meta_data
        if df is None or not self.scope_rows:
            self.figure.set_scope(None)
            self.scope_match_label.setVisible(False)
            return
        mask = self.scope_rows[0].mask(df)
        for row in self.scope_rows[1:]:
            m = row.mask(df)
            if row.combinator.currentText() == "OR":
                mask = mask | m
            else:
                mask = mask & m
        self.figure.set_scope(mask)

        n = int(mask.sum())
        self.scope_match_label.setText(
            f"1 of {len(df)} rows matches"
            if n == 1
            else f"{n} of {len(df)} rows match"
        )
        self.scope_match_label.setVisible(True)

    def update_label_combo_boxes(self):
        """Update the items in the label, color by and size by combo boxes."""
        # First, collect the items we want to have:
        items = {"Default"}
        if self.figure.metadata is not None:
            items.update(
                col for col in self.figure.metadata.columns if not col.startswith("_")
            )
        if getattr(self, "_cluster_labels", None) is not None:
            items.add(CLUSTER_DATA_OPTION)

        # Now edit the combo boxes to match these items, trying to preserve the current selection if possible
        for combo_box in (self.label_combo_box, self.color_combo_box):
            # First clear all items that no longer exist
            for i in reversed(range(combo_box.count())):
                text = combo_box.itemText(i)
                if text in ("Default", CLUSTER_DATA_OPTION):
                    continue
                if (
                    self.figure.metadata is None
                    or text not in self.figure.metadata.columns
                ):
                    combo_box.removeItem(i)

            # Now add missing items
            for item in sorted(items):
                if combo_box.findText(item) < 0:
                    combo_box.addItem(item)

        # The size combo box only offers numeric columns
        size_items = {"Default"}
        if self.figure.metadata is not None:
            size_items.update(
                col
                for col in self.figure.metadata.columns
                if not col.startswith("_")
                and self.figure.metadata[col].dtype.kind in "iuf"
            )

        for i in reversed(range(self.size_combo_box.count())):
            text = self.size_combo_box.itemText(i)
            if text == "Default":
                continue
            if text not in size_items:
                self.size_combo_box.removeItem(i)

        for item in sorted(size_items):
            if self.size_combo_box.findText(item) < 0:
                self.size_combo_box.addItem(item)

    def _has_square_distance_matrix(self):
        """Whether the active source provides a full (square) distance matrix."""
        dists = getattr(self.figure, "dists", None)
        if dists is None:
            return False
        arr = dists.get("distances") if isinstance(dists, dict) else dists
        return (
            arr is not None
            and hasattr(arr, "shape")
            and len(arr.shape) == 2
            and arr.shape[0] == arr.shape[1]
        )

    def update_distance_edges_controls(self):
        """Show distance-edge controls only when a full distance matrix exists.

        The "Distances" edges feature needs all pairwise distances, so it's
        hidden entirely for feature- or KNN-only projects rather than left
        visible-but-disabled.
        """
        has_distances = self._has_square_distance_matrix()

        self.distance_edges_group.setVisible(has_distances)
        self.show_distance_edges_check.setEnabled(has_distances)

        # Drop any stale edges when the source can no longer supply them.
        if not has_distances and self.show_distance_edges_check.isChecked():
            self.show_distance_edges_check.setChecked(False)

        self.distance_edges_slider.setEnabled(
            has_distances and self.show_distance_edges_check.isChecked()
        )

    def update_umap_options(self):
        """Update the Embeddings tab state and the data-source combo box."""
        self.umap_dist_combo_box.clear()

        embeddings_tab_index = self.tabs.indexOf(self.tab5)
        entries = getattr(self.figure, "embedding_entries", None) or []
        dists = getattr(self.figure, "dists", None)

        # The tab is usable whenever there is something to switch between or a
        # high-dim source to recompute from. The recompute controls themselves
        # are only enabled when the active embedding has a source.
        self.tabs.setTabEnabled(
            embeddings_tab_index, (len(entries) > 1) or (dists is not None)
        )
        self.embedding_recompute_widget.setEnabled(dists is not None)

        # Add any distances/features/knn we have for the active embedding
        if isinstance(dists, dict):
            for key in dists.keys():
                self.umap_dist_combo_box.addItem(key)

        # `_update_embedding_input_controls` re-gates methods/sliders/warning
        # for the (now repopulated) selected source.
        self._update_embedding_input_controls()

    def _refresh_embedding_source_gating(self):
        """Limit methods/sliders for the selected source.

        Gating tracks the *selected* source: a KNN graph allows only UMAP/t-SNE
        with capped neighbor counts, while distances/features keep all options.
        The KNN limitation is surfaced via the data-source info label (see
        `_update_embedding_source_info`).
        """
        graph = self._selected_embedding_knn()
        self._populate_embedding_methods(is_knn=graph is not None)
        if graph is not None:
            k = int(graph.k)
            self.umap_n_neighbors_slider.setMaximum(max(1, k))
            # sklearn t-SNE needs ~3*perplexity+1 neighbors in the graph.
            self.tsne_perplexity_slider.setMaximum(max(1.0, (k - 1) / 3.0))
        else:
            self.umap_n_neighbors_slider.setMaximum(200)
            self.tsne_perplexity_slider.setMaximum(200.0)
        # Repopulating may have changed the current method (signals blocked),
        # so re-sync the method-specific settings widgets without recomputing.
        self._sync_method_settings_widgets()

    def _populate_embedding_methods(self, is_knn=False):
        """Populate the embedding method combo, limited to UMAP/t-SNE for KNN."""
        combo = self.umap_method_combo_box
        if is_knn:
            methods = ["UMAP", "t-SNE"]
        else:
            methods = ["UMAP", "MDS", "t-SNE"]
            if getattr(self.figure, "feats", None) is not None:
                methods.append("PaCMAP")

        current = combo.currentText()
        combo.blockSignals(True)
        combo.clear()
        combo.addItems(methods)
        combo.blockSignals(False)
        if current in methods:
            combo.setCurrentText(current)
        elif methods:
            combo.setCurrentIndex(0)

    def on_embedding_switched(self):
        """React to the figure switching its active embedding.

        Syncs the selector, refreshes the data-source and fidelity options for
        the new sources, recomputes the fidelity column if it currently drives
        point size, and mirrors the active artifacts into the owning window.
        """
        active = getattr(self.figure, "active_embedding", None)
        entries = getattr(self.figure, "embedding_entries", None) or []

        # Sync the selector without re-triggering a switch.
        self.embedding_selector_combo.blockSignals(True)
        if active is not None and 0 <= active < len(entries):
            self.embedding_selector_combo.setCurrentIndex(active)
        self.embedding_selector_combo.blockSignals(False)

        self.update_umap_options()
        self.update_fidelity_options()
        self.update_cluster_options()
        self.update_distance_edges_controls()
        self._reconcile_knn_edges()

        # Keep fidelity in sync with the new embedding when it drives sizing.
        if self.size_combo_box.currentText() == FIDELITY_DATA_COLUMN:
            if getattr(self.figure, "dists", None) is None:
                # No source to compute fidelity from -> fall back to default.
                self.size_combo_box.setCurrentText("Default")
            else:
                target = None
                if active is not None and 0 <= active < len(entries):
                    target = np.asarray(entries[active]["embedding"])
                if self._compute_fidelity_column(positions=target):
                    self._on_size_column_changed()

        self._sync_window_data_active()
        self._sync_window_embedding_status()

    def _owner_view(self):
        """Walk up from the canvas to the view (MainWidget) owning this figure.

        Views live as tabs inside the main window, so we stop at the owning
        view rather than the top-level window. Returns None before the figure
        is attached to a view (e.g. during construction).
        """
        owner = self.figure.canvas
        while owner is not None and getattr(owner, "fig_scatter", None) is not self.figure:
            owner = owner.parentWidget()
        return owner

    def _sync_window_embedding_status(self):
        """Ask the owning window to refresh its status-bar embedding indicator."""
        owner = self._owner_view()
        window = owner.window() if owner is not None else None
        if window is not None and hasattr(window, "_update_embedding_status"):
            window._update_embedding_status()

    def _sync_window_data_active(self):
        """Mirror the active embedding's artifacts into the owning view's _data."""
        owner = self._owner_view()
        if owner is None or not isinstance(getattr(owner, "_data", None), dict):
            return

        active = getattr(self.figure, "active_embedding", None)
        entries = getattr(self.figure, "embedding_entries", None) or []
        if active is None or not (0 <= active < len(entries)):
            return

        entry = entries[active]
        owner._data["embeddings"] = entry["embedding"]
        owner._data["features"] = entry["features"]
        owner._data["distances"] = entry["distances"]
        owner._data["knn"] = entry.get("knn")
        owner._data["active_embedding"] = active

        # Menu enablement reflects the active tab only.
        window = owner.window()
        current_view = getattr(window, "current_view", None)
        if hasattr(window, "_update_view_actions") and (
            current_view is None or current_view() is owner
        ):
            window._update_view_actions()

    def update_fidelity_options(self):
        """Update fidelity controls based on available distances/features."""
        fidelity_tab_index = self.tabs.indexOf(self.tab6)
        dists = getattr(self.figure, "dists", None)

        # The active-embedding indicator always reflects the current embedding.
        self._update_fidelity_active_embedding_label()

        if dists is None:
            self.tabs.setTabEnabled(fidelity_tab_index, False)
            for box in (
                self.fidelity_data_combo_box,
                self.fidelity_metric_combo_box,
            ):
                box.blockSignals(True)
                box.clear()
                box.blockSignals(False)
            self.fidelity_source_info_label.setText("")
            self.update_evaluate_labels_data_options()
            return

        self.tabs.setTabEnabled(fidelity_tab_index, True)

        # Available high-dimensional sources, in preference order. A feature
        # metric (euclidean/cosine/...) is a separate choice surfaced only when
        # the "features" source is selected (see `_update_fidelity_input_controls`).
        sources = []
        if isinstance(dists, dict):
            if dists.get("knn") is not None:
                sources.append("knn")
            if "distances" in dists:
                sources.append("precomputed")
            if "features" in dists:
                sources.append("features")
        elif isinstance(dists, (np.ndarray, pd.DataFrame)):
            sources.append(
                "precomputed" if dists.shape[0] == dists.shape[1] else "features"
            )

        combo = self.fidelity_data_combo_box
        current = combo.currentText()
        combo.blockSignals(True)
        combo.clear()
        combo.addItems(sources)
        if current in sources:
            combo.setCurrentText(current)
        elif sources:
            combo.setCurrentIndex(0)
        combo.blockSignals(False)
        # Only worth interacting with when there's something to switch between.
        combo.setEnabled(len(sources) > 1)

        # Repopulate the feature-metric combo (static; preserve the selection).
        metric_combo = self.fidelity_metric_combo_box
        if metric_combo.count() == 0:
            metric_combo.blockSignals(True)
            metric_combo.addItems(
                ["euclidean", "cosine", "manhattan", "correlation", "chebyshev"]
            )
            metric_combo.blockSignals(False)

        self._update_fidelity_input_controls()
        self.update_evaluate_labels_data_options()

    def _update_fidelity_active_embedding_label(self):
        """Mirror the active embedding's name into the Fidelity tab indicator."""
        label = getattr(self, "fidelity_active_embedding_label", None)
        if label is None:
            return
        entries = getattr(self.figure, "embedding_entries", None) or []
        active = getattr(self.figure, "active_embedding", None)
        if active is not None and 0 <= active < len(entries):
            label.setText(str(entries[active].get("name", "embedding")))
        else:
            label.setText("—")

    def _fidelity_source(self):
        """Return the selected fidelity Data source ('knn'/'precomputed'/'features')."""
        combo = getattr(self, "fidelity_data_combo_box", None)
        if combo is None:
            return None
        return combo.currentText() or None

    def _fidelity_metric(self):
        """Resolve the selected Data source + metric to a `metric` argument.

        'knn'/'precomputed' pass straight through; 'features' resolves to the
        chosen feature metric (euclidean/cosine/...). Falls back to 'auto'.
        """
        source = self._fidelity_source()
        if source == "features":
            return self.fidelity_metric_combo_box.currentText() or "euclidean"
        return source or "auto"

    def _on_fidelity_source_changed(self):
        """React to a Data-source change: re-gate controls and refresh edges."""
        self._update_fidelity_input_controls()
        # Keep any live KNN edges in sync with the new source.
        self.set_knn_edges()

    def _update_fidelity_input_controls(self):
        """Toggle the metric row, cap k, and refresh the source info label."""
        if not hasattr(self, "fidelity_data_combo_box"):
            return

        source = self._fidelity_source()
        is_features = source == "features"
        # The metric only applies to the feature source.
        self.fidelity_input_form.setRowVisible(
            self.fidelity_metric_combo_box, is_features
        )

        # Cap neighbor counts to what the KNN graph stores when it's the source.
        graph = self._active_knn() if source == "knn" else None
        kmax = max(1, int(graph.k)) if graph is not None else 200
        self.fidelity_k_slider.setMaximum(kmax)
        self.fidelity_knn_k_slider.setMaximum(kmax)

        self._update_fidelity_source_info()

    def _update_fidelity_source_info(self):
        """Show shape (and type/metric, if available) for the selected source."""
        label = getattr(self, "fidelity_source_info_label", None)
        if label is None:
            return

        dists = getattr(self.figure, "dists", None)
        source = self._fidelity_source()
        if not isinstance(dists, dict) or not source:
            label.setText("")
            return

        if source == "knn":
            graph = dists.get("knn")
            if graph is None:
                label.setText("")
                return
            parts = [f"KNN: {len(graph)}×{graph.k}"]
            info = self._active_source_info("knn")
            if isinstance(info, dict) and info.get("metric"):
                parts.append(f"metric: {info['metric']}")
            label.setText("  ·  ".join(parts))
            return

        key = "distances" if source == "precomputed" else "features"
        arr = dists.get(key)
        if arr is None or not hasattr(arr, "shape"):
            label.setText("")
            return

        parts = [f"shape: {tuple(arr.shape)}"]
        info = self._active_source_info(key)
        if isinstance(info, dict):
            if info.get("type"):
                parts.append(f"type: {info['type']}")
            if info.get("metric"):
                parts.append(f"metric: {info['metric']}")
        label.setText("  ·  ".join(parts))

    def _selected_embedding_data(self):
        """Return the currently selected distance/feature matrix."""
        dists = getattr(self.figure, "dists", None)
        if dists is None:
            return None

        if isinstance(dists, dict):
            key = self.umap_dist_combo_box.currentText()
            if not key:
                return None
            return dists.get(key)

        return dists

    def _embedding_feature_partition_names(self, arr):
        """Return ordered top-level partition names for MultiIndex feature columns."""
        if not isinstance(arr, pd.DataFrame):
            return []
        if not isinstance(arr.columns, pd.MultiIndex):
            return []

        names = []
        seen = set()
        for value in arr.columns.get_level_values(0):
            label = str(value)
            if label in seen:
                continue
            seen.add(label)
            names.append(label)
        return names

    def _apply_embedding_feature_partition_filter(self, arr):
        """Filter feature columns by selected top-level partition checkboxes."""
        if not isinstance(arr, pd.DataFrame):
            return arr
        if not isinstance(arr.columns, pd.MultiIndex):
            return arr
        if not hasattr(self, "_embedding_partition_checks"):
            return arr

        selected = {
            name
            for name, checkbox in self._embedding_partition_checks.items()
            if checkbox.isChecked()
        }
        if not selected:
            return arr.iloc[:, 0:0].copy()

        top_level = arr.columns.get_level_values(0).astype(str)
        mask = np.asarray([name in selected for name in top_level], dtype=bool)
        if not mask.any():
            return arr.iloc[:, 0:0].copy()
        return arr.loc[:, mask].copy()

    def _update_embedding_feature_partition_controls(self, arr, is_feature_input):
        """Show/update partition checkboxes for MultiIndex feature inputs."""
        if not hasattr(self, "umap_feature_partition_widget"):
            return

        names = (
            self._embedding_feature_partition_names(arr)
            if is_feature_input
            else []
        )
        has_partitions = len(names) > 0

        if hasattr(self, "umap_feature_subset_group"):
            self.umap_feature_subset_group.setVisible(has_partitions)
            self.umap_feature_subset_group.setEnabled(is_feature_input)

        old_checks = getattr(self, "_embedding_partition_checks", {})
        old_state = {name: check.isChecked() for name, check in old_checks.items()}

        layout = self.umap_feature_partition_widget.layout()
        while layout.count():
            item = layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

        self._embedding_partition_checks = {}
        for name in names:
            check = QtWidgets.QCheckBox(name)
            check.setChecked(old_state.get(name, True))
            check.stateChanged.connect(self.calculate_embeddings_maybe)
            layout.addWidget(check)
            self._embedding_partition_checks[name] = check

    def _update_embedding_input_controls(self):
        """Toggle embedding controls based on selected input type."""
        if not hasattr(self, "pca_check"):
            return

        # Re-gate available methods/sliders/warning for the selected source.
        self._refresh_embedding_source_gating()

        arr = self._selected_embedding_data()
        is_knn_input = is_knn_graph(arr)
        is_feature_input = (
            (arr is not None)
            and (not is_knn_input)
            and (not is_precomputed_distance_matrix(arr))
        )
        self._update_embedding_feature_partition_controls(arr, is_feature_input)
        arr_filtered = self._apply_embedding_feature_partition_filter(arr)

        has_selected_features = True
        if (
            is_feature_input
            and arr_filtered is not None
            and hasattr(arr_filtered, "shape")
            and len(arr_filtered.shape) > 1
            and arr_filtered.shape[1] == 0
        ):
            has_selected_features = False

        if hasattr(self, "feature_options_group"):
            self.feature_options_group.setEnabled(is_feature_input)

        if hasattr(self, "umap_button"):
            self.umap_button.setEnabled((arr is not None) and (not is_feature_input or has_selected_features))

        # PCA pre-reduction is only meaningful for feature vectors.
        if not is_feature_input:
            self.pca_check.setChecked(False)
        self.pca_check.setEnabled(is_feature_input and has_selected_features)
        self.pca_n_components_slider.setEnabled(
            is_feature_input and has_selected_features and self.pca_check.isChecked()
        )

        self._update_densmap_controls()
        self._update_embedding_source_info()

    def _active_source_info(self, key):
        """Return the raw `info` spec (with type/metric) for the selected source."""
        entries = getattr(self.figure, "embedding_entries", None) or []
        active = getattr(self.figure, "active_embedding", None)
        if active is None or not (0 <= active < len(entries)):
            return None
        entry = entries[active]
        if key == "features":
            return entry.get("features_info")
        # A KNN graph is parsed from the `distances` spec, so its type/metric
        # live in `distances_info`.
        if key in ("distances", "knn"):
            return entry.get("distances_info")
        return None

    def _update_embedding_source_info(self):
        """Show shape (and type/metric, if available) for the selected source."""
        label = getattr(self, "umap_source_info_label", None)
        if label is None:
            return

        dists = getattr(self.figure, "dists", None)
        key = self.umap_dist_combo_box.currentText()
        arr = dists.get(key) if isinstance(dists, dict) else None

        if is_knn_graph(arr):
            parts = [f"KNN: {len(arr)}×{arr.k}"]
            info = self._active_source_info(key)
            if isinstance(info, dict) and info.get("metric"):
                parts.append(f"metric: {info['metric']}")
            # KNN graphs only support a subset of methods/options.
            parts.append("some options unavailable")
            label.setText("  ·  ".join(parts))
            return

        if arr is None or not hasattr(arr, "shape"):
            label.setText("")
            return

        parts = [f"shape: {tuple(arr.shape)}"]
        info = self._active_source_info(key)
        if isinstance(info, dict):
            if info.get("type"):
                parts.append(f"type: {info['type']}")
            if info.get("metric"):
                parts.append(f"metric: {info['metric']}")
        label.setText("  ·  ".join(parts))

    def _update_densmap_controls(self):
        """Show the dens_lambda control only when DensMAP is enabled."""
        show = self.umap_densmap_check.isChecked()
        self.umap_dens_lambda_label.setVisible(show)
        self.umap_dens_lambda_spinbox.setVisible(show)
        if show:
            self.umap_dens_lambda_label.setEnabled(True)
            self.umap_dens_lambda_spinbox.setEnabled(True)

    def set_add_group(self):
        """Set whether to add neurons as group when selected."""
        self.figure._add_as_group = self.add_group_check.isChecked()

    def set_ngl_cache(self):
        """Set whether the ngl viewer should cache neurons."""
        if hasattr(self.figure, "ngl_viewer"):
            self.figure.ngl_viewer.use_cache = self.ngl_cache_neurons.isChecked()

            if not self.ngl_cache_neurons.isChecked():
                self.figure.ngl_viewer.clear_cache()

    def set_ngl_cache_size(self):
        """Set the maximum number of neurons the ngl viewer will cache."""
        if hasattr(self.figure, "ngl_viewer"):
            self.figure.ngl_viewer.max_cache_size = self.ngl_cache_size.value()

    def clear_ngl_cache(self):
        """Clear the ngl viewer's neuron cache."""
        if hasattr(self.figure, "ngl_viewer"):
            self.figure.ngl_viewer.clear_cache()

    def set_dclick_deselect(self):
        """Set whether to deselect on double-click."""
        self.figure.deselect_on_dclick = self.dclick_deselect.isChecked()

    def set_empty_deselect(self):
        """Set whether to deselect on double-click."""
        self.figure.deselect_on_empty = self.empty_deselect.isChecked()

    def set_label_counts(self):
        """Set whether to add counts to the labels."""
        self.set_labels()  # Update the labels

    def find_next(self):
        """Find next occurrence."""
        text = self.searchbar.text()
        if text:
            regex = False
            if text.startswith("/"):
                regex = True
                text = text[1:]

            if (
                not hasattr(self, "_label_search")
                or self._label_search.query != text
                or self._label_search.regex != regex
            ):
                self._label_search = self.figure.find_label(text, regex=regex)

            # LabelSearch can be `None` if no match found
            if self._label_search:
                self._label_search.next()

    def find_previous(self):
        """Find previous occurrence."""
        text = self.searchbar.text()
        if text:
            regex = False
            if text.startswith("/"):
                regex = True
                text = text[1:]

            if (
                not hasattr(self, "_label_search")
                or self._label_search.query != text
                or self._label_search.regex != regex
            ):
                self._label_search = self.figure.find_label(text, regex=regex)

            # LabelSearch can be `None` if no match found
            if self._label_search:
                self._label_search.prev()

    def find_select(self):
        """Find and select all matches."""
        mods = QtWidgets.QApplication.keyboardModifiers()
        text = self.searchbar.text()
        if text:
            regex = False
            if text.startswith("/"):
                regex = True
                text = text[1:]

            if (
                not hasattr(self, "_label_search")
                or self._label_search.query != text
                or self._label_search.regex != regex
            ):
                self._label_search = self.figure.find_label(text, regex=regex)

            # LabelSearch can be `None` if no match found
            if self._label_search:
                if mods & QtCore.Qt.ShiftModifier:
                    self._label_search.select_all(add=True)
                else:
                    self._label_search.select_all(add=False)

    def selected_to_clipboard(self, dataset=None):
        """Copy selected items to clipboard."""
        if self.figure.selected is not None:
            indices = self.selected_indices

            if isinstance(dataset, str):
                indices = [i for i in indices if self.figure._markers[i] == dataset]
            elif isinstance(dataset, (list, set, tuple)):
                indices = [i for i in indices if self.figure._markers[i] in dataset]

            ids = self.figure._ids[indices]
            pyperclip.copy(",".join(np.array(ids).astype(str)))

    def _on_color_column_changed(self):
        """Handle color-by column change: reset range controls then apply colors."""
        color_col = self.color_combo_box.currentText()
        is_numerical = False
        is_color_col = False
        if (
            color_col
            and color_col not in ("Default", CLUSTER_DATA_OPTION)
            and self.meta_data is not None
            and color_col in self.meta_data.columns
        ):
            series = self.meta_data[color_col]
            is_numerical = series.dtype.kind == "f"
            is_color_col = is_color_column(series)

        # Palette is not meaningful when the column already encodes colours or
        # when colouring by cluster assignments.
        palette_enabled = not is_color_col and color_col != CLUSTER_DATA_OPTION
        self.palette_combo_box.setEnabled(palette_enabled)

        if is_numerical:
            data = self.meta_data[color_col].values
            lo = float(np.nanmin(data))
            hi = float(np.nanmax(data))
            mid = (lo + hi) / 2.0
            self.color_range_slider.blockSignals(True)
            self.color_range_slider.set_data_range(lo, hi)
            self.color_range_slider.set_values(lo, mid, hi)
            self.color_range_slider.set_distribution(data)
            self.color_range_slider.blockSignals(False)

        self.color_range_slider.setVisible(is_numerical)
        if is_numerical:
            is_diverging = self.palette_combo_box.currentText() in _DIVERGING_PALETTES
            self.color_range_slider.set_diverging(is_diverging)

        self.set_colors()

    def _on_palette_changed(self):
        """Handle palette change: update centre handle visibility then apply colors."""
        if self.color_range_slider.isVisible():
            is_diverging = self.palette_combo_box.currentText() in _DIVERGING_PALETTES
            self.color_range_slider.set_diverging(is_diverging)
        self.set_colors()

    def set_colors(self, sync_to_viewer=True):
        """Set the color mode."""
        color_col = self.color_combo_box.currentText()

        if not color_col:
            return

        if color_col == "Default":
            color_col = self.figure.default_color_col

        if color_col == CLUSTER_DATA_OPTION:
            colors = self._cluster_colors
        elif is_color_column(self.meta_data[color_col]):
            colors = self.meta_data[color_col].values
        else:
            kwargs = {"palette": self.palette_combo_box.currentText()}
            if self.color_range_slider.isVisible():
                kwargs["vmin"] = self.color_range_slider.vmin
                kwargs["vmax"] = self.color_range_slider.vmax
                if self.color_range_slider.show_center:
                    kwargs["vcenter"] = self.color_range_slider.vcenter
            colors = labels_to_colors(
                self.meta_data[color_col].values,
                **kwargs,
            )

        self.figure.set_colors(colors, sync_to_viewer=sync_to_viewer)

    def _on_size_column_changed(self):
        """Handle size-by column change: reset range controls then apply sizes."""
        size_col = self.size_combo_box.currentText()
        has_range = False
        if (
            size_col
            and size_col != "Default"
            and self.meta_data is not None
            and size_col in self.meta_data.columns
        ):
            data = np.asarray(self.meta_data[size_col].values, dtype=float)
            data = data[np.isfinite(data)]
            has_range = data.size > 0 and data.min() != data.max()
            if has_range:
                lo = float(data.min())
                hi = float(data.max())
                mid = (lo + hi) / 2.0
                self.size_range_slider.blockSignals(True)
                self.size_range_slider.set_data_range(lo, hi)
                self.size_range_slider.set_values(lo, mid, hi)
                self.size_range_slider.set_distribution(data)
                self.size_range_slider.blockSignals(False)

        self.size_range_slider.setVisible(has_range)

        self.set_sizes()

    def set_sizes(self):
        """Apply the 'Size by' selection as per-point sizes."""
        size_col = self.size_combo_box.currentText()

        if not size_col:
            return

        if (
            size_col == "Default"
            or self.meta_data is None
            or size_col not in self.meta_data.columns
        ):
            self.figure.point_size = 1
            return

        kwargs = {}
        if self.size_range_slider.isVisible():
            kwargs = {
                "vmin": self.size_range_slider.vmin,
                "vcenter": self.size_range_slider.vcenter,
                "vmax": self.size_range_slider.vmax,
            }

        self.figure.point_size = _normalize_sizes(
            self.meta_data[size_col].values, **kwargs
        )

    def set_label_outlines(self):
        """Draw polygons around neurons with the same label."""
        self.figure.show_label_lines = self.label_outlines_check.isChecked()

    def _compute_fidelity_column(self, positions=None):
        """Compute neighborhood fidelity and write it to metadata.

        Returns True on success, False otherwise.
        """
        if self.figure.metadata is None:
            self.figure.show_message(
                "Metadata is required to store fidelity scores.",
                color="red",
                duration=3,
            )
            return False

        try:
            scores = self.figure.calculate_embedding_fidelity(
                k=self.fidelity_k_slider.value(),
                metric=self._fidelity_metric(),
                rank=self.fidelity_use_rank_check.isChecked(),
                positions=positions,
            )
        except Exception as e:
            logger.error(f"Fidelity computation failed: {e}")
            self.figure.show_message(
                f"Fidelity computation failed: {e}", color="red", duration=4
            )
            return False

        self.figure.metadata[FIDELITY_DATA_COLUMN] = np.asarray(scores, dtype=float)

        self.label_combo_box.blockSignals(True)
        self.update_label_combo_boxes()  # this updates label, color and size combo boxes
        self.label_combo_box.blockSignals(False)

        return True

    def _on_fidelity_compute(self):
        """Handle the fidelity 'Compute' button."""
        if not self._compute_fidelity_column():
            return

        if self.size_combo_box.currentText() != FIDELITY_DATA_COLUMN:
            self.size_combo_box.setCurrentText(
                FIDELITY_DATA_COLUMN
            )  # this triggers _on_size_column_changed
        else:
            # Force refresh (incl. the slider range) when re-computing the same column
            self._on_size_column_changed()

        self.figure.show_message(
            f"Added fidelity column '{FIDELITY_DATA_COLUMN}' to metadata.",
            color="lightgreen",
            duration=3,
        )

    def set_knn_edges(self):
        """Set the KNN line mode."""
        mode = self.fidelity_knn_combo_box.currentText()
        if mode == "Off":
            self.figure.show_knn_edges = False
        else:
            metric = self._fidelity_metric()
            self.figure.show_knn_edges = {
                "mode": "selected" if mode == "Selected only" else "all",
                "k": self.fidelity_knn_k_slider.value(),
                "metric": metric,
                "distance": metric,
            }

    def _knn_edges_source_available(self, metric):
        """Whether the figure can still draw KNN edges with `metric`."""
        check = getattr(self.figure, "_knn_edges_drawable", None)
        return bool(check(metric)) if callable(check) else False

    def _reconcile_knn_edges(self):
        """Turn off KNN edges when the active embedding lacks their source.

        Mirrors the distance-edge auto-disable in
        `update_distance_edges_controls`: switching to an embedding without the
        source the edges were drawn from would otherwise crash the figure's
        move-completion / selection re-apply (see `make_neighbour_edges`).
        """
        edges = getattr(self.figure, "show_knn_edges", False)
        if not isinstance(edges, dict) or not edges.get("mode"):
            return
        if self._knn_edges_source_available(edges.get("metric", "auto")):
            return

        # Reset the Show combo (the signal turns the figure edges off); fall
        # back to clearing the figure flag directly if it was already "Off".
        if self.fidelity_knn_combo_box.currentText() != "Off":
            self.fidelity_knn_combo_box.setCurrentText("Off")
        else:
            self.figure.show_knn_edges = False

        self.figure.show_message(
            "KNN edges turned off: source unavailable for this embedding.",
            color="orange",
            duration=3,
        )

    def _on_evaluate_labels_toggle(self):
        """React to changes in the evaluate labels checkbox."""
        if self.evaluate_labels_show_check.isChecked():
            self._save_current_color_settings()
            if self._evaluate_labels():
                self._apply_evaluate_label_color()
            else:
                self._restore_previous_color_settings()
        else:
            self._restore_previous_color_settings()

    def _save_current_color_settings(self):
        """Save the current color column and palette so we can restore them later."""
        if not hasattr(self, "_prev_evaluate_color_col"):
            self._prev_evaluate_color_col = self.color_combo_box.currentText()
            self._prev_evaluate_palette = self.palette_combo_box.currentText()

    def _update_evaluate_labels_controls(self):
        """Show or hide evaluation controls depending on selected method."""
        method_key = self.evaluate_labels_method_combo_box.currentText().lower().replace(
            " ", "_"
        )
        show_k = method_key == "neighbor_consistency"
        self.evaluate_labels_k_row.setVisible(show_k)
        if hasattr(self, "evaluate_labels_k_label"):
            self.evaluate_labels_k_label.setVisible(show_k)

    def _apply_evaluate_label_color(self):
        """Switch color settings to the shared evaluation result column and coolwarm palette."""
        self.palette_combo_box.setCurrentText("matplotlib:coolwarm")
        if self.color_combo_box.findText(EVALUATE_DATA_COLUMN) >= 0:
            self.color_combo_box.setCurrentText(EVALUATE_DATA_COLUMN)

        # Force refresh even if the color column text did not change.
        self.set_colors()

    def _evaluate_labels_data_options(self):
        """Return the available high-dimensional data options for label evaluation."""
        options = []

        positions = getattr(self.figure, "positions", None)
        if positions is not None and len(np.asarray(positions)) > 0:
            options.append("Embedding")

        if getattr(self.figure, "feats", None) is not None:
            options.append("Features")

        dists = getattr(self.figure, "dists", None)
        if dists is not None:
            if isinstance(dists, dict) and "distances" in dists:
                options.append("Precomputed distances")
            elif is_precomputed_distance_matrix(dists):
                options.append("Precomputed distances")

        return options

    def update_evaluate_labels_data_options(self):
        """Refresh the Evaluate Labels data source dropdown."""
        if not hasattr(self, "evaluate_labels_data_combo_box"):
            return

        options = self._evaluate_labels_data_options()
        current_text = self.evaluate_labels_data_combo_box.currentText()

        self.evaluate_labels_data_combo_box.blockSignals(True)
        self.evaluate_labels_data_combo_box.clear()
        if options:
            self.evaluate_labels_data_combo_box.addItems(options)
            self.evaluate_labels_data_combo_box.setEnabled(True)
            self.evaluate_labels_show_check.setEnabled(True)
            if current_text in options:
                self.evaluate_labels_data_combo_box.setCurrentText(current_text)
            else:
                self.evaluate_labels_data_combo_box.setCurrentIndex(0)
        else:
            self.evaluate_labels_data_combo_box.addItem("No high-dimensional data available")
            self.evaluate_labels_data_combo_box.setEnabled(False)
            self.evaluate_labels_show_check.setEnabled(False)
        self.evaluate_labels_data_combo_box.blockSignals(False)

    def _restore_previous_color_settings(self):
        """Restore the previous color column and palette after evaluation is turned off."""
        if hasattr(self, "_prev_evaluate_color_col"):
            self.palette_combo_box.setCurrentText(self._prev_evaluate_palette)
            self.color_combo_box.setCurrentText(self._prev_evaluate_color_col)
            del self._prev_evaluate_color_col
            del self._prev_evaluate_palette

    def _maybe_recompute_evaluation(self, _=None):
        """Recompute evaluation when labels change and sync is enabled."""
        if (
            getattr(self, "evaluate_labels_show_check", None)
            and getattr(self, "evaluate_labels_sync_check", None)
            and self.evaluate_labels_show_check.isChecked()
            and self.evaluate_labels_sync_check.isChecked()
        ):
            if self._evaluate_labels():
                self._apply_evaluate_label_color()

    def _evaluate_labels(self):
        """Compute per-sample evaluation scores and store them in metadata."""
        if self.figure.metadata is None:
            self.figure.show_message(
                "Metadata is required for label evaluation.",
                color="red",
                duration=3,
            )
            self.evaluate_labels_show_check.blockSignals(True)
            self.evaluate_labels_show_check.setChecked(False)
            self.evaluate_labels_show_check.blockSignals(False)
            return

        choice = self.evaluate_labels_data_combo_box.currentText()
        if choice == "Embedding":
            data = getattr(self.figure, "positions", None)
            is_precomputed = False
        elif choice == "Features":
            data = getattr(self.figure, "feats", None)
            is_precomputed = False
        elif choice == "Precomputed distances":
            dists = getattr(self.figure, "dists", None)
            if isinstance(dists, dict):
                data = dists.get("distances")
            else:
                data = dists
            is_precomputed = True
        else:
            data = None
            is_precomputed = False

        if data is None:
            self.figure.show_message(
                "No high-dimensional data available for evaluation.",
                color="red",
                duration=3,
            )
            self.evaluate_labels_show_check.blockSignals(True)
            self.evaluate_labels_show_check.setChecked(False)
            self.evaluate_labels_show_check.blockSignals(False)
            return

        method_text = self.evaluate_labels_method_combo_box.currentText()
        method_key = method_text.lower().replace(" ", "_")
        if method_key not in ("silhouette", "neighbor_consistency"):
            self.figure.show_message(
                f"Unknown evaluation method '{method_text}'.",
                color="red",
                duration=3,
            )
            return

        labels = np.asarray(self.labels, dtype=object)
        if labels.size < 2:
            self.figure.show_message(
                "At least 2 samples are required for per-sample evaluation.",
                color="red",
                duration=3,
            )
            return

        labels = pd.factorize(labels)[0]
        metric = self.evaluate_labels_metric_combo_box.currentText()
        k_neighbors = int(self.evaluate_labels_k_spinbox.value())

        try:
            scores = evaluate_clustering_sample(
                data,
                labels,
                method=method_key,
                is_precomputed=is_precomputed,
                metric=metric,
                k_neighbors=k_neighbors,
            )
        except Exception as exc:
            self.figure.show_message(
                f"Label evaluation failed: {exc}",
                color="red",
                duration=4,
            )
            return

        col_name = EVALUATE_DATA_COLUMN
        col_exists = col_name in self.figure.metadata.columns
        self.figure.metadata[col_name] = np.asarray(scores, dtype=float)
        self.update_label_combo_boxes()
        if not col_exists:
            self.figure.show_message(
                f"Added evaluation column '{col_name}' to metadata.",
                color="lightgreen",
                duration=3,
            )
        return True

    def set_labels(self):
        """Set the leaf labels."""
        label = self.label_combo_box.currentText()

        if not label:
            return

        if label == "Default":
            label = self.figure.default_label_col

        # Nothing to do here
        if self._current_leaf_labels != label:
            self._last_leaf_labels, self._current_leaf_labels = (
                self._current_leaf_labels,
                label,
            )

        if label == CLUSTER_DATA_OPTION:
            labels = (
                "cluster_"
                + self.meta_data[CLUSTER_DATA_COLUMN].astype(str).fillna("-1").values
            )
            labels[labels == "cluster_-1"] = "noise"
        else:
            labels = self.meta_data[label].astype(str).fillna("").values

        # For labels that were set manually by the user (e.g. via pushing annotations)
        for i, label in self.label_overrides.items():
            # Label overrides {dend index: label}
            # We need to translate into original indices
            labels[i] = label

        # Add counts - e.g. "CB12345(10)"
        if self.label_count_check.isChecked():
            counts = pd.Series(labels).value_counts().to_dict()  # dict is much faster
            labels = [
                f"{label}({counts[label]})" if counts[label] > 1 else label
                for label in labels
            ]
        self.figure.labels = labels

        # Update searchbar completer
        if not hasattr(self, "_label_models"):
            self._label_models = {}
        if (label, self.label_count_check.isChecked()) not in self._label_models:
            self._label_models[(label, self.label_count_check.isChecked())] = (
                QtCore.QStringListModel(np.unique(labels).tolist())
            )

        self.searchbar_completer.setModel(
            self._label_models[(label, self.label_count_check.isChecked())]
        )

        # Update label lines
        if self.figure.show_label_lines:
            self.figure.make_label_lines()

    def switch_labels(self):
        """Switch between current and last labels."""
        if hasattr(self, "_last_leaf_labels"):
            self.label_combo_box.setCurrentText(self._last_leaf_labels)
            self.set_labels()
            self.figure.show_message(
                f"Labels: {self._current_leaf_labels}", color="lightgreen", duration=2
            )

    def close(self):
        """Close the controls."""
        super().close()

    def ngl_open(self):
        if not hasattr(self.figure, "ngl_viewer"):
            raise ValueError("Figure has no neuroglancer viewer")
        scene = self.figure.ngl_viewer.neuroglancer_scene(
            group_by=self.ngl_split_combo_box.currentText().lower(),
            use_colors=self.ngl_use_colors.isChecked(),
        )
        scene.open()

    def ngl_copy(self):
        if not hasattr(self.figure, "ngl_viewer"):
            raise ValueError("Figure has no neuroglancer viewer")
        scene = self.figure.ngl_viewer.neuroglancer_scene(
            group_by=self.ngl_split_combo_box.currentText().lower(),
            use_colors=self.ngl_use_colors.isChecked(),
        )
        scene.to_clipboard()
        self.figure.show_message(
            "Link copied to clipboard", color="lightgreen", duration=2
        )

    def calculate_embeddings(self):
        """Re-calculate embeddings and move points to their new positions."""
        dists = self._selected_embedding_data()
        if dists is None:
            return

        # KNN graph: feed neighbors straight into UMAP/t-SNE (bypassing the
        # feature-vector preprocessing, which would densify the sparse input).
        if is_knn_graph(dists):
            self._calculate_embeddings_from_knn(dists)
            return

        dists = self._apply_embedding_feature_partition_filter(dists)
        if (
            hasattr(dists, "shape")
            and len(dists.shape) > 1
            and dists.shape[1] == 0
        ):
            self.figure.show_message(
                "Select at least one parcellation to compute embeddings.",
                color="red",
                duration=3,
            )
            return

        is_precomputed = is_precomputed_distance_matrix(dists)
        metric = (
            "precomputed"
            if is_precomputed
            else self.umap_feature_metric_combo_box.currentText().strip().lower()
        )
        method = self.umap_method_combo_box.currentText()
        random_state = (
            int(self.umap_random_seed.text()) if self.umap_random_seed.text() else None
        )
        fit = make_embedding_estimator(
            method,
            metric=metric,
            is_precomputed=is_precomputed,
            random_state=random_state,
            umap_n_neighbors=self.umap_n_neighbors_slider.value(),
            umap_min_dist=self.umap_min_dist_slider.value(),
            umap_spread=self.umap_spread_slider.value(),
            umap_set_op_mix_ratio=self.umap_set_op_mix_ratio_spinbox.value(),
            umap_densmap=self.umap_densmap_check.isChecked(),
            umap_dens_lambda=self.umap_dens_lambda_spinbox.value(),
            mds_n_init=self.mds_n_init_slider.value(),
            mds_max_iter=self.mds_max_iter_slider.value(),
            mds_eps=self.mds_eps_slider.value(),
            tsne_perplexity=self.tsne_perplexity_slider.value(),
            tsne_learning_rate=self.tsne_learning_rate_slider.value(),
            tsne_n_iter=self.tsne_n_iter_slider.value(),
        )

        pca_components = (
            self.pca_n_components_slider.value()
            if ((not is_precomputed) and self.pca_check.isChecked())
            else None
        )
        if pca_components is not None:
            print(
                f" Using PCA to reduce {dists.shape} observation vector to {pca_components} components",
                flush=True,
            )

        dists = prepare_embedding_input(
            dists.values if isinstance(dists, pd.DataFrame) else dists,
            is_precomputed=is_precomputed,
            method=method,
            metric=metric,
            rebalance_mode=self.umap_feature_rebalance_combo_box.currentText(),
            pca_n_components=pca_components,
            random_state=random_state,
        )

        with warnings.catch_warnings(action="ignore"):
            xy = fit.fit_transform(dists)

        self._apply_recomputed_positions(xy)

    def _calculate_embeddings_from_knn(self, graph):
        """Recompute the embedding directly from a precomputed KNN graph."""
        method = self.umap_method_combo_box.currentText()
        random_state = (
            int(self.umap_random_seed.text()) if self.umap_random_seed.text() else None
        )
        try:
            estimator, fit_input = make_knn_embedding_estimator(
                method,
                knn=graph,
                random_state=random_state,
                umap_n_neighbors=self.umap_n_neighbors_slider.value(),
                umap_min_dist=self.umap_min_dist_slider.value(),
                umap_spread=self.umap_spread_slider.value(),
                umap_set_op_mix_ratio=self.umap_set_op_mix_ratio_spinbox.value(),
                umap_densmap=self.umap_densmap_check.isChecked(),
                umap_dens_lambda=self.umap_dens_lambda_spinbox.value(),
                tsne_perplexity=self.tsne_perplexity_slider.value(),
                tsne_learning_rate=self.tsne_learning_rate_slider.value(),
                tsne_n_iter=self.tsne_n_iter_slider.value(),
            )
            with warnings.catch_warnings(action="ignore"):
                xy = estimator.fit_transform(fit_input)
        except Exception as e:
            logger.error(f"KNN embedding failed: {e}")
            self.figure.show_message(
                f"Embedding error: {e}", color="red", duration=4
            )
            return

        self._apply_recomputed_positions(xy)

    def _apply_recomputed_positions(self, xy):
        """Normalize, persist and animate to freshly recomputed positions."""
        # Fully disconnected neurons (no KNN links in this subset) come back from
        # UMAP/t-SNE at NaN; a single non-finite row would otherwise turn the
        # whole normalized layout into NaN and make every point vanish. Relocate
        # them to the edge of the layout instead.
        xy, relocated = sanitize_embedding(xy)
        if relocated.any():
            n = int(relocated.sum())
            self.figure.show_message(
                f"{n} neuron(s) had no neighbors in this subset and were placed "
                f"at the edge of the layout.",
                color="orange",
                duration=4,
            )

        # Normalize into the shared frame so the recomputed layout stays in view
        # and matches the scale of the other embeddings.
        xy = self.figure.normalize_to_frame(xy)

        # Recomputing supersedes the active embedding: persist the new positions
        # into its entry so switching away and back keeps this layout.
        self.figure.update_active_embedding_positions(xy)

        # This moves points to their new positions
        self.figure.move_points(xy)

        # If point sizes are currently driven by the fidelity column, recompute it
        # against the new (post-animation) positions and refresh sizes.
        if self.size_combo_box.currentText() == FIDELITY_DATA_COLUMN:
            if self._compute_fidelity_column(positions=xy):
                self._on_size_column_changed()

        # Keep the owning window's project data pointing at the new positions.
        self._sync_window_data_active()

    def _sync_method_settings_widgets(self):
        """Show the settings widget for the current method (no recompute)."""
        if not hasattr(self, "umap_settings_widget"):
            return
        method = self.umap_method_combo_box.currentText()
        if method == "UMAP":
            self.umap_settings_widget.show()
            self.mds_settings_widget.hide()
            self.tsne_settings_widget.hide()
            self.umap_button.setText("Run UMAP")
        elif method == "MDS":
            self.umap_settings_widget.hide()
            self.mds_settings_widget.show()
            self.tsne_settings_widget.hide()
            self.umap_button.setText("Run MDS")
        elif method == "t-SNE":
            self.umap_settings_widget.hide()
            self.mds_settings_widget.hide()
            self.tsne_settings_widget.show()
            self.umap_button.setText("Run t-SNE")
        else:
            self.umap_settings_widget.hide()
            self.mds_settings_widget.hide()
            self.tsne_settings_widget.hide()
            self.umap_button.setText(f"Run {method}")

    def update_embedding_settings(self):
        """Update the embedding settings based on the selected method."""
        self._sync_method_settings_widgets()
        self.calculate_embeddings_maybe()

    def update_searchbar_completer(self):
        """Update the searchbar completer."""
        if not hasattr(self, "_label_models"):
            self._label_models = {}

        label = self.label_combo_box.currentText()
        labels = self.figure.labels
        logger.debug(
            f"Updating searchbar completer for {label} with {len(labels)} labels"
        )
        if (label, self.label_count_check.isChecked()) not in self._label_models:
            self._label_models[(label, self.label_count_check.isChecked())] = (
                QtCore.QStringListModel(np.unique(labels).tolist())
            )

        self.searchbar_completer.setModel(
            self._label_models[(label, self.label_count_check.isChecked())]
        )

    def calculate_embeddings_maybe(self):
        """Recalculate embeddings if the auto-run checkbox is checked."""
        if self.umap_auto_run.isChecked():
            self.calculate_embeddings()


class QHLine(QtWidgets.QFrame):
    def __init__(self):
        super(QHLine, self).__init__()
        self.setFrameShape(QtWidgets.QFrame.HLine)
        self.setFrameShadow(QtWidgets.QFrame.Sunken)


class QVLine(QtWidgets.QFrame):
    def __init__(self):
        super(QVLine, self).__init__()
        self.setFrameShape(QtWidgets.QFrame.VLine)
        self.setFrameShadow(QtWidgets.QFrame.Sunken)
