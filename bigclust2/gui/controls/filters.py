"""Reusable metadata-filter widgets, shared by the scatter controls' Scope
panel and the Find Nearest widget.

Three self-contained Qt widgets, factored out of ``scatter_control`` so several
places can build the same column/operator/value → boolean-mask filter UI without
depending on the whole controls module:

* :class:`MultiHandleSlider` – a 2/3-handle range slider with an optional
  distribution histogram.
* :class:`FilterableComboBox` – a combo box whose popup has a filter field.
* :class:`ScopeFilterRow` – a single dtype-aware filter row (checkboxes for
  categoricals, a range slider for numeric, a substring/regex field otherwise)
  that turns into a boolean mask over a DataFrame via :meth:`ScopeFilterRow.mask`.
"""

import re

import numpy as np

from PySide6 import QtWidgets, QtCore, QtGui


class MultiHandleSlider(QtWidgets.QWidget):
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
            label = MultiHandleSlider._fmt(val)
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


class FilterableComboBox(QtWidgets.QComboBox):
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


class ScopeFilterRow(QtWidgets.QWidget):
    """A single scope filter: column picker plus a dtype-specific editor.

    Numeric columns get a range slider with editable min/max fields,
    low-cardinality categorical columns get checkboxes (plus an "(empty)" box
    for missing/blank values) and high-cardinality ones a substring/regex
    filter field with an any/empty/non-empty selector.

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
        self._empty_check = None
        self._empty_combo = None

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

        self.column_combo = FilterableComboBox()
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
        self._empty_check = None
        self._empty_combo = None

    @staticmethod
    def _empty_mask(series):
        """Rows where the field is missing or blank (whitespace counts as blank)."""
        return (
            series.isna().to_numpy()
            | (series.astype(str).str.strip() == "").to_numpy()
        )

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
                # Empty/missing values get their own checkbox instead of
                # counting towards the cardinality of the real values
                empty = self._empty_mask(series)
                uniques = series[~empty].astype(str).unique()
                if len(uniques) <= self.MAX_CHECKBOX_VALUES:
                    self._build_checkbox_editor(layout, uniques, bool(empty.any()))
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
        self._range_slider = MultiHandleSlider()
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

    def _build_checkbox_editor(self, layout, uniques, has_empty=False):
        self._editor_kind = "checks"
        for value in sorted(uniques):
            check = QtWidgets.QCheckBox(value)
            check.setChecked(True)
            check.stateChanged.connect(lambda *_: self.changed.emit())
            layout.addWidget(check)
            self._value_checks[value] = check

        if has_empty:
            # Kept out of `_value_checks` so it can't clash with a literal
            # "(empty)" value in the column
            self._empty_check = QtWidgets.QCheckBox("(empty)")
            self._empty_check.setChecked(True)
            self._empty_check.setToolTip(
                "Include rows where this field is missing or blank.\n"
                "Uncheck everything else to see only those rows."
            )
            self._empty_check.setStyleSheet("font-style: italic;")
            self._empty_check.stateChanged.connect(lambda *_: self.changed.emit())
            layout.addWidget(self._empty_check)

    def _build_text_editor(self, layout):
        self._editor_kind = "text"

        row = QtWidgets.QWidget()
        row_layout = QtWidgets.QHBoxLayout()
        row_layout.setContentsMargins(0, 0, 0, 0)
        row_layout.setSpacing(4)
        row.setLayout(row_layout)

        self._text_edit = QtWidgets.QLineEdit()
        self._text_edit.setPlaceholderText("substring filter…")
        self._text_edit.setToolTip(
            "Case-insensitive substring filter; start with '/' for a regex pattern"
        )
        self._text_edit.setClearButtonEnabled(True)
        self._text_edit.textChanged.connect(self._on_text_changed)
        row_layout.addWidget(self._text_edit)

        self._empty_combo = QtWidgets.QComboBox()
        self._empty_combo.addItems(["any", "empty", "non-empty"])
        self._empty_combo.setToolTip(
            "Restrict to rows where this field is empty (missing or blank) "
            "or non-empty"
        )
        self._empty_combo.currentIndexChanged.connect(self._on_empty_changed)
        row_layout.addWidget(self._empty_combo)

        layout.addWidget(row)

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

    def _on_empty_changed(self):
        # A substring only makes sense for rows that have a value at all
        self._text_edit.setEnabled(self._empty_combo.currentText() != "empty")
        self.changed.emit()

    def _text_pattern(self):
        """Return (pattern, is_regex); a leading '/' marks a regex pattern."""
        text = self._text_edit.text()
        if text.startswith("/"):
            return text[1:], True
        return text, False

    def _text_mask(self, series):
        pattern, is_regex = self._text_pattern()
        # `astype(str)` turns missing values into "nan"/"None", which would
        # otherwise match patterns like "na" - exclude them explicitly
        return ~self._empty_mask(series) & (
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
            empty = self._empty_mask(df[col])
            mask = ~empty & df[col].astype(str).isin(
                {v for v, c in self._value_checks.items() if c.isChecked()}
            ).to_numpy()
            if self._empty_check is not None and self._empty_check.isChecked():
                mask |= empty
            return mask
        elif self._editor_kind == "text":
            mode = self._empty_combo.currentText()
            empty = self._empty_mask(df[col])
            if mode == "empty":
                return empty
            mask = ~empty if mode == "non-empty" else np.ones(len(df), dtype=bool)
            if self._text_edit.text():
                try:
                    mask = mask & self._text_mask(df[col])
                except re.error:
                    pass
            return mask
        return np.ones(len(df), dtype=bool)
