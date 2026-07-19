"""VS Code-style command palette.

The palette does not maintain its own registry of commands: the menu bar
already is one. :func:`collect_commands` walks ``QMainWindow.menuBar()`` at open
time, so every action added to a menu shows up here for free, complete with its
shortcut and its enabled/checked state (which the window already keeps correct
via ``_update_view_actions`` / ``_sync_actions_to_view``).

Two details of this codebase's menus need care when walking them, both handled
in :func:`collect_commands`:

* Several menus fill themselves lazily on ``aboutToShow``, so a cold walk would
  see them empty or stale. We emit that signal before descending.
* The "Open Recent" menu uses ``QWidgetAction`` for its rich entries; those have
  no meaningful ``text()`` and would render as blank rows, so they are skipped.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Callable

from PySide6.QtCore import Qt, QEvent, QSize, QRect
from PySide6.QtGui import (
    QColor,
    QFont,
    QFontMetrics,
    QKeySequence,
    QPainter,
    QPalette,
)
from PySide6.QtWidgets import (
    QAbstractItemView,
    QDialog,
    QFrame,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QStyle,
    QStyledItemDelegate,
    QVBoxLayout,
    QWidgetAction,
)

from ...fuzzy import match_positions, rank

#: QSettings key holding the most-recently-used command IDs.
RECENT_COMMANDS_KEY = "commandPalette/recentCommands/v1"

#: How many recently-used commands to remember.
MAX_RECENT = 20

#: Separator between menu levels in a command's breadcrumb.
BREADCRUMB_SEP = " › "

#: ``objectName`` of the action that opens the palette. Skipped when collecting,
#: since offering "open the palette" from inside the palette is just a no-op.
COMMAND_PALETTE_ACTION_NAME = "commandPaletteAction"

_ROLE_COMMAND = Qt.UserRole + 1
_ROLE_QUERY = Qt.UserRole + 2


@dataclass
class Command:
    """A single runnable entry in the palette.

    ``id`` is the stable identity used for the recently-used list. ``QAction``
    objects are anonymous, so it is derived from the menu path (falling back to
    ``objectName()`` when one is set).
    """

    id: str
    title: str
    breadcrumb: str = ""
    shortcut: str = ""
    enabled: bool = True
    checkable: bool = False
    checked: bool = False
    callback: Callable[[], None] | None = None
    aliases: list[str] = field(default_factory=list)

    @property
    def haystack(self) -> str:
        """The text the fuzzy matcher searches."""
        parts = [self.breadcrumb, self.title, *self.aliases]
        return " ".join(p for p in parts if p)

    def run(self):
        if self.callback is not None:
            self.callback()


def _action_shortcut_text(action) -> str:
    seq = action.shortcut()
    if seq is None or seq.isEmpty():
        return ""
    return seq.toString(QKeySequence.NativeText)


def _walk_menu(menu, trail, out, seen):
    """Recursively collect leaf actions from ``menu`` into ``out``."""
    if menu is None or id(menu) in seen:
        # Guard against a menu reachable by two paths (and against cycles).
        return
    seen.add(id(menu))

    # Menus that populate themselves on demand look empty until asked. Emitting
    # the signal here is what makes "Open Recent", "Grow/Shrink Options", the
    # hover-column list and the Window menu searchable.
    try:
        menu.aboutToShow.emit()
    except (RuntimeError, AttributeError):
        pass

    for action in menu.actions():
        if action.isSeparator():
            continue
        # Rich entries (e.g. the recent-projects rows) carry a custom widget and
        # no usable text; showing them would produce blank palette rows.
        if isinstance(action, QWidgetAction):
            continue
        if action.objectName() == COMMAND_PALETTE_ACTION_NAME:
            continue

        submenu = action.menu()
        title = action.text().replace("&", "").strip()
        if not title:
            continue

        if submenu is not None:
            _walk_menu(submenu, trail + [title], out, seen)
            continue

        path = trail + [title]
        out.append(
            Command(
                id=action.objectName() or "/".join(path),
                title=title,
                breadcrumb=BREADCRUMB_SEP.join(trail),
                shortcut=_action_shortcut_text(action),
                enabled=action.isEnabled(),
                checkable=action.isCheckable(),
                checked=action.isChecked(),
                callback=action.trigger,
            )
        )


def command_from_button(button, breadcrumb, *, key, title=None, aliases=None):
    """Wrap a plain ``QPushButton`` as a palette command.

    For controls that live in a panel rather than the menu bar. Because
    :func:`collect_commands` runs on every open, the label and enabled state are
    read live - so a button whose text switches between "Run UMAP" and "Run
    t-SNE" shows up under whichever it currently is.

    Args:
        button: The button to invoke. ``None`` yields ``None``, so callers can
            pass widgets that may not have been built yet.
        breadcrumb: Where it lives, shown dimmed in the row (e.g. "Clustering").
        key: Stable identifier for the recently-used list. Deliberately separate
            from the title, which may change at runtime.
        title: Overrides the button's own text. Use for buttons labelled "Run"
            or "Export", which say nothing out of context.
        aliases: Extra search terms. Worth adding for buttons whose text varies,
            so "umap" still finds the button while it reads "Run t-SNE".
    """
    if button is None:
        return None
    try:
        label = title or button.text().replace("&", "").strip()
        if not label:
            return None
        return Command(
            id=f"{breadcrumb}/{key}",
            title=label,
            breadcrumb=breadcrumb,
            enabled=button.isEnabled(),
            callback=button.click,
            aliases=list(aliases or []),
        )
    except RuntimeError:
        # Underlying C++ widget already deleted.
        return None


def _provider_commands(obj) -> list[Command]:
    """Ask ``obj`` for extra commands via the ``palette_commands`` hook.

    Any object - a view, a control panel - can expose commands that are not in
    the menu bar by defining ``palette_commands()`` returning ``Command``s.
    Failures are swallowed: a broken provider must not take the palette down.
    """
    hook = getattr(obj, "palette_commands", None)
    if not callable(hook):
        return []
    try:
        return [c for c in (hook() or []) if isinstance(c, Command)]
    except Exception:
        return []


def _embedding_commands(window) -> list[Command]:
    """`Switch Embedding: <name>` entries.

    The embedding switcher lives in the status bar rather than the menu bar, so
    it is not reachable by the menu walk and has to be added explicitly.
    """
    view = window.current_view()
    fig = getattr(view, "fig_scatter", None) if view is not None else None
    entries = getattr(fig, "embedding_entries", None) or []
    if len(entries) < 2:
        return []
    active = getattr(fig, "active_embedding", None)

    out = []
    for idx, entry in enumerate(entries):
        name = str(entry.get("name", f"#{idx + 1}"))
        out.append(
            Command(
                id=f"embedding/{name}",
                title=f"Switch Embedding: {name}",
                breadcrumb="Embedding",
                checkable=True,
                checked=idx == active,
                callback=lambda i=idx, f=fig: f.switch_embedding(i, animate=True),
                aliases=["embedding"],
            )
        )
    return out


def _view_commands(window) -> list[Command]:
    """`Go to View: <title>` entries for the window's tabs."""
    tabs = getattr(window, "_view_tabs", None)
    if tabs is None or tabs.count() < 2:
        return []
    current = tabs.currentIndex()
    out = []
    for i in range(tabs.count()):
        title = tabs.tabText(i).replace("&", "").strip() or f"View {i + 1}"
        out.append(
            Command(
                id=f"view/{i}",
                title=f"Go to View: {title}",
                breadcrumb="Window",
                checkable=True,
                checked=i == current,
                callback=lambda idx=i, t=tabs: t.setCurrentIndex(idx),
                aliases=["tab", "view"],
            )
        )
    return out


def collect_commands(window) -> list[Command]:
    """Build the full command list for ``window``.

    Sources, in order: the menu bar, the status-bar embedding switcher, the view
    tabs, and anything the window or its current view exposes through a
    ``palette_commands()`` hook (see :func:`command_from_button`) - which is how
    control-panel buttons such as "Run UMAP" get in.
    """
    commands: list[Command] = []
    menu_bar = window.menuBar() if hasattr(window, "menuBar") else None
    if menu_bar is not None:
        seen: set[int] = set()
        for action in menu_bar.actions():
            submenu = action.menu()
            title = action.text().replace("&", "").strip()
            if submenu is None or not title:
                continue
            _walk_menu(submenu, [title], commands, seen)

    try:
        commands.extend(_embedding_commands(window))
        commands.extend(_view_commands(window))
    except Exception:
        # Extra sources are best-effort; never let them break the palette.
        pass

    commands.extend(_provider_commands(window))
    view = window.current_view() if hasattr(window, "current_view") else None
    if view is not None:
        commands.extend(_provider_commands(view))

    return commands


def load_recent(settings) -> list[str]:
    """Read the recently-used command IDs from ``QSettings``."""
    if settings is None:
        return []
    try:
        raw = settings.value(RECENT_COMMANDS_KEY, "[]")
        recent = json.loads(raw) if isinstance(raw, str) else []
        return [str(x) for x in recent] if isinstance(recent, list) else []
    except (ValueError, TypeError):
        return []


def remember_recent(settings, command_id: str):
    """Push ``command_id`` to the front of the recently-used list."""
    if settings is None or not command_id:
        return
    recent = [x for x in load_recent(settings) if x != command_id]
    recent.insert(0, command_id)
    try:
        settings.setValue(RECENT_COMMANDS_KEY, json.dumps(recent[:MAX_RECENT]))
    except (ValueError, TypeError):
        pass


#: Score added to the most-recently-used command, decaying linearly to zero
#: across the rest of the recent list. Sized to flip near-ties (a matched
#: character is worth 16) without letting a stale favourite outrank a clearly
#: better match.
RECENCY_BONUS = 12.0


def order_commands(commands, query, recent_ids) -> list[Command]:
    """Filter by ``query`` and order for display.

    With no query the recently-used commands lead, then everything else in menu
    order. With a query, fuzzy score decides, nudged by a recency bonus.
    Disabled commands always sink to the bottom but are still listed, so it is
    visible that a command exists and is merely unavailable right now.
    """
    rank_of = {cid: i for i, cid in enumerate(recent_ids)}

    if not query.strip():
        ordered = sorted(
            enumerate(commands),
            key=lambda t: (
                not t[1].enabled,
                rank_of.get(t[1].id, len(rank_of)),
                t[0],
            ),
        )
        return [c for _, c in ordered]

    n_recent = len(recent_ids)

    def _recency_bonus(cmd):
        pos = rank_of.get(cmd.id)
        if pos is None or n_recent == 0:
            return 0.0
        return RECENCY_BONUS * (1.0 - pos / n_recent)

    matched = rank(query, commands, key=lambda c: c.haystack)
    matched.sort(key=lambda t: (not t[0].enabled, -(t[1] + _recency_bonus(t[0]))))
    return [c for c, _ in matched]


def fit_text(fm: QFontMetrics, text: str, avail: int, mode=Qt.ElideRight) -> str:
    """Return ``text``, elided only if it genuinely overflows ``avail``.

    Calling ``elidedText`` with a width equal to the text's own measured advance
    can still elide on rounding, which clipped short breadcrumbs ("View" ->
    "...ew") and silently disabled match highlighting on titles. Checking the
    advance first keeps text intact whenever it actually fits.
    """
    if avail <= 0:
        return ""
    if fm.horizontalAdvance(text) <= avail:
        return text
    return fm.elidedText(text, mode, avail)


class _CommandDelegate(QStyledItemDelegate):
    """Draws: title (with matched characters emphasised), breadcrumb, shortcut."""

    _PAD_X = 10
    _PAD_Y = 6
    _GAP = 8

    def sizeHint(self, option, index):
        fm = QFontMetrics(option.font)
        return QSize(320, fm.height() + 2 * self._PAD_Y + 2)

    def paint(self, painter: QPainter, option, index):
        cmd: Command = index.data(_ROLE_COMMAND)
        query: str = index.data(_ROLE_QUERY) or ""
        if cmd is None:
            super().paint(painter, option, index)
            return

        painter.save()
        selected = bool(option.state & QStyle.State_Selected)

        palette: QPalette = option.palette
        if selected:
            painter.fillRect(option.rect, palette.highlight())
            base_color = palette.highlightedText().color()
        else:
            base_color = palette.text().color()

        if not cmd.enabled:
            base_color = QColor(base_color)
            base_color.setAlpha(110)

        rect: QRect = option.rect.adjusted(self._PAD_X, self._PAD_Y, -self._PAD_X, -self._PAD_Y)
        fm = QFontMetrics(option.font)

        # Shortcut, right-aligned.
        if cmd.shortcut:
            sc_font = QFont(option.font)
            sc_font.setPointSizeF(max(8.0, option.font.pointSizeF() - 1))
            sc_fm = QFontMetrics(sc_font)
            sc_w = sc_fm.horizontalAdvance(cmd.shortcut)
            sc_rect = QRect(rect.right() - sc_w, rect.top(), sc_w, rect.height())
            painter.setFont(sc_font)
            sc_color = QColor(base_color)
            sc_color.setAlpha(140 if cmd.enabled else 90)
            painter.setPen(sc_color)
            painter.drawText(sc_rect, Qt.AlignRight | Qt.AlignVCenter, cmd.shortcut)
            rect.setRight(sc_rect.left() - self._GAP)

        # Breadcrumb, right-aligned before the shortcut.
        if cmd.breadcrumb:
            bc_font = QFont(option.font)
            bc_font.setPointSizeF(max(8.0, option.font.pointSizeF() - 1))
            bc_fm = QFontMetrics(bc_font)
            # Breadcrumbs may take up to half the row before being elided.
            cap = max(0, rect.width() // 2)
            bc_text = fit_text(bc_fm, cmd.breadcrumb, cap, Qt.ElideLeft)
            bc_w = bc_fm.horizontalAdvance(bc_text)
            if bc_w > 0:
                bc_rect = QRect(rect.right() - bc_w, rect.top(), bc_w, rect.height())
                painter.setFont(bc_font)
                bc_color = QColor(base_color)
                bc_color.setAlpha(110 if cmd.enabled else 70)
                painter.setPen(bc_color)
                painter.drawText(bc_rect, Qt.AlignRight | Qt.AlignVCenter, bc_text)
                rect.setRight(bc_rect.left() - self._GAP)

        # Title, with a check glyph for checkable commands and bolded matches.
        prefix = ""
        if cmd.checkable:
            prefix = "✓ " if cmd.checked else "   "

        painter.setFont(option.font)
        x = rect.left()
        y_flags = Qt.AlignLeft | Qt.AlignVCenter

        if prefix:
            pw = fm.horizontalAdvance("✓ ")  # fixed width so titles stay aligned
            painter.setPen(base_color)
            painter.drawText(QRect(x, rect.top(), pw, rect.height()), y_flags, prefix)
            x += pw

        avail = max(0, rect.right() - x)
        title = fit_text(fm, cmd.title, avail)
        highlight = set(match_positions(query, cmd.title)) if query.strip() else set()

        if not highlight or title != cmd.title:
            # No query, or the title got elided (positions would be off) - plain.
            painter.setPen(base_color)
            painter.drawText(QRect(x, rect.top(), max(0, avail), rect.height()), y_flags, title)
        else:
            bold = QFont(option.font)
            bold.setBold(True)
            bold_fm = QFontMetrics(bold)
            for i, ch in enumerate(title):
                is_hit = i in highlight
                painter.setFont(bold if is_hit else option.font)
                painter.setPen(base_color)
                cw = (bold_fm if is_hit else fm).horizontalAdvance(ch)
                if x + cw > rect.right():
                    break
                painter.drawText(QRect(x, rect.top(), cw, rect.height()), y_flags, ch)
                x += cw

        painter.restore()


class CommandPalette(QDialog):
    """Frameless overlay listing every command, filtered as you type.

    Deliberately a **top-level** dialog rather than a child overlay inside the
    view. The wgpu canvases force native window handles into the widget tree; a
    text field parented under one becomes the platform's first responder, and
    when that subwindow goes away key delivery to the whole app can break. Being
    top-level keeps this input out of the canvas hierarchy entirely. Callers
    should still restore canvas focus after the dialog closes.
    """

    def __init__(self, commands, parent=None, recent_ids=None):
        super().__init__(parent)
        self._commands = list(commands)
        self._recent_ids = list(recent_ids or [])
        self.chosen: Command | None = None

        self.setWindowFlags(Qt.Dialog | Qt.FramelessWindowHint)
        self.setModal(True)
        self._build_ui()
        self._refresh("")

    def _build_ui(self):
        self.setMinimumWidth(560)

        frame = QFrame(self)
        frame.setObjectName("paletteFrame")
        frame.setFrameShape(QFrame.StyledPanel)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(frame)

        layout = QVBoxLayout(frame)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        self.search = QLineEdit(frame)
        self.search.setPlaceholderText("Type a command…")
        self.search.setClearButtonEnabled(True)
        self.search.textChanged.connect(self._refresh)
        self.search.installEventFilter(self)
        layout.addWidget(self.search)

        self.list = QListWidget(frame)
        self.list.setUniformItemSizes(True)
        self.list.setSelectionMode(QAbstractItemView.SingleSelection)
        self.list.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.list.setItemDelegate(_CommandDelegate(self.list))
        self.list.itemActivated.connect(self._on_activated)
        self.list.itemClicked.connect(self._on_activated)
        layout.addWidget(self.list)

        self.status = QLabel("", frame)
        self.status.setStyleSheet("color: palette(mid); font-size: 11px;")
        layout.addWidget(self.status)

        self.search.setFocus()

    def _refresh(self, query):
        ordered = order_commands(self._commands, query, self._recent_ids)
        self.list.clear()
        for cmd in ordered:
            item = QListWidgetItem()
            item.setData(_ROLE_COMMAND, cmd)
            item.setData(_ROLE_QUERY, query)
            if not cmd.enabled:
                item.setFlags(item.flags() & ~Qt.ItemIsEnabled)
            self.list.addItem(item)

        first = self._first_enabled_row()
        if first is not None:
            self.list.setCurrentRow(first)

        total = len(ordered)
        n_enabled = sum(1 for c in ordered if c.enabled)
        if total == 0:
            self.status.setText("No matching commands")
        elif n_enabled == total:
            self.status.setText(f"{total} commands")
        else:
            self.status.setText(
                f"{n_enabled} available · {total - n_enabled} unavailable here"
            )

    def _first_enabled_row(self):
        for row in range(self.list.count()):
            cmd = self.list.item(row).data(_ROLE_COMMAND)
            if cmd is not None and cmd.enabled:
                return row
        return None

    def _step(self, delta):
        """Move the selection by ``delta``, skipping disabled rows and wrapping."""
        count = self.list.count()
        if count == 0:
            return
        row = self.list.currentRow()
        for _ in range(count):
            row = (row + delta) % count
            cmd = self.list.item(row).data(_ROLE_COMMAND)
            if cmd is not None and cmd.enabled:
                self.list.setCurrentRow(row)
                return

    def eventFilter(self, obj, event):
        if obj is self.search and event.type() == QEvent.KeyPress:
            key = event.key()
            if key in (Qt.Key_Down, Qt.Key_Up):
                self._step(1 if key == Qt.Key_Down else -1)
                return True
            if key in (Qt.Key_PageDown, Qt.Key_PageUp):
                self._step(10 if key == Qt.Key_PageDown else -10)
                return True
            if key in (Qt.Key_Return, Qt.Key_Enter):
                item = self.list.currentItem()
                if item is not None:
                    self._on_activated(item)
                return True
        return super().eventFilter(obj, event)

    def _on_activated(self, item):
        cmd = item.data(_ROLE_COMMAND) if item is not None else None
        if cmd is None or not cmd.enabled:
            return
        self.chosen = cmd
        self.accept()

    def center_on_parent(self):
        """Position near the top of the parent window, like an editor palette."""
        parent = self.parentWidget()
        self.adjustSize()
        if parent is None:
            return
        geo = parent.geometry()
        width = min(max(560, geo.width() // 2), max(560, geo.width() - 80))
        self.resize(width, self.sizeHint().height())
        x = geo.x() + (geo.width() - self.width()) // 2
        y = geo.y() + max(40, geo.height() // 8)
        self.move(x, y)
