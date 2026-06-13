"""Compact dialog summarising the current project's metadata.

Renders a derived ``Summary`` section plus the raw nested ``info`` file as a
two-column ``QTreeWidget`` with native collapsible expand/collapse. The dialog
takes plain pre-extracted data (a ``summary`` dict and the ``info`` dict) rather
than the live loader, so it imports neither pandas nor the data loader nor the
wgpu canvas and can be constructed/tested in isolation under offscreen Qt.
"""

from PySide6.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QLabel,
    QTreeWidget,
    QTreeWidgetItem,
)


class ProjectDetailsDialog(QDialog):
    """Dialog showing a project's summary stats and its nested ``info`` file."""

    def __init__(self, parent=None, *, summary=None, info=None):
        super().__init__(parent)
        self.setWindowTitle("Project Details")
        self.resize(520, 480)  # compact

        layout = QVBoxLayout(self)

        if not summary and not info:
            layout.addWidget(QLabel("No project loaded."))
            return

        tree = QTreeWidget(self)
        tree.setColumnCount(2)
        tree.setHeaderLabels(["Key", "Value"])
        if summary:
            self._add_node(tree, "Summary", summary)
        if info:
            self._add_node(tree, "Info file", info)
        tree.expandToDepth(1)  # top sections plus their immediate keys
        tree.resizeColumnToContents(0)
        layout.addWidget(tree)

        self.tree = tree

    def _add_node(self, parent, key, value):
        """Add a tree row for ``key: value``, recursing into dicts/lists."""
        if isinstance(value, dict):
            node = self._item(parent, str(key), "" if value else "{}")
            for k, v in value.items():
                self._add_node(node, k, v)
            return node
        if isinstance(value, (list, tuple)):
            node = self._item(parent, str(key), f"[{len(value)} items]")
            for i, item in enumerate(value):
                label = str(i)
                # Name-bearing entries (e.g. the multi-embedding list) read
                # without having to expand them.
                if isinstance(item, dict) and item.get("name") is not None:
                    label = f"{i}: {item['name']}"
                self._add_node(node, label, item)
            return node
        return self._item(parent, str(key), "" if value is None else str(value))

    @staticmethod
    def _item(parent, col0, col1):
        return QTreeWidgetItem(parent, [col0, col1])
