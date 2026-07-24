"""Dialog showing the command line(s) that re-open the current view.

Takes plain, pre-built command strings (see
:func:`bigclust2.utils.build_launch_command`) rather than the live loader, so it
imports only PySide6 and can be constructed/tested in isolation under offscreen
Qt. Each command is shown in a read-only, selectable field with a copy button.
"""

from PySide6.QtCore import QTimer
from PySide6.QtGui import QFontDatabase
from PySide6.QtWidgets import (
    QApplication,
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
)


class CommandDialog(QDialog):
    """Show one or more shell commands, each with a copy-to-clipboard button."""

    def __init__(self, parent=None, *, commands=None, intro=None):
        """Create the dialog.

        Parameters
        ----------
        commands :  list of (label, command) tuples
                    Each becomes a labelled row with its own copy button.
        intro :     str, optional
                    Short explanatory line shown above the commands.
        """
        super().__init__(parent)
        self.setWindowTitle("Command")
        self.setMinimumWidth(560)

        layout = QVBoxLayout(self)

        commands = list(commands or [])
        if not commands:
            layout.addWidget(QLabel("No project loaded."))
            return

        if intro:
            intro_label = QLabel(intro)
            intro_label.setWordWrap(True)
            layout.addWidget(intro_label)

        mono = QFontDatabase.systemFont(QFontDatabase.FixedFont)

        # Keep references so the copy handler can restore each button's caption.
        self._rows = []
        for label, command in commands:
            layout.addWidget(QLabel(label))

            row = QHBoxLayout()
            field = QLineEdit(command)
            field.setReadOnly(True)
            field.setFont(mono)
            field.setCursorPosition(0)
            row.addWidget(field)

            button = QPushButton("Copy")
            button.setToolTip("Copy this command to the clipboard")
            button.clicked.connect(
                lambda _checked=False, c=command, b=button: self._copy(c, b)
            )
            row.addWidget(button)

            layout.addLayout(row)
            self._rows.append((field, button))

        # Close button.
        close_row = QHBoxLayout()
        close_row.addStretch(1)
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.accept)
        close_row.addWidget(close_btn)
        layout.addLayout(close_row)

    def _copy(self, command, button):
        """Copy ``command`` to the clipboard and briefly confirm on ``button``."""
        QApplication.clipboard().setText(command)
        button.setText("Copied!")
        QTimer.singleShot(1200, lambda: button.setText("Copy") if button else None)
