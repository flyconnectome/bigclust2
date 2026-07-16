"""Dialog for configuring the TAB property toggle.

The toggle lets the user flip the whole scatter view between two states (A and
B). Each state assigns a metadata column to one or more properties (Color by,
Labels). Pressing TAB over the plot swaps every enabled property to the other
state at once.
"""

from PySide6 import QtWidgets


def _select(combo, text):
    """Select ``text`` in ``combo`` if it is offered; otherwise leave as-is."""
    if not text:
        return
    idx = combo.findText(text)
    if idx >= 0:
        combo.setCurrentIndex(idx)


def _select_different(combo, avoid, options):
    """Select the first option that differs from ``avoid`` (fallback: ``avoid``)."""
    for i, opt in enumerate(options):
        if opt != avoid:
            combo.setCurrentIndex(i)
            return
    _select(combo, avoid)


class TabToggleDialog(QtWidgets.QDialog):
    """Configure which properties TAB flips and between which two values.

    Parameters
    ----------
    prop_specs :    list of (key, title, options, current_value)
                    One entry per toggle-able property. ``options`` are the
                    available column names (taken from the live combo so the
                    choices stay identical); ``current_value`` seeds the
                    defaults when the property has no saved config yet.
    current :       dict | None
                    Existing config ``{key: (col_a, col_b) | None}`` to restore.
    """

    def __init__(self, prop_specs, current=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Configure Tab toggle")
        self.setModal(True)
        current = current or {}

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(16, 14, 16, 12)
        layout.setSpacing(12)

        intro = QtWidgets.QLabel(
            "Pressing <b>Tab</b> over the plot flips between state <b>A</b> and "
            "state <b>B</b>. Enable a property and pick the column for each state."
        )
        intro.setWordWrap(True)
        layout.addWidget(intro)

        form = QtWidgets.QFormLayout()
        form.setVerticalSpacing(8)
        form.setHorizontalSpacing(10)
        layout.addLayout(form)

        self._rows = {}  # key -> (check, combo_a, combo_b)
        for key, title, options, current_value in prop_specs:
            check = QtWidgets.QCheckBox(title)
            combo_a = QtWidgets.QComboBox()
            combo_b = QtWidgets.QComboBox()
            combo_a.addItems(options)
            combo_b.addItems(options)

            cfg = current.get(key)
            if cfg:
                check.setChecked(True)
                _select(combo_a, cfg[0])
                _select(combo_b, cfg[1])
            else:
                check.setChecked(False)
                _select(combo_a, current_value)
                _select_different(combo_b, current_value, options)

            field = QtWidgets.QWidget()
            fl = QtWidgets.QHBoxLayout(field)
            fl.setContentsMargins(0, 0, 0, 0)
            fl.setSpacing(6)
            fl.addWidget(QtWidgets.QLabel("A"))
            fl.addWidget(combo_a, 1)
            fl.addWidget(QtWidgets.QLabel("B"))
            fl.addWidget(combo_b, 1)
            form.addRow(check, field)

            def _sync(_state, ca=combo_a, cb=combo_b, ck=check):
                on = ck.isChecked()
                ca.setEnabled(on)
                cb.setEnabled(on)

            check.stateChanged.connect(_sync)
            check.stateChanged.connect(self._update_ok)
            _sync(None)

            self._rows[key] = (check, combo_a, combo_b)

        self._buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel
        )
        self._buttons.accepted.connect(self.accept)
        self._buttons.rejected.connect(self.reject)
        layout.addWidget(self._buttons)

        self.setMinimumWidth(440)
        self._update_ok()

    def _update_ok(self, *args):
        """OK is only meaningful when at least one property is enabled."""
        any_on = any(check.isChecked() for check, _, _ in self._rows.values())
        self._buttons.button(QtWidgets.QDialogButtonBox.Ok).setEnabled(any_on)

    def selected_config(self):
        """Return ``{key: (col_a, col_b) | None}`` for the chosen settings."""
        cfg = {}
        for key, (check, combo_a, combo_b) in self._rows.items():
            if check.isChecked():
                cfg[key] = (combo_a.currentText(), combo_b.currentText())
            else:
                cfg[key] = None
        return cfg
