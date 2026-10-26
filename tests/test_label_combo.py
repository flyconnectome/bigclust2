"""Tests for the label column -> figure labels -> completer path.

``ScatterControls.set_labels`` used to build the "(count)" suffixes with a
per-point f-string loop and then feed ``np.unique`` over every point's label
into the completer model - both O(N) in Python for a value set that is tiny by
comparison. It now suffixes once per distinct label and hands the known
distinct set to ``update_searchbar_completer``.

Driven against a lightweight stub owning only the Qt widgets these methods
touch (same approach as ``test_display_state.py``): the full controls panel
needs a Qt-backed wgpu canvas, which does not exist under the offscreen
platform.
"""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402
from PySide6 import QtWidgets  # noqa: E402

from bigclust2.gui.controls.scatter_control import ScatterControls  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    yield QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


class _Figure:
    """Records what `set_labels` pushes onto the figure."""

    default_label_col = "label"
    show_label_lines = False

    def __init__(self):
        self.labels = None

    def make_label_lines(self):  # pragma: no cover - not reached in these tests
        raise AssertionError("label lines are disabled in this stub")


class _Controls:
    set_labels = ScatterControls.set_labels
    update_searchbar_completer = ScatterControls.update_searchbar_completer
    labels_unique = ScatterControls.labels_unique

    def __init__(self, meta, column, counts=False):
        self.meta_data = meta
        self.figure = _Figure()
        self.label_overrides = {}

        self.label_combo_box = QtWidgets.QComboBox()
        self.label_combo_box.addItems(list(meta.columns))
        self.label_combo_box.setCurrentIndex(self.label_combo_box.findText(column))

        self.label_count_check = QtWidgets.QCheckBox()
        self.label_count_check.setChecked(counts)

        self.searchbar_completer = QtWidgets.QCompleter([])

    @property
    def completer_entries(self):
        model = self.searchbar_completer.model()
        return [model.index(i, 0).data() for i in range(model.rowCount())]


@pytest.fixture
def meta():
    # "b" appears twice, "a" and "c" once -> only "b" gets a count suffix.
    return pd.DataFrame(
        {
            "label": ["b", "a", "b", "c"],
            "other": ["x", "x", "y", "y"],
        }
    )


def test_labels_are_pushed_to_the_figure(qapp, meta):
    ctrl = _Controls(meta, "label")
    ctrl.set_labels()
    assert list(ctrl.figure.labels) == ["b", "a", "b", "c"]


def test_counts_are_appended_only_to_repeated_labels(qapp, meta):
    ctrl = _Controls(meta, "label", counts=True)
    ctrl.set_labels()
    assert list(ctrl.figure.labels) == ["b(2)", "a", "b(2)", "c"]


def test_completer_offers_the_distinct_labels_sorted(qapp, meta):
    ctrl = _Controls(meta, "label")
    ctrl.set_labels()
    assert ctrl.completer_entries == ["a", "b", "c"]


def test_completer_offers_the_suffixed_labels(qapp, meta):
    ctrl = _Controls(meta, "label", counts=True)
    ctrl.set_labels()
    assert ctrl.completer_entries == ["a", "b(2)", "c"]


def test_completer_models_are_cached_per_column_and_count_setting(qapp, meta):
    ctrl = _Controls(meta, "label")
    ctrl.set_labels()
    first = ctrl.searchbar_completer.model()

    ctrl.set_labels()
    assert ctrl.searchbar_completer.model() is first, "same key must reuse the model"

    ctrl.label_count_check.setChecked(True)
    ctrl.set_labels()
    assert ctrl.searchbar_completer.model() is not first, "counts are part of the key"

    ctrl.label_count_check.setChecked(False)
    ctrl.set_labels()
    assert ctrl.searchbar_completer.model() is first, "and the old model comes back"


def test_label_overrides_do_not_poison_the_cache_key(qapp, meta):
    """Regression: the completer cache was once keyed by the wrong value.

    ``set_labels`` applies overrides in a loop whose variable used to be called
    ``label``, shadowing the selected column name; the completer block that
    followed was inlined here and keyed ``_label_models`` by that shadowed
    value. It is now keyed inside ``update_searchbar_completer``, which reads
    the combo box itself - so the key is right regardless of the loop.
    """
    ctrl = _Controls(meta, "label")
    ctrl.label_overrides = {0: "OVERRIDDEN"}
    ctrl.set_labels()

    assert list(ctrl.figure.labels)[0] == "OVERRIDDEN"
    assert set(ctrl._label_models) == {("label", False)}


def test_switching_column_switches_labels(qapp, meta):
    ctrl = _Controls(meta, "label")
    ctrl.set_labels()
    ctrl.label_combo_box.setCurrentIndex(ctrl.label_combo_box.findText("other"))
    ctrl.set_labels()

    assert list(ctrl.figure.labels) == ["x", "x", "y", "y"]
    assert ctrl.completer_entries == ["x", "y"]


def test_null_labels_are_stringified(qapp):
    """Documents a long-standing quirk, so a "fix" has to be deliberate.

    ``set_labels`` does ``.astype(str).fillna("")``; ``astype(str)`` has already
    turned nulls into ``"None"``/``"nan"`` by the time ``fillna`` runs, so the
    fill never fires. Swapping the order would render these as empty labels
    instead - a visible change, hence not done as part of the vectorisation.
    """
    frame = pd.DataFrame({"label": ["a", None, "a"]})
    ctrl = _Controls(frame, "label")
    ctrl.set_labels()
    assert list(ctrl.figure.labels) == ["a", "None", "a"]

    frame = pd.DataFrame({"label": ["a", np.nan, "a"]})
    ctrl = _Controls(frame, "label")
    ctrl.set_labels()
    assert list(ctrl.figure.labels) == ["a", "nan", "a"]


def test_categorical_column_is_handled(qapp):
    """Real projects carry `category` dtype columns."""
    frame = pd.DataFrame({"label": pd.Series(["b", "a", "b"], dtype="category")})
    ctrl = _Controls(frame, "label", counts=True)
    ctrl.set_labels()
    assert list(ctrl.figure.labels) == ["b(2)", "a", "b(2)"]
