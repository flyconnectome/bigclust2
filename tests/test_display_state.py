"""Tests for carrying display settings into a selection's new view.

Exercises ``ScatterControls.capture_display_state`` / ``apply_display_state``
(used by ``on_open_selection_in_new_view``) without building the full controls
panel or the wgpu ``MainWindow``: the methods are invoked against a lightweight
stub that owns only the real Qt widgets they touch. Qt widgets run fine under
the offscreen QPA platform; the wgpu canvas does not, so it is avoided here.
"""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pandas as pd
import pytest
from PySide6 import QtWidgets

from bigclust2.gui.controls.scatter_control import (
    CLUSTER_DATA_OPTION,
    ScatterControls,
    _MultiHandleSlider,
)


@pytest.fixture(scope="module")
def qapp():
    yield QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


class _Figure:
    """Stand-in for the ScatterFigure attributes capture/apply read or write."""

    def __init__(self):
        self.font_size = 5.0
        self.point_scale = 1.0
        self.label_vis_limit = 100
        self.show_label_lines = False


class _ControlsStub:
    """Owns the real widgets capture/apply use, with the methods bound to it.

    The capture/apply/helper methods are taken straight from ``ScatterControls``
    so the production code is what runs; only the surrounding panel is faked.
    The settings-tab knobs are wired to write the figure exactly as
    ``ScatterControls.__init__`` does.
    """

    capture_display_state = ScatterControls.capture_display_state
    apply_display_state = ScatterControls.apply_display_state
    _set_combo_text = ScatterControls._set_combo_text
    _apply_slider_state = ScatterControls._apply_slider_state

    def __init__(self, columns, df):
        self.figure = _Figure()
        self.meta_data = df
        self.applied = []

        self.label_combo_box = QtWidgets.QComboBox()
        self.color_combo_box = QtWidgets.QComboBox()
        self.size_combo_box = QtWidgets.QComboBox()
        for c in columns:
            self.label_combo_box.addItem(c)
            self.color_combo_box.addItem(c)
            self.size_combo_box.addItem(c)

        self.palette_combo_box = QtWidgets.QComboBox()
        for p in ("matplotlib:viridis", "seaborn:tab10", "matplotlib:coolwarm"):
            self.palette_combo_box.addItem(p)

        self.label_count_check = QtWidgets.QCheckBox()
        self.label_outlines_check = QtWidgets.QCheckBox()

        self.color_range_slider = _MultiHandleSlider()
        self.size_range_slider = _MultiHandleSlider()
        self.color_range_slider.setVisible(False)
        self.size_range_slider.setVisible(False)

        self.font_size_slider = QtWidgets.QDoubleSpinBox()
        self.font_size_slider.setRange(0.01, 200)
        self.font_size_slider.valueChanged.connect(
            lambda x: setattr(self.figure, "font_size", x)
        )
        self.point_scale_spinbox = QtWidgets.QDoubleSpinBox()
        self.point_scale_spinbox.setRange(0.01, 200)
        self.point_scale_spinbox.valueChanged.connect(
            lambda x: setattr(self.figure, "point_scale", x)
        )
        self.max_label_vis_slider = QtWidgets.QSlider()
        self.max_label_vis_slider.setRange(1, 5000)
        self.max_label_vis_slider.valueChanged.connect(
            lambda x: setattr(self.figure, "label_vis_limit", int(x))
        )

    # Public appliers — record that they ran (the real ones push to the figure).
    def set_labels(self):
        self.applied.append("labels")

    def set_colors(self):
        self.applied.append("colors")

    def set_sizes(self):
        self.applied.append("sizes")


@pytest.fixture
def df():
    return pd.DataFrame(
        {
            "label": ["a", "b", "c", "d"],
            "score": [0.1, 0.5, 0.9, 0.3],
            "size": [1.0, 2.0, 3.0, 4.0],
        }
    )


def _make_source(df):
    src = _ControlsStub(["Default", "label", "score", "size"], df)
    src.label_combo_box.setCurrentText("label")
    src.label_count_check.setChecked(True)
    src.label_outlines_check.setChecked(True)
    src.color_combo_box.setCurrentText("score")
    src.palette_combo_box.setCurrentText("matplotlib:coolwarm")
    src.color_range_slider.setVisible(True)
    src.color_range_slider.set_data_range(0.0, 1.0)
    src.color_range_slider.set_values(0.2, 0.5, 0.8)
    src.color_range_slider.set_diverging(True)
    src.size_combo_box.setCurrentText("size")
    src.size_range_slider.setVisible(True)
    src.size_range_slider.set_data_range(1.0, 4.0)
    src.size_range_slider.set_values(1.5, 2.5, 3.5)
    src.figure.font_size = 18.0
    src.figure.point_scale = 2.5
    src.figure.label_vis_limit = 1234
    return src


def test_capture_snapshot(qapp, df):
    state = _make_source(df).capture_display_state()

    assert state["label_col"] == "label"
    assert state["label_counts"] is True
    assert state["label_outlines"] is True
    assert state["color_col"] == "score"
    assert state["palette"] == "matplotlib:coolwarm"
    assert state["color_range"] == {
        "lo": 0.0,
        "hi": 1.0,
        "vmin": 0.2,
        "vcenter": 0.5,
        "vmax": 0.8,
        "show_center": True,
    }
    assert state["size_col"] == "size"
    assert state["size_range"]["lo"] == 1.0 and state["size_range"]["hi"] == 4.0
    assert (state["font_size"], state["point_scale"], state["label_vis_limit"]) == (
        18.0,
        2.5,
        1234,
    )


def test_apply_round_trip(qapp, df):
    state = _make_source(df).capture_display_state()

    tgt = _ControlsStub(["Default", "label", "score", "size"], df)
    tgt.apply_display_state(state)

    assert tgt.label_combo_box.currentText() == "label"
    assert tgt.label_count_check.isChecked() is True
    assert tgt.label_outlines_check.isChecked() is True
    assert tgt.figure.show_label_lines is True

    assert tgt.color_combo_box.currentText() == "score"
    assert tgt.palette_combo_box.currentText() == "matplotlib:coolwarm"
    assert tgt.color_range_slider.isVisible() is True
    assert (tgt.color_range_slider.lo, tgt.color_range_slider.hi) == (0.0, 1.0)
    assert (tgt.color_range_slider.vmin, tgt.color_range_slider.vmax) == (0.2, 0.8)
    assert tgt.color_range_slider.show_center is True

    assert tgt.size_combo_box.currentText() == "size"
    assert tgt.size_range_slider.isVisible() is True
    assert (tgt.size_range_slider.lo, tgt.size_range_slider.hi) == (1.0, 4.0)

    assert tgt.figure.font_size == 18.0
    assert tgt.figure.point_scale == 2.5
    assert tgt.figure.label_vis_limit == 1234

    # All three appliers run, regardless of whether a combo index changed.
    assert tgt.applied == ["labels", "colors", "sizes"]


def test_apply_skips_missing_option(qapp, df):
    """An option absent from the target (e.g. clusters) is skipped, not forced."""
    src = _ControlsStub(["Default", "label", CLUSTER_DATA_OPTION], df)
    src.color_combo_box.setCurrentText(CLUSTER_DATA_OPTION)
    state = src.capture_display_state()
    assert state["color_col"] == CLUSTER_DATA_OPTION

    # Target has no cluster option: it must fall back to its default selection.
    tgt = _ControlsStub(["Default", "label", "score"], df)
    tgt.apply_display_state(state)

    assert tgt.color_combo_box.currentText() == "Default"
    assert tgt.palette_combo_box.isEnabled() is True
    assert tgt.applied == ["labels", "colors", "sizes"]


def test_apply_hidden_slider_stays_hidden(qapp, df):
    """A categorical source (no range slider) leaves the target slider hidden."""
    src = _ControlsStub(["Default", "label"], df)
    src.color_combo_box.setCurrentText("label")  # categorical -> slider hidden
    state = src.capture_display_state()
    assert state["color_range"] is None

    tgt = _ControlsStub(["Default", "label"], df)
    tgt.color_range_slider.setVisible(True)  # ensure apply actively hides it
    tgt.apply_display_state(state)
    assert tgt.color_range_slider.isVisible() is False


# --------------------------------------------------------------------------- #
# labels_unique
# --------------------------------------------------------------------------- #


class _LabelsStub:
    """Just enough of ScatterControls to exercise the `labels_unique` property."""

    labels_unique = ScatterControls.labels_unique

    def __init__(self, labels):
        self.figure = type("_Fig", (), {"labels": labels})()


def test_labels_unique_is_empty_before_the_figure_has_points():
    """`ScatterControls` is built before `set_points`, so labels can be None.

    The searchbar completer is constructed from this property during window
    setup; returning None here used to crash view creation.
    """
    assert len(_LabelsStub(None).labels_unique) == 0


def test_labels_unique_is_sorted_and_deduplicated():
    stub = _LabelsStub(np.array(["b", "a", "b", "c"], dtype=object))
    assert list(stub.labels_unique) == ["a", "b", "c"]
