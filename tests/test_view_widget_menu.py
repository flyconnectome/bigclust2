"""Tests for the data-dependent widget entries in the View menu.

Two things are covered: the ``can_open_*`` predicates that decide whether a
widget has the data it needs, and the menu section built from them - a plain
action per widget with one embedding, a submenu of embeddings with several.

The menu methods are bound to a lightweight stub owning a real ``QMenu`` and
real ``QAction``s rather than built on the real ``MainWindow``: Qt widgets run
fine under the offscreen QPA platform, the wgpu-backed main window does not.
"""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pandas as pd
import pytest
from PySide6 import QtWidgets
from PySide6.QtGui import QAction

from bigclust2.gui.core import (
    VIEW_WIDGET_SPECS,
    MainWindow,
    can_open_connectivity_table,
    can_open_distances_table,
    can_open_feature_comparison,
    can_open_find_nearest,
    can_open_meta_explorer,
)


@pytest.fixture(scope="module")
def qapp():
    yield QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


META = pd.DataFrame({"id": ["a", "b", "c"], "type": ["x", "y", "z"]})
FEATURES = pd.DataFrame(np.arange(9, dtype=float).reshape(3, 3), index=META["id"])
DISTANCES = pd.DataFrame(
    np.zeros((3, 3)), index=META["id"], columns=META["id"]
)


def _entry(name, *, features=None, distances=None, knn=None):
    """An embedding entry as `ScatterFigure.embedding_entries` holds them."""
    return {
        "name": name,
        "embedding": np.zeros((3, 2), dtype=np.float32),
        "features": features,
        "distances": distances,
        "knn": knn,
    }


# --------------------------------------------------------------------------- #
# predicates
# --------------------------------------------------------------------------- #


def test_connectivity_needs_a_non_empty_feature_table():
    assert can_open_connectivity_table({"features": FEATURES})
    assert not can_open_connectivity_table({"features": None})
    assert not can_open_connectivity_table({"features": pd.DataFrame()})
    assert not can_open_connectivity_table(None)


def test_distances_accepts_a_matrix_or_numeric_features():
    assert can_open_distances_table({"distances": DISTANCES})
    assert can_open_distances_table({"features": FEATURES})
    # A KNN graph stands in for the matrix but cannot fill a heatmap.
    assert not can_open_distances_table({"knn": object()})
    # Non-square distances are not a heatmap either.
    assert not can_open_distances_table({"distances": DISTANCES.iloc[:2]})
    assert not can_open_distances_table({"features": pd.DataFrame({"a": ["x"]})})


def test_feature_comparison_needs_features_and_meta():
    assert can_open_feature_comparison({"features": FEATURES, "meta": META})
    assert not can_open_feature_comparison({"features": FEATURES, "meta": None})
    assert not can_open_feature_comparison({"features": None, "meta": META})


def test_meta_explorer_needs_meta():
    assert can_open_meta_explorer({"meta": META})
    assert not can_open_meta_explorer({"meta": pd.DataFrame()})


def test_find_nearest_needs_meta_and_a_similarity_source():
    assert can_open_find_nearest({"meta": META, "embeddings": np.zeros((3, 2))})
    assert can_open_find_nearest({"meta": META, "features": FEATURES})
    assert not can_open_find_nearest({"meta": META})
    assert not can_open_find_nearest({"meta": pd.DataFrame(), "features": FEATURES})


# --------------------------------------------------------------------------- #
# menu layout
# --------------------------------------------------------------------------- #


class _FigureStub:
    def __init__(self, entries, active=0):
        self.embedding_entries = entries
        self.active_embedding = active if entries else None


class _ViewStub:
    def __init__(self, entries, active=0, meta=META):
        self.fig_scatter = _FigureStub(entries, active)
        data = {"meta": meta, "embedding_entries": entries}
        if entries:
            data.update(
                {
                    "embeddings": entries[active]["embedding"],
                    "features": entries[active]["features"],
                    "distances": entries[active]["distances"],
                    "knn": entries[active].get("knn"),
                    "active_embedding": active,
                }
            )
        self._data = data


class _WindowStub(QtWidgets.QMainWindow):
    """The View-menu machinery, bound to a real menu and fake views."""

    _embedding_entries = MainWindow._embedding_entries
    _embedding_names = MainWindow._embedding_names
    _resolve_embedding = MainWindow._resolve_embedding
    _embedding_data = MainWindow._embedding_data
    _embedding_key = MainWindow._embedding_key
    _refresh_view_widget_entries = MainWindow._refresh_view_widget_entries
    _rebuild_view_widget_entries = MainWindow._rebuild_view_widget_entries

    def __init__(self):
        super().__init__()
        self.view = None
        self.opened = []
        for spec in VIEW_WIDGET_SPECS:
            setattr(self, spec["action"], QAction(spec["title"], self))
        self._view_widget_menu = QtWidgets.QMenu("View", self)
        self._view_widget_menu.addAction(QAction("Command Palette…", self))
        self._view_widget_anchor = self._view_widget_menu.addSeparator()
        self._view_widget_menu.addAction(QAction("Center", self))
        self._view_widget_submenus = {}
        self._view_widget_signature = None

    def current_view(self):
        return self.view

    # The openers only have to record what they were asked for.
    def show_connectivity_table(self, embedding=None):
        self.opened.append(("connectivity", embedding))

    def show_distances_table(self, embedding=None):
        self.opened.append(("distances", embedding))

    def show_feature_comparison(self, embedding=None):
        self.opened.append(("features", embedding))

    def show_meta_explorer(self):
        self.opened.append(("meta", None))


@pytest.fixture
def window(qapp):
    win = _WindowStub()
    yield win
    win.close()
    win.deleteLater()


def _entries(window):
    """The widget entries installed in the menu, in menu order.

    Everything between the leading Command Palette item and the anchor the
    section is inserted before.
    """
    actions = window._view_widget_menu.actions()
    return actions[1 : actions.index(window._view_widget_anchor)]


def _plain_actions(window):
    return [getattr(window, spec["action"]) for spec in VIEW_WIDGET_SPECS]


def _submenu(window, spec_action):
    return window._view_widget_submenus[spec_action][0]


def test_single_embedding_keeps_plain_actions(window):
    window.view = _ViewStub([_entry("emb", features=FEATURES)])
    window._refresh_view_widget_entries()

    assert window._view_widget_submenus == {}
    # Inserted as a block before the anchor, in spec order.
    assert _entries(window) == _plain_actions(window)


def test_no_embeddings_keeps_plain_actions(window):
    window.view = _ViewStub([])
    window._refresh_view_widget_entries()

    assert window._view_widget_submenus == {}
    assert window.meta_explorer_action.isEnabled()
    # Nothing but meta data -> only the explorer can be opened.
    assert not window.feature_comparison_action.isEnabled()


def test_multiple_embeddings_produce_submenus(window):
    window.view = _ViewStub(
        [
            _entry("morphology", features=FEATURES, distances=DISTANCES),
            _entry("connectivity", features=FEATURES),
        ]
    )
    window._refresh_view_widget_entries()

    assert set(window._view_widget_submenus) == {
        "connectivity_table_action",
        "distances_table_action",
        "feature_comparison_action",
    }
    for spec_action in window._view_widget_submenus:
        submenu = _submenu(window, spec_action)
        assert [a.text() for a in submenu.actions()] == ["morphology", "connectivity"]
        assert submenu.menuAction().isEnabled()

    # Meta data is shared by all embeddings, so its entry stays a plain action,
    # and it keeps its place at the end of the section.
    assert _entries(window)[-1] is window.meta_explorer_action


def test_embeddings_without_the_data_are_listed_but_disabled(window):
    window.view = _ViewStub(
        [_entry("with", features=FEATURES), _entry("without")]
    )
    window._refresh_view_widget_entries()

    submenu = _submenu(window, "feature_comparison_action")
    with_it, without_it = submenu.actions()
    assert with_it.isEnabled() and with_it.text() == "with"
    assert not without_it.isEnabled()
    assert without_it.text() == "without (no features)"
    # One embedding has features -> the entry as a whole stays available.
    assert submenu.menuAction().isEnabled()


def test_entry_disabled_when_no_embedding_has_the_data(window):
    window.view = _ViewStub([_entry("a"), _entry("b")])
    window._refresh_view_widget_entries()

    submenu = _submenu(window, "connectivity_table_action")
    assert not submenu.menuAction().isEnabled()
    assert not any(a.isEnabled() for a in submenu.actions())


def test_distance_heatmap_follows_matrix_or_features(window):
    window.view = _ViewStub(
        [
            _entry("matrix", distances=DISTANCES),
            _entry("features", features=FEATURES),
            _entry("knn", knn=object()),
        ]
    )
    window._refresh_view_widget_entries()

    submenu = _submenu(window, "distances_table_action")
    matrix, features, knn = submenu.actions()
    assert matrix.isEnabled()
    assert features.isEnabled()
    assert not knn.isEnabled()
    assert knn.text() == "knn (no distance matrix or features)"


def test_submenu_item_opens_that_embedding(window):
    window.view = _ViewStub(
        [_entry("first", features=FEATURES), _entry("second", features=FEATURES)]
    )
    window._refresh_view_widget_entries()

    _submenu(window, "feature_comparison_action").actions()[1].trigger()
    assert window.opened == [("features", 1)]


def test_plain_action_tracks_the_active_embedding(window):
    """The action the keyboard shortcut fires targets the active embedding."""
    window.view = _ViewStub(
        [_entry("with", features=FEATURES), _entry("without")], active=1
    )
    window._refresh_view_widget_entries()

    assert not window.feature_comparison_action.isEnabled()
    # ... while the submenu still offers the embedding that does have features.
    assert _submenu(window, "feature_comparison_action").menuAction().isEnabled()

    window.view.fig_scatter.active_embedding = 0
    window._refresh_view_widget_entries()
    assert window.feature_comparison_action.isEnabled()


def test_layout_follows_a_change_of_embeddings(window):
    window.view = _ViewStub(
        [_entry("a", features=FEATURES), _entry("b", features=FEATURES)]
    )
    window._refresh_view_widget_entries()
    assert window._view_widget_submenus

    # Switching to a project with a single embedding restores plain entries and
    # leaves no stale submenu behind in the menu.
    window.view = _ViewStub([_entry("only", features=FEATURES)])
    window._refresh_view_widget_entries()

    assert window._view_widget_submenus == {}
    assert _entries(window) == _plain_actions(window)


def test_relayout_only_happens_when_the_embeddings_change(window):
    window.view = _ViewStub(
        [_entry("a", features=FEATURES), _entry("b", features=FEATURES)]
    )
    window._refresh_view_widget_entries()
    submenu = _submenu(window, "feature_comparison_action")

    # Same embeddings, changed data: states refresh, the menu is not rebuilt.
    window.view._data["meta"] = pd.DataFrame()
    for entry in window.view.fig_scatter.embedding_entries:
        entry["features"] = None
    window._refresh_view_widget_entries()

    assert _submenu(window, "feature_comparison_action") is submenu
    assert not submenu.menuAction().isEnabled()


def test_embedding_data_reads_the_requested_embedding(window):
    other = pd.DataFrame(np.ones((3, 2)), index=META["id"])
    window.view = _ViewStub(
        [_entry("a", features=FEATURES), _entry("b", features=other)]
    )

    assert window._embedding_data(window.view, 1)["features"] is other
    # None -> the active embedding; meta is view-level and always carried.
    assert window._embedding_data(window.view)["features"] is FEATURES
    assert window._embedding_data(window.view, 1)["meta"] is META


def test_embedding_key_names_only_when_there_is_a_choice(window):
    single = _ViewStub([_entry("only", features=FEATURES)])
    assert window._embedding_key(single) == (0, None)

    multi = _ViewStub([_entry("a", features=FEATURES), _entry("b")])
    assert window._embedding_key(multi) == (0, "a")
    assert window._embedding_key(multi, 1) == (1, "b")


def test_an_index_out_of_range_falls_back_to_the_active_embedding(window):
    """Key and data must agree so a stray index cannot fork the widget cache."""
    view = _ViewStub([_entry("only", features=FEATURES)])
    assert window._embedding_key(view, 3) == (0, None)
    assert window._embedding_data(view, 3)["features"] is FEATURES

    empty = _ViewStub([])
    assert window._embedding_key(empty) == ("__default__", None)
