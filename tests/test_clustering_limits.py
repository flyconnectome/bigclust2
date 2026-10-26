"""Tests for gating analysis tools that cannot handle the point count.

Drawing was made to scale; these tools cannot be, because they read the data
rather than the screen. Most of them merely get slow, but Agglomerative
clustering builds an O(N^2) distance matrix - past a few tens of thousands of
points it does not run slowly, it exhausts the machine. On an embedding of
tens of millions, picking it from the combo is the difference between a result
and a forced reboot.

That distinction is the thing these tests pin: a limit that is `blocking` is
refused outright, because there is no informed consent to give when the
machine goes down either way; everything else is the user's call and is asked,
with cancel as the default.
"""

from __future__ import annotations

import re
from pathlib import Path
from types import SimpleNamespace

import pytest
from PySide6 import QtWidgets

from bigclust2.clusters import (
    CLUSTERING_LIMITS,
    LABEL_EVALUATION_LIMITS,
    clustering_limit,
    clustering_too_big,
    label_evaluation_too_big,
)
from bigclust2.embeddings import (
    FEATURE_COMPARISON_LIMIT,
    FIDELITY_LIMIT,
    HEATMAP_LIMIT,
    KNN_EDGE_LIMIT,
    feature_comparison_too_big,
    fidelity_too_big,
    heatmap_too_big,
    knn_edges_too_big,
)
from bigclust2.gui.controls.scatter_control import (
    CLUSTER_DATA_EMBEDDING_OPTION,
    ScatterControls,
)


HUGE = 71_000_000


# --------------------------------------------------------------------------- #
# The policy
# --------------------------------------------------------------------------- #


def test_every_method_the_combo_offers_has_been_measured():
    """A new method must be measured, not left to inherit another's ceiling."""
    offered = set(ScatterControls._CLUSTER_METHODS)
    assert offered <= set(CLUSTERING_LIMITS)


def test_an_unmeasured_method_is_not_capped():
    assert clustering_limit("Leiden") is None
    assert clustering_too_big("Leiden", HUGE) is None


def test_a_linear_method_is_not_capped():
    assert clustering_too_big("K-Means", HUGE) is None


def test_the_message_states_the_facts():
    too_big = clustering_too_big("Agglomerative", HUGE)

    assert "71,000,000" in str(too_big)
    assert f"{clustering_limit('Agglomerative'):,}" in str(too_big)
    assert "square" in str(too_big), "it should say *why*, not just that it cannot"


#: Every gate, as (name, predicate, limit). Keeps the boundary and the
#: blocking policy pinned for all of them in one place rather than per site.
GATES = [
    ("agglomerative", lambda n: clustering_too_big("Agglomerative", n), 25_000),
    ("hdbscan", lambda n: clustering_too_big("HDBSCAN", n), 100_000),
    ("spectral", lambda n: clustering_too_big("Spectral", n), 50_000),
    (
        "silhouette",
        lambda n: label_evaluation_too_big("silhouette", n),
        LABEL_EVALUATION_LIMITS["silhouette"].max_points,
    ),
    (
        "neighbour-consistency",
        lambda n: label_evaluation_too_big("neighbor_consistency", n),
        LABEL_EVALUATION_LIMITS["neighbor_consistency"].max_points,
    ),
    ("fidelity", fidelity_too_big, FIDELITY_LIMIT.max_points),
    ("knn-edges", knn_edges_too_big, KNN_EDGE_LIMIT.max_points),
    ("heatmap", heatmap_too_big, HEATMAP_LIMIT.max_points),
    ("features", feature_comparison_too_big, FEATURE_COMPARISON_LIMIT.max_points),
]


@pytest.mark.parametrize("check, limit", [g[1:] for g in GATES], ids=[g[0] for g in GATES])
def test_a_gate_bites_exactly_one_past_its_cap(check, limit):
    assert check(limit) is None
    assert check(limit + 1) is not None


@pytest.mark.parametrize("check", [g[1] for g in GATES], ids=[g[0] for g in GATES])
def test_only_memory_exhaustion_blocks(check):
    """Slow is the user's call; out of memory is not a choice anyone can make."""
    too_big = check(HUGE)
    assert too_big is not None
    assert too_big.blocking == (too_big.operation == "Agglomerative clustering")


def test_label_evaluation_caps_the_two_methods_differently(qapp):
    """One compares every pair, the other searches neighbours - not the same.

    A single cap was wrong about whichever method it was not measured on, and
    told the user something false about it.
    """
    pairwise = LABEL_EVALUATION_LIMITS["silhouette"].max_points
    assert label_evaluation_too_big("silhouette", pairwise + 1) is not None
    assert label_evaluation_too_big("neighbor_consistency", pairwise + 1) is None


def test_an_unmeasured_evaluation_method_is_not_capped():
    assert label_evaluation_too_big("graph_modularity", HUGE) is None


# --------------------------------------------------------------------------- #
# The documented limits are the real ones
# --------------------------------------------------------------------------- #


def test_the_documented_limits_match_the_code():
    """The docs table is the fifth place these numbers appear.

    Nothing else notices if a limit is tuned and the table is not, and a
    stale table is worse than none - it tells the user the tool will work.
    """
    page = Path(__file__).parent.parent / "docs/concepts/large-embeddings.md"
    text = page.read_text()
    documented = dict(
        re.findall(r"\*\*Clustering — ([\w-]+)\*\* \| ([\d,]+) points", text)
    )
    assert documented, "the limits table is gone or its shape changed"

    for method, shown in documented.items():
        assert int(shown.replace(",", "")) == clustering_limit(method), method

    uncapped = re.findall(r"\*\*Clustering — ([\w-]+)\*\* \| none", text)
    for method in uncapped:
        assert clustering_limit(method) is None, method
    assert set(documented) | set(uncapped) == set(ScatterControls._CLUSTER_METHODS)

    for limit in (
        FIDELITY_LIMIT,
        KNN_EDGE_LIMIT,
        HEATMAP_LIMIT,
        FEATURE_COMPARISON_LIMIT,
        LABEL_EVALUATION_LIMITS["silhouette"],
        LABEL_EVALUATION_LIMITS["neighbor_consistency"],
    ):
        assert f"{limit.max_points:,}" in text, limit


# --------------------------------------------------------------------------- #
# The combo box
# --------------------------------------------------------------------------- #


class _Points:
    """Stands in for the position array; only its length is ever read.

    Allocating 71M real rows to ask for `shape[0]` would cost half a gigabyte
    per test.
    """

    def __init__(self, n):
        self.shape = (n, 2)
        self.size = n * 2
        self.ndim = 2


class _Controls:
    _populate_cluster_methods = ScatterControls._populate_cluster_methods
    _clustering_input_size = ScatterControls._clustering_input_size
    _selected_clustering_data = ScatterControls._selected_clustering_data
    _run_clustering = ScatterControls._run_clustering
    _allow_clustering = ScatterControls._allow_clustering
    _current_cluster_method = ScatterControls._current_cluster_method
    _allow_expensive = ScatterControls._allow_expensive
    TOO_BIG_ADVICE = ScatterControls.TOO_BIG_ADVICE
    _CLUSTER_METHODS = ScatterControls._CLUSTER_METHODS
    _CLUSTER_METHODS_KNN = ScatterControls._CLUSTER_METHODS_KNN

    def __init__(self, n, confirm=False):
        self.figure = SimpleNamespace(positions=_Points(n), dists=None, messages=[])
        self.figure.show_message = lambda text, **kw: self.figure.messages.append(text)
        self.confirmed = []
        self._confirm_answer = confirm
        self.cluster_method_combo_box = QtWidgets.QComboBox()
        self.cluster_data_combo_box = QtWidgets.QComboBox()
        self.cluster_data_combo_box.addItem(CLUSTER_DATA_EMBEDDING_OPTION)

    def _confirm_expensive(self, title, html):
        self.confirmed.append(html)
        return self._confirm_answer

    def method_items(self):
        combo = self.cluster_method_combo_box
        model = combo.model()
        return {
            combo.itemText(i): model.item(i).isEnabled() for i in range(combo.count())
        }


def test_a_small_project_offers_every_method(qapp):
    ctrl = _Controls(n=1_000)
    ctrl._populate_cluster_methods()

    assert all(ctrl.method_items().values())


def test_a_huge_project_bars_only_the_blocking_method(qapp):
    """The slow ones stay pickable - they ask before they run."""
    ctrl = _Controls(n=HUGE)
    ctrl._populate_cluster_methods()

    disabled = [name for name, ok in ctrl.method_items().items() if not ok]
    assert len(disabled) == 1
    assert disabled[0].startswith("Agglomerative")


def test_a_barred_method_says_why_in_its_label(qapp):
    """A disabled item's tooltip is not reachable in every Qt style."""
    ctrl = _Controls(n=HUGE)
    ctrl._populate_cluster_methods()

    label = next(name for name in ctrl.method_items() if name.startswith("Agglomerative"))
    assert "25,000" in label


def test_the_selection_lands_on_a_method_that_can_run(qapp):
    ctrl = _Controls(n=HUGE)
    ctrl._populate_cluster_methods()

    assert not ctrl.cluster_method_combo_box.currentText().startswith("Agglomerative")


def test_a_usable_choice_survives_a_repopulate(qapp):
    ctrl = _Controls(n=1_000)
    ctrl._populate_cluster_methods()
    ctrl.cluster_method_combo_box.setCurrentText("Spectral")

    ctrl._populate_cluster_methods()

    assert ctrl.cluster_method_combo_box.currentText() == "Spectral"


def test_growing_past_a_blocking_limit_moves_off_that_method(qapp):
    ctrl = _Controls(n=1_000)
    ctrl._populate_cluster_methods()
    ctrl.cluster_method_combo_box.setCurrentText("Agglomerative")

    ctrl.figure.positions = _Points(HUGE)
    ctrl._populate_cluster_methods()

    assert not ctrl.cluster_method_combo_box.currentText().startswith("Agglomerative")


# --------------------------------------------------------------------------- #
# The gate itself
# --------------------------------------------------------------------------- #


def test_a_blocking_limit_is_refused_without_asking(qapp):
    ctrl = _Controls(n=HUGE, confirm=True)

    allowed = ctrl._allow_expensive(clustering_too_big("Agglomerative", HUGE))

    assert allowed is False
    assert not ctrl.confirmed, "it must not offer to continue into an OOM"
    assert "Agglomerative" in ctrl.figure.messages[-1]
    assert ctrl.TOO_BIG_ADVICE in ctrl.figure.messages[-1]


def test_a_patience_limit_asks_and_honours_yes(qapp):
    ctrl = _Controls(n=HUGE, confirm=True)

    assert ctrl._allow_expensive(clustering_too_big("HDBSCAN", HUGE)) is True
    assert ctrl.confirmed, "it should have asked"


def test_a_patience_limit_asks_and_honours_no(qapp):
    ctrl = _Controls(n=HUGE, confirm=False)

    assert ctrl._allow_expensive(clustering_too_big("HDBSCAN", HUGE)) is False


def test_nothing_is_asked_when_there_is_no_limit(qapp):
    ctrl = _Controls(n=HUGE)

    assert ctrl._allow_expensive(None) is True
    assert not ctrl.confirmed
    assert not ctrl.figure.messages


def test_running_a_barred_method_is_stopped_at_the_run_path(qapp):
    """The combo bars it, but the data can change without a repopulate."""
    ctrl = _Controls(n=HUGE, confirm=True)
    ctrl._populate_cluster_methods()
    combo = ctrl.cluster_method_combo_box
    combo.blockSignals(True)
    combo.setCurrentIndex(
        next(i for i in range(combo.count()) if combo.itemText(i).startswith("Agglomerative"))
    )
    combo.blockSignals(False)

    ctrl._run_clustering()

    assert "Agglomerative" in ctrl.figure.messages[-1]


# --------------------------------------------------------------------------- #
# The other entry points that read every point
# --------------------------------------------------------------------------- #


class _Gated:
    """Minimal stand-in for the controls around one gated action."""

    _allow_expensive = ScatterControls._allow_expensive
    _untick = staticmethod(ScatterControls._untick)
    TOO_BIG_ADVICE = ScatterControls.TOO_BIG_ADVICE

    class _Figure:
        """`len(figure)` is what the gates measure, so it has to be real."""

        def __init__(self, n):
            self.positions = _Points(n)
            self.metadata = {"a": 1}
            self.messages = []
            self.show_knn_edges = False
            self._n = n

        def __len__(self):
            return self._n

        def show_message(self, text, **kw):
            self.messages.append(text)

    def __init__(self, n, confirm=False):
        self.figure = self._Figure(n)
        self.asked = []
        self._confirm_answer = confirm

    def _confirm_expensive(self, title, html):
        self.asked.append(html)
        return self._confirm_answer


def test_evaluating_labels_asks_before_scoring_a_huge_project(qapp):
    """`silhouette_samples` compares every pair - it is the same shape as
    Agglomerative's cost, sitting behind a checkbox."""
    ctrl = _Gated(n=HUGE, confirm=False)
    ctrl._evaluate_labels = ScatterControls._evaluate_labels.__get__(ctrl)
    ctrl.evaluate_labels_show_check = QtWidgets.QCheckBox()
    ctrl.evaluate_labels_show_check.setChecked(True)
    ctrl.evaluate_labels_data_combo_box = QtWidgets.QComboBox()
    ctrl.evaluate_labels_data_combo_box.addItem("Embedding")
    ctrl.evaluate_labels_method_combo_box = QtWidgets.QComboBox()
    ctrl.evaluate_labels_method_combo_box.addItem("Silhouette")

    ctrl._evaluate_labels()

    assert ctrl.asked, "it should have asked before scoring 71M points"
    assert not ctrl.evaluate_labels_show_check.isChecked(), "declining unticks it"


def test_drawing_knn_edges_for_every_point_asks_first(qapp):
    ctrl = _Gated(n=HUGE, confirm=False)
    ctrl.set_knn_edges = ScatterControls.set_knn_edges.__get__(ctrl)
    ctrl._fidelity_metric = lambda: "euclidean"
    ctrl.fidelity_knn_combo_box = QtWidgets.QComboBox()
    ctrl.fidelity_knn_combo_box.addItems(["Off", "Selected only", "All"])
    ctrl.fidelity_knn_k_slider = SimpleNamespace(value=lambda: 10)
    ctrl.fidelity_knn_combo_box.setCurrentText("All")

    ctrl.set_knn_edges()

    assert ctrl.asked, "it should have asked before searching 71M neighbourhoods"
    assert ctrl.figure.show_knn_edges is False
    assert ctrl.fidelity_knn_combo_box.currentText() == "Off", "declining resets it"


def test_changing_k_while_edges_are_already_on_does_not_ask_again(qapp):
    """Regression: this also runs from the k spinbox's `valueChanged`.

    That fires per keystroke, so asking on every call put a modal in front of
    each digit the user typed.
    """
    ctrl = _Gated(n=HUGE, confirm=True)
    ctrl.set_knn_edges = ScatterControls.set_knn_edges.__get__(ctrl)
    ctrl._fidelity_metric = lambda: "euclidean"
    ctrl.fidelity_knn_combo_box = QtWidgets.QComboBox()
    ctrl.fidelity_knn_combo_box.addItems(["Off", "Selected only", "All"])
    ctrl.fidelity_knn_k_slider = SimpleNamespace(value=lambda: 10)
    ctrl.fidelity_knn_combo_box.setCurrentText("All")

    ctrl.set_knn_edges()  # the switch into "All" - asks once
    assert len(ctrl.asked) == 1

    ctrl.fidelity_knn_k_slider = SimpleNamespace(value=lambda: 11)
    ctrl.set_knn_edges()  # a new k, already consented

    assert len(ctrl.asked) == 1
    assert ctrl.figure.show_knn_edges["k"] == 11


def test_declining_keeps_the_edges_already_on_screen(qapp):
    """Saying no to a new setting should not throw away the old one."""
    ctrl = _Gated(n=HUGE, confirm=False)
    ctrl.set_knn_edges = ScatterControls.set_knn_edges.__get__(ctrl)
    ctrl._fidelity_metric = lambda: "euclidean"
    ctrl.fidelity_knn_combo_box = QtWidgets.QComboBox()
    ctrl.fidelity_knn_combo_box.addItems(["Off", "Selected only", "All"])
    ctrl.fidelity_knn_k_slider = SimpleNamespace(value=lambda: 10)
    ctrl.fidelity_knn_combo_box.setCurrentText("Selected only")
    ctrl.set_knn_edges()
    assert ctrl.figure.show_knn_edges["mode"] == "selected"

    ctrl.fidelity_knn_combo_box.setCurrentText("All")
    ctrl.set_knn_edges()

    assert ctrl.figure.show_knn_edges["mode"] == "selected", "the old edges stay"
    assert ctrl.fidelity_knn_combo_box.currentText() == "Selected only"


def test_knn_edges_for_the_selection_are_never_gated(qapp):
    """"Selected only" is bounded by the selection, however big the project."""
    ctrl = _Gated(n=HUGE, confirm=False)
    ctrl.set_knn_edges = ScatterControls.set_knn_edges.__get__(ctrl)
    ctrl._fidelity_metric = lambda: "euclidean"
    ctrl.fidelity_knn_combo_box = QtWidgets.QComboBox()
    ctrl.fidelity_knn_combo_box.addItems(["Off", "Selected only", "All"])
    ctrl.fidelity_knn_k_slider = SimpleNamespace(value=lambda: 10)
    ctrl.fidelity_knn_combo_box.setCurrentText("Selected only")

    ctrl.set_knn_edges()

    assert not ctrl.asked
    assert ctrl.figure.show_knn_edges["mode"] == "selected"


def test_a_method_that_becomes_usable_again_is_kept(qapp):
    """Regression: the label carries the limit, so the *name* has to round-trip.

    A barred method's item reads "Agglomerative (max 25,000)", so comparing
    the item *text* against the bare method names silently reset the choice
    the moment that method came back into range.
    """
    ctrl = _Controls(n=HUGE)
    ctrl._populate_cluster_methods()
    combo = ctrl.cluster_method_combo_box
    # Force the barred (and therefore decorated) method to be current.
    combo.setCurrentIndex(combo.findData("Agglomerative"))
    assert combo.currentText() != "Agglomerative", "the label has to be decorated"

    ctrl.figure.positions = _Points(1_000)
    ctrl._populate_cluster_methods()

    assert ctrl._current_cluster_method() == "Agglomerative"


def test_the_menu_says_too_many_points_not_no_features(qapp):
    """A project that has features and is simply too big is not missing them."""
    import pandas as pd

    from bigclust2.gui.core import can_open_feature_comparison, feature_comparison_missing

    features = pd.DataFrame({"f": [1.0, 2.0]})
    small = {"features": features, "meta": pd.DataFrame({"a": [1, 2]})}
    assert can_open_feature_comparison(small)

    class _Meta:
        def __len__(self):
            return FEATURE_COMPARISON_LIMIT.max_points + 1

        empty = False

    big = {"features": features, "meta": _Meta()}
    assert not can_open_feature_comparison(big)
    assert "points" in feature_comparison_missing(big)
    assert "no features" == feature_comparison_missing({"meta": pd.DataFrame({"a": [1]})})

    # ... and the menu actually asks it, rather than carrying fixed text.
    from bigclust2.gui.core import VIEW_WIDGET_SPECS

    spec = next(
        s for s in VIEW_WIDGET_SPECS if s["action"] == "feature_comparison_action"
    )
    assert spec["missing"] is feature_comparison_missing
