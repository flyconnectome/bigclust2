"""Fixtures shared across the headless (non-GUI) tests."""

from __future__ import annotations

import os

# Must be set before anything imports `rendercanvas.auto` (which
# ``bigclust2.figure`` does at module import), so that the figure fixture below
# gets a canvas that needs neither a window nor a display.
os.environ.setdefault("RENDERCANVAS_FORCE_OFFSCREEN", "1")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

from bigclust2.annotation_backends import AnnotationBackend  # noqa: E402


class FakeBackend(AnnotationBackend):
    """In-memory annotation backend returning a controlled frame indexed by id.

    Lets the merge logic be exercised without a network, credentials or any live
    dataset. `raise_on_read` makes a source fail on demand.
    """

    BACKEND_NAME = "Fake"
    CONFIG_CLASS = None

    def __init__(self, frame=None, *, raise_on_read=None):
        self._frame = frame if frame is not None else pd.DataFrame()
        self._raise_on_read = raise_on_read

    def validate(self):  # pragma: no cover - trivial
        pass

    def validate_field(self, field):  # pragma: no cover - trivial
        return True

    def _write_annotations(self, ids, value, fields):  # pragma: no cover
        raise NotImplementedError("FakeBackend never writes")

    def _read_annotations(self, ids=None):
        if self._raise_on_read is not None:
            raise self._raise_on_read
        if ids is None:
            return self._frame.copy()
        wanted = {str(i) for i in ids}
        return self._frame[self._frame.index.astype(str).isin(wanted)].copy()


@pytest.fixture
def fake_backend():
    """The :class:`FakeBackend` class, for tests that build their own frames."""
    return FakeBackend


class _SelectionCounter:
    """Stand-in for the Qt label `ScatterFigure` writes its selection size to."""

    def __init__(self):
        self.text = None

    def setText(self, text):
        self.text = text


@pytest.fixture
def make_scatter():
    """Factory building a populated `ScatterFigure` on an offscreen canvas.

    Unlike the stub-based tests (e.g. ``test_center_on_selection.py``) this
    exercises the real visual-building path - which is the only way to cover
    point grouping, the colour upload and hover picking. Skips if no GPU
    adapter is available (e.g. a CI box without one).
    """
    from bigclust2.scatter import ScatterFigure

    created = []

    def _make(n=120, markers=None, colors="red", seed=0, hover_col="label: {label}"):
        rng = np.random.default_rng(seed)
        if markers is None:
            markers = np.full(n, "ds", dtype=object)
        if isinstance(colors, str):
            colors = np.full(n, colors, dtype=object)
        meta = pd.DataFrame(
            {
                "id": np.arange(n),
                "label": [f"point-{i}" for i in range(n)],
                "dataset": markers,
                "_color": colors,
            }
        )
        points = rng.random((n, 2)).astype(np.float32)
        try:
            fig = ScatterFigure(selection_counter=_SelectionCounter(), parent=None)
        except Exception as exc:  # pragma: no cover - depends on the host GPU
            pytest.skip(f"no usable render canvas: {exc}")
        created.append(fig)
        fig.set_points(
            points=points,
            metadata=meta,
            color_col="_color",
            hover_col=hover_col,
        )
        return fig, meta, points

    yield _make

    for fig in created:
        fig.canvas.close()
