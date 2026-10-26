"""Fixtures shared across the headless (non-GUI) tests."""

from __future__ import annotations

import os

# Must be set before anything imports `rendercanvas.auto` (which
# ``bigclust2.figure`` does at module import), so that the figure fixture below
# gets a canvas that needs neither a window nor a display.
os.environ.setdefault("RENDERCANVAS_FORCE_OFFSCREEN", "1")
# Qt widgets run fine headless under the offscreen platform (the wgpu canvas
# does not, which is why figures use the offscreen rendercanvas above).
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

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


#: Chunk settings that force a test-sized scatter to split into several chunks.
#: Production thresholds (see `ScatterFigure.chunk_target`/`chunk_min_size`)
#: would leave it as a single visual. `chunk_min_size` has to be small enough
#: for the point counts the tests actually use (tens, not thousands), or the
#: "chunked" runs silently collapse back to one visual and assert nothing.
FORCE_CHUNKS = {"chunk_target": 8, "chunk_min_size": 5}


@pytest.fixture
def force_chunks():
    """`FORCE_CHUNKS`, for tests that build a chunked figure themselves."""
    return dict(FORCE_CHUNKS)


@pytest.fixture
def scatter_factory():
    """Factory building a populated `ScatterFigure` on an offscreen canvas.

    Unlike the stub-based tests (e.g. ``test_center_on_selection.py``) this
    exercises the real visual-building path - which is the only way to cover
    point grouping, the colour upload and hover picking. Skips if no GPU
    adapter is available (e.g. a CI box without one).
    """
    from bigclust2.scatter import ScatterFigure

    created = []

    def _make(
        n=120,
        markers=None,
        colors="red",
        seed=0,
        hover_col="label: {label}",
        chunk_target=None,
        chunk_min_size=None,
    ):
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
        # Must be set before `set_points`, which is what builds the visuals.
        for name, value in (
            ("chunk_target", chunk_target),
            ("chunk_min_size", chunk_min_size),
        ):
            if value is not None:
                setattr(fig, name, value)
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


@pytest.fixture(params=["unchunked", "chunked"])
def make_scatter(request, scatter_factory):
    """`scatter_factory`, run once unchunked and once with chunking forced.

    Production thresholds leave a test-sized scatter unchunked, so without this
    the whole suite would only ever exercise the single-visual path. Everything
    addressed through ``vis._point_ix`` has to behave the same either way, so
    the invariants are asserted once and checked in both layouts.
    """
    forced = {"unchunked": {}, "chunked": FORCE_CHUNKS}

    def _make(**kwargs):
        for key, value in forced[request.param].items():
            kwargs.setdefault(key, value)
        result = scatter_factory(**kwargs)
        if request.param == "chunked":
            # Guard against this parametrization quietly becoming a duplicate
            # of the unchunked run (which is what happens if FORCE_CHUNKS stops
            # matching the point counts the tests use).
            assert len(result[0].point_visuals) > 1, (
                "the 'chunked' layout produced a single visual; "
                "FORCE_CHUNKS no longer chunks at this point count"
            )
        return result

    return _make


@pytest.fixture
def hover_at():
    """Send a pointer_move through the real event pipeline and flush it.

    Shared so the pygfx event payload lives in one place; the figure resolves
    hover from the renderer's pick buffer, so the event has to go through the
    real pipeline rather than calling the handler directly.
    """

    def _hover(fig, x, y):
        fig.canvas.submit_event(
            {
                "event_type": "pointer_move",
                "x": x,
                "y": y,
                "button": 0,
                "buttons": [],
                "modifiers": [],
                "ntouches": 0,
                "touches": {},
                "pointer_id": 1,
            }
        )
        fig.canvas._events.flush()
        fig.renderer.render(fig.scene, fig.camera)
        fig.canvas.draw()

    return _hover


@pytest.fixture
def hover_probe():
    """Capture the text the figure would put in its hover widget."""

    def _probe(fig):
        seen = {}
        fig._update_hover_widget = lambda text, *_: seen.update(text=text)
        return seen

    return _probe


@pytest.fixture(scope="session")
def qapp():
    """The shared QApplication, for tests that build real Qt widgets."""
    from PySide6 import QtWidgets

    yield QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


@pytest.fixture
def chunked(scatter_factory, force_chunks):
    """A figure whose points are split into several chunks per marker.

    Production thresholds would leave a test-sized scatter unchunked, so the
    chunk settings are lowered - but still applied through the real
    `set_points` -> `make_visuals` path, so the visuals are wired up (hover
    handlers included) exactly as they are in the app.

    Uses `scatter_factory` rather than `make_scatter`: these tests force
    chunking anyway, so running them under both of `make_scatter`'s layouts
    would just duplicate every result.
    """

    def _make(**kwargs):
        kwargs.setdefault("n", 6000)
        for key, value in force_chunks.items():
            kwargs.setdefault(key, value)
        return scatter_factory(**kwargs)

    return _make
