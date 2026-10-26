"""Tests for the vectorised point-colour path.

``colors_to_rgba`` and ``labels_to_colors`` used to convert colours one point
at a time via ``cmap.Color`` / ``colormap_obj(value)``; at scatter sizes that
dominated load time (minutes for tens of millions of points). They now
de-duplicate first and gather through a lookup table.

Because the rewrite had to be behaviour-preserving, the pre-vectorisation
implementations are kept below as oracles: the tests assert the fast paths
reproduce them exactly, so a future optimisation that changes any colour is a
test failure rather than a silent visual regression.
"""

from __future__ import annotations

import cmap
import numpy as np
import pandas as pd
import pytest

from bigclust2.utils import colors_to_rgba, labels_to_colors, sorted_unique


# --------------------------------------------------------------------------- #
# Oracles: the per-point implementations these functions replaced
# --------------------------------------------------------------------------- #


def _oracle_colors_to_rgba(colors):
    return np.array([tuple(cmap.Color(c).rgba) for c in colors], dtype=np.float32)


def _oracle_labels_to_colors(
    labels,
    palette="vispy:husl",
    nan_color=(0.5, 0.5, 0.5, 1.0),
    vmin=None,
    vmax=None,
    vcenter=None,
):
    labels = np.asarray(labels)
    colormap_obj = cmap.Colormap(palette)

    if labels.dtype.kind == "f":
        lo = vmin if vmin is not None else float(np.nanmin(labels))
        hi = vmax if vmax is not None else float(np.nanmax(labels))

        if vcenter is not None:
            left_span = max(vcenter - lo, 1e-12)
            right_span = max(hi - vcenter, 1e-12)

            def _diverging_norm(v):
                if v <= vcenter:
                    return 0.5 * (v - lo) / left_span
                return 0.5 + 0.5 * (v - vcenter) / right_span

            normed = np.array([_diverging_norm(v) for v in labels], dtype=np.float64)
        else:
            span = hi - lo if hi != lo else 1.0
            normed = (labels - lo) / span

        normed = np.clip(normed, 0.0, 1.0)
        colors = np.empty((len(labels), 4), dtype=np.float32)
        for i, val in enumerate(normed):
            if np.isnan(val):
                colors[i] = np.array(nan_color, dtype=np.float32)
            else:
                colors[i] = np.array(colormap_obj(val).rgba, dtype=np.float32)
    else:
        unique_labels = sorted(set(labels[~pd.isnull(labels)].tolist()))
        n = max(len(unique_labels), 1)
        palette_colors = list(colormap_obj.iter_colors(n))
        label_to_color = {
            lbl: np.array(palette_colors[i].rgba, dtype=np.float32)
            for i, lbl in enumerate(unique_labels)
        }
        ignore = np.array(nan_color, dtype=np.float32)
        colors = np.empty((len(labels), 4), dtype=np.float32)
        for i, lbl in enumerate(labels):
            colors[i] = label_to_color.get(lbl, ignore)

    return colors


# --------------------------------------------------------------------------- #
# colors_to_rgba
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "colors",
    [
        np.array(["red", "#00ff00", "blue"] * 5),
        np.array(["red"] * 7),  # one distinct colour -> one cmap call
    ],
    ids=["mixed", "uniform"],
)
def test_string_colors_match_per_point_conversion(colors):
    assert np.allclose(colors_to_rgba(colors), _oracle_colors_to_rgba(colors))


def test_unconvertible_color_still_raises():
    """`None` is not a colour - de-duplicating must not swallow that."""
    with pytest.raises(TypeError):
        colors_to_rgba(np.array(["red", None, "blue"], dtype=object))


def test_object_array_of_tuples_and_strings():
    colors = np.empty(4, dtype=object)
    colors[:] = [(1.0, 0.0, 0.0, 1.0), (0.0, 1.0, 0.0, 0.5), (1.0, 0.0, 0.0, 1.0), "blue"]
    assert np.allclose(colors_to_rgba(colors), _oracle_colors_to_rgba(colors))


def test_rgba_float_array_passes_through():
    colors = np.random.default_rng(0).random((10, 4)).astype(np.float32)
    out = colors_to_rgba(colors)
    assert out.dtype == np.float32
    assert np.allclose(out, colors)


def test_rgb_float_array_gains_opaque_alpha():
    colors = np.random.default_rng(0).random((10, 3)).astype(np.float32)
    out = colors_to_rgba(colors)
    assert out.shape == (10, 4)
    assert np.allclose(out[:, :3], colors)
    assert np.allclose(out[:, 3], 1.0)


def test_integer_arrays_are_rejected():
    """Previously these became out-of-range 0-255 floats; now they raise."""
    with pytest.raises(ValueError, match="floats in"):
        colors_to_rgba(np.array([[255, 0, 0], [0, 128, 255]], dtype=np.uint8))


def test_bad_shapes_are_rejected():
    with pytest.raises(ValueError, match=r"\(N, 3\) or \(N, 4\)"):
        colors_to_rgba(np.zeros((4, 2), dtype=np.float32))
    with pytest.raises(ValueError, match="1D or 2D"):
        colors_to_rgba(np.zeros((2, 2, 4), dtype=np.float32))


def test_empty_input():
    assert colors_to_rgba(np.array([], dtype=object)).shape == (0, 4)


def test_distinct_colors_converted_once():
    """The whole point of the rewrite: cost scales with distinct colours."""
    calls = []
    real_color = cmap.Color

    class _CountingColor(real_color):
        def __new__(cls, value, *args, **kwargs):
            calls.append(value)
            return real_color.__new__(cls, value, *args, **kwargs)

    colors = np.array(["red", "blue"] * 500)
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(cmap, "Color", _CountingColor)
        colors_to_rgba(colors)

    assert len(calls) == 2, f"expected 2 conversions for 2 distinct colours, got {len(calls)}"


# --------------------------------------------------------------------------- #
# labels_to_colors
# --------------------------------------------------------------------------- #

_CONTINUOUS_KWARGS = [
    {},
    {"vmin": 0.2, "vmax": 0.8},
    {"vmin": 0.0, "vmax": 1.0, "vcenter": 0.3},
]


@pytest.mark.parametrize("kwargs", _CONTINUOUS_KWARGS)
@pytest.mark.parametrize(
    "labels",
    [
        np.random.default_rng(0).random(200),
        np.random.default_rng(1).random(200).astype(np.float32),
        np.where(np.random.default_rng(2).random(200) < 0.2, np.nan, 0.5),
        np.full(20, np.nan),
        np.full(20, 3.0),  # constant -> zero span
    ],
    ids=["float64", "float32", "with-nan", "all-nan", "constant"],
)
def test_continuous_matches_per_point_oracle(labels, kwargs):
    assert np.allclose(
        labels_to_colors(labels, **kwargs),
        _oracle_labels_to_colors(labels, **kwargs),
        equal_nan=True,
    )


@pytest.mark.parametrize(
    "labels",
    [
        np.random.default_rng(0).choice(list("zyxwabc"), 200),
        np.random.default_rng(1).integers(0, 7, 200),
        np.array(np.random.default_rng(2).choice(["a", "b", None], 200), dtype=object),
        np.array(["only"] * 20),
        np.array([], dtype=object),
    ],
    ids=["strings", "ints", "with-none", "single", "empty"],
)
def test_categorical_matches_per_point_oracle(labels):
    assert np.allclose(
        labels_to_colors(labels), _oracle_labels_to_colors(labels)
    )


def test_categorical_palette_follows_sorted_order():
    """A given label must always get the same colour, whatever the row order."""
    labels = np.array(["c", "a", "b", "a"])
    shuffled = np.array(["a", "b", "c", "a"])
    first = labels_to_colors(labels)
    second = labels_to_colors(shuffled)
    for label in "abc":
        assert np.allclose(first[labels == label][0], second[shuffled == label][0])


def test_nulls_get_nan_color():
    labels = np.array(["a", None, "b"], dtype=object)
    out = labels_to_colors(labels, nan_color=(0.1, 0.2, 0.3, 0.4))
    assert np.allclose(out[1], (0.1, 0.2, 0.3, 0.4))


# --------------------------------------------------------------------------- #
# sorted_unique
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "values",
    [["b", "a", "b"], np.array(["b", "a", "b"]), pd.Series(["b", "a", "b"])],
    ids=["list", "ndarray", "series"],
)
def test_sorted_unique_accepts_list_array_and_series(values):
    assert list(sorted_unique(values)) == ["a", "b"]


def test_sorted_unique_matches_np_unique():
    values = np.random.default_rng(0).choice(list("hello world"), 500)
    assert np.array_equal(sorted_unique(values), np.unique(values))
