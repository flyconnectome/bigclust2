import re
from dataclasses import dataclass
import shlex
import colorsys
import warnings

import numpy as np
import pandas as pd
import polars as pl
import matplotlib.colors as mcl

from pathlib import Path
from urllib.parse import urlparse, urlunparse, urljoin


def is_url(string: str) -> bool:
    """Check if a string is a URL.

    Args:
        string: The string to check

    Returns:
        True if the string is a URL, False otherwise
    """
    try:
        result = urlparse(string)
        return all([result.scheme, result.netloc])
    except Exception:
        return False


def is_relative(path: str | Path) -> bool:
    """Check if a path is relative.

    Args:
        path: The path to check (string or Path object)

    Returns:
        True if the path is relative, False if absolute
    """
    return not Path(path).is_absolute()


def is_list_of_ids(string: str) -> bool:
    """Check if a string represents a list of IDs (comma-separated values).

    Args:
        string: The string to check
    Returns:
        True if the string is a list of IDs, False otherwise
    """
    if not isinstance(string, str) or not string.strip():
        return False

    parts = [part.strip() for part in string.split(",")]
    for part in parts:
        if not part:
            return False
        if not part.isdigit():
            return False
    return True


def build_launch_command(source, filter_expr=None, embedding_mode=None):
    """Build the command line(s) that re-open a given view.

    Produces the two equivalent invocations that reproduce a project view from
    the command line: one via ``uvx`` (runs without installing anything) and one
    via the installed ``bigclust2`` console script. Every token is quoted with
    :func:`shlex.quote` so the strings are safe to paste into a POSIX shell.

    Args:
        source:         Path or URL of the dataset (``--from``). Required.
        filter_expr:    Filter expression string (``--filters``). Optional.
        embedding_mode: Load-time embedding mode (``--embedding``). Optional and
                        omitted for the default "use precomputed".

    Returns:
        dict with keys ``"uvx"`` and ``"installed"`` mapping to the two command
        strings.
    """
    args = ["--from", str(source)]

    if filter_expr and str(filter_expr).strip():
        args += ["--filters", str(filter_expr).strip()]

    mode = (str(embedding_mode).strip() if embedding_mode else "")
    if mode and mode.lower() != "use precomputed":
        args += ["--embedding", mode]

    tail = " ".join(shlex.quote(a) for a in args)
    return {
        "uvx": f"uvx bigclust2@latest {tail}",
        "installed": f"bigclust2 {tail}",
    }


def string_to_polars_filter(filter_expr: str) -> pl.Expr:
    """Convert a string filter expression to a Polars expression.

    Args:
        filter_expr: The filter expression as a string (e.g. 'column_a > 5 & column_b == "test"')

    Returns:
        A Polars expression representing the filter
    """
    if not isinstance(filter_expr, str) or not filter_expr.strip():
        raise ValueError("filter_expr must be a non-empty string")

    s = filter_expr.strip()

    def is_ident_start(ch: str) -> bool:
        return ch.isalpha() or ch == "_"

    def is_ident_part(ch: str) -> bool:
        return ch.isalnum() or ch == "_"

    class Tok:
        def __init__(self, t: str, v):
            self.t = t
            self.v = v

        def __repr__(self):
            return f"Tok({self.t},{self.v})"

    i = 0
    n = len(s)
    tokens = []

    while i < n:
        ch = s[i]
        if ch.isspace():
            i += 1
            continue
        if ch in "()":
            tokens.append(Tok("LPAREN" if ch == "(" else "RPAREN", ch))
            i += 1
            continue
        if ch in "'\"":
            quote = ch
            i += 1
            buf = []
            while i < n:
                c = s[i]
                if c == "\\" and i + 1 < n:
                    buf.append(s[i + 1])
                    i += 2
                    continue
                if c == quote:
                    i += 1
                    break
                buf.append(c)
                i += 1
            tokens.append(Tok("STRING", "".join(buf)))
            continue
        if ch.isdigit() or (ch == "." and i + 1 < n and s[i + 1].isdigit()):
            j = i
            has_dot = False
            while j < n and (s[j].isdigit() or s[j] == "."):
                if s[j] == ".":
                    if has_dot:
                        break
                    has_dot = True
                j += 1
            num_str = s[i:j]
            val = float(num_str) if has_dot else int(num_str)
            tokens.append(Tok("NUMBER", val))
            i = j
            continue
        if is_ident_start(ch):
            j = i + 1
            while j < n and is_ident_part(s[j]):
                j += 1
            ident = s[i:j]
            low = ident.lower()
            if low == "and":
                tokens.append(Tok("OP", "&"))
                i = j
            elif low == "or":
                tokens.append(Tok("OP", "|"))
                i = j
            elif low == "in":
                # Handle "in" operator with list: in (val1, val2, ...) or in [val1, val2, ...]
                # Skip whitespace after "in"
                k = j
                while k < n and s[k].isspace():
                    k += 1
                if k < n and s[k] in "([":
                    # Parse the list
                    closing = ")" if s[k] == "(" else "]"
                    k += 1
                    list_items = []
                    while k < n:
                        # Skip whitespace
                        while k < n and s[k].isspace():
                            k += 1
                        if k >= n:
                            raise ValueError(
                                f"Unclosed list in 'in' operator: expected '{closing}' before end of string"
                            )
                        if s[k] == closing:
                            k += 1
                            break
                        # Parse a value (string, number)
                        if s[k] in "'\"":
                            quote = s[k]
                            k += 1
                            buf = []
                            found_closing_quote = False
                            while k < n:
                                c = s[k]
                                if c == "\\" and k + 1 < n:
                                    buf.append(s[k + 1])
                                    k += 2
                                    continue
                                if c == quote:
                                    k += 1
                                    found_closing_quote = True
                                    break
                                buf.append(c)
                                k += 1
                            if not found_closing_quote:
                                raise ValueError(
                                    f"Unclosed string in list at pos {k}: expected '{quote}'"
                                )
                            list_items.append("".join(buf))
                        elif s[k].isdigit() or (
                            s[k] == "." and k + 1 < n and s[k + 1].isdigit()
                        ):
                            start = k
                            has_dot = False
                            while k < n and (s[k].isdigit() or s[k] == "."):
                                if s[k] == ".":
                                    if has_dot:
                                        break
                                    has_dot = True
                                k += 1
                            num_str = s[start:k]
                            val = float(num_str) if has_dot else int(num_str)
                            list_items.append(val)
                        else:
                            raise ValueError(
                                f"Unexpected token in list at pos {k}: {s[k]}"
                            )
                        # Skip whitespace and comma
                        while k < n and s[k].isspace():
                            k += 1
                        if k < n and s[k] == ",":
                            k += 1
                    tokens.append(Tok("IN", "in"))
                    tokens.append(Tok("LIST", tuple(list_items)))
                    i = k
                else:
                    raise ValueError(
                        f"'in' operator must be followed by a list in parentheses or brackets at pos {j}"
                    )
            else:
                tokens.append(Tok("IDENT", ident))
                i = j
            continue
        # operators
        two = s[i : i + 2]
        if two in ("==", "!=", ">=", "<="):
            tokens.append(Tok("OP", two))
            i += 2
            continue
        if ch in (">", "<", "&", "|"):
            tokens.append(Tok("OP", ch))
            i += 1
            continue
        raise ValueError(f"Unexpected character in filter expression at pos {i}: {ch}")

    prec = {
        "==": 3,
        "!=": 3,
        ">": 3,
        ">=": 3,
        "<": 3,
        "<=": 3,
        "in": 3,
        "&": 2,
        "|": 1,
    }

    output = []
    op_stack = []

    for tok in tokens:
        if tok.t in ("IDENT", "NUMBER", "STRING"):
            output.append(tok)
        elif tok.t == "LIST":
            output.append(tok)
        elif tok.t == "IN":
            while op_stack:
                top = op_stack[-1]
                if top.t == "OP" and prec[top.v] >= prec["in"]:
                    output.append(op_stack.pop())
                elif top.t == "IN":
                    output.append(op_stack.pop())
                else:
                    break
            op_stack.append(tok)
        elif tok.t == "OP":
            while op_stack:
                top = op_stack[-1]
                if top.t == "OP" and prec[top.v] >= prec[tok.v]:
                    output.append(op_stack.pop())
                elif top.t == "IN" and prec["in"] >= prec[tok.v]:
                    output.append(op_stack.pop())
                else:
                    break
            op_stack.append(tok)
        elif tok.t == "LPAREN":
            op_stack.append(tok)
        elif tok.t == "RPAREN":
            while op_stack and op_stack[-1].t != "LPAREN":
                output.append(op_stack.pop())
            if not op_stack:
                raise ValueError("Mismatched parentheses in filter expression")
            op_stack.pop()
    while op_stack:
        t = op_stack.pop()
        if t.t in ("LPAREN", "RPAREN"):
            raise ValueError("Mismatched parentheses in filter expression")
        output.append(t)

    def to_expr(tok: Tok):
        if tok.t == "IDENT":
            return pl.col(tok.v)
        if tok.t == "NUMBER" or tok.t == "STRING":
            return pl.lit(tok.v)
        raise ValueError(f"Invalid token to convert: {tok}")

    stack = []
    for tok in output:
        if tok.t in ("IDENT", "NUMBER", "STRING"):
            stack.append(to_expr(tok))
        elif tok.t == "LIST":
            stack.append(tok.v)
        elif tok.t == "IN":
            if len(stack) < 2:
                raise ValueError("Invalid expression: insufficient operands for 'in'")
            list_val = stack.pop()
            col_expr = stack.pop()
            if not isinstance(list_val, tuple):
                raise ValueError("Right operand of 'in' must be a list")
            stack.append(col_expr.is_in(list_val))
        elif tok.t == "OP":
            if len(stack) < 2:
                raise ValueError("Invalid expression: insufficient operands")
            rhs = stack.pop()
            lhs = stack.pop()
            if tok.v == "==":
                stack.append(lhs == rhs)
            elif tok.v == "!=":
                stack.append(lhs != rhs)
            elif tok.v == ">":
                stack.append(lhs > rhs)
            elif tok.v == ">=":
                stack.append(lhs >= rhs)
            elif tok.v == "<":
                stack.append(lhs < rhs)
            elif tok.v == "<=":
                stack.append(lhs <= rhs)
            elif tok.v == "&":
                stack.append(lhs & rhs)
            elif tok.v == "|":
                stack.append(lhs | rhs)
            else:
                raise ValueError(f"Unsupported operator: {tok.v}")
        else:
            raise ValueError(f"Unexpected token in RPN: {tok}")

    if len(stack) != 1:
        raise ValueError("Invalid filter expression")
    return stack[0]


class Url:
    """Class representing a URL.

    This is analogous to pathlib.Path but for URLs and
    with limited functionality.
    """

    def __init__(self, url: str):
        if isinstance(url, Url):
            self.url = url.url
        elif not isinstance(url, str):
            raise TypeError("Url must be initialized with a string or Url object")

        self.url = url

    def __str__(self):
        return self.url

    def __repr__(self):
        return f"Url({self.url})"

    def __div__(self, other) -> "Url":
        return self.joinpath(other)

    def __truediv__(self, other) -> "Url":
        return self.joinpath(other)

    def joinpath(self, *args) -> "Url":
        new_url = self.url
        for arg in args:
            if not isinstance(arg, str):
                arg = str(arg)

            if not new_url.endswith("/"):
                new_url += "/"
            new_url = urljoin(new_url, arg)
        return Url(new_url)

    @property
    def parent(self) -> "Url":
        parsed = urlparse(self.url)
        path_parts = parsed.path.rstrip("/").split("/")
        parent_path = "/".join(path_parts[:-1])
        new_parsed = parsed._replace(path=parent_path)
        return Url(urlunparse(new_parsed))

    @property
    def name(self) -> str:
        parsed = urlparse(self.url)
        path_parts = parsed.path.rstrip("/").split("/")
        return path_parts[-1] if path_parts else ""

    @property
    def suffix(self) -> str:
        name = self.name
        if "." in name:
            return name[name.rfind(".") :]
        return ""

    @property
    def stem(self) -> str:
        name = self.name
        if "." in name:
            return name[: name.rfind(".")]
        return name


def hash_function(state, value):
    """This is a modified murmur hash.
    """
    k1 = 0xCC9E2D51
    k2 = 0x1B873593
    state = state & 0xFFFFFFFF
    value = (value * k1) & 0xFFFFFFFF
    value = ((value << 15) | value >> 17) & 0xFFFFFFFF
    value = (value * k2) & 0xFFFFFFFF
    state = (state ^ value) & 0xFFFFFFFF
    state = ((state << 13) | state >> 19) & 0xFFFFFFFF
    state = ((state * 5) + 0xE6546B64) & 0xFFFFFFFF
    return state


def rgb_from_segment_id(color_seed, segment_id):
    """Return the RGBA for a segment given a color seed and the segment ID."""
    segment_id = int(segment_id)  # necessary since segment_id is 64 bit originally
    result = hash_function(state=color_seed, value=segment_id)
    newvalue = segment_id >> 32
    result2 = hash_function(state=result, value=newvalue)
    c0 = (result2 & 0xFF) / 255.0
    c1 = ((result2 >> 8) & 0xFF) / 255.0
    h = c0
    s = 0.5 + 0.5 * c1
    v = 1.0
    return tuple([v * 255 for v in colorsys.hsv_to_rgb(h, s, v)])


def take(arr, ix):
    """``arr[ix]``, or ``arr`` itself when ``ix`` is None ("everything, in order")."""
    return arr if ix is None else arr[ix]


def as_xyz(xy, z=0.0, order=None):
    """Promote (N, 2) points to a C-contiguous (N, 3) float32 array at height `z`.

    Parameters
    ----------
    xy :    (N, 2) array
    z :     float
            Value for the third column.
    order : (M,) int array, optional
            Gather rows in this order straight into the output, instead of
            building a reordered (M, 2) copy first. Writing through the strided
            destination measures faster than gather-then-copy.

    Returns
    -------
    xyz :   (N or M, 3) float32 array
    """
    xy = np.asarray(xy)
    n = len(xy) if order is None else len(order)
    xyz = np.empty((n, 3), dtype=np.float32)
    xyz[:, 2] = z
    if order is None:
        xyz[:, :2] = xy
    else:
        # `np.take`'s `out=` enforces safe casting, so an integer input would
        # raise where plain assignment converts; convert up front instead.
        if xy.dtype != np.float32:
            xy = xy.astype(np.float32)
        np.take(xy, order, axis=0, out=xyz[:, :2])
    return xyz


@dataclass(frozen=True)
class TooBig:
    """An operation whose input is larger than it can handle.

    Carries the facts, not the sentence the user sees: these live in modules
    that import without Qt, while what to *do* about it (filter, open a view)
    is a property of the application.

    `blocking` separates the two kinds. An operation that would exhaust memory
    cannot be offered at all - there is no informed consent to give when the
    machine goes down either way. One that is merely slow is the user's call,
    so it is offered with the cost stated and cancel as the default.
    """

    operation: str
    n: int
    limit: int
    reason: str
    blocking: bool = False

    def __str__(self):
        return (
            f"{self.operation} is limited to {self.limit:,} points "
            f"({self.n:,} here) because {self.reason}."
        )


@dataclass(frozen=True)
class MethodLimit:
    """How large an input something can take, and why.

    The table form of :class:`TooBig`: constants declare the limit and the
    reason once, next to the measurements that justify them, and `check`
    turns one into a `TooBig` when an input exceeds it.
    """

    #: Largest input this can handle, or None when it has no practical cap.
    max_points: int | None
    #: Why, as a clause - "because {reason}". Describes the algorithm, not
    #: what the user should do about it, which belongs to the application.
    reason: str
    #: True when going past it exhausts memory rather than merely taking time.
    blocking: bool = False

    def check(self, operation, n):
        """`TooBig` if `operation` cannot handle `n` points, else None."""
        if self.max_points is None or n <= self.max_points:
            return None
        return TooBig(
            operation=operation,
            n=n,
            limit=self.max_points,
            reason=self.reason,
            blocking=self.blocking,
        )


def points_in_rect(xy, rect):
    """Mask of which of ``xy`` ((N, 2)) fall inside ``rect``.

    ``rect`` is ``(xmin, ymin, xmax, ymax)``; the edges are inclusive. Points
    with a non-finite coordinate compare False throughout, so they are never
    reported inside.
    """
    xmin, ymin, xmax, ymax = rect
    with np.errstate(invalid="ignore"):  # NaN compares False, as intended
        return (
            (xy[:, 0] >= xmin)
            & (xy[:, 0] <= xmax)
            & (xy[:, 1] >= ymin)
            & (xy[:, 1] <= ymax)
        )


def sorted_unique(values):
    """Return the sorted unique entries of ``values``.

    Equivalent to ``np.unique`` but hashes instead of sorting the full input,
    which matters for large object/string arrays: ``np.unique`` sorts all N
    entries, while this only sorts the (usually far fewer) distinct ones.
    """
    if isinstance(values, list):
        values = np.asarray(values)  # pd.unique does not accept plain lists
    return np.sort(pd.unique(values))


def colors_to_rgba(colors):
    """Convert per-point colours to an ``(N, 4)`` float32 RGBA array.

    Parameters
    ----------
    colors : array-like
        Either an ``(N,)`` sequence of colour specs (strings or RGB(A) tuples)
        or an already-numeric ``(N, 3)``/``(N, 4)`` float array in ``[0, 1]``.

    Returns
    -------
    colors : np.ndarray of shape (N, 4), dtype float32

    Notes
    -----
    Colour specs are de-duplicated before conversion, so the (comparatively
    expensive) ``cmap.Color`` calls scale with the number of *distinct*
    colours rather than with ``N``. That matters for large scatters, where a
    per-point conversion dominates load time.
    """
    import cmap as _cmap

    colors = np.asarray(colors)

    if colors.ndim == 2:
        if colors.shape[1] not in (3, 4):
            raise ValueError(
                f"Expected (N, 3) or (N, 4) colour array, got {colors.shape}."
            )
        if colors.dtype.kind != "f":
            raise ValueError(
                "Numeric colour arrays must be floats in [0, 1], got dtype "
                f"{colors.dtype}."
            )
        if colors.shape[1] == 4:
            return colors.astype(np.float32, copy=False)
        # Add the alpha channel without the two copies `hstack` would make.
        out = np.empty((len(colors), 4), dtype=np.float32)
        out[:, :3] = colors
        out[:, 3] = 1.0
        return out

    if colors.ndim != 1:
        raise ValueError(f"Expected 1D or 2D colour array, got {colors.ndim}D.")

    # Keep NaN/None as their own category so that an unconvertible colour
    # raises here just as a per-point conversion would have.
    codes, uniques = pd.factorize(colors, use_na_sentinel=False)
    lut = np.array(
        [tuple(_cmap.Color(c).rgba) for c in uniques], dtype=np.float32
    ).reshape(-1, 4)
    return lut[codes]


def labels_to_colors(
    labels,
    palette="vispy:husl",
    nan_color=(0.5, 0.5, 0.5, 1.0),
    vmin=None,
    vmax=None,
    vcenter=None,
):
    """Map per-point values to RGBA colours.

    Parameters
    ----------
    labels : array-like
        Value per point. Float input is mapped continuously, anything else
        categorically. Null entries (NaN/None) get ``nan_color``.
    palette : str
        A ``cmap``-compatible colormap name used for distinct cluster colours.
    nan_color : tuple of 4 floats
        RGBA colour for NaN points.
    vmin : float, optional
        Lower bound for continuous colour mapping. Defaults to data minimum.
    vmax : float, optional
        Upper bound for continuous colour mapping. Defaults to data maximum.
    vcenter : float, optional
        Centre value for diverging colour mapping. When set, maps
        [vmin, vcenter] → [0, 0.5] and [vcenter, vmax] → [0.5, 1.0].

    Returns
    -------
    colors : np.ndarray of shape (N, 4), dtype float32
        Per-point RGBA colours in ``[0, 1]`` range.
    """
    import cmap as _cmap

    labels = np.asarray(labels)
    colormap_obj = _cmap.Colormap(palette)

    if labels.dtype.kind == "f":
        # Continuous mapping
        lo = vmin if vmin is not None else float(np.nanmin(labels))
        hi = vmax if vmax is not None else float(np.nanmax(labels))

        if vcenter is not None:
            # Piecewise diverging normalisation. Folding the division into the
            # multiplier keeps this to one temporary per arm.
            left_scale = 0.5 / max(vcenter - lo, 1e-12)
            right_scale = 0.5 / max(hi - vcenter, 1e-12)
            normed = np.where(
                labels <= vcenter,
                (labels - lo) * left_scale,
                (labels - vcenter) * right_scale + 0.5,
            )
        else:
            span = hi - lo if hi != lo else 1.0
            normed = (labels - lo) * (1.0 / span)

        # `colormap_obj` is vectorised; NaNs come back as transparent black,
        # so blank them out first and paint them with `nan_color` afterwards.
        is_nan = np.isnan(normed)
        np.clip(normed, 0.0, 1.0, out=normed)
        normed[is_nan] = 0.0
        colors = np.asarray(colormap_obj(normed), dtype=np.float32).reshape(-1, 4)
        colors[is_nan] = np.asarray(nan_color, dtype=np.float32)
    else:
        # `sort=True` makes the codes index the sorted unique values directly,
        # so a given label always gets the same palette entry.
        codes, uniques = pd.factorize(labels, sort=True)

        # One extra row holding `nan_color`: `pd.factorize` codes nulls as -1,
        # which indexes exactly that last row, so this is a single gather.
        lut = np.empty((len(uniques) + 1, 4), dtype=np.float32)
        if len(uniques):
            lut[:-1] = [c.rgba for c in colormap_obj.iter_colors(len(uniques))]
        lut[-1] = np.asarray(nan_color, dtype=np.float32)
        colors = lut[codes]

    return colors


def check_finite_features(X, context, action="raise"):
    """Check a feature matrix for non-finite values (NaN/inf).

    Args:
        X: The feature matrix — a numpy array or pandas DataFrame. For
            DataFrames only numeric columns are scanned.
        context: Short description of the operation the features are used for,
            e.g. "grow/shrink selection". Used in the error/warning message.
        action: "raise" to raise a ValueError when non-finite values are found,
            "warn" to emit a warning instead.

    Returns:
        bool: True if the matrix is fully finite, False otherwise (only
        reachable with ``action="warn"``).
    """
    if isinstance(X, pd.DataFrame):
        values = X.select_dtypes(include="number").values
    else:
        values = np.asarray(X)

    if values.dtype.kind not in "fc":  # integer/bool arrays are always finite
        return True

    finite = np.isfinite(values)
    if finite.all():
        return True

    n_bad = int(finite.size - finite.sum())
    bad_rows = int((~finite).any(axis=1).sum()) if finite.ndim == 2 else n_bad
    bad_cols = int((~finite).any(axis=0).sum()) if finite.ndim == 2 else 1
    n_rows, n_cols = (values.shape if finite.ndim == 2 else (values.size, 1))
    msg = (
        f"Feature matrix used for {context} contains {n_bad:,} non-finite "
        f"values (NaN/inf) in {bad_rows:,} of {n_rows:,} rows "
        f"({bad_cols:,} of {n_cols:,} columns). Impute or drop missing values "
        f"before using feature-based {context}."
    )
    if action == "raise":
        raise ValueError(msg)

    warnings.warn(msg)
    return False


def is_color_column(series: pd.Series) -> bool:
    """Check if a pandas Series is a color column (contains valid color strings).

    Args:
        series: The pandas Series to check

    Returns:
        bool: True if the series is a color column, False otherwise.
    """
    if series.dtype.kind != "O":
        return False

    for val in series.dropna().unique():
        if not isinstance(val, str):
            return False
        try:
            mcl.to_rgb(val)
        except (ValueError, TypeError):
            return False

    return True


class SparseMap:
    """A fixed-length ``index -> value`` store whose default is ``None``.

    Stands in for a ``[None] * n`` list where only a handful of the entries are
    ever filled in - which is the case for the figure's per-point label visuals
    and label colours: only labels that have been on screen exist, out of
    however many points the project has. A dense list of 71M ``None``
    references costs 568 MB and makes every traversal O(N) in Python; this
    costs nothing for the empty entries and iterates only the filled ones.

    Indexing, assignment and ``len`` behave as they would on the list, negative
    indices included. Deliberately *not* iterable: ``for v in sparse`` could
    only reasonably yield the filled values, which would silently disagree with
    ``len``. Use :meth:`values` or :meth:`items` instead - both of which make
    the O(filled) cost explicit at the call site.

    How many entries end up set is bounded by how many labels have been shown,
    not by `n` - but nothing removes them, so it grows with how much of a
    project has been explored rather than staying at `label_vis_limit`.
    """

    def __init__(self, n):
        self._n = int(n)
        self._values = {}

    def _key(self, ix):
        ix = int(ix)
        if ix < 0:
            ix += self._n
        if not 0 <= ix < self._n:
            raise IndexError(f"index {ix} out of range for length {self._n}")
        return ix

    def __len__(self):
        return self._n

    def __getitem__(self, ix):
        return self._values.get(self._key(ix))

    def __setitem__(self, ix, value):
        key = self._key(ix)
        if value is None:
            self._values.pop(key, None)
        else:
            self._values[key] = value

    def __iter__(self):
        # Without this, Python's old iteration protocol would fall back to
        # `__getitem__` and happily walk all `n` indices - which is the O(N)
        # traversal this class exists to prevent, only now invisible.
        raise TypeError(
            "SparseMap is not iterable; use .values() or .items() to iterate "
            "the entries that are set"
        )

    def __repr__(self):
        return f"<SparseMap {len(self._values)}/{self._n} set>"

    def values(self):
        """The values that have been set, in no particular order."""
        return self._values.values()

    def items(self):
        """``(index, value)`` for the entries that have been set."""
        return self._values.items()

