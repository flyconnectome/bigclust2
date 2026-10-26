"""Spatial chunking of a 2D point cloud.

A single ``gfx.Points`` holding tens of millions of points can only be drawn or
skipped as a whole, which rules out both view culling and level-of-detail. This
module splits the points into spatially coherent, roughly equal-sized chunks so
the figure can build one visual per chunk and later decide per chunk how much of
it to draw (see ``ScatterFigure._apply_draw_budget``).

Chunk bounds are computed here rather than read from pygfx. Each chunk is one
``gfx.Points``, and pygfx does maintain a cached bounding box per object - but
that cache is keyed on the position buffer's ``rev``, which (as of pygfx 0.16)
does not change on ``set_data``/``update_full``. It would therefore serve stale
bounds after the figure moves its points, which is exactly when a cull must not
be working from the old extents.

The split is by Z-order (Morton) code rather than a recursive quadtree: sorting
by interleaved coordinate bits already groups nearby points together, and
cutting that order into equal-count blocks gives chunks whose *extent* shrinks
with density while their *size* stays uniform. That is the property we want -
a zoomed-in view touches few chunks whatever the local density, and every chunk
costs about the same to draw.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field

import numpy as np

# Bits per axis for the Morton key. 16 gives a 65536x65536 grid, which is finer
# than any display, and keeps the interleaved key inside uint32.
MORTON_BITS = 16

#: Seed for the within-chunk shuffle. Fixed so that a given scatter always
#: samples the same points, rather than changing on every rebuild.
CHUNK_SEED = 0


def _part1by1(values):
    """Spread each value's bits out into every other bit.

    A 4-bit ``0babcd`` becomes ``0b0a0b0c0d``; the same halving pattern (visible
    in the shift widths below) carries up to 16 input bits. Two spread values
    can then be combined with a shift and an OR to interleave them.
    """
    x = values.astype(np.uint32, copy=True)
    x = (x | (x << np.uint32(8))) & np.uint32(0x00FF00FF)
    x = (x | (x << np.uint32(4))) & np.uint32(0x0F0F0F0F)
    x = (x | (x << np.uint32(2))) & np.uint32(0x33333333)
    x = (x | (x << np.uint32(1))) & np.uint32(0x55555555)
    return x


def morton_keys(xy, bits=MORTON_BITS):
    """Z-order keys for the points in ``xy``.

    Parameters
    ----------
    xy :    (N, 2) array
            Point coordinates. Non-finite rows are given key 0 so that they
            sort together instead of poisoning the quantisation.
    bits :  int
            Bits per axis; at most 16.

    Returns
    -------
    keys :  (N,) uint32 array
    """
    xy = np.asarray(xy)
    if xy.ndim != 2 or xy.shape[1] != 2:
        raise ValueError(f"Expected an (N, 2) array, got {xy.shape}.")
    if not 1 <= bits <= 16:
        raise ValueError(f"`bits` must be in 1..16, got {bits}.")

    n = len(xy)
    keys = np.zeros(n, dtype=np.uint32)
    if not n:
        return keys

    finite = np.isfinite(xy).all(axis=1)
    n_finite = int(finite.sum())
    if not n_finite:
        return keys
    all_finite = n_finite == n

    pts = xy if all_finite else xy[finite]
    lo = pts.min(axis=0)
    hi = pts.max(axis=0)
    span = np.where(hi > lo, hi - lo, 1.0)

    scale = float((1 << bits) - 1)
    grid = ((pts - lo) / span) * scale
    # `span` can still be 0 along a degenerate axis once cast to float32.
    grid = np.nan_to_num(grid, nan=0.0, posinf=scale, neginf=0.0)
    grid = np.clip(grid, 0.0, scale).astype(np.uint32)

    interleaved = _part1by1(grid[:, 0]) | (_part1by1(grid[:, 1]) << np.uint32(1))
    if all_finite:
        return interleaved
    keys[finite] = interleaved
    return keys


def chunk_count(n, target, min_size):
    """How many chunks to split ``n`` points into.

    Chunking costs one draw call and a little per-frame bookkeeping per chunk,
    so it only pays off once there are enough points to be worth culling:
    below ``min_size`` points this returns 1.
    """
    if n <= 0:
        return 0
    if target <= 1 or min_size <= 0:
        return 1
    return int(min(target, max(1, n // min_size)))


def group_chunk_target(target, n_group, total):
    """Share a chunk budget across groups in proportion to their size.

    Points are chunked per marker group (each ``gfx.Points`` carries a single
    marker), so the budget has to be divided up or the total number of visuals
    would scale with the number of markers. Rounding means the parts can exceed
    ``target`` by up to half a chunk per group, which is near enough.
    """
    if total <= 0:
        return 1
    return max(1, round(target * n_group / total))


def spatial_chunks(xy, target, min_size, bits=MORTON_BITS, rng=None):
    """Order points by locality and cut them into roughly equal-sized chunks.

    Parameters
    ----------
    xy :        (N, 2) array
                Point coordinates.
    target :    int
                Upper bound on the number of chunks. Required, so that the
                chunking policy lives in one place (the figure) rather than
                being half-specified by defaults here. Per-object overhead grows
                with the chunk count (measurably past ~1000), so this caps it
                rather than scaling freely with N.
    min_size :  int
                Do not chunk at all below this many points.
    bits :      int
                Bits per axis for the Morton key.
    rng :       np.random.Generator, optional
                Shuffle the points *within* each chunk. Chunk membership is
                unaffected, but a prefix of a chunk then becomes an unbiased
                random sample of it - which is what lets the renderer draw
                only part of a chunk without biasing what is shown.

    Returns
    -------
    order :     (N,) intp array | None
                Point indices, grouped so that each chunk is a contiguous
                slice. Only chunk *membership* is part of the contract -
                nearby points share a chunk; the order *within* a chunk is
                unspecified. It is Z-order, or random within each chunk when
                `rng` is given (see `rng`).
                ``None`` when the points are left in their original order as a
                single chunk - callers can then avoid materializing an
                identity permutation.
    starts :    (n_chunks + 1,) intp array
                Chunk boundaries into ``order``: chunk ``i`` is
                ``order[starts[i]:starts[i + 1]]``.
    """
    n = len(xy)
    n_chunks = chunk_count(n, target, min_size)

    if n_chunks <= 1 and rng is None:
        return None, np.array([0, n], dtype=np.intp)

    order = (
        # One chunk: no grouping to do, but still a permutation to shuffle.
        np.arange(n, dtype=np.intp)
        if n_chunks <= 1
        else _argsort_keys(morton_keys(xy, bits=bits))
    )
    # `linspace` spreads the remainder across chunks instead of leaving a
    # short one at the end.
    starts = np.linspace(0, n, max(n_chunks, 1) + 1).astype(np.intp)

    if rng is not None:
        for i in range(len(starts) - 1):
            # Shuffles the slice in place, so membership is untouched.
            rng.shuffle(order[starts[i] : starts[i + 1]])

    return order, starts


def _argsort_keys(keys):
    """Stable argsort of uint32 keys.

    ``np.argsort(kind="stable")`` does not take a radix path for this dtype and
    measures ~5x slower than packing each key together with its index into a
    uint64 and sorting that. Carrying the index in the low bits also makes ties
    resolve by index, which is exactly what "stable" means, so the result is
    identical.
    """
    n = len(keys)
    if n >= 1 << 32:  # indices no longer fit in the low 32 bits
        return np.argsort(keys, kind="stable")
    packed = (keys.astype(np.uint64) << np.uint64(32)) | np.arange(
        n, dtype=np.uint64
    )
    # The packed values are unique (the index is in the low bits), so the
    # default introsort is already stable *for this data* and beats the
    # mergesort `kind="stable"` would force.
    packed.sort()
    # Masking leaves values below 2**32, so the indices can be reinterpreted
    # in place rather than cast into another full-size array.
    packed &= np.uint64(0xFFFFFFFF)
    return packed.view(np.intp)


def chunk_bounds(xyz_ordered, starts):
    """Axis-aligned bounds per chunk, as ``(n_chunks, 4)`` xmin/ymin/xmax/ymax.

    Takes the whole ``(N, 3)`` position buffer, *already* in chunk order (see
    :func:`spatial_chunks`), so each chunk is a contiguous slice and the whole
    thing is two reductions rather than a gather per chunk.

    Non-finite coordinates are ignored; a chunk with no finite point at all
    gets NaN bounds, which never intersects a view rectangle.
    """
    xyz_ordered = np.asarray(xyz_ordered)
    n_chunks = len(starts) - 1
    bounds = np.full((n_chunks, 4), np.nan)
    if n_chunks <= 0 or not len(xyz_ordered):
        return bounds

    xy_ordered = xyz_ordered[:, :2]
    offsets = starts[:-1]

    # Reduce first and check the (tiny) result, rather than probing the whole
    # input: `np.isfinite(x).all()` materializes a full-size bool array, and
    # this runs per frame while points are moving. A non-finite coordinate
    # shows up in the reduction, because min/max propagate it.
    mins = np.minimum.reduceat(xy_ordered, offsets, axis=0)
    maxs = np.maximum.reduceat(xy_ordered, offsets, axis=0)

    if not (np.isfinite(mins).all() and np.isfinite(maxs).all()):
        # Rare: redo it ignoring the non-finite coordinates, which would
        # otherwise poison their whole chunk's extent.
        finite = np.isfinite(xy_ordered)
        mins = np.minimum.reduceat(
            np.where(finite, xy_ordered, np.inf), offsets, axis=0
        )
        maxs = np.maximum.reduceat(
            np.where(finite, xy_ordered, -np.inf), offsets, axis=0
        )

    # A chunk with no finite point at all reduces to +inf/-inf; leave it NaN.
    # `mins` is +inf exactly when `maxs` is -inf, so one test covers both.
    ok = np.isfinite(mins).all(axis=1)
    bounds[ok, :2] = mins[ok]
    bounds[ok, 2:] = maxs[ok]
    return bounds


def visible_chunks(bounds, view):
    """Mask of chunks whose extent intersects the view rectangle.

    Parameters
    ----------
    bounds : (n_chunks, 4) array
             xmin/ymin/xmax/ymax per chunk (see :func:`chunk_bounds`).
    view :   (xmin, ymin, xmax, ymax)
             The viewport in world coordinates.

    Returns
    -------
    visible : (n_chunks,) bool array
              A chunk with NaN bounds (no finite point) compares False
              throughout and is excluded.
    """
    xmin, ymin, xmax, ymax = view
    with np.errstate(invalid="ignore"):  # NaN bounds compare False, as intended
        return (
            (bounds[:, 0] <= xmax)
            & (bounds[:, 2] >= xmin)
            & (bounds[:, 1] <= ymax)
            & (bounds[:, 3] >= ymin)
        )


def draw_counts(sizes, visible, budget, min_per_chunk=1):
    """How many points to draw from each chunk, within a total budget.

    Chunks outside the view draw nothing. If the visible chunks together hold
    no more than `budget` points they are drawn in full; otherwise each gets a
    share in proportion to its size. Points within a chunk are in random order
    (see :func:`spatial_chunks`), so drawing a prefix samples it without bias
    - and as the view narrows, fewer chunks share the budget and the surviving
    ones are drawn more densely, until everything in view is drawn.

    Parameters
    ----------
    sizes :         (n_chunks,) int array
    visible :       (n_chunks,) bool array
    budget :        int
                    Maximum number of points to draw in total.
    min_per_chunk : int
                    Floor for a visible chunk, so that a chunk in view never
                    disappears entirely just because many others share the
                    budget.

    Returns
    -------
    counts : (n_chunks,) int64 array
    """
    sizes = np.asarray(sizes)
    counts = np.zeros(len(sizes), dtype=np.int64)

    vis_sizes = sizes[visible]
    if not len(vis_sizes):
        return counts

    total = int(vis_sizes.sum())
    if total <= budget:
        # Not a separate rule - the proportional share below would also come
        # out at the full size here - just the common case, done without a
        # divide and two extra passes.
        counts[visible] = vis_sizes
        return counts

    share = (budget * (vis_sizes / total)).astype(np.int64)
    # The floor keeps a visible chunk on screen, but it cannot be allowed to
    # push the total past the budget - with many visible chunks and a small
    # budget, `min_per_chunk` each would do exactly that.
    floor = min(min_per_chunk, max(budget // len(vis_sizes), 1))
    # Never below the floor, never more than the chunk actually holds.
    counts[visible] = np.minimum(np.maximum(share, floor), vis_sizes)
    return counts


#: Monotonic, process-wide, so that a *new* layout never compares equal to the
#: one it replaced (which `id()` could, once the old object is collected).
_LAYOUT_VERSIONS = itertools.count()


@dataclass
class ChunkLayout:
    """How a point cloud is laid out across chunks, for the renderer to use.

    The three arrays are produced together, consumed together and have to be
    invalidated together, so they travel as one object rather than as parallel
    attributes on the figure.

    Attributes
    ----------
    order :  (M,) intp array
             Point index per drawn position, chunks back to back.
    starts : (n_chunks + 1,) intp array
             Offsets into `order`, aligned with the chunk visuals.
    bounds : (n_chunks, 4) array
             Each chunk's xmin/ymin/xmax/ymax, so a view test is one
             vectorised pass rather than a loop over visuals.
    version : int
             Changes whenever `bounds` is refreshed or a new layout is built.
    """

    order: np.ndarray
    starts: np.ndarray
    bounds: np.ndarray
    #: Bumped whenever anything here changes, so consumers can cache on it
    #: instead of being told to invalidate.
    version: int = field(default_factory=lambda: next(_LAYOUT_VERSIONS))

    @property
    def n_chunks(self):
        return max(len(self.starts) - 1, 0)

    @property
    def sizes(self):
        """Number of points in each chunk."""
        return np.diff(self.starts)

    def draw_counts(self, view, budget, min_per_chunk=1):
        """How many points to draw from each chunk for a given viewport.

        Composes :func:`visible_chunks` and :func:`draw_counts` so that callers
        state the policy once and never have to touch `bounds` or `sizes`.
        """
        return draw_counts(
            self.sizes,
            visible_chunks(self.bounds, view),
            budget,
            min_per_chunk=min_per_chunk,
        )

    def slice_for(self, i):
        """The slice of `order` (and of any draw-order buffer) for chunk `i`."""
        return slice(self.starts[i], self.starts[i + 1])

    def refresh_bounds(self, xyz_in_draw_order):
        """Recompute `bounds` after the points have moved."""
        self.bounds = chunk_bounds(xyz_in_draw_order, self.starts)
        self.version = next(_LAYOUT_VERSIONS)
