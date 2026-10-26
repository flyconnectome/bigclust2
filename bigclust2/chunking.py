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

from .utils import points_in_rect

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
        else argsort_keys(morton_keys(xy, bits=bits))
    )
    # `linspace` spreads the remainder across chunks instead of leaving a
    # short one at the end.
    starts = np.linspace(0, n, max(n_chunks, 1) + 1).astype(np.intp)

    if rng is not None:
        for i in range(len(starts) - 1):
            # Shuffles the slice in place, so membership is untouched.
            rng.shuffle(order[starts[i] : starts[i + 1]])

    return order, starts


def argsort_keys(keys):
    """Stable argsort of non-negative integer keys that fit in 32 bits.

    ``np.argsort(kind="stable")`` does not take a radix path for this dtype and
    measures ~5x slower than packing each key together with its index into a
    uint64 and sorting that (4.6s -> 0.5s over 20M label codes). Carrying the
    index in the low bits also makes ties resolve by index, which is exactly
    what "stable" means, so the result is identical.

    Falls back to ``np.argsort`` when the keys or the indices do not fit. A
    uint32 input is taken on trust - that is the Morton-key caller, and a range
    check would be a wasted pass over every point.
    """
    n = len(keys)
    if n >= 1 << 32:  # indices no longer fit in the low 32 bits
        return np.argsort(keys, kind="stable")
    # uint32 is taken on trust - that is the Morton-key caller, and the range
    # check would be a wasted pass over every point. For anything else it costs
    # ~1% of the sort it is guarding.
    if keys.dtype != np.uint32 and n and (keys.min() < 0 or keys.max() >= 1 << 32):
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


def contained_chunks(bounds, view):
    """Mask of chunks lying entirely inside the view rectangle.

    A subset of :func:`visible_chunks`. Useful when a caller needs an exact
    count of the points on screen: every point of a contained chunk is on
    screen, so those chunks can be counted from `sizes` alone and only the
    chunks straddling the viewport edge have to be tested point by point.
    """
    # A box is contained exactly when both its corners are - which is the same
    # point-in-rectangle test the figure applies to the points themselves.
    return points_in_rect(bounds[:, :2], view) & points_in_rect(bounds[:, 2:], view)


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

    exact = budget * (vis_sizes / total)
    share = exact.astype(np.int64)
    # Truncating to an integer loses up to a point per chunk, and with many
    # chunks sharing a small budget every share truncates to zero and the
    # scatter goes blank. Hand the lost points to the largest fractions, which
    # spends the budget exactly.
    remainder = budget - int(share.sum())
    if remainder > 0:
        share[np.argsort(share - exact)[:remainder]] += 1

    # The floor keeps a visible chunk on screen, but it cannot be allowed to
    # push the total past the budget, so cap it at what every visible chunk
    # could draw and still fit. A budget smaller than the number of visible
    # chunks leaves no floor at all - there is simply nothing to go round.
    floor = min(min_per_chunk, budget // len(vis_sizes))
    # Never below the floor, never more than the chunk actually holds.
    drawn = np.minimum(np.maximum(share, floor), vis_sizes)

    # The proportional shares already add up to the budget, so raising the
    # small chunks to the floor overshoots it. Take the excess back from the
    # chunks that have room above the floor, in proportion to that room. A
    # chunk holding less than the floor has no room - clamping at zero keeps
    # it from being handed points it does not have.
    excess = int(drawn.sum()) - budget
    if excess > 0:
        headroom = np.maximum(drawn - floor, 0)
        available = int(headroom.sum())
        if available > 0:
            # Rounding up so the give-back always covers the excess; that can
            # land a point or two under budget, never over.
            give = np.ceil(headroom * (excess / available)).astype(np.int64)
            drawn -= np.minimum(headroom, give)

    counts[visible] = drawn
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
    structure : int
             Changes only when a new layout is built. `refresh_bounds` moves
             the boxes but leaves `order` and `starts` alone, so anything
             keyed on *which points are in which chunk* stays valid across a
             refresh and should key on this instead of on `version`.
    """

    order: np.ndarray
    starts: np.ndarray
    bounds: np.ndarray
    #: Bumped whenever anything here changes, so consumers can cache on it
    #: instead of being told to invalidate.
    version: int = field(default_factory=lambda: next(_LAYOUT_VERSIONS))
    #: Identifies the chunk *membership*, which a bounds refresh preserves.
    structure: int = field(default_factory=lambda: next(_LAYOUT_VERSIONS))

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

    def points_in_view(self, xy, view, limit):
        """Indices of the points of `xy` inside `view`, or ``None`` past `limit`.

        Callers that only act on a small number of visible points - the label
        system - do not need the set once it is bigger than they are willing to
        handle, so this stops counting there instead of testing every point.

        Chunks lying entirely inside the view are counted from `sizes` alone,
        which is what makes a zoomed-out view (where the answer is always "too
        many") cost a pass over the chunks rather than over the points. Only
        chunks straddling the edge are tested point by point; the work is
        bounded by how many points those chunks hold.

        `xy` must be in the original point order, as `order` indexes it.
        """
        # Containment implies visibility, so the cheap bail needs only `inside`.
        inside = contained_chunks(self.bounds, view)
        if self.sizes[inside].sum() > limit:
            return None

        found = []
        total = 0
        for i in np.flatnonzero(visible_chunks(self.bounds, view)):
            ix = self.points_in(i)
            if not inside[i]:
                ix = ix[points_in_rect(xy[ix], view)]
            total += len(ix)
            if total > limit:
                return None
            found.append(ix)

        out = np.concatenate(found) if found else np.empty(0, dtype=np.intp)
        # Chunks are visited in draw order, so sort to give callers point
        # order - which is what a plain mask test would have returned, and what
        # lets a caller cache on the result's bytes.
        out.sort()
        return out

    def subset_in_draw_order(self, mask):
        """Where a subset of the points sits in this layout's draw order.

        The renderer draws a prefix of each chunk, so anything that wants to
        annotate exactly what was drawn - the selection overlay - needs its own
        points ordered the same way. Laid out like this, the members of a
        chunk's drawn prefix are a prefix of the subset's chunk too, and
        picking them up is a `searchsorted` rather than a set intersection.

        Parameters
        ----------
        mask : (N,) bool array
               True for the points of interest, indexed by point number.
               A mask rather than indices because this walks `order`, and at
               tens of millions of points one bool per point is far cheaper
               than inverting the permutation.

        Returns
        -------
        ix :        (M,) intp array
                    The chosen point indices, in draw order.
        splits :    (n_chunks + 1,) intp array
                    Offsets into `ix`: chunk `i` holds
                    ``ix[splits[i]:splits[i+1]]``.
        positions : (M,) intp array
                    Each point's position in the full draw order, ascending.
                    Feed back to :meth:`prefix_counts` to find how many of them
                    a given set of per-chunk draw counts covers.
        """
        positions = np.flatnonzero(mask[self.order])
        ix = self.order[positions]
        # `side="left"`: a point sitting exactly on a chunk boundary is the
        # first point of that chunk, not the last of the one before.
        splits = np.searchsorted(positions, self.starts)
        return ix, splits, positions

    def prefix_counts(self, positions, offsets, chunks, counts):
        """How much of a subset falls inside each chunk's drawn prefix.

        The renderer draws the first `counts[i]` points of chunk `i`, and
        `positions` is ascending, so the subset's share of that prefix is found
        by one search rather than a pass per chunk.

        Parameters
        ----------
        positions : (M,) intp array
                    From :meth:`subset_in_draw_order`.
        offsets :   (V,) intp array
                    Where each wanted chunk's points start within `positions`
                    (``splits[chunks]``).
        chunks :    (V,) intp array
                    The chunks being asked about.
        counts :    (n_chunks,) int array
                    Points drawn per chunk, from :meth:`draw_counts`.

        Returns
        -------
        counts : (V,) intp array
                 How many of the subset's points each of `chunks` draws.
        """
        ends = np.searchsorted(positions, self.starts[chunks] + counts[chunks])
        return ends - offsets

    def points_in(self, i):
        """The original point indices making up chunk `i`.

        The one place that spells out how `order` and `starts` combine, so
        callers never have to index `order` themselves.
        """
        return self.order[self.slice_for(i)]

    def refresh_bounds(self, xyz_in_draw_order):
        """Recompute `bounds` after the points have moved."""
        self.bounds = chunk_bounds(xyz_in_draw_order, self.starts)
        self.version = next(_LAYOUT_VERSIONS)
