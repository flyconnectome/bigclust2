"""Greedy label placement for dense scatter plots.

This module is pure numpy (no Qt/pygfx imports) so it can be unit-tested
headlessly. The entry point is :func:`solve_label_placement`, which assigns
each visible point's label to one of a small number of candidate "slots"
around the point such that labels overlap neither point markers nor each
other. Labels for which no free slot exists are reported as suppressed
(slot ``-1``) and should be hidden by the caller.

Because both points and labels in the scatter plot are sized in world units,
whether two labels overlap does not depend on the camera: a solution stays
valid under arbitrary pan/zoom until the set of visible labels - or the
point/label/text geometry itself - changes. ``ScatterFigure`` exploits this
by caching solutions and only re-solving when needed.
"""

import numpy as np

# Candidate slots around a point, in order of preference. Slot 0 (right,
# vertically centered) matches the legacy fixed label position.
SLOT_RIGHT = 0
SLOT_TOP_RIGHT = 1
SLOT_BOTTOM_RIGHT = 2
SLOT_TOP_LEFT = 3
SLOT_BOTTOM_LEFT = 4
SLOT_LEFT = 5
SLOT_TOP = 6
SLOT_BOTTOM = 7
N_SLOTS = 8

_SLOT_NAMES = (
    "right",
    "top-right",
    "bottom-right",
    "top-left",
    "bottom-left",
    "left",
    "top",
    "bottom",
)


def slot_name(slot):
    """Human-readable name for a candidate slot, e.g. "left" or "top (ring 1)"."""
    if slot < 0:
        return "unplaced"
    ring, base = divmod(int(slot), N_SLOTS)
    return _SLOT_NAMES[base] if ring == 0 else f"{_SLOT_NAMES[base]} (ring {ring})"


def slot_box(slot, px, py, r, w, h, pad):
    """Return the label box ``(x0, y0, x1, y1)`` for a candidate slot.

    Coordinates are world units with y pointing up. ``(px, py)`` is the point
    position, ``r`` the point marker radius, ``(w, h)`` the label extent and
    ``pad`` the gap between the marker edge and the label box.

    Slots >= N_SLOTS repeat the eight base directions in successively
    farther rings: ``slot = ring * N_SLOTS + base``, with the gap growing by
    ``3 * pad`` per ring.
    """
    ring, base = divmod(slot, N_SLOTS)
    if ring < 0:
        raise ValueError(f"Invalid slot {slot}.")
    d = r + pad * (1 + 3 * ring)
    dd = d * 0.7071  # diagonal offset such that the box corner sits at ~d
    if base == SLOT_RIGHT:
        return (px + d, py - h / 2, px + d + w, py + h / 2)
    elif base == SLOT_TOP_RIGHT:
        return (px + dd, py + dd, px + dd + w, py + dd + h)
    elif base == SLOT_BOTTOM_RIGHT:
        return (px + dd, py - dd - h, px + dd + w, py - dd)
    elif base == SLOT_TOP_LEFT:
        return (px - dd - w, py + dd, px - dd, py + dd + h)
    elif base == SLOT_BOTTOM_LEFT:
        return (px - dd - w, py - dd - h, px - dd, py - dd)
    elif base == SLOT_LEFT:
        return (px - d - w, py - h / 2, px - d, py + h / 2)
    elif base == SLOT_TOP:
        return (px - w / 2, py + d, px + w / 2, py + d + h)
    elif base == SLOT_BOTTOM:
        return (px - w / 2, py - d - h, px + w / 2, py - d)
    raise ValueError(f"Invalid slot {slot}.")


class _Grid:
    """Uniform-grid spatial hash over axis-aligned boxes.

    Each box is stored with an ``owner`` id; queries can skip a given owner
    (used to ignore a label's own point marker).
    """

    def __init__(self, cell_size):
        self.cell = max(float(cell_size), 1e-12)
        self.cells = {}

    def _cell_range(self, box):
        c = self.cell
        return (
            int(np.floor(box[0] / c)),
            int(np.floor(box[1] / c)),
            int(np.floor(box[2] / c)),
            int(np.floor(box[3] / c)),
        )

    def add(self, box, owner=None):
        i0, j0, i1, j1 = self._cell_range(box)
        for i in range(i0, i1 + 1):
            for j in range(j0, j1 + 1):
                self.cells.setdefault((i, j), []).append((box, owner))

    def collides(self, box, skip_owner=None):
        i0, j0, i1, j1 = self._cell_range(box)
        for i in range(i0, i1 + 1):
            for j in range(j0, j1 + 1):
                for other, owner in self.cells.get((i, j), ()):
                    if owner is not None and owner == skip_owner:
                        continue
                    if (
                        box[0] < other[2]
                        and box[2] > other[0]
                        and box[1] < other[3]
                        and box[3] > other[1]
                    ):
                        return True
        return False

    def blockers(self, box, skip_owner=None):
        """Return the owners of all stored boxes overlapping `box`.

        Debugging helper - like `collides` but exhaustive, returning the
        (deduplicated) owners of everything in the way.
        """
        i0, j0, i1, j1 = self._cell_range(box)
        hits = []
        for i in range(i0, i1 + 1):
            for j in range(j0, j1 + 1):
                for other, owner in self.cells.get((i, j), ()):
                    if owner is not None and owner == skip_owner:
                        continue
                    if owner in hits:
                        continue
                    if (
                        box[0] < other[2]
                        and box[2] > other[0]
                        and box[1] < other[3]
                        and box[3] > other[1]
                    ):
                        hits.append(owner)
        return hits


def solve_label_placement(
    points,
    radii,
    extents,
    priority=None,
    prev_slots=None,
    pad=None,
    rings=1,
    bounds=None,
    obstacles=None,
    obstacle_radii=None,
    obstacle_boxes=None,
    anchor_obstacles=True,
    debug=None,
):
    """Greedily assign non-overlapping positions to point labels.

    Labels are placed in priority order; each label takes the first candidate
    slot (see :func:`slot_box`) that collides with neither a point marker nor
    an already-placed label. Point markers are approximated by their bounding
    squares; a label never collides with its own marker.

    Parameters
    ----------
    points :    (N, 2) array
                World positions of the visible points. Every point is both an
                obstacle and the anchor of one label.
    radii :     float | (N,) array
                World radius of each point marker.
    extents :   (N, 2) array
                (width, height) of each point's label in world units.
    priority :  (N,) array, optional
                Lower values are placed first (e.g. 0 for selected points).
                Ties preserve index order.
    prev_slots : (N,) int array, optional
                Slot each label occupied in a previous solution (-1 = none).
                A label's previous slot is tried first, which keeps placements
                stable across successive re-solves.
    pad :       float, optional
                Gap between marker edge and label box. Defaults to 20% of the
                median label height.
    rings :     int, optional
                Number of concentric candidate rings to try (8 slots each,
                see :func:`slot_box`). Extra rings help in crowded scenes at
                the cost of labels sitting farther from their anchor.
    bounds :    (x0, y0, x1, y1) tuple, optional
                If given, candidate boxes must lie fully inside this
                rectangle (e.g. the current viewport, so labels don't stick
                out of view). Callers caching solutions should treat this as
                a solve-time snapshot - it is not re-checked on pan/zoom.
    obstacles : (K, 2) array, optional
                Additional positions labels must avoid - e.g. the individual
                points when the anchors are whole groups.
    obstacle_radii : float | (K,) array, optional
                World radius of each obstacle.
    obstacle_boxes : (K, 4) array, optional
                Additional axis-aligned boxes to avoid - e.g. labels already
                placed by an earlier solve.
    anchor_obstacles : bool, optional
                Whether the anchor markers themselves act as obstacles
                (default). Pass False when the anchors are coarse group
                discs and exact `obstacles` are supplied instead - a diffuse
                group's bounding disc would otherwise block far more space
                than its points actually occupy.
    debug :     list, optional
                Pass an empty list to collect per-label diagnostics: it is
                filled with one list per label containing a tuple
                ``(slot, kind, blocker)`` for every rejected candidate,
                where kind is "out-of-view" (blocker None) or "collision"
                (blocker is the anchor index of the marker in the way,
                ``("label", i)`` for label i's already-placed box, or
                ``("obstacle", k)`` for row k of `obstacles`).

    Returns
    -------
    slots :     (N,) int array
                Assigned slot per label; -1 means no free slot was found and
                the label should be hidden.
    offsets :   (N, 2) float array
                World offset from the point to the label anchor, assuming a
                ``middle-left`` anchored text visual (i.e. the anchor sits at
                the left edge, vertical center of the label box). Rows with
                slot -1 carry the preferred (right-hand) slot's offset, for
                callers that want to draw unplaceable labels anyway.
    """
    points = np.asarray(points, dtype=float)
    n = len(points)
    slots = np.full(n, -1, dtype=int)
    offsets = np.zeros((n, 2), dtype=float)
    if debug is not None:
        debug.extend([] for _ in range(n))
    if n == 0:
        return slots, offsets

    radii = np.broadcast_to(np.asarray(radii, dtype=float), (n,))
    extents = np.asarray(extents, dtype=float)
    heights = extents[:, 1]

    if pad is None:
        pad = 0.2 * np.median(heights)

    if obstacles is not None:
        obstacles = np.asarray(obstacles, dtype=float)
        obstacle_radii = np.broadcast_to(
            np.asarray(
                obstacle_radii if obstacle_radii is not None else 0.0, dtype=float
            ),
            (len(obstacles),),
        )

    # Cell size on the order of a couple of label heights works well: labels
    # span a handful of cells, points usually one. Also scale with the radii
    # of whatever is actually inserted - a cell size much smaller than the
    # inserted boxes makes grid insertion explode.
    if anchor_obstacles:
        size_radii = radii
    elif obstacles is not None and len(obstacles):
        size_radii = obstacle_radii
    else:
        size_radii = np.zeros(1)
    grid = _Grid(
        cell_size=max(2 * np.median(heights), 2 * np.median(size_radii), 1e-12)
    )

    # Point markers are obstacles for every label (except their own)
    if anchor_obstacles:
        for i in range(n):
            r = radii[i]
            grid.add(
                (
                    points[i, 0] - r,
                    points[i, 1] - r,
                    points[i, 0] + r,
                    points[i, 1] + r,
                ),
                owner=i,
            )

    # Extra obstacles (e.g. the individual points when anchors are groups)
    if obstacles is not None:
        for k in range(len(obstacles)):
            r = obstacle_radii[k]
            grid.add(
                (
                    obstacles[k, 0] - r,
                    obstacles[k, 1] - r,
                    obstacles[k, 0] + r,
                    obstacles[k, 1] + r,
                ),
                owner=("obstacle", k),
            )
    if obstacle_boxes is not None:
        for k, ob in enumerate(np.asarray(obstacle_boxes, dtype=float)):
            grid.add((ob[0], ob[1], ob[2], ob[3]), owner=("box", int(k)))

    if priority is not None:
        order = np.argsort(np.asarray(priority), kind="stable")
    else:
        order = np.arange(n)

    n_slots = N_SLOTS * max(int(rings), 1)
    for i in order:
        # Plain int: np.int64 vs tuple owners would numpy-broadcast in the
        # grid's owner comparison
        i = int(i)
        px, py = points[i]
        candidates = list(range(n_slots))
        if prev_slots is not None and 0 <= prev_slots[i] < n_slots:
            # Hysteresis: prefer the slot this label had last time
            candidates.remove(prev_slots[i])
            candidates.insert(0, prev_slots[i])

        for slot in candidates:
            box = slot_box(slot, px, py, radii[i], extents[i, 0], extents[i, 1], pad)
            if bounds is not None and not (
                box[0] >= bounds[0]
                and box[1] >= bounds[1]
                and box[2] <= bounds[2]
                and box[3] <= bounds[3]
            ):
                if debug is not None:
                    debug[i].append((slot, "out-of-view", None))
                continue
            if debug is not None:
                hits = grid.blockers(box, skip_owner=i)
                if hits:
                    debug[i].append((slot, "collision", hits[0]))
                    continue
            elif grid.collides(box, skip_owner=i):
                continue
            slots[i] = slot
            offsets[i, 0] = box[0] - px
            offsets[i, 1] = (box[1] + box[3]) / 2 - py
            # Placed labels are obstacles for later labels
            grid.add(box, owner=("label", i))
            break
        else:
            # No free slot: report the preferred slot's offset anyway (slot
            # stays -1 and the box is not added as an obstacle)
            box = slot_box(
                SLOT_RIGHT, px, py, radii[i], extents[i, 0], extents[i, 1], pad
            )
            offsets[i, 0] = box[0] - px
            offsets[i, 1] = (box[1] + box[3]) / 2 - py

    return slots, offsets


def spatial_components(points, threshold):
    """Single-linkage clustering: component ids for points within reach.

    Two points belong to the same component if they are connected by a chain
    of points with pairwise distances <= `threshold`. O(N^2), intended for
    small N (a few hundred). Component ids are assigned in first-occurrence
    order, so the result is deterministic.

    Returns
    -------
    (N,) int array of component ids (0-based).
    """
    points = np.asarray(points, dtype=float)
    n = len(points)
    comp = np.full(n, -1, dtype=int)
    if n == 0:
        return comp

    deltas = points[:, None, :] - points[None, :, :]
    adjacent = (deltas**2).sum(axis=-1) <= threshold**2

    current = 0
    for i in range(n):
        if comp[i] >= 0:
            continue
        comp[i] = current
        stack = [i]
        while stack:
            j = stack.pop()
            neighbours = np.where(adjacent[j] & (comp < 0))[0]
            comp[neighbours] = current
            stack.extend(neighbours.tolist())
        current += 1
    return comp


def connector_offsets(slot, w, h, r, gap=0.0):
    """Return the endpoints of a connector line from a point to its label.

    Parameters
    ----------
    slot :  int
            The slot the label occupies (see :func:`slot_box`).
    w, h :  float
            Extents of the label box.
    r :     float
            Radius of the point marker.
    gap :   float, optional
            Standoff between the marker edge and the start of the line.

    Returns
    -------
    start : (2,) tuple
            Offset of the line's start from the *point*: just outside the
            marker edge (``r + gap`` from the center), pointing towards the
            slot.
    end :   (2,) tuple
            Offset of the line's end from the *label's anchor* (the middle
            of the label box's left edge, matching a ``middle-left`` anchored
            text visual): the point on the label box edge nearest the marker.
    """
    if slot < 0:
        raise ValueError(f"Invalid slot {slot}.")
    slot = slot % N_SLOTS  # ring slots keep their base direction
    s = r + gap
    d = 0.7071
    if slot == SLOT_RIGHT:
        return (s, 0.0), (0.0, 0.0)
    elif slot == SLOT_TOP_RIGHT:
        return (s * d, s * d), (0.0, -h / 2)
    elif slot == SLOT_BOTTOM_RIGHT:
        return (s * d, -s * d), (0.0, h / 2)
    elif slot == SLOT_TOP_LEFT:
        return (-s * d, s * d), (w, -h / 2)
    elif slot == SLOT_BOTTOM_LEFT:
        return (-s * d, -s * d), (w, h / 2)
    elif slot == SLOT_LEFT:
        return (-s, 0.0), (w, 0.0)
    elif slot == SLOT_TOP:
        return (0.0, s), (w / 2, -h / 2)
    elif slot == SLOT_BOTTOM:
        return (0.0, -s), (w / 2, h / 2)
    raise ValueError(f"Invalid slot {slot}.")


def estimate_text_wh(text, font_size):
    """Estimate the (width, height) of a text in world units.

    Used for labels whose visual has not been created (and laid out) yet;
    supplanted by exact measurements (see :func:`measure_text_wh`) as soon
    as they are available.
    """
    lines = text.splitlines() or [text]
    return (
        0.60 * font_size * max(len(line) for line in lines),
        1.25 * font_size * len(lines),
    )


def measure_text_wh(vis, font_size):
    """Measure the (width, height) of a laid-out pygfx Text visual.

    Reads the per-block layout rectangles (same trick as
    ``draw_bounding_box_around_text``). For world-space text these are in
    world units. Returns None if the visual has no (plausible) layout yet -
    e.g. because pygfx changed internals or layout has not run - in which
    case the caller should fall back to :func:`estimate_text_wh`.
    """
    try:
        blocks = vis._text_blocks
        positions = vis.geometry.positions.data
        if not len(blocks):
            return None
        left, right = np.inf, -np.inf
        top, bottom = -np.inf, np.inf
        for i, block in enumerate(blocks):
            pos_x, pos_y = positions[i][:2]
            left = min(left, pos_x + block._rect.left)
            right = max(right, pos_x + block._rect.right)
            top = max(top, pos_y + block._rect.top)
            bottom = min(bottom, pos_y + block._rect.bottom)
        w = float(right - left)
        h = float(top - bottom)
    except Exception:
        return None

    # Sanity check against the font size - if the numbers are implausible
    # (pygfx layout units changed?) rather use the estimate.
    if not (np.isfinite(w) and np.isfinite(h)) or w <= 0 or h <= 0:
        return None
    if not (0.4 * font_size <= h <= 6 * font_size * len(blocks)):
        return None
    return (w, h)
