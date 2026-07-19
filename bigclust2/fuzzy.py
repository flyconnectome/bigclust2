"""Small fuzzy subsequence matcher used by the command palette.

Deliberately free of Qt (and any third-party) imports so it can be unit-tested
headlessly and reused anywhere. The scoring follows the same shape as editor
"fuzzy finders": a query matches if its characters appear in order (not
necessarily contiguously) in the candidate, and the score rewards matches that
are contiguous, start on a word boundary, and start early in the string.

Typical use::

    >>> rank("gs", ["Grow Selection", "Shrink Selection", "Show Hidden"])
    [('Grow Selection', ...), ('Show Hidden', ...)]
"""

from __future__ import annotations

# Scoring weights. Tuned so that a boundary-anchored acronym match (e.g. "gs"
# -> "Grow Selection") outranks an incidental in-word match of the same length.
_MATCH = 16.0  # awarded per matched character
_BONUS_BOUNDARY = 8.0  # match starts a word ("grow selection" <- "s")
_BONUS_CAMEL = 6.0  # match at a lower->upper transition ("KNNGraph" <- "G")
_BONUS_CONSECUTIVE = 10.0  # match directly follows the previous match
_PENALTY_GAP_START = -3.0  # first skipped character in a gap
_PENALTY_GAP_EXTEND = -1.0  # each further skipped character
_PENALTY_LEADING = -0.5  # per character skipped before the first match
_MAX_LEADING = 12  # cap the leading penalty so long breadcrumbs aren't buried
_PENALTY_LENGTH = -0.05  # per candidate character; breaks ties toward shorter

_NEG = float("-inf")

_BOUNDARY_BEFORE = frozenset(" \t\n_-/\\.:,;()[]{}<>|›»→")


def _bonuses(candidate: str) -> list[float]:
    """Per-position bonus for starting a match at that character."""
    out = []
    for i, ch in enumerate(candidate):
        if i == 0:
            out.append(_BONUS_BOUNDARY)
            continue
        prev = candidate[i - 1]
        if prev in _BOUNDARY_BEFORE:
            out.append(_BONUS_BOUNDARY)
        elif ch.isupper() and prev.islower():
            out.append(_BONUS_CAMEL)
        else:
            out.append(0.0)
    return out


def _is_subsequence(query: str, candidate: str) -> bool:
    """Cheap reject before running the (more expensive) scoring pass."""
    it = iter(candidate)
    return all(ch in it for ch in query)


def score(query: str, candidate: str) -> float | None:
    """Score how well ``query`` fuzzy-matches ``candidate``.

    Matching is case-insensitive and order-preserving. Returns ``None`` when the
    query is not a subsequence of the candidate; an empty query scores ``0.0``
    (matches everything), which lets callers show a full list before any typing.

    Higher is better. Scores are only meaningful relative to each other for the
    same query - do not compare across queries.
    """
    q = query.strip().lower()
    if not q:
        return 0.0
    if not candidate:
        return None

    cl = candidate.lower()
    n, m = len(q), len(cl)
    if n > m or not _is_subsequence(q, cl):
        return None

    bonuses = _bonuses(candidate)

    # dp[j] = best score for matching the query prefix processed so far, with
    # its final character landing exactly on candidate position j.
    prev = [_NEG] * m
    for j in range(m):
        if cl[j] == q[0]:
            prev[j] = _MATCH + bonuses[j] + _PENALTY_LEADING * min(j, _MAX_LEADING)

    for i in range(1, n):
        qc = q[i]
        cur = [_NEG] * m
        # `running` carries the best previous-row score reachable at j via a
        # gap, decayed as the gap widens. Keeping it as a rolling value is what
        # holds this loop at O(len(query) * len(candidate)).
        running = _NEG
        for j in range(1, m):
            gap_from_prev = prev[j - 1] + _PENALTY_GAP_START
            running = max(
                running + _PENALTY_GAP_EXTEND if running > _NEG else _NEG,
                gap_from_prev if prev[j - 1] > _NEG else _NEG,
            )
            if cl[j] != qc:
                continue
            consecutive = (
                prev[j - 1] + _BONUS_CONSECUTIVE if prev[j - 1] > _NEG else _NEG
            )
            base = max(consecutive, running)
            if base > _NEG:
                cur[j] = base + _MATCH + bonuses[j]
        prev = cur

    best = max(prev)
    if best == _NEG:
        return None
    return best + _PENALTY_LENGTH * m


def match_positions(query: str, candidate: str) -> list[int]:
    """Indices of ``candidate`` to highlight for ``query``.

    Greedy left-to-right, preferring word-boundary positions for each character
    so acronym-style queries light up the right letters. Used purely for
    rendering, so it does not need to agree with :func:`score` in corner cases;
    returns ``[]`` when there is no match.
    """
    q = query.strip().lower()
    if not q or not candidate:
        return []
    cl = candidate.lower()
    bonuses = _bonuses(candidate)

    positions: list[int] = []
    start = 0
    for ch in q:
        found = -1
        fallback = -1
        for j in range(start, len(cl)):
            if cl[j] != ch:
                continue
            if fallback < 0:
                fallback = j
            if bonuses[j] > 0:
                found = j
                break
        pick = found if found >= 0 else fallback
        if pick < 0:
            return []
        positions.append(pick)
        start = pick + 1
    return positions


def rank(query, items, key=None):
    """Filter and sort ``items`` by fuzzy match against ``query``.

    Args:
        query: The search string. Empty keeps every item, in input order.
        items: The candidates to rank.
        key: Optional callable mapping an item to the string to match against.
            Defaults to ``str``.

    Returns:
        A list of ``(item, score)`` pairs, best first. Ties keep their original
        relative order, so callers can pre-sort by their own secondary
        criterion (e.g. recency) and have it respected.
    """
    to_text = key if key is not None else str
    scored = []
    for idx, item in enumerate(items):
        s = score(query, to_text(item))
        if s is not None:
            scored.append((item, s, idx))
    scored.sort(key=lambda t: (-t[1], t[2]))
    return [(item, s) for item, s, _ in scored]
