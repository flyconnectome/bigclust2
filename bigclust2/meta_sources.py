"""Logic for updating a project's meta data from live annotation sources.

A project's meta data is a static snapshot that combines annotations from
several datasets. This module describes where each dataset's annotations come
from (a backend + config + column mapping), pulls fresh data, and merges it back
into the meta table **in place** -- same row order and count, only cell values
change. Reading only; backends are never written to here.

This module is intentionally Qt-free so the merge logic can be unit-tested
without a GUI.
"""

from __future__ import annotations

import logging

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .gui.widgets.annotation_backends import build_backend


logger = logging.getLogger(__name__)


# Columns that are never updated from a source: `id` is the join key, `dataset`
# is the partition key, and `_`-prefixed columns are internal (e.g. `_color`).
RESERVED_COLUMNS = ("id", "dataset")


def _is_reserved(column):
    """Whether a project column must never be a meta-source target."""
    text = str(column)
    return text in RESERVED_COLUMNS or text.startswith("_")


@dataclass(frozen=True)
class MetaSourceSpec:
    """A single dataset's meta-data source definition."""

    dataset: str
    backend: str
    config: dict = field(default_factory=dict)
    # Maps project meta column -> the source's column name.
    columns: dict = field(default_factory=dict)
    last_updated: str | None = None

    def to_dict(self):
        """Serialize for storage under ``info["meta"]["sources"][dataset]``."""
        return {
            "backend": self.backend,
            "config": dict(self.config),
            "columns": dict(self.columns),
            "last_updated": self.last_updated,
        }

    @classmethod
    def from_dict(cls, dataset, raw):
        """Build a spec from a stored entry, or None if it is malformed."""
        if not isinstance(raw, dict):
            return None

        backend = raw.get("backend", "")
        if not isinstance(backend, str) or not backend.strip():
            return None

        config = raw.get("config", {})
        if not isinstance(config, dict):
            config = {}

        columns = raw.get("columns", {})
        if not isinstance(columns, dict):
            columns = {}

        # Drop reserved targets defensively (id/dataset/_*).
        columns = {
            str(proj): str(src)
            for proj, src in columns.items()
            if not _is_reserved(proj) and str(src).strip()
        }

        last_updated = raw.get("last_updated")
        if last_updated is not None and not isinstance(last_updated, str):
            last_updated = str(last_updated)

        return cls(
            dataset=str(dataset),
            backend=backend.strip(),
            config={str(k): v for k, v in config.items()},
            columns=columns,
            last_updated=last_updated,
        )


def source_entry_datasets(key, raw):
    """Return the datasets covered by one ``sources`` entry.

    A single entry can serve several datasets that share the same setup (e.g.
    the left/right side of a brain). The datasets are taken from an explicit
    ``datasets`` list in the value when present, otherwise from a
    comma-separated key (``"left,right"``).
    """
    if isinstance(raw, dict):
        ds_list = raw.get("datasets")
        if isinstance(ds_list, list) and ds_list:
            return [str(d).strip() for d in ds_list if str(d).strip()]
    return [part.strip() for part in str(key).split(",") if part.strip()]


def meta_sources_mapping(meta):
    """Return the per-dataset sources mapping from a ``meta`` dict, or None.

    Accepts the canonical ``sources`` key and the singular ``source`` alias
    (only when it holds the per-dataset dict rather than a legacy URL string).
    """
    if not isinstance(meta, dict):
        return None
    sources = meta.get("sources")
    if isinstance(sources, dict):
        return sources
    legacy = meta.get("source")
    if isinstance(legacy, dict):
        return legacy
    return None


def parse_meta_sources(info):
    """Return the list of ``MetaSourceSpec`` declared in a project's info dict.

    Returns an empty list when ``info["meta"]`` is not a dict or declares no
    sources. Both the canonical ``sources`` key and the singular ``source``
    alias are accepted. Entries covering multiple datasets (comma-separated key
    or a ``datasets`` list) expand into one spec per dataset. Malformed entries
    are skipped.
    """
    if not isinstance(info, dict):
        return []

    sources = meta_sources_mapping(info.get("meta"))
    if not isinstance(sources, dict):
        return []

    specs = []
    for key, raw in sources.items():
        for dataset in source_entry_datasets(key, raw):
            spec = MetaSourceSpec.from_dict(dataset, raw)
            if spec is not None:
                specs.append(spec)
    return specs


def _spec_signature(spec):
    """Hashable signature identifying specs that share an identical setup."""
    return (
        spec.backend,
        tuple(sorted(spec.config.items())),
        tuple(sorted(spec.columns.items())),
        spec.last_updated,
    )


def serialize_meta_sources(specs):
    """Turn a list of specs into the ``{key: {...}}`` storage mapping.

    Specs that share an identical setup (backend, config, columns and
    last_updated) are grouped under a single comma-separated key, so e.g.
    left/right datasets with the same source collapse into one entry.
    """
    groups = []
    index = {}
    for spec in specs:
        sig = _spec_signature(spec)
        if sig in index:
            groups[index[sig]][0].append(spec.dataset)
        else:
            index[sig] = len(groups)
            groups.append(([spec.dataset], spec))

    out = {}
    for datasets, spec in groups:
        out[",".join(datasets)] = spec.to_dict()
    return out


def ensure_meta_dict(info):
    """Return ``info['meta']`` as a dict, converting a plain file-path string.

    The ``meta`` block may be stored as a bare file path string; this upgrades
    it to ``{"file": <str>}`` in place so additional keys (like ``sources``) can
    be attached.
    """
    meta = info.get("meta")
    if isinstance(meta, dict):
        return meta
    if isinstance(meta, str):
        meta = {"file": meta}
    else:
        meta = {}
    info["meta"] = meta
    return meta


def write_meta_sources(info, specs):
    """Store ``specs`` under ``info['meta']['sources']`` (in place).

    Writes the canonical ``sources`` key and migrates a dict-valued ``source``
    alias away so the two keys can't drift apart.
    """
    meta = ensure_meta_dict(info)
    meta["sources"] = serialize_meta_sources(specs)
    if isinstance(meta.get("source"), dict):
        del meta["source"]
    return info


def normalize_colname(name):
    """Collapse a column name to a comparable key.

    Lowercases and strips everything but alphanumerics so that ``"somaSide"``,
    ``"soma_side"`` and ``"SOMA SIDE"`` all map to ``"somaside"``.
    """
    return "".join(ch for ch in str(name).lower() if ch.isalnum())


def auto_match_columns(project_cols, source_cols):
    """Suggest a ``{project_col: source_col}`` mapping by name similarity.

    Three tiers, first match wins: exact, case-insensitive, then normalized
    (see :func:`normalize_colname`). Reserved project columns (``id``,
    ``dataset``, ``_*``) are never matched.
    """
    source_cols = [str(c) for c in source_cols]
    by_exact = {c: c for c in source_cols}
    by_lower = {}
    by_norm = {}
    for c in source_cols:
        by_lower.setdefault(c.lower(), c)
        by_norm.setdefault(normalize_colname(c), c)

    mapping = {}
    for proj in project_cols:
        if _is_reserved(proj):
            continue
        proj_str = str(proj)
        if proj_str in by_exact:
            mapping[proj_str] = by_exact[proj_str]
        elif proj_str.lower() in by_lower:
            mapping[proj_str] = by_lower[proj_str.lower()]
        elif normalize_colname(proj_str) in by_norm:
            mapping[proj_str] = by_norm[normalize_colname(proj_str)]
    return mapping


@dataclass
class DatasetUpdateReport:
    """Per-dataset outcome of a meta update."""

    dataset: str
    rows: int = 0
    ids_found: int = 0
    ids_missing: int = 0
    columns_updated: list = field(default_factory=list)
    columns_added: list = field(default_factory=list)
    columns_skipped: list = field(default_factory=list)
    columns_failed: list = field(default_factory=list)
    cells_changed: int = 0
    error: str | None = None


@dataclass
class UpdateReport:
    """Aggregate outcome of an :func:`update_meta` call."""

    datasets: list = field(default_factory=list)

    @property
    def cells_changed(self):
        return sum(d.cells_changed for d in self.datasets)

    @property
    def ids_missing(self):
        return sum(d.ids_missing for d in self.datasets)

    @property
    def errors(self):
        return [d for d in self.datasets if d.error]

    @property
    def columns_failed(self):
        cols = set()
        for d in self.datasets:
            cols.update(d.columns_failed)
        return sorted(cols)

    def columns_touched(self):
        """Sorted set of project columns updated or added across all datasets."""
        cols = set()
        for d in self.datasets:
            cols.update(d.columns_updated)
            cols.update(d.columns_added)
        return sorted(cols)

    def summary(self):
        """Human-readable one-paragraph summary."""
        n_ds = len([d for d in self.datasets if not d.error])
        parts = [
            f"Updated {self.cells_changed:,} cells across {n_ds} "
            f"dataset{'s' if n_ds != 1 else ''}."
        ]
        if self.ids_missing:
            parts.append(f"{self.ids_missing:,} ids not found in their source.")
        failed_cols = self.columns_failed
        if failed_cols:
            parts.append(f"Skipped columns: {', '.join(failed_cols)}.")
        if self.errors:
            names = ", ".join(d.dataset for d in self.errors)
            parts.append(f"Failed: {names}.")
        return " ".join(parts)


def _normalize_source_frame(src):
    """Return a source DataFrame with a de-duplicated string index."""
    src = src.copy()
    src.index = src.index.astype(str)
    src = src[~src.index.duplicated(keep="first")]
    return src


def _isna_scalar(value):
    """``pd.isna`` reduced to a single bool (arrays/lists count as not-NA)."""
    res = pd.isna(value)
    return bool(res) if isinstance(res, bool) else False


def _values_equal(a, b):
    """Scalar-safe equality that won't raise on array/list cell values."""
    try:
        res = a == b
    except Exception:  # noqa: BLE001 - exotic objects
        return a is b
    if isinstance(res, np.ndarray):
        return bool(res.all()) if res.size else True
    return bool(res)


def _count_changed(old, new):
    """NaN-aware count of differing values (``new`` is assumed all non-null).

    Falls back to an element-wise comparison when cells hold non-scalar values
    (e.g. arrays/lists, which some backends return), since the vectorized object
    comparison raises an ambiguous-truth-value error on those.
    """
    try:
        old_obj = old.astype(object).where(old.notna(), other=pd.NA)
        new_obj = new.astype(object)
        changed = old_obj.isna() | (old_obj != new_obj)
        return int(changed.sum())
    except (ValueError, TypeError):
        count = 0
        for o, n in zip(old.tolist(), new.tolist()):
            if _isna_scalar(o) and _isna_scalar(n):
                continue
            if not _values_equal(o, n):
                count += 1
        return count


def _assign_column(out, project_col, target_index, new_vals):
    """Assign ``new_vals`` into ``out[project_col]`` at ``target_index``.

    Coerces to the existing column dtype when possible, upcasting the column to
    ``object`` rather than raising on a dtype clash.
    """
    existing_dtype = out[project_col].dtype
    try:
        out.loc[target_index, project_col] = new_vals.astype(existing_dtype)
    except (ValueError, TypeError):
        out[project_col] = out[project_col].astype(object)
        out.loc[target_index, project_col] = new_vals.astype(object)


def _update_one_column(out, ds_report, ids_str, src, project_col, source_col):
    """Merge one source column into ``out`` for the current dataset's rows."""
    # Source values aligned to this dataset's masked rows; ids that are absent
    # (or whose source value is null) become NaN and are skipped.
    mapped = ids_str.map(src[source_col])
    found = mapped.notna()
    if not found.any():
        return

    target_index = mapped.index[found]
    new_vals = mapped[found]

    if project_col not in out.columns:
        out[project_col] = pd.NA
        ds_report.columns_added.append(project_col)
        ds_report.cells_changed += int(len(new_vals))
        _assign_column(out, project_col, target_index, new_vals)
    else:
        old_vals = out.loc[target_index, project_col]
        ds_report.cells_changed += _count_changed(old_vals, new_vals)
        _assign_column(out, project_col, target_index, new_vals)
        ds_report.columns_updated.append(project_col)


def update_meta(meta_df, specs, *, backends=None, progress_callback=None):
    """Pull fresh annotations and merge them into a copy of ``meta_df``.

    The returned DataFrame has the same length, row order and index as
    ``meta_df``; only cell values change (and mapped-but-new columns are added,
    initialized to NA where ids don't match). Updates are scoped per dataset via
    the ``dataset`` column, so ids repeated across datasets never collide.

    Parameters
    ----------
    meta_df : pandas.DataFrame
        Project meta data; must have ``id`` and ``dataset`` columns.
    specs : list[MetaSourceSpec]
        One source definition per dataset to refresh.
    backends : dict, optional
        Pre-built ``{dataset: AnnotationBackend}`` instances (e.g. already
        validated by the UI). Missing datasets are built via ``build_backend``.
    progress_callback : callable, optional
        Called as ``progress_callback(done, total, dataset)`` before each source
        is read, and once at the end with ``done == total`` and ``dataset=""``.

    Returns
    -------
    (pandas.DataFrame, UpdateReport)
    """
    out = meta_df.copy()
    report = UpdateReport()
    backends = backends or {}
    total = len(specs)

    for i, spec in enumerate(specs):
        if progress_callback is not None:
            progress_callback(i, total, spec.dataset)

        ds_report = DatasetUpdateReport(dataset=spec.dataset)
        report.datasets.append(ds_report)

        mask = out["dataset"].astype(str) == str(spec.dataset)
        ds_report.rows = int(mask.sum())
        if ds_report.rows == 0:
            continue
        if not spec.columns:
            continue

        try:
            backend = backends.get(spec.dataset)
            if backend is None:
                backend = build_backend(spec.backend, spec.config)

            ids = out.loc[mask, "id"]
            src = backend.read_annotations(ids=ids.tolist())
            if src is None or len(src) == 0:
                src = pd.DataFrame()
            else:
                src = _normalize_source_frame(src)
        except Exception as exc:  # noqa: BLE001 - surfaced in the report
            detail = str(exc).strip() or exc.__class__.__name__
            ds_report.error = detail
            logger.warning(
                "Meta update failed for %s: %s",
                spec.dataset,
                detail,
                exc_info=True,
            )
            continue

        ids_str = out.loc[mask, "id"].astype(str)
        in_source = ids_str.isin(src.index)
        ds_report.ids_found = int(in_source.sum())
        ds_report.ids_missing = int((~in_source).sum())

        for project_col, source_col in spec.columns.items():
            if _is_reserved(project_col):
                continue
            if source_col not in src.columns:
                ds_report.columns_skipped.append(project_col)
                continue

            # A single bad column (e.g. odd dtypes) must not abort the rest of
            # the dataset's update.
            try:
                _update_one_column(out, ds_report, ids_str, src, project_col, source_col)
            except Exception as exc:  # noqa: BLE001 - surfaced in the report
                detail = str(exc).strip() or exc.__class__.__name__
                ds_report.columns_failed.append(project_col)
                logger.warning(
                    "Meta update: column '%s' failed for %s: %s",
                    project_col,
                    spec.dataset,
                    detail,
                    exc_info=True,
                )

    if progress_callback is not None:
        progress_callback(total, total, "")

    return out, report


# ---------------------------------------------------------------------------
# Local-file merge ("Update -> From Local")
# ---------------------------------------------------------------------------


def read_table(path):
    """Read a local table file into a DataFrame by suffix.

    Supports ``.parquet``, ``.feather``, ``.csv`` and ``.tsv``; anything else
    is attempted as CSV. Raises on unreadable files (callers wrap for the GUI).
    """
    path = Path(path)
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    if path.suffix == ".feather":
        return pd.read_feather(path)
    if path.suffix == ".tsv":
        return pd.read_csv(path, sep="\t")
    return pd.read_csv(path)


def _norm_key_col(series):
    """Normalize one join-key column to comparable strings.

    Integral floats (a CSV artifact when an id column contains gaps) are cast
    through ``Int64`` first so ``123.0`` matches an integer ``123``.
    """
    if pd.api.types.is_float_dtype(series):
        non_null = series.dropna()
        if (
            len(non_null)
            and np.isfinite(non_null).all()
            and (non_null == np.round(non_null)).all()
        ):
            try:
                series = series.astype("Int64")
            except (TypeError, ValueError, OverflowError):
                pass
    return series.astype(str).str.strip()


def _join_key_series(df, join_columns):
    """Build a join-key Series aligned to ``df.index``.

    Single-column keys are normalized strings; composite keys become tuples of
    them (tuples cannot collide the way separator-joined strings can).
    """
    normed = [_norm_key_col(df[col]) for col in join_columns]
    if len(normed) == 1:
        return normed[0]
    return pd.Series(list(zip(*normed)), index=df.index)


def _format_key(key):
    """Render a (possibly composite) join key for display."""
    if isinstance(key, tuple):
        return "/".join(str(part) for part in key)
    return str(key)


@dataclass
class LocalMergeReport:
    """Outcome of a :func:`merge_local_meta` call.

    Duck-compatible with :class:`UpdateReport` where the GUI relies on it:
    ``summary()`` and an iterable ``datasets`` (always empty here — a local
    merge is not scoped per dataset).
    """

    join_columns: list = field(default_factory=list)
    rows_meta: int = 0
    rows_source: int = 0
    rows_matched: int = 0
    rows_missing: int = 0
    rows_new: int = 0
    missing_keys_sample: list = field(default_factory=list)
    new_keys_sample: list = field(default_factory=list)
    duplicate_source_keys: int = 0
    duplicate_meta_keys: int = 0
    columns_updated: list = field(default_factory=list)
    columns_added: list = field(default_factory=list)
    columns_skipped: list = field(default_factory=list)
    columns_failed: list = field(default_factory=list)
    cells_changed: int = 0
    cells_skipped_nan: int = 0
    datasets: list = field(default_factory=list)

    def summary(self):
        """Human-readable one-paragraph summary."""
        cols = sorted(set(self.columns_updated) | set(self.columns_added))
        head = (
            f"Updated {self.cells_changed:,} cells in {self.rows_matched:,} "
            f"matched rows"
        )
        parts = [head + (f" ({', '.join(cols)})." if cols else ".")]
        if self.rows_missing:
            parts.append(
                f"{self.rows_missing:,} meta "
                f"row{'s' if self.rows_missing != 1 else ''} had no match "
                "in the file."
            )
        if self.rows_new:
            parts.append(
                f"{self.rows_new:,} file "
                f"row{'s' if self.rows_new != 1 else ''} matched no meta row "
                "(ignored)."
            )
        if self.columns_failed:
            parts.append(f"Failed columns: {', '.join(self.columns_failed)}.")
        return " ".join(parts)


def _clear_cells(out, target, clear_index):
    """Set cells to NA, upcasting to ``object`` if the dtype can't hold NA."""
    try:
        out.loc[clear_index, target] = pd.NA
    except (ValueError, TypeError):
        out[target] = out[target].astype(object)
        out.loc[clear_index, target] = pd.NA


def _merge_one_local_column(
    out,
    report,
    meta_key,
    matched_mask,
    src_key,
    src_vals,
    target,
    *,
    mode,
    apply_source_nan,
):
    """Merge one source column into ``out[target]``."""
    mapped = meta_key.map(dict(zip(src_key, src_vals)))
    found = mapped.notna()
    # Matched rows whose source cell is null (map() can't tell the two apart,
    # but the match mask can).
    matched_null = matched_mask & ~found

    is_new_col = target not in out.columns
    if is_new_col:
        out[target] = pd.NA

    write_mask = found
    clear_mask = matched_null if apply_source_nan else None
    if mode == "fill_missing" and not is_new_col:
        old_obj = out[target].astype(object)
        empty = out[target].isna() | old_obj.map(
            lambda v: isinstance(v, str) and not v.strip()
        )
        write_mask = write_mask & empty
        if clear_mask is not None:
            clear_mask = clear_mask & empty

    target_index = out.index[write_mask]
    new_vals = mapped[write_mask]
    clear_index = out.index[clear_mask] if clear_mask is not None else out.index[:0]
    if clear_mask is None:
        report.cells_skipped_nan += int(matched_null.sum())

    if is_new_col:
        report.columns_added.append(target)
        report.cells_changed += int(len(new_vals))
    else:
        report.columns_updated.append(target)
        old_vals = out.loc[target_index, target]
        report.cells_changed += _count_changed(old_vals, new_vals)
        if len(clear_index):
            report.cells_changed += int(out.loc[clear_index, target].notna().sum())

    if len(new_vals):
        _assign_column(out, target, target_index, new_vals)
    if len(clear_index):
        _clear_cells(out, target, clear_index)


def merge_local_meta(
    meta_df,
    source_df,
    *,
    join_columns,
    columns,
    mode="overwrite",
    apply_source_nan=False,
):
    """Merge selected columns of a local table into a copy of ``meta_df``.

    Rows are matched on ``join_columns`` (which must exist in both frames).
    The returned DataFrame keeps ``meta_df``'s length, row order and index --
    source rows that match no meta row are reported, never appended.

    Parameters
    ----------
    meta_df, source_df : pandas.DataFrame
    join_columns : list[str]
        Column(s) to match rows on, e.g. ``["id"]`` or ``["id", "dataset"]``.
        Duplicate keys in the source keep the first occurrence.
    columns : dict
        ``{target_meta_col: source_col}`` -- which source columns to import
        and under which meta column name. Reserved targets (``id``,
        ``dataset``, ``_*``) and join columns are skipped.
    mode : "overwrite" | "fill_missing"
        With ``"fill_missing"`` only cells currently null/empty-string in the
        meta are written.
    apply_source_nan : bool
        When True, null source cells clear the meta value in matched rows;
        by default they are skipped (and counted in ``cells_skipped_nan``).

    Returns
    -------
    (pandas.DataFrame, LocalMergeReport)
    """
    if mode not in ("overwrite", "fill_missing"):
        raise ValueError(f"Unknown mode: {mode!r}")

    join_columns = [str(c) for c in (join_columns or [])]
    if not join_columns:
        raise ValueError("At least one join column is required.")
    for col in join_columns:
        if col not in meta_df.columns:
            raise ValueError(f"Join column '{col}' not found in the meta data.")
        if col not in source_df.columns:
            raise ValueError(f"Join column '{col}' not found in the file.")

    report = LocalMergeReport(join_columns=list(join_columns))
    report.rows_meta = int(len(meta_df))
    report.rows_source = int(len(source_df))

    usable = {}
    for target, source_col in (columns or {}).items():
        target = str(target)
        if (
            _is_reserved(target)
            or target in join_columns
            or str(source_col) not in source_df.columns
        ):
            report.columns_skipped.append(target)
            continue
        usable[target] = str(source_col)
    if not usable:
        raise ValueError("No importable columns selected.")

    meta_key = _join_key_series(meta_df, join_columns)
    src_key = _join_key_series(source_df, join_columns)

    keep = ~src_key.duplicated(keep="first")
    report.duplicate_source_keys = int((~keep).sum())
    src_key = src_key[keep]
    src_kept = source_df[keep]

    # Meta rows sharing a join key all receive the same source value; the
    # report surfaces this so the UI can hint at adding e.g. `dataset`.
    report.duplicate_meta_keys = int(meta_key.duplicated(keep=False).sum())

    matched_mask = meta_key.isin(set(src_key))
    report.rows_matched = int(matched_mask.sum())
    report.rows_missing = int((~matched_mask).sum())
    new_mask = ~src_key.isin(set(meta_key))
    report.rows_new = int(new_mask.sum())
    report.missing_keys_sample = [
        _format_key(k) for k in meta_key[~matched_mask].unique()[:5]
    ]
    report.new_keys_sample = [
        _format_key(k) for k in src_key[new_mask].unique()[:5]
    ]

    out = meta_df.copy()
    for target, source_col in usable.items():
        # A single bad column must not abort the rest of the merge.
        try:
            _merge_one_local_column(
                out,
                report,
                meta_key,
                matched_mask,
                src_key,
                src_kept[source_col],
                target,
                mode=mode,
                apply_source_nan=apply_source_nan,
            )
        except Exception as exc:  # noqa: BLE001 - surfaced in the report
            detail = str(exc).strip() or exc.__class__.__name__
            report.columns_failed.append(target)
            logger.warning(
                "Local meta merge: column '%s' failed: %s",
                target,
                detail,
                exc_info=True,
            )

    return out, report
