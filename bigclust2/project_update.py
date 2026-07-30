"""Refresh an existing project's meta data on disk.

The GUI pulls fresh annotations into the *loaded* project (View -> Meta Data
Explorer -> Update), where they live only for the session. This module does the
same pull for a project *directory* and saves the result. See
:func:`update_project_meta`; the merge itself lives in
:mod:`bigclust2.meta_sources`, so what lands on disk is what the GUI would show.
"""

from __future__ import annotations

import json
import logging
import os

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from .data import parse_directory, SingleProjectLoader
from .meta_sources import (
    UpdateReport,
    parse_meta_sources,
    stamp_meta_updated,
    update_meta,
    update_source_columns,
    write_table,
)
from .utils import is_url


logger = logging.getLogger(__name__)

__all__ = ["ProjectMetaUpdate", "update_project_meta"]


@dataclass
class ProjectMetaUpdate:
    """Outcome of an :func:`update_project_meta` call."""

    #: The project directory that was updated.
    path: Path
    #: The refreshed meta table (same rows and order as the one on disk).
    meta: pd.DataFrame
    #: Per-dataset merge details.
    report: UpdateReport
    #: The meta table's file.
    meta_file: Path
    #: Whether the project files were written (False for a dry run or a total failure).
    written: bool = False
    #: Whether this was a dry run (nothing was ever going to be written).
    dry_run: bool = False
    #: Datasets whose source mapped no columns, so nothing could be pulled for them.
    unmapped_datasets: list = field(default_factory=list)
    #: Whether newly auto-mapped columns were stored back into ``info``.
    sources_saved: bool = False

    def summary(self):
        """Human-readable one-paragraph summary."""
        parts = [self.report.summary()]

        if self.unmapped_datasets:
            parts.append(
                f"No columns mapped for: {', '.join(self.unmapped_datasets)} — "
                "nothing to pull for them. Match them by name with "
                "auto_map_columns=True (`--auto-map`)."
            )

        if not self.written:
            parts.append(
                "Dry run: nothing written."
                if self.dry_run
                else "Nothing written (no source updated successfully)."
            )
        elif self.report.cells_changed:
            parts.append(f"Wrote {self.meta_file.name} and info.")
        else:
            parts.append("Nothing changed; refreshed the timestamps in info.")

        if self.sources_saved:
            parts.append("Saved the column mapping to the project.")

        return " ".join(parts)


def _write_json(data, path):
    """Write `data` as indented JSON to `path`, atomically (see `write_table`)."""
    path = Path(path)
    tmp = path.with_name(f".{path.name}.bigclust2-tmp")
    try:
        with open(tmp, "w") as handle:
            json.dump(data, handle, indent=4)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)
    return path


def _open_project(path):
    """Load a project that can be written back to, plus its meta table.

    Returns ``(loader, meta, meta_path)``, raising ValueError with an actionable
    message for anything this command cannot refresh in place.
    """
    # Checked before parsing so a remote path fails instantly rather than after
    # fetching an `info` file we could never write back to.
    if is_url(str(path)):
        raise ValueError(
            f"Cannot update a remote project in place: {path}. Save a local copy "
            "first (Export -> Project in the app, or "
            "`SingleProjectLoader.save_snapshot`) and update that."
        )

    loader = parse_directory(path)
    if not isinstance(loader, SingleProjectLoader):
        projects = [str(p) for p in getattr(loader, "info", [])]
        listed = ", ".join(projects[:5]) + (", …" if len(projects) > 5 else "")
        raise ValueError(
            f"{path} holds {len(projects)} projects, not one. Point at a single "
            f"project directory instead: {listed}"
        )

    meta_file = loader.meta_file
    if not meta_file:
        raise ValueError(f"Project '{loader.name}' declares no meta file to update.")
    if is_url(str(meta_file)):
        raise ValueError(
            f"Project '{loader.name}' keeps its meta table at a URL "
            f"({meta_file}); there is nothing local to write back to."
        )

    meta = loader.meta
    missing = {"id", "dataset"} - set(meta.columns)
    if missing:
        raise ValueError(
            f"Meta table of '{loader.name}' has no {sorted(missing)} column(s); "
            "rows cannot be matched to a source without them."
        )

    return loader, meta, Path(loader.resolve_path(meta_file))


def _select_specs(loader, datasets):
    """The source specs to refresh, narrowed to `datasets` when given."""
    specs = parse_meta_sources(loader.info)
    if not specs:
        raise ValueError(
            f"Project '{loader.name}' declares no meta sources. Add a 'sources' "
            "block to its info file, or configure them in the app (View -> Meta "
            "Data Explorer -> Update -> From Remote…)."
        )

    if datasets is None:
        return specs

    wanted = [datasets] if isinstance(datasets, str) else [str(d) for d in datasets]
    selected = [s for s in specs if s.dataset in wanted]
    if not selected:
        declared = ", ".join(sorted({s.dataset for s in specs}))
        raise ValueError(
            f"No meta source for dataset(s) {wanted} in '{loader.name}'. "
            f"Declared: {declared}."
        )
    return selected


def update_project_meta(
    path,
    *,
    auto_map_columns=False,
    datasets=None,
    dry_run=False,
    save_sources=True,
    backends=None,
    progress_callback=None,
):
    """Refresh a project's meta data from its declared sources and save it.

    Reads the project at `path`, pulls current annotations for every dataset
    that declares a `meta source
    <https://flyconnectome.github.io/bigclust2/reference/data-format/#meta-sources>`_,
    merges them into the meta table and writes the table and the ``info`` file
    back. The table keeps its length, row order and index -- only cell values
    change (plus mapped-but-new columns) -- so it stays aligned with the
    project's embeddings, distances and features.

    The backends are only ever read from; the files written are the project's
    own. Credentials come from the environment (see
    :mod:`bigclust2.credentials`).

    Parameters
    ----------
    path :          str | pathlib.Path
                    A local project directory (one holding an ``info`` file).
    auto_map_columns : bool
                    Also refresh project columns the sources do not map, where a
                    source has a column of a matching name. Explicit mappings in
                    ``info`` always win.
    datasets :      str | list of str, optional
                    Only refresh these datasets. Sources for other datasets are
                    left untouched -- both on disk and in ``info``.
    dry_run :       bool
                    Pull and merge, but write nothing. The refreshed table is
                    still returned, so you can inspect it before committing.
    save_sources :  bool
                    Store column mappings discovered by `auto_map_columns` back
                    into ``info``, so the next refresh does not have to match
                    them again. Existing mappings are never overwritten.
    backends :      dict, optional
                    Pre-built ``{dataset: AnnotationBackend}`` instances. Mostly
                    useful for testing; missing datasets are built from ``info``.
    progress_callback : callable, optional
                    Called as ``progress_callback(done, total, dataset)`` before
                    each source is read, and once at the end with
                    ``done == total`` and ``dataset=""``.

    Returns
    -------
    ProjectMetaUpdate
        Carries the refreshed table, the per-dataset report and what was written.

    Raises
    ------
    ValueError
        If `path` is not a single, local project, if its meta table lacks the
        ``id``/``dataset`` columns needed to match rows, or if it declares no
        meta sources to refresh from.

    Examples
    --------
    >>> import bigclust2
    >>> result = bigclust2.update_project_meta(          # doctest: +SKIP
    ...     "my_project/", auto_map_columns=True
    ... )
    >>> print(result.summary())                          # doctest: +SKIP
    """
    loader, meta, meta_path = _open_project(path)
    specs = _select_specs(loader, datasets)

    updated, report = update_meta(
        meta,
        specs,
        auto_map=auto_map_columns,
        backends=backends,
        progress_callback=progress_callback,
    )

    # `update_meta` guarantees this, but the project's other files are matched to
    # the meta table by row position - so check before overwriting it.
    if len(updated) != len(meta) or not updated.index.equals(meta.index):
        raise RuntimeError(
            "Meta update changed the table's row count or order; refusing to "
            "write. This is a bug - please report it."
        )

    result = ProjectMetaUpdate(
        path=Path(loader.path),
        meta=updated,
        report=report,
        dry_run=dry_run,
        meta_file=meta_path,
        unmapped_datasets=[
            d.dataset for d in report.datasets if d.rows and not (d.error or d.columns_resolved)
        ],
    )

    if dry_run:
        return result

    # Every source failed: the "refreshed" table is just the old one, and
    # stamping `last_updated` would hide that.
    if report.datasets and len(report.errors) == len(report.datasets):
        logger.warning("All meta sources failed; leaving the project untouched.")
        return result

    if save_sources:
        mapped = {
            d.dataset: d.columns_resolved for d in report.datasets if d.columns_resolved
        }
        result.sources_saved = bool(update_source_columns(loader.info, mapped))

    stamp_meta_updated(loader.info, report)

    # Rewriting an unchanged table would burn a full serialization on every
    # scheduled run; the stamps in `info` still record that we checked. A column
    # is only ever added when at least one value lands in it, so a zero cell
    # count means the table on disk is already what we would write.
    if report.cells_changed:
        write_table(updated, meta_path)
    _write_json(loader.info, loader.path / "info")
    result.written = True

    return result
