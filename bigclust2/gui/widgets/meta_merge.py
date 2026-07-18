"""UI for merging a local table file into the project's meta data.

``MetaMergeDialog`` lets the user pick a local CSV/TSV/parquet/feather file,
choose which column(s) to join on (e.g. ``id`` or ``id`` + ``dataset``) and
which columns to import, sanity-check the merge (matched/missing/new rows)
and apply it in memory. The meta table's rows never change -- file rows that
match nothing are reported and ignored.

The heavy lifting lives in the Qt-free ``bigclust2.meta_sources`` module
(:func:`bigclust2.meta_sources.merge_local_meta`); this file is only the Qt
layer.
"""

from __future__ import annotations

import json
import logging

from pathlib import Path

import pandas as pd

from PySide6 import QtCore, QtWidgets

from ...meta_sources import merge_local_meta, read_table


logger = logging.getLogger(__name__)


FILE_FILTER = "Data files (*.csv *.tsv *.parquet *.feather);;All files (*)"


def _is_reserved(column):
    """Meta columns that are never valid merge targets."""
    text = str(column)
    return text in ("id", "dataset") or text.startswith("_")


class MetaMergeDialog(QtWidgets.QDialog):
    """Merge a local table into the project's meta data (in memory)."""

    # (updated_meta DataFrame, LocalMergeReport) -- same shape as
    # MetaSourcesDialog.metaUpdated, so core wires both to _apply_meta_update.
    metaUpdated = QtCore.Signal(object, object)

    SETTINGS_KEY_LAST_CONFIG = "metaMerge/last_config"

    # Status colors: neutral grey, success green, warning amber, error red.
    STATUS_NEUTRAL = "#ededed"
    STATUS_SUCCESS = "#05fc6c"
    STATUS_WARNING = "#b26a00"
    STATUS_ERROR = "#c62828"

    STATUS_JOIN = "join column"
    STATUS_RESERVED = "reserved — blocked"
    STATUS_EXISTING = "updates existing"
    STATUS_NEW = "new column"

    def __init__(self, meta, parent=None, settings=None):
        super().__init__(parent)
        if not isinstance(meta, pd.DataFrame):
            raise TypeError("meta must be a pandas DataFrame")

        self._meta = meta
        self._source = None
        self._pending = None  # (merged DataFrame, LocalMergeReport) from Check
        self._loading = False  # suppress change-handlers while populating
        self._settings = (
            settings
            if settings is not None
            else QtCore.QSettings("BigClust", "BigClustGUI")
        )

        self.setWindowTitle("Update Meta From Local File")
        self.resize(760, 640)
        self.setModal(False)

        self._build_ui()
        self._restore_last_config()

    # ------------------------------------------------------------------ #
    # UI construction
    # ------------------------------------------------------------------ #

    def _build_ui(self):
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(10)

        help_label = QtWidgets.QLabel(
            "Load a local table and merge selected columns into the project's "
            "meta data. Rows are matched on the join column(s); the meta "
            "table's rows never change — file rows that match nothing are "
            "reported and ignored. Nothing is written to disk."
        )
        help_label.setWordWrap(True)
        layout.addWidget(help_label)

        file_row = QtWidgets.QHBoxLayout()
        file_row.addWidget(QtWidgets.QLabel("File:"))
        self.path_edit = QtWidgets.QLineEdit()
        self.path_edit.setPlaceholderText("Path to a .csv/.tsv/.parquet/.feather file")
        self.path_edit.returnPressed.connect(self._load_file)
        file_row.addWidget(self.path_edit, 1)
        self.browse_btn = QtWidgets.QPushButton("Browse…")
        self.browse_btn.clicked.connect(self._on_browse)
        file_row.addWidget(self.browse_btn)
        self.load_btn = QtWidgets.QPushButton("Load")
        self.load_btn.setToolTip("Read the file and list its columns below.")
        self.load_btn.clicked.connect(self._load_file)
        file_row.addWidget(self.load_btn)
        layout.addLayout(file_row)

        split_row = QtWidgets.QHBoxLayout()

        join_group = QtWidgets.QGroupBox("Join on")
        join_layout = QtWidgets.QVBoxLayout(join_group)
        self.join_list = QtWidgets.QListWidget()
        self.join_list.setToolTip(
            "Columns used to match file rows to meta rows. Must be present "
            "in both tables."
        )
        self.join_list.itemChanged.connect(self._on_join_changed)
        join_layout.addWidget(self.join_list)
        self.join_hint_label = QtWidgets.QLabel("")
        self.join_hint_label.setWordWrap(True)
        self.join_hint_label.setStyleSheet("color: #7a7a7a; font-size: 11px;")
        join_layout.addWidget(self.join_hint_label)
        split_row.addWidget(join_group, 1)

        import_group = QtWidgets.QGroupBox("Columns to import")
        import_layout = QtWidgets.QVBoxLayout(import_group)
        self.import_table = QtWidgets.QTableWidget(0, 4)
        self.import_table.setHorizontalHeaderLabels(
            ["Import", "File column", "Import as", "Status"]
        )
        self.import_table.verticalHeader().setVisible(False)
        self.import_table.setSelectionMode(QtWidgets.QAbstractItemView.NoSelection)
        header = self.import_table.horizontalHeader()
        header.setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeToContents)
        header.setSectionResizeMode(1, QtWidgets.QHeaderView.Stretch)
        header.setSectionResizeMode(2, QtWidgets.QHeaderView.Stretch)
        header.setSectionResizeMode(3, QtWidgets.QHeaderView.ResizeToContents)
        self.import_table.itemChanged.connect(self._on_import_item_changed)
        import_layout.addWidget(self.import_table)
        split_row.addWidget(import_group, 2)

        layout.addLayout(split_row, 2)

        options_row = QtWidgets.QHBoxLayout()
        self.fill_only_check = QtWidgets.QCheckBox(
            "Only fill empty cells (never overwrite existing values)"
        )
        self.fill_only_check.toggled.connect(self._on_fill_only_toggled)
        options_row.addWidget(self.fill_only_check)
        self.apply_nan_check = QtWidgets.QCheckBox(
            "Empty file cells clear existing values"
        )
        self.apply_nan_check.toggled.connect(self._invalidate_check)
        options_row.addWidget(self.apply_nan_check)
        options_row.addStretch(1)
        layout.addLayout(options_row)

        check_group = QtWidgets.QGroupBox("Sanity check")
        check_layout = QtWidgets.QVBoxLayout(check_group)
        self.check_panel = QtWidgets.QPlainTextEdit()
        self.check_panel.setReadOnly(True)
        self.check_panel.setPlaceholderText(
            "Load a file, pick join and import columns, then press "
            "'Check merge' to preview what would change."
        )
        check_layout.addWidget(self.check_panel)
        layout.addWidget(check_group, 1)

        self.status_label = QtWidgets.QLabel("")
        self.status_label.setWordWrap(True)
        self.status_label.setStyleSheet("color: #7a7a7a;")

        buttons = QtWidgets.QHBoxLayout()
        buttons.addWidget(self.status_label, 1)
        self.check_btn = QtWidgets.QPushButton("Check merge")
        self.check_btn.setToolTip("Preview the merge without changing anything.")
        self.check_btn.clicked.connect(self._on_check)
        buttons.addWidget(self.check_btn)
        self.apply_btn = QtWidgets.QPushButton("Apply")
        self.apply_btn.setToolTip("Apply the checked merge to the meta data.")
        self.apply_btn.setEnabled(False)
        self.apply_btn.clicked.connect(self._on_apply)
        buttons.addWidget(self.apply_btn)
        self.close_btn = QtWidgets.QPushButton("Close")
        self.close_btn.clicked.connect(self.close)
        buttons.addWidget(self.close_btn)
        layout.addLayout(buttons)

    # ------------------------------------------------------------------ #
    # File loading & population
    # ------------------------------------------------------------------ #

    def _on_browse(self):
        start_dir = ""
        path = self.path_edit.text().strip()
        if path:
            start_dir = str(Path(path).parent)
        selected, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Select meta data table", start_dir, FILE_FILTER
        )
        if not selected:
            return
        self.path_edit.setText(selected)
        self._load_file()

    def _load_file(self):
        path = self.path_edit.text().strip()
        if not path:
            return
        try:
            source = read_table(path)
        except Exception as exc:  # noqa: BLE001 - surfaced in UI
            QtWidgets.QMessageBox.warning(
                self,
                "Read error",
                f"Could not read file '{Path(path).name}': {exc}",
            )
            return

        self._source = source
        self._populate_from_source()
        self._invalidate_check()
        n_rows, n_cols = source.shape
        color = None
        text = f"Loaded {n_rows:,} rows / {n_cols} columns from {Path(path).name}."
        if source.empty:
            color = self.STATUS_WARNING
            text += " The file contains no rows."
        self._set_status(text, color=color)

    def _populate_from_source(self):
        """(Re)build the join list and import table for the loaded file."""
        source = self._source
        config = self._last_config()

        self._loading = True
        try:
            # Join candidates: non-internal columns present in both tables.
            meta_cols = [str(c) for c in self._meta.columns if not str(c).startswith("_")]
            common = [c for c in meta_cols if c in source.columns]

            remembered_join = [
                c for c in config.get("join_columns", []) if c in common
            ]
            defaults = remembered_join
            if not defaults:
                defaults = [c for c in ("id", "dataset") if c in common]

            self.join_list.clear()
            for col in common:
                item = QtWidgets.QListWidgetItem(col)
                item.setFlags(item.flags() | QtCore.Qt.ItemIsUserCheckable)
                item.setCheckState(
                    QtCore.Qt.Checked if col in defaults else QtCore.Qt.Unchecked
                )
                self.join_list.addItem(item)

            # Remembered import selection is only trusted when at least one of
            # its source columns exists in this file.
            remembered_imports = {
                str(target): str(src)
                for target, src in config.get("import_columns", {}).items()
                if str(src) in map(str, source.columns)
            }
            by_source = {src: target for target, src in remembered_imports.items()}
            use_remembered = bool(remembered_imports)

            self.import_table.setRowCount(0)
            self.import_table.setRowCount(len(source.columns))
            for row, col in enumerate(map(str, source.columns)):
                check_item = QtWidgets.QTableWidgetItem()
                check_item.setFlags(
                    QtCore.Qt.ItemIsUserCheckable | QtCore.Qt.ItemIsEnabled
                )
                if use_remembered:
                    checked = col in by_source
                else:
                    checked = not _is_reserved(col)
                check_item.setCheckState(
                    QtCore.Qt.Checked if checked else QtCore.Qt.Unchecked
                )
                self.import_table.setItem(row, 0, check_item)

                source_item = QtWidgets.QTableWidgetItem(col)
                source_item.setFlags(
                    QtCore.Qt.ItemIsEnabled | QtCore.Qt.ItemIsSelectable
                )
                self.import_table.setItem(row, 1, source_item)

                target_item = QtWidgets.QTableWidgetItem(by_source.get(col, col))
                self.import_table.setItem(row, 2, target_item)

                status_item = QtWidgets.QTableWidgetItem("")
                status_item.setFlags(QtCore.Qt.ItemIsEnabled)
                self.import_table.setItem(row, 3, status_item)
        finally:
            self._loading = False

        self._refresh_import_states()
        self._refresh_join_hint()

    # ------------------------------------------------------------------ #
    # Selection state
    # ------------------------------------------------------------------ #

    def join_columns(self):
        """Checked join columns, in list order."""
        cols = []
        for i in range(self.join_list.count()):
            item = self.join_list.item(i)
            if item.checkState() == QtCore.Qt.Checked:
                cols.append(item.text())
        return cols

    def import_columns(self):
        """Checked import columns as ``{target_meta_col: file_col}``."""
        mapping = {}
        for row in range(self.import_table.rowCount()):
            check_item = self.import_table.item(row, 0)
            if check_item is None or check_item.checkState() != QtCore.Qt.Checked:
                continue
            if not check_item.flags() & QtCore.Qt.ItemIsEnabled:
                continue
            src = self.import_table.item(row, 1).text()
            target = self.import_table.item(row, 2).text().strip() or src
            mapping[target] = src
        return mapping

    def _row_status(self, src, target, join_cols):
        if src in join_cols:
            return self.STATUS_JOIN
        if _is_reserved(target):
            return self.STATUS_RESERVED
        if target in map(str, self._meta.columns):
            return self.STATUS_EXISTING
        return self.STATUS_NEW

    def _refresh_import_states(self):
        """Recompute each import row's status and enabled state."""
        was_loading = self._loading
        self._loading = True
        try:
            join_cols = set(self.join_columns())
            for row in range(self.import_table.rowCount()):
                check_item = self.import_table.item(row, 0)
                src_item = self.import_table.item(row, 1)
                target_item = self.import_table.item(row, 2)
                status_item = self.import_table.item(row, 3)
                if None in (check_item, src_item, target_item, status_item):
                    continue
                src = src_item.text()
                target = target_item.text().strip() or src
                status = self._row_status(src, target, join_cols)
                status_item.setText(status)

                blocked = status in (self.STATUS_JOIN, self.STATUS_RESERVED)
                if blocked:
                    check_item.setCheckState(QtCore.Qt.Unchecked)
                    check_item.setFlags(QtCore.Qt.ItemIsUserCheckable)
                    # Renaming can unblock a reserved target but not a join column.
                    target_flags = (
                        QtCore.Qt.ItemIsEnabled | QtCore.Qt.ItemIsEditable
                        if status == self.STATUS_RESERVED
                        else QtCore.Qt.ItemIsEnabled
                    )
                    target_item.setFlags(target_flags)
                else:
                    check_item.setFlags(
                        QtCore.Qt.ItemIsUserCheckable | QtCore.Qt.ItemIsEnabled
                    )
                    target_item.setFlags(
                        QtCore.Qt.ItemIsEnabled | QtCore.Qt.ItemIsEditable
                    )
        finally:
            self._loading = was_loading

    def _refresh_join_hint(self):
        join_cols = self.join_columns()
        if not join_cols or self._meta.empty:
            self.join_hint_label.setStyleSheet("color: #7a7a7a; font-size: 11px;")
            self.join_hint_label.setText("")
            return
        key = self._meta[join_cols].astype(str).agg("/".join, axis=1)
        n_dup = int(key.duplicated(keep=False).sum())
        if n_dup:
            self.join_hint_label.setStyleSheet(
                f"color: {self.STATUS_WARNING}; font-size: 11px;"
            )
            hint = f"{n_dup:,} meta rows share a join key."
            if "dataset" not in join_cols and "dataset" in self._meta.columns:
                hint += " Consider adding 'dataset'."
            self.join_hint_label.setText(hint)
        else:
            self.join_hint_label.setStyleSheet("color: #7a7a7a; font-size: 11px;")
            self.join_hint_label.setText("Join key is unique in the meta data.")

    # ------------------------------------------------------------------ #
    # Change handlers
    # ------------------------------------------------------------------ #

    def _on_join_changed(self, _item=None):
        if self._loading:
            return
        self._refresh_import_states()
        self._refresh_join_hint()
        self._invalidate_check()

    def _on_import_item_changed(self, item):
        if self._loading:
            return
        if item.column() == 2:
            self._refresh_import_states()
        self._invalidate_check()

    def _on_fill_only_toggled(self, checked):
        self.apply_nan_check.setEnabled(not checked)
        self._invalidate_check()

    def _invalidate_check(self, *_args):
        had_pending = self._pending is not None
        self._pending = None
        self.apply_btn.setEnabled(False)
        if had_pending and not self._loading:
            self._set_status("Settings changed — run 'Check merge' again.")

    # ------------------------------------------------------------------ #
    # Check & apply
    # ------------------------------------------------------------------ #

    def _merge_kwargs(self):
        return dict(
            join_columns=self.join_columns(),
            columns=self.import_columns(),
            mode="fill_missing" if self.fill_only_check.isChecked() else "overwrite",
            apply_source_nan=(
                self.apply_nan_check.isChecked()
                and self.apply_nan_check.isEnabled()
            ),
        )

    def _on_check(self):
        if self._source is None:
            self._set_status("Load a file first.", color=self.STATUS_ERROR)
            return
        kwargs = self._merge_kwargs()
        if not kwargs["join_columns"]:
            self._set_status(
                "Check at least one join column.", color=self.STATUS_ERROR
            )
            return
        if not kwargs["columns"]:
            self._set_status(
                "Check at least one column to import.", color=self.STATUS_ERROR
            )
            return

        try:
            out, report = merge_local_meta(self._meta, self._source, **kwargs)
        except ValueError as exc:
            self._set_status(str(exc), color=self.STATUS_ERROR)
            return
        except Exception as exc:  # noqa: BLE001 - surfaced in UI
            logger.exception("Local meta merge check failed")
            detail = str(exc).strip() or exc.__class__.__name__
            self._set_status(f"Check failed: {detail}", color=self.STATUS_ERROR)
            return

        self._pending = (out, report)
        self.check_panel.setPlainText(self._describe_report(report))

        if report.rows_matched == 0:
            self.apply_btn.setEnabled(False)
            self._set_status(
                "No file row matches any meta row — nothing to apply.",
                color=self.STATUS_ERROR,
            )
            return

        self.apply_btn.setEnabled(True)
        partial = bool(
            report.rows_missing
            or report.rows_new
            or report.columns_failed
            or report.columns_skipped
        )
        self._set_status(
            "Check passed — review the sanity check, then press Apply.",
            color=self.STATUS_WARNING if partial else self.STATUS_SUCCESS,
        )

    def _describe_report(self, report):
        lines = [
            f"Join on: {', '.join(report.join_columns)}",
            f"Meta rows: {report.rows_meta:,} | file rows: {report.rows_source:,}",
            "",
            f"Matched meta rows: {report.rows_matched:,}",
        ]
        if report.rows_missing:
            sample = ", ".join(report.missing_keys_sample)
            lines.append(
                f"Meta rows without a match (kept as-is): "
                f"{report.rows_missing:,} (e.g. {sample})"
            )
        if report.rows_new:
            sample = ", ".join(report.new_keys_sample)
            lines.append(
                f"File rows matching no meta row (ignored — rows cannot be "
                f"added): {report.rows_new:,} (e.g. {sample})"
            )
        if report.duplicate_source_keys:
            lines.append(
                f"Duplicate join keys in the file (first occurrence used): "
                f"{report.duplicate_source_keys:,}"
            )
        if report.duplicate_meta_keys:
            lines.append(
                f"Meta rows sharing a join key (all receive the same value): "
                f"{report.duplicate_meta_keys:,}"
            )
        lines.append("")
        if report.columns_updated:
            lines.append(f"Columns updated: {', '.join(report.columns_updated)}")
        if report.columns_added:
            lines.append(f"Columns added: {', '.join(report.columns_added)}")
        if report.columns_skipped:
            lines.append(f"Columns skipped: {', '.join(report.columns_skipped)}")
        if report.columns_failed:
            lines.append(f"Columns failed: {', '.join(report.columns_failed)}")
        lines.append(f"Cells that would change: {report.cells_changed:,}")
        if report.cells_skipped_nan:
            lines.append(
                f"Empty file cells skipped (existing values kept): "
                f"{report.cells_skipped_nan:,}"
            )
        return "\n".join(lines)

    def _on_apply(self):
        if self._pending is None:
            return
        out, report = self._pending

        # Keep our reference current so consecutive merges stack.
        self._meta = out
        self._save_last_config()
        self.metaUpdated.emit(out, report)

        # The meta changed; a new merge must be re-checked against it.
        self._pending = None
        self.apply_btn.setEnabled(False)
        self._refresh_import_states()
        self._refresh_join_hint()
        self._set_status(report.summary(), color=self.STATUS_SUCCESS)

    # ------------------------------------------------------------------ #
    # Settings persistence
    # ------------------------------------------------------------------ #

    def _last_config(self):
        raw = self._settings.value(self.SETTINGS_KEY_LAST_CONFIG, "")
        if not raw:
            return {}
        try:
            config = json.loads(raw)
        except (TypeError, ValueError):
            return {}
        return config if isinstance(config, dict) else {}

    def _restore_last_config(self):
        config = self._last_config()
        last_path = str(config.get("last_path", "") or "")
        if last_path and Path(last_path).is_file():
            self.path_edit.setText(last_path)
        self.fill_only_check.setChecked(config.get("mode") == "fill_missing")
        self.apply_nan_check.setChecked(bool(config.get("apply_source_nan")))

    def _save_last_config(self):
        config = {
            "last_path": self.path_edit.text().strip(),
            "join_columns": self.join_columns(),
            "import_columns": self.import_columns(),
            "mode": (
                "fill_missing" if self.fill_only_check.isChecked() else "overwrite"
            ),
            "apply_source_nan": (
                self.apply_nan_check.isChecked()
                and self.apply_nan_check.isEnabled()
            ),
        }
        self._settings.setValue(
            self.SETTINGS_KEY_LAST_CONFIG, json.dumps(config)
        )

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    def _set_status(self, text, color=None):
        """Set the footer status text and color (defaults to neutral grey)."""
        self.status_label.setStyleSheet(f"color: {color or self.STATUS_NEUTRAL};")
        self.status_label.setText(text)


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO)

    app = QtWidgets.QApplication(sys.argv)

    if len(sys.argv) > 1:
        meta = read_table(sys.argv[1])
    else:
        meta = pd.DataFrame(
            {
                "id": [1, 2, 3, 1, 2],
                "dataset": ["A", "A", "A", "B", "B"],
                "label": ["a1", "a2", "a3", "b1", "b2"],
                "soma_side": ["l", "r", "l", "r", "l"],
            }
        )

    dlg = MetaMergeDialog(meta)

    def _print_update(updated, report):
        logger.info("metaUpdated: %s", report.summary())

    dlg.metaUpdated.connect(_print_update)
    dlg.show()
    sys.exit(app.exec())
