"""GUI for authoring a BigClust project (**File -> Build Project**).

A single-form dialog that is a thin wrapper over
:class:`bigclust2.project_builder.ProjectBuilder`: the same engine backs both the
Python API and this dialog, so they produce identical projects. Table files
(meta / embeddings / distances / features) are referenced by path and only read
when the user clicks *Build*, so validation errors surface with a clear message
at build time rather than while filling in the form.
"""

import logging

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QMessageBox,
    QProgressDialog,
    QPushButton,
    QStackedWidget,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from ...project_builder import ProjectBuilder, plan_meta_remap, apply_meta_remap
from ...meta_sources import read_table

logger = logging.getLogger(__name__)

__all__ = ["ProjectBuilderDialog"]

# File dialog filter shared by every table picker.
_TABLE_FILTER = "Tables (*.parquet *.feather *.csv *.tsv);;All files (*)"


def _browse_open(parent, caption, name_filter):
    """Return a chosen existing file path (or "" if cancelled)."""
    path, _ = QFileDialog.getOpenFileName(parent, caption, "", name_filter)
    return path or ""


def _grow_form(parent=None):
    """A QFormLayout whose fields expand to the available width.

    The default (macOS especially) keeps fields at their size hint, which
    truncates long file paths and labels — grow the field column instead.
    """
    form = QFormLayout(parent) if parent is not None else QFormLayout()
    form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
    form.setRowWrapPolicy(QFormLayout.DontWrapRows)
    form.setLabelAlignment(Qt.AlignLeft)
    return form


def _help_button(parent, title, text):
    """A small "?" button that shows ``text`` on hover and on click."""
    btn = QToolButton(parent)
    btn.setText("?")
    btn.setAutoRaise(True)
    btn.setFocusPolicy(Qt.NoFocus)
    btn.setCursor(Qt.WhatsThisCursor)
    btn.setToolTip(text)  # rich text auto-detected by Qt
    # Click gives a persistent, readable copy for discoverability (tooltips are
    # easy to miss).
    btn.clicked.connect(lambda: QMessageBox.information(parent, title, text))
    return btn


class EmbeddingConfigDialog(QDialog):
    """Configure a single embedding: x/y source plus optional distances/features.

    ``meta_columns`` are the columns available in the already-loaded meta table;
    they populate the "from meta columns" choice. Returns a plain config dict via
    :meth:`config` (files are stored as paths and read later by the builder).
    """

    def __init__(self, parent=None, *, meta_columns=None, config=None):
        super().__init__(parent)
        self.setWindowTitle("Add Embedding")
        self.setMinimumWidth(480)
        self._meta_columns = list(meta_columns or [])

        layout = QVBoxLayout(self)

        form = _grow_form()
        self.name_edit = QLineEdit()
        self.name_edit.setPlaceholderText("optional display name")
        form.addRow("Name", self.name_edit)

        # x/y source: either two meta columns or a 2-column embeddings file.
        self.source_combo = QComboBox()
        self.source_combo.addItems(["From meta columns", "From file"])
        form.addRow("Coordinates", self.source_combo)
        layout.addLayout(form)

        # A stacked widget switches between the two coordinate sources so that
        # each option's fields (and their labels) show/hide as a unit.
        self.coord_stack = QStackedWidget()

        columns_page = QWidget()
        columns_form = _grow_form(columns_page)
        self.x_combo = QComboBox()
        self.y_combo = QComboBox()
        self.x_combo.addItems(self._meta_columns)
        self.y_combo.addItems(self._meta_columns)
        # Default to the conventional x/y columns when present.
        if "x" in self._meta_columns:
            self.x_combo.setCurrentText("x")
        if "y" in self._meta_columns:
            self.y_combo.setCurrentText("y")
        columns_form.addRow("X column", self.x_combo)
        columns_form.addRow("Y column", self.y_combo)
        self.coord_stack.addWidget(columns_page)

        file_page = QWidget()
        file_form = _grow_form(file_page)
        self.file_edit, file_row = self._file_row("Pick a 2-column table")
        file_form.addRow("Embeddings file", file_row)
        self.coord_stack.addWidget(file_page)

        self.source_combo.currentIndexChanged.connect(self.coord_stack.setCurrentIndex)
        layout.addWidget(self.coord_stack)

        # Optional distances.
        self.dist_group = QGroupBox("Attach distances (optional)")
        self.dist_group.setCheckable(True)
        self.dist_group.setChecked(False)
        dist_form = _grow_form(self.dist_group)
        self.dist_edit, dist_row = self._file_row("Pick a square distance matrix")
        dist_form.addRow("File", dist_row)
        self.dist_type_edit = QLineEdit("connectivity")
        dist_form.addRow("Type", self.dist_type_edit)
        self.dist_metric_edit = QLineEdit()
        self.dist_metric_edit.setPlaceholderText("e.g. cosine (optional)")
        dist_form.addRow("Metric", self.dist_metric_edit)
        layout.addWidget(self.dist_group)

        # Optional features.
        self.feat_group = QGroupBox("Attach features (optional)")
        self.feat_group.setCheckable(True)
        self.feat_group.setChecked(False)
        feat_form = _grow_form(self.feat_group)
        self.feat_edit, feat_row = self._file_row("Pick a feature matrix")
        feat_form.addRow("File", feat_row)
        self.feat_type_edit = QLineEdit("connectivity")
        feat_form.addRow("Type", self.feat_type_edit)
        layout.addWidget(self.feat_group)

        buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        )
        buttons.accepted.connect(self._on_accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        if config:
            self._apply_config(config)

    def _file_row(self, caption):
        """Build a (line-edit, container-widget) file picker row."""
        container = QWidget()
        row = QHBoxLayout(container)
        row.setContentsMargins(0, 0, 0, 0)
        edit = QLineEdit()
        browse = QPushButton("Browse…")
        browse.clicked.connect(
            lambda: edit.setText(_browse_open(self, caption, _TABLE_FILTER) or edit.text())
        )
        row.addWidget(edit)
        row.addWidget(browse)
        return edit, container

    def _apply_config(self, config):
        self.name_edit.setText(config.get("name") or "")
        if config.get("embeddings_file"):
            self.source_combo.setCurrentText("From file")
            self.coord_stack.setCurrentIndex(1)
            self.file_edit.setText(config["embeddings_file"])
        else:
            cols = config.get("columns") or []
            if len(cols) == 2:
                self.x_combo.setCurrentText(cols[0])
                self.y_combo.setCurrentText(cols[1])
        if config.get("distances_file"):
            self.dist_group.setChecked(True)
            self.dist_edit.setText(config["distances_file"])
            self.dist_type_edit.setText(config.get("distances_type") or "connectivity")
            self.dist_metric_edit.setText(config.get("distances_metric") or "")
        if config.get("features_file"):
            self.feat_group.setChecked(True)
            self.feat_edit.setText(config["features_file"])
            self.feat_type_edit.setText(config.get("features_type") or "connectivity")

    def _on_accept(self):
        """Validate the minimal requirements before closing."""
        if self.source_combo.currentText() == "From meta columns":
            if self.x_combo.currentText() == self.y_combo.currentText():
                QMessageBox.warning(
                    self, "Invalid embedding", "X and Y columns must differ."
                )
                return
        elif not self.file_edit.text().strip():
            QMessageBox.warning(
                self, "Invalid embedding", "Choose an embeddings file or use meta columns."
            )
            return
        if self.dist_group.isChecked() and not self.dist_edit.text().strip():
            QMessageBox.warning(self, "Invalid embedding", "Distances enabled but no file chosen.")
            return
        if self.feat_group.isChecked() and not self.feat_edit.text().strip():
            QMessageBox.warning(self, "Invalid embedding", "Features enabled but no file chosen.")
            return
        self.accept()

    def config(self):
        """Return the embedding config as a plain dict (files as paths)."""
        cfg = {"name": self.name_edit.text().strip() or None}
        if self.source_combo.currentText() == "From meta columns":
            cfg["columns"] = [self.x_combo.currentText(), self.y_combo.currentText()]
        else:
            cfg["embeddings_file"] = self.file_edit.text().strip()
        if self.dist_group.isChecked():
            cfg["distances_file"] = self.dist_edit.text().strip()
            cfg["distances_type"] = self.dist_type_edit.text().strip() or "connectivity"
            cfg["distances_metric"] = self.dist_metric_edit.text().strip() or None
        if self.feat_group.isChecked():
            cfg["features_file"] = self.feat_edit.text().strip()
            cfg["features_type"] = self.feat_type_edit.text().strip() or "connectivity"
        return cfg

    @staticmethod
    def describe(cfg):
        """One-line summary of an embedding config for the list widget."""
        name = cfg.get("name") or "(unnamed)"
        if cfg.get("embeddings_file"):
            src = f"file: {cfg['embeddings_file']}"
        else:
            src = f"columns: {', '.join(cfg.get('columns', []))}"
        extras = []
        if cfg.get("distances_file"):
            extras.append("+distances")
        if cfg.get("features_file"):
            extras.append("+features")
        suffix = f" ({', '.join(extras)})" if extras else ""
        return f"{name} — {src}{suffix}"


# Help text shown by the "?" buttons next to each section.
_HELP_PROJECT = (
    "Basic details recorded in the project's <b>info</b> file.<br><br>"
    "<b>Name</b> defaults to the output folder name. <b>Output folder</b> is the "
    "directory the info file and parquet tables are written into (created if "
    "needed)."
)
_HELP_META = (
    "One row per neuron, read from a <b>.parquet</b>, <b>.feather</b>, <b>.csv</b> "
    "or <b>.tsv</b> table.<br><br>"
    "BigClust needs three fields per row: <b>id</b>, <b>label</b> and "
    "<b>dataset</b>. Pick which of your columns supplies each — they need not be "
    "named that way. If your table already has a column literally named "
    "<code>id</code>, <code>label</code> or <code>dataset</code> that you do "
    "<i>not</i> map, it is kept under a suffixed name so nothing is lost.<br><br>"
    "<b>Color column</b> (optional): per-row colors. <b>Source column</b> "
    "(optional): per-row neuroglancer source URLs for the 3D viewer."
)
_HELP_EMBEDDINGS = (
    "The 2D coordinates drawn in the scatter plot. At least one is required.<br><br>"
    "Each embedding takes its x/y either from two meta columns or from a "
    "2-column table file. You can optionally attach:<br>"
    "• a <b>distances</b> matrix — square, with neuron IDs on both the index and "
    "columns;<br>"
    "• a <b>features</b> table — the high-dimensional vectors the embedding came "
    "from."
)
_HELP_NGL = (
    "Optional. Configures the neuroglancer 3D viewer.<br><br>"
    "<b>Source:</b> a segmentation URL (e.g. <code>precomputed://…</code>) or the "
    "name of a meta column holding per-row URLs. Leave blank to reuse the meta "
    "<i>Source column</i>.<br>"
    "<b>Neuropil mesh:</b> a <code>.ply</code> / <code>.obj</code> / "
    "<code>.stl</code> file or URL shown as anatomical context."
)


class ProjectBuilderDialog(QDialog):
    """Single-form dialog to build a BigClust project from local tables."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Build Project")
        self.setModal(True)
        self.setMinimumWidth(600)
        self.resize(680, 680)

        self._meta = None  # loaded meta DataFrame (for column dropdowns)
        self._embeddings = []  # list of embedding config dicts
        self.built_path = None  # set on successful build
        self._ngl_source_autofilled = False  # 3D-viewer source came from a column pick

        layout = QVBoxLayout(self)

        layout.addWidget(self._build_project_group())
        layout.addWidget(self._build_meta_group())
        layout.addWidget(self._build_embeddings_group())
        layout.addWidget(self._build_neuroglancer_group())

        # "Open after building" preference, then Build / Cancel.
        self.open_after_build = QCheckBox("Open the project after building")
        self.open_after_build.setChecked(True)
        layout.addWidget(self.open_after_build)

        buttons = QDialogButtonBox(QDialogButtonBox.Cancel)
        self.build_btn = QPushButton("Build project")
        self.build_btn.setDefault(True)
        self.build_btn.clicked.connect(self._on_build)
        buttons.addButton(self.build_btn, QDialogButtonBox.AcceptRole)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        # Initial state: no meta yet, so adding embeddings is disabled and the
        # hint spelling out why is shown.
        self._set_embeddings_enabled(False)

    # ---- section scaffolding ---------------------------------------------

    def _section(self, title, help_text, *, checkable=False):
        """A framed section with a bold title, a "?" help button, and a body.

        Returns ``(group, body_layout, toggle)`` where ``toggle`` is the enable
        checkbox for a ``checkable`` section (else ``None``). The title is drawn
        as our own header row (not the QGroupBox title) so the "?" sits right
        next to the label and the body uses the full width.
        """
        group = QGroupBox()
        outer = QVBoxLayout(group)

        header = QHBoxLayout()
        toggle = None
        if checkable:
            toggle = QCheckBox(title)
            font = toggle.font()
            font.setBold(True)
            toggle.setFont(font)
            header.addWidget(toggle)
        else:
            label = QLabel(title)
            font = label.font()
            font.setBold(True)
            label.setFont(font)
            header.addWidget(label)
        header.addWidget(_help_button(self, title, help_text))
        header.addStretch(1)
        outer.addLayout(header)

        body = QWidget()
        body_layout = QVBoxLayout(body)
        body_layout.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(body)

        if toggle is not None:
            toggle.toggled.connect(body.setEnabled)
            body.setEnabled(False)

        return group, body_layout, toggle

    # ---- section builders -------------------------------------------------

    def _build_project_group(self):
        group, body, _ = self._section("Project", _HELP_PROJECT)
        form = _grow_form()
        body.addLayout(form)

        self.name_edit = QLineEdit()
        self.name_edit.setPlaceholderText("shown in the title bar; defaults to the output folder name")
        form.addRow("Name", self.name_edit)
        self.description_edit = QLineEdit()
        self.description_edit.setPlaceholderText("free text, shown in Project Details (optional)")
        form.addRow("Description", self.description_edit)
        self.dataset_edit = QLineEdit()
        self.dataset_edit.setPlaceholderText(
            "just for your reference, not used by BigClust — e.g. hemibrain:v1.2.1"
        )
        form.addRow("Dataset label", self.dataset_edit)

        self.output_edit = QLineEdit()
        self.output_edit.setPlaceholderText("directory to write the project into")
        self.output_edit.textChanged.connect(lambda _t: self._update_build_button())
        form.addRow("Output folder", self._picker_row(self.output_edit, self._browse_output))
        return group

    def _build_meta_group(self):
        group, body, _ = self._section("Meta table (required)", _HELP_META)
        form = _grow_form()
        body.addLayout(form)

        self.meta_edit = QLineEdit()
        self.meta_edit.setPlaceholderText("table with id, label and dataset columns")
        form.addRow("File", self._picker_row(self.meta_edit, self._browse_meta))

        self.meta_status = QLabel("No meta table loaded.")
        self.meta_status.setWordWrap(True)
        self.meta_status.setStyleSheet("color: #666; font-size: 12px;")
        form.addRow("", self.meta_status)

        # Map the user's columns onto the required id / label / dataset fields.
        # Each row carries its own clash line so it is clear which field is
        # affected when a reserved-named column has to be renamed.
        self.id_combo = QComboBox()
        self.label_combo = QComboBox()
        self.dataset_combo = QComboBox()
        self.id_clash_label = self._clash_label()
        self.label_clash_label = self._clash_label()
        self.dataset_clash_label = self._clash_label()
        for combo in (self.id_combo, self.label_combo, self.dataset_combo):
            combo.currentIndexChanged.connect(lambda _i: self._refresh_meta_mapping())
        form.addRow("ID column", self.id_combo)
        form.addRow("", self.id_clash_label)
        form.addRow("Label column", self.label_combo)
        form.addRow("", self.label_clash_label)
        form.addRow("Dataset column", self.dataset_combo)
        form.addRow("", self.dataset_clash_label)

        self.color_combo = QComboBox()
        self.color_combo.addItem("(none)")
        form.addRow("Color column", self.color_combo)

        self.ngl_column_combo = QComboBox()
        self.ngl_column_combo.addItem("(none)")
        self.ngl_column_combo.setToolTip(
            "Meta column holding per-row neuroglancer source URLs. Selecting one "
            "turns on the 3D viewer below and fills in its source."
        )
        self.ngl_column_combo.currentTextChanged.connect(self._on_ngl_column_changed)
        form.addRow("Neuroglancer column", self.ngl_column_combo)
        return group

    def _build_embeddings_group(self):
        group, body, _ = self._section(
            "Embeddings (at least one required)", _HELP_EMBEDDINGS
        )
        self.embedding_list = QListWidget()
        self.embedding_list.itemDoubleClicked.connect(lambda _i: self._edit_embedding())
        self.embedding_list.currentRowChanged.connect(
            lambda _i: self._update_embedding_buttons()
        )
        body.addWidget(self.embedding_list)

        row = QHBoxLayout()
        self.add_emb_btn = QPushButton("Add…")
        self.add_emb_btn.setEnabled(False)
        self.add_emb_btn.clicked.connect(self._add_embedding)
        self.edit_emb_btn = QPushButton("Edit…")
        self.edit_emb_btn.clicked.connect(self._edit_embedding)
        self.remove_emb_btn = QPushButton("Remove")
        self.remove_emb_btn.clicked.connect(self._remove_embedding)
        row.addWidget(self.add_emb_btn)
        row.addWidget(self.edit_emb_btn)
        row.addWidget(self.remove_emb_btn)
        row.addStretch(1)
        body.addLayout(row)

        # Spells out why Add is disabled (points 2): a tooltip alone is missable.
        self.embedding_hint = QLabel("Load a meta table above to add embeddings.")
        self.embedding_hint.setStyleSheet("color: #a06000; font-size: 12px;")
        body.addWidget(self.embedding_hint)
        return group

    def _build_neuroglancer_group(self):
        group, body, toggle = self._section(
            "3D viewer (optional)", _HELP_NGL, checkable=True
        )
        self.ngl_enabled = toggle
        form = _grow_form()
        body.addLayout(form)

        self.ngl_source_edit = QLineEdit()
        self.ngl_source_edit.setPlaceholderText(
            "source URL, a meta column name, or leave blank to use the neuroglancer column"
        )
        # A manual edit opts out of the auto-fill from the neuroglancer column.
        self.ngl_source_edit.textEdited.connect(
            lambda _t: setattr(self, "_ngl_source_autofilled", False)
        )
        form.addRow("Source", self.ngl_source_edit)

        self.ngl_mesh_edit = QLineEdit()
        self.ngl_mesh_edit.setPlaceholderText("neuropil mesh path or URL")
        form.addRow(
            "Neuropil mesh",
            self._picker_row(
                self.ngl_mesh_edit,
                lambda: self.ngl_mesh_edit.setText(
                    _browse_open(self, "Pick a neuropil mesh", "Meshes (*.ply *.obj *.stl);;All files (*)")
                    or self.ngl_mesh_edit.text()
                ),
            ),
        )
        return group

    def _clash_label(self):
        """A hidden, amber, word-wrapped label for a per-field clash note."""
        label = QLabel("")
        label.setWordWrap(True)
        label.setStyleSheet("color: #a06000; font-size: 12px;")
        label.setVisible(False)
        return label

    def _picker_row(self, line_edit, on_browse):
        """Wrap ``line_edit`` + a Browse button into a full-width row widget."""
        container = QWidget()
        row = QHBoxLayout(container)
        row.setContentsMargins(0, 0, 0, 0)
        browse = QPushButton("Browse…")
        browse.clicked.connect(on_browse)
        row.addWidget(line_edit)
        row.addWidget(browse)
        return container

    # ---- actions ----------------------------------------------------------

    def _browse_output(self):
        path = QFileDialog.getExistingDirectory(self, "Choose output folder")
        if path:
            self.output_edit.setText(path)

    def _browse_meta(self):
        path = _browse_open(self, "Pick a meta table", _TABLE_FILTER)
        if path:
            self.meta_edit.setText(path)
            self._load_meta(path)

    def _load_meta(self, path):
        """Read the meta table and refresh the column dropdowns."""
        try:
            meta = read_table(path)
        except Exception as e:
            self._meta = None
            self.meta_status.setText(f"Could not read table: {e}")
            self.meta_status.setStyleSheet("color: #a00; font-size: 12px;")
            for combo in (self.id_combo, self.label_combo, self.dataset_combo,
                          self.color_combo, self.ngl_column_combo):
                combo.blockSignals(True)
                combo.clear()
                combo.blockSignals(False)
            for lbl in (self.id_clash_label, self.label_clash_label, self.dataset_clash_label):
                lbl.setVisible(False)
            self._set_embeddings_enabled(False, "Load a valid meta table to add embeddings.")
            return

        self._meta = meta
        columns = [str(c) for c in meta.columns]

        # Populate the id/label/dataset mapping combos, defaulting to same-named
        # columns when the table already uses the conventional names.
        for combo, target in (
            (self.id_combo, "id"),
            (self.label_combo, "label"),
            (self.dataset_combo, "dataset"),
        ):
            combo.blockSignals(True)
            combo.clear()
            combo.addItems(columns)
            idx = combo.findText(target)
            combo.setCurrentIndex(idx if idx >= 0 else 0)
            combo.blockSignals(False)

        self.meta_status.setText(f"Loaded {len(meta)} rows, {len(columns)} columns.")
        self.meta_status.setStyleSheet("color: #2a2; font-size: 12px;")

        self._refresh_meta_mapping()

        # Default the optional color / neuroglancer columns to the conventional
        # names. Setting the neuroglancer column (unblocked) fires
        # `_on_ngl_column_changed`, which turns the 3D viewer on.
        if self.color_combo.findText("color") >= 0:
            self.color_combo.setCurrentText("color")
        if self.ngl_column_combo.findText("source") >= 0:
            self.ngl_column_combo.setCurrentText("source")

    def _current_mapping(self):
        """Return the (id, label, dataset) source column names currently chosen."""
        return (
            self.id_combo.currentText().strip(),
            self.label_combo.currentText().strip(),
            self.dataset_combo.currentText().strip(),
        )

    def _refresh_meta_mapping(self):
        """React to a mapping change: show clashes, refresh downstream combos."""
        if self._meta is None:
            return
        id_col, label_col, dataset_col = self._current_mapping()
        if not (id_col and label_col and dataset_col):
            return

        plan = plan_meta_remap(
            [str(c) for c in self._meta.columns], id_col, label_col, dataset_col
        )

        # One clash line per field so the problem is pinned to the right row.
        field_warnings = plan["field_warnings"]
        for field, lbl in (
            ("id", self.id_clash_label),
            ("label", self.label_clash_label),
            ("dataset", self.dataset_clash_label),
        ):
            message = field_warnings.get(field)
            lbl.setText(message or "")
            lbl.setVisible(bool(message))

        # Color / neuroglancer columns are picked from what the project will
        # actually contain, so repopulate them from the remapped column names
        # (keeping any selection).
        for combo in (self.color_combo, self.ngl_column_combo):
            current = combo.currentText()
            combo.blockSignals(True)
            combo.clear()
            combo.addItem("(none)")
            combo.addItems(plan["final_columns"])
            idx = combo.findText(current)
            combo.setCurrentIndex(idx if idx >= 0 else 0)
            combo.blockSignals(False)

        self._set_embeddings_enabled(True)

    def _build_meta_frame(self):
        """Return the remapped meta DataFrame that will be saved."""
        id_col, label_col, dataset_col = self._current_mapping()
        return apply_meta_remap(self._meta, id_col, label_col, dataset_col)

    def _set_embeddings_enabled(self, enabled, hint=None):
        """Enable adding embeddings and show/hide the explanatory hint."""
        self.add_emb_btn.setEnabled(enabled)
        if enabled:
            self.embedding_hint.setVisible(False)
        else:
            self.embedding_hint.setText(hint or "Load a meta table above to add embeddings.")
            self.embedding_hint.setVisible(True)
        self._update_embedding_buttons()

    def _meta_columns(self):
        """Columns the embedding config can pick from — i.e. the *remapped* set."""
        if self._meta is None:
            return []
        id_col, label_col, dataset_col = self._current_mapping()
        if not (id_col and label_col and dataset_col):
            return [str(c) for c in self._meta.columns]
        return plan_meta_remap(
            [str(c) for c in self._meta.columns], id_col, label_col, dataset_col
        )["final_columns"]

    def _add_embedding(self):
        dlg = EmbeddingConfigDialog(self, meta_columns=self._meta_columns())
        if dlg.exec() == QDialog.Accepted:
            cfg = dlg.config()
            self._embeddings.append(cfg)
            self.embedding_list.addItem(EmbeddingConfigDialog.describe(cfg))
            self._update_embedding_buttons()

    def _edit_embedding(self):
        row = self.embedding_list.currentRow()
        if row < 0:
            return
        dlg = EmbeddingConfigDialog(
            self, meta_columns=self._meta_columns(), config=self._embeddings[row]
        )
        if dlg.exec() == QDialog.Accepted:
            cfg = dlg.config()
            self._embeddings[row] = cfg
            self.embedding_list.item(row).setText(EmbeddingConfigDialog.describe(cfg))

    def _remove_embedding(self):
        row = self.embedding_list.currentRow()
        if row < 0:
            return
        self._embeddings.pop(row)
        self.embedding_list.takeItem(row)
        self._update_embedding_buttons()

    def _update_embedding_buttons(self):
        """Edit/Remove only make sense with an embedding selected (point 3)."""
        has_selection = self.embedding_list.currentRow() >= 0
        self.edit_emb_btn.setEnabled(has_selection)
        self.remove_emb_btn.setEnabled(has_selection)
        self._update_build_button()

    def _on_ngl_column_changed(self, text):
        """Selecting a neuroglancer meta column turns on and fills the 3D viewer."""
        if not hasattr(self, "ngl_enabled"):
            return
        text = (text or "").strip()
        if not text or text == "(none)":
            return
        self.ngl_enabled.setChecked(True)
        # Fill the viewer's Source with the column name unless the user has typed
        # their own value there (tracked via `_ngl_source_autofilled`).
        if not self.ngl_source_edit.text().strip() or self._ngl_source_autofilled:
            self.ngl_source_edit.setText(text)
            self._ngl_source_autofilled = True

    def _update_build_button(self):
        """Enable Build only once output folder, meta table and >=1 embedding are set."""
        if not hasattr(self, "build_btn"):
            return
        ready = (
            bool(self.output_edit.text().strip())
            and self._meta is not None
            and bool(self._embeddings)
        )
        self.build_btn.setEnabled(ready)

    # ---- build ------------------------------------------------------------

    def _on_build(self):
        """Assemble a ProjectBuilder from the form and save it."""
        output = self.output_edit.text().strip()
        if not output:
            QMessageBox.warning(self, "Missing output", "Choose an output folder.")
            return
        if self._meta is None:
            QMessageBox.warning(self, "Missing meta", "Load a meta table first.")
            return
        if not self._embeddings:
            QMessageBox.warning(self, "Missing embedding", "Add at least one embedding.")
            return

        progress = QProgressDialog("Building project...", None, 0, 0, self)
        progress.setWindowModality(Qt.ApplicationModal)
        progress.setCancelButton(None)
        progress.show()
        try:
            builder = ProjectBuilder(
                output,
                name=self.name_edit.text().strip() or None,
                description=self.description_edit.text().strip() or None,
                dataset=self.dataset_edit.text().strip() or None,
            )

            color = self.color_combo.currentText()
            source = self.ngl_column_combo.currentText()
            builder.set_meta(
                self._build_meta_frame(),
                source=source if source != "(none)" else None,
                color_column=color if color != "(none)" else None,
            )

            for cfg in self._embeddings:
                self._add_embedding_to_builder(builder, cfg)

            if self.ngl_enabled.isChecked():
                ngl_source = self.ngl_source_edit.text().strip() or (
                    source if source != "(none)" else None
                )
                mesh = self.ngl_mesh_edit.text().strip() or None
                if ngl_source or mesh:
                    builder.set_neuroglancer(source=ngl_source, neuropil_mesh=mesh)

            info_path = builder.save()
        except Exception as e:
            progress.close()
            logger.exception("Failed to build project")
            QMessageBox.critical(self, "Build failed", str(e))
            return
        finally:
            progress.close()

        self.built_path = info_path.parent
        # When the project will be opened straight away, that load is the
        # confirmation; only pop a "wrote to ..." notice when it will not.
        if not self.open_after_build.isChecked():
            QMessageBox.information(
                self, "Project built", f"Wrote project to:\n{self.built_path}"
            )
        self.accept()

    def _add_embedding_to_builder(self, builder, cfg):
        """Read this embedding's files and register it on the builder."""
        kwargs = {"name": cfg.get("name")}
        if cfg.get("embeddings_file"):
            kwargs["embeddings"] = read_table(cfg["embeddings_file"])
        else:
            kwargs["columns"] = cfg["columns"]
        if cfg.get("distances_file"):
            kwargs["distances"] = read_table(cfg["distances_file"])
            kwargs["distances_type"] = cfg.get("distances_type", "connectivity")
            kwargs["distances_metric"] = cfg.get("distances_metric")
        if cfg.get("features_file"):
            kwargs["features"] = read_table(cfg["features_file"])
            kwargs["features_type"] = cfg.get("features_type", "connectivity")
        builder.add_embedding(**kwargs)
