import re
import logging
import warnings
import pyperclip

import pandas as pd
import numpy as np

from PySide6 import QtWidgets, QtCore
from concurrent.futures import ThreadPoolExecutor

from ...embeddings import (
    is_precomputed_distance_matrix,
    make_embedding_estimator,
    prepare_embedding_input,
)
from ...utils import labels_to_colors, is_color_column


CLIO_CLIENT = None
CLIO_ANN = None
NEUPRINT_CLIENT = None
FLYWIRE_ANN = None
HB_ANN = None

logger = logging.getLogger(__name__)


CLUSTER_DATA_EMBEDDING_OPTION = "embedding (2D)"
CLUSTER_DATA_OPTION = "cluster (bigclust)"
CLUSTER_DATA_COLUMN = "bigclust_cluster"
CLUSTER_HOMOGENEOUS_LABEL_CURRENT = "Current labels"


def requires_selection(func):
    """Decorator to check if a selection is required."""

    def wrapper(self, *args, **kwargs):
        if self.figure.selected_ids is None or len(self.figure.selected_ids) == 0:
            self.figure.show_message("No neurons selected", color="red", duration=2)
            return
        return func(self, *args, **kwargs)

    return wrapper


class ScatterControls(QtWidgets.QWidget):
    """Controls for the scatter plot."""

    def __init__(self, figure):
        super().__init__()
        self.figure = figure
        self.setWindowTitle("Controls")
        self.label_overrides = {}

        # Build gui
        self.tab_layout = QtWidgets.QVBoxLayout()
        self.setLayout(self.tab_layout)

        self.tabs = QtWidgets.QTabWidget()
        self.tabs.setDocumentMode(True)
        self.tabs.setTabPosition(QtWidgets.QTabWidget.West)
        self.tabs.setMovable(True)

        self.tab_layout.addWidget(self.tabs)

        self.tab1 = QtWidgets.QWidget()
        self.tab3 = QtWidgets.QWidget()
        self.tab4 = QtWidgets.QWidget()
        self.tab5 = QtWidgets.QWidget()
        self.tab6 = QtWidgets.QWidget()
        self.tab7 = QtWidgets.QWidget()
        self.tab1_layout = QtWidgets.QVBoxLayout()
        self.tab4_layout = QtWidgets.QVBoxLayout()
        self.tab5_layout = QtWidgets.QVBoxLayout()
        self.tab6_layout = QtWidgets.QVBoxLayout()
        self.tab7_layout = QtWidgets.QVBoxLayout()
        self.tab1.setLayout(self.tab1_layout)
        self.tab4.setLayout(self.tab4_layout)
        self.tab5.setLayout(self.tab5_layout)
        self.tab6.setLayout(self.tab6_layout)
        self.tab7.setLayout(self.tab7_layout)
        self.tabs.addTab(self.tab1, "General")
        self.tabs.addTab(self.tab5, "Embeddings")
        self.tabs.addTab(self.tab6, "Fidelity")
        self.tabs.addTab(self.tab7, "Cluster")
        self.tabs.addTab(self.tab4, "Settings")

        self.build_control_gui()
        self.build_settings_gui()
        self.build_embeddings_gui()
        self.build_fidelity_gui()
        self.build_clusters_gui()
        # Holds the futures for requested data
        self.futures = {}
        self.pool = ThreadPoolExecutor(4)

    @property
    def meta_data(self):
        """Get the meta data."""
        return self.figure.metadata

    @property
    def labels(self):
        """Get unique labels."""
        if not hasattr(self, "_labels"):
            self._labels = np.unique(self.figure.labels)
        return self._labels

    @property
    def selected_indices(self):
        """Get the selected IDs."""
        selected = getattr(self.figure, "selected", None)
        if selected is None:
            return np.array([], dtype=int)
        return np.asarray(selected, dtype=int)

    def build_control_gui(self):
        """Build the GUI."""
        ########
        # Search
        ########
        search_group = QtWidgets.QGroupBox("Search")
        search_form = QtWidgets.QFormLayout()
        search_form.setContentsMargins(2, 2, 2, 2)
        search_form.setVerticalSpacing(2)
        search_form.setFieldGrowthPolicy(
            QtWidgets.QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow
        )
        search_group.setLayout(search_form)
        self.tab1_layout.addWidget(search_group)

        self.searchbar = QtWidgets.QLineEdit()
        self.searchbar.setToolTip(
            "Search for label(s) in the project. Search syntax:\n"
            " - by default will search for exact match among the labels\n"
            " - use a leading '/' to search for a regex\n"
            " - you can also search for IDs; multiple IDs must be comma-separated, e.g. '1,2,3'"
        )
        self.searchbar.setPlaceholderText("Hover for search syntax")
        self.searchbar.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Fixed,
        )
        self.searchbar.returnPressed.connect(self.find_next)
        # self.searchbar.textChanged.connect(self.figure.highlight_cluster)
        self.searchbar_completer = QtWidgets.QCompleter(self.labels)
        self.searchbar_completer.setCaseSensitivity(QtCore.Qt.CaseInsensitive)
        self.searchbar.setCompleter(self.searchbar_completer)
        search_form.addRow(self.searchbar)

        # Add buttons for previous/next
        self.button_layout = QtWidgets.QHBoxLayout()
        self.button_layout.setContentsMargins(0, 0, 0, 0)
        self.button_layout.setSpacing(0)
        self.prev_button = QtWidgets.QPushButton("Previous")
        self.prev_button.clicked.connect(self.find_previous)
        self.button_layout.addWidget(self.prev_button)
        self.find_sel_button = QtWidgets.QPushButton("Select")
        self.find_sel_button.setToolTip(
            "Select all objects matching the search term. Use Shift-Click to add to current selection."
        )
        self.find_sel_button.clicked.connect(self.find_select)
        self.button_layout.addWidget(self.find_sel_button)
        self.next_button = QtWidgets.QPushButton("Next")
        self.next_button.clicked.connect(self.find_next)
        self.button_layout.addWidget(self.next_button)
        search_form.addRow(self.button_layout)

        ########
        # Labels
        ########

        label_group = QtWidgets.QGroupBox("Labels")
        label_form = QtWidgets.QFormLayout()
        label_form.setContentsMargins(2, 2, 2, 2)
        label_form.setVerticalSpacing(2)
        label_group.setLayout(label_form)
        self.tab1_layout.addWidget(label_group)

        # Add dropdown to choose leaf labels
        self.label_combo_box = QtWidgets.QComboBox()
        label_form.addRow(QtWidgets.QLabel("Labels:"), self.label_combo_box)
        self.label_combo_box.currentIndexChanged.connect(self.set_labels)
        self._current_leaf_labels = self.label_combo_box.currentText()

        # Checkbox for whether to show label counts
        self.label_count_check = QtWidgets.QCheckBox("Show label counts")
        self.label_count_check.setToolTip("Whether to add counts to the labels.")
        self.label_count_check.setChecked(False)
        self.label_count_check.stateChanged.connect(self.set_label_counts)
        label_form.addRow(self.label_count_check)

        # Checkbox for whether to show label outlines
        self.label_outlines_check = QtWidgets.QCheckBox("Show label outlines")
        self.label_outlines_check.setToolTip(
            "Whether to add draw polygons around neurons with the same label."
        )
        self.label_outlines_check.setChecked(False)
        self.label_outlines_check.stateChanged.connect(self.set_label_outlines)
        label_form.addRow(self.label_outlines_check)

        ########
        # Colors
        ########

        color_group = QtWidgets.QGroupBox("Colors")
        color_form = QtWidgets.QFormLayout()
        color_form.setContentsMargins(2, 2, 2, 2)
        color_form.setVerticalSpacing(2)
        color_group.setLayout(color_form)
        self.tab1_layout.addWidget(color_group)

        # Add dropdowns to choose color mode

        self.color_combo_box = QtWidgets.QComboBox()
        color_form.addRow("Color by:", self.color_combo_box)
        self.palette_combo_box = QtWidgets.QComboBox()
        color_form.addRow("Palette:", self.palette_combo_box)
        self.palette_combo_box.setToolTip(
            "The color palette to use when coloring by labels. Ignored if column contains colors or if coloring by clusters."
        )
        self.palette_combo_box.addItems(
            [
                "seaborn:tab20",
                "vispy:husl",
                "matplotlib:coolwarm",
                "matplotlib:viridis",
                "glasbey:glasbey",
            ]
        )

        # Set the action for the color combo box
        self.color_combo_box.currentIndexChanged.connect(self.set_colors)
        self.palette_combo_box.currentIndexChanged.connect(self.set_colors)

        ########
        # Selection behavior
        ########

        selection_group = QtWidgets.QGroupBox("Selection Behavior")
        selection_form = QtWidgets.QFormLayout()
        selection_form.setContentsMargins(2, 2, 2, 2)
        selection_form.setVerticalSpacing(2)
        selection_group.setLayout(selection_form)
        self.tab1_layout.addWidget(selection_group)

        self.selection_restrict_toggle = QtWidgets.QToolButton()
        self.selection_restrict_toggle.setText("Restrict to datasets")
        self.selection_restrict_toggle.setCheckable(True)
        self.selection_restrict_toggle.setChecked(False)
        self.selection_restrict_toggle.setToolButtonStyle(
            QtCore.Qt.ToolButtonTextBesideIcon
        )
        self.selection_restrict_toggle.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Fixed,
        )
        self.selection_restrict_toggle.setIconSize(QtCore.QSize(10, 10))
        self.selection_restrict_toggle.setStyleSheet(
            "QToolButton {"
            "text-align: left;"
            "border: none;"
            "background: transparent;"
            "padding: 0px;"
            "}"
            "QToolButton:pressed, QToolButton:checked, QToolButton:hover {"
            "border: none;"
            "background: transparent;"
            "}"
        )
        self.selection_restrict_toggle.setArrowType(QtCore.Qt.RightArrow)
        self.selection_restrict_toggle.toggled.connect(
            self._toggle_selection_restrict_panel
        )
        selection_form.addRow(self.selection_restrict_toggle)

        self.selection_restrict_panel = QtWidgets.QWidget()
        selection_restrict_layout = QtWidgets.QVBoxLayout()
        selection_restrict_layout.setContentsMargins(12, 0, 0, 0)
        selection_restrict_layout.setSpacing(2)
        self.selection_restrict_panel.setLayout(selection_restrict_layout)
        self.selection_restrict_panel.setVisible(False)
        selection_form.addRow(self.selection_restrict_panel)

        self.selection_restrict_checks = {}
        self.selection_restrict_empty_label = QtWidgets.QLabel(
            "No datasets available"
        )
        self.selection_restrict_empty_label.setStyleSheet("color: palette(mid);")
        selection_restrict_layout.addWidget(self.selection_restrict_empty_label)

        self.update_selection_restrict_options()

        self.add_group_check = QtWidgets.QCheckBox("Add as group")
        self.add_group_check.setToolTip("Whether to add neurons as group to the viewer when selected")
        self.add_group_check.stateChanged.connect(self.set_add_group)
        self.add_group_check.setChecked(True)
        toggle_font = self.selection_restrict_toggle.font()
        if toggle_font.pointSizeF() > 0:
            toggle_font.setPointSizeF(toggle_font.pointSizeF() + 0.01)
        elif toggle_font.pixelSize() > 0:
            toggle_font.setPixelSize(toggle_font.pixelSize() + 0.01)
        self.selection_restrict_toggle.setFont(toggle_font)
        selection_form.addRow(self.add_group_check)

        self.dclick_deselect = QtWidgets.QCheckBox("Deselect on double-click")
        self.dclick_deselect.setToolTip("You can always deselect using ESC")
        self.dclick_deselect.setChecked(self.figure.deselect_on_dclick)
        self.dclick_deselect.stateChanged.connect(self.set_dclick_deselect)
        selection_form.addRow(self.dclick_deselect)

        self.empty_deselect = QtWidgets.QCheckBox("Deselect on empty selection")
        self.empty_deselect.setToolTip("You can always deselect using ESC")
        self.empty_deselect.setChecked(self.figure.deselect_on_empty)
        self.empty_deselect.stateChanged.connect(self.set_empty_deselect)
        selection_form.addRow(self.empty_deselect)

        # This would make it so the legend does not stretch when
        # we resize the window vertically
        self.tab1_layout.addStretch(1)

        return

    def build_settings_gui(self):
        # Add dropdown to determine render mode
        self.tab4_layout.addWidget(QtWidgets.QLabel("Render trigger:"))

        self.render_mode_dropdown = QtWidgets.QComboBox()
        self.render_mode_dropdown.setToolTip(
            "Set trigger for re-rendering the scene. See documentation for details."
        )
        self.render_mode_dropdown.addItems(["Continuous", "Reactive", "Active Window"])
        self.render_mode_dropdown.setItemData(
            0, "Continuously render the scene.", QtCore.Qt.ToolTipRole
        )
        self.render_mode_dropdown.setItemData(
            1,
            "Render only when the scene changes.",
            QtCore.Qt.ToolTipRole,
        )
        self.render_mode_dropdown.setItemData(
            2, "Render only when the window is active.", QtCore.Qt.ToolTipRole
        )
        render_trigger_vals = ["continuous", "reactive", "active_window"]
        # Set default item to whatever the currently set render trigger is
        self.render_mode_dropdown.setCurrentIndex(
            render_trigger_vals.index(self.figure.render_trigger)
        )
        self.render_mode_dropdown.currentIndexChanged.connect(
            lambda x: setattr(
                self.figure,
                "render_trigger",
                render_trigger_vals[self.render_mode_dropdown.currentIndex()],
            )
        )
        self.tab4_layout.addWidget(self.render_mode_dropdown)

        # Add slide for max frame rate
        label = QtWidgets.QLabel("Max frame rate:")
        label.setToolTip(
            "Set the maximum frame rate for the figure. Press F while the figure window is active to show current frame rate."
        )
        self.tab4_layout.addWidget(label)

        self.max_frame_rate_layout = QtWidgets.QHBoxLayout()
        self.tab4_layout.addLayout(self.max_frame_rate_layout)

        self.max_frame_rate_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.max_frame_rate_slider.setRange(5, 100)
        self.max_frame_rate_slider.setValue(self.figure.max_fps)
        self.max_frame_rate_slider.valueChanged.connect(
            lambda x: setattr(self.figure, "max_fps", int(x))
        )
        self.max_frame_rate_slider.valueChanged.connect(
            lambda x: self.max_frame_rate_value_label.setText(f"{x} FPS")
        )
        self.max_frame_rate_layout.addWidget(self.max_frame_rate_slider)

        self.max_frame_rate_value_label = QtWidgets.QLabel(
            f"{int(self.figure.max_fps)} FPS"
        )
        self.max_frame_rate_layout.addWidget(self.max_frame_rate_value_label)

        # Add SpinBox for font size
        hlayout = QtWidgets.QHBoxLayout()
        self.tab4_layout.addLayout(hlayout)
        label = QtWidgets.QLabel("Font size:")
        label.setToolTip("Set the font size for the labels in the figure.")
        hlayout.addWidget(label)
        self.font_size_slider = QtWidgets.QDoubleSpinBox()
        self.font_size_slider.setRange(.01, 200)
        self.font_size_slider.setValue(self.figure.font_size)
        self.font_size_slider.valueChanged.connect(
            lambda x: setattr(self.figure, "font_size", x)
        )
        hlayout.addWidget(self.font_size_slider)

        # Add slider for number of labels visible at once
        label = QtWidgets.QLabel("Max visible labels:")
        label.setToolTip(
            "Set the maximum number of labels visible at once. This is useful for large datasets. May negatively impact performance."
        )
        self.tab4_layout.addWidget(label)

        self.max_label_vis_layout = QtWidgets.QHBoxLayout()
        self.tab4_layout.addLayout(self.max_label_vis_layout)

        self.max_label_vis_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.max_label_vis_slider.setRange(1, 5_000)
        self.max_label_vis_slider.setValue(self.figure.label_vis_limit)
        self.max_label_vis_slider.valueChanged.connect(
            lambda x: setattr(self.figure, "label_vis_limit", int(x))
        )
        self.max_label_vis_slider.valueChanged.connect(
            lambda x: self.max_label_vis_value_label.setText(f"{x} labels")
        )
        self.max_label_vis_layout.addWidget(self.max_label_vis_slider)

        self.max_label_vis_value_label = QtWidgets.QLabel(
            f"{int(self.figure.label_vis_limit)} labels"
        )
        self.max_label_vis_layout.addWidget(self.max_label_vis_value_label)

        # This makes it so the legend does not stretch
        self.tab4_layout.addStretch(1)

    def build_embeddings_gui(self):
        """Build the GUI for the Embeddings tab."""
        # Top action row
        actions_row = QtWidgets.QHBoxLayout()
        self.tab5_layout.addLayout(actions_row)

        self.umap_button = QtWidgets.QPushButton("Re-calculate positions")
        self.umap_button.setToolTip(
            "Run dimensionality reduction on the current dataset. This will overwrite the current positions."
        )
        self.umap_button.clicked.connect(self.calculate_embeddings)
        actions_row.addWidget(self.umap_button)

        actions_row = QtWidgets.QHBoxLayout()
        self.tab5_layout.addLayout(actions_row)

        self.umap_auto_run = QtWidgets.QCheckBox("Auto run")
        self.umap_auto_run.setToolTip(
            "Automatically run dimensionality reduction when changing settings."
        )
        self.umap_auto_run.setChecked(False)
        self.umap_auto_run.stateChanged.connect(
            lambda: setattr(self.figure, "_auto_umap", self.umap_auto_run.isChecked())
        )
        actions_row.addWidget(self.umap_auto_run)

        self.umap_selection_only = QtWidgets.QCheckBox("Selection only")
        self.umap_selection_only.setToolTip(
            "Run on current selection only. This spawns a new figure."
        )
        self.umap_selection_only.setChecked(False)
        actions_row.addWidget(self.umap_selection_only)

        # Input group
        input_group = QtWidgets.QGroupBox("Input")
        input_form = QtWidgets.QFormLayout()
        input_form.setContentsMargins(8, 6, 8, 6)
        input_group.setLayout(input_form)
        self.tab5_layout.addWidget(input_group)

        self.umap_method_combo_box = QtWidgets.QComboBox()
        self.umap_method_combo_box.setToolTip(
            "Select the method to use for dimensionality reduction."
        )
        for item in ("UMAP", "MDS"):
            self.umap_method_combo_box.addItem(item)
        if getattr(self.figure, "feats", None) is not None:
            self.umap_method_combo_box.addItem("PaCMAP")
        self.umap_method_combo_box.currentIndexChanged.connect(
            self.update_embedding_settings
        )
        input_form.addRow("Method:", self.umap_method_combo_box)

        self.umap_dist_combo_box = QtWidgets.QComboBox()
        self.umap_dist_combo_box.setToolTip(
            "Select the data source used for embedding."
        )

        def update_and_calculate_embeddings_maybe():
            """Update the run button when the distance is changed."""
            self._update_embedding_input_controls()
            self.calculate_embeddings_maybe()

        self.umap_dist_combo_box.currentIndexChanged.connect(
            update_and_calculate_embeddings_maybe
        )
        input_form.addRow("Data:", self.umap_dist_combo_box)

        # Optional sub-selection of top-level feature groups for MultiIndex columns.
        self.umap_feature_subset_group = QtWidgets.QGroupBox("Feature Subset")
        feature_subset_layout = QtWidgets.QVBoxLayout()
        feature_subset_layout.setContentsMargins(8, 6, 8, 6)
        feature_subset_layout.setSpacing(2)
        self.umap_feature_subset_group.setLayout(feature_subset_layout)
        input_form.addRow(self.umap_feature_subset_group)

        self.umap_feature_partition_widget = QtWidgets.QWidget()
        partition_layout = QtWidgets.QVBoxLayout()
        partition_layout.setContentsMargins(0, 0, 0, 0)
        partition_layout.setSpacing(2)
        self.umap_feature_partition_widget.setLayout(partition_layout)
        feature_subset_layout.addWidget(self.umap_feature_partition_widget)
        self._embedding_partition_checks = {}

        # Feature-dependent options are grouped together and enabled only for feature vectors.
        self.feature_options_group = QtWidgets.QGroupBox("Feature Options")
        prep_form = QtWidgets.QFormLayout()
        prep_form.setContentsMargins(8, 6, 8, 6)
        self.feature_options_group.setLayout(prep_form)
        self.tab5_layout.addWidget(self.feature_options_group)

        # For non-square inputs (feature vectors), allow choosing a distance metric.
        self.umap_feature_metric_widget = QtWidgets.QWidget()
        metric_layout = QtWidgets.QHBoxLayout()
        metric_layout.setContentsMargins(0, 0, 0, 0)
        self.umap_feature_metric_widget.setLayout(metric_layout)
        self.umap_feature_metric_combo_box = QtWidgets.QComboBox()
        self.umap_feature_metric_combo_box.setToolTip(
            "Distance metric used when embedding feature vectors."
        )
        self.umap_feature_metric_combo_box.addItems(
            ["cosine", "euclidean", "manhattan", "correlation", "chebyshev"]
        )
        self.umap_feature_metric_combo_box.currentIndexChanged.connect(
            self.calculate_embeddings_maybe
        )
        metric_layout.addWidget(self.umap_feature_metric_combo_box)
        prep_form.addRow("Feature metric:", self.umap_feature_metric_widget)

        # Optional feature rebalancing to reduce dominance of a few dimensions.
        self.umap_feature_rebalance_widget = QtWidgets.QWidget()
        rebalance_layout = QtWidgets.QHBoxLayout()
        rebalance_layout.setContentsMargins(0, 0, 0, 0)
        self.umap_feature_rebalance_widget.setLayout(rebalance_layout)
        self.umap_feature_rebalance_combo_box = QtWidgets.QComboBox()
        self.umap_feature_rebalance_combo_box.setToolTip(
            "Preprocessing applied per feature before embedding to reduce feature dominance."
        )
        self.umap_feature_rebalance_combo_box.addItems(
            ["none", "z-score", "robust (median/IQR)", "log1p + z-score"]
        )
        self.umap_feature_rebalance_combo_box.currentIndexChanged.connect(
            self.calculate_embeddings_maybe
        )
        rebalance_layout.addWidget(self.umap_feature_rebalance_combo_box)
        prep_form.addRow("Feature rebalancing:", self.umap_feature_rebalance_widget)

        pca_widget = QtWidgets.QWidget()
        pca_row = QtWidgets.QHBoxLayout()
        pca_row.setContentsMargins(0, 0, 0, 0)
        pca_widget.setLayout(pca_row)

        self.pca_check = QtWidgets.QCheckBox("Enable PCA")
        self.pca_check.setToolTip(
            "Reduce dimensionality before embedding (feature vectors only)."
        )
        self.pca_check.setChecked(False)
        pca_row.addWidget(self.pca_check)

        self.pca_n_components_slider = QtWidgets.QSpinBox()
        self.pca_n_components_slider.setRange(1, 2000)
        self.pca_n_components_slider.setSingleStep(1)
        self.pca_n_components_slider.setValue(100)
        self.pca_n_components_slider.setToolTip(
            "Number of components to keep after PCA."
        )
        self.pca_n_components_slider.valueChanged.connect(
            self.calculate_embeddings_maybe
        )
        self.pca_n_components_slider.setEnabled(self.pca_check.isChecked())
        pca_row.addWidget(self.pca_n_components_slider)

        self.pca_check.stateChanged.connect(
            lambda _: self.pca_n_components_slider.setEnabled(
                self.pca_check.isChecked()
            )
        )
        prep_form.addRow("PCA:", pca_widget)

        # Method-specific settings
        settings_group = QtWidgets.QGroupBox("Method Settings")
        settings_layout = QtWidgets.QVBoxLayout()
        settings_layout.setContentsMargins(8, 6, 8, 6)
        settings_group.setLayout(settings_layout)
        self.tab5_layout.addWidget(settings_group)

        method_common_widget = QtWidgets.QWidget()
        method_common_form = QtWidgets.QFormLayout()
        method_common_form.setContentsMargins(0, 0, 0, 0)
        method_common_widget.setLayout(method_common_form)
        settings_layout.addWidget(method_common_widget)

        self.umap_random_seed = QtWidgets.QLineEdit()
        self.umap_random_seed.setToolTip(
            "Random seed. Leave empty for random initialization."
        )
        self.umap_random_seed.setPlaceholderText("random initialization")
        self.umap_random_seed.setText(str(42))
        self.umap_random_seed.textChanged.connect(
            lambda x: self.calculate_embeddings_maybe()
        )
        method_common_form.addRow("Random seed:", self.umap_random_seed)

        self.umap_settings_widget = QtWidgets.QWidget()
        self.umap_settings_layout = QtWidgets.QFormLayout()
        self.umap_settings_layout.setContentsMargins(0, 0, 0, 0)
        self.umap_settings_widget.setLayout(self.umap_settings_layout)
        settings_layout.addWidget(self.umap_settings_widget)

        self.umap_n_neighbors_slider = QtWidgets.QSpinBox()
        self.umap_n_neighbors_slider.setRange(1, 200)
        self.umap_n_neighbors_slider.setSingleStep(1)
        self.umap_n_neighbors_slider.setValue(10)
        self.umap_n_neighbors_slider.valueChanged.connect(
            self.calculate_embeddings_maybe
        )
        self.umap_settings_layout.addRow(
            "Number of neighbors:", self.umap_n_neighbors_slider
        )

        self.umap_min_dist_slider = QtWidgets.QDoubleSpinBox()
        self.umap_min_dist_slider.setRange(0.0, 100.0)
        self.umap_min_dist_slider.setSingleStep(0.05)
        self.umap_min_dist_slider.setValue(0.1)
        self.umap_min_dist_slider.valueChanged.connect(self.calculate_embeddings_maybe)
        self.umap_settings_layout.addRow("Minimum distance:", self.umap_min_dist_slider)

        self.umap_spread_slider = QtWidgets.QDoubleSpinBox()
        self.umap_spread_slider.setRange(0.0, 1000.0)
        self.umap_spread_slider.setSingleStep(0.05)
        self.umap_spread_slider.setValue(1)
        self.umap_spread_slider.valueChanged.connect(self.calculate_embeddings_maybe)
        self.umap_settings_layout.addRow("Spread:", self.umap_spread_slider)

        self.umap_set_op_mix_ratio_spinbox = QtWidgets.QDoubleSpinBox()
        self.umap_set_op_mix_ratio_spinbox.setRange(0.0, 1.0)
        self.umap_set_op_mix_ratio_spinbox.setSingleStep(0.01)
        self.umap_set_op_mix_ratio_spinbox.setDecimals(2)
        self.umap_set_op_mix_ratio_spinbox.setValue(1.0)
        self.umap_set_op_mix_ratio_spinbox.setToolTip(
            "Adjust the UMAP set_op_mix_ratio between 0 and 1. Lower values encourage the embedding to preserve outliers as outlying."
        )
        self.umap_set_op_mix_ratio_spinbox.valueChanged.connect(
            self.calculate_embeddings_maybe
        )
        self.umap_settings_layout.addRow(
            "Set operation mix ratio:", self.umap_set_op_mix_ratio_spinbox
        )

        self.umap_densmap_check = QtWidgets.QCheckBox("Use DensMAP")
        self.umap_densmap_check.setToolTip(
            "Use UMAP's densMAP extension to better preserve local density relationships in the embedding."
        )
        self.umap_densmap_check.setChecked(False)
        self.umap_densmap_check.stateChanged.connect(
            lambda _: (self._update_densmap_controls(), self.calculate_embeddings_maybe())
        )
        self.umap_settings_layout.addRow("DensMAP:", self.umap_densmap_check)

        self.umap_dens_lambda_label = QtWidgets.QLabel("DensMAP lambda:")
        self.umap_dens_lambda_spinbox = QtWidgets.QDoubleSpinBox()
        self.umap_dens_lambda_spinbox.setRange(0.0, 10.0)
        self.umap_dens_lambda_spinbox.setSingleStep(0.1)
        self.umap_dens_lambda_spinbox.setDecimals(3)
        self.umap_dens_lambda_spinbox.setValue(2.0)
        self.umap_dens_lambda_spinbox.setToolTip(
            "Strength of the DensMAP density-preservation penalty. Higher values preserve density more strongly."
        )
        self.umap_dens_lambda_spinbox.valueChanged.connect(self.calculate_embeddings_maybe)
        self.umap_settings_layout.addRow(
            self.umap_dens_lambda_label,
            self.umap_dens_lambda_spinbox,
        )
        self._update_densmap_controls()

        self.mds_settings_widget = QtWidgets.QWidget()
        self.mds_settings_layout = QtWidgets.QFormLayout()
        self.mds_settings_layout.setContentsMargins(0, 0, 0, 0)
        self.mds_settings_widget.setLayout(self.mds_settings_layout)
        settings_layout.addWidget(self.mds_settings_widget)

        self.mds_n_init_slider = QtWidgets.QSpinBox()
        self.mds_n_init_slider.setRange(1, 200)
        self.mds_n_init_slider.setSingleStep(1)
        self.mds_n_init_slider.setValue(4)
        self.mds_n_init_slider.valueChanged.connect(self.calculate_embeddings_maybe)
        self.mds_settings_layout.addRow(
            "Number of initializations:", self.mds_n_init_slider
        )

        self.mds_max_iter_slider = QtWidgets.QSpinBox()
        self.mds_max_iter_slider.setRange(1, 10000)
        self.mds_max_iter_slider.setSingleStep(1)
        self.mds_max_iter_slider.setValue(300)
        self.mds_max_iter_slider.valueChanged.connect(self.calculate_embeddings_maybe)
        self.mds_settings_layout.addRow("Max iterations:", self.mds_max_iter_slider)

        self.mds_eps_slider = QtWidgets.QDoubleSpinBox()
        self.mds_eps_slider.setRange(0.0000, 1.0000)
        self.mds_eps_slider.setSingleStep(0.001)
        self.mds_eps_slider.setDecimals(4)
        self.mds_eps_slider.setValue(0.001)
        self.mds_eps_slider.valueChanged.connect(self.calculate_embeddings_maybe)
        self.mds_settings_layout.addRow("Relative tolerance:", self.mds_eps_slider)

        # Populate options
        self.update_umap_options()

        self.tab5_layout.addStretch(1)

        # Make sure method settings are in sync by default.
        self.update_embedding_settings()

    def build_fidelity_gui(self):
        """Build the GUI for the Fidelity tab."""

        # Neighboorhood fidelity settings
        settings_group = QtWidgets.QGroupBox("Neighborhood Fidelity")
        settings_form = QtWidgets.QFormLayout()
        settings_form.setContentsMargins(8, 6, 8, 6)
        settings_group.setLayout(settings_form)
        self.tab6_layout.addWidget(settings_group)

        self.add_tooltip(
            settings_group,
            "Neighborhood fidelity measures how well local relationships are "
            "preserved by the 2D embedding. Higher values mean nearby neurons "
            "in the original space remain nearby in the plot.",
            anchor="group_label",
        )

        self.fidelity_point_size_check = QtWidgets.QCheckBox()
        self.fidelity_point_size_check.setToolTip(
            "Show neighborhood fidelity scores as point sizes."
        )
        self.fidelity_point_size_check.setChecked(False)
        self.fidelity_point_size_check.stateChanged.connect(self.set_fidelity_mode)
        settings_form.addRow("Show:", self.fidelity_point_size_check)

        self.fidelity_k_slider = QtWidgets.QSpinBox()
        self.fidelity_k_slider.setRange(1, 200)
        self.fidelity_k_slider.setSingleStep(1)
        self.fidelity_k_slider.setValue(10)
        self.fidelity_k_slider.setToolTip(
            "Number of nearest neighbors (k) used to compute neighborhood fidelity."
        )
        self.fidelity_k_slider.valueChanged.connect(self.set_fidelity_mode)
        settings_form.addRow("k neighbors:", self.fidelity_k_slider)

        self.fidelity_distance_combo_box = QtWidgets.QComboBox()
        self.fidelity_distance_combo_box.setToolTip(
            "Distance metric used when fidelity is computed from feature vectors."
        )
        self.fidelity_distance_combo_box.addItems(
            ["euclidean", "cosine", "manhattan", "correlation", "chebyshev"]
        )
        self.fidelity_distance_combo_box.currentIndexChanged.connect(
            self.set_fidelity_mode
        )
        settings_form.addRow("Distance:", self.fidelity_distance_combo_box)

        self.fidelity_use_rank_check = QtWidgets.QCheckBox("Use rank")
        self.fidelity_use_rank_check.setToolTip(
            "Use rank-based neighborhood overlap instead of distance-weighted overlap."
        )
        self.fidelity_use_rank_check.setChecked(True)
        self.fidelity_use_rank_check.stateChanged.connect(self.set_fidelity_mode)
        settings_form.addRow(self.fidelity_use_rank_check)

        # KNN lines
        settings_group = QtWidgets.QGroupBox("K-Nearest Neighbors")
        settings_form = QtWidgets.QFormLayout()
        settings_form.setContentsMargins(8, 6, 8, 6)
        settings_group.setLayout(settings_form)
        self.tab6_layout.addWidget(settings_group)

        self.add_tooltip(
            settings_group,
            "Show lines connecting each point to its k nearest neighbors in the original feature space. "
            "This can help visualize how well local relationships are preserved in the embedding.",
            anchor="group_label",
        )

        self.fidelity_knn_combo_box = QtWidgets.QComboBox()
        self.fidelity_knn_combo_box.setToolTip(
            "Show lines connecting each point to its k nearest neighbors in the original feature space."
        )
        self.fidelity_knn_combo_box.addItems(["Off", "Selected only", "All points"])
        self.fidelity_knn_combo_box.currentIndexChanged.connect(self.set_knn_edges)
        settings_form.addRow("Show:", self.fidelity_knn_combo_box)

        self.fidelity_knn_k_slider = QtWidgets.QSpinBox()
        self.fidelity_knn_k_slider.setRange(1, 200)
        self.fidelity_knn_k_slider.setSingleStep(1)
        self.fidelity_knn_k_slider.setValue(10)
        self.fidelity_knn_k_slider.setToolTip(
            "Number of nearest neighbors (k) used to compute neighborhood fidelity."
        )
        self.fidelity_knn_k_slider.valueChanged.connect(self.set_knn_edges)
        settings_form.addRow("k neighbors:", self.fidelity_knn_k_slider)

        self.fidelity_knn_distance_combo_box = QtWidgets.QComboBox()
        self.fidelity_knn_distance_combo_box.setToolTip(
            "Distance metric used when fidelity is computed from feature vectors."
        )
        self.fidelity_knn_distance_combo_box.addItems(
            ["euclidean", "cosine", "manhattan", "correlation", "chebyshev"]
        )
        self.fidelity_knn_distance_combo_box.currentIndexChanged.connect(
            self.set_knn_edges
        )
        settings_form.addRow("Distance:", self.fidelity_knn_distance_combo_box)

        # Distances edges
        settings_group = QtWidgets.QGroupBox("Distances")
        settings_form = QtWidgets.QFormLayout()
        settings_form.setContentsMargins(8, 6, 8, 6)
        settings_group.setLayout(settings_form)
        self.tab6_layout.addWidget(settings_group)

        self.add_tooltip(
            settings_group,
            "Show lines connecting points closer than a given distance in the original feature space. "
            "This can help visualize how well local relationships are preserved in the embedding.",
            anchor="group_label",
        )

        # Add a checkbox + spinbox to show distances as edges
        self.show_distance_edges_check = QtWidgets.QCheckBox()
        self.show_distance_edges_check.setToolTip(
            "Whether to show actual distances as edges between points."
        )
        self.show_distance_edges_check.setChecked(False)
        settings_form.addRow("Show:", self.show_distance_edges_check)

        self.distance_edges_slider = QtWidgets.QDoubleSpinBox()
        self.distance_edges_slider.setRange(0.0, 1.0)
        self.distance_edges_slider.setSingleStep(0.05)
        self.distance_edges_slider.setValue(self.figure.distance_edges_threshold)
        self.distance_edges_slider.setDecimals(2)
        settings_form.addRow("Threshold:", self.distance_edges_slider)

        self.distance_edges_slider.valueChanged.connect(
            lambda x: setattr(self.figure, "distance_edges_threshold", float(x))
        )
        self.show_distance_edges_check.stateChanged.connect(
            lambda x: setattr(self.figure, "show_distance_edges", bool(x))
        )
        self.show_distance_edges_check.stateChanged.connect(
            lambda x: self.distance_edges_slider.setEnabled(
                self.show_distance_edges_check.isChecked()
            )
        )

        self.tab6_layout.addStretch(1)

        self.update_fidelity_options()

    def build_clusters_gui(self):
        """Build the GUI for the Clusters tab."""
        # Top action row
        actions_row = QtWidgets.QHBoxLayout()
        actions_row.setContentsMargins(0, 0, 0, 0)
        actions_row.setSpacing(0)
        self.tab7_layout.addLayout(actions_row)

        self.cluster_run_button = QtWidgets.QPushButton("Run")
        self.cluster_run_button.setToolTip(
            "Partition the data into clusters using the selected algorithm."
        )
        self.cluster_run_button.clicked.connect(self._run_clustering)
        actions_row.addWidget(self.cluster_run_button)

        self.cluster_clear_button = QtWidgets.QPushButton("Clear")
        self.cluster_clear_button.setToolTip("Reset points to their defaults.")
        self.cluster_clear_button.clicked.connect(self._clear_cluster)
        actions_row.addWidget(self.cluster_clear_button)

        self.cluster_load_button = QtWidgets.QPushButton()
        self.cluster_load_button.setIcon(
            self.style().standardIcon(QtWidgets.QStyle.SP_DialogOpenButton)
        )
        self.cluster_load_button.setAccessibleName("Load")
        self.cluster_load_button.setToolTip(
            "Load cluster assignments from a previously exported CSV file."
        )
        self.cluster_load_button.clicked.connect(self._load_cluster_labels)
        actions_row.addWidget(self.cluster_load_button)

        # Input group
        input_group = QtWidgets.QGroupBox("Input")
        input_form = QtWidgets.QFormLayout()
        input_form.setContentsMargins(8, 6, 8, 6)
        input_group.setLayout(input_form)
        self.tab7_layout.addWidget(input_group)

        self.cluster_data_combo_box = QtWidgets.QComboBox()
        self.cluster_data_combo_box.setToolTip("Select the data source for clustering.")
        self.cluster_data_combo_box.currentIndexChanged.connect(
            self._update_cluster_input_controls
        )
        input_form.addRow("Data:", self.cluster_data_combo_box)

        self.cluster_metric_combo_box = QtWidgets.QComboBox()
        self.cluster_metric_combo_box.setToolTip(
            "Distance metric used when clustering from feature vectors."
        )
        self.cluster_metric_combo_box.addItems(
            ["euclidean", "cosine", "manhattan", "correlation", "chebyshev"]
        )
        input_form.addRow("Metric:", self.cluster_metric_combo_box)

        # Algorithm group
        algo_group = QtWidgets.QGroupBox("Algorithm")
        algo_layout = QtWidgets.QVBoxLayout()
        algo_layout.setContentsMargins(8, 6, 8, 6)
        algo_group.setLayout(algo_layout)
        self.tab7_layout.addWidget(algo_group)

        algo_form = QtWidgets.QFormLayout()
        algo_form.setContentsMargins(0, 0, 0, 0)
        algo_layout.addLayout(algo_form)

        self.cluster_method_combo_box = QtWidgets.QComboBox()
        self.cluster_method_combo_box.addItems(["HDBSCAN", "Agglomerative", "K-Means"])
        self.cluster_method_combo_box.setToolTip("Select the clustering algorithm.")
        self.cluster_method_combo_box.currentIndexChanged.connect(
            self._update_cluster_method_settings
        )
        algo_form.addRow("Method:", self.cluster_method_combo_box)

        # --- HDBSCAN settings ---
        self.cluster_hdbscan_widget = QtWidgets.QWidget()
        hdbscan_form = QtWidgets.QFormLayout()
        hdbscan_form.setContentsMargins(0, 0, 0, 0)
        self.cluster_hdbscan_widget.setLayout(hdbscan_form)
        algo_layout.addWidget(self.cluster_hdbscan_widget)

        self.cluster_hdbscan_min_cluster_size = QtWidgets.QSpinBox()
        self.cluster_hdbscan_min_cluster_size.setRange(2, 10000)
        self.cluster_hdbscan_min_cluster_size.setValue(5)
        self.cluster_hdbscan_min_cluster_size.setToolTip(
            "Minimum number of points required to form a cluster."
        )
        hdbscan_form.addRow("Min cluster size:", self.cluster_hdbscan_min_cluster_size)

        self.cluster_hdbscan_min_samples = QtWidgets.QSpinBox()
        self.cluster_hdbscan_min_samples.setRange(0, 10000)
        self.cluster_hdbscan_min_samples.setValue(0)
        self.cluster_hdbscan_min_samples.setToolTip(
            "Minimum samples for a core point. 0 defaults to min_cluster_size."
        )
        hdbscan_form.addRow("Min samples:", self.cluster_hdbscan_min_samples)

        self.cluster_hdbscan_epsilon = QtWidgets.QDoubleSpinBox()
        self.cluster_hdbscan_epsilon.setRange(0.0, 100.0)
        self.cluster_hdbscan_epsilon.setSingleStep(0.01)
        self.cluster_hdbscan_epsilon.setDecimals(3)
        self.cluster_hdbscan_epsilon.setValue(0.0)
        self.cluster_hdbscan_epsilon.setToolTip(
            "Distance threshold below which clusters are merged."
        )
        hdbscan_form.addRow("Cluster epsilon:", self.cluster_hdbscan_epsilon)

        self.cluster_hdbscan_method_combo = QtWidgets.QComboBox()
        self.cluster_hdbscan_method_combo.addItems(["eom", "leaf"])
        self.cluster_hdbscan_method_combo.setToolTip(
            "'eom': excess of mass (larger clusters). "
            "'leaf': smaller, more granular clusters."
        )
        hdbscan_form.addRow("Selection method:", self.cluster_hdbscan_method_combo)

        self.cluster_hdbscan_label_noise_check = QtWidgets.QCheckBox(
            "Relabel noise points"
        )
        self.cluster_hdbscan_label_noise_check.setToolTip(
            "Assign HDBSCAN noise points (-1) to the nearest non-noise cluster."
        )
        self.cluster_hdbscan_label_noise_check.setChecked(False)
        self.cluster_hdbscan_label_noise_check.stateChanged.connect(
            self._update_hdbscan_noise_controls
        )
        hdbscan_form.addRow(self.cluster_hdbscan_label_noise_check)

        self.cluster_hdbscan_noise_threshold_row = QtWidgets.QWidget()
        noise_threshold_layout = QtWidgets.QHBoxLayout()
        noise_threshold_layout.setContentsMargins(0, 0, 0, 0)
        self.cluster_hdbscan_noise_threshold_row.setLayout(noise_threshold_layout)

        self.cluster_hdbscan_noise_threshold_check = QtWidgets.QCheckBox(
            "Use threshold"
        )
        self.cluster_hdbscan_noise_threshold_check.setToolTip(
            "Only relabel noise points if nearest-cluster distance is below threshold."
        )
        self.cluster_hdbscan_noise_threshold_check.setChecked(False)
        self.cluster_hdbscan_noise_threshold_check.stateChanged.connect(
            self._update_hdbscan_noise_controls
        )
        noise_threshold_layout.addWidget(self.cluster_hdbscan_noise_threshold_check)

        self.cluster_hdbscan_noise_threshold = QtWidgets.QDoubleSpinBox()
        self.cluster_hdbscan_noise_threshold.setRange(0.0, 1_000_000.0)
        self.cluster_hdbscan_noise_threshold.setSingleStep(0.05)
        self.cluster_hdbscan_noise_threshold.setDecimals(4)
        self.cluster_hdbscan_noise_threshold.setValue(1.0)
        self.cluster_hdbscan_noise_threshold.setToolTip(
            "Maximum distance for relabeling a noise point."
        )
        noise_threshold_layout.addWidget(self.cluster_hdbscan_noise_threshold)

        hdbscan_form.addRow(
            "Noise threshold:", self.cluster_hdbscan_noise_threshold_row
        )

        # --- Agglomerative settings ---
        self.cluster_agg_widget = QtWidgets.QWidget()
        agg_form = QtWidgets.QFormLayout()
        agg_form.setContentsMargins(0, 0, 0, 0)
        self.cluster_agg_widget.setLayout(agg_form)
        algo_layout.addWidget(self.cluster_agg_widget)

        self.cluster_agg_criterion_combo = QtWidgets.QComboBox()
        self.cluster_agg_criterion_combo.addItems(
            ["N clusters", "Distance threshold", "Homogeneous composition"]
        )
        self.cluster_agg_criterion_combo.setCurrentIndex(0)
        self.cluster_agg_criterion_combo.setToolTip(
            "Choose how to stop agglomerative merging: fixed number of "
            "clusters or a linkage distance threshold."
        )
        self.cluster_agg_criterion_combo.currentIndexChanged.connect(
            self._update_agg_controls
        )
        agg_form.addRow("Stop criterion:", self.cluster_agg_criterion_combo)

        self.cluster_agg_n_clusters = QtWidgets.QSpinBox()
        self.cluster_agg_n_clusters.setRange(2, 10000)
        self.cluster_agg_n_clusters.setValue(10)
        self.cluster_agg_n_clusters.setToolTip("Number of clusters to find.")
        agg_form.addRow("N clusters:", self.cluster_agg_n_clusters)

        self.cluster_agg_distance_threshold = QtWidgets.QDoubleSpinBox()
        self.cluster_agg_distance_threshold.setRange(0.0, 1_000_000.0)
        self.cluster_agg_distance_threshold.setSingleStep(0.05)
        self.cluster_agg_distance_threshold.setDecimals(4)
        self.cluster_agg_distance_threshold.setValue(1.0)
        self.cluster_agg_distance_threshold.setToolTip(
            "Stop merging when linkage distance exceeds this threshold."
        )
        agg_form.addRow("Distance threshold:", self.cluster_agg_distance_threshold)

        self.cluster_agg_homogeneous_widget = QtWidgets.QWidget()
        homogeneous_form = QtWidgets.QFormLayout()
        homogeneous_form.setContentsMargins(0, 0, 0, 0)
        self.cluster_agg_homogeneous_widget.setLayout(homogeneous_form)
        agg_form.addRow(self.cluster_agg_homogeneous_widget)

        self.cluster_agg_homogeneous_labels_combo = QtWidgets.QComboBox()
        self.cluster_agg_homogeneous_labels_combo.setToolTip(
            "Label source used to evaluate composition balance per cluster."
        )
        homogeneous_form.addRow(
            "Composition labels:", self.cluster_agg_homogeneous_labels_combo
        )

        self.cluster_agg_homogeneous_max_dist = QtWidgets.QDoubleSpinBox()
        self.cluster_agg_homogeneous_max_dist.setRange(0.0, 1_000_000.0)
        self.cluster_agg_homogeneous_max_dist.setDecimals(4)
        self.cluster_agg_homogeneous_max_dist.setSingleStep(0.05)
        self.cluster_agg_homogeneous_max_dist.setValue(0.0)
        self.cluster_agg_homogeneous_max_dist.setSpecialValueText("None")
        self.cluster_agg_homogeneous_max_dist.setToolTip(
            "Upper split-distance bound. 0 means no upper bound."
        )
        homogeneous_form.addRow(
            "Max split distance:", self.cluster_agg_homogeneous_max_dist
        )

        self.cluster_agg_homogeneous_min_dist = QtWidgets.QDoubleSpinBox()
        self.cluster_agg_homogeneous_min_dist.setRange(0.0, 1_000_000.0)
        self.cluster_agg_homogeneous_min_dist.setDecimals(4)
        self.cluster_agg_homogeneous_min_dist.setSingleStep(0.05)
        self.cluster_agg_homogeneous_min_dist.setValue(0.0)
        self.cluster_agg_homogeneous_min_dist.setSpecialValueText("None")
        self.cluster_agg_homogeneous_min_dist.setToolTip(
            "Lower split-distance bound. 0 means no lower bound."
        )
        homogeneous_form.addRow(
            "Min split distance:", self.cluster_agg_homogeneous_min_dist
        )

        self.cluster_agg_homogeneous_min_dist_diff = QtWidgets.QDoubleSpinBox()
        self.cluster_agg_homogeneous_min_dist_diff.setRange(0.0, 1_000_000.0)
        self.cluster_agg_homogeneous_min_dist_diff.setDecimals(4)
        self.cluster_agg_homogeneous_min_dist_diff.setSingleStep(0.05)
        self.cluster_agg_homogeneous_min_dist_diff.setValue(0.0)
        self.cluster_agg_homogeneous_min_dist_diff.setSpecialValueText("None")
        self.cluster_agg_homogeneous_min_dist_diff.setToolTip(
            "Merge nearby clusters when split distances differ less than this value."
        )
        homogeneous_form.addRow(
            "Merge distance diff:", self.cluster_agg_homogeneous_min_dist_diff
        )

        self.cluster_agg_linkage_combo = QtWidgets.QComboBox()
        self.cluster_agg_linkage_combo.addItems(
            ["ward", "complete", "average", "single"]
        )
        self.cluster_agg_linkage_combo.setToolTip(
            "Ward minimises within-cluster variance. complete/average/single use "
            "max/mean/min inter-cluster distances. "
            "Ward is replaced with 'average' when using a precomputed distance matrix."
        )
        agg_form.addRow("Linkage:", self.cluster_agg_linkage_combo)

        # --- K-Means settings ---
        self.cluster_kmeans_widget = QtWidgets.QWidget()
        kmeans_form = QtWidgets.QFormLayout()
        kmeans_form.setContentsMargins(0, 0, 0, 0)
        self.cluster_kmeans_widget.setLayout(kmeans_form)
        algo_layout.addWidget(self.cluster_kmeans_widget)

        self.cluster_kmeans_n_clusters = QtWidgets.QSpinBox()
        self.cluster_kmeans_n_clusters.setRange(2, 10000)
        self.cluster_kmeans_n_clusters.setValue(10)
        self.cluster_kmeans_n_clusters.setToolTip("Number of clusters.")
        kmeans_form.addRow("N clusters:", self.cluster_kmeans_n_clusters)

        self.cluster_kmeans_n_init = QtWidgets.QSpinBox()
        self.cluster_kmeans_n_init.setRange(1, 100)
        self.cluster_kmeans_n_init.setValue(10)
        self.cluster_kmeans_n_init.setToolTip(
            "Number of random initialisations to try."
        )
        kmeans_form.addRow("N init:", self.cluster_kmeans_n_init)

        self.cluster_kmeans_max_iter = QtWidgets.QSpinBox()
        self.cluster_kmeans_max_iter.setRange(10, 10000)
        self.cluster_kmeans_max_iter.setValue(300)
        self.cluster_kmeans_max_iter.setToolTip("Maximum iterations per run.")
        kmeans_form.addRow("Max iterations:", self.cluster_kmeans_max_iter)

        # Output group
        output_group = QtWidgets.QGroupBox("Output")
        output_form = QtWidgets.QFormLayout()
        output_form.setContentsMargins(8, 6, 8, 6)
        output_group.setLayout(output_form)
        self.tab7_layout.addWidget(output_group)

        self.cluster_result_label = QtWidgets.QLabel("N/A")
        self.cluster_result_label.setToolTip("Result of the last clustering run.")
        output_form.addRow("Result:", self.cluster_result_label)

        cluster_output_buttons = QtWidgets.QWidget()
        cluster_output_buttons_layout = QtWidgets.QHBoxLayout()
        cluster_output_buttons_layout.setContentsMargins(0, 0, 0, 0)
        cluster_output_buttons.setLayout(cluster_output_buttons_layout)

        self.cluster_apply_button = QtWidgets.QPushButton("Apply labels")
        self.cluster_apply_button.setToolTip(
            "Write cluster assignments as a metadata column and switch the "
            "label display to show cluster IDs."
        )
        self.cluster_apply_button.setEnabled(False)
        self.cluster_apply_button.clicked.connect(
            lambda _: self.label_combo_box.setCurrentText(CLUSTER_DATA_OPTION)
        )
        cluster_output_buttons_layout.addWidget(self.cluster_apply_button)

        self.cluster_export_button = QtWidgets.QPushButton("Export")
        self.cluster_export_button.setToolTip(
            "Save the current cluster assignments as a CSV file "
            "with 'id' and 'bigclust_cluster' columns."
        )
        self.cluster_export_button.setEnabled(False)
        self.cluster_export_button.clicked.connect(self._export_cluster_labels)
        cluster_output_buttons_layout.addWidget(self.cluster_export_button)

        output_form.addRow(cluster_output_buttons)

        # Manual refinement group (hidden until cluster labels are available)
        self.cluster_manual_group = QtWidgets.QGroupBox("Manual Refinement")
        manual_form = QtWidgets.QFormLayout()
        manual_form.setContentsMargins(8, 6, 8, 6)
        self.cluster_manual_group.setLayout(manual_form)
        self.tab7_layout.addWidget(self.cluster_manual_group)

        cluster_set_row = QtWidgets.QWidget()
        cluster_set_layout = QtWidgets.QHBoxLayout()
        cluster_set_layout.setContentsMargins(0, 0, 0, 0)
        cluster_set_row.setLayout(cluster_set_layout)

        self.cluster_manual_set_combo = QtWidgets.QComboBox()
        self.cluster_manual_set_combo.setToolTip(
            "Select target cluster ID for the current selection."
        )
        self.cluster_manual_set_combo.setEditable(True)
        self.cluster_manual_set_combo.setInsertPolicy(
            QtWidgets.QComboBox.InsertPolicy.NoInsert
        )
        combo_completer = self.cluster_manual_set_combo.completer()
        if combo_completer is not None:
            combo_completer.setCaseSensitivity(QtCore.Qt.CaseInsensitive)
        cluster_set_layout.addWidget(self.cluster_manual_set_combo)

        self.cluster_manual_set_button = QtWidgets.QPushButton("Set Cluster")
        self.cluster_manual_set_button.setToolTip(
            "Assign current selection to the selected cluster ID."
        )
        self.cluster_manual_set_button.clicked.connect(self._manual_refine_set_cluster)
        cluster_set_layout.addWidget(self.cluster_manual_set_button)

        manual_form.addRow("Target cluster:", cluster_set_row)

        self.cluster_manual_new_button = QtWidgets.QPushButton("New Cluster")
        self.cluster_manual_new_button.setToolTip(
            "Assign current selection to a newly created cluster ID."
        )
        self.cluster_manual_new_button.clicked.connect(self._manual_refine_new_cluster)
        manual_form.addRow(self.cluster_manual_new_button)

        self.cluster_manual_reset_button = QtWidgets.QPushButton("Reset")
        self.cluster_manual_reset_button.setToolTip(
            "Reset current selection to the original cluster assignment."
        )
        self.cluster_manual_reset_button.clicked.connect(self._manual_refine_reset)
        manual_form.addRow(self.cluster_manual_reset_button)

        self.tab7_layout.addStretch(1)

        # Sync initial visibility
        self._update_cluster_method_settings()
        self._update_hdbscan_noise_controls()
        self.update_cluster_options()
        self._refresh_manual_refinement_controls()

    # ------------------------------------------------------------------
    # Clusters tab helpers
    # ------------------------------------------------------------------

    def update_cluster_options(self):
        """Enable/disable the Clusters tab and populate the data combo box."""
        clusters_tab_index = self.tabs.indexOf(self.tab7)
        dists = getattr(self.figure, "dists", None)
        positions = getattr(self.figure, "positions", None)
        has_embedding = positions is not None and len(np.asarray(positions)) > 0

        if dists is None and not has_embedding:
            self.tabs.setTabEnabled(clusters_tab_index, False)
            self._refresh_manual_refinement_controls()
            return

        self.tabs.setTabEnabled(clusters_tab_index, True)

        current_text = self.cluster_data_combo_box.currentText()
        self.cluster_data_combo_box.blockSignals(True)
        self.cluster_data_combo_box.clear()
        if has_embedding:
            self.cluster_data_combo_box.addItem(CLUSTER_DATA_EMBEDDING_OPTION)
        if isinstance(dists, dict):
            for key in dists.keys():
                self.cluster_data_combo_box.addItem(key)
        self.cluster_data_combo_box.blockSignals(False)

        if current_text and self.cluster_data_combo_box.findText(current_text) >= 0:
            self.cluster_data_combo_box.setCurrentText(current_text)

        self._update_homogeneous_label_options()
        self._update_cluster_input_controls()
        self._refresh_manual_refinement_controls()

    def _update_homogeneous_label_options(self):
        """Populate homogeneous-composition label source choices."""
        if not hasattr(self, "cluster_agg_homogeneous_labels_combo"):
            return

        current = self.cluster_agg_homogeneous_labels_combo.currentText()
        self.cluster_agg_homogeneous_labels_combo.blockSignals(True)
        self.cluster_agg_homogeneous_labels_combo.clear()
        self.cluster_agg_homogeneous_labels_combo.addItem(
            CLUSTER_HOMOGENEOUS_LABEL_CURRENT
        )

        meta = getattr(self.figure, "metadata", None)
        if meta is not None:
            for col in sorted(meta.columns):
                if col.startswith("_"):
                    continue
                self.cluster_agg_homogeneous_labels_combo.addItem(col)

        self.cluster_agg_homogeneous_labels_combo.blockSignals(False)

        if current and self.cluster_agg_homogeneous_labels_combo.findText(current) >= 0:
            self.cluster_agg_homogeneous_labels_combo.setCurrentText(current)
        elif self.cluster_agg_homogeneous_labels_combo.findText("dataset") >= 0:
            self.cluster_agg_homogeneous_labels_combo.setCurrentText("dataset")

    def _selected_homogeneous_labels(self):
        """Return labels used for homogeneous-composition clustering."""
        key = self.cluster_agg_homogeneous_labels_combo.currentText()
        if (not key) or (key == CLUSTER_HOMOGENEOUS_LABEL_CURRENT):
            return np.asarray(self.figure.labels)

        meta = getattr(self.figure, "metadata", None)
        if meta is None or key not in meta.columns:
            return None

        return meta[key].astype(str).fillna("").to_numpy()

    def _selected_clustering_data(self):
        """Return the currently selected distance/feature matrix for clustering."""
        key = self.cluster_data_combo_box.currentText()
        if key == CLUSTER_DATA_EMBEDDING_OPTION:
            return getattr(self.figure, "positions", None)

        dists = getattr(self.figure, "dists", None)
        if dists is None:
            return None
        if isinstance(dists, dict):
            if not key:
                return None
            return dists.get(key)
        return dists

    def _update_cluster_input_controls(self):
        """Adjust metric availability based on the selected input type."""
        arr = self._selected_clustering_data()
        is_precomputed = arr is not None and is_precomputed_distance_matrix(arr)
        self.cluster_metric_combo_box.setEnabled(not is_precomputed)

    def _update_cluster_method_settings(self):
        """Show/hide method-specific settings widgets."""
        method = self.cluster_method_combo_box.currentText()
        self.cluster_hdbscan_widget.setVisible(method == "HDBSCAN")
        self.cluster_agg_widget.setVisible(method == "Agglomerative")
        self.cluster_kmeans_widget.setVisible(method == "K-Means")
        self._update_agg_controls()

    def _update_agg_controls(self):
        """Enable Agglomerative fields for the selected stop criterion."""
        if not hasattr(self, "cluster_agg_criterion_combo"):
            return

        criterion = self.cluster_agg_criterion_combo.currentText()
        use_threshold = criterion == "Distance threshold"
        use_homogeneous = criterion == "Homogeneous composition"

        show_n_clusters = not use_threshold and not use_homogeneous
        show_distance_threshold = use_threshold

        self.cluster_agg_n_clusters.setVisible(show_n_clusters)
        self.cluster_agg_distance_threshold.setVisible(show_distance_threshold)

        agg_form = self.cluster_agg_widget.layout()
        if isinstance(agg_form, QtWidgets.QFormLayout):
            n_label = agg_form.labelForField(self.cluster_agg_n_clusters)
            if n_label is not None:
                n_label.setVisible(show_n_clusters)

            d_label = agg_form.labelForField(self.cluster_agg_distance_threshold)
            if d_label is not None:
                d_label.setVisible(show_distance_threshold)

        self.cluster_agg_homogeneous_widget.setVisible(use_homogeneous)

    def _update_hdbscan_noise_controls(self):
        """Enable optional HDBSCAN noise-threshold controls."""
        if not hasattr(self, "cluster_hdbscan_label_noise_check"):
            return

        relabel_noise = self.cluster_hdbscan_label_noise_check.isChecked()
        use_threshold = (
            relabel_noise and self.cluster_hdbscan_noise_threshold_check.isChecked()
        )

        self.cluster_hdbscan_noise_threshold_check.setEnabled(relabel_noise)
        self.cluster_hdbscan_noise_threshold_row.setEnabled(relabel_noise)
        self.cluster_hdbscan_noise_threshold.setEnabled(use_threshold)

    def _run_clustering(self):
        """Run clustering with the current settings and colour points by cluster."""
        from ...clusters import run_clustering

        arr = self._selected_clustering_data()
        if arr is None:
            self.figure.show_message(
                "No data available for clustering", color="red", duration=2
            )
            return

        arr_np = arr.values if hasattr(arr, "values") else np.asarray(arr)
        is_precomputed = is_precomputed_distance_matrix(arr_np)
        metric = self.cluster_metric_combo_box.currentText()
        method = self.cluster_method_combo_box.currentText()

        if (
            method == "Agglomerative"
            and self.cluster_agg_linkage_combo.currentText() == "ward"
            and (is_precomputed or metric != "euclidean")
        ):
            if self.figure is not None:
                self.figure.show_message(
                    "Ward linkage should not be used with non-euclidean distances.\n"
                    "Please use a different linkage method (e.g. 'average') or provide\n"
                    "feature vectors + Euclidean metric instead of a distance matrix.",
                    duration=5,
                    color="orange",
                )

        try:
            labels = run_clustering(
                arr_np,
                method=method,
                is_precomputed=is_precomputed,
                metric=metric,
                hdbscan_min_cluster_size=self.cluster_hdbscan_min_cluster_size.value(),
                hdbscan_min_samples=(self.cluster_hdbscan_min_samples.value() or None),
                hdbscan_cluster_selection_epsilon=self.cluster_hdbscan_epsilon.value(),
                hdbscan_cluster_selection_method=self.cluster_hdbscan_method_combo.currentText(),
                hbdscan_label_noise=self.cluster_hdbscan_label_noise_check.isChecked(),
                hbdscan_label_noise_threshold=(
                    self.cluster_hdbscan_noise_threshold.value()
                    if (
                        self.cluster_hdbscan_label_noise_check.isChecked()
                        and self.cluster_hdbscan_noise_threshold_check.isChecked()
                    )
                    else None
                ),
                agg_stop_criterion={
                    "N clusters": "n_clusters",
                    "Distance threshold": "distance_threshold",
                    "Homogeneous composition": "homogeneous_composition",
                }[self.cluster_agg_criterion_combo.currentText()],
                agg_n_clusters=(
                    self.cluster_agg_n_clusters.value()
                    if self.cluster_agg_criterion_combo.currentText() == "N clusters"
                    else None
                ),
                agg_distance_threshold=(
                    self.cluster_agg_distance_threshold.value()
                    if self.cluster_agg_criterion_combo.currentText()
                    == "Distance threshold"
                    else None
                ),
                agg_linkage=self.cluster_agg_linkage_combo.currentText(),
                agg_homogeneous_labels=(
                    self._selected_homogeneous_labels()
                    if self.cluster_agg_criterion_combo.currentText()
                    == "Homogeneous composition"
                    else None
                ),
                agg_homogeneous_max_dist=(
                    self.cluster_agg_homogeneous_max_dist.value() or None
                    if self.cluster_agg_criterion_combo.currentText()
                    == "Homogeneous composition"
                    else None
                ),
                agg_homogeneous_min_dist=(
                    self.cluster_agg_homogeneous_min_dist.value() or None
                    if self.cluster_agg_criterion_combo.currentText()
                    == "Homogeneous composition"
                    else None
                ),
                agg_homogeneous_min_dist_diff=(
                    self.cluster_agg_homogeneous_min_dist_diff.value() or None
                    if self.cluster_agg_criterion_combo.currentText()
                    == "Homogeneous composition"
                    else None
                ),
                kmeans_n_clusters=self.cluster_kmeans_n_clusters.value(),
                kmeans_n_init=self.cluster_kmeans_n_init.value(),
                kmeans_max_iter=self.cluster_kmeans_max_iter.value(),
            )
        except Exception as e:
            logger.error(f"Clustering failed: {e}")
            self.figure.show_message(f"Clustering error: {e}", color="red", duration=4)
            return

        self._cluster_labels = labels
        self._cluster_labels_original = labels.copy()

        # This function takes care of writing cluster labels to metadata, updating the label/color combo boxes, and refreshing the display
        self._apply_cluster_labels()

    def _clear_cluster(self):
        """Reset viewer to the defaults stored in figure metadata."""
        self._cluster_labels = None
        self._cluster_labels_original = None
        self._cluster_colors = None
        self.cluster_result_label.setText("N/A")
        self.cluster_apply_button.setEnabled(False)
        self.cluster_export_button.setEnabled(False)
        self.color_combo_box.setCurrentText(self._color_col_before_clustering)
        if getattr(self, "_label_before_clustering", None):
            self.label_combo_box.setCurrentText(self._label_before_clustering)
        self._refresh_manual_refinement_controls()
        self.update_label_combo_boxes()

    def _apply_cluster_labels(self):
        """Write cluster assignments to metadata and switch the label display."""
        if not hasattr(self, "_cluster_labels") or self._cluster_labels is None:
            return

        # (Re-)calculate colors
        self._cluster_colors = labels_to_colors(
            self._cluster_labels, palette="vispy:husl"
        )
        self._cluster_colors[self._cluster_labels < 0] = (
            0.5,
            0.5,
            0.5,
            1.0,
        )  # grey for noise/unassigned

        # Write new cluster labels to metadata
        if self.figure.metadata is not None:
            self.figure.metadata[CLUSTER_DATA_COLUMN] = self._cluster_labels.copy()

        # Save the current label and color mode so we can restore it when clearing
        self._label_before_clustering = self.label_combo_box.currentText()
        self._color_col_before_clustering = self.color_combo_box.currentText()

        # Update the cluster summary label
        self._update_cluster_result_label(self._cluster_labels)

        # Make sure the cluster buttons are enabled and the manual refinement controls are visible
        self.cluster_apply_button.setEnabled(True)
        self.cluster_export_button.setEnabled(True)
        self._refresh_manual_refinement_controls()

        # Apply cluster colors
        self.label_combo_box.blockSignals(True)
        self.update_label_combo_boxes()  # this update both label and color combo boxes
        self.label_combo_box.blockSignals(False)

        if self.color_combo_box.currentText() != CLUSTER_DATA_OPTION:
            self.color_combo_box.setCurrentText(
                CLUSTER_DATA_OPTION
            )  # this triggers a refresh
        else:
            self.set_colors()  # just refresh colors without changing the combo box text

        # For the labels: we don't automatically enforce setting them but if they are set to the cluster column,
        # we need to trigger an update
        if self.label_combo_box.currentText() == CLUSTER_DATA_OPTION:
            self.set_labels()

        if self.figure.show_label_lines:
            self.figure.make_label_lines()

        self._refresh_manual_refinement_controls()

    def _load_cluster_labels(self):
        """Open a load dialog and import cluster assignments from a CSV file."""
        import pandas as pd

        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Load cluster labels",
            "",
            "CSV files (*.csv);;All files (*)",
        )
        if not path:
            return

        try:
            df = pd.read_csv(path)
        except Exception as e:
            self.figure.show_message(
                f"Failed to read CSV: {e}", color="red", duration=4
            )
            return

        if "id" not in df.columns:
            self.figure.show_message(
                "CSV must contain 'id' column.", color="red", duration=4
            )
            return

        if CLUSTER_DATA_COLUMN in df.columns:
            cluster_col = CLUSTER_DATA_COLUMN
        elif "cluster" in df.columns:
            cluster_col = "cluster"
        else:
            self.figure.show_message(
                f"CSV must contain '{CLUSTER_DATA_COLUMN}' or 'cluster' column for cluster labels.",
                color="red",
                duration=4,
            )

        df[cluster_col] = df[cluster_col].fillna(-1).astype(self.figure.ids.dtype)

        df = df.set_index("id")
        labels_series = df[cluster_col].reindex(self.figure.ids)
        if labels_series.isna().any():
            self.figure.show_message(
                "Some IDs in the CSV do not match the current data. "
                "Missing entries will be marked as noise (-1).",
                color="orange",
                duration=4,
            )
        labels = labels_series.fillna(-1).astype(int).to_numpy()

        self._cluster_labels = labels
        self._cluster_labels_original = labels.copy()

        # This function takes care of writing cluster labels to metadata, updating the label/color combo boxes, and refreshing the display
        self._apply_cluster_labels()

    def _export_cluster_labels(self):
        """Open a save dialog and write cluster assignments to a CSV file."""
        if not hasattr(self, "_cluster_labels") or self._cluster_labels is None:
            return

        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Export cluster labels",
            "clusters.csv",
            "CSV files (*.csv);;All files (*)",
        )
        if not path:
            return

        import pandas as pd

        labels = self._cluster_labels
        ids = self.figure.ids if hasattr(self.figure, "ids") else np.arange(len(labels))

        df = pd.DataFrame({"id": ids, CLUSTER_DATA_COLUMN: labels})
        df.to_csv(path, index=False)

    def _update_cluster_result_label(self, labels):
        """Update clustering summary label from integer cluster assignments."""
        unique_clusters = (
            np.unique(labels[labels >= 0])
            if (labels >= 0).any()
            else np.array([], dtype=int)
        )
        n_clusters = len(unique_clusters)
        n_noise = int((labels == -1).sum())
        if n_noise > 0:
            self.cluster_result_label.setText(
                f"{n_clusters} clusters, {n_noise} noise points"
            )
        else:
            self.cluster_result_label.setText(f"{n_clusters} clusters")

    def _parse_cluster_series(self, values):
        """Parse a metadata cluster column into integer cluster IDs."""
        parsed = np.empty(len(values), dtype=int)
        for i, value in enumerate(values):
            if pd.isna(value):
                return None

            if isinstance(value, (int, np.integer)):
                parsed[i] = int(value)
                continue

            text = str(value).strip().lower()
            if text == "noise":
                parsed[i] = -1
                continue

            match = re.fullmatch(r"cluster_(-?\d+)", text)
            if match:
                parsed[i] = int(match.group(1))
                continue

            if re.fullmatch(r"-?\d+", text):
                parsed[i] = int(text)
                continue

            return None

        return parsed

    def _ensure_cluster_labels_loaded(self):
        """Ensure editable cluster labels exist, loading from metadata when possible."""
        labels = getattr(self, "_cluster_labels", None)
        if labels is not None:
            return np.asarray(labels, dtype=int)

        meta = getattr(self.figure, "metadata", None)
        if meta is None or CLUSTER_DATA_COLUMN not in meta.columns:
            return None

        parsed = self._parse_cluster_series(meta[CLUSTER_DATA_COLUMN].values)
        if parsed is None:
            return None

        self._cluster_labels = parsed.copy()
        self._cluster_labels_original = parsed.copy()
        return self._cluster_labels

    def _manual_cluster_target_id(self):
        """Return selected target cluster ID from manual refinement combo."""
        text = self.cluster_manual_set_combo.currentText().strip().lower()
        if text in ("noise", "noise (-1)"):
            return -1

        match = re.fullmatch(r"cluster_(-?\d+)", text)
        if match:
            return int(match.group(1))

        if re.fullmatch(r"-?\d+", text):
            return int(text)

        value = self.cluster_manual_set_combo.currentData()
        if value is not None:
            return int(value)

        return None

    def _refresh_manual_refinement_controls(self):
        """Show/hide and repopulate manual refinement controls."""
        if not hasattr(self, "cluster_manual_group"):
            return

        labels = self._ensure_cluster_labels_loaded()
        has_clusters = labels is not None and len(labels) > 0
        self.cluster_manual_group.setVisible(has_clusters)
        if not has_clusters:
            return

        current_target = self._manual_cluster_target_id()
        unique_labels = np.unique(labels)
        non_noise = np.sort(unique_labels[unique_labels >= 0])

        self.cluster_manual_set_combo.blockSignals(True)
        self.cluster_manual_set_combo.clear()
        self.cluster_manual_set_combo.addItem("noise (-1)", -1)
        for cluster_id in non_noise:
            self.cluster_manual_set_combo.addItem(str(int(cluster_id)), int(cluster_id))

        if current_target is None:
            self.cluster_manual_set_combo.setCurrentIndex(0)
        else:
            idx = self.cluster_manual_set_combo.findData(int(current_target))
            if idx >= 0:
                self.cluster_manual_set_combo.setCurrentIndex(idx)
            else:
                self.cluster_manual_set_combo.setEditText(str(int(current_target)))
        self.cluster_manual_set_combo.blockSignals(False)

    @requires_selection
    def _manual_refine_set_cluster(self):
        """Set current selection to an existing cluster ID."""
        labels = self._ensure_cluster_labels_loaded()
        if labels is None:
            self.figure.show_message("No clusters loaded", color="red", duration=2)
            return

        cluster_id = self._manual_cluster_target_id()
        if cluster_id is None:
            self.figure.show_message("Invalid cluster ID", color="red", duration=2)
            return

        indices = self.selected_indices
        if indices.size == 0:
            self.figure.show_message("No points selected", color="red", duration=2)
            return

        labels = labels.copy()
        labels[indices] = int(cluster_id)
        self._cluster_labels = labels
        self._apply_cluster_labels()
        self._refresh_manual_refinement_controls()

    @requires_selection
    def _manual_refine_new_cluster(self):
        """Assign current selection to a new cluster ID."""
        labels = self._ensure_cluster_labels_loaded()
        if labels is None:
            self.figure.show_message("No clusters loaded", color="red", duration=2)
            return

        existing = labels[labels >= 0]
        next_cluster_id = int(existing.max() + 1) if existing.size else 0

        indices = self.selected_indices
        if indices.size == 0:
            self.figure.show_message("No points selected", color="red", duration=2)
            return

        labels = labels.copy()
        labels[indices] = next_cluster_id
        self._cluster_labels = labels
        self._apply_cluster_labels()
        self._refresh_manual_refinement_controls()

    @requires_selection
    def _manual_refine_reset(self):
        """Reset current selection to original automatic cluster assignments."""
        labels = self._ensure_cluster_labels_loaded()
        original = getattr(self, "_cluster_labels_original", None)
        if labels is None or original is None:
            self.figure.show_message("No clusters loaded", color="red", duration=2)
            return

        indices = self.selected_indices
        if indices.size == 0:
            self.figure.show_message("No points selected", color="red", duration=2)
            return

        labels = labels.copy()
        labels[indices] = np.asarray(original, dtype=int)[indices]
        self._cluster_labels = labels
        self._apply_cluster_labels()
        self._refresh_manual_refinement_controls()

    def add_tooltip(self, widget, text, anchor="group_label"):
        """Add a reusable round help icon tooltip to a widget."""
        if not hasattr(self, "_tooltip_anchors"):
            self._tooltip_anchors = {}

        button = QtWidgets.QToolButton(widget)
        button.setText("?")
        button.setAutoRaise(True)
        button.setFixedSize(14, 14)
        button.setStyleSheet(
            "QToolButton {"
            "border: 1px solid palette(mid);"
            "border-radius: 7px;"
            "font-weight: bold;"
            "padding: 0px;"
            "}"
        )
        button.setToolTip(text)

        self._tooltip_anchors.setdefault(widget, []).append((button, anchor))
        widget.installEventFilter(self)
        QtCore.QTimer.singleShot(0, lambda: self._position_tooltips(widget))
        return button

    def _position_tooltips(self, widget):
        """Position one or more tooltip icons for a widget."""
        items = getattr(self, "_tooltip_anchors", {}).get(widget, [])
        if not items:
            return

        for i, (button, anchor) in enumerate(items):
            if anchor == "group_label" and isinstance(widget, QtWidgets.QGroupBox):
                opt = QtWidgets.QStyleOptionGroupBox()
                widget.initStyleOption(opt)
                label_rect = widget.style().subControlRect(
                    QtWidgets.QStyle.CC_GroupBox,
                    opt,
                    QtWidgets.QStyle.SC_GroupBoxLabel,
                    widget,
                )
                if label_rect.isValid():
                    x = label_rect.right() + 4 + i * (button.width() + 3)
                    y = label_rect.center().y() - (button.height() // 2)
                else:
                    x = widget.width() - button.width() - 6 - i * (button.width() + 3)
                    y = 6
            else:
                x = widget.width() - button.width() - 6 - i * (button.width() + 3)
                y = 6

            max_x = max(0, widget.width() - button.width() - 6)
            button.move(max(0, min(x, max_x)), max(0, y))
            button.raise_()

    def eventFilter(self, obj, event):
        """Keep custom tooltip buttons aligned when host widgets change."""
        if obj in getattr(self, "_tooltip_anchors", {}) and event.type() in (
            QtCore.QEvent.Resize,
            QtCore.QEvent.Show,
            QtCore.QEvent.LayoutRequest,
        ):
            self._position_tooltips(obj)
        return super().eventFilter(obj, event)

    def add_split(self, layout):
        """Add horizontal divider."""
        # layout.addSpacing(5)
        layout.addWidget(QHLine())
        # layout.addSpacing(5)

    def update_controls(self):
        """Update the controls based on the current figure state."""
        # self.update_ann_combo_box()
        self.update_umap_options()
        self.update_fidelity_options()
        self.update_cluster_options()
        self.update_label_combo_boxes()
        self.update_searchbar_completer()
        self.update_distance_edges_controls()
        self.update_selection_restrict_options()

    def _toggle_selection_restrict_panel(self, expanded):
        """Expand/collapse dataset restriction controls."""
        self.selection_restrict_panel.setVisible(expanded)
        self.selection_restrict_toggle.setArrowType(
            QtCore.Qt.DownArrow if expanded else QtCore.Qt.RightArrow
        )

    def _available_datasets(self):
        """Return unique dataset names from the loaded figure data."""
        datasets = getattr(self.figure, "datasets", None)
        if datasets is not None:
            names = sorted(
                {
                    str(ds).strip()
                    for ds in datasets
                    if (not pd.isna(ds)) and str(ds).strip()
                }
            )
            if names:
                return names

        meta = getattr(self.figure, "metadata", None)
        if meta is None or "dataset" not in meta.columns:
            return []
        return sorted(
            {
                str(ds).strip()
                for ds in meta["dataset"].dropna().tolist()
                if str(ds).strip()
            }
        )

    def update_selection_restrict_options(self):
        """Refresh dataset restriction checkboxes from currently loaded data."""
        if not hasattr(self, "selection_restrict_panel"):
            return

        datasets = self._available_datasets()

        current_restriction = getattr(self.figure, "_restrict_selection", None)
        if current_restriction is None:
            checked_names = {
                name
                for name, check in self.selection_restrict_checks.items()
                if check.isChecked()
            }
        else:
            checked_names = {str(name).strip() for name in current_restriction}

        layout = self.selection_restrict_panel.layout()
        while layout.count():
            item = layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

        self.selection_restrict_checks = {}

        if not datasets:
            self.selection_restrict_empty_label = QtWidgets.QLabel(
                "No datasets available"
            )
            self.selection_restrict_empty_label.setStyleSheet("color: palette(mid);")
            layout.addWidget(self.selection_restrict_empty_label)
            if hasattr(self.figure, "_restrict_selection"):
                delattr(self.figure, "_restrict_selection")
            return

        for dataset_name in datasets:
            check = QtWidgets.QCheckBox(dataset_name)
            check.setChecked(dataset_name in checked_names or not checked_names)
            check.stateChanged.connect(self.set_selection_restrict)
            layout.addWidget(check)
            self.selection_restrict_checks[dataset_name] = check

        self.set_selection_restrict()

    def update_label_combo_boxes(self):
        """Update the items in the label and color by combo boxes."""
        # First, collect the items we want to have:
        items = {"Default"}
        if self.figure.metadata is not None:
            items.update(
                col for col in self.figure.metadata.columns if not col.startswith("_")
            )
        if getattr(self, "_cluster_labels", None) is not None:
            items.add(CLUSTER_DATA_OPTION)

        # Now edit the combo boxes to match these items, trying to preserve the current selection if possible
        for combo_box in (self.label_combo_box, self.color_combo_box):
            # First clear all items that no longer exist
            for i in reversed(range(combo_box.count())):
                text = combo_box.itemText(i)
                if text in ("Default", CLUSTER_DATA_OPTION):
                    continue
                if (
                    self.figure.metadata is None
                    or text not in self.figure.metadata.columns
                ):
                    combo_box.removeItem(i)

            # Now add missing items
            for item in sorted(items):
                if combo_box.findText(item) < 0:
                    combo_box.addItem(item)

    def update_distance_edges_controls(self):
        """Update the distance edges controls."""
        dists = getattr(self.figure, "dists", None)
        self.show_distance_edges_check.setEnabled(False)
        self.show_distance_edges_check.setEnabled(False)
        self.distance_edges_slider.setEnabled(False)

        if dists is None:
            return
        elif (
            isinstance(dists, (np.ndarray, pd.DataFrame))
            and dists.shape[0] != dists.shape[1]
        ):
            return
        elif (
            isinstance(dists, dict)
            and ("distances" not in dists)
            or (dists["distances"].shape[0] != dists["distances"].shape[1])
        ):
            return

        self.show_distance_edges_check.setEnabled(True)
        self.distance_edges_slider.setEnabled(True)

    def update_umap_options(self):
        """Update the items in the UMAP distance combo box."""
        self.umap_dist_combo_box.clear()

        embeddings_tab_index = self.tabs.indexOf(self.tab5)

        if getattr(self.figure, "dists", None) is None:
            self.tabs.setTabEnabled(embeddings_tab_index, False)
            return
        self.tabs.setTabEnabled(embeddings_tab_index, True)

        # Add any distances we have
        if isinstance(self.figure.dists, dict):
            for key in self.figure.dists.keys():
                self.umap_dist_combo_box.addItem(key)

        self._update_embedding_input_controls()

    def update_fidelity_options(self):
        """Update fidelity controls based on available distances/features."""
        fidelity_tab_index = self.tabs.indexOf(self.tab6)
        dists = getattr(self.figure, "dists", None)

        if dists is None:
            self.tabs.setTabEnabled(fidelity_tab_index, False)
            for box in (
                self.fidelity_distance_combo_box,
                self.fidelity_knn_distance_combo_box,
            ):
                box.blockSignals(True)
                box.clear()
                box.blockSignals(False)
            return

        self.tabs.setTabEnabled(fidelity_tab_index, True)

        options = []
        has_precomputed = False
        has_features = False

        if isinstance(dists, dict):
            has_precomputed = "distances" in dists
            has_features = "features" in dists
        elif isinstance(dists, (np.ndarray, pd.DataFrame)):
            has_precomputed = dists.shape[0] == dists.shape[1]
            has_features = not has_precomputed

        if has_precomputed:
            options.append("precomputed")
        if has_features:
            options.extend(
                ["euclidean", "cosine", "manhattan", "correlation", "chebyshev"]
            )

        for box in (
            self.fidelity_distance_combo_box,
            self.fidelity_knn_distance_combo_box,
        ):
            current_text = box.currentText()
            box.blockSignals(True)
            box.clear()
            box.addItems(options)
            box.blockSignals(False)

            if current_text in options:
                box.setCurrentText(current_text)
            elif options:
                box.setCurrentIndex(0)

            box.setEnabled(len(options) > 1)

    def _selected_embedding_data(self):
        """Return the currently selected distance/feature matrix."""
        dists = getattr(self.figure, "dists", None)
        if dists is None:
            return None

        if isinstance(dists, dict):
            key = self.umap_dist_combo_box.currentText()
            if not key:
                return None
            return dists.get(key)

        return dists

    def _embedding_feature_partition_names(self, arr):
        """Return ordered top-level partition names for MultiIndex feature columns."""
        if not isinstance(arr, pd.DataFrame):
            return []
        if not isinstance(arr.columns, pd.MultiIndex):
            return []

        names = []
        seen = set()
        for value in arr.columns.get_level_values(0):
            label = str(value)
            if label in seen:
                continue
            seen.add(label)
            names.append(label)
        return names

    def _apply_embedding_feature_partition_filter(self, arr):
        """Filter feature columns by selected top-level partition checkboxes."""
        if not isinstance(arr, pd.DataFrame):
            return arr
        if not isinstance(arr.columns, pd.MultiIndex):
            return arr
        if not hasattr(self, "_embedding_partition_checks"):
            return arr

        selected = {
            name
            for name, checkbox in self._embedding_partition_checks.items()
            if checkbox.isChecked()
        }
        if not selected:
            return arr.iloc[:, 0:0].copy()

        top_level = arr.columns.get_level_values(0).astype(str)
        mask = np.asarray([name in selected for name in top_level], dtype=bool)
        if not mask.any():
            return arr.iloc[:, 0:0].copy()
        return arr.loc[:, mask].copy()

    def _update_embedding_feature_partition_controls(self, arr, is_feature_input):
        """Show/update partition checkboxes for MultiIndex feature inputs."""
        if not hasattr(self, "umap_feature_partition_widget"):
            return

        names = (
            self._embedding_feature_partition_names(arr)
            if is_feature_input
            else []
        )
        has_partitions = len(names) > 0

        if hasattr(self, "umap_feature_subset_group"):
            self.umap_feature_subset_group.setVisible(has_partitions)
            self.umap_feature_subset_group.setEnabled(is_feature_input)

        old_checks = getattr(self, "_embedding_partition_checks", {})
        old_state = {name: check.isChecked() for name, check in old_checks.items()}

        layout = self.umap_feature_partition_widget.layout()
        while layout.count():
            item = layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

        self._embedding_partition_checks = {}
        for name in names:
            check = QtWidgets.QCheckBox(name)
            check.setChecked(old_state.get(name, True))
            check.stateChanged.connect(self.calculate_embeddings_maybe)
            layout.addWidget(check)
            self._embedding_partition_checks[name] = check

    def _update_embedding_input_controls(self):
        """Toggle embedding controls based on selected input type."""
        if not hasattr(self, "pca_check"):
            return

        arr = self._selected_embedding_data()
        is_feature_input = (arr is not None) and (
            not is_precomputed_distance_matrix(arr)
        )
        self._update_embedding_feature_partition_controls(arr, is_feature_input)
        arr_filtered = self._apply_embedding_feature_partition_filter(arr)

        has_selected_features = True
        if (
            is_feature_input
            and arr_filtered is not None
            and hasattr(arr_filtered, "shape")
            and len(arr_filtered.shape) > 1
            and arr_filtered.shape[1] == 0
        ):
            has_selected_features = False

        if hasattr(self, "feature_options_group"):
            self.feature_options_group.setEnabled(is_feature_input)

        if hasattr(self, "umap_button"):
            self.umap_button.setEnabled((arr is not None) and (not is_feature_input or has_selected_features))

        # PCA pre-reduction is only meaningful for feature vectors.
        if not is_feature_input:
            self.pca_check.setChecked(False)
        self.pca_check.setEnabled(is_feature_input and has_selected_features)
        self.pca_n_components_slider.setEnabled(
            is_feature_input and has_selected_features and self.pca_check.isChecked()
        )

        self._update_densmap_controls()

    def _update_densmap_controls(self):
        """Show the dens_lambda control only when DensMAP is enabled."""
        show = self.umap_densmap_check.isChecked()
        self.umap_dens_lambda_label.setVisible(show)
        self.umap_dens_lambda_spinbox.setVisible(show)
        if show:
            self.umap_dens_lambda_label.setEnabled(True)
            self.umap_dens_lambda_spinbox.setEnabled(True)

    def set_add_group(self):
        """Set whether to add neurons as group when selected."""
        self.figure._add_as_group = self.add_group_check.isChecked()

    def set_selection_restrict(self):
        """Set optional dataset restriction for point selection."""
        if not hasattr(self, "selection_restrict_checks"):
            return

        selected_datasets = [
            name
            for name, check in self.selection_restrict_checks.items()
            if check.isChecked()
        ]
        self.figure._restrict_selection = selected_datasets

    def set_ngl_cache(self):
        """Set whether the ngl viewer should cache neurons."""
        if hasattr(self.figure, "ngl_viewer"):
            self.figure.ngl_viewer.use_cache = self.ngl_cache_neurons.isChecked()

            if not self.ngl_cache_neurons.isChecked():
                self.figure.ngl_viewer.clear_cache()

    def set_ngl_debug(self):
        """Set debug mode for ngl viewer."""
        if hasattr(self.figure, "ngl_viewer"):
            self.figure.ngl_viewer.debug = self.ngl_debug_mode.isChecked()

    def set_dclick_deselect(self):
        """Set whether to deselect on double-click."""
        self.figure.deselect_on_dclick = self.dclick_deselect.isChecked()

    def set_empty_deselect(self):
        """Set whether to deselect on double-click."""
        self.figure.deselect_on_empty = self.empty_deselect.isChecked()

    def set_label_counts(self):
        """Set whether to add counts to the labels."""
        self.set_labels()  # Update the labels

    def find_next(self):
        """Find next occurrence."""
        text = self.searchbar.text()
        if text:
            regex = False
            if text.startswith("/"):
                regex = True
                text = text[1:]

            if (
                not hasattr(self, "_label_search")
                or self._label_search.query != text
                or self._label_search.regex != regex
            ):
                self._label_search = self.figure.find_label(text, regex=regex)

            # LabelSearch can be `None` if no match found
            if self._label_search:
                self._label_search.next()

    def find_previous(self):
        """Find previous occurrence."""
        text = self.searchbar.text()
        if text:
            regex = False
            if text.startswith("/"):
                regex = True
                text = text[1:]

            if (
                not hasattr(self, "_label_search")
                or self._label_search.query != text
                or self._label_search.regex != regex
            ):
                self._label_search = self.figure.find_label(text, regex=regex)

            # LabelSearch can be `None` if no match found
            if self._label_search:
                self._label_search.prev()

    def find_select(self):
        """Find and select all matches."""
        mods = QtWidgets.QApplication.keyboardModifiers()
        text = self.searchbar.text()
        if text:
            regex = False
            if text.startswith("/"):
                regex = True
                text = text[1:]

            if (
                not hasattr(self, "_label_search")
                or self._label_search.query != text
                or self._label_search.regex != regex
            ):
                self._label_search = self.figure.find_label(text, regex=regex)

            # LabelSearch can be `None` if no match found
            if self._label_search:
                if mods & QtCore.Qt.ShiftModifier:
                    self._label_search.select_all(add=True)
                else:
                    self._label_search.select_all(add=False)

    def selected_to_clipboard(self, dataset=None):
        """Copy selected items to clipboard."""
        if self.figure.selected is not None:
            indices = self.selected_indices

            if isinstance(dataset, str):
                indices = [i for i in indices if self.figure._markers[i] == dataset]
            elif isinstance(dataset, (list, set, tuple)):
                indices = [i for i in indices if self.figure._markers[i] in dataset]

            ids = self.figure._ids[indices]
            pyperclip.copy(",".join(np.array(ids).astype(str)))

    def set_colors(self, sync_to_viewer=True):
        """Set the color mode."""
        color_col = self.color_combo_box.currentText()

        if not color_col:
            return

        if color_col == "Default":
            color_col = self.figure.default_color_col

        if color_col == CLUSTER_DATA_OPTION:
            colors = self._cluster_colors
        elif is_color_column(self.meta_data[color_col]):
            colors = self.meta_data[color_col].values
        else:
            colors = labels_to_colors(
                self.meta_data[color_col].values,
                palette=self.palette_combo_box.currentText(),
            )

        self.figure.set_colors(colors, sync_to_viewer=sync_to_viewer)

    def set_label_outlines(self):
        """Draw polygons around neurons with the same label."""
        self.figure.show_label_lines = self.label_outlines_check.isChecked()

    def set_fidelity_mode(self):
        """Set the fidelity mode."""
        if not self.fidelity_point_size_check.isChecked():
            self.figure.fidelity_mode = False
        else:
            self.figure.fidelity_mode = {
                "mode": "point_size",
                "k": self.fidelity_k_slider.value(),
                "distance": self.fidelity_distance_combo_box.currentText(),
                "rank": self.fidelity_use_rank_check.isChecked(),
            }

    def set_knn_edges(self):
        """Set the KNN line mode."""
        mode = self.fidelity_knn_combo_box.currentText()
        if mode == "Off":
            self.figure.show_knn_edges = False
        else:
            self.figure.show_knn_edges = {
                "mode": "selected" if mode == "Selected only" else "all",
                "k": self.fidelity_knn_k_slider.value(),
                "metric": self.fidelity_knn_distance_combo_box.currentText(),
                "distance": self.fidelity_knn_distance_combo_box.currentText(),
            }

    def set_labels(self):
        """Set the leaf labels."""
        label = self.label_combo_box.currentText()

        if not label:
            return

        if label == "Default":
            label = self.figure.default_label_col

        # Nothing to do here
        if self._current_leaf_labels != label:
            self._last_leaf_labels, self._current_leaf_labels = (
                self._current_leaf_labels,
                label,
            )

        if label == CLUSTER_DATA_OPTION:
            labels = (
                "cluster_"
                + self.meta_data[CLUSTER_DATA_COLUMN].astype(str).fillna("-1").values
            )
            labels[labels == "cluster_-1"] = "noise"
        else:
            labels = self.meta_data[label].astype(str).fillna("").values

        # For labels that were set manually by the user (e.g. via pushing annotations)
        for i, label in self.label_overrides.items():
            # Label overrides {dend index: label}
            # We need to translate into original indices
            labels[i] = label

        # Add counts - e.g. "CB12345(10)"
        if self.label_count_check.isChecked():
            counts = pd.Series(labels).value_counts().to_dict()  # dict is much faster
            labels = [
                f"{label}({counts[label]})" if counts[label] > 1 else label
                for label in labels
            ]
        self.figure.labels = labels

        # Update searchbar completer
        if not hasattr(self, "_label_models"):
            self._label_models = {}
        if (label, self.label_count_check.isChecked()) not in self._label_models:
            self._label_models[(label, self.label_count_check.isChecked())] = (
                QtCore.QStringListModel(np.unique(labels).tolist())
            )

        self.searchbar_completer.setModel(
            self._label_models[(label, self.label_count_check.isChecked())]
        )

        # Update label lines
        if self.figure.show_label_lines:
            self.figure.make_label_lines()

    def switch_labels(self):
        """Switch between current and last labels."""
        if hasattr(self, "_last_leaf_labels"):
            self.label_combo_box.setCurrentText(self._last_leaf_labels)
            self.set_labels()
            self.figure.show_message(
                f"Labels: {self._current_leaf_labels}", color="lightgreen", duration=2
            )

    def close(self):
        """Close the controls."""
        super().close()

    def ngl_open(self):
        if not hasattr(self.figure, "ngl_viewer"):
            raise ValueError("Figure has no neuroglancer viewer")
        scene = self.figure.ngl_viewer.neuroglancer_scene(
            group_by=self.ngl_split_combo_box.currentText().lower(),
            use_colors=self.ngl_use_colors.isChecked(),
        )
        scene.open()

    def ngl_copy(self):
        if not hasattr(self.figure, "ngl_viewer"):
            raise ValueError("Figure has no neuroglancer viewer")
        scene = self.figure.ngl_viewer.neuroglancer_scene(
            group_by=self.ngl_split_combo_box.currentText().lower(),
            use_colors=self.ngl_use_colors.isChecked(),
        )
        scene.to_clipboard()
        self.figure.show_message(
            "Link copied to clipboard", color="lightgreen", duration=2
        )

    def calculate_embeddings(self):
        """Re-calculate embeddings and move points to their new positions."""
        dists = self._selected_embedding_data()
        if dists is None:
            return

        dists = self._apply_embedding_feature_partition_filter(dists)
        if (
            hasattr(dists, "shape")
            and len(dists.shape) > 1
            and dists.shape[1] == 0
        ):
            self.figure.show_message(
                "Select at least one parcellation to compute embeddings.",
                color="red",
                duration=3,
            )
            return

        is_precomputed = is_precomputed_distance_matrix(dists)
        metric = (
            "precomputed"
            if is_precomputed
            else self.umap_feature_metric_combo_box.currentText().strip().lower()
        )
        method = self.umap_method_combo_box.currentText()
        random_state = (
            int(self.umap_random_seed.text()) if self.umap_random_seed.text() else None
        )
        fit = make_embedding_estimator(
            method,
            metric=metric,
            is_precomputed=is_precomputed,
            random_state=random_state,
            umap_n_neighbors=self.umap_n_neighbors_slider.value(),
            umap_min_dist=self.umap_min_dist_slider.value(),
            umap_spread=self.umap_spread_slider.value(),
            umap_set_op_mix_ratio=self.umap_set_op_mix_ratio_spinbox.value(),
            umap_densmap=self.umap_densmap_check.isChecked(),
            umap_dens_lambda=self.umap_dens_lambda_spinbox.value(),
            mds_n_init=self.mds_n_init_slider.value(),
            mds_max_iter=self.mds_max_iter_slider.value(),
            mds_eps=self.mds_eps_slider.value(),
        )

        if self.umap_selection_only.isChecked():
            # Get the selected indices
            selected_indices = self.selected_indices
            if selected_indices.size == 0:
                self.figure.show_message("No points selected", color="red", duration=2)
                return

            # Get the distances for the selected indices
            if dists.shape[0] == dists.shape[1]:
                if isinstance(dists, pd.DataFrame):
                    dists = dists.iloc[selected_indices, selected_indices].copy()
                else:
                    dists = dists[np.ix_(selected_indices, selected_indices)].copy()
            else:
                if isinstance(dists, pd.DataFrame):
                    dists = dists.iloc[selected_indices].copy()
                else:
                    dists = dists[selected_indices].copy()

            # Get the data for the selected indices
            data = self.meta_data.iloc[selected_indices].copy().reset_index(drop=True)

            pca_components = (
                self.pca_n_components_slider.value()
                if ((not is_precomputed) and self.pca_check.isChecked())
                else None
            )
            if pca_components is not None:
                print(
                    f" Using PCA to reduce {dists.shape} observation vector to {pca_components} components",
                    flush=True,
                )

            _dists = prepare_embedding_input(
                dists.values if isinstance(dists, pd.DataFrame) else dists,
                is_precomputed=is_precomputed,
                method=method,
                metric=metric,
                rebalance_mode=self.umap_feature_rebalance_combo_box.currentText(),
                pca_n_components=pca_components,
                random_state=random_state,
            )

            # Re-calculate the x/y coordinates
            with warnings.catch_warnings(action="ignore"):
                xy = fit.fit_transform(_dists)
            data["x"] = xy[:, 0]
            data["y"] = xy[:, 1]

            selected_ids = data["id"].to_numpy()

            selected_distances = None
            selected_features = None

            # Preserve both modalities when both are available in the source figure.
            source_matrices = getattr(self.figure, "dists", None)
            if isinstance(source_matrices, dict):
                source_distances = source_matrices.get("distances")
                source_features = source_matrices.get("features")

                if source_distances is not None:
                    if isinstance(source_distances, pd.DataFrame):
                        selected_distances = source_distances.iloc[
                            selected_indices, selected_indices
                        ].copy()
                    else:
                        selected_distances = pd.DataFrame(
                            source_distances[
                                np.ix_(selected_indices, selected_indices)
                            ],
                            index=selected_ids,
                            columns=selected_ids,
                        )

                if source_features is not None:
                    if isinstance(source_features, pd.DataFrame):
                        selected_features = source_features.iloc[
                            selected_indices
                        ].copy()
                        selected_features.index = selected_ids
                        selected_features = self._apply_embedding_feature_partition_filter(
                            selected_features
                        )
                    else:
                        selected_features = pd.DataFrame(
                            np.asarray(source_features)[selected_indices],
                            index=selected_ids,
                        )

            # Fallback for non-dict sources: preserve whichever modality was used.
            if selected_distances is None and selected_features is None:
                if is_precomputed:
                    if isinstance(dists, pd.DataFrame):
                        selected_distances = dists.copy()
                    else:
                        selected_distances = pd.DataFrame(
                            dists,
                            index=selected_ids,
                            columns=selected_ids,
                        )
                else:
                    if isinstance(dists, pd.DataFrame):
                        selected_features = dists.copy()
                        selected_features.index = selected_ids
                    else:
                        selected_features = pd.DataFrame(dists, index=selected_ids)

            from ..core import MainWindow

            window = MainWindow()
            parent_window = self.window()
            if parent_window is not None:
                window.move(parent_window.pos() + QtCore.QPoint(40, 40))
            window.show()

            main_widget = window.centralWidget()
            fig = main_widget.fig_scatter
            fig.clear()

            fig.set_points(
                points=xy,
                metadata=data,
                label_col="label",
                id_col="id",
                color_col="_color",
                marker_col="dataset",
                hover_col="\n".join(
                    [
                        f"{c}: {{{c}}}"
                        for c in data.columns
                        if not str(c).startswith("_")
                    ]
                ),
                dataset_col="dataset",
                point_size=10,
                distances=selected_distances,
                features=selected_features,
            )
            main_widget.scatter_controls.update_controls()

            # Try to propagate neuroglancer source data for the selected subset.
            try:
                src_viewer = getattr(self.figure, "ngl_viewer", None)
                src_data = getattr(src_viewer, "data", None)
                if src_data is not None and len(src_data):
                    if (
                        isinstance(src_data.index, pd.MultiIndex)
                        and "dataset" in data.columns
                    ):
                        keys = list(zip(data["id"].tolist(), data["dataset"].tolist()))
                        ngl_data = src_data.loc[keys].copy()
                    else:
                        ngl_data = src_data.loc[data["id"].tolist()].copy()

                    main_widget.ngl_viewer.set_data(ngl_data)

                    neuropil_mesh = getattr(src_viewer, "neuropil_mesh", None)
                    if neuropil_mesh is not None:
                        main_widget.ngl_viewer.set_neuropil_mesh(
                            neuropil_mesh,
                            neuropil_source=getattr(
                                src_viewer, "neuropil_source", None
                            ),
                        )
            except Exception as e:
                logger.debug(
                    f"Failed to propagate Neuroglancer data to new window: {e}"
                )

            window._data = {
                "meta": data,
                "embeddings": xy,
                "distances": selected_distances,
                "features": selected_features,
            }
            window._update_view_actions()
            window.setWindowTitle(f"BigClust - {method} selection ({len(data)})")

        else:
            pca_components = (
                self.pca_n_components_slider.value()
                if ((not is_precomputed) and self.pca_check.isChecked())
                else None
            )
            if pca_components is not None:
                print(
                    f" Using PCA to reduce {dists.shape} observation vector to {pca_components} components",
                    flush=True,
                )

            dists = prepare_embedding_input(
                dists.values if isinstance(dists, pd.DataFrame) else dists,
                is_precomputed=is_precomputed,
                method=method,
                metric=metric,
                rebalance_mode=self.umap_feature_rebalance_combo_box.currentText(),
                pca_n_components=pca_components,
                random_state=random_state,
            )

            with warnings.catch_warnings(action="ignore"):
                xy = fit.fit_transform(dists)

            # This moves points to their new positions
            self.figure.move_points(xy)

    def update_embedding_settings(self):
        """Update the embedding settings based on the selected method."""
        method = self.umap_method_combo_box.currentText()
        if method == "UMAP":
            self.umap_settings_widget.show()
            self.mds_settings_widget.hide()
            self.umap_button.setText("Run UMAP")
        elif method == "MDS":
            self.umap_settings_widget.hide()
            self.mds_settings_widget.show()
            self.umap_button.setText("Run MDS")
        else:
            self.umap_settings_widget.hide()
            self.mds_settings_widget.hide()
            self.umap_button.setText(f"Run {method}")

        self.calculate_embeddings_maybe()

    def update_searchbar_completer(self):
        """Update the searchbar completer."""
        if not hasattr(self, "_label_models"):
            self._label_models = {}

        label = self.label_combo_box.currentText()
        labels = self.figure.labels
        logger.debug(
            f"Updating searchbar completer for {label} with {len(labels)} labels"
        )
        if (label, self.label_count_check.isChecked()) not in self._label_models:
            self._label_models[(label, self.label_count_check.isChecked())] = (
                QtCore.QStringListModel(np.unique(labels).tolist())
            )

        self.searchbar_completer.setModel(
            self._label_models[(label, self.label_count_check.isChecked())]
        )

    def calculate_embeddings_maybe(self):
        """Recalculate embeddings if the auto-run checkbox is checked."""
        if self.umap_auto_run.isChecked():
            if self.umap_selection_only.isChecked():
                self.figure.show_message(
                    "Can't automatically run UMAP on a selection. Please run it manually.",
                    duration=5,
                    color="red",
                )
            self.calculate_embeddings()


class QHLine(QtWidgets.QFrame):
    def __init__(self):
        super(QHLine, self).__init__()
        self.setFrameShape(QtWidgets.QFrame.HLine)
        self.setFrameShadow(QtWidgets.QFrame.Sunken)


class QVLine(QtWidgets.QFrame):
    def __init__(self):
        super(QVLine, self).__init__()
        self.setFrameShape(QtWidgets.QFrame.VLine)
        self.setFrameShadow(QtWidgets.QFrame.Sunken)
