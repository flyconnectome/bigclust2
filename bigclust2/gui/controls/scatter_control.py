import re
import os
import uuid
import logging
import warnings
import pyperclip
import traceback

import pandas as pd
import numpy as np

from PySide6 import QtWidgets, QtCore
from concurrent.futures import ThreadPoolExecutor

from ...embeddings import (
    is_precomputed_distance_matrix,
    make_embedding_estimator,
    prepare_embedding_input,
    neighborhood_fidelity,
)


CLIO_CLIENT = None
CLIO_ANN = None
NEUPRINT_CLIENT = None
FLYWIRE_ANN = None
HB_ANN = None

logger = logging.getLogger(__name__)


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
        self.tab2 = QtWidgets.QWidget()
        self.tab3 = QtWidgets.QWidget()
        self.tab4 = QtWidgets.QWidget()
        self.tab5 = QtWidgets.QWidget()
        self.tab6 = QtWidgets.QWidget()
        self.tab1_layout = QtWidgets.QVBoxLayout()
        self.tab2_layout = QtWidgets.QVBoxLayout()
        self.tab3_layout = QtWidgets.QVBoxLayout()
        self.tab4_layout = QtWidgets.QVBoxLayout()
        self.tab5_layout = QtWidgets.QVBoxLayout()
        self.tab6_layout = QtWidgets.QVBoxLayout()
        self.tab1.setLayout(self.tab1_layout)
        self.tab2.setLayout(self.tab2_layout)
        self.tab3.setLayout(self.tab3_layout)
        self.tab4.setLayout(self.tab4_layout)
        self.tab5.setLayout(self.tab5_layout)
        self.tab6.setLayout(self.tab6_layout)
        self.tabs.addTab(self.tab1, "General")
        self.tabs.addTab(self.tab2, "Annotation")
        # self.tabs.addTab(self.tab3, "Neuroglancer")
        self.tabs.addTab(self.tab4, "Settings")
        self.tabs.addTab(self.tab5, "Embeddings")
        self.tabs.addTab(self.tab6, "Fidelity")

        self.build_control_gui()
        # self.build_annotation_gui()
        # self.build_neuroglancer_gui()
        # self.build_settings_gui()
        self.build_embeddings_gui()
        self.build_fidelity_gui()

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
        raise NotImplementedError("This method should be implemented in a subclass.")

    def build_control_gui(self):
        """Build the GUI."""
        # Search bar
        self.tab1_layout.addWidget(QtWidgets.QLabel("Search"))
        self.searchbar = QtWidgets.QLineEdit()
        self.searchbar.setToolTip(
            "Search for a label in the scene. Use a leading '/' to search for a regex."
        )
        self.searchbar.returnPressed.connect(self.find_next)
        # self.searchbar.textChanged.connect(self.figure.highlight_cluster)
        self.searchbar_completer = QtWidgets.QCompleter(self.labels)
        self.searchbar_completer.setCaseSensitivity(QtCore.Qt.CaseInsensitive)
        self.searchbar.setCompleter(self.searchbar_completer)
        self.tab1_layout.addWidget(self.searchbar)

        # Add buttons for previous/next
        self.button_layout = QtWidgets.QHBoxLayout()
        self.button_layout.setContentsMargins(0, 0, 0, 0)
        self.button_layout.setSpacing(0)
        self.prev_button = QtWidgets.QPushButton("Previous")
        self.prev_button.clicked.connect(self.find_previous)
        self.button_layout.addWidget(self.prev_button)
        self.find_sel_button = QtWidgets.QPushButton("Select")
        self.find_sel_button.setToolTip("Select all objects matching the search term.")
        self.find_sel_button.clicked.connect(self.find_select)
        self.button_layout.addWidget(self.find_sel_button)
        self.next_button = QtWidgets.QPushButton("Next")
        self.next_button.clicked.connect(self.find_next)
        self.button_layout.addWidget(self.next_button)
        self.tab1_layout.addLayout(self.button_layout)

        # Add horizontal divider
        self.add_split(self.tab1_layout)

        # Add dropdown to choose leaf labels
        self.label_layout = QtWidgets.QHBoxLayout()
        self.tab1_layout.addLayout(self.label_layout)
        self.label_layout.addWidget(QtWidgets.QLabel("Labels:"))
        self.label_combo_box = QtWidgets.QComboBox()
        self.label_layout.addWidget(self.label_combo_box)
        self.label_combo_box.currentIndexChanged.connect(self.set_labels)
        self._current_leaf_labels = self.label_combo_box.currentText()

        # Checkbox for whether to show label counts
        self.label_count_check = QtWidgets.QCheckBox("Show label counts")
        self.label_count_check.setToolTip("Whether to add counts to the labels.")
        self.label_count_check.setChecked(False)
        self.label_count_check.stateChanged.connect(self.set_label_counts)
        self.tab1_layout.addWidget(self.label_count_check)

        # Checkbox for whether to show label outlines
        self.label_outlines_check = QtWidgets.QCheckBox("Show label outlines")
        self.label_outlines_check.setToolTip(
            "Whether to add draw polygons around neurons with the same label."
        )
        self.label_outlines_check.setChecked(False)
        self.label_outlines_check.stateChanged.connect(self.set_label_outlines)
        self.tab1_layout.addWidget(self.label_outlines_check)

        # Add horizontal divider
        self.add_split(self.tab1_layout)

        # Add dropdown to choose color mode
        self.tab1_layout.addWidget(QtWidgets.QLabel("Color neurons by:"))
        self.color_combo_box = QtWidgets.QComboBox()
        self.color_combo_box.addItem("Default")
        self.color_combo_box.addItem("Dataset")
        self.color_combo_box.addItem("Cluster")
        self.color_combo_box.addItem("Label")
        self.color_combo_box.addItem("Random")
        self.color_combo_box.setItemData(
            0, "Color neurons by viewer default", QtCore.Qt.ToolTipRole
        )
        self.color_combo_box.setItemData(
            1, "Color neurons by dataset", QtCore.Qt.ToolTipRole
        )
        self.color_combo_box.setItemData(
            2, "Color neurons by cluster", QtCore.Qt.ToolTipRole
        )
        self.color_combo_box.setItemData(
            3, "Color neurons by label", QtCore.Qt.ToolTipRole
        )
        self.color_combo_box.setItemData(
            4, "Randomly color neurons", QtCore.Qt.ToolTipRole
        )
        self.tab1_layout.addWidget(self.color_combo_box)

        # Set the action for the color combo box
        self.color_combo_box.currentIndexChanged.connect(self.set_color_mode)

        self.add_group_check = QtWidgets.QCheckBox("Add as group")
        self.add_group_check.setToolTip("Whether to add neurons as group when selected")
        self.add_group_check.setChecked(False)
        self.add_group_check.stateChanged.connect(self.set_add_group)
        self.tab1_layout.addWidget(self.add_group_check)

        self.dclick_deselect = QtWidgets.QCheckBox("Deselect on double-click")
        self.dclick_deselect.setToolTip("You can always deselect using ESC")
        self.dclick_deselect.setChecked(self.figure.deselect_on_dclick)
        self.dclick_deselect.stateChanged.connect(self.set_dclick_deselect)
        self.tab1_layout.addWidget(self.dclick_deselect)

        self.empty_deselect = QtWidgets.QCheckBox("Deselect on empty selection")
        self.empty_deselect.setToolTip("You can always deselect using ESC")
        self.empty_deselect.setChecked(self.figure.deselect_on_empty)
        self.empty_deselect.stateChanged.connect(self.set_empty_deselect)
        self.tab1_layout.addWidget(self.empty_deselect)

        # Add horizontal divider
        self.add_split(self.tab1_layout)

        # Add a checkbox + spinbox to show distances as edges
        self.show_distance_edges_check = QtWidgets.QCheckBox("Show distances as edges")
        self.show_distance_edges_check.setToolTip(
            "Whether to show actual distances as edges between points."
        )
        self.show_distance_edges_check.setChecked(False)
        self.tab1_layout.addWidget(self.show_distance_edges_check)

        hlayout = QtWidgets.QHBoxLayout()
        self.tab1_layout.addLayout(hlayout)
        self.distance_edges_threshold = QtWidgets.QLabel("Edge threshold:")
        self.distance_edges_threshold.setToolTip(
            "Set the threshold for showing edges between points."
        )
        hlayout.addWidget(self.distance_edges_threshold)
        self.distance_edges_slider = QtWidgets.QDoubleSpinBox()
        self.distance_edges_slider.setRange(0.0, 1.0)
        self.distance_edges_slider.setSingleStep(0.05)
        self.distance_edges_slider.setValue(self.figure.distance_edges_threshold)
        self.distance_edges_slider.setDecimals(2)
        self.distance_edges_slider.valueChanged.connect(
            lambda x: setattr(self.figure, "distance_edges_threshold", float(x))
        )
        hlayout.addWidget(self.distance_edges_slider)
        self.show_distance_edges_check.stateChanged.connect(
            lambda x: setattr(self.figure, "show_distance_edges", bool(x))
        )
        self.show_distance_edges_check.stateChanged.connect(
            lambda x: self.distance_edges_slider.setEnabled(
                self.show_distance_edges_check.isChecked()
            )
        )

        # This would make it so the legend does not stretch when
        # we resize the window vertically
        self.tab1_layout.addStretch(1)

        return

    def build_neuroglancer_gui(self):
        # Add buttons to generate neuroglancer scene
        self.ngl_open_button = QtWidgets.QPushButton("Open in browser")
        self.ngl_open_button.setToolTip(
            "Open the current scene in a new browser window"
        )
        self.ngl_open_button.clicked.connect(self.ngl_open)
        self.tab3_layout.addWidget(self.ngl_open_button)

        self.ngl_copy_button = QtWidgets.QPushButton("Copy to clipboard")
        self.ngl_copy_button.setToolTip("Copy the current scene to the clipboard")
        self.ngl_copy_button.clicked.connect(self.ngl_copy)
        self.tab3_layout.addWidget(self.ngl_copy_button)

        # Add checkbox to determine whether to use colours
        self.ngl_use_colors = QtWidgets.QCheckBox("Use colors")
        self.ngl_use_colors.setToolTip(
            "Whether to use re-use colors from bigclust for the neuroglancer scene. If False, colours will be determined by neuroglancer."
        )
        self.ngl_use_colors.setChecked(True)
        self.tab3_layout.addWidget(self.ngl_use_colors)

        # Dropdown to choose whether to split neurons into layers other than source
        self.tab3_layout.addWidget(QtWidgets.QLabel("Group neurons into layers by:"))
        self.ngl_split_combo_box = QtWidgets.QComboBox()
        self.ngl_split_combo_box.addItem("Source")
        self.ngl_split_combo_box.addItem("Color")
        self.ngl_split_combo_box.addItem("Label")
        self.ngl_split_combo_box.setToolTip(
            "Determine how neurons are grouped into layers."
        )
        self.tab3_layout.addWidget(self.ngl_split_combo_box)

        # Checkbox for whether to cache neurons
        self.ngl_cache_neurons = QtWidgets.QCheckBox("Cache neurons")
        self.ngl_cache_neurons.setToolTip("Whether cache neuron meshes.")
        if hasattr(self.figure, "_ngl_viewer"):
            self.ngl_cache_neurons.setChecked(self.figure._ngl_viewer.use_cache)
        else:
            self.ngl_cache_neurons.setChecked(False)
        self.ngl_cache_neurons.stateChanged.connect(self.set_ngl_cache)
        self.tab3_layout.addWidget(self.ngl_cache_neurons)

        # Checkbox for debug mode
        self.ngl_debug_mode = QtWidgets.QCheckBox("Debug mode")
        self.ngl_debug_mode.setToolTip(
            "Whether to show debug information for the neuroglancer view"
        )
        self.ngl_debug_mode.setChecked(False)
        self.ngl_debug_mode.stateChanged.connect(self.set_ngl_debug)
        self.tab3_layout.addWidget(self.ngl_debug_mode)

        # This makes it so the legend does not stretch
        self.tab3_layout.addStretch(1)

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
        self.render_mode_dropdown.currentIndexChanged.connect(
            lambda x: setattr(
                self.figure,
                "render_trigger",
                render_trigger_vals[self.render_mode_dropdown.currentIndex()],
            )
        )
        # Set default item to whatever the currently set render trigger is
        self.render_mode_dropdown.setCurrentIndex(
            render_trigger_vals.index(self.figure.render_trigger)
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
        self.font_size_slider.setRange(1, 200)
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

    def build_annotation_gui(self):
        # Add buttons to push annotations
        self.push_ann_button = QtWidgets.QPushButton("Push annotations")
        self.push_ann_button.setToolTip(
            "Push the current annotation to selected fields"
        )
        self.push_ann_button.clicked.connect(self.push_annotation)
        self.tab2_layout.addWidget(self.push_ann_button)

        self.ann_combo_box = QtWidgets.QComboBox()
        self.ann_combo_box.setEditable(True)
        self.tab2_layout.addWidget(self.ann_combo_box)

        self.clear_ann_button = QtWidgets.QPushButton("Clear annotations")
        self.clear_ann_button.setToolTip("Clear the current annotations")
        self.clear_ann_button.clicked.connect(self.clear_annotation)
        self.tab2_layout.addWidget(self.clear_ann_button)

        self.tab2_layout.addWidget(QtWidgets.QLabel("Which fields to set/clear:"))

        # Create a horizontal layout to hold the two vertical layouts:
        # one for Clio and one for FlyTable
        horizontal_layout = QtWidgets.QHBoxLayout()
        self.tab2_layout.addLayout(horizontal_layout)

        # Create the first vertical layout
        left_vertical_layout = QtWidgets.QVBoxLayout()
        horizontal_layout.addLayout(left_vertical_layout)

        left_vertical_layout.addWidget(QtWidgets.QLabel("Clio:"))
        self.set_clio_type = QtWidgets.QCheckBox("type")
        self.set_clio_type.setTristate(True)
        self.set_clio_type.setToolTip(
            "If fully checked, will edit both type and instance. Set to partially checked leave instance unchanged."
        )
        left_vertical_layout.addWidget(self.set_clio_type)
        self.set_clio_flywire_type = QtWidgets.QCheckBox("flywire_type")
        left_vertical_layout.addWidget(self.set_clio_flywire_type)
        self.set_clio_hemibrain_type = QtWidgets.QCheckBox("hemibrain_type")
        left_vertical_layout.addWidget(self.set_clio_hemibrain_type)
        self.set_clio_manc_type = QtWidgets.QCheckBox("manc_type")
        left_vertical_layout.addWidget(self.set_clio_manc_type)

        # Add a vertical line to separate the layouts
        vertical_line = QVLine()
        horizontal_layout.addWidget(vertical_line)

        # Create the second vertical layout
        right_vertical_layout = QtWidgets.QVBoxLayout()
        horizontal_layout.addLayout(right_vertical_layout)

        right_vertical_layout.addWidget(QtWidgets.QLabel("FlyTable:"))
        self.set_flytable_type = QtWidgets.QCheckBox("cell_type")
        right_vertical_layout.addWidget(self.set_flytable_type)
        self.set_flytable_mcns_type = QtWidgets.QCheckBox("malecns_type")
        right_vertical_layout.addWidget(self.set_flytable_mcns_type)
        self.set_flytable_hemibrain_type = QtWidgets.QCheckBox("hemibrain_type")
        right_vertical_layout.addWidget(self.set_flytable_hemibrain_type)

        self.tab2_layout.addWidget(QtWidgets.QLabel("Settings:"))

        self.set_sanity_check = QtWidgets.QCheckBox("Sanity checks")
        self.set_sanity_check.setToolTip("Whether to perform sanity checks")
        self.set_sanity_check.setChecked(True)
        self.tab2_layout.addWidget(self.set_sanity_check)

        # Add dropdown to set dimorphism status
        self.sel_dimorphism_action = QtWidgets.QPushButton(text="Set dimorphism")
        self.tab2_layout.addWidget(self.sel_dimorphism_action)
        self.sel_dimorphism_action_menu = QtWidgets.QMenu(self)
        self.sel_dimorphism_action.setMenu(self.sel_dimorphism_action_menu)

        # Set actions for the dropdown
        self.sel_dimorphism_action_menu.addAction("Sex-specific")
        self.sel_dimorphism_action_menu.actions()[-1].triggered.connect(
            lambda x: self.selected_set_dimorphism("sex-specific")
        )
        self.sel_dimorphism_action_menu.addAction("Sexually dimorphic")
        self.sel_dimorphism_action_menu.actions()[-1].triggered.connect(
            lambda x: self.selected_set_dimorphism("sexually dimorphic")
        )
        self.sel_dimorphism_action_menu.addAction("Pot. sex-specific")
        self.sel_dimorphism_action_menu.actions()[-1].triggered.connect(
            lambda x: self.selected_set_dimorphism("potentially sex-specific")
        )
        self.sel_dimorphism_action_menu.addAction("Pot. sexually dimorphic")
        self.sel_dimorphism_action_menu.actions()[-1].triggered.connect(
            lambda x: self.selected_set_dimorphism("potentially sexually dimorphic")
        )
        self.sel_dimorphism_action_menu.addAction("Not dimorphic")
        self.sel_dimorphism_action_menu.actions()[-1].triggered.connect(
            lambda x: self.selected_set_dimorphism(None)
        )

        # Make a separate layout with tighter margins for the buttons
        grid_layout = QtWidgets.QGridLayout()
        grid_layout.setContentsMargins(0, 0, 0, 0)  # No margins around the grid
        grid_layout.setSpacing(0)  # No space between buttons
        grid_layout.setColumnStretch(1, 1)  # Make the button column stretchable
        self.tab2_layout.addLayout(grid_layout)

        # Add checkbox + button to set new Clio group
        checkbox = QtWidgets.QCheckBox(" ")  # Do not remove the whitespace!
        checkbox.setToolTip("Check to activate the button to set a new Clio group.")
        checkbox.setChecked(False)
        checkbox.stateChanged.connect(
            lambda x: self.clio_group_button.setEnabled(x == 2)
        )
        grid_layout.addWidget(checkbox, 0, 0)
        self.clio_group_button = QtWidgets.QPushButton("Set new Clio group")
        self.clio_group_button.setToolTip(
            "Assign new Clio group. This will use the lowest body ID as group ID."
        )
        self.clio_group_button.clicked.connect(self.new_clio_group)
        self.clio_group_button.setEnabled(False)
        grid_layout.addWidget(self.clio_group_button, 0, 1)

        # Add button to suggest new MCNS type
        self.suggest_type_button = QtWidgets.QPushButton("Suggest new HB-style type")
        self.suggest_type_button.setToolTip(
            "Suggest new Hemibrain-style type based on main input neuropil(s). See console for output."
        )
        self.suggest_type_button.clicked.connect(self.suggest_type)
        grid_layout.addWidget(self.suggest_type_button, 1, 0, 1, -1)

        # Add button to suggest new MCNS type
        self.suggest_male_type_button = QtWidgets.QPushButton("Suggest male-only type")
        self.suggest_male_type_button.setToolTip(
            "Suggest new male-only type based on main input neuropil(s). See console for output."
        )
        self.suggest_male_type_button.clicked.connect(self.suggest_male_type)
        grid_layout.addWidget(self.suggest_male_type_button, 2, 0, 1, -1)

        # Add button to suggest new CB type
        self.suggest_cb_type_button = QtWidgets.QPushButton("Suggest new CB-type")
        self.suggest_cb_type_button.setToolTip("Suggest new CBXXXX type.")
        self.suggest_cb_type_button.clicked.connect(self.suggest_cb_type)
        grid_layout.addWidget(self.suggest_cb_type_button, 3, 0, 1, -1)

        # Add checkbox + button to set new super type
        checkbox = QtWidgets.QCheckBox(" ")  # Do not remove the whitespace!
        checkbox.setToolTip("Check to activate the button to set a new supertype.")
        checkbox.setChecked(False)
        checkbox.stateChanged.connect(
            lambda x: self.set_supertype_button.setEnabled(x == 2)
        )
        grid_layout.addWidget(checkbox, 4, 0)
        self.set_supertype_button = QtWidgets.QPushButton("Set new SuperType")
        self.set_supertype_button.setToolTip(
            "Assign selected neurons to a supertype. This will use the lowest ID as supertype ID."
        )
        self.set_supertype_button.clicked.connect(self.new_super_type)
        self.set_supertype_button.setEnabled(False)
        grid_layout.addWidget(self.set_supertype_button, 4, 1)

        # This makes it so the legend does not stretch
        self.tab2_layout.addStretch(1)

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

        actions_row.addStretch(1)

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
        self.fidelity_knn_combo_box.currentIndexChanged.connect(
            self.set_knn_edges
        )
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

        self.tab6_layout.addStretch(1)

        self.update_fidelity_options()

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
        self.update_label_combo_box()
        self.update_searchbar_completer()
        self.update_distance_edges_controls()

    def update_label_combo_box(self):
        """Update the items in the label combo box."""
        # First clear all existing items
        self.label_combo_box.clear()

        self.label_combo_box.addItem("Default")
        if self.meta_data is not None:
            for col in sorted(self.meta_data.columns):
                if col.startswith("_"):
                    continue
                self.label_combo_box.addItem(col)

    def update_distance_edges_controls(self):
        """Update the distance edges controls."""
        dists = getattr(self.figure, "dists", None)
        self.show_distance_edges_check.setEnabled(False)
        self.distance_edges_threshold.setEnabled(False)
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
        self.distance_edges_threshold.setEnabled(True)
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

    def _update_embedding_input_controls(self):
        """Toggle embedding controls based on selected input type."""
        if not hasattr(self, "pca_check"):
            return

        arr = self._selected_embedding_data()
        is_feature_input = (arr is not None) and (
            not is_precomputed_distance_matrix(arr)
        )

        if hasattr(self, "feature_options_group"):
            self.feature_options_group.setEnabled(is_feature_input)

        # PCA pre-reduction is only meaningful for feature vectors.
        if not is_feature_input:
            self.pca_check.setChecked(False)
        self.pca_check.setEnabled(is_feature_input)
        self.pca_n_components_slider.setEnabled(
            is_feature_input and self.pca_check.isChecked()
        )

    # def update_ann_combo_box(self):
    #     """Update the items in the annotation combo box."""
    #     # First clear all existing items
    #     self.ann_combo_box.clear()

    #     if self.figure.selected_labels is None:
    #         return

    #     # Now add the new items currently selected
    #     for label in sorted(list(set(self.figure.selected_labels))):
    #         # Skip if this label is NaN or None (i.e. not a string)
    #         if not isinstance(label, str):
    #             continue

    #         if re.match(".*?\([0-9]+\)", label):
    #             label = label.split("(")[0]

    #         # Replace the "*"
    #         label = label.replace("*", "")

    #         if label in ("untyped",):
    #             continue
    #         self.ann_combo_box.addItem(label)

    @requires_selection
    def selected_set_dimorphism(self, dimorphism):
        """Push dimorphism to Clio/FlyTable."""
        assert dimorphism in (
            "sex-specific",
            "sexually dimorphic",
            "potentially sex-specific",
            "potentially sexually dimorphic",
            None,
        )
        selected_ids = self.figure.selected_ids

        # Extract FlyWire root and MaleCNS body IDs from the selected IDs
        # N.B. This requires meta data to be present.
        rootids, bodyids = sort_ids(selected_ids, self.figure.selected_meta)

        # Get the annotation
        import clio

        global CLIO_CLIENT
        if CLIO_CLIENT is None:
            CLIO_CLIENT = clio.Client(dataset="CNS")

        import ftu

        # Submit the annotations
        self.futures[(dimorphism, uuid.uuid4())] = self.pool.submit(
            _push_dimorphism,
            dimorphism=dimorphism,
            bodyids=bodyids,
            rootids=rootids,
            clio=clio,  #  pass the module
            ftu=ftu,  #  pass the module
            figure=self.figure,
        )

    @requires_selection
    def push_annotation(self):
        """Push the current annotation to Clio/FlyTable."""
        if not any(
            (
                self.set_clio_type.isChecked(),
                self.set_clio_flywire_type.isChecked(),
                self.set_clio_hemibrain_type.isChecked(),
                self.set_clio_manc_type.isChecked(),
                self.set_flytable_type.isChecked(),
                self.set_flytable_mcns_type.isChecked(),
                self.set_flytable_hemibrain_type.isChecked(),
            )
        ):
            self.figure.show_message("No fields to push", color="red", duration=2)
            return

        label = self.ann_combo_box.currentText()
        if not label:
            self.figure.show_message("No label to push", color="red", duration=2)
            return

        # Extract FlyWire root and MaleCNS body IDs from the selected IDs
        # N.B. This requires meta data to be present.
        selected_ids = self.figure.selected_ids
        rootids, bodyids = sort_ids(selected_ids, self.figure.selected_meta)

        # Which fields to set
        clio_to_set = []
        if self.set_clio_type.isChecked():
            clio_to_set.append("type")
            # If the checkbox is fully checked also set the instance
            type_state = self.set_clio_type.checkState()
            if type_state == QtCore.Qt.CheckState.Checked:
                clio_to_set.append("instance")
        if self.set_clio_flywire_type.isChecked():
            clio_to_set.append("flywire_type")
        if self.set_clio_hemibrain_type.isChecked():
            clio_to_set.append("hemibrain_type")
        if self.set_clio_manc_type.isChecked():
            clio_to_set.append("manc_type")
        clio_to_set = tuple(clio_to_set)

        flytable_to_set = []
        if self.set_flytable_type.isChecked():
            flytable_to_set.append("cell_type")
        if self.set_flytable_mcns_type.isChecked():
            flytable_to_set.append("malecns_type")
        if self.set_flytable_hemibrain_type.isChecked():
            flytable_to_set.append("hemibrain_type")
        flytable_to_set = tuple(flytable_to_set)

        # Get the annotation
        import clio

        global CLIO_CLIENT
        if CLIO_CLIENT is None:
            CLIO_CLIENT = clio.Client(dataset="CNS")

        import ftu

        # Submit the annotations
        self.futures[(label, uuid.uuid4())] = self.pool.submit(
            _push_annotations,
            label=label,
            clio_to_set=clio_to_set,
            flytable_to_set=flytable_to_set,
            bodyids=bodyids if clio_to_set else None,
            rootids=rootids if flytable_to_set else None,
            clio=clio,  #  pass the module
            ftu=ftu,  #  pass the module
            figure=self.figure,
            controls=self,
        )

        if clio_to_set and len(bodyids) and CLIO_ANN is not None:
            # Update the CLIO annotations
            for col in clio_to_set:
                CLIO_ANN.loc[
                    CLIO_ANN.get("bodyId", CLIO_ANN.get("bodyid")).isin(bodyids), col
                ] = label

    @requires_selection
    def clear_annotation(self):
        """Clear the currently selected fields."""
        if not any(
            (
                self.set_clio_type.isChecked(),
                self.set_clio_flywire_type.isChecked(),
                self.set_clio_manc_type.isChecked(),
                self.set_clio_hemibrain_type.isChecked(),
                self.set_flytable_type.isChecked(),
                self.set_flytable_mcns_type.isChecked(),
                self.set_flytable_hemibrain_type.isChecked(),
            )
        ):
            self.figure.show_message("No fields to clear", color="red", duration=2)
            return

        # Extract FlyWire root and MaleCNS body IDs from the selected IDs
        # N.B. This requires meta data to be present.
        selected_ids = self.figure.selected_ids
        rootids, bodyids = sort_ids(selected_ids, self.figure.selected_meta)

        # Which fields to clear
        clio_to_clear = []
        if self.set_clio_type.isChecked():
            clio_to_clear.append("type")
        if self.set_clio_flywire_type.isChecked():
            clio_to_clear.append("flywire_type")
        if self.set_clio_hemibrain_type.isChecked():
            clio_to_clear.append("hemibrain_type")
        if self.set_clio_manc_type.isChecked():
            clio_to_clear.append("manc_type")
        clio_to_clear = tuple(clio_to_clear)

        flytable_to_clear = []
        if self.set_flytable_type.isChecked():
            flytable_to_clear.append("cell_type")
        if self.set_flytable_mcns_type.isChecked():
            flytable_to_clear.append("malecns_type")
        if self.set_flytable_hemibrain_type.isChecked():
            flytable_to_clear.append("hemibrain_type")
        flytable_to_clear = tuple(flytable_to_clear)

        # Get the annotation
        import clio

        global CLIO_CLIENT
        if CLIO_CLIENT is None:
            CLIO_CLIENT = clio.Client(dataset="CNS")

        import ftu

        # Submit the annotations
        self.futures[uuid.uuid4()] = self.pool.submit(
            _clear_annotations,
            bodyids=bodyids if clio_to_clear else None,
            rootids=rootids if flytable_to_clear else None,
            clio_to_clear=clio_to_clear,
            flytable_to_clear=flytable_to_clear,
            clio=clio,  #  pass the module
            ftu=ftu,  #  pass the module
            figure=self.figure,
            controls=self,
        )

    @requires_selection
    def new_super_type(self):
        """Set a new super type for given IDs."""
        # N.B. This requires meta data to be present.
        selected_ids = self.figure.selected_ids
        rootids, bodyids = sort_ids(selected_ids, self.figure.selected_meta)

        # New type name
        new_type = min(selected_ids)

        # Get the clio module
        import clio

        global CLIO_CLIENT
        if CLIO_CLIENT is None:
            CLIO_CLIENT = clio.Client(dataset="CNS")

        import ftu

        # Submit the annotations
        self.futures[(new_type, uuid.uuid4())] = self.pool.submit(
            _push_super_type,
            super_type=new_type,
            bodyids=bodyids,
            rootids=rootids,
            clio=clio,  #  pass the module
            sanity_checks=self.set_sanity_check.isChecked(),
            ftu=ftu,
            figure=self.figure,
        )

    @requires_selection
    def new_clio_group(self):
        """Set a new Clio group for given IDs."""
        # MaleCNS body IDs from the selected IDs
        # N.B. This requires meta data to be present.
        selected_ids = self.figure.selected_ids
        _, bodyids = sort_ids(selected_ids, self.figure.selected_meta)

        if not len(bodyids):
            self.figure.show_message(
                "No MCNS neurons selected", color="red", duration=2
            )
            return

        group = min(bodyids)

        # Get the clio module
        import clio

        global CLIO_CLIENT
        if CLIO_CLIENT is None:
            CLIO_CLIENT = clio.Client(dataset="CNS")

        # Submit the annotations
        self.futures[(group, uuid.uuid4())] = self.pool.submit(
            _push_group,
            group=group,
            bodyids=bodyids,
            clio=clio,  #  pass the module
            figure=self.figure,
        )

    @requires_selection
    def suggest_type(self):
        """Suggest a new type for given IDs."""
        selected_ids = self.figure.selected_ids
        # Extract FlyWire root and MaleCNS body IDs from the selected IDs
        # N.B. This requires meta data to be present.
        _, bodyids = sort_ids(selected_ids, self.figure.selected_meta)

        if not len(bodyids):
            self.figure.show_message(
                "No MCNS neurons selected", color="red", duration=2
            )
            return

        # Threading this doesn't make much sense
        suggest_new_label(bodyids=bodyids)

    @requires_selection
    def suggest_male_type(self):
        """Suggest a new male-only type for given IDs."""
        selected_ids = self.figure.selected_ids
        # Extract FlyWire root and MaleCNS body IDs from the selected IDs
        # N.B. This requires meta data to be present.
        _, bodyids = sort_ids(selected_ids, self.figure.selected_meta)

        if not len(bodyids):
            self.figure.show_message(
                "No MCNS neurons selected", color="red", duration=2
            )
            return

        # Threading this doesn't make much sense
        suggest_new_label(bodyids=bodyids, male_only=True)

    def suggest_cb_type(self):
        """Suggest a new CB type."""
        # Threading this doesn't make much sense
        import ftu

        print("Next free CB tyoe:", ftu.info.get_next_cb_id())

    def set_add_group(self):
        """Set whether to add neurons as group when selected."""
        self.figure._add_as_group = self.add_group_check.isChecked()

    def set_ngl_cache(self):
        """Set whether the ngl viewer should cache neurons."""
        if hasattr(self.figure, "_ngl_viewer"):
            self.figure._ngl_viewer.use_cache = self.ngl_cache_neurons.isChecked()

            if not self.ngl_cache_neurons.isChecked():
                self.figure._ngl_viewer.clear_cache()

    def set_ngl_debug(self):
        """Set debug mode for ngl viewer."""
        if hasattr(self.figure, "_ngl_viewer"):
            self.figure._ngl_viewer.debug = self.ngl_debug_mode.isChecked()

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
                self._label_search.select_all()

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

    def set_color_mode(self):
        """Set the color mode."""
        mode = self.color_combo_box.currentText()
        self.figure.set_viewer_color_mode(mode.lower())

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

        labels = self.meta_data[label].astype(str).fillna("").values

        # For labels that were set manually by the user (via pushing annotations)
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
        if hasattr(self.figure, "_label_line_group"):
            # Re-trigger making label lines
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
        if not hasattr(self.figure, "_ngl_viewer"):
            raise ValueError("Figure has no neuroglancer viewer")
        scene = self.figure._ngl_viewer.neuroglancer_scene(
            group_by=self.ngl_split_combo_box.currentText().lower(),
            use_colors=self.ngl_use_colors.isChecked(),
        )
        scene.open()

    def ngl_copy(self):
        if not hasattr(self.figure, "_ngl_viewer"):
            raise ValueError("Figure has no neuroglancer viewer")
        scene = self.figure._ngl_viewer.neuroglancer_scene(
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
            mds_n_init=self.mds_n_init_slider.value(),
            mds_max_iter=self.mds_max_iter_slider.value(),
            mds_eps=self.mds_eps_slider.value(),
        )

        if self.umap_selection_only.isChecked():
            assert isinstance(dists, pd.DataFrame)
            # Get the selected indices
            selected_indices = self.selected_indices

            # Get the distances for the selected indices
            if dists.shape[0] == dists.shape[1]:
                dists = dists.iloc[selected_indices, selected_indices].copy()
            else:
                dists = dists.iloc[selected_indices].copy()

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

            # Create a new figure
            new_fig = self.figure.__class__(
                data, dists=dists, hover_info=self.figure._hover_info_org
            )

            # Add neuroglancer viewer (if it exists)
            if hasattr(self.figure, "_ngl_viewer"):
                ngl = self.figure._ngl_viewer.__class__(
                    data,
                    neuropil_mesh=self.figure._ngl_viewer._neuropil_mesh,
                    title="Viewer Selection",
                )
                new_fig.sync_viewer(ngl)

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


def _push_annotations(
    label,
    bodyids,
    rootids,
    clio,
    ftu,
    clio_to_set,
    flytable_to_set,
    figure=None,
    controls=None,
):
    """Push the current annotation to Clio/FlyTable."""
    try:
        if bodyids is not None and len(bodyids) and clio_to_set:
            ann = pd.DataFrame()
            ann["bodyid"] = bodyids
            for field in clio_to_set:
                if field == "instance":
                    continue
                ann[field] = label

            # Generate instances
            if "instance" in clio_to_set and "type" in clio_to_set:
                data = clio.fetch_annotations(bodyids)
                sides = {}
                for col in ("root_side", "soma_side"):
                    if col in data.columns:
                        sides.update(zip(data["bodyid"], data[col]))
                ann["instance"] = [f"{label}_{sides.get(bid, 'NA')}" for bid in bodyids]

            clio.set_annotations(ann)
        if rootids is not None and len(rootids) and flytable_to_set:
            kwargs = {}

            for field in flytable_to_set:
                kwargs[field] = label

                if field in ("cell_type", "hemibrain_type", "malecns_type"):
                    kwargs[f"{field}_source"] = os.environ.get(
                        "BC_ANNOTATION_USER", "bigclust"
                    )

            ftu.info.update_fields(
                rootids,
                **kwargs,
                id_col="root_783",
                dry_run=False,
            )

        if clio_to_set and flytable_to_set:
            msg = f"Set {label} for {len(bodyids)} maleCNS and {len(rootids)} FlyWire neurons"
        elif clio_to_set:
            msg = f"Set {label} for {len(bodyids)} male CNS neurons"
        elif flytable_to_set:
            msg = f"Set {label} for {len(rootids)} FlyWire neurons"

        print(f"{msg}:")
        if bodyids is not None and len(bodyids) and clio_to_set:
            print("  ", bodyids)
        if rootids is not None and len(rootids) and flytable_to_set:
            print("  ", rootids)

        if figure:
            # Update the labels in the figure
            if clio_to_set and bodyids is not None:
                ind = figure.selected[np.isin(figure.selected_ids, bodyids)]
                figure.set_labels(ind, f"{label}(!)")
                controls.label_overrides.update({i: f"{label}(!)" for i in ind})
            if flytable_to_set and rootids is not None:
                ind = figure.selected[np.isin(figure.selected_ids, rootids)]
                figure.set_labels(ind, f"{label}(!)")
                controls.label_overrides.update({i: f"{label}(!)" for i in ind})

            # Show the message
            figure.show_message(msg, color="lightgreen", duration=2)
    except BaseException as e:
        if figure:
            figure.show_message(
                "Error pushing annotations (see console)", color="red", duration=2
            )
        traceback.print_exc()
        raise


def _push_dimorphism(
    dimorphism,
    bodyids,
    rootids,
    clio,
    ftu,
    figure=None,
):
    """Push dimorphism status to Clio/FlyTable."""
    try:
        if bodyids is not None and len(bodyids):
            label = (
                dimorphism.replace("sex-specific", "male-specific")
                if dimorphism
                else None
            )

            clio.set_fields(bodyids, dimorphism=label)

        if rootids is not None and len(rootids):
            label = (
                dimorphism.replace("sex-specific", "female-specific")
                if dimorphism
                else None
            )

            ftu.info.update_fields(
                rootids, dimorphism=label, id_col="root_783", dry_run=False
            )

        if bodyids is not None and rootids is not None:
            msg = f"Set dimorphism to '{dimorphism}' for {len(bodyids)} maleCNS and {len(rootids)} FlyWire neurons"
        elif bodyids is not None:
            msg = (
                f"Set dimorphism to '{dimorphism}' for {len(bodyids)} male CNS neurons"
            )
        elif rootids is not None:
            msg = f"Set dimorphism to '{dimorphism}' for {len(rootids)} FlyWire neurons"

        print(f"{msg}:")
        if bodyids is not None and len(bodyids):
            print("  ", bodyids)
        if rootids is not None and len(rootids):
            print("  ", rootids)

        if figure:
            # Show the message
            figure.show_message(msg, color="lightgreen", duration=2)
    except BaseException as e:
        if figure:
            figure.show_message(
                "Error pushing dimorphism status (see console)", color="red", duration=2
            )
        traceback.print_exc()
        raise


def _push_super_type(
    super_type,
    bodyids,
    rootids,
    clio,
    ftu,
    sanity_checks=True,
    figure=None,
):
    """Push supertype to Clio/FlyTable."""
    try:
        # Make sure supertype is a string
        super_type = str(super_type)

        if bodyids is not None and len(bodyids):
            clio.set_fields(bodyids, supertype=super_type)

        if rootids is not None and len(rootids):
            ftu.info.update_fields(
                rootids, supertype=super_type, id_col="root_783", dry_run=False
            )

        if bodyids is not None and rootids is not None:
            msg = f"Set super type to '{super_type}' for {len(bodyids)} maleCNS and {len(rootids)} FlyWire neurons"
        elif bodyids is not None:
            msg = (
                f"Set super type to '{super_type}' for {len(bodyids)} male CNS neurons"
            )
        elif rootids is not None:
            msg = f"Set super type to '{super_type}' for {len(rootids)} FlyWire neurons"

        print(f"{msg}:")
        if bodyids is not None and len(bodyids):
            print("  ", bodyids)
        if rootids is not None and len(rootids):
            print("  ", rootids)

        if figure:
            # Show the message
            figure.show_message(msg, color="lightgreen", duration=2)
    except BaseException as e:
        if figure:
            figure.show_message(
                "Error pushing super type (see console)", color="red", duration=2
            )
        traceback.print_exc()
        raise


def _clear_annotations(
    bodyids,
    rootids,
    clio,
    ftu,
    clio_to_clear,
    flytable_to_clear,
    figure=None,
    controls=None,
):
    """Clear the given fields from to Clio/FlyTable."""
    cleared_fields = []
    cleared_ids = []
    try:
        if bodyids is not None and len(bodyids) and clio_to_clear:
            kwargs = {}

            for field in clio_to_clear:
                kwargs[field] = None
                cleared_fields.append(f"`{field}`")

            clio.set_fields(bodyids, **kwargs)
            cleared_ids.append(f"{len(bodyids)} maleCNS")

        if rootids is not None and len(rootids) and flytable_to_clear:
            kwargs = {}

            for field in flytable_to_clear:
                kwargs[field] = None
                cleared_fields.append(f"`{field}`")

                if field in ("cell_type", "hemibrain_type", "malecns_type"):
                    kwargs[f"{field}_source"] = None

            ftu.info.update_fields(
                rootids,
                **kwargs,
                id_col="root_783",
                dry_run=False,
            )
            cleared_fields.append("`malecns_type`")
            cleared_ids.append(f"{len(rootids)} FlyWire")

        msg = f"Cleared {', '.join(cleared_fields)} for {' and '.join(cleared_ids)} neuron(s)"

        print(f"{msg}:")
        if bodyids is not None and len(bodyids) and clio_to_clear:
            print("  ", bodyids)
        if rootids is not None and len(rootids) and flytable_to_clear:
            print("  ", rootids)

        if figure:
            # Update the labels in the figure
            if clio_to_clear and bodyids is not None:
                ind = figure.selected[np.isin(figure.selected_ids, bodyids)]
                figure.set_labels(ind, "(cleared)(!)")
                controls.label_overrides.update({i: "(cleared)(!)" for i in ind})
            if flytable_to_clear and rootids is not None:
                ind = figure.selected[np.isin(figure.selected_ids, rootids)]
                figure.set_labels(ind, "(cleared)(!)")
                controls.label_overrides.update({i: "(cleared)(!)" for i in ind})

            # Show the message
            figure.show_message(msg, color="lightgreen", duration=2)
    except:
        if figure:
            figure.show_message(
                "Error pushing annotations (see console)", color="red", duration=2
            )
        traceback.print_exc()
        raise


def _push_group(
    group,
    bodyids,
    clio,
    figure=None,
):
    """Push group to Clio."""
    try:
        if bodyids is not None:
            clio.set_fields(bodyids, group=group)

        msg = f"Set group {group} for {len(bodyids)} maleCNS neurons"

        print(f"{msg}:")
        print("  ", bodyids)

        if figure:
            # Update the labels in the figure
            figure.set_labels(
                figure.selected[np.isin(figure.selected_ids, bodyids)],
                f"group_{group}(!)",
            )
            # Show the message
            figure.show_message(msg, color="lightgreen", duration=2)
    except:
        if figure:
            figure.show_message(
                "Error pushing annotations (see console)", color="red", duration=2
            )
        traceback.print_exc()
        raise


def suggest_new_label(bodyids, male_only=False):
    """Suggest a new male-only label."""

    # First we need to find the main input neuropil for these neurons
    import neuprint as neu

    global NEUPRINT_CLIENT
    if NEUPRINT_CLIENT is None:
        NEUPRINT_CLIENT = neu.Client("https://neuprint-cns.janelia.org", dataset="cns")

    meta, roi = neu.fetch_neurons(
        neu.NeuronCriteria(bodyId=bodyids), client=NEUPRINT_CLIENT
    )

    # Drop non-primary ROIs
    roi = roi[roi.roi.isin(NEUPRINT_CLIENT.primary_rois)]

    # Remove the hemisphere information
    roi["roi"] = roi.roi.str.replace("(R)", "").str.replace("(L)", "")

    # Find the ROIs that collectively hold > 50% of the neurons input
    roi_in = roi.groupby("roi").post.sum().sort_values(ascending=False)
    roi_in = roi_in / roi_in.sum()

    global HB_ANN
    if HB_ANN is None:
        HB_ANN = pd.read_csv(
            "https://github.com/flyconnectome/flywire_annotations/raw/refs/heads/main/supplemental_files/Supplemental_file5_hemibrain_meta.csv"
        )

    import cocoa as cc

    global CLIO_ANN
    if CLIO_ANN is None:
        print("Fetching Clio annotations...")
        CLIO_ANN = cc.MaleCNS().get_annotations()

    print("Suggested cell type for IDs:", bodyids)
    for roi in roi_in.index.values[:4]:
        if not male_only:
            # Check if we already have hemibrain types for this ROI
            this_hb = np.sort(
                HB_ANN[
                    HB_ANN.type.str.match(f"{roi}\d+", na=False)
                ].morphology_type.unique()
            )

            # Extract the highest hemibrain type ID
            min_id = 0
            if len(this_hb):
                min_id = int(this_hb[-1][len(roi) : len(roi) + 3])

            # Check if we already have male CNS types for this ROI
            this_mcns = CLIO_ANN[
                CLIO_ANN.type.str.match(f"{roi}\d+", na=False)
            ].type.unique()
            this_mcns = np.sort([t for ty in this_mcns for t in ty.split()])
            this_mcns_m = this_mcns[[t.endswith("m") for t in this_mcns]]
            this_mcns_non_m = this_mcns[[not t.endswith("m") for t in this_mcns]]

            if len(this_mcns):
                min_id = max(min_id, int(this_mcns_non_m[-1][len(roi) : len(roi) + 3]))

            new_id = min_id + 1

            # Make sure we're not running into m-types
            if len(this_mcns_m):
                min_id_m = int(this_mcns_m[0][len(roi) : len(roi) + 3])
                max_id_m = int(this_mcns_m[-1][len(roi) : len(roi) + 3])

                if (new_id >= min_id_m) and (new_id <= max_id_m):
                    print(
                        f"Next free ID in roi '{roi}' would be {new_id:03}, but this is already taken by an m-type."
                    )
                    continue

            print(f"{roi}{new_id:03} ({roi_in[roi]:.2%})")
        else:
            # Check if we already have male-specific types for this ROI
            this_mcns = CLIO_ANN[
                CLIO_ANN.type.str.match(f"{roi}\d+m", na=False)
            ].type.unique()

            if len(this_mcns):
                new_id = max([int(t[len(roi) : len(roi) + 3]) for t in this_mcns]) + 1
            else:
                # Check if we already have hemibrain types for this ROI
                this_hb = HB_ANN[
                    HB_ANN.type.str.match(f"{roi}\d+", na=False)
                ].morphology_type.unique()

                if len(this_hb):
                    highest_hb = max([int(t[len(roi) : len(roi) + 1]) for t in this_hb])

                    # Start with the next hundred after the highest hemibrain type
                    new_id = (highest_hb // 100 + 1) * 100
                    if (new_id - highest_hb) < 10:
                        new_id += 100
                else:
                    new_id = 1

            print(f"{roi}{new_id:03}m ({roi_in[roi]:.2%})")


def is_root_id(x):
    """Check if the ID is a root ID (as opposed to a body ID) based on its length."""
    if not isinstance(x, (np.ndarray, tuple, list)):
        x = [x]
    return np.array([len(str(i)) > 15 for i in x])


def sort_ids(ids, meta):
    """Sort given IDs into FlyWire root IDs and male CNS body IDs.

    Parameters
    ----------
    ids :       array-like
                IDs to sort.
    meta :      DataFrame
                Meta data for the neurons. Order should match the IDs.
                This is used to determine whether the IDs are FlyWire root IDs
                or Male CNS body IDs. This requires are `dataset` column
                which, by convention, uses e.g. `Fw` or `FlyWire` + a side suffix
                for FlyWire and `Mcns` or `MaleCNS` + a side suffix for the
                Male CNS.

    Returns
    -------
    rootids :   array-like
                FlyWire root IDs.
    bodyids :   array-like
                Male CNS body IDs.

    """
    ids = np.asarray(ids)

    assert "dataset" in meta.columns, "Meta data must have a 'dataset' column"

    # Process dataset column
    dataset_lower = meta.dataset.fillna("").str.lower()

    # Get FlyWire root IDs
    is_fw_root = dataset_lower.str.startswith("fw") | dataset_lower.str.startswith(
        "flywire"
    )
    rootids = ids[is_fw_root]

    # Get MaleCNS body IDs
    is_mcns = dataset_lower.str.startswith("mcns") | dataset_lower.str.startswith(
        "malecns"
    )
    bodyids = ids[is_mcns]

    return rootids, bodyids
