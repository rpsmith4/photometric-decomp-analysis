from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtWidgets import QApplication, QWidget, QMessageBox, QMainWindow, QDialog, QAbstractItemView
from PySide6.QtGui import QColor, QPixmap, QKeySequence, QImage, QBrush
from PySide6.QtWidgets import *
from PySide6.QtCore import QFile
from PySide6.QtUiTools import *
import os
import math
import pyimfit

from utils import *

class CopyParametersDialog(QDialog):
    """Dialog for copying parameters between config files in one galaxy directory."""
    
    def __init__(self, galaxy_path, current_band, fit_type, target_config_path=None, parent=None):
        super().__init__(parent)
        self.galaxy_path = galaxy_path
        self.current_band = current_band
        self.fit_type = fit_type
        self.source_band = None
        self.source_config = None
        self.source_config_path = None
        self.source_fit_params_path = None
        self.source_config_name = None
        self.source_type = "config"  # Can be "config" or "fit_params"
        self.fit_params_values = {}  # Store parsed fit parameters
        self.copy_mappings = []
        self.target_function_parameters = {}
        self.target_config_path = target_config_path or os.path.join(
            self.galaxy_path, f"{self.fit_type}_{self.current_band}.dat"
        )
        self.setWindowTitle("Copy Parameters")
        # self.setMinimumWidth(400)
        # self.setMinimumHeight(500)
        
        layout = QVBoxLayout()
        
        # Band selection
        band_layout = QHBoxLayout()
        band_label = QLabel("Copy from band:")
        self.band_combo = QComboBox()
        available_bands = ["g", "r", "i", "z"]
        self.band_combo.addItems(available_bands)
        self.band_combo.currentTextChanged.connect(self.on_band_changed)
        band_layout.addWidget(band_label)
        band_layout.addWidget(self.band_combo)
        band_layout.addStretch()
        layout.addLayout(band_layout)
        
        # Source config file selection
        source_file_layout = QHBoxLayout()
        source_file_label = QLabel("Source config file:")
        self.config_file_combo = QComboBox()
        self.config_file_combo.currentTextChanged.connect(self.on_source_file_changed)
        source_file_layout.addWidget(source_file_label)
        source_file_layout.addWidget(self.config_file_combo)
        source_file_layout.addStretch()
        layout.addLayout(source_file_layout)

        # Source type selection
        source_layout = QHBoxLayout()
        source_label = QLabel("Source:")
        self.config_radio = QRadioButton("Config File")
        self.config_radio.setChecked(True)
        self.config_radio.toggled.connect(self.on_source_changed)
        self.fitparams_radio = QRadioButton("Fit Parameters")
        self.fitparams_radio.toggled.connect(self.on_source_changed)
        source_layout.addWidget(source_label)
        source_layout.addWidget(self.config_radio)
        source_layout.addWidget(self.fitparams_radio)
        source_layout.addStretch()
        layout.addLayout(source_layout)
        
        # Select source parameters and destination functions independently.
        selection_layout = QHBoxLayout()

        source_layout = QVBoxLayout()
        source_layout.addWidget(QLabel("Source parameters:"))
        self.param_list = QListWidget()
        self.param_list.setSelectionMode(QAbstractItemView.SelectionMode.MultiSelection)
        self.param_list.itemSelectionChanged.connect(self._update_compatibility_indicator)
        source_layout.addWidget(self.param_list)
        selection_layout.addLayout(source_layout)

        target_layout = QVBoxLayout()
        target_layout.addWidget(QLabel("Target functions:"))
        self.target_list = QListWidget()
        self.target_list.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.target_list.itemSelectionChanged.connect(self._update_compatibility_indicator)
        target_layout.addWidget(self.target_list)
        selection_layout.addLayout(target_layout)
        layout.addLayout(selection_layout)

        mapping_button_layout = QHBoxLayout()
        add_mapping_btn = QPushButton("Add Selection")
        add_mapping_btn.clicked.connect(self.add_mapping)
        remove_mapping_btn = QPushButton("Remove Mapping")
        remove_mapping_btn.clicked.connect(self.remove_mapping)
        mapping_button_layout.addWidget(add_mapping_btn)
        mapping_button_layout.addWidget(remove_mapping_btn)
        mapping_button_layout.addStretch()
        layout.addLayout(mapping_button_layout)

        self.compatibility_label = QLabel("Select source parameters and a target function.")
        layout.addWidget(self.compatibility_label)

        layout.addWidget(QLabel("Pending mappings:"))
        self.mapping_list = QListWidget()
        self.mapping_list.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        layout.addWidget(self.mapping_list)
        
        # Buttons
        button_layout = QHBoxLayout()
        select_all_btn = QPushButton("Select All")
        select_all_btn.clicked.connect(self.select_all)
        clear_all_btn = QPushButton("Clear All")
        clear_all_btn.clicked.connect(self.clear_all)
        button_layout.addWidget(select_all_btn)
        button_layout.addWidget(clear_all_btn)
        button_layout.addStretch()
        layout.addLayout(button_layout)
        
        # Dialog buttons
        dialog_button_layout = QHBoxLayout()
        copy_btn = QPushButton("Copy Selected")
        cancel_btn = QPushButton("Cancel")
        copy_btn.clicked.connect(self.accept)
        cancel_btn.clicked.connect(self.reject)
        dialog_button_layout.addWidget(copy_btn)
        dialog_button_layout.addWidget(cancel_btn)
        dialog_button_layout.addStretch()
        layout.addLayout(dialog_button_layout)
        
        self.setLayout(layout)
        
        # Load initial band
        self.on_band_changed(self.band_combo.currentText())
    
    def _get_available_config_files(self, band):
        """Return every config file available in the selected galaxy directory."""
        if not os.path.isdir(self.galaxy_path):
            return []

        candidates = []
        for entry in sorted(os.listdir(self.galaxy_path)):
            if entry.endswith(".dat"):
                candidates.append(entry)

        return candidates

    def _populate_config_file_selector(self, band):
        """Populate the source config file combo for the selected band."""
        self.config_file_combo.blockSignals(True)
        self.config_file_combo.clear()
        available_files = self._get_available_config_files(band)
        self.config_file_combo.addItems(available_files)

        default_name = os.path.basename(self.target_config_path)
        if default_name in available_files:
            index = available_files.index(default_name)
        else:
            index = 0
        self.config_file_combo.setCurrentIndex(index)
        self.config_file_combo.blockSignals(False)
        self.on_source_file_changed()

    def on_source_changed(self):
        """Handle source type change."""
        if not self.config_radio.isChecked() and not self.fitparams_radio.isChecked():
            return
        if self.config_radio.isChecked():
            self.source_type = "config"
        else:
            self.source_type = "fit_params"
        self._load_selected_source_config(preserve_mappings=True)

    def on_source_file_changed(self):
        """Update the selected config and fit-params paths when the source file changes."""
        selected_name = self.config_file_combo.currentText()
        if not selected_name:
            return

        self.source_config_name = selected_name
        self.source_config_path = os.path.join(self.galaxy_path, selected_name)
        self.source_fit_params_path = os.path.join(
            self.galaxy_path,
            os.path.splitext(selected_name)[0] + "_fit_params.txt"
        )
        self._load_selected_source_config()

    def _load_selected_source_config(self, preserve_mappings=False):
        """Load the selected source config and populate the parameter list."""
        self.param_list.clear()
        if not preserve_mappings:
            self.target_list.clear()
            self.mapping_list.clear()
            self.copy_mappings.clear()
            self.target_function_parameters.clear()
        self.fit_params_values = {}

        if not self.source_config_path:
            return

        try:
            self.source_config = pyimfit.parse_config_file(self.source_config_path)
            config_dict = self.source_config.getModelAsDict()
            function_list = config_dict["function_sets"][0]["function_list"]

            if not preserve_mappings and os.path.exists(self.target_config_path):
                target_config = pyimfit.parse_config_file(self.target_config_path)
                target_dict = target_config.getModelAsDict()
                target_labels = read_function_labels(self.target_config_path)
                for target_idx, target_func in enumerate(target_dict["function_sets"][0]["function_list"]):
                    label = target_labels[target_idx] if target_idx < len(target_labels) else None
                    parameter_names = list(target_func["parameters"])
                    self.target_function_parameters[target_idx] = set(parameter_names)
                    item = QListWidgetItem(
                        f"{label or f'Function {target_idx}'}: {', '.join(parameter_names)}"
                    )
                    item.setData(QtCore.Qt.UserRole, target_idx)
                    self.target_list.addItem(item)

            # Load function labels
            labels = read_function_labels(self.source_config_path)

            # Load fit parameters so mappings can mix both source types.
            fit_params_available = os.path.exists(self.source_fit_params_path)
            if fit_params_available:
                self.fit_params_values = parse_results(self.source_fit_params_path)[0]
            elif self.source_type == "fit_params":
                QMessageBox.warning(
                    self, "Warning",
                    f"Fit parameters file not found for {self.source_config_name}.\nFalling back to config file."
                )
                self.config_radio.blockSignals(True)
                self.config_radio.setChecked(True)
                self.config_radio.blockSignals(False)
                self.source_type = "config"

            # Populate the list
            for func_idx, func in enumerate(function_list):
                params = func["parameters"]
                label = labels[func_idx] if func_idx < len(labels) else None
                
                # Add header for function
                label_text = f"{label}" if label else f"Function {func_idx}"
                header_item = QListWidgetItem(label_text)
                header_item.setFlags(header_item.flags() & ~QtCore.Qt.ItemFlag.ItemIsSelectable)
                header_font = header_item.font()
                header_font.setBold(True)
                header_item.setFont(header_font)
                self.param_list.addItem(header_item)
                
                # Add parameters
                for param_name in params.keys():
                    # Add source indicator if using fit_params
                    source_indicator = ""
                    highlight_item = False
                    if self.source_type == "fit_params" and func_idx in self.fit_params_values:
                        fit_entry = self.fit_params_values[func_idx]
                        if param_name in fit_entry["parameters"]:
                            param_val = fit_entry["parameters"][param_name]
                            source_indicator = f" (fit: {param_val:.6g})"
                            param_unc = fit_entry["parameters_unc"].get(param_name)
                            if param_unc == 0:
                                param_bounds = params[param_name]
                                if param_bounds[1] == 'fixed':
                                    lowlim = param_bounds[0]
                                    hilim = param_bounds[0]
                                else:
                                    lowlim = param_bounds[1]
                                    hilim = param_bounds[2]
                                if math.isclose(param_val, lowlim, rel_tol=1e-9, abs_tol=1e-12) or math.isclose(param_val, hilim, rel_tol=1e-9, abs_tol=1e-12):
                                    highlight_item = True
                    
                    item_text = f"  └─ {param_name}{source_indicator}"
                    item = QListWidgetItem(item_text)
                    item.setData(QtCore.Qt.UserRole, (func_idx, param_name))
                    if highlight_item:
                        item.setForeground(QtGui.QBrush(QtGui.QColor(255, 0, 0)))
                    self.param_list.addItem(item)
        
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Could not load config from {self.source_config_name}: {str(e)}")

    def on_band_changed(self, band):
        """Reload the source file list."""
        self.source_band = band
        self._populate_config_file_selector(band)
    
    def select_all(self):
        """Select all parameter items (exclude headers)."""
        self.param_list.selectAll()
    
    def clear_all(self):
        """Deselect all items."""
        self.param_list.clearSelection()

    def _update_compatibility_indicator(self):
        """Show whether the current source selection is supported by the target."""
        selected_parameters = self.get_selected_parameters()
        selected_targets = self.get_selected_target_functions()
        if not selected_parameters or not selected_targets:
            self.compatibility_label.setText("Select source parameters and a target function.")
            self.compatibility_label.setStyleSheet("")
            return

        target_idx = selected_targets[0]
        target_parameters = self.target_function_parameters.get(target_idx, set())
        unsupported = sorted({
            param_name
            for _, param_name in selected_parameters
            if param_name not in target_parameters
        })
        if unsupported:
            self.compatibility_label.setText(
                "Warning: target does not support: " + ", ".join(unsupported)
            )
            self.compatibility_label.setStyleSheet("color: #b00020; font-weight: bold;")
        else:
            self.compatibility_label.setText("All selected parameters are supported by the target.")
            self.compatibility_label.setStyleSheet("color: #176b2c;")

    def _source_type_label(self, source_type=None):
        """Return the source type label shown for pending mappings."""
        source_type = self.source_type if source_type is None else source_type
        return "Fit Parameters" if source_type == "fit_params" else "Config File"

    def _target_item_text(self, target_idx):
        """Return the visible target-function label for a mapping."""
        for row in range(self.target_list.count()):
            item = self.target_list.item(row)
            if item.data(QtCore.Qt.UserRole) == target_idx:
                return item.text()
        return f"Function {target_idx}"

    def add_mapping(self):
        """Add the selected source parameters and target function to the pending list."""
        selected_parameters = self.get_selected_parameters()
        selected_targets = self.get_selected_target_functions()
        if not selected_parameters or not selected_targets:
            QMessageBox.information(
                self,
                "Incomplete Selection",
                "Select source parameters and one target function first."
            )
            return

        target_idx = selected_targets[0]
        target_item = self.target_list.selectedItems()[0]
        target_parameters = self.target_function_parameters.get(target_idx, set())
        unsupported_parameters = sorted({
            param_name
            for _, param_name in selected_parameters
            if param_name not in target_parameters
        })
        compatible_parameters = [
            (func_idx, param_name)
            for func_idx, param_name in selected_parameters
            if param_name in target_parameters
        ]
        if not compatible_parameters:
            QMessageBox.warning(
                self,
                "Unsupported Parameters",
                "The selected target function does not support: "
                + ", ".join(unsupported_parameters)
            )
            return

        if unsupported_parameters:
            QMessageBox.warning(
                self,
                "Unsupported Parameters Skipped",
                "These parameters will not be copied to the selected target function: "
                + ", ".join(unsupported_parameters)
            )

        parameter_names = ", ".join(param_name for _, param_name in compatible_parameters)
        mapping = (target_idx, compatible_parameters, self.source_type)
        self.copy_mappings.append(mapping)

        item = QListWidgetItem(
            f"[{self._source_type_label()}] {self._target_item_text(target_idx)} <- {parameter_names}"
        )
        item.setData(QtCore.Qt.UserRole, mapping)
        self.mapping_list.addItem(item)
        self.param_list.clearSelection()
        self.target_list.clearSelection()

    def remove_mapping(self):
        """Remove the selected pending mapping."""
        row = self.mapping_list.currentRow()
        if row < 0:
            return
        self.mapping_list.takeItem(row)
        del self.copy_mappings[row]
    
    def get_selected_parameters(self):
        """Return list of selected (func_idx, param_name) tuples."""
        selected = []
        for item in self.param_list.selectedItems():
            data = item.data(QtCore.Qt.UserRole)
            if data is not None:
                selected.append(data)
        return selected

    def get_selected_target_functions(self):
        """Return target function indices selected by the user."""
        return [
            item.data(QtCore.Qt.UserRole)
            for item in self.target_list.selectedItems()
            if item.data(QtCore.Qt.UserRole) is not None
        ]

    def get_copy_mappings(self):
        """Return pending (target function, source parameter list) mappings."""
        return list(self.copy_mappings)
    
    def get_source_type(self):
        """Return the source type (config or fit_params)."""
        return self.source_type
    
    def get_fit_params_values(self):
        """Return the parsed fit parameters."""
        return self.fit_params_values

    def get_source_config_path(self):
        """Return the selected source config file path."""
        return self.source_config_path

    def get_source_fit_params_path(self):
        """Return the selected source fit-params file path."""
        return self.source_fit_params_path