from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtWidgets import QApplication, QWidget, QMessageBox, QMainWindow, QDialog, QAbstractItemView
from PySide6.QtGui import QColor, QPixmap, QKeySequence, QImage, QBrush
from PySide6.QtWidgets import *
from PySide6.QtCore import QFile
from PySide6.QtUiTools import *
import os
import pyimfit

from utils import *

class CopyParametersDialog(QDialog):
    """Dialog for copying parameters from one band to another."""
    
    def __init__(self, galaxy_path, current_band, fit_type, parent=None):
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
        self.setWindowTitle("Copy Parameters From Band")
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
        
        # Parameter list with checkboxes
        param_label = QLabel("Select parameters to copy:")
        layout.addWidget(param_label)
        
        self.param_list = QListWidget()
        self.param_list.setSelectionMode(QAbstractItemView.SelectionMode.MultiSelection)
        layout.addWidget(self.param_list)
        
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
        """Return config files for the selected band, including component-specific ones."""
        if not os.path.isdir(self.galaxy_path):
            return []

        prefix = f"{self.fit_type}_{band}"
        candidates = []
        for entry in sorted(os.listdir(self.galaxy_path)):
            if not entry.endswith(".dat"):
                continue
            if entry.startswith(prefix):
                candidates.append(entry)

        return candidates or [f"{prefix}.dat"]

    def _populate_config_file_selector(self, band):
        """Populate the source config file combo for the selected band."""
        self.config_file_combo.blockSignals(True)
        self.config_file_combo.clear()
        available_files = self._get_available_config_files(band)
        self.config_file_combo.addItems(available_files)

        default_name = f"{self.fit_type}_{band}.dat"
        if default_name in available_files:
            index = available_files.index(default_name)
        else:
            index = 0
        self.config_file_combo.setCurrentIndex(index)
        self.config_file_combo.blockSignals(False)
        self.on_source_file_changed()

    def on_source_changed(self):
        """Handle source type change."""
        if self.config_radio.isChecked():
            self.source_type = "config"
        else:
            self.source_type = "fit_params"
        self.on_band_changed(self.band_combo.currentText())

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

    def _load_selected_source_config(self):
        """Load the selected source config and populate the parameter list."""
        self.param_list.clear()
        self.fit_params_values = {}

        if not self.source_config_path:
            return

        try:
            self.source_config = pyimfit.parse_config_file(self.source_config_path)
            config_dict = self.source_config.getModelAsDict()
            function_list = config_dict["function_sets"][0]["function_list"]

            # Load function labels
            labels = read_function_labels(self.source_config_path)

            # If fit_params source is selected, try to load fit parameters
            if self.source_type == "fit_params":
                if os.path.exists(self.source_fit_params_path):
                    self.fit_params_values = parse_results(self.source_fit_params_path)[0]
                else:
                    QMessageBox.warning(
                        self, "Warning",
                        f"Fit parameters file not found for {self.source_config_name}.\nFalling back to config file."
                    )
                    self.config_radio.setChecked(True)
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
            QMessageBox.warning(self, "Error", f"Could not load config from {self.source_config_name or band}: {str(e)}")

    def on_band_changed(self, band):
        """Load parameters from the selected source band."""
        self.source_band = band
        self._populate_config_file_selector(band)
    
    def select_all(self):
        """Select all parameter items (exclude headers)."""
        self.param_list.selectAll()
    
    def clear_all(self):
        """Deselect all items."""
        self.param_list.clearSelection()
    
    def get_selected_parameters(self):
        """Return list of selected (func_idx, param_name) tuples."""
        selected = []
        for item in self.param_list.selectedItems():
            data = item.data(QtCore.Qt.UserRole)
            if data is not None:
                selected.append(data)
        return selected
    
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