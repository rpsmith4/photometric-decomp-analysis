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
    
    def on_source_changed(self):
        """Handle source type change."""
        if self.config_radio.isChecked():
            self.source_type = "config"
        else:
            self.source_type = "fit_params"
        self.on_band_changed(self.band_combo.currentText())
    
    def on_band_changed(self, band):
        """Load parameters from the selected source band."""
        self.source_band = band
        self.param_list.clear()
        self.fit_params_values = {}
        
        config_path = os.path.join(self.galaxy_path, f"{self.fit_type}_{band}.dat")
        try:
            self.source_config = pyimfit.parse_config_file(config_path)
            config_dict = self.source_config.getModelAsDict()
            function_list = config_dict["function_sets"][0]["function_list"]
            
            # Load function labels
            labels = read_function_labels(config_path)
            
            # If fit_params source is selected, try to load fit parameters
            if self.source_type == "fit_params":
                fit_params_path = os.path.join(self.galaxy_path, f"{self.fit_type}_{band}_fit_params.txt")
                if os.path.exists(fit_params_path):
                    self.fit_params_values = parse_results(fit_params_path)[0]
                else:
                    QMessageBox.warning(
                        self, "Warning", 
                        f"Fit parameters file not found for band {band}.\nFalling back to config file."
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
            QMessageBox.warning(self, "Error", f"Could not load config from band {band}: {str(e)}")
    
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