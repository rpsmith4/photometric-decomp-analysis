from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtWidgets import QApplication, QWidget, QMessageBox, QMainWindow, QDialog, QAbstractItemView
from PySide6.QtGui import QColor, QPixmap, QKeySequence, QImage, QBrush
from PySide6.QtWidgets import *
from PySide6.QtCore import QFile
from PySide6.QtUiTools import *
import os
from pathlib import Path
import subprocess
import sys
import argparse
import shutil
from multiprocessing import Process
import json
import matplotlib.pyplot as plt
from astropy.io import fits
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas, NavigationToolbar2QT as NavigationToolbar
from astropy.visualization.stretch import LogStretch, LinearStretch
from astropy.visualization import ImageNormalize
import math
import pyimfit
import shutil
import numpy as np
import re
import pandas as pd
import matplotlib.patches
import glob
from PIL import Image

BASE_DIR = Path(Path(os.path.dirname(__file__)).parent).resolve()
sys.path.append(str(BASE_DIR))
sys.path.append(os.path.join(BASE_DIR, 'decomposer'))
sys.path.append(os.path.join(BASE_DIR, 'decomposer/manual_fitting'))

from generate_imfit_conf import generate_config
from iman_new.imp.masking.convert_reg_to_mask import mask as convert_reg_to_mask
import test_manual_decomposer
from photometric_cut import photometric_cut, fold_cut_to_radial_profile
from photometric_cut_helpers import pixel_scale_from_header_arcsec_per_pix

def open_folder(path): 
    path = os.path.abspath(path) 
    if sys.platform.startswith('win'): 
        os.startfile(path)                   # Windows Explorer 
    elif sys.platform == 'darwin': 
        subprocess.run(['open', path])      # Finder on macOS 
    else: 
        # Linux: try xdg-open, then sensible-browser as fallback 
        try: 
            subprocess.run(['xdg-open', path], check=True) 
        except Exception: 
            subprocess.run(['gio', 'open', path], check=False) 

LOCAL_DIR = "GUI"
MAINDIR = Path(os.path.dirname(__file__).rpartition(LOCAL_DIR)[0])
sys.path.append(os.path.join(MAINDIR, "decomposer"))
import imfit_run
import fit_monitor

class PlotCanvas(FigureCanvas):
    def __init__(self, parent = None):
        self.fig = Figure(figsize=(250/50, 250/50), dpi=50)
        self.ax = self.fig.subplots()
        self.fig.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=1.0)
        super().__init__(self.fig)
        self.setParent(parent)

    def plot(self, im, limits, cmap, stretch=LogStretch(), ellipse_params=pd.DataFrame, plottext=None):
        self.fig.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=1.0)
        self.ax.cla()
        self.ax.set_axis_off()
        if im.any():
            if limits != None:
                norm = ImageNormalize(stretch=stretch, vmin=limits[0], vmax=limits[1])
                self.ax.imshow(im, origin="lower", norm=norm, cmap=cmap)
            else:
                self.ax.imshow(im, origin="lower", cmap=cmap)

            if plottext != None:
                self.ax.text(0.05, 0.95, plottext, size=15,
                        ha="left", va="center",
                        bbox=dict(boxstyle="square",
                                ec=(1., 1, 1),
                                fc=(0,0,0),
                                ),
                        transform=self.ax.transAxes,
                        color="lightgreen"
                        )


            if not ellipse_params.empty:
                host = ellipse_params[ellipse_params["PolarOrHost"] == "Host"].iloc[0]
                polar = ellipse_params[ellipse_params["PolarOrHost"] == "Polar"].iloc[0]
                imshape = im.shape
                ell_host = matplotlib.patches.Ellipse(
                    xy=(imshape[0]/2, imshape[1]/2),
                    height=float(host["semi_minor"]*2),
                    width=float(host["semi_major"]*2),
                    angle=host["angle"],
                    label="Host",
                    ls="--",
                    lw=2,
                    color="red",
                    fill=False
                )
                ell_polar = matplotlib.patches.Ellipse(
                    xy=(imshape[0]/2, imshape[1]/2),
                    height=float(polar["semi_minor"]*2),
                    width=float(polar["semi_major"]*2),
                    angle=polar["angle"],
                    label="Polar",
                    ls="--",
                    lw=2,
                    color="blue",
                    fill=False
                )
                self.ax.add_patch(ell_host)
                self.ax.add_patch(ell_polar)
                self.ax.legend()
        else:
            self.ax.text(0,0.5,"Cannot find FITs image!")

        self.draw()

    def plot_profiles(self, host_data, polar_data, title, overplot=None, y_label='Surface Brightness (mag/arcsec^2)'):
        self.fig.clear()
        self.ax = self.fig.subplots()

        self.fig.subplots_adjust(left=0.13, right=0.9,bottom=0.1,top=0.9)
        self.ax.cla()
        self.ax.set_title(title, fontsize=12)
        self.ax.plot(host_data['r'], host_data['mu'], label='Host', color='blue')
        self.ax.plot(polar_data['r'], polar_data['mu'], label='Polar', color='red')
        if overplot:
            self.ax.plot(overplot['host']['r'], overplot['host']['mu'], 'b--', label='Host Image')
            self.ax.plot(overplot['polar']['r'], overplot['polar']['mu'], 'r--', label='Polar Image')
        self.ax.legend(fontsize=8)
        self.ax.set_xlabel('Radius (arcsec)', fontsize=10)
        self.ax.set_ylabel(y_label, fontsize=10)
        self.ax.tick_params(labelsize=8)
        self.draw()



class DirOnlyChildrenFileSystemModel(QFileSystemModel):
    def __init__(self, mark_colors=None, galmarks=None, parent=None):
        super().__init__(parent)
        self.mark_colors = mark_colors or {}
        self.galmarks = galmarks

    def hasChildren(self, index):
        # For invalid indices, fall back to default behavior
        if not index.isValid():
            return super().hasChildren(index)
        # Files are never considered to have children
        if not self.isDir(index):
            return False
        path = self.filePath(index)
        try:
            # Only report that this index has children if it contains subdirectories
            with os.scandir(path) as it:
                for entry in it:
                    if entry.is_dir():
                        return True
            return False
        except Exception:
            return super().hasChildren(index)

    def data(self, index, role):
        # Remove folder icon for leaf directories (galaxy folders)
        if role == QtCore.Qt.DecorationRole:
            try:
                # If it's a directory but has no subdirectories (a leaf), don't show an icon
                if self.isDir(index) and not self.hasChildren(index):
                    return None
            except Exception:
                pass

        # Color leaf directories based on galmarks
        if role == QtCore.Qt.BackgroundRole:
            try:
                if self.isDir(index) and not self.hasChildren(index):
                    name = Path(self.filePath(index)).name
                    mark = self.galmarks[name]
                    if mark:
                        col = self.mark_colors[mark]
                        if col:
                            return QBrush(QColor(col))
            except Exception as e:
                pass

        return super().data(index, role)

class ParamSliderWidget(QWidget):
    def __init__(self, paramname, initval, lowlim, hilim, fixed=False, ndigits=3, parent=None):
        super().__init__(parent)
        self.paramname = paramname
        self.ndigits = ndigits
        self.scale = 10 ** ndigits
        self.fixed = fixed

        parameter_adjust_layout = QHBoxLayout()
        parameter_adjust_layout.setContentsMargins(0,0,0,0)

        self.text = QTextBrowser()
        self.text.setText(str(paramname))
        self.text.setStyleSheet('font-size: 10px')
        self.text.setMaximumSize(45,25)
        self.text.setSizePolicy(QtWidgets.QSizePolicy.Policy.Maximum,QtWidgets.QSizePolicy.Policy.Maximum)
        self.text.setAlignment(QtCore.Qt.AlignCenter)

        self.fixed_checkbox = QCheckBox("Fixed")
        self.fixed_checkbox.setChecked(fixed)

        self.slider = QSlider(QtCore.Qt.Orientation.Horizontal)
        self.slider.setSizePolicy(QtWidgets.QSizePolicy.Policy.Minimum,QtWidgets.QSizePolicy.Policy.Maximum)
        self.slider.setRange(int(lowlim * self.scale), int(hilim * self.scale))
        # self.slider.setTickInterval(5)
        self.slider.setSingleStep(1)
        self.slider.setValue(int(initval * self.scale))

        parameter_adjust_layout.addWidget(self.text)
        parameter_adjust_layout.addWidget(self.slider)
        parameter_adjust_layout.setStretchFactor(self.slider, 4)
        parameter_adjust_layout.addWidget(self.fixed_checkbox)

        spinboxes_layout = QHBoxLayout()
        from scientific_spinbox import ScientificDoubleSpinBox
        self.minspinbox = ScientificDoubleSpinBox()
        # self.minspinbox.setDecimals(ndigits)
        spinbox_minwidth = 50
        self.minspinbox.setMaximum(hilim)
        self.minspinbox.setMinimum(-1e9)
        self.minspinbox.setValue(lowlim)
        self.minspinbox.setMaximumWidth(100)
        self.minspinbox.setMinimumWidth(spinbox_minwidth)
        self.minspinbox.setSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Maximum)

        self.valspinbox = ScientificDoubleSpinBox()
        # self.valspinbox.setDecimals(ndigits)
        self.valspinbox.setMaximum(hilim)
        self.valspinbox.setMinimum(lowlim)
        self.valspinbox.setValue(initval)
        self.valspinbox.setMaximumWidth(100)
        self.valspinbox.setMinimumWidth(spinbox_minwidth)
        self.valspinbox.setSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Maximum)

        self.maxspinbox = ScientificDoubleSpinBox()
        # self.maxspinbox.setDecimals(ndigits)
        self.maxspinbox.setSingleStep(1e-2)
        self.maxspinbox.setMinimum(lowlim)
        self.maxspinbox.setMaximum(1e9)
        self.maxspinbox.setValue(hilim)
        self.maxspinbox.setMaximumWidth(100)
        self.maxspinbox.setMinimumWidth(spinbox_minwidth)
        self.maxspinbox.setSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Maximum)
        # self.maxspinbox.setFont("arial: size=20px")

        spinboxes_layout.addWidget(self.minspinbox)
        spinboxes_layout.addWidget(self.valspinbox)
        spinboxes_layout.addWidget(self.maxspinbox)

        spinboxes_layout.setStretchFactor(self.minspinbox, 1)
        spinboxes_layout.setStretchFactor(self.valspinbox, 1)
        spinboxes_layout.setStretchFactor(self.maxspinbox, 1)
        parameter_adjust_layout.addLayout(spinboxes_layout)
        
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        self.converted_label = QLabel()
        self.converted_label.setStyleSheet('font-size: 10px')
        self.converted_label.setMaximumHeight(12)
        layout.addWidget(self.converted_label)
        layout.addLayout(parameter_adjust_layout)
        self.setLayout(layout)

        self.set_fixed_state(fixed)

        self.slider.valueChanged.connect(self.slider_changed)
        self.valspinbox.setKeyboardTracking(False)
        self.valspinbox.valueChanged.connect(self.spinbox_changed)
        self.minspinbox.setKeyboardTracking(False)
        self.minspinbox.valueChanged.connect(self.minspinbox_changed)
        self.maxspinbox.setKeyboardTracking(False)
        self.maxspinbox.valueChanged.connect(self.maxspinbox_changed)
        self.fixed_checkbox.stateChanged.connect(lambda state: self.set_fixed_state(state==2))
        
        spinboxes_layout.setSizeConstraint(QLayout.SizeConstraint.SetMinimumSize)
        parameter_adjust_layout.setStretchFactor(spinboxes_layout, 10)
        
        self.update_converted()

    def set_fixed_state(self, is_fixed):
        self.minspinbox.setEnabled(not is_fixed)
        self.maxspinbox.setEnabled(not is_fixed)
        # self.slider.setEnabled(not is_fixed)
        # self.valspinbox.setEnabled(not is_fixed)

    def slider_changed(self, value):
        float_val = value / self.scale
        self.valspinbox.blockSignals(True)
        self.valspinbox.setValue(float_val)
        self.valspinbox.blockSignals(False)
        self.update_converted()

    def spinbox_changed(self, value):
        int_val = int(round(value * self.scale))
        self.slider.blockSignals(True)
        self.slider.setValue(int_val)
        self.slider.blockSignals(False)
        self.update_converted()

    def minspinbox_changed(self, new_min):
        self.slider.setMinimum(int(new_min * self.scale))
        self.valspinbox.setMinimum(new_min)
        self.maxspinbox.setMinimum(new_min)
        if self.valspinbox.value() < new_min:
            self.valspinbox.setValue(new_min)
        if self.slider.value() < int(new_min * self.scale):
            self.slider.setValue(int(new_min * self.scale))

    def maxspinbox_changed(self, new_max):
        self.slider.setMaximum(int(new_max * self.scale))
        self.valspinbox.setMaximum(new_max)
        self.minspinbox.setMaximum(new_max)
        if self.valspinbox.value() > new_max:
            self.valspinbox.setValue(new_max)
        if self.slider.value() > int(new_max * self.scale):
            self.slider.setValue(int(new_max * self.scale))

    def get_values(self):
        return {
            'value': self.valspinbox.value(),
            'min': self.minspinbox.value(),
            'max': self.maxspinbox.value(),
            'fixed': self.fixed_checkbox.isChecked()
        }

    def update_converted(self):
        val = self.valspinbox.value()
        minval = self.minspinbox.value()
        maxval = self.maxspinbox.value()
        if self.paramname in ["r_e", "R_ring"]:
            valarcsec = val * 0.262
            minvalarcsec = minval * 0.262
            maxvalarcsec = maxval * 0.262
            self.converted_label.setText(f"Min: {minvalarcsec:.3f} Val: {valarcsec:.3f} Max: {maxvalarcsec:.3f}arcsec")
        elif self.paramname in ["I_e", "sigma_r", "A"]:
            if val > 0 and minval > 0 and maxval > 0:
                valmag = 22.5 - 2.5 * math.log10(val/0.262**2)
                minvalmag = 22.5 - 2.5 * math.log10(minval/0.262**2)
                maxvalmag = 22.5 - 2.5 * math.log10(maxval/0.262**2)
                self.converted_label.setText(f"Min: {minvalmag:.3f} Val: {valmag:.3f} Max: {maxvalmag:.3f}arcsec")
            else:
                self.converted_label.setText("N/A")
        else:
            self.converted_label.setText("")

def read_function_labels(config_path):
    """
    Reads function labels from a config file and returns a list of labels in order.
    Each FUNCTION line may have a '# LABEL <label>' comment.
    """
    labels = []
    with open(config_path, 'r') as f:
        for line in f:
            if line.strip().startswith('FUNCTION'):
                m = re.search(r'# LABEL\s*(\S+)', line)
                if m:
                    labels.append(m.group(1))
                else:
                    labels.append(None)
    return labels

def parse_fit_params_file(fit_params_path, config_path):
    """
    Parses an imfit fit_params file and returns a dictionary of parameter values
    organized by function index and parameter name.
    """
    params_dict = {}
    
    try:
        # First, get the config structure to know which parameters exist
        config = pyimfit.parse_config_file(config_path)
        config_dict = config.getModelAsDict()
        function_list = config_dict["function_sets"][0]["function_list"]
        
        # Parse the fit_params file
        with open(fit_params_path, 'r') as f:
            lines = f.readlines()
        
        # Track current function index while parsing
        func_idx = 0
        param_idx = 0
        
        # Parse each line looking for parameter values
        for line in lines:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            # Try to parse as a parameter line (typically format: "param_name = value [error]")
            parts = line.split()
            if len(parts) >= 2:
                try:
                    param_value = float(parts[0])
                    
                    # Get parameter name from config
                    if func_idx < len(function_list):
                        func_params = function_list[func_idx]["parameters"]
                        param_names = list(func_params.keys())
                        
                        if param_idx < len(param_names):
                            param_name = param_names[param_idx]
                            
                            if func_idx not in params_dict:
                                params_dict[func_idx] = {}
                            
                            # Store the value - we'll use it as a fixed parameter
                            params_dict[func_idx][param_name] = param_value
                            param_idx += 1
                            
                            # If we've gone through all params in this function, move to next
                            if param_idx >= len(param_names):
                                func_idx += 1
                                param_idx = 0
                except (ValueError, IndexError) as e:
                    print(e)
                    pass
        
        return params_dict
    except Exception as e:
        print(f"Error parsing fit_params file: {e}")
        return {}

def parse_results(file):
    model = pyimfit.parse_config_file(file)
    band = Path(file).stem.rsplit("_")[-3]
    with open(file, "r") as f:
        lines = f.readlines()
    status = lines[5].split(" ")[7]
    status_message = " ".join(lines[5].split(" ")[9:])
    uncs = dict()
    for k, line in enumerate(lines):
        if "FUNCTION" in line:
            func_type = line.split(" ")[1].rstrip()
            func_label = line.split("LABEL ")[-1].rstrip()
            func_params = pyimfit.get_function_dict()[func_type]
            uncs[func_label] = dict()
            for j, func_param in enumerate(func_params):
                try:
                    unc = lines[k + j + 1].split("+/-")[1].split("\t")[0]
                    uncs[func_label][func_param] = float(unc) # Extremely janky way to get the uncertainties
                except:
                    uncs[func_label][func_param] = None
    chi_sq = float(lines[7].split(" ")[-1])
    chi_sq_red = float(lines[8].split(" ")[-1])
    functions = []
    for k, function in enumerate(model.functionList()):
        func_dict = function.getFunctionAsDict()
        for param in func_dict["parameters"]:
            # func_dict["parameters_unc"][param] = func_dict["parameters"][param] 
            func_dict["parameters"][param] = func_dict["parameters"][param][0]
        if k == 0:
            func_dict["label"] = "Host"
        if k == 1:
            func_dict["label"] = "Polar"
        func_dict["parameters_unc"] = uncs[func_dict["label"]]
        func_dict["band"] = band
        functions.append(func_dict)
    
    return functions, chi_sq, chi_sq_red, status, status_message

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
                    if self.source_type == "fit_params" and func_idx in self.fit_params_values:
                        if param_name in self.fit_params_values[func_idx]:
                            param_val = self.fit_params_values[func_idx][param_name]
                            source_indicator = f" (fit: {param_val:.6g})"
                    
                    item_text = f"  └─ {param_name}{source_indicator}"
                    item = QListWidgetItem(item_text)
                    item.setData(QtCore.Qt.UserRole, (func_idx, param_name))
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

class MainWindow(QMainWindow):
    def __init__(self, p=None, master_table_p = None, ellipse_fit_p=None):
        super().__init__()

        # Loading the config file for the GUI
        with open(os.path.join(MAINDIR, LOCAL_DIR, 'config.json')) as config:
            self.gui_config = json.load(config)
            config.close()
            
        # Read in the ellipse fit data 
        if ellipse_fit_p != None:
            # ellipse_fit_data = pd.DataFrame(columns=["file", "PolarOrHost","IsoLevel", "x_center", "y_center", "semi_major", "semi_minor", "angle"])
            ellipse_fit_data = pd.DataFrame(columns=["file","x_center","y_center","semi_major","semi_minor","angle","MorphType","PSType1","PolarOrHost","IsophoteLevel"])
            try:
                dat = pd.read_csv(Path(ellipse_fit_p), sep = ",")
                ellipse_fit_data = pd.concat([ellipse_fit_data, dat])
            except Exception as e:
                print(e)
            self.ellipse_fit_data = ellipse_fit_data

        # Read in the master table (I might change this later so we don't have to do this in the GUI code)
        if master_table_p != None:
            self.master_table_data = pd.read_csv(master_table_p,header=0)

        # Apply global scaling for the application
        scale_factor = self.gui_config["ui_scale"]
        app = QApplication.instance()
        if app is not None:
            # Set a global style sheet for smaller fonts and widgets
            app.setStyleSheet(f"""
                QWidget {{ font-size: {int(10*scale_factor)}pt; }}
                QAbstractButton, QComboBox, QLineEdit, QTextEdit, QTextBrowser, QSpinBox, QDoubleSpinBox, QSlider {{
                    min-height: {int(22*scale_factor)}px;
                    min-width: {int(22*scale_factor)}px;
                }}
                QTreeView {{ font-size: {int(9*scale_factor)}pt; }}
            """)
        # Optionally, set a fixed window scale
        self.setMinimumWidth(int(800*scale_factor))
        self.setMinimumHeight(int(600*scale_factor))

        ui_file = QFile(os.path.join(MAINDIR, LOCAL_DIR, 'fit_gui.ui'))
        loader = QUiLoader()
        self.ui = loader.load(ui_file)

        # Initializing widget types to make my autocomplete work
        self.currentgalaxytext: QTextBrowser = self.ui.currentgalaxytext
        # self.config: QTextBrowser = self.ui.config
        self.params: QTextBrowser = self.ui.params

        # Initializing some variables
        self.curr_gal_index = 0
        self.solvertype = "LM"
        self.band = "g"
        self.param_widgets = {}

        # Setting up the buttons
        self.ui.LMbutton.clicked.connect(lambda: self.set_solver("LM"))
        self.ui.NMbutton.clicked.connect(lambda: self.set_solver("NM"))
        self.ui.DEbutton.clicked.connect(lambda: self.set_solver("DE"))
        self.ui.gbutton.clicked.connect(lambda: self.set_band("g"))
        self.ui.gbutton.setShortcut(QKeySequence("g"))
        self.ui.rbutton.clicked.connect(lambda: self.set_band("r"))
        self.ui.rbutton.setShortcut(QKeySequence("r"))
        self.ui.ibutton.clicked.connect(lambda: self.set_band("i"))
        self.ui.ibutton.setShortcut(QKeySequence("i"))
        self.ui.zbutton.clicked.connect(lambda: self.set_band("z"))
        self.ui.zbutton.setShortcut(QKeySequence("z"))
        self.ui.markfitted.clicked.connect(lambda: self.markgalaxy("fitted"))
        self.ui.markfitted.setShortcut(QKeySequence("f"))
        self.ui.markreturn.clicked.connect(lambda: self.markgalaxy("return"))
        self.ui.markreturn.setShortcut(QKeySequence("e"))
        self.ui.markunable.clicked.connect(lambda: self.markgalaxy("unable"))
        self.ui.markunable.setShortcut(QKeySequence("u"))
        self.ui.saveconfigbutton.clicked.connect(self.saveconfig)
        self.ui.saveconfigbutton.setShortcut(QKeySequence("CTRL+S"))

        self.ui.opends9button.clicked.connect(self.open_ds9)
        self.ui.opends9button.setShortcut(QKeySequence("O"))
        self.ui.importds9maskbutton.clicked.connect(self.import_ds9_mask)
        self.ui.importds9maskbutton.setShortcut(QKeySequence("CTRL+M"))
        self.ui.refitbutton.clicked.connect(self.refit)
        self.ui.refitbutton.setShortcut(QKeySequence("CTRL+R"))
        self.ui.cancelbutton.clicked.connect(self.cancel)
        self.ui.cancelbutton.setShortcut(QKeySequence("CTRL+C"))

        self.ui.fileexplorerbutton.clicked.connect(self.open_gal_fileexplorer)
        self.ui.fileexplorerbutton.setShortcut(QKeySequence("CTRL+F"))

        self.ui.copyparamsbutton.clicked.connect(self.copy_parameters_from_band)
        self.ui.copyparamsbutton.setShortcut(QKeySequence("CTRL+P"))

        # Also connect the generate config button
        self.ui.newconfbutton.clicked.connect(self.regenconf)
        self.ui.newconfbutton.setShortcut(QKeySequence("CTRL+G"))

        # 1D Fitting button
        self.ui.onedfitbutton.clicked.connect(self.openonedfitdialog)
        self.ui.hostradio.toggled.connect(lambda checked: setattr(self, 'component', 'host') if checked else None)
        self.ui.polarradio.toggled.connect(lambda checked: setattr(self, 'component', 'polar') if checked else None)
        self.component = 'host'

        self.fit_type = self.ui.fit_type_combo.currentText()
        self.ui.fit_type_combo.currentIndexChanged.connect(self.change_fit_type)

        self.host_manual = self.ui.host_manual_radio.isChecked()
        self.polar_manual = self.ui.polar_manual_radio.isChecked()

        self.ui.host_auto_radio.toggled.connect(self.on_fitting_mode_changed)
        self.ui.host_manual_radio.toggled.connect(self.on_fitting_mode_changed)
        self.ui.polar_auto_radio.toggled.connect(self.on_fitting_mode_changed)
        self.ui.polar_manual_radio.toggled.connect(self.on_fitting_mode_changed)

        self.host_button_group = QButtonGroup(self)
        self.host_button_group.addButton(self.ui.host_auto_radio)
        self.host_button_group.addButton(self.ui.host_manual_radio)
        self.polar_button_group = QButtonGroup(self)
        self.polar_button_group.addButton(self.ui.polar_auto_radio)
        self.polar_button_group.addButton(self.ui.polar_manual_radio)

        # Loading the list of galaxies
        self.galaxytree: QTreeView = self.ui.galaxytree
        self.galaxytree.setColumnHidden(1, True)

        # Track the currently selected lowest-level galaxy folder (Path) if any
        self.selected_galaxy_path = None


        # Process list of currently running fits
        self.ps = []
        self.fit_dialogs = []

        # Setting up the FITs plots for the image, model, and residual
        self.jpg_img_plot =  PlotCanvas(parent=self.ui.galaxyjpg)
        self.img = PlotCanvas(parent=self.ui.galimg)
        self.model = PlotCanvas(parent=self.ui.galmodel)
        self.resid = PlotCanvas(parent=self.ui.galresid)
        self.configimg = PlotCanvas(parent=self.ui.configimg)
        self.configresid = PlotCanvas(parent=self.ui.configresid)

        # Add 1D toggle from the UI
        self.toggle_1d = self.ui.toggle1dcheckbox
        self.toggle_1d.stateChanged.connect(self.on_toggle_1d)
        self.is_1d_mode = self.toggle_1d.isChecked()

        # Loading the JSON file for the galaxy marks (whether fitted, need to return to, or can't fit)
        try:
            with open(os.path.join(MAINDIR, LOCAL_DIR, 'galmarks.json')) as f:
                self.galmarks = json.load(f)
        except:
            self.galmarks = {}

        # Ensure mark colors mapping exists
        self.colors = self.gui_config.get("mark_colors", {})

        # Use a QFileSystemModel that only reports directories as having children,
        # hides folder icons for leaf directories, and applies background colors.
        model = DirOnlyChildrenFileSystemModel(mark_colors=self.colors, galmarks=self.galmarks)
        model.setRootPath(str(p))
        self.galaxytree.setModel(model)
        self.galaxytree.setRootIndex(model.index(str(p)))
        self.galaxytree.setColumnHidden(1, True)
        self.galaxytree.setColumnHidden(2, True)
        self.galaxytree.setColumnHidden(3, True)
        # Detect selections on the tree to identify when a leaf directory is selected
        self.galaxytree.selectionModel().selectionChanged.connect(self.on_galaxytree_selection_changed)
        
        self.ui.show()

    def on_toggle_1d(self, state):
        self.is_1d_mode =  state == 2
        self.plot_image()
        self.plot_model()
        self.plot_residual()
        self.plot_config()
        self.plot_config_residual()

    def get_suffix(self):
        if self.host_manual and self.polar_manual:
            return "_all_manual"
        elif self.host_manual:
            return "_host_manual"
        elif self.polar_manual:
            return "_polar_manual"
        else:
            return ""

    def get_config_path(self, galaxypath, band, fit_type):
        suffix = self.get_suffix()
        return os.path.join(galaxypath, f"{fit_type}_{band}{suffix}.dat")

    def on_fitting_mode_changed(self):
        self.host_manual = self.ui.host_manual_radio.isChecked()
        self.polar_manual = self.ui.polar_manual_radio.isChecked()
        self.changegal()

    def open_gal_fileexplorer(self):
        open_folder(self.selected_galaxy_path)

    def change_fit_type(self):
        self.fit_type = self.ui.fit_type_combo.currentText()
        self.changegal()

    def open_ds9(self, *args):
        p = self.selected_galaxy_path
        if p is None:
            QMessageBox.warning(self, "No Galaxy Selected", "Please select a galaxy first.")
            return
        files = [f"{os.path.join(p, f'{self.fit_type}_{self.band}_composed.fits')}"]
        ds9_cmd = self.gui_config.get("ds9_path", "ds9") or "ds9"
        arg = [ds9_cmd, "-cmap", self.gui_config["ds9_cmap"], "-scale", self.gui_config["ds9_scale"], "-scale", "limits", f"{self.gui_config['ds9_limits'][0]}", f"{self.gui_config['ds9_limits'][1]}", "-cube", "3"]
        arg.extend(files)
        subprocess.Popen(arg)

    def import_ds9_mask(self):
        if self.selected_galaxy_path is None:
            QMessageBox.warning(self, "No Galaxy Selected", "Please select a galaxy first.")
            return

        galpath = self.selected_galaxy_path
        reg_file, _ = QFileDialog.getOpenFileName(
            self,
            "Select DS9 region file",
            str(galpath),
            "DS9 Region Files (*.reg);;All Files (*)"
        )
        if not reg_file:
            return

        image_file = os.path.join(galpath, f"image_{self.band}.fits")
        if not os.path.isfile(image_file):
            QMessageBox.critical(self, "Missing Image", f"Could not find image file: {image_file}")
            return

        new_mask_file = os.path.join(galpath, f"image_{self.band}_ds9_mask.fits")
        try:
            _, output_mask, mask_data = convert_reg_to_mask(
                image_file,
                reg_file,
                output_mask=new_mask_file,
                mask_value=1,
                verbosity=False
            )
            if output_mask is None:
                raise RuntimeError("Mask conversion did not produce an output mask file")

            old_mask_path = os.path.join(galpath, "image_mask.fits")
            if os.path.exists(old_mask_path):
                with fits.open(old_mask_path) as hdul:
                    old_mask = hdul[0].data
                    header = hdul[0].header
            else:
                header = fits.getheader(image_file)
                old_mask = np.zeros_like(mask_data, dtype=np.uint8)

            if old_mask.shape != mask_data.shape:
                raise ValueError("Existing mask and DS9 mask have different shapes. Please verify the image and region file.")

            combined_mask = np.where((old_mask > 0) | (mask_data > 0), 1, 0).astype(old_mask.dtype)
            try:
                fits.writeto(os.path.join(galpath, "image_mask.fits.bak"), old_mask, header=header, overwrite=False) #Keep a backup
            except:
                pass
            fits.writeto(old_mask_path, combined_mask, header=header, overwrite=True)

            QMessageBox.information(
                self,
                "DS9 Mask Imported",
                f"Imported {os.path.basename(reg_file)} and merged it into image_mask.fits. (Backup is available in the file explorer)"
            )
            self.changegal()
        except Exception as e:
            QMessageBox.critical(self, "Mask Import Failed", f"Failed to import DS9 mask:\n{e}")
            print(e)
            return
    
    def get_composed_data(self, galaxy_path, band, idx, fit_type):
        # idx = 0 -> Regular image with mask applied, 1 -> Model image, 2 -> Residual, 3 -> Percent residual, 4 Onwards -> Components of model
        try:
            if idx == 0:
                im = fits.getdata(os.path.join(galaxy_path, f"image_{band}.fits"))
                mask_path = os.path.join(galaxy_path, "image_mask.fits")
                if os.path.exists(mask_path):
                    mask = fits.getdata(mask_path)
                    if mask.shape == im.shape:
                        im = np.where(mask > 0, 0, im)
                return im
            im = fits.getdata(os.path.join(galaxy_path, f"{fit_type}_{band}_composed.fits"))[idx]
        except:
            if idx != 0:
                im = np.array([])
            else:
                try:
                    im = fits.getdata(os.path.join(galaxy_path, f"image_{band}.fits"))
                except:
                    return np.array([])
        return im

    def getconfigim(self, galaxypath, config_path, shape, maxThreads=4):
        try:
            model_desc = pyimfit.parse_config_file(config_path)
            psf = fits.getdata(os.path.join(galaxypath, f"psf_patched_{self.band}.fits"))
            imfitter = pyimfit.Imfit(model_desc, psf=psf, maxThreads=maxThreads)
            # imfitter = pyimfit.Imfit(model_desc, maxThreads=maxThreads)
            im = imfitter.getModelImage(shape=shape)
        except:
            im = np.array([])

        return im
    
    def getconfigresid(self, im, imconfig, mask=np.array([])):
        try:
            if mask.size != 0:
                return np.where(mask >0, 0, im - imconfig)
            else:
                return im - imconfig
        except:
            return np.array([])


    def changegal(self):
        # Store the current config model for later editing
        self.current_config_model = None
        # Update UI based on the currently selected leaf galaxy folder
        galaxypath = self.selected_galaxy_path
        galaxy = galaxypath.name
        self.currentgalaxytext.setText(f"Current Galaxy: {galaxy}")
        self.currentgalaxytext.repaint()

        try:
            config_path = self.get_config_path(galaxypath, self.band, self.fit_type)
            config_model = pyimfit.parse_config_file(config_path)
            self.current_config_model = config_model
            config_dict = config_model.getModelAsDict()
            # Add function labels to config_dict
            labels = read_function_labels(config_path)
            function_list = config_dict["function_sets"][0]["function_list"]
            for i, func in enumerate(function_list):
                if i < len(labels):
                    func['label'] = labels[i]
                else:
                    func['label'] = None
            
            # Want to ensure that I actually keep the labels since pyimift is incapable of doing so for some reason
            self.current_config_dict = config_dict   

            layout: QVBoxLayout = self.ui.configsliders
            # Reset the layout first
            try:
                self.clearLayout(layout)
            except Exception as e:
                print(e)
                pass
            for func_idx, func in enumerate(function_list):
                params = func["parameters"]
                label = func["label"]

                label_text = QTextBrowser()
                label_text.setText(label)
                label_text.setMaximumHeight(30)
                label_text.setMinimumWidth(50)
                label_text.setSizePolicy(QtWidgets.QSizePolicy.Policy.Minimum, QtWidgets.QSizePolicy.Policy.Minimum)
                label_text.setAlignment(QtCore.Qt.AlignCenter)
                layout.addWidget(label_text)

                for param in params.keys():
                    initval = params[param][0]
                    fixed = False
                    if params[param][1] == 'fixed':
                        lowlim = initval
                        hilim = initval
                        fixed = True
                    else:
                        lowlim = params[param][1]
                        hilim = params[param][2]

                    # Use (func_idx, param) as key to distinguish duplicate param names
                    self.draw_params(initval, lowlim, hilim, fixed, (func_idx, param), label, layout)
        except:
            self.clearLayout(self.ui.configsliders)
            pass


        try:
            with open(os.path.join(galaxypath, f"{self.fit_type}_{self.band}_fit_params.txt"), "r") as f:
                params_file = f.readlines()
            self.params.setPlainText("".join(params_file))
            self.params.repaint()
        except:
            self.params.setPlainText("Fit Params not found!")
            self.params.repaint()

        # pixmap = QPixmap(os.path.join(galaxypath, "image.jpg"))
        self.jpg_img = np.asarray(Image.open(os.path.join(galaxypath, "image.jpg")))
        self.jpg_img = np.flipud(self.jpg_img)
        self.jpg_img_plot.plot(self.jpg_img, limits=None, cmap=None, stretch=None, plottext="JPG Image")

        # self.img.get_composed_data(galaxypath, self.band, idx=0, fit_type=self.fit_type)
        self.sci_im = self.get_composed_data(galaxypath, self.band, idx=0, fit_type=self.fit_type)
        self.sci_fits = fits.open(os.path.join(galaxypath, f"image_{self.band}.fits"))[0]
        self.pixel_scale = pixel_scale_from_header_arcsec_per_pix(self.sci_fits)
        self.mask_fits = fits.open(os.path.join(galaxypath, "image_mask.fits"))[0] if os.path.exists(os.path.join(galaxypath, "image_mask.fits")) else None
        self.invvar_fits = fits.open(os.path.join(galaxypath, f"image_{self.band}_invvar.fits"))[0] if os.path.exists(os.path.join(galaxypath, f"image_{self.band}_invvar.fits")) else None
        self.psf_fits = fits.open(os.path.join(galaxypath, f"psf_patched_{self.band}.fits"))[0] if os.path.exists(os.path.join(galaxypath, f"psf_patched_{self.band}.fits")) else None

        self.model_im = self.get_composed_data(galaxypath, self.band, idx=1, fit_type=self.fit_type)
        self.residual_im = self.get_composed_data(galaxypath, self.band, idx=2, fit_type=self.fit_type)
        
        imconfig = self.getconfigim(galaxypath, config_path, np.shape(self.sci_im), maxThreads=self.gui_config["imfit_maxthreads"])
        self.config_im = imconfig
        self.config_residual_im = self.getconfigresid(self.sci_im, self.config_im, self.mask_fits.data)

        galname = self.selected_galaxy_path.name
        if ellipse_fit_p != None:
            ellipse_params = self.ellipse_fit_data[self.ellipse_fit_data["file"] == galname]
        else:
            ellipse_params = pd.DataFrame()
        self.current_ellipse_params = ellipse_params
        self.toggle_1d.setEnabled(not ellipse_params.empty)

        self.plot_image()
        self.plot_model()
        self.plot_residual()
        self.plot_config()
        self.plot_config_residual()

    def clearLayout(self, layout):
        if isinstance(layout, QLayout):
            while layout.count():
                item = layout.takeAt(0)
                widget = item.widget()
                if widget is not None:
                    widget.deleteLater()
                    # layout.removeWidget(widget)
                else:
                    self.clearLayout(item.layout())
                    # layout.removeItem(item)
 
    def plot_image(self):
        self.img.plot(self.sci_im, limits=self.gui_config["plot_limits"], cmap=self.gui_config["plot_cmap"], ellipse_params=self.current_ellipse_params, plottext=f"{self.band} Band Image")

    def plot_model(self):
        if self.is_1d_mode:
            data = self.get_radial_data(self.band)
            if data is not None:
                self.model.plot_profiles(data['model']['host'], data['model']['polar'], 'Model Radial Profile', overplot=data['image'])
                return
        self.model.plot(self.model_im, limits=self.gui_config["plot_limits"], cmap=self.gui_config["plot_cmap"], plottext=f"2D Model Image")

    def plot_residual(self):
        if self.is_1d_mode:
            data = self.get_radial_data(self.band)
            if data is not None:
                self.resid.plot_profiles(
                    data['residual']['host'],
                    data['residual']['polar'],
                    'Residual Radial Profile',
                    y_label='Residual Flux (nanomaggies/arcsec^2)'
                )
                return
        self.resid.plot(self.residual_im, limits=self.gui_config["plot_resid_limits"], cmap=self.gui_config["plot_resid_cmap"], stretch=LinearStretch(), plottext=f"Image - Model") # Update to show relative residual

    def plot_config(self):
        if self.is_1d_mode:
            data = self.get_radial_data(self.band)
            if data is not None:
                self.configimg.plot_profiles(data['config']['host'], data['config']['polar'], 'Config Radial Profile', overplot=data['image'])
                return
        self.configimg.plot(self.config_im, limits=self.gui_config["plot_limits"], cmap=self.gui_config["plot_cmap"], plottext="2D Config Image")

    def plot_config_residual(self):
        if self.is_1d_mode:
            data = self.get_radial_data(self.band)
            if data is not None:
                self.configresid.plot_profiles(
                    data['config_residual']['host'],
                    data['config_residual']['polar'],
                    'Config Residual Radial Profile',
                    y_label='Config Residual Flux (nanomaggies/arcsec^2)'
                )
                return
        self.configresid.plot(self.config_residual_im, limits=self.gui_config["plot_resid_limits"], cmap=self.gui_config["plot_resid_cmap"], stretch=LinearStretch(), plottext="Image - Config")

    def get_radial_data(self, band):
        if self.current_ellipse_params is None or self.current_ellipse_params.empty:
            return None
        host_e = self.current_ellipse_params[self.current_ellipse_params["PolarOrHost"] == "Host"]
        polar_e = self.current_ellipse_params[self.current_ellipse_params["PolarOrHost"] == "Polar"]
        if host_e.empty or polar_e.empty:
            return None
        host_pa = host_e["angle"].iloc[0]
        polar_pa = polar_e["angle"].iloc[0]
        host_a = host_e["semi_major"].iloc[0]
        polar_a = polar_e["semi_major"].iloc[0]
        zeropoint = 22.5
        center = (float(self.sci_fits.header['CRPIX1']) - 1.0, float(self.sci_fits.header['CRPIX2']) - 1.0)
        host_len = int(max(200, 2.5 * host_a))
        polar_len = int(max(250, 2.5 * polar_a))
        width_pix = 7.0
        oversample = 2

        mask_array = self.mask_fits.data if getattr(self.mask_fits, 'data', None) is not None else self.mask_fits
        invvar_array = self.invvar_fits.data if getattr(self.invvar_fits, 'data', None) is not None else self.invvar_fits
        psf_array = self.psf_fits.data if getattr(self.psf_fits, 'data', None) is not None else self.psf_fits

        def profile_from_image(image_data, pa, length, quantity='mu'):
            if hasattr(image_data, 'data'):
                sci_input = image_data
            else:
                sci_input = type('FITSLike', (), {'data': image_data})()
            cut = photometric_cut(
                sci_fits=sci_input,
                center_xy=center,
                pa_deg=pa,
                length_pix=length,
                width_pix=width_pix,
                oversample=oversample,
                mask_fits=mask_array,
                invvar_fits=invvar_array,
                psf_fits=psf_array,
                zeropoint=zeropoint,
                pixel_scale_arcsec=self.pixel_scale,
            )
            r, values, values_err, _ = fold_cut_to_radial_profile(cut, quantity=quantity)
            return {'r': r, 'mu': values, 'mu_err': values_err}

        image_host = profile_from_image(self.sci_fits, host_pa, host_len)
        image_polar = profile_from_image(self.sci_fits, polar_pa, polar_len)

        model_host = profile_from_image(self.model_im, host_pa, host_len)
        model_polar = profile_from_image(self.model_im, polar_pa, polar_len)

        residual_data = self.sci_fits.data - self.model_im
        residual_host = profile_from_image(residual_data, host_pa, host_len, quantity='I')
        residual_polar = profile_from_image(residual_data, polar_pa, polar_len, quantity='I')

        config_host = profile_from_image(self.config_im, host_pa, host_len)
        config_polar = profile_from_image(self.config_im, polar_pa, polar_len)

        config_residual_data = self.sci_fits.data - self.config_im
        config_residual_host = profile_from_image(config_residual_data, host_pa, host_len, quantity='I')
        config_residual_polar = profile_from_image(config_residual_data, polar_pa, polar_len, quantity='I')

        return {
            'image': {'host': image_host, 'polar': image_polar},
            'model': {'host': model_host, 'polar': model_polar},
            'residual': {'host': residual_host, 'polar': residual_polar},
            'config': {'host': config_host, 'polar': config_polar},
            'config_residual': {'host': config_residual_host, 'polar': config_residual_polar},
        }
 
    def draw_params(self, initval, lowlim, hilim, fixed, paramkey, label, layout):
        func_idx, paramname = paramkey
        ndigits = 6
        widget = ParamSliderWidget(paramname, initval, lowlim, hilim, fixed=fixed, ndigits=ndigits)
        # widget.setSizePolicy(QtWidgets.QSizePolicy.Policy.Maximum, QtWidgets.QSizePolicy.Policy.Maximum)
        widget.setMinimumWidth(100)
        layout.addWidget(widget)
        self.param_widgets[paramkey] = widget
        
    def on_galaxytree_selection_changed(self, selected, deselected):
        """Called when the tree selection changes. If the selected item is a lowest-level
        directory (a directory with no subdirectories) we store it as the current galaxy
        and update the UI."""
        idxs = selected.indexes()
        if not idxs:
            return
        idx = idxs[0]
        model = self.galaxytree.model()
        # Only consider directory selections
        try:
            if not model.isDir(idx):
                return
        except Exception:
            return

        is_leaf = not model.hasChildren(idx)
        path = Path(model.filePath(idx))
        if is_leaf:
            self.selected_galaxy_path = path
            self.changegal()
        else:
            # Clear selection for non-leaf directories
            self.selected_galaxy_path = None

    def set_solver(self, solver):
        self.solvertype = solver 

    def set_band(self, band):
        self.band = band
        self.changegal()

    def refit(self):
        # Open a Fit Monitor dialog which runs IMFIT and streams stdout
        if getattr(self, "selected_galaxy_path", None) is None:
            print("No galaxy selected to refit")
            return
        path = self.selected_galaxy_path
        config_path = self.get_config_path(path, self.band, self.fit_type)
        dlg = fit_monitor.FitMonitorDialog(path, self.band, self.solvertype, max_threads=self.gui_config["imfit_maxthreads"], fit_type=self.fit_type, config_file=config_path, gui_config=self.gui_config, parent=self)
        dlg.show()
        self.fit_dialogs.append(dlg)

        # Just refreshing the configs and stats and whatnot
        self.changegal()
    
    def cancel(self):
        # Try to cancel dialog-based fits first
        if len(self.fit_dialogs) > 0:
            dlg = self.fit_dialogs[-1]
            try:
                dlg.cancel()
                dlg.close()
            except Exception:
                pass
            self.fit_dialogs.pop()
            return

        if len(self.ps) > 0:
            self.ps[-1].terminate()
            self.ps.pop()
        else:
            print("No running IMFIT processes")
    
    def markgalaxy(self, markas):
        if getattr(self, "selected_galaxy_path", None) is None:
            return
        galname = self.selected_galaxy_path.name
        # Save mark state
        self.galmarks[galname] = markas
        with open(os.path.join(MAINDIR, LOCAL_DIR, 'galmarks.json'), 'w') as fp:
            json.dump(self.galmarks, fp)
        # Refresh the view so marked colors are updated
        try:
            model = self.galaxytree.model()
            model.layoutChanged.emit()
        except Exception as e:
            print(e)
            pass
    
    def saveconfig(self):
        if self.selected_galaxy_path is None:
            print("No galaxy selected to save config")
            return
        p = self.selected_galaxy_path
        fit_type = self.ui.fit_type_combo.currentText()
        config_path = self.get_config_path(p, self.band, fit_type)

        # If we have a config model and param_widgets, update the config model with the new values
        model_dict = self.current_config_dict
        function_list = model_dict["function_sets"][0]["function_list"]
        # Update parameter values from widgets
        for func_idx, func in enumerate(function_list):
            params = func["parameters"]
            for param in params.keys():
                key = (func_idx, param)
                if key in self.param_widgets:
                    values = self.param_widgets[key].get_values()
                    if values['fixed']:
                        # Only value and 'fixed' string
                        params[param] = [values['value'], 'fixed']
                    else:
                        # Value, min, max
                        params[param] = [values['value'], values['min'], values['max']]
        
        # Rebuild the config description from the updated dict
        new_model = pyimfit.ModelDescription.dict_to_ModelDescription(model_dict)
        config_text = "".join(new_model.getStringDescription())
        # Backup old config
        if os.path.isfile(config_path):
            shutil.copyfile(src=config_path, dst=config_path + ".bak")
        with open(config_path, "w") as f:
            f.write(config_text)
        if f: f.close()

        # Refresh config image and residual
        self.changegal()
    
    def copy_parameters_from_band(self):
        """Open dialog to copy parameters from another band to current band."""
        if self.selected_galaxy_path is None:
            QMessageBox.warning(self, "No Galaxy Selected", "Please select a galaxy first.")
            return
        
        # Open the copy parameters dialog
        dlg = CopyParametersDialog(
            self.selected_galaxy_path,
            self.band,
            self.fit_type,
            parent=self
        )
        
        if dlg.exec() == QDialog.DialogCode.Accepted:
            source_band = dlg.source_band
            selected_params = dlg.get_selected_parameters()
            source_type = dlg.get_source_type()
            fit_params_values = dlg.get_fit_params_values()
            
            if not selected_params:
                QMessageBox.information(self, "No Parameters", "No parameters selected to copy.")
                return
            
            try:
                # Get source config
                source_config_path = os.path.join(
                    self.selected_galaxy_path,
                    f"{self.fit_type}_{source_band}.dat"
                )
                source_config = pyimfit.parse_config_file(source_config_path)
                source_dict = source_config.getModelAsDict()
                source_functions = source_dict["function_sets"][0]["function_list"]
                source_functions_labels = read_function_labels(source_config_path)

                # Not super necessary to get these labels, but perhaps I may
                # Want to check that both the source and current have the same labels
                for i, func in enumerate(source_functions):
                    if i < len(source_functions_labels):
                        func['label'] = source_functions_labels[i]
                    else:
                        func['label'] = None

                # Get current config
                current_config_path = os.path.join(
                    self.selected_galaxy_path,
                    f"{self.fit_type}_{self.band}.dat"
                )
                current_config = pyimfit.parse_config_file(current_config_path)
                current_dict = current_config.getModelAsDict()
                current_functions = current_dict["function_sets"][0]["function_list"]
                current_functions_labels = read_function_labels(current_config_path)

                for i, func in enumerate(current_functions):
                    if i < len(current_functions_labels):
                        func['label'] = current_functions_labels[i]
                    else:
                        func['label'] = None
                
                if current_functions_labels != source_functions_labels:
                    raise ValueError(f"Bad function labels!\nSource has labels: {source_functions_labels}\nCurrent has labels: {current_functions_labels}")
            
                # Copy selected parameters
                copied_count = 0
                for func_idx, param_name in selected_params:
                    try:
                        if source_type == "fit_params":
                            # Copy from fit parameters - use value as fixed parameter
                            param_value = fit_params_values[func_idx]["parameters"][param_name]
                            param_unc = fit_params_values[func_idx]["parameters_unc"][param_name]
                            # Ngl I don't know how I feel about determining the bounds like this, but it works for now (subject to change)
                            if param_unc != 0:
                                current_functions[func_idx]["parameters"][param_name] = [param_value, param_value-param_unc, param_value+param_unc]
                            else:
                                current_functions[func_idx]["parameters"][param_name] = [param_value, 'fixed']
                            copied_count += 1
                        else:
                            # Copy from config file
                            source_param = source_functions[func_idx]["parameters"][param_name]
                            if source_param is not None:
                                # Copy the parameter value and constraints
                                current_functions[func_idx]["parameters"][param_name] = source_param.copy()
                                copied_count += 1
                    except Exception as e:
                        print(f"Warning: Could not copy {param_name} from function {func_idx}: {e}")
                
                if copied_count == 0:
                    QMessageBox.warning(self, "Copy Failed", "No parameters could be copied. Function count mismatch?")
                    return
                
                # Save updated config
                new_model = pyimfit.ModelDescription.dict_to_ModelDescription(current_dict)
                config_text = "".join(new_model.getStringDescription())
                
                # Backup old config
                if os.path.isfile(current_config_path):
                    shutil.copyfile(src=current_config_path, dst=current_config_path + ".bak")
                
                with open(current_config_path, "w") as f:
                    f.write(config_text)
                
                source_text = "config file" if source_type == "config" else "fit parameters"
                QMessageBox.information(self, "Success", f"Copied {copied_count} parameter(s) from {source_text} of band {source_band}.")
                
                # Refresh the UI with new parameters
                self.changegal()
                
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to copy parameters: {str(e)}")
    
    def regenconf(self):
        ### TODO: Make this a bit more general, ideally shouldn't need to load the data here, but could instead load it in the config generation script itself maybe
        galpath = self.selected_galaxy_path
        img_file = glob.glob(os.path.join(galpath, f"image_{self.band}.fits"))[0]

        img = fits.open(img_file)[0]
        mask = fits.getdata(os.path.join(galpath, "image_mask.fits"))
            
        psf = fits.getdata(os.path.join(galpath, f"psf_patched_{self.band}.fits"))
        invvar = fits.getdata(os.path.join(galpath, f"image_{self.band}_invvar.fits"))
        outfile_name = f"{self.fit_type}_{self.band}.dat" 
        outfile = os.path.join(galpath, outfile_name)
        galname = self.selected_galaxy_path.name
        ellipse_fit_data_gal = self.ellipse_fit_data[self.ellipse_fit_data["file"] == galname]
        if ellipse_fit_data_gal.empty:
            QMessageBox.critical(self, "Failed to Generate Config", f"Galaxy {galname} not in Ellipse Fit Data!")
            return
        model_desc_dict = {} # Not really needed anymore I think
        master_table_data_gal = self.master_table_data[self.master_table_data["NAME"] == galname]


        files = os.listdir(galpath)
        try:
            if f"{self.fit_type}_{self.band}.dat" in files:
                answer = QMessageBox.question(
                    self,
                    'Overwrite Warning',
                    'There is an existing config file. Overwrite?',
                    QMessageBox.StandardButton.Yes |
                    QMessageBox.StandardButton.No
                )
                if answer == QMessageBox.StandardButton.Yes:
                    generate_config(
                        galpath,
                        self.band,
                        img,
                        mask,
                        psf,
                        invvar,
                        type="ring",
                        ellipse_fit_data=ellipse_fit_data_gal,
                        model_desc_dict=model_desc_dict,
                        galaxy_type=master_table_data_gal,
                        plot_slits=self.gui_config["show_phot_slits"],
                        outfile_name=outfile,
                        fit_type=self.fit_type,
                    )
                    QMessageBox.information(self, "Config Generation Information", "Config successfully written")
                else:
                    pass
            else:
                print(self.gui_config["show_phot_slits"])
                generate_config(
                    galpath,
                    self.band,
                    img,
                    mask,
                    psf,
                    invvar,
                    type="ring",
                    ellipse_fit_data=ellipse_fit_data_gal,
                    model_desc_dict=model_desc_dict,
                    galaxy_type=master_table_data_gal,
                    plot_slits=self.gui_config["show_phot_slits"],
                    outfile_name=outfile,
                    fit_type=self.fit_type,
                )
                QMessageBox.information(self, "Config Generation Information", "Config successfully written")
        except Exception as e:
            QMessageBox.critical(self, "Config Generation Information", f"Config generation failed:\n{e}")
            print(e)

        
        self.changegal()
    
    # This currently only allows for manual refitting of the host. For the polar component, just swap out line 1160 'host' with 'polar'. I don't know if you want separate buttons to do that or a toggle, but either way I'm not 100% sure how that would exactly that would need to be included.
    def openonedfitdialog(self):
        if self.selected_galaxy_path is None:
            QMessageBox.warning(self, "No galaxy selected", "Please select a galaxy first.")
            return
        
        galpath = self.selected_galaxy_path
        manual_decomp_path = os.path.join(MAINDIR, "decomposer", "manual_fitting", "test_manual_decomposer.py")
        mask_path = os.path.join(galpath, "image_mask.fits")
        galname = self.selected_galaxy_path.name
        self.component = 'host' if self.ui.hostradio.isChecked() else 'polar'
        try:
            ellipse_fit_data_gal = self.ellipse_fit_data[self.ellipse_fit_data["file"] == galname]
            if ellipse_fit_data_gal.empty:
                QMessageBox.critical(self, "Failed to open 1D fit", f"Galaxy {galname} not in Ellipse Fit Data!")
            else:
                if self.component == 'host':
                    ellipse_fit_data_gal = ellipse_fit_data_gal[ellipse_fit_data_gal["PolarOrHost"] == 'Host']
                elif self.component == 'polar':
                    ellipse_fit_data_gal = ellipse_fit_data_gal[ellipse_fit_data_gal["PolarOrHost"] == 'Polar']
                self.ell = ((ellipse_fit_data_gal["semi_major"] - ellipse_fit_data_gal["semi_minor"])/ellipse_fit_data_gal["semi_major"]).iloc[0]
                # print(self.ell)
                self.pa = ellipse_fit_data_gal["angle"].iloc[0]
                # print(self.pa)
                subprocess.call([sys.executable, manual_decomp_path, "-p", str(self.selected_galaxy_path), "-b", self.band, "-c", self.component, 
                                 "-pa", str(self.pa), "-ell", str(self.ell), 
                                 "-ellipse_path", ellipse_fit_p,
                                 "-m", mask_path,
                                 "-master_table", master_table_p])
                self.changegal()
        except Exception as e:
            QMessageBox.critical(self, "Failed to open 1D fit", f"Could not launch 1D fit: {e}")
        
        


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Open IMFIT helper GUI"
    )
    
    parser.add_argument("-p", help="Path to folder containing galaxies", default=".")
    parser.add_argument("--ellipse_fit", help="Path to ellipse fit data", default=None)
    parser.add_argument("--master_table", help="Path to master table data", default=None)

    args = parser.parse_args()
    p = Path(args.p).resolve()
    if args.ellipse_fit != None:
        ellipse_fit_p = Path(args.ellipse_fit).resolve()
    else:
        ellipse_fit_p = None
    if args.master_table != None:
        master_table_p = Path(args.master_table).resolve()
    else:
        master_table_p = None
    app = QApplication(sys.argv)
    main_win = MainWindow(p,master_table_p, ellipse_fit_p)
    app.setWindowIcon(QtGui.QIcon(os.path.join(Path(__file__).parent, "./car.png")))
    sys.exit(app.exec())
