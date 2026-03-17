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
import pyimfit
import shutil
import numpy as np
import re
import pandas as pd
import matplotlib.patches
import glob

BASE_DIR = Path(Path(os.path.dirname(__file__)).parent).resolve()
sys.path.append(os.path.join(BASE_DIR, 'decomposer'))

from generate_imfit_conf import generate_config

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
        fig = Figure(figsize=(250/100, 250/100), dpi=100)
        self.ax = fig.add_subplot(111)
        fig.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=1.0)
        super().__init__(fig)
        self.setParent(parent)

    def plot(self, im, limits, cmap, stretch=LogStretch(), ellipse_params=pd.DataFrame):
        self.ax.cla()
        self.ax.set_axis_off()
        if im.any():
            norm = ImageNormalize(stretch=stretch, vmin=limits[0], vmax=limits[1])
            self.ax.imshow(im, origin="lower", norm=norm, cmap=cmap)
            if not ellipse_params.empty:
                host = ellipse_params[ellipse_params["label"] == "Host"]
                polar = ellipse_params[ellipse_params["label"] == "Polar"]
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



class DirOnlyChildrenFileSystemModel(QFileSystemModel):
    def __init__(self, mark_colors=None, galmarks=None, parent=None):
        super().__init__(parent)
        self.mark_colors = mark_colors or {}
        self.galmarks = galmarks or {}

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
                    mark = self.galmarks.get(name)
                    if mark:
                        col = self.mark_colors.get(mark)
                        if col:
                            return QBrush(QColor(col))
            except Exception:
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
        self.setLayout(parameter_adjust_layout)

        self.set_fixed_state(fixed)

        self.slider.valueChanged.connect(self.slider_changed)
        self.valspinbox.valueChanged.connect(self.spinbox_changed)
        self.minspinbox.valueChanged.connect(self.minspinbox_changed)
        self.maxspinbox.valueChanged.connect(self.maxspinbox_changed)
        self.fixed_checkbox.stateChanged.connect(lambda state: self.set_fixed_state(state==2))
        
        spinboxes_layout.setSizeConstraint(QLayout.SizeConstraint.SetMinimumSize)
        parameter_adjust_layout.setStretchFactor(spinboxes_layout, 10)

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

    def spinbox_changed(self, value):
        int_val = int(round(value * self.scale))
        self.slider.blockSignals(True)
        self.slider.setValue(int_val)
        self.slider.blockSignals(False)

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

class CopyParametersDialog(QDialog):
    """Dialog for copying parameters from one band to another."""
    
    def __init__(self, galaxy_path, current_band, fit_type, parent=None):
        super().__init__(parent)
        self.galaxy_path = galaxy_path
        self.current_band = current_band
        self.fit_type = fit_type
        self.source_band = None
        self.source_config = None
        self.setWindowTitle("Copy Parameters From Band")
        # self.setMinimumWidth(400)
        # self.setMinimumHeight(500)
        
        layout = QVBoxLayout()
        
        # Band selection
        band_layout = QHBoxLayout()
        band_label = QLabel("Copy from band:")
        self.band_combo = QComboBox()
        available_bands = [b for b in ["g", "r", "i", "z"] if b != current_band]
        self.band_combo.addItems(available_bands)
        self.band_combo.currentTextChanged.connect(self.on_band_changed)
        band_layout.addWidget(band_label)
        band_layout.addWidget(self.band_combo)
        band_layout.addStretch()
        layout.addLayout(band_layout)
        
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
    
    def on_band_changed(self, band):
        """Load parameters from the selected source band."""
        self.source_band = band
        self.param_list.clear()
        
        config_path = os.path.join(self.galaxy_path, f"{self.fit_type}_{band}.dat")
        try:
            self.source_config = pyimfit.parse_config_file(config_path)
            config_dict = self.source_config.getModelAsDict()
            function_list = config_dict["function_sets"][0]["function_list"]
            
            # Load function labels
            labels = read_function_labels(config_path)
            
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
                    item_text = f"  └─ {param_name}"
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

class MainWindow(QMainWindow):
    def __init__(self, p=None, master_table_p = None, ellipse_fit_p=None):
        super().__init__()

        # Loading the config file for the GUI
        with open(os.path.join(MAINDIR, LOCAL_DIR, 'config.json')) as config:
            self.gui_config = json.load(config)
            config.close()
            
        # Read in the ellipse fit data 
        csvs = glob.glob(os.path.join(ellipse_fit_p, "*.csv"))
        ellipse_fit_data = pd.DataFrame(columns=["file", "label","contour", "x_center", "y_center", "semi_major", "semi_minor", "angle", "center_offset", "axis_ratio", "pa_diff"])
        for csv in csvs:
            dat = pd.read_csv(csv)
            ellipse_fit_data = pd.concat([ellipse_fit_data, dat])
        self.ellipse_fit_data = ellipse_fit_data

        # Read in the master table (I might change this later so we don't have to do this in the GUI code)
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
        self.ui.refitbutton.clicked.connect(self.refit)
        self.ui.refitbutton.setShortcut(QKeySequence("CTRL+R"))
        self.ui.cancelbutton.clicked.connect(self.cancel)
        self.ui.cancelbutton.setShortcut(QKeySequence("CTRL+C"))

        self.ui.fileexplorerbutton.clicked.connect(self.open_gal_fileexplorer)
        self.ui.fileexplorerbutton.setShortcut(QKeySequence("CTRL+F"))
        
        # Connect Copy From Band button
        self.ui.copyparamsbutton.clicked.connect(self.copy_parameters_from_band)
        self.ui.copyparamsbutton.setShortcut(QKeySequence("CTRL+P"))

        # Also connect the generate config button
        self.ui.newconfbutton.clicked.connect(self.regenconf)
        self.ui.newconfbutton.setShortcut(QKeySequence("CTRL+G"))

        # Get the fit type
        self.ui.fit_type_combo.currentTextChanged.connect(self.change_fit_type)
        self.fit_type = self.ui.fit_type_combo.currentText()

        # Loading the list of galaxies
        self.galaxytree: QTreeView = self.ui.galaxytree
        self.galaxytree.setColumnHidden(1, True)

        # Track the currently selected lowest-level galaxy folder (Path) if any
        self.selected_galaxy_path = None


        # Process list of currently running fits
        self.ps = []
        self.fit_dialogs = []

        # Setting up the FITs plots for the iamge, model, and residual
        self.img = PlotCanvas(parent=self.ui.galimg)
        self.model = PlotCanvas(parent=self.ui.galmodel)
        self.resid = PlotCanvas(parent=self.ui.galresid)
        self.configimg = PlotCanvas(parent=self.ui.configimg)
        self.configresid = PlotCanvas(parent=self.ui.configresid)

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

    def open_gal_fileexplorer(self):
        open_folder(self.selected_galaxy_path)

    def change_fit_type(self):
        self.fit_type = self.ui.fit_type_combo.currentText()
        self.changegal()

    def open_ds9(self, *args):
        p = self.selected_galaxy_path
        files = [f"{os.path.join(p, f'{self.fit_type}_{self.band}_composed.fits')}"]
        arg = ["ds9", "-cmap", self.gui_config["ds9_cmap"], "-scale", self.gui_config["ds9_scale"], "-scale", "limits", f"{self.gui_config['ds9_limits'][0]}", f"{self.gui_config['ds9_limits'][1]}", "-cube", "3"]
        arg.extend(files)
        subprocess.Popen(arg)
    
    def get_composed_data(self, galaxy_path, band, idx, fit_type):
        # idx = 0 -> Image with mask, 1 -> Model image, 2 -> Residual, 3 -> Percent residual, 4 Onwards -> Components of model
        try:
            im = fits.getdata(os.path.join(galaxy_path, f"{fit_type}_{band}_composed.fits"))[idx]
        except:
            # Regualar image fallback
            if idx != 0:
                im = np.array([])
            else:
                try:
                    im = fits.getdata(os.path.join(galaxy_path, f"image_{band}.fits"))
                except:
                    return np.array([])
        return im

    def getconfigim(self, galaxypath, band, fit_type, shape, maxThreads=4):
        try:
            model_desc = pyimfit.parse_config_file(os.path.join(galaxypath, f"{fit_type}_{band}.dat"))
            psf = fits.getdata(os.path.join(galaxypath, f"psf_patched_{band}.fits"))
            imfitter = pyimfit.Imfit(model_desc, psf=psf, maxThreads=maxThreads)
            # imfitter = pyimfit.Imfit(model_desc, maxThreads=maxThreads)
            im = imfitter.getModelImage(shape=shape)
        except:
            im = np.array([])

        return im
    
    def getconfigresid(self, im, imconfig):
        try:
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
            config_path = os.path.join(galaxypath, f"{self.fit_type}_{self.band}.dat")
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
            pass


        try:
            with open(os.path.join(galaxypath, f"{self.fit_type}_{self.band}_fit_params.txt"), "r") as f:
                params_file = f.readlines()
            self.params.setPlainText("".join(params_file))
            self.params.repaint()
        except:
            self.params.setPlainText("Fit Params not found!")
            self.params.repaint()

        pixmap = QPixmap(os.path.join(galaxypath, "image.jpg"))
        self.ui.galaxyjpg.setPixmap(pixmap)
        

        # self.img.get_composed_data(galaxypath, self.band, idx=0, fit_type=self.fit_type)
        img = self.get_composed_data(galaxypath, self.band, idx=0, fit_type=self.fit_type)
        galname = self.selected_galaxy_path.name
        ellipse_params = self.ellipse_fit_data[self.ellipse_fit_data["file"] == galname]
        self.img.plot(img, limits=self.gui_config["plot_limits"], cmap=self.gui_config["plot_cmap"], ellipse_params=ellipse_params)

        model = self.get_composed_data(galaxypath, self.band, idx=1, fit_type=self.fit_type)
        self.model.plot(model, limits=self.gui_config["plot_limits"], cmap=self.gui_config["plot_cmap"])

        resid = self.get_composed_data(galaxypath, self.band, idx=2, fit_type=self.fit_type)
        self.resid.plot(resid, limits=self.gui_config["plot_resid_limits"], cmap=self.gui_config["plot_resid_cmap"], stretch=LinearStretch())
        
        imconfig = self.getconfigim(galaxypath, self.band, self.fit_type, np.shape(img), maxThreads=self.gui_config["imfit_maxthreads"])
        self.configimg.plot(imconfig, limits=self.gui_config["plot_limits"], cmap=self.gui_config["plot_cmap"])

        imresidconfig = self.getconfigresid(img, imconfig)
        self.configresid.plot(imresidconfig, limits=self.gui_config["plot_resid_limits"], cmap=self.gui_config["plot_resid_cmap"], stretch=LinearStretch())

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
 
    def draw_params(self, initval, lowlim, hilim, fixed, paramkey, label, layout):
        if not hasattr(self, 'param_widgets'):
            self.param_widgets = {}
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
        dlg = fit_monitor.FitMonitorDialog(path, self.band, self.solvertype, max_threads=self.gui_config["imfit_maxthreads"], fit_type=self.fit_type, parent=self)
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
        except Exception:
            pass
    
    def saveconfig(self):
        if getattr(self, "selected_galaxy_path", None) is None:
            print("No galaxy selected to save config")
            return
        p = self.selected_galaxy_path
        fit_type = self.ui.fit_type_combo.currentText()
        config_path = os.path.join(p, f"{fit_type}_{self.band}.dat")

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
                
                QMessageBox.information(self, "Success", f"Copied {copied_count} parameter(s) from band {source_band}.")
                
                # Refresh the UI with new parameters
                self.changegal()
                
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to copy parameters: {str(e)}")
    
    def regenconf(self):
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
                    generate_config(outfile, self.band, img, mask, psf, invvar, self.fit_type, ellipse_fit_data_gal, model_desc_dict, galaxy_type = master_table_data_gal)
                    QMessageBox.information(self, "Config Generation Information", "Config successfully written")
                else:
                    pass
            else:
                generate_config(outfile, self.band, img, mask, psf, invvar, self.fit_type, ellipse_fit_data_gal, model_desc_dict, galaxy_type = master_table_data_gal)
                QMessageBox.information(self, "Config Generation Information", "Config successfully written")
        except Exception as e:
            QMessageBox.critical(self, "Config Generation Information", f"Config generation failed:\n{e}")
            print(e)

        
        self.changegal()
        
        


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Open IMFIT helper GUI"
    )
    
    parser.add_argument("-p", help="Path to folder containing galaxies", default=".")
    parser.add_argument("--ellipse_fit", help="Path to folder ellipse fit data", default=".")
    parser.add_argument("--master_table", help="Path to master table data", default=".")

    args = parser.parse_args()
    p = Path(args.p).resolve()
    ellipse_fit_p = Path(args.ellipse_fit).resolve()
    master_table_p = Path(args.master_table).resolve()
    app = QApplication(sys.argv)
    main_win = MainWindow(p,master_table_p, ellipse_fit_p)
    app.setWindowIcon(QtGui.QIcon(os.path.join(Path(__file__).parent, "./car.png")))
    sys.exit(app.exec())
