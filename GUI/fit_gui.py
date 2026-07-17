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
from matplotlib.ticker import LogLocator
from astropy.visualization.stretch import LogStretch, LinearStretch
from astropy.visualization import ImageNormalize
import math
import pyimfit
import shutil
import numpy as np
import re
import html
import pandas as pd
import matplotlib.patches
import glob
from PIL import Image
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import scipy
import astropy.units as u
from astropy.cosmology import WMAP9 as cosmo
import astropy.cosmology.units as cu
import traceback

BASE_DIR = Path(Path(os.path.dirname(__file__)).parent).resolve()
sys.path.append(str(BASE_DIR))
sys.path.append(os.path.join(BASE_DIR, 'decomposer'))
sys.path.append(os.path.join(BASE_DIR, 'decomposer/manual_fitting'))

from generate_imfit_conf import generate_config
from iman_new.imp.masking.convert_reg_to_mask import mask as convert_reg_to_mask
import test_manual_decomposer
from photometric_cut import photometric_cut, fold_cut_to_radial_profile
from photometric_cut_helpers import pixel_scale_from_header_arcsec_per_pix

from plot_canvas import PlotCanvas
from param_slider import ParamSliderWidget,ConfigAdjustWidget
from copy_params import CopyParametersDialog
from utils import *

from enum import Enum

# For indexing the composed image
class Composed(Enum):
    IMAGE = 0
    MODEL = 1
    RESID = 2
    RELRESID = 3

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
        self.plotrelresid = True
        self.current_selected_indices = None

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
        self.ui.relresidcheckbox.toggled.connect(self.setresidtype)

        self.ui.opends9button.clicked.connect(self.open_ds9)
        self.ui.opends9button.setShortcut(QKeySequence("O"))
        self.ui.importds9maskbutton.clicked.connect(self.import_ds9_mask)
        self.ui.importds9maskbutton.setShortcut(QKeySequence("CTRL+M"))
        self.ui.refitbutton.clicked.connect(self.refit)
        self.ui.refitbutton.setShortcut(QKeySequence("CTRL+R"))

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
            # with open(os.path.join(MAINDIR, LOCAL_DIR, 'galmarks.json')) as f:
            with open(os.path.join(p, 'galmarks.json')) as f:
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
        ds9_cmd = self.gui_config.get("ds9_path", "ds9") or "ds9"
        if f'{self.fit_type}_{self.band}_composed.fits' in os.listdir(p):
            files = [f"{os.path.join(p, f'{self.fit_type}_{self.band}_composed.fits')}"]
            arg = [ds9_cmd, "-cmap", self.gui_config["ds9_cmap"], "-scale", self.gui_config["ds9_scale"], "-scale", "limits", f"{self.gui_config['ds9_limits'][0]}", f"{self.gui_config['ds9_limits'][1]}", "-cube", "3"]
        else:
            files = [f"{os.path.join(p, f'image_{self.band}.fits')}"]
            arg = [ds9_cmd, "-cmap", self.gui_config["ds9_cmap"], "-scale", self.gui_config["ds9_scale"], "-scale", "limits", f"{self.gui_config['ds9_limits'][0]}", f"{self.gui_config['ds9_limits'][1]}"]

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
    
    def refresh_conf(self, redraw=True):
        try:
            layout: QVBoxLayout = self.ui.configsliders
            # Reset the layout first
            try:
                if redraw:
                    clearLayout(layout)
            except Exception as e:
                print(e)
                pass

            if redraw:
                selected_indices = self.current_selected_indices
                if selected_indices is None:
                    selected_indices = set(range(len(self.dataset.config_dict["function_sets"][0]["function_list"]))) if self.dataset.config_dict is not None else None
                self.config_adjust = ConfigAdjustWidget(
                    parent=layout,
                    dataset=self.dataset,
                    config_callback=self.on_component_selection_changed,
                    selected_indices=selected_indices,
                )
                self.config_adjust.draw_config_adjust()
        except:
            clearLayout(self.ui.configsliders)
            print(traceback.format_exc())
            pass
    
    def refresh_fitparams(self):
        try:
            self.highlight_boundary_params(self.dataset.fit_results_text, self.dataset.fit_results)
        except:
            self.params.setPlainText("Fit Params not found!")
            self.params.repaint()
            print(tb.format_exc())
    def on_component_selection_changed(self, selected_indices):
        #TODO: implement
        pass
    
    def refresh_plots(self):
        self.jpg_img_plot.plot(self.dataset.jpg_image, limits=None, cmap=None, stretch=None, plottext="JPG Image", cbar=False)


        galname = self.selected_galaxy_path.name
        if ellipse_fit_p != None:
            ellipse_params = self.ellipse_fit_data[self.ellipse_fit_data["file"] == galname]
        else:
            ellipse_params = pd.DataFrame()
        self.current_ellipse_params = ellipse_params
        self.toggle_1d.setEnabled(not ellipse_params.empty)

        self.radial_data = self.get_radial_data()

        self.plot_image()
        self.plot_model()
        self.plot_residual()
        self.plot_config()
        self.plot_config_residual()

    def refresh_tabledata(self):
        galaxypath = self.selected_galaxy_path
        galaxy = galaxypath.name
        try:
            master_table_data_gal = self.master_table_data[self.master_table_data["NAME"] == galaxy]
            self.dataset.z = master_table_data_gal["REDSHIFT"].iloc[0]
            self.dataset.d_A = cosmo.angular_diameter_distance(z=self.dataset.z)
        except:
            self.dataset.z = self.dataset.d_A = None
        if self.dataset.z != self.dataset.z:
            self.dataset.z = self.dataset.d_A = None

    def changegal(self):
        # Update UI based on the currently selected leaf galaxy folder
        galaxypath = self.selected_galaxy_path
        galaxy = galaxypath.name
        self.currentgalaxytext.setText(f"Current Galaxy: {galaxy}")
        self.currentgalaxytext.repaint()

        jpg_image_path = os.path.join(galaxypath, "image.jpg")
        fits_image_path = os.path.join(galaxypath, f"image_{self.band}.fits") 
        fits_invvar_image_path = os.path.join(galaxypath, f"image_{self.band}_invvar.fits")
        fits_psf_path = os.path.join(galaxypath, f"psf_patched_{self.band}.fits")

        mask_path = os.path.join(galaxypath, f"image_mask.fits")
        config_path = os.path.join(galaxypath, f"{self.fit_type}_{self.band}.dat")
        fits_composed_path = os.path.join(galaxypath, f"{self.fit_type}_{self.band}_composed.fits")
        fit_results_path = os.path.join(galaxypath, f"{self.fit_type}_{self.band}_fit_params.txt")

        if self.gui_config["data_type"] == "DESI":
            self.dataset = DESIDataSet(jpg_image_path, fits_image_path, fits_invvar_image_path, fits_psf_path, mask_path, config_path, fit_results_path, fits_composed_path)
        else:
            self.dataset = DataSet(jpg_image_path, fits_image_path, fits_invvar_image_path, fits_psf_path, mask_path, config_path, fit_results_path, fits_composed_path)
        self.dataset.load_all()
        # self.current_selected_indices = set(range(len(self.base_config_dict["function_sets"][0]["function_list"]))) if self.base_config_dict is not None else None

        self.refresh_tabledata()
        self.refresh_conf()
        self.refresh_fitparams()
        self.refresh_plots()

    def plot_image(self):
        self.img.plot(self.dataset.fits_image, limits=self.gui_config["plot_limits"], cmap=self.gui_config["plot_cmap"], ellipse_params=self.current_ellipse_params, plottext=f"{self.band} Band Image")

    def plot_model(self):
        if self.is_1d_mode:
            if self.radial_data is not None:
                self.model.plot_profiles(self.radial_data['model']['host'], self.radial_data['model']['polar'], 'Model Radial Profile', overplot=self.radial_data['image'], surfbright=True, d_A=self.dataset.d_A)
                return
        try:
            im = self.dataset.fits_composed[Composed.MODEL.value]
        except:
            im = np.array([])
        self.model.plot(im, limits=self.gui_config["plot_limits"], cmap=self.gui_config["plot_cmap"], plottext=f"2D Model Image")

    def plot_residual(self):
        if self.is_1d_mode:
            if self.radial_data is not None:
                if self.plotrelresid:
                    y_label='Relative Residual'
                else:
                    y_label='Residual Flux (nanomaggies/arcsec^2)'
                self.resid.plot_profiles(
                    self.radial_data['residual']['host'],
                    self.radial_data['residual']['polar'],
                    'Residual Radial Profile',
                    y_label=y_label,
                    is_resid=True,
                    d_A=self.dataset.d_A
                )
                return
        if self.plotrelresid:
            plottext = "(Image - Model)/Image" 
            limits = self.gui_config["plot_relresid_limits"]
            try:
                im = self.dataset.fits_composed[Composed.RELRESID.value]
            except:
                im = np.array([])
            self.resid.plot(im, limits=limits, cmap=self.gui_config["plot_resid_cmap"], stretch=LinearStretch(), plottext=plottext) 
        else:
            plottext = "Image - Model"
            limits = self.gui_config["plot_resid_limits"]
            try:
                im = self.dataset.fits_composed[Composed.RESID.value]
            except:
                im = np.array([])
            self.resid.plot(im, limits=limits, cmap=self.gui_config["plot_resid_cmap"], stretch=LinearStretch(), plottext=plottext) 

    def plot_config(self):
        if self.is_1d_mode:
            if self.radial_data is not None:
                self.configimg.plot_profiles(self.radial_data['config']['host'], self.radial_data['config']['polar'], 'Config Radial Profile', overplot=self.radial_data['image'], surfbright=True, d_A=self.dataset.d_A)
                return
        self.configimg.plot(self.dataset.config_im, limits=self.gui_config["plot_limits"], cmap=self.gui_config["plot_cmap"], plottext="2D Config Image")

    def plot_config_residual(self):
        if self.is_1d_mode:
            if self.plotrelresid:
                y_label = 'Relative Config Residual'
            else:
                y_label = 'Config Residual Flux (nanomaggies/arcsec^2)'
            if self.radial_data is not None:
                self.configresid.plot_profiles(
                    self.radial_data['config_residual']['host'],
                    self.radial_data['config_residual']['polar'],
                    'Config Residual Radial Profile',
                    y_label=y_label,
                    is_resid=True,
                    d_A=self.dataset.d_A
                )
                return
        if self.plotrelresid:
            plottext = "(Image - Config)/Image" 
            limits = self.gui_config["plot_relresid_limits"]
        else:
            plottext = "Image - Config"
            limits = self.gui_config["plot_resid_limits"]
        self.configresid.plot(self.dataset.getconfigresid(self.plotrelresid), limits=limits, cmap=self.gui_config["plot_resid_cmap"], stretch=LinearStretch(), plottext=plottext)

    def get_radial_data(self):
        if self.current_ellipse_params is None or self.current_ellipse_params.empty:
            return None
        host_e = self.current_ellipse_params[self.current_ellipse_params["PolarOrHost"] == "Host"]
        polar_e = self.current_ellipse_params[self.current_ellipse_params["PolarOrHost"] == "Polar"]
        if host_e.empty or polar_e.empty:
            return None
        host_pa = host_e["angle"].iloc[0]
        polar_pa = polar_e["angle"].iloc[0]
        host_len = host_e["semi_major"].iloc[0] * 2 # Just a bit larger
        polar_len = polar_e["semi_major"].iloc[0] * 2

        zeropoint = 22.5

        mask = self.dataset.fits_mask
        sci_image = np.where(mask == 0, self.dataset.fits_image, 0)
        config_im = self.dataset.config_im

        image_host = profile_from_image(sci_image, host_pa, host_len, self.dataset.pixel_scale, surf_bright=True)
        image_polar = profile_from_image(sci_image, polar_pa, polar_len, self.dataset.pixel_scale, surf_bright=True)

        model_host = profile_from_image(config_im, host_pa, host_len, self.dataset.pixel_scale, surf_bright=True)
        model_polar = profile_from_image(config_im, polar_pa, polar_len, self.dataset.pixel_scale, surf_bright=True)
        
        try:
            if self.plotrelresid:
                residual_data = np.where(sci_image != 0, (sci_image - config_im)/sci_image, 0)
            else:
                residual_data = sci_image - config_im
            residual_host = profile_from_image(residual_data, host_pa, host_len, self.dataset.pixel_scale)
            residual_polar = profile_from_image(residual_data, polar_pa, polar_len, self.dataset.pixel_scale) 
        except:
            residual_host = residual_polar = None

        if config_im.size != 0:
            config_host = profile_from_image(config_im, host_pa, host_len, self.dataset.pixel_scale, surf_bright=True) 
            config_polar = profile_from_image(config_im, polar_pa, polar_len, self.dataset.pixel_scale, surf_bright=True)

            if self.plotrelresid:
                config_residual_data = np.where(sci_image !=0, (sci_image - config_im)/sci_image, 0)
            else:
                config_residual_data = sci_image - config_im
            config_residual_host = profile_from_image(config_residual_data, host_pa, host_len, self.dataset.pixel_scale)
            config_residual_polar = profile_from_image(config_residual_data, polar_pa, polar_len, self.dataset.pixel_scale)
        else:
            config_host = config_polar = config_residual_host = config_residual_polar = None

        return {
            'image': {'host': image_host, 'polar': image_polar},
            'model': {'host': model_host, 'polar': model_polar},
            'residual': {'host': residual_host, 'polar': residual_polar},
            'config': {'host': config_host, 'polar': config_polar},
            'config_residual': {'host': config_residual_host, 'polar': config_residual_polar},
        }
 

    def highlight_boundary_params(self, params_file_lines, fit_results):
        if params_file_lines is None:
            return 

        if fit_results is None or self.dataset.config_dict is None:
            self.params.setHtml("<pre>" + html.escape("".join(params_file_lines)) + "</pre>")
            return

        try:
            function_list = self.dataset.config_dict["function_sets"][0]["function_list"]
        except Exception:
            self.params.setHtml("<pre>" + html.escape("".join(params_file_lines)) + "</pre>")
            raise

        html_lines = []
        func_idx = -1
        for line in params_file_lines:
            stripped = line.strip()
            if stripped.startswith("FUNCTION"):
                func_idx += 1
                html_lines.append(html.escape(line))
                continue

            highlight = False
            if "+/-" in line and func_idx >= 0 and func_idx in fit_results[0]:
                m = re.match(r"^\s*(\S+)\s+([+-]?[0-9]*\.?[0-9]+(?:[eE][+-]?[0-9]+)?)\s*#\s*\+/\-\s*([+-]?[0-9]*\.?[0-9]+(?:[eE][+-]?[0-9]+)?)", line)
                if m:
                    param_name = m.group(1)
                    try:
                        param_val = float(m.group(2))
                        param_unc = float(m.group(3))
                    except ValueError:
                        param_val = None
                        param_unc = None

                    if param_val is not None and param_unc == 0:
                        func_params = function_list[func_idx]["parameters"]
                        if param_name in func_params:
                            bounds = func_params[param_name]
                            if bounds[1] == 'fixed':
                                lowlim = bounds[0]
                                hilim = bounds[0]
                            else:
                                lowlim = bounds[1]
                                hilim = bounds[2]
                            if lowlim != hilim: # Parameter is fixed if these are equal
                                highlight = True

            escaped_line = html.escape(line)
            if highlight:
                html_lines.append(f"<span style='background-color: yellow; color: red;'>%s</span>" % escaped_line)
            else:
                html_lines.append(escaped_line)

        self.params.setHtml("<pre>" + "".join(html_lines) + "</pre>")

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
    
    def markgalaxy(self, markas):
        if getattr(self, "selected_galaxy_path", None) is None:
            return
        galname = self.selected_galaxy_path.name
        # Save mark state
        self.galmarks[galname] = markas
        # with open(os.path.join(MAINDIR, LOCAL_DIR, 'galmarks.json'), 'w') as fp:
        with open(os.path.join(p, 'galmarks.json'), 'w') as fp:
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

        config_path = self.dataset.config_path

        # If we have a config model and param_widgets, update the config model with the new values
        model_dict = self.dataset.config_dict
        function_list = model_dict["function_sets"][0]["function_list"]
        # Update parameter values from widgets
        for func_idx, func in enumerate(function_list):
            params = func["parameters"]
            for param in params.keys():
                key = (func_idx, param)
                if key in self.config_adjust.param_widgets:
                    values = self.config_adjust.param_widgets[key].get_values()
                    if values['fixed']:
                        # Only value and 'fixed' string
                        params[param] = [values['value'], 'fixed']
                    else:
                        # Value, min, max
                        params[param] = [values['value'], values['min'], values['max']]
        
        # Rebuild the config description from the updated dict
        try:
            new_model = pyimfit.ModelDescription.dict_to_ModelDescription(model_dict)
        except Exception as e:
            QMessageBox.warning(
                self, "Warning", f"Warning: {e}"
            )
            return

        config_text = "".join(new_model.getStringDescription())
        # Backup old config
        if os.path.isfile(config_path):
            shutil.copyfile(src=config_path, dst=config_path + ".bak")
        with open(config_path, "w") as f:
            f.write(config_text)
        if f: f.close()

        # Refresh config image and residual
        self.dataset.load_config()
        self.refresh_conf(redraw=False)
        # self.refresh_conf()
        self.refresh_plots()
    
    def copy_parameters_from_band(self): # TODO need to update this to work with the new refactor
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
                            # param_unc = fit_params_values[func_idx]["parameters_unc"][param_name]
                            # Ngl I don't know how I feel about determining the bounds like this, but it works for now (subject to change)
                            # Now actually changed to just using the config bounds
                            # if param_unc != 0:
                            #     current_functions[func_idx]["parameters"][param_name] = [param_value, param_value-param_unc, param_value+param_unc]
                            # else:
                            #     current_functions[func_idx]["parameters"][param_name] = [param_value, 'fixed']
                            current_functions[func_idx]["parameters"][param_name] = [param_value,  source_functions[func_idx]["parameters"][param_name][1],  source_functions[func_idx]["parameters"][param_name][2]]
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
        ### Actually nevermind, the new dataset class kinda helps here
        galpath = self.selected_galaxy_path

        img = fits.open(self.dataset.fits_image_path)[0]
        mask = self.dataset.fits_mask
            
        psf = self.dataset.fits_psf
        invvar = self.dataset.fits_invvar_image
        outfile_name = f"{self.fit_type}_{self.band}.dat" 
        outfile = os.path.join(galpath, outfile_name)
        galname = self.selected_galaxy_path.name
        ellipse_fit_data_gal = self.ellipse_fit_data[self.ellipse_fit_data["file"] == galname]
        if ellipse_fit_data_gal.empty:
            QMessageBox.critical(self, "Failed to Generate Config", f"Galaxy {galname} not in Ellipse Fit Data!")
            return
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
                    self.dataset.config_model_desc = generate_config(
                        galpath,
                        self.band,
                        img,
                        mask,
                        psf,
                        invvar,
                        type="ring",
                        ellipse_fit_data=ellipse_fit_data_gal,
                        galaxy_type=master_table_data_gal,
                        plot_slits=self.gui_config["show_phot_slits"],
                        outfile_name=outfile,
                        fit_type=self.fit_type,
                    )
                    QMessageBox.information(self, "Config Generation Information", "Config successfully written")
                else:
                    pass
            else:
                self.dataset.config_model_desc = generate_config(
                    galpath,
                    self.band,
                    img,
                    mask,
                    psf,
                    invvar,
                    type="ring",
                    ellipse_fit_data=ellipse_fit_data_gal,
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
    
    def openonedfitdialog(self):
        if self.selected_galaxy_path is None:
            QMessageBox.warning(self, "No galaxy selected", "Please select a galaxy first.")
            return
        
        galpath = self.selected_galaxy_path
        manual_decomp_path = os.path.join(MAINDIR, "decomposer", "manual_fitting", "test_manual_decomposer.py")
        mask_path = self.dataset.mask_path
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
        
    
    def setresidtype(self, state):
        self.plotrelresid = state 
        self.changegal()

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
