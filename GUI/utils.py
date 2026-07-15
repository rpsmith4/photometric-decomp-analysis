import pyimfit
from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtWidgets import QApplication, QWidget, QMessageBox, QMainWindow, QDialog, QAbstractItemView
from PySide6.QtGui import QColor, QPixmap, QKeySequence, QImage, QBrush
from PySide6.QtWidgets import *
from PySide6.QtCore import QFile
from PySide6.QtUiTools import *
import os
from pathlib import Path
import re
import math
from PIL import Image
from astropy.io import fits
import numpy as np
import pathlib
import traceback as tb
import scipy
import astropy.units as u
import sys

BASE_DIR = Path(Path(os.path.dirname(__file__)).parent).resolve()
sys.path.append(str(BASE_DIR))
sys.path.append(os.path.join(BASE_DIR, 'decomposer'))
sys.path.append(os.path.join(BASE_DIR, 'decomposer/manual_fitting'))

from photometric_cut_helpers import pixel_scale_from_header_arcsec_per_pix

class DataSet():
    def __init__(self, jpg_image_path: pathlib.PosixPath = None, fits_image_path: pathlib.PosixPath = None, fits_invvar_image_path: pathlib.PosixPath = None, fits_psf_path: pathlib.PosixPath = None, mask_path: pathlib.PosixPath = None, config_path: pathlib.PosixPath = None, fit_results_path:pathlib.PosixPath = None, fits_composed_path: pathlib.PosixPath = None, z: float = None):
        self.jpg_image_path = jpg_image_path
        self.jpg_image = np.array([])

        self.fits_image_path = fits_image_path
        self.fits_image = np.array([])
        self.pixel_scale = None

        self.fits_invvar_image_path = fits_invvar_image_path
        self.fits_invvar_image = np.array([])

        self.fits_psf_path = fits_psf_path
        self.fits_psf = np.array([])

        self.mask_path = mask_path
        self.fits_mask = np.array([])

        self.config_path = config_path
        self.config_dict = None
        self.config_model_desc = None
        self.config_im = np.array([])
        
        self.fit_results_path = fit_results_path
        self.fit_results_text = None
        self.fit_results = None

        self.fits_composed_path = fits_composed_path
        self.fits_composed = np.array([])

        self.z = z
        self.d_A = None

    def load_jpg(self):
        self.jpg_image = np.asarray(Image.open(self.jpg_image_path))
        self.jpg_image = np.flipud(self.jpg_image)
    
    def load_fits_image(self, apply_mask = True):
        try:
            fits_file = fits.open(self.fits_image_path)[0]
            self.fits_image = fits.getdata(self.fits_image_path)
            self.pixel_scale = pixel_scale_from_header_arcsec_per_pix(fits_file)
            if self.apply_mask:
                self.fits_image = self.apply_mask(self.fits_image)
        except:
            print(tb.format_exc())
            pass
    
    def load_fits_invvar_image(self):
        try:
            self.fits_invvar_image = fits.getdata(self.fits_invvar_image_path)
        except:
            print(tb.format_exc())
            pass
    
    def load_mask(self):
        self.fits_mask = fits.getdata(self.mask_path)
    
    def load_psf(self):
        try:
            self.fits_psf = fits.getdata(self.fits_psf_path)
        except:
            print(tb.format_exc())
            pass
    
    def load_composed(self, apply_mask = True):
        # idx = 0 -> Regular image with mask applied, 1 -> Model image, 2 -> Residual, 3 -> Percent residual, 4 Onwards -> Components of model
        try:
            self.fits_composed = fits.getdata(self.fits_composed_path)
            if apply_mask:
                self.fits_composed[0] = np.where(self.fits_mask > 0, 0, self.fits_composed[0])
        except:
            self.fits_composed = None
    
    def load_config(self):
        try:
            self.config_model_desc = pyimfit.parse_config_file(self.config_path)
            self.config_dict = self.config_model_desc.getModelAsDict()

            # Add function labels to config_dict
            # Want to ensure that I actually keep the labels since pyimift is incapable of doing so for some reason
            labels = read_function_labels(self.config_path)
            function_list = self.config_dict["function_sets"][0]["function_list"]
            for i, func in enumerate(function_list):
                if i < len(labels):
                    func['label'] = labels[i]
                else:
                    func['label'] = None
            self.getconfigim()
        except:
            self.config_model_desc = None
            self.config_dict = None
    
    def load_fit_results(self):
        try:
            with open(self.fit_results_path, "r") as f:
                self.fit_results_text = f.readlines()
            self.fit_results = parse_results(self.fit_results_path)
        except:
            print(tb.format_exc())
            self.fit_results_text = None
            self.fit_results = None
    
    def load_all(self):
        self.load_jpg()
        self.load_mask()
        self.load_fits_image()
        self.load_fits_invvar_image()
        self.load_psf()
        self.load_fit_results()
        self.load_composed()
        self.load_config()

    def getconfigim(self, maxThreads=4, apply_psf = True):
        try:
            if apply_psf:
                imfitter = pyimfit.Imfit(self.config_model_desc, psf=self.fits_psf, maxThreads=maxThreads)
            else:
                imfitter = pyimfit.Imfit(self.config_model_desc, maxThreads=maxThreads)
            im = imfitter.getModelImage(shape=np.shape(self.fits_image))
        except:
            im = np.array([])
            print(tb.format_exc())
        self.config_im = im
    
    def getconfigresid(self, relresid=False):
        try:
            im = self.fits_image
            imconfig = self.config_im
            mask = self.fits_mask

            if relresid:
                resid_im = (im- imconfig)/im
            else:
                resid_im = im-imconfig
            if mask.size != 0:
                return np.where(mask >0, 0, resid_im)
            else:
                return resid_im
        except:
            return np.array([])

    def apply_mask(self, im):
        return np.where(self.fits_mask > 0, 0, im)

class DESIDataSet(DataSet):
    def __init__(self, jpg_image_path = None, fits_image_path = None, fits_invvar_image_path = None, fits_psf_path = None, mask_path = None, config_path = None, fit_results_path = None, fits_composed_path = None):
        super().__init__(jpg_image_path, fits_image_path, fits_invvar_image_path, fits_psf_path, mask_path, config_path, fit_results_path, fits_composed_path)


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
            func_dict["parameters"][param] = func_dict["parameters"][param][0]
        if k == 0:
            func_dict["label"] = "Host"
        if k == 1:
            func_dict["label"] = "Polar"
        func_dict["parameters_unc"] = uncs[func_dict["label"]]
        func_dict["band"] = band
        functions.append(func_dict)

    function_map = {idx: func for idx, func in enumerate(functions)}
    return function_map, chi_sq, chi_sq_red, status, status_message

def profile_from_image(image_data, pa, length, pixel_scale=None, surf_bright=False):
    try:
        pa = np.deg2rad(pa + 90) # Need to account for the offset that imfit gives lol
        # length = int(np.shape(image_data)[0]*np.cos(np.pi/4))

        c = (int(image_data.shape[0]/2)-1, int(image_data.shape[1]/2)-1) # Just assuming the center of the galaxy is the center of the image

        x0 = c[0] + np.cos(pa)*length
        x1 = c[0] + np.cos(pa+np.pi)*length

        y0 = c[1] + np.sin(pa)*length
        y1 = c[1] + np.sin(pa+np.pi)*length

        npts = np.shape(image_data)[0]
        x, y = np.linspace(x0, x1, npts), np.linspace(y0, y1, npts)

        coords = np.array([x,y])
        prof = scipy.ndimage.map_coordinates(image_data, coords, cval=np.nan)
        upper = prof[int(x.size/2):]
        lower = np.flip(prof[:int(x.size/2)])
        r = np.linspace(0, length/2, np.size(upper))
        prof_avg = (upper + lower)/2 


        prof_avg = prof_avg * u.nmgy / pixel_scale**2 # nmgy /pix -> nmgy/arcsec**2

        if surf_bright:
            zero_point_star_equiv = u.zero_point_flux(3631.1 * u.Jy)
            prof_avg = u.Magnitude(prof_avg.to(u.AB, zero_point_star_equiv))

        return {"r": r[prof_avg != np.nan]*pixel_scale, "mu": prof_avg[prof_avg != np.nan].value}
    except:
        print(tb.format_exc())
        return None

def clearLayout(layout):
    if isinstance(layout, QLayout):
        while layout.count():
            item = layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
            else:
                clearLayout(item.layout())