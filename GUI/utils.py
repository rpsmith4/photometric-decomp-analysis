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