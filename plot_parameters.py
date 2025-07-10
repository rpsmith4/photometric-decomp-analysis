import glob
import matplotlib.pyplot as plt
import numpy as np
import os
import math

from astropy.convolution import convolve
from astropy.io import fits

from photutils.background import Background2D, MedianBackground
from photutils.segmentation import detect_sources, make_2dgaussian_kernel, SourceCatalog, deblend_sources

from scipy.ndimage import rotate
import scipy.integrate as integrate
import scipy
import pyimfit
import pandas as pd

import argparse
from pathlib import Path

from photutils.isophote import EllipseGeometry
from photutils.aperture import EllipticalAperture
from photutils.isophote import Ellipse
from photutils.isophote import build_ellipse_model

import astropy.units as u
import warnings
warnings.filterwarnings("ignore")

def parse_results(file, galaxy_name):
    model = pyimfit.parse_config_file(file)
    band = Path(file).stem[0]
    # print(model.functionLabelList())
    # print(model.functionNameList())
    # print(model.functionList()[i].parameterList()[0].value)
    with open(file, "r") as f:
        lines = f.readlines()
    status = lines[5].split(" ")[7]
    status_message = " ".join(lines[5].split(" ")[9:])
    # print(status)
    # print(status_message)

    chi_sq = float(lines[7].split(" ")[-1])
    chi_sq_red = float(lines[8].split(" ")[-1])
    functions = []
    for k, function in enumerate(model.functionList()):
        # if k == 0: # Assume this is the host
        #     print("Host:")
        # elif k == 1: # Assume this is the polar 
        #     print("Polar:")
        # funcname = function._funcName
        # if funcname == "Sersic":
        #     params = function.parameterList()
        #     PA = params[0]
        #     print(PA)
        # for j, parameter in enumerate(function.parameterList()):
        #         print(f"{parameter.name} : {parameter.value}")
        
        func_dict = function.getFunctionAsDict()
        for param in func_dict["parameters"]:
            func_dict["parameters"][param] = func_dict["parameters"][param][0]
        if k == 0:
            func_dict["label"] = "Host"
        if k == 1:
            func_dict["label"] = "Polar"
        func_dict["band"] = band
        func_dict["Galaxy"] = galaxy_name
        # TODO: Get other parameters here (or somewhere somehow)
        e = func_dict["parameters"]["ell"]
        axis_ratio = np.sqrt(-1*(np.square(e) - 1)) # b/a ratio
        func_dict["b/a"] = axis_ratio
        functions.append(func_dict)
    
    return functions, chi_sq, chi_sq_red, status, status_message

def plot(all_functions):
    # TODO: Will have to account for the different types of fits at some point
    # print(all_functions)
    # df = pd.DataFrame(all_functions)
    df = pd.json_normalize(all_functions)
    df = df.groupby(by="Galaxy", group_keys=True)[df.columns].apply(lambda x: x)
    host_PA = df[df["label"] == "Host"]["parameters.PA"]
    plt.hist(host_PA, density=True, histtype='step', facecolor='g')
    plt.savefig("test.png")
    
    #     for function in functions:
    #         if function["label"] == "Host":
    #             print("")
    #             # host_PAs.append(function["PA"])
                
    return None

def main(args):
    p = Path(args.p)
    threshold = 1 # For reduced chi-sq
    if args.r:
        structure = os.walk(p)
        for root, dirs, files in structure:
            if not(files == []):
                model_files = sorted(glob.glob(os.path.join(Path(root), "?_fit_params.txt")))
                # print(model_files)
                for model_file in model_files:
                    functions, chi_sq, chi_sq_red, status, status_message = parse_results(model_file, root.base_name())
    else:
        model_files = sorted(glob.glob(os.path.join(p, "?_fit_params.txt")))
        all_functions = []
        for model_file in model_files:
            functions, chi_sq, chi_sq_red, status, status_message = parse_results(model_file, os.path.basename(p))
            all_functions.extend(functions)
            
            threshold = 1
            if chi_sq_red > threshold:
                warnings.warn(f"{Path(model_file).resolve().relative_to(p.resolve())} has high reduced chi-sq! ({chi_sq_red} > {threshold})")
        plot(all_functions)

def _warning(
    message,
    category = UserWarning,
    filename = '',
    lineno = -1,
    file = '',
    line = -1):
    print(message)
warnings.showwarning = _warning

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hello")
    parser.add_argument("-p", help="Path to file/folder containing models")
    parser.add_argument("--type", help="Type of polar structure", choices=["ring", "bulge", "halo"])
    parser.add_argument("-r", help="Recursively go into subfolders (assumes that fits data is at the end of the filetree)", action="store_true")

    args = parser.parse_args()
    main(args)