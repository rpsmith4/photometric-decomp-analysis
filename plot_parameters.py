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

import argparse
from pathlib import Path

from photutils.isophote import EllipseGeometry
from photutils.aperture import EllipticalAperture
from photutils.isophote import Ellipse
from photutils.isophote import build_ellipse_model

import astropy.units as u

def parse_results(file):
    model = pyimfit.parse_config_file(file)
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
        if k == 0:
            func_dict["label"] = "Host"
        if k == 1:
            func_dict["label"] = "Polar"
        functions.append(func_dict)
        print(functions)
    
    return None


def main(args):
    p = Path(args.p)
    if args.r:
        structure = os.walk(p)
        for root, dirs, files in structure:
            if not(files == []):
                model_files = sorted(glob.glob(os.path.join(Path(root), "?_fit_params.txt")))
                print(model_files)
                for model in model_files:
                    model = pyimfit.ModelDescription.load(model)
                    print(model)
    else:
        # model_files = sorted(glob.glob(os.path.join(p, "?_fit_params.txt")))
        model_files = sorted(glob.glob(os.path.join(p, "?_fit_params.txt")))
        print(model_files)
        for model_file in model_files:
            # model = pyimfit.ModelDescription.load(model)
            resutls = parse_results(model_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hello")
    parser.add_argument("-p", help="Path to file/folder containing models")
    parser.add_argument("--type", help="Type of polar structure", choices=["ring", "bulge", "halo"])
    parser.add_argument("-r", help="Recursively go into subfolders (assumes that fits data is at the end of the filetree)", action="store_true")

    args = parser.parse_args()
    main(args)