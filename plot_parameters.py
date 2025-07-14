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

import sys
IMAN_DIR = os.path.expanduser("~/Documents/iman_new")
sys.path.append(os.path.join(IMAN_DIR, 'decomposition/make_model'))
import make_model_ima_imfit

# warnings.filterwarnings("ignore")

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

def quantities_plot(all_functions):
    # TODO: Will have to account for the different types of fits at some point
    # print(all_functions)
    # df = pd.DataFrame(all_functions)
    df = pd.json_normalize(all_functions)
    df = df.groupby(by="Galaxy", group_keys=True)[df.columns].apply(lambda x: x)
    # TODO: Add each band (maybe overlay them on the same plots, just different colors)
    fig = plt.figure()
    plt.suptitle("Host")
    band_colors = {
        "g": "g",
        "r" : "r",
        "i" : "firebrick",
        "z" : "darkred"
    }
    for band in "griz":
        df_band = df[df["band"] == band].copy()
        host_PA = df_band[df_band["label"] == "Host"]["parameters.PA"]
        host_ax_ratio = df_band[df_band["label"] == "Host"]["b/a"]
        host_I_e = df_band[df_band["label"] == "Host"]["parameters.I_e"]
        host_r_e = df_band[df_band["label"] == "Host"]["parameters.r_e"]
        host_n = df_band[df_band["label"] == "Host"]["parameters.n"]


        plt.subplot(2, 3, 1)
        plt.hist(host_PA, histtype='step', color=band_colors[band], label=band)
        plt.xlabel("PA")
        plt.ylabel("Count")

        plt.subplot(2, 3, 2)
        plt.hist(host_ax_ratio, histtype='step', color=band_colors[band], label=band)
        plt.xlabel("Axis ratio (b/a)")
        plt.ylabel("Count")

        plt.subplot(2, 3, 3)
        plt.hist(host_I_e, histtype='step', color=band_colors[band], label=band)
        plt.xlabel(r"Half light intensity $I_e$")
        plt.ylabel("Count")

        plt.subplot(2, 3, 4)
        plt.hist(host_r_e, histtype='step', color=band_colors[band], label=band)
        plt.xlabel(r"Half light radius $r_e$")
        plt.ylabel("Count")

        ax = plt.subplot(2, 3, 5)
        plt.hist(host_n, histtype='step', color=band_colors[band], label=band)
        plt.xlabel(r"Sersic Index $n$")
        plt.ylabel("Count")
    ax.legend(bbox_to_anchor=(1.15, 1.05))
    plt.tight_layout()
    plt.savefig(os.path.join(Path(args.o), "host.png"))

    fig = plt.figure()
    plt.suptitle("Polar")
    for band in "griz":
        df_band = df[df["band"] == band].copy()
        polar_PA = df_band[df_band["label"] == "Polar"]["parameters.PA"]
        polar_ax_ratio = df_band[df_band["label"] == "Polar"]["b/a"]
        polar_I_e = df_band[df_band["label"] == "Polar"]["parameters.I_e"]
        polar_r_e = df_band[df_band["label"] == "Polar"]["parameters.r_e"]
        polar_n = df_band[df_band["label"] == "Polar"]["parameters.n"]


        plt.subplot(2, 3, 1)
        plt.hist(polar_PA, histtype='step', color=band_colors[band], label=band)
        plt.xlabel("PA")
        plt.ylabel("Count")

        plt.subplot(2, 3, 2)
        plt.hist(polar_ax_ratio, histtype='step', color=band_colors[band], label=band)
        plt.xlabel("Axis ratio (b/a)")
        plt.ylabel("Count")

        plt.subplot(2, 3, 3)
        plt.hist(polar_I_e, histtype='step', color=band_colors[band], label=band)
        plt.xlabel(r"Half light intensity $I_e$")
        plt.ylabel("Count")

        plt.subplot(2, 3, 4)
        plt.hist(polar_r_e, histtype='step', color=band_colors[band], label=band)
        plt.xlabel(r"Half light radius $r_e$")
        plt.ylabel("Count")

        ax = plt.subplot(2, 3, 5)
        plt.hist(polar_n, histtype='step', color=band_colors[band], label=band)
        plt.xlabel(r"Sersic Index $n$")
        plt.ylabel("Count")
    ax.legend(bbox_to_anchor=(1.1, 1.05))
    plt.tight_layout()
    plt.savefig(os.path.join(Path(args.o), "polar.png"))
    
    #     for function in functions:
    #         if function["label"] == "Host":
    #             print("")
    #             # host_PAs.append(function["PA"])
                
    return None

def main(args):
    p = Path(args.p).resolve()
    threshold = args.t # For reduced chi-sq
    all_functions = []
    total_bad_fit = 0
    total_fit = 0
    if args.r:
        structure = os.walk(p)
        for root, dirs, files in structure:
            if not(files == []):
                model_files = sorted(glob.glob(os.path.join(Path(root), "?_fit_params.txt")))
                # print(model_files)
                for model_file in model_files:
                    functions, chi_sq, chi_sq_red, status, status_message = parse_results(model_file, os.path.basename(root))
                    img_file = f"image_{functions[0]["band"]}.fits"
                    psf_file = f"psf_patched_{functions[0]["band"]}.fits"
                    params_file = f"{functions[0]["band"]}_fit_params.txt"
                    os.chdir(Path(root).resolve())
                    try:
                        if args.make_composed and (not(f"{functions[0]["band"]}_composed.fits" in files) or args.overwrite):
                            make_model_ima_imfit.main(img_file, params_file, psf_file, composed_model_file=f"{functions[0]["band"]}_composed.fits", comp_names=["Host", "Polar"])
                    except Exception as e:
                        print(f"Failed for {Path(img_file).stem}")
                        print(e)
                    os.chdir(p)
                    all_functions.extend(functions)
                    
                    total_fit += 1
                    if chi_sq_red > threshold or chi_sq_red != chi_sq_red:
                        print(f"{Path(model_file).resolve().relative_to(p.resolve())} has high reduced chi-sq! ({chi_sq_red} > {threshold})")
                        # warnings.warn(f"{Path(model_file).resolve().relative_to(p.resolve())} has high reduced chi-sq! ({chi_sq_red} > {threshold})")
                        total_bad_fit += 1
    else:
        model_files = sorted(glob.glob(os.path.join(p, "?_fit_params.txt")))
        for model_file in model_files:
            functions, chi_sq, chi_sq_red, status, status_message = parse_results(model_file, os.path.basename(p))
            img_file = f"image_{functions[0]["band"]}.fits"
            psf_file = f"psf_patched_{functions[0]["band"]}.fits"
            params_file = f"{functions[0]["band"]}_fit_params.txt"
            os.chdir(p.resolve())
            try:
                if args.make_composed and (not(f"{functions[0]["band"]}_composed.fits" in files) or args.overwrite):
                    make_model_ima_imfit.main(img_file, params_file, psf_file, composed_model_file=f"{functions[0]["band"]}_composed.fits", comp_names=["Host", "Polar"])
            except Exception as e:
                print(f"Failed for {Path(img_file).stem}")
                print(e)
            all_functions.extend(functions)
            
            total_fit += 1
            if chi_sq_red > threshold:
                warnings.warn(f"{Path(model_file).resolve().relative_to(p.resolve())} has high reduced chi-sq! ({chi_sq_red} > {threshold})")
                total_bad_fit += 1
    if args.plot_stats:
        quantities_plot(all_functions)
    print(f"Total fit: {total_fit}")
    print(f"Total poor fit: {total_bad_fit} ({total_bad_fit/total_fit * 100:.2f}% bad)")

def _warning(
    message,
    category = UserWarning,
    filename = '',
    lineno = -1,
    file = '',
    line = -1):
    if UserWarning:
        print(f"{message}")
    else:
        print(f"{category}: {message} at {lineno}")
warnings.showwarning = _warning
# class customWarning(Warning):
#     def __init__(self, message):
#         self.message = message
#     def __str__(self):
#         return repr(self.message)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hello")
    parser.add_argument("-p", help="Path to file/folder containing models", default=".")
    parser.add_argument("--type", help="Type of polar structure", choices=["ring", "bulge", "halo"])
    parser.add_argument("-r", help="Recursively go into subfolders (assumes that fits data is at the end of the filetree)", action="store_true")
    parser.add_argument("-t", help="Reduced Chi-Sq threshold for a fit to be considered bad",type=float, default=1)
    parser.add_argument("-o", help="Output of overall statistics plot", default=".")
    parser.add_argument("--plot_stats", help="Plot overall statistics", action="store_true")
    parser.add_argument("--make_composed", help="Make a composed image of the galaxy (includes image, model, and components)", action="store_true")
    parser.add_argument("--overwrite", help="Overwrite existing files", action="store_true")
    args = parser.parse_args()
    args.o = Path(args.o).resolve()
    main(args)