import glob
import matplotlib.pyplot as plt
import numpy as np
import os
import math

from astropy.convolution import convolve
from astropy.io import fits
from astropy.constants import c
from astropy.table import Table

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

from get_flux_ratios import get_flux_ratios

# warnings.filterwarnings("ignore")

def nmgy2ABmag(nmgy):
    return 22.5 - 2.5 * np.log10(nmgy)

def parse_results(file, galaxy_name, galaxy_type, table=None):
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
    # print(status)
    # print(status_message)
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
        func_dict["Galaxy"] = galaxy_name
        func_dict["Galaxy_type"] = galaxy_type
        # TODO: Get other parameters here (or somewhere somehow)
        if table:
            H_0 = 70.8 * u.km / u.s / u.Mpc
            z = table["GALAXY" == galaxy_name]["Z_LEDA"]
            if z == -1:
                func_dict["Distance"] = -1 # SGA sets z to -1 if z is not measured
            else:
                d = (z * c / H_0).to(u.Mpc)
                func_dict["Distance"] = d.value
        else:
            func_dict["Distance"] = -1
            
        e = func_dict["parameters"]["ell"]
        axis_ratio = np.sqrt(-1*(np.square(e) - 1)) # b/a ratio
        func_dict["b/a"] = axis_ratio
        flux_ratios = get_flux_ratios(model_file=file)
        if func_dict["label"] == "Host":
            func_dict["flux_ratio"] = flux_ratios["Host"]
        elif func_dict["label"] == "Polar":
            func_dict["flux_ratio"] = flux_ratios["Polar"]

        functions.append(func_dict)
    
    return functions, chi_sq, chi_sq_red, status, status_message

def quantities_plot(all_functions):
    # TODO: Will have to account for the different types of fits at some point
    # print(all_functions)
    # df = pd.DataFrame(all_functions)
    df = pd.json_normalize(all_functions)
    df = df.groupby(by="Galaxy", group_keys=True)[df.columns].apply(lambda x: x)
    df = Table.from_pandas(df)
    df = df[df["Distance"] != -1]
    df = df[df["flux_ratio"] != -1]
    fig = plt.figure()
    band_colors = {
        "g": "g",
        "r" : "r",
        "i" : "firebrick",
        "z" : "blueviolet"
    }
    if args.plot_type == "compare_structure":
        fig = plt.figure(figsize=(16, 8))
        plt.suptitle("Host")
        for band in "griz":
            df_band = df[df["band"] == band].copy()
            host_ax_ratio = df_band[df_band["label"] == "Host"]["b/a"]
            host_I_e = df_band[df_band["label"] == "Host"]["parameters.I_e"] * u.nmgy / u.pix
            host_r_e = df_band[df_band["label"] == "Host"]["parameters.r_e"] * u.pix
            host_n = df_band[df_band["label"] == "Host"]["parameters.n"]
            host_flux_ratio = df_band[df_band["label"] == "Host"]["flux_ratio"]
            d = df_band[df_band["label"] == "Host"]["Distance"] * u.Mpc
            host_PA = df_band[df_band["label"] == "Host"]["parameters.PA"]
            polar_PA = df_band[df_band["label"] == "Polar"]["parameters.PA"]
            diff_PA = host_PA - polar_PA
            diff_PA = np.abs(diff_PA)

            pixscale = 0.262 * u.arcsec / u.pix

            host_r_e = (np.tan(host_r_e * pixscale) * d).to(u.kpc)
            host_I_e = nmgy2ABmag((host_I_e * pixscale).value)


            ax = plt.subplot(2, 3, 1)
            plt.hist(diff_PA, histtype='step', color=band_colors[band], label=band)
            plt.xlabel(r"$PA_{host} - PA_{polar}$ (deg)")
            plt.ylabel("Count")

            plt.subplot(2, 3, 2)
            plt.hist(host_ax_ratio, histtype='step', color=band_colors[band], label=band)
            plt.xlabel("Axis ratio (b/a)")
            plt.ylabel("Count")

            plt.subplot(2, 3, 3)
            plt.hist(host_I_e, histtype='step', color=band_colors[band], label=band)
            plt.xlabel(r"Half light intensity $I_e$ (AB Mag / arcsec)")
            plt.ylabel("Count")

            plt.subplot(2, 3, 4)
            plt.hist(host_r_e, histtype='step', color=band_colors[band], label=band)
            plt.xlabel(r"Half light radius $r_e$ (kpc)")
            plt.ylabel("Count")

            plt.subplot(2, 3, 5)
            plt.hist(host_n, histtype='step', color=band_colors[band], label=band)
            plt.xlabel(r"Sersic Index $n$")
            plt.ylabel("Count")

            plt.subplot(2, 3, 6)
            plt.hist(np.array(host_flux_ratio), histtype='step', color=band_colors[band], label=band)
            plt.xlabel(r"Flux Ratio $f_{Host}/f_{Polar}$")
            plt.ylabel("Count")

        # ax.legend(bbox_to_anchor=(1.15, 1.05))
        ax.legend(loc="upper left")
        plt.tight_layout()
        plt.savefig(os.path.join(Path(args.o), "host.png"))

        fig = plt.figure(figsize=(16, 8))
        plt.suptitle("Polar")
        for band in "griz":
            df_band = df[df["band"] == band].copy()
            polar_ax_ratio = df_band[df_band["label"] == "Polar"]["b/a"]
            polar_I_e = df_band[df_band["label"] == "Polar"]["parameters.I_e"] * u.nmgy / u.pix
            polar_flux_ratio = df_band[df_band["label"] == "Polar"]["flux_ratio"]
            polar_r_e = df_band[df_band["label"] == "Polar"]["parameters.r_e"] * u.pix
            polar_n = df_band[df_band["label"] == "Polar"]["parameters.n"]
            d = df_band[df_band["label"] == "Host"]["Distance"] * u.Mpc
            host_PA = df_band[df_band["label"] == "Host"]["parameters.PA"]
            polar_PA = df_band[df_band["label"] == "Polar"]["parameters.PA"]
            diff_PA = host_PA - polar_PA
            diff_PA = np.abs(diff_PA)

            polar_r_e = (np.tan(polar_r_e * pixscale) * d).to(u.kpc)
            polar_I_e = nmgy2ABmag((polar_I_e * pixscale).value)

            plt.subplot(2, 3, 1)
            plt.hist(diff_PA, histtype='step', color=band_colors[band], label=band)
            plt.xlabel(r"$PA_{host} - PA_{polar}$ (deg)")
            plt.ylabel("Count")

            plt.subplot(2, 3, 2)
            plt.hist(polar_ax_ratio, histtype='step', color=band_colors[band], label=band)
            plt.xlabel("Axis ratio (b/a)")
            plt.ylabel("Count")

            plt.subplot(2, 3, 3)
            plt.hist(polar_I_e, histtype='step', color=band_colors[band], label=band)
            plt.xlabel(r"Half light intensity $I_e$ (AB Mag / arcsec)")
            plt.ylabel("Count")

            plt.subplot(2, 3, 4)
            plt.hist(polar_r_e, histtype='step', color=band_colors[band], label=band)
            plt.xlabel(r"Half light radius $r_e$ (kpc)")
            plt.ylabel("Count")

            ax = plt.subplot(2, 3, 5)
            l = plt.hist(polar_n, histtype='step', color=band_colors[band], label=band)
            plt.xlabel(r"Sersic Index $n$")
            plt.ylabel("Count")

            plt.subplot(2, 3, 6)
            plt.hist(np.array(polar_flux_ratio), histtype='step', color=band_colors[band], label=band)
            plt.xlabel(r"Flux Ratio $f_{Host}/f_{Polar}$")
            plt.ylabel("Count")

        ax.legend(bbox_to_anchor=(1.15, 1.05))
        plt.tight_layout()
        plt.savefig(os.path.join(Path(args.o), "polar.png"))

    elif args.plot_type == "compare_type":
        for galaxy_type in ["ring", "bulge", "halo"]:
            fig = plt.figure(figsize=(16, 8))
            # plt.suptitle(f"Host (Polar {galaxy_type})")
            for structure in ["Host", "Polar"]:
                for band in "griz":
                    df_type = df[df["Galaxy_type"] == galaxy_type].copy()
                    df_band = df_type[df_type["band"] == band]
                    ax_ratio = df_band[df_band["label"] == structure]["b/a"]
                    I_e = df_band[df_band["label"] == structure]["parameters.I_e"] * u.nmgy
                    flux_ratio = df_band[df_band["label"] == "Host"]["flux_ratio"]
                    r_e = df_band[df_band["label"] == structure]["parameters.r_e"] * u.pix
                    n = df_band[df_band["label"] == structure]["parameters.n"]
                    d = df_band[df_band["label"] == structure]["Distance"] * u.Mpc

                    host_PA = df_band[df_band["label"] == "Host"]["parameters.PA"]
                    polar_PA = df_band[df_band["label"] == "Polar"]["parameters.PA"]
                    diff_PA = host_PA - polar_PA
                    diff_PA = np.abs(diff_PA)

                    pixscale = 0.262 * u.arcsec / u.pix

                    r_e = (np.tan(r_e * pixscale) * d).to(u.kpc)
                    I_e = nmgy2ABmag((I_e * pixscale).value)

                    if structure == "Host":
                        label = band
                        ls = "-"
                    else:
                        label = None
                        ls = "--"
                    

                    ax = plt.subplot(2, 3, 1)
                    plt.hist(diff_PA, histtype='step', color=band_colors[band], label=label, ls=ls)
                    plt.xlabel(r"$PA_{host} - PA_{polar}$ (deg)")
                    plt.ylabel("Count")

                    plt.subplot(2, 3, 2)
                    plt.hist(ax_ratio, histtype='step', color=band_colors[band], label=label, ls=ls)
                    plt.xlabel("Axis ratio (b/a)")
                    plt.ylabel("Count")

                    plt.subplot(2, 3, 3)
                    plt.hist(I_e, histtype='step', color=band_colors[band], label=label, ls=ls)
                    plt.xlabel(r"Half light intensity $I_e$ (AB Mag / arcsec)")
                    plt.ylabel("Count")

                    plt.subplot(2, 3, 4)
                    plt.hist(r_e, histtype='step', color=band_colors[band], label=label, ls=ls)
                    plt.xlabel(r"Half light radius $r_e$ (kpc)")
                    plt.ylabel("Count")

                    plt.subplot(2, 3, 5)
                    plt.hist(n, histtype='step', color=band_colors[band], label=label, ls=ls)
                    plt.xlabel(r"Sersic Index $n$")
                    plt.ylabel("Count")

                    plt.subplot(2, 3, 6)
                    plt.hist(np.array(flux_ratio), histtype='step', color=band_colors[band], label=label, ls=ls)
                    plt.xlabel(r"Flux Ratio $f_{Host}/f_{Polar}$")
                    plt.ylabel("Count")


                # ax.legend(bbox_to_anchor=(1.15, 1.05))
                ax.legend(loc="upper left")
                plt.tight_layout()
                plt.savefig(os.path.join(Path(args.o), f"host_{galaxy_type}.png"))
    return None

def get_functions_from_files(root, galaxy_type, table=None):
    threshold = args.t # For reduced chi-sq
    model_files = sorted(glob.glob(os.path.join(Path(root), f"{args.fit_type}_?_fit_params.txt")))
    global total_fit
    global total_bad_fit
    global bound_sticking
    # print(model_files)
    if not args.dont_exclude:
        analysis_path = "/home/ryans/Documents/Photometric Decomp/Analysis/"
        f = open(analysis_path + "exclude.txt", "r")
        gals_to_exclude = f.readlines()
        gals_to_exclude = [gal_to_exclude.strip("\n") for gal_to_exclude in gals_to_exclude]
    files = os.listdir(root)
    for model_file in model_files:
        if not args.dont_exclude:
            if any(gal_to_exclude in model_file for gal_to_exclude in gals_to_exclude):
                continue
        functions, chi_sq, chi_sq_red, status, status_message = parse_results(model_file, os.path.basename(root), galaxy_type, table)
        root = Path(root).resolve()
        img_file = f"image_{functions[0]['band']}.fits"
        psf_file = f"psf_patched_{functions[0]['band']}.fits"
        params_file = f"{args.fit_type}_{functions[0]['band']}_fit_params.txt"
        mask_file = f"image_mask.fits"
        os.chdir(root)
        try:
            if args.make_composed and (not(f"{args.fit_type}_{functions[0]['band']}_composed.fits" in files) or args.overwrite):
                components = {
                    "2_sersic" : ["Host", "Polar"],
                    "3_sersic" : ["Core", "Wings", "Polar"],
                    "1_sersic_1_gauss_ring" : ["Host", "Polar"]
                }
                if args.mask:
                    img_dat = fits.open(img_file)
                    img = img_dat[0].data
                    mask = fits.open(mask_file)[0].data
                    img = img * (1 - mask)
                    fits.writeto("masked.fits", data=img, header=img_dat[0].header)

                    make_model_ima_imfit.main("masked.fits", params_file, psf_file, composed_model_file=f"{args.fit_type}_{functions[0]['band']}_composed.fits", comp_names=components[args.fit_type])
                    os.remove("./masked.fits")
                else:
                    make_model_ima_imfit.main(img_file, params_file, psf_file, composed_model_file=f"{args.fit_type}_{functions[0]['band']}_composed.fits", comp_names=components[args.fit_type])

        except Exception as e:
            print(f"Failed for {Path(img_file)}")
            print(e)


        all_functions.extend(functions)
        total_fit += 1
        if chi_sq_red > threshold or chi_sq_red != chi_sq_red:
            name = Path(model_file).resolve().relative_to(p)
            if args.v: print(f"{name} has high reduced chi-sq! ({chi_sq_red} > {threshold})")
            # warnings.warn(f"{Path(model_file).resolve().relative_to(p.resolve())} has high reduced chi-sq! ({chi_sq_red} > {threshold})")
            total_bad_fit += 1
        bounds_stick = False
        for function in functions:
            for param in function["parameters_unc"].keys():
                # if param not in ["ell", "n", "r_e"]:
                if function["parameters_unc"][param] == 0:
                    name = Path(model_file).resolve().relative_to(p)
                    if args.vvv: print(f"Zero uncertainty for {param} in {name} (possibly sticking to bounds)!")
                    bounds_stick = True
                    bound_sticking += 1
        name = Path(model_file).resolve().relative_to(p)
        if args.vv and bounds_stick: print(f"{name} has sticking bounds!")

    return all_functions

def main(args):
    global p
    p = Path(args.p).resolve()
    table = Table.read(os.path.join(args.c, "SGA-2020.fits"))
    global total_bad_fit
    global total_fit
    global bound_sticking
    total_bad_fit = 0
    total_fit = 0
    bound_sticking = 0
    global all_functions
    all_functions = []
    if args.r:
        structure = os.walk(p)
        for root, dirs, files in structure:
            if not(files == []):
                folder_type_dict = {
                    "Polar Rings": "ring",
                    "Polar_Tilted Bulges": "bulge",
                    "Polar_Tilted Halo": "halo"
                }
                galaxy_type = None
                for folder in ["Polar Rings", "Polar_Tilted Bulges", "Polar_Tilted Halo"]: # Attempt to autodetect type
                    if folder in root:
                        galaxy_type = folder_type_dict[folder]
                # model_files = sorted(glob.glob(os.path.join(Path(root), "?_fit_params.txt")))
                if not(galaxy_type == None):
                    all_functions = get_functions_from_files(Path(root).resolve(), galaxy_type, table)
    else:
        # model_files = sorted(glob.glob(os.path.join(p, "?_fit_params.txt")))
        all_functions = get_functions_from_files(root=Path(p).resolve(), galaxy_type=None, table=table)

    print(f"Total fit: {total_fit}")
    print(f"Total poor fit: {total_bad_fit} ({total_bad_fit/total_fit * 100:.2f}% bad)")
    print(f"Total parameter bounds sticking: {bound_sticking}")

    if args.plot_stats:
        quantities_plot(all_functions)

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
    parser.add_argument("-c", help="Directory to Sienna Galaxy Atlas File (used to get redshift)", default=".")
    parser.add_argument("--fit_type", help="Type of fit done", choices=["2_sersic", "1_sersic_1_gauss_ring", "3_sersic"], default="2_sersic")
    parser.add_argument("--mask", help="Use mask on the original image", action="store_true")
    parser.add_argument("--plot_type", help="Type of plots to make", choices=["compare_structure", "compare_type"], default="compare_structure")
    parser.add_argument("--dont_exclude", help="Don't exclude galaxies in the 'exclude.txt' file", action="store_true")
    parser.add_argument("-v", help="Show chi-sq warnings for fits", action="store_true")
    parser.add_argument("-vv", help="Show chi-sq and parameter bounds warnings for fits", action="store_true")
    parser.add_argument("-vvv", help="Show chi-sq and parameter bounds (specific) warnings for fits", action="store_true")
    args = parser.parse_args()
    args.o = Path(args.o).resolve()
    main(args)