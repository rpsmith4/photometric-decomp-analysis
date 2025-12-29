import glob
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import os

from astropy.io import fits
from astropy.constants import c
from astropy.table import Table

import pyimfit
import pandas as pd

import argparse
from pathlib import Path

import astropy.units as u
import subprocess

import sys

IMAN_DIR = Path(os.path.dirname(__file__))
sys.path.append(os.path.join(IMAN_DIR, 'iman_new/decomposition/make_model'))
import make_model_ima_imfit

# warnings.filterwarnings("ignore")

# TODO: Change nmgy to "flux"
# Range should be -19 to -23 for abs mag
def flux2ABmag(flux):
    return 22.5 - 2.5 * np.log10(flux)

def get_flux(model_file):
    result = subprocess.run(["makeimage", f"{model_file}", "--print-fluxes"], capture_output=True)
    # print(result.stdout.decode("utf-8"))

    lines = result.stdout.decode("utf-8").split("\n")
    flux_ratios = {}
    flux = {}
    for line in lines:
        line = line.split(" ")
        line = list(filter(None, line))

        if "Host" in line:
            flux_ratios["Host"] = float(line[3])
            flux["Host"] = float(line[1])
            if float(line[3]) != float(line[3]):
                print(f"Nan flux ratio for {model_file}")
        elif "Polar" in line:
            flux_ratios["Polar"] = float(line[3])
            flux["Polar"] = float(line[1])
    if flux_ratios == {}:
        print(f"Unable to determine flux ratios for {model_file}")
        flux_ratios = {
            "Host": -1,
            "Polar": -1
        }
    return flux_ratios, flux

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
        
        # Gets other parameters from result
        if table:
            H_0 = 70.8 * u.km / u.s / u.Mpc
            z = table[table["GALAXY"] == galaxy_name]["Z_LEDA"]
            if z == -1:
                func_dict["Distance"] = -1 # SGA sets z to -1 if z is not measured
            else:
                d = (z * c / H_0).to(u.Mpc)
                func_dict["Distance"] = d.value
        else:
            func_dict["Distance"] = -1
            
        e = func_dict["parameters"]["ell"]
        # axis_ratio = np.sqrt(-1*(np.square(e) - 1)) # b/a ratio # THIS IS WRONG I THOUGH IT WAS ECCENTRICITY BUT ITS ELLIPCTICITY
        axis_ratio = -1*(e - 1)# f=1-b/a
        func_dict["b/a"] = axis_ratio
        flux_ratios, flux = get_flux(model_file=file)
        if func_dict["label"] == "Host":
            func_dict["flux_ratio"] = flux_ratios["Host"]
            func_dict["flux"] = flux["Host"]
        elif func_dict["label"] == "Polar":
            func_dict["flux_ratio"] = flux_ratios["Polar"]
            func_dict["flux"] = flux["Polar"]

        functions.append(func_dict)
    
    return functions, chi_sq, chi_sq_red, status, status_message

def quantities_plot(all_functions):
    # TODO: Will probably have to account for the different types of fits at some point, can only really handle
    # 2 sersic models at the moment
    df = pd.json_normalize(all_functions)
    df = df.groupby(by="Galaxy", group_keys=True)[df.columns].apply(lambda x: x)
    df = Table.from_pandas(df)
    # df = df[df["Distance"] != -1]
    df = df[df["flux_ratio"] != -1]
    fig = plt.figure()
    band_colors = {
        "g": "g",
        "r" : "r",
        "i" : "firebrick",
        "z" : "blueviolet"
    }
    if args.plot_type == "compare_structure":
        for structure in ["Host", "Polar"]:
            plt.rc('axes', labelsize=16)
            plt.rc('legend', fontsize=14)
            plt.rc('figure', titlesize=20)
            plt.rc('xtick', labelsize=14)
            plt.rc('ytick', labelsize=14)
            fig = plt.figure(figsize=(16, 8))
            gs = gridspec.GridSpec(2, 4)
            gs.update(wspace=0.5)
            plt.suptitle(f"{structure} Component Properties", y=0.93)
            for band in "griz":
                df_band = df[df["band"] == band]
                ax_ratio = df_band[df_band["label"] == structure]["b/a"]
                I_e = df_band[df_band["label"] == structure]["parameters.I_e"] * u.nmgy
                flux_ratio = df_band[df_band["label"] == "Host"]["flux_ratio"]
                flux = df_band[df_band["label"] == structure]["flux"]
                r_e = df_band[df_band["label"] == structure]["parameters.r_e"] * u.pix
                n = df_band[df_band["label"] == structure]["parameters.n"]
                d = df_band[df_band["label"] == structure]["Distance"] * u.Mpc

                pixscale = 0.262 * u.arcsec / u.pix

                r_e = np.squeeze(np.array(r_e[d != -1]), axis=0) * u.pix # For some reason increases the dimension by one
                r_e = (np.tan(r_e * pixscale) * d).to(u.kpc).value
                I_e = flux2ABmag((I_e * pixscale).value)
                flux = np.squeeze(np.array(flux[d != -1]), axis=0) # For some reason increases the dimension by one
                app_mag = flux2ABmag(flux) # Apparent Magnitude
                abs_mag = app_mag - 5 * np.log10((d.to(u.pc).value/10))

                ls = "-"
                label = band
                color = band_colors[band]

                # ax = plt.subplot(2, 3, 1)
                # plt.hist(diff_PA, histtype='step', color=band_colors[band], label=band)
                # plt.xlabel(r"$PA_{host} - PA_{polar}$ (deg)")
                # plt.ylabel("Count")

                # ax = plt.subplot(2, 2, 1)
                # plt.hist(ax_ratio, histtype='step', color=band_colors[band], label=band)
                # plt.xlabel("Axis ratio (b/a)")
                # plt.ylabel("Count")

                # plt.subplot(2, 2, 2)
                # plt.hist(I_e, histtype='step', color=band_colors[band], label=band)
                # plt.xlabel(r"Half light intensity $I_e$ (AB Mag / arcsec)")
                # plt.ylabel("Count")

                # plt.subplot(2, 2, 3)
                # plt.hist(r_e, histtype='step', color=band_colors[band], label=band)
                # plt.xlabel(r"Half light radius $r_e$ (kpc)")
                # plt.ylabel("Count")

                # plt.subplot(2, 2, 4)
                # plt.hist(n, histtype='step', color=band_colors[band], label=band)
                # plt.xlabel(r"Sersic Index $n$")
                # plt.ylabel("Count")

                # plt.subplot(2, 3, 6)
                # plt.hist(np.array(flux_ratio), histtype='step', color=band_colors[band], label=band)
                # plt.xlabel(r"Flux Ratio $f_{Host}/f_{Polar}$")
                # plt.ylabel("Count")

                ngalaxies = np.size(ax_ratio)
                ax1 = plt.subplot(gs[0, :2])
                bins = np.arange(0, 1 + 0.05, 0.05)
                ax_ratio_binned, bins = np.histogram(ax_ratio, bins=bins)
                plt.stairs(ax_ratio_binned/ngalaxies, edges=bins, color=color, label=label, ls=ls, linewidth=2)
                plt.xlabel("Axis ratio (b/a)")
                plt.ylabel("Fraction of Sample")

                ax2 = plt.subplot(gs[0, 2:])
                bins = np.arange(0, 8 + 0.5, 0.5)
                n_binned, bins = np.histogram(n, bins=bins)
                plt.stairs(n_binned/ngalaxies, edges=bins, color=color, label=label, ls=ls, linewidth=2)
                plt.xlabel(r"Sersic Index $n$")
                plt.ylabel("Fraction of Sample")

                ngalaxies = np.size(r_e) # May be different due to removing the d=-1 cases here
                if structure == "Polar":
                    ax3 = plt.subplot(gs[1, 1:3])
                    bins = np.arange(-23, -13 + 1, 1)
                    abs_mag_binned, bins = np.histogram(abs_mag, bins=bins)
                    plt.stairs(abs_mag_binned/ngalaxies, edges=bins, color=color, label=label, ls=ls, linewidth=2)
                    plt.xlabel(r"Absolute Magnitude")
                    plt.ylabel("Fraction of Sample")

                elif structure == "Host":
                    ax3 = plt.subplot(gs[1, :2])
                    bins = np.arange(-23, -13 + 1, 1)
                    abs_mag_binned, bins = np.histogram(abs_mag, bins=bins)
                    plt.stairs(abs_mag_binned/ngalaxies, edges=bins, color=color, label=label, ls=ls, linewidth=2)
                    plt.xlabel(r"Absolute Magnitude")
                    plt.ylabel("Fraction of Sample")
                    
                    ax4 = plt.subplot(gs[1, 2:])
                    bins = np.arange(0, 11 + 0.5, 0.5)
                    r_e_binned, bins = np.histogram(r_e, bins=bins)
                    plt.stairs(r_e_binned/ngalaxies, edges=bins, color=color, label=label, ls=ls, linewidth=2)
                    plt.xlabel(r"Effective Radius $r_e$ (kpc)")
                    plt.ylabel("Fraction of Sample")

            # ax.legend(bbox_to_anchor=(1.15, 1.05))
            ax1.legend(loc="upper left")
            plt.tight_layout()
            plt.savefig(os.path.join(Path(args.o), f"{structure}_plot.png"))

        plt.rc('axes', labelsize=24)
        plt.rc('legend', fontsize=18)
        plt.rc('figure', titlesize=28)
        plt.rc('xtick', labelsize=20)
        plt.rc('ytick', labelsize=20)
        fig = plt.figure(figsize=(8, 12))
        plt.suptitle(f"Host and Polar Comparisons")
        for band in "griz":
            df_band = df[df["band"] == band]
            flux_ratio = df_band[df_band["label"] == "Host"]["flux_ratio"]
            host_PA = df_band[df_band["label"] == "Host"]["parameters.PA"]
            polar_PA = df_band[df_band["label"] == "Polar"]["parameters.PA"]
            diff_PA = host_PA - polar_PA
            diff_PA = np.abs(diff_PA)

            ls = "-"
            label = band
            color = band_colors[band]

            ngalaxies = np.size(host_PA)
            ax1 = plt.subplot(2, 1, 1)
            bins = np.arange(0, 180 + 15, 15)
            diff_PA_binned, bins = np.histogram(diff_PA, bins=bins)
            plt.stairs(diff_PA_binned/ngalaxies, edges=bins, color=color, label=label, ls=ls, linewidth=3)
            plt.xlabel(r"$PA_{host} - PA_{polar}$ (deg)")
            plt.ylabel("Fraction of Sample")

            ax2 = plt.subplot(2, 1, 2)
            bins = np.arange(0, 1 + 0.1, 0.1)
            flux_ratio_binned, bins = np.histogram(np.array(flux_ratio), bins=bins)
            plt.stairs(flux_ratio_binned/ngalaxies, edges=bins, color=color, label=label, ls=ls, linewidth=3)
            plt.xlabel(r"Flux Ratio $f_{Host}/f_{Total}$")
            plt.ylabel("Fraction of Sample")


        ax1.legend(loc="upper left")
        plt.tight_layout()
        plt.savefig(os.path.join(Path(args.o), "extra_overall.png"))

    elif args.plot_type == "compare_type":
        from matplotlib.ticker import MaxNLocator
        # plt.rc('axes', titlesize=8)


        galaxy_plot_style = {
            "ring": ["Polar Ring/Disk", "red", "-"],
            "bulge": ["Polar Bulge", "blue", "--"],
            "halo": ["Polar Halo", "green", ":"]
        }
        for structure in ["Host", "Polar"]:
            plt.rc('axes', labelsize=16)
            plt.rc('legend', fontsize=14)
            plt.rc('figure', titlesize=20)
            plt.rc('xtick', labelsize=14)
            plt.rc('ytick', labelsize=14)
            fig = plt.figure(figsize=(16, 8))
            gs = gridspec.GridSpec(2, 4)
            gs.update(wspace=0.5)
            plt.suptitle(f"{structure} Component Properties", y=0.93)
            for galaxy_type in ["ring", "bulge", "halo"]:
                # for band in "griz":
                for band in "g":
                    df_type = df[df["Galaxy_type"] == galaxy_type].copy()
                    df_band = df_type[df_type["band"] == band]
                    ax_ratio = df_band[df_band["label"] == structure]["b/a"]
                    I_e = df_band[df_band["label"] == structure]["parameters.I_e"] * u.nmgy
                    flux_ratio = df_band[df_band["label"] == "Host"]["flux_ratio"]
                    flux = df_band[df_band["label"] == structure]["flux"]
                    r_e = df_band[df_band["label"] == structure]["parameters.r_e"] * u.pix
                    n = df_band[df_band["label"] == structure]["parameters.n"]
                    d = df_band[df_band["label"] == structure]["Distance"] * u.Mpc

                    host_PA = df_band[df_band["label"] == "Host"]["parameters.PA"]
                    polar_PA = df_band[df_band["label"] == "Polar"]["parameters.PA"]
                    diff_PA = host_PA - polar_PA
                    diff_PA = np.abs(diff_PA)

                    pixscale = 0.262 * u.arcsec / u.pix

                    r_e = np.squeeze(np.array(r_e[d != -1]), axis=0) * u.pix # For some reason increases the dimension by one
                    r_e = (np.tan(r_e * pixscale) * d).to(u.kpc).value
                    I_e = flux2ABmag((I_e * pixscale).value)
                    flux = np.squeeze(np.array(flux[d != -1]), axis=0) # For some reason increases the dimension by one
                    app_mag = flux2ABmag(flux) # Apparent Magnitude
                    abs_mag = app_mag - 5 * np.log10((d.to(u.pc).value/10))
                    
                    label = galaxy_plot_style[galaxy_type][0]
                    color = galaxy_plot_style[galaxy_type][1]
                    ls = galaxy_plot_style[galaxy_type][2]
                    

                    # ax = plt.subplot(2, 3, 1)
                    # plt.hist(diff_PA, histtype='step', color=band_colors[band], label=label, ls=ls)
                    # plt.xlabel(r"$PA_{host} - PA_{polar}$ (deg)")
                    # plt.ylabel("Count")

                    # plt.subplot(2, 3, 2)
                    # plt.hist(ax_ratio, histtype='step', color=band_colors[band], label=label, ls=ls)
                    # plt.xlabel("Axis ratio (b/a)")
                    # plt.ylabel("Count")

                    # plt.subplot(2, 3, 3)
                    # plt.hist(I_e, histtype='step', color=band_colors[band], label=label, ls=ls)
                    # plt.xlabel(r"Half light intensity $I_e$ (AB Mag / arcsec)")
                    # plt.ylabel("Count")

                    # plt.subplot(2, 3, 4)
                    # plt.hist(r_e, histtype='step', color=band_colors[band], label=label, ls=ls)
                    # plt.xlabel(r"Half light radius $r_e$ (kpc)")
                    # plt.ylabel("Count")

                    # plt.subplot(2, 3, 5)
                    # plt.hist(n, histtype='step', color=band_colors[band], label=label, ls=ls)
                    # plt.xlabel(r"Sersic Index $n$")
                    # plt.ylabel("Count")

                    # plt.subplot(2, 3, 6)
                    # plt.hist(np.array(flux_ratio), histtype='step', color=band_colors[band], label=label, ls=ls)
                    # plt.xlabel(r"Flux Ratio $f_{Host}/f_{Polar}$")
                    # plt.ylabel("Count")
                    
                    ngalaxies = np.size(ax_ratio)
                    ax1 = plt.subplot(gs[0, :2])
                    bins = np.arange(0, 1 + 0.05, 0.05)
                    ax_ratio_binned, bins = np.histogram(ax_ratio, bins=bins)
                    # plt.gca().yaxis.set_major_locator(MaxNLocator(nbins=10))
                    # ax1.locator_params(axis='both', nbins=4)
                    plt.stairs(ax_ratio_binned/ngalaxies, edges=bins, color=color, label=label, ls=ls, linewidth=2)
                    plt.xlabel("Axis ratio (b/a)")
                    plt.ylabel("Fraction of Sample")

                    ax2 = plt.subplot(gs[0, 2:])
                    bins = np.arange(0, 8 + 0.5, 0.5)
                    n_binned, bins = np.histogram(n, bins=bins)
                    plt.stairs(n_binned/ngalaxies, edges=bins, color=color, label=label, ls=ls, linewidth=2)
                    plt.xlabel(r"Sersic Index $n$")
                    plt.ylabel("Fraction of Sample")

                    ngalaxies = np.size(r_e) # May be different due to removing the d=-1 cases here
                    if structure == "Polar":
                        ax3 = plt.subplot(gs[1, 1:3])
                        bins = np.arange(-23, -13 + 1, 1)
                        abs_mag_binned, bins = np.histogram(abs_mag, bins=bins)
                        plt.stairs(abs_mag_binned/ngalaxies, edges=bins, color=color, label=label, ls=ls, linewidth=2)
                        plt.xlabel(r"Absolute Magnitude")
                        plt.ylabel("Fraction of Sample")
                        # ax1.yaxis.set_major_locator(MaxNLocator(integer=True))
                        # ax2.yaxis.set_major_locator(MaxNLocator(integer=True))
                        # ax3.yaxis.set_major_locator(MaxNLocator(integer=True))
                        # for ax in [ax1, ax2, ax3]:
                        #     ax.set_yticks(np.arange(0, 1 + 1/5, 1/5))

                    elif structure == "Host":
                        ax3 = plt.subplot(gs[1, :2])
                        bins = np.arange(-23, -13 + 1, 1)
                        abs_mag_binned, bins = np.histogram(abs_mag, bins=bins)
                        plt.stairs(abs_mag_binned/ngalaxies, edges=bins, color=color, label=label, ls=ls, linewidth=2)
                        plt.xlabel(r"Absolute Magnitude")
                        plt.ylabel("Fraction of Sample")
                        
                        ax4 = plt.subplot(gs[1, 2:])
                        bins = np.arange(0, 11 + 0.5, 0.5)
                        r_e_binned, bins = np.histogram(r_e, bins=bins)
                        plt.stairs(r_e_binned/ngalaxies, edges=bins, color=color, label=label, ls=ls, linewidth=2)
                        plt.xlabel(r"Effective Radius $r_e$ (kpc)")
                        plt.ylabel("Fraction of Sample")
                        # ax1.yaxis.set_major_locator(MaxNLocator(integer=True))
                        # ax2.yaxis.set_major_locator(MaxNLocator(integer=True))
                        # ax3.yaxis.set_major_locator(MaxNLocator(integer=True))
                        # ax4.yaxis.set_major_locator(MaxNLocator(integer=True))
                        # for ax in [ax1, ax2, ax3, ax4]:
                        #     ax.set_yticks(np.arange(0, 1 + 1/5, 1/5))

            ax1.legend(loc="upper right")
            plt.tight_layout()
            plt.savefig(os.path.join(Path(args.o), f"{structure}_type_compare.png"))


        plt.rc('axes', labelsize=24)
        plt.rc('legend', fontsize=18)
        plt.rc('figure', titlesize=28)
        plt.rc('xtick', labelsize=20)
        plt.rc('ytick', labelsize=20)
        fig = plt.figure(figsize=(8, 12))
        plt.suptitle(f"Host and Polar Comparisons")
        for galaxy_type in ["ring", "bulge", "halo"]:
            # for band in "griz":
            for band in "g":
                df_type = df[df["Galaxy_type"] == galaxy_type].copy()
                df_band = df_type[df_type["band"] == band]
                flux_ratio = df_band[df_band["label"] == "Host"]["flux_ratio"]
                host_PA = df_band[df_band["label"] == "Host"]["parameters.PA"]
                polar_PA = df_band[df_band["label"] == "Polar"]["parameters.PA"]
                diff_PA = host_PA - polar_PA
                diff_PA = np.abs(diff_PA)

                label = galaxy_plot_style[galaxy_type][0]
                color = galaxy_plot_style[galaxy_type][1]
                ls = galaxy_plot_style[galaxy_type][2]

                ngalaxies = np.size(host_PA)
                ax1 = plt.subplot(2, 1, 1)
                bins = np.arange(0, 180 + 15, 15)
                diff_PA_binned, bins = np.histogram(diff_PA, bins=bins)
                plt.stairs(diff_PA_binned/ngalaxies, edges=bins, color=color, label=label, ls=ls, linewidth=3)
                plt.xlabel(r"$PA_{host} - PA_{polar}$ (deg)")
                plt.ylabel("Fraction of Sample")

                ax2 = plt.subplot(2, 1, 2)
                bins = np.arange(0, 1 + 0.1, 0.1)
                flux_ratio_binned, bins = np.histogram(np.array(flux_ratio), bins=bins)
                plt.stairs(flux_ratio_binned/ngalaxies, edges=bins, color=color, label=label, ls=ls, linewidth=3)
                plt.xlabel(r"Flux Ratio $f_{Host}/f_{Total}$")
                plt.ylabel("Fraction of Sample")

        ax1.legend(loc="upper left")
        plt.tight_layout()
        plt.savefig(os.path.join(Path(args.o), "extra_compare_type.png"))
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
                if not(galaxy_type == None):
                    all_functions = get_functions_from_files(Path(root).resolve(), galaxy_type, table)
    else:
        all_functions = get_functions_from_files(root=Path(p).resolve(), galaxy_type=None, table=table)

    print(f"Total fit: {total_fit}")
    print(f"Total poor fit: {total_bad_fit} ({total_bad_fit/total_fit * 100:.2f}% bad)")
    print(f"Total parameter bounds sticking: {bound_sticking}")

    if args.plot_stats:
        quantities_plot(all_functions)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Generate plots of the outputs of fits")
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