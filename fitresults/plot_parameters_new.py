import glob
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import os

from astropy.io import fits
from astropy.constants import c
from astropy.table import QTable

import pyimfit
import pandas as pd

import argparse
from pathlib import Path

import astropy.units as u
import subprocess
import traceback as tb
import sys
import json

IMAN_DIR = os.path.join(Path(os.path.dirname(__file__)).parent, "iman_new")
sys.path.append(os.path.join(IMAN_DIR, 'decomposition/make_model'))
import make_model_ima_imfit

from parse_results import *

# warnings.filterwarnings("ignore")

def quantities_plot(all_functions):
    fig = plt.figure()
    band_colors = {
        "g": "g",
        "r" : "r",
        "i" : "firebrick",
        "z" : "blueviolet"
    }
    return None

def main(args):
    global p
    p = Path(args.p).resolve()
    SGAtable = QTable.read(os.path.join(args.c))
    # master_table = QTable.read(os.path.join())
    master_table = QTable.read("/home/ryans/Projects/Photometric Decomp/Analysis/master_table.csv")
    galmarks = json.load(open(os.path.join(p, "galmarks.json")))
    print(galmarks)

    global total_bad_fit
    global total_fit
    global bound_sticking
    total_bad_fit = 0
    total_fit = 0
    bound_sticking = 0

    galaxies = []
    bands = ["g", "r"]

    structure = os.walk(p)
    for root, dirs, files in structure:
        if len(glob.glob(os.path.join(root,"*.fits"))) != 0: # Check to see if we have entered a folder with galaxy data
            galaxies.append([Path(root).resolve(), os.path.basename(os.path.dirname(root))])

    all_results =  get_all_results(galaxies, bands, args.fit_type, galmarks)
    total_fit = len(all_results.groups.indices)
    total_bad_fit = len(all_results[all_results["Reduced ChiSq"] > args.t].groups.indices)

    print(f"Total fit: {total_fit}")
    print(f"Total poor fit: {total_bad_fit} ({total_bad_fit/total_fit * 100:.2f}% bad)")
    print(f"Total parameter bounds sticking: {bound_sticking}")

    print(all_results)

    if args.plot_stats:
        quantities_plot(all_results)

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
    parser.add_argument("-c", help="Path to Sienna Galaxy Atlas File (used to get redshift)", default="SGA-2020.fits")
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