import argparse
from pathlib import Path
import glob
import numpy as np
import os
import subprocess


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

def main(args):
    pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hello")
    parser.add_argument("-p", help="Path to file/folder containing models", default=".")
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
    parser.add_argument("-v", help="Show chi-sq warnings for fits", action="store_true")
    parser.add_argument("-vv", help="Show chi-sq and parameter bounds warnings for fits", action="store_true")
    parser.add_argument("-vvv", help="Show chi-sq and parameter bounds (specific) warnings for fits", action="store_true")
    args = parser.parse_args()
    args.o = Path(args.o).resolve()
    main(args)
