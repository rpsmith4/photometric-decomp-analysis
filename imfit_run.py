from astropy.io import fits
import numpy as np
from astropy.table import Table
import astropy
import pandas as pd
import argparse
import sys
import os
from pathlib import Path
import pyimfit
import subprocess


def run_imfit(args):
    # Assumes alread in directory
    #imfit -c config.dat image_g.fits --mask image_mask.fits --psf psf_patched_g.fits --noise image_g_invvar.fits --save-model g_model.fits --save-residual g_residual.fits --max-threads 4 --errors-are-weights
    bands = "griz"
    for band in bands:
        subprocess.run(["imfit", "-c", "config.dat", f"image_{band}.fits", "--mask", "image_mask.fits", "--psf", f"psf_patched_{band}.fits", "--noise", f"image_{band}_invvar.fits", "--save-model", f"{band}_model.fits", "--save-residual", f"{band}_residual.fits", "--errors-are-weights", "--save-params", f"{band}_fit_params.txt"])

def main(args):
    p = Path(args.p)
    os.chdir(p)

    run_imfit(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("-p", help="Path to folder containing galaxies")
    parser.add_argument("-r", help="Recursively go into subfolders to find")
    # TODO: Add more arguments for IMFIT options

    args = parser.parse_args()

    main(args)