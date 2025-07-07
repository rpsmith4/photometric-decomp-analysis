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
import glob


def run_imfit(args):
    # Assumes alread in directory
    #imfit -c config.dat image_g.fits --mask image_mask.fits --psf psf_patched_g.fits --noise image_g_invvar.fits --save-model g_model.fits --save-residual g_residual.fits --max-threads 4 --errors-are-weights
    bands = "griz"
    for band in bands:
        subprocess.run(["imfit", "-c", "config.dat", f"image_{band}.fits", "--mask", "image_mask.fits", "--psf", f"psf_patched_{band}.fits", "--noise", f"image_{band}_invvar.fits", "--save-model", f"{band}_model.fits", "--save-residual", f"{band}_residual.fits", "--errors-are-weights", "--save-params", f"{band}_fit_params.txt"])

def main(args):
    if not(args.p == None):
        p = Path(args.p)
        os.chdir(p)

    # run_imfit(args)

    if args.r:
        structure = os.walk(".")
        for root, dirs, files in structure:
            if not(files == []):
                img_files = sorted(glob.glob(os.path.join(Path(root), "image_?.fits")))

                for i in range(len(img_files)):
                        band = img_file[-6] # Yes I know this is not the best way
                        os.chdir(Path(root))

                        # Assumes the names of the files for the most part
                        # config file should be called config_[band].dat, may also include a way to change that 
                        subprocess.run(["imfit", "-c", f"config_{band}.dat", f"image_{band}.fits", "--mask", "image_mask.fits", "--psf", f"psf_patched_{band}.fits", "--noise", f"image_{band}_invvar.fits", "--save-model", f"{band}_model.fits", "--save-residual", f"{band}_residual.fits", "--errors-are-weights", "--save-params", f"{band}_fit_params.txt", "--nm"])
                        os.chdir(p)
    else:
        img_files = sorted(glob.glob(os.path.join(Path("."), "image_?.fits")))

        for i in range(len(img_files)):
                band = img_files[i][-6] # Yes I know this is not the best way

                # Assumes the names of the files for the most part
                # config file should be called config_[band].dat, may also include a way to change that 
                subprocess.run(["imfit", "-c", f"config_{band}.dat", f"image_{band}.fits", "--mask", "image_mask.fits", "--psf", f"psf_patched_{band}.fits", "--noise", f"image_{band}_invvar.fits", "--save-model", f"{band}_model.fits", "--save-residual", f"{band}_residual.fits", "--errors-are-weights", "--save-params", f"{band}_fit_params.txt"])



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("-p", help="Path to folder containing galaxies")
    parser.add_argument("-r", help="Recursively go into subfolders to find")
    # TODO: Add more arguments for IMFIT options

    args = parser.parse_args()

    main(args)