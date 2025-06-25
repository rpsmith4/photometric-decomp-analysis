from astropy.io import fits
import numpy as np
from astropy.table import Table
import astropy
import pandas as pd
import argparse
import sys
import os
from pathlib import Path
from astropy.coordinates import SkyCoord
import matplotlib.pyplot as plt
from reproject.mosaicking import find_optimal_celestial_wcs
from reproject.mosaicking import reproject_and_coadd
from reproject import reproject_interp
from reproject import reproject_exact

def combine_wm(name, weights):
    # Assuming that the wm from DESI is in the order of image, weight, image, weight, ...
    # Weights should be an HDUlist
    print("Coadding Weight Map...")
    primary = weights[0]
    image_header = weights[2].header # To remake the header at the end
    weights = weights[2::2] # Get inverse-variance maps

    wcs_out, shape_out = find_optimal_celestial_wcs([weight for weight in weights]) 

    bands = {"g": [], "r": [], "i": [], "z": []}
    for weight in weights:
        band = weight.header["BAND"]
        bands[band].append(weight)

    out_ims = list()


    header_wcs = wcs_out.to_header()


    for band in bands.keys():
        weights_band = bands[band]
        out_im, footprint = reproject_and_coadd(weights_band, wcs_out, shape_out=shape_out, reproject_function=reproject_interp)
        out_im_n = fits.ImageHDU(out_im)

        header_n = image_header # Replacing the header WCS
        for h in header_wcs:
            header_n[h] = header_wcs[h]
        header_n["BAND"] = band

        out_im_n.header = header_n.copy()

        out_ims.append(out_im_n)

    out_ims.insert(0, primary)
    hdul = fits.HDUList(out_ims)

    hdul.writeto(name, overwrite=True)
    print("Done with WM Coadd")

def main(args):
    # TODO: Add logic if file is called directly
    print("")

if __name__ == "__main__":
    print("")