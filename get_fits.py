from astropy.io import fits
import numpy as np
from astropy.table import Table
import astropy
import matplotlib.pyplot as plt
from astropy.nddata import CCDData
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.nddata import Cutout2D
import astropy.visualization as viz
import pandas as pd
import matplotlib.patches as mpatches
import argparse
import sys
import os

def get_image_names(path):
    files = os.listdir(path)
    names = list()
    for i in files:
        if os.path.isdir(path + "/" + i):
            names.extend(get_image_names(path + "/" + i))
        else:
            names.append(i.split(".")[0])

    return names

def main(args):
    all_data = fits.open(args.c + "SGA-2020.fits")
    print(all_data)
    print(get_image_names(args.p))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Hello")
    parser.add_argument("-p", help="Path to folder containing galaxy samples", default="./")
    parser.add_argument("-c", help="Catalogue of galaxy data (fits)", default="./")
    parser.add_argument("-r", help="Recursively go into subfolders", action="store_true")
    parser.add_argument("-o", help="Output directory")

    args = parser.parse_args()
    main(args)

