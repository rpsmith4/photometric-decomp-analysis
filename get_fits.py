from astropy.io import fits
import numpy as np
from astropy.table import Table
import astropy
import pandas as pd
import argparse
import sys
import os

iman_dir = os.path.expanduser('~') + '/Documents/iman_new'

sys.path.append(os.path.join(iman_dir, 'misc_funcs/'))
import download_legacy_fits
import get_mask

'''
Need to:
 - Get image names
 - Get coordinates from the catalogue
 - Get R26 from the catalogue
 - Get a fits cutout of the region from legacy_survey (need to use iman)
 - Download the output to somewhere reasonable (ideally keeping thins in the same folder stucture)
'''

def get_image_names(path):
    try:
        files = os.listdir(path)
    except:
        files = [path.split("/")[-1]]
    names = list()
    for i in files:
        if os.path.isdir(path + "/" + i):
            if args.r == True:
                names.extend(get_image_names(path + "/" + i))
        else:
            names.append(i.split(".")[0])

    return names

def main(args):
    all_data = fits.open(args.c + "SGA-2020.fits")
    data = all_data[1].data # Select the right data
    names = get_image_names(args.p)
    if not(args.o == None):
        os.chdir(args.o)
    for name in names:
        RA = data[data["GALAXY"] == name]["RA"]
        DEC = data[data["GALAXY"] == name]["DEC"]
        R26 = data[data["GALAXY"] == name]["D26"]/2 # arcmin
        download_legacy_fits.main([name], [RA], [DEC], [R26*args.factor], bands=args.bands, dr=args.dr)

        if args.mask:
            image_dat = fits.open(name + ".fits")
            print(image_dat.info)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Hello")
    parser.add_argument("-p", help="Path to file/folder containing galaxy samples", default="./")
    parser.add_argument("-c", help="Catalogue of galaxy data (fits)", default="./")
    parser.add_argument("-r", help="Recursively go into subfolders", action="store_true")
    parser.add_argument("-o", help="Output directory", default="./")
    parser.add_argument("--dr", help="Data Release (dr9 or dr10)", default="dr10")
    parser.add_argument("--factor", help="Factor by which to multiply the R26 isphote radius by", default=3)
    parser.add_argument("--bands", help="Bands to download", default="griz")
    parser.add_argument("--mask", help="Estimates and creates a mask", action="store_true")

    args = parser.parse_args()
    main(args)

