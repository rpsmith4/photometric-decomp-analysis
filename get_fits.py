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
import download_legacy_PSF
import download_legacy_WM
# import convert_npy_to_fits
sys.path.append(os.path.join(iman_dir, 'imp/psf/'))

import create_extended_PSF_DESI

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
        names = {path: list()}
    except:
        file = path.split("/")[-1]
        file = file.split(".")[0]
        path = "/".join(path.split("/")[:-1])
        names = {path : [file]}
        return names

    for i in files:
        if os.path.isdir(path + i + "/"):
            if args.r == True:
                names[path].append(get_image_names(path + i + "/"))
        else:
            names[path].append(".".join(i.split(".")[:-1])) # Some files have "." in them that have nothing to do with the file extension

    return names

# TODO: Might want this to just take in paramters like RA, DEC, and R26 (as well as name) so that I can put things not in the catalogue into this
def get_fits(name, data=None, RA=None, DEC=None, R26=None):
    """
    Retreive FITS with correct parameters, calculate and make mask, as well as get weight map and PSF for each band
    """
    # TODO: Still need to find a way to get the weight map
    if not(data == None): # Allows to override needing to use the catalogue
        RA = data[data["GALAXY"] == name]["RA"]
        DEC = data[data["GALAXY"] == name]["DEC"]
        R26 = data[data["GALAXY"] == name]["D26"]/2 # arcmin
    
    if not(os.path.isfile(name + ".fits")) or (os.path.isfile(name + ".fits") and args.overwrite):
        download_legacy_fits.main([name], [RA], [DEC], [R26*args.factor], bands=args.bands, dr=args.dr)
        if args.psf:
            download_legacy_PSF.main([name + "_psf"], [RA], [DEC], [R26*args.factor], bands=args.bands, dr=args.dr)
            images_dat_psf = fits.open(name + "_psf.fits")

            for i, image_dat_psf in enumerate(images_dat_psf):
                band = images_dat_psf[0].header["BAND" + str(i)]
                create_extended_PSF_DESI.main(name + "_psf.fits", name + "_psf_ex_" + band + ".fits", band=band)
                # TODO: Recombine the bands of the PSF into one file so its less of a mess

        if args.mask:
            images_dat = fits.open(name + ".fits")

            total_mask = np.zeros_like(images_dat[0].data[0])
            for i, image_dat in enumerate(images_dat[0].data):
                band = images_dat[0].header["BAND" + str(i)]
                try:
                    image, mask, theta, sma, smb = get_mask.prepare_rotated(image_dat, subtract=False, rotate_ok=False)
                    total_mask += mask
                except:
                    continue 

            total_mask[total_mask >= 1] = 1
            file_name = name + "_mask.fits"
            fits.PrimaryHDU(total_mask).writeto(file_name, overwrite=args.overwrite)
        
        if args.wm:
            download_legacy_WM.main([name + "_wm"], [RA], [DEC], [R26*args.factor], bands=args.bands, dr=args.dr)

def make_filestructure_and_download(names, data):
    for name in names.keys():
        files = names[name]
        for file in files:
            if type(file) == str:
                get_fits(file, data)
            elif type(file) == dict:
                folder = list(file)[0].split("/")
                folder = folder[-2]
                try:
                    os.mkdir(folder)
                except FileExistsError:
                    continue
                finally:
                    os.chdir(folder)
                    make_filestructure_and_download(file, data)
                    os.chdir("../")

def main(args):
    all_data = fits.open(args.c + "SGA-2020.fits")
    data = all_data[1].data # Select the right data

    names = get_image_names(args.p)
    if not(args.o == None):
        os.chdir(args.o)
    
    make_filestructure_and_download(names, data)

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
    parser.add_argument("--psf", help="Downloads the core PSF and estimates and extended PSF", action="store_true")
    parser.add_argument("--wm", help="Downloads the inverse varianve map (weight map)", action="store_true")
    parser.add_argument("--overwrite", help="Overwrite existing fits files", action="store_true")

    args = parser.parse_args()
    main(args)

