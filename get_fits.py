from astropy.io import fits
import numpy as np
from astropy.table import Table
import astropy
import pandas as pd
import argparse
import sys
import os
from pathlib import Path

iman_dir = os.path.expanduser('~') + '/Documents/iman_new'

sys.path.append(os.path.join(iman_dir, 'misc_funcs/'))
import download_legacy_fits
import get_mask
import download_legacy_PSF
import download_legacy_WM
sys.path.append(os.path.join(iman_dir, 'imp/psf/'))

import create_extended_PSF_DESI

# TODO: Maybe reintroduce the --overwrite functionality
def get_fits(file_names, RA, DEC, R26, args):
    for i, file in enumerate(file_names):
        if not(os.path.isfile(file + ".fits")) or args.overwrite:
            download_legacy_fits.main([file], [RA[i]], [DEC[i]], R=[R26[i]*args.factor], bands=args.bands, dr=args.dr)

        if args.psf and (not(os.path.isfile(file + "_psf.fits")) or args.overwrite):
            download_legacy_PSF.main([file + "_psf"], [RA[i]], [DEC[i]], R=[R26[i]*args.factor], bands=args.bands, dr=args.dr)

            images_dat_psf = fits.open(file + "_psf.fits")

            for i, image_dat_psf in enumerate(images_dat_psf):
                band = images_dat_psf[0].header["BAND" + str(i)]
                create_extended_PSF_DESI.main(file + "_psf.fits", file + "_psf_ex_" + band + ".fits", band=band, layer=i)
                # TODO: Recombine the bands of the PSF into one file so its less of a mess

        if args.mask and (not(os.path.isfile(file + "_mask.fits")) or args.overwrite):
            images_dat = fits.open(file + ".fits")
            total_mask = np.zeros_like(images_dat[0].data[0])

            for i, image_dat in enumerate(images_dat[0].data):
                band = images_dat[0].header["BAND" + str(i)]
                try:
                    image, mask, theta, sma, smb = get_mask.prepare_rotated(image_dat, subtract=False, rotate_ok=False)
                    total_mask += mask
                except:
                    continue 

            total_mask[total_mask >= 1] = 1
            file_name = file + "_mask.fits"
            fits.PrimaryHDU(total_mask).writeto(file_name, overwrite=args.overwrite)
        
        if args.wm and (not(os.path.isfile(file + "_wm")) or args.overwrite):
            download_legacy_WM.main([file + "_wm"], [RA[i]], [DEC[i]], R=[R26[i]*args.factor], bands=args.bands, dr=args.dr)
            # TODO: Might want to remove regular images and just keep weight maps

def get_quantities(files, data):
    RA = [float(data[data["GALAXY"] == file]["RA"]) for file in files]
    DEC = [float(data[data["GALAXY"] == file]["DEC"]) for file in files]
    R26 = [float(data[data["GALAXY"] == file]["D26"])/2 for file in files]

    return RA, DEC, R26

def main(args):
    data = Table.read(args.c + "SGA-2020.fits")

    if not(args.o == None):
        os.chdir(Path(args.o))

    if args.r and not(args.p == None):
        structure = os.walk(args.p)
        main = Path(args.p)
        for root, dirs, files in structure:
            root = root.replace(str(main) + "/", "")
            try:
                if not(root==""):
                    os.mkdir(root) # Remaking folder structure in output folder
            except FileExistsError:
                continue
            finally:
                files = [file.rsplit(".", maxsplit=1)[0] for file in files]
                RA, DEC, R26 = get_quantities(files, data)

                out = Path(args.o)
                if not(root == ''):
                    os.chdir(Path(root))
                    get_fits(files, RA, DEC, R26, args)
                    os.chdir("../")
                else:
                    get_fits(files, RA, DEC, R26, args)
    else:
        files = args.f
        files = [os.path.basename(file) for file in files]

        files = [file.rsplit(".", maxsplit=1)[0] for file in files]
        RA, DEC, R26 = get_quantities(files, data)

        get_fits(files, RA, DEC, R26, args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Hello")
    parser.add_argument("-p", help="Path to file/folder containing galaxy sample names")
    parser.add_argument("-f", help="List of files (don't use in combination with -p)", nargs="+")
    parser.add_argument("-c", help="Catalogue of galaxy data (fits)", default="./")
    parser.add_argument("-r", help="Recursively go into subfolders", action="store_true")
    parser.add_argument("-o", help="Output directory", default="./")
    parser.add_argument("--dr", help="Data Release (dr9 or dr10)", default="dr10")
    parser.add_argument("--factor", help="Factor by which to multiply the R26 isphote radius by", default=3)
    parser.add_argument("--bands", help="Bands to download", default="griz")
    parser.add_argument("--mask", help="Estimates and creates a mask", action="store_true")
    parser.add_argument("--psf", help="Downloads the core PSF and estimates and extended PSF", action="store_true")
    parser.add_argument("--wm", help="Downloads the inverse variance map (weight map)", action="store_true")
    parser.add_argument("--overwrite", help="Overwrite existing fits files", action="store_true")

    args = parser.parse_args()
    main(args)