from astropy.io import fits
import numpy as np
from astropy.table import Table
import astropy
import pandas as pd
import argparse
import sys
import os
from pathlib import Path

import download_legacy_DESI
import prepare_images
import get_mask
import combine_wm

def get_fits(file_names, RA, DEC, R26, args):
    args.factor = float(args.factor)
    # pixscale = 0.262
    # RR = int(np.ceil(R26[k]*60. * args.factor/pixscale))
    for i, file in enumerate(file_names):
        os.makedirs(file, exist_ok=True)
        os.chdir(file)
        if args.fits and (not any(os.path.isfile(f"image_{band}.fits") for band in args.bands) or args.overwrite):
            download_legacy_DESI.main([file], [RA[i]], [DEC[i]], R=[R26[i]*args.factor], file_types=["fits"], bands=args.bands, dr=args.dr)

        if args.psf and (not any(os.path.isfile(f"psf_core_{band}.fits") for band in args.bands) or args.overwrite):
            download_legacy_DESI.main([file + "_psf"], [RA[i]], [DEC[i]], R=[R26[i]*args.factor], file_types=["psf"], bands=args.bands, dr=args.dr)


        if args.mask and (not(os.path.isfile("image_mask.fits")) or args.overwrite):
            # images_dat = fits.open(file + ".fits")
            pixscale = 0.262
            RR = int(np.ceil(R26[i]*60. * args.factor/pixscale))

            total_mask = np.zeros((RR*2, RR*2))

            for k, band in enumerate(args.bands):
                try:
                    image_dat = fits.open("image_" + band + ".fits")[0].data
                    # band = images_dat[0].header["BAND" + str(k)]
                    try:
                        image, mask, theta, sma, smb = get_mask.prepare_rotated(image_dat, subtract=False, rotate_ok=False)
                        total_mask += mask
                    except:
                        continue 
                except:
                    continue

            total_mask[total_mask >= 1] = 1
            file_name = "image_mask.fits"
            fits.PrimaryHDU(total_mask).writeto(file_name, overwrite=args.overwrite)
        
        # if args.wm and (not(os.path.isfile(file + "_wm.fits")) or args.overwrite):
        #     try:
        #         download_legacy_DESI.main([file + "_wm"], [RA[i]], [DEC[i]], R=[R26[i]*args.factor], file_types=["wm"], bands=args.bands, dr=args.dr)
        #     except Exception as e:
        #         print(f"Failed to download psf for {file} ({e})! Continuing...")
        #         pass

            # images_dat = fits.open(file + ".fits")
            # out_shape = images_dat[0].data[0].shape
            # weights = fits.open(file + "_wm.fits")  
            # combine_wm.combine_wm(file + "_wm.fits", weights, out_shape)

        if args.jpg and (not(os.path.isfile("image.jpg")) or args.overwrite):
            download_legacy_DESI.main([file], [RA[i]], [DEC[i]], R=[R26[i]*args.factor], file_types=["jpg"], bands=args.bands, dr=args.dr)
        os.chdir("..")


def get_quantities(files, data):
    RA = [float(data[data["GALAXY"] == file]["RA"]) for file in files]
    DEC = [float(data[data["GALAXY"] == file]["DEC"]) for file in files]
    R26 = [float(data[data["GALAXY"] == file]["D26"])/2 for file in files]

    return RA, DEC, R26

def main(args):
    data = Table.read(args.c + "SGA-2020.fits")

    if not(args.p == None):
        in_path = Path(args.p).resolve()
        structure = os.walk(in_path)
        main = Path(args.p).resolve()

    if not(args.o == None):
        output = Path(args.o).resolve()
        os.chdir(output)

    if args.r and not(args.p == None):
        for root, dirs, files in structure:
            root = Path(root).relative_to(main)
            try:
                if not(root==Path(".")):
                    os.mkdir(root) # Remaking folder structure in output folder
            except FileExistsError:
                continue
            finally:
                files = [file.rsplit(".", maxsplit=1)[0] for file in files]
                RA, DEC, R26 = get_quantities(files, data)
                out = Path(args.o)
                if not(root == Path('.')):
                    os.chdir(Path(root))
                    get_fits(files, RA, DEC, R26, args)
                    os.chdir(output)
                else:
                    get_fits(files, RA, DEC, R26, args)
    elif not(args.r) and not(args.p == None):
        files = os.listdir(in_path)

        files = [os.path.basename(file) for file in files]

        files = [file.rsplit(".", maxsplit=1)[0] for file in files]
        RA, DEC, R26 = get_quantities(files, data)

        get_fits(files, RA, DEC, R26, args)

    elif not (args.f == None):
        files = args.f
        files = [os.path.basename(file) for file in files]

        files = [file.rsplit(".", maxsplit=1)[0] for file in files]
        RA, DEC, R26 = get_quantities(files, data)

        get_fits(files, RA, DEC, R26, args)
    
    elif not(args.n == None):
        file = args.n
        RA, DEC, R26 = get_quantities([file], data)

        get_fits([file], RA, DEC, R26, args)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Hello")
    parser.add_argument("-p", help="Path to file/folder containing galaxy sample names")
    parser.add_argument("-f", help="List of files (don't use in combination with -p)", nargs="+")
    parser.add_argument("-n", help="Name of galaxy")
    parser.add_argument("-c", help="Catalogue of galaxy data (fits)", default="./")
    parser.add_argument("-r", help="Recursively go into subfolders", action="store_true")
    parser.add_argument("-o", help="Output directory", default="./")
    parser.add_argument("--dr", help="Data Release (dr9 or dr10)", default="dr10")
    parser.add_argument("--factor", help="Factor by which to multiply the R26 isphote radius by", default=3)
    parser.add_argument("--bands", help="Bands to download", default="griz")
    parser.add_argument("--fits", help="Download the FITS file", action="store_true")
    parser.add_argument("--mask", help="Estimates and creates a mask", action="store_true")
    parser.add_argument("--psf", help="Downloads the core PSF and estimates and extended PSF", action="store_true")
    parser.add_argument("--wm", help="Downloads the inverse variance map (weight map)", action="store_true")
    parser.add_argument("--jpg", help="Downloads a jpg image cutout", action="store_true")
    parser.add_argument("--overwrite", help="Overwrite existing fits files", action="store_true")

    args = parser.parse_args()
    main(args)