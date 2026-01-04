from astropy.io import fits
import numpy as np
from astropy.table import QTable
import argparse
import os
from pathlib import Path

import download_legacy_DESI
import get_mask
import astropy.units as u

from mpi4py import MPI
from mpi4py.futures import MPIPoolExecutor
from functools import partial

def get_fits(file_names, RA, DEC, R26, psg_types, args):
    args.factor = float(args.factor)
    curr_root = Path(os.getcwd())
    # pixscale = 0.262
    # RR = int(np.ceil(R26[k]*60. * args.factor/pixscale))
    num_gals = len(file_names)
    for i in range(num_gals):
        print(f"Status: [{i}/{num_gals}] (%{i/num_gals*100:.2f})")
        file = file_names[i]
        psg_type = psg_types[i]

        directory = os.path.join(psg_type, file)
        print(f"Downloading {file}...")
        if not(args.no_make_folder):
            os.makedirs(directory, exist_ok=True)
            os.chdir(directory)
        if "fits" in args.files and (not any(os.path.isfile(f"image_{band}.fits") for band in args.bands) or args.overwrite):
            download_legacy_DESI.main([file], [RA[i]], [DEC[i]], R=[R26[i]*args.factor], file_types=["fits"], bands=args.bands, dr=args.dr)

        if "psf" in args.files and (not any(os.path.isfile(f"psf_core_{band}.fits") for band in args.bands) or args.overwrite):
            download_legacy_DESI.main([file + "_psf"], [RA[i]], [DEC[i]], R=[R26[i]*args.factor], file_types=["psf"], bands=args.bands, dr=args.dr)


        if "mask" in args.files and (not(os.path.isfile("image_mask.fits")) or args.overwrite):
            pixscale = 0.262
            RR = int(np.ceil(R26[i]*60. * args.factor/pixscale))

            total_mask = np.zeros((RR*2, RR*2))

            for k, band in enumerate(args.bands):
                try:
                    image_dat = fits.open("image_" + band + ".fits")[0].data
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

        if "jpg" in args.files and (not(os.path.isfile(f"image.jpg")) or args.overwrite):
            download_legacy_DESI.main([file], [RA[i]], [DEC[i]], R=[R26[i]*args.factor], file_types=["jpg"], bands=args.bands, dr=args.dr)
        if not(args.no_make_folder):
            os.chdir(curr_root)


def get_quantities(files, data):
    RA = [float(data[data["GALAXY"] == file]["RA"]) for file in files]
    DEC = [float(data[data["GALAXY"] == file]["DEC"]) for file in files]
    R26 = [float(data[data["GALAXY"] == file]["D26"])/2 for file in files]

    return RA, DEC, R26

def main(args):
    catalog = QTable.read("master_table.csv", data_start=2)

    if not(args.o == None):
        output = Path(args.o).resolve()
        os.makedirs(output, exist_ok=True)
        os.chdir(output)
    
    file_names = catalog["NAME"]
    RA = np.array(catalog["RA"]).astype(np.float64)
    DEC = np.array(catalog["DEC"].astype(np.float64))
    R26_SGA = np.array(catalog["D26_SGA"]/2).astype(np.float64)
    R26_IRAF = np.array([catalog["R26_G_IRAF"], catalog["R26_R_IRAF"], catalog["R26_I_IRAF"], catalog["R26_Z_IRAF"]])
    R26 = np.append(R26_IRAF, np.expand_dims(R26_SGA, 0), axis=0)
    R26 = np.nanmean(R26, axis=0, where=(R26 != 0))
    # R26[R26 == 0] = 0.3 # Get rid of empty values

    types = catalog["PSG_TYPE_1"]

    idx = np.arange(len(file_names))

    # part = partial(get_fits, file_names=file_names, RA=RA, DEC=DEC, R26=R26, psg_types = types, args=args)
    # with MPIPoolExecutor(max_workers=1) as pool:
    #     pool.map(part, idx)
    get_fits(file_names=file_names, RA=RA, DEC=DEC, R26=R26, psg_types = types, args=args)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Download data from DESI Legacy by using an existing folder structure containing files have their galaxy name as their file name. Uses the 26 mag/arcsecond^2 isophote radius to determine size of the image (which is multiplied by a chosen constant factor).")
    parser.add_argument("-n", help="List of files/names (don't use in combination with -p)", nargs="+")
    parser.add_argument("-o", help="Output directory", default="./")
    parser.add_argument("--dr", help="Data Release (dr9 or dr10)", default="dr10")
    parser.add_argument("--factor", help="Factor by which to multiply the R26 isphote radius by", default=3)
    parser.add_argument("--bands", help="Bands to download", default="griz")
    parser.add_argument("--files", nargs="+", choices=["fits", "mask", "psf", "wm", "jpg"],
                        help="Files to download. 'fits' for FITS image file, 'mask' to generate a mask with SExtractor (FITS image must exist), 'psf' for the core and estimated extended PSF, 'wm' for inverse-variance (weight) map (must be used with 'fits'), and 'jpg' for a coadded JPG image.", default=["fits", "mask", "psf", "wm", "jpg"])
    parser.add_argument("--no_make_folder", help="Decide whether to make separate folders for each galaxy", action="store_true")
    parser.add_argument("--overwrite", help="Overwrite existing downloaded data", action="store_true")

    args = parser.parse_args()
    main(args)