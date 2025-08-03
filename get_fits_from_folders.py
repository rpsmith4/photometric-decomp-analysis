from astropy.io import fits
import numpy as np
from astropy.table import Table
import argparse
import os
from pathlib import Path

import download_legacy_DESI
import get_mask

def get_fits(file_names, RA, DEC, R26, args):
    args.factor = float(args.factor)
    # pixscale = 0.262
    # RR = int(np.ceil(R26[k]*60. * args.factor/pixscale))
    for i, file in enumerate(file_names):
        if not(args.no_make_folder):
            os.makedirs(file, exist_ok=True)
            os.chdir(file)
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

        # Used if you need to coadd segmented weight maps       
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

        if "jpg" in args.files and (not(os.path.isfile(f"image.jpg")) or args.overwrite):
            download_legacy_DESI.main([file], [RA[i]], [DEC[i]], R=[R26[i]*args.factor], file_types=["jpg"], bands=args.bands, dr=args.dr)
        if not(args.no_make_folder):
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
        p = Path(args.p).resolve()

    if not(args.o == None):
        output = Path(args.o).resolve()
        os.chdir(output)

    if args.r and not(args.p == None):
        for root, dirs, files in structure:
            root = Path(root).relative_to(p)
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

    elif not (args.n == None):
        files = args.n
        files = [os.path.basename(file) for file in files]

        files = [file.rsplit(".", maxsplit=1)[0] for file in files]
        RA, DEC, R26 = get_quantities(files, data)

        get_fits(files, RA, DEC, R26, args)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Download data from DESI Legacy by using an existing folder structure containing files have their galaxy name as their file name. Uses the 26 mag/arcsecond^2 isophote radius to determine size of the image (which is multiplied by a chosen constant factor).")
    parser.add_argument("-p", help="Path to file/folder containing galaxy sample names")
    parser.add_argument("-n", help="List of files/names (don't use in combination with -p)", nargs="+")
    parser.add_argument("-c", help="Path to Siena Galaxy Catalogue", default="./")
    parser.add_argument("-r", help="Recursively go into subfolders (also regenerates file structure)", action="store_true")
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