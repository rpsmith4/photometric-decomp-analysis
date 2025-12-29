import pandas as pd
import argparse
import os
from pathlib import Path
import get_fits_from_folders


def main(args):

    df = pd.read_excel(Path(args.i))
    name_cols = list(df)[:10]

    os.chdir(Path(args.o))

    for name_col in name_cols:
        df_n = df[df[name_col].notna()]
        RA = df_n["RA"].to_list()
        DEC = df_n["Dec"].to_list()
        names = df_n[name_col].to_list()
        # TODO: Need to check what units these are in, too big right now
        size = df_n["size"]
        size[size.isna()] = 1
        size = size.to_list()


        df = df.drop(df[df[name_col].notna()].index)
        
        get_fits_from_folders.get_fits(names, RA, DEC, size, args)
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Read an excel spreadsheet and download the galaxies inside. Also does not download duplicate galaxies that may be under a different name.")
    parser.add_argument("-i", help="Input spreadsheet")
    parser.add_argument("-o", help="Output directory for files")
    parser.add_argument("--dr", help="Data Release (dr9 or dr10)", default="dr10")
    parser.add_argument("--factor", help="Factor by which to multiply the R26 isphote radius by", default=3)
    parser.add_argument("--bands", help="Bands to download", default="griz")
    parser.add_argument("--files", nargs="+", choices=["fits", "mask", "psf", "wm", "jpg"],
                        help="Files to download. 'fits' for FITS image file, 'mask' to generate a mask with SExtractor (FITS image must exist), 'psf' for the core and estimated extended PSF, 'wm' for inverse-variance (weight) map (must be used with 'fits'), and 'jpg' for a coadded JPG image.", default=["fits", "mask", "psf", "wm", "jpg"])
    parser.add_argument("--no_make_folder", help="Decide whether to make separate folders for each galaxy", action="store_true")
    parser.add_argument("--overwrite", help="Overwrite existing fits files", action="store_true")

    args = parser.parse_args()
    main(args)