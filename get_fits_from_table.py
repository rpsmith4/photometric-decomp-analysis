import pandas as pd
import argparse
import os
from pathlib import Path
import get_fits


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
        
        get_fits.get_fits(names, RA, DEC, size, args)
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Hello")
    parser.add_argument("-i", help="Input spreadsheet")
    parser.add_argument("-o", help="Output directory for files")
    parser.add_argument("--dr", help="Data Release (dr9 or dr10)", default="dr10")
    parser.add_argument("--factor", help="Factor by which to multiply the R26 isphote radius by", default=3)
    parser.add_argument("--bands", help="Bands to download", default="griz")
    parser.add_argument("--mask", help="Estimates and creates a mask", action="store_true")
    parser.add_argument("--psf", help="Downloads the core PSF and estimates and extended PSF", action="store_true")
    parser.add_argument("--wm", help="Downloads the inverse variance map (weight map)", action="store_true")
    parser.add_argument("--overwrite", help="Overwrite existing fits files", action="store_true")

    args = parser.parse_args()
    main(args)