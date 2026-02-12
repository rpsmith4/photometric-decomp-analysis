import glob
import numpy as np
import os

from astropy.io import fits

import argparse
from pathlib import Path

import multiprocessing as mp
import warnings
warnings.filterwarnings("ignore")
# from plot_parameters import parse_results
import pyimfit

# def generate_config(sci: np.array, mask: np.array = None, psf: np.array = None, invvar: np.array = None, type: str = "ring") -> pyimfit.ModelDescription:
#     print("Hello")
def generate_config(model_desc_dict: dict, outfile: str, band: str, sci: np.array, mask: np.array = None, psf: np.array = None, invvar: np.array = None, type: str = "ring") -> pyimfit.ModelDescription:

    print("Hello")


def main(args):
    if not(args.p == None):
        p = Path(args.p)
    else:
        p = Path(".")

    manager = mp.Manager()
    
    galpathlist = []

    if args.r:
        structure = os.walk(p)
        for root, dirs, files in structure:
            if not(files == []):
                galpathlist.append(Path(root).resolve())
    else:
        galpathlist.append(p.resolve())
        
    for galpath in galpathlist:
        img_files = sorted(glob.glob(os.path.join(galpath, "image_?.fits")))

        jobs = []
        model_desc_dict = manager.dict()
        for img_file in img_files:
            band = img_file[-6] # Yes I know this is not the best way
            
            if not(f"{args.fit_type}_{band}.dat" in files) or args.overwrite:
                print(f"Generating config for {img_file}")
                img = fits.getdata(img_file)

                if args.mask:
                    mask = fits.getdata(os.path.join(galpath, "image_mask.fits"))
                else:
                    mask = None
                    # img = img * (1-mask)
                    
                psf = fits.getdata(os.path.join(galpath, f"psf_patched_{band}.fits"))
                invvar = fits.getdata(os.path.join(galpath, f"image_{band}_invvar.fits"))
                
                folder_type_dict = {
                    "Polar Rings": "ring",
                    "Polar_Tilted Bulges": "bulge",
                    "Polar_Tilted Halo": "halo"
                }
                if not args.type:
                    for folder in ["Polar Rings", "Polar_Tilted Bulges", "Polar_Tilted Halo"]: # Attempt to autodetect type
                        if folder in root:
                            args.type = folder_type_dict[folder]

                outfile = f"{args.fit_type}_{band}_fit_params.txt"
                if args.fit_type == "2_sersic":
                    # p = mp.Process(target = generate_config, args=(img, str(args.type).lower(), model_desc, band))
                    p = mp.Process(target = generate_config, args=(model_desc_dict, outfile, band, img, mask, psf, invvar, args.fit_type))
                # elif args.fit_type == "1_sersic_1_gauss_ring":
                #     two_sersic_fit_params = f"2_sersic_{band}_fit_params.txt"
                #     p = mp.Process(target = init_guess_1_sersic_1_gauss_ring, args=(img, two_sersic_fit_params, str(args.type).lower(), model_desc, band))
                
                jobs.append(p)  
        
        if not(len(jobs) == 0):
            for p in jobs:
                p.start()

            for p in jobs:
                p.join()

            for band in model_desc_dict.keys():
                if not(f"{args.fit_type}_{band}.dat" in files) or args.overwrite:
                    with open(os.path.join(galpath, f"{args.fit_type}_{band}.dat"), "w") as f:
                        print(model_desc_dict[band])
                        # f.write(model_desc_dict[band])
    print(galpathlist)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Attempt to generate initial guess and bounds and write to an IMFIT config file")
    parser.add_argument("-p", help="Path to file/folder containing galaxy FITS")
    parser.add_argument("--overwrite", help="Overwrite existing config files", action="store_true")
    parser.add_argument("--mask", help="Use the mask to guess initial values", action="store_true")
    parser.add_argument("--type", help="Type of polar structure", choices=["ring", "bulge", "halo"], default="ring")
    parser.add_argument("--dont_fit", help="Don't use DE imfitting to try and do another guess at initial parameters", action="store_true")
    parser.add_argument("--fit_type", help="Type of fit done", choices=["2_sersic", "1_sersic_1_gauss_ring", "3_sersic"], default="2_sersic")
    parser.add_argument("-r", help="Recursively go into subfolders (assumes that fits data is at the end of the filetree)", action="store_true")
    parser.add_argument("--new", help="Use new version of the photometric decomp", action="store_true")
    

    args = parser.parse_args()
    if args.new:
        from initial_parameterization import gather_parameters as genparams
    # Here I will make my own separate file
    # else:
    #     from 
    main(args)
