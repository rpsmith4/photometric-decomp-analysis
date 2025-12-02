from astropy.io import fits
import numpy as np
from astropy.table import Table
import astropy
import pandas as pd
import argparse
import sys
import os
from pathlib import Path
import pyimfit
import subprocess
import glob
IMAN_DIR = os.path.expanduser("~/Documents/iman_new")
sys.path.append(os.path.join(IMAN_DIR, 'decomposition/make_model'))
import make_model_ima_imfit
import signal


def run_imfit(band, mask=True, psf=True, invvar=True, alg="LM", max_threads=4, fit_type="2_sersic", stdout_callback=None):
    # Assumes alread in directory
    #imfit -c config.dat image_g.fits --mask image_mask.fits --psf psf_patched_g.fits --noise image_g_invvar.fits --save-model g_model.fits --save-residual g_residual.fits --max-threads 4 --errors-are-weights
    # command = ["imfit", "-c", f"config_{band}.dat", f"image_{band}.fits", "--save-model", f"{band}_model.fits", "--save-residual", f"{band}_residual.fits", "--save-params", f"{band}_fit_params.txt", "--max-threads", f"{args.max_threads}"]
    command = ["imfit", "-c", f"{fit_type}_{band}.dat", f"image_{band}.fits", "--save-params", f"{fit_type}_{band}_fit_params.txt", "--max-threads", f"{max_threads}"]
    if mask:
        command.extend(["--mask", "image_mask.fits"])
    if psf:
        command.extend(["--psf", f"psf_patched_{band}.fits"])
    if invvar:
        command.extend(["--noise", f"image_{band}_invvar.fits", "--errors-are-weights"])
    if alg=="NM":
        # command.extend(["--nm", "--bootstrap 50", "--save-bootstrap", f"bootstrap_{band}.dat"])
        command.extend(["--nm"])
    if alg=="DE":
        command.extend(["--de"])
    if alg=="DE_LHS":
        command.extend(["--de_lhs"])
    
    # Launch imfit and optionally stream stdout to a callback
    global p
    # Use line-buffering where possible so output is delivered in real time
    p = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    if stdout_callback is None:
        # If no callback provided, just wait until completion
        p.wait()
    else:
        # Stream output lines to callback in real time using readline loop
        try:
            # Ensure p.stdout is not None (type check for static analyzers)
            if p.stdout is not None:
                for line in iter(p.stdout.readline, ''):
                    if line:
                        try:
                            stdout_callback(line)
                        except Exception:
                            # Ensure exceptions in the callback don't break the loop
                            pass
                    else:
                        break
        except Exception:
            pass
        finally:
            try:
                p.wait()
            except Exception:
                pass

def handler(signum, frame):
    print("Terminating IMFIT process...")
    try:
        p.terminate()
    except Exception:
        pass
    print("IMFIT process terminated.")
    sys.exit(-1)
    # raise IOError("Quitting on {}".format(signum))

signal.signal(signal.SIGTERM, handler)

def main(p, bands, r=False, overwrite=False, mask=True, psf=True, invvar=True, alg="LM", max_threads=4, fit_type="2_sersic", make_composed=True):
    if not(p == None):
        p = Path(p).resolve()
        os.chdir(p)
    if r:
        structure = os.walk(".")
        for root, dirs, files in structure:
            if not(files == []):
                # Assumes data is at the end of the file tree
                img_files = sorted(glob.glob(os.path.join(Path(root), "image_?.fits")))

                for img_file in img_files:
                        band = img_file[-6] # Yes I know this is not the best way
                        if band in bands:
                            os.chdir(Path(root))
                            if not(any([f"{fit_type}_{band}_composed.fits" in files, f"{fit_type}_{band}_fit_params.txt" in files])) or overwrite:
                                # Assumes the names of the files for the most part
                                # config file should be called config_[band].dat, may also include a way to change that 
                                run_imfit(band, mask, psf, invvar, alg, max_threads, fit_type)
                            os.chdir(p)
    else:
        img_files = sorted(glob.glob(os.path.join(Path("."), "image_?.fits")))

        for img_file in img_files:
                band = img_file[-6] # Yes I know this is not the best way
                if band in bands:
                    files = os.listdir(".")
                    if not(any([f"{fit_type}_{band}_composed.fits" in files, f"{fit_type}_{band}_fit_params.txt" in files])) or overwrite:
                        # Assumes the names of the files for the most part
                        # config file should be called config_[band].dat, may also include a way to change that 
                        run_imfit(band, mask, psf, invvar, alg, max_threads, fit_type)
                        img_file = f"image_{band}.fits"
                        psf_file = f"psf_patched_{band}.fits"
                        params_file = f"{fit_type}_{band}_fit_params.txt"
                        mask_file = f"image_mask.fits"
                        if make_composed and (not(f"{fit_type}_{band}_composed.fits" in files) or overwrite):
                            if mask:
                                img_dat = fits.open(img_file)
                                img = img_dat[0].data
                                mask_img = fits.open(mask_file)[0].data
                                # areas to be removed are marked "1" in the mask
                                img = img * (1 - mask_img)
                                fits.writeto("masked.fits", data=img, header=img_dat[0].header)

                                print(os.getcwd())
                                make_model_ima_imfit.main("masked.fits", params_file, psf_file, composed_model_file=f"{fit_type}_{band}_composed.fits", comp_names=["Host", "Polar"])
                                os.remove("./masked.fits")
                            else:
                                make_model_ima_imfit.main(img_file, params_file, psf_file, composed_model_file=f"{fit_type}_{band}_composed.fits", comp_names=["Host", "Polar"])


def terminate_imfit():
    global p
    try:
        if p is not None:
            p.terminate()
            # give it a short grace period
            try:
                p.wait(timeout=1)
            except Exception:
                try:
                    p.kill()
                except Exception:
                    pass
    except Exception:
        pass



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Run IMFIT on a sample of galaxies."
    )
    
    parser.add_argument("-p", help="Path to folder containing galaxies")
    parser.add_argument("-r", help="Recursively go into subfolders to find", action="store_true")
    parser.add_argument("--overwrite", help="Overwrites existing configs", action="store_true")
    parser.add_argument("--mask", help="Use mask image", action="store_true")
    parser.add_argument("--psf", help="Use psf image", action="store_true")
    parser.add_argument("--invvar", help="Use invvar map", action="store_true")
    parser.add_argument("--all", help="Use mask, psf, and invvar map", action="store_true")
    parser.add_argument("--nm", help="Use Nelder-Mead simplex solver (instead of Levenberg-Marquardt)", action="store_true")
    parser.add_argument("--de", help="Use differential evolution solver", action="store_true")
    parser.add_argument("--de_lhs", help="Use differential evolution solver (with Latin hypercube sampling)", action="store_true")
    parser.add_argument("--max_threads", help="Max number of threads to use for a fit", type=int, default=4)
    parser.add_argument("--fit_type", choices=["2_sersic", "1_sersic_1_gauss_ring", "3_sersic"], default="2_sersic")
    parser.add_argument("--make_composed", help="Make a composed image of the galaxy (includes image, model, and components)", action="store_true")
    parser.add_argument("--bands", help="Image bands to fit", nargs="+", default=["g", "r", "i", "z"])
    # TODO: Add more arguments for IMFIT options

    args = parser.parse_args()

    alg = "LM"
    if args.nm:
        alg = "NM"
    if args.de:
        alg = "DE"
    if args.de_lhs:
        alg = "DE_LHS"

    main(args.p, args.bands, r=args.r, overwrite=args.overwrite, mask=args.mask, psf=args.psf, invvar=args.invvar, alg=alg, max_threads=args.max_threads, fit_type=args.fit_type, make_composed=args.make_composed)
