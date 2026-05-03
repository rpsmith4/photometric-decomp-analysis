#!/usr/bin/env python


import sys
from pathlib import Path

# Add the parent directory (decomposer/) to Python path
sys.path.append(str(Path(__file__).resolve().parent.parent))


import argparse
from pathlib import Path
import os
import numpy as np
from astropy.io import fits
import generate_imfit_conf
import glob
import decomposer_updated

BASE_DIR = Path(Path(os.path.dirname(__file__)).parent.parent).resolve()
sys.path.append(os.path.join(BASE_DIR, 'decomposer'))
from photometric_cut import photometric_cut






from scipy.ndimage import map_coordinates
from scipy.signal import fftconvolve
from scipy.optimize import least_squares
from typing import Dict, Optional, Tuple, Union






def fold_cut_to_radial_profile_I(cut, min_frac_good=0.5):
    """
    Take a cut dict from photometric_cut() and produce a folded radial profile:
      R = |s| (arcsec), mu(R) = median of +/- sides in bins.

    Returns: R, mu, mu_err, mask_used (boolean mask on the *original* samples)
    """
    s = np.asarray(cut["s_pix"], float)
    I = np.asarray(cut["I"], float)
    I_err = np.asarray(cut.get("I_err", np.full_like(I, np.nan)), float)
    frac_good = np.asarray(cut.get("frac_good", np.ones_like(I)), float)
    valid = np.asarray(cut.get("valid", np.isfinite(I)), bool)

    base_mask = valid & np.isfinite(s) & np.isfinite(I) & (frac_good >= float(min_frac_good))
    if np.sum(base_mask) < 10:
        return np.array([]), np.array([]), np.array([]), base_mask

    s0 = s[base_mask]
    I0 = I[base_mask]
    I_e0 = I_err[base_mask] if I_err is not None else np.full_like(I0, np.nan)

    R = np.abs(s0)
    order = np.argsort(R)
    R, I0, I_e0 = R[order], I0[order], I_e0[order]

    # Bin in R with an automatic bin width
    dR = np.median(np.diff(R)) if R.size > 10 else (R.max() - R.min()) / max(R.size - 1, 1)
    dR = float(dR) if np.isfinite(dR) and dR > 0 else 0.5

    edges = np.arange(R.min(), R.max() + 1e-9, dR)
    if edges.size < 3:
        return np.array([]), np.array([]), np.array([]), base_mask

    idx = np.digitize(R, edges) - 1
    nb = edges.size - 1

    Rb, Ib, Ieb = [], [], []
    for b in range(nb):
        sel = idx == b
        if np.sum(sel) < 2:
            continue
        Rb.append(np.nanmedian(R[sel]))
        Ib.append(np.nanmedian(I0[sel]))
        # conservative: take median error in bin
        Ieb.append(np.nanmedian(I_e0[sel]) if np.any(np.isfinite(I_e0[sel])) else np.nan)

    return np.asarray(Rb), np.asarray(Ib), np.asarray(Ieb), base_mask


def main(galaxy_directory: str = "", band: str | None = None, pa: float | None = None, ell: float | None = None, component: str | None = None, mask: Path | None = None, ellipse_path: Path | None = None, master_table_path: Path | None = None) -> None:
    """ A function that prepares the necessary information to pass along to the manual decomposition script, then generates the appropriate manual config files for IMFIT.

    
    Parameters:
        galaxy_directory: A string containing the path to the directory with the galaxy images
        band: A string containing the band to manually fit [griz]
        ell: A float of the ellipticity for the component being fit, taken from ellipse results
        component: A string detailing whether the host or polar strucutre is being fit ['host','polar']
        mask: A path to the location of the mask for this object
        ellipse_path: A path to the location where the ellipse results are stored
        master_table_path: A path to the master_table.csv file (included)
    """

    
    # Location of galaxy files and to save all new files
    data_dir = str(galaxy_directory) + "/" 

    # Make sure the band is passed in properly
    if band is not None:
        psf_info = fits.open(data_dir + f'psf_patched_{band}.fits')
    else: # Raise an error if no band was passed.
        raise ValueError("band must be provided as g, r, i, or z")

    # Get psf profile as a slit along the same axis as what is being manually fit. Technically not exactly how it should be done, but I think it works well enough for our purposes to avoid doing a full 2-D deconvolution of the image when making the original slit. Saves the results into a text file for usage in the manual fitter.
    psf_array = psf_info[0].data
    psf_center = (psf_array.shape[0] // 2, psf_array.shape[1] // 2)
    psf_cut = photometric_cut(psf_array, psf_center, pa, length_pix = 40, pixel_scale_arcsec=0.262) # 
    R_psf, I_psf, Ierr_psf, radial_mask = fold_cut_to_radial_profile_I(psf_cut)
    data = np.column_stack((R_psf, I_psf, Ierr_psf))
    save_path = data_dir
    # Save to file with formatting
    np.savetxt(
        save_path + "psf_profile.txt",
        data,
        header="sma[pix]\tflux[DN]\tflux_err[DN]",
        fmt=["%6.2f", "% .8e", "% .8e"],
        comments="# ")
    print(f"Successfully created and saved psf_profile.txt to {save_path}\n")
    psf_info = np.genfromtxt(save_path + "psf_profile.txt", unpack=True, usecols=[1])
   

    # Arguments needed to get the galaxy data in the manual fitter
    gal_info = fits.open(data_dir + f'image_{band}.fits')
    gal_array = gal_info[0].data
    gal_center = (gal_array.shape[0] // 2, gal_array.shape[1] // 2)
    mask_data = fits.getdata(mask)

    # Call the photometric cut function to get the galaxy data, and write in into the appropriate format for use with the manual fitter
    gal_cut = photometric_cut(gal_array, gal_center, pa, mask_fits= mask_data, length_pix = 600, pixel_scale_arcsec=0.262) # Come back to the length_pix value perhaps. I haven't seen it cause issues ever but is currently hardcoded.
    R_gal, mu_gal, muerr_gal, mask = fold_cut_to_radial_profile_I(gal_cut)
    data_gal = np.column_stack((R_gal, mu_gal, muerr_gal))
    np.savetxt(
        save_path + "gal_profile_" + component + ".txt",
        data_gal,
        header="sma[pix]\tI\tI_err",
        fmt=["%6.2f", "% .8e", "% .8e"],
        comments="# ")
    print(f"Successfully created and saved gal_profile_{component}.txt to {save_path}\n")



    # Arguments to pass into the manual fit script
    # Hard coded
    args = argparse.Namespace()
    args.ZP = 22.5 # Magnitude zero point is not obtained from GUI script
    args.pix2sec = 0.262 # pixel scale is likewise not obtained from GUI script
    args.SBlim = 30.0 # this was the default for the base manual fitter script
    args.adderr = 0.000 # to my knowledge there is no flat error that we need to add to our data
    args.mask_radii = "0:0" # we shouldn't need to mask out any radii at this step
    args.profile_type = 'photcut' # every time this is used, it is a photcut
    # Soft coded
    args.image = data_dir + f"image_{band}.fits"
    args.pa = pa
    args.ell = ell
    args.component = component
    args.profile = save_path + "gal_profile_" + component + ".txt"
    args.psf = save_path + "psf_profile.txt"

    # Call manual fit script
    decomposer_updated.main(args, data_dir)

    # Generate new config file
    # Define all necessary arguments to call the config generation script
    gen_args = argparse.Namespace()
    gen_args.p = p
    gen_args.overwrite = True
    gen_args.mask = True
    if ellipse_path is not None:
        gen_args.ellipse_fit = ellipse_path
    else:
        gen_args.ellipse_fit = Path("./decomposer/EllipseFitResults") # If not passed in properly, deaults to where I have it stored. Probably unnecessary
    if master_table_path is not None:
        gen_args.master_table = Path(master_table_path).parent
    else:
        gen_args.master_table = Path("./decomposer") # Same story as with ellipse_fit
    gen_args.fit_type = "2_sersic"
    gen_args.r = False
    gen_args.dont_fit = True
    gen_args.new = True

    # Create manual host config file if there is a saved host file and the host was being fit
    if component == 'host':
        gen_args.component = 'host_manual'
        if len(glob.glob(str(p) + "/*host.json")) != 0:
            generate_imfit_conf.main(gen_args, fit_band = band)
    
    # Create manual polar config file if there is a saved polar file and the polar was being fit
    elif component == 'polar': 
        gen_args.component = 'polar_manual'
        if len(glob.glob(str(p) + "/*polar.json")) != 0:
            generate_imfit_conf.main(gen_args, fit_band = band)
  
    # Create a fully manual file if both host and polar have been manually fit
    all_json = glob.glob(str(p) + "/*.json")
    if len(all_json) > 1:
        gen_args.component = 'all_manual'
        generate_imfit_conf.main(gen_args, fit_band = band)

    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    parser.add_argument("-p", help="Path to galaxy folder", default=".")
    parser.add_argument("-b", help="Band to fit", default=None, choices=['g','r','i','z'])
    parser.add_argument("-c", help="Component to fit", default=None, choices = ['host', 'polar']) 
    parser.add_argument("-pa", help="Position angle results for the correct component", default = None)
    parser.add_argument("-ell", help="Ellipticity results for the correct component", default = None)
    parser.add_argument("-ellipse_path", help="Path to the ellipse file for this galaxy", default = None)
    parser.add_argument("-m", help="Path to the mask file for this galaxy", default = None)
    parser.add_argument("-master_table", help="Path to the master_table file", default = None)

    args = parser.parse_args()
    p = Path(args.p)

    main(p, band=args.b, pa = float(args.pa), ell = float(args.ell), component = args.c, mask = args.m, ellipse_path = args.ellipse_path, master_table_path = args.master_table)