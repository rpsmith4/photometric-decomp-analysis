#!/usr/bin/env python


import sys
from pathlib import Path

# Add the parent directory (decomposer/) to Python path
sys.path.append(str(Path(__file__).resolve().parent.parent))


import argparse
from pathlib import Path
import os
import math
import shutil
import json
import warnings
from string import Template

from scipy.ndimage import zoom
from scipy.optimize import fmin
import numpy as np
from numpy.typing import NDArray
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse as MPLEllipse

from astropy.io import fits
from astropy.modeling import models, fitting






from scipy.ndimage import map_coordinates
from scipy.signal import fftconvolve
from scipy.optimize import least_squares
from typing import Dict, Optional, Tuple, Union

# Reuse your helper utilities directly
from photometric_cut_helpers import (
    load_fits_array,
    pixel_scale_from_header_arcsec_per_pix,
    plot_slit_overlay,
    bn_of_n,
    sersic_mu,
    initial_guesses_mu,
)

ArrayLike = Union[np.ndarray, str, Path]

def _mu_from_I(I: np.ndarray, zeropoint: float, pixscale_arcsec: float) -> np.ndarray:
    """
    Convert linear flux per pixel to mag/arcsec^2.
    """
    I = np.asarray(I, float)
    pixarea = float(pixscale_arcsec) ** 2
    with np.errstate(divide="ignore", invalid="ignore"):
        mu = zeropoint - 2.5 * np.log10(I / pixarea)
    return mu

def _normalize_kernel(psf: np.ndarray) -> np.ndarray:
    psf = np.asarray(psf, float)
    psf = np.where(np.isfinite(psf), psf, 0.0)
    s = psf.sum()
    if s <= 0:
        raise ValueError("PSF kernel sum <= 0; cannot normalize.")
    return psf / s

def photometric_cut(
    sci_fits: ArrayLike,
    center_xy: Tuple[float, float],
    pa_deg: float,
    *,
    # geometry
    length_pix: float = 240.0,
    width_pix: float = 5.0,
    oversample: int = 2,
    # optional maps
    mask_fits: Optional[ArrayLike] = None,
    invvar_fits: Optional[ArrayLike] = None,
    psf_fits: Optional[ArrayLike] = None,
    # photometric calibration (only needed for mu; relative ZP is fine)
    zeropoint: float = 22.5,
    pixel_scale_arcsec: Optional[float] = None,
    # # background handling
    # subtract_background: bool = False,
    # background_region: str = "ends",   # "ends" or "none"
    #endcap_frac: float = 0.18,
    # interpolation
    interpolation_order: int = 1,
    # center refinement (folding metric)
    refine_center: bool = False,
    refine_kwargs: Optional[dict] = None,
    # debugging / visualization
    show_slit: bool = False,
    slit_overlay_savepath: Optional[str] = None,
) -> Dict[str, np.ndarray]:
    """
    Extract a robust photometric cut along a specified PA through a center.

    Returns a dict with keys:
      - center_xy_initial, center_xy_used, center_refined, center_refine_metric
      - pa_deg, pixel_scale_arcsec
      - s_pix, s_arcsec
      - I, I_err
      - mu, mu_err
      - frac_good
      - background_estimate, background_sigma_robust (if computed)
    """
    sci = sci_fits.data

    if sci.ndim != 2:
        raise ValueError(f"sci must be 2D; got shape {sci.shape}")

    # ny, nx = sci.shape

    # Pixel scale
    if pixel_scale_arcsec is None:
        # DESI DR10 cutouts have WCS; use header when sci_fits is a path-like
        try:
            pixel_scale_arcsec = pixel_scale_from_header_arcsec_per_pix(sci_fits)
        except Exception:
            pixel_scale_arcsec = None
    if pixel_scale_arcsec is None:
        raise ValueError("pixel_scale_arcsec could not be inferred; pass pixel_scale_arcsec explicitly.")

    # Optional arrays
    msk = mask_fits if mask_fits is not None else None
    inv = invvar_fits.data if invvar_fits is not None else None

    if msk is not None and msk.shape != sci.shape:
        raise ValueError(f"mask shape {msk.shape} != sci shape {sci.shape}")
    if inv is not None and inv.shape != sci.shape:
        raise ValueError(f"invvar shape {inv.shape} != sci shape {sci.shape}")

    bad = (msk > 0) if msk is not None else np.zeros_like(sci, dtype=bool)

    # Record initial center (always)
    center_xy_initial = tuple(map(float, center_xy))
    center_xy_used = center_xy_initial
    center_refine_metric: Optional[float] = None

    # Optional PSF convolution (science only; invvar propagation is non-trivial)
    if psf_fits is not None:
        psf = load_fits_array(psf_fits)
        psf = _normalize_kernel(psf)
        # sci = fftconvolve(np.nan_to_num(sci, nan=0.0), psf, mode="same") # DO NOT DO THIS !!!! 
                                                                         # Sci already contains the actual science data, so don't convolve again
        # mask left as-is; invvar left as-is (uncertainties become approximate)



    # Debug overlay
    if show_slit or slit_overlay_savepath:
        plot_slit_overlay(
            sci_fits=sci,  # array (already convolved/bg-subtracted if applied)
            mask_fits=msk,
            center_xy=center_xy_used,
            pa_deg=pa_deg,
            length_pix=length_pix,
            half_width_pix=0.5 * width_pix,
            savepath=slit_overlay_savepath,
            show=bool(show_slit),
        )

    # Build sampling grid
    th = np.deg2rad(pa_deg)
    u = np.array([np.cos(th), np.sin(th)])       # along slit
    v = np.array([-np.sin(th), np.cos(th)])      # across slit

    # Along-slit sample points
    step = 1.0 / max(int(oversample), 1)
    s_pix = np.arange(-0.5 * length_pix, 0.5 * length_pix + 1e-9, step, dtype=float)

    # Across-slit samples
    half_w = 0.5 * float(width_pix)
    t_pix = np.arange(-half_w, half_w + 1e-9, step, dtype=float)

    x0, y0 = center_xy_used

    xs = x0 + s_pix[:, None] * u[0] + t_pix[None, :] * v[0]
    ys = y0 + s_pix[:, None] * u[1] + t_pix[None, :] * v[1]
    coords = np.vstack([ys.ravel(), xs.ravel()])

    # Sample science
    sci_samp = map_coordinates(
        sci,
        coords,
        order=int(interpolation_order),
        mode="constant",
        cval=np.nan,
    ).reshape(s_pix.size, t_pix.size)

    # Sample mask (nearest)
    bad_samp = map_coordinates(
        bad.astype(float),
        coords,
        order=0,
        mode="constant",
        cval=1.0,   # outside image treated as bad
    ).reshape(s_pix.size, t_pix.size) > 0.5

    # Sample invvar (if any)
    if inv is not None:
        inv_samp = map_coordinates(
            inv,
            coords,
            order=0,
            mode="constant",
            cval=0.0,
        ).reshape(s_pix.size, t_pix.size)
        inv_samp = np.where(np.isfinite(inv_samp) & (inv_samp > 0), inv_samp, 0.0)
    else:
        inv_samp = None

    # Combine across width
    good = (~bad_samp) & np.isfinite(sci_samp)

    if inv_samp is None:
        w = good.astype(float)
    else:
        w = inv_samp * good.astype(float)

    wsum = np.sum(w, axis=1)
    frac_good = np.sum(good, axis=1) / float(good.shape[1])

    I = np.full(s_pix.size, np.nan, float)
    I_err = np.full(s_pix.size, np.nan, float)

    ok = wsum > 0
    if np.any(ok):
        I[ok] = np.sum(w[ok] * sci_samp[ok], axis=1) / wsum[ok]

        # Uncertainty
        if inv_samp is not None:
            I_err[ok] = np.sqrt(1.0 / np.maximum(wsum[ok], 1e-30))
        else:
            resid = sci_samp[ok] - I[ok, None]
            mad = np.nanmedian(np.abs(resid), axis=1)
            sigma = 1.4826 * mad
            Ng = np.sum(good[ok], axis=1).astype(float)
            I_err[ok] = sigma / np.sqrt(np.maximum(Ng, 1.0))

    # Convert to arcsec and surface brightness
    s_arcsec = s_pix * float(pixel_scale_arcsec)

    mu = _mu_from_I(I, zeropoint=zeropoint, pixscale_arcsec=float(pixel_scale_arcsec))
    with np.errstate(divide="ignore", invalid="ignore"):
        mu_err = (2.5 / np.log(10.0)) * (I_err / np.maximum(I, 1e-30))

    out: Dict[str, np.ndarray] = {
        "center_xy_initial": np.array(center_xy_initial, float),
        "center_xy_used": np.array(center_xy_used, float),
        "center_refined": bool(refine_center),
        "center_refine_metric": center_refine_metric,
        "pa_deg": float(pa_deg),
        "pixel_scale_arcsec": float(pixel_scale_arcsec),
        "length_pix": float(length_pix),
        "width_pix": float(width_pix),
        "oversample": int(max(1, oversample)),
        "s_pix": s_pix,
        "s_arcsec": s_arcsec,
        "I": I,
        "I_err": I_err,
        "mu": mu,
        "mu_err": mu_err,
        "frac_good": frac_good,
        #"background_estimate": background_estimate,
        #"background_sigma_robust": background_sigma_robust,
    }

    out["valid"] = np.isfinite(I) & (frac_good > 0)

    return out

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


def main(galaxy_directory: str = "", psf: NDArray | None = None, image_name: str = "", data_array: NDArray | None = None, band: str | None = None, pa: float | None = None, ell: float | None = None, component: str | None = None, mask: Path | None = None) -> None:
    """ A function that prepares the necessary information to pass along to the manual decomposition script.

    
    Parameters:
        galaxy_directory: A string containing the path to the directory with the galaxy images
        psf: An array of a 1-D slice of the psf
        image_name: A string of the .fits image file
        data_array: An array of the 1-D profile for manual estimation

    """

    args = argparse.Namespace()

    # test_data_dir = str(galaxy_directory) + "/"
    data_dir = str(galaxy_directory) + "/" # Incorporate whether it is the polar or host being saved. Only needs to be included here.
    args.profile = data_dir + "ellipse.txt"
    if band:
        args.image = data_dir + f"image_{band}.fits"
        psf_info = fits.open(data_dir + f'psf_patched_{band}.fits')
    else:
        args.image = data_dir + f"image_g.fits"
        psf_info = fits.open(data_dir + 'psf_patched_g.fits')

    args.psf = data_dir + "azim_model_psf.txt"
    
    psf_array = psf_info[0].data
    psf_center = (psf_array.shape[0] // 2, psf_array.shape[1] // 2)
    psf_cut = photometric_cut(psf_array, psf_center, 172.6, length_pix = 40, pixel_scale_arcsec=0.262) # I don't know if this is how one does a 1D PSF, you may want to read up on this since I think its different
    # It is definitely not perfect, as it misses contributions from outside the slit, but was sufficient for a first approximation. The full process would require some type of deconvolution which would take me a bit to figure out how to properly include, and hasn't been impactful enough yet.
    R_psf, I_psf, Ierr_psf, radial_mask = fold_cut_to_radial_profile_I(psf_cut)

        # Stack columns together
    data = np.column_stack((R_psf, I_psf, Ierr_psf))
    save_path = data_dir
    # Save to file with formatting
    np.savetxt(
        save_path + "psf_profile.txt",
        data,
        header="sma[pix]\tflux[DN]\tflux_err[DN]",
        fmt=["%6.2f", "% .8e", "% .8e"],
        comments="# ")
    print(f"Successfully created and saved psf_profile.txt to {save_path}")
    psf_info = np.genfromtxt(save_path + "psf_profile.txt", unpack=True, usecols=[1])
   
    args.psf = save_path + "psf_profile.txt"

    if band:
        gal_info = fits.open(data_dir + f'image_{band}.fits')
    else:
        gal_info = fits.open(data_dir + f'image_g.fits')
    gal_array = gal_info[0].data
    gal_center = (gal_array.shape[0] // 2, gal_array.shape[1] // 2)

    mask_data = fits.getdata(mask)
    gal_cut = photometric_cut(gal_array, gal_center, pa, mask_fits= mask_data, length_pix = 600, pixel_scale_arcsec=0.262)
    R_gal, mu_gal, muerr_gal, mask = fold_cut_to_radial_profile_I(gal_cut)
    data_gal = np.column_stack((R_gal, mu_gal, muerr_gal))
    np.savetxt(
        save_path + "gal_profile_" + component + ".txt",
        data_gal,
        header="sma[pix]\tI\tI_err",
        fmt=["%6.2f", "% .8e", "% .8e"],
        comments="# ")
    print(f"Successfully created and saved gal_profile_{component}.txt to {save_path}")
    print('\n')
    # psf_info = np.genfromtxt(save_path + "gal_profile.txt", unpack=True, usecols=[1])

    args.profile = save_path + "gal_profile_" + component + ".txt"

    # Currently, all of these are hard coded. They shouldn't need to change with any of the files to my knowledge.
    args.ZP = 22.5
    args.pix2sec = 0.262
    args.SBlim = 30.0
    args.adderr = 0.000
    args.mask_radii = "0:0"
    args.profile_type = 'photcut'
    args.pa = pa
    args.ell = ell
    args.component = component

    import decomposer_updated
    decomposer_updated.main(args, data_dir)

    # Generate new config file
    # I'll need to clean up a bunch of this but I'm wanting to get it working first

    import generate_imfit_conf
    import glob
    import pandas as pd
    gen_args = argparse.Namespace()
    gen_args.p = p
    gen_args.overwrite = True
    gen_args.mask = True
    gen_args.ellipse_fit = Path("./decomposer/EllipseFitResults")
    gen_args.fit_type = "2_sersic"
    gen_args.master_table = Path("./decomposer")
    gen_args.r = False
    gen_args.dont_fit = True
    gen_args.new = True

    csvs = glob.glob(os.path.join(Path(gen_args.ellipse_fit), "*.ecsv"))
    ellipse_fit_data = pd.DataFrame(columns=["file", "PolarOrHost","IsoLevel", "x_center", "y_center", "semi_major", "semi_minor", "angle"])

    if component == 'host':
        gen_args.component = 'host_manual'
        if len(glob.glob(str(p) + "/*host.json")) != 0:
            generate_imfit_conf.main(gen_args, fit_band = band)
  
    elif component == 'polar': 
        gen_args.component = 'polar_manual'
        if len(glob.glob(str(p) + "/*polar.json")) != 0:
            generate_imfit_conf.main(gen_args, fit_band = band)
  

    
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
    parser.add_argument("-b", help="Band to fit", default="b", choices=['g','r','i','z'])
    # I don't really know how the GUI interacts with this script, so this is my guess for how the following can be passed in?
    parser.add_argument("-c", help="Component to fit", default=None, choices = ['host', 'polar']) 
    parser.add_argument("-pa", help="Position angle results for the correct component", default = None)
    parser.add_argument("-ell", help="Ellipticity results for the correct component", default = None)
    parser.add_argument("-m", help="Path to the mask file for this galaxy", default = None)

    args = parser.parse_args()
    p = Path(args.p)

    main(p, band=args.b, pa = float(args.pa), ell = float(args.ell), component = args.c, mask = args.m)