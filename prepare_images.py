#! /usr/bin/env python

import argparse
import shutil
import subprocess
import requests
from pathlib import Path
from io import BytesIO
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from imfittools.ImfitModel import ImfitModel


def download_legacy_survey_region(ra, dec, size, passband=None):
    """
    Downloads a specified region from the Legacy Survey and saves it as a FITS file.

    Parameters:
        ra (float): Right Ascension (in degrees) of the center of the region.
        dec (float): Declination (in degrees) of the center of the region.
        size (int): Size of the region in pixels (e.g., 256 for a 256x256 image).
        passband (str, optional): Photometric passband (e.g., "g", "r", "z"). If None, default combined image is used.

    Returns:
        None
    """
    # Legacy Survey API URL
    url = "https://www.legacysurvey.org/viewer/cutout.fits"

    # Define query parameters
    params = {
        "ra": ra,
        "dec": dec,
        "size": size,
        "layer": "ls-dr10",  # Change the layer if needed (e.g., ls-dr9, ls-dr8, etc.)
        # "pixscale": 0.262,    # Pixel scale in arcseconds (default for Legacy Survey)
        "invvar": True
    }
    # Add passband if specified
    if passband:
        params["bands"] = passband

    attempts = 3
    for attempt in range(attempts):
        try:
            # Make the request to the Legacy Survey API
            print("Requesting image")
            response = requests.get(url, params=params)
            print(f"Response status: {response.reason}")

            if response.status_code == 500:
                print(f"Server error (500): {response.reason}.")
                return None
            response.raise_for_status()

            # Open the response as a FITS file and return
            print("Download OK")
            return fits.open(BytesIO(response.content))
        except requests.exceptions.RequestException as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            if attempt == attempts - 1:
                print("Max retries reached. Unable to download the file.")
        except Exception as e:
            print(f"Error handling FITS file: {e}")
            return None
    return None


def download_coadded_psf(ra, dec):
    """
    Download coadded PSF from a legacy server. This appears to be a
    first eigenvalue of the PSF output
    """
    # Legacy Survey API URL
    url = "https://www.legacysurvey.org/viewer/coadd-psf/"

    # Define query parameters
    params = {
        "ra": ra,
        "dec": dec,
        "layer": "ls-dr10",  # Change the layer if needed (e.g., ls-dr9, ls-dr8, etc.)
    }
    attempts = 3
    for attempt in range(attempts):
        try:
            # Make the request to the Legacy Survey API
            print("Requesting PSF")
            response = requests.get(url, params=params)
            print(f"Response status: {response.reason}")

            if response.status_code == 500:
                print(f"Server error (500): {response.reason}.")
                return None
            response.raise_for_status()
            print("Download OK")
            # Open the response as a FITS file and save it to disk
            return fits.open(BytesIO(response.content))
        except requests.exceptions.RequestException as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            if attempt == attempts - 1:
                print("Max retries reached. Unable to download the file.")
        except Exception as e:
            print(f"Error handling FITS file: {e}")
            return None
    return None


def fit_by_moffat(psf_file_name, target_size):
    workdir = Path("workdir")
    if workdir.exists():
        shutil.rmtree(workdir)
    workdir.mkdir()
    psf_data = fits.getdata(psf_file_name)
    y_size, x_size = psf_data.shape
    x_cen = x_size / 2
    y_cen = y_size / 2

    # Make a mask to fit only a region between 1.8 and 5 arcseconds
    mask = np.ones_like(psf_data)
    xx, yy = np.meshgrid(np.arange(x_size), np.arange(y_size))
    distances = np.hypot(x_cen-xx, y_cen-yy)
    mask[(distances > 6) * (distances < 20)] = 0
    fits.PrimaryHDU(data=mask).writeto("workdir/moffat_mask.fits", overwrite=True)

    # Setup fitting
    max_flux = np.max(psf_data)
    dflux = max_flux / 2
    config_str = f"X0 {x_cen}  {x_cen-5},{x_cen+5}\n" \
        f"Y0 {y_cen}  {y_cen-5},{y_cen+5}\n" \
        "FUNCTION Moffat\n" \
        f"PA 0 fixed\n" \
        f"ell 0 fixed \n" \
        f"I_0 {max_flux}  {max_flux-dflux},{max_flux+dflux}\n" \
        f"fwhm 4 1,20\n" \
        f"beta 4  1,20\n"
    with open("./workdir/input_moffat.imfit", 'w') as imfit_config:
        imfit_config.write(config_str)
    call_str = f"imfit {psf_file_name} -c ./workdir/input_moffat.imfit " \
        "--save-params ./workdir/fitted_moffat.imfit --save-model ./workdir/fitted_star.fits "\
        "--save-residual ./workdir/residuals.fits --de --mlr >/dev/null"
    subprocess.call(call_str, shell=True)
    fit_res = ImfitModel("./workdir/fitted_moffat.imfit")
    fit_res.recenter_all_components(x_new=target_size/2, y_new=target_size/2, dx=0, dy=0)
    fit_res.to_fits("./workdir/moffat_larger.fits", target_size, target_size)
    fitted_data = fits.getdata("./workdir/moffat_larger.fits")
    shutil.rmtree(workdir)
    return fitted_data


def make_patched_psf(psf_file_name, band,  target_size):
    print("Making patched PSF")
    psfex_data = fits.getdata(psf_file_name)
    # Fit by Moffat function
    moffat_fit = fit_by_moffat(psf_file_name, target_size)

    xx, yy = np.meshgrid(np.arange(target_size), np.arange(target_size))
    distances = np.hypot(target_size//2-xx, target_size//2-yy)
    pixscale = 0.262 # Arcsec/pixel
    distances[distances < 0.01] = 0.01

    match band:
        case 'g':
            outer_psf = moffat_fit + 0.00045 * 1 / (distances*pixscale)**2
        case 'r' | 'i':
            outer_psf = moffat_fit + 0.00033 * 1 / (distances*pixscale)**2
        case 'z':
            alpha = 17.650
            beta = 1.7
            w = 0.0145
            outer_psf = moffat_fit + w * (beta-1) / ((np.pi * alpha**2) * (1+(distances*pixscale)/alpha**2) ** beta)

    # Compute weights maps
    R1 = 5 / 0.262
    R2 = 6 / 0.262
    R3 = 7 / 0.262
    R4 = 8 / 0.262
    psfex_weight = np.zeros_like(distances, dtype=float)
    psfex_weight[distances < R1] = 1.0
    psfex_weight[(distances > R1) * (distances < R2)] = (R2 - distances[(distances > R1) * (distances < R2)]) / (R2-R1)
    outer_weight = np.zeros_like(distances, dtype=float)
    outer_weight[distances > R2] = 1.0
    outer_weight[(distances > R1) * (distances < R2)] = (distances[(distances > R1) * (distances < R2)] - R1) / (R2-R1)

    outer_weight[distances > R4] = 0
    outer_weight[(distances > R3) * (distances < R4)] = (distances[(distances > R3) * (distances < R4)] - R4) / (R3-R4)

    # Combine PSF
    psf_orig_size = psfex_data.shape[0]
    pad_before = (target_size - psf_orig_size) // 2
    pad_after = target_size - psf_orig_size - pad_before
    pad_width = ((pad_before, pad_after), (pad_before, pad_after))
    psfex_data_padded = np.pad(array=psfex_data, pad_width=pad_width, constant_values=0)
    psf_combined = psfex_data_padded * psfex_weight + outer_psf * outer_weight
    psf_combined /= np.sum(psf_combined)
    return psf_combined


def main(args):
    outdir = Path('downloads')
    outdir.mkdir(exist_ok=True)
    # Download and unpack images and invvar maps for all available passbands
    hdu = download_legacy_survey_region(args.ra, args.dec, args.size)
    bands = hdu[0].header["BANDS"].strip()
    wcs = WCS(hdu[0].header)
    for idx, band in enumerate(bands):
        data = hdu[0].data[idx, ...]
        if np.sum(data) == 0:
            continue
        fits.PrimaryHDU(data=data, header=wcs.to_header()).writeto(outdir/f"image_{band}.fits", overwrite=True)
    for idx, band in enumerate(bands):
        data = hdu[1].data[idx, ...]
        if np.sum(data) == 0:
            continue
        fits.PrimaryHDU(data=data, header=wcs.to_header()).writeto(outdir/f"image_{band}_invvar.fits", overwrite=True)

    # Download PSF images
    psf_hdu_list = download_coadded_psf(args.ra, args.dec)
    for hdu in psf_hdu_list:
        band = hdu.header['BAND']
        data = hdu.data
        fits.PrimaryHDU(data=data).writeto(outdir / f"psf_core_{band}.fits", overwrite=True)
        psf_combined = make_patched_psf(outdir / f"psf_core_{band}.fits", band, 150)
        fits.PrimaryHDU(data=psf_combined).writeto(outdir / f"psf_patched_{band}.fits", overwrite=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--ra", type=float)
    parser.add_argument("--dec", type=float)
    parser.add_argument("--size", type=int, help="Size in pixels")
    args = parser.parse_args()
    main(args)
