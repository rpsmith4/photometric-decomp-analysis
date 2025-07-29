import ferengi
import argparse
import numpy as np
from pathlib import Path
from astropy.io import fits
import astropy.units as u
import matplotlib.pyplot as plt

# INPUTS:
#   sky = high redshift sky background image
#         2d float array in units [counts/sec]
#         layout is: [x,y] (pixels x, pixels y)
#   im = local input image cube
#        3d float array in units [cts]
#        layout is: [x,y,bands] (pixels x, pixels y, filters)
#   imerr = local input error image cube (Poisson noise image)
#           3d float array in units [cts]
#           layout is: [x,y,bands] (pixels x, pixels y, filters)
#   psflo = low redshift PSF
#           3d float array
#           layout is: [x,y,bands] (pixels x, pixels y, filters)
#           [total flux normalised to 1!]
#   err0_mag = minimum errors to apply to K-correction
#              1d float array in magnitudes
#              layout is: [bands] (filters)
#              for SDSS: err0_mag=[0.05, 0.02, 0.02, 0.02, 0.03]
#   psfhi = high redshift PSF
#           2d float array
#           layout is: [x,y] (pixels x, pixels y)
#           [total flux normalised to 1!]
#   lambda_lo = central/effective wavelength of local filters
#               1d float array in units [angstroems]
#               layout is: [bands] (filters)
#   filter_lo = local filters
#               1d string array
#               layout is: [bands] (filters)
#               for details see description of parameter FILTERLIST to KCORRECT
#   zlo = local redshift
#         float
#   scllo = local pixel scale
#           float in units [arcseconds per pixel]
#   zplo = local zeropoints
#          1d float array in units [magnitudes]
#          layout is: [bands] (filters)
#   tlo = local exposure time (array: [nbands])
#         1d float array in units [seconds]
#         layout is: [bands] (filters)
#   lambda_hi = high redshift wavelength
#               float in units [angstroems]
#   filter_hi = high redshift filters
#               string
#               for details see description of parameter FILTERLIST to KCORRECT
#   zhi = high redshift
#         float
#   sclhi = high redshift pixel scale
#           float in units [arcseconds per pixel]
#   zphi = high redshift zeropoint
#          float in units [magnitudes]
#   thi = high redshift exposure time
#         float in units [seconds]

# TODO: Figure out how to get MJy/sr -> counts/pixel
def simunits2cts(simim, pixarea):
    simim = simim * 10 ** 6 * 10 ** (-23) * pixarea # MJy/sr to ergs/s/cm**2/Hz (fnu) /pixel?
    return ferengi.maggies2cts(ferengi.fnu2maggies(simim), expt=3000000, zp=22.5) # fnu/pixel -> maggies/pixel -> cnts/pixel

def main(im, psf, sky):
    # 1 pixel is 100 pc 
    pixscale = 0.262 * u.arcsecond
    pixarea = pixscale ** 2
    pixarea = pixarea.to(u.sr).value

    im = simunits2cts(im, pixarea) 
    # TODO: Figure out what the exposure time should be
    # for the sky as well as the simulated image
    sky = ferengi.maggies2cts(sky * 10 ** (-9), expt=1, zp=22.5) # sky is in nmgy
    # sky = np.zeros_like(sky)
    # lam = 0.000799898 
    # imerr = np.random.poisson(lam=lam, size=im.shape) # nanomaggies
    # imerr = ferengi.maggies2cts(imerr, expt=1, zp = 22.5)
    imerr = np.zeros_like(im) # Poisson Noise alread added
    psflo = psf
    psfhi = psf[0] # Using just the g band psf
    # erro0_mag = np.array([0, 0, 0, 0])
    erro0_mag = np.array([0.02, 0.02, 0.02, 0.03]) # From SDSS
    filter_lo = ["g", "r", "i", "z"]
    lambda_lo = np.array([4640, 6580, 8060, 9000]) # Taken from Wikipedia
    # lambda_lo = [4640]
    lambda_lo = 4640
    zlo = 0.3 # TODO: Figure out issues with zlo < ~0.3
    zhi = 2
    lambda_hi = lambda_lo
    scllo = pixscale.value
    zplo = [22.5, 22.5, 22.5, 22.5] # magnitudes
    tlo = [1, 1, 1, 1] 
    filter_hi = filter_lo
    sclhi = pixscale.value
    zphi = 22.5
    thi = 1

    im_out_file = "output.fits"
    psf_out_file = "output_psf.fits"

    # TODO: Currently produces weird looking images, need to check indices of some parts of the code
    ferengi.ferengi(sky, im, imerr, psflo, erro0_mag, psfhi, lambda_lo, filter_lo, zlo, scllo, zplo, tlo, lambda_hi, filter_hi, zhi, sclhi, zphi, thi, im_out_file, psf_out_file, noflux=False, evo=False, noconv=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hello")
    # parser.add_argument("-p", help="Path to fits simulation image", default=".")
    # args = parser.parse_args()

    # args.p = Path(args.p)
    p = "/home/ryans/Documents/Photometric Decomp/Outputs/TNG50/PolarRing/TNG167392/"
    im = list()
    bands = "griz"
    for band in bands:
        im.append(fits.open(p + f"TNG167392_E_SDSS_{band}.fits")[0].data)
    im = np.array(im)
    im = np.moveaxis(im, 0, -1)
    # im = im.reshape(im.shape[1], im.shape[2], len(bands))
    # In the shape of (x, y, bands)

    # im = fits.open(p + f"TNG167392_E_SDSS_g.fits")[0].data

    psf = list()
    for band in bands:
        # psf.append(fits.open(p + f"psf_core_{band}.fits")[0].data)
        psf.append(fits.open(p + f"psf_patched_g.fits")[0].data)
    psf = np.array(psf)
    psf = np.moveaxis(psf, 0, -1)
    # In the shape of (x, y, bands)
    # psf = fits.open(p + f"psf_core_g.fits")[0].data

    
    rng = np.random.default_rng()
    mu = 0
    stddev = np.sqrt(4.00737e-06) # Taken from a DESI image
    sky = rng.normal(mu, stddev, size=np.shape(im)[:2]) # nmgy

    main(im, psf, sky)