import ferengi
import argparse
import numpy as np
from pathlib import Path
from astropy.io import fits
import astropy.units as u
import matplotlib.pyplot as plt
import os
import glob

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
def simunits2cts(simim, pixarea, expt):
    simim = simim * 10 ** 6 * 10 ** (-23) * pixarea # MJy/sr to ergs/s/cm**2/Hz(fnu) /pixel
    for k in range(np.shape(simim)[-1]):
        simim[:, :, k] = ferengi.maggies2cts(ferengi.fnu2maggies(simim[:, :, k]), expt=expt[k], zp=22.5)
    return simim # fnu/pixel -> maggies/pixel -> cnts/pixel

def cts2simunits(im, pixarea, expt):
    for k in range(np.shape(im)[-1]):
        im[:, :, k] = ferengi.maggies2fnu(ferengi.cts2maggies(im[:, :, k], expt=expt[k], zp=22.5))
    im = im / (10 ** 6 * 10 ** (-23)) / pixarea
    return im 

def main(im, psf, sky, out_bands, galaxy_name):
    # 1 pixel is 100 pc 
    # TODO: Figure out the "distance" to the object in the SKIRT image
    pixscale = 0.262 * u.arcsecond
    pixarea = pixscale ** 2
    pixarea = pixarea.to(u.sr).value
    scllo = pixscale.value * 2
    sclhi = pixscale.value

    tlo = [1, 1, 1, 1]
    tlo = [t/100 for t in tlo]
    # TODO: Figure out what the exposure time should be
    # for the sky as well as the simulated image

    im = simunits2cts(im, pixarea, tlo) 

    sky = ferengi.maggies2cts(sky * 10 ** (-9), expt=1, zp=22.5) # sky is in nmgy
    sky = np.zeros_like(sky) 

    thi = 200 # Based vaguely off of DESI images
    sky = sky/thi # cnts / second

    imerr = np.zeros_like(im) # Poisson Noise alread added

    psflo = psf
    psfhi = psf[:, :, 0] # Using just the g band psf
    # erro0_mag = np.array([0, 0, 0, 0])
    erro0_mag = np.array([0.02, 0.02, 0.02, 0.03]) # From SDSS
    filter_lo = ["g", "r", "i", "z"]
    lambda_lo = np.array([4640, 6580, 8060, 9000]) # Taken from Wikipedia

    zlo = 0.03 # TODO: Figure out issues with zlo < ~0.03 # Seems to be that the PSF is being downscaled way too much (becomes empty)
    # In part due to the scllo pixel scale being too small, so magnification (and resulting image) is smaller
    zhi = 0.5

    zplo = [22.5, 22.5, 22.5, 22.5] # magnitudes
    zphi = 22.5

    band_wav = {
        "g": 4640,
        "r": 6580,
        "i": 8060,
        "z": 9000
    } # Taken from Wikipedia, Angstroms

    for band in out_bands:
        filter_hi = band
        lambda_hi = band_wav[band]
        im_out_file = f"{galaxy_name}_{band}_redshift.fits"
        psf_out_file = f"{galaxy_name}_psf_{band}_recon.fits"
        ferengi.ferengi(sky, im, imerr, psflo, erro0_mag, psfhi, lambda_lo, filter_lo, zlo, scllo, zplo, tlo, lambda_hi, filter_hi, zhi, sclhi, zphi, thi, im_out_file, psf_out_file, noflux=False, evo=None, noconv=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                     description="Artificially redshift TNG50 simulation galaxies with FERENGI.")
    parser.add_argument("-p", nargs="+", help="Path to folder containing fits simulation images", default=".")
    parser.add_argument("-o", help="Output of high-redshift PSFs and FITs images", default=".")
    parser.add_argument("-b", help="Output bands", nargs="+", choices=["g", "r", "i", "z"], default=["g", "r", "i", "z"])
    parser.add_argument("-t", help="Type of SKIRT output to use", choices=["SDSS"], default="SDSS")
    args = parser.parse_args()
    ps = [Path(p).resolve() for p in args.p]

    if not any(os.path.isdir(p) for p in ps):
        galaxy_names = [p.name for p in ps]
        ps = [p.parent for p in ps]
    else:
        if args.t == "SDSS":
            galaxy_names = glob.glob(f"*SDSS*[{"".join(args.b)}]*", root_dir=ps[0])

    o = Path(args.o).resolve()
    os.chdir(o)

    psf = list()
    for band in args.b:
        psf.append(fits.getdata(os.path.join(ps[0], f"psf_patched_{band}.fits"))) # Assume everything is in the same parent directory
    psf = np.array(psf)
    psf = np.moveaxis(psf, 0, -1)
    # In the shape of (x, y, bands)

    rng = np.random.default_rng()
    mu = 0
    stddev = np.sqrt(4.00737e-06) # Taken from a DESI image

    if args.t == "SDSS":
        galaxy_names = set([galaxy_name.split('_')[0] for galaxy_name in galaxy_names])

    for galaxy_name in galaxy_names:
        im = list()
        for band in args.b:
            im.append(fits.getdata(os.path.join(ps[0], f"{galaxy_name}_E_SDSS_{band}.fits"))) # Assume everything is in the same parent directory
        im = np.array(im)
        im = np.moveaxis(im, 0, -1)
        # In the shape of (x, y, bands)

        sky_shape = (np.shape(im)[0]*3, np.shape(im)[1]*3) # Adjust as needed to make the code not error out if the sky is too small
        sky = rng.normal(mu, stddev, size=sky_shape) # nmgy

        print(f"Performing redshift on {galaxy_name}")
        main(im, psf, sky, out_bands=args.b, galaxy_name=galaxy_name)