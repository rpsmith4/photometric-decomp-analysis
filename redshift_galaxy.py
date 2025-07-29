import ferengi
import argparse
import numpy as np
from pathlib import Path
from astropy.io import fits

def simunits2cnts(simim, pixscale):
    simim = simim * 10 ** 6 * 10 ** (-23) * pixscale # MJy/sr to ergs/s/cm**2/Hz
    return ferengi.maggies2cts(ferengi.fnu2maggies(simim), expt=1, zp=22.5)

def main(im, psf, sky):
    pixscale = 0.262
    im = simunits2cnts(im, pixscale) 
    # im = ferengi.maggies2cts(im*10**(9), expt=1, zp=22.5)
    sky = simunits2cnts(sky, pixscale)
    lam = 1 
    imerr = np.random.poisson(lam=lam, size=im.shape)
    # psflo = np.array([psf])
    # psfhi = np.array([psf])
    psflo = psf
    psfhi = psf
    erro0_mag = [0, 0, 0, 0]
    filter_lo = ["g", "r", "i", "z"]
    filter_lo = ["g"]
    lambda_lo = [4640, 6580, 8060, 9000]
    # lambda_lo = [4640]
    lambda_lo = 4640
    zlo = 0.03
    zhi = 2
    lambda_hi = lambda_lo
    lambda_hi = 4640
    scllo = 0.262
    zplo = 22.5
    tlo = [1, 1, 1, 1]
    tlo = [1]
    filter_hi = filter_lo
    sclhi = 0.262
    zphi = 22.5
    thi = [1, 1, 1, 1]
    thi = [1]

    im_out_file = "output.fits"
    psf_out_file = "output_psf.fits"

    # im = np.array([im])
    # imerr = np.array([imerr])
    ferengi.ferengi(sky, im, imerr, psflo, erro0_mag, psfhi, lambda_lo, filter_lo, zlo, scllo, zplo, tlo, lambda_hi, filter_hi, zhi, sclhi, zphi, thi, im_out_file, psf_out_file, noflux=False, evo=None, noconv=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hello")
    # parser.add_argument("-p", help="Path to fits simulation image", default=".")
    # args = parser.parse_args()

    # args.p = Path(args.p)
    p = "/home/ryans/Documents/Photometric Decomp/Outputs/TNG50/PolarRing/TNG167392/"
    im = fits.open(p + "TNG167392_E_SDSS_g.fits")[0].data
    psf = fits.open(p + "psf_patched_g.fits")[0].data
    
    rng = np.random.default_rng()
    mu = 0.000799898
    stddev = np.sqrt(4.00737e-06)
    sky = rng.normal(mu, stddev, size=np.shape(im))/10000
    # sky = np.zeros_like(im)

    main(im, psf, sky)