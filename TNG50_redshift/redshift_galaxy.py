import ferengi
import argparse
import numpy as np
from pathlib import Path
from astropy.io import fits
import astropy.units as u
import matplotlib.pyplot as plt
import os
import glob
import multiprocessing as mp
from mpi4py import MPI
from mpi4py.futures import MPIPoolExecutor
from functools import partial
from astropy.cosmology import FlatLambdaCDM
cosmo = FlatLambdaCDM(H0=70, Om0=0.3)

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

def simunits2cts2(simim, pixarea, expt):
    simim = simim * 10 ** 6 * 3.631e-6 * pixarea # MJy/sr to nmgy /pixel
    for k in range(np.shape(simim)[-1]):
        simim[:, :, k] = ferengi.maggies2cts(simim[:,:,k], expt=expt[k], zp=22.5)
    return simim 

def simunits2maggies2(simim, pixarea):
    # simim = simim * 10 ** 6 / 3.631e-6 * pixarea # MJy/sr to nmgy /pixel
    simim = simim * 10 ** 6 / 3.631e-6 # MJy/sr to nmgy /sr
    return simim 

def simunits2maggies(simim, pixarea):
    simim = simim * 10 ** 6 * 10 ** (-23) * pixarea # MJy/sr to ergs/s/cm**2/Hz(fnu) /pixel
    for k in range(np.shape(simim)[-1]):
        simim[:, :, k] = ferengi.fnu2maggies(simim[:, :, k])
    return simim # fnu/pixel -> maggies/pixel -> cnts/pixel

def cts2simunits(im, pixarea, expt):
    for k in range(np.shape(im)[-1]):
        im[:, :, k] = ferengi.maggies2fnu(ferengi.cts2maggies(im[:, :, k], expt=expt[k], zp=22.5))
    im = im / (10 ** 6 * 10 ** (-23)) / pixarea
    return im 

def redshift(im, psf, sky, out_bands, galaxy_name, lerp_scheme):
    # 1 pixel is 100 pc 
    # TODO: Figure out the "distance" to the object in the SKIRT image
    pixscale = 0.262 * u.arcsecond
    pixarea = pixscale ** 2
    pixarea = pixarea.to(u.sr).value
    sclhi = pixscale.value

    zlo = 0 # TODO: Figure out issues with low zlo # Seems to be that the PSF is being downscaled way too much (becomes empty)
    # In part due to the scllo pixel scale being too small, so magnification (and resulting image) is smaller

    scllo = np.arctan2(100*u.pc, cosmo.luminosity_distance(zlo)).to(u.arcsec).value
    pixscale_lo = scllo * u.arcsecond
    pixarea_lo = pixscale_lo ** 2
    pixarea_lo = pixarea_lo.to(u.sr).value

    # scllo = pixscale.value 
    # scllo = sclhi

    tlo = [1, 1, 1, 1]
    tlo = [1, 1, 1, 1, 1]
    tlo = [t/100 for t in tlo]

    # im = im * 10**(-1) # Needed to get the order of magnitude right 
    # im = simunits2cts2(im, pixarea_lo, tlo) 
    im = simunits2maggies2(im, pixarea_lo) 
    # im = simunits2cts(im, pixarea_lo, tlo) 

    # sky = np.zeros_like(sky) 

    thi = 200 # Based vaguely off of DESI images


    imerr = np.zeros_like(im) # Poisson Noise alread added

    psflo = psf
    erro0_mag = np.array([0.05, 0.02, 0.02, 0.02, 0.03]) # From SDSS
    filter_lo = ["u", "g", "r", "i", "z"]
    lambda_lo = np.array([4640, 6580, 8060, 9000]) # Taken from Wikipedia
    lambda_lo = np.array([3551, 4686, 6166, 7480, 8932]) # Taken from https://www.sdss4.org/instruments/camera/


    zplo = [22.5, 22.5, 22.5, 22.5] # magnitudes
    zplo = [22.5, 22.5, 22.5, 22.5, 22.5] # magnitudes
    zphi = 22.5

    # band_wav = {
    #     "g": 4640,
    #     "r": 6580,
    #     "i": 8060,
    #     "z": 9000
    # } # Taken from Wikipedia, Angstroms

    band_wav = {
        "g": 4686,
        "r": 6165,
        "i": 7481,
        "z": 8931
    } #http://astro.vaporia.com/start/sdss.html, Angstroms

    band_wav = {
        "u": 3551,
        "g": 4686,
        "r": 6166,
        "i": 7480,
        "z": 8932
    } #https://www.sdss4.org/instruments/camera/, Angstroms

    # sky = sky*0
    for zhi in [0.05, 0.1, 0.15, 0.2]:
    # for zhi in [zlo+0.001]:
        for k,band in enumerate(out_bands):
            filter_hi = band
            lambda_hi = band_wav[band]
            im_out_file = f"{galaxy_name}_{band}_z={zhi}.fits"
            psf_out_file = f"{galaxy_name}_psf_{band}_recon_z={zhi}.fits"
            psfhi = psflo[:, :, k]
            ferengi.ferengi(sky, im, imerr, psflo, erro0_mag, psfhi, lambda_lo, filter_lo, zlo, scllo, zplo, tlo, lambda_hi, filter_hi, zhi, sclhi, zphi, thi, im_out_file, psf_out_file, noflux=False, evo=None, noconv=False, lerp_scheme=lerp_scheme)
    # os.chdir("../")

def load_data_and_run(galaxy_name, p, psf_path, out_path, lerp_scheme):
    # TODO: Come back and make this take input and output bands as an arugment (not really important right now)
    p = os.path.join(p, galaxy_name)
    try:
        im = list()
        for band in "ugriz":
            im.append(fits.getdata(os.path.join(p, f"{galaxy_name}_E_SDSS_{band}.fits"))) # Assume everything is in the same parent directory
        im = np.array(im)
        im = np.moveaxis(im, 0, -1)
        # In the shape of (x, y, bands)

        rng = np.random.default_rng()
        mu = 0
        stddevs = np.array([0.00247579, 0.00247579, 0.0037597, 0.0074736, 0.0108026]) # Taken from a DESI image; u, g, r, i, and z band

        # sky_shape = (np.shape(im)[0]*3, np.shape(im)[1]*3) # Adjust as needed to make the code not error out if the sky is too small
        sky_shape = (np.shape(im)[0]*10, np.shape(im)[1]*10) # Adjust as needed to make the code not error out if the sky is too small
        sky = rng.normal(mu, stddevs, size=(sky_shape[0], sky_shape[1], 5)) # nmgy
        # sky = 100*sky # Needed to make it even visible (may have to change thi or something)

        psf = list()
        for band in "ugriz":
            # psf.append(fits.getdata(os.path.join(psf_path, f"psf_patched_{band}.fits"))) # Assume everything is in the same parent directory
            # if band != "z" and band != "u":
            if band != "u":
                psf.append(fits.getdata(os.path.join(psf_path, f"psf_patched_{band}.fits"))) 
            else:
                psf.append(fits.getdata(os.path.join(psf_path, f"psf_patched_g.fits"))) 
        psf = np.array(psf)
        psf = np.moveaxis(psf, 0, -1)
        # In the shape of (x, y, bands)func

        print(f"Performing redshift on {galaxy_name}")
        Path(os.path.join(out_path, f'{galaxy_name}')).mkdir(exist_ok=True, parents=True)
        os.chdir(Path(os.path.join(out_path, f'{galaxy_name}')))
        redshift(im, psf, sky, out_bands=["g", "r", "i", "z"], galaxy_name=galaxy_name, lerp_scheme=lerp_scheme)
    except MemoryError as e:
        print("Memory Error!")
        print(e)
        return -1
    except Exception as e:
        print(e)
        return -1

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                     description="Artificially redshift TNG50 simulation galaxies with FERENGI.")
    parser.add_argument("-p", help="Path to folder containing fits simulation images", default=".")
    parser.add_argument("--psf", help="Path to folder containing PSF images", default=".")
    parser.add_argument("-o", help="Output of high-redshift PSFs and FITs images", default=None)
    parser.add_argument("-ib", help="Input bands", nargs="+", choices=["g", "r", "i", "z"], default=["g", "r", "i", "z"])
    parser.add_argument("-b", help="Output bands", nargs="+", choices=["g", "r", "i", "z"], default=["g", "r", "i", "z"])
    parser.add_argument("-t", help="Type of SKIRT output to use", choices=["SDSS"], default="SDSS")
    parser.add_argument("-n", help="Number of parallel redshifts", default=1, type=int)
    parser.add_argument("-l", help="Lerp Scheme", default=0, type=int)
    args = parser.parse_args()

    p = Path(args.p).resolve() 
    if args.o == None:
        o = p
    else:
        o = Path(args.o).resolve()

    psf_path = Path(args.psf).resolve()
    lerp_scheme = args.l

    galaxy_names = []

    structure = os.walk(p)
    for root, dirs, files in structure:
        if not(files == []):
            galaxy_name = os.path.basename(root)
            galaxy_names.append(galaxy_name)


    # for galaxy in galaxy_names:
    #     load_data_and_run(galaxy, p, psf_path, o, lerp_scheme)
    part = partial(load_data_and_run, p=p, psf_path=psf_path, out_path=o, lerp_scheme=lerp_scheme)
    with MPIPoolExecutor(max_workers=args.n) as pool:
        pool.map(part, galaxy_names)

    # pool = mp.Pool(processes=args.n)
    # pool.map(func, galaxy_names)
    