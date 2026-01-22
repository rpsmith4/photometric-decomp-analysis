import RGB_gui
# import RGB_gui_two
import astropy.io.fits as fits
import argparse
from pathlib import Path
import os
import glob
import re
import numpy as np
from functools import partial
import astropy.units as u
from mpi4py import MPI
from mpi4py.futures import MPIPoolExecutor
from scipy.optimize import curve_fit
from functools import partial
import astropy
from astropy.cosmology import FlatLambdaCDM
cosmo = FlatLambdaCDM(H0=70, Om0=0.3)

def ferengi_odd_n_square(psf0, centre=None):
    """
    Makes the input PSF array square, the number of pixels along each axis odd,
    and centers the result.
    """
    psf = np.copy(psf0)

    # Resize to odd number of pixels
    sz_x, sz_y = psf.shape
    if (sz_x % 2) == 0: # If even, add a row
        psf = np.pad(psf, ((0, 1), (0, 0)), 'constant')
    sz_x, sz_y = psf.shape # Update sizes
    if (sz_y % 2) == 0: # If even, add a column
        psf = np.pad(psf, ((0, 0), (0, 1)), 'constant')

    # Make array square
    sz_x, sz_y = psf.shape
    if sz_x > sz_y:
        pad_y = sz_x - sz_y
        psf = np.pad(psf, ((0, 0), (0, pad_y)), 'constant')
    elif sz_y > sz_x:
        pad_x = sz_y - sz_x
        psf = np.pad(psf, ((0, pad_x), (0, 0)), 'constant')
    
    # Center array
    if centre is not None and len(centre) == 2:
        # Assuming centre is (shift_x, shift_y)
        psf = np.roll(psf, int(np.round(centre[0])), axis=0)
        psf = np.roll(psf, int(np.round(centre[1])), axis=1)
    else:
        # Use 2D Gaussian fit for centering
        # This part of the original IDL code is quite complex and uses tricks with rebin and padding.
        # A direct 2D Gaussian fit to find the centroid and then shifting is more robust in Python.
        
        # Simple 2D Gaussian function for fitting
        def gaussian_2d_fit(coords, amplitude, xo, yo, sigma_x, sigma_y, theta, offset):
            x, y = coords
            a = (np.cos(theta)**2)/(2*sigma_x**2) + (np.sin(theta)**2)/(2*sigma_y**2)
            b = -(np.sin(2*theta))/(4*sigma_x**2) + (np.sin(2*theta))/(4*sigma_y**2)
            c = (np.sin(theta)**2)/(2*sigma_x**2) + (np.cos(theta)**2)/(2*sigma_y**2)
            g = offset + amplitude*np.exp( - (a*((x-xo)**2) + 2*b*(x-xo)*(y-yo) + c*((y-yo)**2)))
            return g.ravel()

        current_sz_x, current_sz_y = psf.shape
        x_coords, y_coords = np.meshgrid(np.arange(current_sz_x), np.arange(current_sz_y))
        
        # Initial guess for the Gaussian fit
        initial_guess = (psf.max(), current_sz_x/2, current_sz_y/2, current_sz_x/4, current_sz_y/4, 0, psf.min())
        try:
            popt, _ = curve_fit(gaussian_2d_fit, (x_coords, y_coords), psf.ravel(), p0=initial_guess)
            centroid_x, centroid_y = popt[1], popt[2]
        except RuntimeError: # If fitting fails, fall back to center of mass or peak pixel
            print("Gaussian fit for PSF centering failed, falling back to peak pixel.")
            peak_idx = np.unravel_index(np.argmax(psf), psf.shape)
            centroid_x, centroid_y = peak_idx[0], peak_idx[1]

        # Calculate shift needed to move centroid to geometric center
        geometric_center_x = (current_sz_x - 1) / 2.0
        geometric_center_y = (current_sz_y - 1) / 2.0
        
        shift_x = geometric_center_x - centroid_x
        shift_y = geometric_center_y - centroid_y

        # Apply the shift by rolling and then interpolating for sub-pixel accuracy if needed
        # For simplicity, using np.roll for integer shifts. For sub-pixel, use scipy.ndimage.shift.
        # IDL's SSHIFT2D is a circular shift, which is important for Fourier transforms.
        psf = np.roll(psf, int(np.round(shift_x)), axis=0)
        psf = np.roll(psf, int(np.round(shift_y)), axis=1)

    return psf

def fnu2maggies(fnu):
    """
    Converts flux in erg s-1 Hz-1 cm-2 to maggies.
    """
    return fnu / 3631e-23

def simunits2maggies(simim, pixarea):
    simim = simim * 10 ** 6 / 3.631e-6 * pixarea
    return simim 

def maggies2simunits(im, pixarea):
    for k in range(np.shape(im)[0]):
        im[k, :, :] = im[k, :, :] * 3631e23
    im = im /(10 ** 6 * 10 **(-23) * pixarea)
    return im

def TNG_rgb(galaxy_name, path, path_redshift, psf_path, out):
    print(f"Making image for {galaxy_name}")

    # zs = [0, 0.05, 0.1, 0.15, 0.2] # List of redshifts
    zs = [0.003] # List of redshifts

    bands = "grz"
    # pixscale = 0.262 * u.arcsecond
    # # pixarea = pixscale ** 2
    # # pixarea = pixarea.to(u.sr).value
    zlo = 0.00001 # Arbitary small distance
    scllo = np.arctan2(100*u.pc, cosmo.luminosity_distance(zlo)).to(u.arcsec)
    pixarea = (scllo **2).to(u.sr).value # This is just for the z=0 images

    for z_factor in zs:
        if z_factor == 0:
            filenames = [os.path.join(path, galaxy_name, f"{galaxy_name}_E_SDSS_{band}.fits") for band in bands]
            im = [fits.getdata(filename) for filename in filenames]
            im = np.array(im) # Shape of (3, x, x), order of g, r, z

            # im = simunits2maggies(im, pixarea) * 10 **9 # maggie to nmgy
            im = simunits2maggies(im, pixarea) * 10 **9 # maggie to nmgy
            im = im/4e6 # Fudge factor as to not make simulation RGB images mega bright
            im = im/4e9 

            rng = np.random.default_rng()
            mu = 0
            stddevs = np.array([0.00247579, 0.0037597, 0.0108026]) # Taken from a DESI image; g, r, and z band
            sky_shape = (np.shape(im)[1], np.shape(im)[2]) 
            sky = rng.normal(mu, stddevs, size=(sky_shape[0], sky_shape[1], 3)) # nmgy


            psf = list()
            # for band in "grz":
            for band in "grg":
                psf.append(fits.getdata(os.path.join(psf_path, f"psf_patched_{band}.fits"))) 

            psf = np.array(psf)
            psf = np.moveaxis(psf, 0, -1)

            # Convolve with the PSF
            g = astropy.convolution.convolve(im[0], ferengi_odd_n_square(psf[:, :, 0]), boundary="fill", fill_value=0)
            r = astropy.convolution.convolve(im[1], ferengi_odd_n_square(psf[:, :, 1]), boundary="fill", fill_value=0)
            z = astropy.convolution.convolve(im[2], ferengi_odd_n_square(psf[:, :, 2]), boundary="fill", fill_value=0)

            # Add sky
            g = g + sky[:, :, 0]
            r = r + sky[:, :, 1]
            z = z + sky[:, :, 2]

        else:
            filenames = [os.path.join(path_redshift, galaxy_name, f"{galaxy_name}_{band}_z={z_factor}.fits") for band in bands]
            im = [fits.getdata(filename) for filename in filenames]
            im = np.array(im) # Shape of (3, x, x), order of g, r, z
            g = im[0]
            r = im[1]
            z = im[2]

        # rgb = RGB_gui.render_rgb_with_isomask(
        # g, r, z,
        # zeropoint=22.5, pixscale=0.262,
        # scale='linear', bias=0., contrast=1.0, brightness=1.,
        # mu_r=25.5, sigma=1.4)
        if z_factor == 0:
            isTNG = True
        else:
            isTNG = False

        rgb = RGB_gui.render_rgb_with_isomask(
        g, r, z,
        zeropoint=22.5, pixscale=0.262,
        scale='linear', bias=0., contrast=1.0, brightness=1.0,
        mu_r=25.5, sigma=0.0)
        Path(os.path.join(out, f'{galaxy_name}')).mkdir(exist_ok=True, parents=True)
        RGB_gui.save_rgb_annotated(rgb, 0.262, os.path.join(out, f'{galaxy_name}', f'{galaxy_name}_z={z_factor}.png'), galaxy_name=galaxy_name, isTNG=isTNG)  

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                     description="")
    parser.add_argument("-p", help="Path to folder containing fits simulation images", default=".")
    parser.add_argument("--pr", help="Path to folder containing redshifted fits simulation images", default=None)
    parser.add_argument("--psf", help="Path to folder containing psf", default=".")
    parser.add_argument("-o", help="Output Folder", default=None)
    args = parser.parse_args()

    p = Path(args.p)
    if args.o == None:
        o = p
    else:
        o = Path(args.o)

    if args.pr == None:
        pr = p
    else:
        pr = Path(args.pr)
    
    psf_path = Path(args.psf)


    structure = os.walk(p)
    galaxy_names = []
    for root, dirs, files in structure:
        if not(files == []):
            galaxy_name = os.path.basename(root) 
            galaxy_names.append(galaxy_name)

    part = partial(TNG_rgb, path=p, path_redshift=pr, psf_path=psf_path, out=o)
    # with MPIPoolExecutor(max_workers=15) as pool:
    #     pool.map(part, galaxy_names)
    for galaxy_name in galaxy_names:
        TNG_rgb(galaxy_name, p, pr, psf_path, o)
