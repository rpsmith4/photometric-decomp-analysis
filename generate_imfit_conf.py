import glob
import matplotlib.pyplot as plt
import numpy as np
import os
import math

from astropy.convolution import convolve
from astropy.io import fits

from photutils.background import Background2D, MedianBackground
from photutils.segmentation import detect_sources, make_2dgaussian_kernel, SourceCatalog, deblend_sources

from scipy.ndimage import rotate
import scipy.integrate as integrate
import scipy
import pyimfit

import argparse
from pathlib import Path

from photutils.isophote import EllipseGeometry
from photutils.aperture import EllipticalAperture
from photutils.isophote import Ellipse
from photutils.isophote import build_ellipse_model

import astropy.units as u

import glob
import itertools
from threading import Thread
import multiprocessing as mp
import warnings
warnings.filterwarnings("ignore")

def get_PA(img):
    # Estimate background
    bkg_estimator = MedianBackground()
    kernel = make_2dgaussian_kernel(3.0, size=5)  # FWHM = 3.0
    img_c = convolve(img, kernel)

    bkg = Background2D(img_c, (50, 50), filter_size=(3, 3), bkg_estimator=bkg_estimator)


    # Detect objects in the image
    threshold = 1.5 * bkg.background_rms
    segment_map = detect_sources(img_c, threshold, npixels=10)

    # Catalogize the identified objects:
    cat = SourceCatalog(img, segment_map, convolved_data=img_c)
    tab = cat.to_table()

    # Find the target galaxy in the image:
    ys, xs = np.shape(img)
    ys = ys/2.
    xs = xs/2.
    
    xmin = list(tab['bbox_xmin'])
    xmax = list(tab['bbox_xmax'])
    ymin = list(tab['bbox_ymin'])
    ymax = list(tab['bbox_ymax'])  
    
    # In fact, some sources in the image can be larger than the target galaxy, so we select the source which intersects the center:
    for ii in range(len(xmin)):
        if xs > xmin[ii] and xs < xmax[ii] and ys > ymin[ii] and ys < ymax[ii]:
            label_gal = ii + 1

    gal_params = cat.get_labels(label_gal).to_table()

    PA = gal_params['orientation'].value[0] # the position angle

    return PA

def fit_iso(img, geometry):

    aper = EllipticalAperture((geometry.x0, geometry.y0), geometry.sma, geometry.sma * (1 - geometry.eps), geometry.pa)

    ellipse = Ellipse(img, geometry)
    isolist = ellipse.fit_image()

    return isolist

def normAngle(angle):
    # Return angle to be between -180 and 180
    angle = angle % 360
    if np.abs(angle) > 180:
        angle = angle - np.sign(angle) * 360
    return angle

def get_PA2(img): # Probably better
    shape = img.shape
    x0 = shape[0]/2
    y0 = shape[1]/2
    I_e = np.max(img)/2

    #model_image = build_ellipse_model(data.shape, isolist)
    #residual = data - model_image
    min_residual = np.inf
    out_isolist = None

    for init_pa in [0, 45, 90]:
        for init_sma_fact in [10, 30, 50]:
            for init_eps in [0.0, 0.3, 0.5]:
                geometry = EllipseGeometry(x0=x0, y0=y0, sma=img.shape[0]/init_sma_fact, eps=init_eps, pa=init_pa * np.pi / 180.0)
                try:
                    # print(f"Trying {init_pa}, {init_sma_fact}, {init_eps}")
                    isolist = fit_iso(img, geometry)
                    table = isolist.to_table()
                    model_image = build_ellipse_model(img.shape, isolist)
                    residual = img - model_image
                    square_residual = np.sum(np.square(residual))
                    if square_residual < min_residual:
                        min_residual = square_residual
                        out_isolist = isolist
                except Exception as e:
                    continue
                    # print(e)


    table = out_isolist.to_table()
    PA = table["pa"]

    # host_PA = np.average(PA[(table["intens"] > I_e/100) & (table["intens"] < I_e/10)])
    # polar_PA = np.average(PA[table["intens"] < I_e/10])
    # host_PA = np.average(PA[:int(np.size(PA)/4)])
    # polar_PA = np.average(PA[int(np.size(PA)/4 * 3):])

    host_PA = np.average(PA[:int(np.size(PA)/2)])
    polar_PA = np.average(PA[-3:])
    host_PA = np.average(PA[10:10+int(np.size(PA)/4)]).value 
    polar_PA = np.average(PA[-3:]).value

    host_PA = normAngle(host_PA)
    polar_PA = normAngle(polar_PA)

    if np.abs(host_PA - polar_PA) < 30 or np.abs(host_PA - polar_PA) > 150: # Check to see if the angle between then is close to the same line
        polar_PA = host_PA + 90

    return host_PA, polar_PA



def init_guess_2_sersic(img, pol_str_type, model_desc, band):
    model = pyimfit.SimpleModelDescription()
    shape = img.shape
    model.x0.setValue(shape[0]/2 - 1, [shape[0]/2 - 30, shape[0]/2 + 30])
    model.y0.setValue(shape[1]/2 - 1, [shape[1]/2 - 30, shape[1]/2 + 30])

    if pol_str_type == "ring":
        # Inner Sersic (Host)
        # Assuming galaxy is at the center
        host = pyimfit.make_imfit_function("Sersic", label="Host")

        # img_reduce = img.copy()
        # I_e = np.max(img)/2
        # img_reduce[img_reduce < I_e] = 0
        
        host_PA, polar_PA = get_PA2(img)
        host_PA = host_PA - 90 # CCW from +x axis to CCW from +y axis 
        polar_PA = polar_PA - 90

        host.PA.setValue(host_PA, [host_PA - 10, host_PA + 10]) # Hard to find probably, might get from the IMAN script ellipse fitting thing (might mask low intensities) # Actually seems to work well with the isophote fitting

        host.ell.setValue(0.5, [0, 0.75]) # Probably doesn't really have to be changed

        img_rot_host = scipy.ndimage.rotate(img, angle=host_PA + 90)
        rad_slc = img_rot_host[int(img_rot_host.shape[0]/2), int(img_rot_host.shape[0]/2):]
        S = rad_slc * 2 * np.pi * (np.arange(int(rad_slc.size)))
        B = np.cumsum(S)
        B_e = B[-1]/2
        r_e = np.argmin(np.abs(B - B_e)) # num of pixels from center
        I_e = rad_slc[r_e]

        host.I_e.setValue(I_e, [I_e/10, I_e*10]) 
        host.r_e.setValue(r_e, [r_e/2, r_e*5]) # Maybe get an azimuthal average
        host.n.setValue(3, [1, 10]) # Maybe keep as is

        model.addFunction(host)

        # Outer Sersic (Polar)
        # Assuming galaxy is at the center

        polar = pyimfit.make_imfit_function("Sersic", label="Polar")
        polar.PA.setValue(polar_PA, [polar_PA - 10, polar_PA + 10]) # Can probably just be the host PA +/- 90
        polar.ell.setValue(0.5, [0, 0.75]) # Probably doesn't really have to be changed

        img_rot_polar = scipy.ndimage.rotate(img, angle=polar_PA + 90)
        rad_slc = img_rot_polar[int(img_rot_polar.shape[0]/2), int(img_rot_polar.shape[0]/2):]
        S = rad_slc * 2 * np.pi * (np.arange(int(rad_slc.size)))
        B = np.cumsum(S)
        B_e = B[-1]/2
        r_e = np.argmin(np.abs(B - B_e)) # num of pixels from center
        I_e = rad_slc[r_e]

        polar.I_e.setValue(I_e/10, [I_e/1000, I_e]) # Probably signifigantly dimmer

        polar.r_e.setValue(r_e, [r_e/10, r_e*10]) # Maybe get an azimuthal average
        polar.n.setValue(3, [1, 10]) # Maybe keep as is

        model.addFunction(polar)

    # Can perhaps run the DE solve on just the host (with isophote clipped) then on the polar component somehow (maybe by removing the center handful of pixels)
    model_desc[band] = str(model)

    # return model

def main(args):
    if not(args.p == None):
        os.chdir(Path(args.p))
    manager = mp.Manager()
    model_desc = manager.dict()

    if args.r:
        structure = os.walk(".")
        for root, dirs, files in structure:
            if not(files == []):
                img_files = sorted(glob.glob(os.path.join(Path(root), "image_?.fits")))

                # invvar_files = sorted(glob.glob(os.path.join(Path(root), "image_?_invvar.fits")))
                # # Assuming the use of the patched PSF
                # psf_files = sorted(glob.glob(os.path.join(Path(root), "psf_patched_?.fits")))
                # assert(len(img_files) == len(invvar_files) == len(psf_files)), "Amount of image, invvar, and psf files unequal!"
                jobs = []
                for img_file in img_files:
                    band = img_file[-6] # Yes I know this is not the best way
                    # outputs[band] = None
                    if not(f"config_{band}.dat" in files) or args.overwrite:
                        print(f"Generating configs for {img_file}")
                        img = fits.getdata(img_file)

                        if args.mask:
                            mask = fits.getdata(os.path.join(Path(root), "image_mask.fits"))
                            img = img * (1-mask)

                        p = mp.Process(target = init_guess_2_sersic, args=(img, str(args.type).lower(), model_desc, band))
                        jobs.append(p)  
                
                for p in jobs:
                    p.start()

                for p in jobs:
                    p.join()

                for band in model_desc.keys():
                    with open(os.path.join(Path(root), f"config_{band}.dat"), "w") as f:
                        f.write(model_desc[band])

    else:
        img_files = sorted(glob.glob(os.path.join(Path("."), "image_?.fits")))

        jobs = []
        for img_file in img_files:
            band = img_file[-6] # Yes I know this is not the best way
            # outputs[band] = None
            if not(f"config_{band}.dat" in files) or args.overwrite:
                print(f"Generating configs for {img_file}")
                img = fits.getdata(img_file)

                if args.mask:
                    mask = fits.getdata(os.path.join(Path("."), "image_mask.fits"))
                    img = img * (1-mask)

                p = mp.Process(target = init_guess_2_sersic, args=(img, str(args.type).lower(), model_desc, band))
                jobs.append(p)  
        
        for p in jobs:
            p.start()

        for p in jobs:
            p.join()

        for band in model_desc.keys():
            with open(os.path.join(Path("."), f"config_{band}.dat"), "w") as f:
                f.write(model_desc[band])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hello")
    parser.add_argument("-p", help="Path to file/folder containing galaxy FITS")
    parser.add_argument("--overwrite", help="Overwrite existing config files", action="store_true")
    parser.add_argument("--mask", help="Use the mask to guess initial values", action="store_true")
    parser.add_argument("--type", help="Type of polar structure", choices=["ring", "bulge", "halo"])
    parser.add_argument("-r", help="Recursively go into subfolders (assumes that fits data is at the end of the filetree)", action="store_true")
    

    args = parser.parse_args()
    main(args)
