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

def fix_close_angles(host_PA, polar_PA, pol_str_type):
    # Angles should be in the range [-180, 180]
    if pol_str_type == "ring" or pol_str_type == "halo":
        if np.abs(host_PA - polar_PA) < 15 or np.abs(host_PA - polar_PA) > 165: 
            host_PA = polar_PA + 90 # Often due to the host being rather circular and small
    if pol_str_type == "bulge":
        if np.abs(host_PA - polar_PA) < 15 or np.abs(host_PA - polar_PA) > 165: 
            polar_PA = host_PA + 90 # Often due to the polar part being small
    
    return host_PA, polar_PA

def get_PA2_and_table(img): # Probably better
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

    # host_PA = np.average(PA[:int(np.size(PA)/2)])
    # polar_PA = np.average(PA[-3:])
    host_PA = np.average(PA[10:10+int(np.size(PA)/4)]).value 
    polar_PA = np.average(PA[-3:]).value

    host_PA = normAngle(host_PA)
    polar_PA = normAngle(polar_PA)

    return host_PA, polar_PA, table

def fit_model(model, img):
    # Used for doing a minifit to get an initial guess
    imfit_imfitter = pyimfit.Imfit(model, maxThreads=1)
    imfit_imfitter.loadData(img, original_sky=10)
    result = imfit_imfitter.doFit(solver="DE")
    result_img = imfit_imfitter.getModelImage()
    return result, result_img

def init_guess_2_sersic(img, pol_str_type, model_desc, band):
    model = pyimfit.SimpleModelDescription()
    host_model = pyimfit.SimpleModelDescription()
    polar_model = pyimfit.SimpleModelDescription()
    shape = img.shape
    model.x0.setValue(shape[0]/2 - 1, [shape[0]/2 - 30, shape[0]/2 + 30])
    model.y0.setValue(shape[1]/2 - 1, [shape[1]/2 - 30, shape[1]/2 + 30])

    host_model.x0.setValue(shape[0]/2 - 1, [shape[0]/2 - 30, shape[0]/2 + 30])
    host_model.y0.setValue(shape[1]/2 - 1, [shape[1]/2 - 30, shape[1]/2 + 30])
    polar_model.x0.setValue(shape[0]/2 - 1, [shape[0]/2 - 30, shape[0]/2 + 30])
    polar_model.y0.setValue(shape[1]/2 - 1, [shape[1]/2 - 30, shape[1]/2 + 30])

    bounds_dict = { # Amount to add or subtract from guess/factor to multiply to guess
        "ring" : {
            "host": {
                "PA_bounds": [-10, 10],
                "ell_bounds": [-0.3, 0.4], # Possibly underestimated due to external dust/whatever
                "n_bounds": [-1, 5], # Possibly underestimated
                "I_e_factor": [1/10, 5], # May be underestimated except in the case that the host PA is perfect
                "r_e_factor": [1/2, 5] # Same as above
            },
            "polar": {
                "PA_bounds": [-15, 15], # May need more variablility since the polar part is very dim
                "ell_bounds": [-0.5, 0.5], # Hard to estimate for the polar part, so using a large range
                "n_bounds": [-1, 5], # Likely to be underestimated
                "I_e_factor": [1/100, 2], # Likely to be overestimated due to host component 
                "r_e_factor": [1/5, 2] # Similar story
            }
        },
        "bulge" : {
            "host": {
                "PA_bounds": [-10, 10],
                "ell_bounds": [-0.3, 0.4], # Possibly underestimated due to external dust/whatever
                "n_bounds": [-1, 5], # Possibly underestimated
                "I_e_factor": [1/10, 5], # May be underestimated except in the case that the host PA is perfect
                "r_e_factor": [1/2, 5] # Same as above
            },
            "polar": {
                "PA_bounds": [-15, 15], # May need more variablility since the polar part is very dim
                "ell_bounds": [-0.5, 0.5], # Hard to estimate for the polar part, so using a large range
                "n_bounds": [-1, 5], # Likely to be underestimated
                "I_e_factor": [1/100, 2], # Likely to be overestimated due to host component 
                "r_e_factor": [1/5, 2] # Similar story
            }
        },
        "halo" : {
            "host": {
                "PA_bounds": [-10, 10],
                "ell_bounds": [-0.3, 0.4], # Possibly underestimated due to external dust/whatever
                "n_bounds": [-1, 5], # Possibly underestimated
                "I_e_factor": [1/10, 5], # May be underestimated except in the case that the host PA is perfect
                "r_e_factor": [1/2, 5] # Same as above
            },
            "polar": {
                "PA_bounds": [-15, 15], # May need more variablility since the polar part is very dim
                "ell_bounds": [-0.5, 0.5], # Hard to estimate for the polar part, so using a large range
                "n_bounds": [-1, 5], # Likely to be underestimated
                "I_e_factor": [1/100, 2], # Likely to be overestimated due to host component 
                "r_e_factor": [1/5, 2] # Similar story
            }
        }
    }

    # Inner Sersic (Host)
    # Assuming galaxy is at the center
    host = pyimfit.make_imfit_function("Sersic", label="Host")
    
    host_PA, polar_PA, table = get_PA2_and_table(img)
    # host_PA, polar_PA = fix_close_angles(host_PA, polar_PA, pol_str_type)
    host_PA = host_PA - 90 # CCW from +x axis to CCW from +y axis 
    polar_PA = polar_PA - 90

    bounds_host = bounds_dict[pol_str_type]["host"]
    host.PA.setValue(host_PA, [host_PA + bounds_host["PA_bounds"][0], host_PA + bounds_host["PA_bounds"][1]]) # Hard to find probably, might get from the IMAN script ellipse fitting thing (might mask low intensities) # Actually seems to work well with the isophote fitting

    img_rot_host = scipy.ndimage.rotate(img, angle=host_PA + 90)
    rad_slc = img_rot_host[int(img_rot_host.shape[0]/2), int(img_rot_host.shape[0]/2):]
    S = rad_slc * 2 * np.pi * (np.arange(int(rad_slc.size)))
    B = np.cumsum(S)
    B_e = B[-1]/2
    r_e = np.argmin(np.abs(B - B_e)) # num of pixels from center
    I_e = rad_slc[r_e]
    if I_e < 0:
        I_e = -1 * I_e # Occasionaly negative due to sky subtractions, though magnitude matters more
    e = np.average(table["ellipticity"][:r_e +1])
    if e != e: # Sometimes is Nan for some reason
        e = 0.2

    # host.ell.setValue(e, [(e - bounds_host["ell_bounds"][0]).clip(min=0), (bounds_host["ell_bounds"][1]).clip(max=0.75)])
    host.ell.setValue(e, np.clip(np.array([(e + bounds_host["ell_bounds"][0]), e + (bounds_host["ell_bounds"][1])]), 0, 0.75))
    host.I_e.setValue(I_e, [I_e * bounds_host["I_e_factor"][0], I_e * bounds_host["I_e_factor"][1]]) 
    host.r_e.setValue(r_e, [r_e * bounds_host["r_e_factor"][0], r_e * bounds_host["r_e_factor"][1]]) # Maybe get an azimuthal average
    host.n.setValue(3, [0, 10]) # Maybe keep as is

    host_model.addFunction(host)
    img_host_reduced = img.copy()
    img_host_reduced[img_host_reduced < I_e/10] = 0
    if not(args.dont_fit):
        result, img_host_refined = fit_model(host_model, img_host_reduced)
        # Order of PA, ell, n, I_e, r_e 
        if result.fitConverged:
            fitparams = result.params[2:]
            PA = fitparams[0]
            ell = fitparams[1]
            n = fitparams[2]
            I_e = fitparams[3]
            r_e = fitparams[4]
            if I_e < 0:
                I_e = -1 * I_e # Occasionally negative due to sky subtractions, though magnitude matters more

            host.PA.setValue(PA, [PA + bounds_host["PA_bounds"][0], PA + bounds_host["PA_bounds"][1]])
            host.ell.setValue(ell, np.clip(np.array([(ell + bounds_host["ell_bounds"][0]), (ell + bounds_host["ell_bounds"][1])]), 0, 0.75)) # May overestimate
            host.n.setValue(n, np.clip(np.array([(n + bounds_host["n_bounds"][0]), (n + bounds_host["n_bounds"][1])]), 0, 10))# Likely to underestimate
            host.I_e.setValue(I_e, [I_e * bounds_host["I_e_factor"][0], I_e * bounds_host["I_e_factor"][1]]) # Likely to underestimate
            host.r_e.setValue(r_e, [r_e * bounds_host["r_e_factor"][0], r_e * bounds_host["r_e_factor"][1]]) # Likely to underestimate
        
    model.addFunction(host)

    # Outer Sersic (Polar)
    # Assuming galaxy is at the center

    polar = pyimfit.make_imfit_function("Sersic", label="Polar")

    bounds_polar = bounds_dict[pol_str_type]["polar"]

    polar.PA.setValue(polar_PA, [polar_PA + bounds_host["PA_bounds"][0], polar_PA + bounds_host["PA_bounds"][1]])
    img_rot_polar = scipy.ndimage.rotate(img, angle=polar_PA + 90)

    rad_slc = img_rot_polar[int(img_rot_polar.shape[0]/2), int(img_rot_polar.shape[0]/2):]
    S = rad_slc * 2 * np.pi * (np.arange(int(rad_slc.size)))
    B = np.cumsum(S)
    B_e = B[-1]/2
    r_e = np.argmin(np.abs(B - B_e)) # num of pixels from center
    I_e = rad_slc[r_e]
    if I_e < 0:
        I_e = -1 * I_e # Occasionally negative due to sky subtractions, though magnitude matters more
    e = np.average(table["ellipticity"][r_e-1:])
    if e != e: # Sometimes is Nan for some reason
        e = 0.2

    polar.ell.setValue(e, np.clip(np.array([(e + bounds_polar["ell_bounds"][0]), (e + bounds_polar["ell_bounds"][1])]), 0, 0.75)) # May overestimate
    # polar.n.setValue(3, np.clip(np.array([(n + bounds_polar["n_bounds"][0]), (n + bounds_polar["n_bounds"][1])]), 0, 10))# Likely to underestimate
    polar.n.setValue(3, [0, 10]) # Maybe leave as is
    polar.I_e.setValue(I_e, [I_e * bounds_polar["I_e_factor"][0], I_e * bounds_polar["I_e_factor"][1]]) # Likely to underestimate, probably very dim
    polar.r_e.setValue(r_e, [r_e * bounds_polar["r_e_factor"][0], r_e * bounds_polar["r_e_factor"][1]]) # Likely to underestimate, maybe get azimuthal average

    polar_model.addFunction(polar)
    if not(args.dont_fit):
        img_polar_reduced = img - img_host_refined # Trying to remove the host component to only fit the polar structure
        img_polar_reduced[img_polar_reduced < I_e/10] = 0
        result, img_polar_refined = fit_model(polar_model, img_polar_reduced)
        # order of pa, ell, n, i_e, r_e 
        if result.fitConverged:
            fitparams = result.params[2:]
            PA = fitparams[0]
            ell = fitparams[1]
            n = fitparams[2]
            I_e = fitparams[3]
            r_e = fitparams[4]
            if I_e < 0:
                I_e = -1 * I_e # occasionally negative due to sky subtractions, though magnitude matters more

            polar.PA.setValue(PA, [PA + bounds_polar["PA_bounds"][0], PA + bounds_polar["PA_bounds"][1]])
            polar.ell.setValue(ell, np.clip(np.array([(ell + bounds_polar["ell_bounds"][0]), (ell + bounds_polar["ell_bounds"][1])]), 0, 0.75)) # may overestimate
            polar.n.setValue(n, np.clip(np.array([(n + bounds_polar["n_bounds"][0]), (n + bounds_polar["n_bounds"][1])]), 0, 10))# likely to underestimate
            polar.I_e.setValue(I_e, [I_e * bounds_polar["I_e_factor"][0], I_e * bounds_polar["I_e_factor"][1]]) # likely to underestimate
            polar.r_e.setValue(r_e, [r_e * bounds_polar["r_e_factor"][0], r_e * bounds_polar["r_e_factor"][1]]) # likely to underestimate

    model.addFunction(polar)

    # Can perhaps run the DE solve on just the host (with isophote clipped) then on the polar component somehow (maybe by removing the center handful of pixels)
    model_desc[band] = str(model)

def main(args):
    if not(args.p == None):
        os.chdir(Path(args.p))
    manager = mp.Manager()

    if args.r:
        structure = os.walk(".")
        for root, dirs, files in structure:
            if not(files == []):
                img_files = sorted(glob.glob(os.path.join(Path(root), "image_?.fits")))

                jobs = []
                model_desc = manager.dict()
                for img_file in img_files:
                    band = img_file[-6] # Yes I know this is not the best way
                    # outputs[band] = None
                    if not(f"2_sersic_{band}.dat" in files) or args.overwrite:
                        print(f"Generating config for {img_file}")
                        img = fits.getdata(img_file)

                        if args.mask:
                            mask = fits.getdata(os.path.join(Path(root), "image_mask.fits"))
                            img = img * (1-mask)
                        
                        folder_type_dict = {
                            "Polar Rings": "ring",
                            "Polar_Tilted Bulges": "bulge",
                            "Polar_Tilted Halo": "halo"
                        }
                        for folder in ["Polar Rings", "Polar_Tilted Bulges", "Polar_Tilted Halo"]: # Attempt to autodetect type
                            if folder in root:
                                args.type = folder_type_dict[folder]
                        p = mp.Process(target = init_guess_2_sersic, args=(img, str(args.type).lower(), model_desc, band))
                        jobs.append(p)  
                
                if not(len(jobs) == 0):
                    for p in jobs:
                        p.start()

                    for p in jobs:
                        p.join()

                    for band in model_desc.keys():
                        if not(f"2_sersic_{band}.dat" in files) or args.overwrite:
                            with open(os.path.join(Path(root), f"2_sersic_{band}.dat"), "w") as f:
                                f.write(model_desc[band])

    else:
        img_files = sorted(glob.glob(os.path.join(Path("."), "image_?.fits")))

        jobs = []
        for img_file in img_files:
            band = img_file[-6] # Yes I know this is not the best way
            files = os.listdir(".")
            if not(f"2_sersic_{band}.dat" in files) or args.overwrite:
                print(f"Generating configs for {img_file}")
                img = fits.getdata(img_file)

                if args.mask:
                    mask = fits.getdata(os.path.join(Path("."), "image_mask.fits"))
                    img = img * (1-mask)

                p = mp.Process(target = init_guess_2_sersic, args=(img, str(args.type).lower(), model_desc, band))
                jobs.append(p)  

        if not(len(jobs) == 0):
            for p in jobs:
                p.start()

            for p in jobs:
                p.join()

            for band in model_desc.keys():
                if not(f"2_sersic_{band}.dat" in files) or args.overwrite:
                    with open(os.path.join(Path("."), f"2_sersic_{band}.dat"), "w") as f:
                        f.write(model_desc[band])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hello")
    parser.add_argument("-p", help="Path to file/folder containing galaxy FITS")
    parser.add_argument("--overwrite", help="Overwrite existing config files", action="store_true")
    parser.add_argument("--mask", help="Use the mask to guess initial values", action="store_true")
    parser.add_argument("--type", help="Type of polar structure", choices=["ring", "bulge", "halo"], default="ring")
    parser.add_argument("--dont_fit", help="Don't use DE imfitting to try and do another guess at initial parameters", action="store_true")
    parser.add_argument("-r", help="Recursively go into subfolders (assumes that fits data is at the end of the filetree)", action="store_true")
    

    args = parser.parse_args()
    main(args)
