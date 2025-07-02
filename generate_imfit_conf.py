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
import pyimfit

import argparse


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

def init_guess_2_sersic(fits_dat, pol_str_type):
    model = pyimfit.SimpleModelDescription()

    # Inner Sersic (Host)
    # Assuming galaxy is at the center
    shape = fits_dat[0].data.shape
    img = fits_dat[0].data

    if args.mask:
        mask = fits.getdata("image_mask.fits")
        img = img * (1-mask)


    model.x0.setValue(shape[0]/2 - 1)
    model.y0.setValue(shape[1]/2 - 1)

    host = pyimfit.make_imfit_function("Sersic", label="Host")

    img_reduce = img.copy()
    print(np.max(img))
    I_e = (np.max(img) - np.min(img))/2
    img_reduce[img_reduce < I_e] = 0
    
    PA = get_PA(img_reduce)
    PA = PA - 90 # CCW from +x axis to CCW from +y axis 

    host.PA.setValue(PA, [PA-30, PA+30]) # Hard to find probably, might get from the IMAN script ellipse fitting thing (might mask low intensities)
    host.ell.setValue(0.5, [0, 1]) # Probably doesn't really have to be changed


    host.I_e.setValue(I_e, [I_e/2, I_e*10]) # Get from (max - min)/2

    host.r_e.setValue(10, [0, 100]) # Maybe get an azimuthal average
    host.n.setValue(3, [0, 5]) # Maybe keep as is

    model.addFunction(host)


    # Outer Sersic (Polar)
    # Assuming galaxy is at the center
    model.x0.setValue(shape[0]/2 - 1, [shape[0]/2 - 30, shape[0]/2 + 30])
    model.y0.setValue(shape[1]/2 - 1, [shape[1]/2 - 30, shape[1]/2 + 30])

    polar = pyimfit.make_imfit_function("Sersic", label="Polar")
    polar.PA.setValue(PA + 90, [PA + 90 - 30, PA + 90 + 30]) # Can probably just be the host PA +/- 90
    polar.ell.setValue(0.5, [0, 1]) # Probably doesn't really have to be changed


    I_e = (np.max(img) - np.min(img))/2
    polar.I_e.setValue(I_e/100, [0, I_e*10/100]) # Get from (max - min)/2

    polar.r_e.setValue(50, [0, 100]) # Maybe get an azimuthal average
    polar.n.setValue(3, [0, 10]) # Maybe keep as is

    model.addFunction(polar)

    # Can perhaps run the DE solve on just the host (with isophote clipped) then on the polar component somehow (maybe by removing the center handful of pixels)

    return model

def main(args):
    fits_file = fits.open("image_g.fits")
    model_desc = init_guess_2_sersic(fits_file, None)

    print(model_desc)

    with open("config.dat", "w") as f:
        f.write(str(model_desc))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hello")
    parser.add_argument("-p", help="Path to file/folder containing galaxy FITS")
    parser.add_argument("--overwrite", help="Overwrite existing config files", action="store_true")
    parser.add_argument("--mask", help="Use the mask to guess initial values", action="store_true")

    args = parser.parse_args()
    main(args)
