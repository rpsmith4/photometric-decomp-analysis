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


def init_guess_2_sersic(fits, type):
    model = pyimfit.SimpleModelDescription()

    # Inner Sersic (Host)
    # Assuming galaxy is at the center
    shape = fits[0].data.shape
    img = fits[0].data
    model.x0.setValue(shape[0]/2 - 1)
    model.y0.setValue(shape[1]/2 - 1)

    host = pyimfit.make_imfit_function("Sersic", label="Host")
    host.PA.setValue(10, [0, 180]) # Hard to find probably
    host.ell.setValue(0.5, [0, 1]) # Probably doesn't really have to be changed


    I_e = (np.max(img) - np.min(img))/2
    host.I_e.setValue(I_e, [0, I_e*10]) # Get from (max - min)/2

    host.r_e.setValue(10, [0, 100]) # Maybe get an azimuthal average
    host.n.setValue(3, [0.5, 5]) # Maybe keep as is

    model.addFunction(host)


    # Outer Sersic (Polar)
    # Assuming galaxy is at the center
    shape = fits[0].data.shape
    img = fits[0].data
    model.x0.setValue(shape[0]/2 - 1)
    model.y0.setValue(shape[1]/2 - 1)

    polar = pyimfit.make_imfit_function("Sersic", label="Polar")
    polar.PA.setValue(10, [0, 180]) # Hard to find probably
    polar.ell.setValue(0.5, [0, 1]) # Probably doesn't really have to be changed


    I_e = (np.max(img) - np.min(img))/2
    polar.I_e.setValue(I_e, [0, I_e*10]) # Get from (max - min)/2

    polar.r_e.setValue(10, [0, 100]) # Maybe get an azimuthal average
    polar.n.setValue(3, [0.5, 5]) # Maybe keep as is

    model.addFunction(polar)

    return model

def main():
    fits_file = fits.open("test.fits")
    model_desc = init_guess_2_sersic(fits_file, None)
    print(model_desc)
    with open("config.dat", "w+") as f:
        f.write(str(model_desc))


if __name__ == "__main__":
    main()
