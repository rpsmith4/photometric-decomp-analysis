#! /usr/bin/env python

import subprocess
import math
from time import time, sleep
import urllib.request
import sys
import os
import glob
import shutil
import argparse
import numpy as np


def download(names, RA, DEC, R, file_types, bands='grz', pixscale=0.262, dr='dr9'):
    url_base = "http://legacysurvey.org/viewer/"

    for k in range(len(RA)):
        # R in arcmin
        RA = np.array(RA, float)
        DEC = np.array(DEC, float)
        R = np.array(R, float)
        RR = int(math.ceil(R[k]*60./pixscale))

        options = "?ra=%f&dec=%f&width=%i&height=%i&layer=ls-%s&pixscale=%.3f&bands=%s" % (RA[k], DEC[k], 2*RR, 2*RR, dr, pixscale,bands)

        file_types_dict = {
            "jpg": ["cutout.jpg", "jpg"],
            "fits": ["fits-cutout", "fits"],
            "psf": ["coadd-psf/", "fits"],
            "wm": ["fits-cutout", "fits"]
        } # file type : [url_option, file extension]

        for file_type in file_types:
            extra_option = ""
            if file_type == "wm":
                extra_option = "&subimage"
            
            url_full = url_base + file_types_dict[file_type][0] + options + extra_option
            
            output_file = names[k] + "." + file_types_dict[file_type][1]
            print(f"{k}  {output_file}")
            urllib.request.urlretrieve(url_full, output_file)



def main(names, RA, DEC, R, bands='grz', pixscale=0.262, dr='dr9', file_types=["jpg"]):
    print('Downloading...')

    download(names, RA, DEC, R, file_types, bands, pixscale, dr)

    print('Done!')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Download from Legacysurvey")
    parser.add_argument("ra", help="Right ascension [deg]", type=float)
    parser.add_argument("dec", help="Declination [deg]", type=float)
    parser.add_argument("width", help="Width of the image [arcmin]", type=float)
    parser.add_argument("--name", help="Optional: Name of the object", type=str, default=None)
    parser.add_argument("--bands", help="Optional: Bands to be downloaded, e.g. grz", type=str, default='grz')


    args = parser.parse_args()

    ra = args.ra
    dec = args.dec
    width = args.width

    name = args.name
    bands = args.bands

    R = width/2.

    main([name], [ra], [dec], [R], args, bands, pixscale=0.262)