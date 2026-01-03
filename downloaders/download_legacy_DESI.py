#! /usr/bin/env python

import subprocess
import math
from time import time, sleep
import argparse
import numpy as np
from io import BytesIO
from astropy.io import fits
from astropy.wcs import WCS
import requests
from prepare_images import make_patched_psf
from PIL import Image


def download(names, RA, DEC, R, file_types, bands='grz', pixscale=0.262, dr='dr9'):
    url = "http://legacysurvey.org/viewer/"

    RA = np.array(RA, float)
    DEC = np.array(DEC, float)
    R = np.array(R, float)

    for k in range(len(RA)):
        # R in arcmin
        RR = int(math.ceil(R[k]*60./pixscale))

        options = "?ra=%f&dec=%f&width=%i&height=%i&layer=ls-%s&pixscale=%.3f&bands=%s" % (RA[k], DEC[k], 2*RR, 2*RR, dr, pixscale,bands)

        file_types_dict = {
            "jpg": ["cutout.jpg", "jpg"],
            "fits": ["fits-cutout", "fits"],
            "psf": ["coadd-psf/", "fits"],
            "wm": ["fits-cutout", "fits"]
        } # file type : [url_option, file extension]


        output_file = names[k]
        print(f"{k}  {output_file}")

        if "fits" in file_types:
            url_new = url + file_types_dict["fits"][0]
            params = {
                "ra": RA[k],
                "dec": DEC[k],
                "size": 2*RR,
                "layer": "ls-" + dr,
                "pixscale": 0.262,
                "invvar": ("wm" in file_types)
            }
            try:
                hdu = fits.open(get_data(url_new, params))

                if not(hdu == None):
                    bands = hdu[0].header["BANDS"].strip()
                    wcs = WCS(hdu[0].header)
                    for idx, band in enumerate(bands):
                        data = hdu[0].data[idx, ...]
                        if np.sum(data) == 0:
                            continue
                        fits.PrimaryHDU(data=data, header=wcs.to_header()).writeto(f"image_{band}.fits", overwrite=True)
                    for idx, band in enumerate(bands):
                        data = hdu[1].data[idx, ...]
                        if np.sum(data) == 0:
                            continue
                        fits.PrimaryHDU(data=data, header=wcs.to_header()).writeto(f"image_{band}_invvar.fits", overwrite=True)
            except Exception as e:
                print(e)
                pass

        if "psf" in file_types:
            url_new = url + file_types_dict["psf"][0]
            params = {
                "ra": RA[k],
                "dec": DEC[k],
                "layer": "ls-" + dr,
                "pixscale": 0.262
            }
            try:
                psf_hdu = fits.open(get_data(url_new, params))

                if not(psf_hdu == None):
                    for hdu in psf_hdu:
                        band = hdu.header['BAND']
                        data = hdu.data
                        fits.PrimaryHDU(data=data).writeto(f"psf_core_{band}.fits", overwrite=True)
                        psf_size = 65
                        # if band == "z":
                        #     psf_size = 600
                        psf_combined = make_patched_psf(f"psf_core_{band}.fits", band, psf_size)
                        fits.PrimaryHDU(data=psf_combined).writeto(f"psf_patched_{band}.fits", overwrite=True)
            except Exception as e:
                print(e)
                pass
        
        if "jpg" in file_types:
            url_new = url + file_types_dict["jpg"][0]
            params = {
                "ra": RA[k],
                "dec": DEC[k],
                "layer": "ls-" + dr,  
                "size": 2*RR,
                "pixscale": 0.262
            }
            try:
                stream = get_data(url_new, params)
                img = Image.open(stream)
                # img.save(f"{names[k]}.jpg")
                img.save(f"image.jpg")
            except:
                pass



def get_data(url, params):
    attempts = 3
    for attempt in range(attempts):
        try:
            # Make the request to the Legacy Survey API
            print("Requesting data")
            print(url)
            print(params)
            response = requests.get(url, params=params)
            print(f"Response status: {response.reason}")

            if response.status_code == 500:
                print(f"Server error (500): {response.reason}.")
                return None
            response.raise_for_status()
            print("Download OK")
            # Open the response as a FITS file and save it to disk
            return BytesIO(response.content)
        except requests.exceptions.RequestException as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            if attempt == attempts - 1:
                print("Max retries reached. Unable to download the file.")
        except Exception as e:
            print(f"Error handling file: {e}")
            return None
    return None

def main(names, RA, DEC, R, bands='grz', pixscale=0.262, dr='dr9', file_types=["jpg"]):
    download(names, RA, DEC, R, file_types, bands, pixscale, dr)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Download from DESI Legacy Survey",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
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