import numpy as np
from astropy.io import fits
import os
import sys


def file_size(fits_file):
    header = fits.getheader(fits_file)
    x_size = header["NAXIS1"]
    y_size = header["NAXIS2"]
    return x_size, y_size


def exact_copy(input_file, output_file):
    with fits.open(input_file) as hdul:
        source_hdu = hdul[0]

        new_hdul = fits.HDUList([source_hdu])
        new_hdul.writeto(output_file, overwrite=True)


def resize_image(input_file, reference_file, output_file):
    with fits.open(input_file) as hdul_s, fits.open(reference_file) as hdul_t:
        data = hdul_s[0].data
        reference_shape = hdul_t[0].data.shape  # (Y, X)
        input_shape = data.shape  # (Y, X)

        y_start = max(0, (input_shape[0] - reference_shape[0]) // 2)
        x_start = max(0, (input_shape[1] - reference_shape[1]) // 2)

        cropped_data = data[
                       y_start: y_start + reference_shape[0],
                       x_start: x_start + reference_shape[1]
                       ]

        new_shape = cropped_data.shape
        pad_y = reference_shape[0] - new_shape[0]
        pad_x = reference_shape[1] - new_shape[1]

        padding = (
            (pad_y // 2, pad_y - pad_y // 2),  # Top, Bottom
            (pad_x // 2, pad_x - pad_x // 2)  # Left, Right
        )

        final_data = np.pad(cropped_data, padding, mode='constant', constant_values=0)
        new_hdu = fits.PrimaryHDU(data=final_data, header=hdul_t[0].header)
        new_hdu.writeto(output_file, overwrite=True)


def main(incorrect_file, correct_file):
    incorrect_x, incorrect_y = file_size(incorrect_file)
    correct_x, correct_y = file_size(correct_file)
    if incorrect_x == correct_x and incorrect_y == correct_y:
        exact_copy(incorrect_file, "galaxy_mask_fitted.fits")
    else:
        resize_image(incorrect_file, correct_file, "galaxy_mask_fitted.fits")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
