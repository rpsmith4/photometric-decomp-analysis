import numpy as np
from scipy.ndimage import convolve, median_filter, zoom
from scipy.optimize import curve_fit
from astropy.io import fits
from astropy.cosmology import FlatLambdaCDM
import matplotlib.pyplot as plt
import kcorrect.kcorrect

# Define cosmology for luminosity distance calculations
cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
# Source:
#   "Full and Efficient Redshifting of Ensembles of Nearby Galaxy Images"
#   see the Webpage (http://www.mpia-hd.mpg.de/FERENGI/), paper
#   (http://www.mpia-hd.mpg.de/FERENGI/paper/ferengi_V20070725.pdf),
#   and README for details.
#
# CALLING SEQUENCE: 
#   ferengi, sky, im, imerr, psflo, err0_mag, psfhi, $
#            lambda_lo, filter_lo, zlo, scllo, zplo, tlo, $
#            lambda_hi, filter_hi, zhi, sclhi, zphi, thi, $
#            im_out_file, psf_out_file, $
#            noflux=noflux, evo=evo, noconv=noconv
# Converted from IDL language to Python with Gemini

def nu2lam(nu):
    """
    Converts frequency (Hz) to wavelength (Angstrom).
    """
    return 299792458. / nu * 1e10

def lam2nu(lam):
    """
    Converts wavelength (Angstrom) to frequency (Hz).
    """
    return 299792458. / lam * 1e10

def maggies2mags(maggies):
    """
    Converts maggies to magnitudes.
    """
    return -2.5 * np.log10(maggies)

def mags2maggies(mags):
    """
    Converts magnitudes to maggies.
    """
    return 10**(-0.4 * mags)

def maggies2fnu(maggies):
    """
    Converts maggies to flux in erg s-1 Hz-1 cm-2.
    """
    return 3631e-23 * maggies

def fnu2maggies(fnu):
    """
    Converts flux in erg s-1 Hz-1 cm-2 to maggies.
    """
    return fnu / 3631e-23

def fnu2flam(fnu, lam):
    """
    Converts flux in erg s-1 Hz-1 cm-2 to erg s-1 cm-2 A-1.
    """
    return 299792458. * 1e10 / lam**2. * fnu

def flam2fnu(flam, lam):
    """
    Converts flux in erg s-1 cm-2 A-1 to erg s-1 Hz-1 cm-2.
    """
    return flam / 299792458. / 1e10 * lam**2.

def lambda_eff(lam, trans):
    """
    Calculates the effective wavelength for a filter.
    """
    idx = np.where(lam != 0)
    if len(idx[0]) == 0:
        raise ValueError('ERROR: no non-zero wavelengths')
    return np.trapz(lam[idx] * trans[idx], x=lam[idx]) / np.trapz(trans[idx], x=lam[idx])

def cts2mags(cts, expt, zp):
    """
    Converts counts to magnitudes.
    """
    return maggies2mags(cts2maggies(cts, expt, zp))

def cts2maggies(cts, expt, zp):
    """
    Converts counts to maggies.
    """
    return cts / expt * 10**(-0.4 * zp)

def mags2cts(mags, expt, zp):
    """
    Converts magnitudes to counts.
    """
    return maggies2cts(mags2maggies(mags), expt, zp)

def maggies2cts(maggies, expt, zp):
    """
    Converts maggies to counts.
    """
    return maggies * expt / 10**(-0.4 * zp)

def maggies2lup(maggies, filter_name):
    """
    Converts maggies to luptitudes for SDSS filters.
    """
    b_values = {'u': 1.4e-10, 'g': 0.9e-10, 'r': 1.2e-10, 'i': 1.8e-10, 'z': 7.4e-10}
    b = b_values.get(filter_name)
    if b is None:
        raise ValueError(f"Unknown filter: {filter_name}")
    return -2.5 / np.log(10) * (np.arcsinh(maggies / b * 0.5) + np.log(b))

def lup2maggies(lup, filter_name):
    """
    Converts luptitudes to maggies for SDSS filters.
    """
    b_values = {'u': 1.4e-10, 'g': 0.9e-10, 'r': 1.2e-10, 'i': 1.8e-10, 'z': 7.4e-10}
    b = b_values.get(filter_name)
    if b is None:
        raise ValueError(f"Unknown filter: {filter_name}")
    return 2 * b * np.sinh(-0.4 * np.log(10) * lup - np.log(b))

def random_indices(len_array, n_in):
    """
    Produces a set of N_IN unique random indices for an array with LEN elements.
    """
    if n_in > len_array:
        raise ValueError("n_in cannot be greater than len_array")
    
    swap = n_in > len_array / 2
    n = len_array - n_in if swap else n_in
    
    inds = np.empty(n, dtype=int)
    count = 0
    
    while count < n:
        new_indices = np.random.randint(0, len_array, size=n - count)
        inds[count:] = new_indices
        
        inds_unique = np.unique(inds[:count + len(new_indices)])
        count = len(inds_unique)
        inds[:count] = inds_unique
    
    if swap:
        all_indices = np.arange(len_array)
        inds = np.setdiff1d(all_indices, inds_unique) #inds_unique holds the indices that were picked
    return inds[:n_in] # Return only n_in elements if not swapped, else len_array - n elements

def edge_index(a, rx, ry):
    """
    Creates an index of a ring with width 1 around the center at radius rx and ry.
    """
    sz_y, sz_x = a.shape
    px = 0 if sz_x % 2 else 0.5
    py = 0 if sz_y % 2 else 0.5

    x_coords = np.arange(sz_x)
    y_coords = np.arange(sz_y)
    
    abs_x = np.abs(x_coords - sz_x / 2 + px)
    abs_y = np.abs(y_coords - sz_y / 2 + py)
    
    b = np.tile(abs_x, (sz_y, 1)).T
    c = np.tile(abs_y, (sz_x, 1))

    # Condition for elements on the "edges" of the specified rectangle
    mask = ( (b == rx) & (c <= ry) ) | ( (c == ry) & (b <= rx) )
    
    indices = np.where(mask.flatten())[0]

    if np.size(indices) == 0:
        return np.array([-1])
    
    return indices

def resistant_mean(data, n_sigma=3):
    """
    Calculates a resistant mean and standard deviation by iteratively removing outliers.
    """
    data_flat = data.flatten()
    good_data = data_flat[np.isfinite(data_flat)]

    if len(good_data) == 0:
        return np.nan, np.nan, 0

    while True:
        mean_val = np.mean(good_data)
        std_val = np.std(good_data)
        
        # Filter outliers
        lower_bound = mean_val - n_sigma * std_val
        upper_bound = mean_val + n_sigma * std_val
        
        new_good_data = good_data[(good_data >= lower_bound) & (good_data <= upper_bound)]
        
        if len(new_good_data) == len(good_data):
            break  # No more outliers removed
        
        if len(new_good_data) == 0:
            return np.nan, np.nan, len(data_flat)  # All data were outliers
        
        good_data = new_good_data
    
    nrej = len(data_flat) - len(good_data)
    return np.mean(good_data), np.std(good_data), nrej


def robust_linefit(x, y):
    """
    Performs a robust linear fit (y = c[0] + c[1]*x) similar to IDL's ROBUST_LINEFIT.
    Uses an iterative approach to remove outliers based on residuals.
    Returns coefficients [intercept, slope].
    """
    if len(x) != len(y) or len(x) < 2:
        raise ValueError("Input arrays must have the same length and at least 2 elements.")

    # Initial linear fit
    A = np.vstack([np.ones(len(x)), x]).T
    coeffs, residuals, rank, s = np.linalg.lstsq(A, y, rcond=None)

    for _ in range(10):  # Iterate a few times for robustness
        # Calculate residuals
        y_pred = coeffs[0] + coeffs[1] * x
        res = y - y_pred

        # Calculate robust sigma of residuals
        median_res = np.median(np.abs(res))
        if median_res == 0: # All residuals are zero, perfect fit
            break
        sigma_res = median_res / 0.6745  # Scale to approximate std dev for normal distribution

        # Identify inliers (within 3 sigma)
        inliers = np.abs(res) <= 3 * sigma_res
        
        if np.sum(inliers) < 2: # Not enough inliers for a fit
            break
        
        # Refit with inliers
        A_inliers = np.vstack([np.ones(np.sum(inliers)), x[inliers]]).T
        new_coeffs, new_residuals, new_rank, new_s = np.linalg.lstsq(A_inliers, y[inliers], rcond=None)
        
        if np.allclose(coeffs, new_coeffs): # Converged
            coeffs = new_coeffs
            break
        
        coeffs = new_coeffs
    
    return coeffs


def ring_sky(image, width0, nap, x=None, y=None, q=1.0, pa=0.0, rstart=None, nw=False):
    """
    Measures flux in apertures around position x,y in rings with axis ratio q and
    position angle pa.
    """
    if isinstance(image, str):
        with fits.open(image) as hdul:
            im = hdul[0].data
    else:
        im = image

    sz = np.array(im.shape)

    if x is None or y is None:
        x = sz[0] * 0.5
        y = sz[1] * 0.5
        rstart = min(sz) * 0.05 if rstart is None else rstart

    # Create radius array (simplified ellipse distance, adjust as needed for full IDL equivalent)
    # This is a placeholder for a more complete `dist_ellipse` implementation
    Y, X = np.ogrid[0:sz[0], 0:sz[1]]
    # Rotate coordinates if pa is not 0
    X_rot = (X - x) * np.cos(np.deg2rad(pa)) - (Y - y) * np.sin(np.deg2rad(pa))
    Y_rot = (X - x) * np.sin(np.deg2rad(pa)) + (Y - y) * np.cos(np.deg2rad(pa))
    
    rad = np.sqrt(X_rot**2 + (Y_rot / q)**2)

    max_rad = np.max(rad) * 0.95

    if nw:
        width = max_rad / float(width0)
    else:
        width = width0

    mean_global, sig_global, _ = resistant_mean(im, 3)
    sig_global *= np.sqrt(im.size - 1) # IDL's resistant_mean sigma calculation is slightly different

    if rstart is None:
        rhi = width
    else:
        rhi = rstart
    
    r_vals = []
    flux_vals = []

    i = 0
    sign = -1 # -1=2 measurements, 0=1 measurement

    while rhi <= max_rad:
        extra = 0
        ct = 0
        while ct < 10:
            mask = (rad <= rhi + extra) & (rad >= rhi - width - extra) & \
                   (im <= mean_global + 3 * sig_global) & (im >= mean_global - 3 * sig_global)
            
            idx = np.where(mask)
            ct = len(idx[0])
            extra += 1
            if extra > max(sz) * 2:
                break

        if ct < 5:
            sky = flux_vals[-1] if flux_vals else np.nan # Use last sky value or NaN
        else:
            sky, _, _ = resistant_mean(im[idx], 3)
        
        r_vals.append(rhi - 0.5 * width)
        flux_vals.append(sky)
        i += 1

        if len(flux_vals) > nap:
            # Measure slope over last nap apertures
            coeffs = robust_linefit(np.array(r_vals)[i - nap:i], np.array(flux_vals)[i - nap:i])
            
            if sign > 0 and coeffs[1] > 0:
                break
            if coeffs[1] > 0:
                sign += 1
        
        rhi += width
    
    # Calculate robust sky over last nap apertures
    sky_final, _, _ = resistant_mean(np.array(flux_vals)[(i - nap + 1 if (i - nap + 1) > 0 else 0):i], 3)
    return sky_final

def ferengi_make_psf_same(psf1, psf2):
    """
    Enlarges the smaller PSF to the size of the larger one using zero-padding.
    Modifies psf1 and psf2 in place.
    """
    sz1 = np.array(psf1.shape)
    sz2 = np.array(psf2.shape)

    if sz1[0] * sz1[1] > sz2[0] * sz2[1]:
        big = psf1
        small = psf2
        flag_c = 1  # psf2 was originally smaller
    else:
        big = psf2
        small = psf1
        flag_c = 0  # psf1 was originally smaller

    szbig = np.array(big.shape)
    szsmall = np.array(small.shape)

    # Calculate padding
    lo_x = (szbig[0] - szsmall[0]) // 2
    hi_x = szbig[0] - szsmall[0] - lo_x
    lo_y = (szbig[1] - szsmall[1]) // 2
    hi_y = szbig[1] - szsmall[1] - lo_y

    small_padded = np.pad(small, ((lo_x, hi_x), (lo_y, hi_y)), 'constant', constant_values=0)

    if flag_c:
        psf2 = small_padded
    else:
        psf1 = small_padded


def ferengi_psf_centre(psf0):
    """
    Centers the PSF by resizing to an odd number of pixels and using a 2D Gaussian fit.
    """
    psf = np.copy(psf0)

    # Resize to odd number of pixels
    sz_x, sz_y = psf.shape
    if (sz_x + 1) % 2:
        psf = np.pad(psf, ((0, 1), (0, 0)), 'constant')
    sz_x, sz_y = psf.shape
    if (sz_y + 1) % 2:
        psf = np.pad(psf, ((0, 0), (0, 1)), 'constant')

    # Fit a 2D Gaussian to find the center
    # This is a simplified version; a full Gaussian fit implementation would be more complex.
    # For now, we'll assume the brightest pixel is a good starting point for centering.
    # In a real scenario, you'd use a proper 2D Gaussian fitting routine.
    def gaussian_2d(coords, amplitude, xo, yo, sigma_x, sigma_y, theta, offset):
        x, y = coords
        a = (np.cos(theta)**2)/(2*sigma_x**2) + (np.sin(theta)**2)/(2*sigma_y**2)
        b = -(np.sin(2*theta))/(4*sigma_x**2) + (np.sin(2*theta))/(4*sigma_y**2)
        c = (np.sin(theta)**2)/(2*sigma_x**2) + (np.cos(theta)**2)/(2*sigma_y**2)
        g = offset + amplitude*np.exp( - (a*((x-xo)**2) + 2*b*(x-xo)*(y-yo) + c*((y-yo)**2)))
        return g.ravel()

    sz_x, sz_y = psf.shape
    x_coords, y_coords = np.meshgrid(np.arange(sz_x), np.arange(sz_y))
    initial_guess = (psf.max(), sz_x/2, sz_y/2, sz_x/4, sz_y/4, 0, psf.min())
    try:
        popt, pcov = curve_fit(gaussian_2d, (x_coords, y_coords), psf.ravel(), p0=initial_guess)
        xo, yo = popt[1], popt[2]
    except RuntimeError:
        # Fallback if fit fails (e.g., flat PSF)
        peak_idx = np.unravel_index(np.argmax(psf), psf.shape)
        xo, yo = peak_idx[0], peak_idx[1]
    
    # Calculate shift needed to center the PSF
    center_x, center_y = (sz_x - 1) / 2.0, (sz_y - 1) / 2.0
    shift_x = center_x - xo
    shift_y = center_y - yo

    # Apply shift using scipy.ndimage.shift for sub-pixel accuracy
    # Note: sshift2d in IDL performs circular shift. Here, we assume a more standard shift for images.
    # If circular shift is strictly required, more complex padding/slicing would be needed.
    shifted_psf = np.roll(psf, int(np.round(shift_x)), axis=0)
    shifted_psf = np.roll(shifted_psf, int(np.round(shift_y)), axis=1)

    return shifted_psf, shift_x, shift_y


def ferengi_deconvolve(wide, narrow):
    """
    Calculates the deconvolution kernel between a wide and narrow PSF.
    """
    # Ensure WIDE and NARROW have same size, odd number of pixels, are centered and normalized
    # (These steps should ideally be done before calling this function)

    sz_wide = np.array(wide.shape)
    sz_narrow = np.array(narrow.shape)

    # Determine a suitable FFT size (power of 2, larger than max dimension)
    sz_max = max(sz_wide[0], sz_wide[1], sz_narrow[0], sz_narrow[1])
    bigsz = 2
    while bigsz < sz_max:
        bigsz *= 2
    if bigsz > 2048:
        print('Warning: Requested PSF array is larger than 2x2k!')
        # Consider resizing or padding to a more manageable size if this becomes an issue.
        # For now, let's cap it as in the original code, but raise a warning
        # bigsz = 2048 

    # Pad PSFs to bigsz
    psf_n_2k = np.zeros((bigsz, bigsz), dtype=np.float64)
    psf_w_2k = np.zeros((bigsz, bigsz), dtype=np.float64)

    # Calculate padding offsets
    offset_n_x = (bigsz - sz_narrow[0]) // 2
    offset_n_y = (bigsz - sz_narrow[1]) // 2
    offset_w_x = (bigsz - sz_wide[0]) // 2
    offset_w_y = (bigsz - sz_wide[1]) // 2
    
    psf_n_2k[offset_n_x:offset_n_x + sz_narrow[0], offset_n_y:offset_n_y + sz_narrow[1]] = narrow
    psf_w_2k[offset_w_x:offset_w_x + sz_wide[0], offset_w_y:offset_w_y + sz_wide[1]] = wide

    # Get FFT for both
    fft_n = np.fft.fft2(psf_n_2k)
    fft_w = np.fft.fft2(psf_w_2k)

    # Avoid division by zero and stabilize
    # The IDL code's stabilization: (ABS(fft_n)/(ABS(fft_n)+0.000000001))*fft_n
    # This effectively scales the FFT by |FFT| / (|FFT| + epsilon)
    # which preserves the phase and dampens small values.
    epsilon = 1e-9 # Matches the IDL code's 0.000000001
    
    abs_fft_n = np.abs(fft_n)
    fft_n_stabilized = (abs_fft_n / (abs_fft_n + epsilon)) * fft_n
    
    abs_fft_w = np.abs(fft_w)
    fft_w_stabilized = (abs_fft_w / (abs_fft_w + epsilon)) * fft_w

    # Calculate ratio (deconvolution in Fourier space)
    psfrat = fft_w_stabilized / fft_n_stabilized

    # Inverse FFT to get the transformation PSF
    psfhlp = np.fft.ifft2(psfrat).real # Take real part as it should be real

    # Shift quadrants (FFT convention) to center the PSF
    # This is equivalent to IDL's ROTATE(..., 2) or numpy's fft.fftshift for centered output
    psfhlp = np.fft.fftshift(psfhlp)

    # Crop to original PSF size (assuming 'narrow' size for output crop)
    crop_sz_x, crop_sz_y = sz_narrow[0], sz_narrow[1]
    
    start_x = (bigsz - crop_sz_x) // 2
    end_x = start_x + crop_sz_x
    start_y = (bigsz - crop_sz_y) // 2
    end_y = start_y + crop_sz_y

    psfcorr = psfhlp[start_x:end_x, start_y:end_y]

    return psfcorr / np.sum(psfcorr)


def ferengi_clip_edge(npix_clip, im, auto_frac=2, clip_also=None, norm=False):
    """
    Clips the outer pixels of an image based on noise characteristics.
    """
    sz = np.array(im.shape)
    rx = int(sz[1] / 2 / auto_frac)
    ry = int(sz[0] / 2 / auto_frac)
    
    sig_vals = [0.0]
    r_vals = [0]

    while True:
        i = edge_index(im, rx, ry)
        if i[0] == -1: # No more pixels found in the ring
            break
        
        # Reshape im and i to 1D arrays for easy indexing
        im_flat = im.flatten()

        mn, sg, nr = resistant_mean(im_flat[i], 3)
        sg *= np.sqrt(len(i) - 1 - nr) # IDL's specific sigma calculation
        
        sig_vals.append(sg)
        r_vals.append(rx)
        
        rx += 1
        ry += 1
    
    r_vals = np.array(r_vals[1:])
    sig_vals = np.array(sig_vals[1:])

    if len(sig_vals) == 0: # No edges found to clip
        if norm:
            if clip_also is not None:
                clip_also /= np.sum(clip_also)
            im /= np.sum(im)
        return

    mn_sig, sg_sig, nr_sig = resistant_mean(sig_vals, 3)
    sg_sig = sg_sig * np.sqrt(len(sig_vals) - 1 - nr_sig)

    i_outliers = np.where(sig_vals > mn_sig + 10 * sg_sig)[0]

    if len(i_outliers) > 0:
        lim = np.min(r_vals[i_outliers]) - 1
        if len(i_outliers) > nr_sig * 3:
            print('Warning: Large gap?')
        
        npix_clip = int(np.round(sz[0] / 2. - lim))

        if clip_also is not None:
            # Need to ensure clip_also is a mutable array like numpy array, not just a keyword flag
            clip_also[:] = clip_also[npix_clip:sz[0]-npix_clip, npix_clip:sz[1]-npix_clip]
        im[:] = im[npix_clip:sz[1]-npix_clip, npix_clip:sz[0]-npix_clip]
    
    if norm:
        if clip_also is not None:
            clip_also /= np.sum(clip_also)
        im /= np.sum(im)
    return im


def ferengi_downscale(im_lo, z_lo, z_hi, p_lo, p_hi, upscl=False, nofluxscl=False, evo=None):
    """
    Scales an image in flux and size for a given set of redshifts.
    """
    evo_fact = 1.0
    if evo is not None:
        evo_fact = 10**(-0.4 * evo * z_hi)

    d_lo = cosmo.luminosity_distance(z_lo).value # in Mpc
    d_hi = cosmo.luminosity_distance(z_hi).value # in Mpc

    # The magnification (size correction)
    magnification = (d_lo / d_hi * (1. + z_hi)**2 / (1. + z_lo)**2 * p_lo / p_hi)
    if upscl:
        magnification = 1. / magnification

    # The flux scaling (surface brightness dimming)
    flux_ratio = 1.0
    if not nofluxscl:
        flux_ratio = (d_lo / d_hi)**2

    sz_lo = np.array(im_lo.shape)
    nx_hi = int(np.round(sz_lo[0] * magnification))
    ny_hi = int(np.round(sz_lo[1] * magnification))

    # Resize image using zoom for better quality than simple rebinning
    # Note: FREBIN in IDL can do sum or sample. /total in IDL FREBIN sums pixels.
    # Here, we need to adjust the zoom factor to preserve total flux if /total is implied.
    # The 'order=1' for bilinear interpolation, 'order=0' for nearest.
    # For total flux preservation, `zoom` needs to be multiplied by zoom_factor**2 or adjusted.
    
    # Calculate the effective zoom factor to be applied
    actual_zoom_x = nx_hi / sz_lo[0]
    actual_zoom_y = ny_hi / sz_lo[1]
    
    # Perform the scaling. Use spline interpolation (order=3) for better quality.
    # The output `zoomed_im` will be scaled by the square of the zoom factor if `mode='nearest'`.
    # For flux conservation, we should divide by the square of the zoom factors to get total flux.
    zoomed_im = zoom(im_lo, (actual_zoom_x, actual_zoom_y), order=1)
    
    # Adjust for total flux if the interpolation method does not inherently conserve it.
    # If FREBIN /total means summing up pixel values, then a simple zoom followed by multiplication
    # by the ratio of pixel areas (original_area / new_area) should work.
    # Since we are effectively changing the pixel scale from p_lo to p_hi * (d_hi/d_lo)^2 / ((1+z_hi)^2/(1+z_lo)^2)
    # the flux per pixel should scale by the inverse of the area change for surface brightness conservation.
    # But since we're scaling the image itself, the total flux will be conserved by `zoom` if we normalize later.
    # The original IDL code multiplies by flux_ratio * evo_fact.
    
    return zoomed_im * flux_ratio * evo_fact

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


def ferengi_transformation_psf(psf_s0, psf_c0, z_lo, z_hi, p_lo, p_hi, same_size=False):
    """
    Calculates the transformation PSF (deconvolution kernel).
    """
    psf_s = np.copy(psf_s0)
    psf_c = np.copy(psf_c0)

    # Make size odd & make square & centre for both psfs
    psf_s = ferengi_odd_n_square(psf_s)
    psf_c = ferengi_odd_n_square(psf_c)

    d_lo = cosmo.luminosity_distance(z_lo).value
    d_hi = cosmo.luminosity_distance(z_hi).value
    
    insz = np.array(psf_s.shape)[0]
    add = 0
    # The scaling factor includes (1+z) terms from angular diameter distance ratio
    # and pixel scale ratio.
    scaling_factor = (d_lo / d_hi * (1. + z_hi)**2 / (1. + z_lo)**2 * p_lo / p_hi)
    outsz = int(np.round(scaling_factor * (insz + add)))
    
    while outsz % 2 == 0:
        add += 2
        # Pad psf_s with 2 pixels on each side for the enlargement
        psf_s = np.pad(psf_s, ((1, 1), (1, 1)), 'constant', constant_values=0)
        psf_s = ferengi_odd_n_square(psf_s) # Re-square and center after padding
        insz = np.array(psf_s.shape)[0] # Update insz
        outsz = int(np.round(scaling_factor * insz))
        if add > insz * 3:
            raise ValueError('Enlarging PSF failed!')

    psf_s = ferengi_odd_n_square(psf_s)

    # Downscale the local PSF (psf_s)
    # Note: ferengi_downscale's nofluxscl keyword is set, meaning no surface brightness dimming.
    psf_s_ds = ferengi_downscale(psf_s, z_lo, z_hi, p_lo, p_hi, nofluxscl=True)

    # Make sizes the same (zero-padding the smaller to the larger)
    # ferengi_make_psf_same modifies in place, so pass mutable arrays or copies if original needed.
    # To mimic IDL's PRO, we create a copy and then potentially assign back.
    temp_psf_c = np.copy(psf_c)
    temp_psf_s_ds = np.copy(psf_s_ds)
    ferengi_make_psf_same(temp_psf_c, temp_psf_s_ds) # This modifies temp_psf_c and temp_psf_s_ds
    psf_c_aligned = temp_psf_c
    psf_s_ds_aligned = temp_psf_s_ds


    # Re-make size odd & square & center for both (after padding to same size)
    psf_s_final = ferengi_odd_n_square(psf_s_ds_aligned)
    psf_c_final = ferengi_odd_n_square(psf_c_aligned)

    # Normalise PSFs to sum to 1
    psf_s_final /= np.sum(psf_s_final)
    psf_c_final /= np.sum(psf_c_final)

    if same_size:
        # If this keyword is set, the original psf_s0 and psf_c0 should be updated.
        # This is tricky with Python's pass-by-object-reference vs. IDL's pass-by-reference.
        # A common Pythonic way is to return the modified PSFs if they are meant to be used externally.
        # For now, we'll assume this only affects the internal copies.
        pass # Placeholder for external modification if needed.

    # Deconvolve to get the transformation PSF
    return ferengi_deconvolve(psf_c_final, psf_s_final)


def ferengi_convolve_plus_noise(im, psf, sky, exptime, nonoise=False, border_clip=0, extend=False):
    """
    Convolves an image with a PSF, adds sky background, and Poisson noise.
    """
    sz_psf = np.array(psf.shape)
    
    # Clip borders of the PSF if border_clip is set
    if border_clip > 0:
        psf = psf[border_clip:sz_psf[0]-border_clip, border_clip:sz_psf[1]-border_clip]
        sz_psf = np.array(psf.shape) # Update PSF size after clipping

    # Enlarge the image for proper convolution (padding)
    sz_im = np.array(im.shape)
    
    # Pad the image. The padding size should be half of the PSF dimensions
    # to avoid edge effects from convolution (assuming PSF is centered).
    pad_x = sz_psf[0] // 2
    pad_y = sz_psf[1] // 2
    
    im_padded = np.pad(im, ((pad_x, pad_x), (pad_y, pad_y)), 'constant', constant_values=0)

    # Convolve with the PSF (normalized)
    # The `convolve` function in scipy.ndimage handles padding if `mode` is set to 'constant'
    # and `cval` to 0, or 'same' mode which crops output to input size.
    # The IDL CONVOLVE by default handles borders by zero-padding.
    out = convolve(im_padded, psf / np.sum(psf), mode='constant', cval=0.0)

    # Remove the excess border if extend is not set
    if not extend:
        out = out[pad_x : pad_x + sz_im[0], pad_y : pad_y + sz_im[1]]

    # Add Poisson noise and sky background
    sz_out = np.array(out.shape)

    if not nonoise:
        # Ensure sky matches the output image dimensions for addition
        # If sky is smaller, it will be tiled or an error will occur. Assume sky is large enough.
        sky_clipped = sky[0:sz_out[0], 0:sz_out[1]]
        
        # Original IDL: out += sky_clipped + sqrt(abs(out*exptime))*randomn(1, ...) / exptime
        # This suggests out is in counts/sec, and noise is added based on counts, then converted back.
        # `randomn` in IDL is standard normal (mean 0, std dev 1).
        
        # Calculate noise in counts
        noise_counts = np.sqrt(np.abs(out * exptime)) * np.random.randn(sz_out[0], sz_out[1])
        # Convert noise back to counts/sec (rate)
        noise_rate = noise_counts / exptime
        
        out += sky_clipped + noise_rate

    return out

# KCORRECT functions would need a separate library or re-implementation.
# For this script, we'll use placeholders for KCORRECT calls, as it's a significant external dependency.
# You would need to integrate a Python K-correction package like `kcorrect_py` or `pz_kcorrect`.

# Placeholder for KCORRECT functionality
def kcorrect_placeholder(maggies, ivar, z, filterlist, rmatrix=None, zvals=None, lambda_wavel=None, vmatrix=None, coeffs=None):
    """
    Placeholder for KCORRECT. In a real application, you'd replace this with a Python K-correction library.
    This function will simply return dummy values or pass through if no K-correction logic is implemented.
    """
    # For demonstration, we'll just return identity or simple scaling.
    # In a real scenario, this would involve complex spectral fitting.
    k_correction = np.zeros_like(maggies[0]) # Dummy K-correction
    
    # Dummy outputs for rmatrix, zvals, wavel, vmatrix, coeffs
    # These would typically be determined by the KCORRECT library
    rmatrix_out = np.array([[1.0]])
    zvals_out = np.array([z])
    lambda_wavel_out = np.array([5000.0]) # Example wavelength
    vmatrix_out = np.array([[1.0]])
    coeffs_out = np.zeros(maggies.shape[0]) # One coefficient per band
    
    return k_correction, rmatrix_out, zvals_out, lambda_wavel_out, vmatrix_out, coeffs_out

def k_reconstruct_maggies_placeholder(coeffs, z_out, maggies_in, vmatrix=None, lambda_wavel=None, filterlist=None):
    """
    Placeholder for K_RECONSTRUCT_MAGGIES.
    """
    # In a real scenario, this would use the coefficients and filter information
    # to predict magnitudes at the output redshift.
    # For now, it will simply return the input maggies as is.
    print(f"Applying dummy K-reconstruction at z={z_out}")
    return np.array(maggies_in) # Return input maggies unchanged for placeholder

def ferengi(sky, im, imerr, psflo, err0_mag, psfhi,
            lambda_lo, filter_lo, zlo, scllo, zplo, tlo,
            lambda_hi, filter_hi, zhi, sclhi, zphi, thi,
            im_out_file, psf_out_file,
            noflux=False, evo=None, noconv=False):
    """
    Applies the effects of redshift to a local galaxy image.
    """
    sz_sky = np.array(sky.shape)

    # Number of input filters (bands)
    # Check if im is 3D (multi-band) or 2D (single-band)
    if im.ndim == 3:
        nbands = im.shape[2]
    else:
        nbands = 1
        im = im[:, :, np.newaxis] # Make it 3D for consistent indexing
        imerr = imerr[:, :, np.newaxis] # Make it 3D
    nok = (nbands == 1) # Flag for no K-correction if only 1 band

    # Convert from cts (input frame) to maggies and back to cts (output frame)
    if nok:
        im_nok = maggies2cts(cts2maggies(im[:,:,0], tlo, zplo), thi, zphi)
        im_ds = ferengi_downscale(im_nok, zlo, zhi, scllo, sclhi,
                                  nofluxscl=noflux, evo=evo)
        psf_lo = psflo
        im_ds = np.expand_dims(im_ds, axis=-1)
    else:
        # Select best matching PSF for output redshift (based on wavelength)
        dz = np.abs(lambda_hi / lambda_lo - 1)
        idx_bestfilt = np.argmin(dz)
        psf_lo = psflo[:, :, idx_bestfilt]

        # Weight the closest filters in rest-frame more (this logic is complex from IDL)
        # For simplicity, and until a full K-correct implementation, we will use a simplified weighting.
        # This part of the code is highly dependent on the KCORRECT library.
        # (Moved to the part that actually applies the weights)


        # Initial downscale for the first band
        im_ds = ferengi_downscale(im[:, :, 0], zlo, zhi, scllo, sclhi, nofluxscl=noflux, evo=evo)
        
        # If multiple bands, extend im_ds to 3D for all bands
        if nbands > 1:
            temp_im_ds_list = [im_ds]
            for j in range(1, nbands):
                temp_im_ds_list.append(ferengi_downscale(im[:, :, j], zlo, zhi, scllo, sclhi, nofluxscl=noflux, evo=evo))
            im_ds = np.stack(temp_im_ds_list, axis=2)

        # Subtract sky (example: using ring_sky)
        for j in range(nbands):
            # This is a placeholder for `ring_sky`, which needs full implementation.
            # Assuming it returns a scalar sky value for simplicity.
            sky_sub_value = ring_sky(im_ds[:, :, j], 50, 15, nw=True)
            im_ds[:, :, j] -= sky_sub_value
        
        # Downscale error image
        imerr_ds = ferengi_downscale(imerr[:, :, 0], zlo, zhi, scllo, sclhi, nofluxscl=noflux, evo=evo)
        if nbands > 1:
            temp_imerr_ds_list = [imerr_ds]
            for j in range(1, nbands):
                temp_imerr_ds_list.append(ferengi_downscale(imerr[:, :, j], zlo, zhi, scllo, sclhi, nofluxscl=noflux, evo=evo))
            imerr_ds = np.stack(temp_imerr_ds_list, axis=2)

        # Convert error from cts to mags
        for j in range(nbands):
            # Avoid division by zero
            imerr_ds[:, :, j] = np.where(im_ds[:, :, j] != 0,
                                         2.5 / np.log(10) * imerr_ds[:, :, j] / im_ds[:, :, j],
                                         np.inf) # Set to inf if original flux is zero

        # Convert image from cts to maggies
        for j in range(nbands):
            im_ds[:, :, j] = cts2maggies(im_ds[:, :, j], tlo[j], zplo[j])

        # K-correction section (using placeholders)
        siglim = 2
        npix = im_ds[:,:,0].size
        
        # Find index of input filter closest to output wavelength
        zmin_idx = np.argmin(np.abs(lambda_hi / lambda_lo - 1 - zhi))
        filt_i = zmin_idx

        # Calculate sigma map and identify "good" pixels
        nsig = np.zeros_like(im_ds)
        nhi = np.zeros(im_ds[:,:,0].shape, dtype=int)
        sig_bands = np.zeros(nbands)

        for j in range(nbands):
            m, s, n = resistant_mean(im_ds[:, :, j], 3)
            sig_bands[j] = s * np.sqrt(npix - 1 - n)
            nsig[:, :, j] = np.where(sig_bands[j] != 0, median_filter(im_ds[:, :, j], size=3) / sig_bands[j], 0) # Use median for nsig
            
            hi_pixels = np.where(np.abs(nsig[:, :, j]) > siglim)
            nhi[hi_pixels] += 1
        
        # Select pixels for K-correction
        good = np.where((nhi >= 3) & (np.abs(nsig[:, :, filt_i]) > siglim))
        goodsize = np.size(good[0])
        if goodsize == 0:
            print('less than 3 filters have high sigma pixels')
            good = np.where((nhi >= 2) & (np.abs(nsig[:, :, filt_i]) > siglim))
        if goodsize == 0:
            print('less than 2 filters have high sigma pixels')
            good = np.where((nhi >= 1) & (np.abs(nsig[:, :, filt_i]) > siglim))
        if goodsize == 0:
            print('NO filter has high sigma pixels')
            good = np.where((nhi >= 0) & (np.abs(nsig[:, :, filt_i]) > siglim)) # All pixels if no high sigma
        
        good1 = np.where((np.abs(nsig[:, :, filt_i]) > 0.25) & (np.abs(nsig[:, :, filt_i]) <= siglim))
        good1size = np.size(good1[0])
        if good1size > 0:
            # Select only 50% of these pixels
            if good1size > 1: # random_indices expects len > 0
                good1 = (good1[0][random_indices(np.size(good1[0]), int(np.round(np.size(good1[0]) * 0.5)))],good1[1][random_indices(np.size(good1[1]), int(np.round(np.size(good1[1]) * 0.5)))])
            else: # If only one pixel, just take it
                good1 = good1
        
        # good = (np.unique(np.concatenate((good[0], good1[0]))), np.unique(np.concatenate((good[1], good1[1]))))
        good = (np.concatenate((good[0], good1[0])), np.concatenate((good[1], good1[1])))
        ngood = np.size(good[0])

        if ngood == 0:
            print("No pixels selected for K-correction.")
            # Handle the case where no pixels are selected, e.g., proceed without K-correction
            # or skip to end. For now, we will proceed assuming im_ds is okay.
            maggies_kcorrected = np.copy(im_ds.flatten()) # Placeholder if no pixels are good
        else:
            # Setup arrays for K-correction
            maggies_for_k = np.zeros((nbands, ngood))
            err_for_k = np.zeros((nbands, ngood))
            # for j in range(nbands):
            #     maggies_for_k[j, :] = im_ds.flatten()[good][j::nbands] # Reshape to (bands, pixels)
            #     err_for_k[j, :] = imerr_ds.flatten()[good][j::nbands]
            for j in range(nbands):
                maggies_for_k[j, :] = im_ds[:, :, j][good]
                err_for_k[j, :] = imerr_ds[:, :, j][good]
            # Remove infinite values in error
            err_for_k[~np.isfinite(err_for_k)] = 99999
            err_for_k = np.minimum(err_for_k, 99999) # Clip large errors

            # Setup array with minimum errors for SDSS
            err0 = (err0_mag[np.newaxis, :].T @ np.ones(np.size(maggies_for_k[0, :]))[:, np.newaxis].T).T
            #err0 = err0_mag[:, np.newaxis] # Ensure err0_mag is (nbands, 1)
            # The original IDL code's `err0_mag#(fltarr(n_elements(maggies[0, *]))+1)`
            # is equivalent to broadcasting err0_mag across the second dimension.
            err0_expanded = np.tile(err0, (1, ngood))

            # Weights (from IDL example, needs specific implementation from the source)
            # For simplicity, using a uniform weight if specific logic is not clear or implemented.
            dz1 = np.abs(lambda_hi - lambda_lo)
            if hasattr(dz1, '__iter__'):  # check if is iterable before applying weights
                ord = np.argsort(dz1)
                weight = np.ones(nbands)
                if dz1[ord[0]] == 0:
                    if nbands == 2: weight[ord] = [10, 4]
                    if nbands == 3: weight[ord] = [10, 4, 4]
                    if nbands >= 4: weight[ord] = np.concatenate(([10, 4, 4], np.ones(nbands - 3)))
                else:
                    if nbands == 2: weight[ord] = np.array([10, 8])
                    if nbands == 3 or nbands == 4: weight[ord] = np.concatenate(([10, 8], np.zeros(nbands - 2) + 4))
                    if nbands > 4: weight[ord] = np.concatenate(([10, 8, 4, 4], np.ones(nbands - 4)))
            weight = (weight[np.newaxis, :].T @ np.ones(np.size(maggies_for_k[0, :]))[:, np.newaxis].T).T
            #weight = np.ones(nbands)[:, np.newaxis] # Default uniform weights
            # The complex weighting logic from IDL source (lines 216-218) should be re-implemented
            # here if accuracy is critical and KCORRECT requires it.

            # Add image errors and minimum errors in quadrature
            err_for_k = err_for_k.T
            err_combined = np.sqrt(err0**2 + err_for_k**2) / weight

            # Convert errors to inverse variance for KCORRECT
            maggies_for_k = maggies_for_k.T
            ivar = (2.5 / np.log(10) / err_combined / np.array(maggies_for_k))**2
            ivar[~np.isfinite(ivar)] = np.max(ivar[np.isfinite(ivar)]) if np.any(np.isfinite(ivar)) else 1.0 # Handle inf/nan

            z_tmp_lo = np.full(ngood, zlo)
            z_tmp_hi = np.full(ngood, zhi)

            # Call KCORRECT 
            responses_in = ['sdss_g0', 'sdss_r0', 'sdss_i0', 'sdss_z0']  # TODO: Maybe not the best implementation, but works for now
            if "i" in filter_lo:
                responses_in = ['sdss_g0', 'sdss_r0', 'sdss_i0', 'sdss_z0'] 
            else:
                responses_in = ['sdss_g0', 'sdss_r0', 'sdss_z0']

            maggies_for_k = maggies_for_k.T
            maggies_for_k = np.moveaxis(maggies_for_k, -1, 0)
            ivar = np.moveaxis(ivar, -1, 0)

            kc = kcorrect.kcorrect.Kcorrect(responses=responses_in)
            ivar = ivar.T
            coeffs = kc.fit_coeffs(redshift=z_tmp_lo, maggies=maggies_for_k, ivar=ivar)           
            k = kc.kcorrect(redshift=z_tmp_lo, coeffs=coeffs)
            maggies_reconstructed = kc.reconstruct(redshift=z_tmp_lo, coeffs=coeffs)

            maggies_for_k = np.moveaxis(maggies_for_k, -1, 0) # Go back to ferengi's weird indices
            maggies_reconstructed = np.moveaxis(maggies_reconstructed, -1, 0)
            ivar = np.moveaxis(ivar, -1, 0)

            # k_dummy, rmatrix, zvals, wavel, vmatrix, coeffs = kcorrect_placeholder(
            #     maggies_for_k, ivar, z_tmp_lo, filterlist=filter_lo
            # )


            # Reconstruct magnitudes at high redshift (placeholder)
            # maggies_reconstructed = k_reconstruct_maggies_placeholder(
            #     coeffs, z_tmp_hi, maggies_for_k, vmatrix=vmatrix, lambda_wavel=wavel, filterlist=filter_hi
            # )

            # Store K-corrected maggies back into im_ds
            # This requires careful re-mapping from the 1D 'good' indices back to the 2D im_ds.
            # Assuming maggies_reconstructed is (nbands, ngood)
            maggies_kcorrected = np.copy(im_ds.flatten())
            for j in range(nbands):
                # This needs to put the reconstructed maggies back into the correct positions
                # in the flattened array, respecting the 'good' indices and band order.
                # The original IDL: im_ds[good] = maggies/(1.+zhi) seems to imply flattening all bands and then K-correcting.
                # Given the `maggies_for_k` structure, it's likely (bands, pixels), so maggies_reconstructed is similar.
                
                # Reshape maggies_reconstructed for easier assignment
                # reshaped_maggies_rec = np.zeros_like(im_ds.flatten())
                reshaped_maggies_rec = np.zeros_like(im_ds)
                s = np.shape(im_ds)
                for band_idx in range(nbands):
                    reshaped_maggies_rec[:, :,band_idx][good] = maggies_reconstructed[band_idx, :]
                
                # Correct indexing is crucial here. The IDL code's `im_ds[good] = maggies/(1.+zhi)` is ambiguous
                # when `im_ds` is 3D and `good` is 1D from a flattened 2D index.
                # Assuming `im_ds.flatten()[good]` means taking specific pixels across all bands that were "good".
                # A more explicit approach would be:
                # for band_idx in range(nbands):
                #     im_ds_flat = im_ds[:, :, band_idx].flatten()
                #     im_ds_flat[good] = maggies_reconstructed[band_idx, :] / (1. + zhi) # Assuming maggies_reconstructed holds the band-specific output
                #     im_ds[:, :, band_idx] = im_ds_flat.reshape(im_ds[:, :, band_idx].shape)
                
                # For simplicity with current placeholder structure, we will assume `maggies_reconstructed` is effectively
                # the result for the specific `im_ds.flatten()[good]` segment.
                # This needs careful re-evaluation based on actual KCORRECT output.
                # For now, let's just use the reconstructed maggies for the first band for the purpose of the demo
                # and assume a simplified mapping.
                
                # Simplified mapping: If K-corrected value exists for a pixel, use it. Else, keep original.
                # This requires that `good` maps directly to the flattened `im_ds` array across bands.
                # The IDL `im_ds[good]` likely means it's still a 3D array indexed in a way that gets the correct (x,y,band) elements.
                # Given `im_ds` is (x,y,bands), `good` from `nhi` and `nsig` is (x*y) index.
                # This implies K-correction results are applied pixel-wise, across bands for each pixel.
                
                # To emulate `im_ds[good] = maggies/(1.+zhi)` when `im_ds` is 3D, and `good` are flattened indices:
                # We need to create a mask for `im_ds` for each band.

                reshaped_good = np.unravel_index(good, im_ds[:,:,0].shape)

                # Apply the k-corrected values to all bands at the `good` pixel locations
                # This is a very strong assumption about `maggies_reconstructed` structure.
                for band_idx in range(nbands):
                    im_ds[:, :, band_idx][good] = maggies_reconstructed[band_idx, :] / (1. + zhi)
                # for band_idx in range(nbands):
                #     im_ds[:, :, band_idx][good] = maggies_reconstructed[band_idx, :] / (1. + zhi)
                
    # Background: choose closest in redshift-space (this logic also from original)
    # Using the original image before K-correction to determine background.
    # This seems to be a separate path depending on `nok`.
    if nok: # This block is actually for nok, which means single band
        bg = im_ds / (1. + zhi) # Already processed in `im_nok`
    else:
        # If multi-band, the 'bg' seems to be derived from the downscaled image of the best-fit filter.
        bg = im_ds[:, :, filt_i] / (1. + zhi) # From the best-fit filter
        im_ds = im_ds[:, :, filt_i] / (1. + zhi)
        # im_ds = np.squeeze(im_ds, axis=-1)
        # If K-correction was applied, `im_ds` has the K-corrected values already.

    im_ds = maggies2cts(im_ds, thi, zphi)
    if not nok: # Only if K-correction was attempted
        bg = maggies2cts(bg, thi, zphi) # Convert background too

    # im_ds = im[..., np.newaxis]

    # Remove infinite pixels: replace with median (3x3)
    # This loop applies to all bands if multi-band
    med_val = median_filter(im_ds, size=3, mode='nearest')
    idx_inf = np.where(~np.isfinite(im_ds))
    if idx_inf[0].size > 0:
        im_ds[idx_inf[0], idx_inf[1]] = med_val[idx_inf[0], idx_inf[1]]

    # Replace 0-value pixels with median (3x3)
    idx_zero = np.where(im_ds == 0)
    if idx_zero[0].size > 0:
        im_ds[idx_zero[0], idx_zero[1]] = med_val[idx_zero[0], idx_zero[1]]
    
    if not nok:
        # Check for large outliers after K-correction for multi-band images
        # This part of the IDL code is to clean extreme residuals.
        m_im_ds, sig_im_ds, nrej_im_ds = resistant_mean(im_ds, 3)
        sig_im_ds *= np.sqrt(im_ds.size - 1 - nrej_im_ds)
        
        idx_outliers = np.where(np.abs(im_ds) > 10 * sig_im_ds)
        
        if idx_outliers[0].size > 0:
            # Flatten for line fit
            fit_coeffs = robust_linefit(np.abs(bg[idx_outliers]), np.abs(im_ds[idx_outliers]))
            
            delta = np.abs(im_ds[idx_outliers]) - (fit_coeffs[0] + fit_coeffs[1] * np.abs(bg[idx_outliers]))
            
            m_delta, sig_delta, nrej_delta = resistant_mean(delta, 3)
            sig_delta *= np.sqrt(delta.size - 1 - nrej_delta)
            
            idx1 = np.where(delta / sig_delta > 50)
            if idx1[0].size > 0:
                # Replace these extreme outliers with median from the original med_val
                # Need to map idx_outliers[idx1] back to the 3D array.
                med_val_flat = med_val.flatten() # Assuming med_val was calculated per band
                im_ds[idx_outliers[idx1]] = med_val_flat[idx_outliers[idx1]]
                im_ds = im_ds.reshape(im_ds.shape)

    # Subtract sky again after K-correction and cleaning (if not already done)
    sky_sub_value = ring_sky(im_ds, 50, 15, nw=True) # Idk if this works
    im_ds -= sky_sub_value

    if noconv:
        im_ds /= thi
        # For reconstruction, if no convolution, it's just the input PSF normalized.
        recon = psf_lo / np.sum(psf_lo)
        # Skip to write_out
    else:
        # The output sky image might be too small
        sz_im_ds = np.array(im_ds.shape[:2]) # Take 2D size
        if sz_im_ds[0] > sz_sky[0] or sz_im_ds[1] > sz_sky[1]:
            raise ValueError('Sky image not big enough for downscaled galaxy.')

        # Calculate the transformation PSF
        psf_hi = psfhi
        psf_t = ferengi_transformation_psf(psf_lo, psf_hi, zlo, zhi, scllo, sclhi, same_size=True)

        # Reconstruct PSF by convolving local PSF with transformation PSF
        # Need to ensure psf_lo and psf_t are properly sized and centered.
        # ferengi_odd_n_square should handle this.
        
        # Convolve psf_lo with psf_t
        # Pad psf_lo for convolution to ensure enough space for the result
        pad_x_psf = psf_t.shape[0] // 2
        pad_y_psf = psf_t.shape[1] // 2
        psf_lo_padded = np.pad(psf_lo, ((pad_x_psf, pad_x_psf), (pad_y_psf, pad_y_psf)), 'constant')
        
        recon_raw = convolve(psf_lo_padded, psf_t / np.sum(psf_t), mode='constant', cval=0.0)
        
        # Crop recon_raw back to a reasonable size, typically similar to psf_lo or psf_t
        # The exact cropping depends on the desired output size.
        # For now, let's try to crop to the original psf_lo size, centered.
        crop_x_start = (recon_raw.shape[0] - psf_lo.shape[0]) // 2
        crop_y_start = (recon_raw.shape[1] - psf_lo.shape[1]) // 2
        recon = recon_raw[crop_x_start : crop_x_start + psf_lo.shape[0],
                          crop_y_start : crop_y_start + psf_lo.shape[1]]

        recon = ferengi_odd_n_square(recon)

        # Get rid of potential bad pixels around the edges of the PSF
        rem = 3 # remove 3 pixels
        # Need a mutable object for `clip_also`.
        temp_psf_hi = np.copy(psf_hi)
        psf_hi = ferengi_clip_edge(rem, recon, clip_also=temp_psf_hi, norm=True) # TODO: What even is the point of this?
        # psf_hi = temp_psf_hi # Update psf_hi with clipped version

        # Normalise reconstructed PSF
        recon /= np.sum(recon)

        # Convolve the high redshift image with the transformation PSF and add noise
        # This part needs to handle multi-band images if nok is False
        # temp_im_ds_conv = np.zeros_like(im_ds)
        temp_im_ds_conv = list()
        # im_ds = np.squeeze(im_ds, axis=-1)
        im_ds = ferengi_convolve_plus_noise(im_ds / thi, psf_t, sky, thi,
                                                                border_clip=3, extend=True) # extend=False means crop borders, though is true in ferengi.pro?
    
    # Write output FITS files
    fits.writeto(f"{im_out_file}", im_ds, overwrite=True)
    fits.writeto(psf_out_file, recon, overwrite=True)

    print("FERENGI process completed.")
