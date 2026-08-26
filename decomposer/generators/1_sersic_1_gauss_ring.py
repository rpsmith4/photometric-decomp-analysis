"""Generate a generic IMFIT config for a host Sersic plus a polar Gaussian ring."""

import numpy as np
import pyimfit
from astropy.modeling import fitting, models
import scipy
import matplotlib.pyplot as plt
import traceback as tb
import astropy
import astropy.units as u
from astropy.stats import sigma_clipped_stats
import os
import re

def clamp(x, lo, hi):
    return max(lo, min(hi, x))

def _safe_ellipticity(ellipse_fit_data, component, fallback=0.3):
    subset = ellipse_fit_data[ellipse_fit_data["PolarOrHost"] == component]
    if subset.empty:
        return float(fallback)
    if "ellipticity" in subset.columns:
        val = subset["ellipticity"].iloc[0]
        if np.isfinite(val):
            return float(np.clip(val, 0.0, 0.95))
    if "semi_major" in subset.columns and "semi_minor" in subset.columns:
        a = float(subset["semi_major"].iloc[0])
        b = float(subset["semi_minor"].iloc[0])
        if a > 0:
            return float(np.clip((a - b) / a, 0.0, 0.95))
    return float(fallback)


def pa_to_imfit(pa_deg):
    return (float(pa_deg) + 90.0) % 180.0


def est_noise_floor(img, mask=None):
    valid = np.isfinite(img)
    if mask is not None:
        valid &= mask <= 0

    img = img[valid]

    background, _, noise = sigma_clipped_stats(img, sigma=3.0, maxiters=100)
    return background + noise


def _surface_brightness_to_nmgy_per_pixel(mu, zeropoint, pixel_scale):
    pixel_area = (float(pixel_scale) * u.arcsec) ** 2
    nanomaggy = 3631.0e-6 * u.Jy
    flux_per_arcsec2 = 10.0 ** ((float(zeropoint) - float(mu)) / 2.5) * nanomaggy / u.arcsec**2
    return (flux_per_arcsec2 * pixel_area / nanomaggy).to_value(u.dimensionless_unscaled)

def radial_slc(pa, img, c=None, mask=None, samples=1000, spline_order=1):
    pa = np.deg2rad(pa)
    if c is None:
        c = (img.shape[1] / 2.0, img.shape[0] / 2.0)

    max_radius = min(
        (c[0] / abs(np.cos(pa))) if np.cos(pa) !=0 else np.inf,
        ((img.shape[1] - 1 - c[0]) / abs(np.cos(pa))) if np.cos(pa) !=0 else np.inf,
        (c[1] / abs(np.sin(pa))) if np.sin(pa) !=0 else np.inf,
        ((img.shape[0] - 1 - c[1]) / abs(np.sin(pa))) if np.sin(pa) !=0 else np.inf,
    )
    radius = np.linspace(0.0, max_radius, samples // 2)
    x = c[0] + np.concatenate((radius, -radius)) * np.cos(pa)
    y = c[1] + np.concatenate((radius, -radius)) * np.sin(pa)
    prof = scipy.ndimage.map_coordinates(
        img, np.array([y, x]), order=spline_order, mode="constant", cval=np.nan
    )
    if mask is not None:
        mask_values = scipy.ndimage.map_coordinates(mask, np.array([y, x]), order=0, mode="constant", cval=np.nan)
        prof[mask_values > 0] = np.nan

    n_radius = radius.size
    upper = prof[:n_radius]
    lower = prof[n_radius:]
    prof = np.nanmean(np.vstack((upper, lower)), axis=0)

    return radius, prof, x, y

def _fit_sersic_profile(radius, intensity, w=None):
    valid = np.isfinite(radius) & np.isfinite(intensity) & (radius >= 0) & (intensity > 0)
    if np.count_nonzero(valid) < 8:
        raise ValueError("Host radial profile has too few positive samples for a Sersic fit")

    radius = radius[valid]
    intensity = intensity[valid] 
    peak = np.nanmax(intensity)
    if w is not None:
        w = w[valid]
    Ie_initial = peak / 2.0
    re_initial = radius[np.argmin(np.abs(intensity - Ie_initial))]
    model = models.Sersic1D(
        amplitude=Ie_initial,
        r_eff=re_initial,
        n=0.5, # Randomly guess this is a guassian profile 
        bounds={"amplitude": (0, None), "r_eff": (0.0, None), "n": (0.0, 15.0)},
    )
    fitter = fitting.TRFLSQFitter()
    fitted = fitter(model, radius, intensity, estimate_jacobian=True, weights=w)
    # print(fitter.fit_info)
    if not np.all(np.isfinite([fitted.amplitude.value, fitted.r_eff.value, fitted.n.value])):
        raise ValueError("Host radial profile Sersic fit returned non-finite parameters")
    return fitted


def _plot_sersic_fit(image, radius, intensity, fitted, sample_x, sample_y, center):
    fit_radius = np.linspace(float(radius.min()), float(radius.max()), 500)
    figure, (image_axis, profile_axis) = plt.subplots(1, 2, figsize=(12, 5))

    finite_image = image[np.isfinite(image)]
    vmin, vmax = np.nanpercentile(finite_image, [1, 99])
    image_axis.imshow(image, origin="lower", cmap="gray", vmin=vmin, vmax=vmax)
    image_axis.scatter(sample_x, sample_y, s=4, c="tab:red", alpha=0.7, label="Samples")
    image_axis.scatter(*center, s=35, c="tab:blue", marker="+", label="Center")
    image_axis.set_title("Host radial-profile samples")
    image_axis.set_xlabel("X (pixels)")
    image_axis.set_ylabel("Y (pixels)")
    image_axis.legend(loc="upper right")

    profile_axis.loglog(radius, intensity, ".", ms=3, label="Profile")
    profile_axis.loglog(fit_radius, fitted(fit_radius), "-", label="1D Sersic fit")
    profile_axis.axvline(x=fitted.r_eff.value)
    profile_axis.set_xlabel("Radius (pixels)")
    profile_axis.set_ylabel("Intensity")
    profile_axis.legend()
    figure.tight_layout()
    plt.show()

def read_reg_file(file_p):
    ell_params = {"host": {}, "annulus_inner": {}, "annulus_outer": {}}
    
    with open(file_p) as f:
        lines = f.readlines()
    for line in lines:
        isophote_level = re.search(".*isophote=([0-9]+)", line)
        params = re.search("ellipse\(([+-]?\d*\.?\d+),([+-]?\d*\.?\d+),([+-]?\d*\.?\d+),([+-]?\d*\.?\d+),([+-]?\d*\.?\d+)\)", line)

        if isophote_level is not None:
            isophote_level = isophote_level.group(1)
        if params is not None:
            xctr, yctr, rmaj, rmin, pa = [float(i) for i in params.groups()]
            params_dict = {"ctr": [xctr, yctr], "rmaj": rmaj, "rmin": rmin, "PA": pa, "ell": (rmaj-rmin)/rmaj, "ecc": np.sqrt(1-rmin**2/rmaj**2), "isophote": isophote_level}

            if "host" in line:
                ell_params["host"] = params_dict
            elif "annulus_inner" in line:
                ell_params["annulus_inner"] = params_dict
            elif "annulus_outer" in line:
                ell_params["annulus_outer"] = params_dict

    return ell_params
def generate_init_guess_fallback(fltr: str,
                      sci_fits: np.array,
                      mask_fits: np.array = None,
                      psf_fits: np.array = None,
                      invvar_fits: np.array = None,
                      psg_type: str = "ring",
                      ellipse_fit_data=None,
                      zeropoint: float = 22.5,
                      pixel_scale: float = 0.262,
                      galaxy_type=None,
                      phot_params: str = "automatic",
                      plot_slits: bool = False,
                      data_loc: str | None = None,
                      **kwargs):

    if ellipse_fit_data is None:
        raise ValueError("Ellipse fit data is required for gaussian ring config generation")

    sci_header = sci_fits.header
    cx = float(sci_header.get("CRPIX1", sci_fits.shape[1] / 2.0))
    cy = float(sci_header.get("CRPIX2", sci_fits.shape[0] / 2.0))

    host_row = ellipse_fit_data[ellipse_fit_data["PolarOrHost"] == "Host"]
    polar_row = ellipse_fit_data[ellipse_fit_data["PolarOrHost"] == "Polar"]

    if host_row.empty or polar_row.empty:
        raise ValueError("ellipse_fit_data must contain both Host and Polar entries")

    host_pa_imfit = pa_to_imfit(float(host_row["angle"].iloc[0]))
    polar_pa_imfit = pa_to_imfit(float(polar_row["angle"].iloc[0]))
    host_ell = _safe_ellipticity(ellipse_fit_data, "Host", fallback=0.25)
    polar_ell = _safe_ellipticity(ellipse_fit_data, "Polar", fallback=0.25)

    img_shape = sci_fits.data.shape
    img = sci_fits.data
    try:
        host_radius, host_profile, _, _ = radial_slc(
            host_pa_imfit - 90, img, c=(cx, cy), mask=mask_fits
        )
        host_fit = _fit_sersic_profile(host_radius, host_profile)
        if False:
            _plot_sersic_fit(host_radius, host_profile, host_fit)
    except Exception as exc:
        print(tb.format_exc())
        raise ValueError("Unable to fit the host radial profile with a Sersic model") from exc

    host_re_pix = np.clip(host_fit.r_eff.value, 0, min(img_shape))
    host_Ie_pix = max(host_fit.amplitude.value, 0)
    host_n = np.clip(host_fit.n.value, 0, 15.0)

    polar_A = max(1e-4, host_Ie_pix * 0.15)
    polar_R = max(10.0, host_re_pix * 2.0)
    polar_sigma_r = max(5.0, host_re_pix * 0.5)

    # prepare for extreme magic number disaster
    model = pyimfit.SimpleModelDescription()
    model.x0.setValue(cx, [cx - 15.0, cx + 15.0])
    model.y0.setValue(cy, [cy - 15.0, cy + 15.0])

    host = pyimfit.make_imfit_function("Sersic", label="Host")
    host.PA.setValue(host_pa_imfit, [max(0.0, host_pa_imfit - 10.0), min(180.0, host_pa_imfit + 10.0)])
    host.ell.setValue(host_ell, [max(0.0, host_ell - 0.10), min(0.95, host_ell + 0.10)])
    host.n.setValue(host_n, [0.5, 6.0])
    host.I_e.setValue(host_Ie_pix, [max(1e-6, host_Ie_pix * 0.02),host_Ie_pix * 1.5])
    host.r_e.setValue(host_re_pix, [max(1.0, host_re_pix * 0.3), max(host_re_pix * 2.0, host_re_pix + 1.0)])

    polar = pyimfit.make_imfit_function("GaussianRing", label="Polar")
    polar.PA.setValue(polar_pa_imfit, [max(0.0, polar_pa_imfit - 10.0), min(180.0, polar_pa_imfit + 10.0)])
    polar.ell.setValue(polar_ell, [max(0.0, polar_ell - 0.15), min(0.95, polar_ell + 0.15)])
    polar.A.setValue(polar_A, [max(1e-6, polar_A * 0.02), max(polar_A * 2.0, polar_A + 1e-6)])
    polar.R_ring.setValue(polar_R, [max(1.0, polar_R * 0.5), max(1.0, polar_R * 2.5)])
    polar.sigma_r.setValue(polar_sigma_r, [max(1.0, polar_sigma_r * 0.3), max(1.0, polar_sigma_r * 2.5)])

    model.addFunction(host)
    model.addFunction(polar)

    return model



def generate_init_guess_main(fltr: str,
                      sci_fits: np.array,
                      mask_fits: np.array = None,
                      psf_fits: np.array = None,
                      invvar_fits: np.array = None,
                      psg_type: str = "ring",
                      ellipse_fit_data=None,
                      zeropoint: float = 22.5,
                      pixel_scale: float = 0.262,
                      galaxy_type=None,
                      phot_params: str = "automatic",
                      plot_slits: bool = False,
                      data_loc: str | None = None,
                      **kwargs):

    galname = os.path.basename(data_loc)
    reg_file = os.path.join(data_loc, f"{galname}_regions.reg")
    ell_params = read_reg_file(reg_file)
    
    if ellipse_fit_data is None:
        raise ValueError("Ellipse fit data is required for gaussian ring config generation")
    cx = sci_fits.shape[1] / 2.0
    cy = sci_fits.shape[0] / 2.0

    host_row = ellipse_fit_data[ellipse_fit_data["PolarOrHost"] == "Host"]
    polar_row = ellipse_fit_data[ellipse_fit_data["PolarOrHost"] == "Polar"]

    if host_row.empty or polar_row.empty:
        raise ValueError("ellipse_fit_data must contain both Host and Polar entries")

    host_pa_imfit = pa_to_imfit(host_row["angle"].iloc[0])
    polar_pa_imfit = pa_to_imfit(ell_params["annulus_inner"]["PA"])
    host_ell = _safe_ellipticity(ellipse_fit_data, "Host", fallback=0.25)
    polar_ell = ell_params["annulus_inner"]["ell"]

    img = sci_fits.data
    try:
        noise_floor = est_noise_floor(img, mask=mask_fits)
        fit_img = np.array(img, dtype=float, copy=True)
        fit_img[~np.isfinite(fit_img) | (fit_img <= noise_floor)] = np.nan
        host_radius, host_profile, sample_x, sample_y = radial_slc(
            host_pa_imfit - 90, fit_img, c=(cx, cy), mask=mask_fits
        )

        _, invvar_prof, _, _ = radial_slc(
            host_pa_imfit - 90, invvar_fits, c=(cx, cy), mask=mask_fits
        )

        host_fit = _fit_sersic_profile(host_radius, host_profile, np.sqrt(invvar_prof))
        if False:
            _plot_sersic_fit(img, host_radius, host_profile, host_fit, sample_x, sample_y, (cx, cy))
    except Exception as exc:
        print(tb.format_exc())
        raise ValueError("Unable to fit the host radial profile with a Sersic model") from exc

    host_re_pix = max(host_fit.r_eff.value, 0)
    host_Ie_pix = max(host_fit.amplitude.value, 0)
    host_n = np.clip(host_fit.n.value, 0, 15.0)

    polar_A = _surface_brightness_to_nmgy_per_pixel(ell_params["annulus_inner"]["isophote"], zeropoint, pixel_scale)
    polar_R = (ell_params["annulus_inner"]["rmaj"] + ell_params["annulus_outer"]["rmin"])/2
    # polar_sigma_r = 1/(np.sqrt(2*np.pi)*polar_A)
    polar_sigma_r = 2*(ell_params["annulus_inner"]["rmaj"] - ell_params["annulus_outer"]["rmin"])

    # prepare for extreme magic number disaster
    pa_tol = 5.0
    ell_tol = 0.1
    n_tol = 0.75

    model = pyimfit.SimpleModelDescription()
    model.x0.setValue(cx, [cx - 15.0, cx + 15.0])
    model.y0.setValue(cy, [cy - 15.0, cy + 15.0])

    host = pyimfit.make_imfit_function("Sersic", label="Host")

    host.PA.setValue(host_pa_imfit, [host_pa_imfit - pa_tol, host_pa_imfit + pa_tol])
    host.ell.setValue(host_ell, [clamp(host_ell - ell_tol, 0,0.95), clamp(host_ell + ell_tol*2, 0,0.95)])
    host.n.setValue(host_n)
    host.n.setTolerance(n_tol)
    host.I_e.setValue(host_Ie_pix, [max(1e-6, host_Ie_pix /100),host_Ie_pix * 5])
    host.r_e.setValue(host_re_pix, [0.1*host_re_pix, 1.5*host_re_pix])

    polar = pyimfit.make_imfit_function("GaussianRing", label="Polar Ring")
    polar.PA.setValue(polar_pa_imfit, [polar_pa_imfit - pa_tol, polar_pa_imfit + pa_tol])
    polar.ell.setValue(polar_ell, [max(0.0, polar_ell - ell_tol), min(0.95, polar_ell + ell_tol)])
    polar.A.setValue(polar_A, [max(1e-6, polar_A * 0.02), max(polar_A * 2.0, polar_A + 1e-6)])
    polar.R_ring.setValue(polar_R, [polar_R * 0.5, polar_R * 2.5])
    polar.sigma_r.setValue(polar_sigma_r, [polar_sigma_r * 0.3, polar_sigma_r * 2.5])

    model.addFunction(host)
    model.addFunction(polar)

    return model


def generate_init_guess(fltr: str,
                      sci_fits: np.array,
                      mask_fits: np.array = None,
                      psf_fits: np.array = None,
                      invvar_fits: np.array = None,
                      psg_type: str = "ring",
                      ellipse_fit_data=None,
                      zeropoint: float = 22.5,
                      pixel_scale: float = 0.262,
                      galaxy_type=None,
                      phot_params: str = "automatic",
                      plot_slits: bool = False,
                      data_loc: str | None = None,
                      **kwargs):
    try:
        return generate_init_guess_main(fltr,
                      sci_fits,
                      mask_fits,
                      psf_fits,
                      invvar_fits,
                      psg_type,
                      ellipse_fit_data,
                      zeropoint,
                      pixel_scale,
                      galaxy_type,
                      phot_params,
                      plot_slits,
                      data_loc,
                      **kwargs)
    except Exception as e:
        print(f"{e}")
        print("Error using regular init guess, using fallback")
        return generate_init_guess_fallback(fltr,
                      sci_fits,
                      mask_fits,
                      psf_fits,
                      invvar_fits,
                      psg_type,
                      ellipse_fit_data,
                      zeropoint,
                      pixel_scale,
                      galaxy_type,
                      phot_params,
                      plot_slits,
                      data_loc,
                      **kwargs)