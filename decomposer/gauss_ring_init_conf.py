"""Generate a generic IMFIT config for a host Sersic plus a polar Gaussian ring."""

import numpy as np
import pyimfit
from astropy.modeling import fitting, models
from photometric_cut_helpers import pixel_scale_from_header_arcsec_per_pix
import scipy
import matplotlib.pyplot as plt
import traceback as tb


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


def _pa_to_imfit(pa_deg):
    return (float(pa_deg) + 90.0) % 180.0

def radial_slc(pa, img, c=None, mask=None, samples=1000):
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
        img, np.array([y, x]), order=1, mode="constant", cval=np.nan
    )
    if mask is not None:
        mask_values = scipy.ndimage.map_coordinates(mask, np.array([y, x]), order=0, mode="constant", cval=np.nan)
        prof[mask_values > 0] = np.nan

    n_radius = radius.size
    upper = prof[:n_radius]
    lower = prof[n_radius:]
    prof = np.nanmean(np.vstack((upper, lower)), axis=0)
    fig,ax = plt.subplots(2,1)
    ax[0].imshow(img)
    ax[0].plot(x,y)
    ax[1].plot(radius, prof)
    plt.show()
    return radius, prof


def _fit_sersic_profile(radius, intensity):
    valid = np.isfinite(radius) & np.isfinite(intensity) & (radius >= 0) & (intensity > 0)
    if np.count_nonzero(valid) < 8:
        raise ValueError("Host radial profile has too few positive samples for a Sersic fit")

    radius = np.asarray(radius[valid], dtype=float)
    intensity = np.asarray(intensity[valid], dtype=float)
    peak = float(np.nanmax(intensity))
    half_max = peak / 2.0
    re_initial = float(radius[np.argmin(np.abs(intensity - half_max))])
    re_initial = max(re_initial, 1.0)
    model = models.Sersic1D(
        amplitude=max(peak / 2.0, 1e-6),
        r_eff=re_initial,
        n=2.0,
        bounds={"amplitude": (1e-8, None), "r_eff": (0.5, None), "n": (0.3, 8.0)},
    )
    fitted = fitting.TRFLSQFitter()(model, radius, intensity)
    if not np.all(np.isfinite([fitted.amplitude.value, fitted.r_eff.value, fitted.n.value])):
        raise ValueError("Host radial profile Sersic fit returned non-finite parameters")
    return fitted



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

    if ellipse_fit_data is None:
        raise ValueError("Ellipse fit data is required for gaussian ring config generation")

    sci_header = sci_fits.header
    cx = float(sci_header.get("CRPIX1", sci_fits.shape[1] / 2.0))
    cy = float(sci_header.get("CRPIX2", sci_fits.shape[0] / 2.0))

    host_row = ellipse_fit_data[ellipse_fit_data["PolarOrHost"] == "Host"]
    polar_row = ellipse_fit_data[ellipse_fit_data["PolarOrHost"] == "Polar"]

    if host_row.empty or polar_row.empty:
        raise ValueError("ellipse_fit_data must contain both Host and Polar entries")

    host_pa_imfit = _pa_to_imfit(float(host_row["angle"].iloc[0]))
    polar_pa_imfit = _pa_to_imfit(float(polar_row["angle"].iloc[0]))
    host_ell = _safe_ellipticity(ellipse_fit_data, "Host", fallback=0.25)
    polar_ell = _safe_ellipticity(ellipse_fit_data, "Polar", fallback=0.25)

    img_shape = sci_fits.data.shape
    img = sci_fits.data
    mask = None if mask_fits is None else mask_fits > 0
    try:
        host_radius, host_profile = radial_slc(
            host_pa_imfit - 90, img, c=(cx, cy), mask=mask
        )
        host_fit = _fit_sersic_profile(host_radius, host_profile)
    except Exception as exc:
        print(tb.format_exc())
        raise
        # raise ValueError("Unable to fit the host radial profile with a Sersic model") from exc

    host_re_pix = float(np.clip(host_fit.r_eff.value, 1.0, min(img_shape)))
    host_Ie_pix = max(float(host_fit.amplitude.value), 1e-6)
    host_n = float(np.clip(host_fit.n.value, 0.5, 6.0))

    polar_A = max(1e-4, host_Ie_pix * 0.15)
    polar_R = max(10.0, host_re_pix * 2.0)
    polar_sigma_r = max(5.0, host_re_pix * 0.5)

    model = pyimfit.SimpleModelDescription()
    model.x0.setValue(cx, [cx - 15.0, cx + 15.0])
    model.y0.setValue(cy, [cy - 15.0, cy + 15.0])

    host = pyimfit.make_imfit_function("Sersic", label="Host")
    host.PA.setValue(host_pa_imfit, [max(0.0, host_pa_imfit - 10.0), min(180.0, host_pa_imfit + 10.0)])
    host.ell.setValue(host_ell, [max(0.0, host_ell - 0.10), min(0.95, host_ell + 0.10)])
    host.n.setValue(host_n, [0.5, 6.0])
    host.I_e.setValue(host_Ie_pix, [max(1e-6, host_Ie_pix * 0.02), max(host_Ie_pix * 0.5, host_Ie_pix)])
    host.r_e.setValue(host_re_pix, [max(1.0, host_re_pix * 0.3), max(host_re_pix * 2.0, host_re_pix + 1.0)])

    polar = pyimfit.make_imfit_function("GaussianRing", label="Polar")
    polar.PA.setValue(polar_pa_imfit, [max(0.0, polar_pa_imfit - 10.0), min(180.0, polar_pa_imfit + 10.0)])
    polar.ell.setValue(polar_ell, [max(0.0, polar_ell - 0.15), min(0.95, polar_ell + 0.15)])
    polar.A.setValue(polar_A, [max(1e-6, polar_A * 0.02), max(polar_A * 2.0, polar_A + 1e-6)])
    polar.R_ring.setValue(polar_R, [max(1.0, polar_R * 0.5), max(1.0, polar_R * 2.5)])
    polar.sigma_r.setValue(polar_sigma_r, [max(1.0, polar_sigma_r * 0.3), max(1.0, polar_sigma_r * 2.5)])

    model.addFunction(host)
    model.addFunction(polar)

    return model, pixel_scale, zeropoint
