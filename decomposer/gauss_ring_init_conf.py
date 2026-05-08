"""Generate a generic IMFIT config for a host Sersic plus a polar Gaussian ring."""

import numpy as np
import pyimfit
from photometric_cut_helpers import pixel_scale_from_header_arcsec_per_pix


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


def gather_parameters(fltr: str,
                      sci_fits: np.array,
                      mask_fits: np.array = None,
                      psf_fits: np.array = None,
                      invvar_fits: np.array = None,
                      psg_type: str = "ring",
                      ellipse_fit_data=None,
                      zeropoint: float = None,
                      pixel_scale: float = None,
                      galaxy_type=None,
                      phot_params: str = "automatic",
                      plot_slits: bool = False,
                      data_loc: str | None = None):
    """Return a generic host Sersic + polar GaussianRing model description."""
    if ellipse_fit_data is None:
        raise ValueError("ellipse_fit_data is required for gaussian ring config generation")

    if pixel_scale is None:
        pixel_scale = float(pixel_scale_from_header_arcsec_per_pix(sci_fits))
    if zeropoint is None:
        zeropoint = 22.5

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

    shape = getattr(sci_fits, "data", None)
    if shape is None:
        shape = sci_fits
    if hasattr(shape, "shape"):
        img_shape = shape.shape
    else:
        img_shape = (0, 0)

    # Generic starting guesses based on image scale and geometry.
    host_re_pix = max(5.0, min(80.0, min(img_shape) / 8.0))
    host_Ie_pix = max(1e-3, float(np.nanmax(sci_fits.data)) / 50.0)
    host_n = 2.0

    polar_A = max(1e-4, host_Ie_pix * 0.15)
    polar_R = max(10.0, host_re_pix * 2.0)
    polar_sigma_r = max(5.0, host_re_pix * 0.5)

    model = pyimfit.SimpleModelDescription()
    model.x0.setValue(cx, [cx - 5.0, cx + 5.0])
    model.y0.setValue(cy, [cy - 5.0, cy + 5.0])

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
