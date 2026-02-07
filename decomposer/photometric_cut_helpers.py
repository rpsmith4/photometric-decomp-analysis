#!/usr/bin/env python3


# =========================
# Imports
# =========================
import ast
import math
import os
import glob
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Circle

from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales

from scipy.ndimage import map_coordinates
from scipy.optimize import least_squares
from scipy.signal import fftconvolve

from table_info import get_galaxy_info
from initial_parameterization import get_galaxy_files


# =====================================================================
# Utilities & helpers
# =====================================================================
def load_fits_array(arr_or_path):
    """Load FITS or ndarray → float array (or None)."""
    if arr_or_path is None:
        return None
    if isinstance(arr_or_path, (str, bytes, Path)):
        return fits.getdata(arr_or_path).astype(float)
    if isinstance(arr_or_path, np.ndarray):
        return arr_or_path.astype(float)
    try:
        return arr_or_path[0].data.astype(float)
    except Exception:
        raise TypeError("Unsupported FITS/array input type")


def _get_fits_header(hdu_or_path):
    """Get a primary header from a FITS path or HDU-like object."""
    if isinstance(hdu_or_path, (str, bytes, Path)):
        return fits.getheader(hdu_or_path)
    try:
        return hdu_or_path[0].header
    except Exception as e:
        raise TypeError("sci_fits must be a FITS filepath or HDU-like with header") from e


def pixel_scale_from_header_arcsec_per_pix(sci_fits, return_axes=False):
    """Arcsec/pixel (geometric mean) from WCS; optionally the axes scales."""
    hdr = _get_fits_header(sci_fits)
    w = WCS(hdr, naxis=2)  # force 2D celestial WCS
    scales_deg = proj_plane_pixel_scales(w)  # deg/pix
    scales_as = np.abs(scales_deg[:2]) * 3600.0
    geo = float(np.sqrt(scales_as[0] * scales_as[1]))
    return (geo, (float(scales_as[0]), float(scales_as[1]))) if return_axes else geo


def ellipse_fit(name: str, base: str = "./EllipseFitResults", to_pixels: float = 2.0):
    """
    Load 'host' & 'polar' ellipse rows for a given `file` from CSVs in `base`.
    Returns two dicts with center/PA/axes/μ etc. (see keys below).
    """
    base_path = Path(base)
    if not base_path.is_dir():
        raise FileNotFoundError(f"Directory not found: {base_path.resolve()}")

    candidates = []
    for csv_path in sorted(base_path.glob("*.csv")):
        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            raise RuntimeError(f"Failed to read {csv_path}: {e}") from e
        if not {"file", "label"}.issubset(df.columns):
            continue
        subset = df[df["file"] == name]
        if len(subset):
            candidates.append((csv_path, subset))

    if not candidates:
        raise FileNotFoundError(f"No rows with file == {name!r} in {base_path}")
    if len(candidates) > 1:
        files = ", ".join(str(p) for p, _ in candidates)
        raise ValueError(
            f"Rows for {name!r} found in multiple CSVs: {files}. Expected exactly one CSV."
        )

    csv_path, subset = candidates[0]
    if len(subset) != 2:
        raise ValueError(
            f"Expected exactly 2 rows for file == {name!r} in {csv_path}, found {len(subset)}."
        )

    subset = subset.copy()
    subset["label"] = subset["label"].astype(str).str.lower()
    if set(subset["label"]) != {"host", "polar"}:
        raise ValueError(f"Expected labels 'host' and 'polar' for {name!r} in {csv_path}.")

    def _parse(v):
        if isinstance(v, str):
            s = v.strip()
            if s and (s[0] in "[{(" or s.lower() in {"true", "false", "none"}):
                try:
                    return ast.literal_eval(s)
                except Exception:
                    return v
        return v

    def _get(row, col):
        if col not in row.index:
            raise KeyError(f"Required column {col!r} missing in {csv_path}")
        return _parse(row[col])

    def _build(row, is_host: bool):
        xc = _get(row, "x_center")
        yc = _get(row, "y_center")
        angle = _get(row, "angle")
        contour_val = _get(row, "contour")
        a = _get(row, "semi_major")
        b = _get(row, "semi_minor")
        q = _get(row, "axis_ratio")
        try:
            xc, yc = float(xc) * to_pixels, float(yc) * to_pixels
        except Exception:
            pass
        try:
            q_num = float(q)
            ell = None if math.isnan(q_num) else 1.0 - q_num
        except Exception:
            q_num, ell = q, None

        out = {
            "center": (xc, yc),
            "PA": angle,
            "semi_major_axis": a,
            "semi_minor_axis": b,
            "axis_ratio": q,
            "ellipticity": ell,
        }
        try:
            out["μ"] = contour_val
        except Exception:
            out["isophote"] = contour_val

        if is_host:
            out["PA_diff"] = _get(row, "pa_diff")

        return out

    host_row = subset[subset["label"] == "host"].iloc[0]
    polar_row = subset[subset["label"] == "polar"].iloc[0]
    host = _build(host_row, is_host=True)
    polar = _build(polar_row, is_host=False)
    return host, polar
# =====================================================================
# Sérsic helpers
# =====================================================================
def bn_of_n(n):
    n = np.asarray(n, dtype=float)
    return 2 * n - 1 / 3 + 4 / (405 * n) + 46 / (25515 * n ** 2)


def sersic_mu(R, n, Re, mu_e):
    bn = bn_of_n(n)
    R = np.asarray(R, dtype=float)
    Re = max(float(Re), 1e-12)
    return mu_e + 1.085736 * bn * (np.power(np.clip(R / Re, 1.0e-12, None), 1.0 / n) - 1.0)


def initial_guesses_mu(R, mu):
    R = np.asarray(R, float)
    mu = np.asarray(mu, float)
    gd = np.isfinite(R) & np.isfinite(mu)
    if not np.any(gd):
        return 2.0, 0.5, (np.nanmedian(mu) if np.isfinite(np.nanmedian(mu)) else 25.0)

    Rg, mug = R[gd], mu[gd]
    if Rg.size < 10:
        Re0 = (np.nanmedian(Rg) if Rg.size else 1.0)
        mu_e0 = (np.nanmedian(mug) if mug.size else 25.0)
        return 2.0, float(Re0), float(mu_e0)

    Re0 = 0.5 * np.nanmax(Rg)
    band = (Rg > 0.8 * Re0) & (Rg < 1.2 * Re0)
    mu_e0 = np.nanmedian(mug[band]) if np.any(band) else np.nanmedian(mug[-max(5, Rg.size // 10) :])
    Re0 = max(Re0, np.nanpercentile(Rg, 30))
    return 2.0, float(Re0), float(mu_e0)


def sersic_I(R, n, Re, Ie):
    bn = bn_of_n(n)
    R = np.asarray(R, float)
    Re = max(float(Re), 1e-12)
    return Ie * np.exp(-bn * (np.power(np.clip(R / Re, 1.0e-12, None), 1.0 / n) - 1.0))



# =====================================================================
# Slit overlay (y up)
# =====================================================================
def plot_slit_overlay(
    sci_fits,
    mask_fits=None,
    center_xy=None,
    pa_deg=0.0,
    length_pix=200,
    half_width_pix=2.0,
    contrast_perc=(1, 99.5),
    interpolation="nearest",
    slit_color='black',
    centerline_color='red',
    center_color='cyan',
    center_mask_radius_pix=None,  # NEW: draw central red circle if provided
    savepath=None,
    show=True,
    return_artists=False
):
    """Display a masked science image with a slit overlay (y-up geometry)."""
    sci = load_fits_array(sci_fits)
    if mask_fits is not None:
        msk = load_fits_array(mask_fits)
        bad = (msk > 0)
    else:
        bad = np.zeros_like(sci, dtype=bool)

    ny, nx = sci.shape
    if center_xy is None:
        x0, y0 = (nx - 1) / 2.0, (ny - 1) / 2.0
    else:
        x0, y0 = center_xy

    finite_vals = sci[np.isfinite(sci)]
    vmin, vmax = np.percentile(finite_vals, contrast_perc) if finite_vals.size > 0 else (0, 1)

    theta = np.deg2rad(pa_deg)
    u = np.array([np.cos(theta), np.sin(theta)])
    v = np.array([-np.sin(theta), np.cos(theta)])

    L2 = 0.5 * float(length_pix)
    W = float(half_width_pix)

    p1 = np.array([x0, y0]) + (-L2) * u + (-W) * v
    p2 = np.array([x0, y0]) + (+L2) * u + (-W) * v
    p3 = np.array([x0, y0]) + (+L2) * u + (+W) * v
    p4 = np.array([x0, y0]) + (-L2) * u + (+W) * v
    corners = np.vstack([p1, p2, p3, p4])

    c1 = np.array([x0, y0]) - L2 * u
    c2 = np.array([x0, y0]) + L2 * u

    fig, ax = plt.subplots(figsize=(6.5, 6))
    ax.imshow(sci, origin="lower", vmin=vmin, vmax=vmax, interpolation=interpolation)

    if bad.any():
        bad_alpha = np.zeros_like(sci, dtype=float)
        bad_alpha[bad] = 0.35
        ax.imshow(np.ones_like(sci), origin="lower", alpha=bad_alpha, interpolation="nearest")

    poly = Polygon(corners, closed=True, fill=False, color=slit_color, linewidth=1.6, zorder=5)
    ax.add_patch(poly)
    ax.plot([c1[0], c2[0]], [c1[1], c2[1]], color=centerline_color, linewidth=1.6, zorder=6)
    ax.plot(x0, y0, marker='x', color=center_color, markersize=6, linewidth=1.2, zorder=7)

    # --- NEW: draw central mask circle (thin red border, semi-transparent interior), on top
    if center_mask_radius_pix is not None and center_mask_radius_pix > 0:
        circ = Circle(
            (x0, y0),
            radius=float(center_mask_radius_pix),
            edgecolor="red",
            facecolor=(1, 0, 0, 0.18),
            linewidth=1.0,
            zorder=20,
            fill=True
        )
        ax.add_patch(circ)

    ax.set_xlim(-0.5, nx - 0.5)
    ax.set_ylim(-0.5, ny - 0.5)
    ax.set_aspect('equal')
    ax.set_xlabel("x [pix]")
    ax.set_ylabel("y [pix]")
    ax.set_title(f"Slit overlay: center=({x0:.1f},{y0:.1f}), PA={pa_deg:.1f}°, L={length_pix}px")

    if savepath:
        fig.savefig(savepath, bbox_inches='tight', dpi=200)
    if show:
        plt.show()
    if return_artists:
        return fig, ax, {"polygon": poly}


# # =====================================================================
# # Cut extraction (y-up) with optional invvar, center refinement, and central *fit* mask
# # =====================================================================
# def _fold_mismatch_metric(s_arcsec, I, frac_good):
#     """
#     Compute a folding mismatch metric: median absolute fractional difference
#     between I(+R) and I(-R) over overlapping radii, weighting by frac_good.
#     Smaller is better.
#     """
#     s = np.asarray(s_arcsec, float)
#     I = np.asarray(I, float)
#     fg = np.asarray(frac_good, float) if frac_good is not None else np.ones_like(I)

#     ok = np.isfinite(s) & np.isfinite(I)
#     s, I, fg = s[ok], I[ok], fg[ok]
#     if I.size < 10:
#         return np.inf

#     # Bin by |s| to pair ±R samples
#     R = np.abs(s)
#     order = np.argsort(R)
#     R, I, fg = R[order], I[order], fg[order]

#     # Merge near-duplicate radii
#     dR = np.median(np.diff(R)) if R.size > 5 else (np.max(R) - np.min(R)) / max(R.size - 1, 1)
#     if not np.isfinite(dR) or dR <= 0:
#         dR = (np.max(R) - np.min(R)) / max(R.size - 1, 1)
#         if not np.isfinite(dR) or dR <= 0:
#             return np.inf

#     edges = np.arange(R.min(), R.max() + 1e-9, dR)
#     if edges.size < 3:
#         return np.inf

#     idx = np.digitize(R, edges) - 1
#     nb = edges.size - 1
#     mismatch = []
#     for b in range(nb):
#         sel = idx == b
#         if np.sum(sel) < 2:
#             continue
#         Ib = I[sel]
#         fgb = fg[sel]
#         if np.nanmin(Ib) <= 0:
#             continue
#         # fractional spread in the bin (robust)
#         m = np.nanmedian(Ib)
#         spread = np.nanmedian(np.abs(Ib - m)) / max(m, 1e-20)
#         w = np.nanmedian(fgb)
#         mismatch.append(spread / max(w, 1e-3))
#     if not mismatch:
#         return np.inf
#     return float(np.nanmedian(mismatch))

