#!/usr/bin/env python3
"""
photometric_cut.py

Robust photometric major-axis cut extraction for DESI DR10 science FITS images.

Assumptions / conventions:
- Pixel coordinates are 0-indexed (numpy), (x, y).
- "y-up" visualization convention (origin='lower') for overlays, but extraction
  is purely coordinate-based so orientation is consistent either way.
- mask_fits: nonzero pixels are treated as bad.
- invvar_fits: inverse variance map in science-image pixel grid.
- psf_fits: PSF image (kernel) to convolve the science image before extraction.

Outputs:
- I is in the same linear units as the science image (after optional convolution/background subtraction).
- mu is surface brightness in mag/arcsec^2 using:
    mu = zeropoint - 2.5 log10(I / pixarea_arcsec2)
  (i.e., converting “counts per pixel” to “counts per arcsec^2”).

NEW REPORT OUTPUT:
- If `report_txt_path` is provided, a small .txt report is written containing:
  (1) initial and final center (if refinement was requested they may differ)
  (2) estimated background (and robust sigma), if background estimation was performed
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import numpy as np

from scipy.ndimage import map_coordinates
from scipy.signal import fftconvolve
from scipy.optimize import least_squares


# Reuse your helper utilities directly
from photometric_cut_helpers import (
    load_fits_array,
    pixel_scale_from_header_arcsec_per_pix,
    plot_slit_overlay,
    bn_of_n,
    sersic_mu,
    initial_guesses_mu,
)

ArrayLike = Union[np.ndarray, str, Path]


def _normalize_kernel(psf: np.ndarray) -> np.ndarray:
    psf = np.asarray(psf, float)
    psf = np.where(np.isfinite(psf), psf, 0.0)
    s = psf.sum()
    if s <= 0:
        raise ValueError("PSF kernel sum <= 0; cannot normalize.")
    return psf / s


def _estimate_background(img: np.ndarray, region_mask: np.ndarray) -> Tuple[float, float]:
    """
    Robust-ish background estimate from selected pixels.
    Returns (median, robust_sigma).
    """
    vals = img[region_mask & np.isfinite(img)]
    if vals.size < 50:
        return 0.0, np.nan
    med = float(np.nanmedian(vals))
    mad = float(np.nanmedian(np.abs(vals - med)))
    sigma = 1.4826 * mad if np.isfinite(mad) else np.nan
    return med, sigma


def _build_endcaps_region_mask(
    shape: Tuple[int, int],
    center_xy: Tuple[float, float],
    pa_deg: float,
    length_pix: float,
    width_pix: float,
    endcap_frac: float = 0.18,
) -> np.ndarray:
    """
    Select “ends of slit” pixels (a rectangular endcap at each end) in a raster sense.
    Used for background estimation. This is approximate but works well in practice.
    """
    ny, nx = shape
    x0, y0 = center_xy
    th = np.deg2rad(pa_deg)
    u = np.array([np.cos(th), np.sin(th)])       # along slit
    v = np.array([-np.sin(th), np.cos(th)])      # across slit

    L2 = 0.5 * float(length_pix)
    W = 0.5 * float(width_pix)
    cap = float(endcap_frac) * (2 * L2)
    cap = max(cap, 5.0)

    # Coordinate grid
    yy, xx = np.mgrid[0:ny, 0:nx]
    dx = xx - x0
    dy = yy - y0

    s = dx * u[0] + dy * u[1]   # along-axis coordinate
    t = dx * v[0] + dy * v[1]   # cross-axis coordinate

    in_width = np.abs(t) <= W
    in_ends = (np.abs(s) >= (L2 - cap)) & (np.abs(s) <= L2)
    return in_width & in_ends


def _mu_from_I(I: np.ndarray, zeropoint: float, pixscale_arcsec: float) -> np.ndarray:
    """
    Convert linear flux per pixel to mag/arcsec^2.
    """
    I = np.asarray(I, float)
    pixarea = float(pixscale_arcsec) ** 2
    with np.errstate(divide="ignore", invalid="ignore"):
        mu = zeropoint - 2.5 * np.log10(I / pixarea)
    return mu


def _fmt_center(c: Tuple[float, float]) -> str:
    return f"({c[0]:.3f}, {c[1]:.3f})"


def _write_cut_report_txt(
    filepath: str,
    *,
    sci_fits,
    pa_deg: float,
    refine_center: bool,
    center_init: Tuple[float, float],
    center_final: Tuple[float, float],
    subtract_background: bool,
    background_region: str,
    background_estimate: Optional[float],
    background_sigma: Optional[float],
    refine_metric: Optional[float] = None,
    note: Optional[str] = None,
):
    lines = []
    lines.append("# photometric_cut report")
    lines.append(
    f"timestamp_utc: {datetime.now(timezone.utc).isoformat(timespec='seconds')}")
    lines.append(f"sci_fits: {sci_fits}")
    lines.append(f"pa_deg: {pa_deg:.6f}")
    lines.append("")
    lines.append("[center_refinement]")
    lines.append(f"refine_center_requested: {bool(refine_center)}")
    lines.append(f"center_initial_xy_pix: {_fmt_center(center_init)}")
    lines.append(f"center_final_xy_pix:   {_fmt_center(center_final)}")
    if refine_metric is None:
        lines.append("refine_metric: None")
    else:
        lines.append(f"refine_metric: {float(refine_metric):.8g}")
    lines.append("")
    lines.append("[background]")
    lines.append(f"subtract_background: {bool(subtract_background)}")
    lines.append(f"background_region: {str(background_region)}")
    if background_estimate is None:
        lines.append("background_estimate: None (not computed)")
    else:
        lines.append(f"background_estimate: {float(background_estimate):.8g}   # image units per pixel")
    if background_sigma is None:
        lines.append("background_sigma_robust: None")
    else:
        lines.append(f"background_sigma_robust: {float(background_sigma):.8g}   # robust sigma (MAD*1.4826)")
    if note:
        lines.append("")
        lines.append("[note]")
        lines.append(str(note).strip())

    with open(filepath, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def photometric_cut(
    sci_fits: ArrayLike,
    center_xy: Tuple[float, float],
    pa_deg: float,
    *,
    # geometry
    length_pix: float = 240.0,
    width_pix: float = 5.0,
    oversample: int = 2,
    # optional maps
    mask_fits: Optional[ArrayLike] = None,
    invvar_fits: Optional[ArrayLike] = None,
    psf_fits: Optional[ArrayLike] = None,
    # photometric calibration (only needed for mu; relative ZP is fine)
    zeropoint: float = 22.5,
    pixel_scale_arcsec: Optional[float] = None,
    # background handling
    subtract_background: bool = False,
    background_region: str = "ends",   # "ends" or "none"
    endcap_frac: float = 0.18,
    # interpolation
    interpolation_order: int = 1,
    # center refinement (folding metric)
    refine_center: bool = False,
    refine_kwargs: Optional[dict] = None,
    # debugging / visualization
    show_slit: bool = False,
    slit_overlay_savepath: Optional[str] = None,
    # NEW: report output
    report_txt_path: Optional[str] = None,
    report_note: Optional[str] = None,
) -> Dict[str, np.ndarray]:
    """
    Extract a robust photometric cut along a specified PA through a center.

    Returns a dict with keys:
      - center_xy_initial, center_xy_used, center_refined, center_refine_metric
      - pa_deg, pixel_scale_arcsec
      - s_pix, s_arcsec
      - I, I_err
      - mu, mu_err
      - frac_good
      - background_estimate, background_sigma_robust (if computed)
    """
    sci = sci_fits.data

    if sci.ndim != 2:
        raise ValueError(f"sci must be 2D; got shape {sci.shape}")

    ny, nx = sci.shape

    # Pixel scale
    if pixel_scale_arcsec is None:
        # DESI DR10 cutouts have WCS; use header when sci_fits is a path-like
        try:
            pixel_scale_arcsec = pixel_scale_from_header_arcsec_per_pix(sci_fits)
        except Exception:
            pixel_scale_arcsec = None
    if pixel_scale_arcsec is None:
        raise ValueError("pixel_scale_arcsec could not be inferred; pass pixel_scale_arcsec explicitly.")

    # Optional arrays
    msk = mask_fits if mask_fits is not None else None
    inv = invvar_fits if invvar_fits is not None else None

    if msk is not None and msk.shape != sci.shape:
        raise ValueError(f"mask shape {msk.shape} != sci shape {sci.shape}")
    if inv is not None and inv.shape != sci.shape:
        raise ValueError(f"invvar shape {inv.shape} != sci shape {sci.shape}")

    bad = (msk > 0) if msk is not None else np.zeros_like(sci, dtype=bool)

    # Record initial center (always)
    center_xy_initial = tuple(map(float, center_xy))
    center_xy_used = center_xy_initial
    center_refine_metric: Optional[float] = None

    # Optional PSF convolution (science only; invvar propagation is non-trivial)
    if psf_fits is not None:
        psf = load_fits_array(psf_fits)
        psf = _normalize_kernel(psf)
        # sci = fftconvolve(np.nan_to_num(sci, nan=0.0), psf, mode="same") # DO NOT DO THIS !!!! 
                                                                         # Sci already contains the actual science data, so don't convolve again
        # mask left as-is; invvar left as-is (uncertainties become approximate)

    # Optionally refine center via folding metric (uses your helper)
    if refine_center:
        rk = dict(refine_kwargs or {})
        rk.setdefault("search_mode", "axis")
        rk.setdefault("search_radius_pix", 8)
        rk.setdefault("step_pix", 1)
        rk.setdefault("oversample", oversample)
        rk.setdefault("subtract_background", subtract_background)
        rk.setdefault("background_region", background_region)

        best_center, best_metric = refine_center_by_folding( # What is this even doing ??? We already know that the center of the image is the center of the galaxy
            sci_fits=sci_fits,
            mask_fits=mask_fits,
            pa_deg=pa_deg,
            length_pix=length_pix,
            width_pix=width_pix,
            pixel_scale=pixel_scale_arcsec,
            center_xy_init=center_xy_used,
            zeropoint=zeropoint,
            invvar_fits=invvar_fits,
            psf_fits=psf_fits,
            interpolation_order=interpolation_order,
            **rk,
        )

        center_xy_used = (float(best_center[0]), float(best_center[1]))
        center_refine_metric = float(best_metric) if best_metric is not None else None

    # Background subtraction (estimated from slit endcaps) # Background is already removed, though there is sensor noise, but can't really subtract that since its a gaussian with mean 0
    background_estimate: Optional[float] = None
    background_sigma_robust: Optional[float] = None

    if subtract_background and str(background_region).lower() == "ends":
        region_mask = _build_endcaps_region_mask(
            shape=sci.shape,
            center_xy=center_xy_used,
            pa_deg=pa_deg,
            length_pix=length_pix,
            width_pix=width_pix,
            endcap_frac=endcap_frac,
        )
        region_mask = region_mask & (~bad)

        bg, bg_sigma = _estimate_background(sci, region_mask)
        # Treat non-finite as "not computed"
        if np.isfinite(bg):
            background_estimate = float(bg)
            background_sigma_robust = float(bg_sigma) if np.isfinite(bg_sigma) else None
            sci = sci - background_estimate
        else:
            background_estimate = None
            background_sigma_robust = None

    # Debug overlay
    if show_slit or slit_overlay_savepath:
        plot_slit_overlay(
            sci_fits=sci,  # array (already convolved/bg-subtracted if applied)
            mask_fits=msk,
            center_xy=center_xy_used,
            pa_deg=pa_deg,
            length_pix=length_pix,
            half_width_pix=0.5 * width_pix,
            savepath=slit_overlay_savepath,
            show=bool(show_slit),
        )

    # Build sampling grid
    th = np.deg2rad(pa_deg)
    u = np.array([np.cos(th), np.sin(th)])       # along slit
    v = np.array([-np.sin(th), np.cos(th)])      # across slit

    # Along-slit sample points
    step = 1.0 / max(int(oversample), 1)
    s_pix = np.arange(-0.5 * length_pix, 0.5 * length_pix + 1e-9, step, dtype=float)

    # Across-slit samples
    half_w = 0.5 * float(width_pix)
    t_pix = np.arange(-half_w, half_w + 1e-9, step, dtype=float)

    x0, y0 = center_xy_used

    xs = x0 + s_pix[:, None] * u[0] + t_pix[None, :] * v[0]
    ys = y0 + s_pix[:, None] * u[1] + t_pix[None, :] * v[1]
    coords = np.vstack([ys.ravel(), xs.ravel()])

    # Sample science
    sci_samp = map_coordinates(
        sci,
        coords,
        order=int(interpolation_order),
        mode="constant",
        cval=np.nan,
    ).reshape(s_pix.size, t_pix.size)

    # Sample mask (nearest)
    bad_samp = map_coordinates(
        bad.astype(float),
        coords,
        order=0,
        mode="constant",
        cval=1.0,   # outside image treated as bad
    ).reshape(s_pix.size, t_pix.size) > 0.5

    # Sample invvar (if any)
    if inv is not None:
        inv_samp = map_coordinates(
            inv,
            coords,
            order=0,
            mode="constant",
            cval=0.0,
        ).reshape(s_pix.size, t_pix.size)
        inv_samp = np.where(np.isfinite(inv_samp) & (inv_samp > 0), inv_samp, 0.0)
    else:
        inv_samp = None

    # Combine across width
    good = (~bad_samp) & np.isfinite(sci_samp)

    if inv_samp is None:
        w = good.astype(float)
    else:
        w = inv_samp * good.astype(float)

    wsum = np.sum(w, axis=1)
    frac_good = np.sum(good, axis=1) / float(good.shape[1])

    I = np.full(s_pix.size, np.nan, float)
    I_err = np.full(s_pix.size, np.nan, float)

    ok = wsum > 0
    if np.any(ok):
        I[ok] = np.sum(w[ok] * sci_samp[ok], axis=1) / wsum[ok]

        # Uncertainty
        if inv_samp is not None:
            I_err[ok] = np.sqrt(1.0 / np.maximum(wsum[ok], 1e-30))
        else:
            resid = sci_samp[ok] - I[ok, None]
            mad = np.nanmedian(np.abs(resid), axis=1)
            sigma = 1.4826 * mad
            Ng = np.sum(good[ok], axis=1).astype(float)
            I_err[ok] = sigma / np.sqrt(np.maximum(Ng, 1.0))

    # Convert to arcsec and surface brightness
    s_arcsec = s_pix * float(pixel_scale_arcsec)

    mu = _mu_from_I(I, zeropoint=zeropoint, pixscale_arcsec=float(pixel_scale_arcsec))
    with np.errstate(divide="ignore", invalid="ignore"):
        mu_err = (2.5 / np.log(10.0)) * (I_err / np.maximum(I, 1e-30))

    out: Dict[str, np.ndarray] = {
        "center_xy_initial": np.array(center_xy_initial, float),
        "center_xy_used": np.array(center_xy_used, float),
        "center_refined": bool(refine_center),
        "center_refine_metric": center_refine_metric,
        "pa_deg": float(pa_deg),
        "pixel_scale_arcsec": float(pixel_scale_arcsec),
        "length_pix": float(length_pix),
        "width_pix": float(width_pix),
        "oversample": int(max(1, oversample)),
        "s_pix": s_pix,
        "s_arcsec": s_arcsec,
        "I": I,
        "I_err": I_err,
        "mu": mu,
        "mu_err": mu_err,
        "frac_good": frac_good,
        "background_estimate": background_estimate,
        "background_sigma_robust": background_sigma_robust,
    }

    out["valid"] = np.isfinite(I) & (frac_good > 0)

    # Write report if requested
    if report_txt_path:
        _write_cut_report_txt(
            report_txt_path,
            sci_fits=sci_fits,
            pa_deg=float(pa_deg),
            refine_center=bool(refine_center),
            center_init=center_xy_initial,
            center_final=center_xy_used,
            subtract_background=bool(subtract_background),
            background_region=str(background_region),
            background_estimate=background_estimate,
            background_sigma=background_sigma_robust,
            refine_metric=center_refine_metric,
            note=report_note,
        )

    return out


def photometric_major_axis_cut(*args, **kwargs) -> Dict[str, np.ndarray]:
    """Compatibility wrapper."""
    return photometric_cut(*args, **kwargs)


# =============================================================================
# Center refinement by folding (NEW: self-contained, uses photometric_cut safely)
# =============================================================================

def _fold_mismatch_metric(s_arcsec, I, frac_good):
    """
    Folding mismatch metric: robust fractional spread of samples in bins of |s|.
    Smaller is better. Uses frac_good to downweight poor coverage.
    """
    s = np.asarray(s_arcsec, float)
    I = np.asarray(I, float)
    fg = np.asarray(frac_good, float) if frac_good is not None else np.ones_like(I)

    ok = np.isfinite(s) & np.isfinite(I)
    s, I, fg = s[ok], I[ok], fg[ok]
    if I.size < 10:
        return np.inf

    R = np.abs(s)
    order = np.argsort(R)
    R, I, fg = R[order], I[order], fg[order]

    # Typical bin width
    dR = np.median(np.diff(R)) if R.size > 5 else (np.max(R) - np.min(R)) / max(R.size - 1, 1)
    if not np.isfinite(dR) or dR <= 0:
        dR = (np.max(R) - np.min(R)) / max(R.size - 1, 1)
        if not np.isfinite(dR) or dR <= 0:
            return np.inf

    edges = np.arange(R.min(), R.max() + 1e-9, dR)
    if edges.size < 3:
        return np.inf

    idx = np.digitize(R, edges) - 1
    nb = edges.size - 1

    mismatch = []
    for b in range(nb):
        sel = idx == b
        if np.sum(sel) < 2:
            continue
        Ib = I[sel]
        fgb = fg[sel]

        # If values cross <=0, fractional metric isn't meaningful
        if np.nanmin(Ib) <= 0:
            continue

        m = np.nanmedian(Ib)
        spread = np.nanmedian(np.abs(Ib - m)) / max(m, 1e-20)
        w = np.nanmedian(fgb)
        mismatch.append(spread / max(w, 1e-3))

    if not mismatch:
        return np.inf
    return float(np.nanmedian(mismatch))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Rectangle

def _slit_geometry(center_xy, pa_deg, length_pix, width_pix):
    """Return (corners[4,2], c1[2], c2[2]) for a slit rectangle + centerline endpoints."""
    x0, y0 = center_xy
    th = np.deg2rad(pa_deg)
    u = np.array([np.cos(th), np.sin(th)])       # along slit
    v = np.array([-np.sin(th), np.cos(th)])      # across slit
    L2 = 0.5 * float(length_pix)
    W2 = 0.5 * float(width_pix)

    p1 = np.array([x0, y0]) + (-L2) * u + (-W2) * v
    p2 = np.array([x0, y0]) + (+L2) * u + (-W2) * v
    p3 = np.array([x0, y0]) + (+L2) * u + (+W2) * v
    p4 = np.array([x0, y0]) + (-L2) * u + (+W2) * v
    corners = np.vstack([p1, p2, p3, p4])

    c1 = np.array([x0, y0]) - L2 * u
    c2 = np.array([x0, y0]) + L2 * u
    return corners, c1, c2


def plot_dual_slit_mu_figure(
    *,
    sci_fits,
    mask_fits=None,
    results: dict,
    title: str = "Simultaneous Sérsic fit to Host & Polar cuts",
    contrast_perc=(1, 99.5),
    savepath: str | None = None,
    show: bool = True,
):
    """
    Creates one figure with:
      - Left: image with both slit overlays (top) + info panel (bottom)
      - Middle: Host folded mu(|s|) (top) + Host unfolded mu(s) (bottom)
      - Right: Polar folded mu(|s|) (top) + Polar unfolded mu(s) (bottom)

    Data are gray dots. Fit lines match slit colors.
    """
    # --- palette (as requested) ---
    host_color = "#D62728"   # red
    polar_color = "#6EC6FF"  # light blue
    data_color = "0.35"      # gray dots

    # --- load image/mask ---
    sci = load_fits_array(sci_fits)
    msk = load_fits_array(mask_fits) if mask_fits is not None else None
    bad = (msk > 0) if msk is not None else np.zeros_like(sci, dtype=bool)

    finite_vals = sci[np.isfinite(sci)]
    if finite_vals.size:
        vmin, vmax = np.percentile(finite_vals, contrast_perc)
    else:
        vmin, vmax = 0, 1

    host_cut = results["host"]["cut"]
    polar_cut = results["polar"]["cut"]
    host_fit = results["host"]["fit"]
    polar_fit = results["polar"]["fit"]

    pixscale = float(results["meta"]["pixel_scale_arcsec"])
    zp = float(results["meta"]["zeropoint"])

    # centers used (after refinement if enabled)
    host_center = tuple(map(float, host_cut["center_xy_used"]))
    polar_center = tuple(map(float, polar_cut["center_xy_used"]))
    host_pa = float(host_cut["pa_deg"])
    polar_pa = float(polar_cut["pa_deg"])
    host_len = float(host_cut["length_pix"])
    polar_len = float(polar_cut["length_pix"])
    width_pix = float(host_cut["width_pix"])

    # folded arrays (already |s| binned)
    Rh = np.asarray(results["host"]["R_arcsec"], float)
    muh_fold = np.asarray(results["host"]["mu"], float)

    Rp = np.asarray(results["polar"]["R_arcsec"], float)
    mup_fold = np.asarray(results["polar"]["mu"], float)

    # unfolded arrays from the photometric cut (signed s)
    sh = np.asarray(host_cut["s_arcsec"], float)
    muh_un = np.asarray(host_cut["mu"], float)
    vh = np.asarray(host_cut.get("valid", np.isfinite(muh_un)), bool) & np.isfinite(sh) & np.isfinite(muh_un)

    sp = np.asarray(polar_cut["s_arcsec"], float)
    mup_un = np.asarray(polar_cut["mu"], float)
    vp = np.asarray(polar_cut.get("valid", np.isfinite(mup_un)), bool) & np.isfinite(sp) & np.isfinite(mup_un)

    # figure layout: 2 rows x 3 cols
    fig = plt.figure(figsize=(18, 7))
    gs = fig.add_gridspec(
        2, 3,
        width_ratios=[1.15, 1.0, 1.0],
        height_ratios=[3.2, 1.0],
        wspace=0.30,
        hspace=0.20,
    )

    ax_img = fig.add_subplot(gs[0, 0])
    ax_info = fig.add_subplot(gs[1, 0])

    ax_host_fold = fig.add_subplot(gs[0, 1])
    ax_host_un = fig.add_subplot(gs[1, 1], sharex=None)

    ax_polar_fold = fig.add_subplot(gs[0, 2])
    ax_polar_un = fig.add_subplot(gs[1, 2], sharex=None)

    fig.suptitle(title, y=0.98)

    # =========================
    # Left top: image + slits
    # =========================
    ax_img.imshow(sci, origin="lower", vmin=vmin, vmax=vmax, interpolation="nearest")

    if bad.any():
        bad_alpha = np.zeros_like(sci, dtype=float)
        bad_alpha[bad] = 0.35
        ax_img.imshow(np.ones_like(sci), origin="lower", alpha=bad_alpha, interpolation="nearest")

    # host slit: black outline underlay + red overlay
    host_corners, host_c1, host_c2 = _slit_geometry(host_center, host_pa, host_len, width_pix)

    ax_img.add_patch(Polygon(host_corners, closed=True, fill=False,
                            edgecolor="black", linewidth=3.4, zorder=9))
    ax_img.add_patch(Polygon(host_corners, closed=True, fill=False,
                            edgecolor=host_color, linewidth=2.3, zorder=10))

    ax_img.plot([host_c1[0], host_c2[0]], [host_c1[1], host_c2[1]],
                color="black", linewidth=3.0, zorder=11)
    ax_img.plot([host_c1[0], host_c2[0]], [host_c1[1], host_c2[1]],
                color=host_color, linewidth=2.0, zorder=12)

    # polar slit: black outline underlay + light-blue overlay
    polar_corners, polar_c1, polar_c2 = _slit_geometry(polar_center, polar_pa, polar_len, width_pix)

    ax_img.add_patch(Polygon(polar_corners, closed=True, fill=False,
                            edgecolor="black", linewidth=3.4, zorder=9))
    ax_img.add_patch(Polygon(polar_corners, closed=True, fill=False,
                            edgecolor=polar_color, linewidth=2.3, zorder=10))

    ax_img.plot([polar_c1[0], polar_c2[0]], [polar_c1[1], polar_c2[1]],
                color="black", linewidth=3.0, zorder=11)
    ax_img.plot([polar_c1[0], polar_c2[0]], [polar_c1[1], polar_c2[1]],
                color=polar_color, linewidth=2.0, zorder=12)


    # center markers
    ax_img.plot(host_center[0], host_center[1], marker="x", color=host_color, markersize=9, mew=2)
    ax_img.plot(polar_center[0], polar_center[1], marker="x", color=polar_color, markersize=9, mew=2)

    ax_img.set_title("Host & Polar slits")
    ax_img.set_xlabel("x [pix]")
    ax_img.set_ylabel("y [pix]")
    ax_img.set_aspect("equal")

    # =========================
    # Left bottom: info panel (HOST left, POLAR right)
    # =========================
    ax_info.set_axis_off()
    ax_info.set_xlim(0, 1)
    ax_info.set_ylim(0, 1)

    # background halves (left = host, right = polar)
    ax_info.add_patch(Rectangle((0.00, 0.00), 0.50, 1.00, facecolor=host_color, alpha=0.10, edgecolor="none"))
    ax_info.add_patch(Rectangle((0.50, 0.00), 0.50, 1.00, facecolor=polar_color, alpha=0.16, edgecolor="none"))

    def _fmt_arcsec_pix(x_arcsec):
        x_pix = x_arcsec / pixscale
        return f'{x_arcsec:.2f}" ({x_pix:.1f} px)'

    # --- Host text (left side) ---
    if host_fit.get("success", False):
        host_text = (
            "HOST\n"
            f"n = {host_fit['n']:.2f}\n"
            f"Re = {_fmt_arcsec_pix(host_fit['Re_arcsec'])}\n"
            f"μe = {host_fit['mu_e']:.2f}"
        )
    else:
        host_text = "HOST\nfit: FAILED"

    ax_info.text(
        0.03, 0.92, host_text,
        ha="left", va="top",
        fontsize=12, color="black",
    )

    # --- Polar text (right side) ---
    if polar_fit.get("success", False):
        polar_text = (
            "POLAR\n"
            f"n = {polar_fit['n']:.2f}\n"
            f"Re = {_fmt_arcsec_pix(polar_fit['Re_arcsec'])}\n"
            f"μe = {polar_fit['mu_e']:.2f}"
        )
    else:
        polar_text = "POLAR\nfit: FAILED"

    ax_info.text(
        0.53, 0.92, polar_text,
        ha="left", va="top",
        fontsize=12, color="black",
    )


    # =========================
    # Host folded μ(|s|) + fit
    # =========================
    ax_host_fold.set_title("Host")
    ax_host_fold.set_xlabel(r"Radius $|s|$ [arcsec]")
    ax_host_fold.set_ylabel(r"$\mu$ [mag/arcsec$^2$]")
    ax_host_fold.invert_yaxis()
    ax_host_fold.grid(True, alpha=0.25)

    if Rh.size:
        ax_host_fold.plot(Rh, muh_fold, linestyle="none", marker="o", markersize=4,
                          color=data_color, label="Host data")

    if host_fit.get("success", False) and Rh.size:
        model_h = sersic_mu(Rh, host_fit["n"], host_fit["Re_arcsec"], host_fit["mu_e"])
        ax_host_fold.plot(Rh, model_h, color=host_color, linewidth=2.5, label="Host Sérsic")

    ax_host_fold.legend(loc="best")

    # =========================
    # Host unfolded μ(s) + symmetric model
    # =========================
    ax_host_un.set_xlabel(r"Signed $s$ [arcsec]")
    ax_host_un.set_ylabel(r"$\mu$ [mag/arcsec$^2$]")
    ax_host_un.invert_yaxis()
    ax_host_un.grid(True, alpha=0.25)

    if np.any(vh):
        ax_host_un.plot(sh[vh], muh_un[vh], linestyle="none", marker="o", markersize=3,
                        color=data_color, label="Host unfolded")

        if host_fit.get("success", False):
            s_sorted = np.sort(sh[vh])
            model_un = sersic_mu(np.abs(s_sorted), host_fit["n"], host_fit["Re_arcsec"], host_fit["mu_e"])
            ax_host_un.plot(s_sorted, model_un, color=host_color, linewidth=2.0, label="Host Sérsic (sym)")

    ax_host_un.legend(loc="best")

    # =========================
    # Polar folded μ(|s|) + fit
    # =========================
    ax_polar_fold.set_title("Polar")
    ax_polar_fold.set_xlabel(r"Radius $|s|$ [arcsec]")
    ax_polar_fold.set_ylabel(r"$\mu$ [mag/arcsec$^2$]")
    ax_polar_fold.invert_yaxis()
    ax_polar_fold.grid(True, alpha=0.25)

    if Rp.size:
        ax_polar_fold.plot(Rp, mup_fold, linestyle="none", marker="o", markersize=4,
                           color=data_color, label="Polar data")

    if polar_fit.get("success", False) and Rp.size:
        model_p = sersic_mu(Rp, polar_fit["n"], polar_fit["Re_arcsec"], polar_fit["mu_e"])
        ax_polar_fold.plot(Rp, model_p, color=polar_color, linewidth=2.5, label="Polar Sérsic")

    ax_polar_fold.legend(loc="best")

    # =========================
    # Polar unfolded μ(s) + symmetric model
    # =========================
    ax_polar_un.set_xlabel(r"Signed $s$ [arcsec]")
    ax_polar_un.set_ylabel(r"$\mu$ [mag/arcsec$^2$]")
    ax_polar_un.invert_yaxis()
    ax_polar_un.grid(True, alpha=0.25)

    if np.any(vp):
        ax_polar_un.plot(sp[vp], mup_un[vp], linestyle="none", marker="o", markersize=3,
                         color=data_color, label="Polar unfolded")

        if polar_fit.get("success", False):
            s_sorted = np.sort(sp[vp])
            model_un = sersic_mu(np.abs(s_sorted), polar_fit["n"], polar_fit["Re_arcsec"], polar_fit["mu_e"])
            ax_polar_un.plot(s_sorted, model_un, color=polar_color, linewidth=2.0, label="Polar Sérsic (sym)")

    ax_polar_un.legend(loc="best")

    if savepath:
        fig.savefig(savepath, bbox_inches="tight", dpi=200)
    if show:
        plt.show()

    return fig



def refine_center_by_folding(
    sci_fits,
    mask_fits,
    pa_deg,
    length_pix,
    width_pix,
    pixel_scale,
    center_xy_init,
    zeropoint=22.5,
    oversample=1,
    subtract_background=True,
    background_region="ends",
    # search controls
    search_radius_pix=8,
    step_pix=1,
    search_mode="axis",            # "axis" or "disk"
    search_half_range_pix=None,    # for "axis"
    # pass-through options for scoring cuts
    invvar_fits=None,
    psf_fits=None,
    interpolation_order=1,
):
    """
    Brute-force search for the center minimizing the fold-mismatch metric.

    This function evaluates candidate centers by calling `photometric_cut(...)`
    with `refine_center=False` to avoid recursion.

    Recommendation 4 included:
    - If psf_fits is provided, the *bounding image* used only to compute nx,ny
      and to check candidate bounds is also convolved so bounds/behavior remain
      consistent with the data used for scoring.
      (The scoring itself already uses photometric_cut which will convolve.)
    """
    x0, y0 = tuple(map(float, center_xy_init))

    img = load_fits_array(sci_fits)
    if img is None or img.ndim != 2:
        raise ValueError("sci_fits must be a 2D FITS image or 2D ndarray.")

    # --- Recommendation 4: keep the "reference image" consistent with PSF convolution ---
    if psf_fits is not None:
        psf = load_fits_array(psf_fits)
        if psf is None:
            raise ValueError("psf_fits provided but could not be loaded.")
        psf = _normalize_kernel(psf)
        img = fftconvolve(np.nan_to_num(img, nan=0.0), psf, mode="same")

    ny, nx = img.shape

    best_center = (x0, y0)
    best_metric = np.inf

    def _score_center(cx, cy):
        cut = photometric_cut(
            sci_fits=sci_fits,
            center_xy=(cx, cy),
            pa_deg=pa_deg,
            length_pix=length_pix,
            width_pix=width_pix,
            oversample=oversample,
            mask_fits=mask_fits,
            invvar_fits=invvar_fits,
            psf_fits=psf_fits,
            zeropoint=zeropoint,
            pixel_scale_arcsec=pixel_scale,
            subtract_background=subtract_background,
            background_region=background_region,
            endcap_frac=0.18,  # keep default unless you add as an arg
            interpolation_order=interpolation_order,
            refine_center=False,     # <-- critical: prevent recursion
            refine_kwargs=None,
            show_slit=False,
            slit_overlay_savepath=None,
            report_txt_path=None,
            report_note=None,
        )
        return _fold_mismatch_metric(cut.get("s_arcsec"), cut.get("I"), cut.get("frac_good"))

    mode = str(search_mode).lower().strip()

    if mode == "axis":
        th = np.deg2rad(pa_deg)
        u = np.array([np.cos(th), np.sin(th)])
        rng = float(search_half_range_pix if search_half_range_pix is not None else search_radius_pix)
        steps = np.arange(-int(rng), int(rng) + 1, int(max(1, step_pix)), dtype=float)

        for t in steps:
            cx, cy = x0 + t * u[0], y0 + t * u[1]
            if not (0 <= cx < nx and 0 <= cy < ny):
                continue
            metric = _score_center(cx, cy)
            if metric < best_metric:
                best_metric = metric
                best_center = (cx, cy)

        return best_center, best_metric

    if mode == "disk":
        steps = np.arange(-int(search_radius_pix), int(search_radius_pix) + 1, int(max(1, step_pix)), dtype=int)
        cand = []
        r2 = float(search_radius_pix) ** 2
        for dy in steps:
            for dx in steps:
                if dx * dx + dy * dy <= r2:
                    cand.append((dx, dy))

        for dx, dy in cand:
            cx, cy = x0 + dx, y0 + dy
            if not (0 <= cx < nx and 0 <= cy < ny):
                continue
            metric = _score_center(cx, cy)
            if metric < best_metric:
                best_metric = metric
                best_center = (cx, cy)

        return best_center, best_metric

    raise ValueError(f"Unknown search_mode={search_mode!r}; expected 'axis' or 'disk'.")

def fold_cut_to_radial_profile(cut, min_frac_good=0.5):
    """
    Take a cut dict from photometric_cut() and produce a folded radial profile:
      R = |s| (arcsec), mu(R) = median of +/- sides in bins.

    Returns: R, mu, mu_err, mask_used (boolean mask on the *original* samples)
    """
    s = np.asarray(cut["s_arcsec"], float)
    mu = np.asarray(cut["mu"], float)
    mu_err = np.asarray(cut.get("mu_err", np.full_like(mu, np.nan)), float)
    frac_good = np.asarray(cut.get("frac_good", np.ones_like(mu)), float)
    valid = np.asarray(cut.get("valid", np.isfinite(mu)), bool)

    base_mask = valid & np.isfinite(s) & np.isfinite(mu) & (frac_good >= float(min_frac_good))
    if np.sum(base_mask) < 10:
        return np.array([]), np.array([]), np.array([]), base_mask

    s0 = s[base_mask]
    mu0 = mu[base_mask]
    mu_e0 = mu_err[base_mask] if mu_err is not None else np.full_like(mu0, np.nan)

    R = np.abs(s0)
    order = np.argsort(R)
    R, mu0, mu_e0 = R[order], mu0[order], mu_e0[order]

    # Bin in R with an automatic bin width
    dR = np.median(np.diff(R)) if R.size > 10 else (R.max() - R.min()) / max(R.size - 1, 1)
    dR = float(dR) if np.isfinite(dR) and dR > 0 else 0.5

    edges = np.arange(R.min(), R.max() + 1e-9, dR)
    if edges.size < 3:
        return np.array([]), np.array([]), np.array([]), base_mask

    idx = np.digitize(R, edges) - 1
    nb = edges.size - 1

    Rb, mub, mueb = [], [], []
    for b in range(nb):
        sel = idx == b
        if np.sum(sel) < 2:
            continue
        Rb.append(np.nanmedian(R[sel]))
        mub.append(np.nanmedian(mu0[sel]))
        # conservative: take median error in bin
        mueb.append(np.nanmedian(mu_e0[sel]) if np.any(np.isfinite(mu_e0[sel])) else np.nan)

    return np.asarray(Rb), np.asarray(mub), np.asarray(mueb), base_mask

def fit_sersic_mu(
    R_arcsec,
    mu,
    mu_err=None,
    *,
    R_min_arcsec=0.0,     # set >0 for polar to avoid host-dominated core
    R_max_arcsec=np.inf,
    n_bounds=(0.3, 8.0),
    Re_bounds=(1e-3, 1e4),
    mu_e_bounds=(5.0, 40.0),
):
    R = np.asarray(R_arcsec, float)
    mu = np.asarray(mu, float)
    mu_err = np.asarray(mu_err, float) if mu_err is not None else np.full_like(mu, np.nan)

    mask = np.isfinite(R) & np.isfinite(mu) & (R >= float(R_min_arcsec)) & (R <= float(R_max_arcsec))
    if np.sum(mask) < 8:
        return {"success": False, "reason": "too_few_points", "mask": mask}

    Rf, muf, ef = R[mask], mu[mask], mu_err[mask]
    w = np.where(np.isfinite(ef) & (ef > 0), 1.0 / ef, 1.0)

    n0, Re0, mu_e0 = initial_guesses_mu(Rf, muf)

    lb = [float(n_bounds[0]), float(Re_bounds[0]), float(mu_e_bounds[0])]
    ub = [float(n_bounds[1]), float(Re_bounds[1]), float(mu_e_bounds[1])]
    x0 = [float(np.clip(n0, lb[0], ub[0])),
          float(np.clip(Re0, lb[1], ub[1])),
          float(np.clip(mu_e0, lb[2], ub[2]))]

    def resid(p):
        n, Re, mu_e = p
        model = sersic_mu(Rf, n, Re, mu_e)
        return (muf - model) * w

    res = least_squares(resid, x0=x0, bounds=(lb, ub), method="trf")

    n, Re, mu_e = res.x
    return {
        "success": bool(res.success),
        "n": float(n),
        "Re_arcsec": float(Re),
        "mu_e": float(mu_e),
        "cost": float(res.cost),
        "mask": mask,
        "nfev": int(res.nfev),
        "message": str(res.message),
    }

def dual_component_slits_and_sersic(
    *,
    sci_fits,
    host_center,
    host_pa,
    polar_center,
    polar_pa,
    mask_fits=None,
    invvar_fits=None,
    psf_fits=None,
    zeropoint=22.5,
    pixel_scale_arcsec=None,
    # slit geometry
    host_length_pix=300,
    polar_length_pix=400,
    width_pix=7.0,
    oversample=2,
    # fitting controls
    host_R_min_arcsec=0.0,
    polar_R_min_arcsec=None,          # if set, overrides scaling rule
    polar_R_min_hostRe_factor=3.5,    # default: exclude inner 3.5 * host Re
    min_frac_good=0.5,
    # background (you defaulted off; keep explicit)
    subtract_background=False,
    background_region="ends",
    # center refine options (optional)
    refine_center=False,
    refine_kwargs=None,
    # write reports
    report_prefix=None,   # e.g. f"./reports/{galaxy}_r"
):
    if pixel_scale_arcsec is None:
        pixel_scale_arcsec = pixel_scale_from_header_arcsec_per_pix(sci_fits)

    host_report = f"{report_prefix}_host.txt" if report_prefix else None
    polar_report = f"{report_prefix}_polar.txt" if report_prefix else None

    host_cut = photometric_cut(
        sci_fits=sci_fits,
        center_xy=host_center,
        pa_deg=host_pa,
        length_pix=host_length_pix,
        width_pix=width_pix,
        oversample=oversample,
        mask_fits=mask_fits,
        invvar_fits=invvar_fits,
        psf_fits=psf_fits,
        zeropoint=zeropoint,
        pixel_scale_arcsec=pixel_scale_arcsec,
        subtract_background=subtract_background,
        background_region=background_region,
        refine_center=refine_center,
        refine_kwargs=refine_kwargs,
        report_txt_path=host_report,
        report_note="Host slit",
    )

    polar_cut = photometric_cut(
        sci_fits=sci_fits,
        center_xy=polar_center,
        pa_deg=polar_pa,
        length_pix=polar_length_pix,
        width_pix=width_pix,
        oversample=oversample,
        mask_fits=mask_fits,
        invvar_fits=invvar_fits,
        psf_fits=psf_fits,
        zeropoint=zeropoint,
        pixel_scale_arcsec=pixel_scale_arcsec,
        subtract_background=subtract_background,
        background_region=background_region,
        refine_center=refine_center,
        refine_kwargs=refine_kwargs,
        report_txt_path=polar_report,
        report_note="Polar slit",
    )

    # Fold -> radial profiles
    Rh, muh, muh_err, host_mask_used = fold_cut_to_radial_profile(host_cut, min_frac_good=min_frac_good) # I gotta figure out what this does
    Rp, mup, mup_err, polar_mask_used = fold_cut_to_radial_profile(polar_cut, min_frac_good=min_frac_good)

    # Fit host Sérsic
    host_fit = fit_sersic_mu(Rh, muh, muh_err, R_min_arcsec=host_R_min_arcsec) # While I think this is a good idea, its also verging on
    # just "fitting before the fitting". It is something I trid but it doesn't really work well, unless this is just a 1d fit

    # Decide polar inner cutoff
    polar_Rmin_used = None
    if polar_R_min_arcsec is not None:
        polar_Rmin_used = float(polar_R_min_arcsec)
    else:
        if host_fit.get("success", False):
            f = float(polar_R_min_hostRe_factor)
            polar_Rmin_used = f * float(host_fit["Re_arcsec"])
        else:
            polar_Rmin_used = 0.0  # fallback if host fit failed

    # Fit polar Sérsic
    polar_fit = fit_sersic_mu(Rp, mup, mup_err, R_min_arcsec=polar_Rmin_used)

    # --------------------------------------------------
    # Clean summary results (human-facing)
    # --------------------------------------------------
    summary = {
        "host": {},
        "polar": {},
    }

    hf = host_fit
    if hf.get("success", False):
        summary["host"] = {
            "success": True,
            "n": float(hf["n"]),
            "Re_arcsec": float(hf["Re_arcsec"]),
            "mu_e": float(hf["mu_e"]),
        }
    else:
        summary["host"] = {"success": False}

    pf = polar_fit
    if pf.get("success", False):
        summary["polar"] = {
            "success": True,
            "n": float(pf["n"]),
            "Re_arcsec": float(pf["Re_arcsec"]),
            "mu_e": float(pf["mu_e"]),
            "R_min_arcsec_used": float(polar_Rmin_used),
        }
    else:
        summary["polar"] = {"success": False}


    return {
        "host": {
            "cut": host_cut,
            "R_arcsec": Rh,
            "mu": muh,
            "mu_err": muh_err,
            "fit": host_fit,
        },
        "polar": {
            "cut": polar_cut,
            "R_arcsec": Rp,
            "mu": mup,
            "mu_err": mup_err,
            "fit": polar_fit,
        },
        "results": summary,
        "meta": {
            "pixel_scale_arcsec": float(pixel_scale_arcsec),
            "zeropoint": float(zeropoint),
        },
    }

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

def _slit_geometry(center_xy, pa_deg, length_pix, width_pix):
    """Return (corners[4,2], c1[2], c2[2]) for a slit rectangle + centerline endpoints."""
    x0, y0 = center_xy
    th = np.deg2rad(pa_deg)
    u = np.array([np.cos(th), np.sin(th)])       # along slit
    v = np.array([-np.sin(th), np.cos(th)])      # across slit

    L2 = 0.5 * float(length_pix)
    W2 = 0.5 * float(width_pix)

    p1 = np.array([x0, y0]) + (-L2) * u + (-W2) * v
    p2 = np.array([x0, y0]) + (+L2) * u + (-W2) * v
    p3 = np.array([x0, y0]) + (+L2) * u + (+W2) * v
    p4 = np.array([x0, y0]) + (-L2) * u + (+W2) * v
    corners = np.vstack([p1, p2, p3, p4])

    c1 = np.array([x0, y0]) - L2 * u
    c2 = np.array([x0, y0]) + L2 * u
    return corners, c1, c2


def plot_dual_slit_and_mu_profiles(
    *,
    sci_fits,
    mask_fits=None,
    results: dict,
    title: str = "Simultaneous Sérsic fit to Host & Polar cuts",
    contrast_perc=(1, 99.5),
    savepath: str | None = None,
    show: bool = True,
):
    """
    Single figure:
      [0] image + host slit + polar slit overlay
      [1] host mu(|s|) + host Sérsic mu(R)
      [2] polar mu(|s|) + polar Sérsic mu(R)

    Expects `results` from dual_component_slits_and_sersic(...).
    Uses folded radial arrays in results: results[comp]["R_arcsec"], ["mu"].
    """
    # --- image + mask ---
    sci = load_fits_array(sci_fits)
    msk = load_fits_array(mask_fits) if mask_fits is not None else None
    bad = (msk > 0) if msk is not None else np.zeros_like(sci, dtype=bool)

    finite_vals = sci[np.isfinite(sci)]
    if finite_vals.size:
        vmin, vmax = np.percentile(finite_vals, contrast_perc)
    else:
        vmin, vmax = 0, 1

    host_cut = results["host"]["cut"]
    polar_cut = results["polar"]["cut"]

    # centers used in the *cuts* (after optional refinement)
    host_center = tuple(map(float, host_cut["center_xy_used"]))
    polar_center = tuple(map(float, polar_cut["center_xy_used"]))

    host_pa = float(host_cut["pa_deg"])
    polar_pa = float(polar_cut["pa_deg"])

    host_len = float(host_cut["length_pix"])
    polar_len = float(polar_cut["length_pix"])
    width_pix = float(host_cut["width_pix"])  # should match polar in your call

    # --- make figure ---
    fig = plt.figure(figsize=(18, 6))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.1, 1.0, 1.0], wspace=0.30)

    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])
    ax2 = fig.add_subplot(gs[0, 2])

    fig.suptitle(title, y=0.98)

    # =========================
    # Panel 0: image + both slits
    # =========================
    ax0.imshow(sci, origin="lower", vmin=vmin, vmax=vmax, interpolation="nearest")

    if bad.any():
        bad_alpha = np.zeros_like(sci, dtype=float)
        bad_alpha[bad] = 0.35
        ax0.imshow(np.ones_like(sci), origin="lower", alpha=bad_alpha, interpolation="nearest")

    # host slit overlay
    host_corners, host_c1, host_c2 = _slit_geometry(host_center, host_pa, host_len, width_pix)
    ax0.add_patch(Polygon(host_corners, closed=True, fill=False, linewidth=2.2, label="Host slit"))
    ax0.plot([host_c1[0], host_c2[0]], [host_c1[1], host_c2[1]], linewidth=2.2)

    # polar slit overlay
    polar_corners, polar_c1, polar_c2 = _slit_geometry(polar_center, polar_pa, polar_len, width_pix)
    ax0.add_patch(Polygon(polar_corners, closed=True, fill=False, linewidth=2.2, label="Polar slit"))
    ax0.plot([polar_c1[0], polar_c2[0]], [polar_c1[1], polar_c2[1]], linewidth=2.2)

    # centers
    ax0.plot(host_center[0], host_center[1], marker="x", markersize=8, linewidth=2.0)
    ax0.plot(polar_center[0], polar_center[1], marker="x", markersize=8, linewidth=2.0)

    ax0.set_title("Host & Polar slits")
    ax0.set_xlabel("x [pix]")
    ax0.set_ylabel("y [pix]")
    ax0.set_aspect("equal")

    # =========================
    # Panel 1: host mu(|s|) + model
    # =========================
    Rh = np.asarray(results["host"]["R_arcsec"], float)
    muh = np.asarray(results["host"]["mu"], float)
    hf = results["host"]["fit"]

    ax1.set_title("Host")
    ax1.set_xlabel(r"Radius $|s|$ [arcsec]")
    ax1.set_ylabel(r"$\mu$ [mag/arcsec$^2$]")
    ax1.invert_yaxis()

    if Rh.size:
        ax1.plot(Rh, muh, marker="o", linestyle="none", label="Host data")

    if hf.get("success", False) and Rh.size:
        model_h = sersic_mu(Rh, hf["n"], hf["Re_arcsec"], hf["mu_e"])
        ax1.plot(Rh, model_h, linestyle="-", label="Host Sérsic")

        ax1.text(
            0.98, 0.05,
            f"n={hf['n']:.2f}, Re={hf['Re_arcsec']:.2f}\" \nμe={hf['mu_e']:.2f}",
            transform=ax1.transAxes,
            ha="right", va="bottom",
            bbox=dict(boxstyle="round", alpha=0.8),
        )
    else:
        ax1.text(0.5, 0.5, "Host fit failed", transform=ax1.transAxes, ha="center")

    ax1.grid(True, alpha=0.25)
    ax1.legend(loc="best")

    # =========================
    # Panel 2: polar mu(|s|) + model
    # =========================
    Rp = np.asarray(results["polar"]["R_arcsec"], float)
    mup = np.asarray(results["polar"]["mu"], float)
    pf = results["polar"]["fit"]

    ax2.set_title("Polar")
    ax2.set_xlabel(r"Radius $|s|$ [arcsec]")
    ax2.set_ylabel(r"$\mu$ [mag/arcsec$^2$]")
    ax2.invert_yaxis()

    if Rp.size:
        ax2.plot(Rp, mup, marker="o", linestyle="none", label="Polar data")

    if pf.get("success", False) and Rp.size:
        model_p = sersic_mu(Rp, pf["n"], pf["Re_arcsec"], pf["mu_e"])
        ax2.plot(Rp, model_p, linestyle="-", label="Polar Sérsic")

        ax2.text(
            0.98, 0.05,
            f"n={pf['n']:.2f}, Re={pf['Re_arcsec']:.2f}\" \nμe={pf['mu_e']:.2f}",
            transform=ax2.transAxes,
            ha="right", va="bottom",
            bbox=dict(boxstyle="round", alpha=0.8),
        )
    else:
        ax2.text(0.5, 0.5, "Polar fit failed", transform=ax2.transAxes, ha="center")

    ax2.grid(True, alpha=0.25)
    ax2.legend(loc="best")

    if savepath:
        fig.savefig(savepath, bbox_inches="tight", dpi=200)
    if show:
        plt.show()

    return fig




def main():
    import table_info
    import glob
    import os
    from photometric_cut_helpers import ellipse_fit
    from sersic_init_conf import get_galaxy_files
    from table_info import get_galaxy_info
    import numpy as np
    import matplotlib.pyplot as plt


    table_info.set_directory()
    GalaxyDirectories = glob.glob("./GalaxyFiles/*")

    filenames = [os.path.basename(f) for f in GalaxyDirectories]
    test_galaxy = filenames[0]

    host_ellipse_results, polar_ellipse_results = ellipse_fit(test_galaxy)
    # print("Host results:", host_ellipse_results)
    # print("Polar results:", polar_ellipse_results)

    TestFiles = get_galaxy_files(test_galaxy, fltr="r")
    sci_fits = TestFiles["science"]
    mask_fits = TestFiles["mask"]
    invvar_fits = TestFiles.get("invvar", None)
    psf_fits = TestFiles.get("psf", None)

    psg_type = get_galaxy_info(test_galaxy)["psg_type"]

    # Calibration
    pixel_scale = pixel_scale_from_header_arcsec_per_pix(sci_fits)



    print(
        f"\n[Calibration]\npixel_scale = {pixel_scale:.4f} arcsec/pix, "
    )
        # ----------------------------
    # Dual-component slit + Sérsic test
    # ----------------------------
    report_prefix = f"./dualcut_{test_galaxy}_r"

    results = dual_component_slits_and_sersic(
        sci_fits=sci_fits,
        host_ellipse_results=host_ellipse_results,
        polar_ellipse_results=polar_ellipse_results,
        mask_fits=mask_fits,
        invvar_fits=invvar_fits,
        psf_fits=psf_fits,
        zeropoint=22.5,
        pixel_scale_arcsec=pixel_scale,
        host_length_pix=300,
        polar_length_pix=450,
        width_pix=7.0,
        oversample=2,
        host_R_min_arcsec=0.0,
        polar_R_min_arcsec=None,          # if set, overrides scaling rule
        polar_R_min_hostRe_factor=3.5,    # default: exclude inner 3.5 * host Re
        min_frac_good=0.5,
        subtract_background=False,
        background_region="ends",
        refine_center=False,      # keep False for first test; turn on after you like the behavior
        refine_kwargs=None,
        report_prefix=report_prefix,
    )

    plot_dual_slit_mu_figure(
        sci_fits=sci_fits,
        mask_fits=mask_fits,
        results=results,
        title="Simultaneous Sérsic fit to Host & Polar cuts",
        savepath=f"./joint_two_cuts_{test_galaxy}.png",
        show=True,
    )



    summary = results["results"]
    print("\n[Dual-component Sérsic results]")

    if summary["host"]["success"]:
        h = summary["host"]
        print(f"Host:  n={h['n']:.2f}, Re={h['Re_arcsec']:.2f}\", μe={h['mu_e']:.2f}")
    else:
        print("Host:  FAILED")

    if summary["polar"]["success"]:
        p = summary["polar"]
        print(
            f"Polar: n={p['n']:.2f}, Re={p['Re_arcsec']:.2f}\", μe={p['mu_e']:.2f} "
            f"(R > {p['R_min_arcsec_used']:.2f}\")"
        )
    else:
        print("Polar: FAILED")

    print("\n")


    return
    


if __name__ == "__main__":
    main()
