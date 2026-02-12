"""
Purpose: Generate initial guess parameters for photometric decomposition.

- Reads .fits files (science, masks, psf/weight optional)
- Uses improved ellipse_fit(name) that returns (host, polar) dicts
- Builds a 2x Sérsic imfit model file with reasonable bounds

Assumptions:
- ellipse_fit(name) is defined elsewhere per the user-supplied implementation
- analyze_two_sersic_cuts_joint(...) exists and accepts the arguments used below
"""

from pathlib import Path
from typing import Dict, Tuple, Optional

import os
import textwrap

# --- file discovery ---------------------------------------------------------

def get_galaxy_files(name: str, base: str = "./", fltr: str = "r") -> Dict[str, Optional[str]]:
    """
    Return paths to key .fits files for galaxy `name` in base.

    Returns dict with keys: science, invvar, psf, mask (some may be None).
    Raises if `science` is missing.
    """
    d = Path(base) / name
    if not d.is_dir():
        raise FileNotFoundError(f"Directory not found: {d.resolve()}")

    def first_match(patterns):
        for pat in patterns:
            m = sorted(d.glob(pat))
            if m:
                return str(m[0])
        return None

    files = {
        "science": first_match([f"image*{fltr}*interp.fits"]),
        "invvar":  first_match([f"*image*{fltr}*invvar.fits"]),
        "psf":     first_match([f"psf_patched*{fltr}.fits"]),
        "mask":    first_match(["*mask_all.fits"]),
    }

    if files["science"] is None:
        raise FileNotFoundError(f"Missing required science image in {d}")

    return files


# --- parameterization & model writing --------------------------------------

def _frac_bounds(val: float, frac: float, lo: Optional[float] = None, hi: Optional[float] = None) -> Tuple[float, float]:
    lower = max(lo if lo is not None else 0.0, val * (1 - frac))
    upper = min(hi if hi is not None else float("inf"), val * (1 + frac))
    return lower, upper


def _pad_comment(line: str, comment: str, width: int = 40) -> str:
    """Pad each parameter line so comments align at a fixed column."""
    return f"{line:<{width}}# {comment}"


def _safe_ellipticity(d: Dict, fallback: float = 0.3) -> float:
    """
    Prefer 'ellipticity' if present/finite; else derive from axis_ratio if numeric; else fallback.
    (ellipse_fit already tries to populate 'ellipticity', but this is a guard.)
    """
    import math
    ell = d.get("ellipticity", None)
    if isinstance(ell, (int, float)) and math.isfinite(ell):
        return float(ell)

    q = d.get("axis_ratio", None)
    try:
        qf = float(q)
        if math.isfinite(qf) and qf > 0:
            return max(0.0, min(0.99, 1.0 - qf))
    except Exception:
        pass

    return fallback


def gather_parameters(name: str, path: str = "./GalaxyFiles", fltr: str = "r") -> tuple[str, float, float]:
    """
    Generate an imfit 2xSersic config using:
      - geometry (center, PA, ellipticity) from ellipse_fit()
      - photometric shape (n, Re_arcsec, mu_e) from dual_component_slits_and_sersic()
        in photometric_cut.py

    Writes: path/<name>/two_sersic_<fltr>.imfit
    Returns: (output_path, pixel_scale_arcsec_per_pix, zeropoint_mag)
    """
    import math
    import textwrap
    from pathlib import Path

    import table_info
    # table_info.set_directory()

    # ---------- helpers (local, from scratch) ----------
    def clamp(x, lo, hi):
        return max(lo, min(hi, x))

    def safe_float(x, default=None):
        try:
            v = float(x)
            if math.isfinite(v):
                return v
        except Exception:
            pass
        return default

    def get_ellipticity(d, fallback=0.3):
        # Prefer ellipticity; else derive from axis_ratio (b/a); else fallback.
        e = safe_float(d.get("ellipticity", None), None)
        if e is not None:
            return clamp(e, 0.0, 0.95)
        q = safe_float(d.get("axis_ratio", None), None)
        if q is not None and q > 0:
            return clamp(1.0 - q, 0.0, 0.95)
        return clamp(float(fallback), 0.0, 0.95)

    def fmt_param(name, val, lo=None, hi=None, comment=""):
        if lo is not None and hi is not None:
            core = f"{name} {val:.6f} {lo:.6f},{hi:.6f}"
        else:
            core = f"{name} {val:.6f}"
        if comment:
            return f"{core:<38}# {comment}"
        return core

    def pa_to_imfit(pa_deg):
        # User-requested conversion: add 90 deg.
        # Keep in [0,180) because PA has 180-deg degeneracy for ellipses.
        return (float(pa_deg) + 90.0) % 180.0

    def mu_e_to_Ie_pix(mu_e, zeropoint, pixscale_arcsec):
        # mu [mag/arcsec^2] = ZP - 2.5 log10(I_arcsec2)
        # => I_arcsec2 = 10^((ZP - mu)/2.5)
        # Convert to counts/pixel by multiplying by pixarea (arcsec^2/pix).
        I_as2 = 10.0 ** ((float(zeropoint) - float(mu_e)) / 2.5)
        return I_as2 * (float(pixscale_arcsec) ** 2)

    # ---------- inputs: files + ellipse geometry ----------
    files = get_galaxy_files(name, base=path, fltr=fltr)
    sci_fits = files["science"]
    mask_fits = files.get("mask", None)
    invvar_fits = files.get("invvar", None)
    psf_fits = files.get("psf", None)

    from photometric_cut_helpers import ellipse_fit, pixel_scale_from_header_arcsec_per_pix
    host_e, polar_e = ellipse_fit(name)

    pixel_scale = float(pixel_scale_from_header_arcsec_per_pix(sci_fits))
    zeropoint = 22.5  # keep consistent with your 1-D mu conversion unless you intentionally change it

    # Geometry from ellipse fitting
    cx, cy = map(float, host_e["center"])  # shared center
    host_pa_imfit = pa_to_imfit(host_e["PA"])
    polar_pa_imfit = pa_to_imfit(polar_e["PA"])
    host_ell = get_ellipticity(host_e, fallback=0.3)
    polar_ell = get_ellipticity(polar_e, fallback=0.8)

    # Choose slit lengths from ellipse sizes if available, else defaults.
    # (These are ONLY for the 1-D estimate step.)
    host_a = safe_float(host_e.get("semi_major_axis", None), None)
    polar_a = safe_float(polar_e.get("semi_major_axis", None), None)
    host_len = int(max(200, 2.5 * host_a)) if host_a is not None else 300
    polar_len = int(max(250, 2.5 * polar_a)) if polar_a is not None else 450

    # ---------- photometric estimates (n, Re_arcsec, mu_e) ----------
    # This calls your new machinery in photometric_cut.py
    from photometric_cut import dual_component_slits_and_sersic

    results = dual_component_slits_and_sersic(
        sci_fits=sci_fits,
        host_ellipse_results=host_e,
        polar_ellipse_results=polar_e,
        mask_fits=mask_fits,
        invvar_fits=invvar_fits,
        psf_fits=psf_fits,
        zeropoint=zeropoint,
        pixel_scale_arcsec=pixel_scale,
        host_length_pix=host_len,
        polar_length_pix=polar_len,
        width_pix=7.0,
        oversample=2,
        subtract_background=False,
        background_region="ends",
        refine_center=False,
        refine_kwargs=None,
        report_prefix=None,
    )

    host_fit = results["host"]["fit"]
    polar_fit = results["polar"]["fit"]

    if not host_fit.get("success", False):
        raise RuntimeError("Host 1-D Sérsic estimate failed; cannot seed imfit reliably.")
    if not polar_fit.get("success", False):
        raise RuntimeError("Polar 1-D Sérsic estimate failed; cannot seed imfit reliably.")

    # Convert to imfit parameters (pixels + intensities/pixel)
    host_n = float(host_fit["n"])
    host_Re_pix = float(host_fit["Re_arcsec"]) / pixel_scale
    host_Ie_pix = mu_e_to_Ie_pix(host_fit["mu_e"], zeropoint, pixel_scale)

    polar_n = float(polar_fit["n"])
    polar_Re_pix = float(polar_fit["Re_arcsec"]) / pixel_scale
    polar_Ie_pix = mu_e_to_Ie_pix(polar_fit["mu_e"], zeropoint, pixel_scale)

    # ---------- bounds (starter defaults; you said you’ll tune these next) ----------
    # Center bounds (pixels)
    c_tol = 5.0
    x0_lo, x0_hi = cx - c_tol, cx + c_tol
    y0_lo, y0_hi = cy - c_tol, cy + c_tol

    # We should also provide some justifications as to why we choose these bounds

    # PA bounds (deg)
    pa_tol = 10.0
    # Note: wrap issues at 0/180 are annoying; simplest is to allow a broad range.
    # We’ll just clamp to [0,180] and let the solver move.
    def pa_bounds(pa):
        return clamp(pa - pa_tol, 0.0, 180.0), clamp(pa + pa_tol, 0.0, 180.0)

    host_pa_lo, host_pa_hi = pa_bounds(host_pa_imfit)
    polar_pa_lo, polar_pa_hi = pa_bounds(polar_pa_imfit)

    # Ellipticity bounds
    host_ell_lo, host_ell_hi = clamp(host_ell - 0.10, 0.0, 0.95), clamp(host_ell + 0.10, 0.0, 0.95)
    polar_ell_lo, polar_ell_hi = clamp(polar_ell - 0.10, 0.0, 0.95), clamp(polar_ell + 0.10, 0.0, 0.95)

    # Sérsic n bounds
    host_n_lo, host_n_hi = _frac_bounds(host_n, 2)
    host_n_lo = max(host_n_lo, 0.5)
    host_n_hi = max(5.0, host_n_hi)
    polar_n_lo, polar_n_hi = _frac_bounds(polar_n, 2)
    polar_n_lo = max(polar_n_lo, 0.1)

    # Re bounds (pix)
    def re_bounds(re_pix):
        # re_pix = max(re_pix, 0.5)
        return 0.7 * re_pix, 1.3 * re_pix

    host_re_lo, host_re_hi = re_bounds(host_Re_pix)
    polar_re_lo, polar_re_hi = re_bounds(polar_Re_pix)

    # Ie bounds (counts/pix) — wide multiplicative
    def ie_bounds(ie):
        ie = max(ie, 1e-6)
        return ie / 2.0, ie * 2.0

    host_ie_lo, host_ie_hi = ie_bounds(host_Ie_pix)
    polar_ie_lo, polar_ie_hi = ie_bounds(polar_Ie_pix)

    # ---------- write config ----------
    lines = []
    lines.append("# Two-component Sérsic model")
    lines.append("# NOTE: PA here uses your requested +90deg conversion relative to ellipse-fit PA.")
    lines.append("# Sérsic uses I_e (counts/pixel), not mu_e; values below were converted from mu_e.")
    lines.append("")

    lines.append(fmt_param("X0", cx, x0_lo, x0_hi, "shared center x (pix)"))
    lines.append(fmt_param("Y0", cy, y0_lo, y0_hi, "shared center y (pix)"))
    lines.append("")

    # Function labels are supported by imfit (handy in outputs). :contentReference[oaicite:2]{index=2}
    lines.append("FUNCTION Sersic  # LABEL host")
    lines.append(fmt_param("PA", host_pa_imfit, host_pa_lo, host_pa_hi, "host PA (deg)"))
    lines.append(fmt_param("ell", host_ell, host_ell_lo, host_ell_hi, "host ellipticity"))
    lines.append(fmt_param("n", host_n, host_n_lo, host_n_hi, "host Sérsic n"))
    lines.append(fmt_param("I_e", host_Ie_pix, host_ie_lo, host_ie_hi, "host I_e (cts/pix)"))
    lines.append(fmt_param("r_e", host_Re_pix, host_re_lo, host_re_hi, "host r_e (pix)"))
    lines.append("")

    lines.append("FUNCTION Sersic  # LABEL polar")
    lines.append(fmt_param("PA", polar_pa_imfit, polar_pa_lo, polar_pa_hi, "polar PA (deg)"))
    lines.append(fmt_param("ell", polar_ell, polar_ell_lo, polar_ell_hi, "polar ellipticity"))
    lines.append(fmt_param("n", polar_n, polar_n_lo, polar_n_hi, "polar Sérsic n"))
    lines.append(fmt_param("I_e", polar_Ie_pix, polar_ie_lo, polar_ie_hi, "polar I_e (cts/pix)"))
    lines.append(fmt_param("r_e", polar_Re_pix, polar_re_lo, polar_re_hi, "polar r_e (pix)"))
    lines.append("")

    model_text = textwrap.dedent("\n".join(lines)).strip() + "\n"

    out_dir = Path("./GalaxyFiles") / name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"two_sersic_{fltr}.imfit"
    out_path.write_text(model_text)

    print(f"Wrote {out_path}")
    return str(out_path), pixel_scale, zeropoint


def main():
    """
    Quick test: pick the first galaxy directory and write its model using r-band.
    """
    import os 
    import table_info 
    table_info.set_directory()
    import glob

    galaxy_dirs = glob.glob("./GalaxyFiles/*")
    if not galaxy_dirs:
        raise RuntimeError("No ./GalaxyFiles/* directories found.")
    test_galaxy = os.path.basename(galaxy_dirs[0])
    _ = gather_parameters(test_galaxy, fltr="r")
    return


if __name__ == "__main__":
    main()
