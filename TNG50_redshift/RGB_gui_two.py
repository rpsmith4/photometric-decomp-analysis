#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import warnings
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from astropy.io import fits
from astropy.visualization import (
    LinearStretch, SqrtStretch, AsinhStretch, LogStretch, SquaredStretch,
    HistEqStretch, ImageNormalize
)

# Pick an interactive backend if possible
import matplotlib
if matplotlib.get_backend().lower() in ("agg", "template"):
    for candidate in ("QtAgg", "Qt5Agg", "TkAgg"):
        try:
            matplotlib.use(candidate, force=True)
            break
        except Exception:
            pass

import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons
import cv2
from scipy.ndimage import gaussian_filter

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ----------------------------- SDSS RGB -----------------------------

def sdss_rgb(imgs, bands, m=0.03):
    """
    Create an RGB image from g,r,z FITS arrays (float32). Output in [0,1].
    """
    rgbscales = {'g': (2, 6.0), 'r': (1, 3.4), 'z': (0, 2.2)}
    I = 0
    for img, band in zip(imgs, bands):
        plane, scale = rgbscales[band]
        img = np.maximum(0, img * scale + m)
        I = I + img
    I /= len(bands)
    Q = 20.0
    fI = np.arcsinh(Q * I) / np.sqrt(Q)
    I += (I == 0.) * 1e-6
    H, W = I.shape
    rgb = np.zeros((H, W, 3), np.float32)
    for img, band in zip(imgs, bands):
        plane, scale = rgbscales[band]
        rgb[:, :, plane] = (img * scale + m) * fI / I
    return np.clip(rgb, 0, 1)


# --------------------------- DS9-like stretch -----------------------

SCALE_MAP = {
    'linear': LinearStretch,
    'log': LogStretch,
    'sqrt': SqrtStretch,
    'squared': SquaredStretch,
    'asinh': AsinhStretch,
    'histogram': HistEqStretch,
}

def apply_stretch(rgb_image, scale='log', bias=0.0, contrast=1.0, brightness=1.0):
    """
    Apply DS9-style scale/bias/contrast/brightness by operating on HSV Value.
    """
    if scale not in SCALE_MAP:
        raise ValueError(f"Unknown scale '{scale}'")

    # RGB->HSV
    hsv = cv2.cvtColor(rgb_image.astype(np.float32), cv2.COLOR_RGB2HSV)
    h, s, v = cv2.split(hsv)

    stretch_class = SCALE_MAP[scale]
    if scale == 'histogram':
        stretch_instance = stretch_class(v)
        norm = ImageNormalize(v, stretch=stretch_instance, clip=True)
    else:
        vmin_data = float(np.nanmin(v))
        vmax_data = float(np.nanmax(v))
        data_range = vmax_data - vmin_data
        if data_range <= 0:
            data_range = 1.0
        vmin_stretch = vmin_data + bias * data_range
        vmax_stretch = vmin_stretch + (vmax_data - vmin_stretch) / max(contrast, 1e-6)
        norm = ImageNormalize(v, stretch=stretch_class(), vmin=vmin_stretch, vmax=vmax_stretch, clip=True)

    v_stretched = norm(v)

    hsv_stretched = cv2.merge([h, s, v_stretched.astype(np.float32)])
    rgb_stretched = cv2.cvtColor(hsv_stretched, cv2.COLOR_HSV2RGB)
    rgb_final = np.clip(rgb_stretched * float(brightness), 0, 1)
    return rgb_final


# --------------------------- photometric helper ---------------------

def intensity_from_mu(mu_mag_arcsec2: float, zeropoint: float, pixscale_arcsec: float) -> float:
    """
    Convert surface brightness μ [mag/arcsec^2] to counts/pixel using:
        I = pixscale^2 * 10^((ZP - μ)/2.5)
    """
    return (pixscale_arcsec ** 2) * (10.0 ** ((zeropoint - mu_mag_arcsec2) / 2.5))


# --------------- two-branch render (bright & faint) helpers ---------

def render_branch(
    base_rgb: np.ndarray,
    scale: str, bias: float, contrast: float, brightness: float,
    sigma: float
) -> np.ndarray:
    out = apply_stretch(base_rgb, scale=scale, bias=bias, contrast=contrast, brightness=brightness)
    if sigma and sigma > 0:
        out = gaussian_filter(out, sigma=(sigma, sigma, 0))
    return np.clip(out, 0, 1)


def color_match_faint_to_bright_at_threshold(
    bright_rgb: np.ndarray,
    faint_rgb: np.ndarray,
    r_img: np.ndarray,
    I_thr: float,
    ring_frac: float = 0.02,
    gain_clip: Tuple[float, float] = (0.5, 2.0)
) -> np.ndarray:
    """
    Scale faint_rgb (per channel) so that in a thin ring around the threshold
    its colors match bright_rgb on average. This avoids a color jump.
    """
    eps = 1e-12
    ring_w = max(eps, ring_frac * max(I_thr, eps))
    ring = np.isfinite(r_img) & (np.abs(r_img - I_thr) <= ring_w)

    if ring.sum() < 50:  # not enough pixels; skip
        return faint_rgb

    gains = []
    for c in range(3):
        b = np.mean(bright_rgb[..., c][ring])
        f = np.mean(faint_rgb[..., c][ring])
        if not np.isfinite(b) or not np.isfinite(f) or f <= eps:
            gains.append(1.0)
        else:
            gains.append(np.clip(b / f, gain_clip[0], gain_clip[1]))
    gains = np.array(gains, dtype=np.float32)[None, None, :]
    return np.clip(faint_rgb * gains, 0, 1)


def render_rgb_two_branch(
    g: np.ndarray, r: np.ndarray, z: np.ndarray,
    zeropoint: float, pixscale: float,
    # bright branch:
    scale_b: str, bias_b: float, contrast_b: float, brightness_b: float, sigma_b: float,
    # faint branch:
    scale_f: str, bias_f: float, contrast_f: float, brightness_f: float, sigma_f: float,
    # threshold:
    mu_r: float,
    # feathering:
    feather_frac: float = 0.10
) -> np.ndarray:
    """
    Build SDSS RGB, render two versions (bright & faint), adjust faint colors at the threshold,
    and feather-blend across a band around the isophote for smooth transition.
    """
    base = sdss_rgb([g, r, z], ['g', 'r', 'z'])

    rgb_b = render_branch(base, scale_b, bias_b, contrast_b, brightness_b, sigma_b)
    rgb_f = render_branch(base, scale_f, bias_f, contrast_f, brightness_f, sigma_f)

    r_img = np.asarray(r, dtype=np.float32)
    I_thr = intensity_from_mu(mu_r, zeropoint, pixscale)

    # Color-match faint to bright at the threshold ring
    rgb_f = color_match_faint_to_bright_at_threshold(rgb_b, rgb_f, r_img, I_thr, ring_frac=0.02)

    # Feather blend across band around threshold
    w = max(1e-12, feather_frac * max(I_thr, 1e-12))  # half-width of blend band in counts
    # alpha: 0 uses faint; 1 uses bright
    t = (r_img - (I_thr - w)) / (2.0 * w)
    alpha = np.clip(t, 0.0, 1.0).astype(np.float32)
    alpha = alpha[..., None]  # broadcast on channels

    out = alpha * rgb_b + (1.0 - alpha) * rgb_f
    return np.clip(out, 0, 1)


# --------------------------- render & save helpers ------------------

def save_rgb_annotated(
    rgb: np.ndarray,
    pixscale: float,
    out_path: Path,
    galaxy_name: Optional[str] = None
):
    """
    Save the RGB image without GUI chrome, with:
      - FOV (arcmin×arcmin) in bottom-left (white)
      - Galaxy name (if provided) in top-left (bright green)

    Uses an off-screen Agg canvas.
    """
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    ny, nx, _ = rgb.shape
    fov_x_arcmin = pixscale * nx / 60.0
    fov_y_arcmin = pixscale * ny / 60.0

    fig = Figure(figsize=(8, 8), dpi=150)
    canvas = FigureCanvasAgg(fig)
    ax = fig.add_subplot(111)
    ax.imshow(rgb, origin='lower')
    ax.set_axis_off()

    # Galaxy name (top-left)
    if galaxy_name:
        ax.text(0.03, 0.94, str(galaxy_name), fontsize=15, color='lime', transform=ax.transAxes,
                ha='left', va='baseline',
                bbox=dict(facecolor='black', alpha=0.5, edgecolor='none', boxstyle='round,pad=0.2'))

    # FOV (bottom-left)
    ax.text(0.03, 0.03, f"{fov_x_arcmin:.1f}'×{fov_y_arcmin:.1f}'",
            fontsize=14, color='white', transform=ax.transAxes,
            ha='left', va='baseline',
            bbox=dict(facecolor='black', alpha=0.5, edgecolor='none', boxstyle='round,pad=0.2'))

    fig.savefig(out_path, dpi=300, bbox_inches='tight', pad_inches=0)


# ------------------------------ GUI --------------------------------

class RGBAdjustGUI:
    def __init__(self, g_path, r_path, z_path, out_png="adjusted_RGB.png",
                 zero_point: float = None, pixscale: float = None,
                 mu_r_init: float = 25.5, mu_r_min: float = 20.0, mu_r_max: float = 30.0,
                 galaxy_name: Optional[str] = None,
                 # BRIGHT (left)
                 scale_b: str = "log", bias_b: float = 0.18,
                 contrast_b: float = 1.08, brightness_b: float = 1.11,
                 sigma_b: float = 0.0,
                 # FAINT (right)
                 scale_f: str = "log", bias_f: float = 0.18,
                 contrast_f: float = 1.08, brightness_f: float = 1.11,
                 sigma_f: float = 3.0):
        self.g_path = Path(g_path)
        self.r_path = Path(r_path)
        self.z_path = Path(z_path)
        self.out_png = Path(out_png)

        if zero_point is None or pixscale is None:
            raise ValueError("zero_point and pixscale are required.")
        self.zero_point = float(zero_point)
        self.pixscale = float(pixscale)
        self.galaxy_name = galaxy_name

        # Load data
        self.g = self._read_fits(self.g_path)
        self.r = self._read_fits(self.r_path)
        self.z = self._read_fits(self.z_path)

        # Isophote (in mag/arcsec^2)
        self.mu_r = float(mu_r_init)
        self.mu_r_min = float(mu_r_min)
        self.mu_r_max = float(mu_r_max)

        # Params (bright/faint)
        self.scale_b = str(scale_b)
        self.bias_b = float(bias_b)
        self.contrast_b = float(contrast_b)
        self.brightness_b = float(brightness_b)
        self.sigma_b = float(sigma_b)

        self.scale_f = str(scale_f)
        self.bias_f = float(bias_f)
        self.contrast_f = float(contrast_f)
        self.brightness_f = float(brightness_f)
        self.sigma_f = float(sigma_f)

        # Figure layout
        self.fig = plt.figure(figsize=(10.5, 7.6), dpi=110)
        gs = self.fig.add_gridspec(3, 2, height_ratios=[12, 3, 3], width_ratios=[1, 1],
                                   hspace=0.25, wspace=0.18)
        self.ax_img = self.fig.add_subplot(gs[0, :])
        self.ax_img.set_title(
            f"SDSS-style RGB • Bright/Faint branches • ZP={self.zero_point:.3f}, pix={self.pixscale:.4f}\"/px"
        )
        self.ax_img.set_axis_off()

        # Initial render
        rgb0 = self.render()
        self.im_artist = self.ax_img.imshow(rgb0, origin='lower')

        # Overlay texts (name top-left, FOV bottom-left)
        ny, nx, _ = rgb0.shape
        fov_x_arcmin = self.pixscale * nx / 60.0
        fov_y_arcmin = self.pixscale * ny / 60.0
        if self.galaxy_name:
            self.ax_img.text(0.03, 0.94, str(self.galaxy_name), fontsize=15, color='lime',
                             transform=self.ax_img.transAxes, ha='left', va='baseline',
                             bbox=dict(facecolor='black', alpha=0.5, edgecolor='none', boxstyle='round,pad=0.2'))
        self.ax_img.text(0.03, 0.03, f"{fov_x_arcmin:.1f}'×{fov_y_arcmin:.1f}'", fontsize=14, color='white',
                         transform=self.ax_img.transAxes, ha='left', va='baseline',
                         bbox=dict(facecolor='black', alpha=0.5, edgecolor='none', boxstyle='round,pad=0.2'))

        # ------- Left (BRIGHT) controls -------
        ax_radio_b = self.fig.add_subplot(gs[1, 0])
        ax_radio_b.set_title("Bright scale (≥ μ)", fontsize=10)
        scale_options = ('linear', 'log', 'sqrt', 'squared', 'asinh', 'histogram')
        self.radio_b = RadioButtons(ax_radio_b, scale_options, active=0)
        try:
            self.radio_b.set_active(scale_options.index(self.scale_b))
        except ValueError:
            self.scale_b = 'linear'
            self.radio_b.set_active(0)
        self.radio_b.on_clicked(self._on_scale_b)

        # sliders (bright)
        ax_bb = self.fig.add_axes([0.10, 0.24, 0.33, 0.03])
        self.s_bias_b = Slider(ax_bb, 'Bias (bright)', 0.0, 1.0, valinit=self.bias_b, valstep=0.005)
        self.s_bias_b.on_changed(self._on_sliders)

        ax_cb = self.fig.add_axes([0.10, 0.19, 0.33, 0.03])
        self.s_contrast_b = Slider(ax_cb, 'Contrast (bright)', 0.2, 3.0, valinit=self.contrast_b, valstep=0.01)
        self.s_contrast_b.on_changed(self._on_sliders)

        ax_vb = self.fig.add_axes([0.10, 0.14, 0.33, 0.03])
        self.s_bright_b = Slider(ax_vb, 'Brightness (bright)', 0.1, 5.0, valinit=self.brightness_b, valstep=0.01)
        self.s_bright_b.on_changed(self._on_sliders)

        ax_sb = self.fig.add_axes([0.10, 0.09, 0.33, 0.03])
        self.s_sigma_b = Slider(ax_sb, 'Sigma (bright, px)', 0.0, 10.0, valinit=self.sigma_b, valstep=0.1)
        self.s_sigma_b.on_changed(self._on_sliders)

        # ------- Right (FAINT) controls -------
        ax_radio_f = self.fig.add_subplot(gs[1, 1])
        ax_radio_f.set_title("Faint scale (< μ)", fontsize=10)
        self.radio_f = RadioButtons(ax_radio_f, scale_options, active=0)
        try:
            self.radio_f.set_active(scale_options.index(self.scale_f))
        except ValueError:
            self.scale_f = 'linear'
            self.radio_f.set_active(0)
        self.radio_f.on_clicked(self._on_scale_f)

        ax_bf = self.fig.add_axes([0.57, 0.24, 0.33, 0.03])
        self.s_bias_f = Slider(ax_bf, 'Bias (faint)', 0.0, 1.0, valinit=self.bias_f, valstep=0.005)
        self.s_bias_f.on_changed(self._on_sliders)

        ax_cf = self.fig.add_axes([0.57, 0.19, 0.33, 0.03])
        self.s_contrast_f = Slider(ax_cf, 'Contrast (faint)', 0.2, 3.0, valinit=self.contrast_f, valstep=0.01)
        self.s_contrast_f.on_changed(self._on_sliders)

        ax_vf = self.fig.add_axes([0.57, 0.14, 0.33, 0.03])
        self.s_bright_f = Slider(ax_vf, 'Brightness (faint)', 0.1, 5.0, valinit=self.brightness_f, valstep=0.01)
        self.s_bright_f.on_changed(self._on_sliders)

        ax_sf = self.fig.add_axes([0.57, 0.09, 0.33, 0.03])
        self.s_sigma_f = Slider(ax_sf, 'Sigma (faint, px)', 0.0, 10.0, valinit=self.sigma_f, valstep=0.1)
        self.s_sigma_f.on_changed(self._on_sliders)

        # ------- Middle row: Isophote + buttons -------
        ax_mu = self.fig.add_axes([0.28, 0.045, 0.44, 0.03])
        self.s_mu = Slider(ax_mu, r'$\mu_r$ (mag/arcsec$^2$)', self.mu_r_min, self.mu_r_max,
                           valinit=self.mu_r, valstep=0.05)
        self.s_mu.on_changed(self._on_sliders)

        ax_reset = self.fig.add_axes([0.10, 0.045, 0.12, 0.035])
        self.b_reset = Button(ax_reset, 'Reset')
        self.b_reset.on_clicked(self._on_reset)

        ax_save = self.fig.add_axes([0.78, 0.045, 0.12, 0.035])
        self.b_save = Button(ax_save, 'Save PNG')
        self.b_save.on_clicked(self._on_save)

        plt.show()

    @staticmethod
    def _read_fits(path: Path):
        with fits.open(path, memmap=True) as hdul:
            arr = hdul[0].data
        if arr is None:
            raise RuntimeError(f"No data in {path}")
        if arr.ndim > 2:
            arr = arr[0]
        return np.array(arr, dtype=np.float32)

    def render(self):
        return render_rgb_two_branch(
            self.g, self.r, self.z,
            zeropoint=self.zero_point, pixscale=self.pixscale,
            scale_b=self.scale_b, bias_b=self.bias_b, contrast_b=self.contrast_b,
            brightness_b=self.brightness_b, sigma_b=self.sigma_b,
            scale_f=self.scale_f, bias_f=self.bias_f, contrast_f=self.contrast_f,
            brightness_f=self.brightness_f, sigma_f=self.sigma_f,
            mu_r=self.mu_r,
            feather_frac=0.10  # fixed smooth band (±10% of threshold in counts)
        )

    # ---- callbacks ----
    def _on_scale_b(self, label):
        self.scale_b = str(label)
        self._refresh()

    def _on_scale_f(self, label):
        self.scale_f = str(label)
        self._refresh()

    def _on_sliders(self, _=None):
        # bright
        self.bias_b = float(self.s_bias_b.val)
        self.contrast_b = float(self.s_contrast_b.val)
        self.brightness_b = float(self.s_bright_b.val)
        self.sigma_b = float(self.s_sigma_b.val)
        # faint
        self.bias_f = float(self.s_bias_f.val)
        self.contrast_f = float(self.s_contrast_f.val)
        self.brightness_f = float(self.s_bright_f.val)
        self.sigma_f = float(self.s_sigma_f.val)
        # threshold
        self.mu_r = float(self.s_mu.val)
        self._refresh(lazy=True)

    def _on_reset(self, _event):
        # sensible defaults
        self.scale_b = 'log'; self.radio_b.set_active(1)
        self.bias_b = 0.18; self.s_bias_b.set_val(self.bias_b)
        self.contrast_b = 1.08; self.s_contrast_b.set_val(self.contrast_b)
        self.brightness_b = 1.11; self.s_bright_b.set_val(self.brightness_b)
        self.sigma_b = 0.0; self.s_sigma_b.set_val(self.sigma_b)

        self.scale_f = 'log'; self.radio_f.set_active(1)
        self.bias_f = 0.18; self.s_bias_f.set_val(self.bias_f)
        self.contrast_f = 1.08; self.s_contrast_f.set_val(self.contrast_f)
        self.brightness_f = 1.11; self.s_bright_f.set_val(self.brightness_f)
        self.sigma_f = 3.0; self.s_sigma_f.set_val(self.sigma_f)

        self.mu_r = 25.5; self.s_mu.set_val(self.mu_r)
        self._refresh()

    def _on_save(self, _event):
        rgb = self.render()
        out = self.out_png
        if self.galaxy_name and (out.name == "adjusted_RGB.png"):
            out = out.with_name(f"{self.galaxy_name}.png")
        save_rgb_annotated(rgb, self.pixscale, out, galaxy_name=self.galaxy_name)
        print(f"✅ Saved: {out}")

    def _refresh(self, lazy=False):
        self.im_artist.set_data(self.render())
        if lazy:
            self.fig.canvas.draw_idle()
        else:
            self.fig.canvas.draw()


# ------------------------------ CLI --------------------------------

def parse_args():
    ap = argparse.ArgumentParser(
        description="SDSS-style RGB viewer with *two* branches: bright (≥ μ) and faint (< μ), "
                    "each with its own scale/bias/contrast/brightness/sigma. "
                    "Smoothly blends at the isophote with color matching."
    )
    ap.add_argument("g", help="Path to g-band FITS")
    ap.add_argument("r", help="Path to r-band FITS")
    ap.add_argument("z", help="Path to z-band FITS")
    ap.add_argument("--zero_point", type=float, required=True,
                    help="Photometric zero point (mag) for r band.")
    ap.add_argument("--pixscale", type=float, required=True,
                    help="Pixel scale (arcsec/pixel).")
    ap.add_argument("-o", "--out", default="adjusted_RGB.png", help="Output PNG filename")
    ap.add_argument("--galaxy-name", type=str, default=None,
                    help="Optional galaxy name for annotation/filename.")

    # Isophote
    ap.add_argument("--mu_init", type=float, default=25.5,
                    help="Isophote level μ_r (mag/arcsec^2).")
    ap.add_argument("--mu_min", type=float, default=20.0,
                    help="Minimum μ slider.")
    ap.add_argument("--mu_max", type=float, default=30.0,
                    help="Maximum μ slider.")

    # Bright branch (≥ μ)
    ap.add_argument("--scale_bright", type=str, default="log",
                    choices=['linear', 'log', 'sqrt', 'squared', 'asinh', 'histogram'])
    ap.add_argument("--bias_bright", type=float, default=0.18)
    ap.add_argument("--contrast_bright", type=float, default=1.08)
    ap.add_argument("--brightness_bright", type=float, default=1.11)
    ap.add_argument("--sigma_bright", type=float, default=0.0)

    # Faint branch (< μ)
    ap.add_argument("--scale_faint", type=str, default="log",
                    choices=['linear', 'log', 'sqrt', 'squared', 'asinh', 'histogram'])
    ap.add_argument("--bias_faint", type=float, default=0.18)
    ap.add_argument("--contrast_faint", type=float, default=1.08)
    ap.add_argument("--brightness_faint", type=float, default=1.11)
    ap.add_argument("--sigma_faint", type=float, default=3.0)

    ap.add_argument("--no-gui", action="store_true",
                    help="Render once with given parameters and save PNG (no GUI).")
    return ap.parse_args()

def main():
    args = parse_args()

    if args.no_gui:
        # Headless render
        def read_fits(p: str) -> np.ndarray:
            with fits.open(p, memmap=True) as hdul:
                arr = hdul[0].data
            if arr.ndim > 2:
                arr = arr[0]
            return np.array(arr, dtype=np.float32)

        g = read_fits(args.g)
        r = read_fits(args.r)
        z = read_fits(args.z)

        rgb = render_rgb_two_branch(
            g, r, z,
            zeropoint=args.zero_point, pixscale=args.pixscale,
            scale_b=args.scale_bright, bias_b=args.bias_bright, contrast_b=args.contrast_bright,
            brightness_b=args.brightness_bright, sigma_b=args.sigma_bright,
            scale_f=args.scale_faint, bias_f=args.bias_faint, contrast_f=args.contrast_faint,
            brightness_f=args.brightness_faint, sigma_f=args.sigma_faint,
            mu_r=args.mu_init,
            feather_frac=0.10
        )

        out_path = Path(args.out)
        if args.galaxy_name and (out_path.name == "adjusted_RGB.png"):
            out_path = out_path.with_name(f"{args.galaxy_name}.png")
        save_rgb_annotated(rgb, args.pixscale, out_path, galaxy_name=args.galaxy_name)
        print(f"✅ Saved (no GUI): {out_path}")
        return

    # GUI mode
    RGBAdjustGUI(
        args.g, args.r, args.z,
        out_png=args.out,
        zero_point=args.zero_point,
        pixscale=args.pixscale,
        mu_r_init=args.mu_init,
        mu_r_min=args.mu_min,
        mu_r_max=args.mu_max,
        galaxy_name=args.galaxy_name,
        scale_b=args.scale_bright,
        bias_b=args.bias_bright,
        contrast_b=args.contrast_bright,
        brightness_b=args.brightness_bright,
        sigma_b=args.sigma_bright,
        scale_f=args.scale_faint,
        bias_f=args.bias_faint,
        contrast_f=args.contrast_faint,
        brightness_f=args.brightness_faint,
        sigma_f=args.sigma_faint
    )

if __name__ == "__main__":
    main()
