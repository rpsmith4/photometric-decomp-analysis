#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import warnings
from pathlib import Path
from typing import Optional

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


# --------------------------- render & save helpers ------------------

def render_rgb_with_isomask(
    g: np.ndarray, r: np.ndarray, z: np.ndarray,
    zeropoint: float, pixscale: float,
    scale: str, bias: float, contrast: float, brightness: float,
    mu_r: float, sigma: float
) -> np.ndarray:
    """
    Build an SDSS-style RGB, apply stretch, and outside r-band isophote smooth by sigma.
    Returns RGB float array in [0,1].
    """
    base = sdss_rgb([g, r, z], ['g', 'r', 'z'])
    rgb = apply_stretch(base, scale=scale, bias=bias, contrast=contrast, brightness=brightness)

    I_thr = intensity_from_mu(mu_r, zeropoint, pixscale)
    r_img = np.asarray(r, dtype=np.float32)
    mask_inside = r_img >= I_thr

    rgb_smooth = gaussian_filter(rgb, sigma=(sigma, sigma, 0))
    out = rgb.copy()
    out[~mask_inside] = rgb_smooth[~mask_inside]
    return np.clip(out, 0, 1)


def save_rgb_annotated(
    rgb: np.ndarray,
    pixscale: float,
    out_path: Path,
    galaxy_name: Optional[str] = None,
    isTNG: bool = False
):
    """
    Save the RGB image without GUI chrome, with:
      - FOV (arcmin×arcmin) in bottom-left (white)
      - Galaxy name (if provided) in top-left (bright green)

    Uses an off-screen Agg canvas to avoid interacting with the GUI backend.
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
    if not isTNG:
        ax.text(0.03, 0.03, f"{fov_x_arcmin:.1f}'×{fov_y_arcmin:.1f}'",
            fontsize=14, color='white', transform=ax.transAxes,
            ha='left', va='baseline',
            bbox=dict(facecolor='black', alpha=0.5, edgecolor='none', boxstyle='round,pad=0.2'))
    else:
        fov_x = 80 #kpc
        fov_y = 80 #kpc
        ax.text(0.03, 0.03, f"{fov_x:.1f}kpc×{fov_y:.1f}kpc",
            fontsize=14, color='white', transform=ax.transAxes,
            ha='left', va='baseline',
            bbox=dict(facecolor='black', alpha=0.5, edgecolor='none', boxstyle='round,pad=0.2'))

    fig.savefig(out_path, dpi=300, bbox_inches='tight', pad_inches=0)
    # No plt.close(), no GUI interaction


# ------------------------------ GUI --------------------------------

class RGBAdjustGUI:
    def __init__(self, g_path, r_path, z_path, out_png="adjusted_RGB.png",
                 zero_point: float = None, pixscale: float = None,
                 mu_r_init: float = 25.5, mu_r_min: float = 20.0, mu_r_max: float = 30.0,
                 sigma_init: float = 3.0, galaxy_name: Optional[str] = None,
                 scale: str = "log", bias: float = 0.18,
                 contrast: float = 1.08, brightness: float = 1.11):
        self.g_path = Path(g_path)
        self.r_path = Path(r_path)
        self.z_path = Path(z_path)
        self.out_png = Path(out_png)

        # Require calibration parameters for r-band μ thresholding
        if zero_point is None or pixscale is None:
            raise ValueError("zero_point and pixscale are required for isophote masking in mag/arcsec^2.")
        self.zero_point = float(zero_point)
        self.pixscale = float(pixscale)
        self.galaxy_name = galaxy_name

        # Load data
        self.g = self._read_fits(self.g_path)
        self.r = self._read_fits(self.r_path)
        self.z = self._read_fits(self.z_path)

        # Defaults from CLI / constructor
        self.scale = str(scale)
        self.bias = float(bias)
        self.contrast = float(contrast)
        self.brightness = float(brightness)

        # Isophote in mag/arcsec^2 (r band) and smoothing sigma
        self.mu_r = float(mu_r_init)
        self.mu_r_min = float(mu_r_min)
        self.mu_r_max = float(mu_r_max)
        self.sigma = float(sigma_init)

        # Figure & axes
        self.fig = plt.figure(figsize=(9.5, 7.2), dpi=110)
        gs = self.fig.add_gridspec(2, 2, height_ratios=[12, 3], width_ratios=[1, 1], hspace=0.2, wspace=0.18)
        self.ax_img = self.fig.add_subplot(gs[0, :])
        self.ax_img.set_title(
            f"SDSS-style RGB • r-band μ mask • ZP={self.zero_point:.3f}, pix={self.pixscale:.4f}\"/px"
        )
        self.ax_img.set_axis_off()

        # Initial render
        rgb0 = self.render()
        self.im_artist = self.ax_img.imshow(rgb0, origin='lower')

        # Overlay texts (name top-left, FOV bottom-left)
        ny, nx, _ = rgb0.shape
        fov_x_arcmin = self.pixscale * nx / 60.0
        fov_y_arcmin = self.pixscale * ny / 60.0
        self.txt_name = None
        if self.galaxy_name:
            self.txt_name = self.ax_img.text(
                0.03, 0.94, str(self.galaxy_name), fontsize=15, color='lime',
                transform=self.ax_img.transAxes, ha='left', va='baseline',
                bbox=dict(facecolor='black', alpha=0.5, edgecolor='none', boxstyle='round,pad=0.2')
            )
        self.txt_fov = self.ax_img.text(
            0.03, 0.03, f"{fov_x_arcmin:.1f}'×{fov_y_arcmin:.1f}'", fontsize=14, color='white',
            transform=self.ax_img.transAxes, ha='left', va='baseline',
            bbox=dict(facecolor='black', alpha=0.5, edgecolor='none', boxstyle='round,pad=0.2')
        )

        self.fig.canvas.draw_idle()

        # ---- Controls ----
        # Radio for scale (initialize to provided value)
        ax_radio = self.fig.add_subplot(gs[1, 0])
        ax_radio.set_title("Scale", fontsize=10)
        scale_options = ('linear', 'log', 'sqrt', 'squared', 'asinh', 'histogram')
        self.radio = RadioButtons(ax_radio, scale_options, active=0)
        # Set the active radio to match the provided scale, if possible
        try:
            active_idx = scale_options.index(self.scale)
        except ValueError:
            active_idx = 0
            self.scale = scale_options[active_idx]
        self.radio.set_active(active_idx)
        self.radio.on_clicked(self._on_scale)

        # Sliders: bias, contrast, brightness (initialize to provided values)
        ax_bias = self.fig.add_axes([0.55, 0.29, 0.35, 0.03])
        self.s_bias = Slider(ax_bias, 'Bias', 0.0, 1.0, valinit=self.bias, valstep=0.005)
        self.s_bias.on_changed(self._on_slider)

        ax_con = self.fig.add_axes([0.55, 0.24, 0.35, 0.03])
        self.s_contrast = Slider(ax_con, 'Contrast', 0.2, 3.0, valinit=self.contrast, valstep=0.01)
        self.s_contrast.on_changed(self._on_slider)

        ax_bri = self.fig.add_axes([0.55, 0.19, 0.35, 0.03])
        self.s_bright = Slider(ax_bri, 'Brightness', 0.1, 5.0, valinit=self.brightness, valstep=0.01)
        self.s_bright.on_changed(self._on_slider)

        # Slider for μ_r (mag/arcsec^2) and smoothing sigma
        ax_mu = self.fig.add_axes([0.55, 0.14, 0.35, 0.03])
        self.s_mu = Slider(ax_mu, r'$\mu_r$ (mag/arcsec$^2$)',
                           self.mu_r_min, self.mu_r_max, valinit=self.mu_r, valstep=0.05)
        self.s_mu.on_changed(self._on_slider)

        ax_sig = self.fig.add_axes([0.55, 0.09, 0.35, 0.03])
        self.s_sigma = Slider(ax_sig, 'Sigma (px)', 0.1, 10.0, valinit=self.sigma, valstep=0.1)
        self.s_sigma.on_changed(self._on_slider)

        # Reset & Save
        ax_reset = self.fig.add_axes([0.55, 0.04, 0.12, 0.04])
        self.b_reset = Button(ax_reset, 'Reset')
        self.b_reset.on_clicked(self._on_reset)

        ax_save = self.fig.add_axes([0.73, 0.04, 0.17, 0.04])
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
        """
        Stretch RGB and apply r-band isophote mask (outside smoothed by sigma).
        """
        return render_rgb_with_isomask(
            self.g, self.r, self.z,
            zeropoint=self.zero_point, pixscale=self.pixscale,
            scale=self.scale, bias=self.bias, contrast=self.contrast, brightness=self.brightness,
            mu_r=self.mu_r, sigma=self.sigma
        )

    # ------------------ Callbacks ------------------

    def _on_scale(self, label):
        self.scale = str(label)
        self._refresh()

    def _on_slider(self, _=None):
        self.bias = float(self.s_bias.val)
        self.contrast = float(self.s_contrast.val)
        self.brightness = float(self.s_bright.val)
        self.mu_r = float(self.s_mu.val)
        self.sigma = float(self.s_sigma.val)
        self._refresh(lazy=True)

    def _on_reset(self, _event):
        self.scale = 'linear'
        self.bias = 0.10
        self.contrast = 1.00
        self.brightness = 1.00
        self.mu_r = 25.5
        self.sigma = 3.0
        self.radio.set_active(0)  # linear
        self.s_bias.set_val(self.bias)
        self.s_contrast.set_val(self.contrast)
        self.s_bright.set_val(self.brightness)
        self.s_mu.set_val(self.mu_r)
        self.s_sigma.set_val(self.sigma)
        self._refresh()

    def _on_save(self, _event):
        # Render fresh RGB with current settings and save a clean image (no GUI chrome)
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
        description="Interactive SDSS-style RGB viewer with DS9-like stretch controls. "
                    "Outside an r-band isophote (μ in mag/arcsec^2), the image is smoothed."
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
                    help="Optional galaxy name to annotate and to use as output filename if not specified.")

    # Optional initial values (still changeable in the GUI)
    ap.add_argument("--mu_init", type=float, default=25.5,
                    help="Initial isophote level μ_r (mag/arcsec^2).")
    ap.add_argument("--mu_min", type=float, default=20.0,
                    help="Minimum μ slider (mag/arcsec^2).")
    ap.add_argument("--mu_max", type=float, default=30.0,
                    help="Maximum μ slider (mag/arcsec^2).")
    ap.add_argument("--sigma_init", type=float, default=3.0,
                    help="Initial Gaussian smoothing sigma (pixels).")

    # Initial stretch settings (now passed into the GUI and used to initialize controls)
    ap.add_argument("--scale", type=str, default="log",
                    choices=['linear', 'log', 'sqrt', 'squared', 'asinh', 'histogram'],
                    help="Initial display scale.")
    ap.add_argument("--bias", type=float, default=0.18,
                    help="Initial bias (0–1).")
    ap.add_argument("--contrast", type=float, default=1.08,
                    help="Initial contrast.")
    ap.add_argument("--brightness", type=float, default=1.11,
                    help="Initial brightness multiplier.")

    ap.add_argument("--no-gui", action="store_true",
                    help="Render once using current/default parameters and save PNG without opening the GUI.")
    return ap.parse_args()

def main():
    args = parse_args()

    if args.no_gui:
        # Headless render & save
        def read_fits(p: str) -> np.ndarray:
            with fits.open(p, memmap=True) as hdul:
                arr = hdul[0].data
            if arr.ndim > 2:
                arr = arr[0]
            return np.array(arr, dtype=np.float32)

        g = read_fits(args.g)
        r = read_fits(args.r)
        z = read_fits(args.z)

        rgb = render_rgb_with_isomask(
            g, r, z,
            zeropoint=args.zero_point, pixscale=args.pixscale,
            scale='log', bias=0.18, contrast=1.08, brightness=1.11,
            mu_r=args.mu_init, sigma=args.sigma_init
        )

        out_path = Path(args.out)
        if args.galaxy_name and (out_path.name == "adjusted_RGB.png"):
            out_path = out_path.with_name(f"{args.galaxy_name}.png")

        save_rgb_annotated(rgb, args.pixscale, out_path, galaxy_name=args.galaxy_name)
        print(f"✅ Saved (no GUI): {out_path}")
        return

    # GUI mode — pass initial stretch settings so widgets start with CLI-provided values
    RGBAdjustGUI(args.g, args.r, args.z,
                 out_png=args.out,
                 zero_point=args.zero_point,
                 pixscale=args.pixscale,
                 mu_r_init=args.mu_init,
                 mu_r_min=args.mu_min,
                 mu_r_max=args.mu_max,
                 sigma_init=args.sigma_init,
                 galaxy_name=args.galaxy_name,
                 scale=args.scale,
                 bias=args.bias,
                 contrast=args.contrast,
                 brightness=args.brightness)

if __name__ == "__main__":
    main()
