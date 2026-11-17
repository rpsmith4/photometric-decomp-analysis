#!/usr/bin/env python3
"""Create a per-galaxy PDF report with JPGs, PyImfit models, and fit parameters."""

import argparse
import math
import re
from pathlib import Path


BANDS = ("g", "r")
FIT_TYPES = ("2_sersic", "1_sersic_1_gauss_ring", "3_sersic")
PIXEL_SCALE_ARCSEC = 0.262
DEFAULT_LOG_SCALE_FACTOR = 1000.0
NANOMAGGIES_AB_ZEROPOINT = 22.5
RADIUS_PARAMETERS = {"r_e", "r_ring", "sigma_r"}
INTENSITY_PARAMETERS = {"i_0", "i_e"}
UNCERTAINTY_RE = re.compile(
    r"\+/-\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?|nan|inf)",
    re.IGNORECASE,
)


def find_galaxy_directories(root: Path, fit_type: str, recursive: bool) -> list[Path]:
    parameter_filenames = {
        f"{fit_type}_{band}_fit_params.txt" for band in BANDS
    }
    if recursive:
        candidates = root.rglob("*_fit_params.txt")
    else:
        candidates = root.glob("*_fit_params.txt")

    return sorted(
        {
            candidate.parent
            for candidate in candidates
            if candidate.name in parameter_filenames
        }
    )


def read_scale_table(scale_table_path: Path) -> dict[str, float]:
    import pandas as pd

    scale_table = pd.read_csv(scale_table_path, encoding="utf-8-sig", dtype=str)
    columns = {
        str(column).strip().casefold(): column for column in scale_table.columns
    }
    if "name" not in columns or "scale" not in columns:
        raise ValueError(
            f"Scale table must contain NAME and SCALE columns: {scale_table_path}"
        )

    galaxy_scales = {}
    for line_number, (name_value, scale_value) in enumerate(
        scale_table[[columns["name"], columns["scale"]]].itertuples(
            index=False, name=None
        ),
        start=2,
    ):
        if pd.isna(name_value) or not str(name_value).strip():
            continue
        if pd.isna(scale_value) or not str(scale_value).strip():
            continue
        galaxy_name = str(name_value).strip()
        try:
            scale = float(scale_value)
        except ValueError as error:
            raise ValueError(
                f"Invalid SCALE value {scale_value!r} in {scale_table_path} "
                f"at line {line_number}"
            ) from error
        if math.isnan(scale):
            continue
        if not math.isfinite(scale) or scale <= 0:
            raise ValueError(
                f"SCALE must be positive and finite in {scale_table_path} "
                f"at line {line_number}"
            )

        key = galaxy_name.casefold()
        if key in galaxy_scales:
            raise ValueError(
                f"Duplicate galaxy name {galaxy_name!r} in {scale_table_path}"
            )
        galaxy_scales[key] = scale

    return galaxy_scales


def read_components(
    parameter_path: Path,
    model: object,
    kpc_per_arcsec: float | None = None,
) -> list[str]:
    """Format the saved best-fit values and their reported uncertainties."""
    lines = parameter_path.read_text(encoding="utf-8").splitlines()
    function_lines = [
        index for index, line in enumerate(lines) if "FUNCTION" in line
    ]
    components = []

    for index, function in enumerate(model.functionList()):
        function_data = function.getFunctionAsDict()
        parameter_values = function_data["parameters"]
        header_index = function_lines[index] if index < len(function_lines) else None
        next_header = (
            function_lines[index + 1]
            if index + 1 < len(function_lines)
            else len(lines)
        )

        label = f"Component {index + 1}"
        if header_index is not None:
            label_match = re.search(r"\bLABEL\s+(.+?)\s*$", lines[header_index])
            if label_match:
                label = label_match.group(1).strip().strip('"')

        function_type = str(function_data.get("name", "Model"))
        component_lines = [f"{label} ({function_type})"]
        for parameter_index, (parameter_name, raw_value) in enumerate(
            parameter_values.items()
        ):
            if hasattr(raw_value, "__len__") and not isinstance(raw_value, (str, bytes)):
                raw_value = raw_value[0]
            value = float(raw_value)
            uncertainty = None
            if (
                header_index is not None
                and header_index + parameter_index + 1 < next_header
            ):
                match = UNCERTAINTY_RE.search(
                    lines[header_index + parameter_index + 1]
                )
                if match:
                    uncertainty = match.group(1)

            value_text = f"{value:.6g}"
            if uncertainty is not None:
                value_text += f" +/- {uncertainty}"
            value_text = _format_fit_parameter(
                parameter_name,
                function_type,
                value_text,
                value,
                uncertainty,
                kpc_per_arcsec,
            )
            component_lines.append(f"  {parameter_name:<16} {value_text}")
        components.append("\n".join(component_lines))

    return components


def _format_fit_parameter(
    parameter_name: str,
    function_type: str,
    value_text: str,
    value: float,
    uncertainty: str | None,
    kpc_per_arcsec: float | None,
) -> str:
    parameter_key = parameter_name.casefold()
    uncertainty_value = None
    if uncertainty is not None:
        try:
            parsed_uncertainty = float(uncertainty)
            if math.isfinite(parsed_uncertainty):
                uncertainty_value = abs(parsed_uncertainty)
        except ValueError:
            pass

    if parameter_key in RADIUS_PARAMETERS:
        value_text += " pix"
        radius_arcsec = value * PIXEL_SCALE_ARCSEC
        physical_text = f"{radius_arcsec:.6g}"
        if uncertainty_value is not None:
            physical_text += (
                f" +/- {uncertainty_value * PIXEL_SCALE_ARCSEC:.3g}"
            )
        physical_text += " arcsec"
        if kpc_per_arcsec is not None:
            radius_kpc = radius_arcsec * kpc_per_arcsec
            kpc_text = f"{radius_kpc:.6g}"
            if uncertainty_value is not None:
                kpc_text += (
                    f" +/- "
                    f"{uncertainty_value * PIXEL_SCALE_ARCSEC * kpc_per_arcsec:.3g}"
                )
            physical_text += f"; {kpc_text} kpc"
        return f"{value_text} ({physical_text})"

    is_gaussian_ring_amplitude = (
        function_type.casefold() == "gaussianring" and parameter_key == "a"
    )
    if parameter_key in INTENSITY_PARAMETERS or is_gaussian_ring_amplitude:
        value_text += " nmgy/pix"
        surface_brightness = _nanomaggies_per_pixel_to_mag_arcsec2(value)
        if surface_brightness is None:
            return f"{value_text} (mag/arcsec^2 undefined)"

        physical_text = f"{surface_brightness:.3f}"
        if uncertainty_value is not None and value > 0:
            magnitude_uncertainty = (
                2.5 / math.log(10) * uncertainty_value / value
            )
            physical_text += f" +/- {magnitude_uncertainty:.2g}"
        return f"{value_text} ({physical_text} mag/arcsec^2)"

    return value_text


def _nanomaggies_per_pixel_to_mag_arcsec2(
    nanomaggies_per_pixel: float,
) -> float | None:
    if not math.isfinite(nanomaggies_per_pixel) or nanomaggies_per_pixel <= 0:
        return None
    pixel_area = PIXEL_SCALE_ARCSEC**2
    return NANOMAGGIES_AB_ZEROPOINT - 2.5 * math.log10(
        nanomaggies_per_pixel / pixel_area
    )


def make_model_image(model: object, image_path: Path):
    import numpy as np
    import pyimfit
    from astropy.io import fits

    with fits.open(image_path, memmap=False) as image_hdul:
        image_shape = np.asarray(image_hdul[0].data).shape
    if len(image_shape) != 2:
        raise ValueError(f"Expected a 2D science image: {image_path}")

    band = image_path.stem.rsplit("_", 1)[-1]
    psf_path = image_path.parent / f"psf_patched_{band}.fits"
    if psf_path.exists():
        psf = fits.getdata(psf_path)
        fitter = pyimfit.Imfit(model, psf=psf, maxThreads=1)
    else:
        fitter = pyimfit.Imfit(model, maxThreads=1)

    model_image = np.asarray(fitter.getModelImage(shape=image_shape))
    if model_image.shape != image_shape:
        raise ValueError(
            f"PyImfit returned {model_image.shape} for {image_path}; "
            f"expected {image_shape}"
        )
    return model_image


def _format_surface_brightness(nanomaggies_per_pixel: float, _position: float) -> str:
    mag_per_arcsec_squared = _nanomaggies_per_pixel_to_mag_arcsec2(
        nanomaggies_per_pixel
    )
    if mag_per_arcsec_squared is None:
        return ""
    return f"{mag_per_arcsec_squared:.1f}"


def _show_image(
    axis,
    image,
    title: str,
    log_scale_factor: float = DEFAULT_LOG_SCALE_FACTOR,
    kpc_per_arcsec: float | None = None,
) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    from matplotlib.ticker import FuncFormatter, MaxNLocator
    import numpy as np

    if kpc_per_arcsec is None:
        axis.set_title(title, fontsize=11)
    else:
        axis.set_title(title, fontsize=11, pad=30)
    if image is None:
        axis.text(
            0.5,
            0.5,
            "Model unavailable",
            ha="center",
            va="center",
            transform=axis.transAxes,
            fontsize=10,
        )
        axis.set_axis_off()
        return

    if image.ndim != 2:
        raise ValueError(f"Expected a 2D image for {title}; got {image.shape}")
    positive_values = image[np.isfinite(image) & (image > 0)]
    if positive_values.size == 0:
        axis.text(
            0.5,
            0.5,
            "No positive pixels for log scale",
            ha="center",
            va="center",
            transform=axis.transAxes,
            fontsize=8,
        )
        axis.set_axis_off()
        return

    high = float(np.percentile(positive_values, 99.5))
    low = high / log_scale_factor
    height, width = image.shape
    extent = (
        -width * PIXEL_SCALE_ARCSEC / 2,
        width * PIXEL_SCALE_ARCSEC / 2,
        -height * PIXEL_SCALE_ARCSEC / 2,
        height * PIXEL_SCALE_ARCSEC / 2,
    )
    display_image = np.ma.masked_invalid(np.ma.masked_less_equal(image, 0))
    artist = axis.imshow(
        display_image,
        origin="lower",
        extent=extent,
        cmap="magma",
        norm=LogNorm(vmin=low, vmax=high),
        aspect="equal",
    )
    axis.set_xlabel('Relative x [arcsec]', fontsize=8)
    axis.set_ylabel('Relative y [arcsec]', fontsize=7)
    axis.xaxis.set_major_locator(MaxNLocator(4))
    axis.yaxis.set_major_locator(MaxNLocator(4))
    axis.tick_params(labelsize=7)
    _add_kpc_axes(axis, kpc_per_arcsec)
    colorbar = plt.colorbar(
        artist,
        ax=axis,
        orientation="horizontal",
        fraction=0.06,
        pad=0.24,
    )
    colorbar.ax.xaxis.set_major_formatter(
        FuncFormatter(_format_surface_brightness)
    )
    colorbar.ax.tick_params(axis="x", labelsize=5, pad=1)
    colorbar.ax.xaxis.set_label_position("top")
    colorbar.set_label(r"AB mag arcsec$^{-2}$", fontsize=5, labelpad=1)


def _show_jpg(
    axis, image, title: str, kpc_per_arcsec: float | None = None
) -> None:
    import matplotlib.ticker as ticker

    height, width = image.shape[:2]
    extent = (
        -width * PIXEL_SCALE_ARCSEC / 2,
        width * PIXEL_SCALE_ARCSEC / 2,
        -height * PIXEL_SCALE_ARCSEC / 2,
        height * PIXEL_SCALE_ARCSEC / 2,
    )
    axis.imshow(image, origin="lower", extent=extent, aspect="equal")
    if kpc_per_arcsec is None:
        axis.set_title(title, fontsize=11)
    else:
        axis.set_title(title, fontsize=11, pad=30)
    axis.set_xlabel('Relative x [arcsec]', fontsize=8)
    axis.set_ylabel('Relative y [arcsec]', fontsize=7)
    axis.xaxis.set_major_locator(ticker.MaxNLocator(4))
    axis.yaxis.set_major_locator(ticker.MaxNLocator(4))
    axis.tick_params(labelsize=7)
    _add_kpc_axes(axis, kpc_per_arcsec)


def _add_kpc_axes(axis, kpc_per_arcsec: float | None) -> None:
    from matplotlib.ticker import MaxNLocator

    if kpc_per_arcsec is None:
        return

    def arcsec_to_kpc(arcsec):
        return arcsec * kpc_per_arcsec

    def kpc_to_arcsec(kpc):
        return kpc / kpc_per_arcsec

    x_axis = axis.secondary_xaxis(
        "top", functions=(arcsec_to_kpc, kpc_to_arcsec)
    )
    x_axis.set_xlabel("x [kpc]", fontsize=6, labelpad=2)
    x_axis.xaxis.set_major_locator(MaxNLocator(4))
    x_axis.tick_params(labelsize=6, pad=1)

    y_axis = axis.secondary_yaxis(
        "right", functions=(arcsec_to_kpc, kpc_to_arcsec)
    )
    y_axis.set_ylabel("y [kpc]", fontsize=5, labelpad=2)
    y_axis.yaxis.set_major_locator(MaxNLocator(4))
    y_axis.tick_params(labelsize=5, pad=1)


def _show_parameters(axis, band: str, components: list[str], unavailable: str | None):
    axis.set_axis_off()
    axis.set_title(f"{band.upper()}-band fit parameters", loc="left", fontsize=11)
    if components:
        text = "\n\n".join(components)
    else:
        text = unavailable or "No fit parameter file found."

    line_count = text.count("\n") + 1
    fontsize = min(8.5, max(5.5, 170 / line_count))
    axis.text(
        0,
        0.94,
        text,
        transform=axis.transAxes,
        va="top",
        fontsize=fontsize,
        family="monospace",
        linespacing=1.15,
    )


def create_page(
    galaxy_directory: Path,
    fit_type: str,
    log_scale_factor: float,
    kpc_per_arcsec: float | None,
    pdf,
) -> None:
    import matplotlib.pyplot as plt
    import pyimfit
    from astropy.io import fits

    fig = plt.figure(figsize=(17, 9))
    grid = fig.add_gridspec(2, 10, height_ratios=(1.35, 1), hspace=0.4, wspace=1.8)
    axes = [fig.add_subplot(grid[0, 2 * i : 2 * i + 2]) for i in range(5)]
    parameter_axes = [
        fig.add_subplot(grid[1, 0:5]),
        fig.add_subplot(grid[1, 5:10]),
    ]

    title = galaxy_directory.name
    if kpc_per_arcsec is not None:
        title += f" (scale: {kpc_per_arcsec:g} kpc/arcsec)"
    fig.suptitle(title, fontsize=16, y=0.98)
    jpg_path = galaxy_directory / "image.jpg"
    if jpg_path.exists():
        _show_jpg(
            axes[0], plt.imread(jpg_path), "Galaxy JPG", kpc_per_arcsec
        )
    else:
        _show_image(axes[0], None, "Galaxy JPG")
        axes[0].texts[0].set_text("image.jpg not found")

    for band_index, (band, parameter_axis) in enumerate(zip(BANDS, parameter_axes)):
        data_axis = axes[1 + 2 * band_index]
        model_axis = axes[2 + 2 * band_index]
        parameter_path = galaxy_directory / f"{fit_type}_{band}_fit_params.txt"
        image_path = galaxy_directory / f"image_{band}.fits"
        model_image = None
        band_image = None
        components = []
        unavailable = None

        if image_path.exists():
            band_image = fits.getdata(image_path)
            if band_image.ndim != 2:
                raise ValueError(f"Expected a 2D science image: {image_path}")
        _show_image(
            data_axis,
            band_image,
            f"{band.upper()}-band image",
            log_scale_factor,
            kpc_per_arcsec,
        )

        if parameter_path.exists():
            model = pyimfit.parse_config_file(str(parameter_path))
            components = read_components(
                parameter_path, model, kpc_per_arcsec
            )
            if band_image is not None:
                model_image = make_model_image(model, image_path)
            else:
                unavailable = f"{image_path.name} not found; model not generated."
        else:
            unavailable = f"{parameter_path.name} not found."

        _show_image(
            model_axis,
            model_image,
            f"{band.upper()}-band PyImfit model",
            log_scale_factor,
            kpc_per_arcsec,
        )
        if band_image is None:
            data_axis.texts[0].set_text(f"{image_path.name} not found.")
        if model_image is None:
            model_axis.texts[0].set_text(unavailable)
        _show_parameters(parameter_axis, band, components, unavailable)

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def create_report(
    root: Path,
    output: Path,
    fit_type: str,
    recursive: bool,
    log_scale_factor: float = DEFAULT_LOG_SCALE_FACTOR,
    scale_table_path: Path | None = None,
) -> int:
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib.backends.backend_pdf import PdfPages

    if not root.is_dir():
        raise NotADirectoryError(f"Galaxy root is not a directory: {root}")

    galaxy_directories = find_galaxy_directories(root, fit_type, recursive)
    if not galaxy_directories:
        raise FileNotFoundError(
            f"No {fit_type} G/R-band fit parameter files found under {root}"
        )

    galaxy_scales = (
        read_scale_table(scale_table_path) if scale_table_path is not None else {}
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(output) as pdf:
        for galaxy_directory in galaxy_directories:
            kpc_per_arcsec = galaxy_scales.get(galaxy_directory.name.casefold())
            create_page(
                galaxy_directory,
                fit_type,
                log_scale_factor,
                kpc_per_arcsec,
                pdf,
            )
    return len(galaxy_directories)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Create one PDF page per galaxy with its JPG, generated G/R PyImfit "
            "models, and best-fit parameter values."
        )
    )
    parser.add_argument(
        "-p",
        "--path",
        type=Path,
        default=Path("."),
        help="Galaxy folder, or root folder when using --recursive",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("fit_report.pdf"),
        help="Output PDF path",
    )
    parser.add_argument(
        "--fit-type",
        choices=FIT_TYPES,
        default="2_sersic",
        help="Fit type used in the parameter filenames",
    )
    parser.add_argument(
        "-r",
        "--recursive",
        action="store_true",
        help="Find galaxy folders recursively under --path",
    )
    parser.add_argument(
        "--log-scale-factor",
        type=float,
        default=DEFAULT_LOG_SCALE_FACTOR,
        help=(
            "Log color scale range as vmax/vmin; increase to reveal fainter "
            "pixels (must be greater than 1)"
        ),
    )
    parser.add_argument(
        "--scale-table",
        type=Path,
        help="CSV with NAME and SCALE (kpc/arcsec) columns for physical size axes",
    )
    args = parser.parse_args()

    if not math.isfinite(args.log_scale_factor) or args.log_scale_factor <= 1:
        parser.error("--log-scale-factor must be finite and greater than 1")

    page_count = create_report(
        args.path.resolve(),
        args.output.resolve(),
        args.fit_type,
        args.recursive,
        args.log_scale_factor,
        args.scale_table.resolve() if args.scale_table is not None else None,
    )
    print(f"Created {args.output.resolve()} with {page_count} galaxy pages.")


if __name__ == "__main__":
    main()
