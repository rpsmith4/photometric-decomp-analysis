#!/usr/bin/env python3
"""Create a per-galaxy PDF report with JPGs, PyImfit models, and fit parameters."""

import argparse
import re
from pathlib import Path


BANDS = ("g", "r")
FIT_TYPES = ("2_sersic", "1_sersic_1_gauss_ring", "3_sersic")
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


def read_components(parameter_path: Path, model: object) -> list[str]:
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
            component_lines.append(f"  {parameter_name:<16} {value_text}")
        components.append("\n".join(component_lines))

    return components


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


def _show_image(axis, image, title: str) -> None:
    import numpy as np

    axis.set_title(title, fontsize=11)
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

    finite_values = image[np.isfinite(image)]
    if finite_values.size == 0:
        raise ValueError(f"Cannot plot an image with no finite pixels: {title}")
    low, high = np.percentile(finite_values, (1, 99.5))
    if high <= low:
        high = low + 1
    axis.imshow(image, origin="lower", cmap="magma", vmin=low, vmax=high)
    axis.set_axis_off()


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


def create_page(galaxy_directory: Path, fit_type: str, pdf) -> None:
    import matplotlib.pyplot as plt
    import pyimfit

    fig = plt.figure(figsize=(11.69, 8.27))
    grid = fig.add_gridspec(2, 6, height_ratios=(1.05, 1), hspace=0.25)
    axes = [
        fig.add_subplot(grid[0, 0:2]),
        fig.add_subplot(grid[0, 2:4]),
        fig.add_subplot(grid[0, 4:6]),
    ]
    parameter_axes = [
        fig.add_subplot(grid[1, 0:3]),
        fig.add_subplot(grid[1, 3:6]),
    ]

    fig.suptitle(galaxy_directory.name, fontsize=16, y=0.98)
    jpg_path = galaxy_directory / "image.jpg"
    if jpg_path.exists():
        axes[0].imshow(plt.imread(jpg_path))
        axes[0].set_title("Galaxy JPG", fontsize=11)
        axes[0].set_axis_off()
    else:
        _show_image(axes[0], None, "Galaxy JPG")
        axes[0].texts[0].set_text("image.jpg not found")

    for axis, band, parameter_axis in zip(axes[1:], BANDS, parameter_axes):
        parameter_path = galaxy_directory / f"{fit_type}_{band}_fit_params.txt"
        image_path = galaxy_directory / f"image_{band}.fits"
        model_image = None
        components = []
        unavailable = None

        if parameter_path.exists():
            model = pyimfit.parse_config_file(str(parameter_path))
            components = read_components(parameter_path, model)
            if image_path.exists():
                model_image = make_model_image(model, image_path)
            else:
                unavailable = f"{image_path.name} not found; model not generated."
        else:
            unavailable = f"{parameter_path.name} not found."

        _show_image(axis, model_image, f"{band.upper()}-band PyImfit model")
        if model_image is None:
            axis.texts[0].set_text(unavailable)
        _show_parameters(parameter_axis, band, components, unavailable)

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def create_report(root: Path, output: Path, fit_type: str, recursive: bool) -> int:
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

    output.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(output) as pdf:
        for galaxy_directory in galaxy_directories:
            create_page(galaxy_directory, fit_type, pdf)
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
    args = parser.parse_args()

    page_count = create_report(
        args.path.resolve(),
        args.output.resolve(),
        args.fit_type,
        args.recursive,
    )
    print(f"Created {args.output.resolve()} with {page_count} galaxy pages.")


if __name__ == "__main__":
    main()
