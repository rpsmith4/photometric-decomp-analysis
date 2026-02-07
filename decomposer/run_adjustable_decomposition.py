## Creates display. This will be the main function in this project. 


# When called:
# - Create GUI display
# - Allow for adjusting of guess parameters
# - Display the .fits files of an original image, the model fit, and the residuals
# - Show initial guess parameters, plus data from them (such as the SB profiles of the 1-D photometric cuts for estimating semimajor axis)

# To do this, need to generate initial parameterization (make a separate function entirely, input initial parameters into a text file to read in)
# 
"""
Imfit runner + report formatter + optional DS9 viewer.

Features
--------
1) Runs Imfit with streamed console output (PTY on POSIX for responsive logs).
2) Parses the initial .imfit parameter file and Imfit's saved best-fit file.
3) Detects parameters that hit constraints and annotates them.
4) Rewrites bestfit_params.txt in-place with a clean, human-readable layout:
   - Unique Imfit metadata preserved at the top (commented).
   - A warnings section listing any bound hits.
   - Parameters rendered in the same order as the initial .imfit file.
   - Position angle comments normalized to the +y convention.
5) (Optional) Opens science/model/residual FITS in DS9 with chosen tiling,
   scale, colormap, zoom, and explicit scale limits. If ds9_exe is not given,
   common macOS/Windows/Linux install paths are tried automatically.

Typical Use
-----------
result = run_imfit(
    science_fits="image.fits",
    params_path="two_sersic.imfit",
    psf_fits="psf.fits",
    noise_fits="invvar.fits",
    mask_fits="mask.fits",
    outdir="out/",
    ds9_show=True, ds9_scale="zscale", ds9_cmap="grey", ds9_tile=True
)
"""

from pathlib import Path
import subprocess
import shutil
import sys
import os
import re
from math import isfinite
import textwrap

# =========================
# Parsing & formatting utils
# =========================

# Robust float regex used to detect/parse numeric tokens, including scientific notation.
_FLOAT_RE = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"

def _parse_param_file(initial_path: Path):
    """
    Parse an initial .imfit file to extract:
      - 'order': a sequence describing the layout for pretty rendering
      - 'init_map': per-parameter initial values and bounds keyed by (component_index, NAME_UC)

    Returns
    -------
    order : list[tuple]
        Elements are:
          ("blank",)
          ("function",)
          ("param", original_name, right_comment)
          ("literal", full_line)
    init_map : dict[(int, str) -> dict]
        Keys are (component_index, NAME_UC) -> {"init": float, "lo": float|None, "hi": float|None}
    """
    order = []
    init_map = {}
    comp = -1
    with open(initial_path, "r") as f:
        for raw in f:
            line = raw.rstrip("\n")
            if line.strip() == "":
                order.append(("blank",))
                continue
            if line.lstrip().upper().startswith("FUNCTION"):
                comp += 1
                order.append(("function",))
                continue

            left, sep, right_comment = line.partition("#")
            right_comment = right_comment.strip() if sep else ""
            content = left.strip()
            parts = content.split()
            if len(parts) < 2:
                order.append(("literal", line))
                continue

            name = parts[0]
            name_uc = name.upper()
            mval = re.match(_FLOAT_RE, parts[1])
            if not mval:
                order.append(("literal", line))
                continue
            init_val = float(mval.group(0))

            rest = " ".join(parts[2:]).strip()
            lo = hi = None
            # Infer bounds unless the parameter is explicitly fixed.
            if rest and "fixed" not in rest.lower():
                nums = re.findall(_FLOAT_RE, rest)
                if len(nums) >= 2:
                    lo, hi = float(nums[0]), float(nums[1])

            init_map[(comp, name_uc)] = {"init": init_val, "lo": lo, "hi": hi}
            order.append(("param", name, right_comment))
    return order, init_map

def _extract_after_value(raw_line: str) -> str:
    """
    Return all tokens after the first numeric value on a parameter line (before '#').

    This preserves supplemental annotations that may carry units or flags.
    """
    line = raw_line.rstrip("\n")
    nohash, _, _comment = line.partition("#")
    tokens = nohash.strip().split()
    if len(tokens) < 2:
        return ""
    # Find the first numeric token after the name.
    val_idx = None
    for i, tok in enumerate(tokens[1:], start=1):
        if re.match(_FLOAT_RE, tok or ""):
            val_idx = i
            break
    if val_idx is None:
        return ""
    tail = " ".join(tokens[val_idx+1:]).replace("\t", " ").strip()
    return tail

def _parse_bestfit_file_with_annotations(bestfit_path: Path):
    """
    Parse Imfit's --save-params output, keeping parameter values and free-form annotations.

    Returns
    -------
    fit_map : dict[(int, str) -> {"fit": float, "ann": str}]
        'ann' includes inline text after the numeric value and any trailing '# comment'.
    metadata_lines : list[str]
        Lines that are not recognized as parameter lines (stats, FUNCTION blocks, blanks, notes).
    """
    comp = -1
    fit_map = {}
    metadata_lines = []
    with open(bestfit_path, "r") as f:
        lines = [ln.rstrip("\n") for ln in f]

    for s in lines:
        stripped = s.split("#", 1)[0].strip()
        if stripped == "":
            metadata_lines.append(s)
            continue
        if stripped.upper().startswith("FUNCTION"):
            comp += 1
            metadata_lines.append(s)
            continue

        parts = stripped.split()
        if len(parts) >= 2 and re.match(_FLOAT_RE, parts[1] or ""):
            name = parts[0]
            name_uc = name.upper()
            mval = re.match(_FLOAT_RE, parts[1])
            if not mval:
                metadata_lines.append(s)
                continue
            fit_val = float(mval.group(0))

            tail_ann = _extract_after_value(s)
            trailing_comment = s.split("#", 1)[1].strip() if "#" in s else ""
            ann = " ".join(x for x in [tail_ann, trailing_comment] if x).strip()
            ann = re.sub(r"\s+", " ", ann)
            fit_map[(comp, name_uc)] = {"fit": fit_val, "ann": ann}
        else:
            metadata_lines.append(s)

    return fit_map, metadata_lines

def _near(a, b, rtol=1e-6, atol=1e-6):
    """Numerical near-equality check for floats; tolerant to catch bound hits."""
    if a is None or b is None:
        return False
    if not (isfinite(a) and isfinite(b)):
        return False
    return abs(a - b) <= max(atol, rtol * max(1.0, abs(a), abs(b)))

def _pretty_param_name(component_index: int, short: str) -> str:
    """
    Produce a human-friendly label for parameters, with component context.

    Position angle (PA) is documented as degrees CCW from +y to avoid frame ambiguity.
    """
    comp_prefix = {
        -1: "Shared ",
         0: "Host ",
         1: "Polar ",
    }.get(component_index, f"Component {component_index+1} ")

    base = short.upper()
    if base == "X0":  return f"{comp_prefix}Center X (pixels)"
    if base == "Y0":  return f"{comp_prefix}Center Y (pixels)"
    if base == "PA":  return f"{comp_prefix}Position Angle (deg CCW from +y)"
    if base == "ELL": return f"{comp_prefix}Ellipticity (1 - b/a)"
    if base == "N":   return f"{comp_prefix}Sérsic Index"
    if base == "I_E": return f"{comp_prefix}Effective Intensity (counts/pixel)"
    if base == "R_E": return f"{comp_prefix}Effective Radius (pixels)"
    return f"{comp_prefix}{short}"

def _pad_comment(line: str, comment: str, width: int = 40) -> str:
    """Left-pad parameter lines so inline comments align at a common column."""
    return f"{line:<{width}}# {comment}"

def _normalize_pa_annotation(ann: str) -> str:
    """
    Remove Imfit's 'CCW from +x/+y' clarifiers from PA annotations; keep units/notes.
    We standardize to a +y frame elsewhere.
    """
    if not ann:
        return ann
    ann = re.sub(r"\s*\(CCW from \+\s*[xy]\s*(axis)?\)\s*", "", ann, flags=re.IGNORECASE)
    ann = re.sub(r"\s*\(degrees?\s*CCW\s*from\s*\+\s*[xy]\s*(axis)?\)\s*", "", ann, flags=re.IGNORECASE)
    return ann.strip()

def _force_pa_comment_to_plus_y(comment: str) -> str:
    """
    Ensure any PA-related initial comment mentions '+y' and is explicit about direction.
    """
    if not comment:
        return comment
    comment = re.sub(r"\+x\b", "+y", comment, flags=re.IGNORECASE)
    comment = re.sub(r"\(.*CCW.*from\s*\+\s*x.*\)", "(degrees CCW from +y)", comment, flags=re.IGNORECASE)
    if "PA" in comment.upper() and "CCW" not in comment.upper():
        comment = comment + " (degrees CCW from +y)"
    return comment

# ==========================================
# Pretty writer (metadata, warnings, +y PA)
# ==========================================

def _rewrite_bestfit_pretty_overwrite(
    initial_path: Path,
    bestfit_path: Path,
    hits_list,                  # list[(label, init_val, "lower bound X"|"upper bound Y")]
    hits_by_key,                # dict[(comp_idx, NAME_UC) -> "HIT ..."]
    pad_width: int = 40
):
    """
    Overwrite bestfit_path with a readable file that:
      - Preserves unique metadata at the top (commented).
      - Adds a warnings block for any parameters that hit constraint bounds.
      - Writes parameters in the initial .imfit order with best-fit values only.
      - Annotates PA to +y convention and harmonizes PA comments.
    """
    order, _init_map = _parse_param_file(initial_path)
    fit_map, metadata_lines = _parse_bestfit_file_with_annotations(bestfit_path)

    header = [
        "# Two-component Sérsic model (pixels; PA in deg CCW from +y)",
        "# Each FUNCTION block defines one Sérsic component.",
        "# Parameter syntax:",
        "#     param  value                → free parameter",
        "#     param  value lower,upper    → constrained between limits",
        "#     param  value fixed          → fixed at value",
        "",
    ]

    out_lines = []

    # Metadata block
    if metadata_lines:
        out_lines.append("# === Imfit metadata (from original bestfit_params.txt) ===")
        out_lines.extend([f"# {ln}" if ln != "" else "#" for ln in metadata_lines])
        out_lines.append("")

    # Bound-hit warnings
    if hits_list:
        out_lines.append("# === Solver constraint warnings (detected by wrapper) ===")
        out_lines.append("# The solver appears to have hit the following constraint boundaries:")
        for label, init_val, bound_txt in hits_list:
            out_lines.append(f"#  - {label}: {init_val} -> {bound_txt}")
        out_lines.append("")

    # Standard header
    out_lines.extend(header)

    # Parameters in the initial layout order
    comp_idx = -1
    for item in order:
        tag = item[0]
        if tag == "blank":
            out_lines.append("")
            continue
        if tag == "literal":
            out_lines.append(item[1])
            continue
        if tag == "function":
            comp_idx += 1
            out_lines.append("FUNCTION Sersic")
            continue

        # Parameter rows
        orig_name, initial_comment = item[1], item[2]
        key_uc = (comp_idx, orig_name.upper())
        if key_uc not in fit_map:
            core = f"{orig_name} NaN"
            pieces = ["missing in bestfit"]
            if initial_comment:
                if orig_name.upper() == "PA":
                    initial_comment = _force_pa_comment_to_plus_y(initial_comment)
                pieces.append(initial_comment)
            out_lines.append(_pad_comment(core, " | ".join(pieces), pad_width))
            continue

        fit_val = fit_map[key_uc]["fit"]
        ann = fit_map[key_uc]["ann"]

        if orig_name.upper() == "PA":
            ann = _normalize_pa_annotation(ann)
            if initial_comment:
                initial_comment = _force_pa_comment_to_plus_y(initial_comment)

        core = f"{orig_name} {fit_val:.6f}"
        hit_note = hits_by_key.get(key_uc)

        # If a bound was hit, keep a dedicated line for the note and another for any annotations.
        if hit_note:
            out_lines.append(_pad_comment(core, hit_note, pad_width))
            rem_pieces = []
            if ann:
                rem_pieces.append(ann)
            if initial_comment:
                rem_pieces.append(initial_comment)
            if rem_pieces:
                out_lines.append(_pad_comment("", " | ".join(rem_pieces), pad_width))
        else:
            pieces = []
            if ann:
                pieces.append(ann)
            if initial_comment:
                pieces.append(initial_comment)
            if pieces:
                out_lines.append(_pad_comment(core, " | ".join(pieces), pad_width))
            else:
                out_lines.append(core)

    pretty_text = textwrap.dedent("\n".join(out_lines)).strip() + "\n"
    bestfit_path.write_text(pretty_text)
# === DS9: simple, reliable tiling (revert) ===

def _guess_ds9_path():
    """
    Try to find SAOImage DS9 in common macOS, Linux, and Windows installation paths.
    Returns the path if found, else None.
    """
    candidates = [
        # macOS app bundle
        "/Applications/SAOImageDS9.app/Contents/MacOS/ds9",
        # Homebrew / MacPorts / Unix-like installs
        "/opt/homebrew/bin/ds9",
        "/usr/local/bin/ds9",
        "/opt/local/bin/ds9",
        "/usr/bin/ds9",
        # Windows installers
        r"C:\SAOImageDS9\ds9.exe",
        r"C:\Program Files\SAOImageDS9\ds9.exe",
    ]
    for p in candidates:
        if Path(p).exists():
            return p
    return None


def _show_in_ds9(
    science=None,
    model=None,
    residual=None,
    *,
    ds9_exe=None,              # None => auto-find via _guess_ds9_path()
    show_science=True,
    show_model=True,
    show_residual=True,
    tile=True,
    scale="zscale",            # DS9 may still default to 'minmax' on initial load on some builds
    cmap="grey",
    scale_limits=None,         # (lo, hi) forces linear+limits if provided
    zoom=None,                 # numeric zoom (1, 2, 4, ...) or None
    extra_args=None
):
    """
    Open DS9 in the straightforward, reliable way:
      - Pass images positionally so DS9 creates one frame per image automatically.
      - Apply '-tile yes' so frames appear side-by-side.
      - Apply '-global yes' so subsequent view settings affect all frames.
    This restores the behavior where tiles reliably appear.
    """
    # Resolve DS9 path
    if ds9_exe is None:
        ds9_exe = _guess_ds9_path()
    if ds9_exe is None or (not Path(str(ds9_exe)).exists() and shutil.which(str(ds9_exe)) is None):
        raise FileNotFoundError(
            "Could not locate SAOImage DS9. Install it or pass an explicit path to 'ds9_exe'."
        )

    # Collect images
    images = []
    if show_science and science and Path(science).exists():
        images.append(str(science))
    if show_model and model and Path(model).exists():
        images.append(str(model))
    if show_residual and residual and Path(residual).exists():
        images.append(str(residual))

    if not images:
        raise ValueError("No images to display in DS9 (none selected or missing).")

    # Build command (simple + reliable ordering)
    cmd = [str(ds9_exe)]
    cmd += images
    if tile:
        cmd += ["-tile", "yes"]
    cmd += ["-global", "yes"]

    # View settings (kept simple to avoid interfering with frame creation)
    if scale_limits is not None and len(scale_limits) == 2:
        cmd += ["-scale", "linear", "-scale", "limits", str(scale_limits[0]), str(scale_limits[1])]
    elif scale:
        cmd += ["-scale", str(scale)]
    if cmap:
        cmd += ["-cmap", str(cmap)]
    if zoom is not None:
        cmd += ["-zoom", str(zoom)]

    if extra_args:
        cmd += list(extra_args)

    subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

# ===============================
# Streaming (PTY on POSIX) runner
# ===============================

def _run_with_pty(cmd):
    """
    Run a command attached to a PTY to get line-buffered, responsive stdout on POSIX.
    Returns (returncode, captured_text) while echoing output to the console.
    """
    import pty, select, codecs

    master_fd, slave_fd = pty.openpty()
    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.DEVNULL,
        stdout=slave_fd,
        stderr=slave_fd,
        close_fds=True,
        text=False,
    )
    os.close(slave_fd)

    decoder = codecs.getincrementaldecoder(sys.getdefaultencoding())(errors="replace")
    captured = []
    try:
        while True:
            r, _, _ = select.select([master_fd], [], [], 0.1)
            if master_fd in r:
                try:
                    data = os.read(master_fd, 4096)
                except OSError:
                    data = b""
                if not data:
                    if proc.poll() is not None:
                        break
                else:
                    text = decoder.decode(data)
                    captured.append(text)
                    sys.stdout.write(text); sys.stdout.flush()
            if proc.poll() is not None:
                try:
                    while True:
                        data = os.read(master_fd, 4096)
                        if not data: break
                        text = decoder.decode(data)
                        captured.append(text)
                        sys.stdout.write(text); sys.stdout.flush()
                except OSError:
                    pass
                break
    finally:
        try:
            os.close(master_fd)
        except OSError:
            pass

    return proc.returncode, "".join(captured)

# =================
# Public API
# =================

def _pretty_param_label(comp_idx: int, short: str) -> str:
    """Public wrapper around _pretty_param_name for external callers."""
    return _pretty_param_name(comp_idx, short)
def run_imfit(
    science_fits,
    params_text=None,        # if provided, written to params_path before running
    params_path=None,        # path to an existing .imfit/.txt parameter file
    psf_fits=None,
    mask_fits=None,
    noise_fits=None,         # e.g., inverse-variance/variance/sigma map
    noise_kind="invvar",     # one of: "invvar", "variance", "sigma", or None
    outdir=".",
    sky=None,                # counts/pixel
    gain=None,               # e-/ADU
    readnoise=None,          # e-
    imfit_exe="imfit",
    extra_args=None,         # any extra flags for Imfit
    pretty_pad=40,           # column where inline comments begin
    fltr = "r",
    zeropoint=None,
    pixel_scale=None,

    # ---- DS9 viewing controls ----
    ds9_show=False,          # open DS9 after fit completes
    ds9_exe=None,            # None => auto-locate DS9 via _guess_ds9_path()
    ds9_show_science=True,
    ds9_show_model=True,
    ds9_show_residual=True,
    ds9_tile=True,
    ds9_scale="zscale",
    ds9_cmap="grey",
    ds9_scale_limits=None,   # (lo, hi) to override scaling with explicit limits
    ds9_zoom=None,
    ds9_extra_args=None,
):
    """
    Run a 2-D photometric decomposition with Imfit, rewrite the report cleanly,
    and optionally visualize outputs in DS9.

    DS9 Modes
    ---------
    - ds9_per_frame=False (default): simple, reliable tiling. DS9 creates one frame per image,
      we apply shared view settings once (some DS9 builds may keep 'minmax' on load).
    - ds9_per_frame=True: load each image into a new frame and apply scale/cmap/zoom right after
      load, which helps ensure 'zscale' (or chosen scale) sticks per frame.
    """
    # --- sanity checks ---
    if shutil.which(imfit_exe) is None:
        raise FileNotFoundError(
            f"Could not find '{imfit_exe}' on PATH. Install Imfit or pass an absolute path."
        )

    outdir = Path(outdir); outdir.mkdir(parents=True, exist_ok=True)
    science_fits = Path(science_fits)
    if not science_fits.exists():
        raise FileNotFoundError(f"Science image not found: {science_fits}")

    # Params file: either write given text or use an existing file.
    if params_text is not None:
        params_path = Path(params_path or (outdir / "two_sersic.imfit"))
        params_path.write_text(params_text)
    elif params_path is not None:
        params_path = Path(params_path)
        if not params_path.exists():
            raise FileNotFoundError(f"Parameter file not found: {params_path}")
    else:
        raise ValueError("Provide either params_text or params_path.")

    # Output paths
    model_path = outdir / "model.fits"
    resid_path = outdir / "residual.fits"
    bestfit_txt = outdir / f"bestfit_params_{fltr}.txt"

    # --- build Imfit command ---
    cmd = [
        imfit_exe,
        str(science_fits),
        "-c", str(params_path),
        "--save-model", str(model_path),
        "--save-residual", str(resid_path),
        "--save-params", str(bestfit_txt),
    ]

    if psf_fits:
        cmd += ["--psf", str(psf_fits)]
    if mask_fits:
        cmd += ["--mask", str(mask_fits)]

    # Noise / error handling flags depend on the map's meaning.
    if noise_fits:
        if noise_kind == "invvar":
            cmd += ["--noise", str(noise_fits), "--errors-are-weights"]
        elif noise_kind == "variance":
            cmd += ["--noise", str(noise_fits), "--errors-are-variances"]
        elif noise_kind == "sigma":
            cmd += ["--noise", str(noise_fits)]
        else:
            raise ValueError("noise_kind must be one of: 'invvar', 'variance', 'sigma'")

    # Optional detector/statistics info.
    if sky is not None:
        cmd += ["--sky", str(sky)]
    if gain is not None:
        cmd += ["--gain", str(gain)]
    if readnoise is not None:
        cmd += ["--readnoise", str(readnoise)]

    if extra_args:
        cmd += list(extra_args)
    
   


    # --- run (streaming logs) ---
    print("Running:", " ".join(cmd))
    if os.name != "nt":
        return_code, _captured = _run_with_pty(cmd)
        stderr_text = ""
    else:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True,
        )
        for line in proc.stdout:
            sys.stdout.write(line); sys.stdout.flush()
        proc.stdout.close()
        return_code = proc.wait()
        stderr_text = ""

    if return_code != 0:
        raise RuntimeError(f"Imfit failed with return code {return_code}")

    # --- detect constraint hits by comparing best-fit values to declared bounds ---
    hits_list = []          # list of (label, init_val, "lower bound X"/"upper bound Y")
    hits_by_key = {}        # (comp_idx, NAME_UC) -> message for inline annotation
    try:
        if bestfit_txt.exists():
            _order, init_map = _parse_param_file(params_path)
            fit_map, meta_lines = _parse_bestfit_file_with_annotations(bestfit_txt)

            for (comp_idx, name_uc), info in init_map.items():
                lo, hi = info["lo"], info["hi"]
                if lo is None and hi is None:
                    continue
                fit = fit_map.get((comp_idx, name_uc), {}).get("fit")
                if fit is None:
                    continue
                if lo is not None and _near(fit, lo):
                    label = _pretty_param_name(comp_idx, name_uc)
                    note = f"HIT {label} \u2192 lower bound {lo}"
                    hits_list.append((label, info["init"], f"lower bound {lo}"))
                    hits_by_key[(comp_idx, name_uc)] = note
                elif hi is not None and _near(fit, hi):
                    label = _pretty_param_name(comp_idx, name_uc)
                    note = f"HIT {label} \u2192 upper bound {hi}"
                    hits_list.append((label, info["init"], f"upper bound {hi}"))
                    hits_by_key[(comp_idx, name_uc)] = note

            if hits_list:
                print("\n⚠️ Constraint warnings")
                print("The solver appears to have hit the following boundaries:")
                for label, init_val, bound_txt in hits_list:
                    print(f" - {label}: {init_val} -> {bound_txt}")
                print()
    except Exception as e:
        print(f"[constraint-check error] {e}")


    # --- rewrite bestfit_params.txt (pretty layout, +y PA, warnings/metadata at top) ---
    try:
        if bestfit_txt.exists():
            _rewrite_bestfit_pretty_overwrite(
                initial_path=params_path,
                bestfit_path=bestfit_txt,
                hits_list=hits_list,
                hits_by_key=hits_by_key,
                pad_width=pretty_pad
            )
            print(f"[info] Rewrote best-fit params (pretty, +y PA): {bestfit_txt}")
    except Exception as e:
        print(f"[bestfit-format error] {e}")
    
    # --- Append derived μ_e and R_e(kpc) after pretty formatting ---
    try:
        if pixel_scale is not None and zeropoint is not None:
            from table_info import get_galaxy_info
            galname = science_fits.parent.name
            info = get_galaxy_info(galname)
            kpc_per_arcsec = float(info.get("scale", 0.0))

            fit_map, _ = _parse_bestfit_file_with_annotations(bestfit_txt)

            derived = []
            derived.append("")
            derived.append("# === Derived photometric quantities ===")

            for (comp_idx, name_uc), entry in fit_map.items():
                if name_uc == "I_E":
                    Ie_pix = entry["fit"]
                    Ie_as2 = Ie_pix / (pixel_scale**2)

                    from math import log10
                    mu_e = zeropoint - 2.5 * log10(Ie_as2)

                    # match Re
                    re_entry = fit_map.get((comp_idx, "R_E"))
                    if re_entry:
                        Re_pix = re_entry["fit"]
                        Re_arcsec = Re_pix * pixel_scale
                        Re_kpc = Re_arcsec * kpc_per_arcsec
                    else:
                        Re_kpc = float("nan")

                    label = ("Host" if comp_idx == 0 
                             else "Polar" if comp_idx == 1 
                             else f"Component {comp_idx+1}")

                    derived.append(f"# {label}:")
                    derived.append(f"#   μ_e = {mu_e:.3f} mag/arcsec^2")
                    derived.append(f"#   R_e = {Re_kpc:.3f} kpc   (scale={kpc_per_arcsec} kpc/arcsec)")
                    derived.append("")

            with open(bestfit_txt, "a") as f:
                f.write("\n".join(derived) + "\n")

            print("[info] Added μ_e and R_e(kpc) to best-fit file.")

    except Exception as e:
        print(f"[derived-append error] {e}")


    # --- optional DS9 viewer ---
    if ds9_show:
        try:
            _show_in_ds9(
                science=science_fits,
                model=model_path,
                residual=resid_path,
                ds9_exe=ds9_exe,                 # None allowed; auto-discovery will run
                show_science=ds9_show_science,
                show_model=ds9_show_model,
                show_residual=ds9_show_residual,
                tile=ds9_tile,
                scale=ds9_scale,
                cmap=ds9_cmap,
                scale_limits=ds9_scale_limits,
                zoom=ds9_zoom,
                extra_args=ds9_extra_args,
            )
        except Exception as e:
            print(f"[ds9 error] {e}")

    return {
        "model_fits": model_path,
        "residual_fits": resid_path,
        "bestfit_params": bestfit_txt,  # formatted in-place
        "params_path": params_path,
        "returncode": return_code,
        "stderr_text": stderr_text,     # empty on POSIX PTY path
    }


# =============
# Optional test
# =============
def main():
    """
    Minimal test hook:
      - Uses user helpers to locate a test galaxy and parameter file.
      - Runs the fit and (optionally) opens DS9 for quick inspection.
    """
    import glob
    import table_info
    from initial_parameterization import gather_parameters
    from initial_parameterization import get_galaxy_files

    table_info.set_directory()

    GalaxyDirectories = glob.glob("./GalaxyFiles/*")
    import os
    filenames = [os.path.basename(f) for f in GalaxyDirectories]
    if not filenames:
        raise RuntimeError("No ./GalaxyFiles/* directories found for testing.")
    test_galaxy = filenames[0]

    filter = "r"
    output_path, pixel_scale, zeropoint = gather_parameters(test_galaxy, fltr=filter)


    galaxy_files = get_galaxy_files(test_galaxy, fltr=filter)
    science_file = galaxy_files["science"]
    psf_file = galaxy_files["psf"]
    noise_file = galaxy_files["invvar"]
    mask_file = galaxy_files["mask"]

    output_dir = "./GalaxyFiles/" + test_galaxy

    return run_imfit(
        science_fits=science_file,
        params_path=output_path,
        psf_fits=psf_file,
        mask_fits=mask_file,
        noise_fits=noise_file,
        outdir=output_dir,
        pretty_pad=28,
        ds9_show=True,
        ds9_tile=True,
        ds9_scale="zscale",
        ds9_cmap="grey",
        fltr=filter,
        pixel_scale=pixel_scale,
        zeropoint=zeropoint,
    )


if __name__ == "__main__":
    _ = main()
