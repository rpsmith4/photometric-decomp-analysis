#!/usr/bin/env python3
"""Submit and run one IMFIT-MCMC Slurm array task per galaxy directory."""

import argparse
from functools import partial
from pathlib import Path
import subprocess

import submitit
import os


def find_galaxies(root: Path, fit_type: str, bands: list[str]) -> list[Path]:
    """Find directories with at least one requested image/config pair."""
    galaxy_dirs = set()
    for image in root.rglob("image_?.fits"):
        band = image.stem[-1]
        if band in bands and (image.parent / f"{fit_type}_{band}.dat").is_file():
            galaxy_dirs.add(image.parent.resolve())
    return sorted(galaxy_dirs)


def run_galaxy(
    galaxy_path: str,
    fit_type: str,
    bands: list[str],
    threads: int,
    max_chain_length: int,
    burnin_length: int,
    nchains: int,
    append: bool,
) -> None:
    galaxy = Path(galaxy_path)
    failed_bands = []
    print(f"Fitting galaxy: {galaxy}", flush=True)
    for band in bands:
        image = galaxy / f"image_{band}.fits"
        config = galaxy / f"{fit_type}_{band}.dat"
        if not image.is_file() or not config.is_file():
            print(f"Skipping {band}: missing image or config", flush=True)
            continue

        output_root = f"{fit_type}_{band}_mcmc"
        existing_chains = list(galaxy.glob(f"{output_root}.*.txt"))
        if existing_chains and not append:
            failed_bands.append(band)
            print(f"Refusing to overwrite existing {band}-band chains; use --append to continue.", flush=True)
            continue

        command = [
            "imfit-mcmc",
            "-c", str(config),
            str(image),
            "--output", output_root,
            "--max-threads", str(threads),
            "--max-chain-length", str(max_chain_length),
            "--burnin-length", str(burnin_length),
        ]
        if nchains:
            command.extend(["--nchains", str(nchains)])
        if append:
            command.append("--append")

        mask = galaxy / "image_mask.fits"
        psf = galaxy / f"psf_patched_{band}.fits"
        noise = galaxy / f"image_{band}_invvar.fits"
        if mask.is_file():
            command.extend(["--mask", str(mask)])
        if psf.is_file():
            command.extend(["--psf", str(psf)])
        if noise.is_file():
            command.extend(["--noise", str(noise), "--errors-are-weights"])

        print(f"Running {band}: {' '.join(command)}", flush=True)
        result = subprocess.run(command, cwd=galaxy, check=False)
        if result.returncode:
            failed_bands.append(band)

    if failed_bands:
        raise RuntimeError(f"IMFIT-MCMC failed or refused existing chains for galaxy {os.path.basename(galaxy_path)} for bands: {', '.join(failed_bands)}")


def submit(args: argparse.Namespace) -> None:
    root = Path(args.root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Galaxy root is not a directory: {root}")

    galaxies = find_galaxies(root, args.fit_type, args.bands)
    if not galaxies:
        raise RuntimeError(
            f"No folders under {root} contain both image_<band>.fits and "
            f"{args.fit_type}_<band>.dat for bands {','.join(args.bands)}"
        )

    executor = submitit.AutoExecutor(folder=str(root / "slurm_logs" / "%j"))
    executor.update_parameters(
        name="imfit-mcmc",
        timeout_min=args.timeout_min,
        mem_gb=args.mem_gb,
        cpus_per_task=args.threads,
        slurm_partition=args.partition,
        slurm_array_parallelism=args.max_parallel,
        stderr_to_stdout=True,
    )
    worker = partial(
        run_galaxy,
        fit_type=args.fit_type,
        bands=args.bands,
        threads=args.threads,
        max_chain_length=args.max_chain_length,
        burnin_length=args.burnin_length,
        nchains=args.nchains or 0,
        append=args.append,
    )
    jobs = executor.map_array(worker, [str(galaxy) for galaxy in galaxies])
    print(f"Submitted {len(jobs)} galaxies to Slurm job array {jobs[0].job_id}")
    print(f"Submitit logs: {root / 'slurm_logs'}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, help="Root directory containing processed galaxy folders")
    parser.add_argument("--fit-type", default="2_sersic")
    parser.add_argument("--bands", nargs="+", default=["g", "r", "i", "z"])
    parser.add_argument("--threads", type=int, default=4, help="CPUs per task and IMFIT thread limit")
    parser.add_argument("--max-parallel", type=int, default=20, help="Maximum simultaneous array tasks")
    parser.add_argument("--max-chain-length", type=int, default=100000)
    parser.add_argument("--burnin-length", type=int, default=5000)
    parser.add_argument("--nchains", type=int, help="MCMC chains per band; default is IMFIT-MCMC's choice")
    parser.add_argument("--append", action="store_true", help="Continue existing MCMC chains")
    parser.add_argument("--timeout-min", type=int, default=1440, help="Slurm time limit in minutes")
    parser.add_argument("--mem-gb", type=float, default=8, help="Memory per array task in GB")
    parser.add_argument("--partition", help="Optional Slurm partition")

    args = parser.parse_args()
    if args.threads < 1 or args.max_parallel < 1 or args.timeout_min < 1 or args.mem_gb <= 0:
        parser.error("threads, max-parallel, timeout-min, and mem-gb must be positive")
    submit(args)


if __name__ == "__main__":
    main()