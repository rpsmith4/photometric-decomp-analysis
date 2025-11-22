import ferengi
import argparse
import numpy as np
from pathlib import Path
from astropy.io import fits
import astropy.units as u
import matplotlib.pyplot as plt
import os
import glob
import multiprocessing as mp
from astropy.cosmology import FlatLambdaCDM
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import astropy.units as u, astropy.constants as c
from astropy.table import Table

# TODO: Figure out how to get MJy/sr -> counts/pixel
def simunits2cts(simim, pixarea, expt):
    simim = simim * 10 ** 6 * 10 ** (-23) * pixarea # MJy/sr to ergs/s/cm**2/Hz(fnu) /pixel
    for k in range(np.shape(simim)[-1]):
        simim[:, :, k] = ferengi.maggies2cts(ferengi.fnu2maggies(simim[:, :, k]), expt=expt[k], zp=22.5)
    return simim # fnu/pixel -> maggies/pixel -> cnts/pixel

def simunits2maggies(simim, pixarea):
    simim = simim * 10 ** 6 * 10 ** (-23) * pixarea # MJy/sr to ergs/s/cm**2/Hz(fnu) /pixel
    for k in range(np.shape(simim)[-1]):
        simim[:, :, k] = ferengi.fnu2maggies(simim[:, :, k])
    return simim # fnu/pixel -> maggies/pixel -> cnts/pixel

def cts2simunits(im, pixarea, expt):
    for k in range(np.shape(im)[-1]):
        im[:, :, k] = ferengi.maggies2fnu(ferengi.cts2maggies(im[:, :, k], expt=expt[k], zp=22.5))
    im = im / (10 ** 6 * 10 ** (-23)) / pixarea
    return im 

def flux2ABmag(flux):
    return 22.5 - 2.5 * np.log10(flux)

def get_stats(ims, d):
    bands = ["g", "r", "i", "z"]

    stats = {}
    for k, band in enumerate(bands):
        stats[band] = {}
        app_mag = flux2ABmag(np.sum(ims[:, :, k]))
        abs_mag = app_mag - 5 * np.log10((d.to(u.pc).value/10))
        stats[band]["app_mag"] = app_mag
        stats[band]["abs_mag"] = abs_mag

    # stats = pd.DataFrame([[band, app_mag, abs_mag] for band,d in stats.items()]columns=["Band", "App. Mag", "Abs. Mag"])

    return stats

def simunits2maggies(simim, pixarea):
    simim = simim * 10 ** 6 * 10 ** (-23) * pixarea # MJy/sr to ergs/s/cm**2/Hz(fnu) /pixel
    for k in range(np.shape(simim)[-1]):
        simim[:, :, k] = ferengi.fnu2maggies(simim[:, :, k])
    return simim # fnu/pixel -> maggies/pixel -> cnts/pixel

def get_stats(data):
    bands = ["g", "r", "i", "z"]

    cosmo = FlatLambdaCDM(H0=70, Om0=0.3)

    cols = ["Name", "z", "band", "App. Mag", "Abs. Mag"]
    # all_stats = pd.DataFrame(columns=cols)
    all_stats = Table(names=cols)


    names = []
    zs = []
    all_bands = []
    app_mags = []
    abs_mags = []
    for i, galaxy in enumerate(data.keys()):
        for z in data[galaxy].keys():
            ims = data[galaxy][z]
            for k, band in enumerate(bands):
                d = cosmo.luminosity_distance(z) # Mpc
                # print(u.spectral_density(4640 * u.AA))
                f = np.sum(ims[:, :, k]) * u.nmgy
                zero_point_star_equiv = u.zero_point_flux(3631.1 * u.Jy)

                app_mag = u.Magnitude(f.to(u.AB, zero_point_star_equiv))
                # app_mag = flux2ABmag(np.sum(ims[:, :, k]))
                abs_mag = app_mag - u.Magnitude(5 * np.log10((d.to(u.pc).value/10)))
                
                names.append(galaxy)
                zs.append(z)
                all_bands.append(band)
                app_mags.append(app_mag.value)
                abs_mags.append(abs_mag.value)

                # stats.loc[k] = [galaxy, z, band, app_mag, abs_mag]
        # all_stats = pd.concat([all_stats, stats], ignore_index=True)
    all_stats_dict = {
        "Name" : names,
        "z": zs,
        "Band": all_bands,
        "App. Mag": app_mags * u.AB,
        "Abs. Mag": abs_mags * u.AB
    }
    all_stats = pd.DataFrame(all_stats_dict)
    all_stats = Table.from_pandas(all_stats)
    # all_stats = all_stats.groupby(by="Name", group_keys=True)[all_stats.columns].apply(lambda x: x)

    # stats = pd.DataFrame([[band, app_mag, abs_mag] for band,d in stats.items()]columns=["Band", "App. Mag", "Abs. Mag"])

    return all_stats
    
def plot_stats(all_stats):
    print(all_stats)
    names = list(set(all_stats["Name"]))
    for name in names:
        stats = all_stats[all_stats["Name"] == name]



    band_colors = {
        "g": "g",
        "r" : "r",
        "i" : "firebrick",
        "z" : "blueviolet"
    }
    bands = ["g", "r", "i", "z"]
    fig = plt.figure(figsize=(20, 20))
    # for band in bands:
    #     stats_band = all_stats[all_stats["Band"] == band]

    #     # App. mag in each band, Abs. mag in each band
    #     # same but with high redshift
    #     # same but with higher redshift when I get to it

    #     # OR

    #     # App. mag of each type, Abs. mag of each type
    #     # same but with high redshift
    #     # same but with higher redshift when I get to it

    #     # gs = gridspec.GridSpec(2, 4)
    #     # gs.update(wspace=0.5)

    #     # plt.subplot(gs[0, :2])
    #     for k,z in enumerate([0.03, 0.5, 1.0]):
    #         stats_z = stats_band[stats_band["z"] == z]
    #         app_mag = stats_z["App. Mag"]
    #         abs_mag = stats_z["Abs. Mag"]

    #         print(app_mag.value)
    #         ax1 = plt.subplot(3, 2, 1 + k*2)
    #         plt.hist(app_mag.value, label=f"{band}", color=band_colors[band], histtype="step")
    #         plt.xlabel(rf"Apparent Magnitude at $z={z}$")
    #         plt.ylabel("Count")

    #         plt.subplot(3, 2, 2 + k*2)
    #         plt.hist(abs_mag.value, label=rf"{band}", color=band_colors[band], histtype="step")
    #         plt.xlabel(rf"Apparent Magnitude at $z={z}$")
    #         plt.ylabel("Count")


    # App. mag in each band, Abs. mag in each band
    # same but with high redshift
    # same but with higher redshift when I get to it

    # OR

    # App. mag of each type, Abs. mag of each type
    # same but with high redshift
    # same but with higher redshift when I get to it

    # gs = gridspec.GridSpec(2, 4)
    # gs.update(wspace=0.5)

    # plt.subplot(gs[0, :2])

    plt.rc('axes', labelsize=20)
    plt.rc('legend', fontsize=28)
    plt.rc('figure', titlesize=34)
    plt.rc('xtick', labelsize=28)
    plt.rc('ytick', labelsize=28)
    for k,z in enumerate([0.03, 0.5, 1.0]):
        stats_z = all_stats[(all_stats["z"] == z) & (all_stats["Band"] == "r")]
        app_mag = stats_z["App. Mag"]
        abs_mag = stats_z["Abs. Mag"]

        ax1 = plt.subplot(3, 2, 1 + k*2)
        plt.hist(app_mag, label=rf"", histtype="bar", color="lightgrey")
        plt.hist(app_mag, label=rf"", histtype="step", color="black", lw=1)
        plt.xlabel(rf"$m_r$ at $z={z}$", fontsize=28)
        plt.ylabel("Count", fontsize=28)
        if k == 0:
            plt.title("Apparent Magnitude", fontsize=30)

        plt.subplot(3, 2, 2 + k*2)
        plt.hist(abs_mag, label=rf"", histtype="bar", color="lightgrey")
        plt.hist(abs_mag, label=rf"", histtype="step", color="black", lw=1)
        plt.xlabel(rf"$M_r$ at $z={z}$", fontsize=28)
        plt.ylabel("Count", fontsize=28)
        if k == 0:
            plt.title("Absolute Magnitude", fontsize=30)


    plt.tight_layout()
    plt.savefig("mags.png")

    fig = plt.figure(figsize=(10, 20))

    # Color magnitude <g-r>

    plt.rc('axes', labelsize=20)
    plt.rc('legend', fontsize=28)
    plt.rc('figure', titlesize=34)
    plt.rc('xtick', labelsize=28)
    plt.rc('ytick', labelsize=28)

    for k,z in enumerate([0.03, 0.5, 1.0]):
        stats_z = all_stats[all_stats["z"] == z]
        app_mag = stats_z["App. Mag"]
        abs_mag = stats_z["Abs. Mag"]
        abs_mag_g = stats_z[stats_z["Band"] == "g"]["Abs. Mag"]
        abs_mag_r = stats_z[stats_z["Band"] == "r"]["Abs. Mag"] 

        g_r = abs_mag_g - abs_mag_r

        ax1 = plt.subplot(3, 1, 1 + k)
        plt.hist(g_r, label=rf"", histtype="bar", color="lightgrey")
        plt.hist(g_r, label=rf"", histtype="step", color="black", lw=1)
        plt.xlabel(fr"$g-r$ at $z={z}$", fontsize=28)
        plt.ylabel("Count", fontsize=28)
        if k==0:
            plt.title(r"$<g-r>$ Color", fontsize=30)

        # plt.subplot(3, 2, 2 + k*2)
        # plt.hist(abs_mag, label=rf"$z={z}$", color=band_colors[band], histtype="step")
        # plt.xlabel("Absolute Magnitude")
        # plt.ylabel("Count")

    # ax1.legend()
    plt.tight_layout()
    plt.savefig("g-r.png")

    

# Need abs mag, D26 is possible, color mag graph, something else idk
# Probably want the magnitude within the D26 isophote

def main(data):
    pixscale = 0.262 * u.arcsecond
    pixarea = pixscale ** 2
    pixarea = pixarea.to(u.sr).value

    # ds = [cosmo.luminosity_distance(z).value for z in zs]
    # all_stats = {}
    # for galaxy in data.keys():
    #     all_stats[galaxy] = {}
    #     for z in data[galaxy].keys():
    #         # ds = np.array(ds)
    #         ims = data[galaxy][z]
    #         all_stats = get_stats(ims, z, galaxy_name)
    all_stats = get_stats(data)
    plot_stats(all_stats)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                     description="Artificially redshift TNG50 simulation galaxies with FERENGI.")
    parser.add_argument("-p", help="Path to folder containing fits simulation images", default=".")
    parser.add_argument("-pr", help="Path to folder containing fits simulation images (redshifted)", default="../")
    parser.add_argument("-b", help="Input bands", nargs="+", choices=["g", "r", "i", "z"], default=["g", "r", "i", "z"])
    # parser.add_argument("-n", help="Number of parallel redshifts", default=1, type=int)

    args = parser.parse_args()
    p = Path(args.p).resolve()
    print(args.pr)
    pr = os.path.join(p, Path(args.pr))
    print(pr)

    galaxy_names = glob.glob(f"*", root_dir=p)
    # galaxy_names = [galaxy_name for galaxy_name in galaxy_names if "psf" not in galaxy_name]
    print(len(galaxy_names))

    
    galaxy_names = [Path(galaxy_name).stem for galaxy_name in galaxy_names]

    # zs = [galaxy_name.split('_')[2][2:] for galaxy_name in galaxy_names]
    # zs = np.array(zs, dtype=np.float64)
    # zs = [float(z) for z in zs]
    zs = [0.05, 0.1, 0.15, 0.2]

    galaxy_names = [galaxy_name.split('_')[0] for galaxy_name in galaxy_names]

    pixscale = 0.262 * u.arcsecond
    pixarea = pixscale ** 2
    pixarea = pixarea.to(u.sr).value

    data = {}
    for k, galaxy_name in enumerate(galaxy_names):
        im = list()
        im_lo = list()
        for band in args.b:
            im.append(fits.getdata(os.path.join(p, f"{galaxy_name}_{band}_z={zs[k]}.fits"))) # Assume everything is in the same parent directory
            im_lo.append(fits.getdata(os.path.join(pr, f"{galaxy_name}_E_SDSS_{band}.fits")))

        im = np.array(im)
        im_lo = np.array(im_lo)

        im = np.moveaxis(im, 0, -1)
        im_lo = np.moveaxis(im_lo, 0, -1)
        # im_lo = im_lo * 10**4
        im_lo = simunits2maggies(im_lo, pixarea)
        # In the shape of (x, y, bands)

        im = im * 10 ** (-9)

        try:
            data[galaxy_name][0] = im_lo
        except:
            data[galaxy_name] = {0: im_lo}
        data[galaxy_name][zs[k]] = im
        print(galaxy_name, zs[k])
        # data[galaxy_name] = {zs[k]: im}

    # print(data["TNG167392"].keys())
    
    main(data)