#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import warnings
from pathlib import Path
import pandas as pd
import numpy as np
from astropy.io import fits
from astropy.visualization import (
    LinearStretch, SqrtStretch, AsinhStretch, LogStretch, SquaredStretch,
    HistEqStretch, ImageNormalize
)

from joblib import Parallel, delayed
import shutil


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
import os

import RGB_gui
import RGB_gui_two

warnings.filterwarnings("ignore", category=RuntimeWarning)

def read_fits(p: str) -> np.ndarray:
            with fits.open(p, memmap=True) as hdul:
                arr = hdul[0].data
            if arr.ndim > 2:
                arr = arr[0]
            return np.array(arr, dtype=np.float32)




def main(N, galaxy_name, PSG_type):
  try:
    print(N, galaxy_name)
    file_g = f'./galaxies/{galaxy_name}/image_g.fits'
    file_r = f'./galaxies/{galaxy_name}/image_r.fits'
    file_z = f'./galaxies/{galaxy_name}/image_z.fits'
    
    g = read_fits(file_g)
    r = read_fits(file_r)
    z = read_fits(file_z)
    
    if PSG_type=="Polar_Tilted Halos":
        rgb = RGB_gui_two.render_rgb_two_branch(g, r, z, 22.5, 0.262, 
                                        'linear',0.,1.,1.0,0.,
                                        'log',0.2,1.0,1.0,1.4,
                                        20.5)
        RGB_gui_two.save_rgb_annotated(rgb, 0.262, f'./RGBs/{PSG_type}/{galaxy_name}.png', galaxy_name=galaxy_name)
        shutil.copy(f'./RGBs/{PSG_type}/{galaxy_name}.png', f'./galaxies/{galaxy_name}/RGB.png')
    elif PSG_type=="Polar_Tilted Rings" or PSG_type=="Polar_Tilted Bulges" or PSG_type=='Polar_Tilted Tidal Structures' or PSG_type=='Forming Polar_Tilted Rings' or PSG_type=='Polar_Tilted Dust Lanes':
        rgb = RGB_gui.render_rgb_with_isomask(
        g, r, z,
        zeropoint=22.5, pixscale=0.262,
        scale='linear', bias=0., contrast=1.0, brightness=1.,
        mu_r=25.5, sigma=1.4)
        RGB_gui.save_rgb_annotated(rgb, 0.262, f'./RGBs/{PSG_type}/{galaxy_name}.png', galaxy_name=galaxy_name)  
        shutil.copy(f'./RGBs/{PSG_type}/{galaxy_name}.png', f'./galaxies/{galaxy_name}/RGB.png')
  except:
    return 1

# Path to your CSV file
csv_file = "psg_catalog_v5.csv"

# Explicitly list the expected columns
columns = [
    "Name", "Category", "Category2", "SGA ID", "PGC",
    "RA_LEDA", "DEC_LEDA", "MORPHTYPE", "PA_LEDA", "D25_LEDA",
    "BA_LEDA", "Z_LEDA", "MAG_LEDA", "RA", "DEC", "D26", "D26_REF",
    "PA", "BA", "G_MAG_SB26", "R_MAG_SB26", "Z_MAG_SB26",
    "G_MAG_SB26_ERR", "R_MAG_SB26_ERR", "Z_MAG_SB26_ERR", "Reference"
]
# Name,Category,Category2,SGA ID,PGC,RA_LEDA,DEC_LEDA,MORPHTYPE,PA_LEDA,D25_LEDA,BA_LEDA,Z_LEDA,MAG_LEDA,RA,DEC,D26,D26_REF,PA,BA,G_MAG_SB26,R_MAG_SB26,Z_MAG_SB26,G_MAG_SB26_ERR,R_MAG_SB26_ERR,Z_MAG_SB26_ERR,Reference

# Read the CSV file
df = pd.read_csv(csv_file)

# Optional: Check if all expected columns are present
missing_columns = [col for col in columns if col not in df.columns]
if missing_columns:
    print("Warning: Missing columns:", missing_columns)
else:
    print("All expected columns found.")

# Display first few rows
print(df.head())

Names = np.array(df["Name"])
RA = np.array(df["RA"])
DEC = np.array(df["DEC"])
D26 = np.array(df["D26"])
RA_LEDA = np.array(df["RA_LEDA"])
DEC_LEDA = np.array(df["DEC_LEDA"])
D25_LEDA = np.array(df["D25_LEDA"])
PSG_TYPE = np.array(df["Category"])

inds_halos = np.where(PSG_TYPE == "Polar_Tilted Halos")[0]
inds_rings = np.where(PSG_TYPE == "Polar_Tilted Rings")[0]
inds_bulges = np.where(PSG_TYPE == "Polar_Tilted Bulges")[0]
inds_pts = np.where(PSG_TYPE == "Polar_Tilted Tidal Structures")[0]
inds_forming = np.where(PSG_TYPE == "Forming Polar_Tilted Rings")[0]
inds_dust = np.where(PSG_TYPE == "Polar_Tilted Dust Lanes")[0]

#print(len(inds_halos),len(inds_rings),len(inds_bulges),len(inds_pts),len(inds_forming),len(inds_dust))


os.makedirs('./RGBs/Polar_Tilted Halos', exist_ok=True)
os.makedirs('./RGBs/Polar_Tilted Rings', exist_ok=True)
os.makedirs('./RGBs/Polar_Tilted Bulges', exist_ok=True)
os.makedirs('./RGBs/Polar_Tilted Tidal Structures', exist_ok=True)
os.makedirs('./RGBs/Forming Polar_Tilted Rings', exist_ok=True)
os.makedirs('./RGBs/Polar_Tilted Dust Lanes', exist_ok=True)


n_jobs = 20

Parallel(n_jobs=n_jobs)(delayed(main)(k, Names[k], 'Polar_Tilted Dust Lanes') for k in inds_dust) #







'''
####################### Manual method ####################### 
N=0
for ind in inds_rings[0:10]: #inds_halos:
    N+=1
    galaxy_name = Names[ind]

    #RGB_gui.RGBAdjustGUI(file_g, file_r, file_z, zero_point=22.5, pixscale=0.262, mu_r_init=22., sigma_init=1.3, galaxy_name=galaxy_name) # For halos
    #RGB_gui.RGBAdjustGUI(file_g, file_r, file_z, zero_point=22.5, pixscale=0.262, mu_r_init=25.5, sigma_init=1.4, galaxy_name=galaxy_name, scale='linear', bias=0., contrast=1., brightness=1.) # For rings
    

    # Good for halos:
    RGB_gui_two.RGBAdjustGUI(file_g, file_r, file_z, zero_point=22.5, pixscale=0.262,
                             mu_r_init=21.,
                             scale_b='linear',
                             bias_b=0.,
                             contrast_b=1.,
                             brightness_b=1.
                             sigma_b=0.,
                             scale_f='log',
                             bias_f=0.2,
                             contrast_f=1.0,
                             brightness_f=1.0,
                             sigma_f=1.4,
                             galaxy_name=galaxy_name) # For halos

############################################################### 
'''
