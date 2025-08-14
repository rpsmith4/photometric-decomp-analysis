# Photometric Decomposition of Polar Structure Galaxies Analysis Code
This project is intended to be used for:

- Downloading FITs image, PSFs (core and extended), inverse-variance maps, and jpg color images from DESI Legacy Survey
- Generating a mask using ```photutils```
- Generate an initial config file for each galaxy for IMFIT 

    - Includes the ability to generate configs for a two 2D Sérsic functions (1 host + 1 polar) and 1 Sérsic function (host) and 1 Gaussian ring (polar) for polar rings
    - Still needs a lot more work to make the guess better

- Run IMFIT on each galaxy
- Read from results and store in a table
- Plot results

## Requirements

- [IMFIT](https://imfit.readthedocs.io/en/latest/imfit_tutorial.html) (Also, [here](https://www.mpe.mpg.de/~erwin/resources/imfit/imfit_howto.pdf) is some more documentation for IMFIT)
- [Siena Galaxy Atlas FITs table](https://sga.legacysurvey.org/)
- [IMAN](https://bitbucket.org/mosenkov/iman_new/src/master/)
- Python 3.1X.XX
- Some list of python packages (will add later)

## Examples
Here are some examples of how to use the code from downloading data to fitting it.

> [!NOTE] 
> Not all of these options have to be set, make sure to check the help section of a script (python3 some_script.py -h) to see the defaults and specific usage for each option
---
```get_fits_from_folders.py```

Used to download fits from DESI given an existing file structure that contains files that are named the same as their corresponding galaxy. For example, you may have the file structure ```my_galaxies/Polar\ Rings/Chosen_Galaxies/galaxy 1, galaxy 2, ...```. To download data for each of these galaxies run:

```python3 get_fits_from_folders.py -p my_galaxies/ -r -c Path/to/SGA2020.fits -o Ouput/ --dr dr10 --factor 3 --bands griz --files fits psf wm jpg mask```

Which will recursively move into folders and (using the SGA catalogue) download the fits, psf, inverse variance (weight) map, jpg, and generate a mask to Ouput/ using dr10, multiplying the R26 isophote level by a factor of 3, getting data from the g,r,i, and z bands.

The outputs will be in ```Output/Polar\ Rings/Chosen_Galaxies/galaxy 1/data... .../galaxy 2/data... ... ``` (since the ```-r``` option regenerates the file structure), where the data for a galaxy is put into a folder with its corresponding galaxy name.

---

```generate_imfit_conf.py```

Used to generate a config file for a given galaxy for each color band.

> [!IMPORTANT] 
> Make sure that the folder containing polar rings is called "Polar\ Rings", polar bulges is called "Polar_Tilted\ Bulges", and halos is called "Polar_Tilted Halo", since the script attempts to determine the type of polar structure, in which some parts of the script depend on.

Using the previous file structure, the configs can be generated with:

```python3 generate_imfit_conf.py -p my_galaxies/ -r --mask --fit_type 2_sersic```

Which will generate configs for each galaxy in ```my_galaxies/``` for a 2 Sérsic fit, and will also use the mask when doing so.

---

```imfit_run.py```

Used to run IMFIT on a galaxy that has config files for each band. Once again, using the same folder:

```python3 imfit_run.py -p my_galaxies/ -r --all --mask --make_composed --fit_type 2_sersic --max_threads 4```

Which will run IMFIT on each galaxy in ```my_galaxies/``` with the PSF, weight map, and mask, as well as make a composed image (FITs image cube with original image, model, residual, residual percent, and each component of the model separately), while using a max of 4 threads.

> [!IMPORTANT] 
> Make sure to set --mask (even if using --all) to apply the mask to the composed image

> [!WARNING]
> Check IMFIT's main [documentation](https://www.mpe.mpg.de/~erwin/resources/imfit/imfit_howto.pdf), which describes some of the caveats to using multiple threads.

---

```plot_parameters.py```

Used to make plots of the outputs of the fitting process, as well as determine (at least somewhat) the quality of the fits. Can also be used to make composed images.

> [!WARNING]
> This script is very much a WIP at the moment, and so may be subject to change (it is also a bit hard to read).

> [!NOTE]
> This script makes use of an ```exclude.txt``` file, in which one can input a list of galaxy names of which to exclude from plotting. Must be in the same directory in which this script is being run from.

> [!IMPORTANT] 
> Make sure that the folder containing polar rings is called "Polar\ Rings", polar bulges is called "Polar_Tilted\ Bulges", and halos is called "Polar_Tilted Halo", since the script attempts to determine the type of polar structure, in which some parts of the script depend on.


To get the overall output of our fits, use:

```python3 plot_parameters.py -p my_galaxies/ -r -o Plots/ --fit_type 2_sersic --plot_stats --plot_type compare_structure -c path/to/SGA2020.fits```

---
```redshift_galaxy.py```

Used to artificially redshift galaxy images that are output from TNG50 (and post-processed with SKIRT).

Can be used by running:

```python3 -p path/to/TNG50data/ -b g r i z -t SDSS```

To run the redshift code on all the SDSS type images with all bands for every galaxy in the directory.

Can also use:

```python3 -p path/to/TNG50data/TNGxxxxxx_E_SDSS* -b g r i z -t SDSS```

To run the code on a particular galaxy. The "*" wildcard is needed at the end to ensure that all of the bands are loaded into the script.

> [!WARNING]
> This script is also mostly a work in progress. Currently does not work in the case of using less than all of the g, r, i, and z bands (due to the way I made all the input parameters assume that I was using all 4 bands). Assumes currently that you are using the SDSS TNG50 outputs at the moment. 

> [!WARNING]
> Does not currently work for an input redshift of less than 0.03. Also the image is very noisy unless a ridiculously large input exposure is used.

---
Any other scripts are mostly helpers in some shape or form that are not intended to be run on their own for this pipeline (though some can).
