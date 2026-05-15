# read in info from galmarks.json
# create/overwrite directory to save results
# copy in the galmarks.json file
# generate folders for fitted, review, and unable
# within folders, generate subfolders for configs, fits (default) or using name conventions of where galaxies are stored
# write in all info
import sys
import os
from pathlib import Path
import argparse
import glob
import shutil
import json

file_loc = Path(os.path.dirname(__file__))
sys.path.append(os.path.join(file_loc))


def main(galpath: Path, name: str, confpath: Path, overwrite: bool = False, match_dir: bool = False) -> None:

    # If the directory containing galmarks.json is passed, try to find the exact file
    if not confpath.parts[-1].endswith(".json"):
        confpath = glob.glob(str(confpath) + "/*.json")
        if len(confpath) != 1:
            found_conf = False
            for json_file in confpath:
                if json_file.endswith("galmarks.json"):
                    confpath = json_file
                    found_conf = True
            if found_conf == False:
                raise NameError("Could not find the galmarks.json file. Try passing in the full path, including the file name.")
            
    # Ensure that the directory is made 
    out_path = Path(name).resolve()
    if overwrite:
        if out_path.exists() and out_path.is_dir():
            shutil.rmtree(out_path)

        out_path.mkdir(parents=True) # Create parents as needed
    else:
        try:
            out_path.mkdir(parents = True)
        except FileExistsError as e:
            print(f"\nDirectory could not be created as it already exists. Set --overwrite or choose a new name. \n{e}\n")
            return
    
    shutil.copy2(confpath, out_path)
    # Create output directories to discriminate the how good the user-supplied fit is
    fitted_path = out_path / "fitted"
    review_path = out_path / "return"
    unable_path = out_path / "unable"
    fit_status_paths = [fitted_path, review_path, unable_path]
    for subpath in fit_status_paths:
        subpath.mkdir()
    
    # Match structure of user's galaxy files (organized by PS type)
    if match_dir:
        match_names = glob.glob(str(galpath) + "/*")
        for subpath in fit_status_paths:
            for subsubdir in match_names:
                new_dir = subpath / Path(subsubdir).parts[-1]
                new_dir.mkdir(parents=True)


    # Write all configs and fit results into the appropriate locations
    with open(confpath, 'r') as json_file:
        data = json.load(json_file)

        # Find a galaxy directory given the name
        def find_subdirectory(root_dir, target_name):
            root = Path(root_dir)
            for all_path in root.rglob(target_name):
                if galpath.is_dir():
                    return all_path
            return None

        # Iterate through the list of fir objects in the json file
        for item in data:
            # print(f"{item}: {data[item]}")
            result = find_subdirectory(galpath, item)
            if data[item] == "fitted": # Create directory within fitted directory, with PS type if desired
                if match_dir:
                    ps_type = result.parent.parts[-1]
                    gal_dir = fitted_path / ps_type / item
                    gal_dir.mkdir()
                else: 
                    gal_dir = fitted_path / item
                    gal_dir.mkdir()
            if data[item] == "return": # Create directory within revisit directory, with PS type if desired
                if match_dir:
                    ps_type = result.parent.parts[-1]
                    gal_dir = review_path / ps_type / item
                    gal_dir.mkdir()
                else: 
                    gal_dir = review_path / item
                    gal_dir.mkdir()
            if data[item] == "unable": # Create directory within unable directory, with PS type if desired
                if match_dir:
                    ps_type = result.parent.parts[-1]
                    gal_dir = unable_path / ps_type / item
                    gal_dir.mkdir()
                else: 
                    gal_dir = unable_path / item
                    gal_dir.mkdir()
            
            # Copy over the configs and fit results (all .dat and .txt files)
            dat_path = str(result) +  "/*.dat"
            txt_path = str(result) +  "/*.txt"
            # print(file_path)
            # print(type(file_path))
            dat_files = glob.glob(dat_path)
            txt_files = glob.glob(txt_path)
            for file in dat_files:
                shutil.copy2(file, gal_dir)
            for file in txt_files:
                shutil.copy2(file, gal_dir)
    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Create folders to save all configs and fit results")
    parser.add_argument("-p", help="Path to root folder containing galaxies to save", default=".")
    parser.add_argument("-conf", help="Path to config.json file", default=".")
    parser.add_argument("-n", help="Path/to/name of folder to create to save results to", default="saved_results")
    parser.add_argument("--overwrite", help="Overwrite folder (if exists)", action="store_true")
    parser.add_argument("--match_directory", help = "Match the naming scheme of the subdirectories of stored galaxies", action="store_true")




    args = parser.parse_args()
    galpath = Path(args.p).resolve()
    confpath = Path(args.conf).resolve()
    name = args.n
    main(galpath, name, confpath, overwrite = args.overwrite, match_dir = args.match_directory)

    # Example usage (from project base directory):
    # python3 path/to/fitresults/save_results.py -p path/to/galaxy/root/folder -n save_location_name -conf path/to/GUI --overwrite --match_directory