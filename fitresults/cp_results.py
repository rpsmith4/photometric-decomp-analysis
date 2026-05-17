import sys
import os
from pathlib import Path
import argparse
import glob
import shutil
import json



def main(galpath, save_path, copy_list: list = ['fitted'], overwrite: bool = True, json_path: Path | None = None):
    # Find galmarks.json in saved results location
    gal_json = glob.glob(str(save_path) + "/*.json" )
    print('')

    if len(gal_json) != 1: 
        found_conf = False
        for json_file in gal_json:
                if json_file.endswith("galmarks.json"):
                    gal_json = json_file
                    found_conf = True
        if not found_conf:
            raise NameError("Could not find the marked galaxies .json file. Make sure that the saved location has only one .json file.")

    else:
        gal_json = gal_json[0]
    print(f"Copying galaxies listed in {gal_json}\n\n")

    with open(gal_json, 'r') as json_file:
        data = json.load(json_file)
    # print(data)

    # Gather all galaxy directories for saved results into gal_dirs
    gal_dirs = []
    for fit_type in copy_list:
        fit_path = save_path / fit_type
        # gals = fit_path.rglob("*")
        for path in fit_path.rglob("*"):
            if path.is_dir():
                # Check if this directory contains any subdirectories or files
                has_files = any(file.is_file() for file in path.iterdir())
                has_subdir = any(p.is_dir() for p in path.iterdir())
                name = path.parts[-1]

                if has_files and name in data and not has_subdir:
                    gal_dirs.append(path)
    
    # Iterate through each of the galaxy directories and copy results over
    for gal_dir in gal_dirs:
        save_loc = list(galpath.rglob(gal_dir.parts[-1]))
        if len(save_loc) == 0:
            print(f"Cound not find {gal_dir.parts[-1]} in the galaxy directories. Skipping.\n")
            continue
    
        files_tocopy = glob.glob(str(gal_dir) + "/*")
        # print(f"{files_tocopy}\n")
        for part in save_loc:
            print(f"Copying files from {gal_dir} to {part}")
            for file in files_tocopy:
                new_file =str(part) + "/" + Path(file).parts[-1]
                # print(new_file)
                if os.path.exists(new_file):
                    # print(f"File already exists.")

                    if overwrite:
                        # os.remove(new_file)
                        print(f"🗑️  Removed old {new_file}")
                    else:
                        print(f"⛔️ Skipped copying {new_file}")
                        continue

                print(f"📝 Creating new {Path(file).parts[-1]} at {Path(new_file).parent}")
                try: 
                    shutil.copy2(Path(file), Path(new_file).parent)
                except Exception as e:
                    print(f"❌ Failed to write file: {e}")
                print(f"✅ Success!")

            print('')    
                    
                    

    if json_path is not None:
        print("Updating .json file...")
        if not json_path.parts[-1].endswith(".json"):
            json_path = glob.glob(str(json_path) + "/*.json")
            if len(json_path) != 1:
                found_conf = False
                for json_file in json_path:
                    if json_file.endswith("galmarks.json"):
                        json_path = json_file
                        found_conf = True
                if found_conf == False:
                    raise NameError("Could not find the galmarks.json file in the GUI directory. Try passing in the full path, including the file name.")
        with open(json_path, mode="r") as old_json:
            is_changed = False
            old_data = json.load(old_json)
            for item in data:
                if item in old_data:
                    # print(f"Found {item} in both .json files.")
                    if data[item] != old_data[item]:
                        old_data[item] = data[item]
                        print(f"Updated {{{item}: {old_data[item]}}} to {{{item}: {data[item]}}}")
                        is_changed = True
                    # else:
                    #     print("Both are the same")
                else:
                    print(f"Added {item} to the new .json")
                    old_data[item] = data[item]
                    is_changed = True
        if is_changed:
            with open(json_path, mode="w") as new_json:
                json.dump(old_data, new_json)
            print("Successfully rewrote .json file")
        else:
            print("No new information to change\n")


        
    return



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Copies the fit results into the galaxies' folders in another location")
    parser.add_argument("-p", help="Path to root folder containing galaxies' data", default=".")
    parser.add_argument("-n", help="Path to folder containing saved results", default=".")
    parser.add_argument("--no_overwrite", help="Don't overwrite files", action="store_false")
    parser.add_argument("--no_fitted", help = "Don't copy files marked as 'fitted'", action="store_false")
    parser.add_argument("--no_return", help = "Don't copy files marked as 'return'", action="store_false")
    parser.add_argument("--no_unable", help = "Don't copy files marked as 'unable'", action="store_false")

    parser.add_argument("--update_json", help = "Update the galmarks.json file located in the GUI", action="store_true")
    parser.add_argument("-json", help="Path to the galmarks.json file in the GUI directory", default=".")

    args = parser.parse_args()
    galpath = Path(args.p).resolve()
    save_path = Path(args.n).resolve()
    overwrite = args.no_overwrite
    copy_list = []
    if args.no_fitted:
        copy_list.append('fitted')
    if args.no_return:
        copy_list.append('return')
    if args.no_unable:
        copy_list.append('unable')
    if len(copy_list) == 0:
        raise ValueError("Must pick at least one of fitted, return, and unable to copy over.")
    
    json_path = None
    if args.update_json:
        json_path = Path(args.json).resolve()
    


    main(galpath, save_path, copy_list, overwrite = overwrite, json_path = json_path)

    # Example usage (from project base directory):
    # python3 path/to/cp_results.py -p path/to/galaxy/root/folder -n path/to/save/location --no_overwrite --no_unable --update_json -json path/to/GUI/galmarks.json
    
    # by default, overwrites files if they exist and copies over all of the results in the path