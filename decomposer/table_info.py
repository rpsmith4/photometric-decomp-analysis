"""
All functions related to getting information from the table (currently master_table.csv). Ensure that master_table.csv is in the same directory as this file. 

Uses
----
(1) Create a list of all of the galaxy names from the table. 
(2) Given a particular galaxy name, give the type of PSG (PSG_TYPE_1) and host morphology (MORPHTYPE)
"""


import pandas as pd
import glob

def get_galaxy_names(csv_file = None):
    """
    Reads a CSV of galaxy data and returns a list of all galaxy names.

    Parameters
    ----------
    csv_file : Name of .csv file containing all of the pertinent information. Defaults to "none".

    Returns
    -------
    list
        A list of galaxy names (from the NAME column).
    """
    # Get the master_table.csv file if none is provided
    if csv_file == None:
        csv_file = find_table()
        if csv_file == None:
            print("Couldn't find the proper .csv file.")
            return None

    
    # Add somethign to make sure that csv_file has something??


    # Read the CSV file into a pandas DataFrame
    df = pd.read_csv(csv_file)

    # Get the NAME column as a list
    names = df['NAME'].dropna().tolist()

    return names


def get_galaxy_info(galaxy_name, csv_file = None):
    """
    Reads a CSV of galaxy data and returns PSG_TYPE_1 and MORPHTYPE
    for the specified galaxy name.

    Parameters
    ----------
    csv_file : str
        Path to the CSV file containing galaxy data.
    galaxy_name : str
        The NAME value of the galaxy to look up.

    Returns
    -------
    dict
        A dictionary with 'psg_type' and 'MORPHTYPE' values,
        or None if the galaxy name isn't found.
    """
    # Get the master_table.csv file if none is provided
    if csv_file == None:
        csv_file = find_table()
        if csv_file == None:
            print("Couldn't find the proper .csv file.")
            return None
        
    # Read the CSV file into a pandas DataFrame
    df = pd.read_csv(csv_file)

    # Find the row where NAME matches the provided galaxy_name
    row = df.loc[df['NAME'] == galaxy_name]

    if row.empty:
        print(f"Galaxy '{galaxy_name}' not found in file.")
        return None

    # Extract the values for PSG_TYPE_1 and MORPHTYPE
    result = {
        'psg_type': row['PSG_TYPE_1'].values[0],
        'MORPHTYPE': row['MORPHTYPE'].values[0],
        'scale': row['SCALE'].values[0],
    }

    return result




def set_directory():
    import os
    """
    When called: sets the current working directory to be the directory where this script is located. 

    Parameters
    ----------
    none

    Returns
    -------
    nothing
    """
    # Get the directory of the current script
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Change working directory to the script's directory
    os.chdir(script_dir)

    return None

def find_table(name = "master_table.csv"):
    """
    Parameters
    ----------
    name : the name of the table you want to get

    Returns
    -------
    list
        a list containing all files ending in .csv located in the same directory as this file
    """
    # Sets directory to script location
    set_directory()

    # makes a list of all files ening in .csv
    csv_files = glob.glob("*.csv")
    if name in csv_files:
        return name
    else:
        return None





def main():
    """ 
    Prints off the information for a random galaxy from the table.
    """
    import numpy as np
    gal_names = get_galaxy_names()
    n = np.random.randint(1, len(gal_names))
    print(n, get_galaxy_info(gal_names[n]))
    return None

if __name__ == "__main__":
    main()