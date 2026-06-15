import pyimfit
from pathlib import Path
import astropy.units as u
import astropy.constants as c
import subprocess
import numpy as np
import os
from astropy.table import QTable
import traceback as tb
import astropy


def get_all_results(galaxies, bands, fit_type, galmarks):
    # galnames = [os.path.basename(galaxy) for galaxy,_ in galaxies]
    # results["Galaxy_Name"] = galnames

    parameter_names = set()
    rows = []
    for galaxy_path, galaxy_type in galaxies:
        galaxy_name =  os.path.basename(galaxy_path)
        if galaxy_name in galmarks.keys():
            if galmarks[galaxy_name] not in ["unable", "return"]:
                for band in bands:
                    try:
                        file = os.path.join(galaxy_path, f"{fit_type}_{band}_fit_params.txt")
                        functions, chi_sq, chi_sq_red, status, status_message = parse_results(file)
                        for function in functions:
                            parameter_names.update(list(function["parameters"].keys()))
                            rows.append(
                                {
                                    "Galaxy Name": galaxy_name,
                                    "Galaxy Type": galaxy_type,
                                    "Fit Function": function["name"],
                                    "Function Label": function["label"],
                                    "band": band,
                                    "b/a": function["b/a"],
                                    "Flux Ratio": function["flux_ratio"],
                                    "Flux": function["flux"],
                                    "parameters": function["parameters"],
                                    "parameters_unc": function["parameters_unc"],
                                    "ChiSq": chi_sq,
                                    "Reduced ChiSq": chi_sq_red,
                                    "Fit Status": status.strip(),
                                    "Fit Status Message": status_message.strip()
                                }
                            )

                    except Exception:
                        print(f"Couldn't read {file}")
                        # print(tb.format_exc())

    names = [name for name in rows[0].keys() if name!="parameters" and name!="parameters_unc"]
    column_data = {name: [] for name in names}
    for parameter_name in sorted(parameter_names):
        column_data[parameter_name] = []
        column_data[f"{parameter_name}_unc"] = []

    for row in rows:
        for name in names:
            column_data[name].append(row[name])
        for parameter_name in sorted(parameter_names):
            column_data[parameter_name].append(row["parameters"].get(parameter_name, np.nan))
            column_data[f"{parameter_name}_unc"].append(row["parameters_unc"].get("parameter_name", np.nan))

    results = QTable(column_data)
    results = results.group_by("Galaxy Name")
    return results
    


def parse_results(file):
    model = pyimfit.parse_config_file(file)
    band = Path(file).stem.rsplit("_")[-3]
    with open(file, "r") as f:
        lines = f.readlines()
    status = lines[5].split(" ")[7]
    status_message = " ".join(lines[5].split(" ")[9:])
    uncs = dict()
    func_labels = list()
    for k, line in enumerate(lines):
        if "FUNCTION" in line:
            func_type = line.split(" ")[1].rstrip()
            func_label = line.split("LABEL ")[-1].rstrip()
            func_labels.append(func_label)
            func_params = pyimfit.get_function_dict()[func_type]
            uncs[func_label] = dict()
            for j, func_param in enumerate(func_params):
                try:
                    unc = lines[k + j + 1].split("+/-")[1].split("\t")[0]
                    uncs[func_label][func_param] = float(unc) # Extremely janky way to get the uncertainties
                except:
                    uncs[func_label][func_param] = None
    chi_sq = float(lines[7].split(" ")[-1])
    chi_sq_red = float(lines[8].split(" ")[-1])
    functions = []

    flux_ratios, flux = get_flux(model_file=file, labels=func_labels)
    for k, function in enumerate(model.functionList()):
        func_dict = function.getFunctionAsDict()
        for param in func_dict["parameters"]:
            # func_dict["parameters_unc"][param] = func_dict["parameters"][param] 
            func_dict["parameters"][param] = func_dict["parameters"][param][0]
        func_dict["label"] = func_labels[k]
        func_dict["parameters_unc"] = uncs[func_dict["label"]]
        func_dict["band"] = band
        
        e = func_dict["parameters"]["ell"]
        # axis_ratio = np.sqrt(-1*(np.square(e) - 1)) # b/a ratio # THIS IS WRONG I THOUGH IT WAS ECCENTRICITY BUT ITS ELLIPCTICITY
        axis_ratio = -1*(e - 1)# f=1-b/a
        func_dict["b/a"] = axis_ratio
        func_dict["flux_ratio"] = flux_ratios[func_labels[k]]
        func_dict["flux"] = flux[func_labels[k]]

        functions.append(func_dict)
    
    return functions, chi_sq, chi_sq_red, status, status_message

# TODO: Change nmgy to "flux"
# Range should be -19 to -23 for abs mag
def flux2ABmag(flux):
    return 22.5 - 2.5 * np.log10(flux)

def flux2ABmag(flux):
    zero_point_star_equiv = u.zero_point_flux(3631.1 * u.Jy)
    return flux.to(u.mag("AB"), zero_point_star_equiv)

def get_flux(model_file, labels):
    result = subprocess.run(["makeimage", f"{model_file}", "--print-fluxes"], capture_output=True)
    # print(result.stdout.decode("utf-8"))

    lines = result.stdout.decode("utf-8").split("\n")
    flux_ratios = {}
    flux = {}
    for line in lines:
        line = line.split(" ")
        line = list(filter(None, line))

        for label in labels:
            if label in line:
                flux_ratios[label] = float(line[3])
                flux[label] = float(line[1])
                if float(line[3]) != float(line[3]):
                    print(f"Nan flux ratio for {model_file}")
    if flux_ratios == {}:
        print(f"Unable to determine flux ratios for {model_file}")
        for label in labels:
            flux_ratios[label] = -1
    return flux_ratios, flux