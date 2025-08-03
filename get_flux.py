import subprocess


def get_flux(model_file):
    result = subprocess.run(["makeimage", f"{model_file}", "--print-fluxes"], capture_output=True)
    # print(result.stdout.decode("utf-8"))

    lines = result.stdout.decode("utf-8").split("\n")
    flux_ratios = {}
    flux = {}
    for line in lines:
        line = line.split(" ")
        line = list(filter(None, line))

        if "Host" in line:
            flux_ratios["Host"] = float(line[3])
            flux["Host"] = float(line[1])
            if float(line[3]) != float(line[3]):
                print(f"Nan flux ratio for {model_file}")
        elif "Polar" in line:
            flux_ratios["Polar"] = float(line[3])
            flux["Polar"] = float(line[1])
    if flux_ratios == {}:
        print(f"Unable to determine flux ratios for {model_file}")
        flux_ratios = {
            "Host": -1,
            "Polar": -1
        }
    return flux_ratios, flux