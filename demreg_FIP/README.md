# demreg_FIP

This repository contains code for Differential Emission Measure (DEM) calculations using the regularisation method and FIP bias composition calculation.

## Overview

The main script `mcmc_para.py` performs the following tasks:
- Downloads and processes solar physics data
- Calculates DEM using a parallelized approach
- Computes FIP bias for different elements

## Requirements for my environment

- Python 3.11.9
- Libraries: numpy, astropy, tqdm, multiprocessing, sunpy, matplotlib

For a complete list of my environment dependencies, see `requirements.txt`.
This is a bit of a mess because I am using a general-purpose environment for all kinds of tasks.

## Usage

1. Create a `config.txt` file with the filenames of the data files to process, one per line. For example:
```
SO_EIS_data/eis_20230327_061218.data.h5
```

2. Run the script in terminal:
```bash
python mcmc_para.py –-cores <number_of_cores>
```

Replace `<number_of_cores>` with the number of CPU cores you want to use. The default is 60 for Linux and 10 for macOS.

## Output

The script generates several output files in the specified output directory:
- DEM results (`.npz` files)
- Composition maps (`.fits` and `.png` files)

## Configuration

Some settings can be customized:
- Abundance file: modify in `ashmcmc.py`
- Density interpolation files: set directory in `asheis.py`
- Calibration: currently hard-coded to Warren et al. 2014 calibration in `asheis.py`. Del Zanna's 2023 calibration is available too.

## Author

Andy To
European Space Agency
