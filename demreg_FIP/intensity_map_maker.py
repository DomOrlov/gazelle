#!/usr/bin/env python
import argparse
import numpy as np
import os
import shutil
from ashmcmc import ashmcmc
from pathlib import Path
import matplotlib
matplotlib.use("Agg")  
from sunpy.map import Map


def make_intensity_maps_for_file(filename, line_databases, ncpu=4, test_mode=False):
    """
    Generate intensity maps for all lines in `line_databases`, or just one in test mode.
    """
    a = ashmcmc(filename, ncpu=ncpu)
    print(f"DEBUG: ashmcmc outdir => {a.outdir}")

    custom_intensity_dir = r"C:\Users\domor\clean_demreg_FIP\intensity_maps"

    if test_mode:
        #line_databases = {"CaAr": ["ca_14_193.87"], "FeS": ["fe_16_262.98"]}
        line_databases = {"FeS": ["s_13_256.69"]}  # only S XIII 256.69


    for ratio_key, ratio_lines in line_databases.items():
        for line in ratio_lines[:2]:
            print(f"\n--- Generating intensity map for {line} in {filename} ---")
            try:
                m = a.ash.get_intensity(
                    line,
                    outdir=custom_intensity_dir,  
                    refit=False,
                    plot=True,
                    mcmc=False,
                    calib=True,
                    calib_year="2014"
                )
                print(f"DEBUG: Intensity Stats for {line} -> Min={m.data.min()}, Max={m.data.max()}, Mean={m.data.mean()}")
                print(f"DEBUG: Nonzero pixel count for {line}: {m.data.nonzero()[0].size}")
                # Force-save FITS file
                # === Convert line (e.g., 'ca_14_193.87') into compact label (e.g., 'ca14193_87') ===
                element_label = (
                    line.replace("ca_14_193.87", "ca14193_87")
                        .replace("fe_16_262.98", "fe16262_98")
                        .replace("s_11_188.68", "s11188_68")
                        .replace("si_10_258.37", "si10258_37")
                        .replace("ar_14_194.40", "ar14194_40")  
                        .replace("ar_11_188.81", "ar11188_81")  
                        .replace("s_10_264.23", "s10264_23")   
                        .replace("s_13_256.69", "s13256_69")
                )

                # === Extract datetime from filename (e.g., 'eis_20140202_122934.data.h5') ===
                datetime_str = Path(filename).stem.split("eis_")[-1].split(".")[0]
                datetime_str = f"{datetime_str[:4]}_{datetime_str[4:6]}_{datetime_str[6:8]}__{datetime_str[9:11]}_{datetime_str[11:13]}_{datetime_str[13:15]}"

                # === Final filename ===
                fits_filename = f"{datetime_str}_{element_label}.fits"
                fits_path = os.path.join(custom_intensity_dir, fits_filename)
                if m.data is None:
                    print(f"ERROR: No data returned for line={line}")
                elif not np.any(np.isfinite(m.data)):
                    print(f"ERROR: Data for line={line} is all NaNs or non-finite values")
                elif np.all(m.data == 0):
                    print(f"WARNING: Data for line={line} is all zeros")
                else:
                    print(f"Data looks valid — attempting to save FITS")

                    Map(m.data, m.meta).save(fits_path, overwrite=True)

                print("============================================")
                print(f"Saved intensity map for line={line} in file={filename}")
                print(f"Output location: {custom_intensity_dir}")
                print("============================================")

            except Exception as e:
                print(f"Error generating intensity for line={line} in {filename}: {e}")


def main():
    parser = argparse.ArgumentParser(description="Generate intensity maps for lines used in composition.")
    parser.add_argument('-c', '--cores', type=int, default=4, help='Number of cores to use.')
    parser.add_argument('--test', action="store_true", help="Run in test mode (only ca_14_193.87 on eis_20140206_234547)")
    args = parser.parse_args()

    line_databases = {
        "sis": ['si_10_258.37', 's_10_264.23', 'SiX_SX'],
        "sar": ['s_11_188.68', 'ar_11_188.81', 'SXI_ArXI'],
        "CaAr": ['ca_14_193.87', 'ar_14_194.40', 'CaXIV_ArXIV'],
        "FeS": ['fe_16_262.98', 's_13_256.69', 'FeXVI_SXIII'],
    }

    if args.test:
        filenames = ["SO_EIS_data/eis_20140202_122934.data.h5"]
    else:
        # Read filenames from config.txt
        try:
            with open("config.txt", "r", encoding="utf-8", newline="") as file:
                filenames = [line.strip() for line in file if line.strip()]
        except FileNotFoundError:
            print("ERROR: config.txt not found!")
            return

    # Process each file
    for filename_full in filenames:

        filename = filename_full.replace(" [processing]", "").replace(" [processed]", "")
        if not filename:
            continue

        print(f"\n==========\nProcessing file: {filename}\n==========")
        try:
            make_intensity_maps_for_file(filename, line_databases, ncpu=args.cores, test_mode=args.test)
        except Exception as e:
            print(f"Error while making intensity maps for {filename}: {e}")

if __name__ == "__main__":
    main()
