from multiprocessing import Pool
from functools import partial
from time import sleep
from tqdm import tqdm
import numpy as np
import astropy.units as u
from ashmcmc import ashmcmc, interp_emis_temp
import argparse
import platform

from mcmc.mcmc_utils import calc_chi2
from demregpy import dn2dem
import demregpy
import shutil
import os

#adding error logging
from log_config import error_log

'''
This script processes EIS (EUV Imaging Spectrometer) observations to create Differential Emission Measure (DEM) maps and elemental composition maps.

DEM:
- The DEM is a measure of the amount of plasma at different temperatures in the solar corona.
- It is recovered by solving an ill-posed inverse problem using regularized inversion (demregpy).

Why not pure MCMC:
- True MCMC is slow and computationally heavy.
- The DEM is solved using a regularized inversion method (demregpy) which is faster and more efficient.

Main Purpose:
- Solve the DEM for every pixel in an EIS raster using demregpy (regularized inversion).
- The DEM describes how much plasma is present at each temperature for each pixel.
- Different spectral lines (Fe, S, Ca, Ar) emit only at specific temperature ranges (via G(T)).
- By integrating G(T) . DEM(T) over temperature, we predict how bright each line should be if abundances were standard.
- This removes the temperature/density dependence of line brightness and isolates the abundance effect.
- Then, compare the observed brightness to the predicted brightness for certain lines:
    - If the predicted intensity of the high-FIP line is lower than observed, high-FIP abundance is higher than expected.
    - If the low-FIP line intensity is higher than expected, low-FIP element is enhanced.
- These differences are captured as FIP bias (elemental composition maps).

Workflow Overview:
1. Download the EIS data file if not already present (using eispac).
2. Read EIS spectral line intensity and error maps (ashmcmc).
   - Loads the selected spectral lines and their corresponding intensity and uncertainty maps.
   - Reads the density map (needed later for emissivity corrections).
   - Prepares data structures for later per-pixel DEM inversion.
3. For each x-pixel:
    - Loop over all y-pixels in that x-column.
    - For each (x,y) pixel:
        - Select useful emission lines.
        - Retrieve emissivity functions (G(T)) for those lines at the pixel's density.
        - Set up the temperature grid dynamically based on the lines used (adaptive temperature range).
        - Solve the Differential Emission Measure (DEM) inversion using regularized least-squares (demregpy).
        - Store:
            - The reconstructed DEM(T) profile (amount of plasma vs temperature).
            - The chi² value.
            - The list of lines actually used in the fit.
    - After finishing the full column (all y-pixels at a fixed x):
        - Save the results into a temporary file in `dem_columns/`. This minimizes RAM usage and allows parallel column processing.
4. After all columns are processed:
    - Load all the saved per-x-column .npz files from `dem_columns/`.
    - Assemble them into a full 3D DEM cube:
        - 1st axis = Y-pixel (vertical position)
        - 2nd axis = X-pixel (horizontal position)
        - 3rd axis = Temperature bins (logT grid)
    - Also build:
        - A 2D map of chi² values.
        - A 2D map of number of lines used per pixel.
    - Save the full DEM cube and auxiliary maps into a single `.npz` file (*_dem_combined.npz`).
5. For each specified composition ratio (e.g., Si/S, S/Ar, Ca/Ar, Fe/S):
    - Predict the low-FIP line intensity by integrating emissivity DEM over temperature.
    - Rescale the DEM slightly to match the observed low-FIP line intensity (compensating for calibration drifts).
    - Predict the high-FIP line intensity similarly using the scaled DEM.
    - Calculate the composition ratio:
        - Ratio = Predicted High-FIP Line / Observed High-FIP Line
    - Assemble the composition map over the full raster.
    - Save outputs:
        - .fits map of the FIP bias ratio (and png).
'''


def check_dem_exists(filename: str) -> bool:
    # Check if the DEM file exists
    from os.path import exists
    return exists(filename)    

def process_pixel(args: tuple[int, np.ndarray, np.ndarray, list[str], np.ndarray, ashmcmc]) -> None:
    # where I do the DEM calculation
    from pathlib import Path
    # Process a single pixel with the given arguments
    xpix, Intensity, Int_error, Lines, ldens, a = args
    output_file = f'{a.outdir}/dem_columns/dem_{xpix}.npz'
    # Extract the directory path from the output_file
    output_dir = Path(output_file).parent

    # Check if the directory exists, and create it if it doesn't
    output_dir.mkdir(parents=True, exist_ok=True)

    ycoords_out = []
    dem_results = []
    chi2_results = []
    linenames_list = []

    if not check_dem_exists(output_file):
        for ypix in range(Intensity.shape[0]):

            logt, emis, linenames = a.read_emissivity(ldens[ypix, xpix])
            logt_interp = np.log10(interp_emis_temp(logt.value))
            # loc = np.where((np.log10(logt_interp) >= 4) & (np.logt_interp <= 8))
            emis_sorted = a.emis_filter(emis, linenames, Lines)
            mcmc_lines = []

            mcmc_intensity = []
            mcmc_int_error = []
            mcmc_emis_sorted = []
            # Original parameters
            # Full default logT grid: 4.0 to 8.01 in 0.04 steps
            original_dlogt = 0.04
            original_mint = 4 - original_dlogt/2
            original_maxt = 8.01 + original_dlogt/2
            original_temps = 10**np.arange(original_mint, original_maxt, original_dlogt)

            # Dynamically determine temperature range based on lines used
            line_temps = []
            for ind, line in enumerate(Lines): # Use 'Lines' here as it's before filtering, ensuring all possible lines are considered
                if (line[:2] == 'fe') and (Intensity[ypix, xpix, ind] > 5): # Consider only Fe lines for temperature range determination, you can adjust this if needed
                    if line in a.ash.dict:
                        line_temps.append(a.ash.dict[line][2]) # index 2 is peak logT

            if line_temps:
                min_logt_line = min(line_temps)
                max_logt_line = max(line_temps)

                dlogt=0.04
                mint= min_logt_line - 0.1 # Adjust buffer as needed, e.g., 0.2 below min temp line
                maxt= max_logt_line + 0.1 # Adjust buffer as needed, e.g., 0.2 above max temp line
                # Ensure mint and maxt are within a reasonable range (e.g., 4 to 8)
                mint = max(mint, 4.0)
                maxt = min(maxt, 8.01)
                
                temps=10**np.arange(mint,maxt,dlogt) # Temperture grid you want to solve

                start_index = np.searchsorted(original_temps, temps[0])
                end_index = np.searchsorted(original_temps, temps[-1], side='right')

                for ind, line in enumerate(Lines):
                    if (line[:2] == 'fe') and (Intensity[ypix, xpix, ind] > 5):
                        mcmc_intensity.append(Intensity[ypix, xpix, ind])
                        mcmc_int_error.append(Int_error[ypix, xpix,ind]+Intensity[ypix, xpix,ind]*0.23)
                        # mcmc_int_error.append(Intensity[ypix, xpix,ind]*0.2)
                        mcmc_emis_sorted.append(emis_sorted[ind, :])
                        mcmc_lines.append(line)
                        line_temps.append(a.ash.dict[line][2]) # index 2 is peak logT

                if mcmc_emis_sorted:
                    nt = len(mcmc_emis_sorted[0])
                    nf = len(mcmc_emis_sorted)
                    trmatrix = np.zeros((nt,nf))
                    for i in range(0,nf):
                        trmatrix[:,i] = mcmc_emis_sorted[i]

                    # doing DEM calculation
                    dem,edem0,elogt0,chisq0,dn_reg0=demreg_process_wrapper(np.array(mcmc_intensity),np.array(mcmc_int_error),np.array(mcmc_emis_sorted),logt_interp,temps)
                    dem0 = np.zeros(len(original_temps) - 1)
                    # Fill in the calculated DEM values at the correct indices
                    dem0[start_index:end_index] = dem

                    chi2 = np.sum(((pred_intensity_compact(mcmc_emis_sorted, logt, dem0) - np.array(mcmc_intensity))/np.array(mcmc_int_error))**2)
                    # chi2 = calc_chi2(dn_reg0, np.array(mcmc_intensity), np.array(mcmc_int_error))
                    dem_results.append(dem0)
                    chi2_results.append(chi2)
            else:
                dem_results.append(np.zeros(len(original_temps)-1))
                chi2_results.append(np.inf)

            ycoords_out.append(ypix)
            linenames_list.append(mcmc_lines)

        dem_results = np.array(dem_results)
        chi2_results = np.array(chi2_results)
        linenames_list = np.array(linenames_list, dtype=object)

        np.savez(output_file, dem_results=dem_results, chi2=chi2_results, ycoords_out=ycoords_out, lines_used=linenames_list, logt = np.array(logt_interp))

def download_data(filename: str) -> None:
    from eispac.download import download_hdf5_data
    import os

    if not filename:  # Check if filename is empty
        error_log.append("Filename cannot be empty")
        raise ValueError("Filename cannot be empty")

    # Extract the local_top directory from the filename
    local_top = os.path.dirname(filename)

    # If local_top is empty, use current directory
    if not local_top:
        local_top = '.'

    # Create the directory if it doesn't exist
    if not os.path.exists(local_top):
        os.makedirs(local_top)
        print(f"Created directory: {local_top}")

    # Extract just the filename without the path
    file_name = os.path.basename(filename)
    
    if not file_name:  # Check if file_name is empty after basename
        error_log.append("Invalid filename format")
        raise ValueError("Invalid filename format")

    print(f"Attempting to download {file_name} to {local_top}...")

    download_hdf5_data(file_name, local_top=local_top, overwrite=False)
    #try:
    #    # Attempt to download the file
    #    download_hdf5_data(file_name, local_top=local_top, overwrite=False)

    #    # Verify if the file was successfully downloaded and is not empty
    #    expected_path = os.path.join(local_top, file_name)
    #    if not os.path.isfile(expected_path) or os.path.getsize(expected_path) == 0:
    #        print(f"File {file_name} does not exist on the server or is incomplete. Skipping download.")
    #        if os.path.exists(expected_path):
    #            os.remove(expected_path)  # Clean up any partial downloads
    #        return
    
    #    print(f"Download completed: {file_name}")
    
    #except Exception as e:
    #    print(f"Error downloading {file_name}: {str(e)}. Skipping download.")

def combine_dem_files(xdim:int, ydim:int, dir: str, delete=False) -> np.array:
    from glob import glob
    from re import search

    dem_files = sorted(glob(f'{dir}/dem_columns/dem*.npz'))
    # print(dem_files)
    ref = np.load(dem_files[0])['dem_results'].shape
    dem_combined = np.zeros((ydim,xdim,ref[1]))
    chi2_combined = np.zeros((ydim,xdim))
    lines_used = np.zeros((ydim,xdim))
    logt = np.load(dem_files[0])['logt']

    for file_num, dem_file in enumerate(tqdm(dem_files)): # Goes through all column files, file_num is the column number, dem_file is the file name
        # print(dem_file)
        xpix_loc = search(r'dem_(\d+)\.npz$', dem_file).group(1) # Extract the X-pixel index from filename
        # print(xpix_loc)
        dem_combined[:,int(xpix_loc), :] = np.load(dem_file)['dem_results'] # Loads the column's DEM result (shape: [Y, T]) and inserts it into the correct X position.
        chi2_combined[:,int(xpix_loc)] = np.load(dem_file)['chi2'] 
        lines_used[:,int(xpix_loc)] = np.array([len(line) for line in np.load(dem_file, allow_pickle=True)['lines_used']]) #Loads the list of spectral lines used per pixel in that column.

        # Collect full line name list per pixel
        if file_num == 0:
            lines_used_names = np.empty((ydim, xdim), dtype=object)

        line_list_column = np.load(dem_file, allow_pickle=True)['lines_used'] # Save the line names used at each pixel.
        for ypix in range(len(line_list_column)):
            lines_used_names[ypix, int(xpix_loc)] = line_list_column[ypix]

    directory_to_delete = os.path.join(dir, 'dem_columns')
    if os.path.exists(directory_to_delete):
        shutil.rmtree(directory_to_delete)
        print(f'Directory {directory_to_delete} has been deleted successfully.')
    else:
        print(f'Directory {directory_to_delete} does not exist.')

    return dem_combined, chi2_combined, lines_used, logt, lines_used_names

def demreg_process_wrapper(mcmc_intensity, mcmc_int_error, mcmc_emis_sorted, logt_interp, temps) -> float: # solves how much plasma is present at each temperature
    max_iter = 1000
    l_emd = False
    reg_tweak = 1
    rgt_fact = 2
    dn_in=np.array(mcmc_intensity) # The observed line intensities
    edn_in=np.array(mcmc_int_error) # The error bars for each of those intensities.
    tresp_logt = logt_interp
    # set up our target dem temps
    nt = len(mcmc_emis_sorted[0])
    nf = len(mcmc_emis_sorted) 
    trmatrix = np.zeros((nt,nf)) # Creates empty matrix 
    trmatrix = np.array(mcmc_emis_sorted).T # Emissivity, how bright that line is per unit plasma at that temperature.
    dem1,edem1,elogt1,chisq1,dn_reg1=dn2dem(dn_in,edn_in,trmatrix,tresp_logt,temps,max_iter=1000,l_emd=True,emd_int=True,gloci=1,reg_tweak=0.001,rgt_fact=1.05) # core demreg function, takes observed intensities and errors, emisitivity fucntions, tempature grid
    # max_iter=1000 is how many iterations the solver can do. 
    # l_emd=True is our zeroth order constraint, controls how smooth the dem is.
    # emd_int=true is our positivity constraint, forces ≥ 0 at every bin.
    # gloci=1 EM_loci(T) = I_obs / G(T): how much plasma would be needed at each T to produce the observed line intensity. 
    # i.e computes EM loci curves, uses their minimum as initial guess for DEM(T), prevents starting at 0, which would take longer.
    # reg_tweak=0.001 tries hard to match the data.
    # rgt_fact=1.05 how agressively λ is adapted. In each iteration, λ can increase up to 5% if needed to improve stability.

    # dem1,edem1,elogt1,chisq1,dn_reg1=dn2dem(dn_in,edn_in,trmatrix,tresp_logt,temps,max_iter=1000,l_emd=True,emd_int=True,gloci=1,reg_tweak=0.3,rgt_fact=1.01)
    return dem1,edem1,elogt1,chisq1,dn_reg1


def process_data(filename: str, num_processes: int) -> None:
    # Create an ashmcmc object with the specified filename
    download_data(filename)
    a = ashmcmc(filename, ncpu = num_processes)

    # Retrieve necessary data from ashmcmc object
    Lines, Intensity, Int_error = a.fit_data(plot=False)
    ldens = a.read_density()

    # Generate a list of arguments for process_pixel function
    args_list = [(xpix, Intensity, Int_error, Lines, ldens, a) for xpix in range(Intensity.shape[1])]

    # Create a Pool of processes for parallel execution
    with Pool(processes=num_processes) as pool:
        # Use imap to apply process_pixel to each xpix in args_list
        # Wrap the pool.imap call with tqdm for a progress bar on the columns
        list(tqdm(pool.imap(process_pixel, args_list), total=len(args_list), desc="Processing Columns"))
        

    # Combine the DEM files into a single array
    print('------------------------------Combining DEM files------------------------------')
    dem_combined, chi2_combined, lines_used, logt, lines_used_names = combine_dem_files(Intensity.shape[1], Intensity.shape[0], a.outdir, delete=True)
    np.savez(f'{a.outdir}/{a.outdir.split("/")[-1]}_dem_combined.npz',
         dem_combined=dem_combined,
         chi2_combined=chi2_combined,
         lines_used=lines_used,
         lines_used_names=lines_used_names,
         logt=logt)
    
    return f'{a.outdir}/{a.outdir.split("/")[-1]}_dem_combined.npz', a.outdir

def pred_intensity_compact(emis: np.array, logt: np.array, dem: np.array) -> float:
    """
    Calculate the predicted intensity for a given emissivity, temperature, and DEM.
    
    Parameters:
    emis (np.array): Emissivity array
    logt (np.array): Log temperature array
    dem (np.array): Differential Emission Measure array
    
    Returns:
    float: Predicted intensity
    """
    # Ensure all inputs are numpy arrays
    emis = np.array(emis)
    logt = np.array(logt)
    dem = np.array(dem)
    # print(emis.shape, logt.shape, dem.shape)
    # Calculate the temperature array
    temp = logt
    
    # Calculate the integrand
    integrand = emis * dem
    
    # Perform the integration using the trapezoidal rule
    intensity = np.trapz(integrand, temp)
    
    return intensity
    
def correct_metadata(map, ratio_name):
    # Correct the metadata of the map
    map.meta['measrmnt'] = 'FIP Bias'
    map.meta.pop('bunit', None)
    map.meta['line_id'] = ratio_name
    return map

def calc_composition_parallel(args):
    ypix, xpix, ldens, dem_median, intensities, line_databases, comp_ratio, a = args
    logt, emis, linenames = a.read_emissivity(ldens[ypix, xpix])
    logt_interp = interp_emis_temp(logt.value)
    emis_sorted = a.emis_filter(emis, linenames, line_databases[comp_ratio][:2])
    int_lf = pred_intensity_compact(emis_sorted[0], logt_interp, dem_median[ypix, xpix])
    dem_scaled = dem_median[ypix, xpix] * (intensities[ypix, xpix, 0] / int_lf)
    int_hf = pred_intensity_compact(emis_sorted[1], logt_interp, dem_scaled)
    fip_ratio = int_hf / intensities[ypix, xpix, 1]
    return ypix, xpix, fip_ratio

def calc_composition(filename, np_file, line_databases, num_processes):
    """
    Calculate the composition of a given file using multiprocessing.

    Parameters:
    - filename (str): The name of the file to calculate the composition for.
    - np_file (str): The name of the numpy file containing the DEM data.
    - line_databases (dict): A dictionary containing line databases for different composition ratios.
    - num_processes (int): The number of processes to use for parallel processing.

    Returns:
    None
    """
    from sunpy.map import Map
    from multiprocessing import Pool

    a = ashmcmc(filename, ncpu=num_processes)
    ldens = a.read_density()
    dem_data = np.load(np_file)
    dem_median = dem_data['dem_combined']

    for comp_ratio in line_databases:
        try:
            intensities = np.zeros((ldens.shape[0], ldens.shape[1], 2))
            composition = np.zeros_like(ldens)

            # Read the intensity maps for the composition lines
            for num, fip_line in enumerate(line_databases[comp_ratio][:2]):
                print('getting intensity \n')
                map = a.ash.get_intensity(fip_line, outdir=a.outdir, plot=False, calib=True)
                intensities[:, :, num] = map.data

            # Create argument list for parallel processing
            args_list = [(ypix, xpix, ldens, dem_median, intensities, line_databases, comp_ratio, a)
                        for ypix, xpix in np.ndindex(ldens.shape)]

            # Create a pool of worker processes
            with Pool(processes=num_processes) as pool:
                results = pool.map(calc_composition_parallel, args_list)

            # Update composition array with the results
            for ypix, xpix, fip_ratio in results:
                composition[ypix, xpix] = fip_ratio

            np.savez(f'{a.outdir}/{a.outdir.split("/")[-1]}_composition_{comp_ratio}.npz',
                    composition=composition, chi2=dem_data['chi2_combined'], no_lines=dem_data['lines_used'])

            map_fip = Map(composition, map.meta)
            map_fip = correct_metadata(map_fip, comp_ratio)
            map_fip.save(f'{a.outdir}/{a.outdir.split("/")[-1]}_{comp_ratio}.fits', overwrite=True)
            # Plot and save map_fip as PNG
            import matplotlib.pyplot as plt
            from astropy.visualization import ImageNormalize, LinearStretch

            plt.figure(figsize=(5, 5))
            if comp_ratio == "sar":  # Assuming "sar" corresponds to s_11 and ar_11
                norm = ImageNormalize(vmin=1, vmax=1.5)
            else:
                norm = ImageNormalize(vmin=1, vmax=4)
            plt.imshow(map_fip.data, origin='lower', norm=norm, cmap='viridis')
            plt.colorbar(label=f'{comp_ratio} Ratio')
            plt.title(f'{comp_ratio} Composition Map - {a.outdir.split("/")[-1]}')
            plt.xlabel('Solar X [pixels]')
            plt.ylabel('Solar Y [pixels]')
            plt.tight_layout()
            plt.savefig(f'{a.outdir}/{a.outdir.split("/")[-1]}_{comp_ratio}_map.png', dpi=300, bbox_inches='tight')
            plt.close()
        except Exception as e:
            print(f"Error processing {comp_ratio}: {str(e)}")
            error_log.append(f"Error processing {comp_ratio}: {e}")
            continue


def update_filenames_txt(old_filename, new_filename):
    try:
        with open("config.txt", "r") as file:
            lines = file.readlines()

        with open("config.txt", "w") as file:
            for line in lines:
                if line.strip() == old_filename:
                    file.write(new_filename + "\n")
                else:
                    file.write(line)
    except Exception as e:
            print(f"Error updating config.txt: {str(e)}")
            error_log.append(f"Error updating config.txt: {e}")

if __name__ == "__main__":
    # Determine the operating system type (Linux or macOS)
    # Set the default number of cores based on the operating system
    if platform.system() == "Linux":
        default_cores = 60  # above 64 seems to break the MSSL machine - probably due to no. cores = 64?
    elif platform.system() == "Darwin":
        default_cores = 10
    else:
        default_cores = 10

    # Create an argument parser
    parser = argparse.ArgumentParser(description='Process data using multiprocessing.')
    parser.add_argument('-c', '--cores', type=int, default=default_cores,
                        help='Number of cores to use (default: {})'.format(default_cores))
    parser.add_argument('--unsave', action='store_true',
                        help='Delete dem_columns folder after processing (default: False)')
    args = parser.parse_args()

    try:
        # Read filenames from a text file
        with open("config.txt", "r") as file:
            filenames = [line.strip() for line in file if line.strip()]  # Skip empty lines

        for file_num, filename_full in enumerate(filenames):
            filename = filename_full.replace(" [processing]", '').replace(" [processed]", '')
            
            if not filename:  # Skip if filename is empty
                continue

            # Re-read the config.txt file to get the latest information
            with open("config.txt", "r") as file:
                current_filenames = [line.strip() for line in file]

            filename_full = current_filenames[file_num]
            if not filename_full.endswith("[processed]") and not filename_full.endswith("[processing]"):
                try:
                    # Add "[processing]" to the end of the filename in filenames.txt
                    processing_filename = filename + " [processing]"
                    update_filenames_txt(filename_full, processing_filename)
                    print(f"Processing: {filename}")
                    np_file, outdir = process_data(filename, args.cores)
                    print(f"Processed: {filename}")
                    
                    line_databases = {
                        "sis": ['si_10_258.37', 's_10_264.23', 'SiX_SX'],
                        "sar": ['s_11_188.68', 'ar_11_188.81', 'SXI_ArXI'],
                        "CaAr": ['ca_14_193.87', 'ar_14_194.40', 'CaXIV_ArXIV'],
                        "FeS": ['fe_16_262.98', 's_13_256.69', 'FeXVI_SXIII'],
                    }
                    calc_composition(filename, np_file, line_databases, args.cores)

                    # Change "[processing]" to "[processed]" in filenames.txt after processing is finished
                    processed_filename = filename + " [processed]"
                    update_filenames_txt(processing_filename, processed_filename)

                    if args.unsave:
                        dem_column_dir = os.path.join(outdir, 'dem_columns')
                        if os.path.exists(dem_column_dir):
                            shutil.rmtree(dem_column_dir)
                            print(f"Deleted folder: {dem_column_dir}")

                except Exception as e:
                    print(f"Error processing {filename}: {str(e)}")
                    error_log.append(f"Error processing {filename}: {e}")
                    #If there's an error, remove the [processing] tag
                    update_filenames_txt(processing_filename, filename)
                    continue

    except Exception as e:
        print(f"Fatal error: {str(e)}")
        error_log.append(f"Fatal error: {e}")

# Write errors to a log file
with open("error_log.txt", "w", encoding="utf-8") as log_file:
    for err in error_log:
        log_file.write(err + "\n")



# how to call
# need to create a config.txt file with the filenames of the data files
# e.g.
# SO_EIS_data/eis_20230327_061218.data.h5
# Then run the following command in terminal:
# python mcmc_para.py –-cores 50
#
# Custom setting locations:
# ashmcmc.py - read_emissivity - abund_file = 'emissivities_sun_photospheric_2015_scott'
# ashmcmc.py - find_matching_file - change abundance file directory
# asheis.py - class asheis (line 71) - input the directory where the IDL density interpolation files are stored
# asheis.py - line 137 - 2014 calibration hard coded
