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
                
                temps=10**np.arange(mint,maxt,dlogt)

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
        raise ValueError("Invalid filename format")
        
    download_hdf5_data(file_name, local_top=local_top, overwrite=False)

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

    for dem_file in tqdm(dem_files):
        # print(dem_file)
        xpix_loc = search(r'dem_(\d+)\.npz$', dem_file).group(1)
        # print(xpix_loc)
        dem_combined[:,int(xpix_loc), :] = np.load(dem_file)['dem_results'] 
        chi2_combined[:,int(xpix_loc)] = np.load(dem_file)['chi2'] 
        lines_used[:,int(xpix_loc)] = np.array([len(line) for line in np.load(dem_file, allow_pickle=True)['lines_used']])

    directory_to_delete = os.path.join(dir, 'dem_columns')
    if os.path.exists(directory_to_delete):
        shutil.rmtree(directory_to_delete)
        print(f'Directory {directory_to_delete} has been deleted successfully.')
    else:
        print(f'Directory {directory_to_delete} does not exist.')

    return dem_combined, chi2_combined, lines_used, logt

def demreg_process_wrapper(mcmc_intensity, mcmc_int_error, mcmc_emis_sorted, logt_interp, temps) -> float:
    max_iter = 1000
    l_emd = False
    reg_tweak = 1
    rgt_fact = 2
    dn_in=np.array(mcmc_intensity)
    edn_in=np.array(mcmc_int_error)
    tresp_logt = logt_interp
    # set up our target dem temps
    nt = len(mcmc_emis_sorted[0])
    nf = len(mcmc_emis_sorted) 
    trmatrix = np.zeros((nt,nf))
    trmatrix = np.array(mcmc_emis_sorted).T
    dem1,edem1,elogt1,chisq1,dn_reg1=dn2dem(dn_in,edn_in,trmatrix,tresp_logt,temps,max_iter=1000,l_emd=True,emd_int=True,gloci=1,reg_tweak=0.001,rgt_fact=1.05)

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
    dem_combined, chi2_combined, lines_used, logt = combine_dem_files(Intensity.shape[1], Intensity.shape[0], a.outdir, delete=True)
    np.savez(f'{a.outdir}/{a.outdir.split("/")[-1]}_dem_combined.npz', dem_combined=dem_combined, chi2_combined=chi2_combined, lines_used=lines_used, logt=logt)
    
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
        #try:
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
        #except Exception as e:
        #    print(f"Error processing {comp_ratio}: {str(e)}")
        #    continue


def update_filenames_txt(old_filename, new_filename):
    #try:
    with open("config.txt", "r") as file:
        lines = file.readlines()

    with open("config.txt", "w") as file:
        for line in lines:
            if line.strip() == old_filename:
                file.write(new_filename + "\n")
            else:
                file.write(line)
#except Exception as e:
#        print(f"Error updating config.txt: {str(e)}")

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

    #try:
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
            #try:
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

            #except Exception as e:
            #    print(f"Error processing {filename}: {str(e)}")
                # If there's an error, remove the [processing] tag
                update_filenames_txt(processing_filename, filename)
                continue

    #except Exception as e:
    #    print(f"Fatal error: {str(e)}")

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
