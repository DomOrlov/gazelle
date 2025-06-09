#================================================================================
import os
import re
import ChiantiPy.core as ch  # type: ignore
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib
import matplotlib.cm as cm
from datetime import datetime


template_dir = "/home/ug/orlovsd2/eispac/eispac/data/templates"

error_log = []

# Get all .h5 template files
template_files = [f for f in os.listdir(template_dir) if f.endswith(".h5")]

low_fip_elements = {"al", "ca", "cr", "fe", "mg", "mn", "ni", "si", "s"}
high_fip_elements = {"ar", "he", "o", "s"} 

spectral_lines = {}

title = {
    "CaAr": "Ca XIV 193.87 Å / Ar XIV 194.40 Å",
    "FeS": "Fe XVI 262.98 Å / S XIII 256.69 Å",
    "SiS": "Si X 258.37 Å / S X 264.23 Å",
    "SAr": "S XI 188.68 Å / Ar XI 188.81 Å"
}

pair_to_element = {
    ("ca_14", "ar_14"): "CaAr",
    ("fe_16", "s_13"): "FeS",
    ("si_10", "s_10"): "SiS",
    ("s_11", "ar_11"): "SAr"
}

plt.rcParams.update({
    "font.size": 10,
    "axes.labelsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "legend.labelspacing": 0.4,
    "axes.titlesize": 10
})

# Extract Fe ions and wavelengths from filenames
for filename in template_files:
    parts = filename.replace("__", "_").split("_")
    if len(parts) >= 3:
        element = parts[0]  # Extract element name
        ion = f"{element}_{parts[1]}"  # Extract ionization state
        wavelength_match = re.search(r"_(\d{3})_(\d{3})", filename)  # Extract wavelength
        if wavelength_match:
            try:
                wavelength = float(f"{wavelength_match.group(1)}.{wavelength_match.group(2)}")  # Convert to Ångströms
            except ValueError:
                print(f"Skipping {filename} due to invalid wavelength format")
                error_log.append(f"Skipping {filename} due to invalid wavelength format")
                continue
            spectral_lines.setdefault(ion, set()).add(wavelength) 


# Convert sets to sorted lists
spectral_lines = {ion: sorted(wavelengths) for ion, wavelengths in spectral_lines.items()}

low_fip_lines = {ion: wavelengths for ion, wavelengths in spectral_lines.items() if ion.split("_")[0] in low_fip_elements and ion.split("_")[1].isdigit()}
high_fip_lines = {ion: wavelengths for ion, wavelengths in spectral_lines.items() if ion.split("_")[0] in high_fip_elements and ion.split("_")[1].isdigit()}

# Temporarily override for validation test
test_mode = True  # Set to False after testing

if test_mode == True:
    matched_pairs = [
        ("ca_14", 193.87, "ar_14", 194.40),
        ("fe_16", 262.98, "s_13", 256.69),
        ("si_10", 258.37, "s_10", 264.23),
        ("s_11", 188.68, "ar_11", 188.81)
    ]

else:
    matched_pairs = [
        (low_ion, low_wvl, high_ion, high_wvl)
        for low_ion, low_wavelengths in low_fip_lines.items()
        for low_wvl in low_wavelengths
        for high_ion, high_wavelengths in high_fip_lines.items()
        for high_wvl in high_wavelengths
        if 2.0 <= abs(low_wvl - high_wvl) <= 8.0  # Adjust for increasing Ångström range
    ]

for pair in matched_pairs:
    print(f"{pair[0]} {pair[1]:.3f} Å / {pair[2]} {pair[3]:.3f} Å")
    error_log.append(f"{pair[0]} {pair[1]:.3f} Å / {pair[2]} {pair[3]:.3f} Å")

## Print matched pairs
#for pair in matched_pairs:
#    print(f"{pair[0]} {pair[1]} Å / {pair[2]} {pair[3]} Å")
#    error_log.append(f"{pair[0]} {pair[1]} Å / {pair[2]} {pair[3]} Å")
    


#================================================================================
# Process matched pairs and calculate emissivities

# Function to find closest spectral lines



T_range = np.logspace(5.8, 7.0, num=120)  # Extend to log(T) = 7 
logT = np.log10(T_range)
electron_densities = [1e8, 1e9, 1e10]
emissivity_data = {}
# ion_cache is already defined earlier in the script, so this line is removed

def find_line_index(ion_obj, target_wavelength):
    """Find the index of the spectral line closest to the target wavelength."""
    if not hasattr(ion_obj, "Wgfa") or "wvl" not in ion_obj.Wgfa:  # Check if the ion has wavelength data
        print(f"Skipping {ion_obj.Spectroscopic} - No 'wvl' data available")
        error_log.append(f"Skipping {ion_obj.Spectroscopic} - No 'wvl' data available")
        return None  # Skip this ion if no wavelength data exists
    wavelengths = np.asarray(ion_obj.Wgfa.get('wvl', []))  # Retrieve the wavelength array
    if wavelengths.size == 0:  # Check if the array is empty
        print(f"Skipping {ion_obj.Spectroscopic} - Empty wavelength data")
        error_log.append(f"Skipping {ion_obj.Spectroscopic} - Empty wavelength data")
        return None
    if low_index is not None and high_index is not None:
        print(f"{low_ion}: Target {low_wvl}Å, CHIANTI closest: {low_ion_obj.Wgfa['wvl'][low_index]}Å")
        error_log.append(f"{low_ion}: Target {low_wvl}Å, CHIANTI closest: {low_ion_obj.Wgfa['wvl'][low_index]}Å")
        print(f"{high_ion}: Target {high_wvl}Å, CHIANTI closest: {high_ion_obj.Wgfa['wvl'][high_index]}Å")   
        error_log.append(f"{high_ion}: Target {high_wvl}Å, CHIANTI closest: {high_ion_obj.Wgfa['wvl'][high_index]}Å") 
    return np.argmin(np.abs(wavelengths - target_wavelength))  # Return the index of the closest wavelength



ion_cache = {}

def get_chianti_ion(ion_name, temperature, eDensity):
    global ion_cache
    temperature_key = tuple(np.round(np.atleast_1d(temperature), decimals=6))
    eDensity_key = float(np.round(eDensity, decimals=6))  # Make it a single float (not a tuple)
    key = (ion_name, temperature_key, eDensity_key)
    if key in ion_cache:
        print(f"Using cached {ion_name} for T={temperature_key}, ne={eDensity_key}")
        error_log.append(f"Using cached {ion_name} for T={temperature_key}, ne={eDensity_key}")
        return ion_cache[key]
    print(f"Loading {ion_name} for T={temperature_key}, ne={eDensity_key}...")
    error_log.append(f"Loading {ion_name} for T={temperature_key}, ne={eDensity_key}...")
    try:
        ion_obj = ch.ion(ion_name, temperature=np.array(temperature), eDensity=np.array([eDensity]))  # Use single-value array
        ion_cache[key] = ion_obj
        return ion_obj
    except Exception as e:
        print(f"Error loading {ion_name}: {e}")
        error_log.append(f"Error loading {ion_name}: {e}")
        return None


## Limit processing to only the first matched pair (e.g., ca_14 and ar_14)
processed_pairs = 0
max_pairs = None  # Allow all pairs
logT_stop = 7

for low_ion, low_wvl, high_ion, high_wvl in matched_pairs:
    print(f"Processing pair {processed_pairs + 1} / {len(matched_pairs)}: {low_ion} {low_wvl}Å / {high_ion} {high_wvl}Å")
    error_log.append(f"Processing pair {processed_pairs + 1} / {len(matched_pairs)}: {low_ion} {low_wvl}Å / {high_ion} {high_wvl}Å")
    
    processed_pairs += 1


print(f"T_range min: {np.log10(min(T_range)):.2f}, max: {np.log10(max(T_range)):.2f}")
error_log.append(f"T_range min: {np.log10(min(T_range)):.2f}, max: {np.log10(max(T_range)):.2f}")

for low_ion, low_wvl, high_ion, high_wvl in matched_pairs:
    if max_pairs is not None and processed_pairs >= max_pairs:
        print(f"Stopping after {processed_pairs} pairs processed.")
        error_log.append(f"Stopping after {processed_pairs} pairs processed.")
        break
    try:
        print(f"\n========== Processing Pair: {low_ion} ({low_wvl}Å) / {high_ion} ({high_wvl}Å) ==========\n")
        error_log.append(f"\n========== Processing Pair: {low_ion} ({low_wvl}Å) / {high_ion} ({high_wvl}Å) ==========\n")
        pair_fully_processed = False  # Track if this pair has been fully processed
        for ne in electron_densities:
            ne = float(ne)  # Ensure electron density is a float
            ne_array = np.array([ne], dtype=float)  # Convert to a 1D NumPy array
            print(f"\nDebug: Electron Density ne={ne}, Type={type(ne)}, Shape={ne_array.shape}")
            error_log.append(f"\nDebug: Electron Density ne={ne}, Type={type(ne)}, Shape={ne_array.shape}")
            try:
                print(f"Debug: Creating ion objects for {low_ion} and {high_ion} with ne_array={ne_array}")
                error_log.append(f"Debug: Creating ion objects for {low_ion} and {high_ion} with ne_array={ne_array}")
                low_ion_obj = ch.ion(low_ion, temperature=np.array([T_range[0]], dtype=float), eDensity=ne_array)
                high_ion_obj = ch.ion(high_ion, temperature=np.array([T_range[0]], dtype=float), eDensity=ne_array)
                print(f"Debug: Successfully created ion objects for {low_ion} and {high_ion}")
                error_log.append(f"Debug: Successfully created ion objects for {low_ion} and {high_ion}")
                # Initialize `low_index` and `high_index` before use
                low_index, high_index = None, None
                try:
                    low_index = find_line_index(low_ion_obj, low_wvl)
                    high_index = find_line_index(high_ion_obj, high_wvl)
                except Exception as e:
                    print(f"Error: Exception occurred in find_line_index - {e}")
                    error_log.append(f"Error: Exception occurred in find_line_index - {e}")
                    continue  # Skip this pair
                if low_index is None:
                    print(f"Warning: No valid index found for {low_ion} at {low_wvl}Å, skipping...")
                    error_log.append(f"Warning: No valid index found for {low_ion} at {low_wvl}Å, skipping...")
                    continue
                if high_index is None:
                    print(f"Warning: No valid index found for {high_ion} at {high_wvl}Å, skipping...")
                    error_log.append(f"Warning: No valid index found for {high_ion} at {high_wvl}Å, skipping...")
                    continue
                low_emissivities = []
                high_emissivities = []
                for T in T_range:
                    print(f"Processing {low_ion} / {high_ion} at log(T) = {np.log10(T):.2f}, ne={ne}")
                    if np.log10(T) > logT_stop:
                        print(f"Skipping log(T) = {np.log10(T):.2f} as it exceeds limit, ne={ne}")
                        continue
                    try:
                        print(f"Debug: Creating ion objects at T={T}, ne={ne}")
                        low_ion_obj = ch.ion(low_ion, temperature=np.array([T], dtype=float), eDensity=ne_array)
                        high_ion_obj = ch.ion(high_ion, temperature=np.array([T], dtype=float), eDensity=ne_array)
                        print(f"Debug: Calling intensity calculation")
                        low_ion_obj.intensity()
                        high_ion_obj.intensity()
                        try:
                            low_emiss = low_ion_obj.Intensity['intensity'][0, low_index]
                            high_emiss = high_ion_obj.Intensity['intensity'][0, high_index]
                            print(f"Debug: Extracted intensities low={low_emiss}, high={high_emiss}")
                            low_emissivities.append(low_emiss)
                            high_emissivities.append(high_emiss)
                        except IndexError as e:
                            print(f"Error: IndexError when accessing intensity data - {e}")
                            continue
                    except Exception as e:
                        print(f"Skipping temperature {T} for {low_ion} / {high_ion} at ne={ne} - CHIANTI error: {e}")
                        continue
                if len(low_emissivities) > 0 and len(high_emissivities) > 0:
                    emissivity_data[(low_ion, low_wvl, high_ion, high_wvl, ne)] = (
                        np.array(low_emissivities),
                        np.array(high_emissivities)
                    )
                    pair_fully_processed = True  # Successfully processed at least one density
            except Exception as e:
                print(f"Skipping density {ne} for {low_ion} {low_wvl}Å / {high_ion} {high_wvl}Å - CHIANTI error: {e}")
                error_log.append(f"Skipping density {ne} for {low_ion} {low_wvl}Å / {high_ion} {high_wvl}Å - CHIANTI error: {e}")
                continue
        if pair_fully_processed:
            processed_pairs += 1
    except Exception as e:
        print(f"Skipping {low_ion} {low_wvl} Å / {high_ion} {high_wvl} Å due to error: {e}")
        error_log.append(f"Skipping {low_ion} {low_wvl} Å / {high_ion} {high_wvl} Å due to error: {e}")


for key, (low_emiss, high_emiss) in emissivity_data.items():
    print(f"{key}: Low FIP Emissivity = {low_emiss}, High FIP Emissivity = {high_emiss}")
    error_log.append(f"{key}: Low FIP Emissivity = {low_emiss}, High FIP Emissivity = {high_emiss}")


for key, (low_emiss, high_emiss) in emissivity_data.items():
    ratio = low_emiss / high_emiss
    print(f"{key}: Ratio (Low FIP / High FIP) = {ratio}")
    error_log.append(f"{key}: Ratio (Low FIP / High FIP) = {ratio}")


#--------------------------------------------------------------------------------
#now we finally plot the data




colors_ratio = ['purple', 'green', 'yellow']  # Colors for the ratio curves
colors_non_ratio = 'black'  # Black for normal emissivity curves
matplotlib.use('Agg')  # Change this to 'TkAgg' if needed, or 'Agg' for saving only

def plot_emissivity_ratios(emissivity_data, logT):
    plt.ioff()  # Turn off interactive mode for better script-based plotting
    unique_pairs = set((key[:4] for key in emissivity_data.keys()))  # Get unique element pairs
    for (low_ion, low_wvl, high_ion, high_wvl) in unique_pairs:
        #plt.title(f"{low_ion.replace('_', ' ')} {low_wvl} & {high_ion.replace('_', ' ')} {high_wvl} emissivities and ratio vs. log T")
        element_key = pair_to_element.get((low_ion, high_ion), f"{low_ion}_{high_ion}")
        low_label, high_label = title[element_key].split(" / ")
        fig, ax = plt.subplots(figsize=(8, 6))  # Single figure per ion pair
        ax.set_yscale('log')
        ax.set_xlabel('Log T (K)')
        ax.set_ylabel('Emissivity and emissivity ratio')
        #ax.set_xlim(6.0, 7.2)  # **Updated to match the good plot**
        ax.set_xlim(5.8, 7.2)
        ax.set_ylim(0.1, 10)  # **Updated to match the good plot**
        ax.set_yticks([1e-2, 1e-1, 1e0, 1e1])
        ax.get_yaxis().set_major_formatter(ticker.ScalarFormatter())
        # Normalize emissivities at ne = 1e9
        ne_ref = 1e9
        key_ref = (low_ion, low_wvl, high_ion, high_wvl, ne_ref)
        if key_ref in emissivity_data:
            low_emiss, high_emiss = emissivity_data[key_ref]
            valid_indices = np.where((low_emiss > 0) & (high_emiss > 0))  # Only valid points
            low_emiss_valid = low_emiss[valid_indices]
            high_emiss_valid = high_emiss[valid_indices]
            if len(low_emiss_valid) > 0 and len(high_emiss_valid) > 0:
                low_emiss_norm = low_emiss / np.max(low_emiss_valid)
                high_emiss_norm = high_emiss / np.max(high_emiss_valid)
            else:
                print(f"Warning: No valid emissivity data for {low_ion} {low_wvl}Å / {high_ion} {high_wvl}Å at ne={ne}")
                error_log.append(f"Warning: No valid emissivity data for {low_ion} {low_wvl}Å / {high_ion} {high_wvl}Å at ne={ne}")
            #ax.plot(logT, low_emiss_norm, 'k-', label=f'{low_ion.upper()} {low_wvl}Å at 1e9')
            #ax.plot(logT, high_emiss_norm, 'k--', label=f'{high_ion.upper()} {high_wvl}Å at 1e9')
            ax.plot(logT, low_emiss_norm, 'k-', label=f'{low_label} at 1e9')
            ax.plot(logT, high_emiss_norm, 'k--', label=f'{high_label} at 1e9')
        # Plot ratios at different densities
        for ne, color, linestyle in zip([1e8, 1e9, 1e10], colors_ratio, [':', '--', '-.']):
            key = (low_ion, low_wvl, high_ion, high_wvl, ne)
            if key in emissivity_data:
                low_emiss, high_emiss = emissivity_data[key]
                low_emiss = np.where(low_emiss > 1e-30, low_emiss, np.nan)
                high_emiss = np.where(high_emiss > 1e-30, high_emiss, np.nan)
                ratio = np.full_like(low_emiss, np.nan)  # Initialize with NaN values
                valid_ratio_indices = (high_emiss > 1e-30)  # Only compute ratio where high_emiss is significant
                ratio[valid_ratio_indices] = low_emiss[valid_ratio_indices] / high_emiss[valid_ratio_indices]
                #ax.plot(logT, ratio, color=color, linestyle=linestyle, label=f'{low_ion.upper()} / {high_ion.upper()} at {int(ne):.0e}')
                ax.plot(logT, ratio, color=color, linestyle=linestyle, label=f'{low_label} / {high_label} at {int(ne):.0e}')
        ax.axhline(1.0, color='gray', linewidth=1)
        ax.legend(loc='best')
        plt.title(f"{title[element_key]} emissivities and ratio vs log T")
        filename = f"{low_ion.replace('_', ' ')} {low_wvl} & {high_ion.replace('_', ' ')} {high_wvl} emissivities and ratio vs. log T".title()
        filename = filename.replace(" ", "_").replace(".", "_") + ".png"        
        plt.savefig(filename, dpi=300)
        plt.show(block=True)  
        plt.close(fig)
        print(f"Saved: {filename}")

# Define log file path (modify the folder as needed)
log_folder = "/home/ug/orlovsd2/gazelle"  # Change this to your desired directory
os.makedirs(log_folder, exist_ok=True)  # Ensure the folder exists

error_log_path = os.path.join(log_folder, "error_log.txt")

# Write the error log to a file
with open(error_log_path, "w", encoding="utf-8") as log_file:
    log_file.write(f"Log start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    log_file.write("=" * 50 + "\n")
    for err in error_log:
        log_file.write(err + "\n")
    log_file.write("\nLog end.\n")

# Run the function
plot_emissivity_ratios(emissivity_data, logT)

