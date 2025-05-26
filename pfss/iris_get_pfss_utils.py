from astropy.io import fits
import numpy as np
import astropy.coordinates
import astropy.units as u
from pfss.functions_data import aia_download_from_date, hmi_daily_download, aia_correction, PrepHMIdaily
import matplotlib.pyplot as plt
import pfsspy
import sunpy
from sunpy.map import Map
from tqdm import tqdm
from glob import glob
from aiapy.calibrate import correct_degradation, update_pointing
from os import makedirs
from astropy.coordinates import SkyCoord
from sunpy.net import Fido, attrs
from sunpy.coordinates.sun import carrington_rotation_number, carrington_rotation_time
from sunpy.time import parse_time
from datetime import datetime, timedelta
import warnings
import pickle
import time
import re
import glob
from pfsspy.fieldline import OpenFieldLines, ClosedFieldLines
from sunpy.coordinates import Helioprojective
from astropy.coordinates import BaseCoordinateFrame
from datetime import timedelta
from sunpy.physics.differential_rotation import solar_rotate_coordinate



# Additional imports that might be required based on the code context
from sunpy.net import attrs as a
from astropy.wcs import WCS
from astropy.visualization import ImageNormalize, SqrtStretch
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.patches import Rectangle

warnings.filterwarnings('ignore')

def get_closest_aia(date_time_obj, wavelength = 193):
    # Add this line near the top of the file, after the imports
    DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

    """
    Get the closest AIA image to a given datetime for a specified wavelength.

    This function searches for and uses the local AIA image closest to the input datetime.
    If no local file is found, Fido can be used as a fallback (currently commented out).

    Parameters:
    -----------
    date_time_obj : datetime.datetime
        The target datetime for which to find the closest AIA image.
    wavelength : int, optional
        The desired wavelength of the AIA image in Angstroms. Default is 193.

    Returns:
    --------
    sunpy.map.sources.sdo.AIAMap
        A SunPy Map object containing the closest AIA image for the specified wavelength.

    Notes:
    ------
    - Uses Fido to search and fetch AIA data.
    - The search window is set to ±30 seconds around the input datetime.
    - Assumes DATE_FORMAT and necessary imports (e.g., Fido, attrs) are defined elsewhere in the code.
    """
    aia_start = date_time_obj - timedelta(minutes=0.5)
    aia_end = date_time_obj + timedelta(minutes=0.5)
    aia_start_str = aia_start.strftime(DATE_FORMAT)[:-3]
    aia_end_str = aia_end.strftime(DATE_FORMAT)[:-3]
    
    aia_local_dir = "/home/ug/orlovsd2/gazelle/pfss/AIA/" #change this to your local directory
    aia_files = sorted(glob.glob(f"{aia_local_dir}/*.fits"))

    for aia_file in aia_files:
        match = re.search(r"(\d{4}_\d{2}_\d{2}T\d{2}_\d{2}_\d{2})", aia_file)
        if match:
            filename_time_str = match.group(1)  # Extract timestamp from filename
            formatted_time_str = filename_time_str.replace("_", "-").replace("T", " ")  # Convert format
            filename_time = datetime.strptime(formatted_time_str, "%Y-%m-%d %H-%M-%S")  # Convert to datetime

            if aia_start <= filename_time <= aia_end:
                print(f"Using local AIA file: {aia_file}")
                return sunpy.map.Map(aia_file)  

    print(f"No local AIA file found within ±30 seconds for {date_time_obj}")
    return None  # If no valid file is found


    #max_retries = 5
    #retry_delay = 120
    
    #for attempt in range(max_retries):
    #    try:
    #        # Search for AIA data
    #        search_results = Fido.search(a.Time(aia_start_str, aia_end_str), a.Instrument.aia, a.Wavelength(wavelength*u.angstrom))
                    
    #        if search_results:
    #            # Download the AIA file
    #            aia_downloads = Fido.fetch(search_results)
                        
    #            if aia_downloads:  # Ensure at least one file was downloaded
    #                return sunpy.map.Map(aia_downloads[0])
    #            else:
    #                raise ValueError("AIA data download failed, empty list returned.")

    #        else:
    #            raise ValueError("No AIA data found for the requested time.")

    #    except Exception as e:
    #        print(f"Attempt {attempt+1} failed: {e}")
    #        time.sleep(retry_delay)

    #    print(f"Skipping AIA data for {date_time_obj} after {max_retries} failed attempts.")
    #    return None  # Return None instead of crashing


def plot_context_pfss(eis_map):
    """
    Plot the PFSS model field lines over an AIA context image with a zoomed-in view.

    This function takes an EIS map, generates PFSS field lines,
    plots them over a corresponding AIA 193Å image for context,
    and adds a zoomed-in view of the AIA image around the EIS field of view.

    Parameters:
    -----------
    eis_map : sunpy.map.GenericMap
        The input EIS map used to define the field of view.

    Returns:
    --------
    None
        The function displays the plot but does not return any value.
    """
    # Get the AIA map closest to the EIS observation time
    aia_map = get_closest_aia(eis_map.date.datetime)
    
    # Generate PFSS field lines
    pfss_fieldlines = get_pfss_from_map(eis_map)
    
    # Create the figure with two subplots
    fig = plt.figure(figsize=(20, 10))
    ax1 = fig.add_subplot(121, projection=aia_map)
    ax2 = fig.add_subplot(122, projection=aia_map)
    
    # Plot the full AIA map
    aia_map.plot(axes=ax1, vmin=0, vmax=5000)
    
    # Define functions to change the observer time and frame
    change_obstime = lambda x,y: SkyCoord(x.replicate(observer=x.observer.replicate(obstime=y), obstime=y))
    change_obstime_frame = lambda x,y: x.replicate_without_data(observer=x.observer.replicate(obstime=y), obstime=y)

    # Define field line categories and their properties
    fieldline_categories = [
        ('Large closed', lambda f: f.coords.shape[0] > 9000, '#E6BC52', 2.0),
        ('Small closed', lambda f: f.coords.shape[0] < 1800, 'black', 1.0),
        ('Open', lambda f: True, 'white', 1.5)
    ]
    
    # Plot field lines on the full map
    for category, condition, color, linewidth in fieldline_categories:
        fieldlines = [f for f in pfss_fieldlines.closed_field_lines if condition(f)] if category != 'Open' else pfss_fieldlines.open_field_lines
        for fline in tqdm(fieldlines, desc=f'Plotting {category} field lines'):
            coords = fline.coords if category != 'Open' else fline.coords
            if coords.shape[0] > 0:  # Only plot if there are coordinates
                # Change the observer time of the coordinates to match the AIA map
                coords_updated = change_obstime(coords, aia_map.date)
                ax1.plot_coord(coords_updated, color=color, linewidth=linewidth, alpha=0.7)
    
    # Plot EIS map field of view as a rectangle on both plots
    bottom_left = eis_map.bottom_left_coord
    top_right = eis_map.top_right_coord

    aia_map.draw_quadrangle(bottom_left, axes=ax1, top_right=top_right, edgecolor="black", linestyle="-", linewidth=2)

    # Set plot properties for the full map
    ax1.set_title('AIA 193 Å with PFSS Field Lines and EIS FOV', fontsize=16)
    ax1.set_xlabel('Solar-X [arcsec]', fontsize=14)
    ax1.set_ylabel('Solar-Y [arcsec]', fontsize=14)
    ax1.tick_params(axis='both', labelsize=12)
    
    # Add legend to the full map
    legend_elements = [
        plt.Line2D([0], [0], color='#E6BC52', lw=2, label='Large closed'),
        plt.Line2D([0], [0], color='black', lw=1, label='Small closed'),
        plt.Line2D([0], [0], color='white', lw=1.5, label='Open'),
        plt.Rectangle((0, 0), 1, 1, fill=False, edgecolor='black', linewidth=2, label='EIS FOV')
    ]
    ax1.legend(handles=legend_elements, loc='upper right', fontsize=12)
    
    # Plot zoomed-in AIA map
    center_x = (bottom_left.Tx + top_right.Tx) / 2
    center_y = (bottom_left.Ty + top_right.Ty) / 2
    width = (top_right.Tx - bottom_left.Tx) * 1.2  # Add 20% buffer
    height = (top_right.Ty - bottom_left.Ty) * 1.2  # Add 20% buffer
    
    # Ensure the zoomed region is square
    size = max(width, height)
    
    bottom_left_zoom = SkyCoord(center_x - size/1.2, center_y - size/1.2, unit=u.arcsec, frame=aia_map.coordinate_frame)
    top_right_zoom = SkyCoord(center_x + size/1.2, center_y + size/1.2, unit=u.arcsec, frame=aia_map.coordinate_frame)
    
    aia_submap = aia_map.submap(bottom_left_zoom, top_right=top_right_zoom)
    aia_submap.plot(axes=ax2)
    aia_submap.draw_quadrangle(bottom_left, axes=ax2, top_right=top_right, edgecolor="black", linestyle="-", linewidth=1)

    # Set plot properties for the zoomed-in map
    ax2.set_title('Zoomed AIA 193 Å with EIS FOV', fontsize=16)
    ax2.set_xlabel('Solar-X [arcsec]', fontsize=14)
    ax2.set_ylabel('Solar-Y [arcsec]', fontsize=14)
    ax2.tick_params(axis='both', labelsize=12)
    
    # Show the plot
    plt.tight_layout()
    plt.show()
    
def get_pfss_from_map(map, min_gauss = -20, max_gauss = 20, dimension = (1080, 540)):
    
    map.plot()
    plt.title("Raw EIS Map")
    plt.show() # This is the region of interest, the EIS map, that we want to trace fieldlines.

    print(f"EIS map shape: {map.data.shape}")
    print(f"WCS dimensions: {map.dimensions}")


    ## hmi synoptic maps provide a global magentic map of the sun to trace fieldlines
    #m_hmi = hmi_daily_download(map.date.value)

    # Get both HMI maps
    m_hmi_today = hmi_daily_download(map.date)
    m_hmi_yesterday = hmi_daily_download(map.date - timedelta(days=1))

    # Choose the one closest in time to the EIS observation
    delta_today = abs(m_hmi_today.date - map.date)
    delta_yesterday = abs(m_hmi_yesterday.date - map.date)

    m_hmi = m_hmi_today if delta_today < delta_yesterday else m_hmi_yesterday
    print(f"Selected HMI date: {m_hmi.date} (closer to EIS time {map.date})")

    m_hmi.plot()
    plt.title("Raw HMI Magnetogram")
    plt.show() # Confirms it covers the area of interest.

    # Define functions to change the observer time and frame
    change_obstime = lambda x,y: SkyCoord( # x original Skycoord, y = new time
        x.replicate( # makes a copy of x
            observer=x.observer.replicate(obstime=y), # takes original observer and makes a copy of it with a new time, y.
            obstime=y # sets the new time for the copy of x
        )
    )
    #change_obstime = lambda x, y: SkyCoord(
    #    x.replicate(
    #        observer=x.observer if isinstance(x.observer, BaseCoordinateFrame) else x.observer.frame,
    #        obstime=y
    #    )
    #)

    change_obstime_frame = lambda x,y: x.replicate_without_data( #original frame, y = new time
        observer=x.observer.replicate(obstime=y), # Makes a copy of the frame without copying any coordinate data inside.
        obstime=y
    )
    
    # Synchronize the original HMI magnetogram frame to EIS observation time
    new_frame = change_obstime_frame(m_hmi.coordinate_frame, map.date)

    # Resample the HMI data to a specific resolution
    m_hmi_resample = m_hmi.resample(dimension * u.pix) # .resample changes the number of pixels in the map, and stretching/compressing the coordinate system to match the new pixel size.

    m_hmi_resample.plot()
    plt.title("Resampled HMI Magnetogram")
    plt.show() # Check that resampling didn't distort or lose critical details, if it matches better the EIS map scale (pixels).

    # Synchronize again guaranteeing that the HMI data is in the same frame as the EIS map.
    new_frame = change_obstime_frame(m_hmi_resample.coordinate_frame, map.date)

    print("===== FRAME COMPARISON =====")
    print("EIS observer:", map.observer_coordinate)
    print("EIS obstime:", map.date)
    print("HMI observer:", m_hmi_resample.observer_coordinate)
    print("HMI obstime:", m_hmi_resample.date)
    print("EIS coordinate frame:", map.coordinate_frame)
    print("HMI coordinate frame:", m_hmi_resample.coordinate_frame)
    print("============================")

    # Expand the coordinates by 10% in each direction, ensures you don’t accidentally miss important fieldlines touching the edges.
    # Magnetic fields often curve outward, fieldlines that start near the edge might still be important.
    blc_ar_synop = change_obstime(
        SkyCoord(
            map.bottom_left_coord.Tx - 0.1 * (map.top_right_coord.Tx - map.bottom_left_coord.Tx),
            map.bottom_left_coord.Ty - 0.1 * (map.top_right_coord.Ty - map.bottom_left_coord.Ty),
            frame=map.coordinate_frame
        ).transform_to(new_frame), # transform the coordinate to the new magnetogram frame.
        m_hmi_resample.date
    )

    trc_ar_synop = change_obstime(
        SkyCoord(
            map.top_right_coord.Tx + 0.1 * (map.top_right_coord.Tx - map.bottom_left_coord.Tx),
            map.top_right_coord.Ty + 0.1 * (map.top_right_coord.Ty - map.bottom_left_coord.Ty),
            frame=map.coordinate_frame
        ).transform_to(new_frame),
        m_hmi_resample.date
    )

    # Rotate bounding coordinates forward to the EIS time
    blc_ar_synop_rot = solar_rotate_coordinate(blc_ar_synop, time=map.date)
    trc_ar_synop_rot = solar_rotate_coordinate(trc_ar_synop, time=map.date)



    # Select pixels that are either above or below the gauss values, these pixels will be used as seed points for PFSS fieldline tracing.
    masked_pix_y, masked_pix_x = np.where((m_hmi_resample.data >=max_gauss) | (m_hmi_resample.data < min_gauss)) # np.where returns the (row, column) indices (masked_pix_y, masked_pix_x) of the selected pixels.

    print(f"Number of masked pixels: {len(masked_pix_x)}")
    print("Filtered Values:")
    print(m_hmi_resample.data[masked_pix_y, masked_pix_x])

    plt.hist(m_hmi_resample.data[masked_pix_y, masked_pix_x].flatten(), bins=50)
    plt.title("Histogram of Magnetic Field Strengths of Masked Pixels")
    plt.xlabel("Magnetic Field Strength (Gauss)")
    plt.ylabel("Number of Pixels")
    plt.grid(True)
    plt.show() # Checks the distribution of field strengths among selected seeds.

    # Convert the masked strong-field pixel positions (masked_pix_x, masked_pix_y) into real-world solar coordinates (longitude, latitude).
    seeds = m_hmi_resample.pixel_to_world(masked_pix_x*u.pix, masked_pix_y*u.pix,).make_3d() #.make_3d() converts the 2D pixel coordinates to 3D spherical coords.

    plt.figure(figsize=(8,6))
    plt.scatter(seeds.lon.to(u.deg), seeds.lat.to(u.deg), s=1, c='r')
    plt.title("Initial Seeds Before FOV Filtering")
    plt.xlabel("Carrington Longitude (deg)")
    plt.ylabel("Carrington Latitude (deg)")

    plt.grid(True)
    plt.show() # Checks that strong-field seeds are planted everywhere there should be magnetic activity.

    print(f"Number of initial seed points: {len(seeds)}")

    ## Selects seeds based on if they reside within our HMI magnetogram FOV (+10%).
    #in_lon = np.logical_and(seeds.lon > blc_ar_synop.lon, seeds.lon < trc_ar_synop.lon)
    #in_lat = np.logical_and(seeds.lat > blc_ar_synop.lat, seeds.lat < trc_ar_synop.lat)

    in_lon = np.logical_and(seeds.lon > blc_ar_synop_rot.lon, seeds.lon < trc_ar_synop_rot.lon)
    in_lat = np.logical_and(seeds.lat > blc_ar_synop_rot.lat, seeds.lat < trc_ar_synop_rot.lat)


    # Filters based on the previous set HMI magnetogram FOV, only keeping the seeds that are within the FOV.
    seeds = seeds[np.where(np.logical_and(in_lon, in_lat))]
    plt.figure(figsize=(8,6))
    plt.scatter(seeds.lon.to(u.deg), seeds.lat.to(u.deg), s=1, c='b')
    plt.title("Seeds After FOV Filtering (EIS area)")
    plt.xlabel("Carrington Longitude (deg)")
    plt.ylabel("Carrington Latitude (deg)")

    ## Add rectangle to show the EIS FOV (in Carrington coordinates)
    #lon_min = blc_ar_synop.lon.deg
    #lon_max = trc_ar_synop.lon.deg
    #lat_min = blc_ar_synop.lat.deg
    #lat_max = trc_ar_synop.lat.deg
    #plt.gca().add_patch(Rectangle(
    #    (lon_min, lat_min),
    #    lon_max - lon_min,
    #    lat_max - lat_min,
    #    edgecolor='orange',
    #    facecolor='none',
    #    lw=2,
    #    label='EIS FOV'
    #))

    plt.legend()
    plt.grid(True)
    plt.show() # Confirm that after masking, seeds correspond only to the EIS field-of-view (plus 10% buffer).
    print(f"Number of seeds after FOV filtering: {len(seeds)}")
    
    nrho = 70 # Number of radial grid points(steps form solar surface to source surface, like resolution).
    rss = 2.5  # Source surface radius (in solar radii, where the fieldlines are traced to, boundary condition for the model).
    pfss_input = pfsspy.Input(m_hmi_resample, nrho, rss) # .Input tell pfsspy what magentogram to use what radial grid to use and where to place the source surface.
    pfss_output = pfsspy.pfss(pfss_input) # .pfss solves the pfss problom and outputs a solution.

    ds = 0.01 # Step size for fieldline tracing (Each tracing step moves the fieldline by 0.01 R☉ before recalculating direction).
    max_steps = int(np.ceil(10 * nrho / ds)) # .ceil rounds to the nearest integer, this computes a maximum number of steps that guarantees a fieldline can reach the top (2.5 R☉) or bottom (1 R☉) without runnin g out of steps.
    tracer = pfsspy.tracing.FortranTracer(step_size=ds, max_steps=max_steps) # Initialize a tracer to follow magnetic fieldlines step-by-step through the solved PFSS field.
    print('processing fieldlines')
    fieldlines = tracer.trace(SkyCoord(seeds), pfss_output) # .trace takes list of seed starting points, takes magentic field solution, tracing the fieldlines starting at each seed point. Still spherical coords.
    # Fieldline reaches the source surface (2.5) = open fieldline. Fieldline reaches the solar surface (1) = closed fieldline. The fieldline hits max_steps and is forcibly stopped.
    empty_coords_count = sum(len(f.coords) == 0 for f in fieldlines)
    print(f"Fieldlines with empty coords: {empty_coords_count} / {len(fieldlines)}")

    #for f in fieldlines:
    #    try:
    #        f.b = pfss_output.get_bvec(f.coords, out_type="cartesian") * u.G # * u.G converts the unitless output to the correct units.
    #    except Exception as e:
    #        f.b = None

    #all_magnitudes = []
    #for f in fieldlines:
    #    if hasattr(f, 'b') and f.b is not None:
    #        mags = np.linalg.norm(f.b.value, axis=1)
    #        mags = mags[np.isfinite(mags)]
    #        all_magnitudes.extend(mags)


    #if all_magnitudes:
    #    all_magnitudes = np.array(all_magnitudes)
    #    print(f"B magnitude range: {all_magnitudes.min():.2f} G – {all_magnitudes.max():.2f} G")
    #    print(f"B mean: {np.mean(all_magnitudes):.2f} G, median: {np.median(all_magnitudes):.2f} G")

    ## DEBUG BLOCK — Check radii
    #first_fline = fieldlines[0]
    #radii = first_fline.coords.radius.to(u.solRad)
    #print(f"Fieldline 0 radius range: min = {radii.min():.2f}, max = {radii.max():.2f}")

    ## DEBUG BLOCK — Try evaluating B-field directly
    #try:
    #    B_test = pfss_output.get_bvec(first_fline.coords) * u.T
    #    print("Test B field shape:", B_test.shape)
    #    print("Test B field sample (Gauss):", B_test[0].to(u.Gauss))
    #except Exception as e:
    #    print("b_eval() failed with error:", e)

    footpoints_lon = [f.coords.lon[0].to(u.deg).value for f in fieldlines if len(f.coords.lon) > 0]
    footpoints_lat = [f.coords.lat[0].to(u.deg).value for f in fieldlines if len(f.coords.lat) > 0]


    plt.figure(figsize=(8,6))
    plt.scatter(footpoints_lon, footpoints_lat, s=1, c='g')
    plt.title("Fieldline Starting Footpoints After Tracing")
    #plt.gca().add_patch(Rectangle(
    #    (lon_min, lat_min),
    #    lon_max - lon_min,
    #    lat_max - lat_min,
    #    edgecolor='orange',
    #    facecolor='none',
    #    lw=2,
    #    label='EIS FOV'
    #))
    #plt.legend()

    lon_min_rot = blc_ar_synop_rot.lon.deg
    lon_max_rot = trc_ar_synop_rot.lon.deg
    lat_min_rot = blc_ar_synop_rot.lat.deg
    lat_max_rot = trc_ar_synop_rot.lat.deg

    plt.gca().add_patch(Rectangle(
        (lon_min_rot, lat_min_rot),
        lon_max_rot - lon_min_rot,
        lat_max_rot - lat_min_rot,
        edgecolor='cyan',
        facecolor='none',
        lw=2,
        linestyle='--',
        label='Rotated EIS FOV'
    ))

    print(f"Δ Carrington Lon after rotation: {(trc_ar_synop_rot.lon - trc_ar_synop.lon).to(u.deg)}")
    print(f"Δ Carrington Lat after rotation: {(trc_ar_synop_rot.lat - trc_ar_synop.lat).to(u.deg)}")


    plt.grid(True)
    plt.show() # Confirms that fieldlines were successfully traced from the seeds.
    print("==========================")
    print("Adding seed metadata to fieldlines...")
    seeds = SkyCoord(seeds) # Ensure seeds are full SkyCoord objects (not just frame instances) to enable transformation and pixel mapping.
    print("Seed frame before transformation:", seeds.frame.name)
    print("Map frame (EIS):", map.coordinate_frame)
    print("Sample seed Carrington lon/lat before transform:", seeds[0].lon.deg, seeds[0].lat.deg)
    #seeds_2d = seeds.transform_to(map.coordinate_frame) # Convert seed coordinates to the 2D helioprojective frame of the EIS map.
    #seeds_2d = seeds.transform_to(map.coordinate_frame.replicate(obstime=map.date)) # Convert seed coordinates to the 2D helioprojective frame of the EIS map.
    #seeds_2d = seeds.transform_to(Helioprojective(obstime=map.date, observer=map.observer_coordinate))
    seeds_2d = seeds.transform_to(map.pixel_to_world(0*u.pix, 0*u.pix).frame)
    print("Sample seed Solar-X/Y after transform:", seeds_2d[0].Tx.to(u.arcsec), seeds_2d[0].Ty.to(u.arcsec))
    # Print transformed world coordinate and resulting pixel coordinate
    test_coord = seeds_2d[0]
    print("Seed 0 world coordinate (Tx, Ty):", test_coord.Tx.to(u.arcsec), test_coord.Ty.to(u.arcsec))
    print("Mapped to pixel:", map.world_to_pixel(test_coord))

    print("===========================")
    x_pix, y_pix = map.world_to_pixel(seeds_2d) # Map each transformed seed coordinate to its corresponding (x, y) pixel location on the EIS image.
    print(f"x_pix range: {x_pix.min()} to {x_pix.max()}")
    print(f"y_pix range: {y_pix.min()} to {y_pix.max()}")
    print(f"Map shape (Y, X): {map.data.shape}")
    x_vals = x_pix.value # Extract raw pixel values from astropy Quantity objects.
    y_vals = y_pix.value 
    valid = np.isfinite(x_vals) & np.isfinite(y_vals) # Identify seeds with valid pixel mappings.
    print(f"Skipped {np.count_nonzero(~valid)} fieldlines due to invalid pixel mapping.")
    valid_fieldlines = np.array(fieldlines)[valid] # Keep only fieldlines whose seeds mapped successfully to image pixels.
    for f, x, y in zip(valid_fieldlines, x_vals[valid], y_vals[valid]):

        # Loop length and lopp starting pixel metadata
        f.start_pix = (int(x), int(y)) # Round to nearest pixel since image indices must be integers, not sub-pixel floats.

        # Original Euclidean distance method, faster, but harder to read. 
        ## f.coords is a list of 3D coordinate points along the fieldline f.
        ## .cartesian gives us acess to the cartesian version of the coordinates, without this we could be dealing with spherical coordinates.
        ## .xyz gives you a 3xN Quantity array, x, y and z.
        ## .to_value() removes the physical units, returning a plain NumPy array of floats
        ## .T transposes the array from (3,N) to (N,3)
        #coords = f.coords.cartesian.xyz.to_value().T 
        ## np.diff computes the difference between the points, subracting each row form the next row.
        ## axis=0 means operate between rows (downward, row-wise)
        ## axis=1 means operate within rows (across columns)
        #diffs = np.diff(coords, axis=0) # Stepwise differences between points along the line.
        ## np.linalg is NumPy's linear algebra library.
        ## .norm computes the vector norm, i.e the length of the vector.
        #f.length = np.sum(np.linalg.norm(diffs, axis=1)) # Arc length of the field line via Euclidean distance.

        coords = f.coords.cartesian # Converts from spherical to x, y, z, FortranTracer is done in spherical coords.
        # Extract the x, y, z coords:
        x = coords.x.value # .value removes the units so it is compatible with numpy, creating a float.
        y = coords.y.value
        z = coords.z.value
        #print("Number of points along fieldline:", len(x))

        points = []
        for i in range(len(x)):
            point = [x[i], y[i], z[i]] # creates a single point x_i, y_i, z_i
            points.append(point) # adds the point to the list of points.
        points_diff = [] # This will store the distance between each point and the next.
        for i in range(len(points)-1):
            dx = points[i+1][0] - points[i][0] # This grabs the x coord of the next point and subtracts the x coord of the current point.
            dy = points[i+1][1] - points[i][1]
            dz = points[i+1][2] - points[i][2]
            point_diff = [dx, dy, dz] # This creates a list of the differences in x, y, z.
            points_diff.append(point_diff) # This adds the point_diff to the list of points_diff.

        total_length = 0
        for i in range(len(points_diff)):
            points_diff_x_squared = points_diff[i][0] ** 2 # This grabs the x coord of the current point_diff, and squarses it.
            points_diff_y_squared = points_diff[i][1] ** 2 
            points_diff_z_squared = points_diff[i][2] ** 2 
            point_mag = np.sqrt(points_diff_x_squared + points_diff_y_squared + points_diff_z_squared)
            total_length += point_mag # This adds the absolute value of the point_diff to the total length of the fieldline.
        f.length = total_length

        ## Mean magnetic field strength metadata
        #if hasattr(f, 'b') and f.b is not None: # Check if the fieldline has magnetic field data.
        #    B_magnitude = np.linalg.norm(f.b.value, axis=1) # f.b.to(u.Gauss) converts units from Tesla to Gauss, .value strips away the units, np.linalg.norm() computes the vector magnitude.
        #    f.mean_B = np.mean(B_magnitude) # Take the average of all |B| values along the fieldline.
        #else:
        #    f.mean_B = np.nan
        
    # we no longer need to make use of a try block since we already check for valid_fieldlines, so all fieldlines should have valid data.
    for f in valid_fieldlines:
        ## pfss_output is the output of psspy.pfss()
        ## get_bvec() tells pfsspy to calcualte the magnetic field vector at the coords of the fieldline f.
        ## out_type="cartesian" means we get the output in cartesian coords, isntead of spherical.
        #bvec = pfss_output.get_bvec(f.coords, out_type="cartesian") * u.G
        #f.b = bvec # does nothing.
        ## np.linalg.norm calculates the length(magnitude) of the vector bvec, which is the magnetic field vector in this case.
        ## axis=1 means operate within rows (across columns).
        ## .value removes the units
        ## .mean gets the average of all the values.
        #f.mean_B = np.mean(np.linalg.norm(bvec.value, axis=1))

        coords = f.coords # f.coords is a list of 3D coord points along the fieldline f.
        if len(coords) == 0: # Skip fieldlines with no coords.
            f.mean_B = np.nan
            #print(f"Skipping fieldline with no coordinates: {f}")
            continue
        #bvec_unitless = pfss_output.get_bvec(coords, out_type="cartesian") # This gets the magnetic field vector at the coords of the fieldline f.
        #bvec = bvec_unitless * u.G # This converts the units from Tesla to Gauss.
        coords.representation_type = "spherical" # Makes sure the coord representation is in spherical form.
        #print("Max radius after filtering:", np.max(coords.radius.to(u.R_sun).value))
        phi = coords.lon.to("rad").value # Extracts the longitude of the coords in radians.
        sin_theta = np.sin(coords.lat).value # Extracts the sine of the latitude of the coords.
        log_r = np.log(coords.radius.to(u.R_sun).value) # Extracts the log of the radius of the coords in solar radii.
        N = len(phi)
        interp_input = np.zeros((N, 3))  # create empty (N, 3) array.
        for i in range(N):
            interp_input[i, 0] = phi[i]         # φ (longitude, in radians)
            interp_input[i, 1] = sin_theta[i]   # sin(θ)
            interp_input[i, 2] = log_r[i]       # log(r)
        bvec_unitless = pfss_output._brgi(interp_input) # Use PFSSPy's internal interpolator _brgi to get the B-vector at each coord point.
        ## DEBUG BLOCK
        #nan_mask = np.isnan(bvec_unitless).any(axis=1)  # True if any component (Bx, By, Bz) is NaN
        #num_nans = np.sum(nan_mask)
        #num_valid = len(nan_mask) - num_nans
        #if num_nans > 0:
        #    print(f"NaNs in B-field vector for fieldline {f}: {num_nans} NaNs out of {len(nan_mask)} points ({num_valid} valid)")
        ## END DEBUG BLOCK
        unit_str = pfss_output.input_map.meta.get("bunit", None) # Attempt to get the unit string from the metadata of the input map.
        bunit = u.Unit(unit_str) if unit_str is not None else u.dimensionless_unscaled # In our case the bunit is unitless.
        bvec = bvec_unitless * bunit
        bvec = bvec * u.G # This converts the units from Tesla to Gauss (works because we know bunit in this case is unitless, but we use anyway to stay consistent with the original function).
        bvec_mag = []
        for i in range(len(bvec)):
            bvec_x = bvec[i][0].value # This grabs the x coord of the magnetic field vector.
            bvec_y = bvec[i][1].value 
            bvec_z = bvec[i][2].value 
            mag = np.sqrt(bvec_x ** 2 + bvec_y ** 2 + bvec_z ** 2) # This calculates the magnitude of the magnetic field vector.
            bvec_mag.append(mag)
        ## DEBUG BLOCK # I was right the last value is Nan, I assume because it's something like 2.50001.
        #if np.any(np.isnan(bvec_mag)):
        #    print("bvec_unitless[-100:]:\n", bvec_unitless[-100:])  # raw interpolated unitless vectors
        ##    print("bvec (with units):\n", bvec)        # vectors with unit applied
        ##    print("bvec_mag:\n", bvec_mag)             # computed magnitudes
        ##    print("bvec_mean (will be NaN):", np.mean(bvec_mag))
        #    break  # Stop after first one to inspect it
        ## END DEBUG BLOCK
        #bvec_mean = np.mean(bvec_mag) # This takes the average of all |B| values along the fieldline. If any value is NaN, the mean will be NaN.
        bvec_mean = np.nanmean(bvec_mag) # This takes the average of all |B| values along the fieldline, ignoring NaN values.
        f.mean_B = bvec_mean # This adds the mean magnetic field strength to the fieldline object.

        #f.custom["expansion_factor"] = f.expansion_factor # This adds the expansion factor to the fieldline object, if it exists.
        if not hasattr(f, "custom"):
            f.custom = {}

        f.custom["expansion_factor"] = f.expansion_factor if f.expansion_factor is not None else np.nan # This adds the expansion factor to the fieldline object, if it exists. If the expansion factor is None, set it to NaN.
        #print(f"Expansion factor: {f.expansion_factor}")
        #print(f"Custom stored expansion factor: {f.custom['expansion_factor']}")

        ## Expansion factor : a measure of how much the magnetic field expands from the solar surface to the source surface.
        ## First thing we need to do is find out r0 and B0, which are the values at the footpoint of the fieldline (1 R☉).
        #r_values = coords.radius.to(u.R_sun).value # Extract the radius for each point along the fieldline.
        #smallest_diff = float('inf')  # Start with a huge difference.
        #r_closest_to_1 = None
        #B_at_r_closest_to_1 = None
        #r_at_r_closest_to_1 = None
        #for i in range(len(r_values)):
        #    r_curr = r_values[i]
        #    B_curr = bvec_mag[i]
        #    diff_from_1 = abs(r_curr - 1.0) # This calculates the difference between the current radius and 1.0.
        #    if diff_from_1 < smallest_diff: # This checks if the current difference is smaller than the smallest difference.
        #        smallest_diff = diff_from_1
        #        r_closest_to_1 = r_curr
        #        B_at_r_closest_to_1 = B_curr
        #        r_at_r_closest_to_1 = r_curr
        #B0 = B_at_r_closest_to_1 
        #r0 = r_at_r_closest_to_1

        ## Next we need to find out r1 and B1, which are the values at the source surface (2.5 R☉).
        #r_target = 2.5
        #smallest_diff = float('inf')
        #r_closest_to_2_5 = None
        #B_at_r_closest_to_2_5 = None
        #for i in range(len(r_values)):
        #    r_curr = r_values[i]
        #    B_curr = bvec_mag[i]
        #    diff_from_2_5 = abs(r_curr - r_target)
        #    if diff_from_2_5 < smallest_diff: # This checks if the current difference is smaller than the smallest difference.
        #        smallest_diff = diff_from_2_5
        #        r_closest_to_2_5 = r_curr
        #        B_at_r_closest_to_2_5 = B_curr
        #B1 = B_at_r_closest_to_2_5
        #r1 = r_closest_to_2_5

        #if B1 > 0 and r0 > 0 and r1 > 0:
        #    f_expansion = (B0 / B1) * (r1 / r0)**2
        #else:
        #    f_expansion = np.nan
        #if not hasattr(f, "custom"):
        #    f.custom = {}
        #f.custom["expansion_factor"] = f_expansion # Not a predefined attribute in pfsspy, but we can add it as a custom attribute. 

    
    num_with_length = sum(np.isfinite(f.length) for f in valid_fieldlines)
    print(f"Fieldlines with valid length metadata: {num_with_length} / {len(valid_fieldlines)}")

    num_with_mean_B = sum(np.isfinite(f.mean_B) for f in valid_fieldlines)
    print(f"Fieldlines with valid mean_B metadata: {num_with_mean_B} / {len(valid_fieldlines)}")

    num_with_expansion_factor = sum(np.isfinite(f.expansion_factor) for f in valid_fieldlines)
    print(f"Fieldlines with valid expansion factor metadata: {num_with_expansion_factor} / {len(valid_fieldlines)}")

    # Plot to verify that start_pix aligns with EIS data.
    plt.figure(figsize=(6, 10))
    plt.imshow(map.data, origin='lower', cmap='gray', aspect='auto')
    x_pix = [f.start_pix[0] for f in valid_fieldlines]
    y_pix = [f.start_pix[1] for f in valid_fieldlines]
    plt.scatter(x_pix, y_pix, s=2, color='cyan', label='start_pix')
    plt.title("EIS Raster with Fieldline Start Pixels (start_pix)")
    plt.xlabel("X Pixel")
    plt.ylabel("Y Pixel")
    plt.legend()
    plt.grid(True)
    plt.show() # Shows whether the fieldlines actually map back to EIS data in the correct orientation.
    # Optional sanity check
    x_check, y_check = map.world_to_pixel(map.pixel_to_world(0*u.pix, 0*u.pix))
    print(f"Pixel (0,0) round-trip lands at: ({x_check}, {y_check})")

    plt.figure()
    plt.hist([f.mean_B for f in valid_fieldlines if np.isfinite(f.mean_B)], bins=50)
    plt.title("Distribution of Mean Magnetic Field Strengths")
    plt.xlabel("Mean |B| (Gauss)")
    plt.ylabel("Number of Fieldlines")
    plt.grid(True)
    plt.show()

    open_lines = [f for f in fieldlines if f.is_open] # For each fieldline f in fieldlines, check if f.is_open == True, if yes add to open_lines
    closed_lines = [f for f in fieldlines if not f.is_open] # For each fieldline f in fieldlines, check if f.is_open == False, if yes add to closed_lines

    open_fieldlines = OpenFieldLines(open_lines) if open_lines else OpenFieldLines([]) # If open_lines is not empty, create OpenFieldLines object, else create an empty one.
    closed_fieldlines = ClosedFieldLines(closed_lines) if closed_lines else ClosedFieldLines([]) # If closed_lines is not empty, create ClosedFieldLines object, else create an empty one.


    print(f"Total field lines: {len(fieldlines)}")
    print(f"Open field lines: {len(open_fieldlines)}")
    print(f"Closed field lines: {len(closed_fieldlines)}")
    #print("Frame of seeds_2d:", seeds_2d.frame)
    #print("EIS map frame:", map.coordinate_frame)
    return open_fieldlines, closed_fieldlines



def get_pfss(IRIS_map_dir):
    IRIS_dir = '/'.join(IRIS_map_dir.split('/')[0:-1])
    IRIS_map = Map(IRIS_map_dir)
    hp_lon = np.linspace(IRIS_map.bottom_left_coord.Tx/u.arcsec, IRIS_map.top_right_coord.Tx/u.arcsec, round(len(IRIS_map.data[0,:]))) * u.arcsec
    hp_lat = np.linspace(IRIS_map.bottom_left_coord.Ty/u.arcsec, IRIS_map.top_right_coord.Ty/u.arcsec, round(len(IRIS_map.data[0,::]))) * u.arcsec
    
    
    # Make a 2D grid from these 1D points
    lon, lat = np.meshgrid(hp_lon, hp_lat)
    seeds = SkyCoord(lon.ravel(), lat.ravel(),
                     frame=IRIS_map.coordinate_frame).make_3d()

    m_hmi = hmi_daily_download(IRIS_map.date.value)
    
    # Define functions to change the observer time and frame
    change_obstime = lambda x,y: SkyCoord(x.replicate(observer=x.observer.replicate(obstime=y), obstime=y))
    change_obstime_frame = lambda x,y: x.replicate_without_data(observer=x.observer.replicate(obstime=y), obstime=y)
    
    # Change the observer time and frame of the synoptic data
    new_frame = change_obstime_frame(m_hmi.coordinate_frame, IRIS_map.date)
    
    # Resample the HMI data to a specific resolution
    m_hmi_resample = m_hmi.resample((2160, 1080)*u.pix)

    new_frame = change_obstime_frame(m_hmi_resample.coordinate_frame, IRIS_map.date)
    blc_ar_synop = change_obstime(IRIS_map.bottom_left_coord.transform_to(new_frame),
                                  m_hmi_resample.date)
    trc_ar_synop = change_obstime(IRIS_map.top_right_coord.transform_to(new_frame),
                                  m_hmi_resample.date)

    # masked_pix_y, masked_pix_x = np.where((m_hmi_resample.data > 10) | (m_hmi_resample.data < 10))
    # seeds = m_hmi_resample.pixel_to_world(masked_pix_x*u.pix, masked_pix_y*u.pix,).make_3d()
    masked_pix_y, masked_pix_x = np.where((m_hmi_resample.data > 7) | (m_hmi_resample.data < 7))
    seeds = m_hmi_resample.pixel_to_world(masked_pix_x*u.pix, masked_pix_y*u.pix,).make_3d()
    in_lon = np.logical_and(seeds.lon > blc_ar_synop.lon, seeds.lon < trc_ar_synop.lon)
    in_lat = np.logical_and(seeds.lat > blc_ar_synop.lat, seeds.lat < trc_ar_synop.lat)
    seeds = seeds[np.where(np.logical_and(in_lon, in_lat))]

        
    # Find the masked pixels based on a condition
    masked_pix_y, masked_pix_x = np.where((m_hmi_resample.data >=10) | (m_hmi_resample.data < 10))
    # masked_pix_y, masked_pix_x = np.where((m_hmi_resample.data <=-7))
    # masked_pix_y, masked_pix_x = np.where((m_hmi_resample.data > 7))
    
    # Convert the masked pixel coordinates to world coordinates
    seeds = m_hmi_resample.pixel_to_world(masked_pix_x*u.pix, masked_pix_y*u.pix,).make_3d()
    
    # Filter the seeds based on longitude and latitude ranges
    in_lon = np.logical_and(seeds.lon > blc_ar_synop.lon, seeds.lon < trc_ar_synop.lon)
    in_lat = np.logical_and(seeds.lat > blc_ar_synop.lat, seeds.lat < trc_ar_synop.lat)
    seeds = seeds[np.where(np.logical_and(in_lon, in_lat))]
    
    print("Processing PFSS...")
    nrho = 70
    rss = 2.5
    pfss_input = pfsspy.Input(m_hmi_resample, nrho, rss)
    pfss_output = pfsspy.pfss(pfss_input)
    
    print("PFSS Done, Tracing Fieldlines...")
    ds = 0.01
    max_steps = int(np.ceil(10 * nrho / ds))
    tracer = pfsspy.tracing.FortranTracer(step_size=ds, max_steps=max_steps)
    # tracer = pfsspy.tracing.FortranTracer(max_steps=max_steps)
    fieldlines = tracer.trace(SkyCoord(seeds), pfss_output,)

    all_lines = [change_obstime(f.coords, IRIS_map.date) for f in fieldlines if f.coords.shape[0]>0]
    time_iden = IRIS_map_dir.split('/')[-1].replace('_v_turb_map.fits','')
    with open(f'{IRIS_dir}/{time_iden}_pfss_fieldlines.pickle', 'wb') as f:
        pickle.dump(all_lines, f)
        
if __name__ == '__main__':
    import os 
    IRIS_map_dirs = sorted(glob('/Users/andysh.to/Script/Data/IRIS_output/201904*/*_v_turb_map.fits'))
    for iris_map_dir in IRIS_map_dirs:
        name = iris_map_dir.replace('_v_turb_map.fits','_pfss_fieldlines.pickle')
        # if os.path.exists(name):
        #     print(f'{name.split("/")[-1]} exists... Skipping...')
        # else:
        try:
            get_pfss(iris_map_dir)
        except Exception as e:
            print(f'Error: {e}')
            print(f'{name.split("/")[-1]} did not process...')

            continue

    
