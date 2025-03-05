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

# Additional imports that might be required based on the code context
from sunpy.net import attrs as a
from astropy.wcs import WCS
from astropy.visualization import ImageNormalize, SqrtStretch
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable

warnings.filterwarnings('ignore')

def get_closest_aia(date_time_obj, wavelength = 193):
    # Add this line near the top of the file, after the imports
    DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

    """
    Get the closest AIA image to a given datetime for a specified wavelength.

    This function searches for and downloads the AIA image closest to the input datetime,
    within a 1-minute window (±30 seconds) for the specified wavelength.

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
    
    aia_res = Fido.search(a.Time(aia_start_str, aia_end_str), a.Instrument.aia, a.Wavelength(wavelength*u.angstrom))[0][0]
    aia_downloads = Fido.fetch(aia_res, path="./{instrument}/{file}")
    aia_map_plot = sunpy.map.Map(aia_downloads[0])
    return aia_map_plot

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

    # Create date_time_obj for get_closest_aia
    date_time_obj = map.date.datetime
    aia_map = get_closest_aia(date_time_obj)
    
    hp_lon = np.linspace(map.bottom_left_coord.Tx/u.arcsec, map.top_right_coord.Tx/u.arcsec, round(len(map.data[0,:]))) * u.arcsec
    hp_lat = np.linspace(map.bottom_left_coord.Ty/u.arcsec, map.top_right_coord.Ty/u.arcsec, round(len(map.data[0,::]))) * u.arcsec
    
    
    # Make a 2D grid from these 1D points
    lon, lat = np.meshgrid(hp_lon, hp_lat)

    seeds = SkyCoord(lon.ravel(), lat.ravel(),
                     frame=map.coordinate_frame).make_3d()

    m_hmi = hmi_daily_download(map.date.value)
    
    # Define functions to change the observer time and frame
    change_obstime = lambda x,y: SkyCoord(x.replicate(observer=x.observer.replicate(obstime=y), obstime=y))
    change_obstime_frame = lambda x,y: x.replicate_without_data(observer=x.observer.replicate(obstime=y), obstime=y)
    
    # Change the observer time and frame of the synoptic data
    new_frame = change_obstime_frame(m_hmi.coordinate_frame, map.date)
    
    # Resample the HMI data to a specific resolution
    m_hmi_resample = m_hmi.resample(dimension*u.pix)

    new_frame = change_obstime_frame(m_hmi_resample.coordinate_frame, map.date)
    # Expand the coordinates by 10% in each direction
    blc_ar_synop = change_obstime(
        SkyCoord(
            map.bottom_left_coord.Tx - 0.1 * (map.top_right_coord.Tx - map.bottom_left_coord.Tx),
            map.bottom_left_coord.Ty - 0.1 * (map.top_right_coord.Ty - map.bottom_left_coord.Ty),
            frame=map.coordinate_frame
        ).transform_to(new_frame),
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
    # masked_pix_y, masked_pix_x = np.where((m_hmi_resample.data > 10) | (m_hmi_resample.data < 10))
    # seeds = m_hmi_resample.pixel_to_world(masked_pix_x*u.pix, masked_pix_y*u.pix,).make_3d()

    # Find the masked pixels based on a condition
    masked_pix_y, masked_pix_x = np.where((m_hmi_resample.data >=max_gauss) | (m_hmi_resample.data < min_gauss))

    seeds = m_hmi_resample.pixel_to_world(masked_pix_x*u.pix, masked_pix_y*u.pix,).make_3d()
    in_lon = np.logical_and(seeds.lon > blc_ar_synop.lon, seeds.lon < trc_ar_synop.lon)
    in_lat = np.logical_and(seeds.lat > blc_ar_synop.lat, seeds.lat < trc_ar_synop.lat)
    seeds = seeds[np.where(np.logical_and(in_lon, in_lat))]

        
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
    print('processing max_steps')    
    max_steps = int(np.ceil(10 * nrho / ds))
    print('processing tracer')
    tracer = pfsspy.tracing.FortranTracer(step_size=ds, max_steps=max_steps)
    # tracer = pfsspy.tracing.FortranTracer(max_steps=max_steps)
    print('processing fieldlines')
    fieldlines = tracer.trace(SkyCoord(seeds), pfss_output,)
    return fieldlines
    
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

    