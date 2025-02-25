from aiapiper.tools import PipeFix # type: ignore
import astropy.units as u # type: ignore

downloader = PipeFix()
downloader.fetch(
	start_date="2014-02-01T10:00:00.000",
	end_date="2014-02-07T23:59:00.000",
	wavelength=193,
	cadence=1*u.hour,
)

#============================================================================================

#eyeball an active region, using the top_right, bottom_left system, of solarmonitor.org, and just make a png


import glob, os, matplotlib.pyplot as plt, astropy.units as u
from sunpy.map import Map
from astropy.coordinates import SkyCoord
from astropy.visualization import ImageNormalize, SqrtStretch
from sunpy.physics.differential_rotation import solar_rotate_coordinate
import imageio.v2 as imageio


base_file = "sdo_downloads/aia.lev1.193A_2014-01-29T20-00-08.22Z.image_lev1.fits"
# Load the base map from the FITS file
base_map = Map(base_file)
base_time = base_map.date  # Extract the observation time of the base map
base_center = SkyCoord((-600 + -1050) / 2 * u.arcsec, (0 - 350) / 2 * u.arcsec, frame=base_map.coordinate_frame)
half_width = 250 * u.arcsec
half_height = 150 * u.arcsec
base_bottom_left = SkyCoord(base_center.Tx - half_width, base_center.Ty - half_height, frame=base_map.coordinate_frame)
base_top_right = SkyCoord(base_center.Tx + half_width, base_center.Ty + half_height, frame=base_map.coordinate_frame)
base_fixed_crop = base_map.submap(base_bottom_left, top_right=base_top_right)
fig = plt.figure()
ax = fig.add_subplot(projection=base_fixed_crop)
base_fixed_crop.plot(axes=ax, norm=ImageNormalize(stretch=SqrtStretch()))
base_fixed_crop.draw_limb(axes=ax)
base_fixed_crop.draw_grid(axes=ax)

#test for preview
ax.set_title(f"Base Fixed Crop at {base_time.iso}", pad=10)
plt.show()
plt.close(fig)
# Get a list of all FITS files in the directory sorted by time
all_files = sorted(glob.glob("sdo_downloads/aia.lev1.193A_*.fits"))
selected_files = []
for f in all_files:
    try:
        m = Map(f)
        if m.date >= base_time:
            selected_files.append(f)
    except Exception as e:
        print(f"Error reading {f}: {e}")


print(f"Found {len(selected_files)} files for tracking.")

output_dir="frames_center_fixed"
os.makedirs(output_dir,exist_ok=True)
frame_filenames=[]
# Loop through selected FITS files and process each one
for i,f in enumerate(selected_files):
    if i == 9:
        print("Skipping frame 9 due to artifact.")
        continue
    try: 
        current_map=Map(f)
    except Exception as e:
        print(f"Could not load file {f}: {e}")
        continue
    current_time=current_map.date
    print(f"Processing {f} (Time: {current_time.iso})")
    # Compute the rotated center of the region using solar differential rotation
    rotated_center=solar_rotate_coordinate(base_center,time=current_time)
    # Compute the new bottom-left and top-right coordinates for the cropped region
    new_bottom_left=SkyCoord(rotated_center.Tx-half_width,rotated_center.Ty-half_height,frame=current_map.coordinate_frame)
    new_top_right=SkyCoord(rotated_center.Tx+half_width,rotated_center.Ty+half_height,frame=current_map.coordinate_frame)
    try:
        current_crop=current_map.submap(new_bottom_left,top_right=new_top_right)
    except Exception as e:
        print(f"Error creating submap for {f}: {e}")
        continue
    fig=plt.figure()
    ax=fig.add_subplot(projection=current_crop)
    current_crop.plot(axes=ax,norm=ImageNormalize(stretch=SqrtStretch()))
    #current_crop.plot(axes=ax)
    #current_crop.draw_limb(axes=ax)
    #current_crop.draw_grid(axes=ax)
    ax.set_title(f"Active Region at {current_time.iso}",pad=10)
    plt.tight_layout()
    #ax.coords[1].ticklabels.set_visible(False)
    #ax.coords[1].ticks.set_visible(False)
    #ax.coords[2].ticklabels.set_visible(False)
    #ax.coords[2].ticks.set_visible(False)
    frame_filename=os.path.join(output_dir,f"frame_{i:03d}.png")
    plt.savefig(frame_filename,bbox_inches='tight')
    plt.close(fig)
    frame_filenames.append(frame_filename)


print(f"Saved {len(frame_filenames)} frames.")
gif_filename="active_region_sequence_full.gif"
with imageio.get_writer(gif_filename,mode="I",fps=1) as writer:
    for filename in sorted(frame_filenames):
        image=imageio.imread(filename)
        writer.append_data(image)


print(f"GIF saved as {gif_filename}")
