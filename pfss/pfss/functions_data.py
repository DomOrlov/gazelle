# Python is very modular, so you import packages before using them.

from sunpy.net import Fido, attrs as a
# from sunpy.image import coalignment
import astropy
import astropy.units as u
from pathlib import Path
from datetime import datetime, timedelta
import sunpy
# import eispac
from astropy.coordinates import SkyCoord
from reproject import reproject_interp
import numpy as np
import pfsspy
from astropy.visualization import ImageNormalize, quantity_support
# from asheis import asheis
# from functions_pickle import *

def PrepHMIdaily(hmidailyfilepath, numpix_x=3600, numpix_y= 1440):
    # Prep HMI daily synoptic map by transforming it into CEA coordinates
    from astropy.io import fits
    import numpy as np
    import sunpy.map
    import astropy.time
    import matplotlib.pyplot as plt
    import astropy.units as u
    from sunpy.coordinates import sun


    #Can load only hmi.mrdailysynframe_polfil using astropy.io.fits
    hdus = fits.open(hmidailyfilepath)[0]
    # hdus.verify('fix')
    #Metadata fixing
    #Change sin(deg) to degree
    hdus.header["CUNIT1"] = 'deg'
    hdus.header["CUNIT2"] = 'deg'
    hdus.header["CDELT1"] = np.abs(hdus.header['cdelt1'])
    # print(hdus.header["CDELT2"])
    # hdus.header["CRVAL1"] = sun.L0(time=hdus.header["DATE-OBS"],aberration_correction=True).value + 120
    # hmi_map.meta['CRVAL1'] = 120 + sun.L0(time=hmi_map.meta['T_OBS']).value

    hdus.header["CDELT2"] = 180 / np.pi * hdus.header["CDELT2"]
    hdus.header["CRLN_OBS"] = hdus.header["CRLN_OBS"] +120
    hdus.header["RSUN_REF"] = 6.957e8
    hdus.header["CADENCE"] = 360
    # original rsun =. 695500000.0
    del hdus.header["CRDER1"]
    del hdus.header["CRDER2"]
    del hdus.header["CSYSER1"]
    del hdus.header["CSYSER2"]

    hmi_syn_map = sunpy.map.Map(np.nan_to_num(hdus.data), hdus.header)
    time = hmi_syn_map.meta['T_OBS'][0:10].replace('.','-')+'T'+hmi_syn_map.meta['T_OBS'][11:-4]
    # time = '2011-04-13T20:42:28.087'
    t_hmi_syn = astropy.time.Time(time, scale='tai')
    # print(t_hmi_syn)
    hdus.header["DATE-OBS"] = t_hmi_syn.utc.value
    # print(hdus.header["CRVAL1"])
    # print(sun.L0(time=t_hmi_syn.utc.value).value)
    # print(hdus.header["CRVAL1"])
    # hdus.header["CRVAL1"] = sun.L0(time=t_hmi_syn.utc.value).value + 120
    hdus.header["CRVAL1"] = sun.L0(time=t_hmi_syn.utc.value).value + 120 # some dark magic hack to roll PFSS to the correct position - 100 is arbitrary (120 originally)
    hdus.header["CRVAL2"] = 0
    # print(hdus.header["CRVAL1"])
    
    br = hdus.data - np.nanmean(hdus.data)
    br = np.nan_to_num(br, nan=-np.nanmean(hdus.data))

    br = np.roll(br, np.int32((hdus.header['CRVAL1'] + 180)/np.abs(hdus.header['cdelt1'])), axis=1)
    # print(np.int32((hdus.header['CRVAL1'] + 180)/np.abs(hdus.header['cdelt1'])))
    hdus.header['CRVAL1'] = 759060.0
    hmi_syn_map = sunpy.map.Map(br, hdus.header)
    # print(hdus.header["CRVAL1"])
    # hmi_syn_map.meta["CRVAL1"] = hdus.header['CRLN_OBS']
    hmi_syn_map.plot_settings['norm'] =plt.Normalize(-200,200)


#    #Resample to reduced computational time
    
    # print(hmi_syn_map.dimensions)
    hmi_syn_map.meta['T_OBS'] = hmi_syn_map.meta['date-obs']
    # print(f"CRVAL1: {hmi_syn_map.meta['CRVAL1']}")
    # print(f"NEW CRVAL1: {hmi_syn_map.meta['CRVAL1']}")
    print(f"HMI Daily Synoptic Map captured at: {hmi_syn_map.meta['T_OBS']}")
    # hmi_syn_map = hmi_syn_map.resample([numpix_x, numpix_y] * u.pix)## [longitude, latitude]
    
#     print(hmi_syn_map.date)   
#     print('New shape: ', hmi_syn_map.data.shape)

    return hmi_syn_map

def eis_aia_align(eis, aia):
    n_x = (aia.scale.axis1 * aia.dimensions.x) / eis.scale.axis1
    n_y = (aia.scale.axis2 * aia.dimensions.y) / eis.scale.axis2
    aia_r = aia.resample(u.Quantity([n_x, n_y]))
    yshift, xshift = coalignment.calculate_shift(aia_r.data, eis.data)
    reference_coord = aia_r.pixel_to_world(xshift, yshift)
    Txshift = reference_coord.Tx - eis.bottom_left_coord.Tx
    Tyshift = reference_coord.Ty - eis.bottom_left_coord.Ty
    print(f'x shift = {Txshift}')
    print(f'y shift = {Tyshift}')
    eis_fixed = eis.shift(Txshift, Tyshift)
    return eis_fixed

def aia_correction(m_aia):
    from aiapy.calibrate import correct_degradation, update_pointing
    m_aia = correct_degradation(update_pointing(m_aia))
    m_aia = m_aia / m_aia.exposure_time
    return m_aia

def hmi_daily_download(date_obs):
    # Return a prepped HMI daily synoptic map using time provided.
    
    from sunpy.net import Fido,attrs
    ar_date = astropy.time.Time(date_obs)
    jsoc_email = 'toshuho@gmail.com'
    print('Finding Data...')
    q = Fido.search(
        attrs.Time(ar_date, ar_date+1*u.day),
        attrs.jsoc.Series('hmi.mrdailysynframe_720s'),
        # attrs.jsoc.Series('hmi.Mrdailysynframe_720s'),
        attrs.jsoc.Notify(jsoc_email),
    )
    print('Downloading Data to ./data/...')
    print(q)
    f = Fido.fetch(q, path='./data/')
    print('Transforming Data to the Correct Frame...')
    m_hmi = PrepHMIdaily(f[0])
    # m_hmi.peek()
    return m_hmi

# def aia_single_download(date_obs, wavelength):
#     from sunpy.net import Fido,attrs
#     ar_date = astropy.time.Time(date_obs)
#     print('Finding Data...')
#     Fido.search(ar_date, ar_date+1*u.minutes, a.Instrument.aia,
#             a.Wavelength(int(wavelength)*u.angstrom))  

#     print('Downloading Data...')
#     f = Fido.fetch(q[0], path='./data/')
#     print('Transforming Data to the Correct Frame...')
#     m_hmi = PrepHMIdaily(f[0])
#     m_hmi.peek()
#     return m_hmi

def aia_download_from_date(date_obs, wavelength=193):
    # Return a prepped AIA map using time provided.
    from sunpy.net import Fido,attrs
    m_hmi_time = astropy.time.Time(date_obs)
    aia_or_secchi = ((attrs.Instrument.aia)
                 & attrs.Wavelength(wavelength*u.angstrom)
                 & attrs.Sample(5*u.minute))
    print('Finding AIA data...')
    q = Fido.search(attrs.Time(m_hmi_time-2*u.minute, m_hmi_time+2*u.minute),
                aia_or_secchi)
    print('Downloading AIA data to ./data/...')
    files = Fido.fetch(q, path='./data/')
    m_aia = sunpy.map.Map(sorted(files))
    print('Prepping AIA data...')
    m_aia = aia_correction(m_aia)
    return m_aia

def data_download(eis_name): # filename with .fits.gz
    time = datetime.strptime(eis_name[7:].replace('.fits.gz',''),'%Y%m%d_%H%M%f')
    eis_name = eis_name.replace('.fits.gz','').replace('_l0','')+'.data.h5'
    path = eis_name[4:12]
    Path(path).mkdir(parents=True, exist_ok=True)
    eis_download = eispac.download.download_hdf5_data(eis_name, local_top=path)
    time_range = a.Time(time,end=time+timedelta(minutes=30))
    # eis_query = time_range & a.Instrument('EIS')& a.Source('Hinode') & a.Provider('NRL') & a.Level('1')
    aia_query = time_range & a.Wavelength(193*u.angstrom) & a.Physobs.intensity & a.Instrument.aia
    hmi_query = time_range & a.Instrument.hmi & a.Physobs.los_magnetic_field
    gong_query = time_range & a.Instrument.gong
    aia_result, hmi_result, gong_result = Fido.search(aia_query | hmi_query | gong_query) # Find data
    # files = Fido.fetch(result, path=f'{path}/')
    # result_list = []
    # for i in range(len(result)):
    #     if len(result[i]) != 0:
    #         result_list.append(result[i][0])
    files = Fido.fetch(aia_result[0], hmi_result[0], path=path)
    files.append(f'{path}/{eis_name}')
    aia_map, aia_submap, hmi_map, eis_map = ebtel_data_prep(path, files)
    return aia_map, aia_submap, hmi_map, eis_map

def aia_submap(eis, aia):
    top_right = SkyCoord(eis.top_right_coord.Tx, eis.top_right_coord.Ty, frame=aia.coordinate_frame)
    bottom_left = SkyCoord(eis.bottom_left_coord.Tx, eis.bottom_left_coord.Ty, frame=aia.coordinate_frame)
    aia_submap = aia.submap(bottom_left, top_right=top_right)
    return aia_submap

def ebtel_data_prep(path, files):
    eis_file = asheis(files[-1])
    eis_map = eis_file.get_intensity('fe_12_195')
    aia_map, hmi_map = sunpy.map.Map(files[1], files[0])
    eis_fixed = eis_aia_align(eis_map, aia_map)
    aia_sub_map = aia_submap(eis_fixed, aia_map)
    save(path+'/aia_map', aia_map)
    save(path+'/eis_fixed', eis_fixed)
    save(path+'/hmi_map', hmi_map)
    save(path+'/aia_sub_map', aia_submap)
    return aia_map, aia_submap, hmi_map, eis_map

def hmi_to_cea(m_hmi):
    shape_out=[1080, 2160]
    frame_out = SkyCoord(0, 0, unit=u.deg, rsun=m_hmi.coordinate_frame.rsun, frame="heliographic_stonyhurst", obstime=m_hmi.date)
    header = sunpy.map.make_fitswcs_header(
        shape_out,
        frame_out,
        scale=[180 / shape_out[0], 360 / shape_out[1]] * u.deg / u.pix,
        projection_code="CAR",
    )
    out_wcs = astropy.wcs.WCS(header)
    array, _ = reproject_interp(m_hmi, out_wcs, shape_out=shape_out)
    array = np.where(np.isnan(array), 0, array)
    m_hmi_cea = pfsspy.utils.car_to_cea(sunpy.map.Map(array, header))
    m_hmi_cea.meta['TELESCOP'] = m_hmi.meta['TELESCOP']
    m_hmi_cea.meta['CONTENT'] = 'Carrington Synoptic Chart Of Br Field'
    m_hmi_cea.meta['T_OBS'] = m_hmi_cea.meta.pop('DATE-OBS')  # This is because of a bug where the date accidentally returns None if it is in the date-obs key
    m_hmi_cea = sunpy.map.Map(
        m_hmi_cea.data,
        m_hmi_cea.meta,
        plot_settings={'norm': ImageNormalize(vmin=-1.5e3,vmax=1.5e3), 'cmap': 'hmimag'}
    )
    return m_hmi_cea

