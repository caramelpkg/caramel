import numpy as np
import os
from pathlib import Path
import datetime
import glob
from joblib import Parallel, delayed
from netCDF4 import Dataset
from config import satellite, ctm_model
from interpolator import interpolator
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)


def _read_nc(filename, var):
    # reading nc files without a group
    nc_f = filename
    nc_fid = Dataset(nc_f, 'r')
    out = np.array(nc_fid.variables[var])
    nc_fid.close()
    return np.squeeze(out)


def _read_group_nc(filename, num_groups, group, var):
    # reading nc files with a group structure
    nc_f = filename
    nc_fid = Dataset(nc_f, 'r')
    if num_groups == 1:
        out = np.array(nc_fid.groups[group].variables[var])
    elif num_groups == 2:
        out = np.array(nc_fid.groups[group[0]].groups[group[1]].variables[var])
    elif num_groups == 3:
        out = np.array(
            nc_fid.groups[group[0]].groups[group[1]].groups[group[2]].variables[var])
    nc_fid.close()
    return np.squeeze(out)


def tropomi_reader_hcho(fname: str, interpolation_flag=True) -> satellite:
    '''
       TROPOMI HCHO L2 reader
       Inputs:
             fname [str]: the name path of the L2 file
             interpolation_flag [bool]: if the L2>L3 should be done
       Output:
             tropomi_hcho [satellite]: a dataclass format (see config.py)
    '''
    # hcho reader
    # read time

    time = _read_group_nc(fname, 1, 'PRODUCT', 'time') +\
        np.nanmean(np.array(_read_group_nc(
            fname, 1, 'PRODUCT', 'delta_time')), axis=1)/1000.0
    time = np.nanmean(time, axis=0)
    time = np.squeeze(time)
    tropomi_hcho = satellite
    tropomi_hcho.time = datetime.datetime(
        2010, 1, 1) + datetime.timedelta(seconds=int(time))
    #print(datetime.datetime.strptime(str(tropomi_hcho.time),"%Y-%m-%d %H:%M:%S"))
    # read lat/lon at corners
    tropomi_hcho.latitude_corner = _read_group_nc(fname, 3, ['PRODUCT', 'SUPPORT_DATA', 'GEOLOCATIONS'],
                                                  'latitude_bounds').astype('float32')
    tropomi_hcho.longitude_corner = _read_group_nc(fname, 3, ['PRODUCT', 'SUPPORT_DATA',
                                                              'GEOLOCATIONS'], 'longitude_bounds').astype('float32')
    # read total amf
    amf_total = _read_group_nc(fname, 3, ['PRODUCT', 'SUPPORT_DATA', 'DETAILED_RESULTS'],
                               'formaldehyde_tropospheric_air_mass_factor')
    # read trop no2
    vcd = _read_group_nc(fname, 1, 'PRODUCT',
                         'formaldehyde_tropospheric_vertical_column').astype('float32')
    scd = _read_group_nc(fname, 1, 'PRODUCT', 'formaldehyde_tropospheric_vertical_column').astype('float32') *\
        amf_total
    tropomi_hcho.vcd = vcd*6.02214*1e19*1e-15
    tropomi_hcho.scd = scd*6.02214*1e19*1e-15
    # read quality flag
    tropomi_hcho.qa = _read_group_nc(
        fname, 1, 'PRODUCT', 'qa_value').astype('float32')
    # read pressures for SWs
    tm5_a = _read_group_nc(
        fname, 3, ['PRODUCT', 'SUPPORT_DATA', 'INPUT_DATA'], 'tm5_constant_a')
    tm5_b = _read_group_nc(
        fname, 3, ['PRODUCT', 'SUPPORT_DATA', 'INPUT_DATA'], 'tm5_constant_b')
    ps = _read_group_nc(fname, 3, [
                        'PRODUCT', 'SUPPORT_DATA', 'INPUT_DATA'], 'surface_pressure').astype('float32')
    p_mid = np.zeros(
        (34, np.shape(vcd)[0], np.shape(vcd)[1])).astype('float32')
    SWs = np.zeros((34, np.shape(vcd)[0], np.shape(vcd)[1])).astype('float32')
    AKs = _read_group_nc(fname, 3, [
                         'PRODUCT', 'SUPPORT_DATA', 'DETAILED_RESULTS'], 'averaging_kernel').astype('float32')
    for z in range(0, 34):
        p_mid[z, :, :] = (tm5_a[z]+tm5_b[z]*ps[:, :])/100.0
        SWs[z, :, :] = AKs[:, :, z]*amf_total
    tropomi_hcho.pressure_mid = p_mid
    tropomi_hcho.averaging_kernels = SWs
    # read the precision
    tropomi_hcho.uncertainty = _read_group_nc(fname, 1, 'PRODUCT',
                                              'formaldehyde_tropospheric_vertical_column_precision').astype('float32')
    # interpolation
    if interpolation_flag == True:
        grid_size = 1.0  # degree
        interpolator(1, grid_size, tropomi_hcho, geos_model)
    # return
    return tropomi_hcho


def tropomi_reader_no2(fname: str, interpolation_flag=True) -> satellite:
    '''
       TROPOMI NO2 L2 reader
       Inputs:
             fname [str]: the name path of the L2 file
             interpolation_flag [bool]: if the L2>L3 should be done
       Output:
             tropomi_hcho [satellite]: a dataclass format (see config.py)
    '''
    # read time
    time = _read_group_nc(fname, 1, 'PRODUCT', 'time') +\
        np.nanmean(np.array(_read_group_nc(
            fname, 1, 'PRODUCT', 'delta_time')), axis=0)/1000.0
    time = np.squeeze(time)
    tropomi_no2 = satellite
    tropomi_no2.time = datetime.datetime(
        2010, 1, 1) + datetime.timedelta(seconds=int(time))
    #print(datetime.datetime.strptime(str(tropomi_no2.time),"%Y-%m-%d %H:%M:%S"))
    # read lat/lon at corners
    tropomi_no2.latitude_corner = _read_group_nc(fname, 3, ['PRODUCT', 'SUPPORT_DATA',
                                                            'GEOLOCATIONS'], 'latitude_bounds').astype('float32')
    tropomi_no2.longitude_corner = _read_group_nc(fname, 3, ['PRODUCT', 'SUPPORT_DATA',
                                                             'GEOLOCATIONS'], 'longitude_bounds').astype('float32')
    # read total amf
    amf_total = _read_group_nc(fname, 1, 'PRODUCT', 'air_mass_factor_total')
    # read trop no2
    vcd = _read_group_nc(
        fname, 1, 'PRODUCT', 'nitrogendioxide_tropospheric_column').astype('float32')
    scd = _read_group_nc(fname, 1, 'PRODUCT', 'nitrogendioxide_tropospheric_column').astype('float32') *\
        _read_group_nc(fname, 1, 'PRODUCT', 'air_mass_factor_troposphere')
    tropomi_no2.vcd = vcd*6.02214*1e19*1e-15
    tropomi_no2.scd = scd*6.02214*1e19*1e-15
    # read quality flag
    tropomi_no2.qa = _read_group_nc(
        fname, 1, 'PRODUCT', 'qa_value').astype('float32')
    # read pressures for SWs
    tm5_a = _read_group_nc(fname, 1, 'PRODUCT', 'tm5_constant_a')
    tm5_a = np.concatenate((tm5_a[:, 0], 0), axis=None)
    tm5_b = _read_group_nc(fname, 1, 'PRODUCT', 'tm5_constant_b')
    tm5_b = np.concatenate((tm5_b[:, 0], 0), axis=None)
    ps = _read_group_nc(fname, 3, [
                        'PRODUCT', 'SUPPORT_DATA', 'INPUT_DATA'], 'surface_pressure').astype('float32')
    p_mid = np.zeros(
        (34, np.shape(vcd)[0], np.shape(vcd)[1])).astype('float32')
    SWs = np.zeros((34, np.shape(vcd)[0], np.shape(vcd)[1])).astype('float32')
    AKs = _read_group_nc(fname, 1, 'PRODUCT',
                         'averaging_kernel').astype('float32')
    for z in range(0, 34):
        p_mid[z, :, :] = 0.5*(tm5_a[z]+tm5_b[z]*ps[:, :] +
                              tm5_a[z+1]+tm5_b[z+1]*ps[:, :])/100
        SWs[z, :, :] = AKs[:, :, z]*amf_total
    tropomi_no2.pressure_mid = p_mid
    tropomi_no2.averaging_kernels = SWs
    # read the tropopause layer index
    trop_layer = _read_group_nc(
        fname, 1, 'PRODUCT', 'tm5_tropopause_layer_index')
    tropomi_no2.tropopause = np.zeros_like(trop_layer).astype('float32')
    for i in range(0, np.shape(trop_layer)[0]):
        for j in range(0, np.shape(trop_layer)[1]):
            if trop_layer[i, j] > 0:
                tropomi_no2.tropopause[i, j] = p_mid[trop_layer[i, j], i, j]
            else:
                tropomi_no2.tropopause[i, j] = np.nan
    # read the precision
    tropomi_no2.uncertainty = _read_group_nc(fname, 1, 'PRODUCT',
                                             'nitrogendioxide_tropospheric_column_precision').astype('float32')
    # interpolation
    if interpolation_flag == True:
        interpolator(1, 1.0, tropomi_no2, geos_model)
    # return
    return tropomi_no2


def tropomi_reader(product_dir: str, satellite_product_num, num_job=1):
    '''
        reading tropomi data
        Output [tropomi]: the tropomi @dataclass
    '''

    # find L2 files first
    L2_files = sorted(glob.glob(product_dir + "/*.nc"))
    # read the files in parallel
    if satellite_product_num == 1:
        outputs = Parallel(n_jobs=num_job)(delayed(tropomi_reader_no2)(
            L2_files[k]) for k in range(len(L2_files)))
    elif satellite_product_num == 2:
        outputs = Parallel(n_jobs=num_job)(delayed(tropomi_reader_hcho)(
            L2_files[k]) for k in range(len(L2_files)))


class readers(object):
    def __init__(self) -> None:
        pass

    def add_satellite_data(self, product_tag: int, product_dir: Path):
        '''
            add L2 data
            Input:
                product_tag [int]: an index specifying the type of data to read:
                                   1 > TROPOMI_NO2
                                   2 > TROPOMI_HCHO
                                   3 > TROPOMI_CH4
                                   4 > TROPOMI_CO     
                product_dir  [Path]: a path object describing the path of L2 files
        '''
        self.satellite_product_dir = product_dir
        self.satellite_product_num = product_tag

    def add_ctm_data(self, product_tag: int, product_dir: Path):
        '''
            add CTM data
            Input:
                product_tag [int]: an index specifying the type of data to read:
                                   1 > GEOS-CCM   
                product_dir  [Path]: a path object describing the path of CTM files
        '''

        self.ctm_product_dir = product_dir
        self.ctm_product_num = product_tag

    def read_satellite_data(self, num_job=1):
        '''
            read L2 satellite data
            Input:
                num_job[int]: the number of cpus for parallel computation
        '''

        tropomi_reader(self.satellite_product_dir.as_posix(),
                       self.satellite_product_num, num_job=num_job)


# testing
if __name__ == "__main__":

    reader_obj = readers()
    reader_obj.add_satellite_data(2, Path('download_bucket/hcho/'))
    reader_obj.read_satellite_data()
