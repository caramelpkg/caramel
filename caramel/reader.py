import numpy as np
import os
from pathlib import Path
from dataclasses import dataclass
import datetime
import glob
from joblib import Parallel, delayed
from netCDF4 import Dataset

@dataclass
class satellite:
    vcd: np.ndarray
    time: datetime.datetime
    profile: np.ndarray
    tropopause: np.ndarray
    latitude_corner: np.ndarray
    longitude_corner: np.ndarray
    uncertainty: np.ndarray
    quality_flag: np.ndarray
    pressure_mid: np.ndarray
    averaging_kernels: np.ndarray
    amf_total: np.ndarray

def _read_nc(filename,var):
    # reading nc files without a group
    nc_f = filename
    nc_fid = Dataset(nc_f, 'r')
    out = np.array(nc_fid.variables[var])
    nc_fid.close()
    return np.squeeze(out)
def _read_group_nc(filename,num_groups,group,var):
    # reading nc files with a group structure
    nc_f = filename
    nc_fid = Dataset(nc_f, 'r')
    if num_groups == 1:
       out = np.array(nc_fid.groups[group].variables[var])
    elif num_groups == 2:
       out = np.array(nc_fid.groups[group[0]].groups[group[1]].variables[var])
    elif num_groups == 3:
       out = np.array(nc_fid.groups[group[0]].groups[group[1]].groups[group[2]].variables[var])
    nc_fid.close()
    return np.squeeze(out)

def tropomi_reader_no2(product_dir:str,num_job = 1) -> satellite:
    '''
        reading tropomi no2 data
        Output [tropomi]: the tropomi @dataclass
    '''
    def reader_wrapper(fname):
        # a wrapper to run in parallel
        # read time
        time = np.array(_read_group_nc(fname,1,'PRODUCT','time')) +\
               np.nanmean(np.array(_read_group_nc(fname,1,'PRODUCT','delta_time')), axis=0)/1000
        time = np.squeeze(time)
        satellite.time = datetime.datetime(2010, 1, 1) + datetime.timedelta(seconds=int(time))
        # read lat/lon at corners
        satellite.latitude_corner = np.array(_read_group_nc(fname,1,'PRODUCT','latitude'))
        satellite.longitude_corner = np.array(_read_group_nc(fname,1,'PRODUCT','longitude'))
        # read total amf
        amf_total = _read_group_nc(fname,1,'PRODUCT','air_mass_factor_total')
        # read trop no2
        vcd = _read_group_nc(fname,1,'PRODUCT','nitrogendioxide_tropospheric_column')
        satellite.vcd = vcd*6.02214*1e19*1e-15
        # read quality flag
        satellite.qa = _read_group_nc(fname,1,'PRODUCT','qa_value')
        # read pressures for SWs
        tm5_a = np.array(_read_group_nc(fname,1,'PRODUCT','tm5_constant_a'))
        tm5_a = np.concatenate((tm5_a[:,0], 0), axis=None)
        tm5_b = np.array(_read_group_nc(fname,1,'PRODUCT','tm5_constant_b'))
        tm5_b = np.concatenate((tm5_b[:,0], 0), axis=None)
        ps = np.array(_read_group_nc(fname,3,['PRODUCT','SUPPORT_DATA','INPUT_DATA'],'surface_pressure'))
        p_mid = np.zeros((34,np.shape(vcd)[0], np.shape(vcd)[1]))
        SWs = np.zeros((34,np.shape(vcd)[0], np.shape(vcd)[1]))
        AKs = np.array(_read_group_nc(fname,1,'PRODUCT','averaging_kernel'))
        for z in range(0,34):
            p_mid[z,:,:] = 0.5*(tm5_a[z]+tm5_b[z]*ps[:,:]+tm5_a[z+1]+tm5_b[z+1]*ps[:,:])/100
            SWs[z,:,:] = AKs[:,:,z]*amf_total
        satellite.pressure_mid = p_mid
        satellite.averaging_kernels = SWs
        # read the precision
        satellite.uncertainty = np.array(_read_group_nc(fname,1,'PRODUCT','nitrogendioxide_tropospheric_column_precision'))
        # return
        return satellite

    # find L2 files first
    L2_files = sorted(glob.glob(product_dir + "/*.nc"))
    # read the files in parallel
    outputs = Parallel(n_jobs=num_job)(delayed(reader_wrapper)(L2_files[k]) for k in range(len(L2_files)))

class readers(object):
    def __init__(self,product_name:int,product_dir:Path,num_job = 1) -> None:
        '''
            Initialize the reader object
            Input:
                product_name [int]: an index specifying the type of data to read:
                                   1 > TROPOMI_NO2
                                   2 > TROPOMI_HCHO
                                   3 > TROPOMI_CH4
                                   4 > TROPOMI_CO     
                product_dir  [Path]: a path object describing the path of L2 files
        '''
        self.product_dir = product_dir

        if product_name == 1:
           self.output = tropomi_reader_no2(self.product_dir.stem,num_job = num_job)
# testing
if __name__ == "__main__":

    reader_obj = readers(1,Path('download_bucket/'),1)
    reader_obj.output