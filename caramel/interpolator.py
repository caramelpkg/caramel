import numpy as np
from config import satellite, ctm_model
from scipy.spatial import Delaunay
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator
from test_plotter import test_plotter

def _interpolosis(tri: Delaunay, Z, X, Y, interpolator_type):
    # to make the interpolator() shorter
    if interpolator_type == 1:
        interpolator = LinearNDInterpolator(tri, (Z).flatten())
        ZZ = interpolator((X, Y))
    elif interpolator_type == 2:
        interpolator = NearestNDInterpolator(tri, (Z).flatten())
        ZZ = interpolator((X, Y))
    else:
        raise Exception(
            "other type of interpolation methods has not been implemented yet")
    return ZZ


def interpolator(interpolator_type: int, grid_size: float, sat_data: satellite, ctm_models_coordinate: dict):
    '''
        Initialize the interpolator function
        Input:
            interpolator_type [int]: an index specifying the type of interpolator
                    1 > Bilinear interpolation (recommended)
                    2 > Nearest neighbour
                    3 > Cressman (not implemented yet)
                    4 > Poppy (not implemented yet)     
            sat_data  [satellite]: a dataclass for satellite data
            ctm_models_coordinate [dic]: a dictionary containing lat and lon of the model
    '''

    # creating the delaunay triangulation on satellite coordinates
    # get the center lat/lon
    sat_center_lat = np.nanmean(sat_data.latitude_corner, axis=2).squeeze()
    sat_center_lon = np.nanmean(sat_data.longitude_corner, axis=2).squeeze()
    # mask bad data
    mask = sat_data.quality_flag <= 0.75
    mask = np.multiply(mask, 1.0).squeeze()
    mask[mask == 0] = np.nan
    # define the triangulation
    points = np.zeros((np.size(sat_center_lat), 2))
    points[:, 0] = sat_center_lon.flatten()
    points[:, 1] = sat_center_lat.flatten()
    tri = Delaunay(points)
    # define the grid
    lat_ctm_min = np.min(ctm_models_coordinate['Latitude'].flatten())
    lat_ctm_max = np.max(ctm_models_coordinate['Latitude'].flatten())
    lon_ctm_min = np.min(ctm_models_coordinate['Longitude'].flatten())
    lon_ctm_max = np.max(ctm_models_coordinate['Longitude'].flatten())

    dx = 0.0  # buffer
    lon_grid = np.arange(lon_ctm_min-dx, lon_ctm_max+dx, grid_size)
    lat_grid = np.arange(lat_ctm_min-dx, lat_ctm_max+dx, grid_size)
    lons_grid, lats_grid = np.meshgrid(lon_grid, lat_grid)

    # interpolate 2Ds fields
    interpolated_sat = satellite

    interpolated_sat.vcd = _interpolosis(
        tri, sat_data.vcd*mask, lons_grid, lats_grid, interpolator_type)
    test_plotter(lons_grid,lats_grid,interpolated_sat.vcd)

    interpolated_sat.scd = _interpolosis(
        tri, sat_data.scd*mask, lons_grid, lats_grid, interpolator_type)
    interpolated_sat.tropopause = _interpolosis(
        tri, sat_data.tropopause*mask, lons_grid, lats_grid, interpolator_type)
    interpolated_sat.latitude_center = lats_grid
    interpolated_sat.longitude_center = lons_grid
    interpolated_sat.uncertainty = _interpolosis(
        tri, sat_data.uncertainty*mask, lons_grid, lats_grid, interpolator_type)

    # interpolate 3Ds fields
    for z in range(0, np.shape(sat_data.pressure_mid)[0]):
        interpolated_sat.averaging_kernels[z, :, :] = _interpolosis(tri, sat_data.scattering_weights[z, :, :].squeeze()
                                                                    * mask, lons_grid, lats_grid, interpolator_type)
        interpolated_sat.pressure_mid[z, :, :] = _interpolosis(tri, sat_data.pressure_mid[z, :, :].squeeze()
                                                               * mask, lons_grid, lats_grid, interpolator_type)
