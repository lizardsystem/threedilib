# Run this code in linux, no errors with combination gdal and netCDF4.
# Depends on lizard_raster.

import os

from read_3di import to_dataset
from nc import Data

from netCDF4 import Dataset
from osgeo import gdal
import matplotlib as mpl
from PIL import Image


#from lizard_raster.raster import get_ahn_indices
from lizard_raster import models
import logging


logger = logging.getLogger(__name__)

subgrid_root = "/home/jack/3di-subgrid/bin"
subgrid_exe = os.path.join(subgrid_root, "subgridf90")
copy_to_work_dir = [
    'subgrid.ini',
    'unstruc.hlp',
    'ISOCOLOUR.hls',
    'land.rgb',
    'water.rgb',
    'interception.rgb',
    'land.zrgb',
    'interact.ini',
    'UNSA_SIM.INP',
    'ROOT_SIM.INP',
    'CROP_OW.PRN',
    'CROPFACT',
    'EVAPOR.GEM',
    ]


def setup_3di(full_path, source_files_dir=subgrid_root):
    """
    Copies default files to 3di work folder
    """
    logger.info('Setting up working directory in %s...' % full_path)

    # Copy default files
    for filename in copy_to_work_dir:
        dst_filename = os.path.join(full_path, filename)
        if not os.path.exists(dst_filename):
            src_filename = os.path.join(source_files_dir, filename)
            logger.info('Copying %s from defaults...' % filename)
            copyfile(src_filename, dst_filename)
        else:
            logger.info('%s exists, ok.' % filename)
    # TODO: Extra checks, update files, etc.
    # subgrid.ini


def setup_and_run_3di(
    mdu_full_path,
    skip_if_results_available=True,
    source_files_dir=subgrid_root,
    subgrid_exe=subgrid_exe):
    """This wil produce a file located on result_filename.

    You are responsible for moving the file away.
    """
    full_path = os.path.dirname(mdu_full_path)
    result_filename = os.path.join(full_path, 'subgrid_map.nc')

    # Go to working dir
    os.chdir(full_path)

    if os.path.exists(result_filename) and skip_if_results_available:
        print 'skipping calculation, already calculated.'
    else:
        setup_3di(full_path, source_files_dir)
        # Run
        os.system('%s %s' % (subgrid_exe, mdu_full_path))

        # Results in full_path + subgrid_map.nc
        # *.tim is also produced.

    # # process results
    # process_3di_nc(result_filename)
    return result_filename


def process_3di_nc(filename):
    """Process result netcdf file"""

    #from netCDF4 import Dataset
    rootgrp = Dataset(filename, 'r', format='NETCDF4')
    # for variable_name, variable in rootgrp.variables.items():
    #     print variable_name, variable

    # Depth
    dep = rootgrp.variables['dep']  # timestep, elem
    flow_elem_contour_x = rootgrp.variables['FlowElemContour_x']  # elem, value
    flow_elem_contour_y = rootgrp.variables['FlowElemContour_y']

    #import pdb; pdb.set_trace()

    num_timesteps, num_elements = dep.shape

    # Safety checks
    assert flow_elem_contour_x.shape == flow_elem_contour_y.shape
    assert num_elements == flow_elem_contour_x.shape[0]

    #
    for timestep in range(num_timesteps):
        print('Working on timestep %d...' % timestep)
        for elem in range(num_elements):
            dep[timestep, elem]
            flow_elem_contour_x[elem]
            flow_elem_contour_y[elem]

    rootgrp.close()


def post_process_3di(full_path, dst_basefilename='_step%d.png'):
    """
    Simple version: do not use AHN tiles to do the calculation

    This method is quite fast, but the result has squares.

    Input: full path of the .nc netcdf file

    Output: png files on disk (specified by dst_basefilename) and a
    dictionary with timesteps and the corresponding files.
    """
    print 'post processing %s...' % full_path
    data = Data(full_path)  # NetCDF data
    #process_3di_nc(full_path)

    result_filenames = {}

    for timestep in range(data.num_timesteps):
        print('Working on timestep %d...' % timestep)

        ma_3di = data.to_masked_array(data.depth, timestep)
        ds_3di = to_dataset(ma_3di, data.geotransform)
        # testing
        #print ', '.join([i.bladnr for i in get_ahn_indices(ds_3di)])


        cdict = {
            'red': ((0.0, 51./256, 51./256),
                    (0.5, 237./256, 237./256),
                    (1.0, 83./256, 83./256)),
            'green': ((0.0, 114./256, 114./256),
                      (0.5, 245./256, 245./256),
                      (1.0, 83./256, 83./256)),
            'blue': ((0.0, 54./256, 54./256),
                     (0.5, 170./256, 170./256),
                     (1.0, 83./256, 83./256)),
            }
        colormap = mpl.colors.LinearSegmentedColormap('something', cdict, N=1024)

        min_value, max_value = 0.0, 4.0
        normalize = mpl.colors.Normalize(vmin=min_value, vmax=max_value)
        rgba = colormap(normalize(ma_3di), bytes=True)
        #rgba[:,:,3] = np.where(rgba[:,:,0], 153 , 0)

        dst_filename = dst_basefilename % timestep
        Image.fromarray(rgba).save(dst_filename, 'PNG')

        #write_pgw(tmp_base + '.pgw', extent)
        result_filenames[timestep] = dst_filename

        # gdal.GetDriverByName('Gtiff').CreateCopy(filename_base + '.tif', ds_3di)
        # gdal.GetDriverByName('AAIGrid').CreateCopy(filename_base + '.asc', ds_3di)
    return result_filenames


def post_process_detailed_3di(full_path):
    """
    Make detailed images using a 0.5m height map.

    TODO: implement this
    """
    print 'post processing %s...' % full_path
    data = Data(full_path)  # NetCDF data

    # TODO: Find out which AHN tiles

    for timestep in range(data.num_timesteps):
        print('Working on timestep %d...' % timestep)

        ma_3di = data.to_masked_array(data.depth, timestep)
        ds_3di = to_dataset(ma_3di, data.geotransform)
        # testing
        ahn_indices = models.AhnIndex.get_ahn_indices(ds_3di)
        print ahn_indices
        for ahn_index in ahn_indices:
            print 'reading ahn data... %s' % ahn_index
            ahn_index.get_ds()

        filename_base = '_step%d' % timestep

        cdict = {
            'red': ((0.0, 51./256, 51./256),
                    (0.5, 237./256, 237./256),
                    (1.0, 83./256, 83./256)),
            'green': ((0.0, 114./256, 114./256),
                      (0.5, 245./256, 245./256),
                      (1.0, 83./256, 83./256)),
            'blue': ((0.0, 54./256, 54./256),
                     (0.5, 170./256, 170./256),
                     (1.0, 83./256, 83./256)),
            }
        colormap = mpl.colors.LinearSegmentedColormap('something', cdict, N=1024)

        min_value, max_value = 0.0, 4.0
        normalize = mpl.colors.Normalize(vmin=min_value, vmax=max_value)
        rgba = colormap(normalize(ma_3di), bytes=True)
        #rgba[:,:,3] = np.where(rgba[:,:,0], 153 , 0)

        Image.fromarray(rgba).save(filename_base + '.png', 'PNG')

        #write_pgw(tmp_base + '.pgw', extent)
