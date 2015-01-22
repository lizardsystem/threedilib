# -*- coding: utf-8 -*-
# (c) Nelen & Schuurmans.  GPL licensed, see LICENSE.rst.

# -*- coding: utf-8 -*-

from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import
from __future__ import division

import argparse
import os

from netCDF4 import Dataset
from osgeo import gdal
from osgeo import gdal_array

import numpy as np

from threedilib.subgrid import quads

gdal.UseExceptions()


def get_parser():
    """ Return argument parser. """
    parser = argparse.ArgumentParser(
        description=""
    )
    parser.add_argument('sourcepath',
                        metavar='SOURCE',
                        help="Path to subgrid netCDF file.")
    parser.add_argument('-v', '--variables',
                        nargs='+',
                        default=['s1', 'dep'],
                        metavar='VARIABLES')
    parser.add_argument('-t', '--timestep', metavar='TIMESTEP', type=int)
    return parser


class SubgridExtractor(object):

    def __init__(self, sourcepath):
        """ Set sourcepath. """
        self.sourcepath = sourcepath

    def __enter__(self):
        """ Init quads and arrays. """
        self.driver = gdal.GetDriverByName(b'aaigrid')
        self.nodata = -999

        self.dsnetcdf = Dataset(self.sourcepath)
        self.dsquads = quads.get_dataset(self.sourcepath)

        length = self.dsnetcdf.variables['s1'].shape[1]
        self.array1d = self.nodata * np.ones(1 + length)
        self.array2d = self.dsquads.ReadAsArray()
        return self

    def __exit__(self, exception_type, error, traceback):
        """ Close netcdf dataset. """
        self.dsnetcdf.close()

    def save(self, variable, timestep, targetpath):
        """ Read variable at timestep, save as asc. """
        ncvariable = self.dsnetcdf.variables[variable]
        if timestep is None:
            print('Getting maximum value of {}.'.format(variable))
            values = ncvariable[:].max(0)
        else:
            print('Getting value of {} at {}.'.format(variable, timestep))
            values = ncvariable[:][timestep]

        # paste
        if np.ma.isMaskedArray(values):
            self.array1d[0: -1] = values.filled(self.nodata)
        else:
            self.array1d[0: -1] = values

        # Combine arrays
        result = self.array1d[self.array2d]

        # Create asciifile
        dsresult = gdal_array.OpenArray(result)
        dsresult.SetGeoTransform(self.dsquads.GetGeoTransform())
        dsresult.GetRasterBand(1).SetNoDataValue(self.nodata)
        self.driver.CreateCopy(
            targetpath,
            dsresult,
            options=[b'DECIMAL_PRECISION=3']
        )


def command(sourcepath, variables, timestep=None):
    """ Do something spectacular. """
    with SubgridExtractor(sourcepath) as extractor:
        for variable in variables:
            root, ext = os.path.splitext(sourcepath)
            targetpath = '{root}_{variable}_{timestep}.asc'.format(
                root=root,
                variable=variable,
                timestep='max' if timestep is None else timestep,
            )
            extractor.save(variable=variable,
                           timestep=timestep,
                           targetpath=targetpath)


def main():
    """ Call command with args from parser. """
    command(**vars(get_parser().parse_args()))
