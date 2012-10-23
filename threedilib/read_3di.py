# -*- coding: utf-8 -*-
# (c) Nelen & Schuurmans.  GPL licensed, see LICENSE.rst.
# from lizard_mekong
from __future__ import (
  print_function,
  # unicode_literals,  # gdal can't live with that, yet.
  absolute_import,
  division,
)

from django.core.management.base import BaseCommand, CommandError
from django.conf import settings

from osgeo import (
    osr,
    gdal,
    gdalconst,
)
import numpy
import os

# Some projections
RD = 28992
UTM = 3405
WGS84 = 4326
GOOGLE = 900913


def projection(epsg):
    sr = osr.SpatialReference()
    sr.ImportFromEPSG(epsg)
    return sr.ExportToWkt()


def to_dataset(masked_array,
               geotransform=None,
               dtype=gdalconst.GDT_Float64):
    """
    Return gdal dataset.
    """

    # Create in memory array
    ds = gdal.GetDriverByName('MEM').Create(
        '',  # No filename
        masked_array.shape[1],
        masked_array.shape[0],
        1,  # number of bands
        dtype
    )

    # Coordinates
    ds.SetGeoTransform(geotransform)

    # Write data
    ds.GetRasterBand(1).WriteArray(masked_array.filled())
    ds.GetRasterBand(1).SetNoDataValue(masked_array.fill_value)
    return ds


def to_masked_array(ds):
    """
    Return numpy masked array.
    """
    array = ds.ReadAsArray()
    nodatavalue = ds.GetRasterBand(1).GetNoDataValue()
    result = numpy.ma.array(
        array,
        mask=numpy.equal(array, nodatavalue),
    )
    return result


def reproject(source, match,
              source_projection,
              match_projection,
              method=gdalconst.GRA_Cubic):
    """
    Accepts and resturns gdal datasets.
    """
    destination = gdal.GetDriverByName('MEM').CreateCopy('', match)
    band = destination.GetRasterBand(1)
    band.Fill(band.GetNoDataValue())

    gdal.ReprojectImage(
        source, destination,
        source_projection, match_projection,
        method,
    )

    return destination


def main(*args, **options):
    ds_height_ori = gdal.Open(
        os.path.join(settings.BUILDOUT_DIR, 'data/Mekong_small90.asc'),
    )
    ds_height = gdal.GetDriverByName('MEM').CreateCopy('', ds_height_ori)
    ma_height = to_masked_array(ds_height)

    data = Data(
        os.path.join(settings.BUILDOUT_DIR, 'data/subgrid_map1500.nc'),
    )

    for i in range(0, data.time.size, 300)[0:1]:
        filename = 'result%04i' % i
        print('Processing %s' % filename)
        print('Calculate grid.')
        ma_3di = data.to_masked_array(data.depth, i)
        ds_3di = to_dataset(ma_3di, data.geotransform)

        print('Reproject.')
        # Mess with the geotransform: Shift target coordinates.
        gt_old = ds_height.GetGeoTransform()
        gt_tmp = (
            gt_old[0] - 891 * gt_old[1],
            gt_old[1],
            gt_old[2],
            gt_old[3] - 17 * gt_old[5],
            gt_old[4],
            gt_old[5],
        )
        ds_height.SetGeoTransform(gt_tmp)

        ds_depth = reproject(
            ds_3di, ds_height,
            projection(UTM),
            projection(WGS84),
            gdalconst.GRA_NearestNeighbour,
        )

        # Put correct geotransform back
        ds_height.SetGeoTransform(gt_old)
        ds_depth.SetGeoTransform(gt_old)
        ma_depth = to_masked_array(ds_depth)

        gdal.GetDriverByName('Gtiff').CreateCopy(filename + '.tif', ds_depth)
        gdal.GetDriverByName('AAIGrid').CreateCopy(filename + '.asc', ds_depth)


class Command(BaseCommand):
    args = ''
    help = 'Read netcdf with results 3di results and produce some data.'

    def handle(self, *args, **options):
        main(*args, **options)

