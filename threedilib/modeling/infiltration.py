#!/usr/bin/python
# -*- coding: utf-8 -*-

from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import
from __future__ import division

import argparse
import csv
import itertools
import os

import numpy as np
import gdal

from threedilib.modeling import progress

description = """
    Create an infiltration map from a conversion table, a landuse map
    and a soil map.
"""

def get_parser():
    """ Return argument parser. """
    parser = argparse.ArgumentParser(
        description=description
    )
    parser.add_argument('csvpath',
                        metavar='CSV', help='parameterfile')
    parser.add_argument('soilpath',
                        metavar='SOIL', help='Path to soil grid.')
    parser.add_argument('landusepath',
                        metavar='USE', help='Path to landuse grid')
    parser.add_argument('outputpath',
                        metavar='OUTPUT', help='Path to output geotiff file')
    # Add arguments here.
    return parser


def get_conversion_arrays(csvpath):
    """ Get np conversionarrays. """
    # Make numpy arrays where the position in the array is the id.
    landuse = np.ma.array(np.ma.empty(256), mask=True, dtype=np.uint16)
    soil = np.ma.array(np.ma.empty(256), mask=True, dtype=np.uint16)

    with open(csvpath) as csvfile:
        table = csv.DictReader(csvfile)

        # Fill these arrays from the csv
        for rec in table:
            landuse[int(rec['LG,N,19,2'])] = float(rec['INFIL_PERC,N,5,0'])
            soil[int(rec['SOIL,N,4,0'])] = float(rec['INFIL,N,5,0'])

        return dict(landuse=landuse, soil=soil)


class Dataset(object):
    """ Dataset specifically intended for line by line reading and writing. """
    def __init__(self, path, templatepath=None):
        """ Init. """
        if templatepath is None:
            # Read mode
            self.dataset = gdal.Open(path)
        else:
            # Write mode
            templatedataset = gdal.Open(templatepath)
            driver = gdal.GetDriverByName(b'gtiff')
            self.dataset = driver.Create(
                str(path),
                templatedataset.RasterXSize,
                templatedataset.RasterYSize,
                1,
                gdal.GDT_UInt16,
                ['COMPRESS=DEFLATE'],
            )
            self.dataset.SetGeoTransform(templatedataset.GetGeoTransform())
            self.dataset.GetRasterBand(1).SetNoDataValue(65535)

        self.band = self.dataset.GetRasterBand(1)
        self.width = self.dataset.RasterXSize
        self.height = self.dataset.RasterYSize
        self.nodatavalue = self.band.GetNoDataValue()

    def __len__(self):
        """ Return y size. """
        return self.height

    def __iter__(self):
        """ Iterate masked_arrays representing line-by-line rasterdata. """
        dtype = {gdal.GDT_Byte: np.uint8,
                 gdal.GDT_Float32: np.float32}[self.band.DataType]

        for i in range(self.height):
            data = np.fromstring(
                self.band.ReadRaster(0, i, self.width, 1),
                dtype=dtype,
            )
            yield np.ma.array(data=data,
                              mask=np.equal(data, self.nodatavalue))

    def writelines(self, lines):
        """ Write a number of masked arrays to the dataset. """
        for i, line in enumerate(lines):
            self.band.WriteRaster(
                0, i, self.width, 1,
                np.uint16(line.filled(self.nodatavalue)).tostring(),
            )


def convert(soil, landuse, tables):
    """ Return a generator of output lines. """
    soil_table = tables['soil']
    landuse_table = tables['landuse']
    indicator = progress.Indicator(len(soil))
    for soil_ma, landuse_ma in itertools.izip(soil, landuse):
        infiltration_ma = (soil_table[soil_ma] / 100 *
                           landuse_table[np.uint8(landuse_ma)])
        yield infiltration_ma
        indicator.update()


def command(csvpath, soilpath, landusepath, outputpath):
    """ Do something spectacular. """
    if os.path.exists(outputpath):
        print('{} already exists!'.format(outputpath))
        exit()
    tables = get_conversion_arrays(csvpath)

    # Instantiate datasets
    soil = Dataset(soilpath)
    landuse = Dataset(landusepath)
    output = Dataset(outputpath, templatepath=landusepath)

    lines = convert(soil=soil, landuse=landuse, tables=tables)
    output.writelines(lines)


def main():
    """ Call command with args from parser. """
    command(**vars(get_parser().parse_args()))


if __name__ == '__main__':
    exit(main())
