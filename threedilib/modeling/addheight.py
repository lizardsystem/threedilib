# -*- coding: utf-8 -*-
# (c) Nelen & Schuurmans.  GPL licensed, see LICENSE.rst.

from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import
from __future__ import division

import argparse
import os
import re

from osgeo import gdal
from osgeo import ogr

import numpy as np

from threedilib.modeling import progress
from threedilib import config

SHEET = re.compile('^i(?P<unit>[0-9]{2}[a-z])[a-z][0-9]_[0-9]{2}$')


def get_index():
    """ Return index from container or open from config location. """
    key = 'index'
    if key not in cache:
        dataset = ogr.Open(config.INDEX_PATH)
        cache[key] = dataset
    return cache[key][0]
    

def get_args():
    """ Return arguments dictionary. """
    parser = argparse.ArgumentParser(description='No description yet.')
    parser.add_argument('source',
                        metavar='SOURCE',
                        help=('Shapefile with linestrings in a single layer.'))
    parser.add_argument('target',
                        metavar='TARGET',
                        help=('Target shapefile, will be overwritten.'))
    return vars(parser.parse_args())


def get_vector_and_offset(line):
    """ Return vector, offset tuple. """
    x0, x1 = np.array(line)
    return x1 - x0, x0


def parameterize_intersects(cellsize, vector, offset):
    """
    Return numpy array of parameters.

    Parameters represent start, optional intersections, and end.
    Each parameter can be converted to a point via
    (parameter * vector + offset)
    """
    # Determine extents (two times ceil, because of how np.arange works)
    points = np.array([offset, offset + vector])
    left, bottom = np.ceil(np.min(points, axis=0) / cellsize) * cellsize
    right, top = np.ceil(np.max(points, axis=0) / cellsize) * cellsize
    # Determine intersections with gridlines
    width, height = cellsize
    x_intersects = np.arange(left, right, width)
    y_intersects = np.arange(bottom, top, height)
    # Determine parameters corresponding to intersections
    x_parameters = (x_intersects - offset[0]) / vector[0]
    y_parameters = (y_intersects - offset[1]) / vector[1]
    # Return sorted, distinct parameters, including start and and.
    return tuple(np.sort(np.unique(np.concatenate([(0, 1),
                                                   x_parameters,
                                                   y_parameters]))))


def segmentize_by_tiles(line):
    """ Return generator of line tuples. """
    vector, offset = get_vector_and_offset(line)
    parameters = parameterize_intersects(vector=vector,
                                         offset=offset,
                                         cellsize=(1000, 1250))
    for i in range(len(parameters) - 1):
        result = (tuple(offset + vector * parameters[i]),
                  tuple(offset + vector * parameters[i + 1]))
        yield result


def segmentize_by_points(linestring):
    """ Return generator of line tuples. """
    for i in range(linestring.GetPointCount() - 1):
        yield linestring.GetPoint_2D(i), linestring.GetPoint_2D(i + 1)


def segmentize(linestring):
    """ Return generator of linestring geometries. """
    for line in segmentize_by_points(linestring):
        for (x0, y0), (x1, y1) in segmentize_by_tiles(line):
            segment = ogr.Geometry(ogr.wkbLineString)
            segment.AddPoint_2D(x0, y0)
            segment.AddPoint_2D(x1, y1)
            yield segment


def get_dataset(leaf):
    """ Return gdal_dataset from cache or file. """
    leafno = leaf[b'BLADNR']
    if leafno in cache:
        return cache[leafno]

    for key in cache.keys():
        del cache[key]  # Maybe unnecessary, see top and lsof.

    # Add to cache and return.
    unit = SHEET.match(leafno).group('unit')
    path = os.path.join(config.AHN_PATH, unit, leafno + '.tif')
    cache[leafno] = gdal.Open(path)
    return cache[leafno]


def get_values(dataset, lines):
    """ Return the height from dataset. """
    geotransform = np.array(dataset.GetGeoTransform())
    cellsize = geotransform[np.array([[1, 5]])]  # Note no abs() now!
    origin = geotransform[np.array([[0, 3]])]
    # Find coordinates of top left points of lines
    coordinates = np.array([lines[..., 0].min(1), lines[..., 1].max(1)]).T
    # Make indices
    indices = tuple(np.int64((coordinates - origin) / cellsize).T)[::-1]
    # Use indices to query the data
    return dataset.ReadAsArray()[indices]


def pixelize(segment):
    """ Return lines, values tuple of numpy arrays. """
    # Get tile and check if it is the only tile
    index = get_index()
    index.SetSpatialFilter(segment.Centroid())
    if index.GetFeatureCount() > 1:
        raise ValueError('There should be only one tile per segment!')
    leaf = index.GetNextFeature()
    dataset = get_dataset(leaf)
    # Determine lines
    geotransform = dataset.GetGeoTransform()
    cellsize = abs(geotransform[1]), abs(geotransform[5])
    vector, offset = get_vector_and_offset(segment.GetPoints())
    parameters = parameterize_intersects(vector=vector,
                                         offset=offset,
                                         cellsize=cellsize)
    points = np.array([offset]) + np.array([vector]) * np.array([parameters]).T
    lines = np.array([points[:-1].T, points[1:].T]).transpose(2, 0, 1)
    # Get values
    values = get_values(dataset, lines)
    return lines, values


def get_initialized_shape(path):
    """ Return ogr dataset. """
    # Prepare in-memory ogr layer
    # driver = ogr.GetDriverByName(b'Memory')
    driver = ogr.GetDriverByName(b'ESRI Shapefile')
    if os.path.exists(path):
        driver.DeleteDataSource(str(path))

    dataset = driver.CreateDataSource(str(path))
    layer = dataset.CreateLayer(b'')

    # Add type field
    field_definition_type = ogr.FieldDefn(b'Type', ogr.OFTString)
    field_definition_type.SetWidth(64)
    layer.CreateField(field_definition_type)

    # Add height field
    field_definition_height = ogr.FieldDefn(b'Height', ogr.OFTReal)
    layer.CreateField(field_definition_height)

    return dataset


def write_to_layer(layer, lines, values, name='Sjef'):
        layer_definition = layer.GetLayerDefn()
        for line, value in zip(lines, values):
            geometry = ogr.Geometry(ogr.wkbLineString)
            geometry.AddPoint_2D(*line[0])
            geometry.AddPoint_2D(*line[1])
            feature = ogr.Feature(layer_definition)
            feature.SetGeometry(geometry)
            feature.SetField(b'Type', str(name))
            feature.SetField(b'Height', float(value))
            layer.CreateFeature(feature)


def main():
    args = get_args()
    source_path = args['source']
    source_dataset = ogr.Open(source_path)
    source_layer = source_dataset[0]

    target_path = args['target']
    target_dataset = get_initialized_shape(target_path)
    target_layer = target_dataset[0]

    # Count planned work
    total = 0
    for feature in source_layer:
        for segment in segmentize(feature.geometry()):
            total += 1
    source_layer.ResetReading()

    indicator = progress.Indicator(total)
    for feature in source_layer:
        name = feature[b'TYPE']
        for i, segment in enumerate(segmentize(feature.geometry())):
            lines, values = pixelize(segment)
            write_to_layer(layer=target_layer,
                           lines=lines, values=values, name=name)
            indicator.update()

    # Close the datasets
    source_dataset = None
    target_dataset = None
    cache = None


cache = {}  # Contains leafno's and the index

if __name__ == '__main__':
    exit(main())
