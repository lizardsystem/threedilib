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

DESCRIPTION = """
    Convert a shapefile containing 2D linestrings to a shapefile
    containing 3D linestrings where the third coordinate is the height
    according to the a resolution height map. Each line is segmentized
    in lines that span exactly one pixel of the height map, after which
    the midpoint of these lines becomes an 3D point in the resulting line.

    For the script to work, a configuration variable AHN_PATH must be
    set in threedilib/localconfig.py pointing to the location of the
    height map, and a variable INDEX_PATH pointing to the .shp file that
    contains the index to the heightmap.
"""

SHEET = re.compile('^i(?P<unit>[0-9]{2}[a-z])[a-z][0-9]_[0-9]{2}$')


def get_args():
    """ Return arguments dictionary. """
    parser = argparse.ArgumentParser(
        description=DESCRIPTION,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('source_path',
                        metavar='SOURCE',
                        help=('Path to shapefile with 2D linestrings.'))
    parser.add_argument('target_path',
                        metavar='TARGET',
                        help=('Path to target shapefile, to be overwritten.'))
    parser.add_argument('-a', '--attribute',
                        metavar='HEIGHT_ATTRIBUTE',
                        help=('Create layer per feature and a feature '
                              'per segment, each feature having a height '
                              'attribute with the name <HEIGHT_ATTRIBUTE>.'))
    parser.add_argument('-d', '--distance',
                        metavar='DISTANCE',
                        type=float,
                        help=('Distance to look perpendicular to the '
                              'segments to find the highest points '
                              'on the height map. Disabled by default.'))
    parser.add_argument('-r', '--relocate',
                        metavar='RELOCATE',
                        type=float,
                        help=('Modify the line to follow the '
                              'highest points on the height map.'))

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


def get_index():
    """ Return index from container or open from config location. """
    key = 'index'
    if key not in cache:
        if os.path.exists(config.INDEX_PATH):
            dataset = ogr.Open(config.INDEX_PATH)
        else:
            raise OSError('File not found :{}'.format(config.INDEX_PATH))
        cache[key] = dataset
    return cache[key][0]


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


def get_values(dataset, points):
    """ Return the height from dataset. """
    geotransform = np.array(dataset.GetGeoTransform())
    cellsize = geotransform[np.array([[1, 5]])]  # Note no abs() now!
    origin = geotransform[np.array([[0, 3]])]
    # Make indices
    indices = tuple(np.int64((points - origin) / cellsize).T)[::-1]
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
    ends = np.array([offset]) + np.array([vector]) * np.array([parameters]).T
    lines = np.array([ends[:-1].T, ends[1:].T]).transpose(2, 0, 1)
    points = lines.mean(1)

    # Get values
    values = get_values(dataset, points)
    return lines, points, values


def get_target_dataset(path):
    """
    Return ogr dataset.

    delete dataset at path if it exists.
    """
    # driver = ogr.GetDriverByName(b'Memory')
    driver = ogr.GetDriverByName(b'ESRI Shapefile')
    if os.path.exists(path):
        driver.DeleteDataSource(str(path))

    dataset = driver.CreateDataSource(str(path))
    return dataset


def get_target_layer(target_dataset, source_layer):
    """
    Return layer.

    Adds an empty layer with same definition as source layer on target dataset.
    """
    target_layer = target_dataset.CreateLayer(source_layer.GetName())

    # Copy field definitions
    source_layer_definition = source_layer.GetLayerDefn()
    for i in range(source_layer_definition.GetFieldCount()):
        target_layer.CreateField(source_layer_definition.GetFieldDefn(i))
    return target_layer


def get_target_feature(target_layer, source_feature):
    """
    Return feature.

    Adds empty feature with same attributes as source feature to target layer.
    """
    target_layer_definition = target_layer.GetLayerDefn()
    target_feature = ogr.Feature(target_layer_definition)

    target_feature = ogr.Feature(target_layer_definition)
    for key, value in source_feature.items().items():
        target_feature[key] = value
    return target_feature


def convert_geometry(source_geometry, distance, indicator):
    """
    Set target feature's geometry to converted source feature's geometry.
    """
    target_geometry = ogr.Geometry(ogr.wkbLineString)
    for i, segment in enumerate(segmentize(source_geometry)):
        lines, points, values = pixelize(segment)

        # Add first point of the first line if this is the first segment
        if i == 0:
            target_geometry.AddPoint(float(lines[0, 0, 0]),
                                     float(lines[0, 0, 1]),
                                     float(values[0]))

        # Add the rest of the points (x, y) and values (z)
        for (x, y), z in zip(points, values):
            target_geometry.AddPoint(float(x), float(y), float(z))

        indicator.update()

    # Add the last point of the last line of the last segment
    target_geometry.AddPoint(float(lines[-1, 1, 0]),
                             float(lines[-1, 1, 1]),
                             float(values[-1]))
    return target_geometry


def addheight(source_path, target_path, distance, relocate, attribute):
    """
    Take linestrings from source and create target with height added.

    Source and target are both shapefiles.
    """
    # Open datasets
    source_dataset = ogr.Open(source_path)
    target_dataset = get_target_dataset(target_path)

    # Count work
    count = 0
    for source_layer in source_dataset:
        for source_feature in source_layer:
            for segment in segmentize(source_feature.geometry()):
                count += 1
        source_layer.ResetReading()
    indicator = progress.Indicator(count)

    for source_layer in source_dataset:
        target_layer = get_target_layer(target_dataset, source_layer)

        for source_feature in source_layer:
            source_geometry = source_feature.geometry()
            target_geometry = convert_geometry(source_geometry=source_geometry,
                                               distance=distance,
                                               indicator=indicator)

            target_feature = get_target_feature(target_layer, source_feature)
            target_feature.SetGeometry(target_geometry)
            target_layer.CreateFeature(target_feature)

    # Close the datasets
    source_dataset = None
    target_dataset = None


def main():
    """ Calls addheight function with args from commandline. """
    args = get_args()
    addheight(**args)


cache = {}  # Contains leafno's and the index

if __name__ == '__main__':
    exit(main())
