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
from threedilib.modeling import vector
from threedilib import config

DESCRIPTION = """
    Convert a shapefile containing 2D linestrings to a shapefile with
    embedded elevation from an elevation map

    Target shapefile can have two layouts: A 'point' layout where the
    elevation is stored in the third coordinate of a 3D linstring, and
    a 'line' layout where a separate feature is created in the target
    shapefile for each segment of each feature in the source shapefile,
    with two extra attributes compared to the original shapefile, one
    to store the elevation, and another to store an arbitrary feature
    id referring to the source feature in the source shapefile.

    In the simplest mode 'fixed' the horizontal geometry is not altered
    and the elevation at the line is used. In 'range' mode the maximum
    elevation is taken from a line perpendicular to the linestring. In
    'change' mode, the horizontal geometry is adjusted to match the
    maxima as found by the 'range' mode.

    containing 3D linestrings where the third coordinate is the elevation
    according to the a resolution elevation map. Each line is segmentized
    in lines that span exactly one pixel of the elevation map, after which
    the midpoint of these lines becomes an 3D point in the resulting line.

    For the script to work, a configuration variable AHN_PATH must be
    set in threedilib/localconfig.py pointing to the location of the
    elevation map, and a variable INDEX_PATH pointing to the .shp file
    that contains the index to the elevation map.
"""

MODE_FIXED = 'simple'
MODE_RANGE = 'range'
MODE_CHANGE = 'change'

LAYOUT_POINT = 'point'
LAYOUT_LINE = 'line'

SHEET = re.compile('^i(?P<unit>[0-9]{2}[a-z])[a-z][0-9]_[0-9]{2}$')


def get_parser():
    """ Return arguments dictionary. """
    parser = argparse.ArgumentParser(
        description=DESCRIPTION,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('source_path',
                        metavar='SOURCE',
                        help='Path to shapefile with 2D linestrings.')
    parser.add_argument('target_path',
                        metavar='TARGET',
                        help='Path to target shapefile.')
    parser.add_argument('-o', '--overwrite',
                        action='store_true',
                        help='Overwrite TARGET if it exists.')
    parser.add_argument('-l', '--layout',
                        metavar='LAYOUT',
                        choices=[LAYOUT_POINT, LAYOUT_LINE],
                        default=LAYOUT_POINT,
                        help="Target shapefile layout.")
    parser.add_argument('-m', '--mode',
                        metavar='MODE',
                        choices=[MODE_FIXED, MODE_RANGE, MODE_CHANGE],
                        default=MODE_FIXED,
                        help="Mode of geometry finding.")
    parser.add_argument('-f', '--feature-id-attribute',
                        metavar='FEATURE_ID_ATTRIBUTE',
                        default='_feat_id',
                        help='Attribute name for the feature id.')
    parser.add_argument('-e', '--elevation-attribute',
                        metavar='ELEVATION_ATTRIBUTE',
                        default='_elevation',
                        help='Attribute name for the elevation.')
    parser.add_argument('-d', '--distance',
                        metavar='DISTANCE',
                        type=float,
                        default=10,
                        help=('Distance (half-width) to look '
                              'perpendicular to the segments to '
                              'find the highest points on the '
                              'elevation map. Disabled by default.'))
    return parser


def get_direction_and_offset(line):
    """ Return direction, offset tuple. """
    x0, x1 = np.array(line)
    return x1 - x0, x0


def parameterize_intersects(cellsize, direction, offset):
    """
    Return numpy array of parameters.

    Parameters represent start, optional intersections, and end.
    Each parameter can be converted to a point via
    (parameter * direction + offset)
    """
    # Determine extents (two times ceil, because of how np.arange works)
    points = np.array([offset, offset + direction])
    left, bottom = np.ceil(np.min(points, axis=0) / cellsize) * cellsize
    right, top = np.ceil(np.max(points, axis=0) / cellsize) * cellsize
    # Determine intersections with gridlines
    width, height = cellsize
    x_intersects = np.arange(left, right, width)
    y_intersects = np.arange(bottom, top, height)
    # Determine parameters corresponding to intersections
    x_parameters = (x_intersects - offset[0]) / direction[0]
    y_parameters = (y_intersects - offset[1]) / direction[1]
    # Return sorted, distinct parameters, including start and and.
    return tuple(np.sort(np.unique(np.concatenate([(0, 1),
                                                   x_parameters,
                                                   y_parameters]))))


def segmentize_by_tiles(line):
    """ Return generator of line tuples. """
    direction, offset = get_direction_and_offset(line)
    parameters = parameterize_intersects(direction=direction,
                                         offset=offset,
                                         cellsize=(1000, 1250))
    for i in range(len(parameters) - 1):
        result = (tuple(offset + direction * parameters[i]),
                  tuple(offset + direction * parameters[i + 1]))
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
    """ Return lines, points, values tuple of numpy arrays. """
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
    direction, offset = get_direction_and_offset(segment.GetPoints())
    parameters = parameterize_intersects(direction=direction,
                                         offset=offset,
                                         cellsize=cellsize)
    offset_array, direction_array = map(np.array, (offset, direction))
    ends = offset_array + direction_array * np.array([parameters]).T
    lines = np.array([ends[:-1].T, ends[1:].T]).transpose(2, 0, 1)
    points = lines.mean(1)

    # Get values
    values = get_values(dataset, points)
    return lines, points, values


def pixelize_range(segment, distance):
    """
    Return lines, points, values tuple of numpy arrays.

    Like pixelize, but searches perpendicular at most distance for a maximum.

    V = original vectors
    U = unit vectors
    P = perpendicular unit vectors
    R = range lines
    """
    # Get data
    lines, points, values = pixelize(segment)
    # Calculate perpendicular lines
    V = lines[:, 1] - lines[:, 0]
    U = V / np.sqrt((V ** 2).sum(1)).reshape(-1, 1)
    P = U[:, ::-1] * np.array([[1, -1]])
    rangelines = np.array([points - P * distance,
                           points + P * distance]).transpose(1, 0, 2)
    rangevalues = []
    for rangeline in rangelines:
        for pseg in segmentize(vector.line2geometry(rangeline)):
            l, p, v = pixelize(pseg)
            rangevalues.append(v.max())

    return lines, points, rangevalues


class AbstractWriter(object):
    """ Base class for common writer methods. """
    def __init__(self, path, **kwargs):
        self.path = path
        for key, value in kwargs.items():
            setattr(self, key, value)

    def __enter__(self):
        """ Creates or replaces the target shapefile. """
        driver = ogr.GetDriverByName(b'ESRI Shapefile')
        if os.path.exists(self.path):
            driver.DeleteDataSource(str(self.path))
        self.dataset = driver.CreateDataSource(str(self.path))
        return self

    def __exit__(self, type, value, traceback):
        """ Close dataset. """
        self.layer = None
        self.dataset = None

    def _count(self, dataset):
        """ Return amount of updates expected for progress indicator. """
        count = 0
        for layer in dataset:
            for feature in layer:
                for segment in segmentize(feature.geometry()):
                    count += 1
            layer.ResetReading()
        return count

    def _add_layer(self, layer):
        """ Add empty copy of layer. """
        # Create layer
        self.layer = self.dataset.CreateLayer(layer.GetName())
        # Copy field definitions
        layer_definition = layer.GetLayerDefn()
        for i in range(layer_definition.GetFieldCount()):
            self.layer.CreateField(layer_definition.GetFieldDefn(i))

    def _pixelize(self, segment):
        """ Pixelize according to mode. """
        if self.mode == MODE_FIXED:
            return pixelize(segment)
        if self.mode == MODE_RANGE:
            return pixelize_range(segment, self.distance)
        if self.mode == MODE_CHANGE:
            raise NotImplementedError


class CoordinateWriter(AbstractWriter):
    """ Writes a shapefile with height in z coordinate. """
    def _convert(self, source_geometry):
        """
        Return converted geometry.
        """
        target_geometry = ogr.Geometry(ogr.wkbLineString)
        for i, segment in enumerate(segmentize(source_geometry)):
            lines, points, values = self._pixelize(segment)

            # Add first point of the first line if this is the first segment
            if i == 0:
                target_geometry.AddPoint(float(lines[0, 0, 0]),
                                         float(lines[0, 0, 1]),
                                         float(values[0]))

            # Add the rest of the points (x, y) and values (z)
            for (x, y), z in zip(points, values):
                target_geometry.AddPoint(float(x), float(y), float(z))

            self.indicator.update()

        # Add the last point of the last line of the last segment
        target_geometry.AddPoint(float(lines[-1, 1, 0]),
                                 float(lines[-1, 1, 1]),
                                 float(values[-1]))
        return target_geometry

    def _add_feature(self, feature):
        """ Add converted feature. """
        # Create feature
        layer_definition = self.layer.GetLayerDefn()
        new_feature = ogr.Feature(layer_definition)
        # Copy attributes
        for key, value in feature.items().items():
            new_feature[key] = value
        # Set geometry and add to layer
        geometry = self._convert(source_geometry=feature.geometry())
        new_feature.SetGeometry(geometry)
        self.layer.CreateFeature(new_feature)

    def add(self, path, **kwargs):
        """ Convert dataset at path. """
        dataset = ogr.Open(path)
        self.indicator = progress.Indicator(self._count(dataset))
        for layer in dataset:
            self._add_layer(layer)
            for feature in layer:
                self._add_feature(feature)
        dataset = None


class AttributeWriter(AbstractWriter):
    """ Writes a shapefile with height in z attribute. """
    def _convert(self, source_geometry):
        """
        Return generator of (geometry, height) tuples.
        """
        for i, segment in enumerate(segmentize(source_geometry)):
            lines, points, values = self._pixelize(segment)
            for line, value in zip(lines, values):
                yield vector.line2geometry(line), str(value)
            self.indicator.update()

    def _add_fields(self):
        """ Create extra fields. """
        for name, kind in ((str(self.elevation_attribute), ogr.OFTReal),
                           (str(self.feature_id_attribute), ogr.OFTInteger)):
            definition = ogr.FieldDefn(name, kind)
            self.layer.CreateField(definition)

    def _add_feature(self, feature_id, feature):
        """ Add converted features. """
        layer_definition = self.layer.GetLayerDefn()
        generator = self._convert(source_geometry=feature.geometry())
        for geometry, elevation in generator:
            # Create feature
            new_feature = ogr.Feature(layer_definition)
            # Copy attributes
            for key, value in feature.items().items():
                new_feature[key] = value
            # Add special attributes
            new_feature[str(self.elevation_attribute)] = elevation
            new_feature[str(self.feature_id_attribute)] = feature_id
            # Set geometry and add to layer
            new_feature.SetGeometry(geometry)
            self.layer.CreateFeature(new_feature)

    def add(self, path):
        """ Convert dataset at path. """
        dataset = ogr.Open(path)
        self.indicator = progress.Indicator(self._count(dataset))
        for layer in dataset:
            self._add_layer(layer)
            self._add_fields()
            for feature_id, feature in enumerate(layer):
                self._add_feature(feature_id=feature_id, feature=feature)
        dataset = None


def addheight(source_path, target_path, overwrite,
              layout, mode, distance,
              elevation_attribute, feature_id_attribute):
    """
    Take linestrings from source and create target with height added.

    Source and target are both shapefiles.
    """
    if os.path.exists(target_path) and not overwrite:
        print("'{}' already exists. Use --overwrite.".format(target_path))
        return 1

    Writer = CoordinateWriter if layout == LAYOUT_POINT else AttributeWriter
    with Writer(target_path,
                mode=mode,
                distance=distance,
                elevation_attribute=elevation_attribute,
                feature_id_attribute=feature_id_attribute) as writer:
        writer.add(source_path)
    return 0


def main():
    """ Call addheight() with commandline args. """
    addheight(**vars(get_parser().parse_args()))


cache = {}  # Contains leafno's and the index

if __name__ == '__main__':
    exit(main())
