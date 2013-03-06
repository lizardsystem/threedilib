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
    embedded elevation from an elevation map.

    Target shapefile can have two layouts: A 'point' layout where the
    elevation is stored in the third coordinate of a 3D linstring, and
    a 'line' layout where a separate feature is created in the target
    shapefile for each segment of each feature in the source shapefile,
    with two extra attributes compared to the original shapefile, one
    to store the elevation, and another to store an arbitrary feature
    id referring to the source feature in the source shapefile.

    For the script to work, a configuration variable AHN_PATH must be
    set in threedilib/localconfig.py pointing to the location of the
    elevation map, and a variable INDEX_PATH pointing to the .shp file
    that contains the index to the elevation map.
"""

LAYOUT_POINT = 'point'
LAYOUT_LINE = 'line'

PIXELSIZE = 0.5  # AHN2
STEPSIZE = 0.5  # For looking perpendicular to line.

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
    parser.add_argument('-d', '--distance',
                        metavar='DISTANCE',
                        type=float,
                        default=0,
                        help=('Distance (half-width) to look '
                              'perpendicular to the segments to '
                              'find the highest points on the '
                              'elevation map. Defaults to 0.0.'))
    parser.add_argument('-m', '--modify',
                        action='store_true',
                        help='Change horizontal geometry.')
    parser.add_argument('-a', '--average',
                        metavar='AMOUNT',
                        type=int,
                        default=0,
                        help='Average points and values.')
    parser.add_argument('-l', '--layout',
                        metavar='LAYOUT',
                        choices=[LAYOUT_POINT, LAYOUT_LINE],
                        default=LAYOUT_POINT,
                        help="Target shapefile layout.")
    parser.add_argument('-f', '--feature-id-attribute',
                        metavar='FEATURE_ID_ATTRIBUTE',
                        default='_feat_id',
                        help='Attribute name for the feature id.')
    parser.add_argument('-e', '--elevation-attribute',
                        metavar='ELEVATION_ATTRIBUTE',
                        default='_elevation',
                        help='Attribute name for the elevation.')
    return parser


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


def get_dataset(leafno):
    """ Return gdal_dataset from cache or file. """
    if leafno in cache:
        return cache[leafno]

    if len(cache) > 10:
        for key in cache.keys():
            if SHEET.match(key):
                del cache[key]  # Maybe unnecessary, see top and lsof.

    # Add to cache and return.
    unit = SHEET.match(leafno).group('unit')
    path = os.path.join(config.AHN_PATH, unit, leafno + '.tif')
    dataset = gdal.Open(path)
    cache[leafno] = Dataset(dataset)
    dataset = None
    return cache[leafno]


def get_carpet(magic_line, distance, step=None):
    """
    Return MxNx2 numpy array.

    It contains the midpoints of the MagicLine, but perpendicularly
    repeated along the normals to the segments of the MagicLine, up to
    distance, with step.
    """
    if step is None or step == 0:
        length = 2
    else:
        # Length must be uneven, and no less than 2 * distance / step + 1
        length = 2 * np.round(0.5 + distance / step) + 1
    offsets_1d = np.mgrid[-distance:distance:length * 1j]
    vectors = vector.normalize(vector.rotate(magic_line.vectors, 270))
    offsets_2d = vectors.reshape(-1, 1, 2) * offsets_1d.reshape(1, -1, 1)
    return offsets_2d + magic_line.centers.reshape(-1, 1, 2)


def get_leafnos(magic_line, distance):
    """ Return the leafnos for the outermost lines of the carpet. """
    # Convert magic line to carpet to linestring around carpet
    pixel_line = magic_line.pixelize(size=PIXELSIZE, endsonly=True)
    carpet_points = get_carpet(magic_line=pixel_line,
                               distance=distance)
    # Create multipoint containing outermost lines
    line1, line2 = carpet_points[:, np.array([0, -1])].transpose(1, 0, 2)
    linestring = vector.line2geometry(np.vstack([line1, line2[::-1]]))
    # Query the index with it
    index = get_index()
    index.SetSpatialFilter(linestring)
    return [feature[b'BLADNR'] for feature in index]


def paste_values(points, values, leafno):
    """ Paste values of evelation pixels at points. """
    dataset = get_dataset(leafno)
    xmin, ymin, xmax, ymax = dataset.get_extent()
    cellsize = dataset.get_cellsize()
    origin = dataset.get_origin()

    # Determine which points are outside leaf's extent.
    # '=' added for the corner where the index origin is.
    index = np.logical_and(np.logical_and(points[..., 0] >= xmin,
                                          points[..., 0] < xmax),
                           np.logical_and(points[..., 1] > ymin,
                                          points[..., 1] <= ymax))
    # Determine indices for these points
    indices = tuple(np.uint64(
        (points[index] - origin) / cellsize,
    ).transpose())[::-1]

    # Assign data for these points to corresponding values.
    values[index] = dataset.data[indices]


def average_result(amount, lines, centers, values):
    """
    Return dictionary of numpy arrays.

    Points and values are averaged in groups of amount, but lines are
    converted per group to a line from the start point of the first line
    in the group to the end point of the last line in the group.
    """
    # Determine the size needed to fit an integer multiple of amount
    oldsize = values.size
    newsize = int(np.ceil(values.size / amount) * amount)
    # Determine lines
    ma_lines = np.ma.array(np.empty((newsize, 2, 2)), mask=True)
    ma_lines[:oldsize] = lines
    ma_lines[oldsize:] = lines[-1]  # Repeat last line
    result_lines = np.array([
        ma_lines.reshape(-1, amount, 2, 2)[:, 0, 0],
        ma_lines.reshape(-1, amount, 2, 2)[:, -1, 1],
    ]).transpose(1, 0, 2)
    # Calculate points and values by averaging
    ma_centers = np.ma.array(np.empty((newsize, 2)), mask=True)
    ma_centers[:oldsize] = centers
    ma_values = np.ma.array(np.empty(newsize), mask=True)
    ma_values[:oldsize] = values
    return dict(lines=result_lines,
                values=ma_values.reshape(-1, amount).mean(1),
                centers=ma_centers.reshape(-1, amount, 2).mean(1))


class Dataset(object):
    def __init__(self, dataset):
        """ Initialize from gdal dataset. """
        self.geotransform = dataset.GetGeoTransform()
        self.size = dataset.RasterXSize, dataset.RasterYSize
        self.data = dataset.ReadAsArray()

        # Check for holes in the data
        nodatavalue = dataset.GetRasterBand(1).GetNoDataValue()
        if nodatavalue in self.data:
            raise ValueError('Dataset {} contains nodatavalues!'.format(
                dataset.GetFileList()[0]
            ))

    def get_extent(self):
        """ Return tuple of xmin, ymin, xmax, ymax. """
        return (self.geotransform[0],
                self.geotransform[3] + self.size[1] * self.geotransform[5],
                self.geotransform[0] + self.size[0] * self.geotransform[1],
                self.geotransform[3])

    def get_cellsize(self):
        """ Return numpy array. """
        return np.array([[self.geotransform[1], self.geotransform[5]]])

    def get_origin(self):
        """ Return numpy array. """
        return np.array([[self.geotransform[0], self.geotransform[3]]])


class BaseWriter(object):
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
        """
        Return amount of updates expected for progress indicator.
        """
        count = 0
        for layer in dataset:
            indicator = progress.Indicator(layer.GetFeatureCount())
            for feature in layer:
                geometry = feature.geometry()
                geometry_type = geometry.GetGeometryType()
                if geometry_type == ogr.wkbLineString:
                    wkb_line_strings = [geometry]
                elif geometry_type == ogr.wkbMultiLineString:
                    wkb_line_strings = [line for line in geometry]
                for wkb_line_string in wkb_line_strings:
                    magic_line = vector.MagicLine(wkb_line_string.GetPoints())
                    count += len(get_leafnos(magic_line=magic_line,
                                             distance=self.distance))
                indicator.update()
            layer.ResetReading()
        return count

    def _calculate(self, wkb_line_string):
        """ Return lines, points, values tuple of numpy arrays. """
        # Determine the leafnos
        magic_line = vector.MagicLine(wkb_line_string.GetPoints())
        leafnos = get_leafnos(magic_line=magic_line, distance=self.distance)
        # Determine the point and values carpets
        pixel_line = magic_line.pixelize(size=PIXELSIZE)
        carpet_points = get_carpet(
            magic_line=pixel_line,
            distance=self.distance,
            step=STEPSIZE,
        )
        carpet_values = np.ma.array(
            np.empty(carpet_points.shape[:2]),
            mask=True,
        )
        # Get the values into the carpet per leafno
        for leafno in leafnos:
            paste_values(carpet_points, carpet_values, leafno)
            self.indicator.update()
        if carpet_values.mask.any():
            raise ValueError('Masked values remaining after filling!')

        # Return lines, centers, values
        result = dict(lines=pixel_line.lines,
                      centers=pixel_line.centers,
                      values=carpet_values.data.max(1))

        if self.average:
            return average_result(amount=self.average, **result)
        else:
            return result

    def _add_layer(self, layer):
        """ Add empty copy of layer. """
        # Create layer
        self.layer = self.dataset.CreateLayer(layer.GetName())
        # Copy field definitions
        layer_definition = layer.GetLayerDefn()
        for i in range(layer_definition.GetFieldCount()):
            self.layer.CreateField(layer_definition.GetFieldDefn(i))


class CoordinateWriter(BaseWriter):
    """ Writes a shapefile with height in z coordinate. """
    def _convert_wkb_line_string(self, source_wkb_line_string):
        """ Return a wkb line string. """
        result = self._calculate(wkb_line_string=source_wkb_line_string)
        target_wkb_line_string = ogr.Geometry(ogr.wkbLineString)

        # Add the first point of the first line
        (x, y), z = result['lines'][0, 0], result['values'][0]
        target_wkb_line_string.AddPoint(float(x), float(y), float(z))

        # Add the centers (x, y) and values (z)
        for (x, y), z in zip(result['centers'], result['values']):
            target_wkb_line_string.AddPoint(float(x), float(y), float(z))

        # Add the last point of the last line
        (x, y), z = result['lines'][-1, 1], result['values'][-1]
        target_wkb_line_string.AddPoint(float(x), float(y), float(z))
        return target_wkb_line_string

    def _convert(self, source_geometry):
        """
        Return converted linestring or multiline.
        """
        geometry_type = source_geometry.GetGeometryType()
        if geometry_type == ogr.wkbLineString:
            return self._convert_wkb_line_string(source_geometry)
        if geometry_type == ogr.wkbMultiLineString:
            target_geometry = ogr.Geometry(source_geometry.GetGeometryType())
            for source_wkb_line_string in source_geometry:
                target_geometry.AddGeometry(
                    self._convert_wkb_line_string(source_wkb_line_string),
                )
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
        print('Estimating work...')
        count = self._count(dataset)
        print('Converting:...')
        self.indicator = progress.Indicator(count)
        for layer in dataset:
            self._add_layer(layer)
            for feature in layer:
                self._add_feature(feature)
        dataset = None


class AttributeWriter(BaseWriter):
    """ Writes a shapefile with height in z attribute. """
    def _convert(self, source_geometry):
        """
        Return generator of (geometry, height) tuples.
        """
        geometry_type = source_geometry.GetGeometryType()
        if geometry_type == ogr.wkbLineString:
            source_wkb_line_strings = [source_geometry]
        elif geometry_type == ogr.wkbMultiLineString:
            source_wkb_line_strings = [line for line in source_geometry]
        for source_wkb_line_string in source_wkb_line_strings:
            result = self._calculate(wkb_line_string=source_wkb_line_string)
            for line, value in zip(result['lines'], result['values']):
                yield vector.line2geometry(line), str(value)

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


def addheight(source_path, target_path, overwrite, distance, modify, average,
              layout, elevation_attribute, feature_id_attribute):
    """
    Take linestrings from source and create target with height added.

    Source and target are both shapefiles.
    """
    if os.path.exists(target_path) and not overwrite:
        print("'{}' already exists. Use --overwrite.".format(target_path))
        return 1

    Writer = CoordinateWriter if layout == LAYOUT_POINT else AttributeWriter
    with Writer(target_path,
                distance=distance,
                modify=modify,
                average=average,
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
