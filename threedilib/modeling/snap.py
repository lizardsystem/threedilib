# (c) Nelen & Schuurmans.  GPL licensed, see LICENSE.rst.
# -*- coding: utf-8 -*-

from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import
from __future__ import division

import argparse
import math
import os
import shutil
import tempfile

import numpy as np

from matplotlib.backends import backend_agg
from matplotlib import figure
from osgeo import ogr
from PIL import Image
from scipy import interpolate

from threedilib.modeling import progress


INTERSECT_TOLERANCE = 0.01


class Artwork(object):
    def __init__(self):
        """ Create axes. """
        fig = figure.Figure()
        backend_agg.FigureCanvasAgg(fig)
        self.axes = fig.add_axes([0.1, 0.1, 0.8, 0.8])

    def add(self, geometries, color):
        """ Add some geometries to plot. """
        for geometry in geometries:
            x, y = zip(*geometry.GetPoints())
            self.axes.plot(x, y, color)

    def show(self):
        """ Show to image. """
        buf, size = self.axes.figure.canvas.print_to_buffer()
        image = Image.fromstring('RGBA', size, buf)
        image.show()


def get_args():
    """ Return arguments dictionary. """
    parser = argparse.ArgumentParser(
        description='Convert shapefiles with Z components.',
    )
    parser.add_argument('network_path',
                        metavar='NETWORK',
                        help=('Path to network shapefile.'))
    parser.add_argument('loose_path',
                        metavar='LOOSE',
                        help=('Path to loose lines shapefile.'))
    parser.add_argument('target_path',
                        metavar='TARGET',
                        help=('Path to target shapefile.'))
    parser.add_argument('distance',
                        metavar='DISTANCE',
                        type=float,
                        help=('Distance to seek snap points.'))
    return vars(parser.parse_args())


def point2geometry(point):
    """ Return geometry. """
    geometry = ogr.Geometry(ogr.wkbPoint)
    geometry.AddPoint_2D(*point)
    return geometry


def line2geometry(line):
    """ Return geometry. """
    geometry = ogr.Geometry(ogr.wkbLineString)
    for point in line:
        geometry.AddPoint_2D(*point)
    return geometry


def get_endpoints(geometry):
    """ Return tuple of geometries. """
    indices = 0, geometry.GetPointCount() - 1
    points = map(geometry.GetPoint_2D, indices)
    return map(point2geometry, points)


def get_feature_geometries(feature):
    """ Return geometry generator. """
    feature_geometry = feature.geometry()
    if feature_geometry.GetGeometryType() in (1, 2, 3):
        yield feature_geometry
    else:
        for geometry in feature_geometry:
            yield geometry


def get_layer_geometries(layer):
    """
    Return geometry generator. 

    Inproper use of this one will definately cause segmentation faults.
    """
    for feature in layer:
        for geometry in get_feature_geometries(feature):
            yield geometry


def get_snap_geometries(geometry, layer, distance):
    """ Return generator of snapping geometries. """
    for point in get_endpoints(geometry):
        # Some visualization
        
        #img = Artwork()
        #img.add(point.Buffer(distance), '--k')
        #img.add(point.Buffer(2), 'g')
        #img.add([geometry], 'r')

        # The real stuff    
        layer.SetSpatialFilter(point.Buffer(distance))
        target_points = []
        for target_feature in layer:
            target_geometry = target_feature.geometry()
            if point.Buffer(INTERSECT_TOLERANCE).Intersects(target_geometry):
                # This is probably points own feature, leave it out.
                continue
            for target_geometry in get_feature_geometries(target_feature):
                target_points.extend(target_geometry.GetPoints())
                #img.add([target_geometry], 'b')
        
        if target_points:
            interpolator = interpolate.NearestNDInterpolator(
                np.array(target_points), np.array(target_points),
            )
            nearest_point = interpolator(point.GetPoints())[0]
            snap_geometry = line2geometry([point.GetPoint_2D(), nearest_point])
            
            #img.add(point2geometry(nearest_point).Buffer(2), 'g')
            #img.add([snap_geometry], 'm')
            
            yield snap_geometry
        #img.show()


def snap(network_path, loose_path, target_path, distance):
    """ Create connections from loose lines to network. """
    # Open input datasets
    network_dataset = ogr.Open(network_path)
    loose_dataset = ogr.Open(loose_path)
    network_layer = network_dataset[0]
    loose_layer = loose_dataset[0]

    # Prepare output dataset
    driver = ogr.GetDriverByName(b'ESRI Shapefile')
    if os.path.exists(target_path):
        driver.DeleteDataSource(str(target_path))
    target_dataset = driver.CreateDataSource(str(target_path))
    target_layer = target_dataset.CreateLayer(b'Snapped objects')
    target_layer_definition = target_layer.GetLayerDefn()

    # Count work
    count = 0
    for geometry in get_layer_geometries(loose_layer):
        count += 1
    indicator = progress.Indicator(count)
    loose_layer.ResetReading()

    # Do the work
    count = 0
    for geometry in get_layer_geometries(loose_layer):
        indicator.update()
        target_geometries = get_snap_geometries(
            geometry=geometry, layer=network_layer, distance=distance,
        )
        for target_geometry in target_geometries:
            target_feature = ogr.Feature(target_layer_definition)
            target_feature.SetGeometry(target_geometry)
            target_layer.CreateFeature(target_feature)

    # Properly close datasets
    network_dataset = None
    loose_dataset = None
    target_dataset = None
        

def main():
    args = get_args()
    snap(**args)


if __name__ == '__main__':
    exit(main())
