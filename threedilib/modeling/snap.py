# (c) Nelen & Schuurmans.  GPL licensed, see LICENSE.rst.
# -*- coding: utf-8 -*-

from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import
from __future__ import division

import argparse
import os

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
        self.axes.axis('equal')

    def add(self, geometries, style):
        """ Add some geometries to plot. """
        for geometry in geometries:
            x, y = zip(*geometry.GetPoints())
            self.axes.plot(x, y, style)

    def get(self):
        """ Return image object. """
        buf, size = self.axes.figure.canvas.print_to_buffer()
        return Image.fromstring('RGBA', size, buf)


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
    geometry.AddPoint_2D(*map(float, point))
    return geometry


def line2geometry(line):
    """ Return geometry. """
    geometry = ogr.Geometry(ogr.wkbLineString)
    for point in line:
        geometry.AddPoint_2D(*map(float, point))
    return geometry


def get_endpoints(feature):
    """ Return generator of end points. """
    for geometry in get_feature_geometries(feature):
        indices = 0, geometry.GetPointCount() - 1
        points = map(geometry.GetPoint_2D, indices)
        for point in points:
            yield point2geometry(point)


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


def get_projections(point, geometry):
    """
    Return list of points.

    Those points are returned where the projection of point on the
    geometry segment is on the segment itself.
    """
    source = np.array(point.GetPoint_2D())
    nodes = np.array(geometry.GetPoints())
    start = nodes[:-1]
    end = nodes[1:]

    # Determine projections
    v1 = end - start
    v1_length = np.sqrt((v1 ** 2).sum(1)).reshape(-1, 1)
    v2 = source - start
    v1n = v1 / v1_length

    # Perform the dot product and the projections.
    dotprod = (v1n * v2).sum(1).reshape(-1, 1)
    projections = start + dotprod * v1n

    # Determine which targets are on the segment
    online = np.logical_and(
        np.greater_equal(dotprod, 0),
        np.less(dotprod, v1_length),
    ).reshape(-1)

    return projections[online].tolist()


def get_snap_geometry(feature, layer, distance, artwork=None):
    """ Return generator of snapping geometries. """
    # Artwork
    if artwork:
        art = Artwork()
        art.add(get_feature_geometries(feature), 'b')

    snaplines = []
    for endpoint in get_endpoints(feature):
        # Artwork
        if artwork:
            art.add([endpoint.Buffer(distance).GetBoundary()], '--k')
            art.add([endpoint], 'ob')

        # Determine nodes and projections per endpoint
        nodes = []
        projections = []
        endpoint_buffered = endpoint.Buffer(INTERSECT_TOLERANCE)
        layer.SetSpatialFilter(endpoint.Buffer(distance))
        for target_geometry in get_layer_geometries(layer):
            if endpoint_buffered.Intersects(target_geometry):
                # This is probably points own feature, leave it out.
                continue

            # Artwork
            if artwork:
                art.add([target_geometry], 'k')

            nodes.extend(target_geometry.GetPoints())
            projections.extend(get_projections(endpoint, target_geometry))

            # Artwork
            if artwork:
                art.add(map(point2geometry, nodes), 'og')
                art.add(map(point2geometry, projections), 'oc')

        # Determine possible snapline for this endpoint
        target_points = nodes + projections
        if target_points:
            interpolator = interpolate.NearestNDInterpolator(
                np.array(target_points), np.array(target_points),
            )
            nearest_point = interpolator(endpoint.GetPoints())[0]
            endpoint_snapline = (endpoint.GetPoint_2D(),
                                 nearest_point.tolist())
            snaplines.append(endpoint_snapline)

            # Artwork
            if artwork:
                art.add([line2geometry(endpoint_snapline)], ':g')

    if snaplines:
        # Determine shortest snapline
        snaparray = np.array(snaplines)
        snapvectors = snaparray[:, 1, :] - snaparray[:, 0, :]
        snaplengths = np.sqrt((snapvectors ** 2).sum(1))
        index = np.where(np.equal(snaplengths, snaplengths.min()))
        snapgeometry = line2geometry(snaparray[index][0])

        # Artwork
        if artwork:
            art.add([snapgeometry], 'm')
            art.get().save('artwork{:04.0f}.png'.format(artwork))

        return snapgeometry


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

    # Do the work
    indicator = progress.Indicator(loose_layer.GetFeatureCount())
    artwork = 0
    for feature in loose_layer:
        # artwork += 1
        target_geometry = get_snap_geometry(feature=feature,
                                            layer=network_layer,
                                            distance=distance,
                                            artwork=artwork)
        if target_geometry is not None:
            target_feature = ogr.Feature(target_layer_definition)
            target_feature.SetGeometry(target_geometry)
            target_layer.CreateFeature(target_feature)
        indicator.update()

    # Properly close datasets
    network_dataset = None
    loose_dataset = None
    target_dataset = None


def main():
    args = get_args()
    snap(**args)


if __name__ == '__main__':
    exit(main())
