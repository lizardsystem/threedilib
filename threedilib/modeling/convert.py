# (c) Nelen & Schuurmans.  GPL licensed, see LICENSE.rst.
# -*- coding: utf-8 -*-
"""
Convert shapefiles with z coordinates. Choose from the following formats:
'inp' to create an inp file, 'img' to create an image with a plot of the
feature, or 'shp' to output a shapefile with the average height of a
feature stored in an extra attribute.
"""

from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import
from __future__ import division

import argparse
import math
import os
import shutil
import tempfile

from matplotlib.backends import backend_agg
from matplotlib import figure
from osgeo import gdal
from osgeo import ogr
from PIL import Image

ogr.UseExceptions()


def get_parser():
    """ Return argument parser. """
    parser = argparse.ArgumentParser(
        description=__doc__,
    )
    parser.add_argument('source_path',
                        metavar='SOURCE',
                        help=('Path to source shapefile.'))
    parser.add_argument('target_path',
                        metavar='TARGET',
                        help=('Path to target file.'))
    parser.add_argument('-of', '--output-format',
                        metavar='FORMAT',
                        choices=['inp', 'img', 'shp'],
                        default='shp',
                        help=("Path to output."))
    return parser


class InputFileWriter(object):
    """ Writer for input files. """
    def __init__(self, path):
        """
        Init the counters and tmpdirs
        """
        self.path = path
        self.node_count = 0
        self.link_count = 0

    def __enter__(self):
        """ Setup tempfiles. """
        self.temp_directory = tempfile.mkdtemp()
        self.node_file = open(
            os.path.join(self.temp_directory, 'nodes'), 'a+',
        )
        self.link_file = open(
            os.path.join(self.temp_directory, 'links'), 'a+',
        )
        return self

    def __exit__(self, type, value, traceback):
        """ Write 'inputfile' at path. """
        with open(self.path, 'w') as input_file:
            self.node_file.seek(0)
            input_file.write(self.node_file.read())
            input_file.write('-1\n')
            self.link_file.seek(0)
            input_file.write(self.link_file.read())
        self.node_file.close()
        self.link_file.close()
        shutil.rmtree(self.temp_directory)

    def _write_node(self, node):
        """ Write a node. """
        self.node_count += 1
        self.node_file.write('{} {} {} {}\n'.format(
            self.node_count, node[0], node[1], -node[2]  # Depth, not height!
        ))

    def _write_link(self):
        """ Write a link between previous node and next node."""
        self.link_count += 1
        self.link_file.write('{} {} {}\n'.format(
            self.link_count, self.node_count, self.node_count + 1,
        ))

    def _add_wkb_line_string(self, wkb_line_string):
        """ Add linestring as nodes and links. """
        nodes = [wkb_line_string.GetPoint(i)
                 for i in range(wkb_line_string.GetPointCount())]
        # Add nodes and links up to the last node
        for i in range(len(nodes) - 1):
            self._write_node(nodes[i])
            self._write_link()
        # Add last node, link already covered.
        self._write_node(nodes[-1])

    def add_feature(self, feature):
        """ Add feature as nodes and links. """
        geometry = feature.geometry()
        geometry_type = geometry.GetGeometryType()
        if geometry_type == ogr.wkbLineString25D:
            self._add_wkb_line_string(geometry)
        elif geometry_type == ogr.wkbMultiLineString25D:
            for wkb_line_string in geometry:
                self._add_wkb_line_string(wkb_line_string)


class ImageWriter(object):
    """ Writer for images. """

    def __init__(self, path):
        self.count = 0
        self.path = path

    def __enter__(self):
        return self

    def _add_wkb_line_string(self, wkb_line_string, label):
        """ Plot linestring as separate image. """
        # Get data
        x, y, z = zip(*[wkb_line_string.GetPoint(i)
                        for i in range(wkb_line_string.GetPointCount())])
        # Determine distance along line
        l = [0]
        for i in range(len(z) - 1):
            l.append(l[-1] + math.sqrt(
                (x[i + 1] - x[i]) ** 2 + (y[i + 1] - y[i]) ** 2,
            ))
        # Plot in matplotlib
        fig = figure.Figure()
        axes = fig.add_axes([0.1, 0.1, 0.8, 0.8])
        axes.plot(l, z, label=label)
        axes.legend(loc='best', frameon=False)
        # Write to image
        backend_agg.FigureCanvasAgg(fig)
        buf, size = fig.canvas.print_to_buffer()
        image = Image.fromstring('RGBA', size, buf)
        root, ext = os.path.splitext(self.path)
        image.save(root + '{:00.0f}'.format(self.count) + ext)
        self.count += 1

    def add_feature(self, feature):
        """ Currently saves every feature in a separate image. """
        # Plotlabel
        label = '\n'.join([': '.join(str(v) for v in item)
                           for item in feature.items().items()])
        # Plot according to geometry type
        geometry = feature.geometry()
        geometry_type = geometry.GetGeometryType()
        if geometry_type == ogr.wkbLineString25D:
            self._add_wkb_line_string(geometry, label=label)
        elif geometry_type == ogr.wkbMultiLineString25D:
            for wkb_line_string in geometry:
                self._add_wkb_line_string(wkb_line_string, label=label)

    def __exit__(self, type, value, traceback):
        pass


class ShapefileWriter(object):
    """ Writer for shapefiles. """
    ATTRIBUTE = b'kruinhoogt'

    def __init__(self, path):
        self.count = 0
        self.path = path
        self.datasource = None
        self.layer = None

    def __enter__(self):
        return self

    def create_datasource(self, feature):
        """ Create a datasource based on feature. """
        root, ext = os.path.splitext(os.path.basename(self.path))
        driver = ogr.GetDriverByName(b'ESRI Shapefile')
        datasource = driver.CreateDataSource(self.path)
        layer = datasource.CreateLayer(root)
        for i in range(feature.GetFieldCount()):
            layer.CreateField(feature.GetFieldDefnRef(i))
        field_defn = ogr.FieldDefn(self.ATTRIBUTE, ogr.OFTReal)
        layer.CreateField(field_defn)

        self.datasource = datasource
        self.layer = layer

    def add_feature(self, feature):
        """ Currently saves every feature in a separate image. """
        if self.layer is None:
            self.create_datasource(feature)
        layer_defn = self.layer.GetLayerDefn()

        # elevation
        geometry = feature.geometry().Clone()
        geometry_type = geometry.GetGeometryType()
        if geometry_type == ogr.wkbLineString25D:
            elevation = min([p[2] for p in geometry.GetPoints()])
        else:
            # multilinestring
            elevation = min([p[2]
                             for g in geometry
                             for p in g.GetPoints()])
        geometry.FlattenTo2D()

        new_feature = ogr.Feature(layer_defn)
        new_feature.SetGeometry(geometry)
        for k, v in feature.items().items():
            new_feature[k] = v
        new_feature[self.ATTRIBUTE] = elevation
        self.layer.CreateFeature(new_feature)

    def __exit__(self, type, value, traceback):
        pass


def convert(source_path, target_path, output_format):
    """ Convert shapefile to inp file."""

    source_dataset = ogr.Open(str(source_path))

    writers = dict(
        inp=InputFileWriter,
        img=ImageWriter,
        shp=ShapefileWriter,
    )
    with writers[output_format](target_path) as writer:
        for source_layer in source_dataset:
            total = source_layer.GetFeatureCount()
            for count, source_feature in enumerate(source_layer, 1):
                writer.add_feature(source_feature)
                gdal.TermProgress_nocb(count / total)


def main():
    """ Call convert() with commandline args. """
    convert(**vars(get_parser().parse_args()))


if __name__ == '__main__':
    exit(main())
