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

from matplotlib.backends import backend_agg
from matplotlib import figure
from osgeo import ogr
from PIL import Image


def get_args():
    """ Return arguments dictionary. """
    parser = argparse.ArgumentParser(
        description='Convert shapefiles with Z components.',
    )
    parser.add_argument('source_path',
                        metavar='SOURCE',
                        help=('Path to source shapefile.'))
    parser.add_argument('target_path',
                        metavar='TARGET',
                        help=('Path to target file.'))
    parser.add_argument('-of', '--output-format',
                        metavar='FORMAT',
                        choices=['inp', 'img'],
                        default='inp',
                        help=("Input file 'inp' or image 'img'"))
    return vars(parser.parse_args())


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

    def add_feature(self, feature):
        """ Add feature as nodes and links. """
        geometry = feature.geometry()
        nodes = geometry.GetPoints()
        # Add nodes and links up to the last node
        for i in range(len(nodes) - 1):
            self._write_node(nodes[i])
            self._write_link()
        # Add last node, link already covered.
        self._write_node(nodes[-1])


class ImageWriter(object):
    """ Writer for images. """

    def __init__(self, path):
        self.count = 0
        self.path = path

    def __enter__(self):
        return self

    def add_feature(self, feature):
        """ Currently saves every feature in a separate image. """
        # Get data
        geometry = feature.geometry()
        x, y, z = zip(*geometry.GetPoints())
        # Determine distance along line
        l = [0]
        for i in range(len(z) - 1):
            l.append(l[-1] + math.sqrt(
                (x[i + 1] - x[i])**2 + (y[i + 1] - y[i])**2,
            ))
        # Plot in matplotlib
        label = '\n'.join([': '.join(str(v) for v in item)
                           for item in feature.items().items()])
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

    def __exit__(self, type, value, traceback):
        pass


def convert(source_path, target_path, output_format):
    """ Convert shapefile to inp file."""

    source_dataset = ogr.Open(source_path)

    writers = dict(inp=InputFileWriter, img=ImageWriter)
    with writers[output_format](target_path) as writer:
        for source_layer in source_dataset:
            for source_feature in source_layer:
                writer.add_feature(source_feature)


def main():
    args = get_args()
    convert(**args)


if __name__ == '__main__':
    exit(main())
