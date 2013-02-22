# (c) Nelen & Schuurmans.  GPL licensed, see LICENSE.rst.
# -*- coding: utf-8 -*-

from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import
from __future__ import division

import argparse
import collections

from osgeo import ogr

from threedilib.modeling import progress


def get_args():
    """ Return arguments dictionary. """
    parser = argparse.ArgumentParser(description='No description yet.')
    parser.add_argument('source',
                        metavar='SOURCE',
                        help=('Shapefile with height attributes.'))
    parser.add_argument('target',
                        metavar='TARGET',
                        help=('Target .inp file.'))
    parser.add_argument('-d', '--depth',
                        action='store_true',
                        default=False,
                        help=('If true, negative height is used.'))
    parser.add_argument('-a', '--attribute',
                        default=b'Type',
                        help=('Feature attribute to identify lines.'))
    parser.add_argument('-i', '--image',
                        help=('If supplied, save image.'))
    return vars(parser.parse_args())


def main():
    """ Convert shapefile to inp file."""
    args = get_args()
    name_attribute = args['attribute']
    sign = -1 if args['depth'] else 1
    source_path = args['source']
    target_path = args['target']

    lines = collections.defaultdict(dict)
    source_dataset = ogr.Open(source_path)
    source_layer = source_dataset[0]
    total = source_layer.GetFeatureCount()
    print('Going to process {} feature(s).'.format(total))
    indicator = progress.Indicator(total)
    for feature in source_layer:
        name = feature[name_attribute]
        start, end = feature.geometry().GetPoints()  # Crash with segments > 1
        indicator.update()

    sign, target_path, lines, name  # Pyflakes

if __name__ == '__main__':
    exit(main())
