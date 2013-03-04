# (c) Nelen & Schuurmans.  GPL licensed, see LICENSE.rst.
# -*- coding: utf-8 -*-

from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import
from __future__ import division

from osgeo import ogr

import numpy as np


def line2geometry(line):
    """ Return geometry. """
    geometry = ogr.Geometry(ogr.wkbLineString)
    for point in line:
        geometry.AddPoint_2D(*map(float, point))
    return geometry


def magnitude(vectors):
    """ Return magnitudes. """
    return np.sqrt((vectors ** 2).sum(1))


def normalize(vectors):
    """ Return unit vectors. """
    return vectors / magnitude(vectors).reshape(-1, 1)


def rotate(vectors, degrees):
    """ Return vectors rotated by degrees. """
    return np.vstack([
        +np.cos(np.radians(degrees)) * vectors[:, 0] +
        -np.sin(np.radians(degrees)) * vectors[:, 1],
        +np.sin(np.radians(degrees)) * vectors[:, 0] +
        +np.cos(np.radians(degrees)) * vectors[:, 1],
    ]).transpose()


def array(vectors, distance, step):
    """ Return MxNx2x2 vector array. """
    positions = np.mgrid[-distance:distance:(1 + 2 * distance / step) * 1j]
    import ipdb; ipdb.set_trace() 
    result = positions.reshape(1, -1) * normalize(rotate(vectors, 90))
    import ipdb; ipdb.set_trace()
    return result
