# (c) Nelen & Schuurmans.  GPL licensed, see LICENSE.rst.
# -*- coding: utf-8 -*-

from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import
from __future__ import division

import contextlib
import logging
import time

import numpy as np
from osgeo import ogr
from osgeo import osr

ogr.UseExceptions()
osr.UseExceptions()

logger = logging.getLogger(__name__)

CONFIG_SET = set(['store.json', 'group.json', 'result.json'])


def get_inverse(a, b, c, d):
    """ Return inverse for a 2 x 2 matrix with elements (a, b), (c, d). """
    D = 1 / (a * d - b * c)
    return d * D, -b * D,  -c * D,  a * D


class GeoTransform(tuple):
    """ Convenience wrapper adding all sorts of handy methods. """
    def scale(self, scale=2):
        """
        return scaled geo transform.

        :param scale: Multiplication factor for the pixel size.

        Adjust the second, third, fifth and sixth elements of the geo
        transform so that the extent of the respective image is multiplied
        by scale.
        """
        p, a, b, q, c, d = self
        s = scale
        return self.__class__([p, a * s, b * s, q, c * s, d * s])

    def get_indices(self, points):
        """
        Return coordinate pixel indices as a numpy tuple.
        """
        # inverse transformation
        p, a, b, q, c, d = self
        e, f, g, h = get_inverse(a, b, c, d)

        # calculate
        x, y = points.transpose()
        return (np.uint64(g * (x - p) + h * (y - q)),
                np.uint64(e * (x - p) + f * (y - q)))


@contextlib.contextmanager
def timer(name):
        start = time.time()
        yield
        logger.debug('{}: {:0.3f} s.'.format(name, time.time() - start))
