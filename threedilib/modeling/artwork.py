# (c) Nelen & Schuurmans.  GPL licensed, see LICENSE.rst.
# -*- coding: utf-8 -*-

from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import
from __future__ import division

from matplotlib.backends import backend_agg
from matplotlib import figure
from PIL import Image


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
