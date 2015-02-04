# -*- coding: utf-8 -*-

from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import
from __future__ import division

import logging
import math

from osgeo import gdal
from osgeo import osr

import netCDF4
import numpy as np

from threedilib.subgrid import datasets
from threedilib.subgrid import utils

gdal.UseExceptions()
osr.UseExceptions()

logger = logging.getLogger(__name__)

CORRECTIONS = {'EPGS:28992': 'EPSG:28992'}
DRIVER = gdal.GetDriverByName(b'mem')


class Data(object):
    def __init__(self, nc_path):
        """ Load and init from netCDF. """
        # load data
        with netCDF4.Dataset(nc_path) as nc:
            v = nc.variables
            e = v['projected_coordinate_system'].EPSG_code
            s = slice(0, int(nc.nFlowElem2d))
            fex, fey = v['FlowElemContour_x'][s], v['FlowElemContour_y'][s]
            fcx, fcy = v['FlowElem_xcc'][s], v['FlowElem_ycc'][s]

        x1, y1, x2, y2 = fex.min(1), fey.min(1), fex.max(1), fey.max(1)
        X1, Y1, X2, Y2 = x1.min(), y1.min(), x2.max(), y2.max()
        epsg = CORRECTIONS.get(e, e)

        # attributes
        self.projection = osr.GetUserInputAsWKT(str(epsg))
        self.no_data_value = s.stop

        # convenient arrays
        self.widths = x2 - x1
        self.heights = y2 - y1
        self.centers = np.array([fcx, fcy]).T

        # indexes
        unique = np.sort(np.unique(self.widths))
        self.index = [np.where(self.widths == w) for w in unique]

        # stretch extent
        i = self.index[-1][0][0]
        w = self.widths[i]
        h = self.heights[i]
        self.X1 = x1[i] + w * math.floor((X1 - x1[i]) / w)
        self.Y1 = y1[i] + h * math.floor((Y1 - y1[i]) / h)
        self.X2 = x2[i] + w * math.ceil((X2 - x2[i]) / w)
        self.Y2 = y2[i] + h * math.ceil((Y2 - y2[i]) / h)

    def get_geo_transform(self):
        """ Return GeoTransform based on smallest quads. """
        i = self.index[0][0][0]
        g = self.X1, self.widths[i], 0, self.Y2, 0, -self.heights[i]
        return utils.GeoTransform(g)

    def get_array(self):
        """ Return array based on smallest quads. """
        i = self.index[0][0][0]
        w = int(round((self.X2 - self.X1) / self.widths[i]))
        h = int(round((self.Y2 - self.Y1) / self.heights[i]))
        return self.no_data_value * np.ones((h, w), dtype='u4')


def get_dataset_kwargs(nc_path):
    """ Return a gdaldataset containing the quad positions. """
    # load
    data = Data(nc_path)
    array = data.get_array()
    projection = data.projection
    no_data_value = data.no_data_value
    geo_transform = data.get_geo_transform()

    # loop quads grouped by width
    msg = 'Add {} wide quads to dataset.'
    for index1 in data.index:
        # determine zoom factor, unzoomed array and geo_transform
        first = index1[0][0]
        zoom = int(round(data.widths[first] / geo_transform[1]))
        unz_shape = tuple(int(round(n / zoom)) for n in array.shape)
        unz_array = no_data_value * np.ones(unz_shape, dtype='u4')
        unz_geo_transform = geo_transform.scale(zoom)

        # fill-in the quads in the unzoomed array
        unz_centers = data.centers[index1]
        unz_indices = unz_geo_transform.get_indices(unz_centers)
        unz_array[unz_indices] = index1[0]

        # zoom in using broadcasting
        unz_array.shape = (unz_shape + (1, 1))
        multiplier = np.ones((1, 1, zoom, zoom), dtype='u4')
        zoomed_unz_array = (unz_array * multiplier).transpose(0, 2, 1, 3)
        zoomed_unz_array = zoomed_unz_array.reshape(array.shape)

        # paste in to target array
        index2 = zoomed_unz_array != no_data_value
        array[index2] = zoomed_unz_array[index2]

        logger.debug(msg.format(data.widths[first]))

    kwargs = {'array': array[np.newaxis, ...],
              'projection': projection,
              'geo_transform': geo_transform,
              'no_data_value': no_data_value}
    return kwargs


def get_dataset(nc_path):
    """ Creates a copy to be safe against segfaults. """
    kwargs = get_dataset_kwargs(nc_path)
    with datasets.Dataset(**kwargs) as dataset:
        return DRIVER.CreateCopy('', dataset)
