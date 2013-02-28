#!/usr/bin/python
# -*- coding: utf-8 -*-
# (c) Nelen & Schuurmans.  GPL licensed, see LICENSE.rst.

from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import
from __future__ import division

import unittest

from threedilib.pointcloud import sync


class TestSync(unittest.TestCase):
    """ Test sync utility. """
    def test_smoke(self):
        sql = ''.join(sync.sync('select'))
        self.assertEqual(type(sql), unicode)
