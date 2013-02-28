#!/usr/bin/python
# -*- coding: utf-8 -*-
# (c) Nelen & Schuurmans.  GPL licensed, see LICENSE.rst.

from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import
from __future__ import division

import unittest

from threedilib.modeling import vector


class TestVectorOps(unittest.TestCase):
    ''' Testing Test functions '''

    def setUp(self):
        pass

    def test_line2geometry(self):
        line = [(0, 0), (1, 1)]
        geometry = vector.line2geometry(line)
        self.assertEqual(line, geometry.GetPoints())
