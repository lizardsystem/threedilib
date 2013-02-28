#!/usr/bin/python
# -*- coding: utf-8 -*-
# (c) Nelen & Schuurmans.  GPL licensed, see LICENSE.rst.

from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import
from __future__ import division

import os
import shutil
import tempfile
import unittest

from threedilib.modeling import convert

TESTDATA_DIR = os.path.join(
    os.path.dirname(os.path.realpath(__file__)), 'testdata',
)


class TestConvert(unittest.TestCase):
    """ Integration tests. """

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

    def test_convert_to_inp(self):
        source_path = os.path.join(TESTDATA_DIR, 'polyline_z.shp')
        target_path = os.path.join(self.temp_dir, 'polyline_z.inp')
        output_format = 'inp'
        convert.convert(source_path=source_path,
                        target_path=target_path,
                        output_format=output_format)
        self.assertTrue(os.path.exists(target_path))

    def test_convert_to_png(self):
        source_path = os.path.join(TESTDATA_DIR, 'polyline_z.shp')
        target_path = os.path.join(self.temp_dir, 'polyline_z.png')
        output_format = 'img'
        convert.convert(source_path=source_path,
                        target_path=target_path,
                        output_format=output_format)
        self.assertEqual(
            os.listdir(self.temp_dir),
            ['polyline_z1.png', 'polyline_z0.png'],
        )

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

class TestParser(unittest.TestCase):
    """ Test parser. """
    def test_parser(self):
        parser = convert.get_parser()
        args = vars(parser.parse_args(['foo', 'bar']))
        self.assertEqual(args['output_format'], 'inp')

