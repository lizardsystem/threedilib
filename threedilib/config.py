# -*- coding: utf-8 -*-

from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import
from __future__ import division

import os

# Directories
BUILDOUT_DIR = os.path.join(
    os.path.dirname(os.path.realpath(__file__)),
    '..',
)
INDEX_PATH = os.path.join(BUILDOUT_DIR, 'var', 'index', 'ahn2_05_int_index.shp')
AHN_PATH = os.path.join(BUILDOUT_DIR, 'var', 'ahn')

# Import local settings
try:
    from threedilib.localconfig import *
except ImportError:
    pass
