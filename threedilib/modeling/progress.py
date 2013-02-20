#!/usr/bin/python
# -*- coding: utf-8 -*-

from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import
from __future__ import division


test = """0...10...20...30...40...50...60...70...80...90...100 - done."""

import sys
import time


class Indicator(object):
    """ Progress indicator. """

    INDICATOR = []
    for text in map(str, range(00, 100, 10)):
        INDICATOR.append(text)
        INDICATOR.extend(3 * '.')
    INDICATOR.append('100 - done.\n')

    def __init__(self, length):
        """ Set the expected length of the job. """
        self.end = length - 1
        self.position = 0  # Indicator position
        self.value = 0  # Jobs counter
        self._update()  # Display the first item

    def _update(self):
        """ Update indicator one position. """
        sys.stdout.write(self.INDICATOR[self.position])
        sys.stdout.flush()
        self.position += 1

    def update(self):
        """ Update progress according to value. """
        fraction = self.value / self.end
        position = fraction * len(self.INDICATOR)
        while self.position < position:
            self._update()
        self.value += 1


def main():
    length = 5
    p = Indicator(length=length)
    for i in range(length):
        s = 0.1
        time.sleep(s)
        p.update()
    length = 133
    p = Indicator(length=length)
    for i in range(length):
        s = 0.01
        time.sleep(s)
        p.update()


if __name__ == '__main__':
    exit(main())
