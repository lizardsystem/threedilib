#!/usr/bin/python
# -*- coding: utf-8 -*-

from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import
from __future__ import division

import os
import re
import sys
import argparse

from threedilib import config

SHEET = re.compile('^i(?P<unit>[0-9]{2}[a-z])[a-z][0-9]_[0-9]{2}$')
SUBUNIT = re.compile('^(?P<subunit>[0-9]{2}[a-z][a-z][0-9])\.tar\.(gz|xz)$')


def get_args():
    parser = argparse.ArgumentParser(
        description='Usage: command query | psql | command update | psql')
    parser.add_argument('dml',
                        choices=['create', 'select', 'update', 'drop'],
                        help=(''))
    return vars(parser.parse_args())


class Dml(object):
    """
    Method container.
    """
    def create(self):
        """ Return create sql. """
        table = config.POINTCLOUD_TABLE
        template = """
            CREATE TABLE "{table}" (
                "leafno" varchar(9) NOT NULL
            )
            ;
        """
        yield template.format(table=table)

    def drop(self):
        """ Return delete sql. """
        table = config.POINTCLOUD_TABLE
        template = """
            DROP TABLE "{table}"
            ;
        """
        yield template.format(table=table)

    def select(self):
        """ Return select sql. """
        table = config.POINTCLOUD_TABLE
        template = """
            SELECT
                leafno
            FROM
                "{table}"
            ;
        """
        yield template.format(table=table)

    def update(self):
        """ Return update sql. """
        table = config.POINTCLOUD_TABLE
        # Check what's in database
        in_database = []
        for line in sys.stdin:

            match = SHEET.match(line.strip())
            if match:
                in_database.append(match.string)

        # Check what's in POINTCLOUD_DIR
        in_pointcloud_dir = []
        for name in os.listdir(config.POINTCLOUD_DIR):
            match = SUBUNIT.match(name)
            if match:
                for i in range(1, 26):
                    in_pointcloud_dir.append(
                        'i' + match.group('subunit') + '_{:02.0f}'.format(i),
                    )

        # Yield delete sql
        template_delete = """DELETE FROM "{table}" WHERE leafno='{leafno}';"""
        for leafno in set(in_database) - set(in_pointcloud_dir):
            yield template_delete.format(table=table, leafno=leafno)

        # Yield insert sql
        template_insert = """INSERT INTO "{table}" VALUES ('{leafno}');"""
        for leafno in set(in_pointcloud_dir) - set(in_database):
            yield template_insert.format(table=table, leafno=leafno)


def main():
    args = get_args()
    sql = getattr(Dml(), args['dml'])()  # Weird code.
    for line in sql:
        sys.stdout.write(line)


if __name__ == '__main__':
    exit(main())
