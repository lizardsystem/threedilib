threedilib
==========================================

Installation
-------------

Standard setup::

    $ python bootstrap.py
    $ bin/buildout

Some of the tools need separate configuration. The easiest way to do
this is to create a file localconfig.py next to threedilib/config.py,
with overrides for variables defined in threedilib/config.py.

To install with pip in a virtualenv::

    $ virtualenv threedilib --system-site-packages
    $ . threedilib/bin/activate
    $ pip install git+ssh://git@github.com/lizardsystem/threedilib.git --index=http://packages.lizardsystem.nl


Schematisation tools
--------------------
The following commands are available::

    $ bin/modeling_addheight
    $ bin/modeling_convert
    $ bin/modeling_snap

Each command can be run with --help as argument for usage instructions.


Remains from history
--------------------
This library was originally intended for preprocessing of 3Di calculation
results to high resolution images, but since then a realtime solution
was developed leaving most of the old functionality stored here
redundant. This includes the subdirectory threedi and the following
modules::

    nc.py
    read_3di.py
    threedi.py
    threedi_win.py
