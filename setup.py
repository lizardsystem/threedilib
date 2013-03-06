from setuptools import setup

version = '0.7.3'

long_description = '\n\n'.join([
    open('README.rst').read(),
    open('CREDITS.rst').read(),
    open('CHANGES.rst').read(),
    ])

install_requires = [
    'setuptools',
    'lizard_raster',
    'Pillow',
    ],

tests_require = [
    ]

setup(name='threedilib',
      version=version,
      description="A library to work with 3Di",
      long_description=long_description,
      # Get strings from http://www.python.org/pypi?%3Aaction=list_classifiers
      classifiers=[],
      keywords=[],
      author='Jack Ha',
      author_email='jack.ha@nelen-schuurmans.nl',
      url='',
      license='GPL',
      packages=['threedilib'],
      include_package_data=True,
      zip_safe=False,
      install_requires=install_requires,
      tests_require=tests_require,
      extras_require = {'test': tests_require},
      entry_points={
          'console_scripts': [
              'modeling_addheight = threedilib.modeling.addheight:main',
              'modeling_convert = threedilib.modeling.convert:main',
              'modeling_snap = threedilib.modeling.snap:main',
              'pointcloud_sync = threedilib.pointcloud.sync:main',
          ]},
      )
