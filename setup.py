#! /usr/bin/env python

# System imports
from distutils.core import *
from distutils      import sysconfig
from pip import __file__ as pip_loc
from os import path
import os
# Third-party modules - we depend on numpy for everything
import numpy

# Obtain the numpy include directory.  This logic works across numpy versions.
try:
    numpy_include = numpy.get_include()
except AttributeError:
    numpy_include = numpy.get_numpy_include()

install_to = path.join(path.split(path.split(pip_loc)[0])[0],
                   'cosmo_pol', 'templates')
# Create the symlinks to lut
try:
    os.symlink('/store/msrad/utils/anaconda3/envs/radop_dw/cosmo_pol/cosmo_pol/lookup/lut_tmatrix/','./build')
except:
    print('Could not create symlink to lookup tables, maybe they are already present')
# interp1 extension module
_doppler_c = Extension("_interp1_c",
                   ["./cosmo_pol/scatter/doppler_c.i","./cosmo_pol/scatter/doppler_c.c"],
                   include_dirs = [numpy_include],
                   )
_interpolation_c_c = Extension("_interpolation_c_c",
                   ["./cosmo_pol/interpolation/interpolation_c.i","./cosmo_pol/interpolation/interpolation_c.c"],
                   include_dirs = [numpy_include],
                  )
# ezrange setup
setup(  name        = "cosmo_pol",
        description = "Polarimetric radar operator for the COSMO NWP model",
        version     = "1.0",
        url='https://github.com/wolfidan/cosmo_pol',
        author='Daniel Wolfensberger - LTE EPFL',
        author_email='daniel.wolfensberger@epfl.ch',
        license='GPL-3.0',
        packages=['cosmo_pol','cosmo_pol/interpolation','cosmo_pol/radar','cosmo_pol/utilities','cosmo_pol/constants','cosmo_pol/lookup','cosmo_pol/scatter','cosmo_pol/hydrometeors',
		'cosmo_pol/config'],
        package_data   = {'cosmo_pol/interpolation' : ['*.o','*.i','*.c','*.so'], 'cosmo_pol/scatter' : ['*.o','*.i','*.c','*.so']},
        data_files = [(install_to, ["LICENSE"])],
        include_package_data=True,
        install_requires=[
          'pyproj',
          'numpy',
          'scipy',
	  'pynio',
        ],
        zip_safe=False,
        ext_modules = [_doppler_c,_interpolation_c_c ]
        )


