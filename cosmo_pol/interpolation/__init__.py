# -*- coding: utf-8 -*-

"""__init__.py: Performs relevant imports for the interpolation submodule"""

__author__ = "Daniel Wolfensberger"
__copyright__ = "Copyright 2017, COSMO_POL"
__credits__ = ["Daniel Wolfensberger"]
__license__ = "GPL"
__version__ = "1.0"
__maintainer__ = "Daniel Wolfensberger"
__email__ = "daniel.wolfensberger@epfl.ch"


from .radial import Radial
from .atm_refraction import get_GPM_refraction, get_radar_refraction
from .melting import melting
from .quadrature import get_points_and_weights
from .interpolation_c import get_all_radar_pts