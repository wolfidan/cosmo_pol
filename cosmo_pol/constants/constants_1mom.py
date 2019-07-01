# -*- coding: utf-8 -*-

"""constants_1mom.py: defines a certain number of constants, for hydrometeors
in the one-moment (operational) microphysical scheme"""

__author__ = "Daniel Wolfensberger"
__copyright__ = "Copyright 2017, COSMO_POL"
__credits__ = ["Daniel Wolfensberger"]
__license__ = "GPL"
__version__ = "1.0"
__maintainer__ = "Daniel Wolfensberger"
__email__ = "daniel.wolfensberger@epfl.ch"

# Global imports
import numpy as np
np.seterr(divide='ignore') # Disable divide by zero error
import scipy.special as spe

# Local imports
from . import global_constants as constants

'''
The list of constants are the following
N0: Intercept parameter in the exponential or gamma PSD [mm-1 m-3]
MU: Shape parameter in the PSD [-]
BM: exponent in the mass diameter relation M = AM * D^BM [-]
AM: intercept in the mass diameter relation M = AM * D^BM [mm^-BM kg]
BV: exponent in the velocity diameter relation V = AV * D^BV [-]
AV: intercept in the velocity diameter relation V = AV * D^BV [mm^-BM m s-1]
D_MIN: minimum considered diameter in the PSD
D_MAX: maximum considered diameter in the PSD
LAMBDA_FACTOR: constant factor used in the integration of the mass to get
    lambda of the PSD, is computed from other parameters, should not be
    changed. Note that it is precomputed to save computation time
VEL_FACTOR: constant factor used in the integration of the fall velocity
    over the PSD, is computed from other parameters, should not be
    changed
NTOT_FACTOR: constant factor used in the integration of the PSD,
     is computed from other parameters, should not be
    changed

'''


# Graupel
N0_G = 4*1E3
BM_G = 3.1
BV_G = 0.89
AM_G = 169.6*(1000**-BM_G)
AV_G = 442.0*(1000**-BV_G)
MU_G = 0.0
D_MIN_G = 0.2
D_MAX_G = 15
LAMBDA_FACTOR_G = AM_G*N0_G*spe.gamma(BM_G + 1)
VEL_FACTOR_G = spe.gamma(MU_G+BV_G + 1)
NTOT_FACTOR_G = spe.gamma(MU_G + 1)

# Snow
BM_S = 2.
BV_S = 0.25
AM_S = 0.038 * (1000 ** -BM_S)
AV_S = 4.9 * (1000 ** -BV_S)
MU_S = 0.0
D_MIN_S = 0.2
D_MAX_S = 20
LAMBDA_FACTOR_S = spe.gamma(BM_S + 1)
VEL_FACTOR_S = spe.gamma(MU_S + BV_S + 1)
NTOT_FACTOR_S = spe.gamma(MU_S + 1)

# Rain
RAIN_FACTOR = 0.1
MU_R = 0.5
N00_r = 8E6 / (1000 ** (1+MU_R)) * (0.01) ** (-MU_R)
# see http://www.cosmo-model.org/content/model/releases/histories/cosmo_4.21.htm
N0_R = RAIN_FACTOR*N00_r * np.exp(3.2 * MU_R)
BM_R = 3.
BV_R=0.5
AM_R = np.pi/6.*constants.RHO_W
AV_R = 130* (1000**-BV_R) # mm^0.5*s-1
D_MIN_R=0.1
D_MAX_R=8
LAMBDA_FACTOR_R = AM_R * N0_R * spe.gamma(1. + BM_R + MU_R)
VEL_FACTOR_R = spe.gamma(MU_R + BV_R + 1)
NTOT_FACTOR_R = spe.gamma(MU_R + 1)

# ICE crystals
BM_I = 3   # Exponent of m-D relation
AM_I = 130 * (1000 ** -BM_I)
MU_I = 0.0
D_MIN_I = 0.05
D_MAX_I = 2
# These are actually taken from two-moments scheme, because in the one-
# moment scheme, ice crystals have no fall speeds, which is problematic to
# get the Doppler spectrum
AV_I = 0.9655930341942476
BV_I = 1.2019867549668874
LAMBDA_FACTOR_I = spe.gamma(BM_I + 1)
NTOT_FACTOR_I = spe.gamma(MU_I + 1)
VEL_FACTOR_I = spe.gamma(MU_I + BV_I + 1)

# This is the phi function for the double-normalized PSD for moments 2,3
# Taken from Field et al. (2005)
PHI_23_I = lambda x : (490.6*np.exp(-20.78 * x) +
                       17.46 * x ** (0.6357) * np.exp(-3.290 * x))


