# -*- coding: utf-8 -*-
"""
Created on Thu Feb  4 13:46:57 2016

@author: wolfensb
"""
import scipy.special as spe
from cosmo_pol.constants.constants import constants
import numpy as np

#,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,
# PSD Parameters

###############################################################################
# Graupel PSD
N0_G=4*1E3 # mm-1 m-3
B_G=3.1 # Exponent of m-D relation
BETA_G=0.89 # Exponent of v-D relation
A_G=169.6*(1000**-B_G) # m_g = am_g * D**3.1 (with m_g in kg and D in m)
ALPHA_G=442.0*(1000**-BETA_G) # mm^0.89*s-1
MU_G=0.0
D_MIN_G=0.2
D_MAX_G=15

# Compute constant integration factors (saves time since evaluation
# of gamma function is done only once)
LAMBDA_FACTOR_G=A_G*N0_G*spe.gamma(B_G+1)
VEL_FACTOR_G=spe.gamma(MU_G+BETA_G+1)
NTOT_FACTOR_G=spe.gamma(MU_G+1)

###############################################################################
# Snow PSD
B_S = 2 # Exponent of m-D relation
BETA_S = 0.25 # Exponent of v-D relation
A_S=0.038*(1000**-B_S)  # kg * mm^-2
ALPHA_S=4.9*(1000**-BETA_S) # mm^0.5*s-1
MU_S=0.0
D_MIN_S=0.2
D_MAX_S=20

# Compute constant integration factors (saves time since evaluation
# of gamma function is done only once)
LAMBDA_FACTOR_S=spe.gamma(B_S+1)
VEL_FACTOR_S=spe.gamma(MU_S+BETA_S+1)
NTOT_FACTOR_S=spe.gamma(MU_S+1)

###############################################################################
# Rain DSD
RAIN_FACTOR=0.1
MU_R=0.5
N00_r=8E6/(1000**(1+MU_R))*(0.01)**(-MU_R) # m^(-3)*mm^(-1-mu)
N0_R=RAIN_FACTOR*N00_r*np.exp(3.2*MU_R)
B_R=3. # Exponent of m-D relation
BETA_R=0.5  # Exponent of v-D relation
A_R = np.pi/6.*constants.RHO_W
ALPHA_R = 130* (1000**-BETA_R) # mm^0.5*s-1
D_MIN_R=0.1
D_MAX_R=8

# Compute constant integration factors (saves time since evaluation
# of gamma function is done only once)
LAMBDA_FACTOR_R=A_R*N0_R*spe.gamma(1.+B_R+MU_R)
VEL_FACTOR_R=spe.gamma(MU_R+BETA_R+1)
NTOT_FACTOR_R=spe.gamma(MU_R+1)

###############################################################################
# ICE Particles
B_I = 3   # Exponent of m-D relation
A_I = 130 * (1000**-B_I)
MU_I = 0.0
D_MIN_I = 0.05
D_MAX_I = 2
# From two-moments scheme, see Seifert and Beheng
ALPHA_I = 0.9655930341942476
BETA_I = 1.2019867549668874

# Compute constant integration factors (saves time since evaluation
# of gamma function is done only once)
LAMBDA_FACTOR_I = spe.gamma(B_I+1)
NTOT_FACTOR_I = spe.gamma(MU_I+1)
VEL_FACTOR_I=spe.gamma(MU_I+BETA_I+1)

# This is the phi function for the double-normalized PSD for moments 2,3
# Taken from Field et al. (2005)


PHI_23_I = lambda x : 490.6*np.exp(-20.78 * x) + 17.46*x**(0.6357) * np.exp(-3.290 * x)


###############################################################################

# This part below is unused
# Vectorize the constant parameters (for rain, snow and graupel) in order to avoir recreating these arrays for every pixel
ALPHA_ALL=np.asarray([ALPHA_R, ALPHA_S, ALPHA_G, 0],dtype='float32')
BETA_ALL=np.asarray([BETA_R, BETA_S, BETA_G, 0],dtype='float32')
MU_ALL=np.asarray([MU_R, MU_S, MU_G, MU_I],dtype='float32')



