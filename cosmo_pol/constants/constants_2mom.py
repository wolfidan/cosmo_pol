# -*- coding: utf-8 -*-
"""
Created on Thu Feb  4 13:46:57 2016

@author: wolfensb
"""
import scipy.special as spe
import constants_1mom as c1

#,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,
# PSD Parameters
# Note that all the constants with _ (underscore) at the end are massic, whereas
# the ones without _ are diameter-based (see doc)

# Graupel PSD (graupelhail2test4)
AM_G_=0.142 # from D = a * x**b
BM_G_=0.314 # from D = a * x**b
AV_G_=86.89371
BV_G_=0.268325
NU_G_=1./3.
MU_G_=1.0
X_MIN_G = 1E-09
X_MAX_G = 5E-04

BM_G = 1./BM_G_ # from x= a * D**b
AM_G = AM_G_**(-1/BM_G_) # from x= a * D**b
BV_G=BV_G_/BM_G_
AV_G=AV_G_*AM_G_**(-BV_G_/BM_G_)
NU_G=NU_G_/BM_G_
MU_G=(MU_G_+1)/BM_G_ - 1
D_MIN_G=0.2
D_MAX_G=15

LAMBDA_FACTOR_G=1./AM_G*spe.gamma((MU_G+1)/NU_G)/spe.gamma((MU_G+BM_G+1)/NU_G)

# Get correct units
AM_G=AM_G*1000**(-BM_G)
AV_G=AV_G*1000**(-BV_G)

# Compute constant integration factors (saves time since evaluation
# of gamma function is done only once)

VEL_FACTOR_G=spe.gamma((MU_G+BV_G+1)/NU_G)
NTOT_FACTOR_G=spe.gamma((MU_G+1)/NU_G)

###############################################################################

# Snow PSD (snowCRYSTALuli2)
AM_S_=2.4 # from D = a * x**b
BM_S_=0.455 # from D = a * x**b
AV_S_=4.2
BV_S_=0.092
NU_S_=0.5
MU_S_=0.0
X_MIN_S = 1E-10
X_MAX_S = 2E-05

BM_S = 1./BM_S_ # from x= a * D**b
AM_S = AM_S_**(-1/BM_S_)
BV_S=BV_S_/BM_S_
AV_S=AV_S_*AM_S_**(-BV_S_/BM_S_)
NU_S=NU_S_/BM_S_
MU_S=(MU_S_+1)/BM_S_ - 1
D_MIN_S=0.2
D_MAX_S=20

LAMBDA_FACTOR_S=1./AM_S*spe.gamma((MU_S+1)/NU_S)/spe.gamma((MU_S+BM_S+1)/NU_S)

# Get correct units
AM_S = AM_S*1000**(-BM_S)
AV_S = AV_S*1000**(-BV_S)

# Compute constant integration factors (saves time since evaluation
# of gamma function is done only once)
VEL_FACTOR_S=spe.gamma((MU_S+BV_S+1)/NU_S)
NTOT_FACTOR_S=spe.gamma((MU_S+1)/NU_S)

###############################################################################
# Rain DSD (rainULI)
AM_R_ = 0.124 # from D = a * x**b
BM_R_ = 1./3. # from D = a * x**b
AV_R_ = 114.0137
BV_R_ = 0.23437
NU_R_ = 1./3.
MU_R_ = 0.0
X_MIN_R = 2.6E-10
X_MAX_R = 3E-06

BM_R =  1./BM_R_ # from x= a * D**b
AM_R = AM_R_**(-1/BM_R_) # frMU x= a * D**b
BV_R = BV_R_/BM_R_
AV_R = AV_R_*AM_R_**(-BV_R_/BM_R_)
NU_R = NU_R_/BM_R_
MU_R = (MU_R_+1)/BM_R_ - 1
D_MIN_R = 0.2
D_MAX_R = 8


C_1 = 9.65 # Parameters from the V-D relation
C_2 = 10.3
C_3 = 600.*1000.**(-1)
TAU_1 = 4 # Parameters from the MU-D relation for sedimentation
TAU_2  = 1
D_EQ = 1.1 # mm

LAMBDA_FACTOR_R = 1./AM_R*spe.gamma((MU_R+1)/NU_R)/spe.gamma((MU_R+BM_R+1)/NU_R)

# Get correct units
AM_R = AM_R * 1000**(-BM_R)
AV_R = AV_R * 1000**(-BV_R)

# Compute constant integration factors (saves time since evaluation
# of gamma function is done only once)
VEL_FACTOR_R = spe.gamma((MU_R+BV_R+1)/NU_R)
NTOT_FACTOR_R = spe.gamma((MU_R+1)/NU_R)

###############################################################################
# Hail DSD (hailULItest)
AM_H_=0.1366 # from D = a * x**b
BM_H_=1./3. # from D = a * x**b
AV_H_=39.3
BV_H_=1./6.
NU_H_=1./3.
MU_H_=1.0
X_MIN_H = 2.6E-09
X_MAX_H = 5E-04

BM_H = 1./BM_H_ # from x= a * D**b
AM_H = AM_H_**(-1/BM_H_) # from x= a * D**b
BV_H=BV_H_/BM_H_
AV_H=AV_H_*AM_H_**(-BV_H_/BM_H_)
NU_H=NU_H_/BM_H_
MU_H=(MU_H_+1)/BM_H_ - 1
D_MIN_H=0.2
D_MAX_H=15

LAMBDA_FACTOR_H=1./AM_H*spe.gamma((MU_H+1)/NU_H)/spe.gamma((MU_H+BM_H+1)/NU_H)

# Get correct units
AM_H = AM_H * 1000**(-BM_H)
AV_H = AV_H * 1000**(-BV_H)

# Compute constant integration factors (saves time since evaluation
# of gamma function is done only once)
VEL_FACTOR_H = spe.gamma((MU_H+BV_H+1)/NU_H)
NTOT_FACTOR_H = spe.gamma((MU_H+1)/NU_H)
###############################################################################

# Cloud ice DSD
AM_I_ = 0.124 # from D = a * x**b
BM_I_ = 0.302 # from D = a * x**b
AV_I_= 317
BV_I_ =  0.363
NU_I_ = 1./3.
MU_I_ = 0.0
X_MIN_I = 1E-12
X_MAX_I = 1E-6

BM_I = 1./BM_I_  # from x= a * D**b
AM_I = AM_I_**(-1/BM_I_)  # from x= a * D**b
BV_I = BV_I_/BM_I_
AV_I = AV_I_ * AM_I_**(-BV_I_/BM_I_)
NU_I = NU_I_/BM_I_
MU_I = (MU_I_+1)/BM_I_ - 1
D_MIN_I = 0.05
D_MAX_I = 2

LAMBDA_FACTOR_I = 1./AM_I*spe.gamma((MU_I + 1) /NU_I)/spe.gamma((MU_I+BM_I+1)/NU_I)

# Get correct units
AM_I = AM_I * 1000**(-BM_I)
AV_I = AV_I * 1000**(-BV_I)

# Compute constant integration factors (saves time since evaluation
# of gamma function is done only once)
VEL_FACTOR_I = spe.gamma((MU_I + BV_I + 1)/NU_I)
NTOT_FACTOR_I = spe.gamma((MU_I + 1) / NU_I)
###############################################################################





