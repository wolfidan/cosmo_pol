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
A_G_=0.142 # from D = a * x**b
B_G_=0.314 # from D = a * x**b
ALPHA_G_=86.89371
BETA_G_=0.268325
NU_G_=1./3.
MU_G_=1.0
X_MIN_G = 1E-09
X_MAX_G = 5E-04

B_G = 1./B_G_ # from x= a * D**b
A_G = A_G_**(-1/B_G_) # from x= a * D**b
BETA_G=BETA_G_/B_G_
ALPHA_G=ALPHA_G_*A_G_**(-BETA_G_/B_G_)
NU_G=NU_G_/B_G_
MU_G=(MU_G_+1)/B_G_ - 1
D_MIN_G=0.2
D_MAX_G=15

LAMBDA_FACTOR_G=1./A_G*spe.gamma((MU_G+1)/NU_G)/spe.gamma((MU_G+B_G+1)/NU_G)

# Get correct units
A_G=A_G*1000**(-B_G)
ALPHA_G=ALPHA_G*1000**(-BETA_G)

# Compute constant integration factors (saves time since evaluation
# of gamma function is done only once)

VEL_FACTOR_G=spe.gamma((MU_G+BETA_G+1)/NU_G)
NTOT_FACTOR_G=spe.gamma((MU_G+1)/NU_G)

###############################################################################

# Snow PSD (snowCRYSTALuli2)
A_S_=2.4 # from D = a * x**b
B_S_=0.455 # from D = a * x**b
ALPHA_S_=4.2
BETA_S_=0.092
NU_S_=0.5
MU_S_=0.0
X_MIN_S = 1E-10
X_MAX_S = 2E-05

B_S = 1./B_S_ # from x= a * D**b
A_S = A_S_**(-1/B_S_)
BETA_S=BETA_S_/B_S_
ALPHA_S=ALPHA_S_*A_S_**(-BETA_S_/B_S_)
NU_S=NU_S_/B_S_
MU_S=(MU_S_+1)/B_S_ - 1
D_MIN_S=0.2
D_MAX_S=20

LAMBDA_FACTOR_S=1./A_S*spe.gamma((MU_S+1)/NU_S)/spe.gamma((MU_S+B_S+1)/NU_S)

# Get correct units
A_S = A_S*1000**(-B_S)
ALPHA_S = ALPHA_S*1000**(-BETA_S)

# Compute constant integration factors (saves time since evaluation
# of gamma function is done only once)
VEL_FACTOR_S=spe.gamma((MU_S+BETA_S+1)/NU_S)
NTOT_FACTOR_S=spe.gamma((MU_S+1)/NU_S)

###############################################################################
# Rain DSD (rainULI)
A_R_ = 0.124 # from D = a * x**b
B_R_ = 1./3. # from D = a * x**b
ALPHA_R_ = 114.0137
BETA_R_ = 0.23437
NU_R_ = 1./3.
MU_R_ = 0.0
X_MIN_R = 2.6E-10
X_MAX_R = 3E-06

B_R =  1./B_R_ # from x= a * D**b
A_R = A_R_**(-1/B_R_) # frMU x= a * D**b
BETA_R = BETA_R_/B_R_
ALPHA_R = ALPHA_R_*A_R_**(-BETA_R_/B_R_)
NU_R = NU_R_/B_R_
MU_R = (MU_R_+1)/B_R_ - 1
D_MIN_R = 0.2
D_MAX_R = 8


C_1 = 9.65 # Parameters from the V-D relation
C_2 = 10.3
C_3 = 600.*1000.**(-1)
TAU_1 = 4 # Parameters from the MU-D relation for sedimentation
TAU_2  = 1
D_EQ = 1.1 # mm

LAMBDA_FACTOR_R = 1./A_R*spe.gamma((MU_R+1)/NU_R)/spe.gamma((MU_R+B_R+1)/NU_R)

# Get correct units
A_R = A_R * 1000**(-B_R)
ALPHA_R = ALPHA_R * 1000**(-BETA_R)

# Compute constant integration factors (saves time since evaluation
# of gamma function is done only once)
VEL_FACTOR_R = spe.gamma((MU_R+BETA_R+1)/NU_R)
NTOT_FACTOR_R = spe.gamma((MU_R+1)/NU_R)

###############################################################################
# Hail DSD (hailULItest)
A_H_=0.1366 # from D = a * x**b
B_H_=1./3. # from D = a * x**b
ALPHA_H_=39.3
BETA_H_=1./6.
NU_H_=1./3.
MU_H_=1.0
X_MIN_H = 2.6E-09
X_MAX_H = 5E-04

B_H = 1./B_H_ # from x= a * D**b
A_H = A_H_**(-1/B_H_) # from x= a * D**b
BETA_H=BETA_H_/B_H_
ALPHA_H=ALPHA_H_*A_H_**(-BETA_H_/B_H_)
NU_H=NU_H_/B_H_
MU_H=(MU_H_+1)/B_H_ - 1
D_MIN_H=0.2
D_MAX_H=15

LAMBDA_FACTOR_H=1./A_H*spe.gamma((MU_H+1)/NU_H)/spe.gamma((MU_H+B_H+1)/NU_H)

# Get correct units
A_H = A_H * 1000**(-B_H)
ALPHA_H = ALPHA_H * 1000**(-BETA_H)

# Compute constant integration factors (saves time since evaluation
# of gamma function is done only once)
VEL_FACTOR_H = spe.gamma((MU_H+BETA_H+1)/NU_H)
NTOT_FACTOR_H = spe.gamma((MU_H+1)/NU_H)
###############################################################################

# Cloud ice DSD
A_I_ = 0.124 # from D = a * x**b
B_I_ = 0.302 # from D = a * x**b
ALPHA_I_= 317
BETA_I_ =  0.363
NU_I_ = 1./3.
MU_I_ = 0.0
X_MIN_I = 1E-12
X_MAX_I = 1E-6

B_I = 1./B_I_  # from x= a * D**b
A_I = A_I_**(-1/B_I_)  # from x= a * D**b
BETA_I = BETA_I_/B_I_
ALPHA_I = ALPHA_I_ * A_I_**(-BETA_I_/B_I_)
NU_I = NU_I_/B_I_
MU_I = (MU_I_+1)/B_I_ - 1
D_MIN_I = 0.05
D_MAX_I = 2

LAMBDA_FACTOR_I = 1./A_I*spe.gamma((MU_I + 1) /NU_I)/spe.gamma((MU_I+B_I+1)/NU_I)

# Get correct units
A_I = A_I * 1000**(-B_I)
ALPHA_I = ALPHA_I * 1000**(-BETA_I)

# Compute constant integration factors (saves time since evaluation
# of gamma function is done only once)
VEL_FACTOR_I = spe.gamma((MU_I + BETA_I + 1)/NU_I)
NTOT_FACTOR_I = spe.gamma((MU_I + 1) / NU_I)
###############################################################################





