#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 19 10:38:35 2016

@author: wolfensb
"""

import numpy as np
import scipy.special as spe
import matplotlib.pyplot as plt

from pytmatrix.tmatrix import Scatterer
from pytmatrix.psd import PSDIntegrator, UnnormalizedGammaPSD
from pytmatrix import orientation, radar, tmatrix_aux, refractive
def drop_ar(D_eq):
    if D_eq < 0.7:
        return 1.0;
    elif D_eq < 1.5:
        return 1.173 - 0.5165*D_eq + 0.4698*D_eq**2 - 0.1317*D_eq**3 - \
            8.5e-3*D_eq**4
    else:
        return 1.065 - 6.25e-2*D_eq - 3.99e-3*D_eq**2 + 7.66e-4*D_eq**3 - \
            4.095e-5*D_eq**4 
            
scatterer = Scatterer(wavelength=tmatrix_aux.wl_C, m=refractive.m_w_10C[tmatrix_aux.wl_C])
scatterer.psd_integrator = PSDIntegrator()
scatterer.psd_integrator.axis_ratio_func = lambda D: 1.0/drop_ar(D)
scatterer.psd_integrator.D_max = 8.
scatterer.psd_integrator.geometries = (tmatrix_aux.geom_horiz_back, tmatrix_aux.geom_horiz_forw)
scatterer.or_pdf = orientation.gaussian_pdf(10.0)
scatterer.orient = orientation.orient_averaged_fixed
scatterer.psd_integrator.init_scatter_table(scatterer)
            
EPS = np.finfo(np.float32).eps
A_R_ = 0.124 # from D = a * x**b
B_R_ = 1./3. # from D = a * x**b
ALPHA_R_ = 114.0137
BETA_R_ = 0.23437
NU_R_ = 1./3.
MU_R_ = 1.0
X_MIN_R = 2.6E-10
X_MAX_R = 3E-06
LAMBDA_FACTOR_R_=spe.gamma((MU_R_+1)/NU_R_)/spe.gamma((MU_R_+2)/NU_R_)
NTOT_FACTOR_R_=spe.gamma((MU_R_+1)/NU_R_)

B_R =  1./B_R_ # from x= a * D**b
A_R = A_R_**(-1/B_R_) # frMU x= a * D**b
BETA_R=BETA_R_/B_R_
ALPHA_R=ALPHA_R_*A_R_**(-BETA_R_/B_R_)
NU_R=NU_R_/B_R_
MU_R=(MU_R_+1)/B_R_ - 1
MU_R = 5
D_MIN_R=0.2
D_MAX_R=8
A_R = A_R * 1000**(-B_R)
LAMBDA_FACTOR_R=1./A_R*spe.gamma((MU_R+1)/NU_R)/spe.gamma((MU_R+B_R+1)/NU_R)
NTOT_FACTOR_R=spe.gamma((MU_R+1)/NU_R)

QN = 1000
Q = 0.001


x_mean = np.minimum(np.maximum(Q*1.0/(QN+EPS),X_MIN_R),X_MAX_R)
_lambda = (LAMBDA_FACTOR_R_*x_mean)**(-NU_R_)
_N0 = (NU_R_/NTOT_FACTOR_R_)*QN*_lambda**((MU_R_+1)/NU_R_)

N = lambda x: _N0*x**MU_R_*np.exp(-_lambda*x**NU_R_)


x = np.linspace(0.5E-10,1E-4,1000)

print(np.nansum(x*N(x)*(x[1]-x[0])))

#plt.plot(x,N(x))
#D= A_R_*x**B_R_
#plt.figure()
#plt.plot(D*1000,N(x)*D**((1/B_R_)-1))
#
#
x_mean = np.minimum(np.maximum(Q*1.0/(QN+EPS),X_MIN_R),X_MAX_R)
_lambda = (LAMBDA_FACTOR_R*x_mean)**(-NU_R/B_R)
_N0 = (NU_R/NTOT_FACTOR_R)*QN*_lambda**((MU_R+1)/NU_R)

N = lambda D: _N0*D**MU_R*np.exp(-_lambda*D**NU_R)

D = np.linspace(0.0,10,1000)

plt.plot(D,N(D))

scatterer.psd = UnnormalizedGammaPSD(N0=_N0, mu=MU_R,Lambda=_lambda,D_max=8)

print(np.log10(radar.refl(scatterer))*10,np.log10(radar.refl(scatterer,False))*10)
print(10*np.log10(radar.Zdr(scatterer)))



#
#print(np.nansum(A_R*D**B_R*N(D)*(D[1]-D[0])))
#print(np.nansum(N(D)*(D[1]-D[0])))
#
#print(np.nansum(D**6*N(D)*(D[1]-D[0])))
