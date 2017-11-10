#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 31 16:07:34 2017

@author: wolfensb
"""
import numpy as np
import pytmatrix
import pickle
import gzip
import pycosmo
import matplotlib.pyplot as plt
import glob


##quad = pickle.load(gzip.open('/data/cosmo_pol/lookup/quad_pts/quad_pts_mS.p','rb'))
##quad = np.array(quad)
#
##max_ar = quad[:,1,0,-1]
##max_ar = [np.max(a) for a in max_ar]
#
##np.save('max_ar',max_ar)
#
max_ar = np.load('max_ar.npy')
from cosmo_pol.hydrometeors.hydrometeors import create_hydrometeor

from pytmatrix.tmatrix import Scatterer
from pytmatrix.psd import PSDIntegrator, GammaPSD
from pytmatrix import orientation, radar, tmatrix_aux, refractive

h = create_hydrometeor('mS','1mom')
#ar_max = q[1][0]
for i,f in enumerate(np.linspace(1E-3,1,100)):
    print(f)
    h.f_wet = f
    print(h.d_max)
    MAX_AR = 7
    print(h.get_m_func(270,9.8)(h.d_max))
    scatterer = Scatterer(radius=h.d_max/2.0, radius_type = Scatterer.RADIUS_MAXIMUM,
                          wavelength=33., m=h.get_m_func(270,9.41)(h.d_max), axis_ratio=max_ar[i],ndgs=12)
    scatterer.or_pdf = orientation.gaussian_pdf(40.0)
    scatterer.get_SZ()

#files_C = glob.glob('/ltedata/COSMO/GPM_2MOM/*.grb')
#
#hydro = ['R','S','G','H','I']
#
#corr = {}
#for h in hydro:
#    corr[h] = []
#
#for f in files_C:
#
##from cosmo_pol.hydrometeors.hydrometeors import create_hydrometeor
#
##R = create_hydrometeor('R','2mom')
##f = '/ltedata/COSMO/GPM_2MOM/2015-01-11-06-05.grb'
#
#    f = pycosmo.open_file(f)
#
#    for h in hydro:
#        QN = f.get_variable('QN'+h)
#        Q = f.get_variable('Q'+h)
#
#        qn = QN[:][Q[:]>0]
#        q = Q[:][Q[:]>0].ravel()
#        corr[h].append(np.corrcoef(qn,q)[0,1])
#
#    f.close()
#
#for h in hydro:
#    corr[h] = np.array(corr[h])
#
#import pickle
#pickle.dump(corr,open('corr.p','wb'))
#
#means = {}
#std_dev = {}
#for h in hydro:
#    means[h] = np.mean(corr[h])
#    std_dev[h] = np.std(corr[h])
#
