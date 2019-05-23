#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 24 14:26:36 2018

@author: wolfensb
"""

import pyart
import pyart_wrapper

ff = '/ltedata/HYMEX/SOP_2012/Radar/Proc_data/2012/09/29/MXPol-polar-20120929-064418-RHI-166_6.nc'
f = pyart_wrapper.PyartMXPOL(ff)

for k in f.fields.keys():
    if k not in ['Zh','Rhohv']:
        f.fields.pop(k)

f.fixed_angle['data'] = [54]
pyart.io.write_cfradial('rhi_mxpol_ex.nc',f)
plt.figure()
plt.imshow(f.get_field(0,'Rhohv'))


f = pyart.io.read_cfradial('rhi_mxpol_ex.nc')
plt.figure()
plt.imshow(f.get_field(0,'Rhohv'))
