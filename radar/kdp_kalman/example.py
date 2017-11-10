# -*- coding: utf-8 -*-

from kdp_estimation import filter_psidp, estimate_kdp_vulpiani, estimate_kdp_kalman
import netCDF4
import matplotlib.pyplot as plt
import numpy as np

# Read example file
file_rad = netCDF4.Dataset('./example_radar_ppi.nc')

# Read range, angles psidp and rhohv from file
ranges = file_rad.variables['Range'][:]
angles = file_rad.variables['Azimuth'][:]

psidp = file_rad.variables['Psidp'][:]
rhohv = file_rad.variables['Rhohv'][:]

# You should replace your masking values with nans in order for the code to work
psidp[psidp==-99900.0] = np.nan
rhohv[rhohv==-99900.0] = np.nan 
dr = (ranges[1]-ranges[0])/1000. # convert to km

# Filter psidp before running the Kdp estimation methods
psidp_filt = filter_psidp(psidp,rhohv)


# Vulpiani method
kdp_vulp, phidp_rec_vulp = estimate_kdp_vulpiani(psidp_filt,dr)

kdp_levs = np.arange(-1,15,0.2)
plt.figure()
plt.contourf(angles,ranges,kdp_vulp,levels=kdp_levs)
plt.title('Vulpiani method')
plt.xlabel(r'Angle $^{\circ}$')
plt.ylabel('Range [m]')

# Kalman method
kdp_kalm, kdp_std_kalm, phidp_rec_kalm = estimate_kdp_kalman(psidp_filt,dr)    

plt.figure()
plt.contourf(angles,ranges,kdp_kalm,levels=kdp_levs)
plt.title('Kalman method')
plt.xlabel(r'Angle $^{\circ}$')
plt.ylabel('Range [m]')


