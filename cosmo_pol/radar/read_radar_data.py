# -*- coding: utf-8 -*-
"""
Created on Mon Feb  1 11:21:32 2016

@author: wolfensb
"""


import numpy as np
import h5py, netCDF4
import re
from datetime import datetime

# Define globals for MCH radars

ELEVATION_ANGLES=[-0.2,0.4,1,1.6,2.5,3.5,4.5,5.5,6.5,7.5,8.5,9.5,11,13,16,20,25,30,35,40]
dic_elev={str(e):i for (i,e) in enumerate(ELEVATION_ANGLES)}

NYQUIST_VEL=[8.3, 9.6,8.3,12.4,11.0,12.4,13.8,12.4,13.8,16.5,16.5,16.5,20.6,20.6,20.6,20.6,20.6,20.6,20.6,20.6]
ANG_RES=1


def row_stack(a1,a2):
    [N1,M1]=a1.shape
    [N2,M2]=a2.shape

    if N1>N2:
        a2=np.pad(a2,((0,0),(0,M1-M2)),mode='constant',constant_values=float('nan'))
    elif N2<N1:
        a1=np.pad(a2,((0,0),(0,M2-M1)),mode='constant',constant_values=float('nan'))
    return np.vstack((a1,a2))

def int2float_radar(data, varname, index_angle):
    output=np.zeros(data.shape)
    if varname in ['Z','ZH','ZV','Zh','Zv']:
        output[data!=0]=(data[data!=0]-64)*0.5
        output[data==0]=float('nan')
    elif varname == 'V':
        output[data!=0]=(data[data!=0]-128)/127*NYQUIST_VEL[index_angle]
        output[data==0]=float('nan')
    elif varname == 'W':
        output = data/255*NYQUIST_VEL[index_angle]
    elif varname == 'ZDR':
        output[data!=0]=data[data!=0]*1.0/16.1259842 - 7.9375
        output[data==0]=float('nan')
    elif varname == 'RHO':
        output[data!=0]=1.003-10**(-(data[data!=0]-1.0)/100)
        output[data==0]=float('nan')
    elif varname == 'PHIDP':
        output[data!=0]=(data[data!=0]-32768)/32767*180
        output[data==0]=float('nan')
    elif varname == 'CLUT':
        output=data
    return output

def readCHRadData(filename, variableList, radial_resolution, max_range=np.Inf, min_range=10000):
    # This function reads a HDF5 containing processed radar data in polar coordinates
    # Input  : filename (complete path of the file
    #          variableList, list of variables to be projected
    # Output : varPol, dictionary containing the projected variables, the azimuth and the range

    varPol={}
    h5id=h5py.File(filename,'r')

    # Get dimensions
    siz=h5id['moments']['Z'].shape
    range = np.arange(0,siz[1])*radial_resolution
    idx2keep=np.where(np.logical_and(range<max_range,range>min_range))[0]
    range=range[idx2keep]
    azimuth = (np.arange(0,siz[0])*ANG_RES)
    index_angle=int(re.findall(r"\.([0-9]{3})\.",filename)[0])-1
    elevation=ELEVATION_ANGLES[index_angle]
    # Get variables in polar coordinates
    for varname in variableList:
        data=[]
        data = h5id['moments'][varname][:]
        data=np.asarray(data)
        data=data.astype(float)
        clut= h5id['moments']['CLUT'][:]
        data[clut>=100]=float('nan') # Remove clutter
        data=data[:,idx2keep]
        varPol[varname]=int2float_radar(data,varname,index_angle)

    varPol['resolution']=range[3]-range[2]
    varPol['range']=range
    varPol['azimuth']=azimuth
    varPol['elevation']=elevation
    varPol['nyquist_vel']=NYQUIST_VEL[index_angle]
    # Close netcdf
    h5id.close()

    return varPol

def readMXPOLRadData(filename, variableList, max_range=np.Inf,min_range=0):
        # This function reads a netcdf containing processed radar data in polar coordinates
    # Input  : filename (complete path of the file
    #          variableList, list of variables to be projected
    # Output : varPol, dictionary containing the projected variables, the azimuth and the range

    varPol={}
    ncid=netCDF4.Dataset(filename)

    time=ncid.variables['Time']
    time-=time[0] # To get time in seconds from beginning of scan

    rrange= ncid.variables['Range'][:]

    # Get indexes between min_range and max_range
    idx2keep=np.where(np.logical_and(rrange<max_range,rrange>min_range))[0]
    rrange=rrange[idx2keep]

    # Get variables in polar coordinates
    for varname in variableList:
        if varname in ncid.variables.keys():
            varPol[varname]=ncid.variables[varname][:].T

    try:
        varPol['resolution']=ncid.__dict__['RangeResolution-value']
        varPol['range']=rrange
        varPol['azimuth']=ncid.variables['Azimuth'][:]
        varPol['elevation']=ncid.variables['Elevation'][:]
        varPol['nyquist_vel']=ncid.__dict__['NyquistVelocity-value']
        varPol['longitude']=ncid.__dict__['Longitude-value']
        varPol['latitude']=ncid.__dict__['Latitude-value']
        varPol['altitude']=ncid.__dict__['Altitude-value']
    except:
        varPol['resolution']=ncid.__dict__['RangeResolution_value']
        varPol['range']=rrange
        varPol['azimuth']=ncid.variables['Azimuth'][:]
        varPol['elevation']=ncid.variables['Elevation'][:]
        varPol['nyquist_vel']=ncid.__dict__['NyquistVelocity_value']
        varPol['longitude']=ncid.__dict__['Longitude_value']
        varPol['latitude']=ncid.__dict__['Latitude_value']
        varPol['altitude']=ncid.__dict__['Altitude_value']
    if 'Elevation' in ncid.dimensions:
        scan_type = 'rhi'
    elif 'Azimuth' in ncid.dimensions:
        scan_type = 'ppi'
    varPol['scan_type'] = scan_type
    varPol['time'] = time
    print(time)
    varPol['date'] = datetime.fromtimestamp(ncid.variables['Time'][-1])
    # Close netcdf
    ncid.close()

    return varPol

