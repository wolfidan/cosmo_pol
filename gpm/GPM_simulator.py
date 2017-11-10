
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  3 11:24:58 2015

@author: wolfensb
"""

import h5py
import pyproj as pp
import numpy as np
import copy
from cosmo_pol.constants import constants
import matplotlib.pyplot as plt
#from interpolation import get_profiles_GH

g = pp.Geod(ellps='WGS84')

ang2rad=1./180.*np.pi
maxHeightCOSMO=35000

def get_group(band):
    if band == 'Ku':
        group = 'NS'
    elif band == 'Ka':
        group = 'HS'
    elif band == 'both':
        group = 'MS'
    return group

class SimulatedGPM():
    def __init__(self,list_beams,dim, band):
        # Reorganize output to make it easier to compare with GPM
        # The idea is to keep only the bins that are above ground and to put
        # every beam into a common matrix

        [N,M]=dim

        pol_vars=list_beams[0].values.keys() # List of simulated variables

        list_beams_cp = copy.deepcopy(list_beams) # make a deepcopy to avoid overwriting the original one

        bin_surface=np.zeros((N,M))
        for idx in range(len(list_beams_cp)): # Here we remove all points that are below the topography COSMO
            i = int(np.floor(idx/M))
            j = idx-i*M
            try:
                bin_surface[i,j] = len(list_beams_cp[idx].mask)-np.where(list_beams_cp[idx].mask>=1)[0][0]
            except:
                bin_surface[i,j]=0
            for k in pol_vars:
                list_beams_cp[idx].values[k]=list_beams_cp[idx].values[k][list_beams_cp[idx].mask<1] # Remove points that are below topo
            list_beams_cp[idx].lats_profile=list_beams_cp[idx].lats_profile[list_beams_cp[idx].mask<1]
            list_beams_cp[idx].lons_profile=list_beams_cp[idx].lons_profile[list_beams_cp[idx].mask<1]

        # Length of longest beam
        max_len=np.max([len(b.dist_profile) for b in list_beams])

        # Initalize output dictionary
        list_beams_formatted={}
        for k in pol_vars:
            list_beams_formatted[k]=np.zeros((N,M,max_len))
        # Initialize lats and lons 3D array

        lats = np.zeros((N,M,max_len))*float('nan')
        lons = np.zeros((N,M,max_len))*float('nan')

        # Fill the output dictionary starting from the ground
        for idx in range(len(list_beams_cp)):
            i = int(np.floor(idx/M))
            j = idx-i*M
            len_beam=len(list_beams_cp[idx].values[k])
            # Flip because we want to start from the ground
            l=list_beams_cp[idx].lats_profile
            ll=list_beams_cp[idx].lons_profile

            lats[i,j,0:len_beam]=l[::-1]
            lons[i,j,0:len_beam]=ll[::-1]
            # Flip [::-1] because we want to start from the ground
            for k in pol_vars:
                list_beams_formatted[k][i,j,0:len_beam] = list_beams_cp[idx].values[k][::-1]

        self.bin_surface = bin_surface
        self.band = band
        self.lats = lats
        self.lons = lons
        self.data = list_beams_formatted
        return


def compare_operator_with_GPM(simulated_GPM_swath,GPM_filename):
    ''' Inputs:
        -simulated_GPM_swath : GPM swath (output of the radar operator)
        -GPM_filename        : Path of the corresponding GPM scan

        Outputs:
        -ZH_ground           : ZH measured and simulated at the ground
        -ZH_everywhere       : ZH measured and simulated everywhere
        -diff                : number of bins between clutter free GPM bin and
                               last simulated bin
    '''

    gpm_f = h5py.File(GPM_filename,'r')

    band = simulated_GPM_swath.band
    group = get_group(band)

    # Get latlon
    lat_simul = simulated_GPM_swath.lats
    lon_simul = simulated_GPM_swath.lons

    # First bin without clutter
    binNoClutter = gpm_f[group]['PRE']['binClutterFreeBottom'][:]

    # Get # of GPM bins
    GPM_N_bins =  \
            constants.GPM_NO_BINS_KA if band == 'Ka' \
                else constants.GPM_NO_BINS_KU

    diff = (GPM_N_bins - binNoClutter) - simulated_GPM_swath.bin_surface

    diff[diff<0] = 0
    diff = diff.astype(int)

#    binSurface=gpm_f[group]['PRE']['binRealSurface'][:]


    # AT GROUND
    ##########################################################################
    # ZH simulated at ground
    ZH_s_dBZ = 10*np.log10(simulated_GPM_swath.data['ZH'])
    [N,M,K]=ZH_s_dBZ.shape
    k,j = np.meshgrid(np.arange(M),np.arange(N)) # Create 2D index
    ZH_s_grd=ZH_s_dBZ[j,k,diff]

    # ZH measured at ground
    ZH_gpm = gpm_f[group]['PRE']['zFactorMeasured'][:]
    ZH_gpm[ZH_gpm<-1000] = float('nan') # Put Nan where data is missing

    ZH_m_grd = ZH_gpm[j,k,binNoClutter]
    ZH_m_grd[ZH_m_grd<-1000] = float('nan')
    # GPM seems to measure lots of noise, so we apply a mask which corresponds
    # to all pixels classified as precip by gpm
    mask = gpm_f[group]['PRE']['flagPrecip'][:]
    ZH_m_grd[mask] = np.nan

    ZH_ground={}
    ZH_ground['measured'] = ZH_m_grd
    ZH_ground['simulated'] = ZH_s_grd
    ZH_ground['lat'] = lat_simul[j,k,diff]
    ZH_ground['lon'] = lon_simul[j,k,diff]

    # EVERYWHERE
    ##########################################################################
    K = np.min([K,ZH_gpm.shape[2]])

    ZH_m_everywhere = np.zeros((N,M,K))
    lat_everywhere = np.zeros((N,M,K))
    lon_everywhere = np.zeros((N,M,K))

    ZH_s_everywhere = np.zeros((N,M,K))

    for i in range(K):
        ZH_s_everywhere[j,k,i] = ZH_s_dBZ[j,k,diff+i]
        lat_everywhere[j,k,i] = lat_simul[j,k,diff+i]
        lon_everywhere[j,k,i] = lon_simul[j,k,diff+i]
        idx = binNoClutter-i
        ZH_m_everywhere[j,k,i] = ZH_gpm[j,k,binNoClutter-i]
        ZH_m_everywhere[:,:,i][idx < 0] = np.nan

    ZH_everywhere={}
    ZH_everywhere['measured'] = ZH_m_everywhere
    ZH_everywhere['simulated'] = ZH_s_everywhere
    ZH_everywhere['lon'] = lon_everywhere
    ZH_everywhere['lat'] = lat_everywhere

    return ZH_ground,ZH_everywhere, diff

#def correct_latlon(lat,lon,az,ellips_offsets,elev_ang,altitudes):
#    distances_to_ground=(ellips_offsets+altitudes)/np.arctan(elev_ang*ang2rad)
#    corr_lon,corr_lat,ang=g.fwd(lon,lat,az,distances_to_ground)
#    return corr_lon,corr_lat


def get_GPM_angles(GPM_file,band):

    group = get_group(band)

    gpm_f = h5py.File(GPM_file,'r')
    lat_2D = gpm_f[group]['Latitude'][:]
    lon_2D = gpm_f[group]['Longitude'][:]
#    altitudes_2D=gpm_f[group]['PRE']['elevation'][:]

    center_lat_sc = gpm_f[group]['navigation']['scLat'][:]
    center_lon_sc = gpm_f[group]['navigation']['scLon'][:]
    altitudes_sc = gpm_f[group]['navigation']['dprAlt'][:]

    pos_sc = gpm_f[group]['navigation']['scPos'][:]

#    elev_ang=90-gpm_f[group]['PRE']['localZenithAngle'][:]
#    ellips_offsets=gpm_f[group]['PRE']['ellipsoidBinOffset'][:]

    azimuths = np.zeros(lat_2D.shape)
    ranges = np.zeros(lat_2D.shape)
    elevations = np.zeros(lon_2D.shape)

    # Projection from lat/long/alt to eced
    ecef = pp.Proj(proj='geocent', ellps='WGS84', datum='WGS84')
    lla = pp.Proj(proj='latlong', ellps='WGS84', datum='WGS84')


    [N,M]=lat_2D.shape

    for i in range(N):
        for j in range(M):

            a,b,d=g.inv(center_lon_sc[i],center_lat_sc[i],lon_2D[i,j],lat_2D[i,j])
            azimuths[i,j]=a

            surf_x,surf_y,surf_z = pp.transform(lla,ecef,lon_2D[i,j] ,lat_2D[i,j], 0)
            range_targ=np.sqrt((surf_x-pos_sc[i,0])**2+(surf_y-pos_sc[i,1])**2+(surf_z-pos_sc[i,2])**2)
            ranges[i,j]=range_targ

            H=np.sqrt((pos_sc[i,0])**2+(pos_sc[i,1])**2+(pos_sc[i,2])**2)
            RE=H-altitudes_sc[i]

            theta=-np.arcsin((H**2+range_targ**2-RE**2)/(2*H*range_targ))/np.pi*180.

            if np.isnan(theta): # Can happen for angles very close to pi
                theta=-90
            elevations[i,j]=-theta # Flip sign since elevations are defined positively in lut

    coords_GPM=np.vstack((center_lat_sc,center_lon_sc,altitudes_sc)).T
    return azimuths, elevations, ranges,coords_GPM

def get_earth_radius(latitude):
    a=6378.1370*1000
    b=6356.7523*1000
    return np.sqrt(((a**2*np.cos(latitude))**2+(b**2*np.sin(latitude))**2)/((a*np.cos(latitude))**2+(b*np.sin(latitude))**2))

#def get_GPM_refraction2(latitude, altitude_radar, max_range, elevation):
#    deg2rad=np.pi/180.
#    rad2deg=180./np.pi
#
#    elev_rad=-elevation*deg2rad
#    ke=1
#    maxHeightCOSMO=constants.MAX_HEIGHT_COSMO
#    RE=get_earth_radius(latitude)
#    # Compute maximum range to target (using cosinus law in the triangle earth center-radar-target)
#    range_vec=np.arange(constants.GPM_RADIAL_RES/2,max_range,constants.GPM_RADIAL_RES)
#
#    H=-(np.sqrt(range_vec**2 + (ke*RE)**2+2*range_vec*ke*RE*np.sin(elev_rad))-ke*RE)+altitude_radar
#
#    S=ke*RE*np.arcsin((range_vec*np.cos(elev_rad))/(ke*RE+H))
#    E=elevation-np.arctan(range_vec*np.cos(elev_rad)/(range_vec*np.sin(elev_rad)+ke*RE+altitude_radar))*rad2deg
##
#    in_lower_atm=[H<maxHeightCOSMO]
#
#    H = H[in_lower_atm]
#    S = S[in_lower_atm]
#    E = E[in_lower_atm]
#
#    return S.astype('float32'),H.astype('float32'), E.astype('float32')

def test_accuracy(GPM_file,band):
    import pyproj

    group=get_group(band)

    g = pyproj.Geod(ellps='WGS84')
    azimuths, elevations, ranges,coords_GPM=get_GPM_angles(GPM_file,band)

    [N,M]=azimuths.shape

    lat=np.zeros((N,M))
    lon=np.zeros((N,M))
    for i in range(N):
        c=coords_GPM[i]
        for j in range(M):
            S,H,E=get_GPM_refraction2(c[0],c[2],ranges[i,j],elevations[i,j])
            lon[i,j],lat[i,j],ang=g.fwd(c[0],c[1],azimuths[i,j],S[-1])

    gpm_f=h5py.File(gpm_fname,'r')
    real_lat=gpm_f[group]['Latitude'][:]
    real_lon=gpm_f[group]['Longitude'][:]

    plt.imshow(real_lat-lat)
    plt.colorbar()

    return

if __name__=='__main__':
    from cosmo_pol.utilities import cfg
    cfg.init('') # Initialize options with 'options_radop.txt'
    constants.update() # Update constants now that we know user config
    import pickle
    f = '/ltedata/COSMO/GPM_data/2014-10-20-22-10.HDF5'
    a = pickle.load(open('../validation/ex_gpm.p','rb'))
    ground,all,c = compare_operator_with_GPM(a,f)


#    S,H,E = get_GPM_refraction(90,47,)

#    import pickle
#    u=pickle.load(open('/media/wolfensb/Storage/cosmo_pol/mom1.pz','rb'))
#    cl = SimulatedGPM(a,[81,49],'Ku')
#    a = SimulatedGPM(u,az.shape,'Ku')
#    uu = compare_operator_with_GPM(a,gpm_file)
#    g,a=SimulatedGPM_Swath(u['r'],u['s'],u['b'])
#    import pyproj
#    gpm_fname='./GPM_files/2014-08-13-02-28.HDF5'
#    band='Ku'
#
#
#    test_accuracy(gpm_fname,band)
#    import pickle
#    simulated_GPM_swath=pickle.load(open('ex_output_GPM.txt','rb'))
#    gpm_fname='./GPM_files/2014-08-13-02-28.HDF5'
#    a,b=compare_operator_with_GPM(simulated_GPM_swath,gpm_fname)

#    plt.figure()
#    plt.subplot(1,2,1)
#    plt.contourf(a['measured'],levels=np.arange(0,50,1))
#    plt.subplot(1,2,2)
#    plt.contourf(a['simulated'],levels=np.arange(0,50,1))
##
#    plt.figure()
#    plt.subplot(1,2,1)
#    plt.contourf(b['measured'][:,:,5],levels=np.arange(0,50,1))
#    plt.subplot(1,2,2)
#    plt.contourf(b['simulated'][:,:,5],levels=np.arange(0,50,1))
#
#    plt.figure()
#    plt.subplot(1,2,1)
#    plt.contourf(b['measured'][:,:,20],levels=np.arange(0,50,1))
#    plt.subplot(1,2,2)
#    plt.contourf(b['simulated'][:,:,20],levels=np.arange(0,50,1))
#
#    plt.figure()
#    plt.subplot(1,2,1)
#    plt.contourf(b['measured'][:,:,30],levels=np.arange(0,50,1))
#    plt.subplot(1,2,2)
#    plt.contourf(b['simulated'][:,:,30],levels=np.arange(0,50,1))
#
#    plt.figure()
#    plt.subplot(1,2,1)
#    plt.contourf(b['measured'][:,:,40],levels=np.arange(0,50,1))
#    plt.subplot(1,2,2)
#    plt.contourf(b['simulated'][:,:,40],levels=np.arange(0,50,1))
#
