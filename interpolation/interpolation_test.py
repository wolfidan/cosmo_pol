# -*- coding: utf-8 -*-
"""
Created on Fri Jul 24 17:19:22 2015

@author: wolfensb
"""
import numpy as np
import pyproj
import interpolation_c
import cosmo_pol.pycosmo as pycosmo
import pickle

from scipy.stats import norm

from cosmo_pol.refraction import atm_refraction
from cosmo_pol.utilities.config import config
from cosmo_pol.utilities.beam import Beam
from cosmo_pol.utilities import utilities


def integrate_GH_pts(list_GH_pts):
    num_beams=len(list_GH_pts)
    
    list_variables=list_GH_pts[0].values.keys()
    
    integrated_variables={}
    for k in list_variables:
        integrated_variables[k]=[float('nan')]
        for i in list_GH_pts:
            integrated_variables[k]=utilities.nansum_arr(integrated_variables[k],i.values[k]*i.GH_weight)
    
    # Get index of central beam
    idx_0=int(num_beams/2)
    
    # Sum the mask of all beams to get overall mask
    mask=np.zeros(num_beams,) # This mask serves to tell if the measured point is ok, or below topo or above COSMO domain
    for i,p in enumerate(list_GH_pts):
        mask=utilities.sum_arr(mask,p.mask) # Get mask of every Beam
    mask/=float(num_beams) # Larger than 1 means that every Beam is below TOPO, smaller than 0 that at least one Beam is above COSMO domain
    mask[np.logical_and(mask>=0,mask<1)]=0

    heights_radar=list_GH_pts[idx_0].heights_profile
    distances_radar=list_GH_pts[idx_0].dist_profile
    lats=list_GH_pts[idx_0].lats_profile
    lons=list_GH_pts[idx_0].lons_profile

    integrated_beam=Beam(integrated_variables,mask,lats, lons, distances_radar, heights_radar)
    return integrated_beam

def get_profiles(interpolation_mode,dic_variables, azimuth, elevation, radar_range=0,N=0, list_refraction=0):

    list_variables=dic_variables.values()
    keys=dic_variables.keys()
    
    # Get options
    bandwidth_3dB=config['radar_3dB_beamwidth']
    
    ###########################################################################
    if interpolation_mode == 'GH':
        nh_GH=int(config['nh_GH'])
        nv_GH=int(config['nv_GH'])
       
        # Get GH points and weights
        sigma=bandwidth_3dB/(2*np.sqrt(2*np.log(2)))
    
        pts_hor, weights_hor=np.polynomial.hermite.hermgauss(nh_GH)
        pts_hor=pts_hor*np.sqrt(2)*sigma
       
        pts_ver, weights_ver=np.polynomial.hermite.hermgauss(nv_GH)
        pts_ver=pts_ver*np.sqrt(2)*sigma
    
        weights = np.outer(weights_hor,weights_ver)
        
        threshold=np.mean([(weights_hor[0]*weights_hor[int(nh_GH/2)])/(nv_GH*nh_GH), (weights_ver[0]*weights_ver[int(nv_GH/2)])/(nv_GH*nh_GH)])
        sum_weights=np.pi
        
        beam_broadening=nh_GH>1 and nv_GH>1 # Boolean for beam-broadening (if only one GH point : No beam-broadening)

    ###########################################################################

    if interpolation_mode == 'GH_improved':
        
        n_rad=9
        n_ang=9
       
        x_leg, w_leg = np.polynomial.legendre.leggauss(n_ang)
        x_her, w_her = np.polynomial.hermite.hermgauss(n_rad)
        
#        bw = [5,1.5,1.5,1.8,1.5,1.5,5]
        sigma = [2,2,0.68574,2,2,0.39374,1.21239]
        mu = [-12.86,-7.0931,0,0.2948,8.37059,10.05448,13.20]
        a_dB = [-44.219,-37.7577,0,-24.0645,-42.1912,-41.4037,-44.7814]

        list_pts = []
        sum_weights = 0
        for i in range(n_rad):
            for j in range(len(sigma)):
                for k in range(n_ang):
                    r = mu[j]+np.sqrt(2)*sigma[j]*x_her[i]
                    theta = np.pi * x_leg[k] + np.pi
                    weight = np.pi*w_her[i]*w_leg[k]*10**(0.1*a_dB[j])*np.sqrt(2)*sigma[j]*abs(r) # Laplacian
                    sum_weights+=weight
                    list_pts.append([weight,[r*np.cos(theta)+azimuth,r*np.sin(theta)+elevation]])


        beam_broadening=n_rad>1 or n_ang>1 # Boolean for beam-broadening (if only one GH point : No beam-broadening)

    ###########################################################################
        
    elif interpolation_mode == 'Gauss':
        # Get points and weights
        sigma=bandwidth_3dB/(2*np.sqrt(2*np.log(2)))
        
        data = pickle.load(open('real_antenna_s.p','rb'))
        angles = data['angles']
        
        pts_hor = angles
        pts_ver = angles
    
        
        threshold = -np.Inf
        beam_broadening = True
        
        ax,ay = np.meshgrid(angles,angles)
        d = (ax**2 + ay**2)
        weights = 1/(2*np.pi*sigma)*np.exp(-0.5*d/sigma**2)*(angles[1]-angles[0])**2
        
        sum_weights= np.sum(weights)
        
    ###########################################################################
        
    elif interpolation_mode == 'Real':
        
        data = pickle.load(open('real_antenna.p','rb'))
        angles = data['angles']
        
        pts_hor = angles
        pts_ver = angles
        
        threshold = -np.Inf
        beam_broadening = True
        
        weights = data['data']
        sum_weights= np.sum(weights)
        
    ###########################################################################
        
    elif interpolation_mode == 'Real_s':
        
        data = pickle.load(open('real_antenna_s.p','rb'))
        angles = data['angles']
        
        pts_hor = angles
        pts_ver = angles
        
        threshold = -np.Inf
        beam_broadening = True
        
        weights = data['data']
        sum_weights= np.sum(weights)
    
    
    list_beams=[]     
    
    if interpolation_mode == 'GH_improved':
        # create vector of bin positions
        bins_ranges=np.arange(config['radar_rres']/2,config['radar_range'],config['radar_rres'])
        # Get coordinates of virtual radar
        radar_pos=config['radar_coords']
        
        refraction_method='4/3' # Other methods take too much time if no quadrature scheme is used
        
        for i in range(len(list_pts)):
            S,H, E = atm_refraction.get_radar_refraction(bins_ranges, list_pts[i][1][1], radar_pos, refraction_method, N)
            lats,lons,b=get_radar_beam_trilin(list_variables, list_pts[i][0], S,H)
            # Create dictionary of beams
            dic_beams={}
            for k, bi in enumerate(b): # Loop on interpolated variables
                if k == 0: # Do this only for the first variable (same mask for all variables)
                    mask_beam=np.zeros((len(bi)))
                    mask_beam[bi==-9999]=-1 # Means that the interpolated point is above COSMO domain
                    mask_beam[np.isnan(bi)]=1  # NaN means that the interpolated point is below COSMO terrain
                bi[mask_beam!=0]=float('nan') # Assign NaN to all missing data
                dic_beams[keys[k]]=bi # Create dictionary
            list_beams.append(Beam(dic_beams, mask_beam, lats, lons, S,H,E,list_pts[i][1], list_pts[i][0]/sum_weights))       
    else:
        print interpolation_mode
        
        if list_refraction==0: # Calculate refraction for vertical GH points
            list_refraction=[]
            
            # create vector of bin positions
            bins_ranges=np.arange(config['radar_rres']/2,config['radar_range'],config['radar_rres'])
            # Get coordinates of virtual radar
            radar_pos=config['radar_coords']
            
            refraction_method='4/3' # Other methods take too much time if no quadrature scheme is used
    
            for pt in pts_ver:
                S,H, E = atm_refraction.get_radar_refraction(bins_ranges, pt+elevation, radar_pos, refraction_method, N)
                list_refraction.append((S,H,E))
        
        for i in range(len(pts_hor)): 
            print i
            for j in range(len(pts_ver)):
                if weights[i,j]>threshold or not beam_broadening:
                    
                    # GH coordinates
                    pt=[pts_hor[i]+azimuth,pts_ver[j]+elevation]
                    weight=weights[i,j]/sum_weights
                    # Interpolate beam
                    lats,lons,b=get_radar_beam_trilin(list_variables, pts_hor[i]+azimuth, list_refraction[j][0],list_refraction[j][1])
                    # Create dictionary of beams
                    dic_beams={}
                    for k, bi in enumerate(b): # Loop on interpolated variables
                        if k == 0: # Do this only for the first variable (same mask for all variables)
                            mask_beam = np.zeros((len(bi)))
                            mask_beam[bi == -9999] =- 1 # Means that the interpolated point is above COSMO domain
                            mask_beam[np.isnan(bi)] = 1  # NaN means that the interpolated point is below COSMO terrain
                        bi[mask_beam!=0] = float('nan') # Assign NaN to all missing data
                        dic_beams[keys[k]] = bi # Create dictionary
                    list_beams.append(Beam(dic_beams, mask_beam, lats, lons, list_refraction[j][0],list_refraction[j][1],list_refraction[j][2],pt, weight))        
    return list_beams
        
            
    
def get_radar_beam_trilin(list_vars, azimuth, distances_profile, heights_profile):
    # Get position of virtual radar from config
    radar_pos=config['radar_coords']

    # Initialize WGS84 geoid
    g = pyproj.Geod(ellps='WGS84')

    # Get radar bins coordinates
    lons_rad=[]
    lats_rad=[]
    # Using the distance on ground of every radar gate, we get its latlon coordinates
    for d in distances_profile:
        lon,lat,ang=g.fwd(radar_pos[1],radar_pos[0],azimuth,d) # Note that pyproj uses lon, lat whereas I used lat, lon
        lons_rad.append(lon)
        lats_rad.append(lat)

    # Convert to numpy array
    lons_rad=np.array(lons_rad)
    lats_rad=np.array(lats_rad)
    
    # Initialize interpolated beams
    all_beams=[]
    
    # Get model heights and COSMO proj from first variable    
    ###########################################################################
    model_heights=list_vars[0].attributes['z-levels']
    rad_interp_values=np.zeros(len(distances_profile),)*float('nan')
    
    # Get COSMO local coordinates info
    proj_COSMO=list_vars[0].attributes['proj_info']
    # Get lower left corner of COSMO domain in local coordinates
    llc_COSMO=(float(proj_COSMO['Lo1']), float(proj_COSMO['La1']))
    res_COSMO=list_vars[0].attributes['resolution']

    # Get resolution 
    # Transform radar WGS coordinates into local COSMO coordinates

    coords_rad_loc=pycosmo.WGS_to_COSMO((lats_rad,lons_rad),[proj_COSMO['Latitude_of_southern_pole'],proj_COSMO['Longitude_of_southern_pole']])  
    llc_COSMO=np.asarray(llc_COSMO).astype('float32')
    

    # Now we interpolate all variables along beam using C-code file
    ###########################################################################
    for n,var in enumerate(list_vars):           

        model_data=var.data
        rad_interp_values=interpolation_c.get_all_radar_pts(len(distances_profile),coords_rad_loc,heights_profile,model_data,model_heights\
        , llc_COSMO,res_COSMO)
        all_beams.append(rad_interp_values[1][:])

    return lats_rad, lons_rad, all_beams
