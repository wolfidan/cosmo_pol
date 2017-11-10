# -*- coding: utf-8 -*-
"""
Created on Tue Aug  4 12:35:47 2015

@author: wolfensb
"""

import numpy as np

from cosmo_pol.utilities import cfg 
from cosmo_pol.utilities.beam import Beam
from cosmo_pol.utilities.utilities import nansum_arr, sum_arr, nan_cumprod, nan_cumsum


LIST_rad_obs_integratedERVABLES=['ZH','ZV','PHIDP','ZDR','KDP','RHOHV']

def get_radar_observables(list_beams, lut):
    ###########################################################################             
    # Get setup
    att_corr = cfg.CONFIG['attenuation']['correction']
    hydrom_scheme = cfg.CONFIG['microphysics']['scheme']
    
    # Get dimensions
    num_beams=len(list_beams)
    idx_0=int(num_beams/2)
    len_beams=len(list_beams[idx_0].dist_profile)
    
    # Initialize
    if att_corr: # Compute radar bins range
        radial_res=cfg.CONFIG['radar']['radial_resolution']
        
    if hydrom_scheme == 1:
        hydrom_types=['R','S','G'] # Rain, snow and graupel
    elif hydrom_scheme == 2:
        hydrom_types=['R','S','G','H']   # Add hail 
            
    rad_obs_integrated={}
    rad_obs={}
    
    for o in LIST_rad_obs_integratedERVABLES:
        rad_obs_integrated[o]=np.zeros((len_beams,len(hydrom_types)))*float('nan')
        rad_obs[o]=np.zeros((len_beams,len(hydrom_types),num_beams))*float('nan')
   
    ###########################################################################             
    for j,h in enumerate(hydrom_types): # Loop on hydrometeors
        sum_weights=np.zeros((len_beams,))    
        for i,beam in enumerate(list_beams[0:]): # Loop on subbeams

            elev=beam.elev_profile
            
            # Since lookup tables are defined for angles >0, we have to check
            # if angles are larger than 90°, in that case we take 180-elevation
            # by symmetricity
            elev[elev>90]=180-elev[elev>90]
            # Also check if angles are smaller than 0, in that case, flip sign
            elev[elev<0]=-elev[elev<0]            
            
            T=beam.values['T']
            QM=np.log10(beam.values['Q'+h+'_v']) # Get log mass densities
            # Remove values that are not in lookup table
            QM[np.logical_or(QM<lut[h]['ZH'].axes_limits[lut[h]['ZH'].axes_names['q']][0],
                 QM>lut[h]['ZH'].axes_limits[lut[h]['ZH'].axes_names['q']][1])]=float('nan') 
            
            if hydrom_scheme == 2: # Get log number densities as well
                QN=np.log10(beam.values['QN'+h+'_v'] )
                # Remove values that are not in lookup table
                QN[np.logical_or(QN<lut[h]['ZH'].axes_limits[lut[h]['ZH'].axes_names['qn']][0],
                     QN>lut[h]['ZH'].axes_limits[lut[h]['ZH'].axes_names['qn']][1])]=float('nan') 
                     
                lut_pts=np.column_stack((elev,T,QN,QM)).T
            elif hydrom_scheme == 1:
                lut_pts=np.column_stack((elev,T,QM)).T

            # Get polarimetric variables from lookup-table
            ZH_prof=lut[h]['ZH'].lookup_pts(lut_pts)
            ZDR_prof=lut[h]['ZDR'].lookup_pts(lut_pts)    
            KDP_prof=lut[h]['KDP'].lookup_pts(lut_pts)
            RHOHV_prof=lut[h]['RHOHV'].lookup_pts(lut_pts)
            PHIDP_prof=nan_cumsum(KDP_prof)   
            ZV_prof=ZH_prof/ZDR_prof # Use ZDR and ZH to get ZV
            
            # Note that Z2=Z1-a*r in dB gives Z2_l = Z1_l * (1/a_l)**r in linear
            if att_corr:
                # AH and AV are in dB so we need to convert them to linear
                Av_prof=lut[h]['AV'].lookup_pts(lut_pts)
                ZV_prof=ZH_prof/ZDR_prof
                ZV_prof*=nan_cumprod(10**(-Av_prof/10.*(radial_res/1000.))) # divide to get dist in km 
                Ah_prof=lut[h]['AH'].lookup_pts(lut_pts) # convert to linear
                ZH_prof*=nan_cumprod(10**(-Ah_prof/10.*(radial_res/1000.)))
                ZDR_prof=ZH_prof/ZV_prof

            # Add contributions from this subbeam
            rad_obs_integrated['ZH'][:,j]=nansum_arr(rad_obs_integrated['ZH'][:,j],ZH_prof*beam.GH_weight)
            rad_obs_integrated['ZV'][:,j]=nansum_arr(rad_obs_integrated['ZV'][:,j],ZV_prof*beam.GH_weight)
            rad_obs_integrated['PHIDP'][:,j]=nansum_arr(rad_obs_integrated['PHIDP'][:,j],PHIDP_prof*beam.GH_weight)
            rad_obs_integrated['RHOHV'][:,j]=nansum_arr(rad_obs_integrated['RHOHV'][:,j],RHOHV_prof*beam.GH_weight)
            
            rad_obs['ZH'][:,j,i]=ZH_prof
            rad_obs['ZV'][:,j,i]=ZV_prof
            rad_obs['PHIDP'][:,j,i]=PHIDP_prof
            rad_obs['RHOHV'][:,j,i]=RHOHV_prof
            
            sum_weights=sum_arr(sum_weights,~np.isnan(ZDR_prof)*beam.GH_weight)
    
    # Rhohv and Phidp are divided by the total received power
    rad_obs_integrated['RHOHV'][:,j]/=sum_weights
    rad_obs_integrated['PHIDP'][:,j]/=sum_weights

    ###########################################################################     

    # This mask serves to tell if the measured point is ok, or below topo or above COSMO domain
    mask=np.zeros(len_beams,)
    for i,beam in enumerate(list_beams):
        mask=sum_arr(mask,beam.mask) # Get mask of every Beam
        
    # Larger than 1 means that every Beam is below TOPO, smaller than 0 that at least one Beam is above COSMO domain    
    mask/=num_beams 
    mask[np.logical_and(mask>=0,mask<1)]=0
    
    rad_obs_integrated = combine(rad_obs_integrated)
    rad_obs = combine(rad_obs)
    
    # Add standard deviation to output
    for var in rad_obs.keys():
        rad_obs_integrated['std_'+var] = np.nanstd(rad_obs[var],axis=1)
        
    # Finally get vectors of distances, height and lat/lon at the central beam
    idx_0=int(len(list_beams)/2)
    heights_radar=list_beams[idx_0].heights_profile
    distances_radar=list_beams[idx_0].dist_profile
    lats=list_beams[idx_0].lats_profile
    lons=list_beams[idx_0].lons_profile
    
    beam_pol=Beam(rad_obs_integrated,mask,lats, lons, distances_radar, heights_radar)
    
    return beam_pol

def combine(rad_obs_integrated):
    output={}
    # Combines radar observables for different hydrometeors, depending on the type of variable
    # ZH for example is summed and converted to dB, whereas ZDR is combined by taking the ratio of
    # summed ZH and ZV in dB
    
    # ZH
    # Take the sum (in linear scale)
    output['ZH']=np.nansum(rad_obs_integrated['ZH'],axis=1)
    # ZV
    # Take the sum (in linear scale)
    output['ZV']=np.nansum(rad_obs_integrated['ZV'],axis=1)    
    
    # ZDR
    # Take the ratio of ZH and ZV (summed in linear scale)
    output['ZDR']=output['ZH']/output['ZV']
    
    # RHOHV
    # Take the average
    output['RHOHV']=np.nansum(rad_obs_integrated['RHOHV'], axis=1)/rad_obs_integrated['RHOHV'].shape[1]
    
    # PHIDP
    # Just sum
    output['PHIDP']=np.nansum(rad_obs_integrated['PHIDP'],axis=1)
    # KDP
    kdp_full = np.zeros(output['PHIDP'].shape)
    # Add the first element of PHIDP at beginning of Kdp to keep them at same size
    kdp_full[0] = output['PHIDP'][0]
    kdp_full[1:] = np.diff(output['PHIDP'],axis=0)
    
    output['KDP'] = kdp_full
    
    return output
    
