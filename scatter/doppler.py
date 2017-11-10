# -*- coding: utf-8 -*-
"""
Created on Wed Aug 12 15:34:56 2015

@author: wolfensb

TODO : CORRECTION FOR AIR DENSITY RHO !!!
"""

import numpy as np
from scipy.ndimage.filters import gaussian_filter

import doppler_c
from cosmo_pol.hydrometeors import hydrometeors
from cosmo_pol.utilities.beam import Beam
from cosmo_pol.utilities import cfg
from cosmo_pol.utilities import utilities
from cosmo_pol.constants import constants

DEG2RAD=np.pi/180.0

def proj_vel(U,V,W,VH,theta,phi):
    return (U*np.sin(phi)+V*np.cos(phi))*np.cos(theta)+(W-VH)*np.sin(theta)

def get_doppler_velocity(list_beams, lut=0):
    ###########################################################################
    # Get setup
    global doppler_scheme
    global microphysics_scheme
    
    doppler_scheme=cfg.CONFIG['doppler']['scheme']
    microphysics_scheme=cfg.CONFIG['microphysics']['scheme']

    # Check if necessary variables for turbulence scheme are present
    if doppler_scheme == 4 and 'EDR' in list_beams.keys(): 
        add_turb=True
    else:
        add_turb=False
        if doppler_scheme == 4:
             print('No eddy dissipitation rate variable found in COSMO file, could not use doppler_scheme == 4, using doppler_scheme == 3 instead')
    
    # Get some dimensions
    # Length of the beams (= number of bins)
    len_beam=len(list_beams[0].dist_profile)
    num_beams=len(list_beams) # Number of beams (= number of GH points)
    
    if microphysics_scheme == 1:
        hydrom_types=['R','S','G'] # Rain, snow and graupel
    elif microphysics_scheme == 2:
        hydrom_types=['R','S','G','H'] # Add hail 
            
    # Create dic of hydrometeors
    list_hydrom={}
    for h in hydrom_types:
        list_hydrom[h]=hydrometeors.create_hydrometeor(h,microphysics_scheme)
    
    
    ###########################################################################    
    # Get radial wind and doppler spectrum (if scheme == 1 or 2)
    if doppler_scheme == 1 or doppler_scheme ==2:
        
        rvel_avg=np.zeros(len_beam,)*float('nan') # average radial velocity 
        sum_weights=np.zeros(len_beam,) # mask of GH weights
        
        for beam in list_beams:
            if doppler_scheme == 1:
#                if microphysics_scheme == '1mom':
                v_hydro=get_v_hydro_unweighted(beam,list_hydrom)
                
            elif doppler_scheme == 2:
                v_hydro=get_v_hydro_weighted(beam,list_hydrom,lut)

            # Get radial velocity knowing hydrometeor fall speed and U,V,W from model
            theta = beam.elev_profile*DEG2RAD
            phi = beam.GH_pt[0]*DEG2RAD

            proj_wind=proj_vel(beam.values['U'],beam.values['V'],beam.values['W'],v_hydro,theta,phi)

            # Get mask of valid values
            sum_weights=utilities.sum_arr(sum_weights,~np.isnan(proj_wind)*beam.GH_weight)

            # Average radial velocity for all sub-beams
            rvel_avg=utilities.nansum_arr(rvel_avg,(proj_wind)*beam.GH_weight)
#  
    elif doppler_scheme == 3:
        rvel_avg=np.zeros(len_beam,)
        doppler_spectrum=np.zeros((len_beam,len(constants.VARRAY)))
        for beam in list_beams:
            beam_spectrum=get_doppler_spectrum(beam,list_hydrom,lut)*beam.GH_weight # Multiply by GH weight
            if add_turb: # Spectrum spread caused by turbulence
                turb_std=get_turb_std(constants.RANGE_RADAR,beam.values['EDR'])
                beam_spectrum=turb_spectrum_spread(beam_spectrum,turb_std)
            doppler_spectrum+=beam_spectrum
        try:
            rvel_avg=np.sum(np.tile(constants.VARRAY,(len_beam,1))*doppler_spectrum,axis=1)/np.sum(doppler_spectrum,axis=1)
        except:
            rvel_avg*=float('nan')
            
    if doppler_scheme == 1 or doppler_scheme == 2: # For schemes 1 and 2 where there is no computation of the doppler spectrum
        # We need to divide by the total weights of valid beams at every bin
        rvel_avg/=sum_weights
        
    ###########################################################################    
    # Get mask
    # This mask serves to tell if the measured point is ok, or below topo or above COSMO domain
    mask=np.zeros(len_beam,)

    for i,beam in enumerate(list_beams):
        mask=utilities.sum_arr(mask,beam.mask) # Get mask of every Beam
    mask/=num_beams # Larger than 1 means that every Beam is below TOPO, smaller than 0 that at least one Beam is above COSMO domain
    mask[np.logical_and(mask>=0,mask<1)]=0
    
    # Finally get vectors of distances, height and lat/lon at the central beam
    idx_0=int(len(list_beams)/2)
    heights_radar=list_beams[idx_0].heights_profile
    distances_radar=list_beams[idx_0].dist_profile
    lats=list_beams[idx_0].lats_profile
    lons=list_beams[idx_0].lons_profile

    if doppler_scheme == 3:
        dic_vars={'RVel':rvel_avg,'DSpectrum':doppler_spectrum}
    else:
        # No doppler spectrum is computed
        dic_vars={'RVel':rvel_avg}
        
    beam_doppler=Beam(dic_vars,mask,lats, lons, distances_radar, heights_radar)    
    return beam_doppler
        
def get_v_hydro_unweighted(beam, list_hydrom):
    vh_avg=np.zeros(beam.values['T'].shape)*float('nan')
    n_avg=np.zeros(beam.values['T'].shape)*float('nan')
    for i,h in enumerate(list_hydrom.keys()):
        if cfg.CONFIG['microphysics']['scheme'] == 1:
            if h=='S':
                list_hydrom['S'].set_psd(beam.values['T'],beam.values['Q'+h+'_v'])
            else:
                list_hydrom[h].set_psd(beam.values['Q'+h+'_v'])
        elif cfg.CONFIG['microphysics']['scheme'] == 2:
            list_hydrom[h].set_psd(beam.values['QN'+h+'_v'],beam.values['Q'+h+'_v'])
            
        # Get fall speed
        vh,n=list_hydrom[h].integrate_V()
        
        vh_avg=utilities.nansum_arr(vh_avg,vh)
        n_avg=utilities.nansum_arr(n_avg,n)
    
    v_hydro_unweighted=vh_avg/n_avg # Average over all hydrometeors

    return v_hydro_unweighted*(beam.values['RHO']/beam.values['RHO'][0])**(0.5)
    
def get_v_hydro_weighted(beam, list_hydrom, lut):
    # For improved performance, the weighted (by rcs) fall velocities are 
    # precomputed and stored in lookup-tables

    vh_weighted_avg=0
    n_weighted_avg=0
    for i,h in enumerate(list_hydrom.keys()):
        QM=np.log10(beam.values['Q'+h+'_v']) # Get log mass densities
        # Remove values that are not in lookup table
        QM[np.logical_or(QM<lut[h]['VH'].axes_limits[lut[h]['VH'].axes_names['q']][0],
             QM>lut[h]['VH'].axes_limits[lut[h]['VH'].axes_names['q']][1])]=float('nan') 
            
        # Get parameters of the PSD (lambda or lambda/N0)
        if cfg.CONFIG['microphysics']['scheme'] == 1:
            lut_pts=np.column_stack((beam.elev_profile,beam.values['T'],
                    QM)).T
        elif cfg.CONFIG['microphysics']['scheme'] == 2:
            QN=np.log10(beam.values['QN'+h+'_v']) # Get log mass densities
            # Remove values that are not in lookup table
            QN[np.logical_or(QN<lut[h]['VH'].axes_limits[lut[h]['VH'].axes_names['qn']][0],
                 QN>lut[h]['VH'].axes_limits[lut[h]['VH'].axes_names['qn']][1])]=float('nan') 
            lut_pts=np.column_stack((beam.elev_profile,beam.values['T'],
                    QM,QN)).T            

        # Get fall speed
        vh_w=lut[h]['VH'].lookup_pts(lut_pts)
        n_w=lut[h]['ZH'].lookup_pts(lut_pts)

        vh_weighted_avg=vh_weighted_avg+vh_w*n_w
        n_weighted_avg=n_weighted_avg+n_w
        
    v_hydro_weighted=vh_weighted_avg/n_weighted_avg # Average over all hydrometeors

    return v_hydro_weighted*(beam.values['RHO']/beam.values['RHO'][0])**(0.5)

def get_doppler_spectrum(beam, list_hydrom, lut_rcs):

    hydrom_types = list_hydrom.keys()
    
    D=np.column_stack((lut_rcs[h].axes[lut_rcs[h].axes_names['d']] for h in hydrom_types))
    D_min = [list_hydrom[h].d_min for h in hydrom_types]
    
    step_D=D[1,:]-D[0,:]
    
    len_beam = len(beam.dist_profile) # Length of the considered beam
    
    refl=np.zeros((len_beam, len(constants.VARRAY)),dtype='float32')

    for i in range(len_beam): 
        if beam.mask[i]==0:
            # Get parameters of the PSD (lambda or lambda/N0)
            for j,h in enumerate(hydrom_types):
                if cfg.CONFIG['microphysics']['scheme'] == 1:
                    if h=='S':
                        list_hydrom['S'].set_psd(beam.values['T'][i],beam.values['Q'+h+'_v'][i])
                    else:
                        list_hydrom[h].set_psd(beam.values['Q'+h+'_v'][i])
                elif cfg.CONFIG['microphysics']['scheme'] == 2:
                    list_hydrom[h].set_psd(beam.values['QN'+h+'_v'][i],beam.values['Q'+h+'_v'][i])

            N0=np.array([list_hydrom[h].N0 for h in hydrom_types],dtype='float32')
            lambdas=np.array([list_hydrom[h].lambda_ for h in hydrom_types],dtype='float32')
            mu=np.array([list_hydrom[h].mu for h in hydrom_types],dtype='float32')
            nu=np.array([list_hydrom[h].nu for h in hydrom_types],dtype='float32')
            
            elevation = beam.elev_profile[i]
            # Correction of velocity for air density
            rho_corr = (beam.values['RHO'][i]/beam.values['RHO'][0])**0.5 
            
            if elevation>90: # Check for angle as lookup tables are defined only for angles 0 to 90
                elevation=180-elevation

            rcs = np.column_stack([lut_rcs[h].lookup_line(t=beam.values['T'][i],e=elevation) for h in hydrom_types])
            rcs = rcs.astype('float32')
            Da, Db, idx = get_diameter_from_rad_vel(list_hydrom,beam.GH_pt[0],
                          beam.elev_profile[i],beam.values['U'][i],
                          beam.values['V'][i],beam.values['W'][i],rho_corr) 
            try:
                refl[i,idx]=doppler_c.get_refl(len(idx),Da, Db,D,rcs,N0,mu,lambdas,nu,step_D,D_min)[1]
            except:
                print('An error occured in the Doppler spectrum calculation...')
                raise

    return refl
    
def get_turb_std(ranges,EDR):
    sigma_r=cfg.CONFIG['radar']['radial_resolution']
    sigma_theta=cfg.CONFIG['radar']['3dB_beamwidth']*DEG2RAD # Convert to rad
    
    turb_std=np.zeros((len(EDR),))
    
    # Method of calculation follow Doviak and Zrnic (p.409)
    # Case 1 : sigma_r << r*sigma_theta
    idx_r=sigma_r<0.1*ranges*sigma_theta
    turb_std[idx_r]=((ranges[idx_r]*EDR[idx_r]*sigma_theta*constants.A**(3/2))/0.72)**(1/3)
    # Case 2 : r*sigma_theta <= sigma_r
    idx_r=sigma_r>=0.1*ranges*sigma_theta
    turb_std[idx_r]=(((EDR[idx_r]*sigma_r*(1.35*constants.A)**(3/2))/(11/15+4/15*(ranges[idx_r]*sigma_theta/sigma_r)**2)**(-3/2))**(1/3))
    return turb_std
    
def turb_spectrum_spread(spectrum, turb_std):
    v=constants.VARRAY
    # Convolve spectrum and turbulence gaussian distributions
    # Get resolution in velocity
    v_res=v[2]-v[1]
    
    original_power=np.sum(spectrum,1) # Power of every spectrum (at all radar gates)
    spectrum=gaussian_filter(spectrum,[0, turb_std/v_res]) # Filter only columnwise (i.e. on velocity bins)
    convolved_power=np.sum(spectrum,1) # Power of every convolved spectrum (at all radar gates)
    
    spectrum=spectrum/convolved_power[:,None]*original_power[:,None]# Rescale to original power
    
    return spectrum

def get_diameter_from_rad_vel(list_hydrom, phi,theta,U,V,W,rho_corr):
    theta=theta*DEG2RAD
    phi=phi*DEG2RAD
    
    wh=1./rho_corr*(W+(U*np.sin(phi)+V*np.cos(phi))/np.tan(theta)\
        -constants.VARRAY/np.sin(theta))
    idx=np.where(wh>=0)[0] # We are only interested in positive fall speeds 

    wh=wh[idx]
    
    hydrom_types = list_hydrom.keys()
    
    D=np.zeros((len(idx),len(hydrom_types)), dtype='float32')
    
    # Get D bins from V bins
    for i,h in enumerate(list_hydrom.keys()): # Loop on hydrometeors
        D[:,i]=list_hydrom[h].get_D_from_V(wh)
        # Threshold to valid diameters
        D[D>=list_hydrom[h].d_max]=list_hydrom[h].d_max
        D[D<=list_hydrom[h].d_min]=list_hydrom[h].d_min
    
    # Array of left bin limits
    Da=np.minimum(D[0:-1,:],D[1:,:])
    # Array of right bin limits
    Db=np.maximum(D[0:-1,:],D[1:,:])

    # Get indice of lines where at least one bin width is larger than 0
    mask=np.where(np.sum((Db-Da)==0.0,axis=1)<len(hydrom_types))[0]
    
    return Da[mask,:], Db[mask,:],idx[mask]

 
if __name__=='__main__':
    from cosmo_pol.lookup.read_lut import get_lookup_tables
    import pickle

    l=pickle.load(open('../ex_beams_rhi.txt','rb'))
    lut_pol,lut_rcs=get_lookup_tables('1mom',5.6,True)
    
#
#    config['doppler_vel_method']=2
#    rvel=get_doppler_velocity(l,lut_rcs)
#    plt.figure()
#    plt.plot(rvel.values['v_radial'])   
#    
#    
    cfg.CONFIG['doppler_scheme']=3
    rvel=get_doppler_velocity(l,lut_rcs)

    cfg.CONFIG['doppler_scheme']=2
    rvel3=get_doppler_velocity(l,lut_pol)
    
    cfg.CONFIG['doppler_scheme']=1
    rvel2=get_doppler_velocity(l)
    import matplotlib.pyplot as plt
    plt.plot(rvel.values['RVel'])
    plt.hold(True)
    plt.plot(rvel2.values['RVel'])
    plt.plot(rvel3.values['RVel'])
    plt.legend(['Dop','Unweighted','Weighted'])