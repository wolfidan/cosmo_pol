# -*- coding: utf-8 -*-
"""
Created on Wed Aug 12 15:34:56 2015

@author: wolfensb

TODO : CORRECTION FOR AIR DENSITY RHO !!!
"""

import numpy as np
from scipy.ndimage.filters import gaussian_filter
import copy

import doppler_c_new
from cosmo_pol.hydrometeors import hydrometeors
from cosmo_pol.utilities.beam import Beam
from cosmo_pol.utilities import cfg
from cosmo_pol.utilities import utilities
from cosmo_pol.constants import constants

constants.N_BINS_D = 1024
DEG2RAD=np.pi/180.0

def proj_vel(U,V,W,VH,theta,phi):
    return (U*np.sin(phi)+V*np.cos(phi))*np.cos(theta)+(W-VH)*np.sin(theta)

def get_doppler_velocity(list_beams, lut_sz = 0):
    ###########################################################################
    # Get setup
    global doppler_scheme
    global microphysics_scheme

    doppler_scheme= cfg.CONFIG['doppler']['scheme']
    add_turb = cfg.CONFIG['doppler']['turbulence_correction']
    microphysics_scheme = cfg.CONFIG['microphysics']['scheme']
    with_ice_crystals = cfg.CONFIG['microphysics']['with_ice_crystals']
    melting = cfg.CONFIG['microphysics']['with_melting']
    att_corr = cfg.CONFIG['microphysics']['attenuation']

   # Get dimensions
    num_beams=len(list_beams) # Number of beams
    idx_0=int(num_beams/2) # Index of central beam
    len_beams=max([len(l.dist_profile) for l in list_beams]) # Beam length

    hydrom_types = []
    if melting: # IMPORTANT: melting hydrometeors must be placed first
        hydrom_types.extend(['mS','mG'])
    if microphysics_scheme == '1mom':
        hydrom_types.extend(['R','S','G']) # Rain, snow and graupel
    elif microphysics_scheme == '2mom':
        hydrom_types.extend(['H']) # Add hail
    if with_ice_crystals:
        hydrom_types.extend('I')
    hydrom_types = np.array(hydrom_types)

    # Create dic of hydrometeors
    list_hydrom={}
    for h in hydrom_types:
        list_hydrom[h] = hydrometeors.create_hydrometeor(h,microphysics_scheme)

#    if cfg.CONFIG['integration']['scheme'] == 'ml':
#        all_weights = np.array([b.GH_weight for b in list_beams])
#        sums_weights = np.sum(all_weights, axis = 0)

    ###########################################################################
    # Get radial wind and doppler spectrum (if scheme == 1 or 2)
    if doppler_scheme == 1 or doppler_scheme ==2:

        rvel_avg=np.zeros(len_beams,)*float('nan') # average radial velocity
        sum_weights=np.zeros(len_beams,) # mask of GH weights

        for beam in list_beams:

            hydros_to_process = list_hydrom.keys()
            if melting: # If melting mode is on, we remove the melting hydrometeors for the beams where no melting is detected
                if not beam.has_melting:
                    hydros_to_process.remove('mS')
                    hydros_to_process.remove('mG')


            if doppler_scheme == 1: # Weighting by PSD only
                v_hydro=get_v_hydro_unweighted(beam, list_hydrom,
                                               hydros_to_process)

            elif doppler_scheme == 2: # Weighting by RCS and PSD
                v_hydro=get_v_hydro_weighted(beam,list_hydrom, lut_sz,
                                             hydros_to_process)

            # Get radial velocity knowing hydrometeor fall speed and U,V,W from model
            theta = beam.elev_profile*DEG2RAD
            phi = beam.GH_pt[0]*DEG2RAD

            proj_wind=proj_vel(beam.values['U'],beam.values['V'], beam.values['W'],
                               v_hydro, theta,phi)

            # Get mask of valid values
            sum_weights=utilities.sum_arr(sum_weights,
                                          ~np.isnan(proj_wind)*beam.GH_weight)


            # Average radial velocity for all sub-beams
            rvel_avg = utilities.nansum_arr(rvel_avg,
                                            (proj_wind)*beam.GH_weight)

        # We need to divide by the total weights of valid beams at every bin
        rvel_avg /= sum_weights

    elif doppler_scheme == 3:
        rvel_avg = np.zeros(len_beams,)
        doppler_spectrum = np.zeros((len_beams, len(constants.VARRAY)))
        for beam in list_beams:

            # Get list of hydrometeors to process
            hydros_to_process = list_hydrom.keys()
            if melting: # If melting mode is on, we remove the melting hydrometeors for the beams where no melting is detected
                if not beam.has_melting:
                    hydros_to_process.remove('mS')
                    hydros_to_process.remove('mG')

            if not np.isscalar(beam.GH_weight):
                beam_spectrum = get_doppler_spectrum(beam, list_hydrom,
                                                 lut_sz,
                                                 hydros_to_process) * beam.GH_weight[:,None] # Multiply by GH weight
            else:
                beam_spectrum = get_doppler_spectrum(beam, list_hydrom,
                                                 lut_sz,
                                                 hydros_to_process) * beam.GH_weight # Multiply by GH weight


            if add_turb: # Spectrum spread caused by turbulence
                turb_std = get_turb_std(constants.RANGE_RADAR, beam.values['EDR'])
                beam_spectrum = turb_spectrum_spread(beam_spectrum,turb_std)
            doppler_spectrum += beam_spectrum
        try:
            rvel_avg = np.nansum(np.tile(constants.VARRAY,(len_beams,1))*doppler_spectrum,
                                 axis=1)/np.nansum(doppler_spectrum,axis=1)
        except:
            rvel_avg *= np.nan


    ###########################################################################
    # Get mask
    # This mask serves to tell if the measured point is ok, or below topo or above COSMO domain
    mask=np.zeros(len_beams,)

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
        dic_vars={'RVEL':rvel_avg,'DSPECTRUM':doppler_spectrum}
    else:
        # No doppler spectrum is computed
        dic_vars={'RVEL':rvel_avg}

    beam_doppler=Beam(dic_vars,mask,lats, lons, distances_radar, heights_radar)
    return beam_doppler

def get_v_hydro_unweighted(beam, list_hydrom, hydros_to_process):
    vh_avg = np.zeros(beam.values['T'].shape)
    vh_avg.fill(np.nan)
    n_avg = np.zeros(beam.values['T'].shape)
    n_avg.fill(np.nan)
    for i,h in enumerate(hydros_to_process):
        valid_data = beam.values['Q'+h+'_v'] > 0
        if not np.isscalar(beam.GH_weight):
            valid_data = np.logical_and(valid_data, beam.GH_weight > 0)
        if not np.any(valid_data):
            continue # Skip

        if not np.any(valid_data):
            continue # skip

        if cfg.CONFIG['microphysics']['scheme'] == '1mom':
            if h == 'mS':
                list_hydrom[h].set_psd(beam.values['T'][valid_data],
                           beam.values['Q' + h + '_v'][valid_data],
                           beam.values['fwet_' + h][valid_data])
            elif h == 'mG':
                list_hydrom[h].set_psd(beam.values['Q'+h+'_v'][valid_data],
                           beam.values['fwet_' + h][valid_data])
            elif h in ['S','I']:
                list_hydrom[h].set_psd(beam.values['T'][valid_data],
                           beam.values['Q' + h + '_v'][valid_data])
            else:
                list_hydrom[h].set_psd(beam.values['Q'+h+'_v'][valid_data])
        elif cfg.CONFIG['microphysics']['scheme'] == '2mom':
            list_hydrom[h].set_psd(beam.values['QN'+h+'_v'][valid_data],
                       beam.values['Q'+h+'_v'][valid_data])

        # Get fall speed
        vh,n = list_hydrom[h].integrate_V()
        vh_avg[valid_data] = utilities.nansum_arr(vh_avg[valid_data],vh)
        n_avg[valid_data] = utilities.nansum_arr(n_avg[valid_data],n)

    v_hydro_unweighted = vh_avg/n_avg # Average over all hydrometeors

    return v_hydro_unweighted*(beam.values['RHO']/beam.values['RHO'][0])**(0.5)

def get_v_hydro_weighted(beam, list_hydrom, lut_sz, hydros_to_process):

    hydrom_scheme = cfg.CONFIG['microphysics']['scheme']

    vh_avg = np.zeros(beam.values['T'].shape)
    vh_avg.fill(np.nan)
    n_avg = np.zeros(beam.values['T'].shape)
    n_avg.fill(np.nan)

    for i,h in enumerate(hydros_to_process):
        valid_data = beam.values['Q'+h+'_v'] > 0
        if not np.isscalar(beam.GH_weight):
            valid_data = np.logical_and(valid_data, beam.GH_weight > 0)
        if not np.any(valid_data):
            continue # Skip

        if not np.any(valid_data):
            continue # skip

        # Get all elevations
        elev = beam.elev_profile
        # Since lookup tables are defined for angles >0, we have to check
        # if angles are larger than 90°, in that case we take 180-elevation
        # by symmetricity
        elev_lut = copy.deepcopy(elev)
        elev_lut[elev_lut>90]=180-elev_lut[elev_lut>90]
        # Also check if angles are smaller than 0, in that case, flip sign
        elev_lut[elev_lut<0]=-elev_lut[elev_lut<0]

        T=beam.values['T']

        '''
        Part 1 : Get the PSD of the particles
        '''
        QM = beam.values['Q'+h+'_v'] # Get mass densities
        # 1 Moment case
        if hydrom_scheme == '1mom':
            if h == 'mS':
                fwet = beam.values['fwet_'+h]
                list_hydrom[h].set_psd(T[valid_data],
                           QM[valid_data],
                           fwet[valid_data])
            elif h == 'mG':
                fwet = beam.values['fwet_'+h]
                list_hydrom[h].set_psd(QM[valid_data],
                           beam.values['fwet_'+h][valid_data])

            elif h in ['S','I']:
                # For ice N0 is Q and temperature dependent
                list_hydrom[h].set_psd(T[valid_data],QM[valid_data])
            else:
                list_hydrom[h].set_psd(QM[valid_data])
        # 2 Moment case
        elif hydrom_scheme == '2mom':
            QN = beam.values['QN'+h+'_v']  # Get concentrations as well
            list_hydrom[h].set_psd(QN[valid_data],QM[valid_data])


        # Get list of diameters for this hydrometeor
        if h in ['mS','mG']: # For melting hydrometeor, diameters depend on wet fraction...
            # Number of diameter bins in lookup table
            n_d_bins = lut_sz[h].axes[lut_sz[h].axes_names['d']].shape[1]
            list_D = utilities.vlinspace(list_hydrom[h].d_min, list_hydrom[h].d_max,
                                         n_d_bins)
        else:
            list_D = lut_sz[h].axes[lut_sz[h].axes_names['d']]


        N = list_hydrom[h].get_N(list_D)

        if len(N.shape) == 1:
            N = np.reshape(N,[len(N),1]) # To be consistent with the einsum dimensions

        '''
        Part 2: Query of the SZ Lookup table  and RCS computation
        '''
        # Get SZ matrix
        if h in ['mS','mG']:
            sz = lut_sz[h].lookup_line(e = elev_lut[valid_data],
                                       wc = fwet[valid_data])
        else:
            sz = lut_sz[h].lookup_line(e = elev_lut[valid_data],
                                       t = T[valid_data])

        # get RCS
        rcs = 2*np.pi*(sz[:,:,0] - sz[:,:,1] - sz[:,:,2] + sz[:,:,3])

        '''
        Part 3 : Integrate
        '''
        # Get fall speed

        v_f = list_hydrom[h].get_V(list_D)


        vh_w = np.trapz(np.multiply(v_f, N*rcs),axis=1)
        n_w = np.trapz(N*rcs, axis=1)
#        if h in ['mS','mG']:
#            print(np.where(valid_data),vh_w/n_w)

        vh_avg[valid_data] = utilities.nansum_arr(vh_avg[valid_data],
                                                  vh_w)
        n_avg[valid_data] = utilities.nansum_arr(n_avg[valid_data],
                                                  n_w)

#        print(h,time.time() -t0 )


    v_hydro_weighted = vh_avg/n_avg # Average over all hydrometeors

    return v_hydro_weighted*(beam.values['RHO']/beam.values['RHO'][0])**(0.5)

def get_doppler_spectrum(beam, list_hydrom, lut_sz, hydros_to_process):

    # Get dimensions
    len_beam = len(beam.dist_profile) # Length of the considered bea

    # Initialize matrix of reflectivities
    refl=np.zeros((len_beam, len(constants.VARRAY)),dtype='float32')

    # Get all elevations
    elev = beam.elev_profile
    # Since lookup tables are defined for angles >0, we have to check
    # if angles are larger than 90°, in that case we take 180-elevation
    # by symmetricity
    elev_lut = copy.deepcopy(elev)
    elev_lut[elev_lut>90]=180-elev_lut[elev_lut>90]
    # Also check if angles are smaller than 0, in that case, flip sign
    elev_lut[elev_lut<0]=-elev_lut[elev_lut<0]

    # Get azimuth angle
    phi = beam.GH_pt[0]

    # Correction of velocity for air density
    rho_corr = (beam.values['RHO']/beam.values['RHO'][0])**0.5

    for i in range(len_beam):  # Loop on all radar gates
        if beam.mask[i] == 0:
            if not np.isscalar(beam.GH_weight):
                if beam.GH_weight[i] == 0:
                    continue
            # Get parameters of the PSD (lambda or lambda/N0) for present hydrom
            # meteors

            list_hydrom_gate = {}

            for j,h in enumerate(hydros_to_process):
                Q = beam.values['Q'+h+'_v'][i]
                T = beam.values['T'][i]
                if Q > 0:
                    list_hydrom_gate[h] = list_hydrom[h]

                    if cfg.CONFIG['microphysics']['scheme'] == '1mom':
                        if h =='mG':
                            fwet = beam.values['fwet_'+h][i]
                            list_hydrom_gate[h].set_psd(np.array([Q]),
                                                 np.array([fwet]))
                        elif h == 'mS':
                            fwet = beam.values['fwet_'+h][i]
                            list_hydrom_gate[h].set_psd(np.array([T]),
                                                 np.array([Q]),
                                                 np.array([fwet]))
                        elif h in ['S','I']:
                            list_hydrom_gate[h].set_psd(np.array([T]),
                                                        np.array([Q]))
                        else:
                            list_hydrom_gate[h].set_psd(np.array([Q]))

                    elif cfg.CONFIG['microphysics']['scheme'] == '2mom':
                        QN = beam.values['QN'+h+'_v'][i]
                        list_hydrom_gate[h].set_psd(np.array([QN]),
                                                    np.array([Q]))


            # Initialize matrix of radar cross sections, N and D
            n_hydrom = len(list_hydrom_gate.keys())
            rcs = np.zeros((constants.N_BINS_D,n_hydrom),dtype='float32') + np.nan
            N = np.zeros((constants.N_BINS_D,n_hydrom),dtype='float32') + np.nan
            D = np.zeros((constants.N_BINS_D,n_hydrom),dtype='float32') + np.nan
            D_min = np.zeros((n_hydrom),dtype = 'float32') + np.nan
            step_D = np.zeros((n_hydrom),dtype = 'float32') + np.nan

            # Get N and D for all hydrometeors that are present
            for j,h in enumerate(list_hydrom_gate.keys()):
                D[:,j] = np.linspace(list_hydrom_gate[h].d_min,
                                     list_hydrom_gate[h].d_max,
                                     constants.N_BINS_D)

                D_min[j] = D[0,j]
                step_D[j] = D[1,j] - D[0,j]
                N[:,j] = list_hydrom_gate[h].get_N(D[:,j])

            # Compute RCS for all hydrometeors that are present
            for j,h in enumerate(list_hydrom_gate.keys()):
                if h in ['mS','mG']:
                    sz = lut_sz[h].lookup_line(e = elev_lut[i],
                               wc = beam.values['fwet_'+h][i])
                else:
                    sz = lut_sz[h].lookup_line(e = elev_lut[i],t = beam.values['T'][i])
                # get RCS
                rcs[:,j]= (2*np.pi*(sz[:,0] - sz[:,1] - sz[:,2] + sz[:,3])).T


            # Important we use symetrical elevations only for lookup querying, not
            # for actual trigonometrical velocity estimation
            Da, Db, idx = get_diameter_from_rad_vel(list_hydrom_gate,phi,
                          elev[i],beam.values['U'][i],
                          beam.values['V'][i],beam.values['W'][i],rho_corr[i])
            try:

#                hydrom_dry = np.logical_and(hydrom_types != 'mS',
#                                            hydrom_types != 'mG')
#
#                # Treat non-melting hydrometeors first
#                N0=np.array([list_hydrom[h].N0 for h in hydrom_types[hydrom_dry]],
#                            dtype='float32')
#                lambdas=np.array([list_hydrom[h].lambda_ for h in hydrom_types[hydrom_dry]],
#                                 dtype='float32')
#                mu=np.array([list_hydrom[h].mu for h in hydrom_types[hydrom_dry]],
#                            dtype='float32')
#                nu=np.array([list_hydrom[h].nu for h in hydrom_types[hydrom_dry]],
#                            dtype='float32')
#
                refl[i,idx] = doppler_c_new.get_refl(len(idx),Da,
                                Db,rcs,D,N,step_D,D_min)[1]


                wavelength = constants.WAVELENGTH
                refl[i,idx] *= wavelength**4/(np.pi**5*constants.KW**2)
            except:
                print('An error occured in the Doppler spectrum calculation...')
                raise

    return refl

def get_turb_std(ranges,EDR):
    sigma_r = 0.35 * cfg.CONFIG['radar']['radial_resolution']
    sigma_theta = cfg.CONFIG['radar']['3dB_beamwidth']*DEG2RAD/(4.*np.sqrt(np.log(2))) # Convert to rad

    turb_std=np.zeros((len(EDR),))

    # Method of calculation follow Doviak and Zrnic (p.409)
    # Case 1 : sigma_r << r*sigma_theta
    idx_r = sigma_r<0.1*ranges*sigma_theta
    turb_std[idx_r] = ((ranges[idx_r]*EDR[idx_r]*sigma_theta*
                        constants.A**(3/2.))/0.72)**(1/3.)
    # Case 2 : r*sigma_theta <= sigma_r
    idx_r = sigma_r>=0.1*ranges*sigma_theta
    turb_std[idx_r]= (((EDR[idx_r]*sigma_r*(1.35*constants.A)**(3/2))/
            (11./15.+4./15.*(ranges[idx_r]*sigma_theta/sigma_r)**2)
            **(-3/2.))**(1/3.))
    return turb_std

def turb_spectrum_spread(spectrum, turb_std):
    v=constants.VARRAY
    # Convolve spectrum and turbulence gaussian distributions
    # Get resolution in velocity
    v_res=v[2]-v[1]

    original_power=np.sum(spectrum,1) # Power of every spectrum (at all radar gates)
    for i,t in enumerate(turb_std):
    	spectrum[i,:] = gaussian_filter(spectrum[i,:],t/v_res) # Filter only columnwise (i.e. on velocity bins)
    convolved_power=np.sum(spectrum,1) # Power of every convolved spectrum (at all radar gates)

    spectrum=spectrum/convolved_power[:,None]*original_power[:,None]# Rescale to original power

    return spectrum

def get_diameter_from_rad_vel(list_hydrom, phi,theta,U,V,W,rho_corr):
    theta=theta*DEG2RAD
    phi=phi*DEG2RAD

    wh = 1./rho_corr*(W+(U*np.sin(phi)+V*np.cos(phi))/np.tan(theta)\
        -constants.VARRAY/np.sin(theta))

    idx = np.where(wh>=0)[0] # We are only interested in positive fall speeds

    wh = wh[idx]

    hydrom_types = list_hydrom.keys()

    D = np.zeros((len(idx), len(hydrom_types)), dtype='float32')

    # Get D bins from V bins
    for i,h in enumerate(list_hydrom.keys()): # Loop on hydrometeors
        D[:,i] = list_hydrom[h].get_D_from_V(wh)
        # Threshold to valid diameters
        D[D>=list_hydrom[h].d_max]=list_hydrom[h].d_max
        D[D<=list_hydrom[h].d_min]=list_hydrom[h].d_min

    # Array of left bin limits
    Da = np.minimum(D[0:-1,:],D[1:,:])
    # Array of right bin limits
    Db = np.maximum(D[0:-1,:],D[1:,:])

    # Get indice of lines where at least one bin width is larger than 0
    mask = np.where(np.sum((Db-Da) == 0.0,axis=1) < len(hydrom_types))[0]

    return Da[mask,:], Db[mask,:],idx[mask]


if __name__=='__main__':


    import pickle
    import gzip
    from cosmo_pol.lookup.lut import Lookup_table, load_all_lut
    beams = pickle.load(open('/data/cosmo_pol/ex_new_pts.p','rb'))
    lut = load_all_lut('1mom', ['S','R','G','mS','mG','I'], 9.41, 'tmatrix_new')

#    lut = pickle.load(gzip.open('/data/cosmo_pol/lookup/final_lut_quad/all_luts_SZ_f_2_7_1mom.pz','rb'))


    from cosmo_pol.utilities import cfg
    cfg.init('./option_files/MXPOL_RHI.yml') # Initialize options with 'options_radop.txt'
    constants.update()
#    from cosmo_pol.utilities import tictoc
#    tictoc.tic()
    cfg.CONFIG['doppler']['scheme'] = 2
    cfg.CONFIG['microphysics']['with_ice_crystals'] = True
    cfg.CONFIG['microphysics']['with_melting'] = True
    c = get_doppler_velocity(beams, lut)


#    from cosmo_pol.lookup.read_lut import get_lookup_tables
#    import pickle
#
#    l=pickle.load(open('../ex_beams_rhi.txt','rb'))
#    lut_pol,lut_rcs=get_lookup_tables('1mom',5.6,True)
#
##
##    config['doppler_vel_method']=2
##    rvel=get_doppler_velocity(l,lut_rcs)
##    plt.figure()
##    plt.plot(rvel.values['v_radial'])
##
##
#    cfg.CONFIG['doppler_scheme']=3
#    rvel=get_doppler_velocity(l,lut_rcs)
#
#    cfg.CONFIG['doppler_scheme']=2
#    rvel3=get_doppler_velocity(l,lut_pol)
#
#    cfg.CONFIG['doppler_scheme']=1
#    rvel2=get_doppler_velocity(l)
#    import matplotlib.pyplot as plt
#    plt.plot(rvel.values['RVel'])
#    plt.hold(True)
#    plt.plot(rvel2.values['RVel'])
#    plt.plot(rvel3.values['RVel'])
#    plt.legend(['Dop','Unweighted','Weighted'])
