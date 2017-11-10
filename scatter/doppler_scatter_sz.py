# -*- coding: utf-8 -*-
"""
Created on Wed Aug 12 15:34:56 2015

@author: wolfensb

TODO : CORRECTION FOR AIR DENSITY RHO !!!
"""

import numpy as np
from scipy.ndimage.filters import gaussian_filter
import copy
from scipy.optimize import fsolve

import doppler_c_new
from cosmo_pol.hydrometeors import hydrometeors
from cosmo_pol.utilities.utilities import nansum_arr, nan_cumsum, sum_arr
from cosmo_pol.utilities.utilities import nan_cumprod, vlinspace
from cosmo_pol.utilities.beam import Beam
from cosmo_pol.utilities import cfg
from cosmo_pol.utilities import utilities
from cosmo_pol.constants import constants

constants.N_BINS_D = 1024
DEG2RAD=np.pi/180.0

def proj_vel(U,V,W,VH,theta,phi):
    return (U*np.sin(phi)+V*np.cos(phi))*np.cos(theta)+(W-VH)*np.sin(theta)

def get_radar_observables(list_beams, lut_sz = 0):
    ###########################################################################
    # Get setup
    global doppler_scheme
    global microphysics_scheme

    # Get info from config
    doppler_scheme= cfg.CONFIG['doppler']['scheme']
    add_turb = cfg.CONFIG['doppler']['turbulence_correction']
    microphysics_scheme = cfg.CONFIG['microphysics']['scheme']
    with_ice_crystals = cfg.CONFIG['microphysics']['with_ice_crystals']
    melting = cfg.CONFIG['microphysics']['with_melting']
    att_corr = cfg.CONFIG['microphysics']['attenuation']
    radar_type = cfg.CONFIG['radar']['type']
    radial_res = cfg.CONFIG['radar']['radial_resolution']

    if radar_type != 'GPM':
        simulate_doppler = True
    else:
        simulate_doppler = False

    # Get dimensions
    num_beams = len(list_beams) # Number of beams
    idx_0 = int(num_beams/2) # Index of central beam
    len_beams = max([len(l.dist_profile) for l in list_beams]) # Beam length

    hydrom_types = []
    hydrom_types.extend(['R','S','G']) # Rain, snow and graupel
    if melting: # IMPORTANT: melting hydrometeors must be placed first
        hydrom_types.extend(['mS','mG'])
    if microphysics_scheme == '2mom':
        hydrom_types.extend(['H']) # Add hail
    if with_ice_crystals:
        hydrom_types.extend('I')

    # Initialize
    # Create hydrometeors
    dic_hydro = {}
    for h in hydrom_types:
        dic_hydro[h] = hydrometeors.create_hydrometeor(h,microphysics_scheme)

    if cfg.CONFIG['integration']['scheme'] == 'ml':
        all_weights = np.array([b.GH_weight for b in list_beams])
        total_weight_at_gates = np.sum(all_weights, axis = 0)

    # Create scattering matrices
    sz_integ = np.zeros((len_beams,len(hydrom_types),12),dtype = 'float32') + np.nan

    # Doppler variables
    if simulate_doppler: # No doppler info for GPM
        rvel_avg = np.zeros(len_beams,)  + np.nan  # average terminal velocity
        if doppler_scheme == 1 or doppler_scheme ==2:
            total_weight_rvel = np.zeros(len_beams,) # total weight where the radial velocity is finite
        elif doppler_scheme == 3:
            doppler_spectrum = np.zeros((len_beams, len(constants.VARRAY)))

    ###########################################################################
    for i,beam in enumerate(list_beams): # Loop on subbeams
        if radar_type != 'GPM': # No doppler info for GPM
            v_integ = np.zeros(len_beams,) # Integrated fall velocity
            n_integ = np.zeros(len_beams,) # Integrated number of particles
            if doppler_scheme == 3:
                # Total attenuation at every subbeam, needed to take into
                # account attenuation
                ah_per_beam = np.zeros((len_beams,),dtype = 'float32') + np.nan

        for j,h in enumerate(hydrom_types): # Loop on hydrometeors
            # PART 2: SCATTERING
            # ---------------

            if melting: # If melting mode is on, we skip the melting hydrometeors for the beams where no melting is detected
                if not beam.has_melting:
                    if h in ['mS','mG']:
                        continue

            elev = beam.elev_profile

            # Since lookup tables are defined for angles >0, we have to check
            # if angles are larger than 90°, in that case we take 180-elevation
            # by symmetricity
            elev[elev>90] = 180-elev[elev>90]
            # Also check if angles are smaller than 0, in that case, flip sign
            elev[elev<0]=-elev[elev<0]

            T = beam.values['T']

            '''
            Part 1 : Get the PSD of the particles
            '''

            QM = beam.values['Q'+h+'_v'] # Get mass densities
            valid_data = QM > 0
            if not np.isscalar(beam.GH_weight):
                valid_data = np.logical_and(valid_data, beam.GH_weight > 0)
            if not np.any(valid_data):
                continue # Skip

            # 1 Moment case
            if microphysics_scheme == '1mom':
                if h == 'mG': # For wet hydrometeor, we need the wet fraction
                    fwet = beam.values['fwet_'+h]
                    dic_hydro[h].set_psd(QM[valid_data], fwet[valid_data])
                elif h == 'mS': # For wet hydrometeor, we need the wet fraction
                    fwet = beam.values['fwet_'+h]
                    dic_hydro[h].set_psd(T[valid_data],QM[valid_data],
                                   fwet[valid_data])
                elif h in ['S','I'] :
                    # For snow N0 is Q and temperature dependent
                    dic_hydro[h].set_psd(T[valid_data], QM[valid_data])
                else: # Rain and graupel
                    dic_hydro[h].set_psd(QM[valid_data])

            # 2 Moment case
            elif microphysics_scheme == '2mom':
                QN = beam.values['QN'+h+'_v']  # Get concentrations as well
                dic_hydro[h].set_psd(QN[valid_data],QM[valid_data])

            # Get list of diameters for this hydrometeor
            if h in ['mS','mG']: # For melting hydrometeor, diameters depend on wet fraction...
                # Number of diameter bins in lookup table
                n_d_bins = lut_sz[h].axes[lut_sz[h].axes_names['d']].shape[1]
                list_D = vlinspace(dic_hydro[h].d_min, dic_hydro[h].d_max,
                                   n_d_bins)

                dD = list_D[:,1] - list_D[:,0]
            else:
                list_D = lut_sz[h].axes[lut_sz[h].axes_names['d']]

                dD = list_D[1] - list_D[0]

            # Compute particle numbers for all diameters
            N = dic_hydro[h].get_N(list_D)

            if len(N.shape) == 1:
                N = np.reshape(N,[len(N),1]) # To be consistent with the einsum dimensions

            '''
            Part 1: Query of the SZ Lookup table
            '''
            # Get SZ matrix
            if h in ['mS','mG']:
                sz = lut_sz[h].lookup_line(e = elev[valid_data],
                                           wc = fwet[valid_data])
            else:
                sz = lut_sz[h].lookup_line(e = elev[valid_data],
                                           t = T[valid_data])
            '''
            Part 3 : Integrate the SZ coefficients over PSD
            '''


            if h in ['mS','mG']:
                # dD is a vector
                sz_psd_integ = np.einsum('ijk,ij->ik',sz,N) * dD[:,None]
            else:
                # dD is a scalar
                sz_psd_integ = np.einsum('ijk,ij->ik',sz,N) * dD

            if len(valid_data) < len_beams:
                # Check for special cases
                valid_data = np.pad(valid_data,(0,len_beams - len(valid_data)),
                                    mode = 'constant', constant_values = False)

            if not np.isscalar(beam.GH_weight):
                weights = beam.GH_weight[valid_data] / total_weight_at_gates[valid_data]
                sz_integ[valid_data,j,:] = nansum_arr(sz_integ[valid_data,j,:],
                            weights[:,None] *
                            sz_psd_integ)
            else:
                sz_integ[valid_data,j,:] = nansum_arr(sz_integ[valid_data,j,:],
                                                      sz_psd_integ *
                                                      beam.GH_weight)
            # PART 2: DOPPLER
            # ---------------
            if not simulate_doppler:
                continue # No Doppler info for GPM

            if doppler_scheme == 1:
                # Get integrated fall speed
                vh,n = dic_hydro[h].integrate_V()
                v_integ[valid_data] = nansum_arr(v_integ[valid_data],vh)
                n_integ[valid_data] = nansum_arr(n_integ[valid_data],n)

            elif doppler_scheme == 2:
                # Get PSD integrated rcs
                rcs = 2*np.pi*(sz[:,:,0] - sz[:,:,1] -
                               sz[:,:,2] + sz[:,:,3])
                # Get fall speed
                v_f = dic_hydro[h].get_V(list_D)

                vh_w = np.trapz(np.multiply(v_f, N * rcs),axis=1)
                n_w = np.trapz(N * rcs, axis=1)

                v_integ[valid_data] = nansum_arr(v_integ[valid_data], vh_w)
                n_integ[valid_data] = nansum_arr(n_integ[valid_data], n_w)

            elif doppler_scheme == 3:
                wavelength = constants.WAVELENGTH
                ah = 4.343e-3 * 2 * wavelength * sz_psd_integ[:,11]
                ah *= radial_res/1000. # Multiply by bin length
                ah_per_beam = nansum_arr(ah_per_beam, ah)

        # For every beam, we get the average fall velocity for all hydrometeors
        # and the resulting radial velocity

        if not simulate_doppler:
            continue

        if doppler_scheme in [1, 2]:
            # Obtain hydrometeor average fall velocity
            v_hydro = v_integ/n_integ
            # Add density weighting
            v_hydro*(beam.values['RHO']/beam.values['RHO'][0])**(0.5)

            # Get radial velocity knowing hydrometeor fall speed and U,V,W from model
            theta = np.deg2rad(beam.elev_profile) # elevation
            phi = np.deg2rad(beam.GH_pt[0])       # azimuth
            proj_wind = proj_vel(beam.values['U'],beam.values['V'],
                               beam.values['W'], v_hydro, theta,phi)

            # Get mask of valid values
            total_weight_rvel = sum_arr(total_weight_rvel,
                                          ~np.isnan(proj_wind)*beam.GH_weight)


            # Average radial velocity for all sub-beams
            rvel_avg = utilities.nansum_arr(rvel_avg,
                                            (proj_wind)*beam.GH_weight)
        elif doppler_scheme == 3: # Full Doppler
            '''
            NOTE: the full Doppler scheme is kind of badly integrated
            within the overall routine and there are lots of code repetitions
            as for exemple RCS are recomputed even though they were computed
            above.
            It could be reimplemented in a better way
            '''
            # Get list of hydrometeors to process
            hydros_to_process = dic_hydro.keys()
            # If melting mode is on, we remove the melting hydrometeors
            # for the beams where no melting is detected
            if melting:
                if not beam.has_melting:
                    hydros_to_process.remove('mS')
                    hydros_to_process.remove('mG')


            beam_spectrum = get_doppler_spectrum(beam,
                             dic_hydro,
                             lut_sz,
                             hydros_to_process)

            if add_turb: # Spectrum spread caused by turbulence
                turb_std = get_turb_std(constants.RANGE_RADAR, beam.values['EDR'])
                beam_spectrum = turb_spectrum_spread(beam_spectrum,turb_std)

            # Correct spectrum for attenuation
            if att_corr:
                wavelength = constants.WAVELENGTH
                # Total number of velocity bins with positive reflectivity
                ah_per_beam = nan_cumsum(ah_per_beam) # cumulated attenuation
                sum_power = np.nansum(beam_spectrum, axis = 1)
                idx_valid = sum_power > 0
                sum_power_db = 10 * np.log10(sum_power)
                # Total attenuation in linear Z
                diff = sum_power[idx_valid] - 10**(0.1 *( sum_power_db[idx_valid] -
                                                   ah_per_beam[idx_valid]))

                func = lambda a : diff - np.sum(np.minimum(a[:,None] ,
                                              beam_spectrum[idx_valid]),axis=1)
                ah_per_bin = fsolve(func, x0 = diff/len(beam_spectrum[0]))

                beam_spectrum[idx_valid,:] -= ah_per_bin[:,None]

            if not np.isscalar(beam.GH_weight):
                beam_spectrum *= beam.GH_weight[:,None] # Multiply by GH weight
            else:
                beam_spectrum *= beam.GH_weight # Multiply by GH weight


            doppler_spectrum += beam_spectrum

    # Finally we integrate the scattering properties over all hydrometeors and beams
    sz_integ = np.nansum(sz_integ,axis=1)
    sz_integ[sz_integ == 0] = np.nan

    # Get radar observables
    ZH,ZV,ZDR,RHOHV,KDP,AH,AV,DELTA_HV = get_pol_from_sz(sz_integ)
    KDP_m = KDP + 0.5*DELTA_HV # Account for differential phase on prop.
    PHIDP = nan_cumsum(2 * KDP_m) * radial_res/1000.

    ZV_ATT = ZV.copy()
    ZH_ATT = ZH.copy()

    if att_corr:
        # AH and AV are in dB so we need to convert them to linear
        ZV_ATT *= nan_cumprod(10**(-0.1*AV*(radial_res/1000.))) # divide to get dist in km
        ZH_ATT *= nan_cumprod(10**(-0.1*AH*(radial_res/1000.)))
        ZDR = ZH_ATT / ZV_ATT


    if simulate_doppler:
        if doppler_scheme in [1, 2]:
            rvel_avg /= total_weight_rvel
        elif doppler_scheme == 3:
            try:
                rvel_avg = np.nansum(np.tile(constants.VARRAY,(len_beams,1))*doppler_spectrum,
                                         axis=1)
                rvel_avg /= np.nansum(doppler_spectrum,axis=1)
            except:
                rvel_avg *= np.nan
    ###########################################################################

    # Create outputs
    rad_obs = {}
    rad_obs['ZH'] = ZH
    rad_obs['ZDR'] = ZDR
    rad_obs['ZV'] = ZV
    rad_obs['KDP'] = KDP_m
    rad_obs['DELTA_HV'] = DELTA_HV
    rad_obs['PHIDP'] = PHIDP
    rad_obs['RHOHV'] = RHOHV
    rad_obs['ATT_H'] = 10*np.log10(ZH) - 10*np.log10(ZH_ATT) # In dB
    rad_obs['ATT_V'] = 10*np.log10(ZV) - 10*np.log10(ZV_ATT) # In dB

    if simulate_doppler:
        rad_obs['RVEL'] = rvel_avg
        if doppler_scheme == 3:
            rad_obs['DSPECTRUM'] = doppler_spectrum

    # This mask serves to tell if the measured point is ok, or below topo or above COSMO domain
    mask = np.zeros(len_beams,)
    for i,beam in enumerate(list_beams):
        mask = sum_arr(mask,beam.mask[0:len_beams],cst = 1) # Get mask of every Beam


    # Larger than 0 means that at least one Beam is below TOPO, smaller than 0 that at least one Beam is above COSMO domain
    mask/=num_beams
    mask[np.logical_and(mask>=0,mask<1)] = 0 # If at least one beam is above topo, we still consider this gate

    # Finally get vectors of distances, height and lat/lon at the central beam
    heights_radar = list_beams[idx_0].heights_profile
    distances_radar = list_beams[idx_0].dist_profile
    lats = list_beams[idx_0].lats_profile
    lons = list_beams[idx_0].lons_profile

    beam_pol = Beam(rad_obs,mask,lats, lons, distances_radar, heights_radar)

    return beam_pol

def get_pol_from_sz(sz):
    wavelength = constants.WAVELENGTH

    # Horizontal reflectivity
    radar_xsect_h = 2*np.pi*(sz[:,0]-sz[:,1]-sz[:,2]+sz[:,3])
    z_h=wavelength**4/(np.pi**5*constants.KW)*radar_xsect_h

    # Vertical reflectivity
    radar_xsect_v = 2*np.pi*(sz[:,0]+sz[:,1]+sz[:,2]+sz[:,3])
    z_v = wavelength**4/(np.pi**5*constants.KW)*radar_xsect_v

    # Differential reflectivity
    zdr=radar_xsect_h/radar_xsect_v

    # Differential phase shift
    kdp=1e-3 * (180.0/np.pi) * wavelength * (sz[:,10]-sz[:,8])

    # Attenuation
    ext_xsect_h = 2 * wavelength * sz[:,11]
    ext_xsect_v = 2 * wavelength * sz[:,9]
    ah= 4.343e-3 * ext_xsect_h
    av= 4.343e-3 * ext_xsect_v

    # Copolar correlation coeff.
    a = (sz[:,4] + sz[:,7])**2 + (sz[:,6] - sz[:,5])**2
    b = (sz[:,0] - sz[:,1] - sz[:,2] + sz[:,3])
    c = (sz[:,0] + sz[:,1] + sz[:,2] + sz[:,3])
    rhohv = np.sqrt(a / (b*c))

    # Backscattering differential phase
    delta_hv = np.arctan2(sz[:,5] - sz[:,6], -sz[:,4] - sz[:,7])

    return z_h,z_v,zdr,rhohv,kdp,ah,av,delta_hv

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

    spectrum = spectrum/convolved_power[:,None] * original_power[:,None]# Rescale to original power

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
