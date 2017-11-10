# -*- coding: utf-8 -*-
"""
Created on Tue Aug  4 12:35:47 2015

@author: wolfensb
"""

import numpy as np

from cosmo_pol.constants import constants
from cosmo_pol.utilities import cfg
from cosmo_pol.utilities.beam import Beam
from cosmo_pol.utilities.utilities import nansum_arr, nan_cumsum, sum_arr, nan_cumprod, vlinspace
from cosmo_pol.hydrometeors.hydrometeors_zaw import create_hydrometeor


def get_radar_observables(list_beams, lut_sz):
    ###########################################################################
    # Get setup
    att_corr = cfg.CONFIG['microphysics']['attenuation']
    microphysics_scheme = cfg.CONFIG['microphysics']['scheme']
    with_ice_crystals = cfg.CONFIG['microphysics']['with_ice_crystals']
    melting = cfg.CONFIG['microphysics']['with_melting']

    # Get dimensions
    num_beams = len(list_beams) # Number of beams
    idx_0 = int(num_beams/2) # Index of central beam
    len_beams = len(list_beams[idx_0].dist_profile)  # Beam length

    # Initialize

    radial_res=cfg.CONFIG['radar']['radial_resolution']

    hydrom_types = []
    hydrom_types.extend(['R','S','G']) # Rain, snow and graupel
    if melting: # IMPORTANT: melting hydrometeors must be placed first
        hydrom_types.extend(['mS','mG'])
    if microphysics_scheme == '2mom':
        hydrom_types.extend(['H']) # Add hail
    if with_ice_crystals:
        hydrom_types.extend('I')

    # Initialize matrices
    sz_integ = np.zeros((len_beams,len(hydrom_types),12),dtype = 'float32')
    sz_integ.fill(np.nan)

    if cfg.CONFIG['integration']['scheme'] in ['ml','ml2']:
        all_weights = np.array([b.GH_weight for b in list_beams])
        sums_weights = np.sum(all_weights, axis = 0)

    ###########################################################################
    for j,h in enumerate(hydrom_types): # Loop on hydrometeors
        # Create a hydrometeor instance
        hydrom = create_hydrometeor(h, microphysics_scheme)

        for i,beam in enumerate(list_beams): # Loop on subbeams
            if melting: # If melting mode is on, we skip the melting hydrometeors for the beams where no melting is detected
                if not beam.has_melting:
                    if h in ['mS','mG']:
                        continue

            # For GPM some sub-beams are longer than the main beam, so we discard
            # the "excess" part
            for k in beam.values.keys():
                beam.values[k] = beam.values[k][0:len_beams]

            elev = beam.elev_profile[0:len_beams]

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
                    hydrom.set_psd(QM[valid_data], fwet[valid_data])
                elif h == 'mS': # For wet hydrometeor, we need the wet fraction
                    fwet = beam.values['fwet_'+h]
                    hydrom.set_psd(T[valid_data],QM[valid_data],
                                   fwet[valid_data])
                elif h in ['S','I'] :
                    # For snow N0 is Q and temperature dependent
                    hydrom.set_psd(T[valid_data], QM[valid_data])
                else: # Rain and graupel
                    hydrom.set_psd(QM[valid_data])

            # 2 Moment case
            elif microphysics_scheme == '2mom':
                QN = beam.values['QN'+h+'_v']  # Get concentrations as well
                hydrom.set_psd(QN[valid_data],QM[valid_data])

            # Get list of diameters for this hydrometeor
            if h in ['mS','mG']: # For melting hydrometeor, diameters depend on wet fraction...
                # Number of diameter bins in lookup table
                n_d_bins = lut_sz[h].axes[lut_sz[h].axes_names['d']].shape[1]
                list_D = vlinspace(hydrom.d_min, hydrom.d_max, n_d_bins)

                dD = list_D[:,1] - list_D[:,0]
            else:
                list_D = lut_sz[h].axes[lut_sz[h].axes_names['d']]

                dD = list_D[1] - list_D[0]

            # Compute particle numbers for all diameters
            N = hydrom.get_N(list_D)

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
            Part 3 : Integrate the SZ coefficients
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
                weights = beam.GH_weight[valid_data] / sums_weights[valid_data]
                sz_integ[valid_data,j,:] = nansum_arr(sz_integ[valid_data,j,:],
                            weights[:,None] *
                            sz_psd_integ)
            else:
                sz_integ[valid_data,j,:] = nansum_arr(sz_integ[valid_data,j,:],
                                                      sz_psd_integ *
                                                      beam.GH_weight)


    # Finally we integrate for all hydrometeors
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
#        print(nan_cumprod(10**(-0.1*AH*(radial_res/1000.))))
        ZDR = ZH_ATT / ZV_ATT

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

if __name__ == '__main__':
    import pickle
    import gzip
    from cosmo_pol.lookup.lut import Lookup_table
    beams = pickle.load(open('/media/wolfensb/Storage/cosmo_pol/ex_beams_rhi.txt','rb'))
    beams = [beams[7]]
    beams[0].weight = 1.0
    beams[0].values['QS_v']*=0.
    beams[0].values['QG_v']*=0.
    beams[0].values['QR_v']= (np.isfinite(beams[0].values['QR_v'])).astype(int)*0.0001
#    lut = pickle.load(gzip.open('/media/wolfensb/Storage/cosmo_pol/lookup/final_lut/all_luts_SZ_f_2_7_1mom.pz','rb'))

    from cosmo_pol.utilities import cfg
    cfg.init('/media/wolfensb/Storage/cosmo_pol/option_files/MXPOL_PPI.yml') # Initialize options with 'options_radop.txt'
    cfg.CONFIG['attenuation']['correction'] = 1
    from cosmo_pol.utilities import tictoc
    tictoc.tic()

    c = get_radar_observables(beams, lut)
#    tictoc.toc()
#
    print(10*np.log10(c.values['ZH'][0]))