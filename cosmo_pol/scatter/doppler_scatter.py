# -*- coding: utf-8 -*-

"""doppler_scatter.py: computes all radar observables for a given radial"""

__author__ = "Daniel Wolfensberger"
__copyright__ = "Copyright 2017, COSMO_POL"
__credits__ = ["Daniel Wolfensberger"]
__license__ = "GPL"
__version__ = "1.0"
__maintainer__ = "Daniel Wolfensberger"
__email__ = "daniel.wolfensberger@epfl.ch"


# Global imports
import numpy as np
np.warnings.filterwarnings('ignore')
from scipy.ndimage.filters import gaussian_filter
import copy
from scipy.optimize import fsolve

# Local imports
from cosmo_pol.hydrometeors import create_hydrometeor
from cosmo_pol.utilities import (nansum_arr, nan_cumsum, sum_arr,
                                 nan_cumprod, vlinspace)
from cosmo_pol.interpolation import Radial
from cosmo_pol.config.cfg import CONFIG
from cosmo_pol.scatter import get_refl
from cosmo_pol.constants import global_constants as constants

def proj_vel(U, V, W, vf, theta,phi):
    """
    Gets the radial velocity from the 3D wind field and hydrometeor
    fall velocity
    Args:
        U: eastward wind component [m/s]
        V: northward wind component [m/s]
        W: vertical wind component [m/s]
        vf: terminal fall velocity averaged over all hydrometeors [m/s]
        theta: elevation angle in degrees
        phi: azimuth angle in degrees

    Returns:
        The radial velocity, with reference to the radar beam
        positive values represent flow away from the radar
    """
    return ((U*np.sin(phi) + V * np.cos(phi)) * np.cos(theta)
            + (W - vf) * np.sin(theta))

def get_radar_observables(list_subradials, lut_sz):
    """
    Computes Doppler and polarimetric radar variables for all subradials
    over ensembles of hydrometeors and integrates them over all subradials at
    the end
    Args:
        list_subradials: list of subradials (Radial claass instances) as
            returned by the interpolation.py code
        lut_sz: Lookup tables for all hydrometeor species as returned by the
            load_all_lut function in the lut submodule

    Returns:
        A radial class instance containing the integrated radar observables
    """

    # Get setup
    global doppler_scheme
    global microphysics_scheme

    # Get info from user config
    from cosmo_pol.config.cfg import CONFIG
    doppler_scheme = CONFIG['doppler']['scheme']
    add_turb = CONFIG['doppler']['turbulence_correction']
    add_antenna_motion = CONFIG['doppler']['motion_correction']
    microphysics_scheme = CONFIG['microphysics']['scheme']
    with_ice_crystals = CONFIG['microphysics']['with_ice_crystals']
    melting = CONFIG['microphysics']['with_melting']
    att_corr = CONFIG['microphysics']['with_attenuation']
    radar_type = CONFIG['radar']['type']
    radial_res = CONFIG['radar']['radial_resolution']
    integration_scheme = CONFIG['integration']['scheme']
    KW = CONFIG['radar']['K_squared']

    if radar_type == 'GPM':
        # For GPM no need to simulate Doppler variables
        simulate_doppler = False
    else:
        simulate_doppler = True

    # Get dimensions of subradials
    num_beams = len(list_subradials) # Number of subradials (quad. pts)
    idx_0 = int(num_beams/2) # Index of central subradial
    # Nb of gates in final integrated radial ( = max length of all subradials)
    n_gates = max([len(l.dist_profile) for l in list_subradials])

    # Here we get the list of all hydrometeor types that must be considered
    hydrom_types = []
    hydrom_types.extend(['R','S','G']) # Rain, snow and graupel
    if melting: # IMPORTANT: melting hydrometeors must be placed first
        hydrom_types.extend(['mS','mG'])
    if microphysics_scheme == '2mom':
        hydrom_types.extend(['H']) # Add hail
    if with_ice_crystals:
        hydrom_types.extend('I')

    # Initialize

    # Create dictionnary with all hydrometeor Class instances
    dic_hydro = {}
    for h in hydrom_types:
        dic_hydro[h] = create_hydrometeor(h, microphysics_scheme)
        # Add info on number of bins to use for numerical integrations
        # Needs to be the same as in the lookup tables
        _nbins_D = lut_sz[h].value_table.shape[-2]
        _dmin = lut_sz[h].axes[2][:,0] if h in ['mS','mG'] else lut_sz[h].axes[2][0]
        _dmax = lut_sz[h].axes[2][:,-1] if h in ['mS','mG'] else lut_sz[h].axes[2][-1]

        dic_hydro[h].nbins_D = _nbins_D
        dic_hydro[h].d_max = _dmax
        dic_hydro[h].d_min = _dmin

    # Consider special case of 'ml' quadrature scheme, where most
    # quadrature points are used only near the melting layer edges

    if integration_scheme == 'ml':
        all_weights = np.array([b.quad_weight for b in list_subradials])
        total_weight_at_gates = np.sum(all_weights, axis = 0)

    # Initialize integrated scattering matrix, see lut submodule for info
    # about the 12 columns
    sz_integ = np.zeros((n_gates,len(hydrom_types),12),
                        dtype = 'float32') + np.nan

    # Doppler variables
    if simulate_doppler:
        # average terminal velocity
        rvel_avg = np.zeros(n_gates,)  + np.nan
        if doppler_scheme == 1 or doppler_scheme ==2:
            # total weight where the radial velocity is finite
            total_weight_rvel = np.zeros(n_gates,)
        elif doppler_scheme == 3:
            # Full Doppler spectrum
            doppler_spectrum = np.zeros((n_gates, len(constants.VARRAY)))

    ###########################################################################
    for i, subrad in enumerate(list_subradials): # Loop on subradials (quad pts)
        if simulate_doppler:
            v_integ = np.zeros(n_gates,) # Integrated fall velocity
            n_integ = np.zeros(n_gates,) # Integrated number of particles

            if doppler_scheme == 3:
                # Total attenuation at every subbeam, needed to take into
                # account attenuation
                ah_per_beam = np.zeros((n_gates,),dtype = 'float32') + np.nan

        for j, h in enumerate(hydrom_types): # Loop on hydrometeors

            # If melting mode is on, we skip the melting hydrometeors
            # for the beams where no melting is detected
            if melting:
                if not subrad.has_melting:
                    if h in ['mS','mG']:
                        continue

            """
            Since lookup tables are defined for angles in [0,90], we have to
            check if elevations are larger than 90°, in that case we take
            180-elevation by symmetricity. Also check if angles are smaller
            than 0, in that case, flip sign
            """
            elev_lut = subrad.elev_profile
            elev_lut[elev_lut > 90] = 180 - elev_lut[elev_lut > 90]
            # Also check if angles are smaller than 0, in that case, flip sign
            elev_lut[elev_lut < 0] =- elev_lut[elev_lut < 0]

            T = subrad.values['T']

            '''
            Part 1 : Compute the PSD of the particles
            '''

            QM = subrad.values['Q'+h+'_v'] # Get mass densities
            valid_data = QM > 0
            if not np.isscalar(subrad.quad_weight):
                # Consider only gates where QM > 0 for subradials with non
                # zero weight
                valid_data = np.logical_and(valid_data, subrad.quad_weight > 0)
            if not np.any(valid_data):
                continue # Skip

            # 1 Moment case
            if microphysics_scheme == '1mom':
                if h == 'mG': # For melting graupel, need QM and wet fraction
                    fwet = subrad.values['fwet_'+h]
                    dic_hydro[h].set_psd(QM[valid_data], fwet[valid_data])
                elif h == 'mS': # For melting snow, need T, QM, wet fraction
                    fwet = subrad.values['fwet_'+h]
                    dic_hydro[h].set_psd(T[valid_data],QM[valid_data],
                                   fwet[valid_data])
                elif h in ['S','I'] :
                    # For snow and ice crystals, we need T and QM
                    dic_hydro[h].set_psd(T[valid_data], QM[valid_data])
                else: # Rain and graupel
                    dic_hydro[h].set_psd(QM[valid_data])

            # 2 Moment case
            elif microphysics_scheme == '2mom':
                QN = subrad.values['QN'+h+'_v']  # Get nb concentrations as well
                dic_hydro[h].set_psd(QN[valid_data], QM[valid_data])

            # Get list of diameters for this hydrometeor

            # For melting hydrometeor, diameters depend on wet fraction...
            if h in ['mS','mG']:
                # Number of diameter bins in lookup table
                list_D = vlinspace(dic_hydro[h].d_min, dic_hydro[h].d_max,
                                   dic_hydro[h].nbins_D)

                dD = list_D[:,1] - list_D[:,0]
            else:
                list_D = lut_sz[h].axes[lut_sz[h].axes_names['d']]
                dD = list_D[1] - list_D[0]

            # Compute particle numbers for all diameters
            N = dic_hydro[h].get_N(list_D)

            if len(N.shape) == 1:
                N = np.reshape(N,[len(N),1]) # To be consistent with the einsum dimensions

            '''
            Part 2: Query of the scattering Lookup table
            '''
            # Get SZ matrix
            if h in ['mS','mG']:
                sz = lut_sz[h].lookup_line(e = elev_lut[valid_data],
                                           wc = fwet[valid_data])
            else:
                sz = lut_sz[h].lookup_line(e = elev_lut[valid_data],
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

            if len(valid_data) < n_gates:
                # Check for special cases
                valid_data = np.pad(valid_data,(0,n_gates - len(valid_data)),
                                    mode = 'constant',
                                    constant_values = False)

            if not np.isscalar(subrad.quad_weight):
                weights = (subrad.quad_weight[valid_data] /
                           total_weight_at_gates[valid_data])
                sz_integ[valid_data,j,:] = nansum_arr(sz_integ[valid_data,j,:],
                            weights[:,None] *
                            sz_psd_integ)
            else:
                sz_integ[valid_data,j,:] = nansum_arr(sz_integ[valid_data,j,:],
                                                      sz_psd_integ *
                                                      subrad.quad_weight)
            '''
            Part 4 : Doppler
            '''

            if not simulate_doppler:
                continue # No Doppler info for GPM

            if doppler_scheme == 1:
                # Get terminal velocity integrated over PSD
                vh,n = dic_hydro[h].integrate_V()

                v_integ[valid_data] = nansum_arr(v_integ[valid_data],vh)
                n_integ[valid_data] = nansum_arr(n_integ[valid_data],n)

            elif doppler_scheme == 2:
                # Get PSD integrated rcs at hor. pol.
                rcs = 2*np.pi*(sz[:,:,0] - sz[:,:,1] -
                               sz[:,:,2] + sz[:,:,3])

                # Get terminal velocity
                v_f = dic_hydro[h].get_V(list_D)

                # Integrate terminal velocity over PSD with rcs weighting
                vh_w = np.trapz(np.multiply(v_f, N * rcs), axis=1)
                n_w = np.trapz(N * rcs, axis=1)

                v_integ[valid_data] = nansum_arr(v_integ[valid_data], vh_w)
                n_integ[valid_data] = nansum_arr(n_integ[valid_data], n_w)

            elif doppler_scheme == 3:
                """ Computation of Doppler reflectivities will be done at the
                the loop, but we need to know the attenuation at every gate
                """
                wavelength = constants.WAVELENGTH
                ah = 4.343e-3 * 2 * wavelength * sz_psd_integ[:,11]
                ah *= radial_res/1000. # Multiply by bin length
                ah_per_beam = nansum_arr(ah_per_beam, ah)

        """ For every beam, we get the average fall velocity for
        all hydrometeors and the resulting radial velocity """

        if not simulate_doppler:
            continue

        if doppler_scheme in [1, 2]:
            # Obtain hydrometeor average fall velocity
            v_hydro = v_integ/n_integ
            # Add density weighting
            v_hydro*(subrad.values['RHO']/subrad.values['RHO'][0])**(0.5)

            # Get radial velocity knowing hydrometeor fall speed and U,V,W from model
            theta = np.deg2rad(subrad.elev_profile) # elevation
            phi = np.deg2rad(subrad.quad_pt[0])       # azimuth
            proj_wind = proj_vel(subrad.values['U'],subrad.values['V'],
                               subrad.values['W'], v_hydro, theta,phi)

            # Get mask of valid values
            total_weight_rvel = sum_arr(total_weight_rvel,
                                          ~np.isnan(proj_wind)*subrad.quad_weight)


            # Average radial velocity for all sub-beams
            rvel_avg = nansum_arr(rvel_avg, (proj_wind) * subrad.quad_weight)

        elif doppler_scheme == 3: # Full Doppler
            '''
            NOTE: the full Doppler scheme is kind of badly integrated
            within the overall routine and there are lots of code repetitions
            for exemple RCS are recomputed even though they were computed
            above.
            It could be reimplemented in a better way
            '''
            # Get list of hydrometeors to process
            hydros_to_process = dic_hydro.keys()
            # If melting mode is on, we remove the melting hydrometeors
            # for the beams where no melting is detected
            if melting:
                if not subrad.has_melting:
                    hydros_to_process.remove('mS')
                    hydros_to_process.remove('mG')


            beam_spectrum = get_doppler_spectrum(subrad,
                                                 dic_hydro,
                                                 lut_sz,
                                                 hydros_to_process,
                                                 KW)

            # Account for spectrum spread caused by turbulence and antenna motion
            add_specwidth = np.zeros(len(beam_spectrum))
            if add_turb:
                add_specwidth += spectral_width_turb(constants.RANGE_RADAR,
                                               subrad.values['EDR'])
            if add_antenna_motion:
                add_specwidth += spectral_width_motion(subrad.elev_profile)

            if np.sum(add_specwidth) > 0:
                beam_spectrum = broaden_spectrum(beam_spectrum, add_specwidth)

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

            if not np.isscalar(subrad.quad_weight):
                beam_spectrum *= subrad.quad_weight[:,None] # Multiply by quad weight
            else:
                beam_spectrum *= subrad.quad_weight # Multiply by quad weight


            doppler_spectrum += beam_spectrum

    '''
    Here we derive the final quadrature integrated radar observables after
    integrating all scattering properties over hydrometeor types and
    all subradials
    '''

    sz_integ = np.nansum(sz_integ,axis=1)
    sz_integ[sz_integ == 0] = np.nan

    # Get radar observables
    ZH, ZV, ZDR, RHOHV, KDP, AH, AV, DELTA_HV = get_pol_from_sz(sz_integ, KW)

    PHIDP = nan_cumsum(2 * KDP) * radial_res/1000. + DELTA_HV

    ZV_ATT = ZV.copy()
    ZH_ATT = ZH.copy()

    if att_corr:
        # AH and AV are in dB so we need to convert them to linear
        ZV_ATT *= nan_cumprod(10**(-0.2*AV*(radial_res/1000.))) # divide to get dist in km
        ZH_ATT *= nan_cumprod(10**(-0.2*AH*(radial_res/1000.)))
        ZDR = ZH_ATT / ZV_ATT


    if simulate_doppler:
        if doppler_scheme in [1, 2]:
            rvel_avg /= total_weight_rvel

        elif doppler_scheme == 3:
            try:
                rvel_avg = np.nansum(np.tile(constants.VARRAY,
                                         (n_gates,1))*doppler_spectrum,
                                         axis=1)
                rvel_avg /= np.nansum(doppler_spectrum,axis=1)
            except:
                rvel_avg *= np.nan
    ###########################################################################

    '''
    Create the final Radial class instance containing all radar observables
    '''

    # Create outputs
    rad_obs = {}
    rad_obs['ZH'] = ZH
    rad_obs['ZDR'] = ZDR
    rad_obs['ZV'] = ZV
    rad_obs['KDP'] = KDP
    rad_obs['DELTA_HV'] = DELTA_HV
    rad_obs['PHIDP'] = PHIDP
    rad_obs['RHOHV'] = RHOHV
    # Add attenuation at every gate
    rad_obs['ATT_H'] = 10*np.log10(ZH) - 10*np.log10(ZH_ATT) # In dB
    rad_obs['ATT_V'] = 10*np.log10(ZV) - 10*np.log10(ZV_ATT) # In dB

    if simulate_doppler:
        rad_obs['RVEL'] = rvel_avg
        if doppler_scheme == 3:
            rad_obs['DSPECTRUM'] = doppler_spectrum
    '''
    Once averaged , the meaning of the mask is the following
    mask == -1 : all beams are below topography
    mask == 1 : all beams are above COSMO top
    mask > 0 : at least one beam is above COSMO top
    We will keep only gates where no beam is above COSMO top and at least
    one beam is above topography
    '''

    # Sum the mask of all beams to get overall average mask
    mask = np.zeros(n_gates,)
    for i,beam in enumerate(list_subradials):
        mask = sum_arr(mask,beam.mask[0:n_gates], cst = 1) # Get mask of every Beam

    mask /= float(num_beams)
    mask[np.logical_and(mask > -1, mask <= 0)] = 0

    # Finally get vectors of distances, height and lat/lon at the central beam
    heights_radar = list_subradials[idx_0].heights_profile
    distances_radar = list_subradials[idx_0].dist_profile
    lats = list_subradials[idx_0].lats_profile
    lons = list_subradials[idx_0].lons_profile

    # Create final radial
    radar_radial = Radial(rad_obs, mask, lats, lons, distances_radar,
                          heights_radar)

    return radar_radial

def get_pol_from_sz(sz, KW):
    '''
    Computes polarimetric radar observables from integrated scattering properties
    Args:
        sz: integrated scattering matrix, with an arbitrary number of rows
            (gates) and 12 columns (seet lut submodule)
        KW: the refractive factor of water, usually 0.93 for radar applications

    Returns:
         z_h: radar refl. factor at hor. pol. in linear units [mm6 m-3]
         z_v: radar refl. factor at vert. pol. in linear units [mm6 m-3]
         zdr: diff. refl. = z_h / z_v [-]
         rhohv: copolar. corr. coeff [-]
         kdp: spec. diff. phase shift upon propagation [deg km-1]
         ah: spec. att. at hor. pol. [dB km-1]
         av: spec. att. at vert. pol. [dB km-1]
         delta_hv: total phase shift upon backscattering [deg]
    '''

    wavelength = constants.WAVELENGTH

    from cosmo_pol.config.cfg import CONFIG
    K_squared = CONFIG['radar']['K_squared']

    # Horizontal reflectivity
    radar_xsect_h = 2*np.pi*(sz[:,0]-sz[:,1]-sz[:,2]+sz[:,3])
    z_h = wavelength**4/(np.pi**5*K_squared)*radar_xsect_h

    # Vertical reflectivity
    radar_xsect_v = 2*np.pi*(sz[:,0]+sz[:,1]+sz[:,2]+sz[:,3])
    z_v = wavelength**4/(np.pi**5*K_squared)*radar_xsect_v

    # Differential reflectivity
    zdr = radar_xsect_h/radar_xsect_v

    # Differential phase shift
    kdp = 1e-3 * (180.0/np.pi) * wavelength * (sz[:,10]-sz[:,8])

    # Attenuation
    ext_xsect_h = 2 * wavelength * sz[:,11]
    ext_xsect_v = 2 * wavelength * sz[:,9]
    ah = 4.343e-3 * ext_xsect_h
    av = 4.343e-3 * ext_xsect_v

    # Copolar correlation coeff.
    a = (sz[:,4] + sz[:,7])**2 + (sz[:,6] - sz[:,5])**2
    b = (sz[:,0] - sz[:,1] - sz[:,2] + sz[:,3])
    c = (sz[:,0] + sz[:,1] + sz[:,2] + sz[:,3])
    rhohv = np.sqrt(a / (b*c))

    # Backscattering differential phase
    delta_hv = np.arctan2(sz[:,5] - sz[:,6], -sz[:,4] - sz[:,7])

    return z_h,z_v,zdr,rhohv,kdp,ah,av,delta_hv

def get_diameter_from_rad_vel(dic_hydro, phi, theta, U, V, W, rho_corr):
    '''
    Retrieves the diameters corresponding to all radial velocity bins, by
    getting the corresponding terminal velocity and inverting the
    diameter-velocity relations
    Args:
        dic_hydro: a dictionary containing the hydrometeor Class instances
        phi: the azimuth angle in degrees
        theta: the elevation angle in degrees
        U: the eastward wind component of COSMO
        V: the northward wind component of COSMO
        W: the vertical wind component of COSMO
        rho_corr: the correction for density (rho / rho_0)
    Returns:
         Da: the diameters corresponding to the left edge of all velocity bins
         Db: the diameters corresponding to the right edge of all velocity bins
         idx: the indices of the radar gates where Da and Db are valid values
             (i.e. between dmin and dmax)

    '''
    theta = np.deg2rad(theta)
    phi = np.deg2rad(phi)

    wh = (1./rho_corr * (W + (U * np.sin(phi) +
          V * np.cos(phi)) / np.tan(theta) - constants.VARRAY / np.sin(theta)))

    idx = np.where(wh>=0)[0] # We are only interested in positive fall speeds

    wh = wh[idx]

    hydrom_types = dic_hydro.keys()

    D = np.zeros((len(idx), len(hydrom_types)), dtype='float32')

    # Get D bins from V bins
    for i,h in enumerate(dic_hydro.keys()): # Loop on hydrometeors
        D[:,i] = dic_hydro[h].get_D_from_V(wh)
        # Threshold to valid diameters
        D[D >= dic_hydro[h].d_max] = dic_hydro[h].d_max
        D[D <= dic_hydro[h].d_min] = dic_hydro[h].d_min

    # Array of left bin limits
    Da = np.minimum(D[0:-1,:], D[1:,:])
    # Array of right bin limits
    Db = np.maximum(D[0:-1,:], D[1:,:])

    # Get indice of lines where at least one bin width is larger than 0
    mask = np.where(np.sum((Db-Da) == 0.0, axis = 1) < len(hydrom_types))[0]

    return Da[mask,:], Db[mask,:], idx[mask]


def get_doppler_spectrum(subrad, dic_hydro, lut_sz, hydros_to_process, KW = 0.93):
    '''
    Computes the reflectivity within every bin of the Doppler spectrum
    Args:
        subrad: a subradial, containing all necessary info
        dic_hydro: a dictionary containing the hydrometeor Class instances
        lut_sz: dictionary containing the lookup tables of scattering
            properties for every hydrometeor type
        hydros_to_process: list of hydrometeors that effectively need to
            be considered, for example if no melting is occuring on the
            subradial, mS and mG will not be in hydros_to_process, even
            thought they might be keys in dic_hydro
        KW: the refractive factor of water, usually 0.93 for radar applications
    Returns:
         refl: array of size [n_gates, len_FFT] containing the reflectivities
             at every range gate and for every velocity bin
    '''

    # Get dimensions
    n_gates = len(subrad.dist_profile) # Number of radar gates

    # Initialize matrix of reflectivities
    refl = np.zeros((n_gates, len(constants.VARRAY)), dtype='float32')

    # Get all elevations
    elev = subrad.elev_profile
    # Since lookup tables are defined for angles >0, we have to check
    # if angles are larger than 90°, in that case we take 180-elevation
    # by symmetricity
    elev_lut = copy.deepcopy(elev)
    elev_lut[elev_lut>90] = 180 - elev_lut[elev_lut>90]
    # Also check if angles are smaller than 0, in that case, flip sign
    elev_lut[elev_lut<0] = - elev_lut[elev_lut<0]

    # Get azimuth angle
    phi = subrad.quad_pt[0]

    # Correction of velocity for air density
    rho_corr = (subrad.values['RHO']/subrad.values['RHO'][0])**0.5

    for i in range(n_gates):  # Loop on all radar gates
        if subrad.mask[i] == 0:
            if not np.isscalar(subrad.quad_weight):
                if subrad.quad_weight[i] == 0:
                    continue
            # Get parameters of the PSD (lambda or lambda/N0) for present hydrom
            # meteors

            dic_hydrom_gate = {}

            for j,h in enumerate(hydros_to_process):
                Q = subrad.values['Q'+h+'_v'][i]
                T = subrad.values['T'][i]
                if Q > 0:
                    dic_hydrom_gate[h] = dic_hydro[h]

                    if microphysics_scheme == '1mom':
                        if h =='mG':
                            fwet = subrad.values['fwet_'+h][i]
                            dic_hydrom_gate[h].set_psd(np.array([Q]),
                                                 np.array([fwet]))
                        elif h == 'mS':
                            fwet = subrad.values['fwet_'+h][i]
                            dic_hydrom_gate[h].set_psd(np.array([T]),
                                                 np.array([Q]),
                                                 np.array([fwet]))
                        elif h in ['S','I']:
                            dic_hydrom_gate[h].set_psd(np.array([T]),
                                                        np.array([Q]))
                        else:
                            dic_hydrom_gate[h].set_psd(np.array([Q]))

                    elif microphysics_scheme == '2mom':
                        QN = subrad.values['QN'+h+'_v'][i]
                        dic_hydrom_gate[h].set_psd(np.array([QN]),
                                                    np.array([Q]))


            # Initialize matrix of radar cross sections, N and D
            n_hydrom = len(dic_hydrom_gate.keys())
            n_d_bins = lut_sz[h].value_table.shape[-2]
            rcs = np.zeros((n_d_bins, n_hydrom),dtype='float32') + np.nan
            N = np.zeros((n_d_bins, n_hydrom),dtype='float32') + np.nan
            D = np.zeros((n_d_bins, n_hydrom),dtype='float32') + np.nan
            D_min = np.zeros((n_hydrom),dtype = 'float32') + np.nan
            step_D = np.zeros((n_hydrom),dtype = 'float32') + np.nan

            # Get N and D for all hydrometeors that are present
            for j,h in enumerate(dic_hydrom_gate.keys()):
                D[:,j] = np.linspace(dic_hydrom_gate[h].d_min,
                                     dic_hydrom_gate[h].d_max,
                                     dic_hydrom_gate[h].nbins_D)

                D_min[j] = D[0,j]
                step_D[j] = D[1,j] - D[0,j]
                N[:,j] = dic_hydrom_gate[h].get_N(D[:,j])

            # Compute RCS for all hydrometeors that are present
            for j,h in enumerate(dic_hydrom_gate.keys()):
                if h in ['mS','mG']:
                    sz = lut_sz[h].lookup_line(e = elev_lut[i],
                               wc = subrad.values['fwet_'+h][i])
                else:
                    sz = lut_sz[h].lookup_line(e = elev_lut[i],
                                               t = subrad.values['T'][i])
                # get RCS
                rcs[:,j]= (2*np.pi*(sz[:,0] - sz[:,1] - sz[:,2] + sz[:,3])).T


            # Important we use symetrical elevations only for lookup querying, not
            # for actual trigonometrical velocity estimation
            Da, Db, idx = get_diameter_from_rad_vel(dic_hydrom_gate, phi,
                          elev[i],subrad.values['U'][i],
                          subrad.values['V'][i],
                          subrad.values['W'][i],
                          rho_corr[i])
            try:
                refl[i,idx] = get_refl(len(idx), Da,
                                Db, rcs, D, N, step_D, D_min)[1]


                wavelength = constants.WAVELENGTH
                refl[i,idx] *= wavelength**4/(np.pi**5*KW**2)
            except:
                print('An error occured in the Doppler spectrum calculation...')
                raise

    return refl

def spectral_width_turb( ranges, EDR):
    '''
    Computes the spectral width caused by turbulence, according to
    Doviak and Zrnic

    Args:
        ranges: the distance to every radar gate along the subradial
        EDR: the eddy dissipitation rate at every radar gate,
            is needed for the turbulence correction, so should be in the
            provided COSMO variables
    Returns:
         std_turb: the spectral width (stdev) caused by turbulence
    '''

    sigma_r = 0.35 * CONFIG['radar']['radial_resolution']
    sigma_theta = (np.deg2rad(CONFIG['radar']['3dB_beamwidth']) /
                   (4.*np.sqrt(np.log(2))))

    std_turb = np.zeros((len(EDR),))

    # Method of calculation follow Doviak and Zrnic (p.409)
    # Case 1 : sigma_r << r*sigma_theta
    idx_r = sigma_r<0.1*ranges*sigma_theta
    std_turb[idx_r] = ((ranges[idx_r]*EDR[idx_r]*sigma_theta*
                        constants.A**(3/2.))/0.72)**(1/3.)
    # Case 2 : r*sigma_theta <= sigma_r
    idx_r = sigma_r >= 0.1*ranges*sigma_theta
    std_turb[idx_r]= (((EDR[idx_r]*sigma_r*(1.35*constants.A)**(3/2))/
            (11./15.+4./15.*(ranges[idx_r]*sigma_theta/sigma_r)**2)
            **(-3/2.))**(1/3.))

    return std_turb

def spectral_width_motion(elevations):
    '''
    Computes the spectral width caused by antenna motion, according to
    Doviak and Zrnic

    Args:
        elevations: elevation angles (theta) in degrees for every gate along
    Returns:
         std_motion: the spectral width (stdev) caused by antenna motion
    '''
    wavelength = constants.WAVELENGTH/100. # Get it in m
    bandwidth_3dB = CONFIG['radar']['3dB_beamwidth']
    ang_vel = CONFIG['radar']['antenna_speed']

    std_motion = ((wavelength * ang_vel * np.cos(np.deg2rad(elevations))) /
                    (2 * np.pi * np.deg2rad(bandwidth_3dB)))

    return std_motion

def broaden_spectrum(spectrum, std):
    '''
    Broadens the spectrum with the specified standard deviations, by
    applying a gaussian filter
    Args:
        spectrum: the Doppler spectrum as a 2D array, range and bin
        std: the list of standard  deviations
    Returns:
         the Gaussian filtered (broadened) spectrum
    '''
    v = constants.VARRAY
    # Convolve spectrum and turbulence gaussian distributions
    # Get resolution in velocity
    v_res=v[2]-v[1]

    original_power = np.sum(spectrum,1) # Power of every spectrum (at all radar gates)
    for i,t in enumerate(std):
    	spectrum[i,:] = gaussian_filter(spectrum[i,:],t/v_res) # Filter only columnwise (i.e. on velocity bins)
    convolved_power=np.sum(spectrum,1) # Power of every convolved spectrum (at all radar gates)

    spectrum = spectrum/convolved_power[:,None] * original_power[:,None]# Rescale to original power

    return spectrum


def cut_at_sensitivity(list_subradials):
    '''
    Censors simulated measurements where the reflectivity falls below the
    sensitivity specified by the user, see the wiki for how to define
    the sensitivity in the configuration files
    Args:
        list_subradials: a list of subradials containing the computed radar
            observables
    Returns:
         the list_subradials but censored with the radar sensitivity
    '''
    from cosmo_pol.config.cfg import CONFIG
    sens_config = CONFIG['radar']['sensitivity']
    if not isinstance(sens_config,list):
        sens_config = [sens_config]

    if len(sens_config) == 3: # Sensitivity - gain - snr
        threshold_func = lambda r: (sens_config[0] + constants.RADAR_CONSTANT_DB
                         + sens_config[2] + 20*np.log10(r/1000.))
    elif len(sens_config) == 2: # ZH - range
        threshold_func = lambda r: ((sens_config[0] -
                                     20 * np.log10(sens_config[1] / 1000.)) +
                                     20 * np.log10(r / 1000.))
    elif len(sens_config) == 1: # ZH
        threshold_func = lambda r: sens_config[0]

    else:
        print('Sensitivity parameters are invalid, cannot cut at specified sensitivity')
        print('Enter either a single value of refl (dBZ) or a list of',
              '[refl (dBZ), distance (m)] or a list of [sensitivity (dBm), gain (dBm) and snr (dB)]')

        return list_subradials

    if isinstance(list_subradials[0],list): # Loop on list of lists
        for i,sweep in enumerate(list_subradials):
            for j,b, in enumerate(sweep):
                rranges = (CONFIG['radar']['radial_resolution'] *
                           np.arange(len(b.dist_profile)))
                mask = 10*np.log10(b.values['ZH']) < threshold_func(rranges)
                for k in b.values.keys():
                    if k in constants.SIMULATED_VARIABLES:
                        if k == 'DSPECTRUM':
                            logspectrum = 10 * np.log10(list_subradials[i][j].values[k])
                            thresh = threshold_func(rranges)
                            thresh =  np.tile(thresh,
                                              (logspectrum.shape[1],1)).T
                            list_subradials[i][j].values[k][logspectrum < thresh] = np.nan
                        else:
                            list_subradials[i][j].values[k][mask] = np.nan

    else:
        for i, subradial in enumerate(list_subradials): # Loop on simple list
            rranges = (CONFIG['radar']['radial_resolution'] *
                       np.arange(len(subradial.dist_profile)))
            mask = 10 * np.log10(subradial.values['ZH']) < threshold_func(rranges)
            for k in subradial.values.keys():
                if k in constants.SIMULATED_VARIABLES:
                    list_subradials[i].values[k][mask] = np.nan
    return list_subradials

