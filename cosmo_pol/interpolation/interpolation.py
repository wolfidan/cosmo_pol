# -*- coding: utf-8 -*-

"""interpolatiom.py: Provides routines for the trilinear interpolation of
COSMO variables to the radar gates"""

__author__ = "Daniel Wolfensberger"
__copyright__ = "Copyright 2017, COSMO_POL"
__credits__ = ["Daniel Wolfensberger"]
__license__ = "GPL"
__version__ = "1.0"
__maintainer__ = "Daniel Wolfensberger"
__email__ = "daniel.wolfensberger@epfl.ch"


# Global imports
import numpy as np
np.seterr(divide='ignore') # Disable divide by zero error
import pyproj
import pickle
import pycosmo as pc
import scipy.interpolate as interp
from scipy.ndimage import gaussian_filter
from textwrap import dedent


# Local imports
from cosmo_pol.interpolation import (get_GPM_refraction,
                                     compute_trajectory_radial,
                                     Radial, melting,
                                     gautschi_points_and_weights,
                                     get_all_radar_pts)

from cosmo_pol.config import CONFIG
from cosmo_pol.constants import global_constants as constants
from cosmo_pol.utilities import nansum_arr, sum_arr, vector_1d_to_polar

def integrate_radials(list_subradials):
    '''
    Integrates a set of radials corresponding to different quadrature points
    Args:
        list_subradials: list of Radial class instances corresponding to
            different subradials

    Returns:
        integrated_radial: an integrated Radial class instance

    '''

    num_subradials = len(list_subradials)
    list_variables = list_subradials[0].values.keys()

    integrated_variables={}
    for k in list_variables:
        integrated_variables[k] = np.nan
        sum_weights=0
        for i in list_subradials:
            sum_weights+=i.GH_weight
        for i in list_subradials:
            integrated_variables[k] = nansum_arr(integrated_variables[k],
                                      i.values[k] * i.GH_weight / sum_weights)

    # Get index of central beam
    idx_0 = int(num_subradials/2)

    # Sum the mask of all beams to get overall average mask
    mask = np.zeros(num_subradials,)
    for i,p in enumerate(list_subradials):
        mask = sum_arr(mask, p.mask)

    '''
    Once averaged , the meaning of the mask is the following
    mask == -1 : all subradials are below topography
    mask == 1 : all subradials are above COSMO top
    mask > 0 : at least one subradials is above COSMO top
    We will keep only gates where no subradials is above COSMO top and at least
    one subradials is above topography
    '''
    mask /= float(num_subradials)
    mask[np.logical_and(mask > -1, mask <= 0)] = 0

    heights_radar = list_subradials[idx_0].heights_profile
    distances_radar = list_subradials[idx_0].dist_profile
    lats = list_subradials[idx_0].lats_profile
    lons = list_subradials[idx_0].lons_profile

    # Create new beam with summed up variables
    integrated_radial = Radial(integrated_variables, mask, lats, lons,
                             distances_radar, heights_radar)

    return integrated_radial

def get_interpolated_radial(dic_variables, azimuth, elevation, N = None,
                            list_refraction = None):
    '''
    Interpolates a radar radial using a specified quadrature and outputs
    a list of subradials
    Args:
        dic_variables: dictionary containing the COSMO variables to be
            interpolated
        azimuth: the azimuth angle in degrees (phi) of the radial
        elevation: the elevation angle in degrees (theta) of the radial
        N : if the differential refraction scheme by Zeng and Blahak (2014) is
            used, the refractivity of the atmosphere must be provided as an
            additional COSMO variable
        list_refraction : To save time, a list of (s,h,e) tuples corresponding
            to the dist at ground, height above ground and incident elev. ang.
            for all quadrature points (output of atmospheric refraction)
            can be provided, in which case the atmospheric refraction will
            not be recalculated. This should be done only if the elevation
            angle is the same from one interpolated radial to the other
            (PPI). Also this is not possible for quadrature schemes 2 and 6
            which have irregular grids.
    Returns:
        list_subradials: a list of Radial class instances containing all
            subradials corresponding to all quadrature points
                         defined along the specified radial
        list_refraction: outputs all refraction tuples (s,h,e) computed for the
            subradials, can be used in the next call to the
            get_interpolated_radial function, but ONLY if the
            elevation angle is conserved!
    '''

    list_variables=dic_variables.values()
    keys=dic_variables.keys()

    # Get options
    bandwidth_3dB = CONFIG['radar']['3dB_beamwidth']
    integration_scheme = CONFIG['integration']['scheme']
    refraction_method = CONFIG['refraction']['scheme']
    has_melting = CONFIG['microphysics']['with_melting']

    list_variables = dic_variables.values()
    keys = dic_variables.keys()

    if integration_scheme == 2 and N == None:
        msg = """
        When using integration scheme 2 (Zeng and Blahak, 2014), you must
        provide the refractivity of the atmosphere (N) as an additional variable
        4/3 Earth model will be used instead...(integration scheme 1)
        """
        print(dedent(msg))
        integration_scheme = 1

    # Calculate quadrature weights
    if integration_scheme == 1: # Classical single gaussian scheme
        nh_GH = int(CONFIG['integration']['nh_GH'])
        nv_GH = int(CONFIG['integration']['nv_GH'])

        # Get GH points and weights
        sigma = bandwidth_3dB/(2*np.sqrt(2*np.log(2)))

        pts_hor, weights_hor=np.polynomial.hermite.hermgauss(nh_GH)
        pts_hor = pts_hor*sigma

        pts_ver, weights_ver=np.polynomial.hermite.hermgauss(nv_GH)
        pts_ver = pts_ver*sigma

        weights = np.outer(weights_hor*sigma,weights_ver*sigma)
        weights *= np.abs(np.cos(np.deg2rad(pts_ver)))
        sum_weights = np.sum(weights.ravel())
        weights /= sum_weights # Normalize weights

        beam_broadening=nh_GH>1 or nv_GH>1 # Boolean for beam-broadening (if only one GH point : No beam-broadening)

    elif integration_scheme == 'ml': # Method with oversampling in ml
        nh_GH = int(CONFIG['integration']['nh_GH'])
        nv_GH = int(CONFIG['integration']['nv_GH'])
        nv_GH_ml = 10 * nv_GH
        if not nv_GH_ml % 2:
            nv_GH_ml += 1 # use an odd number

        # Get GH points and weights
        sigma = bandwidth_3dB/(2*np.sqrt(2*np.log(2)))

        pts_hor, weights_hor = np.polynomial.hermite.hermgauss(nh_GH)
        pts_hor = pts_hor*sigma

        pts_ver, weights_ver = np.polynomial.hermite.hermgauss(nv_GH_ml)
        pts_ver = pts_ver*sigma

        # Sort by weights
        idx_sort = np.argsort(weights_ver)[::-1]
        weights_ver = weights_ver[idx_sort]
        pts_ver = pts_ver[idx_sort]

        weights = np.outer(weights_hor * sigma, weights_ver*sigma)
        weights *= np.abs(np.cos(np.deg2rad(pts_ver)))

        sum_weights = np.sum(weights.ravel())
        weights /= sum_weights # Normalize weights

        beam_broadening = nh_GH>1 or nv_GH>1 # Boolean for beam-broadening (if only one GH point : No beam-broadening)

    elif integration_scheme == 2: # Improved multi-gaussian scheme
        nr_GH = int(CONFIG['integration']['nr_GH'])
        na_GL= int(CONFIG['integration']['na_GL'])

        antenna_params = CONFIG['integration']['antenna_params']

        pts_ang, w_ang = np.polynomial.legendre.leggauss(na_GL)
        pts_rad, w_rad = np.polynomial.hermite.hermgauss(nr_GH)

        a_dB = antenna_params[:,0]
        mu = antenna_params[:,1]
        sigma = antenna_params[:,2]

        list_pts = []
        weights = []
        sum_weights = 0

        for i in range(nr_GH):
            for j in range(len(sigma)):
                for k in range(na_GL):
                    r = mu[j]+np.sqrt(2)*sigma[j]*pts_rad[i]
                    theta = np.pi * pts_ang[k] + np.pi
                    weight = (np.pi * w_ang[k] * w_rad[i] * 10 ** (0.1*a_dB[j])
                        * np.sqrt(2) * sigma[j] * abs(r)) # Laplacian
                    weight *= np.cos(r*np.sin(theta))
                    weights.append(weight)

                    sum_weights += weight

                    list_pts.append([r*np.cos(theta)+azimuth,
                                     r*np.sin(theta)+elevation])

        weights = np.array(weights)
        weights /= sum_weights # Normalize weights

    elif integration_scheme == 3: # Gauss-Legendre with real-antenna weighting
        nh_GH = int(CONFIG['integration']['nh_GH'])
        nv_GH = int(CONFIG['integration']['nv_GH'])

        antenna = np.genfromtxt(CONFIG['integration']['antenna_diagram'],
                                delimiter=',')

        angles =  antenna[:,0]
        power_sq =  (10**(0.1*antenna[:,1]))**2
        bounds = np.max(angles)

        pts_hor, weights_hor=np.polynomial.legendre.leggauss(nh_GH)
        pts_hor=pts_hor*bounds

        pts_ver, weights_ver=np.polynomial.legendre.leggauss(nv_GH)
        pts_ver=pts_ver*bounds

        power_sq_pts = (vector_1d_to_polar(angles,power_sq,pts_hor,
                                                     pts_ver).T)
        weights = power_sq_pts * np.outer(weights_hor,weights_ver)
        weights *= np.abs(np.cos(np.deg2rad(pts_ver)))
        weights *= 2*bounds

        sum_weights = np.sum(weights.ravel())
        weights /= sum_weights # Normalize weights

        beam_broadening=nh_GH>1 or nv_GH>1 # Boolean for beam-broadening (if only one GH point : No beam-broadening)

    elif integration_scheme == 4: # Real antenna, for testing only

        data = pickle.load(open('/storage/cosmo_pol/tests/real_antenna_ss.p',
                                'rb'))
        angles = data['angles']
        power_squ = (data['data'].T)**2

        pts_hor = angles
        pts_ver = angles

        beam_broadening = True # In this scheme we always consider several beams

        weights = power_squ * np.abs(np.cos(np.deg2rad(pts_ver)))
        sum_weights = np.sum(weights)
        weights /= sum_weights # Normalize weights

    elif integration_scheme == 5: # Discrete Gautschi quadrature
        nh_GA = int(CONFIG['integration']['nh_GH'])
        nv_GA = int(CONFIG['integration']['nv_GH'])

        antenna = np.genfromtxt(CONFIG['integration']['antenna_diagram'],
                                delimiter=',')

        angles =  antenna[:,0]
        power_sq =  (10**(0.1*antenna[:,1]))**2
        bounds = np.max(angles)

        antenna_fit = interp.interp1d(angles,power_sq,fill_value=0)

        # Add cosinus weighting (in the vertical)
        antenna_fit_weighted = interp.interp1d(angles,
               power_sq*np.cos(np.abs(np.deg2rad(angles))),fill_value=0)

        pts_hor, weights_hor = gautschi_points_and_weights(antenna_fit, -bounds,
                                                      bounds,nh_GA)
        pts_ver, weights_ver = gautschi_points_and_weights(antenna_fit_weighted,
                                                      -bounds,bounds,nv_GA)

        weights = np.outer(weights_hor, weights_ver)
        sum_weights = np.sum(weights.ravel())
        weights /= sum_weights # Normalize weights

        # Boolean for beam-broadening (if only one point : No beam-broadening)
        beam_broadening=nh_GA>1 or nv_GA>1

    elif integration_scheme == 6: # Sparse Gauss-Hermite
        try:
            from SpectralToolbox import SparseGrids
        except:
            msg =  """
            Could not find SpectralToolbox, which is required for integration
            method 6, please make sure it is installed
            Aborting...
            """
            raise ImportError(dedent(msg))

        nh_GH = int(CONFIG['integration']['nh_GH'])
        nv_GH = int(CONFIG['integration']['nv_GH'])
        # Get GH points and weights
        sigma=bandwidth_3dB/(2*np.sqrt(2*np.log(2)))

        grid = SparseGrids.SparseGrid(dim=2,qrule=SparseGrids.GQN,
                                      k=int(CONFIG['integration']['nh_GH']),
                                      sym=True)

        XF,W = grid.sparseGrid()
        W = np.squeeze(W)
        XF *= sigma

        weights = W
        weights *= np.abs(np.cos(np.deg2rad(XF[:,1]))) * sigma

        pts_hor = XF[:,0] + azimuth
        pts_ver = XF[:,1] + elevation
        list_pts = [pt for pt in zip(pts_hor,pts_ver)]

        sum_weights = np.sum(weights)
        weights /= sum_weights # Normalize weights

        # Boolean for beam-broadening (if only one point : No beam-broadening)
        beam_broadening=nh_GH>1 or nv_GH>1

    # Keep only weights above threshold
    if integration_scheme != 6:
        weights_sort = np.sort(np.array(weights).ravel())[::-1] # Desc. order

        weights_cumsum = np.cumsum(weights_sort/np.sum(weights_sort))
        weights_cumsum[-1] = 1. # Avoid floating precision issues
        idx_above = np.where(weights_cumsum >=
                             CONFIG['integration']['weight_threshold'])[0][0]

        threshold = weights_sort[idx_above]
        sum_weights = np.sum(weights_sort[weights_sort>=threshold])
    else:
        threshold = -np.inf

    # Initialize list of subradials
    list_subradials = []
    # create vector of bin positions
    rranges = constants.RANGE_RADAR

    if integration_scheme not in [2,6]: # Only regular grids!
        if list_refraction == None: # Calculate refraction for vertical GH points
            list_refraction = []

            # Get coordinates of virtual radar
            radar_pos = CONFIG['radar']['coords']

            for pt in pts_ver:
                if CONFIG['radar']['type'] == 'GPM':
                    s, h, e = get_GPM_refraction(pt+elevation)
                else:
                    s, h, e = compute_trajectory_radial(
                                                  rranges,
                                                  pt+elevation,
                                                  radar_pos,
                                                  refraction_method,
                                                  N
                                                  )
                list_refraction.append((s, h, e))

        for i in range(len(pts_hor)):
            for j in range(len(pts_ver)):
                if weights[i,j] >= threshold or not beam_broadening:

                    # GH coordinates
                    pt = [pts_hor[i]+azimuth, pts_ver[j]+elevation]
                    # Interpolate beam
                    lats,lons,list_vars = trilin_interp_radial(list_variables,
                                                      pts_hor[i]+azimuth,
                                                      list_refraction[j][0],
                                                      list_refraction[j][1])

                    weight = weights[i,j]

                    # Create dictionary of beams
                    dic_beams={}
                    # Loop on interpolated variables
                    for k, bi in enumerate(list_vars):
                        # Do this only for the first variable
                        # (same mask for all variables)
                        if k == 0:
                            '''
                            mask = 1 : interpolated pt is above COSMO top
                            mask = -1 : intepolated pt is below topography
                            mask = 0 : interpolated pt is ok
                            '''
                            mask_beam = np.zeros((len(bi)))
                            mask_beam[bi == -9999] = 1
                            mask_beam[np.isnan(bi)] = -1
                        bi[mask_beam!=0] = np.nan # Assign NaN to all missing data
                        dic_beams[keys[k]] = bi # Create dictionary

                    subradial = Radial(dic_beams, mask_beam, lats, lons,
                                           list_refraction[j][0],
                                           list_refraction[j][1],
                                           list_refraction[j][2],
                                           pt, weight)

                    # If both QS and/or QG coexist with QR : perform melting
                    if has_melting:
                        subradial = melting(subradial)

                    if integration_scheme == 'ml':

                        if j > nv_GH:
                            smoothing_mask = np.zeros(len(lats))
                            idx_ml = np.where(subradial.mask_ml)[0]
                            if len(idx_ml):
                                smoothing_mask[idx_ml[0]] = 1
                                smoothing_mask[idx_ml[-1]] = 1
                                smoothing_mask = gaussian_filter(smoothing_mask,
                                                                 2)

                            subradial.quad_weight *= smoothing_mask
                        else:
                            subradial.quad_weight *= np.ones(len(lats))

                    list_subradials.append(subradial)

    else:
        # create vector of bin positions
        # Get coordinates of virtual radar

        radar_pos = CONFIG['radar']['coords']

        for i in range(len(list_pts)):
            if weights[i] >= threshold:

                if CONFIG['radar']['type'] == 'GPM':
                    s, h, e = get_GPM_refraction(list_pts[i][1])
                else:
                    s, h, e = compute_trajectory_radial(rranges,
                                                        list_pts[i][1],
                                                        radar_pos,
                                                        refraction_method,
                                                        N)

                lats,lons,list_vars = trilin_interp_radial(list_variables,
                                                          list_pts[i][0],
                                                          s,
                                                          h)
                # Create dictionary of beams
                dic_beams={}
                # Loop on interpolated variables
                for k, bi in enumerate(list_vars):
                    # Do this only for the first variable
                    # (same mask for all variables)

                    if k == 0:
                        '''
                        mask = 1 : interpolated pt is above COSMO top
                        mask = -1 : intepolated pt is below topography
                        mask = 0 : interpolated pt is ok
                        '''
                        mask_beam=np.zeros((len(bi)))
                        mask_beam[bi==-9999]= 1
                        mask_beam[np.isnan(bi)]= -1
                    bi[mask_beam!=0]=float('nan') # Assign NaN to all missing data
                    dic_beams[keys[k]]=bi # Create dictionary

                subradial = Radial(dic_beams,
                              mask_beam,
                              lats,
                              lons,
                              s,
                              h,
                              e,
                              list_pts[i],
                              weights[i])

                if has_melting:
                    subradial = melting(subradial)

                list_subradials.append(subradial)

    return list_subradials, list_refraction

def trilin_interp_radial(list_vars, azimuth, distances_profile, heights_profile):

    """
    Interpolates a radar radial using a specified quadrature and outputs
    a list of subradials
    Args:
        list_vars: list of COSMO variables to be interpolated
        azimuth: the azimuth angle in degrees (phi) of the subradial
        distances_profile: vector of distances in meters of all gates
            along the subradial (computed with the atmospheric refraction
            scheme)
        heights_profile: vector of heights above ground in meters of all
            gates along the subradial (computed with the atmospheric refraction
            scheme)
    Returns:
        lats_rad: vector of all latitudes along the subradial
        lons_rad: vector of all longitudes along the subradial
        interp_data: dictionary containing all interpolated variables along
            the subradial
    """



    # Get position of virtual radar from user configuration
    radar_pos = CONFIG['radar']['coords']

    # Initialize WGS84 geoid
    g = pyproj.Geod(ellps='WGS84')

    # Get radar bins coordinates
    lons_rad=[]
    lats_rad=[]
    # Using the distance on ground of every radar gate, we get its latlon coordinates
    for d in distances_profile:
        # Note that pyproj uses lon/lat whereas I used lat/lon
        lon,lat,ang=g.fwd(radar_pos[1], radar_pos[0], azimuth,d)
        lons_rad.append(lon)
        lats_rad.append(lat)

    # Convert to numpy array
    lons_rad = np.array(lons_rad)
    lats_rad = np.array(lats_rad)

    # Initialize interpolated variables
    interp_data = []

    # Get model heights and COSMO proj from first variable
    ###########################################################################
    model_heights=list_vars[0].attributes['z-levels']
    rad_interp_values=np.zeros(len(distances_profile),)*float('nan')

    # Get COSMO local coordinates info
    proj_COSMO=list_vars[0].attributes['proj_info']

    # Get lower left corner of COSMO domain in local coordinates
    llc_COSMO=(float(proj_COSMO['Lo1']), float(proj_COSMO['La1']))
    llc_COSMO=np.asarray(llc_COSMO).astype('float32')

    # Get upper left corner of COSMO domain in local coordinates
    urc_COSMO=(float(proj_COSMO['Lo2']), float(proj_COSMO['La2']))
    urc_COSMO=np.asarray(urc_COSMO).astype('float32')

    res_COSMO=list_vars[0].attributes['resolution']

    # Get resolution

    # Transform radar gate coordinates into local COSMO coordinates
    coords_rad_loc = pc.WGS_to_COSMO((lats_rad,lons_rad),
                                 [proj_COSMO['Latitude_of_southern_pole']
                                 ,proj_COSMO['Longitude_of_southern_pole']])


    # Check if all points are within COSMO domain
    if np.any(coords_rad_loc[:,1]<llc_COSMO[0]) or\
        np.any(coords_rad_loc[:,0]<llc_COSMO[1]) or \
            np.any(coords_rad_loc[:,1]>urc_COSMO[0]) or \
                np.any(coords_rad_loc[:,0]>urc_COSMO[1]):
                    msg = """
                    ERROR: RADAR DOMAIN IS NOT ENTIRELY CONTAINED IN COSMO
                    SIMULATION DOMAIN: ABORTING
                    """
                    raise(IndexError(dedent(msg)))

    # Now we interpolate all variables along beam using C-code file
    ###########################################################################
    for n,var in enumerate(list_vars):
        model_data=var.data
        arguments_c_code = (len(distances_profile),
                            coords_rad_loc,
                            heights_profile,
                            model_data,
                            model_heights,
                            llc_COSMO,
                            res_COSMO)

        rad_interp_values = get_all_radar_pts(arguments_c_code)
        interp_data.append(rad_interp_values[1][:])

    return lats_rad, lons_rad, interp_data

