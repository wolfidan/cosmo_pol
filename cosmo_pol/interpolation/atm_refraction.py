# -*- coding: utf-8 -*-

"""atm_refraction.py: defines a set of functions to compute the path of a
radar beam while taking into account atmospheric refraction"""

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
import pycosmo as pc
from scipy.interpolate import interp1d
from scipy.integrate import odeint

# Local imports
from  cosmo_pol.config import CONFIG
from cosmo_pol.utilities import get_earth_radius
from cosmo_pol.constants import global_constants as constants


def compute_trajectory_radial(range_vec, elevation_angle, coords_radar,
                         refraction_method, N = 0):
    '''
    Computes the trajectory of a radar beam along a specified radial by
    calling the appropriate subfunction depending on the desired method
    Args:
        range_vec: vector of all ranges along the radial [m]
        elevation_angle: elevation angle of the beam [degrees]
        coord_radar: radar coordinates as 3D tuple [lat, lon, alt], ex
            [47.35, 12.3, 682]
        refraction_method: the method to be used to compute the trajectory
            can be either 1, for the 4/3 method, or 2 for the Zeng and Blahak
            (2014) ODE method
        N: atmospheric refractivity as a COSMO variable, needs to be provided
            only for refraction_method == 2

    Returns:
        s: vector of distance at the ground along the radial [m]
        h: vector of heights above ground along the radial [m]
        e: vector of incident elevation angles along the radial [degrees]
    '''

    if refraction_method==1:
        s, h, e = _ref_4_3(range_vec, elevation_angle, coords_radar)
    elif refraction_method==2:
        s, h, e = _ref_ODE(range_vec, elevation_angle, coords_radar, N)

    return s, h, e

def _deriv_z(z, r, n_h_int, dn_dh_int, RE):
    '''
    Updates the state vector of the system of ODE used in the ODE refraction
    by Blahak and Zeng
    (2014)
    Args:
        z: state vector in the form of a tuple (height, sin(theta))
        r: range vector in m (not actually used in the state vector)
        n_h_spline: piecewise linear interpolator for the refractive index as
            a function of the altitude
        dn_dh_int: piecewise linear interpolator for the derivative of the
            refractive index as a function of the altitude
        RE: earth radius [m]
    Returns:
        An updated state vector
    '''
    # Computes the derivatives (RHS) of the system of ODEs
    h, u = z
    n = n_h_int(h)
    dn_dh = dn_dh_int(h)
    return [u, (-u ** 2 * ((1./n) * dn_dh + 1./ (RE + h)) +
                ((1. / n) * dn_dh + 1. / (RE + h)))]

def _ref_ODE(range_vec, elevation_angle, coords_radar, N):
    '''
    Computes the trajectory of a radar beam along a specified radial with the
    ODE method of Zeng and Blahak (2014)
    Args:
        range_vec: vector of all ranges along the radial [m]
        elevation_angle: elevation angle of the beam [degrees]
        coord_radar: radar coordinates as 3D tuple [lat, lon, alt], ex
            [47.35, 12.3, 682]
        N: atmospheric refractivity as a COSMO variable

    Returns:
        s: vector of distance at the ground along the radial [m]
        h: vector of heights above ground along the radial [m]
        e: vector of incident elevation angles along the radial [degrees]
    '''

    # Get info about COSMO coordinate system
    proj_COSMO = N.attributes['proj_info']
    # Convert WGS84 coordinates to COSMO rotated pole coordinates
    coords_rad_in_COSMO = pc.WGS_to_COSMO(coords_radar,
                              [proj_COSMO['Latitude_of_southern_pole'],
                               proj_COSMO['Longitude_of_southern_pole']])

    llc_COSMO = (float(proj_COSMO['Lo1']), float(proj_COSMO['La1']))
    res_COSMO = N.attributes['resolution']

    # Get index of radar in COSMO rotated pole coordinates
    pos_radar_bin = [(coords_rad_in_COSMO[0]-llc_COSMO[1]) / res_COSMO[1],
                    (coords_rad_in_COSMO[1]-llc_COSMO[0]) / res_COSMO[0]]

    # Get refractive index profile from refractivity estimated from COSMO variables
    n_vert_profile = 1 + (N.data[:,int(np.round(pos_radar_bin[0])),
                             int(np.round(pos_radar_bin[0]))]) * 1E-6
    # Get corresponding altitudes
    h = N.attributes['z-levels'][:,int(np.round(pos_radar_bin[0])),
                                int(np.round(pos_radar_bin[0]))]

    # Get earth radius at radar latitude
    RE = get_earth_radius(coords_radar[0])

    if CONFIG['radar']['type']  == 'ground':
        # Invert to get from ground to top of model domain, COSMO first vert.
        # layer is the highest one...
        h = h[::-1]
        n_vert_profile = n_vert_profile[::-1] # Refractivity

    # Create piecewise linear interpolation for n as a function of height
    n_h_int = _piecewise_linear(h, n_vert_profile)
    dn_dh_int = _piecewise_linear(h[0:-1],
                                     np.diff(n_vert_profile) / np.diff(h))

    z_0 = [coords_radar[2], np.sin(np.deg2rad(elevation_angle))]
    # Solve second-order ODE
    Z = odeint(_deriv_z, z_0, range_vec, args = (n_h_int, dn_dh_int, RE))
    h = Z[:,0] # Heights above ground
    e = np.arcsin(Z[:,1]) # Elevations
    s = np.zeros(h.shape) # Arc distances
    dR = range_vec[1]-range_vec[0]
    s[0] = 0

    for i in range(1,len(s)): # Solve for arc distances
        s[i] = s[i-1] + RE * np.arcsin((np.cos(e[i-1]) * dR) / (RE + h[i]))

    s = s.astype('float32')
    h = h.astype('float32')
    e = np.rad2deg(e.astype('float32'))
    return s, h, e


def _piecewise_linear(x,y):
    '''
    Defines a piecewise linear interpolator, used to interpolate refractivity
    values between COSMO vertical coordinates
    Args:
        x: vector of independent variable
        y: vector of dependent variable

    Returns:
        A piecewise linear interpolator
    '''
    interpolator=interp1d(x,y)
    xs = interpolator.x
    ys = interpolator.y

    def pointwise(x):
        if x < xs[0]:
            return ys[0]+(x-xs[0])*(ys[1]-ys[0])/(xs[1]-xs[0])
        elif x > xs[-1]:
            return ys[-1]+(x-xs[-1])*(ys[-1]-ys[-2])/(xs[-1]-xs[-2])
        else:
            return interpolator(x)

    def ufunclike(xs):
        if np.isscalar(xs):
            xs=[xs]
        return np.array([pointwise(xi) for xi in xs])

    return ufunclike

def _ref_4_3(range_vec, elevation_angle, coords_radar):
    '''
    Computes the trajectory of a radar beam along a specified radial wwith
    the simple 4/3 earth radius model (see Doviak and Zrnic, p.21)
    Args:
        range_vec: vector of all ranges along the radial [m]
        elevation_angle: elevation angle of the beam [degrees]
        coord_radar: radar coordinates as 3D tuple [lat, lon, alt], ex
            [47.35, 12.3, 682]

    Returns:
        s: vector of distance at the ground along the radial [m]
        h: vector of heights above ground along the radial [m]
        e: vector of incident elevation angles along the radial [degrees]
    '''
    # elevation_angle must be in radians in the formula
    elevation_angle = np.deg2rad(elevation_angle)
    KE = constants.KE

    altitude_radar=coords_radar[2]
    latitude_radar=coords_radar[1]

    # Compute earth radius at radar latitude
    RE = get_earth_radius(latitude_radar)
    # Compute height over radar of every range_bin
    temp = np.sqrt(range_vec ** 2 + (KE * RE) ** 2 + 2 * range_vec *
                  KE * RE * np.sin(elevation_angle))

    h = temp - KE * RE + altitude_radar
    # Compute arc distance of every range bin
    s = KE * RE * np.arcsin((range_vec * np.cos(elevation_angle)) /
                                   (KE * RE + h))
    e = elevation_angle + np.arctan(range_vec * np.cos(elevation_angle) /
                                    (range_vec*np.sin(elevation_angle) + KE *
                                     RE + altitude_radar))
    s = s.astype('float32')
    h = h.astype('float32')
    e = np.rad2deg(e.astype('float32'))

    return s,h,e

def compute_trajectory_GPM(elevation):
    '''
    Computes the trajectory of a GPM beam along a specified radial,
    currently atmospheric refraction is not taken into account
    Args:
        elevation: elevation angle of the radial in degrees

    Returns:
        s: vector of distance at the ground along the radial [m]
        h: vector of heights above ground along the radial [m]
        e: vector of incident elevation angles along the radial [degrees]
    '''

    # Get info about GPM satellite position
    latitude = CONFIG['radar']['coords'][0]
    altitude_radar = CONFIG['radar']['coords'][2]
    max_range = CONFIG['radar']['range']
    radial_resolution = CONFIG['radar']['radial_resolution']

    elev_rad = np.deg2rad(elevation)

    # For GPM refraction is simply ignored...
    KE = 1
    maxHeightCOSMO = constants.MAX_HEIGHT_COSMO
    RE = get_earth_radius(latitude)
    # Compute maximum range to target (using cosinus law in the triangle
    # earth center-radar-target)

    range_vec=np.arange(radial_resolution/2.,max_range,radial_resolution)

    h = - (np.sqrt(range_vec**2 + (KE * RE)**2 + 2*range_vec*
                  KE * RE*np.sin(elev_rad))- KE * RE) + altitude_radar

    s = KE * RE * np.arcsin((range_vec * np.cos(elev_rad)) / (KE * RE + h))
    e = elevation - np.rad2deg(np.arctan(range_vec * np.cos(elev_rad)
                            / (range_vec *
                              np.sin(elev_rad) + KE * RE + altitude_radar)))

    in_lower_atm = [h < maxHeightCOSMO]

    h = h[in_lower_atm]
    s = s[in_lower_atm]
    e = e[in_lower_atm]

    s = s.astype('float32')
    h = h.astype('float32')
    e = np.rad2deg(e.astype('float32'))

    return s,h,e
