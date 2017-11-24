# -*- coding: utf-8 -*-

""" gpm_wrapper.py: Provides routines to convert radar operator outputs
to the format used in GPM DPR files, as well as routines to retrieve
all azimuth and elevation angles from a GPM DPR file, which needed before
simulating the scan """

__author__ = "Daniel Wolfensberger"
__copyright__ = "Copyright 2017, COSMO_POL"
__credits__ = ["Daniel Wolfensberger"]
__license__ = "GPL"
__version__ = "1.0"
__maintainer__ = "Daniel Wolfensberger"
__email__ = "daniel.wolfensberger@epfl.ch"

# Global imports
import h5py
import pyproj as pp
import numpy as np
np.seterr(divide='ignore') # Disable divide by zero error
import copy

# Local imports
from cosmo_pol.constants import global_constants as constants

def _get_group(band):
    '''
    Returns the GPM group as used in the HDF5 files
    Args:
        band: the simulated GPM band, can be either 'Ka', 'Ku', 'Ka_matched',
        or 'Ku_matched'

    Returns:
        group: the name of the group in the GPM DPR H5DF files, can be
        either 'NS' (normal scan), 'MS' (matched scan) or 'HS' (high-
        sensitivity scan)
    '''
    if band == 'Ku':
        group = 'NS'
    elif band == 'Ka':
        group = 'HS'
    elif band == 'Ku_matched' or band == 'Ka_matched':
        group = 'MS'
    return group


class SimulatedGPM():
    '''
    The output class of simulated GPM swaths in the radar operator
    '''
    def __init__(self, list_radials, dim, band):
        '''
        Returns a SimulatedGPM Class instance
        Args:
            list_beams: the list of simulated radials as returned in the
                main RadarOperator class
            dim: the horizontal dimension of the simulated GPM swath, this
                is needed because it can not be guessed from the data
                ex: [88, 49]
            band: the simulated GPM band, can be either 'Ka', 'Ku', 'Ka_matched',
                or 'Ku_matched'

        Returns:
            a SimulatedGPM with five fields:
                bin_surface: a 2D array giving the indexes of  radar bins
                    that correspond to the ground level
                band: the radar band of the simulated swath
                lats: a 3D array containing the latitudes at the all GPM
                    gates (also in vertical)
                lons: a 3D array containing the longitudes at the all GPM
                    gates (also in vertical)
                data: a dictionary of 3D arrays with all simulated variables

        '''

        # Reorganize output to make it easier to compare with GPM
        # The idea is to keep only the bins that are above ground and to put
        # every beam into a common matrix

        [N,M] = dim

        pol_vars = list_radials[0].values.keys() # List of simulated variables

        # make a deepcopy to avoid overwriting the original one
        list_beams_cp = copy.deepcopy(list_radials)

        bin_surface = np.zeros((N,M))
        # Here we remove all points that are below the topography COSMO
        for idx in range(len(list_beams_cp)):
            i = int(np.floor(idx / M))
            j = idx - i * M
            try:
                bin_surface[i,j] = len(list_beams_cp[idx].mask) - \
                    np.where(list_beams_cp[idx].mask>=1)[0][0]
            except:
                bin_surface[i,j]=0
            for k in pol_vars:
                # Remove points that are below topo
                list_beams_cp[idx].values[k] = \
                    list_beams_cp[idx].values[k][list_beams_cp[idx].mask > -1]
            # Take only lat and lon profile where data is valid (above topo)
            list_beams_cp[idx].lats_profile = \
                list_beams_cp[idx].lats_profile[list_beams_cp[idx].mask > -1]

            list_beams_cp[idx].lons_profile =  \
                list_beams_cp[idx].lons_profile[list_beams_cp[idx].mask > -1]

        # Length of longest beam
        max_len=np.max([len(r.dist_profile) for r in list_radials])

        # Initalize output dictionary
        list_beams_formatted = {}
        for k in pol_vars:
            list_beams_formatted[k]=np.zeros((N,M,max_len))
        # Initialize lats and lons 3D array

        lats = np.zeros((N,M,max_len))*float('nan')
        lons = np.zeros((N,M,max_len))*float('nan')

        # Fill the output dictionary starting from the ground
        for idx in range(len(list_beams_cp)):
            i = int(np.floor(idx/M))
            j = idx - i * M
            len_beam=len(list_beams_cp[idx].values[k])
            # Flip because we want to start from the ground
            l = list_beams_cp[idx].lats_profile
            ll = list_beams_cp[idx].lons_profile

            lats[i, j, 0:len_beam] = l[::-1]
            lons[i, j, 0:len_beam] = ll[::-1]
            # Flip [::-1] because we want to start from the ground
            for k in pol_vars:
                list_beams_formatted[k][i,j,0:len_beam] = \
                    list_beams_cp[idx].values[k][::-1]

        self.bin_surface = bin_surface
        self.band = band
        self.lats = lats
        self.lons = lons
        self.data = list_beams_formatted
        return


def compare_operator_with_GPM(simulated_GPM_swath, GPM_filename,
                              additional = False):
    '''
    Is used to compare a simulated GPM swath with data from a GPM-DPR
    HDF5 file, by formating both data to the same shape and storing them in
    the same dictionary
    Args:
        simulated_GPM_swath: SimulatedGPM class instance
            (output of the radar operator)
        GPM_filename: path of the corresponding GPM DPR file
        addtional: flag to indicate if all other simulated variables
            that are not measured by GPM (temp, ZDR, etc)
            should be added to the output, default = False

    Returns:
        values_ground: ZH measured and simulated at the ground as well
            their latitude and longitude
        values_everywhere: ZH measured and simulated everywhere as well
            their latitude and longitude. If additional == True an additional
                field named 'other_vars' is added that contains all additional
                simulated variables
        bin_surface: number of bins between clutter free GPM bin and the lowest
            simulated bin
    '''

    gpm_f = h5py.File(GPM_filename,'r')

    band = simulated_GPM_swath.band
    group = _get_group(band)

    # Get latlon
    lat_simul = simulated_GPM_swath.lats
    lon_simul = simulated_GPM_swath.lons

    # First bin without clutter
    binNoClutter = gpm_f[group]['PRE']['binClutterFreeBottom'][:]

    # Get number of vertical GPM bins
    # Identical for Ku, Ku_matched and Ka_matched, different for Ka band
    GPM_N_bins =  \
            constants.GPM_NO_BINS_KA if band == 'Ka' \
                else constants.GPM_NO_BINS_KU

    # Number of bins between COSMO surface and first bin with no clutter on GPM
    diff = (GPM_N_bins - binNoClutter) - simulated_GPM_swath.bin_surface

    diff[diff<0] = 0
    diff = diff.astype(int)


    # AT GROUND
    ##########################################################################
    # ZH simulated at ground
    ZH_s_dBZ = 10*np.log10(simulated_GPM_swath.data['ZH'])
    [N,M,K] = ZH_s_dBZ.shape
    k,j = np.meshgrid(np.arange(M),np.arange(N)) # Create 2D index
    ZH_s_grd = ZH_s_dBZ[j,k,diff]

    # ZH measured at ground
    ZH_gpm = gpm_f[group]['PRE']['zFactorMeasured'][:]
    ZH_gpm[ZH_gpm<-1000] = float('nan') # Put Nan where data is missing

    ZH_m_grd = ZH_gpm[j,k,binNoClutter]
    ZH_m_grd[ZH_m_grd<-1000] = float('nan')
    # GPM seems to measure lots of noise, so we apply a mask which corresponds
    # to all pixels classified as precip by gpm
    mask = gpm_f[group]['PRE']['flagPrecip'][:]
    ZH_m_grd[mask] = np.nan

    values_ground={}
    values_ground['measured_refl'] = ZH_m_grd
    values_ground['simulated_refl'] = ZH_s_grd
    values_ground['lat'] = lat_simul[j,k,diff]
    values_ground['lon'] = lon_simul[j,k,diff]

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

    values_everywhere={}
    values_everywhere['measured_refl'] = ZH_m_everywhere
    values_everywhere['simulated_refl'] = ZH_s_everywhere
    values_everywhere['lon'] = lon_everywhere
    values_everywhere['lat'] = lat_everywhere

    if additional:
        # Add all other simulated variables to output with the same coords
        other_vars = {}
        for key in simulated_GPM_swath.data.keys():
            if key == 'ZH':
                continue # Ignore reflectivity (already in corresponding dic)
            data = simulated_GPM_swath.data[key]
            other_vars[key] = np.zeros((N,M,K))
            for i in range(K):
                other_vars[key][j,k,i] = data[j,k,diff+i]

        values_everywhere['other_vars'] = other_vars

    return values_ground, values_everywhere, diff


def get_GPM_angles(GPM_filename, band):
    '''
    From as specified GPM DPR HDF5 file, gets the azimuth (phi), elevation
    (theta) angles and range bins for every radar radial
    (there is one radial for every measurement at the ground),
    based on the position of the satellite given in the file
    Args:
        GPM_filename: path of the corresponding GPM DPR file
        band: the desired simulated GPM band, can be either 'Ka', 'Ku',
            'Ka_matched' or 'Ku_matched'
    Returns:
        azimuths: all azimuths in the form of a 2D array [degrees]
        elevations: all elevation angles in the form of a 2D array [degrees]
        ranges: all

    '''
    # Initialize geoid for inverse distance computations
    geoid = pp.Geod(ellps = 'WGS84')

    group = _get_group(band)

    gpm_f = h5py.File(GPM_file,'r')
    lat_2D = gpm_f[group]['Latitude'][:]
    lon_2D = gpm_f[group]['Longitude'][:]

    center_lat_sc = gpm_f[group]['navigation']['scLat'][:]
    center_lon_sc = gpm_f[group]['navigation']['scLon'][:]
    altitudes_sc = gpm_f[group]['navigation']['dprAlt'][:]

    pos_sc = gpm_f[group]['navigation']['scPos'][:]

    azimuths = np.zeros(lat_2D.shape)
    ranges = np.zeros(lat_2D.shape)
    elevations = np.zeros(lon_2D.shape)

    # Projection from lat/long/alt to eced
    ecef = pp.Proj(proj='geocent', ellps='WGS84', datum='WGS84')
    lla = pp.Proj(proj='latlong', ellps='WGS84', datum='WGS84')


    [N,M]=lat_2D.shape

    for i in range(N):
        for j in range(M):

            a,b,d = geoid.inv(center_lon_sc[i], center_lat_sc[i],
                              lon_2D[i,j], lat_2D[i,j])
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
