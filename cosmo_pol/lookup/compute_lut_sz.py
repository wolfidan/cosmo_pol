# -*- coding: utf-8 -*-

"""compute_lut_sz.py: computes scattering lookup tables, with the pytmatrix
library of Jussi Leinonen  https://github.com/jleinonen/pytmatrix/wiki"""

__author__ = "Daniel Wolfensberger"
__copyright__ = "Copyright 2017, COSMO_POL"
__credits__ = ["Daniel Wolfensberger"]
__license__ = "GPL"
__version__ = "1.0"
__maintainer__ = "Daniel Wolfensberger"
__email__ = "daniel.wolfensberger@epfl.ch"

# Global imports
import os
import numpy as np
from pytmatrix import orientation
from pytmatrix.tmatrix import Scatterer
import multiprocessing
from joblib import Parallel, delayed
import stats

# Local imports
from cosmo_pol.constants import global_constants as constants
from cosmo_pol.interpolation import quadrature
from cosmo_pol.lookup import Lookup_table, save_lut
from cosmo_pol.hydrometeors import create_hydrometeor

# Define constants

# The name of the scattering method that is being used
SCATTERING_METHOD = 'tmatrix_masc'
# The base directory from which other directories are defined, by default
# the directory of this file
BASE_FOLDER = os.path.dirname(os.path.realpath(__file__)) +'/'
# The folder where the lookup tables must be stored
FOLDER_LUT = BASE_FOLDER + SCATTERING_METHOD
# The folder where the computed quadrature points must be stored
FOLDER_QUAD = BASE_FOLDER + 'quad_pts/'

# Specify if one-moment and/or two-moments lookup table should be computed
GENERATE_1MOM = True
GENERATE_2MOM = False


# Whether to regenerate  all lookup tables, even if already present
FORCE_REGENERATION_SCATTER_TABLES = True

'''
Define all axes values for all hydrometeor types
Rain : ELEVATIONS, TEMPERATURES_LIQ, DIAMETER
Snow: ELEVATIONS, TEMPERATURES_SOL, DIAMETER
Graupel: ELEVATIONS, TEMPERATURES_SOL, DIAMETER
Hail: ELEVATIONS, TEMPERATURES_SOL, DIAMETER
Ice crystals: ELEVATIONS, TEMPERATURES_SOL, DIAMETER
Melting Snow: ELEVATIONS, W_CONTENTS, DIAMETER
Melting Graupel: ELEVATIONS, W_CONTENTS, DIAMETER
'''

ELEVATIONS = range(0,91,2)
TEMPERATURES_LIQ = range(262,316,2)
TEMPERATURES_SOL = range(200,278,2)
W_CONTENTS = np.linspace(1E-3,0.999,100)

# The frequencies in GHz for which the lookup tables will be computed
FREQUENCIES=[9.41, 5.6, 2.7, 13.6, 35.6]
# The number of diameter bins to use
# DIAMETER = np.linspace(hydrom.dmin, hydrom.dmax, NUM_DIAMETERS)
NUM_DIAMETERS = 1024

# Number of quadrature points used in the aspect-ratio integration
N_QUAD_PTS = 5
# The maximum aspect-ratio to consider, avoids crashing the T-matrix
MAX_AR = 7
# The fixed temperature to use for melting hydrometeors
T_MELTING = 273.15

HYDROM_TYPES=['S','G','mS','mG'] # Snow, graupel, hail, rain and ice
global SCATTERER

def _create_scatterer(wavelength, orientation_std):
    """
        Create a scatterer instance that will be used later
        Args:
            wavelength: wavelength in mm
            orientation_std: standard deviation of the Gaussian distribution
                of orientations (canting angles)
        Returns:
            scatt: a pytmatrix Scatterer class instance
    """
    scatt = Scatterer(radius = 1.0, wavelength = wavelength, ndgs = 10)
    scatt.or_pdf = orientation.gaussian_pdf(std = orientation_std)
    scatt.orient = orientation.orient_averaged_fixed
    return scatt

def _compute_gautschi_canting(list_of_std):
    """
        Computes the quadrature points and weights for a list of standard
        deviations for the Gaussian distribution of canting angles
        Args:
            list_of_std: list of standard deviations for which a
                quadrature rule must be computed (one set of points and weights
                per standard deviation)
        Returns:
            A tuple, containing two lists, one with the quadrature points
            for every stdev, one with the quadrature weights for every
            stdev
    """
    scatt = Scatterer(radius = 5.0)
    gautschi_pts = []
    gautschi_w = []
    for l in list_of_std:
        scatt.or_pdf = orientation.gaussian_pdf(std=l)
        scatt.orient = orientation.orient_averaged_fixed
        scatt._init_orient()
        gautschi_pts.append(scatt.beta_p)
        # Check if sum of weights if 1, if not set it to 1
        scatt.beta_w /= np.sum(scatt.beta_w)
        gautschi_w.append(scatt.beta_w)

    return(gautschi_pts, gautschi_w)

def _compute_gautschi_ar(ar_lambda, ar_loc, ar_mu):
    """
        Computes the quadrature points and weights for a list of parameters
        for the Gamma distribution of aspect ratios
        The parameters are lambda, loc and mu in the Gamma pdf:
        p(x) = ((a - loc)^(lamba-1) exp(-(x - 1)/mu)) / mu^lambda* Gamma(lamb)
        Args:
            ar_lambda: list of lambda (slope) parameters
            ar_loc: list of location parameters
            ar_mu: list of mu (shape) parameters
        Returns:
            A tuple, containing two lists, one list with the quadrature points
            for every lambda, loc and mu parameter, and one list
            with the quadrature weights for every lambda, loc and
            mu parameter
    """

    gautschi_pts = []
    gautschi_w = []
    for l in zip(ar_lambda, ar_loc, ar_mu):
        gamm = stats.gamma(l[0],l[1],l[2])
        pts,wei = quadrature.get_points_and_weights(lambda x: gamm.pdf(x),
                                                    num_points = N_QUAD_PTS,
                                                    left=l[1],right=MAX_AR)
        # Check if sum of weights if 1, if not set it to 1
        wei /= np.sum(wei)
        gautschi_pts.append(pts)
        gautschi_w.append(wei)

    return(gautschi_pts, gautschi_w)

def _compute_sz_with_quad(hydrom, freq, elevation, T, quad_pts_o,
                         quad_pts_ar, list_D):
    """
        Computes all scattering properties for a given set of parameters and
        for a given hydrometeor using a set of quadrature points for integration
        in orientations and aspect-ratios
        This is to be used for all hydrometeors except melting snow and
        melting graupel
        Args:
            hydrom: a Hydrometeor Class instance (see hydrometeors.py)
            freq: the frequency in GHz
            elevation: incident elevation angle in degrees
            T: the temperature in K
            quad_pts_o: a tuple (pts, weights) containing the quadrature
                points and weights for the canting angle integration
            quad_pts_ar: a tuple (pts, weights) containing the quadrature
                points and weights for the aspect ratio integration
            list_D: list of diameters in mm for which to compute the scattering
                properties
        Returns:
            list_SZ: a list which contains for every diameter a tuple (Z,S)
                where Z is the 4x4 phase matrix at backward scattering and
                S the 2x2 amplitude matrix at forward scattering
    """
    list_SZ=[]
    m_func = hydrom.get_m_func(T,freq)

    geom_back=(90-elevation, 180-(90-elevation), 0., 180, 0.0,0.0)
    geom_forw=(90-elevation, 90-elevation, 0., 0.0, 0.0,0.0)

    for i,D in enumerate(list_D):
        SCATTERER.radius = D/2.
        SCATTERER.m = m_func(D)

        SCATTERER.beta_p = quad_pts_o[0][i]
        SCATTERER.beta_w = quad_pts_o[1][i]

        Z_back = np.zeros((4,4))
        SCATTERER.set_geometry(geom_back)
        for pt, we in zip(quad_pts_ar[0][i],quad_pts_ar[1][i]):
            SCATTERER.axis_ratio = pt
            Z_ar = SCATTERER.get_Z()
            Z_back += we * Z_ar

        S_forw = np.zeros((2,2), dtype=complex)
        SCATTERER.set_geometry(geom_forw)
        for pt, we in zip(quad_pts_ar[0][i],quad_pts_ar[1][i]):
            SCATTERER.axis_ratio = pt
            S_ar = SCATTERER.get_S()
            S_forw += we * S_ar
        list_SZ.append([Z_back,S_forw])
    print('done')
    return list_SZ


def _compute_sz_with_quad_melting(hydrom, freq, elevation, w_content, quad_pts_o,
                                 quad_pts_ar):
    """
        Computes all scattering properties for a given set of parameters and
        for a given hydrometeor using a set of quadrature points for integration
        in orientations and aspect-ratios
        This is to be used for the melting hydrometeors, the diameters are
        not specified explicitely because they depend on the water content
        Args:
            hydrom: a Hydrometeor Class instance (see hydrometeors.py)
            freq: the frequency in GHz
            elevation: incident elevation angle in degrees
            w_contents: a list of water contents, ranging from 0 to 1
            quad_pts_o: a tuple (pts, weights) containing the quadrature
                points and weights for the canting angle integration
            quad_pts_ar: a tuple (pts, weights) containing the quadrature
                points and weights for the aspect ratio integration
        Returns:
            list_SZ: a list which contains for every water fraction a tuple
                (Z, S) where Z is the 4x4 phase matrix at backward scattering
                and S the 2x2 amplitude matrix at forward scattering
    """
    list_SZ=[]
    hydrom.f_wet = w_content
    list_D = np.linspace(hydrom.d_min,hydrom.d_max,
                         NUM_DIAMETERS).astype('float32')

    m_func = hydrom.get_m_func(T_MELTING,freq)

    geom_back=(90-elevation, 180-(90-elevation), 0., 180, 0.0,0.0)
    geom_forw=(90-elevation, 90-elevation, 0., 0.0, 0.0,0.0)

    quad_pts_w = quad_pts[0]
    quad_pts_ar = quad_pts[1]
    for i,D in enumerate(list_D):
        SCATTERER.radius = D/2.
        SCATTERER.m = m_func(D)
        print(m_func(D),w_content)
        SCATTERER.beta_p = quad_pts_w[0][i]
        SCATTERER.beta_w = quad_pts_w[1][i]

        Z_back = np.zeros((4,4))
        SCATTERER.set_geometry(geom_back)
        for pt, we in zip(quad_pts_ar[0][i],quad_pts_ar[1][i]):
            SCATTERER.axis_ratio = pt
            Z_ar = SCATTERER.get_Z()
            Z_back += we * Z_ar

        S_forw = np.zeros((2,2), dtype=complex)
        SCATTERER.set_geometry(geom_forw)
        for pt, we in zip(quad_pts_ar[0][i],quad_pts_ar[1][i]):
            SCATTERER.axis_ratio = pt
            S_ar = SCATTERER.get_S()
            S_forw += we * S_ar
        list_SZ.append([Z_back,S_forw])
    print('done')
    return list_SZ

def _flatten_matrices(list_matrices):
    """
        Flattens a list of lists of Z and S matrices computed with the two
        previous functions to a 3D array.
        Args:
            list_matrices: a list of lists of tuples (Z_back, S_forw)
        Returns:
            arr_SZ: a 3D array of shape N x M x 12,
                where N is the number of lists of S and Z matrices
                M is the number of S and Z matrices within every list
                and 12, is the number of entries from each tuple (Z,S) which
                are relevant for the derivation of polarimetric variables
    """
    arr_SZ = np.zeros((len(list_matrices), len(list_matrices[0]), 12))
    for i,mat_t in enumerate(list_matrices):
        for j,mat_SZ in enumerate(mat_t):
            S = mat_SZ[1]
            Z = mat_SZ[0]

            arr_SZ[i,j,0] = Z[0,0]
            arr_SZ[i,j,1] = Z[0,1]
            arr_SZ[i,j,2] = Z[1,0]
            arr_SZ[i,j,3] = Z[1,1]
            arr_SZ[i,j,4] = Z[2,2]
            arr_SZ[i,j,5] = Z[2,3]
            arr_SZ[i,j,6] = Z[3,2]
            arr_SZ[i,j,7] = Z[3,3]
            arr_SZ[i,j,8] = S[0,0].real
            arr_SZ[i,j,9] = S[0,0].imag
            arr_SZ[i,j,10] = S[1,1].real
            arr_SZ[i,j,11] = S[1,1].imag

    return arr_SZ

def sz_lut_melting(scheme, hydrom_type, list_frequencies, list_elevations,
                   list_wcontent, quad_pts):
    """
        Computes and saves a scattering lookup table for a given melting
        hydrometeortype and various frequencies, elevations, etc.
        Args:
            scheme: microphysical scheme, '1mom' or '2mom'
            hydrom_type: the hydrometeor type, either 'mS' (melt. snow)
                or 'mG' (melt. graup)
            list_frequencies: list of frequencies for which to obtain the
                lookup tables, in GHz
            list_elevations: list of incident elevation angles for which to
                obtain the lookup tables, in degrees
            list_wcontent: list of water contents for which to obtain the
                lookup tables, unitless, from 0 to 1
            quad_pts: the quadrature points computed with the
                calculate_quadrature_points function (see below)
        Returns:
            No output but saves a lookup table
    """

    if np.isscalar(list_frequencies):
        list_frequencies=[list_frequencies]

    hydrom = create_hydrometeor(hydrom_type,scheme)

    array_D = []
    for wc in W_CONTENTS:
        hydrom.f_wet = wc
        list_D = np.linspace(hydrom.d_min,hydrom.d_max,NUM_DIAMETERS).astype('float32')
        array_D.append(list_D)
    array_D = np.array(array_D)

    num_cores = multiprocessing.cpu_count()

    for f in list_frequencies:
        global SCATTERER

        wavelength=constants.C/(f*1E09)*1000 # in mm

        # The 40 here is the orientation std, but it doesn't matter since
        # we integrate "manually" over the distributions, we just have
        # to set something to start
        SCATTERER = _create_scatterer(wavelength,40)

        SZ_matrices=np.zeros((len(list_elevations), len(list_wcontent),
                              len(list_D), 12))

        for i,e in enumerate(list_elevations):
            print('Running elevation : ' + str(e))
            results = (Parallel(n_jobs=num_cores)(delayed(
                    _compute_sz_with_quad_melting)(hydrom, f, e, wc,
                                         quad_pts[j][0], quad_pts[j][1])
                                         for j,wc in enumerate(list_wcontent)))
            arr_SZ = _flatten_matrices(results)

            SZ_matrices[i,:,:,:] = arr_SZ

        # Create lookup table for a given frequency
        lut_SZ = Lookup_table()
        lut_SZ.add_axis('e',list_elevations)
        lut_SZ.add_axis('wc', list_wcontent)
        lut_SZ.add_axis('d',array_D)
        lut_SZ.add_axis('sz', np.arange(12))
        lut_SZ.set_value_table(SZ_matrices)

        # The name of  the lookup table is lut_SZ_<hydro_name>_<freq>_<scheme>.lut
        filename = (FOLDER_LUT + "lut_SZ_" + hydrom_type+'_' +
                    str(f).replace('.','_')+'_' + scheme+".lut")
        save_lut(lut_SZ, filename)

def sz_lut(scheme, hydrom_type, list_frequencies, list_elevations,
           list_temperatures, quad_pts):
    """
        Computes and saves a scattering lookup table for a given
        hydrometeor type (non melting) and various frequencies, elevations, etc.
        Args:
            scheme: microphysical scheme, '1mom' or '2mom'
            hydrom_type: the hydrometeor type, either 'mS' (melt. snow)
                or 'mG' (melt. graup)
            list_frequencies: list of frequencies for which to obtain the
                lookup tables, in GHz
            list_elevations: list of incident elevation angles for which to
                obtain the lookup tables, in degrees
            list_temperatures: list of temperatures for which to obtain the
                lookup tables, in K
            quad_pts: the quadrature points computed with the
                calculate_quadrature_points function (see below)
        Returns:
            No output but saves a lookup table
    """

    if np.isscalar(list_frequencies):
        list_frequencies=[list_frequencies]


    hydrom = create_hydrometeor(hydrom_type,scheme)
    list_D=np.linspace(hydrom.d_min,hydrom.d_max,NUM_DIAMETERS).astype('float32')


    num_cores = multiprocessing.cpu_count()

    for f in list_frequencies:
        global SCATTERER

        wavelength=constants.C/(f*1E09)*1000 # in mm

        SCATTERER = _create_scatterer(wavelength,hydrom.canting_angle_std)

        SZ_matrices=np.zeros((len(list_elevations), len(list_temperatures),
                              len(list_D), 12))

        for i,e in enumerate(list_elevations):
            print('Running elevation : ' + str(e))
            results = (Parallel(n_jobs = num_cores)(delayed(_compute_sz_with_quad)
                (hydrom,f,e,t,quad_pts[0],quad_pts[1], list_D) for t in
                list_temperatures))

            arr_SZ = _flatten_matrices(results)

            SZ_matrices[i,:,:,:] = arr_SZ

        # Create lookup table for a given frequency
        lut_SZ = Lookup_table()
        lut_SZ.add_axis('e',list_elevations)
        lut_SZ.add_axis('t', list_temperatures)
        lut_SZ.add_axis('d',list_D)
        lut_SZ.add_axis('sz', np.arange(12))
        lut_SZ.set_value_table(SZ_matrices)
        # The name of  the lookup table is lut_SZ_<hydro_name>_<freq>_<scheme>.lut
        filename = (FOLDER_LUT+"lut_SZ_"+hydrom_type+'_'+
                    str(f).replace('.','_')+'_'+scheme+".lut")
        save_lut(lut_SZ, filename)

def _quadrature_parallel_melting(hydrom, f_wet):
    """
        Computes the quadrature points and weights for a given water fraction
        for all diameters
        Args:
            hydrom: tan hydrometeor Class instance, either of melting snow
                or melting graupel
            f_wet: the wet fraction unitless
        Returns:
            quad_pts_o: a tuple (pts, weights) containing the quadrature
                points and weights for all diameters
            quad_pts_ar: a tuple (pts, weights) containing the quadrature
                points and weights for all diameters
    """

    print('Wet fraction = '+ str(f_wet))
    hydrom.f_wet = f_wet
    list_D = np.linspace(hydrom.d_min, hydrom.d_max, NUM_DIAMETERS).astype('float32')

    canting_stdevs = hydrom.get_canting_angle_std_masc(list_D)

    quad_pts_o = _compute_gautschi_canting(canting_stdevs)

    ar_alpha, ar_loc, ar_scale = hydrom.get_aspect_ratio_pdf_masc(list_D)
    quad_pts_ar = _compute_gautschi_ar(ar_alpha, ar_loc, ar_scale)

    return quad_pts_o, quad_pts_ar

def _calculate_quadrature_points(hydrom_type, folder):
    """
        Computes the quadrature points and weights for the integration of the
        distributions of orientations and aspect ratio type for a given
        hydrometeor type and saves them to the file
        Args:
            hydrom_type: the hydrometeor type, can be either 'R', 'S', 'G', 'H',
                'mS' or 'mG'
        Returns:
            None but saves the quadrature points to the drive
    """
    # Scheme doesn't matter here
    scheme = '1mom'
    hydrom = create_hydrometeor(hydrom_type,scheme)
    if hydrom_type in ['S','G', 'R','H', 'I']:
        list_D=np.linspace(hydrom.d_min,hydrom.d_max,NUM_DIAMETERS).astype('float32')
        hydrom = create_hydrometeor(hydrom_type,scheme)

        # In order to save time, we precompute the points of the canting quadrature
        # (since they are independent of temp, elev and frequency)
        if hasattr(hydrom,'get_aspect_ratio_pdf_masc'):
            canting_stdevs = hydrom.get_canting_angle_std_masc(list_D)
        else:
            canting_stdevs = hydrom.canting_angle_std * np.ones((len(list_D,)))

        quad_pts_canting = _compute_gautschi_canting(canting_stdevs)

        # In order to save time, we precompute the points of the ar quadrature
        # (since they are independent of temp, elev and frequency)
        if hasattr(hydrom,'get_aspect_ratio_pdf_masc'):
            ar_alpha, ar_loc, ar_scale = hydrom.get_aspect_ratio_pdf_masc(list_D)
            quad_pts_ar = _compute_gautschi_ar(ar_alpha, ar_loc, ar_scale)
        else:
            # If no pdf is available we just take a quadrature of one single point
            # (the axis-ratio) with a weight of one, for sake of generality
            ar = hydrom.get_aspect_ratio(list_D)
            quad_pts_ar = ([[a] for a in ar],[[1]]*len(ar))

        quad_pts = (quad_pts_canting, quad_pts_ar)

    elif hydrom_type in ['mS','mG']: #  wet hydrometeors
        num_cores = multiprocessing.cpu_count()
        quad_pts = (Parallel(n_jobs=num_cores)(delayed(
                    _quadrature_parallel_melting)(hydrom,w)
                    for w in W_CONTENTS))

    quad_pts = np.array(quad_pts)

    np.save(FOLDER_QUAD + '/quad_pts_'+hydrom_type, quad_pts)


if __name__=='__main__':
    """
        Create all lookup tables for the specified hydrometeor types and
        microphysical schemes

        If no quadrature points are available, they are previously precomputed
    """
    quad_pts = {}
    for hydrom_type in HYDROM_TYPES:
        fname = FOLDER_QUAD + 'quad_pts_'+hydrom_type+'.npy'
        if not os.path.exists(fname):
            print(hydrom_type)
            _calculate_quadrature_points(hydrom_type)

        quad_pts[hydrom_type] = np.load(fname)

    for f in FREQUENCIES:
        for hydrom_type in HYDROM_TYPES:
            if GENERATE_1MOM:
                name_lut = (FOLDER_LUT + 'lut_SZ_' + hydrom_type + '_' +
                            str(f).replace('.','_') + "_1mom.lut")
                if (FORCE_REGENERATION_SCATTER_TABLES
                    or not os.path.exists(name_lut)):
                    if hydrom_type!='H':
                        msg = '''
                        Generating scatter table for 1 moment scheme,
                        hydrometeor = {:s}
                        freq = {:s}
                        '''.format(hydrom_type, str(f))
                        print(msg)
                        if hydrom_type in ['S','G']:
                            temp = TEMPERATURES_SOL
                        else:
                            temp = TEMPERATURES_LIQ
                        if hydrom_type in ['mS','mG']:
                            sz_lut_melting('1mom',hydrom_type, f, ELEVATIONS,
                                           W_CONTENTS, quad_pts[hydrom_type])
                        else:
                            sz_lut('1mom' ,hydrom_type, f, ELEVATIONS, temp,
                                   quad_pts[hydrom_type])
            if GENERATE_2MOM:
                name_lut = (FOLDER_LUT + 'lut_SZ_' + hydrom_type + '_' +
                            str(f).replace('.','_') + "_2mom.lut")
                if (FORCE_REGENERATION_SCATTER_TABLES
                    or not os.path.exists(name_lut)):
                    msg = '''
                        Generating scatter table for 2 moment scheme,
                        hydrometeor = {:s}
                        freq = {:s}
                        '''.format(hydrom_type, str(f))
                    print(msg)
                    if hydrom_type in ['S','G','H']:

                        temp = TEMPERATURES_SOL
                    else:
                        temp = TEMPERATURES_LIQ
                    if hydrom_type in ['mS','mG']:
                        sz_lut_melting('2mom',hydrom_type, f, ELEVATIONS,
                                       W_CONTENTS, quad_pts[hydrom_type])
                    else:
                        sz_lut('2mom', hydrom_type, f, ELEVATIONS, temp,
                               quad_pts[hydrom_type])


