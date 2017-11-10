# -*- coding: utf-8 -*-
"""
Created on Fri Jan 29 17:25:11 2016

@author: wolfensb
"""
import os
import numpy as np
import pickle
import gzip
from pytmatrix import orientation
from pytmatrix.tmatrix import Scatterer
import multiprocessing

from joblib import Parallel, delayed

from cosmo_pol.constants import constants
from cosmo_pol.hydrometeors import hydrometeors
from cosmo_pol.lookup.lut import Lookup_table

FOLDER_LUT=os.path.dirname(os.path.realpath(__file__))+'/stored_lut/'
FOLDER_FINAL_LUT=os.path.dirname(os.path.realpath(__file__))+'/final_lut/'

GENERATE_1MOM=True
GENERATE_2MOM=False

FORCE_REGENERATION_SCATTER_TABLES=False

ELEVATIONS = range(0,91,2)
TEMPERATURES_LIQ = range(262,316,2)
TEMPERATURES_SOL = range(200,278,2)

FREQUENCIES=[2.7,4.15,5.6,7.7,9.41,9.8,11.7,13.6,24.6,35.6]
FREQUENCIES=[9.41]

NUM_DIAMETERS=1024

HYDROM_TYPES=['R','S','G','I'] # Rain, snow, graupel, hail and ice crystals

global SCATTERER

def create_scatterer(wavelength,orientation_std):

    scatt = Scatterer(radius = 1.0, wavelength = wavelength)
    scatt.or_pdf = orientation.gaussian_pdf(std=orientation_std)
    scatt.orient = orientation.orient_averaged_fixed
    return scatt

def compute_sz(hydrom,freq,elevation,T, list_D):
    list_SZ=[]

    m_func = hydrom.get_m_func(T,freq)

    geom_back=(90-elevation, 180-(90-elevation), 0., 180, 0.0,0.0)
    geom_forw=(90-elevation, 90-elevation, 0., 0.0, 0.0,0.0)


    for i,D in enumerate(list_D):

        ar = hydrom.get_axis_ratio(D)

        SCATTERER.radius = D/2.
        SCATTERER.m = m_func(D)
        SCATTERER.axis_ratio = ar

        # Backward scattering (note that we do not need amplitude matrix for backward scattering)
        SCATTERER.set_geometry(geom_back)
        Z_back=SCATTERER.get_Z()
        # Forward scattering (note that we do not need phase matrix for forward scattering)
        SCATTERER.set_geometry(geom_forw)
        S_forw=SCATTERER.get_S()

        list_SZ.append([Z_back,S_forw])

    return list_SZ

def flatten_matrices(list_matrices):
    arr_SZ=np.zeros((len(list_matrices),len(list_matrices[0]),12))

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

def sz_lut(scheme,hydrom_type,list_frequencies,list_elevations, list_temperatures):
    if np.isscalar(list_frequencies):
        list_frequencies=[list_frequencies]

    hydrom = hydrometeors.create_hydrometeor(hydrom_type,scheme)

    list_D=np.linspace(hydrom.d_min,hydrom.d_max,NUM_DIAMETERS).astype('float32')

    num_cores = multiprocessing.cpu_count()

    for f in list_frequencies:
        global SCATTERER

        wavelength=constants.C/(f*1E09)*1000 # in mm

        SCATTERER = create_scatterer(wavelength,hydrom.canting_angle_std)

        SZ_matrices=np.zeros((len(list_elevations),len(list_temperatures),len(list_D),12))

        for i,e in enumerate(list_elevations):
            print 'Running elevation : ' + str(e)
            results = (Parallel(n_jobs=num_cores)(delayed(compute_sz)(hydrom,f,e,t, list_D) for t in list_temperatures))
            arr_SZ = flatten_matrices(results)

            SZ_matrices[i,:,:,:] = arr_SZ

        lut_SZ = Lookup_table()
        lut_SZ.add_axis('e',list_elevations)
        lut_SZ.add_axis('t', list_temperatures)
        lut_SZ.add_axis('d',list_D)
        lut_SZ.add_axis('sz', np.arange(12))
        lut_SZ.set_value_table(SZ_matrices)

        pickle.dump(lut_SZ, gzip.open( FOLDER_LUT+"lut_SZ_"+hydrom_type+'_'+str(f).replace('.','_')+'_'+scheme+".pz", "wb" ) )

def prepare_global_lookup_tables(scheme,freq):
    os.chdir(FOLDER_LUT)
    lut_sz = dict.fromkeys(HYDROM_TYPES)

    for h in HYDROM_TYPES:
        print(h)
        try:
            lut_sz[h]=pickle.load(gzip.open(FOLDER_LUT+ "lut_SZ_"+h+"_"+str(freq).replace('.','_')+'_'+scheme+'.pz','rb'))
        except:
            raise
            print('No SZ lookup table was found for frequency ' + str(freq)+' and hydrometeor '+h+'...')

    pickle.dump(lut_sz,gzip.open(FOLDER_FINAL_LUT+'all_luts_SZ_f_'+str(freq).replace('.','_')+'_'+scheme+'.pz','wb'))

if __name__=='__main__':
#    # Generate scattering tables (if needed)
#    for f in FREQUENCIES:
#        for hydrom_type in HYDROM_TYPES:
#            if GENERATE_1MOM:
#                if FORCE_REGENERATION_SCATTER_TABLES or not os.path.exists(FOLDER_LUT+'lut_SZ_'+hydrom_type+'_'+str(f).replace('.','_')+"_1mom.pz"):
#                    if hydrom_type!='H':
#                        print('Generating scatter table for 1 moment scheme, hydrometeor='+hydrom_type+' and freq='+str(f))
#                        if hydrom_type in ['S','G']:
#                            temp = TEMPERATURES_SOL
#                        else:
#                            temp = TEMPERATURES_LIQ
#                        sz_lut('1mom',hydrom_type,f,ELEVATIONS,temp)
#            if GENERATE_2MOM:
#                if FORCE_REGENERATION_SCATTER_TABLES or not os.path.exists(FOLDER_LUT+'lut_SZ_'+hydrom_type+'_'+str(f).replace('.','_')+"_2mom.pz"):
#                    print('Generating scatter table for 2 moment scheme, hydrometeor='+hydrom_type+' and freq='+str(f))
#                    if hydrom_type in ['S','G','H']:
#                        temp = TEMPERATURES_SOL
#                    else:
#                        temp = TEMPERATURES_LIQ
#                    sz_lut('2mom',hydrom_type,f,ELEVATIONS,temp)

    # Create global lookup-tables
    for f in FREQUENCIES:
        if GENERATE_1MOM:
            prepare_global_lookup_tables('1mom',f)
        if GENERATE_2MOM:
            prepare_global_lookup_tables('2mom',f)

