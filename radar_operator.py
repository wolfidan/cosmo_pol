# -*- coding: utf-8 -*-
"""
Created on Wed Aug  5 10:51:15 2015

@author: wolfensb
"""
# Reload modules

import sys
from functools import partial
import multiprocess as mp
import numpy as np
import copy
import pycosmo as pc
import gc

from cosmo_pol.radar.pyart_wrapper import PyartRadop,PyartRadopVProf, RadarDisplay
from cosmo_pol.utilities import cfg, utilities,tictoc
from cosmo_pol.constants.constants import constants
from cosmo_pol.interpolation import interpolation
from cosmo_pol.scatter import doppler_scatter_sz
from cosmo_pol.gpm import GPM_simulator
from cosmo_pol.lookup.lut import load_all_lut

BASE_VARIABLES=['U','V','W','QR_v','QS_v','QG_v','QI_v','RHO','T']
BASE_VARIABLES_2MOM=['QH_v','QNH_v','QNR_v','QNS_v','QNG_v','QNI_v']

class RadarOperator():
    def __init__(self, options_file='', output_variables = 'all'):
        '''
        Inputs:
            options_file : link to the .yml file specifying the user config.
            diagnostic_mode: if true, all model variables will be
        '''
        # delete the module's globals
        print('Reading options defined in options file')
        cfg.init(options_file) # Initialize options with 'options_radop.txt'

        constants.update() # Update constants now that we know user config
        self.config = cfg.CONFIG
        self.lut_sz = None

        self.current_microphys_scheme = '1mom'
        self.dic_vars = {}
        self.N = 0 # atmospheric refractivity

        if output_variables in ['all','only_model','only_radar']:
            self.output_variables = output_variables
        else:
            self.output_variables = 'all'
            msg = """Invalid output_variables input, must be either
            'all', 'only_model' or 'only_radar'
            """
            print(msg)

    def close(self):
        try:
            del dic_vars, N, lut_sz, output_variables
        except:
            pass
        self.config = None
        cfg.CONFIG = None

    def update_config(self, config_dic, check = True):
        # if check == False, no sanity check will be done, do this only
        # if you are sure of what you are doing

        if check:
            checked_config = cfg.sanity_check(config_dic)
        else:
            checked_config = config_dic

        # If frequency was changed or if melting is considered and not before
        #  reload appropriate lookup tables
        if checked_config['radar']['frequency'] != \
             self.config['radar']['frequency'] or \
             checked_config['microphysics']['with_melting'] != \
             self.config['microphysics']['with_melting'] \
             or not self.lut_sz:

             self.config = checked_config
             cfg.CONFIG = checked_config
             # Recompute constants
             constants.update() # Update constants now that we know user config
             self.set_lut()

        else:

            self.config = checked_config
            cfg.CONFIG = checked_config
            # Recompute constants
            constants.update() # Update constants now that we know user config


    def set_lut(self):
        micro_scheme = self.current_microphys_scheme
        has_ice = self.config['microphysics']['with_ice_crystals']
        has_melting = self.config['microphysics']['with_melting']
        scattering_method = self.config['microphysics']['scattering']
        freq = self.config['radar']['frequency']
        list_hydrom = ['R','S','G']
        if micro_scheme == '2mom':
            list_hydrom.extend(['H'])
        if has_melting:
            list_hydrom.extend(['mS','mG'])
        if has_ice:
            list_hydrom.extend(['I'])


        lut = load_all_lut(micro_scheme, list_hydrom, freq, scattering_method)

        self.lut_sz = lut

    def get_pos_and_time(self):
        latitude = self.config['radar']['coords'][0]
        longitude = self.config['radar']['coords'][1]
        altitude = self.config['radar']['coords'][2]
        time=self.dic_vars['T'].attributes['time'] # We could read any variable, T or others

        out={'latitude':latitude,'longitude':longitude,'altitude':altitude,\
        'time':time}

        return out

    def get_config(self):
        return self.config

    def define_globals(self):
        global output_variables
        output_variables = self.output_variables

        global dic_vars
        dic_vars=self.dic_vars

        global N
        N=self.N

        global lut_sz
        lut_sz = self.lut_sz

        return dic_vars, N, lut_sz, output_variables

    def load_model_file(self, filename, cfilename=None):
        file_h = pc.open_file(filename)
        vars_to_load = copy.deepcopy(BASE_VARIABLES)

        # Check if necessary variables are present in file

        base_vars_ok = file_h.check_if_variables_in_file(['P','T','QV','QR','QC','QI','QS','QG','U','V','W'])
        two_mom_vars_ok = file_h.check_if_variables_in_file(['QH','QNH','QNR','QNS','QNG'])

        if self.config['refraction']['scheme'] == 2:
            if file_h.check_if_variables_in_file(['T','P','QV']):
                vars_to_load.extend('N')
            else:
                print('Necessary variables for computation of atm. refractivity: ',
                      'Pressure, Water vapour mass density and temperature',
                      'were not found in file. Using 4/3 method instead.')
                self.config['refraction_method']='4/3'

        # To consider the effect of turbulence we need to get the eddy dissipation rate as well

        if self.config['doppler']['scheme'] == 3 and \
            self.config['doppler']['turbulence_correction'] == 1:
            if file_h.check_if_variables_in_file(['EDR']):
                vars_to_load.extend(['EDR'])
            else:
                print('Necessary variable for correction of turbulence broadening'\
                      'eddy dissipitation rate'\
                      'was not found in file. No correction will be done')
                self.config['doppler']['turbulence_correction']=False

        if not base_vars_ok:
            print('Not all necessary variables could be found in file')
            print('For 1-moment scheme, the COSMO file must contain:')
            print('Temperature, Pressure, U-wind component, V-wind component,',
                  'W-wind component, and mass-densities (Q) for Vapour, Rain, Snow,',
                  ' Graupel, Ice cloud and Cloud water')
            print('For 2-moment scheme, the COSMO file must AS WELL contain:')
            print('Number densitities (QN) for Rain, Snow, Graupel and Hail ',
                    'as well as mass density (Q) of hail')
            sys.exit()
        elif base_vars_ok and not two_mom_vars_ok:
            print('Using 1-moment scheme')
            self.config['microphysics']['scheme'] = '1mom'
            self.current_microphys_scheme = '1mom'
        elif base_vars_ok and two_mom_vars_ok:
            vars_to_load.extend(BASE_VARIABLES_2MOM)
            print('Using 2-moment scheme')
            self.current_microphys_scheme = '2mom'


        # Read variables from GRIB file
        print('Reading variables ',vars_to_load,' from file')

        loaded_vars = file_h.get_variable(vars_to_load,get_proj_info=True,
                                          shared_heights=False,assign_heights=True,
                                          cfile_name=cfilename)
        self.dic_vars = loaded_vars #  Assign to class
        if 'N' in loaded_vars.keys():
            self.N=loaded_vars['N']
            self.dic_vars.pop('N',None) # Remove N from the variable dictionnary (we won't need it there)
        file_h.close()
        gc.collect()
        print('-------done------')

        # Check if lookup tables are deprecated
        if not self.output_variables == 'model_only':
            if not self.lut_sz or \
                self.current_microphys_scheme != \
                    self.config['microphysics']['scheme']:
                print('Loading lookup-tables for current specification')
                self.set_lut()
                self.config['microphysics']['scheme'] = self.current_microphys_scheme
        del loaded_vars

    def get_VPROF(self):
        # Check if model file has been loaded
        if self.dic_vars=={}:
            print('No model file has been loaded! Aborting...')
            return

        if self.config['radar']['type'] != 'ground':
            print('RHI profiles only possible for ground radars, please use\n'+\
                 ' get_GPM_swath instead')
            return []
        # Needs to be done in order to deal with Multiprocessing's annoying limitations
        global dic_vars, N, lut_sz, output_variables
        dic_vars, N, lut_sz, output_variables = self.define_globals()
        # Define list of angles that need to be resolved

        # Define  ranges
        rranges = constants.RANGE_RADAR

        # Initialize computing pool
        list_GH_pts = interpolation.get_profiles_GH(dic_vars,0.,90.,N=N)

        beam = doppler_scatter_sz.get_radar_observables(list_GH_pts,
                                                       lut_sz)

        # Threshold at given sensitivity
        beam = utilities.cut_at_sensitivity(beam,self.config['radar']['sensitivity'])

        if output_variables == 'all':
            beam=utilities.combine_beams((beam, interpolation.integrate_GH_pts(list_GH_pts)))

        del dic_vars
        del N
        del lut_sz
        gc.collect()

        simulated_sweep={'ranges':rranges,'pos_time':self.get_pos_and_time(),'data':beam}

        pyrad_instance=PyartRadopVProf(simulated_sweep)
        return  pyrad_instance


    def get_PPI(self, elevations, azimuths = [], az_step=-1, az_start=0, az_stop=359):

        # Check if model file has been loaded
        if self.dic_vars=={}:
            print('No model file has been loaded! Aborting...')
            return

        # Check if list of elevations is scalar
        if np.isscalar(elevations):
            elevations=[elevations]

        if self.config['radar']['type'] != 'ground':
            print('PPI profiles only possible for ground radars, please use')
            print(' get_GPM_swath instead')
            return []
        # Needs to be done in order to deal with Multiprocessing's annoying limitations
        global dic_vars, N, lut_sz, output_variables
        dic_vars, N, lut_sz, output_variables = self.define_globals()
        # Define list of angles that need to be resolved
        if az_step==-1:
            az_step=self.config['radar']['3dB_beamwidth']

        # Define list of angles that need to be resolved
        if not len(azimuths):
            # Define azimuths and ranges
            if az_start>az_stop:
                azimuths=np.hstack((np.arange(az_start,360.,az_step),np.arange(0,az_stop+az_step,az_step)))
            else:
                azimuths=np.arange(az_start,az_stop+az_step,az_step)

        # Define  ranges
        rranges = constants.RANGE_RADAR

        # Initialize computing pool
        pool = mp.Pool(processes = mp.cpu_count(),maxtasksperchild=1)
        m = mp.Manager()
        event = m.Event()

        list_sweeps=[]

        def worker(event,elev, azimuth):#
            print(azimuth)
            try:
                if not event.is_set():
                    list_GH_pts = interpolation.get_profiles_GH(dic_vars,azimuth, elev,N=N)

                    if output_variables in ['all','only_radar']:
                        output = doppler_scatter_sz.get_radar_observables(list_GH_pts,
                                                                       lut_sz)
                    if output_variables == 'only_model':
                        output =  interpolation.integrate_GH_pts(list_GH_pts)
                    elif output_variables == 'all':
                        output = utilities.combine_beams((output, interpolation.integrate_GH_pts(list_GH_pts)))

                    return output
            except:
                # Throw signal back
                raise
                event.set()

        for e in elevations: # Loop on the elevations
            func = partial(worker,event,e)
            list_beams = pool.map(func,azimuths)
            list_sweeps.append(list_beams)

        pool.close()
        pool.join()

        del dic_vars
        del N
        del lut_sz
        gc.collect()

        if not event.is_set():
            # Threshold at given sensitivity
            if output_variables in ['all','only_radar']:
                list_sweeps = utilities.cut_at_sensitivity(list_sweeps)

            simulated_sweep={'elevations':elevations,'azimuths':azimuths,
            'ranges':rranges,'pos_time':self.get_pos_and_time(),
            'data':list_sweeps}

            pyrad_instance = PyartRadop('ppi',simulated_sweep)

            return pyrad_instance

    def get_RHI(self, azimuths, elevations = [], elev_step=-1, elev_start=0, elev_stop=90):
        # Check if model file has been loaded
        if self.dic_vars=={}:
            print('No model file has been loaded! Aborting...')
            return

        # Check if list of azimuths is scalar
        if np.isscalar(azimuths):
            azimuths=[azimuths]

        if self.config['radar']['type'] != 'ground':
            print 'RHI profiles only possible for ground radars, please use'
            print ' get_GPM_swath instead'
            return []
        # Needs to be done in order to deal with Multiprocessing's annoying limitations
        global dic_vars, N, lut_sz, output_variables
        dic_vars, N, lut_sz, output_variables=self.define_globals()

        # Define list of angles that need to be resolved
        if not len(elevations):
            if elev_step==-1:
                elev_step=self.config['radar']['3dB_beamwidth']
            # Define elevation and ranges
            elevations=np.arange(elev_start,elev_stop+elev_step,elev_step)

        # Define  ranges
        rranges = constants.RANGE_RADAR

        # Initialize computing pool
        pool = mp.Pool(processes = mp.cpu_count(),maxtasksperchild=1)
        m = mp.Manager()
        event = m.Event()

        list_sweeps=[]

        def worker(event, azimuth, elev):
            print(elev)
            try:
                if not event.is_set():
                    list_GH_pts = interpolation.get_profiles_GH(dic_vars,azimuth, elev,N=N)

                    if output_variables in ['all','only_radar']:
                        output = doppler_scatter_sz.get_radar_observables(list_GH_pts,
                                                                       lut_sz)
                    if output_variables == 'only_model':
                        output =  interpolation.integrate_GH_pts(list_GH_pts)
                    elif output_variables == 'all':
                        output = utilities.combine_beams((output,
                                 interpolation.integrate_GH_pts(list_GH_pts)))

                    return output
            except:
                # Throw signal back
                raise
                event.set()

        for a in azimuths: # Loop on the o
            func=partial(worker,event,a) # Partial function
            list_beams = pool.map(func,elevations)
            list_sweeps.append(list_beams)

        pool.close()
        pool.join()

        del dic_vars
        del N
        del lut_sz

        if not event.is_set():
            # Threshold at given sensitivity
            if output_variables in ['all','only_radar']:
                list_sweeps = utilities.cut_at_sensitivity(list_sweeps)

            simulated_sweep={'elevations':elevations,'azimuths':azimuths,
            'ranges':rranges,'pos_time':self.get_pos_and_time(),
            'data':list_sweeps}

            pyrad_instance=PyartRadop('rhi',simulated_sweep)
            return  pyrad_instance

    def get_GPM_swath(self, GPM_file, band='Ku'):
        # Check if model file has been loaded
        if self.dic_vars=={}:
            print('No model file has been loaded! Aborting...')
            return

        # Assign correct frequencies and 3dB beamwidth, whatever user config
        cfg_copy = copy.deepcopy(self.config)
        if band == 'Ku' or band == 'Ku_matched':
            cfg_copy['radar']['frequency'] = constants.GPM_KU_FREQUENCY
        elif band == 'Ka' or band == 'Ka_matched':
            cfg_copy['radar']['frequency'] = constants.GPM_KA_FREQUENCY

        cfg_copy['radar']['3dB_beamwidth'] = constants.GPM_3DB_BEAMWIDTH
        cfg_copy['radar']['sensitivity'] = constants.GPM_SENSITIVITY
        cfg_copy['radar']['type'] = 'GPM'
        cfg_copy['radar']['radial_resolution'] = \
            constants.GPM_RADIAL_RES_KA if band == 'Ka' \
                else constants.GPM_RADIAL_RES_KU

        self.update_config(cfg_copy, check = False)

        # Needs to be done in order to deal with Multiprocessing's annoying limitations
        global dic_vars, N, lut_sz, output_variables
        dic_vars, N, lut_sz, output_variables = self.define_globals()

        az,elev,rang,coords_GPM = GPM_simulator.get_GPM_angles(GPM_file,band)
        # Initialize computing pool
        pool = mp.Pool(processes = mp.cpu_count(), maxtasksperchild=1)
        m = mp.Manager()
        event = m.Event()


        def worker(event,params):
            try:
                if not event.is_set():
                    azimuth=params[0]
                    elev=params[1]

                    """ For some reason modifying self.config instead of
                    cfg.CONFIG throws in error about not being able to pickle
                    the pycosmo variables. This is indeed very weird and I
                    have not been able to figure out why...However since
                    self.config is just a shallow copy of cfg.CONFIG, it doesn't
                    really matter...
                    """

                    # Update GPM position and range vector
                    cfg.CONFIG['radar']['range'] = params[2]
                    cfg.CONFIG['radar']['coords'] = [params[3], params[4], params[5]]

                    list_GH_pts = interpolation.get_profiles_GH(dic_vars,azimuth,
                                                                elev,N=N)

                    output = doppler_scatter_sz.get_radar_observables(list_GH_pts,lut_sz)

                    if output_variables in ['all','only_radar']:
                        output = doppler_scatter_sz.get_radar_observables(list_GH_pts,lut_sz)
                    if output_variables == 'only_model':
                        output =  interpolation.integrate_GH_pts(list_GH_pts)
                    elif output_variables == 'all':
                        output = utilities.combine_beams((output,
                                interpolation.integrate_GH_pts(list_GH_pts)))

                    return output
            except:
                # Throw signal back
                raise
                event.set()

        dim_swath = az.shape


        list_beams=[]
        for i in range(dim_swath[0]):
            print('running slice '+str(i))
            # Update radar position
            c0 = np.repeat(coords_GPM[i,0],len(az[i]))
            c1 = np.repeat(coords_GPM[i,1],len(az[i]))
            c2 = np.repeat(coords_GPM[i,2],len(az[i]))
            worker_partial = partial(worker,event)
            list_beams.extend(map(worker_partial,zip(az[i],elev[i],
                                                          rang[i],c0,c1,c2)))

        pool.close()
        pool.join()

        del dic_vars
        del N
        del lut_sz
        gc.collect()

        if not event.is_set():
            # Threshold at given sensitivity
            if output_variables in ['all','only_radar']:
                list_beams = utilities.cut_at_sensitivity(list_beams)

            list_beams_formatted = GPM_simulator.SimulatedGPM(list_beams, dim_swath, band)
            return list_beams_formatted


    def set_config(self,options_file):

        cfg.init(options_file) # Initialize options with 'options_radop.txt'

if __name__=='__main__':


#    gpm_file = '/ltedata/COSMO/GPM_data/2014-08-13-02-28.HDF5'
#    cosmo_file = '/ltedata/COSMO/GPM_1MOM/2014-08-13-02-28.grb'
##    #
##    #
##
#    a  = RadarOperator()
#    a.load_model_file(cosmo_file,cfilename = '/ltedata/COSMO/GPM_FULL_1MOM/2014-04-04-16-18/lfff00000000c')
#    cfg.CONFIG['integration']['nh_GH'] = 3
#    cfg.CONFIG['integration']['nv_GH'] = 3
#    cfg.CONFIG['attenuation']['correction'] = 1
#    swath = a.get_GPM_swath(gpm_file,'Ku')
##
#    from cosmo_pol.gpm.GPM_simulator import compare_operator_with_GPM
#    a,b,c = compare_operator_with_GPM(swath,gpm_file)

#    import matplotlib.pyplot as plt
#    from cosmo_pol.radar import small_radar_db, pyart_wrapper
##    import glob
#
#    print(constants.WAVELENGTH)
#vsimeono
    files_c = pc.get_model_filenames('/ltedata/COSMO/Multifractal_analysis/case2014040802_ONEMOM/')
    a=RadarOperator(options_file='/data/cosmo_pol/option_files/MXPOL_RHI_PAYERNE.yml')
####
#    a.config['microphysics']['with_melting'] = 1
#    a.config['microphysics']['with_ice_crystals'] = 1
#    a.config['microphysics']['scattering']='tmatrix_new'
#    a.config['radar']['radial_resolution'] = 75
#    a.config['radar']['3dB_beamwidth'] = 1.41
#    a.config['integration']['scheme']='ml'
#    a.config['integration']['nh_GH']=1
#    a.config['integration']['nv_GH']=5
#    a.config['doppler']['scheme'] = 3
#    a.config['radar']['range']=45000
#    a.change_frequency(9.41)
#    a.config['doppler']['turbulence_correction'] = 0
#    a.config['integration']['weight_threshold'] = 1.
##    a.change_frequency(35.6)
##    print(constants.WAVELENGTH)
##    a.config['radar']['sensitivity']= [-35,10000]
##
#    import matplotlib.pyplot as plt
#    a.load_model_file(files_c['h'][60],cfilename = '/ltedata/COSMO/Multifractal_analysis/case2014040802_ONEMOM/lfsf00000000c')
#    r=a.get_PPI(elevations=[3])
#    plt.imshow(r.get_field(0,'ZH'))
#    from cosmo_pol.radar.pyart_wrapper import RadarDisplay
#
#    from scipy.ndimage import gaussian_filter


#    a.config['integration']['scheme']=1
#    a.config['integration']['nv_GH']=1


#    plt.plot(r.get_field(0,'ZDR')[0])
#    part_with_ml = np.where(r.get_field(0,'QmS_v')[0]>0)[0]
#    zh_ml = r.get_field(0,'ZH')[0][part_with_ml]
#    dist = r.range['data'][part_with_ml]
#    ang = 3.
#    sigma=1.41/(2*np.sqrt(2*np.log(2)))
#    dvert = sigma/180.*np.pi *dist
#    nbins_h = (dvert / np.tan(ang/180.*np.pi)) / 100.
#    from scipy.ndimage.filters import gaussian_filter

#    fuu = gaussian_filter(zh_ml,42)
#    display =RadarDisplay(r)
#    plt.figure(figsize=(12,10))
#    plt.subplot(2,2,1)
#    display.plot('ZH',0,vmin=0,vmax=40, title='ZH')
#    plt.subplot(2,2,2)
#    display.plot('ZDR',0,vmin=0,vmax=3, title='ZDR')
#    plt.subplot(2,2,3)
#    display.plot('KDP',0,vmin=0,vmax=3, title='KDP')
#    plt.savefig('ex_soft_ml_ppi_meth2.png',dpi=200,bbox_inches='tight')
##
##    plt.figure()
##    if a.config['microphysics']['with_melting']:
##        aa = r.get_field(0,'QR_v')
##        b = r.get_field(0,'QS_v')
##        c =r.get_field(0,'QG_v')
##        d = r.get_field(0,'QmS_v')
##        e = r.get_field(0,'QmG_v')
##        plt.imshow(d+e)
##    else:
##        aa = r.get_field(0,'QR_v')
##        b = r.get_field(0,'QS_v')
##        c =r.get_field(0,'QG_v')
##        plt.imshow(a+b+c)
#
#    a.config['doppler']['scheme']=3
##    a.config['integration']['weight_threshold'] = 1.
#    r2 = a.get_PPI(elevations = [90],azimuths = [0])
#    plt.figure()
#    varray = r2.instrument_parameters['varray']['data']
#    rrange = r2.range['data']
#    plt.contourf(varray,rrange,10*np.log10(r2.get_field(0,'DSPECTRUM')[0]),levels=np.arange(-40,35,1),extend='max')
#    plt.colorbar()
#    plt.xlim([-12,0])
#    plt.ylim([0,10000])

#    a.config['integration']['scheme']='ml2'
#    r2 = a.get_PPI(elevations = [90],azimuths = [0])
#    plt.figure()
#    varray = r2.instrument_parameters['varray']['data']
#    rrange = r2.range['data']
#    plt.contourf(varray,rrange,10*np.log10(r2.get_field(0,'DSPECTRUM')[0]),levels=np.arange(-40,35,1),extend='max')
#    plt.colorbar()
#    plt.xlim([-12,0])
#    plt.ylim([0,10000])

#    plt.savefig('ex_spectrum_meth2.png',dpi=200,bbox_inches='tight')
#    plt.imshow(10*np.log10(r.get_field(0,'DSPECTRUM')[0]))
#    display.plot('fwet_mS',0,vmin=0,vmax=1, title='Raw Reflectivity')

#    plt.savefig('ex_ml.png',dpi=200,bbox_inches='tight')
#    display.plot_vprof('DSPECTRUM', 0, vmin=-50, vmax=10.,
#             colorbar_label='', title='Raw Reflectivity')
#    plt.plot(r.get_field(0,'RVEL')[0])
#    plt.xlim([10,30])
#    import matplotlib.pyplot as plt
#
#    a.config['microphysics']['with_melting']=False
#
#    r2=a.get_RHI(azimuths=[2],elevations=[90])
#
#
#    plt.plot(r.get_field(0,'QR_v')[0])
#    plt.plot(r2.get_field(0,'QR_v')[0])
#
#    plt.plot(r.get_field(0,'QS_v')[0])
#    plt.plot(r2.get_field(0,'QS_v')[0])

#    plt.ylim([0,10])
#
#    gpm_s = a.get_GPM_swath('/ltedata/COSMO/GPM_data/2014-10-20-22-10.HDF5','Ku')
#    a.close()

#    r2=a.get_RHI(azimuths=[2])


#    plt.figure()
#    plt.imshow(r2[0].get_field(0,'ZDR')-r[0].get_field(0,'ZDR'))


#    f1 = '/media/wolfensb/Storage/cosmo_pol/validation/DATA_ALL_RADAR_MU_2/20130201/model_1mom/MODEL_70.p'
#    f2 = '/media/wolfensb/Storage/cosmo_pol/validation/DATA_WITH_ICE/20130201/model_1mom/MODEL_70.p'
#    import pickle
#    f1 = pickle.load(open(f1,'r'))
#    f2 = pickle.load(open(f2,'r'))

#    a.load_model_file(files_c['h'][30],cfilename = '/ltedata/COSMO/Multifractal_analysis/case2014040802_ONEMOM/lfsf00000000c')
#    a.config.CONFIG['integration']['nh_GH']=1
#    a.config.CONFIG['integration']['nv_GH']=1
#    r1=a.get_PPI(elevations=[5])
#
#    a.load_model_file(files_c2['h'][30],cfilename = '/ltedata/COSMO/Multifractal_analysis/case2014040802_TWOMOM/lfsf00000000c')
#    a.config.CONFIG['integration']['nh_GH']=1
#    a.config.CONFIG['integration']['nv_GH']=1
#    r2=a.get_PPI(elevations=[5])
#
#    plt.figure()
#    plt.imshow(r1.get_field(0,'ZH'))
#
#    plt.figure()
#    plt.imshow(r2.get_field(0,'ZH'))


    #a.config.CONFIG['integration']['scheme'] = 1
    #a.config.CONFIG['integration']['weight_threshold']=0.99
    #a.config.CONFIG['doppler']['scheme'] = 3
    #a.config.CONFIG['doppler']['turbulence_correction'] = 1
    #a.config.CONFIG['radar']['radial_resolution'] = 75
    #a.config.CONFIG['radar']['range'] = 50000


    #a.load_model_file(files_c['h'][40],cfilename = '/ltedata/COSMO/Multifractal_analysis/case2014040802_ONEMOM/lfsf00000000c')

    #r1=a.get_RHI(azimuths=0,elevations=[90])

#    from cosmo_pol.radar.pyart_wrapper import RadarDisplay
#
#    display = RadarDisplay(r1)
#    display.plot('ZH', 0,title='Hor. Reflectivity')
#    plt.ylim([0,10])
#    a.config.CONFIG['attenuation']['correction']=0.
#    r2=a.get_PPI(elevations = [2],azimuths=[0])
#
#    sz_psd_integ = np.zeros((12,))
#    import pickle
#    import gzip
#    lut = pickle.load(gzip.open('/media/wolfensb/Storage/cosmo_pol/lookup/final_lut/all_luts_SZ_f_5_6_1mom.pz','rb'))
#    from cosmo_pol.hydrometeors.hydrometeors import create_hydrometeor
#    gr = create_hydrometeor('G',scheme='1mom')
#    D = lut['G'].axes[2]
#
#    gr.set_psd(np.array([2.590333838980996e-05]))
#    sz = lut['G'].lookup_line(e = 2, t = 273)
#
#    dD = D[1]-D[0]
#    sz_psd_integ += np.einsum('ij,i->j',sz,gr.get_N(D)) * dD
#    print(np.einsum('ij,i->j',sz,gr.get_N(D)) * dD )
#    print(np.trapz(sz[:,1]*gr.get_N(D),dx=dD))
#
#    sn = create_hydrometeor('S',scheme='1mom')
#    D = lut['S'].axes[2]
#
#    sn.set_psd(np.array([273]),np.array([r1.get_field(0,'QS_v')[0][20]]))
#    sz = lut['S'].lookup_line(e = 2, t = 273)
#
#    dD = D[1]-D[0]
#    sz_psd_integ += np.einsum('ij,i->j',sz,sn.get_N(D)) * dD
#    print(np.einsum('ij,i->j',sz,sn.get_N(D)) * dD )
#
#    r = create_hydrometeor('R',scheme='1mom')
#    D = lut['R'].axes[2]
#
#    r.set_psd(np.array([r1.get_field(0,'QR_v')[0][20]]))
#    sz = lut['R'].lookup_line(e = 2, t = 273)
#
#    dD = D[1]-D[0]
#    sz_psd_integ += np.einsum('ij,i->j',sz,r.get_N(D)) * dD
#    print(np.einsum('ij,i->j',sz,r.get_N(D)) * dD )
#
#    ZH = constants.WAVELENGTH**4/(np.pi**5*constants.KW)*2*np.pi*(sz_psd_integ[0]-sz_psd_integ[1]-sz_psd_integ[2]+sz_psd_integ[3])


#    tictoc.tic()
#    a.config.CONFIG['integration']['scheme'] = 6
#    a.config.CONFIG['integration']['weight_threshold']=1.
#    r1=a.get_RHI(azimuths=0,elevations = [2])
#    a.config.CONFIG['integration']['weight_threshold']=0.9
#    a.config.CONFIG['integration']['scheme'] = 2
#    r2=a.get_RHI(azimuths=0,elevations = [2])
#    a.config.CONFIG['integration']['scheme'] = 1
#    r3=a.get_RHI(azimuths=0,elevations = [2])
#    a.config.CONFIG['integration']['weight_threshold']=0.999
#    a.config.CONFIG['integration']['scheme'] = 4
#    r4=a.get_RHI(azimuths=0,elevations = [2])
#    a.config.CONFIG['integration']['scheme'] = 5
#    a.config.CONFIG['integration']['nv_GH']=13
#    a.config.CONFIG['integration']['weight_threshold']=0.9
#    r5=a.get_RHI(azimuths=0,elevations = [2])
#
#    import matplotlib.pyplot as plt
#    plt.plot(r1.get_field(0,'ZH').data.ravel())
#    plt.plot(r2.get_field(0,'ZH').data.ravel())
#    plt.plot(r3.get_field(0,'ZH').data.ravel())
#    plt.plot(r4.get_field(0,'ZH').data.ravel())
#    plt.plot(r5.get_field(0,'ZH').data.ravel())
#    plt.legend(['1','2','3','4','5'])
#
#    r1=a.get_PPI(elevations=5)
#
#
#    display = RadarDisplay(r1, shift=(0.0, 0.0))
#    display.plot('ZH', 0, 100000,colorbar_flag=True,title="ZH (radar)",mask_outside = True)
#
#    time = pycosmo.get_time_from_COSMO_filename(files_c['h'][10])
#
#    aaaa = small_radar_db.CH_RADAR_db()
#    files = aaaa.query(date=[str(time)],radar='D',angle=1)
#    rad = pyart_wrapper.PyradCH(files[0][0].rstrip(),False)
#    display = RadarDisplay(rad, shift=(0.0, 0.0))
#    display.plot('ZDR', 0, 150000,colorbar_flag=True,title="ZH (radar)",vmin=0,mask_outside = True)
#    import matplotlib.pyplot as plt
#    for i,f in enumerate(['/ltedata/COSMO/Validation_operator/case2015081312_TWOMOM/lfsf00153500']):
#        time = pc.get_time_from_COSMO_filename(f)
#        aaaa = small_radar_db.CH_RADAR_db()
#        files = aaaa.query(date=[str(time)],radar='D',angle=1)
#
#        if i<-9:
#	    continue
##        rad = pyart_wrapper.PyradCH(files[0].rstrip(),False)
##
##        rad.correct_velocity()
##        display = RadarDisplay(rad, shift=(0.0, 0.0))
##
##        plt.figure(figsize=(14,6))
##        plt.subplot(1,2,1)
##        display.plot('Z', 0, 150000,colorbar_flag=True,title="ZH (radar)",vmin=0,mask_outside = True)
##        display.set_limits(xlim=(-150,150),ylim=(-150,150))
##
##        plt.subplot(1,2,2)
##        display.plot('V_corr', 0, 150000,colorbar_flag=True,title="Mean Doppler velocitiy (radar)",mask_outside = True,vmin=-25,vmax=25)
##        display.set_limits(xlim=(-150,150),ylim=(-150,150))
##
##        plt.savefig('example_ppi_radar_'+str(i)+'.png',dpi=200,bbox_inches='tight')
##
##        #
##
#        cfg.CONFIG['doppler']['scheme']=1
#        a.load_model_file(f,cfilename = '/ltedata/COSMO/Multifractal_analysis/case2015081312_TWOMOM/lfsf00000000c')
#        r=a.get_PPI(1)
##
#        fig = plt.figure()
##
#        display = RadarDisplay(r, shift=(0.0, 0.0))
#        plt.figure(figsize=(14,12))
#        plt.subplot(2,2,1)
#        display.plot('ZH', 0, 150000,colorbar_flag=True,title="ZH")
#        plt.subplot(2,2,2)
#        display.plot('RVEL', 0, 150000,colorbar_flag=True,title="Mean Doppler velocity")
#        plt.subplot(2,2,3)
#        display.plot('ZDR', 0,150000, colorbar_flag=True,title="Diff. reflectivity",vmax=3)
#        plt.subplot(2,2,4)
#        display.plot('KDP', 0, 150000,colorbar_flag=True,title="Spec. diff. phase",vmax=0.5)
#        plt.savefig('example_ppi_'+str(i)+'.png',dpi=200,bbox_inches='tight')
##
#    r=a.get_GPM_swath('./GPM_files/2014-08-13-02-28.HDF5','Ku')
#
#    ZH_ground,ZH_everywhere=compare_operator_with_GPM(r,'./GPM_files/2014-08-13-02-28.HDF5')
#    import pickle
##    pickle.dump(a,open('ex_beams.txt','wb'))
##    pickle.dump(b,open('ex_output_GPM.txt','wb'))
#    g='./GPM_files/2014-08-13-02-28.HDF5'
#    import h5py
#    group='NS'
#    gpm_f=h5py.File(g,'r')
#    lat_2D=gpm_f[group]['Latitude'][:]
#    lon_2D=gpm_f[group]['Longitude'][:]



#    a.load_model_file('./cosmo_files/GRIB/2014-05-13-20.grb')
#    a.load_model_file('/ltedata/COSMO/case2014040802_PAYERNE_analysis_ONEMOM/lfsf00105000')
#
#    results=a.get_PPI(2)
#    o=polar_to_cartesian_PPI(results[0],results[1])
#    plt.contourf(o[2]['ZH'],levels=np.arange(0,40,5),extend='both')
#    c=10*np.log10(o[2]['ZH'])
#    plt.contourf(c,levels=np.arange(0,40,1),extend='max')
#    c=np.log10(o[2]['KDP'])
#    plt.contourf(c,extend='max')
#    plt.colorbar()
#    cfg.CONFIG['attenuation_correction']=False
#    results=a.get_PPI(5)
#    o=polar_to_cartesian_PPI(results[0],results[1])
#    d=10*np.log10(o[2]['ZH'])
#    plt.figure()
#    plt.contourf(d-c,levels=np.arange(0,2,0.1))
#    results=a.get_PPI(5)
#    o=polar_to_cartesian_PPI(results[0],results[1])
#    plt.figure()
#    plt.contourf(c,levels=np.arange(0,40,0.5),extend='both')
#
#    d=10*np.log10(o[2]['ZH'])
#    plt.figure()
#    plt.contourf(d,levels=np.arange(0,40,0.5),extend='both')
#
#    plt.figure()
#    plt.contourf(d-c)

#    print np.nansum(np.array([l.values['ZH'] for l in list_beams]))
#    cPickle.dump((elev,list_beams),open('test.p','w'))

#    o=polar_to_cartesian_PPI(az,list_beams)
#    plt.figure()
#    plt.imshow(o[2]['ZH'])
#    cPickle.dump((az,list_beams),open('test.p','w'))

#    a=cPickle.load(open('test.p','r'))
#    o=polar_to_cartesian_PPI(az,list_beams)
#    plt.imshow(10*np.log10(o[2]['ZH']))
#    plt.figure()
#    tic()
#    az,list_beams=a.get_PPI(20)
#    toc()
#    o=polar_to_cartesian_PPI(az,list_beams)
#    plt.imshow(10*np.log10(o[2]['ZH']))
#    plt.colorbar()
