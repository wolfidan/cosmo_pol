# -*- coding: utf-8 -*-

""" radar_operator.py: Defines the main radar operator class RadarOperator
which is the only one you should access as a user. This class allows to
compute GPM swaths, PPI scans, vertical profiles and RHI profiles"""

__author__ = "Daniel Wolfensberger"
__copyright__ = "Copyright 2017, COSMO_POL"
__credits__ = ["Daniel Wolfensberger"]
__license__ = "GPL"
__version__ = "1.0"
__maintainer__ = "Daniel Wolfensberger"
__email__ = "daniel.wolfensberger@epfl.ch"


# Global imports
from functools import partial
import multiprocess as mp
import numpy as np
import copy
import pycosmo as pc
import gc
from textwrap import dedent

# Local imports
from cosmo_pol.radar import PyartRadop, get_GPM_angles, SimulatedGPM
from cosmo_pol.config import cfg
from cosmo_pol.interpolation import get_interpolated_radial, integrate_radials


from cosmo_pol.constants import global_constants as constants
from cosmo_pol.lookup import load_all_lut
from cosmo_pol.utilities import combine_subradials

BASE_VARIABLES=['U','V','W','QR_v','QS_v','QG_v','QI_v','RHO','T']
BASE_VARIABLES_2MOM=['QH_v','QNH_v','QNR_v','QNS_v','QNG_v','QNI_v']

EXHAUSTIVE = False

if EXHAUSTIVE:
    from cosmo_pol.scatter.doppler_scatter_exhaustive import get_radar_observables, cut_at_sensitivity
else:
    from cosmo_pol.scatter import get_radar_observables, cut_at_sensitivity


class RadarOperator(object):
    def __init__(self, options_file = None, output_variables = 'all'):
        '''
        Creates a RadarOperator class instance that can be used to compute
        radar profiles (PPI, RHI, GPM)
        Args:
            options_file: a .yml file containing the user configuration
                (see examples in the options_files folder)
            output_variables: can be either 'all', 'only_model', or 'only_radar'
                if 'only_model', only COSMO native variables used for the
                radar operator (e.g. temp, concentrations, etc.) will be
                returned at the radar gates, no radar observables will be
                computed. This is fast and can be of use in particular
                circonstances
                if 'only_radar', only radar simulated radar observables will
                be returned at the radar gates (i.e. ZH, ZV, ZDR, KDP,...)
                if 'all', both radar observables and model variables will
                be returned at the radar gates

        Returns:
            A RadarOperator class instance
        '''


        # delete the module's globals
        print('Reading options defined in options file')
        cfg.init(options_file) # Initialize options with specified file

        self.current_microphys_scheme = '1mom'
        self.dic_vars = None
        self.N = 0 # atmospheric refractivity

        self.config = cfg.CONFIG
        self.lut_sz = None


        if output_variables in ['all','only_model','only_radar']:
            self.output_variables = output_variables
        else:
            self.output_variables = 'all'
            msg = """Invalid output_variables input, must be either
            'all', 'only_model' or 'only_radar'
            """
            print(msg)

    def close(self):
        '''
        Closes the RadarOperator class instance and deletes its content
        '''
        try:
            del dic_vars, N, lut_sz, output_variables
        except:
            pass
        self.config = None
        cfg.CONFIG = None

    @property
    def config(self):
        return copy.deepcopy(self.__config)

    @config.setter
    def config(self, config_dic):
        '''
        Update the content of the user configuration, applies the necessary
        changes to the constants and if needed reloads the lookup tables
        The best way to use this function is to retrieve the current
        configuration using  deepcopy: copy = copy.deepcopy(radop.config)
        then modify this copy: ex. copy['radar']['frequency'] = 9.41
        and then modify config.
		config = deepcopy.copy(radop.config)
		# change config
		radop.config = config

        Args:
            config_dic: a dictionnary specifying the new configuration to
                use.
            check: boolean to state if the specified new configuration
                dictionary should be checked and missing/invalid values should
                be replaced (as when a new .yml file is loaded). True by
                default
        '''
        # if check == False, no sanity check will be done, do this only
        # if you are sure of what you are doing

        print('Loading new configuration...')
        checked_config = cfg.sanity_check(config_dic)

        if hasattr(self, 'config'):
            # If frequency was changed or if melting is considered and not before
            #  reload appropriate lookup tables
            if checked_config['radar']['frequency'] != \
                 self.config['radar']['frequency'] or \
                 checked_config['microphysics']['with_melting'] != \
                 self.config['microphysics']['with_melting'] \
                 or not self.lut_sz:

                 print('Reloading lookup tables...')
                 self.__config = checked_config
                 cfg.CONFIG = checked_config
                 # Recompute constants
                 constants.update() # Update constants now that we know user config
                 self.set_lut()

            else:

                self.__config = checked_config
                cfg.CONFIG = checked_config
                # Recompute constants
                constants.update() # Update constants now that we know user config
        else:
            self.__config = checked_config
            cfg.CONFIG = checked_config
            constants.update()
            self.set_lut()

    def set_lut(self):
        '''
        Load a new set of lookup tables for the current radar operator
        based on the user configuration
        '''
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
        '''
        Get the position of the radar and time
        '''
        latitude = self.config['radar']['coords'][0]
        longitude = self.config['radar']['coords'][1]
        altitude = self.config['radar']['coords'][2]
        time=self.dic_vars['T'].attributes['time'] # We could read any variable, T or others

        out={'latitude':latitude,'longitude':longitude,'altitude':altitude,\
        'time':time}

        return out


    def define_globals(self):
        '''
        This is used in the parallelization to get all global variables
        '''
        global output_variables
        output_variables = self.output_variables

        global dic_vars
        dic_vars=self.dic_vars

        global N
        N=self.N

        global lut_sz
        lut_sz = self.lut_sz

        return dic_vars, N, lut_sz, output_variables

    def load_model_file(self, filename, cfilename = None):
        '''
        Loads data from a COSMO file, which is a pre-requirement for
        simulating radar observables

        Args:
            filename: the name of the COSMO GRIB hybrid level file
            cfilename: The name the of corresponding c-file, the c-file contains
                all constant variables such as the height of all grid cells,
                the altitude, etc. Usually it is in the same folder as the
                COSMO file
        '''
        file_h = pc.open_file(filename)
        vars_to_load = copy.deepcopy(BASE_VARIABLES)

        # Check if necessary variables are present in file

        base_vars_ok = file_h.check_if_variables_in_file(['P','T','QV','QR','QC','QI','QS','QG','U','V','W'])
        two_mom_vars_ok = file_h.check_if_variables_in_file(['QH','QNH','QNR','QNS','QNG'])
       
        if self.config['refraction']['scheme'] == 2:
            if file_h.check_if_variables_in_file(['T','P','QV']):
                vars_to_load.extend('N')
            else:
                msg = '''
                Necessary variables for computation of atm. refractivity:
                Pressure, Water vapour mass density and temperature
                were not found in file. Using 4/3 method instead.
                '''
                print(dedent(msg))
                self.config['refraction_method']='4/3'

        # To consider the effect of turbulence we need to get the eddy dissipation rate as well

        if self.config['doppler']['scheme'] == 3 and \
            self.config['doppler']['turbulence_correction'] == 1:
            if file_h.check_if_variables_in_file(['EDR']):
                vars_to_load.extend(['EDR'])
            else:
                msg = '''
                Necessary variable for correction of turbulence broadening:
                Eddy dissipitation rate
                was not found in file. No  turbulence correction will be done.
                '''
                print(dedent(msg))
                self.config['doppler']['turbulence_correction']=False

        if not base_vars_ok:
            msg = '''
            Not all necessary variables could be found in file
            For 1-moment scheme, the COSMO file must contain
            Temperature, Pressure, U-wind component, V-wind component,
            W-wind component, and mass-densities (Q) for Vapour, Rain, Snow,
            Graupel, Ice cloud
            For 2-moment scheme, the COSMO file must AS WELL contain:
            Number densitities (QN) for Rain, Snow, Graupel and Hail
            as well as mass density (Q) of hail
            '''
            raise ValueError(dedent(msg))

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
        '''
        Simulates a 90° elevation profile based on the user configuration

        Returns:
            A vertical profile at 90° elevation, in the form of a PyART
            class
        '''
        # Check if model file has been loaded
        if self.dic_vars=={}:
            print('No model file has been loaded! Aborting...')
            return

        # Needs to be done in order to deal with Multiprocessing's annoying limitations
        global dic_vars, N, lut_sz, output_variables
        dic_vars, N, lut_sz, output_variables = self.define_globals()
        # Define list of angles that need to be resolved

        # Define  ranges
        rranges = constants.RANGE_RADAR

        # Initialize computing pool
        list_GH_pts = get_interpolated_radial(dic_vars, 0., 90., N = N)

        beam = get_radar_observables(list_GH_pts,
                                                       lut_sz)

        # Threshold at given sensitivity
        beam = cut_at_sensitivity(beam,self.config['radar']['sensitivity'])

        if output_variables == 'all':
            beam= combine_subradials((beam, integrate_radials(list_GH_pts)))

        del dic_vars
        del N
        del lut_sz
        gc.collect()

        simulated_sweep={'ranges':rranges,
                         'pos_time':self.get_pos_and_time(),
                         'data':beam}

        pyrad_instance = PyartRadop(simulated_sweep)
        return  pyrad_instance


    def get_PPI(self, elevations, azimuths = None, az_step = None, az_start = 0,
                az_stop = 359):
        '''
        Simulates a PPI scan based on the user configuration
        Args:
            elevations: a single scalar or a list of elevation angles in
                degrees. If a list is provided, the output will consist
                of several PPI scans (sweeps in the PyART class)
            azimuths: (Optional) a list of azimuth angles in degrees
            az_start az_step, az_stop: (Optional) If 'azimuths' is undefined
                these three arguments will be used to create a list of azimuths
                angles. Defaults are (0, 3dB_beamwidth, 359)
        Returns:
            A PPI profile at the specified elevation(s), in the form of a PyART
            class. To every elevation angle corresponds at sweep
        '''
        # Check if model file has been loaded
        if self.dic_vars=={}:
            print('No model file has been loaded! Aborting...')
            return

        # Check if list of elevations is scalar
        if np.isscalar(elevations):
            elevations=[elevations]

        # Needs to be done in order to deal with Multiprocessing's annoying limitations
        global dic_vars, N, lut_sz, output_variables
        dic_vars, N, lut_sz, output_variables = self.define_globals()
        # Define list of angles that need to be resolved
        if az_step == None:
            az_step=self.config['radar']['3dB_beamwidth']

        # Define list of angles that need to be resolved
        if np.any(azimuths == None):
            # Define azimuths and ranges
            if az_start>az_stop:
                azimuths=np.hstack((np.arange(az_start, 360., az_step),
                                    np.arange(0, az_stop + az_step, az_step)))
            else:
                azimuths=np.arange(az_start, az_stop + az_step, az_step)

        # Define  ranges
        rranges = constants.RANGE_RADAR

        # Initialize computing pool
        pool = mp.Pool(processes = mp.cpu_count(), maxtasksperchild=1)
        m = mp.Manager()
        event = m.Event()

        list_sweeps=[]
        def worker(event,elev, azimuth):#
            print(azimuth)
            try:
                if not event.is_set():
                    list_subradials = get_interpolated_radial(dic_vars,
                                                          azimuth,
                                                          elev,
                                                          N)
                    if output_variables in ['all','only_radar']:
                        output = get_radar_observables(list_subradials, lut_sz)
                    if output_variables == 'only_model':
                        output =  integrate_radials(list_subradials)
                    elif output_variables == 'all':
                        output = combine_subradials((output,
                                 integrate_radials(list_subradials)))

                    return output
            except:
                # Throw signal back
                raise
                event.set()

        for e in elevations: # Loop on the elevations
            func = partial(worker,event,e)
            list_beams = pool.map(func,azimuths)
            list_sweeps.append(list(list_beams))

        pool.close()
        pool.join()

        del dic_vars
        del N
        del lut_sz
        gc.collect()

        if not event.is_set():
            # Threshold at given sensitivity
            if output_variables in ['all','only_radar']:
                list_sweeps = cut_at_sensitivity(list_sweeps)

            simulated_sweep={'elevations':elevations,'azimuths':azimuths,
            'ranges':rranges,'pos_time':self.get_pos_and_time(),
            'data':list_sweeps}

            pyrad_instance = PyartRadop('ppi',simulated_sweep)

            return pyrad_instance

    def get_RHI(self, azimuths, elevations = None, elev_step = None,
                                            elev_start = 0, elev_stop = 90):
        '''
        Simulates a RHI scan based on the user configuration
        Args:
            azimuths: a single scalar or a list of azimuth angles in
                degrees. If a list is provided, the output will consist
                of several RHI scans (sweeps in the PyART class)
            elevations: (Optional) a list of elevation angles in degrees
            elev_start elev_step, elev_stop: (Optional) If 'elevations' is
                undefinedthese three arguments will be used to create a list
                of elevations angles. Defaults are (0, 3dB_beamwidth, 359)
        Returns:
            A RHI profile at the specified elevation(s), in the form of a PyART
            class. To every azimuth angle corresponds at sweep
        '''
        # Check if model file has been loaded
        if self.dic_vars=={}:
            print('No model file has been loaded! Aborting...')
            return

        # Check if list of azimuths is scalar
        if np.isscalar(azimuths):
            azimuths=[azimuths]


        # Needs to be done in order to deal with Multiprocessing's annoying limitations
        global dic_vars, N, lut_sz, output_variables
        dic_vars, N, lut_sz, output_variables=self.define_globals()

        # Define list of angles that need to be resolved
        if np.any(elevations == None):
            if elev_step == None:
                elev_step = self.config['radar']['3dB_beamwidth']
            # Define elevation and ranges
            elevations = np.arange(elev_start, elev_stop + elev_step,
                                   elev_step)


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
                    list_subradials = get_interpolated_radial(dic_vars,
                                                          azimuth,
                                                          elev,
                                                          N)

                    if output_variables in ['all','only_radar']:
                        output = get_radar_observables(list_subradials, lut_sz)
                    if output_variables == 'only_model':
                        output =  integrate_radials(list_subradials)
                    elif output_variables == 'all':
                        output = combine_subradials((output,
                                 integrate_radials(list_subradials)))

                    return output
            except:
                # Throw signal back
                raise
                event.set()

        for a in azimuths: # Loop on the o
            func = partial(worker, event, a) # Partial function
            list_beams = pool.map(func,elevations)
            list_sweeps.append(list(list_beams))

        pool.close()
        pool.join()

        del dic_vars
        del N
        del lut_sz

        if not event.is_set():
            # Threshold at given sensitivity
            if output_variables in ['all','only_radar']:
                list_sweeps = cut_at_sensitivity(list_sweeps)

            simulated_sweep={'elevations':elevations,'azimuths':azimuths,
            'ranges':rranges,'pos_time':self.get_pos_and_time(),
            'data':list_sweeps}

            pyrad_instance = PyartRadop('rhi',simulated_sweep)
            return  pyrad_instance

    def get_GPM_swath(self, GPM_file, band = 'Ku'):
        '''
        Simulates a GPM swath
        Args:
            GPM_file: a GPM-DPR file in the HDF5 format
            band: can be either 'Ka', 'Ku', 'Ku_matched', 'Ka_matched'
                'Ka' will be at the location of the HS (high sensitivity)
                coordinates and at 35.6 GHz
                'Ka_matched' will be at the location of the MS (matched scan)
                coordinates and at 35.6 GHz
                'Ku' will be at the location of the NS (high sensitivity)
                coordinates and at 13.6 GHz
                'Ku_matched' will be at the location of the MS (matched scan)
                coordinates and at 13.6 GHz

        Returns:
            An instance of the SimulatedGPM class (see gpm_wrapper.py) which
            contains the simulated radar observables at the coordinates in
            the GPM file
        '''
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

        az,elev,rang,coords_GPM = get_GPM_angles(GPM_file,band)
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
                    cfg.CONFIG['radar']['coords'] = [params[3],
                                                    params[4],
                                                    params[5]]

                    list_subradials = get_interpolated_radial(dic_vars,
                                                                azimuth,
                                                                elev,N = N)

                    output = get_radar_observables(list_subradials,lut_sz)

                    if output_variables in ['all','only_radar']:
                        output = get_radar_observables(list_subradials,lut_sz)
                    if output_variables == 'only_model':
                        output =  integrate_radials(list_subradials)
                    elif output_variables == 'all':
                        output = combine_subradials((output,
                                integrate_radials(list_subradials)))

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
            list_beams.extend(pool.map(worker_partial,zip(az[i],
                                                     elev[i],
                                                     rang[i],
                                                     c0,
                                                     c1,
                                                     c2)))

        pool.close()
        pool.join()

        del dic_vars
        del N
        del lut_sz
        gc.collect()

        if not event.is_set():
            # Threshold at given sensitivity
            if output_variables in ['all','only_radar']:
                list_beams = cut_at_sensitivity(list_beams)

            list_beams_formatted = SimulatedGPM(list_beams,
                                                dim_swath,
                                                band)
            return list_beams_formatted


if __name__ == '__main__':

    a = RadarOperator(options_file='/ltedata/Daniel/cosmo_pol/cosmo_pol/option_files/CH_PPI_albis_alias.yml')
#    files_c = pc.get_model_filenames('/ltedata/Daniel/cosmo_event_monika/')
    cfile = '/ltedata/Daniel/cfiles_COSMO/lmConstPara_1158x774_0.01_20160106.grib'
    a.load_model_file('/data/cosmo_runs_monika/inlaf201708020015',cfilename = cfile)

    r = a.get_PPI(elevations = 1)
    from cosmo_pol.radar.pyart_wrapper import RadarDisplay
    display = RadarDisplay(r, shift=(0.0, 0.0))
    import matplotlib.pyplot as plt
    plt.figure()
    display.plot('RVEL',0,vmin=-8.3,vmax=8.3,
                 title='aliased RVEL',
                 cmap = plt.cm.RdBu_r,
                 shading = 'flat',
                 max_range = 150000)

    plt.savefig('rvel_aliased.png',dpi=300,bbox_inches='tight')

    cc = a.config
    cc['radar']['nyquist_velocity']=None
    a.config = cc

    r = a.get_PPI(elevations = 1)
    from cosmo_pol.radar.pyart_wrapper import RadarDisplay
    display = RadarDisplay(r, shift=(0.0, 0.0))
    import matplotlib.pyplot as plt
    plt.figure()
    display.plot('RVEL',0,vmin=-30,vmax=30,
                 title='real RVEL',
                 cmap = plt.cm.RdBu_r,
                 shading = 'flat',
                 max_range = 150000)


    plt.savefig('rvel_unaliased.png',dpi=300,bbox_inches='tight')