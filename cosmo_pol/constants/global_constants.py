# -*- coding: utf-8 -*-

"""global_constants.py: defines a certain number of global constants, used
throughout the radar operator"""

__author__ = "Daniel Wolfensberger"
__copyright__ = "Copyright 2017, COSMO_POL"
__credits__ = ["Daniel Wolfensberger"]
__license__ = "GPL"
__version__ = "1.0"
__maintainer__ = "Daniel Wolfensberger"
__email__ = "daniel.wolfensberger@epfl.ch"

# Global imports

from cosmo_pol.config import CONFIG
import numpy as np
np.seterr(divide='ignore') # Disable divide by zero error

'''
constants are defined in the form of a class, because they need to be updated
at runtime depending on the specified user configuration

Here is a list

--Independent constants--
EPS: machine epsilon, numbers which differ by less than machine epsilon are
    numerically the same
SIMULATED_VARIABLES:  list of all variables simulated by the radar operator
C: speed of light [m/s]
RHO_W: density of liquid water [kg/mm3 ]
RHO_I: density of ice [kg/mm3 ]
T0 : water freezing temperature [K]
A : Constant used to compute the spectral width due to turbulence
    see Doviak and Zrnic (p.409) [-]
RHO_0: Density of air at the sea level [kg/m3]
KE: 4/3 parameter in the 4/3 refraction model
MAX_MODEL_HEIGHT: The maximum height above which simulated radar gates are
    immediately discarded, i.e. there is not chance the model simulates
    anything so high. This is used when interpolation COSMO variables to the
    GPM beam, because GPM is at 407 km from the Earth, so there is no
    need at all to simulate all radar gates (this would make more than
    3000 gates at Ka band...)
A_AR_LAMBDA_AGG: intercept parameter a in the power-law relation defining the
    value of the Lambda in the gamma distribution for aggregate aspect-ratios
    as a function of diameter
    Lambda(D) = a * D^b
B_AR_LAMBDA_AGG: exponent parameter b in the power-law relation defining the
    value of the Lambda in the gamma distribution for aggregate aspect-ratios
    as a function of diameter
    Lambda(D) = a * D^b
A_AR_LAMBDA_GRAU: intercept parameter a in the power-law relation defining the
    value of the Lambda in the gamma distribution for graupel aspect-ratios
    as a function of diameter
    Lambda(D) = a * D^b
B_AR_LAMBDA_GRAU: exponent parameter b in the power-law relation defining the
    value of the Lambda in the gamma distribution for graupel aspect-ratios
    as a function of diameter
    Lambda(D) = a * D^b
A_CANT_STD_AGG: intercept parameter a in the power-law relation defining the
    value of the standard deviation of aggregates orientations as a function
    of the diameter
    sigma_o(D) = a * D^b
B_CANT_STD_AGG: exponent parameter b in the power-law relation defining the
    value of the standard deviation of aggregates orientations as a function
    of the diameter
    sigma_o(D) = a * D^b
A_CANT_STD_AGG: intercept parameter a in the power-law relation defining the
    value of the standard deviation of graupels orientations as a function
    of the diameter
    sigma_o(D) = a * D^b
B_CANT_STD_AGG: exponent parameter b in the power-law relation defining the
    value of the standard deviation of graupels orientations as a function
    of the diameter
    sigma_o(D) = a * D^b
GPM_SENSITIVITY: maximum sensitivity of GPM measurements [dBZ], everything
    below will be removed
GPM_RADIAL_RES_KA: GPM radial (vertical) resolution at Ka band [m]
GPM_RADIAL_RES_KU: GPM radial (vertical) resolution at Ku band [m]
GPM_NO_BINS_KA: Number of vertical bins at Ka band [-]
GPM_NO_BINS_Ku: Number of vertical bins at Ku band [-]
GPM_KA_FREQUENCY: GPM frequency at Ka band [GHz]
GPM_KU_FREQUENCY: GPM frequency at Ku band [GHz]
GPM_3DB_BEAMWIDTH: 3dB beamwidth of the GPM antenna [degrees]

--Dependent constants--
NVEL: Nyquist velocity of the radar [m/s]
VRES: velocity resolution in the Doppler spectrum (bin width) [m/s]
WAVELENGTH: radar wavelength [mm]
PULSE_WIDTH: width of an emitted radar pulse [m]
RADAR_CONSTANT_DB: The radar contant relating power to reflectivity in dB
RANGE_RADAR: the ranges of all radar gates
'''
class Constant_class(object):
    def __init__(self):

        #,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,
        # Numerical parameters
        self.EPS = np.finfo(np.float).eps
        self.SIMULATED_VARIABLES = ['ZH','DSPECTRUM','RVEL','ZV','PHIDP',
                                    'ZDR','RHOHV','KDP']
        #,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,
        # Physical parameters
        self.C = 299792458.
        self.RHO_W = 1000./(1000**3)
        self.RHO_I = 916./(1000**3)
        self.A = 1.6
        self.RHO_0 = 1.225
        self.KE = 4./3.
        self.MAX_MODEL_HEIGHT = 35000

        #,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,
        # Power laws based on MASC observations
        # Axis-ratios:
        # Aggregates
        self.A_AR_LAMBDA_AGG =  8.42003348664
        self.B_AR_LAMBDA_AGG =  -0.568465084269
        self.A_AR_M_AGG =  0.0527252217284
        self.B_AR_M_AGG = 0.792594862923

        # Graupel
        self.A_AR_LAMBDA_GRAU =  1.97869286543
        self.B_AR_LAMBDA_GRAU =  -0.426770312328
        self.A_AR_M_GRAU =  0.0743715480794
        self.B_AR_M_GRAU =  0.672627814141

        # Canting angles std
        self.A_CANT_STD_AGG = 30.2393875
        self.B_CANT_STD_AGG = -0.077397563
        self.A_CANT_STD_GRAU = 26.65795932
        self.B_CANT_STD_GRAU = -0.10082787

        # GPM constants
        self.GPM_SENSITIVITY = 12
        self.GPM_RADIAL_RES_KA = 250
        self.GPM_RADIAL_RES_KU = 125
        self.GPM_NO_BINS_KA = 88 + 1
        self.GPM_NO_BINS_KU = 176 + 1
        self.GPM_KA_FREQUENCY = 35.6
        self.GPM_KU_FREQUENCY = 13.6
        self.GPM_3DB_BEAMWIDTH = 0.5

        if CONFIG != {}:
            #,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,
            # Radar parameters
            self.NVEL = self.C / (4*1E-6 * CONFIG['radar']['PRI'] *
                                  CONFIG['radar']['frequency']*1E09)
            self.VRES=2*self.NVEL/CONFIG['radar']['FFT_length']
            self.VARRAY=np.arange(-self.NVEL,self.NVEL+self.VRES,self.VRES)
            self.WAVELENGTH=self.C/(CONFIG['radar']['frequency']*1E09)*1000
            self.PULSE_WIDTH = 2*CONFIG['radar']['radial_resolution']
            try:
                if len(CONFIG['radar']['sensitivity']) == 3:
                    K_squared = CONFIG['radar']['K_squared']
                    self.RADAR_CONSTANT_DB = (180 - 10*np.log10(np.pi**3 *
                        CONFIG['radar']['3dB_beamwidth']**2 \
                        * K_squared * self.PULSE_WIDTH/1000)
                        - 2*CONFIG['radar']['sensitivity'][1]\
                        + 10*np.log10(1024 * np.log(2) *
                        (self.WAVELENGTH/1000.)**2))
            except:
                # If it fails we won't need it anyway
                pass
            if CONFIG['radar']['type']=='ground':
                self.RANGE_RADAR=np.arange(
                   CONFIG['radar']['radial_resolution']/2,
                   CONFIG['radar']['range'],
                   CONFIG['radar']['radial_resolution'])



    def update(self):
        self.__init__()

global_constants = Constant_class()