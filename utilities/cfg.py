
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 11 09:57:28 2016

@author: wolfensb
"""
# Setup DEFAULTS
global CONFIG


import numpy as np
import numbers
import yaml
import sys
import copy

from cosmo_pol.utilities.antenna_fit import optimize_gaussians

FLAG_NUMBER = 'ANY_NUMBER'

#class warning_dict(dict):
#    def __init__(self, *args, **kwargs):
#        self.update(*args, **kwargs)
#
#    def __getitem__(self, key):
#        val = dict.__getitem__(self, key)
##        print 'GET', key
#        return val
#
#    def __setitem__(self, key, val):
##        print 'SET', key, val
#        caller = sys._getframe().f_back.f_code.co_name
#        if key == 'frequency' and key in self.keys() \
#            and caller != 'change_frequency':
#            msg = """
#            Since manually changing the frequency requires to reload the lookup tables,
#            as well as recomputing some constants, you need to use the
#            "change_frequency" function of the RadarOperator class to change it.
#            The frequency will NOT be changed now.
#            """
#            print(msg)
#            raw_input("Press Enter to continue...")
#        else:
#            dict.__setitem__(self, key, val)
#    def __repr__(self):
#        dictrepr = dict.__repr__(self)
#        return '%s(%s)' % (type(self).__name__, dictrepr)
#
#    def update(self, *args, **kwargs):
#        for k, v in dict(*args, **kwargs).iteritems():
#            self[k] = v
#    def change_frequency(self,val):
#        # "Legal" way of changing the frequency
#        self['frequency'] = val

CONFIG = {}

DEFAULTS={
    'radar':
        {'coords':[46.42563,6.01,1672.7],\
        'type':'ground',\
        'frequency':5.6,\
        'range':150000,\
        'radial_resolution':500,\
        'PRI':700,\
        'FFT_length':256,\
        'sensitivity':[-5,10000],\
        '3dB_beamwidth':1.},\
    'refraction':
        {'scheme':1},\
    'integration':
        {'scheme':1,\
        'nv_GH':9,\
        'nh_GH':3,\
        'n_gaussians':7,\
        'antenna_diagram':'',\
        'weight_threshold':1.,\
        'nr_GH':7,\
        'na_GL':7},\
    'doppler':
        {'scheme':1,\
        'turbulence_correction':0},\
    'microphysics':
        {'scheme':'1mom',\
         'with_melting': 0,\
         'with_ice_crystals':1,\
         'attenuation': 1,\
         'scattering':'tmatrix_masc'},\
    }

#DEFAULTS['radar'] = warning_dict(DEFAULTS['radar'])

# Setup range of valid values

VALID_VALUES={
    'radar':
        {'coords':FLAG_NUMBER,\
        'type':['ground','GPM'],\
        'frequency':[2,'to',35.6],\
        'range':[5000,'to',500000],\
        'radial_resolution':[25,'to',5000],\
        'PRI':[10,'to',3000],\
        'FFT_length':[16,'to',2048],\
        'sensitivity':FLAG_NUMBER,\
        '3dB_beamwidth':[0.1,'to',10]},\
    'refraction':
        {'scheme':[1,2]},\
    'integration':
        {'scheme':[1,2,3,4,'ml'],\
        'nv_GH':np.arange(1,31,2),\
        'nh_GH':np.arange(1,31,2),\
        'n_gaussians':np.arange(1,13,2),\
        'weight_threshold':[0.,'to',1.],\
        'nr_GH':np.arange(1,31,2),\
        'na_GL':np.arange(1,31,2)},\
    'doppler':
        {'scheme':[1,2,3],\
        'turbulence_correction':[0,1]},\
    'microphysics':
        {'scheme':['1mom','2mom'],\
        'with_ice_crystals':[0,1],\
        'with_melting':[0,1],\
        'attenuation':[0,1],\
        'scattering':['tmatrix_masc','tmatrix','dda']}\
    }

def check_valid(section,key, value):

    flag_ok = False
    out = value
    message = ''

    if section in VALID_VALUES.keys() and key in VALID_VALUES[section].keys():
        valid_vals=VALID_VALUES[section][key]
        if isinstance(valid_vals, list):
            if 'to' in valid_vals:
                if value>=float(valid_vals[0]) and value<=float(valid_vals[2]):
                    flag_ok = True
                    message = ''
                else:
                    flag_ok = False
                    message = 'Invalid value for the '+section+': '+key+' parameter'+\
                    'Please choose one of the following values: '+\
                    'from '+str(valid_vals[0])+' to '+str(valid_vals[2])+\
                    'Using default option: '+section+': '+key+' = '+str(DEFAULTS[section][key]+'\n')
                    out = DEFAULTS[section][key]
            elif value in valid_vals:
                flag_ok = True
                message = ''
            else:
                flag_ok = False
                message = 'Invalid value for the "'+section+': '+key+'" parameter \n'+\
                    'Please choose one of the following values: '+\
                    '['+', '.join([str(s) for s in valid_vals])+'] \n'\
                    'Using default option: "'+section+': '+key+'" = '+str(DEFAULTS[section][key])+'\n'
        elif valid_vals == FLAG_NUMBER:
            if not hasattr(value, '__len__'):
                value = [value]
            for v in value:
                if not isinstance(v,numbers.Number):
                    flag_ok = False
                    message = 'Invalid value for the "'+section+': '+key+'" parameter \n'+\
                    'All values must be numbers \n' \
                    'Using default option: "'+section+': '+key+'" = '+str(DEFAULTS[section][key])+'\n'
                    out = DEFAULTS[section][key]
    else:
        flag_ok = True
        message='Parameter "'+section+': '+key+'" was not tested for valid range.\n'+\
        'It is your responsability to give relevant values \n'
    return flag_ok, out, message


def init(options_file):
    global CONFIG
    CONFIG = None
    try:
        with open(options_file, 'r') as ymlfile:
            CONFIG = yaml.load(ymlfile)
    except:
        CONFIG = copy.deepcopy(DEFAULTS)
        print('Could not find or read '+options_file+' file, using default options')
        return
    CONFIG = sanity_check(CONFIG)

def sanity_check(config):
    # Parsing values
    for section in DEFAULTS:
        if section not in config.keys():
            config[section] = {}
        for key in DEFAULTS[section]:
            if key not in config[section].keys():
                config[section][key] = DEFAULTS[section][key]

            flag_ok, value, message = check_valid(section,key, config[section][key])

            if message != '':
                print(message)

            config[section][key] = value

        if section == 'integration': # Treat special case
            if config[section]['antenna_diagram'] != '' :

                try:
                    print('Trying to fit sum of gaussians on the provided antenna diagram...\n')
                    data = np.genfromtxt(CONFIG[section]['antenna_diagram'], delimiter=',')
                    x = data[:,0]
                    y = 10*np.log10((10**(0.1*data[:,1]))**2)
                    gauss_params = optimize_gaussians(x,y,config[section]['n_gaussians'])
                    config[section]['antenna_params'] = gauss_params
                    print('Fit was successful !\n')
                except:
                    raise
                    print('Could not fit a sum of gaussians with the file provided in "\n'+\
                    '"'+section+':'+' antenna_diagram'+'"\n'+\
                    'Please proviide a comma-separated file with two columns'+\
                    ', one for the angles, one for the power in dB \n')

#    config['radar'] = warning_dict(config['radar'])

    return config

if __name__ == '__main__':
    init('/media/wolfensb/Storage/cosmo_pol/option_files/MXPOL_PPI.yml')
    print(CONFIG['radar']['frequency'])
    CONFIG['radar'].change_frequency(5)
    print(CONFIG['radar']['frequency'])
    CONFIG = None
    init('/media/wolfensb/Storage/cosmo_pol/option_files/MXPOL_PPI.yml')
    print(CONFIG['radar']['frequency'])

    #print c
    #import yaml
    #
    #with open("test_yaml.yaml", 'r') as ymlfile:
    #    CONFIG = yaml.load(ymlfile)
    #
    #for section in CONFIG:
    #    print(section)
    #print(CONFIG['radar'])
    #print(CONFIG['doppler'])

