# -*- coding: utf-8 -*-

"""pyart_wrapper.py: Provides routines to convert radar operator outputs
to PyART classes"""

__author__ = "Daniel Wolfensberger"
__copyright__ = "Copyright 2017, COSMO_POL"
__credits__ = ["Daniel Wolfensberger"]
__license__ = "GPL"
__version__ = "1.0"
__maintainer__ = "Daniel Wolfensberger"
__email__ = "daniel.wolfensberger@epfl.ch"

# Global imports
from pyart import graph, filters, config, core
import numpy as np
np.seterr(divide='ignore') # Disable divide by zero error

# Local imports
from cosmo_pol.constants import global_constants as constants
from cosmo_pol.utilities import row_stack

# Define the units, labels and vmin and vmax values for the most commonly
# simulated variables

UNITS_SIMUL  = {'ZH':'dBZ',
               'KDP':'deg/km',
               'PHIDP':'deg',
               'RHOHV':'-',
               'ZDR':'dB',
               'RVEL':'m/s',
               'DSPECTRUM':'dBZ',
               'ZV':'dBZ',
               'U':'m/s',
               'V':'m/s',
               'W':'m/s',
               'T':'K',
               'RHO':'kg/m3',
               'QR_v':'kg/m3',
               'QS_v':'kg/m3',
               'QG_v':'kg/m3',
               'QH_v':'kg/m3',
               'ZV':'dBZ',
               'ATT_H':'dBZ',
               'ATT_V':'dBZ'}

VAR_LABELS_SIMUL = {'ZH':'Reflectivity',
                    'KDP':'Specific diff. phase',
                    'RHOHV':'Copolar corr. coeff.',
                    'ZDR':'Diff. reflectivity',
                    'RVEL':'Mean doppler velocity',
                    'DSPECTRUM':'Doppler spectrum',
                    'ZV':'Vert. reflectivity',
                    'U':'U-wind component',
                    'V':'V-wind component',
                    'W':'Vertical wind component',
                    'T':'Temperature',
                    'RHO':'Air density',
                    'QR_v':'Mass density of rain',
                    'QS_v':'Mass density of snow',
                    'QG_v':'Mass density of graupel',
                    'QH_v':'Mass density of hail',
                    'PHIDP':'Diff. phase shift',
                    'ATT_H':'Attenuation at hor. pol.',
                    'ATT_V':'Attenuation at vert. pol.'}

VMIN_SIMUL = {'ZH': 0.,
              'KDP': 0.,
              'PHIDP': 0.,
              'RHOHV': 0.6,
              'ZDR': 0.,
              'RVEL': -25,
              'DSPECTRUM': -50,
              'ZV': 0.,
              'U': -30,
              'V': -30,
              'W': -10,
              'T': 200,
              'RHO': 0.5,
              'QR_v': 0.,
              'QS_v': 0.,
              'QG_v': 0.,
              'QH_v': 0.,
              'ATT_H': 0,
              'ATT_V': 0}

VMAX_SIMUL = {'ZH': 55,
              'KDP': 1,
              'PHIDP': 20,
              'RHOHV': 1,
              'ZDR': 2,
              'RVEL': 25,
              'DSPECTRUM': 30,
              'ZV': 45,
              'U': 30,
              'V': 30,
              'W': 10,
              'T': 300,
              'RHO': 1.4,
              'QR_v': 1E-3,
              'QS_v': 1E-3,
              'QG_v': 1E-3,
              'QH_v': 1E-2,
              'ATT_H': 5,
              'ATT_V': 5}


class RadarDisplay(graph.radardisplay.RadarDisplay):
    '''
    This is a class for radar operator outputs that inheritates the
    pyart RadarDisplay Class and thus has the same functionalities
    It adds some functionalities that described below
    '''
    def __init__(self, radar, shift = (0.0, 0.0)):

        self.scan_type = None
        self._radar = None
        super(RadarDisplay,self).__init__(radar, shift=(0.0, 0.0))

    def plot(self, fields, sweep = 0, max_range = 100000, **kwargs):
        '''
        Plots the data to a Cartesian grid,
        IT is the same as the original pyART function
        see http://arm-doe.github.io/pyart/, except that it adds an option
        to mask all values further away than a specified max_range
        '''

        filt = filters.GateFilter(self._radar)
        filt.exclude_above('range',max_range)


        super(RadarDisplay,self).plot(field = fields,
             sweep = sweep, gatefilter = filt, **kwargs)

    def plot_vprof(self, fields, sweep=0, mask_tuple=None,
            vmin = None, vmax=None, norm=None, cmap=None, mask_outside=False,
            title=None, title_flag=True,
            axislabels=(None, None), axislabels_flag=True,
            colorbar_flag=True, colorbar_label=None,
            colorbar_orient='vertical', edges=True, gatefilter=None,
            filter_transitions=True, ax=None, fig=None, **kwargs):
        '''
        Plots the data of a vertical profile, with possibly a full
        Doppler spectrum to a Cartesian grid
        IT basically refined the original plot pyART function
        see http://arm-doe.github.io/pyart/
        '''
        if np.isscalar(fields):
            fields=[fields]

        ranges = self._radar.range['data']

        ax, fig = graph.common.parse_ax_fig(ax, fig)
        ax.hold(True)
        for field in fields:
            data = self._radar.get_field(sweep, field)
            if field == 'DSpectrum': # 1D data
                varray = constants.VARRAY
                pm = ax.pcolormesh(varray, ranges, data, vmin=vmin,
                                   vmax=vmax, cmap=cmap, norm=norm, **kwargs)
                self.set_limits(ylim=[0,10000],xlim=[np.min(varray),2])
                ax.set_xlabel('Velocity [m/s]')
                self.plot_colorbar(
                mappable=pm, label=colorbar_label, orient=colorbar_orient,
                field=field, ax=ax, fig=fig)
            else:
                if norm is None:
                    vmin, vmax = graph.common.parse_vmin_vmax(self._radar,
                                                              field, vmin, vmax)
                else:
                    vmin = None
                    vmax = None
                if cmap is None:
                    cmap = config.get_field_colormap(field)
                    ax.plot(data,ranges,linewidth=2)
                self.set_limits(ylim=[0,10000])
                ax.set_xlabel(VAR_LABELS_SIMUL[field]+' ('+UNITS_SIMUL[field]+')')
        ax.grid()
        ax.set_ylabel('Height above radar [m]')

class PyartRadop(core.Radar):
    """
    This is a class for radar operator outputs that inheritates the
    pyart core Class and thus has the same functionalities
    """
    def __init__(self, scan_type, scan):
        '''
        Creates a PyartRadop Class instance
        Args:
            scan_type: the type of scan can be either 'PPI' or 'RHI', for
                vertical scans it doesn't matter, they can be PPI or RHI
            scan: the output of the get_PPI or get_RHI functions of the main
                RadarOperator class

        Returns:
            a PyartRadop instance which can be used as specified in the
            pyART doc: http://arm-doe.github.io/pyart/

        '''

        N_sweeps=len(scan['data'])

        fields={}
        fixed_angle={}
        fixed_angle['data']=np.zeros(N_sweeps,)

        sweep_start_ray_index={}
        sweep_start_ray_index['data']=[]
        sweep_stop_ray_index={}
        sweep_stop_ray_index['data']=[]

        # Get all variables names
        varnames=scan['data'][0][0].values.keys()

        for i,k in enumerate(varnames):

            fields[k]={}
            fields[k]['data']=[]
            try: # No info found in header
                fields[k]['long_name'] = VAR_LABELS_SIMUL[k]
            except:
                pass
            try:
                fields[k]['units'] = UNITS_SIMUL[k]
            except:
                pass
            try:
                fields[k]['valid_min'] = VMIN_SIMUL[k]
            except:
                pass
            try:
                fields[k]['valid_max'] = VMAX_SIMUL[k]
            except:
                pass
        # Add latitude and longitude
        fields['Latitude'] = {'data':[],'units':['degrees']}
        fields['Longitude'] = {'data':[],'units':['degrees']}

        # Initialize
        idx_start=0
        idx_stop=0
        elevations=[]
        azimuths=[]

        for i in range(N_sweeps):
            # Convert list of beams to array of data in polar coordinates
            polar_data_sweep={}
            for k in varnames:
                polar_data_sweep[k] = np.array([it.values[k] for
                                               it in scan['data'][i]])
                if k in ['ZDR','ZV','ZH','DSpectrum']: # Convert to dB
                    polar_data_sweep[k][polar_data_sweep[k]==0]=float('nan')
                    polar_data_sweep[k]=10*np.log10(polar_data_sweep[k])

            # Add also latitude and longitude to variables
            polar_data_sweep['Latitude'] = \
                np.array([it.lats_profile for it in scan['data'][i]])
            polar_data_sweep['Longitude'] = \
                np.array([it.lons_profile for it in scan['data'][i]])


            [N_angles,N_ranges] = polar_data_sweep[varnames[0]].shape

            idx_stop=idx_start+N_angles-1
            sweep_start_ray_index['data'].append(idx_start)
            sweep_stop_ray_index['data'].append(idx_stop)
            idx_start = idx_stop + 1

            if scan_type == 'PPI':
                fixed_angle['data'] = scan['elevations'][i]
                elevations.extend(list([scan['elevations'][i]] * N_angles))
                azimuths.extend(list(scan['azimuths']))
            elif scan_type == 'RHI':
                fixed_angle['data']=scan['azimuths'][i]
                elevations.extend(list(scan['elevations']))
                azimuths.extend(list([scan['azimuths'][i]] * N_angles))

            for k in polar_data_sweep.keys():
                if not len(fields[k]['data']):
                    fields[k]['data'] = polar_data_sweep[k]
                else:
                    fields[k]['data'] = row_stack(fields[k]['data'],
                                                    polar_data_sweep[k])

        for k in polar_data_sweep.keys():
            fields[k]['data'] = np.ma.array(fields[k]['data'],
                                mask=np.isnan(fields[k]['data']))
        # Add velocities (for Doppler spectrum) in instrument param
        try:
            instrument_parameters={'varray': {'data':constants.VARRAY}}
        except:
            pass

        '''
        Position and time are obtained from the pos_time field of the
        scan dictionary. Note that these latitude and longitude are the
        coordinates of the radar, whereas the latitude and longitude fields
        are the coords of every gate
        '''

        latitude={'data' : np.array(scan['pos_time']['latitude'])}
        longitude={'data' :  np.array(scan['pos_time']['longitude'])}
        altitude={'data' :  np.array(scan['pos_time']['altitude'])}

        time_units='seconds since '+scan['pos_time']['time']
        time={'data' : np.zeros((N_angles,)),'units': time_units}


        sweep_number={'data' : np.arange(0,N_sweeps)}
        sweep_mode={'data' : [scan_type]*N_sweeps}

        azimuth={'data' : np.array(azimuths)}
        rrange={'data': scan['ranges']}
        elevation={'data' :np.array(elevations)}

        '''
        Finally add ranges as an additional variable, for convenience in order
        to filter gates based on their range
        '''

        fields['range'] = {}
        fields['range']['data'] = np.tile(rrange['data'],(len(elevation['data']),1))

        metadata = {}

        # Create PyART instance
        super(PyartRadop,self).__init__(time, rrange, fields, metadata,
        scan_type, latitude, longitude, altitude, sweep_number, sweep_mode,
        fixed_angle, sweep_start_ray_index, sweep_stop_ray_index, azimuth,
        elevation, instrument_parameters = instrument_parameters)

    def get_field(self,sweep_idx, variable):
        # see the get_field function in the PyART doc, this just flattens
        # the output (squeeze)
        out = super(PyartRadop,self).get_field(sweep_idx, variable)
        return np.squeeze(out)
