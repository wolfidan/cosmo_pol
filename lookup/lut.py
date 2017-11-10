
# lookup_table - multidimensional lookup table class
# Copyright (C) 2007 RADLogic
#
# This library is free software; you can redistribute it and/or
# modify it under the terms of the GNU Lesser General Public
# License as published by the Free Software Foundation;
# version 2.1 of the License.
#
# This library is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# Lesser General Public License for more details.
#
# See http://www.fsf.org/licensing/licenses/lgpl.txt for full license text.
"""Define a multidimensional lookup table class.

This uses (piecewise) linear interpolation for table lookups.

- axes are defined in order, and can be added in stages
- axes are either purely increasing or purely decreasing
- this module is units-agnostic

This is intended for use with lookup tables compiled from Liberty files,
so must meet the needs for that application.

It might be useful for other applications as well.

This is designed to be compatible with Python 2.2.

The unittests (doctest) can be run by running this script directly with Python:
python lookup_table.py

Make sure the tests pass before checking in changes to this module.

This module is probably a good candidate for releasing as open source.
(concise, well-defined scope, doctests, finished, would benefit from many eyes)

"""

__author__ = 'Tim Wegener <twegener@radlogic.com.au>'
__version__ = '$Revision: 0.7 $'
__date__ = '$Date: 2008/04/18 08:01:00 $'



import numpy as np
from  scipy import ndimage
from cosmo_pol.utilities.tictoc import *
import tempfile
import os
import tarfile
import glob
import StringIO
import shutil
import pickle

# Get current directory
FOLDER_LUT=os.path.dirname(os.path.realpath(__file__))
def load_all_lut(scheme, list_hydrom, freq, scattering_method):
    lut_sz = {}

    default_lut = FOLDER_LUT +'/stored_lut_quad/'
    if scattering_method == 'tmatrix_old':
        folder_lut = FOLDER_LUT +'/stored_lut/'
    elif scattering_method == 'tmatrix_new':
        folder_lut = FOLDER_LUT +'/stored_lut_quad/'
    elif scattering_method == 'dda':
        folder_lut = FOLDER_LUT +'/stored_lut_dda/'

    for h in list_hydrom:
        freq_str = str(freq).replace('.','_')
        name = 'lut_SZ_'+h+'_'+freq_str+'_'+scheme+'.lut'
        try:
            if scattering_method == 'dda' and h in ['R','H']:
                lut_sz[h] = load_lut(default_lut + name)
            else:
                lut_sz[h] = load_lut(folder_lut + name)
        except:
            msg = """
    Could not find lookup table for scheme = {:s}, hydrometeor =
    {:s}, frequency = {:f} and scattering method = {:s}
        """.format(scheme,h,freq,scattering_method)
            print(msg)

    return lut_sz



# Saves a lookup table in numpy format, this is the fastest read/write method
def save_lut(lut, name):
    tmp = tempfile.gettempdir()

    bname = os.path.basename(name).split('.')[0]
    tmp_dir = tmp + '/' + bname + '/'

    if not os.path.exists(tmp_dir):
        os.makedirs(tmp_dir)

    # Save value table
    np.save(tmp_dir + 'value_table',lut.value_table)
    # Save axes
    np.save(tmp_dir + 'axes',lut.axes)
    # Save axes names
    np.save(tmp_dir + 'axes_names',lut.axes_names)

    # Save axes steps
    try:
        np.save(tmp_dir + 'axes_step',lut.axes_step)
    except:
        pickle.dump(lut.axes_step,open(tmp_dir + 'axes_step.npy','wb'))

    # Save axes limits
    try:
        np.save(tmp_dir + 'axes_limits',lut.axes_limits)
    except:
        pickle.dump(lut.axes_limits,open(tmp_dir + 'axes_limits.npy','wb'))

    all_new_files = glob.glob(tmp_dir + '/*.npy')

    tar = tarfile.open(name, "w")
    for name in all_new_files:
        tar.add(name, arcname=os.path.basename(name))

    tar.close()
    # Clear temporary directory
    shutil.rmtree(tmp_dir)

def load_lut(name):
    tar = tarfile.open(name, "r")
    lut = Lookup_table()
    for member in tar.getmembers():
        name = member.name.replace('.npy','')
        f = tar.extractfile(member)

        data = np.load(f)

        if name == 'axes_names':
            data = data.all()
        setattr(lut, name, data)


    tar.close()
    return lut

def float_to_uint16(x, use_log=False):
    eps = np.finfo(float).eps
    log_offset = 0
    if use_log:
        log_offset=eps+np.nanmin(x)
        x=np.log(x+log_offset)

    x[x==np.Inf] = float('nan')

    int_min = -32768
    int_max = 32767
    float_min = np.nanmin(x)
    float_max = np.nanmax(x)

    x_d=np.zeros(x.shape)
    x_d[~np.isnan(x)]=(int_min+1)+(x[~np.isnan(x)]-float_min)/(float_max-float_min)*(int_max-(int_min+1))
    x_d[np.isnan(x)]=int_min
    x_d = x_d.astype('int16')

    out={'data':x_d,'use_log':use_log,'log_offset':log_offset,'range':[float_min,float_max]}

    return out

class Error(Exception):
    """Lookup Table Error"""
    pass


class Lookup_table:
    def __init__(self):

        # axes_names - map name->index
        self.axes_names = {}

        # axes_is_regular : checks if the axis has constant spacing

        self.axes_limits= []
        self.axes_step = []

        # axes - define independent variable values known along each axis
        #    [[x0, x1, ..., xn], [y0, y1, ..., yn], ...]
        self.axes = []

        # value_table - store dependent variable values for each point in the
        #               table
        #    [[[val_x0_y0_...z0, val_x0_y0..._z1, ...], [], ...], ...]
        self.value_table = []


    def add_axis(self, name, axis_values=None, is_regular=False):
        """Add an axis definition."""

        if self.axes_names.has_key(name):
            raise Error("Axis already exists with name: '%s'" % name)
        axis_i = len(self.axes)

        self.axes_names[name] = axis_i
        axis_values=np.asarray(axis_values).astype('float32')

        self.axes_limits.append([np.min(axis_values),np.max(axis_values)])
        self.axes_step.append(axis_values[1]-axis_values[0])
        self.axes.append(axis_values)

    def set_axis_values(self, axis_name, axis_values, is_regular=False):
        """Set the axis values for the specified axis.

        Axis values define points along the axis at which measurements
        were taken.

        This will raise an error if the value table already exists.

        # todo: Add doctests.

        """
        # todo: Is raising an error here necessary?
        if self.value_table is not None:
            raise Error("Cannot define axis once value table has been set.")
        axis_i = self.axes_names[axis_name]
##         if len(axis_values) != len(self.axes[axis_i]):
##             print 'warning: number of axis values changed'
        axis_values=np.asarray(axis_values).astype('float32')

        self.axes_limits[axis_i]=[np.min(axis_values),np.max(axis_values)]

        self.axes[axis_i] = axis_values

    def set_value_table(self, value_table, map_to_int=False):
        """Set the value table to the specified sequence of sequences.

        Nesting should correspond to value_table[axis0_i][axis1_i]...[axisn_i]

        """
        if isinstance(value_table,np.ndarray):
            self.value_table=value_table
        else:
            self.value_table=np.array(value_table)

    def get_axis_name(self, axis_i):
        """Return the name of the specified axis. (Index starts at 0)"""

        result = None
        for name, i in self.axes_names.items():
            if i == axis_i:
                result = name
                break
        return result


    def lookup_pts(self, coords, mode='nearest', order=1):
        """Lookup the interpolated value for given axis values.

        Arguments:
        Specify the axis values for the lookup, using the axis names as
        keyword arguments.

        """
        # Check that a value table exists.

        if not len(self.value_table):
            raise Error("No values set for lookup table")

        if coords.shape[0] != len(self.axes_names):
            raise Error("Invalid coordinates dimension")

        coords = [(c - lo) * (n - 1) / (hi - lo) for (lo, hi), c, n in zip(self.axes_limits,
                  coords, self.value_table.shape)]

        out = ndimage.map_coordinates(self.value_table, coords,mode=mode,order=order)

        if self.int_transform:
            return self.get_float_data(out)
        else:
            return out
#
#    def lookup_line(self,**kwargs):
#        ''' Extracts a slice of the lookup table by setting some dimensions to
#        a fixed value
#
#        Examples:
#        1) lut.lookup_line(t=298,e=10) : extracts all values for all
#        diameters, for a given temperature and elevation
#
#        2) lut.lookup_line(e=10) extracts all values for all diameters
#        all temperatures for a fixed elevation of 10 deg '''
#
#        v=self.value_table
#        num_axes=len(self.axes)
#        slc = [slice(None)] * num_axes
#        dim = self.value_table.shape
#        for i,k in enumerate(kwargs.keys()):
#            ax_idx=self.axes_names[k]
#            closest = round((kwargs[k]-self.axes_limits[ax_idx][0])/self.axes_step[ax_idx])
#            closest = min(max(0,closest),dim[i]-1)
#            slc[ax_idx]=slice(int(closest),int(closest)+1)
#        return np.squeeze(v[slc])

    def lookup_line(self,**kwargs):
        v = self.value_table
        dim = self.value_table.shape

        # Boolean indexer
#        boolean = np.zeros(v.shape).astype(bool)
        I = [slice(None)]*v.ndim
        for i,k in enumerate(kwargs.keys()):
            if k in self.axes_names.keys():
                ax_idx = self.axes_names[k]

                closest = np.floor((kwargs[k]-
                                    self.axes_limits[ax_idx][0])
                                    /self.axes_step[ax_idx])
                closest = closest.astype(int)
                closest[closest < 0] = 0
                closest[closest >= dim[ax_idx]] = dim[ax_idx] - 1
                I[ax_idx] = closest
#                print(boolean[tuple(I)].shape)
#                boolean[tuple(I)] = Tru

        return v[tuple(I)]
#
if __name__ == '__main__':
    a = load_all_lut('1mom',['mS'],freq=5.6,scattering_method = 'tmatrix_new')
    lmg = a['mS'].lookup_line(e = np.array([0,1]), wc = np.array([0.1,0.99]))
    import matplotlib.pyplot as plt
    plt.plot(lmg[0,:,0])
    plt.plot(lmg[1,:,0])
    plt.legend(['0.1','0.99'])
#    import pickle
#
#    list_elevations=range(0,91,)
#    list_temperatures=range(200,316,2)
#    list_D=np.linspace(0.2,10,1024).astype('float32')
#    list_scatt=np.arange(0,12)
#    lut_rcs = Lookup_table()
#    lut_rcs.add_axis('e',list_elevations)
#    lut_rcs.add_axis('t', list_temperatures)
#    lut_rcs.add_axis('D',[1,2])
#    lut_rcs.add_axis('Scatt',list_scatt)
#
#    v=np.random.rand(len(list_elevations),len(list_temperatures),len(list_D),len(list_scatt))
#    lut_rcs.set_value_table(v.astype('float32'),map_to_int=False)



#    import glob
#    import pickle
#    import gzip
#    files = glob.glob('/media/wolfensb/Storage/cosmo_pol/lookup/sz_lut_64/*')
#    for f in files:
#        print(f)
#        lut_rcs = pickle.load(gzip.open(f))
#        lut_rcs.__class__ = Lookup_table
#        lut_rcs.value_table = lut_rcs.value_table.astype('float32')
#        pickle.dump(lut_rcs,gzip.open(f.replace('sz_lut_64','sz_lut'),'wb'))

#    FREQUENCIES=[2.7,4.15,5.6,7.7,9.8,11.7,13.6,24.6,35.6]
#    for fr in FREQUENCIES:
#        files = glob.glob('/media/wolfensb/Storage/cosmo_pol/lookup/sz_lut/*'+str(fr).replace('.','_')+'*2mom.pz')
#        for f in files:
#            bname = os.path.basename(f)
#


#    lut_rcs = pickle.load(gzip.open('/media/wolfensb/Storage/cosmo_pol/lookup/sz_lut/lut_SZ_R_7_7_1mom.pz','rb'))
#    N = np.random.rand(500,1024).astype('float32')
#
#    tic()
#    for i in range(40):
#        for j in range(3):
#            S = lut_rcs.lookup_line(e=np.ones(500,)*10.5,t=np.ones(500,)*298.9)
#            bb = np.einsum('ijk,ij->ik',S,N)
#    toc()


#    coords=[]
#    for i in range(500):
#        coords.append([list_elevations[15],list_temperatures[10],0])
#    coords=np.asarray(coords).T
#    w=lut_rcs.lookup_pts(coords)
#
#    tic()
#    for i in range(50*4):
#        w=lut_rcs.lookup_pts(coords)
#    toc()
##
#    print v[15,10,150,3]
#
#
#    lut_rcs.set_value_table(v,map_to_int=True)
#
#    #
#    tic()
#    for u in range(360):
#        for i in range(45):
#            w=lut_rcs.lookup_pts(coords)
#    print w[0]
#    toc()
