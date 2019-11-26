# -*- coding: utf-8 -*-

"""lut.py: defines the Lookup Class as well as a set of functions
used to load and save scattering lookup tables"""

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
from  scipy import ndimage
import tempfile
import os
import tarfile
import glob
import shutil
import pickle
from io import BytesIO
from textwrap import dedent

def load_all_lut(scheme, list_hydrom, frequency, scattering_method):
    '''
    Loads all scattering lookup tables for the user specified parameters
    Args:
        scheme: the microphysical scheme, either '1mom' or '2mom'
        list_hydrom: the list of hydrometeors for which scattering properties
            should be obtained: 'R': rain, 'S': snow, 'G': graupel, 'H': hail
            'mS': melting snow, 'mG': melting graupel. Ex: ['R','S','G']
        frequency: the frequency in GHz, make sure the lookup tables have
            been previously computed for the corresponding frequency!
        scattering_method: the scattering method that is used, can be either
            'tmatrix', 'tmatrix_masc' or 'dda', which correspond to subfolders
            in the lookup folder. You could add more...

    Returns:
        lut_sz: dictionary containing the lookup table for every hydrometeor
            type given in 'list_hydrom', the lookup tables are instances of
            the class Lookup_table (see below)
    '''

    # Get current directory
    folder_lut = os.path.dirname(os.path.realpath(__file__))+'/'
    lut_sz = {}

    default_lut = folder_lut +'/lut_tmatrix_masc/'
    if scattering_method == 'tmatrix':
        folder_lut = folder_lut +'/lut_tmatrix/'
    elif scattering_method == 'tmatrix_masc':
        folder_lut = folder_lut +'/lut_tmatrix_masc/'
    elif scattering_method == 'dda':
        folder_lut = folder_lut +'/lut_dda/'

    for h in list_hydrom:
        freq_str = str(frequency).replace('.','_')
        name = 'lut_SZ_' + h + '_' + freq_str + '_' + scheme + '.lut'
        print(folder_lut + name)
        try:
            if scattering_method == 'dda' and h in ['R','H']:
                lut_sz[h] = load_lut(default_lut + name)
            else:
                lut_sz[h] = load_lut(folder_lut + name)
        except:
            raise
            msg = """
            Could not find lookup table for scheme = {:s}, hydrometeor =
            {:s}, frequency = {:f} and scattering method = {:s}
            """.format(scheme, h, frequency, scattering_method)
            raise IOError(dedent(msg))

    return lut_sz

def save_lut(lut, filename):
    '''
    Saves an instance of the Lookup_table to the drive, in the form of multiple
    numpy arrays stored together in a tar file. Use load_lut just below to
    load it to memory
    Args:
        lut: an instance of the Lookup_table class
        filename: the complete filename (with path), indicating where the
            lookup table should be saved
    '''

    tmp = tempfile.gettempdir()

    bname = os.path.basename(filename).split('.')[0]
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

    tar = tarfile.open(filename, "w")
    for name in all_new_files:
        tar.add(name, arcname=os.path.basename(name))

    tar.close()
    # Clear temporary directory
    shutil.rmtree(tmp_dir)


def load_lut(filename):
    '''
    Loads an instance of the Loookup_table class, previously saved to the
    drive with the save_lut function, to memory
    Args:
        filename: the complete filename (with path), indicating where the
            lookup table is stored

    Returns:
        lut: the lookup table as an instance of the class Lookup_table Class
        (see below)
    '''
    tar = tarfile.open(filename, "r")
    
    lut = Lookup_table()
    for member in tar.getmembers():
        array_file = BytesIO()
        array_file.write(tar.extractfile(member).read())
        name = member.name.replace('.npy','')
        array_file.seek(0)
        data = np.load(array_file, allow_pickle = True, encoding = 'latin1')

        if name == 'axes_names':
            data = data.all()
        setattr(lut, name, data)

    tar.close()
    return lut



class Error(Exception):
    """Lookup Table Error"""
    pass

class Lookup_table:
    '''
    The Lookup_table class used to store scattering properties of hydrometeors
    and perform queries of the scattering properties, this class assumes all
    stored data to be defined on regular grids
    '''
    def __init__(self):

        # Contains the independent variable values known along
        # each axis of the data [[x0, x1, ..., xn], [y0, y1, ..., yn], ...]
        self.axes = []

        # Dictionary whose keys are the name of all axes and its values are
        # the index of the axis, ex: {'x': 0, 'y': 1, 'z': 2}, for 3D
        # Cartesian coordinates
        self.axes_names = {}

        # The limits for every axis (min and max values)
        self.axes_limits= []
         # the step for every axis: x1 - x0
        self.axes_step = []

        # to avoid the array shape check error in set_value_table for axis 'd',
        # which has different axes value for different 'wc'
        self.axes_len = []

        # Stores the dependent data for each point of the regular grid defined
        # the axes
        self.value_table = []


    def add_axis(self, name, axis_values=None):
        '''
        Add an axis to the lookup table. Axis correspond to the dimension of
        the data, for example for 3D coordinates, the axis would be 'x', 'y'
        and 'z'
        Args:
            name: name of the axis, for example 'd' for diameters
            axis_values: 'the values corresponding to the specified new axis
                ex for diameters: np.linspace(0.2, 8, 1024)
        '''
        if self.axes_names.has_key(name):
            raise Error("Axis already exists with name: '%s'" % name)
        axis_i = len(self.axes)

        self.axes_names[name] = axis_i
        axis_values = np.asarray(axis_values).astype('float32')

        self.axes_limits.append([np.min(axis_values), np.max(axis_values)])
        self.axes_step.append(axis_values[1]-axis_values[0])
        self.axes_len.append(len(axis_values[0]) if isinstance(axis_values[0], np.ndarray) else len(axis_values))
        self.axes.append(axis_values)

    def set_axis_values(self, axis_name, axis_values):
        '''
        Set the axis values for the specified axis.
        Axis values define points along the axis at which measurements
        were taken.

        Args:
            axis_name: name of the axis, for example 'd' for diameters
            axis_values: 'the values corresponding to the specified new axis
                ex for diameters: np.linspace(0.2, 8, 1024)
        '''

        axis_i = self.axes_names[axis_name]
        axis_values=np.asarray(axis_values).astype('float32')

        self.axes_limits[axis_i]=[np.min(axis_values),np.max(axis_values)]

        self.axes[axis_i] = axis_values


    def set_value_table(self, value_table):
        """Set the value table to the specified axes
        The shape of the data should correspond to the length of all axis
        i.e. value_table.shape = (len(axis[0]), len(axis[1]),...,len(axis[n]))

        Args:
            value_table: the independent data as a list of lists of numpy
            array
        """
        if not isinstance(value_table,np.ndarray):
            value_table = np.array(value_table)

        # Check dimensions
        tuple_axes_len = tuple(self.axes_len)
        if value_table.shape != tuple_axes_len:
            msg = '''
            The shape of the specified data does not match with the length
            of all axes, please ensure that
            data.shape = (len(axis[0]), len(axis[1]),...,len(axis[n]))
            '''
            return ValueError(dedent(msg))
        else:
            self.value_table = value_table


    def get_axis_name(self, axis_i):
        """Returns the name of an axis given its index

        Args:
            axis_i: the index of the axis (zero-based)
        Returns:
            The name of the axis for example 'd' or 'x' or 'y'
        """
        result = None
        for name, i in self.axes_names.items():
            if i == axis_i:
                result = name
                break
        return result


    def lookup_pts(self, coords, order = 1):
        """ Lookup the interpolated value for given axis values

        Args:
            coords: the coordinates where you want to interpolate the data
                must be contained within the bounds for every axis
                (no extrapolation). The coordinates are in the form of
                 a N x M array, where N is the number of points
                you want to interpolate and M is the number of axes
            order: interpolation order, must be in the range 0 to 5
                0 = nearest neighbour, 1 = linear interpolation,...
        Returns:
            out: the interpolated values at the specified coordinates
        """
        # Check that a value table exists.

        if not len(self.value_table):
            raise Error("No values set for lookup table")

        if coords.shape[0] != len(self.axes_names):
            raise Error("Invalid coordinates dimension")

        coords = [(c - lo) * (n - 1) / (hi - lo) for (lo, hi), c, n in zip(self.axes_limits,
                  coords, self.value_table.shape)]

        out = ndimage.map_coordinates(self.value_table, coords, order=order,
                                      mode = 'constant', cval = np.nan )

        if self.int_transform:
            return self.get_float_data(out)
        else:
            return out

    def lookup_line(self,**kwargs):
        """ Lookup a N-D plane from the data, by fixing one axis or more
            Currently only performs nearest neighbour interpolation
        Args:
            The number of arguments is variable. The function is used like
            this lookup_line(name_0 = val_0,name_1 = al_1,...)
            where 'name_0' and 'name_1' are the names of two axes in the
            lookup table, and val_0 and val_1, are the values used for these
            axes. Basically you extract a slice from the value_table by
            setting a certain number of axes to a fixed value
            For example if you have three axes: 'x', 'y', 'z',
            lookup_line(x = 1) will return the plane corresponding to x = 1
            lookup_line(x = 1, y = 2) will return the line corresponding
            to x = 1 and y = 2
            lookup_line(x = 1, y = 2, z = 4) will return a single line
            THE ORDER OF THE ARGUMENTS IS NOT IMPORTANT
        Returns:
            out: the values at the specified hyperplane
        """
        v = self.value_table
        dim = self.value_table.shape

        I = [slice(None)]*v.ndim
        for i,k in enumerate(kwargs.keys()):
            if k in self.axes_names.keys():
                ax_idx = self.axes_names[k]

                closest = np.floor((kwargs[k]-
                                    self.axes_limits[ax_idx][0])
                                    /self.axes_step[ax_idx])
                closest = np.array(closest, dtype= int)
                closest[closest < 0] = 0
                closest[closest >= dim[ax_idx]] = dim[ax_idx] - 1
                I[ax_idx] = closest

        return v[tuple(I)]
