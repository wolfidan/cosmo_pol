# -*- coding: utf-8 -*-

"""utilities.py: provides a set of convenient functions that are used
throughout the radar operator"""

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
from textwrap import dedent


BASIC_TYPES = [float, int, str]
class Range(object):
    def __init__(self,x0,x1):
        """
        The Range class extends the range native python class to floats
        It can be used as a convenient way to test if a given values fall
        within a certain range: ex 2.3 in Range(1.2,5.3) returns True, but
        1.9 in Range(2.0, 10.0) return False.
        Args:
            x0: lower bound of the range
            x1: lower bound of the range
        Returns:
            A Range class instance
        """

        if type(x0) != type(x1):
            raise ValueError('range bounds are not of the same type!')
        if x1 <= x0:
            raise ValueError('Lowe bound is larger than upper bound!')
        self.x0 = x0
        self.x1 = x1
        self.type = type(x0)
    def __contains__(self,z):
        return type(z) == self.type and z <= self.x1 and z >= self.x0
    def __str__(self):
        return 'Range of values from {:f} to {:f}'.format(self.x0,self.x1)

def generic_type(value):
    """
        Gets the type of any input, if the type is a numpy type (ex. np.float),
        the equivalent native python type is returned

        Args:
            value: the value whose type should be returned
        Returns:
            The type of the value
    """
    type_ = type(value)
    if type_ == np.int64:
        type_ = int
    elif type_ == np.float or type_ == np.float64 or type_ == np.float128:
        type_ = float
    elif type_ == np.str_:
        type_ = str
    return type_

# The TypeList class is used to check if a given array or list is of appropriate
# dimension and type

class TypeList(object):
    def __init__(self, types, dim = []):
        """
        Checks if a given array or list has the right type(s) and the right
            dimensions

        Args:
            types : a single python type or a list of types, to be checked for
            dim: a tuple e.g. (3,2), which specifies which shape the checked
                array must have (Optional). When not specified, the checked
                array can have any arbitrary dimensions
        Returns:
            A TypeList class instance, which can be used to check an array
                ex. np.array([1.,2.,'a']) == TypeList([float,str],[3,])
                will return True, but np.array(['a','b']) = TypeList([float])
                will return False
        """

        if type(types) != list:
            types = [types]
        # Note that dim = [], is used to indicate an arbitrary length
        if list(set(types)-set(BASIC_TYPES)):
            msg = ('''
            One of the specified types is invalid! Must be int, float, str'
            ''')
            raise ValueError(dedent(msg))
        if any([d<0 for d in dim]):
            raise(ValueError('Specified dimension is invalid (<0)!'))

        self.types = types
        self.dim = dim
    def __eq__(self, array):
        flag = False
        try:
            array = np.array(array)
            dim = array.shape
            # Check if dimensions are ok
            if len(self.dim): # Check only if dim != []
                flag = all([d1 == d2 for d1,d2 in zip(dim,self.dim)])
            else:
                flag = True
            # Check if all types are ok
            flag *= all([generic_type(v) in self.types for v in array.ravel()])
        except:
            pass
        return flag

    def __str__(self):
        if self.dim != []:
            msg = 'Array of {:s}, with dimensions {:s}'.format(self.types,self.dim)
        else:
            msg = 'Array of {:s}, with arbitrary dimensions'.format(self.type_)

        return dedent(msg)

def get_earth_radius(latitude):
    '''
    Computes the radius of the earth at a specified latitude
    Args:
        latitude: latitude in degrees

    Returns:
        The earth radius in meters
    '''
    a = 6378.1370*1000 # Minimal earth radius (pole) in m
    b = 6356.7523*1000 # Maximal earth radius (equator) in m
    num = ((a** 2 * np.cos(latitude)) ** 2 + (b ** 2 * np.sin(latitude)) ** 2)
    den = ((a * np.cos(latitude)) ** 2+(b * np.sin(latitude)) ** 2)
    return np.sqrt(num / den)


def vlinspace(a, b, N, endpoint=True):
    """
        Vectorized equivalent of numpy's linspace

        Args:
            a: list of starting points
            b: list of ending points
            N: number of linearly spaced values to compute between a and b
            endpoint: boolean (optional), if True, the endpoint will be included
                in the resulting series
        Returns:
            A matrix, where every column i is a linearly spaced vector between
                a[i] and b[i]
    """
    a, b = np.asanyarray(a), np.asanyarray(b)
    return a[..., None] + (b-a)[..., None]/(N-endpoint) * np.arange(N)

def nan_cumprod(x):
    """
    An equivalent of np.cumprod, where NaN are ignored

    Args:
        x: a 1D array
    Returns:
        The 1D array of cumulated products,
            ex: [1,2,NaN,5] will return [1,2,2,10]
    """
    x[np.isnan(x)]=1
    return np.cumprod(x)

def nan_cumsum(x):
    """
    An equivalent of np.cumsum, where NaN are ignored

    Args:
        x: a 1D array
    Returns:
        The 1D array of cumulated sums,
            ex: [1,2,NaN,5] will return [1,3,3,8]
    """
    x[np.isnan(x)]=0
    return np.cumsum(x)

def sum_arr(x,y, cst = 0):
    """
    Sums up two arrays with possibly different shapes, by padding them before
    with zeros so they have the same dimensions

    Args:
        x: first array
        y: second array
    Returns:
        The summed up array
    """
    diff = np.array(x.shape) - np.array(y.shape)
    pad_1 = []
    pad_2 = []
    for d in diff:
        if d < 0:
            pad_1.append((0,-d))
            pad_2.append((0,0))
        else:
            pad_2.append((0,d))
            pad_1.append((0,0))


    x = np.pad(x,pad_1,'constant',constant_values=cst)
    y = np.pad(y,pad_2,'constant',constant_values=cst)

    z = np.sum([x,y],axis=0)

    return z

def nansum_arr(x,y, cst = 0):
    """
    Sums up two arrays with possibly different shapes, by padding them before
    with zeros so they have the same dimensions. Ignores NaN values (they
    are treated as zeros)

    Args:
        x: first array
        y: second array
    Returns:
        The summed up array
    """
    x = np.array(x)
    y = np.array(y)

    diff = np.array(x.shape) - np.array(y.shape)
    pad_1 = []
    pad_2 = []
    for d in diff:
        if d < 0:
            pad_1.append((0,-d))
            pad_2.append((0,0))
        else:
            pad_2.append((0,d))
            pad_1.append((0,0))

    x = np.pad(x,pad_1,'constant',constant_values=cst)
    y = np.pad(y,pad_2,'constant',constant_values=cst)

    z = np.nansum([x,y],axis=0)
    return z

def combine_subradials(list_of_subradials):
    """
    Combines two subradials by combining their variables. The subradials
    should correspond to the same radar profile (theta and phi angles)

    Args:
        list_of_subradials: the list of subradials
    Returns:
        The combined subradial
    """
    x = list_of_subradials[0]
    for l in list_of_subradials:
        if (l.dist_profile == x.dist_profile).all():
            x.values.update(l.values)
        else:
            msg = '''
            Beams are not defined on the same set of coordinates, aborting
            '''
            print(dedent(msg))
            return
    return x


def row_stack(a1, a2):
    """
    Vertically stacks two rows with possibly different shapes,
    padding them with nan before, so they have the same shape

    Args:
        a1: the first row
        a2: the second row
    Returns:
        The array corresponding to the stacked rows
    """
    [N1, M1] = a1.shape
    [N2, M2] = a2.shape

    if N1 > N2:
        a2 = np.pad(a2,((0,0),(0,M1-M2)), mode='constant',
                    constant_values = np.nan)
    elif N2 < N1:
        a1=np.pad(a2,((0,0),(0,M2-M1)), mode='constant',
                  constant_values = np.nan)
    return np.vstack((a1, a2))