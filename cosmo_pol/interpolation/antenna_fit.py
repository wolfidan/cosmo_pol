# -*- coding: utf-8 -*-

"""antenna_fit.py: Provides routines to fit a sum of Gaussians to a
provided real antenna diagram """

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
from scipy.optimize import minimize
from scipy.signal import argrelextrema



def _gaussian_sum(x,params):
    """
    Weighted sum gaussians
    Args:
        x: angle in degrees
        params: parameters of the sum of gaussians
    Returns:
        the value f(x) where f is a function defined by the weighted sum of
        gaussians
    """
    return 10*np.log10(np.sum([10**(0.1*p[0])* np.exp(-(x-p[1])**2/(2*p[2]**2))
                               for p in params],axis=0))

def _obj(params,x,y):
    """
    Computes the objective (cost) function needed for optimization
    Args:
        params: parameters of the sum of gaussians
        x: vector of angles in degrees
        y: real powers in dB from the real antenna diagram
    Returns:
        the value of the objective function corresponding to the specified
        parameters of the sum of Gaussians
    """
    params = np.reshape(params,(len(params)/3,3))
    est = _gaussian_sum(x,params)

    error = np.sqrt(np.sum((est-y)**2))

    return error


def antenna_diagram(x, bw, ampl, mu):
    """
    Creates an antenna diagram for a specified set of parameters corresponding
    to a weighted sum of Gaussians
    Args:
        x: vector of angles in degrees
        bw: 3dB beamwidths for all gaussians
        ampl: amplitude in dB of every gaussian
        mu: offset of every Gaussian (with respect to zero degrees)
    Returns:
        y: the powers corresponding to every angle in x
    """
    sigma = bw / (2 * np.sqrt(2 * np.log(2)))
    y = np.zeros(x.shape)
    for i in range(len(bw)):
        y = y + 10**(0.1 * ampl[i]) * np.exp(-(x - mu[i])**2/(2 * sigma[i]**2))
    return y



def optimize_gaussians(x,y,n_gaussians):
    """
    Fits a weighted sum of Gaussians to a specified real antenna diagram
    Args:
        x: vector of angles in degrees
        y: vector of powers in dB corresponding to the angles x
        n_gaussians: numbers of Gaussians to fit
    Returns:
        y: the powers corresponding to every angle in x
    """
    peaks = argrelextrema(y, np.greater)
    a_lobes = y[peaks]
    mu_lobes = x[peaks]

    if 0 not in mu_lobes:
        mu_lobes=np.append(mu_lobes,0)
        a_lobes=np.append(a_lobes,0)

    params = np.column_stack((a_lobes,mu_lobes))
    params = params[params[:,0].argsort()]
    params = np.flipud(params)

    selected = params[0:n_gaussians,:]

    p0 = np.column_stack((selected[:,0],selected[:,1],np.array([0.5]*n_gaussians)))

    bounds=[]
    for i in range(n_gaussians):
        for j in range(3):
            if j!=2:
                bounds.append([None,None])
            else:
                bounds.append([0.1,2])

    bounds[0]=[0,0]
    bounds[1]=[0,0]


    params = minimize(_obj,p0,args=(x,y),bounds=bounds,method='SLSQP')
    params = np.reshape(params['x'],(n_gaussians,3))
    return params
