# -*- coding: utf-8 -*-

"""dielectric.py: Provides routines to fit a sum of Gaussians to a
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

# Local imports
from cosmo_pol.constants import global_constants as constants

def dielectric_ice(t,f):
    """
    Compute the complex dielectric constant of pure ice, based on a the article
    of G. Huffort "A model for the complex permittivity of ice at frequencies
    below 1 THz"
    Args:
        t: temperature in K
        f: frequency in GHz
    Returns:
        m: the complex dielectric constant m = x + iy
    """

    t = min(t, 273.15) # Stop at solidification temp

    theta = 300./t-1.
    alpha = (50.4 + 62.*theta)*10 ** -4 * np.exp(-22.1 * theta)
    beta = ((0.502 - 0.131 * theta) / (1. + theta) * 10**-4 + 0.542 * 10**-6
            *((1 + theta)/(theta + 0.0073)) ** 2)

    Epsilon_real=3.15
    Epsilon_imag=alpha/f+beta*f

    Epsilon = complex(Epsilon_real, Epsilon_imag)

    m = np.sqrt(Epsilon)
    return m


def dielectric_water(t,f):
    """
    Compute the complex dielectric constant of pure liquid water, based on
    the article of H.Liebe: "A model for the complex permittivity of
    water at frequencies below 1 THz"
    Args:
        t: temperature in K
        f: frequency in GHz
    Returns:
        m: the complex dielectric constant m = x + iy
    """

    Theta = 1 - 300./t
    Epsilon_0 = 77.66 - 103.3*Theta
    Epsilon_1 = 0.0671*Epsilon_0
    Epsilon_2 = 3.52 + 7.52*Theta
    Gamma_1 = 20.20 + 146.5*Theta + 316*Theta**2
    Gamma_2 = 39.8*Gamma_1

    term1 = Epsilon_0-Epsilon_1
    term2 = 1+(f/Gamma_1)**2
    term3 = 1+(f/Gamma_2)**2
    term4 = Epsilon_1-Epsilon_2
    term5 = Epsilon_2

    Epsilon_real = term1/term2 + term4/term3 + term5
    Epsilon_imag = (term1 / term2) * (f / Gamma_1) + (term4/term3) * (f / Gamma_2)

    Epsilon = complex(Epsilon_real, Epsilon_imag)

    m = np.sqrt(Epsilon)
    return m

def K_squared(frequency):
    """
    Computes the value of |K|^2 used in the definition of the refl. factor
    The temperature is assumed to be of 10°C
    Args:
        f: the frequency
    Returns:
        the value of |K|^2 at 10°C and the specified frequency
    """
    m = dielectric_water(constants.T_K_SQUARED, frequency)
    k = (m**2-1)/(m**2 + 2)
    return np.abs(k)**2

def dielectric_mixture(mix, m):
    """
    Recursive function that uses the Maxwell-Garnett Effective Medium
    Approximation to compute the dielectric constant of a mixture of
    two or more components.
    Args:
       m: Tuple of the complex dielectic constants of the components.
       mix: Tuple of the volume fractions of the components, len(mix)==len(m)
            (if sum(mix)!=1, these are taken relative to sum(mix))
    Returns:
       The Maxwell-Garnett approximation for the dielectric constant of
       the effective medium
    If len(m)==2, the first element is taken as the matrix and the second as
    the inclusion. If len(m)>2, the components are mixed recursively so that the
    last element is used as the inclusion and the second to last as the
    matrix, then this mixture is used as the last element on the next
    iteration, and so on.
    """

    mix = tuple(mix)
    m = tuple(m)
    if len(m) == 2:
        cF = float(mix[1]) / (mix[0]+mix[1]) * \
            (m[1]**2-m[0]**2) / (m[1]**2+2*m[0]**2)
        er = m[0]**2 * (1.0+2.0*cF) / (1.0-cF)
        m = np.sqrt(er)
    else:
        m_last = dielectric_mixture(mix[-2:], m[-2:])
        mix_last = mix[-2] + mix[-1]
        m = dielectric_mixture(mix[:-2] + (mix_last,), m[:-2] + (m_last,))
    return m

