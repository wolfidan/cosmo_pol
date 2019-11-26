# -*- coding: utf-8 -*-

"""hydrometeors.py: Provides classes with relevant functions for all
hydrometeor types considered in the radar operator.
Computes all diameter dependent properties (orientation, aspect-ratio,
dielectric constants, velocity, mass... """

__author__ = "Daniel Wolfensberger"
__copyright__ = "Copyright 2017, COSMO_POL"
__credits__ = ["Daniel Wolfensberger"]
__license__ = "GPL"
__version__ = "1.0"
__maintainer__ = "Daniel Wolfensberger"
__email__ = "daniel.wolfensberger@epfl.ch"

# Global importants
import numpy as np
np.seterr(divide='ignore') # Disable divide by zero error

from scipy.integrate import trapz
from scipy.optimize import fsolve, root, newton, brentq
from scipy.stats import gamma as gampdf
from scipy.special import gamma, erf
from scipy.interpolate import interp1d
from textwrap import dedent

# Local imports
from cosmo_pol.hydrometeors.dielectric import (dielectric_ice, dielectric_water,
                                    dielectric_mixture)

from cosmo_pol.constants import constants_1mom
from cosmo_pol.constants import constants_2mom
from cosmo_pol.constants import global_constants as constants
from cosmo_pol.utilities import vlinspace


###############################################################################

def create_hydrometeor(hydrom_type, scheme = '1mom', sedi = False):
    """
    Creates a hydrometeor class instance, for a specified microphysical
    scheme
    Args:
        hydrom_type: the hydrometeor types, can be either
            'R': rain, 'S': snow aggregates, 'G': graupel, 'H': hail (2-moment
            scheme only), 'mS': melting snow, 'mG': melting graupel
        scheme: microphysical scheme to use, can be either '1mom' (operational
           one-moment scheme) or '2mom' (non-operational two-moment scheme)
        sedi: boolean flag. If true, a more complex empirical
            velocity-diameter relation will be used for raindrops
            (in the two-moments scheme only). It is set to False by default
    Returns:
        A hydrometeor class instance (see below)
    """

    if  hydrom_type == 'R':
       return Rain(scheme, sedi)
    elif hydrom_type == 'S':
        return Snow(scheme)
    elif hydrom_type == 'G':
        return Graupel(scheme)
    elif hydrom_type == 'H':
        return Hail(scheme)
    elif hydrom_type == 'I':
        return IceParticle(scheme)
    elif hydrom_type == 'mS':
        return MeltingSnow(scheme)
    elif hydrom_type == 'mG':
        return MeltingGraupel(scheme)
    else:
        msg = """
        Invalid hydrometeor type, must be R, S, G, H, mS or mG
        """
        return ValueError(dedent(msg))




###############################################################################

class _Hydrometeor(object):
    '''
    Base class for rain, snow, hail and graupel, should not be initialized
    directly
    '''
    def __init__(self, scheme):
        """
        Create a Hydrometeor Class instance
        Args:
            scheme: microphysical scheme to use, can be either '1mom' (operational
               one-moment scheme) or '2mom' (non-operational two-moment scheme)

        Returns:
            A Hydrometeor class instance (see below)
        """
        if scheme not in ['1mom','2mom']:
            scheme = '1mom'
        self.precipitating = True
        self.scheme = scheme
        self.d_max = None # Max diam in the integration over D
        self.d_min = None # Min diam in the integration over D
        self.nbins_D = 1024 # Number of diameter bins used in the numerical integrations

        # Power-law parameters
        self.a = None
        self.b = None
        self.alpha = None
        self.beta = None

        # PSD parameters
        self.lambda_ = None
        self.N0 = None
        self.mu = None
        self.nu = None

        # Scattering parameters
        self.canting_angle_std = None

        # Integration factors
        self.lambda_factor = None
        self.vel_factor = None
        self.ntot_factor = None

        if self.scheme == '2mom':
            self.x_max = None
            self.x_min = None

    def get_N(self, D):
        """
        Returns the PSD in mm-1 m-3
        Args:
            D: vector or matrix of diameters in mm

        Returns:
            N: the number of particles for every diameter, same dimensions as D
        """
        try:
            if len(self.lambda_.shape) >= D.ndim:
                operator = lambda x,y: np.outer(x,y)
            else:
                operator = lambda x,y: np.multiply(x[:,None],y)

            if np.isscalar(self.N0):
                return self.N0*D**self.mu * np.exp(- operator(self.lambda_,D**self.nu))
            else:
                return operator(self.N0,D**self.mu) * \
                              np.exp(- operator(self.lambda_,D**self.nu))


        except:
            raise
            print('Wrong dimensions of input')

    def get_V(self,D):
        """
        Returns the terminal fall velocity i m/s
        Args:
            D: vector or matrix of diameters in mm

        Returns:
            V: the terminal fall velocities, same dimensions as D
        """
        V = self.alpha * D ** self.beta
        return V

    def get_D_from_V(self,V):
        """
        Returns the diameter for a specified terminal fall velocity by
        simply inverting the power-law relation
        Args:
            V: the terminal fall velocity in m/s

        Returns:
            the corresponding diameters in mm
        """
        return (V / self.alpha) ** (1. / self.beta)

    def integrate_V(self):
        """
        Integrates the terminal fall velocity over the PSD
        Args:
            None

        Returns:
            v: the integral of V(D) * (D)
            n: the integratal of the PSD: N(D)
        """

        v = (self.vel_factor * self.N0 * self.alpha / self.nu *
             self.lambda_ ** (-(self.beta + self.mu + 1) / self.nu))
        if self.scheme == '2mom':
            n = self.ntot
        else:
            n = self.ntot_factor*self.N0/self.nu*self.lambda_**(-(self.mu+1)/self.nu)
        if np.isscalar(v):
            v = np.array([v])
            n = np.array([n])

        return v, n

    def get_M(self,D):
        """
        Returns the mass of a particle in kg
        Args:
            D: vector or matrix of diameters in mm

        Returns:
            m: the particle masses, same dimensions as D
        """
        return self.a*D**self.b

    def set_psd(self,*args):
        """
        Sets the particle size distribution parameters
        Args:
            *args: for the one-moment scheme, see function redefinition in
                the more specific classes Rain, Snow, Graupel and Hail
                for the two-moment scheme, a tuple (QN, QM), where
                QN is the number concentration in m-3 (not used currently...)

        Returns:
            No return but initializes class attributes for psd estimation
        """
        if len(args) == 2 and self.scheme == '2mom':
            # First argument must be number density and second argument mass
            # density
            with np.errstate(divide='ignore'):
                qn = args[0]
                q = args[1]

                x_mean = np.minimum(np.maximum(q*1.0/(qn+constants.EPS),self.x_min),self.x_max)
                if len(x_mean.shape) > 1:
                    x_mean = np.squeeze(x_mean)
                _lambda = np.asarray((self.lambda_factor*x_mean)**(-self.nu/self.b))

                # This is done in order to make sure that lambda is always a 1D
                # array

                _lambda[q == 0] = np.nan
                if _lambda.shape == ():
                    _lambda = np.array([_lambda])

                _N0 = np.asarray((self.nu/self.ntot_factor)*qn*_lambda**((self.mu+1)/self.nu))
                _N0 = _N0 * 1000**(-(1+self.mu))

                _lambda = _lambda * 1000**(-self.nu)

                self.N0 = _N0.T
                self.lambda_ = _lambda.T

                if np.isscalar(self.N0):
                    self.N0 = np.array([self.N0])
                if np.isscalar(self.lambda_):
                    self.lambda_ = np.array([self.lambda_])
                self.ntot = args[0]


###############################################################################

class _Solid(_Hydrometeor):
    '''
    Base class for snow, hail and graupel, should not be initialized
    directly
    '''
    def get_fractions(self,D):
        """
        Returns the volumic fractions of pure ice and air within
        the melting snow particles
        Args:
            D: vector of diameters in mm

        Returns:
            A nx2 matrtix of fractions, where n is the number of dimensions.
            The first column is the fraction of ice, the second the
            fraction of air
        """
        # Uses COSMO mass-diameter rule
        f_ice = 6*self.a/(np.pi*constants.RHO_I)*D**(self.b-3)
        f_air = 1-f_ice
        return [f_ice, f_air]

    def get_m_func(self,T,f):
        """
        Returns a function giving the dielectric constant as a function of
        the diameters, depending on the frequency and the temperature.
        Used for the computation of scattering properties (see lut submodule)
        Args:
            T: temperature in K
            f: frequency in GHz

        Returns:
            A lambda function f(D) giving the dielectric constant as a function
            of the particle diameter
        """
        def func(D, T, f):
            frac = self.get_fractions(D)
            m_ice = dielectric_ice(T,f)
            return dielectric_mixture(frac, [m_ice, constants.M_AIR])
        return lambda D: func(D, T, f)

###############################################################################

class _MeltingHydrometeor(object):
    '''
    Base class for melting hydrometeors from which MeltingSnow and
    MeltingGraupel inherit, should not be used as such
    '''
    def __init__(self, scheme):
        """
        Create a melting Hydrometeor Class instance
        Args:
            scheme: microphysical scheme to use, can be either '1mom' (operational
               one-moment scheme) or '2mom' (non-operational two-moment scheme)

        Returns:
            A MeltingHydrometeor class instance (see below)
        """

        if scheme not in ['1mom','2mom']:
            scheme = '1mom'

        self.nbins_D = 1024
        self.scheme = scheme # Microphyscal scheme

        self.equivalent_rain = Rain(scheme)  # Rain class instance
        self.equivalent_solid = Graupel(scheme)  # Solid phase hydrometeor class instance
        self.prop_factor = None # see set_psd

    @property
    def f_wet(self): # wet fraction
        return self._f_wet
    @f_wet.setter
    def f_wet(self, fw):
        # The min and max diameters are recalculated when f_wet is modified
        self._f_wet = fw
        self.d_max = (self._f_wet * self.equivalent_rain.d_max +
                          (1 - self._f_wet) * self.equivalent_solid.d_max)
        self.d_min = (self._f_wet * self.equivalent_rain.d_min +
                          (1 - self._f_wet) * self.equivalent_solid.d_min)

    def get_N_new(self,D):
        """
        Returns the PSD in mm-1 m-3
        Args:
            D: vector or matrix of diameters in mm

        Returns:
            N: the number of particles for every diameter, same dimensions as D
        """
        if len(self.f_wet.shape) >= D.ndim:
            operator = lambda x,y: np.outer(x,y)
        else:
            operator = lambda x,y:  np.multiply(x[:,None],y)

        try:
            Rho_m = lambda d: self.get_M(d)/(np.pi/6 * d**3)
            D_r = lambda d: (Rho_m(d)/constants.RHO_W)**(1/3.) * d
            dD_r_over_dD = (D_r(D + 0.01) - D_r(D))/0.01

            c1 = operator(1 - self.f_wet, self.equivalent_solid.get_N(D)) + \
                operator(self.f_wet, self.equivalent_rain.get_N(D) *dD_r_over_dD)
            if self.prop_factor is not None:
                c2 = operator(self.prop_factor, c1)
            else:
                c2 = c1

            return c2
        except:
            raise
            print('Wrong dimensions of input')

    def get_N(self,D):
        if self.prop_factor is not None:
            if len(self.prop_factor.shape) > D.ndim:
                operator = lambda x,y: np.outer(x,y)
            else:
                operator = lambda x,y:  np.multiply(x[:,None],y)
        else:
            operator = lambda x,y: y

        try:
            Rho_m = lambda d: self.get_M(d)/(np.pi/6 * d**3)
            D_r = lambda d: (Rho_m(d)/constants.RHO_W)**(1/3.) * d
            dD_r_over_dD = (D_r(D + 0.01) - D_r(D))/0.01

            return operator(self.prop_factor, self.equivalent_rain.get_N(D_r(D))) \
                      * self.equivalent_rain.get_V(D_r(D)) / self.get_V(D) * dD_r_over_dD
        except:
            raise
            print('Wrong dimensions of input')


    def get_M(self,D):
        """
        Returns the mass of a particle in kg
        Args:
            D: vector or matrix of diameters in mm

        Returns:
            m: the particle masses, same dimensions as D
        """

        # Relation here is quadratic according to Jung et al. (2008)
        M_rain = self.equivalent_rain.get_M(D)
        M_dry = self.equivalent_solid.get_M(D)
        if len(self.f_wet.shape) >= M_dry.ndim:
            operator = lambda x,y: np.outer(x,y)
        else:
            operator = lambda x,y:  np.multiply(x[:,None],y)

        m = operator(self.f_wet**2, M_rain) + operator((1 - self.f_wet**2),
                     M_dry)
        return m

    def get_V(self,D):
        """
        Returns the terminal fall velocity i m/s
        Args:
            D: vector or matrix of diameters in mm

        Returns:
            V: the terminal fall velocities, same dimensions as D
        """

        Rho_m = lambda d: self.get_M(d)/(np.pi/6 * d**3)
        D_r = lambda d: (Rho_m(d)/constants.RHO_W)**(1/3.) * d

        V_rain = self.equivalent_rain.get_V(D_r(D))
        V_dry = self.equivalent_solid.get_V(D)
        # See Frick et al. 2013, for the phi function
        phi = 0.246 * self.f_wet + (1 - 0.246)*self.f_wet**7


        if len(self.f_wet.shape) > V_dry.ndim:
            operator = lambda x,y: np.outer(x,y)
        else:
            operator = lambda x,y:  np.multiply(x[:,None],y)

        return operator(phi, V_rain) + operator((1 - phi), V_dry)

    def integrate_V(self):
        """
        Integrates the terminal fall velocity over the PSD
        Args:
            None

        Returns:
            v: the integral of V(D) * (D)
            n: the integratal of the PSD: N(D)
        """
        # I don't think there is any analytical solution for this hydrometeor
        D = vlinspace(self.d_min,self.d_max, self.nbins_D)
        dD = D[:,1] - D[:,0]

        N = self.get_N(D)

        v = np.sum(N * self.get_V(D),axis=1) * dD
        n = np.sum(N, axis=1) * dD

        return v, n

    def integrate_M(self):
        """
        Integrates the particles mass over the PSD to get the total mass
        Args:
            None

        Returns:
            the total mass in kg
        """
        # I don't think there is any analytical solution for this hydrometeor
        D = vlinspace(self.d_min, self.d_max, self.nbins_D)
        dD = D[:,1] - D[:,0]

        if np.isscalar(self.d_min):
            return np.sum(self.get_N(D) * self.get_M(D)) * dD
        else:
            return np.sum(self.get_N(D) * self.get_M(D),axis=1) * dD

    def get_D_from_V(self,V):
        """
        Returns the diameter for a specified terminal fall velocity, this is
        done by using a linear interpolation
        Args:
            V: the terminal fall velocity in m/s

        Returns:
            the corresponding diameters in mm
        """
        if not self.vd_interpolator:
            D_all = np.linspace(np.min(self.d_min),np.max(self.d_max),
                            self.nbins_D)
            V_all = np.squeeze(self.get_V(D_all))
            self.vd_interpolator = interp1d(V_all, D_all,
                                            bounds_error= False,
                                            fill_value= np.nan,
                                            assume_sorted=False,
                                            copy = False)

        return self.vd_interpolator(V)


    def get_aspect_ratio(self, D):
        """
        Returns the aspect ratio for a given diameter, the aspect-ratio is
        defined here as the smallest dimension over the largest (between 0
        and 1)
        Args:
            D: vector or matrix of diameters in mm

        Returns:
            The vector or matrix of aspect-ratios, same dimensions as D
        """
        c_rain = self.equivalent_rain.canting_angle_std
        c_snow = self.equivalent_snow.get_canting_angle_std_masc(D)

        return self.f_wet * c_rain + (1 - self.f_wet) * c_snow

    def get_fractions(self, D):
        """
        Returns the volumic fractions of water, pure ice and air within
        the melting snow particles
        Args:
            D: vector of diameters in mm

        Returns:
            A nx3 matrtix of fractions, where n is the number of dimensions.
            The first column is the fraction of water, the second the fraction
            of ice and the third the fraction of air
        """
        # Dry fractions consist of air and ice
        # Uses COSMO mass-diameter rule
        D = np.array([D])
        self.f_wet = np.array([self.f_wet])
        rho_melt = self.get_M(D) / (1/6. * np.pi * D**3)

        f_wat = self.f_wet * rho_melt / constants.RHO_W
        f_ice = (rho_melt - f_wat * constants.RHO_W)/constants.RHO_I
        f_air = 1 - f_wat - f_ice
        fracs = np.squeeze(np.array([f_wat, f_ice, f_air]))
        fracs[fracs < 0] = 0.
        return fracs

    def get_m_func(self, T, f):
        """
        Returns a function giving the dielectric constant as a function of
        the diameters, depending on the frequency and the temperature.
        Used for the computation of scattering properties (see lut submodule)
        Args:
            T: temperature in K
            f: frequency in GHz

        Returns:
            A lambda function f(D) giving the dielectric constant as a function
            of the particle diameter
        """
        def func(D, T, f):
            frac = self.get_fractions(D)
            m_water = dielectric_water(T,f)
            m_ice = dielectric_ice(T,f)
            m = np.array([m_water, m_ice, constants.M_AIR])
            m_water_matrix = dielectric_mixture(frac,m)
            m_dry_snow = dielectric_mixture(frac[-2:], m[-2:])
            frac_dry_snow = frac[1] + frac[2]
            m_snow_matrix = dielectric_mixture((frac_dry_snow, frac[0]),
                                                     (m_dry_snow, m[0]))
            f_wat = frac[0]

            # See Ryzhkov 2010
            tau = erf( 2 * (1 - f_wat)/ f_wat - 1)
            return  0.5 * ((1 + tau) * m_snow_matrix + (1 - tau) * m_water_matrix)
        return lambda D: func(D, T, f)

    def get_canting_angle_std_masc(self, D):
        """
        Returns the standard deviation of the distribution of canting angles
        of melting snow hydrometeors for a given diameter
        Args:
            D: vector of diameters in mm

        Returns:
            the standard deviations of the distribution of canting angles
            for the specified diameters
        """
        c_rain = self.equivalent_rain.canting_angle_std
        c_snow = self.equivalent_solid.get_canting_angle_std_masc(D)

        return self.f_wet * c_rain + (1 - self.f_wet) * c_snow


    def get_aspect_ratio_pdf_masc(self, D):
        """
        Returns the parameters of the gamma distribution of aspect-ratios
        of melting snow for the specific diameter.
        the gamma pdf has the following form, where x is the aspect ratio
        p(x) = ((a - loc)^(lamb-1) exp(-(x - 1)/mu)) / mu^lamb* Gamma(lamb)

        Args:
            D: vector of diameters in mm

        Returns:
            lamb_wet: lambda parameter of the gamma pdf
            loc_wet: location parameter of the gamma pdf
            mu_wet: shape parameter of the gamma pdf

        """
        lamb_snow, loc_snow, mu_snow = self.equivalent_solid.get_aspect_ratio_pdf_masc(D)

        lamb_rain = lamb_snow

        if np.isscalar(lamb_snow):
            loc_rain = self.equivalent_rain.get_aspect_ratio(D)
            mu_rain = 1E-4
        else:
            n = len(lamb_snow)
            loc_rain = np.ones((n)) * self.equivalent_rain.get_aspect_ratio(D)
            mu_rain = 1E-4 * np.ones((n))

        lamb_wet = self.f_wet * lamb_rain + (1 - self.f_wet) * lamb_snow
        loc_wet = self.f_wet * loc_rain + (1 - self.f_wet) * loc_snow
        mu_wet = self.f_wet * mu_rain + (1 - self.f_wet) * mu_snow
        return lamb_wet, loc_wet, mu_wet


###############################################################################

class Rain(_Hydrometeor):
    '''
    Class for raindrops
    '''
    def __init__(self, scheme, sedi = True):
        """
        Create a Rain Class instance
        Args:
            scheme: microphysical scheme to use, can be either '1mom' (operational
               one-moment scheme) or '2mom' (non-operational two-moment scheme)
            sedi: boolean flag. If true, a more complex empirical
                velocity-diameter relation will be used for raindrops
                (in the two-moments scheme only)
        Returns:
            A Rain class instance (see below)
        """
        if scheme not in ['1mom','2mom']:
            scheme = '1mom'

        self.scheme = scheme
        self.sedi = (sedi if self.scheme == '2mom' else False)
        self.nbins_D = 1024

        if self.scheme == '2mom':
            self.d_max = constants_2mom.D_MAX_R
            self.d_min = constants_2mom.D_MIN_R
        else:
            self.d_max = constants_1mom.D_MAX_R
            self.d_min = constants_1mom.D_MIN_R

        # Power-law parameters
        self.a = (constants_1mom.AM_R if self.scheme == '1mom'
                  else constants_2mom.AM_R)
        self.b = (constants_1mom.BM_R if self.scheme == '1mom'
                  else constants_2mom.BM_R)
        self.alpha = (constants_1mom.AV_R if self.scheme == '1mom'
                      else constants_2mom.AV_R)
        self.beta = (constants_1mom.BV_R if self.scheme == '1mom'
                     else constants_2mom.BV_R)

        # PSD parameters
        self.lambda_ = None
        self.N0 = (constants_1mom.N0_R if self.scheme == '1mom' else None)
        self.mu = (constants_1mom.MU_R if self.scheme == '1mom'
                   else constants_2mom.MU_R)
        self.nu = (1.0 if self.scheme == '1mom' else constants_2mom.NU_R)

        # Canting angle stdev, taken from Bringi
        self.canting_angle_std = 10.

        # Others
        self.lambda_factor = (constants_1mom.LAMBDA_FACTOR_R if
                              self.scheme == '1mom' else
                              constants_2mom.LAMBDA_FACTOR_R)
        self.vel_factor = (constants_1mom.VEL_FACTOR_R if
                           self.scheme == '1mom' else
                           constants_2mom.VEL_FACTOR_R)
        self.ntot_factor=(constants_1mom.NTOT_FACTOR_R if
                          self.scheme == '1mom' else
                          constants_2mom.NTOT_FACTOR_R)
        self.ntot = None # Total number of particles

        if self.scheme == '2mom':
            self.x_max = constants_2mom.X_MAX_R
            self.x_min = constants_2mom.X_MIN_R
        else:
            self.vel_factor = constants_1mom.VEL_FACTOR_R

    def get_V(self, D):
        """
        Returns the terminal fall velocity i m/s
        Args:
            D: vector or matrix of diameters in mm

        Returns:
            V: the terminal fall velocities, same dimensions as D
        """
        if self.sedi:
            # Specific V-D relation for sedimentation
            # For small drops the velocity might be zero with this model...
            V = np.maximum(0, constants_2mom.C_1 - constants_2mom.C_2 *
                           np.exp(-constants_2mom.C_3*D))
            return V
        else:
            return super(Rain,self).get_V(D)

    def get_D_from_V(self,V):
        """
        Returns the diameter for a specified terminal fall velocity by
        simply inverting the power-law relation
        Args:
            V: the terminal fall velocity in m/s

        Returns:
            the corresponding diameters in mm
        """
        if self.sedi:
            # Specific V-D relation for sedimentation
            return (-1/constants_2mom.C_3 *
                    np.log((constants_2mom.C_1-V) / constants_2mom.C_2))
        else:
            return super(Rain,self).get_D_from_V(V)

    def integrate_V(self,*args):
        """
        Integrates the terminal fall velocity over the PSD
        Args:
            None

        Returns:
            v: the integral of V(D) * (D)
            n: the integratal of the PSD: N(D)
        """
        if not self.sedi or not self.scheme == '2mom':
            return super(Rain,self).integrate_V()
        else:
            D = args[0]
            dD = D[1] - D[0]
            return np.einsum('ij,i->j', self.get_N(D), self.get_V(D)) * dD, self.ntot

    def set_psd(self, *args):
        """
        Sets the particle size distribution parameters
        Args:
            *args: for the one-moment scheme, a tuple (QM) containing the
                mass concentration, for the two-moment scheme, a tuple
                (QN, QM), where QN is the number concentration in m-3
                (not used currently...)

        Returns:
            No return but initializes class attributes for psd estimation
        """
        if len(args) == 2 and self.scheme == '2mom':
            super(Rain,self).set_psd(*args)
        elif self.scheme == '1mom':
            with np.errstate(divide='ignore'):
                _lambda = np.array((self.lambda_factor/args[0])**
                                   (1./(4.+self.mu)))
                _lambda[args[0]==0] = np.nan
                self.lambda_ = _lambda
                self.ntot = (self.ntot_factor *
                             self.N0 * self.lambda_**(self.mu - 1))
                if np.isscalar(self.lambda_):
                    self.lambda_ = np.array([self.lambda_])
                if np.isscalar(self.ntot):
                    self.ntot = np.array([self.ntot])
        else:
            msg = '''
            Invalid call to function, if scheme == ''2mom'',
            input must be tuple of (QN,QM) if scheme == '1mom',
            input must be (QM)
            '''
            print(dedent(msg))

    def get_aspect_ratio(self, D):
        """
        Return the aspect ratio for specified diameters, based on Thurai et
        al (2007)
        Args:
            D: the vector of diameters

        Returns:
            The aspect-ratios defined by the smaller dimension over the
            larger dimension
        """

        if np.isscalar(D):
            D = np.array([D])

        ar = np.zeros((len(D),))
        ar[D<0.7] = 1.0
        mid_diam = np.logical_and(D<1.5,D>=0.7)
        ar[mid_diam] = (1.173 - 0.5165 * D[mid_diam] + 0.4698*D[mid_diam]**2
            - 0.1317*D[mid_diam]**3 - 8.5e-3*D[mid_diam]**4)

        ar[D>=1.5] = (1.065 - 6.25e-2 * D[D>=1.5] - 3.99e-3 * D[D>=1.5] ** 2 +
            7.66e-4 * D[D>=1.5] ** 3 - 4.095e-5 * D[D>=1.5] ** 4)

        # This model tends to diverge for large drops so we threshold it to
        # a reasonable max drop size
        ar[D>=self.d_max] = ar[D<=self.d_max][-1]

        return 1./ar


    def get_m_func(self,T,f):
        return lambda D: dielectric_water(T, f)

###############################################################################

class Snow(_Solid):
    '''
    Class for snow in the form of aggregates
    '''
    def __init__(self, scheme):
        """
        Create a Snow Class instance
        Args:
            scheme: microphysical scheme to use, can be either '1mom' (operational
               one-moment scheme) or '2mom' (non-operational two-moment scheme)
        Returns:
            A Snow class instance (see below)
        """
        if scheme not in ['1mom','2mom']:
            scheme = '1mom'

        self.scheme = scheme
        self.nbins_D = 1024

        if self.scheme == '2mom':
            self.d_max = constants_2mom.D_MAX_S
            self.d_min = constants_2mom.D_MIN_S
        else:
            self.d_max = constants_1mom.D_MAX_S
            self.d_min = constants_1mom.D_MIN_S

        # Power-law parameters
        self.a = (constants_1mom.AM_S if self.scheme == '1mom'
                  else constants_2mom.AM_S)
        self.b = (constants_1mom.BM_S if self.scheme == '1mom'
                  else constants_2mom.BM_S)
        self.alpha = (constants_1mom.AV_S if self.scheme == '1mom'
                      else constants_2mom.AV_S)
        self.beta = (constants_1mom.BV_S if self.scheme == '1mom'
                     else constants_2mom.BV_S)

        # PSD parameters
        self.lambda_ = None
        self.N0 = None
        self.mu = (constants_1mom.MU_S if self.scheme == '1mom'
                   else constants_2mom.MU_S)
        self.nu = (1 if self.scheme == '1mom' else constants_2mom.NU_S)

        # Scattering parameters
        self.canting_angle_std = 20.

        # Others
        self.lambda_factor = (constants_1mom.LAMBDA_FACTOR_S
                              if self.scheme == '1mom'
                              else constants_2mom.LAMBDA_FACTOR_S)
        self.vel_factor = (constants_1mom.VEL_FACTOR_S
                           if self.scheme == '1mom'
                           else constants_2mom.VEL_FACTOR_S)
        self.ntot_factor=(constants_1mom.NTOT_FACTOR_S
                          if self.scheme == '1mom'
                          else constants_2mom.NTOT_FACTOR_S)

        if self.scheme == '2mom':
            self.x_max = constants_2mom.X_MAX_S
            self.x_min = constants_2mom.X_MIN_S


    def set_psd(self,*args):
        """
        Sets the particle size distribution parameters
        Args:
            *args: for the one-moment scheme, a tuple (T,QM) containing the
                temperatue in K and the mass concentration QM,
                for the two-moment scheme, a tuple (QN, QM), where QN is the
                number concentration in m-3 (not used currently...)

        Returns:
            No return but initializes class attributes for psd estimation
        """
        if len(args) == 2 and self.scheme == '2mom':
            super(Snow,self).set_psd(*args)

        elif len(args) == 2 and self.scheme == '1mom':
            # For N0 use relation by Field et al. 2005 (QJRMS)
            self.N0 = 13.5*(5.65*10**5*np.exp(-0.107*(args[0]-273.15)))/1000 # mm^-1 m^-3
            with np.errstate(divide='ignore'):
                _lambda = np.array((self.a * self.N0 * self.lambda_factor
                                    / args[1]) ** (1. / (self.b + 1))) # in m-1
                _lambda[args[1] == 0] = np.nan
                self.lambda_ = _lambda
                self.ntot = (self.ntot_factor * self.N0 *
                             self.lambda_ ** (self.mu - 1))
                if np.isscalar(self.lambda_):
                    self.lambda_ = np.array([self.lambda_])
                if np.isscalar(self.ntot):
                    self.ntot = np.array([self.ntot])
        else:
            msg = '''
            Invalid call to function, if scheme == ''2mom'',
            input must be tuple of (QN,QM) if scheme == '1mom',
            input must be (T,QM)
            '''
            print(dedent(msg))

    def get_aspect_ratio(self, D):
        """
        Return the aspect ratio for specified diameters, based on Brandes
         et al (2007)
        Args:
            D: the vector of diameters

        Returns:
            The aspect-ratios defined by the smaller dimension over the
            larger dimension
        """
        ar = (0.01714 * D + 0.8467) # Brandes et al 2007 (Colorado snowstorms)
        return 1.0 / ar

    def get_aspect_ratio_pdf_masc(self,D):
        """
        Returns the parameters of the gamma distribution of aspect-ratios
        of snow for the specific diameter.
        the gamma pdf has the following form, where x is the aspect ratio
        p(x) = ((a - loc)^(lamb-1) exp(-(x - 1)/mu)) / mu^lamb* Gamma(lamb)

        Args:
            D: vector of diameters in mm

        Returns:
            lamb: lambda parameter of the gamma pdf
            loc: location parameter of the gamma pdf
            mu: shape parameter of the gamma pdf

        """
        lambd = constants.A_AR_LAMBDA_AGG*D**constants.B_AR_LAMBDA_AGG
        loc = np.ones(len(lambd))
        mu = constants.A_AR_M_AGG*D**constants.B_AR_M_AGG
        return lambd, loc, mu


    def get_canting_angle_std_masc(self, D):
        """
        Returns the standard deviation of the distribution of canting angles
        of snow hydrometeors for a given diameter
        Args:
            D: vector of diameters in mm

        Returns:
            the standard deviations of the distribution of canting angles
            for the specified diameters
        """
        cant_std = constants.A_CANT_STD_AGG * D ** constants.B_CANT_STD_AGG
        return cant_std

###############################################################################


class Graupel(_Solid):
    '''
    Class for graupel
    '''
    def __init__(self, scheme):
        """
        Create a Graupel Class instance
        Args:
            scheme: microphysical scheme to use, can be either '1mom' (operational
               one-moment scheme) or '2mom' (non-operational two-moment scheme)
        Returns:
            A Graupel class instance (see below)
        """
        if scheme not in ['1mom','2mom']:
            scheme = '1mom'

        self.scheme = scheme
        self.nbins_D = 1024

        if self.scheme == '2mom':
            self.d_max = constants_2mom.D_MAX_G
            self.d_min = constants_2mom.D_MIN_G
        else:
            self.d_max = constants_1mom.D_MAX_G
            self.d_min = constants_1mom.D_MIN_G

        # Power-law parameters
        self.a = (constants_1mom.AM_G if self.scheme == '1mom'
                  else constants_2mom.AM_G)
        self.b = (constants_1mom.BM_G if self.scheme == '1mom'
                  else constants_2mom.BM_G)
        self.alpha = (constants_1mom.AV_G if self.scheme == '1mom'
                      else constants_2mom.AV_G)
        self.beta = (constants_1mom.BV_G if self.scheme == '1mom'
                     else constants_2mom.BV_G)

        # PSD parameters

        self.lambda_ = None
        self.N0 = constants_1mom.N0_G
        self.mu = (constants_1mom.MU_G if self.scheme == '1mom' else constants_2mom.MU_G)
        self.nu = (1 if self.scheme == '1mom' else constants_2mom.NU_G)

        # Scattering parameters
        self.canting_angle_std = 40.

        # Others
        self.lambda_factor = (constants_1mom.LAMBDA_FACTOR_G if self.scheme == '1mom' else constants_2mom.LAMBDA_FACTOR_G)
        self.vel_factor = (constants_1mom.VEL_FACTOR_G if self.scheme == '1mom' else constants_2mom.VEL_FACTOR_G)
        self.ntot_factor=(constants_1mom.NTOT_FACTOR_G if self.scheme == '1mom' else constants_2mom.NTOT_FACTOR_G)

        if self.scheme == '2mom':
            self.x_max = constants_2mom.X_MAX_G
            self.x_min = constants_2mom.X_MIN_G


    def set_psd(self,*args):
        """
        Sets the particle size distribution parameters
        Args:
            *args: for the one-moment scheme, a tuple (QM) containing the mass
                concentration QM, for the two-moment scheme, a tuple (QN, QM),
                where QN is the number concentration in m-3
                (not used currently...)

        Returns:
            No return but initializes class attributes for psd estimation
        """
        if len(args) == 2 and self.scheme == '2mom':
            super(Graupel,self).set_psd(*args)
        elif self.scheme == '1mom':
            with np.errstate(divide='ignore'):
                _lambda = np.array((self.lambda_factor/args[0]) **
                                    (1./(4.+self.mu)))
                _lambda[args[0] == 0] = np.nan
                self.lambda_ = _lambda
                self.ntot = self.ntot_factor * self.N0 * self.lambda_**(self.mu - 1)
                if np.isscalar(self.lambda_):
                    self.lambda_ = np.array([self.lambda_])
                if np.isscalar(self.ntot):
                    self.ntot = np.array([self.ntot])
        else:
            msg = '''
            Invalid call to function, if scheme == ''2mom'',
            input must be tuple of (QN,QM) if scheme == '1mom',
            input must be (QM)
            '''
            print(dedent(msg))

    def get_aspect_ratio(self,D):
        """
        Return the aspect ratio for specified diameters, based on Garrett (2015)
        Args:
            D: the vector of diameters

        Returns:
            The aspect-ratios defined by the smaller dimension over the
            larger dimension
        """
        if np.isscalar(D):
            D=np.array([D])
        ar = 0.9*np.ones(len(D),)
        return 1.0/ar

    def get_aspect_ratio_pdf_masc(self,D):
        """
        Returns the parameters of the gamma distribution of aspect-ratios
        of snow for the specific diameter.
        the gamma pdf has the following form, where x is the aspect ratio
        p(x) = ((a - loc)^(lamb-1) exp(-(x - 1)/mu)) / mu^lamb* Gamma(lamb)

        Args:
            D: vector of diameters in mm

        Returns:
            lamb: lambda parameter of the gamma pdf
            loc: location parameter of the gamma pdf
            mu: shape parameter of the gamma pdf

        """
        lamb = constants.A_AR_LAMBDA_GRAU * D ** constants.B_AR_LAMBDA_GRAU
        loc =  np.ones(len(lamb))
        mu = constants.A_AR_M_GRAU*D**constants.B_AR_M_GRAU
        return lamb, loc, mu

    def get_canting_angle_std_masc(self,D):
        """
        Returns the standard deviation of the distribution of canting angles
        of snow hydrometeors for a given diameter
        Args:
            D: vector of diameters in mm

        Returns:
            the standard deviations of the distribution of canting angles
            for the specified diameters
        """
        cant_std = constants.A_CANT_STD_GRAU*D**constants.B_CANT_STD_GRAU
        return cant_std

###############################################################################


class Hail(_Solid):
    '''
    Class for Hail, exists only in the two-moment scheme
    '''
    def __init__(self, scheme='2mom'):
        """
        Create a Hail Class instance
        Args:
            scheme: microphysical scheme to use, can be either '1mom' (operational
               one-moment scheme) or '2mom' (non-operational two-moment scheme)
        Returns:
            A Hail class instance (see below)
        """
        self.scheme = '2mom' # No 1-moment scheme for hail
        self.nbins_D = 1024

        if self.scheme == '2mom':
            self.d_max = constants_2mom.D_MAX_H
            self.d_min = constants_2mom.D_MIN_H
        else:
            self.d_max = constants_1mom.D_MAX_H
            self.d_min = constants_1mom.D_MIN_H

        # Power-law parameters
        self.a = constants_2mom.AM_H
        self.b = constants_2mom.BM_H
        self.alpha = constants_2mom.AV_H
        self.beta = constants_2mom.BV_H

        # PSD parameters
        self.lambda_ = None
        self.N0 = None
        self.mu = constants_2mom.MU_H
        self.nu = constants_2mom.NU_H

        # Scattering parameters
        self.canting_angle_std = 40.

        # Others
        self.lambda_factor = constants_2mom.LAMBDA_FACTOR_H
        self.vel_factor = constants_2mom.VEL_FACTOR_H
        self.ntot_factor=  constants_2mom.NTOT_FACTOR_H

        self.x_max = constants_2mom.X_MAX_H
        self.x_min = constants_2mom.X_MIN_H

    def get_aspect_ratio(self,D):
        """
        Return the aspect ratio for specified diameters, based on Ryzhkov et
        al (2011)
        Args:
            D: the vector of diameters

        Returns:
            The aspect-ratios defined by the smaller dimension over the
            larger dimension
        """
        ar = np.ones((len(D),))
        ar[ar < 10] = 1 - 0.02*D
        ar[D >= 10] = 0.8
        return 1.0/ar


###############################################################################


class IceParticle(_Solid):
    '''
    Class for ice crystals
    '''
    def __init__(self,scheme='2mom'):
        """
        Create a IceParticle Class instance
        Args:
            scheme: microphysical scheme to use, can be either '1mom' (operational
               one-moment scheme) or '2mom' (non-operational two-moment scheme)
        Returns:
            An IceParticle class instance (see below)
        """
        self.scheme = scheme
        self.nbins_D = 1024

        if self.scheme == '2mom':
            self.d_max = constants_2mom.D_MAX_I
            self.d_min = constants_2mom.D_MIN_I
        else:
            self.d_max = constants_1mom.D_MAX_I
            self.d_min = constants_1mom.D_MIN_I

        # Power-law parameters
        self.a = (constants_1mom.AM_I if self.scheme == '1mom'
                  else constants_2mom.AM_I)
        self.b = (constants_1mom.BM_I if self.scheme == '1mom'
                  else constants_2mom.BM_I)
        self.alpha = (constants_1mom.AV_I if self.scheme == '1mom'
                      else constants_2mom.AV_I)
        self.beta = (constants_1mom.BV_I if self.scheme == '1mom'
                     else constants_2mom.BV_I)

        # PSD parameters
        self.lambda_ = None
        self.N0 = None
        self.mu = (constants_1mom.MU_I if self.scheme == '1mom' else constants_2mom.MU_I)
        self.nu = (1 if self.scheme == '1mom' else constants_2mom.NU_I)

        # Scattering parameters
        # See Noel and Sassen 2004
        self.canting_angle_std = 5.

        # Others
        self.lambda_factor =  (constants_1mom.LAMBDA_FACTOR_I if self.scheme == '1mom'
                               else constants_2mom.LAMBDA_FACTOR_I)
        self.ntot_factor = (constants_1mom.NTOT_FACTOR_I if self.scheme == '1mom'
                            else constants_2mom.NTOT_FACTOR_I)
        self.vel_factor = (constants_1mom.VEL_FACTOR_I if self.scheme == '1mom'
                            else constants_2mom.VEL_FACTOR_I)

        self.x_max = constants_2mom.X_MAX_I
        self.x_min = constants_2mom.X_MIN_I

    def get_N(self,D):
        """
        Returns the PSD in mm-1 m-3 using the moment explained in the paper
        based on the double-normalized PSD for moments 2,3, from
        Field et al. (2005)
        Args:
            D: vector or matrix of diameters in mm

        Returns:
            N: the number of particles for every diameter, same dimensions as D
        """

        try:
            if self.scheme == '1mom':
                # Taken from Field et al (2005)
                x = self.lambda_[:,None] * D / 1000.
                return self.N0[:,None] * constants_1mom.PHI_23_I(x)
            else:
                return (self.N0[:,None] * D ** self.mu *
                        np.exp(-self.lambda_[:,None]*D**self.nu))

        except:
            raise
            print('Wrong dimensions of input')

    def integrate_V(self):
        """
        Integrates the terminal fall velocity over the PSD
        Args:
            None

        Returns:
            v: the integral of V(D) * (D)
            n: the integratal of the PSD: N(D)
        """
        # Again no analytical solution here
        D = np.linspace(self.d_min, self.d_max, self.nbins_D)
        dD = D[1] - D[0]
        N = self.get_N(D)
        v =  np.sum(N * self.get_V(D)) * dD
        n = np.sum(N) * dD
        if np.isscalar(v):
            v = np.array([v])
            n = np.array([n])
        return v, n

    def get_mom_2(self, T, QM):
        """
        Get second moment from third moment using the best fit provided
        by Field et al (2005)
        Args:
            T: temperature in K
            QM: mass concentration in kg/m3

        Returns:
            The estimated moment of orf_{\text{vol}}^{\text{water}} & =  f_{\text{wet}}^m \frac{\rho^{\text{water}}}{\rho^{m}} \\der 2 of the PSD
        """
        n = 3
        T = T - constants.T0 # Convert to celcius
        a = 5.065339 - 0.062659 * T -3.032362 * n + 0.029469 * T * n \
            - 0.000285 * T**2  + 0.312550 * n**2 + 0.000204 * T**2*n \
            + 0.003199 * T* n**2 - 0.015952 * n**3

        a  = 10**(a)
        b = 0.476221 - 0.015896 * T + 0.165977 * n + 0.007468 * T * n \
            - 0.000141 * T**2 + 0.060366 * n**2 + 0.000079 * T**2 * n \
            + 0.000594 * T * n**2 -0.003577 * n**3

        return (QM/a) ** (1/b)


    def set_psd(self, arg1, arg2):
        """
        Sets the particle size distribution parameters
        Args:
            *args: for the one-moment scheme, a tuple (T, QM) containing the
                temperature in K and the mass concentration QM,
                for the two-moment scheme, a tuple (QN, QM), where QN is the
                number concentration in m-3 (not used currently...)

        Returns:
            No return but initializes class attributes for psd estimation
        """

        # Reference is Field et al. (2005)
        # We use the two-moment normalization to get a PSD
        # if one moment scheme, arg1 = T, arg2 = Q(M)I
        # if two moments scheme, arg1 =  QNI, arg2 = Q(M)I

        QM = arg2.astype(np.float64)

        if self.scheme == '1mom':
            T = arg1
            Q2 = self.get_mom_2(T,QM/constants_1mom.BM_I)
            N0 = Q2**((self.b + 1)/(self.b - 2)) * QM**((2 + 1)/(2 - self.b))
            N0 /= 10**5 # From m^-1 to mm^-1

            lambda_ = (Q2/QM) ** (1/(self.b - 2))

            # Apply correction factor to match third moment
            D = np.linspace(self.d_min, self.d_max, self.nbins_D)

            x = lambda_[:,None] * D.T / 1000

            N = N0[:,None] * constants_1mom.PHI_23_I(x)

            QM_est = np.nansum(self.a * D ** self.b * N, axis = 1) * (D[1]-D[0])

            N0 = N0/QM_est * QM

        else:
            QN = arg1.astype(np.float64)
            with np.errstate(divide='ignore'):
                x_mean = np.minimum(np.maximum(QM * 1.0/(QN+constants.EPS),
                                               self.x_min), self.x_max)
                if len(x_mean.shape) > 1:
                    x_mean = np.squeeze(x_mean)

                lambda_ = np.array((self.lambda_factor*x_mean)**(-self.nu/self.b))
                # This is done in order to make sure that lambda is always a 1D
                # array
                lambda_[QM == 0] = float('nan')
                if not lambda_.shape:
                    lambda_ = np.array([lambda_])

                N0 = np.asarray((self.nu/self.ntot_factor)*QN*lambda_**((self.mu+1)/self.nu))
                N0 = N0 * 1000**(-(1+self.mu))

                lambda_ = lambda_ * 1000**(-self.nu)


        self.N0 = N0.T
        self.lambda_ = lambda_.T
        if self.scheme == '2mom':
            self.ntot = self.ntot_factor * self.N0 * self.lambda_**(-self.mu - 1)
        else:
            self.ntot = np.nansum(N, axis = 1) *(D[1]-D[0])


        if np.isscalar(self.lambda_):
            self.lambda_ = np.array([self.lambda_])
        if np.isscalar(self.ntot):
            self.ntot = np.array([self.ntot])

    def get_aspect_ratio(self,D):
        """
        Return the aspect ratio for specified diameters, based on  Auer and
        Veal (1970)
        Args:
            D: the vector of diameters

        Returns:
            The aspect-ratios defined by the smaller dimension over the
            larger dimension
        """
        ar = 11.3 * D ** 0.414 * 1000 ** (-0.414)
        return 1/ar



###############################################################################

class MeltingSnow(_MeltingHydrometeor):
    def __init__(self, scheme):
        super(MeltingSnow, self).__init__(scheme)
        self.equivalent_solid = Snow(scheme)

    def set_psd(self, *args):
        """
        Sets the particle size distribution parameters
        Args:
            *args: for the one-moment scheme a tuple (T,QM,fw) containing
                the temperature T in K, the mass concentration QM in kg/m3,
                the wet fraction fw, (from 0 to 1)
                for the two-moment scheme, a tuple (QN, QM), where
                QN is the number concentration in m-3 (not used currently...)

        Returns:
            No return but initializes class attributes for psd estimation
        """
        self.prop_factor = None
        if (len(args) == 3 and self.scheme == '2mom') or \
            (len(args) == 3 and self.scheme == '1mom'):
            with np.errstate(divide='ignore'):

                if self.scheme == '1mom':
                    T = np.array(args[0])
                    q = np.array(args[1])
                    fw =  np.array(args[2])
                    self.equivalent_solid.set_psd(T,q)
                    self.equivalent_rain.set_psd(q)
                    self.f_wet = fw

                elif self.scheme == '2mom':
                    q = np.squeeze(np.array([args[1]]))
            # Reset vd_interpolator since f_wet has changed
            self.vd_interpolator = None
            self.prop_factor = q / self.integrate_M()

            if np.isscalar(self.f_wet):
                self.f_wet = np.array([self.f_wet])

        else:
            raise ValueError('Invalid parameters, cannot set psd')

###############################################################################

class MeltingGraupel(_MeltingHydrometeor):
    '''
    Melting graupel class
    '''
    def __init__(self, scheme):
        super(MeltingGraupel, self).__init__(scheme)
        self.equivalent_solid = Graupel(scheme)

    def set_psd(self, *args):
        """
        Sets the particle size distribution parameters
        Args:
            *args: for the one-moment scheme, aa tuple (QM,fw) containing
                the mass concentration QM in kg/m3, the wet fraction
                fw, (from 0 to 1)
                for the two-moment scheme, a tuple (QN, QM), where
                QN is the number concentration in m-3 (not used currently...)

        Returns:
            No return but initializes class attributes for psd estimation
        """
        self.prop_factor = None
        if (len(args) == 3 and self.scheme == '2mom') or \
            (len(args) == 2 and self.scheme == '1mom'):
            with np.errstate(divide='ignore'):

                if self.scheme == '1mom':
                    q = np.array(args[0])
                    fw =  np.array(args[1])
                    self.equivalent_solid.set_psd(q)
                    self.equivalent_rain.set_psd(q)
                    self.f_wet = fw

                elif self.scheme == '2mom':
                    q = np.array([args[1]])
            # Reset vd_interpolator since f_wet has changed
            self.vd_interpolator = None
            self.prop_factor = q / self.integrate_M()

            if np.isscalar(self.f_wet):
                self.f_wet = np.array([self.f_wet])

        else:
            raise ValueError('Invalid parameters, cannot set psd')

if __name__ == '__main__':

    ms = MeltingGraupel('1mom')

    D = np.linspace(0.5,15,110)
    ms.set_psd(np.array([0.01]),np.array([0.7]))

    s = Graupel('1mom')

    N = ms.get_N(D)
    s.set_psd(np.array([0.01]))
    N2 = s.get_N(D)
    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(D,N2[0], D , N[0])
#    fwet = np.linspace(0.01,0.99,120)
#    D = np.linspace(0.5,15,110)
#    all_frac = []
#    for f in fwet:
#        mg = MeltingGraupel('1mom')
#        mg.f_wet = f
#        ff = mg.get_fractions(D)
#        all_frac.append(ff[0])
#    all_frac_g = np.array(all_frac)
#
#    fwet = np.linspace(0.01,0.99,120)
#    D = np.linspace(0.5,15,110)
#    all_frac = []
#    for f in fwet:
#        mg = MeltingSnow('1mom')
#        mg.f_wet = f
#        ff = mg.get_fractions(D)
#        all_frac.append(ff[0])
#    all_frac_s = np.array(all_frac)
#    import matplotlib.pyplot as plt
#    plt.figure()
#    plt.plot(D,all_frac_g[60])
#    plt.plot(D,all_frac_s[60])
#    plt.legend(['Graupel','Snow'])
#    plt.grid()
#
#    plt.xlabel('Diameter [mm]')
#    plt.ylabel('Volume fraction of water: vol. water / vol. hydrometeor [-]')
#    plt.title('Wet fraction = 0.5 [-]')
#    plt.savefig('ex_diam.pdf',bbox_inches='tight')
#    plt.figure()
#    plt.grid()
#    plt.plot(fwet,all_frac_g[:,49])
#    plt.plot(fwet,all_frac_s[:,49])
#    plt.legend(['Graupel','Snow'])
#    plt.xlabel('Wet fraction : mass water / mass hydrometeor [-]')
#    plt.ylabel('Volume fraction of water: vol. water / vol. hydrometeor [-]')
#    plt.title('Diameter = 7 [mm]')
#    plt.savefig('ex_fwet.pdf',bbox_inches='tight')
#
#
#    import matplotlib.pyplot as plt
#    all_frac = np.array(all_frac)
