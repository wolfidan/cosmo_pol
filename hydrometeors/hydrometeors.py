
import numpy as np
from scipy.integrate import trapz
from scipy.optimize import fsolve, root, newton, brentq
from scipy.stats import gamma as gampdf
from scipy.special import gamma
from scipy.interpolate import interp1d

import dielectric

from cosmo_pol.constants import constants
from cosmo_pol.constants import constants_1mom
from cosmo_pol.constants import constants_2mom
from cosmo_pol.utilities.utilities import vlinspace

import matplotlib as mpl
label_size = 14
mpl.rcParams['xtick.labelsize'] = label_size
mpl.rcParams['ytick.labelsize'] = label_size
import matplotlib.pyplot as plt

###############################################################################

def create_hydrometeor(hydrom_type,scheme, sedi = False):
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

###############################################################################

class MeltingSnow(object):
    def __init__(self,scheme):
        if scheme not in ['1mom','2mom']:
            scheme = '1mom'

        self.scheme = scheme

        # These two correspond to the extreme behaviours of melting snow
        # f_wet -> 1: melting snow tends to equivalent rain
        # f_wet -> 0: melting snow tends to equivalent (dry) snow
        self.equivalent_rain = Rain(scheme)
        self.equivalent_snow = Snow(scheme)

        self.prop_factor = None

    @property
    def f_wet(self):
        return self._f_wet
    @f_wet.setter
    def f_wet(self, fw):
        self._f_wet = fw
        self.d_max = (self._f_wet * self.equivalent_rain.d_max +
                          (1 - self._f_wet) * self.equivalent_snow.d_max)
        self.d_min = (self._f_wet * self.equivalent_rain.d_min +
                          (1 - self._f_wet) * self.equivalent_snow.d_min)

    def get_N(self,D):
        if len(self.f_wet.shape) >= D.ndim:
            operator = lambda x,y: np.outer(x,y)
        else:
            operator = lambda x,y:  np.multiply(x[:,None],y)


        try:
#            return operator(self.prop_factor, self.equivalent_snow.get_N(D)) \
#                      * self.equivalent_snow.get_V(D) / self.get_V(D)
            c1 = operator(1 - self.f_wet, self.equivalent_snow.get_N(D)) + \
                operator(self.f_wet, self.equivalent_rain.get_N(D))
            if self.prop_factor is not None:
                c2 = operator(self.prop_factor, c1)
            else:
                c2 = c1

            return c2
        except:
            raise
            print('Wrong dimensions of input')


    def get_V(self,D):
        V_rain = self.equivalent_rain.get_V(D)
        V_snow = self.equivalent_snow.get_V(D)
        # See Frick et al. 2013, for the phi function
        phi = 0.246 * self.f_wet + (1 - 0.246)*self.f_wet**7

        if len(self.f_wet.shape) >= V_snow.ndim:
            operator = lambda x,y: np.outer(x,y)
        else:
            operator = lambda x,y:  np.multiply(x[:,None],y)

        return operator(phi, V_rain) + operator((1 - phi), V_snow)

    def integrate_V(self):
        # I don't think there is any analytical solution for this hydrometeor
        D = vlinspace(self.d_min,self.d_max,constants.N_BINS_D)
        dD = D[:,1] - D[:,0]

        N = self.get_N(D)

        v = np.sum(N * self.get_V(D),axis=1) * dD
        n = np.sum(N, axis=1) * dD

        return v, n

    def integrate_M(self):
        # I don't think there is any analytical solution for this hydrometeor
        D = vlinspace(self.d_min,self.d_max,constants.N_BINS_D)
        dD = D[:,1] - D[:,0]

        if np.isscalar(self.d_min):
            return np.sum(self.get_N(D) * self.get_M(D)) * dD
        else:
            return np.sum(self.get_N(D) * self.get_M(D),axis=1) * dD

    def get_D_from_V(self,V):
        # NOTE : it is faster to create a linear interpolator in set_psd
        # than solving a system for every velocity...

        if not self.vd_interpolator:
            D_all = np.linspace(np.min(self.d_min),np.max(self.d_max),
                            constants.N_BINS_D)
            V_all = np.squeeze(self.get_V(D_all))
            self.vd_interpolator = interp1d(V_all, D_all,
                                            bounds_error= False,
                                            fill_value= np.nan,
                                            assume_sorted=False,
                                            copy = False)

        return self.vd_interpolator(V)

    def get_M(self,D):
        # Relation here is quadratic according to Jung et al. (2008)
        M_rain = self.equivalent_rain.get_M(D)
        M_snow = self.equivalent_snow.get_M(D)
        if len(self.f_wet.shape) >= M_snow.ndim:
            operator = lambda x,y: np.outer(x,y)
        else:
            operator = lambda x,y:  np.multiply(x[:,None],y)


        return operator(self.f_wet**2, M_rain) + operator((1 - self.f_wet**2), M_snow)
#        else:
#            return np.outer(self.f_wet**2, M_rain) + np.outer((1 - self.f_wet**2), M_snow)

    def set_psd(self,*args):
#        D = np.linspace(self.d_min,self.d_max,1024)
#        dD = D[1] - D[0]
        self.prop_factor = None
        if (len(args) == 3 and self.scheme == '2mom') or \
            (len(args) == 3 and self.scheme == '1mom'):
            with np.errstate(divide='ignore'):

                if self.scheme == '1mom':
                    T = np.array(args[0])
                    q = np.array(args[1])
                    fw =  np.array(args[2])
                    self.equivalent_snow.set_psd(T,q)
                    self.equivalent_rain.set_psd(q)
                    self.f_wet = fw

                elif self.scheme == '2mom':
                    q = np.squeeze(np.array([args[1]]))
#            # Reset vd_interpolator since f_wet has changed
            self.vd_interpolator = None
            self.prop_factor = q / self.integrate_M()

            if np.isscalar(self.f_wet):
                self.f_wet = np.array([self.f_wet])

        else:
            raise ValueError('Invalid parameters, cannot set psd')

    def get_aspect_ratio(self,D):
        c_rain = self.equivalent_rain.canting_angle_std
        c_snow = self.equivalent_snow.get_canting_angle_std_masc(D)

        return self.f_wet * c_rain + (1 - self.f_wet) * c_snow

    def get_fractions(self,D):
        # Dry fractions consist of air and ice
        # Uses COSMO mass-diameter rule
        D = np.array([D])
        self.f_wet = np.array([self.f_wet])
        rho_melt = self.get_M(D) / (1/6. * np.pi * D**3)# Density of melting hydrom.

        f_wat = self.f_wet * rho_melt / constants.RHO_W
        f_ice = (rho_melt - f_wat * constants.RHO_W)/constants.RHO_I
        f_air = 1 - f_wat - f_ice
        fracs = np.squeeze(np.array([f_wat, f_ice, f_air]))
        fracs[fracs < 0] = 0.
        return fracs

    def get_m_func(self,T,f):
        def func(D, T, f):
            frac = self.get_fractions(D)
            m_water = dielectric.dielectric_water(T,f)
            m_ice = dielectric.dielectric_ice(T,f)
            m = np.array([m_water, m_ice, constants.M_AIR])
            return dielectric.dielectric_mixture(frac,m)
        return lambda D: func(D, T, f)

    def get_canting_angle_std_masc(self,D):
        c_rain = self.equivalent_rain.canting_angle_std
        c_snow = self.equivalent_snow.get_canting_angle_std_masc(D)

        return self.f_wet * c_rain + (1 - self.f_wet) * c_snow

    def get_aspect_ratio_pdf_masc(self,D):
        alpha_snow, loc_snow, scale_snow = self.equivalent_snow.get_aspect_ratio_pdf_masc(D)

        alpha_rain = alpha_snow

        if np.isscalar(alpha_snow):
            loc_rain = self.equivalent_rain.get_aspect_ratio(D)
            scale_rain = 1E-4
        else:
            n = len(alpha_snow)
            loc_rain = np.ones((n)) * self.equivalent_rain.get_aspect_ratio(D)
            scale_rain = 1E-4 * np.ones((n))

        alpha_wet = self.f_wet * alpha_rain + (1 - self.f_wet) * alpha_snow
        loc_wet = self.f_wet * loc_rain + (1 - self.f_wet) * loc_snow
        scale_wet = self.f_wet * scale_rain + (1 - self.f_wet) * scale_snow
        return alpha_wet, loc_wet, scale_wet

###############################################################################

class MeltingGraupel(object):
    def __init__(self,scheme):
        if scheme not in ['1mom','2mom']:
            scheme = '1mom'

        self.scheme = scheme

        # These two correspond to the extreme behaviours of melting snow
        # f_wet -> 1: melting snow tends to equivalent rain
        # f_wet -> 0: melting snow tends to equivalent (dry) snow
        self.equivalent_rain = Rain(scheme)
        self.equivalent_graupel = Graupel(scheme)

        self.prop_factor = None

    @property
    def f_wet(self):
        return self._f_wet
    @f_wet.setter
    def f_wet(self, fw):
        self._f_wet = fw
        self.d_max = (self._f_wet * self.equivalent_rain.d_max +
                          (1 - self._f_wet) * self.equivalent_graupel.d_max)
        self.d_min = (self._f_wet * self.equivalent_rain.d_min +
                          (1 - self._f_wet) * self.equivalent_graupel.d_min)

    def get_N(self,D):
        if len(self.f_wet.shape) >= D.ndim:
            operator = lambda x,y: np.outer(x,y)
        else:
            operator = lambda x,y:  np.multiply(x[:,None],y)


        try:
#            return operator(self.prop_factor, self.equivalent_snow.get_N(D)) \
#                      * self.equivalent_snow.get_V(D) / self.get_V(D)
            c1 = operator(1 - self.f_wet, self.equivalent_graupel.get_N(D)) + \
                operator(self.f_wet, self.equivalent_rain.get_N(D))
            if self.prop_factor is not None:
                c2 = operator(self.prop_factor, c1)
            else:
                c2 = c1

            return c2
        except:
            raise
            print('Wrong dimensions of input')

    def get_V(self,D):
        V_rain = self.equivalent_rain.get_V(D)
        V_graupel = self.equivalent_graupel.get_V(D)
        # See Frick et al. 2013, for the phi function
        phi = 0.246 * self.f_wet + (1 - 0.246)*self.f_wet**7

        if len(self.f_wet.shape) >= V_graupel.ndim:
            operator = lambda x,y: np.outer(x,y)
        else:
            operator = lambda x,y:  np.multiply(x[:,None],y)

        return operator(phi, V_rain) + operator((1 - phi), V_graupel)

    def integrate_V(self):
        # I don't think there is any analytical solution for this hydrometeor
        D = vlinspace(self.d_min,self.d_max,constants.N_BINS_D)
        dD = D[:,1] - D[:,0]

        N = self.get_N(D)

        v = np.sum(N * self.get_V(D),axis=1) * dD
        n = np.sum(N, axis=1) * dD

        return v,n

    def integrate_M(self):
        # I don't think there is any analytical solution for this hydrometeor
        D = vlinspace(self.d_min,self.d_max,constants.N_BINS_D)
        dD = D[:,1] - D[:,0]
        if np.isscalar(self.d_min):
            return np.sum(self.get_N(D) * self.get_M(D)) * dD
        else:
            return np.sum(self.get_N(D) * self.get_M(D),axis=1) * dD

    def get_D_from_V(self,V):
        # NOTE : it is faster to create a linear interpolator in set_psd
        # than solving a system for every velocity...

        if not self.vd_interpolator:
            D_all = np.linspace(np.min(self.d_min),np.max(self.d_max),
                            constants.N_BINS_D)
            V_all = np.squeeze(self.get_V(D_all))
            self.vd_interpolator = interp1d(V_all, D_all,
                                            bounds_error= False,
                                            fill_value= np.nan,
                                            assume_sorted=False,
                                            copy = False)

        return self.vd_interpolator(V)

    def get_M(self,D):
        # Relation here is quadratic according to Jung et al. (2008)
        M_rain = self.equivalent_rain.get_M(D)
        M_graupel = self.equivalent_graupel.get_M(D)

        if len(self.f_wet.shape) >= M_graupel.ndim:
            operator = lambda x,y: np.outer(x,y)
        else:
            operator = lambda x,y:  np.multiply(x[:,None],y)


        return operator(self.f_wet**2, M_rain) + operator((1 - self.f_wet**2), M_graupel)
#        else:
#            return np.outer(self.f_wet**2, M_rain) + np.outer((1 - self.f_wet**2), M_graupel)

    def set_psd(self,*args):
        self.prop_factor = None
        if (len(args) == 3 and self.scheme == '2mom') or \
            (len(args) == 2 and self.scheme == '1mom'):
            with np.errstate(divide='ignore'):

                if self.scheme == '1mom':
                    q = np.array(args[0])
                    fw =  np.array(args[1])
                    self.equivalent_graupel.set_psd(q)
                    self.equivalent_rain.set_psd(q)
                    self.f_wet = fw

                elif self.scheme == '2mom':
                    q = np.array([args[1]])
#            # Reset vd_interpolator since f_wet has changed
            self.vd_interpolator = None
            self.prop_factor = q / self.integrate_M()

            if np.isscalar(self.f_wet):
                self.f_wet = np.array([self.f_wet])

        else:
            raise ValueError('Invalid parameters, cannot set psd')

    def get_aspect_ratio(self,D):
        c_rain = self.equivalent_rain.canting_angle_std
        c_graupel = self.equivalent_graupel.get_canting_angle_std_masc(D)

        return self.f_wet * c_rain + (1 - self.f_wet) * c_graupel

    def get_fractions(self,D):
        # Dry fractions consist of air and ice
        # Uses COSMO mass-diameter rule
        D = np.array([D])
        self.f_wet = np.array([self.f_wet])
        rho_melt = self.get_M(D) / (1/6. * np.pi * D**3)# Density of melting hydrom.

        f_wat = self.f_wet * rho_melt / constants.RHO_W
        f_ice = (rho_melt - f_wat * constants.RHO_W)/constants.RHO_I
        f_air = 1 - f_wat - f_ice
        fracs = np.squeeze(np.array([f_wat, f_ice, f_air]))
        fracs[fracs < 0] = 0.
        return fracs

    def get_m_func(self,T,f):
        def func(D, T, f):
            frac = self.get_fractions(D)
            m_water = dielectric.dielectric_water(T,f)
            m_ice = dielectric.dielectric_ice(T,f)
            m = np.array([m_water, m_ice, constants.M_AIR])
            return dielectric.dielectric_mixture(frac,m)
        return lambda D: func(D, T, f)

    def get_canting_angle_std_masc(self,D):
        c_rain = self.equivalent_rain.canting_angle_std
        c_graupel = self.equivalent_graupel.get_canting_angle_std_masc(D)

        return self.f_wet * c_rain + (1 - self.f_wet) * c_graupel

    def get_aspect_ratio_pdf_masc(self,D):
        alpha_graupel, loc_graupel, scale_graupel = self.equivalent_graupel.get_aspect_ratio_pdf_masc(D)

        alpha_rain = alpha_graupel

        if np.isscalar(alpha_graupel):
            loc_rain = self.equivalent_rain.get_aspect_ratio(D)
            scale_rain = 1E-4
        else:
            n = len(alpha_graupel)
            loc_rain = np.ones((n)) * self.equivalent_rain.get_aspect_ratio(D)
            scale_rain = 1E-4 * np.ones((n))

        alpha_wet = self.f_wet * alpha_rain + (1 - self.f_wet) * alpha_graupel
        loc_wet = self.f_wet * loc_rain + (1 - self.f_wet) * loc_graupel
        scale_wet = self.f_wet * scale_rain + (1 - self.f_wet) * scale_graupel
        return alpha_wet, loc_wet, scale_wet



###############################################################################

class Hydrometeor(object):
    def __init__(self, scheme):
        if scheme not in ['1mom','2mom']:
            scheme = '1mom'
        self.precipitating = True
        self.scheme = scheme
        self.d_max=None # Max diam in the integration over D
        self.d_min=None # Min diam in the integration over D

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
        self.lambda_factor=None
        self.vel_factor=None
        self.ntot_factor=None

        if self.scheme == '2mom':
            self.x_max = None
            self.x_min = None

    def get_N(self,D):
        try:
#            l = self.lambda_.shape
#            if len(l) > D.ndim:
#                if len(l)  == 1:
#                    D = np.repeat(D[:,None],l[0], axis=1)
#                if len(l)  == 2:
#                    D = (np.repeat(D[:,:,None],l[1], axis=2))
#            print(l,D.shape)
#
#            if D.ndim > len(l) and l[0] == D.shape[0]:
#                lamb = np.repeat(self.lambda_[:,None],D.shape[1],axis=1)
#                plt.plot(lamb)
#            else:
#                lamb = self.lambda_
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
        return self.alpha*D**self.beta

    def get_D_from_V(self,V):
        return (V/self.alpha)**(1./self.beta)

    def integrate_V(self):
        v = (self.vel_factor*self.N0*self.alpha/self.nu*
                 self.lambda_**(-(self.beta+self.mu+1)/self.nu))
        if self.scheme == '2mom':
            n = self.ntot
        else:
            n = self.ntot_factor*self.N0/self.nu*self.lambda_**(-(self.mu+1)/self.nu)
        if np.isscalar(v):
            v = np.array([v])
            n = np.array([n])

        return v, n

#    def integrate_V_weighted(self,D,weights, method='sum'):
#        try:
#            if method not in ['sum','trapz']:
#                print('Invalid integration method')
#                print('Choosing sum instead')
#                method = 'sum'
#
#            V=self.get_V(D)
#            N=self.get_N(D)
#
#            dD = D[1]-D[0]
#
#            if method == 'sum':
#                # Fast version using einsum
#                if len(N.shape) == 1:
#                    V_weighted = np.sum(N*weights*V) * dD
#                    N_weighted = np.sum(N*weights) * dD
#                elif len(N.shape) == 2:
#                    V_weighted = np.einsum('ij,i,i,->j',N,weights,V) * dD
#                    N_weighted = np.einsum('ij,i->j',N,weights) * dD
#                elif len(N.shape) == 3:
#                    V_weighted = np.einsum('ijk,i,i->jk',N,weights,V) * dD
#                    N_weighted = np.einsum('ijk,i->jk',N,weights) * dD
#
#            elif method == 'trapz':
#                # More precise but slow (using trapz)
#                if len(N.shape) == 2:
#                    V=np.repeat(V[:,None],N.shape[1],axis=1)
#                    weights=np.repeat(weights[:,None],N.shape[1],axis=1)
#                elif len(N.shape) == 3:
#                    V=np.repeat(np.repeat(V[:,None,None],N.shape[1],axis=1),N.shape[2],axis=2)
#                    weights=np.repeat(np.repeat(weights[:,None,None],N.shape[1],axis=1),N.shape[2],axis=2)
#
#                V_weighted = trapz(N*weights*V,dx=dD,axis=0)# multiply by integration step
#                N_weighted = trapz(N*weights,dx=dD,axis=0)# multiply by integration step
#
#            return V_weighted, N_weighted
#        except:
#            raise
#            print('Wrong dimensions of input')

    def get_M(self,D):
        return self.a*D**self.b

    def set_psd(self,*args):
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

class Solid(Hydrometeor):
    def get_fractions(self,D):
        # Uses COSMO mass-diameter rule
        f_ice = 6*self.a/(np.pi*constants.RHO_I)*D**(self.b-3)
        f_air = 1-f_ice
        return [f_ice, f_air]

    def get_m_func(self,T,f):
        def func(D, T, f):
            frac = self.get_fractions(D)
            m_ice = dielectric.dielectric_ice(T,f)
            return dielectric.dielectric_mixture(frac, [m_ice, constants.M_AIR])
        return lambda D: func(D, T, f)


###############################################################################

class Rain(Hydrometeor):
    def __init__(self, scheme, sedi=True):

        if scheme not in ['1mom','2mom']:
            scheme = '1mom'

        self.scheme = scheme
        self.sedi = (sedi if self.scheme == '2mom' else False)

        if self.scheme == '2mom':
            self.d_max = constants_2mom.D_MAX_R
            self.d_min = constants_2mom.D_MIN_R
        else:
            self.d_max = constants_1mom.D_MAX_R
            self.d_min = constants_1mom.D_MIN_R

        # Power-law parameters
        self.a = (constants_1mom.A_R if self.scheme == '1mom' else constants_2mom.A_R)
        self.b = (constants_1mom.B_R if self.scheme == '1mom' else constants_2mom.B_R)
        self.alpha = (constants_1mom.ALPHA_R if self.scheme == '1mom' else constants_2mom.ALPHA_R)
        self.beta = (constants_1mom.BETA_R if self.scheme == '1mom' else constants_2mom.BETA_R)

        # PSD parameters
        self.lambda_ = None
        self.N0 = (constants_1mom.N0_R if self.scheme == '1mom' else None)
        self.mu = (constants_1mom.MU_R if self.scheme == '1mom' else constants_2mom.MU_R)
        self.nu = (1 if self.scheme == '1mom' else constants_2mom.NU_R)

        # Scattering parameters
        self.canting_angle_std = 10.

        # Others
        self.lambda_factor = (constants_1mom.LAMBDA_FACTOR_R if self.scheme == '1mom' else constants_2mom.LAMBDA_FACTOR_R)
        self.vel_factor = (constants_1mom.VEL_FACTOR_R if self.scheme == '1mom' else constants_2mom.VEL_FACTOR_R)
        self.ntot_factor=(constants_1mom.NTOT_FACTOR_R if self.scheme == '1mom' else constants_2mom.NTOT_FACTOR_R)
        self.ntot = None

        if self.scheme == '2mom':
            self.x_max = constants_2mom.X_MAX_R
            self.x_min = constants_2mom.X_MIN_R
        else:
            self.vel_factor = constants_1mom.VEL_FACTOR_R

    def get_V(self,D):
        if self.sedi:
            # Specific V-D relation for sedimentation
            # For small drops the velocity might be zero with this model...
            V = np.maximum(0,constants_2mom.C_1-constants_2mom.C_2*np.exp(-constants_2mom.C_3*D))
            return V
        else:
            return super(Rain,self).get_V(D) # Call general Hydrometeor method for

    def get_D_from_V(self,V):
        if self.sedi:
            # Specific V-D relation for sedimentation
            return -1/constants_2mom.C_3 * np.log((constants_2mom.C_1-V) / constants_2mom.C_2)
        else:
            return super(Rain,self).get_D_from_V(V)

    def integrate_V(self,*args):
        if not self.sedi or not self.scheme == '2mom':
            return super(Rain,self).integrate_V()
        else:
            D=args[0]
            dD=D[1]-D[0]
            return np.einsum('ij,i->j',self.get_N(D),self.get_V(D))*dD,self.ntot

    def set_mu(self,QN,QM,QC):
        if self.sedi and self.scheme == '2mom':
            if QC<constants.EPS:
                D_mean=self.a*(QN/QN)**self.b * 1000. # To get in mm (and not m)
                if D_mean <= constants_2mom.D_EQ:
                    mu = 9*np.tanh((constants_2mom.TAU_1*(D_mean-constants_2mom.D_EQ))**2)+1
                else:
                    mu = 9*np.tanh((constants_2mom.TAU_1*(D_mean-constants_2mom.D_EQ))**2)+1

                self.mu = mu

    def set_psd(self,*args):
        if len(args) == 2 and self.scheme == '2mom':
            super(Rain,self).set_psd(*args)
        elif self.scheme == '1mom':
            with np.errstate(divide='ignore'):
                _lambda = np.array((self.lambda_factor/args[0])**(1./(4.+self.mu)))
                _lambda[args[0]==0] = np.nan
                self.lambda_ = _lambda
                self.ntot = self.ntot_factor * self.N0 * self.lambda_**(self.mu - 1)
                if np.isscalar(self.lambda_):
                    self.lambda_ = np.array([self.lambda_])
                if np.isscalar(self.ntot):
                    self.ntot = np.array([self.ntot])
        else:
            print('Invalid call to function, if scheme == ''2mom'', input must be tuple of (QN,QM)')
            print('if scheme == ''1mom'', input must be (QM)')

    def get_aspect_ratio(self,D):
        if np.isscalar(D):
            D = np.array([D])

        ar = np.zeros((len(D),))
        ar[D<0.7] = 1.0
        mid_diam = np.logical_and(D<1.5,D>=0.7)
        ar[mid_diam] = 1.173 - 0.5165*D[mid_diam] + 0.4698*D[mid_diam]**2 - 0.1317*D[mid_diam]**3 - \
                8.5e-3*D[mid_diam]**4
        ar[D>=1.5] = 1.065 - 6.25e-2*D[D>=1.5] - 3.99e-3*D[D>=1.5]**2 + 7.66e-4*D[D>=1.5]**3 - \
                4.095e-5*D[D>=1.5]**4
        # This model tends to diverge for large drops so we threshold it to
        # a reasonable max drop size
        ar[D>=self.d_max] = ar[D<=self.d_max][-1]
        return 1./ar


    def get_m_func(self,T,f):
        return lambda D: dielectric.dielectric_water(T, f)

###############################################################################

class Snow(Solid):
    def __init__(self, scheme):
        if scheme not in ['1mom','2mom']:
            scheme = '1mom'

        self.scheme = scheme

        if self.scheme == '2mom':
            self.d_max = constants_2mom.D_MAX_S
            self.d_min = constants_2mom.D_MIN_S
        else:
            self.d_max = constants_1mom.D_MAX_S
            self.d_min = constants_1mom.D_MIN_S

        # Power-law parameters
        self.a = (constants_1mom.A_S if self.scheme == '1mom' else constants_2mom.A_S)
        self.b = (constants_1mom.B_S if self.scheme == '1mom' else constants_2mom.B_S)
        self.alpha = (constants_1mom.ALPHA_S if self.scheme == '1mom' else constants_2mom.ALPHA_S)
        self.beta = (constants_1mom.BETA_S if self.scheme == '1mom' else constants_2mom.BETA_S)

        # PSD parameters
        self.lambda_ = None
        self.N0 = None
        self.mu = (constants_1mom.MU_S if self.scheme == '1mom' else constants_2mom.MU_S)
        self.nu = (1 if self.scheme == '1mom' else constants_2mom.NU_S)

        # Scattering parameters
        self.canting_angle_std = 20.

        # Others
        self.lambda_factor = (constants_1mom.LAMBDA_FACTOR_S if self.scheme == '1mom' else constants_2mom.LAMBDA_FACTOR_S)
        self.vel_factor = (constants_1mom.VEL_FACTOR_S if self.scheme == '1mom' else constants_2mom.VEL_FACTOR_S)
        self.ntot_factor=(constants_1mom.NTOT_FACTOR_S if self.scheme == '1mom' else constants_2mom.NTOT_FACTOR_S)

        if self.scheme == '2mom':
            self.x_max = constants_2mom.X_MAX_S
            self.x_min = constants_2mom.X_MIN_S


    def set_psd(self,*args):
        if len(args) == 2 and self.scheme == '2mom':
            super(Snow,self).set_psd(*args)

        elif len(args) == 2 and self.scheme == '1mom':
            # For N0 use relation by Field et al. 2005 (QJRMS)
            self.N0 = 13.5*(5.65*10**5*np.exp(-0.107*(args[0]-273.15)))/1000 # mm^-1 m^-3
            with np.errstate(divide='ignore'):
                _lambda = np.array((self.a*self.N0*self.lambda_factor/args[1])**(1./(self.b+1))) # in m-1
                _lambda[args[1] == 0] = np.nan
                self.lambda_ = _lambda
                self.ntot = self.ntot_factor * self.N0 * self.lambda_**(self.mu - 1)
                if np.isscalar(self.lambda_):
                    self.lambda_ = np.array([self.lambda_])
                if np.isscalar(self.ntot):
                    self.ntot = np.array([self.ntot])
        else:
            print('Invalid call to function, if scheme == ''2mom'', input must be tuple of (QN,QM)')
            print('if scheme == ''2mom'', input must be tuple of (T,QM)')

    def get_aspect_ratio(self,D):
        ar=(0.01714*D+0.8467) # Brandes et al 2007 (Colorado snowstorms)
        return 1.0/ar

    def get_aspect_ratio_pdf_masc(self,D):
        alpha = constants.A_AR_ALPHA_AGG*D**constants.B_AR_ALPHA_AGG
        loc = constants.A_AR_LOC_AGG*D**constants.B_AR_LOC_AGG
        scale = constants.A_AR_SCALE_AGG*D**constants.B_AR_SCALE_AGG
        return alpha,loc,scale


    def get_canting_angle_std_masc(self,D):
        cant_std = constants.A_CANT_STD_AGG*D**constants.B_CANT_STD_AGG
        return cant_std

###############################################################################


class Graupel(Solid):
    def __init__(self, scheme):
        if scheme not in ['1mom','2mom']:
            scheme = '1mom'

        self.scheme = scheme

        if self.scheme == '2mom':
            self.d_max = constants_2mom.D_MAX_G
            self.d_min = constants_2mom.D_MIN_G
        else:
            self.d_max = constants_1mom.D_MAX_G
            self.d_min = constants_1mom.D_MIN_G

        # Power-law parameters
        self.a = (constants_1mom.A_G if self.scheme == '1mom' else constants_2mom.A_G)
        self.b = (constants_1mom.B_G if self.scheme == '1mom' else constants_2mom.B_G)
        self.alpha = (constants_1mom.ALPHA_G if self.scheme == '1mom' else constants_2mom.ALPHA_G)
        self.beta = (constants_1mom.BETA_G if self.scheme == '1mom' else constants_2mom.BETA_G)

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
            print('Invalid call to function, if scheme == ''2mom'', input must be tuple of (QN,QM)')
            print('if scheme == ''2mom'', input must be (QM)')

    def get_aspect_ratio(self,D): # Garrett, 2015 http://onlinelibrary.wiley.com/doi/10.1002/2015GL064040/full
        if np.isscalar(D):
            D=np.array([D])
        ar=0.9*np.ones(len(D),)
        return 1.0/ar

    def get_aspect_ratio_pdf_masc(self,D):
        alpha = constants.A_AR_ALPHA_GRAU*D**constants.B_AR_ALPHA_GRAU
        loc = constants.A_AR_LOC_GRAU*D**constants.B_AR_LOC_GRAU
        scale = constants.A_AR_SCALE_GRAU*D**constants.B_AR_SCALE_GRAU
        return alpha,loc,scale

    def get_canting_angle_std_masc(self,D):
        cant_std = constants.A_CANT_STD_GRAU*D**constants.B_CANT_STD_GRAU
        return cant_std

###############################################################################


class Hail(Solid):
    def __init__(self,scheme='2mom'):

        self.scheme = '2mom' # No 1-moment scheme for hail

        if self.scheme == '2mom':
            self.d_max = constants_2mom.D_MAX_H
            self.d_min = constants_2mom.D_MIN_H
        else:
            self.d_max = constants_1mom.D_MAX_H
            self.d_min = constants_1mom.D_MIN_H

        # Power-law parameters
        self.a = constants_2mom.A_H
        self.b = constants_2mom.B_H
        self.alpha = constants_2mom.ALPHA_H
        self.beta = constants_2mom.BETA_H

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
        ar = 0.9 * np.ones((len(D),))
        return 1.0/ar


###############################################################################


class IceParticle(Solid):
    def __init__(self,scheme='2mom'):

        self.scheme = scheme

        if self.scheme == '2mom':
            self.d_max = constants_2mom.D_MAX_I
            self.d_min = constants_2mom.D_MIN_I
        else:
            self.d_max = constants_1mom.D_MAX_I
            self.d_min = constants_1mom.D_MIN_I

        # Power-law parameters
        self.a = (constants_1mom.A_I if self.scheme == '1mom' else constants_2mom.A_I)
        self.b = (constants_1mom.B_I if self.scheme == '1mom' else constants_2mom.B_I)
        self.alpha = (constants_1mom.ALPHA_I if self.scheme == '1mom' else constants_2mom.ALPHA_I)
        self.beta = (constants_1mom.BETA_I if self.scheme == '1mom' else constants_2mom.BETA_I)

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
        try:
            if self.scheme == '1mom':
                # Taken from Field et al (2005)
                x = self.lambda_[:,None] * D / 1000.
                return self.N0[:,None] * constants_1mom.PHI_23_I(x)
            else:
                return self.N0[:,None] *D**self.mu*np.exp(-self.lambda_[:,None]*D**self.nu)

        except:
            raise
            print('Wrong dimensions of input')

    def integrate_V(self):
        # Again no analytical solution here
        D = np.linspace(self.d_min, self.d_max, constants.N_BINS_D)
        dD = D[1] - D[0]
        N = self.get_N(D)
        v =  np.sum(N * self.get_V(D)) * dD
        n = np.sum(N) * dD
        if np.isscalar(v):
            v = np.array([v])
            n = np.array([n])
        return v, n

    def get_mom_2(self,T, QM):
        n = 3
        T = T - constants.T0 # Convert to celcius
        a = 5.065339 - 0.062659 * T -3.032362 * n + 0.029469 * T * n \
            - 0.000285 * T**2  + 0.312550 * n**2 + 0.000204 * T**2*n \
            + 0.003199 * T* n**2 - 0.015952 * n**3

        a  = 10**(a)
        b = 0.476221 - 0.015896 * T + 0.165977 * n + 0.007468 * T * n \
            - 0.000141 * T**2 + 0.060366 * n**2 + 0.000079 * T**2 * n \
            + 0.000594 * T * n**2 -0.003577 * n**3

        return (QM/a)**(1/b)


    def set_psd(self,arg1,arg2):
        # Reference is Field et al. (2005)
        # We use the two-moment normalization to get a PSD
        # if one moment scheme, arg1 = T, arg2 = Q(M)I
        # if two moments scheme, arg1 =  QNI, arg2 = Q(M)I


        QM = arg2.astype(np.float64)
        if self.scheme == '1mom':
            T = arg1
            Q2 = self.get_mom_2(T,QM/constants_1mom.B_I)
            N0 = Q2**((self.b + 1)/(self.b - 2)) * QM**((2 + 1)/(2 - self.b))
            N0 /= 10**5 # From m^-1 to mm^-1

            lambda_ = (Q2/QM) ** (1/(self.b - 2))

            # Apply correction factor to match third moment
            D = np.linspace(self.d_min, self.d_max, constants.N_BINS_D)

            x = lambda_[:,None] * D.T / 1000

            N = N0[:,None] * constants_1mom.PHI_23_I(x)

            QM_est = np.nansum(self.a*D**self.b*N, axis = 1) *(D[1]-D[0])

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
        # Taken from Auer and Veal (1970)
        ar = 11.3 * D ** 0.414 * 1000**(-0.414)
        return 1/ar

if __name__=='__main__':


    import time
    plt.close('all')
    from cosmo_pol.lookup.lut import load_all_lut
    luts = load_all_lut('1mom',['S','R','mS'],9.41,'tmatrix_new')
    lut = luts['mS'].value_table[0,-1]
    z = lut[:,0] -lut[:,1] -lut[:,2] +lut[:,3]

    mS = create_hydrometeor('mS','1mom')
    r = create_hydrometeor('R','1mom')
    s = create_hydrometeor('S','1mom')
    D = np.linspace(r.d_min,r.d_max,1024)
#    # Case 1
#    r.set_psd(np.array([0.001]))
##    print(np.nansum(r.get_N(D)[0]*z))
#    plt.figure()
##    plt.plot(D,r.get_N(D)[0])
#    # Case 2
#    f = 0.2
#
#    s.set_psd(np.array([273.15]),np.array([ 0.001 ]))
#
#    mS.set_psd(np.array([273.15]),np.array([0.001]),np.array([0.0]))
#    D = np.linspace(mS.d_min[0],mS.d_max[0],1024)
#    plt.plot(D,mS.get_N(D)[0],D,s.get_N(D)[0])
#    print(np.nansum(s.get_N(D)[0]*z ),np.nansum(mS.get_N(D)[0]*z))

    for f in np.linspace(0.0001,0.9999,100):
        print(f)
        mS.f_wet = f
        f = mS.get_fractions(np.array([2.,3.,4.]))
        mf  = mS.get_m_func(273.15,9.41)
        D = np.linspace(mS.d_min,mS.d_max,1024)
        print(np.max(D))
        for d in D:
            mm = mf(d)
            print(mm)
#    r = create_hydrometeor('I','1mom')
#
#    r.set_psd(np.array([270,269]),np.array([ 0.0001,0.001]))
#    print(r.get_N(D).shape)

#    plt.plot(D,r.get_N(D)[0],'C0')
#    plt.plot(D,mG.get_N(D)[0],'C0--',label='_nolegend_')
#
#    print(np.sum(r.get_N(D)**7)/np.sum(r.get_N(D)))
#    print(np.sum(mG.get_N(D)**7)/np.sum(mG.get_N(D)))
#
#    r.set_psd(np.array([ 0.001]))
#    mG.set_psd(np.array([270]),np.array([ 0.001]),
#              np.array([  1.]))
#
#    plt.plot(D,r.get_N(D)[0],'C1')
#    plt.plot(D,mG.get_N(D)[0],'C1--',label='_nolegend_')
#
#
#    r.set_psd(np.array([ 0.01]))
#    mG.set_psd(np.array([270]),np.array([ 0.01]),
#              np.array([  1.]))
#
#    plt.plot(D,r.get_N(D)[0],'C2')
#    plt.plot(D,mG.get_N(D)[0],'C2--',label='_nolegend_')
#
#    plt.grid()
#
#    plt.xlabel('Diameter [mm]',fontsize = 16)
#    plt.ylabel(r'N part. m$^{-3}$',fontsize=16)
#    plt.title('Rain DSD vs Totally melted Snow PSD')
#    plt.legend(['q = 1E-4 kg/m3','q = 1E-3 kg/m3','q = 1E-2 kg/m3'])
#    plt.savefig('rain_vs_snow.pdf',bbox_inches = 'tight',dpi=200)
#    n,v = r.integrate_V()
#    N = r.get_N(np.linspace(1,4,100))
#    r = create_hydrometeor('R','1mom')
#    ms.f_wet = 0.0

#    fu = ms.get_aspect_ratio_pdf_masc(D)

#    from cosmo_pol.lookup.lut import Lookup_table, load_all_lut
#    lut = load_all_lut('1mom', ['R','mS','S','mG'], 9.41, 'tmatrix_new')
#
#    r.set_psd(np.array([0.001]),np.array([0.99]))
#    plt.plot(np.linspace(0.1,10,1000),np.squeeze(r.get_N(np.linspace(0.1,10,1000))))
#
#    r2 = create_hydrometeor('R','1mom')
#    r2.set_psd(np.array([0.001]))
#    plt.plot(np.linspace(0.1,10,1000),np.squeeze(r2.get_N(np.linspace(0.1,10,1000))))
#
#    sz_r = lut['S'].lookup_line(e=0,t=278)
#    sz_mg = lut['mG'].lookup_line(e=0,wc=0.99)
#

#    D = np.linspace(r.d_min,r.d_max,1000)
#    N = r.get_N(D)
#    print((np.sum(N*r.get_V(D)))/np.sum(N))
#    D = vlinspace(ms.d_min,ms.d_max,1024)
#    v,n = r.integrate_V(+)
#    print(v/n)
#    import time
#    t0 = time.time()
#    meu = ms.get_D_from_V(np.array([1,3,4]))
#    print(t0-time.time())

#    r.set_psd(np.array((0.001)))
#    fu = ms.integrate_V()
#    D = vlinspace(np.ones((1))*1,np.ones((1))*10,1024)
#    D = np.linspace(ms.d_min,ms.d_max,1024)
#    N = ms.get_N(D)
#    N2 = r.get_N(D)
#    M = ms.get_M(D)
#    V = ms.get_V(D)
#    plt.plot(D,N[0],D,N2)
#    plt.grid()
#    plt.xlabel('Drop size [mm]')
#    plt.ylabel(r'N drops [$m^{-3}$]')
#    plt.legend(['Very wet graupels (f_wet = 0.99)','Rain with exp. DSD'])
#    plt.savefig('wet_graupels_dsd.png',dpi=200)
#    dD = D[1] - D[0]
#    print(np.sum(r.get_N(D)* r.get_M(D))*dD)
#    print(np.sum(ms.get_N(D)* ms.get_M(D))*dD)
    # Try to fit marshall palmer


#    ms.set_psd(np.array([0.01,0.001]),np.array([0.5,0.99]))

#    N = ms.get_N(D)
#    print(ms.integrate_V())
#    print(ms.f_wet,ms.d_max)
#    t0 = time.time()
#    for i in range(100):
#        fu = ms.set_psd(np.array([270]),np.array([0.01]),np.array([0.99]))
#    print(t0 - time.time())
##    a = ms.get_aspect_ratio_pdf_masc(2)
#
#    D = np.linspace(ms.d_min,ms.d_max,1024)
#    a = ms.get_aspect_ratio_pdf_masc(D)
#
#    plt.plot(np.linspace(1,5,100),a(np.linspace(1,5,100)))
#
#    import scipy
#    t0 = time.time()
#    ip = scipy.interpolate.interp1d(ms.equivalent_rain.get_V(D),D)
#
#    t0 = time.time()
#    for i in range(1000):
#        D = ms.get_D_from_V(np.linspace(1.6,3,256))
#    print(t0 - time.time())

#    plt.figure()

#    x = np.linspace(1,5,1000)
#    plt.plot(x,a.pdf(x))
##    print(t0 - time.time())
##    print(ms.get_fractions(3))
#    print(ms.get_m_func(270,9.8)(5))
#    g = create_hydrometeor('G','1mom')
#
#    g.set_psd(np.array([0.001]))
#
#    r = create_hydrometeor('R','1mom')
#
##    print(r.get_m_func(270,9.8)(5))
#    s = create_hydrometeor('S','1mom')
#    s.set_psd(np.array([270]),np.array([0.001]))
#
#    plt.plot(D,ms.get_N(D))
#    D = np.linspace(ms.d_min,ms.d_max,1024)
#
#    import matplotlib.pyplot as plt
#    plt.plot(D,ms.get_N(D))
#    plt.plot(D,g.get_N(D))
#    plt.plot(D,r.get_N(D))
#
#    plt.legend(['S','MG','G','R'])
#
#




#    import pickle
#    import gzip
##    plt.close('all')
###
##    lut1 = pickle.load(gzip.open('/media/wolfensb/Storage/cosmo_pol/lookup/stored_lut/lut_SZ_G_9_41_1mom.pz','rb'))
##    dd2 = lut1.value_table[10,10,:,0]
##
##    lut2 = pickle.load(gzip.open('/media/wolfensb/Storage/cosmo_pol/lookup/stored_lut_m21/lut_SZ_G_9_41_2mom.pz','rb'))
##    dd = lut2.value_table[10,10,:,0]
###
##    plt.figure()
##    plt.plot(dd)
##    plt.hold(True)
##    plt.plot(dd2)
##
##
#    from pytmatrix.tmatrix import Scatterer
#    from pytmatrix.psd import PSDIntegrator, UnnormalizedGammaPSD
#    from pytmatrix import orientation, radar, tmatrix_aux, refractive
#    def drop_ar(D_eq):
#        if D_eq < 0.7:
#            return 1.0;
#        elif D_eq < 1.5:
#            return 1.173 - 0.5165*D_eq + 0.4698*D_eq**2 - 0.1317*D_eq**3 - \
#                8.5e-3*D_eq**4
#        else:
#            return 1.065 - 6.25e-2*D_eq - 3.99e-3*D_eq**2 + 7.66e-4*D_eq**3 - \
#                4.095e-5*D_eq**4
#
#    scatterer = Scatterer(wavelength=tmatrix_aux.wl_C, m=refractive.m_w_10C[tmatrix_aux.wl_C])
#    scatterer.psd_integrator = PSDIntegrator()
#    scatterer.psd_integrator.aspect_ratio_func = lambda D: 1.0/drop_ar(D)
#    scatterer.psd_integrator.D_max = 8.
#    scatterer.psd_integrator.geometries = (tmatrix_aux.geom_horiz_back, tmatrix_aux.geom_horiz_forw)
#    scatterer.or_pdf = orientation.gaussian_pdf(10.0)
#    scatterer.orient = orientation.orient_averaged_fixed
#    scatterer.psd_integrator.init_scatter_table(scatterer)
#
    import matplotlib.pyplot as plt
#    plt.close('all')
#
##
#    plt.figure(figsize=(6,4))
#    D = np.linspace(0.01,5,1000)
#
#    a = create_hydrometeor('I','1mom', sedi=False)
#    a.set_psd(np.array([250,250]),np.array([0.001,0.001]))
#
#    D = np.linspace(0.01,8,1000)
#    fu=(a.get_N(D))
#    plt.plot(D,fu[:,0],D,fu[:,1])
#    a = create_hydrometeor('R','1mom')
#    a.set_psd(np.array([0.0000001]))

#    D = np.linspace(0.05,2,500)
#    plt.plot(D,a.get_N(D))
#    print(np.sum(a.a*D**a.b*a.get_N(D))*(D[1]-D[0]))

#    plt.plot(D,b.get_N(D))

#    a=create_PrecipitatingHydrometeor('I','1mom')
#    a.set_psd(np.array([0.0001]),np.array([260]))
#    print(np.sum(a.get_N(D)*(D[1]-D[0])))
#    D = np.linspace(0.01,5,1000)
#
#    plt.plot(D,a.get_N(D))
#
#    N = a.get_N(D)
#    plt.plot(D,N,linewidth=1.5)
#    plt.title(r'DSD for $Q_{rain}$ = 1 g$\cdot$m$^{-3}$')
#    plt.xlim([0.2,5])
#    plt.grid()
#    plt.xlabel('Diameter [mm]')
#    plt.ylabel(r'N [mm$\cdot$m$^{-3}$]')
#    plt.legend([r'$\mu$ = 0.5',r'$\mu$ = 2'])
#    plt.text(2,475,r'$N(D) = N_0 \cdot D^{\mu} $ exp $\left(-\lambda D\right)$',
#             size=14)
#    plt.savefig('example_mu.pdf',dpi=200,bbox_inches='tight')

#
#    scatterer.psd = UnnormalizedGammaPSD(N0=a.N0, mu=a.mu,Lambda=a.lambda_,D_max=8)
#
#    plt.plot(D,scatterer.psd(D),linewidth=1.5)
##    print(np.log10(radar.refl(scatterer))*10,np.log10(radar.refl(scatterer,False))*10)
#    print(np.log10(radar.refl(scatterer))*10,10*np.log10(radar.Zdr(scatterer)))
#
#
##    print(np.sum(N*(D[1]-D[0])))
#
#    a=create_PrecipitatingHydrometeor('R','2mom')
#    a.set_psd(np.ones(1,)*433,np.ones(1,)*0.001)
#    D = np.linspace(0.2,15,1024)
#    N = a.get_N(D)
##
#    scatterer.psd = UnnormalizedGammaPSD(N0=a.N0+5000, mu=a.mu,Lambda=a.lambda_,D_max=8)
#    plt.plot(D,scatterer.psd(D),linewidth=1.5)
##    print(np.log10(radar.refl(scatterer))*10,np.log10(radar.refl(scatterer,False))*10)
#    print(np.log10(radar.refl(scatterer))*10,10*np.log10(radar.Zdr(scatterer)))
#    print(np.sum(N*D**6*(D[1]-D[0])))
#    print(np.sum(N*(D[1]-D[0])))

#    import pycosmo
#    fi = pycosmo.open_file('/ltedata/COSMO/Validation_operator/case2014040802_TWOMOM/lfsf00182000')
#    N = fi.get_variable('QNR')
#    Q = fi.get_variable('QR')
#
#    plt.figure()
#    D = np.linspace(0.2,15,1024)
#    a=create_hydrometeor('R','2mom')
#    a.set_psd(np.ones(1,)*500,np.ones(1,)*0.001)
#    N = a.get_N(D)
#    plt.plot(D,N,linewidth=1.5)
#
#    a=create_hydrometeor('S','2mom')
#    a.set_psd(np.ones(1,)*500,np.ones(1,)*0.001)
#    N = a.get_N(D)
#    plt.plot(D,N,linewidth=1.5)
#
#    a=create_hydrometeor('G','2mom')
#    a.set_psd(np.ones(1,)*500,np.ones(1,)*0.001)
#    N = a.get_N(D)
#    plt.plot(D,N,linewidth=1.5)
#
#    a=create_hydrometeor('H','2mom')
#    a.set_psd(np.ones(1,)*500,np.ones(1,)*0.001)
#    N = a.get_N(D)
#    plt.plot(D,N,linewidth=1.5)
#
#    a=create_hydrometeor('I','2mom')
#    a.set_psd(np.ones(1,)*300,np.ones(1,)*0.00001)
#    N = a.get_N(D)
#    plt.plot(D,N,linewidth=1.5)
#
#    plt.xlim([0.2,12])
#    plt.grid()
#    plt.xlabel('Diameter [mm]',fontsize=16)
#    plt.ylabel(r'N. of particles / m$^3$',fontsize=16)
#    plt.legend(['Rain','Snow','Graupel','Hail','Ice'],prop={'size': 12})
#    plt.savefig('ex_psd.pdf',dpi=200,bbox_inches = 'tight')
#
##
##    v,n = a.integrate_V()
##    a.set_psd(np.array([10**-3]))
##    print(a.lambda_)
#    D = np.linspace(0.2,8,1024)
#    plt.plot(D,a.get_N(D))
#    plt.hold(True)
#    print(np.trapz(dd*a.get_N(D),D))
#    a=create_PrecipitatingHydrometeor('R','1mom')
#    a.set_psd(np.ones(1,)*0.0025270581)
##    print(np.trapz(a.a*D**(a.b)*a.get_N(D),D))
#    plt.plot(D,a.get_N(D))
##    print(np.trapz(a.a*D**(a.b)*a.get_N(D),D))
#    print(np.trapz(dd*a.get_N(D),D))


    #    frac = a.get_fractions(D)
#    plt.plot(D,frac[0],D,frac[1]    )
#    plt.hold(True)
#
#    a1=create_PrecipitatingHydrometeor('S','1mom')
#    frac1 = a1.get_fractions(D)
#    diel1 = a1.get_m_func(270,9.41)
#    plt.plot(D,frac1[0],D,frac1[1]   )
#    plt.hold(True)
#    a2=create_PrecipitatingHydrometeor('S','2mom')
#    frac2 = a2.get_fractions(D)
#    diel2 = a2.get_m_func(270,9.41)
#    plt.plot(D,frac2[0],D,frac2[1]   )
#    plt.legend(['ice1','air1','ice2','air2'])
#
#    plt.figure()
#    plt.plot(D,a1.a*D**a1.b)
#    plt.plot(D,a2.a*D**a2.b)
##
#    plt.figure()
#    plt.plot(D,diel1(D),D,diel2(D))
#

#
#    from pytmatrix import orientation
#    from pytmatrix.tmatrix import Scatterer
#
#    f=9.41
#    wavelength=constants.C/(f*1E09)*1000 # in mm
#    scatt = Scatterer(radius = 1.0, wavelength = wavelength)
#    scatt.or_pdf = orientation.gaussian_pdf(std=20)
#    scatt.orient = orientation.orient_averaged_fixed
#
#    list_SZ_1=[]
#    list_SZ_2=[]
#
#    elevation = 5
#    m_func1= a1.get_m_func(270,f)
#    m_func2= a2.get_m_func(270,f)
#    geom_back=(90-elevation, 180-(90-elevation), 0., 180, 0.0,0.0)
#    geom_forw=(90-elevation, 90-elevation, 0., 0.0, 0.0,0.0)
#
#
#    for i,d in enumerate(D):
#
#        ar = a1.get_aspect_ratio(d)
#
#        scatt.radius = d/2.
#        scatt.m = m_func1(d)
#        scatt.aspect_ratio = ar
#
#        # Backward scattering (note that we do not need amplitude matrix for backward scattering)
#        scatt.set_geometry(geom_back)
#        Z_back = scatt.get_Z()
#        # Forward scattering (note that we do not need phase matrix for forward scattering)
#        scatt.set_geometry(geom_forw)
#        S_forw=scatt.get_S()
#
#        list_SZ_1.append(Z_back[0,0])
#
#        ar = a2.get_aspect_ratio(d)
#
#        scatt.radius = d/2.
#        scatt.m = m_func2(d)
#        scatt.aspect_ratio = ar
#
#        # Backward scattering (note that we do not need amplitude matrix for backward scattering)
#        scatt.set_geometry(geom_back)
#        Z_back=scatt.get_Z()
#        # Forward scattering (note that we do not need phase matrix for forward scattering)
#        scatt.set_geometry(geom_forw)
#        S_forw=scatt.get_S()
#
#        list_SZ_2.append(Z_back[1,1])

#    a.set_psd(np.array([10**-4]))
#    print(a.lambda_)
#    N2= a.get_N(np.linspace(0.1,6,100))
#
#    a.set_psd(np.array([0.5*(10**-4+10**-3)]))
#    N4= a.get_N(np.linspace(0.1,6,100))
#
#    a.lambda_ = np.array([(1.57084824+2.62033279)/2])
#    N5= a.get_N(np.linspace(0.1,6,100))
#    plt.plot(D,0.5*(N+N2),D,N4,D,N5)
#    QN = 10**(np.linspace(0,8,150))
#    Q= 10**(np.linspace(-8,-2,150))
#    a.set_psd(QN,Q)
#    lambdas = a.lambda_
#    n0s = a.N0
#    N= a.get_N(np.linspace(0.1,6,100))
#
#    uuu = a.integrate_V_weighted(np.linspace(0.1,6,100),np.ones((100,)))
#
#    uuu=uuu[0]/uuu[1]
#    lambdas_2 = np.zeros((len(QN),len(Q)))
#    uuu_2 =  np.zeros((len(QN),len(Q)))
#    n0s_2 = np.zeros((len(QN),len(Q)))
#    for i,qi in enumerate(QN):
#        for j,qj in enumerate(Q):
#            a.set_psd(qi,qj)
#            lambdas_2[i,j]=a.lambda_
#            n0s_2[i,j]=a.N0
#            N= a.get_N(np.linspace(0.1,6,100))
#            fu= a.integrate_V_weighted(np.linspace(0.1,6,100),np.ones((100,)))
#            uuu_2[i,j]=fu[0]/fu[1]
#
##    print a.get_D_from_V(np.array([6]*20))
#    a.set_psd(1000,0.01)
#    D=np.arange(0,10,0.01)
#    N=a.get_N(D)
#    W=np.random.rand(1000,)*0+1
##
##
#    a.set_psd(np.array([0.001]))
##    tic()
#    for i in range(30):
#        v=a.integrate_V_weighted(D,W)
#    toc()
##    ##print v[0]/v[1]


