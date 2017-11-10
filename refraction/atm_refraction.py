# -*- coding: utf-8 -*-
"""
Created on Mon Aug  3 15:35:21 2015

@author: wolfensb
"""
import numpy as np
import pycosmo as pc

from scipy.interpolate import interp1d
from scipy.integrate import odeint
from  cosmo_pol.utilities import cfg
from cosmo_pol.constants import constants

rad2deg=180.0/np.pi
deg2rad=1/rad2deg

def get_earth_radius(latitude):
    a=6378.1370*1000
    b=6356.7523*1000
    return np.sqrt(((a**2*np.cos(latitude))**2+(b**2*np.sin(latitude))**2)/((a*np.cos(latitude))**2+(b*np.sin(latitude))**2))


def get_radar_refraction(range_vec, elevation_angle, coords_radar, refraction_method, N=0):
    # Method can be '4/3', 'ODE_s', or 'ODE_c'
    # '4/3': Standard 4/3 refraction model (offline, very fast)
    # ODE_s: differential equation of atmospheric refraction assuming horizontal homogeneity
    # ODE_s: differential equation of atmospheric refraction with no assumption

    if refraction_method==1:
        S,H,E=ref_4_3(range_vec, elevation_angle, coords_radar)
    elif refraction_method==2:
        S,H,E=ref_ODE_s(range_vec, elevation_angle, coords_radar, N)

    
    return S,H, E   
    
def deriv_z(z,r,n_h_spline, dn_dh_spline, RE):
    # Computes the derivatives (RHS) of the system of ODEs
    h,u=z
    n=n_h_spline(h)
    dn_dh=dn_dh_spline(h)
    return [u, -u**2*((1./n)*dn_dh+1./(RE+h))+((1./n)*dn_dh+1./(RE+h))]
    
def ref_ODE_s(range_vec, elevation_angle, coords_radar, N):
    
    # Get info about COSMO coordinate system
    proj_COSMO=N.attributes['proj_info']
    coords_rad_in_COSMO=pc.WGS_to_COSMO(coords_radar,[proj_COSMO['Latitude_of_southern_pole'],proj_COSMO['Longitude_of_southern_pole']])
    llc_COSMO=(float(proj_COSMO['Lo1']), float(proj_COSMO['La1']))
    res_COSMO=N.attributes['resolution']
    
    # Get position of radar in COSMO coordinates
    pos_radar_bin=[(coords_rad_in_COSMO[0]-llc_COSMO[1])/res_COSMO[1], (coords_rad_in_COSMO[1]-llc_COSMO[0])/res_COSMO[0]] # Note that for lat and lon we stay with indexes but for the vertical we have real altitude 
    
    # Get refractive index profile from refractivity estimated from COSMO variables
    n_vert_profile=1+(N.data[:,int(np.round(pos_radar_bin[0])),
                             int(np.round(pos_radar_bin[0]))])*1E-6
    # Get corresponding altitudes
    h = N.attributes['z-levels'][:,int(np.round(pos_radar_bin[0])),
                                int(np.round(pos_radar_bin[0]))] 
    
    # Get earth radius at radar latitude
    RE=get_earth_radius(coords_radar[0])
    
    if cfg.CONFIG['radar']['type']  == 'ground':
        # Invert to get from ground to top of model domain
        h=h[::-1]
        n_vert_profile=n_vert_profile[::-1] # Refractivity
        
    # Create piecewise linear interpolation for n as a function of height
    n_h_spline=piecewise_linear(h, n_vert_profile)
    dn_dh_spline=piecewise_linear(h[0:-1],np.diff(n_vert_profile)/np.diff(h))
    
    z_0 = [coords_radar[2],np.sin(np.deg2rad(elevation_angle))]
    # Solve second-order ODE
    Z = odeint(deriv_z,z_0,range_vec,args=(n_h_spline,dn_dh_spline,RE))
    H = Z[:,0] # Heights above ground
    E = np.arcsin(Z[:,1]) # Elevations
    S = np.zeros(H.shape) # Arc distances
    dR = range_vec[1]-range_vec[0]
    S[0] = 0
    for i in range(1,len(S)): # Solve for arc distances
        S[i]=S[i-1]+RE*np.arcsin((np.cos(E[i-1])*dR)/(RE+H[i]))
        
    return S.astype('float32'), H.astype('float32'), E.astype('float32')*rad2deg
    
def ref_ODE_f(range_vec, elevation_angle, coords_radar, N_field):    
    #TODO
    return
    
def piecewise_linear(x,y):
    interpolator=interp1d(x,y)
    xs = interpolator.x
    ys = interpolator.y

    def pointwise(x):
        if x < xs[0]:
            return ys[0]+(x-xs[0])*(ys[1]-ys[0])/(xs[1]-xs[0])
        elif x > xs[-1]:
            return ys[-1]+(x-xs[-1])*(ys[-1]-ys[-2])/(xs[-1]-xs[-2])
        else:
            return interpolator(x)

    def ufunclike(xs):
        if np.isscalar(xs):
            xs=[xs]
        return np.array([pointwise(xi) for xi in xs])
        
    return ufunclike
    
def ref_4_3(range_vec, elevation_angle, coords_radar):
    elevation_angle=elevation_angle*np.pi/180.
    ke=4./3.
    altitude_radar=coords_radar[2]
    latitude_radar=coords_radar[1]
    # elevation_angle must be in radians!
    # Compute earth radius at radar latitude 
    EarthRadius=get_earth_radius(latitude_radar)
    # Compute height over radar of every range_bin        
    H=np.sqrt(range_vec**2 + (ke*EarthRadius)**2+2*range_vec*ke*EarthRadius*np.sin(elevation_angle))-ke*EarthRadius+altitude_radar
    # Compute arc distance of every range bin
    S=ke*EarthRadius*np.arcsin((range_vec*np.cos(elevation_angle))/(ke*EarthRadius+H))
    E=elevation_angle+np.arctan(range_vec*np.cos(elevation_angle)/(range_vec*np.sin(elevation_angle)+ke*EarthRadius+altitude_radar))
    return S.astype('float32'),H.astype('float32'), E.astype('float32')*rad2deg
    
def get_GPM_refraction(elevation):
    latitude = cfg.CONFIG['radar']['coords'][0]
    altitude_radar = cfg.CONFIG['radar']['coords'][2]
    max_range = cfg.CONFIG['radar']['range']
    radial_resolution = cfg.CONFIG['radar']['radial_resolution']
    
    elev_rad=elevation*deg2rad
    ke=1
    maxHeightCOSMO=constants.MAX_HEIGHT_COSMO
    RE=get_earth_radius(latitude)
    # Compute maximum range to target (using cosinus law in the triangle earth center-radar-target)
    range_vec=np.arange(radial_resolution/2.,max_range,radial_resolution)

    H=-(np.sqrt(range_vec**2 + (ke*RE)**2+2*range_vec*ke*RE*np.sin(elev_rad))-ke*RE)+altitude_radar
    
    S=ke*RE*np.arcsin((range_vec*np.cos(elev_rad))/(ke*RE+H))
    E=elevation-np.arctan(range_vec*np.cos(elev_rad)/(range_vec*np.sin(elev_rad)+ke*RE+altitude_radar))*rad2deg
#
    in_lower_atm=[H<maxHeightCOSMO]

    H=H[in_lower_atm]
    S=S[in_lower_atm]
    E=E[in_lower_atm]

    return S.astype('float32'),H.astype('float32'), E.astype('float32')
    
if __name__=='__main__':
    
    import matplotlib.pyplot as plt
    plt.close('all')
    elev=0.1
    coords_radar=[46.42562,6.0999,200]
    RE=get_earth_radius(coords_radar[0])
    range_vec=np.arange(0,300000,5000)
    
    S1,H1,E1=ref_4_3(range_vec,elev,coords_radar)
    
    # Test case of temperature inversion
    h=np.arange(0,10000,10)
    N=h*0.
    N[0:35]=400-0.25*h[0:35]
    N[35:]=N[34]-0.04*(h[35:]-h[34])
    
    n43=1+(360-(1./(4.*RE))*10**(6)*h)*10**(-6)

    n_vert_profile=1+N*10**(-6) # Refractive index
#    n_vert_profile=n43
    
    n_h_spline=piecewise_linear(h, n_vert_profile)
    dn_dh_spline=piecewise_linear(h[0:-1],np.diff(n_vert_profile)/np.diff(h))
    z_0=[coords_radar[2],np.sin(elev/180.*np.pi)]

    # Solve second-order ODE
    Z=odeint(deriv_z,z_0,range_vec,args=(n_h_spline,dn_dh_spline,RE))
    H=Z[:,0] # Heights above ground
    E=np.arcsin(Z[:,1]) # Elevations
    S=np.zeros(H.shape) # Arc distances
    dR=range_vec[1]-range_vec[0]
    S[0]=0
    for i in range(1,len(S)): # Solve for arc distances
        S[i]=S[i-1]+RE*np.arcsin((np.cos(E[i-1])*dR)/(RE+H[i]))
        
    plt.figure()
    plt.plot(N,h,'k',400-(1./(4.*RE))*10**(6)*h,h,'--k', linewidth=2)
    plt.xlabel('Refractivity N')
    plt.ylabel('Height above surface [m]')
    plt.legend(['Surface duct profile','4/3 profile'])
    plt.xlim([0,400])
    plt.savefig('ex_profile_refractivity.pdf',dpi=200,bbox_inches='tight')
    
    
    elev=1.1
    coords_radar=[46.42562,6.0999,200]
    RE=get_earth_radius(coords_radar[0])
    range_vec=np.arange(0,300000,5000)
    
    S2,H2,E2=ref_4_3(range_vec,elev,coords_radar)
    
    # Test case of temperature inversion
    h=np.arange(0,10000,10)
    N=h*0.
    N[0:35]=400-0.25*h[0:35]
    N[35:]=N[34]-0.04*(h[35:]-h[34])
    
    n43=1+(360-(1./(4.*RE))*10**(6)*h)*10**(-6)

    n_vert_profile=1+N*10**(-6) # Refractive index
#    n_vert_profile=n43
    
    n_h_spline=piecewise_linear(h, n_vert_profile)
    dn_dh_spline=piecewise_linear(h[0:-1],np.diff(n_vert_profile)/np.diff(h))
    z_0=[coords_radar[2],np.sin(elev/180.*np.pi)]

    # Solve second-order ODE
    Z=odeint(deriv_z,z_0,range_vec,args=(n_h_spline,dn_dh_spline,RE))
    H3=Z[:,0] # Heights above ground
    E3=np.arcsin(Z[:,1]) # Elevations
    S3=np.zeros(H.shape) # Arc distances
    dR=range_vec[1]-range_vec[0]
    S3[0]=0
    for i in range(1,len(S)): # Solve for arc distances
        S3[i]=S3[i-1]+RE*np.arcsin((np.cos(E3[i-1])*dR)/(RE+H[i]))

    
#    

    plt.figure()
    plt.plot(S1,H1,'^',S,H,'b',S2,H2,'g^',S3,H3,'g',linewidth=2)
#    plt.legend(['4/3 Earth model','ODE model'],loc='best')
    plt.xlabel('Arc distance [m]')
    plt.ylabel('Height above surface [m]')
    plt.ylim([0,1600])
    plt.xlim([0,200000])
    plt.grid()
    plt.savefig('ex_refraction.pdf',dpi=200,bbox_inches='tight')
#    plt.figure()
#    
#    plt.plot(h[0:-1],dn_dh_spline(h[0:-1]))
#
#    plt.figure()
#    plt.plot(h,n_vert_profile,h,n43)

    
#    import pycosmo as pc
#    coords_rad_in_COSMO = pc.WGS_to_COSMO([46.8,6.94],[-43,10])
#    llc_COSMO=[-2.1,-0.98]
#    res_COSMO = [0.02,0.02]
#    pos_radar_bin=[(coords_rad_in_COSMO[0]-llc_COSMO[0])/res_COSMO[0], (coords_rad_in_COSMO[1]-llc_COSMO[1])/res_COSMO[1]]
#    
#    range_vec=np.arange(0,100000,500)
#    elevation_angle=5
#    coords_radar=[46.844,6.856,500]
#    a,b,c=ref_4_3(range_vec, elevation_angle, coords_radar)