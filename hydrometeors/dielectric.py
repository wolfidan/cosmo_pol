# -*- coding: utf-8 -*-
"""
Created on Wed Aug 19 14:54:56 2015

@author: wolfensb
"""

import numpy as np

def dielectric_ice(t,f):
   ## From the article of G.Hufford: "A model for the complex permittivity of
   ## ice at frequencies below 1 THz"

   ## Inputs :
   ##   t = the temperature (in K)
   ##   f = the frequency (in GHz)

   ## Outputs :
   ##   m = m' + im'' the complex refractive index
   ## NOTE THAT THERE IS A DISCONTINUITY AT 300 K...
    
    t=min(t,273.15) # Stop at solidification temp
        
    theta=300./t-1.
    alpha=(50.4+62.*theta)*10**-4*np.exp(-22.1*theta)
    beta=(0.502-0.131*theta)/(1.+theta)*10**-4+0.542*10**-6*((1+theta)/(theta+0.0073))**2
    
    Epsilon_real=3.15
    Epsilon_imag=alpha/f+beta*f
    
    Epsilon = complex(Epsilon_real,Epsilon_imag)
    
    m=np.sqrt(Epsilon)
    return m
    
    
def dielectric_water(t,f):
   ## From the article of H.Liebe: "A model for the complex permittivity of
   ## water at frequencies below 1 THz"

   ## More general than the Debye equation this formula is valid for
   ## frequencies up to 1THz.
   ## Replaces the work of P.Ray, Applied Optics Vol.8,p.1836-1844, 1972

   ## Inputs :
   ##   t = the temperature (in K)
   ##   f = the frequency (in GHz)

   ## Outputs :
   ##   m = m' + im'' the complex refractive index

   ## Source : M.Schleiss (May.2008)
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
    Epsilon_imag = (term1/term2)*(f/Gamma_1) + (term4/term3)*(f/Gamma_2)

    Epsilon = complex(Epsilon_real,Epsilon_imag)

    m=np.sqrt(Epsilon)
    return m
    
def dielectric_mixture(fracs, m):
   ## From the article of Bohren and Battan : "Radar Backscattering by Inhomogeneous
   # Precipitation Particles

   ## Inputs :
   ##   fracs = array containing the volume fraction of the inclusions
   ##   m = array containing the dielectric constants of the inclusions

   ## Outputs :
   ##   m_mix = m' + im'' the complex refractive index of the mixture

   ## Source : D.Wolfensberger (November 2015)

    if type(fracs)!=np.ndarray: # Convert to numpy array
        fracs=np.asarray(fracs)

    tot_frac=np.sum(fracs)
    if tot_frac!=1:
        fracs=fracs/tot_frac

    fracs_stack=fracs[0]
    m_mix=m[0]
    idx=0
    while idx<len(fracs)-1:
        fracs_sub=np.asarray([fracs_stack,fracs[idx+1]])
        m_sub=np.asarray([m_mix, m[idx+1]])

        tot_frac=np.sum(fracs_sub)
        if tot_frac!=1:
            fracs_sub=fracs_sub/tot_frac
        
        beta=2*m_sub[0]/(m_sub[1]-m_sub[0])*(m_sub[1]/(m_sub[1]-m_sub[0])*np.log(m_sub[1]/m_sub[0])-1)
        m_mix=((1-fracs_sub[1])*m_sub[0]+fracs_sub[1]*beta)/(1-fracs_sub[1]+fracs_sub[1]*beta)
        idx+=1
        fracs_stack+=fracs[1]
        
    return m_mix
				