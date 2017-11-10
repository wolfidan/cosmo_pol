'''

  NAME:
     kdp_estimation_vulpiani
 
  PURPOSE:
     Calculate kdp from the method suggested by Vulpiani et al, JAOT 2012
     The method calculate Kdp by iterative fixed-window derivation.
     Phidp is iteratively reconstructe and differentiated. This method
     treats the differential phase shift upon backscattering as noise
     and interpolates any gap (NaN) found in the input Psidp.
     Note that this method is not convergent and when n_iter--> infinity
     then Kdp collapses to a costant value over the profile


 
  AUTHOR(S):
     Jacopo Grazioli (16.07.2015) -- from previous codes of J. Grazioli
     Daniel Wolfensberger (August 2016) -- rewritten in python
 
  CALLING SEQUENCE:
     ok = kdp_estimation_vulpiani(psidpvolume,windsize=windsize,band=band,n_iter=n_iter,nonegative=nonegative)
 
  INPUT PARAMETERS:
     psidp_in    : one-dimensional vector of length -nrg-
                  containining the input psidp [degrees]
    windsize: size in [m] of the range derivative  window.
     The larger this window  the smoother Kdp will be. Default is 750 m
 
    band: radar frequency band string. Accepted "x", "c", "s" (capital
       or not). IT is used to set default boundaries for expected
       values of Kdp
   
    n_iter: number of iterationsof the method. Default is 10.
 
    nonegative: pure keyword. If set then Kdp is estimated GE 0
    (it loses generality but it is better for rainfall estimation)
 
    interpolate: pure keyword. If set all the nans are interpolated.
                 The advantage is that less data are lost (the
                 iterations in fact are "eating the edges") but some
                 non-linear errors may be introduced
 
  OUTPUT KEYWORDS:
 
  COMMON BLOCKS:
 
  OUTPUT:
    It returns the volume (structure) of Kdp in degrees per km with the same fields as
    psidpvolume
 
  DEPENDENCIES:
 
  MODIFICATION HISTORY:
     16.7.2015:  Creation J. Grazioli (Meteo Swiss)
     August 2016: Conversion to python, D. Wolfensberger

'''

import numpy as np
import numpy.ma as ma
from scipy.interpolate import interp1d


def kdp_estimation_vulpiani(psidp_in,dr, windsize,band = 'X', n_iter = 10, interpolate = False):
    if not np.isfinite(psidp_in).any(): # Check if psidp has at least one finite value
        return psidp_in,psidp_in, psidp_in # Return the NaNs...
    
    l = windsize
    
    #Thresholds in kdp calculation
    if band == 'X':   
        th1 = -2.
        th2 = 25.  
    elif band == 'C':   
        th1 = -0.5
        th2 = 15.
    elif band == 'S':   
        th1 = -0.5
        th2 = 10.
    else:   
        print('Unexpected value set for the band keyword ')
        print(band)
        return None
    
    psidp = psidp_in
    nn = len(psidp_in)
   
    #Get information of valid and non valid points in psidp the new psidp
    nonan = np.where(np.isfinite(psidp))[0]
    nan =  np.where(np.isnan(psidp))[0]
    if interpolate:
        ranged = np.arange(0,nn)
        psidp_interp = psidp
        # interpolate
        if len(nan):
            interp = interp1d(ranged[nonan],psidp[nonan],kind='zero')
            psidp_interp[nan] = interp(ranged[nan])
            
        psidp = psidp_interp
    
    kdp_calc = np.zeros([nn]) * np.nan

    #Loop over range profile and iteration
    for ii in range(0, n_iter):
        for ir in range(0, nn ):
            # In the core of the profile
            if ir >= l / 2 and ir <= nn - 1 - l / 2:   
                kdp_calc[ir] = (psidp[ir + l / 2] - psidp[ir - l / 2]) / (2. * l * dr)
            # In the beginnning of the profile: use all the available data on the RHS
            if ir < l / 2:   
                dummy = l / 2 - ir
                kdp_calc[ir] = (psidp[ir + l / 2 + dummy] - psidp[ir - l / 2 + dummy]) / (2. * l * dr)
            # In the end of the profile: use the  LHS available data
            if ir > nn - 1 - l / 2:   
                dummy = nn - 1 - ir
                kdp_calc[ir] = (psidp[nn - 1] - psidp[ir - l + dummy]) / (2. * l * dr)
                
            #apply thresholds
            if kdp_calc[ir] <= th1:   
                kdp_calc[ir] = th1
            if kdp_calc[ir] >= th2:   
                kdp_calc[ir] = th2

        #Erase first and last gate
        kdp_calc[0] = np.nan
        kdp_calc[nn - 1] = np.nan
    
    kdp_calc = ma.masked_array(kdp_calc, mask = np.isnan(kdp_calc))
    #Reconstruct Phidp from Kdp
    phidp_rec = np.cumsum(kdp_calc) * 2. * dr

    #Censor Kdp where Psidp was not defined
    if len(nan):   
        kdp_calc[nan] = np.nan
        
    #Fill the output
      
    return kdp_calc, phidp_rec