import numpy as np
import numpy.ma as ma
from scipy.signal import medfilt
from functools import partial
from kdp_estimation_vulpiani import kdp_estimation_vulpiani
from kdp_estimation_kalman import kdp_estimation_kalman_v11

RAD2DEG = 180./np.pi
DEG2RAD = np.pi/180.

def estimate_kdp_kalman(psidp,dr, band = 'C', rcov = 0, pcov = 0, parallel = True):
    '''
  NAME:
     estimate_kdp_kalman
 
  PURPOSE:
      Estimates Kdp with the Kalman filter method for a 2D array of psidp measurements with the first dimension 
      being the distance from radar and the second dimension being the angles (azimuths for PPI, elev for RHI).
      The input psidp is assumed to be pre-filtered (for ex. with the filter_psidp function)
 
  AUTHOR(S):
     Daniel Wolfensberger (August 2016)
 
  CALLING SEQUENCE:
     ok = estimate_kdp_vulpiani(psidp,dr, band = 'C', rcov = 0, pcov = 0, parallel = True)
 
  INPUT PARAMETERS:
    psidp_in    :  one-dimensional vector of length -nrg-
                   containining the input psidp [degrees]
    dr:            scalar [km], range resolution
    band: radar frequency band string. Accepted "X", "C", "S" (capital
       or not). IT is used to compute Rcov and Pcov when not provided
       
    Rcov, Pcov:  error covariance matrix of state transition
                 (Pcov), and of measurements (Rcov).
                 If not set, or if set to objects with less
                 than 2 elements, a default parametrization
                 is used, valid for X-band and 75m gate resolution.
                 Pcov is a 4x4 and Rcov a 3x3 matrix
    
    parallel :   flag to enable parallel computation (one core for every radar beam)

  OUTPUT KEYWORDS:
 
  COMMON BLOCKS:
 
  OUTPUT:
    It returns a filtered version of psidp in the form of a masked array
 
  DEPENDENCIES:
  MODIFICATION HISTORY:
     August 2016: Creation D. Wolfensberger

    '''

    if parallel:
        import multiprocessing as mp
        pool = mp.Pool(processes = mp.cpu_count(),maxtasksperchild=1)
        
    masked = psidp.mask
    
    pool = mp.Pool(processes = mp.cpu_count(),maxtasksperchild=1)
    func = partial(kdp_estimation_kalman_v11,dr = dr, band = band, rcov = rcov, pcov = pcov)
    all_psidp = list(psidp)
    
    if parallel:
        list_est = pool.map(func,all_psidp)
    else:
        list_est = map(func,all_psidp)    
        
    kdp = np.zeros(psidp.shape)*np.nan
    kdp_stdev = np.zeros(psidp.shape)*np.nan
    phidp_rec = np.zeros(psidp.shape)*np.nan

    for i,l in enumerate(list_est):
        kdp[i,0:len(l[0])] = l[0]
        kdp_stdev[i,0:len(l[1])] = l[1]
        phidp_rec[i,0:len(l[2])] = l[2]
    
    kdp = ma.asarray(kdp)
    kdp.mask = masked
    kdp_stdev = ma.asarray(kdp_stdev)
    kdp_stdev.mask = masked
    phidp_rec = ma.asarray(phidp_rec)
    phidp_rec.mask = masked

    if parallel:
        pool.close()
    
    return kdp, kdp_stdev, phidp_rec

    
def estimate_kdp_vulpiani(psidp,dr, windsize = 7,  band = 'C', n_iter = 10, interpolate = False, parallel = True):
    '''
  NAME:
     estimate_kdp_vulpiani
 
  PURPOSE:
      Estimates Kdp with the Vulpiani method for a 2D array of psidp measurements with the first dimension 
      being the distance from radar and the second dimension being the angles (azimuths for PPI, elev for RHI).
      The input psidp is assumed to be pre-filtered (for ex. with the filter_psidp function)
 
  AUTHOR(S):
     Daniel Wolfensberger (August 2016)
 
  CALLING SEQUENCE:
     ok = estimate_kdp_vulpiani(psidp,dr,windsize = 7,  band = 'C', interpolate = False, parallel = True)
 
  INPUT PARAMETERS:
    psidp    : two dimensional numpy array of previously filtered psidp measurements
    
    dr       : radial resolution of the radar (in KM)
    
    windsize: size in # of gates of the range derivative  window.
    
    The larger this window  the smoother Kdp will be. Default is 5 gates
    
    band: radar frequency band string. Accepted "X", "C", "S" (capital
       or not). IT is used to set default boundaries for expected
       values of Kdp
   
    n_iter: number of iterations of the method. Default is 10.
    
    interpolate: pure keyword. If set all the nans are interpolated.
                 The advantage is that less data are lost (the
                 iterations in fact are "eating the edges") but some
                 non-linear errors may be introduced
    parallel :   flag to enable parallel computation (one core for every radar beam)
       
    
  OUTPUT KEYWORDS:
 
  COMMON BLOCKS:
 
  OUTPUT:
    It returns a filtered version of psidp in the form of a masked array
 
  DEPENDENCIES:
  MODIFICATION HISTORY:
     August 2016: Creation D. Wolfensberger

    '''
    
    if parallel:
        import multiprocessing as mp
        pool = mp.Pool(processes = mp.cpu_count(),maxtasksperchild=1)
        
    masked = psidp.mask
    func=partial(kdp_estimation_vulpiani,dr = dr, band = band, windsize = windsize, n_iter = n_iter, interpolate = interpolate)
    all_psidp = list(psidp)

    if parallel:
        list_est = pool.map(func,all_psidp)
    else:
        list_est = map(func,all_psidp)    
        
    kdp = np.zeros(psidp.shape)*np.nan
    phidp_rec = np.zeros(psidp.shape)*np.nan

    for i,l in enumerate(list_est):
        kdp[i,0:len(l[0])] = l[0]
        phidp_rec[i,0:len(l[1])] = l[1]
    
    kdp = ma.asarray(kdp)
    kdp.mask = masked
    phidp_rec = ma.asarray(phidp_rec)
    phidp_rec.mask = masked
    

    if parallel:
        pool.close()
    
        
    return kdp, phidp_rec
    
def filter_psidp(psidp, rhohv, minsize = 5, thresh_rhohv=0.65, max_discont = 90):
    '''

  NAME:
     filter_psidp
 
  PURPOSE:
     Filter psidp to remove spurious data in three steps:
         1. Censor it with Rhohv
         2. Unravel angles when strong discontinuities are detected
         3. Remove very short sequences
 
  AUTHOR(S):
     Daniel Wolfensberger (August 2016)
 
  CALLING SEQUENCE:
     ok = filter_psidp(psidp,rhohv=rhohv, minsize = 5, thresh_rhohv=0.65, max_discont = 90)
 
  INPUT PARAMETERS:
    psidp    : two (or one) dimensional numpy array of psidp measurements
    rhohv    : two (or one) dimensional numpy array of rhohv measurements, must be of the same size as psidp 
    minsize  : minimal len (in radar gates) of sequences of valid data to be accepted
    thresh_rhohv : censoring threshold in rhohv (gates with rhohv < thresh_rhohv) will be rejected
    max_discont : Maximum discontinuity between values, default is 90 deg 
  OUTPUT KEYWORDS:
 
  COMMON BLOCKS:
 
  OUTPUT:
    It returns a filtered version of psidp in the form of a masked array
 
  DEPENDENCIES:
  MODIFICATION HISTORY:
     August 2016: Creation D. Wolfensberger

    '''
    # Filter with RHOHV
    psidp[np.isnan(rhohv)] = np.nan
    psidp[rhohv<thresh_rhohv] = np.nan
    
    # Remove short sequences and unwrap    
    psidp_filt = np.nan * psidp    
    for i,psi_row in enumerate(psidp):
        idx = np.where(np.isfinite(psi_row))[0]
        if len(idx):
            psi_row = psi_row[0:idx[-1]]
            
            psi_row = RAD2DEG*np.unwrap(np.nan_to_num(psi_row)*DEG2RAD,max_discont*DEG2RAD)
            # To be sure to always have a left and right neighbour, we need to pad
            # signal with NaN
            psi_row = np.pad(psi_row,(1,1),'constant',constant_values=(np.nan,))
            idx = np.where(np.isfinite(psi_row))[0]
            nan_left = idx[np.where(np.isnan(psi_row[idx-1]))[0]]
            nan_right = idx[np.where(np.isnan(psi_row[idx+1]))[0]]
            
            len_sub = nan_right-nan_left
            
            for j,l in enumerate(len_sub):
                if l < minsize:
                    psi_row[nan_left[j]:nan_right[j]+1]*=np.nan
                    
            # median filter
            psi_row = medfilt(psi_row,11)
            psidp_filt[i,0:len(psi_row[1:-1])] = psi_row[1:-1]
            
    psidp_filt = ma.masked_array(psidp_filt,mask = np.isnan(psidp_filt))

    return psidp_filt

