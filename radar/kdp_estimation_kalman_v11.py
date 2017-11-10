import numpy as np
import numpy.ma as ma
from scipy.interpolate import interp1d
from scipy.signal import medfilt
import matplotlib.pyplot as plt
import multiprocessing as mp
from functools import partial

from kdp_estimation_backward_fixed import kdp_estimation_backward_fixed
from kdp_estimation_forward_fixed import kdp_estimation_forward_fixed


'''
 NAME:
    kdp_estimation_kalman_v11

 PURPOSE:
    Processing one profile of Psidp and estimating Kdp and Phidp
    with the KFE algorithm described in Schneebeli et al, 2014
    IEEE_TGRS.
    NOTE: Psidp should be unwrapped and censored BEFORE this routine

 AUTHOR(S):
    Marc Schneebeli: original code

    Jacopo Grazioli: current version

    Daniel Wolfensberger: Python version
    
 CALLING SEQUENCE:
  Kdp_estimation_Kalman_V11,Psidp_in,$
                              dr,$
                              Kdp_filter_out,$
                              phidp_filter_out,$
                              Kdp_std,$
                              Rcov=Rcov,$
                              Pcov=Pcov

 INPUT PARAMETERS:
    psidp_in    :  one-dimensional vector of length -nrg-
                   containining the input psidp [degrees]
    dr:            scalar [km], range resolution


 OUTPUT
    kdp_filter_out: vector, same length of psidp_in
                    containing the estimated kdp profile
    phidp_filter_out: estimated phidp (smooth psidp)

 INPUT KEYWORDS:
    Rcov, Pcov:  error covariance matrix of state transition
                 (Pcov), and of measurements (Rcov).
                 If not set, or if set to objects with less
                 than 2 elements, a default parametrization
                 is used, valid for X-band and 75m gate resolution.
                 Pcov is a 4x4 and Rcov a 3x3 matrix

 OUTPUT KEYWORDS:


 COMMON BLOCKS:
   ;
 DEPENDENCIES:
   kdp_estimation_forward_fixed
   kdp_Estimation_backward_fixed

 MODIFICATION HISTORY:
   2012/2013: Schneebeli, creation
   July 2015: re-implementation J. Grazioli
    speed optimization, vectorization, set the defaults and group
    the parameters to allow future flexibility
    but the global
    structure is kept, to be consistent with pre-existing
    routines
   August 2016: Rewritten in Python, D.Wolfensberger

SCALERS = [0.1,10**(-0.8),10**(-0.6), 10**(-0.4), 10**(-0.2), 1,
           10**(-0.2), 10**(-0.4), 10**(-0.6), 10**(-0.8),1, 10]

'''
def estimate_kdp_kalman(psidp,dr, band = 'C', rcov = 0, pcov = 0):
    masked = psidp.mask
    
    pool = mp.Pool(processes = mp.cpu_count(),maxtasksperchild=1)
    func = partial(kdp_estimation_kalman_v11,dr = dr, band = band, rcov = rcov, pcov = pcov)
    all_psidp = list(psidp)
    list_est = pool.map(func,all_psidp)
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

    pool.close()
    
    return kdp, kdp_stdev, phidp_rec

    
def estimate_kdp_vulpiani(psidp,dr,windsize = 7,  band = 'C', interpolate = False):
    masked = psidp.mask
    
    pool = mp.Pool(processes = mp.cpu_count(),maxtasksperchild=1)
    func=partial(kdp_estimation_vulpiani,dr = dr, band = band, windsize = windsize, interpolate = interpolate)
    all_psidp = list(psidp)
    list_est = pool.map(func,all_psidp)
    kdp = np.zeros(psidp.shape)*np.nan
    phidp_rec = np.zeros(psidp.shape)*np.nan

    for i,l in enumerate(list_est):
        kdp[i,0:len(l[0])] = l[0]
        phidp_rec[i,0:len(l[1])] = l[1]
    
    kdp = ma.asarray(kdp)
    kdp.mask = masked
    phidp_rec = ma.asarray(phidp_rec)
    phidp_rec.mask = masked
    
    pool.close()
    
    return kdp, phidp_rec
    
def kdp_estimation_kalman_v11(psidp_in, dr, band = 'X', rcov = 0, pcov = 0):

   # COMPILE_OPT STRICTARR
   
   #NOTE! Parameters are not checked to save as much time as possible
   
   
   if not np.isfinite(psidp_in).any(): # Check if psidp has at least one finite value
       return psidp_in,psidp_in, psidp_in # Return the NaNs...
       
   #Set deafault of the error covariance matrices
   if not isinstance(pcov,np.ndarray):
       pcov = np.array([[(0.11+1.56*dr)**2,(0.11+1.85*dr)**2,0,(0.01+1.1*dr)**2]
                       ,[(0.11+1.85*dr)**2,(0.18+3.03*dr)**2,0,(0.01+1.23*dr)**2]
                       ,[0,      0,       0,     0]
                       ,[(0.01+1.1*dr)**2,(0.01+1.23*dr)**2,0,(-0.04+1.27*dr)**2]])
   
   if not isinstance(rcov,np.ndarray):
      rcov = np.array([[4.10625,-0.0498779,-0.0634192],
                       [-0.0498779,4.02369,-0.0421455],
                       [-0.0634192,-0.0421455,1.44300]])
                       
                       
   #=========================Parameters================================
   # ----defaults well suited for X-band 75m----
   extra_gates = 15 #per side
   
   #Intercepts -c and slope b of delta=b*Kdp+c
   #According to the Kdp threshold selected
   
   if band == 'X':
       c1 = -0.054 ; c2 = -6.155
       b1 = 2.3688 ; b2 = 0.2734
       kdp_th = 2.5
   elif band == 'C':
       c1 = -0.036 ; c2 = -1.03
       b1 = 0.53 ; b2 = 0.15
       kdp_th = 1.1
   elif band =='S':
       c1 = -0.024 ; c2 = -0.15
       b1 = 0.19 ; b2 = 0.019
       kdp_th = 2.5
        
   #Parameters for the final selection from the KF ensemble members
   fac1 = 1.2
   fac2 = 3.
   
   th1_comp = -0.1
   th2_comp = 0.1
   
   th1_final = -0.25
   
   
   #Kalman matrices
   
   #State matrix
   f = np.array([[1,0,0,0],[0,1,0,0],[0,0,0,1],[2*dr,0,0,1]],dtype=float)
   f_transposed = f.T

   # Measurement prediction matrix--------------------------------
   #J. Grazioli modification 07.2015 --previous H_plus buggy--
   h_plus = np.array([[-2*dr,1,0,1],[2*dr,1,1,0],[0,-1,0,0]],dtype=float)
   
      
   #Define the input
   psidp = psidp_in

      #Get indices of finite data
   real_data_ind = np.where(np.isfinite(psidp.ravel()))[0]
#   import pdb
#   pdb.set_trace()
   if len(real_data_ind):   
      mpsidp = psidp.ravel()[real_data_ind[-1]]
   else:   
      mpsidp = np.nan
   
   psidp = psidp[0:real_data_ind[-1]]   

   
   nrg = len(psidp)

   #Define the output
   kdp_filter_out = np.zeros([nrg,])
   
   kdp_mat = np.zeros([nrg,2*len(SCALERS)])
   kdp_sim = np.zeros([nrg,len(SCALERS)])
   phidp_filter_out = np.zeros([nrg])
   
   #==================================================================
   #==================================================================
   #Prepare a longer array with some extra gates on each side
   
   #add  values at the beginning and at the end of the profile
   psidp_long = np.zeros([nrg + extra_gates * 2,]) * np.nan
   
   nn = nrg + extra_gates * 2
   

   noise = 2*np.random.randn(extra_gates)
   psidp_long[0:extra_gates] = noise
   psidp_long[real_data_ind[-1] + extra_gates:real_data_ind[-1] + 2*extra_gates] = mpsidp + noise
   psidp_long[extra_gates:nrg + extra_gates] = psidp

   psidp = psidp_long
   
   #Get information of valid and non valid points in psidp the new psidp
   nonan = np.where(np.isfinite(psidp))[0]
   nan =  np.where(np.isnan(psidp))[0]
   
   ranged = np.arange(0,nn)
   
   psidp_interp = psidp
   
   # interpolate
   if len(nan):
       interp = interp1d(ranged[nonan],psidp[nonan],kind='zero')
       psidp_interp[nan] = interp(ranged[nan])
   else:
       psidp_interp = psidp
       
   # add noise
   if len(nan) > 0:   
      psidp_interp[nan] = psidp_interp[nan] + 2 * np.random.randn(len(nan))
   
   #Define the final input and output
   psidp = psidp_interp

   
#   plt.plot(psidp)
   #=========================================================

   #Generate the ensemble of Kalman filters estimates in backward and
   #Forward directions
   #--------------------Smallest scaler------------------------
   scaler = 10 ** (-2.)
   

   #Backward
   kdp_dummy_b2,error_kdp = kdp_estimation_backward_fixed(psidp, rcov, pcov * scaler, f, f_transposed, h_plus, c1, c2, b1, b2, kdp_th)
   kdp002 = kdp_dummy_b2[extra_gates:(nn - (extra_gates + 1))+1]
   
   #Forward
   kdp_dummy_b2,error_kdp  = kdp_estimation_forward_fixed(psidp, rcov, pcov * scaler, f, f_transposed, h_plus, c1, c2, b1, b2, kdp_th)
   kdp002f = kdp_dummy_b2[extra_gates:(nn - (extra_gates + 1))+1]
#
#   #--------------------Loop on scalers-------------------------------
   
   for i, sc in enumerate(SCALERS):
       #Backward
       kdp_dummy_b2,error_kdp = kdp_estimation_backward_fixed(psidp, rcov, pcov * scaler, f, f_transposed, h_plus, c1, c2, b1, b2, kdp_th)
       kdp_mat[:,i+1] = kdp_dummy_b2[extra_gates:(nn - (extra_gates + 1))+1]
   
       #Forward
       kdp_dummy_b2,error_kdp = kdp_estimation_forward_fixed(psidp, rcov, pcov * scaler, f, f_transposed, h_plus, c1, c2, b1, b2, kdp_th)
       kdp_mat[:,i] = kdp_dummy_b2[extra_gates:(nn - (extra_gates + 1))+1]

   #-----------------------------------------------------------------------------------------
   #=========================================================================================
   #COmpile the final estimate
   
   #Get some reference mean values
   kdp_mean = np.nanmean(kdp_mat, axis=1)
   kdp_std = np.nanstd(kdp_mat, axis=1)
   

   kdp_low_mean2 = np.nanmean(np.vstack((kdp002,kdp002f)).T,axis=1)

   diff_mean_smooth = np.convolve(kdp_low_mean2, np.ones((4,))/4, mode='same')
   
   #Backward estimate if diff_mean greater than a defined threshold
   condi = np.where(diff_mean_smooth > th2_comp)[0]
   if len(condi):   
      kdp_dummy = kdp_mat[:,np.arange(len(SCALERS)) * 2 + 1]
      kdp_sim[condi,:] = kdp_dummy[condi,:]
   
   #Forward estimate if diff_mean lower than a defined threshold
   condi = np.where(diff_mean_smooth < th1_comp)[0]
   if len(condi):   
      kdp_dummy = kdp_mat[:,np.arange(len(SCALERS)) * 2]
      kdp_sim[condi,:] = kdp_dummy[condi,:]
      
   #Combination of the two in the middle
   condi = np.where(np.logical_and(diff_mean_smooth >= th1_comp, diff_mean_smooth <= th2_comp))[0]
   if len(condi):  
      weight2 = (-0.5 / 0.15) * diff_mean_smooth + 0.5

      weight2 = np.tile(weight2,(len(SCALERS),1)).T

      kdp_dummy = (1 - weight2) * kdp_mat[:,np.arange(len(SCALERS)) * 2 + 1] + weight2 * kdp_mat[:,np.arange(len(SCALERS)) * 2]
      kdp_sim[condi,:] = kdp_dummy[condi,:]

#   #Now we reduced to 11 ensemble members: compile the final one
   kdp_mean_sim = np.nanmean(kdp_sim, axis=1)
   kdp_std_sim = np.nanstd(kdp_sim, axis=1)
#   
#   #Get the range of ensemble members that compile
#   #a final estimate
#   
#   #Lower bounds
   lower_bound = np.round(kdp_mean_sim * fac1) - np.round(kdp_std_sim * fac2)
   lower_bound = np.maximum(lower_bound, 0)
   lower_bound = np.minimum(lower_bound, 10)

#   #Upper bounds
   upper_bound = np.round(kdp_mean_sim * fac1) + np.round(kdp_std_sim * fac2)
   upper_bound = np.minimum(upper_bound, 10)
   upper_bound = np.maximum(upper_bound, 0)

   #Final selection of the ensemble members
   for uu in range(0, (nrg - 2)+(1)):
      selection_vector = np.arange(upper_bound[uu] - lower_bound[uu] + 1) + lower_bound[uu]
      selection_vector = selection_vector.astype(int)
      kdp_filter_out[uu] = np.mean(kdp_sim[uu,selection_vector])
#   
#   
#   #Final filtering of excessively negative values:
#   #TO DO: It would be better to get rid of this filtering
   ind_lt_0 = np.where(kdp_filter_out < th1_final)[0]
   if len(ind_lt_0):   
      kdp_filter_out[ind_lt_0] = kdp_low_mean2[ind_lt_0]
#   
#   
   phidp_filter_out = np.cumsum(kdp_filter_out)
   phinan = np.where(np.isnan(psidp))[0]
   if len(phinan):   
      phidp_filter_out[phinan] = np.nan
      kdp_filter_out[phinan] =  np.nan
   phidp_filter_out[nrg - 1] = np.nan; kdp_filter_out[nrg - 1] =  np.nan


   return kdp_filter_out, kdp_std, phidp_filter_out


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
    
def filter_psidp(psidp, rhohv, minsize = 5, thresh_rhohv=0.65, max_discont = 90):
    
    # Filter with RHOHV
    psidp[np.isnan(rhohv)] = np.nan
    psidp[rhohv<thresh_rhohv] = np.nan
    
    # Remove short sequences and unwrap    
    psidp_filt = np.nan * psidp    
    for i,psi_row in enumerate(psidp):
        idx = np.where(np.isfinite(psi_row))[0]
        if len(idx):
            psi_row = psi_row[0:idx[-1]]
            
            # To be sure to always have a left and right neighbour, we need to pad
            # signal with NaN
            psi_row = np.pad(psi_row,(1,1),'constant',constant_values=(np.nan,))
            idx = np.where(np.isfinite(psi_row))[0]
            nan_left = idx[np.where(np.isnan(psi_row[idx-1]))[0]]
            nan_right = idx[np.where(np.isnan(psi_row[idx+1]))[0]]
#            print psi_row[idx+1]
            
            len_sub = nan_right-nan_left
            
            for j,l in enumerate(len_sub):
                if l < minsize:
                    psi_row[nan_left[j]:nan_right[j]+1]*=np.nan
                    
            # median filter
            psi_row = medfilt(psi_row,11)
            psidp_filt[i,0:len(psi_row[1:-1])] = psi_row[1:-1]
            
    psidp_filt = ma.masked_array(psidp_filt,mask = np.isnan(psidp_filt))

    return psidp_filt


if __name__ == '__main__':
    import pickle
#    r = pickle.load(open('ex_rad.p'))
    files_c = pycosmo.get_model_filenames('/ltedata/COSMO/Multifractal_analysis/case2014040802_ONEMOM')
    time = pycosmo.get_time_from_COSMO_filename(files_c['h'][30])
    aaaa = small_radar_db.CH_RADAR_db()
    files = aaaa.query(date=[str(time)],radar='D',angle=1)
    
    rad = pyart_wrapper.PyradCH(files[0].rstrip(),False)
    phidp = rad.get_field(0,'PHIDP')
#    phidp = (np.arange(0,300)/300.)**2*10
    dr = 500./1000
    band = 'C'
    
#    a  = estimate_kdp(phidp,dr,band)
    rhohv = rad.get_field(0,'RHO')
    psidp = filter_psidp(phidp,rhohv,10,0.6)
    KDP,PHIDP = estimate_kdp_vulpiani(psidp,dr)
    KDP2,stdev,PHIDP2 = estimate_kdp_kalman(psidp,dr)
    
    plt.figure()
    plt.subplot(2,1,1)
    plt.contourf(KDP,levels=np.arange(-1,3,0.1))
    plt.subplot(2,1,2)
    plt.contourf(KDP2,levels=np.arange(-1,3,0.1))  
    
    plt.figure()
    plt.subplot(2,1,1)
    plt.contourf(PHIDP,levels=np.arange(0,90))
    plt.subplot(2,1,2)
    plt.contourf(PHIDP2,levels=np.arange(0,90))      
    
    a = rad.get_field(0,'Z').ravel()
    b = KDP2.ravel()
    mask = np.logical_and(np.isfinite(a),np.isfinite(b))
    plt.hist2d(a[mask],b[mask],bins=100)
    xlim([20,50])
    ylim([-1,4])
    #    plt.figure()
#    plt.imshow(psidp)
#    a,b,c= estimate_kdp(psidp,dr,band)
#    plt.figure()
#    plt.imshow(a)    
#    plt.figure()
#    plt.imshow(b)      
#    plt.figure()
#    plt.imshow(c)    
#    tictoc.toc()
    
#    a = kdp_estimation_kalman_v11(psidp[230,:],0.5,band='C')
    
