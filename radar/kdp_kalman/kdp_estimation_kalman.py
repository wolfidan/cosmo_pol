# -*- coding: utf-8 -*-


import numpy as np
import scipy
from scipy.interpolate import interp1d


SCALERS = [0.1,10**(-0.8),10**(-0.6), 10**(-0.4), 10**(-0.2), 1,
           10**(-0.2), 10**(-0.4), 10**(-0.6), 10**(-0.8),1, 10]


'''
 NAME:
    kdp_estimation_backward_fixed

 PURPOSE:
    Processing one profile of Psidp and estimating Kdp and Phidp
    with the KFE algorithm described in Schneebeli et al, 2014
    IEEE_TGRS. This routine estimates Kdp in the backward
    direction given a set of matrices that define the Kalman
    filter.

 AUTHOR(S):
    Marc Schneebeli: original code
    Jacopo Grazioli: current version
    Daniel Wolfensberger: Python version

 CALLING SEQUENCE:

Kdp_estimation_backward_fixed,Psidp_in,$
                                Rcov,$
                                Pcov_scale,$
                                F,$
                                F_transposed,$
                                H_plus,$
                                c1,$
                                c2,$
                                b1,$
                                b2,$
                                kdp_th,$

 INPUT PARAMETERS:
    psidp_in    : one-dimensional vector of length -nrg-
                  containining the input psidp [degrees]
    Rcov        : measurement error covariance matrix [3x3]
    Pcov        : scaled state transition error covariance matrix
                   [4x4]
    F           : forward state prediction matrix [4x4]
    F_transposed: transposed of F
    H_plus      : measurement prediction matrix [4x3]
    c1, c2,b1,b2: the values of the intercept of the relation
                  c  = b*Kdp -delta. This relation uses
                  b1, c1 IF kdp is lower than a kdp_th and b2, c2
                  otherwise
   kdp_th:        see above

 OUTPUT:

 INPUT KEYWORDS:


 OUTPUT KEYWORDS:
    kdp:  filtered Kdp [degrees/km]. Same length as Psidp
    error_kdp: estimated error on Kdp values
NOZERO
 COMMON BLOCKS:
   ;
 DEPENDENCIES:


 MODIFICATION HISTORY:
   2012/2013: Schneebeli, creation
   July 2015: re-implementation J. Grazioli
   August 2016: Rewritten in Python, D. Wolfensberger
'''


def kdp_estimation_backward_fixed(psidp_in, rcov, pcov_scale, f, f_transposed, h_plus, c1, c2, b1, b2, kdp_th):
   # COMPILE_OPT STRICTARR
   
   #--------------------------------------------------------
   #Define the input
   psidp = psidp_in
   
   #Get indices of finite data
   real_data_ind = np.where(np.isfinite(psidp))[0]
   
   #Get indices of finite data
   real_data_ind = np.where(np.isfinite(psidp.ravel()))[0]
#   import pdb
#   pdb.set_trace()
   if len(real_data_ind):   
      mpsidp = psidp.ravel()[real_data_ind[-1]]
   else:   
      mpsidp = np.nan
   
   # invert Psidp (backward estimation)
   psidp = -1 * (psidp[::-1] - mpsidp)
   nrg_new = len(psidp)
   
   #==========================================================================
   
   #Initialize the state vector to 0 (J. Grazioli 2015)
   s = np.zeros([4,1])  #first state estimate

   #define measurement vector
   z = np.zeros([3,1])
   
   #Define the identity matrix
   identity_i = np.eye(4)
   p = identity_i * 4.
   
   kdp = np.zeros([nrg_new])
   kdp_error = np.zeros([nrg_new])
   

   #Loop on all the gates and apply the filter
   
   for ii in range(0, (nrg_new - 2)+(1)):
      z[0] = psidp[ii+1]
      z[1] = psidp[ii]

#      print s
      s_pred = np.dot(f, s)   #state prediciton     
      p_pred = np.dot(np.dot(f, p), f_transposed) + pcov_scale #error prediction

      if (s_pred[0] > kdp_th): 
         h_plus[2,0] = b2
         z[2] = c2
      else:   
         h_plus[2,0] = b1
         z[2] = c1

#      print h_plus
      #as far as i see aludc is symmetrical, so i do not transpose it
      #aludc = H_plus ## P_pred ## TRANSPOSE(H_plus)+Rcov ;OR A_mat
      aludc = np.dot(h_plus, ( np.dot(h_plus, p_pred).T)) + rcov
      
      #B_mat = P_pred ## TRANSPOSE(H_plus)--here below we get the transposed of B_mat directly
      # B_mat=MATRIX_MULTIPLY(P_pred,H_plus,/ATRANSPOSE) ;But  P_pred is symmetrical
      b_mat =  np.dot(h_plus, p_pred)
#      print b_mat
      #; LA_LUDC, aludc, index   ; LU decomposition
      cho = scipy.linalg.cho_factor(aludc)
      k = scipy.linalg.cho_solve(cho,b_mat,check_finite=False,overwrite_b=True).T

      #Update state and error
      s =  np.dot(k, ( np.dot(-h_plus, s_pred) + z)) + s_pred
      p =  np.dot((identity_i -  np.dot(k, h_plus)), p_pred)

      #Fill the output
      kdp[ii] = s[0]
      kdp_error[ii] = p[0,0]
   
   #Reverse the estimates (backward direction)
   kdp =  kdp[::-1]
   kdp_error = kdp_error[::-1]
   
   return kdp, kdp_error

'''
 NAME:
    kdp_estimation_forward_fixed

 PURPOSE:
    Processing one profile of Psidp and estimating Kdp and Phidp
    with the KFE algorithm described in Schneebeli et al, 2014
    IEEE_TGRS. This routine estimates Kdp in the forward
    direction given a set of matrices that define the Kalman
    filter.

 AUTHOR(S):
    Marc Schneebeli: original code
    Jacopo Grazioli: current version
    Daniel Wolfensberger: Python version
 CALLING SEQUENCE:

Kdp_estimation_backward_fixed,Psidp_in,$
                                Rcov,$
                                Pcov_scale,$
                                F,$
                                F_transposed,$
                                H_plus,$
                                c1,$
                                c2,$
                                b1,$
                                b2,$
                                kdp_th,$


 INPUT PARAMETERS:
    psidp_in    : one-dimensional vector of length -nrg-
                  containining the input psidp [degrees]
    Rcov        : measurement error covariance matrix [3x3]
    Pcov        : scaled state transition error covariance matrix
                   [4x4]
    F           : forward state prediction matrix [4x4]
    F_transposed: transposed of F
    H_plus      : measurement prediction matrix [4x3]
    c1, c2,b1,b2: the values of the intercept of the relation
                  c  = b*Kdp -delta. This relation uses
                  b1, c1 IF kdp is lower than a kdp_th and b2, c2
                  otherwise
   kdp_th:        see above

 OUTPUT:

 INPUT KEYWORDS:


 OUTPUT KEYWORDS:
    kdp:  filtered Kdp [degrees/km]. Same length as Psidp
    error_kdp: estimated error on Kdp values
NOZERO
 COMMON BLOCKS:
   ;
 DEPENDENCIES:


 MODIFICATION HISTORY:
   2012/2013: Schneebeli, creation
   July 2015: re-implementation J. Grazioli
   August 2016: Conversion to Python, D. Wolfensberger

'''
def kdp_estimation_forward_fixed(psidp_in, rcov, pcov_scale, f, f_transposed, h_plus, c1, c2, b1, b2, kdp_th):
   
   # COMPILE_OPT STRICTARR
   
   #--------------------------------------------------------
   #Define the input
   psidp = psidp_in
   nrg_new = len(psidp)

   #==========================================================================
   
   #Initialize the state vector to 0 (J. Grazioli 2015)
   s = np.zeros([4,1])  #first state estimate

   #define measurement vector
   z = np.zeros([3,1])
   
   #Define the identity matrix
   identity_i = np.eye(4)
   p = identity_i * 4.
   
   phidp = np.zeros([nrg_new])
   kdp = np.zeros([nrg_new])
   kdp_error = np.zeros([nrg_new])

   #Loop on all the gates and apply the filter
   for ii in range(0, (nrg_new - 2)+(1)):
      z[0] = psidp[ii +1]
      z[1] = psidp[ii]
      
      s_pred = np.dot(f, s)   #state prediciton     
      p_pred = np.dot(np.dot(f, p), f_transposed) + pcov_scale #error prediction
      
      if (s_pred[0] > kdp_th):   
         h_plus[2,0] = b2
         z[2] = c2
      else:   
         h_plus[2,0] = b1
         z[2] = c1
      
      #as far as i see aludc is symmetrical, so i do not transpose it
      aludc = np.dot(h_plus, ( np.dot(h_plus, p_pred).T)) + rcov
      
      # below we get the transposed of B_mat directly
      b_mat = np.dot(h_plus, p_pred)

      #Solve the linear system
      cho = scipy.linalg.cho_factor(aludc)
      k = scipy.linalg.cho_solve(cho,b_mat,check_finite=False,overwrite_b=True).T
  
      #Update state and error
      s =  np.dot(k, ( np.dot(-h_plus, s_pred) + z)) + s_pred
      p =  np.dot((identity_i -  np.dot(k, h_plus)), p_pred)
      
      #Fill the output
      kdp[ii] = s[0]
      kdp_error[ii] = p[0,0]
      phidp[ii]     = s[2]

      
   return kdp, kdp_error


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
    band: radar frequency band string. Accepted "X", "C", "S" (capital
       or not). IT is used to compute Rcov and Pcov when not provided
    Rcov, Pcov:  error covariance matrix of state transition
                 (Pcov), and of measurements (Rcov).
                 If not set, or if set to objects with less
                 than 2 elements, a default parametrization
                 is used, valid for X-band and 75m gate resolution.
                 Pcov is a 4x4 and Rcov a 3x3 matrix

 OUTPUT
    kdp_filter_out: vector, same length of psidp_in
                    containing the estimated kdp profile
    kdp_std       : estimated standard deviation of kdp
    phidp_filter_out: estimated phidp (smooth psidp)
    
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


'''

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
