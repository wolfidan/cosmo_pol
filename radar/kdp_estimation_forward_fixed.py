import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy.linalg import blas as FB
super_dot = lambda v, w: FB.dgemm(alpha=1., a=v.T, b=w.T, trans_b=True)
 #+
# NAME:
#    kdp_estimation_backward_fixed
#
# PURPOSE:
#    Processing one profile of Psidp and estimating Kdp and Phidp
#    with the KFE algorithm described in Schneebeli et al, 2014
#    IEEE_TGRS. This routine estimates Kdp in the backward
#    direction given a set of matrices that define the Kalman
#    filter.
#
# AUTHOR(S):
#    Marc Schneebeli: original code
#    Jacopo Grazioli: current version
#
# CALLING SEQUENCE:
#
#Kdp_estimation_backward_fixed,Psidp_in,$
#                                Rcov,$
#                                Pcov_scale,$
#                                F,$
#                                F_transposed,$
#                                H_plus,$
#                                c1,$
#                                c2,$
#                                b1,$
#                                b2,$
#                                kdp_th,$


# INPUT PARAMETERS:
#    psidp_in    : one-dimensional vector of length -nrg-
#                  containining the input psidp [degrees]
#    Rcov        : measurement error covariance matrix [3x3]
#    Pcov        : scaled state transition error covariance matrix
#                   [4x4]
#    F           : forward state prediction matrix [4x4]
#    F_transposed: transposed of F
#    H_plus      : measurement prediction matrix [4x3]
#    c1, c2,b1,b2: the values of the intercept of the relation
#                  c  = b*Kdp -delta. This relation uses
#                  b1, c1 IF kdp is lower than a kdp_th and b2, c2
#                  otherwise
#   kdp_th:        see above
#
# OUTPUT:

# INPUT KEYWORDS:
#
#
# OUTPUT KEYWORDS:
#    kdp:  filtered Kdp [degrees/km]. Same length as Psidp
#    error_kdp: estimated error on Kdp values
#NOZERO
# COMMON BLOCKS:
#   ;
# DEPENDENCIES:
#
#
# MODIFICATION HISTORY:
#   2012/2013: Schneebeli, creation
#   July 2015: re-implementation J. Grazioli

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
      #aludc = H_plus ## P_pred ## TRANSPOSE(H_plus)+Rcov ;OR A_mat
      aludc = np.dot(h_plus, ( np.dot(h_plus, p_pred).T)) + rcov
      
      #B_mat = P_pred ## TRANSPOSE(H_plus)--here below we get the transposed of B_mat directly
      # B_mat=MATRIX_MULTIPLY(P_pred,H_plus,/ATRANSPOSE) ;But  P_pred is symmetrical
      b_mat = np.dot(h_plus, p_pred)
#      print b_mat
      #; LA_LUDC, aludc, index   ; LU decomposition

#      k = np.linalg.solve(aludc,b_mat).T #Solve the linear system
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





