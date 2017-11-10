 %module doppler_c
 %{
 /* Put header files here or function declarations like below */
float* get_refl(float *refl, int len, float *Da, int len_D_bins_x, int o2,  float *Db, int o3, int o4, float *D, int o5, int o6, float *rcs, int o8, int o9, float *N0, int nhydro, float *mu, int o11,  float *lambda, int o12, float *nu, int o13, float *step_D, int o14, float *Dmin, int o15);
 %}
 
%{
#define SWIG_FILE_WITH_INIT
%}
%include "numpy.i"
%init %{
import_array();
%}


%apply (float* ARGOUT_ARRAY1, int DIM1) {(float *refl, int len)}
%apply (float* IN_ARRAY2, int DIM1, int DIM2) {(float *Da, int len_D_bins_x,int o2)}
%apply (float* IN_ARRAY2, int DIM1, int DIM2) {(float *Db, int o3,int o4)}
%apply (float* IN_ARRAY2, int DIM1, int DIM2) {(float *D,int o5,int o6)}
%apply (float* IN_ARRAY2, int DIM1, int DIM2) {(float *rcs,int o8,int o9)}
%apply (float* IN_ARRAY1, int DIM1) {(float *N0, int nhydro)}
%apply (float* IN_ARRAY1, int DIM1) {(float *mu, int o11)}
%apply (float* IN_ARRAY1, int DIM1) {(float *lambda, int o12)}
%apply (float* IN_ARRAY1, int DIM1) {(float *nu, int o13)}
%apply (float* IN_ARRAY1, int DIM1) {(float *step_D, int o14)}
%apply (float* IN_ARRAY1, int DIM1) {(float *Dmin, int o15)}

float* get_refl(float *refl, int len, float *Da, int len_D_bins_x, int o2,  float *Db, int o3, int o4, float *D, int o5, int o6, float *rcs, int o8, int o9, float *N0, int nhydro, float *mu, int o11,  float *lambda, int o12, float *nu, int o13, float *step_D, int o14, float *Dmin, int o15);
