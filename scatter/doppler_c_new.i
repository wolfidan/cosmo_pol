 %module doppler_c_new
 %{
 /* Put header files here or function declarations like below */
float* get_refl(float *refl, int len, float *Da, int len_D_bins_x, int nhydro,  float *Db, int o1, int o2, float *rcs, int o3, int o4, float *D, int o5, int o6, float *N, int o7, int o8, float *step_D, int o9, float *Dmin, int o10);
 %}
 
%{
#define SWIG_FILE_WITH_INIT
%}
%include "numpy.i"
%init %{
import_array();
%}


%apply (float* ARGOUT_ARRAY1, int DIM1) {(float *refl, int len)}
%apply (float* IN_ARRAY2, int DIM1, int DIM2) {(float *Da, int len_D_bins_x,int nhydro)}
%apply (float* IN_ARRAY2, int DIM1, int DIM2) {(float *Db, int o1,int o2)}
%apply (float* IN_ARRAY2, int DIM1, int DIM2) {(float *rcs,int o3,int o4)}
%apply (float* IN_ARRAY2, int DIM1, int DIM2) {(float *D,int o5,int o6)}
%apply (float* IN_ARRAY2, int DIM1, int DIM2) {(float *N,int o7,int o8)}
%apply (float* IN_ARRAY1, int DIM1) {(float *step_D, int o9)}
%apply (float* IN_ARRAY1, int DIM1) {(float *Dmin, int o10)}

float* get_refl(float *refl, int len, float *Da, int len_D_bins_x, int nhydro,  float *Db, int o1, int o2, float *rcs, int o3, int o4, float *D, int o5, int o6, float *N, int o7, int o8, float *step_D, int o9, float *Dmin, int o10);
