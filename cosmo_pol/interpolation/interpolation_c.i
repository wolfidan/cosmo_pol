 %module radar_interp_c
 %{
 /* Put header files here or function declarations like below */
extern float* get_all_radar_pts(float *output, int len, float *coords_rad_pts, int crp_x, int crp_y, float *radar_heights, int rh_x, float *model_data, int md_x, int md_y, int md_z, float *model_heights, int mh_x, int mh_y, int mh_z, float *llc_cosmo, int llc_cosmo_x, float *res_cosmo, int res_cosmo_x);
 %}
 
%{
#define SWIG_FILE_WITH_INIT
%}
%include "numpy.i"
%init %{
import_array();
%}

%apply (float* ARGOUT_ARRAY1, int DIM1) {(float *output, int len)}
%apply (float* IN_ARRAY2, int DIM1, int DIM2) {(float *coords_rad_pts, int crp_x, int crp_y)}
%apply (float* IN_ARRAY1, int DIM1) {(float *radar_heights, int rh_x)}
%apply (float* IN_ARRAY3, int DIM1, int DIM2, int DIM3) {(float *model_data, int md_x, int md_y, int md_z)}
%apply (float* IN_ARRAY3, int DIM1, int DIM2, int DIM3) {(float *model_heights, int mh_x, int mh_y, int mh_z)}
%apply (float* IN_ARRAY1, int DIM1) {(float *llc_cosmo, int llc_cosmo_x)}
%apply (float* IN_ARRAY1, int DIM1) {(float *res_cosmo, int res_cosmo_x)}

float* get_all_radar_pts(float *output, int len, float *coords_rad_pts, int crp_x, int crp_y, float *radar_heights, int rh_x, float *model_data, int md_x, int md_y, int md_z, float *model_heights, int mh_x, int mh_y, int mh_z, float *llc_cosmo, int llc_cosmo_x, float *res_cosmo, int res_cosmo_x);
int binary_search(float *arr, int dim, float key);
float trilinear_interp(float *c_n, int c_n_x, float *v_n, int v_n_x, float *pos_radar, int p_r_x, float radar_bin_height);
