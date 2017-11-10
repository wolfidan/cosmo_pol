 /* File : example.c */

 #include <time.h>
 #include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h> /* memset */

float* get_refl(float *refl, int len, float *Da, int len_D_bins_x, int nhydro,  float *Db, int o1, int o2, float *rcs, int o3, int o4, float *D, int o5, int o6, float *N, int o7, int o8, float *step_D, int o9, float *Dmin, int o10);

float* get_refl(float *refl, int len, float *Da, int len_D_bins_x, int nhydro,  float *Db, int o1, int o2, float *rcs, int o3, int o4, float *D, int o5, int o6, float *N, int o7, int o8, float *step_D, int o9, float *Dmin, int o10){
   memset(refl, 0, len*sizeof(float));
   float sum=0;
   int i,j,k;
   int idx_D_a, idx_D_b;
   float D_a, D_b;
   for(i=0;i<len_D_bins_x;i++){
       for(j=0;j<nhydro;j++){
	sum=0;
	D_a=Da[i*nhydro+j];
	D_b=Db[i*nhydro+j];

	idx_D_a=(int)((D_a-Dmin[j])/step_D[j]);
	idx_D_b=(int)((D_b-Dmin[j])/step_D[j]);

	for(k=idx_D_a;k<idx_D_b;k++){
	    sum += N[k*nhydro+j]*rcs[k*nhydro+j];
	}
	refl[i] += sum*step_D[j];
}
}
return refl;}
