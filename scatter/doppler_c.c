 /* File : example.c */

 #include <time.h>
 #include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h> /* memset */

float* get_refl(float *refl, int len, float *Da, int len_D_bins_x, int o2,  float *Db, int o3, int o4, float *D, int o5, int o6, float *rcs, int o8, int o9, float *N0, int nhydro, float *mu, int o11,  float *lambda, int o12, float *nu, int o13, float *step_D, int o14, float *Dmin, int o15);

float* get_refl(float *refl, int len, float *Da, int len_D_bins_x, int o2,  float *Db, int o3, int o4, float *D, int o5, int o6, float *rcs, int o8, int o9, float *N0, int nhydro, float *mu, int o11,  float *lambda, int o12, float *nu, int o13, float *step_D, int o14, float *Dmin, int o15){
   memset(refl, 0, len*sizeof(float));
   float sum=0;
   int i,j,k;
   int idx_D_a, idx_D_b;
   float D_a, D_b;
   for(i=0;i<len_D_bins_x;i++){
       for(j=0;j<nhydro;j++){
	if(lambda[j]>0){
	sum=0;
	D_a=Da[i*3+j];
	D_b=Db[i*3+j];

	idx_D_a=(int)((D_a-Dmin[j])/step_D[j]);
	idx_D_b=(int)((D_b-Dmin[j])/step_D[j]);

	for(k=idx_D_a;k<idx_D_b;k++){
	    sum+=pow(D[k*3+j], mu[j])*exp(-lambda[j]*pow(D[k*3+j], nu[j]))*rcs[k*3+j];
	}
	refl[i]+=sum*step_D[j]*N0[j];
}}
}
return refl;}
