/*
 * constant.h
 *
 *  Created on: 7 Nov, 2016
 *      Author: cg
 */

#ifndef CONSTANT_H_
#define CONSTANT_H_

#include <cuda_runtime.h>
#include <string>
#include <iomanip>

#define RANDOM_STD   1e1

#define ABS(X)   (((X)>0)?(X):(-X))
#define FOR01(x)    for(int (x)=0; (x)<2; x++)
#define FOR(x,a,b)    for(int (x)=a; (x)<b; x++)

#define CONST_PI    3.14159265358
#define CONST_K_BOLTZMANN     1.3806488E-23		//   J/K
#define EPSILON     1e-200
#define MU0 	 	1.256637061E-6//4e-7 * CONST_PI


#define CUDA_CALL(x) do { if((x)!=cudaSuccess) { \
    printf("Error at %s:%d\n",__FILE__,__LINE__);\
    return EXIT_FAILURE;}} while(0)
#define CURAND_CALL(x) do { if((x)!=CURAND_STATUS_SUCCESS) { \
    printf("Error at %s:%d\n",__FILE__,__LINE__);\
    return EXIT_FAILURE;}} while(0)

//#define USE_CPU_TO_CALCULATE_K_TENSOR

//#define DISPLAY_VERBOSE_FFT
//#define DEBUG_H_FIELD
//#define USE_ONE_HOT_MX_INIT


//#define USE_DOULBE_IN_GPU



#ifdef USE_DOULBE_IN_GPU
	#define FLOAT_  double
	#define FLOAT2 double2
#else
	#define FLOAT_  float
	#define FLOAT2 float2
#endif




#endif /* CONSTANT_H_ */
