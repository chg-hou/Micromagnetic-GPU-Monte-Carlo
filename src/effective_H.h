/*
 * effective_H.h
 *
 *  Created on: Nov 9, 2016
 *      Author: cg
 */

#ifndef EFFECTIVE_H_H_
#define EFFECTIVE_H_H_

#include "constant.h"

__inline__ __device__ int il_min(int a, int b) {
	return (a < b) ? a : b;
}
__inline__ __device__ int il_max(int a, int b) {
	return (a > b) ? a : b;
}

#define INDEX(X,Y,Z)   ((Z) + (Y) * nz_padded + (X) * nz_padded * ny_padded)
//#define MIN(X,Y)   		(  ((X)<(Y))?(X):(Y)  )



__global__ void effectiveHKernel(FLOAT_ ms,FLOAT_ factor_H_exch, FLOAT_ Dind,
		FLOAT_ Ku1, FLOAT_ anisUx, FLOAT_ anisUy, FLOAT_ anisUz,
		FLOAT_ Bextx, FLOAT_ Bexty, FLOAT_ BextZ,
		int nx, int ny, int nz,
		int nx_padded, int ny_padded, int nz_padded, double dx, double dy,
		double dz, int pbc_x, int pbc_y, int pbc_z,
		FLOAT_ * Mx_padded, FLOAT_ *My_padded, FLOAT_ *Mz_padded,
		FLOAT2 * Hx_padded_fft, FLOAT2 *Hy_padded_fft, FLOAT2 *Hz_padded_fft,
		FLOAT_ * d_energy, bool cal_demag_flag) ;

__global__ void calEnergyKernel(FLOAT_ ms, FLOAT_ mu0,FLOAT_ factor_H_exch, FLOAT_ Aex, FLOAT_ Dind,
		FLOAT_ Ku1, FLOAT_ anisUx, FLOAT_ anisUy, FLOAT_ anisUz,
		FLOAT_ Bextx, FLOAT_ Bexty, FLOAT_ Bextz,
		int nx, int ny, int nz,
		int nx_padded, int ny_padded, int nz_padded, double dx, double dy,
		double dz, int pbc_x, int pbc_y, int pbc_z,
		FLOAT_ * Mx_padded, FLOAT_ *My_padded, FLOAT_ *Mz_padded,
		FLOAT2 * Hx_padded_fft, FLOAT2 *Hy_padded_fft, FLOAT2 *Hz_padded_fft,
		FLOAT_ * d_energy, bool cal_demag_flag) ;

#endif /* EFFECTIVE_H_H_ */
