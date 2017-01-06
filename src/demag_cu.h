/*
 * demag_cu.h
 *
 *  Created on: 2 Nov, 2016
 *      Author: cg
 */

#ifndef DEMAG_CU_H_
#define DEMAG_CU_H_

#include <cuda_runtime.h>
#include "constant.h"
#include <iostream>
#include <helper_cuda.h>
#include <boost/progress.hpp>

__global__ void ktensorKernel2_step0_init(int nx_padded, int ny_padded, int nz_padded,
		double * Kxx, double * Kxy, double * Kxz, double * Kyy, double * Kyz,
		double * Kzz);
__global__ void ktensorKernel2_step1(int nx_padded, int ny_padded, int nz_padded,
		double dx, double dy, double dz,
		double * Kxx, double * Kxy, double * Kxz, double * Kyy, double * Kyz,
		double * Kzz,
		int pbc_idx_x, int pbc_idx_y, int pbc_idx_z);
__global__ void ktensorKernel2_step2_div_const(int nx_padded, int ny_padded, int nz_padded,
		double dx, double dy, double dz,
		double * Kxx, double * Kxy, double * Kxz, double * Kyy, double * Kyz,
		double * Kzz) ;


__global__ void ktensorKernel(int nx_padded, int ny_padded, int nz_padded,
		double dx, double dy, double dz, int pbc_x, int pbc_y, int pbc_z,
		double * Kxx, double * Kxy, double * Kxz, double * Kyy, double * Kyz,
		double * Kzz,
		volatile int *progress);
__global__ void calHexchangeKernel(int nx_padded, int ny_padded, int nz_padded,
		FLOAT2 * Mx_padded_fft, FLOAT2 * My_padded_fft, FLOAT2 * Mz_padded_fft,
		FLOAT2 * Kxx_fft, FLOAT2 * Kxy_fft, FLOAT2 * Kxz_fft, FLOAT2 * Kyy_fft,	FLOAT2 * Kyz_fft, FLOAT2 * Kzz_fft,
		FLOAT2 * Hx_padded_fft,  FLOAT2 * Hy_padded_fft, FLOAT2 * Hz_padded_fft);
;
namespace ktensorcpu
{
__inline__ double f(double x, double y, double z) {
	x = ABS(x);
	y = ABS(y);
	z = ABS(z);

	return +y / 2.0 * (z * z - x * x)
			* asinh(y / (sqrt(x * x + z * z) + EPSILON))
			+ z / 2.0 * (y * y - x * x)
					* asinh(z / (sqrt(x * x + y * y) + EPSILON))
			- x * y * z
					* atan(y * z / (x * sqrt(x * x + y * y + z * z) + EPSILON))
			+ 1.0 / 6.0 * (2 * x * x - y * y - z * z)
					* sqrt(x * x + y * y + z * z);
}
__inline__ double g(double x, double y, double z) {
	z = ABS(z);
	return +x * y * z * asinh(z / (sqrt(x * x + y * y) + EPSILON))
			+ y / 6.0 * (3.0 * z * z - y * y)
					* asinh(x / (sqrt(y * y + z * z) + EPSILON))
			+ x / 6.0 * (3.0 * z * z - x * x)
					* asinh(y / (sqrt(x * x + z * z) + EPSILON))
			- z * z * z / 6.0
					* atan(x * y / (z * sqrt(x * x + y * y + z * z) + EPSILON))
			- z * y * y / 2.0
					* atan(x * z / (y * sqrt(x * x + y * y + z * z) + EPSILON))
			- z * x * x / 2.0
					* atan(y * z / (x * sqrt(x * x + y * y + z * z) + EPSILON))
			- x * y * sqrt(x * x + y * y + z * z) / 3.0;
}

void ktensorCPU (int nx_padded, int ny_padded, int nz_padded,
		double dx, double dy, double dz, int pbc_x, int pbc_y, int pbc_z,
		double * Kxx_d, double * Kxy_d, double * Kxz_d, double * Kyy_d, double * Kyz_d,
		double * Kzz_d, bool & CTRL_C_QUIT_FLAG);
}
#endif /* DEMAG_CU_H_ */
