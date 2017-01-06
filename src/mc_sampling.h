/*
 * mc_sampling.h
 *
 *  Created on: 10 Nov, 2016
 *      Author: cg
 */

#ifndef MC_SAMPLING_H_
#define MC_SAMPLING_H_

#include "constant.h"
#include "effective_H.h"

__global__ void mcKernel(
		FLOAT_ ms, FLOAT_ mu0, FLOAT_ Aex, FLOAT_ Dind,
		FLOAT_ Ku1, FLOAT_ anisUx, FLOAT_ anisUy, FLOAT_ anisUz,
		FLOAT_ Bextx, FLOAT_ Bexty, FLOAT_ Bextz,
		FLOAT_ kT,
		int nx, int ny, int nz, int nx_padded, int ny_padded,
		int nz_padded, int meshsize, double dx, double dy, double dz, int pbc_x, int pbc_y,
		int pbc_z,
		FLOAT_  Kxx_mirror, FLOAT_ Kyy_mirror, FLOAT_  Kzz_mirror,
		FLOAT_ * Mx_padded, FLOAT_ *My_padded, FLOAT_ *Mz_padded,
		FLOAT_ * Mx_padded_output, FLOAT_ *My_padded_output, FLOAT_ *Mz_padded_output,
		FLOAT2 * Hx_padded_fft, FLOAT2 *Hy_padded_fft, FLOAT2 *Hz_padded_fft,
		FLOAT_ * dRandPool, FLOAT_ * dRandPool_reject, int randPoolCount, bool cal_demag_flag);
__global__ void mcKernel2(
		FLOAT_ ms, FLOAT_ mu0, FLOAT_ Aex, FLOAT_ Dind,
		FLOAT_ Ku1, FLOAT_ anisUx, FLOAT_ anisUy, FLOAT_ anisUz,
		FLOAT_ Bextx, FLOAT_ Bexty, FLOAT_ Bextz,
		FLOAT_ kT,
		int nx, int ny, int nz, int nx_padded, int ny_padded,
		int nz_padded, int meshsize, double dx, double dy, double dz, int pbc_x, int pbc_y,
		int pbc_z,
		FLOAT_  Kxx_mirror, FLOAT_ Kxy_mirror, FLOAT_  Kxz_mirror,
		FLOAT_  Kyy_mirror, FLOAT_ Kyz_mirror, FLOAT_  Kzz_mirror,
		FLOAT_ * Mx_padded, FLOAT_ *My_padded, FLOAT_ *Mz_padded,
		FLOAT_ * Mx_padded_output, FLOAT_ *My_padded_output, FLOAT_ *Mz_padded_output,
		FLOAT2 * Hx_padded_fft, FLOAT2 *Hy_padded_fft, FLOAT2 *Hz_padded_fft,
		FLOAT_ * dRandPool, FLOAT_ * dRandPool_reject, int randPoolCount, bool cal_demag_flag
		, FLOAT_ * d_energy);

#endif /* MC_SAMPLING_H_ */
