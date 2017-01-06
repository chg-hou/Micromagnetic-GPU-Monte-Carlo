/*
 * mx_init.h
 *
 *  Created on: 11 Nov, 2016
 *      Author: cg
 */

#ifndef MX_INIT_H_
#define MX_INIT_H_

#include <random> // rand
#include "constant.h"
#include <stdio.h>

#include <curand.h>
#include <cublas_v2.h>
#include <helper_functions.h>
#include <helper_cuda.h>


void randInitMx(double * Mx, double * My, double * Mz, int nx, int ny, int nz, unsigned long long seed )

{
	using namespace  std;
	//std::random_device rd;
	//std::mt19937 gen(rd());
	std::mt19937 gen(seed);
	std::normal_distribution<> cpurand(0.0 , RANDOM_STD );

	curandGenerator_t randgen;
	curandCreateGenerator(&randgen, CURAND_RNG_PSEUDO_DEFAULT);
	curandSetPseudoRandomGeneratorSeed(randgen, seed);

	double * d_Mx;
	double * d_My;
	double * d_Mz;

	checkCudaErrors(
					cudaMalloc((void ** )&d_Mx, sizeof(double) * nx * ny * nz));
	checkCudaErrors(
					cudaMalloc((void ** )&d_My, sizeof(double) * nx * ny * nz));
	checkCudaErrors(
					cudaMalloc((void ** )&d_Mz, sizeof(double) * nx * ny * nz));

	curandGenerateNormalDouble(randgen, d_Mx, nx * ny * nz, 0.0, RANDOM_STD);
	curandGenerateNormalDouble(randgen, d_My, nx * ny * nz, 0.0, RANDOM_STD);
	curandGenerateNormalDouble(randgen, d_Mz, nx * ny * nz, 0.0, RANDOM_STD);

	cudaMemcpy(Mx, d_Mx, sizeof(FLOAT_) * nx * ny * nz,
				cudaMemcpyDeviceToHost);
	cudaMemcpy(My, d_My, sizeof(FLOAT_) * nx * ny * nz,
				cudaMemcpyDeviceToHost);
	cudaMemcpy(Mz, d_Mz, sizeof(FLOAT_) * nx * ny * nz,
				cudaMemcpyDeviceToHost);

	checkCudaErrors(cudaFree(d_Mx));
	checkCudaErrors(cudaFree(d_My));
	checkCudaErrors(cudaFree(d_Mz));


	double Mx_new, My_new, Mz_new;
	double normal_length;
	FOR (i,0,nx * ny * nz)
	{
		Mx_new = Mx[i];
		My_new = My[i];
		Mz_new = Mz[i];

		normal_length = sqrt(Mx_new*Mx_new + My_new*My_new + Mz_new*Mz_new );
		while( (normal_length == 0) || (Mx_new ==0 && My_new ==0 && Mz_new ==0) ||
				!(std::isfinite(Mx_new) && std::isfinite(My_new) && std::isfinite(Mz_new) ) )
		{
			Mx_new = cpurand(gen);
			My_new = cpurand(gen);
			Mz_new = cpurand(gen);
			normal_length = sqrt(Mx_new*Mx_new + My_new*My_new + Mz_new*Mz_new );
		}


		Mx_new /= normal_length;
		My_new /= normal_length;
		Mz_new /= normal_length;
#ifdef DISPLAY_VERBOSE_FFT
		cout << "Mx_new: "<<Mx_new<<endl;
		cout << "My_new: "<<My_new<<endl;
		cout << "Mz_new: "<<Mz_new<<endl;
		cout << "normal_length: "<<normal_length<<endl;
		cout << "sqrt(x^2 + y^2 + z^2): "<<(Mx_new*Mx_new + My_new*My_new + Mz_new*Mz_new)<<endl;
#endif
		Mx[i] = Mx_new ;
		My[i] =	My_new ;
		Mz[i] = Mz_new ;

	}


};
template<typename T1, typename T2>  void copyMxtoMxpadded(T1 * Mx, T1 * My, T1 * Mz, int nx, int ny, int nz,
		T2 * Mx_padded, T2 * My_padded, T2 * Mz_padded, int nx_padded, int ny_padded, int nz_padded)

{

	int threadId, threadId_padded;

	FOR(x,0,nx_padded)
	{
		FOR(y,0,ny_padded)
		{
			FOR(z,0,nz_padded)
			{
				threadId= z + y * nz + x * nz* ny;
				threadId_padded = z + y * nz_padded + x * nz_padded * ny_padded;
				if (x < nx && y < ny && z < nz)
				{
					Mx_padded[threadId_padded] = Mx[threadId];
					My_padded[threadId_padded] = My[threadId];
					Mz_padded[threadId_padded] = Mz[threadId];
				}
				else{
					Mx_padded[threadId_padded] = 0;
					My_padded[threadId_padded] = 0;
					Mz_padded[threadId_padded] = 0;
				}

			}
		}

	}
};
template<typename T1, typename T2> void copyMxpaddedtoMx(
		T1 * Mx_padded, T1 * My_padded, T1 * Mz_padded, int nx_padded, int ny_padded, int nz_padded,
		T2 * Mx, T2 * My, T2 * Mz, int nx, int ny, int nz)

{
	int threadId, threadId_padded;
	FOR(x,0,nx)
	{
		FOR(y,0,ny)
		{
			FOR(z,0,nz)
			{
				threadId= z + y * nz + x * nz* ny;
				threadId_padded = z + y * nz_padded + x * nz_padded * ny_padded;

				Mx[threadId] = 	Mx_padded[threadId_padded] ;
				My[threadId] =  My_padded[threadId_padded] ;
				Mz[threadId] = 	Mz_padded[threadId_padded] ;

			}
		}

	}
}


#endif /* MX_INIT_H_ */
