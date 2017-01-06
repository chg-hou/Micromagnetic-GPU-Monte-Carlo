/*
 * demag_cu.h
 *
 *  Created on: 2 Nov, 2016
 *      Author: cg
 */
#include "demag_cu.h"
#include <cuComplex.h>

#define ComplexXreal(a,b) ((a).x * (b).x - (a).y * (b).y)
#define ComplexXimag(a,b) ((a).x * (b).y + (a).y * (b).x)
#define ComplexSet(a,b)  (a).x = (b).x ; (a).y = (b).y;

#define ABS(X)   (((X)>0)?(X):(-X))

__inline__ __device__ double f(double x, double y, double z) {
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
__inline__ __device__ double g(double x, double y, double z) {
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

__global__ void ktensorKernel2_step0_init(int nx_padded, int ny_padded, int nz_padded,
		double * Kxx, double * Kxy, double * Kxz, double * Kyy, double * Kyz,
		double * Kzz) {

	// TODO: ignore K(0,0) term, to remove H acting on (0,0)
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int z = blockIdx.z * blockDim.z + threadIdx.z;
	int threadId = z + y * nz_padded + x * nz_padded * ny_padded;
	if (x >= nx_padded || y >= ny_padded || z >= nz_padded)
		return;

	Kxx[threadId] = 0;
	Kxy[threadId] = 0;
	Kxz[threadId] = 0;
	Kyy[threadId] = 0;
	Kyz[threadId] = 0;
	Kzz[threadId] = 0;

}


__global__ void calHexchangeKernel(int nx_padded, int ny_padded, int nz_padded,
		FLOAT2 * Mx_padded_fft, FLOAT2 * My_padded_fft, FLOAT2 * Mz_padded_fft,
		FLOAT2 * Kxx_fft, FLOAT2 * Kxy_fft, FLOAT2 * Kxz_fft, FLOAT2 * Kyy_fft,
		FLOAT2 * Kyz_fft, FLOAT2 * Kzz_fft, FLOAT2 * Hx_padded_fft,
		FLOAT2 * Hy_padded_fft, FLOAT2 * Hz_padded_fft) {

	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int z = blockIdx.z * blockDim.z + threadIdx.z;
	int threadId = z + y * nz_padded + x * ny_padded * nz_padded;
	if (x >= nx_padded || y >= ny_padded || z >= nz_padded)
		return;

	unsigned int  normal_factor = nx_padded * ny_padded * nz_padded;

	FLOAT2 Mx, My, Mz;
	FLOAT2 Kxx, Kxy, Kxz, Kyy, Kyz, Kzz;

	ComplexSet(Mx, Mx_padded_fft[threadId]);
	ComplexSet(My, My_padded_fft[threadId]);
	ComplexSet(Mz, Mz_padded_fft[threadId]);

	ComplexSet(Kxx, Kxx_fft[threadId]);
	ComplexSet(Kxy, Kxy_fft[threadId]);
	ComplexSet(Kxz, Kxz_fft[threadId]);
	ComplexSet(Kyy, Kyy_fft[threadId]);
	ComplexSet(Kyz, Kyz_fft[threadId]);
	ComplexSet(Kzz, Kzz_fft[threadId]);


	Hx_padded_fft[threadId].x = (ComplexXreal(Mx, Kxx) + ComplexXreal(My,Kxy)
	+ ComplexXreal(Mz,Kxz))/normal_factor;
	Hx_padded_fft[threadId].y = (ComplexXimag(Mx, Kxx) + ComplexXimag(My,Kxy)
	+ ComplexXimag(Mz,Kxz))/normal_factor;

	Hy_padded_fft[threadId].x = (ComplexXreal(Mx, Kxy) + ComplexXreal(My,Kyy)
	+ ComplexXreal(Mz,Kyz))/normal_factor;
	Hy_padded_fft[threadId].y = (ComplexXimag(Mx, Kxy) + ComplexXimag(My,Kyy)
	+ ComplexXimag(Mz,Kyz))/normal_factor;

	Hz_padded_fft[threadId].x = (ComplexXreal(Mx, Kxz) + ComplexXreal(My,Kyz)
	+ ComplexXreal(Mz,Kzz))/normal_factor;
	Hz_padded_fft[threadId].y = (ComplexXimag(Mx, Kxz) + ComplexXimag(My,Kyz)
	+ ComplexXimag(Mz,Kzz))/normal_factor;




}


void ktensorcpu::ktensorCPU (int nx_padded, int ny_padded, int nz_padded,
		double dx, double dy, double dz, int pbc_x, int pbc_y, int pbc_z,
		double * Kxx_d, double * Kxy_d, double * Kxz_d, double * Kyy_d, double * Kyz_d,
		double * Kzz_d, bool & CTRL_C_QUIT_FLAG)
{
		using namespace std;


	    long meshsize_padded = nx_padded * ny_padded * nz_padded;

	    printf("      ");

	    double * Kxx = new double[sizeof(double)*meshsize_padded];
	    double * Kxy = new double[sizeof(double)*meshsize_padded];
	    double * Kyy = new double[sizeof(double)*meshsize_padded];
	    double * Kxz = new double[sizeof(double)*meshsize_padded];
	    double * Kyz = new double[sizeof(double)*meshsize_padded];
	    double * Kzz = new double[sizeof(double)*meshsize_padded];

	    boost::progress_display show_progress( meshsize_padded );

	    FOR(x,0,nx_padded)
	    {
	        FOR(y,0,ny_padded)
	        {
	            FOR(z,0,nz_padded)
	            {
	                int threadId = z + y * nz_padded + x * nz_padded * ny_padded;

	                int idx_x, idx_y, idx_z;
	                idx_x = (x + (nx_padded+1)/2-1  )%( nx_padded ) - (nx_padded+1)/2 + 1;
	                idx_y = (y + (ny_padded+1)/2-1  )%( ny_padded ) - (ny_padded+1)/2 + 1;
	                idx_z = (z + (nz_padded+1)/2-1  )%( nz_padded ) - (nz_padded+1)/2 + 1;

	                double tmp;
	                double kxx, kxy, kxz, kyy, kyz, kzz;
	                kxx=0;kxy=0;kxz=0;kyy=0;kyz=0;kzz=0;

	                for (int pbc_idx_x = -pbc_x; pbc_idx_x <= pbc_x; pbc_idx_x++)
	                {
	                    for (int pbc_idx_y = -pbc_y; pbc_idx_y <= pbc_y; pbc_idx_y++)
	                    {
	                        for (int pbc_idx_z = -pbc_z; pbc_idx_z <= pbc_z; pbc_idx_z++)
	                        {

	                            int ix = idx_x + pbc_idx_x*nx_padded;
	                            int iy = idx_y + pbc_idx_y*ny_padded;
	                            int iz = idx_z + pbc_idx_z*nz_padded;

	                            int iix,iiy,iiz;
	                            bool sign;
	                            FOR01(ikx)
	                            {
	                                FOR01(iky)
	                                {
	                                    FOR01(ikz)
	                                    {
	                                        FOR01(ilx)
	                                        {
	                                            FOR01(ily)
	                                            {
	                                                FOR01(ilz)
	                                                {

	                                                    iix = ix+ikx-ilx;
	                                                    iiy = iy+iky-ily;
	                                                    iiz = iz+ikz-ilz;


	                                                    sign = ((ikx + iky + ikz + ilx + ily + ilz)%2 == 0);

	                                                    tmp = f(iix*dx,iiy*dy,iiz*dz);
	                                                    //cout<<"f :"<<tmp<<endl;
	                                                    kxx += (sign? tmp: -tmp);

	                                                    tmp = g(iix*dx,iiy*dy,iiz*dz);	//xy
	                                                    kxy += (sign? tmp: -tmp);

	                                                    tmp = g(iix*dx,iiz*dz,iiy*dy);	//xz
	                                                    kxz += (sign? tmp: -tmp);

	                                                    tmp = f(iiy*dy,iiz*dz,iix*dx);
	                                                    kyy += (sign? tmp: -tmp);

	                                                    tmp = g(iiy*dy,iiz*dz,iix*dx);	//yz
	                                                    kyz += (sign? tmp: -tmp);

	                                                    tmp = f(iiz*dz,iix*dx,iiy*dy);
	                                                    kzz += (sign? tmp: -tmp);
	                                                }
	                                            }
	                                        }
	                                    }
	                                }
	                            }



	                            //end PBC
	                        }
	                    }
	                }
	                tmp = -CONST_PI*4.0*dx*dy*dz;

	                Kxx[threadId] = kxx / tmp;
	                Kxy[threadId] = kxy / tmp;
	                Kxz[threadId] = kxz / tmp;
	                Kyy[threadId] = kyy / tmp;
	                Kyz[threadId] = kyz / tmp;
	                Kzz[threadId] = kzz / tmp;

	                ++show_progress;
	                if (CTRL_C_QUIT_FLAG) break;
	            }
	            if (CTRL_C_QUIT_FLAG) break;
	        }
	        if (CTRL_C_QUIT_FLAG) break;
	    }
	    double * KK[6]={Kxx,Kxy,Kxz,Kyy,Kyz,Kzz};

	    double * d_K_array[6]={Kxx_d, Kxy_d, Kxz_d, Kyy_d, Kyz_d, Kzz_d};
		for (int i = 0; i <6; i++)
		{
			checkCudaErrors(cudaMemcpy(d_K_array[i], KK[i], sizeof(double) * meshsize_padded,
											cudaMemcpyHostToDevice));
		}

	    delete Kxx,Kxy,Kxz,Kyy,Kyz,Kzz;

}


