/*
 * link  :     cufft cublas curand boost_system boost_filesystem
 *               // must link boost_system before boost_filesystem    http://stackoverflow.com/questions/9723793/undefined-reference-to-boostsystemsystem-category-when-compiling
 * 				-lboost_system -lboost_filesystem
 * include: /usr/local/cuda-8.0/samples/common/inc/
 * c++11:   -std=c++11 -stdlib=libc++     //needed by boost     //http://stackoverflow.com/questions/19469887/segmentation-fault-with-boostfilesystem
 *
 * -I/usr/local/cuda-8.0/samples/common/inc/ -G -g -O0 -std=c++11
 *
 * environment COPTS   = '-g -O0'  to disable optim
 */


/*
 *    /home/cg/cuda-workspace/abc.ini -gpu=0  -tensorcache="/home/cg/__tensor_cache/"
 *    abc.ini -gpu=0  -tensorcache="/home/cg/__tensor_cache/"
 */

// windows copy along with boost_system-vc140-mt-1_62
// include path :     E:\local\boost_1_62_0\         C:\ProgramData\NVIDIA Corporation\CUDA Samples\v8.0\common\inc
//					compute_60,sm_60
// linker:  addtion path :  E:\local\boost_1_62_0\lib64-msvc-14.0;%(AdditionalLibraryDirectories)
// input cufft.lib;cublas.lib;curand.lib;boost_system-vc140-mt-1_62.lib;boost_filesystem-vc140-mt-1_62.lib;%(AdditionalDependencies)
// TODO: cufft plan many
#include "demag_cu.h"
#include "constant.h"
#include "effective_H.h"
#include "mc_sampling.h"
#include "mx_init.h"
#include "ovf_io.h"
#include "terminal_print.h"
#include "parse_input_ini.h"
#include "GPU_selector.h"
#include "demag_mirror.h"

#include "device_launch_parameters.h"


#include <boost/filesystem.hpp>
#include <boost/filesystem/path.hpp>
#include <boost/filesystem/operations.hpp>
#include <boost/progress.hpp>

#include <cuda_runtime.h>
#include <curand.h>
#include <cublas_v2.h>
#include <helper_functions.h>
#include <helper_cuda.h>

#include <ctime>
#include <time.h>
#include <stdio.h>
#include <iostream>
#include <math.h>
#include <cufft.h>
#include <fstream>

#include <string>


#include <chrono>
#include <thread>
#include <thrust/device_ptr.h>
#include <thrust/reduce.h>

#ifdef _WIN32
        
#else
     #include <unistd.h>
#endif

#ifdef DEBUG_H_FIELD
	#include "mc_sampling_debug.h"
#endif

#include <signal.h>
#include <stdlib.h>

#include <timer.h>


bool CTRL_C_QUIT_FLAG = false;
void ctrl_c_handler(int s)
{
	printf("Caught signal %d\n",s);
	CTRL_C_QUIT_FLAG = true;
	return;
}

int main(int argc, char *argv[]) {



	//================================
	using namespace std;
	setbuf(stdout,NULL);
	if (argc==1)
	{
		cout<<"need input file. \n";
		cout<<"arg format:  input_cfg_file.ini  -gpu=0   -tensorcache=\"/mnt/cachepath/\" \n";
		return 0;
	}

	string input_cfg_filename = getCfgFilename( argc,(const char **) argv);
	getCudaDevice( argc, (const char **) argv);

//=========================init para=======
	int  nx; int  ny; int  nz;
	double  dx;double  dy;double  dz;

	FLOAT_   Bextx;FLOAT_   Bexty;FLOAT_   Bextz;

	int  pbc_x; int  pbc_y; int  pbc_z;

	FLOAT_   ms; FLOAT_   Aex;FLOAT_   Dind;

	FLOAT_   anisUx; FLOAT_   anisUy;FLOAT_   anisUz;FLOAT_   Ku1;

	bool   use_random_init;
	long long   randomseed;
	string   ovf_filename;

	FLOAT_   Temperature_start; FLOAT_   Temperature_end;
	FLOAT_   Temperature_step;
	bool Temperature_use_exp;

	long   circle_per_stage ;
	long   terminal_output_period;
	long   energy_output_period;
	long   ms_output_period;

	int   rand_block_size;
	bool cal_demag_flag;

	//-----------------------------

	//============catch ctrl-c=================
	signal(SIGINT, &ctrl_c_handler);


	//----------------------------------------




	boost::filesystem::path input_cfg_filename_path(input_cfg_filename);
	if (!boost::filesystem::exists(input_cfg_filename_path))
		input_cfg_filename_path = boost::filesystem::current_path() / input_cfg_filename_path ;
	input_cfg_filename_path = boost::filesystem::absolute(input_cfg_filename_path);
	std::cout << "input_cfg_path is : "<<input_cfg_filename_path<<endl;
	input_cfg_filename = input_cfg_filename_path.string();

	if (!parse_input_ini(input_cfg_filename,
			  nx,   ny,   nz,
			  dx,  dy,  dz,
			   Bextx,   Bexty,   Bextz,
			  pbc_x,   pbc_y,   pbc_z,
			   ms,    Aex,   Dind,
			   anisUx,    anisUy,   anisUz,   Ku1,
			  use_random_init,
			    randomseed,
			ovf_filename,
			   Temperature_start,    Temperature_end,
			   Temperature_step,
			   circle_per_stage ,
			   terminal_output_period,
			   energy_output_period,
			   ms_output_period,
			   rand_block_size,
			   cal_demag_flag,
			   Temperature_use_exp))
		return 0;
	cout_input_ini(
			  nx,   ny,   nz,
			  dx,  dy,  dz,
			   Bextx,   Bexty,   Bextz,
			  pbc_x,   pbc_y,   pbc_z,
			   ms,    Aex,   Dind,
			   anisUx,    anisUy,   anisUz,   Ku1,
			   use_random_init,
			    randomseed,
			   ovf_filename,
			   Temperature_start,    Temperature_end,
			   Temperature_step,
			   circle_per_stage ,
			   terminal_output_period,
			   energy_output_period,
			   ms_output_period,
			   rand_block_size,
			   cal_demag_flag,
			   Temperature_use_exp);

	boost::filesystem::path output_path =
			(input_cfg_filename_path.parent_path()/input_cfg_filename_path.stem()).string()
			+ ".output";	;//boost::filesystem::path(".\\energy");
	if (! boost::filesystem::exists(output_path) )
	{
		if(boost::filesystem::create_directories(output_path)) {
				std::cout << "Successfully create output folder: "<< output_path << "\n";
			}
		else
		{
			std::cout << "Fail to create output folder: "<< output_path << "\n";
			return 0;
		}
	}else
	{
		namespace fs = boost::filesystem;
		fs::directory_iterator end_iter;
		for (fs::directory_iterator dir_itr(output_path);
		          dir_itr != end_iter;
		          ++dir_itr)
		    {
		      try
		      {
		    	  std::string extension = dir_itr->path().extension().string();
		    	  if (extension==".ovf" || extension==".png"|| extension==".jpg"
		    			  || extension==".jpeg")
		    	  {
		    		  fs::remove(dir_itr->path());
		    	  }

		      }
		      catch (const std::exception & ex)
		      {
		        std::cout << dir_itr->path().filename() << " " << ex.what() << std::endl;
		      }
		    }

	}
	cout<<"output_path: "<<output_path<<endl;

	string energy_filename_string = (output_path  / boost::filesystem::path("energy.txt")).string();//output_path ;// / boost::filesystem::path("energy.txt"); //output_path.string() ;
	cout<<"energy_filename_string: "<<energy_filename_string<<endl;
	{
		FILE * fp;
		fp = fopen(energy_filename_string.c_str(),"w");
		fclose(fp);
	}


	FLOAT_ Temperature = Temperature_start;

	FLOAT_ mu0 = 4e-7 * CONST_PI;
	FLOAT_ factor_H_exch = 2.0 * Aex / mu0 / ms;
	if (randomseed<0)
	{
		randomseed = time(NULL);
		cout<<"Use random seed: "<<randomseed<<endl;
	}



	bool padding_x = (pbc_x == 0);
	bool padding_y = (pbc_y == 0);
	bool padding_z = (pbc_z == 0);

	int nx_padded = padding_x ? (2 * nx - 1) : nx;
	int ny_padded = padding_y ? (2 * ny - 1) : ny;
	int nz_padded = padding_z ? (2 * nz - 1) : nz;

	long meshsize = nx * ny * nz;
	long meshsize_padded = nx_padded * ny_padded * nz_padded;

	//===============
	ovf_io::OvfInfo ovfinfo;
	ovfinfo.xnodes = nx;
	ovfinfo.ynodes = ny;
	ovfinfo.znodes = nz;
	ovfinfo.xstepsize = dx;
	ovfinfo.ystepsize = dy;
	ovfinfo.zstepsize = dz;
	//===============
	//const int blocksize = 10;
	int blockDim_x = 5;
	int blockDim_y = 5;
	int blockDim_z = 1;

	dim3 dimBlock(blockDim_x, blockDim_y, blockDim_z);
	dim3 dimGrid((nx_padded + blockDim_x - 1) / blockDim_x,
			(ny_padded + blockDim_y - 1) / blockDim_y,
			(nz_padded + blockDim_z - 1) / blockDim_z);

	dim3 dimBlock_nopadding(blockDim_x, blockDim_y, blockDim_z);
	dim3 dimGrid_nopadding((nx + blockDim_x - 1) / blockDim_x,
			(ny + blockDim_y - 1) / blockDim_y,
			(nz + blockDim_z - 1) / blockDim_z);

	/*==========================load Mx ===========================*/
	FLOAT_ * h_Mx_padded;
	FLOAT_ * h_My_padded;
	FLOAT_ * h_Mz_padded;
	{
		//
//		double h_Mx[meshsize];
//		double h_My[meshsize];
//		double h_Mz[meshsize];
		double * h_Mx = new double[meshsize];
		double * h_My = new double[meshsize];
		double * h_Mz = new double[meshsize];

#ifdef USE_ONE_HOT_MX_INIT
		for(int i=0;i<meshsize;i++)
		{
			h_Mx[i]=0;
			h_My[i]=0;
			h_Mz[i]=0;
		}
		h_Mx[0]=1;
#else
		if (use_random_init)
		{
			//randInitMx(T * Mx, T * My, T * Mz, int nx, int ny, int nz, unsigned long long seed = 1234L )
			randInitMx( h_Mx,  h_My,  h_Mz,  nx, ny, nz, randomseed);

		}
		else
		{
			if (!boost::filesystem::exists(boost::filesystem::path(ovf_filename)))
				{cout<<"OVF file not found: "<<ovf_filename<<endl;return 0;}
			if (!(ovf_io::parseOvfFile<double>(ovf_filename, ovfinfo, h_Mx,  h_My,  h_Mz)))
				{cout<<"Error, quit now.";return 0;}
		}
#endif
		h_Mx_padded = new FLOAT_[meshsize_padded];
		h_My_padded = new FLOAT_[meshsize_padded];
		h_Mz_padded = new FLOAT_[meshsize_padded];

		copyMxtoMxpadded<double,FLOAT_>(h_Mx, h_My, h_Mz,  nx,  ny,  nz,
				h_Mx_padded, h_My_padded, h_Mz_padded, nx_padded, ny_padded, nz_padded);
		delete h_Mx,h_My,h_Mz;
	}

	/*--------------------------load Mx ---------------------------*/



	/*==========================K tensor for pbc=========*/
	double Kxx_mirror, Kyy_mirror, Kzz_mirror;
	double Kxy_mirror, Kxz_mirror, Kyz_mirror;
	demag_mirror::calDemagMirrorTensor( nx_padded,  ny_padded,  nz_padded,
			 pbc_x,  pbc_y,  pbc_z,
			 dx, dy ,  dz,
			 Kxx_mirror, Kxy_mirror, Kxz_mirror,
			 Kyy_mirror, Kyz_mirror, Kzz_mirror
	);


	/*-----------------------K tensor for pbc-----------------------*/

	/*=======================init random block================*/
	curandGenerator_t randgen;
	FLOAT_ * dRandPool;
	FLOAT_ * dRandPool_reject;
	int randPoolCount = rand_block_size;
	//cout<<sizeof(FLOAT_) * 3 * meshsize *  rand_block_size ;
	checkCudaErrors(
			cudaMalloc((void ** )&dRandPool, sizeof(FLOAT_) * 3 * meshsize *  rand_block_size ));
	checkCudaErrors(
			cudaMalloc((void ** )&dRandPool_reject, sizeof(FLOAT_) * meshsize *  rand_block_size ));
	CURAND_CALL(curandCreateGenerator(&randgen, CURAND_RNG_PSEUDO_DEFAULT));
//	CURAND_RNG_PSEUDO_DEFAULT
//	CURAND_RNG_PSEUDO_XORWOW
//	CURAND_RNG_PSEUDO_MRG32K3A
//	CURAND_RNG_PSEUDO_MTGP32
//	CURAND_RNG_PSEUDO_MT19937
//	CURAND_RNG_PSEUDO_PHILOX4_32_10
//	CURAND_RNG_QUASI_DEFAULT
//	CURAND_RNG_QUASI_SOBOL32
//	CURAND_RNG_QUASI_SCRAMBLED_SOBOL32
//	CURAND_RNG_QUASI_SOBOL64
//	CURAND_RNG_QUASI_SCRAMBLED_SOBOL64

	CURAND_CALL(curandSetPseudoRandomGeneratorSeed(randgen, randomseed));
//	 /* Generate n floats on device */
//#ifdef USE_DOULBE_IN_GPU
//	CURAND_CALL(curandGenerateNormalDouble(randgen, dRandPool, 3 * meshsize * rand_block_size,
//				0.0,1.0));	//double mean, double stddev
//	CURAND_CALL(curandGenerateUniformDouble(randgen, dRandPool_reject,  meshsize *  rand_block_size));
//#else
//	CURAND_CALL(curandGenerateNormal(randgen, dRandPool, 3 * meshsize * rand_block_size,
//			0.0f,1.0f));	//float mean, float stddev
//	// excluding 0.0 and including 1.0      ( 0.0 , 1.0]
//	CURAND_CALL(curandGenerateUniform(randgen, dRandPool_reject, meshsize * rand_block_size));
//#endif
//	/*------------ init random block           */


	/* ================================init Kxx fft==================================*/
	cout << endl << "Dimension: nx ny nz meshsize";
	cout << endl << nx << " " << ny << " " << nz <<" "<<meshsize<< endl;
	cout << "nx_padded ny_padded nz_padded meshsize_padded";
	cout << endl << nx_padded << " " << ny_padded << " " << nz_padded <<" "<<meshsize_padded<< endl;

	cout<<"Calculating demag tensor...\n";
	double * Kxx, *Kxy, *Kxz, *Kyy, *Kyz, *Kzz;

	checkCudaErrors(
			cudaMalloc((void ** )&Kxx, sizeof(double) * meshsize_padded));
	checkCudaErrors(
			cudaMalloc((void ** )&Kxy, sizeof(double) * meshsize_padded));
	checkCudaErrors(
			cudaMalloc((void ** )&Kxz, sizeof(double) * meshsize_padded));
	checkCudaErrors(
			cudaMalloc((void ** )&Kyy, sizeof(double) * meshsize_padded));
	checkCudaErrors(
			cudaMalloc((void ** )&Kyz, sizeof(double) * meshsize_padded));
	checkCudaErrors(
			cudaMalloc((void ** )&Kzz, sizeof(double) * meshsize_padded));






//	ktensorKernel<<<dimGrid, dimBlock>>>(nx, ny, nz, dx, dy, dz, Kxx, Kxy, Kxz,
//			Kyy, Kyz, Kzz);

	string tensor_cache_filename;
	{
		char filename_buff[200];
		snprintf(filename_buff,200,"cache_%d_%d_%d-%g_%g_%g-%d_%d_%d.ovf",nx_padded, ny_padded, nz_padded, dx,
				dy, dz, pbc_x, pbc_y, pbc_z);
		tensor_cache_filename = filename_buff;
	}
	bool succefully_load_tensor_cache = false;
	if(boost::filesystem::exists(boost::filesystem::path(tensor_cache_path) / boost::filesystem::path(tensor_cache_filename) ))
	{
		//load
		cout<<"  loading cached tensor... "<<boost::filesystem::path(tensor_cache_path) / boost::filesystem::path(tensor_cache_filename) ;
		ifstream ifs( (boost::filesystem::path(tensor_cache_path) / boost::filesystem::path(tensor_cache_filename)).string(),
				ios::binary);
		if (ifs)
		{
			double * h_K_tmp = new double[meshsize_padded];
			try
			{

				double * d_K_array[6]={Kxx, Kxy, Kxz, Kyy, Kyz, Kzz};
				for (int i = 0; i <6; i++)
				{
					ifs.read(reinterpret_cast<char *>(h_K_tmp), sizeof(double) * meshsize_padded );
					checkCudaErrors(cudaMemcpy(d_K_array[i], h_K_tmp, sizeof(double) * meshsize_padded,
													cudaMemcpyHostToDevice));
				}
				succefully_load_tensor_cache = true;
				cout<<"  Done.\n";

			}
			catch(const std::exception &e)
			{succefully_load_tensor_cache = false;cout<<"  ERROR.\n";};
			delete h_K_tmp;
		}
		ifs.close();
	}
	if (!succefully_load_tensor_cache)
	{
		//cal
		{
			cout<<"		Calculating in real space ...";

#ifdef USE_CPU_TO_CALCULATE_K_TENSOR
			ktensorcpu::ktensorCPU(nx_padded, ny_padded, nz_padded, dx,
					dy, dz, pbc_x, pbc_y, pbc_z, Kxx, Kxy, Kxz, Kyy, Kyz, Kzz,CTRL_C_QUIT_FLAG);
#else
			ktensorKernel2_step0_init<<<dimGrid, dimBlock>>>(nx_padded, ny_padded, nz_padded,
					Kxx, Kxy, Kxz, Kyy, Kyz, Kzz);

			{
				printf("....................");

//				float count = 0;
				long total = (2*max(pbc_x,0)+1)*(2*max(pbc_y,0)+1)*(2*max(pbc_z,0)+1);
				boost::progress_display show_progress( total );
				for (int pbc_idx_x = -max(pbc_x,0) ; pbc_idx_x <= max(pbc_x,0); pbc_idx_x++) {
					for (int pbc_idx_y = -max(pbc_y,0); pbc_idx_y <= max(pbc_y,0); pbc_idx_y++) {
						for (int pbc_idx_z = -max(pbc_z,0); pbc_idx_z <= max(pbc_z,0); pbc_idx_z++) {

							if (CTRL_C_QUIT_FLAG) break;
//							printf("\b\b\b\b\b\b\b\b\b\b\b\b...% 3.1f %% ...",(count++)*100.0/total);

	//						ktensorKernel<<<dimGrid, dimBlock>>>(nx_padded, ny_padded, nz_padded, dx,
	//											dy, dz, pbc_x, pbc_y, pbc_z, Kxx, Kxy, Kxz, Kyy, Kyz, Kzz, d_data);

							ktensorKernel2_step1<<<dimGrid, dimBlock>>>(nx_padded, ny_padded, nz_padded, dx,
									dy, dz,  Kxx, Kxy, Kxz, Kyy, Kyz, Kzz,
									pbc_idx_x, pbc_idx_y, pbc_idx_z);
							++show_progress;
							cudaDeviceSynchronize();
						}
					}
				}
//				printf("\b\b\b\b\b\b\b\b\b\b\b\b...100%% ...\n");
//				cout<<count<<"  "<<total<<endl;
				printf("   Done.\n");
			}
			ktensorKernel2_step2_div_const<<<dimGrid, dimBlock>>>(nx_padded, ny_padded, nz_padded,
					dx, dy, dz,
					Kxx, Kxy, Kxz, Kyy, Kyz, Kzz);

//			volatile int * d_data, * h_data;
//			cudaHostAlloc((void **)& h_data,sizeof(int), cudaHostAllocMapped);
//			cudaHostGetDevicePointer((int **) &d_data, (int *) h_data,0);
//			* h_data = 0;
//
////			cudaEvent_t start,stop;
////			cudaEventCreate(&start);cudaEventCreate(&stop);
////			cudaEventRecord(start);
//
//			ktensorKernel<<<dimGrid, dimBlock>>>(nx_padded, ny_padded, nz_padded, dx,
//					dy, dz, pbc_x, pbc_y, pbc_z, Kxx, Kxy, Kxz, Kyy, Kyz, Kzz, d_data);
//
//
////			cudaEventRecord(stop);
//
//			unsigned int num_blocks = dimGrid.x * dimGrid.y;
//			float my_progress = 0.0f;
//
//	#ifndef  _WIN32
//				do{
//					//boost::this_thread::sleep(boost::posix_time::milliseconds(100));
//					std::this_thread::sleep_for(std::chrono::microseconds(100000));
//					//usleep(100000);
//					int value1 = *h_data;
//					float kern_progress = (float)value1 /(float) num_blocks * ((2*max(pbc_x,0)+1)* (2*max(pbc_y,0)+1) * (2*max(pbc_z,1)+1));
//					if((kern_progress-my_progress)>0.05f)
//					{
//						printf("...%3.1f %% ...\n", kern_progress*100.0);
//
//						my_progress= kern_progress;
//					}
//				}while (my_progress<0.99f);
//	#endif
#endif
			cout<<"Done.\n";

			if (pbc_x<0) pbc_x=1;
			if (pbc_y<0) pbc_y=1;
			if (pbc_z<0) pbc_z=1;
			printf("pbc_x = %d, pbc_y = %d, pbc_z = %d\n",pbc_x,pbc_y,pbc_z);
//			cudaEventSynchronize(stop);
//			float et;
//			cudaEventElapsedTime(&et, start, stop);
//			cudaDeviceSynchronize();
//			printf("Elaspsed time = %f ms", et);
		}
		//save
		if(boost::filesystem::exists(boost::filesystem::path(tensor_cache_path) ))
		{
			string cache_filename = (boost::filesystem::path(tensor_cache_path) / boost::filesystem::path(tensor_cache_filename)).string();
			cout<<"Write tensor to cache: "<<cache_filename<<endl;
			ofstream ofs( cache_filename,
							ios::binary);
			if (ofs)
			{
				try
				{
					double *  h_K_tmp = new double[meshsize_padded];
					double * d_K_array[6]={Kxx, Kxy, Kxz, Kyy, Kyz, Kzz};
					for (int i = 0; i <6; i++)
					{
						checkCudaErrors(cudaMemcpy( h_K_tmp,d_K_array[i], sizeof(double) * meshsize_padded,
																						cudaMemcpyDeviceToHost));
						ofs.write(reinterpret_cast<char *>(h_K_tmp), sizeof(double) * meshsize_padded );
					}
					delete h_K_tmp;
				}
				catch(const std::exception &e)
				{cout<<"Fail to write cached tensor.\n";};
			}
			ofs.close();
		}
	}
;

#ifdef DEBUG_H_FIELD
	ovf_io::writetoOvfFile_device<double>(
						(output_path  / boost::filesystem::path("Kxx.ovf")).string(),
						ovfinfo,Kxx,Kyy,Kzz);
	ovf_io::writetoOvfFile_device<double>(
							(output_path  / boost::filesystem::path("Kxy.ovf")).string(),
							ovfinfo,Kxy,Kxz,Kyz);
#endif

#ifdef DISPLAY_VERBOSE_FFT
	double * KK[6] = { Kxx, Kxy, Kxz, Kyy, Kyz, Kzz };
	print_real<double>("K tensor ", KK, 6, nx_padded, ny_padded, nz_padded);
#endif
	//K tensor fft
	double2 * Kxx_fft_64, *Kxy_fft_64, *Kxz_fft_64, *Kyy_fft_64, *Kyz_fft_64,
			*Kzz_fft_64;
	FLOAT2 * Kxx_fft, *Kxy_fft, *Kxz_fft, *Kyy_fft, *Kyz_fft, *Kzz_fft;
	cufftHandle plan_D2Z;
	cout<<"		Calculating FFT ...";
	cufftPlan3d(&plan_D2Z, nx_padded, ny_padded, nz_padded, CUFFT_D2Z);
	cout<<"plan done...";
	checkCudaErrors(
			cudaMalloc((void ** )&Kxx_fft_64,
					sizeof(double2) * meshsize_padded));
	checkCudaErrors(
			cudaMalloc((void ** )&Kxx_fft, sizeof(FLOAT2) * meshsize_padded));

	checkCudaErrors(
			cudaMalloc((void ** )&Kxy_fft_64,
					sizeof(double2) * meshsize_padded));
	checkCudaErrors(
			cudaMalloc((void ** )&Kxy_fft, sizeof(FLOAT2) * meshsize_padded));

	checkCudaErrors(
			cudaMalloc((void ** )&Kxz_fft_64,
					sizeof(double2) * meshsize_padded));
	checkCudaErrors(
			cudaMalloc((void ** )&Kxz_fft, sizeof(FLOAT2) * meshsize_padded));

	checkCudaErrors(
			cudaMalloc((void ** )&Kyy_fft_64,
					sizeof(double2) * meshsize_padded));
	checkCudaErrors(
			cudaMalloc((void ** )&Kyy_fft, sizeof(FLOAT2) * meshsize_padded));

	checkCudaErrors(
			cudaMalloc((void ** )&Kyz_fft_64,
					sizeof(double2) * meshsize_padded));
	checkCudaErrors(
			cudaMalloc((void ** )&Kyz_fft, sizeof(FLOAT2) * meshsize_padded));

	checkCudaErrors(
			cudaMalloc((void ** )&Kzz_fft_64,
					sizeof(double2) * meshsize_padded));
	checkCudaErrors(
			cudaMalloc((void ** )&Kzz_fft, sizeof(FLOAT2) * meshsize_padded));

	//TODO: stream these things
	cudaDeviceSynchronize();


	cufftExecD2Z(plan_D2Z, Kxx, Kxx_fft_64);cout<<"1 done...";
	cufftExecD2Z(plan_D2Z, Kxy, Kxy_fft_64);cout<<"2 done...";
	cufftExecD2Z(plan_D2Z, Kxz, Kxz_fft_64);cout<<"3 done...";
	cufftExecD2Z(plan_D2Z, Kyy, Kyy_fft_64);cout<<"4 done...";
	cufftExecD2Z(plan_D2Z, Kyz, Kyz_fft_64);cout<<"5 done...";
	cufftExecD2Z(plan_D2Z, Kzz, Kzz_fft_64);cout<<"6 done...\n";

	checkCudaErrors(cudaFree(Kxx));
	checkCudaErrors(cudaFree(Kxy));
	checkCudaErrors(cudaFree(Kxz));
	checkCudaErrors(cudaFree(Kyy));
	checkCudaErrors(cudaFree(Kyz));
	checkCudaErrors(cudaFree(Kzz));

	cout<<"		Copying ...";
	// copy K_fft_64 to K_fft (32)
	cudaDeviceSynchronize();
	copyFFT64to32Kernel<<<dimGrid, dimBlock>>>(nx_padded, ny_padded, nz_padded,
			Kxx_fft_64, Kxy_fft_64, Kxz_fft_64, Kyy_fft_64, Kyz_fft_64,
			Kzz_fft_64, Kxx_fft, Kxy_fft, Kxz_fft, Kyy_fft, Kyz_fft, Kzz_fft);
	cudaDeviceSynchronize();

	checkCudaErrors(cudaFree(Kxx_fft_64));
	checkCudaErrors(cudaFree(Kxy_fft_64));
	checkCudaErrors(cudaFree(Kxz_fft_64));
	checkCudaErrors(cudaFree(Kyy_fft_64));
	checkCudaErrors(cudaFree(Kyz_fft_64));
	checkCudaErrors(cudaFree(Kzz_fft_64));

	FLOAT2 *h_Kxx_fft = new FLOAT2[meshsize_padded]; //new FLOAT2[sizeof(FLOAT2) * meshsize_padded];

#ifdef DISPLAY_VERBOSE_FFT
	FLOAT2 * KK_fft[6] =
				{ Kxx_fft, Kxy_fft, Kxz_fft, Kyy_fft, Kyz_fft, Kzz_fft };
	print_complex<FLOAT2>("K tensor fft ", KK_fft, 6, nx_padded, ny_padded,
			nz_padded);
#endif
	delete h_Kxx_fft;
	cout<<"Done.\n";
	/* --------------------------  init Kxx fft-----------------------------*/

	/*=============================init Mx , Mx fft========================*/
//	float * Mx, * My, * Mz;
	FLOAT_ * Mx_padded, *My_padded, *Mz_padded;
	FLOAT_ * Mx_padded_output, *My_padded_output, *Mz_padded_output;

	FLOAT2 * Mx_padded_fft, *My_padded_fft, *Mz_padded_fft;

//	FLOAT_ * Hx_padded, *Hy_padded, *Hz_padded;
	FLOAT2 * Hx_padded_fft, *Hy_padded_fft, *Hz_padded_fft;

	checkCudaErrors(
			cudaMalloc((void ** )&Mx_padded, sizeof(FLOAT_) * meshsize_padded));
	checkCudaErrors(
			cudaMalloc((void ** )&My_padded, sizeof(FLOAT_) * meshsize_padded));
	checkCudaErrors(
			cudaMalloc((void ** )&Mz_padded, sizeof(FLOAT_) * meshsize_padded));
	checkCudaErrors(
			cudaMalloc((void ** )&Mx_padded_output, sizeof(FLOAT_) * meshsize_padded));
	checkCudaErrors(
			cudaMalloc((void ** )&My_padded_output, sizeof(FLOAT_) * meshsize_padded));
	checkCudaErrors(
			cudaMalloc((void ** )&Mz_padded_output, sizeof(FLOAT_) * meshsize_padded));
	checkCudaErrors(
			cudaMalloc((void ** )&Mx_padded_fft, sizeof(FLOAT2) * meshsize_padded));
	checkCudaErrors(
			cudaMalloc((void ** )&My_padded_fft, sizeof(FLOAT2) * meshsize_padded));
	checkCudaErrors(
			cudaMalloc((void ** )&Mz_padded_fft, sizeof(FLOAT2) * meshsize_padded));

	checkCudaErrors(
			cudaMalloc((void ** )&Hx_padded_fft, sizeof(FLOAT2) * meshsize_padded));
	checkCudaErrors(
			cudaMalloc((void ** )&Hy_padded_fft, sizeof(FLOAT2) * meshsize_padded));
	checkCudaErrors(
			cudaMalloc((void ** )&Hz_padded_fft, sizeof(FLOAT2) * meshsize_padded));
	/*----------------------------init Mx , Mx fft----------------------------------------*/

	//========================copy Mx to GPU================



	cudaMemcpy(Mx_padded, h_Mx_padded, sizeof(FLOAT_) * meshsize_padded,
			cudaMemcpyHostToDevice);
	cudaMemcpy(My_padded, h_My_padded, sizeof(FLOAT_) * meshsize_padded,
			cudaMemcpyHostToDevice);
	cudaMemcpy(Mz_padded, h_Mz_padded, sizeof(FLOAT_) * meshsize_padded,
			cudaMemcpyHostToDevice);
	cudaMemcpy(Mx_padded_output, h_Mx_padded, sizeof(FLOAT_) * meshsize_padded,
			cudaMemcpyHostToDevice);
	cudaMemcpy(My_padded_output, h_My_padded, sizeof(FLOAT_) * meshsize_padded,
			cudaMemcpyHostToDevice);
	cudaMemcpy(Mz_padded_output, h_Mz_padded, sizeof(FLOAT_) * meshsize_padded,
			cudaMemcpyHostToDevice);
	delete h_Mx_padded, h_My_padded, h_Mz_padded;
	//-----------------------copy Mx to GPU------------------

	//=====================init energy=================
	FLOAT_ * d_energy;
	checkCudaErrors(
				cudaMalloc((void ** )&d_energy, sizeof(FLOAT_) * meshsize));


	/*=====================init  FFT plan=======================*/
	cout<<"Initializing FFT...";
#ifdef USE_DOULBE_IN_GPU
//	cufftHandle plan_D2Z;
//	cufftPlan3d(&plan_D2Z, nx_padded, ny_padded, nz_padded, CUFFT_D2Z);
	cufftHandle plan_Z2D;
	cufftPlan3d(&plan_Z2D, nx_padded, ny_padded, nz_padded, CUFFT_Z2D);
	cufftHandle plan_Z2Z;
	cufftPlan3d(&plan_Z2Z, nx_padded, ny_padded, nz_padded, CUFFT_Z2Z);
#else
	cufftHandle plan_R2C;
	cufftHandle plan_C2R;
	cufftPlan3d(&plan_R2C, nx_padded, ny_padded, nz_padded, CUFFT_R2C);
	cufftPlan3d(&plan_C2R, nx_padded, ny_padded, nz_padded, CUFFT_C2R);
	cufftHandle plan_C2C;
	cufftPlan3d(&plan_C2C, nx_padded, ny_padded, nz_padded, CUFFT_C2C);
#endif

	/*-----------------init  FFT plan-----------------------*/

#ifdef DISPLAY_VERBOSE_FFT
	FLOAT_ * handle_Mx_padded[3] = { Mx_padded, My_padded, Mz_padded };
	FLOAT2 * handle_M_fft_padded[3] = { Mx_padded_fft, My_padded_fft,
				Mz_padded_fft };
	FLOAT2 * handle_H_fft_padded[3] = { Hx_padded_fft, Hy_padded_fft,
				Hz_padded_fft };
	FLOAT_ * d_energy_p []= {d_energy};
#endif
	cout<<"Done.\n";


#ifdef DEBUG_H_FIELD
	FLOAT_ * d_debug_Hx, * d_debug_Hy, * d_debug_Hz;
	checkCudaErrors(
				cudaMalloc((void ** )&d_debug_Hx, sizeof(FLOAT_) * meshsize_padded));
	checkCudaErrors(
					cudaMalloc((void ** )&d_debug_Hy, sizeof(FLOAT_) * meshsize_padded));
	checkCudaErrors(
					cudaMalloc((void ** )&d_debug_Hz, sizeof(FLOAT_) * meshsize_padded));
#endif

	// ==================================================================================================================
	// ==================================================================================================================
	// ==================================================================================================================
	// ======================================      where game begins   ==================================================
	// ==================================================================================================================
	// ==================================================================================================================
	// ==================================================================================================================
	cout<<"Start...\n";

	// ========= elapsed time====
//	std::chrono::time_point<std::chrono::system_clock> time_start, time_end;		//Method 'count' could not be resolved
//	std::chrono::duration<double> stage_elapsed_seconds;// = time_end - time_start;
//	time_start = std::chrono::system_clock::now();

	clock_t time_start;//, time_end;
	double stage_elapsed_seconds;
	//------------------------

	long output_file_count = 0;
	char filename_buff[200];
	while ( Temperature > Temperature_end)

	{
		printf("\n\nTemperature: %.2f K\n",Temperature);
		for (long stage=0; stage <circle_per_stage;stage++ )		//	FOR(stage,0,circle_per_stage)
		{
			if (CTRL_C_QUIT_FLAG) break;
			if (stage%terminal_output_period ==0 )
			{
				printf("Stage: %ld",stage);
				if (stage ==0)
				{
//					time_start = std::chrono::system_clock::now();
					time_start = clock();
					printf(" ");
				}
				else
				{
//					std::chrono::duration<double> vasdfa;// = time_end - time_start;
//					time_end = std::chrono::system_clock::now();
//					stage_elapsed_seconds = time_end - time_start;
					//printf(", elapsed time: %f s, speed: %f stages/s ",(stage_elapsed_seconds.count()), (stage_elapsed_seconds.count()/stage) );
					stage_elapsed_seconds = (double)(clock()-time_start)/CLOCKS_PER_SEC;
					printf(", elapsed time: %f s, speed: %f stages/s ",stage_elapsed_seconds, (stage/stage_elapsed_seconds) );

				}
			}
//			long energy_output_period = 1000;
//			long ms_output_period = 1000;
			if (stage % ms_output_period == 0) {
				{
//				double h_Mx[meshsize];
//				double h_My[meshsize];
//				double h_Mz[meshsize];
				double * h_Mx= new double[meshsize];
				double * h_My= new double[meshsize];
				double * h_Mz= new double[meshsize];

				h_Mx_padded = new FLOAT_[meshsize_padded];
				h_My_padded = new FLOAT_[meshsize_padded];
				h_Mz_padded = new FLOAT_[meshsize_padded];
				cudaMemcpy(h_Mx_padded,Mx_padded,
						sizeof(FLOAT_) * meshsize_padded,
						cudaMemcpyDeviceToHost);
				cudaMemcpy( h_My_padded,My_padded,
						sizeof(FLOAT_) * meshsize_padded,
						cudaMemcpyDeviceToHost);
				cudaMemcpy(h_Mz_padded,Mz_padded,
						sizeof(FLOAT_) * meshsize_padded,
						cudaMemcpyDeviceToHost);
				copyMxpaddedtoMx<FLOAT_,double>(
						h_Mx_padded,  h_My_padded, h_Mz_padded,  nx_padded,  ny_padded,  nz_padded,
						h_Mx,  h_My,  h_Mz,  nx,  ny,  nz);

				snprintf(filename_buff,200,"%06ld_%.2fK_%ld.ovf",output_file_count, Temperature, stage );
				std::string filename = filename_buff;
				filename =  (output_path / boost::filesystem::path(filename)).string();
				ovf_io::writetoOvfFile<double>( filename,ovfinfo,	h_Mx,  h_My,  h_Mz);

				delete h_Mx_padded, h_My_padded, h_Mz_padded;

				output_file_count++;
				delete h_Mx,h_My,h_Mz;
				}

			}

			if(cal_demag_flag)
			{
			#ifdef USE_DOULBE_IN_GPU
				cufftExecD2Z(plan_D2Z, Mx_padded, Mx_padded_fft);
				cufftExecD2Z(plan_D2Z, My_padded, My_padded_fft);
				cufftExecD2Z(plan_D2Z, Mz_padded, Mz_padded_fft);
			#else
				cufftExecR2C(plan_R2C, Mx_padded, Mx_padded_fft);
				cufftExecR2C(plan_R2C, My_padded, My_padded_fft);
				cufftExecR2C(plan_R2C, Mz_padded, Mz_padded_fft);
			#endif
				cudaDeviceSynchronize();
				calHexchangeKernel<<<dimGrid, dimBlock>>>(nx_padded, ny_padded, nz_padded,
						Mx_padded_fft, My_padded_fft, Mz_padded_fft, Kxx_fft, Kxy_fft,
						Kxz_fft, Kyy_fft, Kyz_fft, Kzz_fft, Hx_padded_fft, Hy_padded_fft,
						Hz_padded_fft);
				cudaDeviceSynchronize();

			#ifdef DISPLAY_VERBOSE_FFT
				print_real<FLOAT_>("M padded ", handle_Mx_padded, 3, nx_padded, ny_padded,
						nz_padded, 0);
				print_complex<FLOAT2>("M padded FFT ", handle_M_fft_padded, 3, nx_padded,
						ny_padded, nz_padded, 0);
				print_complex<FLOAT2>("H_fft_ (M_fft*K_fft) ", handle_H_fft_padded, 3, nx_padded,
						ny_padded, nz_padded, 0);
			#endif


			#ifdef DISPLAY_VERBOSE_FFT
				print_complex<FLOAT2>("H_fft_INVERSE ", handle_H_fft_padded, 3, nx_padded,
						ny_padded, nz_padded, 0);
			#endif
			}

			if (stage%energy_output_period == 0)
			{
				// =========================== cal exchange field energy=========
						cudaDeviceSynchronize();
//						effectiveHKernel<<<dimGrid_nopadding, dimBlock_nopadding>>>( ms, factor_H_exch, Dind,
//									Ku1, anisUx, anisUy, anisUz,
//									Bextx, Bexty, Bextz,
//									nx, ny, nz,
//									nx_padded, ny_padded, nz_padded, dx, dy, dz,
//									pbc_x, pbc_y, pbc_z,
//									Mx_padded, My_padded, Mz_padded,
//									Hx_padded_fft, Hy_padded_fft, Hz_padded_fft,
//									d_energy, cal_demag_flag);
						calEnergyKernel<<<dimGrid_nopadding, dimBlock_nopadding>>>( ms,  mu0, factor_H_exch,  Aex,  Dind,
											 Ku1,  anisUx,  anisUy,  anisUz,
											 Bextx,  Bexty,  Bextz,
											 nx,  ny,  nz,
											 nx_padded,  ny_padded,  nz_padded,  dx,  dy,
											 dz,  pbc_x,  pbc_y,  pbc_z,
											  Mx_padded, My_padded, Mz_padded,
											 Hx_padded_fft, Hy_padded_fft, Hz_padded_fft,
											 d_energy,  cal_demag_flag) ;
						cudaDeviceSynchronize();
//						print_real<FLOAT_>("M padded ", handle_Mx_padded, 3, nx_padded, ny_padded,
//											nz_padded, 0);
						#ifdef DISPLAY_VERBOSE_FFT
//							print_complex<FLOAT2>("H demag + i * H exch ", handle_H_fft_padded, 3,
//									nx_padded, ny_padded, nz_padded, 0);
						// DISPALY_H_demag_H_demag_
							print_complex_real_nopadding<FLOAT2>("H demag ", handle_H_fft_padded, 3,
															nx_padded, ny_padded, nz_padded,
															nx,ny,nz,0);
							print_complex_imag_nopadding<FLOAT2>("H exch ", handle_H_fft_padded, 3,
															nx_padded, ny_padded, nz_padded,
															nx,ny,nz,0);
							print_real<FLOAT_>("Energy ", d_energy_p, 1, nx, ny, nz,0);
						#endif
//							FLOAT_ * handle_Mx_padded[3] = { Mx_padded, My_padded, Mz_padded };
//							print_real<FLOAT_>("M padded ", handle_Mx_padded, 3, nx_padded, ny_padded,
//												nz_padded, 0);
//							FLOAT_ * d_energy_p []= {d_energy};
//							print_real<FLOAT_>("Energy ", d_energy_p, 1, nx, ny, nz,0);
							//cout<<" thrust reduce "<<endl;
							thrust::device_ptr<FLOAT_> d_sum_ptr = thrust::device_pointer_cast(d_energy);
//							FLOAT_ tot_energy = thrust::reduce(d_sum_ptr, d_sum_ptr + meshsize);
							FLOAT_ tot_energy = thrust::reduce(d_sum_ptr, d_sum_ptr + meshsize)*dx*dy*dz;
							printf("\nEnergy:  %g\n", tot_energy);

							{
								FILE *fp;

								fp = fopen(energy_filename_string.c_str(), "a");
								fprintf(fp,"%f %ld %g\n",Temperature,stage,tot_energy);
								fclose(fp);
							}



			}

			//====================================main Monte ======================================


#ifdef DEBUG_H_FIELD
			FLOAT_ *  h_debug_H[3] = {d_debug_Hx, d_debug_Hy, d_debug_Hz};
			cout<<"###########################DEBUG_B_demag##############\n";
			mcKernel_debug_demag<<<dimGrid_nopadding, dimBlock_nopadding>>>(
								 ms,  mu0,  Aex,  Dind,
								 Ku1,  anisUx,  anisUy,  anisUz,
								 Bextx, Bexty, Bextz,
								 (FLOAT_)(CONST_K_BOLTZMANN * Temperature),
								 nx,  ny,  nz,  nx_padded,  ny_padded,
								 nz_padded,  meshsize,  dx,  dy,  dz,  pbc_x,  pbc_y,  pbc_z,
								  Kxx_mirror,  Kyy_mirror,   Kzz_mirror,
								 Mx_padded, My_padded, Mz_padded,
								 Mx_padded_output, My_padded_output, Mz_padded_output,
								 Hx_padded_fft, Hy_padded_fft, Hz_padded_fft,
								 dRandPool,  dRandPool_reject,  randPoolCount, cal_demag_flag,
								 d_debug_Hx, d_debug_Hy, d_debug_Hz);
//			print_real<FLOAT_>("B demag", h_debug_H, 3,
//					nx_padded, ny_padded, nz_padded, 0);
			print_real_along_col<FLOAT_>("B demag", d_debug_Hx,d_debug_Hy,d_debug_Hz, 3,
								nx_padded, ny_padded, nz_padded, 0);
			ovf_io::writetoOvfFile_device<FLOAT_>(
					(output_path  / boost::filesystem::path("B_demag.ovf")).string(),
					ovfinfo,	d_debug_Hx,d_debug_Hy,d_debug_Hz);
			//cout<<"###########################DEBUG_H_FIELD##############\n";
			cout<<"###########################DEBUG_B_anis##############\n";
			mcKernel_debug_anis<<<dimGrid_nopadding, dimBlock_nopadding>>>(
								 ms,  mu0,  Aex,  Dind,
								 Ku1,  anisUx,  anisUy,  anisUz,
								 Bextx, Bexty, Bextz,
								 (FLOAT_)(CONST_K_BOLTZMANN * Temperature),
								 nx,  ny,  nz,  nx_padded,  ny_padded,
								 nz_padded,  meshsize,  dx,  dy,  dz,  pbc_x,  pbc_y,  pbc_z,
								  Kxx_mirror,  Kyy_mirror,   Kzz_mirror,
								 Mx_padded, My_padded, Mz_padded,
								 Mx_padded_output, My_padded_output, Mz_padded_output,
								 Hx_padded_fft, Hy_padded_fft, Hz_padded_fft,
								 dRandPool,  dRandPool_reject,  randPoolCount, cal_demag_flag,
								 d_debug_Hx, d_debug_Hy, d_debug_Hz);
//			print_real<FLOAT_>("H anis", h_debug_H, 3,
//					nx_padded, ny_padded, nz_padded, 0);
			print_real_along_col<FLOAT_>("B anis", d_debug_Hx,d_debug_Hy,d_debug_Hz, 3,
											nx_padded, ny_padded, nz_padded, 0);
			ovf_io::writetoOvfFile_device<FLOAT_>(
					(output_path  / boost::filesystem::path("B_anis.ovf")).string(),
					ovfinfo,d_debug_Hx,d_debug_Hy,d_debug_Hz);
			cout<<"###########################DEBUG_B_exch##############\n";
			mcKernel_debug_exch<<<dimGrid_nopadding, dimBlock_nopadding>>>(
											 ms,  mu0,  Aex,  Dind,
											 Ku1,  anisUx,  anisUy,  anisUz,
											 Bextx, Bexty, Bextz,
											 (FLOAT_)(CONST_K_BOLTZMANN * Temperature),
											 nx,  ny,  nz,  nx_padded,  ny_padded,
											 nz_padded,  meshsize,  dx,  dy,  dz,  pbc_x,  pbc_y,  pbc_z,
											  Kxx_mirror,  Kyy_mirror,   Kzz_mirror,
											 Mx_padded, My_padded, Mz_padded,
											 Mx_padded_output, My_padded_output, Mz_padded_output,
											 Hx_padded_fft, Hy_padded_fft, Hz_padded_fft,
											 dRandPool,  dRandPool_reject,  randPoolCount, cal_demag_flag,
											 d_debug_Hx, d_debug_Hy, d_debug_Hz);
//			print_real<FLOAT_>("B_exch", h_debug_H, 3,
//					nx_padded, ny_padded, nz_padded, 0);
			print_real_along_col<FLOAT_>("B exch", d_debug_Hx,d_debug_Hy,d_debug_Hz, 3,
														nx_padded, ny_padded, nz_padded, 0);
			ovf_io::writetoOvfFile_device<FLOAT_>(
					(output_path  / boost::filesystem::path("B_exch.ovf")).string(),
					ovfinfo,d_debug_Hx,d_debug_Hy,d_debug_Hz);
			cout<<"###########################DEBUG_B_effective##############\n";
			mcKernel_debug_total<<<dimGrid_nopadding, dimBlock_nopadding>>>(
											 ms,  mu0,  Aex,  Dind,
											 Ku1,  anisUx,  anisUy,  anisUz,
											 Bextx, Bexty, Bextz,
											 (FLOAT_)(CONST_K_BOLTZMANN * Temperature),
											 nx,  ny,  nz,  nx_padded,  ny_padded,
											 nz_padded,  meshsize,  dx,  dy,  dz,  pbc_x,  pbc_y,  pbc_z,
											  Kxx_mirror,  Kyy_mirror,   Kzz_mirror,
											 Mx_padded, My_padded, Mz_padded,
											 Mx_padded_output, My_padded_output, Mz_padded_output,
											 Hx_padded_fft, Hy_padded_fft, Hz_padded_fft,
											 dRandPool,  dRandPool_reject,  randPoolCount, cal_demag_flag,
											 d_debug_Hx, d_debug_Hy, d_debug_Hz);
			print_real_along_col<FLOAT_>("B_effective", d_debug_Hx,d_debug_Hy,d_debug_Hz, 3,
					nx_padded, ny_padded, nz_padded, 0);
			ovf_io::writetoOvfFile_device<FLOAT_>(
					(output_path  / boost::filesystem::path("B_effective.ovf")).string(),
					ovfinfo,d_debug_Hx,d_debug_Hy,d_debug_Hz);
			cout<<"###########################DEBUG_E_density##############\n";
			calEnergyKernel<<<dimGrid_nopadding, dimBlock_nopadding>>>( ms,  mu0, factor_H_exch,  Aex,  Dind,
					 Ku1,  anisUx,  anisUy,  anisUz,
					 Bextx,  Bexty,  Bextz,
					 nx,  ny,  nz,
					 nx_padded,  ny_padded,  nz_padded,  dx,  dy,
					 dz,  pbc_x,  pbc_y,  pbc_z,
					  Mx_padded, My_padded, Mz_padded,
					 Hx_padded_fft, Hy_padded_fft, Hz_padded_fft,
					 d_energy,  cal_demag_flag) ;
			print_real_along_col<FLOAT_>("E_density", d_energy,d_energy,d_energy, 3,
					nx_padded, ny_padded, nz_padded, 0);
			ovf_io::writetoOvfFile_device<FLOAT_>(
					(output_path  / boost::filesystem::path("E_density.ovf")).string(),
					ovfinfo,d_energy,d_energy,d_energy);
			cout<<"###########################DEBUG_mc_E_density##############\n";
			mcKernel2<<<dimGrid_nopadding, dimBlock_nopadding>>>(
								 ms,  mu0,  Aex,  Dind,
								 Ku1,  anisUx,  anisUy,  anisUz,
								 Bextx, Bexty, Bextz,
								 (FLOAT_)(CONST_K_BOLTZMANN * Temperature),
								 nx,  ny,  nz,  nx_padded,  ny_padded,
								 nz_padded,  meshsize,  dx,  dy,  dz,  pbc_x,  pbc_y,  pbc_z,
								 Kxx_mirror, Kxy_mirror, Kxz_mirror,
								 Kyy_mirror, Kyz_mirror, Kzz_mirror,
								 Mx_padded, My_padded, Mz_padded,
								 Mx_padded_output, My_padded_output, Mz_padded_output,
								 Hx_padded_fft, Hy_padded_fft, Hz_padded_fft,
								 dRandPool,  dRandPool_reject,  randPoolCount, cal_demag_flag,
								 d_energy);
			print_real_along_col<FLOAT_>("mc_E_density", d_energy,d_energy,d_energy, 3,
					nx_padded, ny_padded, nz_padded, 0);
			ovf_io::writetoOvfFile_device<FLOAT_>(
					(output_path  / boost::filesystem::path("mc_E_density.ovf")).string(),
					ovfinfo,d_energy,d_energy,d_energy);
			cout<<"###########################DEBUG_mc_E_density##############\n";
#endif

//			mcKernel<<<dimGrid_nopadding, dimBlock_nopadding>>>(
//					 ms,  mu0,  Aex,  Dind,
//					 Ku1,  anisUx,  anisUy,  anisUz,
//					 Bextx, Bexty, Bextz,
//					 (FLOAT_)(CONST_K_BOLTZMANN * Temperature),
//					 nx,  ny,  nz,  nx_padded,  ny_padded,
//					 nz_padded,  meshsize,  dx,  dy,  dz,  pbc_x,  pbc_y,  pbc_z,
//					  Kxx_mirror,  Kyy_mirror,   Kzz_mirror,
//					 Mx_padded, My_padded, Mz_padded,
//					 Mx_padded_output, My_padded_output, Mz_padded_output,
//					 Hx_padded_fft, Hy_padded_fft, Hz_padded_fft,
//					 dRandPool,  dRandPool_reject,  randPoolCount, cal_demag_flag);
			// TODO: CONST_K_BOLTZMANN * Temperature * 3
			mcKernel2<<<dimGrid_nopadding, dimBlock_nopadding>>>(
											 ms,  mu0,  Aex,  Dind,
											 Ku1,  anisUx,  anisUy,  anisUz,
											 Bextx, Bexty, Bextz,
											 (FLOAT_)(CONST_K_BOLTZMANN * Temperature),
											 nx,  ny,  nz,  nx_padded,  ny_padded,
											 nz_padded,  meshsize,  dx,  dy,  dz,  pbc_x,  pbc_y,  pbc_z,
											 Kxx_mirror, Kxy_mirror, Kxz_mirror,
											 Kyy_mirror, Kyz_mirror, Kzz_mirror,
											 Mx_padded, My_padded, Mz_padded,
											 Mx_padded_output, My_padded_output, Mz_padded_output,
											 Hx_padded_fft, Hy_padded_fft, Hz_padded_fft,
											 dRandPool,  dRandPool_reject,  randPoolCount, cal_demag_flag,
											 d_energy);
			cudaDeviceSynchronize();
			randPoolCount++;

			swap(Mx_padded,Mx_padded_output);
			swap(My_padded,My_padded_output);
			swap(Mz_padded,Mz_padded_output);

			if (CTRL_C_QUIT_FLAG) break;
		}
		//stage end

		if (CTRL_C_QUIT_FLAG) break;
	// ==================================  update Temperature ================

		if (Temperature_use_exp)
		{
			Temperature =  Temperature * Temperature_step;
		}
		else
		{
			Temperature =  Temperature - Temperature_step;
		}


	}

	//----------------------------------------------------------------------------------------------------------------
	//----------------------------------------------------------------------------------------------------------------
	//----------------------------------------------------------------------------------------------------------------
	//--------------------------------------------      running end   ------------------------------------------
	//----------------------------------------------------------------------------------------------------------------
	//----------------------------------------------------------------------------------------------------------------
	//----------------------------------------------------------------------------------------------------------------

//	int x, y, z;
//	int shifted_x, shifted_y, shifted_z;
//	shifted_x = (x + nx_padded / 2 - 1) % nx_padded;
//	shifted_y = (x + ny_padded / 2 - 1) % ny_padded;
//	shifted_z = (x + nz_padded / 2 - 1) % nz_padded;

//	cout << "----------------" << endl;
	//!!!   Hx_padded = Hx_padded/meshsize_padded;
//	cudaMemcpy(h_float_padded, Hx_padded, sizeof(FLOAT_) * meshsize_padded,
//			cudaMemcpyDeviceToHost);
//	cout << "--------Hx_padded--------" << endl;
//	for (int kk = 0; kk < nz_padded; kk++) {
//		cout << "nz = " << kk << endl;
//		for (int ii = 0; ii < nx_padded; ii++) {
//			for (int jj = 0; jj < ny_padded; jj++)
////		 				cout << h_Hx[kk + jj * nz_padded + ii * nz_padded * ny_padded]/meshsize_padded
////		 						<< " ";
//				printf("%.4f ",
//						h_float_padded[kk + jj * nz_padded
//								+ ii * nz_padded * ny_padded]
//								/ meshsize_padded);
//			cout << endl;
//		}
//	}
//	cout << "----------------" << endl;

//	cout << "----------------" << endl;
	//!!!   Hx_padded = Hx_padded/meshsize_padded;
//	cudaMemcpy(h_float_padded, Hx_padded, sizeof(FLOAT_) * meshsize_padded,
//			cudaMemcpyDeviceToHost);
//	cout << "--------Hx_padded shifted--------" << endl;
//	for (int kk = 0; kk < nz_padded; kk++) {
//		cout << "nz = " << kk << endl;
//		for (int ii = 0; ii < nx_padded; ii++) {
//			for (int jj = 0; jj < ny_padded; jj++) {
//				int x = (ii + nx_padded / 2 - 1) % nx_padded;
//				int y = (jj + ny_padded / 2 - 1) % ny_padded;
//				int z = (kk + nz_padded / 2 - 1) % nz_padded;
//				printf("%.4f ",
//						h_float_padded[z + y * nz_padded
//								+ x * nz_padded * ny_padded] / meshsize_padded);
//			}
//			cout << endl;
//		}
//	}
//	cout << "----------------" << endl;


	//=========================  clean field==========================


	checkCudaErrors(cudaFree(Kxx_fft));
	checkCudaErrors(cudaFree(Kxy_fft));
	checkCudaErrors(cudaFree(Kxz_fft));
	checkCudaErrors(cudaFree(Kyy_fft));
	checkCudaErrors(cudaFree(Kyz_fft));
	checkCudaErrors(cudaFree(Kzz_fft));

//	float * Mx_padded, * My_padded, * Mz_padded;
//	float2 * Mx_padded_fft, * My_padded_fft,* Mz_padded_fft;
	checkCudaErrors(cudaFree(Mx_padded));
	checkCudaErrors(cudaFree(My_padded));
	checkCudaErrors(cudaFree(Mz_padded));
	checkCudaErrors(cudaFree(Mx_padded_output));
	checkCudaErrors(cudaFree(My_padded_output));
	checkCudaErrors(cudaFree(Mz_padded_output));
	checkCudaErrors(cudaFree(Mx_padded_fft));
	checkCudaErrors(cudaFree(My_padded_fft));
	checkCudaErrors(cudaFree(Mz_padded_fft));
//	float * Hx_padded, * Hy_padded, * Hz_padded;
//	float2 * Hx_padded_fft, * Hy_padded_fft,* Hz_padded_fft;

//	checkCudaErrors(cudaFree(Hx_padded));
//	checkCudaErrors(cudaFree(Hy_padded));
//	checkCudaErrors(cudaFree(Hz_padded));

	checkCudaErrors(cudaFree(Hx_padded_fft));
	checkCudaErrors(cudaFree(Hy_padded_fft));
	checkCudaErrors(cudaFree(Hz_padded_fft));

	checkCudaErrors(cudaFree(dRandPool));
	CURAND_CALL(curandDestroyGenerator(randgen));

	checkCudaErrors(cudaFree(d_energy));


	cufftDestroy(plan_D2Z);

//	cufftDestroy(plan_Z2Z);
#ifdef USE_DOULBE_IN_GPU
	cufftDestroy(plan_Z2D);
	cufftDestroy(plan_Z2Z);
#else
	cufftDestroy(plan_R2C);
	cufftDestroy(plan_C2R);
	cufftDestroy(plan_C2C);
#endif


#ifdef DEBUG_H_FIELD
	checkCudaErrors(cudaFree(d_debug_Hx));
	checkCudaErrors(cudaFree(d_debug_Hy));
	checkCudaErrors(cudaFree(d_debug_Hz));
#endif




//	cuFloatComplex a,b;
//	a.x=1;a.y=1;
//	b.x =3;b.y=-1;
//	cuDoubleComplex c;
//	c = cuCmul(a,b);
//	cout<<c.x<<endl;
//	cout<<c.y<<endl;
	cout<<"~~~~~~~~~~~~~~END~~~~~~~~~~~~~~~~~~~~~~~~~ \n";
	return 0;
}
