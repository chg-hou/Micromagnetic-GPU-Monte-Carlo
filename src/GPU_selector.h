/*
 * GPU_selector.h
 *
 *  Created on: 14 Nov, 2016
 *      Author: cg
 */

#ifndef GPU_SELECTOR_H_
#define GPU_SELECTOR_H_

inline std::string getCfgFilename(int argc, const char **argv)
{
	for(int i=1; i<argc;i++)
	{
		if(argv[i][0]!='-')
			return std::string(argv[i]);
	}
	return "";
}
inline std::string getTensorCachePath(int argc,const char ** argv)
{
	if (checkCmdLineFlag(argc, argv, "tensorcache"))
	{
		char *cache;
		if(getCmdLineArgumentString(argc,argv,"tensorcache",&cache))
		{
			std::cout<<"cache path: "<<cache<<std::endl;
			return std::string(cache);
		}
		else
			return "";
	}
	return "";
}
inline int getCudaDevice(int argc, const char **argv)
{
    cudaDeviceProp deviceProp;
    int devID = 0;

    // If the command-line has a device number specified, use it
    if (checkCmdLineFlag(argc, argv, "gpu"))
    {
        devID = getCmdLineArgumentInt(argc, argv, "gpu=");

        if (devID < 0)
        {
            printf("Invalid command line parameter: -gpu=\n ");
            exit(EXIT_FAILURE);
        }
        else
        {
            devID = gpuDeviceInit(devID);

            if (devID < 0)
            {
                printf("exiting...\n");
                exit(EXIT_FAILURE);
            }

        }
    }
    else
    {
        // Otherwise pick the device with highest Gflops/s
        devID = gpuGetMaxGflopsDeviceId();
    }

    checkCudaErrors(cudaSetDevice(devID));
	checkCudaErrors(cudaGetDeviceProperties(&deviceProp, devID));

	printf("GPU Device %d: \"%s\" with compute capability %d.%d, multiProcessor: %d,\n", devID, deviceProp.name, deviceProp.major, deviceProp.minor,deviceProp.multiProcessorCount);
	printf(" total Global Mem: %ldMB, clock Rate: %dMHz, \n", deviceProp.totalGlobalMem/1024/1024,deviceProp.clockRate/1024);
	printf(" singleToDoublePrecisionPerfRatio: %d.\n\n", deviceProp.singleToDoublePrecisionPerfRatio);

    return devID;
}


#endif /* GPU_SELECTOR_H_ */
