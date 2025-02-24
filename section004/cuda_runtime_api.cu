#include <stdio.h>
//we dont nned to
int main(){
    int nDevices;

    cudaGetDeviceCount(&nDevices);//count of devices of our system

    for(int i=0;i<nDevices;i++){
        cudaDeviceProp prop;// variable to store the properties
        cudaGetDeviceProperties( &prop,i);//read all the available properties
        printf("Device number: %d \n",i);
        printf("Device name: %s \n",prop.name);//porp is a structure thta is why we use "." 
        printf("Memory clock rate (KHZ) : %d \n",prop.memoryClockRate);
        printf("Memory bus width (bits): %d\n",prop.memoryBusWidth);
        printf("Peak memory banwidth (Gbits/s): %f \n",2.0*prop.memoryClockRate*prop.memoryBusWidth/8.0/1.0e6);// this metric is very important and we'll need it when comparing the GPU performance for some applications
        printf("Total global memory: %lu \n",prop.totalGlobalMem);
        printf("Compute capability: %d.%d \n",prop.major,prop.minor);//explained in the second section
        printf("Number of Ss: %d \n",prop.multiProcessorCount);
        printf("Max threads per block: %d \n",prop.maxThreadsPerBlock);
        printf("Max threads dimensions: x= %d, y= %d, z=%d \n",prop.maxThreadsDim[0],prop.maxThreadsDim[1],prop.maxThreadsDim[2]);
        printf("Max grid dimensions: x= %d, y=%d, z=%d \n",prop.maxGridSize[0],prop.maxGridSize[1],prop.maxGridSize[3]);

    }
    return 0;

}