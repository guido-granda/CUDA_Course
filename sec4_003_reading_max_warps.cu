#include <cuda_runtime.h>
#include <stdio.h>

int main() {
    int device;
    cudaGetDevice(&device); // Get current CUDA device. Provides the total number of NVIDIA GPUs available
    cudaDeviceProp prop;

    cudaGetDeviceProperties(&prop, device); // we dont iterate here because there is only one GPU
    printf("Max_threads_per_SM  0: %d \n", prop.maxThreadsPerMultiProcessor);
    printf("Max_warps_per_SM    0: %d \n\n\n", (prop.maxThreadsPerMultiProcessor)/32);



    //another method to get this information
    int maxThreadsPerMP = 0;
    cudaDeviceGetAttribute(&maxThreadsPerMP, cudaDevAttrMaxThreadsPerMultiProcessor, device);// write the attribute name sthat you want to read. chech documentation for details
    printf("Max_threads_per_SM  1: %d   \n", maxThreadsPerMP);
    printf("Max_warps_per_SM    1: %d   \n", maxThreadsPerMP/32);

    return 0;
}
