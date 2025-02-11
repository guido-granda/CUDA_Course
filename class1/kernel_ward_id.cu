#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>

// here we have 1 kernel and 1 function

// this is the kernel. 
__global__ void test01()
{
    //print the blocks and the threads IDs
    // ward = 32 threads, lets assume we  have 128 threads/block (change this in line 19: 128>>>) in our project, so 128/32 wards/block
    int ward_ID_value=0;
    ward_ID_value=threadIdx.x/32; 
    printf("\n The block ID is %d , the thread ID is %d , the ward ID is %d  \n", blockIdx.x,threadIdx.x,ward_ID_value);
    // here blockIdx takes .x because in cuda the cuda numbers can be distributed across dimensions x, y, z 
}
int main() //main function
{
    test01 <<<1, 128>>> ();//the kernel is called here
    //test01 <<<1, 64>>> ();//in this case we have 2 wards
    test01 <<<2, 64>>> ();//in this case we have 2 wards per block, so 4 wards in the GPU
    // here we determine 1 thread and 1 block, so the code prints block ID 0 and ThreadID 0
    // Syntax: kernel_name <<< number of blocks, number of threads per block >>> ();
    // when we use number of blocks=2 and number of threads per block 8, the block ID=1 is executed first 
    cudaDeviceSynchronize();//this make the CPU wait for the GPU to finsh runnning the applictaion before continuing. If you comment this the output might not show.
    return 0; 
}
