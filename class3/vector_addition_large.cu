#include "cuda_runtime.h"
#include <stdio.h>

#define SIZE 2048 //define the size of vectors
// CUDA kernel for vetcor addition. 
__global__ void vectorAdd(int* A, int* B, int* C, int n) // n varibale is the size but it is not used here
{
    int i=threadIdx.x+blockIdx.x*blockDim.x;
        C[i]=A[i]+B[i];
}
int main() //main function
{
    int* A,* B,* C; //alloctae some pointers for host (CPU)
    int* d_A,* d_B,* d_C; //allocate pointers for device (GPU)
    int size=SIZE*sizeof(int); // size in bytes for the arrays
    // Alloctae host vectors
    A=(int*)malloc(size);
    B=(int*)malloc(size);
    C=(int*)malloc(size);
    // Alloctae device vectors
    cudaMalloc((void**)&d_A,size);
    cudaMalloc((void**)&d_B,size);
    cudaMalloc((void**)&d_C,size);
    // Initialize vetcor A and B
    for(int i=0; i<SIZE; i++){
        A[i]=i;
        B[i]=SIZE-i;

    }
    // copy vectors from host to device
    cudaMemcpy(d_A,A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B,B, size, cudaMemcpyHostToDevice);
    // launch the vectorAdd CUDA kernel
    vectorAdd <<<2, 1024>>> (d_A, d_B, d_C, SIZE);//the kernel is called here

    //copy the result back to CPU
    cudaMemcpy(C,d_C,size, cudaMemcpyDeviceToHost); 
    printf("execution finished");
    for(int i; i<SIZE; i++){
        printf("%d + %d = %d \n",A[i],B[i],C[i]);
    }
    // free memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(A);
    free(B);
    free(C);

    // here we determine 1 thread and 1 block, so the code prints block ID 0 and ThreadID 0
    // Syntax: kernel_name <<< number of blocks, number of threads per block >>> ();
    // when we use number of blocks=2 and number of threads per block 8, the block ID=1 is executed first 
    cudaDeviceSynchronize();//this make the CPU wait for the GPU to finsh runnning the applictaion before continuing. If you comment this the output might not show.
    return 0; 
}
