// here we use the CUDA API to measure performance
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <device_launch_parameters.h>

#define TOTAL_SIZE 1024*1024*1024 //define the size of vectors
#define CHUNK_SIZE 1024*1024*128 // Elements per chunk, adjust based on available host memory. Each vector divided into 8 pieces
#define BLOCK_SIZE 1024 // number of threads per block (depend on GPU capabilities)
// CUDA kernel for vetcor addition. 
__global__ void vectorAdd(int* A, int* B, int* C, int chunk_size) // n varibale is the size but it is not used here
{
    int i=threadIdx.x+blockIdx.x*blockDim.x;
    if(i< chunk_size){ // ensure we are insite the bounds 
        C[i]=A[i]+B[i];
    }
}
void random_ints(int* x, int size)
{
    for(int i=0; i<size;i++)
    {
      x[i]=rand()%100;  
    }
}
int main() //main function
{
    int* chunk_a,* chunk_b,* chunk_c; //alloctae some pointers for host chunks (CPU)
    int* d_A,* d_B,* d_C; //allocate pointers for device (GPU)
    size_t chunkSizeBytes= CHUNK_SIZE*sizeof(int); // size in bytes for the arrays chunks

    // Alloctae host vectors
    chunk_a=(int*)malloc(chunkSizeBytes);
    chunk_b=(int*)malloc(chunkSizeBytes);
    chunk_c=(int*)malloc(chunkSizeBytes);
    // Alloctae device vectors
    cudaMalloc((void**)&d_A,chunkSizeBytes);
    cudaMalloc((void**)&d_B,chunkSizeBytes);
    cudaMalloc((void**)&d_C,chunkSizeBytes);
    // calculate the number of blocks for the kernel
    int numBlocks=(CHUNK_SIZE+BLOCK_SIZE-1)/BLOCK_SIZE;
    for(long long offset=0; offset<TOTAL_SIZE;offset+=CHUNK_SIZE){
        int current_chunk_size= (TOTAL_SIZE-offset)<CHUNK_SIZE ? (TOTAL_SIZE-offset):CHUNK_SIZE;
        printf("\n Offset %lld \n",offset);
        random_ints(chunk_a,current_chunk_size);
        random_ints(chunk_b,current_chunk_size);

        cudaMemcpy(d_A,chunk_a, current_chunk_size*sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_B,chunk_b, current_chunk_size*sizeof(int), cudaMemcpyHostToDevice);


        vectorAdd <<<numBlocks, BLOCK_SIZE>>> (d_A, d_B, d_C,current_chunk_size);//the kernel is called here
    
        cudaMemcpy(chunk_c,d_C,current_chunk_size*sizeof(int), cudaMemcpyDeviceToHost);
    }
    // copy vectors from host to device
    //copy the result back to CPU

    for(int i; i<10; i++){
        printf("%d + %d = %d \n",chunk_a[i],chunk_b[i],chunk_c[i]);
    }
    // free memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(chunk_a);
    free(chunk_b);
    free(chunk_c);

    // here we determine 1 thread and 1 block, so the code prints block ID 0 and ThreadID 0
    // Syntax: kernel_name <<< number of blocks, number of threads per block >>> ();
    // when we use number of blocks=2 and number of threads per block 8, the block ID=1 is executed first 
    return 0; 
}
