#include "device_launch_parameters.h"  
#include "../../include/kernels.cuh/addKernel.cuh"

__global__ void addKernel(float *c, const float *a, const float *b, unsigned int vectorSize)
{
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    if(i < vectorSize/2 ){
        c[i] = ((i & 1) == 0) * b[i] * b[i] + ((i & 1) == 1) * (a[i] + b[i]);
        c[i + vectorSize/2] = ((i & 1) == 0) * b[i + vectorSize/2] * b[i + vectorSize/2] + (((i + vectorSize/2) & 1) == 1) * (a[i + vectorSize/2] + b[i + vectorSize/2]);
    }
        
    
}

cudaError_t addWithCuda(float *c, const float *a, const float *b, unsigned int size,unsigned int ThreadsPerBlock)
{
    float *dev_a = 0, *dev_b = 0, *dev_c = 0;
    cudaError_t cudaStatus;
    int numBlocks = (size/2 + ThreadsPerBlock - 1)/ThreadsPerBlock;
    cudaStatus = cudaSetDevice(0);

    cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(float));
    cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(float));
    cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(float));

    cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(float), cudaMemcpyHostToDevice);
    cudaStatus = cudaMemcpy(dev_b, b, size * sizeof(float), cudaMemcpyHostToDevice);

    addKernel<<<numBlocks, ThreadsPerBlock>>>(dev_c, dev_a, dev_b,size);

    cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(dev_c);
    cudaFree(dev_a);
    cudaFree(dev_b);

    return cudaStatus;
}