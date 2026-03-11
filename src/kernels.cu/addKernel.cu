#include "device_launch_parameters.h"  
#include "../../include/kernels.cuh/addKernel.cuh"

__global__ void addKernel(float *c, const float *a, const float *b, unsigned int vectorSize, unsigned int n)
{
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    int index;
    int indexB;
    if(i < vectorSize/n ){
        for (int j = 0; j < n; j++){
            index = i + j * vectorSize/n;
            indexB =  (index / n + 1) * n - 1 - (index % n);
            c[index] = a[index] + b[indexB]; 
        }
    }
}

cudaError_t addWithCuda(float *c, const float *a, const float *b, unsigned int size,unsigned int ThreadsPerBlock, unsigned int n)
{
    float *dev_a = 0, *dev_b = 0, *dev_c = 0;
    cudaError_t cudaStatus;
    int numBlocks = (size/n + ThreadsPerBlock - 1)/ThreadsPerBlock;
    cudaStatus = cudaSetDevice(0);

    cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(float));
    cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(float));
    cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(float));

    cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(float), cudaMemcpyHostToDevice);
    cudaStatus = cudaMemcpy(dev_b, b, size * sizeof(float), cudaMemcpyHostToDevice);

    addKernel<<<numBlocks, ThreadsPerBlock>>>(dev_c, dev_a, dev_b,size,n);

    cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(dev_c);
    cudaFree(dev_a);
    cudaFree(dev_b);

    return cudaStatus;
}