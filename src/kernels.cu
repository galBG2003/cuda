#include "../include/kernels.cuh"


__global__ void addKernelSections23(float *c, const float *a, const float *b,unsigned int vectorSize)
{
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    if(i < vectorSize)
        c[i] = a[i] + b[i];
}

__global__ void addKernelSections45(float *c, const float *a, const float *b, unsigned int vectorSize)
{
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    if(i < vectorSize )
        c[i] = ((i & 1) == 0) * b[i] * b[i] + ((i & 1) == 1) * (a[i] + b[i]);
}

__global__ void addKernelSections6(float *c, const float *a, const float *b, unsigned int vectorSize)
{
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    if(i < vectorSize/2 ){
        c[i] = ((i & 1) == 0) * b[i] * b[i] + ((i & 1) == 1) * (a[i] + b[i]);
        c[i + vectorSize/2] = ((i & 1) == 0) * b[i + vectorSize/2] * b[i + vectorSize/2] + (((i + vectorSize/2) & 1) == 1) * (a[i + vectorSize/2] + b[i + vectorSize/2]);
    }
}

__global__ void addKernelSections7(float *c, const float *a, const float *b, unsigned int vectorSize, unsigned int n)
{
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    int index;
    if(i < vectorSize/n ){
        for (int j = 0; j < n; j++){
            index = i + j * vectorSize/n;
            c[index] = ((index & 1) == 0) * b[index] * b[index] + (((index) & 1) == 1) * (a[index] + b[index]);
        }
    }
}

__global__ void addKernelSections8(float *c, const float *a, const float *b, unsigned int vectorSize, unsigned int n)
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