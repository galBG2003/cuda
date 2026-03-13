#pragma once
#include "cuda_runtime.h"

__global__ void addKernelSections23(float *c, const float *a, const float *b,unsigned int vectorSize);
__global__ void addKernelSections45(float *c, const float *a, const float *b,unsigned int vectorSize);
__global__ void addKernelSections6(float *c, const float *a, const float *b,unsigned int vectorSize);
__global__ void addKernelSections7(float *c, const float *a, const float *b,unsigned int vectorSize,int unsigned n);
__global__ void addKernelSections8(float *c, const float *a, const float *b,unsigned int vectorSize,int unsigned n);
