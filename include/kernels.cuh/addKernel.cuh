#pragma once
#include "cuda_runtime.h"

__global__ void addKernel(float *c, const float *a, const float *b);
cudaError_t addWithCuda(float *c, const float *a, const float *b, unsigned int size);