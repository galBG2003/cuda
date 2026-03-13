#pragma once
#include "cuda_runtime.h"


cudaError_t addWithCuda(float *c, const float *a, const float *b, unsigned int size,unsigned int ThreadsPerBlock, unsigned int n, void(*Kernel)(float*, const float*, const float*, unsigned int));
cudaError_t addWithCuda(float *c, const float *a, const float *b, unsigned int size,unsigned int ThreadsPerBlock, unsigned int n, void(*Kernel)(float*, const float*, const float*, unsigned int, unsigned int));
