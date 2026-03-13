#pragma once
#include "cuda_runtime.h"

/**
 * @brief Manages GPU operations for kernels that accept four arguments.
 * @explanation This wrapper handles the full CUDA lifecycle: setting the device, allocating VRAM, 
 * copying input data to the GPU, launching the kernel, and retrieving the results.
 * @param c Pointer to the host array where the final result will be stored.
 * @param a Pointer to the first input host array (Vector A).
 * @param b Pointer to the second input host array (Vector B).
 * @param size Total number of elements in the vectors.
 * @param ThreadsPerBlock Number of threads per block (Block Dimension).
 * @param n How many operations is each thread is responsible for.
 * @param Kernel A function pointer to a __global__ kernel expecting (float*, float*, float*, unsigned int).
 * @return cudaError_t Returns cudaSuccess if all steps (malloc, memcpy, launch) succeeded.
 */
cudaError_t addWithCuda(float *c, const float *a, const float *b, unsigned int size, unsigned int ThreadsPerBlock, unsigned int n, void(*Kernel)(float*, const float*, const float*, unsigned int));

/**
 * @brief Manages GPU operations for kernels that accept five arguments, including an extra 'n' parameter.
 * @explanation Similar to the primary wrapper, but specifically designed for kernels that require the 'n' 
 * parameter passed directly into the GPU logic.
 * @param c Pointer to the host array where the final result will be stored.
 * @param a Pointer to the first input host array (Vector A).
 * @param b Pointer to the second input host array (Vector B).
 * @param size Total number of elements in the vectors.
 * @param ThreadsPerBlock Number of threads per block (Block Dimension).
 * @param n How many operations is each thread is responsible for.
 * @param Kernel A function pointer to a __global__ kernel expecting (float*, float*, float*, unsigned int, unsigned int).
 * @return cudaError_t Returns cudaSuccess if all steps succeeded.
 */
cudaError_t addWithCuda(float *c, const float *a, const float *b, unsigned int size, unsigned int ThreadsPerBlock, unsigned int n, void(*Kernel)(float*, const float*, const float*, unsigned int, unsigned int));