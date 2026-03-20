#pragma once
#include "cuda_runtime.h"

/**
 * @brief Runs a basic parallel addition where each thread handles one element.
 * @param c Result vector.
 * @param a First input vector.
 * @param b Second input vector.
 * @param vectorSize Total number of elements.
 */
__global__ void addKernelSections23(float *c, const float *a, const float *b, unsigned int vectorSize);

/**
 * @brief Calculates different math based on even/odd indices (uses bitwise & 1).
 * @explanation Even indices calculate b*b, odd indices calculate a+b.
 * @param c Result vector.
 * @param a First input vector.
 * @param b Second input vector.
 * @param vectorSize Total number of elements.
 */
__global__ void addKernelSections45(float *c, const float *a, const float *b, unsigned int vectorSize);

/**
 * @brief Each thread processes two elements: one in the first half and one in the second half(stride equals number of threads).
 * @param c Result vector.
 * @param a First input vector.
 * @param b Second input vector.
 * @param vectorSize Total number of elements.
 */
__global__ void addKernelSections6(float *c, const float *a, const float *b, unsigned int vectorSize);

/**
 * @brief Each thread runs a loop to process 'n' elements with stride equals to number of threads.
 * @param c Result vector.
 * @param a First input vector.
 * @param b Second input vector.
 * @param vectorSize Total number of elements.
 * @param n Number of elements each thread is responsible for.
 */
__global__ void addKernelSections7(float *c, const float *a, const float *b, unsigned int vectorSize, unsigned int n);

/**
 * @brief Runs complex addition where input 'b' is accessed in a reversed or mirrored pattern within chunks.
 * @param c Result vector.
 * @param a First input vector.
 * @param b Second input vector.
 * @param vectorSize Total number of elements.
 * @param n Number of elements each thread is responsible for and how mant elements are reversed.
 */
__global__ void addKernelSections8(float *c, const float *a, const float *b, unsigned int vectorSize, unsigned int n);