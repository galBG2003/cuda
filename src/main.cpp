#include "../include/kernels.cuh/addKernel.cuh"
#include "../include/utils.h"
#include <stdio.h>
#include <fstream>
#include <iostream>

int main()
{
    std::vector<float> a = generateRandomVector(5);
    std::vector<float> valuesForB = {10.0f, 20.0f, 30.0f, 40.0f, 50.0f};
    std::vector<float> b;
    try {
    checkValidInput(a, valuesForB);
    createBinaryFile("/home/test4/Desktop/git/cuda/vectorB.32f", valuesForB);
    b = readBinaryFile("/home/test4/Desktop/git/cuda/vectorB.32f");
    }
    catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return 1;
    }
       
    std::vector<float> c(a.size(), 0);

    cudaError_t cudaStatus = addWithCuda(c.data(), a.data(), b.data(), a.size());
    if (cudaStatus != cudaSuccess) {
        std::cerr << "addWithCuda failed!" << std::endl;
        return 1;
    }
   
    printResult(a,b,c);

    cudaDeviceReset();
    
    return 0;
}

