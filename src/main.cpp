#include "../include/kernels.cuh/addKernel.cuh"
#include "../include/utils.h"
#include <stdio.h>
#include <fstream>
#include <iostream>

int main()
{
    int vectorSize = 40 * 32;
    int numThreadsPerBlock = 256;
    std::vector<float> a = generateRandomVector(vectorSize);
    std::vector<float> valuesForB =  generateRandomVector(vectorSize) ;
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

    cudaError_t cudaStatus = addWithCuda(c.data(), a.data(), b.data(), a.size(), numThreadsPerBlock);
    if (cudaStatus != cudaSuccess) {
        std::cerr << "addWithCuda failed!" << std::endl;
        return 1;
    }
   
    printResult(a,b,c);

    try{
        writeResultToFile(c);
    }
    catch(const std::exception &e){
        std::cerr << e.what() << std::endl;
    }

    cudaDeviceReset();

    return 0;
}

