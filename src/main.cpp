#include "../include/cudaFunctions.cuh"
#include "../include/kernels.cuh"
#include "../include/utils.h"
#include "cuda_runtime.h"

int main(){
    int n = 10;
    int vectorSize = 5 * 32 * n;
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
       
    std::vector<float> c1(a.size(), 0);
    std::vector<float> c2(a.size(), 0);
    std::vector<float> c3(a.size(), 0);
    std::vector<float> c4(a.size(), 0);
    std::vector<float> c5(a.size(), 0);
    cudaError_t statuses[] = {
        addWithCuda(c1.data(), a.data(), b.data(), a.size(),numThreadsPerBlock,1,addKernelSections23),
        addWithCuda(c2.data(), a.data(), b.data(), a.size(), numThreadsPerBlock,1,addKernelSections45),
        addWithCuda(c3.data(), a.data(), b.data(), a.size(), numThreadsPerBlock,2,addKernelSections7),
        addWithCuda(c4.data(), a.data(), b.data(), a.size(), numThreadsPerBlock,n,addKernelSections7),
        addWithCuda(c5.data(), a.data(), b.data(), a.size(), numThreadsPerBlock,n,addKernelSections8)
    };
     
    const char* sectionNames[] = {"23", "45", "6", "7", "8"};

    for(int i = 0; i < 5; i++){
        if(statuses[i] != cudaSuccess){
            std::cerr << "addWithCuda section " << sectionNames[i] << " failed!" << std::endl;
            return 1;
        }
    }
    printResult(a,b,c1);

    try{
        writeResultToFile(c1);
    }
    catch(const std::exception &e){
        std::cerr << e.what() << std::endl;
    }

    cudaDeviceReset();

    return 0;
}




