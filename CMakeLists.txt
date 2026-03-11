cmake_minimum_required(VERSION 3.18)
project(MyCUDA CXX CUDA)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD 17)

find_package(CUDAToolkit REQUIRED)

include_directories(include)
include_directories(include/kernels.cuh)

add_executable(output
    src/main.cpp
    src/kernels.cu/addKernel.cu
    src/utils.cpp
)

target_link_libraries(output CUDA::cudart)