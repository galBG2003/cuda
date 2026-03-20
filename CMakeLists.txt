cmake_minimum_required(VERSION 3.18)
project(MyCUDA CXX CUDA)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD 17)

find_package(CUDAToolkit REQUIRED)

include_directories(include)

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -g -G")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g")

add_executable(output
    src/main.cpp
    src/kernels.cu
    src/cudaFunctions.cu
    src/utils.cpp
)

target_link_libraries(output CUDA::cudart)