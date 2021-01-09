cmake_minimum_required(VERSION 3.17)
set(CMAKE_CUDA_COMPILER /usr/local/cuda-11.2/bin/nvcc)
project(kmeans CUDA)

set(CMAKE_CUDA_STANDARD 14)

add_executable(kmeans main.cu km-cpu.cpp)

set_target_properties(
        kmeans
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)