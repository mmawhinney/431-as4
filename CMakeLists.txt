cmake_minimum_required(VERSION 2.8)

project(5kk70-assignment-gpu)

find_package(CUDA REQUIRED)

set(CMAKE_CXX_FLAG "-g -Wall -Werror")

# Adjust according to your platform
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-O3 -fmad=false -gencode arch=compute_35,code=sm_35)

# The second option is to make use of utility functions defined in samples/common/inc
include_directories(${CUDA_INCLUDE_DIRS} "${CUDA_TOOLKIT_ROOT_DIR}/samples/common/inc")

message(STATUS ${CUDA_INCLUDE_DIRS})
message(STATUS ${CUDA_TOOLKIT_ROOT_DIR})
# set(CMAKE_MODULE_PATH "/usr/share/cmake/Modules" ${CMAKE_MODULE_PATH})

set(CPPSRCS
   main.cpp
   histogram-equalization.cpp
   contrast-enhancement.cpp
)

set(CUDASRCS
   contrast-enhancement-cuda.cu
   histogram-equalization.cu
)

CUDA_ADD_EXECUTABLE(${PROJECT_NAME} ${CUDASRCS} ${CPPSRCS})

