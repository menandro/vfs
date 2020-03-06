CMAKE_MINIMUM_REQUIRED(VERSION 2.8)
PROJECT(libvfs)

set( CMAKE_BUILD_TYPE RelWithDebInfo)
# CUDA package
FIND_PACKAGE(CUDA REQUIRED)
# OpenCV package
FIND_PACKAGE(OpenCV REQUIRED)

INCLUDE(FindCUDA)

# CUDA include directories
INCLUDE_DIRECTORIES(/usr/local/cuda/include /usr/local/cuda/samples/common/inc)

# OpenCV include directories
INCLUDE_DIRECTORIES(${OpenCV_INCLUDE_DIRS})

# file name
FILE(GLOB SOURCES "*.cu" "*.cpp" "*.c" "*.h")

# CUDA library
CUDA_ADD_LIBRARY(vfs ${SOURCES})

# C++ list
LIST(APPEND CMAKE_CXX_FLAGS "-std=c++0x -O3 -ffast-math -Wall")

# CUDA list
LIST(APPEND CUDA_NVCC_FLAGS --compiler-options -fno-strict-aliasing -lineinfo -use_fast_math -Xptxas -dlcm=cg)
LIST(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_60,code=sm_60)
LIST(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_75,code=sm_75)
LIST(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_35,code=sm_35)

#ADD_LIBRARY(vfs static  ${OpenCV_LIBS})
# target
TARGET_LINK_LIBRARIES(vfs /usr/local/cuda/lib64/libcudart.so ${OpenCV_LIBS})