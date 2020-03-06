CMAKE_MINIMUM_REQUIRED(VERSION 2.8)
PROJECT(test_vfs)
set( CMAKE_BUILD_TYPE RelWithDebInfo)

# OpenCV package
FIND_PACKAGE(OpenCV REQUIRED)

# OpenCV include directories
INCLUDE_DIRECTORIES(${OpenCV_INCLUDE_DIRS})
# CUDA include directories
INCLUDE_DIRECTORIES(/usr/local/cuda/include /usr/local/cuda/samples/common/inc)

INCLUDE_DIRECTORIES(../)

# file name
FILE(GLOB SOURCES "*.cu" "*.cpp" "*.c" "*.h")

# C++ list
LIST(APPEND CMAKE_CXX_FLAGS "-std=c++0x -O3 -ffast-math -Wall")

add_executable(test_vfs ${SOURCES})

# target
TARGET_LINK_LIBRARIES(test_vfs vfs /usr/local/cuda/lib64/libcudart.so ${OpenCV_LIBS})