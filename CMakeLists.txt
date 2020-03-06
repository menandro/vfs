cmake_minimum_required(VERSION 2.8 FATAL_ERROR)
project(vfs)

set( CMAKE_BUILD_TYPE RelWithDebInfo)
set(OpenCV_DIR "/usr/local/lib/cmake/opencv4/") # which change to your own path

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

add_subdirectory(stereotgv)
add_subdirectory(test_vfs)