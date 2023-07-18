project (oneapi-samples CXX)

cmake_minimum_required (VERSION 3.9)

enable_testing()

if(IS_ABSOLUTE "${CL_LIB_NAME}" AND EXISTS "${CL_LIB_NAME}")
  message(STATUS "Using OpenCL (absolute path) library: ${CL_LIB_NAME}")
  set(OCL_LIB "${CL_LIB_NAME}")
  get_filename_component(OCL_DIR "${CL_LIB_NAME}" DIRECTORY)
else()
  message(STATUS "Using OpenCL (named) library: ${CL_LIB_NAME}")
  find_library(OCL_LIB "${CL_LIB_NAME}" PATHS "${CL_LIB_DIR}" NO_DEFAULT_PATH REQUIRED)
  message(STATUS "Found OpenCL library at: ${OCL_LIB}")
endif()

set(SYCL_LIBRARIES ${OCL_LIB})

set(CMAKE_CXX_STANDARD 17)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsycl -fsycl-targets=spir64")

include_directories(util)

add_subdirectory(1d_HeatTransfer)
add_subdirectory(bitonic-sort)
# add_subdirectory(mandelbrot)
add_subdirectory(Nbody)
add_subdirectory(simple-add)
add_subdirectory(vector-add)
