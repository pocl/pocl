#=============================================================================
#   CMake build system files
#
#   Copyright (c) 2016 pocl developers
#
#   Permission is hereby granted, free of charge, to any person obtaining a copy
#   of this software and associated documentation files (the "Software"), to deal
#   in the Software without restriction, including without limitation the rights
#   to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#   copies of the Software, and to permit persons to whom the Software is
#   furnished to do so, subject to the following conditions:
#
#   The above copyright notice and this permission notice shall be included in
#   all copies or substantial portions of the Software.
#
#   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#   OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
#   THE SOFTWARE.
#
#=============================================================================

find_package(CUDA REQUIRED)
message(STATUS "CUDA_TOOLKIT_ROOT_DIR = ${CUDA_TOOLKIT_ROOT_DIR}")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${LLVM_CXXFLAGS}")
include_directories(${LLVM_INCLUDE_DIRS} ${CUDA_INCLUDE_DIRS} ${CMAKE_CURRENT_SOURCE_DIR}/../../../llvmopencl)

add_pocl_device_library("pocl-devices-cuda" pocl-cuda.c pocl-cuda.h pocl-ptx-gen.cc pocl-ptx-gen.h)
target_compile_definitions("pocl-devices-cuda" PRIVATE "-DCUDA_TOOLKIT_ROOT_DIR=\"${CUDA_TOOLKIT_ROOT_DIR}\"")
if(ENABLE_LOADABLE_DRIVERS)
  target_link_libraries(pocl-devices-cuda PRIVATE cuda ${POCL_LLVM_LIBS} ${PTHREAD_LIBRARY})
endif()

install(FILES "builtins.cl" "builtins_sm50.ptx" "builtins_sm70.ptx"
        DESTINATION "${POCL_INSTALL_PRIVATE_DATADIR}/cuda" COMPONENT "lib")
