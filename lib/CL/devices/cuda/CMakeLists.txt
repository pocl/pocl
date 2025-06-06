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

# this is required, because we're compiling the C++ files of this directory
# with LLVM-provided flags, which usually contain their own setting of
# C++ standard, which may be in conflict with pocl's default C++ standard.
unset(CMAKE_CXX_STANDARD)
unset(CMAKE_CXX_STANDARD_REQUIRED)
set_source_files_properties(pocl-ptx-gen.cc pocl-ptx-gen.h PROPERTIES COMPILE_FLAGS "${LLVM_CXXFLAGS}")
include_directories(${LLVM_INCLUDE_DIRS} ${CUDA_INCLUDE_DIRS} ${CMAKE_CURRENT_SOURCE_DIR}/../../../llvmopencl)
set(CMAKE_CXX_STANDARD ${POCL_CMAKE_CXX_STANDARD})
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_pocl_device_library("pocl-devices-cuda" pocl-cuda.c pocl-cuda.h pocl-ptx-gen.cc pocl-ptx-gen.h)
target_compile_definitions("pocl-devices-cuda" PRIVATE "-DCUDA_TOOLKIT_ROOT_DIR=\"${CUDA_TOOLKIT_ROOT_DIR}\"")
if(ENABLE_LOADABLE_DRIVERS)
  target_link_libraries(pocl-devices-cuda PRIVATE ${CUDA_POCL_TARGETS} ${CLANG_LINK_LIBRARIES} ${LLVM_LINK_LIBRARIES} ${PTHREAD_LIBRARY})
  target_link_directories(pocl-devices-cuda PRIVATE ${CLANG_LINK_DIRECTORIES} ${LLVM_LINK_DIRECTORIES})
endif()


install(FILES "builtins.cl" "builtins_sm50.ptx" "builtins_sm70.ptx"
        DESTINATION "${POCL_INSTALL_PRIVATE_DATADIR_REL}/cuda" COMPONENT "lib")
