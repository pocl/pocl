#=============================================================================
#   CMake build system files
#
#   Copyright (c) 2014 pocl developers
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

message(STATUS "Trying HSA support in LLVM")
# test that Clang supports the amdgcn--amdhsa target
custom_try_compile_clangxx("" "return 0;" RESULT "-target" "amdgcn--amdhsa" "-emit-llvm" "-S")
if(RESULT)
  message(FATAL_ERROR "LLVM support for amdgcn--amdhsa target is required")
endif()

# find the headers & the library
message(STATUS "Searching for HSA runtime")
if(WITH_HSA_RUNTIME_DIR)
  if(IS_DIRECTORY "${WITH_HSA_RUNTIME_DIR}")
    set(HSA_RUNTIME_DIR "${WITH_HSA_RUNTIME_DIR}")
    message(STATUS "using ${WITH_HSA_RUNTIME_DIR}")
  else()
    message(FATAL_ERROR "${WITH_HSA_RUNTIME_DIR} is not a directory")
  endif()
else()
  message(STATUS "WITH_HSA_RUNTIME_DIR not given, trying default paths")
  if(IS_DIRECTORY "/opt/rocm/hsa")
    set(HSA_RUNTIME_DIR "/opt/rocm/hsa")
#    set(HSA_RUNTIME_IS_ROCM 1)
    message(STATUS "using /opt/rocm/hsa")
  endif()
  if(IS_DIRECTORY "/opt/hsa")
    set(HSA_RUNTIME_DIR "/opt/hsa")
#    set(HSA_RUNTIME_IS_ROCM 0)
    message(STATUS "using /opt/hsa")
  endif()
endif()

if((IS_ABSOLUTE "${HSA_RUNTIME_DIR}") AND (EXISTS "${HSA_RUNTIME_DIR}"))
  set(HSA_INCLUDEDIR "${HSA_RUNTIME_DIR}/include")
  set(HSA_LIBDIR "${HSA_RUNTIME_DIR}/lib")
else()
  message(WARNING "${HSA_RUNTIME_DIR} is not a directory (using default system paths for search)")
  set(HSA_INCLUDEDIR "")
  set(HSA_LIBDIR "")
endif()

find_path(HSA_INCLUDES "hsa.h" PATHS "${HSA_INCLUDEDIR}")
if(NOT HSA_INCLUDES)
  message(FATAL_ERROR "hsa.h header not found in ${HSA_INCLUDEDIR}")
endif()

find_library(HSALIB NAMES "hsa-runtime64" "hsa-runtime" "phsa-runtime64" PATHS "${HSA_LIBDIR}")
if(NOT HSALIB)
  message(FATAL_ERROR "libhsa-runtime{,64} not found in ${HSA_LIBDIR}")
endif()

###############################################################

if(HSA_RUNTIME_IS_ROCM)

  if(NOT LLVM_LD)
    message(FATAL_ERROR "Build with ROCM runtime requires a HSACO linker (lld.ld)!")
  endif()

  # Default: /opt/rocm/libamdgcn (/include, /lib/libamdgcn.{CPU}.bc ...)
  message(STATUS "Searching for ROCM kernel library")
  if(WITH_ROCM_LIBAMDGCN_DIR)
    if(IS_DIRECTORY "${WITH_ROCM_LIBAMDGCN_DIR}")
      set(ROCM_LIBAMDGCN_DIR "${WITH_ROCM_LIBAMDGCN_DIR}")
      message(STATUS "using ${WITH_ROCM_LIBAMDGCN_DIR}")
    else()
      message(FATAL_ERROR "${WITH_ROCM_LIBAMDGCN_DIR} is not a directory")
    endif()
  else()
    message(STATUS "WITH_ROCM_LIBAMDGCN_DIR not given, trying default paths")
    if(IS_DIRECTORY "/opt/rocm/libamdgcn")
      set(ROCM_LIBAMDGCN_DIR "/opt/rocm/libamdgcn")
      message(STATUS "using /opt/rocm/libamdgcn")
    else()
      message(FATAL_ERROR "Cannot find kernel libraries, please use -DWITH_ROCM_LIBAMDGCN_DIR")
    endif()
  endif()

else() # old HSA

  if(DEFINED WITH_HSAILASM_PATH)
    set(HSAILASM_SEARCH_PATH "${WITH_HSAILASM_PATH}")
  else()
    set(HSAILASM_SEARCH_PATH "${HSA_RUNTIME_DIR}")
  endif()

  if((EXISTS "${HSAILASM_SEARCH_PATH}") AND
      (NOT IS_DIRECTORY "${HSAILASM_SEARCH_PATH}"))
    set(HSAIL_ASM "${HSAILASM_SEARCH_PATH}")
  else()
    find_program(HSAIL_ASM "HSAILasm${CMAKE_EXECUTABLE_SUFFIX}" PATHS "${HSAILASM_SEARCH_PATH}" "${HSAILASM_SEARCH_PATH}/bin")
  endif()

  if(NOT HSAIL_ASM)
    message(FATAL_ERROR "HSAILasm executable not found (use -DWITH_HSAILASM_PATH=... to specify)")
  endif()

endif()

message(STATUS "OK, building HSA")
