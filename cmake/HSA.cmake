#=============================================================================
#   CMake build system files
#
#   Copyright (c) 2014-2018 pocl developers
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

if (DEFINED ENABLE_HSAIL AND NOT ENABLE_HSAIL)
  set(HSAIL_ENABLED 0)
else()
  message(STATUS "Trying HSA support in LLVM")
  # test that Clang supports the amdgcn--amdhsa target
  custom_try_compile_clangxx("" "return 0;" RESULT "-target" "amdgcn--amdhsa" "-emit-llvm" "-S")
  if(RESULT)
    message(FATAL_ERROR "LLVM support for amdgcn--amdhsa target is required")
  endif()
  set(HSAIL_ENABLED 1)
endif()

if (NOT DEFINED AMD_HSA)
  set(AMD_HSA 1)
endif()

# find the headers & the library
if(DEFINED WITH_HSA_RUNTIME_DIR AND WITH_HSA_RUNTIME_DIR)
  set(HSA_RUNTIME_DIR "${WITH_HSA_RUNTIME_DIR}")
else()
  message(STATUS "WITH_HSA_RUNTIME_DIR not given, trying default path")
  set(HSA_RUNTIME_DIR "/opt/hsa")
endif()

if(DEFINED WITH_HSA_RUNTIME_LIB_DIR AND WITH_HSA_RUNTIME_LIB_DIR)
  set(HSA_LIBDIR "${WITH_HSA_RUNTIME_LIB_DIR}")
elseif((IS_ABSOLUTE "${HSA_RUNTIME_DIR}") AND (EXISTS "${HSA_RUNTIME_DIR}"))
  set(HSA_INCLUDEDIR "${HSA_RUNTIME_DIR}/include")
  set(HSA_LIBDIR "${HSA_RUNTIME_DIR}/lib")
else()
  message(WARNING "${HSA_RUNTIME_DIR} is not a directory (using default system paths for search)")
  set(HSA_INCLUDEDIR "")
  set(HSA_LIBDIR "")
endif()

if(DEFINED WITH_HSA_RUNTIME_INCLUDE_DIR AND WITH_HSA_RUNTIME_INCLUDE_DIR)
  set(HSA_INCLUDEDIR "${WITH_HSA_RUNTIME_INCLUDE_DIR}")
elseif((IS_ABSOLUTE "${HSA_RUNTIME_DIR}") AND (EXISTS "${HSA_RUNTIME_DIR}"))
  set(HSA_INCLUDEDIR "${HSA_RUNTIME_DIR}/include")
else()
  message(WARNING "${HSA_RUNTIME_DIR} is not a directory (using default system paths for search)")
  set(HSA_INCLUDEDIR "")
endif()


find_path(HSA_INCLUDES "hsa.h" PATHS "${HSA_INCLUDEDIR}" NO_DEFAULT_PATH)
find_path(HSA_INCLUDES "hsa.h")
if(NOT HSA_INCLUDES)
  message(FATAL_ERROR "hsa.h header not found (use -DHSA_RUNTIME_DIR=... to specify path to HSA runtime)")
endif()

find_library(HSALIB NAMES "hsa-runtime64" "hsa-runtime" "phsa-runtime64" PATHS "${HSA_LIBDIR}" NO_DEFAULT_PATH)
find_library(HSALIB NAMES "hsa-runtime64" "hsa-runtime" "phsa-runtime64")
if(NOT HSALIB)
  message(FATAL_ERROR "libhsa-runtime not found (use -DWITH_HSA_RUNTIME_DIR=... to specify path to HSA runtime) ${HSA_LIBDIR}")
endif()

if (HSAIL_ENABLED)
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

if (HSAIL_ENABLED)
  message(STATUS "OK, building HSA with HSAIL")
else()
  message(STATUS "OK, building HSA with native code generation")
endif()
