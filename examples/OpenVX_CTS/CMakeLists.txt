#=============================================================================
#   CMake build system files
#
#   Copyright (c) 2025 Michal Babej / Intel Finland Oy
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

set(TS_NAME "OpenVX_CTS")
set(TS_BASEDIR "${TESTSUITE_BASEDIR}/${TS_NAME}")
set(TS_BUILDDIR "${TS_BASEDIR}/src/${TS_NAME}-build")
set(TS_SRCDIR "${TS_BASEDIR}/src/${TS_NAME}")

if(NOT HAVE_GIT)
  message(STATUS "Disabling testsuite ${TS_NAME}, requires git to checkout sources")
  return()
endif()

if(NOT BASH)
  message(STATUS "Disabling testsuite ${TS_NAME}, can't find bash shell")
  return()
endif()

if(ENABLE_ICD AND (NOT OPENCL_LIBRARIES))
  message(STATUS "Disabling ${TS_NAME}, can't find OpenCL library")
  return()
endif()

if(NOT EXISTS /opt/rocm)
  message(STATUS "Disabling ${TS_NAME}, can't find ROCm")
  return()
endif()

if((NOT EXISTS /opt/rocm/include/mivisionx) OR (NOT EXISTS /opt/rocm/lib/libopenvx.so))
  message(STATUS "Disabling ${TS_NAME}, can't find MIVisionX")
  return()
endif()

find_program(CLANGXX_18 clang++-18)
find_program(CLANG_18 clang-18)

if(NOT CLANG_18 OR NOT CLANGXX_18)
  message(STATUS "Disabling ${TS_NAME}, can't find Clang 18")
  return()
endif()

message(STATUS "Enabling testsuite ${TS_NAME}")
list(APPEND ACTUALLY_ENABLED_TESTSUITES "${TS_NAME}")
set(ACTUALLY_ENABLED_TESTSUITES ${ACTUALLY_ENABLED_TESTSUITES} PARENT_SCOPE)

if(CUSTOM_OPENVX_CTS_GIT_REPO)
  set(OPENVX_CTS_GIT_REPO "${CUSTOM_OPENVX_CTS_GIT_REPO}")
else()
  set(OPENVX_CTS_GIT_REPO "https://github.com/KhronosGroup/OpenVX-cts.git")
endif()
if(CUSTOM_OPENVX_CTS_GIT_TAG)
  set(OPENVX_CTS_GIT_TAG "${CUSTOM_OPENVX_CTS_GIT_TAG}")
else()
  set(OPENVX_CTS_GIT_TAG "openvx_1.3")
endif()

set(ROCM_ROOT "/opt/rocm")

ExternalProject_Add(
  ${TS_NAME}
  PREFIX "${TS_BASEDIR}"
  GIT_REPOSITORY "${OPENVX_CTS_GIT_REPO}"
  GIT_TAG "${OPENVX_CTS_GIT_TAG}"
  ${GIT_OPTIONS}
  CMAKE_ARGS
    -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
    -DOPENVX_CONFORMANCE_VISION=ON
    -DCMAKE_BUILD_TYPE=Release
    -DOPENVX_INCLUDES=${ROCM_ROOT}/include/mivisionx
    "-DOPENVX_LIBRARIES=${ROCM_ROOT}/lib/libopenvx.so$<SEMICOLON>${ROCM_ROOT}/lib/libvxu.so$<SEMICOLON>pthread$<SEMICOLON>dl$<SEMICOLON>m$<SEMICOLON>rt"
    "-DCMAKE_CXX_COMPILER_PATH=${CLANGXX_18}"
    "-DCMAKE_C_COMPILER_PATH=${CLANG_18}"
    "${TS_BASEDIR}/src/${TS_NAME}"
  INSTALL_COMMAND ""
)

set_target_properties(${TS_NAME} PROPERTIES EXCLUDE_FROM_ALL TRUE)
add_dependencies(prepare_examples ${TS_NAME})

add_test(NAME "${TS_NAME}"
         COMMAND "${TS_BUILDDIR}/bin/vx_test_conformance" "--filter=*:-SmokeTestBase.vxRelease*:-SmokeTestBase.vxLoadKernels:-SmokeTestBase.vxUnloadKernels:-Array.vxCreateArray/?/array/VX_TYPE*:-vxConvertDepth.BitExact*:-GaussianPyramid.GraphProcessing/?/randomInput/VX_BORDER_UNDEFINED*:-GaussianPyramid.ImmediateProcessing/*/VX_BORDER_UNDEFINED*:-GaussianPyramid.GraphProcessing/37/lena/VX_BORDER_UNDEFINED*"
         WORKING_DIRECTORY "${TS_BUILDDIR}")

# Unit_hipMemset_SetMemoryWithOffset require CL_DEVICE_MAX_MEM_ALLOC_SIZE of >= 1GB,
# so PoCL CPU device memory must be >= 4GB
set_tests_properties(${TS_NAME}
  PROPERTIES
  ENVIRONMENT "AGO_DEFAULT_TARGET=GPU;LD_LIBRARY_PATH=$ENV{LD_LIBRARY_PATH}$<SEMICOLON>${TS_BUILDDIR}/lib;VX_TEST_DATA_PATH=${TS_SRCDIR}/test_data;POCL_CPU_MAX_CU_COUNT=2;POCL_MEMORY_LIMIT=6"
  LABELS "${TS_NAME};openvx")
