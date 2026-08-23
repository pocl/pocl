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

# MIVisionX provides the OpenVX implementation the CTS links against. By default
# it is taken from the ROCm install at /opt/rocm, but CUSTOM_MIVISIONX_ROOT lets
# CI (or a developer) point at a self-built MIVisionX -- e.g. the latest develop
# built with the OpenCL backend -- so the tested implementation is not tied to
# whatever MIVisionX version happens to be preinstalled on the runner.
if(CUSTOM_MIVISIONX_ROOT)
  set(MIVISIONX_ROOT "${CUSTOM_MIVISIONX_ROOT}")
else()
  set(MIVISIONX_ROOT "/opt/rocm")
endif()

if((NOT EXISTS ${MIVISIONX_ROOT}/include/mivisionx) OR (NOT EXISTS ${MIVISIONX_ROOT}/lib/libopenvx.so))
  message(STATUS "Disabling ${TS_NAME}, can't find MIVisionX in ${MIVISIONX_ROOT}")
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
  set(OPENVX_CTS_GIT_TAG "openvx_1.3.2")
endif()

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
    -DOPENVX_INCLUDES=${MIVISIONX_ROOT}/include/mivisionx
    "-DOPENVX_LIBRARIES=${MIVISIONX_ROOT}/lib/libopenvx.so$<SEMICOLON>${MIVISIONX_ROOT}/lib/libvxu.so$<SEMICOLON>pthread$<SEMICOLON>dl$<SEMICOLON>m$<SEMICOLON>rt"
    "-DCMAKE_CXX_COMPILER_PATH=${CLANGXX_18}"
    "-DCMAKE_C_COMPILER_PATH=${CLANG_18}"
    "${TS_BASEDIR}/src/${TS_NAME}"
  INSTALL_COMMAND ""
)

set_target_properties(${TS_NAME} PROPERTIES EXCLUDE_FROM_ALL TRUE)
add_dependencies(prepare_examples ${TS_NAME})

# The only excluded tests are the ORB-scale Gaussian pyramid cases with
# VX_BORDER_UNDEFINED. MIVisionX's OpenCL Graph/Immediate implementations
# produce incorrect results there on non-AMD OpenCL devices (its own CPU
# Reference model passes, and the tests pass on native AMD hardware), so this
# is a known upstream MIVisionX bug in the non-AMD OpenCL path, not a PoCL
# issue. All other vision tests pass.
#
# ROCm/MIVisionX#1750 (merged) fixed one contributing bug in this path (the
# amd_bytealign software emulation). A second, independent defect in the ORB
# GaussianPyramid OpenCL kernel remains, so these cases still fail on PoCL and
# the filter is retained. Once the remaining upstream fix lands and is
# verified here, this exclusion can be dropped for full vision conformance.
add_test(NAME "${TS_NAME}"
         COMMAND "${TS_BUILDDIR}/bin/vx_test_conformance" "--filter=*:-GaussianPyramid.GraphProcessing/*VX_BORDER_UNDEFINED*VX_SCALE_PYRAMID_ORB:-GaussianPyramid.ImmediateProcessing/*VX_BORDER_UNDEFINED*VX_SCALE_PYRAMID_ORB"
         WORKING_DIRECTORY "${TS_BUILDDIR}")

# Unit_hipMemset_SetMemoryWithOffset require CL_DEVICE_MAX_MEM_ALLOC_SIZE of >= 1GB,
# so PoCL CPU device memory must be >= 4GB
set_tests_properties(${TS_NAME}
  PROPERTIES
  ENVIRONMENT "AGO_DEFAULT_TARGET=GPU;LD_LIBRARY_PATH=${MIVISIONX_ROOT}/lib$<SEMICOLON>$ENV{LD_LIBRARY_PATH}$<SEMICOLON>${TS_BUILDDIR}/lib;VX_TEST_DATA_PATH=${TS_SRCDIR}/test_data;POCL_CPU_MAX_CU_COUNT=2;POCL_MEMORY_LIMIT=6"
  LABELS "${TS_NAME};openvx")
