#=============================================================================
#   CMake build system files
#
#   Copyright (c) 2023 Michal Babej / Intel Finland Oy
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

set(TS_NAME "chip-spv")
set(TS_BASEDIR "${TESTSUITE_BASEDIR}/${TS_NAME}")
set(TS_BUILDDIR "${TS_BASEDIR}/src/${TS_NAME}-build")
set(TS_SRCDIR "${TS_BASEDIR}/src/${TS_NAME}")

if(NOT HAVE_GIT)
  message(STATUS "Disabling testsuite ${TS_NAME}, requires git to checkout sources")
  return()
endif()

if(ENABLE_ICD AND (NOT OPENCL_LIBRARIES))
  message(STATUS "Disabling ${TS_NAME}, can't find OpenCL library")
  return()
endif()

if(NOT ENABLE_LLVM)
  message(STATUS "Disabling ${TS_NAME}, needs ENABLE_LLVM=ON")
  return()
endif()

if(NOT ENABLE_HOST_CPU_DEVICES)
  message(STATUS "Disabling ${TS_NAME}, needs ENABLE_HOST_CPU_DEVICES=ON")
  return()
endif()

if(ENABLE_HOST_CPU_DEVICES AND (ENABLE_CONFORMANCE))
  message(WARNING "CHIP-SPV + CPU driver with ENABLE_CONFORMANCE=ON will fail some tests")
endif()

find_package(Python3 3.8 COMPONENTS Interpreter Development)
if(NOT Python3_FOUND)
  message(STATUS "Disabling testsuite ${TS_NAME}, can't find suitable python3")
  return()
else()
  message(STATUS "Using Python3: ${Python3_EXECUTABLE}")
  set(PYTHON_INTERP "${Python3_EXECUTABLE}")
endif()

message(STATUS "Enabling testsuite ${TS_NAME}")
list(APPEND ACTUALLY_ENABLED_TESTSUITES "${TS_NAME}")
set(ACTUALLY_ENABLED_TESTSUITES ${ACTUALLY_ENABLED_TESTSUITES} PARENT_SCOPE)

if(CUSTOM_CHIP_GIT_REPO)
  set(CHIP_GIT_REPO "${CUSTOM_CHIP_GIT_REPO}")
else()
  set(CHIP_GIT_REPO "https://github.com/CHIP-SPV/chip-spv")
endif()
if(CUSTOM_CHIP_GIT_TAG)
  set(CHIP_GIT_TAG "${CUSTOM_CHIP_GIT_TAG}")
else()
  set(CHIP_GIT_TAG "f1ff1ef4749392e5c656663c0c1be5a286c69cd4")
endif()

# if we have disabled FP16, disable also the FP16 tests in chip-spv
# TODO this assumes we're using CPU device
if(NOT HOST_CPU_ENABLE_CL_KHR_FP16)
  set(CHIP_SPV_INSTALL_COMMAND "${CMAKE_COMMAND}" -E copy
    "${CMAKE_CURRENT_SOURCE_DIR}/failed_tests_fp16_disabled.txt"
    "${TS_BUILDDIR}/test_lists/cpu_pocl_failed_tests.txt")
else()
  set(CHIP_SPV_INSTALL_COMMAND "echo" "INSTALLED")
endif()

# set up env variables. Unfortunately this is necessary even just to build chip-spv
# because the catch testsuite's "autodiscovery" feature requires
# a working chip-spv (which requires a working backend)
set(CMAKE_CALL_ENV "-E" "env" "OCL_ICD_VENDORS=${CMAKE_BINARY_DIR}/ocl-vendors/pocl-tests.icd")
list(APPEND CMAKE_CALL_ENV "POCL_BUILDING=1")
list(APPEND CMAKE_CALL_ENV "POCL_KERNEL_CACHE=0")
list(APPEND CMAKE_CALL_ENV "CHIP_BE=opencl")
list(APPEND CMAKE_CALL_ENV "CHIP_LOGLEVEL=warn")
list(APPEND CMAKE_CALL_ENV "CHIP_DEVICE_TYPE=cpu")
list(APPEND CMAKE_CALL_ENV "HIP_COMPILER=clang")
list(APPEND CMAKE_CALL_ENV "HIP_RUNTIME=spirv")
list(APPEND CMAKE_CALL_ENV "HIP_PLATFORM=spirv")
list(APPEND CMAKE_CALL_ENV "HIP_ARCH=spirv")

# add llvm-spirv's directory to the PATH environment var. Necessary
# for chip-spv; in theory passing -DLLVM_SPIRV=... should be enough,
# but it doesn't pass that to Clang (yet) so the compilation fails.
if(LLVM_SPIRV)
  get_filename_component(LLVM_SPIRV_DIR "${LLVM_SPIRV}" DIRECTORY)
  set(NEW_PATH "${LLVM_SPIRV_DIR}:$ENV{PATH}")
  list(APPEND CMAKE_CALL_ENV "PATH=${NEW_PATH}")
endif()

list(APPEND CMAKE_CALL_ENV ${CMAKE_COMMAND})

set(UPSTREAM_CMAKE_ARGS "-DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}")
if(CMAKE_C_COMPILER_LAUNCHER)
  list(APPEND UPSTREAM_CMAKE_ARGS "-DCMAKE_C_COMPILER_LAUNCHER=${CMAKE_C_COMPILER_LAUNCHER}")
endif()
if(CMAKE_CXX_COMPILER_LAUNCHER)
  list(APPEND UPSTREAM_CMAKE_ARGS "-DCMAKE_CXX_COMPILER_LAUNCHER=${CMAKE_CXX_COMPILER_LAUNCHER}")
endif()

# TODO: hardcoded -j8
ExternalProject_Add(
  ${TS_NAME}
  PREFIX "${TS_BASEDIR}"
  GIT_REPOSITORY "${CHIP_GIT_REPO}"
  GIT_TAG "${CHIP_GIT_TAG}"
  CONFIGURE_COMMAND "${CMAKE_COMMAND}" ${CMAKE_CALL_ENV}
    ${UPSTREAM_CMAKE_ARGS}
    "-DLLVM_CONFIG=${LLVM_CONFIG}"
    "-DLAZY_JIT=ON"
    "${TS_BASEDIR}/src/${TS_NAME}"
  PATCH_COMMAND patch -N -r- -p1 -i ${CMAKE_CURRENT_SOURCE_DIR}/HIP_tests_catch.patch
  BUILD_COMMAND ${CMAKE_COMMAND} ${CMAKE_CALL_ENV} --build . --parallel ${CORECOUNT} --target all build_tests
  INSTALL_COMMAND ${CHIP_SPV_INSTALL_COMMAND}
)

set_target_properties(${TS_NAME} PROPERTIES EXCLUDE_FROM_ALL TRUE)
add_dependencies(prepare_examples ${TS_NAME})
# this ensures that libpocl is built before chip-spv testsuite build begins
add_dependencies(${TS_NAME} ${POCL_LIBRARY_NAME})
if(ENABLE_LOADABLE_DRIVERS)
  add_dependencies(${TS_NAME} pocl-devices-basic pocl-devices-pthread pocl-devices-topology)
endif()

# TODO hardcoded -j8 and repeat 2
add_test(NAME "${TS_NAME}"
         COMMAND "${Python3_EXECUTABLE}" "${TS_SRCDIR}/scripts/check.py"
           "${TS_BUILDDIR}" "cpu" "pocl" ${CORECOUNT} 2
         WORKING_DIRECTORY "${TS_BUILDDIR}")

set_tests_properties(${TS_NAME}
  PROPERTIES
  PROCESSORS ${CORECOUNT}
  ENVIRONMENT "POCL_CPU_MAX_CU_COUNT=1"
  PASS_REGULAR_EXPRESSION "100% tests passed, 0 tests failed out of"
  LABELS "chip-spv")
