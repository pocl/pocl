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
  set(CHIP_GIT_TAG "main")
endif()

# TODO: hardcoded -j8
ExternalProject_Add(
  ${TS_NAME}
  PREFIX "${TS_BASEDIR}"
  GIT_REPOSITORY "${CHIP_GIT_REPO}"
  GIT_TAG "${CHIP_GIT_TAG}"
  GIT_SUBMODULES_RECURSE ON
  CONFIGURE_COMMAND ${CMAKE_COMMAND}
    "-DLLVM_CONFIG=${LLVM_CONFIG}"
    "-DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}"
    "-DLAZY_JIT=ON"
    "${TS_BASEDIR}/src/${TS_NAME}"
  INSTALL_COMMAND "${CMAKE_MAKE_PROGRAM}" -j8 build_tests
)

set_target_properties(${TS_NAME} PROPERTIES EXCLUDE_FROM_ALL TRUE)
add_dependencies(prepare_examples ${TS_NAME})

# TODO hardcoded -j8 and repeat 2
add_test(NAME "${TS_NAME}"
         COMMAND "${Python3_EXECUTABLE}" "${TS_SRCDIR}/scripts/check.py"
           "${TS_BUILDDIR}" "cpu" "pocl"  8 2
         WORKING_DIRECTORY "${TS_BUILDDIR}")

set_tests_properties(${TS_NAME}
  PROPERTIES
  ENVIRONMENT "POCL_MAX_PTHREAD_COUNT=1"
  LABELS "chip-spv")
