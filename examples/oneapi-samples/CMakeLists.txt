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

set(TS_NAME "oneapi-samples")
set(TS_BASEDIR "${TESTSUITE_BASEDIR}/${TS_NAME}")
set(TS_BUILDDIR "${TS_BASEDIR}/src/${TS_NAME}-build")
set(TS_SRCDIR "${CMAKE_CURRENT_SOURCE_DIR}/src")

if((NOT SYCL_CXX_COMPILER) OR (NOT SYCL_LIBDIR))
  message(STATUS "Disabling testsuite ${TS_NAME}, requires a SYCL compiler (-DSYCL_CXX_COMPILER=...) and also SYCL_LIBDIR (path to libsycl.so)")
  return()
endif()

# handle ICD on/off
if(ENABLE_ICD)
  if(OPENCL_LIBRARIES AND (NOT OPENCL_LIBDIR))
    get_filename_component(OPENCL_LIBDIR "${OPENCL_LIBRARIES}" DIRECTORY)
  endif()
  set(ONEAPI_OPENCL_LIBDIR "-DCL_LIB_DIR=${OPENCL_LIBDIR}")
  set(ONEAPI_OPENCL_LIBRARY "-DCL_LIB_NAME=${OPENCL_LIBRARIES}")
else()
  set(ONEAPI_OPENCL_LIBDIR "-DCL_LIB_DIR=$<TARGET_FILE_DIR:${POCL_LIBRARY_NAME}>")
  set(ONEAPI_OPENCL_LIBRARY "-DCL_LIB_NAME=$<TARGET_FILE_NAME:${POCL_LIBRARY_NAME}>")
endif()

message(STATUS "Enabling testsuite ${TS_NAME}")
list(APPEND ACTUALLY_ENABLED_TESTSUITES "${TS_NAME}")
set(ACTUALLY_ENABLED_TESTSUITES ${ACTUALLY_ENABLED_TESTSUITES} PARENT_SCOPE)

ExternalProject_Add(
  ${TS_NAME}
  PREFIX "${TS_BASEDIR}"
  SOURCE_DIR "${TS_SRCDIR}"
  BINARY_DIR "${TS_BUILDDIR}"
  CMAKE_ARGS
    -DCMAKE_CXX_COMPILER=${SYCL_CXX_COMPILER}
    ${ONEAPI_OPENCL_LIBDIR} ${ONEAPI_OPENCL_LIBRARY}
    -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
    "${TS_SRCDIR}"

  INSTALL_COMMAND /bin/true
)


set_target_properties(${TS_NAME} PROPERTIES EXCLUDE_FROM_ALL TRUE)
add_dependencies(prepare_examples ${TS_NAME})

add_test(NAME oneapi_samples
         COMMAND "${CMAKE_CTEST_COMMAND}"
         WORKING_DIRECTORY "${TS_BUILDDIR}")

set_tests_properties(
  oneapi_samples
  PROPERTIES
    ENVIRONMENT "LD_LIBRARY_PATH=${SYCL_LIBDIR};ONEAPI_DEVICE_SELECTOR=opencl:*"
    LABELS "${TS_NAME}")
