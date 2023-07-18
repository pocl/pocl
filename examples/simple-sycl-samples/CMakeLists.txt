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

set(TS_NAME "simple-sycl-samples")
set(TS_BASEDIR "${TESTSUITE_BASEDIR}/${TS_NAME}")
set(TS_BUILDDIR "${TS_BASEDIR}/src/${TS_NAME}-build")
set(TS_SRCDIR "${TS_BASEDIR}/src/${TS_NAME}")

if((NOT SYCL_CXX_COMPILER) OR (NOT SYCL_LIBDIR))
  message(STATUS "Disabling testsuite ${TS_NAME}, requires a SYCL compiler (-DSYCL_CXX_COMPILER=...) and also SYCL_LIBDIR (path to libsycl.so)")
  return()
endif()

if(NOT HAVE_GIT)
  message(STATUS "Disabling testsuite ${TS_NAME}, requires git to checkout sources")
  return()
endif()

if(EXAMPLES_USE_GIT_MASTER)
  set(REPO_TAG main)
else()
  set(REPO_TAG 9d57b1c4a38b1c040b75212aad6c88b74aa762f9)
endif()

message(STATUS "Enabling testsuite ${TS_NAME}")
list(APPEND ACTUALLY_ENABLED_TESTSUITES "${TS_NAME}")
set(ACTUALLY_ENABLED_TESTSUITES ${ACTUALLY_ENABLED_TESTSUITES} PARENT_SCOPE)

ExternalProject_Add(
  ${TS_NAME}
  PREFIX "${TS_BASEDIR}"
  GIT_REPOSITORY "https://github.com/bashbaug/simple-sycl-samples.git"
  GIT_TAG ${REPO_TAG}
  CMAKE_ARGS
    -DCMAKE_CXX_COMPILER=${SYCL_CXX_COMPILER}
    "${TS_BASEDIR}/src/${TS_NAME}"

  INSTALL_COMMAND /bin/true
)


set_target_properties(${TS_NAME} PROPERTIES EXCLUDE_FROM_ALL TRUE)
add_dependencies(prepare_examples ${TS_NAME})

add_test(NAME all_simple_sycl_samples
         COMMAND "${CMAKE_CTEST_COMMAND}"
         WORKING_DIRECTORY "${TS_BUILDDIR}")

set_tests_properties(
  all_simple_sycl_samples
  PROPERTIES
  ENVIRONMENT "LD_LIBRARY_PATH=${SYCL_LIBDIR};ONEAPI_DEVICE_SELECTOR=opencl:*"
    LABELS "${TS_NAME};level0")
