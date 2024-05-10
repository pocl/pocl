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

set(TS_NAME "dpcpp-book-samples")
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
  set(REPO_TAG 6a2161df3794fea9e0d16)
endif()

message(STATUS "Enabling testsuite ${TS_NAME}")
list(APPEND ACTUALLY_ENABLED_TESTSUITES "${TS_NAME}")
set(ACTUALLY_ENABLED_TESTSUITES ${ACTUALLY_ENABLED_TESTSUITES} PARENT_SCOPE)

ExternalProject_Add(
  ${TS_NAME}
  PREFIX "${TS_BASEDIR}"
  GIT_REPOSITORY "https://github.com/Apress/data-parallel-CPP.git"
  GIT_TAG ${REPO_TAG}
  ${GIT_OPTIONS}
  CMAKE_ARGS
    "-DCMAKE_CXX_COMPILER=${SYCL_CXX_COMPILER}"
    "-DCMAKE_EXE_LINKER_FLAGS=-L${SYCL_LIBDIR}"
    "${TS_BASEDIR}/src/${TS_NAME}"
  INSTALL_COMMAND /bin/true
)

set_target_properties(${TS_NAME} PROPERTIES EXCLUDE_FROM_ALL TRUE)
add_dependencies(prepare_examples ${TS_NAME})

# Skipped tests for CPU:
#
# fig_2_10_gpu_selector: Fails as we test with a CPU driver.
# fig_6_8_prefetch_memadvise, fig_14_26_local_unpack: Takes minutes with the CPU driver.

set(SKIP_TEST_EXPR "fig_2_10_gpu_selector|fig_6_8_prefetch_memadvise|fig_14_26_local_unpack")
add_test(NAME all_dpcpp_book_samples-cpu
         COMMAND "${CMAKE_CTEST_COMMAND}" --output-on-failure
         -E "${SKIP_TEST_EXPR}"
         WORKING_DIRECTORY "${TS_BUILDDIR}")

         WORKING_DIRECTORY "${TS_BUILDDIR}")

# We need to use the oneAPI environment's LD_LIBRARY_PATH (need to source
# the setvars.sh before testing), but still ensure we use the system's libOpenCL.so
# to use its ICD to direct the OpenCL to libpocl (otherwise setvars.sh overwrites
# it with its own libOpenCL.so which is not ocl-icd). Also we have to fake the CPU device
# to look like Intel OpenCL CPU to force the oneAPI DPC++ to target it.

set_tests_properties(
  all_dpcpp_book_samples-cpu
  PROPERTIES
    ENVIRONMENT "ONEAPI_DEVICE_SELECTOR=opencl:cpu;POCL_CPU_VENDOR_ID_OVERRIDE=32902;POCL_DRIVER_VERSION_OVERRIDE=2023.16.7.0.21_160000;POCL_DEVICES=cpu;LD_LIBRARY_PATH=${OCL_ICD_LIBDIR}:${SYCL_LIBDIR}:$ENV{LD_LIBRARY_PATH}"
    LABELS "${TS_NAME}")
