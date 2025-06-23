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

set(TS_NAME "sycl-bench")
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
  set(REPO_TAG master)
else()
  set(REPO_TAG 31fc70be6266193c4ba60eb1fe3ce26edee4ca5b)
endif()

message(STATUS "Enabling testsuite ${TS_NAME}")
list(APPEND ACTUALLY_ENABLED_TESTSUITES "${TS_NAME}")
set(ACTUALLY_ENABLED_TESTSUITES ${ACTUALLY_ENABLED_TESTSUITES} PARENT_SCOPE)

ExternalProject_Add(
  ${TS_NAME}
  PREFIX "${TS_BASEDIR}"
  GIT_REPOSITORY "https://github.com/unisa-hpc/sycl-bench.git"
  GIT_TAG ${REPO_TAG}
  ${GIT_OPTIONS}
  CMAKE_ARGS
    -DCMAKE_CXX_COMPILER=${SYCL_CXX_COMPILER}
    -DCMAKE_C_COMPILER=${SYCL_CXX_COMPILER}
    -DSYCL_IMPL=dpcpp
    -DSYCL_BENCH_HAS_FP64_SUPPORT=OFF
    "${TS_BASEDIR}/src/${TS_NAME}"

  INSTALL_COMMAND ""
)

set_target_properties(${TS_NAME} PROPERTIES EXCLUDE_FROM_ALL TRUE)
add_dependencies(prepare_examples ${TS_NAME})

# these are unreliable (ocassionally fail with PoCL main & also Intel OpenCL):
# lin_reg_coeff lin_reg_error reduction
set(BENCHMARKS 2DConvolution 2mm 3DConvolution 3mm arith atax atomic_reduction bicg blocked_transform correlation covariance dag_task_throughput_independent dag_task_throughput_sequential DRAM fdtd2d gemm gesummv gramschmidt host_device_bandwidth kmeans local_mem Makefile matmulchain median mol_dyn mvt nbody pattern_L2 scalar_prod segmentedreduction sf sobel sobel5 sobel7 syr2k syrk usm_accessors_latency usm_allocation_latency usm_instr_mix usm_pinned_overhead vec_add)

foreach(BENCH IN LISTS BENCHMARKS)
  add_test(NAME "${TS_NAME}_${BENCH}_short"
           COMMAND "${TS_BUILDDIR}/${BENCH}" --size=4096 --local=256 --warmup-run --num-runs=3
           WORKING_DIRECTORY "${TS_BUILDDIR}")

  # verification is single-threaded & extremely slow with larger problem size
  # RUN_SERIAL to avoid running anything else at the same time
  add_test(NAME "${TS_NAME}_${BENCH}_long"
           COMMAND "${TS_BUILDDIR}/${BENCH}" --size=256000 --local=256 --warmup-run --num-runs=40 --no-verification
           WORKING_DIRECTORY "${TS_BUILDDIR}"
           RUN_SERIAL)
  set_property(TEST "${TS_NAME}_${BENCH}_long" PROPERTY TEST_LAUNCHER  ${CMAKE_COMMAND} -P "${CMAKE_SOURCE_DIR}/cmake/run_sycl_bench.cmake" "--" )

  set_property(TEST "${TS_NAME}_${BENCH}_short"
         PROPERTY LABELS ${TS_NAME})
  set_property(TEST "${TS_NAME}_${BENCH}_short"
         PROPERTY ENVIRONMENT "POCL_MEMORY_LIMIT=4")
  set_property(TEST "${TS_NAME}_${BENCH}_long"
         PROPERTY LABELS "${TS_NAME};long" )

endforeach()
