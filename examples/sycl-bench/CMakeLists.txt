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

if(NOT Python3_EXECUTABLE)
  message(STATUS "Disabling testsuite ${TS_NAME}, can't find suitable python3")
  return()
else()
  message(STATUS "Using Python3: ${Python3_EXECUTABLE}")
  set(PYTHON_INTERP "${Python3_EXECUTABLE}")
endif()

if(EXAMPLES_USE_GIT_MASTER)
  set(REPO_TAG master)
else()
  set(REPO_TAG 31fc70be6266193c4ba60eb1fe3ce26edee4ca5b)
endif()

message(STATUS "Enabling testsuite ${TS_NAME}")
list(APPEND ACTUALLY_ENABLED_TESTSUITES "${TS_NAME}")
set(ACTUALLY_ENABLED_TESTSUITES ${ACTUALLY_ENABLED_TESTSUITES} PARENT_SCOPE)

string(REPLACE "icpx" "icx" SYCL_C_COMPILER "${SYCL_CXX_COMPILER}")

ExternalProject_Add(
  ${TS_NAME}
  PREFIX "${TS_BASEDIR}"
  GIT_REPOSITORY "https://github.com/unisa-hpc/sycl-bench.git"
  GIT_TAG ${REPO_TAG}
  ${GIT_OPTIONS}
  CMAKE_ARGS
    -DCMAKE_CXX_COMPILER=${SYCL_CXX_COMPILER}
    -DCMAKE_C_COMPILER=${SYCL_C_COMPILER}
    -DSYCL_IMPL=dpcpp
    -DSYCL_BENCH_HAS_FP64_SUPPORT=OFF
    "${TS_SRCDIR}"
  INSTALL_COMMAND "" # ${CMAKE_COMMAND} -E copy_directory "${TS_SRCDIR}/share" "${TS_BUILDDIR}"
)

set_target_properties(${TS_NAME} PROPERTIES EXCLUDE_FROM_ALL TRUE)
add_dependencies(prepare_examples ${TS_NAME})

set(TEST_LAUNCHER_SCRIPT ${PYTHON_INTERP} "${CMAKE_CURRENT_SOURCE_DIR}/sycl_bench_runner.py" "--output=${CMAKE_BINARY_DIR}/sycl_bench_result.json" "--")

# benchmark verification is single-threaded & extremely slow with larger problem size

# these are unreliable (ocassionally fail with PoCL & also Intel OpenCL):
# lin_reg_coeff lin_reg_error reduction
set(BENCHMARKS_256K arith atomic_reduction blocked_transform kmeans local_mem mol_dyn pattern_L2 scalar_prod segmentedreduction sf usm_accessors_latency usm_allocation_latency usm_instr_mix usm_pinned_overhead vec_add)
foreach(BENCH IN LISTS BENCHMARKS_256K)
  if(BENCH STREQUAL usm_accessors_latency)
    set(RUNS 10)
  else()
    set(RUNS 50)
  endif()
  add_test(NAME "${TS_NAME}_${BENCH}"
           COMMAND ${TEST_LAUNCHER_SCRIPT} "${TS_BUILDDIR}/${BENCH}" --size=256000 --local=256 --warmup-run --num-runs=${RUNS} --no-verification
           WORKING_DIRECTORY "${TS_BUILDDIR}")
endforeach()

set(BENCHMARKS_64K dag_task_throughput_independent dag_task_throughput_sequential)
foreach(BENCH IN LISTS BENCHMARKS_64K)
  add_test(NAME "${TS_NAME}_${BENCH}"
           COMMAND ${TEST_LAUNCHER_SCRIPT} "${TS_BUILDDIR}/${BENCH}" --size=65536 --local=64 --warmup-run --num-runs=10 --no-verification
           WORKING_DIRECTORY "${TS_BUILDDIR}")
endforeach()

set(BENCHMARKS_8K 2DConvolution atax bicg gesummv mvt nbody)
foreach(BENCH IN LISTS BENCHMARKS_8K)
  add_test(NAME "${TS_NAME}_${BENCH}"
           COMMAND ${TEST_LAUNCHER_SCRIPT} "${TS_BUILDDIR}/${BENCH}" --size=8192 --local=16 --warmup-run --num-runs=20 --no-verification
           WORKING_DIRECTORY "${TS_BUILDDIR}")
endforeach()

set(BENCHMARKS_2K gemm matmulchain correlation covariance 2mm 3mm) # sobel median
foreach(BENCH IN LISTS BENCHMARKS_2K)
  add_test(NAME "${TS_NAME}_${BENCH}"
           COMMAND ${TEST_LAUNCHER_SCRIPT} "${TS_BUILDDIR}/${BENCH}" --size=1024 --local=16 --warmup-run --num-runs=20 --no-verification
           WORKING_DIRECTORY "${TS_BUILDDIR}")
endforeach()

#list(APPEND BENCHMARKS_2K 3DConvolution)
#add_test(NAME "${TS_NAME}_3DConvolution"
#           COMMAND ${TEST_LAUNCHER_SCRIPT} "${TS_BUILDDIR}/${BENCH}" --size=512 --local=16 --warmup-run --num-runs=10 --no-verification
#           WORKING_DIRECTORY "${TS_BUILDDIR}")

foreach(BENCH IN LISTS BENCHMARKS_2K BENCHMARKS_64K BENCHMARKS_8K BENCHMARKS_256K)
  # RUN_SERIAL to avoid running anything else at the same time
  set_tests_properties("${TS_NAME}_${BENCH}" PROPERTIES LABELS "${TS_NAME}" RUN_SERIAL ON)
endforeach()
