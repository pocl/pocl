#=============================================================================
#   CMake build system files
#
#   Copyright (c) 2025 pocl developers
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

set(TS_NAME "polybenchGPU")
set(TS_BASEDIR "${TESTSUITE_BASEDIR}/${TS_NAME}")
set(TS_BUILDDIR "${TS_BASEDIR}/src/${TS_NAME}-build")

if(NOT HAVE_GIT)
  message(STATUS "Disabling testsuite ${TS_NAME}, requires git to checkout sources")
  return()
endif()

if(NOT Python3_EXECUTABLE)
  message(STATUS "Disabling testsuite ${TS_NAME}, can't find suitable python3")
  return()
else()
  set(PYTHON_INTERP "${Python3_EXECUTABLE}")
endif()

message(STATUS "Enabling testsuite ${TS_NAME}")
list(APPEND ACTUALLY_ENABLED_TESTSUITES "${TS_NAME}")
set(ACTUALLY_ENABLED_TESTSUITES ${ACTUALLY_ENABLED_TESTSUITES} PARENT_SCOPE)

if(ENABLE_ICD)
  set(OPENCL_LIBRARY "${OPENCL_LIBRARIES}")
else()
  set(OPENCL_LIBRARY "$<TARGET_FILE:${POCL_LIBRARY_NAME}>")
endif()

ExternalProject_Add(
  ${TS_NAME}
  PREFIX "${TS_BASEDIR}"
  GIT_REPOSITORY "https://github.com/cpc/polybench.git"
  GIT_TAG "abc08a37a035b281b3657f21552925dae8a05d11"
  ${GIT_OPTIONS}
  CMAKE_ARGS
    ${UPSTREAM_CMAKE_ARGS}
    -DUSE_SMALL_DATASET=OFF
    -DENABLE_VERIFICATION=OFF
    "-DOPENCL_INCLUDE_PATH=${CMAKE_SOURCE_DIR}/include/"
    "-DOPENCL_opencl_LIBRARY:STRING=${OPENCL_LIBRARY}"
  INSTALL_COMMAND ""
)

set_target_properties(${TS_NAME} PROPERTIES EXCLUDE_FROM_ALL TRUE)
add_dependencies(prepare_examples ${TS_NAME})

set(TEST_LAUNCHER_SCRIPT ${PYTHON_INTERP} "${CMAKE_CURRENT_SOURCE_DIR}/polybench_runner.py" "--output=${CMAKE_BINARY_DIR}/polybench_result.json")

add_test(NAME polybenchGPU_2DCONV COMMAND ${TEST_LAUNCHER_SCRIPT} ./2DConvolution
       WORKING_DIRECTORY "${TS_BUILDDIR}/OpenCL/2DCONV")
add_test(NAME polybenchGPU_2MM COMMAND ${TEST_LAUNCHER_SCRIPT} ./2mm
       WORKING_DIRECTORY "${TS_BUILDDIR}/OpenCL/2MM")
add_test(NAME polybenchGPU_3DCONV COMMAND ${TEST_LAUNCHER_SCRIPT} ./3DConvolution
       WORKING_DIRECTORY "${TS_BUILDDIR}/OpenCL/3DCONV")
add_test(NAME polybenchGPU_3MM COMMAND ${TEST_LAUNCHER_SCRIPT} ./3mm
       WORKING_DIRECTORY "${TS_BUILDDIR}/OpenCL/3MM")
add_test(NAME polybenchGPU_ADI COMMAND ${TEST_LAUNCHER_SCRIPT} ./adi
       WORKING_DIRECTORY "${TS_BUILDDIR}/OpenCL/ADI")
add_test(NAME polybenchGPU_ATAX COMMAND ${TEST_LAUNCHER_SCRIPT} ./atax
       WORKING_DIRECTORY "${TS_BUILDDIR}/OpenCL/ATAX")
add_test(NAME polybenchGPU_BICG COMMAND ${TEST_LAUNCHER_SCRIPT} ./bicg
       WORKING_DIRECTORY "${TS_BUILDDIR}/OpenCL/BICG")
add_test(NAME polybenchGPU_CORR COMMAND ${TEST_LAUNCHER_SCRIPT} ./correlation
       WORKING_DIRECTORY "${TS_BUILDDIR}/OpenCL/CORR")
add_test(NAME polybenchGPU_COVAR COMMAND ${TEST_LAUNCHER_SCRIPT} ./covariance
       WORKING_DIRECTORY "${TS_BUILDDIR}/OpenCL/COVAR")
add_test(NAME polybenchGPU_FDTD-2D COMMAND ${TEST_LAUNCHER_SCRIPT} ./fdtd2d
        WORKING_DIRECTORY "${TS_BUILDDIR}/OpenCL/FDTD-2D")
add_test(NAME polybenchGPU_GEMM COMMAND ${TEST_LAUNCHER_SCRIPT} ./gemm
       WORKING_DIRECTORY "${TS_BUILDDIR}/OpenCL/GEMM")
add_test(NAME polybenchGPU_GEMVER COMMAND ${TEST_LAUNCHER_SCRIPT} ./gemver
       WORKING_DIRECTORY "${TS_BUILDDIR}/OpenCL/GEMVER")
add_test(NAME polybenchGPU_GESUMMV COMMAND ${TEST_LAUNCHER_SCRIPT} ./gesummv
       WORKING_DIRECTORY "${TS_BUILDDIR}/OpenCL/GESUMMV")
add_test(NAME polybenchGPU_GRAMSCHM COMMAND ${TEST_LAUNCHER_SCRIPT} ./gramschmidt
       WORKING_DIRECTORY "${TS_BUILDDIR}/OpenCL/GRAMSCHM")
add_test(NAME polybenchGPU_JACOBI1D COMMAND ${TEST_LAUNCHER_SCRIPT} ./jacobi1d
       WORKING_DIRECTORY "${TS_BUILDDIR}/OpenCL/JACOBI1D")
add_test(NAME polybenchGPU_JACOBI2D COMMAND ${TEST_LAUNCHER_SCRIPT} ./jacobi2d
       WORKING_DIRECTORY "${TS_BUILDDIR}/OpenCL/JACOBI2D")
add_test(NAME polybenchGPU_LU COMMAND ${TEST_LAUNCHER_SCRIPT} ./lu
       WORKING_DIRECTORY "${TS_BUILDDIR}/OpenCL/LU")
add_test(NAME polybenchGPU_MVT COMMAND ${TEST_LAUNCHER_SCRIPT} ./mvt
       WORKING_DIRECTORY "${TS_BUILDDIR}/OpenCL/MVT")
add_test(NAME polybenchGPU_SYR2K COMMAND ${TEST_LAUNCHER_SCRIPT} ./syr2k
       WORKING_DIRECTORY "${TS_BUILDDIR}/OpenCL/SYR2K")
add_test(NAME polybenchGPU_SYRK COMMAND ${TEST_LAUNCHER_SCRIPT} ./syrk
       WORKING_DIRECTORY "${TS_BUILDDIR}/OpenCL/SYRK")

add_test(NAME polybenchGPU_2DCONV_cmd_buffer COMMAND ${TEST_LAUNCHER_SCRIPT} ./2DConvolution_cmd_buffer
       WORKING_DIRECTORY "${TS_BUILDDIR}/OpenCL-command-buffer/2DCONV")
add_test(NAME polybenchGPU_2MM_cmd_buffer COMMAND ${TEST_LAUNCHER_SCRIPT} ./2mm_cmd_buffer
       WORKING_DIRECTORY "${TS_BUILDDIR}/OpenCL-command-buffer/2MM")
add_test(NAME polybenchGPU_3DCONV_cmd_buffer COMMAND ${TEST_LAUNCHER_SCRIPT} ./3DConvolution_cmd_buffer
       WORKING_DIRECTORY "${TS_BUILDDIR}/OpenCL-command-buffer/3DCONV")
add_test(NAME polybenchGPU_3MM_cmd_buffer COMMAND ${TEST_LAUNCHER_SCRIPT} ./3mm_cmd_buffer
       WORKING_DIRECTORY "${TS_BUILDDIR}/OpenCL-command-buffer/3MM")
add_test(NAME polybenchGPU_ADI_cmd_buffer COMMAND ${TEST_LAUNCHER_SCRIPT} ./adi_cmd_buffer
       WORKING_DIRECTORY "${TS_BUILDDIR}/OpenCL-command-buffer/ADI")
add_test(NAME polybenchGPU_ATAX_cmd_buffer COMMAND ${TEST_LAUNCHER_SCRIPT} ./atax_cmd_buffer
       WORKING_DIRECTORY "${TS_BUILDDIR}/OpenCL-command-buffer/ATAX")
add_test(NAME polybenchGPU_BICG_cmd_buffer COMMAND ${TEST_LAUNCHER_SCRIPT} ./bicg_cmd_buffer
       WORKING_DIRECTORY "${TS_BUILDDIR}/OpenCL-command-buffer/BICG")
add_test(NAME polybenchGPU_CORR_cmd_buffer COMMAND ${TEST_LAUNCHER_SCRIPT} ./correlation_cmd_buffer
       WORKING_DIRECTORY "${TS_BUILDDIR}/OpenCL-command-buffer/CORR")
add_test(NAME polybenchGPU_COVAR_cmd_buffer COMMAND ${TEST_LAUNCHER_SCRIPT} ./covariance_cmd_buffer
       WORKING_DIRECTORY "${TS_BUILDDIR}/OpenCL-command-buffer/COVAR")
add_test(NAME polybenchGPU_FDTD-2D_cmd_buffer COMMAND ${TEST_LAUNCHER_SCRIPT} ./fdtd2d_cmd_buffer
       WORKING_DIRECTORY "${TS_BUILDDIR}/OpenCL-command-buffer/FDTD-2D")
add_test(NAME polybenchGPU_GEMM_cmd_buffer COMMAND ${TEST_LAUNCHER_SCRIPT} ./gemm_cmd_buffer
       WORKING_DIRECTORY "${TS_BUILDDIR}/OpenCL-command-buffer/GEMM")
add_test(NAME polybenchGPU_GEMVER_cmd_buffer COMMAND ${TEST_LAUNCHER_SCRIPT} ./gemver_cmd_buffer
       WORKING_DIRECTORY "${TS_BUILDDIR}/OpenCL-command-buffer/GEMVER")
add_test(NAME polybenchGPU_GESUMMV_cmd_buffer COMMAND ${TEST_LAUNCHER_SCRIPT} ./gesummv_cmd_buffer
       WORKING_DIRECTORY "${TS_BUILDDIR}/OpenCL-command-buffer/GESUMMV")
add_test(NAME polybenchGPU_GRAMSCHM_cmd_buffer COMMAND ${TEST_LAUNCHER_SCRIPT} ./gramschmidt_cmd_buffer
       WORKING_DIRECTORY "${TS_BUILDDIR}/OpenCL-command-buffer/GRAMSCHM")
add_test(NAME polybenchGPU_JACOBI1D_cmd_buffer COMMAND ${TEST_LAUNCHER_SCRIPT} ./jacobi1d_cmd_buffer
       WORKING_DIRECTORY "${TS_BUILDDIR}/OpenCL-command-buffer/JACOBI1D")
add_test(NAME polybenchGPU_JACOBI2D_cmd_buffer COMMAND ${TEST_LAUNCHER_SCRIPT} ./jacobi2d_cmd_buffer
       WORKING_DIRECTORY "${TS_BUILDDIR}/OpenCL-command-buffer/JACOBI2D")
add_test(NAME polybenchGPU_LU_cmd_buffer COMMAND ${TEST_LAUNCHER_SCRIPT} ./lu_cmd_buffer
       WORKING_DIRECTORY "${TS_BUILDDIR}/OpenCL-command-buffer/LU")
add_test(NAME polybenchGPU_MVT_cmd_buffer COMMAND ${TEST_LAUNCHER_SCRIPT} ./mvt_cmd_buffer
       WORKING_DIRECTORY "${TS_BUILDDIR}/OpenCL-command-buffer/MVT")
add_test(NAME polybenchGPU_SYR2K_cmd_buffer COMMAND ${TEST_LAUNCHER_SCRIPT} ./syr2k_cmd_buffer
       WORKING_DIRECTORY "${TS_BUILDDIR}/OpenCL-command-buffer/SYR2K")
add_test(NAME polybenchGPU_SYRK_cmd_buffer COMMAND ${TEST_LAUNCHER_SCRIPT} ./syrk_cmd_buffer
       WORKING_DIRECTORY "${TS_BUILDDIR}/OpenCL-command-buffer/SYRK")

# currently failing verification:
#  polybenchGPU_ADI
#  polybenchGPU_ADI_cmd_buffer
#  polybenchGPU_FDTD-2D
#  polybenchGPU_FDTD-2D_cmd_buffer

set_tests_properties(
  polybenchGPU_2DCONV
  polybenchGPU_2MM
  polybenchGPU_3DCONV
  polybenchGPU_3MM
  polybenchGPU_ATAX
  polybenchGPU_BICG
  polybenchGPU_CORR
  polybenchGPU_COVAR
  polybenchGPU_GEMM
  polybenchGPU_GEMVER
  polybenchGPU_GESUMMV
  polybenchGPU_GRAMSCHM
  polybenchGPU_JACOBI1D
  polybenchGPU_JACOBI2D
  polybenchGPU_LU
  polybenchGPU_MVT
  polybenchGPU_SYR2K
  polybenchGPU_SYRK
  polybenchGPU_2DCONV_cmd_buffer
  polybenchGPU_2MM_cmd_buffer
  polybenchGPU_3DCONV_cmd_buffer
  polybenchGPU_3MM_cmd_buffer
  polybenchGPU_ATAX_cmd_buffer
  polybenchGPU_BICG_cmd_buffer
  polybenchGPU_CORR_cmd_buffer
  polybenchGPU_COVAR_cmd_buffer
  polybenchGPU_GEMM_cmd_buffer
  polybenchGPU_GEMVER_cmd_buffer
  polybenchGPU_GESUMMV_cmd_buffer
  polybenchGPU_GRAMSCHM_cmd_buffer
  polybenchGPU_JACOBI1D_cmd_buffer
  polybenchGPU_JACOBI2D_cmd_buffer
  polybenchGPU_LU_cmd_buffer
  polybenchGPU_MVT_cmd_buffer
  polybenchGPU_SYR2K_cmd_buffer
  polybenchGPU_SYRK_cmd_buffer
  PROPERTIES
  # RUN_SERIAL to avoid running anything else at the same time
  RUN_SERIAL ON
  PASS_REGULAR_EXPRESSION "pass"
  FAIL_REGULAR_EXPRESSION "fail"
  SKIP_REGULAR_EXPRESSION "skipped"
  LABELS "polybenchGPU;mlir")

set_tests_properties(
  polybenchGPU_CORR
  polybenchGPU_GRAMSCHM
  polybenchGPU_2MM_cmd_buffer
  polybenchGPU_2MM
  PROPERTIES
  LABELS "long")

foreach(BENCH IN ITEMS 2DCONV 2MM 3DCONV 3MM ATAX BICG CORR COVAR GEMM GEMVER GESUMMV GRAMSCHM
    JACOBI1D JACOBI2D LU MVT SYR2K SYRK 2DCONV_cmd_buffer 2MM_cmd_buffer
    3DCONV_cmd_buffer 3MM_cmd_buffer ATAX_cmd_buffer BICG_cmd_buffer
    CORR_cmd_buffer COVAR_cmd_buffer GEMM_cmd_buffer GEMVER_cmd_buffer
    GESUMMV_cmd_buffer GRAMSCHM_cmd_buffer JACOBI1D_cmd_buffer JACOBI2D_cmd_buffer
    LU_cmd_buffer MVT_cmd_buffer SYR2K_cmd_buffer SYRK_cmd_buffer)
endforeach()
