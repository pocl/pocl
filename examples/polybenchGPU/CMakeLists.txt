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
else()
  message(STATUS "Enabling testsuite ${TS_NAME}")
  list(APPEND ACTUALLY_ENABLED_TESTSUITES "${TS_NAME}")
  set(ACTUALLY_ENABLED_TESTSUITES ${ACTUALLY_ENABLED_TESTSUITES} PARENT_SCOPE)

  set(OCL_LIBRARY "$<TARGET_FILE:${POCL_LIBRARY_NAME}>")

  ExternalProject_Add(
    ${TS_NAME}
    PREFIX "${TS_BASEDIR}"
    GIT_REPOSITORY "https://github.com/cpc/polybench.git"
    GIT_TAG "f71ac9fb64d0cca7144538efbde7793d60d3b239"
    ${GIT_OPTIONS}
    CMAKE_ARGS
      -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
      "-DOPENCL_INCLUDE_PATH=${CMAKE_SOURCE_DIR}/include/"
      "-DOPENCL_opencl_LIBRARY:STRING=${OCL_LIBRARY}"
    INSTALL_COMMAND ""
  )

  set_target_properties(${TS_NAME} PROPERTIES EXCLUDE_FROM_ALL TRUE)
  add_dependencies(prepare_examples ${TS_NAME})

  add_test(NAME polybenchGPU_2DCONV COMMAND ./2DConvolution
         WORKING_DIRECTORY "${TS_BUILDDIR}/OpenCL/2DCONV")
  add_test(NAME polybenchGPU_2MM COMMAND ./2mm
         WORKING_DIRECTORY "${TS_BUILDDIR}/OpenCL/2MM")
  add_test(NAME polybenchGPU_3DCONV COMMAND ./3DConvolution
         WORKING_DIRECTORY "${TS_BUILDDIR}/OpenCL/3DCONV")
  add_test(NAME polybenchGPU_3MM COMMAND ./3mm
         WORKING_DIRECTORY "${TS_BUILDDIR}/OpenCL/3MM")
  add_test(NAME polybenchGPU_ADI COMMAND ./adi
         WORKING_DIRECTORY "${TS_BUILDDIR}/OpenCL/ADI")
  add_test(NAME polybenchGPU_ATAX COMMAND ./atax
         WORKING_DIRECTORY "${TS_BUILDDIR}/OpenCL/ATAX")
  add_test(NAME polybenchGPU_BICG COMMAND ./bicg
         WORKING_DIRECTORY "${TS_BUILDDIR}/OpenCL/BICG")
  add_test(NAME polybenchGPU_CORR COMMAND ./correlation
         WORKING_DIRECTORY "${TS_BUILDDIR}/OpenCL/CORR")
  add_test(NAME polybenchGPU_COVAR COMMAND ./covariance
         WORKING_DIRECTORY "${TS_BUILDDIR}/OpenCL/COVAR")
  add_test(NAME polybenchGPU_FDTD-2D COMMAND ./fdtd2d
         WORKING_DIRECTORY "${TS_BUILDDIR}/OpenCL/FDTD-2D")
  add_test(NAME polybenchGPU_GEMM COMMAND ./gemm
         WORKING_DIRECTORY "${TS_BUILDDIR}/OpenCL/GEMM")
  add_test(NAME polybenchGPU_GEMVER COMMAND ./gemver
         WORKING_DIRECTORY "${TS_BUILDDIR}/OpenCL/GEMVER")
  add_test(NAME polybenchGPU_GESUMMV COMMAND ./gesummv
         WORKING_DIRECTORY "${TS_BUILDDIR}/OpenCL/GESUMMV")
  add_test(NAME polybenchGPU_GRAMSCHM COMMAND ./gramschmidt
         WORKING_DIRECTORY "${TS_BUILDDIR}/OpenCL/GRAMSCHM")
  add_test(NAME polybenchGPU_JACOBI1D COMMAND ./jacobi1d
         WORKING_DIRECTORY "${TS_BUILDDIR}/OpenCL/JACOBI1D")
  add_test(NAME polybenchGPU_JACOBI2D COMMAND ./jacobi2d
         WORKING_DIRECTORY "${TS_BUILDDIR}/OpenCL/JACOBI2D")
  add_test(NAME polybenchGPU_LU COMMAND ./lu
         WORKING_DIRECTORY "${TS_BUILDDIR}/OpenCL/LU")
  add_test(NAME polybenchGPU_MVT COMMAND ./mvt
         WORKING_DIRECTORY "${TS_BUILDDIR}/OpenCL/MVT")
  add_test(NAME polybenchGPU_SYR2K COMMAND ./syr2k
         WORKING_DIRECTORY "${TS_BUILDDIR}/OpenCL/SYR2K")
  add_test(NAME polybenchGPU_SYRK COMMAND ./syrk
         WORKING_DIRECTORY "${TS_BUILDDIR}/OpenCL/SYRK")

  # Command buffer tests
  add_test(NAME polybenchGPU_2DCONV_cmd_buffer COMMAND ./2DConvolution_cmd_buffer
         WORKING_DIRECTORY "${TS_BUILDDIR}/OpenCL-command-buffer/2DCONV")
  add_test(NAME polybenchGPU_2MM_cmd_buffer COMMAND ./2mm_cmd_buffer
         WORKING_DIRECTORY "${TS_BUILDDIR}/OpenCL-command-buffer/2MM")
  add_test(NAME polybenchGPU_3DCONV_cmd_buffer COMMAND ./3DConvolution_cmd_buffer
         WORKING_DIRECTORY "${TS_BUILDDIR}/OpenCL-command-buffer/3DCONV")
  add_test(NAME polybenchGPU_3MM_cmd_buffer COMMAND ./3mm_cmd_buffer
         WORKING_DIRECTORY "${TS_BUILDDIR}/OpenCL-command-buffer/3MM")
  add_test(NAME polybenchGPU_ADI_cmd_buffer COMMAND ./adi_cmd_buffer
         WORKING_DIRECTORY "${TS_BUILDDIR}/OpenCL-command-buffer/ADI")
  add_test(NAME polybenchGPU_ATAX_cmd_buffer COMMAND ./atax_cmd_buffer
         WORKING_DIRECTORY "${TS_BUILDDIR}/OpenCL-command-buffer/ATAX")
  add_test(NAME polybenchGPU_BICG_cmd_buffer COMMAND ./bicg_cmd_buffer
         WORKING_DIRECTORY "${TS_BUILDDIR}/OpenCL-command-buffer/BICG")
  add_test(NAME polybenchGPU_CORR_cmd_buffer COMMAND ./correlation_cmd_buffer
         WORKING_DIRECTORY "${TS_BUILDDIR}/OpenCL-command-buffer/CORR")
  add_test(NAME polybenchGPU_COVAR_cmd_buffer COMMAND ./covariance_cmd_buffer
         WORKING_DIRECTORY "${TS_BUILDDIR}/OpenCL-command-buffer/COVAR")
  add_test(NAME polybenchGPU_FDTD-2D_cmd_buffer COMMAND ./fdtd2d_cmd_buffer
         WORKING_DIRECTORY "${TS_BUILDDIR}/OpenCL-command-buffer/FDTD-2D")
  add_test(NAME polybenchGPU_GEMM_cmd_buffer COMMAND ./gemm_cmd_buffer
         WORKING_DIRECTORY "${TS_BUILDDIR}/OpenCL-command-buffer/GEMM")
  add_test(NAME polybenchGPU_GEMVER_cmd_buffer COMMAND ./gemver_cmd_buffer
         WORKING_DIRECTORY "${TS_BUILDDIR}/OpenCL-command-buffer/GEMVER")
  add_test(NAME polybenchGPU_GESUMMV_cmd_buffer COMMAND ./gesummv_cmd_buffer
         WORKING_DIRECTORY "${TS_BUILDDIR}/OpenCL-command-buffer/GESUMMV")
  add_test(NAME polybenchGPU_GRAMSCHM_cmd_buffer COMMAND ./gramschmidt_cmd_buffer
         WORKING_DIRECTORY "${TS_BUILDDIR}/OpenCL-command-buffer/GRAMSCHM")
  add_test(NAME polybenchGPU_JACOBI1D_cmd_buffer COMMAND ./jacobi1d_cmd_buffer
         WORKING_DIRECTORY "${TS_BUILDDIR}/OpenCL-command-buffer/JACOBI1D")
  add_test(NAME polybenchGPU_JACOBI2D_cmd_buffer COMMAND ./jacobi2d_cmd_buffer
         WORKING_DIRECTORY "${TS_BUILDDIR}/OpenCL-command-buffer/JACOBI2D")
  add_test(NAME polybenchGPU_LU_cmd_buffer COMMAND ./lu_cmd_buffer
         WORKING_DIRECTORY "${TS_BUILDDIR}/OpenCL-command-buffer/LU")
  add_test(NAME polybenchGPU_MVT_cmd_buffer COMMAND ./mvt_cmd_buffer
         WORKING_DIRECTORY "${TS_BUILDDIR}/OpenCL-command-buffer/MVT")
  add_test(NAME polybenchGPU_SYR2K_cmd_buffer COMMAND ./syr2k_cmd_buffer
         WORKING_DIRECTORY "${TS_BUILDDIR}/OpenCL-command-buffer/SYR2K")
  add_test(NAME polybenchGPU_SYRK_cmd_buffer COMMAND ./syrk_cmd_buffer
         WORKING_DIRECTORY "${TS_BUILDDIR}/OpenCL-command-buffer/SYRK")

  set_tests_properties(
    polybenchGPU_2DCONV
    polybenchGPU_2MM
    polybenchGPU_3DCONV
    polybenchGPU_3MM
    polybenchGPU_ADI
    polybenchGPU_ATAX
    polybenchGPU_BICG
    polybenchGPU_CORR
    polybenchGPU_COVAR
    polybenchGPU_FDTD-2D
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
    polybenchGPU_ADI_cmd_buffer
    polybenchGPU_ATAX_cmd_buffer
    polybenchGPU_BICG_cmd_buffer
    polybenchGPU_CORR_cmd_buffer
    polybenchGPU_COVAR_cmd_buffer
    polybenchGPU_FDTD-2D_cmd_buffer
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
    PASS_REGULAR_EXPRESSION "pass"
    FAIL_REGULAR_EXPRESSION "fail"
    SKIP_REGULAR_EXPRESSION "skipped"
    LABELS "polybenchGPU;mlir")

endif()
