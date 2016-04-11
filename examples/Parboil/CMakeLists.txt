#=============================================================================
#   CMake build system files
#
#   Copyright (c) 2015 pocl developers
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

set(TS_NAME "Parboil")
set(TS_BASEDIR "${TESTSUITE_BASEDIR}/${TS_NAME}")
set(TS_BUILDDIR "${TS_BASEDIR}/src/${TS_NAME}")
set(TS_SRCDIR "${TESTSUITE_SOURCE_BASEDIR}/${TS_NAME}")

set(PB_DRIVER "${TS_SRCDIR}/pb2.5driver.tgz")
set(PB_BENCH "${TS_SRCDIR}/pb2.5benchmarks.tgz")
set(PB_DATASET "${TS_SRCDIR}/pb2.5datasets_standard.tgz")
set(PB_MAKE "${CMAKE_BINARY_DIR}/Parboil.makefile")

find_package(PythonInterp 2.7 QUIET)

if (PYTHONINTERP_FOUND)

if (EXISTS "${PB_DRIVER}" AND
    EXISTS "${PB_BENCH}" AND
    EXISTS "${PB_DATASET}")

  file(WRITE "${PB_MAKE}" "OPENCL_PATH=${CMAKE_SOURCE_DIR}\n")
  #file(APPEND "${PB_MAKE}" "OPENCL_LIB_PATH=/usr/lib\n")

  message(STATUS "Enabling testsuite ${TS_NAME}")
  list(APPEND ACTUALLY_ENABLED_TESTSUITES "${TS_NAME}")
  set(ACTUALLY_ENABLED_TESTSUITES ${ACTUALLY_ENABLED_TESTSUITES} PARENT_SCOPE)

  ExternalProject_Add(
    ${TS_NAME}
    PREFIX "${TS_BASEDIR}"
    DOWNLOAD_COMMAND pwd &&
      tar xvzf "${PB_DRIVER}" && cd parboil &&
      tar xvzf "${PB_BENCH}" &&
      tar xvzf "${PB_DATASET}" && cd .. &&
      rmdir Parboil && mv parboil Parboil &&
      cp "${PB_MAKE}" ./Parboil/common/Makefile.conf

    CONFIGURE_COMMAND /bin/true
    BUILD_IN_SOURCE 1
    BUILD_COMMAND pwd &&
      "${PYTHON_EXECUTABLE}" ./parboil compile spmv opencl_base &&
      "${PYTHON_EXECUTABLE}" ./parboil compile stencil opencl_base &&
      "${PYTHON_EXECUTABLE}" ./parboil compile tpacf opencl_base &&
      "${PYTHON_EXECUTABLE}" ./parboil compile cutcp opencl_base &&
      "${PYTHON_EXECUTABLE}" ./parboil compile mri-gridding opencl_base &&
      "${PYTHON_EXECUTABLE}" ./parboil compile sad opencl_base &&
      "${PYTHON_EXECUTABLE}" ./parboil compile bfs opencl_base &&
      "${PYTHON_EXECUTABLE}" ./parboil compile histo opencl_base &&
      "${PYTHON_EXECUTABLE}" ./parboil compile sgemm opencl_base &&
      "${PYTHON_EXECUTABLE}" ./parboil compile lbm opencl_base &&
      "${PYTHON_EXECUTABLE}" ./parboil compile mri-q opencl

    INSTALL_COMMAND /bin/true
  )

  set_target_properties(${TS_NAME} PROPERTIES EXCLUDE_FROM_ALL TRUE)
  add_dependencies(prepare_examples ${TS_NAME})

  add_test(NAME parboil_spmv
           COMMAND "${PYTHON_EXECUTABLE}" ./parboil run
           spmv opencl_base small
           WORKING_DIRECTORY "${TS_BUILDDIR}")
  add_test(NAME parboil_stencil
           COMMAND "${PYTHON_EXECUTABLE}" ./parboil run
           stencil opencl_base small
           WORKING_DIRECTORY "${TS_BUILDDIR}")
  add_test(NAME parboil_tpacf
           COMMAND "${PYTHON_EXECUTABLE}" ./parboil run
           tpacf opencl_base small
           WORKING_DIRECTORY "${TS_BUILDDIR}")
  add_test(NAME parboil_cutcp
           COMMAND "${PYTHON_EXECUTABLE}" ./parboil run
           cutcp opencl_base small
           WORKING_DIRECTORY "${TS_BUILDDIR}")
  add_test(NAME parboil_mri_gridding
           COMMAND "${PYTHON_EXECUTABLE}" ./parboil run
           mri-gridding opencl_base small
           WORKING_DIRECTORY "${TS_BUILDDIR}")
  add_test(NAME parboil_sad
           COMMAND "${PYTHON_EXECUTABLE}" ./parboil run
           sad opencl_base default
           WORKING_DIRECTORY "${TS_BUILDDIR}")
  add_test(NAME parboil_bfs
           COMMAND "${PYTHON_EXECUTABLE}" ./parboil run
           bfs opencl_base NY
           WORKING_DIRECTORY "${TS_BUILDDIR}")
  add_test(NAME parboil_histo
           COMMAND "${PYTHON_EXECUTABLE}" ./parboil run
           histo opencl_base default
           WORKING_DIRECTORY "${TS_BUILDDIR}")
  add_test(NAME parboil_sgemm
           COMMAND "${PYTHON_EXECUTABLE}" ./parboil run
           sgemm opencl_base medium
           WORKING_DIRECTORY "${TS_BUILDDIR}")
  add_test(NAME parboil_mri_q
           COMMAND "${PYTHON_EXECUTABLE}" ./parboil run
           mri-q opencl small
           WORKING_DIRECTORY "${TS_BUILDDIR}")
  add_test(NAME parboil_lbm
           COMMAND "${PYTHON_EXECUTABLE}" ./parboil run
           lbm opencl_base short
           WORKING_DIRECTORY "${TS_BUILDDIR}")

  set_tests_properties(
    parboil_spmv
    parboil_stencil
    parboil_tpacf
    parboil_cutcp
    parboil_mri_gridding
    parboil_sad
    parboil_bfs
    parboil_histo
    parboil_sgemm
    parboil_mri_q
    parboil_lbm

    PROPERTIES
      LABELS "Parboil")

else()
  message(STATUS "Disabling testsuite ${TS_NAME}, required files (parboil sources) not found" )
endif()
else()
  message(STATUS "Disabling testsuite ${TS_NAME}, requires python 2.7 (Parboil is not python 3 compatible)")
endif()
