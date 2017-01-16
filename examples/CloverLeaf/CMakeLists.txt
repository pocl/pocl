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

set(TS_NAME CloverLeaf)
set(TS_BASEDIR "${TESTSUITE_BASEDIR}/${TS_NAME}")
set(TS_BUILDDIR "${TS_BASEDIR}/src/${TS_NAME}")
set(TS_SRCDIR "${TESTSUITE_SOURCE_BASEDIR}/${TS_NAME}")

if(UNIX)
  pkg_check_modules(OPENMPI QUIET ompi)
  if (OPENMPI_FOUND)
    set(EXTRA_LIB_HINTS "HINTS" "${OPENMPI_LIBRARY_DIRS}")
  endif()
endif()

#You need to have "openmpi" and "gfortran" packages installed to build it.
find_program(MPI_F90 mpif90)
find_program(MPI_CC  mpicc)
find_program(MPI_CXX  mpiCC)
find_program(GFORTRAN gfortran)
find_library(MPI_LIB mpi_cxx ${EXTRA_LIB_HINTS})

if(HAVE_GIT)
if (MPI_F90 AND MPI_CC AND MPI_CXX AND MPI_LIB AND GFORTRAN)

  message(STATUS "Enabling testsuite ${TS_NAME}")
  list(APPEND ACTUALLY_ENABLED_TESTSUITES "${TS_NAME}")
  set(ACTUALLY_ENABLED_TESTSUITES ${ACTUALLY_ENABLED_TESTSUITES} PARENT_SCOPE)

  ExternalProject_Add(
    ${TS_NAME}
    PREFIX "${TS_BASEDIR}"
    #DOWNLOAD_COMMAND "/bin/true"
    GIT_REPOSITORY "https://github.com/UK-MAC/CloverLeaf_OpenCL.git"
    GIT_TAG "Bristol"
    CONFIGURE_COMMAND "/bin/true"
    BUILD_IN_SOURCE 1
    BUILD_COMMAND
      make COMPILER=GNU
      "C_OPTIONS=-g -I${CMAKE_SOURCE_DIR}/include -DCL_USE_DEPRECATED_OPENCL_1_2_APIS -DCL_USE_DEPRECATED_OPENCL_1_1_APIS"
      "MPI_COMPILER=${MPI_F90}"
      "C_MPI_COMPILER=${MPI_CC}"
      "CXX_MPI_COMPILER=${MPI_CXX}"
      "MPICXX_LIB=${MPI_LIB}"
    INSTALL_COMMAND "/bin/true"
  )

  set_target_properties(${TS_NAME} PROPERTIES EXCLUDE_FROM_ALL TRUE)
  add_dependencies(prepare_examples ${TS_NAME})

  add_test(NAME CloverLeaf
           COMMAND "${TS_BUILDDIR}/clover_leaf")

  set_tests_properties(CloverLeaf
    PROPERTIES
      LABELS "CloverLeaf")

else()
  message(STATUS "Disabling testsuite ${TS_NAME}, required files (OpenMPI/gfortran) not found" )
endif()
else()
  message(STATUS "Disabling testsuite ${TS_NAME}, requires git to checkout sources")
endif()
