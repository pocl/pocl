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

set(TS_NAME "PyOpenCL")
set(TS_BASEDIR "${TESTSUITE_BASEDIR}/${TS_NAME}")
set(TS_BUILDDIR "${TS_BASEDIR}/src/${TS_NAME}-build")
set(TS_SRCDIR "${TS_BASEDIR}/src/${TS_NAME}")

# TODO for whatever reason, compyte (required for running tests)
# is not included in pyopencl release tars. For now force using git
set(EXAMPLES_USE_GIT_MASTER 1)

if(EXAMPLES_USE_GIT_MASTER)
  set(FETCH_SOURCE
    GIT_REPOSITORY "https://github.com/inducer/pyopencl.git"
    GIT_TAG v2023.1.4
    GIT_SUBMODULES "pyopencl/compyte")
else()
  set(FETCH_SOURCE URL "https://github.com/inducer/pyopencl/archive/v2023.1.4.tar.gz")
endif()

if(NOT TESTS_USE_ICD)
  message(STATUS "Disabling testsuite ${TS_NAME}, requires ocl-icd")
elseif(NOT BASH)
  message(STATUS "Disabling testsuite ${TS_NAME}, can't find bash shell")
else()

find_program(VIRTUALENV NAMES "virtualenv")
find_package(Python3 3.8 COMPONENTS Interpreter Development)
if(NOT Python3_FOUND)
  message(STATUS "Disabling testsuite ${TS_NAME}, can't find suitable python3")
  return()
else()
  message(STATUS "Using Python3: ${Python3_EXECUTABLE}")
  set(PYTHON_INTERP "${Python3_EXECUTABLE}")
endif()

if(VIRTUALENV)

  message(STATUS "Enabling testsuite ${TS_NAME}")
  list(APPEND ACTUALLY_ENABLED_TESTSUITES "${TS_NAME}")
  set(ACTUALLY_ENABLED_TESTSUITES ${ACTUALLY_ENABLED_TESTSUITES} PARENT_SCOPE)

  ExternalProject_Add(
    ${TS_NAME}
    PREFIX "${TS_BASEDIR}"
    ${FETCH_SOURCE}

    PATCH_COMMAND /bin/sh -c "patch -N -r- -p1 -i ${CMAKE_CURRENT_SOURCE_DIR}/CL_UNORM_INT24.patch || echo patch already applied"

    # have to use external scripts, because CMake has /bin/sh hardcoded,
    # and /bin/sh on Ubuntu is dash, which doesn't support "source"
    # command.
    CONFIGURE_COMMAND "${BASH}" "${CMAKE_CURRENT_SOURCE_DIR}/configure.sh"
      "${TS_BASEDIR}"
      "${TS_BUILDDIR}"
      "${TS_SRCDIR}"
      "${VIRTUALENV}"
      "${PYTHON_INTERP}"

    BUILD_COMMAND "${BASH}" "${CMAKE_CURRENT_SOURCE_DIR}/build.sh"
      "${TS_BASEDIR}"
      "${TS_BUILDDIR}"
      "${TS_SRCDIR}"

    INSTALL_COMMAND "${BASH}" "${CMAKE_CURRENT_SOURCE_DIR}/install.sh"
      "${TS_BASEDIR}"
      "${TS_BUILDDIR}"
      "${TS_SRCDIR}"
  )


  set_target_properties(${TS_NAME} PROPERTIES EXCLUDE_FROM_ALL TRUE)
  add_dependencies(prepare_examples ${TS_NAME})

#  add_test(NAME "PyOpenCL_" COMMAND "${BASH}" "${CMAKE_CURRENT_SOURCE_DIR}/runtest.sh" "${TS_BASEDIR}" "${TS_BUILDDIR}" "${TS_SRCDIR}"  )
  add_test(NAME "PyOpenCL_test_algorithm" COMMAND "${BASH}" "${CMAKE_CURRENT_SOURCE_DIR}/runtest.sh" "${TS_BASEDIR}" "${TS_BUILDDIR}" "${TS_SRCDIR}" test_algorithm.py )
  add_test(NAME "PyOpenCL_test_array" COMMAND "${BASH}" "${CMAKE_CURRENT_SOURCE_DIR}/runtest.sh" "${TS_BASEDIR}" "${TS_BUILDDIR}" "${TS_SRCDIR}" test_array.py )
  add_test(NAME "PyOpenCL_test_arrays_in_structs" COMMAND "${BASH}" "${CMAKE_CURRENT_SOURCE_DIR}/runtest.sh" "${TS_BASEDIR}" "${TS_BUILDDIR}" "${TS_SRCDIR}" test_arrays_in_structs.py )
  add_test(NAME "PyOpenCL_test_clmath" COMMAND "${BASH}" "${CMAKE_CURRENT_SOURCE_DIR}/runtest.sh" "${TS_BASEDIR}" "${TS_BUILDDIR}" "${TS_SRCDIR}" test_clmath.py )
  add_test(NAME "PyOpenCL_test_clrandom" COMMAND "${BASH}" "${CMAKE_CURRENT_SOURCE_DIR}/runtest.sh" "${TS_BASEDIR}" "${TS_BUILDDIR}" "${TS_SRCDIR}" test_clrandom.py )
  add_test(NAME "PyOpenCL_test_enqueue_copy" COMMAND "${BASH}" "${CMAKE_CURRENT_SOURCE_DIR}/runtest.sh" "${TS_BASEDIR}" "${TS_BUILDDIR}" "${TS_SRCDIR}" test_enqueue_copy.py )
# The wrapper test seems to be broken. Tests an AMD topology constructor
# which is not found. Doesn't seem to have anything to do with PoCL-
#  add_test(NAME "PyOpenCL_test_wrapper" COMMAND "${BASH}" "${CMAKE_CURRENT_SOURCE_DIR}/runtest.sh" "${TS_BASEDIR}" "${TS_BUILDDIR}" "${TS_SRCDIR}" test_wrapper.py )

  set_tests_properties("PyOpenCL_test_algorithm"
    "PyOpenCL_test_array" "PyOpenCL_test_arrays_in_structs"
    "PyOpenCL_test_clmath" "PyOpenCL_test_clrandom"
    "PyOpenCL_test_enqueue_copy" #"PyOpenCL_test_wrapper"
    PROPERTIES
      ENVIRONMENT "PYOPENCL_NO_CACHE=1;PYOPENCL_TEST=portable"
      COST 400.0
      LABELS "PyOpenCL")

else()

  message(STATUS "Disabling testsuite ${TS_NAME}, requires python virtualenv")

endif()

endif()
