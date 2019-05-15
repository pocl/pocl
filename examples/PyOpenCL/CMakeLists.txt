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
      GIT_SUBMODULES "pyopencl/compyte")
else()
  set(FETCH_SOURCE URL "https://github.com/inducer/pyopencl/archive/v2017.2.tar.gz")
endif()

# The ICD test is broken. Disable the check for now.
if(FALSE) # NOT TESTS_USE_ICD)
  message(STATUS "Disabling testsuite ${TS_NAME}, requires ocl-icd")
elseif(NOT BASH)
  message(STATUS "Disabling testsuite ${TS_NAME}, can't find bash shell")
else()

find_program(VIRTUALENV NAMES "virtualenv")
find_program(PYTHON_INTERP NAMES python3 python2 python)

if(VIRTUALENV)

  message(STATUS "Enabling testsuite ${TS_NAME}")
  list(APPEND ACTUALLY_ENABLED_TESTSUITES "${TS_NAME}")
  set(ACTUALLY_ENABLED_TESTSUITES ${ACTUALLY_ENABLED_TESTSUITES} PARENT_SCOPE)

  ExternalProject_Add(
    ${TS_NAME}
    PREFIX "${TS_BASEDIR}"
    ${FETCH_SOURCE}

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

  add_test(NAME "PyOpenCL"
           COMMAND "${BASH}" "${CMAKE_CURRENT_SOURCE_DIR}/runtest.sh"
            "${TS_BASEDIR}"
            "${TS_BUILDDIR}"
            "${TS_SRCDIR}"
           )

  set_tests_properties("PyOpenCL"
    PROPERTIES
      ENVIRONMENT "PYOPENCL_NO_CACHE=1;PYOPENCL_TEST=portable"
      LABELS "PyOpenCL")

else()

  message(STATUS "Disabling testsuite ${TS_NAME}, requires python virtualenv")

endif()

endif()
