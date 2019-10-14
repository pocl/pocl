#=============================================================================
#   CMake build file for CLBlast's test suite.
#
#   Copyright (c) 2019 Pekka Jääskeläinen
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

set(TS_NAME "Glow")
set(TS_BASEDIR "${TESTSUITE_BASEDIR}/${TS_NAME}")
set(TS_BUILDDIR "${TS_BASEDIR}/src/${TS_NAME}-build")
set(TS_SRCDIR "${TESTSUITE_SOURCE_BASEDIR}/${TS_NAME}")

# This tests the Parmance-tested Glow branch in 32b dimension mode.
# The branch occasionally has OpenCL-related or misc. patches on top,
# and is rebased on top of the Glow master (with force push).
set(FETCH_SOURCE GIT_REPOSITORY "https://github.com/parmance/glow.git")

message(STATUS "Enabling testsuite ${TS_NAME}")
list(APPEND ACTUALLY_ENABLED_TESTSUITES "${TS_NAME}")
set(ACTUALLY_ENABLED_TESTSUITES ${ACTUALLY_ENABLED_TESTSUITES} PARENT_SCOPE)

ExternalProject_Add(
  ${TS_NAME}
  ${FETCH_SOURCE}
  PREFIX "${TS_BASEDIR}"

  CMAKE_ARGS
    -DCMAKE_BUILD_TYPE=Release
    -DGLOW_WITH_CPU=OFF
    -DGLOW_WITH_OPENCL=ON
    -DTENSOR_DIMS_32_BITS=ON
    # This is needed with some drivers due to https://gcc.gnu.org/bugzilla/show_bug.cgi?id=67791
    -DCMAKE_CXX_FLAGS=-pthread
  INSTALL_COMMAND /bin/true
)

set_target_properties(${TS_NAME} PROPERTIES EXCLUDE_FROM_ALL TRUE)
add_dependencies(prepare_examples ${TS_NAME})

# Run the basic "ninja check" suite excluding known problematic
# cases. MLTest is a very long running suite especially on simulated
# targets, so exclude it for now and perhaps add as a separate test case
# later.
add_test(NAME OpenCLRegressionSuite
  COMMAND "${CMAKE_CTEST_COMMAND}"
  "-E" "^BackendCorrectnessTest|^MLTest|^RecommendationSystemTest"
  "-LE" "EXPENSIVE"
  WORKING_DIRECTORY "${TS_BUILDDIR}")

set_property(TEST
  OpenCLRegressionSuite
  APPEND PROPERTY LABELS "${TS_NAME}")
