#=============================================================================
#   CMake build file for CLBlast's test suite.
#
#   Copyright (c) 2018 Pekka Jääskeläinen
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

set(TS_NAME "CLBlast")
set(TS_BASEDIR "${TESTSUITE_BASEDIR}/${TS_NAME}")
set(TS_BUILDDIR "${TS_BASEDIR}/src/${TS_NAME}-build")
set(TS_SRCDIR "${TESTSUITE_SOURCE_BASEDIR}/${TS_NAME}")

set(FETCH_SOURCE GIT_REPOSITORY "git@github.com:CNugteren/CLBlast.git")

message(STATUS "Enabling testsuite ${TS_NAME}")
list(APPEND ACTUALLY_ENABLED_TESTSUITES "${TS_NAME}")
set(ACTUALLY_ENABLED_TESTSUITES ${ACTUALLY_ENABLED_TESTSUITES} PARENT_SCOPE)

ExternalProject_Add(
  ${TS_NAME}
  ${FETCH_SOURCE}
  PREFIX "${TS_BASEDIR}"

  CMAKE_ARGS
    -DTESTS=ON
    -DCLIENTS=ON
    # This is needed with some drivers due to https://gcc.gnu.org/bugzilla/show_bug.cgi?id=67791
    -DCMAKE_CXX_FLAGS=-pthread
  INSTALL_COMMAND /bin/true
)

set_target_properties(${TS_NAME} PROPERTIES EXCLUDE_FROM_ALL TRUE)
add_dependencies(prepare_examples ${TS_NAME})

add_test(NAME clblast_test_xswap
  COMMAND "${TS_BUILDDIR}/clblast_test_xswap")

add_test(NAME clblast_test_xscal
  COMMAND "${TS_BUILDDIR}/clblast_test_xscal")

add_test(NAME clblast_test_xcopy
  COMMAND "${TS_BUILDDIR}/clblast_test_xcopy")

add_test(NAME clblast_test_xaxpy
  COMMAND "${TS_BUILDDIR}/clblast_test_xaxpy")

add_test(NAME clblast_test_xdot
  COMMAND "${TS_BUILDDIR}/clblast_test_xdot")

add_test(NAME clblast_test_xdotu
  COMMAND "${TS_BUILDDIR}/clblast_test_xdotu")

add_test(NAME clblast_test_xdotc
  COMMAND "${TS_BUILDDIR}/clblast_test_xdotc")

add_test(NAME clblast_test_xnrm2
  COMMAND "${TS_BUILDDIR}/clblast_test_xnrm2")

add_test(NAME clblast_test_xasum
  COMMAND "${TS_BUILDDIR}/clblast_test_xasum")

add_test(NAME clblast_test_xamax
  COMMAND "${TS_BUILDDIR}/clblast_test_xamax")

add_test(NAME clblast_test_xgemv
  COMMAND "${TS_BUILDDIR}/clblast_test_xgemv")

add_test(NAME clblast_test_xgbmv
  COMMAND "${TS_BUILDDIR}/clblast_test_xgbmv")

add_test(NAME clblast_test_xhemv
  COMMAND "${TS_BUILDDIR}/clblast_test_xhemv")

add_test(NAME clblast_test_xhbmv
  COMMAND "${TS_BUILDDIR}/clblast_test_xhbmv")

add_test(NAME clblast_test_xhpmv
  COMMAND "${TS_BUILDDIR}/clblast_test_xhpmv")

add_test(NAME clblast_test_xsymv
  COMMAND "${TS_BUILDDIR}/clblast_test_xsymv")

add_test(NAME clblast_test_xsbmv
  COMMAND "${TS_BUILDDIR}/clblast_test_xsbmv")

add_test(NAME clblast_test_xspmv
  COMMAND "${TS_BUILDDIR}/clblast_test_xspmv")

add_test(NAME clblast_test_xtrmv
  COMMAND "${TS_BUILDDIR}/clblast_test_xtrmv")

add_test(NAME clblast_test_xtbmv
  COMMAND "${TS_BUILDDIR}/clblast_test_xtbmv")

add_test(NAME clblast_test_xtpmv
  COMMAND "${TS_BUILDDIR}/clblast_test_xtpmv")

add_test(NAME clblast_test_xtrsv
  COMMAND "${TS_BUILDDIR}/clblast_test_xtrsv")

add_test(NAME clblast_test_xger
  COMMAND "${TS_BUILDDIR}/clblast_test_xger")

add_test(NAME clblast_test_xgeru
  COMMAND "${TS_BUILDDIR}/clblast_test_xgeru")

add_test(NAME clblast_test_xgerc
  COMMAND "${TS_BUILDDIR}/clblast_test_xgerc")

add_test(NAME clblast_test_xher
  COMMAND "${TS_BUILDDIR}/clblast_test_xher")

add_test(NAME clblast_test_xhpr
  COMMAND "${TS_BUILDDIR}/clblast_test_xhpr")

add_test(NAME clblast_test_xher2
  COMMAND "${TS_BUILDDIR}/clblast_test_xher2")

add_test(NAME clblast_test_xhpr2
  COMMAND "${TS_BUILDDIR}/clblast_test_xhpr2")

add_test(NAME clblast_test_xsyr
  COMMAND "${TS_BUILDDIR}/clblast_test_xsyr")

add_test(NAME clblast_test_xspr
  COMMAND "${TS_BUILDDIR}/clblast_test_xspr")

add_test(NAME clblast_test_xsyr2
  COMMAND "${TS_BUILDDIR}/clblast_test_xsyr2")

add_test(NAME clblast_test_xspr2
  COMMAND "${TS_BUILDDIR}/clblast_test_xspr2")

add_test(NAME clblast_test_xgemm
  COMMAND "${TS_BUILDDIR}/clblast_test_xgemm")

add_test(NAME clblast_test_xsymm
  COMMAND "${TS_BUILDDIR}/clblast_test_xsymm")

add_test(NAME clblast_test_xhemm
  COMMAND "${TS_BUILDDIR}/clblast_test_xhemm")

add_test(NAME clblast_test_xsyrk
  COMMAND "${TS_BUILDDIR}/clblast_test_xsyrk")

add_test(NAME clblast_test_xherk
  COMMAND "${TS_BUILDDIR}/clblast_test_xherk")

add_test(NAME clblast_test_xsyr2k
  COMMAND "${TS_BUILDDIR}/clblast_test_xsyr2k")

add_test(NAME clblast_test_xher2k
  COMMAND "${TS_BUILDDIR}/clblast_test_xher2k")

add_test(NAME clblast_test_xtrmm
  COMMAND "${TS_BUILDDIR}/clblast_test_xtrmm")

add_test(NAME clblast_test_xtrsm
  COMMAND "${TS_BUILDDIR}/clblast_test_xtrsm")

add_test(NAME clblast_test_xhad
  COMMAND "${TS_BUILDDIR}/clblast_test_xhad")

add_test(NAME clblast_test_xomatcopy
  COMMAND "${TS_BUILDDIR}/clblast_test_xomatcopy")

add_test(NAME clblast_test_xim2col
  COMMAND "${TS_BUILDDIR}/clblast_test_xim2col")

add_test(NAME clblast_test_xaxpybatched
  COMMAND "${TS_BUILDDIR}/clblast_test_xaxpybatched")

add_test(NAME clblast_test_xgemmbatched
  COMMAND "${TS_BUILDDIR}/clblast_test_xgemmbatched")

add_test(NAME clblast_test_xgemmstridedbatched
  COMMAND "${TS_BUILDDIR}/clblast_test_xgemmstridedbatched")

add_test(NAME clblast_test_override_parameters
  COMMAND "${TS_BUILDDIR}/clblast_test_override_parameters")

add_test(NAME clblast_test_retrieve_parameters
  COMMAND "${TS_BUILDDIR}/clblast_test_retrieve_parameters")

add_test(NAME clblast_test_preprocessor
  COMMAND "${TS_BUILDDIR}/clblast_test_preprocessor")

set_tests_properties(clblast_test_xswap clblast_test_xscal clblast_test_xcopy
  clblast_test_xaxpy clblast_test_xdot clblast_test_xdotu clblast_test_xdotc
  clblast_test_xnrm2 clblast_test_xasum clblast_test_xamax clblast_test_xgemv
  clblast_test_xhemv clblast_test_xhpmv
  clblast_test_xsymv clblast_test_xspmv
  clblast_test_xtrsv clblast_test_xger
  clblast_test_xgeru clblast_test_xgerc clblast_test_xher clblast_test_xhpr
  clblast_test_xher2 clblast_test_xhpr2 clblast_test_xsyr clblast_test_xspr
  clblast_test_xsyr2 clblast_test_xspr2
  clblast_test_xsyrk clblast_test_xherk clblast_test_xsyr2k
  clblast_test_xher2k
  clblast_test_xomatcopy clblast_test_xim2col clblast_test_xaxpybatched
  clblast_test_retrieve_parameters clblast_test_preprocessor
  PROPERTIES
    LABELS "CLBlast")


set_tests_properties(clblast_test_xswap clblast_test_xscal clblast_test_xcopy
  clblast_test_xdot clblast_test_xdotu clblast_test_xdotc
  clblast_test_xnrm2 clblast_test_xasum clblast_test_xamax clblast_test_xgemv
  clblast_test_xhemv clblast_test_xhpmv
  clblast_test_xsymv clblast_test_xspmv
  clblast_test_xtrsv clblast_test_xger
  clblast_test_xgeru clblast_test_xgerc clblast_test_xher clblast_test_xhpr
  clblast_test_xher2 clblast_test_xhpr2 clblast_test_xsyr clblast_test_xspr
  clblast_test_xsyr2 clblast_test_xspr2
  clblast_test_xim2col clblast_test_xaxpybatched
  clblast_test_retrieve_parameters
  PROPERTIES
  LABELS "CLBlast hsa-native")

# These produce wrong some results with hsa and native compilation (but not
# with straight pthread, which is a bit weird):
#        131 - clblast_test_xaxpy (Failed)
#        164 - clblast_test_xsyrk (Failed)
#        165 - clblast_test_xherk (Failed)
#        166 - clblast_test_xsyr2k (Failed)
#        167 - clblast_test_xher2k (Failed)
#        171 - clblast_test_xomatcopy (Failed)

set_property(TEST
  clblast_test_xtbmv clblast_test_xtpmv
  clblast_test_xgbmv clblast_test_xhbmv clblast_test_xsbmv
  clblast_test_xtrmv clblast_test_xsymm clblast_test_xhemm
  clblast_test_xtrmm clblast_test_xtrsm
  clblast_test_xgemmbatched clblast_test_xgemmstridedbatched
  clblast_test_override_parameters
  APPEND PROPERTY LABELS "hsa-nat-slow")

# Skipped some of the slowest test cases:
# clblast_test_xtbmv clblast_test_xtpmv clblast_test_xgemm
# clblast_test_xgbmv clblast_test_xhbmv clblast_test_xsbmv
# clblast_test_xtrmv clblast_test_xsymm clblast_test_xhemm
# clblast_test_xtrmm clblast_test_xtrsm clblast_test_xhad
# clblast_test_xgemmbatched clblast_test_xgemmstridedbatched
# clblast_test_override_parameters
