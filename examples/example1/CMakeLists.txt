#=============================================================================
#   CMake build system files
#
#   Copyright (c) 2014 pocl developers
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

#AM_CPPFLAGS = -I$(top_srcdir)/fix-include -I$(top_srcdir)/include -DSRCDIR='"$(abs_srcdir)"'
add_definitions("-DSRCDIR=\"${CMAKE_CURRENT_SOURCE_DIR}\"")

# example1_CFLAGS = @OPENCL_CFLAGS@
#set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -std=c99 ${OPENCL_CFLAGS}")
add_compile_options(${OPENCL_CFLAGS})

if (MSVC)
  set_source_files_properties( example1.c example1_exec.c PROPERTIES LANGUAGE CXX )
endif(MSVC)
add_executable("example1" example1.c example1_exec.c example1.cl)

# example1_LDADD = @OPENCL_LIBS@ ../../lib/poclu/libpoclu.la
target_link_libraries("example1" ${POCLU_LINK_OPTIONS})

add_test("examples/example1_dot_product" "example1")

set_tests_properties( "examples/example1_dot_product"
  PROPERTIES
    COST 3.0
    PASS_REGULAR_EXPRESSION "[(]0[.]000000, 0[.]000000, 0[.]000000, 0[.]000000[)] [.] [(]0[.]000000, 0[.]000000, 0[.]000000, 0[.]000000[)] = 0[.]000000
[(]1[.]000000, 1[.]000000, 1[.]000000, 1[.]000000[)] [.] [(]1[.]000000, 1[.]000000, 1[.]000000, 1[.]000000[)] = 4[.]000000
[(]2[.]000000, 2[.]000000, 2[.]000000, 2[.]000000[)] [.] [(]2[.]000000, 2[.]000000, 2[.]000000, 2[.]000000[)] = 16[.]000000
[(]3[.]000000, 3[.]000000, 3[.]000000, 3[.]000000[)] [.] [(]3[.]000000, 3[.]000000, 3[.]000000, 3[.]000000[)] = 36[.]000000
OK"
    PROCESSORS 1
    LABELS "internal;hsa;tce;cuda"
    DEPENDS "pocl_version_check")
