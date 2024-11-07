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

#AM_CPPFLAGS = -I$(top_srcdir)/fix-include -I$(top_srcdir)/include -DSRCDIR='"$(abs_srcdir)"'
add_definitions("-DSRCDIR=\"${CMAKE_CURRENT_SOURCE_DIR}\"")

# EinsteinToolkit_CFLAGS = @OPENCL_CFLAGS@
#set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -std=c99 ${OPENCL_CFLAGS}")
add_compile_options(${OPENCL_CFLAGS} -std=c99)

add_executable("EinsteinToolkit" EinsteinToolkit.c ML_BSSN_CL_RHS1.cl ML_BSSN_CL_RHS2.cl)

# EinsteinToolkit_LDADD = @OPENCL_LIBS@ ../../lib/poclu/libpoclu.la
target_link_libraries("EinsteinToolkit" ${POCLU_LINK_OPTIONS})

add_test(NAME "EinsteinToolkit" COMMAND "EinsteinToolkit")

set_tests_properties( "EinsteinToolkit"
    PROPERTIES
    COST 15.0
    LABELS "internal;EinsteinToolkit"
    DEPENDS "pocl_version_check")

add_test(NAME "EinsteinToolkit_SubDev" COMMAND "EinsteinToolkit" s)

set_tests_properties( "EinsteinToolkit_SubDev"
    PROPERTIES
    COST 15.0
    LABELS "EinsteinToolkit"
    ENVIRONMENT "POCL_AFFINITY=1"
    SKIP_RETURN_CODE 77
    DEPENDS "pocl_version_check")

if(ENABLE_HOST_CPU_DEVICES AND RISCV)
  set_tests_properties("EinsteinToolkit" PROPERTIES DISABLED 1)
endif()
