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

add_definitions("-DSRCDIR=\"${CMAKE_CURRENT_SOURCE_DIR}\"")

# EinsteinToolkit_CFLAGS = @OPENCL_CFLAGS@
add_compile_options(${OPENCL_CFLAGS})

add_executable("EinsteinToolkit" EinsteinToolkit.c ML_BSSN_CL_RHS1.cl ML_BSSN_CL_RHS2.cl)
add_symlink_to_built_opencl_dynlib("EinsteinToolkit")

target_link_libraries("EinsteinToolkit" ${POCLU_LINK_OPTIONS})

add_test(NAME "EinsteinToolkit" COMMAND "EinsteinToolkit")

set_tests_properties( "EinsteinToolkit"
    PROPERTIES
    COST 15.0
    RESOURCE_LOCK "einstein" # these tests use a lot of memory; allow only 1 instance at a time
    LABELS "EinsteinToolkit"
    SKIP_RETURN_CODE 77)

add_test(NAME "EinsteinToolkit_SubDev" COMMAND "EinsteinToolkit" s)

set_tests_properties( "EinsteinToolkit_SubDev"
    PROPERTIES
    COST 15.0
    RESOURCE_LOCK "einstein"
    LABELS "EinsteinToolkit"
    SKIP_RETURN_CODE 77)

if(RISCV)
  set_property(TEST
    "EinsteinToolkit" "EinsteinToolkit_SubDev"
    APPEND PROPERTY LABELS "cpu_fail")
endif()
