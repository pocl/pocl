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

add_definitions("-DSRCDIR=\"${CMAKE_CURRENT_SOURCE_DIR}\"")

add_compile_options(${OPENCL_CFLAGS})

add_executable("trig" trig.c trig.cl trig_exec.c)
target_link_libraries("trig" ${POCLU_LINK_OPTIONS})
add_symlink_to_built_opencl_dynlib("trig")

add_test(NAME "examples/trig" COMMAND "trig")

set(PROPS)
if(NOT ENABLE_ANYSAN)
  set(PROPS
    PASS_REGULAR_EXPRESSION "OK")
endif()

set_tests_properties(
  "examples/trig"
  PROPERTIES
    COST 3.0
    ${PROPS}
    PROCESSORS 1
    LABELS "cuda;hsa-native;vulkan;level0"
    DEPENDS "pocl_version_check")

if(ENABLE_HOST_CPU_DEVICES AND RISCV)
  set_tests_properties("examples/trig" PROPERTIES DISABLED 1)
endif()
