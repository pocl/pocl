#=============================================================================
#   CMake build system files
#
#   Copyright (c) 2018 Michal Babej / Tampere University of Technology
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

add_compile_options(${OPENCL_CFLAGS} -std=c99)

add_executable("matrix1" matrix1.c matrix1.cl)
target_link_libraries("matrix1" ${POCLU_LINK_OPTIONS})
add_symlink_to_built_opencl_dynlib(matrix1)

add_test(NAME "examples/matrix1" COMMAND "matrix1" 512)

add_test(NAME "examples/matrix1_local" COMMAND "matrix1" 512 -l)

set_tests_properties( "examples/matrix1" "examples/matrix1_local" ${SPIREX}
  PROPERTIES
    COST 8.0
    PASS_REGULAR_EXPRESSION "OK"
    PROCESSORS 1
    SKIP_REGULAR_EXPRESSION "SKIPPED"
    LABELS "matrix;level0"
    DEPENDS "pocl_version_check")

# devices which don't support SPIR
set_property(TEST "examples/matrix1" "examples/matrix1_local"
  APPEND PROPERTY LABELS "vulkan")

add_test(NAME "examples/matrix1_poclbin"
         COMMAND ${CMAKE_COMMAND}
            "-DCMD1=${CMAKE_BINARY_DIR}/bin/poclcc####-o####${CMAKE_BINARY_DIR}/matrix1_poclbin.bin####${CMAKE_CURRENT_SOURCE_DIR}/matrix1.cl"
            "-DCMD2=${CMAKE_BINARY_DIR}/examples/matrix1/matrix1####256####-b####${CMAKE_BINARY_DIR}/matrix1_poclbin.bin"
            "-DCMD3=${CMAKE_COMMAND}####-E####remove####-f####${CMAKE_BINARY_DIR}/matrix1_poclbin.bin"
            -P ${CMAKE_SOURCE_DIR}/cmake/multi_exec_test.cmake)

set_tests_properties( "examples/matrix1_poclbin"
  PROPERTIES
    COST 8.0
    PASS_REGULAR_EXPRESSION "OK"
    SKIP_REGULAR_EXPRESSION "SKIPPED"
    PROCESSORS 1
    LABELS "poclbin"
    DEPENDS "pocl_version_check"
    ENVIRONMENT "POCL_KERNEL_CACHE=0")
