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

add_compile_options(${OPENCL_CFLAGS})

add_executable("example2" example2.c example2.cl)
target_link_libraries("example2" ${POCLU_LINK_OPTIONS})
add_symlink_to_built_opencl_dynlib("example2")

add_test(NAME "examples/example2" COMMAND "example2")


set(PROPS)
if(NOT ENABLE_ANYSAN)
  set(PROPS
    PASS_REGULAR_EXPRESSION "OK\n")
endif()

set_tests_properties( "examples/example2"
  PROPERTIES
    COST 3.0
    ${PROPS}
    PROCESSORS 1
    LABELS "cuda;level0;tce;hsa;hsa-native;vulkan"
    DEPENDS "pocl_version_check")


add_test(NAME "examples/example2_poclbin"
         COMMAND ${CMAKE_COMMAND}
            "-DCMD1=${CMAKE_BINARY_DIR}/bin/poclcc####-o####${CMAKE_BINARY_DIR}/example2.bin####${CMAKE_CURRENT_SOURCE_DIR}/example2.cl"
            "-DCMD2=${CMAKE_BINARY_DIR}/examples/example2/example2####b####${CMAKE_BINARY_DIR}/example2.bin"
            "-DCMD3=${CMAKE_COMMAND}####-E####remove####-f####${CMAKE_BINARY_DIR}/example2.bin"
            -P ${CMAKE_SOURCE_DIR}/cmake/multi_exec_test.cmake)

set_tests_properties( "examples/example2_poclbin"
  PROPERTIES
    COST 3.0
    ${PROPS}
    PROCESSORS 1
    ENVIRONMENT "POCL_KERNEL_CACHE=0"
    LABELS "poclbin;level0"
    DEPENDS "pocl_version_check")
