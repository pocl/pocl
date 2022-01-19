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

if (MSVC)
  set_source_files_properties( example2.c PROPERTIES LANGUAGE CXX )
endif(MSVC)
add_executable("example2" example2.c example2.cl)

target_link_libraries("example2" ${POCLU_LINK_OPTIONS})

add_test(NAME "examples/example2" COMMAND "example2")

if (FALSE) # ENABLE_SPIR - disabled for now because missing the SPIR binaries.
  add_test(NAME "examples/example2_spir"
           COMMAND ${CMAKE_COMMAND}
           "-DCMD1=${CLANG}####${SPIR_COMMON_FLAGS}####-target####spir-unknown-unknown####-o####${CMAKE_BINARY_DIR}/example2.spir32####-c####${CMAKE_CURRENT_SOURCE_DIR}/example2.cl"
           "-DCMD2=${CLANG}####${SPIR_COMMON_FLAGS}####-target####spir64-unknown-unknown####-o####${CMAKE_BINARY_DIR}/example2.spir64####-c####${CMAKE_CURRENT_SOURCE_DIR}/example2.cl"
           "-DCMD3=${CMAKE_BINARY_DIR}/examples/example2/example2####s"
           "-DCMD4=${CMAKE_COMMAND}####-E####remove####-f####${CMAKE_BINARY_DIR}/example2.spir32####${CMAKE_BINARY_DIR}/example2.spir64"
            -P ${CMAKE_SOURCE_DIR}/cmake/multi_exec_test.cmake)
  set(SPIREX "examples/example2_spir")
endif()


set(PROPS)
if(NOT ENABLE_ANYSAN)
  set(PROPS
    PASS_REGULAR_EXPRESSION "OK\n")
endif()

set_tests_properties( "examples/example2" ${SPIREX}
  PROPERTIES
    COST 3.0
    ${PROPS}
    PROCESSORS 1
    ENVIRONMENT "POCL_TTASIM0_PARAMETERS=${CMAKE_SOURCE_DIR}/tools/data/test_machine_LE.adf"
    LABELS "internal;cuda"
    DEPENDS "pocl_version_check")

# devices which don't support SPIR
set_property(TEST "examples/example2"
  APPEND PROPERTY LABELS "tce;hsa;hsa-native;vulkan")

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
    LABELS "poclbin;internal"
    DEPENDS "pocl_version_check")
