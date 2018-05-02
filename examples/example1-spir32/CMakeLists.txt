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

if(POCL_DEVICE_ADDRESS_BITS EQUAL "32")

add_definitions("-DSRCDIR=\"${CMAKE_CURRENT_SOURCE_DIR}\"")

add_compile_options(${OPENCL_CFLAGS})

if (MSVC)
  set_source_files_properties( example1.c example1_exec.c PROPERTIES LANGUAGE CXX )
endif(MSVC)
add_executable("example1-spir32" example1.c example1_exec.c example1.cl example1.spir)

target_link_libraries("example1-spir32" ${POCLU_LINK_OPTIONS})

add_test_pocl(NAME "spec_tests/example1_dot_product_spir32"
              EXPECTED_OUTPUT "example1-spir.stdout"
              COMMAND "example1-spir32")

set_tests_properties( "spec_tests/example1_dot_product_spir32"
  PROPERTIES
    COST 3.0
    PROCESSORS 1
    LABELS "internal;spir;cuda"
    DEPENDS "pocl_version_check")

# fails on older LLVMs because SPIR bitcode was generated with LLVM 4.0
if(LLVM_OLDER_THAN_3_9)
  set_tests_properties( "spec_tests/example1_dot_product_spir32"
  PROPERTIES
    WILL_FAIL 1)
endif()

endif()
