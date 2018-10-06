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

add_compile_options(${OPENCL_CFLAGS})

if (MSVC)
  set_source_files_properties( matrix1.c PROPERTIES LANGUAGE CXX )
endif(MSVC)
add_executable("matrix1" matrix1.c matrix1.cl)

target_link_libraries("matrix1" ${POCLU_LINK_OPTIONS})

add_test("examples/matrix1" "matrix1" 512)

add_test("examples/matrix1_local" "matrix1" 512 -l)

if (ENABLE_SPIR)
  add_test("examples/matrix1_spir" "matrix1" 256 -s)

  add_test("examples/matrix1_spir_local" "matrix1" 256 -s -l)

  list(APPEND SPIREX "examples/matrix1_spir" "examples/matrix1_spir_local")
endif()

set_tests_properties( "examples/matrix1" "examples/matrix1_local" ${SPIREX}
  PROPERTIES
    COST 8.0
    PASS_REGULAR_EXPRESSION "OK"
    PROCESSORS 1
    LABELS "matrix"
    DEPENDS "pocl_version_check")
