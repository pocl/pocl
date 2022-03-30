#=============================================================================
#   CMake build system files
#
#   Copyright (c) 2022 Michal Babej
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

if(ENABLE_SPIRV)

add_compile_options( -Wno-deprecated -Wno-deprecated-declarations)

add_definitions("-DSRCDIR=\"${CMAKE_CURRENT_SOURCE_DIR}\"")

add_executable("spirv_printf" printf.cc)

target_link_libraries("spirv_printf" ${POCLU_LINK_OPTIONS})

add_test_pocl(NAME "spirv/printf" COMMAND "spirv_printf")

set_tests_properties( "spirv/printf"
  PROPERTIES
    COST 4.0
    PROCESSORS 1
    PASS_REGULAR_EXPRESSION "OK"
    LABELS "internal;spirv"
    DEPENDS "pocl_version_check")

endif()
