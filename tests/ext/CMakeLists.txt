#=============================================================================
#   CMake build system files
#
#   Copyright (c) 2022 Pekka Jääskeläinen
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

add_executable("relaxed_printf_address_space" relaxed_printf_address_space.cc)

target_link_libraries("relaxed_printf_address_space" ${POCLU_LINK_OPTIONS})

add_test_pocl(NAME "ext/relaxed_printf_address_space"
              EXPECTED_OUTPUT "relaxed_printf_address_space.stdout"
              COMMAND "relaxed_printf_address_space")

set_tests_properties( "ext/relaxed_printf_address_space"
  PROPERTIES
    COST 4.0
    PROCESSORS 1
    LABELS "internal;ext"
    DEPENDS "pocl_version_check")

