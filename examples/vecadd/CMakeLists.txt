#=============================================================================
#   CMake build system files
#
#   Copyright (c) 2018 pocl developers
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
  set_source_files_properties( vecadd.c vecadd_exec.c PROPERTIES LANGUAGE CXX )
endif(MSVC)

#message(STATUS "ld_library_path=$ENV{LD_LIBRARY_PATH}")

add_executable("vecadd" vecadd.c vecadd_exec.c vecadd.cl)

TARGET_LINK_DIRECTORIES(vecadd PUBLIC $ENV{LD_LIBRARY_PATH})
target_link_libraries("vecadd" ${POCLU_LINK_OPTIONS})

add_test(NAME "examples/vecadd" COMMAND "vecadd")

add_test(NAME "examples/vecadd_large_grid" COMMAND "vecadd" "128000" "128" "10000" "100" "1" "1")

set(PROPS)
if(NOT ENABLE_ANYSAN)
  set(PROPS
    PASS_REGULAR_EXPRESSION "OK")
endif()

set_tests_properties(
  "examples/vecadd"
  "examples/vecadd_large_grid"
  PROPERTIES
    COST 3.0
    ${PROPS}
    PROCESSORS 1
    LABELS "internal;vulkan"
    DEPENDS "pocl_version_check")
