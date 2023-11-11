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

add_executable("convol1" convol1.cpp)
target_link_libraries("convol1" ${POCLU_LINK_OPTIONS})
add_test("examples/convol1" "convol1")

add_executable("convol2" convol2.cpp)
target_link_libraries("convol2" ${POCLU_LINK_OPTIONS})
add_test("examples/convol2" "convol2")

set(CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set_property(TARGET convol1 PROPERTY CXX_STANDARD 11)
set_property(TARGET convol2 PROPERTY CXX_STANDARD 11)

set(PROPS)
#if(NOT ENABLE_ANYSAN)
#  set(PROPS
#    PASS_REGULAR_EXPRESSION "OK")
#endif()

set_tests_properties( "examples/convol1"
  PROPERTIES
    COST 30.0
    ${PROPS}
    PROCESSORS 1
    ENVIRONMENT "POCL_MAX_WORK_GROUP_SIZE=64;POCL_TTASIM0_PARAMETERS=${CMAKE_SOURCE_DIR}/tools/data/test_machine_fp16_LE_full.adf"
    LABELS "slow"
    DEPENDS "pocl_version_check")


# "examples/convol2"
