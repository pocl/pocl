#=============================================================================
#   CMake build system files
#
#   Copyright (c) 2016 pocl developers
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

add_executable("fp16_host" host.cpp)

target_link_libraries("fp16_host" ${POCLU_LINK_OPTIONS})

add_compile_options( -Wno-deprecated -Wno-deprecated-declarations)

add_definitions("-DSRCDIR=\"${CMAKE_CURRENT_SOURCE_DIR}\"")

add_test_pocl(NAME "tce/fp16/repl" COMMAND "fp16_host" EXPECTED_OUTPUT "expected_out.txt")

#AT_SETUP([Half-precision floats on ttasim (repl)])
set_tests_properties( "tce/fp16/repl"
  PROPERTIES
    COST 40.0
    PROCESSORS 1
    LABELS "tce;tta;ttasim;half"
    ENVIRONMENT "POCL_DEVICES=ttasim;POCL_TTASIM0_PARAMETERS=${CMAKE_SOURCE_DIR}/tools/data/test_machine_fp16.adf;POCL_WORK_GROUP_METHOD=repl"
    DEPENDS "pocl_version_check")

add_test_pocl(NAME "tce/fp16/loopvec" COMMAND "fp16_host" EXPECTED_OUTPUT "expected_out.txt")


set_tests_properties( "tce/fp16/loopvec"
  PROPERTIES
    COST 40.0
    PROCESSORS 1
    LABELS "tce;tta;ttasim;half"
    ENVIRONMENT "POCL_DEVICES=ttasim;POCL_TTASIM0_PARAMETERS=${CMAKE_SOURCE_DIR}/tools/data/test_machine_fp16.adf;POCL_WORK_GROUP_METHOD=loopvec"
    DEPENDS "pocl_version_check")

if(LLVM_OLDER_THAN_6_0)
  set_property(TEST "tce/fp16/loopvec"
  APPEND PROPERTY WILL_FAIL 1)
endif()
