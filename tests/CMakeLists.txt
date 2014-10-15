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

function(add_test_custom RUN_CMD TEST_NAME RESULT_FILE)
  foreach(LOOPVAR ${ARGN})
    set(RUN_CMD "${RUN_CMD}####${LOOPVAR}")
  endforeach()

  add_test( "${TEST_NAME}"
    "${CMAKE_COMMAND}"
    -Dtest_cmd=${RUN_CMD}
    -Doutput_blessed=${CMAKE_CURRENT_SOURCE_DIR}/${RESULT_FILE}
    -P "${CMAKE_SOURCE_DIR}/cmake/run_test.cmake"
  )
endfunction()

add_test("pocl_version_check" "runtime/test_version")
set_tests_properties("pocl_version_check"
  PROPERTIES
  ENVIRONMENT "POCL_DEVICES=basic"
  PASS_REGULAR_EXPRESSION "basic")

#######################################################################

add_subdirectory("kernel")
if(HAVE_OPENCL_HPP)
  add_subdirectory("regression")
else()
  message(STATUS "cl.hpp unavailable, skipping regression tests")
endif()
add_subdirectory("runtime")
add_subdirectory("workgroup")
if(ENABLE_TCE)
  #add_subdirectory("tce")
endif()
if(ENABLE_SPU)
  #add_subdirectory("cell")
endif()

