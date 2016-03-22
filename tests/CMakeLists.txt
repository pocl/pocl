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

#function(add_test_custom RUN_CMD TEST_NAME RESULT_FILE)
#  foreach(LOOPVAR ${ARGN})
#    set(RUN_CMD "${RUN_CMD}####${LOOPVAR}")
#  endforeach()
#endfunction()

include(CMakeParseArguments)

# This is a wrapper around add_test
# Solves several problems:
# 1) allows expected outputs (optionally sorted)
# 2) handles the exit status problem (test properties WILL_FAIL does not work if
#    the test exits with !0 exit status)

function(add_test_pocl)

  set(options SORT_OUTPUT)
  set(oneValueArgs EXPECTED_OUTPUT NAME WORKING_DIRECTORY)
  set(multiValueArgs COMMAND)
  cmake_parse_arguments(POCL_TEST "${options}" "${oneValueArgs}"
                        "${multiValueArgs}" ${ARGN})

  #message(STATUS "POCL_TEST_NAME: ${POCL_TEST_NAME}")
  #message(STATUS "POCL_TEST_COMMAND: ${POCL_TEST_COMMAND}")

  unset(RUN_CMD)
  foreach(LOOPVAR ${POCL_TEST_COMMAND})
    if(NOT RUN_CMD)
      set(RUN_CMD "${CMAKE_CURRENT_BINARY_DIR}/${LOOPVAR}")
    else()
      set(RUN_CMD "${RUN_CMD}####${LOOPVAR}")
    endif()
  endforeach()

  set(POCL_TEST_ARGLIST "NAME" "${POCL_TEST_NAME}")
  if(POCL_TEST_WORKING_DIRECTORY)
    list(APPEND POCL_TEST_ARGLIST "WORKING_DIRECTORY")
    list(APPEND POCL_TEST_ARGLIST "${POCL_TEST_WORKING_DIRECTORY}")
  endif()

  list(APPEND POCL_TEST_ARGLIST "COMMAND" "${CMAKE_COMMAND}" "-Dtest_cmd=${RUN_CMD}")
  if(POCL_TEST_EXPECTED_OUTPUT)
    list(APPEND POCL_TEST_ARGLIST
      "-Doutput_blessed=${CMAKE_CURRENT_SOURCE_DIR}/${POCL_TEST_EXPECTED_OUTPUT}")
  endif()
  if(POCL_TEST_SORT_OUTPUT)
    list(APPEND POCL_TEST_ARGLIST "-Dsort_output=1")
    endif()
  list(APPEND POCL_TEST_ARGLIST "-P" "${CMAKE_SOURCE_DIR}/cmake/run_test.cmake")

  add_test(${POCL_TEST_ARGLIST} )
  set_tests_properties("${POCL_TEST_NAME}" PROPERTIES
                       PASS_REGULAR_EXPRESSION "OK"
                       FAIL_REGULAR_EXPRESSION "FAIL")

endfunction()


add_test("pocl_version_check" "runtime/test_version")
set_tests_properties("pocl_version_check"
  PROPERTIES
  ENVIRONMENT "POCL_DEVICES=basic"
  PASS_REGULAR_EXPRESSION "basic"
  LABELS "internal")

#######################################################################

add_subdirectory("kernel")
add_subdirectory("regression")
add_subdirectory("runtime")
add_subdirectory("workgroup")
if(ENABLE_TCE)
  add_subdirectory("tce")
endif()

