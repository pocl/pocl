#=============================================================================
#   CMake build system files - add_test_pocl() etc. test wrappers
#
#   Copyright (c) 2014-2017 pocl developers
#                 2024 Pekka Jääskeläinen / Intel Finland Oy
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

include(CMakeParseArguments)

# This is a wrapper around add_test
# Solves several problems:
# 1) allows expected outputs (optionally sorted)
# 2) handles the exit status problem (test properties WILL_FAIL does not work if
#    the test exits with !0 exit status)
#
# If LLVM_FILECHECK is set to an existing FileCheck file, an additional test
# will be added that runs the test with the LLVM IR tester script using the
# loopvec method."

function(add_test_pocl)

  set(options SORT_OUTPUT)
  set(oneValueArgs EXPECTED_OUTPUT NAME WORKING_DIRECTORY LLVM_FILECHECK ONLY_FILECHECK ENVIRONMENT)
  set(multiValueArgs COMMAND WORKITEM_HANDLER)
  cmake_parse_arguments(POCL_TEST "${options}" "${oneValueArgs}"
                        "${multiValueArgs}" ${ARGN})
  if(POCL_TEST_WORKITEM_HANDLER)
    set(VARIANTS ${POCL_TEST_WORKITEM_HANDLER})
  else()
    set(VARIANTS "loopvec" "cbs")
  endif()
  list(LENGTH VARIANTS VARIANTS_COUNT)

  foreach(VARIANT ${VARIANTS})
    if(${VARIANTS_COUNT} GREATER 1)
      set(POCL_VARIANT_TEST_NAME ${POCL_TEST_NAME}_${VARIANT})
    else()
      set(POCL_VARIANT_TEST_NAME ${POCL_TEST_NAME})
    endif()
    unset(RUN_CMD)

    foreach(LOOPVAR ${POCL_TEST_COMMAND})
      if(NOT RUN_CMD)
        # Special command name expansion.
        if(${LOOPVAR} STREQUAL "poclcc")
          set(RUN_CMD "${CMAKE_BINARY_DIR}/bin/${LOOPVAR}")
        else()
          set(RUN_CMD "${CMAKE_CURRENT_BINARY_DIR}/${LOOPVAR}")
        endif()
      else()
        set(RUN_CMD "${RUN_CMD}####${LOOPVAR}")
      endif()
    endforeach()

    set(POCL_TEST_ARGLIST "NAME" "${POCL_VARIANT_TEST_NAME}")
    if(POCL_TEST_WORKING_DIRECTORY)
      list(APPEND POCL_TEST_ARGLIST "WORKING_DIRECTORY")
      list(APPEND POCL_TEST_ARGLIST "${POCL_TEST_WORKING_DIRECTORY}")
    endif()

    list(APPEND POCL_TEST_ARGLIST "COMMAND" "${CMAKE_COMMAND}" "-Dtest_cmd=${RUN_CMD}")
    if(INTEL_SDE_AVX512)
      list(APPEND POCL_TEST_ARGLIST "-DSDE=${INTEL_SDE_AVX512}")
    endif()

    if(POCL_TEST_EXPECTED_OUTPUT)
      if (NOT IS_ABSOLUTE "${POCL_TEST_EXPECTED_OUTPUT}")
        set(POCL_TEST_EXPECTED_OUTPUT "${CMAKE_CURRENT_SOURCE_DIR}/${POCL_TEST_EXPECTED_OUTPUT}")
      endif()
      list(APPEND POCL_TEST_ARGLIST
        "-Doutput_blessed=${POCL_TEST_EXPECTED_OUTPUT}")
    endif()
    if(POCL_TEST_SORT_OUTPUT)
      list(APPEND POCL_TEST_ARGLIST "-Dsort_output=1")
      endif()
    list(APPEND POCL_TEST_ARGLIST "-P" "${CMAKE_SOURCE_DIR}/cmake/run_test.cmake")

    if(NOT POCL_TEST_ONLY_FILECHECK)
      add_test(${POCL_TEST_ARGLIST})

      if(NOT ENABLE_ANYSAN)
        set_tests_properties("${POCL_VARIANT_TEST_NAME}" PROPERTIES
          PASS_REGULAR_EXPRESSION "OK"
          FAIL_REGULAR_EXPRESSION "FAIL")
      endif()
      set_tests_properties("${POCL_VARIANT_TEST_NAME}" PROPERTIES
        SKIP_RETURN_CODE 77)
      if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.16)
        set_tests_properties("${POCL_VARIANT_TEST_NAME}" PROPERTIES
          SKIP_REGULAR_EXPRESSION "SKIP")
      endif()

      set_tests_properties("${POCL_VARIANT_TEST_NAME}" PROPERTIES
        ENVIRONMENT POCL_WORK_GROUP_METHOD=${VARIANT})
    endif()

    if(ENABLE_LLVM_FILECHECKS AND POCL_TEST_LLVM_FILECHECK)
      set(RUN_CMD "${CMAKE_SOURCE_DIR}/tools/scripts/run-and-check-llvm-ir####${LLVM_FILECHECK_BIN}####${LLVM_DIS_BIN}####${CMAKE_CURRENT_SOURCE_DIR}/${POCL_TEST_LLVM_FILECHECK}####${RUN_CMD}")

      set(POCL_TEST_IR_CHECK_NAME "${POCL_VARIANT_TEST_NAME}_llvm-ir-checks")
      set(POCL_TEST_ARGLIST "NAME" ${POCL_TEST_IR_CHECK_NAME})
      if(POCL_TEST_WORKING_DIRECTORY)
        list(APPEND POCL_TEST_ARGLIST "WORKING_DIRECTORY")
        list(APPEND POCL_TEST_ARGLIST "${POCL_TEST_WORKING_DIRECTORY}")
      endif()
      list(APPEND POCL_TEST_ARGLIST "COMMAND" "${CMAKE_COMMAND}" "-Dtest_cmd=${RUN_CMD}")
      list(APPEND POCL_TEST_ARGLIST "-P" "${CMAKE_SOURCE_DIR}/cmake/run_test.cmake")

      add_test(${POCL_TEST_ARGLIST})

      set_tests_properties(${POCL_TEST_IR_CHECK_NAME} PROPERTIES
                          PASS_REGULAR_EXPRESSION "OK"
                          FAIL_REGULAR_EXPRESSION "FAIL"
                          ENVIRONMENT "POCL_WORK_GROUP_METHOD=${VARIANT};${POCL_TEST_ENVIRONMENT}"
                          LABELS "cpu"
                          DEPENDS "pocl_version_check")

    endif()

  endforeach()

endfunction()
