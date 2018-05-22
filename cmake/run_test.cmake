# some argument checking:
# test_cmd is the command to run with all its arguments, separated by "####"
if( NOT test_cmd )
  message( FATAL_ERROR "Variable test_cmd not defined" )
endif()

# output_blessed contains the name of the file with expected output
if(output_blessed)
  message(STATUS "Expecting output: ${output_blessed}")
endif()

string(REPLACE "####" ";" test_cmd_separated "${test_cmd}")

execute_process(
  COMMAND ${test_cmd_separated}
  RESULT_VARIABLE test_not_successful
  OUTPUT_VARIABLE stdout
  ERROR_VARIABLE stderr
)

# the first run would fail, but still pre-compile the kernels
# for the 2nd run through SDE
if(SDE)
execute_process(
  COMMAND "${SDE}" -skx -- ${test_cmd_separated}
  RESULT_VARIABLE test_not_successful
  OUTPUT_VARIABLE stdout
  ERROR_VARIABLE stderr
)
endif()


if( test_not_successful )
  message( SEND_ERROR "FAIL: Test exited with nonzero code: ${test_cmd_separated}\nSTDOUT:\n${stdout}\nSTDERR:\n${stderr}" )
else()
  message("${stdout}")
  message("${stderr}")
endif()

if(output_blessed)

  string(RANDOM RAND_STR)
  set(RANDOM_FILE "/tmp/cmake_testrun_${RAND_STR}")
  file(WRITE "${RANDOM_FILE}" "${stdout}")

  if( sort_output )
    message(STATUS "SORTING FILE")
    file(STRINGS "${RANDOM_FILE}" output_string_list)
    list(SORT output_string_list)
    # for some reason sorting doesn't work when list contains newlines,
    # have to add them after the sort
    file(REMOVE "${RANDOM_FILE}")
    string(REPLACE ";" "\n" OUTPUT "${output_string_list}")
    set(RANDOM_FILE "${RANDOM_FILE}_sorted")
    file(WRITE "${RANDOM_FILE}" "${OUTPUT}\n")
  endif()

  message(STATUS "Comparing output..")
  execute_process(
    COMMAND ${CMAKE_COMMAND} -E compare_files "${output_blessed}" "${RANDOM_FILE}"
    RESULT_VARIABLE test_not_successful
    )

  if( test_not_successful )
    message(SEND_ERROR "FAIL: Test output does not match the expected output; output stored in ${RANDOM_FILE}" )
  else()
    file(REMOVE "${RANDOM_FILE}")
  endif()

endif()

if ((NOT "${stdout}${stderr}" MATCHES "OK")
    AND
    (NOT "${stdout}${stderr}" MATCHES "FAIL"))

  message(STATUS "OK")

endif()
