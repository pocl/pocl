# some argument checking:
# test_cmd is the command to run with all its arguments, separated by "####"
if( NOT test_cmd )
  message( FATAL_ERROR "Variable test_cmd not defined" )
endif()
# output_blessed contains the name of the "blessed" output file
if( NOT output_blessed )
  message( FATAL_ERROR "Variable output_blessed not defined" )
else()
  message(STATUS "Expecting output: ${output_blessed}")
endif()

message(STATUS "POCL_DEVICES: $ENV{POCL_DEVICES}")
message(STATUS "POCL_WORK_GROUP_METHOD: $ENV{POCL_WORK_GROUP_METHOD}")

string(REPLACE "####" ";" test_cmd_separated "${test_cmd}")

string(RANDOM RAND_STR)
# TODO properly handle tmpdir
set(RANDOM_FILE "/tmp/cmake_testrun_${RAND_STR}")

execute_process(
  COMMAND ${test_cmd_separated}
  RESULT_VARIABLE test_not_successful
  OUTPUT_FILE "${RANDOM_FILE}"
  ERROR_VARIABLE stderr
)

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

if( test_not_successful )
  message( SEND_ERROR "Test exited with nonzero code: ${test_cmd_separated}\nSTDERR:\n${stderr}" )
endif()

#~ if( sort_output )
  #~ execute_process(
    #~ COMMAND "sort"
#~ endif()

execute_process(
  COMMAND ${CMAKE_COMMAND} -E compare_files "${output_blessed}" "${RANDOM_FILE}"
  RESULT_VARIABLE test_not_successful
)

if( test_not_successful )
  message( SEND_ERROR "Test output does not match the expected output; output stored in ${RANDOM_FILE}" )
else()
  file(REMOVE "${RANDOM_FILE}")
endif()
