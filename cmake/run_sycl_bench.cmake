# Produces HeCBench-style CSV from syncl-bench style benchmark output
#
# How to use: cmake -DOUTPUT_FILE=/path/to.csv -P <this-file.cmake> -- test_binary [test_binary-arguments...]
#
if( NOT CMAKE_ARGV4 STREQUAL "--" )
  message(FATAL_ERROR "Usage: cmake -DOUTPUT_FILE=/path/to.csv -P <this-file.cmake> -- test_binary [test_binary arguments ...]" )
endif()

if(NOT OUTPUT_FILE)
  message(FATAL_ERROR "Output filename not given")
endif()

set(TEST_BINARY ${CMAKE_ARGV5})
if(NOT EXISTS ${TEST_BINARY} )
  message(FATAL_ERROR "Not a file: ${TEST_BINARY}")
else()
  message(STATUS "Using test binary: ${TEST_BINARY}")
endif()

set(ARG_NUM 6)
math(EXPR ARGC_COUNT "${CMAKE_ARGC}")

while(ARG_NUM LESS ARGC_COUNT)
  set(CURRENT_ARG "${CMAKE_ARGV${ARG_NUM}}")
  list(APPEND TEST_BINARY "${CURRENT_ARG}")
  math(EXPR ARG_NUM "${ARG_NUM}+1")
endwhile()

message(STATUS "Using test binary with args: ${TEST_BINARY}")

execute_process(
  COMMAND ${TEST_BINARY}
  RESULT_VARIABLE test_not_successful
  OUTPUT_VARIABLE bench_stdout
  ERROR_VARIABLE bench_stderr
)

if( test_not_successful )
  message( FATAL_ERROR "FAIL: Benchmark exited with nonzero code (${test_not_successful}): ${TEST_BINARY}\nSTDOUT:\n${bench_stdout}\nSTDERR:\n${bench_stderr}" )
else()

  string(REGEX MATCHALL "Results for ([^*]+)[*]"                        NAME_LIST "${bench_stdout}")
  string(REGEX REPLACE  "Results for ([^*]+)[*]"                      "\\1" NAME_L "${NAME_LIST}")

  string(REGEX MATCHALL "run-time-min: ([0-9]+\\.[0-9]+) "           MIN_TIME_LIST "${bench_stdout}")
  string(REGEX REPLACE  "run-time-min: ([0-9]+\\.[0-9]+) "         "\\1" MIN_TIME_L "${MIN_TIME_LIST}")

  string(REGEX MATCHALL "run-time-mean: ([0-9]+\\.[0-9]+) "          MEAN_TIME_LIST "${bench_stdout}")
  string(REGEX REPLACE  "run-time-mean: ([0-9]+\\.[0-9]+) "        "\\1" MEAN_TIME_L "${MEAN_TIME_LIST}")

  string(REGEX MATCHALL "run-time-median: ([0-9]+\\.[0-9]+) "        MEDIAN_TIME_LIST "${bench_stdout}")
  string(REGEX REPLACE  "run-time-median: ([0-9]+\\.[0-9]+) "      "\\1" MEDIAN_TIME_L "${MEDIAN_TIME_LIST}")

  string(REGEX MATCHALL "run-time-stddev: ([0-9]+\\.[0-9]+) "        STDDEV_TIME_LIST "${bench_stdout}")
  string(REGEX REPLACE  "run-time-stddev: ([0-9]+\\.[0-9]+) "      "\\1" STDDEV_L "${STDDEV_TIME_LIST}")

#  string(REGEX MATCHALL "run-time-throughput: ([0-9]+\\.[0-9]+) "    THROUP_TIME_LIST "${bench_stdout}")
#  string(REGEX REPLACE "run-time-throughput: ([0-9]+\\.[0-9]+) "  "\\1" THROUP_L "${THROUP_TIME_LIST}")

  foreach(NAME MIN_TIME MEAN_TIME MEDIAN_TIME STDDEV  IN ZIP_LISTS NAME_L MIN_TIME_L MEAN_TIME_L MEDIAN_TIME_L STDDEV_L)
    if(MEAN_TIME)
      math(EXPR VARIANCE "${STDDEV} / ${MEAN_TIME}" DECIMAL)
    else()
      set(VARIANCE 0)
    endif()
    file(APPEND "${OUTPUT_FILE}" "${NAME},${MIN_TIME},${MEAN_TIME},${STDDEV},${VARIANCE}")
  endforeach()

endif()

