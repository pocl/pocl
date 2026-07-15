if(NOT DEFINED test_cmd)
  message(FATAL_ERROR "test_cmd is required")
endif()

execute_process(
  COMMAND "${test_cmd}"
  RESULT_VARIABLE result
  OUTPUT_VARIABLE stdout
  ERROR_VARIABLE stderr)

if(result EQUAL 0)
  message(FATAL_ERROR "SVML fault-injection test unexpectedly succeeded")
endif()

set(output "${stdout}\n${stderr}")
if(NOT output MATCHES "injected SVML provider failure \\(POCL_FAULT_INJECT_JIT_SVML\\)")
  message(FATAL_ERROR "SVML fault-injection error was not reported:\n${output}")
endif()
