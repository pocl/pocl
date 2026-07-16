if(NOT DEFINED test_cmd OR NOT DEFINED cache_dir)
  message(FATAL_ERROR "test_cmd and cache_dir are required")
endif()

file(REMOVE_RECURSE "${cache_dir}")
file(MAKE_DIRECTORY "${cache_dir}")

foreach(phase cold warm)
  execute_process(
    COMMAND "${CMAKE_COMMAND}" -E env "POCL_CACHE_DIR=${cache_dir}" "${test_cmd}"
    RESULT_VARIABLE result
    OUTPUT_VARIABLE stdout
    ERROR_VARIABLE stderr)
  if(result OR NOT "${stdout}${stderr}" MATCHES "OK")
    message(FATAL_ERROR "SVML ${phase}-cache run failed:\n${stdout}\n${stderr}")
  endif()
  string(REGEX MATCHALL "activating lazy SVML archive provider" activations
         "${stdout}\n${stderr}")
  list(LENGTH activations activation_count)
  if(NOT activation_count EQUAL 1)
    message(FATAL_ERROR "SVML ${phase}-cache run activated ${activation_count} times")
  endif()
endforeach()
