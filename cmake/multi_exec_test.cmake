macro(execute_command_with_args CMD_WITH_ARGS)

    string(REPLACE "####" ";" CMD_SEPARATED "${CMD_WITH_ARGS}")

    execute_process(COMMAND ${CMD_SEPARATED}
      RESULT_VARIABLE CMD_RESULT
      OUTPUT_VARIABLE stdout
      ERROR_VARIABLE  stderr
    )

    if( CMD_RESULT )
      message( SEND_ERROR "FAIL: Command exited with nonzero code (${CMD_RESULT}): ${CMD}\nSTDOUT:\n${stdout}\nSTDERR:\n${stderr}" )
    else()
      message("${stdout}")
      message("${stderr}")
    endif()

endmacro()

if(CMD1)
  execute_command_with_args(${CMD1})
endif()

if(CMD2)
  execute_command_with_args(${CMD2})
endif()

if(CMD3)
  execute_command_with_args(${CMD3})
endif()

if(CMD4)
  execute_command_with_args(${CMD4})
endif()

if(CMD5)
  execute_command_with_args(${CMD5})
endif()
