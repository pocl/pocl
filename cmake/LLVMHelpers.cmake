
macro(run_llvm_config VARIABLE_NAME)
  execute_process(
    COMMAND "${LLVM_CONFIG_BIN}" ${ARGN}
    OUTPUT_VARIABLE LLVM_CONFIG_VALUE
    RESULT_VARIABLE LLVM_CONFIG_RETVAL
    OUTPUT_STRIP_TRAILING_WHITESPACE
  )
  if(LLVM_CONFIG_RETVAL)
    message(FATAL_ERROR "Error running llvm-config with arguments: ${ARGN}")
  else()
    set(${VARIABLE_NAME} ${LLVM_CONFIG_VALUE} CACHE STRING "llvm-config's ${VARIABLE_NAME} value")
    message(STATUS "llvm-config's ${VARIABLE_NAME} is: ${${VARIABLE_NAME}}")
  endif()
endmacro(run_llvm_config)

macro(remove_prefix_from_filepath PATH_VAR)
  if(DEFINED ${PATH_VAR} AND ${PATH_VAR})
  string(LENGTH "${CMAKE_SYSROOT}" CMAKE_SYSROOT_LENGTH)
  string(SUBSTRING "${${PATH_VAR}}" 0 ${CMAKE_SYSROOT_LENGTH} PATH_VAR_START)
  if(PATH_VAR_START STREQUAL CMAKE_SYSROOT)
    string(SUBSTRING "${${PATH_VAR}}" ${CMAKE_SYSROOT_LENGTH} -1 PATH_VAR_END)
    set(${PATH_VAR} "${PATH_VAR_END}")
    message(STATUS "set ${PATH_VAR} to ${${PATH_VAR}}")
  else()
    message(STATUS "########## ${${PATH_VAR}} could not match ${CMAKE_SYSROOT} to ${PATH_VAR_START}")
  endif()
  endif()
endmacro()

macro(find_llvm_program_or_die OUTPUT_VAR PROG_NAME LOCATION DOCSTRING)
  find_program(${OUTPUT_VAR}
    NAMES "${PROG_NAME}${LLVM_BINARY_SUFFIX}${CMAKE_EXECUTABLE_SUFFIX}"
          "${PROG_NAME}-${LLVM_VERSION_MAJOR}${CMAKE_EXECUTABLE_SUFFIX}"
    # At least the LLVM v21 .deb doesn't have a clang++-21 under
    # /usr/lib/llvm-21/bin, but only 'clang++', 'clang' and
    # clang-21 symlink. Thus when looking only in the install prefix,
    # we don't find the clang++-21 symlink that is only in /usr/bin.
    # So, look also for the non-suffixed ones since we are searching
    # in the install/config dir.
    "${PROG_NAME}${CMAKE_EXECUTABLE_SUFFIX}"
    HINTS ${LOCATION}
    DOC "${DOCSTRING}"
    NO_DEFAULT_PATH
    NO_CMAKE_PATH
    NO_CMAKE_ENVIRONMENT_PATH
  )
  if(EXISTS "${${OUTPUT_VAR}}")
    message(STATUS "Found ${PROG_NAME}: ${${OUTPUT_VAR}}")
  else()
    message(FATAL_ERROR "${PROG_NAME} executable not found in ${LOCATION}!")
  endif()
endmacro()

macro(find_llvm_program OUTPUT_VAR PROG_NAME LOCATION DOCSTRING)
  find_program(${OUTPUT_VAR}
    NAMES "${PROG_NAME}${LLVM_BINARY_SUFFIX}${CMAKE_EXECUTABLE_SUFFIX}"
          "${PROG_NAME}-${LLVM_VERSION_MAJOR}${CMAKE_EXECUTABLE_SUFFIX}"
    # At least the LLVM v21 .deb doesn't have a clang++-21 under
    # /usr/lib/llvm-21/bin, but only 'clang++', 'clang' and
    # clang-21 symlink. Thus when looking only in the install prefix,
    # we don't find the clang++-21 symlink that is only in /usr/bin.
    # So, look also for the non-suffixed ones since we are searching
    # in the install/config dir.
    "${PROG_NAME}${CMAKE_EXECUTABLE_SUFFIX}"
    HINTS ${LOCATION}
    DOC "${DOCSTRING}"
    NO_DEFAULT_PATH
    NO_CMAKE_PATH
    NO_CMAKE_ENVIRONMENT_PATH
  )
  if(EXISTS "${${OUTPUT_VAR}}")
    message(STATUS "Found ${PROG_NAME}: ${${OUTPUT_VAR}}")
  endif()
endmacro()


# try compile with any compiler (supplied as argument)
macro(custom_try_compile_any SILENT COMPILER SUFFIX SOURCE RES_VAR)
  string(RANDOM RNDNAME)
  set(RANDOM_FILENAME "${CMAKE_BINARY_DIR}/compile_test_${RNDNAME}.${SUFFIX}")
  file(WRITE "${RANDOM_FILENAME}" "${SOURCE}")

  math(EXPR LSIZE "${ARGC} - 4")

  execute_process(COMMAND "${COMPILER}" ${ARGN} "${RANDOM_FILENAME}"
                  WORKING_DIRECTORY "${CMAKE_BINARY_DIR}"
                  RESULT_VARIABLE RESV
                  OUTPUT_VARIABLE OV
                  ERROR_VARIABLE EV)
  if((NOT ${RESV} EQUAL 0) AND (NOT ${SILENT}))
    message(STATUS " ########## The command: ")
    string(REPLACE ";" " " ARGN_STR "${ARGN}")
    message(STATUS "${COMPILER} ${ARGN_STR} ${RANDOM_FILENAME}")
    message(STATUS " ########## Exited with nonzero status: ${${RES_VAR}}")
    if(OV)
      message(STATUS "STDOUT: ${OV}")
    endif()
    if(EV)
      message(STATUS "STDERR: ${EV}")
    endif()
  endif()
  file(REMOVE "${RANDOM_FILENAME}")

  set(${RES_VAR} ${RESV})
endmacro()

# convenience c/c++ source wrapper
macro(custom_try_compile_c_cxx COMPILER SUFFIX SOURCE1 SOURCE2 RES_VAR)
  set(SOURCE_PROG "
  ${SOURCE1}

  int main(int argc, char** argv) {

  ${SOURCE2}

  }")
  custom_try_compile_any(FALSE "${COMPILER}" ${SUFFIX} "${SOURCE_PROG}" ${RES_VAR} ${ARGN})
endmacro()

# convenience c/c++ source wrapper
macro(custom_try_compile_c_cxx_silent COMPILER SUFFIX SOURCE1 SOURCE2 RES_VAR)
  set(SOURCE_PROG "
  ${SOURCE1}

  int main(int argc, char** argv) {

  ${SOURCE2}

  }")
  custom_try_compile_any(TRUE "${COMPILER}" ${SUFFIX} "${SOURCE_PROG}" ${RES_VAR} ${ARGN})
endmacro()

# clang++ try-compile macro
macro(custom_try_compile_clangxx SOURCE1 SOURCE2 RES_VAR)
  string(RANDOM RNDNAME)
  set(RANDOM_FILENAME "${CMAKE_BINARY_DIR}/compile_test_${RNDNAME}.o")
  custom_try_compile_c_cxx("${HOST_CLANGXX}" "cc" "${SOURCE1}" "${SOURCE2}" ${RES_VAR} "-o" "${RANDOM_FILENAME}" "-c" ${ARGN})
  file(REMOVE "${RANDOM_FILENAME}")
endmacro()

# clang++ try-compile macro
macro(custom_try_compile_clang SOURCE1 SOURCE2 RES_VAR)
  string(RANDOM RNDNAME)
  set(RANDOM_FILENAME "${CMAKE_BINARY_DIR}/compile_test_${RNDNAME}.o")
  custom_try_compile_c_cxx("${HOST_CLANG}" "c" "${SOURCE1}" "${SOURCE2}" ${RES_VAR} "-o" "${RANDOM_FILENAME}" "-c" ${ARGN})
  file(REMOVE "${RANDOM_FILENAME}")
endmacro()

# clang++ try-compile macro
macro(custom_try_compile_clang_silent SOURCE1 SOURCE2 RES_VAR)
  string(RANDOM RNDNAME)
  set(RANDOM_FILENAME "${CMAKE_BINARY_DIR}/compile_test_${RNDNAME}.o")
  custom_try_compile_c_cxx_silent("${HOST_CLANG}" "c" "${SOURCE1}" "${SOURCE2}" ${RES_VAR} "-o" "${RANDOM_FILENAME}" "-c" ${ARGN})
  file(REMOVE "${RANDOM_FILENAME}")
endmacro()

# clang++ try-link macro
macro(custom_try_link_clang SOURCE1 SOURCE2 RES_VAR)
  string(RANDOM RNDNAME)
  set(RANDOM_FILENAME "${CMAKE_BINARY_DIR}/compile_test_${RNDNAME}${CMAKE_EXECUTABLE_SUFFIX}")
  custom_try_compile_c_cxx_silent("${HOST_CLANG}" "c" "${SOURCE1}" "${SOURCE2}" ${RES_VAR} "-o" "${RANDOM_FILENAME}" ${ARGN})
  file(REMOVE "${RANDOM_FILENAME}")
endmacro()

macro(custom_try_link_clangxx SOURCE1 SOURCE2 RES_VAR)
  string(RANDOM RNDNAME)
  set(RANDOM_FILENAME "${CMAKE_BINARY_DIR}/compile_test_${RNDNAME}${CMAKE_EXECUTABLE_SUFFIX}")
  custom_try_compile_c_cxx_silent("${HOST_CLANGXX}" "cc" "${SOURCE1}" "${SOURCE2}" ${RES_VAR} "-o" "${RANDOM_FILENAME}" ${ARGN})
  file(REMOVE "${RANDOM_FILENAME}")
endmacro()
