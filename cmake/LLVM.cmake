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

if(DEFINED WITH_LLVM_CONFIG AND WITH_LLVM_CONFIG)
  # search for preferred version
  if(IS_ABSOLUTE "${WITH_LLVM_CONFIG}")
    if(EXISTS "${WITH_LLVM_CONFIG}")
      set(LLVM_CONFIG "${WITH_LLVM_CONFIG}")
    endif()
  else()
    find_program(LLVM_CONFIG NAMES "${WITH_LLVM_CONFIG}")
  endif()
else()
  # search for any version
  find_program(LLVM_CONFIG
    NAMES "llvm-config"
      "llvm-config-mp-3.3" "llvm-config-3.3" "llvm-config33"
      "llvm-config-mp-3.4" "llvm-config-3.4" "llvm-config34"
      "llvm-config-mp-3.5" "llvm-config-3.5" "llvm-config35"
      "llvm-config-mp-3.2" "llvm-config-3.2" "llvm-config32"
    DOC "llvm-config executable")
endif()

set(WITH_LLVM_CONFIG "${WITH_LLVM_CONFIG}" CACHE PATH "Path to preferred llvm-config")

if(NOT LLVM_CONFIG)
  message(FATAL_ERROR "llvm-config not found !")
else()
  message(STATUS "Using llvm-config: ${LLVM_CONFIG}")
  if(LLVM_CONFIG MATCHES "llvm-config${CMAKE_EXECUTABLE_SUFFIX}$")
    set(LLVM_BINARY_SUFFIX "")
  elseif(LLVM_CONFIG MATCHES "llvm-config(.*)${CMAKE_EXECUTABLE_SUFFIX}$")
    set(LLVM_BINARY_SUFFIX "${CMAKE_MATCH_1}")
  else()
    message(WARNING "Cannot determine llvm binary suffix from ${LLVM_CONFIG}")
  endif()
  message(STATUS "LLVM binaries suffix : ${LLVM_BINARY_SUFFIX}")
endif()

get_filename_component(LLVM_CONFIG_LOCATION "${LLVM_CONFIG}" DIRECTORY)

##########################################################################

# A macro to run llvm config
macro(run_llvm_config VARIABLE_NAME)
  execute_process(
    COMMAND "${LLVM_CONFIG}" ${ARGN}
    OUTPUT_VARIABLE ${VARIABLE_NAME}
    RESULT_VARIABLE LLVM_CONFIG_RETVAL
    OUTPUT_STRIP_TRAILING_WHITESPACE
  )
  if(LLVM_CONFIG_RETVAL)
    message(SEND_ERROR "Error running llvm-config with arguments: ${ARGN}")
  else()
    message(STATUS "${VARIABLE_NAME} is ${${VARIABLE_NAME}}")
  endif()
endmacro(run_llvm_config)

run_llvm_config(LLVM_PREFIX --prefix)
set(LLVM_PREFIX_BIN "${LLVM_PREFIX}/bin")
run_llvm_config(LLVM_VERSION_FULL --version)
# sigh, sanitize version... `llvm --version` on debian might return 3.4.1 but llvm command names are still <command>-3.4
string(REGEX REPLACE "([0-9]+)\\.([0-9]+).*" "\\1.\\2" LLVM_VERSION "${LLVM_VERSION_FULL}")
message(STATUS "LLVM_VERSION: ${LLVM_VERSION}")

run_llvm_config(LLVM_CFLAGS --cflags)
run_llvm_config(LLVM_CXXFLAGS --cxxflags)
run_llvm_config(LLVM_CPPFLAGS --cppflags)
run_llvm_config(LLVM_LDFLAGS --ldflags)
run_llvm_config(LLVM_LIBDIR --libdir)
run_llvm_config(LLVM_INCLUDEDIR --includedir)
run_llvm_config(LLVM_LIBS --libs)
run_llvm_config(LLVM_LIBFILES --libfiles)
run_llvm_config(LLVM_ALL_TARGETS --targets-built)
run_llvm_config(LLVM_HOST_TARGET --host-target)
# Ubuntu's llvm reports "arm-unknown-linux-gnueabihf" triple, then if one tries
# `clang --target=arm-unknown-linux-gnueabihf ...` it will produce armv6 code,
# even if one's running armv7;
# Here we replace the "arm" string with whatever's in CMAKE_HOST_SYSTEM_PROCESSOR
# which should be "armv6l" on rasp pi, or "armv7l" on my cubieboard, hopefully its
# more reasonable and reliable than llvm's own host flags
string(REPLACE "arm-" "${CMAKE_HOST_SYSTEM_PROCESSOR}-" LLVM_HOST_TARGET "${LLVM_HOST_TARGET}")

# required for sources..
if(LLVM_VERSION MATCHES "3[.]([0-9]+)")
  string(STRIP "${CMAKE_MATCH_1}" LLVM_MINOR)
  message(STATUS "Minor llvm version: ${LLVM_MINOR}")
  if(LLVM_MINOR STREQUAL "1")
    set(LLVM_3_1 1)
  elseif(LLVM_MINOR STREQUAL "2")
    set(LLVM_3_2 1)
  elseif(LLVM_MINOR STREQUAL "3")
    set(LLVM_3_3 1)
  elseif(LLVM_MINOR STREQUAL "4")
    set(LLVM_3_4 1)
  elseif(LLVM_MINOR STREQUAL "5")
    set(LLVM_3_5 1)
  else()
    message(WARNING "Unknown minor llvm version.")
  endif()
endif()

####################################################################

# DONE
if("${LLVM_CXXFLAGS}" MATCHES "-fno-rtti")
  message(WARNING "Your LLVM was not built with RTTI.
       You should rebuild LLVM with 'make REQUIRES_RTTI=1'.
       See the INSTALL file for more information.")
endif()


# A few work-arounds for llvm-config issues

# - pocl doesn't compile with '-pedantic'
#LLVM_CXX_FLAGS=$($LLVM_CONFIG --cxxflags | sed -e 's/ -pedantic / /g')
string(REPLACE " -pedantic" "" LLVM_CXXFLAGS "${LLVM_CXXFLAGS}")

# - '-fno-rtti' is a work-around for llvm bug 14200
#LLVM_CXX_FLAGS="$LLVM_CXX_FLAGS -fno-rtti"
set(LLVM_CXXFLAGS "${LLVM_CXXFLAGS} -fno-rtti")

if(NOT LLVM_VERSION VERSION_LESS "3.5")
  run_llvm_config(LLVM_SYSLIBS --system-libs)
  set(LLVM_LDFLAGS "${LLVM_LDFLAGS} ${LLVM_SYSLIBS}")
endif()

####################################################################

macro(find_program_or_die OUTPUT_VAR PROG_NAME DOCSTRING)
  find_program(${OUTPUT_VAR} NAMES "${PROG_NAME}${LLVM_BINARY_SUFFIX}${CMAKE_EXECUTABLE_SUFFIX}" "${PROG_NAME}${CMAKE_EXECUTABLE_SUFFIX}" HINTS "${LLVM_CONFIG_LOCATION}" "${LLVM_PREFIX}" "${LLVM_PREFIX_BIN}" DOC "${DOCSTRING}")
  if(${OUTPUT_VAR})
    message(STATUS "Found ${PROG_NAME}: ${${OUTPUT_VAR}}")
  else()
    message(FATAL_ERROR "${PROG_NAME} executable not found!")
  endif()
endmacro()

find_program_or_die( CLANG "clang" "clang binary")
execute_process(COMMAND "${CLANG}" "--version" OUTPUT_VARIABLE LLVM_CLANG_VERSION RESULT_VARIABLE CLANG_RES)
# TODO this should be optional
find_program_or_die( CLANGXX "clang++" "clang++ binary")
execute_process(COMMAND "${CLANGXX}" "--version" OUTPUT_VARIABLE LLVM_CLANGXX_VERSION RESULT_VARIABLE CLANGXX_RES)
if(CLANGXX_RES OR CLANG_RES)
  message(FATAL_ERROR "Failed running clang/clang++ --version")
endif()

find_program_or_die(LLVM_OPT "opt" "LLVM optimizer")
find_program_or_die(LLC "llc" "LLVM static compiler") # TODO rename to LLVM_LLC
find_program_or_die(LLVM_AS "llvm-as" "LLVM assembler")
find_program_or_die(LLVM_LINK "llvm-link" "LLVM IR linker")
find_program_or_die(LLVM_LLI "lli" "LLVM interpreter")

####################################################################

# try compile with any compiler (supplied as argument)
macro(custom_try_compile_any COMPILER SUFFIX SOURCE RES_VAR)
  string(RANDOM RNDNAME)
  set(RANDOM_FILENAME "${CMAKE_BINARY_DIR}/compile_test_${RNDNAME}.${SUFFIX}")
  file(WRITE "${RANDOM_FILENAME}" "${SOURCE}")

  execute_process(COMMAND "${COMPILER}" ${ARGN} "${RANDOM_FILENAME}" RESULT_VARIABLE ${RES_VAR} OUTPUT_VARIABLE OV ERROR_VARIABLE EV)
  if(${${RES_VAR}})
    message(STATUS " ########## The command: ")
    message(STATUS "${COMPILER} ${ARGN} ${RANDOM_FILENAME}")
    message(STATUS " ########## Exited with nonzero status: ${${RES_VAR}}")
    if(OV)
      message(STATUS "STDOUT: ${OV}")
    endif()
    if(EV)
      message(STATUS "STDERR: ${EV}")
    endif()
  endif()
  file(REMOVE "${RANDOM_FILENAME}")

endmacro()

# convenience c/c++ source wrapper
macro(custom_try_compile_c_cxx COMPILER SUFFIX SOURCE1 SOURCE2 RES_VAR)
  set(SOURCE_PROG "
  ${SOURCE1}

  int main(int argc, char** argv) {

  ${SOURCE2}

  }")
  custom_try_compile_any("${COMPILER}" ${SUFFIX} "${SOURCE_PROG}" ${RES_VAR} ${ARGN})
endmacro()

# clang++ try-compile macro
macro(custom_try_compile_clangxx SOURCE1 SOURCE2 RES_VAR)
  # this is wrong
  #separate_arguments(FLAGS UNIX_COMMAND "${LLVM_CXXFLAGS}")
  custom_try_compile_c_cxx("${CLANGXX}" "cc" "${SOURCE1}" "${SOURCE2}" ${RES_VAR}  "-c" ${ARGN})
endmacro()

# clang try-compile-run macro, running via native executable
macro(custom_try_run_exe SOURCE1 SOURCE2 OUTPUT_VAR)
  set(OUTF "${CMAKE_BINARY_DIR}/try_run${CMAKE_EXECUTABLE_SUFFIX}")
  if(EXISTS "${OUTF}")
    file(REMOVE "${OUTF}")
  endif()
  custom_try_compile_c_cxx("${CLANG}" "c" "${SOURCE1}" "${SOURCE2}" RESV "-o" "${OUTF}" "-x" "c")
  set(${OUTPUT_VAR} "")
  if(RESV OR (NOT EXISTS "${OUTF}"))
    message(STATUS " ########## Compilation failed")
  else()
    execute_process(COMMAND "${OUTF}" RESULT_VARIABLE RESV OUTPUT_VARIABLE ${OUTPUT_VAR} ERROR_VARIABLE EV)
    file(REMOVE "${OUTF}")
    if(${RESV})
      message(STATUS " ########## Running ${OUTF}")
      message(STATUS " ########## Exited with nonzero status: ${RESV}")
      if(${${OUTPUT_VAR}})
        message(STATUS " ########## STDOUT: ${${OUTPUT_VAR}}")
      endif()
      if(EV)
        message(STATUS " ########## STDERR: ${EV}")
      endif()
    endif()
  endif()
endmacro()

# clang try-compile-run macro, run via lli, the llvm interpreter
macro(custom_try_run_lli SOURCE1 SOURCE2 OUTPUT_VAR)
# this uses "lli" - the interpreter, so we can run any -target
# TODO variable for target !!
  set(OUTF "${CMAKE_BINARY_DIR}/try_run.bc")
  if(EXISTS "${OUTF}")
    file(REMOVE "${OUTF}")
  endif()
  custom_try_compile_c_cxx("${CLANG}" "c" "${SOURCE1}" "${SOURCE2}" RESV "-o" "${OUTF}" "-x" "c" "-emit-llvm" "-c" ${ARGN})
  set(${OUTPUT_VAR} "")
  if(RESV OR (NOT EXISTS "${OUTF}"))
    message(STATUS " ########## Compilation failed")
  else()
    execute_process(COMMAND "${LLVM_LLI}" "${OUTF}" RESULT_VARIABLE RESV OUTPUT_VARIABLE ${OUTPUT_VAR} ERROR_VARIABLE EV)
    file(REMOVE "${OUTF}")
    if(${RESV})
      message(STATUS " ########## The command ${OUTF}")
      message(STATUS " ########## Exited with nonzero status: ${RESV}")
      if(${${OUTPUT_VAR}})
        message(STATUS " ########## STDOUT: ${${OUTPUT_VAR}}")
      endif()
      if(EV)
        message(STATUS " ########## STDERR: ${EV}")
      endif()
    endif()
  endif()
endmacro()

####################################################################

# helpers for caching variables, with cache dependent on supplied string

macro(setup_cache_var_name VARNAME DEP_STRING)
  string(MD5 ${VARNAME}_MD5 "${DEP_STRING}")
  set(CACHE_VAR_NAME "${VARNAME}_CACHE_${${VARNAME}_MD5}")
endmacro()

macro(set_cache_var VARNAME VAR_DOCS)
  if(DEFINED ${CACHE_VAR_NAME})
    set(${VARNAME} "${${CACHE_VAR_NAME}}")
    message(STATUS "${VAR_DOCS} (cached) : ${${CACHE_VAR_NAME}}")
  else()
    set(${CACHE_VAR_NAME} ${${VARNAME}} CACHE INTERNAL "${VAR_DOCS}")
    message(STATUS "${VAR_DOCS} : ${${CACHE_VAR_NAME}}")
  endif()
endmacro()

####################################################################

# The option for specifying the target changed; try the modern syntax
# first, and fall back to the old-style syntax if this failed

setup_cache_var_name(CLANG_TARGET_OPTION "${LLVM_HOST_TARGET}-${CLANG}-${LLVM_CLANG_VERSION}")

if(NOT DEFINED ${CACHE_VAR_NAME})

  custom_try_compile_clangxx("" "return 0;" RES "--target=${LLVM_HOST_TARGET}")
  if(NOT RES)
    set(CLANG_TARGET_OPTION "--target=")
  else()
    #EXECUTE_PROCESS(COMMAND "${CLANG}" "-target ${LLVM_HOST_TARGET}" "-x" "c" "/dev/null" "-S" RESULT_VARIABLE RES)
    custom_try_compile_clangxx("" "return 0;" RES "-target ${LLVM_HOST_TARGET}")
    if(NOT RES)
      set(CLANG_TARGET_OPTION "-target ")
    else()
      message(FATAL_ERROR "Cannot determine Clang option to specify the target")
    endif()
  endif()

endif()

set_cache_var(CLANG_TARGET_OPTION "Clang option used to specify the target" )


####################################################################

macro(CHECK_SIZEOF TYPE RES_VAR TRIPLE)
  setup_cache_var_name(SIZEOF "${TYPE}-${TRIPLE}-${CLANG}")

  if(NOT DEFINED ${CACHE_VAR_NAME})
    custom_try_run_exe("#include <stddef.h>\n #include <stdio.h>" "printf(\"%i\",(int)sizeof(${TYPE})); return 0;" ${RES_VAR} "${CLANG_TARGET_OPTION}${TRIPLE}")
    if(NOT ${RES_VAR})
      message(SEND_ERROR "Could not determine sizeof(${TYPE})")
    endif()
  endif()

  set_cache_var(${RES_VAR} "Size of ${TYPE}")
endmacro()

macro(CHECK_ALIGNOF TYPE TYPEDEF RES_VAR TRIPLE)
  setup_cache_var_name(ALIGNOF "${TYPE}-${TYPEDEF}-${TRIPLE}-${CLANG}")

  if(NOT DEFINED ${CACHE_VAR_NAME})

    custom_try_run_exe("
#include <stddef.h>
#include <stdio.h>

#ifndef offsetof
#define offsetof(type, member) ((char *) &((type *) 0)->member - (char *) 0)
#endif

${TYPEDEF}"  "typedef struct { char x; ${TYPE} y; } ac__type_alignof_;
    int r = offsetof(ac__type_alignof_, y);
    printf(\"%i\",r);
    return 0;" ${RES_VAR} "${CLANG_TARGET_OPTION}${TRIPLE}")

    if(NOT ${RES_VAR})
      message(SEND_ERROR "Could not determine align of(${TYPE})")
    endif()
  endif()

  set_cache_var(${RES_VAR} "Align of ${TYPE}")

endmacro()

####################################################################

# clangxx works check

if(CLANGXX)

  message(STATUS "Checking if clang++ works (required by vecmathlib)")

  setup_cache_var_name(CLANGXX_WORKS "${LLVM_HOST_TARGET}-${CLANGXX}-${LLVM_CLANGXX_VERSION}")

  if(NOT DEFINED ${CACHE_VAR_NAME})
    set(CLANGXX_WORKS 1)
    custom_try_compile_clangxx("namespace std { class type_info; } \n  #include <iostream>" "std::cout << \"Hello clang++ world!\" << std::endl;" COMPILE_RESULT)
    if(COMPILE_RESULT)
      set(CLANGXX_WORKS 0)
    endif()
  endif()

  set_cache_var(CLANGXX_WORKS "Clang++ works")

else()

  set(CLANGXX_WORKS 0)

endif()

####################################################################

# which C++ stdlib clang is using

if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")

  setup_cache_var_name(SYSTEM_CLANGXX_STDLIB "${LLVM_HOST_TARGET}-${CLANGXX}-${LLVM_CLANGXX_VERSION}")

  if(NOT DEFINED ${CACHE_VAR_NAME})
    message(STATUS "Checking if system clang++ uses libstdc++")
    separate_arguments(_FLAGS UNIX_COMMAND "${LLVM_CXXFLAGS}")
    custom_try_compile_clangxx("#include <llvm/Support/Debug.h>\n#include <iostream>\nusing namespace std;\n"
        "cout << 234134; return 0;" COMPILE_RESULT ${_FLAGS} "-stdlib=libstdc++")
    if(COMPILE_RESULT)
      custom_try_compile_clangxx("#include <llvm/Support/Debug.h>\n#include <iostream>\nusing namespace std;\n"
          "cout << 234134; return 0;" COMPILE_RESULT ${_FLAGS} "-stdlib=libc++")
      if(COMPILE_RESULT)
        message(FATAL_ERROR "System clang cannot compile with neither libstdc++ nor libc++; possibly missing llvm heaers?")
      else()
        set(SYSTEM_CLANGXX_STDLIB "-stdlib=libc++")
      endif()
    else()
      set(SYSTEM_CLANGXX_STDLIB "-stdlib=libstdc++")
    endif()
  endif()

  set_cache_var(SYSTEM_CLANGXX_STDLIB "Clang's c++ stdlib")
  set(LLVM_CXXFLAGS "${LLVM_CXXFLAGS} ${SYSTEM_CLANGXX_STDLIB}")
endif()

if("${SYSTEM_CLANGXX_STDLIB}" MATCHES "libc")
  set(LLVM_LDFLAGS "${LLVM_LDFLAGS} -lc++")
else()
  set(LLVM_LDFLAGS "${LLVM_LDFLAGS} -lstdc++")
endif()


####################################################################

# - '-DNDEBUG' is a work-around for llvm bug 18253

# llvm-config does not always report the "-DNDEBUG" flag correctly
# (see LLVM bug 18253). If LLVM and the pocl passes are built with
# different NDEBUG settings, problems arise

if(NOT LLVM_CXXFLAGS MATCHES "-DNDEBUG")

  message(STATUS "Checking if LLVM is built with assertions")
  separate_arguments(_FLAGS UNIX_COMMAND "${LLVM_CXXFLAGS}")
  custom_try_compile_clangxx("#include <llvm/Support/Debug.h>" "llvm::DebugFlag=true;" COMPILE_RESULT ${_FLAGS} "-UNDEBUG")
  if(COMPILE_RESULT)
    message(STATUS "no assertions... adding -DNDEBUG")
    set(LLVM_CXXFLAGS "${LLVM_CXXFLAGS} -DNDEBUG")
  endif()
  unset(_FLAGS)

endif()

####################################################################

# DONE

# TODO: We need to set both target-triple and cpu-type when
# building, since the ABI depends on both. We can either add flags
# to all the scripts, or set the respective flags here in
# *_CLANG_FLAGS and *_LLC_FLAGS. Note that clang and llc use
# different option names to set these. Note that clang calls the
# triple "target" and the cpu "architecture", which is different
# from llc.

# Normalise the triple. Otherwise, clang normalises it when
# passing it to llc, which is then different from the triple we
# pass to llc. This would lead to inconsistent bytecode files,
# depending on whether they are generated via clang or directly
# via llc.

setup_cache_var_name(LLC_TRIPLE "LLC_TRIPLE-${LLVM_HOST_TARGET}-${CLANG}")

if(NOT DEFINED ${CACHE_VAR_NAME})
  message(STATUS "Find out LLC target triple (for host ${LLVM_HOST_TARGET})")
  execute_process(COMMAND ${CLANG} "${CLANG_TARGET_OPTION}${LLVM_HOST_TARGET}" -x c /dev/null -S -emit-llvm -o - RESULT_VARIABLE RES_VAR OUTPUT_VARIABLE OUTPUT_VAR)
  if(RES_VAR)
    message(FATAL_ERROR "Error ${RES_VAR} while determining target triple")
  endif()
  if(OUTPUT_VAR MATCHES "target triple = \"([^\"]+)")
    string(STRIP "${CMAKE_MATCH_1}" LLC_TRIPLE)
  else()
    message(FATAL_ERROR "Could not find target triple in llvm output")
  endif()

  # TODO the armv7hl normalize
  string(REPLACE "armv7l-" "armv7-" LLC_TRIPLE "${LLC_TRIPLE}")

endif()

set_cache_var(LLC_TRIPLE "LLC_TRIPLE")



setup_cache_var_name(LLC_HOST_CPU "LLC_HOST_CPU-${LLVM_HOST_TARGET}-${LLC}")

if(NOT DEFINED ${CACHE_VAR_NAME})
  message(STATUS "Find out LLC host CPU with ${LLC}")
  execute_process(COMMAND ${LLC} "--version" RESULT_VARIABLE RES_VAR OUTPUT_VARIABLE OUTPUT_VAR)
  # WTF, ^^ has return value 1
  #if(RES_VAR)
  #  message(FATAL_ERROR "Error ${RES_VAR} while determining LLC host CPU")
  #endif()

  if(OUTPUT_VAR MATCHES "Host CPU: ([^ ]*)")
    # sigh... STRING(STRIP is to workaround regexp bug in cmake
    string(STRIP "${CMAKE_MATCH_1}" LLC_HOST_CPU)
  else()
    message(FATAL_ERROR "Couldnt determine host CPU from llc output")
  endif()

  #TODO better
  if(CMAKE_LIBRARY_ARCHITECTURE MATCHES "gnueabihf" AND LLC_HOST_CPU MATCHES "arm1176jz-s")
    set(LLC_HOST_CPU "arm1176jzf-s")
  endif()

endif()

set_cache_var(LLC_HOST_CPU "LLC_HOST_CPU")

####################################################################
#X86 has -march and -mcpu reversed, for clang

if(CMAKE_HOST_SYSTEM_PROCESSOR MATCHES "(powerpc|armv7)")
  set(CLANG_MARCH_FLAG "-mcpu=")
else()
  set(CLANG_MARCH_FLAG "-march=")
endif()

####################################################################
# line 823 in configure.ac:
# case $host_cpu in

#~
#~ if(CMAKE_HOST_SYSTEM_PROCESSOR MATCHES "armv6l")
    #~ MESSAGE(STATUS "Using the ARM optimized kernel lib for the native device")
    #~ # TODO better...
    #~ ;;
#~
#~
#~ elseif(CMAKE_HOST_SYSTEM_PROCESSOR MATCHES "(x86_64|AMD64)")
  #~ message(STATUS "using the x86_64 optimized kernel lib for the native device")
#~
#~ endif()


####################################################################

# Work-around a clang bug in LLVM 3.3: On 32-bit platforms, the size
# of Open CL C long is not 8 bytes

#  set(_CL_DISABLE_LONG ${BUG_PRESENT} CACHE INTERNAL "bug in LLVM 3.3: On 32-bit platforms, the size of Open CL C long is not 8 bytes")

setup_cache_var_name(CL_DISABLE_LONG "CL_DISABLE_LONG-${LLVM_HOST_TARGET}-${CLANG}")

if(NOT DEFINED ${CACHE_VAR_NAME})
  set(CL_DISABLE_LONG 0)
  # TODO -march=CPU flags !
  custom_try_compile_any("${CLANG}" "cl" "constant int test[sizeof(long)==8?1:-1]={1};" RESV  -x cl -S ${CLANG_TARGET_OPTION}${LLC_TRIPLE} ${CLANG_MARCH_FLAG}${LLC_HOST_CPU})
  if(RESV)
    message(STATUS "Your llvm has the Open-CL-C-long-is-not-8-bytes bug; disabling cl_khr_int64 on host based devices.")
    set(CL_DISABLE_LONG 1)
  endif()
endif()

set_cache_var(CL_DISABLE_LONG "Disable cl_khr_int64 because of buggy llvm")


####################################################################

setup_cache_var_name(CL_DISABLE_HALF "CL_DISABLE_HALF-${LLVM_HOST_TARGET}-${CLANG}")

if(NOT DEFINED ${CACHE_VAR_NAME})
  set(CL_DISABLE_HALF 0)
  # TODO -march=CPU flags !
  custom_try_compile_c_cxx("${CLANG}" "c" "__fp16 callfp16(__fp16 a) { return a * (__fp16)1.8; };" "__fp16 x=callfp16((__fp16)argc);" RESV -c ${CLANG_TARGET_OPTION}${LLC_TRIPLE} ${CLANG_MARCH_FLAG}${LLC_HOST_CPU})
  if(RESV)
    message(STATUS "Disabling cl_khr_fp16, seems your system doesnt support it")
    set(CL_DISABLE_HALF 1)
  endif()
endif()

set_cache_var(CL_DISABLE_HALF "Disable cl_khr_fp16 because fp16 is not supported")
