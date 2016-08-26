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
      "llvm-config-mp-3.8" "llvm-config-3.8" "llvm-config38"
      "llvm-config-mp-3.7" "llvm-config-3.7" "llvm-config37"
    DOC "llvm-config executable")
endif()

set(WITH_LLVM_CONFIG "${WITH_LLVM_CONFIG}" CACHE PATH "Path to preferred llvm-config")

if(NOT LLVM_CONFIG)
  message(FATAL_ERROR "llvm-config not found !")
else()
  file(TO_CMAKE_PATH "${LLVM_CONFIG}" LLVM_CONFIG)
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
    message(STATUS "llvm-config's ${VARIABLE_NAME} is: ${${VARIABLE_NAME}}")
  endif()
endmacro(run_llvm_config)

run_llvm_config(LLVM_PREFIX --prefix)
# on windows, llvm-config returs "C:\llvm_prefix/bin" mixed style paths,
# and cmake doesn't like the "\" - thinks its an escape char..
file(TO_CMAKE_PATH "${LLVM_PREFIX}" LLVM_PREFIX_CMAKE)

set(LLVM_PREFIX_BIN "${LLVM_PREFIX_CMAKE}/bin")
run_llvm_config(LLVM_VERSION_FULL --version)
# sigh, sanitize version... `llvm --version` on debian might return 3.4.1 but llvm command names are still <command>-3.4
string(REGEX REPLACE "([0-9]+)\\.([0-9]+).*" "\\1.\\2" LLVM_VERSION "${LLVM_VERSION_FULL}")
message(STATUS "LLVM_VERSION: ${LLVM_VERSION}")

run_llvm_config(LLVM_CFLAGS --cflags)
string(REPLACE "${LLVM_PREFIX}" "${LLVM_PREFIX_CMAKE}" LLVM_CFLAGS "${LLVM_CFLAGS}")
run_llvm_config(LLVM_CXXFLAGS --cxxflags)
string(REPLACE "${LLVM_PREFIX}" "${LLVM_PREFIX_CMAKE}" LLVM_CXXFLAGS "${LLVM_CXXFLAGS}")
run_llvm_config(LLVM_CPPFLAGS --cppflags)
string(REPLACE "${LLVM_PREFIX}" "${LLVM_PREFIX_CMAKE}" LLVM_CPPFLAGS "${LLVM_CPPFLAGS}")
run_llvm_config(LLVM_LDFLAGS --ldflags)
string(REPLACE "${LLVM_PREFIX}" "${LLVM_PREFIX_CMAKE}" LLVM_LDFLAGS "${LLVM_LDFLAGS}")
run_llvm_config(LLVM_BINDIR --bindir)
string(REPLACE "${LLVM_PREFIX}" "${LLVM_PREFIX_CMAKE}" LLVM_BINDIR "${LLVM_BINDIR}")
run_llvm_config(LLVM_LIBDIR --libdir)
string(REPLACE "${LLVM_PREFIX}" "${LLVM_PREFIX_CMAKE}" LLVM_LIBDIR "${LLVM_LIBDIR}")
run_llvm_config(LLVM_INCLUDEDIR --includedir)
string(REPLACE "${LLVM_PREFIX}" "${LLVM_PREFIX_CMAKE}" LLVM_INCLUDEDIR "${LLVM_INCLUDEDIR}")
run_llvm_config(LLVM_LIBS --libs)
# Convert LLVM_LIBS from string -> list format to make handling them easier
separate_arguments(LLVM_LIBS)
# workaround for a bug in current HSAIL LLVM
# it forgets to report one HSAIL library in llvm-config
if(ENABLE_HSA)
  if(NOT HSA_RUNTIME_IS_ROCM)
    list(APPEND LLVM_LIBS "-lLLVMHSAILUtil")
  endif()
endif()
run_llvm_config(LLVM_SRC_ROOT --src-root)
run_llvm_config(LLVM_OBJ_ROOT --obj-root)
string(REPLACE "${LLVM_PREFIX}" "${LLVM_PREFIX_CMAKE}" LLVM_OBJ_ROOT "${LLVM_OBJ_ROOT}")
run_llvm_config(LLVM_ALL_TARGETS --targets-built)
run_llvm_config(LLVM_HOST_TARGET --host-target)
run_llvm_config(LLVM_BUILD_MODE --build-mode)
run_llvm_config(LLVM_ASSERTS_BUILD --assertion-mode)
run_llvm_config(LLVM_SYSLIBS --system-libs)
string(STRIP "${LLVM_SYSLIBS}" LLVM_SYSLIBS)

if (MSVC)
  string(REPLACE "-L${LLVM_LIBDIR}" "" LLVM_LDFLAGS "${LLVM_LDFLAGS}")
  string(STRIP "${LLVM_LDFLAGS}" LLVM_LDFLAGS)
endif()

if(LLVM_BUILD_MODE MATCHES "Debug")
  set(LLVM_BUILD_MODE_DEBUG 1)
else()
  set(LLVM_BUILD_MODE_DEBUG 0)
endif()

# Ubuntu's llvm reports "arm-unknown-linux-gnueabihf" triple, then if one tries
# `clang --target=arm-unknown-linux-gnueabihf ...` it will produce armv6 code,
# even if one's running armv7;
# Here we replace the "arm" string with whatever's in CMAKE_HOST_SYSTEM_PROCESSOR
# which should be "armv6l" on rasp pi, or "armv7l" on my cubieboard, hopefully its
# more reasonable and reliable than llvm's own host flags
if(NOT CMAKE_CROSSCOMPILING)
  string(REPLACE "arm-" "${CMAKE_HOST_SYSTEM_PROCESSOR}-" LLVM_HOST_TARGET "${LLVM_HOST_TARGET}")
endif()

# In windows llvm-config reports --target=x86_64-pc-windows-msvc
# however this causes clang to use MicrosoftCXXMangler, which does not
# yet support mangling for extended vector types (with llvm 3.5)
# so for now hardcode LLVM_HOST_TARGET to be x86_64-pc with windows
if(WIN32)
  set(LLVM_HOST_TARGET "x86_64-pc")
endif(WIN32)

# required for sources..
if(LLVM_VERSION MATCHES "3[.]([0-9]+)")
  set(LLVM_MAJOR 3)
  string(STRIP "${CMAKE_MATCH_1}" LLVM_MINOR)
  message(STATUS "Minor llvm version: ${LLVM_MINOR}")
  if(LLVM_MINOR STREQUAL "6")
    set(LLVM_3_6 1)
  elseif(LLVM_MINOR STREQUAL "7")
    set(LLVM_3_7 1)
  elseif(LLVM_MINOR STREQUAL "8")
    set(LLVM_3_8 1)
  elseif(LLVM_MINOR STREQUAL "9")
    set(LLVM_3_9 1)
  else()
    message(FATAL_ERROR "Unknown/unsupported minor llvm version: ${LLVM_MINOR}")
  endif()
else()
  message(FATAL_ERROR "LLVM version 3.x required, found: ${LLVM_VERSION}")
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
# Which according to bug report has been fixed in llvm 3.7
# sadly, that bugreport is mistaken, it's not fixed in 3.7
if (NOT MSVC)
  set(LLVM_CXXFLAGS "${LLVM_CXXFLAGS} -fno-rtti")
endif()


# Llvm-config may be installed or it might be used from build directory, in which case
# we need to add few extra include paths to find clang includes and compiled includes
list(APPEND LLVM_INCLUDE_DIRS 
  "${LLVM_INCLUDEDIR}" 
  "${LLVM_SRC_ROOT}/tools/clang/include" 
  "${LLVM_OBJ_ROOT}/include" 
  "${LLVM_OBJ_ROOT}/tools/clang/include")

# Llvm-config does not include clang libs
set(CLANG_LIBNAMES clangFrontendTool clangFrontend clangDriver clangSerialization
    clangCodeGen clangParse clangSema clangRewrite clangRewriteFrontend
    clangStaticAnalyzerFrontend clangStaticAnalyzerCheckers
    clangStaticAnalyzerCore clangAnalysis clangEdit clangAST clangLex clangBasic)

foreach(LIBNAME ${CLANG_LIBNAMES})
  find_library(C_LIBFILE_${LIBNAME} NAMES "${LIBNAME}" HINTS "${LLVM_LIBDIR}")
  list(APPEND CLANG_LIBFILES "${C_LIBFILE_${LIBNAME}}")
endforeach()

# With Visual Studio llvm-config gives invalid list of static libs (libXXXX.a instead of XXXX.lib)
# we extract the pure names (LLVMLTO, LLVMMipsDesc etc) and let find_library do its job
foreach(LIBFLAG ${LLVM_LIBS})
  STRING(REGEX REPLACE "^-l(.*)$" "\\1" LIB_NAME ${LIBFLAG})
  list(APPEND LLVM_LIBNAMES "${LIB_NAME}")
endforeach()

foreach(LIBNAME ${LLVM_LIBNAMES})
  find_library(L_LIBFILE_${LIBNAME} NAMES "${LIBNAME}" HINTS "${LLVM_LIBDIR}")
  list(APPEND LLVM_LIBFILES "${L_LIBFILE_${LIBNAME}}")
endforeach()


####################################################################

macro(find_program_or_die OUTPUT_VAR PROG_NAME DOCSTRING)
  find_program(${OUTPUT_VAR} NAMES "${PROG_NAME}${LLVM_BINARY_SUFFIX}${CMAKE_EXECUTABLE_SUFFIX}" "${PROG_NAME}${CMAKE_EXECUTABLE_SUFFIX}" HINTS "${LLVM_BINDIR}" "${LLVM_CONFIG_LOCATION}" "${LLVM_PREFIX}" "${LLVM_PREFIX_BIN}" DOC "${DOCSTRING}")
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
find_program_or_die(LLVM_LLC "llc" "LLVM static compiler")
find_program_or_die(LLVM_AS "llvm-as" "LLVM assembler")
find_program_or_die(LLVM_LINK "llvm-link" "LLVM IR linker")
find_program_or_die(LLVM_LLI "lli" "LLVM interpreter")

# only required for ROCM
if(HSA_RUNTIME_IS_ROCM)
  find_program_or_die(LLVM_LD "ld.lld" "LLVM MC linker")
endif()

####################################################################

# try compile with any compiler (supplied as argument)
macro(custom_try_compile_any SILENT COMPILER SUFFIX SOURCE RES_VAR)
  string(RANDOM RNDNAME)
  set(RANDOM_FILENAME "${CMAKE_BINARY_DIR}/compile_test_${RNDNAME}.${SUFFIX}")
  file(WRITE "${RANDOM_FILENAME}" "${SOURCE}")

  math(EXPR LSIZE "${ARGC} - 4")

  execute_process(COMMAND "${COMPILER}" ${ARGN} "${RANDOM_FILENAME}" RESULT_VARIABLE ${RES_VAR} OUTPUT_VARIABLE OV ERROR_VARIABLE EV)
  if(${${RES_VAR}} AND (NOT ${SILENT}))
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
  # this is wrong
  #separate_arguments(FLAGS UNIX_COMMAND "${LLVM_CXXFLAGS}")
  custom_try_compile_c_cxx("${CLANGXX}" "cc" "${SOURCE1}" "${SOURCE2}" ${RES_VAR}  "-c" ${ARGN})
endmacro()

# clang try-compile-run macro, running via native executable
macro(custom_try_run_exe SOURCE1 SOURCE2 OUTPUT_VAR RES_VAR)
  set(OUTF "${CMAKE_BINARY_DIR}/try_run${CMAKE_EXECUTABLE_SUFFIX}")
  if(EXISTS "${OUTF}")
    file(REMOVE "${OUTF}")
  endif()
  custom_try_compile_c_cxx("${CLANG}" "c" "${SOURCE1}" "${SOURCE2}" RESV "-o" "${OUTF}" "-x" "c")
  set(${OUTPUT_VAR} "")
  set(${RES_VAR} "")
  if(RESV OR (NOT EXISTS "${OUTF}"))
    message(STATUS " ########## Compilation failed")
  else()
    execute_process(COMMAND "${OUTF}" RESULT_VARIABLE RESV OUTPUT_VARIABLE ${OUTPUT_VAR} ERROR_VARIABLE EV)
    set(${RES_VAR} ${RESV})
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
macro(custom_try_run_lli SILENT SOURCE1 SOURCE2 OUTPUT_VAR RES_VAR)
# this uses "lli" - the interpreter, so we can run any -target
# TODO variable for target !!
  set(OUTF "${CMAKE_BINARY_DIR}/try_run.bc")
  if(EXISTS "${OUTF}")
    file(REMOVE "${OUTF}")
  endif()
  custom_try_compile_c_cxx("${CLANG}" "c" "${SOURCE1}" "${SOURCE2}" RESV "-o" "${OUTF}" "-x" "c" "-emit-llvm" "-c" ${ARGN})
  set(${OUTPUT_VAR} "")
  set(${RES_VAR} "")
  if(RESV OR (NOT EXISTS "${OUTF}"))
    message(STATUS " ########## Compilation failed")
  else()
    execute_process(COMMAND "${LLVM_LLI}" "-force-interpreter" "${OUTF}" RESULT_VARIABLE RESV OUTPUT_VARIABLE ${OUTPUT_VAR} ERROR_VARIABLE EV)
    set(${RES_VAR} ${RESV})
    file(REMOVE "${OUTF}")
    if(${RESV} AND (NOT ${SILENT}))
      message(STATUS " ########## The command ${LLVM_LLI} -force-interpreter ${OUTF}")
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

macro(CHECK_ALIGNOF TYPE TYPEDEF RES_VAR TRIPLE)
  setup_cache_var_name(ALIGNOF "${TYPE}-${TYPEDEF}-${TRIPLE}-${CLANG}")

  if(NOT DEFINED ${CACHE_VAR_NAME})

    custom_try_run_lli(TRUE "
#ifndef offsetof
#define offsetof(type, member) ((char *) &((type *) 0)->member - (char *) 0)
#endif

${TYPEDEF}" "typedef struct { char x; ${TYPE} y; } ac__type_alignof_;
    int r = offsetof(ac__type_alignof_, y);
    return r;" SIZEOF_STDOUT ${RES_VAR} "${CLANG_TARGET_OPTION}${TRIPLE}")

    if(NOT ${RES_VAR})
      message(SEND_ERROR "Could not determine align of(${TYPE})")
    endif()
  endif()

  set_cache_var(${RES_VAR} "Align of ${TYPE}")

endmacro()

####################################################################
#
# clangxx works check 
#

# TODO clang + vecmathlib doesn't work on Windows yet...
if(CLANGXX AND (NOT WIN32))

  message(STATUS "Checking if clang++ works (required by vecmathlib)")

  setup_cache_var_name(CLANGXX_WORKS "${LLVM_HOST_TARGET}-${CLANGXX}-${LLVM_CLANGXX_VERSION}")

  if(NOT DEFINED ${CACHE_VAR_NAME})
    set(CLANGXX_WORKS 0)

    custom_try_compile_clangxx("namespace std { class type_info; } \n  #include <iostream> \n  #include <type_traits>" "std::cout << \"Hello clang++ world!\" << std::endl;" _STATUS_FAIL "-std=c++11")

    if(NOT _STATUS_FAIL)
      set(CLANGXX_STDLIB "")
      set(CLANGXX_WORKS 1)
    else()
      custom_try_compile_clangxx("namespace std { class type_info; } \n  #include <iostream> \n  #include <type_traits>" "std::cout << \"Hello clang++ world!\" << std::endl;" _STATUS_FAIL "-stdlib=libstdc++" "-std=c++11")
      if (NOT _STATUS_FAIL)
        set(CLANGXX_STDLIB "-stdlib=libstdc++")
        set(CLANGXX_WORKS 1)
      else()
        custom_try_compile_clangxx("namespace std { class type_info; } \n  #include <iostream> \n  #include <type_traits>" "std::cout << \"Hello clang++ world!\" << std::endl;" _STATUS_FAIL "-stdlib=libc++" "-std=c++11")
        if(NOT _STATUS_FAIL)
          set(CLANGXX_STDLIB "-stdlib=libc++")
          set(CLANGXX_WORKS 1)
        endif()
      endif()
    endif()
  endif()

  set_cache_var(CLANGXX_WORKS "Clang++ works with ${CLANGXX_STDLIB}")

else()

  set(CLANGXX_WORKS 0)

endif()

if(CLANGXX_STDLIB AND (CMAKE_CXX_COMPILER_ID MATCHES "Clang"))
  set(LLVM_CXXFLAGS "${CLANGXX_STDLIB} ${LLVM_CXXFLAGS}")
  set(LLVM_LDFLAGS "${CLANGXX_STDLIB} ${LLVM_LDFLAGS}")
endif()

####################################################################
#
# - '-DNDEBUG' is a work-around for llvm bug 18253
#
# llvm-config does not always report the "-DNDEBUG" flag correctly
# (see LLVM bug 18253). If LLVM and the pocl passes are built with
# different NDEBUG settings, problems arise

if(NOT LLVM_CXXFLAGS MATCHES "-DNDEBUG")

  message(STATUS "Checking if LLVM is a DEBUG build")
  separate_arguments(_FLAGS UNIX_COMMAND "${LLVM_CXXFLAGS}")

  set(_TEST_SOURCE "${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/llvmNDEBUG.cc")
  file(WRITE "${_TEST_SOURCE}"
    "
      #include <llvm/Support/Debug.h>
      int main(int argc, char** argv) {
        llvm::DebugFlag=true;
      }
    ")

  set(TRY_COMPILE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${LLVM_CXXFLAGS} -UNDEBUG")

  try_compile(_TRY_SUCCESS ${CMAKE_BINARY_DIR} "${_TEST_SOURCE}"
    CMAKE_FLAGS "-DINCLUDE_DIRECTORIES:STRING=${LLVM_INCLUDE_DIRS}"
    CMAKE_FLAGS "-DLINK_DIRECTORIES:STRING=${LLVM_LIBDIR}"
    LINK_LIBRARIES "${LLVM_LIBS} ${LLVM_SYSLIBS} ${LLVM_LDFLAGS}"
    COMPILE_DEFINITIONS ${TRY_COMPILE_CXX_FLAGS}
    OUTPUT_VARIABLE _TRY_COMPILE_OUTPUT
  )

  file(APPEND "${CMAKE_BINARY_DIR}/CMakeFiles/CMakeOutput.log"
    "Test -NDEBUG flag: ${_TRY_COMPILE_OUTPUT}\n")

  if(_TRY_SUCCESS)
    message(STATUS "DEBUG build")
  else()
    message(STATUS "Not a DEBUG build, adding -DNDEBUG explicitly")
    set(LLVM_CXXFLAGS "${LLVM_CXXFLAGS} -DNDEBUG")
  endif()

endif()

####################################################################

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
  set(_EMPTY_C_FILE "${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/tripletfind.c")
  file(WRITE "${_EMPTY_C_FILE}" "")

  execute_process(COMMAND ${CLANG} "${CLANG_TARGET_OPTION}${LLVM_HOST_TARGET}" -x c ${_EMPTY_C_FILE} -S -emit-llvm -o - RESULT_VARIABLE RES_VAR OUTPUT_VARIABLE OUTPUT_VAR)
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

if(NOT DEFINED LLC_HOST_CPU AND NOT CMAKE_CROSSCOMPILING)
  message(STATUS "Find out LLC host CPU with ${LLVM_LLC}")
  execute_process(COMMAND ${LLVM_LLC} "--version" RESULT_VARIABLE RES_VAR OUTPUT_VARIABLE OUTPUT_VAR)
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

if(LLC_HOST_CPU MATCHES "unknown")
  message(FATAL_ERROR "LLVM could not recognize your CPU model automatically. Please rerun cmake with -DLLC_HOST_CPU=<model> (to see a list of models, try: llc -mcpu help)")
endif()

set(LLC_HOST_CPU "${LLC_HOST_CPU}" CACHE STRING "The Host CPU to use with llc")

####################################################################

# This tests that we can actually link to the llvm libraries.
# Mostly to catch issues like #295 - cannot find -ledit

setup_cache_var_name(LLVM_LINK_TEST "LLVM_LINK_TEST-${LLVM_HOST_TARGET}-${CLANG}")

if(NOT DEFINED ${CACHE_VAR_NAME})

  set(LLVM_LINK_TEST_SOURCE "
    #include <stdio.h>
    #include \"llvm/IR/LLVMContext.h\"
    #include \"llvm/Support/SourceMgr.h\"
    #include \"llvm/IR/Module.h\"
    #include \"llvm/IRReader/IRReader.h\"

    int main( int argc, char* argv[] )
    {
       if( argc < 2 )
         exit(2);

       llvm::LLVMContext context;
       llvm::SMDiagnostic err;
       std::unique_ptr<llvm::Module> module = llvm::parseIRFile( argv[1], err, context );

       if( !module )
         exit(1);
       else
         module->dump();

       return 0;
    }")

  string(RANDOM RNDNAME)
  set(LLVM_LINK_TEST_FILENAME "${CMAKE_BINARY_DIR}/llvm_link_test_${RNDNAME}.cc")
  file(WRITE "${LLVM_LINK_TEST_FILENAME}" "${LLVM_LINK_TEST_SOURCE}")

  try_compile(LLVM_LINK_TEST ${CMAKE_BINARY_DIR} "${LLVM_LINK_TEST_FILENAME}"
              CMAKE_FLAGS "-DINCLUDE_DIRECTORIES:STRING=${LLVM_INCLUDE_DIRS}"
              CMAKE_FLAGS "-DLINK_DIRECTORIES:STRING=${LLVM_LIBDIR}"
              LINK_LIBRARIES "${LLVM_LDFLAGS} ${LLVM_LIBS} ${LLVM_SYSLIBS}"
              COMPILE_DEFINITIONS "${CMAKE_CXX_FLAGS} ${LLVM_CXXFLAGS}"
              OUTPUT_VARIABLE _TRY_COMPILE_OUTPUT)

  if (LLVM_LINK_TEST)
    message(STATUS "LLVM link test OK")
  else()
    message(STATUS "LLVM link test output: ${_TRY_COMPILE_OUTPUT}")
    message(FATAL_ERROR "LLVM link test FAILED. This mostly happens when your LLVM installation does not have all dependencies installed.")
  endif()

endif()

set_cache_var(LLVM_LINK_TEST "LLVM link test result")

####################################################################
#X86 has -march and -mcpu reversed, for clang

if(CMAKE_HOST_SYSTEM_PROCESSOR MATCHES "(powerpc|armv7|aarch64)")
  set(CLANG_MARCH_FLAG "-mcpu=")
else()
  set(CLANG_MARCH_FLAG "-march=")
endif()

####################################################################

if(NOT DEFINED ${CL_DISABLE_HALF})
  set(CL_DISABLE_HALF 0)
  message(STATUS "Checking fp16 support")
  custom_try_compile_c_cxx_silent("${CLANG}" "c" "__fp16 callfp16(__fp16 a) { return a * (__fp16)1.8; };" "__fp16 x=callfp16((__fp16)argc);" RESV -c ${CLANG_TARGET_OPTION}${LLC_TRIPLE} ${CLANG_MARCH_FLAG}${LLC_HOST_CPU})
  if(RESV)
    set(CL_DISABLE_HALF 1)
  endif()
endif()

set(CL_DISABLE_HALF "${CL_DISABLE_HALF}" CACHE BOOL "Disable cl_khr_fp16 because fp16 is not supported")
message(STATUS "fp16 disabled: ${CL_DISABLE_HALF}")

#####################################################################
