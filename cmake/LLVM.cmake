
#=============================================================================
#   CMake build system files for detecting Clang and LLVM
#
#   Copyright (c) 2014-2020 pocl developers
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
    NAMES
      "llvmtce-config"
      "llvm-config"
      "llvm-config-mp-19.0" "llvm-config-mp-19" "llvm-config-19" "llvm-config190"
      "llvm-config-mp-18.0" "llvm-config-mp-18" "llvm-config-18" "llvm-config180"
      "llvm-config-mp-17.0" "llvm-config-mp-17" "llvm-config-17" "llvm-config170"
      "llvm-config-mp-16.0" "llvm-config-mp-16" "llvm-config-16" "llvm-config160"
      "llvm-config-mp-15.0" "llvm-config-mp-15" "llvm-config-15" "llvm-config150"
      "llvm-config-mp-14.0" "llvm-config-mp-14" "llvm-config-14" "llvm-config140"
      "llvm-config"
    DOC "llvm-config executable")
endif()

set(WITH_LLVM_CONFIG "${WITH_LLVM_CONFIG}" CACHE PATH "Path to preferred llvm-config")

if(NOT LLVM_CONFIG)
  message(FATAL_ERROR "llvm-config not found !")
else()
  file(TO_CMAKE_PATH "${LLVM_CONFIG}" LLVM_CONFIG)
  message(STATUS "Using llvm-config: ${LLVM_CONFIG}")
  if(LLVM_CONFIG MATCHES "llvmtce-config${CMAKE_EXECUTABLE_SUFFIX}$")
    set(LLVM_BINARY_SUFFIX "")
  elseif(LLVM_CONFIG MATCHES "llvm-config${CMAKE_EXECUTABLE_SUFFIX}$")
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
    OUTPUT_VARIABLE LLVM_CONFIG_VALUE
    RESULT_VARIABLE LLVM_CONFIG_RETVAL
    OUTPUT_STRIP_TRAILING_WHITESPACE
  )
  if(LLVM_CONFIG_RETVAL)
    message(SEND_ERROR "Error running llvm-config with arguments: ${ARGN}")
  else()
    set(${VARIABLE_NAME} ${LLVM_CONFIG_VALUE} CACHE STRING "llvm-config's ${VARIABLE_NAME} value")
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

# required for sources..
if((LLVM_VERSION_MAJOR LESS 14) OR (LLVM_VERSION_MAJOR GREATER 19))
  message(FATAL_ERROR "LLVM version between 14.0 and 19.0 required, found: ${LLVM_VERSION_MAJOR}")
endif()

string(REPLACE "." ";" LLVM_VERSION_PARSED "${LLVM_VERSION}")
list(GET LLVM_VERSION_PARSED 0 LLVM_VERSION_MAJOR)
list(GET LLVM_VERSION_PARSED 1 LLVM_VERSION_MINOR)

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

if(LLVM_VERSION_MAJOR LESS 16)
  run_llvm_config(LLVM_SRC_ROOT --src-root)
  run_llvm_config(LLVM_OBJ_ROOT --obj-root)
endif()

string(REPLACE "${LLVM_PREFIX}" "${LLVM_PREFIX_CMAKE}" LLVM_OBJ_ROOT "${LLVM_OBJ_ROOT}")
run_llvm_config(LLVM_ALL_TARGETS --targets-built)
run_llvm_config(LLVM_HOST_TARGET --host-target)
run_llvm_config(LLVM_BUILD_MODE --build-mode)
run_llvm_config(LLVM_ASSERTS_BUILD --assertion-mode)

if(MSVC)
  string(REPLACE "-L${LLVM_LIBDIR}" "" LLVM_LDFLAGS "${LLVM_LDFLAGS}")
  string(STRIP "${LLVM_LDFLAGS}" LLVM_LDFLAGS)
endif()

if(LLVM_BUILD_MODE MATCHES "Debug")
  set(LLVM_BUILD_MODE_DEBUG 1)
else()
  set(LLVM_BUILD_MODE_DEBUG 0)
endif()

# A few work-arounds for llvm-config issues

# - pocl doesn't compile with '-pedantic'
#LLVM_CXX_FLAGS=$($LLVM_CONFIG --cxxflags | sed -e 's/ -pedantic / /g')
string(REPLACE " -pedantic" "" LLVM_CXXFLAGS "${LLVM_CXXFLAGS}")

#llvm-config clutters CXXFLAGS with a lot of -W<whatever> flags.
#(They are not needed - we want to use -Wall anyways)
#This is a problem if LLVM was built with a different compiler than we use here,
#and our compiler chokes on unrecognized command-line options.
string(REGEX REPLACE "-W[^ ]*" "" LLVM_CXXFLAGS "${LLVM_CXXFLAGS}")

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
if(WIN32 AND (NOT MINGW))
  set(LLVM_HOST_TARGET "x86_64-pc")
endif()

#############################################################

run_llvm_config(LLVM_HAS_RTTI --has-rtti)

if(STATIC_LLVM)
  set(LLVM_LIB_MODE --link-static)
else()
  set(LLVM_LIB_MODE --link-shared)
endif()

unset(LLVM_LIBS)
run_llvm_config(LLVM_LIBS --libs ${LLVM_LIB_MODE})
string(STRIP "${LLVM_LIBS}" LLVM_LIBS)
if(NOT LLVM_LIBS)
  message(FATAL_ERROR "llvm-config --libs did not return anything, perhaps wrong setting of STATIC_LLVM ?")
endif()
# Convert LLVM_LIBS from string -> list format to make handling them easier
separate_arguments(LLVM_LIBS)

# With Visual Studio llvm-config gives invalid list of static libs (libXXXX.a instead of XXXX.lib)
# we extract the pure names (LLVMLTO, LLVMMipsDesc etc) and let find_library do its job
foreach(LIBFLAG ${LLVM_LIBS})
  STRING(REGEX REPLACE "^-l(.*)$" "\\1" LIB_NAME ${LIBFLAG})
  list(APPEND LLVM_LIBNAMES "${LIB_NAME}")
endforeach()

foreach(LIBNAME ${LLVM_LIBNAMES})
  find_library(L_LIBFILE_${LIBNAME} NAMES "${LIBNAME}" HINTS "${LLVM_LIBDIR}")
  if(NOT L_LIBFILE_${LIBNAME})
    message(FATAL_ERROR "Could not find LLVM library ${LIBNAME}, perhaps wrong setting of STATIC_LLVM ?")
  endif()
  list(APPEND LLVM_LIBFILES "${L_LIBFILE_${LIBNAME}}")
endforeach()

set(POCL_LLVM_LIBS ${LLVM_LIBFILES})

####################################################################

# this needs to be done with LLVM_LIB_MODE because it affects the output
run_llvm_config(LLVM_SYSLIBS --system-libs ${LLVM_LIB_MODE})
string(STRIP "${LLVM_SYSLIBS}" LLVM_SYSLIBS)
# TODO this hack is required for MinGW. Without it,
# Clang link test reports missing symbols like GetFileVersionInfoSizeW
if(MINGW)
  set(LLVM_SYSLIBS "${LLVM_SYSLIBS} -lversion" CACHE STRING "llvm's syslibs" FORCE)
endif()

####################################################################

if(STATIC_LLVM)
  set(CLANG_LIBNAMES clangCodeGen clangFrontendTool clangFrontend clangDriver clangSerialization
      clangParse clangSema clangRewrite clangRewriteFrontend
      clangStaticAnalyzerFrontend clangStaticAnalyzerCheckers
      clangStaticAnalyzerCore clangAnalysis clangEdit clangAST clangASTMatchers clangLex clangBasic)
  if(LLVM_VERSION_MAJOR GREATER 14)
     list(APPEND CLANG_LIBNAMES clangSupport)
  endif()
  # must come after clangFrontend
  if(LLVM_VERSION_MAJOR GREATER 17)
     list(INSERT CLANG_LIBNAMES 4 clangAPINotes)
  endif()
else()
  # For non-static builds, link against a single shared library
  # instead of multiple component shared libraries.
  if("${LLVM_LIBNAMES}" MATCHES "LLVMTCE")
    set(CLANG_LIBNAMES clangTCE-cpp)
  else()
    set(CLANG_LIBNAMES clang-cpp)
  endif()
endif()

foreach(LIBNAME ${CLANG_LIBNAMES})
  list(APPEND CLANG_LIBS "-l${LIBNAME}")
  find_library(C_LIBFILE_${LIBNAME} NAMES "${LIBNAME}" HINTS "${LLVM_LIBDIR}")
  if(NOT C_LIBFILE_${LIBNAME})
    message(FATAL_ERROR "Could not find Clang library ${LIBNAME}, perhaps wrong setting of STATIC_LLVM ?")
  endif()
  list(APPEND CLANG_LIBFILES "${C_LIBFILE_${LIBNAME}}")
endforeach()

####################################################################

macro(find_program_or_die OUTPUT_VAR PROG_NAME DOCSTRING)
  find_program(${OUTPUT_VAR}
    NAMES "${PROG_NAME}${LLVM_BINARY_SUFFIX}${CMAKE_EXECUTABLE_SUFFIX}"
    HINTS "${LLVM_BINDIR}" "${LLVM_CONFIG_LOCATION}"
    DOC "${DOCSTRING}"
    NO_DEFAULT_PATH
    NO_CMAKE_PATH
    NO_CMAKE_ENVIRONMENT_PATH
  )
  if(${OUTPUT_VAR})
    message(STATUS "Found ${PROG_NAME}: ${${OUTPUT_VAR}}")
  else()
    message(FATAL_ERROR "${PROG_NAME} executable not found!")
  endif()
endmacro()

find_program_or_die( CLANG "clang" "clang binary")
execute_process(COMMAND "${CLANG}" "--version" OUTPUT_VARIABLE LLVM_CLANG_VERSION RESULT_VARIABLE CLANG_RES)
find_program_or_die( CLANGXX "clang++" "clang++ binary")
execute_process(COMMAND "${CLANGXX}" "--version" OUTPUT_VARIABLE LLVM_CLANGXX_VERSION RESULT_VARIABLE CLANGXX_RES)
if(CLANGXX_RES OR CLANG_RES)
  message(FATAL_ERROR "Failed running clang/clang++ --version")
endif()

find_program_or_die(LLVM_OPT  "opt"       "LLVM optimizer")
find_program_or_die(LLVM_LLC  "llc"       "LLVM static compiler")
find_program_or_die(LLVM_AS   "llvm-as"   "LLVM assembler")
find_program_or_die(LLVM_LINK "llvm-link" "LLVM IR linker")
find_program_or_die(LLVM_LLI  "lli"       "LLVM interpreter")

if(ENABLE_LLVM_FILECHECKS)
  if(IS_ABSOLUTE "${LLVM_FILECHECK_BIN}" AND EXISTS "${LLVM_FILECHECK_BIN}")
    message(STATUS "LLVM IR checks enabled using ${LLVM_FILECHECK_BIN}.")
  else()
    find_program_or_die(LLVM_FILECHECK_BIN "FileCheck" "LLVM FileCheck (not installed by default)")
  endif()
endif()

if(NOT DEFINED LLVM_SPIRV)
  find_program(LLVM_SPIRV NAMES "llvm-spirv${LLVM_BINARY_SUFFIX}${CMAKE_EXECUTABLE_SUFFIX}" "llvm-spirv-${LLVM_VERSION_MAJOR}${CMAKE_EXECUTABLE_SUFFIX}" "llvm-spirv${CMAKE_EXECUTABLE_SUFFIX}" HINTS "${LLVM_BINDIR}" "${LLVM_CONFIG_LOCATION}" "${LLVM_PREFIX}" "${LLVM_PREFIX_BIN}")
  if(LLVM_SPIRV)
    execute_process(
        COMMAND "${LLVM_SPIRV}" "--version"
        OUTPUT_VARIABLE LLVM_SPIRV_VERSION_VALUE
        RESULT_VARIABLE LLVM_SPIRV_VERSION_RETVAL
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    if(${LLVM_SPIRV_VERSION_RETVAL} EQUAL 0)
        string(REGEX MATCH "LLVM version ([0-9]*)" LLVM_SPIRV_VERSION_MATCH ${LLVM_SPIRV_VERSION_VALUE})
        if(NOT ${CMAKE_MATCH_1} EQUAL ${LLVM_VERSION_MAJOR})
          unset(LLVM_SPIRV)
        endif()
    else()
      unset(LLVM_SPIRV)
    endif()
  endif()
  if(LLVM_SPIRV)
    message(STATUS "Found llvm-spirv: ${LLVM_SPIRV}")
    set(HAVE_LLVM_SPIRV ON CACHE BOOL "llvm-spirv")
  else()
    message(STATUS "Did NOT find llvm-spirv!")
    set(HAVE_LLVM_SPIRV OFF CACHE BOOL "llvm-spirv")
  endif()
endif()

if(NOT DEFINED SPIRV_LINK)
  find_program(SPIRV_LINK NAMES "spirv-link${CMAKE_EXECUTABLE_SUFFIX}" HINTS "${LLVM_BINDIR}" "${LLVM_CONFIG_LOCATION}" "${LLVM_PREFIX}" "${LLVM_PREFIX_BIN}")
  if(SPIRV_LINK)
    message(STATUS "Found spirv-link: ${SPIRV_LINK}")
  else()
    message(STATUS "Did NOT find spirv-link!")
  endif()
endif()

if(NOT DEFINED HAVE_LLVM_SPIRV_LIB)
  find_path(LLVM_SPIRV_INCLUDEDIR "LLVMSPIRVLib.h" PATHS "${LLVM_INCLUDEDIR}/LLVMSPIRVLib" NO_DEFAULT_PATH)
  find_library(LLVM_SPIRV_LIB "LLVMSPIRVLib" PATHS "${LLVM_LIBDIR}" NO_DEFAULT_PATH)
  if(LLVM_SPIRV_INCLUDEDIR AND LLVM_SPIRV_LIB)
    set(HAVE_LLVM_SPIRV_LIB 1 CACHE BOOL "have LLVMSPIRVLib")
  else()
    set(HAVE_LLVM_SPIRV_LIB 0 CACHE BOOL "have LLVMSPIRVLib")
  endif()
endif()

####################################################################

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
  custom_try_compile_c_cxx("${CLANGXX}" "cc" "${SOURCE1}" "${SOURCE2}" ${RES_VAR} "-o" "${RANDOM_FILENAME}" "-c" ${ARGN})
  file(REMOVE "${RANDOM_FILENAME}")
endmacro()

# clang++ try-compile macro
macro(custom_try_compile_clang SOURCE1 SOURCE2 RES_VAR)
  string(RANDOM RNDNAME)
  set(RANDOM_FILENAME "${CMAKE_BINARY_DIR}/compile_test_${RNDNAME}.o")
  custom_try_compile_c_cxx("${CLANG}" "c" "${SOURCE1}" "${SOURCE2}" ${RES_VAR} "-o" "${RANDOM_FILENAME}" "-c" ${ARGN})
  file(REMOVE "${RANDOM_FILENAME}")
endmacro()

# clang++ try-compile macro
macro(custom_try_compile_clang_silent SOURCE1 SOURCE2 RES_VAR)
  string(RANDOM RNDNAME)
  set(RANDOM_FILENAME "${CMAKE_BINARY_DIR}/compile_test_${RNDNAME}.o")
  custom_try_compile_c_cxx_silent("${CLANG}" "c" "${SOURCE1}" "${SOURCE2}" ${RES_VAR} "-o" "${RANDOM_FILENAME}" "-c" ${ARGN})
  file(REMOVE "${RANDOM_FILENAME}")
endmacro()

# clang++ try-link macro
macro(custom_try_link_clang SOURCE1 SOURCE2 RES_VAR)
  string(RANDOM RNDNAME)
  if(WIN32)
    set(RANDOM_FILENAME "${CMAKE_BINARY_DIR}/compile_test_${RNDNAME}.exe")
  else()
    set(RANDOM_FILENAME "${CMAKE_BINARY_DIR}/compile_test_${RNDNAME}")
  endif()
  custom_try_compile_c_cxx_silent("${CLANG}" "c" "${SOURCE1}" "${SOURCE2}" ${RES_VAR} "-o" "${RANDOM_FILENAME}" ${ARGN})
  file(REMOVE "${RANDOM_FILENAME}")
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
    if(NOT ${RESV} EQUAL 0)
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
  string(RANDOM RNDNAME)
  set(OUTF "${CMAKE_BINARY_DIR}/lli_test_${RNDNAME}.bc")
  custom_try_compile_c_cxx("${CLANG}" "c" "${SOURCE1}" "${SOURCE2}" RESV "-o" "${OUTF}" "-x" "c" "-emit-llvm" "-c" ${ARGN})
  set(${OUTPUT_VAR} "")
  set(${RES_VAR} "")
  if(RESV OR (NOT EXISTS "${OUTF}"))
    message(STATUS " ########## Compilation failed")
  else()
    execute_process(COMMAND "${LLVM_LLI}" "-force-interpreter" "${OUTF}" RESULT_VARIABLE RESV OUTPUT_VARIABLE ${OUTPUT_VAR} ERROR_VARIABLE EV)
    set(${RES_VAR} ${RESV})
    file(REMOVE "${OUTF}")
    if((NOT ${RESV} EQUAL 0) AND (NOT ${SILENT}))
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
####################################################################

if(NOT DEFINED CLANG_NEEDS_RTLIB)

  set(RT128 OFF)
  set(RT64 OFF)
  set(NEEDS_RTLIB_FLAG OFF)

  # on 32bit systems, we need 64bit emulation
  if(CMAKE_SIZEOF_VOID_P EQUAL 4)
    set(INC "#include <stdint.h>\n#include <stddef.h>")
    set(SRC "int64_t a = argc; int64_t b = argc-1; int64_t c = a / b; return (int)c; ")
    custom_try_link_clang("${INC}" "${SRC}" RES)
    if(NOT RES)
      message(STATUS "64bit division compiles without extra flags")
      set(RT64 ON)
    else()
      custom_try_link_clang("${INC}" "${SRC}" RES "--rtlib=compiler-rt")
      if(NOT RES)
        message(STATUS "64bit division compiles WITH --rtlib=compiler-rt")
        set(NEEDS_RTLIB_FLAG ON)
        set(RT64 ON)
      else()
        message(WARNING "64bit division doesn't compile at all!")
      endif()
    endif()

  else()

    set(RT64 ON)
    # on 64bit systems, we need 128bit integers for Errol
    set(INC "extern __uint128_t __udivmodti4(__uint128_t a, __uint128_t b, __uint128_t* rem);")
    set(SRC "__uint128_t low, mid, tmp1, pow19 = (__uint128_t)1000000000; mid = __udivmodti4(low, pow19, &tmp1); return 0;")
    custom_try_link_clang("${INC}" "${SRC}" RES)

    if(NOT RES)
      message(STATUS "udivmodti4 compiles without extra flags")
      set(RT128 ON)
    else()
      custom_try_link_clang("${INC}" "${SRC}" RES "--rtlib=compiler-rt")
      if(NOT RES)
        message(STATUS "udivmodti4 compiles WITH --rtlib=compiler-rt")
        set(NEEDS_RTLIB_FLAG ON)
        set(RT128 ON)
      else()
        message(WARNING "udivmodti4 doesn't compile at all!")
      endif()
    endif()

  endif()

  set(CLANG_HAS_64B_MATH ${RT64} CACHE INTERNAL "Clang's available with 64bit math")
  set(CLANG_HAS_128B_MATH ${RT128} CACHE INTERNAL "Clang's available with 128bit math")
  set(CLANG_NEEDS_RTLIB ${NEEDS_RTLIB_FLAG} CACHE INTERNAL "Clang needs extra --rtlib flag for compiler-rt math")

endif()

####################################################################

if(ENABLE_HOST_CPU_DEVICES)

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

if(NOT DEFINED LLC_TRIPLE)
  message(STATUS "Find out LLC target triple (for host ${LLVM_HOST_TARGET})")
  set(_EMPTY_C_FILE "${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/tripletfind.c")
  file(WRITE "${_EMPTY_C_FILE}" "")

  execute_process(COMMAND ${CLANG} "--target=${LLVM_HOST_TARGET}" -x c ${_EMPTY_C_FILE} -S -emit-llvm -o - RESULT_VARIABLE RES_VAR OUTPUT_VARIABLE OUTPUT_VAR)
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

  set(LLC_TRIPLE "${LLC_TRIPLE}" CACHE INTERNAL "LLC_TRIPLE")
endif()

# FIXME: The cpu name printed by llc --version is the same cpu that will be
# targeted if you pass -mcpu=native to llc, so we could replace this auto-detection
# with just: set(LLC_HOST_CPU "native"), however, we can't do this at the moment
# because of the work-around for arm1176jz-s.
if(NOT DEFINED LLC_HOST_CPU_AUTO)
  message(STATUS "Find out LLC host CPU with ${LLVM_LLC}")
  execute_process(COMMAND ${LLVM_LLC} "--version" RESULT_VARIABLE RES_VAR OUTPUT_VARIABLE OUTPUT_VAR)
  if(RES_VAR)
    message(FATAL_ERROR "Error ${RES_VAR} while determining LLC host CPU")
  endif()

  if(OUTPUT_VAR MATCHES "Host CPU: ([^ ]*)")
    # sigh... STRING(STRIP is to workaround regexp bug in cmake
    string(STRIP "${CMAKE_MATCH_1}" LLC_HOST_CPU_AUTO)
  else()
    message(FATAL_ERROR "Couldnt determine host CPU from llc output")
  endif()

  #TODO better
  if(CMAKE_LIBRARY_ARCHITECTURE MATCHES "gnueabihf" AND LLC_HOST_CPU_AUTO MATCHES "arm1176jz-s")
    set(LLC_HOST_CPU_AUTO "arm1176jzf-s")
  endif()
endif()

if((LLC_HOST_CPU_AUTO MATCHES "unknown") AND (NOT LLC_HOST_CPU))
  message(FATAL_ERROR "LLVM could not recognize your CPU model automatically. Please run CMake with -DLLC_HOST_CPU=<cpu> (you can find valid names with: llc -mcpu=help)")
else()
  set(LLC_HOST_CPU_AUTO "${LLC_HOST_CPU_AUTO}" CACHE INTERNAL "Autodetected CPU")
endif()

if(DEFINED LLC_HOST_CPU)
  if(NOT LLC_HOST_CPU STREQUAL LLC_HOST_CPU_AUTO)
    message(STATUS "Autodetected CPU ${LLC_HOST_CPU_AUTO} overridden by user to ${LLC_HOST_CPU}")
  endif()
  set(SELECTED_HOST_CPU "${LLC_HOST_CPU}")
  set(HOST_CPU_FORCED 1 CACHE INTERNAL "CPU is forced by user")
else()
  set(SELECTED_HOST_CPU "${LLC_HOST_CPU_AUTO}")
  set(HOST_CPU_FORCED 0 CACHE INTERNAL "CPU is forced by user")
endif()


# Some architectures have -march and -mcpu reversed
if(NOT DEFINED CLANG_MARCH_FLAG)
  message(STATUS "Checking clang -march vs. -mcpu flag")
  custom_try_compile_clang_silent("" "return 0;" RES --target=${LLC_TRIPLE} -march=${SELECTED_HOST_CPU})
  if(NOT RES)
    set(CLANG_MARCH_FLAG "-march=")
  else()
    custom_try_compile_clang_silent("" "return 0;" RES --target=${LLC_TRIPLE} -mcpu=${SELECTED_HOST_CPU})
    if(NOT RES)
      set(CLANG_MARCH_FLAG "-mcpu=")
    else()
      message(FATAL_ERROR "Could not determine whether to use -march or -mcpu with clang")
    endif()
  endif()
  message(STATUS "  Using ${CLANG_MARCH_FLAG}")

  set(CLANG_MARCH_FLAG ${CLANG_MARCH_FLAG} CACHE INTERNAL "Clang option used to specify the target cpu")
endif()

if(NOT DEFINED ALIGNOF_DOUBLE16)

  custom_try_run_lli(TRUE "
#ifndef offsetof
#define offsetof(type, member) ((char *) &((type *) 0)->member - (char *) 0)
#endif

  typedef double double16  __attribute__((__ext_vector_type__(16)));"

  "typedef struct { char x; double16 y; } ac__type_alignof_;
  int r = offsetof(ac__type_alignof_, y);
  return r;"
  SIZEOF_STDOUT RESULT "--target=${LLC_TRIPLE}")

  if(NOT ${RESULT})
    message(WARNING "Could not determine align of(double16)")
    set(ALIGNOF_DOUBLE16 "128" CACHE INTERNAL "Align of double16")
  else()
    set(ALIGNOF_DOUBLE16 "${RESULT}" CACHE INTERNAL "Align of double16")
  endif()

endif()

endif(ENABLE_HOST_CPU_DEVICES)

####################################################################

# This tests that we can actually link to the llvm libraries.
# Mostly to catch issues like #295 - cannot find -ledit

if(NOT LLVM_LINK_TEST)

  message(STATUS "Running LLVM link test")

  set(LLVM_LINK_TEST_FILENAME "${CMAKE_SOURCE_DIR}/cmake/LinkTestLLVM.cc")

  try_compile(LLVM_LINK_TEST ${CMAKE_BINARY_DIR} "${LLVM_LINK_TEST_FILENAME}"
              CMAKE_FLAGS "-DINCLUDE_DIRECTORIES:STRING=${LLVM_INCLUDE_DIRS}"
              CMAKE_FLAGS "-DLINK_DIRECTORIES:STRING=${LLVM_LIBDIR}"
              LINK_LIBRARIES "${LLVM_LDFLAGS} ${LLVM_LIBS} ${LLVM_SYSLIBS}"
              COMPILE_DEFINITIONS "${CMAKE_CXX_FLAGS} ${LLVM_CXXFLAGS}"
              OUTPUT_VARIABLE _TRY_COMPILE_OUTPUT)

  if (LLVM_LINK_TEST)
    message(STATUS "LLVM link test OK")
    set(LLVM_LINK_TEST 1 CACHE INTERNAL "LLVM link test result")
  else()
    message(STATUS "LLVM link test output: ${_TRY_COMPILE_OUTPUT}")
    message(FATAL_ERROR "LLVM link test FAILED. This mostly happens when your LLVM installation does not have all dependencies installed.")
  endif()

endif()

####################################################################

# This tests that we can actually link to the Clang libraries.

if(NOT CLANG_LINK_TEST)

  message(STATUS "Running Clang link test")

  set(CLANG_LINK_TEST_FILENAME "${CMAKE_SOURCE_DIR}/cmake/LinkTestClang.cc")

  try_compile(CLANG_LINK_TEST ${CMAKE_BINARY_DIR} "${CLANG_LINK_TEST_FILENAME}"
              CMAKE_FLAGS "-DINCLUDE_DIRECTORIES:STRING=${LLVM_INCLUDE_DIRS}"
              CMAKE_FLAGS "-DLINK_DIRECTORIES:STRING=${LLVM_LIBDIR}"
              LINK_LIBRARIES "${LLVM_LDFLAGS} ${CLANG_LIBS} ${LLVM_LIBS} ${LLVM_SYSLIBS}"
              COMPILE_DEFINITIONS "${CMAKE_CXX_FLAGS} ${LLVM_CXXFLAGS} -DLLVM_MAJOR=${LLVM_VERSION_MAJOR}"
              OUTPUT_VARIABLE _TRY_COMPILE_OUTPUT)

  if(CLANG_LINK_TEST)
    message(STATUS "Clang link test OK")
    set(CLANG_LINK_TEST 1 CACHE INTERNAL "Clang link test result")
  else()
    message(STATUS "Clang link test output: ${_TRY_COMPILE_OUTPUT}")
    message(FATAL_ERROR "Clang link test FAILED. This mostly happens when your Clang installation does not have all dependencies and/or headers installed.")
  endif()

endif()


####################################################################

# Clang documentation on Language Extensions:
# __fp16 is supported on every target, as it is purely a storage format
# _Float16 is currently only supported on the following targets... SPIR, x86
# Limitations:
#     The _Float16 type requires SSE2 feature and above due to the instruction
#        limitations. When using it on i386 targets, you need to specify -msse2
#        explicitly.
#     For targets without F16C feature or above, please make sure:
#     Use GCC 12.0 and above if you are using libgcc.
#     If you are using compiler-rt, use the same version with the compiler.
#        Early versions provided FP16 builtins in a different ABI. A workaround is
#        to use a small code snippet to check the ABI if you cannot make sure of it.

if(ENABLE_HOST_CPU_DEVICES AND NOT DEFINED HOST_CPU_SUPPORTS_FLOAT16)
  set(HOST_CPU_SUPPORTS_FLOAT16 0)
  message(STATUS "Checking host support for _Float16 type")
    custom_try_compile_clang_silent("_Float16 callfp16(_Float16 a) { return a * 1.8f16; };" "_Float16 x=callfp16((_Float16)argc);"
    RESV --target=${LLC_TRIPLE} ${CLANG_MARCH_FLAG}${SELECTED_HOST_CPU})
  if(RESV EQUAL 0)
    set(HOST_CPU_SUPPORTS_FLOAT16 1)
  endif()
endif()

####################################################################

# TODO we should check double support of the target somehow (excluding emulation),
# for now just provide an option
if(ENABLE_HOST_CPU_DEVICES AND NOT DEFINED HOST_CPU_SUPPORTS_DOUBLE)
  if(X86)
    set(HOST_CPU_SUPPORTS_DOUBLE ON CACHE INTERNAL "FP64, always enabled on X86(-64)" FORCE)
  else()
    option(HOST_CPU_SUPPORTS_DOUBLE "Enable FP64 support for Host CPU device" ON)
  endif()
endif()

#####################################################################

execute_process(COMMAND "${CLANG}" "--print-resource-dir" OUTPUT_VARIABLE RESOURCE_DIR)
string(STRIP "${RESOURCE_DIR}" RESOURCE_DIR)
set(CLANG_RESOURCE_DIR "${RESOURCE_DIR}" CACHE INTERNAL "Clang resource dir")

