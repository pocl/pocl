
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

include(LLVMHelpers)

# if the user provided CMake package root, use only that (NO_DEFAULT_PATH)
if(DEFINED LLVM_DIR OR DEFINED ENV{LLVM_DIR})
  find_package(LLVM REQUIRED CONFIG NO_DEFAULT_PATH HINTS "${LLVM_DIR}" ENV{LLVM_DIR})
  if((LLVM_VERSION_MAJOR LESS 17) OR (LLVM_VERSION_MAJOR GREATER 21))
    message(FATAL_ERROR "LLVM version between 17.0 and 21.0 required, found: ${LLVM_VERSION_MAJOR}")
  endif()
endif()

# if user provided llvm-config, use the preferred version
if(DEFINED WITH_LLVM_CONFIG AND WITH_LLVM_CONFIG)
  if(IS_ABSOLUTE "${WITH_LLVM_CONFIG}")
    if(EXISTS "${WITH_LLVM_CONFIG}")
      set(LLVM_CONFIG_BIN "${WITH_LLVM_CONFIG}" CACHE PATH "path of llvm-config")
    endif()
  else()
    find_program(LLVM_CONFIG_BIN NAMES "${WITH_LLVM_CONFIG}")
  endif()
endif()

# fallback search for LLVMConfig.cmake of supported versions in descending order
if(NOT LLVM_CONFIG_BIN AND NOT LLVM_PACKAGE_VERSION)
  if(NOT MSVC)
  find_package(LLVM 21.1.0...<21.2 CONFIG)
  if(NOT LLVM_FOUND)
    find_package(LLVM 20.1.0...<20.2 CONFIG)
  endif()
  if(NOT LLVM_FOUND)
    find_package(LLVM 19.1.0...<19.2 CONFIG)
  endif()
  if(NOT LLVM_FOUND)
    find_package(LLVM 18.1.0...<18.2 CONFIG)
  endif()
  if(NOT LLVM_FOUND)
    find_package(LLVM 17.0.0...<17.1 CONFIG)
  endif()
  endif()
  # at last, fallback to finding any llvm-config executable
  if(NOT LLVM_FOUND AND NOT LLVM_CONFIG_BIN)
    find_program(LLVM_CONFIG_BIN
      NAMES
        "llvmtce-config"
        "llvm-config"
        "llvm-config-mp-21.0" "llvm-config-mp-21" "llvm-config-21" "llvm-config210"
        "llvm-config-mp-20.0" "llvm-config-mp-20" "llvm-config-20" "llvm-config200"
        "llvm-config-mp-19.0" "llvm-config-mp-19" "llvm-config-19" "llvm-config190"
        "llvm-config-mp-18.0" "llvm-config-mp-18" "llvm-config-18" "llvm-config180"
        "llvm-config-mp-17.0" "llvm-config-mp-17" "llvm-config-17" "llvm-config170"
        "llvm-config"
      DOC "llvm-config executable")
  endif()
endif()

if(NOT LLVM_CONFIG_BIN AND NOT LLVM_PACKAGE_VERSION)
  message(FATAL_ERROR "Could not find either llvm-config or LLVMConfig.cmake !")
endif()

if(LLVM_CONFIG_BIN)
  if ((NOT IS_ABSOLUTE "${LLVM_CONFIG_BIN}") OR (NOT EXISTS "${LLVM_CONFIG_BIN}"))
    message(FATAL_ERROR "Found LLVM_CONFIG ${LLVM_CONFIG_BIN} but it isn't a valid executable")
  endif()

  # get the LLVM version of the llvm-config executable
  file(TO_CMAKE_PATH "${LLVM_CONFIG_BIN}" LLVM_CONFIG_BIN)
  message(STATUS "Using llvm-config: ${LLVM_CONFIG_BIN}")
  if(LLVM_CONFIG_BIN MATCHES "llvmtce-config${CMAKE_EXECUTABLE_SUFFIX}$")
    set(LLVM_BINARY_SUFFIX "")
  elseif(LLVM_CONFIG_BIN MATCHES "llvm-config${CMAKE_EXECUTABLE_SUFFIX}$")
    set(LLVM_BINARY_SUFFIX "")
  elseif(LLVM_CONFIG_BIN MATCHES "llvm-config(.*)${CMAKE_EXECUTABLE_SUFFIX}$")
    set(LLVM_BINARY_SUFFIX "${CMAKE_MATCH_1}")
  else()
    message(WARNING "Cannot determine llvm binary suffix from ${LLVM_CONFIG_BIN}")
  endif()
  message(STATUS "LLVM binaries suffix : ${LLVM_BINARY_SUFFIX}")

  if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.19)
    # Convert to real path only here, after detecting the binary suffix.
    # This is necessary when symlinked to a non-suffix binary, like
    # with the LLVM's Debian/Ubuntu packages.
    # /usr/bin/llvm-config-21 -> ../lib/llvm-21/bin/llvm-config
    file(REAL_PATH "${LLVM_CONFIG_BIN}"  LLVM_CONFIG_BIN)
  endif()

  get_filename_component(LLVM_CONFIG_LOCATION "${LLVM_CONFIG_BIN}" DIRECTORY)

  run_llvm_config(LLVM_VERSION_FULL --version)
  message(STATUS "llvm-config's LLVM_VERSION_FULL: ${LLVM_VERSION_FULL}")

  # turn to cmake list (18.1.3 -> 18;1;3)
  string(REPLACE "." ";" LLVM_VERSION_PARSED "${LLVM_VERSION_FULL}")

  # check that versions of CMake package & the llvm-config.exe binary agree
  if(LLVM_PACKAGE_VERSION)
    message(STATUS "LLVM Package's LLVM_CMAKE_DIR: ${LLVM_CMAKE_DIR}")
    list(GET LLVM_VERSION_PARSED 0 LLVM_VERSION_MAJOR_BINARY)
    list(GET LLVM_VERSION_PARSED 1 LLVM_VERSION_MINOR_BINARY)
    if(NOT ((LLVM_VERSION_MAJOR EQUAL LLVM_VERSION_MAJOR_BINARY)
       AND (LLVM_VERSION_MINOR EQUAL LLVM_VERSION_MINOR_BINARY)))
     message(FATAL_ERROR "Mismatch between versions of LLVM package and LLVM-config binary!")
    endif()
  else()
    list(GET LLVM_VERSION_PARSED 0 LLVM_VERSION_MAJOR)
    list(GET LLVM_VERSION_PARSED 1 LLVM_VERSION_MINOR)
  endif()
endif()

if(NOT LLVM_VERSION_MAJOR)
  message(FATAL_ERROR "LLVM version unknown")
endif()

if(CMAKE_CROSSCOMPILING)
  if(NOT LLVM_PACKAGE_VERSION OR NOT LLVM_CONFIG_BIN)
  message(FATAL_ERROR "Cross-compiling with LLVM is only supported if both LLVM_DIR\
  and WITH_LLVM_CONFIG variables are provided; WITH_LLVM_CONFIG must point to a Host-side\
  llvm-config executable, and LLVM_DIR must point to a <...llvm/lib/cmake>\
  directory in CMAKE_SYSROOT (Target-side LLVM)")
  endif()
else()
  if(LLVM_PACKAGE_VERSION AND LLVM_CONFIG_BIN)
    message(WARNING "Both LLVM package and LLVM-config binary found; will use the package")
  endif()
endif()

############################################################################

# Prefer the CMake setup when possible, fallback to llvm-config
# This ensures that the LLVM CMake variables are always correctly setup
# for target system
if(LLVM_PACKAGE_VERSION)
  message(STATUS "Setting up LLVM using LLVMConfig.cmake")

  # setup via CMake package
  include(SetupLLVMviaCMake)
else()
  message(STATUS "Setting up LLVM using llvm-config ${CMAKE_EXECUTABLE_SUFFIX}")
  message(STATUS "Using llvm-config: ${LLVM_CONFIG_BIN}")

  # setup via llvm-config binary
  include(SetupLLVMviaBinary)
endif()

############################################################################

if(LLVM_BUILD_MODE MATCHES "Debug")
  set(LLVM_BUILD_MODE_DEBUG 1)
else()
  set(LLVM_BUILD_MODE_DEBUG 0)
endif()

# When cross-compiling, we need some LLVM binaries both from Host and Target,
# since the some of the tools are needed on the Target machine,
# and some are needed on the Host to build the lib/kernel library.
#
# find the Host binaries using LLVM_CONFIG_LOCATION,
# and the Target binaries using LLVM_BINDIR

# find the Host programs first
if(LLVM_CONFIG_LOCATION)
  set(SEARCH_LOCATION ${LLVM_CONFIG_LOCATION})
else()
  set(SEARCH_LOCATION ${LLVM_BINDIR})
endif()
message(STATUS "Searching for Host LLVM binaries in ${SEARCH_LOCATION}")
find_llvm_program_or_die(HOST_CLANG     "clang"    "${SEARCH_LOCATION}"  "Host clang binary")
find_llvm_program_or_die(HOST_CLANGXX   "clang++"  "${SEARCH_LOCATION}"  "Host clang++ binary")
find_llvm_program_or_die(HOST_LLVM_OPT  "opt"      "${SEARCH_LOCATION}"  "Host LLVM optimizer")
find_llvm_program_or_die(HOST_LLVM_LLC  "llc"      "${SEARCH_LOCATION}"  "Host LLVM static compiler")
find_llvm_program_or_die(HOST_LLVM_AS   "llvm-as"  "${SEARCH_LOCATION}"  "Host LLVM assembler")
find_llvm_program_or_die(HOST_LLVM_DIS  "llvm-dis" "${SEARCH_LOCATION}"  "Host LLVM disassembler")
find_llvm_program_or_die(HOST_LLVM_LINK "llvm-link" "${SEARCH_LOCATION}" "Host LLVM IR linker")
find_llvm_program(HOST_LLVM_SPIRV "llvm-spirv" "${SEARCH_LOCATION}"      "Host LLVM spirv translator")
find_llvm_program(HOST_SPIRV_LINK "spirv-link" "${SEARCH_LOCATION}"      "Host spirv-link linker")
find_llvm_program(HOST_LLVM_FILECHECK "FileCheck" "${SEARCH_LOCATION}"   "Host LLVM Filecheck")


if(CMAKE_CROSSCOMPILING)
  message(STATUS "Searching for Target LLVM binaries in ${LLVM_BINDIR}")
  # TODO how to handle LLVM_BINARY_SUFFIX ?
  set(SAVED_LLVM_BINARY_SUFFIX ${LLVM_BINARY_SUFFIX})
  unset(LLVM_BINARY_SUFFIX)
  # must unset the ignore path to find binaries in the sysroot
  set(SAVED_CMAKE_SYSTEM_IGNORE_PATH ${CMAKE_SYSTEM_IGNORE_PATH})
  unset(CMAKE_SYSTEM_IGNORE_PATH)
  find_llvm_program_or_die(TARGET_CLANG     "clang"    "${LLVM_BINDIR}"  "Target clang binary")
  find_llvm_program_or_die(TARGET_CLANGXX   "clang++"  "${LLVM_BINDIR}"  "Target clang++ binary")
  find_llvm_program_or_die(TARGET_LLVM_OPT  "opt"      "${LLVM_BINDIR}"  "Target LLVM optimizer")
  find_llvm_program_or_die(TARGET_LLVM_LLC  "llc"      "${LLVM_BINDIR}"  "Target LLVM static compiler")
  find_llvm_program_or_die(TARGET_LLVM_AS   "llvm-as"  "${LLVM_BINDIR}"  "Target LLVM assembler")
  find_llvm_program_or_die(TARGET_LLVM_DIS  "llvm-dis" "${LLVM_BINDIR}"  "Target LLVM disassembler")
  find_llvm_program_or_die(TARGET_LLVM_LINK "llvm-link" "${LLVM_BINDIR}" "Target LLVM IR linker")
  find_llvm_program(TARGET_LLVM_SPIRV "llvm-spirv" "${LLVM_BINDIR}"      "Target LLVM spirv translator")
  find_llvm_program(TARGET_SPIRV_LINK "spirv-link" "${LLVM_BINDIR}"      "Target spirv-link linker")
  find_llvm_program(TARGET_LLVM_FILECHECK "FileCheck" "${LLVM_BINDIR}"   "Target LLVM Filecheck")
  # reset to original
  set(CMAKE_SYSTEM_IGNORE_PATH ${SAVED_CMAKE_SYSTEM_IGNORE_PATH})
  set(LLVM_BINARY_SUFFIX ${SAVED_LLVM_BINARY_SUFFIX})
  foreach(ITEM IN ITEMS TARGET_CLANG TARGET_CLANGXX TARGET_LLVM_OPT TARGET_LLVM_LLC TARGET_LLVM_AS
          TARGET_LLVM_DIS TARGET_LLVM_LINK TARGET_LLVM_SPIRV TARGET_SPIRV_LINK TARGET_LLVM_FILECHECK)
    remove_prefix_from_filepath(${ITEM})
  endforeach()
else()
  # if not cross-compiling these are identical
  set(TARGET_CLANG ${HOST_CLANG})
  set(TARGET_CLANGXX ${HOST_CLANGXX})
  set(TARGET_LLVM_OPT ${HOST_LLVM_OPT})
  set(TARGET_LLVM_LLC ${HOST_LLVM_LLC})
  set(TARGET_LLVM_AS ${HOST_LLVM_AS})
  set(TARGET_LLVM_DIS ${HOST_LLVM_DIS})
  set(TARGET_LLVM_LINK ${HOST_LLVM_LINK})
  set(TARGET_LLVM_SPIRV ${HOST_LLVM_SPIRV})
  set(TARGET_SPIRV_LINK ${HOST_SPIRV_LINK})
  set(TARGET_LLVM_FILECHECK ${HOST_LLVM_FILECHECK})
endif()

############################################################################

if(ENABLE_LLVM_FILECHECKS)
  if(NOT TARGET_LLVM_FILECHECK)
    message(STATUS "LLVM IR checks not enabled, FileCheck not found.")
    set(ENABLE_LLVM_FILECHECKS OFF)
  endif()
  if(NOT TARGET_LLVM_DIS)
    message(STATUS "LLVM IR checks not enabled, llvm-dis not found.")
    set(ENABLE_LLVM_FILECHECKS OFF)
  endif()
  if(ENABLE_LLVM_FILECHECKS)
    message(STATUS "LLVM IR checks enabled.")
  endif()
endif()

############################################################################

include(SetupLLVMSPIRV)

if(ENABLE_HOST_CPU_DEVICES)
  include(SetupLLVMHostCPU)
endif()

####################################################################

# To catch issue #2041. Determine if kernel compiler LLVM does not have sysroot set.
# Simple include should trigger this. See: https://gitlab.kitware.com/cmake/cmake/-/issues/26863
if(APPLE)
  message(STATUS "Checking LLVM configuration")
  custom_try_compile_clang_silent("#include <stdio.h>" "printf(\"Hello World!\");return 0;" RES --target=${LLC_TRIPLE})
  if(RES)
    message(FATAL_ERROR "LLVM/Clang failed to compile. Maybe an issue with LLVM sysroot? See: https://gitlab.kitware.com/cmake/cmake/-/issues/26863")
  endif()
  message(STATUS "LLVM configuration OK")
endif()

####################################################################

#TODO finish the package version
# From try_compile help:
#    LINK_LIBRARIES <libs>...
#    Specify libraries to be linked in the generated project.
#    The list of libraries may refer to system libraries and to
#    Imported Targets from the calling project.
#
# ... this though doesn't work.
# https://discourse.cmake.org/t/try-compile-try-run-link-against-target/7189/8

# skip the check, if using Package / cross-compiling; unfortunately
# simply using a CMake Target (provided by LLVM's CMake files) in
# LINK_LIBRARIES does not work, we would have to extract flags etc manually
if(NOT LLVM_PACKAGE_VERSION)

# This tests that we can actually link to the llvm libraries.
# Mostly to catch issues like #295 - cannot find -ledit
if(NOT LLVM_LINK_TEST_SUCCESSFUL)
  set(LLVM_LINK_TEST_FILENAME "${CMAKE_SOURCE_DIR}/cmake/LinkTestLLVM.cc")
  try_compile(LLVM_LINK_TEST ${CMAKE_BINARY_DIR} "${LLVM_LINK_TEST_FILENAME}"
              CMAKE_FLAGS "-DINCLUDE_DIRECTORIES:STRING=${LLVM_INCLUDE_DIRS}"
              CMAKE_FLAGS "-DLINK_DIRECTORIES:STRING=${LLVM_LIBDIR}"
              LINK_LIBRARIES "${LLVM_LDFLAGS}" "${LLVM_LINK_LIBRARIES}" "${LLVM_SYSLIBS}"
              COMPILE_DEFINITIONS "${CMAKE_CXX_FLAGS} ${LLVM_CXXFLAGS}"
              OUTPUT_VARIABLE _TRY_COMPILE_OUTPUT)
  if(LLVM_LINK_TEST)
    message(STATUS "LLVM link test OK")
    set(LLVM_LINK_TEST_SUCCESSFUL 1 CACHE INTERNAL "LLVM link test result")
  else()
    message(STATUS "LLVM link test output: ${_TRY_COMPILE_OUTPUT}")
    message(FATAL_ERROR "LLVM link test FAILED. This mostly happens when your LLVM installation does not have all dependencies installed.")
  endif()
endif()

# This tests that we can actually link to the Clang libraries.

if(NOT CLANG_LINK_TEST_SUCCESSFUL)
  message(STATUS "Running Clang link test")
  set(CLANG_LINK_TEST_FILENAME "${CMAKE_SOURCE_DIR}/cmake/LinkTestClang.cc")

  set(CXX_COMPAT_FLAGS "")
  if (MSVC)
    set(CXX_COMPAT_FLAGS "/Zc:preprocessor")
  endif()

  set(CLT_LINK_DIRS ${LLVM_LIBDIR})
  if(CLANG_LINK_DIRS AND (NOT CLANG_LINK_DIRS STREQUAL LLVM_LIBDIR))
    list(APPEND CLT_LINK_DIRS ${CLANG_LINK_DIRS})
  endif()

  try_compile(CLANG_LINK_TEST ${CMAKE_BINARY_DIR} "${CLANG_LINK_TEST_FILENAME}"
              CMAKE_FLAGS "-DINCLUDE_DIRECTORIES:STRING=${LLVM_INCLUDE_DIRS}"
              CMAKE_FLAGS "-DLINK_DIRECTORIES:STRING=${CLT_LINK_DIRS}"
              LINK_LIBRARIES ${LLVM_LDFLAGS} ${CLANG_LINK_LIBRARIES} ${LLVM_LINK_LIBRARIES} ${LLVM_SYSLIBS}
              COMPILE_DEFINITIONS ${CMAKE_CXX_FLAGS} ${LLVM_CXXFLAGS} ${CXX_COMPAT_FLAGS} -DLLVM_MAJOR=${LLVM_VERSION_MAJOR}
              OUTPUT_VARIABLE _TRY_COMPILE_OUTPUT)
  if(CLANG_LINK_TEST)
    message(STATUS "Clang link test OK")
    set(CLANG_LINK_TEST_SUCCESSFUL 1 CACHE INTERNAL "Clang link test result")
  else()
    message(STATUS "Clang link test output: ${_TRY_COMPILE_OUTPUT}")
    message(FATAL_ERROR "Clang link test FAILED. This mostly happens when your Clang installation does not have all dependencies and/or headers installed.")
  endif()
endif()

endif()

#####################################################################

