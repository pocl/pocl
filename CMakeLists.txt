#=============================================================================
#   CMake build system files
#
#   Copyright (c) 2014-2018 pocl developers
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

cmake_minimum_required(VERSION 3.9 FATAL_ERROR)
cmake_policy(SET CMP0067 OLD)

project(pocl)
set(CMAKE_PROJECT_DESCRIPTION "pocl is a portable OpenCl runtime.")

set(LATEST_KNOWN_CXX_STD_VERSION "20")
set(SUPPORTED_CXX_STD_VERSION "11")

option(ENABLE_LATEST_CXX_STD "Upgrade C++ standard version to ${LATEST_KNOWN_CXX_STD_VERSION}. Required to get rid of unused variables warnings in compilers not supporting [[gnu::*]] attributes. Can bring other benefits, including performance and efficiency ones. Before a pull request build with this disabled." OFF)
if(ENABLE_LATEST_CXX_STD)
	set(CMAKE_CXX_STANDARD "${LATEST_KNOWN_CXX_STD_VERSION}")
else()
	set(CMAKE_CXX_STANDARD "${SUPPORTED_CXX_STD_VERSION}")
endif()

include(CheckCCompilerFlag)
include(CPackComponent)
macro(pass_through_cpack_vars)
  get_cmake_property(cpackVarsToPassthrough VARIABLES)
  foreach(varName ${cpackVarsToPassthrough})
    if(varName MATCHES "^CPACK_DEBIAN_")
      message(STATUS "${varName}")
      set("${varName}" "${${varName}}" PARENT_SCOPE)
    endif()
  endforeach()
endmacro()

# don't allow implicit function declarations
if(UNIX)
  if((CMAKE_C_COMPILER_ID STREQUAL "GNU") OR
     (CMAKE_C_COMPILER_ID STREQUAL "Clang"))

    add_compile_options("$<$<COMPILE_LANGUAGE:C>:-Werror=implicit-function-declaration>")

    check_c_compiler_flag("-Wincompatible-pointer-types" HAVE_WARN_INCOMPATIBLE_POINTER_TYPES)
    if (HAVE_WARN_INCOMPATIBLE_POINTER_TYPES)
      add_compile_options("$<$<COMPILE_LANGUAGE:C>:-Wincompatible-pointer-types>")
    endif()

    add_compile_options("-Wno-ignored-attributes")

  else()
    message(WARNING "Don't know how to forbid this compiler from allowing implicit function declarations.")
  endif()

endif()

set(MAJOR_VERSION 3)
set(MINOR_VERSION 1)
set(VERSION_SUFFIX_FIXED_TEXT "-pre")
set(VERSION_SUFFIX "${VERSION_SUFFIX_FIXED_TEXT}")
set(VERSION_STRING ${MAJOR_VERSION}.${MINOR_VERSION}${VERSION_SUFFIX})
set(POCL_VERSION_BASE ${VERSION_STRING})

# required b/c SHARED libs defaults to ON while OBJECT defaults to OFF
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
# CMake doesn't add "-pie" by default for executables (CMake issue #14983)
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -pie")

enable_testing()

#####################################################

if(EXISTS "${CMAKE_SOURCE_DIR}/.git")
  set(DEFAULT_BUILD_TYPE "Debug")
else()
  set(DEFAULT_BUILD_TYPE "RelWithDebInfo")
endif()

if(NOT CMAKE_BUILD_TYPE)
  message(STATUS "Setting build type to '${DEFAULT_BUILD_TYPE}' as none was specified.")
  set(CMAKE_BUILD_TYPE "${DEFAULT_BUILD_TYPE}" CACHE
      STRING "Choose the type of build." FORCE)
  # Set the possible values of build type for cmake-gui
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS
    "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

##################################################################################

macro(set_expr VAR)
  if(${ARGN})
    set(${VAR} 1)
  else()
    set(${VAR} 0)
  endif()
endmacro()

find_program(BASH "bash")
find_program(MAKE_PROGRAM NAMES "make")
find_program(GIT_CMD "git")
set_expr(HAVE_GIT GIT_CMD)

if(HAVE_GIT)
  execute_process(COMMAND "${GIT_CMD}" "rev-parse" "HEAD"
                  OUTPUT_VARIABLE GIT_COMMIT
                  RESULT_VARIABLE EXITCODE
                  WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
                  OUTPUT_STRIP_TRAILING_WHITESPACE)
endif()

if(HAVE_GIT AND (VERSION_SUFFIX MATCHES "pre") AND (EXITCODE EQUAL 0))
  message(STATUS "Pocl source Git commit: ${GIT_COMMIT}")

  execute_process(COMMAND "${GIT_CMD}" "branch" "--contains" "${GIT_COMMIT}"
                  OUTPUT_VARIABLE GIT_BRANCH
                  RESULT_VARIABLE EXITCODE
                  WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
                  OUTPUT_STRIP_TRAILING_WHITESPACE)
  message(STATUS "Pocl source Git branch: ${GIT_BRANCH}")

  execute_process(COMMAND "${GIT_CMD}" describe "--always" "--long" "--all" "${GIT_COMMIT}"
                  OUTPUT_VARIABLE GIT_DESCRIBE
                  RESULT_VARIABLE EXITCODE
                  WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
                  OUTPUT_STRIP_TRAILING_WHITESPACE)

  string(REPLACE "heads/" "" GIT_DESCRIBE "${GIT_DESCRIBE}")
  message(STATUS "Pocl source Git describe: ${GIT_DESCRIBE}")
  set(VERSION_SUFFIX "${VERSION_SUFFIX_FIXED_TEXT} ${GIT_DESCRIBE}")
  set(VERSION_STRING ${MAJOR_VERSION}.${MINOR_VERSION}${VERSION_SUFFIX})
  set(POCL_VERSION_FULL "${VERSION_STRING}")
else()
  message(STATUS "No git and/or not a prerelease -> not adding git commit to version.")
  set(POCL_VERSION_FULL "${POCL_VERSION_BASE}")
endif()

set(CPACK_PACKAGE_NAME pocl)
set(CPACK_PACKAGE_VENDOR pocl)
set(CPACK_PACKAGE_VERSION_MAJOR "${MAJOR_VERSION}")
set(CPACK_PACKAGE_VERSION_MINOR "${MINOR_VERSION}")
set(CPACK_PACKAGE_VERSION "${MAJOR_VERSION}.${MINOR_VERSION}~${VERSION_SUFFIX_FIXED_TEXT}")
if(HAVE_GIT)
	set(CPACK_PACKAGE_VERSION "${CPACK_PACKAGE_VERSION}.${GIT_COMMIT}")
endif()

##################################################################################

if(DEFINED OCS_AVAILABLE)
  message(WARNING "The OCS_AVAILABLE option is deprecated, since it actually meant 'LLVM available', but LLVM is not the only way to get compiler support in a device. Please use ENABLE_LLVM in future if you want to enable/disable building pocl against LLVM.")
  set(ENABLE_LLVM ${OCS_AVAILABLE} CACHE BOOL "build against LLVM" FORCE)
else()
  option(ENABLE_LLVM "Build pocl with LLVM. Default is ON." ON)
endif()

option(STATIC_LLVM "If ON, link to static LLVM libraries. OFF (default) = link to shared LLVM libraries." OFF)

option(BUILD_SHARED_LIBS "ON=Build shared libs, OFF=static libs" ON)

option(POCL_DEBUG_MESSAGES
  "Enable debug messages from pocl (useful for OpenCL developers), must be enabled at runtime, with env var POCL_DEBUG"
  ON)

option(ENABLE_LOADABLE_DRIVERS "Enable drivers to be dlopen()-ed at pocl runtime, instead of being linked into libpocl" ON)

option(ENABLE_HSA "Enable the HSA base profile runtime device driver" OFF)

option(ENABLE_CUDA "Enable the CUDA device driver for NVIDIA devices" OFF)

option(ENABLE_VULKAN "Experimental and incomplete driver that uses the Vulkan API for controlling the device. Please refer to the user manual for the status and open tasks" OFF)

option(ENABLE_PROXY_DEVICE "Enable proxy driver for proxying to another OpenCL implementation" OFF)

option(ENABLE_PROXY_DEVICE_INTEROP "Enable OpenGL- or EGL-interop with the proxy driver" OFF)

option(KERNEL_CACHE_DEFAULT "Default value for the kernel compile cache. If disabled, pocl will still use kernel cache for intermediate compilation files, but will clean up them on exit. You can still enable keeping the files it at runtime with an env var." ON)

option(POCL_ICD_ABSOLUTE_PATH "Use absolute path in pocl.icd" ON)

option(ENABLE_POCL_BUILDING "When OFF, env var POCL_BUILDING has no effect. Defaults to ON" ON)

if (ENABLE_PROXY_DEVICE)
  set(VISIBILITY_HIDDEN_DEFAULT OFF)
else()
  set(VISIBILITY_HIDDEN_DEFAULT ON)
endif()
option(VISIBILITY_HIDDEN "Build with -fvisibility=hidden -fvisibility-inlines-hidden" ${VISIBILITY_HIDDEN_DEFAULT})
if(VISIBILITY_HIDDEN)
  add_compile_options(-fvisibility=hidden)
  add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-fvisibility-inlines-hidden>)
endif()


# Ninja Job Pool support
set(PARALLEL_COMPILE_JOBS "" CACHE STRING
  "Define the maximum number of concurrent compilation jobs (Ninja only).")
if(PARALLEL_COMPILE_JOBS)
  if(CMAKE_GENERATOR STREQUAL "Ninja")
    set_property(GLOBAL APPEND PROPERTY JOB_POOLS compile_job_pool=${PARALLEL_COMPILE_JOBS})
    set(CMAKE_JOB_POOL_COMPILE compile_job_pool)
  endif()
endif()

set(PARALLEL_LINK_JOBS "" CACHE STRING
  "Define the maximum number of concurrent link jobs (Ninja only).")
if(CMAKE_GENERATOR STREQUAL "Ninja")
  if(PARALLEL_LINK_JOBS)
    set_property(GLOBAL APPEND PROPERTY JOB_POOLS link_job_pool=${PARALLEL_LINK_JOBS})
    set(CMAKE_JOB_POOL_LINK link_job_pool)
  endif()
endif()

if(NOT CMAKE_GENERATOR STREQUAL "Ninja" AND (PARALLEL_COMPILE_JOBS OR PARALLEL_LINK_JOBS))
  message(WARNING "Job pooling is only available with Ninja generators.")
endif()

#### these are mostly useful for pocl developers

option(ENABLE_EXTRA_VALIDITY_CHECKS "Enable extra checks on cl_* object validity" OFF)

option(DEVELOPER_MODE "This will SIGNIFICANTLY slow down pocl (but speed up its compilation). Only turn on if you know what you're doing." OFF)

option(USE_POCL_MEMMANAGER "Enables custom memory manager. Except for special circumstances, this should be disabled." OFF)

option(EXAMPLES_USE_GIT_MASTER "If enabled, some of the external testsuites in examples/ will try to use sources from Git master, instead of releases. This may result in failure to build or run the examples" OFF)

option(ENABLE_HOST_CPU_DEVICES "Add host CPUs as OpenCL devices (basic and pthread)." ON)

option(ENABLE_ACCEL_DEVICE "Enable the generic hardware accelerator device driver." OFF)

option(ENABLE_POCLCC "Build poclcc. Defaults to ON" ON)

option(ENABLE_TESTS "Build tests. Defaults to ON" ON)

option(ENABLE_EXAMPLES "Build examples. Defaults to ON" ON)


##########################################################

if(CMAKE_SIZEOF_VOID_P EQUAL 8)
  set(HOST_DEVICE_ADDRESS_BITS 64)
elseif(CMAKE_SIZEOF_VOID_P EQUAL 4)
  set(HOST_DEVICE_ADDRESS_BITS 32)
else()
  message(FATAL_ERROR "Cannot figure out HOST_DEVICE_ADDRESS_BITS")
endif()

# printf buffer size, in KB
if(NOT DEFINED PRINTF_BUFFER_SIZE)
  set(PRINTF_BUFFER_SIZE 16384 CACHE STRING "printf buffer size, in KB")
endif()

##################################################################################

if(CMAKE_SYSTEM_PROCESSOR MATCHES "ppc64le")
  set(POWERPC 1)
  set(POWERPC64LE 1)
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "ppc")
  set(POWERPC 1)
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "mips")
  set(MIPS 1)
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "(arm|aarch64)")
  set(ARM 1)
  if(HOST_DEVICE_ADDRESS_BITS MATCHES "32")
    set(ARM32 1)
  else()
    set(ARM64 1)
  endif()
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "(i.86|AMD64|x86_64|amd64)")
  set(X86 1)
  if(HOST_DEVICE_ADDRESS_BITS MATCHES "32")
    set(I386 1)
  else()
    set(X86_64 1)
  endif()
endif()

if(CMAKE_MAJOR_VERSION GREATER 2)
  include(ProcessorCount)
  ProcessorCount(CORECOUNT)
  if(CORECOUNT LESS 1)
    set(CORECOUNT 1)
  endif()
else()
  set(CORECOUNT 1)
endif()
message(STATUS "Host CPU cores: ${CORECOUNT}")

######################################################################################

function(rename_if_different SRC DST)
  if(EXISTS "${DST}")
    file(MD5 "${SRC}" OLD_MD5)
    file(MD5 "${DST}" NEW_MD5)
    if(NOT OLD_MD5 STREQUAL NEW_MD5)
      file(RENAME "${SRC}" "${DST}")
    endif()
  else()
    file(RENAME "${SRC}" "${DST}")
  endif()
endfunction()

######################################################################################

# Recent versions of CMake can make use of Ninja's console pool to avoid
# buffering the output of particular commands.
if(CMAKE_VERSION VERSION_LESS 3.2.0)
  set(COMMAND_USES_TERMINAL)
else()
  set(COMMAND_USES_TERMINAL USES_TERMINAL)
endif()

include(GNUInstallDirs)

# for libpocl.so
set(POCL_INSTALL_PUBLIC_LIBDIR "${CMAKE_INSTALL_FULL_LIBDIR}" CACHE PATH "POCL public libdir")

# for libpocl-devices-*.so
set(POCL_INSTALL_PRIVATE_LIBDIR "${CMAKE_INSTALL_FULL_LIBDIR}/pocl" CACHE PATH "POCL private libdir")

# for pocl.icd
set(POCL_INSTALL_ICD_VENDORDIR "${CMAKE_INSTALL_FULL_SYSCONFDIR}/OpenCL/vendors" CACHE PATH "POCL ICD file destination")

# for kernel-<target>.bc
set(POCL_INSTALL_PRIVATE_DATADIR "${CMAKE_INSTALL_FULL_DATADIR}/pocl" CACHE PATH "POCL private datadir")

# for poclu.h
set(POCL_INSTALL_PUBLIC_HEADER_DIR "${CMAKE_INSTALL_FULL_INCLUDEDIR}" CACHE PATH "POCL public header dir")

# for _kernel.h et al
set(POCL_INSTALL_PRIVATE_HEADER_DIR "${POCL_INSTALL_PRIVATE_DATADIR}/include" CACHE PATH "POCL private header dir")

# for pocl-standalone et al
set(POCL_INSTALL_PUBLIC_BINDIR "${CMAKE_INSTALL_FULL_BINDIR}" CACHE PATH "POCL public bindir")

# for PoclConfig.cmake & stuff
set(POCL_INSTALL_CMAKE_CONFIG_DIR "${POCL_INSTALL_PRIVATE_LIBDIR}/cmake" CACHE PATH   "Installation directory for CMake files")

# TODO maybe use output of pkg-config --variable=pc_path pkg-config ?
set(POCL_INSTALL_PKGCONFIG_DIR "${POCL_INSTALL_PUBLIC_LIBDIR}/pkgconfig" CACHE PATH "Destination for pocl.pc")

if(APPLE)
  set(CMAKE_MACOSX_RPATH ON)
  set(POCL_INSTALL_OPENCL_HEADER_DIR "${POCL_INSTALL_PUBLIC_HEADER_DIR}/OpenCL" CACHE PATH "POCL header dir for OpenCL headers")
else()
  set(POCL_INSTALL_OPENCL_HEADER_DIR "${POCL_INSTALL_PUBLIC_HEADER_DIR}/CL" CACHE PATH "POCL header dir for OpenCL headers")
endif()

######################################################################################


######################################################################################

set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")
option(HARDENING_ENABLE "Enable hardening against various attacks. May worsen performance" OFF)
if(HARDENING_ENABLE)
  include(Hardening)
else()
  function(harden target)
  endfunction()
endif()

find_package(PkgConfig MODULE)
find_package(Hwloc)

  if(NOT Hwloc_FOUND)
    message(STATUS "hwloc package not found")
    set(ENABLE_HWLOC OFF CACHE BOOL "Hwloc" FORCE)
  else()
      if("${Hwloc_VERSION}" VERSION_LESS "1.0")
        message(FATAL_ERROR "Hwloc version must be >= 1.0 !")
      endif()

      message(STATUS "Hwloc_VERSION ${Hwloc_VERSION}")
      message(STATUS "Hwloc_LIBRARIES ${Hwloc_LIBRARIES}")
      message(STATUS "Hwloc_INCLUDE_DIRS ${Hwloc_INCLUDE_DIRS}")

      set(ENABLE_HWLOC ON CACHE BOOL "Hwloc" FORCE)
  endif()


include(sanitizers)

######################################################################################

if(NOT HOST_CPU_CACHELINE_SIZE)

  set(CL_SIZE 0)
  if(UNIX OR CMAKE_HOST_SYSTEM_NAME MATCHES "Linux|Darwin")
    find_program(GETCONF "getconf")
    if(GETCONF)
      execute_process(COMMAND "getconf" "LEVEL1_DCACHE_LINESIZE"
                      RESULT_VARIABLE RES OUTPUT_VARIABLE CL_SIZE)
      if(RES)
        message(WARNING "getconf exited with nonzero status!")
        set(CL_SIZE 0)
      else()
        # getconf may in rare conditions return "undefined" value
        if (CL_SIZE STREQUAL "undefined\n")
          set(CL_SIZE 0)
        endif()
        # getconf sometimes just returns zero
        if(NOT (CL_SIZE EQUAL 0))
          string(STRIP "${CL_SIZE}" CL_SIZE)
          message(STATUS "L1D Cacheline size detected: ${CL_SIZE}")
          set(HOST_CPU_CACHELINE_SIZE "${CL_SIZE}" CACHE STRING "L1D Cacheline size")
        endif()
      endif()
    endif()
  endif()

  if(CL_SIZE EQUAL 0)
    message(WARNING "Unable to detect cacheline size - assuming 64byte cacheline, override with -DHOST_CPU_CACHELINE_SIZE=<number> (Note: this is merely used for optimization, at worst pocl will be slightly slower)")
    set(HOST_CPU_CACHELINE_SIZE "64" CACHE STRING "L1D Cacheline size")
  endif()
endif()

######################################################################################
#
# Find executables to few tools required during build
#

find_program(PATCH_EXEC
  NAMES patch
  HINTS ENV PATH
)

find_program(XARGS_EXEC
  NAMES xargs
  HINTS ENV PATH
)

if(NOT PATCH_EXEC)
  message(FATAL_ERROR "Could not find patch command.")
endif()

if(NOT XARGS_EXEC)
  message(FATAL_ERROR "Could not find xargs command.")
endif()

######################################################################################

if(ENABLE_LLVM)

  # this must be done before including LLVM,
  # so we can check that TCE's LLVM and the found LLVM is the same.
  if(NOT WITH_TCE)
    set(WITH_TCE ENV PATH)
  endif()

  find_program(TCE_CONFIG NAMES "tce-config" HINTS ${WITH_TCE})
  find_program(TCECC NAMES "tcecc" HINTS ${WITH_TCE})
  find_program(TTASIM NAMES "ttasim" HINTS ${WITH_TCE})

  include(LLVM RESULT_VARIABLE RES)
  if(NOT RES)
    message(FATAL_ERROR "Could not load LLVM.cmake")
  endif()

  if(ENABLE_HOST_CPU_DEVICES)
  if(NOT DEFINED HOST_DEVICE_BUILD_HASH)
    if(KERNELLIB_HOST_CPU_VARIANTS STREQUAL "distro")
      set(HOST_DEVICE_BUILD_HASH "${LLC_TRIPLE}")
    else()
      set(HOST_DEVICE_BUILD_HASH "${LLC_TRIPLE}-${LLC_HOST_CPU}")
    endif()
  endif()

  if(INTEL_SDE_AVX512)
    set(HOST_CPU_FORCED 1 CACHE INTERNAL "CPU is forced by user" FORCE)
    set(LLC_HOST_CPU "skylake-avx512" CACHE STRING "The Host CPU to use with llc" FORCE)
  endif()
  endif()

else()

  if(ENABLE_HOST_CPU_DEVICES AND (NOT DEFINED HOST_DEVICE_BUILD_HASH))
    message(FATAL_ERROR "For compiler-less builds of CPU backend, you must define HOST_DEVICE_BUILD_HASH")
  endif()

endif()

######################################################################################

if(ENABLE_HSA)
  include(HSA RESULT_VARIABLE RES)
  if(NOT RES)
    message(FATAL_ERROR "Could not load HSA.cmake")
  endif()
endif()

######################################################################################

if (NOT MSVC)
  find_program(LINK_COMMAND
    NAMES ld${CMAKE_EXECUTABLE_SUFFIX}
    HINTS ENV PATH
  )
else()
    set(LINK_COMMAND "${CLANGXX}")
endif()

######################################################################################

# if variable FEATURE_X isn't defined, sets it to DEFAULT_FEATURE_X;
# also, if DEFAULT_FEATURE_X is 0, prevents FEATURE_X being 1
# since it takes DEFAULT_FEATURE_X=0 to mean "FEATURE_X is unavailable"
macro(setup_cached_var VARNAME DESCRIPTION DOCS_FEATURE_IS_UNAVAILABLE DOCS_REQUESTED_DISABLING_FEATURE)

  if(DEFINED ${VARNAME})
    set(_CACHED "(cached)")
  else()
    set(_CACHED "")
    set(${VARNAME} ${DEFAULT_${VARNAME}})
  endif()

  if(${VARNAME} AND (NOT ${DEFAULT_${VARNAME}}))
    message(WARNING "${DOCS_FEATURE_IS_UNAVAILABLE}")
    set(${VARNAME} 0)
    set(_CACHED "(override)")
  endif()
  if((NOT ${VARNAME}) AND ${DEFAULT_${VARNAME}} )
    message(STATUS "${DOCS_REQUESTED_DISABLING_FEATURE}")
  endif()
  if(${VARNAME})
    message(STATUS "${DESCRIPTION} ${_CACHED}: 1")
  else()
    message(STATUS "${DESCRIPTION} ${_CACHED}: 0")
  endif()
endmacro()

######################################################################################

if(UNIX)
  include(CheckCSourceCompiles)
  include(CheckSymbolExists)

  # don't allow implicit function declarations
  set(CMAKE_REQUIRED_FLAGS "-std=c99")
  if (CMAKE_SYSTEM_NAME MATCHES "Linux")
    set(CMAKE_REQUIRED_LIBRARIES "rt")
  endif ()

  CHECK_SYMBOL_EXISTS("fork"
                      "sys/types.h;unistd.h"
                      HAVE_FORK)

  CHECK_SYMBOL_EXISTS("fsync"
                      "unistd.h"
                      HAVE_FSYNC)

  CHECK_SYMBOL_EXISTS("sleep"
                      "unistd.h"
                      HAVE_SLEEP)

  CHECK_SYMBOL_EXISTS("getrlimit"
                      "sys/time.h;sys/resource.h"
                      HAVE_GETRLIMIT)

  CHECK_SYMBOL_EXISTS("utime"
                      "sys/types.h;utime.h"
                      HAVE_UTIME)

  CHECK_SYMBOL_EXISTS("ANNOTATE_HAPPENS_BEFORE"
                      "valgrind/helgrind.h"
                      HAVE_VALGRIND)

  set(CMAKE_REQUIRED_DEFINITIONS "-D_POSIX_C_SOURCE=200809L")
  CHECK_SYMBOL_EXISTS("futimens"
                      "fcntl.h;sys/stat.h"
                      HAVE_FUTIMENS)

  set(CMAKE_REQUIRED_DEFINITIONS "-D_POSIX_C_SOURCE=200112L")
  CHECK_SYMBOL_EXISTS("posix_memalign"
                      "stdlib.h"
                      HAVE_POSIX_MEMALIGN)

  set(CMAKE_REQUIRED_DEFINITIONS "-D_POSIX_C_SOURCE=199309L")
  CHECK_SYMBOL_EXISTS("clock_gettime"
                      "time.h"
                      HAVE_CLOCK_GETTIME)

  CHECK_SYMBOL_EXISTS("fdatasync"
                      "unistd.h"
                      HAVE_FDATASYNC)

  set(CMAKE_REQUIRED_DEFINITIONS "-D_BSD_SOURCE" "-D_DEFAULT_SOURCE")
  CHECK_SYMBOL_EXISTS("mkdtemp"
                      "stdlib.h;unistd.h"
                      HAVE_MKDTEMP)

  CHECK_SYMBOL_EXISTS("mkstemps"
                      "stdlib.h;unistd.h"
                      HAVE_MKSTEMPS)

  CHECK_SYMBOL_EXISTS("vfork"
                      "sys/types.h;unistd.h"
                      HAVE_VFORK)

  set(CMAKE_REQUIRED_DEFINITIONS "-D_GNU_SOURCE")
  CHECK_SYMBOL_EXISTS("mkostemps"
                      "stdlib.h"
                      HAVE_MKOSTEMPS)

  set(CMAKE_REQUIRED_LIBRARIES "dl")
  CHECK_SYMBOL_EXISTS("dladdr"
                      "dlfcn.h"
                      HAVE_DLADDR)

  unset(CMAKE_REQUIRED_DEFINITIONS)
  unset(CMAKE_REQUIRED_FLAGS)
  unset(CMAKE_REQUIRED_LIBRARIES)

else()
  set(HAVE_CLOCK_GETTIME 0)
  set(HAVE_FDATASYNC 0)
  set(HAVE_FSYNC 0)
  set(HAVE_SLEEP 0)
  set(HAVE_MKOSTEMPS 0)
  set(HAVE_MKSTEMPS 0)
  set(HAVE_MKDTEMP 0)
  set(HAVE_FUTIMENS 0)
  set(HAVE_FORK 0)
  set(HAVE_GETRLIMIT 0)
  set(HAVE_VFORK 0)
  set(HAVE_UTIME 0)
  set(HAVE_DLADDR 0)
  set(HAVE_VALGRIND 0)
endif()

######################################################################################

function(check_64bit_atomics varname)
    check_c_source_compiles("
#include <stdint.h>
uint64_t x = 0;
int main()
{
  __atomic_add_fetch(&x, 1, __ATOMIC_SEQ_CST);
  __atomic_sub_fetch(&x, 1, __ATOMIC_SEQ_CST);
  return x;
}
" ${varname})
endfunction(check_64bit_atomics)

# platforms w/o lockfree 64bit atomics need to link with -latomic
if(UNIX)
    check_64bit_atomics(HAVE_64BIT_ATOMICS_WITHOUT_LIB)
    if(HAVE_64BIT_ATOMICS_WITHOUT_LIB)
        set(HAVE_64BIT_ATOMICS_WITH_LIB FALSE)
    else()
        set(OLD_CMAKE_REQUIRED_LIBRARIES ${CMAKE_REQUIRED_LIBRARIES})
        list(APPEND CMAKE_REQUIRED_LIBRARIES "atomic")
        check_64bit_atomics(HAVE_64BIT_ATOMICS_WITH_LIB)
        set(CMAKE_REQUIRED_LIBRARIES ${OLD_CMAKE_REQUIRED_LIBRARIES})
        if(NOT HAVE_64BIT_ATOMICS_WITH_LIB)
            message(SEND_ERROR "Missing 64-bit atomic increment")
        endif()
    endif()
endif()

######################################################################################

if(UNIX AND ENABLE_LLVM AND HAVE_DLADDR)
    option(ENABLE_RELOCATION "make libpocl relocatable" ON)
else()
    message(STATUS "Relocation not available")
    set(ENABLE_RELOCATION OFF CACHE INTERNAL "libpocl relocatable" FORCE)
endif()

if(ENABLE_RELOCATION)
  file(RELATIVE_PATH POCL_INSTALL_PRIVATE_DATADIR_REL ${POCL_INSTALL_PUBLIC_LIBDIR} ${POCL_INSTALL_PRIVATE_DATADIR})
  message(STATUS "Private Datadir Relative path: ${POCL_INSTALL_PRIVATE_DATADIR_REL}")
  install(FILES ${CLANG_OPENCL_HEADERS}
          DESTINATION "${POCL_INSTALL_PRIVATE_DATADIR}/include" COMPONENT "dev")
endif()

file(RELATIVE_PATH POCL_INSTALL_PRIVATE_LIBDIR_REL ${POCL_INSTALL_PUBLIC_LIBDIR} ${POCL_INSTALL_PRIVATE_LIBDIR})

######################################################################################

# IPO support for runtime library
if(POLICY CMP0069)
  cmake_policy(SET CMP0069 NEW)
endif()

if(NOT DEFINED DEFAULT_ENABLE_IPO)
  set(DEFAULT_ENABLE_IPO OFF CACHE BOOL "IPO" FORCE)

  if(NOT CMAKE_VERSION VERSION_LESS "3.9")

    include(CheckIPOSupported)
    check_ipo_supported(RESULT IPO OUTPUT IPO_OUTPUT)
    set(DEFAULT_ENABLE_IPO ${IPO} CACHE BOOL "IPO" FORCE)

    message(STATUS "Compiler supports IPO: ${DEFAULT_ENABLE_IPO}")
    #message(STATUS "IPO check message: ${IPO_OUTPUT}")
  endif()
endif()

setup_cached_var(ENABLE_IPO "Enable Link-Time Optimization (IPO) while building pocl runtime"
  "Requested build with IPO, but IPO is not available"
  "IPO available, but requested build without it")

######################################################################################

option(ENABLE_SLEEF "Use SLEEF for kernel library" ON)

option(ENABLE_CONFORMANCE "Enable conformance to OpenCL standard. \
  Enabling this option this does not guarantee conformance (depends on hardware), \
  but CMake will give errors if options that conflict with conformance are used" OFF)

if(ENABLE_CONFORMANCE AND (NOT ENABLE_SLEEF))
  message(FATAL_ERROR "conformance needs enabled SLEEF")
endif()

######################################################################################

# fully device-side printf on devices which support it (only CPU backend ATM), disabled by default.
# this requires 128bit integer support because of the code in "errol" float-to-string conversion routine
# the output is not 100% compatible with glibc's printf (%f with large argument prints zeroes after
# last significant digit - 16-18th digit or so, unlike glibc which prints digits up to decimal point).
if(CLANG_HAS_128B_MATH)
  option(ENABLE_POCL_FLOAT_CONVERSION "Enable use of pocl's own float-to-decimal conversion code in OpenCL printf(). Defaults to OFF (uses snprintf from C library). Requires compiler-rt." OFF)
else()
  set(ENABLE_POCL_FLOAT_CONVERSION OFF CACHE INTERNAL "pocl's own float-to-decimal conversion code")
endif()

unset(FLOATCONV_FLAG)
if(ENABLE_POCL_FLOAT_CONVERSION)
  # force link with Clang; otherwise not needed on x86 but in this case we need rtlib
  set(FLOATCONV_FLAG "-DENABLE_POCL_FLOAT_CONVERSION")
endif()

######################################################################################

# for kernel code, disable PIC & stack protector
#
# it seems PIC and stack-protector defaults somehow depend on
# clang build type or environment. PIC causes problems with
# constant addrspace variables, and stack protector likely slows
# down the kernels, so it needs to be determined whether it's worth
# the trouble.
set(DEFAULT_KERNEL_CL_FLAGS  "-xcl -fno-stack-protector -fPIC ${FLOATCONV_FLAG}")
set(DEFAULT_KERNEL_C_FLAGS "-xc -std=c11 -D__CBUILD__ -fno-math-errno -fno-stack-protector -fPIC ${FLOATCONV_FLAG}")
set(DEFAULT_KERNEL_CXX_FLAGS "-xc++ -std=c++11 -fno-stack-protector -fPIC ${FLOATCONV_FLAG}")

set(EXTRA_KERNEL_FLAGS "" CACHE STRING "Extra arguments to all kernel compilation commands (defaults to empty)")
set(EXTRA_KERNEL_CL_FLAGS "" CACHE STRING "Extra arguments to kernel CL compiler (defaults to empty)")
set(EXTRA_KERNEL_CXX_FLAGS "" CACHE STRING "Extra arguments to kernel CXX compiler (defaults to empty)")
set(EXTRA_KERNEL_C_FLAGS "" CACHE STRING "Extra arguments to kernel C compiler (defaults to empty)")

set(KERNEL_CXX_FLAGS "${DEFAULT_KERNEL_CXX_FLAGS}${EXTRA_KERNEL_FLAGS}${EXTRA_KERNEL_CXX_FLAGS}")
set(KERNEL_CL_FLAGS "${DEFAULT_KERNEL_CL_FLAGS}${EXTRA_KERNEL_FLAGS}${EXTRA_KERNEL_CL_FLAGS}")
set(KERNEL_C_FLAGS "${DEFAULT_KERNEL_C_FLAGS}${EXTRA_KERNEL_FLAGS}${EXTRA_KERNEL_C_FLAGS}")

######################################################################################

if(UNIX)
  if(APPLE)
    # MacOS ld outputs useless warnings like
    # ld: warning: -macosx_version_min not specificed, assuming 10.7
    # suppress them with -w.
    set(DEFAULT_HOST_LD_FLAGS "-dynamiclib -w -lm")
  elseif(ANDROID)
    set(DEFAULT_HOST_LD_FLAGS "-L/system/lib/ -shared -ldl -lc /system/lib/crtbegin_so.o /system/lib/crtend_so.o")
  else()
    set(DEFAULT_HOST_LD_FLAGS "-shared")
  endif()
  set(LIBMATH "-lm")
elseif(WIN32)
  set(LIBMATH)
endif()

if(CLANG_NEEDS_RTLIB)
  set(DEFAULT_HOST_LD_FLAGS "${DEFAULT_HOST_LD_FLAGS} --rtlib=compiler-rt")
endif()

######################################################################################

if(UNIX)
  if(APPLE)
    # TODO MACOSX_BUNDLE target prop
    set(ICD_LD_FLAGS "-single_module")
  else()
    set(ICD_LD_FLAGS "-Wl,-Bsymbolic")
  endif()
endif()

######################################################################################

set(SPIRV OFF)

if(ENABLE_LLVM AND X86 AND (NOT KERNELLIB_HOST_CPU_VARIANTS STREQUAL "distro")
   AND (NOT ENABLE_CONFORMANCE))
  option(ENABLE_SPIR "Enable SPIR support (default ON when available)" ON)
else()
  set(ENABLE_SPIR OFF CACHE INTERNAL "SPIR enabled" FORCE)
endif()

if(ENABLE_SPIR)

  message(WARNING "SPIR support is available but highly experimental; use at your own risk.")

  if(LLVM_SPIRV AND (EXISTS "${LLVM_SPIRV}"))
    message(WARNING "SPIR-V support enabled but highly experimental; you must use a llvm-spirv "
      "converter that produces bitcode FOR YOUR LLVM VERSION. "
      "E.g. if you're compiling pocl against LLVM 5 then using Khronos' "
      "llvm-spirv based on LLVM 3.6 branch WILL NOT WORK.")
    set(SPIRV ON)
  endif()

endif()

set(ENABLE_SPIRV ${SPIRV} CACHE INTERNAL "SPIR-V enabled" FORCE)

if(ENABLE_CONFORMANCE AND (ENABLE_SPIR OR ENABLE_SPIRV))
  message(FATAL_ERROR "conformance needs SPIR AND SPIR-V support disabled")
endif()

######################################################################################

set(HAVE_DLFCN_H OFF CACHE BOOL "dlopen" FORCE)

if(WIN32 AND (NOT MINGW))
  message(STATUS "Using LoadLibrary/FreeLibrary in Windows, libltdl not needed.")

elseif(UNIX)

  if (CMAKE_CROSSCOMPILING AND (NOT ENABLE_HOST_CPU_DEVICES) AND (NOT ENABLE_HSA))
    message(STATUS "Cross-compiling without CPU/HSA devices -> skipping LIBDL search")
  else()

    find_library(DL_LIB "dl")
    find_file(DL_H "dlfcn.h")

    if(DL_LIB AND DL_H)
      message(STATUS "libdl found")
    else()
      message(STATUS "libdl not found, assuming dlopen() is in libc")
      set(DL_LIB "")
    endif()

    if(DL_H)
      get_filename_component(DL_H_INCLUDE_DIR "${DL_H}" DIRECTORY)
      string(FIND "${CMAKE_C_IMPLICIT_INCLUDE_DIRECTORIES}" "${DL_H_INCLUDE_DIR}" LTPOSITION)
      # include the directory of dlfcn.h, if its not in the default system include dirs
      # also when cross-compiling this includes <cross-compile-root>/usr/include, which screws things up
      if((LTPOSITION LESS "0") AND (NOT CMAKE_CROSSCOMPILING))
        include_directories("${DL_H_INCLUDE_DIR}")
      endif()
      set(HAVE_DLFCN_H ON CACHE BOOL "dlfcn.h" FORCE)
    else()
      message(FATAL_ERROR "Could not find dlfcn.h!")
    endif()
  endif()

else()
  message(STATUS "Unknown OS, don't know how to load a dynamic library")
endif()

######################################################################################

set(CMAKE_THREAD_PREFER_PTHREAD TRUE)
set(THREADS_PREFER_PTHREAD_FLAG TRUE)
find_package(Threads REQUIRED)

if(CMAKE_VERSION VERSION_GREATER "3.0.99")
  set(PTHREAD_LIBRARY Threads::Threads)
else()
  set(PTHREAD_LIBRARY ${CMAKE_THREAD_LIBS_INIT})
endif()

######################################################################################
# LTTNG

if(UNIX)
  if(PKG_CONFIG_EXECUTABLE)
    pkg_check_modules(LTTNG_UST lttng-ust>=2.7)
  endif()
  if(LTTNG_UST_FOUND)
    set(HAVE_LTTNG_UST 1)
  else()
    set(HAVE_LTTNG_UST 0)
  endif()
endif()

######################################################################################

if(NOT DEFINED DEFAULT_ENABLE_ICD)

if (MSVC)
  message(STATUS "Building ICD not yet supported on Windows.")
  set(DEFAULT_ENABLE_ICD 0 CACHE INTERNAL "Going to use ICD loader")

else()

  # pkg-config doesn't work with cross-compiling
  if(PKG_CONFIG_EXECUTABLE)
    pkg_check_modules(OCL_ICD ocl-icd>=1.3)
  endif()

  if (NOT OCL_ICD_FOUND)
    find_path(OCL_ICD_INCLUDE_DIRS
      NAMES
        ocl_icd.h
    )
    find_library(OCL_ICD_LIBRARIES
      NAMES
        OpenCL
    )
    if(OCL_ICD_INCLUDE_DIRS AND OCL_ICD_LIBRARIES)
      set(OCL_ICD_FOUND 1)
    endif()
  endif()

  if(OCL_ICD_FOUND)

    set(HAVE_OCL_ICD 1 CACHE INTERNAL "ICL library is ocl-icd")
    set(OPENCL_FOUND 1 CACHE INTERNAL "opencl ICD/library found")

    set(OPENCL_LIBRARIES "OpenCL" CACHE INTERNAL "ocl-icd library")
    set(OPENCL_LIBDIR "${OCL_ICD_LIBDIR}" CACHE INTERNAL "opencl ICD/library path")

    set(DEFAULT_ENABLE_ICD 1 CACHE INTERNAL "ICD loader availability")
    if(DEFINED OCL_ICD_VERSION AND (OCL_ICD_VERSION VERSION_GREATER_EQUAL "2.3.0"))
      set(HAVE_OCL_ICD_30_COMPATIBLE 1 CACHE INTERNAL "ICD 3.0 compat")
    else()
      set(HAVE_OCL_ICD_30_COMPATIBLE 0 CACHE INTERNAL "ICD 3.0 compat")
    endif()
  else()

    set(HAVE_OCL_ICD 0 CACHE INTERNAL "OCL library is ocl-icd")
    unset (OPENCL_FOUND CACHE)

      # fallback to other ICD loaders
      message(STATUS "ocl-icd not found -> trying fallback ICD implementations")
      if(PKG_CONFIG_EXECUTABLE)
        pkg_check_modules(OPENCL OpenCL>=1.2)
      endif()
      if(NOT OPENCL_FOUND)
        find_library(OPENCL_LIBRARIES OpenCL)
        # version check the found library
        if(OPENCL_LIBRARIES)
          set(CMAKE_REQUIRED_LIBRARIES "${OPENCL_LIBRARIES}")
          include(CheckFunctionExists)
          unset (OPENCL_FOUND CACHE)
          CHECK_FUNCTION_EXISTS("clEnqueueFillImage" OPENCL_FOUND)
        endif()
      endif()

    if(OPENCL_FOUND)
      # no ocl-icd, but libopencl
      message(STATUS "libOpenCL (unknown ICD loader) found")
      set(DEFAULT_ENABLE_ICD 1 CACHE INTERNAL "ICD loader availability")
    else()
      message(STATUS "No ICD loader of any kind found (or its OpenCL version is <1.2)")
      # no ocl-icd, no libopencl
      set(DEFAULT_ENABLE_ICD 0 CACHE INTERNAL "no ICL loader found availability")
    endif()

  endif()

endif()

endif()

setup_cached_var(ENABLE_ICD "Using an ICD loader"
  "Requested build with icd, but ICD loader not found! some examples will not work.."
  "ICD loader found, but requested build without it")

if(ENABLE_ICD)
  # only meaningful to link tests with ocl-icd
  set(TESTS_USE_ICD ${HAVE_OCL_ICD})
else()
  set(TESTS_USE_ICD 0)
endif()

if(ENABLE_ICD OR ENABLE_PROXY_DEVICE)
  set(POCL_LIBRARY_NAME "pocl")
else()
  set(POCL_LIBRARY_NAME "OpenCL")
endif()

message(STATUS "Run tests with ICD: ${TESTS_USE_ICD}")

######################################################################################

# Apple's OpenCL headers are too old for building PoCL, always use our own
if(NOT APPLE)
  find_file(OPENCL_H opencl.h PATH_SUFFIXES CL)
  find_file(OPENCL_HPP opencl.hpp PATH_SUFFIXES CL)
endif()

# if ICD is disabled, we might as well compile with our own headers
if(OPENCL_H AND ENABLE_ICD)
  message(STATUS "OpenCL.h found (${OPENCL_H}) and ICD enabled, NOT installing our headers")
  set(HAVE_OPENCL_H ON)
  set(IOH OFF)
else()
  message(STATUS "OpenCL.h not found or ICD disabled, installing our headers")
  set(HAVE_OPENCL_H OFF)
  set(IOH ON)
endif()

if(NOT DEFINED INSTALL_OPENCL_HEADERS)
  option(INSTALL_OPENCL_HEADERS "Install POCL's OpenCL headers. (Ones from Khronos should be installed instead)" ${IOH})
endif()

add_definitions(-DCL_USE_DEPRECATED_OPENCL_1_0_APIS -DCL_USE_DEPRECATED_OPENCL_1_1_APIS
                -DCL_USE_DEPRECATED_OPENCL_1_2_APIS -DCL_USE_DEPRECATED_OPENCL_2_0_APIS
                -DCL_USE_DEPRECATED_OPENCL_2_1_APIS -DCL_USE_DEPRECATED_OPENCL_2_2_APIS)

if(HAVE_OPENCL_H)
  include(CheckSymbolExists)
  check_symbol_exists(CL_VERSION_3_0 ${OPENCL_H} HAVE_OPECL_3_0)
  check_symbol_exists(CL_VERSION_2_2 ${OPENCL_H} HAVE_OPECL_2_2)
  check_symbol_exists(CL_VERSION_2_1 ${OPENCL_H} HAVE_OPECL_2_1)
  check_symbol_exists(CL_VERSION_2_0 ${OPENCL_H} HAVE_OPECL_2_0)
  check_symbol_exists(CL_VERSION_1_2 ${OPENCL_H} HAVE_OPECL_1_2)
  if(HAVE_OPECL_3_0)
    set(OPENCL_HEADER_VERSION 300)
  elseif(HAVE_OPECL_2_2)
    set(OPENCL_HEADER_VERSION 220)
  elseif(HAVE_OPECL_2_1)
    set(OPENCL_HEADER_VERSION 210)
  elseif(HAVE_OPECL_2_0)
    set(OPENCL_HEADER_VERSION 200)
  elseif(HAVE_OPECL_1_2)
    set(OPENCL_HEADER_VERSION 120)
  else()
    message(FATAL_ERROR "Couldn't find OpenCL header with version at least 1.2")
  endif()
else()
  set(OPENCL_HEADER_VERSION 300)
endif()

macro(set_opencl_header_includes)
  add_definitions(-DCL_TARGET_OPENCL_VERSION=${OPENCL_HEADER_VERSION}
                  -DCL_HPP_TARGET_OPENCL_VERSION=${OPENCL_HEADER_VERSION})
  if(NOT HAVE_OPENCL_H)
    include_directories("${CMAKE_SOURCE_DIR}/include")
  endif()
  if(NOT OPENCL_CL2_HPP)
    include_directories("${CMAKE_SOURCE_DIR}/include/hpp")
  endif()
  include_directories("${CMAKE_SOURCE_DIR}/poclu")
endmacro()

######################################################################################

option(PEDANTIC "Compile host library with stricter compiler flags." OFF)
if(PEDANTIC)
  add_compile_options("-Wno-unused-result" "-Werror") # maybe "-Wimplicit"
endif()

######################################################################################

set_expr(POCL_KERNEL_CACHE_DEFAULT KERNEL_CACHE_DEFAULT)

string(TIMESTAMP POCL_BUILD_TIMESTAMP "%d%m%Y%H%M%S")
file(WRITE "${CMAKE_BINARY_DIR}/pocl_build_timestamp.h" "#define POCL_BUILD_TIMESTAMP \"${POCL_BUILD_TIMESTAMP}\"")

####################################################################

# Host (basic/pthread) driver setup

set(DEFAULT_HOST_CLANG_FLAGS "${CLANG_TARGET_OPTION}${LLC_TRIPLE}")
set(DEFAULT_HOST_LLC_FLAGS "-relocation-model=pic -mtriple=${LLC_TRIPLE}")

if(ARM)
  option(ENABLE_FP64 "Enable FP64 on ARM32 - if you have at least VFP support for doubles, you can leave it ON" ON )
else()
  set(ENABLE_FP64 ON CACHE INTERNAL "FP64, always on except ARM")
endif()

if(ARM32 OR (LLC_TRIPLE MATCHES "^arm"))
  if(LLC_TRIPLE MATCHES "gnueabihf")
    # hardfloat
    set(DEFAULT_HOST_LLC_FLAGS "${DEFAULT_HOST_LLC_FLAGS} -float-abi=hard")
    set(DEFAULT_HOST_CLANG_FLAGS "${DEFAULT_HOST_CLANG_FLAGS} -mfloat-abi=hard")
    set(DEFAULT_HOST_AS_FLAGS "${DEFAULT_HOST_AS_FLAGS} -mfloat-abi=hard")
  else()
    # softfloat
    set(HOST_FLOAT_SOFT_ABI 1)
    set(DEFAULT_HOST_LLC_FLAGS "${DEFAULT_HOST_LLC_FLAGS} -float-abi=soft")
    set(DEFAULT_HOST_CLANG_FLAGS "${DEFAULT_HOST_CLANG_FLAGS} -mfloat-abi=soft")
    set(DEFAULT_HOST_AS_FLAGS "${DEFAULT_HOST_AS_FLAGS} -mfloat-abi=soft")
  endif()
endif()

if(CL_DISABLE_HALF)
  set(DEFAULT_HOST_CLANG_FLAGS "${DEFAULT_HOST_CLANG_FLAGS} -D_CL_DISABLE_HALF")
endif()

# set the host device OpenCL version (and OpenCL C version too)
# ..... to enable OpenCL C 3.0 language for the CPU device,
# it must have compiler capable of 3.0
# theoretically LLVM 12 is enough, but it lacks
# https://reviews.llvm.org/D106778
# and so it fails the C11 atomics tests of the CTS
# with '... unknown call to _cl_atomic_store_explicit' or similar
if(LLVM_VERSION VERSION_GREATER_EQUAL 14.0)
  set(HOST_DEVICE_CL_VERSION_MAJOR 3)
  set(HOST_DEVICE_CL_VERSION_MINOR 0)
else()
  set(HOST_DEVICE_CL_VERSION_MAJOR 1)
  set(HOST_DEVICE_CL_VERSION_MINOR 2)
endif()


# These are mandatory for OpenCL 1.1 (except for 3D writes but those are CPU-independent)
set(HOST_DEVICE_EXTENSIONS "cl_khr_byte_addressable_store cl_khr_global_int32_base_atomics \
  cl_khr_global_int32_extended_atomics cl_khr_local_int32_base_atomics \
  cl_khr_local_int32_extended_atomics cl_khr_3d_image_writes")

set(HOST_DEVICE_FEATURES_30 "__opencl_c_3d_image_writes  __opencl_c_images \
  __opencl_c_atomic_order_acq_rel __opencl_c_atomic_order_seq_cst \
  __opencl_c_atomic_scope_device")

# TODO: __opencl_c_atomic_scope_all_devices works with CPU device but not others
# for now it's included because CTS test 'C11 atomics' subtest 'atomic_flag' requires it
# .... seems like a bug in CTS, but needs to be investigated.
if(ENABLE_CONFORMANCE)
  set( HOST_DEVICE_FEATURES_30  "${HOST_DEVICE_FEATURES_30} __opencl_c_atomic_scope_all_devices")
endif()

if(NOT ENABLE_CONFORMANCE)
  set( HOST_DEVICE_FEATURES_30 "${HOST_DEVICE_FEATURES_30} __opencl_c_read_write_images")
endif()

if(ENABLE_SPIR)
  set(HOST_DEVICE_EXTENSIONS "${HOST_DEVICE_EXTENSIONS} cl_khr_spir")
  set(DEFAULT_DEVICE_EXTENSIONS "${DEFAULT_DEVICE_EXTENSIONS} cl_khr_spir")
endif()

if(ENABLE_SPIRV)
  set(HOST_DEVICE_EXTENSIONS "${HOST_DEVICE_EXTENSIONS} cl_khr_il_program")
endif()

if(NOT CL_DISABLE_HALF)
  set(HOST_DEVICE_EXTENSIONS "${HOST_DEVICE_EXTENSIONS} cl_khr_fp16")
  set(HOST_DEVICE_FEATURES_30 "${HOST_DEVICE_FEATURES_30} __opencl_c_fp16")
endif()


# don't include cl_khr_int64 in HOST_DEVICE_EXTENSIONS list, because
# such extension doesn't exist in official extension list
# for FULL profile, int64 is always enabled;
# for embedded profiles, it's optional
set(HOST_DEVICE_EXTENSION_DEFINES "${HOST_DEVICE_EXTENSION_DEFINES} -Dcl_khr_int64=1")
set(HOST_DEVICE_FEATURES_30 "${HOST_DEVICE_FEATURES_30} __opencl_c_int64")
set(HOST_DEVICE_EXTENSIONS "${HOST_DEVICE_EXTENSIONS} cl_khr_int64_base_atomics cl_khr_int64_extended_atomics")

if(ENABLE_FP64)
  set(HOST_DEVICE_EXTENSIONS "${HOST_DEVICE_EXTENSIONS} cl_khr_fp64")
  set(HOST_DEVICE_FEATURES_30 "${HOST_DEVICE_FEATURES_30} __opencl_c_fp64")
  set(_CL_DISABLE_DOUBLE 0)
else()
  set(_CL_DISABLE_DOUBLE 1)
endif()

set(TEMP_EXT "${HOST_DEVICE_EXTENSIONS}")
separate_arguments(TEMP_EXT)
set(TEMP_CLEXT "-Xclang -cl-ext=-all,")
foreach(EXT ${TEMP_EXT})
  set(HOST_DEVICE_EXTENSION_DEFINES "${HOST_DEVICE_EXTENSION_DEFINES} -D${EXT}=1")
  set(TEMP_CLEXT "${TEMP_CLEXT}+${EXT},")
endforeach()

if(HOST_DEVICE_CL_VERSION_MAJOR GREATER_EQUAL 3)
  set(TEMP_EXT "${HOST_DEVICE_FEATURES_30}")
  separate_arguments(TEMP_EXT)
  foreach(EXT ${TEMP_EXT})
    set(HOST_DEVICE_EXTENSION_DEFINES "${HOST_DEVICE_EXTENSION_DEFINES} -D${EXT}=1")
    set(TEMP_CLEXT "${TEMP_CLEXT}+${EXT},")
  endforeach()
endif()

set(HOST_DEVICE_EXTENSION_DEFINES "${HOST_DEVICE_EXTENSION_DEFINES} ${TEMP_CLEXT}")

if(NOT DEFINED KERNELLIB_HOST_CPU_VARIANTS)
  set(KERNELLIB_HOST_CPU_VARIANTS "native")
# else TODO test cpu list for unknown values
endif()

set(KERNELLIB_HOST_DISTRO_VARIANTS 0)
if(KERNELLIB_HOST_CPU_VARIANTS STREQUAL "distro")
  if(HOST_CPU_FORCED)
    message(FATAL_ERROR "Cannot build with CPU autodetection distro variants build, and enforce LLC_HOST_CPU at the same time. Please pick one")
  endif()
  if(X86_64 OR I386)
    set(KERNELLIB_HOST_CPU_VARIANTS sse2 ssse3 sse41 avx avx_f16c avx_fma4 avx2 avx512)
  elseif(POWERPC64LE)
    set(KERNELLIB_HOST_CPU_VARIANTS pwr8 pwr9)
  elseif(APPLE AND ARM64)
    set(KERNELLIB_HOST_CPU_VARIANTS cyclone)
  else()
    message(FATAL_ERROR "Don't know what CPU variants to use for kernel library on this platform.")
  endif()
  set(KERNELLIB_HOST_DISTRO_VARIANTS 1)
endif()

####################################################################

set(EXTRA_HOST_AS_FLAGS "" CACHE STRING "Extra parameters to as for code generation in the host. (default: empty)")
set(EXTRA_HOST_LD_FLAGS "" CACHE STRING "Extra parameter to compiler to generate loadable module. (default: empty)")
set(EXTRA_HOST_CLANG_FLAGS "" CACHE STRING "Extra parameters to clang for host compilation. (default: empty)")
set(EXTRA_HOST_LLC_FLAGS "" CACHE STRING "Extra parameters to llc for code generation in the host. (default: empty)")

####################################################################

set(HOST_AS_FLAGS "${DEFAULT_HOST_AS_FLAGS} ${EXTRA_HOST_AS_FLAGS}")
set(HOST_LD_FLAGS "${DEFAULT_HOST_LD_FLAGS} ${EXTRA_HOST_LD_FLAGS}" )
string(STRIP "${HOST_LD_FLAGS}" HOST_LD_FLAGS_STRIPPED)
string(REGEX REPLACE "[\r\n\t ]+" "\", \"" HOST_LD_FLAGS_ARRAY "${HOST_LD_FLAGS_STRIPPED}")
# string(REPLACE "###, ###" " oo \", \" oo " HOST_LD_FLAGS_ARRAY "${HOST_LD_FLAGS_ARRAY_1}")

set(HOST_CLANG_FLAGS "${DEFAULT_HOST_CLANG_FLAGS} ${EXTRA_HOST_CLANG_FLAGS}")
set(HOST_LLC_FLAGS "${DEFAULT_HOST_LLC_FLAGS} ${EXTRA_HOST_LLC_FLAGS}")

if(ENABLE_HOST_CPU_DEVICES)
  set(OCL_TARGETS "host")
  set(OCL_DRIVERS "basic pthreads")
  # TODO OCL_KERNEL_TARGET -> CPU_TARGET_TRIPLE
  # TODO OCL_KERNEL_TARGET_CPU -> OCL_KERNEL_TARGET_CPU
  set(OCL_KERNEL_TARGET "${LLC_TRIPLE}") #The kernel target triplet.
  set(OCL_KERNEL_TARGET_CPU "${LLC_HOST_CPU}") #The kernel target CPU variant.
  set(BUILD_BASIC 1)
  set(BUILD_PTHREAD 1)
endif()

# The accel device could be built by default, but it's implemented in C++,
# thus requires a C++ compiler, so let's not.
if(ENABLE_ACCEL_DEVICE)
  set(BUILD_ACCEL 1)
  set(OCL_DRIVERS "${OCL_DRIVERS} accel")
endif()

if(DEFINED EXTRA_OCL_TARGETS)
  set(OCL_TARGETS "${OCL_TARGETS} ${EXTRA_OCL_TARGETS}")
endif()


if(ENABLE_PROXY_DEVICE)
  # see doc/sphinx/source/proxy.rst
  if (ENABLE_ICD)
    message(FATAL_ERROR "Proxy driver cannot be used with ICD")
  endif()
  if (VISIBILITY_HIDDEN)
    message(FATAL_ERROR "Proxy driver cannot be used with VISIBILITY_HIDDEN")
  endif()
  if (ENABLE_PROXY_DEVICE_INTEROP)
    if (ANDROID)
      set(ENABLE_OPENGL_INTEROP 0 CACHE BOOL "gl" FORCE)
      set(ENABLE_EGL_INTEROP 1 CACHE BOOL "egl" FORCE)
    else()
      set(ENABLE_OPENGL_INTEROP 1 CACHE BOOL "gl" FORCE)
      set(ENABLE_EGL_INTEROP 0 CACHE BOOL "egl" FORCE)
    endif()

    if(ENABLE_OPENGL_INTEROP)
      set(CMAKE_REQUIRED_LIBRARIES "OpenCL")
      include(CheckFunctionExists)
      unset (ENABLE_CL_GET_GL_CONTEXT CACHE)
      CHECK_FUNCTION_EXISTS("clGetGLContextInfoKHR" ENABLE_CL_GET_GL_CONTEXT)
      unset(CMAKE_REQUIRED_LIBRARIES)
    endif()

  endif()
  set(BUILD_PROXY 1)
  set(OCL_DRIVERS "${OCL_DRIVERS} proxy")
endif()


####################################################################

# Determine which device drivers to build.

if(NOT DEFINED DEFAULT_ENABLE_TCE)

  set(HAVE_TCE 0)
  set(HAVE_TCEMC 0)

  # THESE are only used in makefile.am & scripts/pocl*
  set(TCE_TARGET_CLANG_FLAGS "" CACHE STRING "Extra parameters to Clang for TCE compilation.")
  set(TCE_TARGET_LLC_FLAGS "" CACHE STRING "Extra parameters to LLVM's llc for TCE compilation.")

  if(TCE_CONFIG AND TCECC AND TTASIM)

    message(STATUS "Found tcecc + tce-config + ttasim, testing setup")

    get_filename_component(TCE_BASEDIR "${TCE_CONFIG}" DIRECTORY)
    find_library(TCE_LIBS "tce" HINTS "${TCE_BASEDIR}/../lib" ENV PATH)
    if(NOT TCE_LIBS)
      execute_process(COMMAND "${TCE_CONFIG}" --libs OUTPUT_VARIABLE TCE_LIBS RESULT_VARIABLE RESV1 OUTPUT_STRIP_TRAILING_WHITESPACE)
    endif()
    execute_process(COMMAND "${TCE_CONFIG}" --includes OUTPUT_VARIABLE TCE_INCLUDES RESULT_VARIABLE RESV2 OUTPUT_STRIP_TRAILING_WHITESPACE)
    execute_process(COMMAND "${TCE_CONFIG}" --version OUTPUT_VARIABLE TCE_VERSION RESULT_VARIABLE RESV3 OUTPUT_STRIP_TRAILING_WHITESPACE)
    execute_process(COMMAND "${TCE_CONFIG}" --cxxflags OUTPUT_VARIABLE TCE_CXXFLAGS RESULT_VARIABLE RESV4 OUTPUT_STRIP_TRAILING_WHITESPACE)
    execute_process(COMMAND "${TCE_CONFIG}" --prefix OUTPUT_VARIABLE TCE_PREFIX RESULT_VARIABLE RESV5 OUTPUT_STRIP_TRAILING_WHITESPACE)
    execute_process(COMMAND "${TCE_CONFIG}" --llvm-config OUTPUT_VARIABLE TCE_LLVM_CONFIG RESULT_VARIABLE RESV6 OUTPUT_STRIP_TRAILING_WHITESPACE)
    execute_process(COMMAND "${TTASIM}" --help OUTPUT_VARIABLE TTASIM_HELP RESULT_VARIABLE RESV9)


    if(NOT (LLVM_CONFIG STREQUAL TCE_LLVM_CONFIG))
      message(WARNING "tce-config returned llvm-config is ${TCE_LLVM_CONFIG} but LLVM_CONFIG given to pocl is ${LLVM_CONFIG}")
    endif()

    if (RESV1 OR RESV2 OR RESV3 OR RESV4 OR RESV5)
      message(WARNING "tce-config: Nonzero exit status, disabling TCE")
    elseif (RESV9)
      message(WARNING "ttasim: Nonzero exit status, disabling TCE")
    else()

    string(STRIP "${TCE_LIBS}" TCE_LIBS)
    separate_arguments(TCE_LIBS)
    string(STRIP "${TCE_INCLUDES}" TCE_INCLUDES)
    separate_arguments(TCE_INCLUDES)
    string(STRIP "${TCE_CXXFLAGS}" TCE_CXXFLAGS)
    separate_arguments(TCE_CXXFLAGS)
    string(STRIP "${TCE_VERSION}" TCE_VERSION)
    string(STRIP "${TCE_PREFIX}" TCE_PREFIX)

    set(TCE_LIBS "${TCE_LIBS}" CACHE INTERNAL "tce-config --libs")
    set(TCE_INCLUDES "${TCE_INCLUDES}" CACHE INTERNAL "tce-config --includes")
    set(TCE_VERSION "${TCE_VERSION}" CACHE INTERNAL "tce-config --version")
    set(TCE_CXXFLAGS "${TCE_CXXFLAGS}" CACHE INTERNAL "tce-config --cxxflags")
    set(TCE_PREFIX "${TCE_PREFIX}" CACHE INTERNAL "tce-config --prefix")

    set(HAVE_TCE 1)
    if(TCE_VERSION MATCHES "trunk")
      set(HAVE_TCEMC 1)
    endif()

    endif()

  else()
    message(STATUS "Failed to find tcecc or tce-config, disabling TCE")
  endif()

  set(DEFAULT_ENABLE_TCE ${HAVE_TCE} CACHE INTERNAL "TCE available")
  set(DEFAULT_ENABLE_TCEMC ${HAVE_TCEMC} CACHE INTERNAL "TCEMC available")

endif()

setup_cached_var(ENABLE_TCE "TCE support"
  "Requested enabling TCE, but no usable TCE installation found !"
  "TCE is available, but requested disabling it")

if(ENABLE_TCE)
  set(OCL_DRIVERS "${OCL_DRIVERS} tce")
  set(OCL_TARGETS "${OCL_TARGETS} tce")
  if(DEFAULT_ENABLE_TCEMC)
    set(ENABLE_TCEMC 1)
    set(OCL_DRIVERS "${OCL_DRIVERS} tcemc") # TCEMC is a "superset" of TCE (lp:tce) features.
  endif()
  set(TCE_DEVICE_EXTENSIONS "cl_khr_byte_addressable_store cl_khr_fp16 cl_khr_spir")
  set(TEMP_EXT "${TCE_DEVICE_EXTENSIONS}")
  set(TCE_DEVICE_EXTENSION_DEFINES "")
  separate_arguments(TEMP_EXT)
  foreach(EXT ${TEMP_EXT})
    set(TCE_DEVICE_EXTENSION_DEFINES "${TCE_DEVICE_EXTENSION_DEFINES} -D${EXT}")
  endforeach()

  set(TCE_DEVICE_CL_VERSION "120")
  set(TCE_DEVICE_CL_STD "1.2")

  if("${LLVM_CXXFLAGS}" MATCHES "-fno-rtti")
    message(WARNING "TCE is enabled but your LLVM was not built with RTTI. You should rebuild LLVM with 'make REQUIRES_RTTI=1'. See the INSTALL file for more information.")
  endif()

else()
  set(ENABLE_TCEMC 0)
endif()

##########################################################

if(ENABLE_HSA)
  set(OCL_DRIVERS "${OCL_DRIVERS} hsa")
  if (HSAIL_ENABLED)
    set(OCL_TARGETS "${OCL_TARGETS} hsail64")
  endif()
  # this is for config.h

  set(HSA_DEVICE_EXTENSIONS "cl_khr_byte_addressable_store cl_khr_global_int32_base_atomics cl_khr_global_int32_extended_atomics cl_khr_local_int32_base_atomics cl_khr_local_int32_extended_atomics cl_khr_fp64 cl_khr_int64_base_atomics cl_khr_int64_extended_atomics")
  set(HSA_DEVICE_CL_VERSION "120")
  set(HSA_DEVICE_CL_STD "1.2")
  find_path(HAVE_HSA_EXT_AMD_H "hsa_ext_amd.h" HINTS "${HSA_INCLUDEDIR}" ENV PATH)
endif()

##########################################################

if (ENABLE_VULKAN)

  # Use FindVulkan module added with CMAKE 3.7
  if (NOT Vulkan_FOUND)
    if (NOT CMAKE_VERSION VERSION_LESS 3.7.0)
      find_package(Vulkan)
    endif()
    if (NOT Vulkan_FOUND)
      find_library(Vulkan_LIBRARY NAMES vulkan HINTS "$ENV{VULKAN_SDK}/lib" REQUIRED)
      find_path(Vulkan_INCLUDE_DIR NAMES vulkan.h PATH_SUFFIXES vulkan)
    endif()
    if (Vulkan_LIBRARY AND Vulkan_INCLUDE_DIR)
      set(Vulkan_FOUND ON CACHE INTERNAL "Vulkan found")
    endif()
  endif()

  if(NOT Vulkan_FOUND)
    message(FATAL_ERROR "Vulkan not found")
  else()
    find_program(CLSPV NAMES clspv HINTS ${CLSPV_DIR})
    find_program(CLSPV_REFLECTION NAMES clspv-reflection HINTS ${CLSPV_DIR})
    if (CLSPV AND CLSPV_REFLECTION)
      message(STATUS "using clspv")
      set(HAVE_CLSPV ON)
    else()
      message(STATUS "please provide a path to clspv & clspv-reflection binaries with -DCLSPV_DIR=...")
      set(HAVE_CLSPV OFF)
    endif()
    message(STATUS "Using Vulkan library ${Vulkan_LIBRARY} and Vulkan headers in ${Vulkan_INCLUDE_DIR}")
    message(WARNING "Note: Vulkan backend is highly experimental and very incomplete." )
    set(OCL_DRIVERS "${OCL_DRIVERS} vulkan")
    set(OCL_TARGETS "${OCL_TARGETS} vulkan")
  endif()
endif()

##########################################################

if(ENABLE_CUDA)

  if(NOT "${LLVM_ALL_TARGETS}" MATCHES "NVPTX")
    message(FATAL_ERROR "CUDA build requested but LLVM does not support NVPTX target!")
  endif()

  set(OCL_DRIVERS "${OCL_DRIVERS} cuda")
  set(OCL_TARGETS "${OCL_TARGETS} cuda")
  # this is for config.h
  # TODO unify with autotools
  set(BUILD_CUDA 1)

  set(CUDA_DEVICE_EXTENSIONS "cl_khr_byte_addressable_store cl_khr_global_int32_base_atomics cl_khr_global_int32_extended_atomics cl_khr_local_int32_base_atomics cl_khr_local_int32_extended_atomics cl_khr_fp64 cl_khr_int64_base_atomics cl_khr_int64_extended_atomics cl_nv_device_attribute_query")
  if(ENABLE_SPIR)
    set(CUDA_DEVICE_EXTENSIONS "${CUDA_DEVICE_EXTENSIONS} cl_khr_spir")
  endif()
  set(CUDA_DEVICE_CL_VERSION "120")
  set(CUDA_DEVICE_CL_STD "1.2")
endif()

##########################################################

message(STATUS "Building the following device drivers: ${OCL_DRIVERS}")

set(BUILDDIR "${CMAKE_BINARY_DIR}")
set(SRCDIR "${CMAKE_SOURCE_DIR}")

##########################################################

# Checks for library features.

if(NOT CMAKE_CROSSCOMPILING)
  # AC_C_BIGENDIAN
  include(TestBigEndian)
  TEST_BIG_ENDIAN(WORDS_BIGENDIAN)
else()
  # Set default as little-endian
  set(WORDS_BIGENDIAN 0)
endif()

##########################################################

if (ENABLE_LLVM AND NOT CMAKE_CROSSCOMPILING)
  CHECK_ALIGNOF("double16" "typedef double double16  __attribute__((__ext_vector_type__(16)));" ALIGNOF_DOUBLE16)
else()
  set(ALIGNOF_DOUBLE16 128)
endif()

if(ALIGNOF_DOUBLE16 LESS 128)
  set(ALIGNOF_DOUBLE16 128)
endif()

set(MAX_EXTENDED_ALIGNMENT "${ALIGNOF_DOUBLE16}")

##########################################################


string(TOUPPER "${CMAKE_BUILD_TYPE}" BTYPE)
if("${CMAKE_C_FLAGS_${BTYPE}}" MATCHES "DNDEBUG")
  set(POCL_ASSERTS_BUILD 0)
else()
  set(POCL_ASSERTS_BUILD 1)
endif()

##########################################################


# cmake docs:
# SOVERSION: What version number is this target.

# For shared libraries VERSION and SOVERSION can be used to specify the
#  build version and API version respectively. When building or installing
#  appropriate symlinks are created if the platform supports symlinks and
#  the linker  supports so-names. If only one of both is specified the
#  missing is assumed to have the same version number.
#
# For executables VERSION can be used to specify the build version.
# SOVERSION is ignored if NO_SONAME property is set. For shared libraries
# and executables on Windows the VERSION attribute is parsed to extract
#  a "major.minor" version number. These numbers are used as the
#  image version of the binary.

# cmake usage:
# SET_TARGET_PROPERTIES(pocl PROPERTIES SOVERSION 1.6.3 VERSION 4) ...



# The libtool library version string to use (passed to -version-info).
# See: http://www.nondot.org/sabre/Mirrored/libtool-2.1a/libtool_6.html
# libpocl.so should get only API additions as we are implementing a standard.
#
# The library version encodings into the library file name are platform
# dependent. Therefore we need to be a bit verbose here for the pocl.icd file
# creation to succeed (see Makefile.am).
# Chiefly, GNU differs from BSD, and others are untested. See e.g.
# http://en.opensuse.org/openSUSE%3aShared_library_packaging_policy#Versioning_schemes
#
# 0:0:0 == 0.6
# 1:0:0 == 0.7 (not backwards compatible with 0:0:0 due to the ICD)
# 2:0:1 == 0.8 (currently backwards compatible with 0.7, thus age = 1).
# 3:0:2 == 0.9 (currently backwards compatible with 0.7, thus age = 2).
# 4:0:3 == 0.10 (currently backwards compatible with 0.7, thus age = 3).
# 5:0:4 == 0.11 (currently backwards compatible with 0.7, thus age = 4).
# 6:0:5 == 0.12 (currently backwards compatible with 0.7, thus age = 5).
# 7:0:6 == 0.13 (currently backwards compatible with 0.7, thus age = 6).
# 8:0:7 == 0.14 (currently backwards compatible with 0.7, thus age = 7).
# pocl 1.0 bumped the API version:
# 2:0:0 == 1.0 (the libpocl.so will be named libpocl.so.2.0.X )
# 3:0:1 == 1.1 (the libpocl.so will be named libpocl.so.2.1.X )
# 4:0:2 == 1.2 (the libpocl.so will be named libpocl.so.2.2.X )
# 5:0:3 == 1.3 (the libpocl.so will be named libpocl.so.2.3.X )
# 6:0:4 == 1.4 (the libpocl.so will be named libpocl.so.2.4.X )
# 7:0:5 == 1.5 (the libpocl.so will be named libpocl.so.2.5.X )
# 8:0:6 == 1.6 (the libpocl.so will be named libpocl.so.2.6.X )
# 9:0:7 == 1.7 (the libpocl.so will be named libpocl.so.2.7.X )
# 10:0:8 == 1.8 (the libpocl.so will be named libpocl.so.2.8.X )
# 11:0:9 == 1.9 (the libpocl.so will be named libpocl.so.2.9.X )

set(LIB_CURRENT_VERSION 11)
set(LIB_REVISION_VERSION 0)
set(LIB_AGE_VERSION 9)

math(EXPR LIB_FIRST_VERSION "${LIB_CURRENT_VERSION} - ${LIB_AGE_VERSION}")

# libtool takes "c:r:a" arguments, but the result is "<lib>.so.(c-a).a.r"
# cmake has "build version" and "API version"
# these vars map libtool -> cmake
# for set_target_properties
set(LIB_BUILD_VERSION "${LIB_FIRST_VERSION}.${LIB_AGE_VERSION}.${LIB_REVISION_VERSION}")
set(LIB_API_VERSION "${LIB_FIRST_VERSION}")

# The kernel compiler opt plugin shared library, however, changes more
# drastically. Let's try to follow the similar 'current' numbering as
# the pocl host API library and perhaps tune the 'revision' and 'age' later.

math(EXPR KER_LIB_CURRENT_VERSION "${LIB_CURRENT_VERSION} + 7")
set(KERNEL_COMPILER_LIB_VERSION "${KER_LIB_CURRENT_VERSION}.0.0")

##########################################################

#TODO
# these vars are copies b/c tons of sources use BUILD_ICD etc
set(BUILD_ICD ${ENABLE_ICD})
set(BUILD_HSA ${ENABLE_HSA})
set(BUILD_VULKAN ${ENABLE_VULKAN})
set(TCE_AVAILABLE ${ENABLE_TCE})
set(TCEMC_AVAILABLE ${ENABLE_TCEMC})
set(_CL_DISABLE_HALF ${CL_DISABLE_HALF})

configure_file("config.h.in.cmake" "config.h.new" ESCAPE_QUOTES)
rename_if_different("${CMAKE_BINARY_DIR}/config.h.new" "${CMAKE_BINARY_DIR}/config.h")

configure_file("config2.h.in.cmake" "config2.h.new")
rename_if_different("${CMAKE_BINARY_DIR}/config2.h.new" "${CMAKE_BINARY_DIR}/config2.h")

configure_file("pocl_opencl.h.in.cmake" "pocl_opencl.h.new")
rename_if_different("${CMAKE_BINARY_DIR}/pocl_opencl.h.new" "${CMAKE_BINARY_DIR}/pocl_opencl.h")

include_directories("${CMAKE_BINARY_DIR}")

# This is used to generate the compiler feature detection header.
# Currently it's not enabled because it requires CMake > 3.x and
# also the autogenerated header needs some editing by hand
# (it errors on all compilers except gcc > 4 and clang > 3)
#
#
#include(WriteCompilerDetectionHeader)
#write_compiler_detection_header(
#  FILE "${CMAKE_BINARY_DIR}/compiler_features.h"
#  PREFIX POCL
#  COMPILERS GNU Clang
#  FEATURES
#    c_function_prototypes
#    c_restrict
#    c_static_assert
#    c_variadic_macros
#)

##########################################################

if(ENABLE_ICD)
  if(POCL_ICD_ABSOLUTE_PATH)
    set(CONTENT "${POCL_INSTALL_PUBLIC_LIBDIR}/$<TARGET_FILE_NAME:pocl>")
  else()
    set(CONTENT "$<TARGET_FILE_NAME:pocl>")
  endif()
  file(GENERATE OUTPUT "${CMAKE_BINARY_DIR}/pocl.icd" CONTENT "${CONTENT}" CONDITION 1)
  install(FILES "${CMAKE_BINARY_DIR}/pocl.icd"
         DESTINATION "${POCL_INSTALL_ICD_VENDORDIR}" COMPONENT "icd")

  # write icd file for pocl testing
  file(MAKE_DIRECTORY "${CMAKE_BINARY_DIR}/ocl-vendors")
  file(GENERATE OUTPUT "${CMAKE_BINARY_DIR}/ocl-vendors/pocl-tests.icd" CONTENT "$<TARGET_FILE:pocl>" CONDITION 1)

  cpack_add_component("icd")
  set("CPACK_DEBIAN_ICD_PACKAGE_NAME" "pocl-opencl-icd")
  list(APPEND CPACK_DEBIAN_ICD_PACKAGE_DEPENDS "libpocl2 (>= ${CPACK_PACKAGE_VERSION}~)")
  set(CPACK_DEBIAN_ICD_PACKAGE_PROVIDES "opencl-icd,opencl-icd-1.1-1,opencl-icd-1.2-1")
  set(CPACK_DEBIAN_ICD_PACKAGE_RECOMMENDS "poclcc")
endif()

if(ENABLE_ASAN OR ENABLE_LSAN)
  file(GENERATE OUTPUT "${CMAKE_BINARY_DIR}/lsan.supp" CONTENT "leak:${LLVM_SRC_ROOT}/lib/Support/Unix/Signals.inc")
  set(SAN_EXTRA "set(ENV{LSAN_OPTIONS} \"suppressions=${CMAKE_BINARY_DIR}/lsan.supp\")")
endif()

# pointing OCL_ICD_VENDORS to pocl doesn't make sense with the proxy driver;
# we want the driver to proxy to another OpenCL, so
# we must let ocl-icd load it
if(ENABLE_PROXY_DEVICE)
file(GENERATE OUTPUT "${CMAKE_BINARY_DIR}/CTestCustom.cmake" CONTENT "
  ${SAN_EXTRA}
  set(ENV{POCL_ENABLE_UNINIT} \"1\")
  set(ENV{POCL_BUILDING} \"1\")
")
else()
file(GENERATE OUTPUT "${CMAKE_BINARY_DIR}/CTestCustom.cmake" CONTENT "
  ${SAN_EXTRA}
  set(ENV{POCL_ENABLE_UNINIT} \"1\")
  set(ENV{POCL_BUILDING} \"1\")
  set(ENV{OCL_ICD_VENDORS} \"${CMAKE_BINARY_DIR}/ocl-vendors\")
")
endif()

##########################################################

if(UNIX)

  configure_file("${CMAKE_SOURCE_DIR}/pocl.pc.in.cmake" "${CMAKE_BINARY_DIR}/pocl.pc" @ONLY)
  install(FILES "${CMAKE_BINARY_DIR}/pocl.pc"
         DESTINATION "${POCL_INSTALL_PKGCONFIG_DIR}" COMPONENT "dev")

endif()

# For now always use a mirror copy of ocml, but allow overriding
# this path later to point to an out-of-tree copy.
set(OCML_SOURCE_DIR "${CMAKE_SOURCE_DIR}/lib/kernel/ocml")

#############################################################

add_subdirectory("include")

add_subdirectory("lib")

add_subdirectory("poclu")

# these are set in lib/cmakelists.txt
message(STATUS "OPENCL_LIBS: ${OPENCL_LIBS}")
message(STATUS "OPENCL_CFLAGS: ${OPENCL_CFLAGS}")

# for tests / examples
set(POCLU_LINK_OPTIONS poclu ${OPENCL_LIBS} ${LIBMATH})
message(STATUS "POCLU LINK OPTS: ${POCLU_LINK_OPTIONS}")

# poclcc bin
if(ENABLE_POCLCC)
    add_subdirectory("bin")
endif()


include(add_test_pocl)

if(ENABLE_TESTS)
    add_subdirectory("tests")

    # make check & make check_tier1
    add_custom_target(check COMMAND ${CMAKE_CTEST_COMMAND} "--output-on-failure" -j ${CORECOUNT} ${COMMAND_USES_TERMINAL})
    add_custom_target(check_tier1 COMMAND ${CMAKE_CTEST_COMMAND} "--output-on-failure" -L "'internal|piglit|PyOpenCL|conformance_suite_micro|CLBlast|shoc'" -j ${CORECOUNT} ${COMMAND_USES_TERMINAL})
endif()

if(ENABLE_EXAMPLES)
    add_subdirectory("examples")
endif()

set(CPACK_PACKAGE_DESCRIPTION_FILE "${CMAKE_SOURCE_DIR}/CPack.pocl.description.txt")
set(CPACK_PACKAGE_ICON "${CMAKE_SOURCE_DIR}/doc/www/img/pocl-80x60.png")
set(CPACK_PACKAGE_HOMEPAGE_URL "https://github.com/pocl/pocl")
set(CPACK_PACKAGE_CONTACT "https://github.com/pocl/pocl")
set(CPACK_PACKAGE_CHECKSUM "SHA512")
set(CPACK_RESOURCE_FILE_README "${CMAKE_SOURCE_DIR}/README")
set(CPACK_RESOURCE_FILE_LICENSE "${CMAKE_SOURCE_DIR}/COPYING")
set(CPACK_GENERATOR "DEB")

set("CPACK_DEBIAN_DEV_PACKAGE_NAME" "libpocl-dev")
list(APPEND CPACK_DEBIAN_DEV_PACKAGE_DEPENDS "libpocl2 (>= ${CPACK_PACKAGE_VERSION}~)")
set(CPACK_DEBIAN_DEV_PACKAGE_BREAKS "libpocl1-common (<< 0.13-9)")
set(CPACK_DEBIAN_DEV_PACKAGE_REPLACES "libpocl1-common (<< 0.13-9)")

set(CPACK_RPM_COMPONENT_INSTALL ON)
set(CPACK_DEB_COMPONENT_INSTALL ON)

include(CPack)

##########################################################

MESSAGE(STATUS " ")
MESSAGE(STATUS "*********************** SUMMARY ***************************")
MESSAGE(STATUS " ")
MESSAGE(STATUS "******* Directories:")
MESSAGE(STATUS " ")

MESSAGE(STATUS "CMAKE_INSTALL_PREFIX: ${CMAKE_INSTALL_PREFIX}")
MESSAGE(STATUS "POCL_INSTALL_CMAKE_CONFIG_DIR: ${POCL_INSTALL_CMAKE_CONFIG_DIR}")
MESSAGE(STATUS "POCL_INSTALL_ICD_VENDORDIR: ${POCL_INSTALL_ICD_VENDORDIR}")
MESSAGE(STATUS "POCL_INSTALL_OPENCL_HEADER_DIR: ${POCL_INSTALL_OPENCL_HEADER_DIR}")
MESSAGE(STATUS "POCL_INSTALL_PKGCONFIG_DIR: ${POCL_INSTALL_PKGCONFIG_DIR}")
MESSAGE(STATUS "POCL_INSTALL_PRIVATE_DATADIR: ${POCL_INSTALL_PRIVATE_DATADIR}")
MESSAGE(STATUS "POCL_INSTALL_PRIVATE_HEADER_DIR: ${POCL_INSTALL_PRIVATE_HEADER_DIR}")
MESSAGE(STATUS "POCL_INSTALL_PRIVATE_LIBDIR: ${POCL_INSTALL_PRIVATE_LIBDIR}")
MESSAGE(STATUS "POCL_INSTALL_PUBLIC_BINDIR: ${POCL_INSTALL_PUBLIC_BINDIR}")
MESSAGE(STATUS "POCL_INSTALL_PUBLIC_HEADER_DIR: ${POCL_INSTALL_PUBLIC_HEADER_DIR}")
MESSAGE(STATUS "POCL_INSTALL_PUBLIC_LIBDIR: ${POCL_INSTALL_PUBLIC_LIBDIR}")

MESSAGE(STATUS " ")

if (ENABLE_LLVM)
  MESSAGE(STATUS " ")
  MESSAGE(STATUS "******* LLVM Programs:")
  MESSAGE(STATUS " ")
  MESSAGE(STATUS "LLVM_CONFIG: ${LLVM_CONFIG}")
  MESSAGE(STATUS "LLVM_OPT: ${LLVM_OPT}")
  MESSAGE(STATUS "LLVM_LLC: ${LLVM_LLC}")
  MESSAGE(STATUS "LLVM_AS: ${LLVM_AS}")
  MESSAGE(STATUS "LLVM_LINK: ${LLVM_LINK}")
  MESSAGE(STATUS "LLVM_LLI: ${LLVM_LLI}")
  MESSAGE(STATUS "WITH_LLVM_CONFIG (User preferred llvm-config): ${WITH_LLVM_CONFIG}")
endif()

MESSAGE(STATUS " ")
MESSAGE(STATUS "******* Various Flags:")
MESSAGE(STATUS " ")

MESSAGE(STATUS "CL_DISABLE_HALF: ${CL_DISABLE_HALF}")
MESSAGE(STATUS "HAVE_CLOCK_GETTIME: ${HAVE_CLOCK_GETTIME}")
MESSAGE(STATUS "HAVE_GLEW: ${HAVE_GLEW}")
MESSAGE(STATUS "HAVE_LTTNG_UST: ${HAVE_LTTNG_UST}")
MESSAGE(STATUS "HOST_AS_FLAGS: ${HOST_AS_FLAGS}")
MESSAGE(STATUS "HOST_CLANG_FLAGS: ${HOST_CLANG_FLAGS}")
MESSAGE(STATUS "HOST_LD_FLAGS: ${HOST_LD_FLAGS}")
MESSAGE(STATUS "HOST_LLC_FLAGS: ${HOST_LLC_FLAGS}")
if (ENABLE_HSA)
  MESSAGE(STATUS "")
  MESSAGE(STATUS "HSA_INCLUDES: ${HSA_INCLUDES}")
  MESSAGE(STATUS "HSALIB: ${HSALIB}")
  MESSAGE(STATUS "HSAIL_ASM: ${HSAIL_ASM}")
endif()
if (ENABLE_VULKAN)
  MESSAGE(STATUS "")
  MESSAGE(STATUS "Vulkan includes: ${Vulkan_INCLUDE_DIR}")
  MESSAGE(STATUS "Vulkan library: ${Vulkan_LIBRARY}")
endif()
MESSAGE(STATUS "")
MESSAGE(STATUS "LIB_API_VERSION: ${LIB_API_VERSION}")
MESSAGE(STATUS "LIB_BUILD_VERSION: ${LIB_BUILD_VERSION}")
MESSAGE(STATUS "ICD_LD_FLAGS: ${ICD_LD_FLAGS}")

MESSAGE(STATUS "EXTRA_KERNEL_FLAGS: ${EXTRA_KERNEL_FLAGS}")
MESSAGE(STATUS "EXTRA_KERNEL_CXX_FLAGS: ${EXTRA_KERNEL_CXX_FLAGS}")
MESSAGE(STATUS "EXTRA_KERNEL_CL_FLAGS: ${EXTRA_KERNEL_CL_FLAGS}")
MESSAGE(STATUS "EXTRA_KERNEL_C_FLAGS: ${EXTRA_KERNEL_C_FLAGS}")

MESSAGE(STATUS "final KERNEL_CXX_FLAGS: ${KERNEL_CXX_FLAGS}")
MESSAGE(STATUS "final KERNEL_CL_FLAGS: ${KERNEL_CL_FLAGS}")
MESSAGE(STATUS "final KERNEL_C_FLAGS: ${KERNEL_C_FLAGS}")

if (ENABLE_LLVM)
  MESSAGE(STATUS "")
  MESSAGE(STATUS "CLANG_HAS_64B_MATH: ${CLANG_HAS_64B_MATH}")
  MESSAGE(STATUS "CLANG_HAS_128B_MATH: ${CLANG_HAS_128B_MATH}")
  MESSAGE(STATUS "CLANG_NEEDS_RTLIB: ${CLANG_NEEDS_RTLIB}")
  MESSAGE(STATUS "CLANG_MARCH_FLAG: ${CLANG_MARCH_FLAG}")
  MESSAGE(STATUS "CLANG_TARGET_OPTION: ${CLANG_TARGET_OPTION}")
  MESSAGE(STATUS "LLVM_VERSION: ${LLVM_VERSION}")
  MESSAGE(STATUS "LLVM_LIB_IS_SHARED: ${LLVM_LIB_IS_SHARED}")
  MESSAGE(STATUS "LLVM_HAS_RTTI: ${LLVM_HAS_RTTI}")
  MESSAGE(STATUS "LLVM_LIB_MODE: ${LLVM_LIB_MODE}")
  MESSAGE(STATUS "LLVM_ASSERTS_BUILD: ${LLVM_ASSERTS_BUILD}")
  MESSAGE(STATUS "LLVM_BUILD_MODE: ${LLVM_BUILD_MODE}")
  MESSAGE(STATUS "LLVM_CFLAGS: ${LLVM_CFLAGS}")
  MESSAGE(STATUS "LLVM_CXXFLAGS: ${LLVM_CXXFLAGS}")
  MESSAGE(STATUS "LLVM_CPPFLAGS: ${LLVM_CPPFLAGS}")
  MESSAGE(STATUS "LLVM_LDFLAGS: ${LLVM_LDFLAGS}")
  MESSAGE(STATUS "LLVM_LIBDIR: ${LLVM_LIBDIR}")
  MESSAGE(STATUS "LLVM_INCLUDEDIR: ${LLVM_INCLUDEDIR}")
  MESSAGE(STATUS "LLVM_SRC_ROOT: ${LLVM_SRC_ROOT}")
  MESSAGE(STATUS "LLVM_OBJ_ROOT: ${LLVM_OBJ_ROOT}")
  MESSAGE(STATUS "LLVM_INCLUDE_DIRS: ${LLVM_INCLUDE_DIRS}")
  MESSAGE(STATUS "LLVM_ALL_TARGETS: ${LLVM_ALL_TARGETS}")
  MESSAGE(STATUS "LLVM_HOST_TARGET: ${LLVM_HOST_TARGET}")
  MESSAGE(STATUS "LLC_TRIPLE: ${LLC_TRIPLE}")
  MESSAGE(STATUS "LLC_HOST_CPU: ${LLC_HOST_CPU}")
  MESSAGE(STATUS "")
endif()
MESSAGE(STATUS "MAX_EXTENDED_ALIGNMENT: ${MAX_EXTENDED_ALIGNMENT}")
MESSAGE(STATUS "OCL_KERNEL_TARGET: ${OCL_KERNEL_TARGET}")
MESSAGE(STATUS "OCL_KERNEL_TARGET_CPU: ${OCL_KERNEL_TARGET_CPU}")
MESSAGE(STATUS "HOST_DEVICE_ADDRESS_BITS: ${HOST_DEVICE_ADDRESS_BITS}")
if (ENABLE_TCE)
  MESSAGE(STATUS "")
  MESSAGE(STATUS "TCE_TARGET_CLANG_FLAGS: ${TCE_TARGET_CLANG_FLAGS}")
  MESSAGE(STATUS "TCE_TARGET_LLC_FLAGS: ${TCE_TARGET_LLC_FLAGS}")
  MESSAGE(STATUS "TCE_CXXFLAGS: ${TCE_CXXFLAGS}")
  MESSAGE(STATUS "TCE_INCLUDES: ${TCE_INCLUDES}")
  MESSAGE(STATUS "TCE_LIBS: ${TCE_LIBS}")
  MESSAGE(STATUS "TCE_VERSION: ${TCE_VERSION}")
  MESSAGE(STATUS "TCE_PREFIX: ${TCE_PREFIX}")
endif()
MESSAGE(STATUS "")

if (ENABLE_LLVM)
MESSAGE(STATUS "----------- -------------------------------- --------")
MESSAGE(STATUS "llvm libs libpocl will be linked to (POCL_LLVM_LIBS):")
MESSAGE(STATUS "${POCL_LLVM_LIBS}")
MESSAGE(STATUS "----------- -------------------------------- --------")
MESSAGE(STATUS "clang libs libpocl will be linked to (CLANG_LIBFILES):")
MESSAGE(STATUS "${CLANG_LIBFILES}")
MESSAGE(STATUS "----------- -------------------------------- --------")
MESSAGE(STATUS "system libs libpocl will be linked to (LLVM_SYSLIBS):")
MESSAGE(STATUS "${LLVM_SYSLIBS}")
MESSAGE(STATUS "----------- -------------------------------- --------")
endif()

MESSAGE(STATUS "******* Enabled features:")
MESSAGE(STATUS " ")

MESSAGE(STATUS "DEVELOPER_MODE: ${DEVELOPER_MODE}")
MESSAGE(STATUS "ENABLE_CONFORMANCE: ${ENABLE_CONFORMANCE}")
if(ARM)
MESSAGE(STATUS "ENABLE_FP64: ${ENABLE_FP64}")
endif()
MESSAGE(STATUS "ENABLE_IPO: ${ENABLE_IPO}")
MESSAGE(STATUS "ENABLE_ICD: ${ENABLE_ICD}")
MESSAGE(STATUS "ENABLE_TCE: ${ENABLE_TCE}")
MESSAGE(STATUS "ENABLE_TCEMC: ${ENABLE_TCEMC}")
MESSAGE(STATUS "ENABLE_HSA: ${ENABLE_HSA}")
MESSAGE(STATUS "ENABLE_CUDA: ${ENABLE_CUDA}")
MESSAGE(STATUS "ENABLE_VULKAN: ${ENABLE_VULKAN}")
MESSAGE(STATUS "ENABLE_ASAN (address sanitizer): ${ENABLE_ASAN}")
MESSAGE(STATUS "ENABLE_LSAN (leak sanitizer): ${ENABLE_LSAN}")
MESSAGE(STATUS "ENABLE_TSAN (thread sanitizer): ${ENABLE_TSAN}")
MESSAGE(STATUS "ENABLE_UBSAN (UB sanitizer): ${ENABLE_UBSAN}")
MESSAGE(STATUS "ENABLE_POCL_FLOAT_CONVERSION: ${ENABLE_POCL_FLOAT_CONVERSION}")
MESSAGE(STATUS "ENABLE_RELOCATION: ${ENABLE_RELOCATION}")
MESSAGE(STATUS "ENABLE_PROXY_DEVICE: ${ENABLE_PROXY_DEVICE}")
MESSAGE(STATUS "ENABLE_PROXY_DEVICE_INTEROP: ${ENABLE_PROXY_DEVICE_INTEROP}")
MESSAGE(STATUS "ENABLE_CL_GET_GL_CONTEXT: ${ENABLE_CL_GET_GL_CONTEXT}")
MESSAGE(STATUS "ENABLE_OPENGL_INTEROP: ${ENABLE_OPENGL_INTEROP}")
MESSAGE(STATUS "ENABLE_EGL_INTEROP: ${ENABLE_EGL_INTEROP}")
MESSAGE(STATUS "ENABLE_SLEEF: ${ENABLE_SLEEF}")
MESSAGE(STATUS "ENABLE_SPIR: ${ENABLE_SPIR}")
MESSAGE(STATUS "ENABLE_SPIRV: ${ENABLE_SPIRV}")
MESSAGE(STATUS "ENABLE_POCL_BUILDING: ${ENABLE_POCL_BUILDING}")
MESSAGE(STATUS "INSTALL_OPENCL_HEADERS (Install our headers): ${INSTALL_OPENCL_HEADERS}")
MESSAGE(STATUS "OCL_DRIVERS (Drivers built): ${OCL_DRIVERS}")
MESSAGE(STATUS "OCL_TARGETS (Targets built): ${OCL_TARGETS}")
MESSAGE(STATUS "ENABLE_LLVM: ${ENABLE_LLVM}")
if(PARALLEL_COMPILE_JOBS AND CMAKE_GENERATOR STREQUAL "Ninja")
  MESSAGE(STATUS "PARALLEL_COMPILE_JOBS: ${PARALLEL_COMPILE_JOBS}")
endif()
if(PARALLEL_LINK_JOBS AND CMAKE_GENERATOR STREQUAL "Ninja")
  MESSAGE(STATUS "PARALLEL_LINK_JOBS: ${PARALLEL_LINK_JOBS}")
endif()
MESSAGE(STATUS "POCL_ICD_ABSOLUTE_PATH: ${POCL_ICD_ABSOLUTE_PATH}")
MESSAGE(STATUS "POCL_ASSERTS_BUILD: ${POCL_ASSERTS_BUILD}")
MESSAGE(STATUS "TESTS_USE_ICD: ${TESTS_USE_ICD}")
MESSAGE(STATUS "Available testsuites: ${ALL_TESTSUITES}")
MESSAGE(STATUS "Enabled testsuites: ${ACTUALLY_ENABLED_TESTSUITES}")
MESSAGE(STATUS "Disabled testsuites: ${DISABLED_TESTSUITES}")
MESSAGE(STATUS "Testsuites are built from git master: ${EXAMPLES_USE_GIT_MASTER}")
MESSAGE(STATUS "Kernel caching: ${KERNEL_CACHE_DEFAULT}")
MESSAGE(STATUS "Kernel library CPU variants: ${KERNELLIB_HOST_CPU_VARIANTS}")
MESSAGE(STATUS "Kernel library distro build: ${KERNELLIB_HOST_DISTRO_VARIANTS}")
MESSAGE(STATUS "Use pocl custom memory allocator: ${USE_POCL_MEMMANAGER}")
MESSAGE(STATUS "L1d cacheline size: ${HOST_CPU_CACHELINE_SIZE}")
