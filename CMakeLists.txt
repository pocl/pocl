#=============================================================================
#   CMake build system files
#
#   Copyright (c) 2014-2017 pocl developers
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

cmake_minimum_required(VERSION 2.8.12 FATAL_ERROR)

project(pocl)

# Fix behavior of CMAKE_CXX_STANDARD when targeting macOS.
if(POLICY CMP0025)
  cmake_policy(SET CMP0025 NEW)
endif()

# don't allow implicit function declarations
if(UNIX)
  if((CMAKE_C_COMPILER_ID STREQUAL "GNU") OR
     (CMAKE_C_COMPILER_ID STREQUAL "Clang"))

    set(FORBIT_IMPLICIT_FUNCTIONS "-Werror=implicit-function-declaration")
    add_compile_options("${FORBIT_IMPLICIT_FUNCTIONS}")

  else()
    message(WARNING "Don't know how to forbid this compiler from allowing implicit function declarations.")
  endif()

endif()


set(MAJOR_VERSION 1)
set(MINOR_VERSION 2)
set(VERSION_SUFFIX "-pre")
set(VERSION_STRING ${MAJOR_VERSION}.${MINOR_VERSION}${VERSION_SUFFIX})
set(POCL_VERSION ${VERSION_STRING})

# required b/c SHARED libs defaults to ON while OBJECT defaults to OFF
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

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

if(HAVE_GIT AND (VERSION_SUFFIX MATCHES "pre"))
  execute_process(COMMAND "${GIT_CMD}" "rev-parse" "HEAD"
                  OUTPUT_VARIABLE GIT_COMMIT
                  RESULT_VARIABLE EXITCODE
                  WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
                  OUTPUT_STRIP_TRAILING_WHITESPACE)
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
  set(VERSION_SUFFIX "-pre/${GIT_DESCRIBE}")
  set(VERSION_STRING ${MAJOR_VERSION}.${MINOR_VERSION}${VERSION_SUFFIX})
  set(POCL_VERSION ${VERSION_STRING})
else()
  message(STATUS "No git and/or not a prerelease -> not adding git commit to version.")
endif()

##################################################################################

option(OCS_AVAILABLE "Online compiler support available to build pocl with. Default is available." ON)

option(BUILD_SHARED_LIBS "ON=Build shared libs, OFF=static libs" ON)

option(POCL_DEBUG_MESSAGES
  "Enable debug messages from pocl (useful for OpenCL developers), must be enabled at runtime, with env var POCL_DEBUG"
  ON)

option(ENABLE_HSA "Enable the HSA device driver for AMD GCN devices" OFF)

option(ENABLE_CUDA "Enable the CUDA device driver for NVIDIA devices" OFF)

option(KERNEL_CACHE_DEFAULT "Default value for the kernel compile cache. If disabled, pocl will still use the kernel cache, but will delete cachefiles on exit. You can still enable keeping the files it at runtime with an env var." ON)

option(POCL_ICD_ABSOLUTE_PATH "Use absolute path in pocl.icd" ON)

option(ENABLE_POCL_BUILDING "When OFF, env var POCL_BUILDING has no effect. Defaults to ON" ON)

#### these are mostly useful for pocl developers

option(DEVELOPER_MODE "This will SIGNIFICANTLY slow down pocl (but speed up its compilation). Only turn on if you know what you're doing." OFF)

option(USE_POCL_MEMMANAGER "Enables custom memory manager. Except for special circumstances, this should be disabled." OFF)

option(EXAMPLES_USE_GIT_MASTER "If enabled, some of the external testsuites in examples/ will try to use sources from Git master, instead of releases. This may result in failure to build or run the examples" OFF)

####

# currently only works with gcc as host compiler
if (CMAKE_C_COMPILER_ID STREQUAL "GNU")
  if (CMAKE_C_COMPILER_VERSION VERSION_GREATER "4.7.99")
    option(ENABLE_ASAN "Enable AddressSanitizer" OFF)
    option(ENABLE_TSAN "Enable ThreadSanitizer" OFF)
  else()
    set(ENABLE_ASAN OFF CACHE INTERNAL "Enable AddressSanitizer")
    set(ENABLE_TSAN OFF CACHE INTERNAL "Enable ThreadSanitizer")
  endif()

  if (CMAKE_C_COMPILER_VERSION VERSION_GREATER "4.8.99")
    option(ENABLE_UBSAN "Enable UBSanitizer" OFF)
  else()
    set(ENABLE_UBSAN OFF CACHE INTERNAL "Enable UBSanitizer")
  endif()

  if (CMAKE_C_COMPILER_VERSION VERSION_GREATER "5.0.99")
    option(ENABLE_LSAN "Enable LeakSanitizer" OFF)
  else()
    set(ENABLE_LSAN OFF CACHE INTERNAL "Enable LeakSanitizer")
  endif()

else()
  set(ENABLE_ASAN OFF CACHE INTERNAL "Enable AddressSanitizer")
  set(ENABLE_TSAN OFF CACHE INTERNAL "Enable ThreadSanitizer")
endif()

##########################################################

# POCL_DEVICE_ADDRESS_BITS
# TODO rename to HOST address bits
if(CMAKE_SIZEOF_VOID_P EQUAL 8)
  set(POCL_DEVICE_ADDRESS_BITS 64)
elseif(CMAKE_SIZEOF_VOID_P EQUAL 4)
  set(POCL_DEVICE_ADDRESS_BITS 32)
else()
  message(FATAL_ERROR "Cannot figure out POCL_DEVICE_ADDRESS_BITS")
endif()

##################################################################################

if(CMAKE_SYSTEM_PROCESSOR MATCHES "ppc")
  set(POWERPC 1)
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "mips")
  set(MIPS 1)
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "armv7")
  set(ARMV7 1)
  set(ARM32 1)
  set(ARM 1)
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "armv6")
  set(ARMV6 1)
  set(ARM32 1)
  set(ARM 1)
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
  set(ARM64 1)
  set(ARM 1)
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "(i.86|AMD64|x86_64)")
  set(X86 1)
  if(POCL_DEVICE_ADDRESS_BITS MATCHES "32")
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

if(UNIX)
  include(GNUInstallDirs)
else()
  if (WIN32)
    set(${CMAKE_INSTALL_LIBDIR} "lib")
    set(${CMAKE_INSTALL_DATADIR} "share")
    set(${CMAKE_INSTALL_INCLUDEDIR} "include")
    set(${CMAKE_INSTALL_BINDIR} "bin")
    message(STATUS "Setting installation destination on Windows to: ${CMAKE_INSTALL_PREFIX}")
  else()
    message(FATAL_ERROR "System not UNIX nor WIN32 - not implemented yet")
  endif()
endif()

# for libpocl.so
set(POCL_INSTALL_PUBLIC_LIBDIR "${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}" CACHE PATH "POCL public libdir")

# for llvmopencl.so
set(POCL_INSTALL_PRIVATE_LIBDIR "${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}/pocl" CACHE PATH "POCL private libdir")

# for pocl.icd
if(UNIX AND (NOT CMAKE_CROSSCOMPILING) AND (CMAKE_INSTALL_PREFIX STREQUAL "/usr"))
  set(POCL_INSTALL_ICD_VENDORDIR "/etc/OpenCL/vendors" CACHE PATH "POCL ICD file destination")
else()
  set(POCL_INSTALL_ICD_VENDORDIR "${CMAKE_INSTALL_PREFIX}/etc/OpenCL/vendors" CACHE PATH "POCL ICD file destination")
endif()

# for kernel-<target>.bc
set(POCL_INSTALL_PRIVATE_DATADIR "${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_DATADIR}/pocl" CACHE PATH "POCL private datadir")

# for poclu.h
set(POCL_INSTALL_PUBLIC_HEADER_DIR "${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_INCLUDEDIR}" CACHE PATH "POCL public header dir")

# for _kernel.h et al
set(POCL_INSTALL_PRIVATE_HEADER_DIR "${POCL_INSTALL_PRIVATE_DATADIR}/include" CACHE PATH "POCL private header dir")

# for pocl-standalone et al
set(POCL_INSTALL_PUBLIC_BINDIR "${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_BINDIR}" CACHE PATH "POCL public bindir")

# for PoclConfig.cmake & stuff
set(POCL_INSTALL_CMAKE_CONFIG_DIR "${POCL_INSTALL_PRIVATE_LIBDIR}" CACHE PATH   "Installation directory for CMake files")

# TODO maybe use output of pkg-config --variable=pc_path pkg-config ?
set(POCL_INSTALL_PKGCONFIG_DIR "${POCL_INSTALL_PUBLIC_LIBDIR}/pkgconfig" CACHE PATH "Destination for pocl.pc")

if(APPLE)
  set(CMAKE_MACOSX_RPATH ON)
  set(POCL_INSTALL_OPENCL_HEADER_DIR "${POCL_INSTALL_PUBLIC_HEADER_DIR}/OpenCL" CACHE PATH "POCL header dir for OpenCL headers")
else()
  set(POCL_INSTALL_OPENCL_HEADER_DIR "${POCL_INSTALL_PUBLIC_HEADER_DIR}/CL" CACHE PATH "POCL header dir for OpenCL headers")
endif()

######################################################################################

if(UNIX)
  find_package(PkgConfig MODULE REQUIRED)
endif()

######################################################################################

set(ANDROID_COMPILER 0)
if(CMAKE_C_COMPILER MATCHES "android")
  set(ANDROID_COMPILER 1)
  add_definitions(-DPOCL_ANDROID)
  add_definitions(-DPOCL_ANDROID_PREFIX="/data/data/org.pocl.libs/files")
endif()

######################################################################################

set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")
find_package( Hwloc )

if(NOT Hwloc_FOUND)
  message(FATAL_ERROR "hwloc package is required!")
endif()

if("${Hwloc_VERSION}" VERSION_LESS "1.0")
  message(FATAL_ERROR "Hwloc version must be >= 1.0 !")
endif()

message(STATUS "Hwloc_VERSION ${Hwloc_VERSION}")
message(STATUS "Hwloc_LDFLAGS ${Hwloc_LDFLAGS}")
message(STATUS "Hwloc_CFLAGS ${Hwloc_CFLAGS}")

######################################################################################

if(NOT HOST_CPU_CACHELINE_SIZE)

  set(CL_SIZE 0)
  if(UNIX)
    find_program(GETCONF "getconf")
    if(GETCONF)
      execute_process(COMMAND "getconf" "LEVEL1_DCACHE_LINESIZE"
                      RESULT_VARIABLE RES OUTPUT_VARIABLE CL_SIZE)
      if(RES)
        message(WARNING "getconf exited with nonzero status!")
        set(CL_SIZE 0)
      else()
        # getconf sometimes just returns zero
        if(NOT (CL_SIZE EQUAL 0))
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
  NAMES patch${CMAKE_EXECUTABLE_SUFFIX}
  HINTS ENV PATH
)

find_program(XARGS_EXEC
  NAMES xargs${CMAKE_EXECUTABLE_SUFFIX}
  HINTS ENV PATH
)

if(NOT PATCH_EXEC)
  message(FATAL_ERROR "Could not find patch command.")
endif()

if(NOT XARGS_EXEC)
  message(FATAL_ERROR "Could not find xargs command.")
endif()

######################################################################################

if (OCS_AVAILABLE)

  include(LLVM RESULT_VARIABLE RES)
  if(NOT RES)
    message(FATAL_ERROR "Could not load LLVM.cmake")
  endif()

  if(NOT DEFINED HOST_DEVICE_BUILD_HASH)
    if(KERNELLIB_HOST_CPU_VARIANTS STREQUAL "distro")
      set(HOST_DEVICE_BUILD_HASH "${LLC_TRIPLE}")
    else()
      set(HOST_DEVICE_BUILD_HASH "${LLC_TRIPLE}-${LLC_HOST_CPU}")
    endif()
  endif()

  if(ARM AND LLVM_3_9)
    message(FATAL_ERROR "pocl does not build on ARM with LLVM 3.9 unless is was patched. Try LLVM 3.8 or (preferably) 4.0+")
  endif()

else()

  if(NOT DEFINED HOST_DEVICE_BUILD_HASH)
    message(FATAL_ERROR "For compiler-less builds, you must define HOST_DEVICE_BUILD_HASH")
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
  set(CMAKE_REQUIRED_FLAGS "-std=c99 ${FORBIT_IMPLICIT_FUNCTIONS}")
  if (CMAKE_SYSTEM_NAME MATCHES "Linux")
    set(CMAKE_REQUIRED_LIBRARIES "rt")
  endif ()

  CHECK_SYMBOL_EXISTS("fork"
                      "sys/types.h;unistd.h"
                      HAVE_FORK)

  CHECK_SYMBOL_EXISTS("fsync"
                      "unistd.h"
                      HAVE_FSYNC)

  CHECK_SYMBOL_EXISTS("utime"
                      "sys/types.h;utime.h"
                      HAVE_UTIME)

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

  unset(CMAKE_REQUIRED_DEFINITIONS)
  unset(CMAKE_REQUIRED_FLAGS)

else()
  set(HAVE_CLOCK_GETTIME 0)
  set(HAVE_FDATASYNC 0)
  set(HAVE_FSYNC 0)
  set(HAVE_MKOSTEMPS 0)
  set(HAVE_MKSTEMPS 0)
  set(HAVE_MKDTEMP 0)
  set(HAVE_FUTIMENS 0)
  set(HAVE_FORK 0)
  set(HAVE_VFORK 0)
  set(HAVE_UTIME 0)
endif()


######################################################################################

if((DEFINED ENABLE_VECMATHLIB) AND (DEFINED ENABLE_SLEEF))
  if(ENABLE_SLEEF AND ENABLE_VECMATHLIB)
    message(FATAL_ERROR "requested to use both Vecmathlib and SLEEF - pick one!")
  endif()

  if((NOT ENABLE_SLEEF) AND (NOT ENABLE_VECMATHLIB))
    message(FATAL_ERROR "requested to use neither Vecmathlib nor SLEEF - pick one!")
  endif()
endif()

if(NOT DEFINED ENABLE_CONFORMANCE)
  if(((DEFINED ENABLE_VECMATHLIB) AND ENABLE_VECMATHLIB) OR
     ((DEFINED ENABLE_SLEEF) AND (NOT ENABLE_SLEEF)))
    set(DEFAULT_CONF OFF)
    set(DEFAULT_SLEEF OFF)
    set(DEFAULT_VML ON)
  else()
    set(DEFAULT_CONF ON)
    set(DEFAULT_SLEEF ON)
    set(DEFAULT_VML OFF)
  endif()
else()
  set(DEFAULT_CONF ${ENABLE_CONFORMANCE})
  if (ENABLE_CONFORMANCE)
    # requested conformant
    if((DEFINED ENABLE_VECMATHLIB) AND ENABLE_VECMATHLIB)
      message(FATAL_ERROR "requested to use Vecmathlib with enabled conformance")
    endif()
    if((DEFINED ENABLE_SLEEF) AND (NOT ENABLE_SLEEF))
      message(FATAL_ERROR "conformance needs enabled SLEEF")
    endif()
    set(DEFAULT_SLEEF ON)
    set(DEFAULT_VML OFF)
  else()
    # requested non-conformant
    set(DEFAULT_SLEEF ${ENABLE_SLEEF})
    set(DEFAULT_VML ${ENABLE_VECMATHLIB})
    # at least one
    if((NOT DEFAULT_SLEEF) AND (NOT DEFAULT_VML))
      set(DEFAULT_VML ON)
      set(DEFAULT_SLEEF OFF)
    endif()
  endif()

endif()

option(ENABLE_CONFORMANCE "Enable conformance to OpenCL standard. Disabling this may enable slightly faster kernel library functions (at a price of range/precision). Note that enabling this does not guarantee conformance (depends on hardware). Incompatible with Vecmathlib. Defaults to ON" ${DEFAULT_CONF})

option(ENABLE_SLEEF "Use SLEEF for kernel library, mutually exclusive with ENABLE_VECMATHLIB" ${DEFAULT_SLEEF})

option(ENABLE_VECMATHLIB "Use vecmathlib for kernel library, mutually exclusive with ENABLE_SLEEF" ${DEFAULT_VML})

if((NOT CLANGXX_WORKS) AND ENABLE_VECMATHLIB)
  message(WARNING "Disabling vecmathlib because clang++ doesn't seem to work!")
  set(ENABLE_VECMATHLIB OFF)
  set(ENABLE_VECMATHLIB OFF CACHE BOOL "VML" FORCE)
  set(ENABLE_SLEEF ON)
  set(ENABLE_SLEEF ON CACHE BOOL "SLEEF" FORCE)
endif()

# vecmathlib does not compile with fp16 currently
if(ENABLE_VECMATHLIB AND (NOT CL_DISABLE_HALF))
  message(STATUS "Half available, but disabling half support since vecmathlib is enabled.")
  set(CL_DISABLE_HALF 1)
  set(CL_DISABLE_HALF 1 CACHE BOOL "Disable cl_khr_fp16 because fp16 is not supported")
endif()

######################################################################################

set(SPIRV OFF)

if((NOT LLVM_OLDER_THAN_5_0) AND CLANG_IS_PATCHED_FOR_SPIR_CC AND X86 AND (NOT KERNELLIB_HOST_CPU_VARIANTS STREQUAL "distro"))

  option(ENABLE_SPIR "Enable SPIR support (default ON when available)" ON)

  if(ENABLE_SPIR)

    message(WARNING "SPIR support is available but highly experimental; use at your own risk.")

    if(LLVM_SPIRV AND (EXISTS "${LLVM_SPIRV}"))
      message(WARNING "SPIR-V support enabled but highly experimental; you must use a llvm-spirv "
                      "converter that produces bitcode FOR YOUR LLVM VERSION. "
                      "E.g. if you're compiling pocl against LLVM 5 then using Khronos' "
                      "llvm-spirv based on LLVM 3.6 branch WILL NOT WORK.")
      set(SPIRV ON)
    endif()

    if(CMAKE_SIZEOF_VOID_P EQUAL 8)
      set(SPIR_HOST_TARGET_TRIPLE "spir64-unknown-unknown")
    elseif(CMAKE_SIZEOF_VOID_P EQUAL 4)
      set(SPIR_HOST_TARGET_TRIPLE "spir-unknown-unknown")
    else()
      message(FATAL_ERROR "Cannot figure out POCL_DEVICE_ADDRESS_BITS")
    endif()
  endif()

else()
  set(ENABLE_SPIR OFF CACHE INTERNAL "SPIR enabled" FORCE)
endif()

set(ENABLE_SPIRV ${SPIRV} CACHE INTERNAL "SPIR-V enabled" FORCE)

######################################################################################

if(CLANG_HAS_RTLIB_128)
  option(ENABLE_POCL_FLOAT_CONVERSION "Enable use of pocl's own float-to-decimal conversion code in OpenCL printf(). Defaults to OFF (uses snprintf from C library). Requires compiler-rt." OFF)
else()
  set(ENABLE_POCL_FLOAT_CONVERSION OFF CACHE INTERNAL "rtlib")
endif()

unset(RTLIB_FLAG)
if(ENABLE_POCL_FLOAT_CONVERSION)
  set(RTLIB_FLAG "-DENABLE_POCL_FLOAT_CONVERSION")
endif()

######################################################################################

# for kernel code, disable PIC & stack protector
#
# it seems PIC and stack-protector defaults somehow depend on
# clang build type or environment. PIC causes problems with
# constant addrspace variables, and stack protector likely slows
# down the kernels, so it needs to be determined whether it's worth
# the trouble.
set(DEFAULT_KERNEL_CL_FLAGS  "-x cl -fno-stack-protector -fno-PIC ${RTLIB_FLAG}")
set(DEFAULT_KERNEL_C_FLAGS "-xc -std=c11 -D__CBUILD__ -fno-math-errno -fno-stack-protector -fno-PIC ${RTLIB_FLAG}")
set(DEFAULT_KERNEL_CXX_FLAGS "-xc++ -std=c++11 -fno-stack-protector -fno-PIC ${RTLIB_FLAG}")

if(ENABLE_VECMATHLIB)
  option(USE_VECMATHLIB_BUILTINS_ONLY "Use only __builtin_* functions in the kernel library." OFF)
  set(DEFAULT_KERNEL_CXX_FLAGS "${DEFAULT_KERNEL_CXX_FLAGS} -DVML_NO_IOSTREAM ${CLANGXX_STDLIB} ")
  if(USE_VECMATHLIB_BUILTINS_ONLY)
    set(DEFAULT_KERNEL_CL_FLAGS "${DEFAULT_KERNEL_CL_FLAGS} -DPOCL_VECMATHLIB_BUILTIN ")
    set(DEFAULT_KERNEL_CXX_FLAGS "${DEFAULT_KERNEL_CXX_FLAGS} -DPOCL_VECMATHLIB_BUILTIN ")
  endif()
endif()

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
    set(DEFAULT_HOST_LD_FLAGS "-dylib -w -lm")
  elseif(ANDROID_COMPILER)
    set(DEFAULT_HOST_LD_FLAGS "-L/system/lib/ -shared -ldl -lc -lm /system/lib/crtbegin_so.o /system/lib/crtend_so.o")
  else()
    if(ENABLE_POCL_FLOAT_CONVERSION)
      set(DEFAULT_HOST_LD_FLAGS "-shared -lm --rtlib=compiler-rt")
    else()
      set(DEFAULT_HOST_LD_FLAGS "-shared -lm")
    endif()
  endif()
  set(LIBMATH "-lm")
elseif(WIN32)
  set(LIBMATH)
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

if (OCS_AVAILABLE)

  option(SINGLE_LLVM_LIB "When on, tries to link pocl to the single big libLLVM before falling back to LLVM_LIBFILES)." ON)

  if (DEFINED STATIC_LLVM)
    message(AUTHOR_WARNING "STATIC_LLVM option was renamed, it wasn't really ensuring anything was static. Please see the SINGLE_LLVM_LIB option")
  endif()

  if(NOT SINGLE_LLVM_LIB)
    message(STATUS "Linking LLVM to LLVM_LIBFILES")
    set(POCL_LLVM_LIBS ${LLVM_LIBFILES})
  else()
    message(STATUS "Trying to link LLVM to the single big libLLVM")
    find_library(LLVM_SHARED_LIB_FILE NAMES "LLVM-${LLVM_VERSION}" "LLVM" PATHS "${LLVM_LIBDIR}" NO_DEFAULT_PATH)
    if(LLVM_SHARED_LIB_FILE AND EXISTS "${LLVM_SHARED_LIB_FILE}")
      message(STATUS " .. OK, using ${LLVM_SHARED_LIB_FILE}")
      set(POCL_LLVM_LIBS "${LLVM_SHARED_LIB_FILE}")
    else()
      message(STATUS "single big libLLVM library not found (Probably because LLVM is built with cmake). Falling back to linking libpocl to LLVM_LIBFILES")
      set(POCL_LLVM_LIBS ${LLVM_LIBFILES})
      set(SINGLE_LLVM_LIB OFF CACHE BOOL "single big libLLVM")
    endif()
  endif()

endif()

######################################################################################

include_directories("fix-include" "include")

######################################################################################

if(WIN32)
    message(STATUS "Using LoadLibrary/FreeLibrary in Windows, libltdl not needed.")
else()
  find_library(LTDL_LIB "ltdl")
  find_file(LTDL_H "ltdl.h")
  if(LTDL_LIB AND LTDL_H)
    message(STATUS "ltdl found")

    get_filename_component(LTDL_H_INCLUDE_DIR "${LTDL_H}" DIRECTORY)
    string(FIND "${CMAKE_C_IMPLICIT_INCLUDE_DIRECTORIES}" "${LTDL_H_INCLUDE_DIR}" LTPOSITION)
    # include the directory of ltdl.h, if its not in the default system include dirs
    # also when cross-compiling this includes <cross-compile-root>/usr/include, which screws things up
    if((LTPOSITION LESS "0") AND (NOT CMAKE_CROSSCOMPILING))
      include_directories("${LTDL_H_INCLUDE_DIR}")
    endif()

  else()
    message(FATAL_ERROR "Could not find LTDL library!")
  endif()
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

if(NOT MSVC)
  pkg_check_modules(LTTNG_UST lttng-ust>=2.7)
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
  if (NOT CMAKE_CROSSCOMPILING)
    pkg_check_modules(OCL_ICD ocl-icd>=1.3)
  endif()
  if (CMAKE_CROSSCOMPILING OR (NOT OCL_ICD_FOUND))
    find_path(OCL_ICD_INCLUDE_DIR
      NAMES
        ocl_icd.h
    )
    find_library(OCL_ICD_LIBRARIES
      NAMES
        OpenCL
    )
    if(OCL_ICD_INCLUDE_DIR AND OCL_ICD_LIBRARIES)
      set(OCL_ICD_FOUND 1)
    endif()
  endif()

  if(OCL_ICD_FOUND)

    set(HAVE_OCL_ICD 1 CACHE INTERNAL "ICL library is ocl-icd")
    set(OPENCL_FOUND 1 CACHE INTERNAL "opencl ICD/library found")
    # duh, why doesn't ocl-icd set this in its .pc file ??
    if (CMAKE_CROSSCOMPILING)
      set(OPENCL_LIBRARIES "${OCL_ICD_LIBRARIES}" CACHE INTERNAL "opencl ICD/library paths")
    else()
      set(OPENCL_LIBRARIES "OpenCL" CACHE INTERNAL "opencl ICD/library paths")
    endif()
    set(DEFAULT_ENABLE_ICD 1 CACHE INTERNAL "ICD loader availability")

  else()

    set(HAVE_OCL_ICD 0 CACHE INTERNAL "ICL library is ocl-icd")
    # fallback to other ICD loaders
    message(STATUS "ocl-icd not found -> trying fallback ICD implementations")
    pkg_check_modules(OPENCL OpenCL>=1.2)
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
  set(POCL_LIBRARY_NAME "pocl")
else()
  set(TESTS_USE_ICD 0)
  set(POCL_LIBRARY_NAME "OpenCL")
endif()

message(STATUS "Run tests with ICD: ${TESTS_USE_ICD}")

######################################################################################

if(DEFINED INSTALL_OPENCL_HEADERS)
  message(STATUS "Install POCL's OpenCL headers: ${INSTALL_OPENCL_HEADERS}")
else() # Undefined = auto -> check
  find_file(OPENCL_H opencl.h PATH_SUFFIXES CL OpenCL)
  if(OPENCL_H)
    message(STATUS "OpenCL.h found, NOT installing our headers")
    set(IOH 0)
  else()
    message(STATUS "OpenCL.h not found, installing our headers")
    set(IOH 1)
  endif()
  set(INSTALL_OPENCL_HEADERS ${IOH} CACHE BOOL "Install POCL's OpenCL headers. (Ones from Kronos should be installed instead)")
endif()

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
  # 32bit arm doesn't always support doubles,
  # need to enable FP64 manually with LLVM < 4.0
  if(LLVM_OLDER_THAN_4_0)
    option(ENABLE_FP64 "Enable FP64 on ARM32 - disabled by default for LLVM < 4" OFF)
  else()
    option(ENABLE_FP64 "Enable FP64 on ARM32 - if you have at least VFP support for doubles, you can leave it ON" ON )
  endif()
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

set(HOST_DEVICE_CL_VERSION "120")
set(HOST_DEVICE_CL_STD "1.2")

# define it here, b/c we'll need these both at runtime and buildtime
if(X86 OR ARM)
  set(HOST_DEVICE_EXTENSIONS "cl_khr_byte_addressable_store cl_khr_global_int32_base_atomics cl_khr_global_int32_extended_atomics cl_khr_local_int32_base_atomics cl_khr_local_int32_extended_atomics cl_khr_3d_image_writes")
else()
  # set some conservative defaults
  set(HOST_DEVICE_EXTENSIONS "cl_khr_global_int32_base_atomics cl_khr_local_int32_base_atomics cl_khr_3d_image_writes")
endif()

if(ENABLE_SPIR)
  set(HOST_DEVICE_EXTENSIONS "${HOST_DEVICE_EXTENSIONS} cl_khr_spir")
endif()

if(NOT CL_DISABLE_HALF)
  set(HOST_DEVICE_EXTENSIONS "${HOST_DEVICE_EXTENSIONS} cl_khr_fp16")
endif()

# must not be defined in HOST_DEVICE_EXTENSIONS list, because
# this extension doesn't exist in official extension list
# there is "cles_khr_int64" which indicates int64 support for embedded profiles
set(HOST_DEVICE_EXTENSION_DEFINES "-Dcl_khr_int64")

if(X86)
  set(HOST_DEVICE_EXTENSIONS "${HOST_DEVICE_EXTENSIONS} cl_khr_fp64 cl_khr_int64_base_atomics cl_khr_int64_extended_atomics")
endif()

if(ENABLE_FP64)
  set(HOST_DEVICE_EXTENSIONS "${HOST_DEVICE_EXTENSIONS} cl_khr_fp64")
  set(_CL_DISABLE_DOUBLE 0)
else()
  set(_CL_DISABLE_DOUBLE 1)
endif()

set(TEMP_EXT "${HOST_DEVICE_EXTENSIONS}")
separate_arguments(TEMP_EXT)
set(TEMP_CLEXT "-Xclang -cl-ext=-all,")
foreach(EXT ${TEMP_EXT})
  set(HOST_DEVICE_EXTENSION_DEFINES "${HOST_DEVICE_EXTENSION_DEFINES} -D${EXT}")
  set(TEMP_CLEXT "${TEMP_CLEXT}+${EXT},")
endforeach()

if (NOT LLVM_OLDER_THAN_4_0)
  set(HOST_DEVICE_EXTENSION_DEFINES "${HOST_DEVICE_EXTENSION_DEFINES} ${TEMP_CLEXT}")
endif()

if(NOT DEFINED KERNELLIB_HOST_CPU_VARIANTS)
  set(KERNELLIB_HOST_CPU_VARIANTS "native")
# else TODO test cpu list for unknown values
endif()

set(KERNELLIB_HOST_DISTRO_VARIANTS 0)
if(KERNELLIB_HOST_CPU_VARIANTS STREQUAL "distro")
  if(X86_64 OR I386)
    if(HOST_CPU_FORCED)
      message(FATAL_ERROR "Cannot build with CPU autodetection distro variants build, and enforce LLC_HOST_CPU at the same time. Please pick one")
    endif()
    set(KERNELLIB_HOST_CPU_VARIANTS sse2 ssse3 sse41 avx avx_f16c avx_fma4 avx2)
    if(NOT LLVM_OLDER_THAN_3_9)
      list(APPEND KERNELLIB_HOST_CPU_VARIANTS avx512)
    endif()
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

set(OCL_TARGETS "host")
set(OCL_DRIVERS "basic pthreads")

# TODO OCL_KERNEL_TARGET -> CPU_TARGET_TRIPLE
# TODO OCL_KERNEL_TARGET_CPU -> OCL_KERNEL_TARGET_CPU
set(OCL_KERNEL_TARGET "${LLC_TRIPLE}") #The kernel target triplet.
set(OCL_KERNEL_TARGET_CPU "${LLC_HOST_CPU}") #The kernel target CPU variant.

####################################################################

# Determine which device drivers to build.

if(NOT DEFINED DEFAULT_ENABLE_TCE)

  set(HAVE_TCE 0)
  set(HAVE_TCEMC 0)

  if (NOT WITH_TCE)
    set(WITH_TCE ENV PATH)
  endif()

  # THESE are only used in makefile.am & scripts/pocl*
  set(TCE_TARGET_CLANG_FLAGS "" CACHE STRING "Extra parameters to Clang for TCE compilation.")
  set(TCE_TARGET_LLC_FLAGS "" CACHE STRING "Extra parameters to LLVM's llc for TCE compilation.")

  find_program(TCE_CONFIG NAMES "tce-config" HINTS ${WITH_TCE})
  find_program(TCECC NAMES "tcecc" HINTS ${WITH_TCE})
  find_program(TTASIM NAMES "ttasim" HINTS ${WITH_TCE})

  if(TCE_CONFIG AND TCECC AND TTASIM)

    message(STATUS "Found tcecc + tce-config + ttasim, testing setup")

    get_filename_component(TCE_BASEDIR "${TCE_CONFIG}" DIRECTORY)
    find_library(TCE_LIBS "tce" HINTS "${TCE_BASEDIR}/../lib" ENV PATH)
    if(NOT TCE_LIBS)
      execute_process(COMMAND "${TCE_CONFIG}" --libs OUTPUT_VARIABLE TCE_LIBS RESULT_VARIABLE RESV1)
    endif()
    execute_process(COMMAND "${TCE_CONFIG}" --includes OUTPUT_VARIABLE TCE_INCLUDES RESULT_VARIABLE RESV2)
    execute_process(COMMAND "${TCE_CONFIG}" --version OUTPUT_VARIABLE TCE_VERSION RESULT_VARIABLE RESV3)
    execute_process(COMMAND "${TCE_CONFIG}" --cxxflags OUTPUT_VARIABLE TCE_CXXFLAGS RESULT_VARIABLE RESV4)
    execute_process(COMMAND "${TCE_CONFIG}" --prefix OUTPUT_VARIABLE TCE_PREFIX RESULT_VARIABLE RESV5)
    execute_process(COMMAND "${TTASIM}" --help OUTPUT_VARIABLE TTASIM_HELP RESULT_VARIABLE RESV9)

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
  set(TCE_DEVICE_EXTENSIONS "cl_khr_byte_addressable_store cl_khr_global_int32_base_atomics cl_khr_global_int32_extended_atomics cl_khr_local_int32_base_atomics cl_khr_local_int32_extended_atomics cl_khr_fp16 cl_khr_spir")
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
  set(OCL_TARGETS "${OCL_TARGETS} hsail64")
  # this is for config.h

  set(HSA_DEVICE_EXTENSIONS "cl_khr_byte_addressable_store cl_khr_global_int32_base_atomics cl_khr_global_int32_extended_atomics cl_khr_local_int32_base_atomics cl_khr_local_int32_extended_atomics cl_khr_fp64 cl_khr_int64_base_atomics cl_khr_int64_extended_atomics")
  set(HSA_DEVICE_CL_VERSION "200")
  set(HSA_DEVICE_CL_STD "2.0")
  find_path(HAVE_HSA_EXT_AMD_H "hsa_ext_amd.h" HINTS "${HSA_INCLUDEDIR}" ENV PATH)
endif()

##########################################################

if(ENABLE_CUDA)

  # Require LLVM 4.0 or newer
  if ("${LLVM_MAJOR}" STRLESS "4")
    message(FATAL_ERROR "The CUDA backend requires LLVM 4.0 or newer")
  endif()

  set(OCL_DRIVERS "${OCL_DRIVERS} cuda")
  set(OCL_TARGETS "${OCL_TARGETS} cuda")
  # this is for config.h
  # TODO unify with autotools
  set(BUILD_CUDA 1)

  set(CUDA_DEVICE_EXTENSIONS "cl_khr_byte_addressable_store cl_khr_global_int32_base_atomics cl_khr_global_int32_extended_atomics cl_khr_local_int32_base_atomics cl_khr_local_int32_extended_atomics cl_khr_fp64 cl_khr_int64_base_atomics cl_khr_int64_extended_atomics")
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

if (OCS_AVAILABLE)
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

set(LIB_CURRENT_VERSION 4)
set(LIB_REVISION_VERSION 0)
set(LIB_AGE_VERSION 2)

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
set(TCE_AVAILABLE ${ENABLE_TCE})
set(TCEMC_AVAILABLE ${ENABLE_TCEMC})
set(_CL_DISABLE_HALF ${CL_DISABLE_HALF})
set(PACKAGE_VERSION "${POCL_VERSION}")

configure_file("config.h.in.cmake" "config.h.new" ESCAPE_QUOTES)
rename_if_different("${CMAKE_BINARY_DIR}/config.h.new" "${CMAKE_BINARY_DIR}/config.h")

configure_file("config2.h.in.cmake" "config2.h.new")
rename_if_different("${CMAKE_BINARY_DIR}/config2.h.new" "${CMAKE_BINARY_DIR}/config2.h")

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
         DESTINATION "${POCL_INSTALL_ICD_VENDORDIR}")

  # write icd file for pocl testing
  file(MAKE_DIRECTORY "${CMAKE_BINARY_DIR}/ocl-vendors")
  file(GENERATE OUTPUT "${CMAKE_BINARY_DIR}/ocl-vendors/pocl-tests.icd" CONTENT "$<TARGET_FILE:pocl>" CONDITION 1)
endif()

if(ENABLE_ASAN OR ENABLE_LSAN)
  file(GENERATE OUTPUT "${CMAKE_BINARY_DIR}/lsan.supp" CONTENT "leak:${LLVM_SRC_ROOT}/lib/Support/Unix/Signals.inc")
  set(SAN_EXTRA "set(ENV{LSAN_OPTIONS} \"suppressions=${CMAKE_BINARY_DIR}/lsan.supp\")")
endif()

file(GENERATE OUTPUT "${CMAKE_BINARY_DIR}/CTestCustom.cmake" CONTENT "
  ${SAN_EXTRA}
  set(ENV{POCL_BUILDING} \"1\")
  set(ENV{OCL_ICD_VENDORS} \"${CMAKE_BINARY_DIR}/ocl-vendors\")
")

##########################################################

if(UNIX)

  configure_file("${CMAKE_SOURCE_DIR}/pocl.pc.in.cmake" "${CMAKE_BINARY_DIR}/pocl.pc" @ONLY)
  install(FILES "${CMAKE_BINARY_DIR}/pocl.pc"
         DESTINATION "${POCL_INSTALL_PKGCONFIG_DIR}")

endif()

#############################################################

add_subdirectory("include")

add_subdirectory("lib")

# these are set in lib/cmakelists.txt
message(STATUS "OPENCL_LIBS: ${OPENCL_LIBS}")
message(STATUS "OPENCL_CFLAGS: ${OPENCL_CFLAGS}")

# for tests / examples
set(POCLU_LINK_OPTIONS ${OPENCL_LIBS} ${LIBMATH} "poclu")
message(STATUS "POCLU LINK OPTS: ${POCLU_LINK_OPTIONS}")

# poclcc bin
add_subdirectory("bin")

include(add_test_pocl)

if (OCS_AVAILABLE)
  add_subdirectory("tests")
  add_subdirectory("examples")
endif()

# make check & make check_tier1

add_custom_target(check COMMAND ${CMAKE_CTEST_COMMAND} "--output-on-failure" -j ${CORECOUNT} ${COMMAND_USES_TERMINAL})
add_custom_target(check_tier1 COMMAND ${CMAKE_CTEST_COMMAND} "--output-on-failure" -L "'internal|amdsdk_30|piglit|PyOpenCL|conformance_suite_micro'" -j ${CORECOUNT} ${COMMAND_USES_TERMINAL})

##########################################################

MESSAGE(STATUS " ")
MESSAGE(STATUS "*********************** SUMMARY ***************************")
MESSAGE(STATUS " ")
MESSAGE(STATUS "******* Directories:")
MESSAGE(STATUS " ")

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

if (OCS_AVAILABLE)
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

if (OCS_AVAILABLE)
  MESSAGE(STATUS "")
  MESSAGE(STATUS "CLANG_HAS_RTLIB: ${CLANG_HAS_RTLIB}")
  MESSAGE(STATUS "CLANG_HAS_RTLIB_128: ${CLANG_HAS_RTLIB_128}")
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
MESSAGE(STATUS "POCL_DEVICE_ADDRESS_BITS: ${POCL_DEVICE_ADDRESS_BITS}")
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

if (OCS_AVAILABLE)
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
MESSAGE(STATUS "ENABLE_ICD: ${ENABLE_ICD}")
MESSAGE(STATUS "ENABLE_TCE: ${ENABLE_TCE}")
MESSAGE(STATUS "ENABLE_TCEMC: ${ENABLE_TCEMC}")
MESSAGE(STATUS "ENABLE_HSA: ${ENABLE_HSA}")
MESSAGE(STATUS "ENABLE_CUDA: ${ENABLE_CUDA}")
MESSAGE(STATUS "ENABLE_ASAN (address sanitizer): ${ENABLE_ASAN}")
MESSAGE(STATUS "ENABLE_LSAN (leak sanitizer): ${ENABLE_LSAN}")
MESSAGE(STATUS "ENABLE_TSAN (thread sanitizer): ${ENABLE_TSAN}")
MESSAGE(STATUS "ENABLE_UBSAN (UB sanitizer): ${ENABLE_UBSAN}")
MESSAGE(STATUS "ENABLE_POCL_FLOAT_CONVERSION: ${ENABLE_POCL_FLOAT_CONVERSION}")
MESSAGE(STATUS "ENABLE_VECMATHLIB: ${ENABLE_VECMATHLIB}")
MESSAGE(STATUS "ENABLE_SLEEF: ${ENABLE_SLEEF}")
MESSAGE(STATUS "ENABLE_SPIR: ${ENABLE_SPIR}")
MESSAGE(STATUS "ENABLE_SPIRV: ${ENABLE_SPIRV}")
MESSAGE(STATUS "ENABLE_POCL_BUILDING: ${ENABLE_POCL_BUILDING}")
MESSAGE(STATUS "INSTALL_OPENCL_HEADERS (Install our headers): ${INSTALL_OPENCL_HEADERS}")
MESSAGE(STATUS "OCL_DRIVERS (Drivers built): ${OCL_DRIVERS}")
MESSAGE(STATUS "OCL_TARGETS (Targets built): ${OCL_TARGETS}")
MESSAGE(STATUS "OCS_AVAILABLE: ${OCS_AVAILABLE}")
MESSAGE(STATUS "POCL_ICD_ABSOLUTE_PATH: ${POCL_ICD_ABSOLUTE_PATH}")
MESSAGE(STATUS "POCL_ASSERTS_BUILD: ${POCL_ASSERTS_BUILD}")
MESSAGE(STATUS "SINGLE_LLVM_LIB: ${SINGLE_LLVM_LIB}")
MESSAGE(STATUS "TESTS_USE_ICD: ${TESTS_USE_ICD}")
MESSAGE(STATUS "Available testsuites: ${ALL_TESTSUITES}")
MESSAGE(STATUS "Enabled testsuites: ${ACTUALLY_ENABLED_TESTSUITES}")
MESSAGE(STATUS "Disabled testsuites: ${DISABLED_TESTSUITES}")
MESSAGE(STATUS "Testsuites are built from git master: ${EXAMPLES_USE_GIT_MASTER}")
MESSAGE(STATUS "Kernel caching: ${KERNEL_CACHE_DEFAULT}")
MESSAGE(STATUS "Kernel library CPU variants: ${KERNELLIB_HOST_CPU_VARIANTS}")
MESSAGE(STATUS "Kernel library distro build: ${KERNELLIB_HOST_DISTRO_VARIANTS}")
MESSAGE(STATUS "Use fake address space IDs: ${POCL_USE_FAKE_ADDR_SPACE_IDS}")
MESSAGE(STATUS "Use pocl custom memory allocator: ${USE_POCL_MEMMANAGER}")
MESSAGE(STATUS "L1d cacheline size: ${HOST_CPU_CACHELINE_SIZE}")
