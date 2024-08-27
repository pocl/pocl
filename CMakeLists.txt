#=============================================================================
#   CMake build system files
#
#   Copyright (c) 2014-2023 pocl developers
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

cmake_minimum_required(VERSION 3.12 FATAL_ERROR)
cmake_policy(SET CMP0067 OLD)

project(pocl)
set(CMAKE_PROJECT_DESCRIPTION "pocl is a portable OpenCl runtime.")

MESSAGE(STATUS "CMAKE_SYSTEM_NAME: ${CMAKE_SYSTEM_NAME}")
MESSAGE(STATUS "CMAKE_SYSTEM_PROCESSOR: ${CMAKE_SYSTEM_PROCESSOR}")
MESSAGE(STATUS "CMAKE_LIBRARY_ARCHITECTURE: ${CMAKE_LIBRARY_ARCHITECTURE}")

set(LATEST_KNOWN_CXX_STD_VERSION "20")
set(SUPPORTED_CXX_STD_VERSION "11")

option(ENABLE_LATEST_CXX_STD "Upgrade C++ standard version to ${LATEST_KNOWN_CXX_STD_VERSION}. Required to get rid of unused variables warnings in compilers not supporting [[gnu::*]] attributes. Can bring other benefits, including performance and efficiency ones. Before a pull request build with this disabled." OFF)
if(ENABLE_LATEST_CXX_STD)
	set(CMAKE_CXX_STANDARD "${LATEST_KNOWN_CXX_STD_VERSION}")
else()
	set(CMAKE_CXX_STANDARD "${SUPPORTED_CXX_STD_VERSION}")
endif()
set(POCL_CMAKE_CXX_STANDARD ${CMAKE_CXX_STANDARD})
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

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

    check_c_compiler_flag("-Wincompatible-pointer-types" HAVE_WARN_INCOMPATIBLE_POINTER_TYPES)
    set(FORBID_IMPLICIT_FUNCTIONS "-Werror=implicit-function-declaration")
    if (HAVE_WARN_INCOMPATIBLE_POINTER_TYPES)
        set(FORBID_IMPLICIT_FUNCTIONS ${FORBID_IMPLICIT_FUNCTIONS} "-Werror=incompatible-pointer-types")
    endif()
    add_compile_options("$<$<COMPILE_LANGUAGE:C>:${FORBID_IMPLICIT_FUNCTIONS}>")

    add_compile_options("-Wno-ignored-attributes")

  else()
    message(WARNING "Don't know how to forbid this compiler from allowing implicit function declarations.")
  endif()
endif()

set(MAJOR_VERSION 6)
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

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

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

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
  set(LLVM_VERIFY_MODULE_DEFAULT 1)
else()
  set(LLVM_VERIFY_MODULE_DEFAULT 0)
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

option(ENABLE_LLVM "Build pocl with LLVM. Default is ON." ON)

option(STATIC_LLVM "If ON, link to static LLVM libraries. OFF (default) = link to shared LLVM libraries." OFF)

option(BUILD_SHARED_LIBS "ON=Build shared libs, OFF=static libs" ON)

option(POCL_DEBUG_MESSAGES
  "Enable debug messages from pocl (useful for OpenCL developers), must be enabled at runtime, with env var POCL_DEBUG"
  ON)

option(ENABLE_LOADABLE_DRIVERS "Enable drivers to be dlopen()-ed at pocl runtime, instead of being linked into libpocl" ON)

option(ENABLE_HSA "Enable the HSA base profile runtime device driver" OFF)

option(ENABLE_CUDA "Enable the CUDA device driver for NVIDIA devices" OFF)

option(ENABLE_CUDNN "Enable the CUDNN for the CUDA device driver, requires CUDA" OFF)

option(ENABLE_VULKAN "Experimental and incomplete driver that uses the Vulkan API for controlling the device. Please refer to the user manual for the status and open tasks" OFF)

option(ENABLE_LEVEL0 "Experimental and incomplete driver that uses the Level Zero API for controlling the device. Please refer to the user manual for the status and open tasks" OFF)

option(ENABLE_TBB_DEVICE "Enable the Intel TBB device driver." OFF)

option(ENABLE_HOST_CPU_DEVICES "Add host CPUs as OpenCL devices (cpu & cpu-minimal)." ON)

option(ENABLE_HOST_CPU_DEVICES_OPENMP "Enables OpenMP support for Host CPU devices (cpu driver only, cpu-minimal driver remains single-threaded)" OFF)

option(ENABLE_PROXY_DEVICE "Enable proxy driver for proxying to another OpenCL implementation" OFF)

option(ENABLE_PROXY_DEVICE_INTEROP "Enable OpenGL- or EGL-interop with the proxy driver" OFF)

option(ENABLE_ALMAIF_DEVICE "Enable the generic hardware accelerator device driver." OFF)

option(KERNEL_CACHE_DEFAULT "Default value for the kernel compile cache. If disabled, pocl will still use kernel cache for intermediate compilation files, but will clean up them on exit. You can still enable keeping the files it at runtime with an env var." ON)

option(ENABLE_REMOTE_SERVER "Build the 'pocld' server daemon for the remote driver" OFF)

option(ENABLE_REMOTE_CLIENT "Build the client library of the remote driver" OFF)

option(POCL_ICD_ABSOLUTE_PATH "Use absolute path in pocl.icd" ON)

option(ENABLE_POCL_BUILDING "When OFF, env var POCL_BUILDING has no effect. Defaults to ON" ON)

option(ENABLE_PRINTF_IMMEDIATE_FLUSH "[currently only applies to CPU drivers] if enabled, printf is flushed immediately when encountered, instead of after all NDRange workgroups are finished" ON)

option(ENABLE_VSOCK "Enable vsock transport in the remote driver" OFF)

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

option(DEVELOPER_MODE "This will SIGNIFICANTLY reduce PoCL's performance, but speeds up its compilation for faster development-test cycles. Only turn on if you know what you're doing." OFF)

option(USE_POCL_MEMMANAGER "Enables custom memory manager. Except for special circumstances, this should be disabled." OFF)

option(EXAMPLES_USE_GIT_MASTER "If enabled, some of the external testsuites in examples/ will try to use sources from Git master, instead of releases. This may result in failure to build or run the examples" OFF)

option(ENABLE_POCLCC "Build poclcc. Defaults to ON" ON)

option(ENABLE_TESTS "Build tests. Defaults to ON" ON)

option(ENABLE_EXAMPLES "Build examples. Defaults to ON" ON)

option(ENABLE_RDMA "Enable usage of RDMA libraries for memory allocations. Requires libRDMAcm and libverbs" OFF)

option(ENABLE_TRAFFIC_MONITOR "Enable network traffic monitoring & logging in remote device" OFF)

OPTION(RENAME_POCL "rename PoCL's ocl functions to PO<ocl_function>. Allows an user to call both PoCL and another ocl application" OFF)

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

include(ProcessorCount)
ProcessorCount(HOST_CPU_CORECOUNT)
if(HOST_CPU_CORECOUNT LESS 1)
  set(HOST_CPU_CORECOUNT 1)
endif()
message(STATUS "Host CPU cores: ${HOST_CPU_CORECOUNT}")


######################################################################################

function(chmod FILE EXEC)
  if(EXEC)
    set(PERMS FILE_PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE)
  else()
    #set(PERMS FILE_PERMISSIONS OWNER_READ OWNER_WRITE GROUP_READ WORLD_READ)
    return()
  endif()
  if(CMAKE_VERSION VERSION_LESS 3.19.0)
    find_program(CHMOD_PROG chmod)
    execute_process(COMMAND "${CHMOD_PROG}" "0755" ${FILE})
  else()
    file(CHMOD "${FILE}" ${PERMS})
  endif()
endfunction()


function(rename_if_different SRC DST EXEC)
  if(EXISTS "${DST}")
    file(MD5 "${SRC}" OLD_MD5)
    file(MD5 "${DST}" NEW_MD5)
    if(NOT OLD_MD5 STREQUAL NEW_MD5)
      file(RENAME "${SRC}" "${DST}")
      chmod("${DST}" ${EXEC})
    endif()
  else()
    file(RENAME "${SRC}" "${DST}")
    chmod("${DST}" ${EXEC})
  endif()
endfunction()

######################################################################################

# Recent versions of CMake can make use of Ninja's console pool to avoid
# buffering the output of particular commands.
set(COMMAND_USES_TERMINAL USES_TERMINAL)

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

set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")

option(HARDENING_ENABLE "Enable hardening against various attacks. May worsen performance" OFF)
if(HARDENING_ENABLE)
  include(Hardening)
else()
  function(harden target)
  endfunction()
endif()

find_package(PkgConfig MODULE)

if(NOT DEFINED DEFAULT_ENABLE_HWLOC)
  find_package(Hwloc)

  if(NOT Hwloc_FOUND)
   message(STATUS "hwloc package not found")
   set(DEFAULT_ENABLE_HWLOC OFF CACHE INTERNAL "default hwloc")
  else()
     if("${Hwloc_VERSION}" VERSION_LESS "1.0")
       message(FATAL_ERROR "Hwloc version must be >= 1.0 !")
     endif()

     message(STATUS "Hwloc_VERSION ${Hwloc_VERSION}")
     message(STATUS "Hwloc_LIBRARIES ${Hwloc_LIBRARIES}")
     message(STATUS "Hwloc_INCLUDE_DIRS ${Hwloc_INCLUDE_DIRS}")

     set(DEFAULT_ENABLE_HWLOC ON CACHE INTERNAL "default hwloc")
  endif()
endif()

option(ENABLE_HWLOC "Enable Portable Hardware Locality software package"
        ${DEFAULT_ENABLE_HWLOC})
setup_cached_var(ENABLE_HWLOC "Using hwloc"
        "Requested build with hwloc, but no hwloc found!"
        "Hwloc found, but requested build without it")

include(sanitizers)

######################################################################################

if(ENABLE_TBB_DEVICE)
  # related documentation: https://github.com/oneapi-src/oneTBB/tree/tbb_2020/cmake#binary-package-integration
  find_package(TBB REQUIRED)
  if(TARGET TBB::tbb)
    set(TBB_IMPORTED_TARGETS TBB::tbb)
  elseif(NOT TBB_FOUND AND PKG_CONFIG_FOUND)
    message(STATUS "searching for TBB with pkg-config")
    pkg_check_modules(TBB tbb>=2015)
    if(TBB_FOUND)
      set(TBB_IMPORTED_TARGETS "${TBB_LIBRARIES}" CACHE STRING "TBB libraries" FORCE)
    endif()
  endif()

  if(NOT TBB_INCLUDE_DIRS)
    get_target_property(TBB_INCLUDE_DIRS TBB::tbb INTERFACE_INCLUDE_DIRECTORIES)
  endif()
  if(NOT TBB_LIBRARIES)
    get_target_property(TBB_LIBRARIES TBB::tbb IMPORTED_LOCATION_RELEASE)
  endif()

  if(NOT TBB_FOUND)
    message(FATAL_ERROR "Can't find TBB libraries")
  endif()

  message(STATUS "Found TBB library: ${TBB_LIBRARIES} with include dirs: ${TBB_INCLUDE_DIRS}" )

  # The tbb device depends on the pthread device which depends on the basic device. ENABLE_HOST_CPU_DEVICES enables both.
  set(ENABLE_HOST_CPU_DEVICES 1)
endif()

if(ENABLE_HOST_CPU_DEVICES_OPENMP)
  find_package(OpenMP)
  if(NOT OpenMP_CXX_FOUND)
    message(FATAL_ERROR "ENABLE_HOST_CPU_DEVICES_OPENMP specified but OpenMP not found.")
  endif()
endif()

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

  include(LLVM RESULT_VARIABLE RES)
  if(NOT RES)
    message(FATAL_ERROR "Could not load LLVM.cmake")
  endif()

  if(ENABLE_HOST_CPU_DEVICES)
  if(NOT DEFINED HOST_DEVICE_BUILD_HASH)
      set(HOST_DEVICE_BUILD_HASH "${LLC_TRIPLE}")
  endif()

  if(INTEL_SDE_AVX512)
    set(HOST_CPU_FORCED 1 CACHE INTERNAL "CPU is forced by user" FORCE)
    set(SELECTED_HOST_CPU "skylake-avx512" CACHE STRING "The Host CPU to use with llc" FORCE)
  endif()
  endif()

  #
  # LLVM_OPAQUE_POINTERS
  #
  # llvm has changed from typed pointers to opaque pointers, their guidance is
  #
  # From https://llvm.org/docs/OpaquePointers.html, version support
  # LLVM 14: Supports all necessary APIs for migrating to opaque pointers and deprecates/removes
  #          incompatible APIs. However, using opaque pointers in the optimization pipeline is not fully
  #          supported. This release can be used to make out-of-tree code compatible with opaque
  #          pointers, but opaque pointers should not be enabled in production.
  #
  # LLVM 15: Opaque pointers are enabled by default. Typed pointers are still available, but only
  #          supported on a best-effort basis and may be untested.
  #
  # LLVM 16: Only opaque pointers will be supported. Typed pointers will not be supported.
  #
  # Up to llvm 14 including, ENABLE_LLVM_OPAQUE_POINTERS is force-set to OFF
  # For llvm 15 ENABLE_LLVM_OPAQUE_POINTERS is a CMake option, defaulting to OFF,
  # Post llvm 15, ENABLE_LLVM_OPAQUE_POINTERS is force-set to ON.
  #    Clang 16 accepts -Xclang -no-opaque-pointers but silently produces bitcode with opaque pointers.
  # reason is that 1) it's unfinished 2) for properly handling images, we need opaque types,
  # which will only be available in LLVM 16+
  # TODO: This should be temporary. After the opaque pointer conversion in PoCL is finished,
  # we should remove this macro and use LLVM_OLDER_THAN_X macros only, like in the rest
  # of the code, since this macro adds another layer of complexity.
  if(LLVM_VERSION VERSION_LESS_EQUAL 14.0)
    set(ENABLE_LLVM_OPAQUE_POINTERS OFF CACHE INTERNAL "llvm opaque pointers" FORCE)
  elseif(LLVM_VERSION VERSION_EQUAL 15.0)
    if(ENABLE_LEVEL0)
      set(ENABLE_LLVM_OPAQUE_POINTERS OFF CACHE INTERNAL "llvm opaque pointers" FORCE)
    else()
      option(ENABLE_LLVM_OPAQUE_POINTERS "Handle the change to llvm opaque pointers." ON)
    endif()
  else()
    set(ENABLE_LLVM_OPAQUE_POINTERS ON CACHE INTERNAL "llvm opaque pointers" FORCE)
  endif()
  if (ENABLE_LLVM_OPAQUE_POINTERS)
    set(LLVM_OPAQUE_POINTERS 1)
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

if(UNIX)
  include(CheckCSourceCompiles)
  include(CheckSymbolExists)

  # don't allow implicit function declarations
  set(CMAKE_REQUIRED_FLAGS "-std=c99 ${FORBID_IMPLICIT_FUNCTIONS}")
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
  if(ENABLE_VSOCK)
    find_file(
      HAVE_LINUX_VSOCK_H
      NAMES
        linux/vm_sockets.h
      PATHS
        /usr/include
    )

    if(NOT HAVE_LINUX_VSOCK_H)
      message(FATAL_ERROR "Could not find vsock header file. Please \
        make sure that the kernel is configured with vsock support.
    ")
    endif()
  endif()

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

    if(DEVELOPER_MODE)
      set(IPO 0)
    else()
      include(CheckIPOSupported)
      check_ipo_supported(RESULT IPO OUTPUT IPO_OUTPUT)
    endif()
    set(DEFAULT_ENABLE_IPO ${IPO} CACHE BOOL "IPO" FORCE)

    message(STATUS "Compiler supports IPO: ${DEFAULT_ENABLE_IPO}")
    #message(STATUS "IPO check message: ${IPO_OUTPUT}")
  endif()
endif()

setup_cached_var(ENABLE_IPO "Enable Link-Time Optimization (IPO) while building pocl runtime"
  "Requested build with IPO, but IPO is not available"
  "IPO available, but requested build without it")

if(HAVE_VALGRIND)
  option(ENABLE_VALGRIND "Enable valgrind support (this may slow down PoCL)" OFF)
else()
  set(ENABLE_VALGRIND OFF CACHE BOOL "valgrind is not available" FORCE)
endif()

######################################################################################

option(ENABLE_SLEEF "Use SLEEF for kernel library" ON)

option(ENABLE_CONFORMANCE "Enable conformance to OpenCL standard. \
  Enabling this option this does not guarantee conformance (depends on hardware), \
  but CMake will give errors if options that conflict with conformance \
are used. It also disables advertising incomplete extensions." OFF)

option(ENABLE_UNFINISHED_EXTENSIONS "Advertise extensions in device queries which \
might work partially, but have known unfinished features." OFF)

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

unset(OPAQUE_PTR_FLAGS)
if(LLVM_VERSION VERSION_EQUAL 15.0)
  if(LLVM_OPAQUE_POINTERS)
    set(OPAQUE_PTR_FLAGS "-Xclang -opaque-pointers")
  else()
    set(OPAQUE_PTR_FLAGS "-Xclang -no-opaque-pointers")
  endif()
endif()

######################################################################################

# for kernel code, disable PIC & stack protector
#
# it seems PIC and stack-protector defaults somehow depend on
# clang build type or environment. PIC causes problems with
# constant addrspace variables, and stack protector likely slows
# down the kernels, so it needs to be determined whether it's worth
# the trouble.
#
# no-jump-tables is to avoid LLVM optimization passes introducing switch tables,
# which the variable uniformity analysis cannot analyze
set(DEFAULT_KERNEL_CL_FLAGS  "-xcl -fno-stack-protector -fPIC -fno-jump-tables ${FLOATCONV_FLAG} ${OPAQUE_PTR_FLAGS}")
set(DEFAULT_KERNEL_C_FLAGS "-xc -std=c11 -D__CBUILD__ -fno-math-errno -fno-stack-protector -fPIC -fno-jump-tables ${FLOATCONV_FLAG} ${OPAQUE_PTR_FLAGS}")
set(DEFAULT_KERNEL_CXX_FLAGS "-xc++ -std=c++11 -fno-stack-protector -fPIC -fno-jump-tables ${FLOATCONV_FLAG} ${OPAQUE_PTR_FLAGS}")


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

# drivers that consume SPIRV directly don't need LLVM or LLVM-SPIRV;
# those that consume SPIRV indirectly need both, but it might be useful to disable LLVM-SPIRV translator
if(ENABLE_LLVM AND LLVM_SPIRV AND (EXISTS "${LLVM_SPIRV}"))
  option(ENABLE_SPIRV "Enable SPIR-V support (default ON when available)" ON)
else()
  set(ENABLE_SPIRV OFF CACHE INTERNAL "SPIR-V enabled" FORCE)
endif()

if(ENABLE_SPIRV)
  # required for the wrapper generator
  find_package(Python3 REQUIRED COMPONENTS Interpreter)
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
set(PTHREAD_LIBRARY Threads::Threads)

######################################################################################
# LTTNG
option(ENABLE_LTTNG "build pocl with the LTTng tracing option" OFF)
if(ENABLE_LTTNG)
  if(NOT UNIX)
    message(FATAL_ERROR "ENABLE_LTTNG enabled, but is only available on Unix")
  endif()

  if(PKG_CONFIG_EXECUTABLE)
    pkg_check_modules(LTTNG_UST lttng-ust>=2.7)
  endif()
  if(LTTNG_UST_FOUND)
    set(HAVE_LTTNG_UST 1)
  else()
    message(FATAL_ERROR "ENABLE_LTTNG enabled, but could not find library")
  endif()

else()
  set(HAVE_LTTNG_UST 0)
endif()

######################################################################################

if(ENABLE_RDMA)
  find_package(RDMAcm MODULE REQUIRED)
  find_package(Verbs MODULE REQUIRED)
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


  # Need to avoid finding the Apple OpenCL framework
  set(CMAKE_FIND_FRAMEWORK NEVER CACHE STRING "Find frameworks on macOS")

  if (NOT OCL_ICD_FOUND)
    find_path(OCL_ICD_INCLUDE_DIRS
      NAMES
        ocl_icd.h
    )
  elseif("${OCL_ICD_LIBRARIES}" STREQUAL "")
    # OCL_ICD_LIBRARIES is set to empty because of a bug in ocl-icd.pc
    unset(OCL_ICD_LIBRARIES CACHE)
  endif()

  # Find OCL_ICD_LIBRARIES with the hint from pkg-config
  find_library(OCL_ICD_LIBRARIES
    NAMES
      OpenCL
    HINTS
      ${OCL_ICD_LIBDIR}
  )

  if(OCL_ICD_INCLUDE_DIRS AND OCL_ICD_LIBRARIES)
    set(OCL_ICD_FOUND 1)
  endif()

  if(OCL_ICD_FOUND)

    set(HAVE_OCL_ICD 1 CACHE INTERNAL "ICL library is ocl-icd")
    set(OPENCL_FOUND 1 CACHE INTERNAL "opencl ICD/library found")

    set(OPENCL_LIBRARIES "${OCL_ICD_LIBRARIES}" CACHE INTERNAL "ocl-icd library")
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

if (APPLE AND VISIBILITY_HIDDEN)
  add_compile_definitions("CL_API_CALL=__attribute__ ((visibility (\"default\")))")
endif()

if(ENABLE_ICD OR ENABLE_PROXY_DEVICE OR ANDROID)
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

if(OPENCL_H AND ENABLE_ICD AND (NOT HAVE_OCL_ICD_30_COMPATIBLE))
  # if system-wide headers are present, ICD is enabled and older than 3.0 -> use the system-wide headers
  # TBD we're assuming a sane combination of header & ICD versions
  message(STATUS "opencl.h found (${OPENCL_H}) and ICD <3.0 enabled, using system headers")
  set(HAVE_OPENCL_H ON)
  set_expr(HAVE_OPENCL_HPP OPENCL_HPP)
  set(IOH OFF)
else()
  # if system-wide OpenCL headers are not found, we must use PoCL's
  # if ICD is disabled, we might as well compile with our own headers
  # if ICD is 3.0, force the use of PoCL's own 3.0 headers -> enables some additional tests
  message(STATUS "opencl.h not found or ICD disabled or ICD >= 3.0, using our headers")
  set(HAVE_OPENCL_H OFF)
  set(HAVE_OPENCL_HPP OFF)
  # install headers only if system-wide headers not found
  if(OPENCL_H)
    set(IOH OFF)
  else()
    set(IOH ON)
  endif()
endif()

if(NOT DEFINED INSTALL_OPENCL_HEADERS)
  option(INSTALL_OPENCL_HEADERS "Install POCL's OpenCL headers. (Ones from Khronos should be installed instead)" ${IOH})
endif()

add_definitions(-DCL_USE_DEPRECATED_OPENCL_1_0_APIS -DCL_USE_DEPRECATED_OPENCL_1_1_APIS
                -DCL_USE_DEPRECATED_OPENCL_1_2_APIS -DCL_USE_DEPRECATED_OPENCL_2_0_APIS
                -DCL_USE_DEPRECATED_OPENCL_2_1_APIS -DCL_USE_DEPRECATED_OPENCL_2_2_APIS)

# check the highest supported OpenCL version in the headers
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
  # PoCL's internal headers are OpenCL 3.0
  set(OPENCL_HEADER_VERSION 300)
endif()

# sets a target-version macro
# sets the include directories to point either to system-wide OpenCL headers, or PoCL's own
macro(set_opencl_header_includes)
  add_definitions(-DCL_TARGET_OPENCL_VERSION=${OPENCL_HEADER_VERSION}
                  -DCL_HPP_TARGET_OPENCL_VERSION=${OPENCL_HEADER_VERSION})
  if(NOT HAVE_OPENCL_H)
    include_directories("${CMAKE_SOURCE_DIR}/include")
  endif()
  if(NOT HAVE_OPENCL_HPP)
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

string(TIMESTAMP POCL_BUILD_TIMESTAMP "%d%m%Y%H%M%S" UTC)
file(WRITE "${CMAKE_BINARY_DIR}/pocl_build_timestamp.h" "#define POCL_BUILD_TIMESTAMP \"${POCL_BUILD_TIMESTAMP}\"")

####################################################################

# Host (basic/pthread) driver setup

set(DEFAULT_HOST_CLANG_FLAGS "${CLANG_TARGET_OPTION}${LLC_TRIPLE}")
set(DEFAULT_HOST_LLC_FLAGS "-relocation-model=pic -mtriple=${LLC_TRIPLE}")

# Option for users to define Clang's --target-abi argument.
# Not required on most platforms, but can be useful on RISC-V
if(NOT DEFINED HOST_CPU_TARGET_ABI)
  set(HOST_CPU_TARGET_ABI "")
endif()

# FP16 support requires _Float16 type
# LLVM <15 doesn't support _Float16;
# LLVM 15 crashes on some code; LLVM 16 seems to work
# CONFORMANCE=ON disables FP16 b/c PoCL support is incomplete
if((LLVM_VERSION_MAJOR GREATER_EQUAL 16)
   AND (NOT ENABLE_CONFORMANCE)
   AND HOST_CPU_SUPPORTS_FLOAT16)
  message(STATUS "Host support for cl_khr_fp16 enabled")
  set(HOST_CPU_ENABLE_CL_KHR_FP16 1)
else()
  message(STATUS "Host support for cl_khr_fp16 disabled")
  set(HOST_CPU_ENABLE_CL_KHR_FP16 0)
endif()

if(HOST_CPU_SUPPORTS_DOUBLE)
  set(HOST_CPU_ENABLE_CL_KHR_FP64 1)
else()
  set(HOST_CPU_ENABLE_CL_KHR_FP64 0)
endif()
if(HOST_CPU_ENABLE_CL_KHR_FP16)
  set(DEFAULT_HOST_CLANG_FLAGS "${DEFAULT_HOST_CLANG_FLAGS} -D_HAS_FLOAT16_TYPE")
endif()

# set the host device OpenCL version (and OpenCL C version too)
# LLVM 14 has sufficient OpenCL 3.0 support, so just enable it
set(HOST_DEVICE_CL_VERSION_MAJOR 3)
set(HOST_DEVICE_CL_VERSION_MINOR 0)

# Host CPU device: list of extensions that are always enabled, for both OpenCL 1.2 and 3.0
set(HOST_DEVICE_EXTENSIONS "cl_khr_byte_addressable_store cl_khr_global_int32_base_atomics \
cl_khr_global_int32_extended_atomics cl_khr_local_int32_base_atomics \
cl_khr_local_int32_extended_atomics cl_khr_3d_image_writes \
cl_khr_command_buffer cl_khr_command_buffer_multi_device cl_khr_subgroups \
cl_intel_unified_shared_memory cl_ext_buffer_device_address")

# Host CPU device: list of OpenCL 3.0 features that are always enabled
# TODO: __opencl_c_atomic_scope_all_devices works with CPU device but not others
# for now it's included because CTS test 'C11 atomics' subtest 'atomic_flag' requires it
# .... seems like a bug in CTS, but needs to be investigated.
set(HOST_DEVICE_FEATURES_30 "__opencl_c_3d_image_writes  __opencl_c_images \
__opencl_c_atomic_order_acq_rel __opencl_c_atomic_order_seq_cst \
__opencl_c_atomic_scope_device __opencl_c_program_scope_global_variables \
__opencl_c_atomic_scope_all_devices __opencl_c_generic_address_space \
__opencl_c_work_group_collective_functions")

# Host CPU device: Unfinished/undertested extensions only advertised when conformance
# is OFF.
if(NOT ENABLE_CONFORMANCE)
  set(HOST_DEVICE_EXTENSIONS "${HOST_DEVICE_EXTENSIONS} \
      cl_pocl_svm_rect cl_pocl_command_buffer_svm \
      cl_pocl_command_buffer_host_buffer")
  set(HOST_DEVICE_EXTENSIONS "${HOST_DEVICE_EXTENSIONS} cl_khr_subgroup_ballot \
cl_khr_subgroup_shuffle cl_intel_subgroups cl_intel_subgroups_short cl_intel_subgroups_char \
cl_ext_float_atomics cl_intel_required_subgroup_size")
  # read-write images are still partially broken
  set( HOST_DEVICE_FEATURES_30  "${HOST_DEVICE_FEATURES_30} __opencl_c_read_write_images")
endif()

# Extensions that are considered feature-complete (preferably CTS-tested).
#
# * cl_khr_subgroups: A simple implementation where SG is always the whole local
# X-dimension. NOTE: Independent forward progress is not yet supported, but it's
# not needed for compliance due to the corner case of only one SG in flight.
# Passes the subgroups/test_subgroups CTS.

# Extensions that are work-in-progress with known unfinished aspects.
# These are not advertised with a conformant build.
#
# * cl_khr_subgroup_shuffle: Passes the CTS, but only because it doesn't test
#   non-uniform(lock-step) behavior, see:
#   https://github.com/KhronosGroup/OpenCL-CTS/issues/1236
#
# * cl_khr_subgroup_ballot: sub_group_ballot() works for uniform calls, the rest
#   are unimplemented.
#
# * cl_intel_subgroups: The block reads/writes are unimplemented.
#
# * cl_intel_required_subgroup_size: CL_​KERNEL_​SPILL_​MEM_​SIZE_​INTEL and
#   CL_​KERNEL_​COMPILE_​SUB_​GROUP_​SIZE_​INTEL are yet to implement.
#
# * cl_khr_fp16: only implemented for part of builtin library functions.
#   Those with either 1) expression implementation, or 2) Clang builtin
#   implementation.
#

if(HOST_DEVICE_EXTENSIONS MATCHES "cl_khr_subgroup")
  set(HOST_DEVICE_FEATURES_30 "${HOST_DEVICE_FEATURES_30} __opencl_c_subgroups")
endif()

if(ENABLE_SPIRV)
  set(HOST_DEVICE_EXTENSIONS "${HOST_DEVICE_EXTENSIONS} cl_khr_il_program")
endif()

if(HOST_CPU_ENABLE_CL_KHR_FP16)
  set(HOST_DEVICE_EXTENSIONS "${HOST_DEVICE_EXTENSIONS} cl_khr_fp16")
  set(HOST_DEVICE_FEATURES_30 "${HOST_DEVICE_FEATURES_30} __opencl_c_fp16")
endif()

if(HOST_CPU_ENABLE_CL_KHR_FP64)
  set(HOST_DEVICE_EXTENSIONS "${HOST_DEVICE_EXTENSIONS} cl_khr_fp64")
  set(HOST_DEVICE_FEATURES_30 "${HOST_DEVICE_FEATURES_30} __opencl_c_fp64")
endif()



if(HOST_DEVICE_EXTENSIONS MATCHES "cl_ext_float_atomics")
  set(HOST_DEVICE_FEATURES_30 "${HOST_DEVICE_FEATURES_30} __opencl_c_ext_fp32_global_atomic_add __opencl_c_ext_fp32_local_atomic_add __opencl_c_ext_fp32_global_atomic_min_max __opencl_c_ext_fp32_local_atomic_min_max")
  if(HOST_CPU_ENABLE_CL_KHR_FP64)
    set(HOST_DEVICE_FEATURES_30 "${HOST_DEVICE_FEATURES_30} __opencl_c_ext_fp64_global_atomic_add __opencl_c_ext_fp64_local_atomic_add __opencl_c_ext_fp64_global_atomic_min_max __opencl_c_ext_fp64_local_atomic_min_max")
  endif()
  # __opencl_c_ext_fp16_global_atomic_add __opencl_c_ext_fp16_local_atomic_add __opencl_c_ext_fp16_global_atomic_min_max __opencl_c_ext_fp16_local_atomic_min_max
endif()

# don't include cl_khr_int64 in HOST_DEVICE_EXTENSIONS list, because
# such extension doesn't exist in official extension list
# for FULL profile, int64 is always enabled;
# for embedded profiles, it's optional
set(HOST_DEVICE_EXTENSION_DEFINES "${HOST_DEVICE_EXTENSION_DEFINES} -Dcl_khr_int64=1")
set(HOST_DEVICE_FEATURES_30 "${HOST_DEVICE_FEATURES_30} __opencl_c_int64")
set(HOST_DEVICE_EXTENSIONS "${HOST_DEVICE_EXTENSIONS} cl_khr_int64_base_atomics cl_khr_int64_extended_atomics")


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
    if(I386)
      set(KERNELLIB_HOST_CPU_VARIANTS i386 i686 mmx sse ${KERNELLIB_HOST_CPU_VARIANTS})
    endif()
  elseif(POWERPC64LE)
    set(KERNELLIB_HOST_CPU_VARIANTS pwr8 pwr9 pwr10)
  else()
    message(WARNING "Don't know what CPU variants to use for the kernel built-in function library on this platform, will only provide a generic build.")
    set(KERNELLIB_HOST_CPU_VARIANTS "")
  endif()
  list(APPEND KERNELLIB_HOST_CPU_VARIANTS generic)
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
  set(OCL_KERNEL_TARGET_CPU "${SELECTED_HOST_CPU}") #The kernel target CPU variant.
  set(BUILD_BASIC 1)
  set(BUILD_PTHREAD 1)
endif()

if(ENABLE_TBB_DEVICE)
  set(BUILD_TBB 1)
  set(OCL_DRIVERS "${OCL_DRIVERS} tbb")
endif()

# The almaif device could be built by default, but it's implemented in C++,
# thus requires a C++ compiler, so let's not.
if(ENABLE_ALMAIF_DEVICE)

  if(NOT HAVE_XRT AND DEFINED ENV{XILINX_XRT})
    set(XRT $ENV{XILINX_XRT})

    if(NOT XRT_INCLUDEDIR)
      if(EXISTS "${XRT}/include")
        set(XRT_INCLUDEDIR "${XRT}/include" CACHE PATH "XRT include dir")
      else()
        message(FATAL_ERROR "please provide -DXRT_INCLUDEDIR=... to CMake")
      endif()
    endif()

    if(NOT XRT_LIBDIR)
      if (EXISTS "${XRT}/lib")
       set(XRT_LIBDIR "${XRT}/lib" CACHE PATH "XRT library dir")
      else()
        message(FATAL_ERROR "please provide -DXRT_LIBDIR=... to CMake")
      endif()
    endif()

    if(XRT_INCLUDEDIR AND XRT_LIBDIR)
      set(HAVE_XRT 1 CACHE BOOL "have XRT")
    endif()
  endif()

  set(BUILD_ALMAIF 1)
  set(OCL_DRIVERS "${OCL_DRIVERS} almaif")
endif()

if(DEFINED EXTRA_OCL_TARGETS)
  set(OCL_TARGETS "${OCL_TARGETS} ${EXTRA_OCL_TARGETS}")
endif()

####################################################################

if(ENABLE_REMOTE_CLIENT)
  set(BUILD_REMOTE_CLIENT 1)
  set(OCL_TARGETS "${OCL_TARGETS} remote")
  set(OCL_DRIVERS "${OCL_DRIVERS} remote")
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

  if(RENAME_POCL AND PROXY_USE_LIBOPENCL_STUB)
    message(FATAL_ERROR "RENAME_POCL and PROXY_USE_LIBOPENCL_STUB can not be used together")
  endif()

  if(RENAME_POCL AND ENABLE_LOADABLE_DRIVERS)
    message(FATAL_ERROR "Proxy driver cannot be used with ENABLE_LOADABLE_DRIVERS unless using PROXY_USE_LIBOPENCL_STUB")
  endif()

  if(NOT PROXY_USE_LIBOPENCL_STUB)
    set(RENAME_POCL ON )
  else()
    set(RENAME_POCL OFF )
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

  if(NOT WITH_TCE)
    set(WITH_TCE ENV PATH)
  endif()

  find_program(TCE_CONFIG NAMES "openasip-config" "tce-config" HINTS ${WITH_TCE})
  find_program(TCECC NAMES "tcecc" HINTS ${WITH_TCE})
  find_program(TTASIM NAMES "ttasim" HINTS ${WITH_TCE})

  if(TCE_CONFIG AND TCECC AND TTASIM)

    message(STATUS "Found tcecc + openasip-config + ttasim, testing setup")

    get_filename_component(TCE_BASEDIR "${TCE_CONFIG}" DIRECTORY)
    find_library(TCE_LIBS "openasip" "tce" HINTS "${TCE_BASEDIR}/../lib" ENV PATH)
    if(NOT TCE_LIBS)
      execute_process(COMMAND "${TCE_CONFIG}" --libs OUTPUT_VARIABLE TCE_LIBS RESULT_VARIABLE RESV1 OUTPUT_STRIP_TRAILING_WHITESPACE)
    endif()
    execute_process(COMMAND "${TCE_CONFIG}" --includes OUTPUT_VARIABLE TCE_INCLUDES RESULT_VARIABLE RESV2 OUTPUT_STRIP_TRAILING_WHITESPACE)
    execute_process(COMMAND "${TCE_CONFIG}" --version OUTPUT_VARIABLE TCE_VERSION RESULT_VARIABLE RESV3 OUTPUT_STRIP_TRAILING_WHITESPACE)
    execute_process(COMMAND "${TCE_CONFIG}" --cxxflags OUTPUT_VARIABLE TCE_CXXFLAGS RESULT_VARIABLE RESV4 OUTPUT_STRIP_TRAILING_WHITESPACE)
    execute_process(COMMAND "${TCE_CONFIG}" --prefix OUTPUT_VARIABLE TCE_PREFIX RESULT_VARIABLE RESV5 OUTPUT_STRIP_TRAILING_WHITESPACE)
    execute_process(COMMAND "${TCE_CONFIG}" --llvm-config OUTPUT_VARIABLE TCE_LLVM_CONFIG RESULT_VARIABLE RESV6 OUTPUT_STRIP_TRAILING_WHITESPACE)
    execute_process(COMMAND "${TTASIM}" --help OUTPUT_VARIABLE TTASIM_HELP RESULT_VARIABLE RESV9)

    if (RESV1 OR RESV2 OR RESV3 OR RESV4 OR RESV5)
      message(WARNING "openasip-config: Nonzero exit status, disabling TCE")
    elseif (RESV9)
      message(WARNING "ttasim: Nonzero exit status, disabling TCE")
    else()

    set(TCE_LLVM_CONFIG "unknown")
    execute_process(COMMAND "${TCE_CONFIG}" --llvm-config RESULT_VARIABLE RES OUTPUT_VARIABLE TCE_LLVM_CONFIG OUTPUT_STRIP_TRAILING_WHITESPACE)

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
    message(STATUS "Failed to find tcecc or openasip-config, disabling TCE")
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

  if(TCE_LLVM_CONFIG AND LLVM_CONFIG)
    if(NOT (LLVM_CONFIG STREQUAL TCE_LLVM_CONFIG))
      message(FATAL_ERROR "openasip-config returned llvm-config is ${TCE_LLVM_CONFIG} but LLVM_CONFIG used by PoCL is ${LLVM_CONFIG}")
    endif()
  endif()

  set(TCE_DEVICE_EXTENSIONS "cl_khr_byte_addressable_store cl_khr_fp16")
  set(TEMP_EXT "${TCE_DEVICE_EXTENSIONS}")
  set(TCE_DEVICE_EXTENSION_DEFINES "")
  separate_arguments(TEMP_EXT)
  foreach(EXT ${TEMP_EXT})
    set(TCE_DEVICE_EXTENSION_DEFINES "${TCE_DEVICE_EXTENSION_DEFINES} -D${EXT}")
  endforeach()

  set(TCE_DEVICE_CL_VERSION "120")
  set(TCE_DEVICE_CL_STD "1.2")

  if(LLVM_VERSION VERSION_GREATER_EQUAL 18.0)
    message(FATAL_ERROR "OpenASIP requires LLVM/Clang <= 17")
  endif()

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
    message(FATAL_ERROR "Vulkan library/headers not found")
  else()
    find_program(CLSPV NAMES clspv HINTS ${CLSPV_DIR})
    find_program(CLSPV_REFLECTION NAMES clspv-reflection HINTS ${CLSPV_DIR})
    if (CLSPV AND CLSPV_REFLECTION)
      message(STATUS "Found clspv: ${CLSPV}")
      set(HAVE_CLSPV ON)
    else()
      message(FATAL_ERROR "Could not find clspv. Provide a path to clspv & clspv-reflection binaries with -DCLSPV_DIR=...")
    endif()
    message(STATUS "Using Vulkan library ${Vulkan_LIBRARY} and Vulkan headers in ${Vulkan_INCLUDE_DIR}")
    message(WARNING "Note: Vulkan backend is highly experimental and very incomplete." )
    set(OCL_DRIVERS "${OCL_DRIVERS} vulkan")
    set(OCL_TARGETS "${OCL_TARGETS} vulkan")
  endif()
endif()

##########################################################

if (ENABLE_LEVEL0)
  if(NOT PKG_CONFIG_EXECUTABLE)
    message(FATAL_ERROR "building Level0 requires pkg-config")
  endif()
  pkg_check_modules(LEVEL0 REQUIRED level-zero>=1.3)

  option(ENABLE_LEVEL0_EXTRA_FEATURES "enables extra opencl extensions & features in LevelZero device. these are not checked at runtime for presence " OFF)

  if(NOT ENABLE_LLVM)
    message(FATAL_ERROR "Level Zero requires LLVM/Clang")
  endif()
  if(NOT LLVM_SPIRV)
    message(FATAL_ERROR "Level Zero requires llvm-spirv translator")
  endif()
  if(NOT STATIC_LLVM)
    message(WARNING "Level Zero might run into symbol conflicts when using dynamic LLVM libraries, consider using -DSTATIC_LLVM=ON")
  endif()

  if (NOT LEVEL0_FOUND)
    message(STATUS "Could not find level-zero via pkg-config, trying to find it directly")
    find_library(LEVEL0_LIBRARIES NAMES ze_loader REQUIRED)
    find_path(LEVEL0_INCLUDE_DIRS NAMES ze_api.h PATH_SUFFIXES level_zero)
    if (LEVEL0_LIBRARIES AND LEVEL0_INCLUDE_DIRS)
      set(LEVEL0_FOUND ON CACHE INTERNAL "Level0 found")
    endif()
  endif()

  if(NOT LEVEL0_FOUND)
    message(FATAL_ERROR "Level Zero library/headers not found")
  else()
    message(STATUS "Using Level0 library: '${LEVEL0_LIBRARIES}' and Level0 headers: '${LEVEL0_INCLUDE_DIRS}'")
    message(WARNING "Note: Level0 backend is highly experimental and very incomplete." )
    set(OCL_DRIVERS "${OCL_DRIVERS} level0")
    set(OCL_TARGETS "${OCL_TARGETS} level0")
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

  set(CUDA_DEVICE_CL_VERSION_MAJOR 3)
  set(CUDA_DEVICE_CL_VERSION_MINOR 0)
  set(CUDA_DEVICE_CL_VERSION "300")
  set(CUDA_DEVICE_CL_STD "3.0")

  # CUDA device: list of extensions that are always enabled, for both OpenCL 1.2 and 3.0
  set(CUDA_DEVICE_EXTENSIONS "cl_khr_byte_addressable_store cl_khr_global_int32_base_atomics \
    cl_khr_global_int32_extended_atomics cl_khr_local_int32_base_atomics \
    cl_khr_local_int32_extended_atomics cl_khr_int64_base_atomics \
    cl_khr_int64_extended_atomics cl_nv_device_attribute_query")

  # CUDA device: list of OpenCL 3.0 features that are always enabled
  # __opencl_c_images should not be included since we don't support images yet,
  # but is needed to work around an issue with the opencl-c.h header in Clang versions from 14 onwards
  # https://github.com/llvm/llvm-project/issues/58017
  # __opencl_c_3d_image_writes and other image-related extensions are disabled because not supported yet
  set(CUDA_DEVICE_FEATURES_30 "__opencl_c_images __opencl_c_atomic_order_acq_rel __opencl_c_atomic_order_seq_cst \
    __opencl_c_atomic_scope_device __opencl_c_program_scope_global_variables \
    __opencl_c_generic_address_space")

  if(LLVM_VERSION VERSION_GREATER_EQUAL 17.0)
    set(CUDA_DEVICE_EXTENSIONS "${CUDA_DEVICE_EXTENSIONS} cl_ext_float_atomics")
    set(CUDA_DEVICE_FEATURES_30 "${CUDA_DEVICE_FEATURES_30} \
     __opencl_c_ext_fp32_global_atomic_add __opencl_c_ext_fp32_local_atomic_add \
     __opencl_c_ext_fp32_global_atomic_min_max __opencl_c_ext_fp32_local_atomic_min_max \
     __opencl_c_ext_fp64_global_atomic_add __opencl_c_ext_fp64_local_atomic_add \
     __opencl_c_ext_fp64_global_atomic_min_max __opencl_c_ext_fp64_local_atomic_min_max")
  endif()

  if(ENABLE_SPIRV)
    set(CUDA_DEVICE_EXTENSIONS "${CUDA_DEVICE_EXTENSIONS} cl_khr_il_program")
  endif()

  option(ENABLE_CUDA_EXPERIMENTAL_EXTENSIONS "experimental (and/or unfinished) device extensions" OFF)
  if(ENABLE_CUDA_EXPERIMENTAL_EXTENSIONS)
    # unfinished extensions
    set(CUDA_DEVICE_EXTENSIONS "${CUDA_DEVICE_EXTENSIONS} cl_khr_subgroups")
    #set(CUDA_DEVICE_EXTENSIONS "${CUDA_DEVICE_EXTENSIONS} cl_khr_subgroup_ballot \
    #  cl_khr_subgroup_shuffle cl_intel_subgroups cl_intel_required_subgroup_size")
  endif()

  if(CUDA_DEVICE_EXTENSIONS MATCHES "cl_khr_subgroup")
    set(CUDA_DEVICE_FEATURES_30 "${CUDA_DEVICE_FEATURES_30} __opencl_c_subgroups")
  endif()

  # advertise fp16 support in CUDA device info
  # disable this when using PoCL with CUDA device with Compute Capability < 6.0
  option(ENABLE_CUDA_FP16 "cuda fp16" ON)

  if(ENABLE_CUDA_FP16)
    set(CUDA_DEVICE_EXTENSIONS "${CUDA_DEVICE_EXTENSIONS} cl_khr_fp16")
    set(CUDA_DEVICE_FEATURES_30 "${CUDA_DEVICE_FEATURES_30} __opencl_c_fp16")
  endif()

  # FP64 is always enabled for CUDA, it should be available since Compute Capability 1.3
  set(CUDA_DEVICE_EXTENSIONS "${CUDA_DEVICE_EXTENSIONS} cl_khr_fp64")
  set(CUDA_DEVICE_FEATURES_30 "${CUDA_DEVICE_FEATURES_30} __opencl_c_fp64")

  if(ENABLE_CUDNN)
    add_definitions(-DENABLE_CUDNN=1)
  endif()


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
# 11:0:9 == 3.0 (the libpocl.so will be named libpocl.so.2.9.X )
# 12:0:10 == 3.1 (the libpocl.so will be named libpocl.so.2.10.X )
# 13:0:11 == 4.0 (the libpocl.so will be named libpocl.so.2.11.X )
# 14:0:12 == 5.0 (the libpocl.so will be named libpocl.so.2.12.X )
# 15:0:13 == 6.0 (the libpocl.so will be named libpocl.so.2.13.X )
# 16:0:14 == 6.1 (the libpocl.so will be named libpocl.so.2.14.X )

set(LIB_CURRENT_VERSION 16)
set(LIB_REVISION_VERSION 0)
set(LIB_AGE_VERSION 14)

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
set(BUILD_LEVEL0 ${ENABLE_LEVEL0})
set(TCE_AVAILABLE ${ENABLE_TCE})
set(TCEMC_AVAILABLE ${ENABLE_TCEMC})

configure_file("config.h.in.cmake" "config.h.new" ESCAPE_QUOTES)
rename_if_different("${CMAKE_BINARY_DIR}/config.h.new" "${CMAKE_BINARY_DIR}/config.h" 0)

configure_file("config2.h.in.cmake" "config2.h.new")
rename_if_different("${CMAKE_BINARY_DIR}/config2.h.new" "${CMAKE_BINARY_DIR}/config2.h" 0)

configure_file("pocl_version.h.in.cmake" "pocl_version.h.new" ESCAPE_QUOTES)
rename_if_different("${CMAKE_BINARY_DIR}/pocl_version.h.new" "${CMAKE_BINARY_DIR}/pocl_version.h" 0)

configure_file("pocl_opencl.h.in.cmake" "pocl_opencl.h.new")
rename_if_different("${CMAKE_BINARY_DIR}/pocl_opencl.h.new" "${CMAKE_BINARY_DIR}/pocl_opencl.h" 0)

configure_file("cl_offline_compiler.sh.in.cmake" "cl_offline_compiler.sh.in.new" @ONLY)
rename_if_different("${CMAKE_BINARY_DIR}/cl_offline_compiler.sh.in.new" "${CMAKE_BINARY_DIR}/cl_offline_compiler.sh" 1)

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
message(STATUS "OPENCL_LIBDIR: ${OPENCL_LIBDIR}")
message(STATUS "OPENCL_LIBNAME: ${OPENCL_LIBNAME}")

##########################################################
## generate CTestCustom.cmake

if(ENABLE_ASAN OR ENABLE_LSAN)
  file(GENERATE OUTPUT "${CMAKE_BINARY_DIR}/lsan.supp" CONTENT "leak:${LLVM_SRC_ROOT}/lib/Support/Unix/Signals.inc")
  set(SAN_EXTRA "set(ENV{LSAN_OPTIONS} \"suppressions=${CMAKE_BINARY_DIR}/lsan.supp\")")
endif()

# We need to use the oneAPI environment's LD_LIBRARY_PATH (need to source
# the setvars.sh before testing), but still ensure we use the system's libOpenCL.so
# (ideally ocl-icd) to pick up libpocl. This is ensured by prepending Opencl LIBDIR
# before all paths in LD_LIBRARY_PATH. Otherwise libOpenCL.so will be found in /opt/intel/oneapi
# which is not ocl-icd).
if(SYCL_LIBDIR)
  set(SYCL_EXTRA "set(ENV{LD_LIBRARY_PATH} \"${OPENCL_LIBDIR}:${SYCL_LIBDIR}:$ENV{LD_LIBRARY_PATH}\")")
endif()

# pointing OCL_ICD_VENDORS to pocl doesn't make sense with the proxy driver;
# we want the driver to proxy to another OpenCL, so
# we must let ocl-icd load it
if(ENABLE_PROXY_DEVICE)

  file(GENERATE OUTPUT "${CMAKE_BINARY_DIR}/CTestCustom.cmake" CONTENT "
  ${SAN_EXTRA}
  set(ENV{POCL_ENABLE_UNINIT} \"1\")
  set(ENV{POCL_BUILDING} \"1\")
  ${SYCL_EXTRA}
  ")
else()

  file(GENERATE OUTPUT "${CMAKE_BINARY_DIR}/CTestCustom.cmake" CONTENT "
  ${SAN_EXTRA}
  set(ENV{POCL_ENABLE_UNINIT} \"1\")
  set(ENV{POCL_BUILDING} \"1\")
  set(ENV{OCL_ICD_VENDORS} \"${CMAKE_BINARY_DIR}/ocl-vendors/\")
  ${SYCL_EXTRA}
  ")
endif()

##########################################################

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
    add_custom_target(check COMMAND ${CMAKE_CTEST_COMMAND} "--output-on-failure" -j ${HOST_CPU_CORECOUNT} ${COMMAND_USES_TERMINAL})
    add_custom_target(check_tier1 COMMAND ${CMAKE_CTEST_COMMAND} "--output-on-failure" -L "'internal|piglit|PyOpenCL|conformance_suite_micro|shoc|chipStar'" -j ${HOST_CPU_CORECOUNT} ${COMMAND_USES_TERMINAL})

    add_custom_target(check-cuda COMMAND
      "${CMAKE_SOURCE_DIR}/tools/scripts/run_cuda_tests"
      "--output-on-failure" -j ${HOST_CPU_CORECOUNT} ${COMMAND_USES_TERMINAL})
endif()

if(ENABLE_EXAMPLES)
    add_subdirectory("examples")
endif()

# generate doxygen using make targets
option(ENABLE_DOXYGEN "Generate internal developer documentation with doxygen" OFF)
if(ENABLE_DOXYGEN)
  find_package(Doxygen REQUIRED)

  # configure placeholders in doxyfile
  configure_file(${CMAKE_SOURCE_DIR}/CMakeDoxyfile ConfiguredDoxyfile @ONLY)

  add_custom_target(gen_doc
          COMMAND ${DOXYGEN_EXECUTABLE} ConfiguredDoxyfile
          WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
          VERBATIM)

  # os specific ways of opening browsers
  set(DOXY_INDEX doxygen/html/index.xhtml)
  set(HTML_VIEWER_COMMAND)
  if(UNIX)
    if(APPLE)
      set(HTML_VIEWER_COMMAND open)
    else()
      set(HTML_VIEWER_COMMAND xdg-open)
    endif()
  elseif(WIN32)
    set(HTML_VIEWER_COMMAND explorer.exe)
  endif()
  add_custom_target(open_doc
          COMMAND ${HTML_VIEWER_COMMAND} ${DOXY_INDEX}
          WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
          VERBATIM
          ${COMMAND_USES_TERMINAL}
          DEPENDS gen_doc)

  unset(DOXY_INDEX)
  unset(HTML_VIEWER_COMMAND)
endif()

if(ENABLE_REMOTE_SERVER)
    add_subdirectory("pocld")
endif()

set(CPACK_PACKAGE_DESCRIPTION_FILE "${CMAKE_SOURCE_DIR}/CPack.pocl.description.txt")
set(CPACK_PACKAGE_ICON "${CMAKE_SOURCE_DIR}/doc/www/img/pocl-80x60.png")
set(CPACK_PACKAGE_HOMEPAGE_URL "https://github.com/pocl/pocl")
set(CPACK_PACKAGE_CONTACT "https://github.com/pocl/pocl")
set(CPACK_PACKAGE_CHECKSUM "SHA512")
set(CPACK_RESOURCE_FILE_README "${CMAKE_SOURCE_DIR}/README.md")
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

MESSAGE(STATUS "HAVE_CLOCK_GETTIME: ${HAVE_CLOCK_GETTIME}")
MESSAGE(STATUS "HAVE_GLEW: ${HAVE_GLEW}")
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
  if(ENABLE_HOST_CPU_DEVICES)
  MESSAGE(STATUS "CLANG_TARGET_OPTION: ${CLANG_TARGET_OPTION}")
  MESSAGE(STATUS "CLANG_MARCH_FLAG: ${CLANG_MARCH_FLAG}")
  MESSAGE(STATUS "LLC_TRIPLE: ${LLC_TRIPLE}")
  MESSAGE(STATUS "LLC_HOST_CPU_AUTO: ${LLC_HOST_CPU_AUTO}")
  MESSAGE(STATUS "LLC_HOST_CPU: ${LLC_HOST_CPU}")
  MESSAGE(STATUS "SELECTED_HOST_CPU: ${SELECTED_HOST_CPU}")
  MESSAGE(STATUS "HOST_CPU_FORCED: ${HOST_CPU_FORCED}")
  MESSAGE(STATUS "HOST_CPU_SUPPORTS_FLOAT16: ${HOST_CPU_SUPPORTS_FLOAT16}")
  endif()
endif()
MESSAGE(STATUS "")
MESSAGE(STATUS "MAX_EXTENDED_ALIGNMENT: ${MAX_EXTENDED_ALIGNMENT}")
MESSAGE(STATUS "OCL_KERNEL_TARGET: ${OCL_KERNEL_TARGET}")
MESSAGE(STATUS "OCL_KERNEL_TARGET_CPU: ${OCL_KERNEL_TARGET_CPU}")
MESSAGE(STATUS "HOST_DEVICE_ADDRESS_BITS: ${HOST_DEVICE_ADDRESS_BITS}")
MESSAGE(STATUS "HOST_CPU_ENABLE_CL_KHR_FP16: ${HOST_CPU_ENABLE_CL_KHR_FP16}")
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
MESSAGE(STATUS "ENABLE_HWLOC: ${ENABLE_HWLOC}")
MESSAGE(STATUS "RENAME_POCL: ${RENAME_POCL}")
if(ARM)
MESSAGE(STATUS "ENABLE_FP64: ${ENABLE_FP64}")
endif()
MESSAGE(STATUS "ENABLE_IPO: ${ENABLE_IPO}")
MESSAGE(STATUS "ENABLE_ICD: ${ENABLE_ICD}")
MESSAGE(STATUS "ENABLE_TCE: ${ENABLE_TCE}")
MESSAGE(STATUS "ENABLE_TCEMC: ${ENABLE_TCEMC}")
MESSAGE(STATUS "ENABLE_HSA: ${ENABLE_HSA}")
MESSAGE(STATUS "ENABLE_ALMAIF_DEVICE: ${ENABLE_ALMAIF_DEVICE}")
MESSAGE(STATUS "ENABLE_CUDA: ${ENABLE_CUDA}")
MESSAGE(STATUS "ENABLE_CUDNN: ${ENABLE_CUDNN}")
MESSAGE(STATUS "ENABLE_HOST_CPU_DEVICES: ${ENABLE_HOST_CPU_DEVICES}")
MESSAGE(STATUS "ENABLE_VULKAN: ${ENABLE_VULKAN}")
MESSAGE(STATUS "ENABLE_ASAN (address sanitizer): ${ENABLE_ASAN}")
MESSAGE(STATUS "ENABLE_LSAN (leak sanitizer): ${ENABLE_LSAN}")
MESSAGE(STATUS "ENABLE_TSAN (thread sanitizer): ${ENABLE_TSAN}")
MESSAGE(STATUS "ENABLE_UBSAN (UB sanitizer): ${ENABLE_UBSAN}")
MESSAGE(STATUS "ENABLE_LTTNG: ${ENABLE_LTTNG}")
MESSAGE(STATUS "ENABLE_POCL_BUILDING: ${ENABLE_POCL_BUILDING}")
MESSAGE(STATUS "ENABLE_POCL_FLOAT_CONVERSION: ${ENABLE_POCL_FLOAT_CONVERSION}")
MESSAGE(STATUS "ENABLE_RELOCATION: ${ENABLE_RELOCATION}")
MESSAGE(STATUS "ENABLE_PROXY_DEVICE: ${ENABLE_PROXY_DEVICE}")
MESSAGE(STATUS "ENABLE_PROXY_DEVICE_INTEROP: ${ENABLE_PROXY_DEVICE_INTEROP}")
MESSAGE(STATUS "ENABLE_REMOTE_SERVER: ${ENABLE_REMOTE_SERVER}")
MESSAGE(STATUS "ENABLE_REMOTE_CLIENT: ${ENABLE_REMOTE_CLIENT}")
MESSAGE(STATUS "ENABLE_D2D_MIG: ${ENABLE_D2D_MIG}")
MESSAGE(STATUS "ENABLE_RDMA: ${ENABLE_RDMA}")
MESSAGE(STATUS "ENABLE_CL_GET_GL_CONTEXT: ${ENABLE_CL_GET_GL_CONTEXT}")
MESSAGE(STATUS "ENABLE_OPENGL_INTEROP: ${ENABLE_OPENGL_INTEROP}")
MESSAGE(STATUS "ENABLE_EGL_INTEROP: ${ENABLE_EGL_INTEROP}")

MESSAGE(STATUS "ENABLE_SLEEF: ${ENABLE_SLEEF}")
MESSAGE(STATUS "ENABLE_SPIRV: ${ENABLE_SPIRV}")
MESSAGE(STATUS "ENABLE_VALGRIND: ${ENABLE_VALGRIND}")
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
MESSAGE(STATUS "Enable internal doxygen documentation: ${ENABLE_DOXYGEN}")
MESSAGE(STATUS "Kernel caching: ${KERNEL_CACHE_DEFAULT}")
MESSAGE(STATUS "Kernel library CPU variants: ${KERNELLIB_HOST_CPU_VARIANTS}")
MESSAGE(STATUS "Kernel library distro build: ${KERNELLIB_HOST_DISTRO_VARIANTS}")
MESSAGE(STATUS "Use pocl custom memory allocator: ${USE_POCL_MEMMANAGER}")
MESSAGE(STATUS "L1d cacheline size: ${HOST_CPU_CACHELINE_SIZE}")
