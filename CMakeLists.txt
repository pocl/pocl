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

cmake_minimum_required(VERSION 2.8.12 FATAL_ERROR)

project(pocl)
set(MAJOR_VERSION 0)
set(MINOR_VERSION 12)
set(VERSION_SUFFIX "-pre")
set(VERSION_STRING ${MAJOR_VERSION}.${MINOR_VERSION}${VERSION_SUFFIX})
set(POCL_VERSION ${VERSION_STRING})



######################################################################################

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
if (UNIX AND NOT CMAKE_CROSSCOMPILING)
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
  set(POCL_INSTALL_OPENCL_HEADER_DIR "${POCL_INSTALL_PUBLIC_HEADER_DIR}/OpenCL" CACHE PATH "POCL header dir for OpenCL headers")
else()
  set(POCL_INSTALL_OPENCL_HEADER_DIR "${POCL_INSTALL_PUBLIC_HEADER_DIR}/CL" CACHE PATH "POCL header dir for OpenCL headers")
endif()

option(BUILD_SHARED_LIBS "ON=Build shared libs, OFF=static libs" ON)

# TODO measure slowdown with compiled but disabled debug, see if it can default to always compile in
option(
  POCL_DEBUG_MESSAGES
  "Enable debug messages from pocl (useful for OpenCL developers), must be enabled at runtime, with env var POCL_DEBUG"
  ON)

# required b/c SHARED libs defaults to ON while OBJECT defaults to OFF
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

######################################################################################

# AMD HSA
option(ENABLE_HSA "Enable the HSA device driver for AMD GCN devices" OFF)
if(ENABLE_HSA)
  message(FATAL_ERROR "Sorry, HSA support in CMake is unfinished and unusable ATM")
endif()

######################################################################################
# AC_CONFIG_TESTDIR([tests])
enable_testing()

######################################################################################

if (UNIX)
  find_package(PkgConfig MODULE REQUIRED)
endif()

######################################################################################
set(ANDROID_COMPILER 0)
if (CMAKE_C_COMPILER MATCHES "android")
  set(ANDROID_COMPILER 1)
  add_definitions(-DPOCL_ANDROID)
  add_definitions(-DPOCL_ANDROID_PREFIX="/data/data/org.pocl.libs/files")
endif ()

######################################################################################

set(CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake)
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
#
# Find executables to few tools required during build 
#

find_program (PATCH_EXEC 
  NAMES patch patch.exe
  HINTS ENV PATH
)

find_program (XARGS_EXEC 
  NAMES xargs xargs.exe
  HINTS ENV PATH
)

find_program (CAT_EXEC 
  NAMES cat cat.exe
  HINTS ENV PATH
)

if (!PATCH_EXEC)
  message(FATAL_ERROR "Could not find patch command.")
endif()

######################################################################################

set(CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake)
include(LLVM RESULT_VARIABLE RES)
if(NOT RES)
  message(FATAL_ERROR "Could not load LLVM.cmake")
endif()

if("${LLVM_VERSION}" VERSION_LESS "3.2")
  message(FATAL_ERROR "POCL requires LLVM version >= 3.2 !")
endif()

######################################################################################

# if variable FEATURE_X isnt defined, sets it to DEFAULT_FEATURE_X;
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

if(NOT DEFINED DEFAULT_ENABLE_DOCS)
  include(Sphinx RESULT_VARIABLE RES)
  if(SPHINX_EXECUTABLE)
    set(DEFAULT_ENABLE_DOCS 1 CACHE INTERNAL "Enable building of documentation")
  else()
    set(DEFAULT_ENABLE_DOCS 0 CACHE INTERNAL "Enable building of documentation")
  endif()
endif()


setup_cached_var(ENABLE_DOCS "Documentation building"
  "Requested enabling documentation, but could not find Sphinx ! -> disabling docs.."
  "Sphinx found, but requested disabling docs")

######################################################################################

if(UNIX)
  include(CheckCSourceCompiles)
  CHECK_C_SOURCE_COMPILES("
#include <time.h>
int main() {
  struct timespec pocl_debug_timespec;
  clock_gettime(CLOCK_REALTIME, &pocl_debug_timespec);
  return 0;
}
  " HAVE_CLOCK_GETTIME)
else()
  if(WIN32)
    message(STATUS "Using GetSystemTime...() on Win32")
    set(HAVE_CLOCK_GETTIME 1)
  else()
    set(HAVE_CLOCK_GETTIME 0)
  endif()
endif()

######################################################################################

set(NEW_PRINTF_WORKS 1)

if("${LLVM_VERSION}" VERSION_LESS "3.4")
  message(STATUS "Turning off new printf()")
  set(NEW_PRINTF_WORKS 0)
else()
  message(STATUS "Enabling new printf()")
endif()

######################################################################################

if(NOT DEFINED DEFAULT_USE_VECMATHLIB)
  # vecmathlib
  if(CLANGXX_WORKS AND EXISTS "${CMAKE_SOURCE_DIR}/lib/kernel/vecmathlib/vecmathlib.h")
    set(DEFAULT_USE_VECMATHLIB 1 CACHE INTERNAL "vecmathlib availability")
  else()
    set(DEFAULT_USE_VECMATHLIB 0 CACHE INTERNAL "vecmathlib availability")
  endif()
endif()

setup_cached_var(USE_VECMATHLIB "Vecmathlib use"
  "Requested enabling vecmathlib use, but either clang++ doesnt work or vecmathlib sources are missing.. -> disabling vecmathlib use"
  "Vecmathlib is usable, but requested disabling it")

######################################################################################

if(UNIX)
  if(APPLE)
    # MacOS ld outputs useless warnings like
    # ld: warning: -macosx_version_min not specificed, assuming 10.7
    # suppress them with -w.
    set(DEFAULT_HOST_LD_FLAGS "-Wl,-dylib -lm")
  elseif(ANDROID_COMPILER)
    set(DEFAULT_HOST_LD_FLAGS "-L/system/lib/ -shared -ldl -lc -lm /system/lib/crtbegin_so.o /system/lib/crtend_so.o")
  else()
    set(DEFAULT_HOST_LD_FLAGS "-shared -lm")
  endif()
  set(LIBMATH "-lm")
elseif(WIN32)
  set(LIBMATH)
endif()

######################################################################################

option(USE_VECMATHLIB_BUILTINS_ONLY "Use only __builtin_* functions in the kernel library." OFF)

if(USE_VECMATHLIB)
  set(DEFAULT_KERNEL_CLANGXX_FLAGS "-DVML_NO_IOSTREAM ${CLANGXX_STDLIB}")
endif()
set(DEFAULT_KERNEL_CL_FLAGS "-D__OPENCL_VERSION__=120")

if(USE_VECMATHLIB_BUILTINS_ONLY AND USE_VECMATHLIB)
  set(DEFAULT_KERNEL_CL_FLAGS "${DEFAULT_KERNEL_CL_FLAGS} -DPOCL_VECMATHLIB_BUILTIN")
  set(DEFAULT_KERNEL_CLANGXX_FLAGS "${DEFAULT_KERNEL_CLANGXX_FLAGS} -DPOCL_VECMATHLIB_BUILTIN")
endif()

if(NOT DEFINED KERNEL_CLANGXX_FLAGS)
  set(KERNEL_CLANGXX_FLAGS "${DEFAULT_KERNEL_CLANGXX_FLAGS}")
endif()
if(NOT DEFINED KERNEL_CL_FLAGS)
  set(KERNEL_CL_FLAGS "${DEFAULT_KERNEL_CL_FLAGS}")
endif()

message(STATUS "Clang++ flags for compiling kernel library: ${KERNEL_CLANGXX_FLAGS}")
message(STATUS "OpenCL flags for compiling kernel library: ${KERNEL_CL_FLAGS}")


######################################################################################

# DONE
option(STATIC_LLVM "Link LLVM statically. Default is dynamic." OFF)

if(STATIC_LLVM)
  message(STATUS "Linking LLVM statically")
  set(POCL_LLVM_LIBS ${LLVM_LIBFILES})
else()
  message(STATUS "Trying to link LLVM dynamically")
  find_library(LLVM_SHARED_LIB_FILE "LLVM-${LLVM_VERSION}" PATHS "${LLVM_LIBDIR}" NO_DEFAULT_PATH)
  if(LLVM_SHARED_LIB_FILE AND EXISTS "${LLVM_SHARED_LIB_FILE}")
    message(STATUS "..using ${LLVM_SHARED_LIB_FILE}")
    set(POCL_LLVM_LIBS "${LLVM_SHARED_LIB_FILE}")
  else()
    message(STATUS "LLVM-${LLVM_VERSION} library not found (Probably because LLVM is built with cmake).
    Linking libpocl (possibly statically) to LLVM_LIBFILES")
    set(POCL_LLVM_LIBS ${LLVM_LIBFILES})
  endif()
endif()

if (MSVC)
  string(REPLACE "-L${LLVM_LIBDIR}" "" LLVM_LDFLAGS "${LLVM_LDFLAGS}")
  string(STRIP "${LLVM_LDFLAGS}" LLVM_LDFLAGS)
endif()

######################################################################################

option(CUSTOM_BUFFER_ALLOCATOR "Use a custom OpenCL optimized region-based memory allocator instead of allocating buffers with malloc directly" ON)

######################################################################################

# examples are in their own cmakelists.txt

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
    if(LTPOSITION LESS "0")
      include_directories("${LTDL_H_INCLUDE_DIR}")
    endif()

  else()
    message(FATAL_ERROR "Could not find LTDL library!")
  endif()
endif()

######################################################################################

if (MSVC)
  set(CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake)
  find_package( PthreadsWin32 )
  if(NOT Pthreads_FOUND)
    message(FATAL_ERROR "Could not find pthreads-win32 libs!")
  endif()
  include_directories("${Pthreads_INCLUDE_DIRS}")
else()
  include(FindThreads)
  if(Threads_FOUND)
    set(LD_FLAGS_BIN ${CMAKE_THREAD_LIBS_INIT})
  else()
    message(FATAL_ERROR "Could not find threading library for this system")
  endif()
endif()


######################################################################################
# DONE

if (MSVC)
  message(STATUS "Building ICD not yet supported on Windows.")
  set(DEFAULT_ENABLE_ICD 0 CACHE INTERNAL "Going to use ICD loader")
  set(DEFAULT_DIRECT_LINKAGE 1 CACHE INTERNAL "Software links directly to pocl")
  set(DEFAULT_TESTS_USE_ICD 0 CACHE INTERNAL "Tests use ICD")
endif()

if(NOT DEFINED DEFAULT_ENABLE_ICD)

  pkg_check_modules(OCL_ICD ocl-icd>=1.3)

  if(OCL_ICD_FOUND)

    set(HAVE_OCL_ICD 1)
    set(OPENCL_FOUND 1 CACHE INTERNAL "opencl ICD/library found")
    set(OPENCL_LIBRARIES "OpenCL" CACHE INTERNAL "opencl ICD/library paths") # duh, why doesnt ocl-icd set this in .pc file ??
    set(DEFAULT_ENABLE_ICD 1 CACHE INTERNAL "ICD loader availability")
    set(DEFAULT_DIRECT_LINKAGE 0 CACHE INTERNAL "direct linkage")
    set(DEFAULT_TESTS_USE_ICD 1 CACHE INTERNAL "tests use ICD")

  else()

    set(HAVE_OCL_ICD 0)
    # fallback to other ICD loaders
    message(STATUS "ocl-icd not found -> trying fallback ICD implementations")
    pkg_check_modules(OPENCL OpenCL>=1.2)
    if(NOT OPENCL_FOUND)
      find_library(OPENCL_LIBRARIES OpenCL)
      # version check the found library
      if(OPENCL_LIBRARIES)
        set(CMAKE_REQUIRED_LIBRARIES "${OPENCL_LIBRARIES}")
        include(CheckFunctionExists)
        CHECK_FUNCTION_EXISTS("clEnqueueFillImage" OPENCL_FOUND)
      endif()
    endif()

    if(OPENCL_FOUND)
      # no ocl-icd, but libopencl
      message(STATUS "libOpenCL (unknown ICD loader) found")
      set(DEFAULT_ENABLE_ICD 1 CACHE INTERNAL "ICD loader availability")
      # tests can't use the icl loader - must use direct linkage
      set(DEFAULT_DIRECT_LINKAGE 1 CACHE INTERNAL "direct linkage")
      set(DEFAULT_TESTS_USE_ICD 0 CACHE INTERNAL "tests use ICD")
    else()
      message(STATUS "No ICD loader of any kind found (or its OpenCL version is <1.2)")
      # no ocl-icd, no libopencl
      set(DEFAULT_ENABLE_ICD 0 CACHE INTERNAL "no ICL loader found availability")
      set(DEFAULT_DIRECT_LINKAGE 1 CACHE INTERNAL "direct linkage")
      set(DEFAULT_TESTS_USE_ICD 0 CACHE INTERNAL "tests use ICD")
    endif()

  endif()

endif()

setup_cached_var(ENABLE_ICD "Using an ICD loader"
  "Requested build with icd, but ICD loader not found! some examples will not work.."
  "ICD loader found, but requested build without it")


######################################################################################
# DONE

# Option to allow to link a OpenCL program directly against libpocl
# instead of going through an ICD Loader
if(NOT DEFINED DIRECT_LINKAGE)
  set(DIRECT_LINKAGE ${DEFAULT_DIRECT_LINKAGE} CACHE BOOL "Allow OpenCL programs to be linked directly against the pocl library instead of using a ICD Loader (OpenCL library), defaults to !ENABLE_ICD")
endif()

message(STATUS "Direct linkage: ${DIRECT_LINKAGE}")

if(NOT DIRECT_LINKAGE)
  if(ENABLE_ICD)
    message(STATUS "Disabling direct linkage, using ICD loader instead")
  else()
    message(FATAL_ERROR "Either ICD or direct-linkage must be enabled !")
  endif()
endif()


######################################################################################
# DONE

if(NOT DEFINED TESTS_USE_ICD)
  set(TESTS_USE_ICD ${DEFAULT_TESTS_USE_ICD} CACHE BOOL "Tests use the ICD loader")
endif()

message(STATUS "Run tests with ICD: ${TESTS_USE_ICD}")

if(TESTS_USE_ICD AND (NOT ENABLE_ICD))
  message(FATAL_ERROR "Cannot use ICD for tests when its not available")
endif()
if((NOT TESTS_USE_ICD) AND (NOT DIRECT_LINKAGE))
  message(FATAL_ERROR "Tests set to link directly, but direct-linkage is disabled")
endif()


######################################################################################
# DONE

# OpenCL headers

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
# DONE

# TODO Check cl.hpp usability. It is broken on a few platforms.

if(DEFINED HAVE_OPENCL_HPP)
  message(STATUS "cl2.hpp (cached): ${HAVE_OPENCL_HPP}")
else()
  message(STATUS "tests will use pocl's own cl2.hpp")
  set(HAVE_OPENCL_HPP ${_OPENCL_HPP} CACHE INTERNAL "cl2.hpp found & usable")
endif()

######################################################################################
# TODO check if this works!

option(PEDANTIC "Compile host library with stricter compiler flags." OFF)
if(PEDANTIC)
  add_compile_options("-Wno-unused-result" "-Werror") # maybe "-Wimplicit"
endif()

######################################################################################
# DONE

if (MSVC)
  message(STATUS "Finding libGLEW binary distribution (http://glew.sourceforge.net/index.html) for Widows not implemented. Consider writing FindGlew.cmake macro.")
  message(STATUS "libGLEW not found. A few tests will not work")
else()
  pkg_check_modules(GLEW glew)
endif()

if(NOT GLEW_FOUND)
  set(HAVE_GLEW 0)
  message(WARNING "libGLEW not found. A few tests will not work")
else()
  set(HAVE_GLEW 1)
endif()

######################################################################################
# DONE

option(KERNEL_CACHE "Enable the kernel compile cache." ON)
if(KERNEL_CACHE)
  set(POCL_BUILD_KERNEL_CACHE 1)
else()
  set(POCL_BUILD_KERNEL_CACHE 0)
endif()
string(TIMESTAMP POCL_BUILD_TIMESTAMP "%d%m%Y%H%M%S")

######################################################################################
# DONE

if("${LLVM_CLANG_VERSION}" MATCHES "SPIR")
  set(CLANG_SPIR 1)
  message(STATUS "Using a SPIR generator Clang from Khronos.")
else()
  set(CLANG_SPIR 0)
  message(STATUS "NOT using SPIR")
endif()


####################################################################

# The Clang flags that are used always when compiling OpenCL C code.
# Note: We use -Wno-format to avoid warnings for printf, where we
# currently have to use %lld for long arguments (to handle 32-bit
# architectures).
set(FORCED_CLFLAGS "-Xclang -ffake-address-space-map -fno-math-errno -fblocks -fno-builtin -fasm -Wno-format")


####################################################################

# line 620

if(UNIX)
  if(APPLE)
    # TODO MACOSX_BUNDLE target prop
    set(ICD_LD_FLAGS "-single_module")
  else()
    set(ICD_LD_FLAGS "-Wl,-Bsymbolic")
  endif()
endif()

####################################################################

# TODO some more work required here

set(DEFAULT_HOST_LLC_FLAGS "-relocation-model=pic")
# TODO host clang flags / llc flags - properly add_compile_options etc
set(DEFAULT_HOST_CLANG_FLAGS "${DEFAULT_HOST_CLANG_FLAGS} ${CLANG_TARGET_OPTION}${LLC_TRIPLE}")
set(DEFAULT_HOST_LLC_FLAGS "${DEFAULT_HOST_LLC_FLAGS} -mtriple=${LLC_TRIPLE}")
if(NOT LLC_HOST_CPU MATCHES "unknown")
  set(DEFAULT_HOST_CLANG_FLAGS "${DEFAULT_HOST_CLANG_FLAGS} ${CLANG_MARCH_FLAG}${LLC_HOST_CPU}")
  set(DEFAULT_HOST_LLC_FLAGS "${DEFAULT_HOST_LLC_FLAGS} -mcpu=${LLC_HOST_CPU}")
endif()

if(LLC_TRIPLE MATCHES "^arm")
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

if(CL_DISABLE_LONG)
  set(DEFAULT_HOST_CLANG_FLAGS "${DEFAULT_HOST_CLANG_FLAGS} -D_CL_DISABLE_LONG")
endif()
if(CL_DISABLE_HALF)
  set(DEFAULT_HOST_CLANG_FLAGS "${DEFAULT_HOST_CLANG_FLAGS} -D_CL_DISABLE_HALF")
endif()

####################################################################

set(HOST_AS_FLAGS "${DEFAULT_HOST_AS_FLAGS}" CACHE STRING "Parameters to as for code generation in the host.")
set(HOST_LD_FLAGS "${DEFAULT_HOST_LD_FLAGS}" CACHE STRING "Parameter to compiler to generate loadable module.")
set(HOST_CLANG_FLAGS "${DEFAULT_HOST_CLANG_FLAGS}" CACHE STRING "Parameters to clang for host compilation.")
set(HOST_LLC_FLAGS "${DEFAULT_HOST_LLC_FLAGS}" CACHE STRING "Parameters to llc for code generation in the host.")

####################################################################

#line 760

set(OCL_TARGETS "host")


# THESE are only used in makefile.am & scripts/pocl*
set(TCE_TARGET_CLANG_FLAGS "" CACHE STRING "Extra parameters to Clang for TCE compilation.")
set(CELL_TARGET_CLANG_FLAGS "" CACHE STRING "Extra parameters to Clang for CELL compilation.")
# TODO HOST_CLANG_FLAGS -> CPU_TARGET_CLANG_FLAGS
#set(HOST_CLANG_FLAGS "" CACHE STRING "Extra parameters to Clang for cpu target compilation.")
#AC_ARG_VAR([TARGET_CLANG_FLAGS],
#           [Parameters to for target compilation.])

set(TCE_TARGET_LLC_FLAGS "" CACHE STRING "Extra parameters to LLVM's llc for TCE compilation.")
set(CELL_TARGET_LLC_FLAGS "" CACHE STRING "Extra parameters to LLVM's llc for CELL compilation.")
# TODO HOST_LLC_FLAGS -> CPU_TARGET_LLC_FLAGS
#set(HOST_LLC_FLAGS "" CACHE STRING "Extra parameters to LLVM's llc for cpu target compilation.")
#AC_DEFINE_UNQUOTED([TARGET_LLC_FLAGS],
#                   [Parameters to llc for code generation in the target.])

# TODO OCL_KERNEL_TARGET -> CPU_TARGET_TRIPLE
# TODO OCL_KERNEL_TARGET_CPU -> OCL_KERNEL_TARGET_CPU
# these are used here:
#lib/CL/devices/basic/basic.c:  dev->llvm_target_triplet = OCL_KERNEL_TARGET;
#lib/CL/devices/basic/basic.c:  dev->llvm_cpu = OCL_KERNEL_TARGET_CPU;
set(OCL_KERNEL_TARGET "${LLC_TRIPLE}") #The kernel target triplet.
set(OCL_KERNEL_TARGET_CPU "${LLC_HOST_CPU}") #The kernel target CPU variant.

# TODO this is required
# mostly used in tests to skip based on cpu
#AC_SUBST([HOST_CPU], [$host_cpu])
#AC_DEFINE_UNQUOTED([HOST_CPU], ["$host_cpu"], [The host CPU type.])


####################################################################

# Determine which device drivers to build.

set(OCL_DRIVERS "basic pthreads")

if(NOT DEFINED DEFAULT_ENABLE_TCE)

  find_program(TCE_CONFIG NAMES "tce-config")
  if(TCE_CONFIG)

    find_program(TCECC NAMES "tcecc")
    set(TCE_USABLE 1)

  endif()

  if(TCE_CONFIG AND TCECC AND TCE_USABLE)

    execute_process(COMMAND "${TCE_CONFIG}" --libs OUTPUT_VARIABLE TCE_LIBS RESULT_VARIABLE RESV)
    string(STRIP "${TCE_LIBS}" TCE_LIBS)
    set(TCE_LIBS "${TCE_LIBS}" CACHE INTERNAL "tce-config --libs")
    # TODO
    set(LD_FLAGS_BIN ${LD_FLAGS_BIN} ${TCE_LIBS})

    execute_process(COMMAND "${TCE_CONFIG}" --includes OUTPUT_VARIABLE TCE_INCLUDES RESULT_VARIABLE RESV)
    string(STRIP "${TCE_INCLUDES}" TCE_INCLUDES)
    set(TCE_INCLUDES "${TCE_INCLUDES}" CACHE INTERNAL "tce-config --includes")

    execute_process(COMMAND "${TCE_CONFIG}" --version OUTPUT_VARIABLE TCE_VERSION RESULT_VARIABLE RESV)
    set(TCE_VERSION "${TCE_VERSION}" CACHE INTERNAL "tce-config --version")

    set(DEFAULT_ENABLE_TCE 1 CACHE INTERNAL "TCE available")

    if(TCE_VERSION MATCHES "trunk")
      set(DEFAULT_ENABLE_TCEMC 1 CACHE INTERNAL "TCEMC available")
    endif()

  else()
    set(DEFAULT_ENABLE_TCE 0 CACHE INTERNAL "TCE available")
    set(DEFAULT_ENABLE_TCEMC 0 CACHE INTERNAL "TCEMC available")
  endif()

endif()

setup_cached_var(ENABLE_TCE "TCE"
  "Requested enabling TCE, but no usable TCE installation found !"
  "TCE is available, but requested disabling it")

if(ENABLE_TCE)
  set(OCL_DRIVERS "${OCL_DRIVERS} tce")
  set(OCL_TARGETS "${OCL_TARGETS} tce")
  if(DEFAULT_ENABLE_TCEMC)
    set(ENABLE_TCEMC 1)
    set(OCL_DRIVERS "${OCL_DRIVERS} tcemc") # TCEMC is a "superset" of TCE (lp:tce) features.
  endif()
else()
  set(ENABLE_TCEMC 0)
endif()

##########################################################

# Check if CellSPU support is found

# The libspe version requirement is not strict. This is the only one tested.
# SPU backend was removed in LLVM 3.3 (and we don't accept LLVM 3.1 anymore in pocl)
# so compile the spu backend only when LLVM 3.2 is found

if(NOT DEFINED DEFAULT_ENABLE_SPU)
  set(_SPU 0)
  if(LLVM_VERSION MATCHES "3.2")
    pkg_check_modules(LIBSPE "libspe2>=2.2.80")
    if(LIBSPE_FOUND)
      set(_SPU 1)
    endif()
  else()
    message(STATUS "Skipping cellspu target, needs llvm 3.2")
  endif()
  set(DEFAULT_ENABLE_SPU ${_SPU} CACHE INTERNAL "Build cell SPU")
endif()

setup_cached_var(ENABLE_SPU "Cell SPU"
  "Requested enabling Cell SPU, but no usable libspe found !"
  "Cell SPU available, but requested disabling it")

if(ENABLE_SPU)
  set(OCL_DRIVERS "${OCL_DRIVERS} spu")
  set(OCL_TARGETS "${OCL_TARGETS} cellspu")
endif()

##########################################################

if(ENABLE_HSA)
  set(OCL_DRIVERS "${OCL_DRIVERS} hsa")
  set(OCL_TARGETS "${OCL_TARGETS} amdgcn")
  # WIP, this is for config.h
  set(BUILD_HSA 1)
endif()

##########################################################

message(STATUS "Building the following device drivers: ${OCL_DRIVERS}")

set(BUILDDIR "${CMAKE_BINARY_DIR}")
set(SRCDIR "${CMAKE_SOURCE_DIR}")

##########################################################

# Checks for library features.

include(CheckSymbolExists)
CHECK_SYMBOL_EXISTS("posix_memalign" "stdlib.h" HAVE_POSIX_MEMALIGN)

if(NOT CMAKE_CROSSCOMPILING)
  # AC_C_BIGENDIAN
  include(TestBigEndian)
  TEST_BIG_ENDIAN(WORDS_BIGENDIAN)
else()
  # Set default as little-endian
  set(WORDS_BIGENDIAN 0)
endif()

##########################################################

# TODO

#AC_MSG_NOTICE([Checking host compiler characteristics])
#message(STATUS "Checking host compiler characteristics")

#CFLAGS="-h $host"
# -> TRIPLE = LLC_TRIPLE

# HOST_SIZEOF_* is actually unused; some code still uses SIZEOF_* though.

#AC_CHECK_SIZEOF([long])
#AC_SUBST([HOST_SIZEOF_LONG], )
CHECK_SIZEOF("__fp16" SIZEOF___FP16 ${LLC_TRIPLE})
CHECK_SIZEOF("void*" SIZEOF_VOID_P ${LLC_TRIPLE})


CHECK_ALIGNOF("float16" "typedef float float16  __attribute__((__ext_vector_type__(16)));" ALIGNOF_FLOAT16 ${LLC_TRIPLE})

CHECK_ALIGNOF("double16" "typedef double double16  __attribute__((__ext_vector_type__(16)));" ALIGNOF_DOUBLE16 ${LLC_TRIPLE})

if(ALIGNOF_FLOAT16 GREATER ALIGNOF_DOUBLE16)
  set(MAX_EXTENDED_ALIGNMENT "${ALIGNOF_FLOAT16}")
else()
  set(MAX_EXTENDED_ALIGNMENT "${ALIGNOF_DOUBLE16}")
endif()

##########################################################

# POCL_DEVICE_ADDRESS_BITS
# Value based on host processor, for basic and pthreads devices
if("${SIZEOF_VOID_P}" STREQUAL "8")
  set(POCL_DEVICE_ADDRESS_BITS 64)
elseif("${SIZEOF_VOID_P}" STREQUAL "4")
  set(POCL_DEVICE_ADDRESS_BITS 32)
else()
  message(FATAL_ERROR "Cannot figure out POCL_DEVICE_ADDRESS_BITS")
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
# dependant. Therefore we need to be a bit verbose here for the pocl.icd file
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

set(LIB_CURRENT_VERSION 6)
set(LIB_REVISION_VERSION 0)
set(LIB_AGE_VERSION 5)

# LIB_FIRST_VERSION=$(($LIB_CURRENT_VERSION - $LIB_AGE_VERSION))
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

# DONE - lib/llvmopencl..
# AC_SUBST([KERNEL_COMPILER_LIB_VERSION], ["3:0:0"])
set(KERNEL_COMPILER_LIB_VERSION "${LIB_CURRENT_VERSION}.0.0")

##########################################################


#TODO
# these vars are copies b/c tons of sources use BUILD_ICD etc
set(BUILD_ICD ${ENABLE_ICD})
set(BUILD_SPU ${ENABLE_SPU})
set(TCE_AVAILABLE ${ENABLE_TCE})
set(TCEMC_AVAILABLE ${ENABLE_TCEMC})
set(_CL_DISABLE_LONG ${CL_DISABLE_LONG})
set(_CL_DISABLE_HALF ${CL_DISABLE_HALF})
set(PACKAGE_VERSION "${POCL_VERSION}")

configure_file("config.h.in.cmake" "config.h" ESCAPE_QUOTES)
include_directories("${CMAKE_BINARY_DIR}")

# autotools compat
file(WRITE "${CMAKE_BINARY_DIR}/install-paths.h" "#define PKGDATADIR \"${POCL_INSTALL_PRIVATE_DATADIR}\"")

##########################################################

file(GENERATE OUTPUT "${CMAKE_BINARY_DIR}/pocl.icd" CONTENT "${POCL_INSTALL_PUBLIC_LIBDIR}/$<TARGET_FILE_NAME:pocl>" CONDITION 1)
install(FILES "${CMAKE_BINARY_DIR}/pocl.icd"
         DESTINATION "${POCL_INSTALL_ICD_VENDORDIR}")

# write icd file for pocl testing
file(MAKE_DIRECTORY "${CMAKE_BINARY_DIR}/ocl-vendors")
file(GENERATE OUTPUT "${CMAKE_BINARY_DIR}/ocl-vendors/pocl-tests.icd" CONTENT "$<TARGET_FILE:pocl>" CONDITION 1)

file(GENERATE OUTPUT "${CMAKE_BINARY_DIR}/CTestCustom.cmake" CONTENT "
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

include_directories("fix-includes" "include")

add_subdirectory("include")

add_subdirectory("lib")

# these are set in lib/cmakelists.txt
message(STATUS "OPENCL_LIBS: ${OPENCL_LIBS}")
message(STATUS "OPENCL_CFLAGS: ${OPENCL_CFLAGS}")

# for tests / examples
set(POCLU_LINK_OPTIONS ${LIBMATH} ${OPENCL_LIBS} "poclu")
message(STATUS "POCLU LINK OPTS: ${POCLU_LINK_OPTIONS}")

# DONE - just pocl-standalone script
add_subdirectory("scripts")

# for tests & examples
if(CMAKE_SYSTEM_PROCESSOR MATCHES "ppc")
  set(POWERPC 1)
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "armv7")
  set(ARMV7 1)
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "armv6")
  set(ARMV6 1)
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "(i386|AMD64|x86_64)")
  if(POCL_DEVICE_ADDRESS_BITS MATCHES "32")
    set(I386 1)
  else()
    set(X86_64 1)
  endif()
endif()


# TODO In progress
add_subdirectory("tests")

# TODO In progress
set(ALL_TESTSUITES "AMD;AMDSDK2.9;opencl-book-samples;Parboil;Piglit;Rodinia;VexCL;ViennaCL;Halide;OpenCV;CloverLeaf;hsa")
add_subdirectory("examples")

add_custom_target(check COMMAND ${CMAKE_CTEST_COMMAND})

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
MESSAGE(STATUS "******* Enabled features:")
MESSAGE(STATUS " ")

MESSAGE(STATUS "CLANG_SPIR: ${CLANG_SPIR}")
MESSAGE(STATUS "CL_DISABLE_LONG (Disable cl_khr_int64): ${CL_DISABLE_LONG}")
MESSAGE(STATUS "DIRECT_LINKAGE: ${DIRECT_LINKAGE}")
MESSAGE(STATUS "ENABLE_DOCS: ${ENABLE_DOCS}")
MESSAGE(STATUS "ENABLE_ICD: ${ENABLE_ICD}")
MESSAGE(STATUS "ENABLE_SPU (Cell SPU enabled): ${ENABLE_SPU}")
MESSAGE(STATUS "ENABLE_TCE: ${ENABLE_TCE}")
MESSAGE(STATUS "ENABLE_TCEMC: ${ENABLE_TCEMC}")
MESSAGE(STATUS "ENABLE_HSA: ${ENABLE_HSA}")
MESSAGE(STATUS "HAVE_CLOCK_GETTIME: ${HAVE_CLOCK_GETTIME}")
MESSAGE(STATUS "HAVE_GLEW: ${HAVE_GLEW}")
MESSAGE(STATUS "HAVE_OPENCL_HPP: ${HAVE_OPENCL_HPP}")
MESSAGE(STATUS "INSTALL_OPENCL_HEADERS (Install our headers): ${INSTALL_OPENCL_HEADERS}")
MESSAGE(STATUS "NEW_PRINTF_WORKS: ${NEW_PRINTF_WORKS}")
MESSAGE(STATUS "OCL_DRIVERS (Drivers built): ${OCL_DRIVERS}")
MESSAGE(STATUS "OCL_TARGETS (Targets built): ${OCL_TARGETS}")
MESSAGE(STATUS "STATIC_LLVM: ${STATIC_LLVM}")
MESSAGE(STATUS "TESTS_USE_ICD: ${TESTS_USE_ICD}")
MESSAGE(STATUS "USE_VECMATHLIB: ${USE_VECMATHLIB}")
MESSAGE(STATUS "Available testsuites: ${ALL_TESTSUITES}")
MESSAGE(STATUS "Enabled testsuites: ${ENABLED_TESTSUITES}")
MESSAGE(STATUS "Kernel caching: ${KERNEL_CACHE}")

MESSAGE(STATUS " ")
MESSAGE(STATUS "******* Programs:")
MESSAGE(STATUS " ")

MESSAGE(STATUS "LLVM_CONFIG: ${LLVM_CONFIG}")
MESSAGE(STATUS "LLVM_OPT: ${LLVM_OPT}")
MESSAGE(STATUS "LLC: ${LLC}")
MESSAGE(STATUS "LLVM_AS: ${LLVM_AS}")
MESSAGE(STATUS "LLVM_LINK: ${LLVM_LINK}")
MESSAGE(STATUS "LLVM_LLI: ${LLVM_LLI}")
MESSAGE(STATUS "WITH_LLVM_CONFIG (User preferred llvm-config): ${WITH_LLVM_CONFIG}")


MESSAGE(STATUS " ")
MESSAGE(STATUS "******* Various Flags:")
MESSAGE(STATUS " ")

MESSAGE(STATUS "CELL_TARGET_CLANG_FLAGS: ${CELL_TARGET_CLANG_FLAGS}")
MESSAGE(STATUS "CELL_TARGET_LLC_FLAGS: ${CELL_TARGET_LLC_FLAGS}")
MESSAGE(STATUS "CLANG_MARCH_FLAG: ${CLANG_MARCH_FLAG}")
MESSAGE(STATUS "CLANG_TARGET_OPTION: ${CLANG_TARGET_OPTION}")
MESSAGE(STATUS "CL_DISABLE_HALF: ${CL_DISABLE_HALF}")
MESSAGE(STATUS "CL_DISABLE_LONG: ${CL_DISABLE_LONG}")
MESSAGE(STATUS "HOST_AS_FLAGS: ${HOST_AS_FLAGS}")
MESSAGE(STATUS "HOST_CLANG_FLAGS: ${HOST_CLANG_FLAGS}")
MESSAGE(STATUS "HOST_LD_FLAGS: ${HOST_LD_FLAGS}")
MESSAGE(STATUS "HOST_LLC_FLAGS: ${HOST_LLC_FLAGS}")
MESSAGE(STATUS "ICD_LD_FLAGS: ${ICD_LD_FLAGS}")
MESSAGE(STATUS "KERNEL_CLANGXX_FLAGS: ${KERNEL_CLANGXX_FLAGS}")
MESSAGE(STATUS "KERNEL_CL_FLAGS: ${KERNEL_CL_FLAGS}")
MESSAGE(STATUS "LD_FLAGS_BIN: ${LD_FLAGS_BIN}")
MESSAGE(STATUS "LIB_API_VERSION: ${LIB_API_VERSION}")
MESSAGE(STATUS "LIB_BUILD_VERSION: ${LIB_BUILD_VERSION}")
MESSAGE(STATUS "LLVM_VERSION: ${LLVM_VERSION}")
MESSAGE(STATUS "LLVM_ASSERTS_BUILD ${LLVM_ASSERTS_BUILD}")
MESSAGE(STATUS "LLVM_CFLAGS: ${LLVM_CFLAGS}")
MESSAGE(STATUS "LLVM_CXXFLAGS: ${LLVM_CXXFLAGS}")
MESSAGE(STATUS "LLVM_CPPFLAGS: ${LLVM_CPPFLAGS}")
MESSAGE(STATUS "LLVM_LDFLAGS: ${LLVM_LDFLAGS}")
MESSAGE(STATUS "LLVM_LIBDIR: ${LLVM_LIBDIR}")
MESSAGE(STATUS "LLVM_INCLUDEDIR: ${LLVM_INCLUDEDIR}")
MESSAGE(STATUS "LLVM_SRC_ROOT: ${LLVM_SRC_ROOT}")
MESSAGE(STATUS "LLVM_OBJ_ROOT: ${LLVM_OBJ_ROOT}")
MESSAGE(STATUS "LLVM_INCLUDE_DIRS: ${LLVM_INCLUDE_DIRS}")
MESSAGE(STATUS "LLVM_SYSLIBS: ${LLVM_SYSLIBS}")
MESSAGE(STATUS "----------- LLVM_LIBS (LLVM SHARED LIBS) --------")
MESSAGE(STATUS "${LLVM_LIBNAMES}")
MESSAGE(STATUS "----------- LLVM_LIBFILES (LLVM STATIC LIBS) --------")
MESSAGE(STATUS "${LLVM_LIBFILES}")
MESSAGE(STATUS "----------- CLANG_LIBS (LLVM SHARED LIBS) --------")
MESSAGE(STATUS "${CLANG_LIBNAMES}")
MESSAGE(STATUS "----------- CLANG_LIBFILES (LLVM STATIC LIBS) --------")
MESSAGE(STATUS "${CLANG_LIBFILES}")
MESSAGE(STATUS "----------- -------------------------------- --------")
MESSAGE(STATUS "POCL_LLVM_LIBS: ${POCL_LLVM_LIBS}")
MESSAGE(STATUS "----------- -------------------------------- --------")
MESSAGE(STATUS "LLVM_ALL_TARGETS: ${LLVM_ALL_TARGETS}")
MESSAGE(STATUS "LLVM_HOST_TARGET: ${LLVM_HOST_TARGET}")
MESSAGE(STATUS "LLC_TRIPLE: ${LLC_TRIPLE}")
MESSAGE(STATUS "LLC_HOST_CPU: ${LLC_HOST_CPU}")
MESSAGE(STATUS "MAX_EXTENDED_ALIGNMENT: ${MAX_EXTENDED_ALIGNMENT}")
MESSAGE(STATUS "OCL_KERNEL_TARGET: ${OCL_KERNEL_TARGET}")
MESSAGE(STATUS "OCL_KERNEL_TARGET_CPU: ${OCL_KERNEL_TARGET_CPU}")
MESSAGE(STATUS "POCL_DEVICE_ADDRESS_BITS: ${POCL_DEVICE_ADDRESS_BITS}")
MESSAGE(STATUS "TCE_TARGET_CLANG_FLAGS: ${TCE_TARGET_CLANG_FLAGS}")
MESSAGE(STATUS "TCE_TARGET_LLC_FLAGS: ${TCE_TARGET_LLC_FLAGS}")


MESSAGE(STATUS " ")
