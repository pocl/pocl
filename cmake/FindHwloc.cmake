#.rst:
# FindHwloc
# ----------
#
# Try to find Portable Hardware Locality (hwloc) libraries.
# http://www.open-mpi.org/software/hwloc
#
# You may declare HWLOC_ROOT environment variable to tell where
# your hwloc library is installed. 
#
# Once done this will define::
#
#   Hwloc_FOUND            - True if hwloc was found
#   Hwloc_INCLUDE_DIRS     - include directories for hwloc
#   Hwloc_LIBRARIES        - link against these libraries to use hwloc
#   Hwloc_VERSION          - version
#   Hwloc_CFLAGS           - include directories as compiler flags
#   Hwloc_LDLFAGS          - link paths and libs as compiler flags
#

#=============================================================================
# Copyright 2014 Mikael Lepistö
#
# Distributed under the OSI-approved BSD License (the "License");
#
# This software is distributed WITHOUT ANY WARRANTY; without even the
# implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the License for more information.
#=============================================================================

if(WIN32)
  find_path(Hwloc_INCLUDE_DIR
    NAMES
      hwloc.h
    PATHS
      ENV "PROGRAMFILES(X86)"
      ENV HWLOC_ROOT
    PATH_SUFFIXES
      include
  )

  find_library(Hwloc_LIBRARY
    NAMES 
      libhwloc.lib
    PATHS
      ENV "PROGRAMFILES(X86)"
      ENV HWLOC_ROOT
    PATH_SUFFIXES
      lib
  )

  #
  # Check if the found library can be used to linking 
  #
  SET (_TEST_SOURCE "${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/linktest.c")
  FILE (WRITE "${_TEST_SOURCE}"
    "
    #include <hwloc.h>
    int main()
    { 
      hwloc_topology_t topology;
      int nbcores;
      hwloc_topology_init(&topology);
      hwloc_topology_load(topology);
      nbcores = hwloc_get_nbobjs_by_type(topology, HWLOC_OBJ_CORE);
      hwloc_topology_destroy(topology);
      return 0;
    }
    "
  )

  TRY_COMPILE(_LINK_SUCCESS ${CMAKE_BINARY_DIR} "${_TEST_SOURCE}"
    CMAKE_FLAGS
    "-DINCLUDE_DIRECTORIES:STRING=${Hwloc_INCLUDE_DIR}"
    CMAKE_FLAGS
    "-DLINK_LIBRARIES:STRING=${Hwloc_LIBRARY}"
  )

  IF(NOT _LINK_SUCCESS)
    if(CMAKE_SIZEOF_VOID_P EQUAL 8)
      message(STATUS "You are building 64bit target.")
    ELSE()
      message(STATUS "You are building 32bit code. If you like to build x64 use e.g. -G 'Visual Studio 12 Win64' generator." )
    ENDIF()
    message(FATAL_ERROR "Library found, but linking test program failed.")
  ENDIF()

  #
  # Resolve version if some compiled binary found...
  #
  find_program(HWLOC_INFO_EXECUTABLE
    NAMES 
      hwloc-info
    PATHS
      ENV HWLOC_ROOT 
    PATH_SUFFIXES
      bin
  )
  
  if(HWLOC_INFO_EXECUTABLE)
    execute_process(
      COMMAND ${HWLOC_INFO_EXECUTABLE} "--version" 
      OUTPUT_VARIABLE HWLOC_VERSION_LINE 
      OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    string(REGEX MATCH "([0-9]+.[0-9]+)$" 
      Hwloc_VERSION "${HWLOC_VERSION_LINE}")
    unset(HWLOC_VERSION_LINE)
  endif()
  
  #
  # All good
  #

  set(Hwloc_LIBRARIES ${Hwloc_LIBRARY})
  set(Hwloc_INCLUDE_DIRS ${Hwloc_INCLUDE_DIR})

  include(FindPackageHandleStandardArgs)
  find_package_handle_standard_args(
    Hwloc
    FOUND_VAR Hwloc_FOUND
    REQUIRED_VARS Hwloc_LIBRARY Hwloc_INCLUDE_DIR
    VERSION_VAR Hwloc_VERSION)

  mark_as_advanced(
    Hwloc_INCLUDE_DIR
    Hwloc_LIBRARY)

  foreach(arg ${Hwloc_INCLUDE_DIRS})
    set(Hwloc_CFLAGS "${Hwloc_CFLAGS} /I${arg}")
  endforeach()

  GET_FILENAME_COMPONENT(Hwloc_LIBRARY_DIRS ${Hwloc_LIBRARY} PATH)
  set(Hwloc_LDFLAGS "/L${Hwloc_LIBRARY_DIRS}")

else()

  # Find with pkgconfig
  find_package(PkgConfig)

  if(HWLOC_ROOT)
    set(ENV{PKG_CONFIG_PATH} "${HWLOC_ROOT}/lib/pkgconfig")
  else()
    foreach(PREFIX ${CMAKE_PREFIX_PATH})
      set(PKG_CONFIG_PATH "${PKG_CONFIG_PATH}:${PREFIX}/lib/pkgconfig")
    endforeach()
    set(ENV{PKG_CONFIG_PATH} "${PKG_CONFIG_PATH}:$ENV{PKG_CONFIG_PATH}")
  endif()

  if(hwloc_FIND_REQUIRED)
    set(_hwloc_OPTS "REQUIRED")
  elseif(hwloc_FIND_QUIETLY)
    set(_hwloc_OPTS "QUIET")
  else()
    set(_hwloc_output 1)
  endif()

  if(hwloc_FIND_VERSION)
    if(hwloc_FIND_VERSION_EXACT)
      pkg_check_modules(Hwloc ${_hwloc_OPTS} hwloc=${hwloc_FIND_VERSION})
    else()
      pkg_check_modules(Hwloc ${_hwloc_OPTS} hwloc>=${hwloc_FIND_VERSION})
    endif()
  else()
    pkg_check_modules(Hwloc ${_hwloc_OPTS} hwloc)
  endif()

  if(Hwloc_FOUND)
    include(FindPackageHandleStandardArgs)
    find_package_handle_standard_args(Hwloc DEFAULT_MSG Hwloc_LIBRARIES)

    if(NOT ${Hwloc_VERSION} VERSION_LESS 1.7.0)
      set(Hwloc_GL_FOUND 1)
    endif()

    if(_hwloc_output)
      message(STATUS
        "Found hwloc ${Hwloc_VERSION} in ${Hwloc_INCLUDE_DIRS}:${Hwloc_LIBRARIES}")
    endif()
  endif()
endif()

