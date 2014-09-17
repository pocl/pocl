#.rst:
# FindPthreadsWin32 library
# -------------------------
#
# Try to find pthreads libraries.
# https://sourceware.org/pthreads-win32/
#
# You may declare PTHREADS_ROOT environment variable to tell where
# your library is installed. 
#
# Once done this will define::
#
#   Pthreads_FOUND          - True if pthreads was found
#   Pthreads_INCLUDE_DIRS   - include directories for pthreads
#   Pthreads_LIBRARIES      - link against this library to use pthreads
#
# The module will also define two cache variables::
#
#   Pthreads_INCLUDE_DIR    - the pthreads include directory
#   Pthreads_LIBRARY        - the path to the pthreads library
#

find_path(Pthreads_INCLUDE_DIR
  NAMES
    pthread.h
  PATHS
    ENV "PROGRAMFILES(X86)"
    ENV PTHREADS_ROOT
  PATH_SUFFIXES
    include
)

if(CMAKE_SIZEOF_VOID_P EQUAL 8)
  set(LIB_PATH lib/x64)
else()
  set(LIB_PATH lib/x86)
endif()

find_library(Pthreads_LIBRARY
  NAMES 
    pthread.lib
    pthreadVC2.lib
    pthreadVC2.lib
  PATHS
    ENV PTHREADS_ROOT
  PATH_SUFFIXES
    ${LIB_PATH}
)

#
# All good
#

set(Pthreads_LIBRARIES ${Pthreads_LIBRARY})
set(Pthreads_INCLUDE_DIRS ${Pthreads_INCLUDE_DIR})

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(
  Pthreads
  FOUND_VAR Pthreads_FOUND
  REQUIRED_VARS Pthreads_LIBRARY Pthreads_INCLUDE_DIR
  VERSION_VAR Pthreads_VERSION_STRING)

mark_as_advanced(
  Pthreads_INCLUDE_DIR
  Pthreads_LIBRARY)

