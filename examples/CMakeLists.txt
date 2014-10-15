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

if(NOT DEFINED ENABLED_TESTSUITES)
  message(STATUS "Disabling all testsuites")
else()

  include(ExternalProject)

  if("${ENABLED_TESTSUITES}" STREQUAL "all")
    message(STATUS "Enabling all testsuites")
    set(ENABLED_TESTSUITES ${ALL_TESTSUITES})
  endif()

  foreach(TESTSUITE IN LISTS ENABLED_TESTSUITES)
    if(ALL_TESTSUITES MATCHES ${TESTSUITE})
      message(STATUS "Enabling testsuite ${TESTSUITE}")
      add_subdirectory("${TESTSUITE}")
    else()
      message(WARNING "Unknown testsuite ${TESTSUITE} requested")
    endif()
  endforeach()

endif()
set(ENABLED_TESTSUITES ${ENABLED_TESTSUITES} PARENT_SCOPE)


add_subdirectory("example1")
add_subdirectory("example1-spir32")
add_subdirectory("example1-spir64")
add_subdirectory("example2")
add_subdirectory("example2a")
add_subdirectory("standalone")
add_subdirectory("scalarwave")
add_subdirectory("trig")

