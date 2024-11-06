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

add_library("pocl-devices-topology" OBJECT pocl_topology.c pocl_topology.h)
harden("pocl-devices-topology")

if(OCL_ICD_INCLUDE_DIRS)
  target_include_directories("pocl-devices-topology" SYSTEM PUBLIC ${OCL_ICD_INCLUDE_DIRS})
endif()

if(ENABLE_HWLOC)
  if(TARGET PkgConfig::HWLOC)
    target_link_libraries("pocl-devices-topology" PRIVATE PkgConfig::HWLOC)
  else()
    target_include_directories("pocl-devices-topology" PRIVATE ${HWLOC_INCLUDE_DIRS})
  endif()
endif()
