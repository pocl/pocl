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

if(MSVC)
  set_source_files_properties( pocl-pthread.h pthread.c pthread_scheduler.c PROPERTIES LANGUAGE CXX )
endif(MSVC)

set(SOURCES pocl-pthread.h pthread.c pthread_scheduler.c)
if(APPLE)
  list(APPEND SOURCES pthread_barrier.c)
endif(APPLE)
add_pocl_device_library(pocl-devices-pthread ${SOURCES})

if(ENABLE_LOADABLE_DRIVERS)
  target_link_libraries(pocl-devices-pthread PRIVATE pocl-devices-basic ${PTHREAD_LIBRARY})
endif()

if(ENABLE_HOST_CPU_DEVICES_OPENMP)
  target_link_libraries(pocl-devices-pthread PRIVATE OpenMP::OpenMP_C)
endif()
