#=============================================================================
#   CMake build system files
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

function(add_pocl_device_library name)
  add_library(${name} SHARED ${ARGN})
  target_link_libraries(${name} PUBLIC ${POCL_PUBLIC_LINK_LIST} PRIVATE ${POCL_LIBRARY_NAME} ${POCL_PRIVATE_LINK_LIST})
  set_target_properties(${name} PROPERTIES PREFIX "lib" SUFFIX ".so")
  install(TARGETS ${name} LIBRARY DESTINATION "${POCL_INSTALL_PRIVATE_LIBDIR}" COMPONENT "lib")
endfunction()

if(BUILD_BASIC)
  add_subdirectory("basic")
endif()

if(BUILD_PTHREAD)
  add_subdirectory("pthread")
endif()

if(BUILD_ACCEL)
  add_subdirectory("accel")
endif()

add_subdirectory("topology")

set(POCL_DEVICES_OBJS "$<TARGET_OBJECTS:pocl-devices>"
  "$<TARGET_OBJECTS:pocl-devices-topology>")

if(ENABLE_TCE)
  include_directories(AFTER "tce")
  add_subdirectory("tce")
endif()

if(ENABLE_HSA)
  include_directories(AFTER "${HSA_INCLUDES}")
  add_subdirectory("hsa")
endif()

if(ENABLE_CUDA)
  add_subdirectory("cuda")
endif()

set(POCL_DEVICES_SOURCES
  devices.h  devices.c
  bufalloc.c
  common.h common.c
  pocl_local_size.h pocl_local_size.c
  bufalloc.h  cpuinfo.c cpuinfo.h)

if(MSVC)
  set_source_files_properties( ${POCL_DEVICES_SOURCES} PROPERTIES LANGUAGE CXX )
endif(MSVC)
add_library("pocl-devices" OBJECT ${POCL_DEVICES_SOURCES})

set(POCL_DEVICES_LINK_LIST ${POCL_DEVICES_LINK_LIST} PARENT_SCOPE)
set(POCL_DEVICES_OBJS ${POCL_DEVICES_OBJS} PARENT_SCOPE)
