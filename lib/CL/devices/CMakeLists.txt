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

if(ENABLE_LOADABLE_DRIVERS)

  function(add_pocl_device_library name)
    add_library(${name} SHARED ${ARGN})
    harden("${name}")
    if(${OCL_ICD_INCLUDE_DIRS})
      target_include_directories(${name} SYSTEM PUBLIC ${OCL_ICD_INCLUDE_DIRS})
    endif()
    target_link_libraries(${name} PUBLIC ${POCL_PUBLIC_LINK_LIST} PRIVATE ${POCL_LIBRARY_NAME} ${POCL_PRIVATE_LINK_LIST})
    set_target_properties(${name} PROPERTIES PREFIX "lib" SUFFIX ".so")
    install(TARGETS ${name} LIBRARY DESTINATION "${POCL_INSTALL_PRIVATE_LIBDIR}" COMPONENT "lib")
  endfunction()

else()

  function(add_pocl_device_library name)
    add_library(${name} OBJECT ${ARGN})
    if(${OCL_ICD_INCLUDE_DIRS})
      target_include_directories(${name} SYSTEM PUBLIC ${OCL_ICD_INCLUDE_DIRS})
    endif()
  endfunction()

endif()


if(BUILD_BASIC)
  add_subdirectory("basic")
  set(POCL_DEVICES_OBJS "${POCL_DEVICES_OBJS}"
    "$<TARGET_OBJECTS:pocl-devices-basic>")
endif()

if(BUILD_PTHREAD)
  add_subdirectory("pthread")
  set(POCL_DEVICES_OBJS "${POCL_DEVICES_OBJS}"
    "$<TARGET_OBJECTS:pocl-devices-pthread>")
endif()

if(BUILD_ALMAIF)
  add_subdirectory("almaif")
  set(POCL_DEVICES_OBJS "${POCL_DEVICES_OBJS}"
    "$<TARGET_OBJECTS:pocl-devices-almaif>")
endif()

# for these drivers, use HWLOC if found
if(ENABLE_HOST_CPU_DEVICES OR ENABLE_HSA)
  add_subdirectory("topology")
  set(POCL_DEVICES_OBJS "${POCL_DEVICES_OBJS}"
    "$<TARGET_OBJECTS:pocl-devices-topology>")

  if(Hwloc_FOUND)
    if(Hwloc_LDFLAGS)
      list(APPEND POCL_DEVICES_LINK_LIST ${Hwloc_LDFLAGS})
    else()
      list(APPEND POCL_DEVICES_LINK_LIST ${Hwloc_LIBRARIES})
    endif()
  endif()
endif()

if(ENABLE_TCE)
  include_directories(AFTER "tce")
  add_subdirectory("tce")
  set(POCL_DEVICES_OBJS "${POCL_DEVICES_OBJS}"
    "$<TARGET_OBJECTS:pocl-devices-ttasim>")

  if(NOT ENABLE_LOADABLE_DRIVERS)
    list(APPEND POCL_DEVICES_LINK_LIST ${TCE_LIBS})
  endif()
endif()

if(ENABLE_PROXY_DEVICE)
  add_subdirectory("proxy")
  set(POCL_DEVICES_OBJS "${POCL_DEVICES_OBJS}"
    "$<TARGET_OBJECTS:pocl-devices-proxy>")
  list(APPEND POCL_DEVICES_LINK_LIST OpenCL)
endif()

if(ENABLE_HSA)
  include_directories(AFTER "${HSA_INCLUDES}")
  add_subdirectory("hsa")
  set(POCL_DEVICES_OBJS "${POCL_DEVICES_OBJS}"
    "$<TARGET_OBJECTS:pocl-devices-hsa>")

  if(NOT ENABLE_LOADABLE_DRIVERS)
    list(APPEND POCL_DEVICES_LINK_LIST ${HSALIB})
  endif()
endif()

if(ENABLE_VULKAN)
  include_directories(AFTER "${Vulkan_INCLUDE_DIR}")
  add_subdirectory("vulkan")
  set(POCL_DEVICES_OBJS "${POCL_DEVICES_OBJS}"
    "$<TARGET_OBJECTS:pocl-devices-vulkan>")

  if(NOT ENABLE_LOADABLE_DRIVERS)
    list(APPEND POCL_DEVICES_LINK_LIST ${Vulkan_LIBRARY})
  endif()
endif()

if(ENABLE_LEVEL0)
  include_directories(AFTER "${LEVEL0_INCLUDE_DIRS}")
  add_subdirectory("level0")
  set(POCL_DEVICES_OBJS "${POCL_DEVICES_OBJS}"
    "$<TARGET_OBJECTS:pocl-devices-level0>")

  if(NOT ENABLE_LOADABLE_DRIVERS)
    list(APPEND POCL_DEVICES_LINK_LIST ${LEVEL0_LIBRARIES})
  endif()
endif()

if(ENABLE_CUDA)
  add_subdirectory("cuda")
  set(POCL_DEVICES_OBJS "${POCL_DEVICES_OBJS}"
    "$<TARGET_OBJECTS:pocl-devices-cuda>")

  if(NOT ENABLE_LOADABLE_DRIVERS)
    list(APPEND POCL_DEVICES_LINK_LIST cuda)
  endif()
endif()

set(POCL_DEVICES_SOURCES
  devices.h  devices.c
  bufalloc.c  bufalloc.h
  common.h  common.c
  pocl_local_size.h  pocl_local_size.c
  common_driver.h  common_driver.c
  builtin_kernels.cc builtin_kernels.hh
  cpuinfo.c  cpuinfo.h)

if(UNIX AND (CMAKE_SYSTEM_NAME MATCHES "Linux"))
  list(APPEND POCL_DEVICES_SOURCES signal_handlers.c)
endif()

if(MSVC)
  set_source_files_properties( ${POCL_DEVICES_SOURCES} PROPERTIES LANGUAGE CXX )
endif(MSVC)
add_library("pocl-devices" OBJECT ${POCL_DEVICES_SOURCES})
if(${OCL_ICD_INCLUDE_DIRS})
  target_include_directories("pocl-devices" SYSTEM PUBLIC ${OCL_ICD_INCLUDE_DIRS})
endif()
set(POCL_DEVICES_OBJS "${POCL_DEVICES_OBJS}"
    "$<TARGET_OBJECTS:pocl-devices>")
harden("pocl-devices")

set(POCL_DEVICES_LINK_LIST ${POCL_DEVICES_LINK_LIST} PARENT_SCOPE)
set(POCL_DEVICES_OBJS ${POCL_DEVICES_OBJS} PARENT_SCOPE)
