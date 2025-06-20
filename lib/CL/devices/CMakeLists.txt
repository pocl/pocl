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
    if(OCL_ICD_INCLUDE_DIRS)
      target_include_directories(${name} SYSTEM PUBLIC ${OCL_ICD_INCLUDE_DIRS})
    endif()
    target_link_libraries(${name} PUBLIC ${POCL_PUBLIC_LINK_LIBRARIES} PRIVATE ${POCL_LIBRARY_NAME} ${POCL_PRIVATE_LINK_LIBRARIES})
    if(WIN32)
      set_target_properties(${name} PROPERTIES PREFIX "")
    endif()
    install(TARGETS ${name}
            LIBRARY DESTINATION ${POCL_INSTALL_PRIVATE_LIBDIR_REL}
            COMPONENT "lib"
            RUNTIME DESTINATION ${POCL_INSTALL_PRIVATE_LIBDIR_REL}
            COMPONENT "lib"
    )
  endfunction()

else()

  function(add_pocl_device_library name)
    add_library(${name} OBJECT ${ARGN})
    if(OCL_ICD_INCLUDE_DIRS)
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

if(BUILD_TBB)
  add_subdirectory("tbb")
  set(POCL_DEVICES_OBJS "${POCL_DEVICES_OBJS}"
    "$<TARGET_OBJECTS:pocl-devices-tbb>")
  if(NOT ENABLE_LOADABLE_DRIVERS)
    list(APPEND POCL_DEVICES_LINK_LIST ${TBB_IMPORTED_TARGETS})
  endif()
endif()

if(BUILD_REMOTE_CLIENT)
  add_subdirectory("remote")
  set(POCL_DEVICES_OBJS "${POCL_DEVICES_OBJS}"
    "$<TARGET_OBJECTS:pocl-devices-remote>")

  if(ENABLE_REMOTE_DISCOVERY_AVAHI)
    find_library(AVAHI_COMMON_LIB NAMES avahi-common REQUIRED)
    find_library(AVAHI_CLIENT_LIB NAMES avahi-client REQUIRED)
    list(APPEND POCL_DEVICES_LINK_LIST ${AVAHI_COMMON_LIB} ${AVAHI_CLIENT_LIB})
  endif()

  if(ENABLE_REMOTE_DISCOVERY_DHT)
    find_library(OPENDHT NAMES opendht REQUIRED)
    find_library(OPENDHT_C NAMES opendht-c REQUIRED)
    list(APPEND POCL_DEVICES_LINK_LIST ${OPENDHT} ${OPENDHT_C})
  endif()

  if((NOT ENABLE_LOADABLE_DRIVERS) AND ENABLE_RDMA)
    list(APPEND POCL_DEVICES_LINK_LIST RDMAcm::RDMAcm IBVerbs::verbs)
  endif()
endif()

# for these drivers, use HWLOC if found
if(ENABLE_HOST_CPU_DEVICES)
  # topology is always enabled with CPU drivers, and works also without HWLOC
  add_subdirectory("topology")
  if((NOT ENABLE_LOADABLE_DRIVERS) AND ENABLE_HOST_CPU_DEVICES_OPENMP)
    list(APPEND POCL_DEVICES_LINK_LIST OpenMP::OpenMP_C)
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
  if(HAVE_TREE_SITTER)
    add_subdirectory("tree_sitter")
    list(APPEND POCL_DEVICES_LINK_LIST pocl_tree_sitter_utils tree-sitter)
    set(POCL_DEVICES_OBJS "${POCL_DEVICES_OBJS}"
            "$<TARGET_OBJECTS:pocl_tree_sitter_utils>")
  endif ()
  add_subdirectory("proxy")
  set(POCL_DEVICES_OBJS "${POCL_DEVICES_OBJS}"
    "$<TARGET_OBJECTS:pocl-devices-proxy>")
  # link to libOpenCL(-stub)
  if(NOT PROXY_USE_LIBOPENCL_STUB)
    list(APPEND POCL_DEVICES_LINK_LIST OpenCL)
  endif()

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
    list(APPEND POCL_DEVICES_LINK_LIST PkgConfig::LEVEL0)
  endif()
endif()

if(ENABLE_CUDA)
  add_subdirectory("cuda")
  set(POCL_DEVICES_OBJS "${POCL_DEVICES_OBJS}"
    "$<TARGET_OBJECTS:pocl-devices-cuda>")

  if(NOT ENABLE_LOADABLE_DRIVERS)
    list(APPEND POCL_DEVICES_LINK_LIST ${CUDA_POCL_TARGETS})
  endif()
endif()


set(POCL_DEVICES_SOURCES
  devices.h  devices.c
  bufalloc.c  bufalloc.h
  common.h  common.c
  pocl_local_size.h  pocl_local_size.c
  common_driver.h  common_driver.c
  spirv.hh  spirv_parser.hh  spirv_parser.cc
  spirv_queries.h  spirv_queries.cc
  pocl_spirv_utils.cc pocl_spirv_utils.hh
  printf_base.c printf_base.h printf_buffer.c
  cpuinfo.c  cpuinfo.h)

if(ENABLE_HOST_CPU_DEVICES)
  if(HAVE_LIBXSMM)
    list(APPEND POCL_DEVICES_LINK_LIST PkgConfig::LIBXSMM BLAS::BLAS)
  endif()
  if(ENABLE_HWLOC)
    if(TARGET PkgConfig::HWLOC)
      list(APPEND POCL_DEVICES_LINK_LIST PkgConfig::HWLOC)
    else()
      message(FATAL_ERROR "missing target PkgConfig::HWLOC")
    endif()
  endif()
  add_subdirectory(cpu_dbk)
  list(APPEND POCL_DEVICES_SOURCES common_utils.h common_utils.c)
endif()

if(ENABLE_SIGFPE_HANDLER OR ENABLE_SIGUSR2_HANDLER)
  list(APPEND POCL_DEVICES_SOURCES signal_handlers.c)
endif()

add_library("pocl-devices" OBJECT ${POCL_DEVICES_SOURCES})
if(OCL_ICD_INCLUDE_DIRS)
  target_include_directories("pocl-devices" SYSTEM PUBLIC ${OCL_ICD_INCLUDE_DIRS})
endif()
if(HAVE_LIBXSMM) # for includedir flags
  target_link_libraries("pocl-devices" PRIVATE PkgConfig::LIBXSMM)
endif()
if(HAVE_LIBJPEG_TURBO)
  target_link_libraries("pocl-devices" PRIVATE ${LIBJPEG_TARGET})
endif ()
if(HAVE_ONNXRT)
  target_link_libraries("pocl-devices" PRIVATE onnxruntime::onnxruntime)
endif ()
if(HAVE_OPENCV)
  target_link_libraries("pocl-devices" PRIVATE opencv_dnn)
endif ()
if(BUILD_SHARED_LIBS AND MSVC)
  # Inject __declspec(dllexport).
  target_compile_definitions("pocl-devices" PRIVATE EXPORT_POCL_LIB)
  target_compile_definitions("pocl-devices" INTERFACE IMPORT_POCL_LIB)
endif()


if(HAVE_TREE_SITTER)
  target_link_libraries("pocl-devices" PRIVATE tree-sitter)
endif()

set(POCL_DEVICES_OBJS "${POCL_DEVICES_OBJS}"
    "$<TARGET_OBJECTS:pocl-devices>")
harden("pocl-devices")

set(POCL_DEVICES_LINK_LIST ${POCL_DEVICES_LINK_LIST} PARENT_SCOPE)
set(POCL_DEVICES_OBJS ${POCL_DEVICES_OBJS} PARENT_SCOPE)
