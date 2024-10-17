#=============================================================================
#
#   CMake build system files
#
#   Copyright (c) 2019-2023 pocl developers
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

cmake_minimum_required(VERSION 3.0)
project(pocld)

add_definitions(-DCL_TARGET_OPENCL_VERSION=${OPENCL_HEADER_VERSION}
        -DCL_HPP_TARGET_OPENCL_VERSION=${OPENCL_HEADER_VERSION})

set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../cmake")

set(SOURCES pocld.cc daemon.hh daemon.cc cmdline.h cmdline.c
            ../lib/CL/pocl_debug.c ../lib/CL/pocl_debug.h
            ../lib/CL/pocl_threads.c ../lib/CL/pocl_threads.h
            ../lib/CL/pocl_timing.c ../lib/CL/pocl_timing.h
            ../lib/CL/devices/spirv_parser.hh ../lib/CL/devices/spirv_parser.cc
            ../lib/CL/devices/bufalloc.h ../lib/CL/devices/bufalloc.c
            ../lib/CL/pocl_networking.c ../lib/CL/pocl_networking.h
            ../lib/CL/pocl_runtime_config.c
            shared_cl_context.cc shared_cl_context.hh
            virtual_cl_context.cc virtual_cl_context.hh
            cmd_queue.cc cmd_queue.hh common.cc common.hh
            connection.hh connection.cc request.hh request.cc
            reply_th.cc reply_th.hh request_th.cc request_th.hh
            peer_handler.cc peer_handler.hh
            peer.cc peer.hh tracing.h traffic_monitor.hh traffic_monitor.cc)

# required b/c SHARED libs defaults to ON while OBJECT defaults to OFF
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

unset(P_INCLUDE_DIRECTORIES)
set(P_COMPILE_OPTIONS "-DPOCL_DEBUG_LOG_PREFIX=\"PoCL-D\"")
unset(P_LINK_LIST)

############################################
############################################

option(QUEUE_PROFILING "Enable profiling on server queues. Enabled by default" ON)

# messages.h + pocl_remote.h
list(APPEND P_INCLUDE_DIRECTORIES "${CMAKE_CURRENT_SOURCE_DIR}/../include")

list(APPEND P_INCLUDE_DIRECTORIES ${CMAKE_CURRENT_SOURCE_DIR})
# pocld_config.h
list(APPEND P_INCLUDE_DIRECTORIES ${CMAKE_BINARY_DIR})
# pocl_debug.h + dependencies
list(APPEND P_INCLUDE_DIRECTORIES "${CMAKE_CURRENT_SOURCE_DIR}/../lib/CL")
# spirv_parser.hh
list(APPEND P_INCLUDE_DIRECTORIES "${CMAKE_CURRENT_SOURCE_DIR}/../lib/CL/devices")

############################################

if(RDMA_USE_SVM)
  set(ENABLE_RDMA 1)
endif()

if(ENABLE_RDMA)
    list(APPEND SOURCES
                rdma_request_th.cc rdma_request_th.hh
                rdma_reply_th.cc rdma_reply_th.hh
                rdma.cc rdma.hh)

    find_package(RDMAcm MODULE REQUIRED)
    find_package(Verbs MODULE REQUIRED)
    list(APPEND P_LINK_LIST RDMAcm::RDMAcm IBVerbs::verbs)
endif()

############################################

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
# we're building the server as a separate CMake project

  set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/../cmake")

  set(STANDALONE 1)

  include(sanitizers)

  if (MSVC)
    message(FATAL_ERROR "Builds on Windows are only supported with MinGW and MSYS")
  endif()

  if(ENABLE_LTTNG AND UNIX AND (NOT CMAKE_CROSSCOMPILING))
    find_package(PkgConfig MODULE REQUIRED)
    pkg_check_modules(LTTNG_UST lttng-ust>=2.7)
    if(LTTNG_UST_FOUND)
      set(HAVE_LTTNG_UST 1)
    else()
      message(FATAL_ERROR "ENABLE_LTTNG enabled, but could not find library")
    endif()
  else()
    set(HAVE_LTTNG_UST 0)
  endif()

  if(OPENCL_LIBDIR)
    message(STATUS "Searching for OpenCL in extra paths: ${OPENCL_LIBDIR}")
    find_library(OPENCL NAMES OpenCL PATHS "${OPENCL_LIBDIR}" NO_DEFAULT_PATH)
  endif()
  find_library(OPENCL NAMES OpenCL)
  if (NOT OPENCL)
    message(FATAL_ERROR "Can't find libOpenCL")
  endif()
  message(STATUS "Linking pocld against this libOpenCL: ${OPENCL}")

  find_package(Threads REQUIRED)

  find_file(
    HAVE_LINUX_VSOCK_H
    NAMES
      linux/vm_sockets.h
    PATHS
      /usr/include
  )

  list(APPEND P_LINK_LIST ${OPENCL} Threads::Threads)

  set(POCL_INSTALL_PUBLIC_BINDIR "${CMAKE_INSTALL_PREFIX}/bin")

else()
# we're building the server as a part of pocl CMake project
  set(STANDALONE 0)

  list(APPEND P_LINK_LIST ${OPENCL_LIBS})

endif()

############################################

# pocld_config.h

configure_file("${CMAKE_CURRENT_SOURCE_DIR}/pocld_config.h.in.cmake" "${CMAKE_BINARY_DIR}/pocld_config.h" NEWLINE_STYLE UNIX)

############################################

if(SANITIZER_OPTIONS)
  if(ENABLE_ICD)
    message(FATAL_ERROR "Sanitizers don't work with ICD")
  endif()
  list(INSERT P_LINK_LIST 0 ${SANITIZER_LIBS})
endif()

############################################

if(HAVE_LTTNG_UST)
  list(APPEND P_COMPILE_OPTIONS ${LTTNG_UST_CFLAGS})
  list(APPEND SOURCES "pocld_lttng.c" "pocld_lttng.h")
  list(APPEND P_LINK_LIST ${LTTNG_UST_LDFLAGS})
endif()

###############################################################
###############################################################

add_executable(pocld ${SOURCES})

if(P_COMPILE_OPTIONS)
  message(STATUS "COMPILE OPTIONS: ${P_COMPILE_OPTIONS}")
  target_compile_options(pocld PRIVATE ${P_COMPILE_OPTIONS})
endif()
if(P_LINK_LIST)
  message(STATUS "LINK LIST: ${P_LINK_LIST}")
  target_link_libraries(pocld PRIVATE ${P_LINK_LIST})
endif()
if(P_INCLUDE_DIRECTORIES)
  message(STATUS "INCLUDE DIRS: ${P_INCLUDE_DIRECTORIES}")
  target_include_directories(pocld PRIVATE ${P_INCLUDE_DIRECTORIES})
endif()

target_include_directories(pocld PRIVATE ${CMAKE_BINARY_DIR}) # for config.h

set(CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set_property(TARGET pocld PROPERTY CXX_STANDARD 17)

if (STANDALONE EQUAL 0)
  install(TARGETS pocld RUNTIME
        DESTINATION "${POCL_INSTALL_PUBLIC_BINDIR}")
endif()
