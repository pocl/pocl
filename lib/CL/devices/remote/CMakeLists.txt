#=============================================================================
#   CMake build system files
#
#   Copyright (c) 2014-2023 pocl developers
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

set(POCL_REMOTE_CLIENT_SOURCES
    remote.h remote.c communication.h communication.c
      ../../pocl_networking.h ../../pocl_networking.c)

if(ENABLE_RDMA)
    list(APPEND POCL_REMOTE_CLIENT_SOURCES ../../pocl_rdma.h ../../pocl_rdma.c)
endif()

add_pocl_device_library("pocl-devices-remote" ${POCL_REMOTE_CLIENT_SOURCES})

if(ENABLE_LOADABLE_DRIVERS AND ENABLE_RDMA)
  target_link_libraries("pocl-devices-remote" PRIVATE RDMAcm::RDMAcm IBVerbs::verbs)
endif()
