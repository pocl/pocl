#=============================================================================
#   CMake build system files
#
#   Copyright (c) 2021 Michal Babej / Tampere University
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
  set_source_files_properties(pocl_proxy.hpp pocl_proxy.cpp PROPERTIES LANGUAGE CXX )
endif(MSVC)

add_pocl_device_library("pocl-devices-proxy" pocl_proxy.hpp pocl_proxy.cpp)

# an option parameter to link to libopencl-stub instead.
# source code: https://github.com/krrishnarraj/libopencl-stub
OPTION(PROXY_USE_LIBOPENCL_STUB "link the proxy device to libopencl-stub instead of system opencl" OFF)
if(PROXY_USE_LIBOPENCL_STUB)

    set(STUB_SOURCES libopencl_stub/openclstub.c libopencl_stub/openclstub.h libopencl_stub/rename_stub.h)
    if(MSVC)
        set_source_files_properties( ${STUB_SOURCES} PROPERTIES LANGUAGE CXX )
    endif ()
    target_sources("pocl-devices-proxy" PRIVATE ${STUB_SOURCES})

endif()
