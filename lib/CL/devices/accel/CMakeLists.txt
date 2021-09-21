#=============================================================================
#   CMake build system files
#
#   Copyright (c) 2019 Pekka Jääskeläinen / Tampere University
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

add_compile_options("-std=c++14")


set(ACCEL_SOURCES "accel-shared.h"
                  "accel.cc"
                  "accel.h"
                  "MMAPRegion.cc"
                  "MMAPRegion.h"
                  "EmulateMMAPRegion.cc"
                  "EmulateMMAPRegion.h"
                  "XrtMMAPRegion.h"
                  "accel-emulate.cc"
                  "accel-emulate.h"
                  "almaif-compile.cc"
                  "almaif-compile.h"
                  "almaif-compile-tce.cc"
                  "almaif-compile-tce.h"
                  )

if(DEFINED ENV{XILINX_XRT})
    set(XRT $ENV{XILINX_XRT})
    add_compile_options(-I${XRT}/include)
    set(ACCEL_SOURCES ${ACCEL_SOURCES}
                      "XrtMMAPRegion.cc"
                      )
endif()


if(MSVC)
  set_source_files_properties( ${ACCEL_SOURCES} PROPERTIES LANGUAGE CXX )
endif(MSVC)

add_pocl_device_library(pocl-devices-accel ${ACCEL_SOURCES})

if(DEFINED ENV{XILINX_XRT})
    target_link_libraries(pocl-devices-accel PRIVATE "${XRT}/lib/libxrt_coreutil.so")
endif()


target_link_libraries(pocl-devices-accel PRIVATE "${TCE_LIBS}")
