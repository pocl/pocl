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

add_compile_options(${TCE_INCLUDES})

set(ACCEL_SOURCES "accel-shared.h"
                  "accel.cc"
                  "accel.h"
                  "Region.cc"
                  "Region.h"
                  "Device.cc"
                  "Device.h"
                  "MMAPRegion.cc"
                  "MMAPRegion.h"
                  "MMAPDevice.cc"
                  "MMAPDevice.h"
                  "EmulationRegion.cc"
                  "EmulationRegion.h"
                  "EmulationDevice.cc"
                  "EmulationDevice.h"
                  "TTASimDevice.h"
                  "XrtDevice.h"
                  "almaif-compile.cc"
                  "almaif-compile.h"
                  "almaif-compile-tce.h"
                  )

if(HAVE_XRT)
    add_compile_options(-I${XRT_INCLUDEDIR})
    set(ACCEL_SOURCES ${ACCEL_SOURCES}
                      "XrtDevice.cc"
                      "XrtRegion.cc"
                      "XrtRegion.h"
                      )
endif()

if(ENABLE_TCE)
    set(ACCEL_SOURCES ${ACCEL_SOURCES}
                      "TTASimDevice.cc"
                      "TTASimRegion.cc"
                      "TTASimRegion.h"
                      "TTASimControlRegion.cc"
                      "TTASimControlRegion.h"
                      "almaif-compile-tce.cc"
                      )
endif()


if(MSVC)
  set_source_files_properties( ${ACCEL_SOURCES} PROPERTIES LANGUAGE CXX )
endif(MSVC)

add_pocl_device_library(pocl-devices-accel ${ACCEL_SOURCES})

if(HAVE_XRT})
    target_link_libraries(pocl-devices-accel PRIVATE "${XRT_LIBDIR}/libxrt_coreutil.so")
endif()

if(ENABLE_TCE)
    target_link_libraries(pocl-devices-accel PRIVATE ${TCE_LIBS})
endif()
