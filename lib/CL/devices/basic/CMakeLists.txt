#=============================================================================
#   CMake build system files
#
#   Copyright (c) 2014-2021 pocl developers
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

set(BASIC_SOURCES "basic.h"
                  "basic.c"
                  )

set(ONEDNN 1)

if(ONEDNN)
  add_compile_options(-I/opt/intel/oneapi/dnnl/2022.1.0/cpu_gomp/include/)
  set(BASIC_SOURCES ${BASIC_SOURCES}
                      "dnn_cpu_ref.h"
                      "dnn_cpu_ref.c"
                      "pocl_onednn.h"
                      "pocl_onednn.c"
                      )
endif(ONEDNN)

if(MSVC)
  set_source_files_properties( ${BASIC_SOURCES} PROPERTIES LANGUAGE CXX )
endif(MSVC)
add_pocl_device_library("pocl-devices-basic" ${BASIC_SOURCES})

if(ONEDNN)
  target_link_libraries(pocl-devices-basic PRIVATE "/opt/intel/oneapi/dnnl/2022.1.0/cpu_gomp/lib/libdnnl.so")
endif(ONEDNN)
