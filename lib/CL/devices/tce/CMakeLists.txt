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

add_compile_options(${TCE_INCLUDES})
add_compile_options(${TCE_CXXFLAGS})

add_compile_options("-Wno-deprecated" "-Wno-unused-parameter")

if(ENABLE_TCE)
  add_subdirectory("ttasim")
endif()

if(MSVC)
  set_source_files_properties( tce_common.h tce_common.cc PROPERTIES LANGUAGE CXX )
endif(MSVC)

add_library("pocl-devices-tce" OBJECT tce_common.h tce_common.cc)
set(POCL_DEVICES_OBJS "${POCL_DEVICES_OBJS};$<TARGET_OBJECTS:pocl-devices-tce>" PARENT_SCOPE)

# dist_pkgdata_DATA = tta_device_main.c
install(FILES "tta_device_main.c"
        DESTINATION ${POCL_INSTALL_PRIVATE_HEADER_DIR})
