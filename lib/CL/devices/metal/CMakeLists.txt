#=============================================================================
#   CMake build system files
#
#   Copyright (c) 2026 PoCL developers
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

set_source_files_properties(pocl-air-gen.cc PROPERTIES
  COMPILE_FLAGS "${LLVM_CXXFLAGS}")
set_source_files_properties(pocl-metal.m PROPERTIES
  COMPILE_FLAGS "-fobjc-arc")
include_directories(${LLVM_INCLUDE_DIRS} ${LLVM_DOWNGRADE_INCLUDE_DIR})

add_pocl_device_library("pocl-devices-metal"
  pocl-metal.m pocl-metal.h pocl-air-gen.cc pocl-air-gen.h)

if(ENABLE_LOADABLE_DRIVERS)
  if(LLVM_PACKAGE_VERSION)
    set(METAL_LLVM_LINK_LIST ${POCL_LLVM_LINK_TARGETS})
  else()
    set(METAL_LLVM_LINK_LIST ${LLVM_LINK_LIBRARIES} ${LLVM_SYSLIBS})
  endif()
  if(LLVM_LINK_DIRS AND (NOT LLVM_LINK_DIRS STREQUAL LLVM_LIBDIR))
    target_link_directories(pocl-devices-metal PRIVATE ${LLVM_LINK_DIRS})
  endif()
  target_link_libraries(pocl-devices-metal PRIVATE ${METAL_LLVM_LINK_LIST}
    ${METAL_FRAMEWORK} ${FOUNDATION_FRAMEWORK} ${LLVM_DOWNGRADE_LIBRARY}
    ${PTHREAD_LIBRARY})
endif()
