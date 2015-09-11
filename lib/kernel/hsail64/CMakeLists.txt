#=============================================================================
#   CMake build system files
#
#   Copyright (c) 2015 pocl developers
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

include("bitcode_rules")

set(KERNEL_SOURCES ${SOURCES_WITHOUT_VML})

foreach(FILE printf.c barrier.ll)
  list(REMOVE_ITEM KERNEL_SOURCES "${FILE}")
endforeach()

foreach(FILE barrier.c get_global_id.c get_group_id.c get_local_id.c get_local_size.c)
  list(REMOVE_ITEM KERNEL_SOURCES "${FILE}")
  list(APPEND KERNEL_SOURCES "hsail64/${FILE}")
endforeach()

set(CLANG_FLAGS "-Xclang" "-ffake-address-space-map" "-emit-llvm" "-target" "hsail64")
set(LLC_FLAGS "")

make_kernel_bc(KERNEL_BC "hsail64" ${KERNEL_SOURCES})

# just debug
message(STATUS "HSAIL64 Kernel BC: ${KERNEL_BC}")

# a target is needed...
add_custom_target("kernel_hsail64" ALL
    DEPENDS ${KERNEL_BC} "${CMAKE_BINARY_DIR}/kernellib_hash.h")

install(FILES "${KERNEL_BC}"
        DESTINATION "${POCL_INSTALL_PRIVATE_DATADIR}")
