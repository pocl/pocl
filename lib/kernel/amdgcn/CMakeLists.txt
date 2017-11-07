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

foreach(FILE printf.c get_local_id.c get_local_size.c get_group_id.c get_global_size.c
    get_num_groups.c get_work_dim.c get_global_id.c barrier.ll)
  list(REMOVE_ITEM KERNEL_SOURCES "${FILE}")
endforeach()

foreach(FILE get_local_id.ll get_local_size.ll get_group_id.ll get_global_size.ll
  barrier.ll get_num_groups.ll get_work_dim.ll get_global_id.c)
  list(APPEND KERNEL_SOURCES "amdgcn/${FILE}")
endforeach()

set(CLANG_FLAGS "-emit-llvm" "-target" "amdgcn--amdhsa" "-mcpu=kaveri")

if(POCL_USE_FAKE_ADDR_SPACE_IDS)
list(APPEND CLANG_FLAGS "-Xclang" "-ffake-address-space-map")
endif()

set(LLC_FLAGS "")

make_kernel_bc(KERNEL_BC "amdgcn--amdhsa" "gcn" 0 0 0 ${KERNEL_SOURCES})

# just debug
message(STATUS "AMDGCN Kernel BC: ${KERNEL_BC}")

list(APPEND KERNEL_BC_LIST "${KERNEL_BC}")
set(KERNEL_BC_LIST "${KERNEL_BC_LIST}" PARENT_SCOPE)

# a target is needed...
add_custom_target("kernel_amdgcn" DEPENDS ${KERNEL_BC})

list(APPEND KERNEL_TARGET_LIST "kernel_amdgcn")
set(KERNEL_TARGET_LIST "${KERNEL_TARGET_LIST}" PARENT_SCOPE)

install(FILES "${KERNEL_BC}"
        DESTINATION "${POCL_INSTALL_PRIVATE_DATADIR}")
