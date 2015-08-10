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

if(USE_VECMATHLIB)
  set(KERNEL_SOURCES ${SOURCES_WITH_VML})
else()
  set(KERNEL_SOURCES ${SOURCES_WITHOUT_VML})
endif()

foreach(FILE printf.c get_local_id.c get_local_size.c get_group_id.c get_global_size.c
    get_num_groups.c get_work_dim.c get_global_id.c barrier.ll)
  list(REMOVE_ITEM KERNEL_SOURCES "${FILE}")
endforeach()

foreach(FILE get_local_id.ll get_local_size.ll get_group_id.ll get_global_size.ll
  barrier.ll get_num_groups.ll get_work_dim.ll get_global_id.c)
  list(APPEND KERNEL_SOURCES "amdgcn/${FILE}")
endforeach()

# TODO
set(AMDGCN_KERNEL_TARGET "amdgcn--amdhsa")

separate_arguments(HOST_CLANG_FLAGS)
# TODO "-ffp-contract=off" like in HOST ?
# TODO "-target" "amdgcn--amdhsa" "-mcpu=${MCPU}
set(CLANG_FLAGS ${HOST_CLANG_FLAGS} "-Xclang" "-ffake-address-space-map" "-emit-llvm")

# TODO
separate_arguments(HOST_LLC_FLAGS)
set(LLC_FLAGS ${HOST_LLC_FLAGS})

# TODO LLC_flags is used by kernel.bc target, but ld is unused
#LD_FLAGS    = @HOST_LD_FLAGS@

# KERNEL_TARGET = @OCL_KERNEL_TARGET@
make_kernel_bc(KERNEL_BC "${AMDGCN_KERNEL_TARGET}" ${KERNEL_SOURCES})

# just debug
message(STATUS "Host Kernel BC: ${KERNEL_BC}")

# a target is needed...
add_custom_target("kernel_amdgcn" ALL
    DEPENDS ${KERNEL_BC} "${CMAKE_BINARY_DIR}/kernellib_hash.h")

install(FILES "${KERNEL_BC}"
        DESTINATION "${POCL_INSTALL_PRIVATE_DATADIR}")
