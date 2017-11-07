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

include("bitcode_rules")

# Use the libc (newlib) printf() for now because the pocl's
# fails in TCE code gen:
# SplitVectorResult #0: 0x24c5ae0: v8i16,ch = vaarg 0x20628e8, 0x2423ed0, 0x24c59e0, 0x24c4fe0 [ORD=223] [ID=0]
# LLVM ERROR: Do not know how to split the result of this operator!
list(REMOVE_ITEM SOURCES_WITHOUT_VML "printf.c" "atomics.cl")
list(APPEND SOURCES_WITHOUT_VML "printf_constant.c")

set(DEVICE_CL_FLAGS "-D__OPENCL_VERSION__=${TCE_DEVICE_CL_VERSION} ${TCE_DEVICE_EXTENSION_DEFINES}")
separate_arguments(DEVICE_CL_FLAGS)

# Use TARGET flags:
# with some additional CLANG_FLAGS
#CLANG_FLAGS = @TARGET_CLANG_FLAGS@ -Xclang -ffake-address-space-map -emit-llvm -ffp-contract=off
separate_arguments(TCE_TARGET_CLANG_FLAGS)
set(CLANG_FLAGS ${TCE_TARGET_CLANG_FLAGS} "-emit-llvm" "-ffp-contract=off" "-target" "tce-tut-llvm" "-isystem" "${TCE_PREFIX}/tce-llvm/include")

if(POCL_USE_FAKE_ADDR_SPACE_IDS)
list(APPEND CLANG_FLAGS "-Xclang" "-ffake-address-space-map" "-DPOCL_USE_FAKE_ADDR_SPACE_IDS")
endif()

#LLC_FLAGS   = @TARGET_LLC_FLAGS@
separate_arguments(TCE_TARGET_LLC_FLAGS)
set(LLC_FLAGS ${TCE_TARGET_LLC_FLAGS})

set(KERNEL_CL_FLAGS "-D__OPENCL_C_VERSION__=${TCE_DEVICE_CL_VERSION}" "-Xclang" "-cl-std=CL${TCE_DEVICE_CL_STD}")

if(NOT LLVM_OLDER_THAN_4_0)
list(APPEND KERNEL_CL_FLAGS "-Xclang" "-cl-ext=all,-cl_khr_fp64")
endif()

# TODO LLC_flags is used by kernel.bc target, but ld is unused
#LD_FLAGS    = @TARGET_LD_FLAGS@

#KERNEL_TARGET = tce (WRONG)
make_kernel_bc(KERNEL_BC "tce-tut-llvm" "tta" 0 0 0 ${SOURCES_WITHOUT_VML})

# just debug
message(STATUS "TCE Kernel BC: ${KERNEL_BC}")

list(APPEND KERNEL_BC_LIST "${KERNEL_BC}")
set(KERNEL_BC_LIST "${KERNEL_BC_LIST}" PARENT_SCOPE)

# a target is needed...
add_custom_target("kernel_tce" DEPENDS ${KERNEL_BC})

list(APPEND KERNEL_TARGET_LIST "kernel_tce")
set(KERNEL_TARGET_LIST "${KERNEL_TARGET_LIST}" PARENT_SCOPE)

install(FILES "${KERNEL_BC}"
        DESTINATION "${POCL_INSTALL_PRIVATE_DATADIR}")




