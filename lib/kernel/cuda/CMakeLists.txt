#=============================================================================
#   CMake build system files
#
#   Copyright (c) 2016 pocl developers
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

set(KERNEL_SOURCES ${SOURCES_GENERIC})

list(REMOVE_ITEM KERNEL_SOURCES printf_base.c)
list(APPEND KERNEL_SOURCES svm_atomics.cl svm_atomics_host.cl)

foreach(FILE
  async_work_group_copy.cl async_work_group_strided_copy.cl
  get_global_id.c get_global_size.c get_group_id.c
  get_local_id.c get_local_size.c get_num_groups.c
  get_global_offset.c
  native_cos.cl native_sin.cl native_tan.cl
  native_log.cl native_log2.cl native_log10.cl
  native_exp.cl native_exp10.cl
  native_powr.cl
  native_divide.cl native_recip.cl
  rsqrt.cl
  printf.c
  wait_group_events.cl
  barrier.ll nvvm_functions.ll subgroup.ll
  )
  list(REMOVE_ITEM KERNEL_SOURCES "${FILE}")
  list(APPEND KERNEL_SOURCES "cuda/${FILE}")
endforeach()


if(ENABLE_SPIRV)
  generate_cuda_spir_wrapper(SPIR_WRAPPER_FILE)
  list(APPEND KERNEL_SOURCES "${SPIR_WRAPPER_FILE}")
endif()

# Select either NVPTX or NVPTX64
if( CMAKE_SIZEOF_VOID_P EQUAL 8 )
  set(LLVM_TARGET nvptx64)
else( CMAKE_SIZEOF_VOID_P EQUAL 8 )
  set(LLVM_TARGET nvptx)
endif( CMAKE_SIZEOF_VOID_P EQUAL 8 )


set(CLANG_FLAGS "-ffreestanding" "-emit-llvm" "-target" "${LLVM_TARGET}")

# Enable all extensions
set(KERNEL_CL_FLAGS "-Xclang" "-cl-std=CL${CUDA_DEVICE_CL_STD}" "-D__OPENCL_C_VERSION__=${CUDA_DEVICE_CL_VERSION}" "-Xclang" "-cl-ext=all" ${KERNEL_CL_FLAGS})

set(LLC_FLAGS "")
set(DEVICE_CL_FLAGS "-D__OPENCL_VERSION__=${CUDA_DEVICE_CL_VERSION} -Dcl_khr_int64")
separate_arguments(CUDA_DEVICE_EXTENSIONS)
foreach(EXT ${CUDA_DEVICE_EXTENSIONS})
  set(DEVICE_CL_FLAGS "${DEVICE_CL_FLAGS} -D${EXT}")
endforeach()
if(CUDA_DEVICE_CL_VERSION_MAJOR GREATER_EQUAL 3)
  separate_arguments(CUDA_DEVICE_FEATURES_30)
  foreach(EXT ${CUDA_DEVICE_FEATURES_30})
    set(DEVICE_CL_FLAGS "${DEVICE_CL_FLAGS} -D${EXT}")
  endforeach()
endif()
separate_arguments(DEVICE_CL_FLAGS)


# Kernel bitcodes generated:
# * <triple>-sm30-ptx60lt
#   * Includes subgroup shuffles.
#   * Excludes subgroup ballot (see details in subgroup_ballot.cl).
#   * Intended for targeting 3.0 <= SM <= 7.5.
# * <triple>-sm30-ptx60-63:
#   * Includes subgroup ballot and shuffles.
#   * Targeted for PTX 6.0 - 6.3. This library uses intrinsics that work
#     for both the LLVM and PTX assembler.
#   * Intended for targeting 3.0 <= SM <= 7.5.
# * <triple>-sm70-ptx64: Supports ballot and shfl
#   * Excludes subgroup shuffles or ballot. The support would require
#     use of .sync variants.
#   * Inteded for targeting SM >=7.0.

make_kernel_bc(KERNEL_BC_SM30_PTX60LT "${LLVM_TARGET}-sm30-ptx60lt"
  "sm30-ptx60-63-BCs" 0 0 0 ${KERNEL_SOURCES}
  "cuda/subgroup_shuffle.cl")

make_kernel_bc(KERNEL_BC_SM30_PTX60_63 "${LLVM_TARGET}-sm30-ptx60-63"
  "sm30-ptx60-63-BCs" 0 0 0 ${KERNEL_SOURCES}
  "cuda/subgroup_shuffle.cl"
  "cuda/subgroup_ballot.cl")

make_kernel_bc(KERNEL_BC_SM70_PTX64 "${LLVM_TARGET}-sm70-ptx64"
 "sm70-ptx64-BCs" 0 0 0 ${KERNEL_SOURCES})

# just debug
#message(STATUS "${LLVM_TARGET} Kernel BC: ${KERNEL_BC_SM30_PTX60_63}")

list(APPEND KERNEL_BC_LIST "${KERNEL_BC_SM30_PTX60LT}")
list(APPEND KERNEL_BC_LIST "${KERNEL_BC_SM30_PTX60_63}")
list(APPEND KERNEL_BC_LIST "${KERNEL_BC_SM70_PTX64}")
set(KERNEL_BC_LIST "${KERNEL_BC_LIST}" PARENT_SCOPE)

# a target is needed...
add_custom_target("kernel_${LLVM_TARGET}_sm30_ptx60lt"
  DEPENDS ${KERNEL_BC_SM30_PTX60LT})
add_custom_target("kernel_${LLVM_TARGET}_sm30_ptx60_63"
  DEPENDS ${KERNEL_BC_SM30_PTX60_63})
add_custom_target("kernel_${LLVM_TARGET}_sm70_ptx64"
  DEPENDS ${KERNEL_BC_SM70_PTX64})

list(APPEND KERNEL_TARGET_LIST "kernel_${LLVM_TARGET}_sm30_ptx60lt")
list(APPEND KERNEL_TARGET_LIST "kernel_${LLVM_TARGET}_sm30_ptx60_63")
list(APPEND KERNEL_TARGET_LIST "kernel_${LLVM_TARGET}_sm70_ptx64")

set(KERNEL_TARGET_LIST "${KERNEL_TARGET_LIST}" PARENT_SCOPE)

install(FILES
  "${KERNEL_BC_SM30_PTX60LT}" "${KERNEL_BC_SM30_PTX60_63}"
  "${KERNEL_BC_SM70_PTX64}"
  DESTINATION "${POCL_INSTALL_PRIVATE_DATADIR_REL}" COMPONENT "lib")
