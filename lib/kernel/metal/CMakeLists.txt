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

# Kernel library for the Metal driver. Both the library and the user programs
# are compiled for the spir64 target, whose address space numbering (private=0,
# global=1, constant=2, local=3) matches AIR's (thread, device, constant,
# threadgroup). The driver lowers the linked module to AIR at build time.

include("bitcode_rules")

set(KERNEL_SOURCES ${SOURCES_GENERIC})

# CPU-only pieces; printf has a Metal-specific buffer allocator (printf.ll)
foreach(FILE printf.c printf_base.c pocl_spawn_wg.c pocl_run_all_wgs.c
        get_linear_id.cl vload_store_half_f16c.c)
  list(REMOVE_ITEM KERNEL_SOURCES "${FILE}")
endforeach()

foreach(FILE
  get_global_id.c get_global_size.c get_group_id.c
  get_local_id.c get_local_size.c get_num_groups.c
  get_global_offset.c get_work_dim.c
  barrier.ll
  )
  list(REMOVE_ITEM KERNEL_SOURCES "${FILE}")
  list(APPEND KERNEL_SOURCES "metal/${FILE}")
endforeach()

list(APPEND KERNEL_SOURCES "metal/get_linear_id.c" "metal/printf.ll"
  "metal/ldexp_half.cl")

# The generic versions of these use Clang builtins that become C library
# calls (there's no libm on Metal): use the libclc-based implementations, as
# the host CPU library does.
foreach(FN acosh asinh atanh logb remainder remquo atan2pi powr rootn)
  list(REMOVE_ITEM KERNEL_SOURCES "${FN}.cl")
  list(APPEND KERNEL_SOURCES "libclc-pocl/${FN}.cl")
endforeach()
list(APPEND KERNEL_SOURCES libclc-pocl/ep_log.cl libclc-pocl/ocml_helpers.cl
  libclc-pocl/frfrexp.cl libclc-pocl/expfrexp.cl metal/lgamma_r.cl
  libclc-pocl/pow_helpers.cl libclc/vtables_fp32.cl libclc/vtables_fp64.cl)
# ...and SLEEF's portable scalar code for the ones the host library takes
# from SLEEF (metal/sleef_libm_fp32.c provides the C library names the
# generic versions call)
list(APPEND KERNEL_SOURCES "metal/sleef_libm_fp32.c")

# half versions of the functions Clang has no FP16 builtins for (atan2 has
# one, but promotedf16.cl defines half atan2 anyway)
list(REMOVE_ITEM KERNEL_SOURCES atan2.cl)
list(APPEND KERNEL_SOURCES metal/atan2.cl)
list(APPEND KERNEL_SOURCES core-math/promotedf16.cl)
foreach(FN acosh asinh atanh expm1 hypot log1p rsqrt)
  list(APPEND KERNEL_SOURCES "core-math/${FN}f16.cl")
endforeach()

# The SPMD (one work-item per thread) versions of the async copies, shared
# with the CUDA driver.
foreach(FILE async_work_group_copy.cl async_work_group_strided_copy.cl
        wait_group_events.cl)
  list(REMOVE_ITEM KERNEL_SOURCES "${FILE}")
  list(APPEND KERNEL_SOURCES "cuda/${FILE}")
endforeach()

set(LLVM_TARGET "spir64-unknown-unknown")

set(CLANG_FLAGS "-ffreestanding" "-emit-llvm" "-target" "${LLVM_TARGET}")

# AIR has no generic address space, so build the library without it: the
# address-space-taking builtins then get one overload per named address space.
set(KERNEL_CL_FLAGS "-Xclang" "-cl-std=CL${METAL_DEVICE_CL_STD}"
  "-D__OPENCL_C_VERSION__=${METAL_DEVICE_CL_VERSION}"
  "-Xclang" "-cl-ext=+all,-__opencl_c_generic_address_space,-__opencl_c_pipes,-__opencl_c_device_enqueue"
  ${KERNEL_CL_FLAGS})

# the only C sources are the Metal ones and SLEEF's single-precision code,
# which must not use double (Apple GPUs lack it)
set(KERNEL_C_FLAGS ${KERNEL_C_FLAGS} "-Xclang" "-cl-single-precision-constant"
  "-I${CMAKE_SOURCE_DIR}/lib/kernel/sleef/arch"
  "-I${CMAKE_SOURCE_DIR}/lib/kernel/sleef/libm"
  "-I${CMAKE_SOURCE_DIR}/lib/kernel/sleef/include")

set(LLC_FLAGS "")
set(DEVICE_CL_FLAGS "-D__OPENCL_VERSION__=${METAL_DEVICE_CL_VERSION} -Dcl_khr_int64")
separate_arguments(METAL_DEVICE_EXTENSIONS)
foreach(EXT ${METAL_DEVICE_EXTENSIONS})
  set(DEVICE_CL_FLAGS "${DEVICE_CL_FLAGS} -D${EXT}")
endforeach()
separate_arguments(DEVICE_CL_FLAGS)

make_kernel_bc(KERNEL_BC "metal-air64" "air64-BCs" 0 0 0 ${KERNEL_SOURCES})

list(APPEND KERNEL_BC_LIST "${KERNEL_BC}")
set(KERNEL_BC_LIST "${KERNEL_BC_LIST}" PARENT_SCOPE)

add_custom_target("kernel_metal_air64" DEPENDS ${KERNEL_BC})
list(APPEND KERNEL_TARGET_LIST "kernel_metal_air64")
set(KERNEL_TARGET_LIST "${KERNEL_TARGET_LIST}" PARENT_SCOPE)

install(FILES "${KERNEL_BC}"
        DESTINATION "${POCL_INSTALL_PRIVATE_DATADIR_REL}" COMPONENT "lib")
