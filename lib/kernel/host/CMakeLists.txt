#=============================================================================
#   CMake build system files
#
#   Copyright (c) 2014-2023 pocl developers
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


set(SOURCES_WITH_SLEEF abs.cl
abs_diff.cl
add_sat.cl
addrspace_operators.ll
all.cl
any.cl
as_type.cl
async_work_group_copy.cl
async_work_group_strided_copy.cl
atomics.cl
barrier.ll
bitselect.cl
clamp.cl
clamp_int.cl
clz.cl
ctz.cl
convert_type.cl
cross.cl
distance.cl
dot.cl
fast_distance.cl
fast_length.cl
fast_normalize.cl
fmax_common.cl
fmin_common.cl
fract.cl
get_global_id.c
get_linear_id.c
get_global_offset.c
get_global_size.c
get_group_id.c
get_image_array_size.cl
get_image_channel_data_type.cl
get_image_channel_order.cl
get_image_depth.cl
get_image_dim.cl
get_image_height.cl
get_image_width.cl
get_local_id.c
get_local_size.c
get_num_groups.c
get_work_dim.c
hadd.cl
half_cos.cl
half_divide.cl
half_exp10.cl
half_exp2.cl
half_exp.cl
half_log10.cl
half_log2.cl
half_log.cl
half_powr.cl
half_recip.cl
half_rsqrt.cl
half_sin.cl
half_sqrt.cl
half_tan.cl
isequal.cl
isgreater.cl
isgreaterequal.cl
isless.cl
islessequal.cl
islessgreater.cl
isnotequal.cl
isordered.cl
isunordered.cl
mad24.cl
mad.cl
mad_hi.cl
mad_sat.cl
max.cl
max_i.cl
maxmag.cl
mem_fence.c
min.cl
min_i.cl
minmag.cl
mix.cl
mul24.cl
mul_hi.cl
nan.cl
native_divide.cl
native_exp10.cl
native_exp2.cl
native_exp.cl
native_log10.cl
native_log2.cl
native_log.cl
native_powr.cl
native_recip.cl
native_rsqrt.cl
native_sqrt.cl
pocl_spawn_wg.c
pocl_run_all_wgs.c
popcount.cl
prefetch.cl
printf.c
printf_base.c
read_image.cl
rhadd.cl
rotate.cl
rsqrt.cl
select.cl
shuffle.cl
signbit.cl
sign.cl
smoothstep.cl
step.cl
sub_sat.cl
upsample.cl
vload.cl
vload_half.cl
vload_store_half_f16c.c
vstore.cl
vstore_half.cl
wait_group_events.cl
write_image.cl

###################################################################

# from libclc

libclc-pocl/pocl_fma.cl
libclc-pocl/acospi.cl
libclc-pocl/asinpi.cl
libclc-pocl/atan2pi.cl
libclc-pocl/atanpi.cl
libclc-pocl/sinpi.cl
libclc-pocl/cospi.cl
libclc-pocl/tanpi.cl
libclc-pocl/cos.cl
libclc-pocl/cosh.cl
libclc-pocl/sin.cl
libclc-pocl/sinh.cl
libclc-pocl/tan.cl
libclc-pocl/tanh.cl
libclc-pocl/sincos.cl
libclc-pocl/sincos_helpers.cl
libclc-pocl/acosh.cl
libclc-pocl/asinh.cl
libclc-pocl/atanh.cl
libclc-pocl/ep_log.cl
libclc-pocl/radians.cl
libclc-pocl/degrees.cl
libclc-pocl/log2.cl
libclc-pocl/logb.cl
# currently unused
#libclc/log1p.cl
#libclc-pocl/frexp.cl
#libclc-pocl/expfrexp.cl
#libclc-pocl/frfrexp.cl
libclc-pocl/pown.cl
libclc-pocl/powr.cl
libclc-pocl/pow.cl
libclc-pocl/rootn.cl
libclc-pocl/pow_helpers.cl
libclc-pocl/fmod.cl
libclc-pocl/remainder.cl
libclc-pocl/remquo.cl
libclc-pocl/ocml_helpers.cl

libclc-pocl/isinf.cl
libclc-pocl/isnan.cl
libclc-pocl/isfinite.cl
libclc-pocl/isnormal.cl


libclc/vtables_fp32.cl
libclc/vtables_fp64.cl
libclc/normalize.cl
libclc/length.cl

###################################################################

sleef/libm/sleef_glue.cl

sleef-pocl/scalars.cl
sleef-pocl/acos.cl
sleef-pocl/asin.cl
sleef-pocl/atan2.cl
sleef-pocl/atan.cl
sleef-pocl/cbrt.cl
sleef-pocl/ceil.cl
sleef-pocl/copysign.cl
sleef-pocl/erfc.cl
sleef-pocl/erf.cl
sleef-pocl/exp10.cl
sleef-pocl/exp2.cl
sleef-pocl/exp.cl
sleef-pocl/expm1.cl
sleef-pocl/fabs.cl
sleef-pocl/fdim.cl
sleef-pocl/floor.cl
sleef-pocl/fma.cl
sleef-pocl/fmax.cl
sleef-pocl/fmin.cl
sleef-pocl/expfrexp.cl
sleef-pocl/frfrexp.cl
sleef-pocl/frexp.cl
sleef-pocl/hypot.cl
sleef-pocl/ilogb.cl
sleef-pocl/ldexp.cl
sleef-pocl/lgamma.cl
sleef-pocl/lgamma_r.cl
sleef-pocl/log10.cl
sleef-pocl/log1p.cl
sleef-pocl/log.cl
sleef-pocl/modf.cl
sleef-pocl/native_cos.cl
sleef-pocl/native_sin.cl
sleef-pocl/native_tan.cl
sleef-pocl/nextafter.cl
sleef-pocl/rint.cl
sleef-pocl/round.cl
sleef-pocl/sqrt.cl
sleef-pocl/tgamma.cl
sleef-pocl/trunc.cl
)

if(HOST_CPU_ENABLE_CL_KHR_FP16)
  list(REMOVE_ITEM SOURCES_WITH_SLEEF
    libclc-pocl/cos.cl
    libclc-pocl/sin.cl
    libclc-pocl/log2.cl
    libclc-pocl/pow.cl
    libclc-pocl/fmod.cl
    sleef-pocl/scalars.cl
    sleef-pocl/ceil.cl
    sleef-pocl/copysign.cl
    sleef-pocl/exp2.cl
    sleef-pocl/exp.cl
    sleef-pocl/fabs.cl
    sleef-pocl/floor.cl
    sleef-pocl/fma.cl
    sleef-pocl/fmax.cl
    sleef-pocl/fmin.cl
    sleef-pocl/log10.cl
    sleef-pocl/log.cl
    sleef-pocl/ldexp.cl
    sleef-pocl/rint.cl
    sleef-pocl/round.cl
    sleef-pocl/sqrt.cl
    sleef-pocl/trunc.cl
    )

  list(APPEND SOURCES_WITH_SLEEF
    fabs.cl
    fma.cl
    fmod.cl
    copysign.cl
    pow.cl
    ceil.cl
    cos.cl
    exp.cl
    exp2.cl
    floor.cl
    fmax.cl
    fmin.cl
    log10.cl
    log2.cl
    log.cl
    ldexp.cl
    rint.cl
    round.cl
    sin.cl
    sqrt.cl
    trunc.cl)
endif()


if(HOST_DEVICE_EXTENSIONS MATCHES "cl_khr_subgroup")
  list(APPEND SOURCES_WITH_SLEEF subgroups.c subgroups.cl)
endif()


include("bitcode_rules")

foreach(FILE "host/addrspace_operators.ll" "host/addrspace_operators_spir.ll")
  if(LLVM_OPAQUE_POINTERS)
    generate_opaque_ptr_ll("${FILE}" "opaque" OPAQ_PTR_FILE)
    list(APPEND SOURCES_WITH_SLEEF "${OPAQ_PTR_FILE}")
  else()
    list(APPEND SOURCES_WITH_SLEEF "${FILE}")
  endif()
endforeach()

if(ENABLE_POCL_FLOAT_CONVERSION)
  list(APPEND SOURCES_WITH_SLEEF "errol/errol.c")
endif()

set(SLEEF_CL_KERNEL_DEPEND_HEADERS "")
foreach(HEADER helperadvsimd.h   helperavx2.h     helperavx.h     helperpurec.h  helpersse2.h    misc.h helperavx2_128.h  helperavx512f.h  helperneon32.h  helpers.h      helpervecext.h)
  list(APPEND SLEEF_CL_KERNEL_DEPEND_HEADERS "${CMAKE_SOURCE_DIR}/lib/kernel/sleef/arch/${HEADER}")
endforeach()
list(APPEND SLEEF_CL_KERNEL_DEPEND_HEADERS "${CMAKE_SOURCE_DIR}/lib/kernel/sleef/include/sleef_cl.h")

set(SLEEF_C_KERNEL_DEPEND_HEADERS ${SLEEF_CL_KERNEL_DEPEND_HEADERS})

# only CL files depend on these
foreach(SOURCE sleef_builtin.c  sleef_glue_auto.c  sleef_glue.cl)
  list(APPEND SLEEF_CL_KERNEL_DEPEND_HEADERS "${CMAKE_SOURCE_DIR}/lib/kernel/sleef/libm/${SOURCE}")
endforeach()

# only C files depend on these
foreach(SOURCE dd.h df.h rename.h rename_vec128.h rename_vec256.h rename_vec512.h )
  list(APPEND SLEEF_C_KERNEL_DEPEND_HEADERS "${CMAKE_SOURCE_DIR}/lib/kernel/sleef/libm/${SOURCE}")
endforeach()
list(APPEND SLEEF_C_KERNEL_DEPEND_HEADERS "${CMAKE_SOURCE_DIR}/lib/kernel/sleef/include/sleef.h")

set(LIBCLC_KERNEL_DEPEND_HEADERS "")
foreach(HEADER ep_log.h misc.h singlevec.h ocml_helpers.h sincos_helpers_fp32.h sincos_helpers_fp64.h vtables.h vtables_macros.h)
  list(APPEND LIBCLC_KERNEL_DEPEND_HEADERS "${CMAKE_SOURCE_DIR}/lib/kernel/libclc/${HEADER}")
endforeach()

#=============================================================================


if(ENABLE_SLEEF)
  set(KERNEL_SOURCES ${SOURCES_WITH_SLEEF})
else()
  set(KERNEL_SOURCES ${SOURCES_GENERIC})
endif()

if(HOST_DEVICE_CL_VERSION_MAJOR GREATER_EQUAL 2)
if(MIPS)
  message(STATUS "OpenCL 2.0 atomics are currently broken on MIPS")
else()
  message(STATUS "Using generic OpenCL 2.0 atomics")
  list(APPEND KERNEL_SOURCES svm_atomics_host.cl svm_atomics.cl)
endif()
endif()

set(HOST_DEVICE_CL_VERSION_3DIGIT "${HOST_DEVICE_CL_VERSION_MAJOR}${HOST_DEVICE_CL_VERSION_MINOR}0")
set(HOST_DEVICE_CL_VERSION_STD  "${HOST_DEVICE_CL_VERSION_MAJOR}.${HOST_DEVICE_CL_VERSION_MINOR}")

set(KERNEL_CL_FLAGS
      "-Wall" "-Wno-unused-local-typedef" "-Xclang"
      "-cl-std=CL${HOST_DEVICE_CL_VERSION_STD}"
      "-D__OPENCL_C_VERSION__=${HOST_DEVICE_CL_VERSION_3DIGIT}"
      "-DENABLE_SLEEF=1"
      ${KERNEL_CL_FLAGS})

# https://bugzilla.mozilla.org/show_bug.cgi?id=1657502
# this *likely* shouldn't be a problem for PoCL, but needs testing
list(APPEND KERNEL_CL_FLAGS "-Wno-psabi")

set(HOST_DEVICE_VERSION_3DIGIT "${HOST_DEVICE_CL_VERSION_MAJOR}${HOST_DEVICE_CL_VERSION_MINOR}0")

separate_arguments(HOST_CLANG_FLAGS)
separate_arguments(HOST_LLC_FLAGS)
set(DEVICE_C_FLAGS "-DPOCL_DEVICE_ADDRESS_BITS=${HOST_DEVICE_ADDRESS_BITS}")
set(DEVICE_CL_FLAGS "-D__OPENCL_VERSION__=${HOST_DEVICE_VERSION_3DIGIT} -DPOCL_DEVICE_ADDRESS_BITS=${HOST_DEVICE_ADDRESS_BITS}")
set(DEVICE_CL_FLAGS "${DEVICE_CL_FLAGS} ${HOST_DEVICE_EXTENSION_DEFINES}")
separate_arguments(DEVICE_CL_FLAGS)
separate_arguments(DEVICE_C_FLAGS)

function(x86_distro_variant_to_flags VARIANT OUT_LLC_FLAGS OUT_CLANG_FLAGS)

  if("${VARIANT}" STREQUAL "sse2")
    set(CLANG_F "${CLANG_MARCH_FLAG}athlon64")
    set(LLC_F "-mcpu=athlon64")

  elseif("${VARIANT}" STREQUAL "ssse3")
    set(CLANG_F "${CLANG_MARCH_FLAG}core2")
    set(LLC_F "-mcpu=core2")

  elseif("${VARIANT}" STREQUAL "sse41")
    set(CLANG_F "${CLANG_MARCH_FLAG}penryn")
    set(LLC_F "-mcpu=penryn")

  elseif("${VARIANT}" STREQUAL "avx")
    set(CLANG_F "${CLANG_MARCH_FLAG}sandybridge")
    set(LLC_F "-mcpu=sandybridge")

  elseif("${VARIANT}" STREQUAL "avx_f16c")
    set(CLANG_F "${CLANG_MARCH_FLAG}ivybridge")
    set(LLC_F "-mcpu=ivybridge")

  elseif("${VARIANT}" STREQUAL "avx_fma4")
    set(CLANG_F "${CLANG_MARCH_FLAG}bdver1")
    set(LLC_F "-mcpu=bdver1")

  elseif("${VARIANT}" STREQUAL "avx2")
    set(CLANG_F "${CLANG_MARCH_FLAG}haswell")
    set(LLC_F "-mcpu=haswell")

  elseif("${VARIANT}" STREQUAL "avx512")
    set(CLANG_F "${CLANG_MARCH_FLAG}skylake-avx512")
    set(LLC_F "-mcpu=skylake-avx512")

  else()
    set(CLANG_F "${CLANG_MARCH_FLAG}${VARIANT}")
    set(LLC_F "-mcpu=${VARIANT}")

  endif()

  set(${OUT_LLC_FLAGS} "${LLC_F}" PARENT_SCOPE)
  set(${OUT_CLANG_FLAGS} "${CLANG_F}" PARENT_SCOPE)
endfunction()

###############################################################################

function(compile_sleef VARIANT SLEEF_CONFIG SLEEF_CONFIG_NEW SLEEF_BC)

  unset(BC_FILE_LIST)

  set(EXTRA_FLAGS "-DDORENAME;-DPURE_C;-I${CMAKE_SOURCE_DIR}/lib/kernel/sleef/include")

# disabled - this code uses libm
#    compile_sleef_c_to_bc("c" "sleef/libm/sleef_builtin.c"
#                          "${VARIANT}" BC_FILE_LIST ${EXTRA_FLAGS})
  compile_sleef_c_to_bc("c" "sleef/libm/sleefsp.c"
                        "${VARIANT}" BC_FILE_LIST ${EXTRA_FLAGS})

  if(HOST_CPU_ENABLE_CL_KHR_FP64)
    compile_sleef_c_to_bc("c" "sleef/libm/sleefdp.c"
                          "${VARIANT}" BC_FILE_LIST ${EXTRA_FLAGS})
  endif()

  compile_sleef_c_to_bc("c" "sleef/libm/sleef_glue_auto.c"
                        "${VARIANT}" BC_FILE_LIST "-include" "${SLEEF_CONFIG}")

  file(READ "${CMAKE_SOURCE_DIR}/lib/kernel/sleef/test.c" TEST_SRC)
  file(READ "${CMAKE_SOURCE_DIR}/lib/kernel/sleef/fma_test.c" FMA_TEST_SRC)

  set(STR "#define SLEEF_ENABLED\n")
  file(APPEND "${SLEEF_CONFIG_NEW}" "${STR}")

  if(HOST_CPU_ENABLE_CL_KHR_FP64)
    set(STR "#define SLEEF_DOUBLE_AVAILABLE\n")
    file(APPEND "${SLEEF_CONFIG_NEW}" "${STR}")
  endif()
  # current SLEEF code does not have code for
  # ARM32 NEON double vectors (if they even exist)
  if(NOT ARM32)
    set(STR "#define SLEEF_DOUBLE_VEC_AVAILABLE\n")
    file(APPEND "${SLEEF_CONFIG_NEW}" "${STR}")
  endif()
  # this is workaround for an inconsistency problem in SLEEF.
  # certain functions (ldexp, ilogb, pown) with double2 type argument
  # take/return int2 types. There are no int2 vectors on x86(-64)
  # but there are on ARM, and the vint type definition in SLEEF is
  # different - on x86 it's defined to be 128bit int32 vector,
  # but that's not the case on ARM where it is a 64bit int32 vector.
  if(X86)
    set(STR "#define SLEEF_VINT_IS_VLONG\n")
    file(APPEND "${SLEEF_CONFIG_NEW}" "${STR}")
  endif()

  # the "runtime/clGetKernelArgInfo" requires working images with spir,
  # that only works if we're using the new (generated) wrappers
  set(HAVE_WORKING_IMAGES_WITH_SPIR OFF CACHE BOOL "image functions work with SPIR" FORCE)
  if(HOST_DEVICE_ADDRESS_BITS EQUAL 32)
    set(SPIR_WRAPPER_FILE "host/spir_wrapper32.ll")
  else()
    if(X86 OR (ARM AND LLVM_VERSION_MAJOR GREATER 11))
      # for X86-64 & ARM, use the generated SPIR wrapper,
      # based on CPU's largest supported vector size
      set(SPIR_WRAPPER_FILE "")
    else()
      # for older LLVMs or other CPUs, use a generic
      # 64bit wrapper, this doesn't support vectors and misses
      # some functions, but otherwise should mostly work
      set(SPIR_WRAPPER_FILE "host/spir_wrapper64.ll")
    endif()
  endif()

  foreach(VECSIZE "512" "256" "128")

    set(EXTRA_FLAGS "-DDORENAME;-DVEC${VECSIZE}")
    custom_try_compile_any(1 "${CLANG}" "c" "${TEST_SRC}" RES
      ${CLANG_FLAGS} ${EXTRA_FLAGS} "-c")

    if(${RES} EQUAL "0")
      compile_sleef_c_to_bc("v${VECSIZE}" "sleef/libm/sleefsimdsp.c"
                            "${VARIANT}" BC_FILE_LIST ${EXTRA_FLAGS})
      if(NOT ARM32)
      compile_sleef_c_to_bc("v${VECSIZE}" "sleef/libm/sleefsimddp.c"
                            "${VARIANT}" BC_FILE_LIST ${EXTRA_FLAGS})
      endif()
      message(STATUS "${VARIANT} SLEEF: ${VECSIZE}bit vectors available.")

      set(STR "#define SLEEF_VEC_${VECSIZE}_AVAILABLE\n")
      file(APPEND "${SLEEF_CONFIG_NEW}" "${STR}")

      if(ENABLE_SPIR AND NOT SPIR_WRAPPER_FILE)
        if (ARM)
          generate_cpu_spir_wrapper(cpu_arm ${VARIANT} ${VECSIZE} SPIR_WRAPPER_FILE)
        endif()
        if(X86)
          generate_cpu_spir_wrapper(cpu_x86 ${VARIANT} ${VECSIZE} SPIR_WRAPPER_FILE)
        endif()
        set(HAVE_WORKING_IMAGES_WITH_SPIR ON CACHE BOOL "image functions work with SPIR" FORCE)
      endif()
    else()
      message(STATUS "${VARIANT} SLEEF: ${VECSIZE}bit vectors NOT available.")
    endif()

    custom_try_compile_any(1 "${CLANG}" "c" "${FMA_TEST_SRC}" RES
      ${CLANG_FLAGS} ${EXTRA_FLAGS} "-c")
    if(${RES} EQUAL "0")
      unset(STR)
      set(STR "#define HAVE_FMA32_${VECSIZE}\n")
      set(STR "${STR}#define HAVE_FMA64_${VECSIZE}\n")
      file(APPEND "${SLEEF_CONFIG_NEW}" "${STR}")
      message(STATUS "${VARIANT} SLEEF: ${VECSIZE}bit hardware FMA available.")
    endif()

  endforeach()

  if(ENABLE_SPIR)
    compile_ll_to_bc("${SPIR_WRAPPER_FILE}" "${VARIANT}" BC_FILE_LIST)
  endif()

  file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/${VARIANT}")
  set(SLEEF_BC "${CMAKE_CURRENT_BINARY_DIR}/${VARIANT}/sleef.bc")
  set(SLEEF_BC "${CMAKE_CURRENT_BINARY_DIR}/${VARIANT}/sleef.bc" PARENT_SCOPE)

  message(STATUS "${VARIANT} SLEEF bc list: ${BC_FILE_LIST}")
  message(STATUS "${VARIANT} SLEEF bc: ${SLEEF_BC}")

  set(LINK_OPT_COMMAND COMMAND "${LLVM_LINK}" "-o" "${SLEEF_BC}" ${BC_FILE_LIST})

  add_custom_command( OUTPUT "${SLEEF_BC}"
    DEPENDS ${BC_FILE_LIST}
    COMMAND ${LINK_OPT_COMMAND}
    COMMENT "Linking & optimizing SLEEF for ${VARIANT}: ${SLEEF_BC}"
    VERBATIM)

endfunction()

###############################################################################

foreach(CPU_VARIANT IN LISTS KERNELLIB_HOST_CPU_VARIANTS)

if(CPU_VARIANT MATCHES "native")
  set(VARIANT "${SELECTED_HOST_CPU}")
else()
  set(VARIANT "${CPU_VARIANT}")
endif()

if(X86_64 OR I386)
  x86_distro_variant_to_flags("${VARIANT}" LLC_CPUFLAGS CLANG_CPUFLAGS)
else()
  set(CLANG_CPUFLAGS "${CLANG_MARCH_FLAG}${VARIANT}")
  set(LLC_CPUFLAGS "-mcpu=${VARIANT}")
endif()

separate_arguments(CLANG_CPUFLAGS)
separate_arguments(LLC_CPUFLAGS)
set(CLANG_FLAGS ${HOST_CLANG_FLAGS} ${CLANG_CPUFLAGS}
                "-emit-llvm" "-ffp-contract=off" "-DPRINTF_IMMEDIATE_FLUSH")

set(LLC_FLAGS ${HOST_LLC_FLAGS} ${LLC_CPUFLAGS})

if(ENABLE_SLEEF)

  # write SLEEF config for this CPU
  set(SLEEF_CONFIG "${CMAKE_BINARY_DIR}/sleef_config_temp_${VARIANT}.h")
  set(SLEEF_CONFIG_NEW "${SLEEF_CONFIG}.new")
  set(STR "/* SLEEF library configuration for ${VARIANT} CPU */ \n")
  file(WRITE "${SLEEF_CONFIG_NEW}" "${STR}")

  # compile SLEEF library for the cpu variant
  unset(SLEEF_BC)
  compile_sleef("${VARIANT}" "${SLEEF_CONFIG}" "${SLEEF_CONFIG_NEW}" SLEEF_BC)

  unset(EXTRA_PARAMS)
  if(CMAKE_VERSION VERSION_GREATER "3.2")
    set(EXTRA_PARAMS BYPRODUCTS "${SLEEF_CONFIG}")
  endif()

  add_custom_command(
    OUTPUT "${SLEEF_CONFIG}.witness"
    ${EXTRA_PARAMS}
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
          "${SLEEF_CONFIG_NEW}" "${SLEEF_CONFIG}"
    COMMAND ${CMAKE_COMMAND} -E touch "${SLEEF_CONFIG}.witness"
  )
  add_custom_target("sleef_config_${VARIANT}" DEPENDS "${SLEEF_CONFIG}.witness")

  # compile kernel
  make_kernel_bc(KERNEL_BC "${OCL_KERNEL_TARGET}-${VARIANT}" "${VARIANT}"
                 1 "${SLEEF_BC}" "${SLEEF_CONFIG}" ${KERNEL_SOURCES})

else()
  make_kernel_bc(KERNEL_BC "${OCL_KERNEL_TARGET}-${VARIANT}" "${VARIANT}"
                 0 0 0 ${KERNEL_SOURCES})
endif()

# just debug
message(STATUS "Host Kernel BC for \"${VARIANT}\": ${KERNEL_BC}")

list(APPEND KERNEL_BC_LIST "${KERNEL_BC}")
set(KERNEL_BC_LIST "${KERNEL_BC_LIST}" PARENT_SCOPE)

# a target is needed...
add_custom_target("kernel_host_${VARIANT}" DEPENDS ${KERNEL_BC})

list(APPEND KERNEL_TARGET_LIST "kernel_host_${VARIANT}")
set(KERNEL_TARGET_LIST "${KERNEL_TARGET_LIST}" PARENT_SCOPE)

install(FILES "${KERNEL_BC}"
        DESTINATION "${POCL_INSTALL_PRIVATE_DATADIR}" COMPONENT "lib")

endforeach()
