#=============================================================================
#   CMake build system files
#
#   Copyright (c) 2014-2018 pocl developers
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

# half_sqrt.cl

set(TCE_SOURCES
abs.cl
abs_diff.cl
add_sat.cl
all.cl
any.cl
as_type.cl
async_work_group_copy.cl
async_work_group_strided_copy.cl
barrier.ll
bitselect.cl
cbrt.cl
clamp.cl
clamp_int.cl
clz.cl
convert_type.cl
copysign.cl
cross.cl
distance.cl
dot.cl
fabs.cl
fast_distance.cl
fast_length.cl
fast_normalize.cl
fract.cl
get_global_id.c
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
min.cl
min_i.cl
minmag.cl
mix.cl
modf.cl
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
sqrt.cl
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

################

tce/memcpy.c
tce/memcpy.ll

libclc-pocl/pocl_fma.cl

############## might be untested
### new ones

libclc-pocl/expfrexp.cl
libclc-pocl/frfrexp.cl
libclc-pocl/frexp.cl

libclc-pocl/nextafter.cl

libclc-pocl/exp.cl
libclc-pocl/exp2.cl
libclc-pocl/expm1.cl
libclc-pocl/exp10.cl

libclc-pocl/log1p.cl
libclc-pocl/log10.cl
libclc-pocl/log.cl

libclc-pocl/ceil.cl
libclc-pocl/floor.cl
libclc-pocl/fminmax.cl
libclc-pocl/rint.cl
libclc-pocl/round.cl
libclc-pocl/trunc.cl

libclc-pocl/ldexp.cl
libclc-pocl/hypot.cl
libclc-pocl/ilogb.cl
libclc-pocl/fdim.cl

libclc-pocl/atan2.cl
libclc-pocl/atan.cl

tce/acos.cl
tce/asin.cl

)

set(TODO_LIST
sleef-pocl/scalars.cl



sleef-pocl/erfc.cl
sleef-pocl/erf.cl

# builtin
# sleef-pocl/fabs.cl
# sleef-pocl/modf.cl
# sleef-pocl/sqrt.cl

# sleef-pocl/native_cos.cl
# sleef-pocl/native_sin.cl
# sleef-pocl/native_tan.cl

# sleef-pocl/lgamma.cl
# sleef-pocl/lgamma_r.cl
# sleef-pocl/tgamma.cl

)

set(DEVICE_C_FLAGS "-DPOCL_DEVICE_ADDRESS_BITS=32 ${OPAQUE_PTR_FLAGS}")
set(DEVICE_CL_FLAGS "-D__OPENCL_VERSION__=${TCE_DEVICE_CL_VERSION} ${TCE_DEVICE_EXTENSION_DEFINES} -DPOCL_DEVICE_ADDRESS_BITS=32 ${OPAQUE_PTR_FLAGS}")

separate_arguments(DEVICE_CL_FLAGS)
separate_arguments(TCE_TARGET_CLANG_FLAGS)
separate_arguments(TCE_TARGET_LLC_FLAGS)

set(LLC_FLAGS ${TCE_TARGET_LLC_FLAGS})

unset(EXTRA_TCE_CLANG_FLAGS)

set(KERNEL_CL_FLAGS "-D__OPENCL_C_VERSION__=${TCE_DEVICE_CL_VERSION}" "-Xclang" "-cl-std=CL${TCE_DEVICE_CL_STD}" "-Xclang" "-cl-ext=all,-cl_khr_fp64")

########### TCE big endian kernel library

set(TCE_TARGET_TRIPLE "tce-tut-llvm")

set(CLANG_FLAGS ${TCE_TARGET_CLANG_FLAGS} "-DPRINTF_BUFFER_AS_ID=1" "-emit-llvm" "-mllvm" "--disable-loop-idiom-memcpy" "-mllvm" "--disable-loop-idiom-memset" "-fno-vectorize" "-fno-slp-vectorize" "-ffp-contract=off" "-target" ${TCE_TARGET_TRIPLE} "-isystem" "${TCE_PREFIX}/tce-llvm/include" ${EXTRA_TCE_CLANG_FLAGS})

make_kernel_bc(KERNEL_BC ${TCE_TARGET_TRIPLE} "tta" 0 0 0 ${TCE_SOURCES})

message(STATUS "TCE Kernel BC: ${KERNEL_BC}")

list(APPEND KERNEL_BC_LIST "${KERNEL_BC}")
set(KERNEL_BC_LIST "${KERNEL_BC_LIST}" PARENT_SCOPE)

add_custom_target("kernel_tce" DEPENDS ${KERNEL_BC})

list(APPEND KERNEL_TARGET_LIST "kernel_tce")
set(KERNEL_TARGET_LIST "${KERNEL_TARGET_LIST}" PARENT_SCOPE)

install(FILES "${KERNEL_BC}"
        DESTINATION "${POCL_INSTALL_PRIVATE_DATADIR}" COMPONENT "lib")

########### TCE little endian kernel library

set(TCE_TARGET_TRIPLE "tcele-tut-llvm")

set(CLANG_FLAGS ${TCE_TARGET_CLANG_FLAGS} "-DPRINTF_BUFFER_AS_ID=1" "-emit-llvm" "-mllvm" "--disable-loop-idiom-memcpy" "-mllvm" "--disable-loop-idiom-memset" "-fno-vectorize" "-fno-slp-vectorize" "-ffp-contract=off" "-target" ${TCE_TARGET_TRIPLE} "-isystem" "${TCE_PREFIX}/tcele-llvm/include" ${EXTRA_TCE_CLANG_FLAGS})

make_kernel_bc(KERNEL_BC_LE ${TCE_TARGET_TRIPLE} "tta-le" 0 0 0 ${TCE_SOURCES})

message(STATUS "TCE-LE Kernel BC: ${KERNEL_BC_LE}")

list(APPEND KERNEL_BC_LIST "${KERNEL_BC_LE}")
set(KERNEL_BC_LIST "${KERNEL_BC_LIST}" PARENT_SCOPE)

add_custom_target("kernel_tcele" DEPENDS ${KERNEL_BC_LE})

list(APPEND KERNEL_TARGET_LIST "kernel_tcele")
set(KERNEL_TARGET_LIST "${KERNEL_TARGET_LIST}" PARENT_SCOPE)

install(FILES "${KERNEL_BC_LE}"
        DESTINATION "${POCL_INSTALL_PRIVATE_DATADIR}" COMPONENT "lib")
