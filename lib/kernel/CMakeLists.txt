#=============================================================================
#   CMake build system files
#
#   Copyright (c) 2014-2016 pocl developers
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

set(SOURCES_WITHOUT_VML abs.cl
abs_diff.cl
acos.cl
acosh.cl
acospi.cl
add_sat.cl
all.cl
any.cl
as_type.cl
asin.cl
asinh.cl
asinpi.cl
async_work_group_copy.cl
async_work_group_strided_copy.cl
atan.cl
atan2.cl
atan2pi.cl
atanh.cl
atanpi.cl
atomics.cl
barrier.ll
bitselect.cl
cbrt.cl
ceil.cl
clamp.cl
clamp_int.cl
clz.cl
convert_type.cl
copysign.cl
cos.cl
cosh.cl
cospi.cl
cross.cl
degrees.cl
distance.cl
divide.cl
dot.cl
erf.cl
erfc.cl
exp.cl
exp10.cl
exp2.cl
expm1.cl
fabs.cl
fast_distance.cl
fast_length.cl
fast_normalize.cl
fdim.cl
floor.cl
fma.cl
fmax.cl
fmin.cl
fmod.cl
fract.cl
get_global_id.c
get_global_offset.c
get_global_size.c
get_group_id.c
get_local_id.c
get_local_size.c
get_num_groups.c
get_work_dim.c
hadd.cl
hypot.cl
ilogb.cl
isequal.cl
isfinite.cl
isgreater.cl
isgreaterequal.cl
isinf.cl
isless.cl
islessequal.cl
islessgreater.cl
isnan.cl
isnormal.cl
isnotequal.cl
isordered.cl
isunordered.cl
ldexp.cl
length.cl
lgamma.cl
log.cl
log10.cl
log1p.cl
log2.cl
logb.cl
mad.cl
mad24.cl
mad_hi.cl
mad_sat.cl
max.cl
max_i.cl
maxmag.cl
min.cl
min_i.cl
minmag.cl
mix.cl
mul24.cl
mul_hi.cl
nan.cl
native_cos.cl
native_divide.cl
native_exp.cl
native_exp10.cl
native_exp2.cl
native_log.cl
native_log10.cl
native_log2.cl
native_powr.cl
native_recip.cl
native_rsqrt.cl
native_sin.cl
native_sqrt.cl
native_tan.cl
nextafter.cl
normalize.cl
popcount.cl
pow.cl
pown.cl
powr.cl
prefetch.cl
radians.cl
recip.cl
remainder.cl
rhadd.cl
rint.cl
rootn.cl
rotate.cl
round.cl
rsqrt.cl
select.cl
shuffle.cl
sign.cl
signbit.cl
sin.cl
sincos.cl
sinh.cl
sinpi.cl
smoothstep.cl
sqrt.cl
step.cl
sub_sat.cl
tan.cl
tanh.cl
tanpi.cl
tgamma.cl
trunc.cl
upsample.cl
vload.cl
vload_half.cl
vstore.cl
vstore_half.cl
vload_store_half_f16c.c
wait_group_events.cl)

set(SOURCES_WITH_VML abs.cl
abs_diff.cl
add_sat.cl
all.cl
any.cl
as_type.cl
async_work_group_copy.cl
async_work_group_strided_copy.cl
atomics.cl
barrier.ll
bitselect.cl
clamp_int.cl
clz.cl
convert_type.cl
erf.cl
erfc.cl
get_global_id.c
get_global_offset.c
get_global_size.c
get_group_id.c
get_image_array_size.cl
get_image_channel_data_type.cl
get_image_channel_order.cl
get_image_depth.cl
get_image_height.cl
get_image_width.cl
get_image_dim.cl
get_local_id.c
get_local_size.c
get_num_groups.c
get_work_dim.c
hadd.cl
lgamma.cl
mad24.cl
mad_hi.cl
mad_sat.cl
max.cl
max_i.cl
min.cl
min_i.cl
mul24.cl
mul_hi.cl
nextafter.cl
popcount.cl
prefetch.cl
printf.c
printf_base.c
read_image.cl
rhadd.cl
rotate.cl
select.cl
shuffle.cl
sqrt.cl
sub_sat.cl
tgamma.cl
upsample.cl
vload.cl
vload_half.cl
vstore.cl
vstore_half.cl
vload_store_half_f16c.c
wait_group_events.cl
write_image.cl
vecmathlib-pocl/acos.cc
vecmathlib-pocl/acosh.cc
vecmathlib-pocl/acospi.cl
vecmathlib-pocl/asin.cc
vecmathlib-pocl/asinh.cc
vecmathlib-pocl/asinpi.cl
vecmathlib-pocl/atan.cc
vecmathlib-pocl/atan2.cl
vecmathlib-pocl/atan2pi.cl
vecmathlib-pocl/atanh.cc
vecmathlib-pocl/atanpi.cl
vecmathlib-pocl/cbrt.cc
vecmathlib-pocl/ceil.cc
vecmathlib-pocl/clamp.cl
vecmathlib-pocl/copysign.cc
vecmathlib-pocl/cos.cc
vecmathlib-pocl/cosh.cc
vecmathlib-pocl/cospi.cl
vecmathlib-pocl/cross.cl
vecmathlib-pocl/degrees.cl
vecmathlib-pocl/distance.cl
vecmathlib-pocl/dot.cl
vecmathlib-pocl/exp.cc
vecmathlib-pocl/exp10.cc
vecmathlib-pocl/exp2.cc
vecmathlib-pocl/expm1.cc
vecmathlib-pocl/fabs.cc
vecmathlib-pocl/fast_distance.cl
vecmathlib-pocl/fast_length.cl
vecmathlib-pocl/fast_normalize.cl
vecmathlib-pocl/fdim.cc
vecmathlib-pocl/floor.cc
vecmathlib-pocl/fma.cc
vecmathlib-pocl/fmax.cc
vecmathlib-pocl/fmax.cl
vecmathlib-pocl/fmin.cc
vecmathlib-pocl/fmin.cl
vecmathlib-pocl/fmod.cc
vecmathlib-pocl/fract.cl
vecmathlib-pocl/frexp.cl
vecmathlib-pocl/half_cos.cl
vecmathlib-pocl/half_divide.cl
vecmathlib-pocl/half_exp.cl
vecmathlib-pocl/half_exp10.cl
vecmathlib-pocl/half_exp2.cl
vecmathlib-pocl/half_log.cl
vecmathlib-pocl/half_log10.cl
vecmathlib-pocl/half_log2.cl
vecmathlib-pocl/half_powr.cl
vecmathlib-pocl/half_recip.cl
vecmathlib-pocl/half_rsqrt.cl
vecmathlib-pocl/half_sin.cl
vecmathlib-pocl/half_tan.cl
vecmathlib-pocl/hypot.cc
vecmathlib-pocl/ilogb.cl
vecmathlib-pocl/ilogb_.cc
vecmathlib-pocl/isequal.cl
vecmathlib-pocl/isfinite.cc
vecmathlib-pocl/isgreater.cl
vecmathlib-pocl/isgreaterequal.cl
vecmathlib-pocl/isinf.cc
vecmathlib-pocl/isless.cl
vecmathlib-pocl/islessequal.cl
vecmathlib-pocl/islessgreater.cl
vecmathlib-pocl/isnan.cc
vecmathlib-pocl/isnormal.cc
vecmathlib-pocl/isnotequal.cl
vecmathlib-pocl/isordered.cl
vecmathlib-pocl/isunordered.cl
vecmathlib-pocl/ldexp.cl
vecmathlib-pocl/ldexp_.cc
vecmathlib-pocl/length.cl
vecmathlib-pocl/log.cc
vecmathlib-pocl/log10.cc
vecmathlib-pocl/log1p.cc
vecmathlib-pocl/log2.cc
vecmathlib-pocl/logb.cl
vecmathlib-pocl/mad.cl
vecmathlib-pocl/maxmag.cl
vecmathlib-pocl/minmag.cl
vecmathlib-pocl/mix.cl
vecmathlib-pocl/modf.cl
vecmathlib-pocl/nan.cl
vecmathlib-pocl/native_cos.cl
vecmathlib-pocl/native_divide.cl
vecmathlib-pocl/native_exp.cl
vecmathlib-pocl/native_exp10.cl
vecmathlib-pocl/native_exp2.cl
vecmathlib-pocl/native_log.cl
vecmathlib-pocl/native_log10.cl
vecmathlib-pocl/native_log2.cl
vecmathlib-pocl/native_powr.cl
vecmathlib-pocl/native_recip.cl
vecmathlib-pocl/native_rsqrt.cl
vecmathlib-pocl/native_sin.cl
vecmathlib-pocl/native_sqrt.cl
vecmathlib-pocl/native_tan.cl
vecmathlib-pocl/normalize.cl
vecmathlib-pocl/pow.cc
vecmathlib-pocl/pown.cl
vecmathlib-pocl/powr.cl
vecmathlib-pocl/radians.cl
vecmathlib-pocl/remainder.cc
vecmathlib-pocl/remquo.cl
vecmathlib-pocl/rint.cc
vecmathlib-pocl/rootn.cl
vecmathlib-pocl/round.cc
vecmathlib-pocl/rsqrt.cc
vecmathlib-pocl/sign.cl
vecmathlib-pocl/signbit.cc
vecmathlib-pocl/sin.cc
vecmathlib-pocl/sincos.cl
vecmathlib-pocl/sinh.cc
vecmathlib-pocl/sinpi.cl
vecmathlib-pocl/smoothstep.cl
vecmathlib-pocl/step.cl
vecmathlib-pocl/tan.cc
vecmathlib-pocl/tanh.cc
vecmathlib-pocl/tanpi.cl
vecmathlib-pocl/trunc.cc)


set(SOURCES_WITH_SLEEF abs.cl
abs_diff.cl
add_sat.cl
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
convert_type.cl
cross.cl
distance.cl
dot.cl
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

if(ENABLE_POCL_FLOAT_CONVERSION)
  list(APPEND SOURCES_WITH_VML "errol/errol.c")
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

# vecmathlib headers that should be dependencies
set(VML_KERNEL_DEPEND_HEADERS
"${CMAKE_SOURCE_DIR}/lib/kernel/vecmathlib/vec_test.h"
"${CMAKE_SOURCE_DIR}/lib/kernel/vecmathlib/mathfuncs.h"
"${CMAKE_SOURCE_DIR}/lib/kernel/vecmathlib/mathfuncs_fabs.h"
"${CMAKE_SOURCE_DIR}/lib/kernel/vecmathlib/vec_avx_fp8_32.h"
"${CMAKE_SOURCE_DIR}/lib/kernel/vecmathlib/mathfuncs_sinh.h"
"${CMAKE_SOURCE_DIR}/lib/kernel/vecmathlib/mathfuncs_log.h"
"${CMAKE_SOURCE_DIR}/lib/kernel/vecmathlib/mathfuncs_asin.h"
"${CMAKE_SOURCE_DIR}/lib/kernel/vecmathlib/vec_pseudo.h"
"${CMAKE_SOURCE_DIR}/lib/kernel/vecmathlib/mathfuncs_convert.h"
"${CMAKE_SOURCE_DIR}/lib/kernel/vecmathlib/vec_sse_float1.h"
"${CMAKE_SOURCE_DIR}/lib/kernel/vecmathlib/vec_sse_double2.h"
"${CMAKE_SOURCE_DIR}/lib/kernel/vecmathlib/vec_altivec_float4.h"
"${CMAKE_SOURCE_DIR}/lib/kernel/vecmathlib/mathfuncs_int.h"
"${CMAKE_SOURCE_DIR}/lib/kernel/vecmathlib/mathfuncs_sqrt.h"
"${CMAKE_SOURCE_DIR}/lib/kernel/vecmathlib/mathfuncs_exp.h"
"${CMAKE_SOURCE_DIR}/lib/kernel/vecmathlib/floatbuiltins.h"
"${CMAKE_SOURCE_DIR}/lib/kernel/vecmathlib/mathfuncs_sin.h"
"${CMAKE_SOURCE_DIR}/lib/kernel/vecmathlib/vecmathlib.h"
"${CMAKE_SOURCE_DIR}/lib/kernel/vecmathlib/vec_avx_double4.h"
"${CMAKE_SOURCE_DIR}/lib/kernel/vecmathlib/vec_avx_float8.h"
"${CMAKE_SOURCE_DIR}/lib/kernel/vecmathlib/mathfuncs_asinh.h"
"${CMAKE_SOURCE_DIR}/lib/kernel/vecmathlib/vec_qpx_double4.h"
"${CMAKE_SOURCE_DIR}/lib/kernel/vecmathlib/vec_mic_double8.h"
"${CMAKE_SOURCE_DIR}/lib/kernel/vecmathlib/vec_neon_float2.h"
"${CMAKE_SOURCE_DIR}/lib/kernel/vecmathlib/floatprops.h"
"${CMAKE_SOURCE_DIR}/lib/kernel/vecmathlib/vec_neon_float4.h"
"${CMAKE_SOURCE_DIR}/lib/kernel/vecmathlib/vec_avx_fp16_16.h"
"${CMAKE_SOURCE_DIR}/lib/kernel/vecmathlib/vec_builtin.h"
"${CMAKE_SOURCE_DIR}/lib/kernel/vecmathlib/vec_base.h"
"${CMAKE_SOURCE_DIR}/lib/kernel/vecmathlib/vec_sse_double1.h"
"${CMAKE_SOURCE_DIR}/lib/kernel/vecmathlib/mathfuncs_base.h"
"${CMAKE_SOURCE_DIR}/lib/kernel/vecmathlib/floattypes.h"
"${CMAKE_SOURCE_DIR}/lib/kernel/vecmathlib/vec_mask.h"
"${CMAKE_SOURCE_DIR}/lib/kernel/vecmathlib/vec_vsx_double2.h"
"${CMAKE_SOURCE_DIR}/lib/kernel/vecmathlib/mathfuncs_pow.h"
"${CMAKE_SOURCE_DIR}/lib/kernel/vecmathlib/mathfuncs_rcp.h"
"${CMAKE_SOURCE_DIR}/lib/kernel/vecmathlib/vec_sse_float4.h"
)

set(KERNEL_BC_LIST "")
set(KERNEL_TARGET_LIST "")

#*********************************************************************

if(OCL_TARGETS MATCHES "host")
  add_subdirectory("host")
endif()

#*********************************************************************

if(OCL_TARGETS MATCHES "tce")
  add_subdirectory("tce")
endif()

#*********************************************************************

if(OCL_TARGETS MATCHES "amdgcn")
  add_subdirectory("amdgcn")
endif()

if(OCL_TARGETS MATCHES "hsail64")
  add_subdirectory("hsail64")
endif()

#*********************************************************************

if(OCL_TARGETS MATCHES "cuda")
  add_subdirectory("cuda")
endif()

#*********************************************************************

# "Escape" a list before passing to an external command
string(REPLACE ";" "****" KERNEL_BC_LIST_ESCAPED "${KERNEL_BC_LIST}")

add_custom_command( OUTPUT "${CMAKE_BINARY_DIR}/kernellib_hash.h"
  COMMAND "${CMAKE_COMMAND}"
      -DKERNEL_BC_LIST_ESCAPED='${KERNEL_BC_LIST_ESCAPED}'
      -DINCLUDEDIR='${CMAKE_SOURCE_DIR}/include'
      -DOUTPUT='${CMAKE_BINARY_DIR}/kernellib_hash.h'
      -P "${CMAKE_SOURCE_DIR}/cmake/kernellib_hash.cmake"
  DEPENDS ${KERNEL_BC_LIST}
      "${CMAKE_SOURCE_DIR}/include/_kernel.h"
      "${CMAKE_SOURCE_DIR}/include/_kernel_c.h"
      "${CMAKE_SOURCE_DIR}/include/pocl_types.h"
  COMMENT "Generating SHA1 of all kernel libs..."
  VERBATIM)


add_custom_target("kernellib_hash" ALL
    DEPENDS "${CMAKE_BINARY_DIR}/kernellib_hash.h")
foreach(DEP IN LISTS KERNEL_TARGET_LIST)
  add_dependencies("kernellib_hash" ${DEP})
endforeach()
