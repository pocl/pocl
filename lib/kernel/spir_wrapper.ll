; ModuleID = 'spir_wrapper.bc'
source_filename = "generate_spir_wrapper.py"
target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024"
target triple = "spir64-unknown-unknown"

declare float @_Z8_cl_acosf(float) local_unnamed_addr #0

define spir_func float @_Z4acosf(float %x) local_unnamed_addr #0 {
  %call = tail call float @_Z8_cl_acosf(float %x)
  ret float %call
}


declare float @_Z9_cl_acoshf(float) local_unnamed_addr #0

define spir_func float @_Z5acoshf(float %x) local_unnamed_addr #0 {
  %call = tail call float @_Z9_cl_acoshf(float %x)
  ret float %call
}


declare float @_Z10_cl_acospif(float) local_unnamed_addr #0

define spir_func float @_Z6acospif(float %x) local_unnamed_addr #0 {
  %call = tail call float @_Z10_cl_acospif(float %x)
  ret float %call
}


declare float @_Z8_cl_asinf(float) local_unnamed_addr #0

define spir_func float @_Z4asinf(float %x) local_unnamed_addr #0 {
  %call = tail call float @_Z8_cl_asinf(float %x)
  ret float %call
}


declare float @_Z9_cl_asinhf(float) local_unnamed_addr #0

define spir_func float @_Z5asinhf(float %x) local_unnamed_addr #0 {
  %call = tail call float @_Z9_cl_asinhf(float %x)
  ret float %call
}


declare float @_Z10_cl_asinpif(float) local_unnamed_addr #0

define spir_func float @_Z6asinpif(float %x) local_unnamed_addr #0 {
  %call = tail call float @_Z10_cl_asinpif(float %x)
  ret float %call
}


declare float @_Z8_cl_atanf(float) local_unnamed_addr #0

define spir_func float @_Z4atanf(float %x) local_unnamed_addr #0 {
  %call = tail call float @_Z8_cl_atanf(float %x)
  ret float %call
}


declare float @_Z9_cl_atanhf(float) local_unnamed_addr #0

define spir_func float @_Z5atanhf(float %x) local_unnamed_addr #0 {
  %call = tail call float @_Z9_cl_atanhf(float %x)
  ret float %call
}


declare float @_Z10_cl_atanpif(float) local_unnamed_addr #0

define spir_func float @_Z6atanpif(float %x) local_unnamed_addr #0 {
  %call = tail call float @_Z10_cl_atanpif(float %x)
  ret float %call
}


declare float @_Z8_cl_cbrtf(float) local_unnamed_addr #0

define spir_func float @_Z4cbrtf(float %x) local_unnamed_addr #0 {
  %call = tail call float @_Z8_cl_cbrtf(float %x)
  ret float %call
}


declare float @_Z8_cl_ceilf(float) local_unnamed_addr #0

define spir_func float @_Z4ceilf(float %x) local_unnamed_addr #0 {
  %call = tail call float @_Z8_cl_ceilf(float %x)
  ret float %call
}


declare float @_Z7_cl_cosf(float) local_unnamed_addr #0

define spir_func float @_Z3cosf(float %x) local_unnamed_addr #0 {
  %call = tail call float @_Z7_cl_cosf(float %x)
  ret float %call
}


declare float @_Z8_cl_coshf(float) local_unnamed_addr #0

define spir_func float @_Z4coshf(float %x) local_unnamed_addr #0 {
  %call = tail call float @_Z8_cl_coshf(float %x)
  ret float %call
}


declare float @_Z9_cl_cospif(float) local_unnamed_addr #0

define spir_func float @_Z5cospif(float %x) local_unnamed_addr #0 {
  %call = tail call float @_Z9_cl_cospif(float %x)
  ret float %call
}


declare float @_Z8_cl_erfcf(float) local_unnamed_addr #0

define spir_func float @_Z4erfcf(float %x) local_unnamed_addr #0 {
  %call = tail call float @_Z8_cl_erfcf(float %x)
  ret float %call
}


declare float @_Z7_cl_erff(float) local_unnamed_addr #0

define spir_func float @_Z3erff(float %x) local_unnamed_addr #0 {
  %call = tail call float @_Z7_cl_erff(float %x)
  ret float %call
}


declare float @_Z7_cl_expf(float) local_unnamed_addr #0

define spir_func float @_Z3expf(float %x) local_unnamed_addr #0 {
  %call = tail call float @_Z7_cl_expf(float %x)
  ret float %call
}


declare float @_Z8_cl_exp2f(float) local_unnamed_addr #0

define spir_func float @_Z4exp2f(float %x) local_unnamed_addr #0 {
  %call = tail call float @_Z8_cl_exp2f(float %x)
  ret float %call
}


declare float @_Z9_cl_exp10f(float) local_unnamed_addr #0

define spir_func float @_Z5exp10f(float %x) local_unnamed_addr #0 {
  %call = tail call float @_Z9_cl_exp10f(float %x)
  ret float %call
}


declare float @_Z9_cl_expm1f(float) local_unnamed_addr #0

define spir_func float @_Z5expm1f(float %x) local_unnamed_addr #0 {
  %call = tail call float @_Z9_cl_expm1f(float %x)
  ret float %call
}


declare float @_Z8_cl_fabsf(float) local_unnamed_addr #0

define spir_func float @_Z4fabsf(float %x) local_unnamed_addr #0 {
  %call = tail call float @_Z8_cl_fabsf(float %x)
  ret float %call
}


declare float @_Z9_cl_floorf(float) local_unnamed_addr #0

define spir_func float @_Z5floorf(float %x) local_unnamed_addr #0 {
  %call = tail call float @_Z9_cl_floorf(float %x)
  ret float %call
}


declare float @_Z10_cl_lgammaf(float) local_unnamed_addr #0

define spir_func float @_Z6lgammaf(float %x) local_unnamed_addr #0 {
  %call = tail call float @_Z10_cl_lgammaf(float %x)
  ret float %call
}


declare float @_Z7_cl_logf(float) local_unnamed_addr #0

define spir_func float @_Z3logf(float %x) local_unnamed_addr #0 {
  %call = tail call float @_Z7_cl_logf(float %x)
  ret float %call
}


declare float @_Z9_cl_log10f(float) local_unnamed_addr #0

define spir_func float @_Z5log10f(float %x) local_unnamed_addr #0 {
  %call = tail call float @_Z9_cl_log10f(float %x)
  ret float %call
}


declare float @_Z8_cl_log2f(float) local_unnamed_addr #0

define spir_func float @_Z4log2f(float %x) local_unnamed_addr #0 {
  %call = tail call float @_Z8_cl_log2f(float %x)
  ret float %call
}


declare float @_Z9_cl_log1pf(float) local_unnamed_addr #0

define spir_func float @_Z5log1pf(float %x) local_unnamed_addr #0 {
  %call = tail call float @_Z9_cl_log1pf(float %x)
  ret float %call
}


declare float @_Z8_cl_rintf(float) local_unnamed_addr #0

define spir_func float @_Z4rintf(float %x) local_unnamed_addr #0 {
  %call = tail call float @_Z8_cl_rintf(float %x)
  ret float %call
}


declare float @_Z9_cl_roundf(float) local_unnamed_addr #0

define spir_func float @_Z5roundf(float %x) local_unnamed_addr #0 {
  %call = tail call float @_Z9_cl_roundf(float %x)
  ret float %call
}


declare float @_Z9_cl_rsqrtf(float) local_unnamed_addr #0

define spir_func float @_Z5rsqrtf(float %x) local_unnamed_addr #0 {
  %call = tail call float @_Z9_cl_rsqrtf(float %x)
  ret float %call
}


declare float @_Z7_cl_sinf(float) local_unnamed_addr #0

define spir_func float @_Z3sinf(float %x) local_unnamed_addr #0 {
  %call = tail call float @_Z7_cl_sinf(float %x)
  ret float %call
}


declare float @_Z8_cl_sinhf(float) local_unnamed_addr #0

define spir_func float @_Z4sinhf(float %x) local_unnamed_addr #0 {
  %call = tail call float @_Z8_cl_sinhf(float %x)
  ret float %call
}


declare float @_Z9_cl_sinpif(float) local_unnamed_addr #0

define spir_func float @_Z5sinpif(float %x) local_unnamed_addr #0 {
  %call = tail call float @_Z9_cl_sinpif(float %x)
  ret float %call
}


declare float @_Z8_cl_sqrtf(float) local_unnamed_addr #0

define spir_func float @_Z4sqrtf(float %x) local_unnamed_addr #0 {
  %call = tail call float @_Z8_cl_sqrtf(float %x)
  ret float %call
}


declare float @_Z7_cl_tanf(float) local_unnamed_addr #0

define spir_func float @_Z3tanf(float %x) local_unnamed_addr #0 {
  %call = tail call float @_Z7_cl_tanf(float %x)
  ret float %call
}


declare float @_Z8_cl_tanhf(float) local_unnamed_addr #0

define spir_func float @_Z4tanhf(float %x) local_unnamed_addr #0 {
  %call = tail call float @_Z8_cl_tanhf(float %x)
  ret float %call
}


declare float @_Z9_cl_tanpif(float) local_unnamed_addr #0

define spir_func float @_Z5tanpif(float %x) local_unnamed_addr #0 {
  %call = tail call float @_Z9_cl_tanpif(float %x)
  ret float %call
}


declare float @_Z10_cl_tgammaf(float) local_unnamed_addr #0

define spir_func float @_Z6tgammaf(float %x) local_unnamed_addr #0 {
  %call = tail call float @_Z10_cl_tgammaf(float %x)
  ret float %call
}


declare float @_Z9_cl_truncf(float) local_unnamed_addr #0

define spir_func float @_Z5truncf(float %x) local_unnamed_addr #0 {
  %call = tail call float @_Z9_cl_truncf(float %x)
  ret float %call
}


declare float @_Z14_cl_native_cosf(float) local_unnamed_addr #0

define spir_func float @_Z10native_cosf(float %x) local_unnamed_addr #0 {
  %call = tail call float @_Z14_cl_native_cosf(float %x)
  ret float %call
}


declare float @_Z14_cl_native_expf(float) local_unnamed_addr #0

define spir_func float @_Z10native_expf(float %x) local_unnamed_addr #0 {
  %call = tail call float @_Z14_cl_native_expf(float %x)
  ret float %call
}


declare float @_Z15_cl_native_exp2f(float) local_unnamed_addr #0

define spir_func float @_Z11native_exp2f(float %x) local_unnamed_addr #0 {
  %call = tail call float @_Z15_cl_native_exp2f(float %x)
  ret float %call
}


declare float @_Z16_cl_native_exp10f(float) local_unnamed_addr #0

define spir_func float @_Z12native_exp10f(float %x) local_unnamed_addr #0 {
  %call = tail call float @_Z16_cl_native_exp10f(float %x)
  ret float %call
}


declare float @_Z14_cl_native_logf(float) local_unnamed_addr #0

define spir_func float @_Z10native_logf(float %x) local_unnamed_addr #0 {
  %call = tail call float @_Z14_cl_native_logf(float %x)
  ret float %call
}


declare float @_Z15_cl_native_log2f(float) local_unnamed_addr #0

define spir_func float @_Z11native_log2f(float %x) local_unnamed_addr #0 {
  %call = tail call float @_Z15_cl_native_log2f(float %x)
  ret float %call
}


declare float @_Z16_cl_native_log10f(float) local_unnamed_addr #0

define spir_func float @_Z12native_log10f(float %x) local_unnamed_addr #0 {
  %call = tail call float @_Z16_cl_native_log10f(float %x)
  ret float %call
}


declare float @_Z16_cl_native_recipf(float) local_unnamed_addr #0

define spir_func float @_Z12native_recipf(float %x) local_unnamed_addr #0 {
  %call = tail call float @_Z16_cl_native_recipf(float %x)
  ret float %call
}


declare float @_Z16_cl_native_rsqrtf(float) local_unnamed_addr #0

define spir_func float @_Z12native_rsqrtf(float %x) local_unnamed_addr #0 {
  %call = tail call float @_Z16_cl_native_rsqrtf(float %x)
  ret float %call
}


declare float @_Z14_cl_native_sinf(float) local_unnamed_addr #0

define spir_func float @_Z10native_sinf(float %x) local_unnamed_addr #0 {
  %call = tail call float @_Z14_cl_native_sinf(float %x)
  ret float %call
}


declare float @_Z15_cl_native_sqrtf(float) local_unnamed_addr #0

define spir_func float @_Z11native_sqrtf(float %x) local_unnamed_addr #0 {
  %call = tail call float @_Z15_cl_native_sqrtf(float %x)
  ret float %call
}


declare float @_Z14_cl_native_tanf(float) local_unnamed_addr #0

define spir_func float @_Z10native_tanf(float %x) local_unnamed_addr #0 {
  %call = tail call float @_Z14_cl_native_tanf(float %x)
  ret float %call
}


declare float @_Z11_cl_degreesf(float) local_unnamed_addr #0

define spir_func float @_Z7degreesf(float %x) local_unnamed_addr #0 {
  %call = tail call float @_Z11_cl_degreesf(float %x)
  ret float %call
}


declare float @_Z11_cl_radiansf(float) local_unnamed_addr #0

define spir_func float @_Z7radiansf(float %x) local_unnamed_addr #0 {
  %call = tail call float @_Z11_cl_radiansf(float %x)
  ret float %call
}


declare float @_Z8_cl_signf(float) local_unnamed_addr #0

define spir_func float @_Z4signf(float %x) local_unnamed_addr #0 {
  %call = tail call float @_Z8_cl_signf(float %x)
  ret float %call
}


declare float @_Z9_cl_atan2ff(float, float) local_unnamed_addr #0

define spir_func float @_Z5atan2ff(float %x, float %y) local_unnamed_addr #0 {
  %call = tail call float @_Z9_cl_atan2ff(float %x, float %y)
  ret float %call
}


declare float @_Z11_cl_atan2piff(float, float) local_unnamed_addr #0

define spir_func float @_Z7atan2piff(float %x, float %y) local_unnamed_addr #0 {
  %call = tail call float @_Z11_cl_atan2piff(float %x, float %y)
  ret float %call
}


declare float @_Z12_cl_copysignff(float, float) local_unnamed_addr #0

define spir_func float @_Z8copysignff(float %x, float %y) local_unnamed_addr #0 {
  %call = tail call float @_Z12_cl_copysignff(float %x, float %y)
  ret float %call
}


declare float @_Z8_cl_fdimff(float, float) local_unnamed_addr #0

define spir_func float @_Z4fdimff(float %x, float %y) local_unnamed_addr #0 {
  %call = tail call float @_Z8_cl_fdimff(float %x, float %y)
  ret float %call
}


declare float @_Z8_cl_fmaxff(float, float) local_unnamed_addr #0

define spir_func float @_Z4fmaxff(float %x, float %y) local_unnamed_addr #0 {
  %call = tail call float @_Z8_cl_fmaxff(float %x, float %y)
  ret float %call
}


declare float @_Z8_cl_fminff(float, float) local_unnamed_addr #0

define spir_func float @_Z4fminff(float %x, float %y) local_unnamed_addr #0 {
  %call = tail call float @_Z8_cl_fminff(float %x, float %y)
  ret float %call
}


declare float @_Z8_cl_fmodff(float, float) local_unnamed_addr #0

define spir_func float @_Z4fmodff(float %x, float %y) local_unnamed_addr #0 {
  %call = tail call float @_Z8_cl_fmodff(float %x, float %y)
  ret float %call
}


declare float @_Z9_cl_hypotff(float, float) local_unnamed_addr #0

define spir_func float @_Z5hypotff(float %x, float %y) local_unnamed_addr #0 {
  %call = tail call float @_Z9_cl_hypotff(float %x, float %y)
  ret float %call
}


declare float @_Z13_cl_nextafterff(float, float) local_unnamed_addr #0

define spir_func float @_Z9nextafterff(float %x, float %y) local_unnamed_addr #0 {
  %call = tail call float @_Z13_cl_nextafterff(float %x, float %y)
  ret float %call
}


declare float @_Z7_cl_powff(float, float) local_unnamed_addr #0

define spir_func float @_Z3powff(float %x, float %y) local_unnamed_addr #0 {
  %call = tail call float @_Z7_cl_powff(float %x, float %y)
  ret float %call
}


declare float @_Z8_cl_powrff(float, float) local_unnamed_addr #0

define spir_func float @_Z4powrff(float %x, float %y) local_unnamed_addr #0 {
  %call = tail call float @_Z8_cl_powrff(float %x, float %y)
  ret float %call
}


declare float @_Z10_cl_maxmagff(float, float) local_unnamed_addr #0

define spir_func float @_Z6maxmagff(float %x, float %y) local_unnamed_addr #0 {
  %call = tail call float @_Z10_cl_maxmagff(float %x, float %y)
  ret float %call
}


declare float @_Z10_cl_minmagff(float, float) local_unnamed_addr #0

define spir_func float @_Z6minmagff(float %x, float %y) local_unnamed_addr #0 {
  %call = tail call float @_Z10_cl_minmagff(float %x, float %y)
  ret float %call
}


declare float @_Z13_cl_remainderff(float, float) local_unnamed_addr #0

define spir_func float @_Z9remainderff(float %x, float %y) local_unnamed_addr #0 {
  %call = tail call float @_Z13_cl_remainderff(float %x, float %y)
  ret float %call
}


declare float @_Z17_cl_native_divideff(float, float) local_unnamed_addr #0

define spir_func float @_Z13native_divideff(float %x, float %y) local_unnamed_addr #0 {
  %call = tail call float @_Z17_cl_native_divideff(float %x, float %y)
  ret float %call
}


declare float @_Z15_cl_native_powrff(float, float) local_unnamed_addr #0

define spir_func float @_Z11native_powrff(float %x, float %y) local_unnamed_addr #0 {
  %call = tail call float @_Z15_cl_native_powrff(float %x, float %y)
  ret float %call
}


declare float @_Z7_cl_maxff(float, float) local_unnamed_addr #0

define spir_func float @_Z3maxff(float %x, float %y) local_unnamed_addr #0 {
  %call = tail call float @_Z7_cl_maxff(float %x, float %y)
  ret float %call
}


declare float @_Z7_cl_minff(float, float) local_unnamed_addr #0

define spir_func float @_Z3minff(float %x, float %y) local_unnamed_addr #0 {
  %call = tail call float @_Z7_cl_minff(float %x, float %y)
  ret float %call
}


declare float @_Z8_cl_stepff(float, float) local_unnamed_addr #0

define spir_func float @_Z4stepff(float %x, float %y) local_unnamed_addr #0 {
  %call = tail call float @_Z8_cl_stepff(float %x, float %y)
  ret float %call
}


declare float @_Z9_cl_fractfPU8CLglobalf(float, float *) local_unnamed_addr #0

define spir_func float @_Z5fractfPU3AS1f(float %x, float addrspace(1)* %y) local_unnamed_addr #0 {
  %1 = addrspacecast float addrspace(1)* %y to float *
  %call = tail call float @_Z9_cl_fractfPU8CLglobalf(float %x, float * %1)
  ret float %call
}


declare float @_Z9_cl_fractfPU7CLlocalf(float, float *) local_unnamed_addr #0

define spir_func float @_Z5fractfPU3AS3f(float %x, float addrspace(3)* %y) local_unnamed_addr #0 {
  %1 = addrspacecast float addrspace(3)* %y to float *
  %call = tail call float @_Z9_cl_fractfPU7CLlocalf(float %x, float * %1)
  ret float %call
}


declare float @_Z9_cl_fractfPU9CLprivatef(float, float *) local_unnamed_addr #0

define spir_func float @_Z5fractfPf(float %x, float * %y) local_unnamed_addr #0 {
  %call = tail call float @_Z9_cl_fractfPU9CLprivatef(float %x, float * %y)
  ret float %call
}


declare float @_Z10_cl_sincosfPU8CLglobalf(float, float *) local_unnamed_addr #0

define spir_func float @_Z6sincosfPU3AS1f(float %x, float addrspace(1)* %y) local_unnamed_addr #0 {
  %1 = addrspacecast float addrspace(1)* %y to float *
  %call = tail call float @_Z10_cl_sincosfPU8CLglobalf(float %x, float * %1)
  ret float %call
}


declare float @_Z10_cl_sincosfPU7CLlocalf(float, float *) local_unnamed_addr #0

define spir_func float @_Z6sincosfPU3AS3f(float %x, float addrspace(3)* %y) local_unnamed_addr #0 {
  %1 = addrspacecast float addrspace(3)* %y to float *
  %call = tail call float @_Z10_cl_sincosfPU7CLlocalf(float %x, float * %1)
  ret float %call
}


declare float @_Z10_cl_sincosfPU9CLprivatef(float, float *) local_unnamed_addr #0

define spir_func float @_Z6sincosfPf(float %x, float * %y) local_unnamed_addr #0 {
  %call = tail call float @_Z10_cl_sincosfPU9CLprivatef(float %x, float * %y)
  ret float %call
}


declare float @_Z8_cl_modffPU8CLglobalf(float, float *) local_unnamed_addr #0

define spir_func float @_Z4modffPU3AS1f(float %x, float addrspace(1)* %y) local_unnamed_addr #0 {
  %1 = addrspacecast float addrspace(1)* %y to float *
  %call = tail call float @_Z8_cl_modffPU8CLglobalf(float %x, float * %1)
  ret float %call
}


declare float @_Z8_cl_modffPU7CLlocalf(float, float *) local_unnamed_addr #0

define spir_func float @_Z4modffPU3AS3f(float %x, float addrspace(3)* %y) local_unnamed_addr #0 {
  %1 = addrspacecast float addrspace(3)* %y to float *
  %call = tail call float @_Z8_cl_modffPU7CLlocalf(float %x, float * %1)
  ret float %call
}


declare float @_Z8_cl_modffPU9CLprivatef(float, float *) local_unnamed_addr #0

define spir_func float @_Z4modffPf(float %x, float * %y) local_unnamed_addr #0 {
  %call = tail call float @_Z8_cl_modffPU9CLprivatef(float %x, float * %y)
  ret float %call
}


declare float @_Z7_cl_fmafff(float, float, float) local_unnamed_addr #0

define spir_func float @_Z3fmafff(float %x, float %y, float %z) local_unnamed_addr #0 {
  %call = tail call float @_Z7_cl_fmafff(float %x, float %y, float %z)
  ret float %call
}


declare float @_Z7_cl_madfff(float, float, float) local_unnamed_addr #0

define spir_func float @_Z3madfff(float %x, float %y, float %z) local_unnamed_addr #0 {
  %call = tail call float @_Z7_cl_madfff(float %x, float %y, float %z)
  ret float %call
}


declare float @_Z9_cl_clampfff(float, float, float) local_unnamed_addr #0

define spir_func float @_Z5clampfff(float %x, float %y, float %z) local_unnamed_addr #0 {
  %call = tail call float @_Z9_cl_clampfff(float %x, float %y, float %z)
  ret float %call
}


declare float @_Z7_cl_mixfff(float, float, float) local_unnamed_addr #0

define spir_func float @_Z3mixfff(float %x, float %y, float %z) local_unnamed_addr #0 {
  %call = tail call float @_Z7_cl_mixfff(float %x, float %y, float %z)
  ret float %call
}


declare float @_Z14_cl_smoothstepfff(float, float, float) local_unnamed_addr #0

define spir_func float @_Z10smoothstepfff(float %x, float %y, float %z) local_unnamed_addr #0 {
  %call = tail call float @_Z14_cl_smoothstepfff(float %x, float %y, float %z)
  ret float %call
}


declare i32 @_Z9_cl_ilogbf(float) local_unnamed_addr #0

define spir_func i32 @_Z5ilogbf(float %x) local_unnamed_addr #0 {
  %call = tail call i32 @_Z9_cl_ilogbf(float %x)
  ret i32 %call
}


declare float @_Z9_cl_ldexpfi(float, i32) local_unnamed_addr #0

define spir_func float @_Z5ldexpfi(float %x, i32 %y) local_unnamed_addr #0 {
  %call = tail call float @_Z9_cl_ldexpfi(float %x, i32 %y)
  ret float %call
}


declare float @_Z8_cl_pownfi(float, i32) local_unnamed_addr #0

define spir_func float @_Z4pownfi(float %x, i32 %y) local_unnamed_addr #0 {
  %call = tail call float @_Z8_cl_pownfi(float %x, i32 %y)
  ret float %call
}


declare float @_Z9_cl_rootnfi(float, i32) local_unnamed_addr #0

define spir_func float @_Z5rootnfi(float %x, i32 %y) local_unnamed_addr #0 {
  %call = tail call float @_Z9_cl_rootnfi(float %x, i32 %y)
  ret float %call
}


declare float @_Z10_cl_remquoffPU8CLglobali(float, float, i32 *) local_unnamed_addr #0

define spir_func float @_Z6remquoffPU3AS1i(float %x, float %y, i32 addrspace(1)* %z) local_unnamed_addr #0 {
  %1 = addrspacecast i32 addrspace(1)* %z to i32 *
  %call = tail call float @_Z10_cl_remquoffPU8CLglobali(float %x, float %y, i32 * %1)
  ret float %call
}


declare float @_Z10_cl_remquoffPU7CLlocali(float, float, i32 *) local_unnamed_addr #0

define spir_func float @_Z6remquoffPU3AS3i(float %x, float %y, i32 addrspace(3)* %z) local_unnamed_addr #0 {
  %1 = addrspacecast i32 addrspace(3)* %z to i32 *
  %call = tail call float @_Z10_cl_remquoffPU7CLlocali(float %x, float %y, i32 * %1)
  ret float %call
}


declare float @_Z10_cl_remquoffPU9CLprivatei(float, float, i32 *) local_unnamed_addr #0

define spir_func float @_Z6remquoffPi(float %x, float %y, i32 * %z) local_unnamed_addr #0 {
  %call = tail call float @_Z10_cl_remquoffPU9CLprivatei(float %x, float %y, i32 * %z)
  ret float %call
}


declare float @_Z7_cl_nanj(i32) local_unnamed_addr #0

define spir_func float @_Z3nanj(i32 %x) local_unnamed_addr #0 {
  %call = tail call float @_Z7_cl_nanj(i32 %x)
  ret float %call
}


declare float @_Z12_cl_lgamma_rfPU8CLglobali(float, i32 *) local_unnamed_addr #0

define spir_func float @_Z8lgamma_rfPU3AS1i(float %x, i32 addrspace(1)* %y) local_unnamed_addr #0 {
  %1 = addrspacecast i32 addrspace(1)* %y to i32 *
  %call = tail call float @_Z12_cl_lgamma_rfPU8CLglobali(float %x, i32 * %1)
  ret float %call
}


declare float @_Z12_cl_lgamma_rfPU7CLlocali(float, i32 *) local_unnamed_addr #0

define spir_func float @_Z8lgamma_rfPU3AS3i(float %x, i32 addrspace(3)* %y) local_unnamed_addr #0 {
  %1 = addrspacecast i32 addrspace(3)* %y to i32 *
  %call = tail call float @_Z12_cl_lgamma_rfPU7CLlocali(float %x, i32 * %1)
  ret float %call
}


declare float @_Z12_cl_lgamma_rfPU9CLprivatei(float, i32 *) local_unnamed_addr #0

define spir_func float @_Z8lgamma_rfPi(float %x, i32 * %y) local_unnamed_addr #0 {
  %call = tail call float @_Z12_cl_lgamma_rfPU9CLprivatei(float %x, i32 * %y)
  ret float %call
}


declare float @_Z9_cl_frexpfPU8CLglobali(float, i32 *) local_unnamed_addr #0

define spir_func float @_Z5frexpfPU3AS1i(float %x, i32 addrspace(1)* %y) local_unnamed_addr #0 {
  %1 = addrspacecast i32 addrspace(1)* %y to i32 *
  %call = tail call float @_Z9_cl_frexpfPU8CLglobali(float %x, i32 * %1)
  ret float %call
}


declare float @_Z9_cl_frexpfPU7CLlocali(float, i32 *) local_unnamed_addr #0

define spir_func float @_Z5frexpfPU3AS3i(float %x, i32 addrspace(3)* %y) local_unnamed_addr #0 {
  %1 = addrspacecast i32 addrspace(3)* %y to i32 *
  %call = tail call float @_Z9_cl_frexpfPU7CLlocali(float %x, i32 * %1)
  ret float %call
}


declare float @_Z9_cl_frexpfPU9CLprivatei(float, i32 *) local_unnamed_addr #0

define spir_func float @_Z5frexpfPi(float %x, i32 * %y) local_unnamed_addr #0 {
  %call = tail call float @_Z9_cl_frexpfPU9CLprivatei(float %x, i32 * %y)
  ret float %call
}


declare double @_Z8_cl_acosd(double) local_unnamed_addr #0

define spir_func double @_Z4acosd(double %x) local_unnamed_addr #0 {
  %call = tail call double @_Z8_cl_acosd(double %x)
  ret double %call
}


declare double @_Z9_cl_acoshd(double) local_unnamed_addr #0

define spir_func double @_Z5acoshd(double %x) local_unnamed_addr #0 {
  %call = tail call double @_Z9_cl_acoshd(double %x)
  ret double %call
}


declare double @_Z10_cl_acospid(double) local_unnamed_addr #0

define spir_func double @_Z6acospid(double %x) local_unnamed_addr #0 {
  %call = tail call double @_Z10_cl_acospid(double %x)
  ret double %call
}


declare double @_Z8_cl_asind(double) local_unnamed_addr #0

define spir_func double @_Z4asind(double %x) local_unnamed_addr #0 {
  %call = tail call double @_Z8_cl_asind(double %x)
  ret double %call
}


declare double @_Z9_cl_asinhd(double) local_unnamed_addr #0

define spir_func double @_Z5asinhd(double %x) local_unnamed_addr #0 {
  %call = tail call double @_Z9_cl_asinhd(double %x)
  ret double %call
}


declare double @_Z10_cl_asinpid(double) local_unnamed_addr #0

define spir_func double @_Z6asinpid(double %x) local_unnamed_addr #0 {
  %call = tail call double @_Z10_cl_asinpid(double %x)
  ret double %call
}


declare double @_Z8_cl_atand(double) local_unnamed_addr #0

define spir_func double @_Z4atand(double %x) local_unnamed_addr #0 {
  %call = tail call double @_Z8_cl_atand(double %x)
  ret double %call
}


declare double @_Z9_cl_atanhd(double) local_unnamed_addr #0

define spir_func double @_Z5atanhd(double %x) local_unnamed_addr #0 {
  %call = tail call double @_Z9_cl_atanhd(double %x)
  ret double %call
}


declare double @_Z10_cl_atanpid(double) local_unnamed_addr #0

define spir_func double @_Z6atanpid(double %x) local_unnamed_addr #0 {
  %call = tail call double @_Z10_cl_atanpid(double %x)
  ret double %call
}


declare double @_Z8_cl_cbrtd(double) local_unnamed_addr #0

define spir_func double @_Z4cbrtd(double %x) local_unnamed_addr #0 {
  %call = tail call double @_Z8_cl_cbrtd(double %x)
  ret double %call
}


declare double @_Z8_cl_ceild(double) local_unnamed_addr #0

define spir_func double @_Z4ceild(double %x) local_unnamed_addr #0 {
  %call = tail call double @_Z8_cl_ceild(double %x)
  ret double %call
}


declare double @_Z7_cl_cosd(double) local_unnamed_addr #0

define spir_func double @_Z3cosd(double %x) local_unnamed_addr #0 {
  %call = tail call double @_Z7_cl_cosd(double %x)
  ret double %call
}


declare double @_Z8_cl_coshd(double) local_unnamed_addr #0

define spir_func double @_Z4coshd(double %x) local_unnamed_addr #0 {
  %call = tail call double @_Z8_cl_coshd(double %x)
  ret double %call
}


declare double @_Z9_cl_cospid(double) local_unnamed_addr #0

define spir_func double @_Z5cospid(double %x) local_unnamed_addr #0 {
  %call = tail call double @_Z9_cl_cospid(double %x)
  ret double %call
}


declare double @_Z8_cl_erfcd(double) local_unnamed_addr #0

define spir_func double @_Z4erfcd(double %x) local_unnamed_addr #0 {
  %call = tail call double @_Z8_cl_erfcd(double %x)
  ret double %call
}


declare double @_Z7_cl_erfd(double) local_unnamed_addr #0

define spir_func double @_Z3erfd(double %x) local_unnamed_addr #0 {
  %call = tail call double @_Z7_cl_erfd(double %x)
  ret double %call
}


declare double @_Z7_cl_expd(double) local_unnamed_addr #0

define spir_func double @_Z3expd(double %x) local_unnamed_addr #0 {
  %call = tail call double @_Z7_cl_expd(double %x)
  ret double %call
}


declare double @_Z8_cl_exp2d(double) local_unnamed_addr #0

define spir_func double @_Z4exp2d(double %x) local_unnamed_addr #0 {
  %call = tail call double @_Z8_cl_exp2d(double %x)
  ret double %call
}


declare double @_Z9_cl_exp10d(double) local_unnamed_addr #0

define spir_func double @_Z5exp10d(double %x) local_unnamed_addr #0 {
  %call = tail call double @_Z9_cl_exp10d(double %x)
  ret double %call
}


declare double @_Z9_cl_expm1d(double) local_unnamed_addr #0

define spir_func double @_Z5expm1d(double %x) local_unnamed_addr #0 {
  %call = tail call double @_Z9_cl_expm1d(double %x)
  ret double %call
}


declare double @_Z8_cl_fabsd(double) local_unnamed_addr #0

define spir_func double @_Z4fabsd(double %x) local_unnamed_addr #0 {
  %call = tail call double @_Z8_cl_fabsd(double %x)
  ret double %call
}


declare double @_Z9_cl_floord(double) local_unnamed_addr #0

define spir_func double @_Z5floord(double %x) local_unnamed_addr #0 {
  %call = tail call double @_Z9_cl_floord(double %x)
  ret double %call
}


declare double @_Z10_cl_lgammad(double) local_unnamed_addr #0

define spir_func double @_Z6lgammad(double %x) local_unnamed_addr #0 {
  %call = tail call double @_Z10_cl_lgammad(double %x)
  ret double %call
}


declare double @_Z7_cl_logd(double) local_unnamed_addr #0

define spir_func double @_Z3logd(double %x) local_unnamed_addr #0 {
  %call = tail call double @_Z7_cl_logd(double %x)
  ret double %call
}


declare double @_Z9_cl_log10d(double) local_unnamed_addr #0

define spir_func double @_Z5log10d(double %x) local_unnamed_addr #0 {
  %call = tail call double @_Z9_cl_log10d(double %x)
  ret double %call
}


declare double @_Z8_cl_log2d(double) local_unnamed_addr #0

define spir_func double @_Z4log2d(double %x) local_unnamed_addr #0 {
  %call = tail call double @_Z8_cl_log2d(double %x)
  ret double %call
}


declare double @_Z9_cl_log1pd(double) local_unnamed_addr #0

define spir_func double @_Z5log1pd(double %x) local_unnamed_addr #0 {
  %call = tail call double @_Z9_cl_log1pd(double %x)
  ret double %call
}


declare double @_Z8_cl_rintd(double) local_unnamed_addr #0

define spir_func double @_Z4rintd(double %x) local_unnamed_addr #0 {
  %call = tail call double @_Z8_cl_rintd(double %x)
  ret double %call
}


declare double @_Z9_cl_roundd(double) local_unnamed_addr #0

define spir_func double @_Z5roundd(double %x) local_unnamed_addr #0 {
  %call = tail call double @_Z9_cl_roundd(double %x)
  ret double %call
}


declare double @_Z9_cl_rsqrtd(double) local_unnamed_addr #0

define spir_func double @_Z5rsqrtd(double %x) local_unnamed_addr #0 {
  %call = tail call double @_Z9_cl_rsqrtd(double %x)
  ret double %call
}


declare double @_Z7_cl_sind(double) local_unnamed_addr #0

define spir_func double @_Z3sind(double %x) local_unnamed_addr #0 {
  %call = tail call double @_Z7_cl_sind(double %x)
  ret double %call
}


declare double @_Z8_cl_sinhd(double) local_unnamed_addr #0

define spir_func double @_Z4sinhd(double %x) local_unnamed_addr #0 {
  %call = tail call double @_Z8_cl_sinhd(double %x)
  ret double %call
}


declare double @_Z9_cl_sinpid(double) local_unnamed_addr #0

define spir_func double @_Z5sinpid(double %x) local_unnamed_addr #0 {
  %call = tail call double @_Z9_cl_sinpid(double %x)
  ret double %call
}


declare double @_Z8_cl_sqrtd(double) local_unnamed_addr #0

define spir_func double @_Z4sqrtd(double %x) local_unnamed_addr #0 {
  %call = tail call double @_Z8_cl_sqrtd(double %x)
  ret double %call
}


declare double @_Z7_cl_tand(double) local_unnamed_addr #0

define spir_func double @_Z3tand(double %x) local_unnamed_addr #0 {
  %call = tail call double @_Z7_cl_tand(double %x)
  ret double %call
}


declare double @_Z8_cl_tanhd(double) local_unnamed_addr #0

define spir_func double @_Z4tanhd(double %x) local_unnamed_addr #0 {
  %call = tail call double @_Z8_cl_tanhd(double %x)
  ret double %call
}


declare double @_Z9_cl_tanpid(double) local_unnamed_addr #0

define spir_func double @_Z5tanpid(double %x) local_unnamed_addr #0 {
  %call = tail call double @_Z9_cl_tanpid(double %x)
  ret double %call
}


declare double @_Z10_cl_tgammad(double) local_unnamed_addr #0

define spir_func double @_Z6tgammad(double %x) local_unnamed_addr #0 {
  %call = tail call double @_Z10_cl_tgammad(double %x)
  ret double %call
}


declare double @_Z9_cl_truncd(double) local_unnamed_addr #0

define spir_func double @_Z5truncd(double %x) local_unnamed_addr #0 {
  %call = tail call double @_Z9_cl_truncd(double %x)
  ret double %call
}


declare double @_Z14_cl_native_cosd(double) local_unnamed_addr #0

define spir_func double @_Z10native_cosd(double %x) local_unnamed_addr #0 {
  %call = tail call double @_Z14_cl_native_cosd(double %x)
  ret double %call
}


declare double @_Z14_cl_native_expd(double) local_unnamed_addr #0

define spir_func double @_Z10native_expd(double %x) local_unnamed_addr #0 {
  %call = tail call double @_Z14_cl_native_expd(double %x)
  ret double %call
}


declare double @_Z15_cl_native_exp2d(double) local_unnamed_addr #0

define spir_func double @_Z11native_exp2d(double %x) local_unnamed_addr #0 {
  %call = tail call double @_Z15_cl_native_exp2d(double %x)
  ret double %call
}


declare double @_Z16_cl_native_exp10d(double) local_unnamed_addr #0

define spir_func double @_Z12native_exp10d(double %x) local_unnamed_addr #0 {
  %call = tail call double @_Z16_cl_native_exp10d(double %x)
  ret double %call
}


declare double @_Z14_cl_native_logd(double) local_unnamed_addr #0

define spir_func double @_Z10native_logd(double %x) local_unnamed_addr #0 {
  %call = tail call double @_Z14_cl_native_logd(double %x)
  ret double %call
}


declare double @_Z15_cl_native_log2d(double) local_unnamed_addr #0

define spir_func double @_Z11native_log2d(double %x) local_unnamed_addr #0 {
  %call = tail call double @_Z15_cl_native_log2d(double %x)
  ret double %call
}


declare double @_Z16_cl_native_log10d(double) local_unnamed_addr #0

define spir_func double @_Z12native_log10d(double %x) local_unnamed_addr #0 {
  %call = tail call double @_Z16_cl_native_log10d(double %x)
  ret double %call
}


declare double @_Z16_cl_native_recipd(double) local_unnamed_addr #0

define spir_func double @_Z12native_recipd(double %x) local_unnamed_addr #0 {
  %call = tail call double @_Z16_cl_native_recipd(double %x)
  ret double %call
}


declare double @_Z16_cl_native_rsqrtd(double) local_unnamed_addr #0

define spir_func double @_Z12native_rsqrtd(double %x) local_unnamed_addr #0 {
  %call = tail call double @_Z16_cl_native_rsqrtd(double %x)
  ret double %call
}


declare double @_Z14_cl_native_sind(double) local_unnamed_addr #0

define spir_func double @_Z10native_sind(double %x) local_unnamed_addr #0 {
  %call = tail call double @_Z14_cl_native_sind(double %x)
  ret double %call
}


declare double @_Z15_cl_native_sqrtd(double) local_unnamed_addr #0

define spir_func double @_Z11native_sqrtd(double %x) local_unnamed_addr #0 {
  %call = tail call double @_Z15_cl_native_sqrtd(double %x)
  ret double %call
}


declare double @_Z14_cl_native_tand(double) local_unnamed_addr #0

define spir_func double @_Z10native_tand(double %x) local_unnamed_addr #0 {
  %call = tail call double @_Z14_cl_native_tand(double %x)
  ret double %call
}


declare double @_Z11_cl_degreesd(double) local_unnamed_addr #0

define spir_func double @_Z7degreesd(double %x) local_unnamed_addr #0 {
  %call = tail call double @_Z11_cl_degreesd(double %x)
  ret double %call
}


declare double @_Z11_cl_radiansd(double) local_unnamed_addr #0

define spir_func double @_Z7radiansd(double %x) local_unnamed_addr #0 {
  %call = tail call double @_Z11_cl_radiansd(double %x)
  ret double %call
}


declare double @_Z8_cl_signd(double) local_unnamed_addr #0

define spir_func double @_Z4signd(double %x) local_unnamed_addr #0 {
  %call = tail call double @_Z8_cl_signd(double %x)
  ret double %call
}


declare double @_Z9_cl_atan2dd(double, double) local_unnamed_addr #0

define spir_func double @_Z5atan2dd(double %x, double %y) local_unnamed_addr #0 {
  %call = tail call double @_Z9_cl_atan2dd(double %x, double %y)
  ret double %call
}


declare double @_Z11_cl_atan2pidd(double, double) local_unnamed_addr #0

define spir_func double @_Z7atan2pidd(double %x, double %y) local_unnamed_addr #0 {
  %call = tail call double @_Z11_cl_atan2pidd(double %x, double %y)
  ret double %call
}


declare double @_Z12_cl_copysigndd(double, double) local_unnamed_addr #0

define spir_func double @_Z8copysigndd(double %x, double %y) local_unnamed_addr #0 {
  %call = tail call double @_Z12_cl_copysigndd(double %x, double %y)
  ret double %call
}


declare double @_Z8_cl_fdimdd(double, double) local_unnamed_addr #0

define spir_func double @_Z4fdimdd(double %x, double %y) local_unnamed_addr #0 {
  %call = tail call double @_Z8_cl_fdimdd(double %x, double %y)
  ret double %call
}


declare double @_Z8_cl_fmaxdd(double, double) local_unnamed_addr #0

define spir_func double @_Z4fmaxdd(double %x, double %y) local_unnamed_addr #0 {
  %call = tail call double @_Z8_cl_fmaxdd(double %x, double %y)
  ret double %call
}


declare double @_Z8_cl_fmindd(double, double) local_unnamed_addr #0

define spir_func double @_Z4fmindd(double %x, double %y) local_unnamed_addr #0 {
  %call = tail call double @_Z8_cl_fmindd(double %x, double %y)
  ret double %call
}


declare double @_Z8_cl_fmoddd(double, double) local_unnamed_addr #0

define spir_func double @_Z4fmoddd(double %x, double %y) local_unnamed_addr #0 {
  %call = tail call double @_Z8_cl_fmoddd(double %x, double %y)
  ret double %call
}


declare double @_Z9_cl_hypotdd(double, double) local_unnamed_addr #0

define spir_func double @_Z5hypotdd(double %x, double %y) local_unnamed_addr #0 {
  %call = tail call double @_Z9_cl_hypotdd(double %x, double %y)
  ret double %call
}


declare double @_Z13_cl_nextafterdd(double, double) local_unnamed_addr #0

define spir_func double @_Z9nextafterdd(double %x, double %y) local_unnamed_addr #0 {
  %call = tail call double @_Z13_cl_nextafterdd(double %x, double %y)
  ret double %call
}


declare double @_Z7_cl_powdd(double, double) local_unnamed_addr #0

define spir_func double @_Z3powdd(double %x, double %y) local_unnamed_addr #0 {
  %call = tail call double @_Z7_cl_powdd(double %x, double %y)
  ret double %call
}


declare double @_Z8_cl_powrdd(double, double) local_unnamed_addr #0

define spir_func double @_Z4powrdd(double %x, double %y) local_unnamed_addr #0 {
  %call = tail call double @_Z8_cl_powrdd(double %x, double %y)
  ret double %call
}


declare double @_Z10_cl_maxmagdd(double, double) local_unnamed_addr #0

define spir_func double @_Z6maxmagdd(double %x, double %y) local_unnamed_addr #0 {
  %call = tail call double @_Z10_cl_maxmagdd(double %x, double %y)
  ret double %call
}


declare double @_Z10_cl_minmagdd(double, double) local_unnamed_addr #0

define spir_func double @_Z6minmagdd(double %x, double %y) local_unnamed_addr #0 {
  %call = tail call double @_Z10_cl_minmagdd(double %x, double %y)
  ret double %call
}


declare double @_Z13_cl_remainderdd(double, double) local_unnamed_addr #0

define spir_func double @_Z9remainderdd(double %x, double %y) local_unnamed_addr #0 {
  %call = tail call double @_Z13_cl_remainderdd(double %x, double %y)
  ret double %call
}


declare double @_Z17_cl_native_dividedd(double, double) local_unnamed_addr #0

define spir_func double @_Z13native_dividedd(double %x, double %y) local_unnamed_addr #0 {
  %call = tail call double @_Z17_cl_native_dividedd(double %x, double %y)
  ret double %call
}


declare double @_Z15_cl_native_powrdd(double, double) local_unnamed_addr #0

define spir_func double @_Z11native_powrdd(double %x, double %y) local_unnamed_addr #0 {
  %call = tail call double @_Z15_cl_native_powrdd(double %x, double %y)
  ret double %call
}


declare double @_Z7_cl_maxdd(double, double) local_unnamed_addr #0

define spir_func double @_Z3maxdd(double %x, double %y) local_unnamed_addr #0 {
  %call = tail call double @_Z7_cl_maxdd(double %x, double %y)
  ret double %call
}


declare double @_Z7_cl_mindd(double, double) local_unnamed_addr #0

define spir_func double @_Z3mindd(double %x, double %y) local_unnamed_addr #0 {
  %call = tail call double @_Z7_cl_mindd(double %x, double %y)
  ret double %call
}


declare double @_Z8_cl_stepdd(double, double) local_unnamed_addr #0

define spir_func double @_Z4stepdd(double %x, double %y) local_unnamed_addr #0 {
  %call = tail call double @_Z8_cl_stepdd(double %x, double %y)
  ret double %call
}


declare double @_Z9_cl_fractdPU8CLglobald(double, double *) local_unnamed_addr #0

define spir_func double @_Z5fractdPU3AS1d(double %x, double addrspace(1)* %y) local_unnamed_addr #0 {
  %1 = addrspacecast double addrspace(1)* %y to double *
  %call = tail call double @_Z9_cl_fractdPU8CLglobald(double %x, double * %1)
  ret double %call
}


declare double @_Z9_cl_fractdPU7CLlocald(double, double *) local_unnamed_addr #0

define spir_func double @_Z5fractdPU3AS3d(double %x, double addrspace(3)* %y) local_unnamed_addr #0 {
  %1 = addrspacecast double addrspace(3)* %y to double *
  %call = tail call double @_Z9_cl_fractdPU7CLlocald(double %x, double * %1)
  ret double %call
}


declare double @_Z9_cl_fractdPU9CLprivated(double, double *) local_unnamed_addr #0

define spir_func double @_Z5fractdPd(double %x, double * %y) local_unnamed_addr #0 {
  %call = tail call double @_Z9_cl_fractdPU9CLprivated(double %x, double * %y)
  ret double %call
}


declare double @_Z10_cl_sincosdPU8CLglobald(double, double *) local_unnamed_addr #0

define spir_func double @_Z6sincosdPU3AS1d(double %x, double addrspace(1)* %y) local_unnamed_addr #0 {
  %1 = addrspacecast double addrspace(1)* %y to double *
  %call = tail call double @_Z10_cl_sincosdPU8CLglobald(double %x, double * %1)
  ret double %call
}


declare double @_Z10_cl_sincosdPU7CLlocald(double, double *) local_unnamed_addr #0

define spir_func double @_Z6sincosdPU3AS3d(double %x, double addrspace(3)* %y) local_unnamed_addr #0 {
  %1 = addrspacecast double addrspace(3)* %y to double *
  %call = tail call double @_Z10_cl_sincosdPU7CLlocald(double %x, double * %1)
  ret double %call
}


declare double @_Z10_cl_sincosdPU9CLprivated(double, double *) local_unnamed_addr #0

define spir_func double @_Z6sincosdPd(double %x, double * %y) local_unnamed_addr #0 {
  %call = tail call double @_Z10_cl_sincosdPU9CLprivated(double %x, double * %y)
  ret double %call
}


declare double @_Z8_cl_modfdPU8CLglobald(double, double *) local_unnamed_addr #0

define spir_func double @_Z4modfdPU3AS1d(double %x, double addrspace(1)* %y) local_unnamed_addr #0 {
  %1 = addrspacecast double addrspace(1)* %y to double *
  %call = tail call double @_Z8_cl_modfdPU8CLglobald(double %x, double * %1)
  ret double %call
}


declare double @_Z8_cl_modfdPU7CLlocald(double, double *) local_unnamed_addr #0

define spir_func double @_Z4modfdPU3AS3d(double %x, double addrspace(3)* %y) local_unnamed_addr #0 {
  %1 = addrspacecast double addrspace(3)* %y to double *
  %call = tail call double @_Z8_cl_modfdPU7CLlocald(double %x, double * %1)
  ret double %call
}


declare double @_Z8_cl_modfdPU9CLprivated(double, double *) local_unnamed_addr #0

define spir_func double @_Z4modfdPd(double %x, double * %y) local_unnamed_addr #0 {
  %call = tail call double @_Z8_cl_modfdPU9CLprivated(double %x, double * %y)
  ret double %call
}


declare double @_Z7_cl_fmaddd(double, double, double) local_unnamed_addr #0

define spir_func double @_Z3fmaddd(double %x, double %y, double %z) local_unnamed_addr #0 {
  %call = tail call double @_Z7_cl_fmaddd(double %x, double %y, double %z)
  ret double %call
}


declare double @_Z7_cl_madddd(double, double, double) local_unnamed_addr #0

define spir_func double @_Z3madddd(double %x, double %y, double %z) local_unnamed_addr #0 {
  %call = tail call double @_Z7_cl_madddd(double %x, double %y, double %z)
  ret double %call
}


declare double @_Z9_cl_clampddd(double, double, double) local_unnamed_addr #0

define spir_func double @_Z5clampddd(double %x, double %y, double %z) local_unnamed_addr #0 {
  %call = tail call double @_Z9_cl_clampddd(double %x, double %y, double %z)
  ret double %call
}


declare double @_Z7_cl_mixddd(double, double, double) local_unnamed_addr #0

define spir_func double @_Z3mixddd(double %x, double %y, double %z) local_unnamed_addr #0 {
  %call = tail call double @_Z7_cl_mixddd(double %x, double %y, double %z)
  ret double %call
}


declare double @_Z14_cl_smoothstepddd(double, double, double) local_unnamed_addr #0

define spir_func double @_Z10smoothstepddd(double %x, double %y, double %z) local_unnamed_addr #0 {
  %call = tail call double @_Z14_cl_smoothstepddd(double %x, double %y, double %z)
  ret double %call
}


declare i32 @_Z9_cl_ilogbd(double) local_unnamed_addr #0

define spir_func i32 @_Z5ilogbd(double %x) local_unnamed_addr #0 {
  %call = tail call i32 @_Z9_cl_ilogbd(double %x)
  ret i32 %call
}


declare double @_Z9_cl_ldexpdi(double, i32) local_unnamed_addr #0

define spir_func double @_Z5ldexpdi(double %x, i32 %y) local_unnamed_addr #0 {
  %call = tail call double @_Z9_cl_ldexpdi(double %x, i32 %y)
  ret double %call
}


declare double @_Z8_cl_powndi(double, i32) local_unnamed_addr #0

define spir_func double @_Z4powndi(double %x, i32 %y) local_unnamed_addr #0 {
  %call = tail call double @_Z8_cl_powndi(double %x, i32 %y)
  ret double %call
}


declare double @_Z9_cl_rootndi(double, i32) local_unnamed_addr #0

define spir_func double @_Z5rootndi(double %x, i32 %y) local_unnamed_addr #0 {
  %call = tail call double @_Z9_cl_rootndi(double %x, i32 %y)
  ret double %call
}


declare double @_Z10_cl_remquoddPU8CLglobali(double, double, i32 *) local_unnamed_addr #0

define spir_func double @_Z6remquoddPU3AS1i(double %x, double %y, i32 addrspace(1)* %z) local_unnamed_addr #0 {
  %1 = addrspacecast i32 addrspace(1)* %z to i32 *
  %call = tail call double @_Z10_cl_remquoddPU8CLglobali(double %x, double %y, i32 * %1)
  ret double %call
}


declare double @_Z10_cl_remquoddPU7CLlocali(double, double, i32 *) local_unnamed_addr #0

define spir_func double @_Z6remquoddPU3AS3i(double %x, double %y, i32 addrspace(3)* %z) local_unnamed_addr #0 {
  %1 = addrspacecast i32 addrspace(3)* %z to i32 *
  %call = tail call double @_Z10_cl_remquoddPU7CLlocali(double %x, double %y, i32 * %1)
  ret double %call
}


declare double @_Z10_cl_remquoddPU9CLprivatei(double, double, i32 *) local_unnamed_addr #0

define spir_func double @_Z6remquoddPi(double %x, double %y, i32 * %z) local_unnamed_addr #0 {
  %call = tail call double @_Z10_cl_remquoddPU9CLprivatei(double %x, double %y, i32 * %z)
  ret double %call
}


declare double @_Z7_cl_nanm(i64) local_unnamed_addr #0

define spir_func double @_Z3nanm(i64 %x) local_unnamed_addr #0 {
  %call = tail call double @_Z7_cl_nanm(i64 %x)
  ret double %call
}


declare double @_Z12_cl_lgamma_rdPU8CLglobali(double, i32 *) local_unnamed_addr #0

define spir_func double @_Z8lgamma_rdPU3AS1i(double %x, i32 addrspace(1)* %y) local_unnamed_addr #0 {
  %1 = addrspacecast i32 addrspace(1)* %y to i32 *
  %call = tail call double @_Z12_cl_lgamma_rdPU8CLglobali(double %x, i32 * %1)
  ret double %call
}


declare double @_Z12_cl_lgamma_rdPU7CLlocali(double, i32 *) local_unnamed_addr #0

define spir_func double @_Z8lgamma_rdPU3AS3i(double %x, i32 addrspace(3)* %y) local_unnamed_addr #0 {
  %1 = addrspacecast i32 addrspace(3)* %y to i32 *
  %call = tail call double @_Z12_cl_lgamma_rdPU7CLlocali(double %x, i32 * %1)
  ret double %call
}


declare double @_Z12_cl_lgamma_rdPU9CLprivatei(double, i32 *) local_unnamed_addr #0

define spir_func double @_Z8lgamma_rdPi(double %x, i32 * %y) local_unnamed_addr #0 {
  %call = tail call double @_Z12_cl_lgamma_rdPU9CLprivatei(double %x, i32 * %y)
  ret double %call
}


declare double @_Z9_cl_frexpdPU8CLglobali(double, i32 *) local_unnamed_addr #0

define spir_func double @_Z5frexpdPU3AS1i(double %x, i32 addrspace(1)* %y) local_unnamed_addr #0 {
  %1 = addrspacecast i32 addrspace(1)* %y to i32 *
  %call = tail call double @_Z9_cl_frexpdPU8CLglobali(double %x, i32 * %1)
  ret double %call
}


declare double @_Z9_cl_frexpdPU7CLlocali(double, i32 *) local_unnamed_addr #0

define spir_func double @_Z5frexpdPU3AS3i(double %x, i32 addrspace(3)* %y) local_unnamed_addr #0 {
  %1 = addrspacecast i32 addrspace(3)* %y to i32 *
  %call = tail call double @_Z9_cl_frexpdPU7CLlocali(double %x, i32 * %1)
  ret double %call
}


declare double @_Z9_cl_frexpdPU9CLprivatei(double, i32 *) local_unnamed_addr #0

define spir_func double @_Z5frexpdPi(double %x, i32 * %y) local_unnamed_addr #0 {
  %call = tail call double @_Z9_cl_frexpdPU9CLprivatei(double %x, i32 * %y)
  ret double %call
}


declare <4 x float> @_Z10_cl_lengthDv4_f(<4 x float>) local_unnamed_addr #0

define spir_func <4 x float> @_Z6lengthDv4_f(<4 x float> %x) local_unnamed_addr #0 {
  %call = tail call <4 x float> @_Z10_cl_lengthDv4_f(<4 x float> %x)
  ret <4 x float> %call
}


declare <4 x double> @_Z10_cl_lengthDv4_d(<4 x double>) local_unnamed_addr #0

define spir_func <4 x double> @_Z6lengthDv4_d(<4 x double> %x) local_unnamed_addr #0 {
  %call = tail call <4 x double> @_Z10_cl_lengthDv4_d(<4 x double> %x)
  ret <4 x double> %call
}


declare  signext i8 @_Z7_cl_absc(i8 signext ) local_unnamed_addr #0

define spir_func  signext i8 @_Z3absc(i8 signext  %x) local_unnamed_addr #0 {
  %call = tail call i8 @_Z7_cl_absc(i8 %x)
  ret i8 %call
}


declare  signext i8 @_Z7_cl_clzc(i8 signext ) local_unnamed_addr #0

define spir_func  signext i8 @_Z3clzc(i8 signext  %x) local_unnamed_addr #0 {
  %call = tail call i8 @_Z7_cl_clzc(i8 %x)
  ret i8 %call
}


declare  signext i8 @_Z12_cl_popcountc(i8 signext ) local_unnamed_addr #0

define spir_func  signext i8 @_Z8popcountc(i8 signext  %x) local_unnamed_addr #0 {
  %call = tail call i8 @_Z12_cl_popcountc(i8 %x)
  ret i8 %call
}


declare  signext i8 @_Z12_cl_abs_diffcc(i8 signext , i8 signext ) local_unnamed_addr #0

define spir_func  signext i8 @_Z8abs_diffcc(i8 signext  %x, i8 signext  %y) local_unnamed_addr #0 {
  %call = tail call i8 @_Z12_cl_abs_diffcc(i8 %x, i8 %y)
  ret i8 %call
}


declare  signext i8 @_Z11_cl_add_satcc(i8 signext , i8 signext ) local_unnamed_addr #0

define spir_func  signext i8 @_Z7add_satcc(i8 signext  %x, i8 signext  %y) local_unnamed_addr #0 {
  %call = tail call i8 @_Z11_cl_add_satcc(i8 %x, i8 %y)
  ret i8 %call
}


declare  signext i8 @_Z8_cl_haddcc(i8 signext , i8 signext ) local_unnamed_addr #0

define spir_func  signext i8 @_Z4haddcc(i8 signext  %x, i8 signext  %y) local_unnamed_addr #0 {
  %call = tail call i8 @_Z8_cl_haddcc(i8 %x, i8 %y)
  ret i8 %call
}


declare  signext i8 @_Z9_cl_rhaddcc(i8 signext , i8 signext ) local_unnamed_addr #0

define spir_func  signext i8 @_Z5rhaddcc(i8 signext  %x, i8 signext  %y) local_unnamed_addr #0 {
  %call = tail call i8 @_Z9_cl_rhaddcc(i8 %x, i8 %y)
  ret i8 %call
}


declare  signext i8 @_Z7_cl_maxcc(i8 signext , i8 signext ) local_unnamed_addr #0

define spir_func  signext i8 @_Z3maxcc(i8 signext  %x, i8 signext  %y) local_unnamed_addr #0 {
  %call = tail call i8 @_Z7_cl_maxcc(i8 %x, i8 %y)
  ret i8 %call
}


declare  signext i8 @_Z7_cl_mincc(i8 signext , i8 signext ) local_unnamed_addr #0

define spir_func  signext i8 @_Z3mincc(i8 signext  %x, i8 signext  %y) local_unnamed_addr #0 {
  %call = tail call i8 @_Z7_cl_mincc(i8 %x, i8 %y)
  ret i8 %call
}


declare  signext i8 @_Z10_cl_mul_hicc(i8 signext , i8 signext ) local_unnamed_addr #0

define spir_func  signext i8 @_Z6mul_hicc(i8 signext  %x, i8 signext  %y) local_unnamed_addr #0 {
  %call = tail call i8 @_Z10_cl_mul_hicc(i8 %x, i8 %y)
  ret i8 %call
}


declare  signext i8 @_Z10_cl_rotatecc(i8 signext , i8 signext ) local_unnamed_addr #0

define spir_func  signext i8 @_Z6rotatecc(i8 signext  %x, i8 signext  %y) local_unnamed_addr #0 {
  %call = tail call i8 @_Z10_cl_rotatecc(i8 %x, i8 %y)
  ret i8 %call
}


declare  signext i8 @_Z11_cl_sub_satcc(i8 signext , i8 signext ) local_unnamed_addr #0

define spir_func  signext i8 @_Z7sub_satcc(i8 signext  %x, i8 signext  %y) local_unnamed_addr #0 {
  %call = tail call i8 @_Z11_cl_sub_satcc(i8 %x, i8 %y)
  ret i8 %call
}


declare  signext i8 @_Z9_cl_clampccc(i8 signext , i8 signext , i8 signext ) local_unnamed_addr #0

define spir_func  signext i8 @_Z5clampccc(i8 signext  %x, i8 signext  %y, i8 signext  %z) local_unnamed_addr #0 {
  %call = tail call i8 @_Z9_cl_clampccc(i8 %x, i8 %y, i8 %z)
  ret i8 %call
}


declare  signext i8 @_Z10_cl_mad_hiccc(i8 signext , i8 signext , i8 signext ) local_unnamed_addr #0

define spir_func  signext i8 @_Z6mad_hiccc(i8 signext  %x, i8 signext  %y, i8 signext  %z) local_unnamed_addr #0 {
  %call = tail call i8 @_Z10_cl_mad_hiccc(i8 %x, i8 %y, i8 %z)
  ret i8 %call
}


declare  signext i8 @_Z11_cl_mad_satccc(i8 signext , i8 signext , i8 signext ) local_unnamed_addr #0

define spir_func  signext i8 @_Z7mad_satccc(i8 signext  %x, i8 signext  %y, i8 signext  %z) local_unnamed_addr #0 {
  %call = tail call i8 @_Z11_cl_mad_satccc(i8 %x, i8 %y, i8 %z)
  ret i8 %call
}


declare  zeroext i8 @_Z7_cl_absh(i8 zeroext ) local_unnamed_addr #0

define spir_func  zeroext i8 @_Z3absh(i8 zeroext  %x) local_unnamed_addr #0 {
  %call = tail call i8 @_Z7_cl_absh(i8 %x)
  ret i8 %call
}


declare  zeroext i8 @_Z7_cl_clzh(i8 zeroext ) local_unnamed_addr #0

define spir_func  zeroext i8 @_Z3clzh(i8 zeroext  %x) local_unnamed_addr #0 {
  %call = tail call i8 @_Z7_cl_clzh(i8 %x)
  ret i8 %call
}


declare  zeroext i8 @_Z12_cl_popcounth(i8 zeroext ) local_unnamed_addr #0

define spir_func  zeroext i8 @_Z8popcounth(i8 zeroext  %x) local_unnamed_addr #0 {
  %call = tail call i8 @_Z12_cl_popcounth(i8 %x)
  ret i8 %call
}


declare  zeroext i8 @_Z12_cl_abs_diffhh(i8 zeroext , i8 zeroext ) local_unnamed_addr #0

define spir_func  zeroext i8 @_Z8abs_diffhh(i8 zeroext  %x, i8 zeroext  %y) local_unnamed_addr #0 {
  %call = tail call i8 @_Z12_cl_abs_diffhh(i8 %x, i8 %y)
  ret i8 %call
}


declare  zeroext i8 @_Z11_cl_add_sathh(i8 zeroext , i8 zeroext ) local_unnamed_addr #0

define spir_func  zeroext i8 @_Z7add_sathh(i8 zeroext  %x, i8 zeroext  %y) local_unnamed_addr #0 {
  %call = tail call i8 @_Z11_cl_add_sathh(i8 %x, i8 %y)
  ret i8 %call
}


declare  zeroext i8 @_Z8_cl_haddhh(i8 zeroext , i8 zeroext ) local_unnamed_addr #0

define spir_func  zeroext i8 @_Z4haddhh(i8 zeroext  %x, i8 zeroext  %y) local_unnamed_addr #0 {
  %call = tail call i8 @_Z8_cl_haddhh(i8 %x, i8 %y)
  ret i8 %call
}


declare  zeroext i8 @_Z9_cl_rhaddhh(i8 zeroext , i8 zeroext ) local_unnamed_addr #0

define spir_func  zeroext i8 @_Z5rhaddhh(i8 zeroext  %x, i8 zeroext  %y) local_unnamed_addr #0 {
  %call = tail call i8 @_Z9_cl_rhaddhh(i8 %x, i8 %y)
  ret i8 %call
}


declare  zeroext i8 @_Z7_cl_maxhh(i8 zeroext , i8 zeroext ) local_unnamed_addr #0

define spir_func  zeroext i8 @_Z3maxhh(i8 zeroext  %x, i8 zeroext  %y) local_unnamed_addr #0 {
  %call = tail call i8 @_Z7_cl_maxhh(i8 %x, i8 %y)
  ret i8 %call
}


declare  zeroext i8 @_Z7_cl_minhh(i8 zeroext , i8 zeroext ) local_unnamed_addr #0

define spir_func  zeroext i8 @_Z3minhh(i8 zeroext  %x, i8 zeroext  %y) local_unnamed_addr #0 {
  %call = tail call i8 @_Z7_cl_minhh(i8 %x, i8 %y)
  ret i8 %call
}


declare  zeroext i8 @_Z10_cl_mul_hihh(i8 zeroext , i8 zeroext ) local_unnamed_addr #0

define spir_func  zeroext i8 @_Z6mul_hihh(i8 zeroext  %x, i8 zeroext  %y) local_unnamed_addr #0 {
  %call = tail call i8 @_Z10_cl_mul_hihh(i8 %x, i8 %y)
  ret i8 %call
}


declare  zeroext i8 @_Z10_cl_rotatehh(i8 zeroext , i8 zeroext ) local_unnamed_addr #0

define spir_func  zeroext i8 @_Z6rotatehh(i8 zeroext  %x, i8 zeroext  %y) local_unnamed_addr #0 {
  %call = tail call i8 @_Z10_cl_rotatehh(i8 %x, i8 %y)
  ret i8 %call
}


declare  zeroext i8 @_Z11_cl_sub_sathh(i8 zeroext , i8 zeroext ) local_unnamed_addr #0

define spir_func  zeroext i8 @_Z7sub_sathh(i8 zeroext  %x, i8 zeroext  %y) local_unnamed_addr #0 {
  %call = tail call i8 @_Z11_cl_sub_sathh(i8 %x, i8 %y)
  ret i8 %call
}


declare  zeroext i8 @_Z9_cl_clamphhh(i8 zeroext , i8 zeroext , i8 zeroext ) local_unnamed_addr #0

define spir_func  zeroext i8 @_Z5clamphhh(i8 zeroext  %x, i8 zeroext  %y, i8 zeroext  %z) local_unnamed_addr #0 {
  %call = tail call i8 @_Z9_cl_clamphhh(i8 %x, i8 %y, i8 %z)
  ret i8 %call
}


declare  zeroext i8 @_Z10_cl_mad_hihhh(i8 zeroext , i8 zeroext , i8 zeroext ) local_unnamed_addr #0

define spir_func  zeroext i8 @_Z6mad_hihhh(i8 zeroext  %x, i8 zeroext  %y, i8 zeroext  %z) local_unnamed_addr #0 {
  %call = tail call i8 @_Z10_cl_mad_hihhh(i8 %x, i8 %y, i8 %z)
  ret i8 %call
}


declare  zeroext i8 @_Z11_cl_mad_sathhh(i8 zeroext , i8 zeroext , i8 zeroext ) local_unnamed_addr #0

define spir_func  zeroext i8 @_Z7mad_sathhh(i8 zeroext  %x, i8 zeroext  %y, i8 zeroext  %z) local_unnamed_addr #0 {
  %call = tail call i8 @_Z11_cl_mad_sathhh(i8 %x, i8 %y, i8 %z)
  ret i8 %call
}


declare  signext i16 @_Z7_cl_abss(i16 signext ) local_unnamed_addr #0

define spir_func  signext i16 @_Z3abss(i16 signext  %x) local_unnamed_addr #0 {
  %call = tail call i16 @_Z7_cl_abss(i16 %x)
  ret i16 %call
}


declare  signext i16 @_Z7_cl_clzs(i16 signext ) local_unnamed_addr #0

define spir_func  signext i16 @_Z3clzs(i16 signext  %x) local_unnamed_addr #0 {
  %call = tail call i16 @_Z7_cl_clzs(i16 %x)
  ret i16 %call
}


declare  signext i16 @_Z12_cl_popcounts(i16 signext ) local_unnamed_addr #0

define spir_func  signext i16 @_Z8popcounts(i16 signext  %x) local_unnamed_addr #0 {
  %call = tail call i16 @_Z12_cl_popcounts(i16 %x)
  ret i16 %call
}


declare  signext i16 @_Z12_cl_abs_diffss(i16 signext , i16 signext ) local_unnamed_addr #0

define spir_func  signext i16 @_Z8abs_diffss(i16 signext  %x, i16 signext  %y) local_unnamed_addr #0 {
  %call = tail call i16 @_Z12_cl_abs_diffss(i16 %x, i16 %y)
  ret i16 %call
}


declare  signext i16 @_Z11_cl_add_satss(i16 signext , i16 signext ) local_unnamed_addr #0

define spir_func  signext i16 @_Z7add_satss(i16 signext  %x, i16 signext  %y) local_unnamed_addr #0 {
  %call = tail call i16 @_Z11_cl_add_satss(i16 %x, i16 %y)
  ret i16 %call
}


declare  signext i16 @_Z8_cl_haddss(i16 signext , i16 signext ) local_unnamed_addr #0

define spir_func  signext i16 @_Z4haddss(i16 signext  %x, i16 signext  %y) local_unnamed_addr #0 {
  %call = tail call i16 @_Z8_cl_haddss(i16 %x, i16 %y)
  ret i16 %call
}


declare  signext i16 @_Z9_cl_rhaddss(i16 signext , i16 signext ) local_unnamed_addr #0

define spir_func  signext i16 @_Z5rhaddss(i16 signext  %x, i16 signext  %y) local_unnamed_addr #0 {
  %call = tail call i16 @_Z9_cl_rhaddss(i16 %x, i16 %y)
  ret i16 %call
}


declare  signext i16 @_Z7_cl_maxss(i16 signext , i16 signext ) local_unnamed_addr #0

define spir_func  signext i16 @_Z3maxss(i16 signext  %x, i16 signext  %y) local_unnamed_addr #0 {
  %call = tail call i16 @_Z7_cl_maxss(i16 %x, i16 %y)
  ret i16 %call
}


declare  signext i16 @_Z7_cl_minss(i16 signext , i16 signext ) local_unnamed_addr #0

define spir_func  signext i16 @_Z3minss(i16 signext  %x, i16 signext  %y) local_unnamed_addr #0 {
  %call = tail call i16 @_Z7_cl_minss(i16 %x, i16 %y)
  ret i16 %call
}


declare  signext i16 @_Z10_cl_mul_hiss(i16 signext , i16 signext ) local_unnamed_addr #0

define spir_func  signext i16 @_Z6mul_hiss(i16 signext  %x, i16 signext  %y) local_unnamed_addr #0 {
  %call = tail call i16 @_Z10_cl_mul_hiss(i16 %x, i16 %y)
  ret i16 %call
}


declare  signext i16 @_Z10_cl_rotatess(i16 signext , i16 signext ) local_unnamed_addr #0

define spir_func  signext i16 @_Z6rotatess(i16 signext  %x, i16 signext  %y) local_unnamed_addr #0 {
  %call = tail call i16 @_Z10_cl_rotatess(i16 %x, i16 %y)
  ret i16 %call
}


declare  signext i16 @_Z11_cl_sub_satss(i16 signext , i16 signext ) local_unnamed_addr #0

define spir_func  signext i16 @_Z7sub_satss(i16 signext  %x, i16 signext  %y) local_unnamed_addr #0 {
  %call = tail call i16 @_Z11_cl_sub_satss(i16 %x, i16 %y)
  ret i16 %call
}


declare  signext i16 @_Z9_cl_clampsss(i16 signext , i16 signext , i16 signext ) local_unnamed_addr #0

define spir_func  signext i16 @_Z5clampsss(i16 signext  %x, i16 signext  %y, i16 signext  %z) local_unnamed_addr #0 {
  %call = tail call i16 @_Z9_cl_clampsss(i16 %x, i16 %y, i16 %z)
  ret i16 %call
}


declare  signext i16 @_Z10_cl_mad_hisss(i16 signext , i16 signext , i16 signext ) local_unnamed_addr #0

define spir_func  signext i16 @_Z6mad_hisss(i16 signext  %x, i16 signext  %y, i16 signext  %z) local_unnamed_addr #0 {
  %call = tail call i16 @_Z10_cl_mad_hisss(i16 %x, i16 %y, i16 %z)
  ret i16 %call
}


declare  signext i16 @_Z11_cl_mad_satsss(i16 signext , i16 signext , i16 signext ) local_unnamed_addr #0

define spir_func  signext i16 @_Z7mad_satsss(i16 signext  %x, i16 signext  %y, i16 signext  %z) local_unnamed_addr #0 {
  %call = tail call i16 @_Z11_cl_mad_satsss(i16 %x, i16 %y, i16 %z)
  ret i16 %call
}


declare  zeroext i16 @_Z7_cl_abst(i16 zeroext ) local_unnamed_addr #0

define spir_func  zeroext i16 @_Z3abst(i16 zeroext  %x) local_unnamed_addr #0 {
  %call = tail call i16 @_Z7_cl_abst(i16 %x)
  ret i16 %call
}


declare  zeroext i16 @_Z7_cl_clzt(i16 zeroext ) local_unnamed_addr #0

define spir_func  zeroext i16 @_Z3clzt(i16 zeroext  %x) local_unnamed_addr #0 {
  %call = tail call i16 @_Z7_cl_clzt(i16 %x)
  ret i16 %call
}


declare  zeroext i16 @_Z12_cl_popcountt(i16 zeroext ) local_unnamed_addr #0

define spir_func  zeroext i16 @_Z8popcountt(i16 zeroext  %x) local_unnamed_addr #0 {
  %call = tail call i16 @_Z12_cl_popcountt(i16 %x)
  ret i16 %call
}


declare  zeroext i16 @_Z12_cl_abs_difftt(i16 zeroext , i16 zeroext ) local_unnamed_addr #0

define spir_func  zeroext i16 @_Z8abs_difftt(i16 zeroext  %x, i16 zeroext  %y) local_unnamed_addr #0 {
  %call = tail call i16 @_Z12_cl_abs_difftt(i16 %x, i16 %y)
  ret i16 %call
}


declare  zeroext i16 @_Z11_cl_add_sattt(i16 zeroext , i16 zeroext ) local_unnamed_addr #0

define spir_func  zeroext i16 @_Z7add_sattt(i16 zeroext  %x, i16 zeroext  %y) local_unnamed_addr #0 {
  %call = tail call i16 @_Z11_cl_add_sattt(i16 %x, i16 %y)
  ret i16 %call
}


declare  zeroext i16 @_Z8_cl_haddtt(i16 zeroext , i16 zeroext ) local_unnamed_addr #0

define spir_func  zeroext i16 @_Z4haddtt(i16 zeroext  %x, i16 zeroext  %y) local_unnamed_addr #0 {
  %call = tail call i16 @_Z8_cl_haddtt(i16 %x, i16 %y)
  ret i16 %call
}


declare  zeroext i16 @_Z9_cl_rhaddtt(i16 zeroext , i16 zeroext ) local_unnamed_addr #0

define spir_func  zeroext i16 @_Z5rhaddtt(i16 zeroext  %x, i16 zeroext  %y) local_unnamed_addr #0 {
  %call = tail call i16 @_Z9_cl_rhaddtt(i16 %x, i16 %y)
  ret i16 %call
}


declare  zeroext i16 @_Z7_cl_maxtt(i16 zeroext , i16 zeroext ) local_unnamed_addr #0

define spir_func  zeroext i16 @_Z3maxtt(i16 zeroext  %x, i16 zeroext  %y) local_unnamed_addr #0 {
  %call = tail call i16 @_Z7_cl_maxtt(i16 %x, i16 %y)
  ret i16 %call
}


declare  zeroext i16 @_Z7_cl_mintt(i16 zeroext , i16 zeroext ) local_unnamed_addr #0

define spir_func  zeroext i16 @_Z3mintt(i16 zeroext  %x, i16 zeroext  %y) local_unnamed_addr #0 {
  %call = tail call i16 @_Z7_cl_mintt(i16 %x, i16 %y)
  ret i16 %call
}


declare  zeroext i16 @_Z10_cl_mul_hitt(i16 zeroext , i16 zeroext ) local_unnamed_addr #0

define spir_func  zeroext i16 @_Z6mul_hitt(i16 zeroext  %x, i16 zeroext  %y) local_unnamed_addr #0 {
  %call = tail call i16 @_Z10_cl_mul_hitt(i16 %x, i16 %y)
  ret i16 %call
}


declare  zeroext i16 @_Z10_cl_rotatett(i16 zeroext , i16 zeroext ) local_unnamed_addr #0

define spir_func  zeroext i16 @_Z6rotatett(i16 zeroext  %x, i16 zeroext  %y) local_unnamed_addr #0 {
  %call = tail call i16 @_Z10_cl_rotatett(i16 %x, i16 %y)
  ret i16 %call
}


declare  zeroext i16 @_Z11_cl_sub_sattt(i16 zeroext , i16 zeroext ) local_unnamed_addr #0

define spir_func  zeroext i16 @_Z7sub_sattt(i16 zeroext  %x, i16 zeroext  %y) local_unnamed_addr #0 {
  %call = tail call i16 @_Z11_cl_sub_sattt(i16 %x, i16 %y)
  ret i16 %call
}


declare  zeroext i16 @_Z9_cl_clampttt(i16 zeroext , i16 zeroext , i16 zeroext ) local_unnamed_addr #0

define spir_func  zeroext i16 @_Z5clampttt(i16 zeroext  %x, i16 zeroext  %y, i16 zeroext  %z) local_unnamed_addr #0 {
  %call = tail call i16 @_Z9_cl_clampttt(i16 %x, i16 %y, i16 %z)
  ret i16 %call
}


declare  zeroext i16 @_Z10_cl_mad_hittt(i16 zeroext , i16 zeroext , i16 zeroext ) local_unnamed_addr #0

define spir_func  zeroext i16 @_Z6mad_hittt(i16 zeroext  %x, i16 zeroext  %y, i16 zeroext  %z) local_unnamed_addr #0 {
  %call = tail call i16 @_Z10_cl_mad_hittt(i16 %x, i16 %y, i16 %z)
  ret i16 %call
}


declare  zeroext i16 @_Z11_cl_mad_satttt(i16 zeroext , i16 zeroext , i16 zeroext ) local_unnamed_addr #0

define spir_func  zeroext i16 @_Z7mad_satttt(i16 zeroext  %x, i16 zeroext  %y, i16 zeroext  %z) local_unnamed_addr #0 {
  %call = tail call i16 @_Z11_cl_mad_satttt(i16 %x, i16 %y, i16 %z)
  ret i16 %call
}


declare i32 @_Z7_cl_absi(i32) local_unnamed_addr #0

define spir_func i32 @_Z3absi(i32 %x) local_unnamed_addr #0 {
  %call = tail call i32 @_Z7_cl_absi(i32 %x)
  ret i32 %call
}


declare i32 @_Z7_cl_clzi(i32) local_unnamed_addr #0

define spir_func i32 @_Z3clzi(i32 %x) local_unnamed_addr #0 {
  %call = tail call i32 @_Z7_cl_clzi(i32 %x)
  ret i32 %call
}


declare i32 @_Z12_cl_popcounti(i32) local_unnamed_addr #0

define spir_func i32 @_Z8popcounti(i32 %x) local_unnamed_addr #0 {
  %call = tail call i32 @_Z12_cl_popcounti(i32 %x)
  ret i32 %call
}


declare i32 @_Z12_cl_abs_diffii(i32, i32) local_unnamed_addr #0

define spir_func i32 @_Z8abs_diffii(i32 %x, i32 %y) local_unnamed_addr #0 {
  %call = tail call i32 @_Z12_cl_abs_diffii(i32 %x, i32 %y)
  ret i32 %call
}


declare i32 @_Z11_cl_add_satii(i32, i32) local_unnamed_addr #0

define spir_func i32 @_Z7add_satii(i32 %x, i32 %y) local_unnamed_addr #0 {
  %call = tail call i32 @_Z11_cl_add_satii(i32 %x, i32 %y)
  ret i32 %call
}


declare i32 @_Z8_cl_haddii(i32, i32) local_unnamed_addr #0

define spir_func i32 @_Z4haddii(i32 %x, i32 %y) local_unnamed_addr #0 {
  %call = tail call i32 @_Z8_cl_haddii(i32 %x, i32 %y)
  ret i32 %call
}


declare i32 @_Z9_cl_rhaddii(i32, i32) local_unnamed_addr #0

define spir_func i32 @_Z5rhaddii(i32 %x, i32 %y) local_unnamed_addr #0 {
  %call = tail call i32 @_Z9_cl_rhaddii(i32 %x, i32 %y)
  ret i32 %call
}


declare i32 @_Z7_cl_maxii(i32, i32) local_unnamed_addr #0

define spir_func i32 @_Z3maxii(i32 %x, i32 %y) local_unnamed_addr #0 {
  %call = tail call i32 @_Z7_cl_maxii(i32 %x, i32 %y)
  ret i32 %call
}


declare i32 @_Z7_cl_minii(i32, i32) local_unnamed_addr #0

define spir_func i32 @_Z3minii(i32 %x, i32 %y) local_unnamed_addr #0 {
  %call = tail call i32 @_Z7_cl_minii(i32 %x, i32 %y)
  ret i32 %call
}


declare i32 @_Z10_cl_mul_hiii(i32, i32) local_unnamed_addr #0

define spir_func i32 @_Z6mul_hiii(i32 %x, i32 %y) local_unnamed_addr #0 {
  %call = tail call i32 @_Z10_cl_mul_hiii(i32 %x, i32 %y)
  ret i32 %call
}


declare i32 @_Z10_cl_rotateii(i32, i32) local_unnamed_addr #0

define spir_func i32 @_Z6rotateii(i32 %x, i32 %y) local_unnamed_addr #0 {
  %call = tail call i32 @_Z10_cl_rotateii(i32 %x, i32 %y)
  ret i32 %call
}


declare i32 @_Z11_cl_sub_satii(i32, i32) local_unnamed_addr #0

define spir_func i32 @_Z7sub_satii(i32 %x, i32 %y) local_unnamed_addr #0 {
  %call = tail call i32 @_Z11_cl_sub_satii(i32 %x, i32 %y)
  ret i32 %call
}


declare i32 @_Z9_cl_clampiii(i32, i32, i32) local_unnamed_addr #0

define spir_func i32 @_Z5clampiii(i32 %x, i32 %y, i32 %z) local_unnamed_addr #0 {
  %call = tail call i32 @_Z9_cl_clampiii(i32 %x, i32 %y, i32 %z)
  ret i32 %call
}


declare i32 @_Z10_cl_mad_hiiii(i32, i32, i32) local_unnamed_addr #0

define spir_func i32 @_Z6mad_hiiii(i32 %x, i32 %y, i32 %z) local_unnamed_addr #0 {
  %call = tail call i32 @_Z10_cl_mad_hiiii(i32 %x, i32 %y, i32 %z)
  ret i32 %call
}


declare i32 @_Z11_cl_mad_satiii(i32, i32, i32) local_unnamed_addr #0

define spir_func i32 @_Z7mad_satiii(i32 %x, i32 %y, i32 %z) local_unnamed_addr #0 {
  %call = tail call i32 @_Z11_cl_mad_satiii(i32 %x, i32 %y, i32 %z)
  ret i32 %call
}


declare i32 @_Z7_cl_absj(i32) local_unnamed_addr #0

define spir_func i32 @_Z3absj(i32 %x) local_unnamed_addr #0 {
  %call = tail call i32 @_Z7_cl_absj(i32 %x)
  ret i32 %call
}


declare i32 @_Z7_cl_clzj(i32) local_unnamed_addr #0

define spir_func i32 @_Z3clzj(i32 %x) local_unnamed_addr #0 {
  %call = tail call i32 @_Z7_cl_clzj(i32 %x)
  ret i32 %call
}


declare i32 @_Z12_cl_popcountj(i32) local_unnamed_addr #0

define spir_func i32 @_Z8popcountj(i32 %x) local_unnamed_addr #0 {
  %call = tail call i32 @_Z12_cl_popcountj(i32 %x)
  ret i32 %call
}


declare i32 @_Z12_cl_abs_diffjj(i32, i32) local_unnamed_addr #0

define spir_func i32 @_Z8abs_diffjj(i32 %x, i32 %y) local_unnamed_addr #0 {
  %call = tail call i32 @_Z12_cl_abs_diffjj(i32 %x, i32 %y)
  ret i32 %call
}


declare i32 @_Z11_cl_add_satjj(i32, i32) local_unnamed_addr #0

define spir_func i32 @_Z7add_satjj(i32 %x, i32 %y) local_unnamed_addr #0 {
  %call = tail call i32 @_Z11_cl_add_satjj(i32 %x, i32 %y)
  ret i32 %call
}


declare i32 @_Z8_cl_haddjj(i32, i32) local_unnamed_addr #0

define spir_func i32 @_Z4haddjj(i32 %x, i32 %y) local_unnamed_addr #0 {
  %call = tail call i32 @_Z8_cl_haddjj(i32 %x, i32 %y)
  ret i32 %call
}


declare i32 @_Z9_cl_rhaddjj(i32, i32) local_unnamed_addr #0

define spir_func i32 @_Z5rhaddjj(i32 %x, i32 %y) local_unnamed_addr #0 {
  %call = tail call i32 @_Z9_cl_rhaddjj(i32 %x, i32 %y)
  ret i32 %call
}


declare i32 @_Z7_cl_maxjj(i32, i32) local_unnamed_addr #0

define spir_func i32 @_Z3maxjj(i32 %x, i32 %y) local_unnamed_addr #0 {
  %call = tail call i32 @_Z7_cl_maxjj(i32 %x, i32 %y)
  ret i32 %call
}


declare i32 @_Z7_cl_minjj(i32, i32) local_unnamed_addr #0

define spir_func i32 @_Z3minjj(i32 %x, i32 %y) local_unnamed_addr #0 {
  %call = tail call i32 @_Z7_cl_minjj(i32 %x, i32 %y)
  ret i32 %call
}


declare i32 @_Z10_cl_mul_hijj(i32, i32) local_unnamed_addr #0

define spir_func i32 @_Z6mul_hijj(i32 %x, i32 %y) local_unnamed_addr #0 {
  %call = tail call i32 @_Z10_cl_mul_hijj(i32 %x, i32 %y)
  ret i32 %call
}


declare i32 @_Z10_cl_rotatejj(i32, i32) local_unnamed_addr #0

define spir_func i32 @_Z6rotatejj(i32 %x, i32 %y) local_unnamed_addr #0 {
  %call = tail call i32 @_Z10_cl_rotatejj(i32 %x, i32 %y)
  ret i32 %call
}


declare i32 @_Z11_cl_sub_satjj(i32, i32) local_unnamed_addr #0

define spir_func i32 @_Z7sub_satjj(i32 %x, i32 %y) local_unnamed_addr #0 {
  %call = tail call i32 @_Z11_cl_sub_satjj(i32 %x, i32 %y)
  ret i32 %call
}


declare i32 @_Z9_cl_clampjjj(i32, i32, i32) local_unnamed_addr #0

define spir_func i32 @_Z5clampjjj(i32 %x, i32 %y, i32 %z) local_unnamed_addr #0 {
  %call = tail call i32 @_Z9_cl_clampjjj(i32 %x, i32 %y, i32 %z)
  ret i32 %call
}


declare i32 @_Z10_cl_mad_hijjj(i32, i32, i32) local_unnamed_addr #0

define spir_func i32 @_Z6mad_hijjj(i32 %x, i32 %y, i32 %z) local_unnamed_addr #0 {
  %call = tail call i32 @_Z10_cl_mad_hijjj(i32 %x, i32 %y, i32 %z)
  ret i32 %call
}


declare i32 @_Z11_cl_mad_satjjj(i32, i32, i32) local_unnamed_addr #0

define spir_func i32 @_Z7mad_satjjj(i32 %x, i32 %y, i32 %z) local_unnamed_addr #0 {
  %call = tail call i32 @_Z11_cl_mad_satjjj(i32 %x, i32 %y, i32 %z)
  ret i32 %call
}


declare i64 @_Z7_cl_absl(i64) local_unnamed_addr #0

define spir_func i64 @_Z3absl(i64 %x) local_unnamed_addr #0 {
  %call = tail call i64 @_Z7_cl_absl(i64 %x)
  ret i64 %call
}


declare i64 @_Z7_cl_clzl(i64) local_unnamed_addr #0

define spir_func i64 @_Z3clzl(i64 %x) local_unnamed_addr #0 {
  %call = tail call i64 @_Z7_cl_clzl(i64 %x)
  ret i64 %call
}


declare i64 @_Z12_cl_popcountl(i64) local_unnamed_addr #0

define spir_func i64 @_Z8popcountl(i64 %x) local_unnamed_addr #0 {
  %call = tail call i64 @_Z12_cl_popcountl(i64 %x)
  ret i64 %call
}


declare i64 @_Z12_cl_abs_diffll(i64, i64) local_unnamed_addr #0

define spir_func i64 @_Z8abs_diffll(i64 %x, i64 %y) local_unnamed_addr #0 {
  %call = tail call i64 @_Z12_cl_abs_diffll(i64 %x, i64 %y)
  ret i64 %call
}


declare i64 @_Z11_cl_add_satll(i64, i64) local_unnamed_addr #0

define spir_func i64 @_Z7add_satll(i64 %x, i64 %y) local_unnamed_addr #0 {
  %call = tail call i64 @_Z11_cl_add_satll(i64 %x, i64 %y)
  ret i64 %call
}


declare i64 @_Z8_cl_haddll(i64, i64) local_unnamed_addr #0

define spir_func i64 @_Z4haddll(i64 %x, i64 %y) local_unnamed_addr #0 {
  %call = tail call i64 @_Z8_cl_haddll(i64 %x, i64 %y)
  ret i64 %call
}


declare i64 @_Z9_cl_rhaddll(i64, i64) local_unnamed_addr #0

define spir_func i64 @_Z5rhaddll(i64 %x, i64 %y) local_unnamed_addr #0 {
  %call = tail call i64 @_Z9_cl_rhaddll(i64 %x, i64 %y)
  ret i64 %call
}


declare i64 @_Z7_cl_maxll(i64, i64) local_unnamed_addr #0

define spir_func i64 @_Z3maxll(i64 %x, i64 %y) local_unnamed_addr #0 {
  %call = tail call i64 @_Z7_cl_maxll(i64 %x, i64 %y)
  ret i64 %call
}


declare i64 @_Z7_cl_minll(i64, i64) local_unnamed_addr #0

define spir_func i64 @_Z3minll(i64 %x, i64 %y) local_unnamed_addr #0 {
  %call = tail call i64 @_Z7_cl_minll(i64 %x, i64 %y)
  ret i64 %call
}


declare i64 @_Z10_cl_mul_hill(i64, i64) local_unnamed_addr #0

define spir_func i64 @_Z6mul_hill(i64 %x, i64 %y) local_unnamed_addr #0 {
  %call = tail call i64 @_Z10_cl_mul_hill(i64 %x, i64 %y)
  ret i64 %call
}


declare i64 @_Z10_cl_rotatell(i64, i64) local_unnamed_addr #0

define spir_func i64 @_Z6rotatell(i64 %x, i64 %y) local_unnamed_addr #0 {
  %call = tail call i64 @_Z10_cl_rotatell(i64 %x, i64 %y)
  ret i64 %call
}


declare i64 @_Z11_cl_sub_satll(i64, i64) local_unnamed_addr #0

define spir_func i64 @_Z7sub_satll(i64 %x, i64 %y) local_unnamed_addr #0 {
  %call = tail call i64 @_Z11_cl_sub_satll(i64 %x, i64 %y)
  ret i64 %call
}


declare i64 @_Z9_cl_clamplll(i64, i64, i64) local_unnamed_addr #0

define spir_func i64 @_Z5clamplll(i64 %x, i64 %y, i64 %z) local_unnamed_addr #0 {
  %call = tail call i64 @_Z9_cl_clamplll(i64 %x, i64 %y, i64 %z)
  ret i64 %call
}


declare i64 @_Z10_cl_mad_hilll(i64, i64, i64) local_unnamed_addr #0

define spir_func i64 @_Z6mad_hilll(i64 %x, i64 %y, i64 %z) local_unnamed_addr #0 {
  %call = tail call i64 @_Z10_cl_mad_hilll(i64 %x, i64 %y, i64 %z)
  ret i64 %call
}


declare i64 @_Z11_cl_mad_satlll(i64, i64, i64) local_unnamed_addr #0

define spir_func i64 @_Z7mad_satlll(i64 %x, i64 %y, i64 %z) local_unnamed_addr #0 {
  %call = tail call i64 @_Z11_cl_mad_satlll(i64 %x, i64 %y, i64 %z)
  ret i64 %call
}


declare i64 @_Z7_cl_absm(i64) local_unnamed_addr #0

define spir_func i64 @_Z3absm(i64 %x) local_unnamed_addr #0 {
  %call = tail call i64 @_Z7_cl_absm(i64 %x)
  ret i64 %call
}


declare i64 @_Z7_cl_clzm(i64) local_unnamed_addr #0

define spir_func i64 @_Z3clzm(i64 %x) local_unnamed_addr #0 {
  %call = tail call i64 @_Z7_cl_clzm(i64 %x)
  ret i64 %call
}


declare i64 @_Z12_cl_popcountm(i64) local_unnamed_addr #0

define spir_func i64 @_Z8popcountm(i64 %x) local_unnamed_addr #0 {
  %call = tail call i64 @_Z12_cl_popcountm(i64 %x)
  ret i64 %call
}


declare i64 @_Z12_cl_abs_diffmm(i64, i64) local_unnamed_addr #0

define spir_func i64 @_Z8abs_diffmm(i64 %x, i64 %y) local_unnamed_addr #0 {
  %call = tail call i64 @_Z12_cl_abs_diffmm(i64 %x, i64 %y)
  ret i64 %call
}


declare i64 @_Z11_cl_add_satmm(i64, i64) local_unnamed_addr #0

define spir_func i64 @_Z7add_satmm(i64 %x, i64 %y) local_unnamed_addr #0 {
  %call = tail call i64 @_Z11_cl_add_satmm(i64 %x, i64 %y)
  ret i64 %call
}


declare i64 @_Z8_cl_haddmm(i64, i64) local_unnamed_addr #0

define spir_func i64 @_Z4haddmm(i64 %x, i64 %y) local_unnamed_addr #0 {
  %call = tail call i64 @_Z8_cl_haddmm(i64 %x, i64 %y)
  ret i64 %call
}


declare i64 @_Z9_cl_rhaddmm(i64, i64) local_unnamed_addr #0

define spir_func i64 @_Z5rhaddmm(i64 %x, i64 %y) local_unnamed_addr #0 {
  %call = tail call i64 @_Z9_cl_rhaddmm(i64 %x, i64 %y)
  ret i64 %call
}


declare i64 @_Z7_cl_maxmm(i64, i64) local_unnamed_addr #0

define spir_func i64 @_Z3maxmm(i64 %x, i64 %y) local_unnamed_addr #0 {
  %call = tail call i64 @_Z7_cl_maxmm(i64 %x, i64 %y)
  ret i64 %call
}


declare i64 @_Z7_cl_minmm(i64, i64) local_unnamed_addr #0

define spir_func i64 @_Z3minmm(i64 %x, i64 %y) local_unnamed_addr #0 {
  %call = tail call i64 @_Z7_cl_minmm(i64 %x, i64 %y)
  ret i64 %call
}


declare i64 @_Z10_cl_mul_himm(i64, i64) local_unnamed_addr #0

define spir_func i64 @_Z6mul_himm(i64 %x, i64 %y) local_unnamed_addr #0 {
  %call = tail call i64 @_Z10_cl_mul_himm(i64 %x, i64 %y)
  ret i64 %call
}


declare i64 @_Z10_cl_rotatemm(i64, i64) local_unnamed_addr #0

define spir_func i64 @_Z6rotatemm(i64 %x, i64 %y) local_unnamed_addr #0 {
  %call = tail call i64 @_Z10_cl_rotatemm(i64 %x, i64 %y)
  ret i64 %call
}


declare i64 @_Z11_cl_sub_satmm(i64, i64) local_unnamed_addr #0

define spir_func i64 @_Z7sub_satmm(i64 %x, i64 %y) local_unnamed_addr #0 {
  %call = tail call i64 @_Z11_cl_sub_satmm(i64 %x, i64 %y)
  ret i64 %call
}


declare i64 @_Z9_cl_clampmmm(i64, i64, i64) local_unnamed_addr #0

define spir_func i64 @_Z5clampmmm(i64 %x, i64 %y, i64 %z) local_unnamed_addr #0 {
  %call = tail call i64 @_Z9_cl_clampmmm(i64 %x, i64 %y, i64 %z)
  ret i64 %call
}


declare i64 @_Z10_cl_mad_himmm(i64, i64, i64) local_unnamed_addr #0

define spir_func i64 @_Z6mad_himmm(i64 %x, i64 %y, i64 %z) local_unnamed_addr #0 {
  %call = tail call i64 @_Z10_cl_mad_himmm(i64 %x, i64 %y, i64 %z)
  ret i64 %call
}


declare i64 @_Z11_cl_mad_satmmm(i64, i64, i64) local_unnamed_addr #0

define spir_func i64 @_Z7mad_satmmm(i64 %x, i64 %y, i64 %z) local_unnamed_addr #0 {
  %call = tail call i64 @_Z11_cl_mad_satmmm(i64 %x, i64 %y, i64 %z)
  ret i64 %call
}


declare i32 @_Z9_cl_mul24ii(i32, i32) local_unnamed_addr #0

define spir_func i32 @_Z5mul24ii(i32 %x, i32 %y) local_unnamed_addr #0 {
  %call = tail call i32 @_Z9_cl_mul24ii(i32 %x, i32 %y)
  ret i32 %call
}


declare i32 @_Z9_cl_mul24jj(i32, i32) local_unnamed_addr #0

define spir_func i32 @_Z5mul24jj(i32 %x, i32 %y) local_unnamed_addr #0 {
  %call = tail call i32 @_Z9_cl_mul24jj(i32 %x, i32 %y)
  ret i32 %call
}


declare i32 @_Z9_cl_mad24iii(i32, i32, i32) local_unnamed_addr #0

define spir_func i32 @_Z5mad24iii(i32 %x, i32 %y, i32 %z) local_unnamed_addr #0 {
  %call = tail call i32 @_Z9_cl_mad24iii(i32 %x, i32 %y, i32 %z)
  ret i32 %call
}


declare i32 @_Z9_cl_mad24jjj(i32, i32, i32) local_unnamed_addr #0

define spir_func i32 @_Z5mad24jjj(i32 %x, i32 %y, i32 %z) local_unnamed_addr #0 {
  %call = tail call i32 @_Z9_cl_mad24jjj(i32 %x, i32 %y, i32 %z)
  ret i32 %call
}




attributes #0 = { nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "denorms-are-zero"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }

!opencl.ocl.version = !{!0}
!opencl.spir.version = !{!0}
!llvm.ident = !{!1}
!llvm.module.flags = !{!2, !3}

!0 = !{i32 1, i32 2}
!1 = !{!"clang version 6.0.0"}
!2 = !{i32 1, !"wchar_size", i32 4}
!3 = !{i32 7, !"PIC Level", i32 2}


