
; ModuleID = 'spir_wrapper.bc'
source_filename = "generate_spir_wrapper.py"
target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024"
target triple = "spir64-unknown-unknown"

declare float @_Z8_cl_acosf(float) local_unnamed_addr #0

define spir_func float @_Z4acosf(float %a) local_unnamed_addr #0 {
  %call = tail call  float @_Z8_cl_acosf(float %a)
  ret float %call
}


declare float @_Z9_cl_acoshf(float) local_unnamed_addr #0

define spir_func float @_Z5acoshf(float %a) local_unnamed_addr #0 {
  %call = tail call  float @_Z9_cl_acoshf(float %a)
  ret float %call
}


declare float @_Z10_cl_acospif(float) local_unnamed_addr #0

define spir_func float @_Z6acospif(float %a) local_unnamed_addr #0 {
  %call = tail call  float @_Z10_cl_acospif(float %a)
  ret float %call
}


declare float @_Z8_cl_asinf(float) local_unnamed_addr #0

define spir_func float @_Z4asinf(float %a) local_unnamed_addr #0 {
  %call = tail call  float @_Z8_cl_asinf(float %a)
  ret float %call
}


declare float @_Z9_cl_asinhf(float) local_unnamed_addr #0

define spir_func float @_Z5asinhf(float %a) local_unnamed_addr #0 {
  %call = tail call  float @_Z9_cl_asinhf(float %a)
  ret float %call
}


declare float @_Z10_cl_asinpif(float) local_unnamed_addr #0

define spir_func float @_Z6asinpif(float %a) local_unnamed_addr #0 {
  %call = tail call  float @_Z10_cl_asinpif(float %a)
  ret float %call
}


declare float @_Z8_cl_atanf(float) local_unnamed_addr #0

define spir_func float @_Z4atanf(float %a) local_unnamed_addr #0 {
  %call = tail call  float @_Z8_cl_atanf(float %a)
  ret float %call
}


declare float @_Z9_cl_atanhf(float) local_unnamed_addr #0

define spir_func float @_Z5atanhf(float %a) local_unnamed_addr #0 {
  %call = tail call  float @_Z9_cl_atanhf(float %a)
  ret float %call
}


declare float @_Z10_cl_atanpif(float) local_unnamed_addr #0

define spir_func float @_Z6atanpif(float %a) local_unnamed_addr #0 {
  %call = tail call  float @_Z10_cl_atanpif(float %a)
  ret float %call
}


declare float @_Z8_cl_cbrtf(float) local_unnamed_addr #0

define spir_func float @_Z4cbrtf(float %a) local_unnamed_addr #0 {
  %call = tail call  float @_Z8_cl_cbrtf(float %a)
  ret float %call
}


declare float @_Z8_cl_ceilf(float) local_unnamed_addr #0

define spir_func float @_Z4ceilf(float %a) local_unnamed_addr #0 {
  %call = tail call  float @_Z8_cl_ceilf(float %a)
  ret float %call
}


declare float @_Z7_cl_cosf(float) local_unnamed_addr #0

define spir_func float @_Z3cosf(float %a) local_unnamed_addr #0 {
  %call = tail call  float @_Z7_cl_cosf(float %a)
  ret float %call
}


declare float @_Z8_cl_coshf(float) local_unnamed_addr #0

define spir_func float @_Z4coshf(float %a) local_unnamed_addr #0 {
  %call = tail call  float @_Z8_cl_coshf(float %a)
  ret float %call
}


declare float @_Z9_cl_cospif(float) local_unnamed_addr #0

define spir_func float @_Z5cospif(float %a) local_unnamed_addr #0 {
  %call = tail call  float @_Z9_cl_cospif(float %a)
  ret float %call
}


declare float @_Z8_cl_erfcf(float) local_unnamed_addr #0

define spir_func float @_Z4erfcf(float %a) local_unnamed_addr #0 {
  %call = tail call  float @_Z8_cl_erfcf(float %a)
  ret float %call
}


declare float @_Z7_cl_erff(float) local_unnamed_addr #0

define spir_func float @_Z3erff(float %a) local_unnamed_addr #0 {
  %call = tail call  float @_Z7_cl_erff(float %a)
  ret float %call
}


declare float @_Z7_cl_expf(float) local_unnamed_addr #0

define spir_func float @_Z3expf(float %a) local_unnamed_addr #0 {
  %call = tail call  float @_Z7_cl_expf(float %a)
  ret float %call
}


declare float @_Z8_cl_exp2f(float) local_unnamed_addr #0

define spir_func float @_Z4exp2f(float %a) local_unnamed_addr #0 {
  %call = tail call  float @_Z8_cl_exp2f(float %a)
  ret float %call
}


declare float @_Z9_cl_exp10f(float) local_unnamed_addr #0

define spir_func float @_Z5exp10f(float %a) local_unnamed_addr #0 {
  %call = tail call  float @_Z9_cl_exp10f(float %a)
  ret float %call
}


declare float @_Z9_cl_expm1f(float) local_unnamed_addr #0

define spir_func float @_Z5expm1f(float %a) local_unnamed_addr #0 {
  %call = tail call  float @_Z9_cl_expm1f(float %a)
  ret float %call
}


declare float @_Z8_cl_fabsf(float) local_unnamed_addr #0

define spir_func float @_Z4fabsf(float %a) local_unnamed_addr #0 {
  %call = tail call  float @_Z8_cl_fabsf(float %a)
  ret float %call
}


declare float @_Z9_cl_floorf(float) local_unnamed_addr #0

define spir_func float @_Z5floorf(float %a) local_unnamed_addr #0 {
  %call = tail call  float @_Z9_cl_floorf(float %a)
  ret float %call
}


declare float @_Z10_cl_lgammaf(float) local_unnamed_addr #0

define spir_func float @_Z6lgammaf(float %a) local_unnamed_addr #0 {
  %call = tail call  float @_Z10_cl_lgammaf(float %a)
  ret float %call
}


declare float @_Z7_cl_logf(float) local_unnamed_addr #0

define spir_func float @_Z3logf(float %a) local_unnamed_addr #0 {
  %call = tail call  float @_Z7_cl_logf(float %a)
  ret float %call
}


declare float @_Z9_cl_log10f(float) local_unnamed_addr #0

define spir_func float @_Z5log10f(float %a) local_unnamed_addr #0 {
  %call = tail call  float @_Z9_cl_log10f(float %a)
  ret float %call
}


declare float @_Z8_cl_log2f(float) local_unnamed_addr #0

define spir_func float @_Z4log2f(float %a) local_unnamed_addr #0 {
  %call = tail call  float @_Z8_cl_log2f(float %a)
  ret float %call
}


declare float @_Z9_cl_log1pf(float) local_unnamed_addr #0

define spir_func float @_Z5log1pf(float %a) local_unnamed_addr #0 {
  %call = tail call  float @_Z9_cl_log1pf(float %a)
  ret float %call
}


declare float @_Z8_cl_rintf(float) local_unnamed_addr #0

define spir_func float @_Z4rintf(float %a) local_unnamed_addr #0 {
  %call = tail call  float @_Z8_cl_rintf(float %a)
  ret float %call
}


declare float @_Z9_cl_roundf(float) local_unnamed_addr #0

define spir_func float @_Z5roundf(float %a) local_unnamed_addr #0 {
  %call = tail call  float @_Z9_cl_roundf(float %a)
  ret float %call
}


declare float @_Z9_cl_rsqrtf(float) local_unnamed_addr #0

define spir_func float @_Z5rsqrtf(float %a) local_unnamed_addr #0 {
  %call = tail call  float @_Z9_cl_rsqrtf(float %a)
  ret float %call
}


declare float @_Z7_cl_sinf(float) local_unnamed_addr #0

define spir_func float @_Z3sinf(float %a) local_unnamed_addr #0 {
  %call = tail call  float @_Z7_cl_sinf(float %a)
  ret float %call
}


declare float @_Z8_cl_sinhf(float) local_unnamed_addr #0

define spir_func float @_Z4sinhf(float %a) local_unnamed_addr #0 {
  %call = tail call  float @_Z8_cl_sinhf(float %a)
  ret float %call
}


declare float @_Z9_cl_sinpif(float) local_unnamed_addr #0

define spir_func float @_Z5sinpif(float %a) local_unnamed_addr #0 {
  %call = tail call  float @_Z9_cl_sinpif(float %a)
  ret float %call
}


declare float @_Z8_cl_sqrtf(float) local_unnamed_addr #0

define spir_func float @_Z4sqrtf(float %a) local_unnamed_addr #0 {
  %call = tail call  float @_Z8_cl_sqrtf(float %a)
  ret float %call
}


declare float @_Z7_cl_tanf(float) local_unnamed_addr #0

define spir_func float @_Z3tanf(float %a) local_unnamed_addr #0 {
  %call = tail call  float @_Z7_cl_tanf(float %a)
  ret float %call
}


declare float @_Z8_cl_tanhf(float) local_unnamed_addr #0

define spir_func float @_Z4tanhf(float %a) local_unnamed_addr #0 {
  %call = tail call  float @_Z8_cl_tanhf(float %a)
  ret float %call
}


declare float @_Z9_cl_tanpif(float) local_unnamed_addr #0

define spir_func float @_Z5tanpif(float %a) local_unnamed_addr #0 {
  %call = tail call  float @_Z9_cl_tanpif(float %a)
  ret float %call
}


declare float @_Z10_cl_tgammaf(float) local_unnamed_addr #0

define spir_func float @_Z6tgammaf(float %a) local_unnamed_addr #0 {
  %call = tail call  float @_Z10_cl_tgammaf(float %a)
  ret float %call
}


declare float @_Z9_cl_truncf(float) local_unnamed_addr #0

define spir_func float @_Z5truncf(float %a) local_unnamed_addr #0 {
  %call = tail call  float @_Z9_cl_truncf(float %a)
  ret float %call
}


declare float @_Z14_cl_native_cosf(float) local_unnamed_addr #0

define spir_func float @_Z10native_cosf(float %a) local_unnamed_addr #0 {
  %call = tail call  float @_Z14_cl_native_cosf(float %a)
  ret float %call
}


declare float @_Z14_cl_native_expf(float) local_unnamed_addr #0

define spir_func float @_Z10native_expf(float %a) local_unnamed_addr #0 {
  %call = tail call  float @_Z14_cl_native_expf(float %a)
  ret float %call
}


declare float @_Z15_cl_native_exp2f(float) local_unnamed_addr #0

define spir_func float @_Z11native_exp2f(float %a) local_unnamed_addr #0 {
  %call = tail call  float @_Z15_cl_native_exp2f(float %a)
  ret float %call
}


declare float @_Z16_cl_native_exp10f(float) local_unnamed_addr #0

define spir_func float @_Z12native_exp10f(float %a) local_unnamed_addr #0 {
  %call = tail call  float @_Z16_cl_native_exp10f(float %a)
  ret float %call
}


declare float @_Z14_cl_native_logf(float) local_unnamed_addr #0

define spir_func float @_Z10native_logf(float %a) local_unnamed_addr #0 {
  %call = tail call  float @_Z14_cl_native_logf(float %a)
  ret float %call
}


declare float @_Z15_cl_native_log2f(float) local_unnamed_addr #0

define spir_func float @_Z11native_log2f(float %a) local_unnamed_addr #0 {
  %call = tail call  float @_Z15_cl_native_log2f(float %a)
  ret float %call
}


declare float @_Z16_cl_native_log10f(float) local_unnamed_addr #0

define spir_func float @_Z12native_log10f(float %a) local_unnamed_addr #0 {
  %call = tail call  float @_Z16_cl_native_log10f(float %a)
  ret float %call
}


declare float @_Z16_cl_native_recipf(float) local_unnamed_addr #0

define spir_func float @_Z12native_recipf(float %a) local_unnamed_addr #0 {
  %call = tail call  float @_Z16_cl_native_recipf(float %a)
  ret float %call
}


declare float @_Z16_cl_native_rsqrtf(float) local_unnamed_addr #0

define spir_func float @_Z12native_rsqrtf(float %a) local_unnamed_addr #0 {
  %call = tail call  float @_Z16_cl_native_rsqrtf(float %a)
  ret float %call
}


declare float @_Z14_cl_native_sinf(float) local_unnamed_addr #0

define spir_func float @_Z10native_sinf(float %a) local_unnamed_addr #0 {
  %call = tail call  float @_Z14_cl_native_sinf(float %a)
  ret float %call
}


declare float @_Z15_cl_native_sqrtf(float) local_unnamed_addr #0

define spir_func float @_Z11native_sqrtf(float %a) local_unnamed_addr #0 {
  %call = tail call  float @_Z15_cl_native_sqrtf(float %a)
  ret float %call
}


declare float @_Z14_cl_native_tanf(float) local_unnamed_addr #0

define spir_func float @_Z10native_tanf(float %a) local_unnamed_addr #0 {
  %call = tail call  float @_Z14_cl_native_tanf(float %a)
  ret float %call
}


declare float @_Z11_cl_degreesf(float) local_unnamed_addr #0

define spir_func float @_Z7degreesf(float %a) local_unnamed_addr #0 {
  %call = tail call  float @_Z11_cl_degreesf(float %a)
  ret float %call
}


declare float @_Z11_cl_radiansf(float) local_unnamed_addr #0

define spir_func float @_Z7radiansf(float %a) local_unnamed_addr #0 {
  %call = tail call  float @_Z11_cl_radiansf(float %a)
  ret float %call
}


declare float @_Z8_cl_signf(float) local_unnamed_addr #0

define spir_func float @_Z4signf(float %a) local_unnamed_addr #0 {
  %call = tail call  float @_Z8_cl_signf(float %a)
  ret float %call
}


declare float @_Z9_cl_atan2ff(float, float) local_unnamed_addr #0

define spir_func float @_Z5atan2ff(float %a, float %b) local_unnamed_addr #0 {
  %call = tail call  float @_Z9_cl_atan2ff(float %a, float %b)
  ret float %call
}


declare float @_Z11_cl_atan2piff(float, float) local_unnamed_addr #0

define spir_func float @_Z7atan2piff(float %a, float %b) local_unnamed_addr #0 {
  %call = tail call  float @_Z11_cl_atan2piff(float %a, float %b)
  ret float %call
}


declare float @_Z12_cl_copysignff(float, float) local_unnamed_addr #0

define spir_func float @_Z8copysignff(float %a, float %b) local_unnamed_addr #0 {
  %call = tail call  float @_Z12_cl_copysignff(float %a, float %b)
  ret float %call
}


declare float @_Z8_cl_fdimff(float, float) local_unnamed_addr #0

define spir_func float @_Z4fdimff(float %a, float %b) local_unnamed_addr #0 {
  %call = tail call  float @_Z8_cl_fdimff(float %a, float %b)
  ret float %call
}


declare float @_Z8_cl_fmaxff(float, float) local_unnamed_addr #0

define spir_func float @_Z4fmaxff(float %a, float %b) local_unnamed_addr #0 {
  %call = tail call  float @_Z8_cl_fmaxff(float %a, float %b)
  ret float %call
}


declare float @_Z8_cl_fminff(float, float) local_unnamed_addr #0

define spir_func float @_Z4fminff(float %a, float %b) local_unnamed_addr #0 {
  %call = tail call  float @_Z8_cl_fminff(float %a, float %b)
  ret float %call
}


declare float @_Z8_cl_fmodff(float, float) local_unnamed_addr #0

define spir_func float @_Z4fmodff(float %a, float %b) local_unnamed_addr #0 {
  %call = tail call  float @_Z8_cl_fmodff(float %a, float %b)
  ret float %call
}


declare float @_Z9_cl_hypotff(float, float) local_unnamed_addr #0

define spir_func float @_Z5hypotff(float %a, float %b) local_unnamed_addr #0 {
  %call = tail call  float @_Z9_cl_hypotff(float %a, float %b)
  ret float %call
}


declare float @_Z13_cl_nextafterff(float, float) local_unnamed_addr #0

define spir_func float @_Z9nextafterff(float %a, float %b) local_unnamed_addr #0 {
  %call = tail call  float @_Z13_cl_nextafterff(float %a, float %b)
  ret float %call
}


declare float @_Z7_cl_powff(float, float) local_unnamed_addr #0

define spir_func float @_Z3powff(float %a, float %b) local_unnamed_addr #0 {
  %call = tail call  float @_Z7_cl_powff(float %a, float %b)
  ret float %call
}


declare float @_Z8_cl_powrff(float, float) local_unnamed_addr #0

define spir_func float @_Z4powrff(float %a, float %b) local_unnamed_addr #0 {
  %call = tail call  float @_Z8_cl_powrff(float %a, float %b)
  ret float %call
}


declare float @_Z10_cl_maxmagff(float, float) local_unnamed_addr #0

define spir_func float @_Z6maxmagff(float %a, float %b) local_unnamed_addr #0 {
  %call = tail call  float @_Z10_cl_maxmagff(float %a, float %b)
  ret float %call
}


declare float @_Z10_cl_minmagff(float, float) local_unnamed_addr #0

define spir_func float @_Z6minmagff(float %a, float %b) local_unnamed_addr #0 {
  %call = tail call  float @_Z10_cl_minmagff(float %a, float %b)
  ret float %call
}


declare float @_Z13_cl_remainderff(float, float) local_unnamed_addr #0

define spir_func float @_Z9remainderff(float %a, float %b) local_unnamed_addr #0 {
  %call = tail call  float @_Z13_cl_remainderff(float %a, float %b)
  ret float %call
}


declare float @_Z17_cl_native_divideff(float, float) local_unnamed_addr #0

define spir_func float @_Z13native_divideff(float %a, float %b) local_unnamed_addr #0 {
  %call = tail call  float @_Z17_cl_native_divideff(float %a, float %b)
  ret float %call
}


declare float @_Z15_cl_native_powrff(float, float) local_unnamed_addr #0

define spir_func float @_Z11native_powrff(float %a, float %b) local_unnamed_addr #0 {
  %call = tail call  float @_Z15_cl_native_powrff(float %a, float %b)
  ret float %call
}


declare float @_Z7_cl_maxff(float, float) local_unnamed_addr #0

define spir_func float @_Z3maxff(float %a, float %b) local_unnamed_addr #0 {
  %call = tail call  float @_Z7_cl_maxff(float %a, float %b)
  ret float %call
}


declare float @_Z7_cl_minff(float, float) local_unnamed_addr #0

define spir_func float @_Z3minff(float %a, float %b) local_unnamed_addr #0 {
  %call = tail call  float @_Z7_cl_minff(float %a, float %b)
  ret float %call
}


declare float @_Z8_cl_stepff(float, float) local_unnamed_addr #0

define spir_func float @_Z4stepff(float %a, float %b) local_unnamed_addr #0 {
  %call = tail call  float @_Z8_cl_stepff(float %a, float %b)
  ret float %call
}


declare float @_Z9_cl_fractfPU8CLglobalf(float, float *) local_unnamed_addr #0

define spir_func float @_Z5fractfPU3AS1f(float %a, float addrspace(1)* %b) local_unnamed_addr #0 {
  %1 = addrspacecast float addrspace(1)* %b to float *
  %call = tail call  float @_Z9_cl_fractfPU8CLglobalf(float %a, float * %1)
  ret float %call
}


declare float @_Z9_cl_fractfPU7CLlocalf(float, float *) local_unnamed_addr #0

define spir_func float @_Z5fractfPU3AS3f(float %a, float addrspace(3)* %b) local_unnamed_addr #0 {
  %1 = addrspacecast float addrspace(3)* %b to float *
  %call = tail call  float @_Z9_cl_fractfPU7CLlocalf(float %a, float * %1)
  ret float %call
}


declare float @_Z9_cl_fractfPU9CLprivatef(float, float *) local_unnamed_addr #0

define spir_func float @_Z5fractfPf(float %a, float * %b) local_unnamed_addr #0 {
  %call = tail call  float @_Z9_cl_fractfPU9CLprivatef(float %a, float * %b)
  ret float %call
}


declare float @_Z10_cl_sincosfPU8CLglobalf(float, float *) local_unnamed_addr #0

define spir_func float @_Z6sincosfPU3AS1f(float %a, float addrspace(1)* %b) local_unnamed_addr #0 {
  %1 = addrspacecast float addrspace(1)* %b to float *
  %call = tail call  float @_Z10_cl_sincosfPU8CLglobalf(float %a, float * %1)
  ret float %call
}


declare float @_Z10_cl_sincosfPU7CLlocalf(float, float *) local_unnamed_addr #0

define spir_func float @_Z6sincosfPU3AS3f(float %a, float addrspace(3)* %b) local_unnamed_addr #0 {
  %1 = addrspacecast float addrspace(3)* %b to float *
  %call = tail call  float @_Z10_cl_sincosfPU7CLlocalf(float %a, float * %1)
  ret float %call
}


declare float @_Z10_cl_sincosfPU9CLprivatef(float, float *) local_unnamed_addr #0

define spir_func float @_Z6sincosfPf(float %a, float * %b) local_unnamed_addr #0 {
  %call = tail call  float @_Z10_cl_sincosfPU9CLprivatef(float %a, float * %b)
  ret float %call
}


declare float @_Z8_cl_modffPU8CLglobalf(float, float *) local_unnamed_addr #0

define spir_func float @_Z4modffPU3AS1f(float %a, float addrspace(1)* %b) local_unnamed_addr #0 {
  %1 = addrspacecast float addrspace(1)* %b to float *
  %call = tail call  float @_Z8_cl_modffPU8CLglobalf(float %a, float * %1)
  ret float %call
}


declare float @_Z8_cl_modffPU7CLlocalf(float, float *) local_unnamed_addr #0

define spir_func float @_Z4modffPU3AS3f(float %a, float addrspace(3)* %b) local_unnamed_addr #0 {
  %1 = addrspacecast float addrspace(3)* %b to float *
  %call = tail call  float @_Z8_cl_modffPU7CLlocalf(float %a, float * %1)
  ret float %call
}


declare float @_Z8_cl_modffPU9CLprivatef(float, float *) local_unnamed_addr #0

define spir_func float @_Z4modffPf(float %a, float * %b) local_unnamed_addr #0 {
  %call = tail call  float @_Z8_cl_modffPU9CLprivatef(float %a, float * %b)
  ret float %call
}


declare float @_Z7_cl_fmafff(float, float, float) local_unnamed_addr #0

define spir_func float @_Z3fmafff(float %a, float %b, float %c) local_unnamed_addr #0 {
  %call = tail call  float @_Z7_cl_fmafff(float %a, float %b, float %c)
  ret float %call
}


declare float @_Z7_cl_madfff(float, float, float) local_unnamed_addr #0

define spir_func float @_Z3madfff(float %a, float %b, float %c) local_unnamed_addr #0 {
  %call = tail call  float @_Z7_cl_madfff(float %a, float %b, float %c)
  ret float %call
}


declare float @_Z9_cl_clampfff(float, float, float) local_unnamed_addr #0

define spir_func float @_Z5clampfff(float %a, float %b, float %c) local_unnamed_addr #0 {
  %call = tail call  float @_Z9_cl_clampfff(float %a, float %b, float %c)
  ret float %call
}


declare float @_Z7_cl_mixfff(float, float, float) local_unnamed_addr #0

define spir_func float @_Z3mixfff(float %a, float %b, float %c) local_unnamed_addr #0 {
  %call = tail call  float @_Z7_cl_mixfff(float %a, float %b, float %c)
  ret float %call
}


declare float @_Z14_cl_smoothstepfff(float, float, float) local_unnamed_addr #0

define spir_func float @_Z10smoothstepfff(float %a, float %b, float %c) local_unnamed_addr #0 {
  %call = tail call  float @_Z14_cl_smoothstepfff(float %a, float %b, float %c)
  ret float %call
}


declare i32 @_Z9_cl_ilogbf(float) local_unnamed_addr #0

define spir_func i32 @_Z5ilogbf(float %a) local_unnamed_addr #0 {
  %call = tail call  i32 @_Z9_cl_ilogbf(float %a)
  ret i32 %call
}


declare float @_Z9_cl_ldexpfi(float, i32) local_unnamed_addr #0

define spir_func float @_Z5ldexpfi(float %a, i32 %b) local_unnamed_addr #0 {
  %call = tail call  float @_Z9_cl_ldexpfi(float %a, i32 %b)
  ret float %call
}


declare float @_Z8_cl_pownfi(float, i32) local_unnamed_addr #0

define spir_func float @_Z4pownfi(float %a, i32 %b) local_unnamed_addr #0 {
  %call = tail call  float @_Z8_cl_pownfi(float %a, i32 %b)
  ret float %call
}


declare float @_Z9_cl_rootnfi(float, i32) local_unnamed_addr #0

define spir_func float @_Z5rootnfi(float %a, i32 %b) local_unnamed_addr #0 {
  %call = tail call  float @_Z9_cl_rootnfi(float %a, i32 %b)
  ret float %call
}


declare float @_Z10_cl_remquoffPU8CLglobali(float, float, i32 *) local_unnamed_addr #0

define spir_func float @_Z6remquoffPU3AS1i(float %a, float %b, i32 addrspace(1)* %c) local_unnamed_addr #0 {
  %1 = addrspacecast i32 addrspace(1)* %c to i32 *
  %call = tail call  float @_Z10_cl_remquoffPU8CLglobali(float %a, float %b, i32 * %1)
  ret float %call
}


declare float @_Z10_cl_remquoffPU7CLlocali(float, float, i32 *) local_unnamed_addr #0

define spir_func float @_Z6remquoffPU3AS3i(float %a, float %b, i32 addrspace(3)* %c) local_unnamed_addr #0 {
  %1 = addrspacecast i32 addrspace(3)* %c to i32 *
  %call = tail call  float @_Z10_cl_remquoffPU7CLlocali(float %a, float %b, i32 * %1)
  ret float %call
}


declare float @_Z10_cl_remquoffPU9CLprivatei(float, float, i32 *) local_unnamed_addr #0

define spir_func float @_Z6remquoffPi(float %a, float %b, i32 * %c) local_unnamed_addr #0 {
  %call = tail call  float @_Z10_cl_remquoffPU9CLprivatei(float %a, float %b, i32 * %c)
  ret float %call
}


declare float @_Z7_cl_nanj(i32) local_unnamed_addr #0

define spir_func float @_Z3nanj(i32 %a) local_unnamed_addr #0 {
  %call = tail call  float @_Z7_cl_nanj(i32 %a)
  ret float %call
}


declare float @_Z12_cl_lgamma_rfPU8CLglobali(float, i32 *) local_unnamed_addr #0

define spir_func float @_Z8lgamma_rfPU3AS1i(float %a, i32 addrspace(1)* %b) local_unnamed_addr #0 {
  %1 = addrspacecast i32 addrspace(1)* %b to i32 *
  %call = tail call  float @_Z12_cl_lgamma_rfPU8CLglobali(float %a, i32 * %1)
  ret float %call
}


declare float @_Z12_cl_lgamma_rfPU7CLlocali(float, i32 *) local_unnamed_addr #0

define spir_func float @_Z8lgamma_rfPU3AS3i(float %a, i32 addrspace(3)* %b) local_unnamed_addr #0 {
  %1 = addrspacecast i32 addrspace(3)* %b to i32 *
  %call = tail call  float @_Z12_cl_lgamma_rfPU7CLlocali(float %a, i32 * %1)
  ret float %call
}


declare float @_Z12_cl_lgamma_rfPU9CLprivatei(float, i32 *) local_unnamed_addr #0

define spir_func float @_Z8lgamma_rfPi(float %a, i32 * %b) local_unnamed_addr #0 {
  %call = tail call  float @_Z12_cl_lgamma_rfPU9CLprivatei(float %a, i32 * %b)
  ret float %call
}


declare float @_Z9_cl_frexpfPU8CLglobali(float, i32 *) local_unnamed_addr #0

define spir_func float @_Z5frexpfPU3AS1i(float %a, i32 addrspace(1)* %b) local_unnamed_addr #0 {
  %1 = addrspacecast i32 addrspace(1)* %b to i32 *
  %call = tail call  float @_Z9_cl_frexpfPU8CLglobali(float %a, i32 * %1)
  ret float %call
}


declare float @_Z9_cl_frexpfPU7CLlocali(float, i32 *) local_unnamed_addr #0

define spir_func float @_Z5frexpfPU3AS3i(float %a, i32 addrspace(3)* %b) local_unnamed_addr #0 {
  %1 = addrspacecast i32 addrspace(3)* %b to i32 *
  %call = tail call  float @_Z9_cl_frexpfPU7CLlocali(float %a, i32 * %1)
  ret float %call
}


declare float @_Z9_cl_frexpfPU9CLprivatei(float, i32 *) local_unnamed_addr #0

define spir_func float @_Z5frexpfPi(float %a, i32 * %b) local_unnamed_addr #0 {
  %call = tail call  float @_Z9_cl_frexpfPU9CLprivatei(float %a, i32 * %b)
  ret float %call
}


declare double @_Z8_cl_acosd(double) local_unnamed_addr #0

define spir_func double @_Z4acosd(double %a) local_unnamed_addr #0 {
  %call = tail call  double @_Z8_cl_acosd(double %a)
  ret double %call
}


declare double @_Z9_cl_acoshd(double) local_unnamed_addr #0

define spir_func double @_Z5acoshd(double %a) local_unnamed_addr #0 {
  %call = tail call  double @_Z9_cl_acoshd(double %a)
  ret double %call
}


declare double @_Z10_cl_acospid(double) local_unnamed_addr #0

define spir_func double @_Z6acospid(double %a) local_unnamed_addr #0 {
  %call = tail call  double @_Z10_cl_acospid(double %a)
  ret double %call
}


declare double @_Z8_cl_asind(double) local_unnamed_addr #0

define spir_func double @_Z4asind(double %a) local_unnamed_addr #0 {
  %call = tail call  double @_Z8_cl_asind(double %a)
  ret double %call
}


declare double @_Z9_cl_asinhd(double) local_unnamed_addr #0

define spir_func double @_Z5asinhd(double %a) local_unnamed_addr #0 {
  %call = tail call  double @_Z9_cl_asinhd(double %a)
  ret double %call
}


declare double @_Z10_cl_asinpid(double) local_unnamed_addr #0

define spir_func double @_Z6asinpid(double %a) local_unnamed_addr #0 {
  %call = tail call  double @_Z10_cl_asinpid(double %a)
  ret double %call
}


declare double @_Z8_cl_atand(double) local_unnamed_addr #0

define spir_func double @_Z4atand(double %a) local_unnamed_addr #0 {
  %call = tail call  double @_Z8_cl_atand(double %a)
  ret double %call
}


declare double @_Z9_cl_atanhd(double) local_unnamed_addr #0

define spir_func double @_Z5atanhd(double %a) local_unnamed_addr #0 {
  %call = tail call  double @_Z9_cl_atanhd(double %a)
  ret double %call
}


declare double @_Z10_cl_atanpid(double) local_unnamed_addr #0

define spir_func double @_Z6atanpid(double %a) local_unnamed_addr #0 {
  %call = tail call  double @_Z10_cl_atanpid(double %a)
  ret double %call
}


declare double @_Z8_cl_cbrtd(double) local_unnamed_addr #0

define spir_func double @_Z4cbrtd(double %a) local_unnamed_addr #0 {
  %call = tail call  double @_Z8_cl_cbrtd(double %a)
  ret double %call
}


declare double @_Z8_cl_ceild(double) local_unnamed_addr #0

define spir_func double @_Z4ceild(double %a) local_unnamed_addr #0 {
  %call = tail call  double @_Z8_cl_ceild(double %a)
  ret double %call
}


declare double @_Z7_cl_cosd(double) local_unnamed_addr #0

define spir_func double @_Z3cosd(double %a) local_unnamed_addr #0 {
  %call = tail call  double @_Z7_cl_cosd(double %a)
  ret double %call
}


declare double @_Z8_cl_coshd(double) local_unnamed_addr #0

define spir_func double @_Z4coshd(double %a) local_unnamed_addr #0 {
  %call = tail call  double @_Z8_cl_coshd(double %a)
  ret double %call
}


declare double @_Z9_cl_cospid(double) local_unnamed_addr #0

define spir_func double @_Z5cospid(double %a) local_unnamed_addr #0 {
  %call = tail call  double @_Z9_cl_cospid(double %a)
  ret double %call
}


declare double @_Z8_cl_erfcd(double) local_unnamed_addr #0

define spir_func double @_Z4erfcd(double %a) local_unnamed_addr #0 {
  %call = tail call  double @_Z8_cl_erfcd(double %a)
  ret double %call
}


declare double @_Z7_cl_erfd(double) local_unnamed_addr #0

define spir_func double @_Z3erfd(double %a) local_unnamed_addr #0 {
  %call = tail call  double @_Z7_cl_erfd(double %a)
  ret double %call
}


declare double @_Z7_cl_expd(double) local_unnamed_addr #0

define spir_func double @_Z3expd(double %a) local_unnamed_addr #0 {
  %call = tail call  double @_Z7_cl_expd(double %a)
  ret double %call
}


declare double @_Z8_cl_exp2d(double) local_unnamed_addr #0

define spir_func double @_Z4exp2d(double %a) local_unnamed_addr #0 {
  %call = tail call  double @_Z8_cl_exp2d(double %a)
  ret double %call
}


declare double @_Z9_cl_exp10d(double) local_unnamed_addr #0

define spir_func double @_Z5exp10d(double %a) local_unnamed_addr #0 {
  %call = tail call  double @_Z9_cl_exp10d(double %a)
  ret double %call
}


declare double @_Z9_cl_expm1d(double) local_unnamed_addr #0

define spir_func double @_Z5expm1d(double %a) local_unnamed_addr #0 {
  %call = tail call  double @_Z9_cl_expm1d(double %a)
  ret double %call
}


declare double @_Z8_cl_fabsd(double) local_unnamed_addr #0

define spir_func double @_Z4fabsd(double %a) local_unnamed_addr #0 {
  %call = tail call  double @_Z8_cl_fabsd(double %a)
  ret double %call
}


declare double @_Z9_cl_floord(double) local_unnamed_addr #0

define spir_func double @_Z5floord(double %a) local_unnamed_addr #0 {
  %call = tail call  double @_Z9_cl_floord(double %a)
  ret double %call
}


declare double @_Z10_cl_lgammad(double) local_unnamed_addr #0

define spir_func double @_Z6lgammad(double %a) local_unnamed_addr #0 {
  %call = tail call  double @_Z10_cl_lgammad(double %a)
  ret double %call
}


declare double @_Z7_cl_logd(double) local_unnamed_addr #0

define spir_func double @_Z3logd(double %a) local_unnamed_addr #0 {
  %call = tail call  double @_Z7_cl_logd(double %a)
  ret double %call
}


declare double @_Z9_cl_log10d(double) local_unnamed_addr #0

define spir_func double @_Z5log10d(double %a) local_unnamed_addr #0 {
  %call = tail call  double @_Z9_cl_log10d(double %a)
  ret double %call
}


declare double @_Z8_cl_log2d(double) local_unnamed_addr #0

define spir_func double @_Z4log2d(double %a) local_unnamed_addr #0 {
  %call = tail call  double @_Z8_cl_log2d(double %a)
  ret double %call
}


declare double @_Z9_cl_log1pd(double) local_unnamed_addr #0

define spir_func double @_Z5log1pd(double %a) local_unnamed_addr #0 {
  %call = tail call  double @_Z9_cl_log1pd(double %a)
  ret double %call
}


declare double @_Z8_cl_rintd(double) local_unnamed_addr #0

define spir_func double @_Z4rintd(double %a) local_unnamed_addr #0 {
  %call = tail call  double @_Z8_cl_rintd(double %a)
  ret double %call
}


declare double @_Z9_cl_roundd(double) local_unnamed_addr #0

define spir_func double @_Z5roundd(double %a) local_unnamed_addr #0 {
  %call = tail call  double @_Z9_cl_roundd(double %a)
  ret double %call
}


declare double @_Z9_cl_rsqrtd(double) local_unnamed_addr #0

define spir_func double @_Z5rsqrtd(double %a) local_unnamed_addr #0 {
  %call = tail call  double @_Z9_cl_rsqrtd(double %a)
  ret double %call
}


declare double @_Z7_cl_sind(double) local_unnamed_addr #0

define spir_func double @_Z3sind(double %a) local_unnamed_addr #0 {
  %call = tail call  double @_Z7_cl_sind(double %a)
  ret double %call
}


declare double @_Z8_cl_sinhd(double) local_unnamed_addr #0

define spir_func double @_Z4sinhd(double %a) local_unnamed_addr #0 {
  %call = tail call  double @_Z8_cl_sinhd(double %a)
  ret double %call
}


declare double @_Z9_cl_sinpid(double) local_unnamed_addr #0

define spir_func double @_Z5sinpid(double %a) local_unnamed_addr #0 {
  %call = tail call  double @_Z9_cl_sinpid(double %a)
  ret double %call
}


declare double @_Z8_cl_sqrtd(double) local_unnamed_addr #0

define spir_func double @_Z4sqrtd(double %a) local_unnamed_addr #0 {
  %call = tail call  double @_Z8_cl_sqrtd(double %a)
  ret double %call
}


declare double @_Z7_cl_tand(double) local_unnamed_addr #0

define spir_func double @_Z3tand(double %a) local_unnamed_addr #0 {
  %call = tail call  double @_Z7_cl_tand(double %a)
  ret double %call
}


declare double @_Z8_cl_tanhd(double) local_unnamed_addr #0

define spir_func double @_Z4tanhd(double %a) local_unnamed_addr #0 {
  %call = tail call  double @_Z8_cl_tanhd(double %a)
  ret double %call
}


declare double @_Z9_cl_tanpid(double) local_unnamed_addr #0

define spir_func double @_Z5tanpid(double %a) local_unnamed_addr #0 {
  %call = tail call  double @_Z9_cl_tanpid(double %a)
  ret double %call
}


declare double @_Z10_cl_tgammad(double) local_unnamed_addr #0

define spir_func double @_Z6tgammad(double %a) local_unnamed_addr #0 {
  %call = tail call  double @_Z10_cl_tgammad(double %a)
  ret double %call
}


declare double @_Z9_cl_truncd(double) local_unnamed_addr #0

define spir_func double @_Z5truncd(double %a) local_unnamed_addr #0 {
  %call = tail call  double @_Z9_cl_truncd(double %a)
  ret double %call
}


declare double @_Z14_cl_native_cosd(double) local_unnamed_addr #0

define spir_func double @_Z10native_cosd(double %a) local_unnamed_addr #0 {
  %call = tail call  double @_Z14_cl_native_cosd(double %a)
  ret double %call
}


declare double @_Z14_cl_native_expd(double) local_unnamed_addr #0

define spir_func double @_Z10native_expd(double %a) local_unnamed_addr #0 {
  %call = tail call  double @_Z14_cl_native_expd(double %a)
  ret double %call
}


declare double @_Z15_cl_native_exp2d(double) local_unnamed_addr #0

define spir_func double @_Z11native_exp2d(double %a) local_unnamed_addr #0 {
  %call = tail call  double @_Z15_cl_native_exp2d(double %a)
  ret double %call
}


declare double @_Z16_cl_native_exp10d(double) local_unnamed_addr #0

define spir_func double @_Z12native_exp10d(double %a) local_unnamed_addr #0 {
  %call = tail call  double @_Z16_cl_native_exp10d(double %a)
  ret double %call
}


declare double @_Z14_cl_native_logd(double) local_unnamed_addr #0

define spir_func double @_Z10native_logd(double %a) local_unnamed_addr #0 {
  %call = tail call  double @_Z14_cl_native_logd(double %a)
  ret double %call
}


declare double @_Z15_cl_native_log2d(double) local_unnamed_addr #0

define spir_func double @_Z11native_log2d(double %a) local_unnamed_addr #0 {
  %call = tail call  double @_Z15_cl_native_log2d(double %a)
  ret double %call
}


declare double @_Z16_cl_native_log10d(double) local_unnamed_addr #0

define spir_func double @_Z12native_log10d(double %a) local_unnamed_addr #0 {
  %call = tail call  double @_Z16_cl_native_log10d(double %a)
  ret double %call
}


declare double @_Z16_cl_native_recipd(double) local_unnamed_addr #0

define spir_func double @_Z12native_recipd(double %a) local_unnamed_addr #0 {
  %call = tail call  double @_Z16_cl_native_recipd(double %a)
  ret double %call
}


declare double @_Z16_cl_native_rsqrtd(double) local_unnamed_addr #0

define spir_func double @_Z12native_rsqrtd(double %a) local_unnamed_addr #0 {
  %call = tail call  double @_Z16_cl_native_rsqrtd(double %a)
  ret double %call
}


declare double @_Z14_cl_native_sind(double) local_unnamed_addr #0

define spir_func double @_Z10native_sind(double %a) local_unnamed_addr #0 {
  %call = tail call  double @_Z14_cl_native_sind(double %a)
  ret double %call
}


declare double @_Z15_cl_native_sqrtd(double) local_unnamed_addr #0

define spir_func double @_Z11native_sqrtd(double %a) local_unnamed_addr #0 {
  %call = tail call  double @_Z15_cl_native_sqrtd(double %a)
  ret double %call
}


declare double @_Z14_cl_native_tand(double) local_unnamed_addr #0

define spir_func double @_Z10native_tand(double %a) local_unnamed_addr #0 {
  %call = tail call  double @_Z14_cl_native_tand(double %a)
  ret double %call
}


declare double @_Z11_cl_degreesd(double) local_unnamed_addr #0

define spir_func double @_Z7degreesd(double %a) local_unnamed_addr #0 {
  %call = tail call  double @_Z11_cl_degreesd(double %a)
  ret double %call
}


declare double @_Z11_cl_radiansd(double) local_unnamed_addr #0

define spir_func double @_Z7radiansd(double %a) local_unnamed_addr #0 {
  %call = tail call  double @_Z11_cl_radiansd(double %a)
  ret double %call
}


declare double @_Z8_cl_signd(double) local_unnamed_addr #0

define spir_func double @_Z4signd(double %a) local_unnamed_addr #0 {
  %call = tail call  double @_Z8_cl_signd(double %a)
  ret double %call
}


declare double @_Z9_cl_atan2dd(double, double) local_unnamed_addr #0

define spir_func double @_Z5atan2dd(double %a, double %b) local_unnamed_addr #0 {
  %call = tail call  double @_Z9_cl_atan2dd(double %a, double %b)
  ret double %call
}


declare double @_Z11_cl_atan2pidd(double, double) local_unnamed_addr #0

define spir_func double @_Z7atan2pidd(double %a, double %b) local_unnamed_addr #0 {
  %call = tail call  double @_Z11_cl_atan2pidd(double %a, double %b)
  ret double %call
}


declare double @_Z12_cl_copysigndd(double, double) local_unnamed_addr #0

define spir_func double @_Z8copysigndd(double %a, double %b) local_unnamed_addr #0 {
  %call = tail call  double @_Z12_cl_copysigndd(double %a, double %b)
  ret double %call
}


declare double @_Z8_cl_fdimdd(double, double) local_unnamed_addr #0

define spir_func double @_Z4fdimdd(double %a, double %b) local_unnamed_addr #0 {
  %call = tail call  double @_Z8_cl_fdimdd(double %a, double %b)
  ret double %call
}


declare double @_Z8_cl_fmaxdd(double, double) local_unnamed_addr #0

define spir_func double @_Z4fmaxdd(double %a, double %b) local_unnamed_addr #0 {
  %call = tail call  double @_Z8_cl_fmaxdd(double %a, double %b)
  ret double %call
}


declare double @_Z8_cl_fmindd(double, double) local_unnamed_addr #0

define spir_func double @_Z4fmindd(double %a, double %b) local_unnamed_addr #0 {
  %call = tail call  double @_Z8_cl_fmindd(double %a, double %b)
  ret double %call
}


declare double @_Z8_cl_fmoddd(double, double) local_unnamed_addr #0

define spir_func double @_Z4fmoddd(double %a, double %b) local_unnamed_addr #0 {
  %call = tail call  double @_Z8_cl_fmoddd(double %a, double %b)
  ret double %call
}


declare double @_Z9_cl_hypotdd(double, double) local_unnamed_addr #0

define spir_func double @_Z5hypotdd(double %a, double %b) local_unnamed_addr #0 {
  %call = tail call  double @_Z9_cl_hypotdd(double %a, double %b)
  ret double %call
}


declare double @_Z13_cl_nextafterdd(double, double) local_unnamed_addr #0

define spir_func double @_Z9nextafterdd(double %a, double %b) local_unnamed_addr #0 {
  %call = tail call  double @_Z13_cl_nextafterdd(double %a, double %b)
  ret double %call
}


declare double @_Z7_cl_powdd(double, double) local_unnamed_addr #0

define spir_func double @_Z3powdd(double %a, double %b) local_unnamed_addr #0 {
  %call = tail call  double @_Z7_cl_powdd(double %a, double %b)
  ret double %call
}


declare double @_Z8_cl_powrdd(double, double) local_unnamed_addr #0

define spir_func double @_Z4powrdd(double %a, double %b) local_unnamed_addr #0 {
  %call = tail call  double @_Z8_cl_powrdd(double %a, double %b)
  ret double %call
}


declare double @_Z10_cl_maxmagdd(double, double) local_unnamed_addr #0

define spir_func double @_Z6maxmagdd(double %a, double %b) local_unnamed_addr #0 {
  %call = tail call  double @_Z10_cl_maxmagdd(double %a, double %b)
  ret double %call
}


declare double @_Z10_cl_minmagdd(double, double) local_unnamed_addr #0

define spir_func double @_Z6minmagdd(double %a, double %b) local_unnamed_addr #0 {
  %call = tail call  double @_Z10_cl_minmagdd(double %a, double %b)
  ret double %call
}


declare double @_Z13_cl_remainderdd(double, double) local_unnamed_addr #0

define spir_func double @_Z9remainderdd(double %a, double %b) local_unnamed_addr #0 {
  %call = tail call  double @_Z13_cl_remainderdd(double %a, double %b)
  ret double %call
}


declare double @_Z17_cl_native_dividedd(double, double) local_unnamed_addr #0

define spir_func double @_Z13native_dividedd(double %a, double %b) local_unnamed_addr #0 {
  %call = tail call  double @_Z17_cl_native_dividedd(double %a, double %b)
  ret double %call
}


declare double @_Z15_cl_native_powrdd(double, double) local_unnamed_addr #0

define spir_func double @_Z11native_powrdd(double %a, double %b) local_unnamed_addr #0 {
  %call = tail call  double @_Z15_cl_native_powrdd(double %a, double %b)
  ret double %call
}


declare double @_Z7_cl_maxdd(double, double) local_unnamed_addr #0

define spir_func double @_Z3maxdd(double %a, double %b) local_unnamed_addr #0 {
  %call = tail call  double @_Z7_cl_maxdd(double %a, double %b)
  ret double %call
}


declare double @_Z7_cl_mindd(double, double) local_unnamed_addr #0

define spir_func double @_Z3mindd(double %a, double %b) local_unnamed_addr #0 {
  %call = tail call  double @_Z7_cl_mindd(double %a, double %b)
  ret double %call
}


declare double @_Z8_cl_stepdd(double, double) local_unnamed_addr #0

define spir_func double @_Z4stepdd(double %a, double %b) local_unnamed_addr #0 {
  %call = tail call  double @_Z8_cl_stepdd(double %a, double %b)
  ret double %call
}


declare double @_Z9_cl_fractdPU8CLglobald(double, double *) local_unnamed_addr #0

define spir_func double @_Z5fractdPU3AS1d(double %a, double addrspace(1)* %b) local_unnamed_addr #0 {
  %1 = addrspacecast double addrspace(1)* %b to double *
  %call = tail call  double @_Z9_cl_fractdPU8CLglobald(double %a, double * %1)
  ret double %call
}


declare double @_Z9_cl_fractdPU7CLlocald(double, double *) local_unnamed_addr #0

define spir_func double @_Z5fractdPU3AS3d(double %a, double addrspace(3)* %b) local_unnamed_addr #0 {
  %1 = addrspacecast double addrspace(3)* %b to double *
  %call = tail call  double @_Z9_cl_fractdPU7CLlocald(double %a, double * %1)
  ret double %call
}


declare double @_Z9_cl_fractdPU9CLprivated(double, double *) local_unnamed_addr #0

define spir_func double @_Z5fractdPd(double %a, double * %b) local_unnamed_addr #0 {
  %call = tail call  double @_Z9_cl_fractdPU9CLprivated(double %a, double * %b)
  ret double %call
}


declare double @_Z10_cl_sincosdPU8CLglobald(double, double *) local_unnamed_addr #0

define spir_func double @_Z6sincosdPU3AS1d(double %a, double addrspace(1)* %b) local_unnamed_addr #0 {
  %1 = addrspacecast double addrspace(1)* %b to double *
  %call = tail call  double @_Z10_cl_sincosdPU8CLglobald(double %a, double * %1)
  ret double %call
}


declare double @_Z10_cl_sincosdPU7CLlocald(double, double *) local_unnamed_addr #0

define spir_func double @_Z6sincosdPU3AS3d(double %a, double addrspace(3)* %b) local_unnamed_addr #0 {
  %1 = addrspacecast double addrspace(3)* %b to double *
  %call = tail call  double @_Z10_cl_sincosdPU7CLlocald(double %a, double * %1)
  ret double %call
}


declare double @_Z10_cl_sincosdPU9CLprivated(double, double *) local_unnamed_addr #0

define spir_func double @_Z6sincosdPd(double %a, double * %b) local_unnamed_addr #0 {
  %call = tail call  double @_Z10_cl_sincosdPU9CLprivated(double %a, double * %b)
  ret double %call
}


declare double @_Z8_cl_modfdPU8CLglobald(double, double *) local_unnamed_addr #0

define spir_func double @_Z4modfdPU3AS1d(double %a, double addrspace(1)* %b) local_unnamed_addr #0 {
  %1 = addrspacecast double addrspace(1)* %b to double *
  %call = tail call  double @_Z8_cl_modfdPU8CLglobald(double %a, double * %1)
  ret double %call
}


declare double @_Z8_cl_modfdPU7CLlocald(double, double *) local_unnamed_addr #0

define spir_func double @_Z4modfdPU3AS3d(double %a, double addrspace(3)* %b) local_unnamed_addr #0 {
  %1 = addrspacecast double addrspace(3)* %b to double *
  %call = tail call  double @_Z8_cl_modfdPU7CLlocald(double %a, double * %1)
  ret double %call
}


declare double @_Z8_cl_modfdPU9CLprivated(double, double *) local_unnamed_addr #0

define spir_func double @_Z4modfdPd(double %a, double * %b) local_unnamed_addr #0 {
  %call = tail call  double @_Z8_cl_modfdPU9CLprivated(double %a, double * %b)
  ret double %call
}


declare double @_Z7_cl_fmaddd(double, double, double) local_unnamed_addr #0

define spir_func double @_Z3fmaddd(double %a, double %b, double %c) local_unnamed_addr #0 {
  %call = tail call  double @_Z7_cl_fmaddd(double %a, double %b, double %c)
  ret double %call
}


declare double @_Z7_cl_madddd(double, double, double) local_unnamed_addr #0

define spir_func double @_Z3madddd(double %a, double %b, double %c) local_unnamed_addr #0 {
  %call = tail call  double @_Z7_cl_madddd(double %a, double %b, double %c)
  ret double %call
}


declare double @_Z9_cl_clampddd(double, double, double) local_unnamed_addr #0

define spir_func double @_Z5clampddd(double %a, double %b, double %c) local_unnamed_addr #0 {
  %call = tail call  double @_Z9_cl_clampddd(double %a, double %b, double %c)
  ret double %call
}


declare double @_Z7_cl_mixddd(double, double, double) local_unnamed_addr #0

define spir_func double @_Z3mixddd(double %a, double %b, double %c) local_unnamed_addr #0 {
  %call = tail call  double @_Z7_cl_mixddd(double %a, double %b, double %c)
  ret double %call
}


declare double @_Z14_cl_smoothstepddd(double, double, double) local_unnamed_addr #0

define spir_func double @_Z10smoothstepddd(double %a, double %b, double %c) local_unnamed_addr #0 {
  %call = tail call  double @_Z14_cl_smoothstepddd(double %a, double %b, double %c)
  ret double %call
}


declare i32 @_Z9_cl_ilogbd(double) local_unnamed_addr #0

define spir_func i32 @_Z5ilogbd(double %a) local_unnamed_addr #0 {
  %call = tail call  i32 @_Z9_cl_ilogbd(double %a)
  ret i32 %call
}


declare double @_Z9_cl_ldexpdi(double, i32) local_unnamed_addr #0

define spir_func double @_Z5ldexpdi(double %a, i32 %b) local_unnamed_addr #0 {
  %call = tail call  double @_Z9_cl_ldexpdi(double %a, i32 %b)
  ret double %call
}


declare double @_Z8_cl_powndi(double, i32) local_unnamed_addr #0

define spir_func double @_Z4powndi(double %a, i32 %b) local_unnamed_addr #0 {
  %call = tail call  double @_Z8_cl_powndi(double %a, i32 %b)
  ret double %call
}


declare double @_Z9_cl_rootndi(double, i32) local_unnamed_addr #0

define spir_func double @_Z5rootndi(double %a, i32 %b) local_unnamed_addr #0 {
  %call = tail call  double @_Z9_cl_rootndi(double %a, i32 %b)
  ret double %call
}


declare double @_Z10_cl_remquoddPU8CLglobali(double, double, i32 *) local_unnamed_addr #0

define spir_func double @_Z6remquoddPU3AS1i(double %a, double %b, i32 addrspace(1)* %c) local_unnamed_addr #0 {
  %1 = addrspacecast i32 addrspace(1)* %c to i32 *
  %call = tail call  double @_Z10_cl_remquoddPU8CLglobali(double %a, double %b, i32 * %1)
  ret double %call
}


declare double @_Z10_cl_remquoddPU7CLlocali(double, double, i32 *) local_unnamed_addr #0

define spir_func double @_Z6remquoddPU3AS3i(double %a, double %b, i32 addrspace(3)* %c) local_unnamed_addr #0 {
  %1 = addrspacecast i32 addrspace(3)* %c to i32 *
  %call = tail call  double @_Z10_cl_remquoddPU7CLlocali(double %a, double %b, i32 * %1)
  ret double %call
}


declare double @_Z10_cl_remquoddPU9CLprivatei(double, double, i32 *) local_unnamed_addr #0

define spir_func double @_Z6remquoddPi(double %a, double %b, i32 * %c) local_unnamed_addr #0 {
  %call = tail call  double @_Z10_cl_remquoddPU9CLprivatei(double %a, double %b, i32 * %c)
  ret double %call
}


declare double @_Z7_cl_nanm(i64) local_unnamed_addr #0

define spir_func double @_Z3nanm(i64 %a) local_unnamed_addr #0 {
  %call = tail call  double @_Z7_cl_nanm(i64 %a)
  ret double %call
}


declare double @_Z12_cl_lgamma_rdPU8CLglobali(double, i32 *) local_unnamed_addr #0

define spir_func double @_Z8lgamma_rdPU3AS1i(double %a, i32 addrspace(1)* %b) local_unnamed_addr #0 {
  %1 = addrspacecast i32 addrspace(1)* %b to i32 *
  %call = tail call  double @_Z12_cl_lgamma_rdPU8CLglobali(double %a, i32 * %1)
  ret double %call
}


declare double @_Z12_cl_lgamma_rdPU7CLlocali(double, i32 *) local_unnamed_addr #0

define spir_func double @_Z8lgamma_rdPU3AS3i(double %a, i32 addrspace(3)* %b) local_unnamed_addr #0 {
  %1 = addrspacecast i32 addrspace(3)* %b to i32 *
  %call = tail call  double @_Z12_cl_lgamma_rdPU7CLlocali(double %a, i32 * %1)
  ret double %call
}


declare double @_Z12_cl_lgamma_rdPU9CLprivatei(double, i32 *) local_unnamed_addr #0

define spir_func double @_Z8lgamma_rdPi(double %a, i32 * %b) local_unnamed_addr #0 {
  %call = tail call  double @_Z12_cl_lgamma_rdPU9CLprivatei(double %a, i32 * %b)
  ret double %call
}


declare double @_Z9_cl_frexpdPU8CLglobali(double, i32 *) local_unnamed_addr #0

define spir_func double @_Z5frexpdPU3AS1i(double %a, i32 addrspace(1)* %b) local_unnamed_addr #0 {
  %1 = addrspacecast i32 addrspace(1)* %b to i32 *
  %call = tail call  double @_Z9_cl_frexpdPU8CLglobali(double %a, i32 * %1)
  ret double %call
}


declare double @_Z9_cl_frexpdPU7CLlocali(double, i32 *) local_unnamed_addr #0

define spir_func double @_Z5frexpdPU3AS3i(double %a, i32 addrspace(3)* %b) local_unnamed_addr #0 {
  %1 = addrspacecast i32 addrspace(3)* %b to i32 *
  %call = tail call  double @_Z9_cl_frexpdPU7CLlocali(double %a, i32 * %1)
  ret double %call
}


declare double @_Z9_cl_frexpdPU9CLprivatei(double, i32 *) local_unnamed_addr #0

define spir_func double @_Z5frexpdPi(double %a, i32 * %b) local_unnamed_addr #0 {
  %call = tail call  double @_Z9_cl_frexpdPU9CLprivatei(double %a, i32 * %b)
  ret double %call
}


declare <4 x float> @_Z10_cl_lengthDv4_f(<4 x float>) local_unnamed_addr #0

define spir_func <4 x float> @_Z6lengthDv4_f(<4 x float> %a) local_unnamed_addr #0 {
  %call = tail call  <4 x float> @_Z10_cl_lengthDv4_f(<4 x float> %a)
  ret <4 x float> %call
}


declare <4 x double> @_Z10_cl_lengthDv4_d(<4 x double>) local_unnamed_addr #0

define spir_func <4 x double> @_Z6lengthDv4_d(<4 x double> %a) local_unnamed_addr #0 {
  %call = tail call  <4 x double> @_Z10_cl_lengthDv4_d(<4 x double> %a)
  ret <4 x double> %call
}


declare  signext i8 @_Z7_cl_absc(i8) local_unnamed_addr #0

define spir_func  signext i8 @_Z3absc(i8 %a) local_unnamed_addr #0 {
  %call = tail call  signext  i8 @_Z7_cl_absc(i8 %a)
  ret i8 %call
}


declare  signext i8 @_Z7_cl_clzc(i8) local_unnamed_addr #0

define spir_func  signext i8 @_Z3clzc(i8 %a) local_unnamed_addr #0 {
  %call = tail call  signext  i8 @_Z7_cl_clzc(i8 %a)
  ret i8 %call
}


declare  signext i8 @_Z12_cl_popcountc(i8) local_unnamed_addr #0

define spir_func  signext i8 @_Z8popcountc(i8 %a) local_unnamed_addr #0 {
  %call = tail call  signext  i8 @_Z12_cl_popcountc(i8 %a)
  ret i8 %call
}


declare  signext i8 @_Z12_cl_abs_diffcc(i8, i8) local_unnamed_addr #0

define spir_func  signext i8 @_Z8abs_diffcc(i8 %a, i8 %b) local_unnamed_addr #0 {
  %call = tail call  signext  i8 @_Z12_cl_abs_diffcc(i8 %a, i8 %b)
  ret i8 %call
}


declare  signext i8 @_Z11_cl_add_satcc(i8, i8) local_unnamed_addr #0

define spir_func  signext i8 @_Z7add_satcc(i8 %a, i8 %b) local_unnamed_addr #0 {
  %call = tail call  signext  i8 @_Z11_cl_add_satcc(i8 %a, i8 %b)
  ret i8 %call
}


declare  signext i8 @_Z8_cl_haddcc(i8, i8) local_unnamed_addr #0

define spir_func  signext i8 @_Z4haddcc(i8 %a, i8 %b) local_unnamed_addr #0 {
  %call = tail call  signext  i8 @_Z8_cl_haddcc(i8 %a, i8 %b)
  ret i8 %call
}


declare  signext i8 @_Z9_cl_rhaddcc(i8, i8) local_unnamed_addr #0

define spir_func  signext i8 @_Z5rhaddcc(i8 %a, i8 %b) local_unnamed_addr #0 {
  %call = tail call  signext  i8 @_Z9_cl_rhaddcc(i8 %a, i8 %b)
  ret i8 %call
}


declare  signext i8 @_Z7_cl_maxcc(i8, i8) local_unnamed_addr #0

define spir_func  signext i8 @_Z3maxcc(i8 %a, i8 %b) local_unnamed_addr #0 {
  %call = tail call  signext  i8 @_Z7_cl_maxcc(i8 %a, i8 %b)
  ret i8 %call
}


declare  signext i8 @_Z7_cl_mincc(i8, i8) local_unnamed_addr #0

define spir_func  signext i8 @_Z3mincc(i8 %a, i8 %b) local_unnamed_addr #0 {
  %call = tail call  signext  i8 @_Z7_cl_mincc(i8 %a, i8 %b)
  ret i8 %call
}


declare  signext i8 @_Z10_cl_mul_hicc(i8, i8) local_unnamed_addr #0

define spir_func  signext i8 @_Z6mul_hicc(i8 %a, i8 %b) local_unnamed_addr #0 {
  %call = tail call  signext  i8 @_Z10_cl_mul_hicc(i8 %a, i8 %b)
  ret i8 %call
}


declare  signext i8 @_Z10_cl_rotatecc(i8, i8) local_unnamed_addr #0

define spir_func  signext i8 @_Z6rotatecc(i8 %a, i8 %b) local_unnamed_addr #0 {
  %call = tail call  signext  i8 @_Z10_cl_rotatecc(i8 %a, i8 %b)
  ret i8 %call
}


declare  signext i8 @_Z11_cl_sub_satcc(i8, i8) local_unnamed_addr #0

define spir_func  signext i8 @_Z7sub_satcc(i8 %a, i8 %b) local_unnamed_addr #0 {
  %call = tail call  signext  i8 @_Z11_cl_sub_satcc(i8 %a, i8 %b)
  ret i8 %call
}


declare  signext i8 @_Z9_cl_clampccc(i8, i8, i8) local_unnamed_addr #0

define spir_func  signext i8 @_Z5clampccc(i8 %a, i8 %b, i8 %c) local_unnamed_addr #0 {
  %call = tail call  signext  i8 @_Z9_cl_clampccc(i8 %a, i8 %b, i8 %c)
  ret i8 %call
}


declare  signext i8 @_Z10_cl_mad_hiccc(i8, i8, i8) local_unnamed_addr #0

define spir_func  signext i8 @_Z6mad_hiccc(i8 %a, i8 %b, i8 %c) local_unnamed_addr #0 {
  %call = tail call  signext  i8 @_Z10_cl_mad_hiccc(i8 %a, i8 %b, i8 %c)
  ret i8 %call
}


declare  signext i8 @_Z11_cl_mad_satccc(i8, i8, i8) local_unnamed_addr #0

define spir_func  signext i8 @_Z7mad_satccc(i8 %a, i8 %b, i8 %c) local_unnamed_addr #0 {
  %call = tail call  signext  i8 @_Z11_cl_mad_satccc(i8 %a, i8 %b, i8 %c)
  ret i8 %call
}


declare  zeroext i8 @_Z7_cl_absh(i8) local_unnamed_addr #0

define spir_func  zeroext i8 @_Z3absh(i8 %a) local_unnamed_addr #0 {
  %call = tail call  zeroext  i8 @_Z7_cl_absh(i8 %a)
  ret i8 %call
}


declare  zeroext i8 @_Z7_cl_clzh(i8) local_unnamed_addr #0

define spir_func  zeroext i8 @_Z3clzh(i8 %a) local_unnamed_addr #0 {
  %call = tail call  zeroext  i8 @_Z7_cl_clzh(i8 %a)
  ret i8 %call
}


declare  zeroext i8 @_Z12_cl_popcounth(i8) local_unnamed_addr #0

define spir_func  zeroext i8 @_Z8popcounth(i8 %a) local_unnamed_addr #0 {
  %call = tail call  zeroext  i8 @_Z12_cl_popcounth(i8 %a)
  ret i8 %call
}


declare  zeroext i8 @_Z12_cl_abs_diffhh(i8, i8) local_unnamed_addr #0

define spir_func  zeroext i8 @_Z8abs_diffhh(i8 %a, i8 %b) local_unnamed_addr #0 {
  %call = tail call  zeroext  i8 @_Z12_cl_abs_diffhh(i8 %a, i8 %b)
  ret i8 %call
}


declare  zeroext i8 @_Z11_cl_add_sathh(i8, i8) local_unnamed_addr #0

define spir_func  zeroext i8 @_Z7add_sathh(i8 %a, i8 %b) local_unnamed_addr #0 {
  %call = tail call  zeroext  i8 @_Z11_cl_add_sathh(i8 %a, i8 %b)
  ret i8 %call
}


declare  zeroext i8 @_Z8_cl_haddhh(i8, i8) local_unnamed_addr #0

define spir_func  zeroext i8 @_Z4haddhh(i8 %a, i8 %b) local_unnamed_addr #0 {
  %call = tail call  zeroext  i8 @_Z8_cl_haddhh(i8 %a, i8 %b)
  ret i8 %call
}


declare  zeroext i8 @_Z9_cl_rhaddhh(i8, i8) local_unnamed_addr #0

define spir_func  zeroext i8 @_Z5rhaddhh(i8 %a, i8 %b) local_unnamed_addr #0 {
  %call = tail call  zeroext  i8 @_Z9_cl_rhaddhh(i8 %a, i8 %b)
  ret i8 %call
}


declare  zeroext i8 @_Z7_cl_maxhh(i8, i8) local_unnamed_addr #0

define spir_func  zeroext i8 @_Z3maxhh(i8 %a, i8 %b) local_unnamed_addr #0 {
  %call = tail call  zeroext  i8 @_Z7_cl_maxhh(i8 %a, i8 %b)
  ret i8 %call
}


declare  zeroext i8 @_Z7_cl_minhh(i8, i8) local_unnamed_addr #0

define spir_func  zeroext i8 @_Z3minhh(i8 %a, i8 %b) local_unnamed_addr #0 {
  %call = tail call  zeroext  i8 @_Z7_cl_minhh(i8 %a, i8 %b)
  ret i8 %call
}


declare  zeroext i8 @_Z10_cl_mul_hihh(i8, i8) local_unnamed_addr #0

define spir_func  zeroext i8 @_Z6mul_hihh(i8 %a, i8 %b) local_unnamed_addr #0 {
  %call = tail call  zeroext  i8 @_Z10_cl_mul_hihh(i8 %a, i8 %b)
  ret i8 %call
}


declare  zeroext i8 @_Z10_cl_rotatehh(i8, i8) local_unnamed_addr #0

define spir_func  zeroext i8 @_Z6rotatehh(i8 %a, i8 %b) local_unnamed_addr #0 {
  %call = tail call  zeroext  i8 @_Z10_cl_rotatehh(i8 %a, i8 %b)
  ret i8 %call
}


declare  zeroext i8 @_Z11_cl_sub_sathh(i8, i8) local_unnamed_addr #0

define spir_func  zeroext i8 @_Z7sub_sathh(i8 %a, i8 %b) local_unnamed_addr #0 {
  %call = tail call  zeroext  i8 @_Z11_cl_sub_sathh(i8 %a, i8 %b)
  ret i8 %call
}


declare  zeroext i8 @_Z9_cl_clamphhh(i8, i8, i8) local_unnamed_addr #0

define spir_func  zeroext i8 @_Z5clamphhh(i8 %a, i8 %b, i8 %c) local_unnamed_addr #0 {
  %call = tail call  zeroext  i8 @_Z9_cl_clamphhh(i8 %a, i8 %b, i8 %c)
  ret i8 %call
}


declare  zeroext i8 @_Z10_cl_mad_hihhh(i8, i8, i8) local_unnamed_addr #0

define spir_func  zeroext i8 @_Z6mad_hihhh(i8 %a, i8 %b, i8 %c) local_unnamed_addr #0 {
  %call = tail call  zeroext  i8 @_Z10_cl_mad_hihhh(i8 %a, i8 %b, i8 %c)
  ret i8 %call
}


declare  zeroext i8 @_Z11_cl_mad_sathhh(i8, i8, i8) local_unnamed_addr #0

define spir_func  zeroext i8 @_Z7mad_sathhh(i8 %a, i8 %b, i8 %c) local_unnamed_addr #0 {
  %call = tail call  zeroext  i8 @_Z11_cl_mad_sathhh(i8 %a, i8 %b, i8 %c)
  ret i8 %call
}


declare  signext i16 @_Z7_cl_abss(i16) local_unnamed_addr #0

define spir_func  signext i16 @_Z3abss(i16 %a) local_unnamed_addr #0 {
  %call = tail call  signext  i16 @_Z7_cl_abss(i16 %a)
  ret i16 %call
}


declare  signext i16 @_Z7_cl_clzs(i16) local_unnamed_addr #0

define spir_func  signext i16 @_Z3clzs(i16 %a) local_unnamed_addr #0 {
  %call = tail call  signext  i16 @_Z7_cl_clzs(i16 %a)
  ret i16 %call
}


declare  signext i16 @_Z12_cl_popcounts(i16) local_unnamed_addr #0

define spir_func  signext i16 @_Z8popcounts(i16 %a) local_unnamed_addr #0 {
  %call = tail call  signext  i16 @_Z12_cl_popcounts(i16 %a)
  ret i16 %call
}


declare  signext i16 @_Z12_cl_abs_diffss(i16, i16) local_unnamed_addr #0

define spir_func  signext i16 @_Z8abs_diffss(i16 %a, i16 %b) local_unnamed_addr #0 {
  %call = tail call  signext  i16 @_Z12_cl_abs_diffss(i16 %a, i16 %b)
  ret i16 %call
}


declare  signext i16 @_Z11_cl_add_satss(i16, i16) local_unnamed_addr #0

define spir_func  signext i16 @_Z7add_satss(i16 %a, i16 %b) local_unnamed_addr #0 {
  %call = tail call  signext  i16 @_Z11_cl_add_satss(i16 %a, i16 %b)
  ret i16 %call
}


declare  signext i16 @_Z8_cl_haddss(i16, i16) local_unnamed_addr #0

define spir_func  signext i16 @_Z4haddss(i16 %a, i16 %b) local_unnamed_addr #0 {
  %call = tail call  signext  i16 @_Z8_cl_haddss(i16 %a, i16 %b)
  ret i16 %call
}


declare  signext i16 @_Z9_cl_rhaddss(i16, i16) local_unnamed_addr #0

define spir_func  signext i16 @_Z5rhaddss(i16 %a, i16 %b) local_unnamed_addr #0 {
  %call = tail call  signext  i16 @_Z9_cl_rhaddss(i16 %a, i16 %b)
  ret i16 %call
}


declare  signext i16 @_Z7_cl_maxss(i16, i16) local_unnamed_addr #0

define spir_func  signext i16 @_Z3maxss(i16 %a, i16 %b) local_unnamed_addr #0 {
  %call = tail call  signext  i16 @_Z7_cl_maxss(i16 %a, i16 %b)
  ret i16 %call
}


declare  signext i16 @_Z7_cl_minss(i16, i16) local_unnamed_addr #0

define spir_func  signext i16 @_Z3minss(i16 %a, i16 %b) local_unnamed_addr #0 {
  %call = tail call  signext  i16 @_Z7_cl_minss(i16 %a, i16 %b)
  ret i16 %call
}


declare  signext i16 @_Z10_cl_mul_hiss(i16, i16) local_unnamed_addr #0

define spir_func  signext i16 @_Z6mul_hiss(i16 %a, i16 %b) local_unnamed_addr #0 {
  %call = tail call  signext  i16 @_Z10_cl_mul_hiss(i16 %a, i16 %b)
  ret i16 %call
}


declare  signext i16 @_Z10_cl_rotatess(i16, i16) local_unnamed_addr #0

define spir_func  signext i16 @_Z6rotatess(i16 %a, i16 %b) local_unnamed_addr #0 {
  %call = tail call  signext  i16 @_Z10_cl_rotatess(i16 %a, i16 %b)
  ret i16 %call
}


declare  signext i16 @_Z11_cl_sub_satss(i16, i16) local_unnamed_addr #0

define spir_func  signext i16 @_Z7sub_satss(i16 %a, i16 %b) local_unnamed_addr #0 {
  %call = tail call  signext  i16 @_Z11_cl_sub_satss(i16 %a, i16 %b)
  ret i16 %call
}


declare  signext i16 @_Z9_cl_clampsss(i16, i16, i16) local_unnamed_addr #0

define spir_func  signext i16 @_Z5clampsss(i16 %a, i16 %b, i16 %c) local_unnamed_addr #0 {
  %call = tail call  signext  i16 @_Z9_cl_clampsss(i16 %a, i16 %b, i16 %c)
  ret i16 %call
}


declare  signext i16 @_Z10_cl_mad_hisss(i16, i16, i16) local_unnamed_addr #0

define spir_func  signext i16 @_Z6mad_hisss(i16 %a, i16 %b, i16 %c) local_unnamed_addr #0 {
  %call = tail call  signext  i16 @_Z10_cl_mad_hisss(i16 %a, i16 %b, i16 %c)
  ret i16 %call
}


declare  signext i16 @_Z11_cl_mad_satsss(i16, i16, i16) local_unnamed_addr #0

define spir_func  signext i16 @_Z7mad_satsss(i16 %a, i16 %b, i16 %c) local_unnamed_addr #0 {
  %call = tail call  signext  i16 @_Z11_cl_mad_satsss(i16 %a, i16 %b, i16 %c)
  ret i16 %call
}


declare  zeroext i16 @_Z7_cl_abst(i16) local_unnamed_addr #0

define spir_func  zeroext i16 @_Z3abst(i16 %a) local_unnamed_addr #0 {
  %call = tail call  zeroext  i16 @_Z7_cl_abst(i16 %a)
  ret i16 %call
}


declare  zeroext i16 @_Z7_cl_clzt(i16) local_unnamed_addr #0

define spir_func  zeroext i16 @_Z3clzt(i16 %a) local_unnamed_addr #0 {
  %call = tail call  zeroext  i16 @_Z7_cl_clzt(i16 %a)
  ret i16 %call
}


declare  zeroext i16 @_Z12_cl_popcountt(i16) local_unnamed_addr #0

define spir_func  zeroext i16 @_Z8popcountt(i16 %a) local_unnamed_addr #0 {
  %call = tail call  zeroext  i16 @_Z12_cl_popcountt(i16 %a)
  ret i16 %call
}


declare  zeroext i16 @_Z12_cl_abs_difftt(i16, i16) local_unnamed_addr #0

define spir_func  zeroext i16 @_Z8abs_difftt(i16 %a, i16 %b) local_unnamed_addr #0 {
  %call = tail call  zeroext  i16 @_Z12_cl_abs_difftt(i16 %a, i16 %b)
  ret i16 %call
}


declare  zeroext i16 @_Z11_cl_add_sattt(i16, i16) local_unnamed_addr #0

define spir_func  zeroext i16 @_Z7add_sattt(i16 %a, i16 %b) local_unnamed_addr #0 {
  %call = tail call  zeroext  i16 @_Z11_cl_add_sattt(i16 %a, i16 %b)
  ret i16 %call
}


declare  zeroext i16 @_Z8_cl_haddtt(i16, i16) local_unnamed_addr #0

define spir_func  zeroext i16 @_Z4haddtt(i16 %a, i16 %b) local_unnamed_addr #0 {
  %call = tail call  zeroext  i16 @_Z8_cl_haddtt(i16 %a, i16 %b)
  ret i16 %call
}


declare  zeroext i16 @_Z9_cl_rhaddtt(i16, i16) local_unnamed_addr #0

define spir_func  zeroext i16 @_Z5rhaddtt(i16 %a, i16 %b) local_unnamed_addr #0 {
  %call = tail call  zeroext  i16 @_Z9_cl_rhaddtt(i16 %a, i16 %b)
  ret i16 %call
}


declare  zeroext i16 @_Z7_cl_maxtt(i16, i16) local_unnamed_addr #0

define spir_func  zeroext i16 @_Z3maxtt(i16 %a, i16 %b) local_unnamed_addr #0 {
  %call = tail call  zeroext  i16 @_Z7_cl_maxtt(i16 %a, i16 %b)
  ret i16 %call
}


declare  zeroext i16 @_Z7_cl_mintt(i16, i16) local_unnamed_addr #0

define spir_func  zeroext i16 @_Z3mintt(i16 %a, i16 %b) local_unnamed_addr #0 {
  %call = tail call  zeroext  i16 @_Z7_cl_mintt(i16 %a, i16 %b)
  ret i16 %call
}


declare  zeroext i16 @_Z10_cl_mul_hitt(i16, i16) local_unnamed_addr #0

define spir_func  zeroext i16 @_Z6mul_hitt(i16 %a, i16 %b) local_unnamed_addr #0 {
  %call = tail call  zeroext  i16 @_Z10_cl_mul_hitt(i16 %a, i16 %b)
  ret i16 %call
}


declare  zeroext i16 @_Z10_cl_rotatett(i16, i16) local_unnamed_addr #0

define spir_func  zeroext i16 @_Z6rotatett(i16 %a, i16 %b) local_unnamed_addr #0 {
  %call = tail call  zeroext  i16 @_Z10_cl_rotatett(i16 %a, i16 %b)
  ret i16 %call
}


declare  zeroext i16 @_Z11_cl_sub_sattt(i16, i16) local_unnamed_addr #0

define spir_func  zeroext i16 @_Z7sub_sattt(i16 %a, i16 %b) local_unnamed_addr #0 {
  %call = tail call  zeroext  i16 @_Z11_cl_sub_sattt(i16 %a, i16 %b)
  ret i16 %call
}


declare  zeroext i16 @_Z9_cl_clampttt(i16, i16, i16) local_unnamed_addr #0

define spir_func  zeroext i16 @_Z5clampttt(i16 %a, i16 %b, i16 %c) local_unnamed_addr #0 {
  %call = tail call  zeroext  i16 @_Z9_cl_clampttt(i16 %a, i16 %b, i16 %c)
  ret i16 %call
}


declare  zeroext i16 @_Z10_cl_mad_hittt(i16, i16, i16) local_unnamed_addr #0

define spir_func  zeroext i16 @_Z6mad_hittt(i16 %a, i16 %b, i16 %c) local_unnamed_addr #0 {
  %call = tail call  zeroext  i16 @_Z10_cl_mad_hittt(i16 %a, i16 %b, i16 %c)
  ret i16 %call
}


declare  zeroext i16 @_Z11_cl_mad_satttt(i16, i16, i16) local_unnamed_addr #0

define spir_func  zeroext i16 @_Z7mad_satttt(i16 %a, i16 %b, i16 %c) local_unnamed_addr #0 {
  %call = tail call  zeroext  i16 @_Z11_cl_mad_satttt(i16 %a, i16 %b, i16 %c)
  ret i16 %call
}


declare i32 @_Z7_cl_absi(i32) local_unnamed_addr #0

define spir_func i32 @_Z3absi(i32 %a) local_unnamed_addr #0 {
  %call = tail call  i32 @_Z7_cl_absi(i32 %a)
  ret i32 %call
}


declare i32 @_Z7_cl_clzi(i32) local_unnamed_addr #0

define spir_func i32 @_Z3clzi(i32 %a) local_unnamed_addr #0 {
  %call = tail call  i32 @_Z7_cl_clzi(i32 %a)
  ret i32 %call
}


declare i32 @_Z12_cl_popcounti(i32) local_unnamed_addr #0

define spir_func i32 @_Z8popcounti(i32 %a) local_unnamed_addr #0 {
  %call = tail call  i32 @_Z12_cl_popcounti(i32 %a)
  ret i32 %call
}


declare i32 @_Z12_cl_abs_diffii(i32, i32) local_unnamed_addr #0

define spir_func i32 @_Z8abs_diffii(i32 %a, i32 %b) local_unnamed_addr #0 {
  %call = tail call  i32 @_Z12_cl_abs_diffii(i32 %a, i32 %b)
  ret i32 %call
}


declare i32 @_Z11_cl_add_satii(i32, i32) local_unnamed_addr #0

define spir_func i32 @_Z7add_satii(i32 %a, i32 %b) local_unnamed_addr #0 {
  %call = tail call  i32 @_Z11_cl_add_satii(i32 %a, i32 %b)
  ret i32 %call
}


declare i32 @_Z8_cl_haddii(i32, i32) local_unnamed_addr #0

define spir_func i32 @_Z4haddii(i32 %a, i32 %b) local_unnamed_addr #0 {
  %call = tail call  i32 @_Z8_cl_haddii(i32 %a, i32 %b)
  ret i32 %call
}


declare i32 @_Z9_cl_rhaddii(i32, i32) local_unnamed_addr #0

define spir_func i32 @_Z5rhaddii(i32 %a, i32 %b) local_unnamed_addr #0 {
  %call = tail call  i32 @_Z9_cl_rhaddii(i32 %a, i32 %b)
  ret i32 %call
}


declare i32 @_Z7_cl_maxii(i32, i32) local_unnamed_addr #0

define spir_func i32 @_Z3maxii(i32 %a, i32 %b) local_unnamed_addr #0 {
  %call = tail call  i32 @_Z7_cl_maxii(i32 %a, i32 %b)
  ret i32 %call
}


declare i32 @_Z7_cl_minii(i32, i32) local_unnamed_addr #0

define spir_func i32 @_Z3minii(i32 %a, i32 %b) local_unnamed_addr #0 {
  %call = tail call  i32 @_Z7_cl_minii(i32 %a, i32 %b)
  ret i32 %call
}


declare i32 @_Z10_cl_mul_hiii(i32, i32) local_unnamed_addr #0

define spir_func i32 @_Z6mul_hiii(i32 %a, i32 %b) local_unnamed_addr #0 {
  %call = tail call  i32 @_Z10_cl_mul_hiii(i32 %a, i32 %b)
  ret i32 %call
}


declare i32 @_Z10_cl_rotateii(i32, i32) local_unnamed_addr #0

define spir_func i32 @_Z6rotateii(i32 %a, i32 %b) local_unnamed_addr #0 {
  %call = tail call  i32 @_Z10_cl_rotateii(i32 %a, i32 %b)
  ret i32 %call
}


declare i32 @_Z11_cl_sub_satii(i32, i32) local_unnamed_addr #0

define spir_func i32 @_Z7sub_satii(i32 %a, i32 %b) local_unnamed_addr #0 {
  %call = tail call  i32 @_Z11_cl_sub_satii(i32 %a, i32 %b)
  ret i32 %call
}


declare i32 @_Z9_cl_clampiii(i32, i32, i32) local_unnamed_addr #0

define spir_func i32 @_Z5clampiii(i32 %a, i32 %b, i32 %c) local_unnamed_addr #0 {
  %call = tail call  i32 @_Z9_cl_clampiii(i32 %a, i32 %b, i32 %c)
  ret i32 %call
}


declare i32 @_Z10_cl_mad_hiiii(i32, i32, i32) local_unnamed_addr #0

define spir_func i32 @_Z6mad_hiiii(i32 %a, i32 %b, i32 %c) local_unnamed_addr #0 {
  %call = tail call  i32 @_Z10_cl_mad_hiiii(i32 %a, i32 %b, i32 %c)
  ret i32 %call
}


declare i32 @_Z11_cl_mad_satiii(i32, i32, i32) local_unnamed_addr #0

define spir_func i32 @_Z7mad_satiii(i32 %a, i32 %b, i32 %c) local_unnamed_addr #0 {
  %call = tail call  i32 @_Z11_cl_mad_satiii(i32 %a, i32 %b, i32 %c)
  ret i32 %call
}


declare i32 @_Z7_cl_absj(i32) local_unnamed_addr #0

define spir_func i32 @_Z3absj(i32 %a) local_unnamed_addr #0 {
  %call = tail call  i32 @_Z7_cl_absj(i32 %a)
  ret i32 %call
}


declare i32 @_Z7_cl_clzj(i32) local_unnamed_addr #0

define spir_func i32 @_Z3clzj(i32 %a) local_unnamed_addr #0 {
  %call = tail call  i32 @_Z7_cl_clzj(i32 %a)
  ret i32 %call
}


declare i32 @_Z12_cl_popcountj(i32) local_unnamed_addr #0

define spir_func i32 @_Z8popcountj(i32 %a) local_unnamed_addr #0 {
  %call = tail call  i32 @_Z12_cl_popcountj(i32 %a)
  ret i32 %call
}


declare i32 @_Z12_cl_abs_diffjj(i32, i32) local_unnamed_addr #0

define spir_func i32 @_Z8abs_diffjj(i32 %a, i32 %b) local_unnamed_addr #0 {
  %call = tail call  i32 @_Z12_cl_abs_diffjj(i32 %a, i32 %b)
  ret i32 %call
}


declare i32 @_Z11_cl_add_satjj(i32, i32) local_unnamed_addr #0

define spir_func i32 @_Z7add_satjj(i32 %a, i32 %b) local_unnamed_addr #0 {
  %call = tail call  i32 @_Z11_cl_add_satjj(i32 %a, i32 %b)
  ret i32 %call
}


declare i32 @_Z8_cl_haddjj(i32, i32) local_unnamed_addr #0

define spir_func i32 @_Z4haddjj(i32 %a, i32 %b) local_unnamed_addr #0 {
  %call = tail call  i32 @_Z8_cl_haddjj(i32 %a, i32 %b)
  ret i32 %call
}


declare i32 @_Z9_cl_rhaddjj(i32, i32) local_unnamed_addr #0

define spir_func i32 @_Z5rhaddjj(i32 %a, i32 %b) local_unnamed_addr #0 {
  %call = tail call  i32 @_Z9_cl_rhaddjj(i32 %a, i32 %b)
  ret i32 %call
}


declare i32 @_Z7_cl_maxjj(i32, i32) local_unnamed_addr #0

define spir_func i32 @_Z3maxjj(i32 %a, i32 %b) local_unnamed_addr #0 {
  %call = tail call  i32 @_Z7_cl_maxjj(i32 %a, i32 %b)
  ret i32 %call
}


declare i32 @_Z7_cl_minjj(i32, i32) local_unnamed_addr #0

define spir_func i32 @_Z3minjj(i32 %a, i32 %b) local_unnamed_addr #0 {
  %call = tail call  i32 @_Z7_cl_minjj(i32 %a, i32 %b)
  ret i32 %call
}


declare i32 @_Z10_cl_mul_hijj(i32, i32) local_unnamed_addr #0

define spir_func i32 @_Z6mul_hijj(i32 %a, i32 %b) local_unnamed_addr #0 {
  %call = tail call  i32 @_Z10_cl_mul_hijj(i32 %a, i32 %b)
  ret i32 %call
}


declare i32 @_Z10_cl_rotatejj(i32, i32) local_unnamed_addr #0

define spir_func i32 @_Z6rotatejj(i32 %a, i32 %b) local_unnamed_addr #0 {
  %call = tail call  i32 @_Z10_cl_rotatejj(i32 %a, i32 %b)
  ret i32 %call
}


declare i32 @_Z11_cl_sub_satjj(i32, i32) local_unnamed_addr #0

define spir_func i32 @_Z7sub_satjj(i32 %a, i32 %b) local_unnamed_addr #0 {
  %call = tail call  i32 @_Z11_cl_sub_satjj(i32 %a, i32 %b)
  ret i32 %call
}


declare i32 @_Z9_cl_clampjjj(i32, i32, i32) local_unnamed_addr #0

define spir_func i32 @_Z5clampjjj(i32 %a, i32 %b, i32 %c) local_unnamed_addr #0 {
  %call = tail call  i32 @_Z9_cl_clampjjj(i32 %a, i32 %b, i32 %c)
  ret i32 %call
}


declare i32 @_Z10_cl_mad_hijjj(i32, i32, i32) local_unnamed_addr #0

define spir_func i32 @_Z6mad_hijjj(i32 %a, i32 %b, i32 %c) local_unnamed_addr #0 {
  %call = tail call  i32 @_Z10_cl_mad_hijjj(i32 %a, i32 %b, i32 %c)
  ret i32 %call
}


declare i32 @_Z11_cl_mad_satjjj(i32, i32, i32) local_unnamed_addr #0

define spir_func i32 @_Z7mad_satjjj(i32 %a, i32 %b, i32 %c) local_unnamed_addr #0 {
  %call = tail call  i32 @_Z11_cl_mad_satjjj(i32 %a, i32 %b, i32 %c)
  ret i32 %call
}


declare i64 @_Z7_cl_absl(i64) local_unnamed_addr #0

define spir_func i64 @_Z3absl(i64 %a) local_unnamed_addr #0 {
  %call = tail call  i64 @_Z7_cl_absl(i64 %a)
  ret i64 %call
}


declare i64 @_Z7_cl_clzl(i64) local_unnamed_addr #0

define spir_func i64 @_Z3clzl(i64 %a) local_unnamed_addr #0 {
  %call = tail call  i64 @_Z7_cl_clzl(i64 %a)
  ret i64 %call
}


declare i64 @_Z12_cl_popcountl(i64) local_unnamed_addr #0

define spir_func i64 @_Z8popcountl(i64 %a) local_unnamed_addr #0 {
  %call = tail call  i64 @_Z12_cl_popcountl(i64 %a)
  ret i64 %call
}


declare i64 @_Z12_cl_abs_diffll(i64, i64) local_unnamed_addr #0

define spir_func i64 @_Z8abs_diffll(i64 %a, i64 %b) local_unnamed_addr #0 {
  %call = tail call  i64 @_Z12_cl_abs_diffll(i64 %a, i64 %b)
  ret i64 %call
}


declare i64 @_Z11_cl_add_satll(i64, i64) local_unnamed_addr #0

define spir_func i64 @_Z7add_satll(i64 %a, i64 %b) local_unnamed_addr #0 {
  %call = tail call  i64 @_Z11_cl_add_satll(i64 %a, i64 %b)
  ret i64 %call
}


declare i64 @_Z8_cl_haddll(i64, i64) local_unnamed_addr #0

define spir_func i64 @_Z4haddll(i64 %a, i64 %b) local_unnamed_addr #0 {
  %call = tail call  i64 @_Z8_cl_haddll(i64 %a, i64 %b)
  ret i64 %call
}


declare i64 @_Z9_cl_rhaddll(i64, i64) local_unnamed_addr #0

define spir_func i64 @_Z5rhaddll(i64 %a, i64 %b) local_unnamed_addr #0 {
  %call = tail call  i64 @_Z9_cl_rhaddll(i64 %a, i64 %b)
  ret i64 %call
}


declare i64 @_Z7_cl_maxll(i64, i64) local_unnamed_addr #0

define spir_func i64 @_Z3maxll(i64 %a, i64 %b) local_unnamed_addr #0 {
  %call = tail call  i64 @_Z7_cl_maxll(i64 %a, i64 %b)
  ret i64 %call
}


declare i64 @_Z7_cl_minll(i64, i64) local_unnamed_addr #0

define spir_func i64 @_Z3minll(i64 %a, i64 %b) local_unnamed_addr #0 {
  %call = tail call  i64 @_Z7_cl_minll(i64 %a, i64 %b)
  ret i64 %call
}


declare i64 @_Z10_cl_mul_hill(i64, i64) local_unnamed_addr #0

define spir_func i64 @_Z6mul_hill(i64 %a, i64 %b) local_unnamed_addr #0 {
  %call = tail call  i64 @_Z10_cl_mul_hill(i64 %a, i64 %b)
  ret i64 %call
}


declare i64 @_Z10_cl_rotatell(i64, i64) local_unnamed_addr #0

define spir_func i64 @_Z6rotatell(i64 %a, i64 %b) local_unnamed_addr #0 {
  %call = tail call  i64 @_Z10_cl_rotatell(i64 %a, i64 %b)
  ret i64 %call
}


declare i64 @_Z11_cl_sub_satll(i64, i64) local_unnamed_addr #0

define spir_func i64 @_Z7sub_satll(i64 %a, i64 %b) local_unnamed_addr #0 {
  %call = tail call  i64 @_Z11_cl_sub_satll(i64 %a, i64 %b)
  ret i64 %call
}


declare i64 @_Z9_cl_clamplll(i64, i64, i64) local_unnamed_addr #0

define spir_func i64 @_Z5clamplll(i64 %a, i64 %b, i64 %c) local_unnamed_addr #0 {
  %call = tail call  i64 @_Z9_cl_clamplll(i64 %a, i64 %b, i64 %c)
  ret i64 %call
}


declare i64 @_Z10_cl_mad_hilll(i64, i64, i64) local_unnamed_addr #0

define spir_func i64 @_Z6mad_hilll(i64 %a, i64 %b, i64 %c) local_unnamed_addr #0 {
  %call = tail call  i64 @_Z10_cl_mad_hilll(i64 %a, i64 %b, i64 %c)
  ret i64 %call
}


declare i64 @_Z11_cl_mad_satlll(i64, i64, i64) local_unnamed_addr #0

define spir_func i64 @_Z7mad_satlll(i64 %a, i64 %b, i64 %c) local_unnamed_addr #0 {
  %call = tail call  i64 @_Z11_cl_mad_satlll(i64 %a, i64 %b, i64 %c)
  ret i64 %call
}


declare i64 @_Z7_cl_absm(i64) local_unnamed_addr #0

define spir_func i64 @_Z3absm(i64 %a) local_unnamed_addr #0 {
  %call = tail call  i64 @_Z7_cl_absm(i64 %a)
  ret i64 %call
}


declare i64 @_Z7_cl_clzm(i64) local_unnamed_addr #0

define spir_func i64 @_Z3clzm(i64 %a) local_unnamed_addr #0 {
  %call = tail call  i64 @_Z7_cl_clzm(i64 %a)
  ret i64 %call
}


declare i64 @_Z12_cl_popcountm(i64) local_unnamed_addr #0

define spir_func i64 @_Z8popcountm(i64 %a) local_unnamed_addr #0 {
  %call = tail call  i64 @_Z12_cl_popcountm(i64 %a)
  ret i64 %call
}


declare i64 @_Z12_cl_abs_diffmm(i64, i64) local_unnamed_addr #0

define spir_func i64 @_Z8abs_diffmm(i64 %a, i64 %b) local_unnamed_addr #0 {
  %call = tail call  i64 @_Z12_cl_abs_diffmm(i64 %a, i64 %b)
  ret i64 %call
}


declare i64 @_Z11_cl_add_satmm(i64, i64) local_unnamed_addr #0

define spir_func i64 @_Z7add_satmm(i64 %a, i64 %b) local_unnamed_addr #0 {
  %call = tail call  i64 @_Z11_cl_add_satmm(i64 %a, i64 %b)
  ret i64 %call
}


declare i64 @_Z8_cl_haddmm(i64, i64) local_unnamed_addr #0

define spir_func i64 @_Z4haddmm(i64 %a, i64 %b) local_unnamed_addr #0 {
  %call = tail call  i64 @_Z8_cl_haddmm(i64 %a, i64 %b)
  ret i64 %call
}


declare i64 @_Z9_cl_rhaddmm(i64, i64) local_unnamed_addr #0

define spir_func i64 @_Z5rhaddmm(i64 %a, i64 %b) local_unnamed_addr #0 {
  %call = tail call  i64 @_Z9_cl_rhaddmm(i64 %a, i64 %b)
  ret i64 %call
}


declare i64 @_Z7_cl_maxmm(i64, i64) local_unnamed_addr #0

define spir_func i64 @_Z3maxmm(i64 %a, i64 %b) local_unnamed_addr #0 {
  %call = tail call  i64 @_Z7_cl_maxmm(i64 %a, i64 %b)
  ret i64 %call
}


declare i64 @_Z7_cl_minmm(i64, i64) local_unnamed_addr #0

define spir_func i64 @_Z3minmm(i64 %a, i64 %b) local_unnamed_addr #0 {
  %call = tail call  i64 @_Z7_cl_minmm(i64 %a, i64 %b)
  ret i64 %call
}


declare i64 @_Z10_cl_mul_himm(i64, i64) local_unnamed_addr #0

define spir_func i64 @_Z6mul_himm(i64 %a, i64 %b) local_unnamed_addr #0 {
  %call = tail call  i64 @_Z10_cl_mul_himm(i64 %a, i64 %b)
  ret i64 %call
}


declare i64 @_Z10_cl_rotatemm(i64, i64) local_unnamed_addr #0

define spir_func i64 @_Z6rotatemm(i64 %a, i64 %b) local_unnamed_addr #0 {
  %call = tail call  i64 @_Z10_cl_rotatemm(i64 %a, i64 %b)
  ret i64 %call
}


declare i64 @_Z11_cl_sub_satmm(i64, i64) local_unnamed_addr #0

define spir_func i64 @_Z7sub_satmm(i64 %a, i64 %b) local_unnamed_addr #0 {
  %call = tail call  i64 @_Z11_cl_sub_satmm(i64 %a, i64 %b)
  ret i64 %call
}


declare i64 @_Z9_cl_clampmmm(i64, i64, i64) local_unnamed_addr #0

define spir_func i64 @_Z5clampmmm(i64 %a, i64 %b, i64 %c) local_unnamed_addr #0 {
  %call = tail call  i64 @_Z9_cl_clampmmm(i64 %a, i64 %b, i64 %c)
  ret i64 %call
}


declare i64 @_Z10_cl_mad_himmm(i64, i64, i64) local_unnamed_addr #0

define spir_func i64 @_Z6mad_himmm(i64 %a, i64 %b, i64 %c) local_unnamed_addr #0 {
  %call = tail call  i64 @_Z10_cl_mad_himmm(i64 %a, i64 %b, i64 %c)
  ret i64 %call
}


declare i64 @_Z11_cl_mad_satmmm(i64, i64, i64) local_unnamed_addr #0

define spir_func i64 @_Z7mad_satmmm(i64 %a, i64 %b, i64 %c) local_unnamed_addr #0 {
  %call = tail call  i64 @_Z11_cl_mad_satmmm(i64 %a, i64 %b, i64 %c)
  ret i64 %call
}


declare i32 @_Z20_cl_atomic_fetch_addPU8CLglobalVU7_Atomicii(i32 *, i32) local_unnamed_addr #0

define spir_func i32 @_Z16atomic_fetch_addPU3AS1VU7_Atomicii(i32 addrspace(1)* %a, i32 %b) local_unnamed_addr #0 {
  %1 = addrspacecast i32 addrspace(1)* %a to i32 *
  %call = tail call  i32 @_Z20_cl_atomic_fetch_addPU8CLglobalVU7_Atomicii(i32 * %1, i32 %b)
  ret i32 %call
}


declare i32 @_Z20_cl_atomic_fetch_addPU7CLlocalVU7_Atomicii(i32 *, i32) local_unnamed_addr #0

define spir_func i32 @_Z16atomic_fetch_addPU3AS3VU7_Atomicii(i32 addrspace(3)* %a, i32 %b) local_unnamed_addr #0 {
  %1 = addrspacecast i32 addrspace(3)* %a to i32 *
  %call = tail call  i32 @_Z20_cl_atomic_fetch_addPU7CLlocalVU7_Atomicii(i32 * %1, i32 %b)
  ret i32 %call
}


declare i32 @_Z29_cl_atomic_fetch_add_explicitPU8CLglobalVU7_Atomicii12memory_order(i32 *, i32, i32) local_unnamed_addr #0

define spir_func i32 @_Z25atomic_fetch_add_explicitPU3AS1VU7_Atomicii12memory_order(i32 addrspace(1)* %a, i32 %b, i32 %c) local_unnamed_addr #0 {
  %1 = addrspacecast i32 addrspace(1)* %a to i32 *
  %call = tail call  i32 @_Z29_cl_atomic_fetch_add_explicitPU8CLglobalVU7_Atomicii12memory_order(i32 * %1, i32 %b, i32 %c)
  ret i32 %call
}


declare i32 @_Z29_cl_atomic_fetch_add_explicitPU7CLlocalVU7_Atomicii12memory_order(i32 *, i32, i32) local_unnamed_addr #0

define spir_func i32 @_Z25atomic_fetch_add_explicitPU3AS3VU7_Atomicii12memory_order(i32 addrspace(3)* %a, i32 %b, i32 %c) local_unnamed_addr #0 {
  %1 = addrspacecast i32 addrspace(3)* %a to i32 *
  %call = tail call  i32 @_Z29_cl_atomic_fetch_add_explicitPU7CLlocalVU7_Atomicii12memory_order(i32 * %1, i32 %b, i32 %c)
  ret i32 %call
}


declare i32 @_Z29_cl_atomic_fetch_add_explicitPU8CLglobalVU7_Atomicii12memory_order12memory_scope(i32 *, i32, i32, i32) local_unnamed_addr #0

define spir_func i32 @_Z25atomic_fetch_add_explicitPU3AS1VU7_Atomicii12memory_order12memory_scope(i32 addrspace(1)* %a, i32 %b, i32 %c, i32 %d) local_unnamed_addr #0 {
  %1 = addrspacecast i32 addrspace(1)* %a to i32 *
  %call = tail call  i32 @_Z29_cl_atomic_fetch_add_explicitPU8CLglobalVU7_Atomicii12memory_order12memory_scope(i32 * %1, i32 %b, i32 %c, i32 %d)
  ret i32 %call
}


declare i32 @_Z29_cl_atomic_fetch_add_explicitPU7CLlocalVU7_Atomicii12memory_order12memory_scope(i32 *, i32, i32, i32) local_unnamed_addr #0

define spir_func i32 @_Z25atomic_fetch_add_explicitPU3AS3VU7_Atomicii12memory_order12memory_scope(i32 addrspace(3)* %a, i32 %b, i32 %c, i32 %d) local_unnamed_addr #0 {
  %1 = addrspacecast i32 addrspace(3)* %a to i32 *
  %call = tail call  i32 @_Z29_cl_atomic_fetch_add_explicitPU7CLlocalVU7_Atomicii12memory_order12memory_scope(i32 * %1, i32 %b, i32 %c, i32 %d)
  ret i32 %call
}


declare i32 @_Z20_cl_atomic_fetch_subPU8CLglobalVU7_Atomicii(i32 *, i32) local_unnamed_addr #0

define spir_func i32 @_Z16atomic_fetch_subPU3AS1VU7_Atomicii(i32 addrspace(1)* %a, i32 %b) local_unnamed_addr #0 {
  %1 = addrspacecast i32 addrspace(1)* %a to i32 *
  %call = tail call  i32 @_Z20_cl_atomic_fetch_subPU8CLglobalVU7_Atomicii(i32 * %1, i32 %b)
  ret i32 %call
}


declare i32 @_Z20_cl_atomic_fetch_subPU7CLlocalVU7_Atomicii(i32 *, i32) local_unnamed_addr #0

define spir_func i32 @_Z16atomic_fetch_subPU3AS3VU7_Atomicii(i32 addrspace(3)* %a, i32 %b) local_unnamed_addr #0 {
  %1 = addrspacecast i32 addrspace(3)* %a to i32 *
  %call = tail call  i32 @_Z20_cl_atomic_fetch_subPU7CLlocalVU7_Atomicii(i32 * %1, i32 %b)
  ret i32 %call
}


declare i32 @_Z29_cl_atomic_fetch_sub_explicitPU8CLglobalVU7_Atomicii12memory_order(i32 *, i32, i32) local_unnamed_addr #0

define spir_func i32 @_Z25atomic_fetch_sub_explicitPU3AS1VU7_Atomicii12memory_order(i32 addrspace(1)* %a, i32 %b, i32 %c) local_unnamed_addr #0 {
  %1 = addrspacecast i32 addrspace(1)* %a to i32 *
  %call = tail call  i32 @_Z29_cl_atomic_fetch_sub_explicitPU8CLglobalVU7_Atomicii12memory_order(i32 * %1, i32 %b, i32 %c)
  ret i32 %call
}


declare i32 @_Z29_cl_atomic_fetch_sub_explicitPU7CLlocalVU7_Atomicii12memory_order(i32 *, i32, i32) local_unnamed_addr #0

define spir_func i32 @_Z25atomic_fetch_sub_explicitPU3AS3VU7_Atomicii12memory_order(i32 addrspace(3)* %a, i32 %b, i32 %c) local_unnamed_addr #0 {
  %1 = addrspacecast i32 addrspace(3)* %a to i32 *
  %call = tail call  i32 @_Z29_cl_atomic_fetch_sub_explicitPU7CLlocalVU7_Atomicii12memory_order(i32 * %1, i32 %b, i32 %c)
  ret i32 %call
}


declare i32 @_Z29_cl_atomic_fetch_sub_explicitPU8CLglobalVU7_Atomicii12memory_order12memory_scope(i32 *, i32, i32, i32) local_unnamed_addr #0

define spir_func i32 @_Z25atomic_fetch_sub_explicitPU3AS1VU7_Atomicii12memory_order12memory_scope(i32 addrspace(1)* %a, i32 %b, i32 %c, i32 %d) local_unnamed_addr #0 {
  %1 = addrspacecast i32 addrspace(1)* %a to i32 *
  %call = tail call  i32 @_Z29_cl_atomic_fetch_sub_explicitPU8CLglobalVU7_Atomicii12memory_order12memory_scope(i32 * %1, i32 %b, i32 %c, i32 %d)
  ret i32 %call
}


declare i32 @_Z29_cl_atomic_fetch_sub_explicitPU7CLlocalVU7_Atomicii12memory_order12memory_scope(i32 *, i32, i32, i32) local_unnamed_addr #0

define spir_func i32 @_Z25atomic_fetch_sub_explicitPU3AS3VU7_Atomicii12memory_order12memory_scope(i32 addrspace(3)* %a, i32 %b, i32 %c, i32 %d) local_unnamed_addr #0 {
  %1 = addrspacecast i32 addrspace(3)* %a to i32 *
  %call = tail call  i32 @_Z29_cl_atomic_fetch_sub_explicitPU7CLlocalVU7_Atomicii12memory_order12memory_scope(i32 * %1, i32 %b, i32 %c, i32 %d)
  ret i32 %call
}


declare i32 @_Z19_cl_atomic_fetch_orPU8CLglobalVU7_Atomicii(i32 *, i32) local_unnamed_addr #0

define spir_func i32 @_Z15atomic_fetch_orPU3AS1VU7_Atomicii(i32 addrspace(1)* %a, i32 %b) local_unnamed_addr #0 {
  %1 = addrspacecast i32 addrspace(1)* %a to i32 *
  %call = tail call  i32 @_Z19_cl_atomic_fetch_orPU8CLglobalVU7_Atomicii(i32 * %1, i32 %b)
  ret i32 %call
}


declare i32 @_Z19_cl_atomic_fetch_orPU7CLlocalVU7_Atomicii(i32 *, i32) local_unnamed_addr #0

define spir_func i32 @_Z15atomic_fetch_orPU3AS3VU7_Atomicii(i32 addrspace(3)* %a, i32 %b) local_unnamed_addr #0 {
  %1 = addrspacecast i32 addrspace(3)* %a to i32 *
  %call = tail call  i32 @_Z19_cl_atomic_fetch_orPU7CLlocalVU7_Atomicii(i32 * %1, i32 %b)
  ret i32 %call
}


declare i32 @_Z28_cl_atomic_fetch_or_explicitPU8CLglobalVU7_Atomicii12memory_order(i32 *, i32, i32) local_unnamed_addr #0

define spir_func i32 @_Z24atomic_fetch_or_explicitPU3AS1VU7_Atomicii12memory_order(i32 addrspace(1)* %a, i32 %b, i32 %c) local_unnamed_addr #0 {
  %1 = addrspacecast i32 addrspace(1)* %a to i32 *
  %call = tail call  i32 @_Z28_cl_atomic_fetch_or_explicitPU8CLglobalVU7_Atomicii12memory_order(i32 * %1, i32 %b, i32 %c)
  ret i32 %call
}


declare i32 @_Z28_cl_atomic_fetch_or_explicitPU7CLlocalVU7_Atomicii12memory_order(i32 *, i32, i32) local_unnamed_addr #0

define spir_func i32 @_Z24atomic_fetch_or_explicitPU3AS3VU7_Atomicii12memory_order(i32 addrspace(3)* %a, i32 %b, i32 %c) local_unnamed_addr #0 {
  %1 = addrspacecast i32 addrspace(3)* %a to i32 *
  %call = tail call  i32 @_Z28_cl_atomic_fetch_or_explicitPU7CLlocalVU7_Atomicii12memory_order(i32 * %1, i32 %b, i32 %c)
  ret i32 %call
}


declare i32 @_Z28_cl_atomic_fetch_or_explicitPU8CLglobalVU7_Atomicii12memory_order12memory_scope(i32 *, i32, i32, i32) local_unnamed_addr #0

define spir_func i32 @_Z24atomic_fetch_or_explicitPU3AS1VU7_Atomicii12memory_order12memory_scope(i32 addrspace(1)* %a, i32 %b, i32 %c, i32 %d) local_unnamed_addr #0 {
  %1 = addrspacecast i32 addrspace(1)* %a to i32 *
  %call = tail call  i32 @_Z28_cl_atomic_fetch_or_explicitPU8CLglobalVU7_Atomicii12memory_order12memory_scope(i32 * %1, i32 %b, i32 %c, i32 %d)
  ret i32 %call
}


declare i32 @_Z28_cl_atomic_fetch_or_explicitPU7CLlocalVU7_Atomicii12memory_order12memory_scope(i32 *, i32, i32, i32) local_unnamed_addr #0

define spir_func i32 @_Z24atomic_fetch_or_explicitPU3AS3VU7_Atomicii12memory_order12memory_scope(i32 addrspace(3)* %a, i32 %b, i32 %c, i32 %d) local_unnamed_addr #0 {
  %1 = addrspacecast i32 addrspace(3)* %a to i32 *
  %call = tail call  i32 @_Z28_cl_atomic_fetch_or_explicitPU7CLlocalVU7_Atomicii12memory_order12memory_scope(i32 * %1, i32 %b, i32 %c, i32 %d)
  ret i32 %call
}


declare i32 @_Z20_cl_atomic_fetch_xorPU8CLglobalVU7_Atomicii(i32 *, i32) local_unnamed_addr #0

define spir_func i32 @_Z16atomic_fetch_xorPU3AS1VU7_Atomicii(i32 addrspace(1)* %a, i32 %b) local_unnamed_addr #0 {
  %1 = addrspacecast i32 addrspace(1)* %a to i32 *
  %call = tail call  i32 @_Z20_cl_atomic_fetch_xorPU8CLglobalVU7_Atomicii(i32 * %1, i32 %b)
  ret i32 %call
}


declare i32 @_Z20_cl_atomic_fetch_xorPU7CLlocalVU7_Atomicii(i32 *, i32) local_unnamed_addr #0

define spir_func i32 @_Z16atomic_fetch_xorPU3AS3VU7_Atomicii(i32 addrspace(3)* %a, i32 %b) local_unnamed_addr #0 {
  %1 = addrspacecast i32 addrspace(3)* %a to i32 *
  %call = tail call  i32 @_Z20_cl_atomic_fetch_xorPU7CLlocalVU7_Atomicii(i32 * %1, i32 %b)
  ret i32 %call
}


declare i32 @_Z29_cl_atomic_fetch_xor_explicitPU8CLglobalVU7_Atomicii12memory_order(i32 *, i32, i32) local_unnamed_addr #0

define spir_func i32 @_Z25atomic_fetch_xor_explicitPU3AS1VU7_Atomicii12memory_order(i32 addrspace(1)* %a, i32 %b, i32 %c) local_unnamed_addr #0 {
  %1 = addrspacecast i32 addrspace(1)* %a to i32 *
  %call = tail call  i32 @_Z29_cl_atomic_fetch_xor_explicitPU8CLglobalVU7_Atomicii12memory_order(i32 * %1, i32 %b, i32 %c)
  ret i32 %call
}


declare i32 @_Z29_cl_atomic_fetch_xor_explicitPU7CLlocalVU7_Atomicii12memory_order(i32 *, i32, i32) local_unnamed_addr #0

define spir_func i32 @_Z25atomic_fetch_xor_explicitPU3AS3VU7_Atomicii12memory_order(i32 addrspace(3)* %a, i32 %b, i32 %c) local_unnamed_addr #0 {
  %1 = addrspacecast i32 addrspace(3)* %a to i32 *
  %call = tail call  i32 @_Z29_cl_atomic_fetch_xor_explicitPU7CLlocalVU7_Atomicii12memory_order(i32 * %1, i32 %b, i32 %c)
  ret i32 %call
}


declare i32 @_Z29_cl_atomic_fetch_xor_explicitPU8CLglobalVU7_Atomicii12memory_order12memory_scope(i32 *, i32, i32, i32) local_unnamed_addr #0

define spir_func i32 @_Z25atomic_fetch_xor_explicitPU3AS1VU7_Atomicii12memory_order12memory_scope(i32 addrspace(1)* %a, i32 %b, i32 %c, i32 %d) local_unnamed_addr #0 {
  %1 = addrspacecast i32 addrspace(1)* %a to i32 *
  %call = tail call  i32 @_Z29_cl_atomic_fetch_xor_explicitPU8CLglobalVU7_Atomicii12memory_order12memory_scope(i32 * %1, i32 %b, i32 %c, i32 %d)
  ret i32 %call
}


declare i32 @_Z29_cl_atomic_fetch_xor_explicitPU7CLlocalVU7_Atomicii12memory_order12memory_scope(i32 *, i32, i32, i32) local_unnamed_addr #0

define spir_func i32 @_Z25atomic_fetch_xor_explicitPU3AS3VU7_Atomicii12memory_order12memory_scope(i32 addrspace(3)* %a, i32 %b, i32 %c, i32 %d) local_unnamed_addr #0 {
  %1 = addrspacecast i32 addrspace(3)* %a to i32 *
  %call = tail call  i32 @_Z29_cl_atomic_fetch_xor_explicitPU7CLlocalVU7_Atomicii12memory_order12memory_scope(i32 * %1, i32 %b, i32 %c, i32 %d)
  ret i32 %call
}


declare i32 @_Z20_cl_atomic_fetch_andPU8CLglobalVU7_Atomicii(i32 *, i32) local_unnamed_addr #0

define spir_func i32 @_Z16atomic_fetch_andPU3AS1VU7_Atomicii(i32 addrspace(1)* %a, i32 %b) local_unnamed_addr #0 {
  %1 = addrspacecast i32 addrspace(1)* %a to i32 *
  %call = tail call  i32 @_Z20_cl_atomic_fetch_andPU8CLglobalVU7_Atomicii(i32 * %1, i32 %b)
  ret i32 %call
}


declare i32 @_Z20_cl_atomic_fetch_andPU7CLlocalVU7_Atomicii(i32 *, i32) local_unnamed_addr #0

define spir_func i32 @_Z16atomic_fetch_andPU3AS3VU7_Atomicii(i32 addrspace(3)* %a, i32 %b) local_unnamed_addr #0 {
  %1 = addrspacecast i32 addrspace(3)* %a to i32 *
  %call = tail call  i32 @_Z20_cl_atomic_fetch_andPU7CLlocalVU7_Atomicii(i32 * %1, i32 %b)
  ret i32 %call
}


declare i32 @_Z29_cl_atomic_fetch_and_explicitPU8CLglobalVU7_Atomicii12memory_order(i32 *, i32, i32) local_unnamed_addr #0

define spir_func i32 @_Z25atomic_fetch_and_explicitPU3AS1VU7_Atomicii12memory_order(i32 addrspace(1)* %a, i32 %b, i32 %c) local_unnamed_addr #0 {
  %1 = addrspacecast i32 addrspace(1)* %a to i32 *
  %call = tail call  i32 @_Z29_cl_atomic_fetch_and_explicitPU8CLglobalVU7_Atomicii12memory_order(i32 * %1, i32 %b, i32 %c)
  ret i32 %call
}


declare i32 @_Z29_cl_atomic_fetch_and_explicitPU7CLlocalVU7_Atomicii12memory_order(i32 *, i32, i32) local_unnamed_addr #0

define spir_func i32 @_Z25atomic_fetch_and_explicitPU3AS3VU7_Atomicii12memory_order(i32 addrspace(3)* %a, i32 %b, i32 %c) local_unnamed_addr #0 {
  %1 = addrspacecast i32 addrspace(3)* %a to i32 *
  %call = tail call  i32 @_Z29_cl_atomic_fetch_and_explicitPU7CLlocalVU7_Atomicii12memory_order(i32 * %1, i32 %b, i32 %c)
  ret i32 %call
}


declare i32 @_Z29_cl_atomic_fetch_and_explicitPU8CLglobalVU7_Atomicii12memory_order12memory_scope(i32 *, i32, i32, i32) local_unnamed_addr #0

define spir_func i32 @_Z25atomic_fetch_and_explicitPU3AS1VU7_Atomicii12memory_order12memory_scope(i32 addrspace(1)* %a, i32 %b, i32 %c, i32 %d) local_unnamed_addr #0 {
  %1 = addrspacecast i32 addrspace(1)* %a to i32 *
  %call = tail call  i32 @_Z29_cl_atomic_fetch_and_explicitPU8CLglobalVU7_Atomicii12memory_order12memory_scope(i32 * %1, i32 %b, i32 %c, i32 %d)
  ret i32 %call
}


declare i32 @_Z29_cl_atomic_fetch_and_explicitPU7CLlocalVU7_Atomicii12memory_order12memory_scope(i32 *, i32, i32, i32) local_unnamed_addr #0

define spir_func i32 @_Z25atomic_fetch_and_explicitPU3AS3VU7_Atomicii12memory_order12memory_scope(i32 addrspace(3)* %a, i32 %b, i32 %c, i32 %d) local_unnamed_addr #0 {
  %1 = addrspacecast i32 addrspace(3)* %a to i32 *
  %call = tail call  i32 @_Z29_cl_atomic_fetch_and_explicitPU7CLlocalVU7_Atomicii12memory_order12memory_scope(i32 * %1, i32 %b, i32 %c, i32 %d)
  ret i32 %call
}


declare i32 @_Z20_cl_atomic_fetch_minPU8CLglobalVU7_Atomicii(i32 *, i32) local_unnamed_addr #0

define spir_func i32 @_Z16atomic_fetch_minPU3AS1VU7_Atomicii(i32 addrspace(1)* %a, i32 %b) local_unnamed_addr #0 {
  %1 = addrspacecast i32 addrspace(1)* %a to i32 *
  %call = tail call  i32 @_Z20_cl_atomic_fetch_minPU8CLglobalVU7_Atomicii(i32 * %1, i32 %b)
  ret i32 %call
}


declare i32 @_Z20_cl_atomic_fetch_minPU7CLlocalVU7_Atomicii(i32 *, i32) local_unnamed_addr #0

define spir_func i32 @_Z16atomic_fetch_minPU3AS3VU7_Atomicii(i32 addrspace(3)* %a, i32 %b) local_unnamed_addr #0 {
  %1 = addrspacecast i32 addrspace(3)* %a to i32 *
  %call = tail call  i32 @_Z20_cl_atomic_fetch_minPU7CLlocalVU7_Atomicii(i32 * %1, i32 %b)
  ret i32 %call
}


declare i32 @_Z29_cl_atomic_fetch_min_explicitPU8CLglobalVU7_Atomicii12memory_order(i32 *, i32, i32) local_unnamed_addr #0

define spir_func i32 @_Z25atomic_fetch_min_explicitPU3AS1VU7_Atomicii12memory_order(i32 addrspace(1)* %a, i32 %b, i32 %c) local_unnamed_addr #0 {
  %1 = addrspacecast i32 addrspace(1)* %a to i32 *
  %call = tail call  i32 @_Z29_cl_atomic_fetch_min_explicitPU8CLglobalVU7_Atomicii12memory_order(i32 * %1, i32 %b, i32 %c)
  ret i32 %call
}


declare i32 @_Z29_cl_atomic_fetch_min_explicitPU7CLlocalVU7_Atomicii12memory_order(i32 *, i32, i32) local_unnamed_addr #0

define spir_func i32 @_Z25atomic_fetch_min_explicitPU3AS3VU7_Atomicii12memory_order(i32 addrspace(3)* %a, i32 %b, i32 %c) local_unnamed_addr #0 {
  %1 = addrspacecast i32 addrspace(3)* %a to i32 *
  %call = tail call  i32 @_Z29_cl_atomic_fetch_min_explicitPU7CLlocalVU7_Atomicii12memory_order(i32 * %1, i32 %b, i32 %c)
  ret i32 %call
}


declare i32 @_Z29_cl_atomic_fetch_min_explicitPU8CLglobalVU7_Atomicii12memory_order12memory_scope(i32 *, i32, i32, i32) local_unnamed_addr #0

define spir_func i32 @_Z25atomic_fetch_min_explicitPU3AS1VU7_Atomicii12memory_order12memory_scope(i32 addrspace(1)* %a, i32 %b, i32 %c, i32 %d) local_unnamed_addr #0 {
  %1 = addrspacecast i32 addrspace(1)* %a to i32 *
  %call = tail call  i32 @_Z29_cl_atomic_fetch_min_explicitPU8CLglobalVU7_Atomicii12memory_order12memory_scope(i32 * %1, i32 %b, i32 %c, i32 %d)
  ret i32 %call
}


declare i32 @_Z29_cl_atomic_fetch_min_explicitPU7CLlocalVU7_Atomicii12memory_order12memory_scope(i32 *, i32, i32, i32) local_unnamed_addr #0

define spir_func i32 @_Z25atomic_fetch_min_explicitPU3AS3VU7_Atomicii12memory_order12memory_scope(i32 addrspace(3)* %a, i32 %b, i32 %c, i32 %d) local_unnamed_addr #0 {
  %1 = addrspacecast i32 addrspace(3)* %a to i32 *
  %call = tail call  i32 @_Z29_cl_atomic_fetch_min_explicitPU7CLlocalVU7_Atomicii12memory_order12memory_scope(i32 * %1, i32 %b, i32 %c, i32 %d)
  ret i32 %call
}


declare i32 @_Z20_cl_atomic_fetch_maxPU8CLglobalVU7_Atomicii(i32 *, i32) local_unnamed_addr #0

define spir_func i32 @_Z16atomic_fetch_maxPU3AS1VU7_Atomicii(i32 addrspace(1)* %a, i32 %b) local_unnamed_addr #0 {
  %1 = addrspacecast i32 addrspace(1)* %a to i32 *
  %call = tail call  i32 @_Z20_cl_atomic_fetch_maxPU8CLglobalVU7_Atomicii(i32 * %1, i32 %b)
  ret i32 %call
}


declare i32 @_Z20_cl_atomic_fetch_maxPU7CLlocalVU7_Atomicii(i32 *, i32) local_unnamed_addr #0

define spir_func i32 @_Z16atomic_fetch_maxPU3AS3VU7_Atomicii(i32 addrspace(3)* %a, i32 %b) local_unnamed_addr #0 {
  %1 = addrspacecast i32 addrspace(3)* %a to i32 *
  %call = tail call  i32 @_Z20_cl_atomic_fetch_maxPU7CLlocalVU7_Atomicii(i32 * %1, i32 %b)
  ret i32 %call
}


declare i32 @_Z29_cl_atomic_fetch_max_explicitPU8CLglobalVU7_Atomicii12memory_order(i32 *, i32, i32) local_unnamed_addr #0

define spir_func i32 @_Z25atomic_fetch_max_explicitPU3AS1VU7_Atomicii12memory_order(i32 addrspace(1)* %a, i32 %b, i32 %c) local_unnamed_addr #0 {
  %1 = addrspacecast i32 addrspace(1)* %a to i32 *
  %call = tail call  i32 @_Z29_cl_atomic_fetch_max_explicitPU8CLglobalVU7_Atomicii12memory_order(i32 * %1, i32 %b, i32 %c)
  ret i32 %call
}


declare i32 @_Z29_cl_atomic_fetch_max_explicitPU7CLlocalVU7_Atomicii12memory_order(i32 *, i32, i32) local_unnamed_addr #0

define spir_func i32 @_Z25atomic_fetch_max_explicitPU3AS3VU7_Atomicii12memory_order(i32 addrspace(3)* %a, i32 %b, i32 %c) local_unnamed_addr #0 {
  %1 = addrspacecast i32 addrspace(3)* %a to i32 *
  %call = tail call  i32 @_Z29_cl_atomic_fetch_max_explicitPU7CLlocalVU7_Atomicii12memory_order(i32 * %1, i32 %b, i32 %c)
  ret i32 %call
}


declare i32 @_Z29_cl_atomic_fetch_max_explicitPU8CLglobalVU7_Atomicii12memory_order12memory_scope(i32 *, i32, i32, i32) local_unnamed_addr #0

define spir_func i32 @_Z25atomic_fetch_max_explicitPU3AS1VU7_Atomicii12memory_order12memory_scope(i32 addrspace(1)* %a, i32 %b, i32 %c, i32 %d) local_unnamed_addr #0 {
  %1 = addrspacecast i32 addrspace(1)* %a to i32 *
  %call = tail call  i32 @_Z29_cl_atomic_fetch_max_explicitPU8CLglobalVU7_Atomicii12memory_order12memory_scope(i32 * %1, i32 %b, i32 %c, i32 %d)
  ret i32 %call
}


declare i32 @_Z29_cl_atomic_fetch_max_explicitPU7CLlocalVU7_Atomicii12memory_order12memory_scope(i32 *, i32, i32, i32) local_unnamed_addr #0

define spir_func i32 @_Z25atomic_fetch_max_explicitPU3AS3VU7_Atomicii12memory_order12memory_scope(i32 addrspace(3)* %a, i32 %b, i32 %c, i32 %d) local_unnamed_addr #0 {
  %1 = addrspacecast i32 addrspace(3)* %a to i32 *
  %call = tail call  i32 @_Z29_cl_atomic_fetch_max_explicitPU7CLlocalVU7_Atomicii12memory_order12memory_scope(i32 * %1, i32 %b, i32 %c, i32 %d)
  ret i32 %call
}


declare i32 @_Z20_cl_atomic_fetch_addPU8CLglobalVU7_Atomicjj(i32 *, i32) local_unnamed_addr #0

define spir_func i32 @_Z16atomic_fetch_addPU3AS1VU7_Atomicjj(i32 addrspace(1)* %a, i32 %b) local_unnamed_addr #0 {
  %1 = addrspacecast i32 addrspace(1)* %a to i32 *
  %call = tail call  i32 @_Z20_cl_atomic_fetch_addPU8CLglobalVU7_Atomicjj(i32 * %1, i32 %b)
  ret i32 %call
}


declare i32 @_Z20_cl_atomic_fetch_addPU7CLlocalVU7_Atomicjj(i32 *, i32) local_unnamed_addr #0

define spir_func i32 @_Z16atomic_fetch_addPU3AS3VU7_Atomicjj(i32 addrspace(3)* %a, i32 %b) local_unnamed_addr #0 {
  %1 = addrspacecast i32 addrspace(3)* %a to i32 *
  %call = tail call  i32 @_Z20_cl_atomic_fetch_addPU7CLlocalVU7_Atomicjj(i32 * %1, i32 %b)
  ret i32 %call
}


declare i32 @_Z29_cl_atomic_fetch_add_explicitPU8CLglobalVU7_Atomicjj12memory_order(i32 *, i32, i32) local_unnamed_addr #0

define spir_func i32 @_Z25atomic_fetch_add_explicitPU3AS1VU7_Atomicjj12memory_order(i32 addrspace(1)* %a, i32 %b, i32 %c) local_unnamed_addr #0 {
  %1 = addrspacecast i32 addrspace(1)* %a to i32 *
  %call = tail call  i32 @_Z29_cl_atomic_fetch_add_explicitPU8CLglobalVU7_Atomicjj12memory_order(i32 * %1, i32 %b, i32 %c)
  ret i32 %call
}


declare i32 @_Z29_cl_atomic_fetch_add_explicitPU7CLlocalVU7_Atomicjj12memory_order(i32 *, i32, i32) local_unnamed_addr #0

define spir_func i32 @_Z25atomic_fetch_add_explicitPU3AS3VU7_Atomicjj12memory_order(i32 addrspace(3)* %a, i32 %b, i32 %c) local_unnamed_addr #0 {
  %1 = addrspacecast i32 addrspace(3)* %a to i32 *
  %call = tail call  i32 @_Z29_cl_atomic_fetch_add_explicitPU7CLlocalVU7_Atomicjj12memory_order(i32 * %1, i32 %b, i32 %c)
  ret i32 %call
}


declare i32 @_Z29_cl_atomic_fetch_add_explicitPU8CLglobalVU7_Atomicjj12memory_order12memory_scope(i32 *, i32, i32, i32) local_unnamed_addr #0

define spir_func i32 @_Z25atomic_fetch_add_explicitPU3AS1VU7_Atomicjj12memory_order12memory_scope(i32 addrspace(1)* %a, i32 %b, i32 %c, i32 %d) local_unnamed_addr #0 {
  %1 = addrspacecast i32 addrspace(1)* %a to i32 *
  %call = tail call  i32 @_Z29_cl_atomic_fetch_add_explicitPU8CLglobalVU7_Atomicjj12memory_order12memory_scope(i32 * %1, i32 %b, i32 %c, i32 %d)
  ret i32 %call
}


declare i32 @_Z29_cl_atomic_fetch_add_explicitPU7CLlocalVU7_Atomicjj12memory_order12memory_scope(i32 *, i32, i32, i32) local_unnamed_addr #0

define spir_func i32 @_Z25atomic_fetch_add_explicitPU3AS3VU7_Atomicjj12memory_order12memory_scope(i32 addrspace(3)* %a, i32 %b, i32 %c, i32 %d) local_unnamed_addr #0 {
  %1 = addrspacecast i32 addrspace(3)* %a to i32 *
  %call = tail call  i32 @_Z29_cl_atomic_fetch_add_explicitPU7CLlocalVU7_Atomicjj12memory_order12memory_scope(i32 * %1, i32 %b, i32 %c, i32 %d)
  ret i32 %call
}


declare i32 @_Z20_cl_atomic_fetch_subPU8CLglobalVU7_Atomicjj(i32 *, i32) local_unnamed_addr #0

define spir_func i32 @_Z16atomic_fetch_subPU3AS1VU7_Atomicjj(i32 addrspace(1)* %a, i32 %b) local_unnamed_addr #0 {
  %1 = addrspacecast i32 addrspace(1)* %a to i32 *
  %call = tail call  i32 @_Z20_cl_atomic_fetch_subPU8CLglobalVU7_Atomicjj(i32 * %1, i32 %b)
  ret i32 %call
}


declare i32 @_Z20_cl_atomic_fetch_subPU7CLlocalVU7_Atomicjj(i32 *, i32) local_unnamed_addr #0

define spir_func i32 @_Z16atomic_fetch_subPU3AS3VU7_Atomicjj(i32 addrspace(3)* %a, i32 %b) local_unnamed_addr #0 {
  %1 = addrspacecast i32 addrspace(3)* %a to i32 *
  %call = tail call  i32 @_Z20_cl_atomic_fetch_subPU7CLlocalVU7_Atomicjj(i32 * %1, i32 %b)
  ret i32 %call
}


declare i32 @_Z29_cl_atomic_fetch_sub_explicitPU8CLglobalVU7_Atomicjj12memory_order(i32 *, i32, i32) local_unnamed_addr #0

define spir_func i32 @_Z25atomic_fetch_sub_explicitPU3AS1VU7_Atomicjj12memory_order(i32 addrspace(1)* %a, i32 %b, i32 %c) local_unnamed_addr #0 {
  %1 = addrspacecast i32 addrspace(1)* %a to i32 *
  %call = tail call  i32 @_Z29_cl_atomic_fetch_sub_explicitPU8CLglobalVU7_Atomicjj12memory_order(i32 * %1, i32 %b, i32 %c)
  ret i32 %call
}


declare i32 @_Z29_cl_atomic_fetch_sub_explicitPU7CLlocalVU7_Atomicjj12memory_order(i32 *, i32, i32) local_unnamed_addr #0

define spir_func i32 @_Z25atomic_fetch_sub_explicitPU3AS3VU7_Atomicjj12memory_order(i32 addrspace(3)* %a, i32 %b, i32 %c) local_unnamed_addr #0 {
  %1 = addrspacecast i32 addrspace(3)* %a to i32 *
  %call = tail call  i32 @_Z29_cl_atomic_fetch_sub_explicitPU7CLlocalVU7_Atomicjj12memory_order(i32 * %1, i32 %b, i32 %c)
  ret i32 %call
}


declare i32 @_Z29_cl_atomic_fetch_sub_explicitPU8CLglobalVU7_Atomicjj12memory_order12memory_scope(i32 *, i32, i32, i32) local_unnamed_addr #0

define spir_func i32 @_Z25atomic_fetch_sub_explicitPU3AS1VU7_Atomicjj12memory_order12memory_scope(i32 addrspace(1)* %a, i32 %b, i32 %c, i32 %d) local_unnamed_addr #0 {
  %1 = addrspacecast i32 addrspace(1)* %a to i32 *
  %call = tail call  i32 @_Z29_cl_atomic_fetch_sub_explicitPU8CLglobalVU7_Atomicjj12memory_order12memory_scope(i32 * %1, i32 %b, i32 %c, i32 %d)
  ret i32 %call
}


declare i32 @_Z29_cl_atomic_fetch_sub_explicitPU7CLlocalVU7_Atomicjj12memory_order12memory_scope(i32 *, i32, i32, i32) local_unnamed_addr #0

define spir_func i32 @_Z25atomic_fetch_sub_explicitPU3AS3VU7_Atomicjj12memory_order12memory_scope(i32 addrspace(3)* %a, i32 %b, i32 %c, i32 %d) local_unnamed_addr #0 {
  %1 = addrspacecast i32 addrspace(3)* %a to i32 *
  %call = tail call  i32 @_Z29_cl_atomic_fetch_sub_explicitPU7CLlocalVU7_Atomicjj12memory_order12memory_scope(i32 * %1, i32 %b, i32 %c, i32 %d)
  ret i32 %call
}


declare i32 @_Z19_cl_atomic_fetch_orPU8CLglobalVU7_Atomicjj(i32 *, i32) local_unnamed_addr #0

define spir_func i32 @_Z15atomic_fetch_orPU3AS1VU7_Atomicjj(i32 addrspace(1)* %a, i32 %b) local_unnamed_addr #0 {
  %1 = addrspacecast i32 addrspace(1)* %a to i32 *
  %call = tail call  i32 @_Z19_cl_atomic_fetch_orPU8CLglobalVU7_Atomicjj(i32 * %1, i32 %b)
  ret i32 %call
}


declare i32 @_Z19_cl_atomic_fetch_orPU7CLlocalVU7_Atomicjj(i32 *, i32) local_unnamed_addr #0

define spir_func i32 @_Z15atomic_fetch_orPU3AS3VU7_Atomicjj(i32 addrspace(3)* %a, i32 %b) local_unnamed_addr #0 {
  %1 = addrspacecast i32 addrspace(3)* %a to i32 *
  %call = tail call  i32 @_Z19_cl_atomic_fetch_orPU7CLlocalVU7_Atomicjj(i32 * %1, i32 %b)
  ret i32 %call
}


declare i32 @_Z28_cl_atomic_fetch_or_explicitPU8CLglobalVU7_Atomicjj12memory_order(i32 *, i32, i32) local_unnamed_addr #0

define spir_func i32 @_Z24atomic_fetch_or_explicitPU3AS1VU7_Atomicjj12memory_order(i32 addrspace(1)* %a, i32 %b, i32 %c) local_unnamed_addr #0 {
  %1 = addrspacecast i32 addrspace(1)* %a to i32 *
  %call = tail call  i32 @_Z28_cl_atomic_fetch_or_explicitPU8CLglobalVU7_Atomicjj12memory_order(i32 * %1, i32 %b, i32 %c)
  ret i32 %call
}


declare i32 @_Z28_cl_atomic_fetch_or_explicitPU7CLlocalVU7_Atomicjj12memory_order(i32 *, i32, i32) local_unnamed_addr #0

define spir_func i32 @_Z24atomic_fetch_or_explicitPU3AS3VU7_Atomicjj12memory_order(i32 addrspace(3)* %a, i32 %b, i32 %c) local_unnamed_addr #0 {
  %1 = addrspacecast i32 addrspace(3)* %a to i32 *
  %call = tail call  i32 @_Z28_cl_atomic_fetch_or_explicitPU7CLlocalVU7_Atomicjj12memory_order(i32 * %1, i32 %b, i32 %c)
  ret i32 %call
}


declare i32 @_Z28_cl_atomic_fetch_or_explicitPU8CLglobalVU7_Atomicjj12memory_order12memory_scope(i32 *, i32, i32, i32) local_unnamed_addr #0

define spir_func i32 @_Z24atomic_fetch_or_explicitPU3AS1VU7_Atomicjj12memory_order12memory_scope(i32 addrspace(1)* %a, i32 %b, i32 %c, i32 %d) local_unnamed_addr #0 {
  %1 = addrspacecast i32 addrspace(1)* %a to i32 *
  %call = tail call  i32 @_Z28_cl_atomic_fetch_or_explicitPU8CLglobalVU7_Atomicjj12memory_order12memory_scope(i32 * %1, i32 %b, i32 %c, i32 %d)
  ret i32 %call
}


declare i32 @_Z28_cl_atomic_fetch_or_explicitPU7CLlocalVU7_Atomicjj12memory_order12memory_scope(i32 *, i32, i32, i32) local_unnamed_addr #0

define spir_func i32 @_Z24atomic_fetch_or_explicitPU3AS3VU7_Atomicjj12memory_order12memory_scope(i32 addrspace(3)* %a, i32 %b, i32 %c, i32 %d) local_unnamed_addr #0 {
  %1 = addrspacecast i32 addrspace(3)* %a to i32 *
  %call = tail call  i32 @_Z28_cl_atomic_fetch_or_explicitPU7CLlocalVU7_Atomicjj12memory_order12memory_scope(i32 * %1, i32 %b, i32 %c, i32 %d)
  ret i32 %call
}


declare i32 @_Z20_cl_atomic_fetch_xorPU8CLglobalVU7_Atomicjj(i32 *, i32) local_unnamed_addr #0

define spir_func i32 @_Z16atomic_fetch_xorPU3AS1VU7_Atomicjj(i32 addrspace(1)* %a, i32 %b) local_unnamed_addr #0 {
  %1 = addrspacecast i32 addrspace(1)* %a to i32 *
  %call = tail call  i32 @_Z20_cl_atomic_fetch_xorPU8CLglobalVU7_Atomicjj(i32 * %1, i32 %b)
  ret i32 %call
}


declare i32 @_Z20_cl_atomic_fetch_xorPU7CLlocalVU7_Atomicjj(i32 *, i32) local_unnamed_addr #0

define spir_func i32 @_Z16atomic_fetch_xorPU3AS3VU7_Atomicjj(i32 addrspace(3)* %a, i32 %b) local_unnamed_addr #0 {
  %1 = addrspacecast i32 addrspace(3)* %a to i32 *
  %call = tail call  i32 @_Z20_cl_atomic_fetch_xorPU7CLlocalVU7_Atomicjj(i32 * %1, i32 %b)
  ret i32 %call
}


declare i32 @_Z29_cl_atomic_fetch_xor_explicitPU8CLglobalVU7_Atomicjj12memory_order(i32 *, i32, i32) local_unnamed_addr #0

define spir_func i32 @_Z25atomic_fetch_xor_explicitPU3AS1VU7_Atomicjj12memory_order(i32 addrspace(1)* %a, i32 %b, i32 %c) local_unnamed_addr #0 {
  %1 = addrspacecast i32 addrspace(1)* %a to i32 *
  %call = tail call  i32 @_Z29_cl_atomic_fetch_xor_explicitPU8CLglobalVU7_Atomicjj12memory_order(i32 * %1, i32 %b, i32 %c)
  ret i32 %call
}


declare i32 @_Z29_cl_atomic_fetch_xor_explicitPU7CLlocalVU7_Atomicjj12memory_order(i32 *, i32, i32) local_unnamed_addr #0

define spir_func i32 @_Z25atomic_fetch_xor_explicitPU3AS3VU7_Atomicjj12memory_order(i32 addrspace(3)* %a, i32 %b, i32 %c) local_unnamed_addr #0 {
  %1 = addrspacecast i32 addrspace(3)* %a to i32 *
  %call = tail call  i32 @_Z29_cl_atomic_fetch_xor_explicitPU7CLlocalVU7_Atomicjj12memory_order(i32 * %1, i32 %b, i32 %c)
  ret i32 %call
}


declare i32 @_Z29_cl_atomic_fetch_xor_explicitPU8CLglobalVU7_Atomicjj12memory_order12memory_scope(i32 *, i32, i32, i32) local_unnamed_addr #0

define spir_func i32 @_Z25atomic_fetch_xor_explicitPU3AS1VU7_Atomicjj12memory_order12memory_scope(i32 addrspace(1)* %a, i32 %b, i32 %c, i32 %d) local_unnamed_addr #0 {
  %1 = addrspacecast i32 addrspace(1)* %a to i32 *
  %call = tail call  i32 @_Z29_cl_atomic_fetch_xor_explicitPU8CLglobalVU7_Atomicjj12memory_order12memory_scope(i32 * %1, i32 %b, i32 %c, i32 %d)
  ret i32 %call
}


declare i32 @_Z29_cl_atomic_fetch_xor_explicitPU7CLlocalVU7_Atomicjj12memory_order12memory_scope(i32 *, i32, i32, i32) local_unnamed_addr #0

define spir_func i32 @_Z25atomic_fetch_xor_explicitPU3AS3VU7_Atomicjj12memory_order12memory_scope(i32 addrspace(3)* %a, i32 %b, i32 %c, i32 %d) local_unnamed_addr #0 {
  %1 = addrspacecast i32 addrspace(3)* %a to i32 *
  %call = tail call  i32 @_Z29_cl_atomic_fetch_xor_explicitPU7CLlocalVU7_Atomicjj12memory_order12memory_scope(i32 * %1, i32 %b, i32 %c, i32 %d)
  ret i32 %call
}


declare i32 @_Z20_cl_atomic_fetch_andPU8CLglobalVU7_Atomicjj(i32 *, i32) local_unnamed_addr #0

define spir_func i32 @_Z16atomic_fetch_andPU3AS1VU7_Atomicjj(i32 addrspace(1)* %a, i32 %b) local_unnamed_addr #0 {
  %1 = addrspacecast i32 addrspace(1)* %a to i32 *
  %call = tail call  i32 @_Z20_cl_atomic_fetch_andPU8CLglobalVU7_Atomicjj(i32 * %1, i32 %b)
  ret i32 %call
}


declare i32 @_Z20_cl_atomic_fetch_andPU7CLlocalVU7_Atomicjj(i32 *, i32) local_unnamed_addr #0

define spir_func i32 @_Z16atomic_fetch_andPU3AS3VU7_Atomicjj(i32 addrspace(3)* %a, i32 %b) local_unnamed_addr #0 {
  %1 = addrspacecast i32 addrspace(3)* %a to i32 *
  %call = tail call  i32 @_Z20_cl_atomic_fetch_andPU7CLlocalVU7_Atomicjj(i32 * %1, i32 %b)
  ret i32 %call
}


declare i32 @_Z29_cl_atomic_fetch_and_explicitPU8CLglobalVU7_Atomicjj12memory_order(i32 *, i32, i32) local_unnamed_addr #0

define spir_func i32 @_Z25atomic_fetch_and_explicitPU3AS1VU7_Atomicjj12memory_order(i32 addrspace(1)* %a, i32 %b, i32 %c) local_unnamed_addr #0 {
  %1 = addrspacecast i32 addrspace(1)* %a to i32 *
  %call = tail call  i32 @_Z29_cl_atomic_fetch_and_explicitPU8CLglobalVU7_Atomicjj12memory_order(i32 * %1, i32 %b, i32 %c)
  ret i32 %call
}


declare i32 @_Z29_cl_atomic_fetch_and_explicitPU7CLlocalVU7_Atomicjj12memory_order(i32 *, i32, i32) local_unnamed_addr #0

define spir_func i32 @_Z25atomic_fetch_and_explicitPU3AS3VU7_Atomicjj12memory_order(i32 addrspace(3)* %a, i32 %b, i32 %c) local_unnamed_addr #0 {
  %1 = addrspacecast i32 addrspace(3)* %a to i32 *
  %call = tail call  i32 @_Z29_cl_atomic_fetch_and_explicitPU7CLlocalVU7_Atomicjj12memory_order(i32 * %1, i32 %b, i32 %c)
  ret i32 %call
}


declare i32 @_Z29_cl_atomic_fetch_and_explicitPU8CLglobalVU7_Atomicjj12memory_order12memory_scope(i32 *, i32, i32, i32) local_unnamed_addr #0

define spir_func i32 @_Z25atomic_fetch_and_explicitPU3AS1VU7_Atomicjj12memory_order12memory_scope(i32 addrspace(1)* %a, i32 %b, i32 %c, i32 %d) local_unnamed_addr #0 {
  %1 = addrspacecast i32 addrspace(1)* %a to i32 *
  %call = tail call  i32 @_Z29_cl_atomic_fetch_and_explicitPU8CLglobalVU7_Atomicjj12memory_order12memory_scope(i32 * %1, i32 %b, i32 %c, i32 %d)
  ret i32 %call
}


declare i32 @_Z29_cl_atomic_fetch_and_explicitPU7CLlocalVU7_Atomicjj12memory_order12memory_scope(i32 *, i32, i32, i32) local_unnamed_addr #0

define spir_func i32 @_Z25atomic_fetch_and_explicitPU3AS3VU7_Atomicjj12memory_order12memory_scope(i32 addrspace(3)* %a, i32 %b, i32 %c, i32 %d) local_unnamed_addr #0 {
  %1 = addrspacecast i32 addrspace(3)* %a to i32 *
  %call = tail call  i32 @_Z29_cl_atomic_fetch_and_explicitPU7CLlocalVU7_Atomicjj12memory_order12memory_scope(i32 * %1, i32 %b, i32 %c, i32 %d)
  ret i32 %call
}


declare i32 @_Z20_cl_atomic_fetch_minPU8CLglobalVU7_Atomicjj(i32 *, i32) local_unnamed_addr #0

define spir_func i32 @_Z16atomic_fetch_minPU3AS1VU7_Atomicjj(i32 addrspace(1)* %a, i32 %b) local_unnamed_addr #0 {
  %1 = addrspacecast i32 addrspace(1)* %a to i32 *
  %call = tail call  i32 @_Z20_cl_atomic_fetch_minPU8CLglobalVU7_Atomicjj(i32 * %1, i32 %b)
  ret i32 %call
}


declare i32 @_Z20_cl_atomic_fetch_minPU7CLlocalVU7_Atomicjj(i32 *, i32) local_unnamed_addr #0

define spir_func i32 @_Z16atomic_fetch_minPU3AS3VU7_Atomicjj(i32 addrspace(3)* %a, i32 %b) local_unnamed_addr #0 {
  %1 = addrspacecast i32 addrspace(3)* %a to i32 *
  %call = tail call  i32 @_Z20_cl_atomic_fetch_minPU7CLlocalVU7_Atomicjj(i32 * %1, i32 %b)
  ret i32 %call
}


declare i32 @_Z29_cl_atomic_fetch_min_explicitPU8CLglobalVU7_Atomicjj12memory_order(i32 *, i32, i32) local_unnamed_addr #0

define spir_func i32 @_Z25atomic_fetch_min_explicitPU3AS1VU7_Atomicjj12memory_order(i32 addrspace(1)* %a, i32 %b, i32 %c) local_unnamed_addr #0 {
  %1 = addrspacecast i32 addrspace(1)* %a to i32 *
  %call = tail call  i32 @_Z29_cl_atomic_fetch_min_explicitPU8CLglobalVU7_Atomicjj12memory_order(i32 * %1, i32 %b, i32 %c)
  ret i32 %call
}


declare i32 @_Z29_cl_atomic_fetch_min_explicitPU7CLlocalVU7_Atomicjj12memory_order(i32 *, i32, i32) local_unnamed_addr #0

define spir_func i32 @_Z25atomic_fetch_min_explicitPU3AS3VU7_Atomicjj12memory_order(i32 addrspace(3)* %a, i32 %b, i32 %c) local_unnamed_addr #0 {
  %1 = addrspacecast i32 addrspace(3)* %a to i32 *
  %call = tail call  i32 @_Z29_cl_atomic_fetch_min_explicitPU7CLlocalVU7_Atomicjj12memory_order(i32 * %1, i32 %b, i32 %c)
  ret i32 %call
}


declare i32 @_Z29_cl_atomic_fetch_min_explicitPU8CLglobalVU7_Atomicjj12memory_order12memory_scope(i32 *, i32, i32, i32) local_unnamed_addr #0

define spir_func i32 @_Z25atomic_fetch_min_explicitPU3AS1VU7_Atomicjj12memory_order12memory_scope(i32 addrspace(1)* %a, i32 %b, i32 %c, i32 %d) local_unnamed_addr #0 {
  %1 = addrspacecast i32 addrspace(1)* %a to i32 *
  %call = tail call  i32 @_Z29_cl_atomic_fetch_min_explicitPU8CLglobalVU7_Atomicjj12memory_order12memory_scope(i32 * %1, i32 %b, i32 %c, i32 %d)
  ret i32 %call
}


declare i32 @_Z29_cl_atomic_fetch_min_explicitPU7CLlocalVU7_Atomicjj12memory_order12memory_scope(i32 *, i32, i32, i32) local_unnamed_addr #0

define spir_func i32 @_Z25atomic_fetch_min_explicitPU3AS3VU7_Atomicjj12memory_order12memory_scope(i32 addrspace(3)* %a, i32 %b, i32 %c, i32 %d) local_unnamed_addr #0 {
  %1 = addrspacecast i32 addrspace(3)* %a to i32 *
  %call = tail call  i32 @_Z29_cl_atomic_fetch_min_explicitPU7CLlocalVU7_Atomicjj12memory_order12memory_scope(i32 * %1, i32 %b, i32 %c, i32 %d)
  ret i32 %call
}


declare i32 @_Z20_cl_atomic_fetch_maxPU8CLglobalVU7_Atomicjj(i32 *, i32) local_unnamed_addr #0

define spir_func i32 @_Z16atomic_fetch_maxPU3AS1VU7_Atomicjj(i32 addrspace(1)* %a, i32 %b) local_unnamed_addr #0 {
  %1 = addrspacecast i32 addrspace(1)* %a to i32 *
  %call = tail call  i32 @_Z20_cl_atomic_fetch_maxPU8CLglobalVU7_Atomicjj(i32 * %1, i32 %b)
  ret i32 %call
}


declare i32 @_Z20_cl_atomic_fetch_maxPU7CLlocalVU7_Atomicjj(i32 *, i32) local_unnamed_addr #0

define spir_func i32 @_Z16atomic_fetch_maxPU3AS3VU7_Atomicjj(i32 addrspace(3)* %a, i32 %b) local_unnamed_addr #0 {
  %1 = addrspacecast i32 addrspace(3)* %a to i32 *
  %call = tail call  i32 @_Z20_cl_atomic_fetch_maxPU7CLlocalVU7_Atomicjj(i32 * %1, i32 %b)
  ret i32 %call
}


declare i32 @_Z29_cl_atomic_fetch_max_explicitPU8CLglobalVU7_Atomicjj12memory_order(i32 *, i32, i32) local_unnamed_addr #0

define spir_func i32 @_Z25atomic_fetch_max_explicitPU3AS1VU7_Atomicjj12memory_order(i32 addrspace(1)* %a, i32 %b, i32 %c) local_unnamed_addr #0 {
  %1 = addrspacecast i32 addrspace(1)* %a to i32 *
  %call = tail call  i32 @_Z29_cl_atomic_fetch_max_explicitPU8CLglobalVU7_Atomicjj12memory_order(i32 * %1, i32 %b, i32 %c)
  ret i32 %call
}


declare i32 @_Z29_cl_atomic_fetch_max_explicitPU7CLlocalVU7_Atomicjj12memory_order(i32 *, i32, i32) local_unnamed_addr #0

define spir_func i32 @_Z25atomic_fetch_max_explicitPU3AS3VU7_Atomicjj12memory_order(i32 addrspace(3)* %a, i32 %b, i32 %c) local_unnamed_addr #0 {
  %1 = addrspacecast i32 addrspace(3)* %a to i32 *
  %call = tail call  i32 @_Z29_cl_atomic_fetch_max_explicitPU7CLlocalVU7_Atomicjj12memory_order(i32 * %1, i32 %b, i32 %c)
  ret i32 %call
}


declare i32 @_Z29_cl_atomic_fetch_max_explicitPU8CLglobalVU7_Atomicjj12memory_order12memory_scope(i32 *, i32, i32, i32) local_unnamed_addr #0

define spir_func i32 @_Z25atomic_fetch_max_explicitPU3AS1VU7_Atomicjj12memory_order12memory_scope(i32 addrspace(1)* %a, i32 %b, i32 %c, i32 %d) local_unnamed_addr #0 {
  %1 = addrspacecast i32 addrspace(1)* %a to i32 *
  %call = tail call  i32 @_Z29_cl_atomic_fetch_max_explicitPU8CLglobalVU7_Atomicjj12memory_order12memory_scope(i32 * %1, i32 %b, i32 %c, i32 %d)
  ret i32 %call
}


declare i32 @_Z29_cl_atomic_fetch_max_explicitPU7CLlocalVU7_Atomicjj12memory_order12memory_scope(i32 *, i32, i32, i32) local_unnamed_addr #0

define spir_func i32 @_Z25atomic_fetch_max_explicitPU3AS3VU7_Atomicjj12memory_order12memory_scope(i32 addrspace(3)* %a, i32 %b, i32 %c, i32 %d) local_unnamed_addr #0 {
  %1 = addrspacecast i32 addrspace(3)* %a to i32 *
  %call = tail call  i32 @_Z29_cl_atomic_fetch_max_explicitPU7CLlocalVU7_Atomicjj12memory_order12memory_scope(i32 * %1, i32 %b, i32 %c, i32 %d)
  ret i32 %call
}


declare i64 @_Z20_cl_atomic_fetch_addPU8CLglobalVU7_Atomicll(i64 *, i64) local_unnamed_addr #0

define spir_func i64 @_Z16atomic_fetch_addPU3AS1VU7_Atomicll(i64 addrspace(1)* %a, i64 %b) local_unnamed_addr #0 {
  %1 = addrspacecast i64 addrspace(1)* %a to i64 *
  %call = tail call  i64 @_Z20_cl_atomic_fetch_addPU8CLglobalVU7_Atomicll(i64 * %1, i64 %b)
  ret i64 %call
}


declare i64 @_Z20_cl_atomic_fetch_addPU7CLlocalVU7_Atomicll(i64 *, i64) local_unnamed_addr #0

define spir_func i64 @_Z16atomic_fetch_addPU3AS3VU7_Atomicll(i64 addrspace(3)* %a, i64 %b) local_unnamed_addr #0 {
  %1 = addrspacecast i64 addrspace(3)* %a to i64 *
  %call = tail call  i64 @_Z20_cl_atomic_fetch_addPU7CLlocalVU7_Atomicll(i64 * %1, i64 %b)
  ret i64 %call
}


declare i64 @_Z29_cl_atomic_fetch_add_explicitPU8CLglobalVU7_Atomicll12memory_order(i64 *, i64, i32) local_unnamed_addr #0

define spir_func i64 @_Z25atomic_fetch_add_explicitPU3AS1VU7_Atomicll12memory_order(i64 addrspace(1)* %a, i64 %b, i32 %c) local_unnamed_addr #0 {
  %1 = addrspacecast i64 addrspace(1)* %a to i64 *
  %call = tail call  i64 @_Z29_cl_atomic_fetch_add_explicitPU8CLglobalVU7_Atomicll12memory_order(i64 * %1, i64 %b, i32 %c)
  ret i64 %call
}


declare i64 @_Z29_cl_atomic_fetch_add_explicitPU7CLlocalVU7_Atomicll12memory_order(i64 *, i64, i32) local_unnamed_addr #0

define spir_func i64 @_Z25atomic_fetch_add_explicitPU3AS3VU7_Atomicll12memory_order(i64 addrspace(3)* %a, i64 %b, i32 %c) local_unnamed_addr #0 {
  %1 = addrspacecast i64 addrspace(3)* %a to i64 *
  %call = tail call  i64 @_Z29_cl_atomic_fetch_add_explicitPU7CLlocalVU7_Atomicll12memory_order(i64 * %1, i64 %b, i32 %c)
  ret i64 %call
}


declare i64 @_Z29_cl_atomic_fetch_add_explicitPU8CLglobalVU7_Atomicll12memory_order12memory_scope(i64 *, i64, i32, i32) local_unnamed_addr #0

define spir_func i64 @_Z25atomic_fetch_add_explicitPU3AS1VU7_Atomicll12memory_order12memory_scope(i64 addrspace(1)* %a, i64 %b, i32 %c, i32 %d) local_unnamed_addr #0 {
  %1 = addrspacecast i64 addrspace(1)* %a to i64 *
  %call = tail call  i64 @_Z29_cl_atomic_fetch_add_explicitPU8CLglobalVU7_Atomicll12memory_order12memory_scope(i64 * %1, i64 %b, i32 %c, i32 %d)
  ret i64 %call
}


declare i64 @_Z29_cl_atomic_fetch_add_explicitPU7CLlocalVU7_Atomicll12memory_order12memory_scope(i64 *, i64, i32, i32) local_unnamed_addr #0

define spir_func i64 @_Z25atomic_fetch_add_explicitPU3AS3VU7_Atomicll12memory_order12memory_scope(i64 addrspace(3)* %a, i64 %b, i32 %c, i32 %d) local_unnamed_addr #0 {
  %1 = addrspacecast i64 addrspace(3)* %a to i64 *
  %call = tail call  i64 @_Z29_cl_atomic_fetch_add_explicitPU7CLlocalVU7_Atomicll12memory_order12memory_scope(i64 * %1, i64 %b, i32 %c, i32 %d)
  ret i64 %call
}


declare i64 @_Z20_cl_atomic_fetch_subPU8CLglobalVU7_Atomicll(i64 *, i64) local_unnamed_addr #0

define spir_func i64 @_Z16atomic_fetch_subPU3AS1VU7_Atomicll(i64 addrspace(1)* %a, i64 %b) local_unnamed_addr #0 {
  %1 = addrspacecast i64 addrspace(1)* %a to i64 *
  %call = tail call  i64 @_Z20_cl_atomic_fetch_subPU8CLglobalVU7_Atomicll(i64 * %1, i64 %b)
  ret i64 %call
}


declare i64 @_Z20_cl_atomic_fetch_subPU7CLlocalVU7_Atomicll(i64 *, i64) local_unnamed_addr #0

define spir_func i64 @_Z16atomic_fetch_subPU3AS3VU7_Atomicll(i64 addrspace(3)* %a, i64 %b) local_unnamed_addr #0 {
  %1 = addrspacecast i64 addrspace(3)* %a to i64 *
  %call = tail call  i64 @_Z20_cl_atomic_fetch_subPU7CLlocalVU7_Atomicll(i64 * %1, i64 %b)
  ret i64 %call
}


declare i64 @_Z29_cl_atomic_fetch_sub_explicitPU8CLglobalVU7_Atomicll12memory_order(i64 *, i64, i32) local_unnamed_addr #0

define spir_func i64 @_Z25atomic_fetch_sub_explicitPU3AS1VU7_Atomicll12memory_order(i64 addrspace(1)* %a, i64 %b, i32 %c) local_unnamed_addr #0 {
  %1 = addrspacecast i64 addrspace(1)* %a to i64 *
  %call = tail call  i64 @_Z29_cl_atomic_fetch_sub_explicitPU8CLglobalVU7_Atomicll12memory_order(i64 * %1, i64 %b, i32 %c)
  ret i64 %call
}


declare i64 @_Z29_cl_atomic_fetch_sub_explicitPU7CLlocalVU7_Atomicll12memory_order(i64 *, i64, i32) local_unnamed_addr #0

define spir_func i64 @_Z25atomic_fetch_sub_explicitPU3AS3VU7_Atomicll12memory_order(i64 addrspace(3)* %a, i64 %b, i32 %c) local_unnamed_addr #0 {
  %1 = addrspacecast i64 addrspace(3)* %a to i64 *
  %call = tail call  i64 @_Z29_cl_atomic_fetch_sub_explicitPU7CLlocalVU7_Atomicll12memory_order(i64 * %1, i64 %b, i32 %c)
  ret i64 %call
}


declare i64 @_Z29_cl_atomic_fetch_sub_explicitPU8CLglobalVU7_Atomicll12memory_order12memory_scope(i64 *, i64, i32, i32) local_unnamed_addr #0

define spir_func i64 @_Z25atomic_fetch_sub_explicitPU3AS1VU7_Atomicll12memory_order12memory_scope(i64 addrspace(1)* %a, i64 %b, i32 %c, i32 %d) local_unnamed_addr #0 {
  %1 = addrspacecast i64 addrspace(1)* %a to i64 *
  %call = tail call  i64 @_Z29_cl_atomic_fetch_sub_explicitPU8CLglobalVU7_Atomicll12memory_order12memory_scope(i64 * %1, i64 %b, i32 %c, i32 %d)
  ret i64 %call
}


declare i64 @_Z29_cl_atomic_fetch_sub_explicitPU7CLlocalVU7_Atomicll12memory_order12memory_scope(i64 *, i64, i32, i32) local_unnamed_addr #0

define spir_func i64 @_Z25atomic_fetch_sub_explicitPU3AS3VU7_Atomicll12memory_order12memory_scope(i64 addrspace(3)* %a, i64 %b, i32 %c, i32 %d) local_unnamed_addr #0 {
  %1 = addrspacecast i64 addrspace(3)* %a to i64 *
  %call = tail call  i64 @_Z29_cl_atomic_fetch_sub_explicitPU7CLlocalVU7_Atomicll12memory_order12memory_scope(i64 * %1, i64 %b, i32 %c, i32 %d)
  ret i64 %call
}


declare i64 @_Z19_cl_atomic_fetch_orPU8CLglobalVU7_Atomicll(i64 *, i64) local_unnamed_addr #0

define spir_func i64 @_Z15atomic_fetch_orPU3AS1VU7_Atomicll(i64 addrspace(1)* %a, i64 %b) local_unnamed_addr #0 {
  %1 = addrspacecast i64 addrspace(1)* %a to i64 *
  %call = tail call  i64 @_Z19_cl_atomic_fetch_orPU8CLglobalVU7_Atomicll(i64 * %1, i64 %b)
  ret i64 %call
}


declare i64 @_Z19_cl_atomic_fetch_orPU7CLlocalVU7_Atomicll(i64 *, i64) local_unnamed_addr #0

define spir_func i64 @_Z15atomic_fetch_orPU3AS3VU7_Atomicll(i64 addrspace(3)* %a, i64 %b) local_unnamed_addr #0 {
  %1 = addrspacecast i64 addrspace(3)* %a to i64 *
  %call = tail call  i64 @_Z19_cl_atomic_fetch_orPU7CLlocalVU7_Atomicll(i64 * %1, i64 %b)
  ret i64 %call
}


declare i64 @_Z28_cl_atomic_fetch_or_explicitPU8CLglobalVU7_Atomicll12memory_order(i64 *, i64, i32) local_unnamed_addr #0

define spir_func i64 @_Z24atomic_fetch_or_explicitPU3AS1VU7_Atomicll12memory_order(i64 addrspace(1)* %a, i64 %b, i32 %c) local_unnamed_addr #0 {
  %1 = addrspacecast i64 addrspace(1)* %a to i64 *
  %call = tail call  i64 @_Z28_cl_atomic_fetch_or_explicitPU8CLglobalVU7_Atomicll12memory_order(i64 * %1, i64 %b, i32 %c)
  ret i64 %call
}


declare i64 @_Z28_cl_atomic_fetch_or_explicitPU7CLlocalVU7_Atomicll12memory_order(i64 *, i64, i32) local_unnamed_addr #0

define spir_func i64 @_Z24atomic_fetch_or_explicitPU3AS3VU7_Atomicll12memory_order(i64 addrspace(3)* %a, i64 %b, i32 %c) local_unnamed_addr #0 {
  %1 = addrspacecast i64 addrspace(3)* %a to i64 *
  %call = tail call  i64 @_Z28_cl_atomic_fetch_or_explicitPU7CLlocalVU7_Atomicll12memory_order(i64 * %1, i64 %b, i32 %c)
  ret i64 %call
}


declare i64 @_Z28_cl_atomic_fetch_or_explicitPU8CLglobalVU7_Atomicll12memory_order12memory_scope(i64 *, i64, i32, i32) local_unnamed_addr #0

define spir_func i64 @_Z24atomic_fetch_or_explicitPU3AS1VU7_Atomicll12memory_order12memory_scope(i64 addrspace(1)* %a, i64 %b, i32 %c, i32 %d) local_unnamed_addr #0 {
  %1 = addrspacecast i64 addrspace(1)* %a to i64 *
  %call = tail call  i64 @_Z28_cl_atomic_fetch_or_explicitPU8CLglobalVU7_Atomicll12memory_order12memory_scope(i64 * %1, i64 %b, i32 %c, i32 %d)
  ret i64 %call
}


declare i64 @_Z28_cl_atomic_fetch_or_explicitPU7CLlocalVU7_Atomicll12memory_order12memory_scope(i64 *, i64, i32, i32) local_unnamed_addr #0

define spir_func i64 @_Z24atomic_fetch_or_explicitPU3AS3VU7_Atomicll12memory_order12memory_scope(i64 addrspace(3)* %a, i64 %b, i32 %c, i32 %d) local_unnamed_addr #0 {
  %1 = addrspacecast i64 addrspace(3)* %a to i64 *
  %call = tail call  i64 @_Z28_cl_atomic_fetch_or_explicitPU7CLlocalVU7_Atomicll12memory_order12memory_scope(i64 * %1, i64 %b, i32 %c, i32 %d)
  ret i64 %call
}


declare i64 @_Z20_cl_atomic_fetch_xorPU8CLglobalVU7_Atomicll(i64 *, i64) local_unnamed_addr #0

define spir_func i64 @_Z16atomic_fetch_xorPU3AS1VU7_Atomicll(i64 addrspace(1)* %a, i64 %b) local_unnamed_addr #0 {
  %1 = addrspacecast i64 addrspace(1)* %a to i64 *
  %call = tail call  i64 @_Z20_cl_atomic_fetch_xorPU8CLglobalVU7_Atomicll(i64 * %1, i64 %b)
  ret i64 %call
}


declare i64 @_Z20_cl_atomic_fetch_xorPU7CLlocalVU7_Atomicll(i64 *, i64) local_unnamed_addr #0

define spir_func i64 @_Z16atomic_fetch_xorPU3AS3VU7_Atomicll(i64 addrspace(3)* %a, i64 %b) local_unnamed_addr #0 {
  %1 = addrspacecast i64 addrspace(3)* %a to i64 *
  %call = tail call  i64 @_Z20_cl_atomic_fetch_xorPU7CLlocalVU7_Atomicll(i64 * %1, i64 %b)
  ret i64 %call
}


declare i64 @_Z29_cl_atomic_fetch_xor_explicitPU8CLglobalVU7_Atomicll12memory_order(i64 *, i64, i32) local_unnamed_addr #0

define spir_func i64 @_Z25atomic_fetch_xor_explicitPU3AS1VU7_Atomicll12memory_order(i64 addrspace(1)* %a, i64 %b, i32 %c) local_unnamed_addr #0 {
  %1 = addrspacecast i64 addrspace(1)* %a to i64 *
  %call = tail call  i64 @_Z29_cl_atomic_fetch_xor_explicitPU8CLglobalVU7_Atomicll12memory_order(i64 * %1, i64 %b, i32 %c)
  ret i64 %call
}


declare i64 @_Z29_cl_atomic_fetch_xor_explicitPU7CLlocalVU7_Atomicll12memory_order(i64 *, i64, i32) local_unnamed_addr #0

define spir_func i64 @_Z25atomic_fetch_xor_explicitPU3AS3VU7_Atomicll12memory_order(i64 addrspace(3)* %a, i64 %b, i32 %c) local_unnamed_addr #0 {
  %1 = addrspacecast i64 addrspace(3)* %a to i64 *
  %call = tail call  i64 @_Z29_cl_atomic_fetch_xor_explicitPU7CLlocalVU7_Atomicll12memory_order(i64 * %1, i64 %b, i32 %c)
  ret i64 %call
}


declare i64 @_Z29_cl_atomic_fetch_xor_explicitPU8CLglobalVU7_Atomicll12memory_order12memory_scope(i64 *, i64, i32, i32) local_unnamed_addr #0

define spir_func i64 @_Z25atomic_fetch_xor_explicitPU3AS1VU7_Atomicll12memory_order12memory_scope(i64 addrspace(1)* %a, i64 %b, i32 %c, i32 %d) local_unnamed_addr #0 {
  %1 = addrspacecast i64 addrspace(1)* %a to i64 *
  %call = tail call  i64 @_Z29_cl_atomic_fetch_xor_explicitPU8CLglobalVU7_Atomicll12memory_order12memory_scope(i64 * %1, i64 %b, i32 %c, i32 %d)
  ret i64 %call
}


declare i64 @_Z29_cl_atomic_fetch_xor_explicitPU7CLlocalVU7_Atomicll12memory_order12memory_scope(i64 *, i64, i32, i32) local_unnamed_addr #0

define spir_func i64 @_Z25atomic_fetch_xor_explicitPU3AS3VU7_Atomicll12memory_order12memory_scope(i64 addrspace(3)* %a, i64 %b, i32 %c, i32 %d) local_unnamed_addr #0 {
  %1 = addrspacecast i64 addrspace(3)* %a to i64 *
  %call = tail call  i64 @_Z29_cl_atomic_fetch_xor_explicitPU7CLlocalVU7_Atomicll12memory_order12memory_scope(i64 * %1, i64 %b, i32 %c, i32 %d)
  ret i64 %call
}


declare i64 @_Z20_cl_atomic_fetch_andPU8CLglobalVU7_Atomicll(i64 *, i64) local_unnamed_addr #0

define spir_func i64 @_Z16atomic_fetch_andPU3AS1VU7_Atomicll(i64 addrspace(1)* %a, i64 %b) local_unnamed_addr #0 {
  %1 = addrspacecast i64 addrspace(1)* %a to i64 *
  %call = tail call  i64 @_Z20_cl_atomic_fetch_andPU8CLglobalVU7_Atomicll(i64 * %1, i64 %b)
  ret i64 %call
}


declare i64 @_Z20_cl_atomic_fetch_andPU7CLlocalVU7_Atomicll(i64 *, i64) local_unnamed_addr #0

define spir_func i64 @_Z16atomic_fetch_andPU3AS3VU7_Atomicll(i64 addrspace(3)* %a, i64 %b) local_unnamed_addr #0 {
  %1 = addrspacecast i64 addrspace(3)* %a to i64 *
  %call = tail call  i64 @_Z20_cl_atomic_fetch_andPU7CLlocalVU7_Atomicll(i64 * %1, i64 %b)
  ret i64 %call
}


declare i64 @_Z29_cl_atomic_fetch_and_explicitPU8CLglobalVU7_Atomicll12memory_order(i64 *, i64, i32) local_unnamed_addr #0

define spir_func i64 @_Z25atomic_fetch_and_explicitPU3AS1VU7_Atomicll12memory_order(i64 addrspace(1)* %a, i64 %b, i32 %c) local_unnamed_addr #0 {
  %1 = addrspacecast i64 addrspace(1)* %a to i64 *
  %call = tail call  i64 @_Z29_cl_atomic_fetch_and_explicitPU8CLglobalVU7_Atomicll12memory_order(i64 * %1, i64 %b, i32 %c)
  ret i64 %call
}


declare i64 @_Z29_cl_atomic_fetch_and_explicitPU7CLlocalVU7_Atomicll12memory_order(i64 *, i64, i32) local_unnamed_addr #0

define spir_func i64 @_Z25atomic_fetch_and_explicitPU3AS3VU7_Atomicll12memory_order(i64 addrspace(3)* %a, i64 %b, i32 %c) local_unnamed_addr #0 {
  %1 = addrspacecast i64 addrspace(3)* %a to i64 *
  %call = tail call  i64 @_Z29_cl_atomic_fetch_and_explicitPU7CLlocalVU7_Atomicll12memory_order(i64 * %1, i64 %b, i32 %c)
  ret i64 %call
}


declare i64 @_Z29_cl_atomic_fetch_and_explicitPU8CLglobalVU7_Atomicll12memory_order12memory_scope(i64 *, i64, i32, i32) local_unnamed_addr #0

define spir_func i64 @_Z25atomic_fetch_and_explicitPU3AS1VU7_Atomicll12memory_order12memory_scope(i64 addrspace(1)* %a, i64 %b, i32 %c, i32 %d) local_unnamed_addr #0 {
  %1 = addrspacecast i64 addrspace(1)* %a to i64 *
  %call = tail call  i64 @_Z29_cl_atomic_fetch_and_explicitPU8CLglobalVU7_Atomicll12memory_order12memory_scope(i64 * %1, i64 %b, i32 %c, i32 %d)
  ret i64 %call
}


declare i64 @_Z29_cl_atomic_fetch_and_explicitPU7CLlocalVU7_Atomicll12memory_order12memory_scope(i64 *, i64, i32, i32) local_unnamed_addr #0

define spir_func i64 @_Z25atomic_fetch_and_explicitPU3AS3VU7_Atomicll12memory_order12memory_scope(i64 addrspace(3)* %a, i64 %b, i32 %c, i32 %d) local_unnamed_addr #0 {
  %1 = addrspacecast i64 addrspace(3)* %a to i64 *
  %call = tail call  i64 @_Z29_cl_atomic_fetch_and_explicitPU7CLlocalVU7_Atomicll12memory_order12memory_scope(i64 * %1, i64 %b, i32 %c, i32 %d)
  ret i64 %call
}


declare i64 @_Z20_cl_atomic_fetch_minPU8CLglobalVU7_Atomicll(i64 *, i64) local_unnamed_addr #0

define spir_func i64 @_Z16atomic_fetch_minPU3AS1VU7_Atomicll(i64 addrspace(1)* %a, i64 %b) local_unnamed_addr #0 {
  %1 = addrspacecast i64 addrspace(1)* %a to i64 *
  %call = tail call  i64 @_Z20_cl_atomic_fetch_minPU8CLglobalVU7_Atomicll(i64 * %1, i64 %b)
  ret i64 %call
}


declare i64 @_Z20_cl_atomic_fetch_minPU7CLlocalVU7_Atomicll(i64 *, i64) local_unnamed_addr #0

define spir_func i64 @_Z16atomic_fetch_minPU3AS3VU7_Atomicll(i64 addrspace(3)* %a, i64 %b) local_unnamed_addr #0 {
  %1 = addrspacecast i64 addrspace(3)* %a to i64 *
  %call = tail call  i64 @_Z20_cl_atomic_fetch_minPU7CLlocalVU7_Atomicll(i64 * %1, i64 %b)
  ret i64 %call
}


declare i64 @_Z29_cl_atomic_fetch_min_explicitPU8CLglobalVU7_Atomicll12memory_order(i64 *, i64, i32) local_unnamed_addr #0

define spir_func i64 @_Z25atomic_fetch_min_explicitPU3AS1VU7_Atomicll12memory_order(i64 addrspace(1)* %a, i64 %b, i32 %c) local_unnamed_addr #0 {
  %1 = addrspacecast i64 addrspace(1)* %a to i64 *
  %call = tail call  i64 @_Z29_cl_atomic_fetch_min_explicitPU8CLglobalVU7_Atomicll12memory_order(i64 * %1, i64 %b, i32 %c)
  ret i64 %call
}


declare i64 @_Z29_cl_atomic_fetch_min_explicitPU7CLlocalVU7_Atomicll12memory_order(i64 *, i64, i32) local_unnamed_addr #0

define spir_func i64 @_Z25atomic_fetch_min_explicitPU3AS3VU7_Atomicll12memory_order(i64 addrspace(3)* %a, i64 %b, i32 %c) local_unnamed_addr #0 {
  %1 = addrspacecast i64 addrspace(3)* %a to i64 *
  %call = tail call  i64 @_Z29_cl_atomic_fetch_min_explicitPU7CLlocalVU7_Atomicll12memory_order(i64 * %1, i64 %b, i32 %c)
  ret i64 %call
}


declare i64 @_Z29_cl_atomic_fetch_min_explicitPU8CLglobalVU7_Atomicll12memory_order12memory_scope(i64 *, i64, i32, i32) local_unnamed_addr #0

define spir_func i64 @_Z25atomic_fetch_min_explicitPU3AS1VU7_Atomicll12memory_order12memory_scope(i64 addrspace(1)* %a, i64 %b, i32 %c, i32 %d) local_unnamed_addr #0 {
  %1 = addrspacecast i64 addrspace(1)* %a to i64 *
  %call = tail call  i64 @_Z29_cl_atomic_fetch_min_explicitPU8CLglobalVU7_Atomicll12memory_order12memory_scope(i64 * %1, i64 %b, i32 %c, i32 %d)
  ret i64 %call
}


declare i64 @_Z29_cl_atomic_fetch_min_explicitPU7CLlocalVU7_Atomicll12memory_order12memory_scope(i64 *, i64, i32, i32) local_unnamed_addr #0

define spir_func i64 @_Z25atomic_fetch_min_explicitPU3AS3VU7_Atomicll12memory_order12memory_scope(i64 addrspace(3)* %a, i64 %b, i32 %c, i32 %d) local_unnamed_addr #0 {
  %1 = addrspacecast i64 addrspace(3)* %a to i64 *
  %call = tail call  i64 @_Z29_cl_atomic_fetch_min_explicitPU7CLlocalVU7_Atomicll12memory_order12memory_scope(i64 * %1, i64 %b, i32 %c, i32 %d)
  ret i64 %call
}


declare i64 @_Z20_cl_atomic_fetch_maxPU8CLglobalVU7_Atomicll(i64 *, i64) local_unnamed_addr #0

define spir_func i64 @_Z16atomic_fetch_maxPU3AS1VU7_Atomicll(i64 addrspace(1)* %a, i64 %b) local_unnamed_addr #0 {
  %1 = addrspacecast i64 addrspace(1)* %a to i64 *
  %call = tail call  i64 @_Z20_cl_atomic_fetch_maxPU8CLglobalVU7_Atomicll(i64 * %1, i64 %b)
  ret i64 %call
}


declare i64 @_Z20_cl_atomic_fetch_maxPU7CLlocalVU7_Atomicll(i64 *, i64) local_unnamed_addr #0

define spir_func i64 @_Z16atomic_fetch_maxPU3AS3VU7_Atomicll(i64 addrspace(3)* %a, i64 %b) local_unnamed_addr #0 {
  %1 = addrspacecast i64 addrspace(3)* %a to i64 *
  %call = tail call  i64 @_Z20_cl_atomic_fetch_maxPU7CLlocalVU7_Atomicll(i64 * %1, i64 %b)
  ret i64 %call
}


declare i64 @_Z29_cl_atomic_fetch_max_explicitPU8CLglobalVU7_Atomicll12memory_order(i64 *, i64, i32) local_unnamed_addr #0

define spir_func i64 @_Z25atomic_fetch_max_explicitPU3AS1VU7_Atomicll12memory_order(i64 addrspace(1)* %a, i64 %b, i32 %c) local_unnamed_addr #0 {
  %1 = addrspacecast i64 addrspace(1)* %a to i64 *
  %call = tail call  i64 @_Z29_cl_atomic_fetch_max_explicitPU8CLglobalVU7_Atomicll12memory_order(i64 * %1, i64 %b, i32 %c)
  ret i64 %call
}


declare i64 @_Z29_cl_atomic_fetch_max_explicitPU7CLlocalVU7_Atomicll12memory_order(i64 *, i64, i32) local_unnamed_addr #0

define spir_func i64 @_Z25atomic_fetch_max_explicitPU3AS3VU7_Atomicll12memory_order(i64 addrspace(3)* %a, i64 %b, i32 %c) local_unnamed_addr #0 {
  %1 = addrspacecast i64 addrspace(3)* %a to i64 *
  %call = tail call  i64 @_Z29_cl_atomic_fetch_max_explicitPU7CLlocalVU7_Atomicll12memory_order(i64 * %1, i64 %b, i32 %c)
  ret i64 %call
}


declare i64 @_Z29_cl_atomic_fetch_max_explicitPU8CLglobalVU7_Atomicll12memory_order12memory_scope(i64 *, i64, i32, i32) local_unnamed_addr #0

define spir_func i64 @_Z25atomic_fetch_max_explicitPU3AS1VU7_Atomicll12memory_order12memory_scope(i64 addrspace(1)* %a, i64 %b, i32 %c, i32 %d) local_unnamed_addr #0 {
  %1 = addrspacecast i64 addrspace(1)* %a to i64 *
  %call = tail call  i64 @_Z29_cl_atomic_fetch_max_explicitPU8CLglobalVU7_Atomicll12memory_order12memory_scope(i64 * %1, i64 %b, i32 %c, i32 %d)
  ret i64 %call
}


declare i64 @_Z29_cl_atomic_fetch_max_explicitPU7CLlocalVU7_Atomicll12memory_order12memory_scope(i64 *, i64, i32, i32) local_unnamed_addr #0

define spir_func i64 @_Z25atomic_fetch_max_explicitPU3AS3VU7_Atomicll12memory_order12memory_scope(i64 addrspace(3)* %a, i64 %b, i32 %c, i32 %d) local_unnamed_addr #0 {
  %1 = addrspacecast i64 addrspace(3)* %a to i64 *
  %call = tail call  i64 @_Z29_cl_atomic_fetch_max_explicitPU7CLlocalVU7_Atomicll12memory_order12memory_scope(i64 * %1, i64 %b, i32 %c, i32 %d)
  ret i64 %call
}


declare i64 @_Z20_cl_atomic_fetch_addPU8CLglobalVU7_Atomicmm(i64 *, i64) local_unnamed_addr #0

define spir_func i64 @_Z16atomic_fetch_addPU3AS1VU7_Atomicmm(i64 addrspace(1)* %a, i64 %b) local_unnamed_addr #0 {
  %1 = addrspacecast i64 addrspace(1)* %a to i64 *
  %call = tail call  i64 @_Z20_cl_atomic_fetch_addPU8CLglobalVU7_Atomicmm(i64 * %1, i64 %b)
  ret i64 %call
}


declare i64 @_Z20_cl_atomic_fetch_addPU7CLlocalVU7_Atomicmm(i64 *, i64) local_unnamed_addr #0

define spir_func i64 @_Z16atomic_fetch_addPU3AS3VU7_Atomicmm(i64 addrspace(3)* %a, i64 %b) local_unnamed_addr #0 {
  %1 = addrspacecast i64 addrspace(3)* %a to i64 *
  %call = tail call  i64 @_Z20_cl_atomic_fetch_addPU7CLlocalVU7_Atomicmm(i64 * %1, i64 %b)
  ret i64 %call
}


declare i64 @_Z29_cl_atomic_fetch_add_explicitPU8CLglobalVU7_Atomicmm12memory_order(i64 *, i64, i32) local_unnamed_addr #0

define spir_func i64 @_Z25atomic_fetch_add_explicitPU3AS1VU7_Atomicmm12memory_order(i64 addrspace(1)* %a, i64 %b, i32 %c) local_unnamed_addr #0 {
  %1 = addrspacecast i64 addrspace(1)* %a to i64 *
  %call = tail call  i64 @_Z29_cl_atomic_fetch_add_explicitPU8CLglobalVU7_Atomicmm12memory_order(i64 * %1, i64 %b, i32 %c)
  ret i64 %call
}


declare i64 @_Z29_cl_atomic_fetch_add_explicitPU7CLlocalVU7_Atomicmm12memory_order(i64 *, i64, i32) local_unnamed_addr #0

define spir_func i64 @_Z25atomic_fetch_add_explicitPU3AS3VU7_Atomicmm12memory_order(i64 addrspace(3)* %a, i64 %b, i32 %c) local_unnamed_addr #0 {
  %1 = addrspacecast i64 addrspace(3)* %a to i64 *
  %call = tail call  i64 @_Z29_cl_atomic_fetch_add_explicitPU7CLlocalVU7_Atomicmm12memory_order(i64 * %1, i64 %b, i32 %c)
  ret i64 %call
}


declare i64 @_Z29_cl_atomic_fetch_add_explicitPU8CLglobalVU7_Atomicmm12memory_order12memory_scope(i64 *, i64, i32, i32) local_unnamed_addr #0

define spir_func i64 @_Z25atomic_fetch_add_explicitPU3AS1VU7_Atomicmm12memory_order12memory_scope(i64 addrspace(1)* %a, i64 %b, i32 %c, i32 %d) local_unnamed_addr #0 {
  %1 = addrspacecast i64 addrspace(1)* %a to i64 *
  %call = tail call  i64 @_Z29_cl_atomic_fetch_add_explicitPU8CLglobalVU7_Atomicmm12memory_order12memory_scope(i64 * %1, i64 %b, i32 %c, i32 %d)
  ret i64 %call
}


declare i64 @_Z29_cl_atomic_fetch_add_explicitPU7CLlocalVU7_Atomicmm12memory_order12memory_scope(i64 *, i64, i32, i32) local_unnamed_addr #0

define spir_func i64 @_Z25atomic_fetch_add_explicitPU3AS3VU7_Atomicmm12memory_order12memory_scope(i64 addrspace(3)* %a, i64 %b, i32 %c, i32 %d) local_unnamed_addr #0 {
  %1 = addrspacecast i64 addrspace(3)* %a to i64 *
  %call = tail call  i64 @_Z29_cl_atomic_fetch_add_explicitPU7CLlocalVU7_Atomicmm12memory_order12memory_scope(i64 * %1, i64 %b, i32 %c, i32 %d)
  ret i64 %call
}


declare i64 @_Z20_cl_atomic_fetch_subPU8CLglobalVU7_Atomicmm(i64 *, i64) local_unnamed_addr #0

define spir_func i64 @_Z16atomic_fetch_subPU3AS1VU7_Atomicmm(i64 addrspace(1)* %a, i64 %b) local_unnamed_addr #0 {
  %1 = addrspacecast i64 addrspace(1)* %a to i64 *
  %call = tail call  i64 @_Z20_cl_atomic_fetch_subPU8CLglobalVU7_Atomicmm(i64 * %1, i64 %b)
  ret i64 %call
}


declare i64 @_Z20_cl_atomic_fetch_subPU7CLlocalVU7_Atomicmm(i64 *, i64) local_unnamed_addr #0

define spir_func i64 @_Z16atomic_fetch_subPU3AS3VU7_Atomicmm(i64 addrspace(3)* %a, i64 %b) local_unnamed_addr #0 {
  %1 = addrspacecast i64 addrspace(3)* %a to i64 *
  %call = tail call  i64 @_Z20_cl_atomic_fetch_subPU7CLlocalVU7_Atomicmm(i64 * %1, i64 %b)
  ret i64 %call
}


declare i64 @_Z29_cl_atomic_fetch_sub_explicitPU8CLglobalVU7_Atomicmm12memory_order(i64 *, i64, i32) local_unnamed_addr #0

define spir_func i64 @_Z25atomic_fetch_sub_explicitPU3AS1VU7_Atomicmm12memory_order(i64 addrspace(1)* %a, i64 %b, i32 %c) local_unnamed_addr #0 {
  %1 = addrspacecast i64 addrspace(1)* %a to i64 *
  %call = tail call  i64 @_Z29_cl_atomic_fetch_sub_explicitPU8CLglobalVU7_Atomicmm12memory_order(i64 * %1, i64 %b, i32 %c)
  ret i64 %call
}


declare i64 @_Z29_cl_atomic_fetch_sub_explicitPU7CLlocalVU7_Atomicmm12memory_order(i64 *, i64, i32) local_unnamed_addr #0

define spir_func i64 @_Z25atomic_fetch_sub_explicitPU3AS3VU7_Atomicmm12memory_order(i64 addrspace(3)* %a, i64 %b, i32 %c) local_unnamed_addr #0 {
  %1 = addrspacecast i64 addrspace(3)* %a to i64 *
  %call = tail call  i64 @_Z29_cl_atomic_fetch_sub_explicitPU7CLlocalVU7_Atomicmm12memory_order(i64 * %1, i64 %b, i32 %c)
  ret i64 %call
}


declare i64 @_Z29_cl_atomic_fetch_sub_explicitPU8CLglobalVU7_Atomicmm12memory_order12memory_scope(i64 *, i64, i32, i32) local_unnamed_addr #0

define spir_func i64 @_Z25atomic_fetch_sub_explicitPU3AS1VU7_Atomicmm12memory_order12memory_scope(i64 addrspace(1)* %a, i64 %b, i32 %c, i32 %d) local_unnamed_addr #0 {
  %1 = addrspacecast i64 addrspace(1)* %a to i64 *
  %call = tail call  i64 @_Z29_cl_atomic_fetch_sub_explicitPU8CLglobalVU7_Atomicmm12memory_order12memory_scope(i64 * %1, i64 %b, i32 %c, i32 %d)
  ret i64 %call
}


declare i64 @_Z29_cl_atomic_fetch_sub_explicitPU7CLlocalVU7_Atomicmm12memory_order12memory_scope(i64 *, i64, i32, i32) local_unnamed_addr #0

define spir_func i64 @_Z25atomic_fetch_sub_explicitPU3AS3VU7_Atomicmm12memory_order12memory_scope(i64 addrspace(3)* %a, i64 %b, i32 %c, i32 %d) local_unnamed_addr #0 {
  %1 = addrspacecast i64 addrspace(3)* %a to i64 *
  %call = tail call  i64 @_Z29_cl_atomic_fetch_sub_explicitPU7CLlocalVU7_Atomicmm12memory_order12memory_scope(i64 * %1, i64 %b, i32 %c, i32 %d)
  ret i64 %call
}


declare i64 @_Z19_cl_atomic_fetch_orPU8CLglobalVU7_Atomicmm(i64 *, i64) local_unnamed_addr #0

define spir_func i64 @_Z15atomic_fetch_orPU3AS1VU7_Atomicmm(i64 addrspace(1)* %a, i64 %b) local_unnamed_addr #0 {
  %1 = addrspacecast i64 addrspace(1)* %a to i64 *
  %call = tail call  i64 @_Z19_cl_atomic_fetch_orPU8CLglobalVU7_Atomicmm(i64 * %1, i64 %b)
  ret i64 %call
}


declare i64 @_Z19_cl_atomic_fetch_orPU7CLlocalVU7_Atomicmm(i64 *, i64) local_unnamed_addr #0

define spir_func i64 @_Z15atomic_fetch_orPU3AS3VU7_Atomicmm(i64 addrspace(3)* %a, i64 %b) local_unnamed_addr #0 {
  %1 = addrspacecast i64 addrspace(3)* %a to i64 *
  %call = tail call  i64 @_Z19_cl_atomic_fetch_orPU7CLlocalVU7_Atomicmm(i64 * %1, i64 %b)
  ret i64 %call
}


declare i64 @_Z28_cl_atomic_fetch_or_explicitPU8CLglobalVU7_Atomicmm12memory_order(i64 *, i64, i32) local_unnamed_addr #0

define spir_func i64 @_Z24atomic_fetch_or_explicitPU3AS1VU7_Atomicmm12memory_order(i64 addrspace(1)* %a, i64 %b, i32 %c) local_unnamed_addr #0 {
  %1 = addrspacecast i64 addrspace(1)* %a to i64 *
  %call = tail call  i64 @_Z28_cl_atomic_fetch_or_explicitPU8CLglobalVU7_Atomicmm12memory_order(i64 * %1, i64 %b, i32 %c)
  ret i64 %call
}


declare i64 @_Z28_cl_atomic_fetch_or_explicitPU7CLlocalVU7_Atomicmm12memory_order(i64 *, i64, i32) local_unnamed_addr #0

define spir_func i64 @_Z24atomic_fetch_or_explicitPU3AS3VU7_Atomicmm12memory_order(i64 addrspace(3)* %a, i64 %b, i32 %c) local_unnamed_addr #0 {
  %1 = addrspacecast i64 addrspace(3)* %a to i64 *
  %call = tail call  i64 @_Z28_cl_atomic_fetch_or_explicitPU7CLlocalVU7_Atomicmm12memory_order(i64 * %1, i64 %b, i32 %c)
  ret i64 %call
}


declare i64 @_Z28_cl_atomic_fetch_or_explicitPU8CLglobalVU7_Atomicmm12memory_order12memory_scope(i64 *, i64, i32, i32) local_unnamed_addr #0

define spir_func i64 @_Z24atomic_fetch_or_explicitPU3AS1VU7_Atomicmm12memory_order12memory_scope(i64 addrspace(1)* %a, i64 %b, i32 %c, i32 %d) local_unnamed_addr #0 {
  %1 = addrspacecast i64 addrspace(1)* %a to i64 *
  %call = tail call  i64 @_Z28_cl_atomic_fetch_or_explicitPU8CLglobalVU7_Atomicmm12memory_order12memory_scope(i64 * %1, i64 %b, i32 %c, i32 %d)
  ret i64 %call
}


declare i64 @_Z28_cl_atomic_fetch_or_explicitPU7CLlocalVU7_Atomicmm12memory_order12memory_scope(i64 *, i64, i32, i32) local_unnamed_addr #0

define spir_func i64 @_Z24atomic_fetch_or_explicitPU3AS3VU7_Atomicmm12memory_order12memory_scope(i64 addrspace(3)* %a, i64 %b, i32 %c, i32 %d) local_unnamed_addr #0 {
  %1 = addrspacecast i64 addrspace(3)* %a to i64 *
  %call = tail call  i64 @_Z28_cl_atomic_fetch_or_explicitPU7CLlocalVU7_Atomicmm12memory_order12memory_scope(i64 * %1, i64 %b, i32 %c, i32 %d)
  ret i64 %call
}


declare i64 @_Z20_cl_atomic_fetch_xorPU8CLglobalVU7_Atomicmm(i64 *, i64) local_unnamed_addr #0

define spir_func i64 @_Z16atomic_fetch_xorPU3AS1VU7_Atomicmm(i64 addrspace(1)* %a, i64 %b) local_unnamed_addr #0 {
  %1 = addrspacecast i64 addrspace(1)* %a to i64 *
  %call = tail call  i64 @_Z20_cl_atomic_fetch_xorPU8CLglobalVU7_Atomicmm(i64 * %1, i64 %b)
  ret i64 %call
}


declare i64 @_Z20_cl_atomic_fetch_xorPU7CLlocalVU7_Atomicmm(i64 *, i64) local_unnamed_addr #0

define spir_func i64 @_Z16atomic_fetch_xorPU3AS3VU7_Atomicmm(i64 addrspace(3)* %a, i64 %b) local_unnamed_addr #0 {
  %1 = addrspacecast i64 addrspace(3)* %a to i64 *
  %call = tail call  i64 @_Z20_cl_atomic_fetch_xorPU7CLlocalVU7_Atomicmm(i64 * %1, i64 %b)
  ret i64 %call
}


declare i64 @_Z29_cl_atomic_fetch_xor_explicitPU8CLglobalVU7_Atomicmm12memory_order(i64 *, i64, i32) local_unnamed_addr #0

define spir_func i64 @_Z25atomic_fetch_xor_explicitPU3AS1VU7_Atomicmm12memory_order(i64 addrspace(1)* %a, i64 %b, i32 %c) local_unnamed_addr #0 {
  %1 = addrspacecast i64 addrspace(1)* %a to i64 *
  %call = tail call  i64 @_Z29_cl_atomic_fetch_xor_explicitPU8CLglobalVU7_Atomicmm12memory_order(i64 * %1, i64 %b, i32 %c)
  ret i64 %call
}


declare i64 @_Z29_cl_atomic_fetch_xor_explicitPU7CLlocalVU7_Atomicmm12memory_order(i64 *, i64, i32) local_unnamed_addr #0

define spir_func i64 @_Z25atomic_fetch_xor_explicitPU3AS3VU7_Atomicmm12memory_order(i64 addrspace(3)* %a, i64 %b, i32 %c) local_unnamed_addr #0 {
  %1 = addrspacecast i64 addrspace(3)* %a to i64 *
  %call = tail call  i64 @_Z29_cl_atomic_fetch_xor_explicitPU7CLlocalVU7_Atomicmm12memory_order(i64 * %1, i64 %b, i32 %c)
  ret i64 %call
}


declare i64 @_Z29_cl_atomic_fetch_xor_explicitPU8CLglobalVU7_Atomicmm12memory_order12memory_scope(i64 *, i64, i32, i32) local_unnamed_addr #0

define spir_func i64 @_Z25atomic_fetch_xor_explicitPU3AS1VU7_Atomicmm12memory_order12memory_scope(i64 addrspace(1)* %a, i64 %b, i32 %c, i32 %d) local_unnamed_addr #0 {
  %1 = addrspacecast i64 addrspace(1)* %a to i64 *
  %call = tail call  i64 @_Z29_cl_atomic_fetch_xor_explicitPU8CLglobalVU7_Atomicmm12memory_order12memory_scope(i64 * %1, i64 %b, i32 %c, i32 %d)
  ret i64 %call
}


declare i64 @_Z29_cl_atomic_fetch_xor_explicitPU7CLlocalVU7_Atomicmm12memory_order12memory_scope(i64 *, i64, i32, i32) local_unnamed_addr #0

define spir_func i64 @_Z25atomic_fetch_xor_explicitPU3AS3VU7_Atomicmm12memory_order12memory_scope(i64 addrspace(3)* %a, i64 %b, i32 %c, i32 %d) local_unnamed_addr #0 {
  %1 = addrspacecast i64 addrspace(3)* %a to i64 *
  %call = tail call  i64 @_Z29_cl_atomic_fetch_xor_explicitPU7CLlocalVU7_Atomicmm12memory_order12memory_scope(i64 * %1, i64 %b, i32 %c, i32 %d)
  ret i64 %call
}


declare i64 @_Z20_cl_atomic_fetch_andPU8CLglobalVU7_Atomicmm(i64 *, i64) local_unnamed_addr #0

define spir_func i64 @_Z16atomic_fetch_andPU3AS1VU7_Atomicmm(i64 addrspace(1)* %a, i64 %b) local_unnamed_addr #0 {
  %1 = addrspacecast i64 addrspace(1)* %a to i64 *
  %call = tail call  i64 @_Z20_cl_atomic_fetch_andPU8CLglobalVU7_Atomicmm(i64 * %1, i64 %b)
  ret i64 %call
}


declare i64 @_Z20_cl_atomic_fetch_andPU7CLlocalVU7_Atomicmm(i64 *, i64) local_unnamed_addr #0

define spir_func i64 @_Z16atomic_fetch_andPU3AS3VU7_Atomicmm(i64 addrspace(3)* %a, i64 %b) local_unnamed_addr #0 {
  %1 = addrspacecast i64 addrspace(3)* %a to i64 *
  %call = tail call  i64 @_Z20_cl_atomic_fetch_andPU7CLlocalVU7_Atomicmm(i64 * %1, i64 %b)
  ret i64 %call
}


declare i64 @_Z29_cl_atomic_fetch_and_explicitPU8CLglobalVU7_Atomicmm12memory_order(i64 *, i64, i32) local_unnamed_addr #0

define spir_func i64 @_Z25atomic_fetch_and_explicitPU3AS1VU7_Atomicmm12memory_order(i64 addrspace(1)* %a, i64 %b, i32 %c) local_unnamed_addr #0 {
  %1 = addrspacecast i64 addrspace(1)* %a to i64 *
  %call = tail call  i64 @_Z29_cl_atomic_fetch_and_explicitPU8CLglobalVU7_Atomicmm12memory_order(i64 * %1, i64 %b, i32 %c)
  ret i64 %call
}


declare i64 @_Z29_cl_atomic_fetch_and_explicitPU7CLlocalVU7_Atomicmm12memory_order(i64 *, i64, i32) local_unnamed_addr #0

define spir_func i64 @_Z25atomic_fetch_and_explicitPU3AS3VU7_Atomicmm12memory_order(i64 addrspace(3)* %a, i64 %b, i32 %c) local_unnamed_addr #0 {
  %1 = addrspacecast i64 addrspace(3)* %a to i64 *
  %call = tail call  i64 @_Z29_cl_atomic_fetch_and_explicitPU7CLlocalVU7_Atomicmm12memory_order(i64 * %1, i64 %b, i32 %c)
  ret i64 %call
}


declare i64 @_Z29_cl_atomic_fetch_and_explicitPU8CLglobalVU7_Atomicmm12memory_order12memory_scope(i64 *, i64, i32, i32) local_unnamed_addr #0

define spir_func i64 @_Z25atomic_fetch_and_explicitPU3AS1VU7_Atomicmm12memory_order12memory_scope(i64 addrspace(1)* %a, i64 %b, i32 %c, i32 %d) local_unnamed_addr #0 {
  %1 = addrspacecast i64 addrspace(1)* %a to i64 *
  %call = tail call  i64 @_Z29_cl_atomic_fetch_and_explicitPU8CLglobalVU7_Atomicmm12memory_order12memory_scope(i64 * %1, i64 %b, i32 %c, i32 %d)
  ret i64 %call
}


declare i64 @_Z29_cl_atomic_fetch_and_explicitPU7CLlocalVU7_Atomicmm12memory_order12memory_scope(i64 *, i64, i32, i32) local_unnamed_addr #0

define spir_func i64 @_Z25atomic_fetch_and_explicitPU3AS3VU7_Atomicmm12memory_order12memory_scope(i64 addrspace(3)* %a, i64 %b, i32 %c, i32 %d) local_unnamed_addr #0 {
  %1 = addrspacecast i64 addrspace(3)* %a to i64 *
  %call = tail call  i64 @_Z29_cl_atomic_fetch_and_explicitPU7CLlocalVU7_Atomicmm12memory_order12memory_scope(i64 * %1, i64 %b, i32 %c, i32 %d)
  ret i64 %call
}


declare i64 @_Z20_cl_atomic_fetch_minPU8CLglobalVU7_Atomicmm(i64 *, i64) local_unnamed_addr #0

define spir_func i64 @_Z16atomic_fetch_minPU3AS1VU7_Atomicmm(i64 addrspace(1)* %a, i64 %b) local_unnamed_addr #0 {
  %1 = addrspacecast i64 addrspace(1)* %a to i64 *
  %call = tail call  i64 @_Z20_cl_atomic_fetch_minPU8CLglobalVU7_Atomicmm(i64 * %1, i64 %b)
  ret i64 %call
}


declare i64 @_Z20_cl_atomic_fetch_minPU7CLlocalVU7_Atomicmm(i64 *, i64) local_unnamed_addr #0

define spir_func i64 @_Z16atomic_fetch_minPU3AS3VU7_Atomicmm(i64 addrspace(3)* %a, i64 %b) local_unnamed_addr #0 {
  %1 = addrspacecast i64 addrspace(3)* %a to i64 *
  %call = tail call  i64 @_Z20_cl_atomic_fetch_minPU7CLlocalVU7_Atomicmm(i64 * %1, i64 %b)
  ret i64 %call
}


declare i64 @_Z29_cl_atomic_fetch_min_explicitPU8CLglobalVU7_Atomicmm12memory_order(i64 *, i64, i32) local_unnamed_addr #0

define spir_func i64 @_Z25atomic_fetch_min_explicitPU3AS1VU7_Atomicmm12memory_order(i64 addrspace(1)* %a, i64 %b, i32 %c) local_unnamed_addr #0 {
  %1 = addrspacecast i64 addrspace(1)* %a to i64 *
  %call = tail call  i64 @_Z29_cl_atomic_fetch_min_explicitPU8CLglobalVU7_Atomicmm12memory_order(i64 * %1, i64 %b, i32 %c)
  ret i64 %call
}


declare i64 @_Z29_cl_atomic_fetch_min_explicitPU7CLlocalVU7_Atomicmm12memory_order(i64 *, i64, i32) local_unnamed_addr #0

define spir_func i64 @_Z25atomic_fetch_min_explicitPU3AS3VU7_Atomicmm12memory_order(i64 addrspace(3)* %a, i64 %b, i32 %c) local_unnamed_addr #0 {
  %1 = addrspacecast i64 addrspace(3)* %a to i64 *
  %call = tail call  i64 @_Z29_cl_atomic_fetch_min_explicitPU7CLlocalVU7_Atomicmm12memory_order(i64 * %1, i64 %b, i32 %c)
  ret i64 %call
}


declare i64 @_Z29_cl_atomic_fetch_min_explicitPU8CLglobalVU7_Atomicmm12memory_order12memory_scope(i64 *, i64, i32, i32) local_unnamed_addr #0

define spir_func i64 @_Z25atomic_fetch_min_explicitPU3AS1VU7_Atomicmm12memory_order12memory_scope(i64 addrspace(1)* %a, i64 %b, i32 %c, i32 %d) local_unnamed_addr #0 {
  %1 = addrspacecast i64 addrspace(1)* %a to i64 *
  %call = tail call  i64 @_Z29_cl_atomic_fetch_min_explicitPU8CLglobalVU7_Atomicmm12memory_order12memory_scope(i64 * %1, i64 %b, i32 %c, i32 %d)
  ret i64 %call
}


declare i64 @_Z29_cl_atomic_fetch_min_explicitPU7CLlocalVU7_Atomicmm12memory_order12memory_scope(i64 *, i64, i32, i32) local_unnamed_addr #0

define spir_func i64 @_Z25atomic_fetch_min_explicitPU3AS3VU7_Atomicmm12memory_order12memory_scope(i64 addrspace(3)* %a, i64 %b, i32 %c, i32 %d) local_unnamed_addr #0 {
  %1 = addrspacecast i64 addrspace(3)* %a to i64 *
  %call = tail call  i64 @_Z29_cl_atomic_fetch_min_explicitPU7CLlocalVU7_Atomicmm12memory_order12memory_scope(i64 * %1, i64 %b, i32 %c, i32 %d)
  ret i64 %call
}


declare i64 @_Z20_cl_atomic_fetch_maxPU8CLglobalVU7_Atomicmm(i64 *, i64) local_unnamed_addr #0

define spir_func i64 @_Z16atomic_fetch_maxPU3AS1VU7_Atomicmm(i64 addrspace(1)* %a, i64 %b) local_unnamed_addr #0 {
  %1 = addrspacecast i64 addrspace(1)* %a to i64 *
  %call = tail call  i64 @_Z20_cl_atomic_fetch_maxPU8CLglobalVU7_Atomicmm(i64 * %1, i64 %b)
  ret i64 %call
}


declare i64 @_Z20_cl_atomic_fetch_maxPU7CLlocalVU7_Atomicmm(i64 *, i64) local_unnamed_addr #0

define spir_func i64 @_Z16atomic_fetch_maxPU3AS3VU7_Atomicmm(i64 addrspace(3)* %a, i64 %b) local_unnamed_addr #0 {
  %1 = addrspacecast i64 addrspace(3)* %a to i64 *
  %call = tail call  i64 @_Z20_cl_atomic_fetch_maxPU7CLlocalVU7_Atomicmm(i64 * %1, i64 %b)
  ret i64 %call
}


declare i64 @_Z29_cl_atomic_fetch_max_explicitPU8CLglobalVU7_Atomicmm12memory_order(i64 *, i64, i32) local_unnamed_addr #0

define spir_func i64 @_Z25atomic_fetch_max_explicitPU3AS1VU7_Atomicmm12memory_order(i64 addrspace(1)* %a, i64 %b, i32 %c) local_unnamed_addr #0 {
  %1 = addrspacecast i64 addrspace(1)* %a to i64 *
  %call = tail call  i64 @_Z29_cl_atomic_fetch_max_explicitPU8CLglobalVU7_Atomicmm12memory_order(i64 * %1, i64 %b, i32 %c)
  ret i64 %call
}


declare i64 @_Z29_cl_atomic_fetch_max_explicitPU7CLlocalVU7_Atomicmm12memory_order(i64 *, i64, i32) local_unnamed_addr #0

define spir_func i64 @_Z25atomic_fetch_max_explicitPU3AS3VU7_Atomicmm12memory_order(i64 addrspace(3)* %a, i64 %b, i32 %c) local_unnamed_addr #0 {
  %1 = addrspacecast i64 addrspace(3)* %a to i64 *
  %call = tail call  i64 @_Z29_cl_atomic_fetch_max_explicitPU7CLlocalVU7_Atomicmm12memory_order(i64 * %1, i64 %b, i32 %c)
  ret i64 %call
}


declare i64 @_Z29_cl_atomic_fetch_max_explicitPU8CLglobalVU7_Atomicmm12memory_order12memory_scope(i64 *, i64, i32, i32) local_unnamed_addr #0

define spir_func i64 @_Z25atomic_fetch_max_explicitPU3AS1VU7_Atomicmm12memory_order12memory_scope(i64 addrspace(1)* %a, i64 %b, i32 %c, i32 %d) local_unnamed_addr #0 {
  %1 = addrspacecast i64 addrspace(1)* %a to i64 *
  %call = tail call  i64 @_Z29_cl_atomic_fetch_max_explicitPU8CLglobalVU7_Atomicmm12memory_order12memory_scope(i64 * %1, i64 %b, i32 %c, i32 %d)
  ret i64 %call
}


declare i64 @_Z29_cl_atomic_fetch_max_explicitPU7CLlocalVU7_Atomicmm12memory_order12memory_scope(i64 *, i64, i32, i32) local_unnamed_addr #0

define spir_func i64 @_Z25atomic_fetch_max_explicitPU3AS3VU7_Atomicmm12memory_order12memory_scope(i64 addrspace(3)* %a, i64 %b, i32 %c, i32 %d) local_unnamed_addr #0 {
  %1 = addrspacecast i64 addrspace(3)* %a to i64 *
  %call = tail call  i64 @_Z29_cl_atomic_fetch_max_explicitPU7CLlocalVU7_Atomicmm12memory_order12memory_scope(i64 * %1, i64 %b, i32 %c, i32 %d)
  ret i64 %call
}


declare void @_Z15_cl_atomic_initPU8CLglobalVU7_Atomicii(i32 *, i32) local_unnamed_addr #0

define spir_func void @_Z11atomic_initPU3AS1VU7_Atomicii(i32 addrspace(1)* %a, i32 %b) local_unnamed_addr #0 {
  %1 = addrspacecast i32 addrspace(1)* %a to i32 *
  tail call void @_Z15_cl_atomic_initPU8CLglobalVU7_Atomicii(i32 * %1, i32 %b)
  ret void
}


declare void @_Z15_cl_atomic_initPU7CLlocalVU7_Atomicii(i32 *, i32) local_unnamed_addr #0

define spir_func void @_Z11atomic_initPU3AS3VU7_Atomicii(i32 addrspace(3)* %a, i32 %b) local_unnamed_addr #0 {
  %1 = addrspacecast i32 addrspace(3)* %a to i32 *
  tail call void @_Z15_cl_atomic_initPU7CLlocalVU7_Atomicii(i32 * %1, i32 %b)
  ret void
}


declare i32 @_Z34_cl_atomic_compare_exchange_strongPU8CLglobalVU7_AtomiciPU8CLglobalii(i32 *, i32 *, i32) local_unnamed_addr #0

define spir_func i32 @_Z30atomic_compare_exchange_strongPU3AS1VU7_AtomiciPU3AS1ii(i32 addrspace(1)* %a, i32 addrspace(1)* %b, i32 %c) local_unnamed_addr #0 {
  %1 = addrspacecast i32 addrspace(1)* %a to i32 *
  %2 = addrspacecast i32 addrspace(1)* %b to i32 *
  %call = tail call  i32 @_Z34_cl_atomic_compare_exchange_strongPU8CLglobalVU7_AtomiciPU8CLglobalii(i32 * %1, i32 * %2, i32 %c)
  ret i32 %call
}


declare i32 @_Z34_cl_atomic_compare_exchange_strongPU7CLlocalVU7_AtomiciPU7CLlocalii(i32 *, i32 *, i32) local_unnamed_addr #0

define spir_func i32 @_Z30atomic_compare_exchange_strongPU3AS3VU7_AtomiciPU3AS3ii(i32 addrspace(3)* %a, i32 addrspace(3)* %b, i32 %c) local_unnamed_addr #0 {
  %1 = addrspacecast i32 addrspace(3)* %a to i32 *
  %2 = addrspacecast i32 addrspace(3)* %b to i32 *
  %call = tail call  i32 @_Z34_cl_atomic_compare_exchange_strongPU7CLlocalVU7_AtomiciPU7CLlocalii(i32 * %1, i32 * %2, i32 %c)
  ret i32 %call
}


declare i32 @_Z43_cl_atomic_compare_exchange_strong_explicitPU8CLglobalVU7_AtomiciPU8CLglobalii12memory_order12memory_order(i32 *, i32 *, i32, i32, i32) local_unnamed_addr #0

define spir_func i32 @_Z39atomic_compare_exchange_strong_explicitPU3AS1VU7_AtomiciPU3AS1ii12memory_order12memory_order(i32 addrspace(1)* %a, i32 addrspace(1)* %b, i32 %c, i32 %d, i32 %e) local_unnamed_addr #0 {
  %1 = addrspacecast i32 addrspace(1)* %a to i32 *
  %2 = addrspacecast i32 addrspace(1)* %b to i32 *
  %call = tail call  i32 @_Z43_cl_atomic_compare_exchange_strong_explicitPU8CLglobalVU7_AtomiciPU8CLglobalii12memory_order12memory_order(i32 * %1, i32 * %2, i32 %c, i32 %d, i32 %e)
  ret i32 %call
}


declare i32 @_Z43_cl_atomic_compare_exchange_strong_explicitPU7CLlocalVU7_AtomiciPU7CLlocalii12memory_order12memory_order(i32 *, i32 *, i32, i32, i32) local_unnamed_addr #0

define spir_func i32 @_Z39atomic_compare_exchange_strong_explicitPU3AS3VU7_AtomiciPU3AS3ii12memory_order12memory_order(i32 addrspace(3)* %a, i32 addrspace(3)* %b, i32 %c, i32 %d, i32 %e) local_unnamed_addr #0 {
  %1 = addrspacecast i32 addrspace(3)* %a to i32 *
  %2 = addrspacecast i32 addrspace(3)* %b to i32 *
  %call = tail call  i32 @_Z43_cl_atomic_compare_exchange_strong_explicitPU7CLlocalVU7_AtomiciPU7CLlocalii12memory_order12memory_order(i32 * %1, i32 * %2, i32 %c, i32 %d, i32 %e)
  ret i32 %call
}


declare i32 @_Z43_cl_atomic_compare_exchange_strong_explicitPU8CLglobalVU7_AtomiciPU8CLglobalii12memory_order12memory_order12memory_scope(i32 *, i32 *, i32, i32, i32, i32) local_unnamed_addr #0

define spir_func i32 @_Z39atomic_compare_exchange_strong_explicitPU3AS1VU7_AtomiciPU3AS1ii12memory_order12memory_order12memory_scope(i32 addrspace(1)* %a, i32 addrspace(1)* %b, i32 %c, i32 %d, i32 %e, i32 %f) local_unnamed_addr #0 {
  %1 = addrspacecast i32 addrspace(1)* %a to i32 *
  %2 = addrspacecast i32 addrspace(1)* %b to i32 *
  %call = tail call  i32 @_Z43_cl_atomic_compare_exchange_strong_explicitPU8CLglobalVU7_AtomiciPU8CLglobalii12memory_order12memory_order12memory_scope(i32 * %1, i32 * %2, i32 %c, i32 %d, i32 %e, i32 %f)
  ret i32 %call
}


declare i32 @_Z43_cl_atomic_compare_exchange_strong_explicitPU7CLlocalVU7_AtomiciPU7CLlocalii12memory_order12memory_order12memory_scope(i32 *, i32 *, i32, i32, i32, i32) local_unnamed_addr #0

define spir_func i32 @_Z39atomic_compare_exchange_strong_explicitPU3AS3VU7_AtomiciPU3AS3ii12memory_order12memory_order12memory_scope(i32 addrspace(3)* %a, i32 addrspace(3)* %b, i32 %c, i32 %d, i32 %e, i32 %f) local_unnamed_addr #0 {
  %1 = addrspacecast i32 addrspace(3)* %a to i32 *
  %2 = addrspacecast i32 addrspace(3)* %b to i32 *
  %call = tail call  i32 @_Z43_cl_atomic_compare_exchange_strong_explicitPU7CLlocalVU7_AtomiciPU7CLlocalii12memory_order12memory_order12memory_scope(i32 * %1, i32 * %2, i32 %c, i32 %d, i32 %e, i32 %f)
  ret i32 %call
}


declare i32 @_Z16_cl_atomic_storePU8CLglobalVU7_Atomicii(i32 *, i32) local_unnamed_addr #0

define spir_func i32 @_Z12atomic_storePU3AS1VU7_Atomicii(i32 addrspace(1)* %a, i32 %b) local_unnamed_addr #0 {
  %1 = addrspacecast i32 addrspace(1)* %a to i32 *
  %call = tail call  i32 @_Z16_cl_atomic_storePU8CLglobalVU7_Atomicii(i32 * %1, i32 %b)
  ret i32 %call
}


declare i32 @_Z16_cl_atomic_storePU7CLlocalVU7_Atomicii(i32 *, i32) local_unnamed_addr #0

define spir_func i32 @_Z12atomic_storePU3AS3VU7_Atomicii(i32 addrspace(3)* %a, i32 %b) local_unnamed_addr #0 {
  %1 = addrspacecast i32 addrspace(3)* %a to i32 *
  %call = tail call  i32 @_Z16_cl_atomic_storePU7CLlocalVU7_Atomicii(i32 * %1, i32 %b)
  ret i32 %call
}


declare i32 @_Z25_cl_atomic_store_explicitPU8CLglobalVU7_Atomicii12memory_order(i32 *, i32, i32) local_unnamed_addr #0

define spir_func i32 @_Z21atomic_store_explicitPU3AS1VU7_Atomicii12memory_order(i32 addrspace(1)* %a, i32 %b, i32 %c) local_unnamed_addr #0 {
  %1 = addrspacecast i32 addrspace(1)* %a to i32 *
  %call = tail call  i32 @_Z25_cl_atomic_store_explicitPU8CLglobalVU7_Atomicii12memory_order(i32 * %1, i32 %b, i32 %c)
  ret i32 %call
}


declare i32 @_Z25_cl_atomic_store_explicitPU7CLlocalVU7_Atomicii12memory_order(i32 *, i32, i32) local_unnamed_addr #0

define spir_func i32 @_Z21atomic_store_explicitPU3AS3VU7_Atomicii12memory_order(i32 addrspace(3)* %a, i32 %b, i32 %c) local_unnamed_addr #0 {
  %1 = addrspacecast i32 addrspace(3)* %a to i32 *
  %call = tail call  i32 @_Z25_cl_atomic_store_explicitPU7CLlocalVU7_Atomicii12memory_order(i32 * %1, i32 %b, i32 %c)
  ret i32 %call
}


declare i32 @_Z25_cl_atomic_store_explicitPU8CLglobalVU7_Atomicii12memory_order12memory_scope(i32 *, i32, i32, i32) local_unnamed_addr #0

define spir_func i32 @_Z21atomic_store_explicitPU3AS1VU7_Atomicii12memory_order12memory_scope(i32 addrspace(1)* %a, i32 %b, i32 %c, i32 %d) local_unnamed_addr #0 {
  %1 = addrspacecast i32 addrspace(1)* %a to i32 *
  %call = tail call  i32 @_Z25_cl_atomic_store_explicitPU8CLglobalVU7_Atomicii12memory_order12memory_scope(i32 * %1, i32 %b, i32 %c, i32 %d)
  ret i32 %call
}


declare i32 @_Z25_cl_atomic_store_explicitPU7CLlocalVU7_Atomicii12memory_order12memory_scope(i32 *, i32, i32, i32) local_unnamed_addr #0

define spir_func i32 @_Z21atomic_store_explicitPU3AS3VU7_Atomicii12memory_order12memory_scope(i32 addrspace(3)* %a, i32 %b, i32 %c, i32 %d) local_unnamed_addr #0 {
  %1 = addrspacecast i32 addrspace(3)* %a to i32 *
  %call = tail call  i32 @_Z25_cl_atomic_store_explicitPU7CLlocalVU7_Atomicii12memory_order12memory_scope(i32 * %1, i32 %b, i32 %c, i32 %d)
  ret i32 %call
}


declare i32 @_Z15_cl_atomic_loadPU8CLglobalVU7_Atomici(i32 *) local_unnamed_addr #0

define spir_func i32 @_Z11atomic_loadPU3AS1VU7_Atomici(i32 addrspace(1)* %a) local_unnamed_addr #0 {
  %1 = addrspacecast i32 addrspace(1)* %a to i32 *
  %call = tail call  i32 @_Z15_cl_atomic_loadPU8CLglobalVU7_Atomici(i32 * %1)
  ret i32 %call
}


declare i32 @_Z15_cl_atomic_loadPU7CLlocalVU7_Atomici(i32 *) local_unnamed_addr #0

define spir_func i32 @_Z11atomic_loadPU3AS3VU7_Atomici(i32 addrspace(3)* %a) local_unnamed_addr #0 {
  %1 = addrspacecast i32 addrspace(3)* %a to i32 *
  %call = tail call  i32 @_Z15_cl_atomic_loadPU7CLlocalVU7_Atomici(i32 * %1)
  ret i32 %call
}


declare i32 @_Z24_cl_atomic_load_explicitPU8CLglobalVU7_Atomici12memory_order(i32 *, i32) local_unnamed_addr #0

define spir_func i32 @_Z20atomic_load_explicitPU3AS1VU7_Atomici12memory_order(i32 addrspace(1)* %a, i32 %b) local_unnamed_addr #0 {
  %1 = addrspacecast i32 addrspace(1)* %a to i32 *
  %call = tail call  i32 @_Z24_cl_atomic_load_explicitPU8CLglobalVU7_Atomici12memory_order(i32 * %1, i32 %b)
  ret i32 %call
}


declare i32 @_Z24_cl_atomic_load_explicitPU7CLlocalVU7_Atomici12memory_order(i32 *, i32) local_unnamed_addr #0

define spir_func i32 @_Z20atomic_load_explicitPU3AS3VU7_Atomici12memory_order(i32 addrspace(3)* %a, i32 %b) local_unnamed_addr #0 {
  %1 = addrspacecast i32 addrspace(3)* %a to i32 *
  %call = tail call  i32 @_Z24_cl_atomic_load_explicitPU7CLlocalVU7_Atomici12memory_order(i32 * %1, i32 %b)
  ret i32 %call
}


declare i32 @_Z24_cl_atomic_load_explicitPU8CLglobalVU7_Atomici12memory_order12memory_scope(i32 *, i32, i32) local_unnamed_addr #0

define spir_func i32 @_Z20atomic_load_explicitPU3AS1VU7_Atomici12memory_order12memory_scope(i32 addrspace(1)* %a, i32 %b, i32 %c) local_unnamed_addr #0 {
  %1 = addrspacecast i32 addrspace(1)* %a to i32 *
  %call = tail call  i32 @_Z24_cl_atomic_load_explicitPU8CLglobalVU7_Atomici12memory_order12memory_scope(i32 * %1, i32 %b, i32 %c)
  ret i32 %call
}


declare i32 @_Z24_cl_atomic_load_explicitPU7CLlocalVU7_Atomici12memory_order12memory_scope(i32 *, i32, i32) local_unnamed_addr #0

define spir_func i32 @_Z20atomic_load_explicitPU3AS3VU7_Atomici12memory_order12memory_scope(i32 addrspace(3)* %a, i32 %b, i32 %c) local_unnamed_addr #0 {
  %1 = addrspacecast i32 addrspace(3)* %a to i32 *
  %call = tail call  i32 @_Z24_cl_atomic_load_explicitPU7CLlocalVU7_Atomici12memory_order12memory_scope(i32 * %1, i32 %b, i32 %c)
  ret i32 %call
}


declare i32 @_Z32_cl_atomic_compare_exchange_weakPU8CLglobalVU7_AtomiciPU8CLglobalii(i32 *, i32 *, i32) local_unnamed_addr #0

define spir_func i32 @_Z28atomic_compare_exchange_weakPU3AS1VU7_AtomiciPU3AS1ii(i32 addrspace(1)* %a, i32 addrspace(1)* %b, i32 %c) local_unnamed_addr #0 {
  %1 = addrspacecast i32 addrspace(1)* %a to i32 *
  %2 = addrspacecast i32 addrspace(1)* %b to i32 *
  %call = tail call  i32 @_Z32_cl_atomic_compare_exchange_weakPU8CLglobalVU7_AtomiciPU8CLglobalii(i32 * %1, i32 * %2, i32 %c)
  ret i32 %call
}


declare i32 @_Z32_cl_atomic_compare_exchange_weakPU7CLlocalVU7_AtomiciPU7CLlocalii(i32 *, i32 *, i32) local_unnamed_addr #0

define spir_func i32 @_Z28atomic_compare_exchange_weakPU3AS3VU7_AtomiciPU3AS3ii(i32 addrspace(3)* %a, i32 addrspace(3)* %b, i32 %c) local_unnamed_addr #0 {
  %1 = addrspacecast i32 addrspace(3)* %a to i32 *
  %2 = addrspacecast i32 addrspace(3)* %b to i32 *
  %call = tail call  i32 @_Z32_cl_atomic_compare_exchange_weakPU7CLlocalVU7_AtomiciPU7CLlocalii(i32 * %1, i32 * %2, i32 %c)
  ret i32 %call
}


declare i32 @_Z41_cl_atomic_compare_exchange_weak_explicitPU8CLglobalVU7_AtomiciPU8CLglobalii12memory_order12memory_order(i32 *, i32 *, i32, i32, i32) local_unnamed_addr #0

define spir_func i32 @_Z37atomic_compare_exchange_weak_explicitPU3AS1VU7_AtomiciPU3AS1ii12memory_order12memory_order(i32 addrspace(1)* %a, i32 addrspace(1)* %b, i32 %c, i32 %d, i32 %e) local_unnamed_addr #0 {
  %1 = addrspacecast i32 addrspace(1)* %a to i32 *
  %2 = addrspacecast i32 addrspace(1)* %b to i32 *
  %call = tail call  i32 @_Z41_cl_atomic_compare_exchange_weak_explicitPU8CLglobalVU7_AtomiciPU8CLglobalii12memory_order12memory_order(i32 * %1, i32 * %2, i32 %c, i32 %d, i32 %e)
  ret i32 %call
}


declare i32 @_Z41_cl_atomic_compare_exchange_weak_explicitPU7CLlocalVU7_AtomiciPU7CLlocalii12memory_order12memory_order(i32 *, i32 *, i32, i32, i32) local_unnamed_addr #0

define spir_func i32 @_Z37atomic_compare_exchange_weak_explicitPU3AS3VU7_AtomiciPU3AS3ii12memory_order12memory_order(i32 addrspace(3)* %a, i32 addrspace(3)* %b, i32 %c, i32 %d, i32 %e) local_unnamed_addr #0 {
  %1 = addrspacecast i32 addrspace(3)* %a to i32 *
  %2 = addrspacecast i32 addrspace(3)* %b to i32 *
  %call = tail call  i32 @_Z41_cl_atomic_compare_exchange_weak_explicitPU7CLlocalVU7_AtomiciPU7CLlocalii12memory_order12memory_order(i32 * %1, i32 * %2, i32 %c, i32 %d, i32 %e)
  ret i32 %call
}


declare i32 @_Z41_cl_atomic_compare_exchange_weak_explicitPU8CLglobalVU7_AtomiciPU8CLglobalii12memory_order12memory_order12memory_scope(i32 *, i32 *, i32, i32, i32, i32) local_unnamed_addr #0

define spir_func i32 @_Z37atomic_compare_exchange_weak_explicitPU3AS1VU7_AtomiciPU3AS1ii12memory_order12memory_order12memory_scope(i32 addrspace(1)* %a, i32 addrspace(1)* %b, i32 %c, i32 %d, i32 %e, i32 %f) local_unnamed_addr #0 {
  %1 = addrspacecast i32 addrspace(1)* %a to i32 *
  %2 = addrspacecast i32 addrspace(1)* %b to i32 *
  %call = tail call  i32 @_Z41_cl_atomic_compare_exchange_weak_explicitPU8CLglobalVU7_AtomiciPU8CLglobalii12memory_order12memory_order12memory_scope(i32 * %1, i32 * %2, i32 %c, i32 %d, i32 %e, i32 %f)
  ret i32 %call
}


declare i32 @_Z41_cl_atomic_compare_exchange_weak_explicitPU7CLlocalVU7_AtomiciPU7CLlocalii12memory_order12memory_order12memory_scope(i32 *, i32 *, i32, i32, i32, i32) local_unnamed_addr #0

define spir_func i32 @_Z37atomic_compare_exchange_weak_explicitPU3AS3VU7_AtomiciPU3AS3ii12memory_order12memory_order12memory_scope(i32 addrspace(3)* %a, i32 addrspace(3)* %b, i32 %c, i32 %d, i32 %e, i32 %f) local_unnamed_addr #0 {
  %1 = addrspacecast i32 addrspace(3)* %a to i32 *
  %2 = addrspacecast i32 addrspace(3)* %b to i32 *
  %call = tail call  i32 @_Z41_cl_atomic_compare_exchange_weak_explicitPU7CLlocalVU7_AtomiciPU7CLlocalii12memory_order12memory_order12memory_scope(i32 * %1, i32 * %2, i32 %c, i32 %d, i32 %e, i32 %f)
  ret i32 %call
}


declare i32 @_Z19_cl_atomic_exchangePU8CLglobalVU7_Atomicii(i32 *, i32) local_unnamed_addr #0

define spir_func i32 @_Z15atomic_exchangePU3AS1VU7_Atomicii(i32 addrspace(1)* %a, i32 %b) local_unnamed_addr #0 {
  %1 = addrspacecast i32 addrspace(1)* %a to i32 *
  %call = tail call  i32 @_Z19_cl_atomic_exchangePU8CLglobalVU7_Atomicii(i32 * %1, i32 %b)
  ret i32 %call
}


declare i32 @_Z19_cl_atomic_exchangePU7CLlocalVU7_Atomicii(i32 *, i32) local_unnamed_addr #0

define spir_func i32 @_Z15atomic_exchangePU3AS3VU7_Atomicii(i32 addrspace(3)* %a, i32 %b) local_unnamed_addr #0 {
  %1 = addrspacecast i32 addrspace(3)* %a to i32 *
  %call = tail call  i32 @_Z19_cl_atomic_exchangePU7CLlocalVU7_Atomicii(i32 * %1, i32 %b)
  ret i32 %call
}


declare i32 @_Z28_cl_atomic_exchange_explicitPU8CLglobalVU7_Atomicii12memory_order(i32 *, i32, i32) local_unnamed_addr #0

define spir_func i32 @_Z24atomic_exchange_explicitPU3AS1VU7_Atomicii12memory_order(i32 addrspace(1)* %a, i32 %b, i32 %c) local_unnamed_addr #0 {
  %1 = addrspacecast i32 addrspace(1)* %a to i32 *
  %call = tail call  i32 @_Z28_cl_atomic_exchange_explicitPU8CLglobalVU7_Atomicii12memory_order(i32 * %1, i32 %b, i32 %c)
  ret i32 %call
}


declare i32 @_Z28_cl_atomic_exchange_explicitPU7CLlocalVU7_Atomicii12memory_order(i32 *, i32, i32) local_unnamed_addr #0

define spir_func i32 @_Z24atomic_exchange_explicitPU3AS3VU7_Atomicii12memory_order(i32 addrspace(3)* %a, i32 %b, i32 %c) local_unnamed_addr #0 {
  %1 = addrspacecast i32 addrspace(3)* %a to i32 *
  %call = tail call  i32 @_Z28_cl_atomic_exchange_explicitPU7CLlocalVU7_Atomicii12memory_order(i32 * %1, i32 %b, i32 %c)
  ret i32 %call
}


declare i32 @_Z28_cl_atomic_exchange_explicitPU8CLglobalVU7_Atomicii12memory_order12memory_scope(i32 *, i32, i32, i32) local_unnamed_addr #0

define spir_func i32 @_Z24atomic_exchange_explicitPU3AS1VU7_Atomicii12memory_order12memory_scope(i32 addrspace(1)* %a, i32 %b, i32 %c, i32 %d) local_unnamed_addr #0 {
  %1 = addrspacecast i32 addrspace(1)* %a to i32 *
  %call = tail call  i32 @_Z28_cl_atomic_exchange_explicitPU8CLglobalVU7_Atomicii12memory_order12memory_scope(i32 * %1, i32 %b, i32 %c, i32 %d)
  ret i32 %call
}


declare i32 @_Z28_cl_atomic_exchange_explicitPU7CLlocalVU7_Atomicii12memory_order12memory_scope(i32 *, i32, i32, i32) local_unnamed_addr #0

define spir_func i32 @_Z24atomic_exchange_explicitPU3AS3VU7_Atomicii12memory_order12memory_scope(i32 addrspace(3)* %a, i32 %b, i32 %c, i32 %d) local_unnamed_addr #0 {
  %1 = addrspacecast i32 addrspace(3)* %a to i32 *
  %call = tail call  i32 @_Z28_cl_atomic_exchange_explicitPU7CLlocalVU7_Atomicii12memory_order12memory_scope(i32 * %1, i32 %b, i32 %c, i32 %d)
  ret i32 %call
}


declare void @_Z15_cl_atomic_initPU8CLglobalVU7_Atomicjj(i32 *, i32) local_unnamed_addr #0

define spir_func void @_Z11atomic_initPU3AS1VU7_Atomicjj(i32 addrspace(1)* %a, i32 %b) local_unnamed_addr #0 {
  %1 = addrspacecast i32 addrspace(1)* %a to i32 *
  tail call void @_Z15_cl_atomic_initPU8CLglobalVU7_Atomicjj(i32 * %1, i32 %b)
  ret void
}


declare void @_Z15_cl_atomic_initPU7CLlocalVU7_Atomicjj(i32 *, i32) local_unnamed_addr #0

define spir_func void @_Z11atomic_initPU3AS3VU7_Atomicjj(i32 addrspace(3)* %a, i32 %b) local_unnamed_addr #0 {
  %1 = addrspacecast i32 addrspace(3)* %a to i32 *
  tail call void @_Z15_cl_atomic_initPU7CLlocalVU7_Atomicjj(i32 * %1, i32 %b)
  ret void
}


declare i32 @_Z34_cl_atomic_compare_exchange_strongPU8CLglobalVU7_AtomicjPU8CLglobaljj(i32 *, i32 *, i32) local_unnamed_addr #0

define spir_func i32 @_Z30atomic_compare_exchange_strongPU3AS1VU7_AtomicjPU3AS1jj(i32 addrspace(1)* %a, i32 addrspace(1)* %b, i32 %c) local_unnamed_addr #0 {
  %1 = addrspacecast i32 addrspace(1)* %a to i32 *
  %2 = addrspacecast i32 addrspace(1)* %b to i32 *
  %call = tail call  i32 @_Z34_cl_atomic_compare_exchange_strongPU8CLglobalVU7_AtomicjPU8CLglobaljj(i32 * %1, i32 * %2, i32 %c)
  ret i32 %call
}


declare i32 @_Z34_cl_atomic_compare_exchange_strongPU7CLlocalVU7_AtomicjPU7CLlocaljj(i32 *, i32 *, i32) local_unnamed_addr #0

define spir_func i32 @_Z30atomic_compare_exchange_strongPU3AS3VU7_AtomicjPU3AS3jj(i32 addrspace(3)* %a, i32 addrspace(3)* %b, i32 %c) local_unnamed_addr #0 {
  %1 = addrspacecast i32 addrspace(3)* %a to i32 *
  %2 = addrspacecast i32 addrspace(3)* %b to i32 *
  %call = tail call  i32 @_Z34_cl_atomic_compare_exchange_strongPU7CLlocalVU7_AtomicjPU7CLlocaljj(i32 * %1, i32 * %2, i32 %c)
  ret i32 %call
}


declare i32 @_Z43_cl_atomic_compare_exchange_strong_explicitPU8CLglobalVU7_AtomicjPU8CLglobaljj12memory_order12memory_order(i32 *, i32 *, i32, i32, i32) local_unnamed_addr #0

define spir_func i32 @_Z39atomic_compare_exchange_strong_explicitPU3AS1VU7_AtomicjPU3AS1jj12memory_order12memory_order(i32 addrspace(1)* %a, i32 addrspace(1)* %b, i32 %c, i32 %d, i32 %e) local_unnamed_addr #0 {
  %1 = addrspacecast i32 addrspace(1)* %a to i32 *
  %2 = addrspacecast i32 addrspace(1)* %b to i32 *
  %call = tail call  i32 @_Z43_cl_atomic_compare_exchange_strong_explicitPU8CLglobalVU7_AtomicjPU8CLglobaljj12memory_order12memory_order(i32 * %1, i32 * %2, i32 %c, i32 %d, i32 %e)
  ret i32 %call
}


declare i32 @_Z43_cl_atomic_compare_exchange_strong_explicitPU7CLlocalVU7_AtomicjPU7CLlocaljj12memory_order12memory_order(i32 *, i32 *, i32, i32, i32) local_unnamed_addr #0

define spir_func i32 @_Z39atomic_compare_exchange_strong_explicitPU3AS3VU7_AtomicjPU3AS3jj12memory_order12memory_order(i32 addrspace(3)* %a, i32 addrspace(3)* %b, i32 %c, i32 %d, i32 %e) local_unnamed_addr #0 {
  %1 = addrspacecast i32 addrspace(3)* %a to i32 *
  %2 = addrspacecast i32 addrspace(3)* %b to i32 *
  %call = tail call  i32 @_Z43_cl_atomic_compare_exchange_strong_explicitPU7CLlocalVU7_AtomicjPU7CLlocaljj12memory_order12memory_order(i32 * %1, i32 * %2, i32 %c, i32 %d, i32 %e)
  ret i32 %call
}


declare i32 @_Z43_cl_atomic_compare_exchange_strong_explicitPU8CLglobalVU7_AtomicjPU8CLglobaljj12memory_order12memory_order12memory_scope(i32 *, i32 *, i32, i32, i32, i32) local_unnamed_addr #0

define spir_func i32 @_Z39atomic_compare_exchange_strong_explicitPU3AS1VU7_AtomicjPU3AS1jj12memory_order12memory_order12memory_scope(i32 addrspace(1)* %a, i32 addrspace(1)* %b, i32 %c, i32 %d, i32 %e, i32 %f) local_unnamed_addr #0 {
  %1 = addrspacecast i32 addrspace(1)* %a to i32 *
  %2 = addrspacecast i32 addrspace(1)* %b to i32 *
  %call = tail call  i32 @_Z43_cl_atomic_compare_exchange_strong_explicitPU8CLglobalVU7_AtomicjPU8CLglobaljj12memory_order12memory_order12memory_scope(i32 * %1, i32 * %2, i32 %c, i32 %d, i32 %e, i32 %f)
  ret i32 %call
}


declare i32 @_Z43_cl_atomic_compare_exchange_strong_explicitPU7CLlocalVU7_AtomicjPU7CLlocaljj12memory_order12memory_order12memory_scope(i32 *, i32 *, i32, i32, i32, i32) local_unnamed_addr #0

define spir_func i32 @_Z39atomic_compare_exchange_strong_explicitPU3AS3VU7_AtomicjPU3AS3jj12memory_order12memory_order12memory_scope(i32 addrspace(3)* %a, i32 addrspace(3)* %b, i32 %c, i32 %d, i32 %e, i32 %f) local_unnamed_addr #0 {
  %1 = addrspacecast i32 addrspace(3)* %a to i32 *
  %2 = addrspacecast i32 addrspace(3)* %b to i32 *
  %call = tail call  i32 @_Z43_cl_atomic_compare_exchange_strong_explicitPU7CLlocalVU7_AtomicjPU7CLlocaljj12memory_order12memory_order12memory_scope(i32 * %1, i32 * %2, i32 %c, i32 %d, i32 %e, i32 %f)
  ret i32 %call
}


declare i32 @_Z16_cl_atomic_storePU8CLglobalVU7_Atomicjj(i32 *, i32) local_unnamed_addr #0

define spir_func i32 @_Z12atomic_storePU3AS1VU7_Atomicjj(i32 addrspace(1)* %a, i32 %b) local_unnamed_addr #0 {
  %1 = addrspacecast i32 addrspace(1)* %a to i32 *
  %call = tail call  i32 @_Z16_cl_atomic_storePU8CLglobalVU7_Atomicjj(i32 * %1, i32 %b)
  ret i32 %call
}


declare i32 @_Z16_cl_atomic_storePU7CLlocalVU7_Atomicjj(i32 *, i32) local_unnamed_addr #0

define spir_func i32 @_Z12atomic_storePU3AS3VU7_Atomicjj(i32 addrspace(3)* %a, i32 %b) local_unnamed_addr #0 {
  %1 = addrspacecast i32 addrspace(3)* %a to i32 *
  %call = tail call  i32 @_Z16_cl_atomic_storePU7CLlocalVU7_Atomicjj(i32 * %1, i32 %b)
  ret i32 %call
}


declare i32 @_Z25_cl_atomic_store_explicitPU8CLglobalVU7_Atomicjj12memory_order(i32 *, i32, i32) local_unnamed_addr #0

define spir_func i32 @_Z21atomic_store_explicitPU3AS1VU7_Atomicjj12memory_order(i32 addrspace(1)* %a, i32 %b, i32 %c) local_unnamed_addr #0 {
  %1 = addrspacecast i32 addrspace(1)* %a to i32 *
  %call = tail call  i32 @_Z25_cl_atomic_store_explicitPU8CLglobalVU7_Atomicjj12memory_order(i32 * %1, i32 %b, i32 %c)
  ret i32 %call
}


declare i32 @_Z25_cl_atomic_store_explicitPU7CLlocalVU7_Atomicjj12memory_order(i32 *, i32, i32) local_unnamed_addr #0

define spir_func i32 @_Z21atomic_store_explicitPU3AS3VU7_Atomicjj12memory_order(i32 addrspace(3)* %a, i32 %b, i32 %c) local_unnamed_addr #0 {
  %1 = addrspacecast i32 addrspace(3)* %a to i32 *
  %call = tail call  i32 @_Z25_cl_atomic_store_explicitPU7CLlocalVU7_Atomicjj12memory_order(i32 * %1, i32 %b, i32 %c)
  ret i32 %call
}


declare i32 @_Z25_cl_atomic_store_explicitPU8CLglobalVU7_Atomicjj12memory_order12memory_scope(i32 *, i32, i32, i32) local_unnamed_addr #0

define spir_func i32 @_Z21atomic_store_explicitPU3AS1VU7_Atomicjj12memory_order12memory_scope(i32 addrspace(1)* %a, i32 %b, i32 %c, i32 %d) local_unnamed_addr #0 {
  %1 = addrspacecast i32 addrspace(1)* %a to i32 *
  %call = tail call  i32 @_Z25_cl_atomic_store_explicitPU8CLglobalVU7_Atomicjj12memory_order12memory_scope(i32 * %1, i32 %b, i32 %c, i32 %d)
  ret i32 %call
}


declare i32 @_Z25_cl_atomic_store_explicitPU7CLlocalVU7_Atomicjj12memory_order12memory_scope(i32 *, i32, i32, i32) local_unnamed_addr #0

define spir_func i32 @_Z21atomic_store_explicitPU3AS3VU7_Atomicjj12memory_order12memory_scope(i32 addrspace(3)* %a, i32 %b, i32 %c, i32 %d) local_unnamed_addr #0 {
  %1 = addrspacecast i32 addrspace(3)* %a to i32 *
  %call = tail call  i32 @_Z25_cl_atomic_store_explicitPU7CLlocalVU7_Atomicjj12memory_order12memory_scope(i32 * %1, i32 %b, i32 %c, i32 %d)
  ret i32 %call
}


declare i32 @_Z15_cl_atomic_loadPU8CLglobalVU7_Atomicj(i32 *) local_unnamed_addr #0

define spir_func i32 @_Z11atomic_loadPU3AS1VU7_Atomicj(i32 addrspace(1)* %a) local_unnamed_addr #0 {
  %1 = addrspacecast i32 addrspace(1)* %a to i32 *
  %call = tail call  i32 @_Z15_cl_atomic_loadPU8CLglobalVU7_Atomicj(i32 * %1)
  ret i32 %call
}


declare i32 @_Z15_cl_atomic_loadPU7CLlocalVU7_Atomicj(i32 *) local_unnamed_addr #0

define spir_func i32 @_Z11atomic_loadPU3AS3VU7_Atomicj(i32 addrspace(3)* %a) local_unnamed_addr #0 {
  %1 = addrspacecast i32 addrspace(3)* %a to i32 *
  %call = tail call  i32 @_Z15_cl_atomic_loadPU7CLlocalVU7_Atomicj(i32 * %1)
  ret i32 %call
}


declare i32 @_Z24_cl_atomic_load_explicitPU8CLglobalVU7_Atomicj12memory_order(i32 *, i32) local_unnamed_addr #0

define spir_func i32 @_Z20atomic_load_explicitPU3AS1VU7_Atomicj12memory_order(i32 addrspace(1)* %a, i32 %b) local_unnamed_addr #0 {
  %1 = addrspacecast i32 addrspace(1)* %a to i32 *
  %call = tail call  i32 @_Z24_cl_atomic_load_explicitPU8CLglobalVU7_Atomicj12memory_order(i32 * %1, i32 %b)
  ret i32 %call
}


declare i32 @_Z24_cl_atomic_load_explicitPU7CLlocalVU7_Atomicj12memory_order(i32 *, i32) local_unnamed_addr #0

define spir_func i32 @_Z20atomic_load_explicitPU3AS3VU7_Atomicj12memory_order(i32 addrspace(3)* %a, i32 %b) local_unnamed_addr #0 {
  %1 = addrspacecast i32 addrspace(3)* %a to i32 *
  %call = tail call  i32 @_Z24_cl_atomic_load_explicitPU7CLlocalVU7_Atomicj12memory_order(i32 * %1, i32 %b)
  ret i32 %call
}


declare i32 @_Z24_cl_atomic_load_explicitPU8CLglobalVU7_Atomicj12memory_order12memory_scope(i32 *, i32, i32) local_unnamed_addr #0

define spir_func i32 @_Z20atomic_load_explicitPU3AS1VU7_Atomicj12memory_order12memory_scope(i32 addrspace(1)* %a, i32 %b, i32 %c) local_unnamed_addr #0 {
  %1 = addrspacecast i32 addrspace(1)* %a to i32 *
  %call = tail call  i32 @_Z24_cl_atomic_load_explicitPU8CLglobalVU7_Atomicj12memory_order12memory_scope(i32 * %1, i32 %b, i32 %c)
  ret i32 %call
}


declare i32 @_Z24_cl_atomic_load_explicitPU7CLlocalVU7_Atomicj12memory_order12memory_scope(i32 *, i32, i32) local_unnamed_addr #0

define spir_func i32 @_Z20atomic_load_explicitPU3AS3VU7_Atomicj12memory_order12memory_scope(i32 addrspace(3)* %a, i32 %b, i32 %c) local_unnamed_addr #0 {
  %1 = addrspacecast i32 addrspace(3)* %a to i32 *
  %call = tail call  i32 @_Z24_cl_atomic_load_explicitPU7CLlocalVU7_Atomicj12memory_order12memory_scope(i32 * %1, i32 %b, i32 %c)
  ret i32 %call
}


declare i32 @_Z32_cl_atomic_compare_exchange_weakPU8CLglobalVU7_AtomicjPU8CLglobaljj(i32 *, i32 *, i32) local_unnamed_addr #0

define spir_func i32 @_Z28atomic_compare_exchange_weakPU3AS1VU7_AtomicjPU3AS1jj(i32 addrspace(1)* %a, i32 addrspace(1)* %b, i32 %c) local_unnamed_addr #0 {
  %1 = addrspacecast i32 addrspace(1)* %a to i32 *
  %2 = addrspacecast i32 addrspace(1)* %b to i32 *
  %call = tail call  i32 @_Z32_cl_atomic_compare_exchange_weakPU8CLglobalVU7_AtomicjPU8CLglobaljj(i32 * %1, i32 * %2, i32 %c)
  ret i32 %call
}


declare i32 @_Z32_cl_atomic_compare_exchange_weakPU7CLlocalVU7_AtomicjPU7CLlocaljj(i32 *, i32 *, i32) local_unnamed_addr #0

define spir_func i32 @_Z28atomic_compare_exchange_weakPU3AS3VU7_AtomicjPU3AS3jj(i32 addrspace(3)* %a, i32 addrspace(3)* %b, i32 %c) local_unnamed_addr #0 {
  %1 = addrspacecast i32 addrspace(3)* %a to i32 *
  %2 = addrspacecast i32 addrspace(3)* %b to i32 *
  %call = tail call  i32 @_Z32_cl_atomic_compare_exchange_weakPU7CLlocalVU7_AtomicjPU7CLlocaljj(i32 * %1, i32 * %2, i32 %c)
  ret i32 %call
}


declare i32 @_Z41_cl_atomic_compare_exchange_weak_explicitPU8CLglobalVU7_AtomicjPU8CLglobaljj12memory_order12memory_order(i32 *, i32 *, i32, i32, i32) local_unnamed_addr #0

define spir_func i32 @_Z37atomic_compare_exchange_weak_explicitPU3AS1VU7_AtomicjPU3AS1jj12memory_order12memory_order(i32 addrspace(1)* %a, i32 addrspace(1)* %b, i32 %c, i32 %d, i32 %e) local_unnamed_addr #0 {
  %1 = addrspacecast i32 addrspace(1)* %a to i32 *
  %2 = addrspacecast i32 addrspace(1)* %b to i32 *
  %call = tail call  i32 @_Z41_cl_atomic_compare_exchange_weak_explicitPU8CLglobalVU7_AtomicjPU8CLglobaljj12memory_order12memory_order(i32 * %1, i32 * %2, i32 %c, i32 %d, i32 %e)
  ret i32 %call
}


declare i32 @_Z41_cl_atomic_compare_exchange_weak_explicitPU7CLlocalVU7_AtomicjPU7CLlocaljj12memory_order12memory_order(i32 *, i32 *, i32, i32, i32) local_unnamed_addr #0

define spir_func i32 @_Z37atomic_compare_exchange_weak_explicitPU3AS3VU7_AtomicjPU3AS3jj12memory_order12memory_order(i32 addrspace(3)* %a, i32 addrspace(3)* %b, i32 %c, i32 %d, i32 %e) local_unnamed_addr #0 {
  %1 = addrspacecast i32 addrspace(3)* %a to i32 *
  %2 = addrspacecast i32 addrspace(3)* %b to i32 *
  %call = tail call  i32 @_Z41_cl_atomic_compare_exchange_weak_explicitPU7CLlocalVU7_AtomicjPU7CLlocaljj12memory_order12memory_order(i32 * %1, i32 * %2, i32 %c, i32 %d, i32 %e)
  ret i32 %call
}


declare i32 @_Z41_cl_atomic_compare_exchange_weak_explicitPU8CLglobalVU7_AtomicjPU8CLglobaljj12memory_order12memory_order12memory_scope(i32 *, i32 *, i32, i32, i32, i32) local_unnamed_addr #0

define spir_func i32 @_Z37atomic_compare_exchange_weak_explicitPU3AS1VU7_AtomicjPU3AS1jj12memory_order12memory_order12memory_scope(i32 addrspace(1)* %a, i32 addrspace(1)* %b, i32 %c, i32 %d, i32 %e, i32 %f) local_unnamed_addr #0 {
  %1 = addrspacecast i32 addrspace(1)* %a to i32 *
  %2 = addrspacecast i32 addrspace(1)* %b to i32 *
  %call = tail call  i32 @_Z41_cl_atomic_compare_exchange_weak_explicitPU8CLglobalVU7_AtomicjPU8CLglobaljj12memory_order12memory_order12memory_scope(i32 * %1, i32 * %2, i32 %c, i32 %d, i32 %e, i32 %f)
  ret i32 %call
}


declare i32 @_Z41_cl_atomic_compare_exchange_weak_explicitPU7CLlocalVU7_AtomicjPU7CLlocaljj12memory_order12memory_order12memory_scope(i32 *, i32 *, i32, i32, i32, i32) local_unnamed_addr #0

define spir_func i32 @_Z37atomic_compare_exchange_weak_explicitPU3AS3VU7_AtomicjPU3AS3jj12memory_order12memory_order12memory_scope(i32 addrspace(3)* %a, i32 addrspace(3)* %b, i32 %c, i32 %d, i32 %e, i32 %f) local_unnamed_addr #0 {
  %1 = addrspacecast i32 addrspace(3)* %a to i32 *
  %2 = addrspacecast i32 addrspace(3)* %b to i32 *
  %call = tail call  i32 @_Z41_cl_atomic_compare_exchange_weak_explicitPU7CLlocalVU7_AtomicjPU7CLlocaljj12memory_order12memory_order12memory_scope(i32 * %1, i32 * %2, i32 %c, i32 %d, i32 %e, i32 %f)
  ret i32 %call
}


declare i32 @_Z19_cl_atomic_exchangePU8CLglobalVU7_Atomicjj(i32 *, i32) local_unnamed_addr #0

define spir_func i32 @_Z15atomic_exchangePU3AS1VU7_Atomicjj(i32 addrspace(1)* %a, i32 %b) local_unnamed_addr #0 {
  %1 = addrspacecast i32 addrspace(1)* %a to i32 *
  %call = tail call  i32 @_Z19_cl_atomic_exchangePU8CLglobalVU7_Atomicjj(i32 * %1, i32 %b)
  ret i32 %call
}


declare i32 @_Z19_cl_atomic_exchangePU7CLlocalVU7_Atomicjj(i32 *, i32) local_unnamed_addr #0

define spir_func i32 @_Z15atomic_exchangePU3AS3VU7_Atomicjj(i32 addrspace(3)* %a, i32 %b) local_unnamed_addr #0 {
  %1 = addrspacecast i32 addrspace(3)* %a to i32 *
  %call = tail call  i32 @_Z19_cl_atomic_exchangePU7CLlocalVU7_Atomicjj(i32 * %1, i32 %b)
  ret i32 %call
}


declare i32 @_Z28_cl_atomic_exchange_explicitPU8CLglobalVU7_Atomicjj12memory_order(i32 *, i32, i32) local_unnamed_addr #0

define spir_func i32 @_Z24atomic_exchange_explicitPU3AS1VU7_Atomicjj12memory_order(i32 addrspace(1)* %a, i32 %b, i32 %c) local_unnamed_addr #0 {
  %1 = addrspacecast i32 addrspace(1)* %a to i32 *
  %call = tail call  i32 @_Z28_cl_atomic_exchange_explicitPU8CLglobalVU7_Atomicjj12memory_order(i32 * %1, i32 %b, i32 %c)
  ret i32 %call
}


declare i32 @_Z28_cl_atomic_exchange_explicitPU7CLlocalVU7_Atomicjj12memory_order(i32 *, i32, i32) local_unnamed_addr #0

define spir_func i32 @_Z24atomic_exchange_explicitPU3AS3VU7_Atomicjj12memory_order(i32 addrspace(3)* %a, i32 %b, i32 %c) local_unnamed_addr #0 {
  %1 = addrspacecast i32 addrspace(3)* %a to i32 *
  %call = tail call  i32 @_Z28_cl_atomic_exchange_explicitPU7CLlocalVU7_Atomicjj12memory_order(i32 * %1, i32 %b, i32 %c)
  ret i32 %call
}


declare i32 @_Z28_cl_atomic_exchange_explicitPU8CLglobalVU7_Atomicjj12memory_order12memory_scope(i32 *, i32, i32, i32) local_unnamed_addr #0

define spir_func i32 @_Z24atomic_exchange_explicitPU3AS1VU7_Atomicjj12memory_order12memory_scope(i32 addrspace(1)* %a, i32 %b, i32 %c, i32 %d) local_unnamed_addr #0 {
  %1 = addrspacecast i32 addrspace(1)* %a to i32 *
  %call = tail call  i32 @_Z28_cl_atomic_exchange_explicitPU8CLglobalVU7_Atomicjj12memory_order12memory_scope(i32 * %1, i32 %b, i32 %c, i32 %d)
  ret i32 %call
}


declare i32 @_Z28_cl_atomic_exchange_explicitPU7CLlocalVU7_Atomicjj12memory_order12memory_scope(i32 *, i32, i32, i32) local_unnamed_addr #0

define spir_func i32 @_Z24atomic_exchange_explicitPU3AS3VU7_Atomicjj12memory_order12memory_scope(i32 addrspace(3)* %a, i32 %b, i32 %c, i32 %d) local_unnamed_addr #0 {
  %1 = addrspacecast i32 addrspace(3)* %a to i32 *
  %call = tail call  i32 @_Z28_cl_atomic_exchange_explicitPU7CLlocalVU7_Atomicjj12memory_order12memory_scope(i32 * %1, i32 %b, i32 %c, i32 %d)
  ret i32 %call
}


declare void @_Z15_cl_atomic_initPU8CLglobalVU7_Atomicll(i64 *, i64) local_unnamed_addr #0

define spir_func void @_Z11atomic_initPU3AS1VU7_Atomicll(i64 addrspace(1)* %a, i64 %b) local_unnamed_addr #0 {
  %1 = addrspacecast i64 addrspace(1)* %a to i64 *
  tail call void @_Z15_cl_atomic_initPU8CLglobalVU7_Atomicll(i64 * %1, i64 %b)
  ret void
}


declare void @_Z15_cl_atomic_initPU7CLlocalVU7_Atomicll(i64 *, i64) local_unnamed_addr #0

define spir_func void @_Z11atomic_initPU3AS3VU7_Atomicll(i64 addrspace(3)* %a, i64 %b) local_unnamed_addr #0 {
  %1 = addrspacecast i64 addrspace(3)* %a to i64 *
  tail call void @_Z15_cl_atomic_initPU7CLlocalVU7_Atomicll(i64 * %1, i64 %b)
  ret void
}


declare i64 @_Z34_cl_atomic_compare_exchange_strongPU8CLglobalVU7_AtomiclPU8CLgloballl(i64 *, i64 *, i64) local_unnamed_addr #0

define spir_func i64 @_Z30atomic_compare_exchange_strongPU3AS1VU7_AtomiclPU3AS1ll(i64 addrspace(1)* %a, i64 addrspace(1)* %b, i64 %c) local_unnamed_addr #0 {
  %1 = addrspacecast i64 addrspace(1)* %a to i64 *
  %2 = addrspacecast i64 addrspace(1)* %b to i64 *
  %call = tail call  i64 @_Z34_cl_atomic_compare_exchange_strongPU8CLglobalVU7_AtomiclPU8CLgloballl(i64 * %1, i64 * %2, i64 %c)
  ret i64 %call
}


declare i64 @_Z34_cl_atomic_compare_exchange_strongPU7CLlocalVU7_AtomiclPU7CLlocalll(i64 *, i64 *, i64) local_unnamed_addr #0

define spir_func i64 @_Z30atomic_compare_exchange_strongPU3AS3VU7_AtomiclPU3AS3ll(i64 addrspace(3)* %a, i64 addrspace(3)* %b, i64 %c) local_unnamed_addr #0 {
  %1 = addrspacecast i64 addrspace(3)* %a to i64 *
  %2 = addrspacecast i64 addrspace(3)* %b to i64 *
  %call = tail call  i64 @_Z34_cl_atomic_compare_exchange_strongPU7CLlocalVU7_AtomiclPU7CLlocalll(i64 * %1, i64 * %2, i64 %c)
  ret i64 %call
}


declare i64 @_Z43_cl_atomic_compare_exchange_strong_explicitPU8CLglobalVU7_AtomiclPU8CLgloballl12memory_order12memory_order(i64 *, i64 *, i64, i32, i32) local_unnamed_addr #0

define spir_func i64 @_Z39atomic_compare_exchange_strong_explicitPU3AS1VU7_AtomiclPU3AS1ll12memory_order12memory_order(i64 addrspace(1)* %a, i64 addrspace(1)* %b, i64 %c, i32 %d, i32 %e) local_unnamed_addr #0 {
  %1 = addrspacecast i64 addrspace(1)* %a to i64 *
  %2 = addrspacecast i64 addrspace(1)* %b to i64 *
  %call = tail call  i64 @_Z43_cl_atomic_compare_exchange_strong_explicitPU8CLglobalVU7_AtomiclPU8CLgloballl12memory_order12memory_order(i64 * %1, i64 * %2, i64 %c, i32 %d, i32 %e)
  ret i64 %call
}


declare i64 @_Z43_cl_atomic_compare_exchange_strong_explicitPU7CLlocalVU7_AtomiclPU7CLlocalll12memory_order12memory_order(i64 *, i64 *, i64, i32, i32) local_unnamed_addr #0

define spir_func i64 @_Z39atomic_compare_exchange_strong_explicitPU3AS3VU7_AtomiclPU3AS3ll12memory_order12memory_order(i64 addrspace(3)* %a, i64 addrspace(3)* %b, i64 %c, i32 %d, i32 %e) local_unnamed_addr #0 {
  %1 = addrspacecast i64 addrspace(3)* %a to i64 *
  %2 = addrspacecast i64 addrspace(3)* %b to i64 *
  %call = tail call  i64 @_Z43_cl_atomic_compare_exchange_strong_explicitPU7CLlocalVU7_AtomiclPU7CLlocalll12memory_order12memory_order(i64 * %1, i64 * %2, i64 %c, i32 %d, i32 %e)
  ret i64 %call
}


declare i64 @_Z43_cl_atomic_compare_exchange_strong_explicitPU8CLglobalVU7_AtomiclPU8CLgloballl12memory_order12memory_order12memory_scope(i64 *, i64 *, i64, i32, i32, i32) local_unnamed_addr #0

define spir_func i64 @_Z39atomic_compare_exchange_strong_explicitPU3AS1VU7_AtomiclPU3AS1ll12memory_order12memory_order12memory_scope(i64 addrspace(1)* %a, i64 addrspace(1)* %b, i64 %c, i32 %d, i32 %e, i32 %f) local_unnamed_addr #0 {
  %1 = addrspacecast i64 addrspace(1)* %a to i64 *
  %2 = addrspacecast i64 addrspace(1)* %b to i64 *
  %call = tail call  i64 @_Z43_cl_atomic_compare_exchange_strong_explicitPU8CLglobalVU7_AtomiclPU8CLgloballl12memory_order12memory_order12memory_scope(i64 * %1, i64 * %2, i64 %c, i32 %d, i32 %e, i32 %f)
  ret i64 %call
}


declare i64 @_Z43_cl_atomic_compare_exchange_strong_explicitPU7CLlocalVU7_AtomiclPU7CLlocalll12memory_order12memory_order12memory_scope(i64 *, i64 *, i64, i32, i32, i32) local_unnamed_addr #0

define spir_func i64 @_Z39atomic_compare_exchange_strong_explicitPU3AS3VU7_AtomiclPU3AS3ll12memory_order12memory_order12memory_scope(i64 addrspace(3)* %a, i64 addrspace(3)* %b, i64 %c, i32 %d, i32 %e, i32 %f) local_unnamed_addr #0 {
  %1 = addrspacecast i64 addrspace(3)* %a to i64 *
  %2 = addrspacecast i64 addrspace(3)* %b to i64 *
  %call = tail call  i64 @_Z43_cl_atomic_compare_exchange_strong_explicitPU7CLlocalVU7_AtomiclPU7CLlocalll12memory_order12memory_order12memory_scope(i64 * %1, i64 * %2, i64 %c, i32 %d, i32 %e, i32 %f)
  ret i64 %call
}


declare i64 @_Z16_cl_atomic_storePU8CLglobalVU7_Atomicll(i64 *, i64) local_unnamed_addr #0

define spir_func i64 @_Z12atomic_storePU3AS1VU7_Atomicll(i64 addrspace(1)* %a, i64 %b) local_unnamed_addr #0 {
  %1 = addrspacecast i64 addrspace(1)* %a to i64 *
  %call = tail call  i64 @_Z16_cl_atomic_storePU8CLglobalVU7_Atomicll(i64 * %1, i64 %b)
  ret i64 %call
}


declare i64 @_Z16_cl_atomic_storePU7CLlocalVU7_Atomicll(i64 *, i64) local_unnamed_addr #0

define spir_func i64 @_Z12atomic_storePU3AS3VU7_Atomicll(i64 addrspace(3)* %a, i64 %b) local_unnamed_addr #0 {
  %1 = addrspacecast i64 addrspace(3)* %a to i64 *
  %call = tail call  i64 @_Z16_cl_atomic_storePU7CLlocalVU7_Atomicll(i64 * %1, i64 %b)
  ret i64 %call
}


declare i64 @_Z25_cl_atomic_store_explicitPU8CLglobalVU7_Atomicll12memory_order(i64 *, i64, i32) local_unnamed_addr #0

define spir_func i64 @_Z21atomic_store_explicitPU3AS1VU7_Atomicll12memory_order(i64 addrspace(1)* %a, i64 %b, i32 %c) local_unnamed_addr #0 {
  %1 = addrspacecast i64 addrspace(1)* %a to i64 *
  %call = tail call  i64 @_Z25_cl_atomic_store_explicitPU8CLglobalVU7_Atomicll12memory_order(i64 * %1, i64 %b, i32 %c)
  ret i64 %call
}


declare i64 @_Z25_cl_atomic_store_explicitPU7CLlocalVU7_Atomicll12memory_order(i64 *, i64, i32) local_unnamed_addr #0

define spir_func i64 @_Z21atomic_store_explicitPU3AS3VU7_Atomicll12memory_order(i64 addrspace(3)* %a, i64 %b, i32 %c) local_unnamed_addr #0 {
  %1 = addrspacecast i64 addrspace(3)* %a to i64 *
  %call = tail call  i64 @_Z25_cl_atomic_store_explicitPU7CLlocalVU7_Atomicll12memory_order(i64 * %1, i64 %b, i32 %c)
  ret i64 %call
}


declare i64 @_Z25_cl_atomic_store_explicitPU8CLglobalVU7_Atomicll12memory_order12memory_scope(i64 *, i64, i32, i32) local_unnamed_addr #0

define spir_func i64 @_Z21atomic_store_explicitPU3AS1VU7_Atomicll12memory_order12memory_scope(i64 addrspace(1)* %a, i64 %b, i32 %c, i32 %d) local_unnamed_addr #0 {
  %1 = addrspacecast i64 addrspace(1)* %a to i64 *
  %call = tail call  i64 @_Z25_cl_atomic_store_explicitPU8CLglobalVU7_Atomicll12memory_order12memory_scope(i64 * %1, i64 %b, i32 %c, i32 %d)
  ret i64 %call
}


declare i64 @_Z25_cl_atomic_store_explicitPU7CLlocalVU7_Atomicll12memory_order12memory_scope(i64 *, i64, i32, i32) local_unnamed_addr #0

define spir_func i64 @_Z21atomic_store_explicitPU3AS3VU7_Atomicll12memory_order12memory_scope(i64 addrspace(3)* %a, i64 %b, i32 %c, i32 %d) local_unnamed_addr #0 {
  %1 = addrspacecast i64 addrspace(3)* %a to i64 *
  %call = tail call  i64 @_Z25_cl_atomic_store_explicitPU7CLlocalVU7_Atomicll12memory_order12memory_scope(i64 * %1, i64 %b, i32 %c, i32 %d)
  ret i64 %call
}


declare i64 @_Z15_cl_atomic_loadPU8CLglobalVU7_Atomicl(i64 *) local_unnamed_addr #0

define spir_func i64 @_Z11atomic_loadPU3AS1VU7_Atomicl(i64 addrspace(1)* %a) local_unnamed_addr #0 {
  %1 = addrspacecast i64 addrspace(1)* %a to i64 *
  %call = tail call  i64 @_Z15_cl_atomic_loadPU8CLglobalVU7_Atomicl(i64 * %1)
  ret i64 %call
}


declare i64 @_Z15_cl_atomic_loadPU7CLlocalVU7_Atomicl(i64 *) local_unnamed_addr #0

define spir_func i64 @_Z11atomic_loadPU3AS3VU7_Atomicl(i64 addrspace(3)* %a) local_unnamed_addr #0 {
  %1 = addrspacecast i64 addrspace(3)* %a to i64 *
  %call = tail call  i64 @_Z15_cl_atomic_loadPU7CLlocalVU7_Atomicl(i64 * %1)
  ret i64 %call
}


declare i64 @_Z24_cl_atomic_load_explicitPU8CLglobalVU7_Atomicl12memory_order(i64 *, i32) local_unnamed_addr #0

define spir_func i64 @_Z20atomic_load_explicitPU3AS1VU7_Atomicl12memory_order(i64 addrspace(1)* %a, i32 %b) local_unnamed_addr #0 {
  %1 = addrspacecast i64 addrspace(1)* %a to i64 *
  %call = tail call  i64 @_Z24_cl_atomic_load_explicitPU8CLglobalVU7_Atomicl12memory_order(i64 * %1, i32 %b)
  ret i64 %call
}


declare i64 @_Z24_cl_atomic_load_explicitPU7CLlocalVU7_Atomicl12memory_order(i64 *, i32) local_unnamed_addr #0

define spir_func i64 @_Z20atomic_load_explicitPU3AS3VU7_Atomicl12memory_order(i64 addrspace(3)* %a, i32 %b) local_unnamed_addr #0 {
  %1 = addrspacecast i64 addrspace(3)* %a to i64 *
  %call = tail call  i64 @_Z24_cl_atomic_load_explicitPU7CLlocalVU7_Atomicl12memory_order(i64 * %1, i32 %b)
  ret i64 %call
}


declare i64 @_Z24_cl_atomic_load_explicitPU8CLglobalVU7_Atomicl12memory_order12memory_scope(i64 *, i32, i32) local_unnamed_addr #0

define spir_func i64 @_Z20atomic_load_explicitPU3AS1VU7_Atomicl12memory_order12memory_scope(i64 addrspace(1)* %a, i32 %b, i32 %c) local_unnamed_addr #0 {
  %1 = addrspacecast i64 addrspace(1)* %a to i64 *
  %call = tail call  i64 @_Z24_cl_atomic_load_explicitPU8CLglobalVU7_Atomicl12memory_order12memory_scope(i64 * %1, i32 %b, i32 %c)
  ret i64 %call
}


declare i64 @_Z24_cl_atomic_load_explicitPU7CLlocalVU7_Atomicl12memory_order12memory_scope(i64 *, i32, i32) local_unnamed_addr #0

define spir_func i64 @_Z20atomic_load_explicitPU3AS3VU7_Atomicl12memory_order12memory_scope(i64 addrspace(3)* %a, i32 %b, i32 %c) local_unnamed_addr #0 {
  %1 = addrspacecast i64 addrspace(3)* %a to i64 *
  %call = tail call  i64 @_Z24_cl_atomic_load_explicitPU7CLlocalVU7_Atomicl12memory_order12memory_scope(i64 * %1, i32 %b, i32 %c)
  ret i64 %call
}


declare i64 @_Z32_cl_atomic_compare_exchange_weakPU8CLglobalVU7_AtomiclPU8CLgloballl(i64 *, i64 *, i64) local_unnamed_addr #0

define spir_func i64 @_Z28atomic_compare_exchange_weakPU3AS1VU7_AtomiclPU3AS1ll(i64 addrspace(1)* %a, i64 addrspace(1)* %b, i64 %c) local_unnamed_addr #0 {
  %1 = addrspacecast i64 addrspace(1)* %a to i64 *
  %2 = addrspacecast i64 addrspace(1)* %b to i64 *
  %call = tail call  i64 @_Z32_cl_atomic_compare_exchange_weakPU8CLglobalVU7_AtomiclPU8CLgloballl(i64 * %1, i64 * %2, i64 %c)
  ret i64 %call
}


declare i64 @_Z32_cl_atomic_compare_exchange_weakPU7CLlocalVU7_AtomiclPU7CLlocalll(i64 *, i64 *, i64) local_unnamed_addr #0

define spir_func i64 @_Z28atomic_compare_exchange_weakPU3AS3VU7_AtomiclPU3AS3ll(i64 addrspace(3)* %a, i64 addrspace(3)* %b, i64 %c) local_unnamed_addr #0 {
  %1 = addrspacecast i64 addrspace(3)* %a to i64 *
  %2 = addrspacecast i64 addrspace(3)* %b to i64 *
  %call = tail call  i64 @_Z32_cl_atomic_compare_exchange_weakPU7CLlocalVU7_AtomiclPU7CLlocalll(i64 * %1, i64 * %2, i64 %c)
  ret i64 %call
}


declare i64 @_Z41_cl_atomic_compare_exchange_weak_explicitPU8CLglobalVU7_AtomiclPU8CLgloballl12memory_order12memory_order(i64 *, i64 *, i64, i32, i32) local_unnamed_addr #0

define spir_func i64 @_Z37atomic_compare_exchange_weak_explicitPU3AS1VU7_AtomiclPU3AS1ll12memory_order12memory_order(i64 addrspace(1)* %a, i64 addrspace(1)* %b, i64 %c, i32 %d, i32 %e) local_unnamed_addr #0 {
  %1 = addrspacecast i64 addrspace(1)* %a to i64 *
  %2 = addrspacecast i64 addrspace(1)* %b to i64 *
  %call = tail call  i64 @_Z41_cl_atomic_compare_exchange_weak_explicitPU8CLglobalVU7_AtomiclPU8CLgloballl12memory_order12memory_order(i64 * %1, i64 * %2, i64 %c, i32 %d, i32 %e)
  ret i64 %call
}


declare i64 @_Z41_cl_atomic_compare_exchange_weak_explicitPU7CLlocalVU7_AtomiclPU7CLlocalll12memory_order12memory_order(i64 *, i64 *, i64, i32, i32) local_unnamed_addr #0

define spir_func i64 @_Z37atomic_compare_exchange_weak_explicitPU3AS3VU7_AtomiclPU3AS3ll12memory_order12memory_order(i64 addrspace(3)* %a, i64 addrspace(3)* %b, i64 %c, i32 %d, i32 %e) local_unnamed_addr #0 {
  %1 = addrspacecast i64 addrspace(3)* %a to i64 *
  %2 = addrspacecast i64 addrspace(3)* %b to i64 *
  %call = tail call  i64 @_Z41_cl_atomic_compare_exchange_weak_explicitPU7CLlocalVU7_AtomiclPU7CLlocalll12memory_order12memory_order(i64 * %1, i64 * %2, i64 %c, i32 %d, i32 %e)
  ret i64 %call
}


declare i64 @_Z41_cl_atomic_compare_exchange_weak_explicitPU8CLglobalVU7_AtomiclPU8CLgloballl12memory_order12memory_order12memory_scope(i64 *, i64 *, i64, i32, i32, i32) local_unnamed_addr #0

define spir_func i64 @_Z37atomic_compare_exchange_weak_explicitPU3AS1VU7_AtomiclPU3AS1ll12memory_order12memory_order12memory_scope(i64 addrspace(1)* %a, i64 addrspace(1)* %b, i64 %c, i32 %d, i32 %e, i32 %f) local_unnamed_addr #0 {
  %1 = addrspacecast i64 addrspace(1)* %a to i64 *
  %2 = addrspacecast i64 addrspace(1)* %b to i64 *
  %call = tail call  i64 @_Z41_cl_atomic_compare_exchange_weak_explicitPU8CLglobalVU7_AtomiclPU8CLgloballl12memory_order12memory_order12memory_scope(i64 * %1, i64 * %2, i64 %c, i32 %d, i32 %e, i32 %f)
  ret i64 %call
}


declare i64 @_Z41_cl_atomic_compare_exchange_weak_explicitPU7CLlocalVU7_AtomiclPU7CLlocalll12memory_order12memory_order12memory_scope(i64 *, i64 *, i64, i32, i32, i32) local_unnamed_addr #0

define spir_func i64 @_Z37atomic_compare_exchange_weak_explicitPU3AS3VU7_AtomiclPU3AS3ll12memory_order12memory_order12memory_scope(i64 addrspace(3)* %a, i64 addrspace(3)* %b, i64 %c, i32 %d, i32 %e, i32 %f) local_unnamed_addr #0 {
  %1 = addrspacecast i64 addrspace(3)* %a to i64 *
  %2 = addrspacecast i64 addrspace(3)* %b to i64 *
  %call = tail call  i64 @_Z41_cl_atomic_compare_exchange_weak_explicitPU7CLlocalVU7_AtomiclPU7CLlocalll12memory_order12memory_order12memory_scope(i64 * %1, i64 * %2, i64 %c, i32 %d, i32 %e, i32 %f)
  ret i64 %call
}


declare i64 @_Z19_cl_atomic_exchangePU8CLglobalVU7_Atomicll(i64 *, i64) local_unnamed_addr #0

define spir_func i64 @_Z15atomic_exchangePU3AS1VU7_Atomicll(i64 addrspace(1)* %a, i64 %b) local_unnamed_addr #0 {
  %1 = addrspacecast i64 addrspace(1)* %a to i64 *
  %call = tail call  i64 @_Z19_cl_atomic_exchangePU8CLglobalVU7_Atomicll(i64 * %1, i64 %b)
  ret i64 %call
}


declare i64 @_Z19_cl_atomic_exchangePU7CLlocalVU7_Atomicll(i64 *, i64) local_unnamed_addr #0

define spir_func i64 @_Z15atomic_exchangePU3AS3VU7_Atomicll(i64 addrspace(3)* %a, i64 %b) local_unnamed_addr #0 {
  %1 = addrspacecast i64 addrspace(3)* %a to i64 *
  %call = tail call  i64 @_Z19_cl_atomic_exchangePU7CLlocalVU7_Atomicll(i64 * %1, i64 %b)
  ret i64 %call
}


declare i64 @_Z28_cl_atomic_exchange_explicitPU8CLglobalVU7_Atomicll12memory_order(i64 *, i64, i32) local_unnamed_addr #0

define spir_func i64 @_Z24atomic_exchange_explicitPU3AS1VU7_Atomicll12memory_order(i64 addrspace(1)* %a, i64 %b, i32 %c) local_unnamed_addr #0 {
  %1 = addrspacecast i64 addrspace(1)* %a to i64 *
  %call = tail call  i64 @_Z28_cl_atomic_exchange_explicitPU8CLglobalVU7_Atomicll12memory_order(i64 * %1, i64 %b, i32 %c)
  ret i64 %call
}


declare i64 @_Z28_cl_atomic_exchange_explicitPU7CLlocalVU7_Atomicll12memory_order(i64 *, i64, i32) local_unnamed_addr #0

define spir_func i64 @_Z24atomic_exchange_explicitPU3AS3VU7_Atomicll12memory_order(i64 addrspace(3)* %a, i64 %b, i32 %c) local_unnamed_addr #0 {
  %1 = addrspacecast i64 addrspace(3)* %a to i64 *
  %call = tail call  i64 @_Z28_cl_atomic_exchange_explicitPU7CLlocalVU7_Atomicll12memory_order(i64 * %1, i64 %b, i32 %c)
  ret i64 %call
}


declare i64 @_Z28_cl_atomic_exchange_explicitPU8CLglobalVU7_Atomicll12memory_order12memory_scope(i64 *, i64, i32, i32) local_unnamed_addr #0

define spir_func i64 @_Z24atomic_exchange_explicitPU3AS1VU7_Atomicll12memory_order12memory_scope(i64 addrspace(1)* %a, i64 %b, i32 %c, i32 %d) local_unnamed_addr #0 {
  %1 = addrspacecast i64 addrspace(1)* %a to i64 *
  %call = tail call  i64 @_Z28_cl_atomic_exchange_explicitPU8CLglobalVU7_Atomicll12memory_order12memory_scope(i64 * %1, i64 %b, i32 %c, i32 %d)
  ret i64 %call
}


declare i64 @_Z28_cl_atomic_exchange_explicitPU7CLlocalVU7_Atomicll12memory_order12memory_scope(i64 *, i64, i32, i32) local_unnamed_addr #0

define spir_func i64 @_Z24atomic_exchange_explicitPU3AS3VU7_Atomicll12memory_order12memory_scope(i64 addrspace(3)* %a, i64 %b, i32 %c, i32 %d) local_unnamed_addr #0 {
  %1 = addrspacecast i64 addrspace(3)* %a to i64 *
  %call = tail call  i64 @_Z28_cl_atomic_exchange_explicitPU7CLlocalVU7_Atomicll12memory_order12memory_scope(i64 * %1, i64 %b, i32 %c, i32 %d)
  ret i64 %call
}


declare void @_Z15_cl_atomic_initPU8CLglobalVU7_Atomicmm(i64 *, i64) local_unnamed_addr #0

define spir_func void @_Z11atomic_initPU3AS1VU7_Atomicmm(i64 addrspace(1)* %a, i64 %b) local_unnamed_addr #0 {
  %1 = addrspacecast i64 addrspace(1)* %a to i64 *
  tail call void @_Z15_cl_atomic_initPU8CLglobalVU7_Atomicmm(i64 * %1, i64 %b)
  ret void
}


declare void @_Z15_cl_atomic_initPU7CLlocalVU7_Atomicmm(i64 *, i64) local_unnamed_addr #0

define spir_func void @_Z11atomic_initPU3AS3VU7_Atomicmm(i64 addrspace(3)* %a, i64 %b) local_unnamed_addr #0 {
  %1 = addrspacecast i64 addrspace(3)* %a to i64 *
  tail call void @_Z15_cl_atomic_initPU7CLlocalVU7_Atomicmm(i64 * %1, i64 %b)
  ret void
}


declare i64 @_Z34_cl_atomic_compare_exchange_strongPU8CLglobalVU7_AtomicmPU8CLglobalmm(i64 *, i64 *, i64) local_unnamed_addr #0

define spir_func i64 @_Z30atomic_compare_exchange_strongPU3AS1VU7_AtomicmPU3AS1mm(i64 addrspace(1)* %a, i64 addrspace(1)* %b, i64 %c) local_unnamed_addr #0 {
  %1 = addrspacecast i64 addrspace(1)* %a to i64 *
  %2 = addrspacecast i64 addrspace(1)* %b to i64 *
  %call = tail call  i64 @_Z34_cl_atomic_compare_exchange_strongPU8CLglobalVU7_AtomicmPU8CLglobalmm(i64 * %1, i64 * %2, i64 %c)
  ret i64 %call
}


declare i64 @_Z34_cl_atomic_compare_exchange_strongPU7CLlocalVU7_AtomicmPU7CLlocalmm(i64 *, i64 *, i64) local_unnamed_addr #0

define spir_func i64 @_Z30atomic_compare_exchange_strongPU3AS3VU7_AtomicmPU3AS3mm(i64 addrspace(3)* %a, i64 addrspace(3)* %b, i64 %c) local_unnamed_addr #0 {
  %1 = addrspacecast i64 addrspace(3)* %a to i64 *
  %2 = addrspacecast i64 addrspace(3)* %b to i64 *
  %call = tail call  i64 @_Z34_cl_atomic_compare_exchange_strongPU7CLlocalVU7_AtomicmPU7CLlocalmm(i64 * %1, i64 * %2, i64 %c)
  ret i64 %call
}


declare i64 @_Z43_cl_atomic_compare_exchange_strong_explicitPU8CLglobalVU7_AtomicmPU8CLglobalmm12memory_order12memory_order(i64 *, i64 *, i64, i32, i32) local_unnamed_addr #0

define spir_func i64 @_Z39atomic_compare_exchange_strong_explicitPU3AS1VU7_AtomicmPU3AS1mm12memory_order12memory_order(i64 addrspace(1)* %a, i64 addrspace(1)* %b, i64 %c, i32 %d, i32 %e) local_unnamed_addr #0 {
  %1 = addrspacecast i64 addrspace(1)* %a to i64 *
  %2 = addrspacecast i64 addrspace(1)* %b to i64 *
  %call = tail call  i64 @_Z43_cl_atomic_compare_exchange_strong_explicitPU8CLglobalVU7_AtomicmPU8CLglobalmm12memory_order12memory_order(i64 * %1, i64 * %2, i64 %c, i32 %d, i32 %e)
  ret i64 %call
}


declare i64 @_Z43_cl_atomic_compare_exchange_strong_explicitPU7CLlocalVU7_AtomicmPU7CLlocalmm12memory_order12memory_order(i64 *, i64 *, i64, i32, i32) local_unnamed_addr #0

define spir_func i64 @_Z39atomic_compare_exchange_strong_explicitPU3AS3VU7_AtomicmPU3AS3mm12memory_order12memory_order(i64 addrspace(3)* %a, i64 addrspace(3)* %b, i64 %c, i32 %d, i32 %e) local_unnamed_addr #0 {
  %1 = addrspacecast i64 addrspace(3)* %a to i64 *
  %2 = addrspacecast i64 addrspace(3)* %b to i64 *
  %call = tail call  i64 @_Z43_cl_atomic_compare_exchange_strong_explicitPU7CLlocalVU7_AtomicmPU7CLlocalmm12memory_order12memory_order(i64 * %1, i64 * %2, i64 %c, i32 %d, i32 %e)
  ret i64 %call
}


declare i64 @_Z43_cl_atomic_compare_exchange_strong_explicitPU8CLglobalVU7_AtomicmPU8CLglobalmm12memory_order12memory_order12memory_scope(i64 *, i64 *, i64, i32, i32, i32) local_unnamed_addr #0

define spir_func i64 @_Z39atomic_compare_exchange_strong_explicitPU3AS1VU7_AtomicmPU3AS1mm12memory_order12memory_order12memory_scope(i64 addrspace(1)* %a, i64 addrspace(1)* %b, i64 %c, i32 %d, i32 %e, i32 %f) local_unnamed_addr #0 {
  %1 = addrspacecast i64 addrspace(1)* %a to i64 *
  %2 = addrspacecast i64 addrspace(1)* %b to i64 *
  %call = tail call  i64 @_Z43_cl_atomic_compare_exchange_strong_explicitPU8CLglobalVU7_AtomicmPU8CLglobalmm12memory_order12memory_order12memory_scope(i64 * %1, i64 * %2, i64 %c, i32 %d, i32 %e, i32 %f)
  ret i64 %call
}


declare i64 @_Z43_cl_atomic_compare_exchange_strong_explicitPU7CLlocalVU7_AtomicmPU7CLlocalmm12memory_order12memory_order12memory_scope(i64 *, i64 *, i64, i32, i32, i32) local_unnamed_addr #0

define spir_func i64 @_Z39atomic_compare_exchange_strong_explicitPU3AS3VU7_AtomicmPU3AS3mm12memory_order12memory_order12memory_scope(i64 addrspace(3)* %a, i64 addrspace(3)* %b, i64 %c, i32 %d, i32 %e, i32 %f) local_unnamed_addr #0 {
  %1 = addrspacecast i64 addrspace(3)* %a to i64 *
  %2 = addrspacecast i64 addrspace(3)* %b to i64 *
  %call = tail call  i64 @_Z43_cl_atomic_compare_exchange_strong_explicitPU7CLlocalVU7_AtomicmPU7CLlocalmm12memory_order12memory_order12memory_scope(i64 * %1, i64 * %2, i64 %c, i32 %d, i32 %e, i32 %f)
  ret i64 %call
}


declare i64 @_Z16_cl_atomic_storePU8CLglobalVU7_Atomicmm(i64 *, i64) local_unnamed_addr #0

define spir_func i64 @_Z12atomic_storePU3AS1VU7_Atomicmm(i64 addrspace(1)* %a, i64 %b) local_unnamed_addr #0 {
  %1 = addrspacecast i64 addrspace(1)* %a to i64 *
  %call = tail call  i64 @_Z16_cl_atomic_storePU8CLglobalVU7_Atomicmm(i64 * %1, i64 %b)
  ret i64 %call
}


declare i64 @_Z16_cl_atomic_storePU7CLlocalVU7_Atomicmm(i64 *, i64) local_unnamed_addr #0

define spir_func i64 @_Z12atomic_storePU3AS3VU7_Atomicmm(i64 addrspace(3)* %a, i64 %b) local_unnamed_addr #0 {
  %1 = addrspacecast i64 addrspace(3)* %a to i64 *
  %call = tail call  i64 @_Z16_cl_atomic_storePU7CLlocalVU7_Atomicmm(i64 * %1, i64 %b)
  ret i64 %call
}


declare i64 @_Z25_cl_atomic_store_explicitPU8CLglobalVU7_Atomicmm12memory_order(i64 *, i64, i32) local_unnamed_addr #0

define spir_func i64 @_Z21atomic_store_explicitPU3AS1VU7_Atomicmm12memory_order(i64 addrspace(1)* %a, i64 %b, i32 %c) local_unnamed_addr #0 {
  %1 = addrspacecast i64 addrspace(1)* %a to i64 *
  %call = tail call  i64 @_Z25_cl_atomic_store_explicitPU8CLglobalVU7_Atomicmm12memory_order(i64 * %1, i64 %b, i32 %c)
  ret i64 %call
}


declare i64 @_Z25_cl_atomic_store_explicitPU7CLlocalVU7_Atomicmm12memory_order(i64 *, i64, i32) local_unnamed_addr #0

define spir_func i64 @_Z21atomic_store_explicitPU3AS3VU7_Atomicmm12memory_order(i64 addrspace(3)* %a, i64 %b, i32 %c) local_unnamed_addr #0 {
  %1 = addrspacecast i64 addrspace(3)* %a to i64 *
  %call = tail call  i64 @_Z25_cl_atomic_store_explicitPU7CLlocalVU7_Atomicmm12memory_order(i64 * %1, i64 %b, i32 %c)
  ret i64 %call
}


declare i64 @_Z25_cl_atomic_store_explicitPU8CLglobalVU7_Atomicmm12memory_order12memory_scope(i64 *, i64, i32, i32) local_unnamed_addr #0

define spir_func i64 @_Z21atomic_store_explicitPU3AS1VU7_Atomicmm12memory_order12memory_scope(i64 addrspace(1)* %a, i64 %b, i32 %c, i32 %d) local_unnamed_addr #0 {
  %1 = addrspacecast i64 addrspace(1)* %a to i64 *
  %call = tail call  i64 @_Z25_cl_atomic_store_explicitPU8CLglobalVU7_Atomicmm12memory_order12memory_scope(i64 * %1, i64 %b, i32 %c, i32 %d)
  ret i64 %call
}


declare i64 @_Z25_cl_atomic_store_explicitPU7CLlocalVU7_Atomicmm12memory_order12memory_scope(i64 *, i64, i32, i32) local_unnamed_addr #0

define spir_func i64 @_Z21atomic_store_explicitPU3AS3VU7_Atomicmm12memory_order12memory_scope(i64 addrspace(3)* %a, i64 %b, i32 %c, i32 %d) local_unnamed_addr #0 {
  %1 = addrspacecast i64 addrspace(3)* %a to i64 *
  %call = tail call  i64 @_Z25_cl_atomic_store_explicitPU7CLlocalVU7_Atomicmm12memory_order12memory_scope(i64 * %1, i64 %b, i32 %c, i32 %d)
  ret i64 %call
}


declare i64 @_Z15_cl_atomic_loadPU8CLglobalVU7_Atomicm(i64 *) local_unnamed_addr #0

define spir_func i64 @_Z11atomic_loadPU3AS1VU7_Atomicm(i64 addrspace(1)* %a) local_unnamed_addr #0 {
  %1 = addrspacecast i64 addrspace(1)* %a to i64 *
  %call = tail call  i64 @_Z15_cl_atomic_loadPU8CLglobalVU7_Atomicm(i64 * %1)
  ret i64 %call
}


declare i64 @_Z15_cl_atomic_loadPU7CLlocalVU7_Atomicm(i64 *) local_unnamed_addr #0

define spir_func i64 @_Z11atomic_loadPU3AS3VU7_Atomicm(i64 addrspace(3)* %a) local_unnamed_addr #0 {
  %1 = addrspacecast i64 addrspace(3)* %a to i64 *
  %call = tail call  i64 @_Z15_cl_atomic_loadPU7CLlocalVU7_Atomicm(i64 * %1)
  ret i64 %call
}


declare i64 @_Z24_cl_atomic_load_explicitPU8CLglobalVU7_Atomicm12memory_order(i64 *, i32) local_unnamed_addr #0

define spir_func i64 @_Z20atomic_load_explicitPU3AS1VU7_Atomicm12memory_order(i64 addrspace(1)* %a, i32 %b) local_unnamed_addr #0 {
  %1 = addrspacecast i64 addrspace(1)* %a to i64 *
  %call = tail call  i64 @_Z24_cl_atomic_load_explicitPU8CLglobalVU7_Atomicm12memory_order(i64 * %1, i32 %b)
  ret i64 %call
}


declare i64 @_Z24_cl_atomic_load_explicitPU7CLlocalVU7_Atomicm12memory_order(i64 *, i32) local_unnamed_addr #0

define spir_func i64 @_Z20atomic_load_explicitPU3AS3VU7_Atomicm12memory_order(i64 addrspace(3)* %a, i32 %b) local_unnamed_addr #0 {
  %1 = addrspacecast i64 addrspace(3)* %a to i64 *
  %call = tail call  i64 @_Z24_cl_atomic_load_explicitPU7CLlocalVU7_Atomicm12memory_order(i64 * %1, i32 %b)
  ret i64 %call
}


declare i64 @_Z24_cl_atomic_load_explicitPU8CLglobalVU7_Atomicm12memory_order12memory_scope(i64 *, i32, i32) local_unnamed_addr #0

define spir_func i64 @_Z20atomic_load_explicitPU3AS1VU7_Atomicm12memory_order12memory_scope(i64 addrspace(1)* %a, i32 %b, i32 %c) local_unnamed_addr #0 {
  %1 = addrspacecast i64 addrspace(1)* %a to i64 *
  %call = tail call  i64 @_Z24_cl_atomic_load_explicitPU8CLglobalVU7_Atomicm12memory_order12memory_scope(i64 * %1, i32 %b, i32 %c)
  ret i64 %call
}


declare i64 @_Z24_cl_atomic_load_explicitPU7CLlocalVU7_Atomicm12memory_order12memory_scope(i64 *, i32, i32) local_unnamed_addr #0

define spir_func i64 @_Z20atomic_load_explicitPU3AS3VU7_Atomicm12memory_order12memory_scope(i64 addrspace(3)* %a, i32 %b, i32 %c) local_unnamed_addr #0 {
  %1 = addrspacecast i64 addrspace(3)* %a to i64 *
  %call = tail call  i64 @_Z24_cl_atomic_load_explicitPU7CLlocalVU7_Atomicm12memory_order12memory_scope(i64 * %1, i32 %b, i32 %c)
  ret i64 %call
}


declare i64 @_Z32_cl_atomic_compare_exchange_weakPU8CLglobalVU7_AtomicmPU8CLglobalmm(i64 *, i64 *, i64) local_unnamed_addr #0

define spir_func i64 @_Z28atomic_compare_exchange_weakPU3AS1VU7_AtomicmPU3AS1mm(i64 addrspace(1)* %a, i64 addrspace(1)* %b, i64 %c) local_unnamed_addr #0 {
  %1 = addrspacecast i64 addrspace(1)* %a to i64 *
  %2 = addrspacecast i64 addrspace(1)* %b to i64 *
  %call = tail call  i64 @_Z32_cl_atomic_compare_exchange_weakPU8CLglobalVU7_AtomicmPU8CLglobalmm(i64 * %1, i64 * %2, i64 %c)
  ret i64 %call
}


declare i64 @_Z32_cl_atomic_compare_exchange_weakPU7CLlocalVU7_AtomicmPU7CLlocalmm(i64 *, i64 *, i64) local_unnamed_addr #0

define spir_func i64 @_Z28atomic_compare_exchange_weakPU3AS3VU7_AtomicmPU3AS3mm(i64 addrspace(3)* %a, i64 addrspace(3)* %b, i64 %c) local_unnamed_addr #0 {
  %1 = addrspacecast i64 addrspace(3)* %a to i64 *
  %2 = addrspacecast i64 addrspace(3)* %b to i64 *
  %call = tail call  i64 @_Z32_cl_atomic_compare_exchange_weakPU7CLlocalVU7_AtomicmPU7CLlocalmm(i64 * %1, i64 * %2, i64 %c)
  ret i64 %call
}


declare i64 @_Z41_cl_atomic_compare_exchange_weak_explicitPU8CLglobalVU7_AtomicmPU8CLglobalmm12memory_order12memory_order(i64 *, i64 *, i64, i32, i32) local_unnamed_addr #0

define spir_func i64 @_Z37atomic_compare_exchange_weak_explicitPU3AS1VU7_AtomicmPU3AS1mm12memory_order12memory_order(i64 addrspace(1)* %a, i64 addrspace(1)* %b, i64 %c, i32 %d, i32 %e) local_unnamed_addr #0 {
  %1 = addrspacecast i64 addrspace(1)* %a to i64 *
  %2 = addrspacecast i64 addrspace(1)* %b to i64 *
  %call = tail call  i64 @_Z41_cl_atomic_compare_exchange_weak_explicitPU8CLglobalVU7_AtomicmPU8CLglobalmm12memory_order12memory_order(i64 * %1, i64 * %2, i64 %c, i32 %d, i32 %e)
  ret i64 %call
}


declare i64 @_Z41_cl_atomic_compare_exchange_weak_explicitPU7CLlocalVU7_AtomicmPU7CLlocalmm12memory_order12memory_order(i64 *, i64 *, i64, i32, i32) local_unnamed_addr #0

define spir_func i64 @_Z37atomic_compare_exchange_weak_explicitPU3AS3VU7_AtomicmPU3AS3mm12memory_order12memory_order(i64 addrspace(3)* %a, i64 addrspace(3)* %b, i64 %c, i32 %d, i32 %e) local_unnamed_addr #0 {
  %1 = addrspacecast i64 addrspace(3)* %a to i64 *
  %2 = addrspacecast i64 addrspace(3)* %b to i64 *
  %call = tail call  i64 @_Z41_cl_atomic_compare_exchange_weak_explicitPU7CLlocalVU7_AtomicmPU7CLlocalmm12memory_order12memory_order(i64 * %1, i64 * %2, i64 %c, i32 %d, i32 %e)
  ret i64 %call
}


declare i64 @_Z41_cl_atomic_compare_exchange_weak_explicitPU8CLglobalVU7_AtomicmPU8CLglobalmm12memory_order12memory_order12memory_scope(i64 *, i64 *, i64, i32, i32, i32) local_unnamed_addr #0

define spir_func i64 @_Z37atomic_compare_exchange_weak_explicitPU3AS1VU7_AtomicmPU3AS1mm12memory_order12memory_order12memory_scope(i64 addrspace(1)* %a, i64 addrspace(1)* %b, i64 %c, i32 %d, i32 %e, i32 %f) local_unnamed_addr #0 {
  %1 = addrspacecast i64 addrspace(1)* %a to i64 *
  %2 = addrspacecast i64 addrspace(1)* %b to i64 *
  %call = tail call  i64 @_Z41_cl_atomic_compare_exchange_weak_explicitPU8CLglobalVU7_AtomicmPU8CLglobalmm12memory_order12memory_order12memory_scope(i64 * %1, i64 * %2, i64 %c, i32 %d, i32 %e, i32 %f)
  ret i64 %call
}


declare i64 @_Z41_cl_atomic_compare_exchange_weak_explicitPU7CLlocalVU7_AtomicmPU7CLlocalmm12memory_order12memory_order12memory_scope(i64 *, i64 *, i64, i32, i32, i32) local_unnamed_addr #0

define spir_func i64 @_Z37atomic_compare_exchange_weak_explicitPU3AS3VU7_AtomicmPU3AS3mm12memory_order12memory_order12memory_scope(i64 addrspace(3)* %a, i64 addrspace(3)* %b, i64 %c, i32 %d, i32 %e, i32 %f) local_unnamed_addr #0 {
  %1 = addrspacecast i64 addrspace(3)* %a to i64 *
  %2 = addrspacecast i64 addrspace(3)* %b to i64 *
  %call = tail call  i64 @_Z41_cl_atomic_compare_exchange_weak_explicitPU7CLlocalVU7_AtomicmPU7CLlocalmm12memory_order12memory_order12memory_scope(i64 * %1, i64 * %2, i64 %c, i32 %d, i32 %e, i32 %f)
  ret i64 %call
}


declare i64 @_Z19_cl_atomic_exchangePU8CLglobalVU7_Atomicmm(i64 *, i64) local_unnamed_addr #0

define spir_func i64 @_Z15atomic_exchangePU3AS1VU7_Atomicmm(i64 addrspace(1)* %a, i64 %b) local_unnamed_addr #0 {
  %1 = addrspacecast i64 addrspace(1)* %a to i64 *
  %call = tail call  i64 @_Z19_cl_atomic_exchangePU8CLglobalVU7_Atomicmm(i64 * %1, i64 %b)
  ret i64 %call
}


declare i64 @_Z19_cl_atomic_exchangePU7CLlocalVU7_Atomicmm(i64 *, i64) local_unnamed_addr #0

define spir_func i64 @_Z15atomic_exchangePU3AS3VU7_Atomicmm(i64 addrspace(3)* %a, i64 %b) local_unnamed_addr #0 {
  %1 = addrspacecast i64 addrspace(3)* %a to i64 *
  %call = tail call  i64 @_Z19_cl_atomic_exchangePU7CLlocalVU7_Atomicmm(i64 * %1, i64 %b)
  ret i64 %call
}


declare i64 @_Z28_cl_atomic_exchange_explicitPU8CLglobalVU7_Atomicmm12memory_order(i64 *, i64, i32) local_unnamed_addr #0

define spir_func i64 @_Z24atomic_exchange_explicitPU3AS1VU7_Atomicmm12memory_order(i64 addrspace(1)* %a, i64 %b, i32 %c) local_unnamed_addr #0 {
  %1 = addrspacecast i64 addrspace(1)* %a to i64 *
  %call = tail call  i64 @_Z28_cl_atomic_exchange_explicitPU8CLglobalVU7_Atomicmm12memory_order(i64 * %1, i64 %b, i32 %c)
  ret i64 %call
}


declare i64 @_Z28_cl_atomic_exchange_explicitPU7CLlocalVU7_Atomicmm12memory_order(i64 *, i64, i32) local_unnamed_addr #0

define spir_func i64 @_Z24atomic_exchange_explicitPU3AS3VU7_Atomicmm12memory_order(i64 addrspace(3)* %a, i64 %b, i32 %c) local_unnamed_addr #0 {
  %1 = addrspacecast i64 addrspace(3)* %a to i64 *
  %call = tail call  i64 @_Z28_cl_atomic_exchange_explicitPU7CLlocalVU7_Atomicmm12memory_order(i64 * %1, i64 %b, i32 %c)
  ret i64 %call
}


declare i64 @_Z28_cl_atomic_exchange_explicitPU8CLglobalVU7_Atomicmm12memory_order12memory_scope(i64 *, i64, i32, i32) local_unnamed_addr #0

define spir_func i64 @_Z24atomic_exchange_explicitPU3AS1VU7_Atomicmm12memory_order12memory_scope(i64 addrspace(1)* %a, i64 %b, i32 %c, i32 %d) local_unnamed_addr #0 {
  %1 = addrspacecast i64 addrspace(1)* %a to i64 *
  %call = tail call  i64 @_Z28_cl_atomic_exchange_explicitPU8CLglobalVU7_Atomicmm12memory_order12memory_scope(i64 * %1, i64 %b, i32 %c, i32 %d)
  ret i64 %call
}


declare i64 @_Z28_cl_atomic_exchange_explicitPU7CLlocalVU7_Atomicmm12memory_order12memory_scope(i64 *, i64, i32, i32) local_unnamed_addr #0

define spir_func i64 @_Z24atomic_exchange_explicitPU3AS3VU7_Atomicmm12memory_order12memory_scope(i64 addrspace(3)* %a, i64 %b, i32 %c, i32 %d) local_unnamed_addr #0 {
  %1 = addrspacecast i64 addrspace(3)* %a to i64 *
  %call = tail call  i64 @_Z28_cl_atomic_exchange_explicitPU7CLlocalVU7_Atomicmm12memory_order12memory_scope(i64 * %1, i64 %b, i32 %c, i32 %d)
  ret i64 %call
}


declare void @_Z15_cl_atomic_initPU8CLglobalVU7_Atomicff(float *, float) local_unnamed_addr #0

define spir_func void @_Z11atomic_initPU3AS1VU7_Atomicff(float addrspace(1)* %a, float %b) local_unnamed_addr #0 {
  %1 = addrspacecast float addrspace(1)* %a to float *
  tail call void @_Z15_cl_atomic_initPU8CLglobalVU7_Atomicff(float * %1, float %b)
  ret void
}


declare void @_Z15_cl_atomic_initPU7CLlocalVU7_Atomicff(float *, float) local_unnamed_addr #0

define spir_func void @_Z11atomic_initPU3AS3VU7_Atomicff(float addrspace(3)* %a, float %b) local_unnamed_addr #0 {
  %1 = addrspacecast float addrspace(3)* %a to float *
  tail call void @_Z15_cl_atomic_initPU7CLlocalVU7_Atomicff(float * %1, float %b)
  ret void
}


declare float @_Z34_cl_atomic_compare_exchange_strongPU8CLglobalVU7_AtomicfPU8CLglobalff(float *, float *, float) local_unnamed_addr #0

define spir_func float @_Z30atomic_compare_exchange_strongPU3AS1VU7_AtomicfPU3AS1ff(float addrspace(1)* %a, float addrspace(1)* %b, float %c) local_unnamed_addr #0 {
  %1 = addrspacecast float addrspace(1)* %a to float *
  %2 = addrspacecast float addrspace(1)* %b to float *
  %call = tail call  float @_Z34_cl_atomic_compare_exchange_strongPU8CLglobalVU7_AtomicfPU8CLglobalff(float * %1, float * %2, float %c)
  ret float %call
}


declare float @_Z34_cl_atomic_compare_exchange_strongPU7CLlocalVU7_AtomicfPU7CLlocalff(float *, float *, float) local_unnamed_addr #0

define spir_func float @_Z30atomic_compare_exchange_strongPU3AS3VU7_AtomicfPU3AS3ff(float addrspace(3)* %a, float addrspace(3)* %b, float %c) local_unnamed_addr #0 {
  %1 = addrspacecast float addrspace(3)* %a to float *
  %2 = addrspacecast float addrspace(3)* %b to float *
  %call = tail call  float @_Z34_cl_atomic_compare_exchange_strongPU7CLlocalVU7_AtomicfPU7CLlocalff(float * %1, float * %2, float %c)
  ret float %call
}


declare float @_Z43_cl_atomic_compare_exchange_strong_explicitPU8CLglobalVU7_AtomicfPU8CLglobalff12memory_order12memory_order(float *, float *, float, i32, i32) local_unnamed_addr #0

define spir_func float @_Z39atomic_compare_exchange_strong_explicitPU3AS1VU7_AtomicfPU3AS1ff12memory_order12memory_order(float addrspace(1)* %a, float addrspace(1)* %b, float %c, i32 %d, i32 %e) local_unnamed_addr #0 {
  %1 = addrspacecast float addrspace(1)* %a to float *
  %2 = addrspacecast float addrspace(1)* %b to float *
  %call = tail call  float @_Z43_cl_atomic_compare_exchange_strong_explicitPU8CLglobalVU7_AtomicfPU8CLglobalff12memory_order12memory_order(float * %1, float * %2, float %c, i32 %d, i32 %e)
  ret float %call
}


declare float @_Z43_cl_atomic_compare_exchange_strong_explicitPU7CLlocalVU7_AtomicfPU7CLlocalff12memory_order12memory_order(float *, float *, float, i32, i32) local_unnamed_addr #0

define spir_func float @_Z39atomic_compare_exchange_strong_explicitPU3AS3VU7_AtomicfPU3AS3ff12memory_order12memory_order(float addrspace(3)* %a, float addrspace(3)* %b, float %c, i32 %d, i32 %e) local_unnamed_addr #0 {
  %1 = addrspacecast float addrspace(3)* %a to float *
  %2 = addrspacecast float addrspace(3)* %b to float *
  %call = tail call  float @_Z43_cl_atomic_compare_exchange_strong_explicitPU7CLlocalVU7_AtomicfPU7CLlocalff12memory_order12memory_order(float * %1, float * %2, float %c, i32 %d, i32 %e)
  ret float %call
}


declare float @_Z43_cl_atomic_compare_exchange_strong_explicitPU8CLglobalVU7_AtomicfPU8CLglobalff12memory_order12memory_order12memory_scope(float *, float *, float, i32, i32, i32) local_unnamed_addr #0

define spir_func float @_Z39atomic_compare_exchange_strong_explicitPU3AS1VU7_AtomicfPU3AS1ff12memory_order12memory_order12memory_scope(float addrspace(1)* %a, float addrspace(1)* %b, float %c, i32 %d, i32 %e, i32 %f) local_unnamed_addr #0 {
  %1 = addrspacecast float addrspace(1)* %a to float *
  %2 = addrspacecast float addrspace(1)* %b to float *
  %call = tail call  float @_Z43_cl_atomic_compare_exchange_strong_explicitPU8CLglobalVU7_AtomicfPU8CLglobalff12memory_order12memory_order12memory_scope(float * %1, float * %2, float %c, i32 %d, i32 %e, i32 %f)
  ret float %call
}


declare float @_Z43_cl_atomic_compare_exchange_strong_explicitPU7CLlocalVU7_AtomicfPU7CLlocalff12memory_order12memory_order12memory_scope(float *, float *, float, i32, i32, i32) local_unnamed_addr #0

define spir_func float @_Z39atomic_compare_exchange_strong_explicitPU3AS3VU7_AtomicfPU3AS3ff12memory_order12memory_order12memory_scope(float addrspace(3)* %a, float addrspace(3)* %b, float %c, i32 %d, i32 %e, i32 %f) local_unnamed_addr #0 {
  %1 = addrspacecast float addrspace(3)* %a to float *
  %2 = addrspacecast float addrspace(3)* %b to float *
  %call = tail call  float @_Z43_cl_atomic_compare_exchange_strong_explicitPU7CLlocalVU7_AtomicfPU7CLlocalff12memory_order12memory_order12memory_scope(float * %1, float * %2, float %c, i32 %d, i32 %e, i32 %f)
  ret float %call
}


declare float @_Z16_cl_atomic_storePU8CLglobalVU7_Atomicff(float *, float) local_unnamed_addr #0

define spir_func float @_Z12atomic_storePU3AS1VU7_Atomicff(float addrspace(1)* %a, float %b) local_unnamed_addr #0 {
  %1 = addrspacecast float addrspace(1)* %a to float *
  %call = tail call  float @_Z16_cl_atomic_storePU8CLglobalVU7_Atomicff(float * %1, float %b)
  ret float %call
}


declare float @_Z16_cl_atomic_storePU7CLlocalVU7_Atomicff(float *, float) local_unnamed_addr #0

define spir_func float @_Z12atomic_storePU3AS3VU7_Atomicff(float addrspace(3)* %a, float %b) local_unnamed_addr #0 {
  %1 = addrspacecast float addrspace(3)* %a to float *
  %call = tail call  float @_Z16_cl_atomic_storePU7CLlocalVU7_Atomicff(float * %1, float %b)
  ret float %call
}


declare float @_Z25_cl_atomic_store_explicitPU8CLglobalVU7_Atomicff12memory_order(float *, float, i32) local_unnamed_addr #0

define spir_func float @_Z21atomic_store_explicitPU3AS1VU7_Atomicff12memory_order(float addrspace(1)* %a, float %b, i32 %c) local_unnamed_addr #0 {
  %1 = addrspacecast float addrspace(1)* %a to float *
  %call = tail call  float @_Z25_cl_atomic_store_explicitPU8CLglobalVU7_Atomicff12memory_order(float * %1, float %b, i32 %c)
  ret float %call
}


declare float @_Z25_cl_atomic_store_explicitPU7CLlocalVU7_Atomicff12memory_order(float *, float, i32) local_unnamed_addr #0

define spir_func float @_Z21atomic_store_explicitPU3AS3VU7_Atomicff12memory_order(float addrspace(3)* %a, float %b, i32 %c) local_unnamed_addr #0 {
  %1 = addrspacecast float addrspace(3)* %a to float *
  %call = tail call  float @_Z25_cl_atomic_store_explicitPU7CLlocalVU7_Atomicff12memory_order(float * %1, float %b, i32 %c)
  ret float %call
}


declare float @_Z25_cl_atomic_store_explicitPU8CLglobalVU7_Atomicff12memory_order12memory_scope(float *, float, i32, i32) local_unnamed_addr #0

define spir_func float @_Z21atomic_store_explicitPU3AS1VU7_Atomicff12memory_order12memory_scope(float addrspace(1)* %a, float %b, i32 %c, i32 %d) local_unnamed_addr #0 {
  %1 = addrspacecast float addrspace(1)* %a to float *
  %call = tail call  float @_Z25_cl_atomic_store_explicitPU8CLglobalVU7_Atomicff12memory_order12memory_scope(float * %1, float %b, i32 %c, i32 %d)
  ret float %call
}


declare float @_Z25_cl_atomic_store_explicitPU7CLlocalVU7_Atomicff12memory_order12memory_scope(float *, float, i32, i32) local_unnamed_addr #0

define spir_func float @_Z21atomic_store_explicitPU3AS3VU7_Atomicff12memory_order12memory_scope(float addrspace(3)* %a, float %b, i32 %c, i32 %d) local_unnamed_addr #0 {
  %1 = addrspacecast float addrspace(3)* %a to float *
  %call = tail call  float @_Z25_cl_atomic_store_explicitPU7CLlocalVU7_Atomicff12memory_order12memory_scope(float * %1, float %b, i32 %c, i32 %d)
  ret float %call
}


declare float @_Z15_cl_atomic_loadPU8CLglobalVU7_Atomicf(float *) local_unnamed_addr #0

define spir_func float @_Z11atomic_loadPU3AS1VU7_Atomicf(float addrspace(1)* %a) local_unnamed_addr #0 {
  %1 = addrspacecast float addrspace(1)* %a to float *
  %call = tail call  float @_Z15_cl_atomic_loadPU8CLglobalVU7_Atomicf(float * %1)
  ret float %call
}


declare float @_Z15_cl_atomic_loadPU7CLlocalVU7_Atomicf(float *) local_unnamed_addr #0

define spir_func float @_Z11atomic_loadPU3AS3VU7_Atomicf(float addrspace(3)* %a) local_unnamed_addr #0 {
  %1 = addrspacecast float addrspace(3)* %a to float *
  %call = tail call  float @_Z15_cl_atomic_loadPU7CLlocalVU7_Atomicf(float * %1)
  ret float %call
}


declare float @_Z24_cl_atomic_load_explicitPU8CLglobalVU7_Atomicf12memory_order(float *, i32) local_unnamed_addr #0

define spir_func float @_Z20atomic_load_explicitPU3AS1VU7_Atomicf12memory_order(float addrspace(1)* %a, i32 %b) local_unnamed_addr #0 {
  %1 = addrspacecast float addrspace(1)* %a to float *
  %call = tail call  float @_Z24_cl_atomic_load_explicitPU8CLglobalVU7_Atomicf12memory_order(float * %1, i32 %b)
  ret float %call
}


declare float @_Z24_cl_atomic_load_explicitPU7CLlocalVU7_Atomicf12memory_order(float *, i32) local_unnamed_addr #0

define spir_func float @_Z20atomic_load_explicitPU3AS3VU7_Atomicf12memory_order(float addrspace(3)* %a, i32 %b) local_unnamed_addr #0 {
  %1 = addrspacecast float addrspace(3)* %a to float *
  %call = tail call  float @_Z24_cl_atomic_load_explicitPU7CLlocalVU7_Atomicf12memory_order(float * %1, i32 %b)
  ret float %call
}


declare float @_Z24_cl_atomic_load_explicitPU8CLglobalVU7_Atomicf12memory_order12memory_scope(float *, i32, i32) local_unnamed_addr #0

define spir_func float @_Z20atomic_load_explicitPU3AS1VU7_Atomicf12memory_order12memory_scope(float addrspace(1)* %a, i32 %b, i32 %c) local_unnamed_addr #0 {
  %1 = addrspacecast float addrspace(1)* %a to float *
  %call = tail call  float @_Z24_cl_atomic_load_explicitPU8CLglobalVU7_Atomicf12memory_order12memory_scope(float * %1, i32 %b, i32 %c)
  ret float %call
}


declare float @_Z24_cl_atomic_load_explicitPU7CLlocalVU7_Atomicf12memory_order12memory_scope(float *, i32, i32) local_unnamed_addr #0

define spir_func float @_Z20atomic_load_explicitPU3AS3VU7_Atomicf12memory_order12memory_scope(float addrspace(3)* %a, i32 %b, i32 %c) local_unnamed_addr #0 {
  %1 = addrspacecast float addrspace(3)* %a to float *
  %call = tail call  float @_Z24_cl_atomic_load_explicitPU7CLlocalVU7_Atomicf12memory_order12memory_scope(float * %1, i32 %b, i32 %c)
  ret float %call
}


declare float @_Z32_cl_atomic_compare_exchange_weakPU8CLglobalVU7_AtomicfPU8CLglobalff(float *, float *, float) local_unnamed_addr #0

define spir_func float @_Z28atomic_compare_exchange_weakPU3AS1VU7_AtomicfPU3AS1ff(float addrspace(1)* %a, float addrspace(1)* %b, float %c) local_unnamed_addr #0 {
  %1 = addrspacecast float addrspace(1)* %a to float *
  %2 = addrspacecast float addrspace(1)* %b to float *
  %call = tail call  float @_Z32_cl_atomic_compare_exchange_weakPU8CLglobalVU7_AtomicfPU8CLglobalff(float * %1, float * %2, float %c)
  ret float %call
}


declare float @_Z32_cl_atomic_compare_exchange_weakPU7CLlocalVU7_AtomicfPU7CLlocalff(float *, float *, float) local_unnamed_addr #0

define spir_func float @_Z28atomic_compare_exchange_weakPU3AS3VU7_AtomicfPU3AS3ff(float addrspace(3)* %a, float addrspace(3)* %b, float %c) local_unnamed_addr #0 {
  %1 = addrspacecast float addrspace(3)* %a to float *
  %2 = addrspacecast float addrspace(3)* %b to float *
  %call = tail call  float @_Z32_cl_atomic_compare_exchange_weakPU7CLlocalVU7_AtomicfPU7CLlocalff(float * %1, float * %2, float %c)
  ret float %call
}


declare float @_Z41_cl_atomic_compare_exchange_weak_explicitPU8CLglobalVU7_AtomicfPU8CLglobalff12memory_order12memory_order(float *, float *, float, i32, i32) local_unnamed_addr #0

define spir_func float @_Z37atomic_compare_exchange_weak_explicitPU3AS1VU7_AtomicfPU3AS1ff12memory_order12memory_order(float addrspace(1)* %a, float addrspace(1)* %b, float %c, i32 %d, i32 %e) local_unnamed_addr #0 {
  %1 = addrspacecast float addrspace(1)* %a to float *
  %2 = addrspacecast float addrspace(1)* %b to float *
  %call = tail call  float @_Z41_cl_atomic_compare_exchange_weak_explicitPU8CLglobalVU7_AtomicfPU8CLglobalff12memory_order12memory_order(float * %1, float * %2, float %c, i32 %d, i32 %e)
  ret float %call
}


declare float @_Z41_cl_atomic_compare_exchange_weak_explicitPU7CLlocalVU7_AtomicfPU7CLlocalff12memory_order12memory_order(float *, float *, float, i32, i32) local_unnamed_addr #0

define spir_func float @_Z37atomic_compare_exchange_weak_explicitPU3AS3VU7_AtomicfPU3AS3ff12memory_order12memory_order(float addrspace(3)* %a, float addrspace(3)* %b, float %c, i32 %d, i32 %e) local_unnamed_addr #0 {
  %1 = addrspacecast float addrspace(3)* %a to float *
  %2 = addrspacecast float addrspace(3)* %b to float *
  %call = tail call  float @_Z41_cl_atomic_compare_exchange_weak_explicitPU7CLlocalVU7_AtomicfPU7CLlocalff12memory_order12memory_order(float * %1, float * %2, float %c, i32 %d, i32 %e)
  ret float %call
}


declare float @_Z41_cl_atomic_compare_exchange_weak_explicitPU8CLglobalVU7_AtomicfPU8CLglobalff12memory_order12memory_order12memory_scope(float *, float *, float, i32, i32, i32) local_unnamed_addr #0

define spir_func float @_Z37atomic_compare_exchange_weak_explicitPU3AS1VU7_AtomicfPU3AS1ff12memory_order12memory_order12memory_scope(float addrspace(1)* %a, float addrspace(1)* %b, float %c, i32 %d, i32 %e, i32 %f) local_unnamed_addr #0 {
  %1 = addrspacecast float addrspace(1)* %a to float *
  %2 = addrspacecast float addrspace(1)* %b to float *
  %call = tail call  float @_Z41_cl_atomic_compare_exchange_weak_explicitPU8CLglobalVU7_AtomicfPU8CLglobalff12memory_order12memory_order12memory_scope(float * %1, float * %2, float %c, i32 %d, i32 %e, i32 %f)
  ret float %call
}


declare float @_Z41_cl_atomic_compare_exchange_weak_explicitPU7CLlocalVU7_AtomicfPU7CLlocalff12memory_order12memory_order12memory_scope(float *, float *, float, i32, i32, i32) local_unnamed_addr #0

define spir_func float @_Z37atomic_compare_exchange_weak_explicitPU3AS3VU7_AtomicfPU3AS3ff12memory_order12memory_order12memory_scope(float addrspace(3)* %a, float addrspace(3)* %b, float %c, i32 %d, i32 %e, i32 %f) local_unnamed_addr #0 {
  %1 = addrspacecast float addrspace(3)* %a to float *
  %2 = addrspacecast float addrspace(3)* %b to float *
  %call = tail call  float @_Z41_cl_atomic_compare_exchange_weak_explicitPU7CLlocalVU7_AtomicfPU7CLlocalff12memory_order12memory_order12memory_scope(float * %1, float * %2, float %c, i32 %d, i32 %e, i32 %f)
  ret float %call
}


declare float @_Z19_cl_atomic_exchangePU8CLglobalVU7_Atomicff(float *, float) local_unnamed_addr #0

define spir_func float @_Z15atomic_exchangePU3AS1VU7_Atomicff(float addrspace(1)* %a, float %b) local_unnamed_addr #0 {
  %1 = addrspacecast float addrspace(1)* %a to float *
  %call = tail call  float @_Z19_cl_atomic_exchangePU8CLglobalVU7_Atomicff(float * %1, float %b)
  ret float %call
}


declare float @_Z19_cl_atomic_exchangePU7CLlocalVU7_Atomicff(float *, float) local_unnamed_addr #0

define spir_func float @_Z15atomic_exchangePU3AS3VU7_Atomicff(float addrspace(3)* %a, float %b) local_unnamed_addr #0 {
  %1 = addrspacecast float addrspace(3)* %a to float *
  %call = tail call  float @_Z19_cl_atomic_exchangePU7CLlocalVU7_Atomicff(float * %1, float %b)
  ret float %call
}


declare float @_Z28_cl_atomic_exchange_explicitPU8CLglobalVU7_Atomicff12memory_order(float *, float, i32) local_unnamed_addr #0

define spir_func float @_Z24atomic_exchange_explicitPU3AS1VU7_Atomicff12memory_order(float addrspace(1)* %a, float %b, i32 %c) local_unnamed_addr #0 {
  %1 = addrspacecast float addrspace(1)* %a to float *
  %call = tail call  float @_Z28_cl_atomic_exchange_explicitPU8CLglobalVU7_Atomicff12memory_order(float * %1, float %b, i32 %c)
  ret float %call
}


declare float @_Z28_cl_atomic_exchange_explicitPU7CLlocalVU7_Atomicff12memory_order(float *, float, i32) local_unnamed_addr #0

define spir_func float @_Z24atomic_exchange_explicitPU3AS3VU7_Atomicff12memory_order(float addrspace(3)* %a, float %b, i32 %c) local_unnamed_addr #0 {
  %1 = addrspacecast float addrspace(3)* %a to float *
  %call = tail call  float @_Z28_cl_atomic_exchange_explicitPU7CLlocalVU7_Atomicff12memory_order(float * %1, float %b, i32 %c)
  ret float %call
}


declare float @_Z28_cl_atomic_exchange_explicitPU8CLglobalVU7_Atomicff12memory_order12memory_scope(float *, float, i32, i32) local_unnamed_addr #0

define spir_func float @_Z24atomic_exchange_explicitPU3AS1VU7_Atomicff12memory_order12memory_scope(float addrspace(1)* %a, float %b, i32 %c, i32 %d) local_unnamed_addr #0 {
  %1 = addrspacecast float addrspace(1)* %a to float *
  %call = tail call  float @_Z28_cl_atomic_exchange_explicitPU8CLglobalVU7_Atomicff12memory_order12memory_scope(float * %1, float %b, i32 %c, i32 %d)
  ret float %call
}


declare float @_Z28_cl_atomic_exchange_explicitPU7CLlocalVU7_Atomicff12memory_order12memory_scope(float *, float, i32, i32) local_unnamed_addr #0

define spir_func float @_Z24atomic_exchange_explicitPU3AS3VU7_Atomicff12memory_order12memory_scope(float addrspace(3)* %a, float %b, i32 %c, i32 %d) local_unnamed_addr #0 {
  %1 = addrspacecast float addrspace(3)* %a to float *
  %call = tail call  float @_Z28_cl_atomic_exchange_explicitPU7CLlocalVU7_Atomicff12memory_order12memory_scope(float * %1, float %b, i32 %c, i32 %d)
  ret float %call
}


declare void @_Z15_cl_atomic_initPU8CLglobalVU7_Atomicdd(double *, double) local_unnamed_addr #0

define spir_func void @_Z11atomic_initPU3AS1VU7_Atomicdd(double addrspace(1)* %a, double %b) local_unnamed_addr #0 {
  %1 = addrspacecast double addrspace(1)* %a to double *
  tail call void @_Z15_cl_atomic_initPU8CLglobalVU7_Atomicdd(double * %1, double %b)
  ret void
}


declare void @_Z15_cl_atomic_initPU7CLlocalVU7_Atomicdd(double *, double) local_unnamed_addr #0

define spir_func void @_Z11atomic_initPU3AS3VU7_Atomicdd(double addrspace(3)* %a, double %b) local_unnamed_addr #0 {
  %1 = addrspacecast double addrspace(3)* %a to double *
  tail call void @_Z15_cl_atomic_initPU7CLlocalVU7_Atomicdd(double * %1, double %b)
  ret void
}


declare double @_Z34_cl_atomic_compare_exchange_strongPU8CLglobalVU7_AtomicdPU8CLglobaldd(double *, double *, double) local_unnamed_addr #0

define spir_func double @_Z30atomic_compare_exchange_strongPU3AS1VU7_AtomicdPU3AS1dd(double addrspace(1)* %a, double addrspace(1)* %b, double %c) local_unnamed_addr #0 {
  %1 = addrspacecast double addrspace(1)* %a to double *
  %2 = addrspacecast double addrspace(1)* %b to double *
  %call = tail call  double @_Z34_cl_atomic_compare_exchange_strongPU8CLglobalVU7_AtomicdPU8CLglobaldd(double * %1, double * %2, double %c)
  ret double %call
}


declare double @_Z34_cl_atomic_compare_exchange_strongPU7CLlocalVU7_AtomicdPU7CLlocaldd(double *, double *, double) local_unnamed_addr #0

define spir_func double @_Z30atomic_compare_exchange_strongPU3AS3VU7_AtomicdPU3AS3dd(double addrspace(3)* %a, double addrspace(3)* %b, double %c) local_unnamed_addr #0 {
  %1 = addrspacecast double addrspace(3)* %a to double *
  %2 = addrspacecast double addrspace(3)* %b to double *
  %call = tail call  double @_Z34_cl_atomic_compare_exchange_strongPU7CLlocalVU7_AtomicdPU7CLlocaldd(double * %1, double * %2, double %c)
  ret double %call
}


declare double @_Z43_cl_atomic_compare_exchange_strong_explicitPU8CLglobalVU7_AtomicdPU8CLglobaldd12memory_order12memory_order(double *, double *, double, i32, i32) local_unnamed_addr #0

define spir_func double @_Z39atomic_compare_exchange_strong_explicitPU3AS1VU7_AtomicdPU3AS1dd12memory_order12memory_order(double addrspace(1)* %a, double addrspace(1)* %b, double %c, i32 %d, i32 %e) local_unnamed_addr #0 {
  %1 = addrspacecast double addrspace(1)* %a to double *
  %2 = addrspacecast double addrspace(1)* %b to double *
  %call = tail call  double @_Z43_cl_atomic_compare_exchange_strong_explicitPU8CLglobalVU7_AtomicdPU8CLglobaldd12memory_order12memory_order(double * %1, double * %2, double %c, i32 %d, i32 %e)
  ret double %call
}


declare double @_Z43_cl_atomic_compare_exchange_strong_explicitPU7CLlocalVU7_AtomicdPU7CLlocaldd12memory_order12memory_order(double *, double *, double, i32, i32) local_unnamed_addr #0

define spir_func double @_Z39atomic_compare_exchange_strong_explicitPU3AS3VU7_AtomicdPU3AS3dd12memory_order12memory_order(double addrspace(3)* %a, double addrspace(3)* %b, double %c, i32 %d, i32 %e) local_unnamed_addr #0 {
  %1 = addrspacecast double addrspace(3)* %a to double *
  %2 = addrspacecast double addrspace(3)* %b to double *
  %call = tail call  double @_Z43_cl_atomic_compare_exchange_strong_explicitPU7CLlocalVU7_AtomicdPU7CLlocaldd12memory_order12memory_order(double * %1, double * %2, double %c, i32 %d, i32 %e)
  ret double %call
}


declare double @_Z43_cl_atomic_compare_exchange_strong_explicitPU8CLglobalVU7_AtomicdPU8CLglobaldd12memory_order12memory_order12memory_scope(double *, double *, double, i32, i32, i32) local_unnamed_addr #0

define spir_func double @_Z39atomic_compare_exchange_strong_explicitPU3AS1VU7_AtomicdPU3AS1dd12memory_order12memory_order12memory_scope(double addrspace(1)* %a, double addrspace(1)* %b, double %c, i32 %d, i32 %e, i32 %f) local_unnamed_addr #0 {
  %1 = addrspacecast double addrspace(1)* %a to double *
  %2 = addrspacecast double addrspace(1)* %b to double *
  %call = tail call  double @_Z43_cl_atomic_compare_exchange_strong_explicitPU8CLglobalVU7_AtomicdPU8CLglobaldd12memory_order12memory_order12memory_scope(double * %1, double * %2, double %c, i32 %d, i32 %e, i32 %f)
  ret double %call
}


declare double @_Z43_cl_atomic_compare_exchange_strong_explicitPU7CLlocalVU7_AtomicdPU7CLlocaldd12memory_order12memory_order12memory_scope(double *, double *, double, i32, i32, i32) local_unnamed_addr #0

define spir_func double @_Z39atomic_compare_exchange_strong_explicitPU3AS3VU7_AtomicdPU3AS3dd12memory_order12memory_order12memory_scope(double addrspace(3)* %a, double addrspace(3)* %b, double %c, i32 %d, i32 %e, i32 %f) local_unnamed_addr #0 {
  %1 = addrspacecast double addrspace(3)* %a to double *
  %2 = addrspacecast double addrspace(3)* %b to double *
  %call = tail call  double @_Z43_cl_atomic_compare_exchange_strong_explicitPU7CLlocalVU7_AtomicdPU7CLlocaldd12memory_order12memory_order12memory_scope(double * %1, double * %2, double %c, i32 %d, i32 %e, i32 %f)
  ret double %call
}


declare double @_Z16_cl_atomic_storePU8CLglobalVU7_Atomicdd(double *, double) local_unnamed_addr #0

define spir_func double @_Z12atomic_storePU3AS1VU7_Atomicdd(double addrspace(1)* %a, double %b) local_unnamed_addr #0 {
  %1 = addrspacecast double addrspace(1)* %a to double *
  %call = tail call  double @_Z16_cl_atomic_storePU8CLglobalVU7_Atomicdd(double * %1, double %b)
  ret double %call
}


declare double @_Z16_cl_atomic_storePU7CLlocalVU7_Atomicdd(double *, double) local_unnamed_addr #0

define spir_func double @_Z12atomic_storePU3AS3VU7_Atomicdd(double addrspace(3)* %a, double %b) local_unnamed_addr #0 {
  %1 = addrspacecast double addrspace(3)* %a to double *
  %call = tail call  double @_Z16_cl_atomic_storePU7CLlocalVU7_Atomicdd(double * %1, double %b)
  ret double %call
}


declare double @_Z25_cl_atomic_store_explicitPU8CLglobalVU7_Atomicdd12memory_order(double *, double, i32) local_unnamed_addr #0

define spir_func double @_Z21atomic_store_explicitPU3AS1VU7_Atomicdd12memory_order(double addrspace(1)* %a, double %b, i32 %c) local_unnamed_addr #0 {
  %1 = addrspacecast double addrspace(1)* %a to double *
  %call = tail call  double @_Z25_cl_atomic_store_explicitPU8CLglobalVU7_Atomicdd12memory_order(double * %1, double %b, i32 %c)
  ret double %call
}


declare double @_Z25_cl_atomic_store_explicitPU7CLlocalVU7_Atomicdd12memory_order(double *, double, i32) local_unnamed_addr #0

define spir_func double @_Z21atomic_store_explicitPU3AS3VU7_Atomicdd12memory_order(double addrspace(3)* %a, double %b, i32 %c) local_unnamed_addr #0 {
  %1 = addrspacecast double addrspace(3)* %a to double *
  %call = tail call  double @_Z25_cl_atomic_store_explicitPU7CLlocalVU7_Atomicdd12memory_order(double * %1, double %b, i32 %c)
  ret double %call
}


declare double @_Z25_cl_atomic_store_explicitPU8CLglobalVU7_Atomicdd12memory_order12memory_scope(double *, double, i32, i32) local_unnamed_addr #0

define spir_func double @_Z21atomic_store_explicitPU3AS1VU7_Atomicdd12memory_order12memory_scope(double addrspace(1)* %a, double %b, i32 %c, i32 %d) local_unnamed_addr #0 {
  %1 = addrspacecast double addrspace(1)* %a to double *
  %call = tail call  double @_Z25_cl_atomic_store_explicitPU8CLglobalVU7_Atomicdd12memory_order12memory_scope(double * %1, double %b, i32 %c, i32 %d)
  ret double %call
}


declare double @_Z25_cl_atomic_store_explicitPU7CLlocalVU7_Atomicdd12memory_order12memory_scope(double *, double, i32, i32) local_unnamed_addr #0

define spir_func double @_Z21atomic_store_explicitPU3AS3VU7_Atomicdd12memory_order12memory_scope(double addrspace(3)* %a, double %b, i32 %c, i32 %d) local_unnamed_addr #0 {
  %1 = addrspacecast double addrspace(3)* %a to double *
  %call = tail call  double @_Z25_cl_atomic_store_explicitPU7CLlocalVU7_Atomicdd12memory_order12memory_scope(double * %1, double %b, i32 %c, i32 %d)
  ret double %call
}


declare double @_Z15_cl_atomic_loadPU8CLglobalVU7_Atomicd(double *) local_unnamed_addr #0

define spir_func double @_Z11atomic_loadPU3AS1VU7_Atomicd(double addrspace(1)* %a) local_unnamed_addr #0 {
  %1 = addrspacecast double addrspace(1)* %a to double *
  %call = tail call  double @_Z15_cl_atomic_loadPU8CLglobalVU7_Atomicd(double * %1)
  ret double %call
}


declare double @_Z15_cl_atomic_loadPU7CLlocalVU7_Atomicd(double *) local_unnamed_addr #0

define spir_func double @_Z11atomic_loadPU3AS3VU7_Atomicd(double addrspace(3)* %a) local_unnamed_addr #0 {
  %1 = addrspacecast double addrspace(3)* %a to double *
  %call = tail call  double @_Z15_cl_atomic_loadPU7CLlocalVU7_Atomicd(double * %1)
  ret double %call
}


declare double @_Z24_cl_atomic_load_explicitPU8CLglobalVU7_Atomicd12memory_order(double *, i32) local_unnamed_addr #0

define spir_func double @_Z20atomic_load_explicitPU3AS1VU7_Atomicd12memory_order(double addrspace(1)* %a, i32 %b) local_unnamed_addr #0 {
  %1 = addrspacecast double addrspace(1)* %a to double *
  %call = tail call  double @_Z24_cl_atomic_load_explicitPU8CLglobalVU7_Atomicd12memory_order(double * %1, i32 %b)
  ret double %call
}


declare double @_Z24_cl_atomic_load_explicitPU7CLlocalVU7_Atomicd12memory_order(double *, i32) local_unnamed_addr #0

define spir_func double @_Z20atomic_load_explicitPU3AS3VU7_Atomicd12memory_order(double addrspace(3)* %a, i32 %b) local_unnamed_addr #0 {
  %1 = addrspacecast double addrspace(3)* %a to double *
  %call = tail call  double @_Z24_cl_atomic_load_explicitPU7CLlocalVU7_Atomicd12memory_order(double * %1, i32 %b)
  ret double %call
}


declare double @_Z24_cl_atomic_load_explicitPU8CLglobalVU7_Atomicd12memory_order12memory_scope(double *, i32, i32) local_unnamed_addr #0

define spir_func double @_Z20atomic_load_explicitPU3AS1VU7_Atomicd12memory_order12memory_scope(double addrspace(1)* %a, i32 %b, i32 %c) local_unnamed_addr #0 {
  %1 = addrspacecast double addrspace(1)* %a to double *
  %call = tail call  double @_Z24_cl_atomic_load_explicitPU8CLglobalVU7_Atomicd12memory_order12memory_scope(double * %1, i32 %b, i32 %c)
  ret double %call
}


declare double @_Z24_cl_atomic_load_explicitPU7CLlocalVU7_Atomicd12memory_order12memory_scope(double *, i32, i32) local_unnamed_addr #0

define spir_func double @_Z20atomic_load_explicitPU3AS3VU7_Atomicd12memory_order12memory_scope(double addrspace(3)* %a, i32 %b, i32 %c) local_unnamed_addr #0 {
  %1 = addrspacecast double addrspace(3)* %a to double *
  %call = tail call  double @_Z24_cl_atomic_load_explicitPU7CLlocalVU7_Atomicd12memory_order12memory_scope(double * %1, i32 %b, i32 %c)
  ret double %call
}


declare double @_Z32_cl_atomic_compare_exchange_weakPU8CLglobalVU7_AtomicdPU8CLglobaldd(double *, double *, double) local_unnamed_addr #0

define spir_func double @_Z28atomic_compare_exchange_weakPU3AS1VU7_AtomicdPU3AS1dd(double addrspace(1)* %a, double addrspace(1)* %b, double %c) local_unnamed_addr #0 {
  %1 = addrspacecast double addrspace(1)* %a to double *
  %2 = addrspacecast double addrspace(1)* %b to double *
  %call = tail call  double @_Z32_cl_atomic_compare_exchange_weakPU8CLglobalVU7_AtomicdPU8CLglobaldd(double * %1, double * %2, double %c)
  ret double %call
}


declare double @_Z32_cl_atomic_compare_exchange_weakPU7CLlocalVU7_AtomicdPU7CLlocaldd(double *, double *, double) local_unnamed_addr #0

define spir_func double @_Z28atomic_compare_exchange_weakPU3AS3VU7_AtomicdPU3AS3dd(double addrspace(3)* %a, double addrspace(3)* %b, double %c) local_unnamed_addr #0 {
  %1 = addrspacecast double addrspace(3)* %a to double *
  %2 = addrspacecast double addrspace(3)* %b to double *
  %call = tail call  double @_Z32_cl_atomic_compare_exchange_weakPU7CLlocalVU7_AtomicdPU7CLlocaldd(double * %1, double * %2, double %c)
  ret double %call
}


declare double @_Z41_cl_atomic_compare_exchange_weak_explicitPU8CLglobalVU7_AtomicdPU8CLglobaldd12memory_order12memory_order(double *, double *, double, i32, i32) local_unnamed_addr #0

define spir_func double @_Z37atomic_compare_exchange_weak_explicitPU3AS1VU7_AtomicdPU3AS1dd12memory_order12memory_order(double addrspace(1)* %a, double addrspace(1)* %b, double %c, i32 %d, i32 %e) local_unnamed_addr #0 {
  %1 = addrspacecast double addrspace(1)* %a to double *
  %2 = addrspacecast double addrspace(1)* %b to double *
  %call = tail call  double @_Z41_cl_atomic_compare_exchange_weak_explicitPU8CLglobalVU7_AtomicdPU8CLglobaldd12memory_order12memory_order(double * %1, double * %2, double %c, i32 %d, i32 %e)
  ret double %call
}


declare double @_Z41_cl_atomic_compare_exchange_weak_explicitPU7CLlocalVU7_AtomicdPU7CLlocaldd12memory_order12memory_order(double *, double *, double, i32, i32) local_unnamed_addr #0

define spir_func double @_Z37atomic_compare_exchange_weak_explicitPU3AS3VU7_AtomicdPU3AS3dd12memory_order12memory_order(double addrspace(3)* %a, double addrspace(3)* %b, double %c, i32 %d, i32 %e) local_unnamed_addr #0 {
  %1 = addrspacecast double addrspace(3)* %a to double *
  %2 = addrspacecast double addrspace(3)* %b to double *
  %call = tail call  double @_Z41_cl_atomic_compare_exchange_weak_explicitPU7CLlocalVU7_AtomicdPU7CLlocaldd12memory_order12memory_order(double * %1, double * %2, double %c, i32 %d, i32 %e)
  ret double %call
}


declare double @_Z41_cl_atomic_compare_exchange_weak_explicitPU8CLglobalVU7_AtomicdPU8CLglobaldd12memory_order12memory_order12memory_scope(double *, double *, double, i32, i32, i32) local_unnamed_addr #0

define spir_func double @_Z37atomic_compare_exchange_weak_explicitPU3AS1VU7_AtomicdPU3AS1dd12memory_order12memory_order12memory_scope(double addrspace(1)* %a, double addrspace(1)* %b, double %c, i32 %d, i32 %e, i32 %f) local_unnamed_addr #0 {
  %1 = addrspacecast double addrspace(1)* %a to double *
  %2 = addrspacecast double addrspace(1)* %b to double *
  %call = tail call  double @_Z41_cl_atomic_compare_exchange_weak_explicitPU8CLglobalVU7_AtomicdPU8CLglobaldd12memory_order12memory_order12memory_scope(double * %1, double * %2, double %c, i32 %d, i32 %e, i32 %f)
  ret double %call
}


declare double @_Z41_cl_atomic_compare_exchange_weak_explicitPU7CLlocalVU7_AtomicdPU7CLlocaldd12memory_order12memory_order12memory_scope(double *, double *, double, i32, i32, i32) local_unnamed_addr #0

define spir_func double @_Z37atomic_compare_exchange_weak_explicitPU3AS3VU7_AtomicdPU3AS3dd12memory_order12memory_order12memory_scope(double addrspace(3)* %a, double addrspace(3)* %b, double %c, i32 %d, i32 %e, i32 %f) local_unnamed_addr #0 {
  %1 = addrspacecast double addrspace(3)* %a to double *
  %2 = addrspacecast double addrspace(3)* %b to double *
  %call = tail call  double @_Z41_cl_atomic_compare_exchange_weak_explicitPU7CLlocalVU7_AtomicdPU7CLlocaldd12memory_order12memory_order12memory_scope(double * %1, double * %2, double %c, i32 %d, i32 %e, i32 %f)
  ret double %call
}


declare double @_Z19_cl_atomic_exchangePU8CLglobalVU7_Atomicdd(double *, double) local_unnamed_addr #0

define spir_func double @_Z15atomic_exchangePU3AS1VU7_Atomicdd(double addrspace(1)* %a, double %b) local_unnamed_addr #0 {
  %1 = addrspacecast double addrspace(1)* %a to double *
  %call = tail call  double @_Z19_cl_atomic_exchangePU8CLglobalVU7_Atomicdd(double * %1, double %b)
  ret double %call
}


declare double @_Z19_cl_atomic_exchangePU7CLlocalVU7_Atomicdd(double *, double) local_unnamed_addr #0

define spir_func double @_Z15atomic_exchangePU3AS3VU7_Atomicdd(double addrspace(3)* %a, double %b) local_unnamed_addr #0 {
  %1 = addrspacecast double addrspace(3)* %a to double *
  %call = tail call  double @_Z19_cl_atomic_exchangePU7CLlocalVU7_Atomicdd(double * %1, double %b)
  ret double %call
}


declare double @_Z28_cl_atomic_exchange_explicitPU8CLglobalVU7_Atomicdd12memory_order(double *, double, i32) local_unnamed_addr #0

define spir_func double @_Z24atomic_exchange_explicitPU3AS1VU7_Atomicdd12memory_order(double addrspace(1)* %a, double %b, i32 %c) local_unnamed_addr #0 {
  %1 = addrspacecast double addrspace(1)* %a to double *
  %call = tail call  double @_Z28_cl_atomic_exchange_explicitPU8CLglobalVU7_Atomicdd12memory_order(double * %1, double %b, i32 %c)
  ret double %call
}


declare double @_Z28_cl_atomic_exchange_explicitPU7CLlocalVU7_Atomicdd12memory_order(double *, double, i32) local_unnamed_addr #0

define spir_func double @_Z24atomic_exchange_explicitPU3AS3VU7_Atomicdd12memory_order(double addrspace(3)* %a, double %b, i32 %c) local_unnamed_addr #0 {
  %1 = addrspacecast double addrspace(3)* %a to double *
  %call = tail call  double @_Z28_cl_atomic_exchange_explicitPU7CLlocalVU7_Atomicdd12memory_order(double * %1, double %b, i32 %c)
  ret double %call
}


declare double @_Z28_cl_atomic_exchange_explicitPU8CLglobalVU7_Atomicdd12memory_order12memory_scope(double *, double, i32, i32) local_unnamed_addr #0

define spir_func double @_Z24atomic_exchange_explicitPU3AS1VU7_Atomicdd12memory_order12memory_scope(double addrspace(1)* %a, double %b, i32 %c, i32 %d) local_unnamed_addr #0 {
  %1 = addrspacecast double addrspace(1)* %a to double *
  %call = tail call  double @_Z28_cl_atomic_exchange_explicitPU8CLglobalVU7_Atomicdd12memory_order12memory_scope(double * %1, double %b, i32 %c, i32 %d)
  ret double %call
}


declare double @_Z28_cl_atomic_exchange_explicitPU7CLlocalVU7_Atomicdd12memory_order12memory_scope(double *, double, i32, i32) local_unnamed_addr #0

define spir_func double @_Z24atomic_exchange_explicitPU3AS3VU7_Atomicdd12memory_order12memory_scope(double addrspace(3)* %a, double %b, i32 %c, i32 %d) local_unnamed_addr #0 {
  %1 = addrspacecast double addrspace(3)* %a to double *
  %call = tail call  double @_Z28_cl_atomic_exchange_explicitPU7CLlocalVU7_Atomicdd12memory_order12memory_scope(double * %1, double %b, i32 %c, i32 %d)
  ret double %call
}


declare  zeroext i1 @_Z28_cl_atomic_flag_test_and_setPU8CLglobalVU7_Atomici(i32 *) local_unnamed_addr #0

define spir_func  zeroext i1 @_Z24atomic_flag_test_and_setPU3AS1VU7_Atomici(i32 addrspace(1)* %a) local_unnamed_addr #0 {
  %1 = addrspacecast i32 addrspace(1)* %a to i32 *
  %call = tail call  zeroext  i1 @_Z28_cl_atomic_flag_test_and_setPU8CLglobalVU7_Atomici(i32 * %1)
  ret i1 %call
}


declare  zeroext i1 @_Z28_cl_atomic_flag_test_and_setPU7CLlocalVU7_Atomici(i32 *) local_unnamed_addr #0

define spir_func  zeroext i1 @_Z24atomic_flag_test_and_setPU3AS3VU7_Atomici(i32 addrspace(3)* %a) local_unnamed_addr #0 {
  %1 = addrspacecast i32 addrspace(3)* %a to i32 *
  %call = tail call  zeroext  i1 @_Z28_cl_atomic_flag_test_and_setPU7CLlocalVU7_Atomici(i32 * %1)
  ret i1 %call
}


declare  zeroext i1 @_Z37_cl_atomic_flag_test_and_set_explicitPU8CLglobalVU7_Atomici12memory_order(i32 *, i32) local_unnamed_addr #0

define spir_func  zeroext i1 @_Z33atomic_flag_test_and_set_explicitPU3AS1VU7_Atomici12memory_order(i32 addrspace(1)* %a, i32 %b) local_unnamed_addr #0 {
  %1 = addrspacecast i32 addrspace(1)* %a to i32 *
  %call = tail call  zeroext  i1 @_Z37_cl_atomic_flag_test_and_set_explicitPU8CLglobalVU7_Atomici12memory_order(i32 * %1, i32 %b)
  ret i1 %call
}


declare  zeroext i1 @_Z37_cl_atomic_flag_test_and_set_explicitPU7CLlocalVU7_Atomici12memory_order(i32 *, i32) local_unnamed_addr #0

define spir_func  zeroext i1 @_Z33atomic_flag_test_and_set_explicitPU3AS3VU7_Atomici12memory_order(i32 addrspace(3)* %a, i32 %b) local_unnamed_addr #0 {
  %1 = addrspacecast i32 addrspace(3)* %a to i32 *
  %call = tail call  zeroext  i1 @_Z37_cl_atomic_flag_test_and_set_explicitPU7CLlocalVU7_Atomici12memory_order(i32 * %1, i32 %b)
  ret i1 %call
}


declare  zeroext i1 @_Z37_cl_atomic_flag_test_and_set_explicitPU8CLglobalVU7_Atomici12memory_order12memory_scope(i32 *, i32, i32) local_unnamed_addr #0

define spir_func  zeroext i1 @_Z33atomic_flag_test_and_set_explicitPU3AS1VU7_Atomici12memory_order12memory_scope(i32 addrspace(1)* %a, i32 %b, i32 %c) local_unnamed_addr #0 {
  %1 = addrspacecast i32 addrspace(1)* %a to i32 *
  %call = tail call  zeroext  i1 @_Z37_cl_atomic_flag_test_and_set_explicitPU8CLglobalVU7_Atomici12memory_order12memory_scope(i32 * %1, i32 %b, i32 %c)
  ret i1 %call
}


declare  zeroext i1 @_Z37_cl_atomic_flag_test_and_set_explicitPU7CLlocalVU7_Atomici12memory_order12memory_scope(i32 *, i32, i32) local_unnamed_addr #0

define spir_func  zeroext i1 @_Z33atomic_flag_test_and_set_explicitPU3AS3VU7_Atomici12memory_order12memory_scope(i32 addrspace(3)* %a, i32 %b, i32 %c) local_unnamed_addr #0 {
  %1 = addrspacecast i32 addrspace(3)* %a to i32 *
  %call = tail call  zeroext  i1 @_Z37_cl_atomic_flag_test_and_set_explicitPU7CLlocalVU7_Atomici12memory_order12memory_scope(i32 * %1, i32 %b, i32 %c)
  ret i1 %call
}


declare  zeroext i1 @_Z21_cl_atomic_flag_clearPU8CLglobalVU7_Atomici(i32 *) local_unnamed_addr #0

define spir_func  zeroext i1 @_Z17atomic_flag_clearPU3AS1VU7_Atomici(i32 addrspace(1)* %a) local_unnamed_addr #0 {
  %1 = addrspacecast i32 addrspace(1)* %a to i32 *
  %call = tail call  zeroext  i1 @_Z21_cl_atomic_flag_clearPU8CLglobalVU7_Atomici(i32 * %1)
  ret i1 %call
}


declare  zeroext i1 @_Z21_cl_atomic_flag_clearPU7CLlocalVU7_Atomici(i32 *) local_unnamed_addr #0

define spir_func  zeroext i1 @_Z17atomic_flag_clearPU3AS3VU7_Atomici(i32 addrspace(3)* %a) local_unnamed_addr #0 {
  %1 = addrspacecast i32 addrspace(3)* %a to i32 *
  %call = tail call  zeroext  i1 @_Z21_cl_atomic_flag_clearPU7CLlocalVU7_Atomici(i32 * %1)
  ret i1 %call
}


declare  zeroext i1 @_Z30_cl_atomic_flag_clear_explicitPU8CLglobalVU7_Atomici12memory_order(i32 *, i32) local_unnamed_addr #0

define spir_func  zeroext i1 @_Z26atomic_flag_clear_explicitPU3AS1VU7_Atomici12memory_order(i32 addrspace(1)* %a, i32 %b) local_unnamed_addr #0 {
  %1 = addrspacecast i32 addrspace(1)* %a to i32 *
  %call = tail call  zeroext  i1 @_Z30_cl_atomic_flag_clear_explicitPU8CLglobalVU7_Atomici12memory_order(i32 * %1, i32 %b)
  ret i1 %call
}


declare  zeroext i1 @_Z30_cl_atomic_flag_clear_explicitPU7CLlocalVU7_Atomici12memory_order(i32 *, i32) local_unnamed_addr #0

define spir_func  zeroext i1 @_Z26atomic_flag_clear_explicitPU3AS3VU7_Atomici12memory_order(i32 addrspace(3)* %a, i32 %b) local_unnamed_addr #0 {
  %1 = addrspacecast i32 addrspace(3)* %a to i32 *
  %call = tail call  zeroext  i1 @_Z30_cl_atomic_flag_clear_explicitPU7CLlocalVU7_Atomici12memory_order(i32 * %1, i32 %b)
  ret i1 %call
}


declare  zeroext i1 @_Z30_cl_atomic_flag_clear_explicitPU8CLglobalVU7_Atomici12memory_order12memory_scope(i32 *, i32, i32) local_unnamed_addr #0

define spir_func  zeroext i1 @_Z26atomic_flag_clear_explicitPU3AS1VU7_Atomici12memory_order12memory_scope(i32 addrspace(1)* %a, i32 %b, i32 %c) local_unnamed_addr #0 {
  %1 = addrspacecast i32 addrspace(1)* %a to i32 *
  %call = tail call  zeroext  i1 @_Z30_cl_atomic_flag_clear_explicitPU8CLglobalVU7_Atomici12memory_order12memory_scope(i32 * %1, i32 %b, i32 %c)
  ret i1 %call
}


declare  zeroext i1 @_Z30_cl_atomic_flag_clear_explicitPU7CLlocalVU7_Atomici12memory_order12memory_scope(i32 *, i32, i32) local_unnamed_addr #0

define spir_func  zeroext i1 @_Z26atomic_flag_clear_explicitPU3AS3VU7_Atomici12memory_order12memory_scope(i32 addrspace(3)* %a, i32 %b, i32 %c) local_unnamed_addr #0 {
  %1 = addrspacecast i32 addrspace(3)* %a to i32 *
  %call = tail call  zeroext  i1 @_Z30_cl_atomic_flag_clear_explicitPU7CLlocalVU7_Atomici12memory_order12memory_scope(i32 * %1, i32 %b, i32 %c)
  ret i1 %call
}


declare i32 @_Z9_cl_mul24ii(i32, i32) local_unnamed_addr #0

define spir_func i32 @_Z5mul24ii(i32 %a, i32 %b) local_unnamed_addr #0 {
  %call = tail call  i32 @_Z9_cl_mul24ii(i32 %a, i32 %b)
  ret i32 %call
}


declare i32 @_Z9_cl_mul24jj(i32, i32) local_unnamed_addr #0

define spir_func i32 @_Z5mul24jj(i32 %a, i32 %b) local_unnamed_addr #0 {
  %call = tail call  i32 @_Z9_cl_mul24jj(i32 %a, i32 %b)
  ret i32 %call
}


declare i32 @_Z9_cl_mad24iii(i32, i32, i32) local_unnamed_addr #0

define spir_func i32 @_Z5mad24iii(i32 %a, i32 %b, i32 %c) local_unnamed_addr #0 {
  %call = tail call  i32 @_Z9_cl_mad24iii(i32 %a, i32 %b, i32 %c)
  ret i32 %call
}


declare i32 @_Z9_cl_mad24jjj(i32, i32, i32) local_unnamed_addr #0

define spir_func i32 @_Z5mad24jjj(i32 %a, i32 %b, i32 %c) local_unnamed_addr #0 {
  %call = tail call  i32 @_Z9_cl_mad24jjj(i32 %a, i32 %b, i32 %c)
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


