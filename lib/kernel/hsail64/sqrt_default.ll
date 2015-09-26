; This is taken from tests/CodeGen/HSAIL/llvm.hsail.fsqrt.ll

declare float @llvm.hsail.fsqrt.f32(i1, i32, float) #0
declare double @llvm.hsail.fsqrt.f64(i1, i32, double) #0

define float @_Z17_sqrt_default_f32f(float %x) #0 {
  %ret = call float @llvm.hsail.fsqrt.f32(i1 false, i32 1, float %x) #0
  ret float %ret
}

define double @_Z17_sqrt_default_f64d(double %x) #0 {
  %ret = call double @llvm.hsail.fsqrt.f64(i1 false, i32 1, double %x) #0
  ret double %ret
}

attributes #0 = { nounwind readnone }
